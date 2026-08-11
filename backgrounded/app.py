"""App - owns the main loop and the two worker threads.

Thread ownership (see docs/ARCHITECTURE.md section 1):
  main       pygame render loop, fixed-timestep sim, command dispatch
  tray       Win32 message pump, pushes commands onto cmd_q
  wallpaper  blocking disk + SystemParametersInfoW writes

Off Windows there are no worker threads: neither the tray nor the wallpaper has
a counterpart there (see backgrounded.host), both start() calls return without
spawning anything, and the program is exactly its window. The three places that
changes behaviour rather than merely skipping work are marked with
``host.WINDOW_IS_THE_APP``, ``host.WALLPAPER_SUPPORTED`` and the POSIX branch of
:func:`_acquire_single_instance`.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import queue
import sys
import time

import pygame

from . import host, paths, persist
from .config import Config
from .constants import (
    AUTOSAVE_SEC, FOLLOW_PICK_RADIUS, RENDER_SIZE, SCENES, SIM_ACCUM_MAX_SEC,
    SIM_DT, SIM_SPEED_HEADROOM, SIM_STEPS_MAX, SIM_TICK_BUDGET_SEC, TARGET_FPS,
    TOOLS, TOOL_NONE,
)
from .render import toolbar
from .render.renderer import Renderer
from .shell.preview import Preview
from .shell.tools import ToolController
from .shell.tray import Tray
from .shell.wallpaper import WallpaperWriter
from .sim.entities import AGENT_HEIGHT
from .sim.world import World

log = logging.getLogger(__name__)

#: A copy of the save ``--fresh`` was about to walk away from. See
#: :meth:`App._keep_a_copy_before_fresh`.
PREFRESH_PREFIX = "save.prefresh."
#: ...and how many are kept. The OLDEST two are never evicted and the newest one
#: always is kept, so at most three exist. That split is the whole policy and it
#: is chosen against the failure it exists for: Backgrounded.bat forwards %* to
#: run.pyw, so a shortcut carrying --fresh runs it at EVERY login. The first
#: such login copies the user's real colony; every login after that copies a
#: world --fresh itself made fifteen seconds earlier. Keeping the newest few
#: would therefore evict the only copy that mattered within three reboots, which
#: is the exact accident this is here to survive. Keeping the newest one as well
#: is for the ordinary case - a developer who ran --fresh today wants today's.
PREFRESH_KEEP_OLD = 2
PREFRESH_KEEP_NEW = 1

#: Where a world that killed the frame loop is put instead of over save.json.
CRASH_PREFIX = "save.crashed."
#: ...and the save that is suspected of having killed it. Both are capped;
#: see :meth:`App._prune`.
SUSPECT_PREFIX = "save.suspect."
CRASH_KEEP = 3
SUSPECT_KEEP = 3


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _write_atomic(dst, payload: bytes) -> None:
    """Complete or absent, never half. Same shape as persist's own writes."""
    tmp = dst.with_name(dst.name + ".part")
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dst)


def _free_name(prefix: str):
    """The first unused ``<prefix><stamp>.json`` in APP_DIR."""
    base = f"{prefix}{_stamp()}"
    for name in [f"{base}.json"] + [f"{base}-{n}.json" for n in range(1, 100)]:
        p = paths.APP_DIR / name
        if not p.exists():
            return p
    return paths.APP_DIR / f"{base}-{time.time_ns()}.json"


#: Named kernel objects. The "Local\" prefix scopes them to the login session,
#: so two signed-in users can each run their own colony without a collision.
_MUTEX_NAME = "Local\\BackgroundedSingleInstance"
_QUIT_EVENT_NAME = "Local\\BackgroundedQuitRequest"
_INSTANCE_RECORD = "instance.json"
#: The POSIX stand-in for the mutex above: a file held under flock for the life
#: of the process. See :func:`_acquire_posix_lock`.
_LOCK_NAME = "instance.lock"

#: How long an arriving copy waits for the running one to save and go. A clean
#: exit writes the world; a kill costs up to AUTOSAVE_SEC of colony history, so
#: the grace period is deliberately generous - it is cheaper to wait twelve
#: seconds than to throw away a minute of the colony's life.
TAKEOVER_GRACE_SEC = 12.0

_WAIT_OBJECT_0 = 0x00000000
_ERROR_ALREADY_EXISTS = 183
_SYNCHRONIZE = 0x00100000
_PROCESS_TERMINATE = 0x0001
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_EVENT_MODIFY_STATE = 0x0002


class _FILETIME(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    def as_int(self) -> int:
        return (int(self.high) << 32) | int(self.low)


def _k32():
    """kernel32 with argtypes declared. Cached.

    The argtypes are not decoration: HANDLE is a pointer, and letting ctypes
    default it to c_int truncates every handle above 2GB on 64-bit Windows.
    """
    k = globals().get("_K32")
    if k is not None:
        return k
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    H, D, B = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32
    FT = ctypes.POINTER(_FILETIME)
    k.CreateMutexW.restype, k.CreateMutexW.argtypes = H, (H, B, ctypes.c_wchar_p)
    k.CreateEventW.restype, k.CreateEventW.argtypes = H, (H, B, B, ctypes.c_wchar_p)
    k.OpenEventW.restype, k.OpenEventW.argtypes = H, (D, B, ctypes.c_wchar_p)
    k.SetEvent.restype, k.SetEvent.argtypes = B, (H,)
    k.OpenProcess.restype, k.OpenProcess.argtypes = H, (D, B, D)
    k.GetCurrentProcess.restype, k.GetCurrentProcess.argtypes = H, ()
    k.GetProcessTimes.restype, k.GetProcessTimes.argtypes = B, (H, FT, FT, FT, FT)
    k.QueryFullProcessImageNameW.restype = B
    k.QueryFullProcessImageNameW.argtypes = (H, D, ctypes.c_wchar_p, ctypes.POINTER(D))
    k.WaitForSingleObject.restype, k.WaitForSingleObject.argtypes = D, (H, D)
    k.TerminateProcess.restype, k.TerminateProcess.argtypes = B, (H, ctypes.c_uint32)
    k.CloseHandle.restype, k.CloseHandle.argtypes = B, (H,)
    globals()["_K32"] = k
    return k


def _identity(handle) -> "tuple[int, str] | None":
    """(creation time, full image path) for an already-open process handle.

    The creation time is the part that makes a pid trustworthy. Windows
    recycles pids freely, so "pid 9312 is alive" says nothing at all; "pid 9312
    is alive AND started at exactly the 100ns tick we wrote down" identifies
    one specific process and cannot collide with a later tenant of that pid.
    """
    k = _k32()
    ct, et, kt, ut = _FILETIME(), _FILETIME(), _FILETIME(), _FILETIME()
    if not k.GetProcessTimes(handle, ctypes.byref(ct), ctypes.byref(et),
                             ctypes.byref(kt), ctypes.byref(ut)):
        return None
    size = ctypes.c_uint32(32768)
    buf = ctypes.create_unicode_buffer(size.value)
    if not k.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
        return None
    return ct.as_int(), buf.value


def _write_instance_record() -> None:
    """Leave behind what a later copy needs to recognise this process."""
    try:
        ident = _identity(_k32().GetCurrentProcess())
        if ident is None:
            return
        paths.APP_DIR.mkdir(parents=True, exist_ok=True)
        (paths.APP_DIR / _INSTANCE_RECORD).write_text(
            json.dumps({"pid": os.getpid(), "created": ident[0],
                        "image": ident[1],
                        "started": time.strftime("%Y-%m-%d %H:%M:%S")}),
            encoding="utf-8")
    except Exception:
        log.debug("could not write the instance record", exc_info=True)


def _clear_instance_record() -> None:
    """Drop our own record on the way out. Never anyone else's."""
    try:
        p = paths.APP_DIR / _INSTANCE_RECORD
        rec = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(rec, dict) and rec.get("pid") == os.getpid():
            p.unlink()
    except Exception:
        pass


def _open_running_instance():
    """Open the recorded process, but only if it is provably our own old copy.

    Returns ``(handle | None, record | None, why)``. A handle comes back only
    when three separate facts agree: a record exists, its pid is alive, and
    that live process reports the same creation time *and* the same image path
    the running copy wrote down. Anything less means the pid may have been
    recycled onto a stranger's program, and the caller must not terminate it.
    """
    k = _k32()
    try:
        rec = json.loads((paths.APP_DIR / _INSTANCE_RECORD).read_text(encoding="utf-8"))
        if not isinstance(rec, dict):
            return None, None, "the instance record is not an object"
    except FileNotFoundError:
        return None, None, "there is no instance record"
    except Exception:
        return None, None, "the instance record is unreadable"

    pid = rec.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return None, rec, "the instance record has no usable pid"

    proc = k.OpenProcess(
        _SYNCHRONIZE | _PROCESS_TERMINATE | _PROCESS_QUERY_LIMITED_INFORMATION,
        0, pid)
    if not proc:
        return None, rec, f"pid {pid} is not running"
    # OpenProcess still succeeds on a process that has exited, as long as
    # somebody (its parent, usually) still holds a handle keeping the object
    # alive. A signalled process handle means "already dead" - worth saying
    # plainly, because this branch is what the user reads in the log.
    if k.WaitForSingleObject(proc, 0) == _WAIT_OBJECT_0:
        k.CloseHandle(proc)
        return None, rec, f"pid {pid} has already exited"
    ident = _identity(proc)
    if ident is None:
        k.CloseHandle(proc)
        return None, rec, f"pid {pid} would not answer for itself"
    created, image = ident
    if created != rec.get("created") or image != rec.get("image"):
        k.CloseHandle(proc)
        return None, rec, (f"pid {pid} is {image!r} now, not the recorded "
                           f"{rec.get('image')!r} - the pid has been recycled")
    return proc, rec, "verified"


def _request_quit() -> bool:
    """Ask a running copy to stand down. Safe: only we own this event name."""
    k = _k32()
    h = k.OpenEventW(_EVENT_MODIFY_STATE, 0, _QUIT_EVENT_NAME)
    if not h:
        return False
    try:
        return bool(k.SetEvent(h))
    finally:
        k.CloseHandle(h)


def _listen_for_quit() -> None:
    """Create the event a later instance uses to ask us to hand over."""
    try:
        globals()["_QUIT_EVENT"] = _k32().CreateEventW(None, 1, 0, _QUIT_EVENT_NAME)
    except Exception:
        log.debug("could not create the quit event", exc_info=True)


def _quit_requested() -> bool:
    h = globals().get("_QUIT_EVENT")
    if not h:
        return False
    try:
        return _k32().WaitForSingleObject(h, 0) == _WAIT_OBJECT_0
    except Exception:
        return False


def _grab_mutex():
    """One CreateMutexW. Returns ``(handle | None, we_are_the_first)``."""
    k = _k32()
    h = k.CreateMutexW(None, 0, _MUTEX_NAME)
    if not h:
        return None, True                    # cannot tell; let the caller run
    return h, ctypes.get_last_error() != _ERROR_ALREADY_EXISTS


def _claim_mutex_until(deadline: float) -> bool:
    """Poll until the mutex name is free and ours, or the deadline passes.

    Each losing handle is closed straight away. Holding one would keep the
    named object alive after the old process died, and we would then wait
    forever on a name that only we were still propping up.
    """
    k = _k32()
    while True:
        h, first = _grab_mutex()
        if h is None:
            return True
        if first:
            globals()["_INSTANCE_MUTEX"] = h
            return True
        k.CloseHandle(h)
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def _take_over() -> bool:
    """Displace the copy that is already running. True if we now own the lock.

    Asking always comes before killing. The running copy sees the quit event
    inside a frame, saves the world and exits the normal way; terminating it
    instead would discard everything since the last autosave. Terminate is the
    fallback for a copy that has genuinely hung, and it fires only at a process
    that :func:`_open_running_instance` has positively identified as ours.
    """
    k = _k32()
    proc, rec, why = _open_running_instance()
    pid = (rec or {}).get("pid", "?")
    try:
        if _request_quit():
            log.info("another Backgrounded is running (pid %s); "
                     "asked it to save and exit", pid)
        elif proc is None:
            log.error(
                "Backgrounded is already running, but this copy cannot identify "
                "it (%s), so it will not kill anything. Close the other copy "
                "from its tray icon, or start with --allow-multi.", why)
            return False

        if _claim_mutex_until(time.monotonic() + TAKEOVER_GRACE_SEC):
            log.info("the running copy exited cleanly; taking over")
            return True

        if proc is None:
            log.error(
                "the running Backgrounded did not exit within %.0fs, and this "
                "copy cannot identify it (%s), so it will not be killed.",
                TAKEOVER_GRACE_SEC, why)
            return False

        log.warning("the running copy (pid %s) ignored the quit request for "
                    "%.0fs; terminating it", pid, TAKEOVER_GRACE_SEC)
        k.TerminateProcess(proc, 3)
        if _claim_mutex_until(time.monotonic() + 5.0):
            log.warning("took over by force - up to %.0fs of colony history "
                        "may have been lost", AUTOSAVE_SEC)
            return True
        log.error("could not take the single-instance lock even after "
                  "terminating pid %s", pid)
        return False
    finally:
        if proc:
            k.CloseHandle(proc)


def _acquire_posix_lock() -> bool:
    """The single-instance check where there are no named kernel objects.

    ``flock`` rather than a pid file, because the kernel drops the lock when
    the process dies however it dies - a kill -9 or a power cut leaves a stale
    pid file behind, and every pid-file scheme then has to guess whether pid
    4471 is still the colony or somebody else's shell.

    Deliberately NOT the takeover the Windows path does. Takeover exists there
    because two copies fight over the desktop wallpaper and because a copy
    living in the tray with its window hidden is easy to lose track of - a copy
    you cannot see should not beat the one you just launched. Neither is true
    here: there is no wallpaper to contend for, and the other copy is a window
    on screen. So the newcomer stands down and says where the running one is.

    The file object is left open on purpose - closing it releases the lock, so
    it has to outlive this function, and the process holds it until it exits.
    """
    import fcntl

    try:
        paths.APP_DIR.mkdir(parents=True, exist_ok=True)
        fh = open(paths.APP_DIR / _LOCK_NAME, "a+")
    except Exception:
        log.debug("could not open the instance lock; starting anyway",
                  exc_info=True)
        return True
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return False
    except Exception:
        fh.close()
        log.debug("instance lock unusable; starting anyway", exc_info=True)
        return True
    try:
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
    except Exception:
        pass                                  # the lock is what matters, not the pid
    globals()["_INSTANCE_LOCK"] = fh
    return True


def _acquire_single_instance(takeover: bool = True) -> bool:
    """Make this the one running copy, displacing an older one if asked.

    Two instances both write the wallpaper A/B pair and both call
    SystemParametersInfoW, so they overwrite each other's files mid-write and
    the desktop flickers between two different worlds. Worse, closing the
    preview window only *hides* it (the app lives in the tray), so it is easy
    to leave orphans running without noticing - which is exactly why the newer
    copy now wins instead of giving up: the copy you cannot see should not beat
    the one you just launched.

    The winning handle is deliberately leaked; it lives as long as the process
    and Windows frees it on exit.
    """
    if not host.IS_WINDOWS:
        return _acquire_posix_lock()
    try:
        h, first = _grab_mutex()
        if h is None:
            return True
        if first:
            globals()["_INSTANCE_MUTEX"] = h
            return True
        # Not ours to hold: keeping this handle would keep the name alive after
        # the old process exits, and _claim_mutex_until would never succeed.
        _k32().CloseHandle(h)
        return _take_over() if takeover else False
    except Exception:
        log.debug("single-instance check failed; starting anyway", exc_info=True)
        return True


def _screen_size() -> tuple[int, int]:
    """The desktop's pixel size - what a wallpaper frame is resized to.

    Only the wallpaper writer reads this, so off Windows it is answering a
    question nobody asks; it is kept honest anyway rather than stubbed, because
    a wrong number here would be a silent one. pygame's answer needs the
    display module initialised, which it is by the time App constructs the
    writer.
    """
    if host.IS_WINDOWS:
        try:
            u = ctypes.windll.user32
            u.SetProcessDPIAware()
            return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
        except Exception:
            return (1920, 1080)
    try:
        sizes = pygame.display.get_desktop_sizes()
        if sizes and sizes[0][0] > 0 and sizes[0][1] > 0:
            return int(sizes[0][0]), int(sizes[0][1])
    except Exception:
        log.debug("could not read the desktop size", exc_info=True)
    return (1920, 1080)


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        paths.ensure_dirs()
        self.args = args
        self.cfg = Config.load()

        # Command-line flags are *session* overrides, not preferences. Without
        # this split a single `--hide --no-wallpaper` test run would write both
        # into config.json and permanently disable the window and the wallpaper
        # for every later run. _cli_keys records what the CLI touched so
        # _save_config can put the stored value back; _user_keys records what
        # the user changed via the tray during the session, which does win.
        self._pristine = {k: getattr(self.cfg, k) for k in vars(self.cfg)}
        self._cli_keys: set[str] = set()
        self._user_keys: set[str] = set()

        if args.scene:
            self.cfg.scene = args.scene
            self._cli_keys.add("scene")
        if args.hide:
            self.cfg.show_window = False
            self._cli_keys.add("show_window")
        if args.no_wallpaper:
            self.cfg.wallpaper_enabled = False
            self._cli_keys.add("wallpaper_enabled")
        if not host.WALLPAPER_SUPPORTED and self.cfg.wallpaper_enabled:
            # Through _cli_keys, not by writing the config, and for the same
            # reason: this is a fact about the machine the app happens to be
            # running on, not a preference the user expressed. Persisting it
            # would silently turn the wallpaper off in a config.json that is
            # later opened on Windows.
            self.cfg.wallpaper_enabled = False
            self._cli_keys.add("wallpaper_enabled")
        if args.hide and host.WINDOW_IS_THE_APP:
            # --hide is a Windows convenience (the app carries on in the tray).
            # Here it leaves a process with nothing on screen and no icon to
            # bring it back, which is only ever wanted alongside --exit-after.
            log.warning("--hide leaves no visible window and no tray icon on "
                        "this platform; kill the process or pair it with "
                        "--exit-after")

        self.running = True
        self.cmd_q: queue.Queue = queue.Queue()

        # --- world -------------------------------------------------------
        self.world: World | None = None
        #: Did this world come off disk? Only a world that did can be evidence
        #: about the file on disk - see :meth:`_shutdown_after_crash`.
        self._from_disk = False
        if args.fresh:
            # BEFORE the new world exists, because the destruction is not this
            # line - it is the autosave sixty seconds from now.
            self._keep_a_copy_before_fresh()
        else:
            self.world = persist.load_world()
            self._from_disk = self.world is not None
        if self.world is None:
            self.world = World(scene=self.cfg.scene)
            log.info("generated a new world (seed=%d)", self.world.seed)
        else:
            self.world.events.request_scene(self.cfg.scene)
        self._apply_world_config()
        #: world_time as loaded. A loop that dies without ever moving this is
        #: the signature of a save the renderer cannot survive, as opposed to
        #: an unlucky moment hours into a healthy session.
        self._start_world_time = float(getattr(self.world, "world_time", 0.0))
        #: The exception that ended run()'s loop, or None. _shutdown reads it
        #: and refuses to write a suspect world over a good save.
        self._crash: BaseException | None = None

        # --- pygame ------------------------------------------------------
        pygame.init()
        pygame.display.set_caption("Backgrounded - Live View")
        self.preview = Preview(RENDER_SIZE, scale=self.cfg.window_scale)
        self.preview.ensure_window(self.cfg.show_window)
        self.renderer = Renderer()
        self.renderer.show_stats = getattr(self.cfg, "show_stats", True)
        self.renderer.show_names = getattr(self.cfg, "show_names", True)
        self.renderer.show_activity = getattr(self.cfg, "show_activity", True)
        self.renderer.show_log = getattr(self.cfg, "show_log", True)
        # HUD size is a module-level dial rather than a Renderer flag because the
        # wallpaper bake and the window overlay go through the same two draw
        # functions and must agree; applying the stored value here is what makes
        # the [ and ] adjustment survive a restart.
        try:
            from .render import hud as _hud
            _hud.HUD_SCALE = float(getattr(self.cfg, "hud_scale", _hud.HUD_SCALE))
        except Exception:
            log.debug("could not apply stored hud scale", exc_info=True)
        # Cold start and world load are both "the colony is already over there";
        # without this the camera opens at world x 0 and eases in at 220 px/s.
        self._cut_camera()
        self.tools = ToolController()

        # --- threads -----------------------------------------------------
        self.wallpaper = WallpaperWriter(_screen_size())
        self.wallpaper.capture_original()
        if self.cfg.wallpaper_enabled:
            self.wallpaper.start()

        self.tray = Tray(self.cmd_q, self._tray_state)
        self.tray.start()

        self.clock = pygame.time.Clock()
        self._sim_accum = 0.0
        #: Sim-seconds the loop asked for and could not afford, and so threw
        #: away. Counted rather than ignored - see :meth:`_advance_sim`.
        self._sim_dropped = 0.0
        self._sim_drop_log = 0.0
        self._last_save = time.monotonic()
        self._last_wall = 0.0
        self._captures = 0
        # Frame counter, used only to bake the HUD into the world frame at most
        # once per frame - see _bake_hud.
        self._frame_no = 0
        self._hud_frame = -1
        # Mirrors Preview's own fullscreen flag, refreshed from the event pump.
        # Only the tray's window-size command reads it, and only so it can leave
        # fullscreen properly instead of resizing out from under it.
        self._fullscreen = False

    # ----------------------------------------------------- keeping a copy --
    def _prune(self, prefix: str, keep_new: int, keep_old: int = 0) -> None:
        """Cap one family of set-aside saves. Never raises.

        Newest-first, keep *keep_new*; oldest-first, keep *keep_old*; delete the
        rest. Two ends rather than one because the two families want opposite
        things and the code should say so: a crash report is only interesting
        while it is recent, and a pre-fresh copy is only interesting while it is
        the ORIGINAL. This runs at login, forever, on a machine whose owner will
        never look in %LOCALAPPDATA%, and each item is a whole ~160 KB save.
        """
        try:
            items = []
            for p in paths.APP_DIR.glob(prefix + "*"):
                if not p.name.startswith(prefix) or p.name.endswith(".part"):
                    continue
                try:
                    items.append((p.stat().st_mtime, p.name, p))
                except OSError:
                    continue
            items.sort(key=lambda it: (it[0], it[1]))
            spared = set()
            for _mt, name, _p in items[:keep_old]:
                spared.add(name)
            # max(0, ...) or this is a NEGATIVE slice whenever there are fewer
            # items than we mean to keep: at len 2 and keep_new 3 it reads
            # items[-1:], spares only the newest, and deletes the OLDEST - the
            # exact opposite of "keep everything, there is nothing to prune".
            # That turned the second incident into the one that destroyed the
            # copy the first incident had set aside.
            lo = max(0, len(items) - keep_new)
            for _mt, name, _p in (items[lo:] if keep_new else []):
                spared.add(name)
            for _mt, name, p in items:
                if name in spared:
                    continue
                try:
                    p.unlink()
                    log.info("pruned %s", name)
                except OSError:
                    log.debug("could not prune %s", name, exc_info=True)
        except Exception:
            log.exception("could not prune %r", prefix)

    def _keep_a_copy_before_fresh(self) -> None:
        """``--fresh`` means "start a new world", not "destroy the old one".

        --fresh skips ``persist.load_world`` entirely, which is also the only
        place the pre-migration backup is ever taken - so the flag walked past
        every protection the save path has. Nothing was destroyed at the moment
        it was passed either, which is what made it so hard to notice: the world
        on disk survived until the first autosave sixty seconds later, and then
        it was gone with no copy anywhere. Measured on an 88,627-byte colony, a
        byte-for-byte search of the whole app directory afterwards found nothing.

        That is a silent, repeating loss rather than a one-off, because
        Backgrounded.bat forwards %* to run.pyw: a shortcut carrying the flag
        does this at every login.

        So the incoming bytes are copied aside first - the bytes, not a re-read
        and not a re-serialisation, so the copy is the original exactly. Never
        raises and never blocks startup: --fresh must still start a world even
        if the copy cannot be made, and it says so loudly in the log if not.
        """
        try:
            src = paths.SAVE_PATH
            try:
                st = src.stat()
            except FileNotFoundError:
                return                      # nothing to lose; ordinary first run
            except Exception:
                log.exception("--fresh: could not examine the existing save")
                return
            if not st.st_size or st.st_size > persist.SAVE_SIZE_LIMIT:
                log.warning("--fresh: the existing save is %d bytes; not copying "
                            "it aside", st.st_size)
                return
            raw = src.read_bytes()
            paths.ensure_dirs()
            dst = _free_name(PREFRESH_PREFIX)
            _write_atomic(dst, raw)
            log.warning("--fresh: starting a NEW world; the one that was on "
                        "disk (%d bytes) has been copied to %s. To go back, "
                        "close Backgrounded and rename that file to %s.",
                        len(raw), dst, paths.SAVE_PATH.name)
            # NOT pruned, for the same reason the suspect family is not: the
            # keep-2-oldest rule protected the colony only while it WAS one of
            # the two oldest, and Backgrounded.bat forwards %* to run.pyw - so a
            # shortcut carrying --fresh fills both sparing slots with worthless
            # copies after two logins, and the third login evicts the real one.
        except Exception:
            log.exception("--fresh: could not keep a copy of the existing save; "
                          "starting fresh anyway")

    def _keep_a_copy_before_reset(self) -> None:
        """"Start Over" means "start a new colony", not "destroy the old one".

        --fresh got :meth:`_keep_a_copy_before_fresh` and the tray's "Start
        Over" reaches the same destruction by a different road with none of it.
        The shape of the loss is the same one that made --fresh so hard to
        notice: nothing is destroyed at the moment it is clicked, because the
        handler calls ``save_world`` first, so save.json holds the old colony
        for one more minute - and then the autosave lands and it is gone with no
        copy anywhere. Measured on the unfixed path exactly as it was for
        --fresh: a byte-for-byte search of the whole app directory afterwards
        finds nothing.

        SOURCED FROM MEMORY, NOT FROM DISK, and that is what makes the ordering
        safe rather than merely lucky. ``_keep_a_copy_before_fresh`` has to read
        save.json because --fresh never loaded a world and there is nothing else
        to copy. Here there is: ``self.world`` IS the colony being walked away
        from, and it is up to sixty seconds newer than the file. Serialising it
        means this call can sit first in the handler - before the save, before
        the new World exists - and still preserve the exact world the user was
        looking at when they clicked. Reading the file instead would make the
        copy's contents depend on where in the handler the call happened to be
        put: before the save it would be up to a minute stale, and anywhere
        after the new world existed it would be a race against the autosave to
        avoid preserving the *new* world instead of the old one.

        Written into the ``save.prefresh.`` family so there is one place to look
        and one restore instruction, and like the rest of that family it is
        never pruned. It also does not go through ``persist``: a session that
        has blocked saves (an unreadable save.json - see ``persist._block_saves``)
        must still be able to set this aside, since it is writing a new file
        under a new name and cannot overwrite anything.

        Never raises and never blocks the reset: "Start Over" must still start
        over even if the copy cannot be made, and it says so loudly if not.
        """
        world = self.world
        if world is None:
            return
        payload = None
        try:
            payload = json.dumps(world.to_dict(),
                                 separators=(",", ":")).encode("utf-8")
        except Exception:
            log.exception("Start Over: could not serialise the world to copy it "
                          "aside; falling back to the bytes on disk")
        if payload is None:
            # Older and possibly by a whole autosave interval, but a stale
            # colony is a colony and the alternative here is nothing at all.
            self._keep_a_copy_before_fresh()
            return
        try:
            paths.ensure_dirs()
            dst = _free_name(PREFRESH_PREFIX)
            _write_atomic(dst, payload)
            log.warning("Start Over: beginning a NEW world; the colony that was "
                        "running (%d bytes) has been copied to %s. To go back, "
                        "close Backgrounded and rename that file to %s.",
                        len(payload), dst, paths.SAVE_PATH.name)
        except Exception:
            log.exception("Start Over: could not keep a copy of the running "
                          "colony; starting over anyway")

    def _set_aside(self, prefix: str, payload: bytes, why: str):
        """Write *payload* to a fresh ``<prefix><stamp>.json``. Never raises."""
        try:
            paths.ensure_dirs()
            dst = _free_name(prefix)
            _write_atomic(dst, payload)
            log.error("%s -> %s (%d bytes)", why, dst, len(payload))
            return dst
        except Exception:
            log.exception("could not set aside %r", prefix)
            return None

    # -------------------------------------------------------- world config --
    def _apply_world_config(self) -> None:
        """Push the settings the World keeps its own live copy of onto it.

        Called for the world the app starts with *and* for every world built
        afterwards. World's default for auto_scene_rotate is True and has to
        stay that way (from_dict leans on it for saves written before the flag
        existed), so a fresh World from "Start Over" arrives with rotation on no
        matter what the tray checkbox says - and then the menu shows one thing
        while the weather does another until the next restart, which is the
        exact desync toggle_auto_scene exists to prevent.
        """
        if self.world is None:
            return
        self.world.auto_scene_rotate = bool(
            getattr(self.cfg, "auto_scene_change", True))

    # ---------------------------------------------------------- tray glue --
    def _tray_state(self) -> dict:
        return {
            "show_window": self.cfg.show_window,
            "paused": self.cfg.paused,
            "scene": self.world.events.scene if self.world else self.cfg.scene,
            "sim_speed": self.cfg.sim_speed,
            "wallpaper_enabled": self.cfg.wallpaper_enabled,
            "show_stats": getattr(self.cfg, "show_stats", True),
            "show_names": getattr(self.cfg, "show_names", True),
            "show_activity": getattr(self.cfg, "show_activity", True),
            "show_log": getattr(self.cfg, "show_log", True),
            "auto_scene_change": getattr(self.cfg, "auto_scene_change", True),
            "window_scale": getattr(self.cfg, "window_scale", 1.0),
        }

    def _drain_commands(self) -> None:
        while True:
            try:
                kind, payload = self.cmd_q.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle(kind, payload)
            except Exception:
                log.exception("command %r failed", kind)

    def _nudge_hud_scale(self, step: int) -> None:
        """Grow or shrink the stats panel and chronicle log, and remember it.

        Screen-anchoring the HUD means it is drawn at its authored pixel size
        rather than being magnified by the letterbox fit, so how big it *should*
        be depends on the monitor it is being read on. That is a per-user
        judgement, not something a constant can be right about, hence a live
        control - and hence persisting it, since being asked to re-tune it on
        every launch would be worse than the original hard-coded number.
        """
        try:
            from .render import hud
            lo, hi = 1.0, 3.5
            new = round(min(hi, max(lo, hud.HUD_SCALE + 0.15 * step)), 2)
            if abs(new - hud.HUD_SCALE) < 1e-6:
                return
            hud.HUD_SCALE = new
            self.cfg.hud_scale = new
            self._save_config("hud_scale")
            log.info("hud scale %.2f", new)
        except Exception:
            log.debug("could not change hud scale", exc_info=True)

    def _save_config(self, *changed: str) -> None:
        """Persist preferences, without letting this session's CLI flags leak
        into the stored config."""
        self._user_keys.update(changed)
        revert = {k: getattr(self.cfg, k)
                  for k in self._cli_keys - self._user_keys}
        for k in revert:
            setattr(self.cfg, k, self._pristine[k])
        try:
            self.cfg.save()
        finally:
            for k, v in revert.items():       # restore the live session value
                setattr(self.cfg, k, v)

    def _handle(self, kind: str, payload) -> None:
        cfg = self.cfg
        if kind == "quit":
            self.running = False
        elif kind == "toggle_window":
            cfg.show_window = not cfg.show_window
            self.preview.ensure_window(cfg.show_window)
            self._save_config("show_window")
        elif kind == "toggle_wallpaper":
            cfg.wallpaper_enabled = not cfg.wallpaper_enabled
            if cfg.wallpaper_enabled:
                self.wallpaper.start()
            else:
                self.wallpaper.stop()
                self.wallpaper.restore()
            self._save_config("wallpaper_enabled")
        elif kind == "toggle_pause":
            cfg.paused = not cfg.paused
            # The caption carries the marker for anyone watching the window, but
            # the window is usually hidden and `paused` survives a restart, so
            # the tooltip has to say it too or a frozen wallpaper has no
            # explanation anywhere.
            self.tray.refresh()
            self._save_config("paused")
        elif kind == "speed":
            cfg.sim_speed = float(payload)
            self._save_config("sim_speed")
        elif kind == "window_scale":
            scale = float(payload)
            if scale > 0.05:
                cfg.window_scale = scale
                # Resizing a fullscreen window drops it back to a window behind
                # Preview's back, leaving it convinced it is still fullscreen -
                # the next F11 then looks broken. Leave first, via the method
                # that owns the flag.
                if self._fullscreen:
                    self._fullscreen = self.preview.toggle_fullscreen()
                self.preview.set_scale(scale)
                self._save_config("window_scale")
        elif kind == "scene" and payload in SCENES:
            cfg.scene = payload
            self.world.events.request_scene(payload)
            self._save_config("scene")
        elif kind == "toggle_stats":
            self.renderer.show_stats = not self.renderer.show_stats
            cfg.show_stats = self.renderer.show_stats
            self._save_config("show_stats")
        elif kind == "toggle_names":
            self.renderer.show_names = not self.renderer.show_names
            cfg.show_names = self.renderer.show_names
            self._save_config("show_names")
        elif kind == "toggle_activity":
            self.renderer.show_activity = not self.renderer.show_activity
            cfg.show_activity = self.renderer.show_activity
            self._save_config("show_activity")
        elif kind == "toggle_log":
            self.renderer.show_log = not self.renderer.show_log
            cfg.show_log = self.renderer.show_log
            self._save_config("show_log")
        elif kind == "toggle_auto_scene":
            # The setting lives on the config; the world holds the live copy the
            # tick actually reads, so both move together or the menu would show
            # one thing while the weather did another.
            cfg.auto_scene_change = not getattr(cfg, "auto_scene_change", True)
            self._apply_world_config()
            self._save_config("auto_scene_change")
            log.info("auto scene rotation %s",
                     "on" if cfg.auto_scene_change else "off")
        elif kind == "new_terrain":
            self.world.randomise_terrain()
            # Every agent was relocated in one frame, which is precisely the
            # case Camera.SNAP_DIST was sized for - but SNAP_DIST only fires
            # past 3200 px, and a colony reseated 800-3200 px away instead got
            # a slow crawl over land nobody asked to look at. This is a tray
            # command: the user is watching, and waiting seven seconds to see
            # the result of "Randomise Terrain" reads as the command failing.
            self._cut_camera()
            persist.save_world(self.world)
        elif kind == "clear_graves":
            n = self.world.clear_graves()
            log.info("cleared %d graves", n)
        elif kind == "save":
            persist.save_world(self.world)
        elif kind == "reset":
            # FIRST, before the save and before the new world exists. The save
            # below is not what destroys the colony - the autosave a minute
            # later is - but putting the copy first means the ordering of the
            # rest of this branch cannot silently change what gets preserved.
            self._keep_a_copy_before_reset()
            persist.save_world(self.world)
            # Drop any Hand drag first: the Grab holds ids belonging to the
            # world about to be thrown away, and the next mouse move would drag
            # whichever entity in the *new* colony happens to reuse that id.
            self.tools.release_all(self.world)
            self.world = World(scene=cfg.scene)
            self._apply_world_config()
            # This world did not come off disk, so it is not evidence about
            # what is on disk. Without these two lines a crash after Start Over
            # would look exactly like the poisoned-save signature - a world
            # that never advanced past _start_world_time - and
            # _shutdown_after_crash would move a perfectly good save.json aside
            # for something it had nothing to do with.
            self._from_disk = False
            self._start_world_time = float(self.world.world_time)
            # Same story as new_terrain: a brand new colony is seated somewhere
            # the old camera has no reason to be pointing.
            self._cut_camera()
            log.info("world reset (seed=%d)", self.world.seed)

    # --------------------------------------------------------------- loop --
    def run(self) -> int:
        log.info("running: pop=%d scene=%s",
                 len(self.world.population.alive_agents()), self.world.events.scene)
        try:
            while self.running:
                self._frame()
        except KeyboardInterrupt:
            log.info("interrupted")
        except Exception as exc:
            # Recorded, not just logged. _shutdown behaves completely
            # differently for a loop that ENDED and a loop that DIED, and
            # before this it could not tell the two apart - so it wrote the
            # world that had just killed the frame loop straight back over
            # save.json, and the next launch loaded it and died the same way.
            # That re-save is what turned a bad frame into a bricked program.
            self._crash = exc
            log.exception("fatal error in main loop")
        finally:
            self._shutdown()
        return 0

    def _frame(self) -> None:
        dt = self.clock.tick(TARGET_FPS) / 1000.0
        dt = min(dt, 0.1)                     # never let a stall bomb the sim
        self._frame_no += 1

        self._drain_commands()
        pointer = self.preview.pump_events()
        # Held WASD / arrows, polled after the event pump so a key released this
        # frame has already been seen. Self-guarding: a hidden or unfocused
        # preview does nothing here.
        # A right/middle drag and a held key are the same motion by different
        # means, so they spill into the same camera through one call - two calls
        # would just re-arm the manual hold twice in a frame.
        self._pan(self.preview.pan_keys(dt)
                  + float(pointer.get("pan_residual") or 0.0),
                  pointer.get("reset_view"))
        self._fullscreen = bool(pointer.get("fullscreen", self._fullscreen))
        # One list of pointer events, read twice: the camera lock gets first
        # refusal (it only acts when NO tool is selected, so it can never take
        # a click a tool wanted), then the tools take it as they always have.
        events = self._to_world(pointer.get("pointer"))
        # A click that landed on the roster is CONSUMED here and never reaches
        # the tools. It has to be, and not merely deprioritised: the stats panel
        # is drawn in window px on top of the scene, so a click on a name also
        # has a perfectly valid world position underneath it, and letting it
        # through would follow the colonist AND strike lightning wherever he
        # happens to be standing.
        events = self._roster_click(events)
        self._follow_click(events)
        self.tools.handle(events, self.world, self.world.world_time)
        step = pointer.get("hud_scale")
        if step:
            self._nudge_hud_scale(int(step))
        if pointer.get("closed"):
            if host.WINDOW_IS_THE_APP:
                # The X means what it says here. Hiding is only a sensible
                # answer to it when something else is still on screen offering
                # a way back; with no tray icon this would strand a running
                # colony behind an invisible window. Out through the normal
                # door, so _shutdown saves the world.
                log.info("window closed; exiting")
                self.running = False
            else:
                self.cfg.show_window = False
                self.preview.ensure_window(False)
                self._save_config("show_window")

        # fixed-timestep sim, scaled by the speed setting
        if not self.cfg.paused:
            self._advance_sim(dt)

        frame = self.renderer.draw(self.world, dt)
        # AFTER draw, not before: Renderer.draw calls camera.follow as its first
        # statement, and that is where a lock releases itself when its man dies
        # or is abducted. Reading lock_id beforehand would leave the ring on a
        # dead colonist for one frame - on the 4 Hz wallpaper, a quarter of a
        # second of pointing at somebody who is not there.
        self._sync_follow_hud()

        # One render, two destinations, and the HUD has to land differently on
        # each - so the ordering below is load-bearing. The window gets it as a
        # *screen-anchored* overlay painted straight onto the window surface,
        # which is what keeps the panels in the window's corners while the world
        # is zoomed and panned beneath them. Only afterwards does the wallpaper
        # path bake the HUD into the shared frame, world-anchored, because a
        # wallpaper is a bare image with no corner of its own to hold. Present
        # first, bake second: the other order would put both copies on screen.
        if self.cfg.show_window:
            self.preview.present(frame, overlay=self._draw_window_overlay)
            self.preview.set_caption(self._hud())

        now = time.monotonic()
        if self.cfg.wallpaper_enabled and (now - self._last_wall) >= 1.0 / max(1, self.cfg.wallpaper_fps):
            self._last_wall = now
            try:
                self._bake_hud(frame)
                raw = pygame.image.tobytes(frame, "RGB")
                self.wallpaper.submit(raw, RENDER_SIZE)
            except Exception:
                log.exception("wallpaper submit failed")

        if self.args.capture and self._captures < self.args.capture:
            if self.world.tick_count % self.args.capture_every == 0:
                # Captures are meant to look like the wallpaper, so they need
                # the baked-in HUD too. _bake_hud is a no-op if the wallpaper
                # push above already did it this frame.
                self._bake_hud(frame)
                self._save_capture(frame)

        if now - self._last_save >= AUTOSAVE_SEC:
            self._last_save = now
            persist.save_world(self.world)

        if _quit_requested():
            # A newer copy has started and wants the desktop. Leaving through
            # the normal door means _shutdown saves the world and puts the real
            # wallpaper back before the other one captures it.
            log.info("a newer instance asked us to stand down")
            self.running = False

        if self.args.exit_after and self.world.world_time >= self.args.exit_after:
            log.info("exit-after reached")
            self.running = False

    # ------------------------------------------------------- pacing the sim --
    def _advance_sim(self, dt: float) -> int:
        """Turn *dt* real seconds into whole SIM_DT ticks. Returns the count.

        WHAT THIS REPLACED, and why it had to go. The loop used to read::

            self._sim_accum += dt * self.cfg.sim_speed
            steps = 0
            while self._sim_accum >= SIM_DT and steps < 8:
                self.world.tick(SIM_DT); self._sim_accum -= SIM_DT; steps += 1

        Do the arithmetic on the cap. At TARGET_FPS 60, dt is 1/60, so 16x asks
        for 0.0167 * 16 = 0.267 s of sim per frame, and 0.267 / SIM_DT is
        EXACTLY 8. The cap therefore sat precisely on 16x's requirement with
        zero margin, and every frame slower than 60 fps - which is every frame
        at 16x on a machine that has to work for it - asked for more than eight
        ticks, got eight, and dropped the rest on the floor. Nothing was logged
        and nothing was shown: the world simply ran slower than the menu said it
        was running. That silent loss is what this method exists to replace with
        a trade the user can see.

        THE TRADE, in the terms it was asked for: do not fight the frame rate.

        * **A WALL-CLOCK BUDGET, NOT A STEP COUNT.** Tick until the accumulator
          drains or ``SIM_TICK_BUDGET_SEC`` of real time is gone. A count cannot
          know what a tick costs and a tick is not one price - measured on two
          warm colonies it is 0.7 ms typical and 4 ms at p95, so any fixed count
          wastes the cheap frames or truncates the expensive ones. The budget
          gets the right answer on both without being told which it is in.
        * **THE CHECK IS AFTER THE FIRST TICK, DELIBERATELY.** At least one tick
          always runs when one is owed, so a machine where a single tick costs
          more than the whole budget still advances instead of freezing, and
          0.5x/1x - which want well under one tick a frame - can never be
          starved by a slow neighbour frame.
        * **FRAME RATE IS ALLOWED TO FALL.** Nothing here shortens the budget
          when frames get long. A frame that spends its whole budget ticking is
          a frame that took budget + render, and that is the point: fps drops,
          the sim keeps its speed, and what is protected is the *event pump*,
          which still runs once per frame at a period bounded by that budget.
        * **AIM SLIGHTLY UNDER, BUT ONLY WHEN IT IS NEEDED.** ``lagging`` below
          is exactly the condition "the budget stopped me last frame" - the
          drain loop cannot leave a whole tick owed for any other reason - and
          only then is the request scaled by ``SIM_SPEED_HEADROOM``. A machine
          that can deliver 16x delivers 16x; one that cannot asks for 15.04x so
          it has slack to repay the shortfall instead of ratcheting into the
          discard below on every hitch.
        * **THE DEBT IS BOUNDED, AND SHEDDING IT IS COUNTED.** Past
          ``SIM_ACCUM_MAX_SEC`` the surplus is thrown away, in one line, and
          added up so the log can say how much. Dropping sim time is honest when
          it is deliberate and bounded; the version this replaced was neither.

        DETERMINISM IS UNTOUCHED AND THAT IS NOT AN ACCIDENT: every tick is
        still exactly ``SIM_DT`` and nothing in the world's update reads a wall
        clock, so speed changes how fast the colony is *watched* and never what
        happens in it. The same seed run for the same number of ticks lands in
        the same state at 1x and at 16x. Anything added here that varies the
        step size - "catch up with one big tick" is the usual temptation -
        breaks that, and with it every A/B measurement in this project.
        """
        world = self.world
        if world is None:
            return 0
        try:
            speed = float(self.cfg.sim_speed)
        except (TypeError, ValueError):
            speed = 1.0
        if not (speed > 0.0):
            return 0

        lagging = self._sim_accum >= SIM_DT
        self._sim_accum += dt * speed * (SIM_SPEED_HEADROOM if lagging else 1.0)
        if self._sim_accum > SIM_ACCUM_MAX_SEC:
            self._sim_dropped += self._sim_accum - SIM_ACCUM_MAX_SEC
            self._sim_accum = SIM_ACCUM_MAX_SEC

        started = time.perf_counter()
        steps = 0
        while self._sim_accum >= SIM_DT and steps < SIM_STEPS_MAX:
            world.tick(SIM_DT)
            self._sim_accum -= SIM_DT
            steps += 1
            if (time.perf_counter() - started) >= SIM_TICK_BUDGET_SEC:
                break

        # Said out loud, at most twice a minute, and only once it amounts to a
        # tick. The whole complaint about the old cap was that it never said.
        if self._sim_dropped >= SIM_DT:
            now = time.monotonic()
            if self._sim_drop_log <= 0.0:
                self._sim_drop_log = now      # start the window, say nothing yet
            elif now - self._sim_drop_log >= 30.0:
                log.info("sim speed %.1fx: %.1f s of sim time discarded over the "
                         "last %.0f s. This machine cannot render %.1fx, so the "
                         "world is running slower than the menu says rather than "
                         "skipping about.",
                         speed, self._sim_dropped, now - self._sim_drop_log, speed)
                self._sim_drop_log = now
                self._sim_dropped = 0.0
        return steps

    # ------------------------------------------------- the two-camera seam --
    # The world is 6400 px wide and the frame is 1600, so there are now two
    # transforms between a mouse click and a lightning bolt: Renderer.camera
    # (world -> frame) and Preview (frame -> window). Each half knows only its
    # own leg, which is the only arrangement that keeps the wallpaper path -
    # which has a Renderer but no Preview at all - honest. App is where they
    # meet, and both methods below exist purely to join them.

    def _camera(self):
        """The world camera, or None if the renderer has not got one.

        getattr rather than a plain attribute read so this file keeps working
        against a Renderer without a camera: a missing camera is exactly the
        identity camera (cam.x == 0, world == frame), which is the behaviour
        this app had before the world grew. Fails to *old* behaviour, never to
        a crash on the frame path.
        """
        return getattr(self.renderer, "camera", None)

    def _cut_camera(self) -> None:
        """Point the view at the colony *now*, with no ease. Never raises.

        Camera.snap_to has existed since the world was widened, with a docstring
        naming exactly these moments - "for load and for randomise_terrain" -
        and until now nothing in the app called it. The camera therefore always
        started at world x 0 and walked to the colony at MAX_SPEED, 220 px/s.
        Measured on a cold start: seed 4242 spent 7.7 s crossing empty land,
        999 6.7 s, 2718 4.1 s, with a theoretical worst case around 11 s on a
        6400 px map. Camera.SNAP_DIST does not save it - that only cuts past
        3200 px, and most colonies sit inside that.

        The three call sites are the three times the colony did not walk into
        view: startup (fresh world or a loaded save), Randomise Terrain, and
        Start Over. Everything else stays an ease, because everywhere else the
        people really did walk.

        The world-side decision of *where* the colony is stays in
        Camera.cut_to, which reuses follow()'s own cluster rule - deriving a
        second answer here would make the cut land somewhere the ease then
        immediately drifts away from. sim/ is not touched: nothing needed to
        tell us a relocation happened, because all three sites are commands this
        file already handles.
        """
        cam = self._camera()
        if cam is None or self.world is None:
            return
        try:
            cam.cut_to(self.world)
        except Exception:
            log.debug("camera cut failed", exc_info=True)

    # -------------------------------------------- click a man, watch that man --
    def _follow_click(self, events) -> None:
        """Left click a stickman with no tool selected: the view follows him.

        WHICH BUTTON, and the argument for it, because both readings of "if I
        click a stickman" were available and only one of them is safe.

        Left, but ONLY IN WATCH MODE - that is, only while ``tools.tool`` is
        ``TOOL_NONE``. The alternative was "a left click that HITS a stickman
        outranks the selected tool", and that alternative breaks the palette on
        exactly the clicks it exists for. TOOL_HAND's entire purpose is to pick a
        colonist up and throw him; TOOL_LIGHTNING's most obvious use is smiting
        one. Under the priority rule neither could ever be aimed at a person
        again - the two most-used tools would silently become camera controls
        the moment you pointed them at something alive.

        Watch mode is where this belongs anyway, and constants.py has said so
        since the palette was written: TOOL_NONE is documented as "just watch;
        clicks do nothing". So this fills a mode that currently does nothing at
        all rather than taking a click away from one that does something, the
        rule is one sentence the user can hold ("no tool selected? clicking a
        person watches them"), the palette itself is the mode indicator, and
        clicking the selected tool again already puts it away and drops you back
        here. Right/middle are untouched: they pan, and panning while locked is
        deliberately still allowed (see Camera.follow).

        RELEASING, in three ways, because a lock the user cannot get out of is
        worse than no lock: click bare ground, press 0/Home (which already
        reaches ``Camera.resume_follow``), or lose the man - Camera drops the
        lock itself when he dies OR is abducted, which are different code paths
        and are tested separately.

        A click on bare ground releases only if something is actually locked. A
        left click in watch mode did nothing before, and it is better that it
        keeps doing nothing than that it quietly cancels a right-drag's manual
        hold and yanks the view back to the colony.
        """
        cam = self._camera()
        if cam is None or self.world is None or not events:
            return
        if self.tools.tool != TOOL_NONE:
            return
        for ev in events:
            try:
                if ev.get("type") != "down" or int(ev.get("button") or 0) != 1:
                    continue
                # The palette is drawn in window px over the top of the scene,
                # so a click that lands on an icon is a tool choice and never a
                # world click. tools.py makes the same test; it has to be made
                # here too because this runs first.
                if toolbar.hit_test((ev.get("sx", 0), ev.get("sy", 0)),
                                    len(TOOLS)) is not None:
                    continue
                wx, wy = ev.get("wx"), ev.get("wy")
                if wx is None or wy is None:
                    continue                  # a letterbox bar, not the world
                agent = self._pick_agent(float(wx), float(wy))
                if agent is not None:
                    if cam.lock_to(agent.id):
                        log.info("following %s (id %d)",
                                 getattr(agent, "name", "?"), int(agent.id))
                elif cam.locked:
                    cam.resume_follow()
                    log.info("released the view back to the colony")
            except Exception:
                log.debug("follow click failed", exc_info=True)

    def _roster_click(self, events):
        """Follow whoever's name was clicked in the stats panel.

        Returns the events the rest of the frame should still see, with any
        click that hit a roster row removed - see the note at the call site for
        why consuming is required rather than merely going first.

        THE TOOL DOES NOT GET A VETO HERE, unlike the world-side click at
        :meth:`_follow_click`. Out in the world a selected tool must win, because
        the hand exists to pick a colonist up and the lightning exists to smite
        one, and letting follow outrank them would mean neither could ever be
        aimed at a person again. The panel is not the world: nothing in the
        palette acts on a HUD row, so there is no click here for a tool to lose.

        Releasing is deliberately NOT wired to a click on empty panel - the
        panel is mostly empty and a stray click on the stockpile line should not
        drop the lock. Bare ground, 0/Home, death and abduction already release.
        """
        cam = self._camera()
        if cam is None or self.world is None or not events:
            return events
        # Local, like every other hud import in this file - render/ is not on
        # app.py's module-level import graph and this is not the place to put it
        # there.
        from .render import hud
        kept = []
        for ev in events:
            try:
                if ev.get("type") == "down" and int(ev.get("button") or 0) == 1:
                    aid = hud.agent_at((ev.get("sx"), ev.get("sy")))
                    if aid is not None:
                        if cam.lock_to(int(aid)):
                            who = self.world.population.by_id(int(aid))
                            log.info("following %s (id %d) from the roster",
                                     getattr(who, "name", "?"), int(aid))
                        continue                  # consumed
            except Exception:
                log.debug("roster click failed", exc_info=True)
            kept.append(ev)
        return kept

    def _pick_agent(self, wx: float, wy: float):
        """The living colonist under a world-space click, or None.

        FOLLOW_PICK_RADIUS carries the whole justification for the number (it is
        the figure's ink-diffed worst-frame half-width, +/-29 px, not a guess).
        What is decided here is the geometry it is applied to and the crowd rule:

        * measured from the MID-TORSO, ``a.y`` being the feet and the body about
          AGENT_HEIGHT tall, so one radius covers the man in both axes and a
          click on somebody's head is a hit rather than a near miss;
        * overlapping agents resolve to the NEAREST centre - in a crowd the man
          you clicked closest to is the man you meant - with the lowest id as
          the tie-break, so two colonists standing in the same place always give
          the same answer for the same click instead of depending on roster
          order.
        """
        best = None
        best_d = None
        r2 = FOLLOW_PICK_RADIUS * FOLLOW_PICK_RADIUS
        mid = AGENT_HEIGHT * 0.5
        try:
            agents = self.world.population.alive_agents()
        except Exception:
            return None
        for a in agents:
            try:
                dx = float(a.x) - wx
                dy = (float(a.y) - mid) - wy
            except (AttributeError, TypeError, ValueError):
                continue
            d = dx * dx + dy * dy
            if d > r2:
                continue
            if (best is None or d < best_d
                    or (d == best_d and int(a.id) < int(best.id))):
                best, best_d = a, d
        return best

    def _sync_follow_hud(self) -> None:
        """Tell the HUD who is being followed, so the roster can say so.

        A module dial rather than a draw argument for the reason hud.FOLLOW_ID
        gives: the wallpaper is where this app is actually watched, it has no
        window and no caption, and it goes through the same draw_hud. The
        caption gets the name as well, for the preview window - two channels,
        because the two outputs have different amounts of room.
        """
        try:
            from .render import hud
            cam = self._camera()
            hud.FOLLOW_ID = None if cam is None else cam.lock_id
        except Exception:
            log.debug("could not publish the follow id", exc_info=True)

    def _followed_name(self) -> str:
        """Name of the locked colonist, or "" - for the window caption."""
        try:
            cam = self._camera()
            wanted = None if cam is None else cam.lock_id
            if wanted is None or self.world is None:
                return ""
            a = self.world.population.by_id(wanted)
            return "" if a is None else str(getattr(a, "name", ""))
        except Exception:
            return ""

    def _pan(self, residual: float, reset: bool = False) -> None:
        """Spill the preview's leftover pan into the world camera.

        Preview absorbs what it can inside the 1600 px frame and hands back the
        rest (see Preview.pan_world). At zoom 1 it absorbs nothing, so the whole
        step arrives here and WASD scrolls the world; zoomed in, the frame pan
        takes the step until it hits a frame edge and only the surplus arrives.
        One rule, no modes, and no key in this file has to know the zoom.

        A right/middle-button drag arrives by the same road, summed in by the
        caller: Preview.pan_by hands its residual to pump_events exactly as
        pan_keys hands back its own. Without that the mouse was the one input
        that could not move a 6400 px world, which is the first thing anyone
        tries.

        nudge() arms the camera's manual hold, so follow() stops fighting the
        user for a few seconds; 0/Home resets the preview's view *and* hands the
        colony back to follow(), because half a reset is worse than none - which
        now includes dropping a click-to-follow lock, since a key that undid the
        pan but left the camera welded to one colonist would look broken. A pan
        on its own does NOT drop the lock: looking around while following
        somebody is a thing people do, and the lock resumes when the hold
        lapses.
        """
        cam = self._camera()
        if cam is None:
            return
        try:
            if reset:
                cam.resume_follow()
            elif residual:
                cam.nudge(float(residual))
        except Exception:
            log.debug("camera pan failed", exc_info=True)

    def _to_world(self, events):
        """Add the camera offset back onto frame-space pointer events.

        Preview reports fx/fy (frame space); shell/tools.py and sim/interact.py
        read wx/wy and have always meant world space. Unremapped, the two were
        silently the same number and a click near the right edge of the frame
        would strike lightning up to 4800 px from where the user pointed.

        ``camera_x_presented``, not ``camera.x``: it is the offset baked into
        the frame the user was actually looking at when they clicked, which is
        last frame's value, since this runs before draw(). That is the correct
        answer rather than an off-by-one - the click refers to pixels already on
        screen. Defaulting it to 0.0 makes a camera-less renderer the identity
        transform, which is what this app did before the world grew.
        """
        if not events:
            return events
        ox = float(getattr(self.renderer, "camera_x_presented", 0.0) or 0.0)
        for ev in events:
            fx = ev.get("fx")
            ev["wx"] = None if fx is None else fx + ox
            ev["wy"] = ev.get("fy")
        return events

    def _draw_window_overlay(self, screen: pygame.Surface) -> None:
        """Everything drawn in *window* pixels, after the world is scaled in.

        The HUD goes down first and the tool palette on top, so a tooltip is
        never buried under the stats panel. Both halves fail soft - an overlay
        that raises must not cost the frame.

        The pointer is handed to the HUD here and nowhere else. It is in window
        pixels, which is the space the panels are drawn in, and it is passed
        only on this path: the same draw_hud runs again against the 1600x1000
        wallpaper frame, where a pointer position would be meaningless and a
        tooltip would be baked into the desktop permanently.

        ``get_focused`` is what stops a tooltip appearing under a pointer that
        is somewhere else entirely - pygame keeps reporting the last position it
        saw once the mouse leaves the window, so without it the panel would show
        detail for whatever the cursor was over when it left.
        """
        try:
            mouse = pygame.mouse.get_pos() if pygame.mouse.get_focused() else None
            self.renderer.draw_hud(screen, self.world, mouse=mouse)
        except Exception:
            log.debug("window hud draw failed", exc_info=True)
        self.tools.draw_overlay(screen)

    def _bake_hud(self, frame: pygame.Surface) -> None:
        """Bake the HUD into the world frame, at most once per frame.

        The wallpaper has no window for the panels to anchor to, so they are
        drawn into the 1600x1000 image at exactly the position they have always
        held, screen shake included. The once-per-frame guard is not an
        optimisation: the wallpaper push and --capture consume the *same*
        surface, and the panel is translucent, so a second blit would visibly
        darken it.
        """
        if self._hud_frame == self._frame_no:
            return
        self._hud_frame = self._frame_no
        try:
            self.renderer.draw_hud(frame, self.world, shake=True)
        except Exception:
            log.debug("wallpaper hud draw failed", exc_info=True)

    def _hud(self) -> str:
        w = self.world
        pop = len(w.population.alive_agents())
        z = getattr(self.preview, "zoom", 1.0)
        zoom = f" - {z:.1f}x" if z > 1.01 else ""
        # Leading, not trailing: the taskbar button and Alt-Tab both truncate the
        # caption from the right, and a paused world that has been paused since
        # some previous session looks exactly like a broken one.
        paused = "[PAUSED] " if self.cfg.paused else ""
        # Leading too, and for the same reason: it is the state most likely to
        # be mistaken for the app misbehaving ("why won't the view go back?").
        who = self._followed_name()
        follow = f"[following {who}] " if who else ""
        return (f"{paused}{follow}Backgrounded - {w.events.scene} - pop {pop} "
                f"gen {w.population.generation} - "
                f"wood {w.stockpile.get('wood',0)} stone {w.stockpile.get('stone',0)} "
                f"food {w.stockpile.get('food',0)}{zoom} - "
                f"{self.clock.get_fps():.0f} fps  "
                # "roam" and "follow" rather than "pan" and "reset": WASD now
                # walks the camera across a world four frames wide, and 0 does
                # not merely undo a zoom, it hands the colony back to follow().
                f"[wheel=zoom wasd=roam []=text 0=follow F11=full]")

    def _save_capture(self, frame: pygame.Surface) -> None:
        self._captures += 1
        p = paths.CAPTURE_DIR / f"frame_{self._captures:03d}_{self.world.events.scene}.png"
        try:
            pygame.image.save(frame, str(p))
            log.info("capture -> %s", p)
        except Exception:
            log.exception("capture failed")

    def _shutdown(self) -> None:
        log.info("shutting down")
        if self._crash is None:
            try:
                persist.save_world(self.world)
            except Exception:
                log.exception("final save failed")
        else:
            try:
                self._shutdown_after_crash()
            except Exception:
                log.exception("crash shutdown failed; save.json left untouched")
        try:
            self.wallpaper.stop()
            if self.cfg.restore_wallpaper_on_exit:
                self.wallpaper.restore()
        except Exception:
            log.exception("wallpaper shutdown failed")
        try:
            self.tools.release_all(self.world)
        except Exception:
            pass
        try:
            self.tray.stop()
        except Exception:
            log.exception("tray shutdown failed")
        try:
            pygame.quit()
        except Exception:
            pass
        self._save_config()
        _clear_instance_record()

    def _shutdown_after_crash(self) -> None:
        """What to write when the frame loop DIED rather than ended.

        The rule is: a world that just killed the frame loop does not get
        written over a save that started cleanly. Both alternatives are worse.
        Saving anyway is what made a single huge-but-finite coordinate
        permanent - the renderer raised on frame one, the loop ended, _shutdown
        wrote the same poison back, and the next launch did it again forever,
        with no console and no window under run.pyw. Refusing and doing nothing
        else would throw away the session instead.

        So it does three things:

        1. **The world in memory goes to a sidecar**, not to save.json. Nothing
           is lost - it is a complete save under a name a human can rename back.
        2. **save.json is left exactly as it was.** On the normal path that
           costs at most AUTOSAVE_SEC (60 s), because the autosave has been
           writing this same world all along; the *last good* state is already
           there. On the poisoned-save path it costs nothing at all, because the
           loop died before the world ever advanced.
        3. **If the loop died without the world advancing a single tick**, the
           file it was loaded from is the prime suspect, and it is moved aside
           so the next launch starts instead of repeating. That test is the
           narrow one deliberately: it is exactly the poisoned-save signature
           (measured: world_time unchanged at 16966.0999 across two launches),
           and a crash three hours into a healthy session fails it, so an
           unlucky moment can never cost anybody their colony.

        A world built by --fresh, or after Start Over, is not evidence about
        anything on disk, so step 3 only fires for a world that came off disk.
        """
        stalled = False
        try:
            stalled = float(self.world.world_time) <= self._start_world_time
        except Exception:
            stalled = False

        payload = None
        try:
            payload = json.dumps(self.world.to_dict(),
                                 separators=(",", ":")).encode("utf-8")
        except Exception:
            log.exception("the crashed world could not even be serialised")
        if payload:
            self._set_aside(
                CRASH_PREFIX, payload,
                "the frame loop died; NOT writing this world over save.json - "
                "it has been kept separately")
            self._prune(CRASH_PREFIX, CRASH_KEEP)

        if not (stalled and self._from_disk):
            log.error("save.json has been left exactly as it was. The colony in "
                      "it is the last autosave, at most %.0f s behind.",
                      AUTOSAVE_SEC)
            return

        try:
            raw = paths.SAVE_PATH.read_bytes()
        except Exception:
            log.exception("could not read save.json to set it aside")
            return
        kept = self._set_aside(
            SUSPECT_PREFIX, raw,
            "the frame loop died on the very first frame and the world never "
            "advanced a tick, so the save it was loaded from is what did it; "
            "it has been moved aside and the next launch will start a new "
            "colony")
        if kept is None:
            return
        try:
            paths.SAVE_PATH.unlink()
        except Exception:
            log.exception("kept a copy at %s but could not remove save.json; "
                          "the next launch will load it again", kept)
            return
        # DELIBERATELY NOT PRUNED. Every file in this family is a colony this
        # app took away from its owner, and there is no rule for picking which
        # of those to destroy that is worth the disk it saves. Newest-first
        # deletes the original; oldest-first deletes it too as soon as two
        # worthless copies occupy the sparing slots. A retention policy here is
        # all downside: the files are one per brick incident, an incident is
        # rare, and a save is ~160 KB. The crash family above IS capped because
        # those are diagnostic snapshots of a world nobody was playing.


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="backgrounded")
    p.add_argument("--fresh", action="store_true", help="ignore any saved world")
    p.add_argument("--scene", choices=list(SCENES), help="opening scene")
    p.add_argument("--hide", action="store_true", help="start with the window hidden")
    p.add_argument("--no-wallpaper", action="store_true",
                   help="do not touch the desktop wallpaper")
    p.add_argument("--capture", type=int, default=0,
                   help="save N png frames to the captures dir then keep running")
    p.add_argument("--capture-every", type=int, default=60,
                   help="sim ticks between captures")
    p.add_argument("--exit-after", type=float, default=0.0,
                   help="exit after N sim seconds (testing)")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--allow-multi", action="store_true",
                   help="permit a second instance (they will fight over the "
                        "wallpaper; for debugging only)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths.ensure_dirs()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.FileHandler(paths.LOG_PATH, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    if not args.allow_multi:
        # Deliberately before App(): the takeover does not return until the old
        # copy is gone, so by the time the renderer captures the "original"
        # wallpaper the other instance has already restored the real one.
        if not _acquire_single_instance():
            if host.WINDOW_IS_THE_APP:
                log.error("Backgrounded is already running - look for its "
                          "window. Close it, or use --allow-multi to run a "
                          "second copy against the same save.")
            else:
                log.error("Backgrounded is already running and could not be "
                          "taken over - look for the tray icon. Use "
                          "--allow-multi to run a second copy anyway.")
            return 2
        _write_instance_record()
        _listen_for_quit()
    return App(args).run()
