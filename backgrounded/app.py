"""App - owns the main loop and the two worker threads.

Thread ownership (see docs/ARCHITECTURE.md section 1):
  main       pygame render loop, fixed-timestep sim, command dispatch
  tray       Win32 message pump, pushes commands onto cmd_q
  wallpaper  blocking disk + SystemParametersInfoW writes
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

from . import paths, persist
from .config import Config
from .constants import (
    AUTOSAVE_SEC, RENDER_SIZE, SCENES, SIM_DT, TARGET_FPS,
)
from .render.renderer import Renderer
from .shell.preview import Preview
from .shell.tools import ToolController
from .shell.tray import Tray
from .shell.wallpaper import WallpaperWriter
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


def _acquire_single_instance() -> bool:
    """Take a named mutex so only one copy ever runs.

    Two instances both write the wallpaper A/B pair and both call
    SystemParametersInfoW, so they overwrite each other's files mid-write and
    the desktop flickers between two different worlds. Worse, closing the
    preview window only *hides* it (the app lives in the tray), so it is easy
    to leave orphans running without noticing. The handle is deliberately
    leaked: it lives as long as the process and Windows frees it on exit.
    """
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.restype = ctypes.c_void_p
        handle = k32.CreateMutexW(None, False, "Local\\BackgroundedSingleInstance")
        if not handle:
            return True                      # cannot tell; allow startup
        ERROR_ALREADY_EXISTS = 183
        if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
            return False
        globals()["_INSTANCE_MUTEX"] = handle
        return True
    except Exception:
        return True


def _screen_size() -> tuple[int, int]:
    try:
        u = ctypes.windll.user32
        u.SetProcessDPIAware()
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
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
        self.tools.handle(self._to_world(pointer.get("pointer")),
                          self.world, self.world.world_time)
        step = pointer.get("hud_scale")
        if step:
            self._nudge_hud_scale(int(step))
        if pointer.get("closed"):
            self.cfg.show_window = False
            self.preview.ensure_window(False)
            self._save_config("show_window")

        # fixed-timestep sim, scaled by the speed setting
        if not self.cfg.paused:
            self._sim_accum += dt * self.cfg.sim_speed
            steps = 0
            while self._sim_accum >= SIM_DT and steps < 8:
                self.world.tick(SIM_DT)
                self._sim_accum -= SIM_DT
                steps += 1

        frame = self.renderer.draw(self.world, dt)

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

        if self.args.exit_after and self.world.world_time >= self.args.exit_after:
            log.info("exit-after reached")
            self.running = False

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
        colony back to follow(), because half a reset is worse than none.
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
        return (f"{paused}Backgrounded - {w.events.scene} - pop {pop} "
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
    if not args.allow_multi and not _acquire_single_instance():
        log.error("Backgrounded is already running - look for the tray icon. "
                  "Use --allow-multi to override.")
        return 2
    return App(args).run()
