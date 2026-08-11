"""Desktop wallpaper output: an atomic A/B BMP writer on its own thread.

Verified facts this module is built around:

* ``SystemParametersInfoW(SPI_SETDESKWALLPAPER)`` costs roughly 200 ms. It
  must never run on the sim thread, hence the worker.
* Overwriting the *currently active* wallpaper file fails transiently with
  ``OSError EINVAL`` while Explorer or Defender has it open. So: write to a
  ``.tmp`` sibling, ``os.replace`` into place, and alternate between
  ``wallpaper_a.bmp`` and ``wallpaper_b.bmp`` so the file being written is
  never the one Windows currently holds.
* Frames are disposable. The queue holds exactly one; a producer that finds
  it full throws the stale frame away and enqueues the fresh one. Dropping
  frames is correct behaviour, not a bug.

Nothing in here raises. A failed frame is counted and skipped.
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import queue
import shutil
import threading
import time
from ctypes import wintypes
from pathlib import Path

from .. import host, paths

log = logging.getLogger(__name__)


# --------------------------------------------------- original-wallpaper record --
def _is_program_managed(path: str) -> bool:
    """True if *path* is one of our own files - a rendered frame or our backup
    copy - because it sits in the wallpaper output directory.

    Such a path must NEVER be adopted as "the original": it is exactly what a
    crashed run leaves on the desktop, and recording it would overwrite the real
    original with our own image (the failure this whole record exists to prevent).
    """
    if not path:
        return False
    try:
        parent = os.path.normcase(os.path.normpath(os.path.dirname(path)))
        home = os.path.normcase(os.path.normpath(str(paths.WALLPAPER_DIR)))
        return parent == home
    except Exception:
        return False


def _load_record() -> dict:
    try:
        data = json.loads(paths.ORIGINAL_RECORD.read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_record(original: str, backup: str | None) -> None:
    try:
        paths.ensure_dirs()
        rec: dict[str, str] = {"original": original}
        if backup:
            rec["backup"] = backup
        tmp = paths.ORIGINAL_RECORD.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=2), "utf-8")
        os.replace(tmp, paths.ORIGINAL_RECORD)
    except Exception as exc:
        log.warning("could not persist original-wallpaper record: %s", exc)


def _copy_backup(src: str) -> str | None:
    """Keep a byte-for-byte copy of the real original inside the wallpaper dir
    (a location the shell will load from), so it can be re-applied even after
    the user moves or deletes the source. Returns the copy path or None."""
    try:
        paths.ensure_dirs()
        ext = os.path.splitext(src)[1] or ".jpg"
        dst = str(paths.ORIGINAL_BACKUP) + ext
        try:
            if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
                return dst                       # already backed up, no churn
        except Exception:
            pass
        shutil.copy2(src, dst)
        log.info("backed up original wallpaper -> %s", dst)
        return dst
    except Exception as exc:
        log.warning("could not back up original wallpaper %s: %s", src, exc)
        return None

try:
    from PIL import Image
except Exception as _exc:                                     # pragma: no cover
    Image = None                                              # type: ignore[assignment]
    log.error("Pillow unavailable (%s); wallpaper output disabled", _exc)

# ---------------------------------------------------------------- win32 ABI --
#
# Conditional because ``ctypes.WinDLL`` does not exist off Windows - it is an
# AttributeError at import, which would take the whole app down before it drew
# a frame. Where ``host.WALLPAPER_SUPPORTED`` is false the class below never
# starts its thread and none of this is reached; see :meth:`WallpaperWriter.start`.

if host.WALLPAPER_SUPPORTED:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SystemParametersInfoW.restype = wintypes.BOOL
    user32.SystemParametersInfoW.argtypes = [
        wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT]
else:
    user32 = None

SPI_SETDESKWALLPAPER = 0x0014
SPI_GETDESKWALLPAPER = 0x0073
SPIF_UPDATEINIFILE = 0x01
SPIF_SENDCHANGE = 0x02

#: Windows documents MAX_PATH for the wallpaper query; 600 chars is headroom.
_PATH_CHARS = 600

#: Sleep between the write attempts. One initial try plus these three retries.
_BACKOFF: tuple[float, ...] = (0.05, 0.15, 0.4)


class _GUID(ctypes.Structure):
    _fields_ = [("d1", wintypes.DWORD), ("d2", wintypes.WORD),
                ("d3", wintypes.WORD), ("d4", ctypes.c_byte * 8)]


class _DesktopWallpaperCOM:
    """Thin ctypes binding for the shell's IDesktopWallpaper.

    Preferred over SystemParametersInfo for one decisive reason: SPI returns
    *success* for a path the shell will not actually load, updates the registry,
    and leaves the desktop showing its background colour. IDesktopWallpaper
    returns a real HRESULT (0x80070002 for a rejected location), which is the
    only way to find out something is wrong without screenshotting the desktop.
    It also addresses each monitor explicitly, which matters on multi-head
    setups.
    """

    CLSID = "{C2CF3110-460E-4fc1-B9D0-8A1C0C9CC4BD}"
    IID = "{B92B56A9-8B55-4E14-9A89-0199BBB6F93B}"
    DWPOS_FILL = 4

    def __init__(self) -> None:
        self._p: ctypes.c_void_p | None = None
        self._set = None
        self._get = None
        self._setpos = None
        try:
            ole = ctypes.WinDLL("ole32")
            ole.CoInitialize(None)

            def guid(s: str) -> _GUID:
                g = _GUID()
                ole.CLSIDFromString(ctypes.c_wchar_p(s), ctypes.byref(g))
                return g

            p = ctypes.c_void_p()
            CLSCTX_ALL = 23        # INPROC_SERVER is not enough for this class
            hr = ole.CoCreateInstance(ctypes.byref(guid(self.CLSID)), None,
                                      CLSCTX_ALL, ctypes.byref(guid(self.IID)),
                                      ctypes.byref(p))
            if hr != 0 or not p:
                log.info("IDesktopWallpaper unavailable (hr=0x%08X); "
                         "falling back to SystemParametersInfo", hr & 0xFFFFFFFF)
                return
            vt = ctypes.cast(
                p, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            F = ctypes.WINFUNCTYPE
            #  3 SetWallpaper  4 GetWallpaper  10 SetPosition
            self._set = F(ctypes.c_long, ctypes.c_void_p,
                          ctypes.c_wchar_p, ctypes.c_wchar_p)(vt[3])
            self._get = F(ctypes.c_long, ctypes.c_void_p, ctypes.c_wchar_p,
                          ctypes.POINTER(ctypes.c_wchar_p))(vt[4])
            self._setpos = F(ctypes.c_long, ctypes.c_void_p, ctypes.c_int)(vt[10])
            self._p = p
            try:
                self._setpos(self._p, self.DWPOS_FILL)
            except Exception:
                pass
        except Exception as exc:
            log.info("IDesktopWallpaper init failed (%s); using SPI", exc)

    @property
    def available(self) -> bool:
        return self._p is not None and self._set is not None

    def set(self, path: str) -> int:
        """Set on every monitor. Returns an HRESULT (0 == success)."""
        if not self.available:
            return -1
        try:
            return int(self._set(self._p, None, ctypes.c_wchar_p(path)))
        except Exception as exc:
            log.warning("IDesktopWallpaper.SetWallpaper threw: %s", exc)
            return -1

    def get(self) -> str | None:
        if not self.available or self._get is None:
            return None
        try:
            buf = ctypes.c_wchar_p()
            if self._get(self._p, None, ctypes.byref(buf)) != 0:
                return None
            return buf.value or None
        except Exception:
            return None


_com_tls = threading.local()


def _com_for_thread() -> _DesktopWallpaperCOM:
    """One COM object per thread.

    IDesktopWallpaper is apartment-threaded: an interface pointer created on
    one thread cannot be called from another. Creating it at import time on the
    main thread and calling it from the wallpaper worker returns
    RPC_E_WRONG_THREAD (0x8001010E) on every single frame.
    """
    com = getattr(_com_tls, "com", None)
    if com is None:
        com = _DesktopWallpaperCOM()
        _com_tls.com = com
    return com


def _set_wallpaper(path: str, flags: int) -> bool:
    """Apply a wallpaper. Prefers the COM interface, falls back to SPI."""
    com = _com_for_thread()
    if com.available:
        hr = com.set(path)
        if hr == 0:
            return True
        if (hr & 0xFFFFFFFF) == 0x80070002:
            # The shell will not load wallpapers from every directory - notably
            # anywhere under %LOCALAPPDATA%. See paths.WALLPAPER_DIR.
            log.error("the shell refused %s (FILE_NOT_FOUND although the file "
                      "exists) - this location is not usable for wallpapers",
                      path)
            return False
        log.warning("IDesktopWallpaper.SetWallpaper hr=0x%08X; trying SPI",
                    hr & 0xFFFFFFFF)
    try:
        ok = user32.SystemParametersInfoW(
            SPI_SETDESKWALLPAPER, 0, ctypes.c_wchar_p(path), flags)
        return bool(ok)
    except Exception as exc:                                  # pragma: no cover
        log.warning("SystemParametersInfoW threw: %s", exc)
        return False


class WallpaperWriter:
    """Pushes rendered frames onto the desktop wallpaper.

    Usage from the app::

        w = WallpaperWriter(screen_size)
        w.capture_original()      # remember what to put back
        w.start()
        ...
        w.submit(surface_rgb_bytes, (RENDER_W, RENDER_H))    # sim thread
        ...
        w.stop()
        w.restore()               # explicit; stop() alone leaves our frame up

    ``frames_written`` / ``failures`` / ``frames_dropped`` are plain ints kept
    for the HUD and the log; they are only written by the worker thread and
    read everywhere else, which is safe enough for diagnostics.
    """

    def __init__(self, screen_size: tuple[int, int]) -> None:
        w = max(1, int(screen_size[0]))
        h = max(1, int(screen_size[1]))
        self.screen_size: tuple[int, int] = (w, h)

        self.original_path: str | None = None
        self._backup_path: str | None = None
        self.frames_written: int = 0
        self.failures: int = 0
        self.frames_dropped: int = 0
        self.last_error: str | None = None

        self._paths: tuple[Path, Path] = (paths.WALLPAPER_A, paths.WALLPAPER_B)
        self._idx: int = 0
        self._q: "queue.Queue[tuple[bytes, tuple[int, int]] | None]" = \
            queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._restored = False

    # ------------------------------------------------------------- capture --

    def _read_current_wallpaper(self) -> str | None:
        """The path Windows currently shows, or None (query failed / solid)."""
        try:
            buf = ctypes.create_unicode_buffer(_PATH_CHARS)
            ok = user32.SystemParametersInfoW(
                SPI_GETDESKWALLPAPER, _PATH_CHARS, ctypes.byref(buf), 0)
        except Exception as exc:
            log.warning("could not read current wallpaper: %s", exc)
            return None
        if not ok:
            log.warning("SPI_GETDESKWALLPAPER failed")
            return None
        return buf.value.strip() or None

    def capture_original(self) -> str | None:
        """Remember what to put back on exit, durably and crash-safely.

        The rule that makes the original un-loseable: a program frame is never
        adopted as the original. So:

        * If the desktop currently shows the user's real wallpaper (anything not
          in our own wallpaper dir), record its path AND keep a byte copy, both
          persisted to disk so they survive a crash or a machine reboot.
        * If it shows one of our frames - which is exactly what a crashed run
          leaves behind - we keep whatever we recorded on a previous clean start
          instead, and never overwrite it.

        Returns the original path (possibly loaded from the persisted record),
        or None if none is known yet.
        """
        if not host.WALLPAPER_SUPPORTED:
            # Nothing was ever taken, so there is nothing to remember. Guarded
            # here rather than left to fail through _read_current_wallpaper,
            # which would log a warning about a query it was never going to be
            # able to make, on every single launch.
            return None
        record = _load_record()
        rec_original = record.get("original")
        rec_backup = record.get("backup")

        current = self._read_current_wallpaper()

        if current and not _is_program_managed(current) and os.path.exists(current):
            self.original_path = current
            self._backup_path = _copy_backup(current)
            _save_record(current, self._backup_path)
            log.info("original wallpaper recorded: %s", current)
            return current

        # Crash leftover (one of our frames), solid colour, or a missing file:
        # fall back to what we already knew, and leave the record untouched.
        if current and _is_program_managed(current):
            log.info("current wallpaper is one of ours (%s); keeping the stored "
                     "original instead of adopting it", current)
        self.original_path = rec_original or None
        self._backup_path = rec_backup or None
        if self.original_path:
            log.info("using stored original wallpaper: %s", self.original_path)
        else:
            log.info("no original wallpaper known yet (none recorded, desktop not "
                     "on a user image)")
        return self.original_path

    # ----------------------------------------------------------- lifecycle --

    def start(self) -> None:
        """Start the writer thread. Idempotent.

        The platform check comes before the Pillow one so that a machine with
        no desktop-wallpaper API says nothing at all, rather than complaining
        about a dependency it would have no use for. Everything else on this
        class is already safe with the thread never started: submit() drops the
        frame, restore() has nothing recorded to put back, and stats() reports
        the zeroes it was initialised with.
        """
        if not host.WALLPAPER_SUPPORTED:
            return
        if Image is None:
            log.error("wallpaper writer not started: Pillow is missing")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            paths.ensure_dirs()
        except Exception as exc:
            log.warning("could not create app directory: %s", exc)
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._run, name="wallpaper", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0, restore: bool = False) -> None:
        """Drain, stop and join the writer thread.

        ``restore`` is off by default so shutdown ordering stays in the app's
        hands; call :meth:`restore` explicitly (it is idempotent).
        """
        self._stopping.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # Unblock a worker parked in q.get().
            self._offer(None)
            thread.join(timeout=timeout)
            if thread.is_alive():
                log.warning("wallpaper thread did not exit within %.1fs", timeout)
        self._thread = None
        if restore:
            self.restore()

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -------------------------------------------------------------- submit --

    def _offer(self, item: "tuple[bytes, tuple[int, int]] | None") -> None:
        """Put onto the depth-1 queue, evicting anything stale. Never blocks."""
        try:
            self._q.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._q.get_nowait()
            self.frames_dropped += 1
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(item)
        except queue.Full:
            # The worker beat us to it; the frame we hold is expendable.
            self.frames_dropped += 1

    def submit(self, rgb_bytes: bytes, size: tuple[int, int]) -> None:
        """Hand a raw RGB frame to the writer thread. Called from the sim
        thread; never blocks and never raises."""
        if self._thread is None or not self._thread.is_alive():
            return
        if self._stopping.is_set():
            return
        try:
            w, h = int(size[0]), int(size[1])
        except Exception:
            return
        if w <= 0 or h <= 0:
            return
        if len(rgb_bytes) < w * h * 3:
            log.warning("wallpaper: frame buffer too small for %dx%d", w, h)
            return
        self._offer((rgb_bytes, (w, h)))

    # -------------------------------------------------------------- worker --

    def _run(self) -> None:
        while True:
            try:
                item = self._q.get()
            except Exception:                                 # pragma: no cover
                break
            if item is None:
                break
            if self._stopping.is_set():
                break
            try:
                self._write_frame(item[0], item[1])
            except Exception as exc:                          # pragma: no cover
                # Belt and braces: _write_frame already swallows everything.
                self.failures += 1
                self.last_error = str(exc)
                log.warning("wallpaper frame failed: %s", exc)

    def _target_path(self) -> Path:
        """The A/B slot that Windows is *not* currently displaying."""
        return self._paths[self._idx]

    def _write_frame(self, rgb_bytes: bytes, size: tuple[int, int]) -> None:
        if Image is None:
            return
        try:
            img = Image.frombytes("RGB", size, rgb_bytes)
            if size != self.screen_size:
                img = img.resize(self.screen_size, Image.LANCZOS)
        except Exception as exc:
            self.failures += 1
            self.last_error = str(exc)
            log.warning("wallpaper: could not build image: %s", exc)
            return

        target = self._target_path()
        tmp = target.with_name(target.name + ".tmp")

        attempts = 1 + len(_BACKOFF)
        for attempt in range(attempts):
            if attempt:
                time.sleep(_BACKOFF[attempt - 1])
                if self._stopping.is_set():
                    return
            try:
                with open(tmp, "wb") as fh:
                    # quality 88 is visually indistinguishable at desktop
                    # scale; subsampling off keeps the thin stickman limbs and
                    # lightning filaments from smearing into mush.
                    img.save(fh, format="JPEG", quality=88, subsampling=0,
                             optimize=False)
                    fh.flush()
                    os.fsync(fh.fileno())     # Explorer must not read a partial file
                os.replace(tmp, target)
                break
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                if attempt == attempts - 1:
                    self.failures += 1
                    log.warning("wallpaper write to %s failed after %d tries: %s",
                                target.name, attempts, exc)
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except Exception:
                        pass
                    return

        # SPIF_UPDATEINIFILE is required, not optional. With SPIF_SENDCHANGE
        # alone the change is never committed to HKCU\Control Panel\Desktop, so
        # the desktop often does not repaint at all - and on a machine using
        # Windows Spotlight the Iris service re-asserts its own image over ours
        # within seconds. Measured: SPI_GETDESKWALLPAPER reported our BMP while
        # the registry still held the Spotlight JPG and the desktop never
        # changed. The extra registry write is cheap next to the ~200 ms call.
        if not _set_wallpaper(str(target), SPIF_UPDATEINIFILE | SPIF_SENDCHANGE):
            self.failures += 1
            self.last_error = "SPI_SETDESKWALLPAPER returned 0"
            log.warning("SPI_SETDESKWALLPAPER failed for %s", target)
            return

        self.frames_written += 1
        # Only flip once the frame is live, so the next write targets the
        # inactive slot.
        self._idx ^= 1

    # ------------------------------------------------------------- restore --

    def restore(self) -> None:
        """Put the user's original wallpaper back. Idempotent, never raises.

        Tries the real original in its real location first; if that file is gone
        (the user moved or deleted it), falls back to our own byte copy, which is
        what makes the restore survive anything short of losing both at once."""
        if not host.WALLPAPER_SUPPORTED:
            # The counterpart to capture_original's guard: without it every
            # clean exit ends on "no original wallpaper available to restore",
            # which is true and completely uninteresting.
            self._restored = True
            return
        if self._restored:
            return
        self._restored = True
        flags = SPIF_UPDATEINIFILE | SPIF_SENDCHANGE

        candidates: list[str] = []
        if self.original_path and not _is_program_managed(self.original_path):
            candidates.append(self.original_path)
        backup = self._backup_path or _load_record().get("backup")
        if backup:
            candidates.append(backup)

        for path in candidates:
            try:
                if not os.path.exists(path):
                    continue
            except Exception:
                continue
            if _set_wallpaper(path, flags):
                log.info("original wallpaper restored: %s", path)
                return

        log.warning("no original wallpaper available to restore; leaving as-is")

    # --------------------------------------------------------------- stats --

    def stats(self) -> dict[str, object]:
        """Diagnostics for the HUD / log."""
        return {
            "frames_written": self.frames_written,
            "failures": self.failures,
            "frames_dropped": self.frames_dropped,
            "last_error": self.last_error,
            "original": self.original_path,
            "alive": self.alive,
        }


if __name__ == "__main__":                                    # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    size = (320, 200)
    writer = WallpaperWriter((640, 400))
    writer.capture_original()
    writer.start()
    # A cheap gradient so the write path is exercised end to end.
    frame = bytearray()
    for y in range(size[1]):
        for x in range(size[0]):
            frame += bytes(((x * 255) // size[0], (y * 255) // size[1], 64))
    writer.submit(bytes(frame), size)
    time.sleep(1.5)
    writer.stop()
    writer.restore()
    print(writer.stats())
