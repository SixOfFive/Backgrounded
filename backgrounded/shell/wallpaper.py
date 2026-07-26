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
import logging
import os
import queue
import threading
import time
from ctypes import wintypes
from pathlib import Path

from .. import paths

log = logging.getLogger(__name__)

try:
    from PIL import Image
except Exception as _exc:                                     # pragma: no cover
    Image = None                                              # type: ignore[assignment]
    log.error("Pillow unavailable (%s); wallpaper output disabled", _exc)

# ---------------------------------------------------------------- win32 ABI --

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.SystemParametersInfoW.restype = wintypes.BOOL
user32.SystemParametersInfoW.argtypes = [
    wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT]

SPI_SETDESKWALLPAPER = 0x0014
SPI_GETDESKWALLPAPER = 0x0073
SPIF_UPDATEINIFILE = 0x01
SPIF_SENDCHANGE = 0x02

#: Windows documents MAX_PATH for the wallpaper query; 600 chars is headroom.
_PATH_CHARS = 600

#: Sleep between the write attempts. One initial try plus these three retries.
_BACKOFF: tuple[float, ...] = (0.05, 0.15, 0.4)


def _set_wallpaper(path: str, flags: int) -> bool:
    """Call SPI_SETDESKWALLPAPER. Returns success; never raises."""
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

    def capture_original(self) -> str | None:
        """Read and remember the wallpaper Windows is currently showing.

        Returns the path, or None if the query failed or the desktop is set
        to a solid colour (empty string).
        """
        try:
            buf = ctypes.create_unicode_buffer(_PATH_CHARS)
            ok = user32.SystemParametersInfoW(
                SPI_GETDESKWALLPAPER, _PATH_CHARS, ctypes.byref(buf), 0)
        except Exception as exc:
            log.warning("could not read current wallpaper: %s", exc)
            return None
        if not ok:
            log.warning("SPI_GETDESKWALLPAPER failed; nothing to restore later")
            return None
        value = buf.value.strip()
        if not value:
            log.info("no original wallpaper file set (solid colour?)")
            return None
        # Never adopt one of our own frames as "the original" - that happens
        # if a previous run was killed before it could restore.
        try:
            ours = {str(p).lower() for p in self._paths}
            if value.lower() in ours:
                log.info("current wallpaper is one of ours; not recording it")
                return None
        except Exception:
            pass
        self.original_path = value
        log.info("original wallpaper recorded: %s", value)
        return value

    # ----------------------------------------------------------- lifecycle --

    def start(self) -> None:
        """Start the writer thread. Idempotent."""
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
        """Put the user's original wallpaper back. Idempotent, never raises."""
        if self._restored:
            return
        self._restored = True
        original = self.original_path
        if not original:
            log.info("no original wallpaper recorded; leaving desktop as-is")
            return
        try:
            exists = os.path.exists(original)
        except Exception:
            exists = False
        if not exists:
            log.warning("original wallpaper %s no longer exists; "
                        "leaving desktop as-is", original)
            return
        flags = SPIF_UPDATEINIFILE | SPIF_SENDCHANGE
        if _set_wallpaper(original, flags):
            log.info("original wallpaper restored: %s", original)
        else:
            log.warning("could not restore original wallpaper: %s", original)

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
