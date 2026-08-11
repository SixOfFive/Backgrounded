# Changelog

Newest first. This file starts on 2026-08-11; everything before that date is in
`git log`, which is where the earlier history of the project lives.

## 2026-08-11

### Added

- **Runs on Linux as an ordinary windowed app.** The simulation and renderer
  were already portable — nothing in `sim/` or `render/` had ever touched the
  Win32 API — so the port is confined to the four modules that talk to the
  desktop shell. `tools/smoke.py` passes all 80 checks unchanged on Debian 13.
- `backgrounded/host.py` — three flags (`WALLPAPER_SUPPORTED`, `TRAY_SUPPORTED`,
  `WINDOW_IS_THE_APP`) that every platform decision now reads. Flags rather
  than scattered `sys.platform` tests, because the question being asked is
  never "is this Windows" but "is there a tray to hide in"; those two only
  happen to coincide today.
- `backgrounded.sh` — the Linux launcher, the counterpart to
  `Backgrounded.bat`. Runs in the foreground by default (you get the log,
  Ctrl+C exits cleanly); `--detach`, `--setup`, `--yes` and `--` are its own
  flags and everything else is forwarded to the app.
  - It installs the three packages into a virtualenv at
    `~/.local/share/Backgrounded/venv` rather than into the system Python,
    because Debian/Ubuntu mark theirs externally managed (PEP 668) and refuse.
    The virtualenv is deliberately **not** beside the repo: this project is
    routinely kept on a CIFS share, where `python3 -m venv` fails outright with
    EIO because the mount has no Unix extensions and cannot make the
    `bin/python` symlink.
  - If pygame, Pillow and numpy already import from the system interpreter, no
    virtualenv is made and that interpreter is used.
- POSIX single-instance locking (`flock` on `instance.lock` in the app dir).
  `flock` rather than a pid file because the kernel releases it however the
  process dies, so a `kill -9` cannot leave a stale lock that blocks every
  later launch. Deliberately **not** the takeover the Windows path performs:
  takeover exists there because two copies fight over the desktop wallpaper and
  because a copy hidden in the tray is easy to lose track of, and neither is
  true of a window you can see. The newcomer stands down and says so.

### Changed

- **Closing the window now ends the run where there is no tray.** On Windows
  the X hides the window and the tray icon brings it back; doing that on Linux
  would strand a running colony behind an invisible window with no way to reach
  it and no way to quit. `App._frame` branches on `host.WINDOW_IS_THE_APP` and
  leaves through the normal shutdown, so the world is saved.
- `paths.APP_DIR` follows `$XDG_DATA_HOME` (`~/.local/share/Backgrounded`) off
  Windows instead of the literal `~/AppData/Local`. Both branches read their
  environment variable before falling back to the conventional path, which is
  what keeps `tools/probe.py` and `tools/smoke.py` able to isolate a test run
  from the real save.
- `paths.WALLPAPER_DIR` points inside the app dir off Windows and is no longer
  created by `ensure_dirs`, so a Linux run leaves nothing in `~/Pictures`.
- `Tray.start()` and `WallpaperWriter.start()` return without spawning a thread
  where the platform has no counterpart, and `capture_original` / `restore` are
  guarded so a Linux launch and exit are not bracketed by warnings about a
  wallpaper query that was never going to work. The app runs with no worker
  threads at all there.
- `Preview` show/hide goes through SDL (`pygame.Window.from_display_module`)
  instead of `ShowWindow` on an HWND off Windows.
- `wallpaper_enabled` is forced off where unsupported as a **session** override
  via the existing `_cli_keys` mechanism, so it is never written back to
  `config.json` — it is a fact about the machine, not a preference, and
  persisting it would silently disable the wallpaper in a config later opened
  on Windows.
- `tools/probe.py` and `tools/frametime.py` only set `SDL_VIDEODRIVER=windib`
  on Windows; elsewhere SDL picks its own driver. Both still honour an
  `SDL_VIDEODRIVER` already in the environment.
- `README.md` documents both platforms: how to launch, where files live, and
  which sections (tray menu, wallpaper handling) are Windows-only.

### Fixed

- Three module-level bindings made the package unimportable off Windows before
  a single frame was drawn — `ctypes.WinDLL` in `shell/preview.py`,
  `shell/tray.py` and `shell/wallpaper.py`, plus `ctypes.WINFUNCTYPE` and the
  module-scope `_bind()` call in `shell/tray.py`. All are conditional now.
  `ctypes.wintypes` needed no guard: it imports fine on any CPython 3, being
  nothing but type aliases.
