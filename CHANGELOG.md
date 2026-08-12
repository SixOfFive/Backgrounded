# Changelog

Newest first. This file starts on 2026-08-11; everything before that date is in
`git log`, which is where the earlier history of the project lives.

## 2026-08-12

### Added

- **An in-window settings menu** — press `M` or click the gear in the
  bottom-right corner. It exists because the Linux port left no runtime control
  surface at all: the tray menu was the only way to reach scene, speed, window
  size, the HUD toggles, pause, and the world actions, and off Windows there is
  no tray.
- `backgrounded/render/menu.py` (layout, hit-test, drawing) and
  `backgrounded/shell/menu.py` (`MenuController`), split the same way
  `render/toolbar.py` and `shell/tools.py` already are — geometry lives beside
  the drawing so the hit-test cannot disagree with what is on screen.
  - **It is a second producer for `cmd_q`, not a second implementation.** It
    emits exactly the `(verb, payload)` tuples the tray emits and takes the
    same `(cmd_q, get_state)` pair `Tray` takes, so `App._handle` needed no
    change at all and a verb added there is reachable from both menus at once.
  - Flat panel of chips rather than the tray's four nested popups: a submenu
    that opens on hover is miserable over a moving scene, and the flat form
    answers "what is the speed right now?" without opening anything.
  - **`Start over` asks twice.** It is the only entry in the panel that
    destroys a colony that cannot be got back, and it sits four chips from
    `Save now`. The tray never needed a confirm because reaching it meant
    walking a popup; here it is one click. It also carries a red border at all
    times, not only on hover — a colour that appears under the cursor tells you
    what you are about to click, which is too late to be the point.
  - Clicks that land on the menu are **consumed**, like `App._roster_click`:
    the panel is painted over the scene, so every pixel of it also has a
    perfectly good world position underneath, and letting one through would
    change the speed *and* strike lightning wherever the chip was sitting. A
    click on empty scene closes the panel and is spent doing so, rather than
    also planting a tree.
  - The gear button is drawn only where there is no tray. `M` works on both,
    because a second way in costs nothing, but Windows does not grow a button
    for a menu it already has.

### Changed

- `SPEEDS` and `WINDOW_SCALES` moved from `shell/tray.py` (Windows-only code)
  to `constants.py`. Both menus dispatch by index into them, so they have to be
  one list rather than two that drift — a speed the tray offers and the panel
  does not is a setting a Windows user can reach and a Linux user cannot.
- `Preview` reports two new flags, `menu_toggle` (`M`) and `menu_close`
  (`Escape`, once fullscreen has had its claim on the key). Reported rather
  than applied, like `reset_view` and `hud_scale` — the window does not own the
  menu's state.
- The window title now leads with `m=menu`. It is the one place a first-time
  user is already looking, and "where do I change the scene now?" was the first
  question the Linux build got.
- `render/toolbar.py` asks for DejaVu/Liberation as well as Segoe UI and Arial.
  Neither Windows name exists on a stock Linux box, so tool tooltips were
  silently falling through to pygame's builtin font.

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
