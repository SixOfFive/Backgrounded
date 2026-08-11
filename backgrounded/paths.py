"""Filesystem locations. Everything user-writable lives under one app dir:
``%LOCALAPPDATA%\\Backgrounded`` on Windows, ``$XDG_DATA_HOME/Backgrounded``
(i.e. ``~/.local/share/Backgrounded``) everywhere else."""
from __future__ import annotations

import os
from pathlib import Path

from . import host

APP_NAME = "Backgrounded"


def _app_base() -> Path:
    """The per-user data root this platform expects an app to write into.

    Both branches read their environment variable first and only then fall
    back to the conventional path. That is not merely politeness towards an
    unusual setup: tools/probe.py and tools/smoke.py isolate a test run by
    pointing the variable at a scratch directory, so a hard-coded home path
    here would send probe worlds into the real save.
    """
    if host.IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/AppData/Local")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base)


APP_DIR: Path = _app_base() / APP_NAME
SAVE_PATH: Path = APP_DIR / "save.json"
CORRUPT_SAVE_PATH: Path = APP_DIR / "save.corrupt.json"
CONFIG_PATH: Path = APP_DIR / "config.json"
LOG_PATH: Path = APP_DIR / "backgrounded.log"
CHRONICLE_PATH: Path = APP_DIR / "chronicle.txt"

# Wallpaper frames do NOT live in APP_DIR.
#
# Windows refuses to load a desktop wallpaper out of %LOCALAPPDATA%. Measured
# with IDesktopWallpaper::SetWallpaper against one identical JPEG copied to
# several places:
#
#   C:\Users\<u>\Pictures\...              hr=0x00000000  OK
#   C:\Users\<u>\AppData\Local\Temp\...    hr=0x00000000  OK
#   C:\Users\<u>\AppData\Local\...         hr=0x80070002  FILE_NOT_FOUND
#   C:\Users\<u>\AppData\Local\App\sub\... hr=0x80070002  FILE_NOT_FOUND
#
# The file demonstrably exists in every case, so this is a shell restriction on
# the location, not a problem with the file, its format or its permissions. It
# also fails silently through the legacy SystemParametersInfo path - that call
# returns success and updates the registry, and the desktop just stays on the
# background colour, which is why this looked like a rendering bug for so long.
#
# Pictures rather than Temp: if the process dies without restoring, the desktop
# keeps showing the last frame instead of going black when a cleanup tool
# reclaims Temp.
#
# None of the above applies where the desktop background is not an output at
# all (see backgrounded.host). The constant still has to exist and still has to
# be a real path, because WallpaperWriter names it when it is constructed even
# on a platform where it will never run - but it points inside the app dir
# there, and ensure_dirs does not create it, so a Linux run leaves nothing at
# all in the user's Pictures folder.
WALLPAPER_DIR: Path = (
    Path(os.path.expanduser("~")) / "Pictures" / APP_NAME
    if host.WALLPAPER_SUPPORTED else
    APP_DIR / "wallpaper"
)
WALLPAPER_A: Path = WALLPAPER_DIR / "wallpaper_a.jpg"
WALLPAPER_B: Path = WALLPAPER_DIR / "wallpaper_b.jpg"

# The user's ORIGINAL desktop wallpaper - what to put back on exit.
#
# This is the guard against the failure that actually happened once: the app's
# own frames overwrote Windows' wallpaper history until the real wallpaper was
# unrecoverable. So we do not trust Windows to remember it. On every clean start
# we (a) record the real wallpaper's path in ORIGINAL_RECORD, and (b) keep a
# byte-for-byte COPY of it in ORIGINAL_BACKUP, which lives in the wallpaper dir
# (a location the shell will actually load from) so it can be re-applied even if
# the user later moves or deletes the real file. A program frame is NEVER
# adopted as the original, so a crash that leaves one of our frames on screen
# cannot poison the record - the next start keeps the real original instead.
ORIGINAL_RECORD: Path = APP_DIR / "original_wallpaper.json"
ORIGINAL_BACKUP: Path = WALLPAPER_DIR / "_original_backup"   # + real extension

# Frame captures written by --capture, used for visual verification.
CAPTURE_DIR: Path = APP_DIR / "captures"

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
ASSET_DIR: Path = REPO_ROOT / "assets"


def ensure_dirs() -> None:
    """Create every directory the app writes into. Safe to call repeatedly."""
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    if host.WALLPAPER_SUPPORTED:
        WALLPAPER_DIR.mkdir(parents=True, exist_ok=True)
