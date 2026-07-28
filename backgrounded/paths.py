"""Filesystem locations. Everything user-writable lives under %LOCALAPPDATA%."""
from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "Backgrounded"


def _local_appdata() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/AppData/Local")
    return Path(base)


APP_DIR: Path = _local_appdata() / APP_NAME
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
WALLPAPER_DIR: Path = Path(os.path.expanduser("~")) / "Pictures" / APP_NAME
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
    WALLPAPER_DIR.mkdir(parents=True, exist_ok=True)
