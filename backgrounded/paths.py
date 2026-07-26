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

# Two alternating wallpaper targets. Windows caches the wallpaper by path, so
# writing the *same* path repeatedly can be ignored or can collide with
# Explorer/Defender holding the active file open. Alternating sidesteps both.
#
# JPEG, not BMP. A 2560x1600 24-bit BMP is ~12 MB, and Windows silently
# declined to transcode ours: SPI_GETDESKWALLPAPER and the registry both
# reported the file while %APPDATA%\Microsoft\Windows\Themes\TranscodedWallpaper
# stayed 80 minutes stale, so the desktop fell back to the background colour
# (black). JPEG is what the shell itself uses - Windows Spotlight writes .jpg -
# and at ~400 KB it also writes roughly 30x faster.
WALLPAPER_A: Path = APP_DIR / "wallpaper_a.jpg"
WALLPAPER_B: Path = APP_DIR / "wallpaper_b.jpg"

# Frame captures written by --capture, used for visual verification.
CAPTURE_DIR: Path = APP_DIR / "captures"

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
ASSET_DIR: Path = REPO_ROOT / "assets"


def ensure_dirs() -> None:
    """Create every directory the app writes into. Safe to call repeatedly."""
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
