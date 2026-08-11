"""Which pieces of the desktop shell this platform actually has.

Backgrounded was written for Windows, where the program's real output is the
desktop wallpaper and its only control surface is a notification-area icon.
Neither has a portable equivalent, so everywhere else the app is simply its
window: the preview stops being a peripheral view of the program and becomes
the program.

Flags rather than scattered ``sys.platform`` tests, because the interesting
question is never "is this Windows" - it is "is there a tray to hide in", and
those two only happen to coincide today. Read once at import; none of them can
change during a run.
"""
from __future__ import annotations

import sys

IS_WINDOWS: bool = sys.platform == "win32"
IS_MAC: bool = sys.platform == "darwin"
IS_LINUX: bool = sys.platform.startswith("linux")

#: Can rendered frames be pushed onto the desktop background?
#:
#: shell/wallpaper.py is IDesktopWallpaper and SystemParametersInfoW all the
#: way down. There is no Linux counterpart to port it to - setting a wallpaper
#: is a different command per desktop (GNOME's gsettings, KDE's
#: plasma-apply-wallpaperimage, feh or hsetroot under a bare WM, and nothing at
#: all under most Wayland compositors), and at 4 Hz none of them is a sane
#: thing to shell out to anyway. So off, and the window carries the picture.
WALLPAPER_SUPPORTED: bool = IS_WINDOWS

#: Is there a tray icon to live in? shell/tray.py is raw user32/shell32/gdi32.
TRAY_SUPPORTED: bool = IS_WINDOWS

#: True when the window IS the whole program.
#:
#: This is the flag with teeth. On Windows, closing the window hides it and the
#: tray icon brings it back; with no tray, that same code path would leave a
#: process running with no way to see it and no way to quit it. So where this
#: is true, closing the window ends the run - see :meth:`App._frame`.
WINDOW_IS_THE_APP: bool = not (WALLPAPER_SUPPORTED or TRAY_SUPPORTED)
