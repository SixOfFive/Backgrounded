"""Win32 notification-area (tray) icon, driven by ctypes only.

No pywin32, no pystray - this module talks straight to user32/shell32/gdi32.

Design notes that are load-bearing (verified by spike, do not "simplify"):

* The tray icon owns a **dedicated thread** with its own message pump. A Win32
  message pump is thread-affine, and so is ``PostQuitMessage`` - calling it
  from another thread does nothing. To shut the tray down from the outside we
  ``PostMessageW(hwnd, WM_CLOSE, 0, 0)`` and call ``PostQuitMessage`` inside
  the window procedure, on the pump's own thread.
* The ``WNDPROC`` ctypes callback object must be kept alive by a Python
  reference for as long as the window exists. If it is garbage collected,
  Windows calls into freed memory and the process dies.
* ``TrackPopupMenu`` needs ``SetForegroundWindow`` first and a ``WM_NULL``
  nudge afterwards, otherwise the menu stays on screen after you click away.
* Explorer restarts drop every tray icon on the floor. The
  ``"TaskbarCreated"`` broadcast message is our cue to re-add ours.

Everything here fails soft: this program is expected to run unattended for
hours, so a Win32 call that returns 0 gets logged, never raised.
"""
from __future__ import annotations

import ctypes
import logging
import math
import os
import queue
import threading
import uuid
from ctypes import wintypes
from typing import Any, Callable

from .. import host
from ..constants import (
    RENDER_SIZE, SCENES, SCENE_LABELS, SCENE_ROTATE_SEC, SPEEDS, WINDOW_SCALES,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------- win32 ABI --
#
# Everything from here to the end of the file is Windows, and off Windows this
# module is imported only so that ``from .shell.tray import Tray`` keeps
# working - :meth:`Tray.start` then returns without spawning a thread and the
# rest is dead code. Two names have to be bound conditionally to get that far,
# because they are the ones that do not merely *fail* on another platform but
# do not exist at all: ``ctypes.WinDLL`` and ``ctypes.WINFUNCTYPE``. Everything
# else here - including the whole of ``ctypes.wintypes``, which is pure
# type aliases - imports fine anywhere CPython runs.

LRESULT = ctypes.c_ssize_t
HCURSOR = wintypes.HANDLE
LPVOID = ctypes.c_void_p

if host.IS_WINDOWS:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    WNDPROC = ctypes.WINFUNCTYPE(
        LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
    )
else:
    user32 = shell32 = gdi32 = kernel32 = None
    WNDPROC = ctypes.CFUNCTYPE(
        LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
    )

# messages
WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_QUIT = 0x0012
WM_NULL = 0x0000
WM_COMMAND = 0x0111
WM_LBUTTONDBLCLK = 0x0203
WM_RBUTTONUP = 0x0205
WM_LBUTTONUP = 0x0202
WM_CONTEXTMENU = 0x007B
WM_APP = 0x8000
WM_TRAYICON = WM_APP + 17
# Posted by the app when the tooltip's contents have gone stale. It is a posted
# message rather than a direct call because Shell_NotifyIconW is talking about
# *this thread's* window: keeping every touch of the icon on the pump thread is
# the same discipline that makes PostQuitMessage work above.
WM_TRAY_REFRESH = WM_APP + 18

# window styles
WS_OVERLAPPED = 0x00000000
CW_USEDEFAULT = -2147483648  # 0x80000000 as a signed int

# menus
MF_STRING = 0x0000
MF_UNCHECKED = 0x0000
MF_GRAYED = 0x0001
MF_CHECKED = 0x0008
MF_POPUP = 0x0010
MF_SEPARATOR = 0x0800
MF_BYCOMMAND = 0x0000

TPM_LEFTALIGN = 0x0000
TPM_RIGHTBUTTON = 0x0002
TPM_RETURNCMD = 0x0100
TPM_NONOTIFY = 0x0080

# shell notify icon
NIM_ADD = 0x00000000
NIM_MODIFY = 0x00000001
NIM_DELETE = 0x00000002
NIF_MESSAGE = 0x00000001
NIF_ICON = 0x00000002
NIF_TIP = 0x00000004

# command ids -------------------------------------------------------------
ID_SHOW_WINDOW = 1001
ID_WALLPAPER = 1002
ID_PAUSE = 1003
ID_RESET = 1004
ID_NEW_TERRAIN = 1044
ID_CLEAR_GRAVES = 1045
ID_NAMES = 1046
ID_LOG = 1047
ID_AUTO_SCENE = 1048
ID_STATS = 1049
ID_ACTIVITY = 1050
ID_SAVE = 1005
ID_EXIT = 1006
ID_SCENE_BASE = 2000
ID_SPEED_BASE = 3000
ID_SCALE_BASE = 4000

# SPEEDS and WINDOW_SCALES moved to constants.py when the in-window menu
# arrived - both menus dispatch by INDEX into them, so they have to be one
# list rather than two that drift.

ICON_SIZE = 32
TOOLTIP = "Backgrounded"


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", HCURSOR),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HICON),
    ]


class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("hWnd", wintypes.HWND),
        ("uID", wintypes.UINT),
        ("uFlags", wintypes.UINT),
        ("uCallbackMessage", wintypes.UINT),
        ("hIcon", wintypes.HICON),
        ("szTip", wintypes.WCHAR * 128),
        ("dwState", wintypes.DWORD),
        ("dwStateMask", wintypes.DWORD),
        ("szInfo", wintypes.WCHAR * 256),
        ("uVersion", wintypes.UINT),
        ("szInfoTitle", wintypes.WCHAR * 64),
        ("dwInfoFlags", wintypes.DWORD),
        ("guidItem", ctypes.c_byte * 16),
        ("hBalloonIcon", wintypes.HICON),
    ]


class ICONINFO(ctypes.Structure):
    _fields_ = [
        ("fIcon", wintypes.BOOL),
        ("xHotspot", wintypes.DWORD),
        ("yHotspot", wintypes.DWORD),
        ("hbmMask", wintypes.HBITMAP),
        ("hbmColor", wintypes.HBITMAP),
    ]


def _bind() -> None:
    """Pin argtypes/restypes. Without this, 64-bit handles get truncated to
    32 bits by ctypes' default int marshalling and the API calls silently
    fail (or, worse, corrupt something)."""
    user32.DefWindowProcW.restype = LRESULT
    user32.DefWindowProcW.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.RegisterClassExW.restype = wintypes.ATOM
    user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
    user32.UnregisterClassW.restype = wintypes.BOOL
    user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, LPVOID]
    user32.DestroyWindow.restype = wintypes.BOOL
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.GetMessageW.restype = ctypes.c_int
    user32.GetMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.restype = LRESULT
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.PostMessageW.restype = wintypes.BOOL
    user32.PostMessageW.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostQuitMessage.argtypes = [ctypes.c_int]
    user32.RegisterWindowMessageW.restype = wintypes.UINT
    user32.RegisterWindowMessageW.argtypes = [wintypes.LPCWSTR]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.GetCursorPos.restype = wintypes.BOOL
    user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]

    user32.CreatePopupMenu.restype = wintypes.HMENU
    user32.CreatePopupMenu.argtypes = []
    user32.DestroyMenu.restype = wintypes.BOOL
    user32.DestroyMenu.argtypes = [wintypes.HMENU]
    user32.AppendMenuW.restype = wintypes.BOOL
    user32.AppendMenuW.argtypes = [
        wintypes.HMENU, wintypes.UINT, wintypes.WPARAM, wintypes.LPCWSTR]
    user32.CheckMenuRadioItem.restype = wintypes.BOOL
    user32.CheckMenuRadioItem.argtypes = [
        wintypes.HMENU, wintypes.UINT, wintypes.UINT, wintypes.UINT, wintypes.UINT]
    user32.TrackPopupMenu.restype = wintypes.BOOL
    user32.TrackPopupMenu.argtypes = [
        wintypes.HMENU, wintypes.UINT, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, wintypes.HWND, LPVOID]

    user32.CreateIconIndirect.restype = wintypes.HICON
    user32.CreateIconIndirect.argtypes = [ctypes.POINTER(ICONINFO)]
    user32.DestroyIcon.restype = wintypes.BOOL
    user32.DestroyIcon.argtypes = [wintypes.HICON]

    gdi32.CreateBitmap.restype = wintypes.HBITMAP
    gdi32.CreateBitmap.argtypes = [
        ctypes.c_int, ctypes.c_int, wintypes.UINT, wintypes.UINT, LPVOID]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteObject.argtypes = [wintypes.HANDLE]

    shell32.Shell_NotifyIconW.restype = wintypes.BOOL
    shell32.Shell_NotifyIconW.argtypes = [
        wintypes.DWORD, ctypes.POINTER(NOTIFYICONDATAW)]

    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]


# There are no DLLs to describe when the four handles above are None, and
# nothing will ever call the functions this would have annotated.
if host.IS_WINDOWS:
    _bind()


# ------------------------------------------------------------ icon drawing --

def _seg_dist(px: float, py: float,
              ax: float, ay: float, bx: float, by: float) -> float:
    """Distance from point p to the line segment a-b."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    t = 0.0 if denom <= 1e-9 else (wx * vx + wy * vy) / denom
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    dx, dy = wx - vx * t, wy - vy * t
    return math.sqrt(dx * dx + dy * dy)


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _rounded_box_sd(px: float, py: float, half: float, radius: float) -> float:
    """Signed distance to an axis-aligned rounded square centred on (0,0)."""
    dx = abs(px) - (half - radius)
    dy = abs(py) - (half - radius)
    ox, oy = max(dx, 0.0), max(dy, 0.0)
    return math.sqrt(ox * ox + oy * oy) + min(max(dx, dy), 0.0) - radius


def _icon_argb(size: int = ICON_SIZE) -> list[int]:
    """Paint the tray icon procedurally and return a top-down 0xAARRGGBB list.

    A dark night-blue rounded square, a white stick figure, and the candle
    flame it carries. Antialiased by coverage, clipped to the rounded square.
    """
    n = size * size
    # rgba accumulator, straight (non-premultiplied) alpha, 0..255 / 0..1
    buf: list[list[float]] = [[0.0, 0.0, 0.0, 0.0] for _ in range(n)]
    mask: list[float] = [0.0] * n

    c = size * 0.5
    half = size * 0.5 - 0.5
    radius = size * 0.24

    bg = (18.0, 25.0, 48.0)
    border = (46.0, 62.0, 104.0)
    bone = (236.0, 241.0, 250.0)
    wax = (244.0, 236.0, 206.0)
    flame_core = (255.0, 246.0, 196.0)
    flame_warm = (255.0, 172.0, 62.0)

    def over(idx: int, rgb: tuple[float, float, float], a: float) -> None:
        if a <= 0.0:
            return
        sa = _clamp01(a)
        d = buf[idx]
        da = d[3]
        oa = sa + da * (1.0 - sa)
        if oa <= 1e-6:
            return
        inv = da * (1.0 - sa)
        d[0] = (rgb[0] * sa + d[0] * inv) / oa
        d[1] = (rgb[1] * sa + d[1] * inv) / oa
        d[2] = (rgb[2] * sa + d[2] * inv) / oa
        d[3] = oa

    # stick figure skeleton, in icon pixel space
    head = (11.4, 9.4, 2.9)
    spine = (11.4, 12.2, 11.4, 19.8)
    arm_l = (11.4, 14.6, 6.4, 18.2)
    arm_r = (11.4, 14.6, 18.4, 11.4)
    leg_l = (11.4, 19.8, 7.0, 26.2)
    leg_r = (11.4, 19.8, 16.2, 26.2)
    candle = (20.0, 9.0, 20.0, 11.8)
    flame = (20.0, 6.9)

    limbs = (spine, arm_l, arm_r, leg_l, leg_r)

    for y in range(size):
        fy = y + 0.5
        for x in range(size):
            fx = x + 0.5
            i = y * size + x

            sd = _rounded_box_sd(fx - c, fy - c, half, radius)
            cov = _clamp01(0.5 - sd)
            mask[i] = cov
            if cov <= 0.0:
                continue

            over(i, bg, cov)
            # thin lighter rim just inside the edge
            rim = _clamp01(1.0 - abs(sd + 0.9)) * 0.75
            if rim > 0.0:
                over(i, border, rim)

            # warm bloom from the flame washing over the background
            gd = math.hypot(fx - flame[0], fy - flame[1])
            glow = _clamp01(1.0 - gd / 7.0)
            if glow > 0.0:
                over(i, flame_warm, glow * glow * 0.45)

            # candle stem
            a = _clamp01(1.2 - _seg_dist(fx, fy, *candle))
            if a > 0.0:
                over(i, wax, a)

            # stick figure
            best = 9e9
            for seg in limbs:
                d = _seg_dist(fx, fy, *seg)
                if d < best:
                    best = d
            a = _clamp01(1.25 - best)
            hd = math.hypot(fx - head[0], fy - head[1])
            a = max(a, _clamp01(head[2] + 0.5 - hd))
            if a > 0.0:
                over(i, bone, a)

            # flame body: warm halo then a bright core
            a = _clamp01(2.6 - gd) * 0.9
            if a > 0.0:
                over(i, flame_warm, a)
            a = _clamp01(1.5 - gd)
            if a > 0.0:
                over(i, flame_core, a)

    out: list[int] = [0] * n
    for i in range(n):
        r, g, b, a = buf[i]
        a *= mask[i]
        if a <= 0.0:
            continue
        ai = int(_clamp01(a) * 255.0 + 0.5)
        ri = int(_clamp01(r / 255.0) * 255.0 + 0.5)
        gi = int(_clamp01(g / 255.0) * 255.0 + 0.5)
        bi = int(_clamp01(b / 255.0) * 255.0 + 0.5)
        out[i] = (ai << 24) | (ri << 16) | (gi << 8) | bi
    return out


def _create_icon(size: int = ICON_SIZE) -> int:
    """Build an HICON from a procedurally painted ARGB buffer. Returns 0 on
    failure - the caller then falls back to a stock icon."""
    try:
        pixels = _icon_argb(size)
        colour_buf = (ctypes.c_uint32 * len(pixels))(*pixels)
        # Monochrome AND-mask; all zero = "use the colour bitmap's alpha".
        mask_bytes = (size * size) // 8
        mask_buf = (ctypes.c_ubyte * mask_bytes)()

        hbm_colour = gdi32.CreateBitmap(size, size, 1, 32, colour_buf)
        hbm_mask = gdi32.CreateBitmap(size, size, 1, 1, mask_buf)
        if not hbm_colour or not hbm_mask:
            if hbm_colour:
                gdi32.DeleteObject(hbm_colour)
            if hbm_mask:
                gdi32.DeleteObject(hbm_mask)
            return 0

        info = ICONINFO()
        info.fIcon = 1
        info.xHotspot = 0
        info.yHotspot = 0
        info.hbmMask = hbm_mask
        info.hbmColor = hbm_colour
        hicon = user32.CreateIconIndirect(ctypes.byref(info))

        # CreateIconIndirect copies the bitmaps; ours are ours to free.
        gdi32.DeleteObject(hbm_colour)
        gdi32.DeleteObject(hbm_mask)
        return int(hicon) if hicon else 0
    except Exception as exc:                                  # pragma: no cover
        log.warning("tray icon could not be built: %s", exc)
        return 0


_DEFAULT_STATE: dict[str, Any] = {
    "show_window": True,
    "paused": False,
    "scene": SCENES[0],
    "sim_speed": 1.0,
    "wallpaper_enabled": True,
}


class Tray:
    """A Win32 tray icon running its own thread and message pump.

    Menu selections are pushed onto ``cmd_q`` as ``(verb, payload)`` tuples;
    the app drains that queue on its own loop. ``get_state`` is polled every
    time the menu opens so the check marks reflect live state rather than
    whatever the tray last heard about.
    """

    def __init__(self, cmd_q: "queue.Queue[tuple[str, Any]]",
                 get_state: Callable[[], dict]) -> None:
        self.cmd_q = cmd_q
        self.get_state = get_state

        self.hwnd: int | None = None
        self._hicon: int = 0
        self._hinst: int = 0
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopping = False
        self._icon_added = False
        self._destroyed = False
        # Keep a hard reference for the window's lifetime: if this ctypes
        # callback is collected, Windows jumps into freed memory.
        self._wndproc_ref: Any = None
        self._class_name = f"BackgroundedTray_{os.getpid():x}_{uuid.uuid4().hex[:8]}"
        self._class_registered = False
        self._wm_taskbar_created = 0

    # ----------------------------------------------------------- lifecycle --

    def start(self) -> None:
        """Spawn the tray thread and wait briefly for the window to exist.

        A no-op where there is no notification area to put an icon in. Silent
        rather than a warning: on those platforms the app is its window and the
        absence of a tray is the design, not a degraded mode. Every other
        method is already safe with ``self._thread`` left at None.
        """
        if not host.TRAY_SUPPORTED:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping = False
        self._destroyed = False
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._run, name="tray", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            log.warning("tray thread did not report ready within 5s")

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the pump thread to quit and join it.

        ``PostQuitMessage`` only works on the pump's own thread, so we post
        ``WM_CLOSE`` and let the window procedure do the quitting.
        """
        self._stopping = True
        hwnd = self.hwnd
        if hwnd:
            try:
                user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            except Exception as exc:
                log.warning("tray: could not post WM_CLOSE: %s", exc)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                log.warning("tray thread did not exit within %.1fs", timeout)
        self._thread = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -------------------------------------------------------------- thread --

    def _run(self) -> None:
        try:
            self._create_window()
        except Exception as exc:
            log.exception("tray window creation failed: %s", exc)
            self._ready.set()
            return
        self._ready.set()
        try:
            self._pump()
        except Exception as exc:                              # pragma: no cover
            log.exception("tray message pump died: %s", exc)
        finally:
            self._teardown()

    def _create_window(self) -> None:
        self._hinst = int(kernel32.GetModuleHandleW(None) or 0)
        self._wm_taskbar_created = int(
            user32.RegisterWindowMessageW("TaskbarCreated") or 0)

        self._wndproc_ref = WNDPROC(self._wndproc)

        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.style = 0
        wc.lpfnWndProc = self._wndproc_ref
        wc.cbClsExtra = 0
        wc.cbWndExtra = 0
        wc.hInstance = self._hinst
        wc.hIcon = None
        wc.hCursor = None
        wc.hbrBackground = None
        wc.lpszMenuName = None
        wc.lpszClassName = self._class_name
        wc.hIconSm = None
        atom = user32.RegisterClassExW(ctypes.byref(wc))
        if not atom:
            raise ctypes.WinError(ctypes.get_last_error())
        self._class_registered = True

        # A real (if never-shown) top-level window. A message-only window
        # cannot be brought to the foreground, and without foreground the
        # popup menu refuses to dismiss.
        hwnd = user32.CreateWindowExW(
            0, self._class_name, TOOLTIP, WS_OVERLAPPED,
            0, 0, 0, 0, None, None, self._hinst, None)
        if not hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        self.hwnd = int(hwnd)

        self._hicon = _create_icon()
        self._add_icon()

    def _pump(self) -> None:
        msg = wintypes.MSG()
        while True:
            got = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if got == 0:        # WM_QUIT
                break
            if got == -1:       # error; nothing sane left to do
                log.warning("tray GetMessageW failed: %s",
                            ctypes.WinError(ctypes.get_last_error()))
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def _teardown(self) -> None:
        self._remove_icon()
        if self._hicon:
            try:
                user32.DestroyIcon(self._hicon)
            except Exception:
                pass
            self._hicon = 0
        hwnd = self.hwnd
        self.hwnd = None
        # Only if WM_DESTROY never ran (pump died early). Destroying a stale
        # handle risks hitting whatever window inherited the value.
        if hwnd and not self._destroyed:
            try:
                user32.DestroyWindow(hwnd)
            except Exception:
                pass
        if self._class_registered:
            try:
                user32.UnregisterClassW(self._class_name, self._hinst)
            except Exception:
                pass
            self._class_registered = False
        # Only safe to drop once the window is gone.
        self._wndproc_ref = None

    # ---------------------------------------------------------- shell icon --

    def _tip(self) -> str:
        """Tooltip text, which carries the paused state.

        A paused colony is indistinguishable from a running one that happens to
        be quiet - the wallpaper simply stops changing - and ``paused`` persists
        across restarts, so the freeze can outlive any memory of having asked
        for it. The window caption says so too, but the whole point of this app
        is that the window is usually hidden, which leaves the tray as the only
        thing left to ask.
        """
        try:
            if bool(self._state().get("paused")):
                return f"{TOOLTIP} - paused"
        except Exception:
            log.debug("tray: could not read paused state for the tooltip",
                      exc_info=True)
        return TOOLTIP

    def _nid(self) -> NOTIFYICONDATAW:
        nid = NOTIFYICONDATAW()
        nid.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
        nid.hWnd = self.hwnd
        nid.uID = 1
        nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
        nid.uCallbackMessage = WM_TRAYICON
        nid.hIcon = self._hicon or None
        nid.szTip = self._tip()
        return nid

    def refresh(self) -> None:
        """Ask the pump thread to re-read state and repaint the icon's tooltip.

        Safe to call from any thread, and a no-op before the window exists (the
        first ``_add_icon`` reads the state itself, so a run that starts paused
        is already labelled).
        """
        hwnd = self.hwnd
        if not hwnd or self._stopping:
            return
        try:
            user32.PostMessageW(hwnd, WM_TRAY_REFRESH, 0, 0)
        except Exception as exc:
            log.debug("tray: could not post a tooltip refresh: %s", exc)

    def _update_icon(self) -> None:
        """Re-send the icon data (tray thread only). Adds it if it went missing."""
        if not self.hwnd:
            return
        if not self._icon_added:
            self._add_icon()
            return
        try:
            nid = self._nid()
            if not shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(nid)):
                log.debug("tray: NIM_MODIFY failed")
        except Exception as exc:
            log.debug("tray: could not update the icon: %s", exc)

    def _add_icon(self) -> None:
        if not self.hwnd:
            return
        nid = self._nid()
        ok = shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(nid))
        if not ok:
            # Explorer may not be listening yet (fresh boot). Try a modify -
            # that covers the "already added" case too.
            ok = shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(nid))
        self._icon_added = bool(ok)
        if not ok:
            log.warning("tray icon could not be added to the notification area")

    def _remove_icon(self) -> None:
        if not self._icon_added or not self.hwnd:
            self._icon_added = False
            return
        try:
            nid = self._nid()
            shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(nid))
        except Exception:
            pass
        self._icon_added = False

    # ------------------------------------------------------------ wndproc --

    def _wndproc(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        try:
            if msg == WM_TRAYICON:
                self._on_tray_message(int(lparam) & 0xFFFF)
                return 0
            if msg == WM_TRAY_REFRESH:
                self._update_icon()
                return 0
            if self._wm_taskbar_created and msg == self._wm_taskbar_created:
                # Explorer restarted; our icon went with it.
                self._icon_added = False
                self._add_icon()
                return 0
            if msg == WM_CLOSE:
                self._remove_icon()
                user32.DestroyWindow(hwnd)
                return 0
            if msg == WM_DESTROY:
                self._destroyed = True
                # Thread-affine: this is the only place it works.
                user32.PostQuitMessage(0)
                return 0
        except Exception as exc:                              # pragma: no cover
            log.exception("tray wndproc error on msg 0x%04X: %s", msg, exc)
        return int(user32.DefWindowProcW(hwnd, msg, wparam, lparam))

    def _on_tray_message(self, mouse_msg: int) -> None:
        if mouse_msg in (WM_RBUTTONUP, WM_CONTEXTMENU):
            self._show_menu()
        elif mouse_msg == WM_LBUTTONDBLCLK:
            self._emit("toggle_window", None)

    # --------------------------------------------------------------- menu --

    def _state(self) -> dict[str, Any]:
        state = dict(_DEFAULT_STATE)
        try:
            live = self.get_state()
        except Exception as exc:
            log.warning("tray: get_state() failed (%s); using defaults", exc)
            return state
        if isinstance(live, dict):
            state.update(live)
        return state

    def _emit(self, verb: str, payload: Any) -> None:
        try:
            self.cmd_q.put_nowait((verb, payload))
        except Exception as exc:
            log.warning("tray: could not queue command %r: %s", verb, exc)

    def _build_menu(self, state: dict[str, Any]) -> tuple[int, list[int]]:
        """Build the popup fresh so check marks track live state.

        Returns (root menu, submenus) - submenus are owned by the root once
        appended with MF_POPUP, but we keep the handles for a defensive
        cleanup if building blows up halfway through.
        """
        subs: list[int] = []
        menu = int(user32.CreatePopupMenu() or 0)
        if not menu:
            return 0, subs

        def flag(checked: bool) -> int:
            return MF_STRING | (MF_CHECKED if checked else MF_UNCHECKED)

        user32.AppendMenuW(menu, flag(bool(state["show_window"])),
                           ID_SHOW_WINDOW, "Show Window")
        user32.AppendMenuW(menu, flag(bool(state["wallpaper_enabled"])),
                           ID_WALLPAPER, "Wallpaper Output")
        user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)

        # --- Scene submenu, radio-checked on the active scene
        scene_menu = int(user32.CreatePopupMenu() or 0)
        if scene_menu:
            subs.append(scene_menu)
            for i, key in enumerate(SCENES):
                label = SCENE_LABELS.get(key, key)
                user32.AppendMenuW(scene_menu, MF_STRING,
                                   ID_SCENE_BASE + i, label)
            try:
                active = SCENES.index(str(state["scene"]))
            except ValueError:
                active = -1
            if active >= 0:
                user32.CheckMenuRadioItem(
                    scene_menu, ID_SCENE_BASE,
                    ID_SCENE_BASE + len(SCENES) - 1,
                    ID_SCENE_BASE + active, MF_BYCOMMAND)
            # The rotation toggle lives inside the Scene submenu because that is
            # what it is about, and because picking a scene by hand and then
            # having the weather move on by itself in ten minutes is exactly the
            # surprise this switch exists to prevent. The interval is in the
            # label so the menu answers "how often?" without needing a manual.
            mins = max(1, int(round(SCENE_ROTATE_SEC / 60.0)))
            user32.AppendMenuW(scene_menu, MF_SEPARATOR, 0, None)
            user32.AppendMenuW(
                scene_menu, flag(bool(state.get("auto_scene_change", True))),
                ID_AUTO_SCENE, f"Change scene every {mins} min")
            user32.AppendMenuW(menu, MF_STRING | MF_POPUP, scene_menu, "Scene")

        # --- Speed submenu, radio-checked on the active multiplier
        speed_menu = int(user32.CreatePopupMenu() or 0)
        if speed_menu:
            subs.append(speed_menu)
            for i, (label, _value) in enumerate(SPEEDS):
                user32.AppendMenuW(speed_menu, MF_STRING,
                                   ID_SPEED_BASE + i, label)
            try:
                current = float(state["sim_speed"])
            except (TypeError, ValueError):
                current = 1.0
            active = -1
            for i, (_label, value) in enumerate(SPEEDS):
                if abs(value - current) < 1e-6:
                    active = i
                    break
            if active >= 0:
                user32.CheckMenuRadioItem(
                    speed_menu, ID_SPEED_BASE,
                    ID_SPEED_BASE + len(SPEEDS) - 1,
                    ID_SPEED_BASE + active, MF_BYCOMMAND)
            user32.AppendMenuW(menu, MF_STRING | MF_POPUP, speed_menu, "Speed")

        # --- Window size submenu, radio-checked on the active multiplier
        scale_menu = int(user32.CreatePopupMenu() or 0)
        if scale_menu:
            subs.append(scale_menu)
            for i, value in enumerate(WINDOW_SCALES):
                w = max(1, int(RENDER_SIZE[0] * value))
                h = max(1, int(RENDER_SIZE[1] * value))
                user32.AppendMenuW(scale_menu, MF_STRING, ID_SCALE_BASE + i,
                                   f"{value * 100:.0f}%\t{w} x {h}")
            try:
                current = float(state.get("window_scale", 1.0))
            except (TypeError, ValueError):
                current = 1.0
            active = -1
            for i, value in enumerate(WINDOW_SCALES):
                if abs(value - current) < 1e-6:
                    active = i
                    break
            if active >= 0:
                user32.CheckMenuRadioItem(
                    scale_menu, ID_SCALE_BASE,
                    ID_SCALE_BASE + len(WINDOW_SCALES) - 1,
                    ID_SCALE_BASE + active, MF_BYCOMMAND)
            user32.AppendMenuW(menu, MF_STRING | MF_POPUP, scale_menu,
                               "Window Size")

        user32.AppendMenuW(menu, flag(bool(state.get("show_stats", True))),
                           ID_STATS, "Show Stats")
        user32.AppendMenuW(menu, flag(bool(state.get("show_names", True))),
                           ID_NAMES, "Show Names")
        user32.AppendMenuW(menu, flag(bool(state.get("show_activity", True))),
                           ID_ACTIVITY, "Show Activity")
        user32.AppendMenuW(menu, flag(bool(state.get("show_log", True))),
                           ID_LOG, "Show Log")
        user32.AppendMenuW(menu, flag(bool(state["paused"])), ID_PAUSE, "Pause")
        user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)
        user32.AppendMenuW(menu, MF_STRING, ID_NEW_TERRAIN,
                           "New Landscape (keep the colony)")
        user32.AppendMenuW(menu, MF_STRING, ID_CLEAR_GRAVES, "Clear Graves")
        user32.AppendMenuW(menu, MF_STRING, ID_RESET,
                           "Start Over (new world and colony)")
        user32.AppendMenuW(menu, MF_STRING, ID_SAVE, "Save Now")
        user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)
        user32.AppendMenuW(menu, MF_STRING, ID_EXIT, "Exit")
        return menu, subs

    def _show_menu(self) -> None:
        hwnd = self.hwnd
        if not hwnd:
            return
        menu = 0
        try:
            menu, _subs = self._build_menu(self._state())
            if not menu:
                return
            pt = wintypes.POINT()
            if not user32.GetCursorPos(ctypes.byref(pt)):
                pt.x, pt.y = 0, 0
            # Required, or the menu will not dismiss when you click elsewhere.
            user32.SetForegroundWindow(hwnd)
            cmd = int(user32.TrackPopupMenu(
                menu, TPM_RETURNCMD | TPM_RIGHTBUTTON | TPM_LEFTALIGN
                | TPM_NONOTIFY,
                pt.x, pt.y, 0, hwnd, None))
            # The other half of the classic fix; flushes the menu's state.
            user32.PostMessageW(hwnd, WM_NULL, 0, 0)
            if cmd:
                self._dispatch(cmd)
        except Exception as exc:
            log.exception("tray menu failed: %s", exc)
        finally:
            if menu:
                try:
                    # Destroying the root destroys the attached submenus too.
                    user32.DestroyMenu(menu)
                except Exception:
                    pass

    def _dispatch(self, cmd: int) -> None:
        if cmd == ID_SHOW_WINDOW:
            self._emit("toggle_window", None)
        elif cmd == ID_WALLPAPER:
            self._emit("toggle_wallpaper", None)
        elif cmd == ID_PAUSE:
            self._emit("toggle_pause", None)
        elif cmd == ID_RESET:
            self._emit("reset", None)
        elif cmd == ID_NEW_TERRAIN:
            self._emit("new_terrain", None)
        elif cmd == ID_CLEAR_GRAVES:
            self._emit("clear_graves", None)
        elif cmd == ID_STATS:
            self._emit("toggle_stats", None)
        elif cmd == ID_NAMES:
            self._emit("toggle_names", None)
        elif cmd == ID_ACTIVITY:
            self._emit("toggle_activity", None)
        elif cmd == ID_LOG:
            self._emit("toggle_log", None)
        elif cmd == ID_AUTO_SCENE:
            self._emit("toggle_auto_scene", None)
        elif cmd == ID_SAVE:
            self._emit("save", None)
        elif cmd == ID_EXIT:
            self._emit("quit", None)
        elif ID_SCENE_BASE <= cmd < ID_SCENE_BASE + len(SCENES):
            self._emit("scene", SCENES[cmd - ID_SCENE_BASE])
        elif ID_SPEED_BASE <= cmd < ID_SPEED_BASE + len(SPEEDS):
            self._emit("speed", SPEEDS[cmd - ID_SPEED_BASE][1])
        elif ID_SCALE_BASE <= cmd < ID_SCALE_BASE + len(WINDOW_SCALES):
            self._emit("window_scale", WINDOW_SCALES[cmd - ID_SCALE_BASE])
        else:
            log.debug("tray: unknown menu command %d", cmd)


if __name__ == "__main__":                                    # pragma: no cover
    import time

    logging.basicConfig(level=logging.DEBUG)
    q: "queue.Queue[tuple[str, Any]]" = queue.Queue()
    state = {
        "show_window": True, "paused": False, "scene": SCENES[0],
        "sim_speed": 1.0, "wallpaper_enabled": True,
    }
    tray = Tray(q, lambda: state)
    tray.start()
    print("tray running - right-click the icon; choose Exit to stop")
    try:
        while True:
            try:
                verb, payload = q.get(timeout=0.25)
            except queue.Empty:
                continue
            print("command:", verb, payload)
            if verb == "quit":
                break
            if verb == "toggle_window":
                state["show_window"] = not state["show_window"]
            elif verb == "toggle_wallpaper":
                state["wallpaper_enabled"] = not state["wallpaper_enabled"]
            elif verb == "toggle_pause":
                state["paused"] = not state["paused"]
            elif verb == "scene":
                state["scene"] = payload
            elif verb == "speed":
                state["sim_speed"] = payload
    except KeyboardInterrupt:
        pass
    finally:
        tray.stop()
        time.sleep(0.1)
