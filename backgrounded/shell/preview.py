"""The optional on-screen preview window.

The app's real output is the desktop wallpaper; this window is a live view of
the same render surface, handy while the world is being watched or tuned.

Two rules make this module fiddlier than it looks:

* Hiding the window must **not** tear down the pygame video context. Calling
  ``pygame.display.quit()`` invalidates every Surface the renderer is holding,
  which turns into a crash the next frame. So we keep the SDL window alive and
  toggle it with ``ShowWindow`` on its raw HWND instead.
* Clicking the window's X must hide it, not quit. The app lives in the tray;
  the window is a peripheral.

The second rule is a *Windows* rule, and off Windows it inverts: with no tray
there is nothing to bring a hidden window back from, so App treats the close as
a quit. This module still only reports the close either way - see
``pump_events``'s ``closed`` - it does not decide what it means.
"""
from __future__ import annotations

import ctypes
import logging
import math
import os
from ctypes import wintypes

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from .. import host

log = logging.getLogger(__name__)

# ctypes.WinDLL does not exist off Windows - referencing it is an AttributeError
# at import time, not a graceful failure - so the binding is conditional and
# _apply_visibility routes around it. (ctypes.wintypes, by contrast, imports
# fine everywhere on CPython 3, so the annotations above need no guard.)
if host.IS_WINDOWS:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.ShowWindow.restype = wintypes.BOOL
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
else:
    user32 = None

SW_HIDE = 0
SW_SHOW = 5

CAPTION = "Backgrounded - Live View"


class Preview:
    """Wraps the pygame display window: create once, show/hide, present."""

    def __init__(self, size: tuple[int, int], scale: float = 1.0) -> None:
        self.size: tuple[int, int] = (max(1, int(size[0])), max(1, int(size[1])))
        self.scale: float = scale if scale and scale > 0.05 else 1.0
        self.window_size: tuple[int, int] = self._scaled()

        self._screen: pygame.Surface | None = None
        self._hwnd: int = 0
        #: The SDL window behind the display surface, off Windows only. Held
        #: for the process lifetime rather than fetched per call - a proxy that
        #: goes out of scope is one more chance for a binding to decide it owns
        #: the window it wraps.
        self._sdl_win = None
        self._visible: bool = False
        self._created: bool = False
        self._caption: str = CAPTION
        self._failed: bool = False
        self._fullscreen: bool = False

        # --- view transform (preview only) ----------------------------
        # Zoom lives here rather than in the Renderer on purpose: the desktop
        # wallpaper must always show the whole frame, so zooming the render
        # itself would crop what everyone else sees. This crops the finished
        # frame at presentation time instead, which also costs nothing when
        # zoom == 1.
        #
        # THERE ARE NOW TWO TRANSFORMS AND THEY ARE STRICTLY COMPOSED, NEVER
        # MERGED. Renderer.camera maps world -> frame (a 1600 px window sliding
        # over a 6400 px world; pure translation, no scale). Preview maps frame
        # -> window (this zoom + cam + letterbox). Everything in this file is on
        # the second leg only: `self.size` is RENDER_SIZE, `self.cam` is a point
        # in the 1600x1000 frame, and _clamp_cam clamps to the frame. Preview
        # deliberately does not import WORLD_W and must not learn it - the
        # moment this file knows where the world camera is, the two transforms
        # are entangled and the wallpaper path (which has no Preview at all)
        # stops agreeing with the window.
        #
        # Because the world->frame leg is unit-scale, a delta in frame px *is* a
        # delta in world px. That is what lets pan_world hand its leftovers
        # straight to Camera.nudge() with no conversion.
        self.zoom: float = 1.0
        self.cam: list[float] = [self.size[0] / 2.0, self.size[1] / 2.0]
        # --- drag-pan state -------------------------------------------
        # _panning is a *cache* of the device, never an authority: see
        # _sync_pan_buttons for what happens when a button-up goes missing.
        self._panning: bool = False
        self._pan_from: tuple[int, int] | None = None
        #: Which of {2, 3} we believe are held. A set, not a bool, so releasing
        #: one of two chorded buttons does not kill the drag.
        self._pan_buttons: set[int] = set()
        #: Latched buttons the device has *stopped* reporting without ever
        #: sending a MOUSEBUTTONUP. Not a third latch - a holding pen. See
        #: _apply_held for the SDL behaviour that makes it necessary.
        self._pan_suspended: set[int] = set()
        #: pygame tick at which the current suspension began, so it can expire.
        self._pan_suspend_ms: int = 0
        #: Window px travelled since the current press. What separates a click
        #: from a drag for the double-right-click test.
        self._pan_travel: float = 0.0
        # --- left-button (tool) gesture state --------------------------
        # The same lost-button-up problem as the pan latch above, on the other
        # button and with a different consequence: the Hand tool's grab.
        # See _apply_lmb for why it is NOT simply a copy of the pan machinery.
        #: True between a MOUSEBUTTONDOWN(1) this window saw and its release.
        self._lmb_down: bool = False
        #: Down, but no report has said so lately. A holding pen, like
        #: _pan_suspended - except that its EXPIRY emits a release rather than
        #: merely forgetting the gesture, because a forgotten grab is the bug.
        self._lmb_suspended: bool = False
        #: pygame tick at which the suspension began, so it can expire.
        self._lmb_suspend_ms: int = 0
        #: Last window px the pointer was seen at, so a synthesised release has
        #: somewhere to happen. Set down where it was carried to, not (0, 0).
        self._lmb_pos: tuple[int, int] = (0, 0)
        #: Set when the window geometry changes mid-drag: the next motion
        #: event's delta spans two coordinate spaces and is dropped.
        self._pan_reanchor: bool = False
        #: Right-button double-click detection. -10_000 rather than 0 because
        #: pygame.time.get_ticks() counts from init, so a 0 default made the
        #: very first right-click of a session read as the second half of a
        #: double-click and threw the window into fullscreen.
        self._last_rclick_ms: int = -10_000
        #: Whether the previous right-button interaction was a *click* - press
        #: and release with no drag between them - rather than merely a press.
        self._rclick_was_click: bool = False
        #: Set on the press that fired a fullscreen toggle, cleared by that
        #: press's own button-up. Without it the toggle never consumed the
        #: click it spent, and the window strobed; see _release_pan_button.
        self._rclick_toggled: bool = False
        # (x, y, w, h) of the image inside the window, set by present() and by
        # _refresh_letterbox() the moment the window geometry changes.
        # Pointer maths must use this, not the raw window size, or zoom
        # anchoring drifts by the width of the letterbox bars.
        self._letterbox: tuple[int, int, int, int] | None = None

    # ---------------------------------------------------------------- misc --

    def _scaled(self) -> tuple[int, int]:
        return (max(1, int(self.size[0] * self.scale)),
                max(1, int(self.size[1] * self.scale)))

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def created(self) -> bool:
        return self._created

    @property
    def surface(self) -> "pygame.Surface | None":
        """The display surface, if the window exists. Read-only use please."""
        return self._screen

    # -------------------------------------------------------------- window --

    def ensure_window(self, visible: bool) -> None:
        """Create the window if needed, then match `visible`.

        The display is created on the first call regardless of `visible` (as a
        hidden SDL window when it should not be seen), because the renderer
        needs a live video context for ``Surface.convert()``. Hiding and
        showing afterwards is a pure ``ShowWindow`` toggle - the pygame
        context, and every Surface derived from it, survives untouched.
        """
        if self._failed:
            return
        if not self._created:
            if not self._create(visible):
                return
        if visible != self._visible:
            self._apply_visibility(visible)

    def _create(self, visible: bool) -> bool:
        try:
            if not pygame.display.get_init():
                pygame.display.init()
            hidden_flag = getattr(pygame, "HIDDEN", 0)
            # RESIZABLE is what gives the window a drag border, a working
            # maximise button and a snap target. Without it the window is
            # pinned to its initial size and cannot be filled to the screen.
            flags = pygame.RESIZABLE | (0 if visible else hidden_flag)
            self._screen = pygame.display.set_mode(self.window_size, flags)
            pygame.display.set_caption(self._caption)
            self._created = True
            # With the HIDDEN flag SDL never maps the window, so it never
            # flashes on screen; without it we hide immediately below.
            self._visible = bool(visible) or not hidden_flag
            # Only the Windows branch of _apply_visibility has any use for a
            # raw handle, and asking for one under Wayland just logs a warning
            # about a window id that would never be read.
            self._hwnd = self._get_hwnd() if host.IS_WINDOWS else 0
            # Before this, _letterbox stayed None until the first present(), and
            # _image_rect's fallback is the raw window size - so every pointer
            # and drag conversion in the first frame used a rect wider than the
            # picture by both bars.
            self._refresh_letterbox()
            if not visible:
                self._apply_visibility(False)
            return True
        except Exception as exc:
            self._failed = True
            log.error("preview window could not be created: %s", exc)
            return False

    def _get_hwnd(self) -> int:
        try:
            info = pygame.display.get_wm_info()
            return int(info.get("window") or 0)
        except Exception as exc:
            log.warning("preview: no window handle available: %s", exc)
            return 0

    def _sdl_window(self):
        """pygame-ce's handle on the display-module window, or None.

        Only used off Windows. ``from_display_module`` wraps the window
        ``set_mode`` already made rather than creating a second one, which is
        the whole point: the video context and every Surface derived from it
        stay exactly as they were.
        """
        if self._sdl_win is None:
            cls = getattr(pygame, "Window", None)
            factory = getattr(cls, "from_display_module", None) if cls else None
            if factory is None:
                return None
            try:
                self._sdl_win = factory()
            except Exception as exc:
                log.warning("preview: no SDL window handle available: %s", exc)
                return None
        return self._sdl_win

    def _apply_visibility(self, visible: bool) -> None:
        if not host.IS_WINDOWS:
            # SDL_ShowWindow/SDL_HideWindow, which is what ShowWindow amounts
            # to anyway. Note that a compositor is entitled to ignore a show
            # request (nothing on Wayland guarantees a raise), so self._visible
            # remains the app's *intent*, not an assertion about the screen -
            # the same thing it has always been on the branch below.
            win = self._sdl_window()
            if win is not None:
                try:
                    win.show() if visible else win.hide()
                except Exception as exc:
                    log.warning("preview: SDL show/hide failed: %s", exc)
            self._visible = visible
            return
        if not self._hwnd:
            self._hwnd = self._get_hwnd()
        if not self._hwnd:
            # Without an HWND there is nothing to toggle; record intent so
            # present() still behaves sensibly.
            self._visible = visible
            return
        try:
            user32.ShowWindow(self._hwnd, SW_SHOW if visible else SW_HIDE)
            self._visible = visible
        except Exception as exc:
            log.warning("preview: ShowWindow failed: %s", exc)

    def show(self) -> None:
        self.ensure_window(True)

    def hide(self) -> None:
        if self._created:
            self._apply_visibility(False)
        else:
            self._visible = False

    def toggle(self) -> bool:
        """Flip visibility and return the new state."""
        self.ensure_window(not self._visible)
        return self._visible

    # -------------------------------------------------------------- camera --

    ZOOM_MIN = 1.0
    ZOOM_MAX = 8.0
    ZOOM_STEP = 1.18

    def _pointer(self, kind: str, button: int, pos) -> dict:
        """One pointer event carrying both window pixels and *frame* coords.

        The toolbar is hit-tested in window pixels (it is a fixed overlay); the
        world action needs frame coords, which are None when the click landed on
        a letterbox bar rather than the scene.

        These are ``fx``/``fy``, not ``wx``/``wy``, and the rename is the whole
        point. Frame coords used to *be* world coords, because the world was
        exactly one frame wide. Now the renderer's camera slides a 1600 px frame
        over a 6400 px world, so a frame x is short of the world x by the
        camera's left edge. app.py adds that offset back (it is the only place
        that can - Preview must not learn what the camera is doing, or the two
        transforms stop composing). Leaving these named wx/wy would have let
        shell/tools.py keep reading them and silently strike lightning up to
        4800 px from the click.
        """
        fx, fy = self.window_to_frame(pos)
        return {"type": kind, "button": button,
                "sx": int(pos[0]), "sy": int(pos[1]), "fx": fx, "fy": fy}

    def window_to_frame(self, pos) -> tuple:
        """Map a window pixel to a *frame* coordinate, honouring letterbox +
        zoom + pan. Returns (x, y), or (None, None) if the pixel is off the
        scene.

        Frame space is the 1600x1000 render surface. Add the camera's left edge
        to get world space; see :meth:`_pointer`.
        """
        ox, oy, dw, dh = self._image_rect()
        if dw <= 0 or dh <= 0:
            return (None, None)
        u = (pos[0] - ox) / dw - 0.5
        v = (pos[1] - oy) / dh - 0.5
        if not (-0.5 <= u <= 0.5 and -0.5 <= v <= 0.5):
            return (None, None)
        z = self.zoom if self.zoom else 1.0
        fx = self.cam[0] + u * (self.size[0] / z)
        fy = self.cam[1] + v * (self.size[1] / z)
        return (float(fx), float(fy))

    def _image_rect(self) -> tuple[int, int, int, int]:
        """Where the scene actually sits inside the window, letterbox aside."""
        lb = getattr(self, "_letterbox", None)
        if lb:
            return lb
        if self._screen is None:
            return (0, 0, self.size[0], self.size[1])
        sw, sh = self._screen.get_size()
        return (0, 0, sw, sh)

    def _refresh_letterbox(self) -> None:
        """Recompute the image rect now, instead of waiting for present().

        ``_letterbox`` is normally written by :meth:`present`, which runs at the
        *end* of the frame - long after pump_events. So a window that changed
        size during this pump (VIDEORESIZE, F11, the tray's Window Size, all of
        which call ``set_mode``) went on converting the rest of the pump's drag
        deltas through the *previous* window's scale: measured, a 10-window-px
        drag on a window that had just doubled came back as 10.0 world px when
        the right answer was 5.0.

        Safe to compute early because the rect depends only on the window size
        and the frame's fixed aspect ratio - zoom cancels out, since the zoomed
        subsurface present() blits has the same aspect as the whole frame.
        """
        try:
            if self._screen is None:
                self._letterbox = None
                return
            sw, sh = self._screen.get_size()
            iw, ih = self.size
            k = min(sw / iw, sh / ih)
            dw, dh = max(1, int(iw * k)), max(1, int(ih * k))
            self._letterbox = ((sw - dw) // 2, (sh - dh) // 2, dw, dh)
        except Exception:
            self._letterbox = None

    def _centre_px(self) -> tuple[int, int]:
        ox, oy, dw, dh = self._image_rect()
        return (ox + dw // 2, oy + dh // 2)

    def _clamp_cam(self) -> None:
        """Keep the visible rect inside the *frame*, so you can never scroll off
        into empty space.

        Clamped to ``self.size``, never to WORLD_W: past the frame edge there is
        no picture, because the renderer only ever drew 1600x1000. Panning
        further than this is what the residual and Camera.nudge() are for - the
        world camera moves and the frame is redrawn around the new position."""
        w = self.size[0] / self.zoom
        h = self.size[1] / self.zoom
        self.cam[0] = min(max(self.cam[0], w / 2.0), self.size[0] - w / 2.0)
        self.cam[1] = min(max(self.cam[1], h / 2.0), self.size[1] - h / 2.0)

    def zoom_at(self, wheel: int, mouse_px: tuple[int, int] | None = None) -> float:
        """Zoom under the cursor, so the point beneath the pointer stays put."""
        try:
            old = self.zoom
            factor = self.ZOOM_STEP ** wheel
            self.zoom = min(self.ZOOM_MAX, max(self.ZOOM_MIN, self.zoom * factor))
            if abs(self.zoom - old) < 1e-6:
                return self.zoom
            if mouse_px and self._screen is not None:
                ox, oy, dw, dh = self._image_rect()
                # where the cursor points, in frame coords, before the zoom
                u = ((mouse_px[0] - ox) / max(1, dw)) - 0.5
                v = ((mouse_px[1] - oy) / max(1, dh)) - 0.5
                u = min(max(u, -0.5), 0.5)
                v = min(max(v, -0.5), 0.5)
                fx = self.cam[0] + u * (self.size[0] / old)
                fy = self.cam[1] + v * (self.size[1] / old)
                # move the camera so that same frame point lands under it again
                self.cam[0] = fx - u * (self.size[0] / self.zoom)
                self.cam[1] = fy - v * (self.size[1] / self.zoom)
            if self.zoom <= self.ZOOM_MIN + 1e-6:
                self.cam = [self.size[0] / 2.0, self.size[1] / 2.0]
            self._clamp_cam()
            return self.zoom
        except Exception as exc:
            log.debug("preview: zoom failed: %s", exc)
            return self.zoom

    # --- the residual channel ---------------------------------------------
    # app.py forwards ONE float from this window to Renderer.camera.nudge(),
    # summed across pan_keys() and every drag motion in the pump. That float has
    # to carry two different facts - how far the world camera should move, and
    # whether a pan gesture is in progress at all - so its magnitude is banded,
    # and Camera.NUDGE_EPS / Camera.HOLD_EPS read the same bands from the other
    # end. Duplicated constants across the seam rather than an import: shell/
    # learning what render/ does with the number is how the two transforms get
    # entangled, and the protocol is three lines of arithmetic.

    #: Top of the hold band, and the same 1e-6 as ``Camera.NUDGE_EPS``. A
    #: residual below this never moves the world camera.
    _PAN_HOLD_MAX: float = 1e-6
    #: What a real pan that spilled nothing returns instead of 0.0. Three
    #: orders above the float noise Camera.HOLD_EPS rejects and three below the
    #: band's ceiling, so app.py adding this window's two token sources
    #: (pan_keys plus the pump's) still lands inside it.
    _PAN_HOLD: float = 1e-9
    #: Smallest frame-space request that is a person rather than arithmetic.
    #: The smallest a real one can be is a single window pixel at ZOOM_MAX on a
    #: maximised 4K window - (1600/8)/3456 = 0.058 frame px - so this floor sits
    #: ~58x below the quietest real input, not near it. It is a discriminator,
    #: not a tuning knob: nothing about the feel of panning depends on it.
    _PAN_REAL: float = 1e-3

    def _spill(self, left: float, asked: bool) -> float:
        """Band a raw leftover into the residual protocol above.

        Returns the leftover when it is real motion; the hold token when the
        caller asked for a genuine pan and none of it survived; 0.0 otherwise.
        """
        if not math.isfinite(left):
            return self._PAN_HOLD if asked else 0.0
        if abs(left) >= self._PAN_HOLD_MAX:
            return left
        return self._PAN_HOLD if asked else 0.0

    def pan_world(self, dx: float, dy: float) -> float:
        """Slide the view by a *frame*-space offset. Returns the UNABSORBED dx.

        Sign convention is the plain one: +dx moves the view right, so more of
        what lies to the right comes into view. (:meth:`pan_by` is the odd one
        out, because a drag has to move the picture *with* the mouse.)

        THE RETURN VALUE CHANGED from bool to a residual, and that residual is
        the whole mechanism by which WASD scrolls the 6400 px world. This method
        was already a documented no-op at ZOOM_MIN - the frame is entirely on
        screen, so there is nowhere inside it to pan to. When the world *was*
        the frame that meant "the keys do nothing"; now it means "the keys do
        nothing *here*", and the motion the user asked for has to go somewhere
        else. So instead of swallowing it we hand it back, and app.py spills it
        into Camera.nudge().

        That gives one rule with no modes: at zoom 1 nothing is absorbed and
        100% spills, so WASD scrolls the world. Above zoom 1 the frame pan
        absorbs the step until _clamp_cam pins it at a frame edge, after which
        the surplus spills and the world scrolls on - so you can walk the
        pointer off the right-hand side of the frame and just keep going.

        dy has no residual on purpose: RENDER_H is unchanged and the world is
        not taller than one frame, so there is no vertical world camera to spill
        into. Vertical pan is a zoom-only affordance and dies at the frame edge.

        AN ABSORBED PAN RETURNS THE HOLD TOKEN, NOT 0.0, and that reverses a
        judgement this method used to make. It snapped the leftover to a hard
        zero so that panning inside the frame at zoom 2 could not re-arm the
        camera's six-second manual hold on every frame - the argument being
        that a frame-local pan is no business of the world camera's. True as
        far as it went, and it missed the other half: never arming means
        follow() is free to *move* during a gesture the user has not released.
        The gesture is: pan at zoom 1, zoom in, keep dragging inside the frame
        without letting go. Measured on the old rule, 720 frames of that at
        zoom 1.94 all spilled 0.0, follow() started moving at t=6.00 s and
        carried cam.x 1440 px while the button was still down. (An earlier
        report of the same defect, from a different starting position, put it
        at 6.68 s and 1044 px.) On this rule the same 720 frames move 0.0 px.

        Both halves are wanted, and they are not actually in tension once the
        request and the spill are read as separate questions. "Did the user ask
        for a pan?" is answered by ``asked`` below, from the *request*, before
        any absorption. "Should the world camera move?" is answered by the
        leftover. The old code let the second answer stand in for the first,
        which is why float noise ~1e-14 had to be silenced with the same
        threshold that decides motion. Now noise fails ``asked`` (it is
        thousands of times smaller than _PAN_REAL) and so still returns a hard
        0.0, while a real absorbed pan returns _PAN_HOLD, which moves the world
        camera by nothing and only holds follow() off.
        """
        try:
            fdx, fdy = float(dx), float(dy)
        except (TypeError, ValueError):
            return 0.0
        if not (math.isfinite(fdx) and math.isfinite(fdy)):
            return 0.0
        # Read off the REQUEST, so absorption cannot turn a gesture into
        # silence. dy counts: a right-drag straight up at zoom 4 is a pan the
        # user is making even though nothing horizontal comes of it.
        asked = max(abs(fdx), abs(fdy)) >= self._PAN_REAL
        if self._screen is None or self.zoom <= self.ZOOM_MIN + 1e-6:
            return self._spill(fdx, asked)
        try:
            before = self.cam[0]
            self.cam[0] += fdx
            self.cam[1] += fdy
            self._clamp_cam()
            # Sub-pixel is meaningless in the leftover anyway: cam.x is
            # pixel-quantised at the other end.
            return self._spill(fdx - (self.cam[0] - before), asked)
        except Exception:
            return self._spill(fdx, asked)

    def pan_by(self, dx_px: int, dy_px: int) -> float:
        """Pan by a *window*-pixel delta with drag semantics. Returns the
        UNABSORBED dx in world px, for pump_events to spill into the camera.

        Dragging right pushes the picture right, which means the view goes
        left - hence the negation. Window pixels are converted through the
        letterbox rect rather than the raw window size, or a drag drifts by the
        width of the bars.

        THIS USED TO REFUSE TO SPILL, and the reasoning was backwards. The old
        rule was that a drag must track the pointer exactly, so it stopped dead
        at the frame edge and did nothing whatsoever at zoom 1. But at zoom 1
        the frame absorbs nothing, so "never spill" meant the mouse could not
        move the view *at all* on a 6400 px map while WASD scrolled it freely -
        the one gesture everybody tries first, silently doing nothing.

        And spilling is what makes the picture track the pointer, not what
        breaks it: nudging the world camera by the same world-px delta slides
        the content under the cursor 1:1, which is exactly the gesture asked
        for. A drag now stops only when the world camera runs out of map, which
        is honest - past the end there is nothing further to show.

        Frame px are world px (the frame is always 1600 world px wide; zoom only
        magnifies how it is presented), so the residual needs no conversion on
        its way to Camera.nudge.
        """
        if self._screen is None:
            return 0.0
        try:
            _, _, dw, dh = self._image_rect()
            return self.pan_world(-dx_px * (self.size[0] / self.zoom) / max(1, dw),
                                  -dy_px * (self.size[1] / self.zoom) / max(1, dh))
        except Exception:
            return 0.0

    # --- drag-pan button bookkeeping -------------------------------------
    # This is more machinery than "a bool that says the button is down" and
    # every part of it is paying off a specific measured failure. See the three
    # helpers below.

    #: Window px of travel a press may accumulate and still count as a click
    #: rather than a drag. Real hardware jitters a pixel or two under a
    #: deliberate double-click, and a mouse with any weight to it more.
    _CLICK_SLOP: float = 4.0
    #: Press-to-press window for a double right-click, ms.
    _DOUBLE_MS: int = 400
    #: How long a latched button may go unreported before it is given up for
    #: good. The window this has to cover is an SDL glitch, not a human action:
    #: after a display-mode switch SDL emits a burst of MOUSEMOTION carrying
    #: buttons=(0,0,0) while the button is still physically down - 17
    #: consecutive of them, measured. Motion events arrive at 125-1000 Hz, so 17
    #: of them span 17-136 ms and a second is an order of magnitude of headroom.
    #: The cost of the other end is what keeps it from being longer still: this
    #: is exactly how long a genuinely lost button-up can keep a suspended entry
    #: alive that a button held down in *another* window could then revive.
    _PAN_SUSPEND_MS: int = 1000

    def _held_pan_buttons(self, ev=None) -> "set[int] | None":
        """Which of {2, 3} are reported down. None means "cannot tell".

        TWO SOURCES, UNIONED, and the union is the fix rather than an
        indulgence. A MOUSEMOTION carries its own ``buttons`` tuple - the device
        state at the moment the motion happened, which is the right answer for a
        queue drained a frame late, and which used to be taken as the *only*
        answer whenever it was present. ``pygame.mouse.get_pressed()`` is the
        live device, correct at drain time. Neither is authoritative alone:

        * the event snapshot lies after a display-mode switch (the 17
          zero-button motions in :meth:`_apply_held`);
        * the live poll lies about an event that is a frame old - press, motion,
          release all in one pump means the poll already reads "up" while that
          motion genuinely happened with the button down.

        Each source is only ever wrong in the "up" direction, never the "down"
        direction, so the union is right where both are individually wrong.
        Being wrong in the "down" direction would need SDL to invent a press,
        and if it did the latch intersection in :meth:`_apply_held` would ignore
        it anyway - a button we never saw pressed is not in ``_pan_buttons``.

        None (trust the latch) survives as the fallback's fallback, because this
        must never be the thing that breaks panning on a platform where pygame
        declines to answer at all.
        """
        sources = []
        b = getattr(ev, "buttons", None)
        if b is not None and len(b) >= 3:
            sources.append(b)
        try:
            sources.append(pygame.mouse.get_pressed())
        except Exception:
            pass
        if not sources:
            return None
        out: set[int] = set()
        for b in sources:
            try:
                for n, i in ((2, 1), (3, 2)):
                    if b[i]:
                        out.add(n)
            except Exception:
                continue
        return out

    def _apply_held(self, held: "set[int]") -> None:
        """Reconcile the drag latch against the buttons reported down.

        THE LATCH ALONE WAS A STUCK-PAN BUG. ``_panning`` was set on a press
        and cleared only by a matching MOUSEBUTTONUP, and there are several
        ordinary ways to never get that up-event: release the button outside
        the window, or recreate the display mid-drag via F11, the tray's Window
        Size, or a VIDEORESIZE - all of which call ``set_mode``. After any of
        those, every bare mouse movement panned the world and re-armed the
        camera's six-second manual hold with no button held at all. Reproduced
        by withholding the up-event: a MOUSEMOTION carrying buttons=(0,0,0)
        still returned a residual of -100.0 world px.

        SO THE OLD GUARD WAS ``self._pan_buttons &= held`` AND IT COULD NOT
        RECOVER, which is the defect this method exists to pay off. An
        intersection has no path to re-add, so ONE unreported motion mid-gesture
        emptied the set permanently - and SDL emits those on its own: after
        switching the display in and out of fullscreen, a wrapper on this
        function recorded 17 CONSECUTIVE MOUSEMOTION events carrying
        buttons=(0,0,0) with the button still physically held. Measured end to
        end on the old rule: press -> panning=True; one zero-button motion ->
        panning=False, travel 0.0; five further motions with the button
        genuinely held -> still panning=False; release after 400 px of dragging
        -> recorded as a CLICK, which primed the double-click detector and threw
        the next ordinary right-click into fullscreen.

        The distinction the old rule could not draw is "the device says this
        button is up" versus "this one event did not report it", and it is drawn
        in two places. :meth:`_held_pan_buttons` unions the event snapshot with
        the live device poll, so a report that disagrees with the hardware is
        usually outvoted before it gets here. When both are wrong at once, an
        unreported button is SUSPENDED rather than dropped:

        * it stops panning immediately, because the case the old guard was
          written for - a lost button-up must not leave the world panning on
          bare mouse movement - is real and unchanged;
        * it is restored the moment any report shows it down again, so the rest
          of the gesture pans;
        * it expires after ``_PAN_SUSPEND_MS``, so a button genuinely released
          outside the window cannot be revived minutes later by a press this
          window never saw.

        Restoration is still an intersection with the latch, never an
        assignment: a button pressed somewhere else and dragged into this window
        is not in ``_pan_buttons`` or ``_pan_suspended`` and cannot start a pan
        we never saw begin.
        """
        now = pygame.time.get_ticks()
        back = self._pan_suspended & held
        if back:
            # Reported down again: this was a lying event, not a release.
            self._pan_suspended -= back
            self._pan_buttons |= back
        gone = self._pan_buttons - held
        if gone:
            if not self._pan_suspended:
                self._pan_suspend_ms = now
            self._pan_buttons -= gone
            self._pan_suspended |= gone
        if self._pan_suspended and now - self._pan_suspend_ms >= self._PAN_SUSPEND_MS:
            self._pan_suspended.clear()
        self._panning = bool(self._pan_buttons)
        # The anchor is kept alive while anything is suspended, and the motion
        # branch keeps moving it under the pointer, so a gesture that comes back
        # resumes from where the pointer *is* rather than spending the whole
        # suspended stretch as one teleport on its first restored event.
        if not self._panning and not self._pan_suspended:
            self._pan_from = None

    def _sync_pan_buttons(self, ev=None) -> None:
        """Reconcile the drag latch with the buttons that are really down."""
        held = self._held_pan_buttons(ev)
        if held is None:
            return
        self._apply_held(held)

    def _poll_pan_device(self) -> bool:
        """Reconcile the latch against the DEVICE alone, and say whether a pan
        gesture is live afterwards.

        Called once at the end of every pump that has a gesture in any state,
        so the latch does not depend on a MOUSEMOTION arriving to be corrected.
        That matters in both directions: a button-up that went missing is
        noticed within one pump even if the mouse never moves again, and a
        button the device confirms is still down is restored from the holding
        pen without waiting for the 1-px twitch that used to be required.

        Returns False when the poll cannot answer, NOT the latch's opinion, and
        the direction of that fallback is the whole safety argument - see
        :meth:`pump_events`.
        """
        if not (self._panning or self._pan_suspended):
            return False
        held = self._held_pan_buttons()      # device only - no event snapshot
        if held is None:
            return False
        self._apply_held(held)
        return self._panning

    def _begin_pan(self, ev) -> bool:
        """Handle a middle/right press. True if it toggled fullscreen instead.

        THE DOUBLE-CLICK GUARD USED TO SWALLOW EVERY OTHER DRAG. It compared
        press *timestamps* only, and armed the drag exclusively in its else
        branch, so a second right-drag that began within 400 ms of the previous
        right press never panned - and on real hardware flipped the window to
        fullscreen instead. Repeated short grab-drag-release strokes are how
        people pan, and crossing 4800 px of world takes at least three
        full-window drags, so roughly every other stroke was eaten. Measured
        before the fix: two back-to-back right-drags gave residuals of +200 and
        then +0.

        The fix is to require the first half of a double-click to have actually
        *been* a click - pressed, released, and not dragged in between - which
        is what ``_rclick_was_click`` records on button-up. A drag can no longer
        masquerade as the opening click of a double.

        Only button 3 touches any of that state. Writing the right-button
        timestamp on a middle press too meant middle-then-right inside 400 ms
        toggled fullscreen and never started a drag.

        A FIRED TOGGLE HAS TO SPEND THE CLICK IT USED, and until now it did not:
        clearing ``_rclick_was_click`` here was undone moments later by the
        trailing MOUSEBUTTONUP, which recomputes the flag from ``_pan_travel``
        - and ``_end_pan`` below has just zeroed that, so it always recomputed
        to True. Every right-click therefore left the detector armed, the one
        that had just toggled included, and rapid right-clicking strobed the
        window: measured, ten 0-travel clicks fired on 2,4,5,7,8,10 with two
        real display-mode switches 34 ms apart. ``_rclick_toggled`` is the
        consume flag the up-handler reads instead.
        """
        if ev.button == 3:
            last, was_click = self._last_rclick_ms, self._rclick_was_click
            self._last_rclick_ms = pygame.time.get_ticks()
            if was_click and self._last_rclick_ms - last < self._DOUBLE_MS:
                self._rclick_was_click = False
                self._rclick_toggled = True
                self._end_pan()
                return True
            self._rclick_was_click = False
            # Self-healing: if the toggling press's up-event went missing, this
            # press clears the stale consume flag rather than eating the next
            # release as well.
            self._rclick_toggled = False
        # A real press is authoritative: whatever this button was suspected of
        # before, it is unambiguously down now.
        self._pan_suspended.discard(ev.button)
        self._pan_buttons.add(ev.button)
        self._panning = True
        self._pan_from = ev.pos
        self._pan_reanchor = False
        self._pan_travel = 0.0
        return False

    def _release_pan_button(self, button: int) -> None:
        """Handle a middle/right release.

        Chorded buttons desynced before this: releasing either one cleared the
        whole latch even while the other was still held, so holding both and
        letting go of one killed the drag mid-stroke.

        The ``_rclick_toggled`` branch is where a fired fullscreen toggle
        finally spends its click. Note what it does NOT do: it does not
        suppress the *next* press from starting a drag, only from counting as
        the second half of another double. A right-drag begun 30 ms after a
        toggle pans normally, which is the behaviour a user who overshot into
        fullscreen and immediately grabs the view again expects.
        """
        self._pan_buttons.discard(button)
        # An up-event outranks any suspicion: this button is now known to be up,
        # so it must not sit in the holding pen waiting to be revived.
        self._pan_suspended.discard(button)
        if button == 3:
            if self._rclick_toggled:
                self._rclick_toggled = False
                self._rclick_was_click = False
            else:
                self._rclick_was_click = self._pan_travel <= self._CLICK_SLOP
        if not self._pan_buttons:
            self._panning = False
            if not self._pan_suspended:
                self._pan_from = None

    def _end_pan(self) -> None:
        """Drop any drag in progress, without touching the double-click state."""
        self._pan_buttons.clear()
        self._pan_suspended.clear()
        self._panning = False
        self._pan_from = None
        self._pan_reanchor = False
        self._pan_travel = 0.0

    # --- left-button bookkeeping -----------------------------------------
    # Button 1 had none of the machinery above and needed the same reconcile,
    # for the same reason and by the same three ways of losing an up-event: any
    # set_mode mid-drag - F11, Alt+Enter, the tray's Window Size, a VIDEORESIZE
    # - swallows the MOUSEBUTTONUP. The pan latch survives that; the tool grab
    # did not. ToolController releases only on a pointer event of kind "up", and
    # this loop only ever produced one from a real MOUSEBUTTONUP(1), so a grab
    # that lost its release stayed grabbed: the held entity followed the cursor
    # with no button down, frozen out of its own behaviour tree, until the user
    # clicked again. Self-healing on the next click, which is why it is a
    # medium and not the stuck-pan emergency, but it is the same defect.
    #
    # WHAT IS NOT COPIED FROM THE PAN PATH, AND WHY. The suspension pen is
    # kept, because the case it exists for is exactly this bug's trigger: after
    # a display-mode switch SDL emits a burst of MOUSEMOTION carrying
    # buttons=(0,0,0) with the button still physically down (17 consecutive,
    # measured - see _apply_held). Releasing on the first of those would mean
    # pressing F11 mid-drag ALWAYS drops what you are carrying, which is a more
    # visible failure than the one being fixed and would look like the fix
    # broke dragging across a mode switch.
    #
    # But two things about it are deliberately different:
    #
    # * The pen is 250 ms, not _PAN_SUSPEND_MS's 1000. That constant's own
    #   docstring names its cost - a button held down in ANOTHER window and
    #   moved over ours inside the window revives a dead gesture - and the
    #   stakes are worse here. A revived pan moves the view; a revived grab
    #   picks a colonist back up and carries them off under a button this
    #   window never saw pressed. 250 ms is still ~2x the measured 136 ms
    #   worst case for the SDL burst, and quarters the revival window.
    # * Expiry RELEASES rather than forgetting. For a pan, dropping the latch
    #   ends the gesture completely - there is nothing left holding anything.
    #   A grab is an object in someone's hands: forgetting it is precisely the
    #   stranding. So _apply_lmb reports "up" on expiry and pump_events
    #   synthesises the pointer event that ToolController is waiting for.

    #: How long a left button may go unreported before the gesture is ended.
    _LMB_SUSPEND_MS: int = 250

    def _lmb_held(self, ev=None) -> "bool | None":
        """Is button 1 reported down? None means "cannot tell".

        Same two-source union as :meth:`_held_pan_buttons`, and the same
        argument for it: the event snapshot lies after a display-mode switch,
        the live poll lies about an event that is a frame old, and each is only
        ever wrong in the "up" direction - so the union is right where both are
        individually wrong. None survives as the fallback's fallback so that a
        platform where pygame declines to answer keeps the old behaviour rather
        than losing grabs to a poll that cannot speak.
        """
        sources = []
        b = getattr(ev, "buttons", None)
        if b is not None and len(b) >= 1:
            sources.append(b)
        try:
            sources.append(pygame.mouse.get_pressed())
        except Exception:
            pass
        if not sources:
            return None
        for b in sources:
            try:
                if b[0]:
                    return True
            except Exception:
                continue
        return False

    def _apply_lmb(self, held: bool) -> str:
        """Reconcile the left-button latch. Returns "up" when the gesture must
        be ended now, "" otherwise.

        Restoration is an intersection with what this window saw, never an
        assignment: a button pressed somewhere else and dragged in is neither
        latched nor suspended here, so ``held`` alone can never start or revive
        a gesture we never saw begin.
        """
        now = pygame.time.get_ticks()
        if held:
            if self._lmb_suspended:
                # Reported down again: those were lying events, not a release.
                self._lmb_suspended = False
                self._lmb_down = True
            return ""
        if self._lmb_down:
            # Stop feeding the gesture at once - the reports say the button is
            # up - but do not end it yet; that is what the pen is for.
            self._lmb_down = False
            self._lmb_suspended = True
            self._lmb_suspend_ms = now
            return ""
        if self._lmb_suspended and now - self._lmb_suspend_ms >= self._LMB_SUSPEND_MS:
            self._lmb_suspended = False
            return "up"
        return ""

    def _sync_lmb(self, ev=None) -> str:
        """Reconcile against an event snapshot unioned with the device."""
        held = self._lmb_held(ev)
        if held is None:
            return ""
        return self._apply_lmb(held)

    def _poll_lmb_device(self) -> str:
        """Reconcile against the DEVICE alone, at the end of every pump.

        The pen must not depend on a MOUSEMOTION arriving to expire, or a user
        who loses the button-up and then holds perfectly still keeps the grab
        for ever - which is the original bug with extra steps.
        """
        if not (self._lmb_down or self._lmb_suspended):
            return ""
        held = self._lmb_held()              # device only - no event snapshot
        if held is None:
            return ""
        return self._apply_lmb(held)

    def _begin_lmb(self, pos) -> None:
        """A real press is authoritative, whatever was suspected before."""
        self._lmb_down = True
        self._lmb_suspended = False
        self._lmb_pos = (int(pos[0]), int(pos[1]))

    def _end_lmb(self, pos=None) -> None:
        """A real MOUSEBUTTONUP(1) outranks any suspicion."""
        self._lmb_down = False
        self._lmb_suspended = False
        if pos is not None:
            self._lmb_pos = (int(pos[0]), int(pos[1]))

    # --- held-key panning ------------------------------------------------
    # WASD and the arrow keys, polled rather than driven off KEYDOWN. Key repeat
    # would deliver a nudge, a ~500 ms pause and then a stutter of nudges, which
    # is not what "moves the screen around" means; polling gives one smooth
    # continuous slide for as long as the key is down. It also consumes no
    # events, so nothing else in the app can be starved of a keystroke by it.

    #: Base rate, as a fraction of the frame per second before the zoom taper
    #: below. Applied per axis (to size[0] and size[1] respectively) so a
    #: diagonal is isotropic on screen - the letterbox scales both axes equally,
    #: so equal *fractions* per second are equal window pixels per second.
    PAN_VIEW_FRAC = 0.62
    #: Frame speed is divided by ``zoom ** PAN_ZOOM_EXP``. The two obvious
    #: choices are both wrong at one end: a fixed frame speed (exp 0) multiplies
    #: into 8x the on-screen speed at ZOOM_MAX and the picture rockets past,
    #: while a fixed *screen* speed (exp 1) crawls through the frame when zoomed
    #: in - 11 s to cross the pannable range at 8x. The square root splits them:
    #: measured, 1.15 s to cross at 2x and 4.00 s at 8x, which reads as the same
    #: gesture at both ends.
    PAN_ZOOM_EXP = 0.5
    #: Shift multiplier. Free, since the modifier is already in key.get_mods().
    PAN_FAST = 2.5
    #: Largest dt a single pan step will honour. A stall (or a debugger) must
    #: not teleport the camera across the map on the frame it resumes.
    PAN_MAX_DT = 0.1

    _PAN_KEYS: tuple[tuple[tuple[int, ...], int, int], ...] = ()

    def pan_keys(self, dt: float) -> float:
        """Poll WASD / arrows once per frame and slide the view. Never raises.

        Returns the UNABSORBED horizontal step, in world px, for the caller to
        spill into ``Renderer.camera.nudge()``. See :meth:`pan_world` for why
        that is a residual rather than a bool.

        THE ZOOM_MIN EARLY-RETURN IS GONE, and its removal is the point of this
        change. It used to read "the whole world is already on screen, so there
        is nothing to pan" - true when the world was one frame wide, false the
        moment the world became four frames wide. Keeping it would have left
        WASD dead at zoom 1, which is where the preview spends nearly all of its
        time, and the 4x map unreachable without zooming in first.

        The other two guards stay, both deliberate: nothing happens while the
        window is hidden or unfocused (a background window must not eat the
        user's typing), and the step is dt-scaled and dt-capped so the speed is
        the same at 15 fps and 144 fps.

        Speed at zoom 1, measured from the constants below: 0.62 * 1600 = 992
        world px/s, so 4.8 s to walk the camera across the 4800 px of pannable
        world, or 1.9 s with shift held. That is deliberately far above
        Camera.MAX_SPEED (220 px/s) - follow() eases, a held key should not.
        """
        if self._failed or not self._created or not self._visible:
            return 0.0
        try:
            if not pygame.key.get_focused():
                return 0.0
            keys = pygame.key.get_pressed()
        except Exception:
            return 0.0
        try:
            if not self._PAN_KEYS:
                # Built lazily: the pygame key constants are only guaranteed
                # once display/key is initialised, and this module is imported
                # long before that.
                type(self)._PAN_KEYS = (
                    ((pygame.K_d, pygame.K_RIGHT), 1, 0),
                    ((pygame.K_a, pygame.K_LEFT), -1, 0),
                    ((pygame.K_s, pygame.K_DOWN), 0, 1),
                    ((pygame.K_w, pygame.K_UP), 0, -1),
                )
            dx = dy = 0
            for codes, ax, ay in self._PAN_KEYS:
                if any(keys[c] for c in codes):
                    dx += ax
                    dy += ay
            # Opposite keys held together cancel, which is the right answer and
            # falls out for free.
            if not dx and not dy:
                return 0.0

            step = max(0.0, min(self.PAN_MAX_DT, float(dt)))
            if step <= 0.0:
                return 0.0
            rate = self.PAN_VIEW_FRAC / (self.zoom ** self.PAN_ZOOM_EXP)
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                rate *= self.PAN_FAST
            if dx and dy and self.zoom > self.ZOOM_MIN + 1e-6:
                # A diagonal must not be 41% faster - but only when it is
                # actually a diagonal. At zoom 1 the vertical axis is dead (the
                # world is one frame tall and pan_world has no dy residual), so
                # W+D is horizontal motion wearing a diagonal's costume, and
                # taking the 0.707 there would make holding W scrub 30% off the
                # scroll speed for no movement in return.
                rate *= 0.70710678
            return self.pan_world(dx * rate * self.size[0] * step,
                                  dy * rate * self.size[1] * step)
        except Exception as exc:
            log.debug("preview: key pan failed: %s", exc)
            return 0.0

    def reset_view(self) -> None:
        """Back to 1x, frame centred - and, via pump_events, back to follow.

        Preview cannot call ``Renderer.camera.resume_follow()`` itself without
        learning about the world camera, which is exactly what the composition
        rule forbids. So this resets its own half and pump_events reports
        ``reset_view: True`` for app.py to finish the job. Undoing the zoom
        while leaving the world camera parked where a manual pan left it would
        be the worst of both: the 0/Home key is "put it back", and half of
        putting it back is handing the colony to the camera again.
        """
        self.zoom = 1.0
        self.cam = [self.size[0] / 2.0, self.size[1] / 2.0]

    def _view_rect(self) -> pygame.Rect:
        w = max(16, int(self.size[0] / self.zoom))
        h = max(16, int(self.size[1] / self.zoom))
        return pygame.Rect(int(self.cam[0] - w / 2), int(self.cam[1] - h / 2), w, h)

    # ---------------------------------------------------------- fullscreen --

    def toggle_fullscreen(self) -> bool:
        """Borderless-fullscreen the preview, or return it to a window.

        SCALED keeps the render surface at its native size and lets SDL letterbox
        it, so the aspect ratio survives on a screen that is not 16:10.

        THE SIZE ARGUMENT IS ``self.size``, NOT (0, 0), and it has to be. (0, 0)
        is the "use the desktop resolution" idiom for a plain FULLSCREEN, and it
        is the one size SCALED refuses: pygame-ce 2.5.7 raises "Cannot set 0
        sized SCALED display mode" from set_mode, which this method caught,
        logged at warning level and swallowed - so F11, Alt+Enter, Escape and
        the double-right-click all silently did nothing, every time, on every
        machine. SCALED wants the size of the surface it is scaling *from*,
        which is the render size, and passing it is also what makes the
        docstring above true: SDL letterboxes 1600x1000 onto the monitor and
        reports mouse positions back in render space, so _image_rect and
        window_to_frame keep composing.

        Both halves of the drag state have to be repaired here. The letterbox is
        refreshed at once rather than at the next present(), or the rest of this
        pump converts drag deltas through the old window's scale; and the drag
        anchor is marked stale, because going fullscreen genuinely moves the
        window's client origin, so a pointer that has not moved an inch reports
        a different position on either side of the switch and the first delta
        after it would be a teleport.

        THE FLAG IS WRITTEN AFTER set_mode RETURNS, NOT BEFORE. It used to be
        set first and forced to False in the except path, on the assumption
        that only the *entering* call could fail - so a failure to LEAVE
        fullscreen left the display fullscreen while the flag read False, and
        that flag is what pump_events reports to app.py and what the Escape
        binding tests: Escape would then decline to act on a window that was
        still covering the screen. Assigning only on success means the flag
        always describes the mode that is actually on the glass.
        """
        if not self._created or self._failed:
            return False
        want = not getattr(self, "_fullscreen", False)
        try:
            if want:
                self._windowed_size = self._screen.get_size()
                self._screen = pygame.display.set_mode(
                    self.size, pygame.FULLSCREEN | pygame.SCALED)
            else:
                size = getattr(self, "_windowed_size", None) or self.window_size
                self._screen = pygame.display.set_mode(size, pygame.RESIZABLE)
            self._fullscreen = want
        except Exception as exc:
            log.warning("preview: fullscreen toggle failed: %s", exc)
            # self._fullscreen is deliberately untouched: set_mode raised, so
            # the display is still in whatever mode it was already in.
        self._hwnd = self._get_hwnd()
        self._refresh_letterbox()
        self._pan_reanchor = True
        return self._fullscreen

    def handle_resize(self, size: tuple[int, int]) -> None:
        """Adopt a user drag/maximise. Ignored while fullscreen.

        The letterbox is recomputed immediately - see :meth:`_refresh_letterbox`
        for the drag that came back at twice its true size because it was not.

        THE DRAG ANCHOR USED TO BE KEPT HERE, alone among the three set_mode
        sites, on the argument that a resize leaves the client origin where it
        was so window coordinates stay comparable across it. That is true of a
        bottom/right border drag and false of a MAXIMISE, which moves the origin
        to the top-left of the work area - and a maximise arrives as a
        VIDEORESIZE like any other. Measured: a VIDEORESIZE to 3200x2000 with a
        moved client origin sent the camera -350.0 px where 0.0 was wanted,
        because the first motion afterwards is a pointer that has not moved an
        inch reporting a different number, and the camera spends that teleport
        in full.

        The counter-argument was that dropping a delta stutters the pan through
        the burst of VIDEORESIZE events a live border drag produces. It costs
        nothing in practice: ``_pan_reanchor`` is only ever read while a
        right/middle drag is in progress, and a window border drag is the left
        button on the frame, so there is no pan to stutter. Even in the contrived
        case where both are somehow live, one dropped motion delta is cheaper
        than a teleport of the whole origin shift. ``toggle_fullscreen`` and
        ``set_scale`` have always done this; this method is now consistent with
        them.
        """
        if not self._created or self._failed or getattr(self, "_fullscreen", False):
            return
        try:
            w, h = max(320, int(size[0])), max(200, int(size[1]))
            self._screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
            self.window_size = (w, h)
            self._hwnd = self._get_hwnd()
            self._refresh_letterbox()
            self._pan_reanchor = True
        except Exception as exc:
            log.warning("preview: resize failed: %s", exc)

    def set_scale(self, scale: float) -> None:
        """Resize the window. Cheap no-op if the scale is unchanged."""
        scale = scale if scale and scale > 0.05 else 1.0
        if abs(scale - self.scale) < 1e-6:
            return
        self.scale = scale
        self.window_size = self._scaled()
        if not self._created or self._failed:
            return
        try:
            flags = pygame.RESIZABLE | (
                0 if self._visible else getattr(pygame, "HIDDEN", 0))
            self._screen = pygame.display.set_mode(self.window_size, flags)
            pygame.display.set_caption(self._caption)
            self._hwnd = self._get_hwnd()
            self._apply_visibility(self._visible)
            # Same set_mode hazard as handle_resize/toggle_fullscreen: this one
            # arrives from the tray thread's Window Size command, so it can land
            # between two motion events of a drag the user is in the middle of.
            self._refresh_letterbox()
            self._pan_reanchor = True
        except Exception as exc:
            log.warning("preview: could not resize to %s: %s",
                        self.window_size, exc)

    # ------------------------------------------------------------- drawing --

    def set_caption(self, text: str) -> None:
        """Live HUD in the title bar. Skips the syscall when unchanged."""
        if not text or text == self._caption:
            return
        self._caption = text
        if not self._created or self._failed:
            return
        try:
            pygame.display.set_caption(text)
        except Exception as exc:
            log.debug("preview: set_caption failed: %s", exc)

    def present(self, surface: "pygame.Surface", overlay=None) -> None:
        """Scale the render surface onto the window and flip.

        Does nothing at all when hidden - a hidden preview must cost the frame
        loop nothing beyond this branch. ``overlay`` is an optional callable
        ``fn(window_surface)`` run after the world is scaled in but before the
        flip - the tool palette draws there, in window pixels, so it stays a
        fixed size and never appears on the wallpaper.
        """
        if self._failed or not self._created or not self._visible:
            return
        screen = self._screen
        if screen is None or surface is None:
            return
        try:
            if self.zoom > self.ZOOM_MIN + 1e-6:
                try:
                    view = self._view_rect().clip(surface.get_rect())
                    if view.width > 0 and view.height > 0:
                        surface = surface.subsurface(view)
                except Exception:
                    pass

            # Letterbox rather than stretch. The window is resizable, so its
            # aspect rarely matches the 16:10 render surface, and scaling
            # straight onto it distorts everything - most visibly the stats
            # panel, whose text and bars go out of proportion. (The wallpaper
            # never showed this because the desktop is 16:10 too.)
            sw, sh = screen.get_size()
            iw, ih = surface.get_size()
            k = min(sw / iw, sh / ih)
            dw, dh = max(1, int(iw * k)), max(1, int(ih * k))
            ox, oy = (sw - dw) // 2, (sh - dh) // 2

            if (dw, dh) != (sw, sh):
                screen.fill((0, 0, 0))
            if (dw, dh) == (iw, ih):
                screen.blit(surface, (ox, oy))
            else:
                try:
                    scaled = pygame.transform.smoothscale(surface, (dw, dh))
                except Exception:
                    scaled = pygame.transform.scale(surface, (dw, dh))
                screen.blit(scaled, (ox, oy))
            self._letterbox = (ox, oy, dw, dh)
            if overlay is not None:
                try:
                    overlay(screen)
                except Exception:
                    log.debug("preview overlay failed", exc_info=True)
            pygame.display.flip()
        except Exception as exc:
            log.warning("preview: present failed: %s", exc)

    # -------------------------------------------------------------- events --

    def handle_close(self) -> bool:
        """True if the user just closed the window (X / Alt+F4).

        The window is hidden here as a side effect; the app should record that
        the preview is now off. Note that this runs a whole :meth:`pump_events`,
        so it is not a passive query - App calls pump_events directly and reads
        ``closed`` off the same dict.
        """
        return self.pump_events().get("closed", False)

    def pump_events(self) -> dict:
        """Handle the window's own events and report what happened.

        This is the app's ONLY ``pygame.event.get`` - there is no second reader
        anywhere in the repo - so an event this method does not act on is an
        event nobody will. That is why the KEYDOWN ``else`` below drops rather
        than re-posts.

        THE QUEUE IS TAKEN UNFILTERED, AND THAT IS A BUG FIX, NOT A TIDY-UP.
        This loop used to call ``pygame.event.get(<list of types>)``, and pygame
        does not return a filtered queue in queue order - it returns it GROUPED
        BY TYPE, in the order of the filter list. Demonstrated directly: posting
        [BtnDown/3, Motion, Motion, BtnUp/3, BtnDown/1, BtnUp/1] and asking for
        that filter gave back [BtnDown/3, BtnDown/1, BtnUp/3, BtnUp/1, Motion,
        Motion], while ``pygame.event.get()`` with no filter gave the posted
        order back intact.

        So every MOUSEBUTTONUP in a pump ran before every MOUSEMOTION in it, and
        every MOUSEBUTTONDOWN before every MOUSEBUTTONUP - and that is the
        user's original "cannot move the screen with the mouse" report, still
        reproducing after the pan arithmetic itself was made exact. When one
        stroke's release shared a pump with the next stroke's press, the regroup
        ran the press first (arming the drag and zeroing ``_pan_travel``) and the
        release second (killing the drag and, because travel now read 0,
        recording it as a CLICK). The stroke panned nothing AND primed the
        double-click detector, so the following click threw the window into
        fullscreen. The motions of that stroke were then handled last, against a
        latch the release had already cleared, and spent nothing.

        Taking the queue whole is available precisely because this is the only
        reader: nothing can be starved of an event by it, since there is nobody
        else to starve. It is also strictly less wasteful than the filter was -
        types outside the old ``wanted`` list (KEYUP, the WINDOW* family,
        TEXTINPUT) were never taken by anyone and simply accumulated in SDL's
        queue. Unmatched types now fall out of the dispatch chain and are
        dropped, which is what the old filter was trying to express.

        Fullscreen is bound to F11 and Alt+Enter, and Escape leaves it - the
        three bindings people actually try. Double-clicking the view works too.

        WASD and the arrows are deliberately *not* handled here: they are polled
        in :meth:`pan_keys` so a held key pans smoothly instead of stuttering on
        key repeat. They fall through the ``else`` below, which drops them -
        polling reads the keyboard state, not the queue, so there is nothing for
        a KEYDOWN to do once this window has declined it. See the comment there
        for why re-posting them instead was unbounded queue growth.
        """
        out = {"closed": False, "fullscreen": getattr(self, "_fullscreen", False),
               "zoom": self.zoom, "pointer": [], "reset_view": False,
               # M opens the settings panel, Escape closes it. Both are
               # reported for the app to act on; this window does not know the
               # menu exists beyond these two flags.
               "menu_toggle": False, "menu_close": False,
               # World px a right/middle drag asked for and the frame could not
               # absorb. Accumulated across every MOUSEMOTION in this pump, not
               # overwritten: one frame can carry several motion events and
               # keeping only the last would drop most of a fast flick.
               #
               # A magnitude below _PAN_HOLD_MAX is not motion at all, it is the
               # hold token - "a drag is in progress, the frame took all of it".
               # See pan_world. Tokens are latched rather than summed so exactly
               # one leaves per pump, whatever the motion-event rate.
               "pan_residual": 0.0}
        if self._failed or not self._created:
            return out
        pan_hold = False
        try:
            # Unfiltered, so the events arrive in the order they happened. See
            # the docstring: a type filter regroups them and that regrouping is
            # the reported bug.
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    out["closed"] = True
                elif ev.type == pygame.VIDEORESIZE:
                    self.handle_resize((ev.w, ev.h))
                elif ev.type == pygame.KEYDOWN:
                    alt = bool(ev.mod & pygame.KMOD_ALT)
                    if ev.key == pygame.K_F11 or (alt and ev.key == pygame.K_RETURN):
                        out["fullscreen"] = self.toggle_fullscreen()
                    elif ev.key == pygame.K_ESCAPE and getattr(self, "_fullscreen", False):
                        out["fullscreen"] = self.toggle_fullscreen()
                    elif ev.key == pygame.K_ESCAPE:
                        # Leaving fullscreen has first claim on Escape - it is
                        # the more urgent thing to be able to get out of.
                        out["menu_close"] = True
                    elif ev.key == pygame.K_m:
                        # Reported, not applied, like reset_view and hud_scale
                        # below: the menu is the app's state, and this window
                        # has no business owning it.
                        out["menu_toggle"] = True
                    elif ev.key in (pygame.K_0, pygame.K_HOME):
                        self.reset_view()
                        # Reported, not applied: resuming follow means touching
                        # Renderer.camera, and this window has no business
                        # reaching across the frame->world seam. Same pattern as
                        # hud_scale below.
                        out["reset_view"] = True
                    elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.zoom_at(1, self._centre_px())
                    elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.zoom_at(-1, self._centre_px())
                    elif ev.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                        # HUD size, reported rather than applied: the panels
                        # belong to render/, and the window has no business
                        # reaching into them. The app owns the setting and is
                        # the only thing that can persist it.
                        out["hud_scale"] = (
                            -1 if ev.key == pygame.K_LEFTBRACKET else 1)
                    # else: DROPPED, and not re-posted. It used to be handed
                    # back with pygame.event.post "in case someone else wants
                    # it", but pygame.event.get() appears at exactly one site in
                    # this repo - this loop - and this loop now takes the queue
                    # whole, so it would take a re-posted KEYDOWN back as surely
                    # as the old filter list did.
                    # So a re-posted key came straight back on the next pump and
                    # was posted again, for ever: the queue grew without bound,
                    # every pump paid an O(n) walk over it, and once SDL's queue
                    # saturated it began dropping *new* events - including the
                    # MOUSEBUTTONUP that ends a pan drag, which is the stuck-pan
                    # bug _sync_pan_buttons exists to survive.
                    #
                    # Nothing is lost. W/A/S/D and the arrows take this branch by
                    # design and are read by pan_keys through
                    # pygame.key.get_pressed(), which SDL keeps current from the
                    # pump itself and which consumes no events.
                elif ev.type == pygame.MOUSEWHEEL:
                    out["zoom"] = self.zoom_at(int(ev.y), pygame.mouse.get_pos())
                elif ev.type == pygame.MOUSEMOTION:
                    # Right/middle drag pans; left never does - left belongs to
                    # the tool palette now. The sync calls are what stop a bare
                    # mouse movement panning the world, or dragging a colonist
                    # around, after a lost button-up.
                    self._sync_pan_buttons(ev)
                    self._lmb_pos = (int(ev.pos[0]), int(ev.pos[1]))
                    if self._sync_lmb(ev) == "up":
                        out["pointer"].append(self._pointer("up", 1, ev.pos))
                    if self._pan_from is not None and (
                            self._panning or self._pan_suspended):
                        dx = ev.pos[0] - self._pan_from[0]
                        dy = ev.pos[1] - self._pan_from[1]
                        self._pan_from = ev.pos
                        # Travel accrues even while the gesture is suspended,
                        # and that is deliberate. Travel answers only "was this
                        # a click or a drag?", and a pointer that crossed 400 px
                        # between press and release dragged, whatever SDL was
                        # claiming about the button in the middle of it. This is
                        # what stops a suspended stretch turning the release
                        # into a click and priming the fullscreen toggle.
                        self._pan_travel += abs(dx) + abs(dy)
                        if self._pan_reanchor:
                            # The window changed shape since the last motion, so
                            # this delta straddles two coordinate spaces. Re-take
                            # the anchor and spend nothing.
                            self._pan_reanchor = False
                        elif self._panning:
                            spill = self.pan_by(dx, dy)
                            if abs(spill) >= self._PAN_HOLD_MAX:
                                out["pan_residual"] += spill
                            elif spill:
                                pan_hold = True
                        # else: the gesture is SUSPENDED - the reports say the
                        # button is up, so nothing here may move the world. The
                        # anchor was still moved above, so when the button is
                        # reported down again the drag resumes from where the
                        # pointer actually is, rather than spending the whole
                        # suspended stretch as one teleport on its first event.
                    if not self._lmb_suspended:
                        out["pointer"].append(self._pointer("move", 0, ev.pos))
                    # else: the left gesture is SUSPENDED - the reports say the
                    # button is up, so nothing here may move a held entity. If
                    # the button really is still down the pen restores it within
                    # a few motions and the entity resumes from under the
                    # cursor; if it really is up, it is set down where it froze
                    # rather than 400 px further on. A move dropped here costs
                    # nothing else: ToolController's "move" branch drives
                    # interact.move_held and nothing besides, and the toolbar
                    # hover reads pygame.mouse.get_pos() directly.
                elif ev.type == pygame.MOUSEBUTTONUP:
                    if ev.button in (2, 3):
                        self._release_pan_button(ev.button)
                    if ev.button == 1:
                        self._end_lmb(ev.pos)
                        out["pointer"].append(self._pointer("up", 1, ev.pos))
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    if ev.button in (2, 3):
                        # Right/middle: pan, and a genuine double-right toggles
                        # fullscreen. All of the fiddly part is in _begin_pan.
                        if self._begin_pan(ev):
                            out["fullscreen"] = self.toggle_fullscreen()
                    elif ev.button == 1:
                        self._begin_lmb(ev.pos)
                        out["pointer"].append(self._pointer("down", 1, ev.pos))
        except Exception as exc:
            log.debug("preview: event pump failed: %s", exc)
            return out
        # --- the standing hold token -------------------------------------
        # THE HOLD TOKEN USED TO BE MOTION-DRIVEN ONLY, emitted from pan_by on
        # the way through a MOUSEMOTION, so a button held perfectly STILL
        # emitted nothing at all and Camera.MANUAL_HOLD simply ran out under it.
        # Measured with a follow target 2400 px off: dragged to cam.x=3201, held
        # motionless, follow() took over at t=6.017 s and by t=30 s had carried
        # cam.x to 957 - 2244 px of view moved out from under a button that was
        # still down, identical at zoom 4. One 1-px twitch re-armed it.
        #
        # The previous lane declined to emit a standing token from the latch,
        # and the reason was sound at the time: that means trusting ``_panning``
        # between pumps, and a lost button-up would then suppress follow()
        # for ever, with no motion event left to clear it. What changed is that
        # the latch is no longer the only thing that can answer. The claim below
        # is made by the DEVICE, polled fresh this pump, and it is a claim about
        # right now rather than about what a press said some seconds ago.
        #
        # The three cases, stated explicitly:
        #
        # * held still - the poll says the button is down, the token goes out
        #   every pump, MANUAL_HOLD is refreshed every pump, and follow() cannot
        #   move the view until the button is released. No 6 s cliff.
        # * lost button-up - the poll says the button is up, so no token is
        #   emitted and the latch is suspended instead (_apply_held), all within
        #   the first pump after the release, ~16 ms. follow() resumes when the
        #   last genuine MANUAL_HOLD expires, at most 6 s later. The old failure
        #   mode - suppressed for ever - needs the poll to lie about a released
        #   button permanently, which is a different thing from the transient
        #   zero-button motions SDL really does emit.
        # * poll unavailable - _poll_pan_device answers False and leaves the
        #   latch alone, so no token goes out. That is the deliberate direction
        #   of the fallback: on a platform that will not answer, this degrades
        #   exactly to the old motion-driven behaviour (follow takes a still-held
        #   button after 6 s), never to a hold nothing can clear.
        #
        # The same poll is what restores a SUSPENDED gesture without waiting for
        # a motion event, which is the other half of item 2: a burst of lying
        # zero-button motions followed by the user simply holding still used to
        # need a 1-px twitch before the drag came back.
        # --- and the same reconcile for button 1 -------------------------
        # Runs every pump, whether or not the mouse moved, so a lost button-up
        # is noticed and the pen expires on the clock rather than on the next
        # motion event. The synthesised release is appended AFTER the loop on
        # purpose: it must be the last pointer event of the pump, or a "move"
        # queued behind it would arrive at a ToolController that has already
        # dropped the grab and mean nothing.
        if self._poll_lmb_device() == "up":
            out["pointer"].append(self._pointer("up", 1, self._lmb_pos))
        if self._poll_pan_device():
            pan_hold = True
        if pan_hold and abs(out["pan_residual"]) < self._PAN_HOLD_MAX:
            # Real motion already implies the hold at the far end, so the token
            # is only worth sending when nothing else made it out.
            out["pan_residual"] = self._PAN_HOLD
        if out["closed"]:
            self._apply_visibility(False)
        return out

    # ------------------------------------------------------------ shutdown --

    def close(self) -> None:
        """Tear the window down. Shutdown only - this invalidates surfaces."""
        if not self._created:
            return
        self._created = False
        self._visible = False
        self._screen = None
        self._hwnd = 0
        # Dropped before display.quit(), not after: the proxy refers to a
        # window that is about to stop existing.
        self._sdl_win = None
        try:
            pygame.display.quit()
        except Exception as exc:
            log.debug("preview: display.quit failed: %s", exc)


if __name__ == "__main__":                                    # pragma: no cover
    import math
    import time

    logging.basicConfig(level=logging.INFO)
    pygame.init()
    preview = Preview((1280, 800), scale=0.5)
    preview.ensure_window(True)
    scene = pygame.Surface((1280, 800))
    clock = pygame.time.Clock()
    t0 = time.time()
    while time.time() - t0 < 6.0:
        t = time.time() - t0
        scene.fill((10, 14, 30))
        x = 640 + int(400 * math.sin(t))
        pygame.draw.circle(scene, (255, 190, 90), (x, 400), 40)
        preview.set_caption(f"Backgrounded - Live View  t={t:4.1f}s")
        preview.present(scene)
        if preview.handle_close():
            print("closed -> hidden; re-showing in 1s")
            time.sleep(1.0)
            preview.show()
        clock.tick(60)
    preview.close()
    pygame.quit()
