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
"""
from __future__ import annotations

import ctypes
import logging
import os
from ctypes import wintypes
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

log = logging.getLogger(__name__)

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.ShowWindow.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]

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
        self._visible: bool = False
        self._created: bool = False
        self._caption: str = CAPTION
        self._failed: bool = False
        self._fullscreen: bool = False
        self._last_click_ms: int = 0

        # --- camera (preview only) ------------------------------------
        # Zoom lives here rather than in the Renderer on purpose: the desktop
        # wallpaper must always show the whole world, so zooming the render
        # itself would crop what everyone else sees. This crops the finished
        # frame at presentation time instead, which also costs nothing when
        # zoom == 1.
        self.zoom: float = 1.0
        self.cam: list[float] = [self.size[0] / 2.0, self.size[1] / 2.0]
        self._panning: bool = False
        self._pan_from: tuple[int, int] | None = None
        # (x, y, w, h) of the image inside the window, set by present().
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
            self._hwnd = self._get_hwnd()
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

    def _apply_visibility(self, visible: bool) -> None:
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
        """One pointer event carrying both window pixels and world coords.

        The toolbar is hit-tested in window pixels (it is a fixed overlay); the
        world action needs world coords, which are None when the click landed on
        a letterbox bar rather than the scene.
        """
        wx, wy = self.window_to_world(pos)
        return {"type": kind, "button": button,
                "sx": int(pos[0]), "sy": int(pos[1]), "wx": wx, "wy": wy}

    def window_to_world(self, pos) -> tuple:
        """Map a window pixel to a world coordinate, honouring letterbox + zoom
        + pan. Returns (x, y), or (None, None) if the pixel is off the scene."""
        ox, oy, dw, dh = self._image_rect()
        if dw <= 0 or dh <= 0:
            return (None, None)
        fx = (pos[0] - ox) / dw - 0.5
        fy = (pos[1] - oy) / dh - 0.5
        if not (-0.5 <= fx <= 0.5 and -0.5 <= fy <= 0.5):
            return (None, None)
        z = self.zoom if self.zoom else 1.0
        wx = self.cam[0] + fx * (self.size[0] / z)
        wy = self.cam[1] + fy * (self.size[1] / z)
        return (float(wx), float(wy))

    def _image_rect(self) -> tuple[int, int, int, int]:
        """Where the scene actually sits inside the window, letterbox aside."""
        lb = getattr(self, "_letterbox", None)
        if lb:
            return lb
        if self._screen is None:
            return (0, 0, self.size[0], self.size[1])
        sw, sh = self._screen.get_size()
        return (0, 0, sw, sh)

    def _centre_px(self) -> tuple[int, int]:
        ox, oy, dw, dh = self._image_rect()
        return (ox + dw // 2, oy + dh // 2)

    def _clamp_cam(self) -> None:
        """Keep the visible rect inside the world, so you can never scroll off
        into empty space."""
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
                # where the cursor points, in world coords, before the zoom
                fx = ((mouse_px[0] - ox) / max(1, dw)) - 0.5
                fy = ((mouse_px[1] - oy) / max(1, dh)) - 0.5
                fx = min(max(fx, -0.5), 0.5)
                fy = min(max(fy, -0.5), 0.5)
                wx = self.cam[0] + fx * (self.size[0] / old)
                wy = self.cam[1] + fy * (self.size[1] / old)
                # move the camera so that same world point lands under it again
                self.cam[0] = wx - fx * (self.size[0] / self.zoom)
                self.cam[1] = wy - fy * (self.size[1] / self.zoom)
            if self.zoom <= self.ZOOM_MIN + 1e-6:
                self.cam = [self.size[0] / 2.0, self.size[1] / 2.0]
            self._clamp_cam()
            return self.zoom
        except Exception as exc:
            log.debug("preview: zoom failed: %s", exc)
            return self.zoom

    def pan_world(self, dx: float, dy: float) -> bool:
        """Slide the camera by a *world*-space offset. Returns True if it moved.

        Sign convention is the plain one: +dx moves the camera right, so more of
        what lies to the right comes into view. (:meth:`pan_by` is the odd one
        out, because a drag has to move the world *with* the mouse.)

        A no-op at ZOOM_MIN: the whole world is on screen, so there is nowhere
        to pan to and letting the camera wander would only desync it from the
        centre that zoom_at() snaps back to.
        """
        if self.zoom <= self.ZOOM_MIN + 1e-6 or self._screen is None:
            return False
        try:
            before = (self.cam[0], self.cam[1])
            self.cam[0] += dx
            self.cam[1] += dy
            self._clamp_cam()
            return (self.cam[0], self.cam[1]) != before
        except Exception:
            return False

    def pan_by(self, dx_px: int, dy_px: int) -> None:
        """Pan by a *window*-pixel delta with drag semantics.

        Dragging right pushes the world right, which means the camera goes
        left - hence the negation. Window pixels are converted through the
        letterbox rect rather than the raw window size, or a drag drifts by the
        width of the bars.
        """
        if self.zoom <= self.ZOOM_MIN + 1e-6 or self._screen is None:
            return
        try:
            _, _, dw, dh = self._image_rect()
            self.pan_world(-dx_px * (self.size[0] / self.zoom) / max(1, dw),
                           -dy_px * (self.size[1] / self.zoom) / max(1, dh))
        except Exception:
            pass

    # --- held-key panning ------------------------------------------------
    # WASD and the arrow keys, polled rather than driven off KEYDOWN. Key repeat
    # would deliver a nudge, a ~500 ms pause and then a stutter of nudges, which
    # is not what "moves the screen around" means; polling gives one smooth
    # continuous slide for as long as the key is down. It also consumes no
    # events, so nothing else in the app can be starved of a keystroke by it.

    #: Base rate, as a fraction of the world per second before the zoom taper
    #: below. Applied per axis (to size[0] and size[1] respectively) so a
    #: diagonal is isotropic on screen - the letterbox scales both axes equally,
    #: so equal *fractions* per second are equal window pixels per second.
    PAN_VIEW_FRAC = 0.62
    #: World speed is divided by ``zoom ** PAN_ZOOM_EXP``. The two obvious
    #: choices are both wrong at one end: a fixed world speed (exp 0) multiplies
    #: into 8x the on-screen speed at ZOOM_MAX and the picture rockets past,
    #: while a fixed *screen* speed (exp 1) crawls through the world when zoomed
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

    def pan_keys(self, dt: float) -> bool:
        """Poll WASD / arrows once per frame and slide the view. Never raises.

        Returns True if the camera actually moved, which is only of interest to
        tests - the frame loop ignores it.

        Three guards, all deliberate: nothing happens while the window is hidden
        or unfocused (a background window must not eat the user's typing),
        nothing happens at ZOOM_MIN (the whole world is already on screen), and
        the step is dt-scaled and dt-capped so the speed is the same at 15 fps
        and 144 fps.
        """
        if self._failed or not self._created or not self._visible:
            return False
        if self.zoom <= self.ZOOM_MIN + 1e-6:
            return False
        try:
            if not pygame.key.get_focused():
                return False
            keys = pygame.key.get_pressed()
        except Exception:
            return False
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
                return False

            step = max(0.0, min(self.PAN_MAX_DT, float(dt)))
            if step <= 0.0:
                return False
            rate = self.PAN_VIEW_FRAC / (self.zoom ** self.PAN_ZOOM_EXP)
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                rate *= self.PAN_FAST
            if dx and dy:
                rate *= 0.70710678      # a diagonal must not be 41% faster
            return self.pan_world(dx * rate * self.size[0] * step,
                                  dy * rate * self.size[1] * step)
        except Exception as exc:
            log.debug("preview: key pan failed: %s", exc)
            return False

    def reset_view(self) -> None:
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
        """
        if not self._created or self._failed:
            return False
        try:
            self._fullscreen = not getattr(self, "_fullscreen", False)
            if self._fullscreen:
                self._windowed_size = self._screen.get_size()
                self._screen = pygame.display.set_mode(
                    (0, 0), pygame.FULLSCREEN | pygame.SCALED)
            else:
                size = getattr(self, "_windowed_size", None) or self.window_size
                self._screen = pygame.display.set_mode(size, pygame.RESIZABLE)
            self._hwnd = self._get_hwnd()
            return self._fullscreen
        except Exception as exc:
            log.warning("preview: fullscreen toggle failed: %s", exc)
            self._fullscreen = False
            return False

    def handle_resize(self, size: tuple[int, int]) -> None:
        """Adopt a user drag/maximise. Ignored while fullscreen."""
        if not self._created or self._failed or getattr(self, "_fullscreen", False):
            return
        try:
            w, h = max(320, int(size[0])), max(200, int(size[1]))
            self._screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
            self.window_size = (w, h)
            self._hwnd = self._get_hwnd()
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
        the preview is now off. Only QUIT events are consumed, so the app's own
        ``pygame.event.get()`` still sees everything else.
        """
        return self.pump_events().get("closed", False)

    def pump_events(self) -> dict:
        """Handle the window's own events and report what happened.

        Consumes only the events the window owns (close, resize, and the
        fullscreen keys) so the app's own ``pygame.event.get()`` still sees
        everything else.

        Fullscreen is bound to F11 and Alt+Enter, and Escape leaves it - the
        three bindings people actually try. Double-clicking the view works too.

        WASD and the arrows are deliberately *not* handled here: they are polled
        in :meth:`pan_keys` so a held key pans smoothly instead of stuttering on
        key repeat. They fall through the ``else`` below and are re-posted like
        any other key this window does not own.
        """
        out = {"closed": False, "fullscreen": getattr(self, "_fullscreen", False),
               "zoom": self.zoom, "pointer": []}
        if self._failed or not self._created:
            return out
        try:
            wanted = [pygame.QUIT, pygame.VIDEORESIZE, pygame.KEYDOWN,
                      pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                      pygame.MOUSEWHEEL, pygame.MOUSEMOTION]
            for ev in pygame.event.get(wanted):
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
                    elif ev.key in (pygame.K_0, pygame.K_HOME):
                        self.reset_view()
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
                    else:
                        pygame.event.post(ev)     # not ours; hand it back
                elif ev.type == pygame.MOUSEWHEEL:
                    out["zoom"] = self.zoom_at(int(ev.y), pygame.mouse.get_pos())
                elif ev.type == pygame.MOUSEMOTION:
                    # Right/middle drag pans; left never does - left belongs to
                    # the tool palette now.
                    if self._panning and self._pan_from is not None:
                        self.pan_by(ev.pos[0] - self._pan_from[0],
                                    ev.pos[1] - self._pan_from[1])
                        self._pan_from = ev.pos
                    out["pointer"].append(self._pointer("move", 0, ev.pos))
                elif ev.type == pygame.MOUSEBUTTONUP:
                    if ev.button in (2, 3):
                        self._panning = False
                        self._pan_from = None
                    if ev.button == 1:
                        out["pointer"].append(self._pointer("up", 1, ev.pos))
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    now = pygame.time.get_ticks()
                    if ev.button in (2, 3):
                        # Right/middle: pan, and double-right toggles fullscreen.
                        # Default far in the past, not 0: pygame.time.get_ticks()
                        # counts from init, so a 0 default made the very first
                        # right-click of a session - any click inside the first
                        # 400 ms - read as the second half of a double-click and
                        # throw the window into fullscreen instead of starting a
                        # pan drag.
                        last = getattr(self, "_last_rclick_ms", -10_000)
                        self._last_rclick_ms = now
                        if ev.button == 3 and now - last < 400:
                            out["fullscreen"] = self.toggle_fullscreen()
                        else:
                            self._panning = True
                            self._pan_from = ev.pos
                    elif ev.button == 1:
                        out["pointer"].append(self._pointer("down", 1, ev.pos))
        except Exception as exc:
            log.debug("preview: event pump failed: %s", exc)
            return out
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
