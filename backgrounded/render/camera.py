"""The world -> frame transform, and the rule that decides where it points.

Until now the world and the frame were the same 1600 px object, so there was no
transform to write: a sim x *was* a screen x. ``WORLD_W`` is 6400 now and the
frame is still ``RENDER_W`` = 1600, so exactly one conversion stands between
them and every draw path has to go through it:

    sx = wx - cam.x

``cam.x`` is the world x of the LEFT edge of the frame. It is a public attribute
rather than a property, and it is a whole number of pixels, for two measured
reasons:

* the particle system subtracts it from a 3000-element array in one vectorised
  op - a ``cam.sx()`` call per particle would be a Python-level loop on the
  frame path;
* the cached terrain chunks blit at ``-cam.x``, and a fractional offset makes
  pygame resample the whole background every frame. On the 4 Hz wallpaper path,
  where a frame sits on the desktop for 250 ms, a 1 px background crawl is the
  most visible artefact this renderer can produce.

Quantised is not the same as *truncating*, and the difference is a bug this
file shipped: ``nudge`` used to round the new position and throw the leftover
away, so a mouse drag that asked for less than half a world pixel per event
moved the camera by nothing at all - for ever. On a 3840x2160 fullscreen window
one window pixel is 0.463 world px, so dragging the mouse across the whole
screen moved the world exactly 0 px. ``_frac`` is the fix: the presented ``x``
stays a whole number, and the sub-pixel remainder is carried to the next call
instead of being discarded.

There is no y camera. ``RENDER_H`` is unchanged and the world did not get
taller, so there is no vertical clamp here to get wrong.

LAYERING: this lives in ``render/`` and the Renderer owns the only real
instance. It is NOT a field on World and the sim never reads it - ``sim/``
answers "can the player see this?" with ``actions.stage_bounds`` /
``offstage_x``, which derive the same answer from ``colony_center()`` without
either half learning about the other.
"""
from __future__ import annotations

import math
from typing import Any

from ..constants import RENDER_W, WORLD_W

__all__ = ["Camera", "IDENTITY"]

#: The furthest left edge that still shows only real land. Recomputed nowhere
#: else: every clamp in this module goes through it.
MAX_LEFT: float = float(max(0, WORLD_W - RENDER_W))


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class Camera:
    """Where the frame sits on the world, and how it gets there.

    ``follow`` is called once per frame from ``Renderer.draw`` and never raises:
    a camera that throws costs the whole frame, and this thing runs unattended
    for hours behind a wallpaper.
    """

    __slots__ = ("x", "_manual_t", "_frac")

    #: How far the target has to move before the camera does. This is the
    #: single most important number in the file. The follow target is a *mean
    #: over wandering agents*, so it is never still; without a deadzone the
    #: camera would chase it at a pixel or two a frame forever and the entire
    #: background would shimmer. 160 px is a tenth of the frame - wide enough
    #: to swallow the wander of a settled colony, narrow enough that a colonist
    #: walking out to a quarry pulls the view along before they leave it.
    DEADZONE: float = 160.0
    #: Exponential approach rate, per second. 2.4/s closes ~91% of a gap in one
    #: second, which reads as the camera noticing rather than as a hard cut.
    EASE: float = 2.4
    #: ...capped, so the ease cannot outrun the eye on a big jump. Above
    #: WALK_SPEED (34) and above the fastest dragon (DRAGON_STATS wyvern, 96),
    #: so the camera never lags the thing it is following.
    MAX_SPEED: float = 220.0
    #: Beyond this, teleport instead of easing. Sized at 2*RENDER_W because the
    #: case it exists for is the whole roster being relocated at once -
    #: world load, and ``randomise_terrain`` - where easing 5000 px at
    #: MAX_SPEED would crawl for 23 seconds with nothing on screen.
    SNAP_DIST: float = 3200.0
    #: How long a manual pan suppresses ``follow``. Long enough to look around,
    #: short enough that the user never has to ask for the colony back.
    MANUAL_HOLD: float = 6.0
    #: Smallest *requested* pan that moves the camera at all. Deliberately the
    #: same 1e-6 that ``Preview.pan_world`` uses as the top of its hold band.
    #: Anything at or above it is motion - with ``_frac`` carrying the
    #: remainder, a request of 0.46 px is motion, just motion that takes three
    #: events to become visible.
    #:
    #: IT THRESHOLDS THE REQUEST AND NOT THE RESULTING PIXELS, ON PURPOSE, and
    #: that is worth writing down because it looks like an oversight. Any
    #: request in (NUDGE_EPS, 0.5) arms MANUAL_HOLD while moving cam.x by
    #: nothing on that frame. Three reasons it stays that way:
    #:
    #: * with ``_frac`` those requests are not zero motion, they are motion not
    #:   yet shown - a 4K window delivers a drag in 0.463 px steps and
    #:   thresholding on visible pixels is precisely the bug that made the
    #:   mouse unable to move the world at all;
    #: * arming without moving is *required* at the end of the map, where the
    #:   clamp eats the whole request and follow() must still be held off (see
    #:   :meth:`nudge`);
    #: * it is required mid-gesture too - that is what HOLD_EPS below is for -
    #:   so "armed but stationary" is a state this camera is designed to be in,
    #:   not an accident of the threshold.
    NUDGE_EPS: float = 1e-6
    #: Floor of the HOLD BAND: |dx| in [HOLD_EPS, NUDGE_EPS) means "the user is
    #: panning right now and none of it reached me". ``Preview.pan_world``
    #: returns exactly that when a zoomed frame absorbs a whole drag step, and
    #: it is the only way this camera can be told about a gesture that is in
    #: progress but currently spilling nothing - app.py forwards one float and
    #: nothing else. Below HOLD_EPS is float noise and means nothing at all;
    #: pan_world's own `dx - (cam - before)` is exact in theory and ~1e-14 in
    #: practice, so the floor sits two orders above the worst of that and three
    #: below the 1e-9 token pan_world actually sends.
    HOLD_EPS: float = 1e-12

    def __init__(self, x: float = 0.0) -> None:
        self.x: float = float(round(_clamp(float(x), 0.0, MAX_LEFT)))
        self._manual_t: float = 0.0
        #: Sub-pixel pan the user has asked for and not yet been shown. Always
        #: in (-0.5, 0.5); see the module docstring.
        self._frac: float = 0.0

    # ------------------------------------------------------ the conversion --
    def sx(self, wx: float) -> float:
        """World x -> screen x. The one conversion; everything else is this."""
        return float(wx) - self.x

    def wx(self, sx: float) -> float:
        """Screen x -> world x. The inverse, for pointer hit-testing."""
        return float(sx) + self.x

    def visible(self, wx: float, pad: float = 0.0) -> bool:
        """Is world x *wx* inside the frame (with *pad* px of slack)?

        The slack is what callers use to keep an object whose sprite is wider
        than a point from popping in at the edge - pass half the sprite width.
        """
        s = float(wx) - self.x
        return -pad <= s <= RENDER_W + pad

    # ------------------------------------------------------------- follow --
    def follow(self, world: Any, dt: float) -> None:
        """Point the camera at the biggest cluster of people. Never raises.

        Deliberately NOT ``actions.colony_center()``, which is the obvious
        choice and is wrong twice over:

        * it prefers ``structures.colony_center()``, a mean over every
          non-grave structure - and barricades are sited at +/-OFFSTAGE by
          design, so a single unpaired barricade drags the view ~200 px off the
          settlement it is supposed to be framing;
        * a plain mean puts the camera in the *gap* when a colony is split
          across a chasm, showing neither half.

        So the rule here is a cluster rule: slide a frame-wide window over the
        sorted agent xs and take the window holding the most people. The
        tie-break is measured against where the camera already is, which buys
        hysteresis for nothing - a 12/8 split keeps the 12 and never
        oscillates, and a 10/10 split holds whichever side it was already on.
        Half the colony off-camera beats the camera parked in the chasm.

        ``colony_center()`` is still the fallback for an empty roster, so the
        two agree about where "nowhere in particular" is.
        """
        try:
            step = float(dt)
            if not math.isfinite(step) or step < 0.0:
                step = 0.0
            step = min(step, 0.25)

            if self._manual_t > 0.0:
                # Manual pan wins outright - not blended with the follow rule,
                # which would fight the user's own WASD for six seconds.
                self._manual_t = max(0.0, self._manual_t - step)
                return

            # follow() is in charge from here down, so any sub-pixel the user's
            # last drag had pending is stale - it belongs to a gesture that
            # ended six seconds ago. Dropping it here is what stops it being
            # added to the *next* unrelated drag.
            self._frac = 0.0

            xs = self._agent_xs(world)
            if xs:
                centre = self._cluster_centre(xs)
            else:
                centre = self._fallback_centre(world)

            target = _clamp(centre - RENDER_W * 0.5, 0.0, MAX_LEFT)
            gap = target - self.x
            if abs(gap) < self.DEADZONE:
                return                      # the whole anti-shimmer story
            if abs(gap) >= self.SNAP_DIST:
                self.x = float(round(target))
                return
            limit = self.MAX_SPEED * step
            move = _clamp(gap * self.EASE * step, -limit, limit)
            # Rounded before anything reads it, not on the way out: a
            # fractional cam.x resamples the terrain cache every frame.
            self.x = float(round(_clamp(self.x + move, 0.0, MAX_LEFT)))
        except Exception:
            return

    @staticmethod
    def _agent_xs(world: Any) -> list[float]:
        """Sorted x of everyone alive. Empty on any malformed world."""
        try:
            pop = getattr(world, "population", None)
            if pop is None:
                return []
            fn = getattr(pop, "alive_agents", None)
            agents = fn() if callable(fn) else getattr(pop, "agents", ())
            xs = []
            for a in agents:
                try:
                    if not getattr(a, "alive", True):
                        continue
                    v = float(a.x)
                except (AttributeError, TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    xs.append(v)
            xs.sort()
            return xs
        except Exception:
            return []

    @staticmethod
    def _fallback_centre(world: Any) -> float:
        try:
            from ..sim.actions import colony_center
            c = float(colony_center(world))
            return c if math.isfinite(c) else WORLD_W * 0.5
        except Exception:
            return WORLD_W * 0.5

    def _cluster_centre(self, xs: list[float]) -> float:
        """Mean of the densest frame-wide window of *xs* (already sorted).

        Two pointers, so O(n) rather than the O(n^2) the description suggests -
        it runs every frame with MAX_POP agents in it.
        """
        n = len(xs)
        if n == 1:
            return xs[0]
        pre = [0.0] * (n + 1)
        for i, v in enumerate(xs):
            pre[i + 1] = pre[i] + v
        here = self.x + RENDER_W * 0.5
        best_count = -1
        best_mean = here
        j = 0
        for i in range(n):
            if j < i:
                j = i
            while j < n and xs[j] <= xs[i] + RENDER_W:
                j += 1
            count = j - i
            mean = (pre[j] - pre[i]) / count
            if count > best_count or (
                count == best_count and abs(mean - here) < abs(best_mean - here)
            ):
                best_count = count
                best_mean = mean
        return best_mean

    # ------------------------------------------------------------- manual --
    def nudge(self, dx: float) -> bool:
        """Pan by *dx* world px and suspend ``follow`` for MANUAL_HOLD.

        This is where WASD and a right/middle-button drag land once
        ``Preview.pan_world`` has absorbed what the zoomed frame could take: at
        zoom 1 the frame pan is a documented no-op, so 100% of the input spills
        here and the world scrolls.

        *dx* IS ROUTINELY FRACTIONAL and that is the whole of what this method
        has to get right. A drag arrives as one call per MOUSEMOTION, converted
        from window px through the letterbox, so on any window wider than
        RENDER_W one window pixel is *less* than one world pixel: 0.833 at
        1920x1080, 0.625 at 2560x1440, 0.463 at 3840x2160. Rounding each call
        on its own threw all of that away - at 4K a slow drag moved the camera
        exactly nothing, which is the "cannot move the screen with the mouse"
        report. So the remainder is carried in ``_frac`` and spent on the next
        call, while ``self.x`` stays a whole number for the terrain cache and
        the particle array.

        ``_frac`` is dropped whenever the clamp bites, and that is deliberate
        rather than tidy-mindedness: pushing at the end of the map otherwise
        banks every discarded request into a hidden charge that fires as one
        jump the moment the user drags back the other way.

        THERE ARE THREE BANDS, not two, and the middle one is what stops
        follow() stealing the view out from under a gesture that is still
        going. A pan the preview's zoomed frame absorbs completely spills
        nothing, so on the old two-band rule the hold was never armed and never
        *refreshed*: a user who panned, zoomed in and kept dragging inside the
        frame had follow() take the camera six seconds later with the button
        still down. Measured, 1044 world px of it in one 12 s drag. So:

            |dx| <  HOLD_EPS   float noise. Nothing happens.
            |dx| <  NUDGE_EPS  HOLD TOKEN. The user is panning and the frame
                               took all of it: refresh MANUAL_HOLD, move
                               nothing, and leave ``_frac`` strictly alone -
                               a token is not a fraction of a pixel owed.
            otherwise          motion, as below.

        Returns whether the camera actually moved - pushed against the end of
        the map it has not, and on a hold token it has not by definition, but
        the hold is armed for both, or follow would yank the view back the
        instant the user reached the edge or zoomed in.
        """
        try:
            d = float(dx)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(d):
            return False
        mag = abs(d)
        if mag < self.HOLD_EPS:
            return False
        if mag < self.NUDGE_EPS:
            self._manual_t = self.MANUAL_HOLD
            return False
        self._manual_t = self.MANUAL_HOLD
        before = self.x
        want = self.x + self._frac + d
        pos = _clamp(want, 0.0, MAX_LEFT)
        self.x = float(round(pos))
        self._frac = 0.0 if (pos <= 0.0 or pos >= MAX_LEFT) else (pos - self.x)
        return self.x != before

    def snap_to(self, wx_center: float) -> None:
        """Cut - no ease - to a frame centred on world x *wx_center*.

        For load and for ``randomise_terrain``: the colony did not walk there,
        it was simply somewhere else the next time we looked. See
        :meth:`cut_to`, which is what the app actually calls.
        """
        try:
            c = float(wx_center)
        except (TypeError, ValueError):
            return
        if not math.isfinite(c):
            return
        self.x = float(round(_clamp(c - RENDER_W * 0.5, 0.0, MAX_LEFT)))
        # A cut discards history by definition, pending sub-pixel included.
        self._frac = 0.0

    def cut_to(self, world: Any) -> None:
        """Cut straight to where ``follow`` would have crawled. Never raises.

        The reason this exists rather than app.py reaching for ``snap_to`` and
        ``colony_center`` itself: the answer to "where should the frame be?"
        is the cluster rule in :meth:`follow`, and a second, simpler answer
        computed elsewhere would put the cut in one place and the ease in
        another, so the camera would jump on load and then immediately drift.

        Called for the three moments where the colony did not walk into view -
        cold start, world load, and ``randomise_terrain`` - because easing
        instead means up to eleven seconds of empty land at MAX_SPEED, and for
        the tray's randomise command that is eleven seconds of not seeing the
        thing you just asked for. Any manual hold is cleared too: a cut is the
        app deciding where to look, and there is no user pan to protect.
        """
        try:
            xs = self._agent_xs(world)
            centre = self._cluster_centre(xs) if xs else self._fallback_centre(world)
            self.snap_to(centre)
            self._manual_t = 0.0
        except Exception:
            return

    def resume_follow(self) -> None:
        """Cancel a manual hold, so the next ``follow`` takes effect."""
        self._manual_t = 0.0
        self._frac = 0.0

    @property
    def manual(self) -> bool:
        """True while a manual pan is suppressing ``follow``."""
        return self._manual_t > 0.0

    def __repr__(self) -> str:      # pragma: no cover - diagnostics only
        return f"Camera(x={self.x:.0f}{', manual' if self.manual else ''})"


class _IdentityCamera(Camera):
    """A camera that is permanently the identity transform: world x IS screen x.

    THIS IS AN OPT-IN FOR CONTACT SHEETS, NOT A DEFAULT, and the distinction is
    the most expensive thing this project has learned. The original design made
    it the default value of every render entry point's ``cam`` parameter, on the
    argument that a lane which forgot to thread the camera would then draw
    something that sits still while the world scrolls - visible, and therefore
    better than a crash. It is not better. What actually happened was that
    entry points silently drew at world coordinates on a 1600 px frame, so
    every stickman standing past x=1600 was drawn off the right-hand edge and
    the colony was simply invisible, on the wallpaper, with nothing in the log
    and no exception anywhere. A silent fallback does not fail loudly enough to
    be found; it fails quietly enough to be *shipped*.

    So the tree does the opposite now: ``cam`` is a REQUIRED keyword-only
    argument on every entry point that converts a coordinate, and omitting it
    is a TypeError on the first frame. Several ``__main__`` blocks assert that
    TypeError deliberately. If you are adding an entry point, give it
    ``*, cam: Camera`` with no default - do not reintroduce one because this
    class exists.

    What it is still for: the ``__main__`` contact sheets and vignette
    harnesses, which draw one sprite onto a bare surface where world space and
    frame space genuinely are the same thing. Those pass ``cam=IDENTITY``
    *explicitly*, at the call site, which is a statement that the two spaces
    coincide rather than an accident of a missing argument.

    Immutable on purpose: it is a module singleton, and one stray ``follow`` or
    ``nudge`` on it would move that ground truth out from under every caller at
    once.

    THAT SENTENCE USED TO BE FALSE, and it is enforced now rather than asserted.
    The class only ever overrode the mutating *methods*, so the attributes
    underneath them were wide open: ``IDENTITY.x = 4242.0`` succeeded and read
    back 4242.0, and ``IDENTITY._manual_t = 5.0`` made ``IDENTITY.manual`` True -
    on a singleton every contact sheet in the repo shares. No live bug came of
    it (a repo scan finds no such assignment outside ``sky.py``'s ``__main__``,
    which mutates a local ``Camera``), but a docstring that promises a property
    the object does not have is exactly how a lane spends a round measuring
    against something that is not true. ``__setattr__`` and ``__delattr__`` raise
    now, so the promise is checkable instead of aspirational; ``__init__`` goes
    through ``object.__setattr__`` because it is the one writer that is allowed.

    The *methods* stay silent no-ops rather than joining the raisers: callers
    like ``App._pan`` hold whatever camera the renderer gave them and call
    ``nudge`` on it every frame, and an identity camera's answer to that is "I
    did not move", not an exception on the frame path.
    """

    __slots__ = ()

    def __init__(self, x: float = 0.0) -> None:
        # The argument is accepted and ignored: an identity camera sits at 0.0
        # by definition, and honouring an offset would make it something else
        # wearing this class's name.
        object.__setattr__(self, "x", 0.0)
        object.__setattr__(self, "_manual_t", 0.0)
        object.__setattr__(self, "_frac", 0.0)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"IDENTITY is a shared singleton and is immutable; "
            f"refusing to set {name!r}. Construct a Camera() if you need one "
            f"that moves.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"IDENTITY is a shared singleton and is immutable; "
            f"refusing to delete {name!r}.")

    def follow(self, world: Any, dt: float) -> None:
        return

    def nudge(self, dx: float) -> bool:
        return False

    def snap_to(self, wx_center: float) -> None:
        return

    def cut_to(self, world: Any) -> None:
        return

    def resume_follow(self) -> None:
        return


#: The no-op camera. ``IDENTITY.x`` is 0.0 and stays 0.0.
IDENTITY: Camera = _IdentityCamera(0.0)


if __name__ == "__main__":      # pragma: no cover - smoke test
    class _A:
        def __init__(self, x): self.x, self.alive = float(x), True

    class _Pop:
        def __init__(self, xs): self._a = [_A(x) for x in xs]
        def alive_agents(self): return list(self._a)

    class _W:
        def __init__(self, xs): self.population = _Pop(xs)

    cam = Camera()
    # a colony clustered at 4000 should be framed, whatever the map width
    w = _W([3900, 3950, 4000, 4050, 4100])
    for _ in range(600):
        cam.follow(w, 1.0 / 60.0)
    print("clustered:", cam, "centre", cam.x + RENDER_W * 0.5)

    # chasm split 4/2: the camera must hold the larger side, not the gap
    cam2 = Camera()
    cam2.snap_to(1000.0)
    split = _W([900, 950, 1000, 1050, 3000, 3050])
    for _ in range(600):
        cam2.follow(split, 1.0 / 60.0)
    print("split 4/2:", cam2, "centre", cam2.x + RENDER_W * 0.5)

    # deadzone: a settled colony must not crawl
    cam3 = Camera()
    cam3.snap_to(3200.0)
    before = cam3.x
    for _ in range(600):
        cam3.follow(_W([3180, 3200, 3220]), 1.0 / 60.0)
    print("deadzone drift over 10 s:", cam3.x - before, "px")

    # sub-pixel nudges must accumulate, not evaporate. 100 x 0.4 px is 40 px of
    # world, and a 4K window delivers a drag in steps smaller than these.
    cam4 = Camera(1000.0)
    for _ in range(100):
        cam4.nudge(0.4)
    print("100 x nudge(0.4) from 1000:", cam4.x, "(want ~1040)")

    # ...but float noise must not arm the six-second hold.
    cam5 = Camera(1000.0)
    print("nudge(1e-14) armed hold:", cam5.nudge(1e-14) or cam5.manual)

    # ...while a hold token from an absorbed pan must arm it and move nothing.
    cam5b = Camera(1000.0)
    moved = cam5b.nudge(1e-9)
    print("nudge(1e-9): moved", moved, "x", cam5b.x, "armed", cam5b.manual,
          "frac", cam5b._frac)

    # no hidden charge at the end of the map: shove right for a while, then one
    # step left must move exactly one step left.
    cam6 = Camera(MAX_LEFT)
    for _ in range(500):
        cam6.nudge(0.4)
    cam6.nudge(-1.0)
    print("edge charge after 500 pushes then one step back:",
          cam6.x, "(want", MAX_LEFT - 1.0, ")")

    # cut_to lands where follow would have crawled to, in one call.
    far = _W([4200, 4250, 4300])
    cut = Camera()
    cut.cut_to(far)
    eased = Camera()
    for _ in range(600):
        eased.follow(far, 1.0 / 60.0)
    print("cut_to:", cut.x, " ten seconds of ease:", eased.x)

    print("identity:", IDENTITY, "sx(5000) =", IDENTITY.sx(5000.0))
    IDENTITY.follow(w, 1.0)
    IDENTITY.nudge(500.0)
    IDENTITY.cut_to(far)
    print("identity after follow+nudge+cut_to:", IDENTITY.x)

    # ...and "immutable on purpose" now means it, not just that the mutating
    # methods were overridden. Both of these used to succeed silently.
    for name, val in (("x", 4242.0), ("_manual_t", 5.0)):
        try:
            setattr(IDENTITY, name, val)
            print(f"identity.{name} = {val}: NOT REFUSED (x={IDENTITY.x},"
                  f" manual={IDENTITY.manual})")
        except AttributeError:
            print(f"identity.{name} = {val}: refused, x={IDENTITY.x},"
                  f" manual={IDENTITY.manual}")
