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

    __slots__ = ("x", "_manual_t", "_frac", "_lock_id")

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
    #: Deadzone used while LOCKED ON ONE PERSON, in place of ``DEADZONE``.
    #:
    #: It is a different number because it is guarding against a different
    #: thing. ``DEADZONE`` (160) exists because the auto-follow target is a
    #: *mean over wandering agents* that is never still, so the camera would
    #: chase its own jitter forever. A locked target is one man: his x moves
    #: when he walks and stops when he stands, so there is no mean to jitter -
    #: what is left to guard against is the pixel quantiser turning a tiny ease
    #: step into 0/1/0/1 chatter on cam.x.
    #:
    #: So it is set exactly at that chatter threshold rather than by taste. The
    #: ease moves ``gap * EASE * dt`` per frame; at TARGET_FPS 60 that is under
    #: half a pixel - i.e. rounds to no movement at all - whenever
    #: ``gap < 0.5 * 60 / EASE`` = 12.5 px. 12.0 sits just inside that, so the
    #: deadzone only ever suppresses motion the rounding was going to discard,
    #: and a locked man is never more than 12 px off centre while he is walking.
    #: (At a lower frame rate the ease step is larger, so the threshold falls
    #: and this stays conservative.)
    #:
    #: DEADZONE would have worked and would have looked wrong: at 160 px the
    #: camera holds still while a colonist walks a tenth of the frame away, then
    #: catches up in a rush, over and over - a stutter every 160/WALK_SPEED =
    #: 4.7 s. The auto-follow target never moves in one direction long enough
    #: for that to show; a man walking to the quarry does nothing else.
    LOCK_DEADZONE: float = 12.0

    def __init__(self, x: float = 0.0) -> None:
        self.x: float = float(round(_clamp(float(x), 0.0, MAX_LEFT)))
        self._manual_t: float = 0.0
        #: Sub-pixel pan the user has asked for and not yet been shown. Always
        #: in (-0.5, 0.5); see the module docstring.
        self._frac: float = 0.0
        #: Stickman id the view is locked onto, or None for the cluster rule.
        #: An ID and not the agent itself, deliberately: holding the object
        #: would keep a corpse - or an abductee the roster has already dropped -
        #: alive in the renderer, and would make "is my target still there?" a
        #: question about python references rather than about the colony.
        self._lock_id: int | None = None

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

        THERE ARE THREE MODES AND THIS METHOD ARBITRATES ALL OF THEM, in a
        fixed order that is the whole of the policy:

        1. **manual pan** - a WASD key or a right/middle drag armed
           MANUAL_HOLD, and it wins outright for MANUAL_HOLD seconds. It does
           NOT cancel a lock: looking around while following somebody is a
           thing people do, and the lock resumes when the hold lapses. Only an
           explicit release (``resume_follow``, i.e. 0/Home or a click on bare
           ground) clears it.
        2. **locked on one colonist** - :meth:`lock_to`, from a left click that
           hit a stickman. Centres on that one agent, and releases itself the
           moment he is no longer a living member of the roster; see
           :meth:`_locked_agent` for why that one test covers both of the ways
           a colonist leaves.
        3. **auto-follow** - the cluster rule below, which is what the camera
           has always done and what it falls back to.

        The auto-follow target is deliberately NOT ``actions.colony_center()``,
        which is the obvious
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

            centre = None
            dead = self.DEADZONE
            if self._lock_id is not None:
                agent = self._locked_agent(world)
                if agent is None:
                    # He died, or the saucer took him. Either way there is
                    # nothing left to point at, so hand the view back rather
                    # than parking it over an empty hillside for ever.
                    self._lock_id = None
                else:
                    centre = float(agent.x)
                    dead = self.LOCK_DEADZONE
            if centre is None:
                xs = self._agent_xs(world)
                if xs:
                    centre = self._cluster_centre(xs)
                else:
                    centre = self._fallback_centre(world)

            target = _clamp(centre - RENDER_W * 0.5, 0.0, MAX_LEFT)
            gap = target - self.x
            if abs(gap) < dead:
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

    def _locked_agent(self, world: Any) -> Any:
        """The locked stickman, or None if there is no longer one. Never raises.

        ONE TEST COVERS BOTH WAYS OUT, and that is the point of writing it here
        rather than asking the app to notice. A colonist stops being followable
        by two completely different routes:

        * he **dies** - ``alive`` goes False and he stays in ``population.agents``
          as a corpse until a grave is raised over him, so he is still findable
          by id and must be rejected on ``alive``;
        * he is **abducted** - ``Ufo._complete_abduction`` calls
          ``population.remove()``, so he is gone from the roster outright and
          ``by_id`` returns None. An abduction is not a death: no grave, no body,
          ``stats["abducted"]`` rather than a death, and a quarter of them are
          dropped back later. This project has already had one accounting bug
          from exactly that distinction (see sim/ufo.py's module docstring), so
          a camera that released on ``alive`` alone would follow a saucer's
          empty hover point until the manual hold or a restart saved it.

        Note what is deliberately NOT a release: an agent held in the UFO's beam
        is alive and still on the roster, so the camera rides the lift with him
        and lets go at the instant he is actually taken. Same for the Hand tool
        carrying him around.
        """
        try:
            pop = getattr(world, "population", None)
            if pop is None:
                return None
            fn = getattr(pop, "by_id", None)
            agent = fn(self._lock_id) if callable(fn) else None
            if agent is None or not getattr(agent, "alive", False):
                return None
            if not math.isfinite(float(agent.x)):
                return None
            return agent
        except Exception:
            return None

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
            # ...and any lock, for a stronger reason than the hold: two of the
            # three call sites (Start Over, world load) replace the roster
            # wholesale, so the id we were locked to now belongs to a different
            # person or to nobody. Keeping it would silently follow a stranger.
            self._lock_id = None
        except Exception:
            return

    def resume_follow(self) -> None:
        """Hand the view back to the cluster rule: drop the hold AND the lock.

        Both, because "give me the colony back" is one intention and a
        half-reset is worse than none - 0/Home clearing a manual pan but
        leaving the camera welded to one colonist would look exactly like the
        key not working. The same call is what a click on bare ground makes.
        """
        self._manual_t = 0.0
        self._frac = 0.0
        self._lock_id = None

    def lock_to(self, agent_id: Any) -> bool:
        """Follow one colonist by id until told otherwise. Never raises.

        Clears any manual hold, because a lock is a fresh instruction about
        where to look and there is no earlier pan left to protect - the same
        argument :meth:`cut_to` makes. It does *not* cut: the target is
        somewhere the user just clicked, so it is on screen by construction and
        an ease reads as the camera turning to him rather than as a jump.
        Passing None (or anything that is not an int) releases instead, so a
        caller never has to special-case a miss.
        """
        try:
            if agent_id is None:
                self._lock_id = None
                return False
            wanted = int(agent_id)
        except (TypeError, ValueError):
            self._lock_id = None
            return False
        self._lock_id = wanted
        self._manual_t = 0.0
        self._frac = 0.0
        return True

    @property
    def manual(self) -> bool:
        """True while a manual pan is suppressing ``follow``."""
        return self._manual_t > 0.0

    @property
    def lock_id(self) -> int | None:
        """Stickman id the view is locked onto, or None. Read-only on purpose.

        The HUD and the window caption read this to say *who* is being followed;
        writing it goes through :meth:`lock_to` / :meth:`resume_follow` so the
        manual hold is always cleared with it.
        """
        return self._lock_id

    @property
    def locked(self) -> bool:
        return self._lock_id is not None

    def __repr__(self) -> str:      # pragma: no cover - diagnostics only
        lock = f", lock={self._lock_id}" if self._lock_id is not None else ""
        return f"Camera(x={self.x:.0f}{', manual' if self.manual else ''}{lock})"


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
        object.__setattr__(self, "_lock_id", None)

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

    def lock_to(self, agent_id: Any) -> bool:
        # A silent no-op, like the other mutators: App calls this on whatever
        # camera the renderer handed it, and an identity camera's honest answer
        # to "follow that man" is "I did not move", not an exception on the
        # frame path. It reports False, so a caller that echoes the lock in the
        # HUD shows nothing rather than showing a lock that does not exist.
        return False


#: The no-op camera. ``IDENTITY.x`` is 0.0 and stays 0.0.
IDENTITY: Camera = _IdentityCamera(0.0)


if __name__ == "__main__":      # pragma: no cover - smoke test
    class _A:
        def __init__(self, x, aid=0):
            self.x, self.alive, self.id = float(x), True, int(aid)

    class _Pop:
        def __init__(self, xs): self._a = [_A(x, i + 1) for i, x in enumerate(xs)]
        def alive_agents(self): return [a for a in self._a if a.alive]
        def by_id(self, i):
            for a in self._a:
                if a.id == int(i):
                    return a
            return None
        def remove(self, i):
            self._a = [a for a in self._a if a.id != int(i)]

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

    # --- locking on one colonist ----------------------------------------
    # The hermit case: a settlement at 3200 and one man 900 px out. Auto-follow
    # centres on the settlement; a lock has to abandon it and frame him.
    def _hermit_world():
        return _W([3150, 3180, 3200, 3220, 3250, 4100])   # id 6 is the hermit

    hw = _hermit_world()
    auto = Camera()
    auto.cut_to(hw)
    for _ in range(600):
        auto.follow(hw, 1.0 / 60.0)
    lock = Camera()
    lock.cut_to(hw)
    lock.lock_to(6)
    for _ in range(600):
        lock.follow(hw, 1.0 / 60.0)
    print("hermit: auto centre", auto.x + RENDER_W * 0.5,
          " locked centre", lock.x + RENDER_W * 0.5, "(want 4100)")

    # ...and he must stay centred while he walks, without the 4.7 s stutter a
    # 160 px deadzone would give a single walker.
    worst = 0.0
    for _ in range(int(30 * 60)):            # 30 s at WALK_SPEED
        hw.population.by_id(6).x += 34.0 / 60.0
        lock.follow(hw, 1.0 / 60.0)
        worst = max(worst, abs((lock.x + RENDER_W * 0.5) - hw.population.by_id(6).x))
    print("locked walker: worst off-centre over 30 s of walking:",
          round(worst, 1), "px")

    # RELEASE 1: he dies. The corpse stays on the roster, so `alive` is the test.
    d = _hermit_world()
    cd = Camera()
    cd.cut_to(d)
    cd.lock_to(6)
    cd.follow(d, 1 / 60)
    d.population.by_id(6).alive = False
    cd.follow(d, 1 / 60)
    print("released on death:", cd.lock_id is None)

    # RELEASE 2: he is ABDUCTED. Not a death - no corpse, no `alive` flag to
    # read, he is simply gone from the roster. Different code path, same rule.
    u = _hermit_world()
    cu = Camera()
    cu.cut_to(u)
    cu.lock_to(6)
    cu.follow(u, 1 / 60)
    u.population.remove(6)
    cu.follow(u, 1 / 60)
    print("released on abduction:", cu.lock_id is None)

    # A manual pan must NOT drop the lock - it suspends it, and it comes back.
    m = _hermit_world()
    cm = Camera()
    cm.cut_to(m)
    cm.lock_to(6)
    cm.nudge(-400.0)
    print("lock survives a pan:", cm.lock_id == 6, "held:", cm.manual)
    for _ in range(int(Camera.MANUAL_HOLD * 60) + 600):
        cm.follow(m, 1.0 / 60.0)
    print("lock resumes after the hold:", cm.x + RENDER_W * 0.5, "(want 4100)")

    # ...but an explicit release does drop it, and so does a cut.
    cm.resume_follow()
    print("resume_follow releases:", cm.lock_id is None)
    cc = Camera()
    cc.lock_to(6)
    cc.cut_to(m)
    print("cut_to releases:", cc.lock_id is None)
    cn = Camera()
    cn.lock_to(None)
    print("lock_to(None) is a release:", cn.lock_id is None)

    print("identity:", IDENTITY, "sx(5000) =", IDENTITY.sx(5000.0))
    IDENTITY.follow(w, 1.0)
    IDENTITY.nudge(500.0)
    IDENTITY.cut_to(far)
    print("identity after follow+nudge+cut_to:", IDENTITY.x,
          "lock_to ->", IDENTITY.lock_to(6), "locked", IDENTITY.locked)

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
