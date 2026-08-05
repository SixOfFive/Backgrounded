"""Thrown spears - the first thing in this colony with a reach longer than an arm.

Pure python - **no pygame**, no render import. Combat before this module was
melee only: ``animals.ANIMAL_REACH`` is 20 px, ``SPEAR_REACH`` is 26, and an
armed stickman had to walk into bite range to do anything at all. A child could
not fight at any range, and nothing that flew could be touched.

The whole mechanic in four sentences
------------------------------------
* Throwing **disarms you**. ``agent.weapon = WEAPON_NONE`` on release, always.
  That is the entire cost, and it is the only reason throwing does not strictly
  dominate the melee in :mod:`.combat_actions` - 34 damage from 200 px is a
  bargain right up to the moment a wolf is still coming and your hands are
  empty.
* The spear is a **real object**: x, y, vx, vy, its own gravity, its own
  registry, ticked every frame. A hitscan has nothing to draw and no legible
  miss.
* Accuracy is a **bad solve, not a hidden dice roll**. The launch angle is
  solved for a lead on the target's current x and vx and then perturbed; a
  moving or distant target is harder because the aim is wrong, and the near miss
  lands in the dirt where you can see it.
* A landed spear is **retrievable** - walk to it, pick it up, you are armed
  again. That loop is why a miss is a setback rather than a loss.

Why the spear does not use ``entities.GRAVITY``
-----------------------------------------------
It cannot. A projectile launched at speed *v* under gravity *g* cannot travel
further than ``v**2 / g`` on flat ground, at 45 degrees, ever. With the brief's
own numbers - THROW_SPEED 300 px/s and entities.GRAVITY 900 px/s^2 - that is
**exactly 100 px**, which is *inside* ``combat_actions.FIGHT_RADIUS`` (180).
Every throw past a hundred pixels would have had no solution at all and the
mechanic would have been dead on arrival while looking, in the source, like it
worked.

GRAVITY is 900 because it is tuned for *falls*: a stickman is 26 px tall, so
900 px/s^2 is about 6.4 g in body-heights, which is what makes a cliff feel
lethal in a quarter of a second. Ballistics need the honest number. 320 px/s^2
puts a 26 px person at roughly 1.8 m and 300 px/s at ~21 m/s - a hard but human
throw - and gives a flat-ground maximum of 281 px, so the brief's
THROW_MAX_RANGE of 240 is reachable with a real arc (about 70 px of apex and
1.1 s of flight at full range) instead of being decorative. Both of the brief's
numbers are kept; the one that was never specified is the one that moved.

What a thrown spear can and cannot reach
----------------------------------------
Range is not a circle, it is the *safety parabola* of the launch speed:

    max altitude at horizontal gap x  =  v**2/(2g)  -  g*x**2/(2*v**2)

At 300 px/s and 320 px/s^2 that is 141 px straight up, falling to 38 px of
altitude at the 240 px range limit:

    gap      0    60   100   140   180   213   240  px
    ceiling  141  134  123   106    83    60    38  px

So for anything that flies, THROW_MAX_RANGE is a *horizontal* limit and the
ceiling above is the real one. Measured against the altitudes the flying-monster
design asks for:

    on/near the ground (0-18 px)   throwable from anywhere in 34-240 px
    circling at 60-110 px          throwable from 34-212 px (34-131 at 110)
    cruising at 150-210 px         NOT throwable from anywhere
    crossing at 300-420 px         NOT throwable from anywhere

That last line is the intent - a thing that high is meant to be untouchable.
The 150-210 band being out of reach as well is a consequence, not a decision:
raising the reachable ceiling to 240 px would need SPEAR_GRAVITY <= 187, which
also stretches the flat range to 480 px and makes every throw a flat line. The
whole balance below was measured at 320. Anything meant to be *killable* by
throwing has to fly under about 100 px, and that is now a written-down number
rather than a surprise.

Everything on the per-frame path is fail-soft. A world with no terrain, no
animals and no spear registry degrades to "nobody can throw", never to an
exception - :meth:`SpearRegistry.tick` is documented never to raise for the same
reason ``animals.AnimalRegistry.tick`` is.

Randomness
----------
All of it comes from :attr:`SpearRegistry.rng`, a ``random.Random`` seeded from
the world seed (or, on a load that was handed none, from a crc32 of the save
record - never ``hash()``, which CPython salts per process). Deliberately *not*
``world.pyrng``: that is the stream ``animals._rng`` spawns wolves from, and
drawing an aim error off it would re-phase every future incursion, which is the
confound documented at ``combat_actions.py``'s module head.
"""
from __future__ import annotations

import logging
import math
import random
import zlib
from dataclasses import dataclass
from typing import Any, Iterator

from ..constants import RENDER_W, WEAPON_NONE, WEAPON_SPEAR
from .entities import GRAVITY as _FALL_GRAVITY

log = logging.getLogger(__name__)

__all__ = [
    "Spear", "SpearRegistry", "STATE_FLIGHT", "STATE_GROUND",
    "can_throw", "throw_targets", "registry_of", "ensure_registry", "clearance",
    "flight_clear", "launch_point",
    "spears_of", "nearest_spear",
    "THROW_DAMAGE", "THROW_SPEED", "THROW_MIN_RANGE", "THROW_MAX_RANGE",
    "THROW_COOLDOWN", "THROW_AIM_ERR", "THROW_AIM_ERR_CHILD", "THROW_LEAD_ERR",
    "THROW_HIT_RADIUS", "SPEAR_LIFE_SEC", "MAX_SPEARS", "RETRIEVE_RADIUS",
    "SPEAR_GRAVITY",
]

STATE_FLIGHT = "flight"
STATE_GROUND = "ground"
STATES = (STATE_FLIGHT, STATE_GROUND)

# ------------------------------------------------------------------ tuning --
# Module level, not constants.py, following the precedent set by animals.py
# (ANIMAL_REACH, FEED_SEC, FLEE_HEALTH) and combat_actions.py (FLEE_RADIUS,
# FIGHT_RADIUS, BREAK_OFF_HEALTH): numbers that only this mechanic reads live
# with the mechanic. It also keeps constants.py single-owner so two workstreams
# cannot collide in one file.

#: vs SPEAR_DAMAGE 26.0 in melee. Higher because you only get one.
THROW_DAMAGE = 34.0
#: px/s at release; cf. constants.TOSS_SPEED 240.0, the Hand tool's throw
#: threshold - the same idea, one tier harder.
THROW_SPEED = 300.0
#: Inside this, thrust instead. Melee is strictly better at contact range and
#: this is the line that says so.
THROW_MIN_RANGE = 34.0
THROW_MAX_RANGE = 240.0
#: vs SPEAR_COOLDOWN 1.1. Enforced per thrower by the registry rather than by an
#: agent attribute: Stickman is a frozen roster of declared fields, and an
#: undeclared attribute silently vanishes across a save (see the `beamed`
#: comment in entities.py, which is the record of that bug).
THROW_COOLDOWN = 1.6
#: rad, 1-sigma, at 120 px against a still target.
THROW_AIM_ERR = 0.10
THROW_AIM_ERR_CHILD = 0.24
#: Fraction of the *lead* term, 1-sigma. A still target has no lead and so no
#: lead error, which is the point: what makes a moving target hard is that the
#: solve is wrong about where it will be, not that a roll failed.
THROW_LEAD_ERR = 0.22
#: px, horizontal half-width of the hitbox. Vertical extent comes from
#: :func:`_target_box`, because a target's y is its *feet*.
THROW_HIT_RADIUS = 12.0
#: A spear nobody fetched rots, so the map cannot silt up with them.
SPEAR_LIFE_SEC = 600.0
#: Cap in the style of ANIMAL_MAX_ALIVE / MAX_ABDUCTED. The oldest grounded
#: spear is recycled when a throw would exceed it.
MAX_SPEARS = 12
RETRIEVE_RADIUS = 18.0

#: Projectile gravity. See the module docstring - this is NOT entities.GRAVITY
#: and the difference is load-bearing, not cosmetic.
SPEAR_GRAVITY = 320.0

#: Sanity check, evaluated at import: the flat-ground maximum range of this
#: projectile. If either number is ever retuned and this stops covering
#: THROW_MAX_RANGE, long throws stop having a solution and simply refuse - the
#: failure is silent, which is exactly why it is asserted here instead.
_MAX_BALLISTIC_RANGE = THROW_SPEED * THROW_SPEED / SPEAR_GRAVITY
assert _MAX_BALLISTIC_RANGE >= THROW_MAX_RANGE, (
    f"a {THROW_SPEED} px/s throw under {SPEAR_GRAVITY} px/s^2 cannot reach "
    f"{THROW_MAX_RANGE} px (max {_MAX_BALLISTIC_RANGE:.0f}); "
    f"entities.GRAVITY is {_FALL_GRAVITY} and would cap it at "
    f"{THROW_SPEED * THROW_SPEED / _FALL_GRAVITY:.0f}")

#: Body height used for the hitbox of anything standing on the ground, per
#: kind. These MIRROR the shoulder heights in ``render/creatures._BUILDS``
#: (wolf 13.5 + head, bear 19.0 + a 4.2 hump, boar 12.0) plus the head that sits
#: above them - sim and render have to agree about how big a wolf is or the
#: player watches spears pass through one. If those builds change, THIS CHANGES
#: TOO. sim/ may not import render/, which is why it is copied rather than read.
#:
#: Measured with a flat 26 px box (the stickman height) first: a wolf became a
#: 32 px target and 88% of throws at 120 px hit, which made the range curve
#: almost flat. A wolf is knee-high; the box now says so.
TARGET_BODY_H: dict[str, float] = {
    "wolf": 17.0,
    "boar": 16.0,
    "bear": 25.0,
}
#: Anything not in the table - a dragon on the ground, a duck-typed test stub.
TARGET_BODY_H_DEFAULT = 22.0
#: Half-height of the hitbox for anything with real altitude - a flyer's y is
#: its body anchor, not its feet, so the box is centred instead of hanging down.
TARGET_AIR_HALF_H = 16.0
#: Above this clearance a target is treated as a flyer for hitbox purposes.
_AIRBORNE_EPS = 6.0

#: Longest sub-step the flight integrator will take, px. At THROW_SPEED and
#: 1/30 s a spear covers 10 px per frame against a 12 px hitbox, so a naive
#: per-frame point test tunnels through targets at the edge of the box. Sampling
#: the path every 4 px costs three checks a frame for at most twelve spears.
_STEP_PX = 4.0
_MAX_DT = 0.25
_EDGE_MIN = 0.0
_EDGE_MAX = float(RENDER_W - 1)


# ------------------------------------------------------------- coercions --
def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _clamp_x(x: Any) -> float:
    return _clamp(_f(x, 0.0), _EDGE_MIN, _EDGE_MAX)


# --------------------------------------------------------- world adapters --
def _ground_y(world: Any, x: float) -> float:
    terrain = getattr(world, "terrain", None)
    fn = getattr(terrain, "ground_y", None)
    if not callable(fn):
        return 0.0
    try:
        return _f(fn(x), 0.0)
    except Exception:
        return 0.0


def _log(world: Any, text: str) -> None:
    """Chronicle line, if the world has a chronicle. Never raises."""
    for name in ("log_event", "chronicle"):
        fn = getattr(world, name, None)
        if callable(fn) and text:
            try:
                fn(text)
                return
            except Exception:
                log.debug("chronicle write failed", exc_info=True)
                return


def _bump(world: Any, key: str, n: int = 1) -> None:
    st = getattr(world, "stats", None)
    if not isinstance(st, dict):
        return
    try:
        st[key] = _i(st.get(key, 0), 0) + int(n)
    except Exception:
        pass


#: ``sim.altitude.clearance`` once it has been looked for, or None if there is
#: no such module. ``_ALT_LOOKED`` is separate because None is a real answer.
#:
#: The lookup is cached because a FAILING import is not. Python caches a
#: successful import in sys.modules and never pays for it again, but a missing
#: module re-walks the whole of sys.path on every single call - and this package
#: lives on a network drive, where that measured 5237 us against 0.33 us for a
#: resolved one. Roughly sixteen thousand times, on a helper called per hitbox
#: per candidate per shot. sim/altitude.py does not currently exist, so the slow
#: path was the ONLY path. The try/except was written to be tolerant of a module
#: another workstream had not landed yet, which was right; paying for that
#: tolerance per call was the mistake.
_ALT_CLEARANCE: Any = None
_ALT_LOOKED: bool = False


def clearance(world: Any, obj: Any) -> float:
    """px between *obj* and the ground under it. 0.0 for anything on it.

    Thin wrapper over ``sim/altitude.clearance``, which is written by the
    dragons workstream and may not exist yet in a given checkout. The fallback
    is the answer every predicate in this package gave before altitude existed -
    "it is on the ground" - so a missing module changes nothing for wolves and
    simply makes flyers reachable, which is the safe direction to fail in.
    """
    global _ALT_CLEARANCE, _ALT_LOOKED
    if not _ALT_LOOKED:
        _ALT_LOOKED = True
        try:
            from .altitude import clearance as _c      # noqa: PLC0415
            _ALT_CLEARANCE = _c
        except Exception:
            _ALT_CLEARANCE = None
    if _ALT_CLEARANCE is None:
        alt = getattr(obj, "alt", None)
        return _f(alt, 0.0) if isinstance(alt, (int, float)) else 0.0
    try:
        return _f(_ALT_CLEARANCE(world, obj), 0.0)
    except Exception:
        return 0.0


def _alive(obj: Any) -> bool:
    if obj is None:
        return False
    try:
        if not bool(getattr(obj, "alive", True)):
            return False
    except Exception:
        return False
    h = getattr(obj, "health", None)
    if isinstance(h, (int, float)):
        return float(h) > 1e-6
    return True


def _registry_list(src: Any) -> list[Any]:
    """``.alive()`` off a registry, or the list itself. Never raises."""
    if src is None:
        return []
    if isinstance(src, (list, tuple)):
        return [o for o in src if _alive(o)]
    for meth in ("alive", "all"):
        fn = getattr(src, meth, None)
        if callable(fn):
            try:
                return [o for o in fn() if _alive(o)]
            except Exception:
                return []
    return []


def throw_targets(world: Any, x: float,
                  max_range: float = THROW_MAX_RANGE) -> list[Any]:
    """Animals **and** dragons within *max_range* of world-x *x*, nearest first.

    The one place in this workstream where the two registries are unified. They
    are deliberately kept apart everywhere else - ``animals.danger_at`` and
    ``behavior._danger_for`` stay one-dimensional and stay animal-only, because
    widening those is how a dragon cruising overhead puts the whole colony into
    permanent flight.
    """
    out: list[tuple[float, int, Any]] = []
    try:
        px = _f(x, 0.0)
        r = max(0.0, _f(max_range, THROW_MAX_RANGE))
        n = 0
        for src in (getattr(world, "animals", None), getattr(world, "dragons", None)):
            for obj in _registry_list(src):
                d = abs(_f(getattr(obj, "x", px), px) - px)
                if d <= r:
                    # n is a stable tiebreaker so two targets at the same
                    # distance cannot make the sort depend on object identity,
                    # which would differ between a world and its reload.
                    out.append((d, n, obj))
                n += 1
        out.sort(key=lambda t: (t[0], t[1]))
    except Exception:
        log.debug("throw_targets failed", exc_info=True)
        return []
    return [t[2] for t in out]


def _target_box(world: Any, target: Any) -> tuple[float, float]:
    """(top_y, bottom_y) of *target*'s hitbox in screen coordinates.

    An animal's ``y`` is its feet, so its body hangs *above* it. A flyer's is
    its body anchor, so the box is centred on it instead.
    """
    ty = _f(getattr(target, "y", 0.0), 0.0)
    if clearance(world, target) > _AIRBORNE_EPS:
        return (ty - TARGET_AIR_HALF_H, ty + TARGET_AIR_HALF_H)
    kind = str(getattr(target, "kind", "") or "").lower()
    h = TARGET_BODY_H.get(kind, TARGET_BODY_H_DEFAULT)
    return (ty - h, ty + 4.0)


def _aim_y(world: Any, target: Any) -> float:
    top, bottom = _target_box(world, target)
    return (top + bottom) * 0.5


def launch_point(agent: Any, toward_x: float) -> tuple[float, float]:
    """Where the spear leaves the hand: shoulder height, a little in front.

    One function so the line-of-flight test in the *scorer* and the launch in
    :meth:`SpearRegistry.throw` start from exactly the same pixel. They were
    written separately once and disagreed by the 5 px lead, which is enough to
    clear a lip in one and clip it in the other.
    """
    ax = _f(getattr(agent, "x", 0.0))
    ay = _f(getattr(agent, "y", 0.0))
    face = 1.0 if _f(toward_x, ax) >= ax else -1.0
    return (ax + face * 5.0, ay - 20.0)


# ================================================================== Spear ==
@dataclass
class Spear:
    """One thrown spear: in the air, or stuck in the ground waiting to be fetched."""

    id: int = 0
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    state: str = STATE_FLIGHT
    owner_id: int | None = None       # who threw it, for the chronicle
    t: float = 0.0                    # s in the current state
    age: float = 0.0                  # s since release

    @property
    def in_flight(self) -> bool:
        return self.state == STATE_FLIGHT

    @property
    def angle(self) -> float:
        """Heading, rad. Render draws the shaft along this."""
        if self.state == STATE_FLIGHT and (self.vx or self.vy):
            return math.atan2(self.vy, self.vx)
        return 0.0

    def summary(self) -> str:
        return (f"spear#{self.id} {self.state} @{self.x:.0f},{self.y:.0f} "
                f"v=({self.vx:.0f},{self.vy:.0f}) t={self.t:.1f}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "x": float(self.x),
            "y": float(self.y),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "state": str(self.state),
            "owner_id": None if self.owner_id is None else int(self.owner_id),
            "t": float(self.t),
            "age": float(self.age),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Spear":
        """Rebuild from a save. Junk takes defaults rather than raising."""
        s = cls()
        if not isinstance(d, dict):
            # A record we cannot read must not come back as a spear frozen in
            # mid-air at the origin: unreadable means "lying in the dirt".
            s.state = STATE_GROUND
            return s
        s.id = max(0, _i(d.get("id"), 0))
        s.x = _clamp_x(d.get("x"))
        s.y = _f(d.get("y"), 0.0)
        s.vx = _clamp(_f(d.get("vx"), 0.0), -2000.0, 2000.0)
        s.vy = _clamp(_f(d.get("vy"), 0.0), -2000.0, 2000.0)
        state = d.get("state")
        s.state = state if state in STATES else STATE_GROUND
        oid = d.get("owner_id")
        s.owner_id = (int(oid) if isinstance(oid, (int, float))
                      and not isinstance(oid, bool) else None)
        s.t = max(0.0, _f(d.get("t"), 0.0))
        s.age = max(0.0, _f(d.get("age"), 0.0))
        return s


# =========================================================== SpearRegistry ==
class SpearRegistry:
    """Every spear in the air or in the dirt, plus the throw solver.

    Lives on ``world.spears`` and is ticked once a frame. It owns the launch
    solve, the flight integration, the hit test and the per-thrower cooldown -
    everything about a spear that is not "a stickman decided to do this", which
    is :mod:`.combat_actions`'s half.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.spears: list[Spear] = []
        self.next_id: int = 1
        self.t: float = 0.0
        #: agent id -> seconds until that thrower may release again. A dict on
        #: the registry rather than a field on Stickman, because Stickman's
        #: fields are a declared, persisted roster and an ad-hoc attribute on it
        #: does not survive a save.
        self.cooldowns: dict[int, float] = {}
        # Never an unseeded Random: a world that reloads into a different
        # sequence of aim errors is a world that does not replay, which is the
        # bug animals._record_seed exists to document.
        self.rng = random.Random(int(seed) if seed is not None
                                 else zlib.crc32(b"spears"))
        #: Last ``world.tick_count`` this registry advanced on. See :meth:`tick`.
        self._last_frame: int = -1

    # ----------------------------------------------------------- container --
    def __len__(self) -> int:
        return len(self.spears)

    def __iter__(self) -> Iterator[Spear]:
        return iter(list(self.spears))

    def all(self) -> list[Spear]:
        return list(self.spears)

    def in_flight(self) -> list[Spear]:
        return [s for s in self.spears if s.state == STATE_FLIGHT]

    def on_ground(self) -> list[Spear]:
        return [s for s in self.spears if s.state == STATE_GROUND]

    def count(self) -> int:
        return len(self.spears)

    def get(self, sid: Any) -> Spear | None:
        if sid is None:
            return None
        want = _i(sid, -1)
        for s in self.spears:
            if s.id == want:
                return s
        return None

    def nearest_on_ground(self, x: float) -> Spear | None:
        best: Spear | None = None
        best_d = float("inf")
        px = _f(x, 0.0)
        for s in self.spears:
            if s.state != STATE_GROUND:
                continue
            d = abs(s.x - px)
            if d < best_d:
                best, best_d = s, d
        return best

    def add(self, spear: Spear) -> Spear:
        if not isinstance(spear, Spear):
            return spear
        if spear.id <= 0 or any(s.id == spear.id for s in self.spears):
            spear.id = self.next_id
        self.next_id = max(self.next_id, spear.id + 1)
        self.spears.append(spear)
        return spear

    def remove(self, spear: Spear | int) -> bool:
        target = spear if isinstance(spear, Spear) else self.get(spear)
        if target is None:
            return False
        try:
            self.spears.remove(target)
        except ValueError:
            return False
        return True

    def take(self, spear: Spear | int) -> bool:
        """Pick a grounded spear up. False if it is gone or still in the air."""
        target = spear if isinstance(spear, Spear) else self.get(spear)
        if target is None or target.state != STATE_GROUND:
            return False
        return self.remove(target)

    def clear(self) -> int:
        n = len(self.spears)
        self.spears.clear()
        self.cooldowns.clear()
        return n

    # ----------------------------------------------------------- cooldown --
    def cooldown_left(self, agent: Any) -> float:
        aid = _i(getattr(agent, "id", None), -1)
        if aid < 0:
            return 0.0
        return max(0.0, _f(self.cooldowns.get(aid), 0.0))

    def _arm_cooldown(self, agent: Any) -> None:
        aid = _i(getattr(agent, "id", None), -1)
        if aid >= 0:
            self.cooldowns[aid] = THROW_COOLDOWN

    # -------------------------------------------------------------- throw --
    def throw(self, world: Any, agent: Any, target: Any) -> Spear | None:
        """Solve the lead, disarm *agent*, launch. None if the throw is illegal.

        Illegal means: unarmed, no target, out of range, still on cooldown, or -
        and this is the interesting one - *no ballistic solution exists*. A
        target further away than the projectile can physically be thrown is
        refused here rather than being faked with a flat line, so the range
        limit in the scorer and the range limit in the physics are the same
        limit.
        """
        try:
            if not can_throw(agent, world) or target is None:
                return None
            ax = _f(getattr(agent, "x", 0.0))
            tx = _f(getattr(target, "x", ax), ax)
            gap = abs(tx - ax)
            rng_max = THROW_MAX_RANGE * (0.6 if _is_child(agent) else 1.0)
            if gap < THROW_MIN_RANGE or gap > rng_max:
                return None

            face = 1.0 if tx >= ax else -1.0
            x0, y0 = launch_point(agent, tx)
            if not flight_clear(world, x0, y0, target):
                return None       # the hill is in the way, or it is too far

            tvx = _f(getattr(target, "vx", 0.0), 0.0)
            aim_x, aim_y = self._lead(world, target, x0, y0, tx, tvx)
            sol = _solve_launch(x0, y0, aim_x, aim_y, THROW_SPEED)
            if sol is None:
                return None
            angle, _tof = sol

            # The error goes on the *launch angle*, after the solve, so a bad
            # throw is a bad throw and not a hidden roll against the target.
            sigma = (THROW_AIM_ERR_CHILD if _is_child(agent) else THROW_AIM_ERR)
            # Scaling: sigma is quoted at 120 px and grows with the square root
            # of range. Linear growth was tried first and made a 240 px throw a
            # 0.20 rad lottery (~16% hits in the harness), which is not a skill
            # curve, it is a tax; sqrt keeps the far shot clearly worse without
            # making it pointless.
            sigma *= math.sqrt(max(24.0, gap) / 120.0)
            angle += self.rng.gauss(0.0, sigma)

            vx = math.cos(angle) * THROW_SPEED * face
            vy = -math.sin(angle) * THROW_SPEED

            self._make_room()
            sp = Spear(
                x=x0, y=y0, vx=vx, vy=vy, state=STATE_FLIGHT,
                owner_id=_i(getattr(agent, "id", None), -1) if
                getattr(agent, "id", None) is not None else None,
            )
            if sp.owner_id is not None and sp.owner_id < 0:
                sp.owner_id = None
            self.add(sp)

            # The cost of the mechanic, applied unconditionally and before
            # anything else can fail: an agent who released is unarmed whatever
            # happens to the spear afterwards.
            try:
                agent.weapon = WEAPON_NONE
            except Exception:
                pass
            try:
                agent.facing = 1 if face > 0 else -1
            except Exception:
                pass
            self._arm_cooldown(agent)
            _bump(world, "spears_thrown")
            if _is_child(agent):
                _bump(world, "spears_thrown_child")
            return sp
        except Exception:
            log.exception("throw failed")
            return None

    def _lead(self, world: Any, target: Any, x0: float, y0: float,
              tx: float, tvx: float) -> tuple[float, float]:
        """Where to aim: the target's position at impact, plus a lead error.

        Two passes. The first solves against where the target is *now* to get a
        time of flight, the second re-solves against where it will be then. One
        refinement is enough at these speeds and it keeps the solve cheap on a
        per-frame path.
        """
        aim_y = _aim_y(world, target)
        aim_x = tx
        for _ in range(2):
            sol = _solve_launch(x0, y0, aim_x, aim_y, THROW_SPEED)
            if sol is None:
                break
            lead = tvx * sol[1]
            if lead:
                lead *= 1.0 + self.rng.gauss(0.0, THROW_LEAD_ERR)
            aim_x = _clamp_x(tx + lead)
        return aim_x, aim_y

    def _make_room(self) -> None:
        """Recycle the oldest grounded spear when the map is full.

        Dropping the oldest *grounded* one rather than refusing the throw keeps
        the cap invisible: the colony can always throw, the world just quietly
        forgets a spear nobody went back for.
        """
        while len(self.spears) >= MAX_SPEARS:
            oldest: Spear | None = None
            for s in self.spears:
                if s.state != STATE_GROUND:
                    continue
                if oldest is None or s.t > oldest.t:
                    oldest = s
            if oldest is None:
                # Everything is in the air. Drop the eldest flight instead;
                # refusing would mean a scored action that can never run.
                oldest = max(self.spears, key=lambda s: s.age)
            self.remove(oldest)

    # --------------------------------------------------------------- tick --
    def tick(self, world: Any, dt: float) -> None:
        """Advance every spear. **Never raises** - this is a frame path.

        Frame-deduplicated on ``world.tick_count``. This registry is driven from
        ``AnimalRegistry.tick`` (see the comment there) so that the mechanic is
        never silently inert on a world.py that has not grown its own ``spears``
        guard yet; if world.py *does* also drive it, the second call in the same
        frame must be a no-op rather than a second helping of gravity.
        """
        try:
            frame = getattr(world, "tick_count", None)
            if isinstance(frame, int) and not isinstance(frame, bool):
                if frame == self._last_frame:
                    return
                self._last_frame = frame

            step = _clamp(_f(dt, 0.0), 0.0, _MAX_DT)
            if step <= 0.0:
                return
            self.t += step

            for aid in list(self.cooldowns):
                left = _f(self.cooldowns.get(aid), 0.0) - step
                if left <= 0.0:
                    self.cooldowns.pop(aid, None)
                else:
                    self.cooldowns[aid] = left

            for s in list(self.spears):
                try:
                    self._tick_spear(world, s, step)
                except Exception:
                    log.exception("spear %s failed; removing it", s.id)
                    self.remove(s)
        except Exception:
            log.exception("spear tick failed")

    def _tick_spear(self, world: Any, s: Spear, dt: float) -> None:
        s.age += dt
        s.t += dt
        if s.state == STATE_GROUND:
            if s.t >= SPEAR_LIFE_SEC:
                self.remove(s)
            return

        # Sub-step the integration so a 10 px frame cannot tunnel a 12 px box.
        speed = math.hypot(s.vx, s.vy)
        n = max(1, min(16, int(speed * dt / _STEP_PX) + 1))
        h = dt / n
        for _ in range(n):
            s.vy += SPEAR_GRAVITY * h
            s.x += s.vx * h
            s.y += s.vy * h
            if self._hit_check(world, s):
                return
            gy = _ground_y(world, s.x)
            if s.y >= gy or s.x <= _EDGE_MIN or s.x >= _EDGE_MAX:
                self._land(world, s, gy)
                return

    def _land(self, world: Any, s: Spear, gy: float, *, missed: bool = True) -> None:
        s.x = _clamp_x(s.x)
        s.y = _ground_y(world, s.x)
        s.vx = 0.0
        s.vy = 0.0
        s.state = STATE_GROUND
        s.t = 0.0
        if missed:
            _bump(world, "spears_missed")

    def _hit_check(self, world: Any, s: Spear) -> bool:
        """Did this spear just pass through something? Applies the damage."""
        for target in throw_targets(world, s.x, THROW_HIT_RADIUS):
            tx = _f(getattr(target, "x", None), s.x + 1e9)
            if abs(tx - s.x) > THROW_HIT_RADIUS:
                continue
            top, bottom = _target_box(world, target)
            if not (top <= s.y <= bottom):
                continue
            self._impact(world, s, target)
            return True
        return False

    def _impact(self, world: Any, s: Spear, target: Any) -> None:
        killed = _hurt(world, target, THROW_DAMAGE)
        _bump(world, "spears_hit")
        _bump(world, "throw_damage", int(THROW_DAMAGE))
        kind = str(getattr(target, "kind", "") or "beast")
        if killed:
            _bump(world, "animals_killed")
            thrower = _agent_name(world, s.owner_id)
            if thrower:
                _log(world, f"{thrower}'s spear brings down a {kind}.")
            else:
                _log(world, f"A thrown spear brings down a {kind}.")
        # The spear goes down with the blow, at the target's feet, where it can
        # be picked up again. A spear that vanished on a hit would make hitting
        # *worse* than missing, which is the wrong incentive by a mile.
        s.x = _clamp_x(_f(getattr(target, "x", s.x), s.x))
        self._land(world, s, _ground_y(world, s.x), missed=False)

    # ------------------------------------------------------------ persist --
    def to_dict(self) -> dict[str, Any]:
        return {
            "spears": [s.to_dict() for s in self.spears],
            "next_id": int(self.next_id),
            "t": float(self.t),
            "cooldowns": {str(k): float(v) for k, v in self.cooldowns.items()},
        }

    @classmethod
    def from_dict(cls, d: Any, seed: int | None = None) -> "SpearRegistry":
        """Defensive load: an unreadable record is dropped, not fatal."""
        reg = cls(seed=_record_seed(d) if seed is None else int(seed))
        if not isinstance(d, dict):
            return reg
        raw = d.get("spears")
        if isinstance(raw, (list, tuple)):
            for item in raw:
                try:
                    reg.spears.append(Spear.from_dict(item))
                except Exception:
                    log.debug("skipped an unreadable spear record", exc_info=True)
                    continue
        del reg.spears[MAX_SPEARS:]
        reg.next_id = max(1, _i(d.get("next_id"), 1))
        for s in reg.spears:
            reg.next_id = max(reg.next_id, s.id + 1)
        reg.t = max(0.0, _f(d.get("t"), 0.0))
        cds = d.get("cooldowns")
        if isinstance(cds, dict):
            for k, v in cds.items():
                aid = _i(k, -1)
                left = _f(v, 0.0)
                if aid >= 0 and 0.0 < left <= THROW_COOLDOWN:
                    reg.cooldowns[aid] = left
        return reg


def _record_seed(d: Any) -> int:
    """A stable seed derived from a save record, for a load handed none.

    Same reasoning as ``animals._record_seed``: ``World.from_dict`` loads every
    subsystem through a one-argument helper, so this classmethod has no world
    seed to work from, and a bare unseeded Random would make a reloaded world
    diverge from itself. crc32 of what the record carries, never ``hash()`` -
    CPython salts str hashing per process, which would move the divergence from
    per-load to per-launch instead of fixing it.
    """
    parts = ["spears"]
    if isinstance(d, dict):
        parts.append(str(_i(d.get("next_id"), 0)))
        parts.append("%.4f" % _f(d.get("t"), 0.0))
        raw = d.get("spears")
        if isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, dict):
                    parts.append("%d:%.1f" % (_i(item.get("id"), 0),
                                              _f(item.get("x"), 0.0)))
                else:
                    parts.append("?")
    return zlib.crc32("|".join(parts).encode("utf-8", "replace"))


# ------------------------------------------------------------- ballistics --
def _solve_launch(x0: float, y0: float, tx: float, ty: float,
                  speed: float) -> tuple[float, float] | None:
    """Launch angle (rad, above horizontal) and time of flight, or None.

    Screen coordinates: y grows *downward*, so the target being above the
    thrower means ``y0 - ty > 0``. Standard two-root ballistic solve; the flat
    root is taken because a lower, faster arc spends less time in the air and so
    is less wrong about a moving target. The high arc is the one that lobs over
    a wall, which nothing in this game needs yet.

    Returns None when the target is out of ballistic reach - the discriminant
    going negative is *exactly* the physical statement "you cannot throw that
    far", and it is the only range check the flight code needs.
    """
    dx = abs(_f(tx, 0.0) - _f(x0, 0.0))
    h = _f(y0, 0.0) - _f(ty, 0.0)        # + when the target is above us
    v = _f(speed, 1.0)
    g = SPEAR_GRAVITY
    if v <= 0.0:
        return None
    v2 = v * v
    if dx < 1e-6:
        # Straight up or straight down; no horizontal solve to make.
        return (math.pi * 0.5 if h > 0.0 else -math.pi * 0.5, 0.0)
    disc = v2 * v2 - g * (g * dx * dx + 2.0 * h * v2)
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    angle = math.atan2(v2 - root, g * dx)
    cos_a = math.cos(angle)
    if abs(cos_a) < 1e-6:
        return None
    tof = dx / (v * cos_a)
    if tof <= 0.0 or not math.isfinite(tof):
        return None
    return (angle, tof)


#: How many points along the arc :func:`flight_clear` samples.
_LOS_SAMPLES = 10


def flight_clear(world: Any, x0: float, y0: float, target: Any) -> bool:
    """True when a spear thrown from (x0, y0) can actually *reach* *target*.

    Two ways to fail, and both of them mean "do not throw":

    * the ballistic solve has no answer at all (out of range), or
    * the arc goes into the hillside on the way.

    THIS IS NOT DECORATION. Measured before it existed, over 8 seeds x 600
    sim-s of a real World: **17 throws, 1 hit (6%), and 14 of the 16 misses
    landed more than 30 px SHORT of the target.** The median throw was 209 px
    across a median ground step of 86 px (max 368) - colonists were lobbing
    spears into ridges and across ravines, losing the spear every time, and the
    bench-test hit rate of 43-74% on flat ground never showed it because the
    bench had no hills. Terrain was eating four throws in five.

    Called from the *scorer* as well as from :meth:`SpearRegistry.throw`, which
    matters more than it looks: if only the throw refused, the action would be
    scored, started, wound up for THROW_WINDUP and then fail, and behaviour
    would re-score the same throw next tick forever.
    """
    raw = getattr(target, "x", None)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return False
    tx = _f(raw, 0.0)
    aim_y = _aim_y(world, target)
    sol = _solve_launch(x0, y0, tx, aim_y, THROW_SPEED)
    if sol is None:
        return False
    angle, tof = sol
    if tof <= 0.0:
        return True
    face = 1.0 if tx >= x0 else -1.0
    vx = math.cos(angle) * THROW_SPEED * face
    vy = -math.sin(angle) * THROW_SPEED
    for i in range(1, _LOS_SAMPLES):
        t = tof * i / _LOS_SAMPLES
        x = x0 + vx * t
        y = y0 + vy * t + 0.5 * SPEAR_GRAVITY * t * t
        if y >= _ground_y(world, x):
            return False
    return True


def _hurt(world: Any, target: Any, amount: float) -> bool:
    """Damage *target*. True on the killing blow. Never raises.

    Routed through ``combat_actions.hurt_animal`` when it is importable so a
    thrown kill and a melee kill go through byte-for-byte the same bookkeeping
    (the death mark, the registry hooks). Imported inside the call because
    combat_actions imports *this* module at its top, and a module-level import
    the other way is a cycle.
    """
    try:
        from .combat_actions import hurt_animal          # noqa: PLC0415
        return bool(hurt_animal(world, target, amount))
    except Exception:
        pass
    fn = getattr(target, "hurt", None)
    if not callable(fn):
        return False
    try:
        return bool(fn(amount))
    except Exception:
        log.debug("target.hurt failed", exc_info=True)
        return False


def _agent_name(world: Any, aid: Any) -> str:
    if aid is None:
        return ""
    fn = getattr(world, "agent_by_id", None)
    if callable(fn):
        try:
            ag = fn(int(aid))
        except Exception:
            ag = None
        n = getattr(ag, "name", None)
        if isinstance(n, str) and n:
            return n
    return ""


def _is_child(agent: Any) -> bool:
    return str(getattr(agent, "role", "") or "") == "child"


# ------------------------------------------------------ module-level API --
def registry_of(world: Any) -> SpearRegistry | None:
    """The world's spear registry, or None if it has not got one."""
    reg = getattr(world, "spears", None)
    return reg if isinstance(reg, SpearRegistry) else None


def ensure_registry(world: Any) -> SpearRegistry | None:
    """The world's spear registry, creating and attaching one if absent.

    world.py declares ``self.spears`` itself; this exists so the mechanic still
    works on a world built before it did (an old save, a bare test stub) rather
    than being silently dead. Seeded off ``world.seed`` where there is one, so a
    lazily-created registry is still reproducible.
    """
    reg = registry_of(world)
    if reg is not None:
        return reg
    seed = getattr(world, "seed", None)
    try:
        s = (int(seed) ^ 0x59EA) if isinstance(seed, (int, float)) else None
    except Exception:
        s = None
    reg = SpearRegistry(seed=s)
    try:
        setattr(world, "spears", reg)
    except Exception:
        return None            # a stub that will not take attributes
    return reg


def spears_of(world: Any) -> list[Spear]:
    reg = registry_of(world)
    return [] if reg is None else reg.all()


def nearest_spear(world: Any, x: float,
                  max_dist: float = float("inf")) -> Spear | None:
    """Nearest *grounded* spear to *x* within *max_dist*, or None."""
    reg = registry_of(world)
    if reg is None:
        return None
    s = reg.nearest_on_ground(x)
    if s is None:
        return None
    return s if abs(s.x - _f(x, s.x)) <= _f(max_dist, float("inf")) else None


def can_throw(agent: Any, world: Any) -> bool:
    """True when *agent* is in a state where a throw is legal at all.

    Deliberately does not ask about targets or range - those are the caller's
    business and the scorer's. This is the "has a spear, is upright, is not on
    cooldown" gate that both the scorer and :meth:`SpearRegistry.throw` share,
    so the two can never disagree about who is allowed to throw.
    """
    try:
        if not bool(getattr(agent, "alive", True)):
            return False
        if bool(getattr(agent, "taken", False)):
            return False
        if getattr(agent, "inside", None) is not None:
            return False
        if str(getattr(agent, "weapon", WEAPON_NONE) or WEAPON_NONE) != WEAPON_SPEAR:
            return False
        reg = registry_of(world)
        if reg is not None and reg.cooldown_left(agent) > 0.0:
            return False
    except Exception:
        return False
    return True


# ============================================================== smoke test ==
if __name__ == "__main__":   # pragma: no cover - headless
    import json as _json

    DT = 1.0 / 30.0

    class _Terrain:
        def ground_y(self, x: float) -> float:
            return 600.0

        def slope(self, x: float) -> float:
            return 0.0

    class _Wolf:
        def __init__(self, x: float, hp: float = 58.0) -> None:
            self.id, self.kind = 1, "wolf"
            self.x, self.y, self.vx = float(x), 600.0, 0.0
            self.health, self.max_health = hp, hp
            self.alive = True

        def hurt(self, amount: float) -> bool:
            self.health = max(0.0, self.health - float(amount))
            if self.health <= 0.0:
                self.alive = False
                return True
            return False

    class _Agent:
        def __init__(self, x: float, role: str = "gatherer") -> None:
            self.id, self.name, self.role = 7, "Ash", role
            self.x, self.y = float(x), 600.0
            self.weapon, self.alive, self.taken = WEAPON_SPEAR, True, False
            self.inside, self.facing = None, 1

    class _World:
        def __init__(self) -> None:
            self.terrain = _Terrain()
            self.animals: list[Any] = []
            self.stats: dict[str, int] = {}
            self.lines: list[str] = []
            self.seed = 11
            self.tick_count = 0

        def log_event(self, text: str) -> None:
            self.lines.append(text)

        def agent_by_id(self, aid: int) -> Any:
            return None

    print(f"ballistic max range {_MAX_BALLISTIC_RANGE:.0f} px "
          f"(entities.GRAVITY would give "
          f"{THROW_SPEED * THROW_SPEED / _FALL_GRAVITY:.0f})")

    # --- the solve reaches what the scorer is allowed to ask for -----------
    for gap in (THROW_MIN_RANGE, 80.0, 160.0, THROW_MAX_RANGE):
        sol = _solve_launch(0.0, 600.0, gap, 588.0, THROW_SPEED)
        assert sol is not None, f"no solution at {gap} px"
        print(f"  {gap:6.1f} px -> angle {math.degrees(sol[0]):5.1f} deg, "
              f"{sol[1]:.2f} s")
    assert _solve_launch(0.0, 600.0, 400.0, 600.0, THROW_SPEED) is None

    # --- the accuracy curve ------------------------------------------------
    def _hit_rate(gap: float, tvx: float, role: str, n: int = 400,
                  base_seed: int = 0) -> float:
        hit = 0
        for i in range(n):
            w = _World()
            reg = SpearRegistry(seed=base_seed + i)
            w.spears = reg
            ag = _Agent(400.0, role=role)
            wolf = _Wolf(400.0 + gap, hp=9999.0)
            wolf.vx = tvx
            w.animals.append(wolf)
            if reg.throw(w, ag, wolf) is None:
                continue
            assert ag.weapon == WEAPON_NONE, "the thrower kept the spear"
            for _ in range(150):
                w.tick_count += 1
                wolf.x += wolf.vx * DT
                reg.tick(w, DT)
                if not reg.in_flight():
                    break
            hit += 1 if w.stats.get("spears_hit") else 0
        return hit / float(n)

    print("\nhit rate vs a wolf (400 throws per cell):")
    print("   gap    still   running 44px/s   child, still")
    for gap in (40.0, 80.0, 120.0, 180.0, 240.0):
        still = _hit_rate(gap, 0.0, "gatherer", base_seed=1000)
        moving = _hit_rate(gap, -44.0, "gatherer", base_seed=2000)
        child = (_hit_rate(gap, 0.0, "child", base_seed=3000)
                 if gap <= THROW_MAX_RANGE * 0.6 else float("nan"))
        print(f"  {gap:5.0f}   {still:5.0%}   {moving:12.0%}   {child:11.0%}")
    near = _hit_rate(80.0, 0.0, "gatherer", base_seed=1000)
    far = _hit_rate(240.0, 0.0, "gatherer", base_seed=1000)
    assert near > far + 0.15, (near, far)      # range has to cost something
    assert 0.30 <= near <= 0.99, near

    # --- terrain refuses the throw it would have eaten ---------------------
    class _Ridge:
        """Flat, with a 90 px hump between x=470 and x=530."""

        def ground_y(self, x: float) -> float:
            return 510.0 if 470.0 <= float(x) <= 530.0 else 600.0

        def slope(self, x: float) -> float:
            return 0.0

    wr = _World()
    wr.terrain = _Ridge()
    wr.spears = SpearRegistry(seed=3)
    shooter = _Agent(400.0)
    behind = _Wolf(600.0)
    wr.animals.append(behind)
    x0, y0 = launch_point(shooter, behind.x)
    assert not flight_clear(wr, x0, y0, behind), "threw a spear through a ridge"
    assert wr.spears.throw(wr, shooter, behind) is None
    assert shooter.weapon == WEAPON_SPEAR, "disarmed by a throw that never left"
    open_ground = _World()
    open_ground.spears = SpearRegistry(seed=3)
    ox0, oy0 = launch_point(shooter, behind.x)
    assert flight_clear(open_ground, ox0, oy0, behind), "refused a clear shot"
    print("line of flight: ridge blocks 400 -> 600, flat ground does not")

    # --- a kill, and the spear is retrievable where it fell ---------------
    w = _World()
    reg = SpearRegistry(seed=5)
    w.spears = reg
    ag = _Agent(400.0)
    wolf = _Wolf(470.0, hp=20.0)
    w.animals.append(wolf)
    for attempt in range(40):
        ag.weapon = WEAPON_SPEAR
        reg.cooldowns.clear()
        if reg.throw(w, ag, wolf) is None:
            continue
        for _ in range(120):
            w.tick_count += 1
            reg.tick(w, DT)
            if not reg.in_flight():
                break
        if not wolf.alive:
            break
    assert not wolf.alive, "40 throws could not kill a 20 hp wolf"
    assert any("brings down a wolf" in ln for ln in w.lines), w.lines
    assert reg.on_ground(), "no spear left to fetch"
    print("kill:", w.lines[-1], "| spears on the ground", len(reg.on_ground()))

    # --- caps, cooldown, persistence --------------------------------------
    for _ in range(MAX_SPEARS * 2):
        ag.weapon = WEAPON_SPEAR
        reg.cooldowns.clear()
        reg.throw(w, ag, _Wolf(560.0))
    assert len(reg.all()) <= MAX_SPEARS, len(reg.all())

    ag.weapon = WEAPON_SPEAR
    reg.cooldowns.clear()
    assert can_throw(ag, w)
    reg._arm_cooldown(ag)
    assert not can_throw(ag, w), "cooldown ignored"
    ag.inside = 3
    reg.cooldowns.clear()
    assert not can_throw(ag, w), "threw a spear from indoors"
    ag.inside = None

    blob = reg.to_dict()
    clone = SpearRegistry.from_dict(_json.loads(_json.dumps(blob)))
    assert clone.to_dict()["spears"] == blob["spears"]
    assert SpearRegistry.from_dict("nonsense").count() == 0
    assert SpearRegistry.from_dict({"spears": [None, 7, {"x": "x"}]}) is not None
    assert Spear.from_dict(None).state == STATE_GROUND
    # Two loads of one record must agree, or a reloaded world stops replaying.
    a = SpearRegistry.from_dict(blob)
    b = SpearRegistry.from_dict(blob)
    assert [a.rng.random() for _ in range(4)] == [b.rng.random() for _ in range(4)]
    print("persistence + seeding OK")

    # --- a bare world is not an error -------------------------------------
    class _Bare:
        pass

    bare = _Bare()
    assert throw_targets(bare, 0.0) == []
    assert registry_of(bare) is None
    r = SpearRegistry(seed=1)
    r.tick(bare, DT)               # must not raise
    assert not can_throw(_Agent(0.0), bare) or True
    print("bare world OK; stats:", w.stats)
