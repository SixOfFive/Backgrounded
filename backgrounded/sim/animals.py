"""Wildlife - the first thing in this world that hunts the colony back.

Pure python - **no pygame**. Like the rest of ``sim/`` this module must stay
importable and runnable headless; the smoke test at the bottom drives a real
:class:`~backgrounded.sim.world.World` without a display.

What an incursion is
--------------------
Every ``ANIMAL_SPAWN_MIN..MAX`` seconds a pack tries to arrive from just past
the edge of frame - wolves in threes, bears and boar alone, per
``ANIMAL_STATS``; see :func:`_offstage_x` for why that is a colony-relative
distance and not the rim of the map any more. They walk
the terrain surface toward whoever is nearest, bite for ``dps`` while in reach,
break off wounded, and give up after ``ANIMAL_LEAVE_SEC``. Kill one and the
camp gets its hide, which is what leather armour is made of - so surviving an
incursion is what makes the next one survivable.

Design notes that are load-bearing for balance
----------------------------------------------
*Animals spread out.* Target choice is nearest-first, but every animal already
committed to a stickman adds ``FOCUS_PENALTY`` px of virtual distance to them.
Three wolves stacked on one person is 24 dps against 100 hp - a four second
execution, and a colony of four cannot lose a quarter of itself to bad luck
about who happened to be closest. Spread out it is 8 dps a head, twelve seconds,
long enough for the victim to run or for someone with a spear to arrive.

*One kill each, then they go.* After a mauling the animal stands over the body
for ``FEED_SEC`` and then leaves; it never picks a second victim. Measured
without this: three wolves against a colony that cannot fight back produced
**15 deaths per 300 s**, because ``world._spawn_replacement`` kept feeding them
new arrivals - a wolf pack turned into a perpetual motion machine. Sated-and-
gone caps an incursion at its own pack size and makes the losses finite.

*Packs lose their nerve.* Once a wave is down half its members the survivors
break off and run. It is what makes a spear worth carrying: killing one wolf
ends the fight rather than just shortening it by a third.

*Scale is not food.* An animal that bites a colonist wearing the dragon's own
hide (``relic == RELIC_SCALE``) gives up on that person permanently - see
:data:`Animal.scale_refused` and :meth:`AnimalRegistry._refuse_scale`. The bite
itself still goes through ``Stickman.hurt`` unchanged, because the 0.9 clamp in
there is the single source of truth about armour and nothing here is allowed to
special-case it. The clamp alone is not enough, though, and that is the whole
reason this exists: ``ARMOUR_DRAGONSCALE`` lets 10% through, so a full
``ANIMAL_LEAVE_SEC`` (90 s) mauling still does 0.8 dps x 90 = 72 damage from a
wolf and 1.5 x 90 = **135** from a bear, against ``MAX_HEALTH`` of 100. Armour
sold as impenetrable that gets its wearer eaten by a bear in a minute and a half
is a bug, and it is a bug on *this* side of the boundary: the wearer is not
tougher, the predator is meant to lose interest. Refusing the target rather than
zeroing the damage keeps leather (0.45) behaving exactly as it does today.

Measured, and measured honestly. The unit pass below pins a bear on a wearer for
its whole patience: with the refusal the wearer walks away on 99.5 hp, and with
a lone wearer the bear gives up and leaves the map rather than chewing. But over
16 same-seed colonies x 60 sim-min - clean / clamp only / clamp + refusal - the
wearer was mauled in 4 of 16 control runs and **0 of 16 in both** armoured arms,
so in a live world the clamp was already enough and the refusal's marginal
benefit is not measurable at n=16. It is insurance against the pinned-bear case,
not a buff anyone can see in the numbers. What the run does rule out is the
displacement worry - refusing a target sends the animal to somebody else, and
colony mauling deaths went 3.50 -> 3.12 -> 2.81 per run across the three arms,
i.e. down if anything, and inside the noise either way (scale - none is -0.69
maulings, 95% CI [-1.56, +0.18]). Total deaths moved +1.12 [-2.25, +4.50]:
nothing.

*Nothing here flies or falls.* ``y`` is assigned from ``terrain.ground_y(x)``
after every move and there is no ``vy`` at all, so an animal cannot fall to its
death or hang in the air. Steep ground shortens the horizontal step instead
(``CLIMB_RATE``) so a cliff slows a wolf down rather than teleporting it.

Every mutation of world state routes through the small ``_...`` adapters at the
top: they degrade to something harmless if the world is a stub or half-built,
because this runs on a per-frame path and a per-frame path that raises is a
program that dies overnight.
"""
from __future__ import annotations

import logging
import math
import random
import zlib
from dataclasses import dataclass
from typing import Any, Iterator

from ..constants import (
    ANIMAL_BEAR,
    ANIMAL_BOAR,
    ANIMAL_KINDS,
    ANIMAL_LEAVE_SEC,
    ANIMAL_MAX_ALIVE,
    ANIMAL_SPAWN_MAX,
    ANIMAL_SPAWN_MIN,
    ANIMAL_STATS,
    ANIMAL_WOLF,
    HERMIT_FIRE_WARD,
    OFFSTAGE,
    RELIC_SCALE,
    RES_HIDE,
    SCENE_CLEAR,
    WORLD_W,
)

log = logging.getLogger(__name__)

__all__ = [
    "Animal", "AnimalRegistry", "CAUSE_MAULED",
    "STATE_APPROACH", "STATE_ATTACK", "STATE_FLEE", "STATE_LEAVE", "STATES",
    "danger_at", "threats", "registry_of",
]

# ------------------------------------------------------------------ states --
STATE_APPROACH = "approach"
STATE_ATTACK = "attack"
STATE_FLEE = "flee"
STATE_LEAVE = "leave"
STATES = (STATE_APPROACH, STATE_ATTACK, STATE_FLEE, STATE_LEAVE)

#: Death cause handed to ``Stickman.hurt``. Mapped into the chronicle by
#: :func:`_register_chronicle_kind` below.
CAUSE_MAULED = "mauled"

# ------------------------------------------------------------------ tuning --
#: Bite range, px. Deliberately shorter than ``SPEAR_REACH`` (26) so an armed
#: stickman gets the first swing in - that gap is most of why a spear matters.
ANIMAL_REACH = 20.0
#: Pursuit only breaks off past reach x this, so a target shuffling one pixel
#: does not flip the animal between approach and attack every tick.
REACH_HYSTERESIS = 1.6
#: Px an animal can reach ABOVE itself. ``ANIMAL_REACH`` is horizontal and was
#: the only gate there was, which made the bite height-blind - and that turned
#: a lookout tower into a gallows.
#:
#: Measured before this existed: a wolf standing 18.8 px from the foot of a
#: tower bit the man on the deck 68 px above it for 375 consecutive ticks and
#: he could not answer one of them. Not for want of a spear - the bands do not
#: overlap. An animal breaks off past ``ANIMAL_REACH * REACH_HYSTERESIS`` =
#: 32 px, and ``throwing.THROW_MIN_RANGE`` is 34, so there is no horizontal
#: distance at which a perched man may both be bitten and throw back. Arming
#: him changed nothing; he died in phase "watch" without the action ever
#: advancing. Widening the throw band was the other candidate fix and is worse:
#: it lets every colonist stab at arm's length and it leaves the real absurdity
#: - a wolf mauling someone two body-heights over its head - in place.
#:
#: 24 px clears a man on a boulder or on the high side of a step, and refuses
#: nothing that is standing on the same ground as the animal.
ANIMAL_VERT_REACH = 24.0
#: Fraction of max health below which an animal gives up and runs.
FLEE_HEALTH = 0.30

#: A fleeing animal's speed multiplier, interpolated by how hurt it is:
#: FLEE_SPEED_HURT at death's door up to FLEE_SPEED_FRESH at the moment it
#: breaks off.
#:
#: A flat 1.35 made a wounded animal *faster* than a healthy hunter, which is
#: both wrong and unwinnable. A wolf is 58hp and a spear does 26, so two hits
#: leave 6hp - already under FLEE_HEALTH - and it then outran the chase
#: (62.1 px/s vs 49.3) with the gap widening 12.8 px/s, so the parting thrust
#: never landed. Measured 0 kills out of 12 for a lone spearman, every wolf
#: escaping at exactly 6hp. Since wolves are ~68% of arrivals and drop the
#: hides armour is made from, that starved the whole gear loop.
#:
#: Bleeding out now slows you down: at 10% health a wolf makes 36.8 px/s
#: against a 49.3 px/s chase, so it can be run down - while a beast that
#: breaks off early still has a real chance of getting away.
FLEE_SPEED_HURT = 0.75
FLEE_SPEED_FRESH = 1.25
#: Seconds an animal stands over a body before wandering off. See module docs -
#: it leaves afterwards rather than hunting again, and that is the single
#: biggest lever on how much an incursion costs.
FEED_SEC = 7.0
#: Px of virtual distance added per animal already committed to a target.
FOCUS_PENALTY = 260.0
#: Px subtracted for the target already held, which stops two animals swapping
#: victims every tick when their focus penalties are symmetric.
STICKY_BONUS = 140.0
#: Px/s of vertical ground change an animal can follow. Steeper ground
#: shortens the step instead of dragging it up the face at walking speed.
CLIMB_RATE = 150.0
#: Never shorten a step below this fraction, or a sheer face stalls the animal
#: on the spot forever.
MIN_STEP_FRAC = 0.12
#: Default radius for :meth:`AnimalRegistry.danger_at`.
DANGER_RADIUS = 110.0
#: Px past the stage edge counted as "arrived" / "gone". See :func:`_offstage_x`
#: for what the stage edge is now that it is no longer the edge of the map.
EDGE_MARGIN = 2.0
#: Retry gap when a spawn attempt is declined (wrong time of day, world full).
SPAWN_RETRY_SEC = 20.0
#: Spacing between pack members as they come over the edge.
PACK_SPACING = 16.0

#: The WORLD's rim - a hard clamp on where an animal may stand, nothing else.
#: It used to double as "the edge of the screen" because the two were the same
#: pixel; they are not any more, and every use that meant the second one now
#: goes through :func:`_offstage_x`.
_EDGE_MIN = 0.0
_EDGE_MAX = float(WORLD_W - 1)
_MAX_DT = 0.25
_FALLBACK_STATS: tuple[float, float, float, int, int] = (60.0, 8.0, 44.0, 1, 1)

#: Relative arrival weight per kind. Wolves are the common incursion; a bear is
#: the one you remember.
_KIND_WEIGHTS: dict[str, float] = {
    ANIMAL_WOLF: 0.50,
    ANIMAL_BOAR: 0.32,
    ANIMAL_BEAR: 0.18,
}

_ARRIVAL_LINES: dict[str, tuple[str, ...]] = {
    ANIMAL_WOLF: (
        "Wolves come down from the ridge.",
        "A wolf pack slips out of the treeline.",
        "There is howling at the edge of the camp.",
    ),
    ANIMAL_BEAR: (
        "A bear comes down out of the high timber.",
        "A bear walks into the valley, unhurried.",
    ),
    ANIMAL_BOAR: (
        "A boar crashes out of the scrub.",
        "A boar comes up from the low ground, bristling.",
    ),
}

#: How a wave is signed off, by kind and outcome. Written out per kind rather
#: than composed from a noun because "The bear are driven off" is not a
#: sentence, and this chronicle is the thing a person actually reads.
_WAVE_LINES: dict[str, dict[str, str]] = {
    ANIMAL_WOLF: {
        "took":   "The wolves withdraw, fed.",
        "driven": "The wolves are driven off.",
        "gone":   "The wolves melt back into the dark.",
    },
    ANIMAL_BEAR: {
        "took":   "The bear withdraws, fed.",
        "driven": "The bear is driven off.",
        "gone":   "The bear wanders back into the timber.",
    },
    ANIMAL_BOAR: {
        "took":   "The boar withdraws, fed.",
        "driven": "The boar is driven off.",
        "gone":   "The boar loses interest and goes.",
    },
}
_WAVE_FALLBACK = {
    "took":   "The animals withdraw, fed.",
    "driven": "The animals are driven off.",
    "gone":   "The animals melt back into the dark.",
}

#: Said once per colony, the first time dragonscale turns a bite. Per kind for
#: the same reason ``_WAVE_LINES`` is: "The bear's teeth" is a different
#: sentence from "The wolves'", and this is a line a person reads.
_SCALE_LINES: dict[str, str] = {
    ANIMAL_WOLF: "The wolf's teeth find nothing but scale.",
    ANIMAL_BEAR: "The bear's jaws find nothing but scale.",
    ANIMAL_BOAR: "The boar's tusks find nothing but scale.",
}
_SCALE_FALLBACK = "Its teeth find nothing but scale."

_RNG = random.Random()


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


def _b(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _clamp_x(x: Any) -> float:
    return _clamp(_f(x, 0.0), _EDGE_MIN, _EDGE_MAX)


def _stats_for(kind: str) -> tuple[float, float, float, int, int]:
    """``(max_health, dps, speed, hides, pack_size)`` for *kind*, never raises."""
    try:
        got = ANIMAL_STATS.get(kind)
        if got is None:
            return _FALLBACK_STATS
        mh, dps, spd, hides, pack = got
        return (_f(mh, 60.0), _f(dps, 8.0), _f(spd, 44.0),
                max(0, _i(hides, 1)), max(1, _i(pack, 1)))
    except Exception:
        return _FALLBACK_STATS


# --------------------------------------------------------- world adapters --
# The world may be a stub, mid-load or missing a subsystem. Every read and
# write goes through here so none of that reaches the state machine.
def _ground_y(world: Any, x: float) -> float:
    terrain = getattr(world, "terrain", None)
    fn = getattr(terrain, "ground_y", None)
    if not callable(fn):
        return 0.0
    try:
        return _f(fn(x), 0.0)
    except Exception:
        return 0.0


def _agents(world: Any) -> list[Any]:
    """Every living stickman, or an empty list if there is no roster."""
    pop = getattr(world, "population", None)
    fn = getattr(pop, "alive_agents", None)
    if callable(fn):
        try:
            return list(fn())
        except Exception:
            return []
    try:
        return [a for a in (pop or ()) if getattr(a, "alive", False)]
    except Exception:
        return []


def _ward_x(world: Any) -> float | None:
    """Where a hermit fire is BURNING, or None. Never raises.

    The one thing on the map that keeps wolves off somebody, and it is deliberate
    that it reads ``fire_active`` rather than merely ``built``: a cold ring of
    stones is not a ward, and the whole point of HERMIT_FIRE_WARD is to make the
    fuel economy matter. See that constant for the measurements.

    The import is deferred to call time for the same reason ``_colony_x``
    defers its one: ``actions`` imports ``entities``/``structures`` and
    ``world`` imports us.
    """
    if HERMIT_FIRE_WARD <= 0.0:
        return None
    try:
        from . import actions as _actions            # noqa: PLC0415 - see above
        fire = _actions.hermit_fire(world, built_only=True)
    except Exception:
        log.debug("hermit fire unavailable", exc_info=True)
        return None
    if fire is None or not getattr(fire, "fire_active", False):
        return None
    try:
        return _f(getattr(fire, "x", 0.0))
    except Exception:
        return None


def _out_of_reach_above(a: Any, agent: Any) -> bool:
    """Is *agent* standing on something too high for *a* to bite?

    Keyed on ``perch_y`` and not on raw dy, because the terrain produces the
    same arithmetic for an entirely innocent reason: on a slope at
    ``MAX_SLOPE_CLIMB`` a target 20 px away in x is legitimately 52 px higher,
    and refusing that would make every hillside chase unwinnable.
    ``Stickman.perch`` is the only writer of ``perch_y`` - it means a platform
    is holding the man off the ground - and it is renewed every tick under
    ``PERCH_LEASE``, so the flag cannot go stale under a dead action.

    A FILTER, like ``_warded``, and deliberately so: a filter degrades
    correctly when the unreachable man is the last one alive, because
    ``_approach_step`` gets None and walks the animal off the map. A damage
    exemption would instead leave the pack milling at the foot of the tower
    forever.
    """
    try:
        if getattr(agent, "perch_y", None) is None:
            return False
        return (_f(getattr(a, "y", 0.0), 0.0)
                - _f(getattr(agent, "y", 0.0), 0.0)) > ANIMAL_VERT_REACH
    except Exception:
        return False


def _warded(agent: Any, ward: float | None) -> bool:
    """Is *agent* standing in the light of a burning hermit fire?"""
    if ward is None:
        return False
    try:
        return abs(_f(getattr(agent, "x", 0.0)) - ward) <= HERMIT_FIRE_WARD
    except Exception:
        return False


def _colony_x(world: Any) -> float:
    """Where the colony is, in world px. The anchor the stage hangs off.

    Prefers ``actions.colony_center`` so every subsystem agrees on one answer,
    and the import is deferred to call time because ``actions`` imports
    ``entities``/``structures`` and ``world`` imports us - the same idiom
    entities.py:1483 uses for exactly this cycle. The mean of the living roster
    is the fallback, because it is what colony_center itself falls back to and
    it works on the stub worlds the smoke tests drive.
    """
    try:
        from . import actions as _actions            # noqa: PLC0415 - see above
        c = _f(_actions.colony_center(world), -1.0)
        if c >= 0.0:
            return _clamp_x(c)
    except Exception:
        log.debug("colony_center unavailable", exc_info=True)
    crowd = _agents(world)
    if crowd:
        return _clamp_x(sum(_f(getattr(a, "x", 0.0)) for a in crowd) / len(crowd))
    return WORLD_W * 0.5


def _offstage_x(centre: float, side: int, margin: float = 0.0) -> float:
    """The x an animal enters from, or leaves by, on *side* of *centre*.

    THE POINT OF THIS FUNCTION. ``sim/`` may not look at render's camera, and
    does not have to: the camera is RENDER_W wide and follows the colony, so
    anything further than OFFSTAGE from the colony is off-camera by
    construction. That is what "the screen edge" meant in this file when the
    world was exactly one frame wide, and this is how it stays true now that it
    is four.

    Left alone as the world rim, a wolf entering at x=0 on a 6400 px map can be
    5000 px from anybody - a 113 s walk at 44 px/s - and it holds one of only
    ANIMAL_MAX_ALIVE slots for the whole of it, invisible. Exiting was the same
    bug in reverse: measured below, an animal at x=3000 leaving to the "nearer
    screen edge" walked 3000 px of dead time before it despawned.

    The clamp is to the WORLD, so a colony seated within a stage-width of the
    map's rim gets an entry at the rim instead - closer than we would like, and
    the honest alternative (never spawning on that side) is worse.

    ``actions.offstage_x(world, side)`` is the same function anchored on the
    world instead of on a number. This one takes the centre already computed
    because ``_exit_step`` runs per animal per tick and would otherwise pay for
    a fresh ``colony_center()`` on every one of them.
    """
    s = 1.0 if side >= 0 else -1.0
    return _clamp(centre + s * (OFFSTAGE + max(0.0, margin)),
                  _EDGE_MIN, _EDGE_MAX)


def _agent_by_id(world: Any, aid: int | None) -> Any:
    if aid is None:
        return None
    fn = getattr(world, "agent_by_id", None)
    if callable(fn):
        try:
            return fn(int(aid))
        except Exception:
            return None
    for a in _agents(world):
        if _i(getattr(a, "id", -1), -1) == int(aid):
            return a
    return None


def _scaled(agent: Any) -> bool:
    """True when *agent* is wearing the dragon's own hide.

    Reads the relic slot, not ``armour``. Deliberate: ``armour`` is a float and
    a lucky number there would silently switch this on for something that is
    only leather-plus - the relic is a name, and only one thing sets it.
    Duck-typed so a stub agent (or a save from before relics existed, where the
    field is simply absent) reads as unarmoured rather than raising.
    """
    return str(getattr(agent, "relic", "") or "") == RELIC_SCALE


def _log(world: Any, text: str) -> None:
    fn = getattr(world, "log_event", None)
    if callable(fn) and text:
        try:
            fn(text)
        except Exception:
            log.debug("chronicle write failed", exc_info=True)


def _give(world: Any, resource: str, qty: int) -> None:
    if qty <= 0:
        return
    fn = getattr(world, "give", None)
    if callable(fn):
        try:
            fn(resource, qty)
            return
        except Exception:
            pass
    pile = getattr(world, "stockpile", None)
    if isinstance(pile, dict):
        try:
            pile[resource] = _i(pile.get(resource, 0), 0) + qty
        except Exception:
            pass


def _rng(world: Any) -> Any:
    """A ``random.Random``-alike. Prefers the world's own seeded stream."""
    r = getattr(world, "pyrng", None)
    if r is not None and callable(getattr(r, "random", None)):
        return r
    return _RNG


def _rand(rng: Any) -> float:
    try:
        v = float(rng.random())
        if 0.0 <= v < 1.0:
            return v
    except Exception:
        pass
    return _RNG.random()


def _uniform(rng: Any, lo: float, hi: float) -> float:
    return lo + (hi - lo) * _rand(rng)


def _record_seed(d: Any) -> int:
    """A seed derived from a save record, for a load that was handed none.

    ``World.from_dict`` loads every subsystem through a one-argument helper, so
    :meth:`AnimalRegistry.from_dict` has no world seed to work from. Falling
    back to an unseeded registry meant a save written before ``next_spawn``
    existed picked its first incursion from the module RNG: five loads of one
    record gave 282.99 / 164.73 / 180.10 / 189.59 / 182.23 s, i.e. a reloaded
    world diverged from itself. crc32 of what the record *does* carry, never
    hash() - CPython salts str hashing per process, which would only move the
    divergence from per-load to per-launch.
    """
    parts = ["animals"]
    if isinstance(d, dict):
        parts.append(str(_i(d.get("next_id"), 0)))
        parts.append("%.4f" % _f(d.get("t"), 0.0))
        raw = d.get("animals")
        if isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, dict):
                    parts.append("%s:%d" % (item.get("kind"), _i(item.get("id"), 0)))
                else:
                    parts.append("?")
    return zlib.crc32("|".join(parts).encode("utf-8", "replace"))


def _pick(rng: Any, options: tuple[str, ...]) -> str:
    if not options:
        return ""
    return options[min(len(options) - 1, int(_rand(rng) * len(options)))]


def _is_night(world: Any) -> bool:
    try:
        return bool(getattr(world, "is_night", False))
    except Exception:
        return False


def _scene(world: Any) -> str:
    try:
        return str(getattr(getattr(world, "events", None), "scene", "") or "")
    except Exception:
        return ""


# ------------------------------------------------------------- chronicle --
def _register_chronicle_kind() -> None:
    """Teach names.py what a mauling looks like, once, at import.

    ``world._reap_dead`` writes one line per death from ``death_cause``, and
    ``"mauled"`` is not in ``names.DEATH_KINDS``, so every animal kill fell
    through to the generic "Something happened." The kill itself is announced
    from :meth:`AnimalRegistry._attack_step` the instant it happens (that is
    the only place that knows *which* animal did it), so the template
    registered here is the later burial line, not a second death notice.

    ``setdefault`` throughout: if names.py ever grows its own, it wins.
    """
    try:
        from . import names as _names                     # noqa: PLC0415

        _names.DEATH_KINDS.setdefault(CAUSE_MAULED, "died_mauled")
        _names.EVENT_TEMPLATES.setdefault("died_mauled", (
            "{name} is buried on the {where} slope.",
            "The camp buries {name} where the animals found them.",
            "They carried {name} back and buried them at the {where} edge.",
        ))
    except Exception:
        log.debug("could not register the mauling chronicle kind", exc_info=True)


_register_chronicle_kind()


# ================================================================== Animal ==
@dataclass
class Animal:
    """One wolf, bear or boar. Everything here survives a restart."""

    id: int = 0
    kind: str = ANIMAL_WOLF

    # --- kinematics. There is no vy: animals are welded to the surface. ----
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0

    health: float = 1.0
    max_health: float = 1.0

    state: str = STATE_APPROACH
    target_id: int | None = None          # stickman id being hunted
    t: float = 0.0                        # s in the current state
    age: float = 0.0                      # s since arrival, drives LEAVE
    facing: int = 1                       # -1 | +1
    anim_t: float = 0.0                   # animation phase accumulator
    fleeing: bool = False                 # wounded and heading for an edge

    #: Seconds left standing over a fresh kill. Not one of the four states on
    #: purpose - render only ever has to know about STATES.
    feed_t: float = 0.0

    #: Stickman id this animal has given up on because they were wearing
    #: dragonscale, or None. One slot, not a set: an animal only ever holds one
    #: target, ``ANIMAL_LEAVE_SEC`` is 90 s, and the colony mean is ~7 people,
    #: so "the one it just tried" is the whole population of interest. Persisted
    #: so a save taken mid-incursion does not hand the wearer back to the pack -
    #: an old save simply has no key and defaults to None, which is today's
    #: behaviour exactly. Safe to hold as a bare id because
    #: ``entities.Population.spawn`` never reuses one, so a refusal cannot
    #: alias onto a replacement colonist who arrives later. The one case a
    #: single slot does not cover is TWO scale wearers in one colony, where the
    #: animal alternates and overwrites this each time; that costs a few extra
    #: approach/attack flips and ~0.1 damage a bite, and it takes two quadruped
    #: kills to arise at all, so a set is not worth carrying in every save file.
    scale_refused: int | None = None

    # ------------------------------------------------------------ derived --
    @property
    def alive(self) -> bool:
        return self.health > 0.0

    @property
    def dps(self) -> float:
        return _stats_for(self.kind)[1]

    @property
    def speed(self) -> float:
        return _stats_for(self.kind)[2]

    @property
    def hides(self) -> int:
        return _stats_for(self.kind)[3]

    @property
    def pack_size(self) -> int:
        return _stats_for(self.kind)[4]

    @property
    def health_frac(self) -> float:
        return _clamp(self.health / max(1e-6, self.max_health), 0.0, 1.0)

    @property
    def hostile(self) -> bool:
        """Coming for someone, as opposed to fleeing, leaving or feeding."""
        return (self.alive and not self.fleeing and self.feed_t <= 0.0
                and self.state in (STATE_APPROACH, STATE_ATTACK))

    def distance_to(self, x: float) -> float:
        return abs(self.x - _f(x, self.x))

    def summary(self) -> str:
        return (f"{self.kind}#{self.id} {self.state} "
                f"{self.health:.0f}/{self.max_health:.0f} @{self.x:.0f}"
                + (f" -> agent {self.target_id}" if self.target_id is not None else ""))

    # -------------------------------------------------------------- combat --
    def hurt(self, amount: float) -> bool:
        """Take *amount* damage. Returns True only on the live -> dead edge.

        The carcass is not removed here: the registry reaps it on the next tick,
        which is where the world (and therefore the stockpile and the chronicle)
        is in scope. Callers in ``actions.py`` only need the boolean.
        """
        dmg = _f(amount, 0.0)
        if not self.alive or dmg <= 0.0:
            return False
        self.health = max(0.0, self.health - dmg)
        if self.health <= 0.0:
            self.vx = 0.0
            self.target_id = None
            return True
        # Wounded past the flee line it breaks off immediately - being speared
        # is exactly the moment an animal decides the camp is not worth it.
        if not self.fleeing and self.health_frac <= FLEE_HEALTH:
            self.fleeing = True
            self.state = STATE_FLEE
            self.t = 0.0
            self.target_id = None
        return False

    # ------------------------------------------------------------ persist --
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "kind": str(self.kind),
            "x": float(self.x),
            "y": float(self.y),
            "vx": float(self.vx),
            "health": float(self.health),
            "max_health": float(self.max_health),
            "state": str(self.state),
            "target_id": None if self.target_id is None else int(self.target_id),
            "t": float(self.t),
            "age": float(self.age),
            "facing": int(self.facing),
            "anim_t": float(self.anim_t),
            "fleeing": bool(self.fleeing),
            "feed_t": float(self.feed_t),
            "scale_refused": (None if self.scale_refused is None
                              else int(self.scale_refused)),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Animal":
        """Rebuild from a save. Junk takes defaults rather than raising."""
        a = cls()
        if not isinstance(d, dict):
            return a
        a.id = max(0, _i(d.get("id"), 0))
        kind = d.get("kind")
        a.kind = kind if kind in ANIMAL_KINDS else ANIMAL_WOLF
        default_hp = _stats_for(a.kind)[0]
        a.x = _clamp_x(d.get("x"))
        a.y = _f(d.get("y"), 0.0)
        a.vx = _clamp(_f(d.get("vx"), 0.0), -400.0, 400.0)
        a.max_health = max(1.0, _f(d.get("max_health"), default_hp))
        a.health = _clamp(_f(d.get("health"), a.max_health), 0.0, a.max_health)
        state = d.get("state")
        a.state = state if state in STATES else STATE_APPROACH
        tid = d.get("target_id")
        a.target_id = int(tid) if isinstance(tid, (int, float)) and not isinstance(tid, bool) else None
        a.t = max(0.0, _f(d.get("t"), 0.0))
        a.age = max(0.0, _f(d.get("age"), 0.0))
        a.facing = -1 if _i(d.get("facing"), 1) < 0 else 1
        a.anim_t = _f(d.get("anim_t"), 0.0)
        a.fleeing = _b(d.get("fleeing"), False)
        a.feed_t = max(0.0, _f(d.get("feed_t"), 0.0))
        ref = d.get("scale_refused")
        a.scale_refused = (int(ref)
                           if isinstance(ref, (int, float))
                           and not isinstance(ref, bool) else None)
        if a.fleeing and a.state in (STATE_APPROACH, STATE_ATTACK):
            a.state = STATE_FLEE
        if a.state in (STATE_APPROACH, STATE_ATTACK):
            a.feed_t = 0.0        # only an animal on its way out is feeding
        return a


# ========================================================== AnimalRegistry ==
class AnimalRegistry:
    """Every animal in the world, plus the incursion timer that brings them."""

    #: See the note in ``__init__``. Declared here as well so an instance that
    #: never ran ``__init__`` still answers the attribute rather than raising
    #: out of ``_pick_target`` on its first frame.
    _ward: float | None = None

    def __init__(self, seed: int | None = None) -> None:
        self.animals: list[Animal] = []
        self.next_id: int = 1
        self.t: float = 0.0
        rng = random.Random(seed) if seed is not None else _RNG
        # The first incursion lands earlier than the steady-state gap so a
        # fresh world is not silent for seven minutes, but never sooner than
        # ANIMAL_SPAWN_MIN - the colony gets its founding day undisturbed.
        self.next_spawn: float = _uniform(
            rng, ANIMAL_SPAWN_MIN, (ANIMAL_SPAWN_MIN + ANIMAL_SPAWN_MAX) * 0.5)
        # Wave bookkeeping: drives the nerve check and the "they are gone" line.
        self._wave_kind: str = ""
        self._wave_size: int = 0        # how many arrived
        self._wave_killed: int = 0      # how many the colony brought down
        self._wave_took: int = 0        # how many people they killed
        #: One-shot for the dragonscale line. Colony-wide, not per animal and
        #: not per wave: "the first time it saves them" is a moment, and a pack
        #: of three would otherwise print it three times in one incursion.
        #: Persisted, so it stays a moment across a reload too.
        self._scale_told: bool = False
        #: Where the hermit's fire is burning this tick, or None. Recomputed
        #: once per :meth:`tick` and read by every animal in it, because
        #: ``_ward_x`` walks the structure registry and doing that four times a
        #: frame for an answer that cannot change inside the frame is waste.
        #: Transient by design - derived from the structures, never saved, and
        #: a class-level default besides so an object built by ``from_dict`` or
        #: by a test stub has it before the first tick.
        self._ward: float | None = None

    # ----------------------------------------------------------- container --
    def __len__(self) -> int:
        return len(self.animals)

    def __iter__(self) -> Iterator[Animal]:
        return iter(list(self.animals))

    def all(self) -> list[Animal]:
        """Every animal record, living or freshly dead."""
        return list(self.animals)

    def alive(self) -> list[Animal]:
        return [a for a in self.animals if a.alive]

    def count(self) -> int:
        """How many living animals are in the world."""
        return sum(1 for a in self.animals if a.alive)

    def get(self, aid: int | None) -> Animal | None:
        if aid is None:
            return None
        wanted = _i(aid, -1)
        for a in self.animals:
            if a.id == wanted:
                return a
        return None

    def nearest(self, x: float, *, hostile_only: bool = False) -> Animal | None:
        """Closest living animal to world *x*, or None."""
        best: Animal | None = None
        best_d = float("inf")
        px = _f(x, 0.0)
        for a in self.animals:
            if not a.alive or (hostile_only and not a.hostile):
                continue
            d = abs(a.x - px)
            if d < best_d:
                best, best_d = a, d
        return best

    def add(self, animal: Animal) -> Animal:
        """Register an animal, assigning an id if it has none."""
        if not isinstance(animal, Animal):
            return animal
        if animal.id <= 0 or any(a.id == animal.id for a in self.animals):
            animal.id = self.next_id
        self.next_id = max(self.next_id, animal.id + 1)
        self.animals.append(animal)
        return animal

    def remove(self, animal: Animal | int) -> bool:
        target = animal if isinstance(animal, Animal) else self.get(_i(animal, -1))
        if target is None:
            return False
        try:
            self.animals.remove(target)
        except ValueError:
            return False
        return True

    def clear(self) -> int:
        n = len(self.animals)
        self.animals.clear()
        self._reset_wave()
        return n

    def _reset_wave(self) -> None:
        self._wave_kind = ""
        self._wave_size = 0
        self._wave_killed = 0
        self._wave_took = 0

    # ------------------------------------------------------- AI query API --
    def danger_at(self, x: float, radius: float = DANGER_RADIUS) -> bool:
        """True if a hunting animal is within *radius* px of world *x*.

        behaviour.py scores fleeing off this, so it deliberately ignores
        animals that are wounded, leaving or standing over a kill: running from
        something that is already walking away is how a colony stops working.
        """
        try:
            px = _f(x, 0.0)
            r = max(0.0, _f(radius, DANGER_RADIUS))
            for a in self.animals:
                if a.hostile and abs(a.x - px) <= r:
                    return True
        except Exception:
            log.debug("danger_at failed", exc_info=True)
        return False

    def threat_at(self, x: float, radius: float = DANGER_RADIUS) -> Animal | None:
        """The nearest hunting animal within *radius* of *x*, or None."""
        try:
            px = _f(x, 0.0)
            r = max(0.0, _f(radius, DANGER_RADIUS))
            best: Animal | None = None
            best_d = float("inf")
            for a in self.animals:
                if not a.hostile:
                    continue
                d = abs(a.x - px)
                if d <= r and d < best_d:
                    best, best_d = a, d
            return best
        except Exception:
            return None

    def threats(self, world: Any = None) -> list[Animal]:
        """Every animal currently hunting, nearest-to-camp first.

        *world* is optional and only used to order the result by how close each
        animal is to the people - the AI wants the one about to bite someone,
        not the one that happens to be lowest in the list.
        """
        try:
            live = [a for a in self.animals if a.hostile]
            agents = _agents(world) if world is not None else []
            if agents:
                def _gap(an: Animal) -> float:
                    return min(abs(an.x - _f(getattr(g, "x", 0.0))) for g in agents)
                live.sort(key=_gap)
            return live
        except Exception:
            log.debug("threats() failed", exc_info=True)
            return []

    # ---------------------------------------------------------- spawning --
    def spawn_wave(
        self,
        world: Any,
        kind: str | None = None,
        edge: int | None = None,
    ) -> list[Animal]:
        """Bring a pack in over a stage edge. Returns what actually arrived.

        Pack size comes from ``ANIMAL_STATS`` - wolves in threes, bears and
        boar alone - trimmed to whatever room ``ANIMAL_MAX_ALIVE`` leaves.

        NEAR-CAMERA, NOT WORLD-EDGE, and that was the decision this migration
        turned on. The world is four frames wide now; entering at x=0 or
        x=WORLD_W-1 would have put a wolf up to 5000 px from the colony, out of
        frame, walking for two minutes before the incursion could start - and
        holding one of four spawn slots while it did. ``_offstage_x`` puts the
        arrival OFFSTAGE px from the colony instead: just outside the camera, so
        the pack still walks in from the dark rather than materialising in
        frame, and still reaches somebody in the same handful of seconds it
        always did. The look is unchanged; only the map got bigger.
        """
        try:
            pre = self.count()
            room = ANIMAL_MAX_ALIVE - pre
            if room <= 0:
                return []
            rng = _rng(world)
            k = kind if kind in ANIMAL_KINDS else self._pick_kind(world, rng)
            max_health, _dps, _speed, _hides, pack = _stats_for(k)
            n = max(1, min(pack, room))
            side = edge if edge in (-1, 1) else (-1 if _rand(rng) < 0.5 else 1)

            centre = _colony_x(world)
            edge_x = _offstage_x(centre, side)

            born: list[Animal] = []
            for i in range(n):
                # Each member a little further IN than the last, exactly as the
                # world-edge version spaced them: the pack strings out along the
                # approach instead of stacking on one column.
                x = _clamp_x(edge_x - side * (EDGE_MARGIN + i * PACK_SPACING))
                a = Animal(
                    kind=k,
                    x=x,
                    y=_ground_y(world, x),
                    health=max_health,
                    max_health=max_health,
                    state=STATE_APPROACH,
                    facing=1 if side < 0 else -1,
                    anim_t=6.0 * _rand(rng),
                )
                self.add(a)
                born.append(a)

            if born:
                # Arriving while a previous wave is still on the map merges
                # into it rather than resetting the counters, or the nerve
                # check would fire off the newcomers' pack size alone.
                if pre <= 0:
                    self._reset_wave()
                self._wave_kind = k
                self._wave_size += len(born)
                _log(world, _pick(rng, _ARRIVAL_LINES.get(k, ())) or
                     f"A {k} comes out of the dark.")
            return born
        except Exception:
            log.exception("spawn_wave failed")
            return []

    def _pick_kind(self, world: Any, rng: Any) -> str:
        """Weighted kind roll. Bears are likelier when the world is unfriendly."""
        weights = dict(_KIND_WEIGHTS)
        if _is_night(world):
            weights[ANIMAL_BEAR] = weights.get(ANIMAL_BEAR, 0.18) * 1.6
        if _scene(world) not in ("", SCENE_CLEAR):
            weights[ANIMAL_BOAR] = weights.get(ANIMAL_BOAR, 0.32) * 1.3
        total = sum(w for w in weights.values() if w > 0.0)
        if total <= 0.0:
            return ANIMAL_WOLF
        roll = _rand(rng) * total
        for k, w in weights.items():
            if w <= 0.0:
                continue
            roll -= w
            if roll <= 0.0:
                return k
        return ANIMAL_WOLF

    def _spawn_chance(self, world: Any) -> float:
        """0..1 odds that *now* is a good moment for an incursion.

        A declined roll costs SPAWN_RETRY_SEC, not a whole interval, so day and
        clear weather delay a wave by a minute or two rather than cancelling it.
        """
        chance = 0.25
        if _is_night(world):
            chance += 0.45
        if _scene(world) not in ("", SCENE_CLEAR):
            chance += 0.22
        return _clamp(chance, 0.05, 0.95)

    def _tick_spawn(self, world: Any, dt: float) -> None:
        self.next_spawn -= dt
        if self.next_spawn > 0.0:
            return
        rng = _rng(world)
        if not _agents(world) or self.count() >= ANIMAL_MAX_ALIVE:
            self.next_spawn = SPAWN_RETRY_SEC
            return
        if _rand(rng) > self._spawn_chance(world):
            self.next_spawn = SPAWN_RETRY_SEC
            return
        if self.spawn_wave(world):
            self.next_spawn = _uniform(rng, ANIMAL_SPAWN_MIN, ANIMAL_SPAWN_MAX)
        else:
            self.next_spawn = SPAWN_RETRY_SEC

    # -------------------------------------------------------------- tick --
    def tick(self, world: Any, dt: float) -> None:
        """Advance every animal and the incursion timer. Never raises."""
        try:
            step = _clamp(_f(dt, 0.0), 0.0, _MAX_DT)
            if step <= 0.0:
                return
            self.t += step
            # Once per frame, before anything reads it. See HERMIT_FIRE_WARD.
            self._ward = _ward_x(world)
            self._tick_spawn(world, step)
            for a in list(self.animals):
                try:
                    self._tick_animal(world, a, step)
                except Exception:
                    log.exception("animal %s failed; removing it", a.id)
                    self.remove(a)
            self._reap(world)
            self._wave_end(world)
            self._tick_spears(world, step)
        except Exception:
            log.exception("animal tick failed")

    def _tick_spears(self, world: Any, dt: float) -> None:
        """Drive the thrown-spear registry, if this world has one.

        A spear is a projectile the combat loop fires at the things in *this*
        registry, and it has to be integrated every frame or it hangs in the
        air. World.tick is the natural owner and declares its own ``spears``
        guard - but a spear that is only ticked from there is silently inert on
        any world built before that line existed (an old save, a test stub, a
        checkout where the two halves have not met yet), and "inert" here means
        agents throw spears that never land and never come back. Driving it from
        the one subsystem that is already ticked, and that a spear is always
        aimed at, closes that hole.

        ``SpearRegistry.tick`` deduplicates on ``world.tick_count``, so when
        World.tick *does* also drive it the second call in the frame is a no-op
        rather than a second helping of gravity. Wrapped in its own try/except
        as well: this runs inside World._guarded("animals"), and a raise here
        would disable the wildlife for the rest of the session.
        """
        reg = getattr(world, "spears", None)
        fn = getattr(reg, "tick", None)
        if not callable(fn):
            return
        try:
            fn(world, dt)
        except Exception:
            log.exception("spear tick failed")

    def _tick_animal(self, world: Any, a: Animal, dt: float) -> None:
        a.age += dt
        a.t += dt
        ratio = min(1.0, abs(a.vx) / max(1.0, a.speed))
        a.anim_t += dt * (0.5 + 2.4 * ratio)
        if a.anim_t > 1.0e6:
            a.anim_t = math.fmod(a.anim_t, 1.0e6)
        if not a.alive:
            return                                    # reaped below

        # Patience runs out whatever it was doing.
        if a.state in (STATE_APPROACH, STATE_ATTACK) and a.age >= ANIMAL_LEAVE_SEC:
            self._enter(a, STATE_LEAVE)

        if a.state in (STATE_FLEE, STATE_LEAVE):
            self._exit_step(world, a, dt)
        elif a.state == STATE_ATTACK:
            self._attack_step(world, a, dt)
        else:
            self._approach_step(world, a, dt)

    def _enter(self, a: Animal, state: str, *, keep_feed: bool = False) -> None:
        if a.state != state:
            a.state = state
            a.t = 0.0
        if state in (STATE_FLEE, STATE_LEAVE):
            a.target_id = None
            if not keep_feed:
                a.feed_t = 0.0

    # -- states -----------------------------------------------------------
    def _approach_step(self, world: Any, a: Animal, dt: float) -> None:
        target = self._pick_target(world, a)
        if target is None:
            self._enter(a, STATE_LEAVE)               # nobody left to hunt
            return

        dx = _f(getattr(target, "x", a.x), a.x) - a.x
        if abs(dx) <= ANIMAL_REACH:
            a.facing = 1 if dx >= 0.0 else -1
            a.vx = 0.0
            self._stick(world, a)
            self._enter(a, STATE_ATTACK)
            return

        # Stand off to one side, chosen by id, so a pair on one victim does not
        # render as a single animal.
        side = 1.0 if (a.id % 2) else -1.0
        aim = _f(getattr(target, "x", a.x), a.x) + side * ANIMAL_REACH * 0.7
        self._walk(world, a, aim, dt, a.speed)

    def _attack_step(self, world: Any, a: Animal, dt: float) -> None:
        target = _agent_by_id(world, a.target_id)
        if (target is None or not getattr(target, "alive", False)
                or getattr(target, "taken", False)):
            a.target_id = None
            a.vx = 0.0
            self._enter(a, STATE_APPROACH)
            return

        dx = _f(getattr(target, "x", a.x), a.x) - a.x
        if abs(dx) > ANIMAL_REACH * REACH_HYSTERESIS:
            self._enter(a, STATE_APPROACH)
            return

        # Climbed out of reach mid-fight. Checked here as well as in
        # _pick_target for exactly the reason the ward is: an animal that
        # closed on him at ground level otherwise goes on biting after he has
        # gone up the ladder, and the one moment a tower is supposed to be
        # worth having is the one it would never cover. The target is dropped
        # rather than re-approached - approaching something it can never reach
        # is how you get a wolf parked at the foot of a tower for the rest of
        # the session.
        if _out_of_reach_above(a, target):
            a.target_id = None
            a.vx = 0.0
            self._enter(a, STATE_APPROACH)
            return

        # He made it back to his fire. Break off - the ward has to be checked
        # here as well as in `_pick_target`, or an animal that closed on him in
        # the open goes on biting inside the light, and the one moment the ward
        # is supposed to be worth watching is the one it would never cover.
        if _warded(target, self._ward):
            a.target_id = None
            a.vx = 0.0
            self._enter(a, STATE_APPROACH)
            return

        a.facing = 1 if dx >= 0.0 else -1
        a.vx = 0.0
        self._stick(world, a)

        killed = False
        try:
            killed = bool(target.hurt(a.dps * dt, CAUSE_MAULED))
        except Exception:
            log.debug("hurt() failed on agent %s", getattr(target, "id", "?"),
                      exc_info=True)

        if not killed and _scaled(target):
            # The bite landed and did nothing worth doing. Note that the damage
            # above was NOT skipped: Stickman.hurt's clamp is the only thing in
            # the game that knows what armour is worth, and a scale-wearer down
            # to their last few hit points can still be finished by a wolf -
            # which is why this branch sits under `not killed` rather than in
            # front of the bite.
            self._refuse_scale(world, a, target)
            return

        if killed:
            name = str(getattr(target, "name", "Someone"))
            _log(world, f"{name} was killed by a {a.kind}.")
            self._wave_took += 1
            # Sated. The killer stands over the body for FEED_SEC and then
            # goes, and the rest of the pack breaks off with it: an incursion
            # costs the colony at most one person. See the module docstring for
            # the measurements behind that.
            a.target_id = None
            self._enter(a, STATE_LEAVE, keep_feed=True)
            a.feed_t = FEED_SEC
            self._wave_withdraw(a)

    def _refuse_scale(self, world: Any, a: Animal, target: Any) -> None:
        """Give up on a dragonscale wearer, for good.

        Not a flee and not a leave: the animal is neither hurt nor frightened,
        it has simply found nothing it can eat. It goes back to APPROACH with
        the wearer struck off its list, so ``_pick_target`` hands it whoever is
        next - and if the wearer was the last person standing, _pick_target
        returns None and ``_approach_step`` walks it off the map. That fallback
        is why the refusal is a target filter rather than a damage exemption:
        the "everyone else is dead" case needs no special code at all.

        Rejected: adding a large ``FOCUS_PENALTY``-style score penalty in
        ``_pick_target`` instead. It reads the same until the wearer is the
        only one left, at which point the penalty loses to an empty field and
        the pack chews on them for the full 90 s anyway - which is the bear
        case in the module docstring, i.e. exactly the thing this fixes.
        """
        aid = _i(getattr(target, "id", -1), -1)
        if aid >= 0:
            a.scale_refused = aid
        if not self._scale_told:
            self._scale_told = True
            _log(world, _SCALE_LINES.get(a.kind, _SCALE_FALLBACK))
        a.target_id = None
        a.vx = 0.0
        self._enter(a, STATE_APPROACH)

    def _exit_step(self, world: Any, a: Animal, dt: float) -> None:
        """Flee or leave: head off the near side of the stage, then despawn.

        "The nearer screen edge" was one pixel from "the nearer world edge"
        until the world got four times wider than the frame. Left as the world
        rim it is dead time nobody sees: an animal breaking off at x=3000 walks
        3200 px to reach x=6399, ~73 s at a wolf's 44 px/s, holding one of
        ANIMAL_MAX_ALIVE slots and off-camera for all but the first second of
        it. It now leaves the way it came - OFFSTAGE px from the colony, i.e.
        just past the edge of frame - which is the same handful of seconds it
        took before the map grew.
        """
        if a.state == STATE_FLEE:
            a.fleeing = True
        if a.feed_t > 0.0:
            # Feeding over a fresh kill before it withdraws.
            a.feed_t = max(0.0, a.feed_t - dt)
            a.vx = 0.0
            self._stick(world, a)
            return
        centre = _colony_x(world)
        side = 1 if a.x >= centre else -1
        edge = _offstage_x(centre, side)
        if a.state == STATE_FLEE:
            hurt = _clamp(a.health_frac / max(1e-6, FLEE_HEALTH), 0.0, 1.0)
            scale = FLEE_SPEED_HURT + (FLEE_SPEED_FRESH - FLEE_SPEED_HURT) * hurt
        else:
            scale = 1.0
        speed = a.speed * scale
        self._walk(world, a, edge, dt, speed)
        # Two despawn tests, and the second one is not redundant. _offstage_x
        # clamps to the world, so a colony seated within OFFSTAGE of the map's
        # rim gets an exit target AT the rim - and then |a.x - centre| never
        # reaches OFFSTAGE and the animal would pace the last column forever.
        # The world-edge test is what closes that, and it is the only case where
        # the old rim rule is still the right one.
        if abs(a.x - centre) >= OFFSTAGE - EDGE_MARGIN:
            self.remove(a)
        elif a.x <= _EDGE_MIN + EDGE_MARGIN or a.x >= _EDGE_MAX - EDGE_MARGIN:
            self.remove(a)

    # -- targeting --------------------------------------------------------
    def _attackers_on(self, agent_id: int, exclude: int) -> int:
        n = 0
        for a in self.animals:
            if a.id == exclude or not a.hostile:
                continue
            if a.target_id is not None and int(a.target_id) == agent_id:
                n += 1
        return n

    def _pick_target(self, world: Any, a: Animal) -> Any:
        """Nearest stickman, heavily biased away from one already being mauled.

        Distance in pixels is the score; every animal already on a person adds
        FOCUS_PENALTY, and the target already held gets STICKY_BONUS off so the
        choice does not oscillate. Writes ``a.target_id`` and returns the agent.

        Two people are skipped outright rather than scored: a dragonscale wearer
        this animal has already tried (see ``_refuse_scale``) and anybody
        standing in the light of a burning hermit fire (see HERMIT_FIRE_WARD).
        Both are FILTERS rather than damage exemptions, and for the same reason:
        a filter degrades correctly when the protected person is the last one
        alive - ``_approach_step`` gets None and walks the animal off the map -
        where a penalty would lose to an empty field and the pack would chew on
        them anyway.
        """
        best: Any = None
        best_score = float("inf")
        for ag in _agents(world):
            if getattr(ag, "taken", False):
                continue
            aid = _i(getattr(ag, "id", -1), -1)
            if aid < 0:
                continue
            if a.scale_refused is not None and aid == int(a.scale_refused):
                continue          # already tried them; they were wearing scale
            if _warded(ag, self._ward):
                continue          # stood at a lit hermit fire; wolves will not
            if _out_of_reach_above(a, ag):
                continue          # up a tower; nothing here can climb
            score = abs(_f(getattr(ag, "x", 0.0)) - a.x)
            score += FOCUS_PENALTY * self._attackers_on(aid, a.id)
            if a.target_id is not None and aid == int(a.target_id):
                score -= STICKY_BONUS
            if score < best_score:
                best, best_score = ag, score
        a.target_id = None if best is None else _i(getattr(best, "id", 0), 0)
        return best

    # -- movement ---------------------------------------------------------
    def _stick(self, world: Any, a: Animal) -> None:
        """Pin the animal to the terrain surface, inside the world."""
        a.x = _clamp_x(a.x)
        a.y = _ground_y(world, a.x)

    def _walk(self, world: Any, a: Animal, tx: float, dt: float, speed: float) -> None:
        """One step toward *tx* along the ground. No gravity, ever.

        A step that would climb or drop faster than CLIMB_RATE is shortened
        instead, so steep terrain slows an animal down rather than snapping it
        up a cliff face - and MIN_STEP_FRAC keeps a sheer wall from stalling it
        on the spot forever.
        """
        target = _clamp_x(tx)
        dx = target - a.x
        if dt <= 0.0 or abs(dx) < 0.5:
            a.vx = 0.0
            self._stick(world, a)
            return

        d = 1.0 if dx > 0.0 else -1.0
        a.facing = 1 if d > 0.0 else -1
        step = min(abs(dx), max(0.0, _f(speed, 0.0)) * dt)
        if step <= 0.0:
            a.vx = 0.0
            self._stick(world, a)
            return

        gy0 = _ground_y(world, a.x)
        nx = _clamp_x(a.x + d * step)
        gy1 = _ground_y(world, nx)
        rise = abs(gy1 - gy0)
        budget = CLIMB_RATE * dt
        if rise > budget:
            frac = max(MIN_STEP_FRAC, budget / rise)
            nx = _clamp_x(a.x + d * step * frac)
            gy1 = _ground_y(world, nx)

        moved = nx - a.x
        a.x = nx
        a.y = gy1
        a.vx = moved / dt

    # -- bookkeeping ------------------------------------------------------
    def _reap(self, world: Any) -> None:
        """Clear carcasses: hides to the stockpile, a line to the chronicle."""
        fell = 0
        for a in list(self.animals):
            if a.alive:
                continue
            self.remove(a)
            self._wave_killed += 1
            fell += 1
            hides = a.hides
            if hides > 0:
                _give(world, RES_HIDE, hides)
                _log(world, f"The {a.kind} is brought down. The camp takes "
                            f"{hides} hide{'s' if hides != 1 else ''}.")
            else:
                _log(world, f"The {a.kind} is brought down.")
        if fell:
            self._check_nerve()

    def _wave_withdraw(self, killer: Animal) -> None:
        """The pack has what it came for: everyone else breaks off too.

        This is the cap on what one incursion can cost. Measured against a
        colony of four that cannot fight back or run (behaviour.py does not
        score animals yet), a wolf pack allowed to keep hunting took 3 of 4;
        with the wave withdrawing on the first kill it takes one, which is the
        "a real event you can lose someone to" the brief asks for. They are not
        frightened, so this is LEAVE and not FLEE - ``fleeing`` stays false and
        the sign-off line reads "withdraw, fed".
        """
        for a in self.animals:
            if a is killer or not a.alive:
                continue
            if a.state in (STATE_APPROACH, STATE_ATTACK):
                self._enter(a, STATE_LEAVE)

    def _check_nerve(self) -> None:
        """A wave down half its number breaks off and runs.

        Without it, killing one of three wolves only removes a third of the
        incoming damage, and a spear is worth a third of a fight. With it, the
        first kill routinely ends the incursion - which is what makes arming
        the colony the thing that changes the outcome rather than the pace.
        """
        if self._wave_size <= 1 or self._wave_killed * 2 < self._wave_size:
            return
        for a in self.animals:
            if a.alive and a.state in (STATE_APPROACH, STATE_ATTACK):
                a.fleeing = True
                self._enter(a, STATE_FLEE)

    def _wave_end(self, world: Any) -> None:
        """Sign the incursion off once the last of it has gone."""
        if not self._wave_kind or self.animals:
            return
        lines = _WAVE_LINES.get(self._wave_kind, _WAVE_FALLBACK)
        if self._wave_killed >= max(1, self._wave_size):
            key = ""            # every one of them died; _reap already said so
        elif self._wave_killed:
            key = "driven"
        elif self._wave_took:
            key = "took"
        else:
            key = "gone"
        if key:
            _log(world, lines.get(key, ""))
        self._reset_wave()

    # ----------------------------------------------------------- persist --
    def to_dict(self) -> dict[str, Any]:
        return {
            "animals": [a.to_dict() for a in self.animals],
            "next_id": int(self.next_id),
            "next_spawn": float(self.next_spawn),
            "t": float(self.t),
            "wave_kind": str(self._wave_kind),
            "wave_size": int(self._wave_size),
            "wave_killed": int(self._wave_killed),
            "wave_took": int(self._wave_took),
            "scale_told": bool(self._scale_told),
        }

    @classmethod
    def from_dict(cls, d: Any, seed: int | None = None) -> "AnimalRegistry":
        """Defensive load: an unreadable record is dropped, not fatal.

        ``seed`` is optional so a caller that *does* hold the world seed can
        thread it through; World.from_dict cannot, so without one we derive a
        stable seed from the record itself - see :func:`_record_seed` for what
        went wrong when this was a bare ``cls()``.
        """
        reg = cls(seed=_record_seed(d) if seed is None else int(seed))
        if not isinstance(d, dict):
            return reg
        raw = d.get("animals")
        if isinstance(raw, (list, tuple)):
            for item in raw:
                try:
                    a = Animal.from_dict(item)
                except Exception:
                    log.debug("skipped an unreadable animal record", exc_info=True)
                    continue
                if a.alive:
                    reg.animals.append(a)
        reg.next_id = max(1, _i(d.get("next_id"), 1))
        for a in reg.animals:
            reg.next_id = max(reg.next_id, a.id + 1)
        # A save from a world that had already used up its interval must not
        # come back with a wave pending on the first frame.
        reg.next_spawn = max(5.0, _f(d.get("next_spawn"), reg.next_spawn))
        reg.t = max(0.0, _f(d.get("t"), 0.0))
        wk = d.get("wave_kind")
        reg._wave_kind = wk if wk in ANIMAL_KINDS else ""
        reg._wave_size = max(0, _i(d.get("wave_size"), len(reg.animals)))
        reg._wave_killed = max(0, _i(d.get("wave_killed"), 0))
        reg._wave_took = max(0, _i(d.get("wave_took"), 0))
        reg._scale_told = _b(d.get("scale_told"), False)
        if not reg.animals:
            reg._reset_wave()
        return reg


# ------------------------------------------------------ module-level API --
def registry_of(world: Any) -> AnimalRegistry | None:
    """The world's animal registry, or None if it has not got one."""
    reg = getattr(world, "animals", None)
    return reg if isinstance(reg, AnimalRegistry) else None


def danger_at(world: Any, x: float, radius: float = DANGER_RADIUS) -> bool:
    """Free-function form of :meth:`AnimalRegistry.danger_at`.

    Safe on a world with no wildlife at all, so behaviour.py can call it
    unconditionally when scoring flight.
    """
    reg = registry_of(world)
    return False if reg is None else reg.danger_at(x, radius)


def threats(world: Any) -> list[Animal]:
    """Free-function form of :meth:`AnimalRegistry.threats`."""
    reg = registry_of(world)
    return [] if reg is None else reg.threats(world)


def nearest_threat(world: Any, x: float,
                   radius: float = DANGER_RADIUS) -> Animal | None:
    """Nearest hunting animal within *radius* of *x*, or None."""
    reg = registry_of(world)
    return None if reg is None else reg.threat_at(x, radius)


# ============================================================== smoke test ==
if __name__ == "__main__":   # pragma: no cover - headless balance harness
    import statistics
    import sys
    import time

    from ..constants import (
        ARMOUR_LEATHER, SPEAR_COOLDOWN, SPEAR_DAMAGE, SPEAR_REACH, WEAPON_SPEAR,
    )
    from . import entities as _ents
    from .world import World

    DT = 1.0 / 30.0

    # ---------------------------------------------------------- unit pass --
    class _Ramp:
        """Flat, then a 260 px cliff at x=640 - the nastiest ground we make."""

        def ground_y(self, x: float) -> float:
            return 420.0 if float(x) < 640.0 else 680.0

        def slope(self, x: float) -> float:
            return 0.0

    class _StubAgent:
        def __init__(self, aid, x):
            self.id, self.x, self.name = aid, x, f"A{aid}"
            self.alive, self.taken, self.hp = True, False, 100.0

        def hurt(self, amount, cause=""):
            self.hp -= amount
            if self.hp <= 0.0:
                self.alive = False
                return True
            return False

    class _StubWorld:
        def __init__(self):
            self.terrain = _Ramp()
            self.stockpile = {}
            self.lines = []
            self.pyrng = random.Random(3)
            self.is_night = True
            self._agents = [_StubAgent(1, 200.0), _StubAgent(2, 1100.0)]

        class _Pop:
            def __init__(self, outer):
                self.o = outer

            def alive_agents(self):
                return [a for a in self.o._agents if a.alive]

        @property
        def population(self):
            return _StubWorld._Pop(self)

        def agent_by_id(self, aid):
            return next((a for a in self._agents if a.id == aid), None)

        def log_event(self, text):
            self.lines.append(text)

        def give(self, res, qty):
            self.stockpile[res] = self.stockpile.get(res, 0) + qty

    sw = _StubWorld()
    reg = AnimalRegistry(seed=1)
    pack = reg.spawn_wave(sw, kind=ANIMAL_WOLF, edge=-1)
    assert len(pack) == 3, pack
    assert any("ridge" in ln or "treeline" in ln or "howling" in ln
               for ln in sw.lines), sw.lines
    assert reg.count() == 3

    for _ in range(int(30 * 120)):
        reg.tick(sw, DT)
        for an in reg.all():
            assert _EDGE_MIN <= an.x <= _EDGE_MAX, an.summary()
            assert abs(an.y - sw.terrain.ground_y(an.x)) < 1e-6, an.summary()
        if not reg.all():
            break
    print("unit: wolves crossed the cliff; survivors", reg.count(),
          "| stub deaths", sum(1 for a in sw._agents if not a.alive))
    assert any("was killed by a wolf." in ln for ln in sw.lines), sw.lines[-6:]

    # danger_at / threats / round-trip
    sw2 = _StubWorld()
    reg2 = AnimalRegistry(seed=2)
    born = reg2.spawn_wave(sw2, kind=ANIMAL_BEAR, edge=1)
    assert len(born) == 1 and born[0].kind == ANIMAL_BEAR
    bear = born[0]
    bear.x = 500.0
    assert reg2.danger_at(500.0) and not reg2.danger_at(50.0)
    assert danger_at(sw2, 500.0) is False        # sw2 has no .animals attribute
    sw2.animals = reg2
    assert danger_at(sw2, 500.0) and threats(sw2) == [bear]
    assert nearest_threat(sw2, 520.0) is bear

    blob = reg2.to_dict()
    clone = AnimalRegistry.from_dict(blob)
    assert clone.to_dict()["animals"] == blob["animals"], clone.to_dict()
    # The save file is JSON, so every value here has to survive a text round
    # trip - not just a dict copy.
    import json as _json
    assert AnimalRegistry.from_dict(
        _json.loads(_json.dumps(blob))).to_dict() == clone.to_dict()
    assert AnimalRegistry.from_dict({"animals": [None, 7, {"kind": "dragon"}]}) is not None
    assert AnimalRegistry.from_dict("nonsense").count() == 0
    assert Animal.from_dict(None).kind == ANIMAL_WOLF

    # Hurt it back: the bear flees under 30% and drops hides when it dies.
    assert bear.hurt(float("nan")) is False
    while bear.health_frac > FLEE_HEALTH:
        bear.hurt(SPEAR_DAMAGE)
    assert bear.state == STATE_FLEE and bear.fleeing, bear.summary()
    assert bear.hurt(999.0) is True and not bear.alive
    reg2.tick(sw2, DT)
    assert sw2.stockpile.get(RES_HIDE) == 3, sw2.stockpile
    assert any("brought down" in ln for ln in sw2.lines), sw2.lines
    # A wave wiped out needs no sign-off line - "brought down" already said it.
    assert not any("driven off" in ln for ln in sw2.lines), sw2.lines
    print("unit: persistence, danger_at, flee-at-30%, hides ->",
          sw2.stockpile.get(RES_HIDE))

    # Nerve: kill two of three wolves and the last one runs.
    sw3 = _StubWorld()
    reg3 = AnimalRegistry(seed=4)
    trio = reg3.spawn_wave(sw3, kind=ANIMAL_WOLF, edge=-1)
    assert len(trio) == 3
    for _ in range(int(30 * 2.0)):        # still crossing, nobody bitten yet
        reg3.tick(sw3, DT)
    assert all(a.state == STATE_APPROACH for a in trio), [a.summary() for a in trio]
    trio[0].hurt(999.0)
    trio[1].hurt(999.0)
    reg3.tick(sw3, DT)
    assert trio[2].state == STATE_FLEE and trio[2].fleeing, trio[2].summary()
    for _ in range(int(30 * 60)):
        reg3.tick(sw3, DT)
        if not reg3.all():
            break
    assert any("driven off" in ln for ln in sw3.lines), sw3.lines
    print("unit: pack nerve broke after 2 of 3 fell ->",
          [ln for ln in sw3.lines if "driven" in ln][0])

    # Sated: an animal that kills leaves instead of picking a second victim.
    sw4 = _StubWorld()
    reg4 = AnimalRegistry(seed=5)
    assert len(reg4.spawn_wave(sw4, kind=ANIMAL_BOAR, edge=-1)) == 1  # alone
    for _ in range(int(30 * ANIMAL_LEAVE_SEC)):
        reg4.tick(sw4, DT)
        if not reg4.all():
            break
    dead_stubs = sum(1 for a in sw4._agents if not a.alive)
    assert dead_stubs <= 1, dead_stubs
    print("unit: one boar, no escape, no fighting back -> stub deaths",
          dead_stubs)

    # -- dragonscale: the bear case from the module docstring ---------------
    # 90 s of bear at the clamped 10% is 135 damage against 100 health, so the
    # clamp on its own is NOT enough and this is the assertion that says so.
    # Two stubs: one in scale at 200, one bare at 1100. The bear has to end up
    # on the bare one, the wearer has to be untouched after the first bite, and
    # the line has to be said exactly once.
    class _ScaleWorld(_StubWorld):
        def __init__(self, *, lone: bool = False):
            super().__init__()
            self._agents[0].relic = RELIC_SCALE
            self._agents[0].armour = 1.0
            if lone:
                self._agents = self._agents[:1]

    sw5 = _ScaleWorld()
    reg5 = AnimalRegistry(seed=7)
    bear5 = reg5.spawn_wave(sw5, kind=ANIMAL_BEAR, edge=-1)[0]
    bear5.x = 210.0                       # already on top of the wearer
    for _ in range(int(30 * ANIMAL_LEAVE_SEC)):
        reg5.tick(sw5, DT)
        if not reg5.all():
            break
    wearer, bystander = sw5._agents[0], sw5._agents[1]
    assert wearer.alive, "dragonscale wearer was eaten"
    # One bite got through the clamp; nothing like a mauling did.
    assert 100.0 - wearer.hp < 2.0, wearer.hp
    assert sum(1 for ln in sw5.lines if "nothing but scale" in ln) == 1, sw5.lines
    assert bear5.scale_refused == wearer.id, bear5.scale_refused
    print("unit: bear off the scale-wearer | wearer hp", f"{wearer.hp:.1f}",
          "| bystander hp", f"{bystander.hp:.1f}")

    # ...and with nobody else to eat, it walks off the map instead of chewing.
    sw6 = _ScaleWorld(lone=True)
    reg6 = AnimalRegistry(seed=8)
    bear6 = reg6.spawn_wave(sw6, kind=ANIMAL_BEAR, edge=-1)[0]
    bear6.x = 210.0
    for _ in range(int(30 * ANIMAL_LEAVE_SEC)):
        reg6.tick(sw6, DT)
        if not reg6.all():
            break
    assert sw6._agents[0].alive and reg6.count() == 0, (
        sw6._agents[0].hp, reg6.count())
    print("unit: lone scale-wearer, bear gave up and left | hp",
          f"{sw6._agents[0].hp:.1f}")

    # -- the BFG's line of fire uses the ordinary damage path ---------------
    # Lane 3's beam sweeps animals with the same Animal.hurt a spear uses, so
    # the ONLY thing this module has to guarantee is that an absurd amount of
    # damage still reaps like any other kill: hides to the stockpile once, the
    # wave counter moved, the nerve check run. Nothing here special-cases the
    # weapon and nothing should - a beam kill that paid different hides from a
    # spear kill is the "hides and kill counts stay honest" bug.
    from ..constants import BFG_DAMAGE
    sw7 = _StubWorld()
    reg7 = AnimalRegistry(seed=9)
    beamed = reg7.spawn_wave(sw7, kind=ANIMAL_WOLF, edge=-1)
    assert len(beamed) == 3
    for victim in beamed[:2]:             # two of three, in one "shot"
        assert victim.hurt(BFG_DAMAGE) is True
    reg7.tick(sw7, DT)
    assert sw7.stockpile.get(RES_HIDE, 0) == 2, sw7.stockpile
    assert reg7._wave_killed == 2, reg7._wave_killed
    assert reg7.count() == 1 and beamed[2].state == STATE_FLEE, beamed[2].summary()
    print("unit: BFG-sized hit -> hides", sw7.stockpile.get(RES_HIDE, 0),
          "| wave_killed", reg7._wave_killed,
          "| survivor", beamed[2].state)

    # ------------------------------------------------------ live scenarios --
    def run(seed: int, seconds: float, *, gear: str,
            forced: str | None = None, scene: str = SCENE_CLEAR) -> dict:
        """One colony, one clock. Returns what happened to it.

        ``gear`` is "none" | "spear" | "full". "spear" is the honest state of
        the colony at its *first* incursion: leather needs hides and hides need
        a dead animal, so nobody can be wearing armour until they have already
        won one of these.
        """
        armed = gear in ("spear", "full")
        _ents.clear_death_listeners()
        deaths: list[str] = []
        _ents.on_death(lambda s, cause: deaths.append(cause or "?"))

        w = World(seed=seed, scene=scene)
        reg = AnimalRegistry(seed=seed)
        w.animals = reg                       # world.py does not own this yet
        waves = [0]
        _orig = reg.spawn_wave

        def _counted(world, kind=None, edge=None):
            got = _orig(world, kind, edge)
            if got:
                waves[0] += 1
            return got
        reg.spawn_wave = _counted             # type: ignore[assignment]

        if forced:
            reg.spawn_wave(w, kind=forced)

        killed = [0]
        worst_hp = [100.0]                # lowest health any survivor reached
        arm_t = 0.0
        n = int(seconds / DT)
        for _ in range(n):
            for ag in w.population.alive_agents():
                worst_hp[0] = min(worst_hp[0], float(ag.health))
            if armed:
                # Stand-in for the Fight action that will live in actions.py:
                # an armed stickman spears whatever comes inside SPEAR_REACH.
                arm_t += DT
                if arm_t >= 1.0:
                    arm_t = 0.0
                    for ag in w.population.alive_agents():
                        ag.weapon = WEAPON_SPEAR
                        ag.armour = ARMOUR_LEATHER if gear == "full" else 0.0
                for ag in w.population.alive_agents():
                    ag.attack_cd = max(0.0, ag.attack_cd - DT)
                    if ag.attack_cd > 0.0:
                        continue
                    foe = reg.nearest(ag.x)
                    if foe is not None and abs(foe.x - ag.x) <= SPEAR_REACH:
                        ag.attack_cd = SPEAR_COOLDOWN
                        if foe.hurt(SPEAR_DAMAGE):
                            killed[0] += 1
                    ag.heal(DT)
            w.tick(DT)
            reg.tick(w, DT)

        _ents.clear_death_listeners()
        return {
            "waves": waves[0],
            "mauled": sum(1 for c in deaths if c == CAUSE_MAULED),
            "other": sum(1 for c in deaths if c != CAUSE_MAULED),
            "animals_killed": killed[0],
            "worst_hp": worst_hp[0],
            "hides": w.stockpile.get(RES_HIDE, 0),
            "alive": w.population.alive_count(),
        }

    def report(label: str, rows: list[dict]) -> None:
        m = [r["mauled"] for r in rows]
        o = [r["other"] for r in rows]
        k = [r["animals_killed"] for r in rows]
        w = [r["waves"] for r in rows]
        lost = sum(1 for v in m if v)
        print(f"  {label:<22} mauled {statistics.mean(m):.2f}/run "
              f"(max {max(m)}) | lost someone in {lost}/{len(rows)} runs | "
              f"animals killed {statistics.mean(k):.2f}/{statistics.mean(w):.2f} "
              f"waves | worst hp {statistics.mean([r['worst_hp'] for r in rows]):.0f}"
              f" | other deaths {statistics.mean(o):.2f}")

    seeds = [int(s) for s in (sys.argv[1:] or range(101, 111))]
    t0 = time.time()

    for kind, label in ((ANIMAL_WOLF, "wolf pack (3)"), (ANIMAL_BEAR, "bear"),
                        (ANIMAL_BOAR, "boar")):
        print(f"\n300 s, one forced {label} + whatever the timer brings, clear:")
        for gear, name in (("none", "no weapons"), ("spear", "spears only"),
                           ("full", "spears + leather")):
            report(name, [run(s, 300.0, gear=gear, forced=kind) for s in seeds])

    print("\n900 s, natural spawn timers only, night storm:")
    for gear, name in (("none", "no weapons"), ("full", "spears + leather")):
        nat = [run(s, 900.0, gear=gear, scene="night_storm") for s in seeds[:5]]
        report(name, nat)
        print(f"    waves/900 s {statistics.mean([r['waves'] for r in nat]):.2f}"
              f" | hides banked {statistics.mean([r['hides'] for r in nat]):.2f}"
              f" | survivors {statistics.mean([r['alive'] for r in nat]):.2f}")

    print(f"\nharness took {time.time() - t0:.1f}s")
