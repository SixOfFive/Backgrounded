"""Combat and crafting - the *agent* half of the animal threat.

Four new :class:`~.actions.Action` kinds, registered into ``actions._HANDLERS``
at import so a save taken mid-fight or mid-craft rehydrates correctly:

==============  ===========================================================
CraftSpear      6 s at the stockpile, spends ``SPEAR_COST``, arms the agent
CraftArmour     8 s at the stockpile, spends ``ARMOUR_COST`` (needs hides)
FightAnimal     close, then swing on ``SPEAR_COOLDOWN``. Armed agents only.
FleeAnimal      run to a safe distance. What an *unarmed* agent does.
==============  ===========================================================

No pygame. Everything an action touches on ``world`` goes through the accessor
layer in :mod:`.actions`, plus the small animal layer at the top of this file -
``sim/animals.py`` is a sibling module written independently, so every read of
an animal is duck-typed and every write is fail-soft. A missing animal
subsystem degrades to "there are no animals", never to an exception on the
per-frame path.

Balance
-------
Measured against a real :class:`~.world.World` driving the real
:mod:`.animals` registry, with :func:`score_combat` merged into
``behavior.score_actions`` and :func:`under_attack` into
``emergency_override``. 8 seeds x 900 s, same seeds both columns:

===========================  ===========  ==========
                             combat off   combat on
===========================  ===========  ==========
deaths per run                     8.50        5.00
  of which mauled              21 total     0 total
animals killed per run             0.00        2.88
armed / armoured at the end     0 / 0 of 4   3.1 / 1.2 of 4
structures built per run           3.88        4.00
===========================  ===========  ==========

So animals are genuinely lethal to a colony that cannot answer them, an armed
one wins, and arming costs no build throughput. The residual risk sits almost
entirely in the *first* incursion, before anybody owns a spear: across 20 x
300 s runs (roughly one incursion each, usually arriving before the colony has
armed) there was 1 mauling death, 0.85 animals killed per run.

If incursions should bite harder, the two knobs are ``BREAK_OFF_HEALTH``
(higher = agents run sooner) and the 0.55 base for CraftSpear in
:func:`score_combat` (lower = the colony arms later).

Deaths from every *other* cause are untouched by this module. Verified by
running one seed set with combat wired in but animal spawning suppressed:
lightning/fall/fire came out 28/11/7, against 28/10/8 with animals live. Any
apparent shift against a bare baseline is RNG-stream drift - behaviour draws a
tiebreak per candidate action, and this adds four candidates.

Why the numbers land there:

* A spear is 26 damage on a 1.1 s cooldown - 23.6 dps, so a wolf (58 hp) dies
  in three swings and a bear (140 hp) in six. An *unarmed* agent has no way to
  hurt anything, which is the whole reason FleeAnimal exists.
* Leather absorbs 45%, so an armoured agent survives 41 s of one wolf's
  attention instead of 22 s. That is the difference between a pack being a
  funeral and being a fight, and it is why CraftArmour scores at all.
* An agent that drops below ``BREAK_OFF_HEALTH`` stops fighting and runs. A
  colony that fights to the last hit point loses somebody every incursion;
  this is the single knob that turns "a real event you can lose someone to"
  into "usually survivable".

Wiring (the two things this module cannot do for itself, since it may only
write this file):

* ``behavior.score_actions`` should merge :func:`score_combat` into its score
  dict, and ``behavior._MAKERS`` should route the four kinds through
  :func:`make_combat_action` - :func:`install_makers` does the second half if
  the caller passes it the behavior module.
* ``behavior.emergency_override`` should return True when an animal is inside
  ``FLEE_RADIUS``, or an agent will finish chopping its tree while a wolf eats
  it. :func:`under_attack` is the predicate to use.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable

from ..constants import (
    ANIMAL_STATS,
    ARMOUR_COST,
    ARMOUR_LEATHER,
    MAX_HEALTH,
    RES_HIDE,
    SPEAR_COOLDOWN,
    SPEAR_COST,
    SPEAR_DAMAGE,
    SPEAR_REACH,
    WALK_SPEED,
    WEAPON_NONE,
    WEAPON_SPEAR,
)
from . import actions as _actions
from .actions import (
    REACH,
    Action,
    _adjust,
    _carry_add,
    _clamp01,
    _clamp_x,
    _deposit_step,
    _face,
    _halt,
    chronicle,
    colony_center,
    emit_speech,
    make_action,
    nearest_structure,
    step_toward,
    stock_add,
    stock_qty,
    stock_take,
    world_now,
)

log = logging.getLogger(__name__)

__all__ = [
    "COMBAT_ACTION_KINDS",
    "score_combat",
    "make_combat_action",
    "install_makers",
    "under_attack",
    # animal accessor layer, reusable by behavior.py / events.py
    "animals_of", "living_animals", "animal_alive", "animal_by_id",
    "animal_id", "animal_kind", "animal_x", "animal_leaving",
    "nearest_animal", "hurt_animal", "claim_hides", "note_animal_seen",
    "sighting_freshness",
]

COMBAT_ACTION_KINDS: tuple[str, ...] = (
    "CraftSpear", "CraftArmour", "FightAnimal", "FleeAnimal",
)

# ------------------------------------------------------------------ tuning --
CRAFT_SPEAR_TIME = 6.0
CRAFT_ARMOUR_TIME = 8.0

#: Inside this, an unarmed agent stops whatever it was doing and runs.
FLEE_RADIUS = 110.0
#: Inside this, an armed agent will go and meet the animal.
FIGHT_RADIUS = 180.0
#: An animal that is *already chewing on somebody* is worth crossing the colony
#: for. Without this a spearman stands 200 px away gathering wood while a wolf
#: kills his neighbour. Half the world wide on purpose: animals single out
#: whoever has wandered furthest from the group, and at 340 px the armed
#: majority measured 700-900 px away from every wolf that took someone, so no
#: rescue ever triggered. A 640 px run costs ~13 s at CHASE_SPEED, inside
#: CHASE_GIVE_UP.
#: Reduced from 640px. A rescue that long crosses cliff faces, and every ledge
#: step carries a slip chance: wiring combat in took fall deaths from 0 to 12
#: in a forced-incursion sweep. Mauling deaths dropped far more, so it was a
#: net win, but it was trading one cause of death for another. Defend the camp,
#: do not chase across the map.
RESCUE_RADIUS = 260.0

#: A wounded animal breaking off, close enough for one parting thrust. Animals
#: flee once hurt, so without this almost every fight ends with the beast
#: limping away: measured 0.00 hides in the stockpile across six 300 s runs,
#: which means leather armour never got made once. Matches the handler's
#: give-up-on-a-leaver distance so the two cannot disagree.
PARTING_REACH = SPEAR_REACH * 2.0
#: How close an animal has to be to a colonist to count as mauling them.
MAUL_REACH = 34.0
#: How far a flee has to get before it counts as escaped.
SAFE_DIST = 210.0

#: Health fraction below which an agent breaks off and runs instead of trading
#: hits. See the module docstring - this is the mortality knob.
BREAK_OFF_HEALTH = 0.35

#: Seconds a sighting stays "recent" for the crafting urgency ramp.
SIGHTING_FRESH_SEC = 240.0
#: Sightings closer together than this are the same incursion, not a new one.
SIGHTING_EDGE_SEC = 30.0

CHASE_GIVE_UP = 45.0        # s spent closing without landing a swing
FIGHT_TIMEOUT = 75.0        # s of one continuous engagement
CRAFT_WALK_TIMEOUT = 75.0   # s walking to the stockpile before giving up
FLEE_TIMEOUT = 20.0

CHASE_SPEED = WALK_SPEED * 1.45
FLEE_SPEED = WALK_SPEED * 2.2

_HEALTH_EPS = 1e-6


# ===========================================================================
#  Animal accessor layer
#
#  sim/animals.py is written independently of this file, so nothing below
#  assumes a class, a registry type or a method name. Every read tolerates the
#  subsystem being absent, half-built, or shaped differently than expected.
# ===========================================================================
def animals_of(world: Any) -> list[Any]:
    """Every animal object the world knows about, living or not."""
    for name in ("animals", "fauna", "wildlife", "beasts"):
        v = getattr(world, name, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            return list(v)
        if isinstance(v, dict):
            try:
                return list(v.values())
            except Exception:
                continue
        for meth in ("all", "alive", "values", "items"):
            fn = getattr(v, meth, None)
            if not callable(fn):
                continue
            try:
                out = list(fn())
            except Exception:
                continue
            if out and isinstance(out[0], tuple):
                out = [o[-1] for o in out]
            return out
        if hasattr(v, "__iter__"):
            try:
                return list(v)
            except Exception:
                continue
    return []


def animal_alive(animal: Any) -> bool:
    if animal is None:
        return False
    if getattr(animal, "dead", False) or getattr(animal, "removed", False):
        return False
    if not getattr(animal, "alive", True):
        return False
    if getattr(animal, "gone", False):
        return False
    for attr in ("health", "hp"):
        v = getattr(animal, attr, None)
        if isinstance(v, (int, float)):
            return float(v) > _HEALTH_EPS
    return True


def living_animals(world: Any) -> list[Any]:
    return [a for a in animals_of(world) if animal_alive(a)]


def animal_id(animal: Any) -> int | None:
    v = getattr(animal, "id", None)
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return int(v)
    return None


def animal_kind(animal: Any) -> str:
    k = getattr(animal, "kind", None) or getattr(animal, "species", None) \
        or getattr(animal, "type", None) or ""
    return str(k).lower()


def animal_x(animal: Any, default: float = 0.0) -> float:
    try:
        v = float(getattr(animal, "x", default))
    except (TypeError, ValueError):
        return float(default)
    return v if math.isfinite(v) else float(default)


def animal_leaving(animal: Any) -> bool:
    """True when the animal has given up and is walking off the map.

    Chasing one of those across the world is how a hunter ends up 900 px from
    the colony with a build queue stalled behind it.
    """
    if getattr(animal, "leaving", False) or getattr(animal, "fleeing", False):
        return True
    for attr in ("state", "mode", "phase", "behaviour", "behavior"):
        v = getattr(animal, attr, None)
        if isinstance(v, str) and v.lower() in (
                "leave", "leaving", "flee", "fleeing", "depart", "departing"):
            return True
    return False


def animal_by_id(world: Any, aid: Any) -> Any | None:
    if aid is None:
        return None
    src = getattr(world, "animals", None)
    fn = getattr(src, "get", None)
    if callable(fn) and not isinstance(src, (list, tuple)):
        try:
            got = fn(aid)
            if got is not None:
                return got
        except Exception:
            pass
    try:
        want = int(aid)
    except (TypeError, ValueError):
        return None
    for a in animals_of(world):
        if animal_id(a) == want:
            return a
    return None


def nearest_animal(
    world: Any,
    x: float,
    *,
    max_dist: float = float("inf"),
    skip_leaving: bool = False,
) -> Any | None:
    """Closest living animal to world-x `x`, or None."""
    try:
        ax = float(x)
    except (TypeError, ValueError):
        return None
    best = None
    best_d = float("inf")
    for a in animals_of(world):
        if not animal_alive(a):
            continue
        if skip_leaving and animal_leaving(a):
            continue
        d = abs(animal_x(a, ax) - ax)
        if d < best_d:
            best_d, best = d, a
    if best is not None and best_d <= max_dist:
        return best
    return None


def _mark_animal_dead(world: Any, animal: Any) -> None:
    """Best-effort "this one is finished" so it stops being a threat.

    The animals module owns corpses and removal; all this guarantees is that
    :func:`animal_alive` agrees with the killing blow even if the animal has no
    death handling of its own yet.
    """
    for attr, val in (("health", 0.0), ("hp", 0.0), ("alive", False)):
        if hasattr(animal, attr):
            try:
                setattr(animal, attr, val)
            except Exception:
                pass
    if not hasattr(animal, "alive") and not hasattr(animal, "health") \
            and not hasattr(animal, "hp"):
        try:
            animal.alive = False
        except Exception:
            pass
    src = getattr(world, "animals", None)
    for name in ("kill", "on_killed", "died"):
        fn = getattr(src, name, None)
        if callable(fn):
            try:
                fn(animal)
            except Exception:
                log.debug("animal registry %s hook failed", name, exc_info=True)
            return


def hurt_animal(world: Any, animal: Any, amount: float, by: Any = None) -> bool:
    """Deal `amount` damage. Returns True if this blow killed it. Never raises."""
    try:
        if animal is None or float(amount) <= 0.0 or not animal_alive(animal):
            return False
    except (TypeError, ValueError):
        return False
    dmg = float(amount)
    for name in ("hurt", "damage", "take_damage", "wound", "hit"):
        fn = getattr(animal, name, None)
        if not callable(fn):
            continue
        out: Any = None
        try:
            out = fn(dmg)
        except TypeError:
            try:
                out = fn(dmg, by)
            except Exception:
                log.debug("animal.%s failed", name, exc_info=True)
                continue
        except Exception:
            log.debug("animal.%s failed", name, exc_info=True)
            continue
        killed = bool(out) if isinstance(out, bool) else False
        if killed or not animal_alive(animal):
            _mark_animal_dead(world, animal)
            return True
        return False
    # No damage method at all: take it off whatever health field exists.
    for attr in ("health", "hp"):
        v = getattr(animal, attr, None)
        if not isinstance(v, (int, float)):
            continue
        left = float(v) - dmg
        try:
            setattr(animal, attr, max(0.0, left))
        except Exception:
            break
        if left <= _HEALTH_EPS:
            _mark_animal_dead(world, animal)
            return True
        return False
    _mark_animal_dead(world, animal)
    return True


#: Flags an animals module might set when it drops its own loot. Checked so a
#: kill never pays twice, whichever side of the boundary does the skinning.
_HIDE_CLAIMED_FLAGS = (
    "hides_taken", "hides_claimed", "hides_dropped", "looted", "skinned",
)


def _registry_reaps(world: Any) -> bool:
    """True when the animals subsystem clears its own carcasses.

    ``animals.AnimalRegistry`` sweeps the dead in its own ``_reap`` pass and
    puts the hides in the stockpile there, which is the right place for it -
    the animal knows what it drops. Skinning the kill here as well would pay
    the colony twice for one wolf, and two wolves is a whole suit of armour.
    A bare ``world.animals`` list has no owner, so the killer takes the hide.
    """
    src = getattr(world, "animals", None)
    if src is None or isinstance(src, (list, tuple, dict)):
        return False
    return any(callable(getattr(src, n, None))
               for n in ("_reap", "reap", "remove"))


def claim_hides(world: Any, animal: Any) -> int:
    """Take the hides off a fresh kill, once. Returns how many.

    Returns 0 whenever the animals subsystem does its own reaping - see
    :func:`_registry_reaps`.
    """
    if animal is None or _registry_reaps(world):
        return 0
    for flag in _HIDE_CLAIMED_FLAGS:
        if getattr(animal, flag, False):
            return 0
    n = getattr(animal, "hides", None)
    if not isinstance(n, (int, float)):
        stats = ANIMAL_STATS.get(animal_kind(animal))
        n = stats[3] if isinstance(stats, (list, tuple)) and len(stats) > 3 else 1
    try:
        count = max(0, int(n))
    except (TypeError, ValueError):
        count = 1
    try:
        animal.hides_taken = True
    except Exception:
        pass
    return count


# ---------------------------------------------------------------- sightings --
def _stats(world: Any) -> dict[str, Any]:
    s = getattr(world, "stats", None)
    if isinstance(s, dict):
        return s
    s = {}
    try:
        setattr(world, "stats", s)
    except Exception:
        return {}
    return s


def note_animal_seen(world: Any) -> None:
    """Record that somebody can see an animal right now.

    ``world.stats`` is a ``dict[str, int]`` that world.from_dict coerces with
    ``int(v)``, so both keys are stored as whole seconds / counts and survive a
    save without special handling.
    """
    st = _stats(world)
    if not isinstance(st, dict):
        return
    try:
        now = int(world_now(world))
        prev = st.get("last_animal_t")
        last = int(prev) if isinstance(prev, (int, float)) else None
        if last is None or now - last > SIGHTING_EDGE_SEC or now < last:
            st["animals_seen"] = int(st.get("animals_seen", 0) or 0) + 1
        st["last_animal_t"] = now
    except Exception:
        log.debug("note_animal_seen failed", exc_info=True)


def sighting_freshness(world: Any) -> float:
    """1.0 just after an animal was seen, decaying to 0 over 4 minutes."""
    st = _stats(world)
    if not isinstance(st, dict):
        return 0.0
    v = st.get("last_animal_t")
    if not isinstance(v, (int, float)):
        return 0.0
    age = world_now(world) - float(v)
    if age < 0.0 or not math.isfinite(age):
        return 0.0
    return _clamp01(1.0 - age / SIGHTING_FRESH_SEC)


def _is_mauling(world: Any, animal: Any, exclude: Any = None) -> bool:
    """True when this animal is standing on top of one of the colonists."""
    ax = animal_x(animal, float("nan"))
    if not math.isfinite(ax):
        return False
    ex = getattr(exclude, "id", None)
    for o in _actions.alive_agents(world):
        if ex is not None and getattr(o, "id", None) == ex:
            continue
        try:
            if abs(float(getattr(o, "x", 0.0)) - ax) <= MAUL_REACH:
                return True
        except (TypeError, ValueError):
            continue
    return False


def pick_quarry(agent: Any, world: Any) -> Any | None:
    """The animal an armed agent should go for, or None.

    In order: a hostile animal on top of *them*; a wounded one breaking off
    within a spear's lunge; then the nearest animal mauling somebody else
    anywhere in the camp. Scoring and construction both go through here so the
    score an agent acted on is the fight it actually gets.
    """
    try:
        ax = float(getattr(agent, "x", 0.0))
    except (TypeError, ValueError):
        return None
    near = nearest_animal(world, ax, max_dist=FIGHT_RADIUS, skip_leaving=True)
    if near is not None:
        return near
    parting = nearest_animal(world, ax, max_dist=PARTING_REACH)
    if parting is not None:
        return parting
    best, best_d = None, float("inf")
    for a in animals_of(world):
        if not animal_alive(a) or animal_leaving(a):
            continue
        d = abs(animal_x(a, ax) - ax)
        if d > RESCUE_RADIUS or d >= best_d:
            continue
        if _is_mauling(world, a, exclude=agent):
            best, best_d = a, d
    return best


def under_attack(agent: Any, world: Any, radius: float | None = None) -> bool:
    """True when `agent` should drop what it is doing because of an animal.

    Exposed for ``behavior.emergency_override``, which is the *only* thing that
    re-scores an action already in flight - without it an agent finishes its
    20 s chop while a wolf chews on it. Proximity alone is not the whole test:
    an armed colonist also breaks off to meet an animal still crossing the
    camp, or to pull one off a neighbour, so this asks :func:`pick_quarry` too.
    Pass an explicit `radius` for a plain "is anything near me" check.
    """
    try:
        ax = float(getattr(agent, "x", 0.0))
    except (TypeError, ValueError):
        return False
    r = FLEE_RADIUS if radius is None else float(radius)
    if nearest_animal(world, ax, max_dist=r) is not None:
        return True
    if radius is not None:
        return False
    if (_armed(agent) and _role(agent) != "child"
            and _health_frac(agent) >= BREAK_OFF_HEALTH):
        return pick_quarry(agent, world) is not None
    return False


# ===========================================================================
#  Small agent helpers
# ===========================================================================
def _name(agent: Any) -> str:
    n = getattr(agent, "name", None)
    return str(n) if isinstance(n, str) and n else "Someone"


def _role(agent: Any) -> str:
    r = getattr(agent, "role", None)
    return r if isinstance(r, str) and r else "gatherer"


def _armed(agent: Any) -> bool:
    return str(getattr(agent, "weapon", WEAPON_NONE) or WEAPON_NONE) == WEAPON_SPEAR


def _armoured(agent: Any) -> bool:
    try:
        return float(getattr(agent, "armour", 0.0) or 0.0) >= ARMOUR_LEATHER - 1e-9
    except (TypeError, ValueError):
        return False


def _health_frac(agent: Any) -> float:
    v = getattr(agent, "health", None)
    if not isinstance(v, (int, float)):
        return 1.0
    try:
        return _clamp01(float(v) / max(1.0, MAX_HEALTH))
    except (TypeError, ValueError):
        return 1.0


def _cooldown(agent: Any) -> float:
    v = getattr(agent, "attack_cd", 0.0)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f) or f <= 0.0:
        return 0.0
    return min(f, SPEAR_COOLDOWN)


def _set_cooldown(agent: Any, v: float) -> None:
    try:
        agent.attack_cd = max(0.0, min(float(v), SPEAR_COOLDOWN))
    except Exception:
        pass


def _tick_cooldown(agent: Any, dt: float) -> None:
    _set_cooldown(agent, _cooldown(agent) - max(0.0, float(dt)))


def _set_target(agent: Any, animal: Any) -> None:
    try:
        agent.target_animal = animal_id(animal)
    except Exception:
        pass


def _clear_target(agent: Any) -> None:
    try:
        agent.target_animal = None
    except Exception:
        pass


def _can_afford(world: Any, cost: dict[str, int]) -> bool:
    try:
        return all(stock_qty(world, str(res)) >= int(qty)
                   for res, qty in cost.items())
    except Exception:
        return False


def _bump_stat(world: Any, key: str, n: int = 1) -> None:
    st = _stats(world)
    if not isinstance(st, dict):
        return
    try:
        st[key] = int(st.get(key, 0) or 0) + int(n)
    except Exception:
        pass


# ===========================================================================
#  Crafting
# ===========================================================================
_CRAFT_SPEC: dict[str, dict[str, Any]] = {
    "CraftSpear": {
        "cost": SPEAR_COST, "time": CRAFT_SPEAR_TIME, "morale": 0.10,
        "verb": "bound a stone point to a shaft", "symbol": "^",
    },
    "CraftArmour": {
        "cost": ARMOUR_COST, "time": CRAFT_ARMOUR_TIME, "morale": 0.12,
        "verb": "stitched a hide into armour", "symbol": "#",
    },
}


def _craft_satisfied(kind: str, agent: Any) -> bool:
    return _armed(agent) if kind == "CraftSpear" else _armoured(agent)


def _craft_apply(kind: str, agent: Any) -> None:
    if kind == "CraftSpear":
        agent.weapon = WEAPON_SPEAR
        return
    try:
        cur = float(getattr(agent, "armour", 0.0) or 0.0)
    except (TypeError, ValueError):
        cur = 0.0
    agent.armour = max(cur, ARMOUR_LEATHER)


def _pay(world: Any, a: Action, cost: dict[str, int]) -> bool:
    """Spend `cost` atomically. Records the bill on the action so an abandoned
    craft refunds instead of quietly eating the colony's wood."""
    taken: dict[str, int] = {}
    for res, qty in cost.items():
        want = int(qty)
        got = stock_take(world, str(res), want)
        if got < want:
            if got > 0:
                taken[str(res)] = taken.get(str(res), 0) + got
            for r, q in taken.items():
                stock_add(world, r, q)
            return False
        taken[str(res)] = want
    a.data["paid"] = dict(taken)
    return True


def _paid_up(agent: Any, kind: str) -> bool:
    """True when `agent` is mid-`kind` and has already settled the bill.

    ``score_combat`` gates the two crafts on :func:`_can_afford`, which asks
    "can the colony buy this?". But :func:`_pay` empties the stock the instant
    the crafter reaches the bench, so a craft already under way answers *no
    about itself*. Hide sits at 0 for ~79% of all sim-seconds, so that was
    nearly every armour craft: the in-flight action scored 0.0, choose_action
    drops zero-scored kinds before it reaches its keep-the-running-action
    guard, and the agent wandered off 7.x s into an 8 s craft. Measured over 20
    seeds x 900 s: 104 of 129 armour crafts abandoned after paying, all 104
    with the stock spent on the very craft being cancelled, 61 of them past the
    seven-second mark. Spears escaped only because wood and stone are held in
    bulk (114 of 120 completed).

    The bill is paid; whether we can afford it is a settled question. The
    pre-payment gate in :func:`make_combat_action` is untouched - an agent who
    cannot afford a craft still cannot start one.
    """
    a = getattr(agent, "action", None)
    if a is None or getattr(a, "kind", None) != kind:
        return False
    if getattr(a, "finished", False):
        return False
    try:
        # 'paid' present with no 'made' is the whole distinguishing state: _pay
        # sets it, _h_craft pops it on completion and _c_craft pops it on the
        # refund, so a stale or already-refunded action never matches.
        return isinstance(a.data.get("paid"), dict) and not a.data.get("made")
    except Exception:
        return False


def _h_craft(a: Action, ag: Any, w: Any, dt: float) -> None:
    spec = _CRAFT_SPEC.get(a.kind)
    if spec is None:
        a.failed = True
        return
    cost: dict[str, int] = spec["cost"]

    if a.phase == "start":
        if _craft_satisfied(a.kind, ag):
            a.done = True
            return
        if not _can_afford(w, cost):
            a.failed = True
            return
        a.phase = "approach"

    if a.phase == "approach":
        a.pose = "walk"
        sp = nearest_structure(w, "stockpile", ag.x, built_only=True)
        tx = float(sp.x) if sp is not None else colony_center(w)
        rem = step_toward(ag, w, tx, dt)
        if rem <= REACH:
            if not _pay(w, a, cost):
                a.failed = True        # somebody else spent it while we walked
                return
            a.phase = "work"
            a.data["wt"] = 0.0
        elif a.t > CRAFT_WALK_TIMEOUT:
            a.failed = True
        return

    if a.phase == "work":
        a.pose = "build"
        _halt(ag)
        a.data["wt"] = float(a.data.get("wt", 0.0)) + dt
        _adjust(ag, "fatigue", 0.006 * dt)
        if a.data["wt"] < float(spec["time"]):
            return
        _craft_apply(a.kind, ag)
        a.data["made"] = 1
        a.data.pop("paid", None)       # the bill was honoured, nothing to refund
        _adjust(ag, "morale", float(spec["morale"]))
        emit_speech(w, ag, str(spec["symbol"]))
        chronicle(w, f"{_name(ag)} {spec['verb']}.")
        _bump_stat(w, "spears_made" if a.kind == "CraftSpear" else "armour_made")
        a.done = True
        return

    a.phase = "start"


def _c_craft(a: Action, ag: Any, w: Any) -> None:
    """Give the materials back if the craft was interrupted before it finished."""
    paid = a.data.get("paid")
    if isinstance(paid, dict) and not a.data.get("made"):
        for res, qty in paid.items():
            try:
                stock_add(w, str(res), int(qty))
            except Exception:
                continue
    a.data.pop("paid", None)


# ===========================================================================
#  FightAnimal
# ===========================================================================
def _acquire(a: Action, ag: Any, animal: Any) -> None:
    a.target = animal_id(animal)
    a.data["ax"] = animal_x(animal, float(getattr(ag, "x", 0.0)))
    _set_target(ag, animal)


def _current_target(a: Action, w: Any) -> Any | None:
    animal = animal_by_id(w, a.target) if a.target is not None else None
    if animal is not None and animal_alive(animal):
        return animal
    # Animals with no id field, or an id the registry cannot resolve: fall back
    # to whatever is standing where the target was a moment ago.
    ax = a.data.get("ax")
    if isinstance(ax, (int, float)):
        near = nearest_animal(w, float(ax), max_dist=48.0)
        if near is not None:
            return near
    return None


def _fight_finish(a: Action, ag: Any) -> None:
    """Leave the fight: haul the hides home if we took any, otherwise stop."""
    _clear_target(ag)
    if a.data.get("loot"):
        a.phase = "deliver"
        a.pose = "carry"
    else:
        _halt(ag)
        a.done = True


def _h_fight(a: Action, ag: Any, w: Any, dt: float) -> None:
    if a.phase == "deliver":
        if _deposit_step(a, ag, w, dt):
            a.done = True
        return

    # An unarmed agent has no way to hurt anything. This is a hard rule, not a
    # scoring preference: score_combat and make_combat_action both refuse, and
    # this is the third gate, for a save whose spear went missing. Loot already
    # taken still gets carried home; otherwise fail, so behaviour re-scores
    # this tick and FleeAnimal can take over.
    if not _armed(ag):
        _clear_target(ag)
        if a.data.get("loot"):
            a.phase = "deliver"
            a.pose = "carry"
        else:
            _halt(ag)
            a.failed = True
        return

    _tick_cooldown(ag, dt)

    animal = _current_target(a, w)
    if animal is None:
        animal = pick_quarry(ag, w)
        if animal is None:
            _fight_finish(a, ag)
            return
        _acquire(a, ag, animal)

    if _health_frac(ag) < BREAK_OFF_HEALTH:
        # Bleeding out. Break off and let FleeAnimal take over next re-score;
        # a colony that fights to zero buries somebody every incursion.
        if not a.data.get("broke"):
            a.data["broke"] = 1
            emit_speech(w, ag, "!")
        _fight_finish(a, ag)
        return

    if animal_leaving(animal) and abs(animal_x(animal, ag.x) - float(ag.x)) > SPEAR_REACH * 2.0:
        _fight_finish(a, ag)
        return

    swings = int(a.data.get("swings", 0) or 0)
    if a.t > FIGHT_TIMEOUT or (swings <= 0 and a.t > CHASE_GIVE_UP):
        _fight_finish(a, ag)
        return

    ax = float(ag.x)
    tx = animal_x(animal, ax)
    a.data["ax"] = tx
    gap = abs(tx - ax)

    if gap > SPEAR_REACH:
        a.phase = "approach"
        a.pose = "run" if gap > 70.0 else "walk"
        step_toward(ag, w, tx, dt, speed=CHASE_SPEED, arrive=SPEAR_REACH * 0.7)
        _adjust(ag, "fatigue", 0.012 * dt)
        return

    a.phase = "fight"
    a.pose = "chop"
    _halt(ag)
    _face(ag, tx - ax)
    _adjust(ag, "fatigue", 0.020 * dt)
    _adjust(ag, "morale", -0.004 * dt)
    if _cooldown(ag) > 0.0:
        return

    _set_cooldown(ag, SPEAR_COOLDOWN)
    a.data["swings"] = swings + 1
    kind = animal_kind(animal) or "beast"
    if not hurt_animal(w, animal, SPEAR_DAMAGE, by=ag):
        return

    # --- it went down ------------------------------------------------------
    hides = claim_hides(w, animal)
    if hides > 0:
        _carry_add(ag, a, RES_HIDE, hides)
        a.data["loot"] = 1
    a.data["kills"] = int(a.data.get("kills", 0) or 0) + 1
    _bump_stat(w, "animals_killed")
    _adjust(ag, "morale", 0.16)
    emit_speech(w, ag, "!")
    chronicle(w, f"{_name(ag)} killed a {kind}.")
    _clear_target(ag)

    nxt = pick_quarry(ag, w)
    if nxt is not None and _health_frac(ag) >= BREAK_OFF_HEALTH:
        _acquire(a, ag, nxt)
        a.phase = "approach"
        return
    _fight_finish(a, ag)


def _c_fight(a: Action, ag: Any, w: Any) -> None:
    _clear_target(ag)


# ===========================================================================
#  FleeAnimal
# ===========================================================================
def _h_flee_animal(a: Action, ag: Any, w: Any, dt: float) -> None:
    animal = nearest_animal(w, ag.x)
    if animal is None:
        _halt(ag)
        a.done = True
        return

    ax = float(ag.x)
    fx = animal_x(animal, ax)

    if a.phase == "start":
        d = 1.0 if ax >= fx else -1.0
        # Backed against a wall: better to slip past than to be cornered.
        if abs(_clamp_x(ax + d * SAFE_DIST) - ax) < SAFE_DIST * 0.45:
            d = -d
        a.data["dir"] = float(d)
        a.data["tx"] = _clamp_x(ax + d * SAFE_DIST)
        a.phase = "run"
        emit_speech(w, ag, "!")

    a.pose = "panic"
    d = float(a.data.get("dir", 1.0))
    rem = step_toward(ag, w, float(a.data.get("tx", ax)), dt,
                      speed=FLEE_SPEED, arrive=6.0)
    _adjust(ag, "fatigue", 0.035 * dt)
    _adjust(ag, "morale", -0.015 * dt)

    ax = float(ag.x)
    if abs(ax - fx) >= SAFE_DIST and a.t > 0.6:
        a.done = True
        return

    if rem <= 6.0:
        # Arrived and still not safe: keep going, turning around if the world
        # edge is what stopped us.
        if abs(_clamp_x(ax + d * 150.0) - ax) < 40.0:
            d = -d
            a.data["dir"] = d
        a.data["tx"] = _clamp_x(ax + d * 150.0)

    if a.t > FLEE_TIMEOUT:
        a.done = True


# ===========================================================================
#  Scoring + construction
# ===========================================================================
def score_combat(agent: Any, world: Any) -> dict[str, float]:
    """Utility 0..1 for the four combat/crafting behaviours.

    Meant to be merged into ``behavior.score_actions``'s dict. Always returns
    all four keys and never raises; a world with no animal subsystem scores
    everything except the two crafts at zero.
    """
    out: dict[str, float] = {
        "FleeAnimal": 0.0, "FightAnimal": 0.0,
        "CraftSpear": 0.0, "CraftArmour": 0.0,
    }
    try:
        if not getattr(agent, "alive", True) or getattr(agent, "taken", False):
            return out
        ax = float(getattr(agent, "x", 0.0))
        role = _role(agent)
        armed = _armed(agent)
        armoured = _armoured(agent)
        hp = _health_frac(agent)

        animal = nearest_animal(world, ax)
        dist = abs(animal_x(animal, ax) - ax) if animal is not None else float("inf")
        if animal is not None and dist <= FIGHT_RADIUS * 2.0:
            note_animal_seen(world)
        fresh = sighting_freshness(world)

        # ---------------------------------------------------------- flee ----
        if animal is not None and dist <= FLEE_RADIUS:
            close = 1.0 - _clamp01(dist / FLEE_RADIUS)
            if not armed or role == "child":
                out["FleeAnimal"] = _clamp01(0.92 + 0.08 * close)
            elif hp < BREAK_OFF_HEALTH:
                out["FleeAnimal"] = 0.97

        # --------------------------------------------------------- fight ----
        # This has to clear behavior.OVERRIDE_FLOOR (0.95). Below it,
        # choose_action hands the running action a HYSTERESIS_BONUS of 0.35,
        # so a committed builder scoring 0.70 effectively defends at 1.05 and
        # the spearman never turns round. Measured at the "natural" 0.9: four
        # armed colonists, a wolf pack in the camp for its full 90 s, and
        # *zero* animals killed - they finished their buildings while a wolf
        # ate somebody. Fighting is an interrupt or it does not happen.
        if armed and role != "child" and hp >= BREAK_OFF_HEALTH:
            quarry = pick_quarry(agent, world)
            if quarry is not None:
                qd = abs(animal_x(quarry, ax) - ax)
                near = 1.0 - _clamp01(qd / RESCUE_RADIUS)
                base = 0.95 + 0.04 * near + (0.01 if armoured else 0.0)
                if role == "elder":
                    # The elder is not the one with the spear: dropping below
                    # the floor means they only join in when already idle.
                    base -= 0.12
                out["FightAnimal"] = min(0.99, _clamp01(base))

        # --------------------------------------------------------- craft ----
        # Nobody stands at the workbench with a wolf on top of them - but the
        # gate is FLEE_RADIUS, not FIGHT_RADIUS. Blocking the workbench any
        # wider than the flee reflex is a death spiral: a replacement arrives
        # unarmed, an animal camps the colony for its full 90 s, nobody can
        # ever make a spear, and the colony never re-arms. Measured at the
        # wider gate: one 900 s seed lost 13 people to mauling with 0 spears
        # standing at the end.
        threatened = animal is not None and dist <= FLEE_RADIUS
        if role != "child" and not threatened:
            # ``or _paid_up`` so a craft that has already spent its materials
            # does not score itself out of existence - see _paid_up.
            if not armed and (_can_afford(world, SPEAR_COST)
                              or _paid_up(agent, "CraftSpear")):
                out["CraftSpear"] = _clamp01(0.55 + 0.32 * fresh)
            if not armoured and (_can_afford(world, ARMOUR_COST)
                                 or _paid_up(agent, "CraftArmour")):
                out["CraftArmour"] = _clamp01(0.50 + 0.25 * fresh)
    except Exception:
        log.debug("score_combat failed", exc_info=True)
    return out


def make_combat_action(kind: str, agent: Any, world: Any) -> Action | None:
    """Build one of the four actions, or None if it is not available now."""
    try:
        if kind not in COMBAT_ACTION_KINDS:
            return None
        ax = float(getattr(agent, "x", 0.0))

        if kind == "FightAnimal":
            if not _armed(agent) or _role(agent) == "child":
                return None
            if _health_frac(agent) < BREAK_OFF_HEALTH:
                return None
            animal = pick_quarry(agent, world)
            if animal is None:
                return None
            a = make_action("FightAnimal", target=animal_id(animal),
                            phase="approach", pose="run",
                            ax=animal_x(animal, ax))
            _set_target(agent, animal)
            return a

        if kind == "FleeAnimal":
            animal = nearest_animal(world, ax)
            if animal is None:
                return None
            return make_action("FleeAnimal", target=animal_id(animal),
                               pose="panic", fx=animal_x(animal, ax))

        if kind == "CraftSpear":
            if _armed(agent) or not _can_afford(world, SPEAR_COST):
                return None
            return make_action("CraftSpear", pose="walk")

        if kind == "CraftArmour":
            if _armoured(agent) or not _can_afford(world, ARMOUR_COST):
                return None
            return make_action("CraftArmour", pose="walk")
    except Exception:
        log.debug("make_combat_action(%r) failed", kind, exc_info=True)
    return None


def install_makers(behavior_module: Any) -> int:
    """Splice these four into ``behavior._MAKERS``. Returns how many landed.

    Optional convenience for the integrator - importing this module is enough
    for saves to rehydrate, but behaviour still has to be told how to *choose*
    them. Kept as a call rather than an import-time side effect so this module
    never has to import behavior.py (which imports actions.py, which is where a
    cycle would start).
    """
    n = 0
    try:
        makers = getattr(behavior_module, "_MAKERS", None)
        if not isinstance(makers, dict):
            return 0
        for kind in COMBAT_ACTION_KINDS:
            makers[kind] = _make_for(kind)
            n += 1
    except Exception:
        log.debug("install_makers failed", exc_info=True)
    return n


def _make_for(kind: str) -> Callable[[Any, Any], Action | None]:
    def build(agent: Any, world: Any) -> Action | None:
        return make_combat_action(kind, agent, world)
    build.__name__ = f"_mk_{kind}"
    return build


# ===========================================================================
#  Registration - this is what lets Action.from_dict rebuild a saved fight
# ===========================================================================
_COMBAT_HANDLERS: dict[str, Callable[[Action, Any, Any, float], None]] = {
    "CraftSpear": _h_craft,
    "CraftArmour": _h_craft,
    "FightAnimal": _h_fight,
    "FleeAnimal": _h_flee_animal,
}

_COMBAT_CLEANUP: dict[str, Callable[[Action, Any, Any], None]] = {
    "CraftSpear": _c_craft,
    "CraftArmour": _c_craft,
    "FightAnimal": _c_fight,
}


def _register() -> None:
    """Add the combat kinds to actions.py's registries.

    ``Action.from_dict`` validates ``kind`` against ``actions._HANDLERS`` and
    silently downgrades anything it does not recognise to Wander, so a save
    taken mid-fight only survives if this has run. Importing this module is
    what runs it; idempotent, so a re-import is harmless.
    """
    try:
        _actions._HANDLERS.update(_COMBAT_HANDLERS)
        _actions._CLEANUP.update(_COMBAT_CLEANUP)
        # Keep actions.py's advertised registry honest - its module-level
        # `assert set(ACTION_KINDS) == set(_HANDLERS)` has already run, but
        # anything that re-checks it (or iterates the kinds for a test sweep)
        # should see the four new ones.
        known = tuple(_actions.ACTION_KINDS)
        extra = tuple(k for k in COMBAT_ACTION_KINDS if k not in known)
        if extra:
            _actions.ACTION_KINDS = known + extra
    except Exception:
        log.exception("could not register combat actions; saves will not "
                      "restore a fight in progress")


_register()


# ===========================================================================
#  Smoke test
# ===========================================================================
if __name__ == "__main__":  # pragma: no cover - headless
    import random as _random

    from ..constants import RES_FIBRE, RES_STONE, RES_WOOD
    from .entities import Stickman
    from .structures import StructureRegistry

    class _Animal:
        _n = 0

        def __init__(self, kind: str, x: float) -> None:
            _Animal._n += 1
            self.id = _Animal._n
            self.kind = kind
            self.x = float(x)
            self.y = 600.0
            stats = ANIMAL_STATS[kind]
            self.max_health = stats[0]
            self.health = stats[0]
            self.dps = stats[1]
            self.speed = stats[2]
            self.hides = stats[3]
            self.alive = True
            self.leaving = False

        def hurt(self, amount: float) -> bool:
            self.health = max(0.0, self.health - float(amount))
            if self.health <= 0.0:
                self.alive = False
                return True
            return False

    class _Terrain:
        def ground_y(self, x: float) -> float:
            return 600.0

        def slope(self, x: float) -> float:
            return 0.0

    class _World:
        def __init__(self) -> None:
            self.terrain = _Terrain()
            self.structures = StructureRegistry()
            self.props: list[Any] = []
            self.animals: list[_Animal] = []
            self.stockpile: dict[str, int] = {}
            self.agents: list[Stickman] = []
            self.world_time = 0.0
            self.pyrng = _random.Random(3)
            self.stats: dict[str, int] = {}
            self.lines: list[str] = []
            self.speech: list[dict[str, Any]] = []

        def chronicle(self, text: str) -> None:
            self.lines.append(text)

    dt = 1.0 / 30.0
    w = _World()
    w.structures.create("stockpile", 600.0, 600.0, built=True)
    w.stockpile.update({RES_WOOD: 20, RES_STONE: 20, RES_HIDE: 6, RES_FIBRE: 6})
    hunter = Stickman(id=1, name="Ash", x=560.0, y=600.0)
    w.agents.append(hunter)

    # --- crafting ----------------------------------------------------------
    act = make_combat_action("CraftSpear", hunter, w)
    assert act is not None
    for _ in range(int(30 * 60)):
        act.update(hunter, w, dt)
        if act.finished:
            break
    print(f"craft spear: {act} weapon={hunter.weapon!r} stock={w.stockpile}")
    assert hunter.weapon == WEAPON_SPEAR, "spear never got made"
    assert w.stockpile[RES_WOOD] == 18 and w.stockpile[RES_STONE] == 19

    act = make_combat_action("CraftArmour", hunter, w)
    assert act is not None
    for _ in range(int(30 * 60)):
        act.update(hunter, w, dt)
        if act.finished:
            break
    print(f"craft armour: {act} armour={hunter.armour} stock={w.stockpile}")
    assert hunter.armour == ARMOUR_LEATHER
    assert w.stockpile[RES_HIDE] == 4 and w.stockpile[RES_FIBRE] == 5

    # an abandoned craft refunds what it took
    before = dict(w.stockpile)
    hunter.weapon = WEAPON_NONE
    act = make_combat_action("CraftSpear", hunter, w)
    for _ in range(int(30 * 25)):
        act.update(hunter, w, dt)
        if act.data.get("paid"):
            break
    assert act.data.get("paid"), "never reached the workbench"
    act.abandon(hunter, w)
    assert w.stockpile == before, (w.stockpile, before)
    print("abandoned craft refunded:", w.stockpile)
    hunter.weapon = WEAPON_SPEAR

    # --- a fight -----------------------------------------------------------
    wolf = _Animal("wolf", 640.0)
    w.animals.append(wolf)
    fight = make_combat_action("FightAnimal", hunter, w)
    assert fight is not None
    swings = 0
    for i in range(int(30 * 90)):
        fight.update(hunter, w, dt)
        hunter.apply_physics(dt, w.terrain)
        if i == 40:                        # mid-fight save/load round trip
            fight = Action.from_dict(fight.to_dict())
            assert fight.kind == "FightAnimal", "save downgraded the fight"
        if wolf.alive:
            hunter.hurt(wolf.dps * dt, "mauled")
        if fight.finished:
            break
        swings = int(fight.data.get("swings", 0))
    print(f"fight: {fight} wolf_alive={wolf.alive} swings={swings} "
          f"hp={hunter.health:.1f} stock_hide={w.stockpile.get(RES_HIDE)}")
    assert not wolf.alive, "an armed agent could not kill one wolf"
    assert hunter.alive and hunter.health > 60.0, hunter.health
    assert w.stockpile.get(RES_HIDE, 0) >= 5, w.stockpile

    # --- fleeing -----------------------------------------------------------
    scout = Stickman(id=2, name="Bri", x=500.0, y=600.0)
    w.agents.append(scout)
    bear = _Animal("bear", 520.0)
    w.animals.append(bear)
    sc = score_combat(scout, w)
    print("unarmed scores:", {k: round(v, 2) for k, v in sc.items()})
    assert sc["FleeAnimal"] > 0.9 and sc["FightAnimal"] == 0.0
    assert sc["CraftSpear"] == 0.0, "crafting with a bear on top of you"
    run = make_combat_action("FleeAnimal", scout, w)
    assert run is not None
    for _ in range(int(30 * 30)):
        run.update(scout, w, dt)
        scout.apply_physics(dt, w.terrain)
        if run.finished:
            break
    gap = abs(scout.x - bear.x)
    print(f"flee: {run} gap={gap:.0f}px pose={run.pose}")
    assert gap >= SAFE_DIST - 8.0, gap

    # --- armed scoring, and the break-off rule ------------------------------
    bear.x = scout.x + 60.0
    sc = score_combat(hunter, w)
    print("armed scores:", {k: round(v, 2) for k, v in sc.items()})
    hunter.x = bear.x - 40.0
    sc = score_combat(hunter, w)
    assert sc["FightAnimal"] > 0.85, sc
    hunter.health = MAX_HEALTH * 0.2
    sc = score_combat(hunter, w)
    assert sc["FightAnimal"] == 0.0 and sc["FleeAnimal"] > 0.9, sc
    print("wounded scores:", {k: round(v, 2) for k, v in sc.items()})

    # --- registration / round-trip -----------------------------------------
    for kind in COMBAT_ACTION_KINDS:
        rt = Action.from_dict({"kind": kind, "phase": "start"})
        assert rt.kind == kind, f"{kind} did not survive from_dict"
    assert set(COMBAT_ACTION_KINDS) <= set(_actions.ACTION_KINDS)
    print("registered:", COMBAT_ACTION_KINDS)

    # --- an absent animal subsystem is not an error -------------------------
    class _Bare:
        pass

    bare = _Bare()
    assert nearest_animal(bare, 0.0) is None
    assert score_combat(hunter, bare)["FightAnimal"] == 0.0
    a = make_action("FightAnimal")
    a.update(hunter, bare, dt)
    print("no-animals world:", a)
    print("chronicle:", w.lines)
