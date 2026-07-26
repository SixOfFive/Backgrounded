"""Utility AI: what should this stickman do next, and what should the colony
build next.

Two halves:

* :func:`choose_action` scores every candidate behaviour 0..1 from the agent's
  needs and the state of the world, adds a small random tiebreak and a
  hysteresis bonus for whatever it is already doing, and returns the winner as
  an :class:`~.actions.Action`.
* :func:`update_director` is the colony-level planner. It decides the next
  building the colony wants, stakes out a site for it (an unbuilt Structure in
  the registry, which is what gives ``BuildStructure`` something to target),
  publishes ``world.build_queue`` / ``world.build_needs``, and keeps roles
  assigned as children mature and watchtowers get built.

No pygame. Nothing here mutates render state; the director does mutate sim
state, which is its whole job.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Callable

import numpy as np

from ..constants import (
    MAT_LAVA,
    MAX_SLOPE_WALK,
    RENDER_W,
    RES_COOKED,
    RES_FIBRE,
    RES_FOOD,
    RES_STONE,
    RES_WOOD,
    SCENE_BLIZZARD,
    SCENE_NIGHT_STORM,
)
from .actions import (
    CARRY_CAP,
    Action,
    alive_agents,
    chronicle,
    colony_center,
    find_prop,
    food_in_store,
    ground_y,
    hazards_of,
    is_night,
    make_action,
    nearest_structure,
    prop_alive,
    props_of,
    rng_of,
    slope_at,
    stock_qty,
    structures_of,
    world_now,
)
from .structures import Structure, StructureRegistry, structure_spec

log = logging.getLogger(__name__)

__all__ = [
    "choose_action",
    "score_actions",
    "update_director",
    "assign_roles",
    "next_build_kind",
    "find_chasm",
    "HYSTERESIS_BONUS",
    "ROLE_AFFINITY",
    "ROLES",
]

# ------------------------------------------------------------------ tuning --
HYSTERESIS_BONUS = 0.35     # stickiness for the action already running

#: Once an agent commits to something it sticks with it for at least this long
#: unless a genuine emergency (score >= OVERRIDE_FLOOR) interrupts.
#:
#: Without this, two actions with near-equal utility trade places every time the
#: AI re-scores. Measured before the change: one agent racked up 192 action
#: switches and 184 direction reversals in 180 seconds, pacing back and forth
#: over an 11 px stretch between a build site and the stockpile. A hysteresis
#: bonus alone cannot fix that - it only biases the score, and the moment the
#: rival action creeps above the margin the pair simply swap and swap back.
MIN_COMMIT_SEC = 7.0
TIEBREAK = 0.03             # random jitter so identical scores do not lock step
OVERRIDE_FLOOR = 0.95       # scores at or above this ignore hysteresis
DIRECTOR_PERIOD = 2.0       # seconds between director passes
MAX_CONCURRENT_SITES = 2    # unfinished buildings the colony tolerates at once
MAX_HUTS = 6
MAX_WALLS = 4
TOTEM_POP = 8
TREE_TARGET = 10            # below this, someone plants saplings
SITE_RANGE = 300.0          # how far from the colony centre a site may be
CHILD_MATURE_AGE = 480.0    # seconds before a child picks up a trade
BUILDER_RATIO = 0.45
CHASM_DEPTH = 36.0
CHASM_MIN_W = 26.0
CHASM_MAX_W = 190.0
CHASM_RECHECK = 20.0
GRAVE_FRESH = 300.0
CELEBRATION_FRESH = 30.0

ROLES: tuple[str, ...] = ("gatherer", "builder", "elder", "child", "lookout")

# per-role appetite for each family of work
ROLE_AFFINITY: dict[str, dict[str, float]] = {
    "gatherer": {"gather": 1.00, "build": 0.55, "social": 0.85, "watch": 0.35},
    "builder":  {"gather": 0.60, "build": 1.00, "social": 0.70, "watch": 0.35},
    "elder":    {"gather": 0.35, "build": 0.40, "social": 1.00, "watch": 0.55},
    "lookout":  {"gather": 0.55, "build": 0.45, "social": 0.60, "watch": 1.00},
    "child":    {"gather": 0.30, "build": 0.15, "social": 1.00, "watch": 0.10},
}
_DEFAULT_AFFINITY = ROLE_AFFINITY["gatherer"]

KIND_PRIORITY: dict[str, float] = {
    "firepit": 1.00,
    "stockpile": 0.92,
    "hut": 0.85,
    "grave": 0.80,
    "bridge": 0.62,
    "wall": 0.58,
    "watchtower": 0.54,
    "totem": 0.40,
}

_COLD_SCENES = (SCENE_BLIZZARD, SCENE_NIGHT_STORM)
_TREE_KINDS = ("tree", "pine", "oak")


def _clamp01(v: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f):
        return 0.0
    return 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)


def _need(agent: Any, name: str) -> float:
    return _clamp01(getattr(agent, name, 0.0))


def _role(agent: Any) -> str:
    r = getattr(agent, "role", "gatherer")
    return r if isinstance(r, str) and r else "gatherer"


def _aff(agent: Any) -> dict[str, float]:
    return ROLE_AFFINITY.get(_role(agent), _DEFAULT_AFFINITY)


def _scene(world: Any) -> str:
    s = getattr(world, "scene", "")
    return s if isinstance(s, str) else ""


# ===========================================================================
#  Danger
# ===========================================================================
def _danger_for(agent: Any, world: Any) -> dict[str, Any] | None:
    """The most pressing hazard threatening `agent`, as flee parameters."""
    best: dict[str, Any] | None = None
    best_urgency = 0.0
    try:
        ax = float(getattr(agent, "x", 0.0))
    except (TypeError, ValueError):
        return None
    for h in hazards_of(world):
        try:
            if "water_y" in h:
                gy = ground_y(world, ax)
                if gy < float(h["water_y"]) - 14.0:
                    continue
                urgency = 1.0
                cand = {
                    "fx": ax, "radius": 220.0, "uphill": True,
                    "kind": str(h.get("kind", "flood")),
                }
            else:
                hx = float(h.get("x", 0.0))
                radius = max(8.0, float(h.get("radius", 70.0)))
                d = abs(ax - hx)
                if d > radius:
                    continue
                urgency = 1.0 - (d / radius) * 0.5
                cand = {
                    "fx": hx, "radius": radius * 1.45, "uphill": False,
                    "kind": str(h.get("kind", "danger")),
                }
        except Exception:
            continue
        if urgency > best_urgency:
            best_urgency = urgency
            best = cand
    if best is not None:
        best["urgency"] = best_urgency
    return best


# ===========================================================================
#  Scoring
# ===========================================================================
def score_actions(agent: Any, world: Any) -> dict[str, float]:
    """Every candidate behaviour scored 0..1. Public for debugging/tests."""
    s: dict[str, float] = {}
    reg = structures_of(world)
    pop = max(1, len(alive_agents(world)))
    aff = _aff(agent)
    role = _role(agent)
    night = is_night(world)
    scene = _scene(world)
    hunger = _need(agent, "hunger")
    fatigue = _need(agent, "fatigue")
    warmth = _need(agent, "warmth")
    morale = _need(agent, "morale")
    carrying = getattr(agent, "carrying", None)
    carry_qty = int(getattr(agent, "carry_qty", 0) or 0)
    cur = getattr(agent, "action", None)
    cur_kind = getattr(cur, "kind", None) if cur is not None else None
    ax = float(getattr(agent, "x", 0.0))

    # ---------------------------------------------------------- danger ------
    danger = _danger_for(agent, world)
    s["FleeFrom"] = 1.0 if danger else 0.0
    s["Panic"] = 0.62 if (danger and morale < 0.32) else 0.0

    # ------------------------------------------------------------- eat ------
    food_res, food_qty = food_in_store(world)
    holding_food = carrying in (RES_FOOD, RES_COOKED) and carry_qty > 0
    can_eat = food_qty > 0 or holding_food
    s["Eat"] = _clamp01(hunger * hunger * (1.0 if can_eat else 0.12))
    if hunger > 0.85 and can_eat:
        s["Eat"] = 1.0

    # ----------------------------------------------------------- sleep ------
    hut = _free_hut(world, agent)
    if hut is not None:
        s["Sleep"] = _clamp01(fatigue * fatigue * (1.75 if night else 0.55))
        if fatigue > 0.93:
            s["Sleep"] = 1.0
    else:
        s["Sleep"] = 0.0

    # ------------------------------------------------------------ warm ------
    fire = nearest_structure(world, "firepit", ax, built_only=True)
    if fire is not None:
        boost = 1.65 if (night or scene in _COLD_SCENES) else 0.75
        wood_ok = fire.fire_active or stock_qty(world, RES_WOOD) >= 2
        s["WarmAtFire"] = _clamp01(warmth * warmth * boost * (1.0 if wood_ok else 0.25))
        if warmth > 0.9 and wood_ok:
            s["WarmAtFire"] = 1.0
    else:
        s["WarmAtFire"] = 0.0

    # ----------------------------------------------------------- build ------
    queue = _queue_of(world)
    if queue and reg is not None:
        item = queue[0]
        needs = item.get("needs") or {}
        avail = _availability(world, needs)
        unmet = 1.0 - 0.55 * float(item.get("completion", 0.0))
        weight = float(item.get("priority", 0.6))
        s["BuildStructure"] = _clamp01(
            aff["build"] * unmet * (0.28 + 0.72 * avail) * (0.55 + 0.45 * weight)
        )
    else:
        s["BuildStructure"] = 0.0

    # ---------------------------------------------------------- repair ------
    s["RepairStructure"] = 0.0
    if reg is not None:
        try:
            dmg = reg.damaged(threshold=0.9)
        except Exception:
            dmg = []
        if dmg:
            worst = min(st.hp / max(1.0, st.max_hp) for st in dmg)
            s["RepairStructure"] = _clamp01(aff["build"] * (1.0 - worst) * 1.25)

    # ---------------------------------------------------------- gather ------
    # What the *head* of the build queue is short of is the colony's real
    # bottleneck - a build stalled on 2 fibre has to outrank idling.
    shortfall = _needs_of(world)
    blocking = _blocking_resources(world)
    wood_urg = _gather_urgency(RES_WOOD, shortfall, blocking, 12.0)
    stone_urg = _gather_urgency(RES_STONE, shortfall, blocking, 10.0)
    fibre_urg = _gather_urgency(RES_FIBRE, shortfall, blocking, 6.0)
    food_target = float(max(8, pop * 4))
    stored_food = stock_qty(world, RES_FOOD) + stock_qty(world, RES_COOKED)
    food_short = _clamp01((food_target - stored_food) / food_target)
    food_urg = max(0.15 + 0.75 * food_short, fibre_urg)

    s["GatherWood"] = _clamp01(aff["gather"] * (0.10 + 0.90 * wood_urg))
    s["GatherStone"] = _clamp01(aff["gather"] * (0.05 + 0.95 * stone_urg))
    s["ForageBerries"] = _clamp01(aff["gather"] * food_urg + hunger * 0.30)
    if _find_target_prop(world, _TREE_KINDS, ax) is None:
        s["GatherWood"] = 0.0
    if _find_target_prop(world, ("rock", "boulder", "stone", "outcrop"), ax) is None:
        s["GatherStone"] = 0.0
    if _find_target_prop(world, ("bush", "berry", "berrybush", "shrub"), ax) is None:
        s["ForageBerries"] = 0.0

    # ------------------------------------------------------------ haul ------
    s["HaulToStockpile"] = (
        _clamp01(0.42 + 0.42 * min(1.0, carry_qty / float(CARRY_CAP)))
        if (carry_qty > 0 and carrying) else 0.0
    )

    # ------------------------------------------------------------ cook ------
    raw = stock_qty(world, RES_FOOD)
    if fire is not None and raw >= 3:
        want = 0.30 + (0.25 if stock_qty(world, RES_COOKED) == 0 else 0.0)
        s["CookFood"] = _clamp01(aff["gather"] * want)
    else:
        s["CookFood"] = 0.0

    # ----------------------------------------------------------- plant ------
    trees = _count_props(world, _TREE_KINDS)
    s["PlantSapling"] = (
        _clamp01(aff["gather"] * 0.42 * _clamp01((TREE_TARGET - trees) / TREE_TARGET))
        if trees < TREE_TARGET else 0.0
    )

    # -------------------------------------------------------- converse ------
    idle_t = float(getattr(cur, "t", 0.0)) if cur_kind == "Wander" else 0.0
    busy = cur_kind not in (None, "Wander")
    partner = _talk_partner(world, agent)
    if partner is not None and pop >= 2:
        base = 0.22 * morale + 0.16 * _clamp01(idle_t / 12.0)
        s["Converse"] = _clamp01(base * (0.30 if busy else 1.0) * aff["social"])
    else:
        s["Converse"] = 0.0

    # ------------------------------------------------------- celebrate ------
    cel = _fresh_celebration(world)
    s["Celebrate"] = _clamp01(0.82 * morale + 0.18) if cel is not None else 0.0

    # ----------------------------------------------------------- mourn ------
    grave = _grave_to_mourn(world, agent)
    s["Mourn"] = 0.68 if grave is not None else 0.0

    # --------------------------------------------------------- lookout ------
    tower = _free_tower(world, agent)
    if tower is not None:
        s["Lookout"] = _clamp01(aff["watch"] * (0.85 if role == "lookout" else 0.18))
    else:
        s["Lookout"] = 0.0

    # ---------------------------------------------------------- climb -------
    if role == "lookout" and tower is None and (reg is None or reg.count("watchtower") == 0):
        s["ClimbTo"] = 0.25
    else:
        s["ClimbTo"] = 0.0

    # ---------------------------------------------------------- child -------
    s["FollowParent"] = 0.58 if (role == "child" and pop >= 2) else 0.0

    # --------------------------------------------------------- wander -------
    s["Wander"] = 0.10

    # Needs never fully take over from survival: nothing outranks fleeing.
    if danger:
        for k in list(s):
            if k not in ("FleeFrom", "Panic"):
                s[k] = min(s[k], 0.35)
    return s


def _short_frac(qty: Any, scale: float) -> float:
    try:
        return _clamp01(float(qty) / max(1.0, scale))
    except (TypeError, ValueError):
        return 0.0


def _blocking_resources(world: Any) -> dict[str, int]:
    """Resources the head of the build queue cannot get from the stockpile.
    These are what is actually holding the colony up."""
    queue = _queue_of(world)
    if not queue:
        return {}
    out: dict[str, int] = {}
    for res, qty in (queue[0].get("needs") or {}).items():
        try:
            deficit = int(qty) - stock_qty(world, str(res))
        except (TypeError, ValueError):
            continue
        if deficit > 0:
            out[str(res)] = deficit
    return out


def _gather_urgency(
    res: str, shortfall: dict[str, int], blocking: dict[str, int], scale: float
) -> float:
    """0..1 appetite for fetching `res`. Any outstanding need clears the Wander
    floor comfortably; blocking the current build clears everything else."""
    deficit = 0
    try:
        deficit = int(shortfall.get(res, 0))
    except (TypeError, ValueError):
        deficit = 0
    urgency = 0.0
    if deficit > 0:
        urgency = 0.42 + 0.48 * _short_frac(deficit, scale)
    if blocking.get(res, 0) > 0:
        urgency = max(urgency, 0.78)
    return _clamp01(urgency)


def _availability(world: Any, needs: dict[str, Any]) -> float:
    """How much of what a build still wants is sitting in the stockpile, 0..1."""
    if not needs:
        return 1.0
    total = 0.0
    n = 0
    for res, qty in needs.items():
        try:
            q = float(qty)
        except (TypeError, ValueError):
            continue
        if q <= 0:
            continue
        total += min(1.0, stock_qty(world, str(res)) / q)
        n += 1
    return total / n if n else 1.0


def _queue_of(world: Any) -> list[dict[str, Any]]:
    q = getattr(world, "build_queue", None)
    return q if isinstance(q, list) else []


def _needs_of(world: Any) -> dict[str, int]:
    n = getattr(world, "build_needs", None)
    return n if isinstance(n, dict) else {}


def _free_hut(world: Any, agent: Any) -> Structure | None:
    aid = int(getattr(agent, "id", -1))
    return nearest_structure(
        world, "hut", float(getattr(agent, "x", 0.0)), built_only=True,
        predicate=lambda s: s.has_room() or aid in s.occupants,
    )


def _free_tower(world: Any, agent: Any) -> Structure | None:
    aid = int(getattr(agent, "id", -1))
    return nearest_structure(
        world, "watchtower", float(getattr(agent, "x", 0.0)), built_only=True,
        predicate=lambda s: s.has_room() or aid in s.occupants,
    )


def _find_target_prop(world: Any, kinds: tuple[str, ...], x: float) -> Any | None:
    return find_prop(world, kinds, x)


def _count_props(world: Any, kinds: tuple[str, ...]) -> int:
    want = tuple(k.lower() for k in kinds)
    n = 0
    for p in props_of(world):
        if not prop_alive(p):
            continue
        k = getattr(p, "kind", None) or getattr(p, "type", None) or ""
        if str(k).lower() in want:
            n += 1
    return n


def _talk_partner(world: Any, agent: Any) -> Any | None:
    my_id = getattr(agent, "id", None)
    ax = float(getattr(agent, "x", 0.0))
    for o in alive_agents(world):
        if getattr(o, "id", None) == my_id:
            continue
        act = getattr(o, "action", None)
        if act is not None and getattr(act, "kind", "") in (
                "Sleep", "Lookout", "FleeFrom", "Panic"):
            continue
        if abs(float(getattr(o, "x", 0.0)) - ax) <= 240.0:
            return o
    return None


def _fresh_celebration(world: Any) -> dict[str, Any] | None:
    q = getattr(world, "celebrations", None)
    if not isinstance(q, list) or not q:
        return None
    now = world_now(world)
    for c in reversed(q):
        try:
            if now - float(c.get("t", 0.0)) <= CELEBRATION_FRESH:
                return c
        except (TypeError, ValueError, AttributeError):
            continue
    return None


def _grave_to_mourn(world: Any, agent: Any) -> Structure | None:
    reg = structures_of(world)
    if reg is None:
        return None
    aid = int(getattr(agent, "id", -1))
    now = world_now(world)
    try:
        graves = reg.of_kind("grave")
    except Exception:
        return None
    best, best_d = None, float("inf")
    ax = float(getattr(agent, "x", 0.0))
    for g in graves:
        try:
            placed = float(g.state.get("placed_t", 0.0))
            if placed > 0.0 and now - placed > GRAVE_FRESH:
                continue
            if placed <= 0.0 and float(g.state.get("mourn_done", 0.0)) > 0.0:
                continue
            mourners = g.state.get("mourners")
            if isinstance(mourners, list) and aid in mourners:
                continue
            d = abs(float(g.x) - ax)
            if d < best_d:
                best_d, best = d, g
        except Exception:
            continue
    return best


# ===========================================================================
#  Action construction
# ===========================================================================
def _mk_simple(kind: str) -> Callable[[Any, Any], Action | None]:
    def build(agent: Any, world: Any) -> Action | None:
        return make_action(kind)
    return build


def _mk_flee(agent: Any, world: Any) -> Action | None:
    d = _danger_for(agent, world)
    if d is None:
        return None
    return make_action(
        "FleeFrom", target=float(d["fx"]), fx=float(d["fx"]),
        radius=float(d["radius"]), uphill=bool(d.get("uphill")),
        danger=str(d.get("kind", "danger")),
    )


def _mk_sleep(agent: Any, world: Any) -> Action | None:
    hut = _free_hut(world, agent)
    return make_action("Sleep", target=hut.id) if hut is not None else None


def _mk_warm(agent: Any, world: Any) -> Action | None:
    fire = nearest_structure(world, "firepit", float(getattr(agent, "x", 0.0)),
                             built_only=True)
    return make_action("WarmAtFire", target=fire.id) if fire is not None else None


def _mk_cook(agent: Any, world: Any) -> Action | None:
    fire = nearest_structure(world, "firepit", float(getattr(agent, "x", 0.0)),
                             built_only=True)
    return make_action("CookFood", target=fire.id) if fire is not None else None


def _mk_build(agent: Any, world: Any) -> Action | None:
    queue = _queue_of(world)
    if not queue:
        reg = structures_of(world)
        s = reg.find_incomplete(None, float(getattr(agent, "x", 0.0))) if reg else None
        return make_action("BuildStructure", target=s.id) if s is not None else None
    ax = float(getattr(agent, "x", 0.0))
    top = queue[0]
    # Everything at the head priority is fair game; take the closest of those.
    lead = float(top.get("priority", 0.0))
    pool = [q for q in queue if float(q.get("priority", 0.0)) >= lead - 0.06]
    pick = min(pool, key=lambda q: abs(float(q.get("x", ax)) - ax))
    return make_action("BuildStructure", target=pick.get("id"))


def _mk_repair(agent: Any, world: Any) -> Action | None:
    reg = structures_of(world)
    if reg is None:
        return None
    try:
        dmg = reg.damaged(threshold=0.9)
    except Exception:
        return None
    if not dmg:
        return None
    ax = float(getattr(agent, "x", 0.0))
    worst = min(dmg, key=lambda s: (s.hp / max(1.0, s.max_hp), abs(s.x - ax)))
    return make_action("RepairStructure", target=worst.id)


def _mk_converse(agent: Any, world: Any) -> Action | None:
    other = _talk_partner(world, agent)
    if other is None:
        return None
    return make_action("Converse", target=getattr(other, "id", None))


def _mk_celebrate(agent: Any, world: Any) -> Action | None:
    cel = _fresh_celebration(world)
    if cel is None:
        return None
    return make_action("Celebrate", tx=float(cel.get("x", getattr(agent, "x", 0.0))))


def _mk_mourn(agent: Any, world: Any) -> Action | None:
    g = _grave_to_mourn(world, agent)
    return make_action("Mourn", target=g.id) if g is not None else None


def _mk_lookout(agent: Any, world: Any) -> Action | None:
    tower = _free_tower(world, agent)
    return make_action("Lookout", target=tower.id) if tower is not None else None


def _mk_climb(agent: Any, world: Any) -> Action | None:
    ax = float(getattr(agent, "x", 0.0))
    return make_action("ClimbTo", ty=ground_y(world, ax) - 110.0)


def _mk_follow(agent: Any, world: Any) -> Action | None:
    my_id = getattr(agent, "id", None)
    ax = float(getattr(agent, "x", 0.0))
    best, best_d = None, float("inf")
    for o in alive_agents(world):
        if getattr(o, "id", None) == my_id or _role(o) == "child":
            continue
        d = abs(float(getattr(o, "x", 0.0)) - ax)
        if d < best_d:
            best_d, best = d, o
    if best is None:
        return None
    return make_action("FollowParent", target=getattr(best, "id", None))


_MAKERS: dict[str, Callable[[Any, Any], Action | None]] = {
    "Wander": _mk_simple("Wander"),
    "GatherWood": _mk_simple("GatherWood"),
    "GatherStone": _mk_simple("GatherStone"),
    "ForageBerries": _mk_simple("ForageBerries"),
    "HaulToStockpile": _mk_simple("HaulToStockpile"),
    "Eat": _mk_simple("Eat"),
    "PlantSapling": _mk_simple("PlantSapling"),
    "Panic": _mk_simple("Panic"),
    "FleeFrom": _mk_flee,
    "Sleep": _mk_sleep,
    "WarmAtFire": _mk_warm,
    "CookFood": _mk_cook,
    "BuildStructure": _mk_build,
    "RepairStructure": _mk_repair,
    "Converse": _mk_converse,
    "Celebrate": _mk_celebrate,
    "Mourn": _mk_mourn,
    "Lookout": _mk_lookout,
    "ClimbTo": _mk_climb,
    "FollowParent": _mk_follow,
}


def choose_action(agent: Any, world: Any) -> Action:
    """Pick the highest-utility action for `agent`. Never raises."""
    try:
        _ensure_director(world)
        cur = getattr(agent, "action", None)
        cur_kind = getattr(cur, "kind", None) if cur is not None else None
        cur_live = cur is not None and not getattr(cur, "finished", True)

        scores = score_actions(agent, world)
        rng = rng_of(world)

        # Commitment window: keep going unless something is genuinely urgent.
        if cur_live:
            elapsed = float(getattr(cur, "t", 0.0) or 0.0)
            if elapsed < MIN_COMMIT_SEC:
                peak = max(scores.values()) if scores else 0.0
                if peak < OVERRIDE_FLOOR:
                    return cur
        ranked: list[tuple[float, str]] = []
        for kind, base in scores.items():
            if base <= 0.0:
                continue
            v = float(base)
            if cur_live and kind == cur_kind and v < OVERRIDE_FLOOR:
                v += HYSTERESIS_BONUS
            ranked.append((v + rng.uniform(0.0, TIEBREAK), kind))
        ranked.sort(key=lambda kv: (-kv[0], kv[1]))

        for _, kind in ranked:
            if cur_live and kind == cur_kind:
                return cur          # keep the in-flight machine, do not restart it
            maker = _MAKERS.get(kind)
            if maker is None:
                continue
            try:
                act = maker(agent, world)
            except Exception:
                log.debug("maker for %s failed", kind, exc_info=True)
                continue
            if act is None:
                continue
            if cur is not None and cur is not act:
                try:
                    cur.abandon(agent, world)
                except Exception:
                    pass
            return act

        if cur_live and cur_kind == "Wander":
            return cur
        if cur is not None:
            try:
                cur.abandon(agent, world)
            except Exception:
                pass
        return make_action("Wander")
    except Exception:
        log.warning("choose_action failed", exc_info=True)
        return make_action("Wander")


# ===========================================================================
#  Colony director
# ===========================================================================
def _ensure_director(world: Any) -> None:
    """Run the director if nobody else has recently. Keeps behaviour correct
    whether or not world.py drives update_director itself."""
    try:
        now = world_now(world)
        last = getattr(world, "_bhv_dir_t", None)
        if last is None:
            update_director(world, DIRECTOR_PERIOD)
            return
        elapsed = now - float(last)
        if elapsed >= DIRECTOR_PERIOD or elapsed < 0.0:
            update_director(world, max(0.0, min(elapsed, 30.0)))
    except Exception:
        log.debug("_ensure_director failed", exc_info=True)


def update_director(world: Any, dt: float) -> None:
    """Colony-level planning pass. Safe to call every tick or every 2 seconds."""
    try:
        setattr(world, "_bhv_dir_t", world_now(world))
    except Exception:
        pass
    try:
        assign_roles(world, dt)
    except Exception:
        log.debug("assign_roles failed", exc_info=True)

    reg = structures_of(world)
    if reg is None:
        _publish(world, [], {})
        return

    try:
        _stake_out_site(world, reg)
    except Exception:
        log.debug("site selection failed", exc_info=True)

    try:
        queue: list[dict[str, Any]] = []
        for s in reg.incomplete():
            queue.append({
                "id": int(s.id),
                "kind": str(s.kind),
                "x": float(s.x),
                "y": float(s.y),
                "stage": int(s.stage),
                "completion": float(s.completion()),
                "priority": KIND_PRIORITY.get(s.kind, 0.5),
                "needs": {k: int(v) for k, v in s.total_remaining_cost().items()},
            })
        queue.sort(key=lambda q: (-q["priority"], -q["completion"]))

        wanted: dict[str, int] = {}
        for q in queue:
            for res, qty in q["needs"].items():
                wanted[res] = wanted.get(res, 0) + int(qty)
        short: dict[str, int] = {}
        for res, qty in wanted.items():
            deficit = qty - stock_qty(world, res)
            if deficit > 0:
                short[res] = deficit
        _publish(world, queue, short)
    except Exception:
        log.debug("build queue update failed", exc_info=True)
        _publish(world, [], {})


def _publish(world: Any, queue: list[dict[str, Any]], needs: dict[str, int]) -> None:
    for name, value in (("build_queue", queue), ("build_needs", needs)):
        try:
            setattr(world, name, value)
        except Exception:
            pass


def next_build_kind(world: Any, reg: StructureRegistry | None = None) -> str | None:
    """What the colony wants next: firepit -> stockpile -> hut -> more huts ->
    wall -> watchtower -> bridge (if a chasm exists) -> totem at a milestone."""
    reg = reg if reg is not None else structures_of(world)
    if reg is None:
        return None
    pop = len(alive_agents(world))

    def built(kind: str) -> int:
        try:
            return reg.count(kind, built_only=True)
        except Exception:
            return 0

    if built("firepit") < 1:
        return "firepit"
    if built("stockpile") < 1:
        return "stockpile"
    if built("hut") < 1:
        return "hut"
    want_huts = min(MAX_HUTS, max(1, (pop + 1) // 2))
    if built("hut") < want_huts:
        return "hut"
    if pop >= 4 and built("wall") < 1:
        return "wall"
    if pop >= 5 and built("watchtower") < 1:
        return "watchtower"
    if built("bridge") < 1 and find_chasm(world) is not None:
        return "bridge"
    if pop >= TOTEM_POP and built("totem") < 1:
        return "totem"
    want_walls = min(MAX_WALLS, pop // 3)
    if built("wall") < want_walls:
        return "wall"
    if built("firepit") < min(2, 1 + pop // 6):
        return "firepit"
    return None


def _stake_out_site(world: Any, reg: StructureRegistry) -> None:
    """Place the next wanted building as an unbuilt Structure, so builders have
    somewhere concrete to go."""
    try:
        pending = reg.incomplete()
    except Exception:
        return
    if len(pending) >= MAX_CONCURRENT_SITES:
        return
    kind = next_build_kind(world, reg)
    if kind is None:
        return
    if any(s.kind == kind for s in pending):
        return
    site = pick_site(world, reg, kind)
    if site is None:
        return
    x, y, extra = site
    s = reg.create(kind, x, y, rng=rng_of(world), state=extra or None)
    chronicle(world, f"The colony staked out a new {kind}.")
    log.debug("director staked %s at %.0f", kind, x)


def pick_site(
    world: Any, reg: StructureRegistry, kind: str
) -> tuple[float, float, dict[str, Any]] | None:
    """Choose where to put a `kind`. Returns (x, y, extra_state) or None."""
    spec = structure_spec(kind)
    center = colony_center(world)

    if kind == "bridge":
        span = find_chasm(world)
        if span is None:
            return None
        x0, x1 = span
        cx = 0.5 * (x0 + x1)
        rim = min(ground_y(world, x0 - 4.0), ground_y(world, x1 + 4.0))
        return cx, rim, {"w": (x1 - x0) + 18.0, "span": [float(x0), float(x1)]}

    lo = max(24.0, center - SITE_RANGE)
    hi = min(float(RENDER_W - 24), center + SITE_RANGE)
    if hi - lo < 24.0:
        lo, hi = 24.0, float(RENDER_W - 24)
    xs = np.arange(lo, hi, 6.0, dtype=np.float64)
    if xs.size == 0:
        return None

    ys = np.array([ground_y(world, float(x)) for x in xs], dtype=np.float64)
    slopes = np.array([abs(slope_at(world, float(x))) for x in xs], dtype=np.float64)

    score = np.zeros_like(xs)
    score -= slopes * 2.4                                   # flat ground wins
    score -= np.abs(xs - center) / 420.0                    # stay close to home
    valid = slopes < MAX_SLOPE_WALK

    if kind == "watchtower":
        score += (ys.max() - ys) / 45.0                     # high ground
    elif kind == "wall":
        score += np.minimum(np.abs(xs - center) / 220.0, 1.4)   # out on the edge
    elif kind == "firepit":
        score += 0.5 - np.abs(xs - center) / 300.0

    # keep clear of what is already standing
    for s in reg:
        gap = max(spec.spacing, structure_spec(s.kind).spacing) * 0.85
        d = np.abs(xs - float(s.x))
        score -= np.where(d < gap, (gap - d) / gap * 3.2, 0.0)

    # do not build on top of trees and rocks
    for p in props_of(world):
        if not prop_alive(p):
            continue
        try:
            px = float(getattr(p, "x", 0.0))
        except (TypeError, ValueError):
            continue
        d = np.abs(xs - px)
        score -= np.where(d < 18.0, (18.0 - d) / 18.0 * 1.1, 0.0)

    # never on lava
    terr = getattr(world, "terrain", None)
    mat = getattr(terr, "material", None)
    if mat is not None:
        try:
            arr = np.asarray(mat)
            idx = np.clip(xs.astype(np.int64), 0, max(0, arr.size - 1))
            valid &= arr[idx] != MAT_LAVA
        except Exception:
            pass

    if not valid.any():
        return None
    score = np.where(valid, score, -1e9)
    best = int(np.argmax(score))
    if score[best] <= -1e8:
        return None
    bx = float(xs[best])
    return bx, float(ground_y(world, bx)), {}


def find_chasm(world: Any) -> tuple[float, float] | None:
    """(x0, x1) of a bridgeable gap, or None. Cached briefly - terrain moves."""
    now = world_now(world)
    try:
        cached_t = float(getattr(world, "_bhv_chasm_t", -1e9))
        if 0.0 <= now - cached_t < CHASM_RECHECK:
            return getattr(world, "_bhv_chasm", None)
    except Exception:
        pass
    span = _compute_chasm(world)
    try:
        setattr(world, "_bhv_chasm", span)
        setattr(world, "_bhv_chasm_t", now)
    except Exception:
        pass
    return span


def _compute_chasm(world: Any) -> tuple[float, float] | None:
    terr = getattr(world, "terrain", None)
    if terr is None:
        return None
    declared = getattr(terr, "chasms", None)
    if callable(declared):
        try:
            declared = declared()
        except Exception:
            declared = None
    if isinstance(declared, (list, tuple)) and declared:
        first = declared[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            try:
                return float(first[0]), float(first[1])
            except (TypeError, ValueError):
                pass
    h = getattr(terr, "height", None)
    if h is None:
        return None
    try:
        arr = np.asarray(h, dtype=np.float64).ravel()
    except Exception:
        return None
    n = int(arr.size)
    r = int(CHASM_MAX_W)
    if n < 2 * r + 4:
        return None
    # A chasm column sits well below the highest ground on *both* sides within
    # a bridge-span reach. Comparing against rims (not a smoothed baseline)
    # keeps wide gaps detectable - a rolling mean dilutes them away.
    try:
        padded = np.pad(arr, r, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, r)
        left_rim = windows[:n].min(axis=1)
        right_rim = windows[r + 1: r + 1 + n].min(axis=1)
    except Exception:
        return None
    if left_rim.size != n or right_rim.size != n:
        return None
    mask = (arr - np.maximum(left_rim, right_rim)) > CHASM_DEPTH
    if not mask.any():
        return None
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(n)
    best: tuple[float, float] | None = None
    best_w = 0.0
    for s0, e0 in zip(starts, ends):
        w = float(e0 - s0)
        if CHASM_MIN_W <= w <= CHASM_MAX_W and w > best_w:
            best_w = w
            best = (float(s0), float(e0))
    return best


# ===========================================================================
#  Roles
# ===========================================================================
def _age_of(agent: Any) -> float | None:
    for name in ("age", "age_s", "age_seconds"):
        v = getattr(agent, name, None)
        if isinstance(v, (int, float)):
            return float(v)
    st = getattr(agent, "state", None)
    if isinstance(st, dict) and isinstance(st.get("age"), (int, float)):
        return float(st["age"])
    return None


def _worker_role(world: Any) -> str:
    """gatherer or builder, whichever the colony is shorter of."""
    gatherers = builders = 0
    for a in alive_agents(world):
        r = _role(a)
        if r == "gatherer":
            gatherers += 1
        elif r == "builder":
            builders += 1
    total = gatherers + builders
    if total == 0:
        return "gatherer"
    return "builder" if (builders / total) < BUILDER_RATIO else "gatherer"


def _set_role(agent: Any, role: str) -> bool:
    if not hasattr(agent, "role"):
        return False
    if getattr(agent, "role", None) == role:
        return False
    try:
        agent.role = role
        return True
    except Exception:
        return False


def assign_roles(world: Any, dt: float) -> None:
    """Mature children, keep exactly one elder, staff the watchtower."""
    agents = alive_agents(world)
    if not agents:
        return
    rng = rng_of(world)
    reg = structures_of(world)

    # --- children grow up ---------------------------------------------------
    for a in agents:
        if _role(a) != "child":
            continue
        age = _age_of(a)
        if age is not None:
            if age < CHILD_MATURE_AGE:
                continue
        elif rng.random() >= max(0.0, dt) / CHILD_MATURE_AGE:
            continue
        role = _worker_role(world)
        if _set_role(a, role):
            chronicle(world, f"{getattr(a, 'name', 'A child')} came of age as a {role}.")
            try:
                a.morale = _clamp01(float(getattr(a, "morale", 0.5)) + 0.15)
            except Exception:
                pass

    adults = [a for a in agents if _role(a) != "child"]
    if not adults:
        return

    # --- the eldest leads ---------------------------------------------------
    if len(adults) >= 3:
        def seniority(a: Any) -> tuple[int, float, int]:
            return (
                int(getattr(a, "generation", 0) or 0),
                -(_age_of(a) or 0.0),
                int(getattr(a, "id", 0) or 0),
            )
        elder = sorted(adults, key=seniority)[0]
        for a in adults:
            if a is not elder and _role(a) == "elder":
                _set_role(a, _worker_role(world))
        if _role(elder) != "elder" and _set_role(elder, "elder"):
            chronicle(world, f"{getattr(elder, 'name', 'Someone')} is now the elder.")
    else:
        for a in adults:
            if _role(a) == "elder":
                _set_role(a, _worker_role(world))

    # --- someone watches from the tower -------------------------------------
    towers = 0
    if reg is not None:
        try:
            towers = reg.count("watchtower", built_only=True)
        except Exception:
            towers = 0
    lookouts = [a for a in adults if _role(a) == "lookout"]
    if towers >= 1:
        if not lookouts:
            pool = [a for a in adults if _role(a) in ("gatherer", "builder")]
            if pool:
                pick = pool[rng.randrange(len(pool))]
                if _set_role(pick, "lookout"):
                    chronicle(world, f"{getattr(pick, 'name', 'Someone')} took the watch.")
        else:
            for extra in lookouts[towers:]:
                _set_role(extra, _worker_role(world))
    else:
        for a in lookouts:
            _set_role(a, _worker_role(world))


if __name__ == "__main__":  # pragma: no cover - headless smoke test
    from dataclasses import dataclass as _dc

    @_dc
    class _Prop:
        id: int
        kind: str
        x: float
        hp: float = 4.0
        alive: bool = True

    class _Terrain:
        def __init__(self) -> None:
            self.height = (600.0 + 40.0 * np.sin(np.arange(RENDER_W) / 180.0)).astype(
                np.float32)
            self.height[700:760] += 90.0          # a chasm to bridge
            self.material = np.zeros(RENDER_W, dtype=np.uint8)

        def ground_y(self, x: float) -> float:
            i = int(max(0, min(RENDER_W - 1, x)))
            return float(self.height[i])

        def slope(self, x: float) -> float:
            i = int(max(1, min(RENDER_W - 2, x)))
            return float(self.height[i + 1] - self.height[i - 1]) * 0.5

    class _Agent:
        _n = 0

        def __init__(self, x: float, role: str = "gatherer", gen: int = 0) -> None:
            _Agent._n += 1
            self.id = _Agent._n
            self.name = f"Stick{self.id}"
            self.color = (200, 200, 200)
            self.x = x
            self.y = 600.0
            self.vx = self.vy = 0.0
            self.facing = 1
            self.hunger = 0.2
            self.fatigue = 0.2
            self.warmth = 0.2
            self.morale = 0.7
            self.carrying = None
            self.carry_qty = 0
            self.role = role
            self.generation = gen
            self.age = 0.0
            self.action: Action | None = None
            self.alive = True
            self.anim_t = 0.0

    class _World:
        def __init__(self) -> None:
            self.terrain = _Terrain()
            self.structures = StructureRegistry()
            self.props = (
                [_Prop(i, "tree", 200.0 + i * 37.0) for i in range(1, 14)]
                + [_Prop(50 + i, "rock", 300.0 + i * 61.0) for i in range(1, 9)]
                + [_Prop(80 + i, "bush", 250.0 + i * 43.0) for i in range(1, 11)]
            )
            self.stockpile: dict[str, int] = {}
            self.agents = [_Agent(560.0), _Agent(600.0, "builder"),
                           _Agent(640.0, "child", 1), _Agent(520.0, "builder")]
            self.world_time = 0.0
            self.scene = SCENE_NIGHT_STORM
            self.rng = random.Random(11)
            self.lines: list[str] = []
            self.speech: list[dict[str, Any]] = []
            self.celebrations: list[dict[str, Any]] = []

        def chronicle(self, text: str) -> None:
            self.lines.append(text)

    w = _World()
    print("chasm:", find_chasm(w))
    dt = 1.0 / 30.0
    ai_every = 15
    tick = 0
    while tick < 30 * 60 * 22:          # 22 sim-minutes
        tick += 1
        w.world_time += dt
        for a in w.agents:
            a.age += dt
            a.hunger = min(1.0, a.hunger + dt * 0.004)
            a.fatigue = min(1.0, a.fatigue + dt * 0.003)
            a.warmth = min(1.0, a.warmth + dt * 0.002)
            if tick % ai_every == 0 or a.action is None or a.action.finished:
                a.action = choose_action(a, w)
            a.action.update(a, w, dt)
        w.structures.update(dt, w)
        if tick % (30 * 120) == 0:
            print(f"  t={w.world_time/60:5.1f}m  stock={w.stockpile}  "
                  f"structs={[(s.kind, round(s.completion(), 2)) for s in w.structures]}")
    print("roles:", [(a.name, a.role) for a in w.agents])
    print("build_queue:", getattr(w, "build_queue", None))
    print("build_needs:", getattr(w, "build_needs", None))
    print("chronicle:")
    for line in w.lines[:24]:
        print("   ", line)
    a0 = w.agents[0]
    sc = score_actions(a0, w)
    print("scores:", {k: round(v, 3) for k, v in sorted(sc.items(), key=lambda kv: -kv[1])})
