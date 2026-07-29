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
    BARRICADE_EDGE_FRAC,
    BARRICADE_MIN_POP,
    CLEANUP_SCORE_MAX,
    CROSSING_MAX_SPAN,
    CROSSING_MIN_DEPTH,
    FARM_FIELD_SIZE,
    LITTER_CLUSTER_FULL,
    LITTER_CLUSTER_MIN,
    LADDER_MAX_W,
    LADDER_MIN_RISE,
    LADDER_MIN_W,
    LADDER_SLOPE,
    MAT_DIRT,
    MAT_GRASS,
    MAT_LAVA,
    MAT_STONE,
    MAX_SLOPE_CLIMB,
    MAX_SLOPE_WALK,
    RENDER_W,
    RES_COOKED,
    RES_FIBRE,
    RES_FOOD,
    RES_GARBAGE,
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
    densest_litter,
    find_prop,
    food_in_store,
    free_litter_cluster,
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
from .structures import (
    CROSSING_KINDS,
    Structure,
    StructureRegistry,
    plan_ladder,
    structure_spec,
)

log = logging.getLogger(__name__)

__all__ = [
    "choose_action",
    "score_actions",
    "update_director",
    "assign_roles",
    "next_build_kind",
    "find_chasm",
    "find_cutoff",
    "plan_crossing",
    "HYSTERESIS_BONUS",
    "emergency_override",
    "ROLE_AFFINITY",
    "ROLES",
]

# ------------------------------------------------------------------ tuning --
HYSTERESIS_BONUS = 0.35     # stickiness for the action already running

#: When an agent is idle and content, "Wander" wins the utility race - and that
#: is exactly the moment it should sometimes do something *cosmetic* instead:
#: a cartwheel, a handstand, a stretch. These are the odds of playing a vignette
#: rather than plain wandering. A second, lower chance applies right after one
#: finishes so acrobatics do not chain forever without a wander in between.
VIGNETTE_CHANCE = 0.55
VIGNETTE_CHANCE_AFTER = 0.20
#: Of the idle beats that do play, this fraction is aimed at the acrobatics band
#: (cartwheels, handstands, flips) rather than the general "various things".
ACROBATIC_BIAS = 0.45
#: Above any of these need levels an agent has better things to do than show off.
VIGNETTE_NEED_CEIL = 0.72
#: Top utility below this counts as "idle": the only things worth doing are
#: low-value chores, so a villager may mess about instead. Set just above the
#: resting floors of Mine/Farm/gather (~0.30-0.50) so a healthy colony has
#: downtime, but any genuine shortfall (which lifts those scores) closes it.
DOWNTIME_PEAK = 0.62

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
#: Seconds between reachability surveys. The terrain's own ``epoch`` invalidates
#: this the instant the ground moves (a mine, a mudslide, a finished bridge), so
#: this interval only covers the things the terrain does not know about - people
#: walking, trees being felled, a stockpile going up.
BARRIER_RECHECK = 6.0
#: How much lower than the near rim the far side may be and still count as
#: something to bridge *to* rather than climb *down* to. Generous, because
#: ``Terrain.stamp_deck`` sizes its end ramps to whatever step is left over.
CROSSING_RIM_TOL = 70.0
#: A bridge shorter than this is not a bridge; use the ladder instead.
CROSSING_MIN_SPAN = 10.0
#: How far the ground between a deck's two ends may rise above the planking
#: before the span is rejected. Small on purpose: the merge keeps whichever
#: surface is higher, so anything much over this is a lump in the middle of the
#: bridge rather than something the bridge crosses.
CROSSING_DECK_CLEAR = 8.0
#: How long the same barrier has to keep blocking the same thing before the
#: colony commits to building. Four founders land scattered across the map and
#: spend the first half-minute walking toward each other, so an instantaneous
#: "somebody is on the far side" is usually just somebody on their way home.
#:
#: Was 45 s, and half of that was doing a job that ``STRANDED_DWELL`` now does
#: better. This is a colony-wide integrator: it cannot tell *which* villager it
#: is confirming, so the only way it could reject a man walking home was to
#: outlast him. Rejecting him per-agent instead leaves this one job - riding out
#: a survey that flickers because the terrain or the headcount moved - and 24 s
#: is plenty for that. The 21 s bought back is 21 s off every crossing on every
#: map, which is the cheapest lives-per-second in the whole planner.
CUTOFF_DWELL = 24.0
#: How fast that confirmation drains again while nothing is cut off. Under 1 so
#: an intermittent split - the colony crossing a lethal gap several times a
#: minute, which is the chasm case exactly - still converges on a decision.
CUTOFF_LEAK = 0.6
#: A barrier-driven ladder may run this far, overriding LADDER_MAX_W. That cap
#: is a taste rule for the general "is there a nice cliff to ladder" survey; a
#: 392 px face that has the colony's wood behind it (seed 42) needs 178 px of
#: run at LADDER_SLOPE and simply cannot be answered inside 130. A ramp the
#: colony cannot build is not more tasteful than a long one.
LADDER_BARRIER_MAX_W = 220
#: A region narrower than this cannot be the colony's home, however many people
#: happen to be standing in it. Roughly a hut plus its spacing: below that there
#: is nowhere to live, so it is somewhere people are stuck, not somewhere they
#: are from.
HOME_MIN_W = 150
#: Walkable ground between two walls narrower than this is not a place, it is
#: the floor of the thing in the way, and the two walls are one obstacle. This
#: is the single most load-bearing number in the crossing planner: ``barriers()``
#: reports one entry per *face* and splits the map at each face's midpoint, so a
#: chasm arrives as two barriers with its own floor labelled as a region between
#: them. Every part of the planner that reasoned about one face at a time got
#: the chasm wrong - asked about the near wall, ``_bridge_pair`` finds the floor
#: 200 px down with no rim to land on and answers "no gap here, use a ladder",
#: so the rim-to-rim deck that is plainly the answer was never once proposed.
#: Measured: seed 10 laddered a cliff 350 px away and left the chasm split, seed
#: 70 built three ladders (one of them *down into* the hole) and left it split,
#: seed 59 built nothing at all.
OBSTACLE_ISLAND_W = 90
#: Widest run of walls that may be merged into one obstacle. Past this the
#: colony is looking at broken country rather than one thing to span, and no
#: crossing it can afford would cover it anyway.
OBSTACLE_MAX_W = 300
#: How long a remembered fall keeps an obstacle marked lethal. Long enough that
#: a crossing has time to go up on the evidence of the death that asked for it,
#: short enough that a wall the colony has since learned to leave alone stops
#: pulling work toward itself.
HAZARD_MEMORY = 420.0
#: How far outside an obstacle's rims a fall is still that obstacle's doing.
#: Generous: people topple *while walking away from* a rim they just slipped on,
#: and the body lands further out than the edge it went over.
HAZARD_SLACK = 45.0
#: Dwell for an obstacle that has already killed, or that is holding somebody
#: down a hole. ``CUTOFF_DWELL`` exists to confirm that a split is real; a wall
#: with a body at the foot of it needs no confirming, and neither does a man
#: standing on a chasm floor. Measured deck times before this ran out to 587 s
#: on seed 10, where all four fall deaths happened first.
CUTOFF_DWELL_URGENT = 4.0
#: How long a villager has to stay on the wrong side before that is stranding
#: rather than a walk. The colony-wide charge cannot tell the difference - it
#: integrates "is *anything* cut off" - so on any map with a wall on it somebody
#: momentarily over an edge reads the same as somebody trapped.
STRANDED_DWELL = 12.0
#: ...and how long *one* villager on his own has to, which is far longer. One
#: man over an edge is the weakest evidence of a split there is - every plateau
#: and valley map produces one in the first minute - and acting on it at the
#: same speed as a divided colony is what put a bridge in front of seed 7777's
#: firepit.
STRANDED_DWELL_LONE = 90.0
#: A villager stuck inside an obstacle - on the chasm floor, at the bottom of
#: his own quarry - is a rescue from the moment he is seen there. There is
#: nothing down there to eat and no way back up.
STRANDED_HOLE_DWELL = 6.0
#: Below this the colony is genuinely short of a resource, and "every tree is
#: across the gap" is a problem rather than an observation. With a full store
#: the far bank is next month's wood, not today's.
CUTOFF_SHORT_QTY = 8
GRAVE_FRESH = 300.0
CELEBRATION_FRESH = 30.0

ROLES: tuple[str, ...] = ("gatherer", "builder", "elder", "child", "lookout")

# per-role appetite for each family of work. `farm` and `mine` are the two
# sustained jobs: gatherers lean to the fields (reliable food), builders to the
# quarry (they are the ones who burn stone), so a normal colony visibly does
# both without either job being a dedicated role of its own.
#
# `cleanup` is sweeping litter to the fire. Nobody's trade, so nobody's 1.00:
# gatherers (already the fetch-and-carry role) take most of it, builders least
# because pulling one off a site is the most expensive hour in the colony, and
# children a fair share because "go and tidy that up" is exactly the job a camp
# gives a child.
ROLE_AFFINITY: dict[str, dict[str, float]] = {
    "gatherer": {"gather": 1.00, "build": 0.55, "social": 0.85, "watch": 0.35,
                 "farm": 0.95, "mine": 0.55, "cleanup": 0.90},
    "builder":  {"gather": 0.60, "build": 1.00, "social": 0.70, "watch": 0.35,
                 "farm": 0.50, "mine": 1.00, "cleanup": 0.40},
    "elder":    {"gather": 0.35, "build": 0.40, "social": 1.00, "watch": 0.55,
                 "farm": 0.55, "mine": 0.30, "cleanup": 0.70},
    "lookout":  {"gather": 0.55, "build": 0.45, "social": 0.60, "watch": 1.00,
                 "farm": 0.45, "mine": 0.55, "cleanup": 0.55},
    "child":    {"gather": 0.30, "build": 0.15, "social": 1.00, "watch": 0.10,
                 "farm": 0.25, "mine": 0.10, "cleanup": 0.75},
}
_DEFAULT_AFFINITY = ROLE_AFFINITY["gatherer"]

#: A crossing outranks everything, including the firepit. It is the only kind of
#: build that is *blocking*: while it is missing, half the colony's ground, its
#: wood, or one of its people is on the far side of a drop that kills anyone who
#: tries it. Nothing else the colony could be doing is worth more than that, and
#: the old ordering - bridge behind a firepit, a stockpile, three huts, two
#: barricades, a wall and a watchtower - meant a chasm map never got one at all.
KIND_PRIORITY: dict[str, float] = {
    "bridge": 1.10,
    "ladder": 1.05,
    "firepit": 1.00,
    "stockpile": 0.92,
    "hut": 0.85,
    "grave": 0.80,
    "barricade": 0.60,
    "wall": 0.58,
    "watchtower": 0.54,
    "totem": 0.40,
}

_COLD_SCENES = (SCENE_BLIZZARD, SCENE_NIGHT_STORM)
_TREE_KINDS = ("tree", "pine", "oak")
_STONE_KINDS = ("rock", "boulder", "stone", "outcrop")
_BUSH_KINDS = ("bush", "berry", "berrybush", "shrub")
#: What a stranded region has to hold before the colony cares, by resource.
_CUTOFF_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (RES_WOOD, _TREE_KINDS),
    (RES_STONE, _STONE_KINDS),
    (RES_FOOD, _BUSH_KINDS),
)


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
    # Claim-aware: a prop another villager has already reserved does not count as
    # available here, so an agent will not pick GatherWood only to find every
    # tree taken - it scores the job zero and does something else instead.
    if _find_target_prop(world, _TREE_KINDS, ax, agent) is None:
        s["GatherWood"] = 0.0
    if _find_target_prop(world, ("rock", "boulder", "stone", "outcrop"), ax, agent) is None:
        s["GatherStone"] = 0.0
    if _find_target_prop(world, ("bush", "berry", "berrybush", "shrub"), ax, agent) is None:
        s["ForageBerries"] = 0.0

    # ------------------------------------------------------------ farm ------
    # Farming is the RELIABLE food source. A tended field beats stripping wild
    # bushes, so once a field exists (or there is ground to break) Farm is
    # weighted a touch above ForageBerries. Anyone will bring in a ripe crop;
    # the standing work of tilling leans on the `farm` affinity.
    crops_near, ripe_ready, tillable = _farm_feasible(world)
    if ripe_ready or tillable:
        field_bonus = 0.12 if crops_near > 0 else 0.0
        ripe_bonus = 0.15 if ripe_ready else 0.0
        s["Farm"] = _clamp01(
            aff.get("farm", 0.5) * (0.18 + 0.82 * food_short + field_bonus)
            + ripe_bonus + hunger * 0.12
        )
    else:
        s["Farm"] = 0.0

    # ------------------------------------------------------------ mine ------
    # A sustained dig, complementing GatherStone. Rises as stored stone falls or
    # a build is blocked on stone, and is preferred when loose rocks are scarce.
    stored_stone = stock_qty(world, RES_STONE)
    stone_target = float(max(6, pop * 3))
    stone_low = _clamp01((stone_target - stored_stone) / stone_target)
    if _mineable_near(world, ax, agent):
        loose_rock = _find_target_prop(world, ("rock",), ax, agent) is not None
        scarce_bonus = 0.0 if loose_rock else 0.20
        # A higher floor than most jobs on purpose: a colony keeps a quarry
        # ticking over even when the stores are full, so mining is a visible
        # ongoing activity rather than only a stone-emergency response.
        s["Mine"] = _clamp01(
            aff.get("mine", 0.5)
            * (0.30 + 0.70 * max(stone_low, stone_urg) + scarce_bonus)
        )
    else:
        s["Mine"] = 0.0

    # ------------------------------------------------------------ haul ------
    # Garbage is excluded: the stockpile refuses it (see actions.stock_add), so
    # winning this with an armful of sweepings would mean walking to the store
    # to accomplish nothing. A sweeper interrupted mid-load drops it instead -
    # `_c_clean` does that on the way out.
    s["HaulToStockpile"] = (
        _clamp01(0.42 + 0.42 * min(1.0, carry_qty / float(CARRY_CAP)))
        if (carry_qty > 0 and carrying and carrying != RES_GARBAGE) else 0.0
    )

    # ------------------------------------------------------------ cook ------
    raw = stock_qty(world, RES_FOOD)
    if fire is not None and raw >= 3:
        want = 0.30 + (0.25 if stock_qty(world, RES_COOKED) == 0 else 0.0)
        s["CookFood"] = _clamp01(aff["gather"] * want)
    else:
        s["CookFood"] = 0.0

    # --------------------------------------------------------- cleanup ------
    # Sweeping litter to the fire. Everything about this score is deliberately
    # meek:
    #   * it needs a DENSE pile, never one stray item (`densest_litter` returns
    #     None below LITTER_CLUSTER_MIN, so the job simply does not exist);
    #   * it needs somewhere to burn it, so no firepit means no score;
    #   * it needs empty hands - or hands already full of rubbish - so it never
    #     competes with a live haul of something the colony actually wants. The
    #     second half of that is not a nicety: gating on `carry_qty <= 0` alone
    #     zeroes the score the instant a sweeper picks his first piece up, and a
    #     zero score is dropped from the ranking entirely, so *any* re-score mid
    #     sweep guaranteed he abandoned the job and put the load back down.
    #     Measured: 74 loads shed against 63 delivered before this line.
    #   * it is capped at CLEANUP_SCORE_MAX whatever the density, which is what
    #     keeps it structurally under food, shelter, defence and building rather
    #     than merely under them at today's numbers;
    #   * and it is switched off entirely while the larder is genuinely low.
    #     Tidying the camp is the definition of a job that can wait until the
    #     colony is fed.
    s["CleanLitter"] = 0.0
    hands_ok = carry_qty <= 0 or carrying == RES_GARBAGE
    if fire is not None and hands_ok and food_short < 0.75:
        pile = densest_litter(world)
        if pile is not None:
            span = max(1.0, float(LITTER_CLUSTER_FULL - LITTER_CLUSTER_MIN))
            dense = _clamp01((pile[1] - LITTER_CLUSTER_MIN) / span)
            s["CleanLitter"] = min(
                CLEANUP_SCORE_MAX,
                aff.get("cleanup", 0.5) * (0.14 + 0.46 * dense),
            )

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
    # Wildlife scoring lives in combat_actions so the fight/flee/craft logic
    # sits with the actions it drives. Merged last so a wolf at the door can
    # outrank anything the colony was calmly getting on with.
    try:
        from .combat_actions import score_combat
        s.update(score_combat(agent, world))
    except Exception:
        log.debug("combat scoring unavailable", exc_info=True)
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


def _find_target_prop(world: Any, kinds: tuple[str, ...], x: float,
                      agent: Any = None) -> Any | None:
    claimant = getattr(agent, "id", None) if agent is not None else None
    return find_prop(world, kinds, x, claimant=claimant)


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


def _ripe_crop_exists(world: Any) -> bool:
    for p in props_of(world):
        if not prop_alive(p):
            continue
        k = getattr(p, "kind", None) or getattr(p, "type", None) or ""
        if str(k).lower() != "crop":
            continue
        st = getattr(p, "state", None)
        if isinstance(st, dict) and st.get("ripe"):
            return True
    return False


def _tillable_near(world: Any, center: float, radius: float = 260.0) -> bool:
    """True if there is gentle grass/dirt near the colony to break new ground."""
    mat_at = getattr(getattr(world, "terrain", None), "material_at", None)
    if not callable(mat_at):
        return False
    x = center - radius
    while x <= center + radius:
        cx = max(4.0, min(float(RENDER_W - 4), x))
        try:
            if int(mat_at(cx)) in (MAT_GRASS, MAT_DIRT) and \
                    abs(slope_at(world, cx)) < MAX_SLOPE_WALK * 0.7:
                return True
        except Exception:
            pass
        x += 14.0
    return False


def _farm_feasible(world: Any) -> tuple[int, bool, bool]:
    """(#crops near the colony, a ripe crop is ready, ground can be tilled).

    Cached for a few seconds: it walks the prop list and samples the terrain,
    and score_actions runs per agent every AI tick.
    """
    now = world_now(world)
    try:
        t = float(getattr(world, "_bhv_farm_t", -1e9))
        if 0.0 <= now - t < 5.0:
            cached = getattr(world, "_bhv_farm", None)
            if isinstance(cached, tuple) and len(cached) == 3:
                return cached  # type: ignore[return-value]
    except Exception:
        pass
    center = colony_center(world)
    crops = _count_props(world, ("crop",))
    ripe = _ripe_crop_exists(world)
    tillable = _tillable_near(world, center) if crops < FARM_FIELD_SIZE else False
    val = (crops, ripe, tillable)
    try:
        setattr(world, "_bhv_farm", val)
        setattr(world, "_bhv_farm_t", now)
    except Exception:
        pass
    return val


def _stone_terrain_exists(world: Any) -> bool:
    """Any MAT_STONE column at all - cached, the check scans the whole map."""
    now = world_now(world)
    try:
        t = float(getattr(world, "_bhv_mine_t", -1e9))
        if 0.0 <= now - t < 6.0:
            return bool(getattr(world, "_bhv_mine", False))
    except Exception:
        pass
    mat = getattr(getattr(world, "terrain", None), "material", None)
    try:
        found = bool(np.any(np.asarray(mat) == MAT_STONE))
    except Exception:
        found = False
    try:
        setattr(world, "_bhv_mine", found)
        setattr(world, "_bhv_mine_t", now)
    except Exception:
        pass
    return found


def _mineable_near(world: Any, x: float, agent: Any = None) -> bool:
    """An unclaimed boulder within reach, or failing that any stone terrain to
    dig. Ground stone is shared freely - two diggers at one seam barely read -
    but a boulder another miner has claimed no longer counts as mineable here."""
    if _find_target_prop(world, ("boulder",), x, agent) is not None:
        return True
    return _stone_terrain_exists(world)


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


def _content_enough(agent: Any) -> bool:
    """True when an agent has no pressing need and could plausibly mess about.

    A hungry, exhausted, freezing or panicking villager does not do cartwheels;
    it deals with the thing that is wrong. Danger never reaches here anyway - a
    real threat makes FleeFrom win outright - but the needs are checked so an
    agent that is merely *coping* still keeps its head down."""
    try:
        if float(getattr(agent, "panic", 0.0) or 0.0) > 0.0:
            return False
    except (TypeError, ValueError):
        pass
    return (_need(agent, "hunger") < VIGNETTE_NEED_CEIL
            and _need(agent, "fatigue") < VIGNETTE_NEED_CEIL
            and _need(agent, "warmth") < VIGNETTE_NEED_CEIL)


def _maybe_vignette(
    agent: Any, world: Any, scores: dict[str, float], rng: Any
) -> Action | None:
    """A cosmetic vignette for a villager between jobs, or None.

    This is the single seam that turns the (previously dormant) vignette engine
    on. Agents are almost never "idle" in the strict sense - Mine, Farm and the
    gather chores all carry a positive floor, so "Wander" essentially never wins
    the utility race. What *does* happen is stretches where the best thing worth
    doing is low-value busywork (topping up a full stockpile). That is the real
    idle moment, and it is defined here as "nothing scored above
    :data:`DOWNTIME_PEAK`" - so when food or stone actually run short the peak
    climbs, the gate closes, and everyone gets back to work on their own."""
    try:
        if not _content_enough(agent):
            return None
        peak = max(scores.values()) if scores else 0.0
        if peak >= DOWNTIME_PEAK:
            return None
        prev = getattr(getattr(agent, "action", None), "kind", "") or ""
        chance = VIGNETTE_CHANCE_AFTER if prev == "Vignette" else VIGNETTE_CHANCE
        if rng.random() >= chance:
            return None
        from .vignettes import ACROBATIC_POSES, make_vignette_action
        # The user asked for acrobatics specifically, so bias toward them - but
        # they are only ~5% of the 520-vignette pool, so ask for one directly a
        # good fraction of the time and let the rest be the "various things".
        if rng.random() < ACROBATIC_BIAS:
            acro = make_vignette_action(agent, world, rng, poses=ACROBATIC_POSES)
            if acro is not None:
                return acro
        return make_vignette_action(agent, world, rng)
    except Exception:
        log.debug("idle vignette pick failed", exc_info=True)
        return None


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


def _mk_clean(agent: Any, world: Any) -> Action | None:
    """Sweep the densest pile nobody else has taken.

    Returns None when there is nothing dense enough or every heap already has
    somebody on it, so a stale score can never launch a job with no premise -
    the villager drops straight through to the next-best thing instead of
    spending a decision cycle on an action that fails on its first update.
    """
    if getattr(agent, "carrying", None) == RES_GARBAGE and \
            int(getattr(agent, "carry_qty", 0) or 0) > 0:
        # Already holding a swept load. Whether any heap is free is beside the
        # point - he has an armful of rubbish and this is the only action in the
        # colony that knows where to put it. The handler's start phase routes
        # him straight to the fire.
        return make_action("CleanLitter")
    pile = free_litter_cluster(world, getattr(agent, "id", None))
    if pile is None:
        return None
    return make_action("CleanLitter", cx=float(pile[0]))


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
    "Farm": _mk_simple("Farm"),
    "Mine": _mk_simple("Mine"),
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
    "CleanLitter": _mk_clean,
}


def emergency_override(agent: Any, world: Any) -> bool:
    """True only when something is urgent enough to abandon a job mid-way.

    The scheduler leaves in-flight actions alone, so this is the sole way an
    agent drops what it is doing. Keep the bar high: anything that returns True
    too readily reintroduces the walk-there-change-mind-walk-back shuffle this
    exists to prevent.
    """
    try:
        cur = getattr(agent, "action", None)
        kind = getattr(cur, "kind", "") if cur is not None else ""

        # Already responding to the emergency.
        if kind in ("FleeFrom", "Panic", "Eat", "Sleep"):
            return False

        if float(getattr(agent, "panic", 0.0) or 0.0) > 0.0:
            return True
        if float(getattr(agent, "hunger", 0.0)) > 0.92:
            return True
        if float(getattr(agent, "fatigue", 0.0)) > 0.95:
            return True
        if float(getattr(agent, "warmth", 0.0)) > 0.93:
            return True

        # Standing in something that is actively going to kill us.
        try:
            ev = getattr(world, "events", None)
            level = getattr(ev, "water_level", None)
            if level and float(getattr(agent, "y", 0.0)) > float(level):
                return True
            for prop in world.props.burning():
                if abs(float(prop.x) - float(agent.x)) < 90.0:
                    return True
        except Exception:
            pass
        return False
    except Exception:
        return False


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

        # Between jobs, with nothing pressing: sometimes mess about (a cartwheel,
        # a handstand, a stretch) instead of picking up the next low-value chore.
        # Only when the previous action ran to completion - never as an emergency
        # interrupt (those arrive here with cur still live) - so a vignette never
        # pre-empts real work, only fills the gaps between it.
        if not cur_live:
            vig = _maybe_vignette(agent, world, scores, rng)
            if vig is not None:
                if cur is not None and cur is not vig:
                    try:
                        cur.abandon(agent, world)
                    except Exception:
                        pass
                return vig

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
                try:
                    from .combat_actions import make_combat_action
                    act = make_combat_action(kind, agent, world)
                except Exception:
                    act = None
                if act is None:
                    continue
                if cur is not None and cur is not act:
                    try:
                        cur.abandon(agent, world)
                    except Exception:
                        pass
                return act
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

    # Falls and stranding are histories, and the 2 s colony pass is the only
    # cadence that samples them often enough to be one. `find_cutoff` keeps them
    # too, but it is cached for six seconds and skipped entirely on the maps
    # with nothing in the way, and a body reaped into a grave before anybody
    # looked is evidence gone for good.
    _watch_colony(world)

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
    """What the colony wants next: a crossing if one is blocking, then
    firepit -> stockpile -> hut -> more huts -> wall -> watchtower ->
    bridge (if a chasm exists) -> totem at a milestone."""
    reg = reg if reg is not None else structures_of(world)
    if reg is None:
        return None
    pop = len(alive_agents(world))

    def built(kind: str) -> int:
        try:
            return reg.count(kind, built_only=True)
        except Exception:
            return 0

    # Blocking work first, ahead of even the firepit. Everything below this line
    # is expansion - nicer, warmer, bigger - and none of it is worth anything
    # while a villager is stranded on the far side of a 300 px drop or the
    # colony's only remaining wood is across one.
    #
    # What changed is not this ordering but what reaches it. `plan_crossing`
    # used to answer "yes" for any map with a wall on it and somebody
    # momentarily the other side of it, and preempting the firepit for that is
    # how seed 7777 - a plateau, nobody trapped, nobody dying - bought a ladder
    # and a bridge and finished a hut short of the colony that built neither.
    # Deferring the crossing behind fire, stores and a roof was tried as the fix
    # and measured worse across 44 seeds (13 falls against 16, and every
    # crossing 200 s later), because on the maps that do need one the wait is
    # paid in people. Gating what counts as blocking is the fix; the ordering
    # was right all along. This still returns None on a map with no barrier,
    # which is almost every map, so the ordinary order below is untouched.
    blocking = plan_crossing(world, reg)
    if blocking is not None:
        return str(blocking["kind"])

    if built("firepit") < 1:
        return "firepit"
    if built("stockpile") < 1:
        return "stockpile"
    if built("hut") < 1:
        return "hut"
    # Edge defence is interleaved with expansion, not deferred behind it, so it
    # goes up *as the colony progresses* rather than after five huts. First
    # shelter, then the first spiked barricade on one edge, then keep growing,
    # and the second edge once the settlement is a bit bigger. Placement (which
    # edge) is decided in pick_site; here we only ask for the next one.
    if pop >= BARRICADE_MIN_POP and built("barricade") < 1:
        return "barricade"
    if built("hut") < 3:
        return "hut"
    if pop >= BARRICADE_MIN_POP and built("barricade") < 2:
        return "barricade"
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
    kind = next_build_kind(world, reg)
    if kind is None:
        return
    # A blocking crossing is allowed one slot over the concurrency cap. Without
    # it a barrier that appears while two ordinary sites are already open - a
    # miner opening a pit, which is exactly the case this exists for - waits for
    # a hut to finish before anyone even stakes the way out.
    cap = MAX_CONCURRENT_SITES + (1 if kind in CROSSING_KINDS else 0)
    if len(pending) >= cap:
        return
    if any(s.kind == kind for s in pending):
        return
    site = pick_site(world, reg, kind)
    if site is None:
        return
    x, y, extra = site
    s = reg.create(kind, x, y, rng=rng_of(world), state=extra or None)
    chronicle(world, _stake_line(world, kind))
    log.debug("director staked %s at %.0f", kind, x)


#: Why a crossing is going up, in the chronicle's voice. Keyed by the reason
#: :func:`find_cutoff` gave, so the log says what the colony noticed rather than
#: what the director decided.
_CUTOFF_LINES: dict[str, str] = {
    "stranded": "One of theirs is stranded on the far side; the colony staked out a {kind}.",
    "falls": "Too many have gone over that edge; the colony staked out a {kind}.",
    "stockpile": "Cut off from the stores, the colony staked out a {kind}.",
    "firepit": "Cut off from the fire, the colony staked out a {kind}.",
    RES_WOOD: "Every tree left is across the gap; the colony staked out a {kind}.",
    RES_STONE: "The stone is all on the far side; the colony staked out a {kind}.",
    RES_FOOD: "The forage is all across the gap; the colony staked out a {kind}.",
}


def _stake_line(world: Any, kind: str) -> str:
    """Chronicle line for a freshly staked site."""
    if kind in CROSSING_KINDS:
        try:
            cut = find_cutoff(world)
            if cut is not None:
                # The plan's own reason, not the survey's: the wall being staked
                # out is the one the colony ranked worst, which is not always
                # the candidate that made the survey fire.
                plan = cut.get("plan") or {}
                tmpl = _CUTOFF_LINES.get(str(plan.get("reason")
                                             or cut.get("reason")))
                if tmpl:
                    return tmpl.format(kind=kind)
        except Exception:
            pass
    return f"The colony staked out a new {kind}."


def pick_site(
    world: Any, reg: StructureRegistry, kind: str
) -> tuple[float, float, dict[str, Any]] | None:
    """Choose where to put a `kind`. Returns (x, y, extra_state) or None."""
    spec = structure_spec(kind)
    center = colony_center(world)

    if kind in CROSSING_KINDS:
        # The reachability survey is the authority: it knows which side the
        # colony is on and which barrier is the one in its way.
        plan = plan_crossing(world, reg)
        if plan is not None and plan["kind"] == kind:
            return float(plan["x"]), float(plan["y"]), dict(plan["extra"])
        if kind != "bridge":
            return None
        # Fallback for the legacy "there is a chasm, build a bridge eventually"
        # entry in the build order, which fires with nothing actually blocked.
        span = find_chasm(world)
        if span is None:
            return None
        x0, x1 = span
        rim = min(ground_y(world, x0 - 4.0), ground_y(world, x1 + 4.0))
        # Anchored at the near rim rather than mid-span: `x` is the spot a
        # builder walks to, and mid-span is open air until the deck is stamped.
        # It used to hand back 0.5 * (x0 + x1), which sent every builder over
        # the edge of the gap they were there to bridge.
        near = x0 if abs(x0 - center) <= abs(x1 - center) else x1
        anchor = float(np.clip(near + (-4.0 if near == x0 else 4.0),
                               4.0, float(RENDER_W - 5)))
        return anchor, rim, {"w": (x1 - x0) + 18.0, "span": [float(x0), float(x1)]}

    # A barricade does not belong near the colony centre like everything else -
    # it belongs out at the edge the animals arrive from. Only the x-choice
    # differs; the caller still creates it as a normal unbuilt Structure.
    if kind == "barricade":
        return _barricade_site(world, reg)

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


def _is_cliff(world: Any, x: float) -> bool:
    """True where a barricade cannot stand: a fall-risk cliff face.

    Prefers the terrain's own ``is_cliff`` (slope beyond MAX_SLOPE_CLIMB) and
    falls back to the shared ``slope_at`` helper so a stub terrain without the
    method still gets a sane answer instead of raising."""
    terr = getattr(world, "terrain", None)
    fn = getattr(terr, "is_cliff", None)
    if callable(fn):
        try:
            return bool(fn(float(x)))
        except Exception:
            pass
    return abs(slope_at(world, float(x))) > MAX_SLOPE_CLIMB


def _barricade_site(
    world: Any, reg: StructureRegistry
) -> tuple[float, float, dict[str, Any]] | None:
    """A site for a barricade, hard against whichever edge has none yet.

    The animals cross in at ``x = 0`` and ``x = RENDER_W``; a barricade wants to
    sit in the band within ``BARRICADE_EDGE_FRAC * RENDER_W`` of one of them, on
    non-cliff ground, as close to the edge as the terrain allows so it bites an
    incursion the moment it lands. Left edge is filled first, then the right.
    Returns ``(x, y, {})`` in the same shape as :func:`pick_site`, or None if
    both edges are covered or the band is nothing but cliff.
    """
    band = max(24.0, BARRICADE_EDGE_FRAC * float(RENDER_W))
    mid = RENDER_W * 0.5

    # Which edges already hold a barricade (built or still going up)?
    left_has = right_has = False
    try:
        for s in reg.of_kind("barricade"):
            if float(s.x) <= mid:
                left_has = True
            else:
                right_has = True
    except Exception:
        pass

    # Candidate edges, left first, that still need one. Trying both (rather than
    # committing to the left) means an all-cliff left band falls through to the
    # right instead of stalling the build order forever.
    candidates: list[tuple[float, float, float]] = []
    if not left_has:
        candidates.append((6.0, band, 0.0))
    if not right_has:
        candidates.append((float(RENDER_W) - band, float(RENDER_W) - 6.0,
                           float(RENDER_W)))
    if not candidates:
        return None                      # both edges covered - nothing to do

    for lo, hi, edge in candidates:
        lo = max(4.0, lo)
        hi = min(float(RENDER_W - 4), hi)
        if hi <= lo:
            continue
        # Walk the band; keep the non-cliff column closest to the edge and,
        # among near-equals, flattest. Closeness dominates so the spikes really
        # do straddle the entry point rather than drifting toward the camp.
        best_x: float | None = None
        best_score = float("-inf")
        x = lo
        while x <= hi:
            cx = float(x)
            if not _is_cliff(world, cx):
                score = -abs(cx - edge) / band - abs(slope_at(world, cx)) * 0.6
                if score > best_score:
                    best_score = score
                    best_x = cx
            x += 6.0
        if best_x is not None:
            return best_x, float(ground_y(world, best_x)), {}

    return None


# ===========================================================================
#  Reachability: has the colony been cut off, and what would fix it?
# ===========================================================================
#
# Three steps, deliberately separated so each can be tested on its own:
#
#   _reach()        ask the terrain which stretches of map are connected
#   find_cutoff()   decide whether the split actually costs the colony anything
#   plan_crossing() turn that into a bridge or a ladder the director can stake
#
# The terrain owns the graph and caches it against its own ``epoch``, so digging
# a pit, a mudslide or a finished deck all re-survey exactly once. Everything in
# here is on the director's 2 s cadence, never per agent per tick.


def _reach(world: Any) -> tuple[Any, tuple[tuple[int, int, float], ...]]:
    """``(labels, barriers)`` from the terrain, or ``(None, ())`` if it cannot.

    Duck-typed on purpose: the module-level smoke test and several unit stubs
    hand ``choose_action`` a terrain with nothing but ``height`` and
    ``ground_y``, and a colony must keep running against one of those.
    """
    terr = getattr(world, "terrain", None)
    regions = getattr(terr, "regions", None)
    barriers = getattr(terr, "barriers", None)
    if not callable(regions) or not callable(barriers):
        return None, ()
    try:
        lab = np.asarray(regions())
        bars = tuple(barriers())
    except Exception:
        return None, ()
    if lab.ndim != 1 or lab.size < 8:
        return None, ()
    return lab, bars


def _region_of(lab: Any, x: Any) -> int:
    """Region id under ``x``, clamped to the map. ``-1`` for a nonsense ``x``."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return -1
    if not np.isfinite(xf):
        return -1
    i = int(np.clip(int(xf), 0, int(lab.size) - 1))
    return int(lab[i])


def _surface_of(world: Any) -> Any:
    """The terrain's effective surface as an array, or ``None``."""
    terr = getattr(world, "terrain", None)
    fn = getattr(terr, "surface", None)
    if callable(fn):
        try:
            arr = np.asarray(fn(), dtype=np.float64)
            if arr.ndim == 1 and arr.size >= 8:
                return arr
        except Exception:
            pass
    h = getattr(terr, "height", None)
    try:
        arr = np.asarray(h, dtype=np.float64).ravel()
    except Exception:
        return None
    return arr if arr.size >= 8 else None


# ------------------------------------------------------- what is in the way --


def _obstacles(
    world: Any, lab: Any, bars: tuple[tuple[int, int, float], ...]
) -> tuple[dict[str, Any], ...]:
    """The walls, grouped into the things a colony actually has to cross.

    ``Terrain.barriers()`` reports one entry per *face* and ``_build_reach``
    splits the map at each face's midpoint, which makes barrier ``i`` the wall
    between region ``i`` and region ``i+1``. That is exact and it is also why a
    chasm cannot be answered one barrier at a time: it arrives as two barriers
    with its own floor labelled as a region in between, so the near wall taken
    alone has a 200 px drop on its far side and no rim to land a deck on.

    Two walls are one obstacle when the ground between them is too narrow to be
    anywhere (nobody lives on 44 px of chasm floor) and the pair is still short
    enough that a single crossing could cover it.

    Each entry is ``{"lo", "hi", "relief", "left", "right", "inner"}``: the
    outer rims, the two regions the obstacle separates, and the region ids it
    swallows. Cached against the terrain's ``epoch`` - this is pure geometry.
    """
    epoch = int(getattr(getattr(world, "terrain", None), "epoch", 0) or 0)
    try:
        if int(getattr(world, "_bhv_obs_epoch", -1)) == epoch:
            cached = getattr(world, "_bhv_obs", None)
            if cached is not None:
                return cached
    except Exception:
        pass

    groups: list[list[int]] = []
    for i in range(len(bars)):
        if groups:
            first = groups[-1][0]
            # `bars[i][0] - bars[i - 1][1]` is the walkable ground between the
            # two faces - rim to rim - which is exactly the "is there anywhere
            # to stand between them" question.
            island = int(bars[i][0]) - int(bars[i - 1][1])
            reach = int(bars[i][1]) - int(bars[first][0])
            if island < OBSTACLE_ISLAND_W and reach <= OBSTACLE_MAX_W:
                groups[-1].append(i)
                continue
        groups.append([i])

    out: list[dict[str, Any]] = []
    for g in groups:
        lo = int(bars[g[0]][0])
        hi = int(bars[g[-1]][1])
        out.append({
            "lo": lo,
            "hi": hi,
            "relief": max(float(bars[k][2]) for k in g),
            # Barrier k divides region k from region k+1, so a group running
            # k0..kn divides region k0 from region kn+1 and eats the rest.
            "left": int(g[0]),
            "right": int(g[-1]) + 1,
            "inner": frozenset(range(int(g[0]) + 1, int(g[-1]) + 1)),
            "mid": 0.5 * (lo + hi),
        })
    packed = tuple(out)
    for name, value in (("_bhv_obs", packed), ("_bhv_obs_epoch", epoch)):
        try:
            setattr(world, name, value)
        except Exception:
            pass
    return packed


def _obstacle_holding(
    obs: tuple[dict[str, Any], ...], region: int
) -> dict[str, Any] | None:
    """The obstacle whose interior swallowed ``region``, if any."""
    for o in obs:
        if region in o["inner"]:
            return o
    return None


def _must_cross(o: dict[str, Any], home: int, region: int) -> bool:
    """Does getting from ``home`` to ``region`` mean getting past ``o``?

    Region ids increase left to right and an obstacle separates everything at
    or left of ``left`` from everything at or right of ``right``, so this is
    two integer comparisons rather than a graph walk.
    """
    if region in o["inner"]:
        return True
    if home in o["inner"]:
        return True
    if home <= o["left"] and region >= o["right"]:
        return True
    return home >= o["right"] and region <= o["left"]


# ----------------------------------------------------- who is it hurting? --


def _watch_colony(world: Any) -> None:
    """Per-pass bookkeeping the crossing planner reads: falls, and stranding.

    Both are histories rather than snapshots, and both are why the planner can
    now tell a wall that is killing people from a wall that merely exists.
    Deliberately driven off corpses rather than a death hook: ``behavior`` has
    no business installing a module-level callback that would outlive the world
    that wanted it, and a body lies where it landed until somebody digs a grave,
    which is many director passes.

    Never raises. This runs on the 2 s colony pass, forever, unattended.
    """
    try:
        now = world_now(world)
        roster = _roster(world)
        lab, _bars = _reach(world)

        # --- falls. Keyed by agent id so a corpse is counted once, not once
        # per pass for the several minutes it lies there.
        seen = getattr(world, "_bhv_hazard_seen", None)
        if not isinstance(seen, set):
            seen = set()
        hazards = list(getattr(world, "_bhv_hazard", ()) or ())
        for a in roster:
            if getattr(a, "alive", True):
                continue
            if str(getattr(a, "death_cause", "")) != "fall":
                continue
            key = int(getattr(a, "id", 0) or 0) or id(a)
            if key in seen:
                continue
            seen.add(key)
            try:
                hazards.append((float(a.x), now))
            except (TypeError, ValueError):
                pass
        hazards = [(hx, ht) for hx, ht in hazards if now - ht <= HAZARD_MEMORY]
        # Bounded, because this runs for hours: a body leaves the roster when it
        # is buried and cannot come back, so an id no longer on it is an id that
        # can never be double-counted again.
        live_ids = {int(getattr(a, "id", 0) or 0) or id(a) for a in roster}
        seen &= live_ids

        # --- stranding. How long each living agent has been *continuously*
        # outside the region holding the colony. A snapshot cannot tell a man
        # trapped across a chasm from a man three steps down a slope on his way
        # home, and treating the second as the first is what had seed 7777
        # building crossings for a colony that was never divided.
        strand = getattr(world, "_bhv_strand", None)
        if not isinstance(strand, dict):
            strand = {}
        fresh: dict[int, tuple[int, float]] = {}
        if lab is not None:
            for a in roster:
                if not getattr(a, "alive", True):
                    continue
                key = int(getattr(a, "id", 0) or 0) or id(a)
                r = _region_of(lab, getattr(a, "x", None))
                was = strand.get(key)
                # Region ids are renumbered whenever a crossing joins two
                # pieces of map, so "same region as last time" is the wrong
                # test after an epoch bump; the clock restarting on a genuine
                # move is the conservative direction to be wrong in.
                same = was is not None and was[0] == r
                fresh[key] = (r, float(was[1]) if same else now)
        strand = fresh

        for name, value in (("_bhv_hazard", hazards), ("_bhv_hazard_seen", seen),
                            ("_bhv_strand", strand)):
            try:
                setattr(world, name, value)
            except Exception:
                pass
    except Exception:
        log.debug("colony watch failed", exc_info=True)


def _roster(world: Any) -> list[Any]:
    """Every agent including the dead - ``alive_agents`` drops the evidence."""
    pop = getattr(world, "population", None)
    roster = getattr(pop, "agents", None) if pop is not None else None
    if isinstance(roster, (list, tuple)):
        return list(roster)
    try:
        return list(roster) if roster is not None else list(alive_agents(world))
    except Exception:
        return list(alive_agents(world))


def _hazard_hits(world: Any, o: dict[str, Any]) -> int:
    """How many people this obstacle has killed lately."""
    lo = float(o["lo"]) - HAZARD_SLACK
    hi = float(o["hi"]) + HAZARD_SLACK
    n = 0
    for hx, _ht in getattr(world, "_bhv_hazard", ()) or ():
        if lo <= hx <= hi:
            n += 1
    return n


def _stranded_for(world: Any, agent: Any) -> float:
    """Seconds this agent has been standing in the region it is standing in."""
    now = world_now(world)
    strand = getattr(world, "_bhv_strand", None)
    if not isinstance(strand, dict):
        return 0.0
    key = int(getattr(agent, "id", 0) or 0) or id(agent)
    rec = strand.get(key)
    return max(0.0, now - float(rec[1])) if rec else 0.0


def find_cutoff(world: Any) -> dict[str, Any] | None:
    """What the colony has been cut off from, or ``None`` if nothing.

    Returns ``{"reason", "home", "target", "target_x", "plan", "charge"}``, where
    ``plan`` is the sited crossing that would fix it. Cached against the terrain's
    ``epoch`` *and* a short wall-clock interval, because the answer also moves
    when nothing about the ground does: people walk, trees get felled, a
    stockpile finishes.
    """
    now = world_now(world)
    epoch = int(getattr(getattr(world, "terrain", None), "epoch", 0) or 0)
    try:
        if (int(getattr(world, "_bhv_cut_epoch", -1)) == epoch
                and 0.0 <= now - float(getattr(world, "_bhv_cut_t", -1e9))
                < BARRIER_RECHECK):
            return getattr(world, "_bhv_cut", None)
    except Exception:
        pass
    # Ahead of the survey, not inside it: the survey reads these histories and
    # a stub world that only ever calls find_cutoff still gets them kept.
    _watch_colony(world)
    try:
        found = _compute_cutoff(world)
        plan = None if found is None else _crossing_geometry(world, found)
    except Exception:
        log.debug("cutoff survey failed", exc_info=True)
        found, plan = None, None
    if found is not None:
        found["plan"] = plan

    charge = _charge(world, found is not None, now)
    if found is not None:
        found["charge"] = charge

    for name, value in (("_bhv_cut", found), ("_bhv_cut_t", now),
                        ("_bhv_cut_epoch", epoch)):
        try:
            setattr(world, name, value)
        except Exception:
            pass
    return found


def _charge(world: Any, wanted: bool, now: float) -> float:
    """How chronically split the colony is, in seconds of confirmed trouble.

    A leaky integrator over "is anything cut off right now", *not* a streak and
    deliberately *not* keyed on which thing or which region. Both of the tighter
    forms were written first and both were measured failing on the one map that
    needs this most. On seed 5 the colony straddles a 300 px chasm and shuttles
    across it all run: the ``(reason, home, target)`` signature flipped between
    four values inside two minutes, and even keying on the proposed geometry
    flipped, because half the surveys name the chasm *floor* as the far side -
    a 99 px slot between two 300 px walls that nothing can be built into, so
    those surveys correctly answer "nothing to build" and, keyed, wiped the
    evidence for the bridge that the other half kept asking for.

    Charging on the bare fact of a split and reading the geometry fresh at the
    moment of commitment gets both: a wall the colony keeps walking into is
    answered even though nobody is standing on the wrong side of it at any
    particular survey, and what actually gets staked is a plan that is valid
    *now* rather than one remembered from a minute ago.

    The leak is asymmetric (``CUTOFF_LEAK`` < 1) so an intermittent split still
    converges, while a colony that sorts itself out - the founding scatter on an
    ordinary map - drains back to zero and stays there.
    """
    try:
        last = float(getattr(world, "_bhv_cut_charge_t", now))
    except (TypeError, ValueError):
        last = now
    elapsed = min(30.0, max(0.0, now - last))
    try:
        charge = float(getattr(world, "_bhv_cut_charge", 0.0) or 0.0)
    except (TypeError, ValueError):
        charge = 0.0
    charge += elapsed if wanted else -CUTOFF_LEAK * elapsed
    charge = float(min(max(charge, 0.0), CUTOFF_DWELL * 3.0))
    for name, value in (("_bhv_cut_charge", charge), ("_bhv_cut_charge_t", now)):
        try:
            setattr(world, name, value)
        except Exception:
            pass
    return charge


def _compute_cutoff(world: Any) -> dict[str, Any] | None:
    lab, bars = _reach(world)
    if lab is None or not bars:
        return None                     # one connected map: nothing to answer

    alive = alive_agents(world)
    center = colony_center(world)

    # Home is where the *people* are, not where the buildings are: the colony is
    # the colonists. Ties go to the region holding the colony centre.
    counts: dict[int, int] = {}
    for a in alive:
        r = _region_of(lab, getattr(a, "x", None))
        if r >= 0:
            counts[r] = counts.get(r, 0) + 1
    if counts:
        # A hole is not a home. On seed 5 two of three survivors were standing on
        # the chasm floor - 99 px of gravel between two 300 px walls - and by
        # simple headcount that outvoted the 640 px of open country the rest of
        # the colony lives on, so the survey started planning a way *out* to the
        # settlement instead of a way *over* the gap. Regions too narrow to put a
        # hut on are only considered when there is nothing else.
        roomy = {r: n for r, n in counts.items() if _region_w(world, r) >= HOME_MIN_W}
        pool = roomy or counts
        best_n = max(pool.values())
        tied = [r for r, n in pool.items() if n == best_n]
        home = tied[0] if len(tied) == 1 else _pick_home(world, lab, tied, center)
    else:
        home = _region_of(lab, center)
    if home < 0:
        return None

    # Candidates are (weight, x, reason). Weight is how badly it hurts, and is
    # what decides which side of the colony gets its crossing first.
    cands: list[tuple[int, float, str]] = []
    obs = _obstacles(world, lab, bars)

    # 1. Somebody is on the wrong side. This is the case the user named: a
    #    villager who mined the floor out from under himself, or who went over
    #    the rim and lived. Nothing else the colony is short of matters as much.
    #
    #    But "is standing in another region" is not stranding, and taking it for
    #    stranding is what made the colony build for nothing. Regions split at
    #    every wall on the map, agents wander over the low ones all day, and on
    #    seed 7777 - a plateau, nobody trapped, nobody dying - that was enough
    #    to buy a ladder and a bridge and finish the run a hut short of the
    #    colony that built neither. Three things separate the real thing from
    #    that, in rising order of how little patience each deserves: a *hole*
    #    (nothing to eat down there, no way back up) is a rescue on sight; a
    #    *divided colony* - two or more of them still over there a moment later
    #    - is the case a crossing exists for; and one man off on his own is the
    #    weakest evidence there is, so he has to still be out there a minute and
    #    a half later before the colony drops what it is doing. Every plateau
    #    and valley map has a lone wanderer over an edge in the first minute;
    #    almost none of them have two.
    trapped: set[float] = set()
    away_from_home: list[tuple[Any, dict[str, Any] | None]] = []
    for a in alive:
        r = _region_of(lab, getattr(a, "x", None))
        if r < 0 or r == home:
            continue
        hole = _obstacle_holding(obs, r)
        if _stranded_for(world, a) >= (STRANDED_HOLE_DWELL if hole is not None
                                       else STRANDED_DWELL):
            away_from_home.append((a, hole))
    lone = len(away_from_home) < 2
    for a, hole in away_from_home:
        if hole is None and lone and _stranded_for(world, a) < STRANDED_DWELL_LONE:
            continue
        if hole is not None:
            # Inside the thing in the way - the chasm floor, the bottom of his
            # own quarry. Nothing down there to eat and no way back up, so this
            # is the one stranding that cannot wait, and the obstacle holding
            # him is marked so the dwell knows it.
            trapped.add(float(hole["mid"]))
        cands.append((3, float(getattr(a, "x", center)), "stranded"))

    # 2. The fire or the stores ended up across the gap. Both are *the* shared
    #    resource: everyone walks to them all day, so a split from either turns
    #    the whole colony into commuters over a lethal face.
    reg = structures_of(world)
    if reg is not None:
        for kind in ("stockpile", "firepit"):
            try:
                standing = [s for s in reg.of_kind(kind)
                            if getattr(s, "built", False)
                            and not getattr(s, "is_ruined", False)]
            except Exception:
                standing = []
            if not standing or any(_region_of(lab, s.x) == home for s in standing):
                continue
            far = min(standing, key=lambda s: abs(float(s.x) - center))
            cands.append((2, float(far.x), kind))

    # 3. Every last source of something is on the far side. Not "some of the
    #    trees" - a colony with a tree at home is not cut off from wood, it just
    #    has fewer of them - but the case where working the resource at all
    #    means crossing.
    by_kind: dict[str, list[tuple[int, float]]] = {}
    for p in props_of(world):
        if not prop_alive(p):
            continue
        k = str(getattr(p, "kind", None) or getattr(p, "type", None) or "").lower()
        if not k:
            continue
        try:
            px = float(getattr(p, "x", 0.0))
        except (TypeError, ValueError):
            continue
        by_kind.setdefault(k, []).append((_region_of(lab, px), px))
    for _res, kinds in _CUTOFF_SOURCES:
        here = 0
        away: list[float] = []
        for k in kinds:
            for r, px in by_kind.get(k, ()):
                if r == home:
                    here += 1
                elif r >= 0:
                    away.append(px)
        # ...and the colony is actually short of it. With a full store the far
        # bank is next month's wood rather than today's problem, and spending
        # the build order on reaching it is the same waste as any other
        # crossing nobody needed.
        if here == 0 and away and stock_qty(world, _res) < CUTOFF_SHORT_QTY:
            cands.append((1, min(away, key=lambda px: abs(px - center)), _res))

    if not cands:
        # Nothing the colony can name is on the wrong side - and it can still be
        # losing a villager a minute over the same rim. Seed 42 is the whole
        # argument: three walls, nobody trapped, every resource at home, and
        # seven people dead at the foot of one 392 px face while the survey
        # reported a perfectly connected colony with nothing to build. A wall
        # with bodies under it is a thing in the way whatever the reachability
        # graph believes, so it gets to ask for a crossing on its own evidence.
        lethal = [(o, _hazard_hits(world, o)) for o in obs]
        lethal = [(o, n) for o, n in lethal if n]
        if not lethal:
            return None
        o = max(lethal, key=lambda e: (e[1], -abs(e[0]["mid"] - center)))[0]
        # A rim column belongs to the region it overlooks, so the outer rim on
        # the far side is a point in the far region and needs no search.
        cands.append((2, float(o["hi"] if home <= int(o["left"]) else o["lo"]),
                      "falls"))
    # Worst problem first; among equals, the nearest one, because that is also
    # the cheapest to answer and the one the colony is walking into.
    weight, tx, reason = max(cands, key=lambda c: (c[0], -abs(c[1] - center)))
    return {
        "reason": reason,
        "weight": int(weight),
        "home": int(home),
        "target": _region_of(lab, tx),
        "target_x": float(tx),
        "trapped": frozenset(trapped),
        # The whole list travels on, because which *barrier* to answer is not
        # decided by whichever candidate hurts most. A colony must eventually
        # cross every wall between it and the thing it wants, so the wall worth
        # building at is the one doing the damage - see `_crossing_geometry`.
        "cands": tuple(cands),
    }


def _region_w(world: Any, region: int) -> int:
    """Width of a region in columns, or ``0`` if the terrain will not say."""
    fn = getattr(getattr(world, "terrain", None), "region_bounds", None)
    if not callable(fn):
        return 0
    try:
        b = fn(int(region))
    except Exception:
        return 0
    return (int(b[1]) - int(b[0])) if b else 0


def _pick_home(world: Any, lab: Any, tied: list[int], center: float) -> int:
    """Break a headcount tie: the colony centre first, then the bigger region.

    Width matters and is not a tidiness rule. Four founders land scattered, and
    on a chasm map two of them can land *on the chasm floor* (seeds 5 and 10 do
    exactly that), which makes a 99 px slot between two 300 px walls tie 2-2
    with the 575 px of open country everybody else is standing on. Calling the
    hole "home" inverts the whole survey: the colony would then be planning a
    crossing out toward its own settlement.
    """
    at_center = _region_of(lab, center)
    if at_center in tied:
        return at_center
    return max(tied, key=lambda r: (_region_w(world, r), -r))


# ---------------------------------------------------------------- planning --


def plan_crossing(
    world: Any, reg: StructureRegistry | None = None
) -> dict[str, Any] | None:
    """The crossing the colony needs next, or ``None``.

    ``{"kind", "x", "y", "extra", "reason"}`` - ``kind`` is ``"bridge"`` or
    ``"ladder"`` and the rest is exactly what :func:`pick_site` returns, so the
    director stakes it out like any other building.

    The vetoes are applied *here* rather than inside the cached survey, because
    they change the moment a site is staked and a stale "yes" would jam the
    build order behind a bridge that already exists.
    """
    cut = find_cutoff(world)
    if cut is None or cut.get("plan") is None:
        return None
    # The dwell is confirmation that the split is real, and an obstacle with a
    # body at the foot of it - or a villager stuck inside it - is already
    # confirmed. Waiting the full dwell there is not caution, it is the
    # difference between a deck that prevents the next four deaths and one that
    # arrives after them.
    dwell = CUTOFF_DWELL_URGENT if cut["plan"].get("urgent") else CUTOFF_DWELL
    if float(cut.get("charge", 0.0)) < dwell:
        return None
    reg = reg if reg is not None else structures_of(world)
    if reg is not None:
        try:
            live = [s for s in reg
                    if s.kind in CROSSING_KINDS and not getattr(s, "is_ruined", False)]
        except Exception:
            live = []
        # One crossing at a time. A half-built bridge is already the answer to
        # this barrier; asking for a second one every director pass would stall
        # every other build behind a kind that never leaves the queue.
        if any(not getattr(s, "built", False) for s in live):
            return None
    else:
        live = []

    plan = cut["plan"]
    # Already spanned. Either it worked and the survey has not caught up, or it
    # did not and building a second one on top will not help either.
    x0, x1 = plan["span"]
    for s in live:
        try:
            span = s.crossing_span()
        except Exception:
            span = None
        if span and span[0] <= x1 and x0 <= span[1]:
            return None
    return plan


def _crossing_geometry(world: Any, cut: dict[str, Any]) -> dict[str, Any] | None:
    """Turn a cutoff report into a sited bridge or ladder.

    Which wall gets answered is decided here, and it is the question the old
    version got wrong. It took the barrier nearest the colony centre between
    home and whatever hurt most, on the reasoning that the near one is the
    cheapest. But a colony has to cross *every* wall between it and the thing it
    wants, so nearness ranks nothing - and meanwhile the wall that is actually
    killing people may be in the other direction entirely. Measured on seed 42:
    three walls, all seven fall deaths at the 392 px one, and the colony spent
    the run bridging a 197 px wall two hundred pixels past it that nobody ever
    died at and that left the killer untouched.

    So: enumerate the obstacles, score each by what it is doing to the colony -
    bodies first, then what is stranded or unreachable behind it - and stake out
    the first one that a crossing can genuinely join. "Can genuinely join" is
    the other half of the fix: a plan whose two ends do not land on the two
    sides of the obstacle is not a crossing, it is a ladder into a hole, and
    three of fourteen chasm colonies spent their wood on exactly that.
    """
    lab, bars = _reach(world)
    surf = _surface_of(world)
    if lab is None or not bars or surf is None:
        return None
    home = int(cut["home"])

    # Where the colony stands *inside* its own region - the crossing has to be
    # walkable-to, so it is measured from home, not from a centre that may have
    # been dragged across the gap by whoever is stranded on the far side.
    home_x = colony_center(world)
    if _region_of(lab, home_x) != home:
        idx = np.flatnonzero(lab == home)
        if idx.size == 0:
            return None
        home_x = float(0.5 * (idx[0] + idx[-1]))

    ranked = _rank_obstacles(world, lab, bars, home, cut, home_x)
    for o in ranked:
        plan = _site_crossing(world, lab, bars, surf, home, o, cut)
        if plan is not None:
            plan["urgent"] = bool(o["urgent"])
            plan["obstacle"] = (int(o["lo"]), int(o["hi"]))
            return plan
    return None


def _rank_obstacles(
    world: Any, lab: Any, bars: tuple[tuple[int, int, float], ...], home: int,
    cut: dict[str, Any], home_x: float
) -> list[dict[str, Any]]:
    """The obstacles worth answering, worst first.

    Worst means: how many people it has killed, then how badly what is behind it
    is wanted, and only then how close it is. Bodies outrank everything because
    they are the only evidence that does not depend on the survey guessing right
    - an obstacle with a fall death at the foot of it *is* the problem, whatever
    the reachability graph thinks the colony is short of.
    """
    obs = _obstacles(world, lab, bars)
    cands = cut.get("cands") or ((int(cut.get("weight", 1)), float(cut["target_x"]),
                                  str(cut.get("reason", ""))),)
    out: list[dict[str, Any]] = []
    for o in obs:
        weight = 0
        reason = ""
        near = float("inf")
        for cw, cx, cr in cands:
            if not _must_cross(o, home, _region_of(lab, cx)):
                continue
            d = abs(float(cx) - o["mid"])
            if cw > weight or (cw == weight and d < near):
                weight, reason, near = int(cw), str(cr), d
        hits = _hazard_hits(world, o)
        if not hits and not weight:
            continue                    # in the way of nothing, hurting nobody
        entry = dict(o)
        entry["hits"] = int(hits)
        entry["urgent"] = bool(hits) or o["mid"] in (cut.get("trapped") or ())
        entry["weight"] = int(weight)
        # Why *this* obstacle, which is not always why the survey fired: the
        # chronicle should say what the colony noticed at the wall it is
        # actually staking out.
        entry["reason"] = reason or str(cut.get("reason", ""))
        out.append(entry)
    # Bodies first, then a man in the hole, then whatever is merely on the
    # wrong side of it, and only then which is nearest. Nearness ranks nothing
    # on its own: a colony has to cross every wall between it and what it wants,
    # so "closest" is not "cheapest", it is just "first in line" - and the wall
    # doing the killing is regularly not the one in front.
    out.sort(key=lambda e: (-e["hits"], not e["urgent"], -e["weight"],
                            abs(e["mid"] - home_x)))
    return out


def _site_crossing(
    world: Any, lab: Any, bars: tuple[tuple[int, int, float], ...], surf: Any,
    home: int, o: dict[str, Any], cut: dict[str, Any]
) -> dict[str, Any] | None:
    """A bridge or ladder that actually joins the two sides of ``o``."""
    w = int(surf.size)
    a, c, relief = int(o["lo"]), int(o["hi"]), float(o["relief"])

    # Columns strictly inside a wall are the wall itself - a deck that ends
    # there ends in mid-air, and a builder sent to stand there falls. For a
    # grouped obstacle the whole interior counts, floor included: the floor of a
    # chasm is somewhere a deck must pass *over*, never somewhere it may land.
    on_wall = np.zeros(w, dtype=bool)
    for ba, bc, _r in bars:
        if bc - ba > 1:
            on_wall[ba + 1 : bc] = True
    if c - a > 1:
        on_wall[a + 1 : c] = True

    far = int(o["right"]) if home <= int(o["left"]) else int(o["left"])

    # Two passes. First try to reach the thing that is actually cut off; if the
    # geometry will not allow that, settle for making the wall crossable at all.
    # Both are wanted and neither subsumes the other: a man at the bottom of a
    # quarry wants a ladder *down to him*, not a deck over his head - but at a
    # 300 px chasm with a 99 px floor there is no ramp that reaches him, and the
    # rim-to-rim bridge is still worth building because it is what stops the
    # next four people going in after him.
    tgt = int(cut["target"])
    if tgt in o["inner"]:
        # He is *in* the obstacle. Aiming at the hole is what produced a ladder
        # from the near rim down onto the chasm floor (seed 70) - geometry that
        # satisfies the survey, reaches the man, and leaves the gap exactly as
        # lethal as it was. Aim past it; getting the two sides joined is what
        # gets him out and what stops the next one going in.
        tgt = far
    for target in (tgt, far, home):
        # A gap is bridged; a face is laddered. The test is whether there is
        # ground across at roughly the same height to land on: a chasm has two
        # rims with a hole between them, a plateau edge has no far rim at all -
        # it is simply lower from here on, and the way back up is to climb.
        pair = _bridge_pair(surf, lab, on_wall, a, c, home, target)
        if pair is not None:
            p, q = pair                 # p is the home-side end
            x0, x1 = (float(min(p, q)), float(max(p, q)))
            if _joins(lab, o, x0, x1):
                return {
                    "kind": "bridge",
                    # Anchored on home ground at the near end, never mid-span:
                    # `x` is where a builder walks to, and the middle of a
                    # bridge is 300 px of open air until the last stage is
                    # stamped. The deck geometry travels in `span`, which the
                    # stamp and the renderer both read.
                    "x": float(np.clip(p, 4.0, w - 5.0)),
                    "y": float(min(surf[p], surf[q])),
                    "extra": {"w": (x1 - x0) + 18.0, "span": [x0, x1]},
                    "span": (x0, x1),
                    "reason": str(o.get("reason") or cut["reason"]),
                }

        plan = _ladder_plan(world, surf, lab, a, c, relief, cut)
        if plan is not None and _joins(lab, o, *plan["span"]):
            plan["reason"] = str(o.get("reason") or cut["reason"])
            return plan
        if target == home:
            break                       # the unconstrained pass has run
    return None


def _joins(lab: Any, o: dict[str, Any], x0: float, x1: float) -> bool:
    """Would a crossing from ``x0`` to ``x1`` actually put the two sides of
    ``o`` together?

    The two ends have to land in the obstacle's own two outside regions - one
    each, not both on the same side and neither inside it. Everything that gets
    built and changes nothing fails exactly here: a ladder from the near rim
    down onto the chasm floor has one end in ``inner``; a deck that clears the
    obstacle *and* the next one along has an end past ``right``; a ramp too
    short to shed the whole face keeps both ends on the home side. All three
    were measured being built, and none of them joined anything - three of
    fourteen chasm colonies finished the run still split, one of them the
    deadliest seed in the sweep.

    Cheap on purpose: region ids already encode reachability, so this is two
    lookups rather than a re-survey of merged terrain.
    """
    ends = {_region_of(lab, x0), _region_of(lab, x1)}
    return ends == {int(o["left"]), int(o["right"])}


def _bridge_pair(
    surf: Any, lab: Any, on_wall: Any, a: int, c: int, home: int, target: int
) -> tuple[int, int] | None:
    """Narrowest deck that spans the wall between rims ``a`` and ``c``.

    Searches every pair of standable columns, one on each side, and takes the
    shortest that qualifies - the cheapest crossing, which is what the colony
    would pick and what costs the least wood. A pair qualifies when the two ends
    are within ``CROSSING_RIM_TOL`` of each other in height (a deck is roughly
    level; a 250 px step is a face, not a gap) and the ground between them dips
    at least ``CROSSING_MIN_DEPTH`` below the lower end (there is actually a
    hole to span).

    Anchoring on the wall's own two rims is not enough and was measured failing:
    on seed 8 the near rim *is* the chasm floor, 236 px below the far one, so no
    deck could ever join them - while a pair 60 px further back on the home side
    spans the whole chasm at a level the two rims share.

    ``None`` means there is no gap here, only a face, and the answer is a ladder.

    The between-the-ends maximum is what makes this affordable: it separates
    into "highest ground from p to the wall", "the wall", and "the wall to q",
    so the O(n^2) pair test is three broadcasts over ~260x260 rather than a
    range query per pair.
    """
    w = int(surf.size)
    span_max = int(CROSSING_MAX_SPAN)
    left = np.arange(max(0, a - span_max), a + 1)
    right = np.arange(c, min(w - 1, c + span_max) + 1)
    if left.size == 0 or right.size == 0:
        return None
    # Home can be on either side of the wall. Getting this backwards costs the
    # whole answer rather than mirroring it: with the near end forced onto the
    # far side there are no candidate columns at all, so a perfectly good chasm
    # reads as unbridgeable (seed 5, whenever the colony drifted right of it).
    # Both arrays run *outward from the wall*, so a running maximum along each
    # is "the highest ground between this column and the wall".
    if int(lab[a]) == home:
        P, Q = left[::-1], right
    else:
        P, Q = right, left[::-1]

    ok_p = (~on_wall[P]) & (lab[P] == home)
    ok_q = (~on_wall[Q]) & (lab[Q] != home)
    # A deck has to land somewhere that helps. Region ids increase left to
    # right, so "toward the target and no further" is two integer tests - and
    # without them a pit dug in the middle of flat ground gets *bridged over*
    # while the man who dug it stays at the bottom, which is a crossing that
    # answers the geometry and ignores the problem.
    if target != home:
        toward = 1 if target > home else -1
        rel = (lab[Q] - home) * toward
        ok_q &= (rel > 0) & (rel <= abs(target - home))
        if not ok_q.any():
            return None
    if not ok_p.any() or not ok_q.any():
        return None

    yp = surf[P]
    yq = surf[Q]
    pmax = np.maximum.accumulate(yp)
    qmax = np.maximum.accumulate(yq)
    pmin = np.minimum.accumulate(yp)
    qmin = np.minimum.accumulate(yq)
    wmax = float(surf[a : c + 1].max())
    wmin = float(surf[a : c + 1].min())

    span = np.abs(Q[None, :].astype(np.float64) - P[:, None])
    good = ok_p[:, None] & ok_q[None, :]
    good &= (span >= CROSSING_MIN_SPAN) & (span <= float(span_max))
    good &= np.abs(yp[:, None] - yq[None, :]) <= CROSSING_RIM_TOL
    deck = np.minimum(yp[:, None], yq[None, :])          # y of the planking
    deepest = np.maximum(np.maximum(pmax[:, None], qmax[None, :]), wmax)
    good &= (deepest - np.maximum(yp[:, None], yq[None, :])) >= CROSSING_MIN_DEPTH
    # Nothing may stick up through the deck. The overlay merges by "whichever
    # surface is higher", so a ridge between the two ends is not spanned - it
    # pokes out of the planking and leaves the crossing broken in the middle.
    # Measured on a quarry pit dug into seed 11: the pair test happily proposed
    # a deck from the rim to a high spot *inside* the pit, straight through
    # 45 px of intervening ground.
    peak = np.minimum(np.minimum(pmin[:, None], qmin[None, :]), wmin)
    good &= peak >= deck - CROSSING_DECK_CLEAR
    if not good.any():
        return None

    k = int(np.argmin(np.where(good, span, np.inf)))
    i, j = divmod(k, int(Q.size))
    return int(P[i]), int(Q[j])


def _ladder_plan(
    world: Any, surf: Any, lab: Any, a: int, c: int, relief: float,
    cut: dict[str, Any]
) -> dict[str, Any] | None:
    """A ladder against the wall between rims ``a`` and ``c``."""
    span = _ladder_span(surf, a, c, relief, lab)
    if span is None:
        # Nothing verified against this face. The general survey may still find
        # a workable ramp on it (it searches a little differently), so ask - but
        # only take an answer that actually lands on the face we are stuck at.
        got = plan_ladder(getattr(world, "terrain", None), 0.5 * (a + c))
        if got is None:
            return None
        gx, gy, extra = got
        raw = extra.get("span") if isinstance(extra, dict) else None
        if not (isinstance(raw, (list, tuple)) and len(raw) >= 2):
            return None
        sx0, sx1 = float(min(raw[0], raw[1])), float(max(raw[0], raw[1]))
        if sx1 < a - 4 or sx0 > c + 4:
            return None
        return {"kind": "ladder", "x": float(gx), "y": float(gy),
                "extra": dict(extra), "span": (sx0, sx1),
                "reason": str(cut["reason"])}

    x0, x1, y0, y1 = span
    # The foot is the low end: where a builder stands, and where a ladder rests.
    foot_x, foot_y = (x0, y0) if y0 > y1 else (x1, y1)
    return {
        "kind": "ladder",
        "x": float(foot_x),
        "y": float(foot_y),
        "extra": {"w": (x1 - x0) + 6.0, "span": [float(x0), float(x1)],
                  "rise": [float(y0), float(y1)]},
        "span": (float(x0), float(x1)),
        "reason": str(cut["reason"]),
    }


def _ladder_span(
    surf: Any, a: int, c: int, relief: float = 0.0, lab: Any = None
) -> tuple[float, float, float, float] | None:
    """Footprint of a ramp that defeats the wall between rims ``a`` and ``c``.

    Returns ``(x0, x1, y0, y1)`` with ``x0 < x1`` - the arguments
    ``Terrain.stamp_climb`` takes. The ramp runs from the lip out over the *low*
    side, because the overlay only wins where it is above the rock: a ramp laid
    over the high side is simply ignored and leaves the face as impassable as it
    was.

    Every length is verified by merging the proposed ramp onto the real ground
    and measuring the worst gradient over ``3`` px - the span the physics judges
    a step over. Sizing the run from the wall's height alone is not enough:
    where the low ground keeps falling away past the lip, the foot lands below
    where it was aimed and the "ramp" comes out steeper than the face.
    """
    w = int(surf.size)
    a = int(np.clip(a, 0, w - 1))
    c = int(np.clip(c, 0, w - 1))
    y_a, y_c = float(surf[a]), float(surf[c])
    rise = abs(y_a - y_c)
    if rise < LADDER_MIN_RISE:
        return None
    # Smaller y is higher ground, so the lip is whichever rim is smaller, and
    # the low side - where the ramp goes - is the other way.
    if y_a <= y_c:
        x_top, y_top, d = a, y_a, 1
        low_side = c
    else:
        x_top, y_top, d = c, y_c, -1
        low_side = a
    # The foot has to come down in the region on the other side of *this* wall.
    # Without it the search happily walks a "ramp" clear across the next gap as
    # well: on seed 5 it proposed a 156 px near-level line from the chasm's near
    # rim to the far plateau, which passes every gradient test because it is a
    # bridge, and which leaves the wall it was asked about untouched.
    low_region = None if lab is None else _region_of(lab, low_side)
    # Slide the top anchor to the *innermost* column of the flat lip, i.e. as
    # far over the face as the ground stays level. It matters: a straight ramp
    # aimed one column short of the edge runs a pixel or two under the last of
    # the rock, the merge picks the rock, and that 2 px flip is enough to read
    # as 2.57 where the ground it replaced was 4.93 and the limit is 2.39. On
    # seed 42 it is the difference between a 392 px face the colony can answer
    # and one it cannot.
    for _ in range(12):
        nxt = x_top + d              # `d` points down the face, so this is into it
        if not (0 <= nxt < w) or float(surf[nxt]) > y_top + 1.5:
            break
        x_top = nxt

    limit = MAX_SLOPE_CLIMB * 0.92      # leave room for fractional-x sampling
    # Shortest workable ramp wins, so the search crawls up from LADDER_MIN_W
    # rather than starting at rise / LADDER_SLOPE. That shortcut is what
    # Terrain.find_climb_face uses and it is wrong for a *pit*: the wall of a
    # quarry drops 196 px in five columns but the floor climbs straight back up,
    # so a 48 px ramp sheds the whole face while the shortcut refuses to look at
    # anything under 89. `_ramp_gradient` is the judge either way - it merges
    # the ramp onto the ground over a window that always contains the face - so
    # starting low only costs iterations, and buys the cheapest ladder.
    tall = max(rise, float(relief))
    longest = int(min(LADDER_BARRIER_MAX_W,
                      max(LADDER_MAX_W, round(tall / max(0.5, LADDER_SLOPE)) + 40)))
    reached = False
    for run in range(LADDER_MIN_W, longest + 1, 2):
        x_foot = x_top + d * run
        if not (0 <= x_foot < w):
            return None                 # ran off the edge of the world
        if low_region is not None:
            # The foot has to come down on the far side of *this* wall - but a
            # wall 59 columns wide has half of it labelled with the region above
            # it, so the first several runs land on the face itself and are
            # neither there yet nor past it. Stopping at the first mismatch
            # abandoned the search before it began: the loop starts at
            # LADDER_MIN_W = 22, and on seed 42 the 392 px face that killed
            # seven people was refused at run 22 with the foot still 29 columns
            # short of the region boundary. Every wall wider than about twice
            # LADDER_MIN_W was unanswerable for the same reason, which is most
            # of the walls worth answering.
            if _region_of(lab, x_foot) != low_region:
                if reached:
                    break               # past the far side of this wall
                continue                # still on the face, keep walking down
            reached = True
        y_foot = float(surf[x_foot])
        if abs(y_foot - y_top) < LADDER_MIN_RISE * 0.8:
            continue                    # still on the lip, nothing descended
        # Deliberately *not* "has it shed the whole face": where the low ground
        # rises again past the foot of a wall - a quarry floor, the far bank of
        # a pit - the drop at the landing is a fraction of the wall's own
        # relief, and demanding the full height there rejects every run. The
        # gradient probe below is the real test and cannot be fooled: it merges
        # the ramp onto the ground over a window that always contains the face,
        # so a ramp that fails to cover it reads the bare rock and is refused.
        lo_x, hi_x = (x_foot, x_top) if x_foot < x_top else (x_top, x_foot)
        ya, yb = (y_foot, y_top) if x_foot < x_top else (y_top, y_foot)
        if _ramp_gradient(surf, lo_x, hi_x, ya, yb) <= limit:
            return float(lo_x), float(hi_x), float(ya), float(yb)
    return None


def _ramp_gradient(surf: Any, a: int, b: int, ya: float, yb: float) -> float:
    """Worst ``|dy/dx|`` on a ramp once merged onto the ground it sits on."""
    w = int(surf.size)
    lo = max(0, a - 4)
    hi = min(w - 1, b + 4)
    prof = surf[lo : hi + 1].astype(np.float64, copy=True)
    n = b - a + 1
    i = a - lo
    prof[i : i + n] = np.fmin(prof[i : i + n], np.linspace(ya, yb, n))
    if prof.size <= 3:
        return 0.0
    return float(np.max(np.abs(prof[3:] - prof[:-3]))) / 3.0


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
    # Ask the terrain where it *put* the chasm before going looking for one.
    #
    # This read was spelled "chasms" while Terrain has only ever exposed "chasm"
    # - a single (x0, x1) pair - so the fast path was dead code and every call
    # fell through to the widest-qualifying-run scan below. That scan picks by
    # WIDTH, and the deliberately cut chasm is narrow (60-120 px) while natural
    # dips in the noise reach the CHASM_MAX_W cap, so it usually returned some
    # harmless dip elsewhere: measured, the span did not overlap the real cut on
    # 8 of 15 chasm seeds. The consequence was that the bridge got staked over a
    # 79-149 px dimple while the 300-483 px killer went unspanned, which is why
    # no bridge appeared in any of 18 measured runs.
    #
    # Both spellings are accepted, and both shapes: a bare (x0, x1) pair and a
    # sequence of them, since this module duck-types the world it is handed.
    for attr in ("chasm", "chasms"):
        declared = getattr(terr, attr, None)
        if callable(declared):
            try:
                declared = declared()
            except Exception:
                declared = None
        if not isinstance(declared, (list, tuple)) or not declared:
            continue
        first = declared[0]
        pair = declared if isinstance(first, (int, float)) else first
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            try:
                x0, x1 = float(pair[0]), float(pair[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x0) and math.isfinite(x1) and x1 > x0:
                return x0, x1
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
