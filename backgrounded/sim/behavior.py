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
    BARRIER_MIN_RELIEF,
    CHASM_SAFE_GRADIENT,
    CLEANUP_SCORE_MAX,
    CROSSING_MAX_SPAN,
    CROSSING_MIN_DEPTH,
    FALL_LETHAL_SPEED,
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
    MIN_POP,
    POP_PER_HUT,
    RELIC_FETCH_RANGE,
    RENDER_W,
    RES_COOKED,
    RES_FIBRE,
    RES_FOOD,
    RES_GARBAGE,
    RES_STONE,
    RES_WOOD,
    SCENE_BLIZZARD,
    SCENE_NIGHT_STORM,
    STAGE_HALF,
    WALK_SPEED,
    WORLD_W,
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
    settlement_center,
    slope_at,
    stock_qty,
    structures_of,
    world_now,
)
from .entities import GRAVITY
from .structures import (
    CROSSING_KINDS,
    HUT_UPGRADE_COST,
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
    "stone_surplus",
    "pick_hut_upgrade",
    "find_chasm",
    "chasm_in_reach",
    "crossing_reach",
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
#: Raised 6 -> 7 with MAX_POP 10 -> 20, and it is not a taste change: growth is
#: gated on shelter in ``World._tick_hut_tier``'s neighbour at world.py:931,
#: ``n >= huts * POP_PER_HUT``. POP_PER_HUT is 3, so six huts house 18 and the
#: colony could never reach its own cap - the roster would stall two short of 20
#: with no message and no visible cause, and MAX_POP would silently not be the
#: thing limiting the population. Seven huts house 21, which puts the ceiling
#: back where the constant says it is.
MAX_HUTS = 7
MAX_WALLS = 4
TOTEM_POP = 8
TREE_TARGET = 10            # below this, someone plants saplings
#: How far from the colony centre a site may be.
#:
#: Raised 300 -> 500 for MAX_POP 20. Twenty people want seven huts (was four);
#: at the hut ``spacing`` of 54 px that is 378 px of hut alone, inside a band
#: that also has to hold a firepit (58), a stockpile (28), a watchtower (120), a
#: workshop (80), a totem and up to four walls. It does not fit in 600 px, and
#: the failure mode is quiet in the worst way: ``pick_site`` starts returning
#: None because every column is inside somebody's spacing, so ``_stake_out_site``
#: no-ops and the build order stalls forever with nothing logged.
#:
#: 500 is affordable now and was not before. On a 1600 px world a 1000 px
#: settlement was most of the map, wilderness included; on a 6400 px world it is
#: a sixth of the land and still sits comfortably inside the 1600 px camera, so
#: the whole colony stays in one frame.
#:
#: Knock-on worth knowing rather than acting on: spreading twenty colonists over
#: 1000 px instead of 600 px lowers their linear density, which pushes the BFG's
#: measured P(line clear) back UP from where doubling the roster alone put it.
#: The two effects are coupled, so the corridor numbers in constants.py:993 must
#: be re-measured against BOTH changes at once, never against either alone.
SITE_RANGE = 500.0
#: How far out from the settlement a barricade stands, in px.
#:
#: STAGE_HALF (800) is the frame edge and is the wrong answer by exactly the
#: slack it leaves, which is none. Measured on seed 7 at 25 sim-minutes: a
#: barricade staked at -800 when the colony was four buildings old ended up 942
#: px from the settlement centre, because every hut that went up afterwards
#: pulled the mean 140 px the other way - so a defence that was on camera when
#: it was built was off it by the time anything attacked. The camera has the
#: same problem from the other end: it follows the agent CLUSTER, not the
#: structure mean, so the frame is already offset from this centre by whatever
#: the colonists are doing.
#:
#: 160 px of slack, the same margin OFFSTAGE uses over STAGE_HALF, covers both.
#: 640 is still 140 px outside SITE_RANGE, so the band's outer half never
#: overlaps the settlement footprint; its inner end (640 - 256 = 384) does, but
#: the scorer prefers the outermost non-cliff column, so a barricade only comes
#: in that far when the whole outer band is cliff - which is the case where
#: standing it up at all beats standing it up in the right place.
BARRICADE_STANDOFF = STAGE_HALF - 160.0
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

# ------------------------------------------------------ faces that kill only --
# A face can be lethal without being a *barrier*. `Terrain.barriers()` wants
# both an unclimbable slope and BARRIER_MIN_RELIEF (150 px) of relief, because
# relief is what makes a face something the colony has to build its way past -
# cutting on slope alone shatters an ordinary hills map into ~150 regions, since
# generation guarantees a cliff on every map and agents climb the small ones all
# day. That threshold is load-bearing and is left exactly where it is.
#
# But "small enough not to divide the map" is not "small enough to be safe", and
# a 'cliffs' map is made of the difference: it is a staircase of one-column steps
# ~54 px tall. 54 px is under BARRIER_MIN_RELIEF so no barrier is emitted and
# `find_cutoff` sees nothing wrong; it is *also* under `entities.STEP_FALL_MAX`
# (FALL_LETHAL_SPEED**2 / 2g = 64.2 px), so the walk code lets people stroll
# straight off it. Both of those readings are of the face alone, and the face
# alone is not what you fall down: measured on seed 21, an agent came off the lip
# at x=845 (y=589.3) and landed at x=862 (y=654.7) - a 65 px drop, impact
# 342 px/s against a 340 px/s limit - because the ground *below* the step keeps
# falling away at nearly 1:1 while the body is in the air. Twelve extra pixels of
# run-out is the whole difference between a step people hop off all day and a
# funeral, and nothing in the survey was looking at it.
#
# So there is a second kind of thing in the way: steep enough that a slip does
# not self-arrest, and with enough left to fall that the landing kills. It is
# answered with a ladder like any other face - but only where one has actually
# killed somebody, which is the whole of the rationing. See the note below
# HAZARD_FACE_SLACK for what happened when the colony was allowed to go looking.

#: How far a body falls before it is travelling at FALL_LETHAL_SPEED, from rest.
#: The same number `entities.STEP_FALL_MAX` is derived from, and the reason a
#: 54 px step is harmless on its own.
FALL_LETHAL_DROP = FALL_LETHAL_SPEED * FALL_LETHAL_SPEED / (2.0 * GRAVITY)
#: ...and how far it travels sideways in that time, which is the window the drop
#: has to be measured over.
#:
#: The factor of two is measured, not padding. A falling agent is still being
#: *steered*: `_ground_step` runs before `_fall_step` in the same update and each
#: writes its own displacement, so an airborne villager walking toward something
#: covers ground at twice the walk speed. On the seed 21 death the trace reads
#: vx = 31.98 (0.94 * WALK_SPEED) with x advancing 2.19 px per tick - 65.7 px/s.
#: Taking the single walk speed instead reads the drop at x+13 rather than x+26,
#: which on that face is 63.0 px against a 64.2 px threshold: it misses the fall
#: that actually happened by a pixel.
HAZARD_RUNOUT = 2.0 * WALK_SPEED * FALL_LETHAL_SPEED / GRAVITY
#: Probe width for the steepness test, px. Matches `Terrain._build_reach`'s own
#: BARRIER_PROBE so a hazardous face and a barrier are found on the same reading
#: of the ground, and the two can never disagree about where a face begins.
HAZARD_PROBE = 3
#: How far outside a hazardous face's own geometry a body is still that face's
#: doing. Far tighter than HAZARD_SLACK, and it is not a style choice: `lo`/`hi`
#: already span the lip and the ground the fall ends on, so a body from this
#: face is *inside* them by construction, and the margin only covers the launch
#: being a fraction of the way down the step. These faces come in staircases 40
#: px apart, and at HAZARD_SLACK's 45 px one corpse is claimed by both of its
#: neighbours - measured, that buys two ladders for one death, which is the
#: closest this feature came to the failure it was written to avoid.
HAZARD_FACE_SLACK = 4.0
#
# NOTE for anyone tempted to answer these faces *before* one of them kills
# somebody. It was written, measured and taken out again. The survey is cheap
# and the traffic signal is good - counting which faces the colony crosses
# between director passes ranks them clearly (on seed 21: 48, 34, 28, 9, 7, 5, 2
# over ten faces) - and it does not matter, because the cost is not in finding
# them:
#
#   * A crossing is the one *blocking* build, ahead of even the firepit, and a
#     rail against a step people have survived all day is not that. Staked
#     during the founding scatter it finished seed 21 with 6 alive and one hut,
#     against 9 alive and three.
#   * Deferred behind fire, stores and a roof it still finished 7 alive: the
#     ladder makes a one-way step climbable both ways, the settlement's centre
#     follows it out onto the terrace, and the colony is then living somewhere
#     it had no reason to be.
#   * Over 19 'cliffs'-bearing seeds it was a wash - 8 falls against 10 - while
#     costing three survivors.
#
# A body is different: it is evidence rather than a guess, it names one face out
# of ten, and answering it is what the ranking already does for barriers.

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

#: ...but that priority is earned by PROXIMITY, not by kind, and this is the
#: number that says so. A crossing outranks the roof over the colony's head
#: because it is *blocking* - while it is missing, the ground, the wood or one
#: of the people is on the far side of a drop. A crossing the colony has to walk
#: a stage-width to reach is blocking nothing at home; it is an expedition.
#:
#: On a 1600 px world the distinction did not exist, because "the wall in the
#: way" was never more than STAGE_HALF away and the walk was free. Measured on
#: the 6400 px world: seed 2 staked a bridge 1329 px from the settlement with a
#: half-built hut standing at 135 px, and `_mk_build` sent both surviving
#: colonists to the bridge - the queue is sorted by priority and pools only what
#: is within 0.06 of the head, so at 1.10 against 0.85 the hut was not even a
#: candidate. The hut was still at completion 0.0 twelve minutes later, nobody
#: could sleep, fatigue pegged at 1.00, morale fell to 0.00 against a
#: MORALE_TO_GROW of 0.45, and the colony sat at MIN_POP for 73 of 90 minutes.
#:
#: 0.82 is chosen against the two numbers it has to sit between: below `hut`
#: (0.85) so a roof wins outright, and within `_mk_build`'s 0.06 pool window of
#: it so that when both are open the *closer* one wins rather than the kind.
#: Above `grave` and every outwork, because a way across is still worth more
#: than a fence. A crossing inside the stage keeps 1.10 untouched, so the chasm
#: maps KIND_PRIORITY was written for are unaffected.
CROSSING_FAR_PRIORITY = 0.82

#: How far from the settlement the colony will stake a crossing it is not being
#: killed by, and the two numbers that let that grow.
#:
#: This is the barricade migration applied to the crossing director. `_barricade_
#: site` was already moved off "the edge of the MAP" onto "the edge of the
#: COLONY" when the world went 1600 -> 6400; the crossing planner was left
#: reasoning about all 6400 px, and so was the legacy "there is a chasm
#: somewhere, bridge it eventually" line in the build order. Measured at t=0 over
#: the 14-seed sweep, `find_chasm` names a gap 54, 683, 754, 786, 900, 1084,
#: 1308, 1391, 1503, 1653, 2617, 2718, 3856 and 4307 px from the settlement -
#: twelve of fourteen seeds have one, and on the old world every one of them
#: would have been inside a stage-width by construction.
#:
#: THE TWO BASES ARE DIFFERENT NUMBERS BECAUSE THEY ANSWER DIFFERENT QUESTIONS,
#: and collapsing them was measured costing more than it saved. A *blocking*
#: crossing is answering something the colony actually needs - its fire, its
#: stores, one of its own people, or the last wood on its side - and where the
#: wall sits is not the colony's choice. The legacy line is answering nothing at
#: all: "there is a gap over there and one day we should span it".
#:
#: So the blocking planner gets RENDER_W, one whole camera width - the country
#: the colony works, well outside the stage it is drawn on, and the distance
#: animals already walk in from - while the "eventually" line gets STAGE_HALF,
#: the ground the player can actually watch. Run at STAGE_HALF for both, the
#: 14-seed sweep gated out a ladder seed 3 had built at 868 px to get back to
#: its own fire, and fall deaths over the sweep went 55 -> 76: crossings are
#: what stop people going over edges, and a blocking one is usually earning its
#: keep. Run at RENDER_W for both, the legacy line is straight back to staking
#: bridges over dips 1500 px away that nothing needed.
#:
#: Both then GROW with what the colony has actually built, which is the whole
#: reason this is a budget and not a ban - the wide map is the point, and a
#: settlement with huts up has the people and the rested hours to spend on an
#: expedition that a settlement without a roof does not.
#:
#: Nothing here gates an URGENT crossing - one with a body at the foot of it or
#: somebody trapped inside it. Those are rescues, they are the evidence
#: `_rank_obstacles` trusts most, and seed 42's seven fall deaths at one face
#: are the standing argument for answering them wherever they are.
CROSSING_REACH_BASE = float(RENDER_W)   # 1600.0 - a camera width of country
CHASM_REACH_BASE = STAGE_HALF           # 800.0 - the stage, for the idle bridge
CROSSING_REACH_PER_HUT = 400.0          # earned reach per standing hut
CROSSING_REACH_MAX = 3200.0             # half the world, then stop
CHASM_REACH_MAX = 2400.0

#: ...and the colony has to be big enough to be asking. The legacy chasm line is
#: the one entry in the build order with no population floor on it, sitting
#: below three lines that have one (barricade and wall at 4, watchtower at 5)
#: and above the totem at 8. That gap is not cosmetic: a colony at three people
#: falls straight THROUGH the wall and watchtower lines - both of which it is
#: too small for - and lands on "build a bridge". Seed 101 did exactly that at
#: t=555 and staked a bridge 1494 px away with three people alive; it never
#: finished it and spent 78 of 90 minutes at MIN_POP. 6 is one clear of the
#: watchtower and is the first headcount at which the order wants three huts.
CHASM_BRIDGE_MIN_POP = 6

#: The stone-hut upgrade's weight, and the ceiling on how much any villager
#: wants to do it. Both numbers exist to keep the tier a luxury.
#:
#: 0.30 is deliberately *below* the totem (0.40), the least urgent thing the
#: colony ever queues: re-walling a hut that already keeps the rain off must
#: lose to food, warmth, sleep, and to raising a hut that does not exist yet.
#: This project has already shipped a bug where a cleanup job outranked shelter
#: and the colony froze; an upgrade is a strictly more optional job than that.
#:
#: The appetite is a hard multiplicative cap rather than a tuning knob: even a
#: builder (build affinity 1.00) standing in a stockpile full of stone scores
#: this at most 0.45, comfortably under Eat/Sleep/WarmAtFire at any real need
#: and under BuildStructure for every kind above `totem`. Capping the score is
#: what makes that structural rather than merely true at today's numbers - the
#: CLEANUP_SCORE_MAX precedent.
HUT_UPGRADE_PRIORITY = 0.30   # below the totem (0.40), the lowest thing queued
HUT_UPGRADE_APPETITE = 0.45   # ceiling on the UpgradeStructure utility score

#: ...and the ceiling an unclaimed job climbs to, over
#: :data:`HUT_UPGRADE_PATIENCE_SEC` of nobody picking it up. 0.72 is chosen
#: against the two scores it has to sit between: Celebrate is 0.82*morale + 0.18,
#: so 0.72 clears it up to morale 0.66 (the pooled colony mean is ~0.50 and the
#: tier unlocks at 0.55, so that covers the ordinary case and deliberately not a
#: euphoric colony - a settlement having the best day of its life is allowed to
#: keep dancing). Hunger, fatigue and cold all climb past 0.72 well before they
#: are dangerous, so an agent who actually needs something still goes and gets it.
HUT_UPGRADE_APPETITE_MAX = 0.72
#: Seconds of going unclaimed before a job reaches that ceiling. Long enough
#: that a colony with real work to do is never diverted by a fresh job, short
#: enough that the measured 6573 s worst-case wait cannot recur.
HUT_UPGRADE_PATIENCE_SEC = 480.0

#: Wanting a dragon's relic off the ground. Same two-number shape as the hut
#: upgrade above, and for the same reason - but the numbers are not the same,
#: because the two jobs fail in opposite directions.
#:
#: The hut upgrade is permanent: nobody claiming it costs the colony nothing but
#: time, so it is allowed to sit at 0.45 and merely be *bounded*. A relic rots
#: at ``RELIC_DECAY_SEC`` (1800 s) and drops about once every four raw colony-
#: hours, so "nobody picked it up" is not a delay, it is the whole feature never
#: happening. That asymmetry is why the floor here (0.62) is above the upgrade's
#: *ceiling* (0.45), and it is set against the bands in this file rather than
#: chosen: it clears Mourn (0.68) only after ageing, clears Celebrate
#: (0.82*morale + 0.18) at the pooled mean morale of ~0.50 = 0.59, and loses to
#: Eat, Sleep and WarmAtFire from the moment any of those needs is real.
#:
#: MEASURED at these numbers, 20 same-seed colonies x 60 sim-min, a relic
#: forced onto the ground every 600 s (120 drops): 118 of 120 claimed, mean
#: wait 45.0 s, median 27 s, p90 88 s, worst 351 s, and *nothing* rotted. The
#: same 20 seeds with the score forced to 0.001 - the job existing but nobody
#: wanting it, which is the stone-hut failure mode - claimed 0 of 120 and let
#: 60 of them decay. For scale, the stone-hut upgrade's worst measured wait was
#: 6573 s and it went unclaimed for up to 110 sim-minutes.
RELIC_APPETITE = 0.62
#: ...and where it gets to once nobody has claimed it. 0.90 is a hard bound, not
#: a preference. It has to stay under ``combat_actions``' FleeAnimal floor of
#: 0.92 and its FightAnimal floor of 0.95, or a wolf at the door loses to a
#: trinket in the dirt; and it has to stay under :data:`OVERRIDE_FLOOR` (0.95)
#: or the fetch becomes an *interrupt* that yanks people off a build.
RELIC_APPETITE_MAX = 0.90
#: Seconds unclaimed before it reaches that ceiling. Half the hut's 480 for the
#: reason above: this job has a deadline and that one does not. 240 s also buys
#: the ageing term seven full windows inside RELIC_DECAY_SEC, so a relic that
#: lands somewhere awkward gets escalated long before it rots.
RELIC_PATIENCE_SEC = 240.0
#: "No relic is worth walking into a wolf for" used to be a constant here
#: (RELIC_SAFE_RADIUS = 200.0) and a pair of checks in :func:`_score_loot`. It
#: is now ``combat_actions.FETCH_SAFE_RADIUS``, enforced once inside
#: ``combat_actions.fetchable_relic``, which this file asks. The number and its
#: justification moved intact; only the second copy is gone. Reason: this file
#: owning a gate that the *action* also had to enforce is how the feature came
#: to be scored by one lane and unbuildable by another.
#:
#: What the score is worth to somebody who is NOT the closest candidate.
#:
#: Without it the whole colony wants the same trinket at the same score, so the
#: whole colony walks to it. MEASURED flat, 6 seeds x 60 sim-min: seven people
#: fetching at once on the worst seed, 232 FetchRelic actions started for 6
#: relics claimed, 1085 person-seconds walking. The tail is self-inflicted -
#: ``RelicRegistry.take()`` returns False for everyone after the first, so the
#: losers fail on arrival and re-score straight back into it.
#:
#: 0.72 is a damper, not a lock. A hard "only the nearest may score it" hands
#: the relic to one person who may be asleep in a hut all night with nobody
#: able to take over; at 0.72 the far villagers keep a live candidate (0.42 to
#: 0.45 fresh, 0.61 to 0.65 fully aged) that still beats Wander, CleanLitter
#: and topping up
#: a full stockpile, so an idle colonist does go - they just no longer drop a
#: build to race for it.
#:
#: Re-measured on the same 6 seeds, peak people fetching at once, flat -> damped:
#: 4->1, 1->1, 6->2, 7->8, 4->4, 4->3. The crowding it was aimed at is gone on
#: the seeds that had it. Total starts and person-seconds are NOT a clean
#: comparison and are not claimed as one - changing any score re-phases the
#: world, so the two arms diverge into different colonies, and both totals are
#: dominated by two seeds with a different problem entirely.
#:
#: That other problem, diagnosed rather than guessed by logging every start on
#: the worst seed: all 153 of them are in the SAME terrain region as the
#: fetcher, and the colony is pacing back and forth against an unclimbable face
#: at x~1110 trying to reach a relic at 1301. That is ``actions._note_blocked``
#: - a blocked agent fails its goal after BLOCKED_GIVE_UP = 2.0 s, re-scores,
#: and the relic (whose score is still climbing with age) simply wins again.
#: Every goal behind a wall does this today, RetrieveSpear included; it is not
#: a bystander problem and RELIC_BYSTANDER does not and should not fix it.
#: Deaths on that seed were 5 against a control of 18, so it costs the colony
#: nothing measurable. Left alone on purpose: the fix belongs with the blocked/
#: timeout path, not with the appetite.
RELIC_BYSTANDER = 0.72

#: Wanting to spend the cairnstone on a headstone. Same two-number ageing shape
#: as the relic appetite above, and the same ceiling for the same hard reason -
#: it must stay under FleeAnimal's floor (0.92), FightAnimal's (0.95) and
#: :data:`OVERRIDE_FLOOR` (0.95), so a wolf at the door beats a resurrection and
#: a raising never becomes an *interrupt* that yanks somebody off a build.
#:
#: The floor is higher than the relic's 0.62 because the two jobs have opposite
#: crowding problems and this one has none. There is exactly one cairnstone
#: bearer and they are the only person in the world who can do this, so there is
#: nobody to damp against and no stampede to prevent: :data:`RELIC_BYSTANDER`
#: has no analogue here. What is left is only "does the bearer actually go", and
#: at 0.80 they beat Mourn (0.68), Celebrate at the pooled mean morale (0.59),
#: and every chore - while still losing to Eat, Sleep and WarmAtFire the moment
#: any of those is real, which is right: a colonist should not starve holding a
#: rock.
#:
#: The one case 0.80 loses is a euphoric colony mid-party (Celebrate is
#: 0.82*morale + 0.18, so it passes 0.80 at morale 0.76). That resolves itself
#: within CELEBRATION_FRESH, and the ageing term below closes it anyway. It is
#: also the correct outcome for the ten seconds it lasts: let them finish the
#: dance, then go and dig somebody up.
RAISE_APPETITE = 0.80
RAISE_APPETITE_MAX = 0.90
#: Seconds of carrying the stone with a reachable grave before it gets there.
#: :data:`RELIC_PATIENCE_SEC`, deliberately the same clock: this is the same
#: "somebody has a job and nobody is doing it" failure the loot appetite and
#: the stone-hut upgrade both hit, and it should not need a third number.
RAISE_PATIENCE_SEC = RELIC_PATIENCE_SEC

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

    # --------------------------------------------------------- upgrade ------
    # Re-walling a standing hut in stone. Its own channel, never the build
    # queue: `score_actions` only ever looks at queue[0], and an upgrade at
    # priority 0.30 would sort to the back of it and never be seen (and
    # `_h_build` would finish instantly on a hut that is already built).
    #
    # No `unmet` term and no priority weighting - the ceiling does not climb as
    # the job nears completion. That is on purpose: this is the least important
    # job in the colony and it should stay the least important job right up to
    # the last hammer blow.
    #
    # It does climb with how long the job has gone UNCLAIMED, which is a
    # different thing and fixes a measured priority inversion. At the flat 0.45
    # ceiling the upgrade won 403 of 59779 scored samples - 0.7% - and the
    # second most common thing beating it was Celebrate, at 17.1%. Celebrate
    # scores 0.82*morale + 0.18, so at the tier's own unlock threshold of 0.55
    # morale it sits at 0.63 and the upgrade could never win: the colony earns
    # masonry by being happy, and being happy is exactly what makes them dance
    # instead of doing the work they just unlocked. Measured lag from a job
    # opening to a stone hut standing ran 59 s on the best seed and 6573 s on
    # the worst, so the chronicle announced the tier and the player then watched
    # nothing happen for an hour and a half.
    #
    # Raising the flat ceiling over Celebrate would also raise it over hungry
    # and tired agents, which is the wrong trade. Ageing the job leaves the
    # normal priority alone and only bounds the wait.
    job = getattr(world, "upgrade_job", None)
    if isinstance(job, dict) and reg is not None:
        avail = _availability(world, job.get("needs") or {})
        opened = job.get("opened")
        if isinstance(opened, (int, float)) and not isinstance(opened, bool):
            age = max(0.0, float(getattr(world, "world_time", 0.0)) - float(opened))
        else:
            age = 0.0
        patience = _clamp01(age / max(1.0, HUT_UPGRADE_PATIENCE_SEC))
        ceiling = (HUT_UPGRADE_APPETITE
                   + (HUT_UPGRADE_APPETITE_MAX - HUT_UPGRADE_APPETITE) * patience)
        s["UpgradeStructure"] = _clamp01(
            aff["build"] * ceiling * (0.28 + 0.72 * avail))
    else:
        s["UpgradeStructure"] = 0.0

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
    # Wanting the loot is scored last of all, after the combat merge, because
    # it has to be able to *lose* to a wolf that combat_actions has just scored
    # and to *beat* the calm-colony chores scored above. See _score_loot.
    _score_loot(s, agent, world, danger)
    # And spending the cairnstone, for the same reason and in the same place:
    # it has to lose to a wolf combat_actions just scored and beat everything
    # the colony was calmly getting on with. See _score_raise.
    _score_raise(s, agent, world, danger)
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
        cx = max(4.0, min(float(WORLD_W - 4), x))
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


# ------------------------------------------------------------------- loot --
def _relic_target(agent: Any, world: Any) -> Any:
    """The relic *agent* should go and get, straight from the action layer.

    ``combat_actions.fetchable_relic`` is the only answer to this question and
    this file deliberately does not have a second one. It used to: a local
    ``_relic_near`` that asked ``world.relics.nearest`` and a local pair of
    ``_wolf_within`` checks. Two independent notions of "a relic worth walking
    to" in two files owned by two lanes is precisely the arrangement that let
    the appetite be tuned to three decimal places against a walk that did not
    exist.

    Still entirely fail-soft: a checkout with no combat_actions, a test stub
    with no registry and a save written before relics existed must all read as
    "there is no loot", never as an exception on the frame path.
    """
    try:
        from .combat_actions import fetchable_relic
        return fetchable_relic(agent, world)
    except Exception:
        return None


def _raise_target(agent: Any, world: Any) -> Any:
    """The headstone a cairnstone-bearer should walk to. Same contract."""
    try:
        from .combat_actions import raisable_grave
        return raisable_grave(agent, world)
    except Exception:
        return None


def _kind_is_wired(kind: str) -> bool:
    """True once ``combat_actions`` actually owns an action of that kind.

    The score and the action machine live in different files by contract, and
    they land separately. Scoring a kind that ``make_combat_action`` cannot
    build is not harmless: ``choose_action`` skips the candidate but has
    already spent a ``rng.uniform(0, TIEBREAK)`` off ``world.pyrng`` on it, and
    that stream is the one wolves spawn from. So an appetite stays switched
    off until the other half exists, and a checkout where it does not is
    bit-identical to the build before any of this - which is the only safe
    posture when four lanes merge into one branch.

    Fails closed. If a kind is ever renamed, that job stops being done and its
    count in a run goes to zero, which is loud; the alternative failure - phase
    drift nobody can see - is not.

    THIS GUARD DID ITS JOB AND THAT IS THE PROBLEM IT LEAVES BEHIND. It held
    the loot appetite at exactly zero for a whole release, correctly, because
    ``FetchRelic`` genuinely did not exist - and a switch that silently and
    correctly disables a shipped feature reads identically to a feature that
    works. 18 relics dropped, 0 picked up, and loot-on/loot-off bit-identical
    on 16 of 16 seeds. Keep the guard; do not trust a green run that never
    asserted a *non*-zero. The measurement that matters is relics picked up per
    colony-hour, and it must be greater than zero.
    """
    try:
        from .combat_actions import COMBAT_ACTION_KINDS
        return kind in COMBAT_ACTION_KINDS
    except Exception:
        return False


def _fetch_is_wired() -> bool:
    """True once ``combat_actions`` owns FetchRelic. See :func:`_kind_is_wired`."""
    return _kind_is_wired("FetchRelic")


def _closest_claimant(world: Any, x: float) -> Any:
    """Which living colonist is nearest to *x* and could actually fetch it.

    Same eligibility as :func:`_score_loot` itself - not a child, not taken,
    not already carrying a relic - so the "nearest" this hands back is the
    nearest *candidate* rather than the nearest body. Ties break on id so two
    people standing on the same pixel do not both read as closest, which would
    put the stampede straight back.
    """
    best, best_key = None, None
    for ag in alive_agents(world):
        try:
            if getattr(ag, "taken", False) or str(getattr(ag, "relic", "") or ""):
                continue
            if _role(ag) == "child":
                continue
            key = (abs(float(getattr(ag, "x", 0.0)) - x),
                   int(getattr(ag, "id", 0) or 0))
        except (TypeError, ValueError):
            continue
        if best_key is None or key < best_key:
            best, best_key = ag, key
    return best


def _score_loot(s: dict[str, float], agent: Any, world: Any,
                danger: Any) -> None:
    """Score ``FetchRelic`` in place. The last word on it, deliberately.

    ``score_combat`` owns the *action* - the walk, the claim, the re-acquire
    when somebody beat you to it - and may score it too. This runs after that
    merge and overwrites, because every number the score has to be justified
    against lives in THIS file: Celebrate, Eat, Sleep, WarmAtFire, Mourn, the
    0.35 danger cap, :data:`OVERRIDE_FLOOR` and :data:`HYSTERESIS_BONUS`. A
    ceiling tuned anywhere else is tuned against numbers it cannot see, which
    is precisely how the stone-hut upgrade came to score 0.45 against a
    Celebrate of 0.63 and go unclaimed for up to 110 sim-minutes.

    The key is only *raised* here when there is a real relic to fetch. On a
    world with no relics this writes 0.0 and nothing else, and a 0.0 candidate
    is dropped by ``choose_action`` before it draws its tiebreak - so such a
    world is bit-identical to the build before relics existed. That is not a
    nicety: behaviour draws ``rng.uniform(0, TIEBREAK)`` off ``world.pyrng``
    once per positive candidate, the same stream wolves spawn from, and adding
    one inert candidate to it has been measured at +16% mean deaths.

    Measured cost of the candidate itself, 20 same-seed triples of 60 sim-min:
    no relics 12.70 deaths/colony, relics present but scored 0.001 (the
    candidate and nothing else) 13.85, relics present and actually fetched
    13.10. Paired, live - none is +0.40 deaths with a 95% CI of [-2.86, +3.66];
    the phase shuffle alone is +1.15. So the cost of everybody wanting the loot
    is not distinguishable from the cost of *scoring* it at n=20, and neither is
    distinguishable from zero. Cause of death moved nowhere either: mauled
    68 / 81 / 67 across the three arms.
    """
    s["FetchRelic"] = 0.0
    try:
        if not _fetch_is_wired():
            return
        if not getattr(agent, "alive", True) or getattr(agent, "taken", False):
            return
        # One relic slot per person, so somebody already carrying one is not a
        # candidate - and nobody is ever both impenetrable and holding the BFG.
        if str(getattr(agent, "relic", "") or ""):
            return
        # Children follow a parent. Every other gear-and-danger behaviour in
        # this game excludes them (CraftSpear, FightAnimal, ThrowSpear); a
        # nine-year-old picking up the wyrm-gun is not a feature.
        if _role(agent) == "child":
            return
        # Standing in something that is trying to kill us. The colony-wide cap
        # above clamps needs to 0.35 under hazard; loot gets no score at all,
        # because 0.35 still wins a frame where everything else happens to be
        # zero and "walked into the fire for a trinket" is not a story worth
        # shipping.
        if danger:
            return

        ax = float(getattr(agent, "x", 0.0))
        # ONE question, asked of the file that owns the walk. fetchable_relic
        # applies every gate that makes setting off a bad idea - nothing hunting
        # at either end (FETCH_SAFE_RADIUS), inside FETCH_SEEK, and not one this
        # agent has already failed to reach and is snubbing - so a relic that
        # scores here is a relic that make_combat_action will actually build a
        # walk to. The eligibility checks above are kept because they are
        # cheaper than the call and because they are also the ones this file can
        # justify; the object choice is not this file's to make.
        relic = _relic_target(agent, world)
        if relic is None:
            return
        rx = float(getattr(relic, "x", ax))

        dist = abs(rx - ax)
        near = 1.0 - _clamp01(dist / max(1.0, RELIC_FETCH_RANGE))
        age = max(0.0, float(getattr(relic, "age", 0.0) or 0.0))
        patience = _clamp01(age / max(1.0, RELIC_PATIENCE_SEC))
        ceiling = (RELIC_APPETITE
                   + (RELIC_APPETITE_MAX - RELIC_APPETITE) * patience)
        # Distance is a small term on purpose. It decides *who* goes - the
        # nearest villager scores highest and wins the tiebreak - and it is not
        # allowed to decide *whether* anyone does, which is what the ageing
        # term is for. At 0.10 the far end of RELIC_FETCH_RANGE gives up ten
        # percent of the ceiling, which is enough to order seven people and not
        # enough to strand a relic on the far side of the map.
        # Everybody except the nearest candidate takes RELIC_BYSTANDER, so one
        # person goes and the rest only join in if they had nothing better on.
        claimant = _closest_claimant(world, rx)
        damp = 1.0 if (claimant is None or claimant is agent) else RELIC_BYSTANDER
        s["FetchRelic"] = _clamp01(
            min(RELIC_APPETITE_MAX, ceiling * (0.90 + 0.10 * near)) * damp)
    except Exception:
        log.debug("relic scoring failed", exc_info=True)
        s["FetchRelic"] = 0.0


def _score_raise(s: dict[str, float], agent: Any, world: Any,
                 danger: Any) -> None:
    """Score ``RaiseTheDead`` in place. Same construction as :func:`_score_loot`.

    ``World.raise_the_dead()`` shipped with **zero callers**. The cairnstone is
    the second most common drop in the table, so the most likely thing to find
    in the dirt was an item with no use, and the ``stats["raised"]`` line that
    was carefully kept out of ``stats["born"]`` counted nothing for anybody.

    The ageing clock is the GRAVE'S OWN AGE, not a timer on the bearer. props.py
    already advances ``state["age"]`` on every headstone each tick, and it
    persists, so this needs no new state on the agent and no new field in a
    save - the same trick the loot appetite plays with ``Relic.age``. It reads
    correctly too: an old headstone is one nobody has done anything about.

    A fresh grave still scores :data:`RAISE_APPETITE`, which already beats every
    calm-colony job, so the ageing term is a backstop rather than the mechanism.
    That is on purpose: the stone-hut upgrade's failure was a job that was only
    ever *bounded* above, and this one has a deadline - the bearer can die, and
    ``MAX_GRAVES`` weathering will eventually take the headstone.
    """
    s["RaiseTheDead"] = 0.0
    try:
        if not _kind_is_wired("RaiseTheDead"):
            return
        if not getattr(agent, "alive", True) or getattr(agent, "taken", False):
            return
        # Not gated on `_role(agent) == "child"` or on carrying the stone here:
        # raisable_grave applies both, and duplicating them is what this change
        # is undoing. The two gates below are the ones only this file can make.
        if danger:
            return
        grave = _raise_target(agent, world)
        if grave is None:
            return

        age = 0.0
        try:
            state = getattr(grave, "state", None)
            if isinstance(state, dict):
                age = max(0.0, float(state.get("age", 0.0) or 0.0))
        except (TypeError, ValueError):
            age = 0.0
        patience = _clamp01(age / max(1.0, RAISE_PATIENCE_SEC))
        s["RaiseTheDead"] = _clamp01(
            RAISE_APPETITE + (RAISE_APPETITE_MAX - RAISE_APPETITE) * patience)
    except Exception:
        log.debug("raise scoring failed", exc_info=True)
        s["RaiseTheDead"] = 0.0


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


def _mk_upgrade(agent: Any, world: Any) -> Action | None:
    """Target the hut the director published, or nothing at all.

    Returning None when there is no job follows `_mk_clean`: a stale score can
    then never launch an action with no premise, so the villager drops straight
    through to the next-best thing instead of spending a decision cycle on a
    job that fails on its first update. `_h_upgrade` still re-reads the job id
    itself, for the action that was serialised and reloaded.
    """
    job = getattr(world, "upgrade_job", None)
    if not isinstance(job, dict):
        return None
    sid = job.get("id")
    if sid is None:
        return None
    return make_action("UpgradeStructure", target=sid)


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
    "UpgradeStructure": _mk_upgrade,
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
        _stake_upgrade(world, reg)
    except Exception:
        log.debug("upgrade selection failed", exc_info=True)
        try:
            world.upgrade_job = None
        except Exception:
            pass

    try:
        queue: list[dict[str, Any]] = []
        # Where the people live, for the proximity test below. Read once: this
        # is a mean over the registry and the queue can be several entries long.
        # `settlement_center`, not `colony_center`, for the reason that function
        # documents - a barricade sits about STAGE_HALF out and folding one into
        # the mean moves the answer by that over the building count.
        try:
            home_x = settlement_center(world)
        except Exception:
            home_x = None
        for s in reg.incomplete():
            prio = KIND_PRIORITY.get(s.kind, 0.5)
            # A crossing's blocking priority is earned by proximity - see
            # CROSSING_FAR_PRIORITY. Off the stage it is an expedition, and an
            # expedition does not outrank the roof the colony sleeps under.
            if (s.kind in CROSSING_KINDS and home_x is not None
                    and abs(float(s.x) - home_x) > STAGE_HALF):
                prio = CROSSING_FAR_PRIORITY
            queue.append({
                "id": int(s.id),
                "kind": str(s.kind),
                "x": float(s.x),
                "y": float(s.y),
                "stage": int(s.stage),
                "completion": float(s.completion()),
                "priority": prio,
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

        # The upgrade's shortfall is folded in AFTER the build queue's, and with
        # max() rather than +=. `_gather_urgency` reads this dict, so this is
        # what sends somebody to the quarry for the last two stone an upgrade
        # wants when nothing else needs any. max() is the point: an ordinary
        # build's stone need is always the larger number and so always the one
        # that survives, and an upgrade can never inflate a real build's
        # shortfall into a bigger emergency than it is.
        job = getattr(world, "upgrade_job", None)
        if isinstance(job, dict):
            for res, qty in (job.get("needs") or {}).items():
                try:
                    deficit = int(qty) - stock_qty(world, str(res))
                except (TypeError, ValueError):
                    continue
                if deficit > 0:
                    short[str(res)] = max(short.get(str(res), 0), deficit)
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


def crossing_reach(world: Any, reg: StructureRegistry | None = None,
                   blocking: bool = True) -> float:
    """How far from the settlement a non-urgent crossing may be staked.

    A budget, not a ban - see :data:`CROSSING_REACH_BASE`. Starts at a camera
    width for a crossing the colony is blocked by, at the stage for one it
    merely fancies, and grows with the roofs it has actually got up, so a
    settlement with shelter can still go and bridge the far side and one without
    a roof cannot spend its last two pairs of hands walking there.

    Never raises: a world that will not say how many huts it has reads as none,
    which is the tight end of the budget and the safe answer.
    """
    reg = reg if reg is not None else structures_of(world)
    huts = 0
    if reg is not None:
        try:
            huts = int(reg.count("hut", built_only=True))
        except Exception:
            huts = 0
    base = CROSSING_REACH_BASE if blocking else CHASM_REACH_BASE
    cap = CROSSING_REACH_MAX if blocking else CHASM_REACH_MAX
    return float(min(cap, base + CROSSING_REACH_PER_HUT * max(0, huts)))


def _within_reach(world: Any, lo: float, hi: float,
                  reg: StructureRegistry | None = None,
                  home_x: float | None = None, blocking: bool = True) -> bool:
    """Is the near end of ``[lo, hi]`` inside :func:`crossing_reach`?

    Measured to the NEAR rim, because that is the end a builder walks to and
    stands on - `_bridge_pair` anchors the deck on home ground and `pick_site`'s
    chasm fallback anchors at the near rim for the same reason.
    """
    x = settlement_center(world) if home_x is None else float(home_x)
    near = min(abs(float(lo) - x), abs(float(hi) - x))
    return near <= crossing_reach(world, reg, blocking)


def chasm_in_reach(world: Any, reg: StructureRegistry | None = None
                   ) -> tuple[float, float] | None:
    """:func:`find_chasm`, but only if the colony could afford to walk to it.

    The build order's legacy "there is a chasm, bridge it eventually" entry and
    :func:`pick_site`'s fallback for it both go through here, so the kind the
    order asks for and the site it gets can never disagree - a kind that is
    wanted but unsiteable is the jam `_stake_out_site`'s skip set exists for,
    and there is no reason to create another one.
    """
    span = find_chasm(world)
    if span is None:
        return None
    return (span if _within_reach(world, span[0], span[1], reg, blocking=False)
            else None)


def next_build_kind(world: Any, reg: StructureRegistry | None = None,
                    skip: Any = ()) -> str | None:
    """What the colony wants next: a crossing if one is blocking, then
    firepit -> stockpile -> hut -> more huts -> wall -> watchtower ->
    bridge (if a chasm exists) -> totem at a milestone.

    *skip* is the set of kinds the caller has already failed to find a site for
    this pass, so the order can step over one it cannot place instead of jamming
    on it. This is a real stall, not a hypothetical: the order asks for a second
    barricade unconditionally once the colony is BARRICADE_MIN_POP strong, and a
    colony seated within ~544 px of a rim of the map has no land on one of its
    two approaches at all - the band is entirely off the world, `pick_site`
    answers None every pass forever, and every line BELOW the barricade (huts 4
    through 7, walls, watchtower, totem, the second firepit) is never reached.
    The same jam was always reachable via "the band is nothing but cliff"; the
    wide world just made it common.
    """
    reg = reg if reg is not None else structures_of(world)
    if reg is None:
        return None
    pop = len(alive_agents(world))
    try:
        skipped = frozenset(str(k) for k in skip)
    except TypeError:
        skipped = frozenset()

    def want(kind: str) -> bool:
        return kind not in skipped

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
    if blocking is not None and want(str(blocking["kind"])):
        return str(blocking["kind"])

    if built("firepit") < 1 and want("firepit"):
        return "firepit"
    if built("stockpile") < 1 and want("stockpile"):
        return "stockpile"
    if built("hut") < 1 and want("hut"):
        return "hut"
    # Edge defence is interleaved with expansion, not deferred behind it, so it
    # goes up *as the colony progresses* rather than after five huts. First
    # shelter, then the first spiked barricade on one edge, then keep growing,
    # and the second edge once the settlement is a bit bigger. Placement (which
    # edge) is decided in pick_site; here we only ask for the next one.
    if pop >= BARRICADE_MIN_POP and built("barricade") < 1 and want("barricade"):
        return "barricade"
    if built("hut") < 3 and want("hut"):
        return "hut"
    if pop >= BARRICADE_MIN_POP and built("barricade") < 2 and want("barricade"):
        return "barricade"
    want_huts = min(MAX_HUTS, max(1, (pop + 1) // 2))
    if built("hut") < want_huts and want("hut"):
        return "hut"
    if pop >= 4 and built("wall") < 1 and want("wall"):
        return "wall"
    if pop >= 5 and built("watchtower") < 1 and want("watchtower"):
        return "watchtower"
    # The one line in this order that had no population floor and no notion of
    # where the colony lives, sitting between two that have both. It is also the
    # most expensive thing the order ever asks for. A colony too small for a
    # watchtower falls straight through the two lines above and lands here, and
    # `find_chasm` will happily name a dip 4307 px away on a 6400 px map. Both
    # halves of that are now spelled out: see CHASM_BRIDGE_MIN_POP and
    # CROSSING_REACH_BASE. This is the *unblocked* bridge - "there is a gap over
    # there and one day we should span it". The blocked case, where the colony
    # genuinely cannot reach its food or one of its own, is `plan_crossing` at
    # the top of this function and is not gated on headcount at all.
    if (pop >= CHASM_BRIDGE_MIN_POP and built("bridge") < 1 and want("bridge")
            and chasm_in_reach(world, reg) is not None):
        return "bridge"
    if pop >= TOTEM_POP and built("totem") < 1 and want("totem"):
        return "totem"
    want_walls = min(MAX_WALLS, pop // 3)
    if built("wall") < want_walls and want("wall"):
        return "wall"
    if built("firepit") < min(2, 1 + pop // 6) and want("firepit"):
        return "firepit"
    return None


def _stake_out_site(world: Any, reg: StructureRegistry) -> None:
    """Place the next wanted building as an unbuilt Structure, so builders have
    somewhere concrete to go."""
    try:
        pending = reg.incomplete()
    except Exception:
        return
    # An unsiteable kind is stepped over, not stalled on. `pick_site` answering
    # None used to end the pass, so a kind the map cannot hold blocked every
    # kind below it in the order for the rest of the run - see next_build_kind's
    # docstring for the barricade-against-a-rim case that makes this common on a
    # 6400 px world. Bounded at four tries because this is on the director's 2 s
    # cadence and there is no point walking the whole order every time.
    #
    # Safe for determinism: neither next_build_kind nor pick_site draws from the
    # seeded stream (the only rng use is reg.create below, which happens at most
    # once), and find_cutoff/find_chasm are both cached, so the extra passes are
    # reads. Only the accepted site consumes randomness, exactly as before.
    skip: set[str] = set()
    for _ in range(4):
        kind = next_build_kind(world, reg, skip=skip)
        if kind is None:
            return
        # A blocking crossing is allowed one slot over the concurrency cap.
        # Without it a barrier that appears while two ordinary sites are already
        # open - a miner opening a pit, which is exactly the case this exists
        # for - waits for a hut to finish before anyone even stakes the way out.
        cap = MAX_CONCURRENT_SITES + (1 if kind in CROSSING_KINDS else 0)
        if len(pending) >= cap:
            return
        if any(s.kind == kind for s in pending):
            return
        site = pick_site(world, reg, kind)
        if site is None:
            skip.add(kind)
            continue
        x, y, extra = site
        reg.create(kind, x, y, rng=rng_of(world), state=extra or None)
        chronicle(world, _stake_line(world, kind))
        log.debug("director staked %s at %.0f", kind, x)
        return


# ===========================================================================
#  Stone-hut upgrades
# ===========================================================================
def stone_surplus(world: Any, reg: StructureRegistry | None = None) -> int:
    """Stone in the store that nothing else already has a claim on.

    This one subtraction is the whole reason the stone-hut tier cannot starve a
    real build. The upgrade is paid for out of what is left over *after* every
    incomplete site and every other in-flight upgrade has been costed, so an
    upgrade is only ever affordable when the firepit/hut/barricade line is not
    waiting on stone - even though re-walling one hut costs 2.5x a whole timber
    hut's stone.

    It is also why the gate is a surplus rather than a stone *target*: a target
    would be a work order. Working poses multiply hunger 1.6x and fatigue 1.5x
    (entities.py), and morale targets 1 - max(hunger, fatigue, warmth), so
    sending the colony to the quarry to afford the upgrade would lower the very
    morale that opens the tier. A surplus is passive; it is noticed, not chased.

    Returns the raw int, which may be negative when the queue is over-committed.
    """
    try:
        have = int(stock_qty(world, RES_STONE))
    except (TypeError, ValueError):
        return 0
    reg = reg if reg is not None else structures_of(world)
    if reg is None:
        return have
    owed = 0
    try:
        for s in reg.incomplete():
            owed += int(s.total_remaining_cost().get(RES_STONE, 0))
        for s in reg.all():
            if s.is_upgrading:
                owed += int(s.upgrade_missing().get(RES_STONE, 0))
    except Exception:
        log.debug("stone_surplus accounting failed", exc_info=True)
        return 0
    return have - owed


def pick_hut_upgrade(
    world: Any, reg: StructureRegistry | None = None
) -> Structure | None:
    """The one hut the colony should be re-walling in stone, or None.

    The step order below is load-bearing, and step 2 in particular:

    * Resuming an in-flight job is checked FIRST, ahead of both the tier latch
      and the affordability test. If the surplus check came first, a stockpile
      that dipped below the price mid-job would drop `world.upgrade_job` to
      None, `score_actions` would zero UpgradeStructure, and the builder would
      wander off a half-delivered upgrade that nothing would ever pick back up.
      Checking it first also means a save taken mid-upgrade always resumes.
    * Concurrency is 1, colony-wide. Two half-re-walled huts is worse reading
      than one finished one, and it keeps the stone accounting to a single job.

    No RNG is consulted anywhere here, so which hut goes to stone is
    reproducible from the seed.
    """
    reg = reg if reg is not None else structures_of(world)
    if reg is None:
        return None
    try:
        for s in reg.all():
            if s.is_upgrading:
                return s                      # 2. resume, always
        if not getattr(world, "hut_tier_unlocked", False):
            return None                       # 3. the tier is not learned yet
        cands = [s for s in reg.all() if s.can_upgrade()]
        if not cands:
            return None
        if stone_surplus(world, reg) < sum(HUT_UPGRADE_COST.values()):
            return None
        # The oldest hut standing turns to stone first: the founding house being
        # rebuilt, which reads as the settlement returning to its own origin
        # rather than as a random hut changing colour. Tie-broken on the lowest
        # id so two huts raised in the same tick still order deterministically.
        return max(cands, key=lambda s: (s.standing_t, -s.id))
    except Exception:
        log.debug("pick_hut_upgrade failed", exc_info=True)
        return None


def _stake_upgrade(world: Any, reg: StructureRegistry) -> None:
    """Publish `world.upgrade_job` - the upgrade's equivalent of a build site.

    Deliberately NOT an entry in ``world.build_queue``. That is the obvious
    implementation and it fails silently: `score_actions` reads ``queue[0]``
    only, so a 0.30-priority upgrade sorted to the back would never be scored,
    and `_h_build` sets ``a.done`` the instant ``s.built`` is True - which an
    upgrading hut always is - so a builder who did reach it would burn an AI
    tick per pass doing nothing. Its own channel and its own action kind is what
    makes the tier actually run.

    `world.upgrade_job` is derived state, rebuilt from scratch on every director
    pass and never persisted. The truth lives in ``Structure.state["upgrade"]``;
    a saved copy would be a second source of truth that can disagree with the
    first.
    """
    s = pick_hut_upgrade(world, reg)
    if s is None:
        world.upgrade_job = None
        return
    if not s.is_upgrading:
        s.start_upgrade(float(getattr(world, "world_time", 0.0)))
        # Only on the frame the job actually opens - never on the resume path,
        # or a reload would re-announce an upgrade the colony started an hour
        # ago and the chronicle would read as a stutter.
        chronicle(world, "They have stone enough to re-wall the hut.")
    up = s.state.get("upgrade") or {}
    world.upgrade_job = {
        "id": int(s.id),
        "kind": "hut",
        "x": float(s.x),
        "y": float(s.y),
        "completion": float(s.upgrade_completion()),
        "priority": HUT_UPGRADE_PRIORITY,
        "needs": {k: int(v) for k, v in s.upgrade_missing().items()},
        # Carried through from the structure, which is the only copy that
        # survives a reload - this dict is rebuilt every director pass.
        "opened": up.get("opened"),
    }


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
    """Choose where to put a `kind`. Returns (x, y, extra_state) or None.

    Sites are measured from ``settlement_center``, not ``colony_center``: the
    latter averages the barricades in, and a barricade is deliberately about
    STAGE_HALF out, so one unpaired outwork drags every subsequent site a couple
    of hundred px toward it. See :func:`actions.settlement_center`.
    """
    spec = structure_spec(kind)
    center = settlement_center(world)

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
        # Reach-gated through the same helper the order asks with, so the two
        # cannot disagree about which chasms exist.
        span = chasm_in_reach(world, reg)
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
                               4.0, float(WORLD_W - 5)))
        return anchor, rim, {"w": (x1 - x0) + 18.0, "span": [float(x0), float(x1)]}

    # A barricade does not belong near the colony centre like everything else -
    # it belongs out at the edge the animals arrive from. Only the x-choice
    # differs; the caller still creates it as a normal unbuilt Structure.
    if kind == "barricade":
        return _barricade_site(world, reg)

    lo = max(24.0, center - SITE_RANGE)
    hi = min(float(WORLD_W - 24), center + SITE_RANGE)
    if hi - lo < 24.0:
        # The clamp collapsed the window - the colony is jammed against a rim.
        # Fall back to the stage (colony +/- STAGE_HALF), NOT to the whole map,
        # which is what this line used to say back when they were the same
        # thing. On a 6400 px world "the whole map" is both wrong (it will
        # happily stake a hut 3000 px from the people who have to build it) and
        # expensive: `ys`/`slopes` below are Python loops over `xs`, so a
        # 6-px-step sweep of the map is 1063 ground_y calls on the director's
        # 2 s cadence instead of 267.
        lo = max(24.0, center - STAGE_HALF)
        hi = min(float(WORLD_W - 24), center + STAGE_HALF)
        if hi - lo < 24.0:
            return None
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
    """A site for a barricade, hard against whichever approach has none yet.

    A barricade wants to sit in the band within ``BARRICADE_EDGE_FRAC *
    RENDER_W`` of the point where an incursion first becomes visible, on
    non-cliff ground, as close to that point as the terrain allows so it bites
    the moment the animal lands. Near approach is filled first, then the far.
    Returns ``(x, y, {})`` in the same shape as :func:`pick_site`, or None if
    both sides are covered or the band is nothing but cliff.

    THE EDGE IS THE COLONY'S EDGE, NOT THE MAP'S. This used to read "the animals
    cross in at x = 0 and x = RENDER_W", which was true only while the world and
    the view were the same 1600 px. Left as ``RENDER_W -> WORLD_W`` it puts the
    spikes 1000 px from the rims of a 6400 px map - up to 2600 px from the
    settlement, permanently off camera, defending empty hillside, while animals
    that now enter at ``actions.offstage_x`` walk straight past them. That is
    the whole barricade feature quietly deleted by a rename.

    So the anchor is ``settlement_center() +/- BARRICADE_STANDOFF``, just inside
    the frame edge. Animals enter at OFFSTAGE (960) and walk in; STAGE_HALF
    (800) is the first x at which anyone can see it happen, so a barricade a
    little inside that is the first thing an incursion meets on screen. That is
    what the old ``x = 6`` on a 1600 px world was - 794 px from a colony sitting
    at the middle of it. See BARRICADE_STANDOFF for why it is not 800 exactly.

    Deliberately NOT centred on OFFSTAGE, which is where the contract's letter
    points: half of such a band is past STAGE_HALF and therefore never drawn,
    and a defence the player cannot watch work is not a defence, it is
    bookkeeping. The band keeps its px width (BARRICADE_EDGE_FRAC of RENDER_W,
    the stage, ~256 px) and runs INWARD from the anchor, so every candidate is
    on camera.

    The centre is ``settlement_center``, NOT ``colony_center``: a barricade
    that has already gone up must not move the point the next one is measured
    from, and ``colony_center`` averages it in. See that function for the
    measurement.
    """
    band = max(24.0, BARRICADE_EDGE_FRAC * float(RENDER_W))
    mid = settlement_center(world)

    # Which approaches already hold a barricade (built or still going up)?
    #
    # A barricade further than STAGE_HALF from the settlement does not count.
    # Barricades are permanent and colonies are not: measured on seed 88 at 25
    # sim-minutes, a barricade correctly staked at -640 ended up 1022 px out
    # because the colony crossed a ladder and rebuilt itself 380 px to the east,
    # leaving its west approach undefended and *marked as defended*. Ignoring
    # the stranded one lets the director stake a replacement where the people
    # now live. The churn is bounded without any extra bookkeeping: the build
    # order asks for at most two BUILT barricades (`built("barricade") < 2`)
    # and counts them wherever they stand, so a colony that keeps moving gets
    # two and then stops, rather than a fence post every time it wanders.
    left_has = right_has = False
    try:
        for s in reg.of_kind("barricade"):
            if abs(float(s.x) - mid) > STAGE_HALF:
                continue
            if float(s.x) <= mid:
                left_has = True
            else:
                right_has = True
    except Exception:
        pass

    # Candidate edges, left first, that still need one. Trying both (rather than
    # committing to the left) means an all-cliff left band falls through to the
    # right instead of stalling the build order forever.
    # Each candidate is (lo, hi, edge): the inward-running band and the point it
    # is measured from. `edge` is the stage rim, so scoring by -|cx - edge| still
    # means "as far out as the terrain allows" - it just stopped meaning "as far
    # out as the MAP allows", which is now a different and much emptier place.
    #
    # The anchors are clamped into the map before the band is cut from them, so
    # a colony seated near a rim gets its band slid onto whatever land there is
    # rather than losing that approach entirely - the left band for a colony at
    # x=178 would otherwise run [-622, -366], i.e. nowhere. The `mid -/+ 24`
    # cap then stops a slid band crossing the settlement and being mistaken for
    # the other side's barricade by the has-test above.
    left_edge = max(4.0, mid - BARRICADE_STANDOFF)
    right_edge = min(float(WORLD_W - 4), mid + BARRICADE_STANDOFF)
    candidates: list[tuple[float, float, float]] = []
    if not left_has:
        candidates.append((left_edge, min(left_edge + band, mid - 24.0),
                           left_edge))
    if not right_has:
        candidates.append((max(right_edge - band, mid + 24.0), right_edge,
                           right_edge))
    if not candidates:
        return None                      # both approaches covered - nothing to do

    for lo, hi, edge in candidates:
        lo = max(4.0, lo)
        hi = min(float(WORLD_W - 4), hi)
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
# Four steps, deliberately separated so each can be tested on its own:
#
#   _reach()         ask the terrain which stretches of map are connected
#   _hazard_faces()  ...and where it kills people without being divided at all
#   find_cutoff()    decide whether any of that costs the colony anything
#   plan_crossing()  turn that into a bridge or a ladder the director can stake
#
# The terrain owns the connectivity graph and caches it against its own
# ``epoch``, so digging a pit, a mudslide or a finished deck all re-survey
# exactly once; the hazardous-face survey is cached on the world against the
# same epoch, for the same reason. Everything in here is on the director's 2 s
# cadence, never per agent per tick.


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

    Always ``False`` for a hazardous face, which by definition does *not*
    divide the map: both of its sides carry the same region id, and answering
    "yes" for a region against itself - which the comparisons below would - has
    every wanted thing on the map claiming to be behind every lethal step near
    it. A hazardous face is answered on the evidence of what it is doing to
    people, never on what happens to be on the far side of it.
    """
    if o.get("hazard"):
        return False
    if region in o["inner"]:
        return True
    if home in o["inner"]:
        return True
    if home <= o["left"] and region >= o["right"]:
        return True
    return home >= o["right"] and region <= o["left"]


def _bool_runs(mask: Any) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs of a 1-D bool mask as half-open ``(a, b)``.

    Same shape as ``Terrain._runs`` - deliberately a copy rather than an import,
    because that one is module-private and this module is the only other place
    that reads a boundary mask the way ``_build_reach`` does.
    """
    m = np.asarray(mask, dtype=np.int8)
    if m.size == 0:
        return []
    d = np.diff(m)
    starts = (np.flatnonzero(d == 1) + 1).tolist()
    ends = (np.flatnonzero(d == -1) + 1).tolist()
    if m[0]:
        starts.insert(0, 0)
    if m[-1]:
        ends.append(int(m.size))
    return list(zip(starts, ends))


#: Whether the hazardous-face survey runs at all. OFF, on measurement.
#:
#: The idea is sound - a face can kill without dividing the map, so the
#: reachability survey (which needs BARRIER_MIN_RELIEF to avoid shattering an
#: ordinary map into 150 regions) cannot see it - and the implementation below
#: does fire end to end. It just does not help, and two independent
#: verifications said so:
#:
#:   * 40-seed A/B, 12 sim-minutes, 9 agents: with the survey on, 20 falls and
#:     338 survivors of 360; with it stubbed to return (), the same. Inside the
#:     run-to-run noise floor - outcome-neutral, not an improvement.
#:   * 10-seed A/B: nine seeds identical with it off, down to the same crossing
#:     timestamps. The single seed it touches - 21, the only 'cliffs' map -
#:     prevented zero falls there and finished with one FEWER survivor (83 -> 82).
#:
#: The reason is structural rather than a tuning miss, which is why it is gated
#: rather than retuned: the survey only produces faces on 'cliffs' terrain (mean
#: 9.6 per map there, under 0.2 everywhere else, 147 of 160 maps return none),
#: and a face is only ever built for once a body is already lying at it. So it
#: cannot prevent a first fall by construction - it answers a death that has
#: already happened, and spends a colony's wood doing it.
#:
#: Left in place rather than deleted because the analysis and the geometry are
#: worth keeping: making it act on the *risk* rather than on a corpse - and
#: proving that pays for the wood - is the work that would make it earn its
#: place. Turn this on again with that measurement in hand.
HAZARD_FACES_ENABLED = False


def _hazard_faces(world: Any) -> tuple[dict[str, Any], ...]:
    """Faces that kill without dividing the map, shaped like obstacles.

    Each entry carries the same keys ``_rank_obstacles`` and ``_site_crossing``
    read off a barrier group - ``lo``, ``hi``, ``relief``, ``left``, ``right``,
    ``inner``, ``mid`` - plus ``hazard`` (always ``True``), the ``lip`` and
    ``toe`` of the face and the ``edge`` people actually go over. ``lo``/``hi``
    bracket the lip and the ground the fall ends on, and ``relief`` is that whole
    drop rather than the face's own height, which is the number a ladder has to
    defeat.

    This is a *survey*, not a work list. On a 'cliffs' map it returns four to
    twenty-two faces (mean 9.6 over 40 seeds; every other style averages under
    0.2 and 147 of 160 return none at all), and only the one a body is lying at
    is ever built for - see the note at ``HAZARD_FACE_SLACK``.

    Two tests, both from the physics rather than from taste:

    * **A slip here does not self-arrest.**  ``CHASM_SAFE_GRADIENT`` is that
      knee, measured by driving the real fall integrator down synthetic faces:
      a slip pitches you forward at a fixed 16 px/s while gravity pulls at
      ``GRAVITY``, so below about 14:1 the face falls away slower than the body
      does and you re-contact it, and above it you do not.  The gradient is read
      as the steepest *single-column* step in the face, because the heightmap is
      linearly interpolated and that is therefore the true local slope - the
      5 px central difference ``column_slope`` reports would read seed 21's
      54 px step as 11.04, sitting exactly on the threshold by coincidence.
    * **There is enough left to fall.**  Not the face's relief: the drop from
      the lip to the deepest ground within ``HAZARD_RUNOUT`` of its foot, which
      is as far as a body gets before gravity has it at ``FALL_LETHAL_SPEED``.

    Anything that already clears ``BARRIER_MIN_RELIEF`` is skipped: that is a
    barrier, ``barriers()`` reports it, and the existing planner owns it.

    Cached against the terrain's ``epoch`` exactly like the reachability sweep,
    so a stamped ladder re-surveys once and the face it defeated drops off this
    list (the effective surface reads the ramp, and the ramp is not steep).
    Never raises: this is on the director's 2 s pass, forever, unattended.
    """
    if not HAZARD_FACES_ENABLED:
        return ()
    terr = getattr(world, "terrain", None)
    epoch = int(getattr(terr, "epoch", 0) or 0)
    try:
        if int(getattr(world, "_bhv_hzf_epoch", -1)) == epoch:
            cached = getattr(world, "_bhv_hzf", None)
            if cached is not None:
                return cached
    except Exception:
        pass
    try:
        found = _compute_hazard_faces(world)
    except Exception:
        log.debug("hazardous face survey failed", exc_info=True)
        found = ()
    for name, value in (("_bhv_hzf", found), ("_bhv_hzf_epoch", epoch)):
        try:
            setattr(world, name, value)
        except Exception:
            pass
    return found


def _compute_hazard_faces(world: Any) -> tuple[dict[str, Any], ...]:
    surf = _surface_of(world)
    lab, _bars = _reach(world)
    if surf is None or lab is None:
        return ()
    w = int(surf.size)
    p = int(HAZARD_PROBE)
    if w <= p + 2 or int(lab.size) != w:
        return ()

    # Same reading of the ground as `Terrain._build_reach`: |dy/dx| over the
    # probe, in whichever direction is uphill, widened to every boundary the
    # probe window covers.
    g = (surf[: w - p] - surf[p:]) / float(p)
    bad = np.abs(g) > MAX_SLOPE_CLIMB
    aided = _climb_aided(world, w)
    if aided is not None:
        # A finished ladder is precisely the thing that makes a face survivable,
        # so a window touching its ramp is not a hazard however steep the rock
        # under it is. Same exemption `is_cliff` and `_build_reach` make.
        bad &= ~(aided[: w - p] | aided[p:])
    cut = np.zeros(w - 1, dtype=bool)
    for k in range(p):
        cut[k : k + bad.size] |= bad

    drop1 = np.abs(np.diff(surf))
    reach = int(math.ceil(HAZARD_RUNOUT))
    out: list[dict[str, Any]] = []
    for a, b in _bool_runs(cut):
        # Boundaries a..b-1 are cut, so columns a..b are the face and a and b
        # are the last standable column on each side.
        if b <= a:
            continue
        face = surf[a : b + 1]
        if float(face.max() - face.min()) >= BARRIER_MIN_RELIEF:
            continue                    # a barrier; barriers() already has it
        # The edge people actually go over is the steepest single column in the
        # run, not its rim: the probe is 3 px wide, so a run typically opens two
        # columns of ordinary hillside before the step itself. Getting this
        # wrong is not cosmetic - it is what a ladder has to be laid across, and
        # a ramp that starts two columns short of the drop answers nothing.
        edge = a + int(np.argmax(drop1[a:b]))
        if float(drop1[edge]) <= CHASM_SAFE_GRADIENT:
            continue                    # a slip here self-arrests
        # Smaller y is higher ground, so the lip is whichever rim is smaller.
        if float(surf[a]) <= float(surf[b]):
            lip, toe, step = a, b, 1
        else:
            lip, toe, step = b, a, -1
        j0 = int(np.clip(toe, 0, w - 1))
        j1 = int(np.clip(toe + step * reach, 0, w - 1))
        lo_j, hi_j = min(j0, j1), max(j0, j1)
        # The deepest ground the body can still be over when gravity has it at
        # the lethal speed. Deepest rather than furthest: where the run-out dips
        # and rises again the fall ends in the dip, and that is the landing.
        land = lo_j + int(np.argmax(surf[lo_j : hi_j + 1]))
        drop = float(surf[land]) - float(surf[lip])
        if drop < FALL_LETHAL_DROP:
            continue                    # too short to reach a killing speed
        region = _region_of(lab, lip)
        out.append({
            "lo": int(min(lip, land)),
            "hi": int(max(lip, land)),
            "relief": drop,
            # A hazardous face divides nothing, so both sides are one region.
            # `_must_cross` refuses these outright and `_joins` has its own
            # branch; the keys are here so an entry is interchangeable with a
            # barrier group everywhere else.
            "left": int(region),
            "right": int(region),
            "inner": frozenset(),
            # The drop itself, not the middle of the run: it is what people go
            # over, what a ladder has to be laid across, and what "nearest"
            # should mean when two of these are close together.
            "mid": float(edge) + 0.5,
            "hazard": True,
            "lip": int(lip),
            "toe": int(toe),
            "edge": int(edge),
        })
    return tuple(out)


def _climb_aided(world: Any, w: int) -> Any:
    """Per-column "a ladder's ramp is here", or ``None`` if the terrain cannot
    say. Duck-typed: several test stubs have no overlays at all."""
    ov = getattr(getattr(world, "terrain", None), "climb", None)
    try:
        arr = np.asarray(ov)
        return np.isfinite(arr) if arr.ndim == 1 and arr.size == w else None
    except Exception:
        return None


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
    slack = HAZARD_FACE_SLACK if o.get("hazard") else HAZARD_SLACK
    lo = float(o["lo"]) - slack
    hi = float(o["hi"]) + slack
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
    if lab is None:
        return None
    if not bars and not _hazard_faces(world):
        return None                     # one connected map: nothing to answer
    # A map can be perfectly connected and still be killing people: 17 of 40
    # 'cliffs' maps report no barrier at all and every one of them carries four
    # or more lethal steps. Only a map with neither leaves early.

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
        #
        # Hazardous faces are in the same pool, and this is the whole reason
        # they exist: on seed 21 the fall deaths are at a 54 px step that is not
        # a barrier and never will be, so before this the fallback surveyed a
        # perfectly connected colony, found no wall anywhere near the bodies,
        # and answered "nothing to build" for twelve minutes while people kept
        # going over the same edge.
        pool = tuple(obs) + _hazard_faces(world)
        lethal = [(o, _hazard_hits(world, o)) for o in pool]
        lethal = [(o, n) for o, n in lethal if n]
        if not lethal:
            return None
        o = max(lethal, key=lambda e: (e[1], -abs(e[0]["mid"] - center)))[0]
        if o.get("hazard"):
            cands.append((2, float(o["mid"]), "falls"))
        else:
            # A rim column belongs to the region it overlooks, so the outer rim
            # on the far side is a point in the far region and needs no search.
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
    if lab is None or surf is None:
        return None
    if not bars and not _hazard_faces(world):
        return None
    home = int(cut["home"])

    # Where the colony stands *inside* its own region - the crossing has to be
    # walkable-to, so it is measured from home, not from a centre that may have
    # been dragged across the gap by whoever is stranded on the far side.
    #
    # `settlement_center` rather than `colony_center`, which is the same
    # correction `_barricade_site` and MINE_KEEP_OUT already carry: a barricade
    # sits deliberately about STAGE_HALF out, so folding one into the structure
    # mean moves this by that distance over the building count. This value now
    # decides which obstacles are in reach at all, not just how ties are broken,
    # so a couple of hundred px of outwork drag is no longer harmless.
    home_x = settlement_center(world)
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
            plan["hazard"] = bool(o.get("hazard"))
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
    obs = _obstacles(world, lab, bars) + _hazard_faces(world)
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
        if o.get("hazard"):
            # `_must_cross` is always False for a face that divides nothing, so
            # a body under it is the only evidence it can rank on - and that is
            # what the colony noticed, whatever else the survey was reporting.
            if not hits:
                continue
            reason = "falls"
        if not hits and not weight:
            continue                    # in the way of nothing, hurting nobody
        entry = dict(o)
        entry["hits"] = int(hits)
        entry["urgent"] = bool(hits) or o["mid"] in (cut.get("trapped") or ())
        # ...and in the way of somebody who can get to it. Distance used to rank
        # nothing here - it was the last tiebreak, under weight - which was
        # sound on a 1600 px map where the furthest wall on the whole world was
        # a stage-width off. On 6400 px it let a colony with a half-built hut
        # 135 px away commit to a wall 1329 px away (seed 2), and `_mk_build`
        # then sent every builder to the one with the higher KIND_PRIORITY.
        #
        # An urgent obstacle - a body at the foot of it, a man trapped inside it
        # - is exempt: that is a rescue, and seed 42's seven fall deaths at a
        # single face are the standing argument for going wherever it is.
        # Everything else has to be inside CROSSING_REACH, which grows with the
        # roofs the colony has up, so the far side stays reachable eventually.
        if not entry["urgent"] and not _within_reach(
                world, o["lo"], o["hi"], home_x=home_x):
            continue
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

    if o.get("hazard"):
        # A face is laddered, and this one cannot be anything else: there is no
        # hole here, so `_bridge_pair`'s depth test could never pass - and both
        # sides carry the same region id, so its "land somewhere that helps"
        # test is meaningless and its search would happily propose a deck to
        # some unrelated region that happens to be within a span's reach. Go
        # straight to the ramp. `a` and `c` already bracket the lip and the
        # ground the fall ends on, so the rise `_ladder_span` measures is the
        # lethal drop rather than the step's own height.
        plan = _ladder_plan(world, surf, lab, a, c, relief, cut, stand_high=True)
        if plan is not None and _joins(lab, o, *plan["span"]):
            plan["reason"] = str(o.get("reason") or cut["reason"])
            return plan
        return None

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

    A hazardous face is the one case region ids cannot answer, because it does
    not divide the map: both of its sides are the same region, and every ramp
    anywhere near it would pass. What has to be true instead is that the ramp
    lies against the lip people actually go over - a ladder built alongside it
    is a ladder that changes nothing.
    """
    if o.get("hazard"):
        edge = int(o["edge"])
        lo_s, hi_s = min(float(x0), float(x1)), max(float(x0), float(x1))
        return lo_s <= float(edge) and hi_s >= float(edge + 1)
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
    cut: dict[str, Any], *, stand_high: bool = False
) -> dict[str, Any] | None:
    """A ladder against the wall between rims ``a`` and ``c``.

    ``stand_high`` moves the *site* - the column a builder is sent to - from the
    foot of the ramp to its head. It changes nothing about the ramp itself; the
    geometry travels in ``span``/``rise`` and is what gets stamped and drawn.

    It exists for hazardous faces, and it is not a preference. The foot of one
    of those is at the bottom of a drop that kills, and unlike a barrier the
    colony is not already living down there - so the trip to the build site is
    the exact trip the ladder is being built to make survivable. Measured on
    seed 247, whose colony was already losing people over a face at x=59: the
    site landed at x=11, in the pocket below it, and over 36 minutes the ladder
    was never finished while the fall count went from 12 to 37 and the site sat
    staked, holding the one-crossing-at-a-time gate shut against everything
    else. Sending the builder to the lip instead is the same ramp, reached from
    the side people are already on.
    """
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
    low = (x0, y0) if y0 > y1 else (x1, y1)
    high = (x1, y1) if y0 > y1 else (x0, y0)
    site_x, site_y = high if stand_high else low
    return {
        "kind": "ladder",
        "x": float(site_x),
        "y": float(site_y),
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


#: How long the shelter lock has to hold before a gatherer is put on the tools.
#:
#: Measured rather than guessed. A control tree with the promotion removed and
#: a read-only observer in its place (proved inert by identical ``to_dict``
#: digests against the plain control) says the "no builder at all" state is NOT
#: the momentary hand-over the first version assumed: over ten seeds and 90
#: sim-minutes each its median life is 128 s and its p90 is 909 s. A debounce
#: on its own therefore separates nothing.
#:
#: What 120 s buys, on top of the other four gates below, is the firing count
#: on colonies that were never in trouble. Firings per seed for the full
#: condition, from those same observer runs:
#:
#:      hold      2    555   70707 | 42   101  1234  7777 12345 40404 333435
#:      ----   ----   ----   ----  | ---  ---  ----  ----  ----  ----  ------
#:        0 s     2      3      7  |  0    0     0     0     0     2       2
#:       60 s     2      3      7  |  0    0     0     0     0     2       2
#:      120 s     2      3      6  |  0    0     0     0     0     0       2
#:      300 s     2      2      4  |  0    0     0     0     0     0       2
#:
#: 120 s is the knee: it is where the last healthy seed (40404) stops firing,
#: and the three colonies that actually collapse are untouched by it - they are
#: still firing at 300 s and seed 2's lock never breaks at all inside 90
#: minutes. Longer than that only starts costing the tail.
SHELTER_LOCK_HOLD = 120.0


def _shelter_lock_candidate(world: Any, agents: list[Any], adults: list[Any],
                            reg: Any) -> Any | None:
    """The gatherer to put on the tools, or None - and usually None.

    This is the *collapse*, not the symptom. Five things have to be true at
    once, and every one of them is load-bearing:

    * the colony is at or below MIN_POP. Not "small": MIN_POP is the floor
      ``_tick_population`` respawns to, so this is a colony on life support.
    * nobody is on the tools at all. Deliberately NOT "below BUILDER_RATIO" -
      see the note in :func:`assign_roles` for the two wider forms that were
      measured and cost more than they saved.
    * the colony cannot house another head. This is character-for-character the
      birth gate in ``World._tick_population`` (``huts <= 0 or n >= huts *
      POP_PER_HUT``), which is the whole reason the trap is a trap: no roof, no
      birth, and at MIN_POP no child either, so the roster can never change.
    * a SHELTER is what is unfinished. The old condition took any outstanding
      work at all, and 43 of the 114 firings measured across ten seeds were for
      a barricade, a watchtower, a totem or a bridge - none of which lift the
      birth gate, so none of which end the trap.
    * the materials for that hut's current stage are already in the store. If
      they are not, the colony is short of hands on the resources, not on the
      tools, and moving its last gatherer is exactly the wrong way round.

    Order matters for cost, not just for meaning: the population test is a
    single ``len`` and it is first, so on a healthy colony this function is one
    integer compare per tick. The old block called ``reg.incomplete()`` every
    tick on every colony.

    Ties break on the lowest id and nothing here draws from the rng, so on the
    seeds where it never fires it re-phases nothing.
    """
    if len(agents) > MIN_POP:
        return None
    workers = [a for a in adults if _role(a) in ("gatherer", "builder")]
    if not workers or any(_role(a) == "builder" for a in workers):
        return None
    # A child about to come of age supplies a builder by itself: `_worker_role`
    # returns "builder" whenever the roster holds none. Vacuous on a world that
    # started fresh - `Population.spawn` defaults to gatherer and nothing in
    # World passes "child", so 0 of the 114 measured firings had a child
    # standing there - but a save restored through `from_dict` can carry one,
    # and pre-empting an arrival that is already on its way is not a rescue.
    for a in agents:
        if _role(a) != "child":
            continue
        age = _age_of(a)
        if age is None or age >= CHILD_MATURE_AGE - SHELTER_LOCK_HOLD:
            return None
    if reg is None:
        return None
    try:
        huts = int(reg.count("hut"))
    except Exception:
        return None
    if huts > 0 and len(agents) < huts * POP_PER_HUT:
        return None
    try:
        unfinished = [s for s in reg.incomplete()
                      if getattr(s, "kind", "") == "hut"]
    except Exception:
        return None
    for s in unfinished:
        try:
            missing = s.missing_for_stage()
        except Exception:
            continue
        if all(stock_qty(world, res) >= int(qty)
               for res, qty in missing.items()):
            return min(workers, key=lambda a: int(getattr(a, "id", 0) or 0))
    return None


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

    # --- and somebody is on the tools ---------------------------------------
    # BUILDER_RATIO was only ever *consulted*, never *enforced*. `_worker_role`
    # is asked which trade the colony is short of, but the only three callers
    # are a child coming of age, an elder standing down and a lookout being
    # stood down - so a colony that loses its builders keeps none until a child
    # grows up, and a colony pinned at MIN_POP never has a child.
    #
    # That is the terminal state of the MIN_POP trap, and it is a lock rather
    # than a slow patch. Seed 2 loses its only builder at t=240 with the first
    # hut at completion 0.50; seventy minutes later the hut is at 0.75 and the
    # store holds 25 wood and 6 fibre against a remaining cost of 5 and 2.
    # Nobody is missing anything - `aff["build"]` is 1.00 for a builder and 0.55
    # for a gatherer, so BuildStructure scores 0.30 against Farm's ~0.50 and the
    # two survivors farm 239 food while the only roof in the colony stands
    # three quarters up. No roof is no sleep and, via the hut gate in
    # `_tick_population`, no births ever: MIN_POP forever.
    #
    # Deliberately NONE, not "below BUILDER_RATIO". Both wider forms were
    # measured over the 14-seed sweep and both cost more than they saved:
    #
    #   * ratio enforced always - fixed both trapped seeds, then took seed 42
    #     from 17/7 to 7/3 and 808 from 20/7 to 14/6. A thriving colony's
    #     roster is already the shape it wants and moving 45% of it onto the
    #     tools takes hands off the food.
    #   * ratio enforced while roofless - reached 9/14 at MAX_POP and 13/14 at
    #     hut #7, but it fires at FOUNDING on every map, because the founding
    #     four are an elder, two gatherers and one builder (1 of 3 workers,
    #     0.33 < 0.45) and every colony starts with no roof. Seed 7777, the
    #     healthiest map in the sweep, went from 20/7 to 6/2 on it.
    #
    # NONE is the condition that is actually a lock rather than a preference,
    # and that part still stands. What did not stand is the other half of the
    # first version's test: "there is work outstanding". That is true on a
    # perfectly healthy colony - anyone can be the last builder, and any of the
    # ten structure kinds can be half-built - so the rescue fired on colonies
    # that were never in trouble, and every firing re-rolls the seeded rng from
    # that moment on. A hostile control tree differing by exactly the old block
    # put numbers on the cost: seeds 1234, 12345, 40404 and 333435 all reach
    # 20 pop / 7 huts without it and 14/5, 14/5, 13/6 and 12/6 with it, and 42
    # and 7777 go 20/7 -> 11/5 and 20/7 -> 10/5 on top.
    #
    # So the test is now the collapse itself rather than its symptom, in
    # `_shelter_lock_candidate`: at or below MIN_POP, nobody on the tools, no
    # room to house another head, an unfinished HUT specifically, and its
    # materials already in the store. Across the ten diagnostic seeds that
    # takes the firing count from 114 to 13, and every one of the 13 is on one
    # of the four seeds that are genuinely stuck.
    #
    # The hold is the fifth gate and it is not decoration. Measured in a tree
    # with the promotion removed, "no builder" is not momentary - median 128 s,
    # p90 909 s - so the debounce is not filtering a hand-over. It is what
    # takes seed 40404 from two firings to none while leaving seeds 2, 555 and
    # 70707 firing; see SHELTER_LOCK_HOLD for the table.
    #
    # `_bhv_shelter_lock_t` is a transient like `_bhv_dir_t` and the other
    # `_bhv_*` marks: it is not in `to_dict`, so it cannot change a save, and a
    # reload simply starts the two minutes again. Ties still break on id and
    # nothing here draws from the rng.
    pick = _shelter_lock_candidate(world, agents, adults, reg)
    if pick is None:
        setattr(world, "_bhv_shelter_lock_t", None)
    else:
        now = world_now(world)
        since = getattr(world, "_bhv_shelter_lock_t", None)
        if not isinstance(since, (int, float)) or now < float(since):
            # First tick of the lock, or a clock that went backwards under us
            # (a save loaded over a running world). Start the clock, do not
            # act on a duration we cannot vouch for.
            setattr(world, "_bhv_shelter_lock_t", now)
        elif now - float(since) >= SHELTER_LOCK_HOLD:
            if _set_role(pick, "builder"):
                chronicle(world, f"{getattr(pick, 'name', 'Someone')} put the "
                                 f"basket down and picked up the tools.")
            # Restart rather than clear: the next call sees a builder and
            # clears it anyway, but if anything upstream reverts the role this
            # tick the colony still waits another two minutes before trying
            # again instead of promoting somebody every frame.
            setattr(world, "_bhv_shelter_lock_t", now)


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
        # WORLD_W wide, not RENDER_W: this stub stands in for the real terrain,
        # and the point of the harness is to run the director against a map the
        # size of the one it will actually get. The founders below sit at ~580,
        # so the chasm at 700-760 is still right next to them.
        def __init__(self) -> None:
            self.height = (600.0 + 40.0 * np.sin(np.arange(WORLD_W) / 180.0)).astype(
                np.float32)
            self.height[700:760] += 90.0          # a chasm to bridge
            self.material = np.zeros(WORLD_W, dtype=np.uint8)

        def ground_y(self, x: float) -> float:
            i = int(max(0, min(WORLD_W - 1, x)))
            return float(self.height[i])

        def slope(self, x: float) -> float:
            i = int(max(1, min(WORLD_W - 2, x)))
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
