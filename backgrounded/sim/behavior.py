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
    APPETITE_MAX,
    BARRICADE_EDGE_FRAC,
    BARRICADE_MIN_POP,
    BARRIER_MIN_RELIEF,
    BUILD_SURPLUS_BONUS,
    CHASM_SAFE_GRADIENT,
    CLEANUP_SCORE_MAX,
    COOK_PILE_BONUS,
    COOK_PILE_SPAN,
    COOK_RAW_GATE,
    COOK_RAW_RESERVE,
    CROSSING_MAX_SPAN,
    CROSSING_MIN_DEPTH,
    FALL_LETHAL_SPEED,
    FARM_FIELD_SIZE,
    FARM_SURPLUS_DAMP,
    FIBRE_VIA_BERRIES,
    FORAGE_SURPLUS_DAMP,
    GATHER_STONE_FLOOR,
    GATHER_STONE_LOW_ARM,
    GATHER_STONE_NEAR_BONUS,
    GATHER_STONE_NEAR_REF,
    GATHER_STONE_SPAN,
    LITTER_CLUSTER_FULL,
    LITTER_CLUSTER_MIN,
    LADDER_MAX_W,
    HERMIT_BUILD_URGE,
    HERMIT_COOK_URGE,
    HERMIT_STASH_CAP,
    HERMIT_STASH_TAPER,
    HERMIT_EDGE_MARGIN,
    HERMIT_FELL_REACH,
    HERMIT_FELL_URGE,
    HERMIT_FIRE_URGE,
    HERMIT_HOMESICK,
    HERMIT_HOST_URGE,
    HERMIT_HUT,
    HERMIT_KEEP_ADULTS,
    HERMIT_MIN_ADULTS,
    HERMIT_ROAM,
    HERMIT_SHELF_HALF,
    HERMIT_STOKE_BELOW,
    HERMIT_STOKE_URGE,
    HERMIT_VISIT_JITTER,
    HERMIT_VISIT_PERIOD,
    HERMIT_VISIT_URGE,
    HERMIT_WOODPILE,
    HERMIT_WOODPILE_URGE,
    LADDER_MIN_RISE,
    LADDER_MIN_W,
    LADDER_SLOPE,
    MAT_DIRT,
    MAT_GRASS,
    MAT_LAVA,
    MAT_STONE,
    MATERIAL_BASELINE,
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
    STOCK_FLOOR,
    STOCK_GLUT_SPAN,
    STOCK_PER_HEAD,
    SURPLUS_DIVIDEND,
    SURPLUS_FALL_SEC,
    SURPLUS_FATIGUE_DAMP,
    SURPLUS_MORALE_FULL,
    SURPLUS_MORALE_MIN,
    SURPLUS_NIGHT,
    SURPLUS_RESIDUAL,
    SURPLUS_RISE_SEC,
    SURPLUS_STEP_SEC,
    SURPLUS_THRIFT_FLOOR,
    WALK_SPEED,
    WORLD_W,
)
from .actions import (
    CARRY_CAP,
    TALK_REACH,
    Action,
    # Private and imported anyway: the one clamp to WORLD bounds every siting
    # decision in the project goes through, so a camp staked here and a step
    # taken there cannot disagree about where the map ends.
    _clamp_x,
    # Private on purpose and imported anyway: it is `settlement_center` with the
    # hermit himself taken out of the agent fallback, and the re-siting test in
    # `_ensure_hermit_hut` has to ask the same question `hermit_home` asks or
    # the two disagree about where the colony is and the camp oscillates.
    _hermit_base,
    agent_by_id,
    alive_agents,
    chronicle,
    colony_center,
    densest_litter,
    find_prop,
    food_in_store,
    free_litter_cluster,
    ground_y,
    colony_repairable,
    fire_for,
    hazards_of,
    hermit_band,
    hermit_fire,
    hermit_home,
    hermit_hut,
    hermit_stash_food,
    hermit_stash_qty,
    hermit_visit_budget,
    hermit_worksite,
    is_hermit,
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
from .names import ROLE_HERMIT
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

#: Let an animal interrupt an in-flight job. See :func:`emergency_override`,
#: which carries the paired A/B this was turned on by. Module-level rather than
#: inlined so a measurement harness can set it in one process and the paired arm
#: in another, which is the only way to compare the two on one seed.
#: TODO(backlog): belongs in constants.py
UNDER_ATTACK_OVERRIDE = True
#: Action kinds that already ARE the response to an animal. Re-deciding one of
#: these every AI tick cannot improve it - ``choose_action`` returns the running
#: machine untouched whenever the winning kind matches - but it does draw a
#: fresh tiebreak per scored candidate on the way, which re-phases the world RNG
#: for everyone else. Cheap to exclude, and it keeps a fight from being re-rolled
#: 60 times while it lasts.
_ANIMAL_BUSY_KINDS = frozenset((
    "FightAnimal", "FleeAnimal", "ThrowSpear", "FireBfg",
))
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

ROLES: tuple[str, ...] = ("gatherer", "builder", "elder", "child", "lookout",
                          "hermit")

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
    # The hermit is the elder's mirror: the elder is the colony's social centre
    # at 1.00 and he is its edge at 0.04. `gather` is the one thing he is good
    # at, and it is what feeds him - ForageBerries takes its target as the bush
    # nearest HIM, so a high gather affinity is self-sufficiency at his own camp
    # rather than a job he does for anybody.
    #
    # `farm` is deliberately LOW even though farming is the better food. Fields
    # are tilled near the settlement (`_tillable_near` measures from the colony
    # centre), so a farming hermit is a commuting hermit. The un-damped
    # `hunger * 0.12` term in the Farm score is left to speak for itself: a
    # hermit who is genuinely starving will still walk to a field, which is the
    # behaviour we want at the one moment it matters.
    #
    # `mine` and `cleanup` are floors rather than zeroes so nothing here divides
    # by an absent key, and because a hermit cracking a rock at his own camp is
    # fine - it is the quarry and the litter sweep, both sited in town, that the
    # low number is buying his way out of.
    "hermit":   {"gather": 0.85, "build": 0.10, "social": 0.04, "watch": 0.20,
                 "farm": 0.15, "mine": 0.05, "cleanup": 0.05},
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

    # --------------------------------------------------------- economy ------
    # Hoisted above the build block because BuildStructure reads the surplus
    # too. The gather block below is where these used to live and is still where
    # most of them are spent.
    #
    # `stocks` is the colony's whole material position in one shape: per
    # resource, how far below its per-head target the store is and how far past
    # it. `surplus` is the food glut, slewed over minutes and dimmed after dark.
    # `spend` is the fraction of it THIS colonist is willing to lay out: the
    # colony's contentment (`_thrift`, a lagging mean) and his own tiredness.
    # The second is not redundant - a man can be spent while the colony still
    # reads content, and without it the dividend outranks his own daytime Sleep
    # and he works until the cold takes him.
    shortfall = _needs_of(world)
    blocking = _blocking_resources(world)
    stocks = _stock_state(world, pop)
    food_short, food_glut = stocks[RES_FOOD]
    surplus = _surplus(world, food_glut, night)
    spend = (SURPLUS_DIVIDEND * surplus * _thrift(world)
             * (1.0 - SURPLUS_FATIGUE_DAMP * fatigue))

    # ----------------------------------------------------------- build ------
    queue = _queue_of(world)
    if queue and reg is not None:
        item = queue[0]
        needs = item.get("needs") or {}
        avail = _availability(world, needs)
        unmet = 1.0 - 0.55 * float(item.get("completion", 0.0))
        weight = float(item.get("priority", 0.6))
        base = _clamp01(
            aff["build"] * unmet * (0.28 + 0.72 * avail) * (0.55 + 0.45 * weight)
        )
        # A colony with food to spare puts things up. Scaled by `avail` on
        # purpose - this is "the materials are lying at the site, go and raise
        # it", never "walk to a site with nothing at it". At surplus 0 the bonus
        # is exactly 0.0 and this is bit-identical to the old single expression.
        s["BuildStructure"] = _clamp01(base + BUILD_SURPLUS_BONUS * spend * avail)
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
        # `colony_repairable`, not `reg.damaged` - the hermit's hut is not the
        # colony's to mend. See the note on that function.
        dmg = colony_repairable(reg, threshold=0.9)
        if dmg:
            worst = min(st.hp / max(1.0, st.max_hp) for st in dmg)
            s["RepairStructure"] = _clamp01(aff["build"] * (1.0 - worst) * 1.25)

    # ---------------------------------------------------------- gather ------
    # What the *head* of the build queue is short of is the colony's real
    # bottleneck - a build stalled on 2 fibre has to outrank idling. That path
    # is unchanged; it is simply silent most of the time, because with
    # MAX_CONCURRENT_SITES = 2 two structures' worth of cost is the ceiling on
    # queue-derived demand and the store covers it almost always. `_appetite`
    # adds the two demands that do not ask the queue anything.
    wood_urg = _gather_urgency(RES_WOOD, shortfall, blocking, 12.0)
    stone_urg = _gather_urgency(RES_STONE, shortfall, blocking, 10.0)
    fibre_urg = _gather_urgency(RES_FIBRE, shortfall, blocking, 6.0)
    wood_want = _appetite(RES_WOOD, stocks, spend, wood_urg)
    stone_want = _appetite(RES_STONE, stocks, spend, stone_urg)
    fibre_want = _appetite(RES_FIBRE, stocks, spend, fibre_urg)
    cook_want = _appetite(RES_COOKED, stocks, spend)
    # The food arm decays with the glut - including its 0.15 floor, which used
    # to keep foraging alive against a larder twenty times over target. The
    # fibre arms do NOT decay: bushes are the colony's only cordage, so a rich
    # colony strips them for the fibre and the berries are the by-product.
    # `fibre_urg` is kept as its own arm rather than folded into `fibre_want` so
    # a build blocked on fibre pulls exactly as hard as it does today.
    food_urg = max((0.15 + 0.75 * food_short) * (1.0 - FORAGE_SURPLUS_DAMP * surplus),
                   fibre_urg, FIBRE_VIA_BERRIES * fibre_want)

    # `stone_low` is hoisted out of the mine block below because BOTH stone jobs
    # read it now. That is the whole of the GatherStone fix: Mine has always
    # used `max(stone_low, stone_want)` and GatherStone used `stone_want` alone,
    # and since `_appetite` only carries the standing per-head target through at
    # MATERIAL_BASELINE (0.45) strength, Mine saw the full shortfall while
    # GatherStone saw 45% of it. Both then read the same `stone_want`, so the
    # surplus dividend lifted the quarry far harder than the loose rock - which
    # is exactly what shipped (Mine 1.55% -> 3.75% of colonist-time, GatherStone
    # 2.11% -> 2.28%). Cracking a rock and digging a quarry are different things
    # to look at and the user asked for both.
    stone_low = stocks[RES_STONE][0]
    s["GatherWood"] = _clamp01(aff["gather"] * (0.10 + 0.90 * wood_want))
    s["ForageBerries"] = _clamp01(aff["gather"] * food_urg + hunger * 0.30)
    # Claim-aware: a prop another villager has already reserved does not count as
    # available here, so an agent will not pick GatherWood only to find every
    # tree taken - it scores the job zero and does something else instead.
    if _find_target_prop(world, _TREE_KINDS, ax, agent) is None:
        s["GatherWood"] = 0.0
    # The rock is fetched as an object rather than a None-test because its
    # DISTANCE is now part of the score. `find_prop` already refuses anything
    # past 720 px, but a flat score rated a rock at 700 px exactly as highly as
    # one underfoot, and this job had no walk term of any kind while Mine has
    # MINE_MAX_WALK. `near` is the mirror of Mine's `scarce_bonus`: Mine is paid
    # for there being no loose rock about, GatherStone is paid for there being
    # one close. Reachable max is 0.878 against the old expression's 0.905, so
    # the ceiling falls and nothing here can newly cross OVERRIDE_FLOOR.
    rock = _find_target_prop(world, ("rock", "boulder", "stone", "outcrop"), ax, agent)
    if rock is None:
        s["GatherStone"] = 0.0
    else:
        near = _clamp01(
            1.0 - abs(float(getattr(rock, "x", ax)) - ax) / GATHER_STONE_NEAR_REF)
        s["GatherStone"] = _clamp01(
            aff["gather"] * (GATHER_STONE_FLOOR
                             + GATHER_STONE_SPAN * max(GATHER_STONE_LOW_ARM * stone_low,
                                                       stone_want)
                             + GATHER_STONE_NEAR_BONUS * near))
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
        # `field_bonus` and `ripe_bonus` were additive and NOT scaled by
        # food_short, so a ripe field pulled a flat 0.435 against a larder
        # sixteen times over target and Farm ate 28% of all colonist time. Both
        # now sit inside the damp. `hunger` stays OUTSIDE it, and that is the
        # anti-starvation guard: a hungry individual goes to the field however
        # rich the colony is. Safe to damp at all because crops are perennial -
        # harvest resets growth rather than removing the plant, so an unpicked
        # field waits and costs no re-tilling.
        s["Farm"] = _clamp01(
            (aff.get("farm", 0.5) * (0.18 + 0.82 * food_short + field_bonus)
             + ripe_bonus) * (1.0 - FARM_SURPLUS_DAMP * surplus)
            + hunger * 0.12
        )
    else:
        s["Farm"] = 0.0

    # ------------------------------------------------------------ mine ------
    # A sustained dig, complementing GatherStone. Rises as stored stone falls or
    # a build is blocked on stone, and is preferred when loose rocks are scarce.
    # `stone_low` is now `stocks[RES_STONE]`'s short half, and STOCK_PER_HEAD /
    # STOCK_FLOOR reproduce the old inline `max(6, pop * 3)` to the digit - the
    # target that used to be Mine's private special case is now the general
    # rule, and GatherStone reads it too. Kept as its own `max()` arm rather
    # than relying on `stone_want` alone so APPETITE_MAX cannot lower Mine's
    # existing ceiling: at an empty quarry this is bit-identical to today.
    # `stone_low` is now hoisted into the gather block above, which is the only
    # place it moved to - Mine's own expression is untouched.
    if _mineable_near(world, ax, agent):
        loose_rock = _find_target_prop(world, ("rock",), ax, agent) is not None
        scarce_bonus = 0.0 if loose_rock else 0.20
        # A higher floor than most jobs on purpose: a colony keeps a quarry
        # ticking over even when the stores are full, so mining is a visible
        # ongoing activity rather than only a stone-emergency response.
        s["Mine"] = _clamp01(
            aff.get("mine", 0.5)
            * (0.30 + 0.70 * max(stone_low, stone_want) + scarce_bonus)
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
    # Cooked meals are a resource with a per-head target like any other, which
    # replaces the old hand-written "is the pantry literally empty" pair. At
    # zero cooked this now scores 0.74 x aff against the old 0.55, and at a full
    # pantry it decays to 0.20 x aff instead of standing at 0.30 forever.
    #
    # The one job a surplus buys that cannot cost the colony a meal: cooking
    # converts raw to cooked one for one and BOTH count toward stored food, so
    # it can never push the larder down. It self-limits three ways over - on the
    # `raw >= 3` gate, on the cooked target, and on the glut past it - which is
    # what stops the runaway simply moving from the food bin to the cooked one.
    # ...which is all true, and none of it made anyone cook. Measured over 5
    # seeds x 15 sim-min (827 scored samples): mean 105.2 raw food standing
    # against mean 2.7 cooked, and CookFood was top of the board in 7 of 807
    # live samples - 0.87%. Two diagnoses were offered for that and BOTH were
    # incomplete. It is not that BuildStructure blocks it and it is not that
    # ForageBerries blocks it: CookFood was beaten by BuildStructure in 75.5% of
    # samples, Farm 56.0%, GatherWood 49.4%, ForageBerries 45.8% and Mine 45.2%.
    # It loses to everything, because the only demand it could read was "the
    # pantry is empty" and that tops out at aff x 0.74.
    #
    # The missing term is the one thing that makes cooking different from every
    # other job here: it is a CONVERSION, not a production. It cannot cost the
    # colony a unit of food - `_h_cook` puts back exactly the 3 it took, as
    # cooked - and `_stock_state` counts both halves as RES_FOOD. So a big raw
    # pile with a fire standing idle is a reason to cook *by itself*, entirely
    # separate from how many meals are ready. That is `pile`, and it is added on
    # top of the pantry appetite rather than folded into it so the two demands
    # stay legible.
    #
    # Three things stop it running away, and they are the same three as before
    # plus a stricter gate:
    #   * `1.0 - cook_glut` kills the bonus as the pantry fills past target and
    #     zeroes it at 2.5x, while the base appetite decays over the same span;
    #   * `pile` is measured PER HEAD, so a big colony needs a proportionally
    #     bigger pile before the fire is worth lighting;
    #   * the gate is no longer a flat 3. `raw >= 3` let a colony of eight with
    #     three food left and everyone hungry hand its entire larder to one man
    #     for COOK_TIME and leave the store empty behind him. The gate now rises
    #     with the colony's own food shortfall - 35 raw at full shortfall and
    #     pop 8, and still exactly 3 when the larder is comfortable, so it is
    #     stricter than the old one everywhere and looser nowhere.
    raw = stock_qty(world, RES_FOOD)
    gate = COOK_RAW_GATE + COOK_RAW_RESERVE * stocks[RES_FOOD][0] * pop
    if fire is not None and raw >= gate:
        pile = _clamp01((raw - gate) / max(1.0, COOK_PILE_SPAN * pop))
        s["CookFood"] = _clamp01(
            aff["gather"] * (0.20 + 0.60 * cook_want)
            + COOK_PILE_BONUS * pile * (1.0 - stocks[RES_COOKED][1])
        )
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
    # And the hermit, last of all and for the same reason as those two: he has
    # to be able to lose to the wolf combat_actions has just scored - his pull
    # home must never outrank running away - and to beat everything the colony
    # was calmly getting on with. Scored here rather than up with the chores
    # because the combat merge is a `dict.update` and would otherwise put the
    # colony's workbench back on his list after he had walked away from it.
    if role == ROLE_HERMIT:
        _hermit_bias(s, agent, world, night, fatigue, warmth, carry_qty,
                     carrying, danger is not None)
    # And the one colonist who has been sent to see him. Scored here, beside the
    # hermit's own bias and after the combat merge, for the same three reasons:
    # it has to beat the chores the colony was calmly getting on with, it has to
    # lose to a wolf combat_actions has just scored, and a `dict.update` merge
    # after it would put the colony's own Converse target back on his list.
    #
    # Not applied while anything dangerous is on the map. A man who keeps
    # walking toward the far edge of the world with a wolf behind him is not
    # being sociable, he is being killed by a feature, and the visit slot's own
    # watchdog will hand the appointment back inside `hermit_visit_budget`.
    elif danger is None and hermit_guest(world) is agent:
        s["Converse"] = HERMIT_VISIT_URGE
    return s


def _tree_in_reach(world: Any, agent: Any) -> bool:
    """Is there timber near HIS CAMP for him to cut? Never raises.

    Asks the same question `actions._h_gather` will ask when it picks the target
    - nearest tree to HERMIT_HOME, not to him, inside HERMIT_FELL_REACH - so the
    score and the action can never disagree and he never picks a job that fails
    on sight. Two callers now (raising his camp, and feeding the fire once it is
    up), which is why it is a function rather than four lines inlined twice.
    """
    try:
        return find_prop(world, _TREE_KINDS, hermit_home(world),
                         max_dist=HERMIT_FELL_REACH,
                         claimant=getattr(agent, "id", None)) is not None
    except Exception:
        return False


def _stash_appetite(world: Any, res: str) -> float:
    """0..1 appetite for putting MORE *res* into the hermit's own pile.

    1.0 while there is room to spare, easing over the last
    ``HERMIT_STASH_TAPER`` units, and exactly 0.0 at ``HERMIT_STASH_CAP``.

    The colony has had this since ``_stock_state`` - its ``glut`` arm is the
    same idea against a per-head target. The hermit had nothing: his cap was
    enforced only at the till, in ``hermit_stash_add``, which quietly returns 0
    and drops the goods. So every score that sends him to make something was
    flat right up to the cap and past it, and the work he did there was
    destroyed on arrival rather than stored.

    Multiplied into a score rather than tested as a gate, because
    ``choose_action`` SKIPS any candidate at or below 0.0 without drawing its
    tiebreak - so a hard gate changes the number of draws the moment it trips
    and re-phases the shared world stream for every colonist. A taper still
    reaches 0 at the cap and still re-phases there, which is unavoidable and is
    the point, but it does not do it over the whole approach.

    Never raises; an unreadable stash reads as empty, which keeps him working.
    """
    try:
        have = max(0, int(hermit_stash_qty(world, res)))
    except Exception:
        return 1.0
    room = max(0, int(HERMIT_STASH_CAP) - have)
    if room <= 0:
        return 0.0
    span = max(1.0, float(HERMIT_STASH_TAPER))
    return _clamp01(room / span)


def _hermit_bias(s: dict[str, float], agent: Any, world: Any, night: bool,
                 fatigue: float, warmth: float, carry_qty: int, carrying: Any,
                 danger: bool) -> None:
    """Turn a scored roster of colony jobs into a scored roster for a recluse.

    WEIGHTS AND TARGETS, NOT NEW ACTIONS, and that is the constraint this
    function exists to respect. `choose_action` draws one `rng.uniform` tiebreak
    per SCORED candidate, so adding a "GoHome" action would re-phase the world
    stream for every colonist on every tick and move unrelated outcomes by
    ~16%. Everything here either zeroes a key that already existed or re-weights
    one, so a colony with no hermit in it draws exactly what it drew before.

    Five groups, in order of how much argument they need:

    1. THE THINGS A HERMIT DOES NOT DO. Company (Converse dies with the 0.04
       social affinity, but Celebrate is `0.82 * morale + 0.18` and is not
       affinity-scaled at all, so it has to be said out loud), and the colony's
       work: raising, upgrading, repairing, cooking, sweeping, watching, and
       carrying anything to the store. Mourn is NOT in this list. He comes in
       for a funeral, and that is the one thing the whole colony does that he
       is still part of. Neither is anything combat_actions scores: he runs from
       a wolf, fights one that corners him, and walks in to the workbench to arm
       himself, because a hermit who will not defend himself is just a slow
       death (measured at 137 s of workbench time in 75 min, which is a visit,
       not a commute).

       THIS LIST IS NOT THE GATE ON GOODS, and reading it as one is what let
       him rob the colony for a fortnight. Every key here is a SCORE, so it only
       ever stops him CHOOSING the job - and a man promoted to hermit MID-ACTION
       is already inside one. `_h_craft` in particular is deliberately NOT
       zeroed (the workbench visit above), and it was the largest leak of the
       five: it paid for his spear and his armour out of the village stores on
       all five measured seeds. The gate on goods lives at the till instead,
       in `actions.stock_take`/`stock_add`, which every withdrawal passes
       through and which bills whichever agent's handler is running. See
       `actions._TILL_ACTOR`. Nothing here needs to know about it; that is the
       point of putting it there.

    2. THE TWO GATHERS THAT ARE NOT HIS. This is the correction that mattered
       most, and it was measured, not guessed. `gather` is ONE affinity feeding
       THREE jobs, and only one of them - ForageBerries - is food he can eat.
       At 0.85 the first cut of this role sent him to the trees and the loose
       rock as well, and `_deposit_step` then handed him his load back: over 75
       sim-min he spent 13-19k ticks felling the colony's forest and cracking
       its rock and DESTROYED every unit of it. That is not a hermit being
       antisocial, it is a hermit strip-mining the map, and it cost seeds 7 and
       11 their whole colony (peak pop 13 and 10 against a control of 20).
       Mine goes with them: `MINE_KEEP_OUT` sites quarries away from the
       settlement, so a digging hermit is a hermit anywhere but home.

    3. HIS OWN SHELTER, WHICH IS SLEEP AND WARMTH AT ONCE. The base Sleep score
       gates on `_free_hut` - a building in town - so it is recomputed here
       without one, and `WarmAtFire` is thrown away and recomputed against HIS
       fire, because the colony's firepit is the other thing that was dragging
       him back in (8-9k ticks of WarmAtFire on three of five seeds, and he was
       still the coldest man in the colony at the end of it). The cold also
       sends him to BED: `warmth` gets its own arm on the Sleep score and
       `_h_sleep_rough` pays warmth back at a hut's rate. That is a real grant -
       he gets shelter he did not build - and it is deliberate. The alternative
       measured worse on every axis at once, and a hermit who walks into town
       every night to stand at somebody else's fire is not a hermit at all.

       HERMIT_FIRE_DAMP IS GONE AND THIS PARAGRAPH USED TO DESCRIBE IT AS
       LIVE - "damped to HERMIT_FIRE_DAMP", "1.0 * 0.35 still beats an idle
       colony's chores", a safety valve that let a blizzard put him at the
       village hearth. There is no such constant and no such valve: the gate on
       the two fires is absolute in both directions (`actions.fire_for`), and
       what carries the blizzard case now is the cold-scene arm on his OWN
       fire's score, thirty paces away instead of seven hundred. See the note
       where the constant used to live, in constants.py.

    4. THE PULL HOME. `Wander` is normally a flat 0.10 - the floor that means
       "nothing better to do". For a hermit standing outside his own roam radius
       it is HERMIT_HOMESICK instead, which outranks the colony's small chores
       and loses to Eat, Sleep and anything combat scored. That ordering is the
       design: he drifts back out to his camp of his own accord after hunger or
       a wolf has dragged him in, but the pull home never outranks the reason he
       came - and while there is danger on the map it is not applied at all, so
       "get home" can never compete with "get away".

    5. A FULL LARDER STOPS THE FORAGING. `_deposit_step` hands a hermit his food
       back instead of shelving it, so his hands stay full - and a full-handed
       forager re-enters the deliver phase the instant he picks anything up,
       which is a job that starts and finishes every tick. Zeroing the two food
       jobs at CARRY_CAP costs nothing (he is holding a meal, by definition) and
       removes the thrash.
    """
    for dead in ("Converse", "Celebrate", "FollowParent", "Lookout", "ClimbTo",
                 "CleanLitter", "CookFood", "HaulToStockpile",
                 "BuildStructure", "UpgradeStructure", "RepairStructure",
                 "PlantSapling", "GatherWood", "GatherStone", "Mine"):
        s[dead] = 0.0

    # HIS FIRE OR NO FIRE. The score above this line was computed against
    # `nearest_structure(.., "firepit")`, which is the colony's hearth, so it is
    # thrown away and recomputed against his own - and it is recomputed at FULL
    # strength, where the shipped version scaled it by HERMIT_FIRE_DAMP (0.35).
    # The damp is gone and the constant with it: it was a soft gate that existed
    # only because the village fire was the one fire on the map, and a soft gate
    # is what the request was against. See the note where the constant used to
    # live. No fire of his own standing means no WarmAtFire at all, and the cold
    # sends him to bed instead, exactly as it did before he had one.
    own_fire = hermit_fire(world, built_only=True)
    if own_fire is None:
        s["WarmAtFire"] = 0.0
    else:
        # The same shape as the colony's score at the top of `_score_wants`,
        # including the cold-scene arm - THAT arm is what the deleted damp's
        # safety valve turns into. A blizzard chills at 0.030/s against the
        # 0.008/s his shelter pays back, so weather that could kill him has to
        # be able to drive his warmth score to 1.0; the difference now is that
        # it drives him to HIS fire, thirty paces away, instead of to the
        # colony's, seven hundred.
        boost = 1.65 if (night or _scene(world) in _COLD_SCENES) else 0.90
        wood_ok = bool(getattr(own_fire, "fire_active", False)) or (
            carry_qty > 0 and carrying == RES_WOOD)
        s["WarmAtFire"] = _clamp01(warmth * warmth * boost
                                   * (1.0 if wood_ok else 0.25))
        if warmth > 0.9 and wood_ok:
            s["WarmAtFire"] = 1.0
    s["Sleep"] = _clamp01(fatigue * fatigue * (1.55 if night else 0.45)
                          + warmth * warmth * (1.35 if night else 0.60))
    if fatigue > 0.93 or warmth > 0.90:
        s["Sleep"] = 1.0

    # HIS STASH IS FOOD HE CAN GET AT, so it scores like food in his hands. The
    # colony arm of `_score_wants` already raises Eat on `food_qty > 0`, i.e. on
    # the COLONY'S store, which a hermit reaches only by the long walk `_h_eat`
    # keeps as a last resort. Without this line a hermit with a full larder
    # banked ten paces away and nothing in his hands would score Eat off a store
    # he does not want to use - or, on a colony whose own store is empty, score
    # it at 0.12 and go hungry beside his own pile. `max`, never an assignment:
    # this can only ever raise the score, so it cannot weaken the guard above it.
    if hermit_stash_food(world)[0] is not None:
        try:
            hunger = float(getattr(agent, "hunger", 0.0) or 0.0)
        except (TypeError, ValueError):
            hunger = 0.0
        s["Eat"] = max(float(s.get("Eat", 0.0)), _clamp01(hunger * hunger))
        if hunger > 0.85:
            s["Eat"] = 1.0

    if carry_qty >= CARRY_CAP and carrying in (RES_FOOD, RES_COOKED):
        s["ForageBerries"] = 0.0
        s["Farm"] = 0.0
    else:
        # Full HANDS were the only thing that ever stopped him gathering; a full
        # PILE did not, so he foraged into a stash that refused the berries and
        # binned them. Same bug as the cook above and found by the same probe -
        # both scores were flat at every stash level up to and past the cap.
        # He can still Eat: that reads the stash, and a full stash is the one
        # state in which eating is certain to work.
        room = _stash_appetite(world, RES_FOOD)
        s["ForageBerries"] = float(s.get("ForageBerries", 0.0)) * room
        s["Farm"] = float(s.get("Farm", 0.0)) * room

    if danger:
        # This function runs AFTER the danger clamp - it has to, or the combat
        # merge would put the workbench back on his list - so it has to reapply
        # it to the two scores it just wrote, or a cold hermit lies down to
        # sleep in a flood. Same ceiling, same reason: nothing outranks fleeing.
        s["Sleep"] = min(s["Sleep"], 0.35)
        s["WarmAtFire"] = min(s["WarmAtFire"], 0.35)
        return
    try:
        away = abs(float(getattr(agent, "x", 0.0)) - hermit_home(world))
    except (TypeError, ValueError):
        away = 0.0
    if away > HERMIT_ROAM:
        s["Wander"] = HERMIT_HOMESICK

    # 6. HIS OWN HOUSE, which is the only building in the colony he is allowed
    #    to touch. Both keys below were zeroed at the top of this function and
    #    are being handed back CONDITIONALLY - while, and only while, there is an
    #    unfinished hut of his own standing at his camp. A hermit with a
    #    finished hut draws exactly what he drew before this existed.
    #
    #    The loop is: nothing in his hands -> GatherWood (his own trees, near
    #    his own camp, because `_h_gather` takes the nearest one to HIM);
    #    `actions._deposit_step` then declines to destroy the load for once,
    #    because his hut wants it; wood in his hands -> BuildStructure, which
    #    `_mk_build` points at the hut and `_h_build` walks to and delivers.
    #    Repeat until the frame is up. `_h_build`'s "fetch" phase - the one that
    #    would send him to the colony's stockpile - fails on sight of a hermit,
    #    which is what drops him back here to cut his own.
    #
    #    Both sit above HERMIT_HOMESICK (0.45), so building outranks drifting
    #    back to his camp, and well below the Eat/Sleep/flee ceilings, so a
    #    hermit never builds through a blizzard or a wolf. Nothing here is
    #    reachable while `danger` is set: that branch returned above.
    # 6a. AND HIS COOKING, which is the other half of feeding himself. CookFood
    #     is zeroed at the top of this function because the colony's version of
    #     it is a walk to the stockpile for somebody else's berries. His is not:
    #     `actions._h_cook` gives a hermit a fetch-free path that cooks what is
    #     already in his hands at his own fire. Handed back only with all three
    #     premises true - a lit fire, raw food in hand - so he never picks a job
    #     that fails on its first update. Scored below Eat, so a starving hermit
    #     eats the berries raw rather than standing over a fire with them.
    #     RAW FOOD IN HIS HANDS *OR* IN HIS PILE. `_h_cook`'s hermit arm draws
    #     from the stash when his hands are empty of raw, and the two have to
    #     agree or he picks a job that fails on its first update - the standing
    #     rule for every score in this function. Without the stash arm here the
    #     stash arm THERE is unreachable, which is how a feature ships inert: he
    #     banks berries, never scores a cook against them, and `ck` stays 0 on
    #     the panel forever.
    #     AND ONLY WHILE THE PILE HAS ROOM FOR THE MEAL. `hermit_stash_add`
    #     refuses a resource at HERMIT_STASH_CAP and returns 0, so a cook at the
    #     cap does not merely waste his afternoon - `_h_cook` takes the raw food
    #     and the till then throws the meal away. Measured before this line
    #     existed: twelve food units destroyed in four sim-minutes, and the
    #     score sat at a flat 0.42 whether he held 0 cooked or 65. See
    #     `_stash_appetite`.
    if own_fire is not None and bool(getattr(own_fire, "fire_active", False)):
        if ((carry_qty > 0 and carrying == RES_FOOD)
                or hermit_stash_qty(world, RES_FOOD) > 0):
            s["CookFood"] = HERMIT_COOK_URGE * _stash_appetite(world, RES_COOKED)

    # 6b. TENDING IT. A fire nobody feeds is out inside a quarter of an hour,
    #     and a hermit whose fire is always out is a hermit with a cold ring of
    #     stones for decoration - which is how this feature would have shipped
    #     inert for the tenth time. So when his own fire is low he goes and puts
    #     wood on it (WarmAtFire is the action that stokes, and `_stoke_wood`
    #     takes it from his hands and never from the store), and when he has no
    #     wood in hand to put on it he goes and cuts some. Both sit under the
    #     build urges, so raising the thing beats feeding it, and both sit above
    #     HERMIT_HOMESICK, so feeding it beats drifting home.
    if own_fire is not None and getattr(own_fire, "built", False):
        try:
            fuel = float((getattr(own_fire, "state", None) or {}).get("fuel", 0.0))
        except (TypeError, ValueError):
            fuel = 0.0
        if fuel < HERMIT_STOKE_BELOW:
            # Wood in his hands OR wood in his stash both mean "go and feed it"
            # rather than "go and cut some", and that ordering is the whole point
            # of the pile: `_stoke_wood` takes from the hands first and the stash
            # second, so a banked armful is a stoke with no walk in it, and a
            # stoke with no walk is a stoke that never leaves HERMIT_FIRE_WARD.
            if ((carry_qty > 0 and carrying == RES_WOOD)
                    or hermit_stash_qty(world, RES_WOOD) > 0):
                s["WarmAtFire"] = max(s.get("WarmAtFire", 0.0), HERMIT_STOKE_URGE)
            elif _tree_in_reach(world, agent):
                s["GatherWood"] = max(s.get("GatherWood", 0.0), HERMIT_STOKE_URGE)

    # 6c. AND THE WOODPILE, which is the only job here that is not a reaction to
    #     something. Everything above waits for a need - a low fire, an unbuilt
    #     frame, an empty stomach - and at 530..1060 px waiting is what put the
    #     fire out: every reactive trip to timber is twice the walk it used to
    #     be and all of it is outside HERMIT_FIRE_WARD. Measured on the first
    #     pass at the new band, before this existed: his banked wood peaked at 5
    #     units over eight seeds and the fire was alight 74.8% of the time it
    #     stood, against 91.4% at half the distance.
    #
    #     So he cuts AHEAD of the need and banks it, and `_stoke_wood` then
    #     feeds the fire without him leaving the ward at all. Gated on a tree
    #     inside HERMIT_FELL_REACH of the CAMP - the same question `_h_gather`
    #     asks when it picks the tree, so the score and the action cannot
    #     disagree - and on the pile being under HERMIT_WOODPILE, so it is a
    #     job that finishes rather than a treadmill. `max`, so it can never
    #     lower a stoke or a build that already wants the same action.
    if (hermit_stash_qty(world, RES_WOOD) < HERMIT_WOODPILE
            and _tree_in_reach(world, agent)):
        s["GatherWood"] = max(s.get("GatherWood", 0.0), HERMIT_WOODPILE_URGE)

    site = hermit_worksite(world)
    if site is not None:
        try:
            need_now = site.missing_for_stage()
        except Exception:
            need_now = {}
        holding = carry_qty > 0 and carrying in need_now
        # ...or it is in the pile by his door, which `_h_build`'s hermit fetch
        # now draws on without moving. Scoring it here is what keeps the score
        # and the action agreeing: ask a different question from the one the
        # handler asks and he picks a job that fails on its first update, which
        # is the deadlock this whole loop was rebuilt around once already.
        banked = any(hermit_stash_qty(world, r) > 0 for r in need_now)
        if holding or banked or not need_now:
            # Either he is carrying what the frame is short of, or the frame is
            # fully stocked and only wants work. Both are `_h_build`'s job.
            # The fire is worth marginally more than the hut (see
            # HERMIT_FIRE_URGE) so that with both unbuilt he lays the fire
            # first - it is a third of the price and it is his only warmth.
            s["BuildStructure"] = (HERMIT_FIRE_URGE
                                   if str(getattr(site, "kind", "")) == "hermit_fire"
                                   else HERMIT_BUILD_URGE)
        else:
            # Go and cut it - but only if there is something to cut WITHIN REACH
            # OF HIS CAMP. Note there is no test on what his hands are already
            # full of, and that is deliberate: his steady state is a full larder
            # of berries, so a hands-free requirement is a hut that never gets
            # past its first stake. `actions._deposit_step` delivers the log out
            # of the gather action's own stash for exactly that reason.
            #
            # The reach test is the one that keeps him a hermit. It asks the same
            # question `_h_gather` will ask when it picks the target - nearest
            # tree to HERMIT_HOME, not to him, inside HERMIT_FELL_REACH - so the
            # score and the action can never disagree and he never picks a job
            # that fails on its first update. On a camp with no wood near it he
            # simply never gets a hut, which is the honest outcome.
            if _tree_in_reach(world, agent):
                s["GatherWood"] = HERMIT_FELL_URGE

    # 7. AND THE ONE CONVERSATION HE HAS. Converse is dead at the top of this
    #    function - the 0.04 social affinity is the role - and it comes back for
    #    exactly one case: somebody has walked the whole standoff to see him and
    #    is standing at his door. Scoring it rather than forcing it keeps the
    #    meeting mutual and keeps it losing to Eat, Sleep and anything combat
    #    scored, so a visit never gets a man killed being polite.
    #
    #    Two conditions, and the second one was paid for. It is not enough that
    #    a visit is OPEN and the guest is nearby: the guest has to actually be
    #    stood there talking. Measured on seed 7 - a guest reached the hut,
    #    paid the visit out, and was then pulled away by a wolf before his
    #    Converse could close the slot. The slot stayed open for the full 260 s
    #    watchdog, and for all 260 s the hermit re-picked a Converse against a
    #    man who was no longer there, failed it on the 30 s approach timeout, and
    #    re-picked it again: eight and a half minutes of a hermit standing
    #    perfectly still in a walk pose. Asking what the guest is DOING rather
    #    than only where he is costs one attribute read and cannot do that.
    #
    #    THE HERMIT DOES NOT WALK TO MEET ANYBODY. The proximity gate is ~2x
    #    TALK_REACH rather than the ordinary 240 px, so this only ever fires with
    #    the guest already at his elbow and `_h_converse` has at most a couple of
    #    paces to close. That is characterisation and it is also the bug fix: at
    #    240 px he would spend the visit walking, and a walk that terrain will
    #    not let him finish is a 30 s approach that fails and is re-picked for as
    #    long as the guest stands there - measured at two solid minutes of a
    #    hermit frozen in a walk pose. He waits. Guests come to him.
    guest = hermit_guest(world)
    if guest is not None and _is_visiting(guest):
        try:
            near = abs(float(getattr(guest, "x", 0.0))
                       - float(getattr(agent, "x", 0.0))) <= TALK_REACH * 2.0
        except (TypeError, ValueError):
            near = False
        if near:
            s["Converse"] = HERMIT_HOST_URGE


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


def _stock_state(world: Any, pop: int) -> dict[str, tuple[float, float]]:
    """Per resource, ``(short, glut)`` against a standing per-head target.

    ``short`` is 0..1 how far the store is BELOW what a colony of this size
    wants; ``glut`` is 0..1 how far it is PAST it, reaching 1 at
    ``STOCK_GLUT_SPAN`` extra target-multiples. They are mutually exclusive by
    construction and both are 0 exactly at the target, so there is no step
    anywhere on the curve - which is what keeps a colony hovering at its target
    off a knife-edge.

    One idiom for every resource. Before this there were five: food got a target
    and no damper, stone got a target inline in Mine, wood and fibre got only
    whatever the build queue was short of, and cooked got a hand-written
    ``== 0`` special case. RES_FOOD counts raw and cooked together because both
    feed people and the birth gate sums them the same way.
    """
    out: dict[str, tuple[float, float]] = {}
    for res, per_head in STOCK_PER_HEAD.items():
        target = max(float(STOCK_FLOOR.get(res, 1.0)), float(pop) * per_head)
        if res == RES_FOOD:
            have = float(stock_qty(world, RES_FOOD) + stock_qty(world, RES_COOKED))
        else:
            have = float(stock_qty(world, res))
        ratio = have / max(1.0, target)
        out[res] = (_clamp01(1.0 - ratio),
                    _clamp01((ratio - 1.0) / max(1e-6, STOCK_GLUT_SPAN)))
    return out


def _surplus(world: Any, food_glut: float, night: bool) -> float:
    """The food glut, slewed and dimmed after dark. 0..1.

    Rise slow, fall fast. Two minutes of plenty before the colony believes it is
    rich; twelve seconds of famine to put everyone back on the crops. The
    asymmetry is the safety story, and it is why this is a slew rather than the
    hut tier's latch - a famine does not un-teach masonry, but it absolutely
    must reopen the fields.

    The night factor scales the slew's TARGET, not its output, so dusk ramps
    down over SURPLUS_FALL_SEC and dawn ramps back up over SURPLUS_RISE_SEC and
    neither edge is a step.

    Deliberately NOT persisted on the world, and that is the cheap half of the
    design: nothing to add to ``to_dict``, nothing for ``from_dict`` to fail on,
    SAVE_VERSION untouched. A loaded world starts at 0.0 and re-earns the
    surplus in at most SURPLUS_RISE_SEC, which is the same thing a save written
    before this feature existed would do.

    Advanced at most once per SURPLUS_STEP_SEC, on whichever agent is scored
    first in that window; every other agent that window reads the value it
    stored. That is what makes the trajectory independent of how many people are
    alive and identical in two separate processes on the same seed.
    """
    raw = _clamp01(food_glut) * (SURPLUS_NIGHT if night else 1.0)
    now = world_now(world)
    prev = getattr(world, "_bhv_surplus", None)
    then = getattr(world, "_bhv_surplus_t", None)
    if not isinstance(prev, float) or not isinstance(then, float):
        prev, then = 0.0, now - SURPLUS_STEP_SEC
    dt = now - then
    if 0.0 <= dt < SURPLUS_STEP_SEC:
        return prev
    if dt < 0.0 or dt > 60.0:
        # Rewound, or a world that has been sitting - do not integrate a jump.
        dt = SURPLUS_STEP_SEC
    step = dt / (SURPLUS_RISE_SEC if raw >= prev else SURPLUS_FALL_SEC)
    val = _clamp01(prev + max(-step, min(step, raw - prev)))
    try:
        setattr(world, "_bhv_surplus", val)
        setattr(world, "_bhv_surplus_t", now)
    except Exception:
        pass
    return val


def _thrift(world: Any) -> float:
    """How much of its surplus the colony is willing to SPEND, 0..1.

    Ramps with ``World.colony_morale()`` - the same mean the birth gate
    (MORALE_TO_GROW) and the stone-hut tier (HUT_TIER_MORALE) read, so "how is
    the colony doing" cannot come to mean three slightly different numbers.

    This is negative feedback and it is load-bearing. The extra work a surplus
    buys costs fatigue; fatigue drops colony morale; morale under MORALE_TO_GROW
    shuts the BIRTH gate. Measured in design without this term: morale blocked
    27.8% -> 59.7% of sampled birth ticks and one colony went 11 people to 3,
    with nothing in the chronicle saying why. A tired colony rests instead.

    The floor is what stops it being an off switch rather than a governor:
    pooled mean morale runs about 0.50, so a hard ramp from 0.45 would leave a
    typical colony spending under a third of its dividend and the labour freed
    from the fields would land on Wander instead of the quarry.

    Cached for two seconds the way ``_farm_feasible`` is, and for the same
    reason: it is O(roster) and score_actions runs per agent per AI tick.
    """
    now = world_now(world)
    try:
        t = float(getattr(world, "_bhv_thrift_t", -1e9))
        if 0.0 <= now - t < 2.0:
            return float(getattr(world, "_bhv_thrift", SURPLUS_THRIFT_FLOOR))
    except (TypeError, ValueError):
        pass
    fn = getattr(world, "colony_morale", None)
    try:
        m = float(fn()) if callable(fn) else 0.0
    except Exception:
        m = 0.0
    span = max(1e-6, SURPLUS_MORALE_FULL - SURPLUS_MORALE_MIN)
    val = SURPLUS_THRIFT_FLOOR + (1.0 - SURPLUS_THRIFT_FLOOR) * _clamp01(
        (m - SURPLUS_MORALE_MIN) / span)
    try:
        setattr(world, "_bhv_thrift", val)
        setattr(world, "_bhv_thrift_t", now)
    except Exception:
        pass
    return val


def _appetite(res: str, stocks: dict[str, tuple[float, float]],
              dividend: float, queue_urg: float = 0.0) -> float:
    """0..1 appetite for producing `res`, from three independent demands.

    ``max()``, never a sum - it matches the ``max(stone_low, stone_urg)`` this
    generalises and it is what bounds the result:

    * ``queue_urg`` - what the head of the build queue cannot afford. Correct
      code, and untouched; it is just silent 93-96% of the time.
    * ``MATERIAL_BASELINE * short`` - the standing per-head target. This is
      Mine's old ``0.70 * stone_low`` generalised to wood and fibre, and it is
      the arm that stops the wood store hitting literal zero.
    * ``dividend`` - what the food surplus pays for, tapered by how short the
      resource is so a topped-up bin does not hold the same appetite as an empty
      one. That taper is negative feedback between the gathers: as wood fills,
      stone overtakes it, so the three separate instead of landing on one number
      inside TIEBREAK of each other.

    Capped at APPETITE_MAX = ``_gather_urgency``'s own ceiling, so no caller's
    reachable maximum moves and nothing can newly cross OVERRIDE_FLOOR.
    """
    short, glut = stocks.get(res, (0.0, 0.0))
    paid = dividend * (1.0 - glut) * (SURPLUS_RESIDUAL
                                      + (1.0 - SURPLUS_RESIDUAL) * short)
    return min(APPETITE_MAX, max(queue_urg, MATERIAL_BASELINE * short, paid))


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
    # A hermit never books a bed - `_h_sleep` sends him to `_h_sleep_rough`, and
    # a target here would only make `_c_sleep` try to evict him from a hut he
    # was never in.
    if is_hermit(agent):
        return make_action("Sleep")
    hut = _free_hut(world, agent)
    return make_action("Sleep", target=hut.id) if hut is not None else None


def _mk_warm(agent: Any, world: Any) -> Action | None:
    # `fire_for` is the mutual gate: the hermit's own fire for the hermit, the
    # nearest colony pit for everybody else, and never the other way round.
    fire = fire_for(agent, world)
    return make_action("WarmAtFire", target=fire.id) if fire is not None else None


def _mk_cook(agent: Any, world: Any) -> Action | None:
    fire = fire_for(agent, world)
    return make_action("CookFood", target=fire.id) if fire is not None else None


def _mk_build(agent: Any, world: Any) -> Action | None:
    if is_hermit(agent):
        # HIS HUT OR NOTHING. The colony's queue is not his and the "nearest
        # incomplete anything" fallback below is worse still - it would send him
        # to whatever the colony happened to be raising, which is 700 px away in
        # the middle of town. Returning None when his own frame is finished (or
        # was never staked) is the `_mk_clean`/`_mk_upgrade` pattern: no premise,
        # no action, and he drops through to the next-best thing this tick
        # instead of spending a decision cycle on one that fails immediately.
        # His fire first, then his walls - see `actions.hermit_worksite`.
        site = hermit_worksite(world)
        if site is None:
            return None
        return make_action("BuildStructure", target=int(site.id))
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
    dmg = colony_repairable(reg, threshold=0.9)
    if not dmg:
        return None
    ax = float(getattr(agent, "x", 0.0))
    worst = min(dmg, key=lambda s: (s.hp / max(1.0, s.max_hp), abs(s.x - ax)))
    return make_action("RepairStructure", target=worst.id)


def _mk_converse(agent: Any, world: Any) -> Action | None:
    # The appointed visitor gets the hermit as his partner regardless of the
    # 240 px proximity rule below, which is the whole difference between a chat
    # and a visit: `_talk_partner` would never name a man 700 px away. The two
    # data keys are what `_h_converse` reads to give the walk a real budget and
    # the talk a longer stay - see its docstring for why this is that action
    # with a longer leash rather than an action of its own.
    if hermit_guest(world) is agent and agent is not None:
        host = living_hermit(world)
        if host is not None:
            return make_action("Converse", target=int(getattr(host, "id", 0) or 0),
                               visit=1)
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

    The animal clause and its A/B
    -----------------------------
    ``combat_actions.under_attack`` was written for this function and then never
    called from it, which left ``score_combat``'s whole above-the-floor design
    unreachable for anybody mid-job. Wiring it in re-decides an action already
    in flight, so it re-phases the world RNG and cannot be judged on a death
    *total*. 32 same-seed pairs, 30 sim-min each, deaths counted through the
    ``entities.on_death`` hook and cross-checked against ``reconcile["deaths"]``
    (0 disagreements in 64 runs), normalised per 1000 colonist-minutes lived:

    ====================  ==========  ==========  =======  =========
    cause                 off /1000   on /1000    seeds    sign p
                                                  -/0/+
    ====================  ==========  ==========  =======  =========
    mauled                    15.60       10.16   17/10/4      0.007
    fall                       6.41        4.98    6/19/6      1.000
    lightning                 11.33        8.92    6/24/1      0.125
    mudslide                   0.86        3.32    2/25/4      0.688
    all causes                42.10       36.92   17/ 1/13     0.585
    ====================  ==========  ==========  =======  =========

    Read it this way. Mauling is the only cause that moves *systematically* -
    it falls a third, on 17 seeds against 4, and the per-seed drops are small
    and everywhere (3->1, 5->1, 4->2) rather than one seed carrying the column.
    Every other row is one or two seeds: the lightning row is the useful
    control, because nothing here can touch lightning and it still came out
    6-down/1-up at p=0.125, which is the drift floor this design has. Mauling
    sits well outside it; nothing else does.

    The row that had to be checked is ``fall``. The prior change of this class
    took fall deaths from 0 to 12 by sending rescuers across cliff faces, and a
    mauling traded for a fall is not a win. It did not happen: falls went 30 ->
    24 on a 6/6 seed split, i.e. unmoved.

    What this does NOT claim: that fewer people die. All-causes is 197 -> 178 at
    p=0.585 and survivors 270 -> 272. The honest statement is that the colony
    answers animals instead of watching them, at no measured cost in any other
    cause. Proving a total-mortality change would need far more than 32 seeds.

    (Figures above drop seed 20, which flooded in the ``on`` arm and alone
    turned 9 drownings into 40; with it kept, all-causes is 211 -> 215 and every
    conclusion above is unchanged.)
    """
    try:
        cur = getattr(agent, "action", None)
        kind = getattr(cur, "kind", "") if cur is not None else ""

        # Already responding to the emergency.
        if kind in ("FleeFrom", "Panic", "Eat", "Sleep"):
            return False

        # An animal close enough to matter. This is the only override that is
        # not about the agent's own body, and it is the one the whole combat
        # module was written expecting: score_combat puts FightAnimal and
        # FleeAnimal above OVERRIDE_FLOOR precisely so they can beat the
        # hysteresis bonus, but nothing above ever re-scored a live action for
        # an animal, so those numbers were unreachable for anyone mid-job.
        # Measured: of 424 agent-seconds where an armed colonist scored
        # FightAnimal above the floor, 77.6% were locked inside an unfinished
        # non-combat action and 0.9% were free to switch.
        if UNDER_ATTACK_OVERRIDE and kind not in _ANIMAL_BUSY_KINDS:
            try:
                from .combat_actions import under_attack
                if under_attack(agent, world):
                    return True
            except Exception:
                log.debug("under_attack override failed", exc_info=True)

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

    # The hermit's own two lines of colony business, and they are deliberately
    # AFTER the colony's own staking: a hut that failed to go up must never be
    # able to cost the settlement its build slot for the pass. Both swallow
    # everything - a hermit is a garnish, and a garnish does not get to take the
    # director down with it.
    _ensure_hermit_hut(world, reg)
    _ensure_hermit_fire(world, reg)
    _tick_hermit_visit(world)

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
        for s in _colony_sites(reg):
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


def _colony_sites(reg: StructureRegistry) -> list[Structure]:
    """Unfinished buildings THE COLONY is responsible for. Never raises.

    ``reg.incomplete()`` with the hermit's hut taken out, and both callers
    need it out for a different reason.

    `_stake_out_site` treats the list as the colony's open work: it refuses to
    stake anything new once ``len(pending) >= MAX_CONCURRENT_SITES``. A hut
    counted there is a build slot the colony has lost to a building it is not
    allowed to work on - and it sits open for as long as one man takes to carry
    twelve wood a hundred px at a time, which is minutes. That is a colony that
    stops building huts because a hermit is busy.

    `update_director` turns the same list into the build QUEUE and the
    ``build_needs`` totals, which are what `_mk_build` sends builders to and
    what the gather scores read as "what are we short of". A hut in there
    sends the nearest villager on a 700 px walk to build the hermit's hut for
    him, which is the opposite of the point of him.
    """
    try:
        return [s for s in reg.incomplete()
                if str(getattr(s, "kind", "")) not in ("hermit_hut", "hermit_fire")]
    except Exception:
        return []


def _stake_out_site(world: Any, reg: StructureRegistry) -> None:
    """Place the next wanted building as an unbuilt Structure, so builders have
    somewhere concrete to go."""
    pending = _colony_sites(reg)
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
#  The hermit's hut, and the visits to it
# ===========================================================================
#: Seconds a newly appointed visitor is given to actually pick the walk up
#: before the slot is treated as stale. He is scored on his own decision
#: cadence, not on the director's, and `choose_action`'s commitment window can
#: hold him on whatever he was already doing for MIN_COMMIT_SEC first.
VISIT_PICKUP_GRACE = 20.0
#: ...and how soon the colony tries again after one that came to nothing. Short,
#: because the appointment failing is not the same event as a visit having just
#: happened, and the full period is the spacing between VISITS.
VISIT_RETRY = 120.0
#: Margin the SLOT watchdog allows on top of the visitor's own approach budget,
#: covering the stay itself (`_h_converse` gives him CONVERSE_TIME * 3 from the
#: moment of contact) and a few seconds of slop. It exists so the watchdog is
#: strictly the outer bound: the ordinary exits are the action's own, and this
#: only catches an appointment with no exit at all - a guest eaten on the walk
#: out, a host who dies while he is still coming.
VISIT_STAY = 60.0


def living_hermit(world: Any) -> Any | None:
    """The colonist holding the title, or None. Never raises.

    Lowest id wins if a mangled save somehow carries two; `assign_roles` demotes
    the extras on its next pass, but this runs on the same tick and has to give
    the same answer twice in a row.
    """
    best = None
    try:
        for a in alive_agents(world):
            if not is_hermit(a):
                continue
            if best is None or int(getattr(a, "id", 0) or 0) < int(getattr(best, "id", 0) or 0):
                best = a
    except Exception:
        return None
    return best


#: Process-local memo for the peak scan, ``{(seed, epoch, base_q, side): mask}``.
#: The scan is a rolling max over 6400 columns; it is cheap (~1 ms) but it is
#: not free, and `_camp_site` can be asked several times a minute on a colony
#: that keeps losing its hermit. Keyed on ``terrain.epoch``, which is bumped by
#: every change to the walkable surface, so a mudslide or a finished ladder
#: invalidates it for free. Derived and never persisted - a stale process cache
#: cannot outlive the process that made it.
_SHELF_MEMO: dict[tuple[int, int], Any] = {}

#: PURE DIAGNOSTIC. `_camp_site` writes its reasoning here and NOTHING IN THE
#: SIM EVER READS IT - it exists so the measurement harness can ask "was the
#: highest point in the band rejected, and why" without the answer having to be
#: reconstructed from the outside and possibly differ from what actually ran.
#: Overwritten on every call, never persisted, never rendered.
_LAST_CAMP_PICK: dict[str, Any] = {}


def _shelf_mask(terrain: Any) -> "np.ndarray | None":
    """Columns whose whole HERMIT_SHELF_HALF shelf is walkable. Never raises.

    A CAMP IS TWO BUILDINGS, NOT A POINT, which is the whole reason this is a
    window rather than a slope read. `_ensure_hermit_fire` puts the firepit
    34 px off the door and the hut sprite is 46 px across, so a site whose
    single column happens to be flat while the ground falls away 20 px either
    side gets a hut on the level and a fire on a face - or a hermit who cannot
    walk from one to the other. The window is +/-HERMIT_SHELF_HALF and every
    column in it has to be walkable, not merely the mean of them.

    This matters far more now than it did at 520 px. The band is searched for
    its HIGHEST ground, and the highest ground is by definition a place the land
    falls away from in both directions: "peak" and "knife edge" are the same
    word until something tests the shelf.
    """
    try:
        sl = np.abs(terrain.column_slope())
        n = int(sl.size)
        half = int(round(float(HERMIT_SHELF_HALF)))
        if n <= 0 or half <= 0 or 2 * half + 1 > n:
            return None
        pad = np.pad(sl, half, mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
        return np.asarray(win.max(axis=1) <= float(MAX_SLOPE_WALK))
    except Exception:
        log.debug("shelf scan failed", exc_info=True)
        return None


def _tree_xs(world: Any) -> "np.ndarray":
    """Sorted x of every living tree. Deterministic; empty array if none."""
    xs = []
    try:
        for p in props_of(world):
            if not prop_alive(p):
                continue
            if str(getattr(p, "kind", "")).lower() not in _TREE_KINDS:
                continue
            try:
                xs.append(float(getattr(p, "x", 0.0)))
            except (TypeError, ValueError):
                continue
    except Exception:
        return np.empty(0, dtype=np.float64)
    return np.sort(np.asarray(xs, dtype=np.float64))


def _camp_site(world: Any) -> float:
    """Where to stake the hut: THE HIGHEST SAFE GROUND IN THE STANDOFF BAND.

    The band stopped being a seeded pick and became a SEARCH RANGE. The ask was
    "double that distance away ... at the highest point beyond 530px", so the
    distance is a corridor and which point inside it he lives on is the
    hillside's decision.

    THEN THE CORRIDOR STOPPED BEING PIXELS AND BECAME A FRACTION OF THE RUN.
    "Lets send him 3/4 of the way across the map away from the colony ... 3/4 or
    more": see `actions.hermit_band`, which is now the only place the two walls
    are computed. The band is ``0.75 * reach .. reach``, where reach is the room
    between the settlement and the world edge on the roomier shoulder, less
    HERMIT_EDGE_MARGIN. NOTHING IN THIS FUNCTION'S RULE CHANGED - it is the same
    search for the highest safe ground, over a corridor that is now 2200-5200 px
    out instead of 530-1060 and 700-1300 px wide instead of 350. Measured over
    60 fresh worlds: mean camp 3548 px out, 0.55 of WORLD_W, 0.86 of the
    available reach, never below the 0.75 floor, and still standing higher than
    82% of the columns in its own band - the same figure the narrow band gave.

    THE SEARCH RUNS TWICE OVER TWO CORRIDORS AT EACH LEVEL OF THE LADDER, and
    the reason is HERMIT_RESITE_SLACK. The odd passes take the STAKING corridor
    (the band with its inner wall inset by the slack), so a new hut starts with
    room to give and `_ensure_hermit_hut` can enforce the band strictly. The
    even passes take the whole band, because the inset is not free - it throws
    away candidate shelf, and measured on the old narrow band that alone moved
    six seeds of sixty off a clean walk home. A clean route is worth more than a
    wide margin against churn, so the search widens before it settles.

    THE SCAN IS MEMOISED ON (seed, terrain.epoch) AND THE WIDER BAND DID NOT
    COST ANYTHING WORTH NAMING. `_shelf_mask` was always computed over the whole
    heightmap rather than over the band, so widening the band changed only the
    numpy slices taken out of it: 1.01 ms cold (mask included) and 0.42 ms warm,
    over 60 worlds, on a function that runs once per hut staked.

    A PEAK IS SURROUNDED BY DESCENT, WHICH IS WHY MOST OF THIS FUNCTION IS A
    SAFETY TEST. "Highest point in the band" taken literally is an instruction
    to stake a camp on a knife edge, and this project has a documented history
    of steep ground killing colonies - a mining pass once cut a face of |slope|
    19.68 against a MAX_SLOPE_CLIMB of 2.6 and emptied seeds. So a candidate has
    to clear two tests before its height is even looked at:

      * SHELF. Every column within HERMIT_SHELF_HALF of it is walkable
        (|slope| <= MAX_SLOPE_WALK, 0.9). That is the hut's footprint plus the
        34 px the fire sits off the door, so both buildings land on the level
        and he can walk between them. See `_shelf_mask`.
      * ROUTE. There is no CLIFF column (|slope| > MAX_SLOPE_CLIMB, 2.6) between
        the candidate and the settlement it is measured from. A cliff on the way
        home is a slip roll every time he goes for timber, and `entities`
        charges those in fall damage. Counted exactly, with a cumulative sum
        over the cliff mask, rather than sampled - a one-column face cannot hide
        between two samples.

    ...and one preference, which is not a safety test but is the difference
    between a hermit and a man standing next to an empty frame for 45 minutes:

      * TIMBER. Everything in his camp is wood he cut himself within
        HERMIT_FELL_REACH of the camp, so a summit with no tree near it is a
        summit with no hut and no fire on it. Measured on the old code, seed 11:
        not one tree within reach for the entire run. So candidates that have
        timber are preferred to candidates that do not - and among them it is
        still the highest that wins, so this reorders the shortlist rather than
        replacing the rule.

    THE LADDER, in order, first match wins, roomier shoulder before the other
    one inside each rung:

        1. shelf + route + timber      the ordinary answer
        2. shelf + route               a bare summit he can reach
        3. shelf + timber              the settlement is behind a cliff
        4. shelf                       ...and there is no wood either
        5. `hermit_home()`             no shelf anywhere in the band

    Rungs 3 and 4 are the honest admission that a colony can be walled in: the
    alternative is refusing to give him a camp at all, and a hermit with no camp
    is the feature not shipping on that seed. They do NOT ignore the route, they
    minimise it - see `_pick` - and they compare the two shoulders rather than
    taking the roomier one, because with no clean walk anywhere the two sides
    stop being interchangeable.

    RUNGS 1 AND 2 ARE NOW DEAD, AND THE MEASUREMENT SAYING SO IS THE MOST
    IMPORTANT LINE IN THIS DOCSTRING. Over 60 fresh worlds the search lands on
    rung 3 SIXTY TIMES OUT OF SIXTY, and the cliff-crossing rate is 60/60
    against 17/60 at the old band. That is not a regression in siting; it is the
    route test running out of meaning. It asks whether a walk of 2242..5141 px
    across procedural terrain contains a single column of |slope| > 2.6, and on
    a 6400 px map the answer is always yes. At 530..1060 px it was a real
    question with a real answer 72% of the time.

    AND THE RANKING UNDERNEATH IT WENT INERT WITH IT, which is the second half
    of the same measurement and is reported here rather than quietly enjoyed.
    `_pick(gentlest=True)` is supposed to minimise the worst face on the walk
    home before it looks at height, and over 530 px it did. Over 3000 px it
    cannot: `worst[]` is a RUNNING MAX outward from the settlement, so once the
    band is far enough out that the biggest face on the map lies inside the
    approach, every candidate in the band carries the same value and the
    minimisation has nothing to choose between. Measured over 60 fresh worlds,
    the worst face on his chosen route is a median 11.64 against 12.00 for an
    arbitrary walk of the same length on the same map - a 3% improvement, i.e.
    none. Do not cite the route ranking as a safety feature at this range.

    WHAT DOES STILL HOLD IS THE SHELF, and it is the test that was carrying the
    outcome all along: every column within HERMIT_SHELF_HALF walkable, 60 of 60
    fresh worlds, and no hermit fall death on any of 8 seeds x 45 sim-min (his
    four deaths over that sweep were two starvations and two maulings). The
    exposure the cliff column represents is a look-and-feel one and a trap
    waiting for a terrain event, not a body count.

    BOTH TESTS ARE KEPT ANYWAY, and not out of sentiment. They cost two numpy
    comparisons, they are what makes rungs 1-2 fire on a small map or a smoothed
    landscape, and deleting them would delete the ordering that makes pass three
    prefer the corridor - the fallback rungs are the only rungs left, so the
    passes are what carry HERMIT_RESITE_SLACK now. What must NOT happen is
    someone reading "cliff-crossing 100%" as a bug and widening the search to
    fix it: there is nothing to find. One face at 300 px poisons every column
    beyond it, and no camp at 2200 px or more can route around a wall standing
    inside 2200.

    THE SAFETY TESTS MOVE HIM OFF THE OUTRIGHT SUMMIT ROUTINELY, and that is the
    number to watch rather than the average: the summit of a band is very often
    a place with no shelf on it. Zero fall deaths followed, which is what the
    shelf test is for.

    Deterministic: the scan is over the heightmap, the tie-break is a strict
    ``<`` on height, and `props_of` iterates in registry order. Two processes on
    one seed pick the same column. Falls back to `hermit_home` on anything
    unexpected - a hermit in the wrong place is a disappointment, a raised
    exception on a director pass is a disabled subsystem.
    """
    x = float(hermit_home(world))
    _LAST_CAMP_PICK.clear()
    _LAST_CAMP_PICK["fallback"] = x
    try:
        terrain = getattr(world, "terrain", None)
        surf = terrain.surface() if terrain is not None else None
        if surf is None:
            return x
        n = int(surf.size)
        key = (int(getattr(world, "seed", 0) or 0), int(getattr(terrain, "epoch", 0)))
        shelf = _SHELF_MEMO.get(key)
        if shelf is None or getattr(shelf, "size", 0) != n:
            shelf = _shelf_mask(terrain)
            if shelf is None:
                return x
            _SHELF_MEMO.clear()          # one terrain at a time; never grows
            _SHELF_MEMO[key] = shelf

        cs = np.abs(terrain.column_slope())
        base = float(_hermit_base(world))
        bi = int(min(max(round(base), 0), n - 1))
        # THE WORST FACE ON THE WALK HOME, for every column, in two passes.
        # A running max outward from the settlement IS the corridor maximum:
        # `worst[i]` is the steepest ground anywhere between column i and the
        # village. That gives the cliff test (`worst > MAX_SLOPE_CLIMB`) and the
        # fallback's ranking from one array, and it is exact rather than sampled
        # - a one-column face cannot hide between two probes.
        worst = np.empty(n, dtype=np.float64)
        worst[bi:] = np.maximum.accumulate(cs[bi:])
        worst[: bi + 1] = np.maximum.accumulate(cs[bi::-1])[::-1]

        trees = _tree_xs(world)
        edge_lo = float(HERMIT_EDGE_MARGIN)
        edge_hi = float(WORLD_W) - edge_lo

        # THE ROOMIER SHOULDER FIRST, then the other one. `hermit_home` now
        # picks its side by reach rather than by crc32 (the standoff is a
        # fraction of the room, so the side with the room is the side with the
        # distance), and reading `first` off it keeps the two agreeing about
        # which shoulder that is with one derivation rather than two.
        first = 1.0 if x >= base else -1.0

        def _shoulders(stake: bool):
            """Both shoulders of the standoff band, the roomier one first.

            Returns ``(sides, band_top)``, where each side carries its shelf
            columns, their worst face on the walk home, whether they have timber
            in reach, and that shoulder's outright summit.
            """
            sides: list[tuple[float, Any, Any, Any, int]] = []
            band_top: int | None = None
            # ONE BAND FOR BOTH SHOULDERS, taken from the roomier one - see
            # `actions.hermit_band`. Giving the cramped shoulder a band derived
            # from its own small reach would let the fallback rungs below stake
            # him 200 px from a colony seated near a wall. Off the roomier
            # shoulder's band the cramped side usually has no columns left in
            # range at all, which is the correct answer rather than a bad one.
            d_lo, d_hi = hermit_band(world, None, stake=bool(stake))
            for side in (first, -first):
                ends = (base + side * float(d_lo), base + side * float(d_hi))
                lo = max(min(ends), edge_lo)
                hi = min(max(ends), edge_hi)
                i0 = int(min(max(math.ceil(lo), 0), n - 1))
                i1 = int(min(max(math.floor(hi), 0), n - 1))
                if i1 < i0:
                    continue
                idx = np.arange(i0, i1 + 1, dtype=np.int64)
                # The band's OUTRIGHT summit, safe or not, recorded before
                # anything is filtered. It is the number the diagnostic compares
                # against, so "the highest point had to be rejected" means what
                # it says rather than "the highest point that already passed".
                top = int(idx[int(np.argmin(surf[idx]))])
                if band_top is None or surf[top] < surf[band_top]:
                    band_top = top
                ok = shelf[i0 : i1 + 1]
                if not bool(ok.any()):
                    continue
                keep = idx[ok]
                if trees.size:
                    j = np.searchsorted(trees, keep.astype(np.float64))
                    left = np.where(j > 0,
                                    trees[np.clip(j - 1, 0, trees.size - 1)],
                                    -1e18)
                    right = np.where(j < trees.size,
                                     trees[np.clip(j, 0, trees.size - 1)], 1e18)
                    near = np.minimum(np.abs(keep - left), np.abs(right - keep))
                    timber = near <= float(HERMIT_FELL_REACH)
                else:
                    timber = np.zeros(keep.size, dtype=bool)
                sides.append((side, keep, worst[keep], timber, top))
            return sides, band_top

        def _pick(cand: Any, w_worst: Any, gentlest: bool) -> int:
            """Highest column in *cand*; on the fallback rungs, gentlest first.

            The two upper rungs already know the route is cliff-free, so there
            height is the only question. The fallback rungs are the case the
            brief calls out - "if the highest point in the band is unreachable
            or lethal, fall back to the highest point that is not" - and
            ignoring the route there would hand him the summit at the far end of
            the worst face on the map. So they minimise the steepest ground on
            the walk home FIRST (rounded to 0.1, which is finer than any face
            this matters on) and take the highest of what is left.
            """
            if not gentlest:
                return int(cand[int(np.argmin(surf[cand]))])
            key = np.round(w_worst, 1)
            tie = cand[key <= key.min()]
            return int(tie[int(np.argmin(surf[tie]))])

        def _ladder(sides, band_top, rungs, widened: bool):
            """Walk *rungs* over *sides*; return the winning column or None."""
            for rung, need_route, need_timber, gentlest in rungs:
                # WHICH SHOULDER, and the two kinds of rung answer it
                # differently on purpose. On the route-clean rungs the walk home
                # is cliff-free either way, so the seeded side wins and the
                # choice stays cosmetic - `sides` is already in that order. On
                # the fallback rungs there IS no clean route and the sides are
                # not interchangeable: one shoulder may be behind a face of
                # |slope| 16 and the other behind nothing worse than 3, and
                # taking the seeded one regardless would pick the worse walk
                # half the time for no reason at all. So those rungs compare the
                # shoulders and take the gentler, seeded side breaking a tie.
                order = sides
                if gentlest and len(sides) > 1:
                    order = sorted(sides, key=lambda t: float(np.min(t[2])))
                for side, keep, w_worst, timber, top in order:
                    m = np.ones(keep.size, dtype=bool)
                    if need_route:
                        m &= (w_worst <= float(MAX_SLOPE_CLIMB))
                    if need_timber:
                        m &= timber
                    if not bool(m.any()):
                        continue
                    pick = _pick(keep[m], w_worst[m], gentlest)
                    _LAST_CAMP_PICK.update(
                        fallback=x, rung=rung, side=side, x=float(pick),
                        y=float(surf[pick]), route_worst=float(worst[pick]),
                        widened=bool(widened),
                        # The band's unconstrained summit on this shoulder, and
                        # the best on either, so the harness can say how far the
                        # safety tests moved him and how high that cost.
                        top_x=float(top), top_y=float(surf[top]),
                        band_top_x=float(band_top if band_top is not None
                                         else pick),
                    )
                    return float(pick)
            return None

        clean = ((1, True, True, False), (2, True, False, False))
        rough = ((3, False, True, True), (4, False, False, True))

        # PASS ONE: the staking corridor, and only the rungs with a clean walk
        # home. This is the ordinary answer and it keeps HERMIT_RESITE_SLACK px
        # of drift in hand at both walls of the band.
        inner, inner_top = _shoulders(True)
        got = _ladder(inner, inner_top, clean, False)
        if got is not None:
            return got

        # PASS TWO: THE FULL BAND, still route-clean. Insetting the corridor to
        # buy drift give also threw away 180 px of candidate shelf, and that is
        # not free: measured over 60 fresh worlds, cliff-crossing sites went
        # from 17 to 23 out of 60 when the corridor narrowed, six seeds moving
        # to rung 3 purely because the shelf they would have used sat in the
        # 90 px the inset removed. A clean walk home is worth more than a wide
        # margin against churn - the margin only saves a re-site, the route is
        # a slip roll every time he goes for timber - so when nothing inside the
        # corridor has one, the search widens to the whole of 530..1060 before
        # it settles for a face. The camp is still in band; it simply starts
        # nearer a wall, and may re-site sooner if the settlement drifts that
        # way. `widened` in the diagnostic is how often this fires.
        outer, outer_top = _shoulders(False)
        got = _ladder(outer, outer_top, clean, True)
        if got is not None:
            return got

        # PASS THREE: THE STAKING CORRIDOR AGAIN, ON THE FALLBACK RUNGS - and
        # this pass exists because the fractional band made passes one and two
        # dead letters. `worst[]` is a running max outward from the settlement,
        # so the route test asks "is there no |slope| > 2.6 column anywhere
        # between here and the village". Over the 530..1060 px the band used to
        # be, that was a real question and it passed 43 times in 60. Over the
        # 2242..5141 px it is now, procedural terrain has a cliff column in it
        # essentially always: measured 60 of 60 fresh worlds, rung 3 on every
        # one. With no pass three the search fell straight through to the full
        # band every single time, and HERMIT_RESITE_SLACK - which is bought by
        # staking INSIDE the corridor - was silently never applied to anything.
        # So the corridor gets its turn on these rungs too, before the widening
        # does. `widened` in the diagnostic now means what it says again.
        got = _ladder(inner, inner_top, rough, False)
        if got is not None:
            return got

        # PASS FOUR: nothing walkable inside the corridor at all. Take the whole
        # band, since a wider search can only find a gentler face than a
        # narrower one.
        got = _ladder(outer, outer_top, rough, True)
        if got is not None:
            return got
    except Exception:
        log.debug("camp siting fell back to the derived offset", exc_info=True)
    return x


def _ensure_hermit_hut(world: Any, reg: StructureRegistry) -> None:
    """Stake the hermit's hut at his camp when he has none. Never raises.

    ONE UNBUILT HUT AT A TIME AND NOBODY ELSE'S BUSINESS. This is the whole of
    the colony's involvement in his hut: it puts stakes in the ground where he
    lives, and `_colony_sites` then hides those stakes from the build queue, the
    build needs and the concurrency cap, so no villager is ever sent to work on
    it. He builds it himself out of wood he cuts himself - see
    `_hermit_bias`, `actions._deposit_step` and `actions._h_build`.

    It is staked rather than conjured finished on purpose. A building that
    simply appears is the one thing this project has learned to distrust; a
    hermit hauling twelve wood to a frame at the far edge of the frame is
    something to WATCH, it is the only construction in the game done by one man,
    and it is proof the role is running at all.

    A SAVE WRITTEN BEFORE THE STANDOFF BECAME A FRACTION LOADS, AND THEN HE
    MOVES, and that is stated here because it is the one user-visible
    consequence of the change that is not in a constant. Measured end to end:
    a seed-5 save taken with the camp at the old 620 px restores exactly - hut
    id, hut position, built flag, his private stash, `hermit_home` still
    returning the hut so the successor still inherits it, SAVE_VERSION 1,
    reconcile residual 0. On the FIRST director pass after the load the band
    test below reads 620 px against a new band of 2443..3257 and he strikes
    camp: the old hut collapses to rubble with ``ruin_cause = "moved"``, the
    chronicle says the village had grown away from him, `purge_ruins` sweeps it
    after RUIN_LINGER, and he re-stakes 3205 px out and rebuilds. The stash
    survives all of it - it is world state, not hut state - and he spends it on
    the new frame. Nothing is lost and nothing raises; he just emigrates within
    a quarter of a second of the save opening. That is correct rather than
    unfortunate: a hut 620 px from the settlement is, under the rule the user
    chose, a hut in town.

    ...UNLESS THE OLD HUT SITS ON THE CRAMPED SHOULDER, and that exception is
    real, measured, and deliberately NOT fixed. Re-probed on seed 3, whose
    settlement sits at 5377 with 683 px of run to the right and 5037 to the
    left, an old hut planted 530 px out on each side in turn:

        left  (roomier)  530 px vs band 3778..5037  out of band, strikes at
                                                    t=0.03 s, rebuilds 4090 px
        right (cramped)  530 px vs band  512.. 683  IN BAND - he stays put

    The right-hand row is not a bug in the band test; it is the fractional rule
    answering honestly. 530 px of a 683 px run IS 0.78 of the way to that edge,
    so on his own shoulder he already satisfies "three quarters or more". It
    looks wrong only against WORLD_W, where it is 8% of the map. The reason it
    can happen at all is that the standoff used to pick its SIDE with a crc32
    (see `actions.hermit_home`), so a legacy save can have the hut on the
    shoulder a fresh siting would now reject; fresh sitings always take the
    roomier one and cannot land here.

    IT IS LEFT ALONE ON PURPOSE. The obvious fix - judge a standing hut against
    the roomier shoulder instead of its own - is exactly the churn bug the
    comment further down refuses: a settlement drifting across the map's
    midpoint would flip which shoulder is roomier, drop a perfectly good hut
    below the new inner wall without it having moved, and march him the width
    of the world to rebuild. Trading a permanent churn risk for a one-off
    legacy-save cosmetic is a bad trade. In practice it also tends to resolve
    itself: on that same seed 3 the colony's own drift pushed the cramped-side
    hut out of band after 95 s and he re-staked at 4095 px unprompted. That is
    luck rather than design, so it is stated as an observation and not relied
    on.

    WHERE: `hermit_home`, which is the derived seeded offset the first time and
    the standing hut's own x every time after - so this fires once per hut,
    and the successor to a dead hermit walks out to the hut the last one built
    instead of founding a new camp. It fires again only when the hut is gone:
    burned by a dragon or by lightning, ruined, and cleared away by
    `purge_ruins`. Then the offset is re-derived and he may well rebuild
    somewhere slightly different, because the settlement has moved since.

    Costs exactly one draw from the seeded stream (the variant roll inside
    ``reg.create``) and only on the tick a hut is staked.
    """
    if not HERMIT_HUT:
        return
    try:
        # ANY hermit_hut, RUBBLE INCLUDED, and that word is the whole of the
        # cooldown. `hermit_hut` deliberately skips ruins - a burned hut is
        # not shelter and must not go on anchoring the camp - but asking it here
        # would mean re-staking on the very next director pass, 15 s later, into
        # whatever destroyed the last one. Measured, and not hypothetically: seed
        # 7 runs a lava front across the hermit's camp, and this loop staked and
        # lost eleven huts in 135 s, one chronicle line and one rng draw each.
        # Rubble lingers for RUIN_LINGER (240 s), so counting it buys a real
        # cooldown out of machinery that already exists.
        herm = living_hermit(world)
        standing = hermit_hut(world)
        if standing is not None:
            # HAS THE VILLAGE WALKED AWAY FROM HIM? The hut does not move and
            # the settlement does: `settlement_center` is a mean over buildings,
            # and a colony that keeps expanding one way drags it. Measured on
            # seed 7: staked at a correct 642 px, and 25 minutes later the same
            # hut stood 974 px from a settlement that had grown six more huts
            # eastward - past the frame edge, which is the one failure the
            # standoff numbers exist to prevent.
            #
            # So he moves camp, rather than the hut teleporting or the
            # distance being allowed to drift. The old hut is COLLAPSED, not
            # deleted: it falls down, stands as rubble for RUIN_LINGER and is
            # swept up by `purge_ruins` like anything else, so a building the
            # user watched go up never simply vanishes.
            #
            # THE TEST IS THE BAND ITSELF, WITH NO SLACK ON EITHER SIDE, and
            # that is the fix to a measured breach of the rule the user chose
            # ("always at least 530 px out, never further than 1060"). It used
            # to read MIN-SLACK <= d <= MAX+SLACK, i.e. 440..1150, so a hut
            # staked correctly inside the band was then allowed to STAND well
            # outside it: 20.4% of samples over 16 seeds, ten of the sixteen
            # affected, as near as 446 px and as far as 1147.
            #
            # The give the slack was written for has not gone away, it has
            # MOVED: `_camp_site` and `hermit_home` now stake inside
            # `hermit_band(stake=True)`, the band inset by HERMIT_RESITE_SLACK
            # at the inner wall, so every new hut begins with 90 px of drift in
            # hand there and a colony wandering a few metres still does not cost
            # him a hut. The outer wall needs no inset: on a fractional band it
            # moves with the hut's own distance px for px, so drift cannot cross
            # it. What this no longer does is let him end up NEARER than the
            # distance he started at.
            #
            # THE NUMBERS ARE DERIVED, so read them off the constants rather
            # than off this comment; a previous version of it quoted
            # "430..740 against a staking band of 520..650", which matched
            # neither the code beside it nor the constants it named.
            if herm is None:
                return
            try:
                hx = float(standing.x)
                bx = float(_hermit_base(world))
                d = abs(hx - bx)
            except (TypeError, ValueError):
                return
            # THE HUT'S OWN SHOULDER, NOT THE ROOMIER ONE, and that distinction
            # is a churn bug that would otherwise be invisible. `hermit_band`
            # with side=None answers for whichever shoulder has the most room -
            # correct when SITING a camp, wrong when judging one that is already
            # standing, because a settlement drifting across the map's midpoint
            # flips which shoulder that is. The band would jump to the far
            # side's larger reach, the hut would fall below the new inner wall
            # without having moved a pixel, and he would strike a perfectly good
            # camp and march the width of the world to rebuild it. Asked about
            # the shoulder he is actually on, the outer wall tracks him exactly
            # (d - reach is invariant under drift) and the inner wall closes at
            # a quarter of the drift rate.
            lo, hi = hermit_band(world, 1.0 if hx >= bx else -1.0)
            if lo <= d <= hi:
                return
            standing.collapse("moved")
            standing.state["ruin_cause"] = "moved"
            chronicle(world, f"The village had grown away from him; "
                             f"{getattr(herm, 'name', 'the hermit')} struck camp.")
        else:
            # ANY hermit_hut, RUBBLE INCLUDED, and that word is the whole of the
            # cooldown. `hermit_hut` deliberately skips ruins - a burned hut
            # is not shelter and must not go on anchoring the camp - but asking
            # only it here would mean re-staking on the very next director pass,
            # 15 s later, into whatever destroyed the last one. Measured, and not
            # hypothetically: seed 7 runs a lava front across the hermit's camp,
            # and this loop staked and lost eleven huts in 135 s, one chronicle
            # line and one rng draw each. Rubble lingers for RUIN_LINGER (240 s),
            # so counting it buys a real cooldown out of machinery that already
            # exists. A camp he STRUCK himself is exempt - that rubble is a
            # decision, not a disaster, and he starts the new one at once.
            for s in reg:
                if (str(getattr(s, "kind", "")) == "hermit_hut"
                        and str(s.state.get("ruin_cause", "")) != "moved"):
                    return
        if herm is None:
            return
        x = _camp_site(world)
        y = float(ground_y(world, x))
        # ...and do not lay the first pole inside the thing that killed the last
        # one. The cooldown alone would still rebuild into a standing lava field
        # every four minutes for as long as it lasts.
        for h in hazards_of(world):
            try:
                hx = float(h.get("x", 0.0))
                r = float(h.get("radius", 0.0) or 0.0)
            except (TypeError, ValueError, AttributeError):
                continue
            if abs(hx - x) <= max(40.0, r) + 40.0:
                return
        reg.create("hermit_hut", x, y, rng=rng_of(world))
        chronicle(world, f"{getattr(herm, 'name', 'The hermit')} began raising a "
                         f"hut of his own, on the high ground past the village.")
    except Exception:
        log.debug("hermit hut staking failed", exc_info=True)


def _ensure_hermit_fire(world: Any, reg: StructureRegistry) -> None:
    """Stake his firepit at his camp when he has none. Never raises.

    The hut's sibling, and everything `_ensure_hermit_hut` says about being
    staked rather than conjured, about `_colony_sites` hiding it from the build
    queue, and about him raising it himself out of wood he cut himself, applies
    here word for word.

    THREE DIFFERENCES, and each of them is why this is a separate function
    rather than a second `reg.create` inside that one:

    1. IT IS ANCHORED TO THE HUT, not to `hermit_home`. The fire goes a few
       paces off his own doorstep, on the settlement side, so the two read as
       one camp rather than as two buildings that happen to share a hillside -
       and so that a resited camp takes its fire with it. That means it waits
       for the hut: with no hut standing or staked there is no doorstep to
       put it beside, and `hermit_worksite` would have him building a fire in
       the middle of nowhere and then walking away from it.
    2. IT DOES NOT RE-SITE ITSELF. `_ensure_hermit_hut` moves camp when the
       village has grown away from him; this one is collapsed by that move, one
       tick later, by the orphan test below - the fire whose hut is gone or is
       now 300 px away is rubble, and a fresh one is staked at the new door.
       Two rules for one camp would fight; one rule that follows the hut
       cannot.
    3. RUBBLE IS NOT A COOLDOWN HERE. The hut counts its own ruins to avoid
       re-staking into whatever burned it down, at a cost of RUIN_LINGER (240 s)
       of a hermit with no roof. A firepit is `flammable=False` and cannot burn,
       so the only things that ruin it are the hazards this function already
       refuses to build into, and making a man wait four minutes for a ring of
       stones buys nothing.

    Costs one draw from the seeded stream (the variant roll in ``reg.create``)
    and only on the tick a fire is staked.
    """
    if not HERMIT_HUT:
        return
    try:
        herm = living_hermit(world)
        if herm is None:
            return
        house = hermit_hut(world)
        fire = hermit_fire(world)
        if fire is not None:
            # ORPHANED? A fire standing on its own in the middle of nowhere is
            # the mis-sited building this project keeps shipping, so it goes -
            # but the test is "is there a camp here", NOT "is there a roof
            # here", AND RUBBLE COUNTS AS A CAMP.
            #
            # That distinction is a bug fix, not a nicety. `hermit_hut` skips
            # ruins by design, so asking it alone meant that the tick a dragon
            # finished burning his hut down, this line decided the fire was
            # orphaned and collapsed it too: the user watches a dragon burn a
            # hut and the fire beside it silently vanishes in the same second,
            # having been neither burned nor touched (the spec is
            # `flammable=False` precisely so it cannot burn). Measured on seed 7
            # - hut ignited, gone in 10 s, fire gone with it.
            #
            # What should happen, and now does, is that he loses his roof and
            # keeps his hearth: he sleeps rough beside a fire he still has while
            # he rebuilds the walls. So the scan below counts ANY hermit_hut
            # within reach, ruined or not. The fire is only given up when the
            # camp itself is gone - the rubble swept away by `purge_ruins` with
            # no new hut near it, or a hut re-staked somewhere else because
            # the village walked away.
            # A STANDING HUT WINS OVER RUBBLE, and that precedence is the whole
            # of this test. Both halves were measured:
            #
            #   * rubble has to count, or a dragon burning his hut down takes
            #     his fire with it in the same second - see the note above.
            #   * but rubble must not count while a standing hut exists
            #     SOMEWHERE ELSE. When the village grows away from him
            #     `_ensure_hermit_hut` collapses the old hut and stakes a new
            #     one further along; the old rubble lingers for RUIN_LINGER
            #     (240 s) right beside the fire, so a test that accepts any
            #     hermit_hut kept the fire pinned to the abandoned camp for four
            #     minutes while he lived at the new one. Measured on seeds 42 and
            #     1234: the fire was clipped by the frame edge in 43% and 46% of
            #     frames against the hut's 8% and 15% - a firepit stranded
            #     further out than the hut it belongs to, which is exactly the
            #     mis-sited building this function exists to prevent.
            #
            # So: if he has a roof, the fire belongs beside THAT roof. If he has
            # none, the fire waits by the wreckage for him to rebuild.
            try:
                anchor_x = (float(house.x) if house is not None else None)
            except (TypeError, ValueError):
                anchor_x = None
            near_camp = False
            if anchor_x is not None:
                near_camp = abs(anchor_x - float(fire.x)) <= 260.0
            else:
                for s in reg:
                    if str(getattr(s, "kind", "")) != "hermit_hut":
                        continue
                    try:
                        if abs(float(s.x) - float(fire.x)) <= 260.0:
                            near_camp = True
                            break
                    except (TypeError, ValueError):
                        continue
            if not near_camp:
                fire.collapse("moved")
                fire.state["ruin_cause"] = "moved"
            return
        if house is None:
            return                      # no doorstep yet - see (1) above
        # A few paces toward town, so the hut sits between the fire and the
        # wilderness and the camp reads as facing the colony it turned its back
        # on. Falls back to the far side if the near one is unwalkable.
        try:
            hx = float(house.x)
        except (TypeError, ValueError):
            return
        inward = -1.0 if hx > float(_hermit_base(world)) else 1.0
        x = _clamp_x(hx + inward * 34.0)
        if abs(slope_at(world, x)) > MAX_SLOPE_WALK:
            x = _clamp_x(hx - inward * 34.0)
        for h in hazards_of(world):
            try:
                hzx = float(h.get("x", 0.0))
                r = float(h.get("radius", 0.0) or 0.0)
            except (TypeError, ValueError, AttributeError):
                continue
            if abs(hzx - x) <= max(40.0, r) + 40.0:
                return
        reg.create("hermit_fire", x, float(ground_y(world, x)), rng=rng_of(world))
        chronicle(world, f"{getattr(herm, 'name', 'The hermit')} laid a ring of "
                         f"stones by his door and set a fire in it.")
    except Exception:
        log.debug("hermit fire staking failed", exc_info=True)


def hermit_guest(world: Any) -> Any | None:
    """The colonist currently walking out to see the hermit, or None."""
    v = getattr(world, "hermit_visit", None)
    if not isinstance(v, dict):
        return None
    a = agent_by_id(world, v.get("id"))
    return a if a is not None and getattr(a, "alive", True) else None


def _is_visiting(agent: Any) -> bool:
    """Is this colonist right now running the visit action? Never raises."""
    try:
        act = getattr(agent, "action", None)
        if act is None or str(getattr(act, "kind", "")) != "Converse":
            return False
        data = getattr(act, "data", None)
        return bool(isinstance(data, dict) and data.get("visit"))
    except Exception:
        return False


def _visit_gap(world: Any) -> float:
    """Seconds until the next visit is due. One draw from the seeded stream."""
    try:
        j = float(rng_of(world).uniform(-HERMIT_VISIT_JITTER, HERMIT_VISIT_JITTER))
    except Exception:
        j = 0.0
    return max(120.0, float(HERMIT_VISIT_PERIOD) + j)


def _tick_hermit_visit(world: Any) -> None:
    """Open, watch and close the one visit the colony pays him. Never raises.

    THE TRIGGER IS A CLOCK, not a mood, and that is the honest choice. Every
    state-based trigger considered here - his morale, the colony's, a festival,
    a death - either fires in bursts (three visits in a minute while the state
    holds) or never fires at all on the seeds where the state never arrives, and
    a feature that is invisible on half the seeds is the failure mode this
    project keeps hitting. A jittered clock fires at a rate that can be quoted
    and measured, which is the whole ask: rare enough to be an occasion, common
    enough to be seen.

    Two preconditions beyond the clock, and both are about it being a VISIT
    rather than an errand: there has to be a hermit, and his hut has to be
    STANDING. Somebody walking out to a patch of hillside is a man going for a
    walk; somebody walking out to a house is calling on a neighbour. It also
    means the first visit never lands during the twenty minutes he is still
    carrying wood about, which is when the walk out would be least legible.

    The slot is a single dict on the world and it is NOT saved - it is a
    transient like ``upgrade_job``, and a save taken mid-visit reloads with the
    visitor simply going home. Persisting it would be a second source of truth
    about an action that is already serialised on the visitor himself.
    """
    try:
        now = world_now(world)
        cur = getattr(world, "hermit_visit", None)
        if isinstance(cur, dict):
            # Watchdog. The action closes its own slot on every ordinary exit;
            # this is for the ones with no exit at all - the visitor eaten by a
            # wolf on the way out, the host dying while he walks.
            opened = 0.0
            try:
                opened = float(cur.get("opened", now))
            except (TypeError, ValueError):
                opened = now
            guest = hermit_guest(world)
            stale = False
            if guest is not None and now - opened > VISIT_PICKUP_GRACE:
                # He has had long enough to pick the action up and is not
                # running it: a wolf took him, or he ate, or he went to bed.
                # The appointment is over. Without this the slot stays open to
                # the full watchdog and the hermit spends it trying to talk to
                # somebody who left - see `_hermit_bias`.
                stale = not _is_visiting(guest)
            # The visitor's own budget plus the stay he is entitled to once he
            # gets there, so the watchdog can only ever fire AFTER the walk has
            # honestly run out - never during it. Reading a bare
            # HERMIT_VISIT_TIMEOUT here would have closed the slot under a man
            # who was still walking and still inside his own budget, which at
            # the new standoff is most of the walk.
            if (guest is None or living_hermit(world) is None or stale
                    or now - opened > hermit_visit_budget(world) + VISIT_STAY):
                world.hermit_visit = None
                # A failed appointment must not cost a whole period. The clock
                # was advanced when the slot opened, so without this line every
                # visitor who was asleep, eaten or busy when he was picked buys
                # the hermit another fifteen minutes of nobody - measured at 2
                # visits across 8 seeds x 45 min against a nominal ~13. Only the
                # failures land here: a visit that actually happened is closed by
                # `actions._close_hermit_visit`, so this branch never sees it.
                world._hermit_visit_due = now + VISIT_RETRY
            return

        due = float(getattr(world, "_hermit_visit_due", 0.0) or 0.0)
        if due <= 0.0:
            world._hermit_visit_due = now + _visit_gap(world)
            return
        if now < due:
            return
        if living_hermit(world) is None:
            return
        if hermit_hut(world, built_only=True) is None:
            return
        # Never the elder (he is the colony's centre, and sending him out is the
        # one absence a small colony notices), never a child, never the hermit.
        #
        # ...and never somebody who is plainly not going to come. A sleeper, a
        # man running from a wolf or a man mid-panic will not pick the walk up
        # inside VISIT_PICKUP_GRACE, and appointing him is an appointment that
        # is thrown away twenty seconds later. Cheap to check, and it is most of
        # the difference between the measured rate and the nominal one.
        busy = ("Sleep", "FleeFrom", "Panic")
        pool = [a for a in alive_agents(world)
                if not is_hermit(a) and _role(a) not in ("elder", "child")
                and str(getattr(getattr(a, "action", None), "kind", "")) not in busy]
        if len(pool) < 2:
            # Fewer than two people left at home behind the visitor. Hold the
            # clock where it is - do not reschedule - so the visit happens as
            # soon as the colony can spare somebody rather than being skipped.
            return
        pool.sort(key=lambda a: int(getattr(a, "id", 0) or 0))
        guest = pool[rng_of(world).randrange(len(pool))]
        world.hermit_visit = {"id": int(getattr(guest, "id", 0) or 0),
                              "opened": float(now)}
        world._hermit_visit_due = now + _visit_gap(world)
    except Exception:
        log.debug("hermit visit tick failed", exc_info=True)


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
#: THE RISK-BASED VERSION HAS NOW ALSO BEEN MEASURED, AND IT DOES NOT PAY.
#: Attempt 2 spent its effort on the measurement instead of the implementation,
#: and the measurement says the idea fails on arithmetic, not on plumbing. Four
#: probes, 8 'cliffs' seeds x 30 sim-min each, falls counted through the
#: ``entities.on_death`` hook:
#:
#:   1. *The geometry is right.* Stamp ``terrain.stamp_climb`` across EVERY face
#:      this survey reports, at t=0, for free - an oracle with perfect siting,
#:      no wood and no build time - and falls go **29 -> 2**, seven of the eight
#:      seeds to zero. So these faces really are where people die and defeating
#:      one really does work. That is the new fact and it is worth keeping.
#:   2. *Delivery is not the blocker either.* With this switch ON, the colony
#:      staked 16 crossings and FINISHED 12, and they land on the right rock:
#:      seven of them within 31 px of a body, typically 70-240 s after staking.
#:      Falls came out 29 -> 24, which is the same null the 40-seed A/B above
#:      reported. Ladders get built, at the right place, reasonably fast, and it
#:      still does not matter.
#:   3. *There is a risk signal, and it is too weak to aim with.* Counting how
#:      often an agent's x crosses each face's ``edge`` - the cheapest thing a
#:      pre-emptive detector could use - ranks the faces that go on to kill
#:      somebody above chance, but only just: the busiest face contains 2 of the
#:      12 killer faces, the top 3 contain 5, and you need the top TEN per map
#:      to contain all 12.
#:   4. *And here is the arithmetic that ends it.* 228 faces surveyed over the 8
#:      maps - 18 to 40 per map, not the 9.6 recorded above, because the world is
#:      now 6400 px wide - of which **12 (5.3%) ever killed anyone**. A ladder
#:      defeats exactly one face. So the oracle in (1) only works because it
#:      fixes all 28-odd at once; any affordable subset leaves people walking
#:      over the ones it did not fix, which is precisely what (2) measured. To
#:      buy the oracle honestly is ~28 ladders x (8 wood + 2 fibre) = ~224 wood
#:      and 56 fibre per map, laid one crossing at a time at 70-240 s each - more
#:      than the entire build economy of a 30-minute run, spent on rock.
#:
#: Falls are ~7.3 deaths per 1000 colonist-minutes against 44 from all causes
#: (32-seed all-styles pool), 'cliffs' is 31 of 200 seeds, and 5 of the 29 falls
#: above were not within 40 px of ANY surveyed face. Even a perfect three-ladder
#: detector is therefore worth a low-single-digit fraction of one percent of
#: deaths - far under what any seed count anyone will pay for can resolve. Say
#: so rather than measuring it again.
#:
#: If someone wants this to work, (1) says where to look: the win is in making
#: the *ground* survivable wholesale - a cheaper primitive than a per-face
#: ladder, or a fall model that self-arrests more often - not in choosing which
#: face to build at. Choosing better is not the missing piece.
#:
#: Left in place rather than deleted because the geometry is the part that was
#: right: ``_hazard_faces`` finds the rock people actually go over, which is what
#: probe (1) leans on, and anything that attacks falls will want it. What is
#: dead is the plan that hung off it - stake a crossing at the dangerous face -
#: on a corpse OR on a risk score. Do not turn this on to answer falls.
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
        # A hermit standing outside the colony's region is not stranded, he is
        # home. He is alone by design and he stays put for hours, so he clears
        # STRANDED_DWELL_LONE (90 s) permanently and would otherwise stake a
        # weight-3 "stranded" candidate that never expires - a bridge built
        # toward a man who does not want one, paid for out of the same labour
        # budget the MIN_POP trap was fixed by protecting.
        #
        # A HOLE STILL COUNTS, and that exception is the whole reason this is a
        # `continue` here rather than a filter on `alive` above. Living apart is
        # a choice; being at the bottom of a chasm with nothing to eat and no
        # way up is not, and it is a rescue for him on exactly the same terms as
        # for anybody else.
        if hole is None and is_hermit(a):
            continue
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

    def seniority(a: Any) -> tuple[int, float, int]:
        return (
            int(getattr(a, "generation", 0) or 0),
            -(_age_of(a) or 0.0),
            int(getattr(a, "id", 0) or 0),
        )

    # --- the eldest leads ---------------------------------------------------
    # The hermit is out of the running by construction. Without that exclusion
    # the two titles fight: the hermit is picked for seniority too, so on the
    # tick the elder dies the hermit is the next most senior, is made elder,
    # walks back into town, and the hermit slot is refilled behind him - a
    # three-way churn triggered by every elder's death.
    hermits = [a for a in adults if _role(a) == ROLE_HERMIT]
    elder_pool = [a for a in adults if _role(a) != ROLE_HERMIT]
    if len(adults) >= 3 and elder_pool:
        elder = sorted(elder_pool, key=seniority)[0]
        for a in elder_pool:
            if a is not elder and _role(a) == "elder":
                _set_role(a, _worker_role(world))
        if _role(elder) != "elder" and _set_role(elder, "elder"):
            chronicle(world, f"{getattr(elder, 'name', 'Someone')} is now the elder.")
    else:
        for a in elder_pool:
            if _role(a) == "elder":
                _set_role(a, _worker_role(world))

    # --- and one of them lives apart ----------------------------------------
    # Same shape as the elder above, with one difference that is the whole of
    # "when they die, another takes the name": INCUMBENCY. The elder is
    # recomputed from scratch every pass, which is fine for a title that means
    # "the most senior person here". The hermit's means "the one who went away",
    # and recomputing that every pass would move it to whoever happened to be
    # most senior at the time - so a child coming of age, or the elder dying,
    # would march a different colonist out into the wilderness. A sitting hermit
    # therefore keeps the title until he dies or the colony gets too small, and
    # succession is emergent exactly as the elder's is: he dies, he is no longer
    # in `adults`, `hermits` is empty on the next pass, and the next most senior
    # non-elder walks out and takes the name.
    if len(adults) < HERMIT_KEEP_ADULTS:
        # Too few hands left to spare one. Not the same number as the appointing
        # gate: with one threshold a colony sitting on the boundary would send a
        # man out and call him back every time somebody died and was born.
        for a in hermits:
            if _set_role(a, _worker_role(world)):
                chronicle(world,
                          f"{getattr(a, 'name', 'The hermit')} came back in "
                          f"from the wilds.")
    else:
        for extra in hermits[1:]:       # never more than one, whatever a save says
            _set_role(extra, _worker_role(world))
        if not hermits and len(adults) >= HERMIT_MIN_ADULTS:
            pool = [a for a in adults if _role(a) not in ("elder", ROLE_HERMIT)]
            if pool:
                pick = sorted(pool, key=seniority)[0]
                if _set_role(pick, ROLE_HERMIT):
                    chronicle(world,
                              f"{getattr(pick, 'name', 'Someone')} went to live "
                              f"apart, as the hermit.")

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
