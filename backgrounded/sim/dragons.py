"""Dragons - five rare visitors, and the first things in this world with a height.

Pure python - **no pygame**, and nothing from ``render/``. Like the rest of
``sim/`` this module must stay importable and runnable headless; the smoke block
at the bottom drives a real :class:`~backgrounded.sim.world.World` without a
display.

Built on :mod:`.ufo`'s shape - a seeded stream, a state machine, ``to_dict`` /
``from_dict``, and a scheduler that spends nearly all of its life counting down.
The differences from the saucer are the three things the brief asked for.

Five machines, not one body with five skins
-------------------------------------------
Every kind below has its own phase list, its own reach, and its own cost. They
are told apart by what they *do to you*, not by their outline:

``flyover``    THE PASSAGE. ``idle -> cross -> idle``. Enters off one edge,
               holds a constant vx and a constant SCREEN LINE across the whole
               frame, exits the far side. It never targets, never descends,
               never lands, never feeds. It cannot be reached - its band is
               300-420 px above the highest ground on the map, far outside both
               ``throwing.THROW_MAX_RANGE`` (240) and the 141 px ballistic
               ceiling - and it is the only one that is unreachable *by design*.
               It is also the HERALD: the first dragon a colony ever sees is
               always this one, so the tech unlock reads as a thing in the sky
               rather than as a line in a log.

               It is ATMOSPHERE, and that word had to be made true in code.
               Measured with the flat morale nudge it shipped with: dragons cost
               a colony ~14% more deaths and ~14% fewer survivors, and only 9 of
               the 48 extra deaths were anything actually eating anybody. The
               rest was this - 0.05 off everyone's morale, crossing
               behavior.py's Panic threshold at 0.32 for whoever was near it,
               turning a wolf incursion that would have been fought into one
               that was run from. The toll is now bounded so it cannot cross any
               morale gate in the game (see ``FLYOVER_MORALE_FLOOR``), and with
               that fixed, 16 seeds x 90 sim-min put deaths at 19.44 with
               dragons against 19.56 without - inside the noise, which is what
               scenery is supposed to cost.

``wyvern``     THE RAIDER. ``idle -> arrive -> hunt -> stoop -> seize -> leave``.
               Cruises at 150-210 out of everything's reach, stoops to alt 6-14,
               holds a colonist exactly the way ``Ufo._hold`` holds one for
               ``DRAGON_SEIZE_SEC``, and eats them. The seize is the ONE window
               in which it is inside ``BARRICADE_SPIKE_H`` and ``MELEE_CEILING``.
               That window is the fight.

``quadruped``  THE SIEGE. ``idle -> arrive -> roost -> raze -> leave``. Lands and
               STAYS, for ``DRAGON_SIEGE_SEC``, walking the terrain surface at
               alt 0 and razing whatever is built near it. Grounded for its
               whole dangerous phase, so spikes, melee and throws all reach it -
               the fair fight, and it is fair because it is the one that stays.

``serpent``    THE DEEP ONE. ``idle -> surface -> hunt -> seize -> leave``.
               Wingless, so it is not a sky creature and is never given an
               overflight role. It comes up out of the chasm (21.2% of maps,
               measured over 400 seeds) or, failing that, out of the deepest
               column of the heightmap. Its cost is positional: it arrives from
               the one place nobody defends, and ``behavior._barricade_site``
               puts barricades within ``BARRICADE_EDGE_FRAC`` of a screen EDGE,
               so the traps are in the wrong place for it by construction.

``skeletal``   THE REVENANT. ``idle -> arrive -> hunt -> scour -> leave``. Never
               targets a living agent. It circles the graveyard and takes
               headstones - ``DRAGON_SCOUR_MAX`` of them - and if the colony has
               buried nobody it never spawns at all, because there is nothing
               here it wants. Its whole cost is MEMORY. At alt 60-110 it is
               above melee and above the spikes but well inside throwing range,
               which makes it the throw-only dragon and therefore the right one
               to teach the throw with.

Altitude
--------
A dragon AUTHORS ``alt`` - px above the ground directly beneath it - and this
registry re-derives ``y = terrain.ground_y(x) - alt`` every single tick. ``y``
stays the one truth render reads; ``alt`` is the sim's intent. ``alt`` may be
NEGATIVE (the serpent while it is still down its hole) and :func:`clearance`
reports that honestly rather than clamping, because a predicate that lies about
being underground is worse than one that cannot measure.

Anything with no ``alt`` attribute at all - every wolf, bear, boar and stickman
in the game - reads 0.0, so every existing x-only predicate returns exactly the
answer it returned before this module existed.

Dragons are deliberately NOT in ``world.animals``. Three things would break at
once: ``AnimalRegistry._walk`` re-welds ``y = ground_y(x)`` after every move
(animals.py's header: "Nothing here flies or falls"), so a flyer's altitude
would be destroyed every tick; ``ANIMAL_MAX_ALIVE`` is 4, so one dragon would
crowd out an entire wolf incursion; and ``animals.danger_at`` /
``behavior._danger_for`` are x-only, so a dragon cruising 300 px overhead would
put the whole colony into permanent FleeFrom. Separate registry, duck-typed
surface: ``x/y/kind/health/max_health/alive/hurt()`` with Animal's exact
semantics, so ``combat_actions.hurt_animal`` and the throwing lane take a Dragon
through the same code path they take a wolf through.

Feeding, and why it is a rule rather than a big health bar
----------------------------------------------------------
``Dragon.hurt`` DISCARDS all damage while ``sated`` is False. Tuning this as
"very high health" was considered and rejected: the player cannot tell 900 hp
from invulnerable, the spears still visibly chip the bar, and the moment it
feeds nothing about the bar changes. A hard refusal plus a sentence in the
chronicle is the only version of this a player can actually read.

Three guards close the failure that rule creates - an unsated dragon that can
never feed is an immortal monster parked on the map forever:

* ``MIN_LEFT_BEHIND``, lifted verbatim from ufo.py. A dragon will not select a
  victim when the colony is down to one. Ufo's comment puts it exactly right: an
  emptied world "is not an event, it is a bug that looks like one".
* An agent is INELIGIBLE while ``inside`` is not None, while ``taken`` is True,
  or while not alive - so "everyone hides in the huts" is a legal, complete
  defence that must not deadlock.
* ``DRAGON_STARVE_SEC``: every dragon that CAN feed runs ``starve_t`` on any
  tick with no eligible victim, and at 90 s it latches ``sated`` anyway and
  becomes mortal. The clock resets the moment somebody is reachable, so it only
  runs while the colony is genuinely out of reach. Hiding indoors stays a real
  strategy; it costs ninety seconds and turns the monster killable.

``skeletal`` is constructed with ``sated = True``. That one field is the ENTIRE
implementation of "the undead do not eat" - ``hurt()`` has no special case and
so cannot drift.

The gorge, and why the rule shipped with only half of itself
------------------------------------------------------------
The rule is "a dragon gets one of you before it can be killed". The first half
always worked. The second half had NEVER HAPPENED ONCE: over 16 seeds x 90
sim-min plus 369 forced visits - 511 dragon visits - the chronicle wrote "The
wyrm has fed. It can be hurt now." and not one dragon was ever brought down.

Two causes, and they are the same mistake seen from two sides. ``_devour``
latched ``sated`` and moved the dragon straight to ``PHASE_LEAVE``, whose first
act is a 96 px/s climb: a wyvern was inside ``MELEE_CEILING`` for about a third
of a second after the sentence promising it could be hurt. And "leave" is not a
private word - ``combat_actions.animal_leaving`` reads any target whose
``state`` says leave/leaving/flee/depart as walking off the map and refuses to
spend a spear on it, on the entirely reasonable grounds that chasing one is how
a colony ends up unarmed. So a fed dragon was not merely fast, it was INVISIBLE
to every thrower in the colony for the whole rest of its life.

``PHASE_GORGE`` is the answer, and it is a phase rather than a slower leave
precisely because of that second cause: the name is read by another module.

* Leg one, ``DRAGON_GORGE_SEC``: it labours at ``GORGE_ALT`` (26 px), under
  ``MELEE_CEILING`` (44), under ``BARRICADE_SPIKE_H`` (58) and far under a
  thrower's ceiling. Every weapon the colony owns reaches it.
* Leg two: it climbs out at ``GORGE_CLIMB_RATE`` - 13 px/s against the 96 it
  leaves at - dragging back up through melee, then the spikes, then out of
  throwing range, visibly and one at a time.
* Only then does ``PHASE_LEAVE`` take over, by which point the word costs
  nothing because nothing could reach it anyway.

The serpent has no wings, so its leg two is a crawl home along the surface: it
stays at its crest, in reach, the whole way back to the hole. The quadruped
needs none of this - it is already at alt 0 for its entire 150 s siege.

Measured on the same 16 seeds x 90 sim-min, before and after:

    per dragon that became killable         before        after
    median seconds under a thrower           1.2 s       17.7 s
    median seconds inside melee reach        0.3 s       11.8 s
    dragons killed, of those killable       2 of 7       4 of 6
    dragons that FED and were then killed   1 of 5       3 of 4
    WYVERNS killed                          0 of 3       1 of 2
    hp ever taken off a wyvern                   0     34 - 156

The wyvern row is the whole story in one line. Its entire design is that the
seize is the window, and before this every wyvern that ever fed left with its
health bar untouched - not "hard to kill", literally never scratched.

Fifteen visits across sixteen colonies is not a rate, so the fraction was taken
separately under a forced regime of one dragon every five to ten minutes (18
seeds, 203 visits, 168 of them killable): colonies brought down 93, about
half. By kind, of the ones that became killable - quadruped 24 of 31, skeletal
38 of 69, wyvern 23 of 49, serpent 8 of 19. It is not a formality either way:
an armed colony killed 6 of 24 wyverns outright, and 6 of the 18 that escaped
left at 24 hp of 180 - one more spear.

The gorge is deliberately exempt from ``_VISIT_MAX``: a payoff that the patience
timer can cancel is not a payoff. ``GORGE_MAX_SEC`` is what pays for that
exemption, because an exemption without a bound of its own is a hang.

A dragon's victim DIES. It is not an abduction: the kill routes through
``Stickman.hurt`` with ``death_cause = "devoured"``, so ``World._reap_dead``
raises the grave, increments ``stats["died"]`` and ``World.reconcile()``'s
residual stays 0. Copying ufo.py's roster-removal path instead would silently
corrupt those books - that path exists precisely because abduction is NOT death,
and this is.

Randomness
----------
Every draw comes from this registry's OWN ``random.Random``, never from
``world.pyrng`` and never from the bare ``random`` module. That is not tidiness,
it is the single biggest measured risk on this feature: ``behavior.choose_action``
draws off ``world.pyrng``, which is the same stream ``animals._rng(world)``
spawns wolves from, and adding one inert scored candidate moved mean deaths from
11.33 to 13.17 (+16%) and mean animals-killed from 10.00 to 10.71 over 24 seeds
x 60 sim-min, with 0/24 seeds matching on any metric. A dragon scheduler pulling
on that stream would re-phase every wolf in the game. This one cannot.

The stream is REPLAYED on load rather than restarted: ``draws`` counts every
value taken and ``from_dict`` re-advances the fresh ``Random`` by that many, so
a reloaded world continues the sequence instead of starting it again. ufo.py
restarts (``random.Random(seed ^ 0x0F0)`` and nothing else), which is
deterministic but means a save/load can re-draw an interval it had already
spent. This is one integer more state and gets it exactly right.
"""
from __future__ import annotations

import logging
import math
import random
import zlib
from dataclasses import dataclass
from typing import Any, Iterator

from ..constants import (
    BARRICADE_SPIKE_H,
    DRAGON_FIRST_DELAY,
    DRAGON_GATE_STONE_HUTS,
    DRAGON_GORGE_SEC,
    DRAGON_INTERVAL_MAX,
    DRAGON_INTERVAL_MIN,
    DRAGON_KINDS,
    DRAGON_MAW_REACH,
    DRAGON_MAX_FED,
    DRAGON_RAZE_DPS,
    DRAGON_SCOUR_MAX,
    DRAGON_SCOUR_SEC,
    DRAGON_SEIZE_SEC,
    DRAGON_SIEGE_SEC,
    DRAGON_STARVE_SEC,
    DRAGON_STATS,
    MAX_DRAGONS_ALIVE,
    MAX_HEALTH,
    OFFSTAGE,
    RENDER_H,
    STAGE_HALF,
    WORLD_W,
)

log = logging.getLogger(__name__)

__all__ = [
    "Dragon", "DragonRegistry",
    "KIND_FLYOVER", "KIND_WYVERN", "KIND_QUADRUPED", "KIND_SERPENT",
    "KIND_SKELETAL", "KINDS",
    "PHASE_IDLE", "PHASE_ARRIVE", "PHASE_HUNT", "PHASE_STOOP", "PHASE_SEIZE",
    "PHASE_GORGE", "PHASE_ROOST", "PHASE_RAZE", "PHASE_CROSS", "PHASE_SCOUR",
    "PHASE_SURFACE", "PHASE_LEAVE", "PHASES", "KIND_PHASES",
    "CAUSE_DEVOURED", "MIN_LEFT_BEHIND",
    "clearance", "is_grounded",
    "registry_of", "dragons_of", "nearest_dragon",
]

# -------------------------------------------------------------------- kinds --
KIND_FLYOVER = "flyover"
KIND_WYVERN = "wyvern"
KIND_QUADRUPED = "quadruped"
KIND_SERPENT = "serpent"
KIND_SKELETAL = "skeletal"
KINDS: tuple[str, ...] = tuple(DRAGON_KINDS)

# ------------------------------------------------------------------- phases --
PHASE_IDLE = "idle"
PHASE_ARRIVE = "arrive"
PHASE_HUNT = "hunt"
PHASE_STOOP = "stoop"
PHASE_SEIZE = "seize"
#: The window. A dragon that has just eaten is heavy: it labours at
#: GORGE_ALT for DRAGON_GORGE_SEC and then climbs out at GORGE_CLIMB_RATE,
#: which is a fifth of the rate it leaves at. See :meth:`_tick_gorge`.
#:
#: The string is NOT "leave-slowly" or "depart" and that is deliberate:
#: ``combat_actions.animal_leaving`` scans a target's ``state`` for the words
#: leave/leaving/flee/fleeing/depart/departing and refuses to spend a spear on
#: anything wearing one. A fed dragon that goes straight to PHASE_LEAVE is not
#: merely fast, it is INVISIBLE to every thrower in the colony.
PHASE_GORGE = "gorge"
PHASE_ROOST = "roost"
PHASE_RAZE = "raze"
PHASE_CROSS = "cross"
PHASE_SCOUR = "scour"
PHASE_SURFACE = "surface"
PHASE_LEAVE = "leave"
PHASES: tuple[str, ...] = (
    PHASE_IDLE, PHASE_ARRIVE, PHASE_HUNT, PHASE_STOOP, PHASE_SEIZE,
    PHASE_GORGE, PHASE_ROOST, PHASE_RAZE, PHASE_CROSS, PHASE_SCOUR,
    PHASE_SURFACE, PHASE_LEAVE,
)

#: Which phases each kind actually visits. This is the machine, written down: it
#: is what the smoke block asserts every kind reaches, and it is the only place
#: that says a quadruped never stoops and a serpent never crosses. IDLE is the
#: terminal state every visit retires through, which is why it is in all five.
KIND_PHASES: dict[str, tuple[str, ...]] = {
    KIND_FLYOVER:   (PHASE_CROSS, PHASE_IDLE),
    KIND_WYVERN:    (PHASE_ARRIVE, PHASE_HUNT, PHASE_STOOP, PHASE_SEIZE,
                     PHASE_GORGE, PHASE_LEAVE, PHASE_IDLE),
    KIND_QUADRUPED: (PHASE_ARRIVE, PHASE_ROOST, PHASE_RAZE, PHASE_LEAVE,
                     PHASE_IDLE),
    KIND_SERPENT:   (PHASE_SURFACE, PHASE_HUNT, PHASE_SEIZE, PHASE_GORGE,
                     PHASE_LEAVE, PHASE_IDLE),
    KIND_SKELETAL:  (PHASE_ARRIVE, PHASE_HUNT, PHASE_SCOUR, PHASE_LEAVE,
                     PHASE_IDLE),
}

#: Death cause handed to ``Stickman.hurt``. Registered with names.py below so
#: the burial line reads as a devouring rather than falling through to the
#: generic "Something happened."
CAUSE_DEVOURED = "devoured"

#: Verbatim from ufo.py, and for the same reason. A dragon will not take the
#: last colonist: an emptied world is not an event, it is a bug that looks like
#: one, and a dragon must not be the thing that empties it.
MIN_LEFT_BEHIND = 1

# ------------------------------------------------------------------ tuning --
# Module-level, matching animals.py (ANIMAL_REACH, FEED_SEC, FOCUS_PENALTY) and
# combat_actions.py (FLEE_RADIUS, FIGHT_RADIUS). constants.py carries the
# numbers other modules read; these are ours alone.

#: Px outside the frame a dragon appears and disappears. Big enough that the
#: quadruped's whole airborne leg happens off-screen - see _tick_arrive.
SPAWN_MARGIN = 220.0
#: Px inside the frame that counts as "in", for the arrive->working transition.
ENTRY_INSET = 90.0
#: Px/s of climb or descent on an arrive or leave leg.
ALT_RATE = 96.0

#: Px above the highest ground the crossing passes over - see
#: :meth:`DragonRegistry._sky_line`
#: for why it is not "above the ground beneath", which is what it used to mean.
FLYOVER_ALT = (300.0, 420.0)
#: ...and how far inside the top of the frame the crossing is held whatever
#: that band works out to. A peak at y 407 minus a 412 px band is y -5, which is
#: a herald nobody ever sees.
FLYOVER_SCREEN_MIN_Y = 40.0
FLYOVER_SPEED = (36.0, 52.0)
#: What a shape crossing the sky does to the people who see it. Applied ONCE, on
#: entry, and guarded: it is a mood, not a mechanic.
FLYOVER_MORALE = 0.05
#: ...and this is what makes that sentence TRUE rather than an intention.
#:
#: Measured: with dragons on, colonies lost ~14% more people and kept ~14%
#: fewer, and only 9 of the 48 extra deaths were a dragon eating anybody. The
#: rest came from here. behavior.py scores ``Panic`` at 0.62 the moment an agent
#: is in danger with morale below 0.32, so a flat 0.05 off everyone turns a
#: wolf incursion that a colony at 0.34 would have fought into one it runs from
#: - and a colony that runs loses people to the wolves, to falls while running,
#: and to the work that stopped. The flyover was buying a body count with a
#: number that is supposed to be scenery.
#:
#: So the toll now applies only ABOVE this line and never crosses it. Every
#: morale gate in the game sits below it - Panic 0.32, MORALE_TO_GROW 0.45,
#: HUT_TIER_MORALE 0.55 - and an agent below the line is not touched at all,
#: which makes "the flyover cannot change any decision anybody makes" a
#: property of the code instead of a hope. The colony mean still dips, because
#: the content are momentarily less content, and that is the entire point.
FLYOVER_MORALE_FLOOR = 0.62

WYVERN_ALT = (150.0, 210.0)
WYVERN_SPAWN_ALT = 260.0
#: Px of horizontal gap at which the wyvern commits to the dive.
STOOP_TRIGGER = 110.0
#: Seconds the dive takes, cruise altitude down to SEIZE_ALT.
STOOP_SEC = 1.4
#: Where it holds the victim. Inside BARRICADE_SPIKE_H (58) and inside
#: MELEE_CEILING (44) - deliberately, this is the whole fight.
SEIZE_ALT = (6.0, 14.0)
#: Px of horizontal slack that still counts as having hold of somebody.
SEIZE_REACH = 24.0
#: Seconds of the victim's own velocity the stoop leads by. A running target is
#: genuinely harder to take because the aim is wrong, not because a roll failed.
STOOP_LEAD_SEC = 0.9

# -- the gorge: the window a fed dragon is actually killable in ---------------
# The altitudes here are the load-bearing column, exactly as in DRAGON_STATS,
# and they were chosen against the three reaches rather than by eye:
#
#   MELEE_CEILING       44   a stickman's swing
#   BARRICADE_SPIKE_H   58   a built trap
#   throwing            141 px straight up, falling to 38 px at the 240 px
#                            range limit (throwing.py's safety-parabola table);
#                            "anything meant to be killable by throwing has to
#                            fly under about 100 px" is that module's own words.
#
#: Where a gorging dragon labours. Under all three, so for these seconds every
#: weapon the colony owns reaches it and the fight is whatever the colony
#: actually built. 26 rather than 0: it is still a flying animal with a body
#: under it, and render/dragons.py has no landed pose for a wyvern.
GORGE_ALT = 26.0
#: Px/s of the climb out, against ALT_RATE's 96. A dragon heavy with prey
#: climbs slowly - that is the whole fiction, and the five-fold difference is
#: what turns "it can be hurt now" from a sentence into seconds.
GORGE_CLIMB_RATE = 13.0
#: Px. Once it is this high the window is closed on every weapon (the throwing
#: ceiling is 141 px at zero horizontal gap and less at every real one), so
#: there is nothing left to protect and it hands over to PHASE_LEAVE.
GORGE_ESCAPE_ALT = 145.0
#: Px/s it wallows sideways while heavy. Not zero - a dragon parked in the air
#: reads as a bug - and not its cruise, which is what "heavy" has to mean.
GORGE_DRIFT = 16.0
#: Px either side of ``home_x`` it wallows over. Bounded on purpose: an unbounded
#: drift for the whole window is 340 px, which carries a dragon that fed near a
#: frame edge off the map and out of the fight it is the payoff for.
GORGE_WALLOW = 48.0
#: Hard ceiling on the whole phase, whatever the arithmetic above does. The
#: gorge is exempt from _VISIT_MAX (a payoff that patience can cancel is not a
#: payoff), so it owns its own upper bound or it owns a hang.
GORGE_MAX_SEC = DRAGON_GORGE_SEC + 40.0

QUAD_ALT = (90.0, 130.0)
#: Px/s a roosting quadruped walks between the buildings it is wrecking.
QUAD_WALK = 18.0
#: Px. Nearest built structure inside this is what gets razed.
RAZE_RANGE = 40.0
#: Px it wanders past its roost before turning round. Keeps the siege legible as
#: one animal working over a settlement rather than a patrol.
ROOST_RANGE = 190.0

#: Px below the surface the serpent's lair sits, and where it retires to.
SERPENT_DEPTH = 150.0
#: Px/s it rises and sinks. Slower than it swims: coming up is the drama.
SERPENT_RISE = 42.0
#: The crest. alt runs 0..18 for the whole surfaced phase - inside every reach a
#: colonist has, which is precisely why the barricades being at the EDGES and
#: not at the chasm is the interesting part.
SERPENT_CREST = 9.0
SERPENT_SWELL = 8.0

SKELETAL_ALT = (60.0, 110.0)
SKELETAL_SPAWN_ALT = 250.0
#: Px of the circle it flies over the graveyard.
CIRCLE_R = 46.0
#: Px. The nearest headstone inside this is the one it takes.
SCOUR_REACH = 90.0

#: Ground movement, copied from animals.py so a dragon on the surface climbs the
#: same terrain a wolf does: a step that would rise or fall faster than this is
#: SHORTENED, never teleported, and never below MIN_STEP_FRAC of itself or a
#: sheer face stalls the creature on the spot forever.
CLIMB_RATE = 150.0
MIN_STEP_FRAC = 0.12

#: Seconds a visit may last before the dragon gives up and goes, whatever it is
#: doing. This is ``ANIMAL_LEAVE_SEC``'s job ("patience runs out whatever it was
#: doing") and it is NOT optional here, because the starve hatch alone does not
#: close the hiding-indoors case: DRAGON_STARVE_SEC makes an unreachable dragon
#: MORTAL, but nothing in the feeding rule makes it LEAVE, so a colony that
#: locked itself in the huts got a killable wyvern circling the camp forever.
#: Measured before this existed: 260 s of everyone indoors, still hunting.
#:
#: The quadruped's own DRAGON_SIEGE_SEC ends its roost long before this; the
#: entry here is the backstop for a siege that somehow cannot end.
_VISIT_MAX: dict[str, float] = {
    KIND_FLYOVER: 120.0,        # a crossing at the slowest vx takes ~57 s
    KIND_WYVERN: 180.0,
    KIND_QUADRUPED: DRAGON_SIEGE_SEC + 60.0,
    KIND_SERPENT: 180.0,
    KIND_SKELETAL: DRAGON_SCOUR_SEC * DRAGON_SCOUR_MAX + 90.0,
}

#: Relative weight per kind once the herald has been spent. Skeletal is dropped
#: from the roll on a colony with no graves; the remaining weights are
#: renormalised, so removing it makes the others likelier rather than making the
#: whole roll fail.
_KIND_WEIGHTS: dict[str, float] = {
    KIND_WYVERN: 0.34,
    KIND_QUADRUPED: 0.22,
    KIND_SKELETAL: 0.20,
    KIND_SERPENT: 0.14,
    KIND_FLYOVER: 0.10,
}

#: ...and skeletal is the ONE weight that moves, because it is the one kind
#: whose reason for coming is a thing the colony accumulates. It goes up by this
#: much per headstone standing, capped at SKELETAL_WEIGHT_MAX.
#:
#: WHY, and what was actually wrong. The report was "the bone-wyrm never
#: appears - 0 sightings in 16 colonies over 24 sim-hours", and the obvious
#: reading is a broken spawn path. It is not: instrumented over 8 colonies with
#: the interval shortened, ``_pick_kind`` chose skeletal 35 times in 185 rolls
#: (19%, which is exactly its weight) and ``_spawn`` returned a live dragon on
#: all 35. Nothing in the spawn path is broken.
#:
#: What is wrong is the arithmetic of a session. Measured over 16 seeds x 90
#: sim-min with the shipped interval: 13 colonies reached the gate and the
#: sixteen of them saw SEVENTEEN dragons between them - and ten of those were
#: the herald, which is always the flyover. Seven real visits across sixteen
#: colonies, of which skeletal's fair share is 1.4. Zero sightings is not a bug
#: in that distribution, it is Tuesday.
#:
#: So the fix is not to the machine, it is to the roll, and it is the one change
#: that says something true rather than just raising a number: a colony that has
#: buried ten people is a colony the revenant has business with. At MAX_GRAVES
#: (10) skeletal takes ~29% of the roll instead of 20%; a colony that has buried
#: nobody still never sees it at all, which is the rule that was already there.
#:
#: The cap is 0.32 and not higher ON PURPOSE, and it was measured too: at 0.45
#: the bone-wyrm became the MODAL dragon over 203 visits - 69 of them, against
#: the wyvern's 49 - which fixes "you never see it" by making the harmless one
#: the one you mostly see. 0.32 keeps it strictly under the wyvern's 0.34, so
#: the commonest visit is still the dangerous one however full the graveyard is.
SKELETAL_WEIGHT_PER_GRAVE = 0.02
SKELETAL_WEIGHT_MAX = 0.32

#: What the chronicle calls each one. Written per kind rather than composed from
#: a noun because this log is the thing a person actually reads, and because the
#: feeding lines ("The spear glances off the wyrm. It has not fed.") were
#: specified against the wyvern's word.
_NOUN: dict[str, str] = {
    KIND_FLYOVER: "shape",
    KIND_WYVERN: "wyrm",
    KIND_QUADRUPED: "drake",
    KIND_SERPENT: "serpent",
    KIND_SKELETAL: "bone-wyrm",
}

_ARRIVAL_LINES: dict[str, str] = {
    KIND_FLYOVER: "Something crosses the sky, high up.",
    KIND_WYVERN: "A shadow comes over the ridge on wide wings.",
    KIND_QUADRUPED: "Something huge comes down out of the hills.",
    KIND_SERPENT: "The low ground splits, and something comes up out of it.",
    KIND_SKELETAL: "A wyrm of bone turns slow circles above the graves.",
}

#: The WORLD's rim - the hard clamp on where a dragon may stand, and nothing
#: else. It used to double as "the edge of the frame" because the world was
#: exactly one frame wide; on a WORLD_W map those are up to 2400 px apart, and
#: every use that meant the second one now goes through :func:`_stage_edge`,
#: :func:`_spawn_x` or :func:`_on_stage`.
_EDGE_MIN = 0.0
_EDGE_MAX = float(WORLD_W - 1)
_MAX_DT = 0.25
_FALLBACK_STATS: tuple[float, float, float] = (150.0, 60.0, 120.0)
#: Cap on the replay loop in from_dict. A hand-edited "draws": 1e12 must cost a
#: fraction of a second, not the session.
_MAX_REPLAY = 200_000

_RNG = random.Random()


# --------------------------------------------------------------- coercions --
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


# ------------------------------------------------------------------ stage ---
# ``sim/`` may not look at render's camera. It does not have to: the camera is
# RENDER_W wide and follows the colony, so a point further than OFFSTAGE from
# whatever the colony is gathered around is off-camera BY CONSTRUCTION. Every
# statement in this file that used to say "the frame edge" - and there were
# several, including the one _tick_arrive's whole design rests on - said it in
# world coordinates because the world was one frame wide. These three restore
# the meaning without giving sim/ any knowledge of render/.
#
# They take the ANCHOR as a number rather than the world, because a dragon
# already carries the right anchor: ``d.home_x`` is the colony for a wyvern,
# quadruped and flyover, the lair for a serpent and the graveyard for the
# skeletal. ``actions.offstage_x(world, side)`` is the same idea anchored on
# colony_center(), and is the right call anywhere that has no home_x.
def _stage_edge(anchor: float, side: int) -> float:
    """The x on *side* of *anchor* at which a thing becomes visible."""
    s = 1.0 if side >= 0 else -1.0
    return _clamp(anchor + s * OFFSTAGE, _EDGE_MIN, _EDGE_MAX)


def _spawn_x(anchor: float, side: int) -> float:
    """Where a dragon appears: SPAWN_MARGIN px beyond the stage edge.

    Deliberately NOT clamped to the land - dragons arrive and leave through the
    air and are allowed to be outside the map while they do, exactly as they
    were when they entered at ``-SPAWN_MARGIN``. The clamp that IS here only
    stops a runaway anchor putting one an absurd distance out.
    """
    s = 1.0 if side >= 0 else -1.0
    return _clamp(anchor + s * (OFFSTAGE + SPAWN_MARGIN),
                  _EDGE_MIN - OFFSTAGE - SPAWN_MARGIN,
                  _EDGE_MAX + OFFSTAGE + SPAWN_MARGIN)


def _on_stage(anchor: float, x: float, inset: float = 0.0) -> bool:
    """True once *x* is near enough *anchor* that the camera could show it."""
    return abs(_f(x, 0.0) - _f(anchor, 0.0)) <= max(0.0, OFFSTAGE - inset)


def _approach(cur: float, target: float, rate: float, dt: float) -> float:
    step = max(0.0, rate) * dt
    if cur < target:
        return min(target, cur + step)
    if cur > target:
        return max(target, cur - step)
    return cur


def _stats_for(kind: str) -> tuple[float, float, float]:
    """``(max_health, speed, cruise_alt)`` for *kind*. Never raises."""
    try:
        got = DRAGON_STATS.get(kind)
        if got is None:
            return _FALLBACK_STATS
        mh, spd, alt = got
        return (max(1.0, _f(mh, 150.0)), max(1.0, _f(spd, 60.0)), _f(alt, 120.0))
    except Exception:
        return _FALLBACK_STATS


def _cap_for(kind: str) -> int:
    try:
        return max(0, _i(DRAGON_MAX_FED.get(kind), 0))
    except Exception:
        return 0


def _band(d: "Dragon", span: tuple[float, float]) -> float:
    """This visit's altitude inside *span*, jittered off the dragon's own id.

    Two requirements make an rng draw the wrong tool here. It must be the SAME
    number on every tick - an altitude re-rolled each frame is a dragon
    vibrating in the sky, which is what the first cut of _tick_arrive actually
    did - and it must survive a save without costing a persisted field. ``id``
    is already both: stable for the life of the visit and in ``to_dict``.

    Knuth's multiplicative constant against a prime modulus, not ``hash()``:
    CPython salts str hashing per process and this project has shipped six
    seeds that were reproducible within a launch and different on the next one.
    An integer id would not actually be salted, but the habit is the point.
    """
    lo, hi = float(span[0]), float(span[1])
    frac = ((max(0, int(d.id)) * 2654435761) % 1009) / 1008.0
    return lo + (hi - lo) * frac


# ------------------------------------------------------------------ altitude --
# The two predicates the rest of the package needs. They are duplicated into
# sim/altitude.py by the lane that owns it; this copy is here so dragons.py has
# no import that can fail, and so grounded() below works on a checkout where
# that file does not exist yet. The semantics are identical and deliberately
# total.
GROUND_EPS: float = 2.0


def clearance(world: Any, obj: Any) -> float:
    """Px between *obj* and the ground under it. 0.0 for anything on it.

    Prefers an authored ``alt``; falls back to ``ground_y(x) - y``; falls back
    to 0.0. **Never raises and never returns a non-finite number**, because a
    predicate that cannot measure height must answer "on the ground" - which is
    the answer every caller in this package gave before altitude existed.

    That totality is not politeness. ``Structure._tick_barricade`` is on the
    per-frame path and runs inside ``World._guarded("structures")``: a
    clearance() that raises would not just break barricades, it would silently
    disable the entire structures subsystem for the session and the colony would
    stop building, with nothing on screen saying why.

    May return NEGATIVE - the serpent while it is still down its hole - and that
    is reported honestly rather than clamped.
    """
    try:
        alt = getattr(obj, "alt", None)
        if alt is not None:
            out = float(alt)
            return out if math.isfinite(out) else 0.0
    except Exception:
        pass
    try:
        terrain = getattr(world, "terrain", None)
        fn = getattr(terrain, "ground_y", None)
        if callable(fn):
            gy = float(fn(float(getattr(obj, "x", 0.0))))
            oy = float(getattr(obj, "y", gy))
            out = gy - oy
            return out if math.isfinite(out) else 0.0
    except Exception:
        pass
    return 0.0


def is_grounded(world: Any, obj: Any, ceiling: float = GROUND_EPS) -> bool:
    """True when *obj* is within *ceiling* px of the ground under it."""
    try:
        return clearance(world, obj) <= max(0.0, _f(ceiling, GROUND_EPS))
    except Exception:
        return True


# ---------------------------------------------------------- world adapters --
# The world may be a stub, mid-load or missing a subsystem. Every read and write
# goes through one of these, so none of that reaches the state machines. Same
# policy as animals.py and ufo.py, and for the same reason: this runs on a
# per-frame path and a per-frame path that raises is a program that dies
# overnight.
def _ground_y(world: Any, x: float) -> float:
    terrain = getattr(world, "terrain", None)
    fn = getattr(terrain, "ground_y", None)
    if not callable(fn):
        return RENDER_H * 0.72
    try:
        return _f(fn(float(x)), RENDER_H * 0.72)
    except Exception:
        return RENDER_H * 0.72


def _agents(world: Any) -> list[Any]:
    """Every living stickman, or [] if there is no roster."""
    pop = getattr(world, "population", None)
    fn = getattr(pop, "alive_agents", None)
    if callable(fn):
        try:
            return [a for a in fn() if getattr(a, "alive", False)]
        except Exception:
            return []
    try:
        return [a for a in (pop or ()) if getattr(a, "alive", False)]
    except Exception:
        return []


def _agent_by_id(world: Any, aid: Any) -> Any:
    if aid is None:
        return None
    fn = getattr(world, "agent_by_id", None)
    if callable(fn):
        try:
            return fn(_i(aid, -1))
        except Exception:
            return None
    for a in _agents(world):
        if _i(getattr(a, "id", -1), -1) == _i(aid, -2):
            return a
    return None


def _log(world: Any, text: str) -> None:
    fn = getattr(world, "log_event", None)
    if callable(fn) and text:
        try:
            fn(text)
        except Exception:
            log.debug("chronicle write failed", exc_info=True)


def _structures(world: Any) -> list[Any]:
    reg = getattr(world, "structures", None)
    fn = getattr(reg, "all", None)
    if not callable(fn):
        return []
    try:
        return list(fn())
    except Exception:
        return []


def _graves(world: Any) -> list[Any]:
    """A SNAPSHOT of the headstones.

    Snapshot, because the skeletal removes one while walking this list and
    ``World._weather_graves`` independently sorts and drops the oldest past
    MAX_GRAVES on the same tick - two things mutating the same collection.
    Iterating the registry's own view while removing from it is the obvious
    footgun and this is the whole guard against it.
    """
    props = getattr(world, "props", None)
    fn = getattr(props, "all_of", None)
    if not callable(fn):
        return []
    try:
        return [g for g in list(fn("grave")) if getattr(g, "alive", True)]
    except Exception:
        return []


def _remove_prop(world: Any, prop: Any) -> bool:
    props = getattr(world, "props", None)
    fn = getattr(props, "remove", None)
    if not callable(fn):
        return False
    try:
        return bool(fn(prop))
    except Exception:
        return False


# ------------------------------------------------------------- chronicle --
def _register_chronicle_kind() -> None:
    """Teach names.py what being eaten looks like, once, at import.

    ``World._reap_dead`` writes one line per death from ``death_cause``, and
    ``"devoured"`` is not in ``names.DEATH_KINDS``, so without this every dragon
    kill would fall through to the generic "Something happened." - the exact
    failure animals.py fixed for ``"mauled"``. The kill itself is announced from
    :meth:`DragonRegistry._devour` the instant it happens (that is the only
    place that knows *which* dragon did it), so what is registered here is the
    later burial line, not a second death notice.

    ``setdefault`` throughout: if names.py ever grows its own, it wins.
    """
    try:
        from . import names as _names                      # noqa: PLC0415

        _names.DEATH_KINDS.setdefault(CAUSE_DEVOURED, "died_devoured")
        _names.EVENT_TEMPLATES.setdefault("died_devoured", (
            "There was nothing of {name} left to bury.",
            "They raise a stone for {name} on the {where} slope, and bury nothing under it.",
            "{name} is remembered at the {where} edge. There was no body.",
        ))
    except Exception:
        log.debug("could not register the devoured chronicle kind", exc_info=True)


_register_chronicle_kind()


# ================================================================== Dragon ==
@dataclass
class Dragon:
    """One visitor. Everything declared here survives a restart."""

    id: int = 0
    kind: str = KIND_WYVERN

    # --- kinematics -------------------------------------------------------
    x: float = 0.0
    #: DERIVED every tick from ``ground_y(x) - alt``. Render reads this and
    #: only this; never author it directly.
    y: float = 0.0
    #: AUTHORED. Px above the ground directly beneath. May be negative.
    alt: float = 0.0
    vx: float = 0.0

    health: float = 1.0
    max_health: float = 1.0

    state: str = PHASE_ARRIVE
    t: float = 0.0                        # s in the current phase
    age: float = 0.0                      # s since it arrived
    facing: int = 1                       # -1 | +1
    anim_t: float = 0.0
    target_id: int | None = None          # stickman being hunted or held

    #: The feeding latch. False means damage is REFUSED - see :meth:`hurt`.
    #: True at birth for skeletal, which is the entire implementation of "the
    #: undead do not eat".
    sated: bool = False
    #: Once-per-dragon guard on the "it has not fed" chronicle line, so a colony
    #: throwing six spears gets one sentence and not six.
    told_unsated: bool = False
    #: Seconds accumulated with NO eligible victim. The deadlock hatch.
    starve_t: float = 0.0
    fed_count: int = 0
    #: Chasm mouth, graveyard centroid, or roost - whatever this kind orbits.
    home_x: float = 0.0
    #: Seconds since damage was last refused; -1.0 while it never has been.
    #: Persisted because render may flash the refusal, and because a save taken
    #: one tick after a spear bounced should reload having bounced it.
    refused_t: float = -1.0

    def __post_init__(self) -> None:
        # Transient, never persisted: the edge flag hurt() raises for the
        # registry's tick to turn into a chronicle line. hurt() has no world in
        # scope - it is called from actions.py, from throwing.py and from a
        # barricade - so it cannot write the line itself.
        self._refused_pending: bool = False

    # ------------------------------------------------------------ derived --
    @property
    def alive(self) -> bool:
        return self.health > 0.0

    @property
    def hostile(self) -> bool:
        """Worth aiming at: alive, in the world, and not the passing shape.

        A leaving dragon is deliberately still hostile - one that has eaten
        somebody and is climbing away is a perfectly good target, and refusing
        to let the colony throw at it would be a strange lesson. ``flyover`` is
        never hostile: it is atmosphere, it takes nothing, and it is 300 px up
        where nothing can reach it anyway.
        """
        return (self.alive and self.kind != KIND_FLYOVER
                and self.state != PHASE_IDLE)

    @property
    def health_frac(self) -> float:
        return _clamp(self.health / max(1e-6, self.max_health), 0.0, 1.0)

    @property
    def speed(self) -> float:
        return _stats_for(self.kind)[1]

    @property
    def cruise_alt(self) -> float:
        return _stats_for(self.kind)[2]

    @property
    def max_fed(self) -> int:
        return _cap_for(self.kind)

    @property
    def eats(self) -> bool:
        """Whether this kind takes people at all. flyover and skeletal do not."""
        return self.max_fed > 0

    @property
    def noun(self) -> str:
        return _NOUN.get(self.kind, "wyrm")

    def distance_to(self, x: float) -> float:
        """Horizontal gap only - matches ``Animal.distance_to`` exactly, so the
        duck-typed callers in actions.py and throwing.py cannot tell them apart."""
        return abs(self.x - _f(x, self.x))

    def summary(self) -> str:
        bits = [f"{self.kind}#{self.id} {self.state}",
                f"{self.health:.0f}/{self.max_health:.0f}",
                f"@{self.x:.0f} alt {self.alt:.0f}"]
        if self.target_id is not None:
            bits.append(f"-> agent {self.target_id}")
        bits.append("sated" if self.sated else "unfed")
        if self.fed_count:
            bits.append(f"ate {self.fed_count}")
        return " ".join(bits)

    # -------------------------------------------------------------- combat --
    def hurt(self, amount: float) -> bool:
        """Take *amount* damage. True only on the live -> dead edge.

        Same signature and same edge semantics as ``Animal.hurt``, so a Dragon
        can go through ``combat_actions.hurt_animal``, through a thrown spear
        and through ``Structure._tick_barricade`` without any of them knowing
        what they hit.

        **All damage is discarded while ``sated`` is False.** That is a RULE and
        not health tuning, and the alternative was considered and rejected:
        giving it 900 hp instead is invisible - the player cannot tell 900 from
        invulnerable, the spears still visibly chip the bar, and at the moment
        it feeds nothing about the bar changes. A hard refusal plus one sentence
        in the chronicle is the only version of this a player can read.

        The refusal raises a transient flag rather than logging: this method has
        no world in scope and must stay callable from anywhere.
        """
        dmg = _f(amount, 0.0)
        if not self.alive or dmg <= 0.0:
            return False
        if not self.sated:
            self.refused_t = 0.0
            self._refused_pending = True
            return False
        self.health = max(0.0, self.health - dmg)
        if self.health <= 0.0:
            self.health = 0.0
            self.vx = 0.0
            self.target_id = None
            return True
        return False

    # ------------------------------------------------------------ persist --
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "kind": str(self.kind),
            "x": float(self.x),
            "y": float(self.y),
            "alt": float(self.alt),
            "vx": float(self.vx),
            "health": float(self.health),
            "max_health": float(self.max_health),
            "state": str(self.state),
            "t": float(self.t),
            "age": float(self.age),
            "facing": int(self.facing),
            "anim_t": float(self.anim_t),
            "target_id": None if self.target_id is None else int(self.target_id),
            "sated": bool(self.sated),
            "told_unsated": bool(self.told_unsated),
            "starve_t": float(self.starve_t),
            "fed_count": int(self.fed_count),
            "home_x": float(self.home_x),
            "refused_t": float(self.refused_t),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Dragon":
        """Rebuild from a save. Junk takes defaults; never raises."""
        g = cls()
        if not isinstance(d, dict):
            return g
        try:
            g.id = max(0, _i(d.get("id"), 0))
            kind = d.get("kind")
            g.kind = kind if kind in KINDS else KIND_WYVERN
            default_hp, _spd, default_alt = _stats_for(g.kind)

            g.x = _clamp(_f(d.get("x"), 0.0),
                         _EDGE_MIN - SPAWN_MARGIN * 2.0,
                         _EDGE_MAX + SPAWN_MARGIN * 2.0)
            g.alt = _clamp(_f(d.get("alt"), default_alt), -900.0, 1200.0)
            g.y = _f(d.get("y"), 0.0)
            g.vx = _clamp(_f(d.get("vx"), 0.0), -600.0, 600.0)
            g.max_health = max(1.0, _f(d.get("max_health"), default_hp))
            g.health = _clamp(_f(d.get("health"), g.max_health), 0.0, g.max_health)

            state = d.get("state")
            allowed = KIND_PHASES.get(g.kind, PHASES)
            # A phase this kind does not have is not survivable state: a
            # quadruped loaded in "stoop" would fall through the dispatch and
            # hang in the air. Send it home instead - one wasted visit beats a
            # dragon frozen mid-air for the rest of the session.
            g.state = state if isinstance(state, str) and state in allowed \
                else PHASE_LEAVE
            g.t = max(0.0, _f(d.get("t"), 0.0))
            g.age = max(0.0, _f(d.get("age"), 0.0))
            g.facing = -1 if _i(d.get("facing"), 1) < 0 else 1
            g.anim_t = _f(d.get("anim_t"), 0.0)

            tid = d.get("target_id")
            g.target_id = _i(tid, 0) if isinstance(tid, (int, float)) and not \
                isinstance(tid, bool) else None

            g.sated = _b(d.get("sated"), False)
            g.told_unsated = _b(d.get("told_unsated"), False)
            g.starve_t = _clamp(_f(d.get("starve_t"), 0.0), 0.0, DRAGON_STARVE_SEC)
            g.fed_count = max(0, _i(d.get("fed_count"), 0))
            g.home_x = _clamp_x(d.get("home_x"))
            g.refused_t = _f(d.get("refused_t"), -1.0)

            # The feeding exemption is ONE FIELD, and it is re-asserted here so
            # a hand-edited save cannot turn the undead into something that has
            # to eat before it can be fought. Same for the kinds that never feed
            # at all: an unsated flyover is unreachable anyway, but a
            # never-sated skeletal would be immortal and would never starve out
            # either, since the starve clock only runs for kinds that eat.
            if g.kind == KIND_SKELETAL:
                g.sated = True
            if g.fed_count > 0:
                g.sated = True
            # A gorging dragon has eaten by definition - that is the only door
            # into the phase. A hand-edited save that says otherwise would put
            # an INVULNERABLE dragon inside the one window the whole feeding
            # rule exists to open, which is the rule exactly inverted.
            if g.state == PHASE_GORGE:
                g.sated = True
            # A save caught mid-seize whose victim no longer exists holds a
            # phase that can never complete.
            if g.state in (PHASE_SEIZE, PHASE_STOOP) and g.target_id is None:
                g.state = PHASE_HUNT
                g.t = 0.0
        except Exception:
            log.warning("dragon record unreadable; dropping it", exc_info=True)
            return cls(health=0.0, max_health=1.0)
        return g


# ========================================================== DragonRegistry ==
class DragonRegistry:
    """Every dragon in the world, plus the gate and the timer that bring them."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed: int = random.randrange(1 << 30) if seed is None else _i(seed, 0)
        self._rng: random.Random = random.Random(self.seed)
        #: Values taken off ``_rng`` so far. Replayed by from_dict - see the
        #: module docstring for why this is not ufo.py's restart.
        self.draws: int = 0

        self.dragons: list[Dragon] = []
        self.next_id: int = 1
        self.t: float = 0.0

        #: The gate, latched. Once the colony has stood two stone huts it has
        #: been noticed, and a fire that ruins one does not un-notice it - the
        #: same reasoning that makes ``hut_tier_unlocked`` a latch.
        self.seen_gate: bool = False
        #: Seconds until the next visit. Meaningless until the gate latches.
        self.next_in: float = DRAGON_FIRST_DELAY
        #: Whether the guaranteed first-visit flyover has been spent.
        self.herald_done: bool = False
        self.visits: int = 0
        #: A forced kind from :meth:`summon`, consumed by the next spawn.
        self._forced_kind: str | None = None
        #: Set by summon() so the debug hook works on a world that has not
        #: earned the gate - and only by summon().
        self._forced: bool = False

        #: ``(kind, x)`` for every dragon that died since somebody last looked.
        #: TRANSIENT: deliberately absent from to_dict and from_dict.
        #:
        #: This is the same idiom the module already uses for
        #: ``Dragon._refused_pending`` - raise a flag and let somebody else turn
        #: it into an event - and it is here for the same reason. ``_reap`` has
        #: no business knowing what loot is, what the drop table says, or that
        #: relics exist at all; this module still imports nothing from items.py.
        #: ``RelicRegistry.drain_slain`` reads it duck-typed and clears it.
        #:
        #: Transient because ``World.tick`` runs ``_guarded("relics", ...)`` in
        #: the very next line after ``_guarded("dragons", ...)``, so the list is
        #: drained in the same frame it is raised and cannot cross a save. If it
        #: were persisted, a save taken between the raise and the drain would
        #: carry a phantom drop that spawns a second relic on load.
        #:
        #: The trim in ``_reap`` is the bound that makes that safe to be wrong
        #: about: if "relics" is ever ``_guarded``-disabled, or if anyone
        #: reorders those two guards, this list must not grow without limit for
        #: the rest of an unattended overnight run.
        self.slain: list[tuple[str, float]] = []

    # ---------------------------------------------------------- container --
    def __len__(self) -> int:
        return len(self.dragons)

    def __iter__(self) -> Iterator[Dragon]:
        return iter(list(self.dragons))

    def all(self) -> list[Dragon]:
        return list(self.dragons)

    def alive(self) -> list[Dragon]:
        """Same name and same shape as ``AnimalRegistry.alive`` - that is the
        point, so ``_spiked_targets`` can concatenate the two without a
        special case."""
        return [d for d in self.dragons if d.alive]

    def grounded(self, world: Any,
                 ceiling: float = BARRICADE_SPIKE_H) -> list[Dragon]:
        """Living dragons within *ceiling* px of the ground under them.

        The convenience wrapper the barricade uses. Defaults to the spike height
        because that is the only caller that has ever wanted it.
        """
        out: list[Dragon] = []
        limit = _f(ceiling, BARRICADE_SPIKE_H)
        for d in self.dragons:
            if not d.alive:
                continue
            try:
                if clearance(world, d) <= limit:
                    out.append(d)
            except Exception:
                continue
        return out

    def count(self) -> int:
        return sum(1 for d in self.dragons if d.alive)

    def get(self, did: int | None) -> Dragon | None:
        if did is None:
            return None
        wanted = _i(did, -1)
        for d in self.dragons:
            if d.id == wanted:
                return d
        return None

    def nearest(self, x: float, *, hostile_only: bool = False) -> Dragon | None:
        best: Dragon | None = None
        best_d = float("inf")
        px = _f(x, 0.0)
        for d in self.dragons:
            if not d.alive or (hostile_only and not d.hostile):
                continue
            gap = abs(d.x - px)
            if gap < best_d:
                best, best_d = d, gap
        return best

    def add(self, dragon: Dragon) -> Dragon:
        if not isinstance(dragon, Dragon):
            return dragon
        if dragon.id <= 0 or any(d.id == dragon.id for d in self.dragons):
            dragon.id = self.next_id
        self.next_id = max(self.next_id, dragon.id + 1)
        self.dragons.append(dragon)
        return dragon

    def remove(self, dragon: Dragon | int) -> bool:
        target = dragon if isinstance(dragon, Dragon) else self.get(_i(dragon, -1))
        if target is None:
            return False
        try:
            self.dragons.remove(target)
        except ValueError:
            return False
        return True

    def clear(self) -> int:
        n = len(self.dragons)
        self.dragons.clear()
        return n

    # ------------------------------------------------------------- random --
    # Every draw in this module goes through these two, and every one of them
    # bumps `draws`. Never world.pyrng (that is the stream wolves spawn from -
    # see the module docstring for the 24-seed measurement), never the bare
    # random module, and never hash() of a string: CPython salts str hashing per
    # process, so a seed derived from one is reproducible within a run and
    # different on the next launch. zlib.crc32 where a string has to become a
    # number.
    def _rand(self) -> float:
        self.draws += 1
        return self._rng.random()

    def _uni(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self._rand()

    def _interval(self) -> float:
        lo = min(DRAGON_INTERVAL_MIN, DRAGON_INTERVAL_MAX)
        hi = max(DRAGON_INTERVAL_MIN, DRAGON_INTERVAL_MAX)
        return self._uni(lo, hi)

    # --------------------------------------------------------------- gate --
    def _stone_huts_standing(self, world: Any) -> int:
        """Built, unruined stone huts. Never raises.

        World is expected to own this latch (``World.dragons_unlocked``, set in
        ``_init_colony_state`` and persisted), and :meth:`_tick_gate` prefers it
        when it is there. This is the fallback that keeps the feature whole on a
        world that has only been handed the registry - a stub, a test harness,
        or a build where the world-side latch has not landed yet.
        """
        n = 0
        for st in _structures(world):
            try:
                if str(getattr(st, "kind", "")) != "hut":
                    continue
                if not bool(getattr(st, "built", False)):
                    continue
                if bool(getattr(st, "is_ruined", False)):
                    continue
                mat = getattr(st, "material", None)
                if callable(mat):
                    material = str(mat() or "")
                else:
                    state = getattr(st, "state", None)
                    material = str((state or {}).get("material") or "")
                if material.strip().lower() == "stone":
                    n += 1
            except Exception:
                continue
        return n

    def _tick_gate(self, world: Any) -> None:
        if self.seen_gate:
            return
        opened = False
        flag = getattr(world, "dragons_unlocked", None)
        if isinstance(flag, bool):
            opened = flag
        if not opened:
            opened = self._stone_huts_standing(world) >= DRAGON_GATE_STONE_HUTS
        if not opened:
            return
        self.seen_gate = True
        if self._forced:
            # A summon already in hand outranks the schedule. _tick_gate runs
            # BEFORE _tick_schedule, so without this a summon issued on the same
            # tick the colony finished its second stone hut was silently pushed
            # eight minutes into the future - the tray command and the debug
            # hook both did nothing, exactly once, on the one tick it mattered.
            return
        # DRAGON_FIRST_DELAY of quiet after the latch. The gate says the colony
        # is ready; the delay is what stops the world announcing it in the same
        # breath, so the herald reads as a thing that happened rather than as a
        # reward popping.
        self.next_in = DRAGON_FIRST_DELAY

    # ----------------------------------------------------------- schedule --
    def summon(self, kind: str | None = None, delay: float = 0.0) -> None:
        """Bring the next visit forward, optionally of a named kind.

        Debug hook and tray command, and the thing the smoke block drives each
        machine with. Deliberately bypasses the gate: a forced summon on a
        colony that has not earned dragons is exactly what you want when you are
        testing one.
        """
        self._forced = True
        self._forced_kind = kind if kind in KINDS else None
        self.next_in = max(0.0, _f(delay, 0.0))

    def _tick_schedule(self, world: Any, dt: float) -> None:
        if not (self.seen_gate or self._forced):
            return
        if self.count() >= MAX_DRAGONS_ALIVE:
            # The clock does not run while one is already here. Two dragons is
            # not twice the event, and an interval that expired during a visit
            # would stack a second one on the instant the first left.
            return
        self.next_in -= dt
        if self.next_in > 0.0:
            return
        kind = self._pick_kind(world)
        if kind is None or self._spawn(world, kind) is None:
            # Nothing spawnable this cycle (an empty colony, a skeletal with no
            # graves left). Try again well before a full interval, the way
            # Ufo._tick_idle does, rather than skipping forty minutes.
            self.next_in = self._interval() * 0.25
            return
        self.visits += 1
        self.next_in = self._interval()
        self._forced = False
        self._forced_kind = None

    def _pick_kind(self, world: Any) -> str | None:
        """The weighted roll, off this registry's own stream.

        Two exceptions, both deliberate. The FIRST dragon a colony ever sees is
        always the flyover: it is the herald, and it is what makes the gate
        legible - the player watches the tech unlock cross the sky instead of
        reading about it. And skeletal is skipped on a colony with no graves,
        because a colony that has buried nobody has nothing it wants.
        """
        if self._forced_kind in KINDS:
            if self._forced_kind == KIND_SKELETAL and not _graves(world):
                return None
            return self._forced_kind
        if not self.herald_done:
            return KIND_FLYOVER
        weights = dict(_KIND_WEIGHTS)
        stones = len(_graves(world))
        if not stones:
            weights.pop(KIND_SKELETAL, None)
        else:
            # The only weight in this table that moves - see the constants for
            # what was measured and why raising a flat number would have been
            # the wrong answer.
            weights[KIND_SKELETAL] = min(
                SKELETAL_WEIGHT_MAX,
                _KIND_WEIGHTS[KIND_SKELETAL]
                + SKELETAL_WEIGHT_PER_GRAVE * float(stones))
        total = sum(w for w in weights.values() if w > 0.0)
        if total <= 0.0:
            return KIND_WYVERN
        roll = self._rand() * total
        for k, w in weights.items():
            if w <= 0.0:
                continue
            roll -= w
            if roll <= 0.0:
                return k
        return KIND_WYVERN

    # ------------------------------------------------------------- spawn --
    def _colony_x(self, world: Any) -> float:
        crowd = _agents(world)
        if crowd:
            return _clamp_x(sum(_f(getattr(a, "x", 0.0)) for a in crowd) / len(crowd))
        built = [st for st in _structures(world) if getattr(st, "built", False)]
        if built:
            return _clamp_x(sum(_f(getattr(st, "x", 0.0)) for st in built) / len(built))
        return WORLD_W * 0.5

    def _lair_x(self, world: Any) -> float:
        """Where the serpent lives: the chasm's midpoint, or the deepest column
        - both now searched WITHIN SIGHT OF THE COLONY rather than map-wide.

        This is the one place in this file where widening the world silently
        deleted an event rather than just stretching it. The serpent does not
        travel: ``_spawn`` puts it at ``home_x`` and ``_tick_surface`` pins it
        there while it rises. On a 6400 px map the deepest column, or a chasm,
        is a mean of ~1600 px from the colony and can be 3000 - so the serpent
        would surface, hunt, starve and submerge without ever entering frame,
        and the only trace of the whole visit would be two chronicle lines.
        Searching the stage window keeps the promise the docstring below makes.

        Cost fell with it: the scan is 1600 px of heightmap again, not 6400.

        Only 21.2% of maps are chasm-cut (measured, 85 of 400 seeds), so the
        signature entrance - out of the hole the colony has spent the whole run
        trying to bridge - is the minority case and the fallback is what keeps
        the serpent on every map. Relief is 240-579 px on every map measured, so
        a low place always exists.

        Deliberately reads the RAW heightmap for the deepest column while
        ``ground_y`` (used everywhere else) reads the EFFECTIVE surface. That is
        the consequence the brief asked for: bridge the chasm and the serpent
        comes up onto your own decking, because the deck is what ``ground_y``
        now reports there. The bridge is a road it did not have to earn.
        """
        terrain = getattr(world, "terrain", None)
        centre = self._colony_x(world)
        lo = _clamp(centre - STAGE_HALF, _EDGE_MIN, _EDGE_MAX)
        hi = _clamp(centre + STAGE_HALF, _EDGE_MIN, _EDGE_MAX)
        try:
            ch = getattr(terrain, "chasm", None)
            if isinstance(ch, (tuple, list)) and len(ch) >= 2:
                mid = (float(ch[0]) + float(ch[1])) * 0.5
                # A chasm the colony cannot see is not the chasm the colony has
                # spent the run bridging, so it does not get the signature
                # entrance - the deepest visible column below does instead.
                if lo <= mid <= hi:
                    return _clamp_x(mid)
        except Exception:
            pass
        try:
            h = getattr(terrain, "height", None)
            n = len(h)                                   # type: ignore[arg-type]
            if n > 8:
                # y grows downward, so the DEEPEST column is the largest value.
                # Sampled every 4 px: this runs once per serpent visit and the
                # extra precision would buy nothing a 156 px animal can tell.
                step = 4
                i0 = max(0, min(n - 1, int(lo)))
                i1 = max(i0 + 1, min(n, int(hi) + 1))
                best_i, best_v = i0, -1e30
                for i in range(i0, i1, step):
                    v = float(h[i])
                    if v > best_v:
                        best_i, best_v = i, v
                return _clamp_x(best_i)
        except Exception:
            pass
        return WORLD_W * 0.5

    def _graveyard_x(self, world: Any) -> float | None:
        stones = _graves(world)
        if not stones:
            return None
        try:
            return _clamp_x(sum(_f(getattr(g, "x", 0.0)) for g in stones) / len(stones))
        except Exception:
            return None

    def _spawn(self, world: Any, kind: str) -> Dragon | None:
        """Put one dragon in the world. None if this kind has nothing to do."""
        try:
            max_health, _speed, cruise = _stats_for(kind)
            d = Dragon(kind=kind, health=max_health, max_health=max_health,
                       anim_t=6.0 * self._rand())
            # ONE FIELD. This is the entire implementation of "the undead do not
            # eat": hurt() has no special case and therefore cannot drift.
            if kind == KIND_SKELETAL:
                d.sated = True

            if kind == KIND_SERPENT:
                d.home_x = self._lair_x(world)
                d.x = d.home_x
                d.alt = -SERPENT_DEPTH
                d.state = PHASE_SURFACE
                d.facing = 1
            elif kind == KIND_SKELETAL:
                yard = self._graveyard_x(world)
                if yard is None:
                    return None            # nothing here it wants
                d.home_x = yard
                side = -1 if self._rand() < 0.5 else 1
                d.x = _spawn_x(d.home_x, side)
                d.alt = SKELETAL_SPAWN_ALT
                d.state = PHASE_ARRIVE
                d.facing = 1 if side < 0 else -1
            elif kind == KIND_FLYOVER:
                # home_x FIRST, unlike the pre-camera version: it is now what
                # the crossing is measured from, not just a field nothing reads.
                d.home_x = self._colony_x(world)
                side = -1 if self._rand() < 0.5 else 1
                d.x = _spawn_x(d.home_x, side)
                # NOT drawn here. _tick_cross authors alt against the sky line
                # every tick (see there), and the band it uses comes off `id`,
                # which add() has not handed out yet. The bottom of the band is
                # the safe seed for the one frame between here and that tick.
                d.alt = FLYOVER_ALT[0]
                speed = self._uni(*FLYOVER_SPEED)
                d.vx = speed if side < 0 else -speed
                d.facing = 1 if d.vx >= 0.0 else -1
                d.state = PHASE_CROSS
            elif kind == KIND_QUADRUPED:
                d.home_x = self._colony_x(world)
                # WAS "the nearer edge of the map", and the reason was that a
                # far-side spawn spent 40 s of a 150 s siege trudging across an
                # empty map before it reached anything to wreck. On a 6400 px
                # world that argument gets four times stronger and the mechanism
                # stops working: BOTH world edges can be 3000 px away, so there
                # is no near one and the siege never starts.
                #
                # _spawn_x makes the question moot instead of answering it - the
                # walk is OFFSTAGE + SPAWN_MARGIN whichever side it picks, so
                # both are "near" and the 40 s finding can never come back. What
                # is left to choose is the side with room to be off-camera in:
                # against the map's rim _spawn_x has nowhere to put it and the
                # airborne leg would happen in frame, which is the one thing
                # _tick_arrive exists to prevent. So: away from the nearer rim.
                side = 1 if d.home_x <= WORLD_W * 0.5 else -1
                d.x = _spawn_x(d.home_x, side)
                d.alt = self._uni(*QUAD_ALT)
                d.state = PHASE_ARRIVE
                d.facing = 1 if side < 0 else -1
            else:                                        # wyvern
                d.home_x = self._colony_x(world)
                side = -1 if self._rand() < 0.5 else 1
                d.x = _spawn_x(d.home_x, side)
                d.alt = WYVERN_SPAWN_ALT
                d.state = PHASE_ARRIVE
                d.facing = 1 if side < 0 else -1

            d.y = _ground_y(world, d.x) - d.alt
            self.add(d)
            if kind == KIND_FLYOVER:
                self.herald_done = True
                self._flyover_toll(world)
            _log(world, _ARRIVAL_LINES.get(kind, "Something is out there."))
            return d
        except Exception:
            log.exception("dragon spawn failed for kind %r", kind)
            return None

    def _flyover_toll(self, world: Any) -> None:
        """One morale nudge across the CONTENT, once, on entry.

        The flyover's entire cost, and see FLYOVER_MORALE_FLOOR for what it used
        to cost by accident. Two rules, both of them the same rule:

        * an agent already at or below the floor is not touched at all;
        * an agent above it is never taken below it.

        Together those mean the nudge cannot move any morale-gated decision in
        the game, because every one of those gates is below the floor. The
        colony mean still visibly dips - the content are momentarily less
        content - which is what a shape crossing the sky is for.
        """
        for a in _agents(world):
            try:
                cur = _f(getattr(a, "morale", 0.5), 0.5)
                if cur <= FLYOVER_MORALE_FLOOR:
                    continue
                a.morale = _clamp(max(FLYOVER_MORALE_FLOOR,
                                      cur - FLYOVER_MORALE), 0.0, 1.0)
            except Exception:
                continue

    def _sky_line(self, world: Any, band: float, anchor: float) -> float:
        """World y a flyover holds for its whole crossing, *band* px up.

        ``alt`` is px above the ground DIRECTLY BENEATH, so a constant alt over
        broken terrain is not a constant height. Relief runs 400-450 px on the
        maps measured (seeds 5/11/21/31/48: peaks at y 407-522, floors at
        852-956), so a shape holding alt 360 crossed the frame 450 px lower over
        a valley than over the ridge beside it - sagging into every low place
        and rearing over every high one. Nothing could shoot it either way
        (``hostile`` is False for a flyover, so the throw chooser never even
        considers it), but "high up and out of reach" has to LOOK like one
        straight line or it is not saying anything.

        Two clamps, and the second one is the interesting one:

        * the reference is the highest ground the CROSSING PASSES OVER, not the
          ground under the animal, so *band* means "px above the tallest thing
          here" - a strictly stronger claim than the one FLYOVER_ALT was
          written for;
        * ...and then the line is held at least ``FLYOVER_SCREEN_MIN_Y`` inside
          the frame, because on the hilliest of those maps a peak at y 407 minus
          a 412 px band is y -5. The herald is the sentence that makes the whole
          tech unlock legible; a herald drawn above the top of the screen is the
          unlock going back to being a line in a log.

        "The crossing" used to be the whole map, because the map was one screen.
        On a WORLD_W map it is the stage around *anchor* - the span the flyover
        actually crosses and the only span anyone can see it against. Taking the
        peak of all 6400 columns would hold the herald above a mountain 3000 px
        away, which reads as a shape drifting pointlessly high over flat ground,
        and it would scan four times the array on every tick of the crossing.

        Costs no persisted state - the heightmap is already the world's - and
        runs only while a flyover is actually on the map.
        """
        peak = None
        try:
            h = getattr(getattr(world, "terrain", None), "height", None)
            if h is not None and len(h):
                n = len(h)
                i0 = int(_clamp(anchor - OFFSTAGE, 0.0, float(n - 1)))
                i1 = int(_clamp(anchor + OFFSTAGE, 0.0, float(n - 1))) + 1
                span = h[i0:i1] if i1 > i0 else h
                # y grows downward, so the HIGHEST ground is the smallest value.
                # Every column, not a sample: a 16 px stride missed the true
                # peak by up to 40 px, and since the clamp below is what catches
                # that, the miss came back as exactly the sag this is here to
                # remove.
                peak = _f(min(float(v) for v in span), RENDER_H * 0.5)
        except Exception:
            peak = None
        if peak is None or not math.isfinite(peak):
            peak = RENDER_H * 0.45
        return max(FLYOVER_SCREEN_MIN_Y, peak - max(0.0, _f(band, 0.0)))

    # --------------------------------------------------------------- tick --
    def tick(self, world: Any, dt: float) -> None:
        """Advance the gate, the timer and every dragon. Never raises."""
        try:
            step = _clamp(_f(dt, 0.0), 0.0, _MAX_DT)
            if step <= 0.0:
                return
            self.t += step
            self._tick_gate(world)
            # Reap at the TOP of the tick, not the bottom. A dragon that dies
            # or reaches PHASE_IDLE is then visible in that state for the rest
            # of the frame - to render, to the tray, and to anything measuring
            # the machine - and is cleared on the next one. Reaping at the
            # bottom made IDLE a state no observer could ever witness, which is
            # a state machine with a phase that does not exist. Animal.hurt's
            # docstring says the same thing about carcasses for the same reason.
            self._reap(world)
            self._tick_schedule(world, step)
            for d in list(self.dragons):
                try:
                    self._tick_dragon(world, d, step)
                except Exception:
                    log.exception("dragon %s failed in %r; sending it home",
                                  d.id, d.state)
                    self._release_victim(world, d)
                    d.state = PHASE_IDLE
        except Exception:
            log.exception("dragon tick failed")

    def _tick_dragon(self, world: Any, d: Dragon, dt: float) -> None:
        d.age += dt
        d.t += dt
        d.anim_t += dt * 1.6
        if d.anim_t > 1.0e6:
            d.anim_t = math.fmod(d.anim_t, 1.0e6)
        if d.refused_t >= 0.0:
            d.refused_t += dt

        self._tell_unsated(world, d)

        if not d.alive or d.state == PHASE_IDLE:
            return                                       # reaped below

        self._tick_starve(world, d, dt)

        # Patience runs out whatever it was doing - animals.py's rule, and the
        # only thing that gets an unfed dragon off a colony that will not come
        # outside. SEIZE is exempt so a lift already in progress finishes rather
        # than dropping somebody a tenth of a second short, and GORGE is exempt
        # because it IS the payoff: a wyvern that spent 170 of its 180 patient
        # seconds hunting and then ate somebody must still be killable for the
        # full window, or the reward for a long defence is no reward at all.
        # GORGE_MAX_SEC is what stops that exemption becoming a hang.
        if d.state not in (PHASE_LEAVE, PHASE_IDLE, PHASE_SEIZE, PHASE_GORGE) \
                and d.age >= _VISIT_MAX.get(d.kind, 180.0):
            self._enter(d, PHASE_LEAVE)

        handler = {
            PHASE_CROSS: self._tick_cross,
            PHASE_ARRIVE: self._tick_arrive,
            PHASE_SURFACE: self._tick_surface,
            PHASE_HUNT: self._tick_hunt,
            PHASE_STOOP: self._tick_stoop,
            PHASE_SEIZE: self._tick_seize,
            PHASE_GORGE: self._tick_gorge,
            PHASE_ROOST: self._tick_roost,
            PHASE_RAZE: self._tick_raze,
            PHASE_SCOUR: self._tick_scour,
            PHASE_LEAVE: self._tick_leave,
        }.get(d.state)
        if handler is None:
            self._enter(d, PHASE_LEAVE)
            return
        handler(world, d, dt)

        # THE ALTITUDE RULE, in one line, applied after every handler: alt is
        # authored, y is derived, and y is what render reads. Doing this here
        # rather than in each handler is what makes it impossible for a new
        # phase to forget it.
        d.y = _ground_y(world, d.x) - d.alt

    def _enter(self, d: Dragon, state: str) -> None:
        if d.state != state:
            d.state = state
            d.t = 0.0

    @staticmethod
    def _cruise(d: Dragon) -> float:
        """The working altitude this visit holds, per kind and per dragon."""
        if d.kind == KIND_WYVERN:
            return _band(d, WYVERN_ALT)
        if d.kind == KIND_SKELETAL:
            return _band(d, SKELETAL_ALT)
        return d.cruise_alt

    # -- the feeding rule's two chronicle lines ---------------------------
    def _tell_unsated(self, world: Any, d: Dragon) -> None:
        """Signal 1: one sentence, once per dragon, the first time it refuses.

        This is the whole legibility answer for requirement 4 - the rule is
        stated in English in the log the player already reads. ``told_unsated``
        is what turns six thrown spears into one line instead of six.
        """
        if not d._refused_pending:
            return
        d._refused_pending = False
        if d.told_unsated:
            return
        d.told_unsated = True
        _log(world, f"The spear glances off the {d.noun}. It has not fed.")

    def _tick_starve(self, world: Any, d: Dragon, dt: float) -> None:
        """The deadlock hatch, and the drama that falls out of it.

        Only kinds that CAN feed run this clock. flyover never feeds and is
        unreachable; skeletal is sated at birth and never wants a person, so a
        starve line for either would be a sentence about nothing.
        """
        if d.sated or not d.eats:
            d.starve_t = 0.0
            return
        if self._pick_victim(world, d) is not None:
            d.starve_t = 0.0
            return
        d.starve_t += dt
        if d.starve_t < DRAGON_STARVE_SEC:
            return
        d.sated = True
        d.starve_t = 0.0
        _log(world, f"Nothing came out. Hunger has made the {d.noun} careless.")

    # -- eligibility ------------------------------------------------------
    def _pick_victim(self, world: Any, d: Dragon) -> Any:
        """The nearest colonist this dragon is allowed to take, or None.

        Three refusals, all mandatory (see the module docstring):

        * never the last one - ``MIN_LEFT_BEHIND``, lifted from ufo.py;
        * never somebody indoors, already taken, or dead - so hiding in the
          huts is a legal and complete defence;
        * never at all once the per-visit cap is spent.
        """
        if not d.eats or d.fed_count >= d.max_fed:
            return None
        crowd = [a for a in _agents(world) if not getattr(a, "taken", False)]
        if len(crowd) <= MIN_LEFT_BEHIND:
            return None
        best, best_gap = None, float("inf")
        for a in crowd:
            if getattr(a, "inside", None) is not None:
                continue
            gap = abs(_f(getattr(a, "x", 0.0)) - d.x)
            if gap < best_gap:
                best, best_gap = a, gap
        return best

    def _held(self, world: Any, d: Dragon) -> Any:
        """The victim this dragon has hold of, if they are still takeable."""
        a = _agent_by_id(world, d.target_id)
        if a is None or not getattr(a, "alive", False):
            return None
        if getattr(a, "taken", False) or getattr(a, "inside", None) is not None:
            return None
        return a

    # -- phases -----------------------------------------------------------
    def _tick_cross(self, world: Any, d: Dragon, dt: float) -> None:
        """flyover. A constant vx and a constant SCREEN HEIGHT, edge to edge.

        No target, no descent, no interaction of any kind. Its whole existence
        is the arrival line and the morale nudge, both already spent at spawn.

        The height is held against :meth:`_sky_line` - the highest ground on the
        stage it crosses - rather than against the ground directly beneath, so
        it draws one straight line across the frame instead of sagging into
        every valley. It
        is the one kind whose ``alt`` is re-derived rather than authored, which
        is the honest way round for a thing whose whole job is to be
        unmistakably out of reach: ``alt`` then reports the real, larger gap
        over low ground instead of understating it.
        """
        d.x += d.vx * dt
        d.facing = 1 if d.vx >= 0.0 else -1
        band = _band(d, FLYOVER_ALT)
        d.alt = _ground_y(world, d.x) - self._sky_line(world, band, d.home_x)
        # Crossed, when it is a spawn-margin past the far stage edge. Measured
        # against home_x rather than the map, or a herald that spawned near the
        # colony would have to fly up to 5000 px of empty world before it was
        # allowed to stop existing - two full minutes at FLYOVER_SPEED, holding
        # a MAX_DRAGONS_ALIVE slot and blocking every later visit.
        if abs(d.x - d.home_x) > OFFSTAGE + SPAWN_MARGIN + 60.0:
            self._enter(d, PHASE_IDLE)

    def _tick_arrive(self, world: Any, d: Dragon, dt: float) -> None:
        """Fly in from off-screen, easing down to the working altitude.

        The quadruped is the special case and it is a DELIBERATE one.
        ``render/dragons.py``'s quadruped module has no flight pose at all - t
        drives a grounded mantle cycle - so there is nothing correct to draw for
        a quadruped in the air. Rather than accept a visibly wrong pose, its
        entire airborne leg is spent OUTSIDE the frame: it descends at whatever
        rate reaches alt 0 exactly as it crosses the screen edge, and everything
        the player ever sees of it is on the ground. That costs one design
        decision and no drawing. It is also what animals already do - they walk
        in over the edge - so it is not a new idea in this codebase.
        """
        target_x = _clamp(d.home_x, ENTRY_INSET, _EDGE_MAX - ENTRY_INSET)
        d.facing = 1 if target_x >= d.x else -1

        if d.kind == KIND_QUADRUPED:
            on_screen = _on_stage(d.home_x, d.x)
            if d.alt > 0.0 or not on_screen:
                # Still airborne, therefore still off-screen. Distance left to
                # the stage edge is the time left to be in the air; the descent
                # rate is recomputed every tick so it cannot overshoot however
                # SPAWN_MARGIN or the speeds are retuned.
                d.x += (d.speed if d.facing > 0 else -d.speed) * dt
                edge = _stage_edge(d.home_x, -1 if d.facing > 0 else 1)
                gap = abs(edge - d.x)
                t_left = gap / max(1e-3, d.speed)
                if t_left <= dt or gap <= 4.0 or _on_stage(d.home_x, d.x):
                    d.alt = 0.0
                else:
                    d.alt = max(0.0, d.alt - (d.alt / t_left) * dt)
                return
            # Down, and walking in. Still ARRIVE rather than ROOST because it
            # has not got anywhere yet - and it strides at its own speed rather
            # than the roost's 18 px/s, which is a working pace, not a
            # travelling one.
            self._walk(world, d, d.home_x, dt, QUAD_WALK * 3.0)
            if abs(d.home_x - d.x) <= ROOST_RANGE * 0.5:
                self._enter(d, PHASE_ROOST)
            return

        d.x += (d.speed if d.facing > 0 else -d.speed) * dt
        cruise = self._cruise(d)
        d.alt = _approach(d.alt, cruise, ALT_RATE, dt)

        inside = _on_stage(d.home_x, d.x, ENTRY_INSET)
        if inside and abs(d.alt - cruise) <= 6.0:
            self._enter(d, PHASE_HUNT)

    def _tick_surface(self, world: Any, d: Dragon, dt: float) -> None:
        """serpent. Up out of the lair. ``alt`` is NEGATIVE for this whole leg,
        which is exactly the case clearance() is specified not to clamp."""
        d.x = _clamp_x(d.home_x)
        d.alt = min(SERPENT_CREST, d.alt + SERPENT_RISE * dt)
        if d.alt >= SERPENT_CREST - 0.5:
            d.alt = SERPENT_CREST
            self._enter(d, PHASE_HUNT)

    def _tick_hunt(self, world: Any, d: Dragon, dt: float) -> None:
        """Track what this kind came for, at its working altitude."""
        if d.kind == KIND_SKELETAL:
            # It wants the graveyard, not a person.
            yard = self._graveyard_x(world)
            if yard is None:
                self._enter(d, PHASE_LEAVE)
                return
            d.home_x = yard
            d.alt = _approach(d.alt, self._cruise(d), ALT_RATE, dt)
            gap = d.home_x - d.x
            d.facing = 1 if gap >= 0.0 else -1
            d.x += _clamp(gap, -d.speed * dt, d.speed * dt)
            if abs(gap) <= CIRCLE_R:
                self._enter(d, PHASE_SCOUR)
            return

        victim = self._pick_victim(world, d)
        if victim is None:
            if d.fed_count >= d.max_fed:
                self._enter(d, PHASE_LEAVE)      # it got what it came for
                return
            # Nobody reachable. Loiter over the colony while the starve clock
            # runs. That clock is what makes it mortal; _VISIT_MAX is what
            # eventually sends it home.
            self._drift(world, d, d.home_x, dt, d.speed * 0.55)
            return

        vx_ = _f(getattr(victim, "x", d.x), d.x)
        if d.kind == KIND_SERPENT:
            # Undulating along the surface, welded to the ground the way an
            # animal is - it has no wings and must never be posed as flight.
            d.alt = SERPENT_CREST + SERPENT_SWELL * math.sin(d.anim_t * 1.3)
            self._walk(world, d, vx_, dt, d.speed)
            if abs(vx_ - d.x) <= SEIZE_REACH:
                d.target_id = _i(getattr(victim, "id", 0), 0)
                self._enter(d, PHASE_SEIZE)
            return

        # wyvern: hold the cruise and shadow their x until it is worth the dive.
        d.alt = _approach(d.alt, self._cruise(d), ALT_RATE, dt)
        self._drift(world, d, vx_, dt, d.speed)
        if abs(vx_ - d.x) <= STOOP_TRIGGER:
            d.target_id = _i(getattr(victim, "id", 0), 0)
            self._enter(d, PHASE_STOOP)

    def _tick_stoop(self, world: Any, d: Dragon, dt: float) -> None:
        """wyvern only. The dive: cruise down to SEIZE_ALT over STOOP_SEC while
        closing on where the victim is going to be, not where they are."""
        victim = self._held(world, d)
        if victim is None:
            d.target_id = None
            self._enter(d, PHASE_HUNT)
            return

        lead = _f(getattr(victim, "x", d.x), d.x) + \
            _f(getattr(victim, "vx", 0.0)) * STOOP_LEAD_SEC
        self._drift(world, d, lead, dt, d.speed * 1.4)

        floor = _band(d, SEIZE_ALT)
        rate = max(1.0, (d.alt - floor) / max(1e-3, STOOP_SEC - d.t)) \
            if d.t < STOOP_SEC else ALT_RATE * 3.0
        d.alt = max(floor, d.alt - rate * dt)
        if d.alt <= SEIZE_ALT[1] and abs(lead - d.x) <= SEIZE_REACH:
            d.alt = floor
            self._enter(d, PHASE_SEIZE)
        elif d.t >= STOOP_SEC * 2.5:
            # It missed. Climb back and try again rather than skimming the dirt.
            d.target_id = None
            self._enter(d, PHASE_HUNT)

    def _tick_seize(self, world: Any, d: Dragon, dt: float) -> None:
        """Holding a colonist. The one window in which a wyvern can be hit.

        The hold is ``Ufo._hold``'s, minus the beam: action and action_raw
        cleared, target_x and path_goal cleared, vx zeroed, x re-pinned under the
        dragon and pose_override set. Re-pinning x every tick is what makes it
        robust regardless of tick order - even if behaviour hands the agent a
        fresh action first, the agent ends the tick back under the dragon.

        ``beamed`` and ``lift_t`` are deliberately NOT set: those are the ufo's
        flags, and ``Ufo._release_strays`` sweeps anyone carrying them once a
        second while it is idle. Borrowing them would have the saucer fighting
        the dragon for the same body.
        """
        victim = self._held(world, d)
        if victim is None:
            # They died first, or went indoors, or the ufo took them. Back to
            # hunting with nothing latched: an aborted seize is not a meal.
            d.target_id = None
            self._enter(d, PHASE_HUNT)
            return

        d.alt = _band(d, SEIZE_ALT) if d.kind == KIND_WYVERN else \
            SERPENT_CREST + SERPENT_SWELL * 0.4 * math.sin(d.anim_t * 2.2)
        self._hold(d, victim)

        if d.t >= DRAGON_SEIZE_SEC:
            fed = d.fed_count
            self._devour(world, d, victim)
            # THE WINDOW. Straight to LEAVE was the shipped behaviour and it is
            # the reason 511 visits produced 0 kills - see PHASE_GORGE and
            # DRAGON_GORGE_SEC. A seize that killed nobody (armour, a stub
            # agent) earns no gorge: nothing to be heavy with.
            self._enter(d, PHASE_GORGE if d.fed_count > fed else PHASE_LEAVE)

    def _tick_gorge(self, world: Any, d: Dragon, dt: float) -> None:
        """It has eaten, and it is heavy. THE PAYOFF, in two legs.

        Leg one, for ``DRAGON_GORGE_SEC``: down at ``GORGE_ALT`` (26 px), which
        is under MELEE_CEILING, under BARRICADE_SPIKE_H and well under a
        thrower's ceiling - every weapon the colony owns reaches it. It wallows
        sideways at ``GORGE_DRIFT``, a fifth of its cruise, because a dragon
        parked in the air reads as a bug rather than as a full one.

        Leg two: the climb out at ``GORGE_CLIMB_RATE`` - 13 px/s against the 96
        it leaves at - which drags it back up through melee reach, then through
        the spikes, then out of throwing range, one at a time and visibly. At
        ``GORGE_ESCAPE_ALT`` nothing can touch it any more, so PHASE_LEAVE takes
        over and the fast climb is no longer hiding anything.

        The serpent does not fly, so leg two is a crawl home along the surface
        instead of a climb; it stays at its crest, in reach, the whole way. Its
        lair is the safety, not the sky.

        Two things this phase must not do, both learned the hard way:

        * it must not be called "leave" in any spelling - see PHASE_GORGE;
        * it must not be cancellable. ``_tick_dragon`` exempts it from
          ``_VISIT_MAX``, because a payoff that the patience timer can delete is
          not a payoff, and ``GORGE_MAX_SEC`` is what pays for that exemption.
        """
        held = self._held(world, d)                # normally None - it ate them
        if held is not None:
            self._hold(d, held)

        # It wallows AROUND ITS HOME, not off in whatever direction it happened
        # to be facing. A straight drift at GORGE_DRIFT for the full window is
        # 340 px, which is enough to carry a dragon that fed near a frame edge
        # clean off the map and out of everyone's reach - the payoff walking
        # away from the fight it is supposed to be.
        goal = _clamp_x(d.home_x + GORGE_WALLOW * math.sin(d.age * 0.4))
        heavy = d.t < DRAGON_GORGE_SEC
        if d.kind == KIND_SERPENT:
            # Welded to the ground: it has no wings and must never be posed as
            # flight. Its "climb" is the crawl back to the hole, and it stays at
            # its crest - in reach of everything - for the whole way.
            d.alt = SERPENT_CREST + SERPENT_SWELL * math.sin(d.anim_t * 1.3)
            if heavy:
                self._walk(world, d, goal, dt, GORGE_DRIFT)
            else:
                self._walk(world, d, d.home_x, dt, d.speed * 0.45)
                if abs(d.home_x - d.x) <= 6.0:
                    self._enter(d, PHASE_LEAVE)
        elif heavy:
            d.alt = _approach(d.alt, GORGE_ALT, ALT_RATE * 0.5, dt)
            self._drift(world, d, goal, dt, GORGE_DRIFT)
        else:
            d.alt = _approach(d.alt, GORGE_ESCAPE_ALT, GORGE_CLIMB_RATE, dt)
            self._drift(world, d, goal, dt, GORGE_DRIFT)
            if d.alt >= GORGE_ESCAPE_ALT - 0.5:
                self._enter(d, PHASE_LEAVE)

        if d.t >= GORGE_MAX_SEC:
            self._enter(d, PHASE_LEAVE)

    def _tick_roost(self, world: Any, d: Dragon, dt: float) -> None:
        """quadruped. Grounded, walking the surface between what it is wrecking.

        ``alt`` is pinned at 0 and ``y`` is re-derived from ``ground_y(x)``
        every tick, exactly as ``animals._walk`` does it - so BARRICADE_DAMAGE
        applies, melee reaches it and thrown spears reach it. This is the payoff
        for having an altitude at all: for the first time in this game a colony
        that built defences has a trap that does what the player asked it to do.
        """
        d.alt = 0.0
        if d.age >= DRAGON_SIEGE_SEC:
            self._enter(d, PHASE_LEAVE)
            return
        self._maw(world, d)

        st = self._raze_target(world, d)
        if st is not None:
            self._enter(d, PHASE_RAZE)
            return

        # Wander toward the nearest thing worth wrecking, else pace the roost.
        goal = self._nearest_built_x(world, d)
        if goal is None:
            span = ROOST_RANGE * math.sin(d.age * 0.06)
            goal = _clamp_x(d.home_x + span)
        self._walk(world, d, goal, dt, QUAD_WALK)

    def _tick_raze(self, world: Any, d: Dragon, dt: float) -> None:
        """quadruped. Standing over a building, taking it apart."""
        d.alt = 0.0
        if d.age >= DRAGON_SIEGE_SEC:
            self._enter(d, PHASE_LEAVE)
            return
        self._maw(world, d)

        st = self._raze_target(world, d)
        if st is None:
            self._enter(d, PHASE_ROOST)
            return
        try:
            # The EXISTING damage path, so a structure a dragon pulls down
            # collapses, evicts and chronicles exactly like one a fire takes.
            st.damage(DRAGON_RAZE_DPS * dt, "dragon")
        except Exception:
            log.debug("raze failed", exc_info=True)
            self._enter(d, PHASE_ROOST)
        # Standing still, but still welded to the surface: y is re-derived from
        # ground_y(x) at the end of every tick and alt is pinned at 0.
        d.vx = 0.0
        d.x = _clamp_x(d.x)

    def _tick_scour(self, world: Any, d: Dragon, dt: float) -> None:
        """skeletal. Circling the graves and taking the stones.

        It never targets a living agent and never feeds. It leaves after
        DRAGON_SCOUR_MAX headstones or when the graveyard is empty, whichever
        comes first - a colony with three graves loses three, not four.
        """
        stones = _graves(world)
        if not stones or d.fed_count >= DRAGON_SCOUR_MAX:
            self._enter(d, PHASE_LEAVE)
            return
        d.alt = _clamp(d.cruise_alt + 14.0 * math.sin(d.anim_t * 0.7),
                       *SKELETAL_ALT)
        prev = d.x
        d.x = _clamp_x(d.home_x + CIRCLE_R * math.cos(d.age * 0.55))
        d.facing = 1 if d.x >= prev else -1

        if d.t < DRAGON_SCOUR_SEC:
            return
        d.t = 0.0
        # Snapshot taken above; remove from the registry, not from the list we
        # are walking. World._weather_graves is doing its own removals from the
        # same collection on the same tick.
        best, best_gap = None, float("inf")
        for g in stones:
            gap = abs(_f(getattr(g, "x", 0.0)) - d.x)
            if gap < best_gap:
                best, best_gap = g, gap
        if best is None or best_gap > SCOUR_REACH:
            return
        name = ""
        try:
            name = str((getattr(best, "state", None) or {}).get("name") or "")
        except Exception:
            name = ""
        if _remove_prop(world, best):
            # fed_count is the scour counter for this kind. Reusing the field
            # rather than adding a sixth counter keeps to_dict one shape for all
            # five, and "how many times has it taken what it came for" is
            # honestly the same question.
            d.fed_count += 1
            if name:
                _log(world, f"The {d.noun} takes {name}'s stone.")
            else:
                _log(world, f"The {d.noun} takes a headstone.")

    def _tick_leave(self, world: Any, d: Dragon, dt: float) -> None:
        """On its way out. Whatever it did or did not get, it goes.

        A dragon that reaches this phase still unsated simply leaves: no kill,
        no latch, no grave. It is not stuck; it is a visit that cost nothing.
        """
        if d.kind == KIND_SERPENT:
            if abs(d.home_x - d.x) > 6.0:
                d.alt = SERPENT_CREST + SERPENT_SWELL * math.sin(d.anim_t * 1.3)
                self._walk(world, d, d.home_x, dt, d.speed)
                self._drag(world, d)
                return
            d.x = _clamp_x(d.home_x)
            d.alt -= SERPENT_RISE * dt
            self._drag(world, d)
            if d.alt <= -SERPENT_DEPTH:
                self._release_victim(world, d)
                self._enter(d, PHASE_IDLE)
            return

        # Aim PAST the despawn line, not at the stage edge. Aiming at the edge
        # itself made the exit oscillate on the rim forever: one tick outside
        # flips the "which side am I on" test, the next tick flips it back, and
        # the dragon hovered at x=1601 for the rest of the run without ever
        # reaching the despawn threshold at edge + SPAWN_MARGIN. Measured: a
        # wyvern that had already fed and left sat there for 400 s of a 400 s
        # test, so PHASE_IDLE was a state the machine could not reach.
        #
        # "Which side am I on" is now measured against home_x, not the middle of
        # the map. A dragon that fed at x=5200 was "left of centre" and so flew
        # 5200 px WEST to leave, straight back across the colony it had just
        # finished with and then 4000 px of empty land, at 96 px/s: 54 s of a
        # visit whose whole point was over, blocking the next one for all of it.
        side = 1 if d.x >= d.home_x else -1
        going_right = side > 0
        out_x = _spawn_x(d.home_x, side) + side * 40.0
        d.facing = 1 if going_right else -1
        step = (d.speed if going_right else -d.speed) * dt

        if d.kind == KIND_QUADRUPED:
            # Walks out the way it walked in, and only starts climbing once it
            # is off the stage - see _tick_arrive for why the airborne leg has
            # to stay off-screen.
            #
            # THE INSET IS LOAD-BEARING, and leaving it out cost a debugging
            # session: with the test at `<= OFFSTAGE` and the walk target AT
            # OFFSTAGE, the quadruped walked to |x - home| == 960.0 exactly,
            # where the test is still true and _walk's own "gap < 0.5, stop"
            # fires - so it stood there for the remaining 250 s of every run and
            # PHASE_IDLE became unreachable. Measured on seeds 11/23/37, all
            # three identical: stuck at 960.0. It is the same shape as the
            # oscillation the comment above records, and the same cure - the
            # target must be strictly past the test, never equal to it.
            #
            # The world-rim bound is kept and is NOT redundant: _walk() clamps x
            # into [0, WORLD_W-1], so a colony seated within OFFSTAGE of the
            # map's rim would walk to that rim and never get off stage at all.
            # Either test failing hands it to the airborne branch, which is
            # unclamped and therefore always terminates.
            if (_on_stage(d.home_x, d.x, 1.0)
                    and _EDGE_MIN + 1.0 < d.x < _EDGE_MAX - 1.0):
                d.alt = 0.0
                self._walk(world, d, _stage_edge(d.home_x, side),
                           dt, QUAD_WALK * 2.2)
                self._drag(world, d)
                return
            d.x += step
            d.alt = _approach(d.alt, d.cruise_alt, ALT_RATE, dt)
        else:
            d.x += step
            d.alt = _approach(d.alt, d.cruise_alt + 120.0, ALT_RATE, dt)

        self._drag(world, d)
        if (going_right and d.x >= out_x) or (not going_right and d.x <= out_x):
            self._release_victim(world, d)
            self._enter(d, PHASE_IDLE)

    # -- eating ------------------------------------------------------------
    def _maw(self, world: Any, d: Dragon) -> None:
        """Anything that walks within DRAGON_MAW_REACH of a grounded quadruped.

        Capped at DRAGON_MAX_FED["quadruped"] per visit, which is the animals.py
        "one kill each, then they go" rule with the number raised because this
        one STAYS: a colony that throws bodies at it loses three and no more.
        """
        if d.fed_count >= d.max_fed:
            return
        victim = self._pick_victim(world, d)
        if victim is None:
            return
        if abs(_f(getattr(victim, "x", 1e9), 1e9) - d.x) > DRAGON_MAW_REACH:
            return
        self._devour(world, d, victim)

    def _devour(self, world: Any, d: Dragon, victim: Any) -> None:
        """The kill, the latch and the second chronicle line, on one line each.

        The victim DIES - this is not an abduction. It routes through
        ``Stickman.hurt`` with ``death_cause = "devoured"`` so ``_reap_dead``
        raises the grave, ``stats["died"]`` moves, and ``World.reconcile()``'s
        residual stays 0. Copying ufo.py's roster-removal path instead would
        silently corrupt those books, which is precisely the bug that path was
        written to document.

        ``sated`` and ``fed_count`` are booked HERE, on the line the person
        actually dies, mirroring how ``Ufo._complete_abduction`` books the
        roster change where the roster changes.
        """
        name = str(getattr(victim, "name", "Someone"))
        killed = False
        try:
            killed = bool(victim.hurt(MAX_HEALTH * 10.0, CAUSE_DEVOURED))
        except TypeError:
            try:
                killed = bool(victim.hurt(MAX_HEALTH * 10.0))
            except Exception:
                killed = False
        except Exception:
            log.debug("could not devour %s", name, exc_info=True)

        self._release(victim)
        d.target_id = None
        if not killed:
            # Armour, a stub agent, a hook that swallowed it - whatever the
            # reason, nobody died, so nothing is latched and nothing is
            # chronicled. An unfed dragon that thinks it has fed would be
            # killable for free, which is the whole rule inverted.
            return

        d.fed_count += 1
        was_sated = d.sated
        d.sated = True
        _log(world, f"The {d.noun} takes {name}.")
        if not was_sated:
            # Signal 2: the feeding edge, said plainly. The player has already
            # been told once that it could not be hurt; this is the sentence
            # that says the rule has just changed.
            _log(world, f"The {d.noun} has fed. It can be hurt now.")

    @staticmethod
    def _hold(d: Dragon, agent: Any) -> None:
        """Pin an agent under the dragon for this tick. Never raises."""
        try:
            agent.action = None
            agent.action_raw = None
            agent.target_x = None
            agent.path_goal = None
            agent.vx = 0.0
            agent.vy = 0.0
            agent.climbing = False
            agent.climb_t = 0.0
            agent.x = _clamp_x(d.x)
            agent.pose_override = "fall"
        except Exception:
            log.debug("could not hold an agent", exc_info=True)

    @staticmethod
    def _release(agent: Any) -> None:
        try:
            if getattr(agent, "pose_override", None) == "fall":
                agent.pose_override = None
        except Exception:
            pass

    def _release_victim(self, world: Any, d: Dragon) -> None:
        a = _agent_by_id(world, d.target_id)
        if a is not None:
            self._release(a)
        d.target_id = None

    def _drag(self, world: Any, d: Dragon) -> None:
        """Keep hold of a victim through the leave leg, if there still is one."""
        if d.target_id is None:
            return
        a = self._held(world, d)
        if a is None:
            d.target_id = None
            return
        self._hold(d, a)

    # -- razing ------------------------------------------------------------
    def _nearest_built_x(self, world: Any, d: Dragon) -> float | None:
        best, best_gap = None, float("inf")
        for st in _structures(world):
            try:
                if not getattr(st, "built", False) or getattr(st, "is_ruined", False):
                    continue
                gap = abs(_f(getattr(st, "x", 0.0)) - d.x)
                if gap < best_gap:
                    best, best_gap = _f(getattr(st, "x", 0.0)), gap
            except Exception:
                continue
        return best

    def _raze_target(self, world: Any, d: Dragon) -> Any:
        best, best_gap = None, float("inf")
        for st in _structures(world):
            try:
                if not getattr(st, "built", False) or getattr(st, "is_ruined", False):
                    continue
                gap = abs(_f(getattr(st, "x", 0.0)) - d.x)
                if gap <= RAZE_RANGE and gap < best_gap:
                    best, best_gap = st, gap
            except Exception:
                continue
        return best

    # -- movement ----------------------------------------------------------
    def _drift(self, world: Any, d: Dragon, tx: float, dt: float,
               speed: float) -> None:
        """One airborne step toward *tx*. No terrain, no gravity - it is flying."""
        target = _clamp(_f(tx, d.x), -SPAWN_MARGIN, _EDGE_MAX + SPAWN_MARGIN)
        gap = target - d.x
        if abs(gap) < 0.5:
            d.vx = 0.0
            return
        step = min(abs(gap), max(0.0, _f(speed, 0.0)) * dt)
        moved = step if gap > 0.0 else -step
        d.x += moved
        d.vx = moved / max(1e-6, dt)
        d.facing = 1 if moved >= 0.0 else -1

    def _walk(self, world: Any, d: Dragon, tx: float, dt: float,
              speed: float) -> None:
        """One step along the ground toward *tx*, the way ``animals._walk`` does.

        A step that would rise or fall faster than CLIMB_RATE is SHORTENED, so
        steep ground slows a grounded dragon rather than snapping it up a cliff
        face, and MIN_STEP_FRAC keeps a sheer wall from stalling it on the spot
        forever. Lifted rather than reinvented: a quadruped and a wolf crossing
        the same ridge should cross it the same way.
        """
        target = _clamp_x(tx)
        gap = target - d.x
        if dt <= 0.0 or abs(gap) < 0.5 or speed <= 0.0:
            d.vx = 0.0
            d.x = _clamp_x(d.x)
            return

        sign = 1.0 if gap > 0.0 else -1.0
        d.facing = 1 if sign > 0.0 else -1
        step = min(abs(gap), max(0.0, _f(speed, 0.0)) * dt)
        if step <= 0.0:
            d.vx = 0.0
            return

        gy0 = _ground_y(world, d.x)
        nx = _clamp_x(d.x + sign * step)
        gy1 = _ground_y(world, nx)
        rise = abs(gy1 - gy0)
        budget = CLIMB_RATE * dt
        if rise > budget:
            frac = max(MIN_STEP_FRAC, budget / rise)
            nx = _clamp_x(d.x + sign * step * frac)

        moved = nx - d.x
        d.x = nx
        d.vx = moved / dt

    # -- bookkeeping -------------------------------------------------------
    def _reap(self, world: Any) -> None:
        """Clear the dead and the departed."""
        for d in list(self.dragons):
            if d.alive and d.state != PHASE_IDLE:
                continue
            if not d.alive:
                # Same phrasing as animals._reap, deliberately: a dragon brought
                # down should read in the chronicle the way a bear does.
                _log(world, f"The {d.noun} is brought down.")
                # ...and raise the drop signal. Only on the `not alive` branch:
                # the other way out of this loop is PHASE_IDLE, which is a dragon
                # that got bored and LEFT, and a departure must not pay loot.
                #
                # Recorded before _release_victim and remove() so d.x is still
                # where it died rather than wherever the teardown leaves it.
                #
                # Trimmed to the last 8. Nothing here knows whether anyone is
                # listening: the drain is a separate subsystem behind its own
                # _guarded name and can be switched off for the session by one
                # exception, at which point this list is the only thing in the
                # module that would grow forever. 8 is well above any plausible
                # frame's kill count (the measured peak is 3, under a forced
                # summon every 45 s) so the trim never actually discards a drop
                # in play - it is the bound, not a policy.
                try:
                    self.slain.append((str(d.kind), float(d.x)))
                    del self.slain[:-8]
                except Exception:
                    # A registry rehydrated by an older from_dict has no `slain`.
                    # Losing one drop is the right cost; taking the "dragons"
                    # subsystem down for the session over loot is not.
                    log.debug("could not record a slain dragon", exc_info=True)
            self._release_victim(world, d)
            self.remove(d)

    # ----------------------------------------------------------- persist --
    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "draws": int(self.draws),
            "dragons": [d.to_dict() for d in self.dragons],
            "next_id": int(self.next_id),
            "t": float(self.t),
            "seen_gate": bool(self.seen_gate),
            "next_in": float(self.next_in),
            "herald_done": bool(self.herald_done),
            "visits": int(self.visits),
        }

    @classmethod
    def from_dict(cls, d: Any, seed: int | None = None) -> "DragonRegistry":
        """Defensive load: an unreadable record is dropped, never fatal.

        ``seed`` is optional so a caller holding the world seed can thread it
        through. ``World.from_dict`` loads every subsystem through a one-argument
        helper and cannot, so the saved seed is used, and failing that a stable
        one is derived from the record itself with :func:`zlib.crc32` - never
        ``hash()``, which CPython salts per process and which would make a
        reloaded world reproducible within a launch and different on the next
        one. That exact bug has shipped six times in this project.
        """
        base = _i((d or {}).get("seed"), 0) if isinstance(d, dict) else 0
        if seed is not None:
            base = _i(seed, 0)
        elif not base:
            base = _record_seed(d)
        reg = cls(seed=base)
        if not isinstance(d, dict):
            return reg
        try:
            raw = d.get("dragons")
            if isinstance(raw, (list, tuple)):
                for item in raw:
                    try:
                        g = Dragon.from_dict(item)
                    except Exception:
                        log.debug("skipped an unreadable dragon", exc_info=True)
                        continue
                    if g.alive:
                        reg.dragons.append(g)
            reg.next_id = max(1, _i(d.get("next_id"), 1))
            for g in reg.dragons:
                reg.next_id = max(reg.next_id, g.id + 1)
            reg.t = max(0.0, _f(d.get("t"), 0.0))
            reg.seen_gate = _b(d.get("seen_gate"), False)
            reg.herald_done = _b(d.get("herald_done"), False)
            reg.visits = max(0, _i(d.get("visits"), 0))
            # A save from a world that had already used up its interval must not
            # come back with a dragon pending on the first frame - the same
            # floor AnimalRegistry.from_dict puts on next_spawn.
            reg.next_in = max(5.0, _f(d.get("next_in"), DRAGON_FIRST_DELAY))
            # Replay the stream to where it was, so a reload CONTINUES the
            # sequence instead of restarting it. Capped, because a hand-edited
            # "draws": 1e12 must cost a fraction of a second and not the session.
            want = _clamp(_i(d.get("draws"), 0), 0, _MAX_REPLAY)
            for _ in range(int(want)):
                reg._rng.random()
            reg.draws = int(want)
        except Exception:
            log.warning("dragon registry save unreadable; starting empty",
                        exc_info=True)
            return cls(seed=base)
        return reg


def _record_seed(d: Any) -> int:
    """A stable seed derived from a save record that carries none.

    crc32 of what the record *does* hold, never ``hash()`` of a string: CPython
    salts str hashing per process, which would move the irreproducibility from
    per-load to per-launch rather than fixing it.
    """
    parts = ["dragons"]
    if isinstance(d, dict):
        parts.append(str(_i(d.get("next_id"), 0)))
        parts.append("%.4f" % _f(d.get("t"), 0.0))
        parts.append(str(_i(d.get("visits"), 0)))
        raw = d.get("dragons")
        if isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, dict):
                    parts.append("%s:%d" % (item.get("kind"), _i(item.get("id"), 0)))
                else:
                    parts.append("?")
    return zlib.crc32("|".join(parts).encode("utf-8", "replace"))


# ------------------------------------------------------- module-level API --
def registry_of(world: Any) -> DragonRegistry | None:
    """The world's dragon registry, or None if it has not got one."""
    reg = getattr(world, "dragons", None)
    return reg if isinstance(reg, DragonRegistry) else None


def dragons_of(world: Any) -> list[Dragon]:
    """Every living dragon. Safe on a world with none at all, so the throwing
    lane and render can call it unconditionally."""
    reg = registry_of(world)
    return [] if reg is None else reg.alive()


def nearest_dragon(world: Any, x: float, *,
                   hostile_only: bool = True) -> Dragon | None:
    reg = registry_of(world)
    return None if reg is None else reg.nearest(x, hostile_only=hostile_only)


# ============================================================== smoke test ==
if __name__ == "__main__":   # pragma: no cover - headless, drives a real World
    import json
    import sys

    from ..constants import MELEE_CEILING as MELEE_CEILING_
    from .world import World

    DT = 1.0 / 30.0

    #: throwing.py's own words: "anything meant to be killable by throwing has
    #: to fly under about 100 px". Written out rather than imported because that
    #: module is another lane's; if the two ever disagree, the reach table this
    #: block prints is what will say so.
    THROWABLE_CEIL = 100.0

    # world.py now DRIVES the registry itself (`_guarded("dragons", ...)`), so
    # every loop below that calls `w.tick(DT)` must NOT also call `reg.tick`.
    # It used to do both, which double-ticked every dragon: this block was red
    # in the shipped tree, and the failure was worse than a red light, because
    # every window measured in here was measured at twice the true rate. Loops
    # that tick only the registry (the razing bench, _staged) are deliberate -
    # they want a fast world that is not advancing anything else - and are left
    # exactly as they are.
    def drive(kind: str, seed: int, seconds: float = 400.0,
              *, indoors: bool = False) -> dict[str, Any]:
        """Force one visit of *kind* and record every phase it reaches."""
        w = World(seed=seed)
        reg = DragonRegistry(seed=seed ^ 0xD4A)
        w.dragons = reg
        # Give it something to want: a couple of graves for the skeletal, a
        # finished hut for the quadruped to raze.
        # Beside the colony, not at a fixed fraction of the map: on a WORLD_W
        # world a grave at 0.45*W can be 3000 px from anybody, and the skeletal
        # would spend the whole 400 s flying to a graveyard nobody can see.
        cx = reg._colony_x(w)
        for dx, who in ((-40.0, "Ana"), (+30.0, "Bo")):
            gx = _clamp_x(cx + dx)
            w.props.add_grave(gx, w.terrain.ground_y(gx), who)
        # ...and a FINISHED hut, which this docstring has always claimed to
        # provide and never did. It used to get one for free because a colony
        # reliably finished something inside 400 s; it does not any more (a
        # freshly seeded world builds one firepit and never completes it), so
        # PHASE_RAZE became unreachable for a reason that has nothing to do with
        # dragons. Planting the hut here makes this bench test the razing state
        # machine instead of testing the colony's economy through it.
        hx = _clamp_x(cx + 90.0)
        w.structures.create("hut", hx, w.terrain.ground_y(hx), built=True)
        seen: set[str] = set()
        alts: list[float] = []
        windows: dict[int, list[float]] = {}
        reg.summon(kind, 0.5)
        for i in range(int(seconds / DT)):
            if indoors:
                # BEFORE the tick, not after: the dragon's own step runs inside
                # w.tick, so setting this afterwards left it one whole frame of
                # reachable colonists every frame - which is not "everyone is
                # indoors", it is "everyone is indoors except when it looks".
                for a in w.population.alive_agents():
                    a.inside = 1
            w.tick(DT)
            for g in reg.all():
                seen.add(g.state)
                alts.append(g.alt)
                if g.alive and g.sated:
                    c = clearance(w, g)
                    win = windows.setdefault(g.id, [0.0, 0.0])
                    if c <= MELEE_CEILING_:
                        win[0] += DT
                    if c <= THROWABLE_CEIL:
                        win[1] += DT
            if reg.visits and not reg.all() and i > 60:
                break
        return {
            "world": w, "reg": reg, "seen": seen, "windows": windows,
            "alt_min": min(alts) if alts else 0.0,
            "alt_max": max(alts) if alts else 0.0,
            "lines": [c for c in w.chronicle
                      if any(n in c for n in _NOUN.values())
                      or "crosses the sky" in c],
        }

    print("-- every kind reaches every phase in its machine ------------------")
    for kind in KINDS:
        hit: set[str] = set()
        for seed in (11, 23, 37, 41, 59):
            r = drive(kind, seed)
            hit |= r["seen"]
            if KIND_PHASES[kind] and set(KIND_PHASES[kind]) <= hit:
                break
        want = set(KIND_PHASES[kind])
        missing = want - hit
        print(f"  {kind:<10} reached {sorted(hit)}")
        assert not missing, f"{kind} never reached {sorted(missing)}"

    print("-- the bone-wyrm's share follows the graveyard -------------------")
    # The reported defect was "0 skeletal sightings in 16 colonies over 24
    # sim-hours". The spawn path is NOT the reason - instrumented over 8
    # colonies, _pick_kind chose skeletal 35 times in 185 rolls (19%, its exact
    # weight) and _spawn returned a live dragon all 35 times. The reason is that
    # sixteen colonies saw seventeen dragons between them and TEN of those were
    # the herald: skeletal's fair share of what was left is 1.4 visits, so zero
    # is inside the noise rather than outside it. See SKELETAL_WEIGHT_PER_GRAVE.
    shares: list[tuple[int, float]] = []
    for stones in (0, 1, 5, 10):
        w = World(seed=7)
        for k in range(stones):
            gx = 400.0 + 70.0 * k
            w.props.add_grave(gx, w.terrain.ground_y(gx), f"G{k}")
        probe_reg = DragonRegistry(seed=1)
        probe_reg.herald_done = True
        rolls = [probe_reg._pick_kind(w) for _ in range(4000)]
        share = rolls.count(KIND_SKELETAL) / float(len(rolls))
        shares.append((stones, share))
        print(f"  {stones:2d} graves -> skeletal takes {100.0 * share:4.1f}% "
              f"of the roll")
    assert shares[0][1] == 0.0, "it came for a graveyard that is not there"
    assert 0.17 <= shares[1][1] <= 0.25, shares[1]
    assert shares[3][1] > shares[1][1] * 1.3, shares
    # Commoner, never the DEFAULT: at a 0.45 cap the bone-wyrm outnumbered the
    # wyvern 69 to 49 over 203 visits, which trades "you never see it" for "it
    # is mostly what you see", and the one you mostly see should be the one
    # that costs you something.
    assert shares[3][1] < _KIND_WEIGHTS[KIND_WYVERN], shares

    print("-- altitude is authored, y is derived ---------------------------")
    r = drive(KIND_SERPENT, 23, 300.0)
    print(f"  serpent alt ran {r['alt_min']:.0f} .. {r['alt_max']:.0f}")
    assert r["alt_min"] < -50.0, "the serpent never went below the surface"
    r = drive(KIND_FLYOVER, 5, 120.0)
    print(f"  flyover alt held {r['alt_min']:.0f} .. {r['alt_max']:.0f}")
    # The range is the terrain's relief now, not FLYOVER_ALT: the band is
    # measured off the map's highest ground and the animal holds one screen
    # line, so its clearance over a valley is honestly larger. What still has to
    # be true is the floor - even directly over the peak, and even after
    # FLYOVER_SCREEN_MIN_Y has pulled the line down to stay in frame, it is
    # further up than a spear can be thrown at all (240 px horizontally, and a
    # 141 px ceiling straight overhead).
    assert r["alt_min"] > 240.0, r["alt_min"]

    print("-- the flyover holds ONE line, not one clearance -----------------")
    # It is the only kind whose alt is derived rather than authored, so the
    # thing to assert is the screen y, not the alt: over broken ground a
    # constant alt sags into every valley, which is what this used to do. The
    # alt range printed above is now the terrain's relief, and that is correct.
    w = World(seed=21)
    reg = DragonRegistry(seed=21)
    w.dragons = reg
    reg.summon(KIND_FLYOVER, 0.0)
    ys: list[float] = []
    for _ in range(int(150.0 / DT)):
        w.tick(DT)
        for g in reg.all():
            if _EDGE_MIN <= g.x <= _EDGE_MAX:
                ys.append(g.y)
    relief = max(float(v) for v in w.terrain.height) - \
        min(float(v) for v in w.terrain.height)
    print(f"  screen y over {relief:.0f} px of relief: "
          f"{min(ys):.1f} .. {max(ys):.1f} ({max(ys) - min(ys):.2f} px of sag)")
    assert ys and max(ys) - min(ys) <= 1.0, (min(ys), max(ys))

    print("-- the quadruped is NEVER airborne on screen ---------------------")
    # render/dragons.py's quadruped module has no flight pose at all, so its
    # arrive and leave legs have nothing correct to draw. The sim's answer is
    # to spend the whole airborne leg outside the frame; this is the assertion
    # that keeps it true if anyone retunes SPAWN_MARGIN or the speeds.
    #
    # "On screen" is _on_stage, not the map bounds. It USED to be the map bounds
    # and the two were the same thing; on a WORLD_W world they are not, and the
    # world-bounds version of this check reported 111 px of altitude in frame
    # for a quadruped that was in fact 1178 px away and invisible. A test that
    # can only pass on a map exactly one screen wide is not testing the rule the
    # comment above states.
    w = World(seed=11)
    reg = DragonRegistry(seed=11)
    w.dragons = reg
    reg.summon(KIND_QUADRUPED, 0.0)
    worst_on_screen = 0.0
    for _ in range(int(260.0 / DT)):
        w.tick(DT)                           # drives reg - see drive() above
        for g in reg.all():
            if _on_stage(g.home_x, g.x):
                worst_on_screen = max(worst_on_screen, g.alt)
    print(f"  highest a quadruped ever got while on screen: "
          f"{worst_on_screen:.2f} px")
    assert worst_on_screen <= 0.5, worst_on_screen

    print("-- reach: what the ground can and cannot touch -------------------")
    # The whole point of requirement 5, as a table. THROW_MAX_RANGE (240) is
    # throwing.py's constant and that module is another lane's, so it is
    # written out here rather than imported - if the two ever disagree this
    # print is the thing that will say so.
    THROW_MAX_RANGE = 240.0
    from ..constants import MELEE_CEILING as _MELEE
    w = World(seed=5)
    reg = DragonRegistry(seed=5)
    w.dragons = reg
    print(f"  {'kind':<10} {'alt':>6}  {'spikes':>7} {'melee':>6} {'throw':>6}")
    for kind in KINDS:
        alt = {KIND_FLYOVER: 360.0, KIND_WYVERN: 180.0, KIND_QUADRUPED: 0.0,
               KIND_SERPENT: 9.0, KIND_SKELETAL: 85.0}[kind]
        probe = Dragon(kind=kind, x=800.0, alt=alt, health=1.0, max_health=1.0)
        c = clearance(w, probe)
        print(f"  {kind:<10} {c:6.0f}  {'yes' if c <= BARRICADE_SPIKE_H else 'no':>7}"
              f" {'yes' if c <= _MELEE else 'no':>6}"
              f" {'yes' if c <= THROW_MAX_RANGE else 'no':>6}")
    # The four claims the design rests on, as assertions.
    assert clearance(w, Dragon(kind=KIND_FLYOVER, alt=360.0)) > THROW_MAX_RANGE
    assert clearance(w, Dragon(kind=KIND_WYVERN, alt=180.0)) > BARRICADE_SPIKE_H
    assert clearance(w, Dragon(kind=KIND_WYVERN, alt=10.0)) <= _MELEE
    assert clearance(w, Dragon(kind=KIND_SKELETAL, alt=85.0)) > BARRICADE_SPIKE_H
    assert clearance(w, Dragon(kind=KIND_SKELETAL, alt=85.0)) <= THROW_MAX_RANGE
    assert clearance(w, Dragon(kind=KIND_QUADRUPED, alt=0.0)) <= _MELEE

    print("-- grounded() is what the barricade asks --------------------------")
    reg.dragons = [Dragon(id=1, kind=KIND_QUADRUPED, x=800.0, alt=0.0,
                          health=9.0, max_health=9.0),
                   Dragon(id=2, kind=KIND_WYVERN, x=800.0, alt=180.0,
                          health=9.0, max_health=9.0),
                   Dragon(id=3, kind=KIND_SERPENT, x=800.0, alt=9.0,
                          health=9.0, max_health=9.0)]
    got = sorted(g.kind for g in reg.grounded(w))
    print("  within the spikes:", got)
    assert got == [KIND_QUADRUPED, KIND_SERPENT], got
    reg.dragons = []

    print("-- the siege actually costs buildings ---------------------------")
    # Incidental damage in a soak is world-state dependent - a struggling
    # colony has nothing standing to raze - so the razing path is pinned here
    # against a world that definitely has buildings in it.
    w = World(seed=48)
    reg = DragonRegistry(seed=48)
    w.dragons = reg
    for i in range(3):
        gx = 800.0 + 90.0 * i
        w.structures.create("hut", gx, w.terrain.ground_y(gx), built=True)
    before = [st.hp for st in w.structures.all()]
    reg.summon(KIND_QUADRUPED, 0.0)
    razed_ticks = 0
    for _ in range(int(260.0 / DT)):
        reg.tick(w, DT)
        razed_ticks += sum(1 for g in reg.all() if g.state == PHASE_RAZE)
    after = [st.hp for st in w.structures.all()]
    ruined = sum(1 for st in w.structures.all() if st.is_ruined)
    print(f"  hut hp {before} -> {after}; {ruined} ruined after "
          f"{razed_ticks / 30.0:.0f}s of razing")
    assert sum(after) < sum(before), (before, after)
    assert ruined >= 1, "a 150 s siege pulled nothing down"

    print("-- the feeding rule --------------------------------------------")
    d = Dragon(kind=KIND_WYVERN, health=180.0, max_health=180.0)
    assert d.hurt(999.0) is False and d.health == 180.0, "unfed damage landed"
    assert d._refused_pending is True and d.refused_t == 0.0
    d.sated = True
    assert d.hurt(50.0) is False and abs(d.health - 130.0) < 1e-6
    assert d.hurt(999.0) is True and not d.alive
    bones = Dragon(kind=KIND_SKELETAL, health=150.0, max_health=150.0, sated=True)
    assert bones.hurt(150.0) is True, "the undead needed a meal"
    print("  unfed refuses, fed dies, skeletal killable from tick one")

    print("-- a dragon that eats is killable, and the books balance ---------")
    fed = None
    for seed in (11, 23, 37, 41, 59, 71, 83):
        r = drive(KIND_WYVERN, seed, 400.0)
        got = [g for g in r["reg"].all()]
        w = r["world"]
        if any("has fed" in ln for ln in w.chronicle):
            fed = r
            break
    assert fed is not None, "no wyvern ever completed a seize in 7 seeds"
    w = fed["world"]
    book = w.reconcile()
    print("  reconcile:", book)
    assert book["ok"] == 1 and book["residual"] == 0, book
    assert w.stats["died"] >= 1
    assert any("has fed. It can be hurt now." in ln for ln in w.chronicle)
    assert any("takes" in ln for ln in w.chronicle)

    print("-- THE WINDOW: what 'it can be hurt now' is worth in seconds -----")
    # The headline defect this phase exists to close. MEASURED before the gorge:
    # 511 dragon visits over 16 seeds x 90 sim-min plus 369 forced ones, every
    # feeding wyvern and serpent wrote "It can be hurt now.", and ZERO dragons
    # were ever brought down - because _devour went straight to PHASE_LEAVE,
    # which climbs at 96 px/s and is also a word combat_actions.animal_leaving
    # reads as "do not spend a spear on this". The window was 0.35 s of melee
    # and, for throwers, nothing at all.
    from ..constants import SPEAR_COOLDOWN as _SP_CD
    from ..constants import SPEAR_DAMAGE as _SP_DMG
    for kind in (KIND_WYVERN, KIND_SERPENT):
        mel: list[float] = []
        thr: list[float] = []
        for seed in (11, 23, 37, 41, 59, 71):
            r = drive(kind, seed, 400.0)
            for win in r["windows"].values():
                if win[0] > 0.0 or win[1] > 0.0:
                    mel.append(win[0])
                    thr.append(win[1])
        assert mel, f"no {kind} ever fed in six seeds"
        mel.sort()
        thr.sort()
        hp = _stats_for(kind)[0]
        # One armed adult who commits, at the melee numbers the game already
        # ships: SPEAR_DAMAGE per SPEAR_COOLDOWN.
        dps = _SP_DMG / _SP_CD
        print(f"  {kind:<9} fed x{len(mel)}  melee {mel[len(mel) // 2]:5.1f}s  "
              f"throwable {thr[len(thr) // 2]:5.1f}s  "
              f"= {mel[len(mel) // 2] * dps:.0f} hp of one fighter vs {hp:.0f}")
        assert mel[0] >= 6.0, (kind, mel)
        assert thr[0] >= 10.0, (kind, thr)
        assert mel[len(mel) // 2] * dps >= hp, (kind, mel, hp)

    print("-- and a colony that fights really does bring one down -----------")
    # The window above is arithmetic; this is the whole path end to end - the
    # quarry chooser seeing a gorging dragon, FightAnimal and ThrowSpear
    # committing to it, the damage landing on Dragon.hurt now that sated is
    # True, _reap writing the line, and the books still balancing afterwards.
    #
    # The ONE thing forced here is the spears: a colony that has not armed
    # itself is not the colony this claim is about, and how many colonists
    # happen to be carrying a spear when a dragon arrives is behaviour's
    # business, not this module's.
    #
    # Not asserted as "every colony kills it", because it is not true and
    # should not be: over 24 seeds an armed colony brought down 6 of the 24
    # wyverns that fed on it, and of the 18 that escaped, 6 left at 24 hp of
    # 180 - one more spear. A quarter, with visible near-misses, is the shape
    # this is meant to have.
    from ..constants import WEAPON_SPEAR as _WPN
    # "is brought down" alone is NOT the test: animals._reap writes the same
    # sentence for every wolf the colony kills, and the first cut of this block
    # passed on seed 23 with "The wolf is brought down." Scope it to the noun.
    _DOWN = f"The {_NOUN[KIND_WYVERN]} is brought down"
    down, fought, worst = 0, 0, 180.0
    for seed in (1, 2, 3, 4, 5, 6, 7, 8):
        w = World(seed=seed)
        reg = w.dragons
        reg.summon(KIND_WYVERN, 0.5)
        armed = False
        for _ in range(int(150.0 / DT)):
            w.tick(DT)
            beast = reg.all()[0] if reg.all() else None
            if beast is not None:
                worst = min(worst, beast.health)
                if beast.sated and not armed:
                    armed = True
                    for a in w.population.alive_agents():
                        if str(getattr(a, "role", "")) != "child":
                            a.weapon = _WPN
            if any(_DOWN in ln for ln in w.chronicle):
                down += 1
                assert w.reconcile()["residual"] == 0, w.reconcile()
                break
        fought += 1 if armed else 0
    print(f"  {down} of {fought} armed colonies brought the wyrm down; "
          f"the best of the rest got it to {worst:.0f} hp of 180")
    assert fought >= 6, fought
    assert down >= 1, "no armed colony managed it in eight seeds"

    print("-- the gorge outlives the patience timer ------------------------")
    # A wyvern that spends 170 of its 180 patient seconds hunting and THEN eats
    # somebody must still get the whole window. _VISIT_MAX used to fire on the
    # next tick and yank it into PHASE_LEAVE, which would have made the payoff
    # something a slow colony could never see.
    d = Dragon(id=4, kind=KIND_WYVERN, x=640.0, alt=GORGE_ALT, health=180.0,
               max_health=180.0, sated=True, fed_count=1, state=PHASE_GORGE)
    d.age = _VISIT_MAX[KIND_WYVERN] + 50.0
    reg = DragonRegistry(seed=1)
    reg.dragons = [d]
    held = 0.0
    for _ in range(int(DRAGON_GORGE_SEC / DT)):
        reg.tick(None, DT)
        if d.state == PHASE_GORGE:
            held += DT
    print(f"  an over-patient wyvern still gorged for {held:.1f}s")
    assert held >= DRAGON_GORGE_SEC - 0.2, held

    print("-- the flyover cannot move any morale gate -----------------------")
    # Panic 0.32, MORALE_TO_GROW 0.45, HUT_TIER_MORALE 0.55. Every one of them
    # is below FLYOVER_MORALE_FLOOR, and the toll neither touches anyone below
    # the floor nor takes anyone through it - so a shape in the sky cannot be
    # the reason a colony panics, stops breeding or loses its tier. It used to
    # be: dragons cost ~14% more deaths and only 9 of the 48 extra were
    # anything actually eating anybody.
    from ..constants import HUT_TIER_MORALE as _HTM
    from ..constants import MORALE_TO_GROW as _MTG
    _PANIC = 0.32                                        # behavior.py's number
    assert FLYOVER_MORALE_FLOOR - FLYOVER_MORALE > max(_HTM, _MTG, _PANIC), \
        "the flyover's floor no longer clears every morale gate"

    class _Mood:
        alive = True

        def __init__(self, m: float) -> None:
            self.morale = m
    crowd = [_Mood(v) for v in (0.05, _PANIC + 0.01, _MTG + 0.01, _HTM + 0.01,
                                FLYOVER_MORALE_FLOOR, 0.64, 0.9, 1.0)]

    class _W:
        population = None
    _w = _W()
    _w.population = crowd                    # _agents() walks a plain iterable
    before_m = [a.morale for a in crowd]
    DragonRegistry(seed=1)._flyover_toll(_w)
    after_m = [a.morale for a in crowd]
    print("  morale", [f"{b:.2f}->{a:.2f}" for b, a in zip(before_m, after_m)])
    for b, a in zip(before_m, after_m):
        assert a <= b, (b, a)                            # never a gift
        for gate in (_PANIC, _MTG, _HTM):
            assert not (b > gate >= a), (b, a, gate)     # never across a gate

    print("-- a summon survives the gate latching on the same tick ----------")
    # _tick_gate runs before _tick_schedule and used to overwrite next_in with
    # DRAGON_FIRST_DELAY, so a tray summon issued on the exact tick the colony
    # finished its second stone hut was silently pushed eight minutes out.
    class _Gated:
        dragons_unlocked = True
    reg = DragonRegistry(seed=2)
    reg.herald_done = True
    reg.summon(KIND_WYVERN, 0.0)
    reg.tick(_Gated(), DT)
    print(f"  gate latched and the summon still landed: {reg.count()} dragon(s)")
    assert reg.seen_gate and reg.count() == 1, (reg.seen_gate, reg.count())

    print("-- the starve hatch, and the visit that has to end ---------------")
    # Everyone hides in the huts. Two things must happen, and the second one is
    # the half the feeding rule does NOT give you: it becomes mortal, and then
    # it goes home. Before _VISIT_MAX it did the first and never the second, so
    # a colony that locked its doors got a killable wyvern circling forever.
    w = World(seed=23)
    reg = DragonRegistry(seed=23)
    w.dragons = reg
    reg.summon(KIND_WYVERN, 0.0)
    hatch_at = left_at = None
    for i in range(int(400.0 / DT)):
        for a in w.population.alive_agents():
            a.inside = 1                     # before the tick - see drive()
        w.tick(DT)
        beast = reg.all()[0] if reg.all() else None
        if hatch_at is None and beast is not None and beast.sated:
            hatch_at = i * DT
        if left_at is None and hatch_at is not None and not reg.all():
            left_at = i * DT
            break
    print(f"  turned mortal at {hatch_at}s, gone by {left_at}s")
    assert hatch_at is not None and abs(hatch_at - DRAGON_STARVE_SEC) < 2.0, hatch_at
    assert left_at is not None, "it never left a colony that stayed indoors"
    assert any("careless" in ln for ln in w.chronicle), list(w.chronicle)[-4:]
    assert not any("has fed" in ln for ln in w.chronicle), "it fed on nobody"

    print("-- the unsated line is said once --------------------------------")
    w = World(seed=7)
    reg = DragonRegistry(seed=7)
    w.dragons = reg
    reg.summon(KIND_WYVERN, 0.0)
    reg.tick(w, DT)
    beast = reg.all()[0]
    for _ in range(6):
        beast.hurt(34.0)
        reg.tick(w, DT)
    said = [ln for ln in w.chronicle if "has not fed" in ln]
    print(f"  six spears ->{len(said)} line(s):", said[:1])
    assert len(said) == 1, said

    print("-- save round-trip, mid-flight, all five ------------------------")

    def _staged(kind: str, ticks: int) -> tuple[DragonRegistry, World, int]:
        """A registry *ticks* into a forced visit of *kind*, and its world."""
        w = World(seed=13)
        reg = DragonRegistry(seed=13)
        w.dragons = reg
        w.props.add_grave(700.0, w.terrain.ground_y(700.0), "Cai")
        w.props.add_grave(760.0, w.terrain.ground_y(760.0), "Dov")
        reg.summon(kind, 0.0)
        lived = 0
        for i in range(ticks):
            reg.tick(w, DT)
            if reg.all():
                lived = i + 1
        return reg, w, lived

    for kind in KINDS:
        # Two passes: measure how long this kind's visit lasts, then re-run and
        # snapshot at HALF of it. A fixed wall-clock snapshot is no good - a
        # serpent's whole visit is over inside 14 s while a quadruped's siege
        # runs 150 - and "mid-flight" has to mean mid-flight for all five.
        _r0, _w0, span = _staged(kind, int(200.0 / DT))
        assert span > 4, f"{kind} never lived at all"
        reg, w, _ = _staged(kind, max(2, span // 2))
        assert reg.all(), f"{kind} was gone before the save"
        blob = reg.to_dict()
        wire = json.loads(json.dumps(blob))         # the save file is JSON
        clone = DragonRegistry.from_dict(wire)
        assert clone.to_dict() == blob, (kind, clone.to_dict(), blob)
        # ...and it must keep running from where it stopped, identically, in
        # two independently loaded copies.
        a, b = DragonRegistry.from_dict(wire), DragonRegistry.from_dict(wire)
        w2, w3 = World(seed=13), World(seed=13)
        for _ in range(300):
            a.tick(w2, DT)
            b.tick(w3, DT)
        assert a.to_dict() == b.to_dict(), kind
        print(f"  {kind:<10} visit {span} ticks; round-tripped in "
              f"{reg.all()[0].state!r} and resumed identically")

    print("-- junk never raises -------------------------------------------")
    assert DragonRegistry.from_dict(None).count() == 0
    assert DragonRegistry.from_dict("nonsense").count() == 0
    assert DragonRegistry.from_dict({"dragons": [None, 7, {"kind": "sponge"}]}) \
        is not None
    assert Dragon.from_dict(None).kind == KIND_WYVERN
    assert Dragon.from_dict({"kind": KIND_QUADRUPED, "state": "stoop"}).state \
        == PHASE_LEAVE
    junk = Dragon.from_dict({"x": float("nan"), "alt": "boom", "health": None})
    assert math.isfinite(junk.x) and math.isfinite(junk.alt)
    assert Dragon.from_dict({"kind": KIND_SKELETAL, "sated": False}).sated is True
    lone = DragonRegistry(seed=3)
    lone.tick(None, 0.05)                  # no world at all
    lone.tick(object(), 0.05)              # a world with nothing on it
    lone.summon(KIND_WYVERN, 0.0)
    for _ in range(400):
        lone.tick(object(), 0.05)
    print("  survived 400 worldless ticks:", lone.count(), "alive")

    print("-- clearance() is total ----------------------------------------")
    assert clearance(None, None) == 0.0
    assert clearance(object(), object()) == 0.0

    class _NanAlt:
        alt = float("nan")
        x = 0.0
        y = 0.0
    assert clearance(None, _NanAlt()) == 0.0
    assert clearance(None, Dragon(alt=-140.0)) == -140.0, "clearance clamped"
    assert is_grounded(None, Dragon(alt=0.0)) is True
    assert is_grounded(None, Dragon(alt=300.0)) is False

    print("-- reproducible from a seed ------------------------------------")
    outs = []
    for _ in range(2):
        w = World(seed=31)
        reg = DragonRegistry(seed=31)
        w.dragons = reg
        reg.summon(None, 0.0)
        for _ in range(int(240.0 / DT)):
            w.tick(DT)                       # drives reg - see drive() above
        outs.append(json.dumps(reg.to_dict(), sort_keys=True))
    assert outs[0] == outs[1], "two identical runs diverged"
    print("  two runs of seed 31 agree:", outs[0][:78], "...")

    print("ok")
    sys.exit(0)
