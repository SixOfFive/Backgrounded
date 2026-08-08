"""Combat and crafting - the *agent* half of the animal threat.

Nine :class:`~.actions.Action` kinds, registered into ``actions._HANDLERS``
at import so a save taken mid-fight or mid-craft rehydrates correctly:

==============  ===========================================================
CraftSpear      6 s at the stockpile, spends ``SPEAR_COST``, arms the agent
CraftArmour     8 s at the stockpile, spends ``ARMOUR_COST`` (needs hides)
FightAnimal     close, then swing on ``SPEAR_COOLDOWN``. Armed agents only.
FleeAnimal      run to a safe distance. What an *unarmed* agent does.
ThrowSpear      aim, release, and you are now unarmed. See :mod:`.throwing`.
RetrieveSpear   walk to a spear lying in the dirt and pick it up.
FireBfg         the dragon relic. A line kill that does not care whose side
                anyone is on, and an 8% chance of taking the wielder with it.
FetchRelic      walk to a dragon relic lying in the dirt and put it on.
RaiseTheDead    carry the cairnstone to a headstone and spend it.
==============  ===========================================================

The relic half
--------------
FetchRelic and RaiseTheDead are late arrivals, and the reason they are late is
worth recording because it cost the whole feature a release. The lane that was
told to make agents *want* loot did not own this file, and the lane that owned
this file was never told to add a pickup action. Both lanes shipped, both were
individually correct, and the result measured **18 relics dropped, 0 picked up**
over 16 seeds x 120 sim-min: no ``FetchRelic`` kind existed, so
``behavior._fetch_is_wired()`` read False and the appetite - which was written,
tuned and measured - switched itself off. Loot on and loot off were
bit-identical on all 16 seeds. Every downstream effect (``bfg_shots``,
``misfires``, ``raised``, ``scale_refusals``) was zero for the same one reason:
nothing was ever worn.

So the wanting and the doing now live one import apart on purpose.
:func:`fetchable_relic` is the single answer to "which relic, if any, may this
person go and get" and ``behavior._score_loot`` calls it; behaviour still owns
every *number*, because the bands a fetch has to beat (Celebrate, Mourn, Eat)
are declared over there and a ceiling tuned against numbers it cannot see is
exactly how the stone-hut upgrade came to sit unclaimed for 110 sim-minutes.

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

Throwing
--------
:mod:`.throwing` owns the projectile; this file owns the *decision*, and the
decision is narrow on purpose. Melee is still the doctrine at contact range,
because a throw costs you the spear:

* An **adult** throws only when melee is not the better answer - the target is
  further than ``FIGHT_RADIUS`` (so the alternative is a 200 px sprint into a
  wolf that is already eating somebody, which is what the RESCUE_RADIUS comment
  below records going wrong), or the target is above ``MELEE_CEILING`` and
  cannot be reached with a spear in the hand at all. And never while something
  is inside ``FLEE_RADIUS`` of the thrower: disarming yourself with a wolf on
  top of you is not a trade, it is a mistake.
* A **child** throws, and it is the only thing a child can do in a fight -
  :func:`score_combat` hard-excludes ``role == "child"`` from FightAnimal, so
  before this a child was purely a liability. They get a wider aim error and
  0.6x the range, and they still run once anything is closer than
  ``CHILD_THROW_MIN_GAP``; the window between that and their range limit is the
  contribution.
* Whoever threw it can walk over and pick it up again (RetrieveSpear), which is
  why a miss is a setback rather than a loss, and why the retrieval score sits
  next to CraftSpear's rather than under it.

Wiring (the two things this module cannot do for itself, since it may only
write this file):

* ``behavior.score_actions`` already merges :func:`score_combat` into its score
  dict with ``dict.update``, and :func:`score_combat` merges :func:`score_throw`
  into its own result, so the two new kinds arrive with no edit to behavior.py.
  ``behavior.choose_action`` likewise already falls back to
  :func:`make_combat_action` for any kind absent from ``_MAKERS``.
  :func:`install_makers` still splices all six in for a caller that would rather
  be explicit.
* ``behavior.emergency_override`` returns True when an animal is inside
  ``FLEE_RADIUS``, or an agent finishes chopping its tree while a wolf eats it.
  :func:`under_attack` is the predicate it calls. This was the one piece of the
  wiring that stayed a *should* for several releases, and while it did, every
  number in the table above was unreachable for anyone already holding a job:
  of 424 agent-seconds where an armed colonist scored FightAnimal above
  ``behavior.OVERRIDE_FLOOR``, 77.6% were locked inside an unfinished
  non-combat action and 0.9% were free to switch. Now wired, and measured -
  32 same-seed pairs put mauling deaths down a third (15.60 -> 10.16 per 1000
  colonist-minutes, 17 seeds down against 4) with fall deaths unmoved, which
  was the trade that had to be ruled out. The table lives in
  ``behavior.emergency_override``.
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
    hermit_stash_qty,
    is_hermit,
    make_action,
    nearest_structure,
    step_toward,
    stock_add,
    stock_qty,
    stock_take,
    world_now,
)
from .actions import alive_agents as _alive_agents
from .throwing import (
    RETRIEVE_RADIUS,
    THROW_MAX_RANGE,
    THROW_MIN_RANGE,
    can_throw,
    clearance,
    ensure_registry,
    flight_clear,
    launch_point,
    registry_of as spear_registry_of,
    throw_targets,
)
#: The beam's geometry is throwing.py's geometry. Deliberately NOT a defensive
#: import: the whole point of borrowing ``_target_box``/``_ground_y``/
#: ``_STEP_PX`` is that the sweep and ``flight_clear`` agree, to the pixel,
#: about where the hillside is and how tall a wolf is. A silent local copy that
#: drifts out of step with them is a worse outcome than a loud ImportError in
#: the smoke test, which is what a rename here would produce.
from .throwing import (          # noqa: PLC2701 - see above
    _STEP_PX as _BEAM_STEP,
    _aim_y as _beam_aim_y,
    _ground_y as _beam_ground_y,
    _target_box as _beam_box,
)

#: px. AGENT_HEIGHT (26) plus a spear held overhead - above this a stickman
#: cannot reach and must throw. Declared in constants.py by the workstream that
#: owns that file; the fallback is here so this module still imports (and this
#: gate still holds) in a checkout where that append has not landed. Same number
#: either way, and constants.py wins the moment it exists.
try:                                    # pragma: no cover - trivial
    from ..constants import MELEE_CEILING
except ImportError:                     # pragma: no cover - trivial
    MELEE_CEILING = 44.0

#: THE QUIVER. ``sim/throwing.py`` owns how many spears a stickman is carrying;
#: this file only ever asks it questions. Three of its four pinned signatures:
#:
#: * ``spears_carried(agent) -> int``   - how many, clamped, never raises.
#: * ``can_carry_more(agent) -> bool``  - room for another spear?
#: * ``carry_spear(agent, n=1) -> int`` - hand one over, clamped; NEW count.
#:
#: (``drop_spear`` is the debit and belongs to the throw path, not to this
#: file - ``SpearRegistry.throw`` calls it. Nothing here should ever spend a
#: spear behind the registry's back.)
#:
#: THE FALLBACKS ARE NOT PLACEHOLDERS, they are the single-slot rule this file
#: shipped with, spelled out. ``agent.weapon`` is one slot, so "carrying a
#: spear" is a boolean and "room for another" is "not holding one". That means
#: every gate below answers **identically before and after** the quiver lands,
#: and the pickup relaxation is decided by ``PICKUP_WHEN_ARMED`` alone rather
#: than by which half of the arc happens to be merged. A fallback that guessed
#: at ``agent.spears`` instead would read 0 on a Stickman that has no such
#: field and quietly uncap the pickup on a checkout with no cap to enforce.
try:                                    # pragma: no cover - trivial
    from .throwing import can_carry_more, carry_spear, spears_carried
except ImportError:                     # pragma: no cover - trivial
    def spears_carried(agent: Any) -> int:      # type: ignore[misc]
        return (1 if str(getattr(agent, "weapon", WEAPON_NONE) or WEAPON_NONE)
                == WEAPON_SPEAR else 0)

    def can_carry_more(agent: Any) -> bool:     # type: ignore[misc]
        return spears_carried(agent) < 1

    def carry_spear(agent: Any, n: int = 1) -> int:   # type: ignore[misc]
        if int(n) <= 0:
            return spears_carried(agent)
        try:
            agent.weapon = WEAPON_SPEAR
        except Exception:
            # Report the failure through the RETURN VALUE, exactly as the real
            # helper does. See _h_retrieve's take phase: the caller decides
            # what to do about a spear it has already lifted, and it cannot do
            # that if the two implementations disagree about how failure looks.
            return spears_carried(agent)
        return 1

#: The wyrm-gun. Same arrangement as MELEE_CEILING above and for the same
#: reason: constants.py belongs to the relics workstream, and this file has to
#: import (and behave identically) in a checkout where that append has not
#: landed. Every number here is the contract's number.
try:                                    # pragma: no cover - trivial
    from ..constants import (
        BFG_BLAST_R,
        BFG_COOLDOWN,
        BFG_CORRIDOR,
        BFG_DAMAGE,
        BFG_HEEDLESS,
        BFG_MIN_RANGE,
        BFG_MISFIRE,
        BFG_RANGE,
        BFG_WINDUP,
        CAUSE_DISINTEGRATED,
        MAX_POP,
        RELIC_BFG,
        RELIC_CAIRN,
        RELIC_FETCH_RANGE,
        RELIC_NONE,
        RELIC_PICKUP_R,
        WEAPON_BFG,
    )
except ImportError:                     # pragma: no cover - trivial
    WEAPON_BFG = "bfg9000"
    RELIC_BFG = "bfg9000"
    RELIC_CAIRN = "cairnstone"
    RELIC_NONE = ""
    RELIC_FETCH_RANGE = 520.0
    RELIC_PICKUP_R = 18.0
    MAX_POP = 20                        # keep in step with constants.MAX_POP
    BFG_RANGE = 420.0
    BFG_MIN_RANGE = 34.0                # == throwing.THROW_MIN_RANGE
    BFG_CORRIDOR = 22.0
    BFG_BLAST_R = 70.0
    BFG_DAMAGE = MAX_HEALTH * 9.0
    BFG_COOLDOWN = 9.0
    BFG_WINDUP = 1.1
    BFG_MISFIRE = 0.08
    BFG_HEEDLESS = 0.20
    CAUSE_DISINTEGRATED = "disintegrated"

#: The relic layer. The module is ``items.py``. It is NOT ``relics.py``, and
#: that one wrong word is why this import sits at module scope with a working
#: fallback instead of being a lazy ``from .X import y`` inside the function
#: that needs it.
#:
#: What shipped: ``_destroy_bfg`` did ``from .relics import unequip`` inside a
#: try/except that swallowed the ImportError *per call*. ufo.py:399 guessed the
#: same wrong name. Neither ever ran, neither ever said so, and the gun's
#: unequip-before-the-blast rule - the thing that stops a misfire leaving a
#: working BFG on the scorch mark for the next colonist - silently did not
#: exist. A name that is wrong here is wrong once, at import, in front of
#: tools/smoke.py, and it is logged at ERROR rather than DEBUG for the same
#: reason: the previous failure mode was that nobody could see it.
try:                                    # pragma: no cover - trivial
    from .items import (
        consume as relic_consume,
        equip as relic_equip,
        noun_for as relic_noun,
        registry_of as relic_registry_of,
        unequip as relic_unequip,
        worn as relic_worn,
    )
except ImportError:                     # pragma: no cover - trivial
    logging.getLogger(__name__).error(
        "sim.items is missing; relics degrade to inert. Fetching, raising and "
        "the BFG's unequip-on-misfire are all off.")

    def relic_registry_of(world: Any) -> Any:       # type: ignore[misc]
        return getattr(world, "relics", None)

    def relic_worn(agent: Any) -> str:              # type: ignore[misc]
        try:
            return str(getattr(agent, "relic", "") or "")
        except Exception:
            return ""

    def relic_equip(agent: Any, kind: str) -> bool:  # type: ignore[misc]
        return False

    def relic_unequip(agent: Any) -> str:           # type: ignore[misc]
        # The by-hand clearing _destroy_bfg used to fall through to. Kept here
        # so there is exactly one copy of it and every caller gets it.
        kind = relic_worn(agent)
        for attr, val in (("weapon", WEAPON_NONE), ("relic", RELIC_NONE)):
            try:
                setattr(agent, attr, val)
            except Exception:
                pass
        return kind

    def relic_consume(world: Any, agent: Any) -> str:   # type: ignore[misc]
        return relic_unequip(agent)

    def relic_noun(kind: Any) -> str:               # type: ignore[misc]
        return "relic"

log = logging.getLogger(__name__)

__all__ = [
    "COMBAT_ACTION_KINDS",
    "score_combat",
    "score_throw",
    "score_bfg",
    "beam_sweep",
    "make_combat_action",
    "make_throw_action",
    "install_makers",
    "under_attack",
    # the ranged abstraction - actions._h_lookout's attack sub-step fires
    # through these three and knows nothing about which weapon it is holding
    "RANGED_KINDS", "ranged_ready", "ranged_target", "ranged_fire",
    # the relic half - behavior.py imports these to score what it cannot build
    "fetchable_relic", "raisable_grave",
    # animal accessor layer, reusable by behavior.py / events.py
    "animals_of", "living_animals", "animal_alive", "animal_by_id",
    "animal_id", "animal_kind", "animal_x", "animal_leaving",
    "nearest_animal", "hurt_animal", "claim_hides", "note_animal_seen",
    "sighting_freshness", "melee_foes",
]

#: ``behavior._fetch_is_wired()`` tests ``"FetchRelic" in COMBAT_ACTION_KINDS``
#: and keeps the loot appetite at zero until it is true. That is the switch the
#: whole feature hung off for a release; do not rename either string without
#: reading the matching comment over there.
COMBAT_ACTION_KINDS: tuple[str, ...] = (
    "CraftSpear", "CraftArmour", "FightAnimal", "FleeAnimal",
    "ThrowSpear", "RetrieveSpear", "FireBfg",
    "FetchRelic", "RaiseTheDead",
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

# ------------------------------------------------------------- throwing --
#: Seconds spent lining a throw up before the spear leaves the hand. Long
#: enough that a throw is a visible act with a pose rather than a teleporting
#: number, short enough that a wolf at CHILD_THROW_MIN_GAP (72 px, ~1.6 s of
#: wolf) does not arrive mid-wind-up.
THROW_WINDUP = 0.45
#: How far an unarmed colonist will walk to fetch a spear off the ground.
#: Deliberately RESCUE_RADIUS and not further: the reason rescues were cut from
#: 640 px is that a long cross-map walk goes over cliff faces and every ledge
#: step carries a slip chance. Fetching a spear is worth exactly as much risk.
RETRIEVE_SEEK = RESCUE_RADIUS
RETRIEVE_TIMEOUT = 45.0
#: A child throws only while the nearest hostile is at least this far off. A
#: wolf makes ~44 px/s, so 72 px is about 1.6 s - the wind-up plus a head start.
#: Closer than this and a child runs, which is still the right answer.
CHILD_THROW_MIN_GAP = 72.0
#: Child range multiplier, and the reason a child stands behind the adults.
CHILD_RANGE_FRAC = 0.6
#: May an already-armed colonist walk over and collect another spear?
#:
#: ONE SWITCH, ON PURPOSE, because this is the re-phasing half of the quiver.
#: Nothing here adds or removes a scored candidate - ``RetrieveSpear`` is in
#: ``score_throw``'s out-dict either way - but it goes **positive on ticks it
#: used to be exactly 0.0**, and ``behavior.choose_action`` draws one
#: ``rng.uniform(0, TIEBREAK)`` off ``world.pyrng`` per candidate scoring above
#: zero. One extra draw shifts every downstream consumer of that stream: wolf
#: spawns, vignettes, role assignment, planting. The last inert candidate added
#: to it moved mean deaths 11.33 -> 13.17 with 0 of 24 seeds matching.
#:
#: So it is a module global rather than an inlined ``if``, for exactly the
#: reason ``behavior.UNDER_ATTACK_OVERRIDE`` is one: a harness can set this in
#: one process and the other arm in another and compare the two on one seed.
#: Set it False and :func:`_pickup_blocked` is the shipped single-slot rule,
#: character for character.
#:
#: NOTE it is orthogonal to how many spears fit. Until ``throwing`` grows the
#: quiver, ``can_carry_more`` is "not holding one" and this flag changes
#: nothing at all; the count and the relaxation are two separate measurements
#: and must not be stacked into one.
PICKUP_WHEN_ARMED: bool = True

# ------------------------------------------------------------------- bfg --
#: How many candidates deep the BFG scanner looks before giving up. Each one
#: costs a full corridor sweep (BFG_RANGE / _STEP_PX = 105 samples), and this
#: runs once per scoring call per wielder, so it is bounded on purpose. Four is
#: past every realistic engagement: throw_targets is sorted nearest-first and
#: the whole point of the weapon is that it does not need to pick carefully.
BFG_SCAN = 4
#: How long the beam stays on screen, and how many are kept. BOTH ARE NOW DEAD
#: NUMBERS and are left here only so nobody re-derives them: the lifetime is
#: ``items.BEAM_LIFE_SEC`` and the cap is ``items.MAX_BEAMS``, because
#: ``RelicRegistry`` owns the list render reads and the clock that ages it. Sim
#: had its own copy of both, aged neither, and stamped an absolute world time
#: into a field render reads as an age - see :func:`_record_beam` for what that
#: cost. One owner for the beam's clock, and it is not this file.
BFG_BEAM_DRAW_SEC = 0.25            # informational; items.BEAM_LIFE_SEC decides
_BFG_BEAM_KEEP = 8                  # informational; items.MAX_BEAMS decides
#: Right-hand world edge, resolved once. ``actions._clamp_x`` is the one place
#: that knows how wide the world is.
_EDGE_MAX_X = _clamp_x(1e9)

# ----------------------------------------------------------------- relics --
#: How close is close enough to lift a relic. ``RELIC_PICKUP_R`` from
#: constants.py, named locally only so ``step_toward(arrive=...)`` reads.
FETCH_ARRIVE = RELIC_PICKUP_R
#: How far a colonist will walk for one. Deliberately ``RELIC_FETCH_RANGE``
#: (520) and not ``RETRIEVE_SEEK`` (260, what a spear is worth): a spear can be
#: re-made from wood and stone, and a dragon relic drops about once every four
#: raw colony-hours and rots at ``RELIC_DECAY_SEC``. It is the same number
#: ``behavior.RELIC_FETCH_RANGE`` scores against, and it has to be, or the
#: scorer wants something the maker refuses to build.
FETCH_SEEK = RELIC_FETCH_RANGE
#: No relic is worth walking into a wolf for.
#:
#: This number was written and justified in behavior.py, as RELIC_SAFE_RADIUS,
#: by the lane that scored the appetite - and there it sat next to a *second*
#: copy of the gate it guards. It lives here now, once, because the gate belongs
#: with the walk: behaviour decides how badly somebody wants the relic, this
#: file decides whether setting off is survivable, and the scorer asks this file
#: (see :func:`fetchable_relic`). Two implementations of one rule in two files
#: owned by two lanes is the shape of the bug this whole change exists to undo.
#:
#: Checked at BOTH ends - where the walker is standing and where the relic lies
#: - because the second is the one that actually kills: a fetcher who sets off
#: from a quiet corner of the map and arrives next to the pack has run *toward*
#: the animal, which is the exact opposite of what FleeAnimal spent its whole
#: design budget on. Wider than animals.DANGER_RADIUS (110) so the margin
#: survives the walk.
FETCH_SAFE_RADIUS = 200.0
#: 520 px at WALK_SPEED (34 px/s) is ~15 s on the flat; this is four times that,
#: because the walk is over terrain and ``step_toward`` detours round ledges.
#: Not open-ended: see FETCH_SNUB_SEC.
FETCH_TIMEOUT = 60.0
#: Seconds an agent leaves a particular relic alone after failing to reach it.
#:
#: The known failure this exists for, diagnosed on the worst seed of the
#: appetite's own measurement run: a colony pacing back and forth against an
#: unclimbable face at x~1110, trying to reach a relic at 1301. A blocked agent
#: fails its goal after ``actions.BLOCKED_GIVE_UP`` (2.0 s), re-scores, and the
#: relic - whose score is still *climbing* with age - simply wins again. 153
#: starts on one seed, all of them into the same wall.
#:
#: The appetite comment correctly says this is not a bystander problem and that
#: RELIC_BYSTANDER must not try to fix it. It is a *goal* problem, so the memory
#: of having failed lives with the goal, here. One minute is long enough that
#: the agent gets a real amount of other work done and short enough that a relic
#: made briefly unreachable by a wandering wolf is not written off.
#:
#: Transient, in ``agent.__dict__``, exactly like ``_bfg_next``: a save that
#: reloaded mid-snub would resume a grudge nobody can see, and forgetting it
#: costs one wasted walk.
FETCH_SNUB_SEC = 60.0
#: How many of those a single agent remembers. Bounded because MAX_RELICS is 6
#: and a colonist that snubbed every relic in the world should still be holding
#: six entries, not a leak.
_SNUB_KEEP = 12
#: What the cairnstone bearer will cross to reach a headstone. Same seek and the
#: same timeout as the fetch: the walk is the same walk.
RAISE_SEEK = RELIC_FETCH_RANGE
RAISE_TIMEOUT = 60.0
#: A beat of kneeling at the stone before anybody stands up. Same purpose as
#: THROW_WINDUP - the rarest thing in the game should be a visible act with a
#: pose, not a number that changes between frames. Longer than the throw because
#: nothing is about to eat the bearer: raisable_grave refuses within
#: FETCH_SAFE_RADIUS of anything hunting.
RAISE_PLACE_SEC = 1.6

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


def dragons_of(world: Any) -> list[Any]:
    """Living dragons, or [] on a world that has no dragon subsystem.

    Kept separate from :func:`animals_of` on purpose. Dragons must NOT flow into
    ``nearest_animal``, because that is what ``score_combat`` flees from and
    what blocks the workbench: a dragon cruising overhead would put the whole
    colony into permanent flight and stop it ever making another spear. The one
    place they join the animals is :func:`melee_foes` - the armed agent's quarry
    list - and :func:`throwing.throw_targets`.
    """
    src = getattr(world, "dragons", None)
    if src is None:
        return []
    if isinstance(src, (list, tuple)):
        return [d for d in src if animal_alive(d)]
    fn = getattr(src, "alive", None)
    if callable(fn):
        try:
            return [d for d in fn() if animal_alive(d)]
        except Exception:
            return []
    return []


def melee_foes(world: Any) -> list[Any]:
    """Everything an agent with a spear in hand could actually hit.

    Animals (always on the ground - animals.py's header says "nothing here flies
    or falls") plus any dragon low enough to reach. ``MELEE_CEILING`` is the
    whole test: a spearman must not stand under something 200 px up swinging at
    the air, and a grounded siege dragon must be a fair fight.
    """
    out = [a for a in animals_of(world) if animal_alive(a)]
    for d in dragons_of(world):
        try:
            if clearance(world, d) <= MELEE_CEILING:
                out.append(d)
        except Exception:
            continue
    return out


def _is_dragon(world: Any, foe: Any) -> bool:
    """True when *foe* came from the dragon registry rather than the animals.

    Ids are per-registry, so a dragon and a wolf can both be #3 - which means
    an action that remembers a bare target id cannot resolve it without knowing
    which side it came from. That is the whole reason this exists.
    """
    if foe is None:
        return False
    return any(d is foe for d in dragons_of(world))


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
        if not isinstance(stats, (list, tuple)) or len(stats) <= 3:
            # Not a kind this game skins. A dragon has no `hides` field and no
            # ANIMAL_STATS row, and the old fallback of 1 would have paid the
            # colony a leather hide for killing a wyvern on a world whose
            # animals are a bare list. Nothing to take.
            return 0
        n = stats[3]
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


def _reachable(world: Any, foe: Any) -> bool:
    """True when a spear held in the hand could touch *foe*.

    Every animal answers True - none of them fly. This only ever says no to a
    dragon at altitude, and it says it here, in the quarry chooser, so that
    FightAnimal is never even *scored* against something 200 px up. The same
    test runs again inside :func:`_h_fight`, because a target can climb after
    the fight has started.
    """
    try:
        return clearance(world, foe) <= MELEE_CEILING
    except Exception:
        return True


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
    # A dragon standing in the colony is a fight; one in the sky is not. It is
    # checked after the animals because a wolf chewing on somebody is always
    # the more urgent problem, and before the rescue sweep because a landed
    # dragon is closer to hand than a mauling across the camp.
    for d in dragons_of(world):
        if not _reachable(world, d):
            continue
        try:
            if abs(float(getattr(d, "x", ax)) - ax) <= FIGHT_RADIUS:
                return d
        except (TypeError, ValueError):
            continue
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
    """Holding a *spear*. Deliberately not "holding a weapon".

    FightAnimal swings at SPEAR_REACH and ``throwing.can_throw`` launches a
    spear, and neither may fire for somebody carrying a BFG - who has traded the
    spear away and cannot melee or throw at all. Widening this predicate is how
    a BFG carrier would end up walking into a wolf to hit it with a relic.
    Use :func:`_combat_capable` for "can hurt something", :func:`_bfg` for the
    gun.
    """
    return str(getattr(agent, "weapon", WEAPON_NONE) or WEAPON_NONE) == WEAPON_SPEAR


def _bfg(agent: Any) -> bool:
    return str(getattr(agent, "weapon", WEAPON_NONE) or WEAPON_NONE) == WEAPON_BFG


def _combat_capable(agent: Any) -> bool:
    """Can this agent hurt anything at all, by any means?

    The single most load-bearing predicate the relic work adds. Without it a
    BFG carrier reads as *unarmed* in the flee gate below, scores FleeAnimal at
    0.92-1.0, and runs away from every wolf it could have vaporised - carrying
    the rarest item in the game and using it for nothing.
    """
    return _armed(agent) or _bfg(agent)


def _pickup_blocked(agent: Any) -> bool:
    """May *agent* NOT collect another spear off the ground?

    The single gate behind the whole pickup half of the quiver, so the scorer
    (:func:`score_throw`), the maker (:func:`make_throw_action`) and the
    handler's own start phase cannot drift apart - which is the stated reason
    :func:`_retrievable_spear` exists at all.

    Two refusals, and they are different in kind:

    * **The wyrm-gun, always.** ``_h_retrieve``'s take phase ends by arming the
      agent with a spear, and ``agent.weapon`` is one slot, so a BFG carrier
      reading as "has room" would trade the rarest item in the game for a stick
      in the dirt. This one is not behind the flag and never will be.
    * **A full quiver.** ``throwing.can_carry_more`` owns the number. With
      ``PICKUP_WHEN_ARMED`` off this collapses back to plain ``_armed``, which
      together with the line above is ``_combat_capable`` exactly - the
      predicate this used to be.
    """
    if _bfg(agent):
        return True
    if not PICKUP_WHEN_ARMED:
        return _armed(agent)
    return not can_carry_more(agent)


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


def _can_afford(world: Any, cost: dict[str, int], agent: Any = None) -> bool:
    """Can *agent* pay this bill out of the store they actually draw on?

    WHOSE STORE, and the answer stopped being "the colony's" the moment
    `actions.stock_take` grew its till gate. A hermit's `_pay` is billed to his
    own camp stash, so asking the colony's stockpile here would let
    `score_combat` promise a craft the workbench cannot deliver: he scores
    CraftSpear at 0.87 off the village's woodpile, walks the whole standoff,
    `_pay` finds his stash empty, the action fails, and the score is still 0.87
    on the next tick. That is a hermit pacing to the workbench and back for the
    rest of the run. The standing rule for every score in `_hermit_bias` is that
    the score and the handler must ask the SAME question; this is that rule
    reaching across into combat.

    `agent=None` keeps the colony's answer, so every caller that has not got an
    agent in hand - and every colonist who is not a hermit - draws exactly what
    it drew before.
    """
    try:
        qty_of = (hermit_stash_qty if (agent is not None and is_hermit(agent))
                  else stock_qty)
        return all(qty_of(world, str(res)) >= int(qty)
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
    """Does *agent* already have what this craft would give them?

    ``_combat_capable`` and not ``_armed`` for the spear, and that one word is
    a silent data-loss bug: ``agent.weapon`` is a single slot, so a colonist who
    found the wyrm-gun and then decided to whittle a spear had
    ``weapon = WEAPON_SPEAR`` written straight over ``WEAPON_BFG``. The relic
    field still read ``bfg9000``, so ``items.worn()`` said they were carrying it
    while ``_bfg()`` said they were not: the rarest item in the game gone, no
    chronicle line, no drop on the ground, and nothing anywhere that could
    notice. It is latent today only because no BFG has ever dropped in natural
    play; it reproduces in two lines the moment one does.

    Saying "satisfied" rather than adding a fourth refusal is deliberate. It is
    the truthful answer - a colonist holding a weapon that vaporises wolves does
    not need a sharpened stick - and it closes all three doors at once, because
    ``score_combat``, ``make_combat_action`` and :func:`_h_craft` all ask this
    same question. :func:`_craft_apply` refuses as well, for a save rehydrated
    mid-craft that never passes back through any of them.
    """
    return _combat_capable(agent) if kind == "CraftSpear" else _armoured(agent)


def _craft_apply(kind: str, agent: Any) -> None:
    if kind == "CraftSpear":
        if _bfg(agent):
            return          # never over the wyrm-gun - see _craft_satisfied
        # Through the quiver's writer, not a bare ``agent.weapon =``. Once
        # ``spears`` exists, ``weapon`` is its DERIVED mirror and Stickman's
        # from_dict re-asserts that on every load: a craft that set the string
        # without the count would arm a colonist who silently disarms himself
        # the next time the game is saved and reloaded. Behaviour is unchanged
        # today - ``_craft_satisfied`` still refuses the craft to anyone already
        # holding one, so this only ever runs at a count of zero.
        carry_spear(agent)
        return
    try:
        cur = float(getattr(agent, "armour", 0.0) or 0.0)
    except (TypeError, ValueError):
        cur = 0.0
    agent.armour = max(cur, ARMOUR_LEATHER)


def _pay(world: Any, a: Action, cost: dict[str, int]) -> bool:
    """Spend `cost` atomically. Records the bill on the action so an abandoned
    craft refunds instead of quietly eating the colony's wood.

    WHOSE WOOD IS DECIDED AT THE TILL, NOT HERE, and that is why this function
    is unchanged apart from these five lines. It contains no literal
    ``stock_take(`` at the call sites people grep for - it is two calls into
    `actions` - which is exactly how the hermit's largest leak survived a static
    scan and had to be found by runtime attribution instead: `_h_craft` was
    buying his spear and his armour with the village's wood, stone, hide and
    fibre on all five measured seeds. `actions.stock_take`/`stock_add` now bill
    whichever agent's handler is running, so a hermit at the bench pays out of
    his own camp stash and an interrupted craft refunds back into it.
    """
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
        if not _can_afford(w, cost, ag):
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
def _acquire(a: Action, ag: Any, animal: Any, world: Any = None) -> None:
    a.target = animal_id(animal)
    a.data["ax"] = animal_x(animal, float(getattr(ag, "x", 0.0)))
    # Which registry the id belongs to. Without it a saved fight against
    # dragon #3 resolves to wolf #3 on reload - see :func:`_is_dragon`.
    if world is not None and _is_dragon(world, animal):
        a.data["foe"] = "dragon"
    else:
        a.data.pop("foe", None)
    _set_target(ag, animal)


def _dragon_by_id(world: Any, did: Any) -> Any | None:
    if did is None:
        return None
    try:
        want = int(did)
    except (TypeError, ValueError):
        return None
    for d in dragons_of(world):
        if animal_id(d) == want:
            return d
    return None


def _current_target(a: Action, w: Any) -> Any | None:
    if a.data.get("foe") == "dragon":
        d = _dragon_by_id(w, a.target)
        return d if d is not None and animal_alive(d) else None
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
        _acquire(a, ag, animal, w)

    # It went up. A stickman cannot swing at something above head height plus a
    # spear, and standing underneath one waiting is how an agent spends its
    # whole life in a fight it is not having. Break off; ThrowSpear is scored
    # the same tick and is the answer to a flyer.
    if not _reachable(w, animal):
        _fight_finish(a, ag)
        return

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
        _acquire(a, ag, nxt, w)
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
#  ThrowSpear / RetrieveSpear
#
#  The projectile itself lives in throwing.py. Everything here is about who
#  decides to throw, at what, and what they do about the spear afterwards.
# ===========================================================================
def _throw_range(agent: Any) -> float:
    return THROW_MAX_RANGE * (CHILD_RANGE_FRAC if _role(agent) == "child" else 1.0)


def _ranged_scan(agent: Any, world: Any, reach: float, min_range: float,
                 accept: Callable[[Any, float, float], bool],
                 scan_cap: int = 0) -> Any | None:
    """The candidate walk every ranged weapon shares. Nearest first, first yes.

    This is the common half of :func:`_throw_target` and :func:`_bfg_target`,
    which were written independently and came out the same shape line for line:
    one ``throw_targets`` sweep, the three cheap filters (too close, already
    leaving, not hostile), and the weapon's own expensive test dead last.
    Keeping two copies of that meant a new weapon - or a fix to one of the three
    filters - was a coin toss about whether it landed in both.

    *accept* is the weapon's own tail, ``(foe, foe_x, gap) -> bool``. It is
    called only for candidates that survived the cheap filters, and the first
    one it accepts is the answer.

    *scan_cap* bounds how many survivors get that far, for a weapon whose tail
    is expensive enough to need bounding (the BFG's is a full corridor sweep;
    the spear's is one ``flight_clear``). Zero means no bound, which is what the
    spear has always had - the counter still runs, it just never trips, so this
    is arithmetic and not a behaviour change.

    Order is load-bearing and matches both originals exactly: the cap is tested
    BEFORE the candidate is examined, and the counter is incremented AFTER the
    cheap filters and BEFORE *accept*. ``throw_targets`` is sorted by distance
    with a stable ``n`` tiebreak, so "first accepted" is reproducible across a
    save and its reload.
    """
    try:
        ax = float(getattr(agent, "x", 0.0))
    except (TypeError, ValueError):
        return None
    seen = 0
    for foe in throw_targets(world, ax, reach):
        if scan_cap and seen >= scan_cap:
            break
        try:
            fx = float(getattr(foe, "x", ax))
        except (TypeError, ValueError):
            continue
        gap = abs(fx - ax)
        if gap < min_range:
            continue
        if animal_leaving(foe):
            continue
        if getattr(foe, "hostile", True) is False:
            continue
        seen += 1
        if accept(foe, fx, gap):
            return foe
    return None


def _throw_target(agent: Any, world: Any, *, perched: bool = False) -> Any | None:
    """What this agent should throw at, or None. The scorer and the action's
    start phase both go through here, so an agent never commits to a throw it
    would not have scored.

    The rules, in the order they matter:

    * Nothing inside ``THROW_MIN_RANGE`` - that is thrusting distance, and a
      thrust does not cost you the spear.
    * Nothing that is already leaving. Spending a spear on a wolf walking off
      the map is how a colony ends up unarmed for the next wave.
    * A **child** throws at anything hostile from ``CHILD_THROW_MIN_GAP`` out.
    * An **adult** throws only while **nothing at all is inside**
      ``FLEE_RADIUS`` of them. That single gate is the whole doctrine: you
      throw at what is still far enough away that you are not yet in danger,
      and you keep the point in your hand for anything that has closed. It also
      makes the throw the *opener* and the thrust the *finisher*, which is the
      order the two damage numbers were tuned for.

    The first version of this rule was narrower - adults threw only past
    ``FIGHT_RADIUS`` (180), leaving a 60 px band between that and
    ``THROW_MAX_RANGE``. Measured over seed 2001 x 900 sim-s: 9800
    choose_action calls, 63 of them with an animal inside 240 px, and only 11
    that scored a throw above zero. Two throws in fifteen minutes is not a
    mechanic, it is a rumour. The band is now 110-240 px.

    ``perched`` - the one exemption, and where it may come from
    -----------------------------------------------------------
    A man on a tower deck is not "under personal attack" in the sense the adult
    doctrine means, because the thing at the foot of his ladder is not between
    him and anywhere. Left as it stands, the FLEE_RADIUS refusal is the reason a
    lookout with a wolf 60 px below him does not throw: elevation makes throwing
    strictly *worse* today, in this one line.

    It is a keyword with a False default on purpose. ``score_throw`` and
    ``make_throw_action`` call this with the default, so **the scored value of
    ThrowSpear is unchanged for every agent in the game, perched or not**, and
    no scored candidate crosses zero. Only :func:`ranged_target` - the seam the
    tower's own attack phase fires through, which is not part of
    ``choose_action`` at all - passes True. That is what keeps the whole
    fire-from-a-lookout feature off ``world.pyrng``.
    """
    try:
        ax = float(getattr(agent, "x", 0.0))
    except (TypeError, ValueError):
        return None
    child = _role(agent) == "child"

    if not child and not perched:
        # Under personal attack: keep the spear in your hand. Note this also
        # implies every candidate below is at least FLEE_RADIUS away, which is
        # why there is no second distance test for adults.
        if nearest_animal(world, ax, max_dist=FLEE_RADIUS) is not None:
            return None

    def accept(foe: Any, fx: float, gap: float) -> bool:
        if child:
            # Too close to spend the wind-up on. Run instead.
            return gap >= CHILD_THROW_MIN_GAP
        # Dragons are not in nearest_animal (deliberately - see dragons_of), so
        # the FLEE_RADIUS pre-check above did not see this one. Apply it here -
        # and skip it from a deck, for the same reason the pre-check is skipped.
        if not perched and _reachable(world, foe) and gap < FLEE_RADIUS:
            return False
        # Last, because it is the expensive one: is there actually an arc from
        # here to there that misses the hillside? See throwing.flight_clear -
        # without this, terrain ate four throws in five.
        x0, y0 = launch_point(agent, fx)
        return flight_clear(world, x0, y0, foe)

    return _ranged_scan(agent, world, _throw_range(agent), THROW_MIN_RANGE,
                        accept)


def _h_throw(a: Action, ag: Any, w: Any, dt: float) -> None:
    if a.phase == "start":
        if not can_throw(ag, w):
            a.failed = True
            return
        foe = _throw_target(ag, w)
        if foe is None:
            a.failed = True
            return
        a.target = animal_id(foe)
        if _is_dragon(w, foe):
            a.data["foe"] = "dragon"
        a.data["ax"] = animal_x(foe, float(getattr(ag, "x", 0.0)))
        a.phase = "aim"
        a.data["wt"] = 0.0

    if a.phase == "aim":
        # "chop" is the overhead swing pose. There is no throw pose in
        # actions.POSES yet and POSES is a closed tuple that render/ switches
        # on, so inventing a name here would draw nothing at all. The nearest
        # true thing is better than a silent blank.
        a.pose = "chop"
        _halt(ag)
        foe = _current_target(a, w)
        if foe is None or not animal_alive(foe):
            foe = _throw_target(ag, w)
            if foe is None:
                a.failed = True
                return
            a.target = animal_id(foe)
            a.data["foe"] = "dragon" if _is_dragon(w, foe) else None
            if a.data.get("foe") is None:
                a.data.pop("foe", None)
        fx = animal_x(foe, float(getattr(ag, "x", 0.0)))
        a.data["ax"] = fx
        _face(ag, fx - float(getattr(ag, "x", 0.0)))
        a.data["wt"] = float(a.data.get("wt", 0.0)) + dt
        if float(a.data["wt"]) < THROW_WINDUP:
            return
        a.phase = "release"

    if a.phase == "release":
        reg = ensure_registry(w)
        foe = _current_target(a, w)
        if reg is None or foe is None:
            a.failed = True
            return
        sp = reg.throw(w, ag, foe)
        if sp is None:
            # Out of range now, or the solve had no answer. Not an error - it
            # is a shot not taken, and the agent still has the spear.
            a.failed = True
            return
        _adjust(ag, "fatigue", 0.010)
        a.data["thrown"] = 1
        _clear_target(ag)
        a.done = True
        return

    a.phase = "start"


def _c_throw(a: Action, ag: Any, w: Any) -> None:
    _clear_target(ag)


def _retrievable_spear(agent: Any, world: Any) -> Any | None:
    """The spear this agent should go and fetch, or None.

    Shared by the scorer and the maker so the two cannot disagree. Two gates
    beyond "there is one": the walk has to be short (see RETRIEVE_SEEK), and
    nothing may be standing over it. Fetching a spear from under a wolf is how
    an unarmed colonist walks into the one place on the map they must not be.

    The first gate is :func:`_pickup_blocked` and not ``_combat_capable``. It
    used to be the latter, and that one word was the entire reason an armed
    colonist would step over a spear he had just thrown: "can you hurt
    anything?" is not the same question as "have you room for another?". The
    wyrm-gun clause survives inside it verbatim - constants.py's note on
    WEAPON_BFG names both paths and this is the second one - because
    ``_h_retrieve`` ends by arming the agent with a spear.
    """
    try:
        if _pickup_blocked(agent):
            return None
        ax = float(getattr(agent, "x", 0.0))
        if nearest_animal(world, ax, max_dist=FLEE_RADIUS) is not None:
            return None
        reg = spear_registry_of(world)
        if reg is None:
            return None
        sp = reg.nearest_on_ground(ax)
        if sp is None:
            return None
        if abs(float(sp.x) - ax) > RETRIEVE_SEEK:
            return None
        if nearest_animal(world, float(sp.x), max_dist=FLEE_RADIUS) is not None:
            return None
        return sp
    except Exception:
        return None


def _h_retrieve(a: Action, ag: Any, w: Any, dt: float) -> None:
    reg = spear_registry_of(w)
    if reg is None:
        a.failed = True
        return

    if a.phase == "start":
        if _pickup_blocked(ag):
            a.done = True                   # full, or holding the wyrm-gun
            return
        sp = reg.nearest_on_ground(float(getattr(ag, "x", 0.0)))
        if sp is None:
            a.failed = True
            return
        a.target = int(sp.id)
        a.phase = "walk"

    if a.phase == "walk":
        a.pose = "walk"
        sp = reg.get(a.target)
        if sp is None or sp.in_flight:
            # Somebody else got there first, or it was never really down.
            sp = reg.nearest_on_ground(float(getattr(ag, "x", 0.0)))
            if sp is None:
                _halt(ag)
                a.failed = True
                return
            a.target = int(sp.id)
        rem = step_toward(ag, w, float(sp.x), dt, arrive=RETRIEVE_RADIUS)
        if rem <= RETRIEVE_RADIUS:
            a.phase = "take"
        elif a.t > RETRIEVE_TIMEOUT:
            _halt(ag)
            a.failed = True
        return

    if a.phase == "take":
        _halt(ag)
        if _bfg(ag):
            # Never over the wyrm-gun, and checked here as well as in
            # _retrievable_spear because a save rehydrated at this phase never
            # passes through "start". The spear stays where it is.
            a.done = True
            return
        sp = reg.get(a.target)
        if sp is None or not reg.take(sp):
            a.failed = True
            return
        before = spears_carried(ag)
        try:
            got = carry_spear(ag)
        except Exception:
            got = before
        if got <= before:
            # THE COUNT DID NOT GO UP and the spear is already off the ground:
            # reg.take() removed it. Checked on the return value and not only
            # on an exception, because ``throwing.carry_spear`` logs a write
            # failure and hands back the OLD total rather than raising - so a
            # bare try/except here would catch nothing and delete the spear out
            # of the world. Put it back instead: a spear that vanishes costs
            # the colony wood and stone it has no way of knowing it lost.
            #
            # This also covers a full quiver, which start-phase would have
            # refused but a save rehydrated straight into "take" never saw.
            reg.add(sp)
            a.failed = True
            return
        _bump_stat(w, "spears_recovered")
        emit_speech(w, ag, "^")
        a.done = True
        return

    a.phase = "start"


# ===========================================================================
#  FireBfg - the wyrm-gun
#
#  One action, three behaviours, and they are one mechanic rather than three:
#
#  * LINE KILL. A ray from the muzzle to the aim point, clipped at BFG_RANGE
#    and clipped again at the first terrain intersection. TERRAIN stops the
#    beam; a body does NOT. Everything whose hitbox overlaps the corridor takes
#    BFG_DAMAGE - animals, dragons and colonists in the same loop, which is why
#    "wipes out animals" and "kills everything in line of sight" need one
#    implementation and not two.
#  * MISFIRE. Rolled BEFORE the ray exists, p = BFG_MISFIRE. No ray, and no
#    blast: it takes the WIELDER ALONE, and the relic is destroyed outright. It
#    is the only item in the set that can leave the world, and that asymmetry is
#    its balance valve. It was a 70 px sphere that caught the neighbours; see
#    _bfg_misfire for the author's correction and why the two outcomes had to
#    stop being the same outcome.
#  * ENGAGEMENT. Hold fire when a colonist is standing in the corridor, except
#    on a BFG_HEEDLESS roll. THIS is the one that can leave a single stickman
#    standing, and it is meant to.
#
#  WHY THE CORRIDOR NEVER HELD A CROWD. Shipped, this weapon had killed at most
#  ONE colonist per shot in 246 forced shots. Three explanations were on the
#  table and, measured on 381 paired shot geometries from twelve real colonies,
#  all three are wrong:
#
#  * "BFG_CORRIDOR (22 px) is too narrow." It is not. The sweep tests a HITBOX
#    against the band, so the effective vertical tolerance is already ~35 px.
#    Taking the half-width from 22 px to 120 px moves the mean number of
#    colonists in the beam from 1.00 to 1.08. Width was never the lever.
#  * "The shot re-aims, or declines, to spare friendlies." It never re-aims -
#    _bfg_target does not look at colonists at all. It DECLINES, on purpose,
#    4 times in 5 (see BFG_HEEDLESS below), and that is the balance valve.
#  * "BFG_HEEDLESS (0.20) is simply rare." It is rare by design, and it is a
#    multiplier on the census, not the reason the census was empty.
#
#  The cause was in _beam_stop. Every other part of the mechanic treats the shot
#  as a lane BFG_CORRIDOR px to either side - the sweep, the beam record, the
#  stroke render draws - and _beam_stop alone treated it as a HAIRLINE, ending
#  the beam at the first sample where its axis met the ground, with the whole
#  upper half of the beam still in clear air. A colony sits on rolling terrain,
#  so a shot from a 20 px shoulder died at the first 20 px rise. Median reach
#  was 116 px of a nominal 420. The corridor was not narrow; it was short.
#
#  Hairline -> corridor, same 381 geometries:
#
#      median reach            116 px  ->  248 px     full range  15.2% -> 34.9%
#      colonists in the beam   mean 0.63  ->  0.92
#        two or more                17.1%  ->  23.1%
#        three or more               3.4%  ->   8.1%
#        most ever in one line          4  ->      8
#
#  A heedless shot is by definition one with a colonist already in the lane, so
#  the distribution the author asked about - "could possibly leave a single
#  stickman" - is that census conditioned on >= 1: one 57.7%, two 27.4%, three
#  9.1%, four to eight 5.8%; mean 1.69. Rare, because somebody has to be in the
#  lane AND the 0.20 roll has to come up; catastrophic when it lands.
#
#  END TO END, 12 seeds x 60 sim-min of real colonies with the gun handed to a
#  living colonist whenever nobody is carrying one and every roll after that off
#  the real seeded stream. Same seeds both arms:
#
#  ==========================================  ==========  ==========
#                                                hairline    corridor
#  ==========================================  ==========  ==========
#  shots fired                                         39          40
#    of which heedless                                  3           9
#    held (a friendly in the lane)                     14          30
#  colonists killed by one heedless shot        1:1 2:2     1:5 2:1 4:2 5:1
#    mean                                            1.67        2.22
#    MOST EVER KILLED BY ONE SHOT                       2           5
#  shots that left the colony at <= 2 alive             1           4
#    at exactly one stickman standing                   0           2
#  animals killed per fired shot                     1.05        1.40
#  misfires                                             5           7
#    that killed anyone but the wielder                 0           0
#  ==========================================  ==========  ==========
#
#  So the weapon now does the thing it was asked for - a line that clears, and
#  twice in twelve colony-hours a single shot that leaves one stickman standing
#  - without becoming routine: 31 of 40 shots were the ordinary aimed kind and
#  killed no colonist at all, and the engagement gate held fire 30 times.
#  Across all four measurement runs 22 misfires killed 22 people: the wielder,
#  every time, and nobody else.
#
#  Why BFG_DAMAGE is MAX_HEALTH * 9.0 and NOT dragons._devour's * 10.0: through
#  the 0.9 armour clamp in Stickman.hurt, an unarmoured colonist takes 900 and
#  dies, leather takes 495 and dies, and DRAGONSCALE takes 90 against 100 health
#  and walks out of the beam at 10 hp. At 10x the scale wearer takes exactly 100
#  and the one interaction the armour exists for silently disappears. Anyone
#  tuning either constant has to check the other.
#
#  Why the engagement gate is not squeamishness: measured over 8 colony-hours of
#  real geometry, 55.5% of shot opportunities have at least one colonist in the
#  corridor (mean 1.129 in line, mean 1.915 within the blast radius). At
#  heedless = 1.0 the wielder kills 6.12 of its own colony per hour against a
#  baseline of 11.75 deaths/hour, which ends every colony inside the hour. At
#  the chosen 0.20 it is 1.64 friendly deaths per colony-hour of carry, +14%,
#  and one single-shot colony wipe per 59.7 colony-hours of carry. That survey
#  also measured mean 1.915 colonists inside the old blast radius; the misfire
#  no longer damages them, so those are no longer deaths.
# ===========================================================================
def _relic_roll(world: Any) -> float | None:
    """One draw off ``world.relics.roll()``, or None when there is no stream.

    **Every** relic decision in the feature comes from here and from nowhere
    else. Not ``world.pyrng``: ``behavior.choose_action`` draws from that once
    per scored candidate, and adding a single inert candidate to it moved mean
    deaths 11.33 -> 13.17 (+16%) with 0 of 24 seeds matching on any metric. Not
    ``world.spears._rng``, or a misfire re-phases every spear throw. Not bare
    ``random`` and never ``hash()`` of a string - this project has shipped that
    bug seven times.

    No stream means no shot: a BFG only exists because a RelicRegistry put it in
    somebody's hands, so a world without one cannot legitimately be holding the
    gun, and inventing a substitute stream here would be the eighth.
    """
    fn = getattr(getattr(world, "relics", None), "roll", None)
    if not callable(fn):
        return None
    try:
        v = float(fn())
    except Exception:
        log.debug("relic roll failed", exc_info=True)
        return None
    return v if math.isfinite(v) else None


def _bfg_ready_at(agent: Any) -> float:
    try:
        return float(agent.__dict__.get("_bfg_next", 0.0) or 0.0)
    except Exception:
        return 0.0


def _bfg_ready(agent: Any, world: Any) -> bool:
    """True when the gun has cooled.

    Kept off ``attack_cd``, which :func:`_cooldown` clamps to SPEAR_COOLDOWN
    (1.1 s) - eight times too short for this - and stored as an absolute world
    time rather than a counter, because nothing ticks between actions. It is
    transient by choice: it does not ride on Stickman.to_dict (that file belongs
    to another workstream), so a reload grants one free shot. A 9 s edge on a
    weapon that fires 3.3 times an hour is not worth an entity field.
    """
    return world_now(world) >= _bfg_ready_at(agent)


def _bfg_cool(agent: Any, world: Any) -> None:
    try:
        agent.__dict__["_bfg_next"] = world_now(world) + float(BFG_COOLDOWN)
    except Exception:
        pass


def _destroy_bfg(agent: Any) -> None:
    """The gun comes apart. It does NOT go on the ground.

    Called before the blast damage, always. If the wielder dies still holding
    it, ``World._reap_dead`` hands the corpse to ``relics.drop_from`` and the
    one item in the set that can leave the world quietly does not - the next
    colonist picks the same gun up off the scorch mark. Unequip first, then
    hurt people.

    This used to say ``from .relics import unequip`` - a module that has never
    existed under that name - inside a per-call try/except. It raised
    ImportError every single time, fell through to the by-hand clearing below,
    and looked like it worked because the by-hand clearing does the same job for
    the two mirrored fields. It did NOT do ``items.unequip``'s validation, and
    it hid a wrong import for the life of the feature. See the import block at
    the head of this file.
    """
    try:
        relic_unequip(agent)
        return
    except Exception:
        log.debug("items.unequip failed; clearing the slots by hand",
                  exc_info=True)
    for attr, val in (("weapon", WEAPON_NONE), ("relic", RELIC_NONE)):
        try:
            setattr(agent, attr, val)
        except Exception:
            pass


def _scorch(world: Any, x: float) -> None:
    """Burn scar where the gun let go. Lazy import - props.py pulls in numpy."""
    try:
        from .props import place_scorch                  # noqa: PLC0415
    except Exception:
        return
    try:
        props = getattr(world, "props", None)
        terrain = getattr(world, "terrain", None)
        if props is None or terrain is None:
            return
        place_scorch(props, terrain, float(x))
    except Exception:
        log.debug("bfg scorch failed", exc_info=True)


def _record_beam(world: Any, x0: float, y0: float, x1: float, y1: float, *,
                 misfire: bool = False) -> None:
    """Hand render the shot. Transient, capped, self-expiring, unsaved.

    THE BEAM WAS NEVER DRAWN, and this is where it went wrong. This function
    used to append to ``world.bfg_beams`` with ``"t": world_now(world)`` - an
    ABSOLUTE world time. ``render/items._draw_beam_record`` reads the same field
    as an AGE in seconds and discards anything past ``BEAM_LIFE_SEC``. On a
    world older than a quarter of a second - i.e. every world - the record was
    born already expired. The gun killed 0.72 colonists a shot and the viewer
    saw people fall over for no reason at all.

    Two things were wrong and both are fixed by handing the record to
    ``RelicRegistry.add_beam`` instead of writing it here:

    * ``add_beam`` stamps ``t = 0.0`` and ``RelicRegistry._age_beams`` advances
      it once a frame off the same clock that ages the relics. Age, not
      timestamp, and there is now exactly one place that decides which.
    * ``render._iter_beams`` looks at ``world.relics.beams`` FIRST and only
      falls back to ``world.bfg_beams``. The registry already owned the list
      render prefers; sim was writing to the one it checks last.

    No ``world.bfg_beams`` fallback any more, and not for tidiness: a shot only
    exists at all if :func:`_relic_roll` found ``world.relics.roll``, so a world
    that reaches here without a registry is unreachable. A hand-written fallback
    list would be a second ageing rule that nothing ticks - the same class of
    bug, one layer down.
    """
    try:
        reg = relic_registry_of(world)
        add = getattr(reg, "add_beam", None)
        if not callable(add):
            log.debug("no relic registry to hand the beam to")
            return
        add(float(x0), float(y0), float(x1), float(y1),
            misfire=bool(misfire), corridor=float(BFG_CORRIDOR))
    except Exception:
        log.debug("could not record the beam for render", exc_info=True)


def _beam_stop(world: Any, x0: float, y0: float, ux: float, uy: float,
               max_range: float, *, corridor: float = BFG_CORRIDOR) -> float:
    """How far the beam gets: *max_range*, or the first hillside.

    Sampled every ``_STEP_PX`` exactly as ``throwing.flight_clear`` samples a
    spear's arc, so the two cannot disagree about where the hill is.

    THE BEAM HAS WIDTH, AND THIS IS WHERE THE CORRIDOR WAS LOSING ITS CROWD.
    Every other part of the mechanic treats the shot as a lane ``corridor`` px
    to either side of the axis - :func:`beam_sweep` hits a hitbox that overlaps
    that band, ``items.add_beam`` carries the number, render draws a stroke that
    wide. This function alone treated it as a hairline: it returned the first
    sample where the AXIS met the ground, which cuts the shot off while its
    entire upper half is still in clear air. A colony sits on rolling terrain,
    so a shot from a shoulder 20 px up died at the first 20 px rise - and
    everyone standing past that rise, whose hitboxes the sweep would happily
    have caught, was simply out of range.

    Blocked now means the ground has risen above the TOP of the corridor, i.e.
    the hill fills the beam rather than merely touching it. Measured on 381
    paired shot geometries from twelve real colonies, hairline -> corridor:

    ============================================  ==========  ==========
                                                    hairline    corridor
    ============================================  ==========  ==========
    median reach (of a nominal 420 px)                116 px      248 px
    shots reaching full range                          15.2%       34.9%
    mean colonists in the beam                          0.63        0.92
    shot opportunities with two or more in it           17.1%       23.1%
    shot opportunities with three or more in it          3.4%        8.1%
    most ever caught by one line                           4           8
    ============================================  ==========  ==========

    *corridor* is a keyword because :func:`beam_sweep` passes its own - a sweep
    asked for a wider lane must get a correspondingly generous stop, or the two
    halves of one beam disagree about where it ends.
    """
    s = 0.0
    step = max(1.0, float(_BEAM_STEP))
    reach = max(0.0, float(max_range))
    slack = max(0.0, float(corridor))
    while s <= reach:
        x = x0 + ux * s
        y = y0 + uy * s
        if x < 0.0 or x > _EDGE_MAX_X:
            return s
        # Screen y grows downward, so `y - slack` is the corridor's upper edge.
        if y - slack >= _beam_ground_y(world, x):
            return s
        s += step
    return reach


def beam_sweep(world: Any, agent: Any, x0: float, y0: float,
               tx: float, ty: float, *,
               corridor: float = BFG_CORRIDOR,
               max_range: float = BFG_RANGE) -> list[tuple[Any, str]]:
    """Everything a straight beam from (x0,y0) toward (tx,ty) touches.

    Returns ``[(obj, side), ...]`` nearest first, where side is one of
    ``"animal"``, ``"dragon"``, ``"agent"``. The side tag is why this builds its
    own candidate list instead of duck-typing the result: an animal and a
    stickman both have ``hurt()``, and only one of them takes a death *cause*.
    Guessing wrong there is the difference between a colonist dying properly -
    grave, chronicle line, counted in ``stats["died"]`` - and a body that the
    ledger cannot explain.

    *agent* (the wielder) is excluded: the ray starts at their own shoulder, so
    they are inside the corridor at s = 0 by construction. The misfire is the
    path that is allowed to hurt them, and it does so explicitly.

    Non-hostile animals are swept like everything else. This is not a targeting
    system; it is a line.

    The contract names ``throwing.beam_sweep`` as the eventual home for this and
    that is the right place for it. It is here for now because this lane owns
    this file and not that one, and because the one requirement that cannot be
    negotiated - a colonist standing in the beam dies - has to be provable
    today. When the shared version lands, this should delegate to it *after*
    confirming it sweeps agents and not only animals.
    """
    out: list[tuple[float, int, Any, str]] = []
    try:
        dx = float(tx) - float(x0)
        dy = float(ty) - float(y0)
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return []
        ux, uy = dx / d, dy / d
        half = max(1.0, float(corridor))
        reach = _beam_stop(world, float(x0), float(y0), ux, uy, max_range,
                           corridor=half)
        if reach <= 0.0:
            return []

        cands: list[tuple[Any, str]] = []
        for a in animals_of(world):
            if animal_alive(a):
                cands.append((a, "animal"))
        for dgn in dragons_of(world):
            cands.append((dgn, "dragon"))
        for ag in _alive_agents(world):
            if ag is agent or getattr(ag, "taken", False):
                continue
            if getattr(ag, "inside", None) is not None:
                continue        # indoors is not in the line, whatever its x
            cands.append((ag, "agent"))

        n = 0
        for obj, side in cands:
            n += 1
            try:
                ox = float(getattr(obj, "x", None))    # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            # The x test FIRST, and the hitbox only for whatever survives it.
            # ``_beam_box`` reaches ``throwing.clearance``, which does a
            # ``from .altitude import clearance`` that fails in this checkout -
            # sim/altitude.py has never existed - and a failed import is a full
            # sys.path filesystem search every single call. Measured on this
            # repo's mount that is ~32 ms per lookup. Anything behind the muzzle
            # or past the terrain stop cannot be hit whatever shape it is, so it
            # must not pay for one.
            if abs(ux) < 1e-3:
                # Straight up or straight down. There is no x to solve for, so
                # the corridor becomes an x-band and the swept y-span the test.
                if abs(ox - float(x0)) > half:
                    continue
                top, bottom = _beam_box(world, obj)
                lo = min(float(y0), float(y0) + uy * reach) - half
                hi = max(float(y0), float(y0) + uy * reach) + half
                if bottom < lo or top > hi:
                    continue
                s = abs(((top + bottom) * 0.5) - float(y0))
            else:
                s = (ox - float(x0)) / ux
                if s < 0.0 or s > reach:
                    continue
                top, bottom = _beam_box(world, obj)
                line_y = float(y0) + uy * s
                if bottom < line_y - half or top > line_y + half:
                    continue
            # n is a stable tiebreaker, exactly as in throw_targets: two targets
            # at the same distance must not order by object identity, which
            # differs between a world and its reload.
            out.append((s, n, obj, side))
        out.sort(key=lambda t: (t[0], t[1]))
    except Exception:
        log.debug("beam_sweep failed", exc_info=True)
        return []
    return [(o, side) for _, _, o, side in out]


def _beam_ends(agent: Any, world: Any, target: Any) -> tuple[float, float, float, float]:
    """(x0, y0, tx, ty) for a shot from *agent* at *target*.

    The aim point is ``throwing._aim_y``, the centre of the target's hitbox, and
    it stays that way. Clamping the aim into the near edge of the box instead -
    "level the gun, take the flattest line that still connects" - was written
    and measured against this, on 381 paired shot geometries from twelve real
    colonies, and it is NOISE: mean colonists in the corridor 0.63 -> 0.66, and
    the fraction of shots reaching full range actually FELL, 15.2% -> 13.4%,
    because levelling points a shot at an uphill target *less* steeply and buries
    it in the slope sooner. The reach problem was real; the aim was not where it
    lived. See :func:`_beam_stop`.
    """
    ax = float(getattr(agent, "x", 0.0))
    tx = animal_x(target, ax)
    x0, y0 = launch_point(agent, tx)
    return (x0, y0, tx, _beam_aim_y(world, target))


def _bfg_target(agent: Any, world: Any) -> Any | None:
    """What this agent should shoot, or None. Scorer and action share it.

    The last gate is the sweep itself rather than a cheaper approximation of it,
    so a shot is never scored that the terrain would eat - and so the scorer and
    the trigger can never disagree about what is hittable.

    Note what is NOT here: any equivalent of the spear's FLEE_RADIUS refusal.
    The gun has no wind-up cost you can be robbed of and does not leave your
    hand, so there is nothing to keep back - which is why :func:`ranged_target`
    hands the perched exemption to the weapon's own picker rather than applying
    it centrally. Weapons differ; the walk over the candidates does not.
    """
    def accept(foe: Any, fx: float, gap: float) -> bool:
        x0, y0, tx, ty = _beam_ends(agent, world, foe)
        return any(o is foe for o, _ in beam_sweep(world, agent, x0, y0, tx, ty))

    return _ranged_scan(agent, world, BFG_RANGE, BFG_MIN_RANGE, accept,
                        scan_cap=BFG_SCAN)


def _bfg_hurt(world: Any, obj: Any, side: str, by: Any) -> bool:
    """One target, one hit. True if it died.

    A colonist goes through ``Stickman.hurt(amount, cause)`` so the death runs
    the ordinary path - armour clamp, ``die()``, the death hooks,
    ``World._reap_dead``, a grave, ``stats["died"]``. A beam death must be as
    countable as a wolf's.

    Animals and dragons go through :func:`hurt_animal`, which means a dragon
    that has not fed takes *nothing*: ``Dragon.hurt`` discards damage while
    ``sated`` is False. The BFG gets no exemption from the feeding rule and
    needs no special case to be denied one.
    """
    if side == "agent":
        try:
            return bool(obj.hurt(BFG_DAMAGE, CAUSE_DISINTEGRATED))
        except TypeError:
            try:
                return bool(obj.hurt(BFG_DAMAGE))
            except Exception:
                log.debug("could not disintegrate an agent", exc_info=True)
                return False
        except Exception:
            log.debug("could not disintegrate an agent", exc_info=True)
            return False
    killed = hurt_animal(world, obj, BFG_DAMAGE, by=by)
    if killed:
        _bump_stat(world, "animals_killed")
    return killed


def _bfg_misfire(agent: Any, world: Any) -> dict[str, Any]:
    """It comes apart in his hands, and it takes HIM. Nobody else.

    CORRECTED SPEC, in the author's words: *"randomly explodes taking out the
    stickmen" .. "stickman", just himself .. so its an or situation .. could
    possibly leave a single stickman*. Two outcomes, not one:

    * a **misfire** is the wielder's own death and nobody else's, and
    * a **heedless corridor shot** is the one that can wipe most of a colony and
      leave one person standing. That one is in :func:`_bfg_shoot` and stays
      exactly as lethal as it was.

    What this used to do: take every animal and every colonist within
    ``BFG_BLAST_R`` (70 px), which is roughly a hut's width and reliably caught
    the neighbours. That made the two outcomes the same outcome with a different
    radius, and it meant a misfire - a *failure* of the weapon - was the thing
    doing the mass casualties rather than the shot the wielder chose to take.

    ``BFG_BLAST_R`` is deliberately still imported. It is no longer a damage
    radius, but ``render/items.draw_misfire`` draws the flash at exactly that
    size and the two should not drift: the light says "70 px of that went up",
    and it did - it just no longer kills the man standing in it.
    """
    ax = float(getattr(agent, "x", 0.0))
    ay = float(getattr(agent, "y", 0.0))
    name = _name(agent)

    # FIRST, before anybody can die: see _destroy_bfg.
    _destroy_bfg(agent)
    _bump_stat(world, "bfg_misfires")
    chronicle(world, f"The wyrm-gun comes apart in {name}'s hands.")
    _scorch(world, ax)
    # Both ends the wielder's own position: render ignores the endpoint on a
    # misfire record and draws the flash instead, which is why add_beam's
    # docstring calls passing it twice "the honest call". Without this the only
    # thing on screen was a colonist falling over.
    _record_beam(world, ax, ay, ax, ay, misfire=True)

    dead_friends: list[str] = []
    if _bfg_hurt(world, agent, "agent", agent):
        dead_friends.append(name)
        chronicle(world, f"There is not enough of {name} left to bury.")
    return {"shot": "misfire", "kills": 0, "friends": dead_friends}


def _bfg_shoot(agent: Any, world: Any, target: Any) -> dict[str, Any]:
    """Pull the trigger. Returns a small summary, for the test and the log."""
    name = _name(agent)

    # The misfire is rolled BEFORE the ray is computed, so a misfiring shot has
    # no line at all and cannot be aimed out of.
    roll = _relic_roll(world)
    if roll is None:
        return {"shot": "nostream", "kills": 0, "friends": []}
    if roll < BFG_MISFIRE:
        return _bfg_misfire(agent, world)

    x0, y0, tx, ty = _beam_ends(agent, world, target)
    hits = beam_sweep(world, agent, x0, y0, tx, ty)
    friends = [o for o, side in hits if side == "agent"]
    heedless = False
    if friends:
        heed = _relic_roll(world)
        if heed is None or heed >= BFG_HEEDLESS:
            # Held. It still costs the cooldown: a hold that cost nothing would
            # be re-scored at 0.96+ on the very next tick and the wielder would
            # spend its life winding up a shot it never takes.
            _bfg_cool(agent, world)
            return {"shot": "held", "kills": 0, "friends": []}
        heedless = True

    _bfg_cool(agent, world)
    _bump_stat(world, "bfg_shots")
    dx, dy = tx - x0, ty - y0
    d = math.hypot(dx, dy) or 1.0
    reach = _beam_stop(world, x0, y0, dx / d, dy / d, BFG_RANGE)
    _record_beam(world, x0, y0, x0 + dx / d * reach, y0 + dy / d * reach)
    if heedless:
        # NOT in world.STAT_DEFAULTS, like "animals_killed" and the spear
        # counters and for the same reason: that dict is another file's, and its
        # own note says these belong in it whenever its owner next passes
        # through. Until then both of these are conjured on first use, so every
        # reader must use .get() - a fresh colony has no key, not a zero.
        _bump_stat(world, "bfg_heedless")
        chronicle(world, f"{name} does not wait for the line to clear.")
    else:
        chronicle(world, f"{name} looses the wyrm-gun. Nothing on that line is "
                         f"standing.")

    dead_friends: list[str] = []
    kills = 0
    for obj, side in hits:
        who = _name(obj) if side == "agent" else ""
        if _bfg_hurt(world, obj, side, agent):
            if side == "agent":
                dead_friends.append(who)
            else:
                kills += 1
    if dead_friends:
        _bump_stat(world, "bfg_friendly_deaths", len(dead_friends))
        chronicle(world, f"{name}'s shot did not care who was in front of it.")
        if len(dead_friends) > 1:
            chronicle(world, f"{len(dead_friends)} of them were in the light.")
        # The line the user asked for by name, and the only place in the game
        # that can print it. Counted AFTER the damage, so a wielder who is now
        # the last one standing counts themselves.
        if len(_alive_agents(world)) <= 1:
            chronicle(world, "One stickman is left standing.")
    if kills:
        _adjust(agent, "morale", 0.10)
    return {"shot": "fired", "heedless": heedless,
            "kills": kills, "friends": dead_friends}


def _h_fire_bfg(a: Action, ag: Any, w: Any, dt: float) -> None:
    if a.phase == "start":
        if not _bfg(ag) or not _bfg_ready(ag, w):
            a.failed = True
            return
        foe = _bfg_target(ag, w)
        if foe is None:
            a.failed = True
            return
        a.target = animal_id(foe)
        if _is_dragon(w, foe):
            a.data["foe"] = "dragon"
        a.data["ax"] = animal_x(foe, float(getattr(ag, "x", 0.0)))
        a.phase = "aim"
        a.data["wt"] = 0.0

    if a.phase == "aim":
        # "chop" for the same reason ThrowSpear uses it: actions.POSES is a
        # closed tuple that render/ switches on, so a new name here would draw
        # nothing at all.
        a.pose = "chop"
        _halt(ag)
        if not _bfg(ag):
            a.failed = True             # it came apart, or was taken off them
            return
        foe = _current_target(a, w)
        if foe is None or not animal_alive(foe):
            foe = _bfg_target(ag, w)
            if foe is None:
                a.failed = True
                return
            a.target = animal_id(foe)
            if _is_dragon(w, foe):
                a.data["foe"] = "dragon"
            else:
                a.data.pop("foe", None)
        fx = animal_x(foe, float(getattr(ag, "x", 0.0)))
        a.data["ax"] = fx
        _face(ag, fx - float(getattr(ag, "x", 0.0)))
        a.data["wt"] = float(a.data.get("wt", 0.0)) + dt
        if float(a.data["wt"]) < BFG_WINDUP:
            return
        a.phase = "fire"

    if a.phase == "fire":
        foe = _current_target(a, w)
        if foe is None:
            a.failed = True
            return
        out = _bfg_shoot(ag, w, foe)
        a.data["shot"] = str(out.get("shot"))
        a.data["kills"] = int(out.get("kills", 0))
        _adjust(ag, "fatigue", 0.008)
        _clear_target(ag)
        if out.get("shot") == "nostream":
            # No relic stream means this agent cannot legitimately be holding
            # the gun. Fail rather than fire blind - see _relic_roll.
            a.failed = True
            return
        a.done = True
        return

    a.phase = "start"


def _c_fire_bfg(a: Action, ag: Any, w: Any) -> None:
    _clear_target(ag)


def score_bfg(agent: Any, world: Any) -> dict[str, float]:
    """Utility 0..1 for FireBfg. Never raises.

    Always returns the key, and returns it as **exactly 0.0** for everyone who
    is not holding the gun. That is the compatibility rule, not tidiness:
    ``behavior.choose_action`` draws ``rng.uniform(0, TIEBREAK)`` off
    ``world.pyrng`` once per candidate whose score is > 0, so a zero here draws
    nothing and a colony that has never seen a relic stays bit-identical to the
    build that shipped before relics existed.

    0.96 + 0.03 * near is the FightAnimal band and is above
    ``behavior.OVERRIDE_FLOOR`` (0.95) for the reason FightAnimal's comment
    records: below the floor, the running action collects a HYSTERESIS_BONUS of
    0.35 and a committed builder effectively outscores anything under it, so
    nobody ever turns round. Fighting is an interrupt or it does not happen.
    """
    out: dict[str, float] = {"FireBfg": 0.0}
    try:
        if not _bfg(agent):
            return out
        if not getattr(agent, "alive", True) or getattr(agent, "taken", False):
            return out
        if getattr(agent, "inside", None) is not None:
            return out
        # A child does not fight - the same hard exclusion FightAnimal makes.
        # With the flee gate reading _combat_capable, a child holding the gun
        # still runs, which is right.
        if _role(agent) == "child":
            return out
        if _health_frac(agent) < BREAK_OFF_HEALTH:
            return out          # bleeding out: run, do not stand and aim
        if not _bfg_ready(agent, world):
            return out
        foe = _bfg_target(agent, world)
        if foe is None:
            return out
        ax = float(getattr(agent, "x", 0.0))
        near = 1.0 - _clamp01(abs(animal_x(foe, ax) - ax) / BFG_RANGE)
        out["FireBfg"] = min(0.99, _clamp01(0.96 + 0.03 * near))
    except Exception:
        log.debug("score_bfg failed", exc_info=True)
    return out


# ===========================================================================
#  The ranged abstraction - ranged_ready / ranged_target / ranged_fire
#
#  Three questions any ranged weapon can answer, and the ONLY three a caller
#  outside this file needs:
#
#      ranged_ready(agent, world)          may this person shoot at all?
#      ranged_target(agent, world)         at what, if anything?
#      ranged_fire(agent, world, target)   loose it. True if it went.
#
#  WHO THIS IS FOR. ``actions._h_lookout`` - the tower machine, in a file this
#  lane does not own - gains an attack sub-step inside its watch phase, and it
#  must not care which weapon the man on the deck happens to have. Everything
#  weapon-shaped is a row in _RANGED_WEAPONS below; adding a third weapon is
#  adding a row, and the tower learns to use it with no edit at all.
#
#  WHY IT IS A PHASE AND NOT AN ACTION KIND, recorded here because the reflex
#  is the other way round. A lookout-role agent in the watch phase already WINS
#  choose_action outright: Lookout scores 0.85, which is under OVERRIDE_FLOOR
#  (0.95), so it collects HYSTERESIS_BONUS (0.35) for a total of 1.20 - above
#  ThrowSpear's 0.985 and FightAnimal's 0.99 - and the loop then returns the
#  running machine untouched. A new scored key would lose to the very action it
#  was meant to extend, and a new *action kind* would trip Action.abandon ->
#  _c_lookout -> _plant, which is a 68 px drop at ~350 px/s. Firing from inside
#  the machine that already owns the agent adds no candidate, draws nothing off
#  world.pyrng, and cannot re-phase anything.
#
#  WHAT IT DELIBERATELY DOES NOT DECIDE: whether shooting is a good idea.
#  ranged_ready is the mechanical gate - armed, upright, off cooldown - and the
#  caller owns policy (a child, a man at 20% health, a hermit who would rather
#  not). That split is copied straight from throwing.can_throw, whose docstring
#  gives the reason: the scorer and the trigger must share the *legality* test
#  or they can disagree about who is allowed to shoot.
# ===========================================================================
#: Every weapon string that can be fired at range. Supersedes the two-term
#: ``_armed or _bfg`` union for RANGED questions only; ``_armed`` stays
#: spear-only for MELEE, deliberately - see its docstring, and note that a BFG
#: carrier walking into a wolf to hit it with a relic is what that split exists
#: to prevent. One entry per row of :data:`_RANGED_WEAPONS`.
RANGED_KINDS: tuple[str, ...] = (WEAPON_SPEAR, WEAPON_BFG)


def _perched(agent: Any) -> bool:
    """True while *agent* is being held up by a platform rather than the dirt.

    ``Stickman.perch_y`` is the one field that means this and it is persisted,
    so a save taken with a man up his tower reloads still up it. Duck-typed
    like everything else here: a world whose agents have no such field simply
    has nobody perched.
    """
    y = getattr(agent, "perch_y", None)
    if isinstance(y, bool) or not isinstance(y, (int, float)):
        return False
    try:
        return math.isfinite(float(y))
    except (TypeError, ValueError):
        return False


def _spear_ready(agent: Any, world: Any) -> bool:
    return can_throw(agent, world)


def _spear_target(agent: Any, world: Any) -> Any | None:
    # The perched exemption lives here and nowhere else. See _throw_target's
    # ``perched`` note for why it must not reach the scorer.
    return _throw_target(agent, world, perched=_perched(agent))


def _spear_fire(agent: Any, world: Any, target: Any) -> bool:
    reg = ensure_registry(world)
    if reg is None or target is None:
        return False
    if reg.throw(world, agent, target) is None:
        # Out of range now, or the solve had no answer. Not an error - it is a
        # shot not taken, and the agent still has the spear.
        return False
    # The same debit _h_throw's release phase applies, and no more: the spear,
    # the cooldown and the stat all happen inside reg.throw.
    _adjust(agent, "fatigue", 0.010)
    return True


def _bfg_row_ready(agent: Any, world: Any) -> bool:
    return _bfg(agent) and _bfg_ready(agent, world)


def _bfg_fire(agent: Any, world: Any, target: Any) -> bool:
    if target is None:
        return False
    out = _bfg_shoot(agent, world, target)
    _adjust(agent, "fatigue", 0.008)
    # "held" is the engagement gate declining with a colonist in the corridor
    # and "nostream" is a world with no relic registry; neither spent a shot.
    # A misfire very much did.
    return str(out.get("shot")) in ("fired", "misfire")


#: One row per weapon: (ready, target, fire). The whole of "which weapon".
_RANGED_WEAPONS: dict[str, tuple[
    Callable[[Any, Any], bool],
    Callable[[Any, Any], Any],
    Callable[[Any, Any, Any], bool],
]] = {
    WEAPON_SPEAR: (_spear_ready, _spear_target, _spear_fire),
    WEAPON_BFG: (_bfg_row_ready, _bfg_target, _bfg_fire),
}


def _ranged_row(agent: Any) -> tuple[
        Callable[[Any, Any], bool],
        Callable[[Any, Any], Any],
        Callable[[Any, Any, Any], bool]] | None:
    """The weapon row for whatever *agent* is holding, or None."""
    w = str(getattr(agent, "weapon", WEAPON_NONE) or WEAPON_NONE)
    if w not in RANGED_KINDS:
        return None
    return _RANGED_WEAPONS.get(w)


def ranged_ready(agent: Any, world: Any) -> bool:
    """True when *agent* could loose a ranged weapon right now.

    Mechanical only: holding something in :data:`RANGED_KINDS`, upright, not
    indoors, off that weapon's cooldown. Says nothing about whether there is
    anything worth shooting - ask :func:`ranged_target` - and nothing about
    whether shooting is wise. Never raises; it runs on the per-tick path.
    """
    try:
        row = _ranged_row(agent)
        if row is None:
            return False
        if not getattr(agent, "alive", True) or getattr(agent, "taken", False):
            return False
        if getattr(agent, "inside", None) is not None:
            return False
        return bool(row[0](agent, world))
    except Exception:
        log.debug("ranged_ready failed", exc_info=True)
        return False


def ranged_target(agent: Any, world: Any) -> Any | None:
    """What *agent* should shoot with whatever it is holding, or None.

    Goes through the weapon's own picker, so the spear keeps its throwing
    doctrine and the gun keeps its corridor sweep. The one thing this adds over
    calling those directly is the **perched exemption**: a thrower on a tower
    deck is not held back by the adult "nothing inside FLEE_RADIUS" rule, which
    is the single line that today makes elevation a downgrade rather than an
    advantage. The exemption is applied here and not in the scorer, so no
    scored candidate anywhere changes value. Never raises.
    """
    try:
        row = _ranged_row(agent)
        if row is None:
            return None
        return row[1](agent, world)
    except Exception:
        log.debug("ranged_target failed", exc_info=True)
        return None


def ranged_fire(agent: Any, world: Any, target: Any) -> bool:
    """Loose the weapon at *target*. True when the shot was actually spent.

    No wind-up: the caller owns the pose and the timing, because a tower's
    attack sub-step and a ThrowSpear action want very different ones. The
    caller is expected to have asked :func:`ranged_ready` on the same tick -
    the weapon re-checks anyway (``SpearRegistry.throw`` calls ``can_throw``
    itself), so a stale target is a refused shot and not a corrupt one.

    False means nothing left the hand and nothing was consumed: out of range,
    the solve had no answer, the gun's engagement gate declined the line. True
    means the agent has paid - one spear gone, or one BFG cooldown started, or
    the relic came apart in his hands. Never raises.
    """
    try:
        row = _ranged_row(agent)
        if row is None:
            return False
        return bool(row[2](agent, world, target))
    except Exception:
        log.debug("ranged_fire failed", exc_info=True)
        return False


# ===========================================================================
#  FetchRelic - somebody picks the thing up
#
#  The single missing piece. Everything else in the relic feature was written,
#  tuned and measured against a walk that did not exist: the drop table worked
#  (18 relics, all five kinds, 12 of 16 colonies), the appetite was measured to
#  three decimal places, the renderer could draw all five on the ground - and
#  0 of 18 were ever lifted, because no action existed to lift them. See the
#  module docstring.
#
#  Modelled on RetrieveSpear, which is the same shape and has been in the game
#  long enough to have had its edges knocked off: walk to a ground object, and
#  the object may be gone when you arrive because somebody else got there first.
#  Two differences, both deliberate:
#
#  * The claim is RelicRegistry.take(), which removes-then-equips and returns
#    False for everyone after the first. That is the whole double-claim answer
#    and it is somebody else's proven code; this file must not grow a second
#    one. (props.py solved the same problem a different way and paid 691
#    collision events for it.)
#  * A failed walk is REMEMBERED. See FETCH_SNUB_SEC - a relic behind a cliff
#    face is the one thing this action can do that RetrieveSpear cannot get
#    badly wrong, because a spear's score does not climb with age and a relic's
#    does.
# ===========================================================================
def _snubs(agent: Any) -> dict[int, float]:
    """This agent's give-up memory, ``{relic id: world time it expires}``.

    In ``__dict__`` rather than as a field, exactly like ``_bfg_next``: it is
    not worth a save slot, and a grudge that survived a reload would be
    invisible state nobody could explain. Created lazily so an agent that never
    fetches anything carries nothing.
    """
    try:
        d = agent.__dict__.get("_relic_snub")
        if not isinstance(d, dict):
            d = {}
            agent.__dict__["_relic_snub"] = d
        return d
    except Exception:
        return {}


def _snub(agent: Any, rid: Any, world: Any) -> None:
    """Leave relic *rid* alone for FETCH_SNUB_SEC. Never raises."""
    try:
        if rid is None:
            return
        d = _snubs(agent)
        d[int(rid)] = world_now(world) + FETCH_SNUB_SEC
        if len(d) > _SNUB_KEEP:
            # Drop the oldest expiries, not arbitrary keys: the ones about to
            # lapse anyway are the cheapest to forget.
            for k in sorted(d, key=lambda k: d[k])[: len(d) - _SNUB_KEEP]:
                d.pop(k, None)
    except Exception:
        log.debug("could not record a fetch snub", exc_info=True)


def _snubbed(agent: Any, rid: Any, world: Any) -> bool:
    try:
        d = agent.__dict__.get("_relic_snub")
        if not isinstance(d, dict) or not d:
            return False
        until = d.get(int(rid))
        if until is None:
            return False
        if world_now(world) >= float(until):
            d.pop(int(rid), None)
            return False
        return True
    except Exception:
        return False


def _clear_snub(agent: Any, rid: Any) -> None:
    try:
        d = agent.__dict__.get("_relic_snub")
        if isinstance(d, dict):
            d.pop(int(rid), None)
    except Exception:
        pass


def _grave_key(gid: Any) -> int:
    """Snub key for a grave.

    The snub dict is shared between the two relic actions, and relic ids and
    prop ids are both small positives drawn from separate counters - so a
    colonist who gave up on relic#3 would also be refusing grave#3. Graves go in
    negative. ``or 1`` because id 0 has no sign to flip, and a key of 0 would
    collide with relic#0 (which ``RelicRegistry.add`` never issues, but that is
    an invariant in another file and this is one character).
    """
    try:
        return -(int(gid) or 1)
    except (TypeError, ValueError):
        return -1


def _relic_eligible(agent: Any) -> bool:
    """Can this person carry a relic at all? Shared by both relic actions.

    One slot per agent, so somebody already carrying one is out - and that is
    also what stops anyone being both impenetrable and holding the gun. Children
    are out for the same reason ``score_combat`` hard-excludes them from
    FightAnimal and ThrowSpear: every gear-and-danger behaviour in this game
    does, and a nine-year-old picking up the wyrm-gun is not a feature.
    """
    try:
        if not getattr(agent, "alive", True) or getattr(agent, "taken", False):
            return False
        if getattr(agent, "inside", None) is not None:
            return False
        return _role(agent) != "child"
    except Exception:
        return False


def fetchable_relic(agent: Any, world: Any) -> Any | None:
    """The relic this agent should go and get, or None. **The only answer.**

    Public, and imported by ``behavior._score_loot``, because the scorer and the
    maker disagreeing is not a theoretical problem here - it is the exact bug
    that shipped. behaviour still owns every *number* (the appetite floor, the
    ageing ceiling, the bystander damper, the danger gate it can see and this
    file cannot); this owns *which object*, and therefore owns the gates that
    would make a walk a bad idea.

    Nearest-first, skipping anything this agent has recently failed to reach, so
    a colonist stuck behind a ledge moves on to the next relic instead of pacing
    at the wall until it rots. Ties break on id like ``RelicRegistry.nearest``,
    so the choice is stable across a reload.
    """
    try:
        if not _relic_eligible(agent):
            return None
        if relic_worn(agent):
            return None
        reg = relic_registry_of(world)
        if reg is None:
            return None
        try:
            candidates = reg.all()
        except Exception:
            return None
        # Before any animal scan: this runs once per agent per scoring call, and
        # the overwhelmingly common case is a world with nothing on the ground.
        if not candidates:
            return None
        ax = float(getattr(agent, "x", 0.0))
        # Nothing hunting where we stand. One query, and it rules out every
        # relic at once, so it goes outside the loop.
        if nearest_animal(world, ax, max_dist=FETCH_SAFE_RADIUS) is not None:
            return None
        best, best_key = None, None
        for r in candidates:
            try:
                rx = float(getattr(r, "x", ax))
                rid = int(getattr(r, "id", 0) or 0)
            except (TypeError, ValueError):
                continue
            d = abs(rx - ax)
            if d > FETCH_SEEK:
                continue
            if _snubbed(agent, rid, world):
                continue
            key = (d, rid)
            if best_key is not None and key >= best_key:
                continue
            # Last and most expensive: is anything standing over it? Below the
            # distance test on purpose, so the scan only runs for a candidate
            # that is currently the nearest - and a relic with a wolf on it
            # falls through to the next one out rather than cancelling the walk.
            if nearest_animal(world, rx, max_dist=FETCH_SAFE_RADIUS) is not None:
                continue
            best, best_key = r, key
        return best
    except Exception:
        log.debug("fetchable_relic failed", exc_info=True)
        return None


def _h_fetch_relic(a: Action, ag: Any, w: Any, dt: float) -> None:
    reg = relic_registry_of(w)
    if reg is None:
        a.failed = True
        return

    if a.phase == "start":
        r = fetchable_relic(ag, w)
        if r is None:
            a.failed = True
            return
        a.target = int(getattr(r, "id", 0) or 0)
        a.phase = "walk"

    if a.phase == "walk":
        a.pose = "walk"
        if relic_worn(ag):
            # Somebody handed them one, or a save rehydrated mid-walk into an
            # agent who already has a slot full. Not a failure.
            _halt(ag)
            a.done = True
            return
        r = reg.get(a.target)
        if r is None:
            # Taken, or rotted, while we were walking. Re-acquire rather than
            # fail: there may well be another one, and a fail here costs the
            # agent a re-score for nothing.
            r = fetchable_relic(ag, w)
            if r is None:
                _halt(ag)
                a.failed = True
                return
            a.target = int(getattr(r, "id", 0) or 0)
        rx = float(getattr(r, "x", float(getattr(ag, "x", 0.0))))
        rem = step_toward(ag, w, rx, dt, arrive=FETCH_ARRIVE)
        if rem <= FETCH_ARRIVE:
            a.phase = "take"
        elif a.t > FETCH_TIMEOUT:
            # Could not get there. THIS is the branch FETCH_SNUB_SEC exists for
            # - without it the agent re-scores straight back into the same walk
            # against the same cliff, and the relic's score is higher than it
            # was because it has aged.
            _halt(ag)
            _snub(ag, a.target, w)
            a.failed = True
        return

    if a.phase == "take":
        _halt(ag)
        r = reg.get(a.target)
        # take() removes-then-equips and bumps stats["relics_found"], writes the
        # chronicle line, and returns False for everybody who arrived second.
        if r is None or not reg.take(r, ag, world=w):
            _snub(ag, a.target, w)
            a.failed = True
            return
        _clear_snub(ag, a.target)
        emit_speech(w, ag, "^")
        # A dragon's relic is the best thing that happens to a colonist. Same
        # size as the craft bonuses (0.10 / 0.12), because it is the same kind
        # of event - you now have a thing you did not have.
        _adjust(ag, "morale", 0.12)
        a.done = True
        return

    a.phase = "start"


def _c_fetch_relic(a: Action, ag: Any, w: Any) -> None:
    """Abandoned mid-walk. Nothing to undo - the relic never left the ground.

    Present so the kind has an entry in ``_CLEANUP`` alongside the others rather
    than relying on the dict's default, which is the sort of asymmetry that has
    somebody later assuming a missing entry means something.
    """
    _halt(ag)


# ===========================================================================
#  RaiseTheDead - the cairnstone's one use
#
#  World.raise_the_dead() had ZERO callers. The cairnstone is the second most
#  common drop in the table and could never do anything at all: a colonist would
#  pick it up (once fetching existed) and carry it to their own grave.
#
#  The contract is stated in that method's docstring and this is the half of it
#  that lives here: **consume on True, keep on False.** raise_the_dead has four
#  separate ways to decline - no name on the stone, the roster at MAX_POP, the
#  headstone already weathered away, junk state - and a refusal that ate the
#  item would lose the rarest half of the skeletal's drop table for nothing.
# ===========================================================================
def _graves_of(world: Any) -> list[Any]:
    """Living headstones. Duck-typed: props.py is a sibling, not a dependency."""
    try:
        fn = getattr(getattr(world, "props", None), "all_of", None)
        if not callable(fn):
            return []
        return list(fn("grave") or [])
    except Exception:
        return []


def _grave_named(grave: Any) -> str:
    """The name on the stone, "" if there is none.

    Checked *before* the walk because ``raise_the_dead`` refuses a nameless
    headstone - older saves have them, and _reap_dead only started passing the
    generation through recently. Walking the length of the map to be told no is
    a worse outcome than never setting off.
    """
    try:
        state = getattr(grave, "state", None)
        if not isinstance(state, dict):
            return ""
        return str(state.get("name") or "").strip()
    except Exception:
        return ""


def raisable_grave(agent: Any, world: Any) -> Any | None:
    """The headstone this cairnstone-bearer should walk to, or None.

    Same contract as :func:`fetchable_relic` - public, and the single answer
    both the scorer and the maker use. The MAX_POP check is duplicated from
    ``raise_the_dead`` on purpose: it is cheap, and without it the bearer walks
    to a grave in a full colony, is refused, keeps the stone (correctly), and
    re-scores straight back into the same walk.
    """
    try:
        if not _relic_eligible(agent):
            return None
        if relic_worn(agent) != RELIC_CAIRN:
            return None
        # Cheapest first: only the one bearer gets past this line, so every
        # other agent's scoring call costs two getattrs.
        graves = _graves_of(world)
        if not graves:
            return None
        try:
            pop = len(world.population.alive_agents())
        except Exception:
            pop = 0
        if pop >= MAX_POP:
            return None
        ax = float(getattr(agent, "x", 0.0))
        if nearest_animal(world, ax, max_dist=FETCH_SAFE_RADIUS) is not None:
            return None
        best, best_key = None, None
        for g in graves:
            try:
                if not getattr(g, "alive", True):
                    continue
                if not _grave_named(g):
                    continue
                gx = float(getattr(g, "x", ax))
                gid = int(getattr(g, "id", 0) or 0)
            except (TypeError, ValueError):
                continue
            d = abs(gx - ax)
            if d > RAISE_SEEK:
                continue
            if _snubbed(agent, _grave_key(gid), world):
                continue
            key = (d, gid)
            if best_key is not None and key >= best_key:
                continue
            if nearest_animal(world, gx, max_dist=FETCH_SAFE_RADIUS) is not None:
                continue
            best, best_key = g, key
        return best
    except Exception:
        log.debug("raisable_grave failed", exc_info=True)
        return None


def _grave_by_id(world: Any, gid: Any) -> Any | None:
    try:
        want = int(gid)
    except (TypeError, ValueError):
        return None
    for g in _graves_of(world):
        try:
            if int(getattr(g, "id", 0) or 0) == want and getattr(g, "alive", True):
                return g
        except (TypeError, ValueError):
            continue
    return None


def _h_raise_dead(a: Action, ag: Any, w: Any, dt: float) -> None:
    if a.phase == "start":
        g = raisable_grave(ag, w)
        if g is None:
            a.failed = True
            return
        a.target = int(getattr(g, "id", 0) or 0)
        a.phase = "walk"

    if a.phase == "walk":
        a.pose = "walk"
        if relic_worn(ag) != RELIC_CAIRN:
            # Died and dropped it, or a save rehydrated into somebody who is not
            # carrying the stone any more.
            _halt(ag)
            a.failed = True
            return
        g = _grave_by_id(w, a.target)
        if g is None:
            g = raisable_grave(ag, w)
            if g is None:
                _halt(ag)
                a.failed = True
                return
            a.target = int(getattr(g, "id", 0) or 0)
        gx = float(getattr(g, "x", float(getattr(ag, "x", 0.0))))
        rem = step_toward(ag, w, gx, dt, arrive=REACH)
        if rem <= REACH:
            a.phase = "place"
            a.data["wt"] = 0.0
        elif a.t > RAISE_TIMEOUT:
            _halt(ag)
            _snub(ag, _grave_key(a.target), w)
            a.failed = True
        return

    if a.phase == "place":
        # A beat of kneeling at the stone. "chop" for the reason ThrowSpear and
        # FireBfg use it: actions.POSES is a closed tuple render switches on, so
        # a new name here would draw nothing at all.
        a.pose = "chop"
        _halt(ag)
        g = _grave_by_id(w, a.target)
        if g is None:
            a.failed = True
            return
        _face(ag, float(getattr(g, "x", 0.0)) - float(getattr(ag, "x", 0.0)))
        a.data["wt"] = float(a.data.get("wt", 0.0)) + dt
        if float(a.data["wt"]) < RAISE_PLACE_SEC:
            return

        fn = getattr(w, "raise_the_dead", None)
        ok = False
        if callable(fn):
            try:
                ok = bool(fn(g, by=_name(ag)))
            except TypeError:
                # The one-argument shape the contract also allows.
                try:
                    ok = bool(fn(g))
                except Exception:
                    log.debug("raise_the_dead(grave) failed", exc_info=True)
            except Exception:
                log.debug("raise_the_dead failed", exc_info=True)
        if not ok:
            # KEEP THE STONE. See the header above: four ways to decline and
            # none of them should cost the item. Snub this grave so the bearer
            # tries a different one (or gets on with their life) instead of
            # kneeling at the same refusal every two seconds.
            _snub(ag, _grave_key(a.target), w)
            a.failed = True
            return
        # SPEND IT. consume() is not unequip(): it bumps stats["relics_used"]
        # and the stone does not hit the ground, which is the difference between
        # "the item left the world" and "the item changed hands".
        relic_consume(w, ag)
        # "*" is render's "something notable" star. NOT "!", which
        # stickfigure._GLYPH_ALIASES maps to the alarm bubble - somebody
        # standing over a headstone with a danger sign over their head reads as
        # the exact opposite of what just happened.
        emit_speech(w, ag, "*")
        _adjust(ag, "morale", 0.20)
        a.done = True
        return

    a.phase = "start"


def _c_raise_dead(a: Action, ag: Any, w: Any) -> None:
    _halt(ag)


# ===========================================================================
#  Scoring + construction
# ===========================================================================
def score_throw(agent: Any, world: Any) -> dict[str, float]:
    """Utility 0..1 for the two throwing behaviours. Never raises.

    Always returns both keys, so the score dict is the same *shape* whether or
    not anything can be thrown. That is not tidiness: ``behavior.choose_action``
    draws ``rng.uniform(0, TIEBREAK)`` once per candidate whose score is > 0,
    off ``world.pyrng`` - the same stream ``animals._rng`` spawns wolves from.
    Two builds that score a different NUMBER of candidates diverge in what
    wildlife they get, so any A/B of this mechanic has to keep both arms
    scoring both keys and toggle only whether :func:`make_throw_action` returns
    an action. See the module head of this file for the measured size of that
    effect.
    """
    out: dict[str, float] = {"ThrowSpear": 0.0, "RetrieveSpear": 0.0}
    try:
        if not getattr(agent, "alive", True) or getattr(agent, "taken", False):
            return out
        if getattr(agent, "inside", None) is not None:
            return out
        ax = float(getattr(agent, "x", 0.0))
        child = _role(agent) == "child"

        # ---------------------------------------------------------- throw ---
        if can_throw(agent, world) and (child or _health_frac(agent) >= BREAK_OFF_HEALTH):
            foe = _throw_target(agent, world)
            if foe is not None:
                # Above OVERRIDE_FLOOR (0.95) on purpose. Below it the running
                # action collects HYSTERESIS_BONUS (0.35), so a committed
                # builder effectively outscores anything under the floor and
                # nobody ever turns round - the same trap FightAnimal's comment
                # records falling into.
                #
                # 0.985 for an adult is deliberately ABOVE anything FightAnimal
                # can score at this distance (0.95 + 0.04 * near + 0.01
                # armoured, so 0.983 at the FLEE_RADIUS crossover and falling
                # from there). Throw first, close after: the alternative is the
                # long sprint that RESCUE_RADIUS's comment records turning 0
                # fall deaths into 12. A child sits lower, at 0.965, because
                # their FleeAnimal is 0.92-1.0 and running must still win once
                # anything is actually on top of them.
                out["ThrowSpear"] = 0.965 if child else 0.985

        # ------------------------------------------------------- retrieve ---
        sp = _retrievable_spear(agent, world)
        if sp is not None:
            # Sits alongside CraftSpear (0.55 + 0.32 * fresh) rather than under
            # it: fetching a spear that already exists should beat felling a
            # tree to make another one, but only while it is close enough to be
            # worth the walk.
            near = 1.0 - _clamp01(abs(float(sp.x) - ax) / RETRIEVE_SEEK)
            fresh = sighting_freshness(world)
            out["RetrieveSpear"] = _clamp01(
                0.58 + 0.30 * fresh) * (0.65 + 0.35 * near)
    except Exception:
        log.debug("score_throw failed", exc_info=True)
    return out


def make_throw_action(kind: str, agent: Any, world: Any) -> Action | None:
    """Build ThrowSpear / RetrieveSpear, or None if it is not available now.

    Split out of :func:`make_combat_action` and called through the module global
    so an A/B can replace exactly this function - which is the only honest way
    to measure the mechanic. See :func:`score_throw`.
    """
    try:
        if kind == "ThrowSpear":
            if not can_throw(agent, world):
                return None
            foe = _throw_target(agent, world)
            if foe is None:
                return None
            a = make_action("ThrowSpear", target=animal_id(foe), phase="aim",
                            pose="chop",
                            ax=animal_x(foe, float(getattr(agent, "x", 0.0))),
                            wt=0.0)
            if _is_dragon(world, foe):
                a.data["foe"] = "dragon"
            return a

        if kind == "RetrieveSpear":
            sp = _retrievable_spear(agent, world)
            if sp is None:
                return None
            return make_action("RetrieveSpear", target=int(sp.id),
                               phase="walk", pose="walk")
    except Exception:
        log.debug("make_throw_action(%r) failed", kind, exc_info=True)
    return None


def score_combat(agent: Any, world: Any) -> dict[str, float]:
    """Utility 0..1 for the four combat/crafting behaviours.

    Meant to be merged into ``behavior.score_actions``'s dict. Always returns
    all four keys and never raises; a world with no animal subsystem scores
    everything except the two crafts at zero.
    """
    out: dict[str, float] = {
        "FleeAnimal": 0.0, "FightAnimal": 0.0,
        "CraftSpear": 0.0, "CraftArmour": 0.0,
        "ThrowSpear": 0.0, "RetrieveSpear": 0.0,
        "FireBfg": 0.0,
        # Declared here at 0.0 and NEVER raised in this file. behavior.py scores
        # both after merging this dict and has the last word, because every band
        # they have to beat or lose to - Celebrate, Mourn, Eat, Sleep,
        # OVERRIDE_FLOOR, HYSTERESIS_BONUS - is declared over there. They are in
        # this dict at all so the score dict is the same *shape* whether or not
        # behaviour got as far as scoring them; see the docstring on
        # score_throw for why the number of keys matters more than it looks.
        "FetchRelic": 0.0, "RaiseTheDead": 0.0,
    }
    # Merged here rather than in behavior.py because behavior.score_actions
    # already does `s.update(score_combat(...))`, so folding the throwing keys
    # in at this level makes them arrive with no edit to that file at all -
    # which matters when behavior.py belongs to somebody else. The two are kept
    # as separate functions so either can be scored, tested or A/B'd alone.
    try:
        out.update(score_throw(agent, world))
    except Exception:
        log.debug("score_throw failed", exc_info=True)
    # Same argument for the gun: merged here, so behavior.py - somebody else's
    # file - needs no edit at all to learn about it.
    try:
        out.update(score_bfg(agent, world))
    except Exception:
        log.debug("score_bfg failed", exc_info=True)
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
            # _combat_capable, not `armed`. A BFG carrier is not holding a spear
            # and never will be, so the spear-only predicate reads them as
            # defenceless and hands them 0.92-1.0 to run - away from every wolf
            # they could have vaporised, carrying the rarest item in the game
            # and using it for nothing. This single word is what makes the
            # weapon exist at all.
            if not _combat_capable(agent) or role == "child":
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
            # _craft_satisfied, not `not armed`: a BFG carrier must not score a
            # spear craft that would write over the gun. Identical to `not
            # armed` for everybody who has never held a relic, so a colony
            # without one draws exactly the same random numbers as before.
            if not _craft_satisfied("CraftSpear", agent) \
                    and (_can_afford(world, SPEAR_COST, agent)
                         or _paid_up(agent, "CraftSpear")):
                out["CraftSpear"] = _clamp01(0.55 + 0.32 * fresh)
            if not armoured and (_can_afford(world, ARMOUR_COST, agent)
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
        if kind in ("ThrowSpear", "RetrieveSpear"):
            # Module-global lookup, not a direct call: an A/B rig replaces
            # combat_actions.make_throw_action with one that returns None and
            # gets an arm that scores identically and therefore draws the same
            # random numbers at the same points. A local call would bind at
            # import and defeat that.
            return make_throw_action(kind, agent, world)
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
            if _is_dragon(world, animal):
                a.data["foe"] = "dragon"
            _set_target(agent, animal)
            return a

        if kind == "FleeAnimal":
            animal = nearest_animal(world, ax)
            if animal is None:
                return None
            return make_action("FleeAnimal", target=animal_id(animal),
                               pose="panic", fx=animal_x(animal, ax))

        if kind == "FireBfg":
            if not _bfg(agent) or _role(agent) == "child":
                return None
            if not _bfg_ready(agent, world):
                return None
            if _health_frac(agent) < BREAK_OFF_HEALTH:
                return None
            foe = _bfg_target(agent, world)
            if foe is None:
                return None
            a = make_action("FireBfg", target=animal_id(foe), phase="aim",
                            pose="chop", ax=animal_x(foe, ax), wt=0.0)
            if _is_dragon(world, foe):
                a.data["foe"] = "dragon"
            return a

        if kind == "FetchRelic":
            r = fetchable_relic(agent, world)
            if r is None:
                return None
            return make_action("FetchRelic", target=int(getattr(r, "id", 0) or 0),
                               phase="walk", pose="walk")

        if kind == "RaiseTheDead":
            g = raisable_grave(agent, world)
            if g is None:
                return None
            return make_action("RaiseTheDead",
                               target=int(getattr(g, "id", 0) or 0),
                               phase="walk", pose="walk")

        if kind == "CraftSpear":
            if _craft_satisfied("CraftSpear", agent) \
                    or not _can_afford(world, SPEAR_COST, agent):
                return None
            return make_action("CraftSpear", pose="walk")

        if kind == "CraftArmour":
            if _armoured(agent) or not _can_afford(world, ARMOUR_COST, agent):
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
    "ThrowSpear": _h_throw,
    "RetrieveSpear": _h_retrieve,
    "FireBfg": _h_fire_bfg,
    "FetchRelic": _h_fetch_relic,
    "RaiseTheDead": _h_raise_dead,
}

_COMBAT_CLEANUP: dict[str, Callable[[Action, Any, Any], None]] = {
    "CraftSpear": _c_craft,
    "CraftArmour": _c_craft,
    "FightAnimal": _c_fight,
    "ThrowSpear": _c_throw,
    "FireBfg": _c_fire_bfg,
    "FetchRelic": _c_fetch_relic,
    "RaiseTheDead": _c_raise_dead,
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

    # Fail an absent module FAST, before anything below asks for a hitbox.
    # ``throwing.clearance`` does ``from .altitude import clearance`` inside a
    # try/except on EVERY call, and ``sim/altitude.py`` does not exist in this
    # checkout - the fallback is deliberate and correct. But a failed import is
    # a full sys.path filesystem search each time, measured at ~32 ms on this
    # repo's mount, and the rate runs below take ~24000 hitbox lookups: twenty
    # minutes of stat() calls for a test whose arithmetic takes two seconds.
    # Priming sys.modules with None is precisely what the import machinery does
    # for a cached negative result. Guarded on the module genuinely being
    # absent, so a checkout that grows one is unaffected. Test-harness only -
    # nothing at module scope does this, and no behaviour changes either way.
    import importlib.util as _ilu
    import sys as _sys
    if _ilu.find_spec("backgrounded.sim.altitude") is None:
        _sys.modules.setdefault("backgrounded.sim.altitude", None)

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

    # --- throwing ----------------------------------------------------------
    from .throwing import SpearRegistry, Spear, STATE_GROUND

    w2 = _World()
    w2.structures.create("stockpile", 600.0, 600.0, built=True)
    w2.spears = SpearRegistry(seed=17)
    thrower = Stickman(id=10, name="Cal", x=400.0, y=600.0)
    thrower.weapon = WEAPON_SPEAR
    w2.agents.append(thrower)
    far = _Animal("wolf", 600.0)          # 200 px: past FIGHT_RADIUS
    w2.animals.append(far)

    sc = score_throw(thrower, w2)
    print("adult, wolf at 200px:", {k: round(v, 2) for k, v in sc.items()})
    assert sc["ThrowSpear"] > 0.95, sc
    far.x = 460.0                          # 60 px: inside FLEE_RADIUS
    assert score_throw(thrower, w2)["ThrowSpear"] == 0.0, "threw at melee range"
    far.x = 600.0

    act = make_combat_action("ThrowSpear", thrower, w2)
    assert act is not None and act.kind == "ThrowSpear"
    for _ in range(int(30 * 6)):
        act.update(thrower, w2, dt)
        w2.spears.tick(w2, dt)
        if act.finished:
            break
    print("throw:", act, "weapon now", repr(thrower.weapon),
          "| spears", [s.summary() for s in w2.spears.all()])
    assert act.done, act
    assert thrower.weapon == WEAPON_NONE, "the thrower kept the spear"
    assert len(w2.spears.all()) == 1, w2.spears.all()

    # the spear comes down somewhere, and it can be fetched
    for _ in range(int(30 * 8)):
        w2.spears.tick(w2, dt)
        if not w2.spears.in_flight():
            break
    assert w2.spears.on_ground(), "the spear never landed"
    lying = w2.spears.on_ground()[0]
    far.x = 1400.0                         # get the wolf away from the spear
    sc = score_throw(thrower, w2)
    print("unarmed, a spear on the ground:", {k: round(v, 2) for k, v in sc.items()})
    assert sc["RetrieveSpear"] > 0.4, sc
    fetch = make_combat_action("RetrieveSpear", thrower, w2)
    assert fetch is not None
    for _ in range(int(30 * 60)):
        fetch.update(thrower, w2, dt)
        w2.spears.tick(w2, dt)      # also what runs the per-thrower cooldown
        thrower.apply_physics(dt, w2.terrain)
        if fetch.finished:
            break
    print("retrieve:", fetch, "weapon", repr(thrower.weapon),
          f"walked to {thrower.x:.0f} for a spear at {lying.x:.0f}")
    assert fetch.done and thrower.weapon == WEAPON_SPEAR, fetch
    assert not w2.spears.on_ground(), "the spear was picked up twice"
    assert w2.stats.get("spears_recovered") == 1, w2.stats

    # a child throws, and only from far enough out to survive doing it
    kid = Stickman(id=11, name="Wren", x=400.0, y=600.0)
    kid.role = "child"
    kid.weapon = WEAPON_SPEAR
    w2.agents.append(kid)
    far.x = 400.0 + CHILD_THROW_MIN_GAP + 20.0
    sck = score_throw(kid, w2)
    print("child, wolf at 92px:", {k: round(v, 2) for k, v in sck.items()})
    assert sck["ThrowSpear"] > 0.95, sck
    assert score_combat(kid, w2)["FightAnimal"] == 0.0, "a child in melee"
    far.x = 400.0 + CHILD_THROW_MIN_GAP - 20.0
    assert score_throw(kid, w2)["ThrowSpear"] == 0.0, "child threw at 52 px"
    far.x = 400.0 + THROW_MAX_RANGE - 10.0     # inside adult range, past 0.6x
    assert score_throw(kid, w2)["ThrowSpear"] == 0.0, "child out-threw an adult"

    # an adult with a wolf on top of them keeps the spear
    far.x = 600.0
    close = _Animal("wolf", thrower.x + 30.0)
    w2.animals.append(close)
    assert score_throw(thrower, w2)["ThrowSpear"] == 0.0, "disarmed under attack"
    w2.animals.remove(close)

    # --- the melee ceiling --------------------------------------------------
    class _Flyer:
        def __init__(self, x, alt):
            self.id, self.kind = 1, "wyvern"
            self.x, self.alt = float(x), float(alt)
            self.y = 600.0 - float(alt)
            self.health = self.max_health = 200.0
            self.alive, self.hostile = True, True

        def hurt(self, amount):
            self.health = max(0.0, self.health - float(amount))
            if self.health <= 0.0:
                self.alive = False
                return True
            return False

    thrower.x = 400.0
    thrower.weapon = WEAPON_SPEAR
    w2.animals.clear()
    w2.dragons = [_Flyer(500.0, 70.0)]          # above MELEE_CEILING, in reach
    assert pick_quarry(thrower, w2) is None, "melee against something 70 px up"
    assert score_combat(thrower, w2)["FightAnimal"] == 0.0
    assert score_throw(thrower, w2)["ThrowSpear"] > 0.95, "no answer to a flyer"
    # ... and something high enough is out of the ballistic envelope entirely,
    # which is what makes "that one is untouchable" a fact rather than a rule.
    # See the reachability table in throwing.py's docstring.
    w2.dragons = [_Flyer(500.0, 350.0)]
    assert score_throw(thrower, w2)["ThrowSpear"] == 0.0, "speared a thing 350 px up"
    w2.dragons = [_Flyer(500.0, 70.0)]
    w2.dragons = [_Flyer(430.0, 0.0)]           # landed: a fair fight
    # Too much health to kill inside the test, so the take-off branch below is
    # actually reached instead of the fight simply ending in a win.
    w2.dragons[0].health = w2.dragons[0].max_health = 2000.0
    q = pick_quarry(thrower, w2)
    assert q is not None and q.kind == "wyvern", q
    assert score_combat(thrower, w2)["FightAnimal"] > 0.85
    duel = make_combat_action("FightAnimal", thrower, w2)
    assert duel is not None and duel.data.get("foe") == "dragon", duel.data
    for _ in range(int(30 * 20)):
        duel.update(thrower, w2, dt)
        if duel.finished or not w2.dragons[0].alive:
            break
    print(f"melee vs a landed dragon: {duel} dragon hp "
          f"{w2.dragons[0].health:.0f}/2000")
    assert w2.dragons[0].health < 2000.0, "never landed a blow on a grounded dragon"
    assert not duel.finished, "the fight ended before the dragon took off"
    # ... and it walks away rather than standing under it when it takes off
    w2.dragons[0].alt = 300.0
    w2.dragons[0].y = 300.0
    for _ in range(int(30 * 4)):
        duel.update(thrower, w2, dt)
        if duel.finished:
            break
    assert duel.finished, "stood under a flying dragon swinging"
    assert claim_hides(w2, w2.dragons[0]) == 0, "took a hide off a dragon"
    w2.dragons = []

    # --- the wyrm-gun -------------------------------------------------------
    from .items import RelicRegistry as _RelicRegistry

    class _Rolls(_RelicRegistry):
        """A real ``RelicRegistry`` with a scripted ``roll()``.

        Driving the sequence by hand is the only way to test an 8% branch
        without firing ten thousand shots; the rates themselves are measured
        further down against a real seeded stream.

        A SUBCLASS and not a duck-typed stub, and that is the whole point:
        ``items.registry_of`` is an ``isinstance(reg, RelicRegistry)`` check, so
        a hand-rolled object with the right two methods is invisible to it.
        :func:`_record_beam` would find no registry, log at DEBUG, and drop the
        beam on the floor - which is exactly the failure it exists to have
        fixed, and a stub would have hidden it from this test. Everything except
        the dice is therefore the real thing: the real ``add_beam``, the real
        ``beams`` list, the real ``MAX_BEAMS`` cap.
        """

        def __init__(self, seq: list[float]) -> None:
            super().__init__(seed=0)
            self.seq = list(seq)
            self.n = 0

        def roll(self, lo: float = 0.0, hi: float = 1.0) -> float:
            v = self.seq[self.n % len(self.seq)]
            self.n += 1
            return float(v)

    w3 = _World()
    gunner = Stickman(id=20, name="Dax", x=300.0, y=600.0)
    gunner.weapon = WEAPON_BFG
    gunner.relic = RELIC_BFG
    w3.agents.append(gunner)
    near_wolf = _Animal("wolf", 500.0)
    far_wolf = _Animal("wolf", 560.0)          # further than the target
    behind = _Animal("bear", 150.0)            # the other way down the line
    beyond = _Animal("wolf", 300.0 + BFG_RANGE + 60.0)
    w3.animals.extend([near_wolf, far_wolf, behind, beyond])
    bystander = Stickman(id=21, name="Eli", x=540.0, y=600.0)     # in the line
    overhead = Stickman(id=22, name="Fay", x=480.0, y=470.0)      # 130 px up
    w3.agents.extend([bystander, overhead])

    # THE BEAM HAS WIDTH - the geometry half of Fix 2. _beam_stop used to return
    # the first sample where the AXIS met the ground, cutting the shot off with
    # its whole upper half still in clear air, while every other part of the
    # mechanic treated the shot as a BFG_CORRIDOR-wide lane. On this flat world
    # a shot from a 20 px shoulder at a wolf's centre 6.5 px up is angled down
    # and the hairline rule killed it at ~292 px of a nominal 420. The rule is
    # now "the hill fills the beam", not "the hill touches its axis", which is
    # +132 px of median reach on real terrain. See _beam_stop.
    assert _beam_stop(w3, 300.0, 500.0, 1.0, 0.0, BFG_RANGE) == BFG_RANGE

    def _stop_for(corr: float) -> float:
        sx0, sy0, sx1, sy1 = _beam_ends(gunner, w3, near_wolf)
        _d = math.hypot(sx1 - sx0, sy1 - sy0) or 1.0
        return _beam_stop(w3, sx0, sy0, (sx1 - sx0) / _d, (sy1 - sy0) / _d,
                          BFG_RANGE, corridor=corr)

    _hairline, _lane = _stop_for(0.0), _stop_for(BFG_CORRIDOR)
    print(f"beam stop, wolf 200 px off: hairline={_hairline:.0f}px  "
          f"corridor={_lane:.0f}px  (BFG_RANGE {BFG_RANGE:.0f})")
    assert _hairline < BFG_RANGE * 0.8, ("the hairline rule used to bury it",
                                         _hairline)
    assert _lane >= BFG_RANGE - 1e-6, ("the corridor bought no range", _lane)
    # A wider sweep has to get a correspondingly generous stop, or the two
    # halves of one beam disagree about where it ends.
    assert _stop_for(BFG_CORRIDOR * 2.0) >= _lane

    # ...and on a hillside, which is where the change actually pays and where a
    # colony actually stands. Ground rising 1 px in 10 swallows a hairline beam
    # from a 20 px shoulder after 200 px; the same beam is BFG_CORRIDOR thick,
    # so the hill has to climb another 22 px to fill it and the shot carries
    # twice as far. Everyone standing on that stretch was previously out of
    # range of a beam whose corridor would happily have caught them.
    class _Slope:
        def ground_y(self, x: float) -> float:
            return 600.0 - max(0.0, (float(x) - 300.0) * 0.1)

        def slope(self, x: float) -> float:
            return 0.1

    _hill = _World()
    _hill.terrain = _Slope()
    _up_hair = _beam_stop(_hill, 300.0, 580.0, 1.0, 0.0, BFG_RANGE, corridor=0.0)
    _up_lane = _beam_stop(_hill, 300.0, 580.0, 1.0, 0.0, BFG_RANGE)
    print(f"level beam into a 1-in-10 rise: hairline={_up_hair:.0f}px  "
          f"corridor={_up_lane:.0f}px")
    assert 180.0 <= _up_hair <= 210.0, _up_hair
    assert _up_lane > _up_hair * 1.9, (_up_hair, _up_lane)

    x0, y0, tx, ty = _beam_ends(gunner, w3, near_wolf)
    swept = beam_sweep(w3, gunner, x0, y0, tx, ty)
    print("beam sweep:", [(getattr(o, "name", getattr(o, "kind", "?")), s)
                          for o, s in swept])
    hit_set = {id(o) for o, _ in swept}
    assert id(near_wolf) in hit_set and id(far_wolf) in hit_set, "not a line kill"
    assert id(bystander) in hit_set, "a colonist in the beam was spared"
    assert id(behind) not in hit_set, "the beam fired backwards"
    assert id(beyond) not in hit_set, "the beam outranged BFG_RANGE"
    assert id(overhead) not in hit_set, "the corridor is not a corridor"
    assert id(gunner) not in hit_set, "the wielder shot themselves in the line"

    # A colonist in the corridor: hold fire unless the heedless roll says
    # otherwise. 0.5 misses BFG_MISFIRE and misses BFG_HEEDLESS.
    w3.relics = _Rolls([0.5])
    out = _bfg_shoot(gunner, w3, near_wolf)
    print("friendly in the line:", out, "| wolf alive:", near_wolf.alive)
    assert out["shot"] == "held", out
    assert near_wolf.alive and bystander.alive, "held fire and shot anyway"

    # ...and heedless: 0.5 (no misfire), then 0.0 (< BFG_HEEDLESS, fire).
    w3.relics = _Rolls([0.5, 0.0])
    gunner.__dict__["_bfg_next"] = 0.0
    out = _bfg_shoot(gunner, w3, near_wolf)
    print("heedless:", out, "| beams handed to render:", len(w3.relics.beams))
    assert out["shot"] == "fired" and out["heedless"] is True, out
    assert not near_wolf.alive and not far_wolf.alive, "the line did not clear"
    assert not bystander.alive, "a colonist stood in a BFG beam and lived"
    assert bystander.death_cause == CAUSE_DISINTEGRATED, bystander.death_cause
    assert out["friends"] == ["Eli"], out
    assert behind.alive and beyond.alive and overhead.alive, "hit outside the line"
    assert w3.stats.get("animals_killed") == 2, w3.stats
    # ...and render was told. Asserted against the RELIC REGISTRY, which is the
    # list render prefers and the only one _record_beam writes to - there is no
    # world.bfg_beams any more and there must not be, because a second list
    # would be a second ageing rule that nothing ticks. Asserting on the old
    # name is what made `python -m backgrounded.sim.combat_actions` exit 1.
    assert w3.relics.beams, "render was told nothing about the shot"
    assert not w3.relics.beams[-1]["misfire"], w3.relics.beams[-1]
    assert any("did not care who was in front" in ln for ln in w3.lines), w3.lines

    # THE HEADLINE REQUIREMENT: "wipes out all stickmen in the direction that
    # the bfg9000 was fired ... could possibly leave a single stickman."
    # Four colonists queued down one line, one heedless shot, one survivor - the
    # wielder. Measured before the aim was levelled, over 246 forced shots, the
    # most this weapon had EVER killed was one, and the corridor had never held
    # two: the beam was angled into the dirt and died a little past the first
    # target. See _beam_ends.
    w8 = _World()
    w8.relics = _Rolls([0.5, 0.0])
    ace = Stickman(id=50, name="Nyx", x=200.0, y=600.0)
    ace.weapon, ace.relic = WEAPON_BFG, RELIC_BFG
    w8.agents.append(ace)
    w8.agents.extend(Stickman(id=51 + i, name=f"Row{i}",
                              x=320.0 + i * 60.0, y=600.0) for i in range(4))
    w8.animals.append(_Animal("wolf", 200.0 + BFG_RANGE - 40.0))
    out = _bfg_shoot(ace, w8, w8.animals[0])
    standing = [a.name for a in _alive_agents(w8)]
    print(f"four in the line: {out} | left standing: {standing}")
    assert out["shot"] == "fired" and out["heedless"] is True, out
    assert len(out["friends"]) == 4, out
    assert standing == ["Nyx"], standing
    assert w8.stats.get("bfg_friendly_deaths") == 4, w8.stats
    assert any("were in the light" in ln for ln in w8.lines), w8.lines
    assert any("One stickman is left standing" in ln for ln in w8.lines), w8.lines

    # Dragonscale walks out of the beam; leather does not. This is the one
    # interaction BFG_DAMAGE = MAX_HEALTH * 9.0 exists for.
    w4 = _World()
    w4.relics = _Rolls([0.5, 0.0])
    shooter = Stickman(id=23, name="Gil", x=300.0, y=600.0)
    shooter.weapon = WEAPON_BFG
    scaled = Stickman(id=24, name="Hana", x=520.0, y=600.0)
    scaled.armour = 1.0                     # ARMOUR_DRAGONSCALE
    leathered = Stickman(id=25, name="Ivo", x=560.0, y=600.0)
    leathered.armour = ARMOUR_LEATHER
    w4.agents.extend([shooter, scaled, leathered])
    w4.animals.append(_Animal("wolf", 700.0))
    _bfg_shoot(shooter, w4, w4.animals[0])
    print(f"through armour: dragonscale {scaled.health:.0f} hp alive={scaled.alive}"
          f" | leather {leathered.health:.0f} hp alive={leathered.alive}")
    assert scaled.alive and abs(scaled.health - 10.0) < 1e-6, scaled.health
    assert not leathered.alive, "leather survived a BFG beam"

    # THE MISFIRE, AS CORRECTED - EXACTLY ONE DEATH AND IT IS THE WIELDER'S.
    #
    # This block used to assert the PRE-CORRECTION spec: a 70 px blast that
    # killed ``close_friend`` and ``near_beast`` and spared anybody past
    # ``BFG_BLAST_R``. The author corrected that to wielder-only - "just
    # himself" - and _bfg_misfire was changed, but the test was not, so it sat
    # here red, defending the behaviour that had been rejected. A red test
    # asserting a rejected spec is worse than no test: the obvious way to make
    # it pass is to put the blast back.
    #
    # ``close_friend`` stands 40 px away, well inside the old radius, and the
    # requirement is now that they LIVE. ``BFG_BLAST_R`` is still imported and
    # still used - render/items draws the flash at that size - so the number is
    # kept in the geometry here to prove the light is 70 px wide and the damage
    # is not. The relic must also be gone BEFORE anybody dies, or a wielder who
    # kills themselves with it drops it again through World._reap_dead.
    w5 = _World()
    w5.relics = _Rolls([0.0])               # < BFG_MISFIRE
    boom = Stickman(id=26, name="Jed", x=300.0, y=600.0)
    boom.weapon = WEAPON_BFG
    boom.relic = RELIC_BFG
    close_friend = Stickman(id=27, name="Kit", x=340.0, y=600.0)     # 40 px
    clear_friend = Stickman(id=28, name="Lum", x=300.0 + BFG_BLAST_R + 30.0,
                            y=600.0)
    w5.agents.extend([boom, close_friend, clear_friend])
    near_beast = _Animal("wolf", 330.0)
    w5.animals.append(near_beast)
    out = _bfg_shoot(boom, w5, near_beast)
    print("misfire:", out, f"| wielder alive={boom.alive} weapon={boom.weapon!r} "
          f"relic={getattr(boom, 'relic', '')!r} "
          f"| 40px away: {close_friend.name} alive={close_friend.alive}")
    assert out["shot"] == "misfire", out
    assert out["friends"] == ["Jed"] and out["kills"] == 0, out
    assert boom.weapon == WEAPON_NONE and getattr(boom, "relic", "") == RELIC_NONE
    assert not boom.alive, "the wielder walked away from their own misfire"
    assert close_friend.alive, "a misfire took the neighbour: that spec was rejected"
    assert clear_friend.alive, "a misfire took a bystander"
    assert near_beast.alive, "a misfire is not a shot; it killed an animal"
    assert [a.name for a in w5.agents if not a.alive] == ["Jed"], \
        "a misfire is exactly one death and it is the wielder's"
    assert w5.stats.get("bfg_shots") is None, "a misfire counted as a shot fired"
    assert w5.stats.get("bfg_misfires") == 1, w5.stats
    assert w5.relics.beams and w5.relics.beams[-1]["misfire"], w5.relics.beams
    assert any("comes apart" in ln for ln in w5.lines), w5.lines

    # No relic stream means no shot. There is no fallback RNG anywhere in this
    # feature - see _relic_roll.
    w6 = _World()
    solo = Stickman(id=29, name="Mab", x=300.0, y=600.0)
    solo.weapon = WEAPON_BFG
    w6.agents.append(solo)
    w6.animals.append(_Animal("wolf", 500.0))
    assert _bfg_shoot(solo, w6, w6.animals[0])["shot"] == "nostream"
    assert w6.animals[0].alive, "fired without a seeded stream"

    # --- the rates, against a real seeded stream ---------------------------
    _rng = _random.Random(90210)

    class _RngRolls:
        def roll(self, lo: float = 0.0, hi: float = 1.0) -> float:
            return _rng.uniform(lo, hi)

    def _rate_run(with_bystander: bool, n: int) -> dict[str, int]:
        wr = _World()
        wr.relics = _RngRolls()
        g = Stickman(id=30, name="Rue", x=300.0, y=600.0)
        wr.agents.append(g)
        beast = _Animal("wolf", 520.0)
        wr.animals.append(beast)
        friend = Stickman(id=31, name="Sol", x=600.0, y=600.0)
        if with_bystander:
            wr.agents.append(friend)
        tally: dict[str, int] = {}
        for _ in range(n):
            g.alive, g.health = True, MAX_HEALTH
            g.weapon, g.relic = WEAPON_BFG, RELIC_BFG
            g.__dict__["_bfg_next"] = 0.0
            friend.alive, friend.health = True, MAX_HEALTH
            beast.alive, beast.health = True, beast.max_health
            shot = _bfg_shoot(g, wr, beast)["shot"]
            tally[shot] = tally.get(shot, 0) + 1
        return tally

    N = 4000
    clear_tally = _rate_run(False, N)
    mis = clear_tally.get("misfire", 0) / N
    print(f"clear line, {N} shots: {clear_tally} -> misfire {mis:.3%} "
          f"(BFG_MISFIRE {BFG_MISFIRE:.0%})")
    assert 0.06 < mis < 0.10, mis
    assert clear_tally.get("held", 0) == 0, "held fire with nobody in the way"

    blocked = _rate_run(True, N)
    fired = blocked.get("fired", 0)
    held = blocked.get("held", 0)
    heed = fired / max(1, fired + held)
    print(f"friendly in the line, {N} shots: {blocked} -> heedless {heed:.3%} "
          f"(BFG_HEEDLESS {BFG_HEEDLESS:.0%})")
    assert 0.17 < heed < 0.23, heed
    assert 0.06 < blocked.get("misfire", 0) / N < 0.10, blocked

    # --- scoring and the flee gate -----------------------------------------
    w7 = _World()
    w7.relics = _Rolls([0.5])
    carrier = Stickman(id=40, name="Nia", x=300.0, y=600.0)
    carrier.weapon = WEAPON_BFG
    w7.agents.append(carrier)
    prowler = _Animal("wolf", 300.0 + FLEE_RADIUS - 20.0)   # inside FLEE_RADIUS
    w7.animals.append(prowler)
    sc = score_combat(carrier, w7)
    print("bfg carrier, wolf at 90px:", {k: round(v, 2) for k, v in sc.items()})
    assert sc["FleeAnimal"] == 0.0, "the carrier ran from something it can shoot"
    assert sc["FireBfg"] > 0.95, sc
    assert sc["FightAnimal"] == 0.0 and sc["ThrowSpear"] == 0.0, \
        "a bfg carrier tried to use a spear it does not have"
    # ...but a wounded one still runs, and a child never fights.
    carrier.health = MAX_HEALTH * 0.2
    hurt_sc = score_combat(carrier, w7)
    assert hurt_sc["FireBfg"] == 0.0 and hurt_sc["FleeAnimal"] > 0.9, hurt_sc
    carrier.health = MAX_HEALTH
    carrier.role = "child"
    assert score_combat(carrier, w7)["FireBfg"] == 0.0
    assert score_combat(carrier, w7)["FleeAnimal"] > 0.9, "a child stood its ground"
    carrier.role = "gatherer"
    # An agent with no gun scores exactly 0.0 and therefore draws no tiebreak
    # from world.pyrng - which is what keeps a relic-free colony bit-identical
    # to the build that shipped before relics existed.
    assert score_combat(hunter, w2)["FireBfg"] == 0.0

    # and the whole action end to end, through Action.update
    act = make_combat_action("FireBfg", carrier, w7)
    assert act is not None and act.kind == "FireBfg", act
    w7.relics = _Rolls([0.5, 0.0])
    for _ in range(int(30 * 6)):
        act.update(carrier, w7, dt)
        if act.finished:
            break
    print("FireBfg:", act, "| wolf alive:", prowler.alive)
    assert act.done and act.data.get("shot") == "fired", act.data
    assert not prowler.alive, "the action never actually fired"
    assert not _bfg_ready(carrier, w7), "the gun did not cool down"

    # --- CraftSpear must never write over the wyrm-gun ----------------------
    # ``agent.weapon`` is one slot. A colonist who found the BFG and then
    # decided to whittle a spear had WEAPON_BFG replaced by WEAPON_SPEAR with
    # nothing anywhere to notice: relic still read "bfg9000", _bfg() said no,
    # and the rarest item in the game was gone. Two lines to reproduce, which is
    # how long it stayed latent - no BFG had ever dropped in natural play.
    w9 = _World()
    w9.structures.create("stockpile", 600.0, 600.0, built=True)
    w9.stockpile.update({RES_WOOD: 20, RES_STONE: 20})
    keeper = Stickman(id=60, name="Osk", x=560.0, y=600.0)
    keeper.weapon, keeper.relic = WEAPON_BFG, RELIC_BFG
    w9.agents.append(keeper)
    assert _can_afford(w9, SPEAR_COST), "the guard has to be tested with wood in"
    assert score_combat(keeper, w9)["CraftSpear"] == 0.0, "scored a spear over the gun"
    assert make_combat_action("CraftSpear", keeper, w9) is None, "built one anyway"
    # ...and the third door: a save rehydrated mid-craft never passes back
    # through either of those, so the handler and _craft_apply refuse too.
    resumed = make_action("CraftSpear", phase="work", wt=CRAFT_SPEAR_TIME + 1.0)
    resumed.update(keeper, w9, dt)
    _craft_apply("CraftSpear", keeper)
    print(f"craft-over-relic: weapon={keeper.weapon!r} relic={getattr(keeper,'relic','')!r}")
    assert keeper.weapon == WEAPON_BFG and keeper.relic == RELIC_BFG, \
        "CraftSpear overwrote the wyrm-gun"
    # RetrieveSpear is the SECOND path to the same slot - constants.py's note on
    # WEAPON_BFG names both - and it ends in the same bare assignment. A carrier
    # must not trade the gun for a stick lying in the dirt.
    w9.spears = SpearRegistry(seed=5)
    w9.spears.add(Spear(id=1, x=580.0, y=600.0, state=STATE_GROUND))
    assert w9.spears.on_ground(), "the fixture put no spear on the ground"
    assert _retrievable_spear(keeper, w9) is None, "walked off to swap the gun"
    assert score_throw(keeper, w9)["RetrieveSpear"] == 0.0
    lift = make_action("RetrieveSpear", phase="take", target=1)
    lift.update(keeper, w9, dt)
    assert keeper.weapon == WEAPON_BFG, "RetrieveSpear overwrote the wyrm-gun"
    assert w9.spears.on_ground(), "took the spear and threw it away"

    # An ordinary unarmed colonist is unaffected by either gate: it is
    # _combat_capable, which is exactly _armed for anybody who has never held a
    # relic.
    plain = Stickman(id=61, name="Pell", x=560.0, y=600.0)
    w9.agents.append(plain)
    assert score_combat(plain, w9)["CraftSpear"] > 0.0, "broke ordinary crafting"
    assert make_combat_action("CraftSpear", plain, w9) is not None
    assert _retrievable_spear(plain, w9) is not None, "broke ordinary retrieval"
    assert score_throw(plain, w9)["RetrieveSpear"] > 0.0

    # --- a real World: the death has to be a real death ---------------------
    from .world import World as _World_

    rw = _World_(seed=5)
    rw.ufo.next_in = 1e9                    # no second event in the books
    rw.relics = _Rolls([0.5, 0.0])
    crowd = rw.population.alive_agents()
    shooter2, victim2 = crowd[0], crowd[1]
    shooter2.weapon = WEAPON_BFG
    shooter2.relic = RELIC_BFG
    for gap in (90.0, 120.0, 150.0, 180.0, -90.0, -120.0, -150.0):
        victim2.x = _clamp_x(shooter2.x + gap)
        victim2.y = rw.terrain.ground_y(victim2.x)
        bx0, by0, btx, bty = _beam_ends(shooter2, rw, victim2)
        if any(o is victim2 for o, _ in beam_sweep(rw, shooter2, bx0, by0, btx, bty)):
            break
    before = rw.reconcile()
    died_before = rw.stats["died"]
    assert before["ok"] == 1 and before["residual"] == 0, before
    out = _bfg_shoot(shooter2, rw, victim2)
    for _ in range(int(30 * 12)):
        rw.tick(dt)
    after = rw.reconcile()
    graves = [g for g in rw.props.all_of("grave")
              if str(getattr(g, "state", {}).get("name", "")) == victim2.name]
    print(f"real world: {out} | {victim2.name} alive={victim2.alive} "
          f"cause={victim2.death_cause!r} died {died_before}->{rw.stats['died']} "
          f"graves for them={len(graves)} residual={after['residual']}")
    assert out["shot"] == "fired", out
    assert not victim2.alive and victim2.death_cause == CAUSE_DISINTEGRATED
    assert rw.stats["died"] > died_before, "a beam death went uncounted"
    assert graves, "a beam death got no grave"
    assert after["ok"] == 1 and after["residual"] == 0, after

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
