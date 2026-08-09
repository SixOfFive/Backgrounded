"""Multi-stage buildable structures and the colony structure registry.

Pure python: no pygame, no rendering. The renderer *reads* ``kind``, ``stage``,
``progress``, ``variant``, ``state`` and :meth:`Structure.scale` to decide what
to draw and must never write to them.

A finished hut is not a fixed object. It keeps a ``standing_t`` (seconds spent
standing, finished) and a derived ``growth`` 0..1 blended from that age and from
how full the colony's stores are. ``growth`` drives :meth:`Structure.scale` and
:meth:`Structure.capacity`, so a long-lived, well-fed camp becomes a settlement
of bigger huts that house more people - and one that eats its stores down
visibly contracts again. Growth is eased, never snapped: a single haul arriving
must not pop a building bigger.

A structure is built in ``max_stage`` visible stages. Each stage has its own
slice of the total resource cost; builders haul that slice to the site
(:meth:`Structure.deliver`) and only then does :meth:`Structure.advance` make
progress. That is what lets a save taken mid-build resume mid-build.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..constants import (
    BARRICADE_DAMAGE,
    BARRICADE_RANGE,
    BARRICADE_SPIKE_H,
    BONFIRE_GARBAGE_CAP,
    BONFIRE_LIGHT_COLOR,
    BONFIRE_LIGHT_INTENSITY,
    BONFIRE_LIGHT_RADIUS,
    BONFIRE_SEC_PER_GARBAGE,
    HUT_AGE_WEIGHT,
    HUT_GROWTH_AGE_SEC,
    HUT_GROWTH_STORE_REF,
    HUT_SCALE_MAX,
    HUT_SCALE_MIN,
    HERMIT_FIRE_BURN,
    HERMIT_FIRE_WOOD,
    HERMIT_HUT_WOOD,
    HUT_STORE_WEIGHT,
    RENDER_H,
    RES_FIBRE,
    RES_STONE,
    RES_WOOD,
    WORLD_W,
)

# The lookout/rock-tower numbers are Lane 0's, in constants.py, and this module
# is not allowed to edit that file. Imported behind ImportError with fallbacks
# that are the SAME literals so this module is correct whether or not the
# constants commit has landed yet: a merge order must not change behaviour.
try:                                            # pragma: no cover - see above
    from ..constants import (
        ROCK_COOLDOWN,
        ROCK_DAMAGE,
        ROCK_DROP_CEILING,
        ROCK_DROP_RANGE,
        TOWER_AMMO_MAX,
        TOWER_AMMO_PER_STONE,
        TOWER_UPGRADE_COST,
        TOWER_UPGRADE_WORK,
    )
except ImportError:                             # pragma: no cover - see above
    ROCK_COOLDOWN = 2.4
    ROCK_DAMAGE = 30.0
    ROCK_DROP_CEILING = 40.0
    ROCK_DROP_RANGE = 46.0
    TOWER_AMMO_MAX = 8
    TOWER_AMMO_PER_STONE = 2
    TOWER_UPGRADE_COST = {RES_WOOD: 6, RES_STONE: 12}
    TOWER_UPGRADE_WORK = 12.0

# The hermit's workbench numbers are the constants lane's, in the same file this
# module is not allowed to edit. Separate try block rather than folded into the
# one above on purpose: a single block falls back as a UNIT, so one missing name
# would quietly swap the tower's tuning out too. Same discipline as above - the
# fallbacks are the SAME literals, so the spec below is identical whether or not
# that commit has landed, and a merge order cannot change what a bench costs.
try:                                            # pragma: no cover - see above
    from ..constants import (
        HERMIT_WORKBENCH_COST,
        HERMIT_WORKBENCH_WORK,
        KIND_HERMIT_WORKBENCH,
    )
except ImportError:                             # pragma: no cover - see above
    KIND_HERMIT_WORKBENCH = "hermit_workbench"
    HERMIT_WORKBENCH_COST = {RES_WOOD: 8}
    HERMIT_WORKBENCH_WORK = 7.0

log = logging.getLogger(__name__)


def _ground_clearance(world: Any, obj: Any) -> float:
    """Px between *obj* and the ground under it; 0.0 for anything without one.

    The fallback used when :mod:`.throwing` cannot be imported. Answering 0.0 -
    "it is on the ground" - is the answer this package gave before altitude
    existed, so a failure here restores the OLD behaviour rather than inventing
    a new one, and the only thing that changes is that flyers become reachable.
    """
    alt = getattr(obj, "alt", None)
    if not isinstance(alt, (int, float)) or isinstance(alt, bool):
        return 0.0
    f = float(alt)
    return 0.0 if f != f or f in (float("inf"), float("-inf")) else f


_CLEARANCE = None


def _clearance():
    """Resolve ``throwing.clearance`` once, lazily, and cache it.

    NOT a module-level import, and that is load-bearing rather than style:
    ``structures -> throwing -> entities -> actions -> structures`` is a real
    cycle (actions.py imports this module at its top level), so importing
    throwing here at import time is an ImportError on a cold start, not a
    subtle problem. Resolved on first use instead, when every module is built.

    Cached in a module global because ``_tick_barricade`` is a per-frame path
    walked once per barricade per tick; re-importing there would be a
    ``sys.modules`` lookup every frame for a function that never changes.
    Falls back to :func:`_ground_clearance`, which is semantically identical
    for everything that does not fly.
    """
    global _CLEARANCE
    if _CLEARANCE is None:
        try:
            from .throwing import clearance      # noqa: PLC0415 - see above
            _CLEARANCE = clearance
        except Exception:
            _CLEARANCE = _ground_clearance
    return _CLEARANCE


def _direct_hurt(world: Any, obj: Any, amount: float, by: Any = None) -> bool:
    """Fallback damage: call the target's own ``hurt``. Never raises.

    Exactly what :meth:`Structure._tick_barricade` has always done, so a failure
    to import combat_actions restores the behaviour this module had before the
    rock tower existed instead of inventing a new one. It still respects a
    ``hurt`` that refuses damage - which is the whole point: ``Dragon.hurt``
    discards everything while the dragon is unsated, and a weapon that reached
    past that would be the only one in the game that kills an unfed dragon.
    """
    fn = getattr(obj, "hurt", None)
    if not callable(fn):
        return False
    try:
        return bool(fn(float(amount)))
    except Exception:
        return False


_HURT = None


def _hurt():
    """Resolve ``combat_actions.hurt_animal`` once, lazily, and cache it.

    Same reasoning as :func:`_clearance`, and the same cycle:
    ``structures -> combat_actions -> actions -> structures``. combat_actions is
    the right place for a killing blow to land because it is the one that marks
    the target dead, registers the death cause and lets the animals module claim
    its hides - a rock that wrote ``hp`` itself would leave a corpse nothing had
    accounted for. Falls back to :func:`_direct_hurt`, which is what the spikes
    do, so the tower degrades to "damage lands, bookkeeping does not".

    Draws no rng of any kind on either path - checked, because ``_tick_upkeep``
    is a passive per-frame branch and a single draw there would re-phase
    ``world.pyrng`` for the whole colony.
    """
    global _HURT
    if _HURT is None:
        try:
            from .combat_actions import hurt_animal   # noqa: PLC0415 - see above
            _HURT = hurt_animal
        except Exception:
            _HURT = _direct_hurt
    return _HURT

__all__ = [
    "Structure",
    "StructureRegistry",
    "StructureSpec",
    "STRUCTURE_SPECS",
    "STRUCTURE_KINDS",
    "CROSSING_KINDS",
    "NON_SETTLEMENT_KINDS",
    "FIRE_KINDS",
    "KIND_LABELS",
    "structure_spec",
    "plan_ladder",
    "FIRE_LIGHT_COLOR",
    "BURN_LIGHT_COLOR",
    "HUT_MATERIALS",
    "HUT_DEFAULT_MATERIAL",
    "HUT_UPGRADE_COST",
    "HUT_UPGRADE_WORK",
    "HUT_STONE_MAX_HP",
    "TOWER_MATERIALS",
    "TOWER_ROCK_MAX_HP",
    "KIND_MATERIALS",
    "UPGRADE_SPECS",
    "SPIKED_TARGET_SOURCES",
    "materials_for",
]

#: Kinds that write themselves into Terrain's effective-surface overlay when
#: they finish, and take themselves back out when they fall down.
CROSSING_KINDS: tuple[str, ...] = ("bridge", "ladder")

#: Buildings that are NOT part of "where the colony is", for the purpose of
#: averaging structure positions. Graves because they are sited out of the way;
#: the hermit's hut because being a long way from the settlement is the entire
#: point of it, and folding it into the mean would move the settlement toward it.
#:
#: That is not a tidiness argument, it is two feedback loops. ``colony_center``
#: below is what render/camera.py falls back to, what ``actions.stage_bounds``
#: derives the visible slice from, and what the quarry keep-out and the
#: barricade band are measured from - a hut 760 px out with six other
#: buildings standing drags all of them ~110 px toward the wilderness. And
#: ``actions.hermit_home`` is *derived* from the settlement mean, so a hut
#: inside that mean pushes the next camp further out again, every pass.
#: ``actions.OUTWORK_KINDS`` carries the same exclusion for
#: ``settlement_center``; the two lists are separate because they answer to
#: different callers, and they agree about this kind on purpose.
#:
#: Both lookout kinds are here for the same reason and it is the sharper case:
#: a colony lookout is deliberately sited LOOKOUT_STANDOFF (520 px) outside the
#: settlement on each side, so with six other buildings standing a pair of them
#: drags the mean ~150 px outward - and then the next site, the camera fallback,
#: the quarry keep-out, the barricade band and hermit_home all move with it,
#: every pass, outward again. The hermit's is further out still.
#:
#: ``hermit_workbench`` is here for the plain version of the hut's reason and it
#: is not optional. It is staked AT his camp, so it is one more x in the mean
#: that ``actions.hermit_home`` is derived from - leave it out and every
#: director pass sites the next camp a little further into the wilderness than
#: the last, which is the runaway the hut's entry above describes with a smaller
#: step and the same sign. It is also the kind most likely to be missed, because
#: it is the only one of his four that nothing else in the sim looks up by name.
NON_SETTLEMENT_KINDS: tuple[str, ...] = (
    "grave", "hermit_hut", "hermit_fire", "lookout", "hermit_lookout",
    KIND_HERMIT_WORKBENCH,
)

#: Kinds that BEHAVE like a fire - they hold fuel, they can be lit and stoked,
#: they burn it down over time and they cast light while they do.
#:
#: This is the one list the hermit's fire is deliberately IN, and it is worth
#: being clear about why it does not contradict the long note above or the one
#: on ``hermit_fire``'s spec. Those are about CENSUS: how many fires has the
#: colony got, how many roofs, where is the settlement. His fire must be
#: invisible to every one of those, and a separate kind makes it invisible by
#: construction - ``reg.count("firepit")`` cannot see a ``hermit_fire`` however
#: hard it looks, so ``behavior.next_build_kind``'s ``built("firepit") < 1``
#: still sends the colony to build its own.
#:
#: This list is about PHYSICS, and the physics of a fire are the same fire
#: wherever it is: ``light_fire``, ``stoke``, ``fire_active`` and ``_tick_fuel``
#: are guards on the Structure's own methods, and a fire kind left out of them
#: is a fire that can never be lit. That was the first cut and it shipped a
#: hermit sitting next to a permanently cold ring of stones.
#:
#: ``feed_garbage``, ``garbage_left`` and ``bonfire_active`` are POINTEDLY not
#: keyed off this list and still test ``kind == "firepit"``. The bonfire is the
#: payoff of the colony's litter sweep, the hermit does not sweep (CleanLitter
#: is zeroed for him) and no villager can route rubbish to his fire (they look
#: up "firepit"), so a hermit bonfire is unreachable in play - and a state key
#: that can only ever arrive from a corrupt save is a state key worth refusing.
FIRE_KINDS: tuple[str, ...] = ("firepit", "hermit_fire")

#: What the chronicle calls a building. Only kinds whose bare identifier reads
#: badly in a sentence need a row; everything else IS its own name ("the hut
#: burned to the ground" is fine, "the hermit_hut burned to the ground" is not,
#: and that line is the one the user sees when a dragon comes).
#:
#: Deliberately not a general i18n table and deliberately not on StructureSpec.
#: It is consulted by :meth:`Structure.label` and nothing else reads it, so a
#: kind added without a row degrades to the identifier, exactly as today.
KIND_LABELS: dict[str, str] = {
    "hermit_hut": "hermit's hut",
    "hermit_fire": "hermit's fire",
    "hermit_lookout": "hermit's lookout",
    KIND_HERMIT_WORKBENCH: "hermit's workbench",
    "lookout": "lookout",
    "firepit": "firepit",
}

# ------------------------------------------------------------------ tuning --
FIRE_LIGHT_COLOR: tuple[int, int, int] = (255, 168, 82)
BURN_LIGHT_COLOR: tuple[int, int, int] = (255, 112, 40)

BURN_DPS = 9.0              # hp lost per second while burning
BURN_SPREAD_CHANCE = 0.35   # per second, chance a burning structure ignites a neighbour
BURN_NEIGHBOUR_DIST = 70.0
FIRE_FUEL_BURN = 1.0 / 240.0   # a full firepit burns for ~4 minutes
FIRE_STOKE_PER_WOOD = 0.34
RUIN_LINGER = 240.0         # seconds a collapsed structure stays as rubble

#: Garbage burns *instead of* wood, not alongside it: while there is any in the
#: pit, `_tick_upkeep` spends garbage and leaves ``state["fuel"]`` alone. That is
#: what "it uses that as fuel primarily" means mechanically - a colony that
#: sweeps up is a colony that saves firewood - and it is also why the transition
#: is worth drawing: the pit is running on something different.
#:
#: Nothing comes back out. No charcoal, no ash resource, no stockpile credit;
#: `feed_garbage` returns units accepted and touches nothing but this structure.
GARBAGE_PER_SEC = 1.0 / max(0.5, float(BONFIRE_SEC_PER_GARBAGE))
#: Seconds the bonfire's light takes to reach full reach, and how many units of
#: garbage from the end it starts dying back down. Both exist so the night does
#: not step-change by 250 px of light radius inside one frame - the whole effect
#: is meant to read as a fire being built up and burning itself out.
BONFIRE_RAMP_SEC = 2.0
BONFIRE_FADE_UNITS = 4.0

#: Sleepers a hut holds at growth 0 and at growth 1. A well-established hut in
#: a well-stocked camp really is a bigger building, so it houses more people.
HUT_CAPACITY_MIN = 2
HUT_CAPACITY_MAX = 5
#: Fraction of the remaining gap `growth` closes toward its target per second.
#: Deliberately slow: a single haul arriving must not pop the building bigger,
#: and running the stores down has to read as a settlement contracting rather
#: than a building blinking. ~5%/s is a ~20 s time constant.
GROWTH_LERP_RATE = 0.05

#: What a hut's walls can be made of. The colony's building tech tier is a
#: ``material`` key on :attr:`Structure.state`, *not* a second structure kind:
#: a new kind would mean chasing eight literal ``"hut"`` comparisons across five
#: sim files (the birth gate in world.py among them - miss it and births stall
#: silently) plus four hardcoded tables, all to say "same building, different
#: walls". A state key round-trips through :func:`_json_safe` for free and
#: breaks no ``kind == "hut"`` comparison anywhere.
#:
#: This tuple is the sim-side mirror of ``render.atlas.HUT_KINDS``. sim/ must
#: never import render/, so the two are kept in step by hand. The failure mode
#: of drift is cosmetic and one-directional: a material named here that the
#: atlas does not know falls through ``Atlas.resolve`` and draws as timber. It
#: cannot crash.
HUT_MATERIALS: tuple[str, ...] = ("wood", "stone")
#: What a hut is when nothing says otherwise - including every save written
#: before any of this existed, where the key is simply absent.
HUT_DEFAULT_MATERIAL: str = "wood"
#: Stone to re-wall one finished timber hut. Stone only, deliberately: it keeps
#: the cost legible ("a stone hut costs stone") and off the wood/fibre economy
#: the hut ladder and the firepit already compete for. The real gate is not this
#: price but the *surplus* rule in behaviour - stone no pending build has a
#: claim on - which is why a 10-stone upgrade never starves a 4-stone hut.
HUT_UPGRADE_COST: dict[str, int] = {RES_STONE: 10}
#: Seconds of applied work once the stone is on site, mirroring
#: ``StructureSpec.build_time`` for one stage. Longer than a hut stage (9 s)
#: because it is the whole building, not a quarter of one.
HUT_UPGRADE_WORK: float = 14.0
#: A stone hut's max hp, up from the timber spec's 130. Note this changes what
#: counts as damaged: ``StructureRegistry.damaged`` compares hp/max_hp, so a
#: stone hut sitting at 130 hp now reads as damaged where the identical timber
#: hut did not. That is intended - a stone hut has more to lose - but it does
#: mean slightly more repair traffic after the first upgrade lands.
HUT_STONE_MAX_HP: float = 190.0

#: What a lookout's platform can be made of. Same mechanism as
#: :data:`HUT_MATERIALS` - a ``material`` key on :attr:`Structure.state`, not a
#: second kind - and for the sharper version of the same reason: a rock tower
#: must still BE a lookout to `_free_tower`, to `assign_roles`' tower census and
#: to `has_room()` for the man standing on it. A second kind would have to be
#: added back to every one of those by hand, and the one that fails silently is
#: the role census: no lookout role assigned, so nobody ever climbs it.
#:
#: Both ladders start at ``"wood"``, which is what lets :meth:`can_upgrade` ask
#: one question ("is it still on its first material?") for every kind.
TOWER_MATERIALS: tuple[str, ...] = ("wood", "rock")
#: The upgraded tier's name, and the one value :func:`_holds_ammo` accepts. A
#: tower is only allowed to store rocks once it is actually made of them.
ROCK_MATERIAL: str = "rock"
#: A rock tower's max hp, up from the timber spec's 120. Same caveat as
#: HUT_STONE_MAX_HP: `StructureRegistry.damaged` compares hp/max_hp, so a tower
#: sitting at 120 hp now reads as damaged where the timber one did not.
TOWER_ROCK_MAX_HP: float = 160.0

#: Which material ladder each kind climbs. Anything absent answers
#: :data:`HUT_MATERIALS`, which is exactly what :meth:`Structure.material` and
#: :func:`_sanitise_upgrade` did for every kind before towers existed - so an
#: unlisted kind behaves byte-for-byte as it always has.
KIND_MATERIALS: dict[str, tuple[str, ...]] = {
    "hut": HUT_MATERIALS,
    "lookout": TOWER_MATERIALS,
}

#: Wood the hermit puts into his own lookout. Sized between his hut
#: (HERMIT_HUT_WOOD) and the colony's 22-wood watchtower: it is a ladder and a
#: plank platform lashed to a tree, cut and carried by one man.
HERMIT_LOOKOUT_WOOD: int = 14

#: Per-kind upgrade spec: ``kind -> (to_material, cost, work_seconds,
#: new_max_hp)``. This table IS the answer to "can this be upgraded" - a kind
#: absent from it cannot be, which is how :meth:`Structure.can_upgrade` stopped
#: being a hardcoded ``kind == "hut"`` without any caller learning a new
#: question.
#:
#: ``hermit_lookout`` is deliberately NOT here. The upgrade is hauled stone out
#: of the colony's stockpile, published through the colony-wide
#: ``world.upgrade_job`` channel with concurrency 1; the hermit builds out of
#: what he cuts himself and has no claim on that channel. Adding him would put
#: a job 800 px into the wilderness at the head of the colony's only upgrade
#: queue.
UPGRADE_SPECS: dict[str, tuple[str, dict[str, int], float, float]] = {
    "hut":     ("stone", HUT_UPGRADE_COST,   HUT_UPGRADE_WORK,   HUT_STONE_MAX_HP),
    "lookout": (ROCK_MATERIAL, TOWER_UPGRADE_COST, TOWER_UPGRADE_WORK,
                TOWER_ROCK_MAX_HP),
}

#: Registry attribute names a passive structure weapon sweeps, IN ORDER.
#: Append-only. ``"raiders"`` is present from day one and costs a world without
#: one exactly nothing: :meth:`Structure._spiked_targets` skips an attribute
#: that is missing or has no ``alive()``, so the list it returns is identical
#: object-for-object and in the same order as before this entry existed.
SPIKED_TARGET_SOURCES: tuple[str, ...] = ("animals", "dragons", "raiders")


def materials_for(kind: str) -> tuple[str, ...]:
    """The material ladder *kind* climbs; :data:`HUT_MATERIALS` if it has none.

    Never raises: a non-string kind out of a corrupt save answers the default
    rather than exploding inside a loader.
    """
    try:
        return KIND_MATERIALS.get(kind, HUT_MATERIALS)
    except TypeError:                   # unhashable kind from a hand-edited save
        return HUT_MATERIALS


def _holds_ammo(kind: str, material: str) -> bool:
    """Is a ``kind`` made of ``material`` allowed to store rocks?

    Both halves matter. A timber lookout is a lookout with no rocks in it, and
    ``ammo`` on a hut, a grave or a bridge is a hand-edited save, not a feature.
    Answering here rather than on :class:`Structure` is what lets
    :func:`_sanitise_ammo` decide before a Structure exists.
    """
    try:
        spec = UPGRADE_SPECS.get(kind)
    except TypeError:
        return False
    return bool(spec) and spec[0] == ROCK_MATERIAL and material == ROCK_MATERIAL


@dataclass(frozen=True)
class StructureSpec:
    """Static per-kind data. Never mutated at runtime."""

    kind: str
    max_stage: int
    max_hp: float
    cost: dict[str, int]
    width: float
    height: float
    capacity: int = 0
    flammable: bool = True
    build_time: float = 8.0     # seconds of applied work per stage
    spacing: float = 46.0       # minimum gap the director keeps from neighbours
    variants: int = 3


STRUCTURE_SPECS: dict[str, StructureSpec] = {
    "firepit": StructureSpec(
        "firepit", 2, 60.0, {RES_WOOD: 6, RES_STONE: 6},
        width=28.0, height=12.0, capacity=0, flammable=False,
        build_time=5.0, spacing=54.0, variants=2,
    ),
    # A finished hut ignores `capacity` below - see Structure.capacity(), which
    # scales it HUT_CAPACITY_MIN..MAX with the hut's growth. The value is kept
    # only as the fallback for a hut that somehow has no growth state.
    "hut": StructureSpec(
        "hut", 4, 130.0, {RES_WOOD: 18, RES_STONE: 4, RES_FIBRE: 6},
        width=46.0, height=38.0, capacity=3, flammable=True,
        build_time=9.0, spacing=58.0, variants=4,
    ),
    # THE HERMIT'S HUT, AND IT IS ITS OWN KIND RATHER THAN A FLAGGED HUT.
    # That is the opposite of the choice HUT_MATERIALS documents above, and for
    # the opposite reason. A stone hut wants to BE a hut everywhere: it houses
    # people, it satisfies the colony's demand for housing, it raises the birth
    # cap. This does none of those things, and every one of the eight literal
    # `kind == "hut"` comparisons that note complains about is a site where a
    # hermit's hut must NOT count:
    #
    #   world.py             the birth gate. A hut counted here raises the
    #                        colony's population cap on a house nobody lives in.
    #   behavior/actions     `nearest_structure(..., "hut")` - how a tired
    #                        colonist finds a bed. Counted, villagers would trek
    #                        800 px to sleep in it.
    #   behavior             the hut ladder and want_huts. Counted, the hut
    #                        satisfies the colony's own demand for roofs and
    #                        they stop building for themselves.
    #   behavior/actions     the build-queue entry and the celebration table.
    #   dragons              the stone-hut tech gate's census.
    #
    # A flagged hut opts IN to all eight and every one then needs an exclusion
    # bolted on: eight chances to miss one, and the one that fails - the birth
    # gate - fails SILENTLY. A separate kind opts OUT of all eight by
    # construction, and the only places that have to learn about it are the two
    # hermit-only code paths written for it anyway (he sleeps in it, he builds
    # it). The failure modes are not symmetric either: a site missed here means
    # "the hut is not special-cased somewhere", which is cosmetic. A site
    # missed the other way means the colony's own arithmetic is wrong.
    #
    # Three stages, wood only, one occupant, and small: it is a lean-to a single
    # man put up on his own, not village housing, and render/atlas draws it as
    # one. `flammable=True` is not an oversight - see events._FLAMMABLE, which
    # derives from this flag, and dragons._raze_target, which takes the nearest
    # built structure. It can burn, and he rebuilds it.
    "hermit_hut": StructureSpec(
        "hermit_hut", 2, 90.0, {RES_WOOD: HERMIT_HUT_WOOD},
        width=34.0, height=30.0, capacity=1, flammable=True,
        build_time=8.0, spacing=40.0, variants=2,
    ),
    # HIS FIRE, AND IT IS ITS OWN KIND FOR THE SAME REASON THE HUT IS.
    # The note on `hermit_hut` above lists the eight `kind == "hut"` sites a
    # hut must not be counted by; this kind's list is shorter and one entry on
    # it is worse than any of them:
    #
    #   behavior.next_build_kind   `built("firepit") < 1` is the FIRST thing the
    #                              colony ever builds. A hermit fire counted
    #                              there and the colony never lays its own
    #                              hearth at all - and warmth is not a nicety in
    #                              this sim, it is the thing that stops a
    #                              blizzard emptying the map. It fails silently:
    #                              nothing errors, the colony just quietly
    #                              freezes 700 px from a fire it may not use.
    #                              The same line at `min(2, 1 + pop // 6)` costs
    #                              the colony its second pit the same way.
    #   actions/behavior           `nearest_structure(.., "firepit")` - warmth,
    #                              cooking and where swept litter gets burned.
    #                              Counted, a villager on the colony's edge
    #                              walks the whole standoff to warm himself at a
    #                              hermit's fire, which is the mutual gate the
    #                              user asked for failing in the visible
    #                              direction.
    #
    # One stage, wood only, and no capacity: a ring of stones and an armful of
    # sticks, laid in an afternoon by one man out of what he cut himself. It is
    # `flammable=False` exactly as the colony's pit is - a firepit is the thing
    # that sets other things alight, not a thing that catches - so the dragon
    # that burns his hut down leaves his fire standing, and he still has
    # somewhere to cook while he rebuilds.
    #
    # It IS in FIRE_KINDS, which is what makes it a fire rather than a prop:
    # see the note there for why the physics list and the census lists point in
    # opposite directions on purpose.
    "hermit_fire": StructureSpec(
        "hermit_fire", 1, 40.0, {RES_WOOD: HERMIT_FIRE_WOOD},
        width=22.0, height=10.0, capacity=0, flammable=False,
        build_time=5.0, spacing=30.0, variants=2,
    ),
    # HIS LOOKOUT, AND IT IS ITS OWN KIND FOR THE THIRD TIME AND THE SAME
    # REASON. The list this one must stay out of is short and one entry on it
    # decides the whole colony tower arc:
    #
    #   behavior.assign_roles   staffs one lookout role per tower it counts. A
    #                           hermit tower counted there hands a villager a
    #                           lookout job on a platform 800 px out in the
    #                           wilderness that he then walks to - and, worse,
    #                           the colony reads its own perimeter as manned.
    #   behavior                the build rung and `_free_tower`. Counted, the
    #                           colony stops raising its own towers because the
    #                           hermit already built one.
    #
    # Two stages and wood only, shorter and flimsier than the colony's: it is a
    # ladder and a platform one man lashed together, not a framed tower a crew
    # raised. `flammable=False` for the trap the colony lookout documents above
    # AND for a second reason specific to this one: it stands beside a
    # `flammable=True` hermit_hut, which is very likely inside the 70 px
    # neighbour radius, so a flammable version would put a fresh numpy draw on
    # the stream every time his hut caught.
    #
    # It is deliberately absent from UPGRADE_SPECS - see the note there.
    # 62.0 is not a free number: render/atlas.py's TOWER_SPEC_H carries the same
    # value as the fallback it uses before `_sync_with_sim` can read this spec.
    # The sim is authoritative and the sync overwrites it, so a mismatch would
    # only show on a partial checkout - but a deck drawn 6 px off the platform
    # the sim perches a man on is exactly the class of bug trap 6 documents, and
    # it costs nothing to agree.
    "hermit_lookout": StructureSpec(
        "hermit_lookout", 2, 80.0, {RES_WOOD: HERMIT_LOOKOUT_WOOD},
        width=24.0, height=62.0, capacity=1, flammable=False,
        build_time=9.0, spacing=34.0, variants=2,
    ),
    # HIS WORKBENCH, AND IT IS ITS OWN KIND FOR THE FOURTH TIME. The three
    # notes above are the record of getting this right three times; this one is
    # the case where the tempting shortcut is worse than any of theirs.
    #
    # WHY IT EXISTS AT ALL is one line of somebody else's handler:
    # `combat_actions._h_craft`'s approach phase walks to
    # `nearest_structure(w, "stockpile")` and falls back to `colony_center(w)`.
    # Every spear the hermit has ever made was made standing in the middle of
    # the settlement, because there was nowhere else in the world to make one.
    #
    # THE SHORTCUT THAT LOOKS FREE is to let him stake an ordinary `stockpile`
    # at his camp and let that same line find it. Do not. `nearest_structure(w,
    # "stockpile")` is asked by EVERYBODY:
    #
    #   haulers        a stockpile 3000 px out in the wilderness is a legal
    #                  destination for the colony's harvest, and the nearest one
    #                  to a forager working the far treeline IS his.
    #   `_h_eat`       its approach resolves a pile to walk to. A starving
    #                  villager would be sent to a pile he may not draw from -
    #                  the till (`actions._TILL_ACTOR`) refuses him and nothing
    #                  errors, so he walks, is refused, and walks again.
    #   behavior       the build ladder counts stockpiles. One at the hermitage
    #                  and the colony stops raising its own.
    #
    # None of those fail loudly. A kind of its own is found by nobody who is not
    # looking for it, which is the whole trick, and it is why this is the fourth
    # kind rather than the first flag.
    #
    # IT HOLDS NOTHING, and `capacity=0` is only half of that. `_on_complete`
    # deliberately does NOT seed `state["shown"]` for it the way it does for a
    # `stockpile`: a bench is a surface to work on, not a container, and the
    # goods gate lives at the till, never on a building. He crafts out of his
    # own stash and the spear goes back into it.
    #
    # One stage, wood only, 8 = `actions.CARRY_CAP` (see HERMIT_WORKBENCH_COST):
    # one armful, one trip, no banked surplus, because nobody hauls for him.
    #
    # `flammable=True`, MATCHING HIS HUT rather than his fire or his tower, and
    # the cost of that is worth writing down because the `hermit_lookout` note
    # above declines it: `StructureRegistry.update` draws one `rng.random()` off
    # `world.rng` per (burning structure x in-range flammable candidate) pair
    # inside BURN_NEIGHBOUR_DIST (70 px), and this bench stands well inside 70 px
    # of a `flammable=True` hermit_hut. So while his hut burns, this kind adds a
    # draw per frame to the shared numpy stream. That is accepted here where it
    # was refused for the tower for one reason: the tower was fireproof anyway
    # for the colony lookout's reason, and this is a wooden trestle standing in
    # the open beside a burning house. A dragon that razes the hermitage should
    # not leave the bench untouched, and he rebuilds it out of one armful the
    # same way he rebuilds the hut. The extra draws are confined to colonies that
    # have a hermit whose camp is alight - a colony with no hermit has no bench
    # and draws exactly what it drew before.
    KIND_HERMIT_WORKBENCH: StructureSpec(
        KIND_HERMIT_WORKBENCH, 1, 45.0, dict(HERMIT_WORKBENCH_COST),
        width=26.0, height=16.0, capacity=0, flammable=True,
        build_time=float(HERMIT_WORKBENCH_WORK), spacing=30.0, variants=2,
    ),
    "wall": StructureSpec(
        "wall", 2, 170.0, {RES_STONE: 14, RES_WOOD: 4},
        width=30.0, height=30.0, capacity=0, flammable=False,
        build_time=7.0, spacing=28.0, variants=3,
    ),
    # A spiked wooden palisade the colony raises near a screen edge - where the
    # wild animals come in from. Two stages, moderately sturdy, no occupants.
    # `flammable=False` keeps it out of the wildfire spread loop (it is meant to
    # be the thing still standing when everything else has burned). The wide
    # spacing lets one sit at each edge without the director treating the pair
    # as neighbours; behaviour.py picks the edge, this only describes the thing.
    "barricade": StructureSpec(
        "barricade", 2, 150.0, {RES_STONE: 8, RES_WOOD: 6},
        width=40.0, height=34.0, capacity=0, flammable=False,
        build_time=8.0, spacing=120.0, variants=2,
    ),
    # A bridge is not decoration: when it completes it stamps a deck into
    # Terrain's effective-surface overlay across `state["span"]`, and from that
    # moment ground_y() over the gap reports planking instead of the floor 300
    # px down. See Structure._tick_crossing.
    "bridge": StructureSpec(
        "bridge", 3, 120.0, {RES_WOOD: 20, RES_FIBRE: 4},
        width=90.0, height=10.0, capacity=0, flammable=True,
        build_time=10.0, spacing=80.0, variants=2,
    ),
    # The cheap crossing: a ladder against a vertical face. Half a bridge's
    # wood, two stages instead of three and a shorter stage, because it solves
    # a much smaller problem - one face rather than a whole chasm - and the
    # colony should be able to answer a cliff without a full build project.
    # On completion it stamps the *climb* overlay, which is what makes a face
    # steeper than MAX_SLOPE_CLIMB passable at all.
    "ladder": StructureSpec(
        "ladder", 2, 70.0, {RES_WOOD: 8, RES_FIBRE: 2},
        width=40.0, height=60.0, capacity=0, flammable=True,
        build_time=6.0, spacing=60.0, variants=2,
    ),
    "watchtower": StructureSpec(
        "watchtower", 3, 120.0, {RES_WOOD: 22, RES_STONE: 8},
        width=30.0, height=74.0, capacity=1, flammable=True,
        build_time=11.0, spacing=70.0, variants=2,
    ),
    # THE COLONY'S PERIMETER LOOKOUT, AND IT IS ITS OWN KIND RATHER THAN A
    # SITED WATCHTOWER. The dimensions are the watchtower's on purpose - it is
    # the same building - but two things about it are not, and both are
    # per-KIND rather than per-instance:
    #
    #   siting     a watchtower is pulled toward the centre on high ground
    #              (behavior's `+= (ys.max()-ys)/45.0` bias). A lookout is
    #              staked LOOKOUT_STANDOFF px OUTSIDE the settlement on each
    #              side, behind its own side's spikes. One `pick_site` cannot
    #              answer both for one kind.
    #   census     `built("watchtower") < 1` is the colony's existing build
    #              rung and `assign_roles` staffs one lookout per watchtower.
    #              Folding these in as watchtowers would satisfy that rung with
    #              a perimeter tower and stop the colony ever raising its own.
    #
    # `flammable=False`, and that is not tidiness either: StructureRegistry.
    # update draws `rng.random()` off world.rng once per (burning structure x
    # in-range flammable candidate) pair, inside BURN_NEIGHBOUR_DIST (70 px).
    # A flammable tower is a new draw on the numpy stream every time anything
    # near it catches, which re-phases a stream that has nothing to do with
    # towers. The barricade, the wall and the firepit are all False for the
    # same family of reasons.
    "lookout": StructureSpec(
        "lookout", 3, 120.0, {RES_WOOD: 22, RES_STONE: 8},
        width=30.0, height=74.0, capacity=1, flammable=False,
        build_time=11.0, spacing=70.0, variants=2,
    ),
    "totem": StructureSpec(
        "totem", 2, 90.0, {RES_WOOD: 10, RES_STONE: 4, RES_FIBRE: 6},
        width=16.0, height=52.0, capacity=0, flammable=True,
        build_time=8.0, spacing=60.0, variants=3,
    ),
    "stockpile": StructureSpec(
        "stockpile", 1, 80.0, {RES_WOOD: 8},
        width=40.0, height=18.0, capacity=0, flammable=True,
        build_time=6.0, spacing=50.0, variants=2,
    ),
    "grave": StructureSpec(
        "grave", 1, 50.0, {RES_STONE: 2},
        width=12.0, height=16.0, capacity=0, flammable=False,
        build_time=3.0, spacing=15.0, variants=3,
    ),
}

STRUCTURE_KINDS: tuple[str, ...] = tuple(STRUCTURE_SPECS)

_DEFAULT_SPEC = StructureSpec("unknown", 1, 80.0, {RES_WOOD: 4}, 24.0, 24.0)


def structure_spec(kind: str) -> StructureSpec:
    """Spec for `kind`, or a bland fallback so unknown kinds never crash."""
    return STRUCTURE_SPECS.get(kind, _DEFAULT_SPEC)


def plan_ladder(
    terrain: Any, near: float | None = None
) -> tuple[float, float, dict[str, Any]] | None:
    """Site a ladder against a cliff face: ``(x, y, extra_state)`` or ``None``.

    The same shape ``behaviour.pick_site`` returns for every other kind, so
    wiring the ladder into the build order is a two-line change there rather
    than a new code path. ``terrain.find_climb_face`` does the survey; this only
    turns its answer into the state dict a Structure carries -
    ``span``/``rise``, which :meth:`Structure._tick_crossing` later stamps.

    Returns ``None`` on gentle terrain with no face worth laddering, which is
    the common case and is not an error.
    """
    finder = getattr(terrain, "find_climb_face", None)
    if not callable(finder):
        return None
    try:
        face = finder(near)
    except Exception:
        return None
    if not face or len(face) != 4:
        return None
    try:
        x0, x1, y0, y1 = (float(v) for v in face)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x0, x1, y0, y1)) or x1 <= x0:
        return None
    # Anchor the structure at the foot: that is where a builder has to stand,
    # and where a ladder visibly rests. `y` is the ground there.
    foot_x, foot_y = (x0, y0) if y0 > y1 else (x1, y1)
    return (
        foot_x,
        foot_y,
        {"w": (x1 - x0) + 6.0, "span": [x0, x1], "rise": [y0, y1]},
    )


# ------------------------------------------------------------- dict helpers --
def _as_float(d: dict[str, Any], key: str, default: float) -> float:
    try:
        v = d.get(key, default)
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _as_int(d: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(d.get(key, default))
    except (TypeError, ValueError):
        return default


def _as_bool(d: dict[str, Any], key: str, default: bool) -> bool:
    v = d.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return default


def _as_str(d: dict[str, Any], key: str, default: str) -> str:
    v = d.get(key, default)
    return v if isinstance(v, str) else default


def _as_cost(v: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    if isinstance(v, dict):
        for k, q in v.items():
            if not isinstance(k, str):
                continue
            try:
                qi = int(q)
            except (TypeError, ValueError):
                continue
            if qi > 0:
                out[k] = qi
    return out


def _json_safe(v: Any, depth: int = 0) -> Any:
    """Reduce arbitrary state to something json.dump can handle."""
    if depth > 4:
        return None
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else 0.0
    if isinstance(v, (list, tuple)):
        return [_json_safe(x, depth + 1) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x, depth + 1) for k, x in v.items()}
    return None


# ------------------------------------------------------------------ struct --
@dataclass
class Structure:
    """One buildable thing standing on the terrain at `x` with its base at `y`."""

    id: int
    kind: str
    x: float
    y: float
    stage: int = 0
    max_stage: int = 1
    progress: float = 0.0          # 0..1 within the current stage
    hp: float = 100.0
    max_hp: float = 100.0
    built: bool = False
    cost: dict[str, int] = field(default_factory=dict)
    occupants: list[int] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    variant: int = 0
    #: Seconds this structure has stood *finished*. Only advances once `built`,
    #: and pauses while it is rubble. Persisted - it is the settlement's age.
    standing_t: float = 0.0
    #: 0..1 derived from `standing_t` and how well stocked the colony is.
    #: Recomputed (eased) every :meth:`update`; persisted only so a reload does
    #: not visibly re-inflate every hut from nothing.
    growth: float = 0.0

    # ------------------------------------------------------------ factories --
    @classmethod
    def new(
        cls,
        sid: int,
        kind: str,
        x: float,
        y: float,
        *,
        variant: int | None = None,
        built: bool = False,
        rng: random.Random | None = None,
    ) -> "Structure":
        spec = structure_spec(kind)
        r = rng or random
        # Defence in depth against a numpy Generator arriving here: it has
        # .random() but no .randrange(), and this is the only call site that
        # would notice - silently killing every structure the colony builds.
        if variant is not None:
            var = int(variant)
        else:
            try:
                var = r.randrange(max(1, spec.variants))
            except AttributeError:
                var = int(r.random() * max(1, spec.variants))
        s = cls(
            id=int(sid),
            kind=kind,
            x=float(x),
            y=float(y),
            stage=spec.max_stage if built else 0,
            max_stage=spec.max_stage,
            progress=0.0,
            hp=spec.max_hp,
            max_hp=spec.max_hp,
            built=bool(built),
            cost=dict(spec.cost),
            occupants=[],
            state={"delivered": {}},
            variant=int(var),
        )
        if built:
            s._on_complete()
        return s

    # -------------------------------------------------------------- geometry --
    @property
    def spec(self) -> StructureSpec:
        return structure_spec(self.kind)

    def capacity(self) -> int:
        """How many occupants this holds *now*.

        A hut's capacity rides its `growth`: HUT_CAPACITY_MIN sleepers when it
        is a fresh box, HUT_CAPACITY_MAX once it has stood a long time in a
        well-stocked camp. Every other kind keeps its flat spec capacity.
        An explicit ``state["capacity"]`` still overrides everything.

        A shrinking hut never evicts: capacity only gates :meth:`has_room`, so
        someone already asleep is not yanked out of a running action mid-state.
        """
        override = self.state.get("capacity")
        if isinstance(override, (int, float)) and not isinstance(override, bool):
            try:
                return max(0, int(override))
            except (TypeError, ValueError):
                pass
        if self.kind != "hut":
            return int(self.spec.capacity)
        span = HUT_CAPACITY_MAX - HUT_CAPACITY_MIN
        return int(HUT_CAPACITY_MIN + math.floor(_clamp01(self.growth) * span + 0.5))

    def scale(self) -> float:
        """Linear size multiplier from `growth`. 1.0 for anything but a hut."""
        return float(HUT_SCALE_MIN + _clamp01(self.growth) * (HUT_SCALE_MAX - HUT_SCALE_MIN))

    def completion(self) -> float:
        """Overall build completion, 0..1."""
        if self.built:
            return 1.0
        n = max(1, int(self.max_stage))
        v = (float(self.stage) + _clamp01(self.progress)) / n
        return _clamp01(v)

    def width_now(self) -> float:
        w = self.state.get("w")
        if isinstance(w, (int, float)) and w > 0:
            return float(w) * self.scale()
        return float(self.spec.width) * self.scale()

    def height_now(self) -> float:
        """Current visible height - grows with build stage, then with `growth`.

        `scale()` is 1.0 until a hut is finished (growth only accrues on the
        finished form), so the build-stage ramp below is untouched.
        """
        h = float(self.spec.height) * self.scale()
        if self.is_ruined:
            return h * 0.28
        return h * max(0.18, self.completion())

    def top_y(self) -> float:
        """Screen y of the top of the structure (smaller y == higher up)."""
        return self.y - self.height_now()

    def contains_point(self, px: float, py: float, pad: float = 0.0) -> bool:
        hw = self.width_now() * 0.5 + pad
        if abs(px - self.x) > hw:
            return False
        return (self.top_y() - pad) <= py <= (self.y + pad + 4.0)

    def distance_to(self, px: float, py: float | None = None) -> float:
        if py is None:
            return abs(px - self.x)
        return math.hypot(px - self.x, py - self.y)

    # ----------------------------------------------------------- build costs --
    def stage_cost(self, stage: int | None = None) -> dict[str, int]:
        """Resource slice owed for `stage` (defaults to the current stage)."""
        st = self.stage if stage is None else int(stage)
        n = max(1, int(self.max_stage))
        if st < 0 or st >= n:
            return {}
        out: dict[str, int] = {}
        for res, total in self.cost.items():
            try:
                total_i = int(total)
            except (TypeError, ValueError):
                continue
            if total_i <= 0:
                continue
            base = total_i // n
            rem = total_i - base * n
            qty = base + (1 if st >= n - rem else 0)
            if qty > 0:
                out[res] = qty
        return out

    @property
    def delivered(self) -> dict[str, int]:
        d = self.state.get("delivered")
        if not isinstance(d, dict):
            d = {}
            self.state["delivered"] = d
        return d

    def missing_for_stage(self) -> dict[str, int]:
        """What still has to be hauled here before work can proceed."""
        if self.built or self.is_ruined:
            return {}
        have = self.delivered
        out: dict[str, int] = {}
        for res, qty in self.stage_cost().items():
            short = qty - int(have.get(res, 0))
            if short > 0:
                out[res] = short
        return out

    def deliver(self, res: str, qty: int) -> int:
        """Accept up to `qty` of `res`. Returns how much was actually taken."""
        if qty <= 0 or self.built or self.is_ruined:
            return 0
        want = self.missing_for_stage().get(res, 0)
        take = min(int(qty), int(want))
        if take <= 0:
            return 0
        have = self.delivered
        have[res] = int(have.get(res, 0)) + take
        return take

    def total_remaining_cost(self) -> dict[str, int]:
        """Everything still owed across every remaining stage."""
        out: dict[str, int] = {}
        if self.built or self.is_ruined:
            return out
        for res, qty in self.missing_for_stage().items():
            out[res] = out.get(res, 0) + qty
        for st in range(self.stage + 1, max(1, int(self.max_stage))):
            for res, qty in self.stage_cost(st).items():
                out[res] = out.get(res, 0) + qty
        return out

    # ---------------------------------------------------------------- build --
    def advance(self, dt: float, rate: float = 1.0) -> bool:
        """Apply `dt` seconds of work. Returns True on the frame it completes."""
        if self.built or self.is_ruined or dt <= 0.0:
            return False
        if self.missing_for_stage():
            return False
        per = max(0.25, float(self.spec.build_time))
        self.progress += (float(dt) * max(0.0, float(rate))) / per
        completed = False
        guard = 0
        while self.progress >= 1.0 and guard < 16:
            guard += 1
            self.progress -= 1.0
            self.stage += 1
            self.state["delivered"] = {}
            if self.stage >= self.max_stage:
                self.stage = int(self.max_stage)
                self.progress = 0.0
                self.built = True
                self.hp = self.max_hp
                completed = True
                self._on_complete()
                break
        if not self.built:
            self.progress = _clamp01(self.progress)
        return completed

    def _on_complete(self) -> None:
        self.state.pop("delivered", None)
        self.state["delivered"] = {}
        if self.kind in FIRE_KINDS:
            self.state.setdefault("lit", True)
            self.state.setdefault("fuel", 1.0)
        elif self.kind == "totem":
            self.state.setdefault("glow", 1.0)
        elif self.kind == "stockpile":
            self.state.setdefault("shown", {})
        # Not an `elif` on kind: this asks about MATERIAL, and it is reachable.
        # `collapse` deliberately keeps `state["material"]` so a building that
        # burns down and is repaired comes back in the material it had learned -
        # which means a rock tower rebuilt out of its own rubble arrives here
        # still made of rock, and needs its cooldown clock back. Its rocks are
        # not restored: they went down with the platform.
        self._seed_ammo()

    # -------------------------------------------------------------- upgrade --
    # Re-walling a finished hut in stone. The shape below mirrors the build
    # path one-for-one - missing -> deliver -> advance -> completes on one frame
    # - because that is the loop the haulers and _h_build already know how to
    # drive, and mirroring it means a save taken mid-upgrade resumes mid-upgrade
    # for exactly the same reason a save taken mid-build does.
    #
    # What it deliberately does NOT do is reuse `repair()`'s trick of flipping
    # `built` back to False to restart the build in place. That is right for
    # rubble and wrong here: built=False would hide the hut from _free_hut and
    # _pick_hut (both pass built_only=True), break has_room() for anyone asleep
    # inside, and drop it out of the birth gate's hut count in world.py - so
    # starting an upgrade would silently stall births colony-wide. So the
    # upgrade borrows repair()'s *idea* (act on a live Structure without
    # destroying its id or evicting its occupants) while touching none of the
    # fields repair() touches: built, stage, progress, occupants, growth,
    # capacity and hp all carry on exactly as they were. Somebody can be asleep
    # in the hut while it is being re-walled.
    def material(self) -> str:
        """What this building is made of; ``HUT_DEFAULT_MATERIAL`` if unset.

        Answers for kinds with no ladder of their own too, so callers need no
        ``kind == "hut"`` guard. The strip/lower matches ``Atlas.resolve``
        exactly - a hand-edited ``"STONE "`` reads as stone in both places, and
        an unknown material reads as timber in both places rather than as stone
        here and timber on screen. Never raises.

        Per-kind since the rock tower: a lookout's ladder is
        :data:`TOWER_MATERIALS`, so ``"rock"`` is a real answer on a lookout and
        still reads as timber on a hut. Both ladders start at ``"wood"``, which
        is why the fallback is one value rather than a table.
        """
        try:
            mat = str(self.state.get("material") or "").strip().lower()
        except (TypeError, ValueError, AttributeError):
            return HUT_DEFAULT_MATERIAL
        return mat if mat in materials_for(self.kind) else HUT_DEFAULT_MATERIAL

    @property
    def is_upgrading(self) -> bool:
        """True while an upgrade job is pending on this structure."""
        return isinstance(self.state.get("upgrade"), dict)

    def upgrade_spec(self) -> tuple[str, dict[str, int], float, float] | None:
        """This kind's ``(to_material, cost, work, new_max_hp)``, or ``None``.

        ``None`` is the honest answer for every kind that cannot be upgraded,
        and every method below treats it as "no upgrade exists here" rather than
        substituting the hut's numbers - a grave must not quietly acquire a
        10-stone price.
        """
        try:
            return UPGRADE_SPECS.get(self.kind)
        except TypeError:                 # unhashable kind from a corrupt save
            return None

    def can_upgrade(self) -> bool:
        """Is this a finished, still-timber building with nothing in flight?

        The kind test is table-driven (:data:`UPGRADE_SPECS`) rather than the
        ``kind == "hut"`` it used to be, so a lookout answers True for a rock
        tower on exactly the same terms a hut answers True for stone. Every
        ladder starts at wood, so "still on its first material" is one question.
        """
        return (
            self.upgrade_spec() is not None
            and self.built
            and not self.is_ruined
            and not self.is_burning
            and self.material() == HUT_DEFAULT_MATERIAL
            and not self.is_upgrading
        )

    def upgrade_cost(self) -> dict[str, int]:
        """A *copy* of the upgrade price, so a caller cannot edit the module.

        ``{}`` for a kind that cannot be upgraded, which is what every consumer
        of a cost dict already treats as "nothing owed".
        """
        spec = self.upgrade_spec()
        return dict(spec[1]) if spec else {}

    def upgrade_delivered(self) -> dict[str, int]:
        """Stone already hauled here for the upgrade. ``{}`` if not upgrading.

        Mirrors the :attr:`delivered` property: the sub-dict is created in place
        when it is missing, so callers can write to what they get back.
        """
        up = self.state.get("upgrade")
        if not isinstance(up, dict):
            return {}
        d = up.get("delivered")
        if not isinstance(d, dict):
            d = {}
            up["delivered"] = d
        return d

    def upgrade_missing(self) -> dict[str, int]:
        """What still has to be hauled here before the work can proceed.

        Exact mirror of :meth:`missing_for_stage` - positive shortfalls only.
        """
        if not self.is_upgrading:
            return {}
        have = self.upgrade_delivered()
        out: dict[str, int] = {}
        for res, qty in self.upgrade_cost().items():
            short = int(qty) - int(have.get(res, 0))
            if short > 0:
                out[res] = short
        return out

    def upgrade_completion(self) -> float:
        """Applied-work fraction, 0..1. 0.0 when nothing is being upgraded.

        Deliberately *not* blended with how much stone has arrived: both
        consumers - the director's job entry and the builder's handler - want
        the work number, and a blended one would report progress for a hut
        nobody has swung a hammer at yet.
        """
        up = self.state.get("upgrade")
        if not isinstance(up, dict):
            return 0.0
        return _clamp01(_as_float(up, "progress", 0.0))

    def start_upgrade(self, now: float = 0.0) -> bool:
        """Open an upgrade job on this hut. False if it is not eligible.

        Touches nothing but the one state key: not `built`, not `stage`, not
        `progress`, not `occupants`, not `hp`, not `growth`, not `capacity`.
        Everything about the hut carries on as it was until the work lands.

        ``now`` is the world clock, stamped into the job so the director can see
        how long it has gone unclaimed. It lives here rather than on the derived
        ``world.upgrade_job`` because that dict is rebuilt every director pass
        and never persisted - an age kept there would reset to zero on the next
        pass and again on every reload, which is precisely the measurement that
        has to survive. Defaults to 0.0 so an existing caller still works; the
        cost of not passing it is a job that reads as maximally patient.
        """
        if not self.can_upgrade():
            return False
        spec = self.upgrade_spec()
        if spec is None:                  # cannot happen; can_upgrade just said so
            return False
        self.state["upgrade"] = {"to": spec[0], "delivered": {}, "progress": 0.0,
                                 "opened": float(now)}
        return True

    def deliver_upgrade(self, res: str, qty: int) -> int:
        """Accept up to `qty` of `res` toward the upgrade. Returns units taken."""
        if qty <= 0 or not self.is_upgrading or self.is_ruined:
            return 0
        want = self.upgrade_missing().get(res, 0)
        take = min(int(qty), int(want))
        if take <= 0:
            return 0
        have = self.upgrade_delivered()
        have[res] = int(have.get(res, 0)) + take
        return take

    def advance_upgrade(self, dt: float, rate: float = 1.0) -> bool:
        """Apply `dt` seconds of work. True on the frame the upgrade lands.

        Work waits on delivery exactly the way :meth:`advance` does - an
        undelivered upgrade burns no time, so a builder standing at a hut with
        no stone in it makes no progress rather than finishing it for free.
        """
        if not self.is_upgrading or self.is_ruined or dt <= 0.0:
            return False
        if self.upgrade_missing():
            return False
        up = self.state.get("upgrade")
        if not isinstance(up, dict):   # cannot happen; is_upgrading just said so
            return False
        spec = self.upgrade_spec()
        if spec is None:               # nor this; is_upgrading implies a spec
            return False
        to_default, _cost, work, new_max_hp = spec
        per = max(0.25, float(work))
        prog = _as_float(up, "progress", 0.0)
        prog += (float(dt) * max(0.0, float(rate))) / per
        if prog < 1.0:
            up["progress"] = _clamp01(prog)
            return False
        # Landed. This single assignment to state["material"] is the whole of
        # the sim's rendering obligation: renderer.py already passes the state
        # dict into atlas.get, and Atlas.resolve("hut", {"material": "stone"})
        # answers "hut_stone". The kind stays what it was, so the growth ladder,
        # the birth gate, capacity, every hut lookup - and for a lookout, the
        # tower census, `_free_tower` and the role staffing - keep working
        # untouched. That is the entire argument for a material key over a kind.
        to = str(up.get("to") or "").strip().lower()
        self.state["material"] = to if to in materials_for(self.kind) else to_default
        self.state.pop("upgrade", None)
        self.max_hp = float(new_max_hp)
        self.hp = self.max_hp        # the same full heal advance() gives
        # Stone does not burn. If it was alight when the last course went on,
        # it is not any more; from here `ignite` refuses outright.
        self.state.pop("burning", None)
        # A tower that has just become rock is a tower that can hold rocks. The
        # magazine starts EMPTY: the whole point of the feature is that someone
        # has to haul the ammunition up, and a tower that upgraded into eight
        # free rocks would make the stocking leg decorative.
        self._seed_ammo()
        return True

    def cancel_upgrade(self) -> None:
        """Drop an in-flight upgrade job.

        Stone already delivered is *not* refunded - it is sunk into the
        building, exactly as a build stage's delivered resources are when a site
        is abandoned. Refunding would make cancelling free and give the director
        an incentive to churn.
        """
        self.state.pop("upgrade", None)

    # ----------------------------------------------------------- rock ammo --
    # A rock tower's magazine. It lives in `Structure.state` and NOT in a new
    # top-level save key, and that is not a stylistic preference: every new
    # top-level key is 36 fresh hostile load cases in tools/smoke.py, a new
    # junk-sweep surface, and a second source of truth about a building that
    # already serialises its own state dict. `state` round-trips through
    # `_json_safe` for free.
    #
    # Clamped on WRITE (here) as well as on LOAD (`_sanitise_ammo`), so a
    # round-trip is a fixed point and nothing has to be added to smoke.py's
    # CLAMPED_ON_LOAD list.
    @property
    def is_rock_tower(self) -> bool:
        """A tower actually made of rock, so it can hold and drop them.

        Written to short-circuit on the kind rather than as a straight call to
        :func:`_holds_ammo`, because ``_tick_upkeep`` asks this of every
        structure in the colony every tick: a hut, a grave or a firepit costs
        one dict lookup and never touches the string work in
        :meth:`material`.
        """
        spec = self.upgrade_spec()
        if not spec or spec[0] != ROCK_MATERIAL:
            return False
        return self.material() == ROCK_MATERIAL

    def ammo(self) -> int:
        """Rocks stacked on the deck right now. 0..TOWER_AMMO_MAX, never raises.

        Absent and zero mean the same thing, deliberately - see
        :meth:`_set_ammo`, which pops rather than writing a 0.
        """
        raw = self.state.get("ammo")
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return 0
        try:
            f = float(raw)
        except (TypeError, ValueError):
            return 0
        if not math.isfinite(f) or f <= 0.0:
            return 0
        return min(int(f), int(TOWER_AMMO_MAX))

    def ammo_missing(self) -> int:
        """Rocks this tower has room for. 0 for anything that is not one."""
        if not self.is_rock_tower:
            return 0
        return max(0, int(TOWER_AMMO_MAX) - self.ammo())

    def _set_ammo(self, n: int) -> int:
        """Write the magazine, clamped. Returns what it now holds.

        Zero POPS the key rather than storing 0. Absent already reads as empty
        everywhere, and popping means an emptied tower serialises to exactly the
        dict a freshly upgraded one does - one less state shape to reason about,
        and a stable round-trip.
        """
        try:
            v = int(n)
        except (TypeError, ValueError):
            v = 0
        v = max(0, min(int(TOWER_AMMO_MAX), v))
        if v <= 0:
            self.state.pop("ammo", None)
        else:
            self.state["ammo"] = v
        return v

    def _seed_ammo(self) -> None:
        """Give a rock tower the cooldown clock it ticks on. Empty magazine."""
        if not self.is_rock_tower:
            return
        self.state.setdefault("ammo", 0)
        self._set_ammo(self.ammo())          # normalises the 0 straight back out
        if not isinstance(self.state.get("rock_t"), (int, float)) \
                or isinstance(self.state.get("rock_t"), bool):
            self.state["rock_t"] = 0.0

    def deliver_ammo(self, stone: int = 1) -> int:
        """Load hauled stone as rocks. Returns the STONE units actually taken.

        The unit mismatch is the point: a hauler carries stone, a tower drops
        rocks, and one stone splits into :data:`TOWER_AMMO_PER_STONE` of them.
        Returning stone rather than rocks is what lets the stocking handler debit
        the stockpile by exactly what it handed over - the same contract
        :meth:`deliver` and :meth:`deliver_upgrade` already have with the
        builders.

        Refuses everything on a tower that is not rock, not built or in rubble,
        so a caller needs no guard of its own.
        """
        if not self.built or self.is_ruined or not self.is_rock_tower:
            return 0
        try:
            units = int(stone)
        except (TypeError, ValueError):
            return 0
        room = self.ammo_missing()
        if units <= 0 or room <= 0:
            return 0
        per = max(1, int(TOWER_AMMO_PER_STONE))
        # Ceiling division: the last stone is accepted even when it only half
        # fills the rack, because the alternative is a tower that can never be
        # topped to full on an odd shortfall and a hauler that walks back and
        # forth forever.
        take = min(units, (room + per - 1) // per)
        self._set_ammo(self.ammo() + min(room, take * per))
        return take

    def spend_ammo(self, n: int = 1) -> bool:
        """Consume `n` rocks. False - and nothing spent - if there are not `n`."""
        try:
            want = int(n)
        except (TypeError, ValueError):
            return False
        have = self.ammo()
        if want <= 0 or have < want:
            return False
        self._set_ammo(have - want)
        return True

    # --------------------------------------------------------------- damage --
    @property
    def is_ruined(self) -> bool:
        return bool(self.state.get("ruined"))

    @property
    def is_burning(self) -> bool:
        return bool(self.state.get("burning"))

    def label(self) -> str:
        """This building's name in the chronicle's voice. Never raises."""
        try:
            return KIND_LABELS.get(str(self.kind), str(self.kind))
        except Exception:
            return "building"

    def damage(self, amount: float, cause: str = "") -> bool:
        """Apply damage. Returns True if this call collapsed the structure."""
        if amount <= 0.0 or self.is_ruined:
            return False
        self.hp -= float(amount)
        self.state["hurt_t"] = 0.0
        if cause:
            self.state["last_cause"] = str(cause)
        if self.hp <= 0.0:
            self.hp = 0.0
            self.collapse(cause or "damage")
            return True
        return False

    def repair(self, amount: float) -> float:
        """Heal by up to `amount`. Returns hp actually restored."""
        if amount <= 0.0:
            return 0.0
        if self.is_ruined:
            # Rubble can be rebuilt: clearing the ruin flag restarts the build.
            self.state.pop("ruined", None)
            self.state.pop("ruin_t", None)
            self.state["delivered"] = {}
            self.built = False
            self.stage = max(0, int(self.max_stage) - 1)
            self.progress = 0.0
            self.hp = max(1.0, self.max_hp * 0.2)
            return self.hp
        before = self.hp
        self.hp = min(self.max_hp, self.hp + float(amount))
        if self.hp >= self.max_hp * 0.55:
            self.state.pop("burning", None)
        return self.hp - before

    def collapse(self, cause: str = "") -> None:
        """Turn the structure into rubble. Occupants are evicted."""
        self.state["ruined"] = True
        self.state["ruin_t"] = 0.0
        self.state["burning"] = False
        self.state["lit"] = False
        if cause:
            self.state["ruin_cause"] = str(cause)
        self.built = False
        self.hp = 0.0
        self.occupants = []
        # Rubble cannot be mid-upgrade: the job goes, along with any stone
        # already hauled for it. `state["material"]` is deliberately KEPT, so a
        # stone hut that burns down and is repaired comes back in stone. It
        # rebuilds at timber prices, which is slack - and chosen: a fire the
        # player could not prevent should not un-teach the colony masonry.
        self.state.pop("upgrade", None)
        # The rocks DO go, unlike the masonry. They were stacked on a platform
        # that is now on the floor. `material` staying put is what lets
        # `_on_complete` hand the rebuilt tower an empty rack and its clock.
        self.state.pop("ammo", None)
        self.state.pop("rock_t", None)

    # ------------------------------------------------------------------ fire --
    def ignite(self) -> bool:
        """Set alight. Returns True if it actually caught."""
        if self.is_ruined or not self.spec.flammable:
            return False
        # A re-walled hut simply does not catch. This cannot live in
        # `spec.flammable` or in `events._FLAMMABLE`: both are per-KIND (and the
        # second is derived from the first) while material is per-INSTANCE, so
        # one stone hut would make every hut in the colony fireproof.
        # Seeded-stream safe: StructureRegistry.update draws its rng.random()
        # *before* calling ignite(), so refusing here does not change how many
        # numbers come off the stream and a seed still replays identically.
        if self.kind == "hut" and self.material() == "stone":
            return False
        if self.state.get("burning"):
            return False
        self.state["burning"] = True
        self.state["burn_t"] = 0.0
        return True

    def extinguish(self) -> None:
        self.state["burning"] = False

    def light_fire(self, fuel: float = 0.7) -> bool:
        """Light a firepit. Returns True if it is lit afterwards."""
        if self.kind not in FIRE_KINDS or not self.built or self.is_ruined:
            return False
        self.state["lit"] = True
        self.state["fuel"] = max(float(self.state.get("fuel", 0.0)), _clamp01(fuel))
        return True

    def stoke(self, wood: int = 1) -> None:
        """Feed wood to a lit (or unlit but built) firepit."""
        if self.kind not in FIRE_KINDS or self.is_ruined or not self.built:
            return
        fuel = float(self.state.get("fuel", 0.0)) + FIRE_STOKE_PER_WOOD * max(0, int(wood))
        self.state["fuel"] = _clamp01(fuel)
        if self.state["fuel"] > 0.0:
            self.state["lit"] = True

    def feed_garbage(self, qty: int = 1) -> int:
        """Tip swept-up rubbish into a built firepit. Returns units accepted.

        Yields nothing whatsoever - no charcoal, no resource, no stockpile
        credit. The units go in, they burn (see :data:`GARBAGE_PER_SEC`), and
        they are gone. That asymmetry is the point: cleaning up costs the colony
        a villager's time and buys it light, warmth and saved firewood, never
        materials.

        Accepting past :data:`BONFIRE_GARBAGE_CAP` is refused rather than
        clamped silently-in-full, so the caller can tell it still has garbage in
        hand and go and find another fire.
        """
        if self.kind != "firepit" or not self.built or self.is_ruined:
            return 0
        try:
            want = int(qty)
        except (TypeError, ValueError):
            return 0
        if want <= 0:
            return 0
        have = max(0.0, _as_float(self.state, "garbage", 0.0))
        room = float(BONFIRE_GARBAGE_CAP) - have
        take = int(min(float(want), math.floor(room)))
        if take <= 0:
            return 0
        self.state["garbage"] = have + float(take)
        # A pit with garbage in it is a lit pit: somebody has just set the heap
        # going. Deliberately does NOT touch ``fuel`` - the wood the colony had
        # banked is still banked, and is what the fire falls back to afterwards.
        self.state["lit"] = True
        return take

    @property
    def garbage_left(self) -> float:
        """Units of rubbish still to burn. 0.0 for anything but a live firepit."""
        if self.kind != "firepit" or self.is_ruined:
            return 0.0
        return max(0.0, _as_float(self.state, "garbage", 0.0))

    @property
    def bonfire_active(self) -> bool:
        """True while this firepit is running on garbage rather than wood.

        The single flag the renderer, the lightmap and the chronicle all key
        off, so "is it a bonfire" cannot mean three slightly different things.
        """
        return (
            self.kind == "firepit"
            and self.built
            and not self.is_ruined
            and bool(self.state.get("lit"))
            and self.garbage_left > 0.0
        )

    @property
    def fire_active(self) -> bool:
        return (
            self.kind in FIRE_KINDS
            and self.built
            and not self.is_ruined
            and bool(self.state.get("lit"))
            and (
                float(self.state.get("fuel", 0.0)) > 0.0
                or self.garbage_left > 0.0
            )
        )

    def light_source(self) -> dict[str, Any] | None:
        """A light this structure contributes, or None. Fed to Lighting by World."""
        try:
            if self.is_burning and not self.is_ruined:
                t = float(self.state.get("burn_t", 0.0))
                grow = min(1.0, 0.35 + t * 0.25)
                return {
                    "x": float(self.x),
                    "y": float(self.top_y() + self.height_now() * 0.45),
                    "radius": 70.0 + 110.0 * grow,
                    "color": BURN_LIGHT_COLOR,
                    "intensity": 0.55 + 0.4 * grow,
                    "flicker": 0.55,
                }
            # A bonfire before an ordinary fire: this is the payoff of the whole
            # cleanup loop, and it has to reach past the firepit's doorstep or
            # nobody watching a wallpaper at night would ever notice it happened.
            # Ramped over the first second and faded over the last four so the
            # night does not step-change; the flicker is wilder than a banked
            # fire's because a heap of rubbish burns unevenly.
            if self.bonfire_active:
                grow = _clamp01(_as_float(self.state, "bonfire_t", 0.0) / BONFIRE_RAMP_SEC)
                dying = _clamp01(self.garbage_left / BONFIRE_FADE_UNITS)
                k = max(0.25, grow * max(0.35, dying))
                return {
                    "x": float(self.x),
                    "y": float(self.y - 16.0),
                    "radius": float(BONFIRE_LIGHT_RADIUS) * (0.55 + 0.45 * k),
                    "color": BONFIRE_LIGHT_COLOR,
                    "intensity": float(BONFIRE_LIGHT_INTENSITY) * (0.55 + 0.45 * k),
                    "flicker": 0.42,
                }
            if self.fire_active:
                fuel = _clamp01(float(self.state.get("fuel", 0.0)))
                strength = 0.45 + 0.55 * fuel
                return {
                    "x": float(self.x),
                    "y": float(self.y - 6.0),
                    "radius": 96.0 + 96.0 * fuel,
                    "color": FIRE_LIGHT_COLOR,
                    "intensity": 0.55 + 0.4 * strength,
                    "flicker": 0.28,
                }
            if self.kind == "totem" and self.built and not self.is_ruined:
                glow = _clamp01(float(self.state.get("glow", 0.0)))
                if glow > 0.02:
                    return {
                        "x": float(self.x),
                        "y": float(self.top_y() + 6.0),
                        "radius": 46.0 + 30.0 * glow,
                        "color": (150, 210, 255),
                        "intensity": 0.22 * glow,
                        "flicker": 0.08,
                    }
        except Exception:  # pragma: no cover - never break the frame
            log.debug("light_source failed for %s#%s", self.kind, self.id, exc_info=True)
        return None

    # ------------------------------------------------------------ occupancy --
    def has_room(self) -> bool:
        return (
            self.built
            and not self.is_ruined
            and len(self.occupants) < self.capacity()
        )

    def enter(self, agent_id: int) -> bool:
        if agent_id in self.occupants:
            return True
        if not self.has_room():
            return False
        self.occupants.append(int(agent_id))
        return True

    def leave(self, agent_id: int) -> None:
        try:
            while agent_id in self.occupants:
                self.occupants.remove(agent_id)
        except (ValueError, TypeError):
            pass

    # --------------------------------------------------------------- growth --
    def _store_fraction(self, world: Any | None) -> float:
        """0..1 for how well stocked the colony is, against HUT_GROWTH_STORE_REF.

        Cached in `state` so a tick that arrives without a world (the smoke
        tests, a disabled subsystem) does not read as an empty larder and start
        shrinking every hut in the settlement.
        """
        total: float | None = None
        sp = getattr(world, "stockpile", None) if world is not None else None
        if isinstance(sp, dict):
            total = 0.0
            for v in sp.values():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv) and fv > 0.0:
                    total += fv
        if total is None:
            return _clamp01(_as_float(self.state, "store_f", 0.0))
        ref = float(HUT_GROWTH_STORE_REF)
        f = _clamp01(total / ref) if ref > 0.0 else 0.0
        self.state["store_f"] = f
        return f

    def growth_target(self, world: Any | None = None) -> float:
        """Where `growth` is heading right now: part age, part full storehouse."""
        if self.kind != "hut" or not self.built or self.is_ruined:
            return 0.0
        age_ref = float(HUT_GROWTH_AGE_SEC)
        age_f = _clamp01(self.standing_t / age_ref) if age_ref > 0.0 else 1.0
        return _clamp01(
            HUT_AGE_WEIGHT * age_f + HUT_STORE_WEIGHT * self._store_fraction(world)
        )

    def _tick_growth(self, dt: float, world: Any | None) -> None:
        """Age the finished form and ease `growth` toward its target.

        Easing rather than snapping is the whole point: growth falls again when
        the stores are spent, and neither direction is allowed to happen inside
        one frame.
        """
        if self.built and not self.is_ruined:
            self.standing_t = max(0.0, self.standing_t + float(dt))
        target = self.growth_target(world)
        k = GROWTH_LERP_RATE * float(dt)
        if k >= 1.0:
            self.growth = target
        else:
            self.growth = _clamp01(self.growth + (target - self.growth) * k)

    # ------------------------------------------------------------ crossings --
    def crossing_span(self) -> tuple[float, float] | None:
        """``(x0, x1)`` this crossing covers, or ``None`` if it is not one.

        ``behaviour.pick_site`` already records the surveyed span in
        ``state["span"]``; the width fallback is for a crossing placed by hand
        (a test, a debug key) that never went through the director.
        """
        if self.kind not in CROSSING_KINDS:
            return None
        raw = self.state.get("span")
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            try:
                x0, x1 = float(raw[0]), float(raw[1])
                if math.isfinite(x0) and math.isfinite(x1) and abs(x1 - x0) >= 4.0:
                    return (min(x0, x1), max(x0, x1))
            except (TypeError, ValueError):
                pass
        hw = max(8.0, float(self.spec.width) * 0.5)
        return (self.x - hw, self.x + hw)

    def _crossing_stamped(self, terrain: Any, x0: float, x1: float) -> bool:
        """Does the terrain *actually* carry this crossing right now?

        Asked of the terrain rather than of a flag in `state` on purpose. The
        two can disagree - a save whose overlay failed to load, a terrain
        replaced under a surviving registry - and when they do it is the
        terrain that is true. Reconciling against it makes the stamp
        self-healing instead of permanently wrong.
        """
        mid = 0.5 * (x0 + x1)
        probe = "climb_aided" if self.kind == "ladder" else "has_deck"
        fn = getattr(terrain, probe, None)
        if not callable(fn):
            return False
        try:
            return bool(fn(mid))
        except Exception:
            return False

    def _stamp_crossing(self, terrain: Any, x0: float, x1: float) -> bool:
        if self.kind == "ladder":
            rise = self.state.get("rise")
            y0 = y1 = None
            if isinstance(rise, (list, tuple)) and len(rise) >= 2:
                try:
                    y0, y1 = float(rise[0]), float(rise[1])
                except (TypeError, ValueError):
                    y0 = y1 = None
            if y0 is None or y1 is None or not (math.isfinite(y0) and math.isfinite(y1)):
                # No surveyed rise: read the ground at the two anchors. Nothing
                # is stamped there yet, so this is still the bare rock.
                gy = getattr(terrain, "ground_y", None)
                if not callable(gy):
                    return False
                y0, y1 = float(gy(x0)), float(gy(x1))
            fn = getattr(terrain, "stamp_climb", None)
            return bool(fn(x0, x1, y0, y1)) if callable(fn) else False

        fn = getattr(terrain, "stamp_deck", None)
        if not callable(fn):
            return False
        # `y` is the rim behaviour surveyed when it staked the site out, and is
        # also where the renderer draws the deck - so the planking, the sprite
        # and the walkable surface are all the same line by construction.
        return bool(fn(x0, x1, float(self.y)))

    def _tick_crossing(self, world: Any | None) -> None:
        """Keep the terrain overlay in step with whether this crossing stands.

        This is the hook that makes a finished bridge something agents can walk
        on. It is edge-triggered off a comparison with the terrain's real state,
        so it costs two array lookups a tick and fires only on the transitions:
        completed (stamp), collapsed or burned down (clear), rebuilt from rubble
        (stamp again), reloaded from a save with a lost overlay (stamp again).

        Deliberately outside `_tick_upkeep`, which returns early for rubble -
        the collapse of a bridge is exactly when the deck has to come out.
        """
        if self.kind not in CROSSING_KINDS or world is None:
            return
        terrain = getattr(world, "terrain", None)
        if terrain is None:
            return
        span = self.crossing_span()
        if span is None:
            return
        x0, x1 = span
        want = bool(self.built) and not self.is_ruined
        if want == self._crossing_stamped(terrain, x0, x1):
            return
        if want:
            if self._stamp_crossing(terrain, x0, x1):
                self.state["crossing_on"] = True
                self._chronicle(
                    world,
                    f"The {self.label()} is open - the far side can be reached."
                    if self.kind == "bridge"
                    else "The ladder is up; the cliff can be climbed.",
                )
            return
        clear = getattr(terrain, "clear_crossing", None)
        if callable(clear) and clear(x0, x1):
            self.state.pop("crossing_on", None)
            self._chronicle(world, f"The {self.label()} is gone; the way is cut off again.")

    # ----------------------------------------------------------------- tick --
    def update(self, dt: float, world: Any | None = None) -> None:
        """Per-tick upkeep: fire fuel, burning damage, ruin ageing, hut growth.

        Never raises. Growth is ticked separately from - and after - the upkeep
        pass so that an early return (rubble, or a structure that burned down on
        this very tick) still lets the building settle back toward its target.
        """
        try:
            self._tick_upkeep(dt, world)
        except Exception:  # pragma: no cover
            log.debug("structure update failed %s#%s", self.kind, self.id, exc_info=True)
        try:
            if dt > 0.0:
                self._tick_growth(dt, world)
        except Exception:  # pragma: no cover
            log.debug("structure growth failed %s#%s", self.kind, self.id, exc_info=True)
        try:
            self._tick_crossing(world)
        except Exception:  # pragma: no cover
            log.debug("crossing sync failed %s#%s", self.kind, self.id, exc_info=True)

    def _tick_upkeep(self, dt: float, world: Any | None = None) -> None:
        if dt <= 0.0:
            return
        self.state["hurt_t"] = float(self.state.get("hurt_t", 9.9)) + dt
        if self.is_ruined:
            self.state["ruin_t"] = float(self.state.get("ruin_t", 0.0)) + dt
            return
        if self.state.get("burning"):
            self.state["burn_t"] = float(self.state.get("burn_t", 0.0)) + dt
            if self.damage(BURN_DPS * dt, "fire"):
                self._chronicle(world, f"The {self.label()} burned to the ground.")
                return
        if self.kind in FIRE_KINDS and self.state.get("lit"):
            self._tick_fuel(dt, world)
        if self.kind == "totem" and self.built:
            self.state["glow"] = _clamp01(float(self.state.get("glow", 1.0)))
        if self.kind == "barricade" and self.built:
            self._tick_barricade(dt, world)
        if self.built and self.is_rock_tower:
            self._tick_rock_tower(dt, world)

    def _tick_fuel(self, dt: float, world: Any | None) -> None:
        """Spend a lit firepit's fuel. **Garbage first, wood only after.**

        The two are not interchangeable. Garbage is fast (:data:`GARBAGE_PER_SEC`
        is ~6x the wood rate), yields nothing and turns the pit into a bonfire
        while it lasts; wood is the slow, ordinary fuel. Burning garbage first
        means a colony that sweeps up spends no firewood at all for a couple of
        minutes, which is the whole reward for the chore.

        Edge-triggered on both ends, off ``state["bonfire"]``, so the chronicle
        gets exactly one line when it flares and one when it dies back - not one
        a frame.
        """
        garbage = self.garbage_left
        was_bonfire = bool(self.state.get("bonfire"))
        if garbage > 0.0:
            if not was_bonfire:
                self.state["bonfire"] = True
                self.state["bonfire_t"] = 0.0
                self._chronicle(world, "The heap catches: the firepit roars up "
                                       "into a bonfire.")
            self.state["bonfire_t"] = _as_float(self.state, "bonfire_t", 0.0) + dt
            left = garbage - GARBAGE_PER_SEC * dt
            self.state["garbage"] = max(0.0, left)
            if left <= 0.0:
                # Back to exactly a plain firepit's state shape - no zeroed
                # leftovers to persist, reload and reason about later.
                for key in ("garbage", "bonfire", "bonfire_t"):
                    self.state.pop(key, None)
                # Burning out of garbage does NOT put the fire out: it drops
                # back to whatever wood was banked, exactly as if the bonfire
                # had never happened. Only a pit with neither goes dark.
                self._chronicle(world, "The bonfire burns down to an ordinary fire.")
                if float(self.state.get("fuel", 0.0)) <= 0.0:
                    self.state["lit"] = False
            return
        if was_bonfire:                      # e.g. a save loaded mid-bonfire
            for key in ("garbage", "bonfire", "bonfire_t"):
                self.state.pop(key, None)
        # A hermit's fire burns at a quarter of the colony's rate. Not a fudge:
        # the colony pit is a hearth a dozen people walk past with an armful of
        # wood all day, and his is one banked fire one man feeds by hand out of
        # what he personally carried. At the colony rate he stoked it 14 times a
        # colony-hour and did nothing else. See HERMIT_FIRE_BURN.
        rate = HERMIT_FIRE_BURN if self.kind == "hermit_fire" else FIRE_FUEL_BURN
        fuel = float(self.state.get("fuel", 0.0)) - rate * dt
        if fuel <= 0.0:
            self.state["fuel"] = 0.0
            self.state["lit"] = False
        else:
            self.state["fuel"] = fuel

    def _spiked_targets(self, world: Any | None) -> list[Any]:
        """Everything the spikes are allowed to bite: wild animals AND dragons.

        Two registries rather than one, because a dragon is deliberately NOT in
        ``world.animals``: ``AnimalRegistry._walk`` re-welds ``y =
        ground_y(x)`` after every move, which would destroy a flyer's altitude
        every tick, and ``ANIMAL_MAX_ALIVE`` is 4, so a dragon would crowd out
        a whole wolf incursion. They are unified here, at the one predicate
        that genuinely wants both.

        Each side is read behind ``getattr`` and degrades to ``[]`` on its own.
        ``world.animals`` may be absent on a bare test world - the existing
        method already documented that - and ``world.dragons`` is absent on
        every world loaded from a save written before dragons existed, which is
        the far more common case and must not cost the barricade its teeth.

        The list of registries is :data:`SPIKED_TARGET_SOURCES`, append-only, and
        ``"raiders"`` is already on it. That entry is free on a world that has no
        raider registry - the ``getattr`` answers ``None``, ``alive`` is not
        callable, the source contributes nothing - which is what lets the lane
        that owns ``world.raiders`` land later without editing this file.
        """
        out: list[Any] = []
        for attr in SPIKED_TARGET_SOURCES:
            reg = getattr(world, attr, None)
            alive = getattr(reg, "alive", None)
            if not callable(alive):
                continue
            try:
                out.extend(alive())
            except Exception:
                continue
        return out

    def _tick_barricade(self, dt: float, world: Any | None) -> None:
        """A built barricade wounds wild animals lingering within its spikes.

        Passive defence: it damages any *animal* within ``BARRICADE_RANGE`` and
        never a stickman - stickmen are simply not in ``world.animals``. On the
        edge that kills an animal it names the beast in the chronicle.

        The test is x AND height. It used to be x alone, which was correct for
        every entity that had ever existed here - animals.py's own header says
        "Nothing here flies or falls" - and became a real bug the moment
        something could fly. MEASURED BEFORE THE FIX, against this exact method
        with a duck-typed flying stub: a dragon 400 px above the fence took the
        full BARRICADE_DAMAGE of 22.0 hp/s, byte-for-byte identical to a wolf
        standing in the spikes. A colony's barricades would have shredded a
        flyover cruising at alt 360 that the player cannot even reach with a
        spear. ``clearance`` answers 0.0 for anything with no ``alt``, so every
        wolf, bear and boar takes exactly the damage it took before.

        Runs on the per-frame path, so it must never raise: ``world.animals``
        may be absent (a bare test world) or malformed, and an individual
        animal's ``.x``/``.hurt`` might misbehave. Every one of those degrades
        to "no damage this tick" rather than disabling the structures subsystem.
        That is also why ``clearance`` is specified total - a clearance() that
        raises would not merely break barricades, it would disable the entire
        structures subsystem for the session via ``World._guarded``, silently,
        and the colony would simply stop building.
        """
        if world is None or dt <= 0.0:
            return
        dmg = BARRICADE_DAMAGE * float(dt)
        if dmg <= 0.0:
            return
        clearance = _clearance()
        for a in self._spiked_targets(world):
            try:
                if abs(float(a.x) - self.x) > BARRICADE_RANGE:
                    continue
                if clearance(world, a) > BARRICADE_SPIKE_H:
                    continue          # it is flying OVER the spikes
                if a.hurt(dmg):
                    kind = str(getattr(a, "kind", "") or "beast").strip().lower()
                    self._chronicle(world, f"The spikes take a {kind or 'beast'}.")
            except Exception:
                continue

    def _tick_rock_tower(self, dt: float, world: Any | None) -> None:
        """A rock tower drops a rock on whatever walks under it. Never raises.

        Modelled line for line on :meth:`_tick_barricade`, because it is the
        same shape of thing - a built structure that passively hurts what comes
        into range - and the barricade's shape has already had the bugs taken
        out of it. Three differences, all of them deliberate:

        * **It costs ammunition.** The barricade's spikes are free forever; this
          spends one rock per drop out of a magazine capped at TOWER_AMMO_MAX
          that somebody has to climb up and refill. That is what makes it a
          burst rather than a siege engine, and it is the only reason a tower
          that out-damages the spikes is not simply better than them.
        * **It is on a cooldown, not a damage rate.** ROCK_DAMAGE is a discrete
          blow every ROCK_COOLDOWN seconds, not hp/second, so the same wolf
          walking the same path takes the same number of rocks whatever the
          frame rate. `rock_t` is that clock and it is persisted, so a save
          taken mid-reload resumes mid-reload rather than firing instantly.
        * **The reach is tiny.** ROCK_DROP_RANGE is 46 px against the
          barricade's spikes, and ROCK_DROP_CEILING is BELOW the deck, so a
          rock falls and never rises. A tower covers its own footing: the man
          standing on it, and the ladder he came up. Anything that wants to be
          hit at distance has to be thrown at.

        Passive, and that is a design choice with a cost attached: the tower
        drops rocks whether or not a lookout is standing on it. Manning it would
        read better, but it would also mean the drop had to ask the world for an
        occupant every frame - and, worse, it would make the feature's output a
        function of `assign_roles`, which draws off ``world.pyrng``. This branch
        draws NOTHING from any stream, which is what keeps the rock drop
        bit-identical against a world where no tower was ever upgraded.
        """
        if world is None or dt <= 0.0:
            return
        st = self.state
        # The clock runs even with an empty rack, capped at the cooldown, so a
        # tower that has stood idle for an hour is "loaded" rather than holding
        # a 3600-second number - and so the first rock hauled up can be dropped
        # at once rather than after another wait the player cannot see.
        t = min(_as_float(st, "rock_t", 0.0) + float(dt), float(ROCK_COOLDOWN))
        st["rock_t"] = t
        if t < float(ROCK_COOLDOWN) or self.ammo() <= 0:
            return
        clearance = _clearance()
        hurt = _hurt()
        for a in self._spiked_targets(world):
            try:
                if abs(float(a.x) - self.x) > ROCK_DROP_RANGE:
                    continue
                if clearance(world, a) > ROCK_DROP_CEILING:
                    continue      # too high up for something falling to reach it
                if not self.spend_ammo(1):
                    return        # rack emptied under us; nothing dropped
                st["rock_t"] = 0.0
                # Through combat_actions.hurt_animal, not a direct write to
                # `hp`: that is the path that respects Dragon.hurt discarding
                # damage while the dragon is unsated. A rock that set `hp`
                # itself would be the only weapon in the game that kills an
                # unfed dragon, and nothing would fail loudly to say so.
                if hurt(world, a, float(ROCK_DAMAGE), None):
                    kind = str(getattr(a, "kind", "") or "beast").strip().lower()
                    self._chronicle(
                        world, f"A rock from the tower finishes a {kind or 'beast'}."
                    )
                return            # one rock per cooldown, one target
            except Exception:
                continue

    @staticmethod
    def _chronicle(world: Any | None, text: str) -> None:
        """Route a line into the world's chronicle, whatever shape it takes.

        World exposes ``chronicle`` as a bounded ``deque`` and the stamping
        ``log_event`` method; the module smoke tests pass a stub with a callable
        ``chronicle``. Try a callable ``chronicle`` first (the stubs), then the
        canonical ``log_event`` (the real World, so the line gets its day
        stamp), then fall back to appending to a deque/list-like chronicle.
        """
        if world is None or not text:
            return
        try:
            fn = getattr(world, "chronicle", None)
            if callable(fn):
                fn(text)
                return
            log_fn = getattr(world, "log_event", None)
            if callable(log_fn):
                log_fn(text)
                return
            if fn is not None:
                add = getattr(fn, "append", None) or getattr(fn, "add", None)
                if callable(add):
                    add(text)
        except Exception:
            pass

    # ------------------------------------------------------------ serialise --
    def to_dict(self) -> dict[str, Any]:
        # `_json_safe` hands back a fresh dict, so sanitising it here clamps the
        # magazine on WRITE without touching the live structure. Doing it on both
        # sides makes a save/load a fixed point, which is what keeps `ammo` out
        # of smoke.py's CLAMPED_ON_LOAD list (currently exactly one entry) and
        # keeps `test_roundtrip` green without a special case.
        state = _json_safe(self.state) or {}
        _sanitise_ammo(self.kind, state)
        return {
            "id": int(self.id),
            "kind": str(self.kind),
            "x": float(self.x),
            "y": float(self.y),
            "stage": int(self.stage),
            "max_stage": int(self.max_stage),
            "progress": float(self.progress),
            "hp": float(self.hp),
            "max_hp": float(self.max_hp),
            "built": bool(self.built),
            "cost": {str(k): int(v) for k, v in self.cost.items()},
            "occupants": [int(a) for a in self.occupants],
            "state": state,
            "variant": int(self.variant),
            "standing_t": float(self.standing_t),
            "growth": float(self.growth),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Structure":
        """Defensive load: missing keys take defaults, unknown keys ignored."""
        if not isinstance(d, dict):
            d = {}
        kind = _as_str(d, "kind", "hut")
        spec = structure_spec(kind)
        cost = _as_cost(d.get("cost")) or dict(spec.cost)
        max_stage = max(1, _as_int(d, "max_stage", spec.max_stage))
        max_hp = max(1.0, _as_float(d, "max_hp", spec.max_hp))
        occ_raw = d.get("occupants")
        occupants: list[int] = []
        if isinstance(occ_raw, (list, tuple)):
            for a in occ_raw:
                try:
                    occupants.append(int(a))
                except (TypeError, ValueError):
                    continue
        st = d.get("state")
        state: dict[str, Any] = dict(st) if isinstance(st, dict) else {}
        if not isinstance(state.get("delivered"), dict):
            state["delivered"] = {}
        else:
            state["delivered"] = {
                str(k): int(v)
                for k, v in state["delivered"].items()
                if isinstance(v, (int, float))
            }
        _sanitise_bonfire(kind, state)
        _sanitise_upgrade(kind, state)
        # AFTER `_sanitise_upgrade`, which canonicalises `material` - the ammo
        # keys are only legal on a tower that is actually made of rock, so this
        # has to read the cleaned value rather than the raw one.
        _sanitise_ammo(kind, state)
        s = cls(
            id=_as_int(d, "id", 0),
            kind=kind,
            # WORLD_W * 0.5, not RENDER_W * 0.5, and the reasoning is not "rename
            # the constant": this default only fires when a save carries a
            # structure whose x is missing or unparseable, and what it must NOT
            # do is drag `colony_center` - which is the plain mean of these very
            # x values - somewhere the colony has never been. On a 6400 px map
            # RENDER_W * 0.5 is x = 800, i.e. the far-left eighth, so one junk
            # entry hauled the whole settlement's centre (and with it every build
            # site, spawn anchor and camera target derived from it) toward a
            # corner. The world's own midpoint is the only "nowhere in
            # particular" available to a classmethod that cannot see the
            # registry, and it is the same answer every other no-information
            # default in sim/ gives (actions.colony_center, animals, dragons,
            # events and ufo all fall back to WORLD_W * 0.5).
            x=_as_float(d, "x", WORLD_W * 0.5),
            y=_as_float(d, "y", RENDER_H * 0.75),
            stage=max(0, min(max_stage, _as_int(d, "stage", 0))),
            max_stage=max_stage,
            progress=_clamp01(_as_float(d, "progress", 0.0)),
            hp=max(0.0, min(max_hp, _as_float(d, "hp", max_hp))),
            max_hp=max_hp,
            built=_as_bool(d, "built", False),
            cost=cost,
            occupants=occupants,
            state=state,
            variant=max(0, _as_int(d, "variant", 0)),
            standing_t=max(0.0, _as_float(d, "standing_t", 0.0)),
        )
        if s.built:
            s.stage = max_stage
            s.progress = 0.0
        # An upgrade only exists on a standing, finished hut - `_sanitise_upgrade`
        # cannot check that, because `built` and `ruined` are not both readable
        # until the Structure exists. A half-built or collapsed hut carrying an
        # upgrade job is a hand-edited or half-flushed save; drop the job rather
        # than let advance_upgrade re-wall a building that is not there.
        if not s.built or s.is_ruined:
            s.state.pop("upgrade", None)
            # Same reasoning for the magazine, and the same reason
            # `_sanitise_ammo` cannot do it: rubble has no deck to stack rocks
            # on. `collapse` already pops these, so this only ever fires for a
            # hand-edited or half-flushed save.
            s.state.pop("ammo", None)
            s.state.pop("rock_t", None)
        # A save from before hut growth existed has no `growth`. Seeding it from
        # the age term (the store term is not knowable until the world hands us
        # a stockpile) stops every hut in an old colony visibly re-inflating
        # from scratch on load.
        if "growth" in d:
            s.growth = _clamp01(_as_float(d, "growth", 0.0))
        else:
            s.growth = _clamp01(s.growth_target(None))
        # Growth only exists on the finished form. A hand-edited or corrupt save
        # must not load a giant half-built hut (or giant rubble) and then ease
        # back down over the next minute in full view.
        if s.kind != "hut" or not s.built or s.is_ruined:
            s.growth = 0.0
        return s


def _clamp01(v: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f):
        return 0.0
    return 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)


def _sanitise_bonfire(kind: str, state: dict[str, Any]) -> None:
    """Make the bonfire keys in a loaded ``state`` safe, in place.

    The rule is *degrade to a plain firepit, never raise*. A save carrying
    ``garbage: "lots"``, ``garbage: -3``, ``garbage: 1e9``, ``garbage: NaN`` or a
    bonfire flag on a hut has to load as a perfectly ordinary building, because
    the alternative - an exception inside ``StructureRegistry.from_dict`` - loses
    the player's whole settlement to a typo. Every other loader in this file
    works the same way; this is that policy applied to the three new keys.

    Nothing here is reachable from a save this version wrote. It is entirely for
    hand-edited files, files from a future build, and a half-flushed autosave.
    """
    if not isinstance(state, dict):
        return
    if kind != "firepit":
        # Only a firepit can be a bonfire. Anything else carrying these keys is
        # a corrupt or hand-edited save; drop them rather than letting a hut
        # start reporting bonfire_active.
        for key in ("garbage", "bonfire", "bonfire_t"):
            state.pop(key, None)
        return
    raw = state.get("garbage")
    qty = 0.0
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        try:
            f = float(raw)
        except (TypeError, ValueError):
            f = 0.0
        if math.isfinite(f) and f > 0.0:
            qty = min(f, float(BONFIRE_GARBAGE_CAP))
    if qty <= 0.0:
        for key in ("garbage", "bonfire", "bonfire_t"):
            state.pop(key, None)
        return
    state["garbage"] = qty
    state["bonfire"] = bool(state.get("bonfire"))
    t = state.get("bonfire_t")
    ok = isinstance(t, (int, float)) and not isinstance(t, bool)
    if ok:
        try:
            ok = math.isfinite(float(t)) and float(t) >= 0.0
        except (TypeError, ValueError):
            ok = False
    if not ok:
        state.pop("bonfire_t", None)


def _sanitise_upgrade(kind: str, state: dict[str, Any]) -> None:
    """Make the wall-material keys in a loaded ``state`` safe, in place.

    Same rule as :func:`_sanitise_bonfire`: *degrade to a plain timber hut,
    never raise*. A save carrying ``material: "sponge"``, ``material: 7``, a
    ``progress`` of NaN or an upgrade job on a grave has to load as a perfectly
    ordinary building, because the alternative - an exception inside
    ``StructureRegistry.from_dict`` - loses the player's whole settlement to a
    typo.

    Nothing here is reachable from a save this version wrote. It is entirely for
    hand-edited files, files from a future build with a third material, and a
    half-flushed autosave.
    """
    if not isinstance(state, dict):
        return
    mats = materials_for(kind)
    if "material" in state:
        raw = state.get("material")
        mat = raw.strip().lower() if isinstance(raw, str) else ""
        if mat in mats:
            # Store the canonical spelling so the value the renderer strips and
            # the value `material()` compares are the same string on disk too.
            state["material"] = mat
        else:
            # POP rather than coerce to "wood": absent and "wood" mean the same
            # thing everywhere, and popping stops a hand-edited save
            # round-tripping its garbage back out to disk forever.
            state.pop("material", None)
    up = state.get("upgrade")
    spec = UPGRADE_SPECS.get(kind) if isinstance(kind, str) else None
    if not isinstance(up, dict) or spec is None:
        # Only a kind with an entry in UPGRADE_SPECS can be upgraded. Anything
        # else carrying the key is a corrupt save; drop it rather than let a
        # grave report is_upgrading.
        #
        # This test USED to be `kind != "hut"`, and widening it is not optional
        # once a second kind can be upgraded: leave it and every tower upgrade
        # in flight is silently deleted on the next load, with the delivered
        # stone sunk and the job re-opened from scratch on the following
        # director pass. It would look like the feature simply never finished.
        state.pop("upgrade", None)
        return
    to_default, cost = spec[0], spec[1]
    to = up.get("to")
    to = to.strip().lower() if isinstance(to, str) else ""
    clean: dict[str, Any] = {"to": to if to in mats else to_default}
    delivered: dict[str, int] = {}
    raw_delivered = up.get("delivered")
    if isinstance(raw_delivered, dict):
        for res, qty in raw_delivered.items():
            # Only resources the upgrade actually charges for: a save claiming
            # 40 delivered fibre must not make upgrade_missing() look satisfied.
            if not isinstance(res, str) or res not in cost:
                continue
            if isinstance(qty, bool) or not isinstance(qty, (int, float)):
                continue
            try:
                fq = float(qty)
            except (TypeError, ValueError):
                continue
            # isfinite before int(): int(inf) raises OverflowError, which is not
            # in the (TypeError, ValueError) every other loader here catches.
            if not math.isfinite(fq) or fq <= 0.0:
                continue
            delivered[res] = min(int(fq), int(cost[res]))
    clean["delivered"] = delivered
    prog = up.get("progress")
    p = 0.0
    if isinstance(prog, (int, float)) and not isinstance(prog, bool):
        try:
            fp = float(prog)
        except (TypeError, ValueError):
            fp = 0.0
        if math.isfinite(fp):
            p = _clamp01(fp)
    clean["progress"] = p
    # When the job opened, in world seconds. The director ages an unclaimed job
    # off this, so a junk or missing value must not make a job read as infinitely
    # patient (which would let it outrank real work forever) - anything
    # unreadable is dropped, and a dropped stamp reads as "opened just now",
    # the conservative end.
    opened = up.get("opened")
    if isinstance(opened, (int, float)) and not isinstance(opened, bool):
        try:
            fo = float(opened)
        except (TypeError, ValueError):
            fo = float("nan")
        if math.isfinite(fo) and fo >= 0.0:
            clean["opened"] = fo
    state["upgrade"] = clean


def _sanitise_ammo(kind: str, state: dict[str, Any]) -> None:
    """Make a rock tower's magazine keys in a ``state`` safe, in place.

    Same rule as :func:`_sanitise_bonfire` and :func:`_sanitise_upgrade`:
    *degrade to a plain building, never raise, and POP rather than coerce*. A
    save carrying ``ammo: "loads"``, ``ammo: -4``, ``ammo: 1e9``, ``ammo: NaN``,
    a rack on a grave, or a rack on a lookout that is still made of timber has
    to load as a perfectly ordinary building, because the alternative - an
    exception inside ``StructureRegistry.from_dict`` - loses the player's whole
    settlement to a typo.

    Popping rather than coercing to 0 matters more here than it looks. Absent
    and empty already mean the same thing to :meth:`Structure.ammo`, and
    :meth:`Structure._set_ammo` pops on zero too - so an emptied rack, a
    freshly upgraded tower and a hand-edited ``ammo: "eight"`` all serialise to
    the same dict, and a save/load round-trip is a fixed point. That is what
    keeps this key out of ``tools/smoke.py``'s CLAMPED_ON_LOAD list.

    Called from ``Structure.from_dict`` AFTER ``_sanitise_upgrade``, because it
    reads the canonicalised ``material``, and from ``Structure.to_dict``, which
    is the write half of the same clamp.
    """
    if not isinstance(state, dict):
        return
    raw_mat = state.get("material")
    mat = raw_mat.strip().lower() if isinstance(raw_mat, str) else ""
    if not _holds_ammo(kind, mat):
        # Only a tower actually made of rock has a rack. Anything else carrying
        # these keys is a corrupt or hand-edited save, or a tower that was
        # rebuilt in timber; drop them rather than let a hut report ammo.
        for key in ("ammo", "rock_t"):
            state.pop(key, None)
        return
    raw = state.get("ammo")
    n = 0
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        try:
            f = float(raw)
        except (TypeError, ValueError):
            f = 0.0
        if math.isfinite(f) and f > 0.0:
            n = min(int(f), int(TOWER_AMMO_MAX))
    if n <= 0:
        state.pop("ammo", None)
    else:
        state["ammo"] = n
    # The cooldown clock. Bounded at ROCK_COOLDOWN on purpose: `_tick_rock_tower`
    # never lets it grow past that, so a larger value can only have come from
    # outside, and an unbounded one would read as "ready" forever regardless.
    t = state.get("rock_t")
    ok = isinstance(t, (int, float)) and not isinstance(t, bool)
    if ok:
        try:
            ft = float(t)
            ok = math.isfinite(ft) and ft >= 0.0
        except (TypeError, ValueError):
            ok = False
    if ok:
        state["rock_t"] = min(float(t), float(ROCK_COOLDOWN))
    else:
        state.pop("rock_t", None)


# ---------------------------------------------------------------- registry --
class StructureRegistry:
    """Every structure in the world, keyed by id. Cheap linear scans - the
    colony never has more than a few dozen buildings."""

    def __init__(self) -> None:
        self._by_id: dict[int, Structure] = {}
        self._next_id: int = 1

    # ------------------------------------------------------------- mutation --
    def create(
        self,
        kind: str,
        x: float,
        y: float,
        *,
        variant: int | None = None,
        built: bool = False,
        rng: random.Random | None = None,
        state: dict[str, Any] | None = None,
    ) -> Structure:
        """Make a structure, register it and return it."""
        s = Structure.new(self._next_id, kind, x, y, variant=variant, built=built, rng=rng)
        if state:
            s.state.update(state)
        self._next_id += 1
        self._by_id[s.id] = s
        return s

    def add(self, s: Structure) -> Structure:
        """Register an existing Structure, assigning an id if it lacks one."""
        if not isinstance(s, Structure):
            raise TypeError("StructureRegistry.add expects a Structure")
        if not s.id or s.id in self._by_id:
            s.id = self._next_id
        self._by_id[s.id] = s
        self._next_id = max(self._next_id, s.id + 1)
        return s

    def remove(self, target: "Structure | int") -> bool:
        sid = target.id if isinstance(target, Structure) else target
        try:
            sid = int(sid)
        except (TypeError, ValueError):
            return False
        return self._by_id.pop(sid, None) is not None

    def clear(self) -> None:
        self._by_id.clear()
        self._next_id = 1

    # -------------------------------------------------------------- queries --
    def get(self, sid: Any) -> Structure | None:
        try:
            return self._by_id.get(int(sid))
        except (TypeError, ValueError):
            return None

    def __iter__(self) -> Iterator[Structure]:
        return iter(list(self._by_id.values()))

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, sid: Any) -> bool:
        return self.get(sid) is not None

    def all(self) -> list[Structure]:
        return list(self._by_id.values())

    def of_kind(self, kind: str, *, built_only: bool = False) -> list[Structure]:
        out = []
        for s in self._by_id.values():
            if s.kind != kind or s.is_ruined:
                continue
            if built_only and not s.built:
                continue
            out.append(s)
        return out

    def nearest(
        self,
        kind: str | None,
        x: float,
        y: float | None = None,
        *,
        built_only: bool = False,
        max_dist: float | None = None,
        predicate: Any = None,
    ) -> Structure | None:
        """Closest structure of `kind` (None == any kind) to `x` (and `y`)."""
        best: Structure | None = None
        best_d = float("inf")
        for s in self._by_id.values():
            if kind is not None and s.kind != kind:
                continue
            if s.is_ruined:
                continue
            if built_only and not s.built:
                continue
            if predicate is not None:
                try:
                    if not predicate(s):
                        continue
                except Exception:
                    continue
            d = s.distance_to(x, y)
            if d < best_d:
                best_d = d
                best = s
        if max_dist is not None and best_d > max_dist:
            return None
        return best

    def count(self, kind: str | None = None, *, built_only: bool = True) -> int:
        n = 0
        for s in self._by_id.values():
            if kind is not None and s.kind != kind:
                continue
            if s.is_ruined:
                continue
            if built_only and not s.built:
                continue
            n += 1
        return n

    def find_incomplete(self, kind: str | None = None, x: float | None = None) -> Structure | None:
        """The nearest (or first) structure still under construction."""
        best: Structure | None = None
        best_d = float("inf")
        for s in self._by_id.values():
            if s.built or s.is_ruined:
                continue
            if kind is not None and s.kind != kind:
                continue
            d = 0.0 if x is None else abs(s.x - x)
            if d < best_d:
                best_d = d
                best = s
        return best

    def incomplete(self) -> list[Structure]:
        return [s for s in self._by_id.values() if not s.built and not s.is_ruined]

    def damaged(self, *, threshold: float = 0.92) -> list[Structure]:
        out = []
        for s in self._by_id.values():
            if s.is_ruined or not s.built:
                continue
            if s.max_hp > 0 and (s.hp / s.max_hp) < threshold:
                out.append(s)
        return out

    def light_sources(self) -> list[dict[str, Any]]:
        """Every light this registry contributes, for Lighting."""
        out: list[dict[str, Any]] = []
        for s in self._by_id.values():
            ls = s.light_source()
            if ls is not None:
                out.append(ls)
        return out

    def colony_center(self, default: float = WORLD_W * 0.5) -> float:
        """Mean x of the standing, non-grave structures - the settlement.

        The *default* is this registry's answer to "nowhere in particular", and
        it is the centre of the LAND (3200), not half the camera (800). It used
        to be the latter, which was the same number while the world and the
        frame were both 1600 px and was 2400 px adrift the moment they were not:
        a colony that has not finished its first building would have reported
        its home as x = 800 wherever it was actually seeded, and that answer
        anchors the camera, the barricade band, the quarry keep-out and every
        build site the director picks.

        In the main path ``actions.colony_center`` never sees this, because it
        deliberately asks with ``float("nan")`` so that "I have no buildings" is
        distinguishable from a mean that happens to land here, and falls through
        to the agents. That is a guard around this default, not a replacement
        for it: the bare call is still what a duck-typed registry that will not
        take the argument gets, and what any other direct caller gets.
        """
        xs = [s.x for s in self._by_id.values()
              if s.kind not in NON_SETTLEMENT_KINDS and not s.is_ruined]
        if not xs:
            return float(default)
        return float(sum(xs) / len(xs))

    # ----------------------------------------------------------------- tick --
    def update(self, dt: float, world: Any | None = None) -> None:
        """Tick every structure. Fire spreads between close flammable ones."""
        try:
            burning: list[Structure] = []
            for s in list(self._by_id.values()):
                s.update(dt, world)
                if s.is_burning and not s.is_ruined:
                    burning.append(s)
            if burning:
                rng = getattr(world, "rng", None) or random
                chance = BURN_SPREAD_CHANCE * max(0.0, dt)
                for b in burning:
                    for s in self._by_id.values():
                        if s is b or s.is_burning or s.is_ruined or not s.spec.flammable:
                            continue
                        if abs(s.x - b.x) <= BURN_NEIGHBOUR_DIST and rng.random() < chance:
                            s.ignite()
            self.purge_ruins()
        except Exception:  # pragma: no cover
            log.debug("StructureRegistry.update failed", exc_info=True)

    def purge_ruins(self, max_age: float = RUIN_LINGER) -> int:
        """Drop rubble that has sat around too long. Graves are never purged."""
        dead = [
            s.id
            for s in self._by_id.values()
            if s.is_ruined
            and s.kind != "grave"
            and float(s.state.get("ruin_t", 0.0)) > max_age
        ]
        for sid in dead:
            self._by_id.pop(sid, None)
        return len(dead)

    # ------------------------------------------------------------ serialise --
    def to_dict(self) -> dict[str, Any]:
        return {
            "next_id": int(self._next_id),
            "items": [s.to_dict() for s in self._by_id.values()],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StructureRegistry":
        reg = cls()
        if not isinstance(d, dict):
            return reg
        items = d.get("items")
        if isinstance(items, (list, tuple)):
            for raw in items:
                if not isinstance(raw, dict):
                    continue
                try:
                    s = Structure.from_dict(raw)
                except Exception:
                    log.debug("skipping unloadable structure", exc_info=True)
                    continue
                if not s.id or s.id in reg._by_id:
                    s.id = reg._next_id
                reg._by_id[s.id] = s
                reg._next_id = max(reg._next_id, s.id + 1)
        try:
            reg._next_id = max(reg._next_id, int(d.get("next_id", reg._next_id)))
        except (TypeError, ValueError):
            pass
        return reg

    def load_dict(self, d: dict[str, Any]) -> None:
        """In-place variant of from_dict, for callers holding a reference."""
        other = StructureRegistry.from_dict(d)
        self._by_id = other._by_id
        self._next_id = other._next_id


if __name__ == "__main__":  # pragma: no cover - smoke test
    reg = StructureRegistry()
    hut = reg.create("hut", 400.0, 600.0)
    print("hut stage cost", hut.stage_cost(), "missing", hut.missing_for_stage())
    guard = 0
    while not hut.built and guard < 5000:
        guard += 1
        for res, qty in hut.missing_for_stage().items():
            hut.deliver(res, qty)
        hut.advance(0.1)
    print("built:", hut.built, "stage:", hut.stage, "ticks:", guard)
    fire = reg.create("firepit", 420.0, 600.0, built=True)
    print("fire light:", fire.light_source())
    fire.state["fuel"] = 0.0005
    fire.update(1.0)
    print("fire out ->", fire.fire_active)
    hut.ignite()
    for _ in range(200):
        reg.update(0.1)
    print("hut ruined:", hut.is_ruined, "hp", hut.hp)
    blob = reg.to_dict()
    again = StructureRegistry.from_dict(blob)
    print("round trip:", len(again), "count huts", again.count("hut"))
    print("junk load:", len(StructureRegistry.from_dict({"items": [{"kind": "???"}, 5]})))

    # ------------------------------------------------ barricade vs animals --
    class _StubAnimal:
        def __init__(self, x: float, hp: float, kind: str = "wolf") -> None:
            self.x, self.health, self.kind = x, hp, kind

        def hurt(self, amount: float) -> bool:
            self.health -= amount
            return self.health <= 0.0

    class _StubStick:
        def __init__(self, x: float) -> None:
            self.x, self.health = x, 100.0

    class _StubAnimals:
        def __init__(self, items: list[_StubAnimal]) -> None:
            self.items = items

        def alive(self) -> list[_StubAnimal]:
            return [a for a in self.items if a.health > 0.0]

    class _BarWorld:
        def __init__(self) -> None:
            self.lines: list[str] = []
            wolf = _StubAnimal(210.0, 58.0, "wolf")       # in range of barricade
            far = _StubAnimal(900.0, 58.0, "boar")        # out of range
            self.animals = _StubAnimals([wolf, far])
            self.stick = _StubStick(206.0)                # standing in the spikes

        def log_event(self, text: str) -> None:
            self.lines.append(text)

    bw = _BarWorld()
    bar = StructureRegistry().create("barricade", 200.0, 600.0, built=True)
    ticks = 0
    while bw.animals.items[0].health > 0.0 and ticks < 1000:
        ticks += 1
        bar.update(1.0 / 30.0, bw)
    print("barricade: wolf killed in", ticks, "ticks;"
          f" near-wolf hp {bw.animals.items[0].health:.1f}"
          f" far-boar hp {bw.animals.items[1].health:.1f}"
          f" stickman hp {bw.stick.health:.1f}")
    print("  kill line:", [ln for ln in bw.lines if "spikes" in ln][:1])
    # An unfinished barricade must not bite.
    unbuilt_bar = StructureRegistry().create("barricade", 200.0, 600.0)
    before = bw.animals.items[1].health
    unbuilt_bar.update(1.0 / 30.0, bw)
    print("  unbuilt barricade harmless:", bw.animals.items[1].health == before)

    # ------------------------------------------- barricade vs a FLYING thing --
    # The altitude bug, as a regression test. Before the height gate a stub at
    # alt 200 took the full 22.0 hp/s at the barricade's own x - identical to a
    # wolf standing in the spikes - because the predicate tested x alone.
    class _StubFlyer(_StubAnimal):
        def __init__(self, x: float, hp: float, alt: float, kind: str = "wyvern") -> None:
            super().__init__(x, hp, kind)
            self.alt = alt

    class _AltWorld(_BarWorld):
        def __init__(self, flyer: "_StubFlyer") -> None:
            self.lines = []
            self.animals = _StubAnimals([flyer])

    high = _StubFlyer(200.0, 400.0, 200.0)
    bar_hi = StructureRegistry().create("barricade", 200.0, 600.0, built=True)
    aw = _AltWorld(high)
    for _ in range(30):
        bar_hi.update(1.0 / 30.0, aw)
    assert high.health == 400.0, f"spikes hit a flyer 200 px up ({high.health})"
    print("  flyer at alt 200 untouched over 30 ticks:", high.health == 400.0)

    low = _StubFlyer(200.0, 400.0, 0.0)
    bar_lo = StructureRegistry().create("barricade", 200.0, 600.0, built=True)
    aw2 = _AltWorld(low)
    ticks = 0
    while low.health > 0.0 and ticks < 2000:
        ticks += 1
        bar_lo.update(1.0 / 30.0, aw2)
    assert low.health <= 0.0, "a grounded dragon walked out of the spikes alive"
    print(f"  same stub at alt 0 dies in {ticks} ticks:", low.health <= 0.0)

    # ...and the spikes must reach world.dragons, not only world.animals.
    class _DragonOnlyWorld:
        def __init__(self, d) -> None:
            self.lines = []
            self.dragons = _StubAnimals([d])

        def log_event(self, text: str) -> None:
            self.lines.append(text)

    ground_drake = _StubFlyer(200.0, 66.0, 0.0, "quadruped")
    dw = _DragonOnlyWorld(ground_drake)
    bar_dr = StructureRegistry().create("barricade", 200.0, 600.0, built=True)
    for _ in range(120):
        bar_dr.update(1.0 / 30.0, dw)
    assert ground_drake.health <= 0.0, "the spikes cannot see world.dragons"
    print("  grounded dragon reached via world.dragons:",
          [ln for ln in dw.lines if "spikes" in ln][:1])

    # ---------------------------------------------- hut growth, real World --
    from .world import World  # noqa: PLC0415 - smoke test only

    w = World(seed=7)
    h = w.structures.create("hut", 620.0, w.terrain.ground_y(620.0), built=True)
    firepit = w.structures.create("firepit", 660.0, w.terrain.ground_y(660.0), built=True)

    def sweep(seconds: float, dt: float = 1.0 / 30.0) -> None:
        for _ in range(int(seconds / dt)):
            h.update(dt, w)
            firepit.update(dt, w)

    def row(tag: str) -> None:
        print(f"  {tag:<26} store={sum(w.stockpile.values()):>3}"
              f" stand={h.standing_t:7.1f}s growth={h.growth:.3f}"
              f" target={h.growth_target(w):.3f}"
              f" scale={h.scale():.3f} cap={h.capacity()}"
              f" wxh={h.width_now():.1f}x{h.height_now():.1f}")

    print("hut growth against a real World:")
    row("fresh, empty stores")
    w.stockpile[RES_WOOD] = 60          # one big haul: must NOT pop the hut
    h.update(1.0 / 30.0, w)
    row("+60 wood, 1 tick")
    sweep(5.0)
    row("+5s")
    sweep(55.0)
    row("+60s")
    sweep(240.0)
    row("+300s")
    sweep(600.0)
    row("+900s (age maxed)")
    w.stockpile[RES_WOOD] = 0           # stores spent: settlement contracts
    sweep(1.0)
    row("stores spent, 1s")
    sweep(119.0)
    row("stores spent, 120s")
    sweep(480.0)
    row("stores spent, 600s")

    # capacity really gates occupancy
    w.stockpile[RES_WOOD] = 60
    sweep(900.0)
    row("restocked, grown back")
    ids = [i for i in range(1, 9) if h.enter(i)]
    print("  capacity", h.capacity(), "-> admitted", len(ids), "of 8; occupants", h.occupants)
    print("  firepit untouched: growth", firepit.growth, "scale", firepit.scale(),
          "cap", firepit.capacity())

    # unfinished huts must not grow, ruins must shrink back
    unbuilt = w.structures.create("hut", 700.0, w.terrain.ground_y(700.0))
    for _ in range(300):
        unbuilt.update(1.0, w)
    print("  unbuilt hut after 300s: standing_t", unbuilt.standing_t,
          "growth", unbuilt.growth, "scale", unbuilt.scale(), "cap", unbuilt.capacity())
    grown_scale = h.scale()
    h.collapse("test")
    for _ in range(120):
        h.update(1.0, w)
    print(f"  ruined hut: scale {grown_scale:.3f} -> {h.scale():.3f}"
          f" growth {h.growth:.3f} standing_t {h.standing_t:.1f}")

    # --------------------------------------------------- stone hut upgrade --
    print("hut upgrade:")
    up_hut = w.structures.create("hut", 560.0, w.terrain.ground_y(560.0), built=True)
    up_hut.enter(41)
    up_hut.enter(42)
    before_id, before_occ, before_cap = up_hut.id, list(up_hut.occupants), up_hut.capacity()
    print(f"  timber: material={up_hut.material()} can_upgrade={up_hut.can_upgrade()}"
          f" cost={up_hut.upgrade_cost()} missing={up_hut.upgrade_missing()}")
    print("  start:", up_hut.start_upgrade(), "missing now", up_hut.upgrade_missing(),
          f"built={up_hut.built} stage={up_hut.stage} occ={up_hut.occupants}"
          f" cap={up_hut.capacity()}")
    print("  work with no stone on site advances nothing:",
          up_hut.advance_upgrade(5.0), up_hut.upgrade_completion())
    print("  deliver 4 then 20 ->", up_hut.deliver_upgrade(RES_STONE, 4),
          up_hut.deliver_upgrade(RES_STONE, 20), "delivered", up_hut.upgrade_delivered())
    ticks = 0
    while up_hut.is_upgrading and ticks < 2000:
        ticks += 1
        up_hut.advance_upgrade(1.0 / 30.0)
        if ticks == 200:   # mid-upgrade save, reloaded mid-upgrade
            mid = Structure.from_dict(up_hut.to_dict())
            print(f"  mid-upgrade save at {up_hut.upgrade_completion():.3f}"
                  f" -> reloads upgrading={mid.is_upgrading}"
                  f" at {mid.upgrade_completion():.3f}"
                  f" delivered={mid.upgrade_delivered()} occ={mid.occupants}")
    print(f"  finished in {ticks / 30.0:.1f}s: material={up_hut.material()}"
          f" hp={up_hut.hp:.0f}/{up_hut.max_hp:.0f} upgrading={up_hut.is_upgrading}"
          f" id kept={up_hut.id == before_id} occ kept={up_hut.occupants == before_occ}"
          f" cap {before_cap}->{up_hut.capacity()} built={up_hut.built}")
    print("  stone hut refuses to catch:", up_hut.ignite(), "burning", up_hut.is_burning)
    print("  cannot upgrade twice:", up_hut.can_upgrade(), up_hut.start_upgrade())
    stone_back = Structure.from_dict(up_hut.to_dict())
    print("  round trip:", stone_back.material(), "max_hp", stone_back.max_hp)
    # A pre-tier save has no `material` key at all and must load as timber.
    pre = up_hut.to_dict()
    pre["state"] = {k: v for k, v in pre["state"].items() if k != "material"}
    print("  pre-tier save:", Structure.from_dict(pre).material(),
          "state", Structure.from_dict(pre).state.get("material"))
    junk = Structure.from_dict({
        "kind": "hut", "built": True, "state": {
            "material": "  STONE ",
            "upgrade": {"to": "sponge", "delivered": {"fibre": 40, "stone": 1e9},
                        "progress": float("nan")},
        },
    })
    print("  junk save ->", junk.material(), junk.state.get("upgrade"))
    print("  upgrade on rubble dropped:", Structure.from_dict(
        {"kind": "hut", "built": False, "state": {"upgrade": {"to": "stone"}}}).is_upgrading)

    # persistence
    w.stockpile[RES_WOOD] = 60
    h2 = w.structures.create("hut", 520.0, w.terrain.ground_y(520.0), built=True)
    sweep_t = 0.0
    while sweep_t < 600.0:
        h2.update(1.0 / 30.0, w)
        sweep_t += 1.0 / 30.0
    d = h2.to_dict()
    back = Structure.from_dict(d)
    print(f"  round trip: standing_t {h2.standing_t:.1f}->{back.standing_t:.1f}"
          f" growth {h2.growth:.3f}->{back.growth:.3f} scale {back.scale():.3f}")
    legacy = dict(d)
    legacy.pop("growth")
    legacy.pop("state")
    print(f"  legacy save (no growth/store cache): growth {Structure.from_dict(legacy).growth:.3f}"
          f" scale {Structure.from_dict(legacy).scale():.3f}")
    print("  world tick with growth wired:", end=" ")
    for _ in range(60):
        w.tick(1.0 / 30.0)
    print("ok, disabled subsystems:", w._disabled or "none")

    # ------------------------------------------------ the hermit's workbench --
    # Every assertion below has a CONTROL that fails if the separate-kind trick
    # is not actually doing the work - a `stockpile` staked at the same spot,
    # which is the shortcut the spec's note refuses. If the bench were a flagged
    # stockpile, the control and the bench would print the same numbers.
    WB = KIND_HERMIT_WORKBENCH
    print("hermit workbench:")
    wb_spec = structure_spec(WB)
    print(f"  spec: kind={wb_spec.kind} stages={wb_spec.max_stage}"
          f" cost={wb_spec.cost} work={wb_spec.build_time}s hp={wb_spec.max_hp}"
          f" cap={wb_spec.capacity} flammable={wb_spec.flammable}"
          f" {wb_spec.width:.0f}x{wb_spec.height:.0f}")
    assert wb_spec.kind == WB, "workbench fell through to the unknown-kind spec"
    assert WB in STRUCTURE_KINDS, "workbench is not in STRUCTURE_KINDS"

    # ...builds out of one armful, in one stage.
    bench_reg = StructureRegistry()
    bench = bench_reg.create(WB, 3400.0, 600.0)
    print("  stage cost", bench.stage_cost(), "missing", bench.missing_for_stage(),
          "total", bench.total_remaining_cost())
    print("  work with no wood on site advances nothing:", bench.advance(5.0),
          f"progress={bench.progress:.3f}")
    assert not bench.built, "a bench built itself with nothing delivered"
    took = bench.deliver(RES_WOOD, 8)
    wb_ticks = 0
    while not bench.built and wb_ticks < 5000:
        wb_ticks += 1
        bench.advance(1.0 / 30.0)
    print(f"  delivered {took} wood -> built in {wb_ticks / 30.0:.2f}s"
          f" (spec says {wb_spec.build_time})  built={bench.built}"
          f" stage={bench.stage}/{bench.max_stage} label={bench.label()!r}")
    assert bench.built and took == 8, "one armful did not raise the bench"
    print("  holds nothing: capacity", bench.capacity(),
          "state keys", sorted(bench.state), "has_room", bench.has_room())
    assert "shown" not in bench.state, "the bench seeded a stockpile's goods dict"
    assert bench.capacity() == 0 and not bench.has_room(), "the bench admits people"

    # ...and is invisible to every colony census, where a stockpile is not.
    ctrl = bench_reg.create("stockpile", 3400.0, 600.0, built=True)
    print(f"  census: count(hut)={bench_reg.count('hut')}"
          f" count(firepit)={bench_reg.count('firepit')}"
          f" count(stockpile)={bench_reg.count('stockpile')}"
          f" count({WB})={bench_reg.count(WB)}")
    assert bench_reg.count("stockpile") == 1, "control: the stockpile is not counted"
    assert bench_reg.count("hut") == 0 and bench_reg.count("firepit") == 0, \
        "the bench answered somebody else's census"

    # colony_center: the bench must not drag it, the control must.
    town = StructureRegistry()
    for tx in (600.0, 640.0, 680.0):
        town.create("hut", tx, 600.0, built=True)
    base_c = town.colony_center()
    town.create(WB, 3400.0, 600.0, built=True)
    with_bench = town.colony_center()
    town.create("stockpile", 3400.0, 600.0, built=True)
    with_ctrl = town.colony_center()
    print(f"  colony_center: {base_c:.1f} -> +bench {with_bench:.1f}"
          f" -> +stockpile at the same x {with_ctrl:.1f}")
    assert with_bench == base_c, "the bench dragged the settlement into the wild"
    assert with_ctrl > base_c + 100.0, "control: a real building did not move the mean"

    # ...it burns, the way his hut does and his fire does not.
    from .events import _FLAMMABLE  # noqa: PLC0415 - smoke test only
    print(f"  events._FLAMMABLE: bench={WB in _FLAMMABLE}"
          f" hermit_hut={'hermit_hut' in _FLAMMABLE}"
          f" hermit_fire={'hermit_fire' in _FLAMMABLE}")
    assert WB in _FLAMMABLE, "the bench is not in the derived flammable set"
    assert "hermit_fire" not in _FLAMMABLE, "control: his fire became flammable"

    class _CampWorld:
        def __init__(self, seed: int) -> None:
            self.rng = random.Random(seed)
            self.lines: list[str] = []

        def log_event(self, text: str) -> None:
            self.lines.append(text)

    camp = StructureRegistry()
    camp_hut = camp.create("hermit_hut", 3400.0, 600.0, built=True)
    camp_bench = camp.create(WB, 3430.0, 600.0, built=True)
    camp_fire = camp.create("hermit_fire", 3434.0, 600.0, built=True)
    cw = _CampWorld(11)
    camp_hut.ignite()
    for _ in range(40):
        camp.update(1.0, cw)
    print(f"  dragon burns the camp: hut ruined={camp_hut.is_ruined}"
          f" bench caught={camp_bench.is_burning or camp_bench.is_ruined}"
          f" bench ruined={camp_bench.is_ruined}"
          f" fire caught={camp_fire.is_burning or camp_fire.is_ruined}")
    assert camp_bench.is_ruined, "fire beside the bench never reached it"
    assert not (camp_fire.is_burning or camp_fire.is_ruined), \
        "control: the fireproof scrape fire caught alight"
    print("  he rebuilds it:", camp_bench.repair(1.0) > 0.0,
          "built", camp_bench.built, "-> redeliver", camp_bench.deliver(RES_WOOD, 8))

    # ...and round-trips, including mid-build and out of a hand-edited save.
    half = StructureRegistry().create(WB, 3400.0, 600.0)
    half.deliver(RES_WOOD, 5)
    blob_wb = half.to_dict()
    back_wb = Structure.from_dict(blob_wb)
    print(f"  mid-build save: kind={back_wb.kind} built={back_wb.built}"
          f" delivered={back_wb.delivered} missing={back_wb.missing_for_stage()}"
          f" cost={back_wb.cost} max_stage={back_wb.max_stage}")
    assert back_wb.to_dict() == blob_wb, "workbench save/load is not a fixed point"
    done_wb = Structure.from_dict(bench.to_dict())
    print(f"  finished save: built={done_wb.built} hp={done_wb.hp:.0f}"
          f" label={done_wb.label()!r} growth={done_wb.growth}"
          f" scale={done_wb.scale():.3f} cap={done_wb.capacity()}")
    assert done_wb.growth == 0.0, "a bench started growing like a hut"
    nasty = Structure.from_dict({
        "kind": WB, "built": True, "x": "over there", "hp": float("nan"),
        "state": {"shown": {"food": 99}, "material": "sponge", "ammo": 40,
                  "garbage": 12, "upgrade": {"to": "stone"}, "capacity": 9},
    })
    # `shown` and `capacity` survive, and that is PRE-EXISTING behaviour this
    # kind inherits rather than introduces: `Structure.capacity` honours a
    # `state["capacity"]` override for every kind, and nothing outside this
    # module reads `shown` at all. Both are inert on a bench - no code path
    # looks up a `hermit_workbench` to stand in or to draw goods on - and
    # popping them here would change how a hand-edited HUT loads, which is not
    # this lane's to change. The keys that carry real machinery are gone.
    print("  hand-edited save loads:", nasty.kind, "state", sorted(nasty.state),
          "upgrading", nasty.is_upgrading, "ammo", nasty.ammo(),
          "material", nasty.material(), "x", nasty.x, "hp", nasty.hp)
    assert not nasty.is_upgrading and nasty.ammo() == 0, \
        "a bench loaded somebody else's state keys"
    assert "ammo" not in nasty.state and "garbage" not in nasty.state, \
        "a bench kept a magazine or a bonfire"
    print("  registry round trip:", len(StructureRegistry.from_dict(camp.to_dict())),
          "kinds", sorted({s.kind for s in StructureRegistry.from_dict(camp.to_dict())}))

    # ...and a real World, ticked with the bench standing in it, never counts it
    # as colony business. The control is the SAME World ticked without one.
    def _camp_world(with_bench: bool) -> "World":
        ww = World(seed=7)
        bx = min(WORLD_W - 40.0, ww.structures.colony_center() + 900.0)
        if with_bench:
            ww.structures.create(WB, bx, ww.terrain.ground_y(bx), built=True)
        for _ in range(600):
            ww.tick(1.0 / 30.0)
        return ww

    w_ctrl, w_wb = _camp_world(False), _camp_world(True)
    for tag, ww in (("no bench", w_ctrl), ("bench at +900", w_wb)):
        rec = ww.reconcile()
        colony_xs = [s.x for s in ww.structures.all()
                     if s.kind not in NON_SETTLEMENT_KINDS and not s.is_ruined]
        mean = sum(colony_xs) / len(colony_xs) if colony_xs else float("nan")
        print(f"  real World ({tag}): colony_center"
              f" {ww.structures.colony_center():.1f} (mean of the"
              f" {len(colony_xs)} colony buildings = {mean:.1f})"
              f" huts={ww.structures.count('hut')}"
              f" firepits={ww.structures.count('firepit')}"
              f" stockpiles={ww.structures.count('stockpile')}"
              f" benches={ww.structures.count(WB)}"
              f" pop={rec['alive']} residual={rec['residual']} ok={rec['ok']}"
              f" disabled={ww._disabled or 'none'}")
        assert rec["ok"] == 1 and rec["residual"] == 0, "reconcile residual moved"
        assert abs(ww.structures.colony_center() - mean) < 1e-9, \
            "colony_center is not the mean of the colony's own buildings"
    assert w_wb.structures.count(WB) == 1, "the bench vanished over 20 sim-seconds"
    assert (w_ctrl.structures.count("hut"), w_ctrl.structures.count("firepit"),
            w_ctrl.structures.count("stockpile")) == \
           (w_wb.structures.count("hut"), w_wb.structures.count("firepit"),
            w_wb.structures.count("stockpile")), \
        "the bench changed what the colony built for itself"
    print("  colony census identical with and without the bench:", True)

    # WIRING LEDGER, because a kind nothing looks up is a kind that ships inert
    # and this repo has done that twice. Three of the five lists that have to
    # know about this kind live in files this lane may not touch; a bench with
    # `actions.OUTWORK_KINDS` still unpatched drags `settlement_center` 465 px
    # and moves where the colony lays its first firepit by 545 px (measured,
    # seed 7). Printed and not asserted, so this smoke test passes both before
    # and after the integrator applies the handoffs - what it must never do is
    # go quiet about them.
    from . import actions as _acts    # noqa: PLC0415 - smoke test only
    from . import behavior as _beh    # noqa: PLC0415 - smoke test only
    print(f"  wiring ledger for {WB!r}:")
    for _name, _seq in (
        ("structures.NON_SETTLEMENT_KINDS", NON_SETTLEMENT_KINDS),
        ("structures.KIND_LABELS", tuple(KIND_LABELS)),
        ("structures.STRUCTURE_SPECS", STRUCTURE_KINDS),
        ("actions.OUTWORK_KINDS", tuple(_acts.OUTWORK_KINDS)),
        ("actions.HERMIT_KINDS", tuple(_acts.HERMIT_KINDS)),
        ("behavior._HERMIT_SITE_KINDS", tuple(_beh._HERMIT_SITE_KINDS)),
    ):
        print(f"    {'ok  ' if WB in _seq else 'TODO'}  {_name}")
