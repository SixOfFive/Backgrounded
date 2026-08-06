"""Stickmen: identity, needs, physics, and the population that owns them.

Pure python + numpy - **no pygame**. This module must stay importable and
runnable headless; it is how the sim gets tested.

Physics contract
----------------
:meth:`Stickman.apply_physics` is the only place an agent's position changes.
It walks toward ``target_x``, glues the agent to the terrain surface, climbs
steep faces with a per-tick slip chance, falls under gravity once airborne,
kills on a hard landing, and hard-clamps x to the screen because the screen
edges *are* the edge of the world. It never raises: a per-frame path that can
throw is a per-frame path that will eventually kill an app running unattended
overnight.

The terrain is not the only thing to stand on. An action that holds somebody
above the ground - the deck of a watchtower - says so through
:meth:`Stickman.perch` each tick; anything else up there is read as a body in
free fall, and being charged gravity for a job you are standing still doing is
fatal within seconds.

Attrition
---------
:meth:`Stickman.update_needs` is also where a colony can lose someone to
neglect. Hunger and cold both cap at 1.0, and sitting *at* the cap is what
kills: ``starve_t`` and ``exposure_t`` count the sustained seconds at the cap
and reset the instant the need drops. Both are deliberately slow
(``STARVE_LETHAL_SEC``/``EXPOSURE_LETHAL_SEC``, 90 s) so they are a last
resort for a colony that has already run out of food or fire, never a clock
that ticks down on a working one.

Death hand-off
--------------
:meth:`Stickman.die` sets ``alive=False`` and ``dead_t=0.0``, then notifies
every callback registered with :func:`on_death`. The corpse keeps ticking
(gravity still applies, ``dead_t`` accumulates) until it is old enough for
:meth:`Population.collect_graves`, which removes it and returns a plain dict
for props.py to turn into a gravestone.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np

from ..constants import (
    CLIMB_SPEED,
    FALL_LETHAL_SPEED,
    FALL_SURVIVED_DAMAGE,
    MAX_SLOPE_CLIMB,
    MAX_SLOPE_WALK,
    RENDER_H,
    SCENE_ASHFALL,
    SCENE_BLIZZARD,
    SCENE_NIGHT_STORM,
    SCENE_WILDFIRE,
    ARMOUR_DRAGONSCALE,
    ARMOUR_NONE,
    HEALTH_REGEN_PER_SEC,
    MAX_HEALTH,
    MORPH_GIRTH_MAX,
    MORPH_GIRTH_MIN,
    MORPH_NONE,
    MORPH_SCALE_MAX,
    MORPH_SCALE_MIN,
    MORPH_SPAWN_CHANCE,
    MORPH_TABLE,
    RELIC_BFG,
    RELIC_BOOTS,
    RELIC_KINDS,
    RELIC_NONE,
    RELIC_SCALE,
    STICK_PALETTE,
    WALK_SPEED,
    WEAPON_BFG,
    WEAPON_NONE,
    WORLD_W,
)
from .names import (
    ADULT_ROLES,
    ROLE_BUILDER,
    ROLE_CHILD,
    ROLE_ELDER,
    ROLE_GATHERER,
    ROLE_LOOKOUT,
    ROLES,
    death_kind,
    is_role,
    new_name,
    role_label,
)

if TYPE_CHECKING:                                # pragma: no cover
    from .actions import Action
    from .terrain import Terrain

log = logging.getLogger(__name__)

__all__ = [
    "AGENT_HEIGHT", "GRAVE_DELAY", "GRAVITY", "Stickman", "Population",
    "color_for_id", "on_death", "clear_death_listeners",
    "morph_scales", "normalise_morph", "roll_morph",
    "MIN_DRAWN_HEIGHT", "MAX_DRAWN_HEIGHT", "MAX_COUNTER",
    "ROLE_GATHERER", "ROLE_BUILDER", "ROLE_ELDER", "ROLE_CHILD",
    "ROLE_LOOKOUT", "ROLES", "ADULT_ROLES",
]

#: The largest value any integer counter read off a SAVE may take.
#:
#: ``_i`` is a coercion, not a bound: ``int(1e308)`` is a perfectly good python
#: int 309 digits long, and every counter in this file used to take it. A save
#: carrying ``"births": 1e308`` therefore loaded a 309-digit ``births``, which
#: World._rebase_books then solved the population identity with and wrote
#: straight into ``stats["returned"]`` - a 309-digit line in the smoke log, a
#: 309-digit number under the HUD's counter, and unbounded arithmetic anywhere
#: downstream. Nothing FAILED, which is exactly why the 1,222-case junk sweep
#: was green: :meth:`Population.from_dict` never raised and ``reconcile`` still
#: balanced at residual 0, because 1e308 - 1e308 is still 0.
#:
#: 1e12 and not something larger, and the number is not arbitrary: World's own
#: ``stats`` loader already clamps every saved counter to +-1e12, so anything a
#: rebase derives from ``births`` has to fit in the SAME box or the books stop
#: surviving a round trip. Clamp ``births`` to 1e15 and the derived
#: ``stats["died"]`` is saved, re-clamped to 1e12 on the next load, and the
#: identity that held on the way out no longer holds on the way in. Matching the
#: two ceilings is what keeps ``reconcile`` at residual 0 across a save/load.
#:
#: Vastly above anything the simulation can reach - MAX_POP is 20 and a colony
#: books a birth every few minutes - so this is invisible to every honest save
#: and is only ever a bound on a hand-edited or corrupt one.
MAX_COUNTER = 1_000_000_000_000

# ------------------------------------------------------------------ tuning --
#: Nominal head-to-heel height of an adult, in world pixels. render/ scales its
#: skeleton from this so sim and render agree on how big a person is. 26 px in
#: a 1280x800 world is the smallest size at which a two-pixel-wide figure still
#: reads as a *pose* rather than a smudge, once the frame is upscaled to 2560.
AGENT_HEIGHT = 26.0

#: Bounds on the *composed* drawn height (role scale x morph scale). Height is
#: not free to be anything: below ~12 px a figure stops reading as a pose once
#: the frame is upscaled to the desktop, and above ~42 px the body no longer
#: matches the constants the physics is written around (GROUND_SNAP,
#: STEP_DROP_MAX, the STEP_PROBE lookahead), so a giant would start clipping
#: ledges its feet should catch on. Cheaper to clamp here than to make every
#: physics constant a function of body size.
MIN_DRAWN_HEIGHT = 12.0
MAX_DRAWN_HEIGHT = 42.0

GRAVITY = 900.0             # px/s^2
TERMINAL_VY = 900.0         # px/s, matches the lethal threshold scale
AIR_DRAG = 1.6              # per second horizontal damping while airborne
GROUND_SNAP = 1.5           # px tolerance for "standing on the ground"
STEP_DROP_MAX = 5.0         # px of ground drop per step before you're falling
STEP_PROBE = 3.0            # px lookahead used to read the slope ahead

#: The largest drop a walking agent will step off voluntarily, in px.
#:
#: _ground_step used to be asymmetric about vertical steps: a rise steeper than
#: MAX_SLOPE_CLIMB was refused outright (stop(), give the goal up, let behaviour
#: re-score), but the "ground ahead is lower" branch walked up to and over a drop
#: of *any* size at 0.6 * WALK_SPEED. So a villager would decline to climb a
#: waist-high step and then stroll off a 150 px ledge without pausing. Measured:
#: 10 of the 18 residual fall deaths in a chasm sweep launched with vy == 0.0 and
#: |vx| of 4.5-56.3 - the walking signature, not the slip signature (a slip
#: writes exactly vx = +-16.0, vy = 12.0).
#:
#: Derived from the physics rather than picked: a fall from rest reaches
#: FALL_LETHAL_SPEED after FALL_LETHAL_SPEED**2 / (2 * GRAVITY) px. Anything
#: shorter is survivable and agents are still free to hop down it, so this
#: refuses only what would actually kill them. AIR_DRAG means the true lethal
#: height is slightly *greater* than this, which makes the bound conservative in
#: the right direction. Deliberate descents are unaffected: those go through
#: actions' ledge guard into descend_step, which is rate-limited and rolls for
#: slips, so refusing here reroutes rather than strands.
STEP_FALL_MAX = FALL_LETHAL_SPEED * FALL_LETHAL_SPEED / (2.0 * GRAVITY)

#: The steepest face an agent will voluntarily climb DOWN, as |dy/dx|.
#:
#: This is the other half of the same asymmetry, and the bigger killer: 33 of 35
#: fall deaths in a chasm sweep were agents deliberately descending a wall toward
#: a target on the far side and losing grip partway. The slip mechanic itself is
#: fine - a slip on a shallow face self-arrests within 5-45 px, and one map with
#: faces of |dy/dx| 54-64 logged 918 descents, 27 slips and zero deaths. What
#: makes a slip fatal is purely the gradient of what you slip on.
#:
#: _slip pitches you forward at 16 px/s while gravity pulls at GRAVITY, so the
#: fall's own trajectory descends at a fixed rate; a face steeper than that falls
#: away faster than the body does, and you re-contact it further down - or not at
#: all. Measured by driving the real fall integrator down synthetic faces: impact
#: 282 px/s at gradient 12-13, 312 at 13.5-14.5, and 342 at 15 - past
#: FALL_LETHAL_SPEED (340). So 13 sits just under the knee, refusing the descents
#: that kill while leaving every survivable one alone.
#:
#: NOTE for anyone tempted to add the mirror of the above to descend_step - i.e.
#: to also refuse to *climb down* a face a slip would kill you on. It was tried,
#: measured, and reverted, twice, because it makes the colony worse:
#:
#:   refuse faces steeper than 13:1        falls 37->10, survivors 70->59
#:   ...only when the drop below exceeds
#:   STEP_FALL_MAX (a short steep step is
#:   harmless, a long one is not)          falls 37-> 9, survivors 70->56
#:
#: Both cut falling hard and both killed more people overall, by starvation. One
#: 'cliffs' map went from 9 alive with no deaths at all to 2 alive with 6 STARVED:
#: its faces run |dy/dx| 54-64, so every route to food was forbidden at once and
#: the colony was walled into its own corner. Worse, on a chasm map it suppressed
#: the bridge that would have rescued them - the refusal changes where people go,
#: so the site never gets built and the cutoff never resolves.
#:
#: The idea is not wrong, it is just out of order: an agent that refuses a face
#: has to already have somewhere else to go. Descent refusal only becomes safe
#: once the crossing director reliably has a bridge or ladder standing BEFORE the
#: route is withdrawn. Fix that first, then revisit this.
ARRIVE_EPS = 1.5            # px, close enough to the target
SLIP_CHANCE_PER_SEC = 0.42  # base slip probability while climbing, per second
SPEECH_DEFAULT_SEC = 3.0
GRAVE_DELAY = 7.0           # s a corpse lies there before it becomes a grave

#: Going down a face is steadier than hauling yourself up one, so a descent
#: runs the same slip roll at a fraction of the risk. Measured over 10 seeds x
#: 300 s: 0.55 gives 0.9 fall deaths per colony-run, 0.30 gives 0.6, and
#: neither changes how much gets built - so this is purely how often a climb
#: turns into a funeral.
#:
#: 0.30 was still too generous on the chasm and cliff maps, where a descent is
#: 150 px and takes nearly two seconds of slip rolls: a single seed contributed
#: six fall deaths to an otherwise clear, hazard-free day, which is the whole
#: 300 s death budget spent on gravity. 0.12 leaves the clear-weather colony at
#: 0.1 fall deaths per run without changing what gets built.
DESCENT_SLIP_SCALE = 0.12
#: Ceiling on how fast a *deliberate* descent loses height, px/s. Horizontal
#: speed alone does not bound it: on a face that drops 40 px per px of travel,
#: CLIMB_SPEED sideways is 600 px/s downward, which is a fall with extra steps.
DESCENT_DROP_RATE = 90.0

#: How long a perch survives without being renewed, in seconds.
#:
#: A perch (see :meth:`Stickman.perch`) is a *lease*, not a latch: the action
#: standing an agent on a platform has to re-assert it every tick, and it
#: lapses on its own the moment that action stops running. Nothing else can
#: leave someone standing on air - and there are several ways for an action to
#: vanish without its cleanup hook firing (world._tick_agents drops the action
#: outright when a handler raises, die() clears it, a save can be written with
#: the action missing). A latch would strand an agent 68 px up forever; a lease
#: sets them down by itself.
#:
#: Well above one frame (1/30 s, and _MAX_DT caps a stalled one at 0.1 s) so a
#: hitch never drops a working lookout off its tower, and short enough that a
#: genuinely orphaned one is on the ground again within half a second.
PERCH_LEASE = 0.5

# Attrition. A need pinned at its cap for this long is fatal. Sized well above
# the time it takes an agent to notice the need and act on it (behaviour
# re-scores every AI tick and a meal is a ~20 s round trip), so these only
# ever fire for a colony that genuinely cannot feed or warm itself.
NEED_FATAL_LEVEL = 0.999    # "at the cap", allowing for float drift
STARVE_LETHAL_SEC = 90.0
EXPOSURE_LETHAL_SEC = 90.0
CAUSE_STARVED = "hunger"    # -> names.DEATH_KINDS -> "died_hunger"
CAUSE_EXPOSURE = "cold"     # -> names.DEATH_KINDS -> "died_cold"

# Need drift rates, per simulated second. Sized against DAY_LENGTH_SEC (1200s)
# so an agent gets hungry two or three times a day rather than constantly.
HUNGER_PER_SEC = 1.0 / 420.0
FATIGUE_PER_SEC = 1.0 / 600.0
FATIGUE_RECOVER_PER_SEC = 1.0 / 110.0    # sleeping pays fatigue back fast
COLD_PER_SEC = 1.0 / 300.0
WARMTH_RECOVER_PER_SEC = 1.0 / 170.0
MORALE_LERP_PER_SEC = 0.05               # morale eases toward its target
DEFAULT_GROUND_Y = RENDER_H * 0.72   # used only when terrain is unavailable

_MAX_DT = 0.1               # clamp: a stalled frame must not teleport anyone

#: The hard walls a body may not walk past. WORLD_W, not RENDER_W: this is "how
#: much land exists", never "how wide the picture is". It read ``RENDER_W - 1``
#: for as long as those were the same 1600 px, and on a 6400 px map that literal
#: was the single line that made the whole wide world a no-op - every agent in
#: the colony piled up against an invisible wall at x = 1599 and stayed there.
#:
#: MEASURED, 14 seeds x 90 sim-min headless, before -> after:
#:
#:   * agents' x span, as a fraction of the map: 1284 px (20.1%) -> 3156 px
#:     (49.3%). Per-seed, the worst case went from 14.8% to 36.1%;
#:   * seeds ending with somebody standing at exactly x = 1599.0: 13 of 14 -> 0;
#:   * ``stats["built"]``, i.e. whether the colony can reach its own build
#:     sites at all: mean 1.43 -> 7.21.
#:
#: The build number is the one that shows what this literal really cost. It is
#: not that the far side of the map was empty; it is that ``behavior`` sites
#: stakes across the whole WORLD_W and no clamped body could ever walk to one,
#: so a firepit staked past 1599 was a settlement that never started.
#:
#: Every clamp in this module derives from these two, so ``walk_to``, the
#: step/climb/descend probes, ``_clamp_edges`` and ``from_dict`` all move
#: together and cannot drift apart. Two more copies of the same literal live in
#: ``items.py`` and ``throwing.py`` (dropped relics, thrown spears); they are
#: the same bound and ``tools/smoke.py`` asserts all three agree.
_EDGE_MIN = 0.0
_EDGE_MAX = float(WORLD_W - 1)

#: Health at or below this is dead, not "barely alive". See :meth:`Stickman.hurt`
#: for the soft-lock this exists to close - damage sized to land exactly on zero
#: lands a few times 1e-14 ABOVE it in binary, and the difference decided whether
#: a dragon could ever be killed. Nine orders of magnitude below MAX_HEALTH, so
#: no caller can be meaning to leave someone standing inside it.
_HEALTH_EPSILON = 1.0e-9

_RNG = random.Random()

# ------------------------------------------------------------ death hooks --
DeathListener = Callable[["Stickman", str], None]
_DEATH_LISTENERS: list[DeathListener] = []


def on_death(fn: DeathListener) -> DeathListener:
    """Register *fn* to be called as ``fn(stickman, cause)`` when one dies.

    Usable as a decorator. Listeners are called fail-soft: an exception in one
    is logged and the rest still run. This is the hand-off point for the
    chronicle, the mourning behaviour and the gravestone prop.
    """
    if callable(fn) and fn not in _DEATH_LISTENERS:
        _DEATH_LISTENERS.append(fn)
    return fn


def clear_death_listeners() -> None:
    """Drop every registered death listener (used by tests and world reload)."""
    _DEATH_LISTENERS.clear()


def _fire_death(s: "Stickman", cause: str) -> None:
    for fn in tuple(_DEATH_LISTENERS):
        try:
            fn(s, cause)
        except Exception:
            log.exception("death listener %r failed", fn)


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


def _count(value: Any, default: int = 0, lo: int = 0,
           hi: int = MAX_COUNTER) -> int:
    """:func:`_i`, bounded. The coercion every counter off a SAVE goes through.

    Bounds the DEFAULT too, deliberately: several callers pass a value derived
    from the file (``len(pop.agents)``, ``d["deaths"] - corpses``), and a bound
    that only applied to the happy path would leave the same hole one argument
    along. Never raises - it is on the from_dict path. See :data:`MAX_COUNTER`
    for why the ceiling is 1e12 and not merely "large".
    """
    v = _i(value, _i(default, 0))
    return lo if v < lo else (hi if v > hi else v)


def _b(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return default


def _s(value: Any, default: str | None = None) -> str | None:
    if isinstance(value, str):
        return value
    return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _unit(value: Any, default: float = 0.0) -> float:
    """A 0..1 need value, coerced and clamped."""
    return _clamp(_f(value, default), 0.0, 1.0)


def _color(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    try:
        r, g, b = (int(c) for c in tuple(value)[:3])
    except Exception:
        return default
    return (int(_clamp(r, 0, 255)), int(_clamp(g, 0, 255)), int(_clamp(b, 0, 255)))


def color_for_id(agent_id: int) -> tuple[int, int, int]:
    """Identity colour for an agent id - the palette cycles by id."""
    if not STICK_PALETTE:
        return (206, 214, 226)
    return STICK_PALETTE[_i(agent_id) % len(STICK_PALETTE)]


# ------------------------------------------------------------ body morphs --
# An agent stores only the *name* of its morph; these three functions are the
# only readers of the table. Keeping the lookup in one place is what lets a
# corrupt or future name degrade to an ordinary body everywhere at once instead
# of raising somewhere deep in the render path.
def morph_scales(name: Any) -> tuple[float, float]:
    """``(height scale, girth)`` for a morph name. Unknown names are ordinary.

    Always returns finite, clamped numbers, so callers can multiply by them
    without checking anything.
    """
    if not isinstance(name, str) or not name:
        return (1.0, 1.0)
    entry = MORPH_TABLE.get(name)
    if not entry:
        return (1.0, 1.0)
    try:
        scale, girth = float(entry[0]), float(entry[1])
    except Exception:
        return (1.0, 1.0)
    if not (math.isfinite(scale) and math.isfinite(girth)):
        return (1.0, 1.0)
    return (_clamp(scale, MORPH_SCALE_MIN, MORPH_SCALE_MAX),
            _clamp(girth, MORPH_GIRTH_MIN, MORPH_GIRTH_MAX))


def normalise_morph(name: Any) -> str:
    """A morph name safe to store. Anything unrecognised becomes the normal body.

    This is the load-bearing half of "a corrupt save must not raise": a name
    from a newer build, a hand-edited save or a junk type is dropped at the
    boundary rather than carried around as a value nothing can interpret.
    """
    try:
        return name if (isinstance(name, str) and name in MORPH_TABLE) else MORPH_NONE
    except Exception:
        return MORPH_NONE


def roll_morph(rng: Any) -> str:
    """Roll a body for one newcomer. Overwhelmingly returns ``MORPH_NONE``.

    *rng* must be a seeded stream (``world.pyrng``). Unlike the cosmetic rolls
    around it there is deliberately **no** fallback to the module RNG: a mutant
    that appears in only some runs of a seed would quietly destroy seed
    reproducibility, and a run with nothing to roll from is better off with an
    ordinary villager than with an unrepeatable one.
    """
    if rng is None:
        return MORPH_NONE
    try:
        r = float(rng.random())
    except Exception:
        return MORPH_NONE
    if not (0.0 <= r < 1.0):
        return MORPH_NONE
    acc = 0.0
    for name, chance in MORPH_SPAWN_CHANCE.items():
        acc += max(0.0, _f(chance, 0.0))
        if r < acc:
            return normalise_morph(name)
    return MORPH_NONE


# -------------------------------------------------------- terrain adapters --
# Terrain belongs to another module and may be absent, partial or mid-rewrite.
# Every read goes through these so a broken Terrain degrades to flat ground
# instead of taking the app down.
def _ground_y(terrain: "Terrain | None", x: float) -> float:
    if terrain is None:
        return DEFAULT_GROUND_Y
    fn = getattr(terrain, "ground_y", None)
    if fn is None:
        return DEFAULT_GROUND_Y
    try:
        return _f(fn(x), DEFAULT_GROUND_Y)
    except Exception:
        return DEFAULT_GROUND_Y


def _terrain_slope(terrain: "Terrain | None", x: float) -> float:
    if terrain is None:
        return 0.0
    fn = getattr(terrain, "slope", None)
    if fn is None:
        return 0.0
    try:
        return _f(fn(x), 0.0)
    except Exception:
        return 0.0


# ================================================================ Stickman ==
@dataclass
class Stickman:
    """One inhabitant. Everything that must survive a restart is a field here."""

    # --- identity ---------------------------------------------------------
    id: int = 0
    name: str = "?"
    color: tuple[int, int, int] = (206, 214, 226)   # identity colour, persisted

    #: Rare body morph: "" for the ordinary build, otherwise a key of
    #: constants.MORPH_TABLE ("giant", "tiny", "stout", "lanky"). Only the name
    #: lives on the agent - see morph_scales() for the numbers it means.
    morph: str = MORPH_NONE

    # --- kinematics -------------------------------------------------------
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    facing: int = 1                       # -1 | +1

    # --- needs (0..1) -----------------------------------------------------
    hunger: float = 0.0                   # 1 = starving
    fatigue: float = 0.0                  # 1 = exhausted
    warmth: float = 0.0                   # 1 = freezing
    morale: float = 0.6                   # 1 = content

    # --- inventory / role -------------------------------------------------
    carrying: str | None = None           # resource id, see constants.RES_*
    carry_qty: int = 0
    role: str = ROLE_GATHERER
    generation: int = 0

    # --- behaviour --------------------------------------------------------
    action: "Action | None" = None
    holds_candle: bool = False
    holds_torch: bool = True     # everyone carries one; the elder has a candle

    # --- combat & gear ---------------------------------------------------
    # health exists because animals are the first threat you can *survive*.
    # Everything else that kills is instantaneous, so until now an agent was
    # simply alive or not.
    health: float = MAX_HEALTH
    weapon: str = WEAPON_NONE          # "" | "spear"
    armour: float = ARMOUR_NONE        # 0..1 fraction of damage absorbed
    attack_cd: float = 0.0             # seconds until the next swing lands
    target_animal: int | None = None   # animal id this agent is fighting

    #: The one relic slot: "" or a member of ``constants.RELIC_KINDS``.
    #:
    #: A REAL FIELD, and that is the whole point of it being here. ``items.equip``
    #: writes ``agent.relic`` and for one release this class had no such field, so
    #: python happily created a bare instance attribute that ``to_dict`` never saw:
    #: a colony's entire loot evaporated on the next save/load, silently, with no
    #: warning and no way to tell from inside the game. ``beamed`` above is the
    #: same bug, already fixed, with the same note. If a module outside sim/items
    #: needs to remember something about an agent, it goes in this dataclass.
    #:
    #: sim/items.py owns what the value MEANS; this module only stores it and
    #: reads one kind of it (:data:`RELIC_BOOTS`, in :meth:`_fall_step`). The read
    #: is a bare ``==`` against the constant rather than a call to ``items.worn``
    #: on purpose - entities.py sits underneath almost everything in sim/ and is
    #: not going to start importing its dependents to test a string.
    #:
    #: The dragonscale and the wyrm-gun ALSO mirror into ``armour`` and ``weapon``
    #: (see ``items.equip``); ``from_dict`` re-asserts those mirrors on load so a
    #: save cannot come back holding the coat and wearing none of it.
    relic: str = RELIC_NONE

    #: How many times ironshod boots have turned a lethal landing into a bruise.
    #: SAVES, not interventions: a booted faller who lands on 5 hp and dies of
    #: the bruise anyway is not counted here (see :meth:`_boots_catch`).
    #:
    #: Persisted, and it exists to be *measurable*. The relic feature shipped once
    #: already in a state where every effect was inert and nothing in the stats
    #: could show it, which is how it survived review; a counter that rides on the
    #: agent needs no world reference (``_fall_step`` has none) and lets a harness
    #: sum it over the population to prove the boots fired at all.
    fall_saves: int = 0

    # --- abduction -------------------------------------------------------
    # Taken is not the same as dead: no corpse, no grave, and they may come
    # back. lift_t drives the beam animation in render/.
    taken: bool = False
    lift_t: float = 0.0
    # Written by ufo.py while the beam holds someone, and read by behaviour to
    # stop them walking. It was an undeclared attribute, so it silently
    # vanished across a save/load and any gate on it saw nothing.
    beamed: bool = False

    #: Structure id this agent is currently inside, or None. Sleepers go *in*
    #: the hut rather than lying across its doorstep, so render/ skips them and
    #: the building shows an occupied light instead. Persisted, because a save
    #: taken at night is mostly a save of people asleep.
    inside: int | None = None
    alive: bool = True
    anim_t: float = 0.0                   # animation phase accumulator

    # --- movement intent --------------------------------------------------
    target_x: float | None = None         # walk here; None = stand still
    path_goal: str | None = None          # opaque label for *why* ("stockpile",
                                          # "tree:12"); actions own its meaning

    #: y of the platform this agent is standing on, or None when the terrain
    #: is its footing. A watchtower deck is not ground - ``ground_y`` knows
    #: nothing about it - so without this the fall integrator reads anybody up
    #: there as mid-air and charges them gravity for every second of their
    #: watch. Set through :meth:`perch`, never written raw. Persisted: a save
    #: taken with a lookout on his tower must not reload him into a fall.
    perch_y: float | None = None
    _perch_t: float = 0.0                 # s since the perch was last renewed

    # --- timers -----------------------------------------------------------
    climb_t: float = 0.0                  # s spent on the current climb
    fall_t: float = 0.0                   # s spent airborne
    dead_t: float = 0.0                   # s since death, drives the grave
    speech: str | None = None             # symbol key for the speech bubble
    speech_t: float = 0.0                 # s of bubble left
    starve_t: float = 0.0                 # s hunger has been pinned at the cap
    exposure_t: float = 0.0               # s warmth has been pinned at the cap
    _blocked_t: float = 0.0               # s stuck against terrain, see actions

    # --- lineage ----------------------------------------------------------
    birth_time: float = 0.0               # world_time at birth
    parent_ids: list[int] = field(default_factory=list)

    # --- derived / cosmetic (persisted so a reload looks identical) --------
    on_ground: bool = True
    climbing: bool = False
    gait: float = 1.0                     # per-agent walk speed/style variation
    death_cause: str = ""
    pose_override: str | None = None      # forces a render pose without an Action
    action_raw: dict[str, Any] | None = None   # unrehydrated action from a save

    # ------------------------------------------------------------ helpers --
    @property
    def pos(self) -> tuple[float, float]:
        """Feet position in world space."""
        return (self.x, self.y)

    @property
    def head_y(self) -> float:
        """Approximate y of the top of the head.

        Derived from :meth:`height` rather than from AGENT_HEIGHT directly, so
        it follows the role *and* the morph - a giant's head is not where an
        ordinary one's is, and anything hanging a label or a bubble off this
        would otherwise put it through the skull.
        """
        return self.y - self.height()

    def height(self) -> float:
        """Drawn height - children are smaller, elders stoop, morphs rescale.

        The morph *multiplies* the role height instead of replacing it, so a
        giant child is still recognisably a child. The clamp is what makes that
        composition safe: a tiny child would otherwise solve to 11 px, which is
        a smudge once the frame is upscaled, and a giant elder must stay inside
        the band the physics constants assume (see MIN/MAX_DRAWN_HEIGHT).
        """
        h = AGENT_HEIGHT
        if self.role == ROLE_CHILD:
            h *= 0.68
        elif self.role == ROLE_ELDER:
            h *= 0.94
        h *= morph_scales(self.morph)[0]
        return _clamp(h, MIN_DRAWN_HEIGHT, MAX_DRAWN_HEIGHT)

    def girth(self) -> float:
        """Width factor for this body; 1.0 is the ordinary build.

        Read by render/ to widen the stance and thicken the lines. It is not a
        height multiplier: girth is exactly the axis that separates "fat" from
        "big", which a single scale cannot express on a stick figure.
        """
        return morph_scales(self.morph)[1]

    def is_mutant(self) -> bool:
        """True for the rare odd-bodied villagers (a known morph, not "")."""
        try:
            return isinstance(self.morph, str) and self.morph in MORPH_TABLE
        except Exception:
            return False

    def hand_anchor(self) -> tuple[float, float]:
        """Roughly where the front hand is - candle light attaches here.

        An approximation on purpose: lighting.py lives in sim/ and cannot ask
        render/ for the exact skeletal hand joint. The horizontal reach carries
        the girth so a broad villager's light does not sit inside their chest.
        """
        h = self.height()
        return (self.x + self.facing * 0.20 * h * self.girth(), self.y - 0.46 * h)

    def is_child(self) -> bool:
        return self.role == ROLE_CHILD

    def distance_to(self, x: float) -> float:
        return abs(self.x - _f(x, self.x))

    # ------------------------------------------------------------- intent --
    def walk_to(self, x: float, goal: str | None = None) -> None:
        """Set the walk target (clamped into the world) and its goal label."""
        self.target_x = _clamp(_f(x, self.x), _EDGE_MIN, _EDGE_MAX)
        self.path_goal = goal

    def stop(self) -> None:
        """Clear the walk target and kill horizontal velocity."""
        self.target_x = None
        self.path_goal = None
        self.vx = 0.0

    def at_target(self, eps: float = ARRIVE_EPS) -> bool:
        return self.target_x is None or abs(self.target_x - self.x) <= eps

    def say(self, symbol: str, duration: float = SPEECH_DEFAULT_SEC) -> None:
        """Show a symbol speech bubble for *duration* seconds."""
        if isinstance(symbol, str) and symbol:
            self.speech = symbol
            self.speech_t = max(0.1, _f(duration, SPEECH_DEFAULT_SEC))

    # -------------------------------------------------------------- needs --
    def update_needs(self, dt: float, world: Any = None) -> None:
        """Advance hunger, fatigue, cold and morale.

        This is the only writer of the need fields. Without it the needs stay
        at their spawn values forever and every need-driven behaviour (eating,
        sleeping, warming at the fire) is unreachable, so it must never raise:
        a bad world reference degrades to the baseline drift rather than
        stopping the colony.
        """
        if dt <= 0.0:
            return

        hunger_rate = HUNGER_PER_SEC
        fatigue_rate = FATIGUE_PER_SEC
        chill_rate = -WARMTH_RECOVER_PER_SEC        # default: slowly warming

        try:
            night = bool(getattr(world, "is_night", False))
            scene = getattr(getattr(world, "events", None), "scene", "")

            if night:
                fatigue_rate *= 1.9
                chill_rate = COLD_PER_SEC * 0.6
            if scene == SCENE_BLIZZARD:
                chill_rate = COLD_PER_SEC * 2.4
            elif scene == SCENE_NIGHT_STORM:
                chill_rate = COLD_PER_SEC * (1.3 if night else 0.7)
            elif scene in (SCENE_WILDFIRE, SCENE_ASHFALL):
                chill_rate = -WARMTH_RECOVER_PER_SEC * 1.5

            # Working is hungrier and more tiring than standing around.
            act = getattr(self, "action", None)
            pose = getattr(act, "pose", "") if act is not None else ""
            if pose in ("chop", "build", "climb", "carry", "panic"):
                hunger_rate *= 1.6
                fatigue_rate *= 1.5
            elif pose == "sleep":
                fatigue_rate = -FATIGUE_RECOVER_PER_SEC
                hunger_rate *= 0.5
        except Exception:
            pass

        self.hunger = _clamp(self.hunger + hunger_rate * dt, 0.0, 1.0)
        self.fatigue = _clamp(self.fatigue + fatigue_rate * dt, 0.0, 1.0)
        self.warmth = _clamp(self.warmth + chill_rate * dt, 0.0, 1.0)

        # Morale drifts toward contentment but is dragged down by whatever is
        # currently worst, so a comfortable colony recovers and a miserable
        # one visibly does not.
        pressure = max(self.hunger, self.fatigue, self.warmth)
        target = 1.0 - pressure
        self.morale = _clamp(
            self.morale + (target - self.morale) * MORALE_LERP_PER_SEC * dt,
            0.0, 1.0)

        self._tick_attrition(dt)

    def _tick_attrition(self, dt: float) -> None:
        """Age the starvation/exposure clocks and kill when one runs out.

        Nothing used to *consume* a maxed-out need, so hunger pinned at 1.0
        for four minutes killed nobody and the needs were decoration. The
        clocks are sustained-time, not accumulated-damage: they zero the moment
        the need comes off its cap, so one meal or one turn at the fire buys a
        full reprieve and a fed colony never approaches the threshold.

        Runs inside update_needs (a per-frame path) so it never raises: an
        agent who cannot be killed is a far smaller bug than a sim that stops.
        """
        try:
            if self.hunger >= NEED_FATAL_LEVEL:
                self.starve_t += dt
            else:
                self.starve_t = 0.0
            if self.warmth >= NEED_FATAL_LEVEL:
                self.exposure_t += dt
            else:
                self.exposure_t = 0.0

            if self.starve_t >= STARVE_LETHAL_SEC:
                self.die(CAUSE_STARVED)
            elif self.exposure_t >= EXPOSURE_LETHAL_SEC:
                self.die(CAUSE_EXPOSURE)
        except Exception:
            log.exception("attrition failed for agent %s (%s)", self.id, self.name)

    def needs(self) -> dict[str, float]:
        """Current need *pressures*, 0 = fine, 1 = urgent.

        This is the input surface behaviour.py scores actions against; morale
        is inverted here so every entry means the same thing (bigger = worse).
        """
        return {
            "hunger": _clamp(self.hunger, 0.0, 1.0),
            "fatigue": _clamp(self.fatigue, 0.0, 1.0),
            "cold": _clamp(self.warmth, 0.0, 1.0),
            "despair": _clamp(1.0 - self.morale, 0.0, 1.0),
        }

    def worst_need(self) -> tuple[str, float]:
        """The single most urgent need and its pressure."""
        needs = self.needs()
        key = max(needs, key=lambda k: needs[k])
        return key, needs[key]

    def speed_scale(self) -> float:
        """Multiplier on walk speed from condition, load and role."""
        scale = 1.0 - 0.30 * self.fatigue - 0.18 * self.hunger - 0.12 * self.warmth
        if self.carrying:
            scale -= 0.12
        if self.role == ROLE_CHILD:
            scale *= 0.88
        elif self.role == ROLE_ELDER:
            scale *= 0.78
        return _clamp(scale * _clamp(self.gait, 0.5, 1.6), 0.35, 1.6)

    def summary(self) -> str:
        """One-line human description - chronicle, tooltips and debug."""
        bits = [f"{self.name} ({role_label(self.role)}, gen {self.generation})"]
        if self.is_mutant():
            bits.append(self.morph)
        if not self.alive:
            cause = f" ({self.death_cause})" if self.death_cause else ""
            bits.append(f"dead{cause}")
            return " - ".join(bits)
        key, pressure = self.worst_need()
        if pressure > 0.55:
            bits.append({"hunger": "hungry", "fatigue": "tired",
                         "cold": "cold", "despair": "low"}.get(key, key))
        else:
            bits.append("ok")
        if self.carrying and self.carry_qty > 0:
            bits.append(f"carrying {self.carrying} x{self.carry_qty}")
        if self.holds_candle:
            bits.append("candle")
        act = getattr(self.action, "name", None) or self.path_goal
        if isinstance(act, str) and act:
            bits.append(act)
        return " - ".join(bits)

    # -------------------------------------------------------------- death --
    def hurt(self, amount: float, cause: str = "mauled") -> bool:
        """Take damage through armour. Returns True if this killed them.

        Armour is a flat fraction absorbed rather than a hit-point pool: a
        stickman in leather takes roughly half as long to bring down, which is
        the difference between a wolf pack being a funeral and being a fight.

        The epsilon on the death test is not defensive tidying, it is a fix for a
        live soft-lock, and it is worth writing down because the same shape will
        come back. ``dragons._devour`` deals ``MAX_HEALTH * 10.0`` = 1000 and
        ``ARMOUR_DRAGONSCALE`` clamps to 0.9 absorbed, which the constants file
        documents as "exactly MAX_HEALTH, so a dragonscale wearer is still
        eaten". In binary it is not exactly MAX_HEALTH::

            1000.0 * (1.0 - 0.9)  ==  99.99999999999997
            100.0 - that          ==  2.842170943040401e-14   # > 0.0

        So a scale-wearer at EXACTLY full health walked out of a dragon's mouth
        with 2.8e-14 hp while one at 99 hp died - a difference of one part in
        10^15 deciding whether the mechanic works. Worse, ``_devour`` reads
        ``hurt`` returning False as "nobody died, so I have not fed", never sets
        ``sated``, and a dragon that never sates never becomes killable: the
        colony's best armour permanently disabled the dragon fight for that
        world. Tried and rejected: rewriting the arithmetic as
        ``amount - amount * absorbed`` (which happens to be exact for these two
        numbers) - it fixes this pair by luck and moves the residue somewhere
        else, and it perturbs every other damage number in the game by an ulp.
        Snapping the *result* fixes the whole class, and 1e-9 hp against
        MAX_HEALTH = 100 is nine orders of magnitude below anything a caller
        means by "still standing".
        """
        if not self.alive or amount <= 0.0:
            return False
        taken = float(amount) * (1.0 - _clamp(self.armour, 0.0, 0.9))
        self.health = max(0.0, self.health - taken)
        if self.health <= _HEALTH_EPSILON:
            self.health = 0.0
            return self.die(cause)
        return False

    def heal(self, dt: float) -> None:
        if self.alive and self.health < MAX_HEALTH:
            self.health = min(MAX_HEALTH, self.health + HEALTH_REGEN_PER_SEC * dt)

    def die(self, cause: str = "") -> bool:
        """Kill this agent. Idempotent; returns True only on the transition.

        Sets ``alive=False`` and ``dead_t=0.0`` (the grave hand-off clock),
        drops the current action and walk target, then fires the death hooks.
        """
        if not self.alive:
            return False
        self.alive = False
        self.dead_t = 0.0
        self.death_cause = _s(cause, "") or ""
        self.action = None
        self.action_raw = None
        self.pose_override = None
        self.target_x = None
        self.path_goal = None
        self.climbing = False
        self.climb_t = 0.0
        self.speech = None
        self.speech_t = 0.0
        self.vx = 0.0
        # Nothing holds a corpse up. The body drops off the tower it was
        # standing on and lies where it lands, which is also where its grave
        # goes; leaving the perch set would hang it in the air instead.
        self.perch_y = None
        self._perch_t = 0.0
        _fire_death(self, self.death_cause)
        return True

    def death_event_kind(self) -> str:
        """Chronicle event kind matching this agent's death cause."""
        return death_kind(self.death_cause)

    def ready_for_grave(self, delay: float = GRAVE_DELAY) -> bool:
        """True once the corpse has lain long enough to become a gravestone."""
        return (not self.alive) and self.dead_t >= _f(delay, GRAVE_DELAY)

    def grave_record(self) -> dict[str, Any]:
        """Plain dict handed to props.py to build the gravestone."""
        return {
            "agent_id": int(self.id),
            "name": str(self.name),
            "color": [int(c) for c in self.color],
            "x": float(self.x),
            "y": float(self.y),
            "cause": str(self.death_cause),
            "generation": int(self.generation),
            "role": str(self.role),
        }

    # ------------------------------------------------------------ physics --
    def apply_physics(
        self,
        dt: float,
        terrain: "Terrain | None" = None,
        rng: Any = None,
    ) -> None:
        """Advance position, timers and pose state by *dt* seconds.

        Walks toward ``target_x``, sticks to the terrain surface, climbs steep
        faces (with a slip chance that rises with fatigue and steepness), falls
        under gravity once airborne, dies on an impact faster than
        ``FALL_LETHAL_SPEED``, and clamps x to the LAND - 0 .. WORLD_W-1, which
        is not the frame and has not been since the world went to 6400 px. Note
        the ordering that makes that clamp weaker than it looks:
        ``world._tick_agents`` runs the action and THEN this, so anything an
        action writes to ``x`` is clamped in the same tick - but ``ufo`` and
        ``dragons`` tick AFTER the agent loop and write ``x`` themselves, under
        their own bounds. Those bounds must stay WORLD_W-based to agree with
        this one - while this clamp said 1599 and theirs said WORLD_W they did
        not, and a held or carried body ended its tick a little past the wall
        everybody else was pinned to (measured: 1600.4 to 1638.1).

        An agent standing on a platform (see :meth:`perch`) does none of the
        walking or falling: it stands - but ``_perch_step`` clamps too, so the
        edges hold there as well. Never raises.
        """
        try:
            self._physics(_clamp(_f(dt, 0.0), 0.0, _MAX_DT), terrain, rng or _RNG)
        except Exception:
            log.exception("physics failed for agent %s (%s)", self.id, self.name)
            # Leave the agent where it was rather than propagating.

    # -- internals ---------------------------------------------------------
    def _physics(self, dt: float, terrain: "Terrain | None", rng: Any) -> None:
        if dt <= 0.0:
            return
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            self.x = _clamp(_f(self.x, _EDGE_MAX * 0.5), _EDGE_MIN, _EDGE_MAX)
            self.y = _ground_y(terrain, self.x)
            self.vx = self.vy = 0.0

        self._tick_timers(dt)

        if not self.alive:
            self.dead_t += dt
            self.climbing = False
            self._fall_step(dt, terrain, lethal=False)
            self.vx *= max(0.0, 1.0 - 6.0 * dt)
            return

        gy = _ground_y(terrain, self.x)
        if self.perch_y is not None and self._perch_step(dt, gy):
            return                        # standing on a platform, not falling
        airborne = (not self.on_ground) or (self.y < gy - GROUND_SNAP)
        if airborne:
            self._fall_step(dt, terrain, lethal=True)
        else:
            self._ground_step(dt, terrain, rng, gy)
        self._clamp_edges()

    def _tick_timers(self, dt: float) -> None:
        # Walk cycle advances with distance covered, so feet don't skate. The
        # floor keeps idle/chop/build cycles ticking while standing still.
        speed_ratio = abs(self.vx) / max(1.0, WALK_SPEED)
        self.anim_t += dt * _clamp(0.42 + speed_ratio, 0.42, 3.2)
        if self.anim_t > 1.0e6:
            self.anim_t = math.fmod(self.anim_t, 1.0e6)
        if self.speech is not None:
            self.speech_t -= dt
            if self.speech_t <= 0.0:
                self.speech = None
                self.speech_t = 0.0

    def _ground_step(
        self, dt: float, terrain: "Terrain | None", rng: Any, gy: float
    ) -> None:
        self.on_ground = True
        self.fall_t = 0.0
        self.y = gy

        target = self.target_x
        if target is None:
            self.vx *= max(0.0, 1.0 - 9.0 * dt)
            self.climbing = False
            self.climb_t = 0.0
            return

        dx = _f(target, self.x) - self.x
        if abs(dx) <= ARRIVE_EPS:
            self.x = _clamp(_f(target, self.x), _EDGE_MIN, _EDGE_MAX)
            self.y = _ground_y(terrain, self.x)
            # Arriving clears the target but *keeps* path_goal: the owning
            # action still needs to know what it walked here for. Only an
            # abandoned trip (unreachable cliff, world edge) calls stop().
            self.target_x = None
            self.vx = 0.0
            self.climbing = False
            self.climb_t = 0.0
            return

        d = 1.0 if dx > 0.0 else -1.0
        self.facing = 1 if d > 0.0 else -1

        ahead = _clamp(self.x + d * STEP_PROBE, _EDGE_MIN, _EDGE_MAX)
        gy_ahead = _ground_y(terrain, ahead)
        span = max(1e-3, abs(ahead - self.x))
        rise = gy - gy_ahead                       # >0: ground ahead is higher
        steep = abs(rise) / span
        # Terrain.slope() is the authority when it looks sane; the height
        # difference is the fallback so a stub Terrain still behaves.
        ts = abs(_terrain_slope(terrain, self.x + d * STEP_PROBE * 0.5))
        if 0.0 <= ts < 50.0:
            steep = max(steep, ts)

        speed = WALK_SPEED * self.speed_scale()
        if steep <= MAX_SLOPE_WALK:
            self.climbing = False
            self.climb_t = 0.0
            speed *= 1.0 - 0.35 * (steep / max(1e-3, MAX_SLOPE_WALK))
        elif rise > 0.0 and steep <= MAX_SLOPE_CLIMB:
            # A steep but climbable face: slow, and you can come off it.
            self.climbing = True
            self.climb_t += dt
            speed = CLIMB_SPEED * _clamp(self.gait, 0.6, 1.4)
            if _rand_of(rng) < self.slip_risk(dt, steep):
                self._slip(d)
                return
        elif rise <= 0.0:
            if -rise > STEP_FALL_MAX:
                # A killing drop. Refuse it exactly as the cliff branch below
                # refuses an unclimbable rise - walking off this would not be a
                # step, it would be a death. Giving the goal up lets behaviour
                # re-score, and a genuinely wanted descent still has the
                # deliberate route through actions' ledge guard.
                self.stop()
                self.climbing = False
                self.climb_t = 0.0
                return
            # A survivable drop ahead: walk up to the lip, the snap check pushes
            # us off, and the landing is one the agent can take.
            self.climbing = False
            self.climb_t = 0.0
            speed *= 0.6
        else:
            # A cliff, not a slope. Refuse it and give the goal up so behaviour
            # re-scores instead of grinding against a wall forever.
            self.stop()
            self.climbing = False
            self.climb_t = 0.0
            return

        step = min(abs(dx), max(0.0, speed) * dt)
        self.x += d * step
        self.vx = d * speed
        self._clamp_edges()

        ngy = _ground_y(terrain, self.x)
        if (ngy - self.y) > STEP_DROP_MAX and not self.climbing:
            # Walked off an edge.
            self.on_ground = False
            self.fall_t = 0.0
            self.vy = max(self.vy, 0.0)
            self.climbing = False
        else:
            self.y = ngy
            self.vy = 0.0

    def slip_risk(self, dt: float, steep: float, scale: float = 1.0) -> float:
        """Chance of losing grip this tick on a face of steepness *steep*.

        One formula for both directions of travel: tired and hungry agents come
        off a face sooner, and steeper is worse. ``scale`` lets a descent buy
        itself better odds than a climb without forking the maths.
        """
        return (
            SLIP_CHANCE_PER_SEC
            * max(0.0, _f(dt, 0.0))
            * max(0.0, _f(scale, 1.0))
            * (0.55 + 0.85 * self.fatigue + 0.25 * self.hunger)
            * _clamp(abs(_f(steep, 0.0)) / max(1e-3, MAX_SLOPE_CLIMB), 0.2, 1.0)
        )

    def descend_step(
        self,
        dt: float,
        terrain: "Terrain | None" = None,
        *,
        direction: float = 1.0,
        rng: Any = None,
        speed: float = CLIMB_SPEED,
    ) -> float:
        """Climb *down* the ledge ahead, staying attached to the surface.

        The mirror image of the climb branch in :meth:`_ground_step`: the agent
        moves at ``speed`` horizontally, the step is further shortened so the
        face is never descended faster than ``DESCENT_DROP_RATE``, and y is
        pinned to the ground the whole way, so no fall velocity ever
        accumulates and no landing check can fire. The one way down badly is
        the shared slip roll, which pitches you forward off the face.

        Returns the horizontal distance actually covered. ``0.0`` means the
        descent did not happen: there was no drop to descend, or the agent
        slipped and is now airborne (check ``on_ground`` to tell them apart).
        Never raises - callers are on the per-frame path.
        """
        try:
            dt = _clamp(_f(dt, 0.0), 0.0, _MAX_DT)
            if dt <= 0.0 or not self.alive:
                return 0.0
            d = 1.0 if _f(direction, 1.0) >= 0.0 else -1.0

            gy = _ground_y(terrain, self.x)
            step = max(0.0, _f(speed, CLIMB_SPEED)) * dt
            nx = _clamp(self.x + d * step, _EDGE_MIN, _EDGE_MAX)
            step = abs(nx - self.x)
            if step <= 0.0:
                return 0.0                       # against the world edge
            drop = _ground_y(terrain, nx) - gy
            if drop <= 0.0:
                return 0.0                       # nothing to descend

            # Rate-limit the *fall* of the step, not just its length. Terrain is
            # piecewise linear, so scaling the step by the overshoot lands on
            # the wanted drop in one pass.
            budget = DESCENT_DROP_RATE * dt
            if drop > budget:
                short_x = _clamp(self.x + d * step * (budget / drop),
                                 _EDGE_MIN, _EDGE_MAX)
                short_step = abs(short_x - self.x)
                short_drop = _ground_y(terrain, short_x) - gy
                if short_step > 0.0 and short_drop > 0.0:
                    nx, step, drop = short_x, short_step, short_drop
                # Otherwise the surface is a step function here, not a slope:
                # shortening walks back onto the lip and the descent never
                # starts. Take the whole drop instead - still attached to the
                # ground, still not a fall.

            self.facing = 1 if d > 0.0 else -1
            self.climbing = True
            self.climb_t += dt
            steep = drop / max(1e-3, step)
            if _rand_of(rng) < self.slip_risk(dt, steep, DESCENT_SLIP_SCALE):
                self._slip(-d)          # pitched forward, over the lip
                return 0.0

            self.x = nx
            self.vx = d * (step / dt)
            self.y = gy + drop
            self.vy = 0.0
            self.on_ground = True
            self.fall_t = 0.0
            self._clamp_edges()
            return step
        except Exception:
            log.exception("descend failed for agent %s (%s)", self.id, self.name)
            return 0.0

    def perch(self, y: float) -> None:
        """Stand on a platform at *y* for this tick, above the terrain.

        The way to hold an agent off the ground without killing it. Writing y
        alone is not enough and never was: apply_physics reads anything above
        ``ground_y`` as airborne, so an action that pinned a lookout to his
        tower deck had gravity charged against him for every second he stood
        up there. Measured on the ladder alone - 0.7 s of "climbing" during
        which he never actually got clear of the ground - vy reached 630 px/s,
        and the step back onto level ground was scored as a fatal impact. A
        full watch is 70-190 s, which is TERMINAL_VY and a certain death for
        anyone who serves one out.

        The mirror of :func:`actions._plant`, which asserts the same agreement
        against the ground. Renew it every tick for as long as the platform
        holds the agent up; see :data:`PERCH_LEASE` for why it expires.
        """
        try:
            y = _f(y, self.y)
            if not math.isfinite(y):
                return
            self.perch_y = y
            self._perch_t = 0.0
            self.y = y
            self.vy = 0.0
            self.fall_t = 0.0
            self.on_ground = True
            self.climbing = False
        except Exception:
            log.exception("perch failed for agent %s (%s)", self.id, self.name)

    def _perch_step(self, dt: float, gy: float) -> bool:
        """Hold this tick's perch. False once it lapses and physics resumes."""
        y = self.perch_y
        self._perch_t += dt
        if (y is None or not math.isfinite(y) or y >= gy - GROUND_SNAP
                or self._perch_t > PERCH_LEASE):
            # Lapsed, or the ground has risen to meet the platform. Set them
            # down rather than hand a body 68 px up to gravity - the lapse
            # branch only fires when an action vanished mid-watch, and the
            # small snap is invisible next to killing the agent for it.
            self.perch_y = None
            self._perch_t = 0.0
            self.y = gy
            self.vy = 0.0
            self.fall_t = 0.0
            self.on_ground = True
            return False
        self.y = y
        self.vy = 0.0
        self.fall_t = 0.0
        self.on_ground = True
        # A platform is somewhere you stand, not somewhere you walk: bleed off
        # any horizontal speed the trip here left behind so the pose reads as
        # standing and nothing drags the agent off the side of the tower.
        self.vx *= max(0.0, 1.0 - 9.0 * dt)
        self._clamp_edges()
        return True

    def _slip(self, d: float) -> None:
        """Lose grip mid-climb: detach from the face and start falling."""
        self.climbing = False
        self.climb_t = 0.0
        self.on_ground = False
        self.fall_t = 0.0
        self.vy = 12.0
        self.vx = -d * 16.0
        self.target_x = None
        self.morale = _clamp(self.morale - 0.08, 0.0, 1.0)

    def _fall_step(self, dt: float, terrain: "Terrain | None", *, lethal: bool) -> None:
        self.on_ground = False
        self.climbing = False
        self.fall_t += dt
        self.vy = min(TERMINAL_VY, self.vy + GRAVITY * dt)
        self.vx *= max(0.0, 1.0 - AIR_DRAG * dt)
        self.x += self.vx * dt
        self.y += self.vy * dt
        self._clamp_edges()

        gy = _ground_y(terrain, self.x)
        if self.y >= gy:
            impact = self.vy
            self.y = gy
            self.on_ground = True
            self.vy = 0.0
            self.vx *= 0.25
            if lethal and impact > FALL_LETHAL_SPEED and not self._boots_catch():
                self.die("fall")
            else:
                self.fall_t = 0.0

    def _boots_catch(self) -> bool:
        """True when ironshod boots took this landing, so the caller must not kill.

        THE ONLY RELIC EFFECT THAT LIVES IN THIS MODULE, and the one that pays
        best. Falling is the largest single killer in the game (191 of 565 deaths
        over 16 seeds x 120 sim-min), and it is one of the causes that never
        touched ``Stickman.hurt`` at all: a landing above ``FALL_LETHAL_SPEED``
        went straight to ``die()``, so armour, health and every other survivable-
        damage idea in the codebase were simply not consulted. That is why the
        boots were a no-op for a release - ``FALL_SURVIVED_DAMAGE`` existed, was
        documented down to the measured effect size, and had zero readers.

        Routed through :meth:`hurt` deliberately, because constants.py specifies
        it that way ("Still routed through ``Stickman.hurt``, so it goes through
        armour ... and a boot-wearer already at low health can still die of the
        fall"). So the boots remove the instant-death gate, not gravity: land on
        30 hp and you still die, with ``death_cause`` still "fall", and the grave,
        the chronicle line and the stats all read exactly as they did before.

        The boots are NOT consumed. They are worn gear, not a charge - the amulet
        and the cairnstone are the two that spend themselves, and both do it
        through ``items.consume`` where ``relics_used`` can be booked. Nothing
        here touches the relic slot, so one pair saves its wearer as often as the
        wearer falls, which is the point of it being the common drop.

        Named for what the BOOTS did rather than for what the agent did, because
        the caller has to read as "a lethal landing kills you, unless the boots
        catch it" - one exception to ordinary lethality, not a second way to die.
        """
        if self.relic != RELIC_BOOTS:
            return False
        # hurt() may still kill them, and if it does the cause is already the
        # fall, so the caller's die() must not run again either way - which is
        # what returning True here guarantees.
        killed = self.hurt(FALL_SURVIVED_DAMAGE, "fall")
        if not killed:
            # Counted AFTER the damage, and only when they got up again.
            # Bumping it before the hurt (which is how this was first written)
            # counts the branch firing rather than the life saved, so a colony
            # whose every booted faller landed on 5 hp and died anyway would
            # report a tidy row of "saves" next to their graves. Nothing is lost
            # by the stricter reading: "did the branch run at all" is still
            # answerable from the death record, because a booted agent who dies
            # here is still in the histogram under "fall".
            self.fall_saves += 1
        return True

    def _clamp_edges(self) -> None:
        """The edges of the LAND are hard walls, no escape.

        Not the edges of the screen - the two stopped being the same thing when
        WORLD_W grew past RENDER_W. The camera is free to sit anywhere; a body
        is not, and 0 .. WORLD_W-1 is the whole of what it may occupy.
        """
        if self.x < _EDGE_MIN:
            self.x = _EDGE_MIN
            if self.vx < 0.0:
                self.vx = 0.0
            if self.target_x is not None and self.target_x < _EDGE_MIN:
                self.stop()
        elif self.x > _EDGE_MAX:
            self.x = _EDGE_MAX
            if self.vx > 0.0:
                self.vx = 0.0
            if self.target_x is not None and self.target_x > _EDGE_MAX:
                self.stop()

    # ------------------------------------------------------------ persist --
    def to_dict(self) -> dict[str, Any]:
        """Everything needed to rebuild this agent exactly."""
        action_state = self.action_raw
        if self.action is not None:
            fn = getattr(self.action, "to_dict", None)
            if callable(fn):
                try:
                    got = fn()
                    if isinstance(got, dict):
                        action_state = got
                except Exception:
                    log.debug("action to_dict failed for %s", self.name, exc_info=True)
        return {
            "id": int(self.id),
            "name": str(self.name),
            "color": [int(c) for c in self.color],
            "morph": str(self.morph),
            "x": float(self.x), "y": float(self.y),
            "vx": float(self.vx), "vy": float(self.vy),
            "facing": int(self.facing),
            "hunger": float(self.hunger),
            "fatigue": float(self.fatigue),
            "warmth": float(self.warmth),
            "morale": float(self.morale),
            "carrying": self.carrying,
            "carry_qty": int(self.carry_qty),
            "role": str(self.role),
            "generation": int(self.generation),
            "action": action_state,
            "holds_candle": bool(self.holds_candle),
            "holds_torch": bool(self.holds_torch),
            "health": float(self.health),
            "weapon": str(self.weapon),
            "armour": float(self.armour),
            "attack_cd": float(self.attack_cd),
            "target_animal": self.target_animal,
            "relic": str(self.relic),
            "fall_saves": int(self.fall_saves),
            "taken": bool(self.taken),
            "lift_t": float(self.lift_t),
            "beamed": bool(self.beamed),
            "inside": self.inside,
            "perch_y": None if self.perch_y is None else float(self.perch_y),
            "alive": bool(self.alive),
            "anim_t": float(self.anim_t),
            "target_x": None if self.target_x is None else float(self.target_x),
            "path_goal": self.path_goal,
            "climb_t": float(self.climb_t),
            "fall_t": float(self.fall_t),
            "dead_t": float(self.dead_t),
            "speech": self.speech,
            "speech_t": float(self.speech_t),
            "starve_t": float(self.starve_t),
            "exposure_t": float(self.exposure_t),
            "blocked_t": float(self._blocked_t),
            "birth_time": float(self.birth_time),
            "parent_ids": [int(p) for p in self.parent_ids],
            "on_ground": bool(self.on_ground),
            "climbing": bool(self.climbing),
            "gait": float(self.gait),
            "death_cause": str(self.death_cause),
            "pose_override": self.pose_override,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Stickman":
        """Rebuild from a save. Missing keys take defaults, junk is ignored."""
        s = cls()
        if not isinstance(d, dict):
            return s
        # Every integer below comes off a FILE, so every one of them is bounded
        # (see MAX_COUNTER). ``id`` keeps its sign - a negative id is junk but it
        # is junk this build already tolerated, and only the magnitude was ever
        # the hazard: Population.from_dict derives next_id from ``max(a.id + 1)``,
        # so a 309-digit id here is a 309-digit id on every colonist born after
        # the load.
        s.id = _count(d.get("id"), 0, lo=-MAX_COUNTER)
        s.name = _s(d.get("name"), None) or f"Agent{s.id}"
        s.color = _color(d.get("color"), color_for_id(s.id))
        # A morph name from a newer build, a truncated save or a hand edit is
        # dropped here rather than stored: the body degrades to normal, which is
        # invisible, instead of raising in the middle of a load.
        s.morph = normalise_morph(d.get("morph"))
        s.x = _clamp(_f(d.get("x"), 0.0), _EDGE_MIN, _EDGE_MAX)
        s.y = _f(d.get("y"), DEFAULT_GROUND_Y)
        s.vx = _clamp(_f(d.get("vx"), 0.0), -TERMINAL_VY, TERMINAL_VY)
        s.vy = _clamp(_f(d.get("vy"), 0.0), -TERMINAL_VY, TERMINAL_VY)
        s.facing = -1 if _i(d.get("facing"), 1) < 0 else 1
        s.hunger = _unit(d.get("hunger"), 0.0)
        s.fatigue = _unit(d.get("fatigue"), 0.0)
        s.warmth = _unit(d.get("warmth"), 0.0)
        s.morale = _unit(d.get("morale"), 0.6)
        s.carrying = _s(d.get("carrying"), None)
        s.carry_qty = _count(d.get("carry_qty"), 0)
        role = _s(d.get("role"), ROLE_GATHERER) or ROLE_GATHERER
        s.role = role if is_role(role) else ROLE_GATHERER
        s.generation = _count(d.get("generation"), 0)
        s.holds_candle = _b(d.get("holds_candle"), False)
        s.holds_torch = _b(d.get("holds_torch"), True)
        s.health = _f(d.get("health"), MAX_HEALTH)
        s.weapon = _s(d.get("weapon"), WEAPON_NONE) if "_s" in globals() else str(d.get("weapon") or WEAPON_NONE)
        s.armour = _f(d.get("armour"), ARMOUR_NONE)
        s.attack_cd = _f(d.get("attack_cd"), 0.0)
        # ``int(float("nan"))` RAISES, and this used to be a bare int(). Inside
        # Stickman.from_dict that is not a lost field, it is a lost COLONIST:
        # Population.from_dict catches the exception and skips the whole record.
        # _count coerces and bounds in one move and cannot raise.
        ta = d.get("target_animal")
        s.target_animal = (_count(ta, 0, lo=-MAX_COUNTER)
                           if isinstance(ta, (int, float)) else None)
        # The relic slot. Validated against RELIC_KINDS on the way IN, exactly as
        # morph is, so a hand edit or a save from a newer build reads as "carrying
        # nothing" rather than as an item with no behaviour attached and no way to
        # take it off. items.worn() re-checks on the way out; both ends validate
        # because either end can be the one a future caller trusts.
        rel = _s(d.get("relic"), RELIC_NONE) or RELIC_NONE
        s.relic = rel if rel in RELIC_KINDS else RELIC_NONE
        # Re-assert the two mirrors items.equip() writes. A save produced by this
        # build always has them already, so this is a no-op in practice and is
        # here for the case that is not: armour and weapon are separate persisted
        # fields, and a save with the relic but not its mirror would reload
        # someone visibly wearing the dragonscale who takes full damage, or
        # holding the wyrm-gun with a spear's reach. The relic is the source of
        # truth for its own effect; nothing else can be.
        if s.relic == RELIC_SCALE:
            s.armour = max(s.armour, ARMOUR_DRAGONSCALE)
        elif s.relic == RELIC_BFG:
            s.weapon = WEAPON_BFG
        s.fall_saves = _count(d.get("fall_saves"), 0)
        s.taken = _b(d.get("taken"), False)
        s.lift_t = _f(d.get("lift_t"), 0.0)
        s.beamed = _b(d.get("beamed"), False)
        # Same bare-int() hazard as target_animal above, and the same fix.
        _ins = d.get("inside")
        s.inside = (_count(_ins, 0, lo=-MAX_COUNTER)
                    if isinstance(_ins, (int, float)) else None)
        # The lease starts fresh on load: whichever action put them up there
        # gets PERCH_LEASE seconds to claim them again, and if the save had no
        # such action the perch simply lapses and sets them down.
        _pch = d.get("perch_y")
        s.perch_y = _f(_pch, s.y) if isinstance(_pch, (int, float)) else None
        if s.perch_y is not None and not math.isfinite(s.perch_y):
            s.perch_y = None
        s._perch_t = 0.0
        s.alive = _b(d.get("alive"), True)
        s.anim_t = _f(d.get("anim_t"), 0.0)

        tx = d.get("target_x")
        s.target_x = None if tx is None else _clamp(_f(tx, 0.0), _EDGE_MIN, _EDGE_MAX)
        s.path_goal = _s(d.get("path_goal"), None)
        s.climb_t = max(0.0, _f(d.get("climb_t"), 0.0))
        s.fall_t = max(0.0, _f(d.get("fall_t"), 0.0))
        s.dead_t = max(0.0, _f(d.get("dead_t"), 0.0))
        s.speech = _s(d.get("speech"), None)
        s.speech_t = max(0.0, _f(d.get("speech_t"), 0.0))
        if s.speech is None:
            s.speech_t = 0.0
        # Attrition clocks survive a restart: reloading must not hand a
        # starving agent a fresh 90 seconds.
        s.starve_t = max(0.0, _f(d.get("starve_t"), 0.0))
        s.exposure_t = max(0.0, _f(d.get("exposure_t"), 0.0))
        s._blocked_t = max(0.0, _f(d.get("blocked_t"), 0.0))
        s.birth_time = _f(d.get("birth_time"), 0.0)

        parents = d.get("parent_ids")
        if isinstance(parents, (list, tuple)):
            s.parent_ids = [_count(p, 0) for p in parents if _i(p, -1) >= 0]

        s.on_ground = _b(d.get("on_ground"), True)
        s.climbing = _b(d.get("climbing"), False)
        s.gait = _clamp(_f(d.get("gait"), 1.0), 0.5, 1.6)
        s.death_cause = _s(d.get("death_cause"), "") or ""
        s.pose_override = _s(d.get("pose_override"), None)

        raw = d.get("action")
        if isinstance(raw, dict):
            s.action_raw = raw
            s.action = _rehydrate_action(raw)
            if s.action is not None:
                s.action_raw = None
        return s


# ------------------------------------------------------- action rehydration --
#: Set by actions.py (or world.py) to ``Action.from_dict``-alike. Left as a hook
#: so entities.py never has to import actions.py, which would be circular.
ACTION_FROM_DICT: Callable[[dict[str, Any]], Any] | None = None


def _rehydrate_action(raw: dict[str, Any]) -> Any:
    fn = ACTION_FROM_DICT
    if fn is None:
        try:
            from . import actions as _actions            # noqa: PLC0415
        except Exception:
            return None
        fn = getattr(_actions, "action_from_dict", None)
        if fn is None:
            cls = getattr(_actions, "Action", None)
            fn = getattr(cls, "from_dict", None) if cls is not None else None
    if not callable(fn):
        return None
    try:
        return fn(raw)
    except Exception:
        log.debug("could not rehydrate action %r", raw.get("kind"), exc_info=True)
        return None


# ============================================================== Population ==
@dataclass
class Population:
    """The roster: every stickman, plus id/name/generation bookkeeping."""

    agents: list[Stickman] = field(default_factory=list)
    next_id: int = 1
    generation: int = 0                     # highest generation seen so far
    used_names: set[str] = field(default_factory=set)
    births: int = 0
    graves_raised: int = 0                  # corpses already turned into graves
    peak: int = 0

    @property
    def deaths(self) -> int:
        """Total deaths ever. Derived, so a death from *any* cause counts.

        Agents die from several places - a fall inside ``apply_physics``,
        lightning in events.py, :meth:`kill` - and a counter incremented in
        only one of them would silently drift. Corpses still on the ground plus
        graves already raised is exact by construction.
        """
        return self.graves_raised + sum(1 for a in self.agents if not a.alive)

    # ------------------------------------------------------------- access --
    def __len__(self) -> int:
        return len(self.agents)

    def __iter__(self) -> Iterator[Stickman]:
        return iter(self.agents)

    def alive_agents(self) -> list[Stickman]:
        """Every living stickman."""
        return [a for a in self.agents if a.alive]

    def dead_agents(self) -> list[Stickman]:
        """Corpses still on the ground (not yet turned into graves)."""
        return [a for a in self.agents if not a.alive]

    def alive_count(self) -> int:
        return sum(1 for a in self.agents if a.alive)

    def by_id(self, agent_id: int) -> Stickman | None:
        wanted = _i(agent_id, -1)
        for a in self.agents:
            if a.id == wanted:
                return a
        return None

    def by_name(self, name: str) -> Stickman | None:
        for a in self.agents:
            if a.name == name:
                return a
        return None

    def by_role(self, role: str) -> list[Stickman]:
        return [a for a in self.agents if a.alive and a.role == role]

    def candle_bearer(self) -> Stickman | None:
        """The living agent currently carrying the candle, if any."""
        for a in self.agents:
            if a.alive and a.holds_candle:
                return a
        return None

    def nearest(self, x: float, exclude: int = -1) -> Stickman | None:
        """Closest living agent to a world x, ignoring id *exclude*."""
        best: Stickman | None = None
        best_d = float("inf")
        for a in self.agents:
            if not a.alive or a.id == exclude:
                continue
            d = abs(a.x - _f(x, 0.0))
            if d < best_d:
                best, best_d = a, d
        return best

    # ------------------------------------------------------------ mutation --
    def spawn(
        self,
        x: float = 0.0,
        y: float = DEFAULT_GROUND_Y,
        *,
        rng: Any = None,
        name: str | None = None,
        role: str = ROLE_GATHERER,
        generation: int | None = None,
        parent_ids: list[int] | None = None,
        birth_time: float = 0.0,
        holds_candle: bool = False,
    ) -> Stickman:
        """Create, register and return a new stickman.

        Identity colour cycles through ``STICK_PALETTE`` by id, and the name is
        guaranteed unique against every name this population has ever used.
        """
        agent_id = max(1, _i(self.next_id, 1))
        self.next_id = agent_id + 1

        chosen = name if (isinstance(name, str) and name and name not in self.used_names) \
            else new_name(rng, self.used_names)
        if isinstance(name, str) and name:
            self.used_names.add(chosen)

        gen = self.generation if generation is None else max(0, _i(generation, 0))
        self.generation = max(self.generation, gen)

        # The one place a mutant is created. Rolled from the caller's seeded
        # stream and nowhere else, so a seed replays the same rare bodies; with
        # no rng this returns "" rather than reaching for a module RNG.
        morph = roll_morph(rng)

        s = Stickman(
            id=agent_id,
            name=chosen,
            color=color_for_id(agent_id),
            morph=morph,
            x=_clamp(_f(x, 0.0), _EDGE_MIN, _EDGE_MAX),
            y=_f(y, DEFAULT_GROUND_Y),
            facing=1 if _rand_of(rng) < 0.5 else -1,
            role=role if is_role(role) else ROLE_GATHERER,
            generation=gen,
            birth_time=_f(birth_time, 0.0),
            holds_candle=bool(holds_candle),
            gait=0.86 + 0.28 * _rand_of(rng),
            anim_t=6.0 * _rand_of(rng),
        )
        if parent_ids:
            s.parent_ids = [_i(p, 0) for p in parent_ids]
        self.agents.append(s)
        self.births += 1
        self.peak = max(self.peak, self.alive_count())
        return s

    def add(self, agent: Stickman) -> Stickman:
        """Adopt an already-built stickman (used by the loader)."""
        if not isinstance(agent, Stickman):
            return agent
        self.agents.append(agent)
        self.used_names.add(agent.name)
        self.next_id = max(self.next_id, agent.id + 1)
        self.generation = max(self.generation, agent.generation)
        self.peak = max(self.peak, self.alive_count())
        return agent

    def kill(self, agent: Stickman | int, cause: str = "") -> bool:
        """Kill an agent (or an agent id). True only on the live->dead edge."""
        target = agent if isinstance(agent, Stickman) else self.by_id(_i(agent, -1))
        if target is None:
            return False
        return target.die(cause)

    def remove(self, agent: Stickman | int) -> bool:
        """Drop an agent from the roster entirely (no death hooks)."""
        target = agent if isinstance(agent, Stickman) else self.by_id(_i(agent, -1))
        if target is None or target not in self.agents:
            return False
        self.agents.remove(target)
        return True

    def collect_graves(self, delay: float = GRAVE_DELAY) -> list[dict[str, Any]]:
        """Remove corpses that have lain long enough; return grave records.

        This is the hand-off to props.py: each dict carries the position, name,
        colour and cause needed to raise a gravestone.
        """
        out: list[dict[str, Any]] = []
        for a in list(self.agents):
            if a.ready_for_grave(delay):
                out.append(a.grave_record())
                try:
                    self.agents.remove(a)
                    self.graves_raised += 1
                except ValueError:
                    pass
        return out

    def apply_physics(self, dt: float, terrain: "Terrain | None" = None,
                      rng: Any = None) -> None:
        """Step physics for every agent. Convenience for world.tick()."""
        for a in self.agents:
            a.apply_physics(dt, terrain, rng)

    # ------------------------------------------------------------- stats --
    def stats(self) -> dict[str, float]:
        """Aggregate population statistics (numpy-backed means)."""
        living = self.alive_agents()
        out: dict[str, float] = {
            "alive": float(len(living)),
            "dead": float(len(self.agents) - len(living)),
            "births": float(self.births),
            "deaths": float(self.deaths),
            "peak": float(self.peak),
            "generation": float(self.generation),
        }
        if living:
            arr = np.array(
                [[a.hunger, a.fatigue, a.warmth, a.morale] for a in living],
                dtype=np.float32,
            )
            mean = arr.mean(axis=0)
            out["hunger"] = float(mean[0])
            out["fatigue"] = float(mean[1])
            out["warmth"] = float(mean[2])
            out["morale"] = float(mean[3])
        else:
            out["hunger"] = out["fatigue"] = out["warmth"] = 0.0
            out["morale"] = 0.0
        return out

    def role_counts(self) -> dict[str, int]:
        counts = {r: 0 for r in ROLES}
        for a in self.agents:
            if a.alive:
                counts[a.role] = counts.get(a.role, 0) + 1
        return counts

    # ----------------------------------------------------------- persist --
    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": [a.to_dict() for a in self.agents],
            "next_id": int(self.next_id),
            "generation": int(self.generation),
            "used_names": sorted(self.used_names),
            "births": int(self.births),
            "graves_raised": int(self.graves_raised),
            "deaths": int(self.deaths),      # derived; written for readability
            "peak": int(self.peak),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Population":
        pop = cls()
        if not isinstance(d, dict):
            return pop
        raw_agents = d.get("agents")
        if isinstance(raw_agents, (list, tuple)):
            for item in raw_agents:
                try:
                    pop.agents.append(Stickman.from_dict(item))
                except Exception:
                    log.debug("skipped an unreadable agent record", exc_info=True)

        names_raw = d.get("used_names")
        if isinstance(names_raw, (list, tuple, set)):
            pop.used_names = {str(n) for n in names_raw if isinstance(n, str)}
        # A save written before used_names existed still must not collide.
        pop.used_names.update(a.name for a in pop.agents)

        # EVERY counter below is a number a file claims, so every one of them is
        # bounded by MAX_COUNTER - see there for why the ceiling is 1e12 and why
        # an unbounded one was not merely ugly. ``max(0, _i(...))`` bounded them
        # from BELOW only, which is the half that cannot hurt you.
        #
        # The clamps do not disturb reconcile. ``births`` is the only one of
        # these the identity contains, World._rebase_books solves the identity
        # for ``died`` given whatever ``births`` turns out to be, and it does
        # that AFTER this method has returned - so a clamped births is simply a
        # smaller books, still balanced, still residual 0. Measured across every
        # counter here at 1e308: ok=1, residual=0, nothing raised.
        pop.next_id = _count(d.get("next_id"), 1, lo=1)
        for a in pop.agents:
            pop.next_id = _count(max(pop.next_id, a.id + 1), 1, lo=1)
        pop.generation = _count(
            max(_i(d.get("generation"), 0),
                max((a.generation for a in pop.agents), default=0)),
            0,
        )
        pop.births = _count(d.get("births"), len(pop.agents))
        corpses = sum(1 for a in pop.agents if not a.alive)
        if "graves_raised" in d:
            pop.graves_raised = _count(d.get("graves_raised"), 0)
        else:                                 # pre-graves_raised save
            pop.graves_raised = _count(_i(d.get("deaths"), 0) - corpses, 0)
        pop.peak = _count(max(_i(d.get("peak"), 0), pop.alive_count()), 0)
        return pop


def _rand_of(rng: Any) -> float:
    """Uniform [0,1) from a random.Random or numpy Generator; never raises."""
    if rng is not None:
        try:
            v = float(rng.random())
            if 0.0 <= v < 1.0:
                return v
        except Exception:
            pass
    return _RNG.random()


if __name__ == "__main__":   # pragma: no cover - headless smoke test
    class _FlatWithCliff:
        """Flat ground with a 200px drop at x=600, for the physics smoke test."""

        def ground_y(self, x: float) -> float:
            return 400.0 if x < 600.0 else 600.0

        def slope(self, x: float) -> float:
            return 0.0

    rng = random.Random(11)
    terrain = _FlatWithCliff()
    pop = Population()
    for i in range(4):
        pop.spawn(100.0 + i * 60.0, 400.0, rng=rng)

    deaths: list[str] = []
    on_death(lambda s, cause: deaths.append(f"{s.name}: {cause}"))

    # A watchtower deck: held 68 px above flat ground for the longest watch a
    # lookout can draw, then abandoned up there. None of it may become a fall.
    # Kept ahead of everything else in this block deliberately - it is the one
    # regression that reads as a working colony right up until it isn't.
    deck_pop = Population()
    guard = deck_pop.spawn(200.0, 400.0, rng=rng)
    for _ in range(int(30 * 190)):
        guard.perch(400.0 - 68.0)              # what the Lookout action does
        guard.apply_physics(1.0 / 30.0, terrain, rng)
        assert guard.vy == 0.0, ("gravity charged against a perch", guard.vy)
    assert guard.alive and guard.y == 400.0 - 68.0, guard.summary()
    for _ in range(int(30 * 2)):               # nobody renews the lease now
        guard.apply_physics(1.0 / 30.0, terrain, rng)
    assert guard.alive and not deaths, (guard.summary(), deaths)
    assert guard.perch_y is None and guard.y == 400.0, guard.summary()
    print("perch: 190s aloft with vy", guard.vy, "- set down at", guard.y)

    walker = pop.agents[0]
    walker.walk_to(900.0, "beyond the drop")
    for _ in range(int(30 * 40)):
        pop.apply_physics(1.0 / 30.0, terrain, rng)

    # STEP_FALL_MAX is 64 px and this ledge is 200, so a WALKING agent now
    # refuses it and gives the goal up - that refusal is exactly what the
    # constant was added for. It also quietly rotted this block: everything
    # below wanted a fall death and there was no longer any way to walk into
    # one, so `deaths == 1` had been failing here since STEP_FALL_MAX shipped
    # (tools/smoke.py only imports this module, so nothing was watching). Put
    # him over the lip explicitly instead. The subject under test is the fall
    # integrator and the grave hand-off, not the pathing, and the pathing now
    # has its own assertion on the line above.
    assert walker.alive and walker.x < 600.0, walker.summary()
    walker.x, walker.y = 700.0, 200.0
    walker.on_ground, walker.vy = False, 0.0
    # Long enough to fall, die, and lie there past GRAVE_DELAY - collect_graves
    # below needs the corpse to have aged, so this must not break early.
    for _ in range(int(30 * (GRAVE_DELAY + 8.0))):
        pop.apply_physics(1.0 / 30.0, terrain, rng)
    assert not walker.alive and walker.death_cause == "fall", walker.summary()

    print(walker.summary())
    print("deaths:", deaths)
    print("stats:", {k: round(v, 3) for k, v in pop.stats().items()})

    blob = pop.to_dict()
    clone = Population.from_dict(blob)
    assert clone.to_dict() == blob, "round-trip mismatch"
    assert clone.deaths == pop.deaths == 1, (clone.deaths, pop.deaths)
    assert Population.from_dict({"agents": [{"bogus": 1}, 7]}) is not None
    assert Stickman.from_dict(None).name == "?"
    print("round-trip ok;", len(clone), "agents,", clone.alive_count(), "alive")

    graves = pop.collect_graves()
    assert len(graves) == 1 and graves[0]["cause"] == "fall", graves
    assert pop.deaths == 1 and len(pop) == 3, (pop.deaths, len(pop))
    print("grave:", graves[0]["name"], "at", round(graves[0]["x"], 1))
    assert Population.from_dict(pop.to_dict()).deaths == 1

    # The same 200px ledge, taken deliberately instead of walked off.
    solo = Population()
    climber = solo.spawn(599.9, 400.0, rng=rng)   # toes on the lip
    for _ in range(int(30 * 20)):
        climber.descend_step(1.0 / 30.0, terrain, direction=1.0, rng=rng)
        climber.apply_physics(1.0 / 30.0, terrain, rng)
        if climber.y >= 599.5:
            break
    print("descended to", round(climber.y, 1), "alive:", climber.alive)
    assert climber.alive and climber.y >= 599.5, (climber.alive, climber.y)

    # Attrition: only *sustained* time at the cap counts, and it is slow.
    victim = solo.spawn(200.0, 400.0, rng=rng)
    victim.hunger = 1.0
    for _ in range(int(30 * (STARVE_LETHAL_SEC - 5.0))):
        victim.update_needs(1.0 / 30.0)
    assert victim.alive, "starved early"
    victim.hunger = 0.2                        # one meal buys a full reprieve
    victim.update_needs(1.0 / 30.0)
    assert victim.starve_t == 0.0, victim.starve_t
    victim.hunger = 1.0
    for _ in range(int(30 * (STARVE_LETHAL_SEC + 2.0))):
        victim.update_needs(1.0 / 30.0)
    assert not victim.alive and victim.death_cause == CAUSE_STARVED, victim.summary()
    print("attrition:", victim.summary())

    # Morphs: rare, seeded, round-tripping, and degrading on junk.
    counts: dict[str, int] = {}
    pop2 = Population()
    seeded = random.Random(99)
    for _ in range(4000):
        m = roll_morph(seeded)
        counts[m] = counts.get(m, 0) + 1
    odd = sum(v for k, v in counts.items() if k)
    print("morph roll over 4000:", counts, "-> mutant rate", round(odd / 4000, 4))
    assert 0.03 < odd / 4000 < 0.12, counts
    assert roll_morph(None) == MORPH_NONE, "unseeded rolls must never mutate"

    giant = pop2.spawn(300.0, 400.0, rng=seeded)
    giant.morph = "giant"
    assert giant.height() > AGENT_HEIGHT and giant.is_mutant(), giant.summary()
    back = Stickman.from_dict(giant.to_dict())
    assert back.morph == "giant" and back.height() == giant.height(), back.summary()
    junk = Stickman.from_dict({"id": 5, "morph": "wyrm"})
    assert junk.morph == MORPH_NONE and junk.height() == AGENT_HEIGHT, junk.morph
    assert Stickman.from_dict({"id": 6, "morph": ["nope"]}).morph == MORPH_NONE
    print("morph round-trip ok:", back.summary(), round(back.height(), 1), "px")

    # --- relics: the slot survives a save, and the boots actually catch ----
    # All three of these are regressions, not features. The relic feature
    # shipped once measuring 100% inert - 18 drops, 0 pickups, 0 effects, and 16
    # of 16 seeds bit-identical with the whole thing switched off - so every part
    # of it that lives in this module is pinned here by an assertion that fails
    # loudly rather than by a comment saying it ought to work.
    import json as _json

    scribe = Population()
    wearer = scribe.spawn(300.0, 400.0, rng=rng)
    for kind in RELIC_KINDS:
        wearer.relic = kind
        if kind == RELIC_SCALE:
            wearer.armour = ARMOUR_DRAGONSCALE
        elif kind == RELIC_BFG:
            wearer.weapon = WEAPON_BFG
        # Through json, not just to_dict/from_dict: the save file is json, and a
        # value that round-trips in memory but not through the file is the exact
        # shape of bug this whole block exists to catch.
        twin = Stickman.from_dict(_json.loads(_json.dumps(wearer.to_dict())))
        assert twin.relic == kind, (kind, twin.relic)
        assert twin.armour == wearer.armour and twin.weapon == wearer.weapon
    print("relic slot round-trips for all", len(RELIC_KINDS), "kinds")

    # Junk in the slot reads as carrying nothing, and the mirrors are re-asserted
    # even when a save lost them.
    assert Stickman.from_dict({"relic": "excalibur"}).relic == RELIC_NONE
    assert Stickman.from_dict({"relic": 7}).relic == RELIC_NONE
    assert Stickman.from_dict({"relic": None}).relic == RELIC_NONE
    assert Stickman.from_dict({}).relic == RELIC_NONE, "a pre-relic save broke"
    assert Stickman.from_dict(
        {"relic": RELIC_SCALE, "armour": 0.0}).armour == ARMOUR_DRAGONSCALE
    assert Stickman.from_dict(
        {"relic": RELIC_BFG, "weapon": "spear"}).weapon == WEAPON_BFG

    # The boots. Same drop, same seed, same terrain - the only difference is
    # what is on the wearer's feet. 400 px of fall is well past the 64 px it
    # takes to reach FALL_LETHAL_SPEED, so this is not a marginal landing.
    def _drop(relic: str, health: float = MAX_HEALTH) -> Stickman:
        lone = Population()
        faller = lone.spawn(700.0, 600.0, rng=random.Random(4242))
        faller.relic, faller.health = relic, health
        faller.y, faller.vy, faller.on_ground = 200.0, 0.0, False
        drop_rng = random.Random(4242)
        for _ in range(int(30 * 12)):
            faller.apply_physics(1.0 / 30.0, terrain, drop_rng)
            if faller.on_ground:
                break
        return faller

    bare = _drop(RELIC_NONE)
    booted = _drop(RELIC_BOOTS)
    assert not bare.alive and bare.death_cause == "fall", bare.summary()
    assert booted.alive, "ironshod boots did not catch a lethal fall"
    assert booted.fall_saves == 1, booted.fall_saves
    assert booted.health == MAX_HEALTH - FALL_SURVIVED_DAMAGE, booted.health
    assert booted.relic == RELIC_BOOTS, "the boots were consumed"
    # ...and they remove the instant-death gate, not gravity: land hurt enough
    # and the fall still kills you, still as a fall.
    frail = _drop(RELIC_BOOTS, health=FALL_SURVIVED_DAMAGE * 0.5)
    assert not frail.alive and frail.death_cause == "fall", frail.summary()
    # ...and that death is not booked as a save. fall_saves is the number of
    # lives the boots kept, not the number of times the branch ran, or a colony
    # that lost every booted faller would still show a row of saves.
    assert frail.fall_saves == 0, frail.fall_saves
    assert Stickman.from_dict(booted.to_dict()).fall_saves == 1
    print(f"boots: bare {bare.death_cause!r} vs booted "
          f"{booted.health:.0f} hp, saves {booted.fall_saves}")

    # The dragonscale must behave the same at 100 hp as at 99. _devour deals
    # MAX_HEALTH * 10 against a 0.9 clamp, which is "exactly MAX_HEALTH" in
    # decimal and 2.8e-14 short of it in binary; before the epsilon in hurt(),
    # only the wearer at EXACTLY full health walked out of the dragon's mouth,
    # and dragons.py then read the survival as "not fed" and never sated.
    for start in (MAX_HEALTH, MAX_HEALTH * 0.99, MAX_HEALTH * 0.5):
        meal = Stickman(health=start, armour=ARMOUR_DRAGONSCALE,
                        relic=RELIC_SCALE)
        assert meal.hurt(MAX_HEALTH * 10.0, "devoured"), \
            f"a dragonscale wearer at {start} hp survived being eaten"
        assert meal.health == 0.0, meal.health
    # ...while the interaction the scale exists FOR is untouched: BFG_DAMAGE is
    # 9x, not 10x, so the coat still walks out of a beam at 10 hp.
    gunned = Stickman(health=MAX_HEALTH, armour=ARMOUR_DRAGONSCALE,
                      relic=RELIC_SCALE)
    assert not gunned.hurt(MAX_HEALTH * 9.0, "disintegrated"), gunned.health
    assert 9.5 < gunned.health < 10.5, gunned.health
    print(f"dragonscale: eaten at 100/99/50 hp, survives a beam at "
          f"{gunned.health:.0f} hp")

    clear_death_listeners()
