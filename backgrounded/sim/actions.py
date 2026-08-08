"""Concrete stickman behaviours as resumable state machines.

One :class:`Action` class, one handler function per ``kind``. Keeping it a
single dataclass (rather than a subclass per behaviour) is what makes
``to_dict``/``from_dict`` round-tripping trivial: a save taken mid-build
reloads mid-build, because everything the machine needs lives in
``phase``/``t``/``data`` plus the target structure's own persisted progress.

No pygame in here. The only thing an action tells the renderer is ``pose``.

Everything an action touches on ``world`` is reached through the small
accessor layer at the top of this module, so a missing or renamed subsystem
degrades to a sensible default instead of raising inside the per-frame path.
"""
from __future__ import annotations

import logging
import math
import random
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..constants import (
    ALL_RESOURCES,
    CLIMB_SPEED,
    DAY_LENGTH_SEC,
    FARM_FIELD_SIZE,
    FARM_HARVEST_FOOD,
    FARM_TILL_SEC,
    HERMIT_EDGE_MARGIN,
    HERMIT_FELL_REACH,
    HERMIT_RESITE_SLACK,
    HERMIT_ROAM,
    HERMIT_STANDOFF_FRAC,
    HERMIT_STASH_CAP,
    HERMIT_VISIT_GIFT,
    HERMIT_VISIT_MORALE_GUEST,
    HERMIT_VISIT_MORALE_HOST,
    HERMIT_VISIT_PACE,
    HERMIT_VISIT_TIMEOUT,
    LITTER_CLUSTER_MIN,
    LITTER_CLUSTER_R,
    MAT_DIRT,
    MAT_GRASS,
    MAT_STONE,
    MAX_SLOPE_CLIMB,
    MAX_SLOPE_WALK,
    MINE_EDGE_WEIGHT,
    MINE_KEEP_OUT,
    MINE_MAX_WALK,
    MINE_SESSION_SEC,
    MINE_WALK_WEIGHT,
    MINE_YIELD_SEC,
    MINE_YIELD_STONE,
    RENDER_H,
    RENDER_W,
    RES_COOKED,
    RES_FIBRE,
    RES_FOOD,
    RES_GARBAGE,
    RES_STONE,
    RES_WOOD,
    WALK_SPEED,
)
from .entities import GROUND_SNAP, STEP_DROP_MAX
from .names import ROLE_HERMIT

# ---------------------------------------------------------------- geometry --
# RENDER_W used to mean two things at once because the world and the view were
# the same 1600 px. They are not any more: WORLD_W is how much land exists,
# RENDER_W is the camera (and the wallpaper image, which does not change size).
# Almost every RENDER_W in `sim/` meant the world and becomes WORLD_W.
#
# The exception is the third meaning, which is why STAGE_HALF/OFFSTAGE exist and
# why `stage_bounds`/`offstage_x` below are in this module: a fair amount of sim
# code means "somewhere a viewer can see" or "far enough away that nobody can".
# `sim/` must not learn that a camera exists - but it does not have to. The
# camera is RENDER_W wide and centred on the colony, so anything within
# STAGE_HALF of colony_center() is on screen by construction and anything past
# OFFSTAGE is off it. Both derive from colony_center() and the seeded stream, so
# determinism is untouched and no sim -> render dependency is created.
#
# The names live in constants.py, not here, and are imported separately from the
# block above only so this note has somewhere to sit next to the code that reads
# them. Nothing in `sim/` may hardcode 6400 - a second copy is exactly how the
# MAX_POP duplicate in combat_actions.py went stale.
from ..constants import OFFSTAGE, STAGE_HALF, WORLD_W

#: Ground drop, in px, that counts as a ledge rather than a slope. Roughly a
#: body height: below this a stickman steps down, above it they have to climb
#: down (or, panicking, jump).
CLIFF_DROP = 46.0

#: How far ahead to probe the ground for a ledge, in px. Must be several
#: times the per-tick step or a cliff edge is invisible until you are over it.
CLIFF_LOOKAHEAD = 14.0

#: Seconds an agent may stand blocked by terrain before it gives the goal up.
#: The action timeouts below are 75-120 s; waiting that long to notice a wall
#: burns a third of a run standing still, which is how "safe" pathing cost the
#: colony two thirds of its buildings.
BLOCKED_GIVE_UP = 2.0
from .structures import Structure, StructureRegistry

log = logging.getLogger(__name__)

__all__ = [
    "Action",
    "ACTION_KINDS",
    "POSES",
    "make_action",
    "action_from_dict",
    "CARRY_CAP",
    # accessor layer, reused by behavior.py
    "world_now", "is_night", "rng_of", "ground_y", "slope_at",
    "stock_qty", "stock_add", "stock_take", "stockpile_of",
    "agents_of", "agent_by_id", "structures_of", "props_of", "find_prop",
    "prop_alive", "hazards_of", "chronicle", "emit_speech",
    "celebrations_of", "push_celebration", "colony_center",
    "litter_clusters", "densest_litter", "free_litter_cluster",
]

# ------------------------------------------------------------------ tuning --
CARRY_CAP = 8               # units a stickman can shoulder at once
REACH = 14.0                # how close counts as "there"
FIRE_REACH = 26.0
TALK_REACH = 30.0
EAT_TIME = 2.6
COOK_TIME = 6.0
CONVERSE_TIME = 4.0
CELEBRATE_TIME = 6.0
MOURN_TIME = 5.0
PLANT_TIME = 2.5
CLIMB_TIME = 2.5
#: Seconds a hermit will spend walking to his own front door before giving up
#: and lying down where he stands. Generous against the distance he actually has
#: to cover (HERMIT_ROAM is 90 px, ~3 s of clear ground) and deliberately so:
#: the number is not a travel budget, it is the guard that stops a hut he
#: cannot reach - over a ledge, behind a fire - from costing him the night's
#: sleep. Fatigue is the one need with no field remedy; see `_h_sleep_rough`.
HERMIT_BED_WALK = 26.0
CHOP_YIELD = 6
STONE_YIELD = 5
BERRY_YIELD = 4

#: How far from the colony centre a tended field may spread. Crops beyond this
#: are somebody else's plot, not this colony's, for the purpose of "is the field
#: full" and "where do I till the next one".
FARM_FIELD_RADIUS = 260.0
#: A ripe crop this much further out than the field is still worth walking to.
FARM_REACH = FARM_FIELD_RADIUS * 1.8
HARVEST_TIME = 1.6          # seconds bent over a ripe crop before it is in hand
#: A quarry divot stays cosmetic: dug at most this deep, a little per yield, and
#: spread wide enough that it is a scoop in the ground, never a fall-risk cliff.
MINE_DIVOT_MAX = 8.0
MINE_DIVOT_PER = 1.3
MINE_DIVOT_HALF = 12        # px each side of the dig point the divot spans
#: px the working face steps along per yield, away from the colony. This is what
#: makes the excavation a RAMP rather than a shaft: 6 px along for 1.3 px down is
#: a gradient of about 0.22, comfortably inside MAX_SLOPE_WALK (0.9), so the cut
#: stays something a laden villager walks out of. Set it much smaller and the
#: face stops moving and you are digging a well again.
MINE_FACE_STEP = 6.0
#: The steepest ground a quarry may leave behind. Below MAX_SLOPE_CLIMB (2.6),
#: which is where a fall turns lethal, with margin - and deliberately ABOVE
#: MAX_SLOPE_WALK (0.9), because measuring found the flattest stone column
#: outside a real settlement already at 1.60 and a walk-limit gate refused to
#: dig anywhere on the map.
MINE_SAFE_GRADIENT = 2.0
SPEECH_SYMBOLS = ("?", "!", "~", "*", "+", "o", "^", "#")

POSES: tuple[str, ...] = (
    "idle", "walk", "run", "carry", "chop", "mine", "forage", "build",
    "eat", "sleep", "warm", "cook", "plant", "talk", "dance", "mourn",
    "flee", "climb", "lookout", "panic", "haul",
)

ACTION_KINDS: tuple[str, ...] = (
    "Wander", "GatherWood", "GatherStone", "ForageBerries", "HaulToStockpile",
    "BuildStructure", "RepairStructure", "Eat", "Sleep", "WarmAtFire",
    "CookFood", "PlantSapling", "Farm", "Mine", "Converse", "Celebrate",
    "Mourn", "FleeFrom", "ClimbTo", "Lookout", "FollowParent", "Panic",
    "CleanLitter", "UpgradeStructure",
)

_TREE_KINDS = ("tree", "pine", "oak", "deadtree")
_ROCK_KINDS = ("rock", "boulder", "stone", "outcrop")
_BUSH_KINDS = ("bush", "berry", "berrybush", "shrub")

_FALLBACK_RNG = random.Random(0xB16B00B5)


# ===========================================================================
#  World accessor layer - every one of these tolerates a missing subsystem
# ===========================================================================
def _clamp(v: float, lo: float, hi: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return lo
    if math.isnan(f):
        return lo
    return lo if f < lo else (hi if f > hi else f)


def _clamp01(v: float) -> float:
    return _clamp(v, 0.0, 1.0)


def _clamp_x(x: float) -> float:
    """Keep an x on the map. WORLD, not view - agents walk off camera all day."""
    return _clamp(x, 4.0, float(WORLD_W - 4))


def rng_of(world: Any) -> random.Random:
    """The world's RNG so replays stay deterministic; a module RNG otherwise.

    World carries two generators: `rng` is a numpy Generator (for array work)
    and `pyrng` is a stdlib Random. Callers here want the stdlib surface -
    randrange/choice/randint - which numpy's Generator does *not* provide.
    Duck-typing on .random/.uniform alone matches numpy and then blows up later
    at the first .randrange, so prefer pyrng and demand the full surface.
    """
    pr = getattr(world, "pyrng", None)
    if isinstance(pr, random.Random):
        return pr
    r = getattr(world, "rng", None)
    if isinstance(r, random.Random):
        return r
    if r is not None and all(hasattr(r, n) for n in
                             ("random", "uniform", "randrange", "choice")):
        return r  # type: ignore[return-value]
    return _FALLBACK_RNG


def world_now(world: Any) -> float:
    for name in ("world_time", "time_s", "elapsed", "t"):
        v = getattr(world, name, None)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def day_phase(world: Any) -> float:
    """0..1 through the day. 0 == dawn, 0.5 == dusk."""
    v = getattr(world, "day_phase", None)
    if isinstance(v, (int, float)):
        return float(v) % 1.0
    if callable(v):
        try:
            return float(v()) % 1.0
        except Exception:
            pass
    day = float(getattr(world, "day_length", DAY_LENGTH_SEC) or DAY_LENGTH_SEC)
    if day <= 0.0:
        day = float(DAY_LENGTH_SEC)
    return (world_now(world) % day) / day


def is_night(world: Any) -> bool:
    v = getattr(world, "is_night", None)
    if isinstance(v, bool):
        return v
    if callable(v):
        try:
            return bool(v())
        except Exception:
            pass
    return day_phase(world) >= 0.5


def ground_y(world: Any, x: float) -> float:
    t = getattr(world, "terrain", None)
    if t is not None:
        fn = getattr(t, "ground_y", None)
        if callable(fn):
            try:
                y = float(fn(_clamp_x(x)))
                if math.isfinite(y):
                    return y
            except Exception:
                pass
    return RENDER_H * 0.75


def slope_at(world: Any, x: float) -> float:
    t = getattr(world, "terrain", None)
    if t is not None:
        fn = getattr(t, "slope", None)
        if callable(fn):
            try:
                s = float(fn(_clamp_x(x)))
                if math.isfinite(s):
                    return s
            except Exception:
                pass
    return 0.0


# ------------------------------------------------------------- stockpile ----
def stockpile_of(world: Any) -> Any:
    sp = getattr(world, "stockpile", None)
    if sp is None:
        sp = {}
        try:
            setattr(world, "stockpile", sp)
        except Exception:
            pass
    return sp


def stock_qty(world: Any, res: str) -> int:
    sp = stockpile_of(world)
    try:
        if isinstance(sp, dict):
            return max(0, int(sp.get(res, 0)))
        fn = getattr(sp, "get", None) or getattr(sp, "qty", None)
        if callable(fn):
            return max(0, int(fn(res) or 0))
    except Exception:
        pass
    return 0


def stock_add(world: Any, res: str, qty: int) -> int:
    """Put `qty` in the colony store. Returns how much went in.

    THE OTHER HALF OF THE TILL GATE, and leaving it out would have turned the
    withdrawal fix into a slow donation. `combat_actions._c_craft` refunds an
    abandoned craft through here, and once `stock_take` bills a hermit's craft
    to his own stash an un-gated refund credits the COLONY - so every wolf that
    interrupted him at the workbench would have moved wood, stone, hide and
    fibre out of his pile and into theirs, one craft at a time, in the same
    direction as the bug being fixed. Same gate, same reason: see `_TILL_ACTOR`.
    """
    if qty <= 0 or not res:
        return 0
    if _hermit_at_till():
        return hermit_stash_add(world, res, qty)
    if res == RES_GARBAGE:
        # The one resource the stockpile refuses. Enforced here, at the single
        # choke point every deposit path goes through (_drop_all, the build
        # fetch, the cook hand-off, the player's Feed tool), rather than in the
        # cleanup action - because the thing that must never happen is garbage
        # reaching the store by some route nobody thought of. Everything that
        # reads colony wealth sums this dict, and sweepings are not wealth.
        # Returning 0 also leaves the carrier's hand full, so an abandoned haul
        # is dropped as litter by _c_clean instead of vanishing into the pile.
        return 0
    sp = stockpile_of(world)
    try:
        if isinstance(sp, dict):
            sp[res] = int(sp.get(res, 0)) + int(qty)
            return int(qty)
        for name in ("add", "deposit", "put"):
            fn = getattr(sp, name, None)
            if callable(fn):
                fn(res, int(qty))
                return int(qty)
    except Exception:
        log.debug("stock_add failed", exc_info=True)
    return 0


# ------------------------------------------------------- who is at the till --
#: THE ACTING AGENT, and the whole reason this exists is that `stock_take` was
#: never told who was withdrawing. The hermit gate was written as a set of
#: per-handler `is_hermit` branches - `_hermit_bias` zeroing his colony jobs,
#: `_deposit_step` declining the walk, `_h_build`'s fetch arm - and a gate made
#: of branches leaks by exactly one route: a handler nobody remembered to
#: branch. `_h_upgrade` and `_h_repair` carry fetch blocks byte-identical to
#: `_h_build`'s with no `is_hermit` in them at all, `_h_craft` pays through a
#: helper (`combat_actions._pay`) that does not contain the string `stock_take`,
#: and an agent promoted to hermit MID-ACTION walks into any of them holding a
#: job the scorer would never have given him. Measured over 5 seeds x 45 min,
#: every one of those routes fired.
#:
#: So the gate moves to the one place every withdrawal passes through. This is
#: NOT the "flag on the existing functions" the stash note below rejects: there
#: is no argument to get wrong, because the caller does not supply the answer -
#: the identity of the agent whose handler is running does. And it is ONE-WAY.
#: A hermit is diverted to his own stash; nothing here can ever route a villager
#: INTO it, so "a villager eating the hermit's winter stores" stays
#: unrepresentable, which was the property that note was protecting.
#:
#: A list rather than a single slot because handlers nest (`_h_craft` -> `_pay`,
#: a cleanup fired from inside another action's update), and pure transient
#: state: nothing here is saved, `SAVE_VERSION` is untouched, and an empty stack
#: means "no handler is running" - which is what every call from the director,
#: the scorer and the HUD sees, so none of their reads change.
_TILL_ACTOR: list[Any] = []

#: Depth of the deliberate colony-till exemption - see `colony_till`.
_TILL_OPEN: list[int] = []


def till_actor() -> Any | None:
    """The agent whose action handler is currently running, or None."""
    return _TILL_ACTOR[-1] if _TILL_ACTOR else None


class colony_till:
    """Context manager: this withdrawal really is from the COLONY'S store.

    THE ONE DELIBERATE EXCEPTION, WRITTEN DOWN RATHER THAN LEFT TO OMISSION.
    `_h_eat`'s last rung is a starving hermit walking into town to eat somebody
    else's dinner, and it is not a bug - `HERMIT_HOMESICK` is tuned so "the pull
    home loses to a really hungry man", and a hermit who starves beside his
    principles is a worse outcome than one who occasionally eats a meal he did
    not grow. It is reached only after his hands AND his stash are both empty.

    Anything that wants the colony's store while a hermit is acting has to say
    so here, in one line, at the call site. A future handler that forgets is
    routed to his stash and fails honestly; a future handler that MEANS it says
    `with colony_till():` and is visible to grep forever.

    Not a `contextlib.contextmanager`: this is on the frame path and a plain
    class with two methods costs no generator machinery.
    """

    __slots__ = ()

    def __enter__(self) -> "colony_till":
        _TILL_OPEN.append(1)
        return self

    def __exit__(self, *exc: Any) -> bool:
        if _TILL_OPEN:
            _TILL_OPEN.pop()
        return False


def _hermit_at_till() -> bool:
    """True when a hermit's handler is running and has not claimed the exception."""
    if _TILL_OPEN:
        return False
    ag = _TILL_ACTOR[-1] if _TILL_ACTOR else None
    return ag is not None and is_hermit(ag)


def stock_take(world: Any, res: str, qty: int) -> int:
    """Remove up to `qty`. Returns how much was actually taken.

    THE TILL, and the hermit gate is enforced here rather than at the call
    sites - see `_TILL_ACTOR` for the four routes that proved a per-handler
    gate cannot hold. A hermit withdrawing draws on his own camp store instead,
    whatever handler he is inside and however he got there; the colony's pile is
    not touched and cannot be.
    """
    if qty <= 0 or not res:
        return 0
    if _hermit_at_till():
        return hermit_stash_take(world, res, qty)
    sp = stockpile_of(world)
    try:
        if isinstance(sp, dict):
            have = int(sp.get(res, 0))
            take = min(have, int(qty))
            if take > 0:
                sp[res] = have - take
            return max(0, take)
        for name in ("take", "withdraw", "remove"):
            fn = getattr(sp, name, None)
            if callable(fn):
                return max(0, int(fn(res, int(qty)) or 0))
    except Exception:
        log.debug("stock_take failed", exc_info=True)
    return 0


# ------------------------------------------------------- the hermit's stash --
#: WHAT HE KEEPS AT HIS OWN CAMP, and the four functions below are the whole of
#: it. They are a deliberate mirror of `stockpile_of`/`stock_qty`/`stock_add`/
#: `stock_take` above rather than a mode on them, because the ONE property this
#: subsystem has to have is that the two books cannot meet: nothing in the
#: colony's half ever names `hermit_stash` and nothing here ever names
#: `stockpile`. A flag on the existing functions would be one wrong argument
#: away from a villager eating the hermit's winter stores.
#:
#: `World.reconcile` conserves PEOPLE, not resources, so there is no accounting
#: hazard in a second pile - the gate is about the ROLE. See the note in
#: `World._init_colony_state` for why the store lives on the world rather than
#: on the hut, and `_deposit_step` for what it replaced.
def hermit_stash_of(world: Any) -> dict[str, int]:
    """His camp store as a plain dict. Creates one if a world somehow lacks it.

    The fallback is not decoration. `_init_colony_state` puts this on both
    construction paths, but every other consumer in this file reaches worlds
    built by test harnesses and stub objects too, and a stash subsystem that
    raises AttributeError on the gather path would take the whole action down.
    """
    st = getattr(world, "hermit_stash", None)
    if not isinstance(st, dict):
        st = {}
        try:
            setattr(world, "hermit_stash", st)
        except Exception:
            pass
    return st


def hermit_stash_qty(world: Any, res: str) -> int:
    try:
        return max(0, int(hermit_stash_of(world).get(res, 0) or 0))
    except Exception:
        return 0


def hermit_stash_add(world: Any, res: str, qty: int) -> int:
    """Bank up to *qty*. Returns how much actually went in.

    Capped at HERMIT_STASH_CAP per resource, and the cap is what stops a hermit
    who outlives three villages from banking four thousand logs into a HUD row
    that then does not fit on the panel. Garbage is refused for the same reason
    `stock_add` refuses it: sweepings are not wealth, and letting them in here
    would put an `rb 0` on his row forever.
    """
    if qty <= 0 or not res or res == RES_GARBAGE:
        return 0
    if res not in ALL_RESOURCES:
        return 0
    st = hermit_stash_of(world)
    try:
        have = max(0, int(st.get(res, 0) or 0))
        room = max(0, int(HERMIT_STASH_CAP) - have)
        put = min(room, int(qty))
        if put > 0:
            st[res] = have + put
        return put
    except Exception:
        log.debug("hermit_stash_add failed", exc_info=True)
        return 0


def hermit_stash_take(world: Any, res: str, qty: int) -> int:
    """Draw up to *qty* out. Returns how much was actually taken."""
    if qty <= 0 or not res:
        return 0
    st = hermit_stash_of(world)
    try:
        have = max(0, int(st.get(res, 0) or 0))
        take = min(have, int(qty))
        if take > 0:
            st[res] = have - take
        return take
    except Exception:
        log.debug("hermit_stash_take failed", exc_info=True)
        return 0


def hermit_stash_food(world: Any) -> tuple[str | None, int]:
    """Best edible thing in his stash and how much of it - cooked wins.

    The stash twin of `food_in_store`, and the reason `_h_eat` can feed him
    without the walk into town that function's callers make.
    """
    ck = hermit_stash_qty(world, RES_COOKED)
    if ck > 0:
        return RES_COOKED, ck
    raw = hermit_stash_qty(world, RES_FOOD)
    if raw > 0:
        return RES_FOOD, raw
    return None, 0


def food_in_store(world: Any) -> tuple[str | None, int]:
    """Best available food resource and its quantity - cooked wins."""
    ck = stock_qty(world, RES_COOKED)
    if ck > 0:
        return RES_COOKED, ck
    raw = stock_qty(world, RES_FOOD)
    if raw > 0:
        return RES_FOOD, raw
    return None, 0


# ----------------------------------------------------------------- agents ---
def agents_of(world: Any) -> list[Any]:
    """Every agent on the roster, however the world happens to store them.

    World keeps them at ``world.population.agents``, which none of the old
    duck-typed names matched, so this returned [] on the real World. Everything
    downstream that counts people therefore saw a colony of zero: the build
    director never wanted a second hut (want_huts = max(1, 0)), and walls,
    watchtowers and totems - all gated on pop >= 4/5/8 - were unreachable for
    the entire life of the project. Check the real location first.
    """
    pop = getattr(world, "population", None)
    if pop is not None:
        roster = getattr(pop, "agents", None)
        if isinstance(roster, (list, tuple)):
            return list(roster)
        if roster is not None and hasattr(roster, "__iter__"):
            try:
                return list(roster)
            except Exception:
                pass
    for name in ("agents", "stickmen", "people"):
        v = getattr(world, name, None)
        if isinstance(v, (list, tuple)):
            return list(v)
        if v is not None and hasattr(v, "__iter__"):
            try:
                return list(v)
            except Exception:
                continue
    return []


def alive_agents(world: Any) -> list[Any]:
    return [a for a in agents_of(world) if getattr(a, "alive", True)]


def agent_by_id(world: Any, aid: Any) -> Any | None:
    if aid is None:
        return None
    fn = getattr(world, "agent_by_id", None)
    if callable(fn):
        try:
            return fn(aid)
        except Exception:
            pass
    try:
        aid_i = int(aid)
    except (TypeError, ValueError):
        return None
    for a in agents_of(world):
        if getattr(a, "id", None) == aid_i:
            return a
    return None


# ------------------------------------------------------------- structures ---
def structures_of(world: Any) -> StructureRegistry | None:
    reg = getattr(world, "structures", None)
    if isinstance(reg, StructureRegistry):
        return reg
    if reg is not None and hasattr(reg, "nearest") and hasattr(reg, "get"):
        return reg  # type: ignore[return-value]
    return None


def structure_by_id(world: Any, sid: Any) -> Structure | None:
    reg = structures_of(world)
    if reg is None:
        return None
    try:
        s = reg.get(sid)
    except Exception:
        return None
    return s if isinstance(s, Structure) else s


def nearest_structure(
    world: Any,
    kind: str | None,
    x: float,
    *,
    built_only: bool = True,
    predicate: Callable[[Any], bool] | None = None,
) -> Structure | None:
    reg = structures_of(world)
    if reg is None:
        return None
    try:
        return reg.nearest(kind, x, built_only=built_only, predicate=predicate)
    except TypeError:
        try:
            return reg.nearest(kind, x)
        except Exception:
            return None
    except Exception:
        return None


def colony_center(world: Any) -> float:
    """Where the colony is, in WORLD px. Never raises; always finite.

    Prefers the structure registry's own mean, then the living agents, then the
    middle of the map.

    The NaN sentinel is load-bearing, and it is a WORLD_W bug rather than a
    tidy-up. ``StructureRegistry.colony_center`` takes a *default* it returns
    when it has no buildings to average, and that default is ``RENDER_W * 0.5``.
    Called bare, an empty registry therefore answers "800" - a real, finite,
    entirely wrong number that this function used to accept and return. That was
    invisible while the world was 1600 px wide, because 800 was the middle of it
    and a colony had to be near the middle. On a 6400 px world a colony that has
    not yet finished its first building reports its centre as x=800 wherever it
    actually stands: measured here, four founders seated at x=1600 reported 800,
    which puts stage_bounds 800 px off, sites the first firepit 788 px from the
    people building it, and hands the camera the wrong place to look.

    Asking with a NaN default makes "I have nothing" distinguishable from "the
    mean happens to be 800", so the agents get their turn. A duck-typed registry
    that will not take the argument falls back to the bare call - it keeps
    today's behaviour rather than losing the registry entirely.
    """
    reg = structures_of(world)
    if reg is not None:
        c: Any = None
        try:
            c = reg.colony_center(float("nan"))
        except TypeError:
            try:
                c = reg.colony_center()
            except Exception:
                c = None
        except Exception:
            c = None
        if isinstance(c, (int, float)) and math.isfinite(float(c)):
            return float(c)
    ags = alive_agents(world)
    if ags:
        return float(sum(float(getattr(a, "x", 0.0)) for a in ags) / len(ags))
    # Nobody left to average: the middle of the WORLD, not of the frame. This
    # is the value stage_bounds/offstage_x fall back to on an empty roster, and
    # it is also what Camera.follow uses when it has nothing to follow, so the
    # two agree on where "nowhere in particular" is.
    return WORLD_W * 0.5


#: Structures deliberately placed at the colony's EDGE rather than inside it.
#: They have to be left out of any "where is the colony" average, because the
#: whole point of them is to sit a long way from that answer.
#:
#: ``hermit_hut`` is here for a sharper version of the same reason and one extra
#: one. A barricade folded into the mean merely moves it; the hermit's hut
#: folded into the mean moves it and then feeds back, because ``hermit_home``
#: below is *derived* from ``settlement_center`` plus a fixed standoff. Leave
#: the hut in and every director pass sites the camp a little further out
#: than the last, a runaway with a 760 px step. See also
#: ``structures.NON_SETTLEMENT_KINDS``, which carries the same exclusion for the
#: registry's own average (the camera's fallback, and stage_bounds').
OUTWORK_KINDS = ("barricade", "hermit_hut", "hermit_fire")


def settlement_center(world: Any) -> float:
    """Where the colony *lives*, with its own outworks left out of the average.

    :func:`colony_center` prefers ``StructureRegistry.colony_center``, a mean
    over every non-grave, non-ruined structure - barricades included. That is
    right for a settlement whose buildings sit together and wrong for anything
    that reasons about the colony's EDGE, because a barricade is deliberately
    about STAGE_HALF out; fold one into the mean and the answer moves by that
    distance divided by the number of buildings.

    Measured on the lane harness, colony at x=800 on an otherwise empty map: the
    first barricade goes in at x=4 - correct, 796 px out - the structure mean
    then drops from 800 to 4, and the SECOND barricade is sited 800 px right of
    *that*, landing on top of the settlement it was meant to defend while the
    right approach stays open. Two symmetric barricades cancel out again, which
    is why this only bites between the first and the second and would survive
    any test that staked both at once. The same drag pulls MINE_KEEP_OUT off
    centre, which is how a quarry opens inside the town it is keeping out of:
    colony at x=178 with barricades at 4 and 974 reports its centre as 489, and
    the keep-out ring then permits a dig 174 px from the actual settlement.

    :func:`colony_center` is deliberately NOT changed - render/camera.py and the
    rest of the migration are written against its documented behaviour - so this
    is the one to reach for when the question is "where do the people live".
    """
    reg = structures_of(world)
    xs: list[float] = []
    if reg is not None:
        try:
            for s in reg:
                kind = str(getattr(s, "kind", ""))
                if kind in OUTWORK_KINDS or kind == "grave":
                    continue
                if getattr(s, "is_ruined", False):
                    continue
                xs.append(float(s.x))
        except Exception:
            xs = []
    if xs:
        return float(sum(xs) / len(xs))
    ags = alive_agents(world)
    if ags:
        return float(sum(float(getattr(a, "x", 0.0)) for a in ags) / len(ags))
    return colony_center(world)


# ------------------------------------------------------------------ hermit --
def is_hermit(agent: Any) -> bool:
    """True for the colonist holding the hermit title. Never raises."""
    try:
        return str(getattr(agent, "role", "")) == ROLE_HERMIT
    except Exception:
        return False


def _hermit_base(world: Any) -> float:
    """The settlement the hermit is measuring himself away FROM.

    :func:`settlement_center` with one correction, and the correction is the
    whole reason this exists. That function prefers the structure mean and falls
    back to the mean of the LIVING AGENTS - and the hermit is a living agent. On
    a colony that has not put a building up yet, ``home = mean + D`` with the
    hermit's own x inside ``mean`` is a feedback loop: it settles rather than
    running away (``home = m + D*n/(n-1)`` for the other n-1 colonists' mean m),
    but it still over-throws the standoff by a third at n=4 and puts the camp
    past the far side of the budget the constants were picked for.

    So the fallback averages everyone EXCEPT the hermit. The structure branch is
    left to ``settlement_center`` verbatim - buildings do not walk, and having
    two averages of the same buildings that could disagree is worse than the
    duplication it would save.
    """
    reg = structures_of(world)
    n = 0
    if reg is not None:
        try:
            for s in reg:
                kind = str(getattr(s, "kind", ""))
                if kind in OUTWORK_KINDS or kind == "grave":
                    continue
                if getattr(s, "is_ruined", False):
                    continue
                n += 1
        except Exception:
            n = 0
    if n:
        return settlement_center(world)
    ags = [a for a in alive_agents(world) if not is_hermit(a)]
    if ags:
        return float(sum(float(getattr(a, "x", 0.0)) for a in ags) / len(ags))
    return settlement_center(world)


def _hermit_owned(world: Any, kind: str, built_only: bool) -> Any | None:
    """The hermit's one building of *kind*, or None. Never raises.

    There is at most one of each, because :func:`behavior._ensure_hermit_camp`
    only ever stakes one; if a mangled save carries several this returns the one
    with the lowest id, deterministically, rather than whichever the dict
    happened to yield first.

    Ruins are never returned. A burned-out hut is not shelter, is not a build
    target, and must not go on anchoring the camp - and it lingers as rubble for
    RUIN_LINGER (240 s) before ``purge_ruins`` takes it away, which is 240 s of
    a hermit who would otherwise be sleeping in a pile of ash.
    """
    reg = structures_of(world)
    if reg is None:
        return None
    best = None
    try:
        for s in reg:
            if str(getattr(s, "kind", "")) != kind:
                continue
            if getattr(s, "is_ruined", False):
                continue
            if built_only and not getattr(s, "built", False):
                continue
            sid = int(getattr(s, "id", 0) or 0)
            if best is None or sid < int(getattr(best, "id", 0) or 0):
                best = s
    except Exception:
        return None
    return best


def hermit_hut(world: Any, *, built_only: bool = False) -> Any | None:
    """The hermit's hut, or None. Never raises."""
    return _hermit_owned(world, "hermit_hut", built_only)


def hermit_fire(world: Any, *, built_only: bool = False) -> Any | None:
    """The hermit's own firepit, or None. Never raises.

    THIS IS ONE HALF OF THE MUTUAL GATE, and it is the half that needs a
    function rather than a kind check. The other half is free: a villager looks
    up ``nearest_structure(w, "firepit", ..)`` and a ``hermit_fire`` is not a
    ``firepit``, so no villager can reach it however close he is standing. But a
    hermit looking up "firepit" WOULD find the colony's, and geometry is not a
    gate - a hermit dragged into town by a wolf is standing next to it. So every
    warmth and cooking path a hermit can take is routed through here instead,
    and it can only ever return something he built.
    """
    return _hermit_owned(world, "hermit_fire", built_only)


def hermit_worksite(world: Any) -> Any | None:
    """The unfinished building at his camp he should be working on, or None.

    FIRE FIRST, THEN WALLS, and that order is the whole of this function. The
    fire is 4 wood against the hut's 12 and it is what replaced the deleted
    HERMIT_FIRE_DAMP safety valve, so the cheap warmth has to land before the
    expensive roof or a hermit appointed on the eve of a blizzard has neither.
    """
    fire = hermit_fire(world)
    if fire is not None and not getattr(fire, "built", False):
        return fire
    house = hermit_hut(world)
    if house is not None and not getattr(house, "built", False):
        return house
    return None


def fire_for(agent: Any, world: Any) -> Any | None:
    """The one fire *agent* is allowed to use, or None. THE MUTUAL GATE.

    Every warmth and cooking path in the sim resolves its fire through here or
    through :func:`fire_kind_for`, so the gate is one function rather than a
    rule six call sites are each trusted to remember. It is symmetrical and it
    is absolute:

      * a HERMIT gets ``hermit_fire`` and nothing else. Not the colony's, ever,
        at any distance, in any weather - see the deleted HERMIT_FIRE_DAMP note
        in constants.py for what this replaced and why the damp had to go.
      * ANYBODY ELSE gets the nearest built ``firepit`` and nothing else. That
        half is already true by construction (a ``hermit_fire`` is not a
        ``firepit``, so ``nearest_structure`` cannot return one), and it is
        written out anyway because "by construction" is exactly the argument
        that stops being true the day somebody passes a kinds tuple to that
        call.

    Never raises: a world with no fire of the relevant sort returns None and the
    caller fails its action, which is what it did before any of this.
    """
    if is_hermit(agent):
        return hermit_fire(world, built_only=True)
    try:
        ax = float(getattr(agent, "x", 0.0))
    except (TypeError, ValueError):
        ax = 0.0
    return nearest_structure(world, "firepit", ax, built_only=True)


#: The hermit's own buildings. Nothing the COLONY does may touch these - not
#: building them, not repairing them, not celebrating them, not counting them.
#: The census sites get that for free because these are separate kinds; the
#: sites that iterate over ALL structures do not, and they are the ones that
#: need this tuple.
HERMIT_KINDS: tuple[str, ...] = ("hermit_hut", "hermit_fire")


def is_hermit_built(st: Any) -> bool:
    """True for a building that belongs to the hermit rather than the colony."""
    try:
        return str(getattr(st, "kind", "")) in HERMIT_KINDS
    except Exception:
        return False


def colony_repairable(reg: Any, *, threshold: float = 0.9) -> list[Any]:
    """Damaged buildings THE COLONY is allowed to mend. Never raises.

    ``StructureRegistry.damaged`` walks every structure there is, so it hands
    back the hermit's hut along with the huts - and `_mk_repair` then picks
    ``min(..., abs(s.x - ax))`` over the result, which on a hut chewed by a
    wolf is a villager sent on a 600 px walk to mend a building he may not
    enter, for a man who did not ask. That is the same "the colony does his work
    for him" failure `_colony_sites` exists to prevent, arriving through a
    different door: `damaged()` is kind-agnostic where `count()` is not, so the
    separate-kind trick does not cover it and an explicit filter has to.

    Nobody mends his camp, including him - RepairStructure is one of the keys
    `_hermit_bias` zeroes. A hut that takes damage stays damaged and, if it
    is hit enough, falls down, and `_ensure_hermit_hut` stakes him another.
    Rebuilding is the loop this role already has and it is the one worth
    watching; a repair loop would be a second one that looks identical.
    """
    try:
        return [st for st in reg.damaged(threshold=threshold)
                if not is_hermit_built(st)]
    except Exception:
        return []


def fire_kind_for(agent: Any) -> str:
    """The only structure kind *agent* may warm at or cook on."""
    return "hermit_fire" if is_hermit(agent) else "firepit"


def _fire_ok(fire: Any, agent: Any) -> bool:
    """True if *fire* is a live fire this agent is permitted to use.

    The kind test is the load-bearing half. An action carries its fire as an id
    across ticks, and ids are handed out afresh as structures are purged and
    staked, so "the target I was given" and "a fire I am allowed to stand at"
    are two different questions and only this one is safe to act on.

    WHAT FALSE ACTUALLY DOES, because it has been described wrongly: an action
    handed the wrong fire is REDIRECTED, not refused. Every caller (`_h_warm`,
    `_h_cook` through :func:`_regate_fire`, and `_h_clean`'s burn phase) answers
    a False here by asking :func:`fire_for` for the one fire this agent may use,
    rewriting its own target to that, and PUTTING ITSELF BACK INTO ITS APPROACH
    PHASE so the agent walks to it. Only when `fire_for` itself comes back None -
    a hermit with no fire of his own, a villager with no hearth standing - does
    the action fail. So the gate is absolute about WHICH fire and silent about
    it: a hermit dragged into town by a wolf and re-scored onto WarmAtFire does
    not stand there failing, he turns round and walks home to his own.

    The walk is not decoration - see `_regate_fire` for the tick on seed 7 where
    it was missing and a redirect drew warmth off a fire 700 px away.
    """
    return (fire is not None
            and str(getattr(fire, "kind", "")) == fire_kind_for(agent)
            and not getattr(fire, "is_ruined", False)
            and bool(getattr(fire, "built", False)))


def _regate_fire(a: Any, ag: Any, w: Any) -> Any | None:
    """The fire this action may use, re-resolved through the gate every tick.

    ONE CALLER'S WORTH OF LOGIC THAT WAS COPIED INTO THREE, and the copies had
    drifted into a bug. Each read its fire out of ``a.target``, tested it with
    :func:`_fire_ok`, and on a failure swapped in :func:`fire_for`'s answer -
    but left ``a.phase`` alone. A redirect that lands while the phase is already
    ``warm`` or ``cook`` therefore skipped the walk entirely: the agent stood
    where he was and drew warmth, or cooked a meal, at a fire that could be
    seven hundred pixels away. Reachable two ways, both real - a structure id
    reused after a purge, and a villager handed the hermit's title while he was
    stood at the colony hearth (measured on seed 7, tick 34195).

    So the redirect is a redirect: it rewrites the target AND puts the action
    back into its approach phase, with a fresh walk clock in ``walk_t0`` so a
    swap late in a long action does not immediately time the walk out. The agent
    turns round and walks to the fire he is allowed to use.

    ``start`` and ``fetch`` are left alone: neither has arrived anywhere yet, so
    both fall into their own approach in the normal way.

    Returns None and sets ``a.failed`` when there is no fire of the right sort
    at all - a hermit whose camp has not been raised, a colony with no hearth.
    """
    fire = structure_by_id(w, a.target)
    if _fire_ok(fire, ag):
        return fire
    fire = fire_for(ag, w)              # the mutual gate - see `fire_for`
    if fire is None:
        a.failed = True
        return None
    a.target = fire.id
    if a.phase not in ("start", "fetch"):
        a.phase = "approach"
        a.data["walk_t0"] = float(getattr(a, "t", 0.0) or 0.0)
    return fire


def _walked_for(a: Any) -> float:
    """Seconds this action has spent in its CURRENT walk. See `_regate_fire`."""
    try:
        return float(getattr(a, "t", 0.0) or 0.0) - float(a.data.get("walk_t0", 0.0))
    except (TypeError, ValueError):
        return float(getattr(a, "t", 0.0) or 0.0)


def _stoke_wood(ag: Any, w: Any, want: int) -> int:
    """Wood for a fire: hands first, then HIS OWN STASH or the colony's store.

    A HERMIT NEVER TOUCHES THE COLONY STORE. He has no claim on it - the whole
    role is built on "what the hermit picks, the colony never sees" - and it is
    most of a kilometre away besides, so a hermit permitted to draw on it would
    walk into town every time his fire guttered, which is the commute this
    feature exists to end.

    WHAT HE DRAWS ON INSTEAD IS HIS OWN PILE, and that line is why the camp
    could move out to 530..1060 without killing him. HERMIT_FIRE_WARD only holds
    while the fire is BURNING, and at the old band the only fuel he had was
    whatever was in his hands, so every stoke meant a walk to timber and every
    walk to timber meant time outside the ward - the mechanism that made every
    band past 540 px measurably more lethal. Banking the surplus and spending it
    here means he needs FEWER walks rather than shorter ones, which is the one
    shape that pays at twice the distance. Measured over 16 seeds x 45 sim-min:
    the fire is alight 98.1% of the ticks a BUILT one stands, against 91.4% at
    half the range, and he is inside a LIT ward for 63.9% of his ticks.

    THE STASH ALONE WAS NOT ENOUGH AND THE FIRST MEASUREMENT SAID SO. With a
    pile he could draw on but no reason to fill it, his banked wood peaked at 5
    units and the alight fraction was 74.8% - WORSE than the old band, because
    he still only cut wood when something already wanted it and every one of
    those trips was now twice as long. HERMIT_WOODPILE is the other half: it
    gives him a reason to cut ahead of the need. This function is where that
    pile is spent.
    """
    got = 0
    if getattr(ag, "carrying", None) == RES_WOOD:
        got = min(int(want), int(getattr(ag, "carry_qty", 0) or 0))
        if got > 0:
            ag.carry_qty = int(ag.carry_qty) - got
            if ag.carry_qty <= 0:
                ag.carrying = None
    if got <= 0:
        got = (hermit_stash_take(w, RES_WOOD, int(want)) if is_hermit(ag)
               else stock_take(w, RES_WOOD, int(want)))
    return int(got)


def hermit_visit_budget(world: Any) -> float:
    """Seconds a visitor gets to reach the hermit. Never raises.

    The walk, at ``distance / WALK_SPEED``, multiplied by HERMIT_VISIT_PACE for
    the fact that nobody walks a hillside at WALK_SPEED, floored at
    HERMIT_VISIT_TIMEOUT so a camp that happens to be close still gets the old
    generous budget.

    DERIVED BECAUSE THE DISTANCE IS NOW A FRACTION OF THE MAP. A fixed budget is
    a distance limit wearing a clock's clothes, and once the camp moved to 3/4
    of the available run the old 260 s stopped covering the walk: 35 visits
    opened over 6 seeds and 1 was paid, the other 34 being men who walked
    3000-4000 px and turned round short of the door. Deriving it means the
    standoff can move again without this quietly switching the feature off.

    Both the visitor's own approach budget (`_h_converse`) and the director's
    watchdog (`behavior._tick_hermit_visit`) read it, so the two cannot disagree
    about when an appointment has expired - the watchdog allows a margin on top
    for the stay itself.
    """
    try:
        d = abs(float(hermit_home(world)) - float(colony_center(world)))
        if not math.isfinite(d):
            d = 0.0
    except Exception:
        d = 0.0
    walk = (d / float(WALK_SPEED)) * float(HERMIT_VISIT_PACE) if WALK_SPEED else 0.0
    return max(float(HERMIT_VISIT_TIMEOUT), walk)


def hermit_reach(world: Any, side: float | None = None) -> float:
    """How much run there IS for a camp, in px. Never raises, never negative.

    The distance from the settlement to the world edge on *side*, less
    HERMIT_EDGE_MARGIN - the room his hut, his fire and his roam need at the
    far wall. This is the denominator the standoff is a fraction OF, and the
    whole reason the standoff stopped being a pixel count: the user asked for
    "3/4 of the way across the map away from the colony", and 3/4 of 6400 px is
    not available to a colony seated mid-map, where the furthest edge is
    ~2900 px off. A fraction of what is actually there means the same thing on
    every world and is achievable on all of them.

    *side* ``None`` answers for the ROOMIER shoulder, which is the one a new
    camp is staked on ("pick the side with the most room"). ``+1``/``-1`` asks
    about that shoulder specifically, which is what the standing-hut band test
    needs - see `behavior._ensure_hermit_hut` for why it must not be allowed to
    switch shoulders under a hut that is already up.
    """
    try:
        base = float(_hermit_base(world))
        if not math.isfinite(base):
            return 0.0
    except Exception:
        return 0.0
    m = float(HERMIT_EDGE_MARGIN)
    right = (float(WORLD_W) - m) - base
    left = base - m
    if side is None:
        return max(0.0, right, left)
    return max(0.0, right if float(side) >= 0.0 else left)


def hermit_band(world: Any, side: float | None = None,
                stake: bool = False) -> tuple[float, float]:
    """``(lo, hi)`` - how far out the camp may be, in px. Never raises.

    ``HERMIT_STANDOFF_FRAC * reach .. reach``, and the top of the band is the
    edge rather than a second fraction because the ask was "3/4 OR MORE":
    three quarters is a floor, not a target. Where inside the band he actually
    lives is `behavior._camp_site`'s decision and it takes the highest safe
    ground, which on most maps lands well past the floor.

    *stake* insets the INNER wall by HERMIT_RESITE_SLACK, and only the inner
    wall. That asymmetry is not a shortcut - see HERMIT_STANDOFF_FRAC for the
    algebra: with the hut fixed and the settlement drifting, the hut's distance
    and the outer wall move together px for px, so the outer wall cannot be
    crossed by drift at all and there is nothing to reserve against there.

    THE SAME BAND APPLIES TO BOTH SHOULDERS when *side* is None, and that is
    deliberate. Deriving a cramped shoulder's band from its own small reach
    would let the fallback rungs of `_camp_site` stake a "hermit" 200 px from a
    colony seated near a wall - which is to say in town, the one thing the
    standoff exists to prevent. Asking the roomier shoulder instead means the
    cramped side simply has no columns in band, so it drops out of the search
    rather than offering a bad answer.
    """
    reach = hermit_reach(world, side)
    lo = float(HERMIT_STANDOFF_FRAC) * reach
    if stake:
        lo = min(lo + float(HERMIT_RESITE_SLACK), reach)
    return lo, reach


def hermit_home(world: Any) -> float:
    """Where the colony's hermit lives, in WORLD px. Never raises.

    ONCE THE HUT IS STANDING, THE HUT IS THE ANSWER. Everything below this
    paragraph describes how the camp is *sited in the first place*; from the
    moment there is a building on it, the building is the single source of truth
    about where the camp is, and the derivation stops running.

    That is what makes the camp stop moving. The derivation is anchored on
    ``_hermit_base``, which is a mean over the colony's buildings, so it drifts
    every time the colony puts a hut up - and a camp that drifts is a hermit who
    walks a little further from his own front door every ten minutes. It is also
    the answer to what succession does with the hut: the next hermit inherits
    it, because ``hermit_home`` keeps returning the same x whoever holds the
    title, so he walks out to the place the last one lived rather than founding a
    new camp somewhere else. The seeded offset only decides where the FIRST one
    goes, and where the next one goes if the hut has burned down and been
    cleared away.

    DERIVED, not stored, and that is a deliberate choice against adding a world
    field. A persisted anchor has to be initialised in ``_init_colony_state`` or
    it exists on the ``__init__`` path only and the whole subsystem switches
    itself off on a loaded world; it has to be re-sited when "New Landscape"
    picks the colony up and puts it somewhere else; and it is a second source of
    truth about where the colony is. Deriving it from ``_hermit_base`` plus a
    seeded offset gets all three for free: a loaded world computes the same
    answer as the one that saved it because ``world.seed`` round-trips, and a
    re-landscaped world re-sites the camp because ``randomise_terrain`` draws a
    new seed and moves the settlement.

    ``zlib.crc32``, never ``hash()``: str hashing is salted per process, so the
    same seed in two processes would put the hermit on opposite sides of the map
    and no two-process digest comparison could ever match.

    THE SIDE IS THE ROOMIER SHOULDER, not a coin flip, and that changed when
    the standoff became a fraction of the run (see `hermit_band`). A seeded
    side would put him on the cramped shoulder half the time, where three
    quarters of the available run is three quarters of very little - a colony
    founded at x=200 would get a "hermit" living 200 px away, which is to say
    in town. Which side he lives on stops being cosmetic the moment the
    distance is measured against the room, so it is measured too.
    """
    house = hermit_hut(world)
    if house is not None:
        try:
            hx = float(getattr(house, "x", 0.0))
            if math.isfinite(hx):
                return _clamp(hx, 0.0, float(WORLD_W))
        except (TypeError, ValueError):
            pass
    base = _hermit_base(world)
    try:
        seed = int(getattr(world, "seed", 0) or 0) & 0xFFFFFFFF
    except (TypeError, ValueError):
        seed = 0
    h = zlib.crc32(b"hermit:" + str(seed).encode("ascii"))
    frac = ((h >> 8) & 0xFFFF) / 65535.0
    # The STAKING band, not the standing one: a camp is staked inside the band
    # inset by HERMIT_RESITE_SLACK at the inner wall so the settlement can drift
    # under it without putting the hut out of band. `behavior._camp_site`
    # searches the same corridor, and `_ensure_hermit_hut` enforces the full
    # band with no inset.
    d_lo, d_hi = hermit_band(world, None, stake=True)
    dist = d_lo + frac * max(0.0, d_hi - d_lo)
    # THE SIDE IS NO LONGER THE CRC'S TO CHOOSE. It is the shoulder with the
    # most room, because the standoff is now a fraction of that room and the
    # seeded side would have halved it on any colony that is not dead centre.
    # The crc keeps the one job the terrain does not do for it - where in the
    # band a camp with no readable heightmap goes.
    side = 1.0 if hermit_reach(world, 1.0) >= hermit_reach(world, -1.0) else -1.0
    lo = float(HERMIT_EDGE_MARGIN)
    hi = float(WORLD_W) - float(HERMIT_EDGE_MARGIN)
    home = base + side * dist
    if not math.isfinite(home):
        home = float(WORLD_W) * 0.5
    return _clamp(home, lo, hi)


def stage_bounds(world: Any) -> tuple[float, float]:
    """``(lo, hi)`` - the slice of world the camera can be showing right now.

    The camera is RENDER_W wide and centred on the colony, so everything inside
    ``colony_center() +/- STAGE_HALF`` is on screen BY CONSTRUCTION. That lets
    sim answer "somewhere a viewer will actually see" without knowing a camera
    exists.

    Use this for every site that used to be ``uniform(24, RENDER_W - 24)``. Those
    were written when the world *was* the view; left as ``uniform(24, WORLD_W)``
    they put three quarters of every lightning strike, meteor and mudslide on
    empty hillside nobody is looking at, at four times the cost for the same
    visible event count. Keep the RATE, narrow the SITING.

    Clamped to the map, so a colony camped against x=0 still gets a real span.
    """
    c = colony_center(world)
    lo = _clamp(c - STAGE_HALF, 0.0, float(WORLD_W))
    hi = _clamp(c + STAGE_HALF, 0.0, float(WORLD_W))
    if hi - lo < 1.0:               # only reachable if WORLD_W is degenerate
        return 0.0, float(WORLD_W)
    return lo, hi


def offstage_x(world: Any, side: int, margin: float = 0.0) -> float:
    """An x on *side* (-1 left, +1 right) that is guaranteed off camera.

    Anything more than OFFSTAGE from the colony cannot be in frame, so this is
    where things arrive from and leave by: an animal walking in, a dragon
    spawning, a saucer parking between visits.

    It replaces "the edge of the map", which meant the same thing only while the
    map was one screen wide. On a 6400 px world the rim is up to 2600 px past
    the last thing anyone can see - ~76 s of walking at WALK_SPEED, invisible,
    while the creature holds a spawn slot the whole way.

    *margin* pushes further out again and widens the clamp to match, so callers
    that want a spawn point genuinely outside the map (dragons fly in) can ask
    for one without the clamp quietly pulling it back onto the land.
    """
    s = 1.0 if (isinstance(side, (int, float)) and float(side) >= 0.0) else -1.0
    try:
        m = float(margin)
        if not math.isfinite(m) or m < 0.0:
            m = 0.0
    except (TypeError, ValueError):
        m = 0.0
    return _clamp(colony_center(world) + s * (OFFSTAGE + m),
                  -m, float(WORLD_W) + m)


# ------------------------------------------------------------------ props ---
def props_of(world: Any) -> list[Any]:
    for name in ("props", "prop_registry", "scenery"):
        v = getattr(world, name, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            return list(v)
        for meth in ("all", "items", "values"):
            fn = getattr(v, meth, None)
            if callable(fn):
                try:
                    out = list(fn())
                    if out and isinstance(out[0], tuple):
                        out = [o[1] for o in out]
                    return out
                except Exception:
                    continue
        if hasattr(v, "__iter__"):
            try:
                return list(v)
            except Exception:
                continue
    return []


def prop_alive(prop: Any) -> bool:
    if prop is None:
        return False
    if getattr(prop, "removed", False) or getattr(prop, "dead", False):
        return False
    if not getattr(prop, "alive", True):
        return False
    if getattr(prop, "depleted", False):
        return False
    hp = getattr(prop, "hp", None)
    if isinstance(hp, (int, float)) and hp <= 0:
        return False
    return True


def _prop_kind(prop: Any) -> str:
    k = getattr(prop, "kind", None) or getattr(prop, "type", None) or ""
    return str(k).lower()


# --------------------------------------------------------- resource claims --
#: Seconds a harvest claim survives without renewal. An agent renews its claim
#: every tick it is walking to or working a prop, so a live claim never lapses;
#: this only governs how fast a claim frees after its owner stops renewing it
#: (died, fled, switched jobs), and is deliberately short so a dropped resource
#: is available again almost immediately.
CLAIM_TTL = 4.0


def _prop_claims(world: Any) -> dict[Any, tuple[int, float]]:
    """The world's transient prop->(*agent_id*, *expires*) claim table.

    Created on demand and never serialised - on load everyone re-claims what
    they resume working, exactly like the Hand tool's held-item sets."""
    c = getattr(world, "prop_claims", None)
    if not isinstance(c, dict):
        c = {}
        try:
            setattr(world, "prop_claims", c)
        except Exception:
            return {}
    return c


def claim_prop(world: Any, prop: Any, agent: Any, ttl: float = CLAIM_TTL) -> None:
    """Reserve *prop* for *agent* so no one else picks it as a harvest target."""
    if prop is None:
        return
    pid = getattr(prop, "id", None)
    aid = getattr(agent, "id", None)
    if pid is None or aid is None:
        return
    try:
        _prop_claims(world)[pid] = (int(aid), world_now(world) + max(0.5, float(ttl)))
    except Exception:
        pass


def release_claim(world: Any, pid: Any, agent: Any = None) -> None:
    """Drop a claim by prop id. With *agent*, only if that agent still holds it
    (so a later claimant's reservation is never yanked out by the first one)."""
    if pid is None:
        return
    c = _prop_claims(world)
    rec = c.get(pid)
    if rec is None:
        return
    if agent is None or rec[0] == getattr(agent, "id", None):
        c.pop(pid, None)


def prop_claimed_by_other(
    world: Any, prop: Any, claimant_id: Any, now: float | None = None
) -> bool:
    """True if a *different* agent holds an unexpired claim on *prop*."""
    pid = getattr(prop, "id", None)
    if pid is None:
        return False
    rec = _prop_claims(world).get(pid)
    if rec is None:
        return False
    holder, expire = rec
    if now is None:
        now = world_now(world)
    if expire <= now:
        return False
    return holder != claimant_id


def sweep_claims(world: Any) -> None:
    """Drop expired claims and claims on props that are gone. Cheap; the table
    never holds more than a claim per agent, but stale keys would otherwise
    accumulate for every prop ever harvested."""
    c = getattr(world, "prop_claims", None)
    if not isinstance(c, dict) or not c:
        return
    now = world_now(world)
    live = {getattr(p, "id", None) for p in props_of(world) if prop_alive(p)}
    for pid in [k for k, (_, exp) in c.items() if exp <= now or k not in live]:
        c.pop(pid, None)


def find_prop(
    world: Any,
    kinds: Iterable[str],
    x: float,
    *,
    max_dist: float = 720.0,
    exclude: Iterable[Any] = (),
    claimant: Any = None,
) -> Any | None:
    """Nearest living prop whose kind is in `kinds`.

    With *claimant* (an agent id), props another agent has an unexpired claim on
    are skipped, so two villagers never converge on the same tree and one is
    left miming a harvest. The same claim-aware call is used both to *score* a
    gather (is any prop free?) and to *target* one, so the two never disagree.
    """
    want = tuple(k.lower() for k in kinds)
    skip = {id(e) for e in exclude}
    now = world_now(world) if claimant is not None else 0.0
    best = None
    best_d = float("inf")
    for p in props_of(world):
        if id(p) in skip or not prop_alive(p):
            continue
        if _prop_kind(p) not in want:
            continue
        if claimant is not None and prop_claimed_by_other(world, p, claimant, now):
            continue
        try:
            d = abs(float(getattr(p, "x", 0.0)) - float(x))
        except (TypeError, ValueError):
            continue
        if d < best_d:
            best_d = d
            best = p
    if best is not None and best_d <= max_dist:
        return best
    return None


# ------------------------------------------------------------------ litter --
#: How long a cluster survey stays good. The scan is O(n log n) over at most
#: LITTER_MAX props, but `score_actions` runs it per agent per AI tick and the
#: cleanup handler asks again every time it re-targets, so it is cached exactly
#: the way `behavior._farm_feasible` caches its prop-list walk. Four seconds is
#: well under the time it takes anyone to walk to a pile, so nobody ever acts on
#: a survey that has gone meaningfully stale.
LITTER_SCAN_SEC = 4.0
#: Distinct piles reported. Three is enough for the colony's whole workforce to
#: spread out without the scan turning into a clustering algorithm.
LITTER_MAX_CLUSTERS = 3


def litter_positions(world: Any) -> list[float]:
    """Sorted x of every piece of litter lying about."""
    xs: list[float] = []
    for p in props_of(world):
        if _prop_kind(p) != "litter" or not prop_alive(p):
            continue
        try:
            xs.append(float(getattr(p, "x", 0.0)))
        except (TypeError, ValueError):
            continue
    xs.sort()
    return xs


def _scan_clusters(xs: list[float]) -> list[tuple[float, int]]:
    """``[(centre_x, count), ...]`` for the densest non-overlapping piles.

    Greedy over a sorted list with a sliding window of width ``2 *
    LITTER_CLUSTER_R``: for each start index walk forward while the span fits,
    keep the widest run, then strike those items out and look again. Linear per
    pass and at most LITTER_MAX_CLUSTERS passes over <= LITTER_MAX items, which
    is why this can afford to be exact rather than a bucketed approximation.

    "Dense" is a count, never a single item: a lone speck can never be a job,
    which is the behaviour the spec asked for in as many words.
    """
    out: list[tuple[float, int]] = []
    pool = list(xs)
    span = 2.0 * float(LITTER_CLUSTER_R)
    for _ in range(LITTER_MAX_CLUSTERS):
        n = len(pool)
        if n < LITTER_CLUSTER_MIN:
            break
        best_i = best_j = 0
        best_n = 0
        j = 0
        for i in range(n):
            if j < i:
                j = i
            while j + 1 < n and pool[j + 1] - pool[i] <= span:
                j += 1
            if (j - i + 1) > best_n:
                best_n, best_i, best_j = j - i + 1, i, j
        if best_n < LITTER_CLUSTER_MIN:
            break
        # The mean, not the window midpoint: it puts the sweeper where the mess
        # actually is when a run has a straggler at one end.
        chunk = pool[best_i:best_j + 1]
        out.append((sum(chunk) / len(chunk), best_n))
        del pool[best_i:best_j + 1]
    return out


def litter_clusters(world: Any) -> list[tuple[float, int]]:
    """Cached ``[(centre_x, count), ...]``, densest first. Never raises."""
    now = world_now(world)
    try:
        t = float(getattr(world, "_act_litter_t", -1e9))
        if 0.0 <= now - t < LITTER_SCAN_SEC:
            cached = getattr(world, "_act_litter", None)
            if isinstance(cached, list):
                return cached
    except Exception:
        pass
    try:
        val = _scan_clusters(litter_positions(world))
    except Exception:
        log.debug("litter cluster scan failed", exc_info=True)
        val = []
    try:
        setattr(world, "_act_litter", val)
        setattr(world, "_act_litter_t", now)
    except Exception:
        pass
    return val


def densest_litter(world: Any) -> tuple[float, int] | None:
    """The single worst pile, or None if nothing is dense enough to be a job."""
    cl = litter_clusters(world)
    return cl[0] if cl else None


def free_litter_cluster(world: Any, agent_id: Any = None) -> tuple[float, int] | None:
    """The densest pile nobody else is already sweeping, or ``None``.

    Shared by the scorer's action-maker and by the handler's ``start`` phase so
    the two cannot disagree. That matters more than it looks: ``choose_action``
    skips a maker that returns ``None`` and moves straight down its ranking,
    whereas an action that is created and then fails on its first update costs
    the villager a whole decision cycle standing still. Same rule in both places
    means a colony where every heap is taken just gets on with something else.
    """
    for cx, n in litter_clusters(world):
        if _cluster_taken(world, cx, agent_id):
            continue
        return (cx, n)
    return None


def prop_by_id(world: Any, pid: Any) -> Any | None:
    if pid is None:
        return None
    src = getattr(world, "props", None)
    fn = getattr(src, "get", None)
    if callable(fn):
        try:
            p = fn(pid)
            if p is not None:
                return p
        except Exception:
            pass
    for p in props_of(world):
        if getattr(p, "id", None) == pid:
            return p
    return None


def _kill_prop(world: Any, prop: Any) -> None:
    src = getattr(world, "props", None)
    for name in ("remove", "kill", "destroy"):
        fn = getattr(src, name, None)
        if callable(fn):
            try:
                fn(prop)
                return
            except Exception:
                break
    for attr, val in (("alive", False), ("removed", True), ("depleted", True)):
        if hasattr(prop, attr):
            try:
                setattr(prop, attr, val)
            except Exception:
                pass
    if not hasattr(prop, "alive") and not hasattr(prop, "removed"):
        try:
            prop.alive = False
        except Exception:
            pass


def hit_prop(world: Any, prop: Any, amount: float, default_yield: int) -> int:
    """Deal one work-hit to a prop. Returns resource units produced (often 0)."""
    if prop is None:
        return 0
    for name in ("chop", "mine", "harvest", "hit"):
        fn = getattr(prop, name, None)
        if callable(fn):
            try:
                out = fn(amount)
            except TypeError:
                try:
                    out = fn()
                except Exception:
                    return 0
            except Exception:
                return 0
            if isinstance(out, tuple) and out:
                out = out[0]
            if isinstance(out, bool):
                return default_yield if out else 0
            if isinstance(out, (int, float)):
                return max(0, int(out))
            return 0
    for attr in ("hp", "hits", "integrity", "amount", "qty"):
        if hasattr(prop, attr):
            try:
                v = float(getattr(prop, attr)) - float(amount)
            except (TypeError, ValueError):
                break
            try:
                setattr(prop, attr, max(0.0, v))
            except Exception:
                break
            if v <= 0.0:
                _kill_prop(world, prop)
                return default_yield
            return 0
    _kill_prop(world, prop)
    return default_yield


def spawn_sapling(world: Any, x: float) -> bool:
    """Ask props.py to plant a sapling at `x`. Queues it if there is no hook."""
    y = ground_y(world, x)
    src = getattr(world, "props", None)
    for name in ("plant_sapling", "add_sapling", "spawn_sapling"):
        fn = getattr(src, name, None) or getattr(world, name, None)
        if callable(fn):
            try:
                fn(float(x))
                return True
            except TypeError:
                try:
                    fn(float(x), float(y))
                    return True
                except Exception:
                    pass
            except Exception:
                pass
    for name in ("spawn", "create", "add"):
        fn = getattr(src, name, None)
        if callable(fn):
            try:
                fn("sapling", float(x), float(y))
                return True
            except Exception:
                continue
    q = getattr(world, "pending_props", None)
    if not isinstance(q, list):
        q = []
        try:
            setattr(world, "pending_props", q)
        except Exception:
            return False
    q.append({"kind": "sapling", "x": float(x), "y": float(y)})
    return True


# ----------------------------------------------------------------- hazards --
def hazards_of(world: Any) -> list[dict[str, Any]]:
    """Normalised danger zones: {kind, x, y, radius, water_y?}."""
    out: list[dict[str, Any]] = []
    raw: list[Any] = []
    for name in ("hazards", "danger_zones"):
        v = getattr(world, name, None)
        if callable(v):
            try:
                v = v()
            except Exception:
                v = None
        if isinstance(v, (list, tuple)):
            raw.extend(v)
    ev = getattr(world, "events", None)
    if ev is not None:
        v = getattr(ev, "hazards", None)
        if callable(v):
            try:
                v = v()
            except Exception:
                v = None
        if isinstance(v, (list, tuple)):
            raw.extend(v)
    for h in raw:
        d = _normalise_hazard(h)
        if d is not None:
            out.append(d)
    # Burning structures are hazards even if events.py never says so.
    reg = structures_of(world)
    if reg is not None:
        try:
            for s in reg:
                if getattr(s, "is_burning", False) and not getattr(s, "is_ruined", False):
                    out.append({
                        "kind": "fire", "x": float(s.x), "y": float(s.y),
                        "radius": 62.0,
                    })
        except Exception:
            pass
    return out


def _normalise_hazard(h: Any) -> dict[str, Any] | None:
    try:
        if isinstance(h, dict):
            get = h.get
        else:
            def get(k, default=None):  # type: ignore[misc]
                return getattr(h, k, default)
        kind = str(get("kind", None) or get("type", None) or "danger")
        water_y = get("water_y", None)
        x = get("x", None)
        radius = get("radius", None) or get("r", None)
        if x is None and water_y is None:
            return None
        d: dict[str, Any] = {
            "kind": kind,
            "x": float(x) if isinstance(x, (int, float)) else WORLD_W * 0.5,
            "y": float(get("y", RENDER_H * 0.75) or RENDER_H * 0.75),
            "radius": float(radius) if isinstance(radius, (int, float)) else 70.0,
        }
        if isinstance(water_y, (int, float)):
            d["water_y"] = float(water_y)
            d["kind"] = kind or "flood"
        return d
    except Exception:
        return None


# ------------------------------------------------------------ world events --
def chronicle(world: Any, text: str) -> None:
    if not text:
        return
    try:
        fn = getattr(world, "chronicle", None)
        if callable(fn):
            fn(text)
            return
        if fn is not None:
            add = getattr(fn, "add", None) or getattr(fn, "append", None)
            if callable(add):
                add(text)
                return
        fn = getattr(world, "log_event", None)
        if callable(fn):
            fn(text)
    except Exception:
        log.debug("chronicle failed: %s", text, exc_info=True)


def emit_speech(world: Any, agent: Any, symbol: str, ttl: float = 2.2) -> None:
    """Push a symbol speech bubble for the renderer to pick up."""
    try:
        fn = getattr(world, "emit_speech", None)
        if callable(fn):
            fn(agent, symbol)
            return
        q = getattr(world, "speech", None)
        if not isinstance(q, list):
            q = []
            setattr(world, "speech", q)
        q.append({
            "agent": int(getattr(agent, "id", 0)),
            "x": float(getattr(agent, "x", 0.0)),
            "y": float(getattr(agent, "y", 0.0)) - 26.0,
            "symbol": str(symbol),
            "t": 0.0,
            "ttl": float(ttl),
            "color": tuple(getattr(agent, "color", (230, 230, 230))),
        })
        if len(q) > 64:
            del q[: len(q) - 64]
    except Exception:
        log.debug("emit_speech failed", exc_info=True)


def celebrations_of(world: Any) -> list[dict[str, Any]]:
    q = getattr(world, "celebrations", None)
    return q if isinstance(q, list) else []


def push_celebration(world: Any, x: float, y: float, kind: str) -> None:
    try:
        q = getattr(world, "celebrations", None)
        if not isinstance(q, list):
            q = []
            setattr(world, "celebrations", q)
        q.append({"x": float(x), "y": float(y), "kind": str(kind),
                  "t": world_now(world)})
        if len(q) > 8:
            del q[: len(q) - 8]
    except Exception:
        log.debug("push_celebration failed", exc_info=True)


# ===========================================================================
#  Movement primitives
# ===========================================================================
def _face(agent: Any, d: float) -> None:
    if d > 0.01:
        agent.facing = 1
    elif d < -0.01:
        agent.facing = -1



def _plant(agent: Any, gy: float) -> None:
    """Put the agent on the ground and make the rest of its state agree.

    Writing y alone is not enough. entities.apply_physics decides an agent is
    airborne from on_ground/vy, so pinning y here while leaving vy alone let
    gravity accumulate against a body that never moved: measured vy reaching
    780 px/s over 0.87s of "falling" that covered 0.3px. The landing check then
    read that as a fatal impact, which is why every agent died of a fall while
    standing still on flat ground. Position and state must be asserted together.

    Standing on the ground also means standing on nothing else, so this drops
    any platform perch too - see :meth:`entities.Stickman.perch`.
    """
    agent.y = float(gy)
    agent.vy = 0.0
    agent.on_ground = True
    agent.fall_t = 0.0
    agent.perch_y = None


def _perch(agent: Any, y: float) -> None:
    """Stand the agent on a platform at *y*, off the ground, for this tick.

    The other half of _plant, for the handful of places that legitimately hold
    somebody above the terrain (a watchtower deck). entities owns y whenever
    the agent is off the ground, so an action that writes y up there without
    saying why is one entities reads as a body in free fall - which is exactly
    how manning a watchtower became a way to die. Must be re-asserted every
    tick: the perch is a lease (entities.PERCH_LEASE).
    """
    fn = getattr(agent, "perch", None)
    if callable(fn):
        fn(float(y))
        return
    # A stub agent from a test: assert the same state by hand, minus the lease.
    agent.y = float(y)
    agent.vy = 0.0
    agent.on_ground = True
    agent.fall_t = 0.0


def _clear_blocked(agent: Any) -> None:
    try:
        agent._blocked_t = 0.0
    except Exception:
        pass


def _note_blocked(agent: Any, dt: float) -> None:
    """Count time spent stuck against terrain; give the goal up past the limit.

    Refusing a ledge without this is how the cliff guard walled the map off:
    the agent stood at the lip until its action timed out 90 s later, having
    done nothing. There is no second path to try - the ground is a function of
    x, so "re-route" means picking a different goal - and failing the action is
    what makes world._tick_agents re-score on the next tick.
    """
    try:
        t = float(getattr(agent, "_blocked_t", 0.0) or 0.0) + max(0.0, float(dt))
        agent._blocked_t = t
        if t < BLOCKED_GIVE_UP:
            return
        agent._blocked_t = 0.0
        act = getattr(agent, "action", None)
        if act is not None and not getattr(act, "finished", True):
            act.failed = True
    except Exception:
        log.debug("_note_blocked failed", exc_info=True)


def _descend(agent: Any, world: Any, d: float, dt: float) -> float:
    """Ask the agent to climb down the face ahead. Returns px covered.

    entities owns the climb: it has the slip roll, the surface attachment and
    the descent rate limit already. Duplicating any of that here would give
    two subtly different cliffs depending on which module moved the agent.
    """
    fn = getattr(agent, "descend_step", None)
    if not callable(fn):
        return 0.0                      # a stub agent: fall back to refusing
    try:
        return float(fn(dt, getattr(world, "terrain", None),
                        direction=d, rng=rng_of(world)) or 0.0)
    except Exception:
        log.debug("descend_step failed", exc_info=True)
        return 0.0


def _ledge_step(
    agent: Any, world: Any, tx: float, d: float, dt: float, gy_now: float,
) -> float:
    """One tick of crossing a ledge: creep to the lip, then climb down it."""
    if not getattr(agent, "on_ground", True):
        return abs(tx - float(agent.x))      # already airborne; entities owns it

    # Judge the step we are about to take, not the walking one we gave up on.
    # A 1.1px walk step can reach over a lip that the 0.5px climb step does
    # not, and asking for a descent there finds no drop to descend, reports
    # failure, and parks the agent 0.7px short of the edge forever.
    nx = _clamp_x(float(agent.x) + d * CLIMB_SPEED * max(0.0, dt))
    gy_next = ground_y(world, nx)
    if nx != float(agent.x) and gy_next - gy_now <= STEP_DROP_MAX:
        # Still on the flat above the lip. Close the gap at climb speed so we
        # arrive at the edge under control rather than at a walking sprint.
        _clear_blocked(agent)
        agent.x = nx
        agent.vx = d * CLIMB_SPEED
        _plant(agent, gy_next)
        return abs(tx - float(agent.x))

    if _descend(agent, world, d, dt) > 0.0:
        _clear_blocked(agent)
        act = getattr(agent, "action", None)
        if act is not None:
            act.pose = "climb"
        return abs(tx - float(agent.x))

    # No descent happened. Either the agent slipped - in which case it is
    # airborne and entities owns it now - or the face is impassable and we
    # stand, then give up.
    if getattr(agent, "on_ground", True):
        agent.vx = 0.0
        _plant(agent, gy_now)
        _note_blocked(agent, dt)
    return abs(tx - float(agent.x))


def step_toward(
    agent: Any,
    world: Any,
    tx: float,
    dt: float,
    *,
    speed: float = WALK_SPEED,
    arrive: float = REACH,
) -> float:
    """Walk/climb one tick toward `tx`. Returns remaining |dx| afterwards."""
    try:
        tx = _clamp_x(tx)
        dx = tx - float(agent.x)
        dist = abs(dx)
        if dist <= arrive:
            agent.vx = 0.0
            _clear_blocked(agent)
            return dist
        d = 1.0 if dx > 0.0 else -1.0
        _face(agent, d)
        sl = slope_at(world, agent.x)
        climbing = abs(sl) > MAX_SLOPE_WALK
        spd = CLIMB_SPEED if climbing else float(speed)
        if not climbing and sl * d < 0.0:
            spd *= 0.78          # uphill is slower
        elif not climbing and sl * d > 0.0:
            spd *= 1.08          # downhill is quicker
        step = spd * max(0.0, dt)
        if step > dist:
            step = dist
        nx = _clamp_x(float(agent.x) + d * step)

        # Look before stepping. Walking is terrain-following: if the ground
        # ahead drops by more than a body height this is a ledge, not a slope,
        # and a calm stickman does not stroll off it. Letting gravity work
        # without this check turned every cliff into a meat grinder - 30 to 42
        # deaths per 260s, every single one a fall, even in clear weather.
        # Panicking agents DO go over, which is what makes a stampede lethal.
        gy_now = ground_y(world, agent.x)
        gy_next = ground_y(world, nx)
        # Probe a body-width ahead, not one footstep. A step is ~1px, and even
        # a sheer cliff only drops 4-11px across 1px of travel, so a per-step
        # test never fires: agents walked straight off a 146px drop measuring
        # it 8px at a time. LOOKAHEAD is what makes the edge visible.
        gy_ahead = ground_y(world, _clamp_x(float(agent.x) + d * CLIFF_LOOKAHEAD))
        drop_now = gy_next - gy_now
        panicking = float(getattr(agent, "panic", 0.0) or 0.0) > 0.0
        # Two ways to be at a ledge: one is coming up within a body width, or
        # this very step drops further than a walk can absorb. The second test
        # matters once agents are allowed *onto* faces - halfway down a cliff
        # the remaining drop is often under CLIFF_DROP, so only the lookahead
        # test would let go of the surface and hand the rest to gravity. That
        # was 17 of 24 fall deaths in a three-seed trace.
        if not panicking and (max(drop_now, gy_ahead - gy_now) > CLIFF_DROP
                              or drop_now > STEP_DROP_MAX):
            # Refusing a ledge outright kept everyone alive and cut the colony
            # from 3 completed structures per 300s to 1, because whole regions
            # - and the wood and stone in them - stopped being reachable.
            # Climb down instead: attached to the face, so it costs time and a
            # slip chance rather than a life.
            return _ledge_step(agent, world, tx, d, dt, gy_now)
        _clear_blocked(agent)

        agent.x = nx
        agent.vx = d * spd
        # Stick to the surface ONLY when actually on it. Snapping y whenever
        # vy happened to be ~0 teleported falling agents straight to the
        # ground before entities.apply_physics could see them airborne, so
        # gravity never accumulated and nobody could ever fall to their death.
        # Actions own x; entities owns y whenever the agent is off the ground.
        if float(agent.y) >= gy_next - GROUND_SNAP:
            _plant(agent, gy_next)
        return abs(tx - float(agent.x))
    except Exception:
        log.debug("step_toward failed", exc_info=True)
        return 0.0


def _halt(agent: Any) -> None:
    try:
        agent.vx = 0.0
    except Exception:
        pass


def _uphill_dir(world: Any, x: float, probe: float = 60.0) -> float:
    """+1 or -1, whichever direction climbs (lower y == higher ground)."""
    left = ground_y(world, x - probe)
    right = ground_y(world, x + probe)
    if left < right - 1.0:
        return -1.0
    if right < left - 1.0:
        return 1.0
    # Dead flat: break the tie toward the middle of the WORLD rather than of the
    # frame, so a panicked climb on featureless ground heads inland instead of
    # off the far rim of a map that is now four screens wide.
    return 1.0 if x < WORLD_W * 0.5 else -1.0


def _work_rate(agent: Any) -> float:
    role = str(getattr(agent, "role", "gatherer"))
    return {
        "builder": 1.35, "gatherer": 1.0, "elder": 0.8,
        "lookout": 0.9, "child": 0.5,
    }.get(role, 1.0)


def _need(agent: Any, name: str) -> float:
    return _clamp01(getattr(agent, name, 0.0))


def _adjust(agent: Any, name: str, delta: float) -> None:
    try:
        setattr(agent, name, _clamp01(float(getattr(agent, name, 0.0)) + delta))
    except Exception:
        pass


def _carry_add(agent: Any, action: "Action", res: str, qty: int) -> None:
    """Shoulder `qty` of `res`, stashing anything that does not fit the slot."""
    if qty <= 0:
        return
    cur = getattr(agent, "carrying", None)
    have = int(getattr(agent, "carry_qty", 0) or 0)
    if not cur or have <= 0:
        agent.carrying = res
        agent.carry_qty = int(qty)
    elif cur == res:
        agent.carry_qty = have + int(qty)
    else:
        stash = action.data.setdefault("stash", {})
        stash[res] = int(stash.get(res, 0)) + int(qty)


def _shed_litter(world: Any, x: float, n: int) -> int:
    """Put `n` pieces of carried rubbish back on the ground around `x`.

    Anything that takes garbage out of a villager's hands without burning it
    routes through here, so the stuff is *put down*, never destroyed. That keeps
    the density signal honest: an interrupted sweep leaves the mess where it was
    dropped and the job can be picked up again, rather than the colony quietly
    disposing of litter by getting scared of a wolf.

    The import is local for the same reason the vignette and combat rehydrates
    are: props.py must not appear on this module's import graph, since the rest
    of actions.py reaches props purely duck-typed.
    """
    try:
        from .props import drop_litter
    except Exception:
        return 0
    reg = getattr(world, "props", None)
    terr = getattr(world, "terrain", None)
    if reg is None or terr is None:
        return 0
    rng = getattr(world, "pyrng", None)
    now = world_now(world)
    count = max(0, min(int(n), CARRY_CAP))
    made = 0
    for i in range(count):
        try:
            # Fan them out a little so a dropped armful reads as spilled rubbish
            # rather than one speck standing in for eight.
            spread = (float(i) - (count - 1) * 0.5) * 7.0
            if drop_litter(reg, terr, float(x) + spread, rng, now) is not None:
                made += 1
        except Exception:
            break
    return made


def _drop_all(agent: Any, action: "Action", world: Any) -> int:
    """Move everything the agent holds into the stockpile. Returns units moved."""
    moved = 0
    res = getattr(agent, "carrying", None)
    qty = int(getattr(agent, "carry_qty", 0) or 0)
    if res == RES_GARBAGE and qty > 0:
        # `stock_add` refuses garbage, so without this the armful would simply
        # cease to exist the moment any other haul path unloaded it.
        _shed_litter(world, float(getattr(agent, "x", 0.0)), qty)
    elif res and qty > 0:
        moved += stock_add(world, res, qty)
    try:
        agent.carrying = None
        agent.carry_qty = 0
    except Exception:
        pass
    stash = action.data.get("stash")
    if isinstance(stash, dict):
        for r, q in list(stash.items()):
            try:
                moved += stock_add(world, str(r), int(q))
            except Exception:
                continue
        action.data["stash"] = {}
    return moved



def _load_is_useful(ag: Any, a: "Action", need: dict) -> bool:
    """True if anything the agent is carrying is something the site still wants.

    Guards the build state machine against walking a useless load to the site
    forever. Checks both the hand and any stashed load the action tracks.
    """
    if not need:
        return False
    carrying = getattr(ag, "carrying", None)
    if carrying in need and int(getattr(ag, "carry_qty", 0) or 0) > 0:
        return True
    stash = a.data.get("stash") or {}
    try:
        return any(need.get(res, 0) > 0 and int(qty) > 0
                   for res, qty in stash.items())
    except Exception:
        return False

def _has_load(agent: Any, action: "Action") -> bool:
    if int(getattr(agent, "carry_qty", 0) or 0) > 0 and getattr(agent, "carrying", None):
        return True
    stash = action.data.get("stash")
    return bool(isinstance(stash, dict) and any(int(v) > 0 for v in stash.values()))


def _deposit_step(action: "Action", agent: Any, world: Any, dt: float) -> bool:
    """Walk to the stockpile and unload. Returns True when finished."""
    if not _has_load(agent, action):
        return True
    if is_hermit(agent):
        # A hermit has no stockpile and no reason to walk to the colony's. This
        # is the single line that stops him commuting: every gather ends in a
        # deliver phase that routes through here, so without it a hermit who
        # forages one bush walks the full standoff back into the crowd, drops
        # six berries and walks out again - which is a colonist with a long
        # commute, not a hermit.
        #
        # Food stays in his hands. That is his larder, and it is also his
        # starvation guard: `Eat` scores `hunger**2` on a held load and is
        # forced to 1.0 past 0.85, so a hermit with anything edible on him
        # cannot starve, and one with nothing still has the walk to the store as
        # a last resort (see HERMIT_HOMESICK - the pull home loses to a really
        # hungry man).
        #
        # EVERYTHING ELSE USED TO BE DESTROYED HERE AND NOW GOES IN HIS STASH.
        # The line this replaces read "what the hermit picks, the colony never
        # sees", which was true of the colony and, it turned out, also true of
        # the hermit: the surplus was deleted at the bottom of this branch, so a
        # man who cut four logs for a fire that wanted one lost three of them.
        # He now banks them at his camp (`hermit_stash_add`, and see the note on
        # those functions), draws on them to build, to stoke and to eat, and his
        # successor inherits the pile. The colony still never sees it - the two
        # stores are separate dicts touched by disjoint sets of functions - so
        # the role's promise is intact and only the waste is gone.
        #
        # There is no book this breaks either way: World.reconcile conserves
        # PEOPLE, not resources, and both stores are counters rather than
        # inventories of objects.
        # ...UNLESS HIS OWN HUT IS SHORT OF IT, in which case THIS is his
        # delivery. The whole build loop for the hermit lives inside the gather
        # action rather than being handed to `BuildStructure`, and that is not an
        # optimisation, it is the only shape that works.
        #
        # The obvious design - keep the load, let BuildStructure walk it to the
        # site - was written, measured and thrown away. A hermit's `carrying`
        # slot is almost always full of BERRIES: `_hermit_bias` only zeroes
        # ForageBerries at CARRY_CAP, so his steady state is a man holding eight
        # to eleven food, which is his larder and his starvation guard and must
        # not be dropped. `_carry_add` puts a second resource in the ACTION's
        # ``stash``, and a stash does not outlive the action that made it - so
        # every log he cut while holding food was destroyed the moment the gather
        # ended, and on seed 7 he sat at stage 0 with an empty frame for the
        # entire 45 minutes. Delivering here, inside the action that still owns
        # the stash, is what lets him carry a meal in one hand and a log in the
        # other.
        # `hermit_worksite`, NOT `hermit_hut`, AND THE DIFFERENCE IS A
        # DEADLOCK. This delivery and `behavior._mk_build`'s target have to name
        # the same building, because the hermit's whole build loop is the two of
        # them in a ring: the gather delivers the wood, the build spends it, and
        # `_hermit_bias` only scores BuildStructure once the frame is stocked.
        # Point them at different frames and the ring opens - measured on seed 7
        # with this line still saying `hermit_hut`: he cut wood for 45 minutes
        # (396 GatherWood samples), delivered 6 of it into a HOUSE nobody was
        # building, while `_mk_build` sat pointed at a FIRE that never received
        # a single log, and scored BuildStructure exactly zero times in the
        # whole run. Both frames finished the run at stage 0. One accessor for
        # both callers is what makes that unrepresentable.
        site = hermit_worksite(world)
        want: dict[str, int] = {}
        if site is not None:
            try:
                want = site.total_remaining_cost()
            except Exception:
                want = {}
        if want and _load_is_useful(agent, action, want):
            action.pose = "carry"
            rem = step_toward(agent, world, float(site.x), dt,
                              speed=WALK_SPEED * 0.92)
            if rem > REACH * 1.6 and action.t <= 120.0:
                return False
            res = getattr(agent, "carrying", None)
            qty = int(getattr(agent, "carry_qty", 0) or 0)
            if res and qty > 0:
                took = site.deliver(str(res), qty)
                try:
                    agent.carry_qty = qty - took
                    if agent.carry_qty <= 0:
                        agent.carrying = None
                except Exception:
                    pass
            stash = action.data.get("stash")
            if isinstance(stash, dict):
                for r, q in list(stash.items()):
                    took = site.deliver(str(r), int(q))
                    stash[r] = int(q) - took
                    if stash[r] <= 0:
                        stash.pop(r, None)

        # AND WHAT IS LEFT GOES IN HIS OWN STASH, WHICH IS THE NEW PART. Every
        # line above this one is what the role always did; everything the
        # delivery did not want used to be DELETED here, and the paragraph at
        # the top of this branch said so in as many words. That was honest about
        # the cost and wrong about the man: he could cut four logs, put one on
        # his fire and lose three, and a hermit who cannot keep what he gathers
        # has no reason to gather anything.
        #
        # `hermit_stash_add` is his camp store and only his - see the note above
        # it, and `World._init_colony_state` for why it lives on the world
        # rather than on a hut he does not always have. The colony's stockpile is
        # not touched on this path and never was.
        stash = action.data.get("stash")
        if isinstance(stash, dict):
            for r, q in list(stash.items()):
                hermit_stash_add(world, str(r), int(q))
        action.data["stash"] = {}

        res = getattr(agent, "carrying", None)
        qty = int(getattr(agent, "carry_qty", 0) or 0)
        try:
            if qty > 0 and res in (RES_FOOD, RES_COOKED):
                # HIS LARDER, AND THE STARVATION GUARD LIVES INSIDE IT. Food in
                # his hands is why he cannot starve: `_score_wants` forces Eat to
                # 1.0 past 0.85 hunger only for a man who is `holding_food`, so
                # emptying his hands into the stash would have deleted that guard
                # while looking like a tidy-up. So a hermit KEEPS HIS FOOD, and
                # only an overflow past CARRY_CAP is banked.
                #
                # THIS OVERFLOW HAS NEVER FIRED, and the honest reading is that
                # this is a CLAMP rather than a mechanism. There used to be a
                # HERMIT_LARDER_HAND constant here carrying forty lines about
                # how much he banks; it was 8, it was wrapped in
                # ``max(.., CARRY_CAP)`` with CARRY_CAP also 8, so it could not
                # take any other value, and the branch under it was dead. It is
                # dead for a reason that is worth writing down instead: food
                # arrives in fours (BERRY_YIELD 4, FARM_HARVEST_FOOD 4) and
                # `_hermit_bias` stops him foraging at ``carry_qty >=
                # CARRY_CAP``, so his food hands go 0, 4, 8 and stop exactly ON
                # the cap, never over it. Instrumented over 16 seeds x 45
                # sim-min: the highest food carry ever seen is 8 and this branch
                # never ran once.
                #
                # It is kept, one line, because the ceiling is not guaranteed by
                # anything but that arithmetic - change BERRY_YIELD to 3 and a
                # hermit walks around with 9 - and an overflow left in his hands
                # would never reach the stash at all. Food still gets into the
                # pile in quantity (30.1 units at the final tick on average, on
                # 15 seeds of 16): it goes by the MIXED-CARRY path a few lines
                # up, where berries picked while he is already holding wood land
                # in ``action.data["stash"]`` and are banked wholesale.
                if qty > CARRY_CAP:
                    banked = hermit_stash_add(world, str(res), qty - CARRY_CAP)
                    if banked > 0:
                        agent.carry_qty = qty - banked
            elif qty > 0 and res:
                # Wood, stone, fibre: all of it into the store, and the HANDS ARE
                # CLEARED WHETHER OR NOT IT FITS. Leaving an unbanked remainder in
                # them would resurrect the exact bug the old comment here warned
                # about - "a hermit left holding wood forever could never pick up
                # food again" - the moment the stash hit HERMIT_STASH_CAP.
                hermit_stash_add(world, str(res), qty)
                agent.carrying = None
                agent.carry_qty = 0
            else:
                agent.carrying = None
                agent.carry_qty = 0
        except Exception:
            log.debug("hermit stash deposit failed", exc_info=True)
        _halt(agent)
        return True
    sp = nearest_structure(world, "stockpile", agent.x, built_only=True)
    tx = sp.x if sp is not None else colony_center(world)
    action.pose = "carry"
    rem = step_toward(agent, world, tx, dt, speed=WALK_SPEED * 0.92)
    if rem <= REACH:
        _drop_all(agent, action, world)
        _halt(agent)
        return True
    if action.t > 90.0:            # never get stuck hauling forever
        _drop_all(agent, action, world)
        return True
    return False


# ===========================================================================
#  Action
# ===========================================================================
@dataclass
class Action:
    """A resumable behaviour. `kind` selects the handler, everything else is
    the machine's persisted state."""

    kind: str = "Wander"
    pose: str = "idle"
    target: Any = None            # structure id / agent id / prop id / x
    phase: str = "start"
    t: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    failed: bool = False

    # ------------------------------------------------------------------ run --
    def update(self, agent: Any, world: Any, dt: float) -> None:
        """Advance the machine by `dt`. Never raises - this is a frame path."""
        if self.done or self.failed:
            return
        try:
            self.t += float(dt)
        except (TypeError, ValueError):
            dt = 0.0
        handler = _HANDLERS.get(self.kind)
        if handler is None:
            log.debug("unknown action kind %r", self.kind)
            self.failed = True
            return
        # WHO IS AT THE TILL. Every handler runs through this one line, so this
        # is where the acting agent is published for `stock_take`/`stock_add` to
        # gate on - see `_TILL_ACTOR`. try/finally, and a stack rather than a
        # slot, because a handler that raises must not leave a stale actor
        # standing at the till for the next agent in the loop.
        _TILL_ACTOR.append(agent)
        try:
            handler(self, agent, world, float(dt))
        except Exception:
            log.warning("action %s/%s failed", self.kind, self.phase, exc_info=True)
            self.failed = True
            try:
                _halt(agent)
            except Exception:
                pass
        finally:
            _TILL_ACTOR.pop()

    def abandon(self, agent: Any, world: Any) -> None:
        """Release anything held (hut slot, tower slot) before being replaced."""
        cleanup = _CLEANUP.get(self.kind)
        if cleanup is not None:
            # Cleanups move goods too - `_c_craft` refunds a part-paid craft
            # straight into `stock_add` - and they are NOT reached through
            # `update`, so without this the refund half of the gate would be
            # blind on the one path that uses it most. Same stack, same reason.
            _TILL_ACTOR.append(agent)
            try:
                cleanup(self, agent, world)
            except Exception:
                log.debug("cleanup for %s failed", self.kind, exc_info=True)
            finally:
                _TILL_ACTOR.pop()
        self.done = True

    @property
    def finished(self) -> bool:
        return bool(self.done or self.failed)

    # ------------------------------------------------------------ serialise --
    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "pose": str(self.pose),
            "target": _safe(self.target),
            "phase": str(self.phase),
            "t": float(self.t),
            "data": _safe(self.data) or {},
            "done": bool(self.done),
            "failed": bool(self.failed),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Action":
        if not isinstance(d, dict):
            d = {}
        kind = d.get("kind")
        if not isinstance(kind, str) or kind not in _HANDLERS:
            kind = "Wander"
        pose = d.get("pose")
        data = d.get("data")
        try:
            t = float(d.get("t", 0.0))
            if not math.isfinite(t):
                t = 0.0
        except (TypeError, ValueError):
            t = 0.0
        return cls(
            kind=kind,
            pose=pose if isinstance(pose, str) and pose else "idle",
            target=d.get("target"),
            phase=str(d.get("phase", "start")) or "start",
            t=t,
            data=dict(data) if isinstance(data, dict) else {},
            done=bool(d.get("done", False)),
            failed=bool(d.get("failed", False)),
        )

    def __repr__(self) -> str:  # pragma: no cover - debug sugar
        flag = "done" if self.done else ("failed" if self.failed else self.phase)
        return f"<{self.kind}:{flag} t={self.t:.1f} target={self.target!r}>"


def _safe(v: Any, depth: int = 0) -> Any:
    if depth > 4:
        return None
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else 0.0
    if isinstance(v, (list, tuple)):
        return [_safe(x, depth + 1) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe(x, depth + 1) for k, x in v.items()}
    return None


def make_action(kind: str, **kw: Any) -> Action:
    """Build an Action, folding unknown kwargs into `data`."""
    a = Action(kind=kind)
    for field_name in ("pose", "target", "phase"):
        if field_name in kw:
            setattr(a, field_name, kw.pop(field_name))
    data = kw.pop("data", None)
    if isinstance(data, dict):
        a.data.update(data)
    a.data.update(kw)
    return a


def action_from_dict(d: Any) -> Any:
    """Rehydrate a saved action, honouring the vignette engine's own kind.

    ``Action.from_dict`` coerces any unknown ``kind`` to ``Wander``, which would
    quietly turn a saved cosmetic vignette into a wander on every reload. The
    vignette module owns a parallel state machine with the same interface, so
    give it first refusal; fall back to a real :class:`Action` otherwise. The
    import is local to keep ``vignettes`` off actions.py's import graph (it
    imports *from* here)."""
    try:
        from .vignettes import vignette_action_from_dict
        v = vignette_action_from_dict(d)
        if v is not None:
            return v
    except Exception:
        log.debug("vignette rehydrate unavailable", exc_info=True)
    # combat_actions registers its four kinds into _HANDLERS as an import side
    # effect, and nothing on the cold-start path imports it - behaviour only
    # pulls it in lazily the first time it scores a threat. Until then
    # Action.from_dict sees CraftSpear/FightAnimal/... as unknown kinds and
    # silently rewrites them to Wander, so a save taken mid-hunt reloaded with
    # the hunter strolling away from the wolf (and, since the head labels read
    # the action, announcing "wandering" while doing it). Importing it here
    # rather than at module scope keeps the dependency one-way: combat_actions
    # imports this module, so the reverse cannot be a top-level import.
    try:
        from . import combat_actions          # noqa: F401  (registers handlers)
    except Exception:
        log.debug("combat actions unavailable for rehydrate", exc_info=True)
    return Action.from_dict(d if isinstance(d, dict) else {})


# ===========================================================================
#  Handlers
# ===========================================================================
def _h_wander(a: Action, ag: Any, w: Any, dt: float) -> None:
    if a.phase == "start":
        rng = rng_of(w)
        span = rng.uniform(45.0, 230.0) * (1.0 if rng.random() < 0.5 else -1.0)
        if is_hermit(ag):
            # An ordinary colonist wanders FROM WHERE HE STANDS, which is why
            # the colony clusters: everyone's jobs are in the same place, so
            # everyone's random walk starts there. A hermit wanders around HIS
            # place instead. That one substitution is the whole of "hangs around
            # their own area": nothing else in the sim positions an idle agent,
            # so nothing else can hold him anywhere.
            #
            # THE SAME TWO DRAWS, deliberately. Re-using `span` rather than
            # asking the rng for a fresh offset keeps the stream phase identical
            # to a run with no hermit in it, so the only thing that moves is the
            # hermit. It is rescaled to HERMIT_ROAM, which is the radius the
            # camera budget in constants.py was picked against.
            a.data["tx"] = _clamp_x(hermit_home(w) + span * (HERMIT_ROAM / 230.0))
        else:
            a.data["tx"] = _clamp_x(float(ag.x) + span)
        a.data["pause"] = rng.uniform(1.2, 4.5)
        a.data["pt"] = 0.0
        a.phase = "walk"
    if a.phase == "walk":
        a.pose = "walk"
        rem = step_toward(ag, w, float(a.data.get("tx", ag.x)), dt,
                          speed=WALK_SPEED * 0.72)
        if rem <= REACH or a.t > 30.0:
            a.phase = "pause"
        return
    a.pose = "idle"
    _halt(ag)
    a.data["pt"] = float(a.data.get("pt", 0.0)) + dt
    if a.data["pt"] >= float(a.data.get("pause", 2.0)):
        a.done = True


_GATHER_SPEC: dict[str, dict[str, Any]] = {
    "GatherWood": {
        "kinds": _TREE_KINDS, "res": RES_WOOD, "pose": "chop",
        "hit_time": 0.55, "dmg": 1.0, "yield": CHOP_YIELD, "verb": "felled a tree",
    },
    "GatherStone": {
        "kinds": _ROCK_KINDS, "res": RES_STONE, "pose": "mine",
        "hit_time": 0.70, "dmg": 1.0, "yield": STONE_YIELD, "verb": "broke open a rock",
    },
    "ForageBerries": {
        "kinds": _BUSH_KINDS, "res": RES_FOOD, "pose": "forage",
        "hit_time": 0.90, "dmg": 1.0, "yield": BERRY_YIELD, "verb": "stripped a bush",
        # bushes are also where the colony's cordage comes from
        "bonus_res": RES_FIBRE, "bonus_ratio": 0.5,
    },
}


def _h_gather(a: Action, ag: Any, w: Any, dt: float) -> None:
    spec = _GATHER_SPEC.get(a.kind)
    if spec is None:
        a.failed = True
        return

    aid = getattr(ag, "id", None)

    if a.phase == "start":
        # A HERMIT FELLS AT HIS OWN CAMP, not wherever he happens to be standing.
        # `find_prop` takes the nearest to the x it is given, so handing it his
        # own position makes each tree a stepping stone to the next one: measured
        # on seed 7, he walked from his camp at 4133 out to 5469 inside four
        # minutes, ended up level with the settlement, and was by then further
        # from his own building site than `_h_build` will walk. Anchoring the
        # search on the camp bounds the whole loop instead. HERMIT_FELL_REACH,
        # and `_hermit_bias` checks the same thing before it scores this at all.
        origin, reach = float(ag.x), 720.0
        if a.kind == "GatherWood" and is_hermit(ag):
            origin, reach = hermit_home(w), float(HERMIT_FELL_REACH)
        p = find_prop(w, spec["kinds"], origin, max_dist=reach, claimant=aid)
        if p is None:
            a.failed = True
            return
        a.target = getattr(p, "id", None)
        a.data["px"] = float(getattr(p, "x", ag.x))
        a.data["hits"] = 0
        a.data["hit_t"] = 0.0
        claim_prop(w, p, ag)          # this prop is mine; no one else target it
        a.phase = "approach"

    if a.phase in ("approach", "work"):
        p = prop_by_id(w, a.target) if a.target is not None else None
        if p is None:
            # id lookup can fail if props.py has no id field - fall back to
            # position, and give up gracefully if the prop is really gone.
            p = find_prop(w, spec["kinds"], float(a.data.get("px", ag.x)),
                          max_dist=40.0, claimant=aid)
        if p is None or not prop_alive(p):
            release_claim(w, a.target, ag)
            if _has_load(ag, a):
                a.phase = "deliver"
            else:
                a.failed = True
            return
        a.data["px"] = float(getattr(p, "x", a.data.get("px", ag.x)))
        # Renew every tick we are actively pursuing it, so the claim only lapses
        # once we walk away for good.
        claim_prop(w, p, ag)

    if a.phase == "approach":
        a.pose = "walk"
        rem = step_toward(ag, w, float(a.data["px"]), dt, arrive=REACH)
        if rem <= REACH:
            a.phase = "work"
            a.data["hit_t"] = 0.0
        elif a.t > 75.0:
            a.failed = True
        return

    if a.phase == "work":
        a.pose = str(spec["pose"])
        _halt(ag)
        _face(ag, float(a.data["px"]) - float(ag.x))
        a.data["hit_t"] = float(a.data.get("hit_t", 0.0)) + dt
        if a.data["hit_t"] < float(spec["hit_time"]) / max(0.3, _work_rate(ag)):
            return
        a.data["hit_t"] = 0.0
        a.data["hits"] = int(a.data.get("hits", 0)) + 1
        p = prop_by_id(w, a.target) or find_prop(
            w, spec["kinds"], float(a.data["px"]), max_dist=40.0, claimant=aid)
        got = hit_prop(w, p, float(spec["dmg"]), int(spec["yield"]))
        if a.data["hits"] > 24 and got <= 0:
            # Something odd about this prop - take the yield and move on.
            _kill_prop(w, p)
            got = int(spec["yield"])
        if got > 0:
            _carry_add(ag, a, str(spec["res"]), got)
            bonus_res = spec.get("bonus_res")
            if bonus_res:
                bonus = max(1, int(got * float(spec.get("bonus_ratio", 0.5))))
                _carry_add(ag, a, str(bonus_res), bonus)
            _adjust(ag, "fatigue", 0.02)
            if not prop_alive(p):
                chronicle(w, f"{getattr(ag, 'name', 'Someone')} {spec['verb']}.")
                release_claim(w, a.target, ag)   # exhausted: free it for others
            if int(getattr(ag, "carry_qty", 0) or 0) >= CARRY_CAP or not prop_alive(p):
                release_claim(w, a.target, ag)   # done here, hands full
                a.phase = "deliver"
        return

    if a.phase == "deliver":
        if _deposit_step(a, ag, w, dt):
            a.done = True
        return

    a.phase = "start"


def _c_gather(a: Action, ag: Any, w: Any) -> None:
    """Free the claimed prop whenever a gather is cut short (abandoned for an
    emergency, superseded, or killed off). Without this a villager who bolts
    from a wolf mid-chop would keep a tree reserved until the claim timed out."""
    try:
        release_claim(w, a.target, ag)
    except Exception:
        pass


def _h_haul(a: Action, ag: Any, w: Any, dt: float) -> None:
    if not _has_load(ag, a):
        a.done = True
        return
    a.phase = "deliver"
    if _deposit_step(a, ag, w, dt):
        a.done = True


def _h_build(a: Action, ag: Any, w: Any, dt: float) -> None:
    s = structure_by_id(w, a.target)
    if s is None and a.target is None:
        reg = structures_of(w)
        s = reg.find_incomplete(None, float(ag.x)) if reg is not None else None
        if s is not None:
            a.target = s.id
    if s is None or getattr(s, "is_ruined", False):
        a.failed = True
        return
    if getattr(s, "built", False):
        a.done = True
        return

    need = s.missing_for_stage()
    carrying = getattr(ag, "carrying", None)
    carry_qty = int(getattr(ag, "carry_qty", 0) or 0)

    if not need:
        a.phase = "work"
    elif carrying in need and carry_qty > 0:
        a.phase = "approach"
    elif _has_load(ag, a) and _load_is_useful(ag, a, need):
        a.phase = "approach"      # we are carrying something the site wants
    else:
        # Carrying a surplus the site has no use for. Routing to "approach"
        # here (as this did unconditionally) deadlocks: the builder walks to
        # the site, cannot deliver, and the haul action wins the next re-score,
        # so the pair alternate every tick and the agent paces on the spot.
        # Go and fetch what is actually needed instead; the surplus gets
        # dropped off on the way.
        a.phase = "fetch"

    if a.phase == "fetch":
        if is_hermit(ag):
            # THERE IS NO WALK TO TOWN FOR A HERMIT. "fetch" means walk to the
            # colony's stockpile and draw on it, and he has no business at
            # theirs - `_deposit_step` spends a whole docstring on why he does
            # not make that walk in the other direction either.
            #
            # HIS OWN STASH IS THE FETCH, and it is a fetch with no walk in it:
            # the pile is at the camp and the frame is at the camp, so he takes
            # what the stage is short of straight out of it and goes to
            # "approach". Before the stash existed this branch could only fail,
            # which meant a hermit whose fire was one log short went and cut a
            # whole tree for it even with three logs banked ten paces away.
            # CARRY_CAP-bounded like every other fetch in this function, so the
            # hands he arrives with are hands the delivery can actually empty.
            got_any = False
            for res, qty in sorted(need.items(), key=lambda kv: -kv[1]):
                got = hermit_stash_take(w, res, min(int(qty), CARRY_CAP))
                if got > 0:
                    _carry_add(ag, a, res, got)
                    got_any = True
                    break
            if got_any:
                # RETURN, and its absence was a confirmed leak. This branch set
                # the phase and then FELL THROUGH into the colony fetch below,
                # so a hermit who had just drawn a log out of his own pile
                # walked to the colony's stockpile and drew a second armful out
                # of theirs: 6 wood on seed 7 over 45 sim-min, from the one
                # handler that already had a hermit guard in it. The till gate
                # in `stock_take` now bills that second draw to his stash rather
                # than the colony's, so the leak is closed either way - but
                # falling through would still charge him twice for one delivery
                # and walk him into town to do it.
                a.phase = "approach"
                return
            else:
                # Nothing banked either. Failing here is not a dead end, it is
                # the handoff: a failed action drops him straight back into
                # `choose_action`, where `_hermit_bias` has GatherWood scored for
                # exactly this case, and `_deposit_step` then banks or keeps the
                # load instead of destroying it. He goes and cuts his own, and
                # arrives back here in "approach".
                a.data["short"] = {k: int(v) for k, v in need.items()}
                a.failed = True
                return
        want_res, want_qty = None, 0
        for res, qty in sorted(need.items(), key=lambda kv: -kv[1]):
            have = stock_qty(w, res)
            if have > 0:
                want_res, want_qty = res, min(qty, have, CARRY_CAP)
                break
        if want_res is None:
            a.data["short"] = {k: int(v) for k, v in need.items()}
            a.failed = True
            return
        sp = nearest_structure(w, "stockpile", ag.x, built_only=True)
        tx = sp.x if sp is not None else colony_center(w)
        a.pose = "walk"
        rem = step_toward(ag, w, tx, dt)
        if rem <= REACH:
            got = stock_take(w, want_res, want_qty)
            if got <= 0:
                a.failed = True
                return
            _carry_add(ag, a, want_res, got)
            a.phase = "approach"
        elif a.t > 90.0:
            a.failed = True
        return

    if a.phase == "approach":
        a.pose = "carry"
        rem = step_toward(ag, w, float(s.x), dt, speed=WALK_SPEED * 0.92)
        if rem <= REACH * 1.6:
            res = getattr(ag, "carrying", None)
            qty = int(getattr(ag, "carry_qty", 0) or 0)
            if res and qty > 0:
                took = s.deliver(res, qty)
                left = qty - took
                ag.carry_qty = left
                if left <= 0:
                    ag.carrying = None
            stash = a.data.get("stash")
            if isinstance(stash, dict):
                for r, q in list(stash.items()):
                    took = s.deliver(str(r), int(q))
                    stash[r] = int(q) - took
                    if stash[r] <= 0:
                        stash.pop(r, None)
            a.phase = "work" if not s.missing_for_stage() else "fetch"
        elif a.t > 120.0:
            a.failed = True
        return

    if a.phase == "work":
        a.pose = "build"
        _halt(ag)
        _face(ag, float(s.x) - float(ag.x))
        _adjust(ag, "fatigue", 0.012 * dt)
        completed = s.advance(dt, _work_rate(ag))
        if completed:
            name = getattr(ag, "name", "Someone")
            label = s.label() if hasattr(s, "label") else str(s.kind)
            chronicle(w, f"{name} finished the {label}.")
            # NO CELEBRATION AT THE HERMITAGE. `_fresh_celebration` is read by
            # every colonist's Celebrate score, and a celebration is a POINT ON
            # THE MAP they walk to - so pushing one when the hermit finishes his
            # own hut summons the entire colony 600 px out to his door to
            # dance outside the house of the man who left to get away from them.
            # It is also the one build in the game nobody but the builder knows
            # happened, which is the better reason: he raised it alone.
            if not is_hermit_built(s):
                push_celebration(w, float(s.x), float(s.y), str(s.kind))
            _adjust(ag, "morale", 0.18)
            a.done = True
        elif s.missing_for_stage():
            a.phase = "fetch"
        return

    a.phase = "fetch"


def _c_build(a: Action, ag: Any, w: Any) -> None:
    """Put a HERMIT'S un-delivered fetch back in his pile. Never raises.

    The fetch phase above draws out of `hermit_stash_take`, and an Action's
    ``data["stash"]`` does not outlive the action that made it. So a build
    abandoned between the fetch and the delivery - a wolf, a blizzard, a
    re-score - would DESTROY wood that was already his, which is the bug this
    whole feature was written to stop, reintroduced one level down. Villagers
    are deliberately left alone: their surplus goes back to the colony store by
    `_drop_all` on the paths that have one, and reaching into `stock_add` from
    here would double-credit it.
    """
    if not is_hermit(ag):
        return
    try:
        held = a.data.get("stash")
        if isinstance(held, dict):
            for r, q in list(held.items()):
                hermit_stash_add(w, str(r), int(q))
            held.clear()
        res = getattr(ag, "carrying", None)
        qty = int(getattr(ag, "carry_qty", 0) or 0)
        if qty > 0 and res not in (RES_FOOD, RES_COOKED) and res:
            hermit_stash_add(w, str(res), qty)
            ag.carrying = None
            ag.carry_qty = 0
    except Exception:
        log.debug("hermit build cleanup failed", exc_info=True)


def _h_upgrade(a: Action, ag: Any, w: Any, dt: float) -> None:
    """Re-wall a standing timber hut in stone. Same machine as `_h_build`.

    Structurally identical to `_h_build` - fetch, approach, work - against the
    upgrade half of the Structure API instead of the build half. It is a
    separate handler rather than a branch inside `_h_build` because `_h_build`
    sets ``a.done`` the instant ``s.built`` is True, and a hut being upgraded is
    built the whole time: folding this in would have meant a flag threaded
    through every early return in there, on a hot path, for one job.

    Nothing is claimed and nothing is reserved, so there is no `_CLEANUP` entry
    for this kind. An abandoned upgrade leaves the delivered stone in the hut
    and the job open, exactly as an abandoned build leaves a site standing; the
    director hands it straight back out on its next pass.

    THERE IS NO `is_hermit` IN THE FETCH BELOW AND THAT IS ON PURPOSE. This
    handler and `_h_repair` used to be the two identical twins of `_h_build`'s
    fetch block with the hermit branch missing, and one of them quietly took 10
    stone off the colony on seed 11 when a gatherer was promoted mid-action.
    Adding a third copy of the branch is what the bug was made of; the gate is
    at the till now (`stock_take`, and see `_TILL_ACTOR`), so a hermit who ends
    up in here draws on his own pile, comes up empty, and fails the action back
    into `choose_action` where `_hermit_bias` has never scored this job for him
    in the first place. One wasted walk on a role transition, and no branch to
    forget.
    """
    s = structure_by_id(w, a.target)
    if s is None:
        # A reloaded action, or one whose target was collapsed and rebuilt under
        # it. The director's published job is the authority on which hut is
        # being re-walled, so re-read it rather than failing.
        job = getattr(w, "upgrade_job", None)
        if isinstance(job, dict) and job.get("id") is not None:
            a.target = job.get("id")
            s = structure_by_id(w, a.target)
    if s is None or getattr(s, "is_ruined", False):
        a.failed = True
        return
    if not s.is_upgrading:
        # Somebody else laid the last course, or the hut collapsed and
        # `collapse()` dropped the job. Either way this is finished work, not
        # failed work - failing it would cost the agent the hysteresis bonus and
        # look like a botched job in the debug view.
        a.done = True
        return

    need = s.upgrade_missing()
    carrying = getattr(ag, "carrying", None)
    carry_qty = int(getattr(ag, "carry_qty", 0) or 0)

    if not need:
        a.phase = "work"
    elif carrying in need and carry_qty > 0:
        a.phase = "approach"
    elif _has_load(ag, a) and _load_is_useful(ag, a, need):
        # The stash arm, and it is not optional. Routing on the hand alone
        # deadlocks the moment a villager fetches stone with wood already
        # shouldered: `_carry_add` puts the stone in the action's stash, the
        # hand still holds wood, the top of the next tick re-routes to "fetch",
        # and he walks back and draws *another* armful of stone out of the
        # stockpile. Unbounded, once per pass. Same failure `_h_build` records
        # above; same fix.
        a.phase = "approach"
    else:
        a.phase = "fetch"

    if a.phase == "fetch":
        want_res, want_qty = None, 0
        for res, qty in sorted(need.items(), key=lambda kv: -kv[1]):
            have = stock_qty(w, res)
            if have > 0:
                want_res, want_qty = res, min(qty, have, CARRY_CAP)
                break
        if want_res is None:
            a.data["short"] = {k: int(v) for k, v in need.items()}
            a.failed = True
            return
        sp = nearest_structure(w, "stockpile", ag.x, built_only=True)
        tx = sp.x if sp is not None else colony_center(w)
        a.pose = "walk"
        rem = step_toward(ag, w, tx, dt)
        if rem <= REACH:
            got = stock_take(w, want_res, want_qty)
            if got <= 0:
                a.failed = True
                return
            _carry_add(ag, a, want_res, got)
            a.phase = "approach"
        elif a.t > 90.0:
            a.failed = True
        return

    if a.phase == "approach":
        a.pose = "carry"
        rem = step_toward(ag, w, float(s.x), dt, speed=WALK_SPEED * 0.92)
        if rem <= REACH * 1.6:
            res = getattr(ag, "carrying", None)
            qty = int(getattr(ag, "carry_qty", 0) or 0)
            if res and qty > 0:
                took = s.deliver_upgrade(res, qty)
                left = qty - took
                ag.carry_qty = left
                if left <= 0:
                    ag.carrying = None
            stash = a.data.get("stash")
            if isinstance(stash, dict):
                for r, q in list(stash.items()):
                    took = s.deliver_upgrade(str(r), int(q))
                    stash[r] = int(q) - took
                    if stash[r] <= 0:
                        stash.pop(r, None)
            a.phase = "work" if not s.upgrade_missing() else "fetch"
        elif a.t > 120.0:
            a.failed = True
        return

    if a.phase == "work":
        a.pose = "build"        # an existing pose; the tier adds no render work
        _halt(ag)
        _face(ag, float(s.x) - float(ag.x))
        _adjust(ag, "fatigue", 0.012 * dt)
        completed = s.advance_upgrade(dt, _work_rate(ag))
        if completed:
            name = getattr(ag, "name", "Someone")
            chronicle(w, f"{name} re-walled the hut in stone.")
            # "hut", not "hut_stone": the celebration table is keyed on sim
            # kinds, and the sim kind never changes - only state["material"].
            push_celebration(w, float(s.x), float(s.y), "hut")
            _adjust(ag, "morale", 0.18)
            a.done = True
        elif s.upgrade_missing():
            a.phase = "fetch"
        return

    a.phase = "fetch"


def _h_repair(a: Action, ag: Any, w: Any, dt: float) -> None:
    s = structure_by_id(w, a.target)
    if s is None:
        reg = structures_of(w)
        if reg is not None:
            dmg = colony_repairable(reg, threshold=1.0) if reg is not None else []
            if dmg:
                s = min(dmg, key=lambda st: abs(st.x - float(ag.x)))
                a.target = s.id
    # A repair that has ended up pointed at the hermit's camp - by a stale id, a
    # save, or a re-target - is abandoned rather than carried out. See
    # `colony_repairable`.
    if s is None or getattr(s, "is_ruined", False) or is_hermit_built(s):
        a.failed = True
        return
    if s.hp >= s.max_hp:
        a.done = True
        return

    if a.phase in ("start", "approach"):
        a.phase = "approach"
        a.pose = "walk"
        rem = step_toward(ag, w, float(s.x), dt)
        if rem <= REACH * 1.6:
            a.phase = "work"
            a.data["mat_t"] = 0.0
        elif a.t > 75.0:
            a.failed = True
        return

    a.pose = "build"
    _halt(ag)
    _face(ag, float(s.x) - float(ag.x))
    a.data["mat_t"] = float(a.data.get("mat_t", 0.0)) + dt
    rate = 12.0 * _work_rate(ag)
    if a.data["mat_t"] >= 2.0:
        a.data["mat_t"] = 0.0
        primary = RES_STONE if s.kind in ("wall", "grave") else RES_WOOD
        # No hermit branch here either - the till gate covers it. A hermit
        # patching a colony wall (only reachable by a mid-action promotion;
        # `_hermit_bias` zeroes RepairStructure and `is_hermit_built` already
        # keeps him off his own camp) draws on his own pile or patches at the
        # 0.35 rate with what is to hand. See `_h_upgrade` and `_TILL_ACTOR`.
        if stock_take(w, primary, 1) <= 0:
            rate *= 0.35        # patching with what is to hand: slow
    s.repair(rate * dt)
    s.extinguish()
    _adjust(ag, "fatigue", 0.01 * dt)
    if s.hp >= s.max_hp:
        chronicle(w, f"{getattr(ag, 'name', 'Someone')} repaired the {s.kind}.")
        a.done = True


def _h_eat(a: Action, ag: Any, w: Any, dt: float) -> None:
    """Eat what is in hand, then - for a hermit - what is in his stash.

    THE ORDER IS THE STARVATION GUARD AND IT IS UNCHANGED. Hands first, exactly
    as before: `_score_wants` forces Eat to 1.0 past 0.85 hunger for a man who
    is `holding_food`, `_deposit_step` leaves a hermit's food IN HIS HANDS
    rather than banking it, and this branch is what turns that into a meal. Nothing in the stash work below can reach a hermit who still
    has food on him.

    The stash arm is a NEW rung under the old one, not a replacement for it. A
    hermit with empty hands used to fall straight through to the "approach"
    phase, which is a walk to the COLONY'S stockpile - the one errand the role
    exists to end, kept only because starving out there is worse than breaking
    character. Now his own camp store is tried first, so that walk happens only
    when his hands and his pile are both empty. It is strictly a safer path than
    the one it precedes: it can feed him where the old code could only start
    walking.
    """
    if a.phase == "start":
        held = getattr(ag, "carrying", None)
        if held in (RES_FOOD, RES_COOKED) and int(getattr(ag, "carry_qty", 0) or 0) > 0:
            ag.carry_qty = int(ag.carry_qty) - 1
            if ag.carry_qty <= 0:
                ag.carrying = None
            a.data["res"] = held
            a.phase = "eat"
            a.data["et"] = 0.0
        elif is_hermit(ag) and hermit_stash_food(w)[0] is not None:
            sres, _ = hermit_stash_food(w)
            if sres is None or hermit_stash_take(w, sres, 1) <= 0:
                a.failed = True
                return
            a.data["res"] = sres
            a.data["et"] = 0.0
            a.phase = "eat"
        else:
            res, qty = food_in_store(w)
            if res is None or qty <= 0:
                a.failed = True
                return
            a.data["res"] = res
            a.phase = "approach"

    if a.phase == "approach":
        sp = nearest_structure(w, "stockpile", ag.x, built_only=True)
        tx = sp.x if sp is not None else colony_center(w)
        a.pose = "walk"
        rem = step_toward(ag, w, tx, dt, speed=WALK_SPEED * 1.05)
        if rem <= REACH:
            res, qty = food_in_store(w)
            # THE ONE WITHDRAWAL A HERMIT IS ALLOWED TO MAKE, and it is claimed
            # explicitly rather than inherited by omission - see `colony_till`.
            # The till gate would otherwise route this back into his stash, and
            # his stash is empty: that is the only way execution reaches this
            # phase at all (the "start" branch above tries his hands, then his
            # pile, and only falls through to a walk into town when both are
            # bare). Sending him home to eat nothing is how a man starves
            # standing on his principles, which is a worse bug than the one
            # this file is fixing. `with`, not a flag argument, so an exception
            # inside cannot leave the till propped open for the next agent.
            with colony_till():
                got = stock_take(w, res, 1) if res is not None else 0
            if res is None or got <= 0:
                a.failed = True
                return
            a.data["res"] = res
            a.data["et"] = 0.0
            a.phase = "eat"
        elif a.t > 75.0:
            a.failed = True
        return

    if a.phase == "eat":
        a.pose = "eat"
        _halt(ag)
        a.data["et"] = float(a.data.get("et", 0.0)) + dt
        if a.data["et"] >= EAT_TIME:
            cooked = a.data.get("res") == RES_COOKED
            _adjust(ag, "hunger", -0.78 if cooked else -0.52)
            _adjust(ag, "morale", 0.09 if cooked else 0.04)
            a.done = True
        return

    a.phase = "start"


def _pick_hut(w: Any, ag: Any) -> Structure | None:
    return nearest_structure(
        w, "hut", ag.x, built_only=True,
        predicate=lambda s: s.has_room() or int(getattr(ag, "id", -1)) in s.occupants,
    )


def _h_sleep_rough(a: Action, ag: Any, w: Any, dt: float) -> None:
    """A hermit sleeps where he lies, at his own camp. No hut, no walk home.

    This is not a nicety, it is the load-bearing half of the role. Fatigue is
    the one need with no field remedy: `_h_sleep` is the only thing in the sim
    that sets ``pose == "sleep"`` on purpose, and `entities.update_needs` only
    pays fatigue back for that pose. A hermit denied it would sit pinned near
    1.0, and fatigue pins MORALE too (``target = 1 - max(hunger, fatigue,
    warmth)``) - which does not kill him, but does drag the colony MEAN morale
    that `behavior._thrift` and the stone-hut tier's 0.55-for-180 s gate are
    both read off. One permanently miserable man would have quietly cost the
    colony its masonry.

    The alternative was letting him use a hut, and a hermit who walks back into
    town every single night is not a hermit.

    HE NOW HAS A HUT, and when it is standing he sleeps IN it - `hermit_hut`,
    `enter()`, `ag.inside`, the lot, exactly the way a villager uses a hut, so
    the renderer lights his doorway at night for free and he is not sprawled on
    his own doorstep. Two guards make that safe rather than merely nicer:
    HERMIT_BED_WALK bounds the walk (he lies down where he stands if the door is
    somehow unreachable, which is the guarantee this whole function exists to
    keep), and a hut that burns down mid-sleep drops him back out onto the
    ground instead of leaving him flagged inside a building that no longer
    exists - which is invisible forever.
    Indoors he gets a HUT's fatigue and morale rates rather than the rough ones
    below, so the role's measured safety margin can only widen: a bed he built
    himself is strictly better than the ground beside it, never worse.

    SO HIS CAMP IS SHELTER even before the hut goes up, and it shelters him
    from the cold at the same rate a
    hut does (0.030/s). That is a grant - he gets a roof he never built - and it
    was arrived at the hard way. The first cut paid warmth back at half a hut's
    rate on the theory that rough sleep should be strictly worse than a bed, and
    it was worse in exactly the way that ruins the feature: he could not clear a
    night's chill lying down (COLD_PER_SEC * 0.6 = 0.002/s over a ~600 s night),
    so `WarmAtFire` won instead and he spent 8-9k ticks per 75 sim-min walking
    to the colony's firepit - and was still the coldest colonist in the run.
    Sleep is now his answer to being cold as well as to being tired, which is
    what a lean-to is for.

    Rough sleep is still worse than a bed where it costs him rather than the
    feature: fatigue comes back at 0.040/s against a hut's 0.055/s, so he needs
    more of the night to rest, and a blizzard (0.008/s) out-chills the camp and
    can still drive him in to the fire. See `behavior._hermit_bias`.
    """
    house = hermit_hut(w, built_only=True)
    if a.data.get("in"):
        # Already in bed. If the roof has come off in the night - burned by a
        # dragon, or simply gone - get him out of it now. `ag.inside` pointing
        # at a structure that no longer exists is a colonist nothing draws.
        if house is None or structure_by_id(w, a.target) is not house:
            _c_sleep(a, ag, w)
            house = None
    elif house is not None and a.phase != "rough":
        a.target = house.id
        a.phase = "approach"
        a.pose = "walk"
        rem = step_toward(ag, w, float(house.x), dt)
        if rem <= REACH * 1.4:
            if house.enter(int(getattr(ag, "id", 0))):
                a.data["in"] = 1
                a.phase = "bed"
                try:
                    ag.inside = int(house.id)
                except Exception:
                    pass
            else:                       # somebody else's hut, or no room
                a.phase = "rough"
        elif a.t > HERMIT_BED_WALK:
            # Out of walking budget. Lying down here beats pacing toward a door
            # he cannot reach while his fatigue pins at 1.0 - see the docstring.
            a.phase = "rough"
        return

    indoors = bool(a.data.get("in")) and house is not None
    a.pose = "sleep"
    _halt(ag)
    if indoors:
        # Park him at the hut, as `_h_sleep` does with a hut: he is not drawn
        # in there, but the position is where he reappears on waking and where
        # anything he carries shines from.
        try:
            ag.x = float(house.x)
            ag.y = float(house.y) - 2.0
            ag.inside = int(house.id)
        except Exception:
            indoors = False
    _adjust(ag, "fatigue", -(0.055 if indoors else 0.040) * dt)
    _adjust(ag, "warmth", -0.030 * dt)
    _adjust(ag, "hunger", 0.004 * dt)
    _adjust(ag, "morale", (0.006 if indoors else 0.004) * dt)
    fatigue = _need(ag, "fatigue")
    warm_enough = _need(ag, "warmth") < 0.30
    # Warmth is in the wake test as well as the sleep score, or a hermit who
    # bedded down cold would get up the moment he stopped being tired and start
    # the walk to the fire he lay down to avoid.
    if a.t > 400.0 or (warm_enough and (
            fatigue <= 0.02 or (not is_night(w) and fatigue < 0.22))):
        # Free the bed here, not just via abandon(), for the same reason
        # `_h_sleep` does: a hermit who wakes up still counted as an occupant
        # locks the only place his successor could sleep.
        if a.data.get("in"):
            _c_sleep(a, ag, w)
        a.done = True


def _h_sleep(a: Action, ag: Any, w: Any, dt: float) -> None:
    if is_hermit(ag):
        # He never claims a HUT. He does claim his own hut when he has one,
        # and `_h_sleep_rough` sets a.target and a.data["in"] for it, so the
        # shared `_c_sleep` cleanup releases the right building either way -
        # and is still a no-op on the nights he sleeps on the ground.
        _h_sleep_rough(a, ag, w, dt)
        return
    hut = structure_by_id(w, a.target)
    if hut is None or getattr(hut, "is_ruined", False) or not getattr(hut, "built", False):
        if a.phase == "sleep":
            a.done = True           # bed collapsed under them; just get up
            return
        hut = _pick_hut(w, ag)
        if hut is None:
            a.failed = True
            return
        a.target = hut.id

    if a.phase in ("start", "approach"):
        a.phase = "approach"
        a.pose = "walk"
        rem = step_toward(ag, w, float(hut.x), dt)
        if rem <= REACH * 1.4:
            if not hut.enter(int(getattr(ag, "id", 0))):
                a.failed = True
                return
            a.data["in"] = 1
            a.phase = "sleep"
            # Actually go inside. render/ hides anyone with .inside set and
            # lights the hut instead, so they are no longer sprawled across
            # the doorstep in full view.
            try:
                ag.inside = int(hut.id)
            except Exception:
                pass
        elif a.t > 75.0:
            a.failed = True
        return

    a.pose = "sleep"
    _halt(ag)
    # Park them at the hut's centre. They are not drawn while inside, but the
    # position still matters: it is where they reappear on waking, and it keeps
    # any light they carry (a candle) shining from the right building.
    try:
        ag.x = float(hut.x)
        ag.y = float(hut.y) - 2.0
        ag.inside = int(hut.id)
    except Exception:
        pass
    _adjust(ag, "fatigue", -0.055 * dt)
    _adjust(ag, "warmth", -0.030 * dt)
    _adjust(ag, "hunger", 0.004 * dt)
    _adjust(ag, "morale", 0.006 * dt)
    fatigue = _need(ag, "fatigue")
    if fatigue <= 0.02 or (not is_night(w) and fatigue < 0.22) or a.t > 400.0:
        _c_sleep(a, ag, w)          # free the bed here, not just via abandon()
        a.done = True


def _c_sleep(a: Action, ag: Any, w: Any) -> None:
    hut = structure_by_id(w, a.target)
    if hut is not None:
        hut.leave(int(getattr(ag, "id", 0)))
    if a.data.get("in"):
        # _plant, not a bare y write, for the same reason the lookout uses it:
        # a sleeper parked at the hut floor sits a couple of px above the
        # ground line, so entities has been reading them as airborne all night.
        # Waking them without asserting vy hands whatever gravity accrued in
        # there to the landing check.
        try:
            _plant(ag, ground_y(w, ag.x))
        except Exception:
            pass
    # Always clear this, even if the hut is gone: an agent left flagged as
    # inside a building that no longer exists would be invisible forever.
    try:
        ag.inside = None
    except Exception:
        pass
    a.data["in"] = 0


def _h_warm(a: Action, ag: Any, w: Any, dt: float) -> None:
    fire = _regate_fire(a, ag, w)       # the mutual gate - see `fire_for`
    if fire is None:
        return

    if a.phase in ("start", "approach"):
        a.phase = "approach"
        a.pose = "walk"
        rem = step_toward(ag, w, float(fire.x), dt, arrive=FIRE_REACH)
        if rem <= FIRE_REACH:
            a.phase = "warm"
        elif _walked_for(a) > 75.0:
            a.failed = True
        return

    a.pose = "warm"
    _halt(ag)
    _face(ag, float(fire.x) - float(ag.x))
    if not fire.fire_active:
        wood = _stoke_wood(ag, w, 2)
        if wood <= 0:
            a.failed = True
            return
        fire.stoke(wood)
        fire.light_fire(0.5)
        chronicle(w, f"{getattr(ag, 'name', 'Someone')} coaxed the fire back to life.")
    _adjust(ag, "warmth", -0.11 * dt)
    _adjust(ag, "morale", 0.012 * dt)
    _adjust(ag, "fatigue", -0.004 * dt)
    if _need(ag, "warmth") <= 0.04 or a.t > 60.0:
        a.done = True


def _h_cook(a: Action, ag: Any, w: Any, dt: float) -> None:
    fire = _regate_fire(a, ag, w)       # the mutual gate - see `fire_for`
    if fire is None:
        return

    held_food = (getattr(ag, "carrying", None) == RES_FOOD
                 and int(getattr(ag, "carry_qty", 0) or 0) > 0)
    if a.phase == "start":
        # A HERMIT COOKS WHAT HE PICKED. The store test below - and the whole
        # "fetch" phase under it - is a walk to the colony's stockpile, which is
        # the one errand this role must never run. His larder is his hands
        # (`_deposit_step` hands a hermit his forage back instead of shelving
        # it), so nothing raw in hand means nothing to cook, full stop.
        if is_hermit(ag):
            # ...but his own stash counts as "in hand". It sits at the camp, the
            # fire sits at the camp, and taking three berries out of his own pile
            # is not the walk to the stockpile this branch refuses - it is the
            # thing that makes a full larder worth cooking down into meals
            # instead of eaten raw one at a time.
            if not held_food:
                got = hermit_stash_take(w, RES_FOOD, 3)
                if got <= 0:
                    a.failed = True
                    return
                _carry_add(ag, a, RES_FOOD, got)
            a.phase = "approach"
        elif stock_qty(w, RES_FOOD) <= 0 and not held_food:
            a.failed = True
            return
        else:
            a.phase = "fetch"

    if a.phase == "fetch":
        if held_food:
            a.phase = "approach"
        else:
            sp = nearest_structure(w, "stockpile", ag.x, built_only=True)
            tx = sp.x if sp is not None else colony_center(w)
            a.pose = "walk"
            rem = step_toward(ag, w, tx, dt)
            if rem <= REACH:
                got = stock_take(w, RES_FOOD, 3)
                if got <= 0:
                    a.failed = True
                    return
                _carry_add(ag, a, RES_FOOD, got)
                a.phase = "approach"
            elif a.t > 90.0:
                a.failed = True
            return

    if a.phase == "approach":
        a.pose = "carry"
        rem = step_toward(ag, w, float(fire.x), dt, arrive=FIRE_REACH)
        if rem <= FIRE_REACH:
            a.phase = "cook"
            a.data["ct"] = 0.0
        elif _walked_for(a) > 120.0:
            a.failed = True
        return

    if a.phase == "cook":
        a.pose = "cook"
        _halt(ag)
        _face(ag, float(fire.x) - float(ag.x))
        if not fire.fire_active:
            got = _stoke_wood(ag, w, 1)
            if got > 0:
                fire.stoke(got)
                fire.light_fire(0.4)
            else:
                a.phase = "deliver"
                return
        a.data["ct"] = float(a.data.get("ct", 0.0)) + dt
        _adjust(ag, "warmth", -0.03 * dt)
        if a.data["ct"] >= COOK_TIME:
            qty = int(getattr(ag, "carry_qty", 0) or 0)
            cooked_any = False
            if getattr(ag, "carrying", None) == RES_FOOD and qty > 0:
                ag.carrying = RES_COOKED
                cooked_any = True
            # ...AND THE FOOD THAT NEVER REACHED HIS HANDS. This line used to be
            # the whole of it, and it silently un-cooked the meal for anybody
            # already carrying something else. `_carry_add` puts a SECOND
            # resource in the ACTION's stash rather than the agent's slot, so a
            # cook who walked up holding wood had his raw food sitting in
            # `a.data["stash"]` where the test above could not see it - no
            # conversion, no chronicle line, and then `_deposit_step` banked it
            # back as FOOD. The hermit is how it surfaced (reported from play as
            # "it just goes back into food and not cooked") because he is nearly
            # always holding timber for his fire, but a villager mid-haul hits
            # exactly the same path into the colony stockpile.
            st = a.data.get("stash")
            if isinstance(st, dict):
                raw = int(st.pop(RES_FOOD, 0) or 0)
                if raw > 0:
                    st[RES_COOKED] = int(st.get(RES_COOKED, 0) or 0) + raw
                    cooked_any = True
            if cooked_any:
                chronicle(w, f"{getattr(ag, 'name', 'Someone')} cooked a meal.")
            a.phase = "deliver"
        return

    if _deposit_step(a, ag, w, dt):
        a.done = True


def _h_plant(a: Action, ag: Any, w: Any, dt: float) -> None:
    if a.phase == "start":
        rng = rng_of(w)
        base = colony_center(w)
        tx = None
        for _ in range(12):
            cand = _clamp_x(base + rng.uniform(-380.0, 380.0))
            if abs(slope_at(w, cand)) > MAX_SLOPE_WALK:
                continue
            near = find_prop(w, _TREE_KINDS + ("sapling",), cand, max_dist=44.0)
            if near is not None:
                continue
            st = nearest_structure(w, None, cand, built_only=False)
            if st is not None and abs(float(st.x) - cand) < 34.0:
                continue
            tx = cand
            break
        if tx is None:
            a.failed = True
            return
        a.data["tx"] = float(tx)
        a.phase = "approach"

    if a.phase == "approach":
        a.pose = "walk"
        rem = step_toward(ag, w, float(a.data.get("tx", ag.x)), dt)
        if rem <= REACH:
            a.phase = "plant"
            a.data["pt"] = 0.0
        elif a.t > 75.0:
            a.failed = True
        return

    a.pose = "plant"
    _halt(ag)
    a.data["pt"] = float(a.data.get("pt", 0.0)) + dt
    if a.data["pt"] >= PLANT_TIME:
        if spawn_sapling(w, float(a.data.get("tx", ag.x))):
            chronicle(w, f"{getattr(ag, 'name', 'Someone')} planted a sapling.")
            _adjust(ag, "morale", 0.06)
            a.done = True
        else:
            a.failed = True


def _h_converse(a: Action, ag: Any, w: Any, dt: float) -> None:
    """Two colonists stop and talk - and, once in a while, a visit to the hermit.

    A VISIT IS THIS ACTION WITH A LONGER LEASH, not an action of its own, and
    that is a measurement constraint rather than a saving. `choose_action` draws
    one `rng.uniform` tiebreak per POSITIVELY SCORED candidate, so a new action
    kind re-phases the world's random stream for every colonist on every tick -
    the same reason `_hermit_bias` is built entirely out of re-weighting keys
    that already existed. Reusing Converse costs one extra draw, for one
    colonist, on the ticks a visit is actually open.

    Three things change when ``a.data["visit"]`` is set, all of them because the
    partner is 700 px away instead of 30:

    * the approach budget goes from 30 s to `hermit_visit_budget`, which scales
      with the walk. Thirty seconds does not cross the standoff even on flat
      ground, so a visit under the ordinary budget would be a man who sets off,
      gives up halfway and turns round - which is worse than no feature at all.
      A FIXED budget turned out to be the same bug with a longer fuse: 260 s
      covered the old 530..1060 px band and covered about 3900 px of broken
      ground, which is inside the new one, so 34 visits in 35 ended as exactly
      the man this bullet says must not exist. See HERMIT_VISIT_TIMEOUT;
    * the talk runs longer, because somebody who walked a quarter of a mile does
      not say hello and leave;
    * it pays out once, on contact: a chronicle line, morale for both (more for
      the host - he is the one who has not spoken to anybody), and whatever food
      the visitor happened to be carrying.

    The walk home needs no code. The visitor's jobs, his stockpile and his bed
    are all back in the settlement, so the moment this finishes every score he
    has points that way.
    """
    visit = bool(a.data.get("visit"))
    other = agent_by_id(w, a.target)
    if other is None or not getattr(other, "alive", True):
        if visit:                    # the host died on the way out
            _close_hermit_visit(w)
            a.done = True
            return
        other = _nearest_other(w, ag, 240.0)
        if other is None:
            a.done = True
            return
        a.target = getattr(other, "id", None)

    if a.phase in ("start", "approach"):
        a.phase = "approach"
        a.pose = "walk"
        gap = float(other.x) - float(ag.x)
        want = float(other.x) - (TALK_REACH * 0.6) * (1.0 if gap > 0 else -1.0)
        rem = step_toward(ag, w, want, dt)
        if abs(float(other.x) - float(ag.x)) <= TALK_REACH:
            a.phase = "talk"
            a.data["st"] = 0.0
            _pair_up(w, ag, other)
            if visit:
                # Measured from CONTACT, not from setting out. `a.t` has the
                # whole walk in it by now - 60 s and more across the standoff -
                # so a stay expressed as an absolute `a.t` bound is a stay that
                # has already expired the instant he arrives, and the visit is a
                # man who walks 700 px and turns straight round. Caught by the
                # hermit standing frozen in "approach" for the whole 260 s
                # watchdog afterwards, which is what a visit that never opened
                # and never closed looks like from his end.
                a.data["until"] = float(a.t) + CONVERSE_TIME * 3.0
                _arrive_at_hermit(w, ag, other)
        elif a.t > (hermit_visit_budget(w) if visit else 30.0):
            if visit:
                _close_hermit_visit(w)
            a.done = True
        return

    a.pose = "talk"
    _halt(ag)
    _face(ag, float(other.x) - float(ag.x))
    if abs(float(other.x) - float(ag.x)) > TALK_REACH * 2.2:
        if visit:
            _close_hermit_visit(w)
        a.done = True
        return
    a.data["st"] = float(a.data.get("st", 0.0)) + dt
    if a.data["st"] >= 1.15:
        a.data["st"] = 0.0
        rng = rng_of(w)
        emit_speech(w, ag, rng.choice(SPEECH_SYMBOLS))
    _adjust(ag, "morale", 0.030 * dt)
    _adjust(ag, "fatigue", -0.002 * dt)
    if a.t > (a.data.get("until") or (CONVERSE_TIME + 1.0)):
        if visit:
            _close_hermit_visit(w)
        a.done = True


def _close_hermit_visit(w: Any) -> None:
    """Hand the visit slot back. Never raises; safe to call twice."""
    try:
        setattr(w, "hermit_visit", None)
    except Exception:
        pass


def _arrive_at_hermit(w: Any, guest: Any, host: Any) -> None:
    """The moment of the visit. Fires once, on contact, and never raises.

    Deliberately does NOT touch the stockpile. A gift that came out of the
    colony's stores would be a transfer the colony never carried and never
    misses, invented 700 px from the building it came out of; the whole role is
    built on "what the hermit picks, the colony never sees" and this is the same
    rule pointed the other way. So the gift is whatever the visitor already has
    in his hands - which is often something, because a gatherer on his way to
    the store is exactly the sort of person with a spare hand for the walk - and
    on the trips where it is nothing, the visit is just company. That reads
    better than a guaranteed handout anyway.
    """
    try:
        gname = str(getattr(guest, "name", "Someone"))
        hname = str(getattr(host, "name", "the hermit"))
        _adjust(host, "morale", HERMIT_VISIT_MORALE_HOST)
        _adjust(guest, "morale", HERMIT_VISIT_MORALE_GUEST)
        gave = 0
        res = getattr(guest, "carrying", None)
        qty = int(getattr(guest, "carry_qty", 0) or 0)
        if res in (RES_FOOD, RES_COOKED) and qty > 0 and not getattr(host, "carrying", None):
            gave = min(qty, int(HERMIT_VISIT_GIFT), CARRY_CAP)
            try:
                host.carrying = res
                host.carry_qty = gave
                guest.carry_qty = qty - gave
                if guest.carry_qty <= 0:
                    guest.carrying = None
            except Exception:
                gave = 0
        if gave > 0:
            chronicle(w, f"{gname} walked out to the hermit's hut and left "
                         f"{hname} some food.")
        else:
            chronicle(w, f"{gname} walked out to sit a while with {hname}.")
        try:
            w.stats["hermit_visits"] = int(w.stats.get("hermit_visits", 0)) + 1
        except Exception:
            pass
    except Exception:
        log.debug("hermit visit payout failed", exc_info=True)


def _nearest_other(w: Any, ag: Any, max_dist: float) -> Any | None:
    best, best_d = None, float("inf")
    my_id = getattr(ag, "id", None)
    for o in alive_agents(w):
        if getattr(o, "id", None) == my_id:
            continue
        act = getattr(o, "action", None)
        if act is not None and getattr(act, "kind", "") in ("Sleep", "Lookout", "Panic", "FleeFrom"):
            continue
        d = abs(float(getattr(o, "x", 0.0)) - float(ag.x))
        if d < best_d:
            best_d, best = d, o
    return best if best is not None and best_d <= max_dist else None


def _pair_up(w: Any, ag: Any, other: Any) -> None:
    """Give the partner a matching Converse so the pause is mutual."""
    try:
        cur = getattr(other, "action", None)
        if cur is not None and getattr(cur, "kind", "") not in ("Wander", "Converse"):
            return
        if cur is not None and getattr(cur, "kind", "") == "Converse":
            return
        if cur is not None:
            cur.abandon(other, w)
        other.action = make_action(
            "Converse", target=getattr(ag, "id", None), phase="talk", pose="talk")
    except Exception:
        log.debug("pair_up failed", exc_info=True)


def _h_celebrate(a: Action, ag: Any, w: Any, dt: float) -> None:
    if a.phase == "start":
        tx = a.data.get("tx")
        if tx is None:
            s = structure_by_id(w, a.target)
            if s is not None:
                tx = float(s.x)
            else:
                cel = celebrations_of(w)
                tx = float(cel[-1].get("x", ag.x)) if cel else float(ag.x)
        a.data["tx"] = _clamp_x(float(tx))
        a.phase = "approach"

    if a.phase == "approach":
        a.pose = "walk"
        rem = step_toward(ag, w, float(a.data["tx"]), dt, speed=WALK_SPEED * 1.15,
                          arrive=FIRE_REACH)
        if rem <= FIRE_REACH or a.t > 9.0:   # was 25s: a long failed walk reads as pacing
            a.phase = "dance"
            a.data["dt"] = 0.0
        return

    a.pose = "dance"
    _halt(ag)
    a.data["dt"] = float(a.data.get("dt", 0.0)) + dt
    if int(a.data["dt"] * 3.0) % 2 == 0:
        ag.facing = 1
    else:
        ag.facing = -1
    _adjust(ag, "morale", 0.055 * dt)
    _adjust(ag, "fatigue", 0.008 * dt)
    if a.data["dt"] >= CELEBRATE_TIME:
        a.done = True


def _h_mourn(a: Action, ag: Any, w: Any, dt: float) -> None:
    grave = structure_by_id(w, a.target)
    if grave is None or grave.kind != "grave":
        grave = nearest_structure(w, "grave", ag.x, built_only=False)
        if grave is None:
            a.failed = True
            return
        a.target = grave.id

    if a.phase in ("start", "approach"):
        a.phase = "approach"
        a.pose = "walk"
        rem = step_toward(ag, w, float(grave.x) + 10.0, dt, speed=WALK_SPEED * 0.7)
        if rem <= REACH * 1.5:
            a.phase = "bow"
            a.data["bt"] = 0.0
        elif a.t > 90.0:
            a.failed = True
        return

    a.pose = "mourn"
    _halt(ag)
    _face(ag, float(grave.x) - float(ag.x))
    a.data["bt"] = float(a.data.get("bt", 0.0)) + dt
    _adjust(ag, "morale", -0.020 * dt)
    if a.data["bt"] >= MOURN_TIME:
        _adjust(ag, "morale", 0.10)        # grief spent, they carry on
        try:
            mourners = grave.state.get("mourners")
            if not isinstance(mourners, list):
                mourners = []
                grave.state["mourners"] = mourners
            aid = int(getattr(ag, "id", 0))
            if aid not in mourners:
                mourners.append(aid)
            grave.state["mourn_done"] = float(grave.state.get("mourn_done", 0.0)) + 1.0
        except Exception:
            pass
        a.done = True


def _h_flee(a: Action, ag: Any, w: Any, dt: float) -> None:
    fx = a.data.get("fx")
    if not isinstance(fx, (int, float)):
        fx = float(a.target) if isinstance(a.target, (int, float)) else float(ag.x)
        a.data["fx"] = float(fx)
    radius = float(a.data.get("radius", 90.0) or 90.0)

    if a.phase == "start":
        if a.data.get("uphill"):
            d = _uphill_dir(w, float(ag.x))
        else:
            d = 1.0 if float(ag.x) >= float(fx) else -1.0
            edge = _clamp_x(float(ag.x) + d * (radius + 110.0))
            if abs(edge - float(ag.x)) < radius * 0.5:
                d = -d          # backed against a wall, break the other way
        a.data["dir"] = float(d)
        a.data["tx"] = _clamp_x(float(ag.x) + d * (radius + 130.0))
        a.phase = "run"

    a.pose = "flee"
    d = float(a.data.get("dir", 1.0))
    tx = float(a.data.get("tx", ag.x))
    rem = step_toward(ag, w, tx, dt, speed=WALK_SPEED * 2.1, arrive=6.0)
    _adjust(ag, "fatigue", 0.035 * dt)
    _adjust(ag, "morale", -0.012 * dt)
    away = abs(float(ag.x) - float(fx))
    if rem <= 6.0 and away < radius * 1.3:
        a.data["tx"] = _clamp_x(float(ag.x) + d * 150.0)
        return
    if away > radius * 1.3 and a.t > 1.0:
        a.done = True
    elif a.t > 22.0:
        a.done = True


def _h_climb(a: Action, ag: Any, w: Any, dt: float) -> None:
    ty = a.data.get("ty")
    if not isinstance(ty, (int, float)):
        ty = ground_y(w, ag.x) - 90.0
        a.data["ty"] = float(ty)
    if a.phase == "start":
        a.data["dir"] = _uphill_dir(w, float(ag.x))
        a.data["lastx"] = float(ag.x)
        a.data["stuck"] = 0.0
        a.phase = "climb"

    d = float(a.data.get("dir", 1.0))
    sl = slope_at(w, ag.x)
    a.pose = "climb" if abs(sl) > MAX_SLOPE_WALK else "walk"
    step_toward(ag, w, _clamp_x(float(ag.x) + d * 60.0), dt, arrive=1.0)
    _adjust(ag, "fatigue", 0.018 * dt)

    moved = abs(float(ag.x) - float(a.data.get("lastx", ag.x)))
    a.data["lastx"] = float(ag.x)
    a.data["stuck"] = 0.0 if moved > 0.05 else float(a.data.get("stuck", 0.0)) + dt
    if ground_y(w, ag.x) <= float(ty):
        a.done = True
    elif a.data["stuck"] > 2.0 or a.t > 30.0:
        a.done = True


def _h_lookout(a: Action, ag: Any, w: Any, dt: float) -> None:
    tower = structure_by_id(w, a.target)
    if tower is None or tower.kind != "watchtower" or getattr(tower, "is_ruined", False):
        if a.phase in ("ascend", "watch", "descend"):
            # The tower went out from under them - put them back on the ground
            # rather than leaving a stickman hanging in mid-air.
            _c_lookout(a, ag, w)
            a.done = True
            return
        tower = nearest_structure(
            w, "watchtower", ag.x, built_only=True,
            predicate=lambda s: s.has_room() or int(getattr(ag, "id", -1)) in s.occupants,
        )
        if tower is None:
            a.failed = True
            return
        a.target = tower.id

    if a.phase in ("start", "approach"):
        a.phase = "approach"
        a.pose = "walk"
        rem = step_toward(ag, w, float(tower.x), dt)
        if rem <= REACH:
            if not tower.enter(int(getattr(ag, "id", 0))):
                a.failed = True
                return
            a.data["in"] = 1
            a.data["y0"] = float(ag.y)
            a.data["at"] = 0.0
            a.data["dur"] = rng_of(w).uniform(70.0, 190.0)
            a.phase = "ascend"
        elif a.t > 90.0:
            a.failed = True
        return

    top = float(tower.top_y()) + 6.0

    if a.phase == "ascend":
        a.pose = "climb"
        _halt(ag)
        a.data["at"] = float(a.data.get("at", 0.0)) + dt
        k = _clamp01(a.data["at"] / CLIMB_TIME)
        y0 = float(a.data.get("y0", ag.y))
        try:
            ag.x = float(tower.x)
            _perch(ag, y0 + (top - y0) * k)
        except Exception:
            pass
        _adjust(ag, "fatigue", 0.02 * dt)
        if k >= 1.0:
            a.phase = "watch"
            a.data["wt"] = 0.0
        return

    if a.phase == "descend":
        # The way down is a climb, not a step off the edge. 68 px of watchtower
        # is past STEP_FALL_MAX, so letting go at the top lands at ~350 px/s -
        # over FALL_LETHAL_SPEED - and the ladder they walked up would have
        # killed them on the way home.
        a.pose = "climb"
        _halt(ag)
        gy = ground_y(w, ag.x)
        a.data["dsc"] = float(a.data.get("dsc", 0.0)) + dt
        k = _clamp01(a.data["dsc"] / CLIMB_TIME)
        y1 = float(a.data.get("y1", top))
        _adjust(ag, "fatigue", 0.012 * dt)
        if k >= 1.0 or gy <= y1:
            _c_lookout(a, ag, w)
            a.done = True
            return
        try:
            ag.x = float(tower.x)
            _perch(ag, y1 + (gy - y1) * k)
        except Exception:
            pass
        return

    a.pose = "lookout"
    _halt(ag)
    try:
        ag.x = float(tower.x)
        _perch(ag, top)
    except Exception:
        pass
    a.data["wt"] = float(a.data.get("wt", 0.0)) + dt
    if int(a.data["wt"] / 4.0) % 2 == 0:
        ag.facing = 1
    else:
        ag.facing = -1
    _adjust(ag, "fatigue", 0.004 * dt)
    _adjust(ag, "warmth", 0.006 * dt)
    haz = hazards_of(w)
    if haz:
        h = min(haz, key=lambda z: abs(float(z["x"]) - float(ag.x)))
        try:
            setattr(w, "alarm", {"x": float(h["x"]), "kind": str(h["kind"]),
                                 "t": world_now(w)})
        except Exception:
            pass
        emit_speech(w, ag, "!")
        chronicle(w, f"{getattr(ag, 'name', 'The lookout')} spotted danger.")
        _start_climb_down(a, top)
        return
    if a.data["wt"] >= float(a.data.get("dur", 120.0)):
        _start_climb_down(a, top)


def _start_climb_down(a: Action, top: float) -> None:
    """End the watch by climbing down. The alarm, if any, is already raised.

    Shout first, then come down: the hazard branch has already set world.alarm
    and chronicled it, so the colony reacts on the same tick either way. If
    something more urgent than a ladder happens on the way, behaviour's
    override replaces the action and the cleanup hook puts him on the ground.
    """
    a.phase = "descend"
    a.pose = "climb"
    a.data["dsc"] = 0.0
    a.data["y1"] = float(top)


def _c_lookout(a: Action, ag: Any, w: Any) -> None:
    tower = structure_by_id(w, a.target)
    if tower is not None:
        tower.leave(int(getattr(ag, "id", 0)))
    if a.data.get("in"):
        # Off the tower and onto the ground in one step. This is the abrupt
        # path - an interrupted watch, a tower that burned down under him -
        # so it must assert the whole landing, not just y: dropping the perch
        # while leaving him above the ground would hand a man 68 px up to
        # gravity, and that lands at ~350 px/s, which is fatal.
        try:
            _plant(ag, ground_y(w, ag.x))
        except Exception:
            pass
    # Always clear this, even if the tower is gone: an agent left flagged as
    # inside a building that no longer exists would be invisible forever.
    try:
        ag.inside = None
    except Exception:
        pass
    a.data["in"] = 0


def _h_follow(a: Action, ag: Any, w: Any, dt: float) -> None:
    parent = agent_by_id(w, a.target)
    if parent is None or not getattr(parent, "alive", True):
        parent = _nearest_adult(w, ag)
        if parent is None:
            a.failed = True
            return
        a.target = getattr(parent, "id", None)
    if a.phase == "start":
        a.data["dur"] = rng_of(w).uniform(22.0, 50.0)
        a.phase = "follow"

    gap = float(parent.x) - float(ag.x)
    if abs(gap) > 34.0:
        a.pose = "walk"
        step_toward(ag, w, float(parent.x) - 22.0 * (1.0 if gap > 0 else -1.0), dt,
                    speed=WALK_SPEED * 1.12)
    else:
        a.pose = "idle"
        _halt(ag)
        _face(ag, gap)
        _adjust(ag, "morale", 0.008 * dt)
    if a.t >= float(a.data.get("dur", 30.0)):
        a.done = True


def _nearest_adult(w: Any, ag: Any) -> Any | None:
    best, best_d = None, float("inf")
    my_id = getattr(ag, "id", None)
    for o in alive_agents(w):
        if getattr(o, "id", None) == my_id:
            continue
        if str(getattr(o, "role", "")) == "child":
            continue
        d = abs(float(getattr(o, "x", 0.0)) - float(ag.x))
        if d < best_d:
            best_d, best = d, o
    return best


def _h_panic(a: Action, ag: Any, w: Any, dt: float) -> None:
    if a.phase == "start":
        rng = rng_of(w)
        a.data["dir"] = 1.0 if rng.random() < 0.5 else -1.0
        a.data["dur"] = rng.uniform(3.0, 7.0)
        a.data["flip"] = 0.0
        a.phase = "run"
    a.pose = "panic"
    a.data["flip"] = float(a.data.get("flip", 0.0)) + dt
    if a.data["flip"] > 0.9:
        a.data["flip"] = 0.0
        a.data["dir"] = -float(a.data.get("dir", 1.0))
        emit_speech(w, ag, "!")
    d = float(a.data.get("dir", 1.0))
    step_toward(ag, w, _clamp_x(float(ag.x) + d * 90.0), dt,
                speed=WALK_SPEED * 1.8, arrive=2.0)
    _adjust(ag, "morale", -0.045 * dt)
    _adjust(ag, "fatigue", 0.030 * dt)
    if a.t >= float(a.data.get("dur", 4.0)):
        a.done = True


# ===========================================================================
#  Farming - grows FOOD (complements ForageBerries, which strips wild bushes)
# ===========================================================================
def _material_at(world: Any, x: float) -> int:
    """Terrain material id under `x`, or -1 if the terrain cannot say."""
    terr = getattr(world, "terrain", None)
    fn = getattr(terr, "material_at", None)
    if callable(fn):
        try:
            return int(fn(_clamp_x(x)))
        except Exception:
            return -1
    return -1


def _crop_state(prop: Any) -> dict:
    st = getattr(prop, "state", None)
    return st if isinstance(st, dict) else {}


def _count_crops_near(world: Any, center: float,
                      radius: float = FARM_FIELD_RADIUS) -> int:
    """Living crops that count as *this* colony's field."""
    n = 0
    for p in props_of(world):
        if not prop_alive(p) or _prop_kind(p) != "crop":
            continue
        try:
            if abs(float(getattr(p, "x", 0.0)) - float(center)) <= radius:
                n += 1
        except (TypeError, ValueError):
            continue
    return n


def _nearest_ripe_crop(world: Any, x: float,
                       max_dist: float = FARM_REACH) -> Any | None:
    best, best_d = None, float("inf")
    for p in props_of(world):
        if not prop_alive(p) or _prop_kind(p) != "crop":
            continue
        if not _crop_state(p).get("ripe"):
            continue
        try:
            d = abs(float(getattr(p, "x", 0.0)) - float(x))
        except (TypeError, ValueError):
            continue
        if d < best_d:
            best_d, best = d, p
    return best if best is not None and best_d <= max_dist else None


def _till_col_ok(world: Any, cand: float) -> bool:
    """The single tillability test. Used by both the site picker and the
    behaviour feasibility check, so the two can never disagree - the bug where
    Farm was chosen 1449 times and completed 47 was exactly that disagreement,
    a random picker missing ground a grid scan swore was there."""
    if abs(slope_at(world, cand)) > MAX_SLOPE_WALK * 0.7:
        return False
    if _material_at(world, cand) not in (MAT_GRASS, MAT_DIRT):
        return False
    if find_prop(world, ("crop",), cand, max_dist=22.0) is not None:
        return False
    st = nearest_structure(world, None, cand, built_only=False)
    if st is not None and abs(float(getattr(st, "x", 1e9)) - cand) < 26.0:
        return False
    return True


def find_till_site(world: Any, center: float) -> float | None:
    """The nearest tillable column to `center`, or None if there is genuinely
    nowhere. A deterministic outward grid scan rather than random darts, so if a
    site exists it is found - which is what keeps feasibility and execution in
    step. Public so behaviour can ask 'is there anywhere to farm?' with the very
    same test the action will use."""
    for off in range(0, int(FARM_FIELD_RADIUS) + 1, 12):
        for cand in ((center + off, center - off) if off else (center,)):
            cx = _clamp_x(float(cand))
            if _till_col_ok(world, cx):
                return cx
    return None


# Back-compat alias for the old private name.
_till_site = find_till_site


def _plant_crop(world: Any, x: float) -> bool:
    """Break ground: spawn a fresh (unripe) crop on the surface at `x`."""
    src = getattr(world, "props", None)
    spawn = getattr(src, "spawn", None)
    if not callable(spawn):
        return False
    try:
        spawn("crop", float(x), float(ground_y(world, x)),
              state={"growth": 0.0, "ripe": False})
        return True
    except Exception:
        log.debug("_plant_crop failed", exc_info=True)
        return False


def _h_farm(a: Action, ag: Any, w: Any, dt: float) -> None:
    if a.phase == "start":
        center = colony_center(w)
        tx = None
        if _count_crops_near(w, center) < FARM_FIELD_SIZE:
            tx = _till_site(w, center)
        if tx is not None:
            a.data["mode"] = "plant"
            a.data["tx"] = float(tx)
            a.phase = "approach"
        else:
            crop = _nearest_ripe_crop(w, float(ag.x))
            if crop is None:
                a.failed = True         # field full and nothing ripe: do else
                return
            a.data["mode"] = "harvest"
            a.target = getattr(crop, "id", None)
            a.data["px"] = float(getattr(crop, "x", ag.x))
            a.phase = "approach"

    if a.phase == "approach":
        if a.data.get("mode") == "plant":
            a.pose = "walk"
            rem = step_toward(ag, w, float(a.data.get("tx", ag.x)), dt)
            if rem <= REACH:
                a.phase = "till"
                a.data["kt"] = 0.0
            elif a.t > 75.0:
                a.failed = True
            return
        # harvest: re-resolve the crop each tick - a wildfire may have taken it
        p = prop_by_id(w, a.target) if a.target is not None else None
        if p is None:
            p = find_prop(w, ("crop",), float(a.data.get("px", ag.x)), max_dist=40.0)
        if p is None or not prop_alive(p):
            a.failed = True
            return
        a.data["px"] = float(getattr(p, "x", a.data.get("px", ag.x)))
        a.pose = "walk"
        rem = step_toward(ag, w, float(a.data["px"]), dt)
        if rem <= REACH:
            a.phase = "harvest"
            a.data["ht"] = 0.0
        elif a.t > 75.0:
            a.failed = True
        return

    if a.phase == "till":
        a.pose = "build"
        _halt(ag)
        _face(ag, float(a.data.get("tx", ag.x)) - float(ag.x))
        a.data["kt"] = float(a.data.get("kt", 0.0)) + dt
        _adjust(ag, "fatigue", 0.010 * dt)
        if a.data["kt"] >= FARM_TILL_SEC / max(0.3, _work_rate(ag)):
            if _plant_crop(w, float(a.data.get("tx", ag.x))):
                if rng_of(w).random() < 0.30:
                    chronicle(w, f"{getattr(ag, 'name', 'Someone')} broke new ground.")
                _adjust(ag, "morale", 0.04)
                a.done = True
            else:
                a.failed = True
        return

    if a.phase == "harvest":
        a.pose = "forage"
        _halt(ag)
        p = prop_by_id(w, a.target) if a.target is not None else None
        if p is None:
            p = find_prop(w, ("crop",), float(a.data.get("px", ag.x)), max_dist=30.0)
        # Crop gone (burned) or picked already: bail out softly, delivering
        # anything we happen to be holding.
        if p is None or not prop_alive(p) or not _crop_state(p).get("ripe"):
            if _has_load(ag, a):
                a.phase = "deliver"
            else:
                a.failed = True
            return
        _face(ag, float(getattr(p, "x", ag.x)) - float(ag.x))
        a.data["ht"] = float(a.data.get("ht", 0.0)) + dt
        if a.data["ht"] < HARVEST_TIME:
            return
        try:
            p.state["ripe"] = False
            p.state["growth"] = 0.0          # perennial: it will grow back
        except Exception:
            pass
        _carry_add(ag, a, RES_FOOD, FARM_HARVEST_FOOD)
        _adjust(ag, "fatigue", 0.02)
        if rng_of(w).random() < 0.35:
            chronicle(w, f"{getattr(ag, 'name', 'Someone')} brought in the harvest.")
        a.phase = "deliver"
        return

    if a.phase == "deliver":
        if _deposit_step(a, ag, w, dt):
            a.done = True
        return

    a.phase = "start"


# ===========================================================================
#  Mining - a sustained dig (distinct from GatherStone's quick rock-grab)
# ===========================================================================
def _prop_home_d(prop: Any, home: float) -> float:
    """Signed px between a prop and the colony centre; 0.0 if unreadable.

    0.0 on junk is the conservative answer here: it reads as "this thing is
    standing in the middle of town", so an unreadable boulder is excluded from
    quarrying rather than silently dug up next to the huts.
    """
    try:
        return float(getattr(prop, "x", home)) - home
    except (TypeError, ValueError):
        return 0.0


def _quarry_column(world: Any, x: float,
                   max_dist: float = MINE_MAX_WALK) -> float | None:
    """The best MAT_STONE column to open a quarry at, or None.

    Replaces a nearest-first scan, which sited every quarry under the
    settlement: a miner decides to dig while standing in town, the first stone
    column outward is a few paces away, and _dig_divot then deforms that ground
    once per yield tick. The colony mined its own floor out.

    Ranks legal columns instead of taking the first. Legal means standable (the
    old cliff test, unchanged - a miner must not commit to a spot it can only
    reach by falling off it) and at least MINE_KEEP_OUT from settlement_center()
    - the settlement, with the barricades left out, because a keep-out ring
    measured from a centre that a single outwork has dragged 300 px is a ring
    round the wrong place. See :func:`settlement_center`.
    Among those, the score prefers a column further OUT - toward the nearer rim
    of the world - with a smaller term for the walk so two comparable directions
    do not send miners across the map.

    Falls back, in order: legal sites, then any standable stone inside the walk
    radius, then None. The fallback is not a nicety - stone gates huts, walls
    and the whole stone-hut tier, so a map whose only stone lies in the middle
    of town must still be mineable. It just stops being the default.
    """
    terr = getattr(world, "terrain", None)
    if not callable(getattr(terr, "material_at", None)):
        return None
    home = settlement_center(world)
    # How far the miner already stands from the nearer rim of the WORLD. The
    # edge term below is measured as a change against this, not as an absolute
    # fraction of the map - see the comment at the score, it is the difference
    # between a term that ranks and one that is silently dead.
    rim_here = min(max(0.0, float(x)), max(0.0, float(WORLD_W) - float(x)))
    window = max(1.0, float(max_dist))
    best: tuple[float, float] | None = None      # (score, x)
    fallback: tuple[float, float] | None = None  # (walk, x) - nearest legal-ish
    step = 5.0
    d = 0.0
    while d <= max_dist:
        cands = (x,) if d == 0.0 else (x + d, x - d)
        for c in cands:
            cx = _clamp_x(c)
            if _material_at(world, cx) != MAT_STONE:
                continue
            if abs(slope_at(world, cx)) > MAX_SLOPE_CLIMB:
                continue
            walk = abs(cx - x)
            if fallback is None or walk < fallback[0]:
                fallback = (walk, float(cx))
            if abs(cx - home) < MINE_KEEP_OUT:
                continue
            # Both terms are fractions of their own range, so the weights are
            # comparable numbers rather than an accident of units. Lower wins.
            #
            # The edge term is normalised over the REACHABLE WINDOW, not over
            # half the map, and that is a decision rather than a rename. Written
            # as `min(cx, W - cx) / (W * 0.5)` it was fine while W was 1600:
            # half the map (800) and MINE_MAX_WALK (900) were the same size, so
            # the term swung most of 0..1 across everything a miner could reach.
            # At WORLD_W = 6400 the denominator is 3200 while the miner can
            # still only walk 900, so the term varies by at most 0.28 - and for
            # a colony seated anywhere near mid-map it sits pinned at ~1.0 and
            # stops ranking anything at all. Quarries would silently revert to
            # walk-ranked-only, i.e. back to digging the nearest legal column,
            # which is most of the bug this function exists to fix.
            #
            # So: 0.5 at the miner's own feet, 0.0 a full walk further OUT
            # (toward the nearer world rim), 1.0 a full walk further IN. Same
            # meaning - "prefer outward" - with its full dynamic range restored
            # at any map width and at any position on the map. A miner standing
            # exactly at mid-map gets 0.5 in both directions, which is right:
            # neither way is more outward, so the walk term decides.
            edge_frac = _clamp01(
                0.5 + (min(cx, float(WORLD_W) - cx) - rim_here) / (2.0 * window)
            )
            walk_frac = walk / window
            score = (MINE_EDGE_WEIGHT * edge_frac
                     + MINE_WALK_WEIGHT * walk_frac)
            if best is None or score < best[0]:
                best = (score, float(cx))
        d += step
    if best is not None:
        return best[1]
    return fallback[1] if fallback is not None else None


def _quarry_too_steep(world: Any, x: float, span: float = MINE_DIVOT_HALF) -> bool:
    """True if cutting at *x* would leave ground nobody can walk out of.

    Measured off the LIVE terrain, every tick, which is the whole point. The
    previous guard was a per-Action budget in ``a.data["divot"]``, and its
    docstring promised "never a cliff" - but a.data is born with each Action, so
    every fresh mining session started a new 8 px allowance AT THE SAME COLUMN.
    Twenty sessions on one spot is 160 px of shaft. Combined with the old
    nearest-first siting, which put that spot in the middle of the settlement,
    a colony reliably excavated a pit in its own front yard and then fell into
    it. This is that bug: not the depth of one dig, the absence of any memory
    between digs.

    A gradient test needs no memory. The terrain already knows how steep it is,
    so however many sessions have come before, the answer is always current.

    The threshold is MINE_SAFE_GRADIENT and NOT MAX_SLOPE_WALK, which was the
    first attempt and was measured refusing to dig anywhere at all: on a real
    map the flattest MAT_STONE column outside the settlement already sits at
    |slope| 1.60, well past the 0.9 walk limit, so a walk-limit gate stops stone
    production dead. The honest requirement is not "a quarry must be flat" - it
    is "mining must never CREATE a cliff". 2.0 leaves real headroom to dig on
    ordinary stone while stopping a clear margin short of MAX_SLOPE_CLIMB (2.6),
    which is where a fall starts being lethal.
    """
    # Swept rather than sampled at three points: the rim is what gets steep and
    # it does not sit at a fixed offset once cuts start overlapping.
    step = 2.0
    probe = -abs(span)
    while probe <= abs(span):
        if abs(slope_at(world, x + probe)) > MINE_SAFE_GRADIENT:
            return True
        probe += step
    return False


def _dig_divot(world: Any, a: "Action", x: float) -> float:
    """Cut the quarry face a touch deeper and a touch along. Returns the new x.

    The cut ADVANCES as it deepens, which is both what the shape should be and
    what makes it safe. Digging straight down at one column is how you get a
    shaft; stepping the face sideways by MINE_FACE_STEP each time turns the same
    excavation into a ramp leading down into the workings - a quarry rather than
    a well, and walkable by construction.

    The step runs AWAY from the colony, so the cut opens outward toward the map
    edge and the near lip - the side the villagers arrive from - stays the
    shallow end.
    """
    terr = getattr(world, "terrain", None)
    fn = getattr(terr, "deform", None)
    if not callable(fn):
        return x
    if _quarry_too_steep(world, x):
        # Already as deep as this face safely goes. Stop cutting rather than
        # keep taking stone out of a wall that is about to become a cliff.
        a.data["spent"] = True
        return x
    cur = float(a.data.get("divot", 0.0))
    dyy = min(MINE_DIVOT_PER, max(0.0, MINE_DIVOT_MAX - cur))
    if dyy <= 0.0:
        a.data["spent"] = True
        return x
    x0, x1 = int(x - MINE_DIVOT_HALF), int(x + MINE_DIVOT_HALF)
    try:
        # positive dy digs the ground DOWN; a wide bowl keeps the walls gentle.
        fn(x0, x1, float(dyy), "bowl")
        # ...gentle in the middle. The STEEP part of a bowl is its rim, where the
        # cut meets ground nobody has touched, and consecutive overlapping bowls
        # compound that rim into a trench wall. Testing the slope BEFORE digging
        # bounds where the quarry starts and says nothing about what the cut
        # leaves behind - measured, that let 30 sessions reach |slope| 4.47, well
        # past the lethal 2.6. So the cut is PROPOSED and then inspected, and put
        # back if it went too far. deform is linear in dy, so pushing the same
        # span up by the same amount is an exact undo.
        if _quarry_too_steep(world, x, span=MINE_DIVOT_HALF * 3):
            fn(x0, x1, -float(dyy), "bowl")
            a.data["spent"] = True
            return x
        a.data["divot"] = cur + dyy
    except Exception:
        log.debug("_dig_divot failed", exc_info=True)
        return x
    away = 1.0 if x >= colony_center(world) else -1.0
    return _clamp_x(x + away * MINE_FACE_STEP)


def _h_mine(a: Action, ag: Any, w: Any, dt: float) -> None:
    if a.phase == "start":
        # Boulders get the same keep-out as a ground quarry. It is the digging
        # that deforms the terrain, not what is being dug - _dig_divot runs in
        # the dig phase for both sources - so a boulder sitting in the middle of
        # the settlement is exactly as bad a place to excavate as bare stone
        # there. Preferred, then any boulder, then ground; the two-pass shape
        # keeps the old behaviour reachable when a map has no outlying boulder.
        home = colony_center(w)
        in_town = [p for p in props_of(w)
                   if _prop_kind(p) == "boulder"
                   and abs(_prop_home_d(p, home)) < MINE_KEEP_OUT]
        b = find_prop(w, ("boulder",), float(ag.x), max_dist=MINE_MAX_WALK,
                      claimant=getattr(ag, "id", None), exclude=in_town)
        if b is None:
            b = find_prop(w, ("boulder",), float(ag.x), max_dist=520.0,
                          claimant=getattr(ag, "id", None))
        if b is not None:
            a.target = getattr(b, "id", None)
            a.data["mx"] = float(getattr(b, "x", ag.x))
            a.data["src"] = "boulder"
            claim_prop(w, b, ag)      # two miners never share one boulder
        else:
            sx = _quarry_column(w, float(ag.x))
            if sx is None:
                a.failed = True
                return
            a.target = None
            a.data["mx"] = float(sx)
            a.data["src"] = "ground"
        a.data["dig_t"] = 0.0
        a.data["yt"] = 0.0
        a.data["divot"] = 0.0
        a.phase = "approach"

    if a.phase == "approach":
        try:
            ag.mining = False
        except Exception:
            pass
        a.pose = "walk"
        rem = step_toward(ag, w, float(a.data.get("mx", ag.x)), dt, arrive=REACH)
        if rem <= REACH:
            a.phase = "dig"
            a.data["dig_t"] = 0.0
            a.data["yt"] = 0.0
        elif a.t > 75.0:
            a.failed = True
        return

    if a.phase == "dig":
        mx = float(a.data.get("mx", ag.x))
        if a.data.get("src") == "boulder" and a.target is not None:
            claim_prop(w, prop_by_id(w, a.target), ag)   # hold it while digging
        _halt(ag)
        _face(ag, mx - float(ag.x))
        a.data["dig_t"] = float(a.data.get("dig_t", 0.0)) + dt
        # Alternate the swing so the figure reads as working, not frozen.
        a.pose = "chop" if int(a.data["dig_t"] * 2.0) % 2 == 0 else "build"
        # Transient flag for the renderer's dust; runtime-only, never persisted.
        try:
            ag.mining = True
        except Exception:
            pass
        _adjust(ag, "fatigue", 0.012 * dt)
        a.data["yt"] = float(a.data.get("yt", 0.0)) + dt
        if a.data["yt"] >= MINE_YIELD_SEC / max(0.3, _work_rate(ag)):
            a.data["yt"] = 0.0
            stock_add(w, RES_STONE, MINE_YIELD_STONE)
            # The face moves, so the miner works along it rather than standing in
            # one spot deepening a shaft. mx is written back so the next swing,
            # the next divot and the pose all follow the cut.
            a.data["mx"] = _dig_divot(w, a, mx)
            if rng_of(w).random() < 0.22:
                chronicle(w, f"{getattr(ag, 'name', 'Someone')} hewed stone from the earth.")
        # NOTE "spent" stops the CUTTING, not the session. Ending the session
        # here was the first shape and it was wrong twice over: the miner walked
        # away after a single yield, and stone output collapsed with it. The
        # divot is scenery - stock_add above grants the stone whether or not the
        # ground moved - so a face that cannot safely be cut any deeper is a
        # perfectly good face to keep working. They hew at it for the usual
        # session and simply stop making the hole worse.
        if a.data["dig_t"] >= MINE_SESSION_SEC:
            try:
                ag.mining = False
            except Exception:
                pass
            release_claim(w, a.target, ag)
            a.done = True
        return

    a.phase = "start"


#: Seconds spent stooping over each piece. Short - this is meant to read as
#: someone moving along picking things up, not as a work session.
CLEAN_PICK_SEC = 0.5
#: How far from the pile's centre a sweeper will chase a stray piece. Just over
#: the cluster radius, so he clears the pile he came for and does not wander off
#: across the map following a trail of single items.
CLEAN_RANGE = LITTER_CLUSTER_R + 30.0
#: Give-up timers. Both generous relative to the work, both mandatory: this
#: action can be started on a pile behind a chasm wall, and an unattended run
#: must not park a villager against a cliff forever.
CLEAN_WALK_TIMEOUT = 75.0
CLEAN_TOTAL_TIMEOUT = 150.0


def _nearest_litter(w: Any, x: float, cx: float, aid: Any) -> Any | None:
    """Nearest unclaimed piece of litter to `x` that still belongs to the pile
    centred on `cx`. Returns None once the pile is clear."""
    best = None
    best_d = float("inf")
    now = world_now(w)
    for p in props_of(w):
        if _prop_kind(p) != "litter" or not prop_alive(p):
            continue
        try:
            px = float(getattr(p, "x", 0.0))
        except (TypeError, ValueError):
            continue
        if abs(px - float(cx)) > CLEAN_RANGE:
            continue
        if aid is not None and prop_claimed_by_other(w, p, aid, now):
            continue
        d = abs(px - float(x))
        if d < best_d:
            best_d, best = d, p
    return best


def _cluster_taken(w: Any, cx: float, aid: Any) -> bool:
    """Is somebody else already sweeping the pile centred on `cx`?

    Asking "is any piece of this pile claimed" rather than "is the anchor
    claimed": a sweeper re-claims whichever piece he is walking to on every
    tick, so he always holds exactly one piece of his own pile, and the piece he
    holds moves as he works along it. Testing only the anchor would let a second
    villager shadow the same heap the moment the first picked the anchor up.
    """
    if aid is None:
        return False
    now = world_now(w)
    for p in props_of(w):
        if _prop_kind(p) != "litter" or not prop_alive(p):
            continue
        try:
            if abs(float(getattr(p, "x", 0.0)) - float(cx)) > CLEAN_RANGE:
                continue
        except (TypeError, ValueError):
            continue
        if prop_claimed_by_other(w, p, aid, now):
            return True
    return False


def _h_clean(a: Action, ag: Any, w: Any, dt: float) -> None:
    """Sweep the densest pile of litter and burn it in the nearest firepit.

    Four phases: pick a pile, walk to it, work along it filling both arms, then
    haul the load to a fire and tip it in. The load is :data:`RES_GARBAGE`, which
    ``stock_add`` refuses outright - it can only ever end up burned or, if this
    is interrupted, back on the ground via :func:`_shed_litter`.

    Deliberately fails rather than improvising whenever the premise stops
    holding: no dense pile, nowhere to burn it, hands already full. This is the
    lowest-value job in the colony and it should evaporate at the first excuse.
    """
    aid = getattr(ag, "id", None)

    if a.phase == "start":
        held = int(getattr(ag, "carry_qty", 0) or 0)
        if held > 0 and getattr(ag, "carrying", None) == RES_GARBAGE:
            # He is already holding a swept load - a previous run that got
            # interrupted between the heap and the fire. Finish that errand
            # rather than starting a new one; nothing else in the colony knows
            # what to do with garbage, so failing here would strand it in his
            # hands until something abandoned the action for him.
            a.data["got"] = held
            a.data["burn_t0"] = float(a.t)
            a.data.setdefault("cx", float(getattr(ag, "x", 0.0)))
            a.phase = "burn"
            return                  # pick the errand up again next tick
        if held > 0:
            # Hands full of something the colony wants. `_carry_add` would stash
            # the rubbish behind it and the tip-in below reads the hand, so the
            # load would arrive at the fire invisible.
            a.failed = True
            return
        # THE SAME QUESTION THE "burn" PHASE WILL ASK, and it has to be, because
        # the two used to disagree. This was a raw `nearest_structure(w,
        # "firepit", ..)` while the phase seventy lines below picks its fire
        # through `fire_for` - so for a hermit the premise passed on the
        # COLONY's hearth and the delivery then resolved to his own or to
        # nothing at all. Unreachable today only because `_hermit_bias` zeroes
        # CleanLitter, which is a gate held shut by a second gate somewhere
        # else: exactly how one rots. One function, asked twice.
        if fire_for(ag, w) is None:
            a.failed = True
            return
        pile = free_litter_cluster(w, aid)
        anchor = _nearest_litter(w, pile[0], pile[0], aid) if pile else None
        if pile is None or anchor is None:
            a.failed = True         # nothing dense enough, or all of it taken
            return
        cx = pile[0]
        a.data["cx"] = float(cx)
        a.data["got"] = 0
        a.data["pt"] = 0.0
        a.target = getattr(anchor, "id", None)
        claim_prop(w, anchor, ag)   # this pile is mine
        a.phase = "approach"

    cx = float(a.data.get("cx", getattr(ag, "x", 0.0)))

    if a.phase == "approach":
        a.pose = "walk"
        rem = step_toward(ag, w, cx, dt, arrive=REACH)
        if rem <= REACH:
            a.phase = "collect"
            a.data["pt"] = 0.0
        elif a.t > CLEAN_WALK_TIMEOUT:
            a.failed = True
        return

    if a.phase == "collect":
        p = _nearest_litter(w, float(ag.x), cx, aid)
        got = int(a.data.get("got", 0))
        if p is None or got >= CARRY_CAP or a.t > CLEAN_TOTAL_TIMEOUT:
            release_claim(w, a.target, ag)
            if got > 0:
                a.phase = "burn"
                # The haul gets its own clock. Sharing `a.t` would mean a sweep
                # that only just made the total timeout arrives at the burn
                # phase already expired and dumps the load on the spot.
                a.data["burn_t0"] = float(a.t)
                return
            a.failed = True
            return
        # Hold whatever piece is next, so a second sweeper picks a different
        # pile rather than shadowing this one.
        a.target = getattr(p, "id", None)
        claim_prop(w, p, ag)
        px = float(getattr(p, "x", ag.x))
        if abs(px - float(ag.x)) > REACH:
            a.pose = "walk"
            step_toward(ag, w, px, dt, arrive=REACH)
            a.data["pt"] = 0.0
            return
        a.pose = "forage"
        _halt(ag)
        _face(ag, px - float(ag.x))
        a.data["pt"] = float(a.data.get("pt", 0.0)) + dt
        if a.data["pt"] < CLEAN_PICK_SEC / max(0.3, _work_rate(ag)):
            return
        a.data["pt"] = 0.0
        release_claim(w, a.target, ag)
        _kill_prop(w, p)
        _carry_add(ag, a, RES_GARBAGE, 1)
        a.data["got"] = got + 1
        return

    if a.phase == "burn":
        # The fire is tracked in `data`, not in `target`: `target` still holds
        # the last claimed litter id, and `_c_clean` releases that claim by it.
        fire = structure_by_id(w, a.data.get("fire"))
        if not _fire_ok(fire, ag):
            fire = fire_for(ag, w)      # the mutual gate - see `fire_for`
            a.data["fire"] = getattr(fire, "id", None) if fire is not None else None
            # A REDIRECT IS A NEW WALK. This phase already re-walks every tick,
            # so unlike `_h_warm`/`_h_cook` there is no phase to rewind - but the
            # CLEAN_WALK_TIMEOUT below is measured from `burn_t0`, and a swap
            # late in the errand would otherwise time the new walk out on its
            # first step and shed the load. Same rule as `_regate_fire`.
            a.data["burn_t0"] = float(a.t)
        if fire is None:
            # The fire burned down while we were sweeping. Put the load back on
            # the ground rather than deleting it.
            _shed_litter(w, float(getattr(ag, "x", 0.0)),
                         int(getattr(ag, "carry_qty", 0) or 0))
            _clear_carry(ag)
            a.done = True
            return
        a.pose = "carry"
        rem = step_toward(ag, w, float(fire.x), dt, arrive=FIRE_REACH,
                          speed=WALK_SPEED * 0.92)
        if rem > FIRE_REACH:
            if a.t - float(a.data.get("burn_t0", 0.0)) > CLEAN_WALK_TIMEOUT:
                _shed_litter(w, float(getattr(ag, "x", 0.0)),
                             int(getattr(ag, "carry_qty", 0) or 0))
                _clear_carry(ag)
                a.done = True
            return
        _halt(ag)
        qty = int(getattr(ag, "carry_qty", 0) or 0)
        taken = 0
        # Was the pit already burning rubbish? Only a load that *starts* a
        # bonfire gets a line - topping one up is the same event continuing, and
        # a keen colony tips a load in every couple of minutes.
        fresh = float(getattr(fire, "garbage_left", 0.0) or 0.0) <= 0.0
        fn = getattr(fire, "feed_garbage", None)
        if callable(fn):
            try:
                taken = int(fn(qty))
            except Exception:
                taken = 0
        if taken > 0:
            if fresh:
                chronicle(w, f"{getattr(ag, 'name', 'Someone')} tipped a "
                             f"load of swept-up rubbish onto the fire.")
            _adjust(ag, "morale", 0.03)
        # Whatever the pit would not take stays IN HAND. It emphatically does not
        # go back on the ground here: the pit refuses anything past
        # BONFIRE_GARBAGE_CAP, so shedding the remainder at the villager's feet
        # turned the whole job into a treadmill - measured, 1012 of 1181 pieces
        # swept up over 45 minutes were tipped out again at the fire, one step
        # from where they would be picked up next. Carrying it means the next
        # delivery finishes the load once the blaze has burned down, which is
        # also just what a person would do.
        #
        # Only a load with nowhere to go at all is put down (the walk-timeout
        # branch above), because a villager holding rubbish forever would never
        # pick up a resource again.
        if taken >= qty:
            _clear_carry(ag)
        elif taken > 0:
            _carry_take(ag, taken)
        a.done = True
        return

    a.phase = "start"


def _clear_carry(ag: Any) -> None:
    """Empty the hands. Only ever used where the load has been accounted for."""
    try:
        ag.carrying = None
        ag.carry_qty = 0
    except Exception:
        pass


def _carry_take(ag: Any, qty: int) -> None:
    """Remove *qty* from the load, keeping the remainder in hand.

    Used where a destination accepts only part of what was brought - the fire
    refusing rubbish past its cap - so the rest is still carried rather than
    dumped. Empties the hands if that takes the load to nothing, because a
    carrying kind with a zero quantity reads as "holding something" to the
    renderer and to every ``_has_load`` check.
    """
    try:
        left = int(getattr(ag, "carry_qty", 0) or 0) - max(0, int(qty))
        if left > 0:
            ag.carry_qty = left
        else:
            _clear_carry(ag)
    except Exception:
        pass


def _c_clean(a: Action, ag: Any, w: Any) -> None:
    """Release the claimed piece and put any swept load back down.

    Mirrors :func:`_c_gather` on the claim, and goes one step further because
    this action is the only one that carries something the stockpile will not
    accept: a sweeper who bolts from a wolf has to *drop* his armful, or it
    would ride around in his hands forever (nothing else knows what to do with
    garbage) and the pile he took it from would read as cleaned.
    """
    try:
        release_claim(w, a.target, ag)
    except Exception:
        pass
    try:
        if getattr(ag, "carrying", None) == RES_GARBAGE:
            _shed_litter(w, float(getattr(ag, "x", 0.0)),
                         int(getattr(ag, "carry_qty", 0) or 0))
            _clear_carry(ag)
    except Exception:
        pass


def _c_mine(a: Action, ag: Any, w: Any) -> None:
    """Drop the transient dust flag and free the boulder when a dig is cut short."""
    try:
        ag.mining = False
    except Exception:
        pass
    try:
        release_claim(w, a.target, ag)
    except Exception:
        pass


_HANDLERS: dict[str, Callable[[Action, Any, Any, float], None]] = {
    "Wander": _h_wander,
    "GatherWood": _h_gather,
    "GatherStone": _h_gather,
    "ForageBerries": _h_gather,
    "HaulToStockpile": _h_haul,
    "BuildStructure": _h_build,
    "UpgradeStructure": _h_upgrade,
    "RepairStructure": _h_repair,
    "Eat": _h_eat,
    "Sleep": _h_sleep,
    "WarmAtFire": _h_warm,
    "CookFood": _h_cook,
    "PlantSapling": _h_plant,
    "Farm": _h_farm,
    "Mine": _h_mine,
    "Converse": _h_converse,
    "Celebrate": _h_celebrate,
    "Mourn": _h_mourn,
    "FleeFrom": _h_flee,
    "ClimbTo": _h_climb,
    "Lookout": _h_lookout,
    "FollowParent": _h_follow,
    "Panic": _h_panic,
    "CleanLitter": _h_clean,
}

_CLEANUP: dict[str, Callable[[Action, Any, Any], None]] = {
    "BuildStructure": _c_build,
    "Sleep": _c_sleep,
    "Lookout": _c_lookout,
    "Mine": _c_mine,
    "GatherWood": _c_gather,
    "GatherStone": _c_gather,
    "ForageBerries": _c_gather,
    "CleanLitter": _c_clean,
}

# Sanity: every advertised kind must have a real handler.
assert set(ACTION_KINDS) == set(_HANDLERS), "ACTION_KINDS/_HANDLERS mismatch"


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
        def ground_y(self, x: float) -> float:
            return 600.0 + 40.0 * math.sin(float(x) / 180.0)

        def slope(self, x: float) -> float:
            return (40.0 / 180.0) * math.cos(float(x) / 180.0)

    class _Agent:
        def __init__(self, aid: int, x: float) -> None:
            self.id = aid
            self.name = f"Agent{aid}"
            self.color = (200, 200, 200)
            self.x = x
            self.y = 600.0
            self.vx = self.vy = 0.0
            self.facing = 1
            self.hunger = self.fatigue = self.warmth = 0.2
            self.morale = 0.6
            self.carrying = None
            self.carry_qty = 0
            self.role = "gatherer"
            self.generation = 0
            self.action = None
            self.alive = True
            self.anim_t = 0.0

    class _World:
        def __init__(self) -> None:
            self.terrain = _Terrain()
            self.structures = StructureRegistry()
            self.props = [_Prop(1, "tree", 500.0), _Prop(2, "rock", 700.0),
                          _Prop(3, "bush", 620.0)]
            self.stockpile: dict[str, int] = {}
            self.agents: list[Any] = []
            self.world_time = 0.0
            self.rng = random.Random(7)
            self.lines: list[str] = []

        def chronicle(self, text: str) -> None:
            self.lines.append(text)

    w = _World()
    a1 = _Agent(1, 400.0)
    w.agents.append(a1)
    w.structures.create("stockpile", 460.0, w.terrain.ground_y(460.0), built=True)

    act = make_action("GatherWood")
    for _ in range(4000):
        act.update(a1, w, 1 / 30)
        if act.finished:
            break
    print("gather:", act, "stock:", w.stockpile)

    site = w.structures.create("firepit", 480.0, w.terrain.ground_y(480.0))
    w.stockpile[RES_WOOD] = 20
    w.stockpile[RES_STONE] = 20
    b = make_action("BuildStructure", target=site.id)
    for i in range(6000):
        b.update(a1, w, 1 / 30)
        if i == 300:                      # mid-build save/load round trip
            b = Action.from_dict(b.to_dict())
        if b.finished:
            break
    print("build:", b, "built:", site.built, "chronicle:", w.lines)

    for kind in ACTION_KINDS:
        act = make_action(kind, ty=520.0, fx=400.0, radius=80.0)
        for _ in range(600):
            act.update(a1, w, 1 / 30)
            if act.finished:
                break
        rt = Action.from_dict(act.to_dict())
        assert rt.kind == act.kind
        print(f"  {kind:16s} -> {act.phase:9s} done={act.done} failed={act.failed}")
    print("junk action:", Action.from_dict({"kind": "Nope"}))

    # ---- ledges: crossable by climbing down, and never a silent standstill --
    from .entities import Stickman

    class _Cliff:
        """Flat, a 150px face over 6px of run, then flat again."""

        def ground_y(self, x: float) -> float:
            x = float(x)
            if x <= 600.0:
                return 500.0
            if x >= 606.0:
                return 650.0
            return 500.0 + 150.0 * (x - 600.0) / 6.0

        def slope(self, x: float) -> float:
            return 25.0 if 600.0 < float(x) < 606.0 else 0.0

    cw = _World()
    cw.terrain = _Cliff()
    walker = Stickman(id=9, name="Ledge", x=560.0, y=500.0)
    for _ in range(int(30 * 40)):
        step_toward(walker, cw, 900.0, 1 / 30)
        walker.apply_physics(1 / 30, cw.terrain)
        if walker.x > 640.0:
            break
    print(f"ledge: x={walker.x:.1f} y={walker.y:.1f} alive={walker.alive}")
    assert walker.alive, "climbing down a 150px ledge should not kill anyone"
    assert walker.x > 640.0, f"never got across the ledge (x={walker.x:.1f})"

    # An agent that cannot get onto the face at all must abandon the goal
    # rather than stand at the lip until the action's 90s timeout.
    stuck = _Agent(2, 600.5)          # on the face; a stub has no descend_step
    stuck.action = make_action("Wander")
    for _ in range(int(30 * (BLOCKED_GIVE_UP + 0.5))):
        _ledge_step(stuck, cw, 900.0, 1.0, 1 / 30, cw.terrain.ground_y(stuck.x))
    assert stuck.action.failed, "a blocked agent never gave its goal up"
    print(f"blocked: goal abandoned after {BLOCKED_GIVE_UP}s")
