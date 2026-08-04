"""Props: the world's scenery and harvestable objects.

Pure python + numpy.  **No pygame** - render/ turns these into sprites, but the
behaviour lives here so the sim runs headless.

Kinds
-----
``tree``     multi-hit choppable, falls over ~1.2 s, yields 3-5 wood
``sapling``  grows into a tree over ~4 minutes of sim time
``rock``     multi-hit mineable, yields stone
``bush``     0..3 berries, regrow on a timer
``boulder``  sits until nudged, then rolls downhill damaging what it hits
``water``    a pond filling a terrain basin (decorative + flood anchor)
``grave``    left where a stickman died, permanent
``scorch``   burn scar left by fire or lightning, fades away
``litter``   rubbish a villager dropped; inert, never a resource node, only a
             cleanup crew or the recycling cap takes it away

Driving it from World.tick
--------------------------
``registry.tick(terrain, rng, dt)`` (or the module level :func:`tick_props`)
advances every behaviour and returns a list of **event dicts**.  Each has a
``"type"`` key; the world is expected to react to these and ignore the ones it
does not care about::

    tree_felled      id kind x y wood      tree finished falling, prop removed
    sapling_grown    id kind x y           sapling became a tree (same id)
    berries_grown    id kind x y berries   a bush regrew one berry
    prop_ignited     id kind x y           caught fire
    fire_spread      id kind x y src       caught fire *from* prop ``src``
    prop_burned_out  id kind x y           burned away, scorch left, removed
    rock_depleted    id kind x y           mined out, removed
    boulder_rolling  id kind x y vx        moving this tick (see nudge())
    boulder_impact   id kind x y force     hit something at x (world hurts agents)
    boulder_stopped  id kind x y
    prop_regrown     id kind x y target    land reseeded itself back toward target
    prop_destroyed   id kind x y cause     removed for any other reason

Actions call the mutators directly: :func:`chop`, :func:`mine`,
:func:`harvest_berries`, :func:`ignite`, :func:`nudge`, :func:`plant_sapling`,
:func:`place_grave`, :func:`place_scorch`.

Headcount-scaled recovery
-------------------------
``reg.tick(world, dt)`` reads ``world.regrowth_factor()`` (1.0 at MIN_POP,
rising to REGROW_MAX) and multiplies every **natural recovery** rate by it:
sapling growth, berry regrow, and the reseeding in :func:`_regrow`.  A colony
of ten therefore repopulates the map roughly three times as fast as a pair,
which is what stops ten mouths from stripping the world and starving.

Nothing *destructive* is scaled - fire spread, burn-out, tree falls, boulders
and scorch fading all run at their own fixed rate regardless of headcount.
Call shapes without a world (``reg.tick(terrain, rng, dt)``) simply use 1.0.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np

from ..constants import (
    LITTER_DROP_SEC,
    LITTER_MAX,
    MAT_ASH,
    MAT_DIRT,
    MAT_GRASS,
    MAT_SAND,
    MAT_STONE,
)

try:  # pragma: no cover - constants predate the population director
    from ..constants import REGROW_MAX as _REGROW_MAX
except ImportError:  # pragma: no cover
    _REGROW_MAX = 3.0

if TYPE_CHECKING:  # pragma: no cover - import cycle free at runtime
    from .terrain import Terrain

__all__ = [
    "Prop",
    "PropRegistry",
    "tick_props",
    "scatter",
    "chop",
    "mine",
    "harvest_berries",
    "ignite",
    "extinguish",
    "nudge",
    "plant_sapling",
    "place_grave",
    "place_scorch",
    "drop_litter",
    "litter_count",
    "growth_factor",
    "KINDS",
    "FLAMMABLE",
    "DEFAULT_COUNTS",
]

# --------------------------------------------------------------------- kinds --

KIND_TREE = "tree"
KIND_SAPLING = "sapling"
KIND_ROCK = "rock"
KIND_BUSH = "bush"
KIND_BOULDER = "boulder"
KIND_WATER = "water"
KIND_GRAVE = "grave"
KIND_SCORCH = "scorch"
KIND_CROP = "crop"        # a farmed food plant; grows, is harvested, regrows
KIND_LITTER = "litter"    # inert rubbish a villager dropped; fuel, not a resource

KINDS: tuple[str, ...] = (
    KIND_TREE, KIND_SAPLING, KIND_ROCK, KIND_BUSH,
    KIND_BOULDER, KIND_WATER, KIND_GRAVE, KIND_SCORCH, KIND_CROP,
    KIND_LITTER,
)

#: A crop grows over this long, then reads ripe until a farmer harvests it, at
#: which point it drops back to a seedling and grows again - a field is
#: perennial, unlike a felled tree.
CROP_GROW_SEC = 115.0
CROP_HP = 3.0

FLAMMABLE: frozenset[str] = frozenset({KIND_TREE, KIND_SAPLING, KIND_BUSH, KIND_CROP})

# -------------------------------------------------------------------- tuning --

TREE_HP = 5.0
TREE_FALL_SEC = 1.2
TREE_WOOD_MIN, TREE_WOOD_MAX = 3, 5          # inclusive
TREE_SCALE = (0.82, 1.25)

SAPLING_HP = 1.0
SAPLING_GROW_SEC = 240.0                     # ~4 minutes of sim time

ROCK_HP = 6.0
ROCK_STONE_PER_HIT = 1

BUSH_HP = 2.0
BUSH_MAX_BERRIES = 3
BERRY_REGROW_SEC = 42.0

BURN_SEC: dict[str, float] = {KIND_TREE: 11.0, KIND_SAPLING: 4.0, KIND_BUSH: 5.5}
FIRE_SPREAD_RADIUS = 70.0
FIRE_SPREAD_PER_SEC = 0.45                   # per neighbour in range
FIRE_MIN_AGE = 0.8                           # must burn this long before spreading
SCORCH_RADIUS = 11
SCORCH_FADE_SEC = 240.0

BOULDER_HP = 12.0
BOULDER_ROLL_SLOPE = 0.30                    # |dy/dx| needed to get moving
BOULDER_ACCEL = 300.0                        # px/s^2 at slope 1.0
BOULDER_DRAG = 0.55                          # per second
BOULDER_MAX_SPEED = 220.0
BOULDER_STOP_SPEED = 7.0
BOULDER_HIT_R = 16.0
#: hp/s at 100 px/s.  A boulder is only in contact for two or three ticks, so
#: this has to be brutal enough to actually flatten what it hits.
BOULDER_DAMAGE = 26.0

WATER_MAX_DEPTH = 24.0

# -- natural regrowth ---------------------------------------------------------
# Vegetation has three sinks (chopping, foraging, fire) and, without this, no
# source: a world left running goes permanently bald in under an hour and
# features 2/4/17 become impossible.  So the land slowly reseeds itself back
# up to - never past - the density :func:`scatter` was asked for.
#
# Every rate here is multiplied by the colony's regrowth factor (see
# ``world.regrowth_factor()``), so the same map that goes bald under ten
# gatherers also grows back roughly three times as fast for them.
REGROW_CHECK_SEC = 12.0        # how often the check even runs
REGROW_TREE_SEC = 90.0         # mean gap between new saplings when short
REGROW_BUSH_SEC = 60.0         # mean gap between new bushes when short
REGROW_ROCK_SEC = 150.0        # stone weathers back slowest of the three
REGROW_TRIES = 24              # candidate columns examined per attempt

#: The renewable resources and how fast each tops back up. Rock was missing
#: from this list entirely, so once a colony mined its last rock it could never
#: make stone again - walls, watchtowers, firepits and spears all need it, so
#: the settlement simply stopped, which is the "no stone halting their progress"
#: a real run hit. Everything here also honours a hard floor (see _regrow):
#: the map is never allowed to hold zero of any of them.
REGROW_KINDS: tuple[tuple[str, float], ...] = (
    (KIND_TREE, REGROW_TREE_SEC),
    (KIND_BUSH, REGROW_BUSH_SEC),
    (KIND_ROCK, REGROW_ROCK_SEC),
)
#: Never fewer than this many usable sources of each renewable kind. One is
#: enough to break a deadlock; the probabilistic top-up carries it the rest of
#: the way back to the scatter target.
RESOURCE_FLOOR = 1

#: Hard ceiling on the recovery multiplier, whatever a caller hands us.  Keeps
#: a bad ``regrowth_factor`` (or a future constant change) from turning the map
#: into a jungle in one tick.
try:
    GROWTH_FACTOR_MAX = max(1.0, float(_REGROW_MAX))
except (TypeError, ValueError):  # pragma: no cover - constants gone weird
    GROWTH_FACTOR_MAX = 3.0

#: Used by :func:`scatter` when the caller does not supply counts.
DEFAULT_COUNTS: dict[str, int] = {
    KIND_TREE: 22,
    KIND_BUSH: 14,
    KIND_ROCK: 12,
    KIND_BOULDER: 4,
    KIND_SAPLING: 3,
    KIND_WATER: 2,
}

#: Minimum gap between two props of the *same* kind.  Different kinds only
#: need ``_CROSS_KIND`` of the larger of the two, so a bush can tuck in under
#: a tree without the forest eating the whole map.
_MIN_SPACING: dict[str, float] = {
    KIND_TREE: 30.0,
    KIND_SAPLING: 26.0,
    KIND_BUSH: 16.0,
    KIND_ROCK: 22.0,
    KIND_BOULDER: 60.0,
    KIND_GRAVE: 14.0,
    KIND_SCORCH: 10.0,
    KIND_WATER: 40.0,
}
_CROSS_KIND = 0.55


# ------------------------------------------------------------------ helpers --


def _name_variant(name: object) -> int:
    """A stable per-name variant, in 0..0xFFFF.

    crc32 rather than hash(): CPython salts str hashing per process, so
    ``hash("Vessa")`` differs on every launch. Using it for a headstone's shape
    meant the same seed produced a different grave each run - measured
    26436 / 11962 / 12501 for one name across three processes - so a save
    diverged byte-for-byte the moment anybody died, and a headstone standing all
    evening changed shape across a restart. A seeded run has to be reproducible;
    hash() of a str never is.
    """
    return zlib.crc32(str(name).encode("utf-8", "replace")) & 0xFFFF


def _f(v: object, default: float = 0.0) -> float:
    try:
        out = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _i(v: object, default: int = 0) -> int:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def growth_factor(source: object) -> float:
    """Natural-recovery multiplier from a World, a bare number, or nothing.

    ``source`` is normally the World: we look for ``regrowth_factor`` and call
    it if it is callable, tolerate it being a plain attribute, and fall back to
    ``1.0`` for anything else (including an old World without the method, a
    stub used by a test, or a call that raises).  Clamped to
    ``[0.0, GROWTH_FACTOR_MAX]`` so nothing here can run away.
    """
    if source is None:
        return 1.0
    raw: object = source
    if not isinstance(source, (int, float)) or isinstance(source, bool):
        fn = getattr(source, "regrowth_factor", None)
        if fn is None:
            return 1.0
        if callable(fn):
            try:
                raw = fn()
            except Exception:
                return 1.0
        else:
            raw = fn
    val = _f(raw, 1.0)
    if val <= 0.0:
        return 0.0
    return min(val, GROWTH_FACTOR_MAX)


def _json_safe(v: object) -> object:
    """Coerce a value into something ``json.dump`` will accept."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_json_safe(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_json_safe(x) for x in v]
    return str(v)


def _default_state(kind: str) -> dict:
    """Fresh per-kind state.  Also used to backfill keys on an old save."""
    if kind == KIND_TREE:
        return {"fallen": False, "fall_t": 0.0, "fall_angle": 0.0, "fall_dir": 1,
                "wood": 0, "burning": False, "burn_t": 0.0, "sway": 0.0}
    if kind == KIND_SAPLING:
        return {"growth": 0.0, "burning": False, "burn_t": 0.0}
    if kind == KIND_BUSH:
        return {"berries_left": BUSH_MAX_BERRIES, "regrow_t": 0.0,
                "burning": False, "burn_t": 0.0}
    if kind == KIND_BOULDER:
        return {"vx": 0.0, "rolling": False, "roll_t": 0.0, "spin": 0.0}
    if kind == KIND_WATER:
        return {"x0": 0, "x1": 0, "surface_y": 0.0, "depth": 0.0, "ripple": 0.0}
    if kind == KIND_GRAVE:
        return {"name": "", "generation": 0, "age": 0.0}
    if kind == KIND_SCORCH:
        return {"radius": float(SCORCH_RADIUS), "age": 0.0, "fade": 1.0}
    if kind == KIND_CROP:
        return {"growth": 0.0, "ripe": False, "burning": False, "burn_t": 0.0}
    if kind == KIND_LITTER:
        # Two ints and nothing that ticks. Litter is the one kind with no
        # behaviour at all - it sits where it was dropped until somebody picks
        # it up - so giving it a timer would cost a branch per prop per frame
        # for a thing that never changes. ``shape`` picks the debris silhouette
        # (see render), ``drop_t`` is the world clock it landed on, kept purely
        # so a debug pass can tell fresh mess from old.
        return {"shape": 0, "drop_t": 0.0}
    return {}


def _default_hp(kind: str) -> float:
    return {
        KIND_TREE: TREE_HP,
        KIND_SAPLING: SAPLING_HP,
        KIND_ROCK: ROCK_HP,
        KIND_BUSH: BUSH_HP,
        KIND_CROP: CROP_HP,
        KIND_BOULDER: BOULDER_HP,
    }.get(kind, 1.0)


def _ev(kind_ev: str, p: "Prop", **extra: object) -> dict:
    d: dict = {"type": kind_ev, "id": p.id, "kind": p.kind, "x": float(p.x), "y": float(p.y)}
    d.update(extra)
    return d


# ---------------------------------------------------------------------- Prop --


@dataclass
class Prop:
    """One piece of scenery.  ``state`` carries the kind-specific bits."""

    id: int = 0
    kind: str = KIND_ROCK
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    hp: float = 1.0
    max_hp: float = 1.0
    alive: bool = True
    #: Stable per-prop randomness so the renderer can vary silhouettes.
    variant: int = 0
    state: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = _i(self.id, 0)
        self.kind = str(self.kind) if str(self.kind) in KINDS else KIND_ROCK
        self.x = _f(self.x, 0.0)
        self.y = _f(self.y, 0.0)
        self.scale = max(0.05, _f(self.scale, 1.0))
        self.max_hp = max(0.01, _f(self.max_hp, _default_hp(self.kind)))
        self.hp = _f(self.hp, self.max_hp)
        self.alive = bool(self.alive)
        self.variant = _i(self.variant, 0) & 0x7FFFFFFF
        if not isinstance(self.state, dict):
            self.state = {}
        base = _default_state(self.kind)
        for k, v in base.items():
            self.state.setdefault(k, v)

    # -- convenience -------------------------------------------------------

    @property
    def burning(self) -> bool:
        return bool(self.state.get("burning", False))

    @burning.setter
    def burning(self, value: object) -> None:
        """Writable so ``prop.burning = False`` (events.py rain/flood) works."""
        on = bool(value)
        if on and not self.state.get("burning"):
            self.state["burn_t"] = 0.0
        elif not on:
            self.state["burn_t"] = 0.0
        self.state["burning"] = on

    @property
    def burn_t(self) -> float:
        return _f(self.state.get("burn_t"), 0.0)

    @burn_t.setter
    def burn_t(self, value: object) -> None:
        self.state["burn_t"] = _f(value, 0.0)

    @property
    def flammable(self) -> bool:
        return self.kind in FLAMMABLE and self.alive

    def ignite(self) -> bool:
        """Catch fire.  Method form, preferred by events.py."""
        return ignite(self)

    def extinguish(self) -> bool:
        """Go out.  Method form, preferred by events.py."""
        return extinguish(self)

    def dist_x(self, x: float) -> float:
        return abs(self.x - _f(x, 0.0))

    def dist(self, x: float, y: float) -> float:
        return math.hypot(self.x - _f(x, 0.0), self.y - _f(y, 0.0))

    def damage(self, amount: float) -> bool:
        """Subtract hp.  Returns True if this took it to zero."""
        amt = _f(amount, 0.0)
        if amt <= 0.0:
            return self.hp <= 0.0
        self.hp = max(0.0, self.hp - amt)
        return self.hp <= 0.0

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "kind": str(self.kind),
            "x": float(self.x),
            "y": float(self.y),
            "scale": float(self.scale),
            "hp": float(self.hp),
            "max_hp": float(self.max_hp),
            "alive": bool(self.alive),
            "variant": int(self.variant),
            "state": {str(k): _json_safe(v) for k, v in self.state.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Prop":
        """Rebuild a prop.  Missing keys take defaults, unknown keys ignored."""
        if not isinstance(d, dict):
            return cls()
        kind = d.get("kind", KIND_ROCK)
        kind = kind if isinstance(kind, str) and kind in KINDS else KIND_ROCK
        raw_state = d.get("state")
        state = dict(raw_state) if isinstance(raw_state, dict) else {}
        return cls(
            id=_i(d.get("id"), 0),
            kind=kind,
            x=_f(d.get("x"), 0.0),
            y=_f(d.get("y"), 0.0),
            scale=_f(d.get("scale"), 1.0),
            hp=_f(d.get("hp"), _default_hp(kind)),
            max_hp=_f(d.get("max_hp"), _default_hp(kind)),
            alive=bool(d.get("alive", True)),
            variant=_i(d.get("variant"), 0),
            state=state,
        )


# -------------------------------------------------------------- the registry --


class PropRegistry:
    """Every prop in the world, keyed by id, iterated in stable insertion order."""

    def __init__(self) -> None:
        self._props: dict[int, Prop] = {}
        self.next_id: int = 1
        #: Only used if a caller ticks us without a usable Generator.
        self._fallback_rng: np.random.Generator | None = None
        #: Vegetation density to reseed back toward, set by :func:`scatter`.
        self.targets: dict[str, int] = {}
        self._regrow_t: float = 0.0
        #: Last recovery multiplier applied, for debugging/inspection only.
        #: Derived from the World every tick, so it is not persisted.
        self.growth: float = 1.0

    # -- container ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._props)

    def __iter__(self) -> Iterator[Prop]:
        return iter(list(self._props.values()))

    def __contains__(self, pid: object) -> bool:
        return _i(pid, -1) in self._props

    def iter_alive(self) -> Iterator[Prop]:
        for p in list(self._props.values()):
            if p.alive:
                yield p

    def get(self, pid: int) -> Prop | None:
        return self._props.get(_i(pid, -1))

    # -- mutation ----------------------------------------------------------

    def add(self, prop: Prop) -> Prop:
        """Insert a prop, assigning an id if it does not have one."""
        if not isinstance(prop, Prop):
            raise TypeError("PropRegistry.add expects a Prop")
        if prop.id <= 0 or prop.id in self._props:
            prop.id = self.next_id
        self.next_id = max(self.next_id, prop.id + 1)
        self._props[prop.id] = prop
        return prop

    def spawn(self, kind: str, x: float, y: float, **kw: object) -> Prop:
        """Create and register a prop of ``kind`` at ``(x, y)``."""
        k = kind if kind in KINDS else KIND_ROCK
        hp = _f(kw.pop("hp", None), _default_hp(k))
        return self.add(
            Prop(
                id=0,
                kind=k,
                x=_f(x),
                y=_f(y),
                scale=_f(kw.pop("scale", 1.0), 1.0),
                hp=hp,
                max_hp=_f(kw.pop("max_hp", None), hp),
                alive=bool(kw.pop("alive", True)),
                variant=_i(kw.pop("variant", 0), 0),
                state=dict(kw.pop("state", {}) or {}),  # type: ignore[arg-type]
            )
        )

    def remove(self, prop: "Prop | int") -> bool:
        """Drop a prop by id or by reference.  Returns True if it was there."""
        pid = prop.id if isinstance(prop, Prop) else _i(prop, -1)
        return self._props.pop(pid, None) is not None

    def clear(self) -> None:
        self._props.clear()
        self.next_id = 1

    # -- queries -----------------------------------------------------------

    def all(self, include_dead: bool = False) -> list[Prop]:
        """Every living prop, in id order.  The renderer depth-sorts this."""
        return [p for p in self._props.values() if include_dead or p.alive]

    def all_of(self, kind: str, include_dead: bool = False) -> list[Prop]:
        """Every prop of ``kind``, in id order."""
        return [
            p for p in self._props.values()
            if p.kind == kind and (include_dead or p.alive)
        ]

    def burning(self) -> list[Prop]:
        """Everything currently on fire - each one is a light source."""
        return [p for p in self._props.values() if p.alive and p.state.get("burning")]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self._props.values():
            if p.alive:
                out[p.kind] = out.get(p.kind, 0) + 1
        return out

    def nearest(
        self,
        kind: str | None,
        x: float,
        y: float | None = None,
        max_dist: float | None = None,
        predicate: Callable[[Prop], bool] | None = None,
    ) -> Prop | None:
        """Closest living prop of ``kind`` to ``x`` (or to ``(x, y)`` if given)."""
        px = _f(x)
        py = _f(y) if y is not None else None
        limit = _f(max_dist, float("inf")) if max_dist is not None else float("inf")
        best: Prop | None = None
        best_d = float("inf")
        for p in self._props.values():
            if not p.alive:
                continue
            if kind is not None and p.kind != kind:
                continue
            if predicate is not None:
                try:
                    if not predicate(p):
                        continue
                except Exception:
                    continue
            d = p.dist_x(px) if py is None else p.dist(px, py)
            if d < best_d and d <= limit:
                best_d = d
                best = p
        return best

    def within(self, x: float, radius: float, kind: str | None = None) -> list[Prop]:
        """Living props whose x is within ``radius`` of ``x``."""
        px = _f(x)
        r = _f(radius, 0.0)
        return [
            p for p in self._props.values()
            if p.alive and (kind is None or p.kind == kind) and abs(p.x - px) <= r
        ]

    # -- behaviour ---------------------------------------------------------

    def add_grave(self, x: float, y: float, name: str = "", generation: int = 0) -> Prop:
        """Headstone at ``(x, y)``.  Called by World when an agent is buried."""
        return self.spawn(
            KIND_GRAVE,
            x,
            y,
            variant=_name_variant(name),
            state={"name": str(name), "generation": _i(generation, 0), "age": 0.0},
        )

    def plant_sapling(self, x: float, y: float) -> Prop:
        """Plant a sapling at ``(x, y)``.

        ``y`` is deliberately required: actions.py probes single-argument
        hooks first, and a sapling planted at an assumed y would hang in
        mid-air.  Letting that probe raise TypeError makes it pass the real
        ground height on the retry.
        """
        return self.spawn(
            KIND_SAPLING, x, y, scale=0.32, hp=SAPLING_HP, state={"growth": 0.0}
        )

    def tick(self, *args: object, world: object = None) -> list[dict]:
        """Advance every prop behaviour.  See :func:`tick_props`.

        Three call shapes are accepted::

            reg.tick(terrain, rng, dt)          # direct, used by the sim tests
            reg.tick(terrain, rng, dt, world)   # ...with headcount scaling
            reg.tick(world, dt)                 # World.tick style; pulls
                                                # world.terrain / world.rng and
                                                # world.regrowth_factor() off it

        Anything with a ``regrowth_factor`` (or a plain number) works as
        ``world``; without one, natural recovery runs at 1.0.
        """
        src: object = world
        if len(args) >= 3:
            terrain, rng, dt = args[0], args[1], args[2]
            if src is None and len(args) >= 4:
                src = args[3]
        elif len(args) == 2:
            w = args[0]
            terrain = getattr(w, "terrain", None)
            rng = getattr(w, "rng", None)
            dt = args[1]
            if src is None:
                src = w
        else:
            return []
        if terrain is None or not hasattr(terrain, "ground_y"):
            return []
        if not (hasattr(rng, "random") and hasattr(rng, "integers")):
            if self._fallback_rng is None:
                self._fallback_rng = np.random.default_rng(self._fallback_seed(src))
            rng = self._fallback_rng
        return tick_props(self, terrain, rng, _f(dt, 0.0), src)  # type: ignore[arg-type]

    def _fallback_seed(self, src: object) -> int:
        """Seed for the stream used when the caller ticked us without one.

        The guard above has to stay - ``reg.tick(world, dt)`` against a world
        that has no ``rng`` (a stub, a half-built World) must not raise on a
        per-frame path - but ``default_rng()`` with no argument draws from the
        OS entropy pool, which quietly made every such run irreproducible:
        measured across three processes, 20 sim-minutes of the same registry
        and seed ended in three different forests.  Take the World's own seed
        when there is one; otherwise crc32 the registry's identity so a given
        registry at least always falls back to the same stream.  crc32 and not
        hash() for the reason spelled out in :func:`_name_variant`.
        """
        s = getattr(src, "seed", None)
        if isinstance(s, (int, float)) and not isinstance(s, bool):
            return int(s) & 0xFFFFFFFF
        ident = "props|%d|%s" % (
            self.next_id,
            ",".join("%d:%s" % (p.id, p.kind) for p in self._props.values()),
        )
        return zlib.crc32(ident.encode("utf-8", "replace"))

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "next_id": int(self.next_id),
            "targets": {str(k): int(v) for k, v in self.targets.items()},
            "props": [p.to_dict() for p in self._props.values()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PropRegistry":
        """Rebuild the registry.  Broken entries are skipped, never fatal."""
        reg = cls()
        if not isinstance(d, dict):
            return reg
        raw = d.get("props")
        if isinstance(raw, (list, tuple)):
            for entry in raw:
                try:
                    p = Prop.from_dict(entry)
                except Exception:
                    continue
                if p.id <= 0 or p.id in reg._props:
                    p.id = reg.next_id
                reg._props[p.id] = p
                reg.next_id = max(reg.next_id, p.id + 1)
        # Keep ids ascending so iteration order is save-stable.
        reg._props = dict(sorted(reg._props.items()))
        nid = _i(d.get("next_id"), reg.next_id)
        reg.next_id = max(reg.next_id, nid, 1)

        raw_t = d.get("targets")
        if isinstance(raw_t, dict):
            reg.targets = {
                str(k): _i(v, 0) for k, v in raw_t.items() if _i(v, 0) > 0
            }
        else:
            # Save predates regrowth: hold the land at whatever it has now
            # rather than letting it decay to nothing.
            counts = reg.counts()
            reg.targets = {
                KIND_TREE: counts.get(KIND_TREE, 0) + counts.get(KIND_SAPLING, 0),
                KIND_BUSH: counts.get(KIND_BUSH, 0),
                KIND_ROCK: counts.get(KIND_ROCK, 0),
            }
            reg.targets = {k: v for k, v in reg.targets.items() if v > 0}
        return reg


# ------------------------------------------------------------- action verbs --


def chop(prop: Prop, damage: float = 1.0, rng: np.random.Generator | None = None) -> bool:
    """Hit a tree with an axe.

    Returns ``True`` on the blow that topples it.  The wood yield is rolled
    now and stored in ``state['wood']`` so it survives a save mid-fall; the
    ``tree_felled`` event carries it once the fall animation finishes.
    """
    if not isinstance(prop, Prop) or prop.kind != KIND_TREE or not prop.alive:
        return False
    if prop.state.get("fallen"):
        return False
    if not prop.damage(damage):
        prop.state["sway"] = min(1.0, _f(prop.state.get("sway"), 0.0) + 0.5)
        return False
    gen = rng if rng is not None else np.random.default_rng(prop.variant or 1)
    prop.state["fallen"] = True
    prop.state["fall_t"] = 0.0
    prop.state["fall_angle"] = 0.0
    prop.state["fall_dir"] = 1 if int(gen.integers(0, 2)) else -1
    prop.state["wood"] = int(gen.integers(TREE_WOOD_MIN, TREE_WOOD_MAX + 1))
    return True


def mine(prop: Prop, damage: float = 1.0, rng: np.random.Generator | None = None) -> int:
    """Hit a rock.  Returns the stone knocked loose by this blow (0 or more)."""
    if not isinstance(prop, Prop) or prop.kind != KIND_ROCK or not prop.alive:
        return 0
    if _f(damage, 0.0) <= 0.0:
        return 0
    # Leave `alive` alone: the next tick sees hp<=0 and emits 'rock_depleted'.
    # Clearing it here would instead trip the generic 'prop_destroyed' path.
    prop.damage(damage)
    return int(ROCK_STONE_PER_HIT)


def harvest_berries(prop: Prop, qty: int = 1) -> int:
    """Take up to ``qty`` berries from a bush.  Returns how many were taken."""
    if not isinstance(prop, Prop) or prop.kind != KIND_BUSH or not prop.alive:
        return 0
    have = _i(prop.state.get("berries_left"), 0)
    take = max(0, min(_i(qty, 1), have))
    if take:
        prop.state["berries_left"] = have - take
        prop.state["regrow_t"] = 0.0
    return take


def ignite(prop: Prop) -> bool:
    """Set a flammable prop alight.  Returns True if it was not already."""
    if not isinstance(prop, Prop) or not prop.flammable:
        return False
    if prop.state.get("burning"):
        return False
    prop.state["burning"] = True
    prop.state["burn_t"] = 0.0
    return True


def nudge(prop: Prop, vx: float = 0.0) -> bool:
    """Start a boulder rolling (earthquake, mudslide, or a hard shove).

    Boulders sit still until this is called; once rolling they follow the
    terrain slope until they run out of speed on level ground.  ``vx`` is an
    optional initial push in px/s.  Returns True if it is now in motion.
    """
    if not isinstance(prop, Prop) or prop.kind != KIND_BOULDER or not prop.alive:
        return False
    prop.state["rolling"] = True
    prop.state["vx"] = float(np.clip(_f(vx, 0.0), -BOULDER_MAX_SPEED, BOULDER_MAX_SPEED))
    return True


def extinguish(prop: Prop) -> bool:
    """Put a prop out (rain, a bucket).  Returns True if it was burning."""
    if not isinstance(prop, Prop) or not prop.state.get("burning"):
        return False
    prop.state["burning"] = False
    prop.state["burn_t"] = 0.0
    return True


def plant_sapling(
    reg: PropRegistry,
    terrain: "Terrain",
    x: float,
    rng: np.random.Generator | None = None,
) -> Prop:
    """Feature 17 - a stickman plants a sapling that will become a tree."""
    px = float(np.clip(_f(x), 4.0, float(terrain.W - 5)))
    # ``rng`` stays optional - not every caller holds a Generator - but an
    # unseeded ``default_rng()`` here meant the sapling's variant, and so the
    # shape of the tree it becomes, was drawn from OS entropy: three processes
    # planting the same four saplings got three different sets of variants.
    # Derive it from where the sapling went in and which one it is instead;
    # crc32 rather than hash(), see :func:`_name_variant`.
    gen = rng if rng is not None else np.random.default_rng(
        zlib.crc32(("sapling|%d|%.3f" % (reg.next_id, px)).encode("utf-8"))
    )
    return reg.spawn(
        KIND_SAPLING,
        px,
        terrain.ground_y(px),
        scale=0.34,
        hp=SAPLING_HP,
        variant=int(gen.integers(0, 1 << 30)),
        state={"growth": 0.0},
    )


def place_grave(
    reg: PropRegistry,
    terrain: "Terrain",
    x: float,
    name: str = "",
    generation: int = 0,
) -> Prop:
    """Feature 27 - mark where somebody died."""
    px = float(np.clip(_f(x), 4.0, float(terrain.W - 5)))
    return reg.spawn(
        KIND_GRAVE,
        px,
        terrain.ground_y(px),
        scale=1.0,
        variant=_name_variant(name),
        state={"name": str(name), "generation": _i(generation, 0), "age": 0.0},
    )


def place_scorch(
    reg: PropRegistry,
    terrain: "Terrain",
    x: float,
    radius: float = SCORCH_RADIUS,
) -> Prop:
    """Burn scar - lightning strikes and burnt-out props both leave one."""
    px = float(np.clip(_f(x), 2.0, float(terrain.W - 3)))
    r = max(3.0, _f(radius, SCORCH_RADIUS))
    try:
        terrain.set_material_span(int(px - r), int(px + r), MAT_ASH)
    except Exception:
        pass
    return reg.spawn(
        KIND_SCORCH,
        px,
        terrain.ground_y(px),
        scale=r / float(SCORCH_RADIUS),
        variant=int(px) & 0xFFFF,
        state={"radius": r, "age": 0.0, "fade": 1.0},
    )


# -------------------------------------------------------------------- litter --


def litter_count(reg: PropRegistry) -> int:
    """How many pieces of litter are lying about. Cheap: one pass, no allocs."""
    n = 0
    try:
        for p in reg:  # type: ignore[union-attr]
            if p.alive and p.kind == KIND_LITTER:
                n += 1
    except Exception:
        return 0
    return n


def drop_litter(
    reg: PropRegistry,
    terrain: "Terrain",
    x: float,
    rng: object = None,
    now: float = 0.0,
) -> Prop | None:
    """Drop one piece of litter at ``x``, honouring :data:`LITTER_MAX`.

    ``rng`` may be anything with ``random()`` - in practice ``world.pyrng``, so
    the roll stays on the seeded stream. It only picks the debris silhouette;
    the position is the dropper's.

    At the cap the **oldest** piece is recycled rather than the drop refused.
    See the constant for why: refusing would let one unreachable pile freeze
    littering everywhere else on the map for the rest of the run.
    """
    if not isinstance(reg, PropRegistry):
        return None
    try:
        px = float(np.clip(_f(x), 2.0, float(terrain.W - 3)))
        py = float(terrain.ground_y(px))
    except Exception:
        return None
    try:
        if litter_count(reg) >= max(1, int(LITTER_MAX)):
            oldest = None
            for p in reg:
                if p.alive and p.kind == KIND_LITTER and (
                    oldest is None or p.id < oldest.id
                ):
                    oldest = p          # ids ascend, so lowest id is oldest
            if oldest is not None:
                reg.remove(oldest)
    except Exception:
        pass
    shape = 0
    try:
        fn = getattr(rng, "random", None)
        if callable(fn):
            shape = int(float(fn()) * 4.0) & 3
    except Exception:
        shape = 0
    return reg.spawn(
        KIND_LITTER,
        px,
        py,
        scale=1.0,
        hp=1.0,
        variant=(int(px) * 2654435761 + shape) & 0xFFFF,
        state={"shape": shape, "drop_t": _f(now, 0.0)},
    )


def _litter_droppers(world: object) -> list:
    """Living, outdoor agents off whatever shape of world we were handed.

    Duck-typed like the rest of this module's world contact: the registry is
    ticked by the real World, by the module smoke test, and by a bare stub in
    other modules' self-tests, and none of them may be able to break littering
    (or, worse, the whole prop loop) by not having a population.
    """
    roster: object = None
    pop = getattr(world, "population", None)
    if pop is not None:
        fn = getattr(pop, "alive_agents", None)
        if callable(fn):
            try:
                roster = fn()
            except Exception:
                roster = None
        if roster is None:
            roster = getattr(pop, "agents", None)
    if roster is None:
        roster = getattr(world, "agents", None)
    if not isinstance(roster, (list, tuple)):
        return []
    out = []
    for a in roster:
        try:
            if not getattr(a, "alive", True):
                continue
            # Indoors: the speck would be drawn under a hut, and a sleeper is
            # not "normal activity" in any case.
            if getattr(a, "inside", None) is not None:
                continue
            out.append(a)
        except Exception:
            continue
    return out


def _tick_litter(
    reg: PropRegistry, terrain: "Terrain", world: object, dt: float
) -> None:
    """Roll each living villager for a dropped piece of litter.

    Poisson-ish: probability ``dt / LITTER_DROP_SEC`` per agent per tick, which
    at 30 Hz is ~0.00044 - far too small to double-fire, so one roll per agent
    per tick is both correct and the cheapest possible form.

    **All randomness comes from ``world.pyrng``**, the seeded python stream, not
    from the numpy generator this module ticks with and not from bare
    ``random.*``. Two runs of the same seed must litter identically or nothing
    measured on top of this is comparable.
    """
    if world is None or dt <= 0.0:
        return
    pyrng = getattr(world, "pyrng", None)
    roll = getattr(pyrng, "random", None)
    if not callable(roll):
        return                      # no seeded stream: do not litter at all
    agents = _litter_droppers(world)
    if not agents:
        return
    chance = dt / max(1.0, float(LITTER_DROP_SEC))
    now = _f(getattr(world, "world_time", 0.0), 0.0)
    for a in agents:
        try:
            if roll() >= chance:
                continue
            drop_litter(reg, terrain, float(getattr(a, "x", 0.0)), pyrng, now)
        except Exception:
            continue


# ------------------------------------------------------------------- ticking --


def tick_props(
    reg: PropRegistry,
    terrain: "Terrain",
    rng: np.random.Generator,
    dt: float,
    world: object = None,
) -> list[dict]:
    """Advance every prop behaviour by ``dt`` seconds.

    ``world`` is optional and only used for :func:`growth_factor`: sapling
    growth, berry regrow and reseeding run ``factor`` times faster for a big
    colony.  Burning, fire spread, falling and fading are untouched by it.

    Returns the event list documented in the module docstring.  Fails soft:
    a misbehaving prop is skipped, never allowed to kill the frame.
    """
    events: list[dict] = []
    if not isinstance(reg, PropRegistry):
        return events
    step = _f(dt, 0.0)
    if step <= 0.0:
        return events
    step = min(step, 0.25)

    grow = growth_factor(world)
    try:
        reg.growth = grow
    except Exception:  # pragma: no cover - reg is ours, but never die for this
        pass

    doomed: list[Prop] = []
    try:
        props = list(reg)
    except Exception:
        return events

    for p in props:
        try:
            if not p.alive:
                doomed.append(p)
                events.append(_ev("prop_destroyed", p, cause=str(p.state.get("cause", "killed"))))
                continue
            if p.state.get("burning"):
                _tick_burning(p, terrain, reg, rng, step, events, doomed)
                continue
            if p.kind == KIND_TREE:
                _tick_tree(p, step, events, doomed)
            elif p.kind == KIND_SAPLING:
                _tick_sapling(p, rng, step, events, grow)
            elif p.kind == KIND_BUSH:
                _tick_bush(p, step, events, grow)
            elif p.kind == KIND_ROCK:
                if p.hp <= 0.0:
                    doomed.append(p)
                    events.append(_ev("rock_depleted", p))
            elif p.kind == KIND_BOULDER:
                _tick_boulder(p, terrain, reg, step, events)
            elif p.kind == KIND_CROP:
                _tick_crop(p, step, grow)
            elif p.kind == KIND_SCORCH:
                _tick_scorch(p, step, events, doomed)
            elif p.kind == KIND_GRAVE:
                p.state["age"] = _f(p.state.get("age"), 0.0) + step
            elif p.kind == KIND_WATER:
                p.state["ripple"] = (_f(p.state.get("ripple"), 0.0) + step) % 1000.0
        except Exception:
            continue

    # Fire spread runs after the per-prop pass so ignitions land on a settled world.
    try:
        _spread_fire(reg, rng, step, events)
    except Exception:
        pass
    try:
        _regrow(reg, terrain, rng, step, events, grow)
    except Exception:
        pass
    # Littering is driven from here rather than from the agent loop because this
    # is the one per-tick pass that already owns the registry and the terrain,
    # and because a dropped piece is a *prop* - putting the spawn anywhere else
    # would mean a second module reaching into PropRegistry every frame.
    try:
        _tick_litter(reg, terrain, world, step)
    except Exception:
        pass

    for p in doomed:
        try:
            reg.remove(p)
        except Exception:
            continue
    return events


def _tick_tree(p: Prop, dt: float, events: list[dict], doomed: list[Prop]) -> None:
    sway = _f(p.state.get("sway"), 0.0)
    if sway > 0.0:
        p.state["sway"] = max(0.0, sway - dt * 1.6)
    if not p.state.get("fallen"):
        return
    t = _f(p.state.get("fall_t"), 0.0) + dt
    p.state["fall_t"] = t
    u = min(1.0, t / TREE_FALL_SEC)
    # Ease-in: slow topple then an accelerating slam.
    p.state["fall_angle"] = float(
        _i(p.state.get("fall_dir"), 1) * (math.pi * 0.5) * (u * u * (3.0 - 2.0 * u)) * u ** 0.5
    )
    if t >= TREE_FALL_SEC:
        wood = _i(p.state.get("wood"), TREE_WOOD_MIN)
        wood = int(np.clip(wood, TREE_WOOD_MIN, TREE_WOOD_MAX))
        doomed.append(p)
        events.append(_ev("tree_felled", p, wood=wood))


def _tick_sapling(
    p: Prop,
    rng: np.random.Generator,
    dt: float,
    events: list[dict],
    grow: float = 1.0,
) -> None:
    # Natural growth, so it takes the colony's recovery multiplier.
    g = _f(p.state.get("growth"), 0.0) + dt * max(0.0, grow) / SAPLING_GROW_SEC
    if g < 1.0:
        p.state["growth"] = g
        p.scale = 0.30 + 0.55 * g
        return
    # Promote in place: same id, so anything holding a reference still works.
    p.kind = KIND_TREE
    p.max_hp = TREE_HP
    p.hp = TREE_HP
    p.scale = float(np.clip(0.86 + 0.25 * float(rng.random()), *TREE_SCALE))
    p.state = _default_state(KIND_TREE)
    events.append(_ev("sapling_grown", p))


def _tick_bush(p: Prop, dt: float, events: list[dict], grow: float = 1.0) -> None:
    have = _i(p.state.get("berries_left"), 0)
    if have >= BUSH_MAX_BERRIES:
        p.state["berries_left"] = BUSH_MAX_BERRIES
        p.state["regrow_t"] = 0.0
        return
    # Berries are food regrowing: the whole point of the headcount scaling is
    # that ten foragers do not out-eat the bushes.
    t = _f(p.state.get("regrow_t"), 0.0) + dt * max(0.0, grow)
    if t >= BERRY_REGROW_SEC:
        # Carry the remainder so a high factor can add more than one berry per
        # regrow window instead of silently throwing the overshoot away.
        p.state["regrow_t"] = 0.0 if have + 1 >= BUSH_MAX_BERRIES else t - BERRY_REGROW_SEC
        p.state["berries_left"] = have + 1
        events.append(_ev("berries_grown", p, berries=have + 1))
    else:
        p.state["regrow_t"] = t


def _tick_burning(
    p: Prop,
    terrain: "Terrain",
    reg: PropRegistry,
    rng: np.random.Generator,
    dt: float,
    events: list[dict],
    doomed: list[Prop],
) -> None:
    was = _f(p.state.get("burn_t"), 0.0)
    t = was + dt
    p.state["burn_t"] = t
    if was <= 0.0:
        # First tick alight, whoever struck the match (lightning, a campfire,
        # another prop).  'fire_spread' is the more specific variant.
        events.append(_ev("prop_ignited", p))
    span = BURN_SEC.get(p.kind, 6.0)
    p.hp = max(0.0, p.max_hp * (1.0 - t / max(span, 0.01)))
    if t < span:
        return
    x = p.x
    r = SCORCH_RADIUS * (0.7 + 0.6 * p.scale)
    doomed.append(p)
    events.append(_ev("prop_burned_out", p))
    try:
        place_scorch(reg, terrain, x, r)
    except Exception:
        pass


def _spread_fire(
    reg: PropRegistry, rng: np.random.Generator, dt: float, events: list[dict]
) -> None:
    burning = [p for p in reg.iter_alive() if p.state.get("burning")]
    if not burning:
        return
    fuel = [p for p in reg.iter_alive() if p.flammable and not p.state.get("burning")]
    if not fuel:
        return
    chance = 1.0 - math.exp(-FIRE_SPREAD_PER_SEC * dt)
    for src in burning:
        if _f(src.state.get("burn_t"), 0.0) < FIRE_MIN_AGE:
            continue
        for tgt in fuel:
            if tgt.state.get("burning"):
                continue
            if abs(tgt.x - src.x) > FIRE_SPREAD_RADIUS:
                continue
            if float(rng.random()) < chance and ignite(tgt):
                events.append(_ev("fire_spread", tgt, src=int(src.id)))


def _tick_boulder(
    p: Prop, terrain: "Terrain", reg: PropRegistry, dt: float, events: list[dict]
) -> None:
    # A boulder never starts moving by itself - otherwise every world would
    # open with an avalanche flattening the forest.  Something has to shove it:
    # an earthquake, a mudslide, or a stickman.  See :func:`nudge`.
    if not p.state.get("rolling"):
        p.state["vx"] = 0.0
        p.y = terrain.ground_y(p.x)
        return

    vx = _f(p.state.get("vx"), 0.0)
    try:
        s = float(terrain.slope(p.x))
    except Exception:
        s = 0.0

    # y grows downward, so a positive slope means downhill to the right.
    vx += BOULDER_ACCEL * float(np.clip(s, -2.5, 2.5)) * dt
    vx -= vx * BOULDER_DRAG * dt
    vx = float(np.clip(vx, -BOULDER_MAX_SPEED, BOULDER_MAX_SPEED))

    if abs(vx) < BOULDER_STOP_SPEED and abs(s) <= BOULDER_ROLL_SLOPE:
        events.append(_ev("boulder_stopped", p))
        p.state["vx"] = 0.0
        p.state["rolling"] = False
        p.y = terrain.ground_y(p.x)
        return

    p.state["vx"] = vx
    p.state["roll_t"] = _f(p.state.get("roll_t"), 0.0) + dt
    p.x = float(np.clip(p.x + vx * dt, 2.0, float(terrain.W - 3)))
    p.y = terrain.ground_y(p.x)
    p.state["spin"] = (_f(p.state.get("spin"), 0.0) + vx * dt / max(6.0, 9.0 * p.scale)) % (
        2.0 * math.pi
    )

    force = abs(vx)
    events.append(_ev("boulder_rolling", p, vx=float(vx)))
    if force < 30.0:
        return
    for other in reg.within(p.x, BOULDER_HIT_R):
        # Litter is excluded with the other unhittable kinds so that the only
        # things that ever remove it are the two documented sinks (a villager
        # carrying it to a fire, and the recycling cap). A boulder quietly
        # flattening a pile would be a third, invisible one.
        if other.id == p.id or other.kind in (
            KIND_WATER, KIND_SCORCH, KIND_BOULDER, KIND_LITTER
        ):
            continue
        other.hp = max(0.0, other.hp - BOULDER_DAMAGE * dt * (force / 100.0))
        if other.hp <= 0.0 and other.alive:
            other.alive = False
            other.state["cause"] = "boulder"
    events.append(_ev("boulder_impact", p, force=float(force)))


def _ensure_targets(reg: PropRegistry) -> None:
    """Adopt the standing vegetation as the reseed target if nobody set one.

    :func:`scatter` normally records what it was asked for, and
    :meth:`PropRegistry.from_dict` backfills it for old saves.  This covers the
    remaining case - a registry assembled by hand (tests, vignettes) - so the
    map it starts with is also the map it recovers to, and never more.
    """
    try:
        if getattr(reg, "targets", None):
            return
        counts = reg.counts()
        trees = counts.get(KIND_TREE, 0) + counts.get(KIND_SAPLING, 0)
        bushes = counts.get(KIND_BUSH, 0)
        rocks = counts.get(KIND_ROCK, 0)
        if trees <= 0 and bushes <= 0 and rocks <= 0:
            return
        reg.targets = {k: v for k, v in
                       ((KIND_TREE, trees), (KIND_BUSH, bushes), (KIND_ROCK, rocks))
                       if v > 0}
    except Exception:  # pragma: no cover - never worth a frame
        pass


def _regrow(
    reg: PropRegistry,
    terrain: "Terrain",
    rng: np.random.Generator,
    dt: float,
    events: list[dict],
    grow: float = 1.0,
) -> None:
    """Slowly reseed vegetation back toward the density scatter() was asked for.

    Only ever tops *up* to the target, so this cannot run away; a world that is
    already green does nothing but a cheap counter check.  ``grow`` is the
    colony's recovery multiplier: a stripped map comes back about three times
    faster for ten colonists than for two.  Silent by design - the events are
    for the stats counters, not the chronicle.
    """
    _ensure_targets(reg)
    targets = getattr(reg, "targets", None)
    if not targets:
        return
    reg._regrow_t = _f(getattr(reg, "_regrow_t", 0.0), 0.0) + dt
    if reg._regrow_t < REGROW_CHECK_SEC:
        return
    elapsed = reg._regrow_t
    reg._regrow_t = 0.0
    factor = max(0.0, _f(grow, 1.0))
    if factor <= 0.0:
        return

    for kind, gap in REGROW_KINDS:
        target = _i(targets.get(kind), 0)
        if target <= 0:
            continue
        have = _usable_count(reg, kind)
        if have >= target:
            continue

        # Hard floor: if the map holds none of this resource, reseed at once and
        # skip the probabilistic timer. Otherwise a colony that mined its last
        # rock (or burned its last tree) would wait out a random gap it can no
        # longer afford - or, for rock, never recover at all.
        forced = have < RESOURCE_FLOOR
        if not forced:
            if float(rng.random()) > min(1.0, elapsed * factor / max(gap, 1e-3)):
                continue

        x = _regrow_site(reg, terrain, rng, kind)
        if x is None and forced:
            x = _guaranteed_site(reg, terrain, rng)   # the floor must not fail
        if x is None:
            continue
        p = _reseed(reg, terrain, rng, kind, x, mature=forced)
        if p is not None:
            events.append(_ev("prop_regrown", p, target=kind))


def _usable_count(reg: PropRegistry, kind: str) -> int:
    """Sources of `kind` a colonist could actually harvest right now.

    A fallen tree, a mined-out rock or a dead bush does not count - and a
    sapling counts toward trees because it is one on its way up.
    """
    if kind == KIND_TREE:
        trees = sum(1 for p in reg.all_of(KIND_TREE)
                    if p.alive and not p.state.get("fallen"))
        return trees + len(reg.all_of(KIND_SAPLING))
    return sum(1 for p in reg.all_of(kind) if p.alive)


def _reseed(reg: PropRegistry, terrain: "Terrain", rng: np.random.Generator,
            kind: str, x: float, mature: bool) -> "Prop | None":
    """Place one fresh source. Trees arrive as a sapling during ordinary
    top-up, but as a full-grown tree when reseeded to break a floor deadlock,
    so the colony is not left waiting minutes for the sapling to mature."""
    y = terrain.ground_y(x)
    var = int(rng.integers(0, 1 << 30))
    if kind == KIND_TREE:
        if mature:
            return reg.spawn(KIND_TREE, x, y, scale=0.85 + 0.3 * float(rng.random()),
                             hp=TREE_HP, variant=var, state=_default_state(KIND_TREE))
        return reg.spawn(KIND_SAPLING, x, y, scale=0.30, hp=SAPLING_HP,
                         variant=var, state={"growth": 0.0})
    if kind == KIND_ROCK:
        return reg.spawn(KIND_ROCK, x, y, scale=0.7 + 0.4 * float(rng.random()),
                         hp=ROCK_HP, variant=var, state=_default_state(KIND_ROCK))
    return reg.spawn(KIND_BUSH, x, y, scale=0.55 + 0.35 * float(rng.random()),
                     hp=BUSH_HP, variant=var,
                     state={"berries_left": BUSH_MAX_BERRIES if mature else 0,
                            "regrow_t": 0.0, "burning": False, "burn_t": 0.0})


def _guaranteed_site(reg: PropRegistry, terrain: "Terrain",
                     rng: np.random.Generator) -> float | None:
    """A placement that relaxes spacing so the resource floor can never fail.

    _regrow_site respects spacing and can legitimately find nowhere; but the
    floor exists precisely to rescue a bare map, so when it triggers we fall
    back to any non-cliff, non-submerged column, spacing be damned.
    """
    try:
        w = int(terrain.W)
    except Exception:
        return None
    for _ in range(64):
        xi = int(rng.integers(8, max(9, w - 8)))
        try:
            if terrain.is_cliff(xi):
                continue
        except Exception:
            pass
        return float(xi)
    return float(w // 2)


def _regrow_site(
    reg: PropRegistry, terrain: "Terrain", rng: np.random.Generator, kind: str
) -> float | None:
    """A free, gently sloped, non-submerged column to reseed into."""
    try:
        w = int(terrain.W)
        mat = terrain.material
    except Exception:
        return None
    # Ash counts as fertile: burnt ground is exactly what needs to come back.
    soils = (MAT_GRASS, MAT_DIRT, MAT_ASH) if kind == KIND_TREE else (
        MAT_GRASS, MAT_DIRT, MAT_SAND, MAT_ASH
    )
    limit = 0.55 if kind == KIND_TREE else 0.85
    spacing = _MIN_SPACING.get(kind, 20.0)
    ponds = [
        (_f(p.state.get("x0"), 0.0), _f(p.state.get("x1"), 0.0))
        for p in reg.all_of(KIND_WATER)
    ]
    for _ in range(REGROW_TRIES):
        xi = int(rng.integers(6, max(7, w - 6)))
        if int(mat[xi]) not in soils:
            continue
        if abs(terrain.slope(float(xi))) > limit:
            continue
        # Belt and braces: the slope test above is per-column, is_cliff() is the
        # terrain's own verdict on whether anything can stand there.
        try:
            if bool(terrain.is_cliff(float(xi))):
                continue
        except Exception:
            pass
        if any(a - 6 <= xi <= b + 6 for a, b in ponds):
            continue
        near = False
        for other in reg._props.values():
            if not other.alive or other.kind in (KIND_SCORCH, KIND_WATER):
                continue
            need = spacing if other.kind == kind else _CROSS_KIND * max(
                spacing, _MIN_SPACING.get(other.kind, 20.0)
            )
            if abs(other.x - xi) < need:
                near = True
                break
        if not near:
            return float(xi)
    return None


def _tick_crop(p: Prop, dt: float, grow: float = 1.0) -> None:
    """Ripen toward harvest. `grow` (the colony regrowth factor) speeds it, so a
    bigger settlement's fields come in faster - the same logic as vegetation."""
    if p.state.get("ripe"):
        return
    g = _f(p.state.get("growth"), 0.0) + dt * max(0.2, grow) / max(1.0, CROP_GROW_SEC)
    p.state["growth"] = min(1.0, g)
    if g >= 1.0:
        p.state["ripe"] = True


def _tick_scorch(p: Prop, dt: float, events: list[dict], doomed: list[Prop]) -> None:
    age = _f(p.state.get("age"), 0.0) + dt
    p.state["age"] = age
    p.state["fade"] = float(max(0.0, 1.0 - age / SCORCH_FADE_SEC))
    if age >= SCORCH_FADE_SEC:
        doomed.append(p)
        events.append(_ev("prop_destroyed", p, cause="faded"))


# ------------------------------------------------------------------ scatter --


def scatter(
    terrain: "Terrain",
    rng: np.random.Generator,
    counts: dict[str, int] | None = None,
) -> PropRegistry:
    """Populate a fresh terrain with props sitting **on** the ground surface.

    Trees and bushes avoid cliffs, rocks favour stone, boulders want a slope to
    roll down, and water fills the deepest basins.  Deterministic for a given
    ``rng`` state.
    """
    reg = PropRegistry()
    want = dict(DEFAULT_COUNTS)
    if isinstance(counts, dict):
        for k, v in counts.items():
            if isinstance(k, str):
                want[k] = max(0, _i(v, 0))
    # Remember the requested vegetation density so the land can grow back to
    # it after fires and felling - see _regrow().
    reg.targets = {
        k: want.get(k, 0)
        for k in (KIND_TREE, KIND_BUSH, KIND_ROCK) if want.get(k, 0) > 0
    }

    try:
        w = int(terrain.W)
        slope_col = np.abs(terrain.column_slope())
        mat = terrain.material
    except Exception:
        return reg

    taken: list[tuple[float, str]] = []            # (x, kind)
    wet: list[tuple[int, int]] = []                # water spans, nothing grows here

    def free(x: float, kind: str, factor: float = 1.0) -> bool:
        r = _MIN_SPACING.get(kind, 20.0)
        for tx, tkind in taken:
            other = _MIN_SPACING.get(tkind, 20.0)
            need = r if tkind == kind else _CROSS_KIND * max(r, other)
            if abs(tx - x) < need * factor:
                return False
        return True

    def dry(x: float) -> bool:
        return not any(a - 6 <= x <= b + 6 for a, b in wet)

    # -- water first: everything else has to route around it ---------------
    for a, b, surface_y, floor_y in terrain.find_basins(min_width=34, min_depth=7.0):
        if len(wet) >= want.get(KIND_WATER, 0):
            break
        depth = min(float(floor_y - surface_y), WATER_MAX_DEPTH)
        if depth < 5.0:
            continue
        top = float(floor_y - depth)
        cx = 0.5 * (a + b)
        reg.spawn(
            KIND_WATER,
            cx,
            top,
            scale=float(b - a),
            variant=int(rng.integers(0, 1 << 30)),
            state={"x0": int(a), "x1": int(b), "surface_y": top, "depth": float(depth)},
        )
        try:
            terrain.set_material_span(int(a) - 3, int(b) + 2, MAT_SAND)
        except Exception:
            pass
        wet.append((int(a), int(b)))
        taken.append((cx, KIND_WATER))

    # -- land props --------------------------------------------------------
    def place(kind: str, n: int, ok: Callable[[int], bool], relaxed: Callable[[int], bool]) -> None:
        """Three passes: ideal spot, then relaxed terrain, then tighter packing."""
        placed = 0
        for test, factor, tries in (
            (ok, 1.0, n * 30),
            (relaxed, 1.0, n * 20),
            (relaxed, 0.6, n * 20),
        ):
            for _ in range(max(0, tries)):
                if placed >= n:
                    return
                xi = int(rng.integers(6, max(7, w - 6)))
                if not dry(xi) or not free(float(xi), kind, factor):
                    continue
                try:
                    if not test(xi):
                        continue
                except Exception:
                    continue
                _spawn_on_ground(reg, terrain, rng, kind, float(xi))
                taken.append((float(xi), kind))
                placed += 1

    def soil(i: int) -> bool:
        return int(mat[i]) in (MAT_GRASS, MAT_DIRT)

    # Boulders are the fussiest (they need a slope to roll down) and the
    # sparsest, so they get first refusal on the steep ground.
    place(
        KIND_BOULDER,
        want.get(KIND_BOULDER, 0),
        lambda i: 0.32 <= slope_col[i] <= 1.4,
        lambda i: 0.2 <= slope_col[i] <= 2.2,
    )
    place(
        KIND_TREE,
        want.get(KIND_TREE, 0),
        lambda i: slope_col[i] <= 0.55 and soil(i),
        lambda i: slope_col[i] <= 0.95,
    )
    place(
        KIND_SAPLING,
        want.get(KIND_SAPLING, 0),
        lambda i: slope_col[i] <= 0.5 and soil(i),
        lambda i: slope_col[i] <= 0.9,
    )
    place(
        KIND_BUSH,
        want.get(KIND_BUSH, 0),
        lambda i: slope_col[i] <= 0.85 and int(mat[i]) in (MAT_GRASS, MAT_DIRT, MAT_SAND),
        lambda i: slope_col[i] <= 1.3,
    )
    place(
        KIND_ROCK,
        want.get(KIND_ROCK, 0),
        lambda i: int(mat[i]) == MAT_STONE or slope_col[i] >= 0.45,
        lambda i: slope_col[i] <= 2.0,
    )
    return reg


def _spawn_on_ground(
    reg: PropRegistry,
    terrain: "Terrain",
    rng: np.random.Generator,
    kind: str,
    x: float,
) -> Prop:
    """Place one prop with kind-appropriate scale, hp and starting state."""
    y = terrain.ground_y(x)
    variant = int(rng.integers(0, 1 << 30))
    if kind == KIND_TREE:
        scale = float(TREE_SCALE[0] + (TREE_SCALE[1] - TREE_SCALE[0]) * float(rng.random()))
        return reg.spawn(kind, x, y, scale=scale, hp=TREE_HP, variant=variant)
    if kind == KIND_SAPLING:
        p = reg.spawn(kind, x, y, scale=0.32, hp=SAPLING_HP, variant=variant)
        p.state["growth"] = float(rng.random()) * 0.35
        return p
    if kind == KIND_BUSH:
        p = reg.spawn(
            kind, x, y, scale=0.55 + 0.45 * float(rng.random()), hp=BUSH_HP, variant=variant
        )
        p.state["berries_left"] = int(rng.integers(1, BUSH_MAX_BERRIES + 1))
        return p
    if kind == KIND_ROCK:
        return reg.spawn(
            kind, x, y, scale=0.5 + 0.7 * float(rng.random()), hp=ROCK_HP, variant=variant
        )
    if kind == KIND_BOULDER:
        return reg.spawn(
            kind, x, y, scale=0.9 + 0.6 * float(rng.random()), hp=BOULDER_HP, variant=variant
        )
    return reg.spawn(kind, x, y, variant=variant)


if __name__ == "__main__":  # pragma: no cover - run with: python -m backgrounded.sim.props
    import random as _pyrandom

    from .terrain import Terrain as _T

    _terr = _T.generate(11, "hills")
    _rng = np.random.default_rng(11)
    _reg = scatter(_terr, _rng, None)
    print("scattered:", _reg.counts(), "total", len(_reg))
    for _p in _reg:
        assert abs(_p.y - _terr.ground_y(_p.x)) < 1.0 or _p.kind == "water", _p.kind

    # Determinism.
    _reg2 = scatter(_T.generate(11, "hills"), np.random.default_rng(11), None)
    assert [p.to_dict() for p in _reg] == [p.to_dict() for p in _reg2], "scatter not deterministic"

    # Chop a tree down.
    _tree = _reg.all_of("tree")[0]
    while not chop(_tree, 1.0, _rng):
        pass
    _wood = 0
    for _ in range(60):
        for _e in _reg.tick(_terr, _rng, 1 / 30):
            if _e["type"] == "tree_felled":
                _wood = _e["wood"]
    assert 3 <= _wood <= 5 and _reg.get(_tree.id) is None, "tree felling"

    # Sapling matures.
    _sap = plant_sapling(_reg, _terr, 640.0, _rng)
    for _ in range(int(SAPLING_GROW_SEC * 31)):
        _reg.tick(_terr, _rng, 1 / 30)
        if _sap.kind == "tree":
            break
    assert _sap.kind == "tree" and _sap.hp == TREE_HP, "sapling growth"

    # Berries deplete and regrow.
    _bush = _reg.all_of("bush")[0]
    _bush.state["berries_left"] = 3
    assert harvest_berries(_bush, 3) == 3 and _bush.state["berries_left"] == 0
    for _ in range(int(BERRY_REGROW_SEC * 31)):
        _reg.tick(_terr, _rng, 1 / 30)
    assert _bush.state["berries_left"] >= 1, "berry regrow"

    # Fire spreads and leaves ash.
    _before = len(_reg.all_of("tree"))
    ignite(_reg.all_of("tree")[0])
    for _ in range(int(90 * 30)):
        _reg.tick(_terr, _rng, 1 / 30)
    _scorch = len(_reg.all_of("scorch"))
    print("fire: trees", _before, "->", len(_reg.all_of("tree")), "scorches", _scorch)
    assert _scorch >= 1 and (_terr.material == MAT_ASH).any(), "fire left no scar"

    # Boulders sit still until nudged, then roll downhill and crush things.
    # Start at the top of the longest sustained descent, not merely the
    # steepest pixel - a lone steep pixel can drop straight into a hollow.
    _h = _terr.height
    _drop = np.maximum(_h[220:1060] - _h[100:940], _h[100:940] - _h[220:1060])
    _bx = float(int(np.argmax(_drop)) + 100)
    _b = _reg.spawn("boulder", _bx, _terr.ground_y(_bx))
    for _ in range(120):
        _reg.tick(_terr, _rng, 1 / 30)
    assert _b.x == _bx, "a resting boulder moved on its own"
    assert nudge(_b)
    _y0 = _b.y
    for _ in range(900):
        if not _b.state["rolling"]:
            break
        _reg.tick(_terr, _rng, 1 / 30)
    print(f"boulder {_bx:.0f} -> {_b.x:.0f}  (y {_y0:.0f} -> {_b.y:.0f})")
    assert abs(_b.x - _bx) > 25.0 and _b.y >= _y0 - 1.0, "boulder did not roll downhill"

    # ------------------------------------------------ headcount-scaled recovery
    class _W:  # stands in for World: all props needs is regrowth_factor()
        def __init__(self, f: float) -> None:
            self._f = f

        def regrowth_factor(self) -> float:
            return self._f

    assert growth_factor(None) == 1.0
    assert growth_factor(_W(2.5)) == 2.5
    assert growth_factor(object()) == 1.0                 # no such method
    assert growth_factor(_W(float("nan"))) == 1.0         # garbage -> 1.0
    assert growth_factor(_W(99.0)) == GROWTH_FACTOR_MAX   # clamped
    assert growth_factor(2.0) == 2.0                      # a bare number works

    # Saplings mature `factor` times faster.
    def _mature_secs(factor: float) -> float:
        t = _T.generate(3, "hills")
        r = PropRegistry()
        s = plant_sapling(r, t, 600.0, np.random.default_rng(3))
        w = _W(factor)
        g = np.random.default_rng(3)
        secs = 0.0
        while s.kind == "sapling" and secs < SAPLING_GROW_SEC * 2:
            r.tick(t, g, 1 / 30, w)
            secs += 1 / 30
        return secs

    _one, _three = _mature_secs(1.0), _mature_secs(3.0)
    print(f"sapling matures: x1 {_one:.0f}s  x3 {_three:.0f}s")
    assert abs(_one - SAPLING_GROW_SEC) < 1.0, _one
    assert abs(_three - SAPLING_GROW_SEC / 3.0) < 1.0, _three

    # Berries too.
    def _berries(factor: float) -> int:
        t = _T.generate(3, "hills")
        r = PropRegistry()
        b = r.spawn("bush", 600.0, t.ground_y(600.0), state={"berries_left": 0})
        g = np.random.default_rng(3)
        for _ in range(int(BERRY_REGROW_SEC * 30)):
            r.tick(t, g, 1 / 30, _W(factor))
        return _i(b.state.get("berries_left"), 0)

    assert _berries(1.0) == 1 and _berries(3.0) == 3, (_berries(1.0), _berries(3.0))

    # A stripped map reseeds itself, faster for a big colony than a small one.
    def _stripped(seed: int) -> tuple:
        t = _T.generate(seed, "hills")
        r = scatter(t, np.random.default_rng(seed),
                    {"tree": 14, "bush": 10, "rock": 6, "boulder": 2, "sapling": 0})
        for q in list(r):
            if q.kind in ("tree", "sapling", "bush"):
                r.remove(q)
        return t, r

    def _recover(factor: float, seconds: float, seed: int = 5) -> tuple[int, int]:
        t, r = _stripped(seed)
        w = _W(factor)
        g = np.random.default_rng(seed ^ 0xA5)
        for _ in range(int(seconds * 30)):
            r.tick(t, g, 1 / 30, w)
        c = r.counts()
        return c.get("tree", 0) + c.get("sapling", 0), c.get("bush", 0)

    _slow = _recover(1.0, 400.0)
    _fast = _recover(2.76, 400.0)
    print(f"stripped map after 400s: pop2 {_slow[0]} trees {_slow[1]} bushes | "
          f"pop10 {_fast[0]} trees {_fast[1]} bushes")
    assert _slow[0] > 0 and _slow[1] > 0, "a stripped map never recovered at all"
    assert _fast[0] >= _slow[0] and _fast[1] >= _slow[1], "factor did not speed recovery"
    assert sum(_fast) > sum(_slow), "recovery ignored the headcount factor"

    # ...and never past the density it started with.
    _cap_t, _cap_r = _stripped(5)
    _cap_g = np.random.default_rng(99)
    for _ in range(int(2400 * 30)):
        _cap_r.tick(_cap_t, _cap_g, 1 / 30, _W(3.0))
    _cap = _cap_r.counts()
    assert _cap.get("tree", 0) + _cap.get("sapling", 0) <= 14, _cap
    assert _cap.get("bush", 0) <= 10, _cap

    # Destruction must NOT scale: a tree burns for BURN_SEC either way.
    def _burn_secs(factor: float) -> float:
        t = _T.generate(7, "hills")
        r = PropRegistry()
        v = r.spawn("tree", 600.0, t.ground_y(600.0), hp=TREE_HP)
        ignite(v)
        g = np.random.default_rng(7)
        secs = 0.0
        while r.get(v.id) is not None and secs < BURN_SEC["tree"] * 3:
            r.tick(t, g, 1 / 30, _W(factor))
            secs += 1 / 30
        return secs

    _b1, _b3 = _burn_secs(1.0), _burn_secs(3.0)
    print(f"tree burns out: x1 {_b1:.1f}s  x3 {_b3:.1f}s  (must match)")
    assert abs(_b1 - _b3) < 0.1 and abs(_b1 - BURN_SEC["tree"]) < 0.2, (_b1, _b3)

    # ---------------------------------------------------------------- litter --
    from ..constants import LITTER_DECAYS as _DECAYS

    class _LW:  # stands in for World: littering needs pyrng, agents, world_time
        def __init__(self, xs: list[float]) -> None:
            self.pyrng = _pyrandom.Random(4)
            self.world_time = 0.0
            self.agents = [type("A", (), {"x": x, "alive": True, "inside": None})()
                           for x in xs]

    _lt = _T.generate(5, "hills")
    _lr = PropRegistry()
    _lw = _LW([300.0, 320.0, 340.0, 900.0])
    for _ in range(int(600 * 30)):                 # ten sim-minutes, four people
        _lr.tick(_lt, np.random.default_rng(5), 1 / 30, _lw)
    _n = len(_lr.all_of(KIND_LITTER))
    print(f"litter after 600s with 4 people: {_n}  (expect ~{4 * 600 / LITTER_DROP_SEC:.0f})")
    assert _n > 0, "nobody littered at all"
    assert abs(_n - 4 * 600 / LITTER_DROP_SEC) < 20, _n

    # Same seed, same litter - the drop rolls come off world.pyrng, so a run
    # that is not reproducible here is not reproducible anywhere.
    _lr2 = PropRegistry()
    _lw2 = _LW([300.0, 320.0, 340.0, 900.0])
    for _ in range(int(600 * 30)):
        _lr2.tick(_lt, np.random.default_rng(5), 1 / 30, _lw2)
    assert [p.to_dict() for p in _lr2.all_of(KIND_LITTER)] == \
           [p.to_dict() for p in _lr.all_of(KIND_LITTER)], "littering is not seeded"

    # ...and it does NOT decay. This is the assertion that makes LITTER_DECAYS a
    # decision rather than a comment: flipping it means changing this test too.
    assert not _DECAYS
    _lw.agents = []                                # everyone leaves; nobody drops
    for _ in range(int(900 * 30)):
        _lr.tick(_lt, np.random.default_rng(5), 1 / 30, _lw)
    assert len(_lr.all_of(KIND_LITTER)) == _n, "litter decayed on its own"

    # The cap recycles the oldest rather than refusing to drop.
    _lw.agents = _LW([500.0] * 6).agents
    for _ in range(int(4000 * 30)):
        _lr.tick(_lt, np.random.default_rng(5), 1 / 30, _lw)
    _capped = len(_lr.all_of(KIND_LITTER))
    print(f"litter cap holds at {_capped} (max {LITTER_MAX})")
    assert _capped <= LITTER_MAX, _capped

    # Persistence round trip.
    _rt = PropRegistry.from_dict(_reg.to_dict())
    assert [p.to_dict() for p in _rt] == [p.to_dict() for p in _reg], "registry round trip"
    for _bad in [None, {}, {"props": "nope"}, {"props": [None, 3, {"kind": "zzz"}]}]:
        assert isinstance(PropRegistry.from_dict(_bad), PropRegistry)
    assert Prop.from_dict({}).kind in KINDS
    print("props smoke test OK")
