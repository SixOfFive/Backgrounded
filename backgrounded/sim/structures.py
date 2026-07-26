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
    HUT_AGE_WEIGHT,
    HUT_GROWTH_AGE_SEC,
    HUT_GROWTH_STORE_REF,
    HUT_SCALE_MAX,
    HUT_SCALE_MIN,
    HUT_STORE_WEIGHT,
    RENDER_H,
    RENDER_W,
    RES_FIBRE,
    RES_STONE,
    RES_WOOD,
)

log = logging.getLogger(__name__)

__all__ = [
    "Structure",
    "StructureRegistry",
    "StructureSpec",
    "STRUCTURE_SPECS",
    "STRUCTURE_KINDS",
    "structure_spec",
    "FIRE_LIGHT_COLOR",
    "BURN_LIGHT_COLOR",
]

# ------------------------------------------------------------------ tuning --
FIRE_LIGHT_COLOR: tuple[int, int, int] = (255, 168, 82)
BURN_LIGHT_COLOR: tuple[int, int, int] = (255, 112, 40)

BURN_DPS = 9.0              # hp lost per second while burning
BURN_SPREAD_CHANCE = 0.35   # per second, chance a burning structure ignites a neighbour
BURN_NEIGHBOUR_DIST = 70.0
FIRE_FUEL_BURN = 1.0 / 240.0   # a full firepit burns for ~4 minutes
FIRE_STOKE_PER_WOOD = 0.34
RUIN_LINGER = 240.0         # seconds a collapsed structure stays as rubble

#: Sleepers a hut holds at growth 0 and at growth 1. A well-established hut in
#: a well-stocked camp really is a bigger building, so it houses more people.
HUT_CAPACITY_MIN = 2
HUT_CAPACITY_MAX = 5
#: Fraction of the remaining gap `growth` closes toward its target per second.
#: Deliberately slow: a single haul arriving must not pop the building bigger,
#: and running the stores down has to read as a settlement contracting rather
#: than a building blinking. ~5%/s is a ~20 s time constant.
GROWTH_LERP_RATE = 0.05


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
    "wall": StructureSpec(
        "wall", 2, 170.0, {RES_STONE: 14, RES_WOOD: 4},
        width=30.0, height=30.0, capacity=0, flammable=False,
        build_time=7.0, spacing=28.0, variants=3,
    ),
    "bridge": StructureSpec(
        "bridge", 3, 120.0, {RES_WOOD: 20, RES_FIBRE: 4},
        width=90.0, height=10.0, capacity=0, flammable=True,
        build_time=10.0, spacing=80.0, variants=2,
    ),
    "watchtower": StructureSpec(
        "watchtower", 3, 120.0, {RES_WOOD: 22, RES_STONE: 8},
        width=30.0, height=74.0, capacity=1, flammable=True,
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
        if self.kind == "firepit":
            self.state.setdefault("lit", True)
            self.state.setdefault("fuel", 1.0)
        elif self.kind == "totem":
            self.state.setdefault("glow", 1.0)
        elif self.kind == "stockpile":
            self.state.setdefault("shown", {})

    # --------------------------------------------------------------- damage --
    @property
    def is_ruined(self) -> bool:
        return bool(self.state.get("ruined"))

    @property
    def is_burning(self) -> bool:
        return bool(self.state.get("burning"))

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

    # ------------------------------------------------------------------ fire --
    def ignite(self) -> bool:
        """Set alight. Returns True if it actually caught."""
        if self.is_ruined or not self.spec.flammable:
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
        if self.kind != "firepit" or not self.built or self.is_ruined:
            return False
        self.state["lit"] = True
        self.state["fuel"] = max(float(self.state.get("fuel", 0.0)), _clamp01(fuel))
        return True

    def stoke(self, wood: int = 1) -> None:
        """Feed wood to a lit (or unlit but built) firepit."""
        if self.kind != "firepit" or self.is_ruined or not self.built:
            return
        fuel = float(self.state.get("fuel", 0.0)) + FIRE_STOKE_PER_WOOD * max(0, int(wood))
        self.state["fuel"] = _clamp01(fuel)
        if self.state["fuel"] > 0.0:
            self.state["lit"] = True

    @property
    def fire_active(self) -> bool:
        return (
            self.kind == "firepit"
            and self.built
            and not self.is_ruined
            and bool(self.state.get("lit"))
            and float(self.state.get("fuel", 0.0)) > 0.0
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
                self._chronicle(world, f"The {self.kind} burned to the ground.")
                return
        if self.kind == "firepit" and self.state.get("lit"):
            fuel = float(self.state.get("fuel", 0.0)) - FIRE_FUEL_BURN * dt
            if fuel <= 0.0:
                self.state["fuel"] = 0.0
                self.state["lit"] = False
            else:
                self.state["fuel"] = fuel
        if self.kind == "totem" and self.built:
            self.state["glow"] = _clamp01(float(self.state.get("glow", 1.0)))

    @staticmethod
    def _chronicle(world: Any | None, text: str) -> None:
        if world is None:
            return
        try:
            fn = getattr(world, "chronicle", None)
            if callable(fn):
                fn(text)
            elif fn is not None and hasattr(fn, "add"):
                fn.add(text)
        except Exception:
            pass

    # ------------------------------------------------------------ serialise --
    def to_dict(self) -> dict[str, Any]:
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
            "state": _json_safe(self.state) or {},
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
        s = cls(
            id=_as_int(d, "id", 0),
            kind=kind,
            x=_as_float(d, "x", RENDER_W * 0.5),
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

    def colony_center(self, default: float = RENDER_W * 0.5) -> float:
        xs = [s.x for s in self._by_id.values() if s.kind != "grave" and not s.is_ruined]
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
