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
    MAX_SLOPE_CLIMB,
    MAX_SLOPE_WALK,
    RENDER_H,
    RENDER_W,
    SCENE_ASHFALL,
    SCENE_BLIZZARD,
    SCENE_NIGHT_STORM,
    SCENE_WILDFIRE,
    STICK_PALETTE,
    WALK_SPEED,
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
    "ROLE_GATHERER", "ROLE_BUILDER", "ROLE_ELDER", "ROLE_CHILD",
    "ROLE_LOOKOUT", "ROLES", "ADULT_ROLES",
]

# ------------------------------------------------------------------ tuning --
#: Nominal head-to-heel height of an adult, in world pixels. render/ scales its
#: skeleton from this so sim and render agree on how big a person is. 26 px in
#: a 1280x800 world is the smallest size at which a two-pixel-wide figure still
#: reads as a *pose* rather than a smudge, once the frame is upscaled to 2560.
AGENT_HEIGHT = 26.0

GRAVITY = 900.0             # px/s^2
TERMINAL_VY = 900.0         # px/s, matches the lethal threshold scale
AIR_DRAG = 1.6              # per second horizontal damping while airborne
GROUND_SNAP = 1.5           # px tolerance for "standing on the ground"
STEP_DROP_MAX = 5.0         # px of ground drop per step before you're falling
STEP_PROBE = 3.0            # px lookahead used to read the slope ahead
ARRIVE_EPS = 1.5            # px, close enough to the target
SLIP_CHANCE_PER_SEC = 0.42  # base slip probability while climbing, per second
SPEECH_DEFAULT_SEC = 3.0
GRAVE_DELAY = 7.0           # s a corpse lies there before it becomes a grave

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
_EDGE_MIN = 0.0
_EDGE_MAX = float(RENDER_W - 1)

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
    alive: bool = True
    anim_t: float = 0.0                   # animation phase accumulator

    # --- movement intent --------------------------------------------------
    target_x: float | None = None         # walk here; None = stand still
    path_goal: str | None = None          # opaque label for *why* ("stockpile",
                                          # "tree:12"); actions own its meaning

    # --- timers -----------------------------------------------------------
    climb_t: float = 0.0                  # s spent on the current climb
    fall_t: float = 0.0                   # s spent airborne
    dead_t: float = 0.0                   # s since death, drives the grave
    speech: str | None = None             # symbol key for the speech bubble
    speech_t: float = 0.0                 # s of bubble left

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
        """Approximate y of the top of the head."""
        return self.y - AGENT_HEIGHT * (0.72 if self.role == ROLE_CHILD else 1.0)

    def height(self) -> float:
        """Drawn height for this agent - children are smaller, elders stoop."""
        if self.role == ROLE_CHILD:
            return AGENT_HEIGHT * 0.68
        if self.role == ROLE_ELDER:
            return AGENT_HEIGHT * 0.94
        return AGENT_HEIGHT

    def hand_anchor(self) -> tuple[float, float]:
        """Roughly where the front hand is - candle light attaches here.

        An approximation on purpose: lighting.py lives in sim/ and cannot ask
        render/ for the exact skeletal hand joint.
        """
        h = self.height()
        return (self.x + self.facing * 0.20 * h, self.y - 0.46 * h)

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
        ``FALL_LETHAL_SPEED``, and clamps x to the screen. Never raises.
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
            risk = (
                SLIP_CHANCE_PER_SEC
                * dt
                * (0.55 + 0.85 * self.fatigue + 0.25 * self.hunger)
                * _clamp(steep / max(1e-3, MAX_SLOPE_CLIMB), 0.2, 1.0)
            )
            if _rand_of(rng) < risk:
                self._slip(d)
                return
        elif rise <= 0.0:
            # A drop ahead: walk up to the lip, the snap check pushes us off.
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
            if lethal and impact > FALL_LETHAL_SPEED:
                self.die("fall")
            else:
                self.fall_t = 0.0

    def _clamp_edges(self) -> None:
        """The screen edges are the edge of the world. Hard walls, no escape."""
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
            "alive": bool(self.alive),
            "anim_t": float(self.anim_t),
            "target_x": None if self.target_x is None else float(self.target_x),
            "path_goal": self.path_goal,
            "climb_t": float(self.climb_t),
            "fall_t": float(self.fall_t),
            "dead_t": float(self.dead_t),
            "speech": self.speech,
            "speech_t": float(self.speech_t),
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
        s.id = _i(d.get("id"), 0)
        s.name = _s(d.get("name"), None) or f"Agent{s.id}"
        s.color = _color(d.get("color"), color_for_id(s.id))
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
        s.carry_qty = max(0, _i(d.get("carry_qty"), 0))
        role = _s(d.get("role"), ROLE_GATHERER) or ROLE_GATHERER
        s.role = role if is_role(role) else ROLE_GATHERER
        s.generation = max(0, _i(d.get("generation"), 0))
        s.holds_candle = _b(d.get("holds_candle"), False)
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
        s.birth_time = _f(d.get("birth_time"), 0.0)

        parents = d.get("parent_ids")
        if isinstance(parents, (list, tuple)):
            s.parent_ids = [_i(p, 0) for p in parents if _i(p, -1) >= 0]

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

        s = Stickman(
            id=agent_id,
            name=chosen,
            color=color_for_id(agent_id),
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

        pop.next_id = max(1, _i(d.get("next_id"), 1))
        for a in pop.agents:
            pop.next_id = max(pop.next_id, a.id + 1)
        pop.generation = max(
            _i(d.get("generation"), 0),
            max((a.generation for a in pop.agents), default=0),
        )
        pop.births = max(0, _i(d.get("births"), len(pop.agents)))
        corpses = sum(1 for a in pop.agents if not a.alive)
        if "graves_raised" in d:
            pop.graves_raised = max(0, _i(d.get("graves_raised"), 0))
        else:                                 # pre-graves_raised save
            pop.graves_raised = max(0, _i(d.get("deaths"), 0) - corpses)
        pop.peak = max(_i(d.get("peak"), 0), pop.alive_count())
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

    walker = pop.agents[0]
    walker.walk_to(900.0, "beyond the drop")
    for _ in range(int(30 * 40)):
        pop.apply_physics(1.0 / 30.0, terrain, rng)

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
    clear_death_listeners()
