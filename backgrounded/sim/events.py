"""EventSystem: weather scenes, disasters and the things they do to the world.

Pure data + numpy. **No pygame** — this module must stay importable headless.

The system owns the current scene, its intensity envelope, the weather
channels (``wind``/``rain``/``snow``/``ash``), scheduled one-shot events and
the transient geometry the renderer draws (``strikes``, ``meteors``,
``water_level``, ``shake_offset()``).

``tick(world, dt)`` is the only entry point that mutates anything. It applies
scene consequences to the world: terrain deformation and repainting, prop
ignition, agent panic and death, and flashes pushed into ``world.lighting``.
Every phase is individually guarded — a broken subsystem disables itself for
that tick rather than taking the whole app down.

World coupling is deliberately duck-typed so the modules other agents own can
evolve. What is actually used, all optional:

    world.terrain      .height (np.float32[W])  .material (np.uint8[W])
                       .ground_y(x) .deform(x0,x1,dy) .crater(x,y,r) .paint(x0,x1,m)
    world.lighting     .add_flash(i, decay, color) ; .wind_gust attribute
    world.agents       iterable of Stickman (.x .y .alive .warmth .morale .id)
    world.props        iterable of Prop (.x .y .kind .burning .ignite())
    world.structures   iterable, same shape as props
    world.cfg/.config  Config (auto_scene_change, scene_min_sec)
    world.kill_agent(agent, cause)      preferred death path if present
    world.chronicle.add(text)           preferred history path if present

Sign convention used for terrain: ``height[x]`` is the *y* of the surface, so
**+dy lowers the ground (digs) and -dy raises it**.

Panic protocol (honoured by behavior.py if it wants to): events set
``agent.panic`` (seconds), ``agent.panic_target_x`` and ``agent.panic_reason``.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any

import numpy as np

from ..constants import (
    MAT_ASH,
    MAT_DIRT,
    MAT_MUD,
    MAT_SNOW,
    RENDER_H,
    RENDER_W,
    SCENE_ASHFALL,
    SCENE_BLIZZARD,
    SCENE_CLEAR,
    SCENE_FLOOD,
    SCENE_METEOR,
    SCENE_MUDSLIDE,
    SCENE_NIGHT_STORM,
    SCENE_WILDFIRE,
    SCENES,
)
from .lighting import LIGHTNING_COLOR, METEOR_COLOR, daylight_factor, is_night

log = logging.getLogger(__name__)

__all__ = ["EventSystem"]

# ------------------------------------------------------------------ tuning --
STRIKE_TTL = 0.5                 # seconds a bolt stays drawable
STRIKE_MIN, STRIKE_MAX = 4.0, 12.0
DIRECT_HIT_CHANCE = 0.08
DIRECT_KILL_RADIUS = 25.0
SCORCH_HALF_WIDTH = 14.0

METEOR_MIN, METEOR_MAX = 1.6, 5.0
METEOR_IMPACT_CHANCE = 0.32
METEOR_TTL = 0.45                # afterglow once the streak lands

MUDSLIDE_WARN = 4.0
MUDSLIDE_SLIDE = 3.0
MUDSLIDE_SETTLE = 2.5

FLOOD_RISE = 40.0
FLOOD_HOLD = 22.0
FLOOD_FALL = 34.0
DROWN_SEC = 7.0

SNOW_MAX_DEPTH = 3.2             # px the ground rises under a full snow layer
SNOW_RATE = 0.055                # px/s
ASH_MAX_DEPTH = 2.4
ASH_RATE = 0.030

BURN_DEATH_SEC = 3.0
BURN_TOUCH_DIST = 13.0
FLEE_DIST = 92.0

SHAKE_DECAY = 3.4                # exponential decay rate of screen shake

#: scene -> handler method name. Kept as strings so the instance holds no
#: bound-method cycles and stays trivially serialisable.
_HANDLERS: dict[str, str] = {
    SCENE_NIGHT_STORM: "_scene_night_storm",
    SCENE_CLEAR: "_scene_clear",
    SCENE_WILDFIRE: "_scene_wildfire",
    SCENE_MUDSLIDE: "_scene_mudslide",
    SCENE_BLIZZARD: "_scene_blizzard",
    SCENE_FLOOD: "_scene_flood",
    SCENE_METEOR: "_scene_meteor",
    SCENE_ASHFALL: "_scene_ashfall",
}

#: plausible successors, weighted. Storms clear, fires leave ash, etc.
_TRANSITIONS: dict[str, dict[str, float]] = {
    SCENE_NIGHT_STORM: {SCENE_CLEAR: 5, SCENE_FLOOD: 2, SCENE_MUDSLIDE: 2,
                        SCENE_WILDFIRE: 1, SCENE_ASHFALL: 1},
    SCENE_CLEAR:       {SCENE_NIGHT_STORM: 4, SCENE_WILDFIRE: 3, SCENE_BLIZZARD: 2,
                        SCENE_METEOR: 2, SCENE_ASHFALL: 1},
    SCENE_WILDFIRE:    {SCENE_ASHFALL: 3, SCENE_CLEAR: 3, SCENE_NIGHT_STORM: 2},
    SCENE_MUDSLIDE:    {SCENE_CLEAR: 4, SCENE_NIGHT_STORM: 2, SCENE_FLOOD: 2},
    SCENE_BLIZZARD:    {SCENE_CLEAR: 5, SCENE_NIGHT_STORM: 2},
    SCENE_FLOOD:       {SCENE_CLEAR: 4, SCENE_NIGHT_STORM: 2, SCENE_MUDSLIDE: 1},
    SCENE_METEOR:      {SCENE_WILDFIRE: 3, SCENE_CLEAR: 3, SCENE_ASHFALL: 2,
                        SCENE_NIGHT_STORM: 2},
    SCENE_ASHFALL:     {SCENE_CLEAR: 4, SCENE_NIGHT_STORM: 2, SCENE_BLIZZARD: 1},
}

_FLAMMABLE = ("tree", "bush", "sapling", "shrub", "log", "grass", "hut", "wall",
              "watchtower", "bridge", "totem", "scaffold")


# ------------------------------------------------------- generic utilities --
def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _fnum(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):
        return default
    return f


def _approach(cur: float, target: float, rate: float, dt: float) -> float:
    step = rate * dt
    if cur < target:
        return min(target, cur + step)
    if cur > target:
        return max(target, cur - step)
    return cur


def _iter(obj: Any, names: tuple[str, ...]) -> list[Any]:
    if obj is None:
        return []
    for name in names:
        v = getattr(obj, name, None)
        if v is None:
            continue
        if isinstance(v, dict):
            return list(v.values())
        try:
            return list(v)
        except TypeError:
            continue
    return []


def _agents(world: Any) -> list[Any]:
    return [a for a in _iter(world, ("agents", "stickmen", "people"))
            if getattr(a, "alive", True)]


def _props(world: Any) -> list[Any]:
    out = _iter(world, ("props",))
    out.extend(_iter(world, ("structures",)))
    return out


def _aid(obj: Any) -> int | None:
    v = getattr(obj, "id", None)
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------ world access --
def _terrain(world: Any) -> Any:
    return getattr(world, "terrain", None)


def _height(world: Any) -> np.ndarray | None:
    h = getattr(_terrain(world), "height", None)
    if isinstance(h, np.ndarray) and h.ndim == 1 and h.size > 0:
        return h
    return None


def _material(world: Any) -> np.ndarray | None:
    m = getattr(_terrain(world), "material", None)
    if isinstance(m, np.ndarray) and m.ndim == 1 and m.size > 0:
        return m
    return None


def _ground_y(world: Any, x: float) -> float:
    t = _terrain(world)
    if t is not None:
        fn = getattr(t, "ground_y", None)
        if callable(fn):
            try:
                v = float(fn(float(x)))
                if v == v:
                    return v
            except Exception:
                pass
        h = _height(world)
        if h is not None:
            i = int(max(0, min(h.size - 1, int(x))))
            return float(h[i])
    return RENDER_H * 0.72


def _deform(world: Any, x0: float, x1: float, dy: float) -> None:
    """+dy lowers the surface (digs), -dy raises it. Never raises."""
    t = _terrain(world)
    if t is None:
        return
    fn = getattr(t, "deform", None)
    if callable(fn):
        try:
            fn(int(x0), int(x1), float(dy))
            return
        except Exception:
            pass
    h = _height(world)
    if h is None:
        return
    a = max(0, min(h.size, int(x0)))
    b = max(0, min(h.size, int(x1)))
    if b > a:
        h[a:b] = np.clip(h[a:b] + np.float32(dy), 12.0, float(RENDER_H - 4))


def _paint(world: Any, x0: float, x1: float, mat: int) -> None:
    t = _terrain(world)
    if t is None:
        return
    fn = getattr(t, "paint", None)
    if callable(fn):
        try:
            fn(int(x0), int(x1), int(mat))
            return
        except Exception:
            pass
    m = _material(world)
    if m is None:
        return
    a = max(0, min(m.size, int(x0)))
    b = max(0, min(m.size, int(x1)))
    if b > a:
        m[a:b] = np.uint8(mat)


def _crater(world: Any, x: float, y: float, radius: float) -> None:
    t = _terrain(world)
    if t is None:
        return
    fn = getattr(t, "crater", None)
    if callable(fn):
        # Terrain.crater is heightmap-column based: crater(cx, radius, depth).
        # It takes no y - the impact height is implicit in the surface. Passing
        # the screen y here as `radius` is arity-valid, so it raises nothing and
        # silently craters the entire 1280-column map instead of a small bowl.
        try:
            fn(int(x), int(max(3.0, radius)), float(max(3.0, radius) * 0.55))
            return
        except Exception:
            pass
    h = _height(world)
    if h is None:
        return
    r = max(3.0, float(radius))
    depth = r * 0.55
    a = max(0, int(x - r))
    b = min(h.size, int(x + r) + 1)
    if b <= a:
        return
    u = (np.arange(a, b, dtype=np.float32) - np.float32(x)) / np.float32(r)
    bowl = np.clip(1.0 - u * u, 0.0, 1.0)
    h[a:b] = np.clip(h[a:b] + bowl * np.float32(depth), 12.0, float(RENDER_H - 4))


def _lighting(world: Any) -> Any:
    return getattr(world, "lighting", None)


def _flash(world: Any, intensity: float, decay: float,
           color: tuple[int, int, int]) -> None:
    lig = _lighting(world)
    fn = getattr(lig, "add_flash", None)
    if callable(fn):
        try:
            fn(float(intensity), float(decay), tuple(color))
        except Exception:
            pass


def _chronicle(world: Any, text: str) -> None:
    if world is None or not text:
        return
    for name in ("chronicle", "history", "log"):
        obj = getattr(world, name, None)
        if obj is None:
            continue
        for meth in ("add", "record", "append", "log"):
            fn = getattr(obj, meth, None)
            if callable(fn):
                try:
                    fn(text)
                    return
                except Exception:
                    continue
    fn = getattr(world, "log_event", None)
    if callable(fn):
        try:
            fn(text)
        except Exception:
            pass


def _kill(world: Any, agent: Any, cause: str) -> bool:
    """Best-effort death. Prefers world.kill_agent, falls back to alive=False."""
    if agent is None or not getattr(agent, "alive", True):
        return False
    fn = getattr(world, "kill_agent", None)
    if callable(fn):
        try:
            fn(agent, cause)
            return True
        except TypeError:
            try:
                fn(agent)
                return True
            except Exception:
                pass
        except Exception:
            pass
    for meth in ("kill", "die"):
        fn = getattr(agent, meth, None)
        if callable(fn):
            try:
                fn(cause)
                return True
            except TypeError:
                try:
                    fn()
                    return True
                except Exception:
                    pass
            except Exception:
                pass
    try:
        agent.alive = False
    except Exception:
        return False
    try:
        agent.death_cause = cause
    except Exception:
        pass
    _chronicle(world, f"{getattr(agent, 'name', 'A stickman')} {cause}.")
    return True


def _panic(world: Any, agent: Any, target_x: float, duration: float,
           reason: str) -> None:
    fn = getattr(world, "panic", None)
    if callable(fn):
        try:
            fn(agent, float(target_x), float(duration), reason)
            return
        except TypeError:
            try:
                fn(agent, float(target_x), float(duration))
                return
            except Exception:
                pass
        except Exception:
            pass
    try:
        cur = _fnum(getattr(agent, "panic", 0.0), 0.0)
        agent.panic = max(cur, float(duration))
        agent.panic_target_x = _clamp(float(target_x), 6.0, RENDER_W - 6.0)
        agent.panic_reason = reason
    except Exception:
        pass
    try:
        agent.morale = _clamp(_fnum(getattr(agent, "morale", 0.5), 0.5) - 0.02)
    except Exception:
        pass


def _is_flammable(prop: Any) -> bool:
    if getattr(prop, "burning", False) or getattr(prop, "burnt", False):
        return False
    if getattr(prop, "destroyed", False) or not getattr(prop, "alive", True):
        return False
    flag = getattr(prop, "flammable", None)
    if isinstance(flag, bool):
        return flag
    for attr in ("kind", "type", "name"):
        v = getattr(prop, attr, None)
        if isinstance(v, str) and v:
            return v.lower() in _FLAMMABLE
    return False


def _ignite(world: Any, prop: Any) -> bool:
    if prop is None:
        return False
    for meth in ("ignite", "set_fire"):
        fn = getattr(prop, meth, None)
        if callable(fn):
            try:
                fn()
                return True
            except TypeError:
                try:
                    fn(1.0)
                    return True
                except Exception:
                    pass
            except Exception:
                pass
    try:
        if getattr(prop, "burning", False):
            return False
        prop.burning = True
        prop.burn_t = 0.0
        return True
    except Exception:
        return False


def _extinguish(prop: Any) -> None:
    fn = getattr(prop, "extinguish", None)
    if callable(fn):
        try:
            fn()
            return
        except Exception:
            pass
    try:
        prop.burning = False
    except Exception:
        pass


def _nearest_flammable(world: Any, x: float, max_dist: float) -> Any:
    best, best_d = None, max_dist
    for p in _props(world):
        if not _is_flammable(p):
            continue
        d = abs(_fnum(getattr(p, "x", 1e9), 1e9) - x)
        if d < best_d:
            best, best_d = p, d
    return best


# ================================================================= system ==
class EventSystem:
    """Drives weather and disasters, and applies their consequences."""

    def __init__(self, scene: str = SCENE_NIGHT_STORM, seed: int | None = None) -> None:
        self.scene: str = scene if scene in SCENES else SCENE_NIGHT_STORM
        self.scene_t: float = 0.0
        self.intensity: float = 0.0            # 0..1 envelope, ramps in on entry

        # weather channels ------------------------------------------------
        self.wind: float = 0.0                 # -1..1, signed (screen x)
        self.rain: float = 0.0                 # 0..1
        self.snow: float = 0.0                 # 0..1
        self.ash: float = 0.0                  # 0..1
        self.gust: float = 0.0                 # 0..1 candle guttering strength
        self.water_level: float | None = None  # y of the flood line, or None
        self.quake_t: float = 0.0              # seconds of quake left

        # renderer-facing transients --------------------------------------
        self.pending: list[dict[str, Any]] = []
        self.strikes: list[dict[str, Any]] = []
        self.meteors: list[dict[str, Any]] = []
        self.ember_rate: float = 0.0           # 0..1 particle budget hint
        self.tint: tuple[int, int, int] = (255, 255, 255)
        self.rumble: float = 0.0               # 0..1 mudslide warning
        self.fireflies: bool = False
        self.shake_amp: float = 0.0
        self.shake_t: float = 0.0

        # internal --------------------------------------------------------
        self.t: float = 0.0
        self.seed: int = int(seed) if seed is not None else random.randrange(1 << 30)
        self._rng = random.Random(self.seed)
        self.next_strike: float = self._rng.uniform(2.0, 6.0)
        self.next_meteor: float = self._rng.uniform(1.0, 3.0)
        self.next_ignite: float = 0.5
        self.snow_depth: float = 0.0
        self.ash_depth: float = 0.0
        self._snow_prev: list[int] | None = None
        self.slide_phase: str = "idle"
        self.slide_x0: float = 0.0
        self.slide_x1: float = 0.0
        self.slide_t: float = 0.0
        self.flood_y0: float | None = None
        self.flood_y1: float | None = None
        self._submerged: dict[int, float] = {}
        self._burning: dict[int, float] = {}
        self._want_advance: bool = False
        self._cov_order: np.ndarray | None = None
        self._errors: int = 0
        self._last_err_log: float = -1e9

    # --------------------------------------------------------------- tick --
    def tick(self, world: Any, dt: float) -> None:
        """Advance the scene and apply its consequences. Never raises."""
        step = _fnum(dt, 0.0)
        if step <= 0.0:
            return
        step = min(step, 0.25)
        self.t += step
        self.scene_t += step
        self.intensity = _approach(self.intensity, 1.0, 1.0 / 6.0, step)

        self._safe(self._tick_transients, world, step)
        handler = getattr(self, _HANDLERS.get(self.scene, "_scene_clear"), None)
        if handler is not None:
            self._safe(handler, world, step)
        self._safe(self._tick_pending, world, step)
        self._safe(self._tick_quake, world, step)
        self._safe(self._publish, world, step)

        cfg = getattr(world, "cfg", None) or getattr(world, "config", None)
        if cfg is not None:
            self._safe(self.auto_advance, cfg)

    def _safe(self, fn: Any, *args: Any) -> None:
        try:
            fn(*args)
        except Exception as exc:                            # pragma: no cover
            self._note_error(getattr(fn, "__name__", "?"), exc)

    def _note_error(self, where: str, exc: BaseException) -> None:
        self._errors += 1
        if self.t - self._last_err_log > 30.0:
            self._last_err_log = self.t
            log.warning("events: %s failed (%s: %s) [%d total]",
                        where, type(exc).__name__, exc, self._errors)

    # ------------------------------------------------------ shared per-tick --
    def _tick_transients(self, world: Any, dt: float) -> None:
        for s in self.strikes:
            s["t"] = _fnum(s.get("t"), 0.0) + dt
        if self.strikes:
            self.strikes = [s for s in self.strikes if s["t"] < STRIKE_TTL]
        self.shake_t += dt
        if self.shake_amp > 0.0:
            self.shake_amp *= math.exp(-SHAKE_DECAY * dt)
            if self.shake_amp < 0.05:
                self.shake_amp = 0.0
        self.rumble = max(0.0, self.rumble - dt * 0.8)
        self.fireflies = False

    def _publish(self, world: Any, dt: float) -> None:
        """Push the few values other subsystems read off us."""
        lig = _lighting(world)
        if lig is not None:
            try:
                lig.wind_gust = _clamp(self.gust)
            except Exception:
                pass

    def _tick_pending(self, world: Any, dt: float) -> None:
        if not self.pending:
            return
        due: list[dict[str, Any]] = []
        keep: list[dict[str, Any]] = []
        for ev in self.pending:
            if not isinstance(ev, dict):
                continue
            ev["t"] = _fnum(ev.get("t"), 0.0) - dt
            (due if ev["t"] <= 0.0 else keep).append(ev)
        self.pending = keep[:32]
        for ev in due:
            self._safe(self._fire, world, str(ev.get("kind", "")), ev)

    def schedule(self, delay: float, kind: str, **data: Any) -> None:
        """Queue a one-shot event ``delay`` seconds from now."""
        if len(self.pending) >= 32:
            return
        ev: dict[str, Any] = {"t": max(0.0, _fnum(delay, 0.0)), "kind": str(kind)}
        ev.update(data)
        self.pending.append(ev)

    def _fire(self, world: Any, kind: str, ev: dict[str, Any]) -> None:
        if kind == "ignite":
            x = _fnum(ev.get("x"), self._rng.uniform(0, RENDER_W))
            p = _nearest_flammable(world, x, 400.0)
            if p is not None and _ignite(world, p):
                _chronicle(world, "Flames take hold in the brush.")
        elif kind == "strike":
            self._lightning_strike(world)
        elif kind == "quake":
            self.trigger_quake(world, _fnum(ev.get("dur"), 2.5),
                               _fnum(ev.get("mag"), 6.0))
        elif kind == "slide":
            self.slide_phase = "slide"
            self.slide_t = 0.0
        elif kind == "advance":
            self._want_advance = True

    # -------------------------------------------------------------- weather --
    def _noise(self, slow: float = 0.31, fast: float = 1.13, off: float = 0.0) -> float:
        """Cheap smooth -1..1 noise from two incommensurate sines."""
        return (0.6 * math.sin(self.t * slow + off)
                + 0.4 * math.sin(self.t * fast + off * 1.7 + 1.3))

    def _approach_env(self, dt: float, rain: float = 0.0, snow: float = 0.0,
                      ash: float = 0.0, rate: float = 0.5) -> None:
        k = self.intensity
        self.rain = _approach(self.rain, _clamp(rain) * k, rate, dt)
        self.snow = _approach(self.snow, _clamp(snow) * k, rate, dt)
        self.ash = _approach(self.ash, _clamp(ash) * k, rate, dt)

    def _set_wind(self, dt: float, amplitude: float, gustiness: float,
                  slow: float = 0.27, fast: float = 1.07) -> None:
        target = _clamp(amplitude * self._noise(slow, fast), -1.0, 1.0) * self.intensity
        self.wind = _approach(self.wind, target, 0.9, dt)
        g = 0.5 + 0.5 * math.sin(self.t * 2.3 + 0.7)
        self.gust = _clamp(abs(self.wind) * (0.5 + 0.5 * g) * gustiness)

    # ------------------------------------------------------ accumulation fx --
    def _coverage_order(self, width: int) -> np.ndarray:
        if (self._cov_order is None or self._cov_order.size != width):
            rng = np.random.default_rng(self.seed & 0xFFFFFFFF)
            self._cov_order = rng.permutation(width).astype(np.int32)
        return self._cov_order

    def _accumulate(self, world: Any, dt: float, mat: int, depth: float,
                    max_depth: float, rate: float, remember: bool) -> float:
        """Raise the ground a little and repaint a growing share of columns."""
        h = _height(world)
        m = _material(world)
        if h is None:
            return depth
        new_depth = min(max_depth, depth + rate * dt * self.intensity)
        delta = new_depth - depth
        if delta <= 1e-4:
            return depth
        h -= np.float32(delta)                    # ground surface rises (y down)
        np.clip(h, 12.0, float(RENDER_H - 4), out=h)
        if m is not None:
            order = self._coverage_order(int(m.size))
            n = int(_clamp(new_depth / max_depth) * m.size)
            if n > 0:
                cols = order[:n]
                if remember and self._snow_prev is None:
                    self._snow_prev = [int(v) for v in m.tolist()]
                m[cols] = np.uint8(mat)
        return new_depth

    def _melt_snow(self, world: Any, dt: float) -> None:
        if self.snow_depth <= 0.0:
            return
        h = _height(world)
        m = _material(world)
        back = min(self.snow_depth, 0.09 * dt)
        if h is not None and back > 0.0:
            h += np.float32(back)
            np.clip(h, 12.0, float(RENDER_H - 4), out=h)
        self.snow_depth = max(0.0, self.snow_depth - back)
        if m is not None and self._snow_prev is not None:
            prev = np.asarray(self._snow_prev[:m.size], dtype=np.uint8)
            if prev.size == m.size:
                frac = _clamp(self.snow_depth / SNOW_MAX_DEPTH)
                order = self._coverage_order(int(m.size))
                keep = order[:int(frac * m.size)]
                mask = np.zeros(m.size, dtype=bool)
                if keep.size:
                    mask[keep] = True
                thawed = (m == np.uint8(MAT_SNOW)) & (~mask)
                m[thawed] = prev[thawed]
        if self.snow_depth <= 0.0:
            self._snow_prev = None

    # ================================================== scene: night storm ==
    def _scene_night_storm(self, world: Any, dt: float) -> None:
        swell = 0.5 + 0.5 * (0.5 + 0.5 * math.sin(self.t * 0.11))
        self._approach_env(dt, rain=_clamp(swell, 0.5, 1.0), rate=0.35)
        self._set_wind(dt, 0.85, 1.0, slow=0.23, fast=0.97)
        self.tint = (206, 218, 255)
        self.ember_rate = 0.0
        self.next_strike -= dt * (0.4 + 0.6 * self.intensity)
        if self.next_strike <= 0.0:
            self.next_strike = self._rng.uniform(STRIKE_MIN, STRIKE_MAX)
            self._lightning_strike(world)
        self._rain_douses(world, dt)

    def _lightning_strike(self, world: Any) -> None:
        direct = self._rng.random() < DIRECT_HIT_CHANCE
        x = self._rng.uniform(24.0, RENDER_W - 24.0)

        # A "direct" strike has to actually aim. Rolling a uniform x across
        # 1280px and then killing only within 25px meant a direct hit landed on
        # somebody with probability ~4%, so across a whole storm nobody was
        # ever struck and feature 29 never once fired in testing. When the roll
        # says direct, pick a victim and strike near them - slightly off, so it
        # is still luck whether it lands.
        if direct:
            crowd = [a for a in _agents(world) if getattr(a, "alive", False)]
            if crowd:
                victim = crowd[self._rng.randrange(len(crowd))]
                x = _fnum(getattr(victim, "x", x), x) + self._rng.uniform(-34.0, 34.0)
                x = max(24.0, min(RENDER_W - 24.0, x))
        gy = _ground_y(world, x)
        self.strikes.append({
            "x": float(x), "t": 0.0,
            "seed": int(self._rng.randrange(1 << 20)),
            "ground_y": float(gy), "direct": bool(direct),
        })
        if len(self.strikes) > 6:
            del self.strikes[0]
        if direct:
            _flash(world, 1.0, 0.45, LIGHTNING_COLOR)
            self.add_shake(7.5)
            self._strike_damage(world, x, gy)
        else:
            _flash(world, self._rng.uniform(0.5, 0.85),
                   self._rng.uniform(0.22, 0.38), LIGHTNING_COLOR)

    def _strike_damage(self, world: Any, x: float, gy: float) -> None:
        for ag in _agents(world):
            ax = _fnum(getattr(ag, "x", 1e9), 1e9)
            ay = _fnum(getattr(ag, "y", 1e9), 1e9)
            if math.hypot(ax - x, ay - gy) <= DIRECT_KILL_RADIUS:
                if _kill(world, ag, "was struck by lightning"):
                    _chronicle(world, f"{getattr(ag, 'name', 'A stickman')} "
                                      f"was struck by lightning.")
        _paint(world, x - SCORCH_HALF_WIDTH, x + SCORCH_HALF_WIDTH, MAT_ASH)
        p = _nearest_flammable(world, x, 70.0)
        if p is not None:
            _ignite(world, p)
        for ag in _agents(world):
            ax = _fnum(getattr(ag, "x", 1e9), 1e9)
            if abs(ax - x) < 180.0:
                away = -180.0 if ax < x else 180.0
                _panic(world, ag, ax + away, 3.0, "lightning")

    def _rain_douses(self, world: Any, dt: float) -> None:
        if self.rain < 0.55:
            return
        if self._rng.random() > dt * 0.25 * self.rain:
            return
        for p in _props(world):
            if getattr(p, "burning", False):
                _extinguish(p)
                break

    # ======================================================== scene: clear ==
    def _scene_clear(self, world: Any, dt: float) -> None:
        self._approach_env(dt, rate=0.25)
        self._set_wind(dt, 0.28, 0.35, slow=0.19, fast=0.61)
        self.tint = (255, 255, 255)
        self.ember_rate = 0.0
        self.water_level = None
        wt = _fnum(getattr(world, "world_time", self.t), self.t)
        self.fireflies = is_night(wt)
        self._melt_snow(world, dt)

    # ===================================================== scene: wildfire ==
    def _scene_wildfire(self, world: Any, dt: float) -> None:
        self._approach_env(dt, rate=0.6)
        self._set_wind(dt, 0.55, 0.7, slow=0.33, fast=1.31)
        self.tint = (255, 176, 118)
        self.ember_rate = 0.55 + 0.45 * self.intensity
        burning = [p for p in _props(world) if getattr(p, "burning", False)]
        self.next_ignite -= dt
        if not burning and self.next_ignite <= 0.0:
            self.next_ignite = self._rng.uniform(8.0, 20.0)
            x = self._rng.uniform(40.0, RENDER_W - 40.0)
            p = _nearest_flammable(world, x, RENDER_W)
            if p is not None and _ignite(world, p):
                _chronicle(world, "A wildfire catches in the dry brush.")
        if burning:
            self._fire_harms_agents(world, dt, burning)
            if self._rng.random() < dt * 0.35 * (0.3 + 0.7 * abs(self.wind)):
                lead = burning[self._rng.randrange(len(burning))]
                fx = _fnum(getattr(lead, "x", 0.0))
                nxt = _nearest_flammable(world, fx + 40.0 * (1 if self.wind >= 0 else -1), 120.0)
                if nxt is not None:
                    _ignite(world, nxt)

    def _fire_harms_agents(self, world: Any, dt: float, burning: list[Any]) -> None:
        fires = [(_fnum(getattr(p, "x", 1e9), 1e9), _fnum(getattr(p, "y", 1e9), 1e9))
                 for p in burning]
        for ag in _agents(world):
            aid = _aid(ag)
            ax = _fnum(getattr(ag, "x", 1e9), 1e9)
            ay = _fnum(getattr(ag, "y", 1e9), 1e9)
            near = min((math.hypot(ax - fx, ay - fy) for fx, fy in fires),
                       default=1e9)
            if near < FLEE_DIST:
                fx = min(fires, key=lambda f: math.hypot(ax - f[0], ay - f[1]))[0]
                away = ax + (140.0 if ax >= fx else -140.0)
                _panic(world, ag, away, 2.5, "fire")
            if aid is None:
                continue
            if near < BURN_TOUCH_DIST:
                burnt = self._burning.get(aid, 0.0) + dt
                self._burning[aid] = burnt
                try:
                    ag.warmth = _clamp(_fnum(getattr(ag, "warmth", 0.5), 0.5) - dt * 0.4)
                except Exception:
                    pass
                if burnt >= BURN_DEATH_SEC:
                    self._burning.pop(aid, None)
                    if _kill(world, ag, "burned in the wildfire"):
                        _chronicle(world, f"{getattr(ag, 'name', 'A stickman')} "
                                          f"burned in the wildfire.")
            elif aid in self._burning:
                self._burning[aid] = max(0.0, self._burning[aid] - dt * 2.0)
                if self._burning[aid] <= 0.0:
                    self._burning.pop(aid, None)

    # ===================================================== scene: mudslide ==
    def _scene_mudslide(self, world: Any, dt: float) -> None:
        self._approach_env(dt, rain=0.55, rate=0.4)
        self._set_wind(dt, 0.4, 0.5)
        self.tint = (226, 214, 198)
        self.ember_rate = 0.0
        if self.slide_phase == "idle":
            self._pick_slide_span(world)
            self.slide_phase = "warn"
            self.slide_t = 0.0
            _chronicle(world, "The hillside groans; loose earth begins to shift.")
        elif self.slide_phase == "warn":
            self.slide_t += dt
            self.rumble = _clamp(self.slide_t / MUDSLIDE_WARN)
            self.add_shake(1.4 * self.rumble)
            if self.slide_t >= MUDSLIDE_WARN:
                self.slide_phase = "slide"
                self.slide_t = 0.0
                self.trigger_quake(world, 1.6, 5.0)
        elif self.slide_phase == "slide":
            self.slide_t += dt
            self.rumble = 1.0
            self.add_shake(5.0)
            self._slide_step(world, dt)
            if self.slide_t >= MUDSLIDE_SLIDE:
                self.slide_phase = "settle"
                self.slide_t = 0.0
                _paint(world, self.slide_x0, self.slide_x1, MAT_MUD)
                _chronicle(world, "A mudslide reshapes the slope.")
        elif self.slide_phase == "settle":
            self.slide_t += dt
            self.rumble = max(0.0, 1.0 - self.slide_t / MUDSLIDE_SETTLE)
            if self.slide_t >= MUDSLIDE_SETTLE:
                self.slide_phase = "done"
                self._want_advance = True

    def _pick_slide_span(self, world: Any) -> None:
        width = self._rng.uniform(140.0, 260.0)
        h = _height(world)
        x0 = self._rng.uniform(0.0, max(1.0, RENDER_W - width))
        if h is not None and h.size > 16:
            grad = np.gradient(h.astype(np.float32))
            win = max(8, int(width) // 8)
            kern = np.ones(win, dtype=np.float32) / float(win)
            smooth = np.convolve(np.abs(grad), kern, mode="same")
            lo = int(width * 0.5)
            hi = max(lo + 1, int(h.size - width * 0.5))
            centre = int(np.argmax(smooth[lo:hi])) + lo
            x0 = _clamp(centre - width * 0.5, 0.0, max(0.0, RENDER_W - width))
        self.slide_x0 = float(x0)
        self.slide_x1 = float(min(RENDER_W, x0 + width))

    def _slide_step(self, world: Any, dt: float) -> None:
        x0, x1 = self.slide_x0, self.slide_x1
        if x1 - x0 < 8.0:
            return
        mid = (x0 + x1) * 0.5
        amount = 26.0 * dt / max(0.1, MUDSLIDE_SLIDE)
        _deform(world, x0, mid, amount)          # scour the upper slope
        _deform(world, mid, x1, -amount * 0.85)  # pile it at the toe
        _paint(world, x0, x1, MAT_MUD)
        for ag in _agents(world):
            ax = _fnum(getattr(ag, "x", 1e9), 1e9)
            if x0 - 10.0 <= ax <= x1 + 10.0:
                ay = _fnum(getattr(ag, "y", 0.0))
                if ay >= _ground_y(world, ax) - 26.0:
                    if _kill(world, ag, "was buried by the mudslide"):
                        _chronicle(world, f"{getattr(ag, 'name', 'A stickman')} "
                                          f"was buried by the mudslide.")
                continue
            edge = x0 if ax < x0 else x1
            if abs(ax - edge) < 130.0:
                _panic(world, ag, ax + (-140.0 if ax < x0 else 140.0), 3.0, "mudslide")

    # ===================================================== scene: blizzard ==
    def _scene_blizzard(self, world: Any, dt: float) -> None:
        self._approach_env(dt, snow=1.0, rate=0.35)
        self._set_wind(dt, 0.95, 0.9, slow=0.29, fast=1.51)
        self.tint = (214, 226, 244)
        self.ember_rate = 0.0
        self.snow_depth = self._accumulate(world, dt, MAT_SNOW, self.snow_depth,
                                           SNOW_MAX_DEPTH, SNOW_RATE, True)
        chill = self.exposure_rate() * dt
        if chill > 0.0:
            for ag in _agents(world):
                try:
                    ag.warmth = _clamp(_fnum(getattr(ag, "warmth", 0.0), 0.0) + chill)
                except Exception:
                    break

    def exposure_rate(self) -> float:
        """Extra ``warmth`` (0..1, 1 = freezing) per second from the weather.

        Already applied by ``tick``; exposed for UI/telemetry so entities.py
        does not need to apply it a second time.
        """
        if self.scene == SCENE_BLIZZARD:
            return 0.012 * self.intensity
        if self.scene == SCENE_NIGHT_STORM:
            return 0.004 * self.rain
        return 0.0

    # ======================================================== scene: flood ==
    def _scene_flood(self, world: Any, dt: float) -> None:
        self._approach_env(dt, rain=0.35, rate=0.3)
        self._set_wind(dt, 0.35, 0.4)
        self.tint = (206, 224, 246)
        self.ember_rate = 0.0
        if self.flood_y0 is None or self.flood_y1 is None:
            h = _height(world)
            if h is not None:
                low = float(np.max(h))
                high = float(np.min(h))
            else:
                low, high = RENDER_H * 0.8, RENDER_H * 0.4
            depth = _clamp(0.45 * (low - high), 40.0, 180.0)
            self.flood_y0 = low + 4.0
            self.flood_y1 = low - depth
            _chronicle(world, "Water begins to pool in the low ground.")
        t = self.scene_t
        if t < FLOOD_RISE:
            u = _smooth(t / FLOOD_RISE)
        elif t < FLOOD_RISE + FLOOD_HOLD:
            u = 1.0
        elif t < FLOOD_RISE + FLOOD_HOLD + FLOOD_FALL:
            u = 1.0 - _smooth((t - FLOOD_RISE - FLOOD_HOLD) / FLOOD_FALL)
        else:
            u = 0.0
            if not self._want_advance:
                self._want_advance = True
                _chronicle(world, "The floodwater drains away.")
        self.water_level = self.flood_y0 + (self.flood_y1 - self.flood_y0) * u
        if u <= 0.0:
            self.water_level = None
            return
        self._flood_effects(world, dt, float(self.water_level))

    def _flood_effects(self, world: Any, dt: float, level: float) -> None:
        for p in _props(world):
            if getattr(p, "burning", False) and _fnum(getattr(p, "y", -1e9), -1e9) > level:
                _extinguish(p)
        h = _height(world)
        dry: np.ndarray | None = None
        if h is not None:
            found = np.flatnonzero(h < np.float32(level - 12.0))
            if found.size:
                dry = found
        for ag in _agents(world):
            ax = _fnum(getattr(ag, "x", 0.0))
            ay = _fnum(getattr(ag, "y", 0.0))
            aid = _aid(ag)
            if ay <= level:
                if aid is not None and aid in self._submerged:
                    self._submerged[aid] = max(0.0, self._submerged[aid] - dt)
                continue
            target = ax
            if dry is not None:
                j = int(np.argmin(np.abs(dry - int(ax))))
                target = float(dry[j])
            else:
                target = 20.0 if ax > RENDER_W * 0.5 else RENDER_W - 20.0
            _panic(world, ag, target, 4.0, "flood")
            if aid is None:
                continue
            sub = self._submerged.get(aid, 0.0) + dt
            self._submerged[aid] = sub
            if sub >= DROWN_SEC:
                self._submerged.pop(aid, None)
                if _kill(world, ag, "drowned in the flood"):
                    _chronicle(world, f"{getattr(ag, 'name', 'A stickman')} "
                                      f"drowned in the flood.")

    # ======================================================= scene: meteor ==
    def _scene_meteor(self, world: Any, dt: float) -> None:
        self._approach_env(dt, rate=0.4)
        self._set_wind(dt, 0.3, 0.35)
        self.tint = (226, 226, 255)
        self.ember_rate = 0.25
        wt = _fnum(getattr(world, "world_time", self.t), self.t)
        self.fireflies = is_night(wt) and daylight_factor(wt) < 0.1
        self.next_meteor -= dt * (0.5 + 0.5 * self.intensity)
        if self.next_meteor <= 0.0:
            self.next_meteor = self._rng.uniform(METEOR_MIN, METEOR_MAX)
            self._spawn_meteor(world)
        self._advance_meteors(world, dt)

    def _spawn_meteor(self, world: Any) -> None:
        impact = self._rng.random() < METEOR_IMPACT_CHANCE
        x1 = self._rng.uniform(40.0, RENDER_W - 40.0)
        drift = self._rng.uniform(180.0, 420.0) * (1 if self._rng.random() < 0.5 else -1)
        x0 = x1 - drift
        y0 = -self._rng.uniform(40.0, 220.0)
        y1 = _ground_y(world, x1) if impact else self._rng.uniform(0.25, 0.5) * RENDER_H
        self.meteors.append({
            "x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1),
            "t": 0.0, "ttl": float(self._rng.uniform(0.7, 1.3)),
            "seed": int(self._rng.randrange(1 << 20)),
            "impact": bool(impact), "done": False,
        })
        if len(self.meteors) > 12:
            del self.meteors[0]

    def _advance_meteors(self, world: Any, dt: float) -> None:
        if not self.meteors:
            return
        keep: list[dict[str, Any]] = []
        for m in self.meteors:
            m["t"] = _fnum(m.get("t"), 0.0) + dt
            ttl = _fnum(m.get("ttl"), 1.0)
            if m["t"] >= ttl and not m.get("done"):
                m["done"] = True
                if m.get("impact"):
                    self._meteor_impact(world, _fnum(m.get("x1")), _fnum(m.get("y1")))
            if m["t"] < ttl + METEOR_TTL:
                keep.append(m)
        self.meteors = keep

    def _meteor_impact(self, world: Any, x: float, y: float) -> None:
        r = self._rng.uniform(18.0, 40.0)
        _crater(world, x, y, r)
        _paint(world, x - r * 0.9, x + r * 0.9, MAT_ASH)
        _flash(world, 0.85, 0.35, METEOR_COLOR)
        self.add_shake(9.0)
        self.trigger_quake(world, 1.2, 4.0)
        p = _nearest_flammable(world, x, r * 2.5)
        if p is not None:
            _ignite(world, p)
        for ag in _agents(world):
            ax = _fnum(getattr(ag, "x", 1e9), 1e9)
            ay = _fnum(getattr(ag, "y", 1e9), 1e9)
            d = math.hypot(ax - x, ay - y)
            if d <= r:
                if _kill(world, ag, "was crushed by a meteor"):
                    _chronicle(world, f"{getattr(ag, 'name', 'A stickman')} "
                                      f"was crushed by a falling star.")
            elif d < r * 5.0:
                _panic(world, ag, ax + (150.0 if ax >= x else -150.0), 3.0, "meteor")
        _chronicle(world, "A meteor slams into the hillside.")

    # ====================================================== scene: ashfall ==
    def _scene_ashfall(self, world: Any, dt: float) -> None:
        self._approach_env(dt, ash=1.0, rate=0.3)
        self._set_wind(dt, 0.45, 0.5, slow=0.17, fast=0.73)
        self.tint = (214, 128, 104)
        self.ember_rate = 0.18
        self.ash_depth = self._accumulate(world, dt, MAT_ASH, self.ash_depth,
                                          ASH_MAX_DEPTH, ASH_RATE, False)

    # ================================================= quakes / shake / fx ==
    def trigger_quake(self, world: Any, duration: float = 2.5,
                      magnitude: float = 6.0) -> None:
        """Start (or extend) an earthquake: shake plus a chance of a fissure."""
        self.quake_t = max(self.quake_t, max(0.2, _fnum(duration, 2.5)))
        self.add_shake(max(1.0, _fnum(magnitude, 6.0)))

    def _tick_quake(self, world: Any, dt: float) -> None:
        if self.quake_t <= 0.0:
            if self.scene in (SCENE_METEOR, SCENE_MUDSLIDE):
                if self._rng.random() < dt / 150.0:
                    self.trigger_quake(world, self._rng.uniform(2.0, 4.5), 6.0)
            return
        self.quake_t = max(0.0, self.quake_t - dt)
        self.add_shake(6.0 if self.quake_t > 0.6 else 3.0)
        if self._rng.random() < dt * 0.35:
            x = self._rng.uniform(30.0, RENDER_W - 30.0)
            w = self._rng.uniform(6.0, 16.0)
            _deform(world, x - w, x + w, self._rng.uniform(4.0, 11.0))
            _paint(world, x - w, x + w, MAT_DIRT)
        if self.quake_t <= 0.0:
            _chronicle(world, "The ground stops shaking.")

    def add_shake(self, amplitude: float) -> None:
        """Request screen shake. Takes the max of current and requested."""
        a = _fnum(amplitude, 0.0)
        if a > self.shake_amp:
            self.shake_amp = min(18.0, a)

    def shake_offset(self) -> tuple[float, float]:
        """Renderer-side camera offset in px. Pure: safe to call many times."""
        if self.shake_amp <= 0.05:
            return (0.0, 0.0)
        a = self.shake_amp
        dx = math.sin(self.shake_t * 47.0) * a
        dy = math.cos(self.shake_t * 39.3) * a * 0.6
        return (dx, dy)

    # ============================================================ control ==
    def request_scene(self, name: str) -> bool:
        """Switch scene (tray menu / commands). Returns False for a bad name."""
        if name not in SCENES:
            return False
        self.scene = name
        self._reset_scene_state()
        return True

    def _reset_scene_state(self) -> None:
        self.scene_t = 0.0
        self.intensity = 0.0
        self._want_advance = False
        self.pending.clear()
        self.strikes.clear()
        self.meteors.clear()
        self.water_level = None
        self.flood_y0 = None
        self.flood_y1 = None
        self.slide_phase = "idle"
        self.slide_t = 0.0
        self.rumble = 0.0
        self.ember_rate = 0.0
        self._submerged.clear()
        self._burning.clear()
        self.next_strike = self._rng.uniform(2.0, 6.0)
        self.next_meteor = self._rng.uniform(1.0, 3.0)
        self.next_ignite = 0.5

    def auto_advance(self, cfg: Any) -> bool:
        """Pick a plausible next scene once the current one has had its run."""
        if cfg is None or not getattr(cfg, "auto_scene_change", False):
            return False
        min_sec = _fnum(getattr(cfg, "scene_min_sec", 180.0), 180.0)
        if not self._want_advance and self.scene_t < max(20.0, min_sec):
            return False
        nxt = self._pick_next()
        if nxt is None:
            self._want_advance = False
            return False
        prev = self.scene
        ok = self.request_scene(nxt)
        if ok:
            log.info("scene %s -> %s", prev, nxt)
        return ok

    def _pick_next(self) -> str | None:
        table = _TRANSITIONS.get(self.scene) or {}
        options = [(s, w) for s, w in table.items() if s in SCENES and s != self.scene]
        if not options:
            options = [(s, 1.0) for s in SCENES if s != self.scene]
        total = sum(w for _, w in options)
        if total <= 0.0:
            return None
        r = self._rng.uniform(0.0, total)
        for name, w in options:
            r -= w
            if r <= 0.0:
                return name
        return options[-1][0]

    # ================================================================= io ==
    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "scene_t": float(self.scene_t),
            "intensity": float(self.intensity),
            "t": float(self.t),
            "seed": int(self.seed),
            "wind": float(self.wind),
            "rain": float(self.rain),
            "snow": float(self.snow),
            "ash": float(self.ash),
            "gust": float(self.gust),
            "water_level": None if self.water_level is None else float(self.water_level),
            "quake_t": float(self.quake_t),
            "pending": [dict(p) for p in self.pending if isinstance(p, dict)],
            "strikes": [dict(s) for s in self.strikes],
            "meteors": [dict(m) for m in self.meteors],
            "ember_rate": float(self.ember_rate),
            "tint": list(self.tint),
            "rumble": float(self.rumble),
            "shake_amp": float(self.shake_amp),
            "shake_t": float(self.shake_t),
            "snow_depth": float(self.snow_depth),
            "ash_depth": float(self.ash_depth),
            "snow_prev": self._snow_prev,
            "slide_phase": self.slide_phase,
            "slide_x0": float(self.slide_x0),
            "slide_x1": float(self.slide_x1),
            "slide_t": float(self.slide_t),
            "flood_y0": None if self.flood_y0 is None else float(self.flood_y0),
            "flood_y1": None if self.flood_y1 is None else float(self.flood_y1),
            "next_strike": float(self.next_strike),
            "next_meteor": float(self.next_meteor),
            "next_ignite": float(self.next_ignite),
            "submerged": {str(k): float(v) for k, v in self._submerged.items()},
            "burning": {str(k): float(v) for k, v in self._burning.items()},
            "want_advance": bool(self._want_advance),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "EventSystem":
        ev = cls()
        if not isinstance(d, dict):
            return ev
        scene = d.get("scene")
        ev.scene = scene if isinstance(scene, str) and scene in SCENES else SCENE_NIGHT_STORM
        ev.scene_t = max(0.0, _fnum(d.get("scene_t"), 0.0))
        ev.intensity = _clamp(_fnum(d.get("intensity"), 1.0))
        ev.t = max(0.0, _fnum(d.get("t"), 0.0))
        seed = d.get("seed")
        if isinstance(seed, (int, float)) and not isinstance(seed, bool):
            ev.seed = int(seed)
            ev._rng = random.Random(ev.seed)
        ev.wind = _clamp(_fnum(d.get("wind"), 0.0), -1.0, 1.0)
        ev.rain = _clamp(_fnum(d.get("rain"), 0.0))
        ev.snow = _clamp(_fnum(d.get("snow"), 0.0))
        ev.ash = _clamp(_fnum(d.get("ash"), 0.0))
        ev.gust = _clamp(_fnum(d.get("gust"), 0.0))
        wl = d.get("water_level")
        ev.water_level = None if wl is None else _fnum(wl, 0.0)
        ev.quake_t = max(0.0, _fnum(d.get("quake_t"), 0.0))
        ev.pending = [dict(p) for p in _listof(d.get("pending")) if isinstance(p, dict)][:32]
        ev.strikes = [dict(s) for s in _listof(d.get("strikes")) if isinstance(s, dict)][:6]
        ev.meteors = [dict(m) for m in _listof(d.get("meteors")) if isinstance(m, dict)][:12]
        ev.ember_rate = _clamp(_fnum(d.get("ember_rate"), 0.0))
        tint = d.get("tint")
        try:
            r, g, b = (int(c) for c in tuple(tint)[:3])
            ev.tint = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
        except (TypeError, ValueError):
            ev.tint = (255, 255, 255)
        ev.rumble = _clamp(_fnum(d.get("rumble"), 0.0))
        ev.shake_amp = max(0.0, _fnum(d.get("shake_amp"), 0.0))
        ev.shake_t = max(0.0, _fnum(d.get("shake_t"), 0.0))
        ev.snow_depth = _clamp(_fnum(d.get("snow_depth"), 0.0), 0.0, SNOW_MAX_DEPTH)
        ev.ash_depth = _clamp(_fnum(d.get("ash_depth"), 0.0), 0.0, ASH_MAX_DEPTH)
        prev = d.get("snow_prev")
        if isinstance(prev, list) and prev:
            try:
                ev._snow_prev = [int(v) & 0xFF for v in prev]
            except (TypeError, ValueError):
                ev._snow_prev = None
        phase = d.get("slide_phase")
        ev.slide_phase = phase if phase in ("idle", "warn", "slide", "settle", "done") else "idle"
        ev.slide_x0 = _fnum(d.get("slide_x0"), 0.0)
        ev.slide_x1 = _fnum(d.get("slide_x1"), 0.0)
        ev.slide_t = max(0.0, _fnum(d.get("slide_t"), 0.0))
        f0, f1 = d.get("flood_y0"), d.get("flood_y1")
        ev.flood_y0 = None if f0 is None else _fnum(f0, 0.0)
        ev.flood_y1 = None if f1 is None else _fnum(f1, 0.0)
        ev.next_strike = max(0.0, _fnum(d.get("next_strike"), 5.0))
        ev.next_meteor = max(0.0, _fnum(d.get("next_meteor"), 2.0))
        ev.next_ignite = max(0.0, _fnum(d.get("next_ignite"), 0.5))
        ev._submerged = _intmap(d.get("submerged"))
        ev._burning = _intmap(d.get("burning"))
        ev._want_advance = bool(d.get("want_advance", False))
        return ev


def _smooth(u: float) -> float:
    u = _clamp(u)
    return u * u * (3.0 - 2.0 * u)


def _listof(v: Any) -> list[Any]:
    return list(v) if isinstance(v, list) else []


def _intmap(v: Any) -> dict[int, float]:
    out: dict[int, float] = {}
    if not isinstance(v, dict):
        return out
    for k, val in v.items():
        try:
            out[int(k)] = float(val)
        except (TypeError, ValueError):
            continue
    return out


# ------------------------------------------------------------ smoke test --
if __name__ == "__main__":                                  # pragma: no cover
    from .lighting import Lighting

    class _T:
        def __init__(self) -> None:
            self.height = np.full(RENDER_W, RENDER_H * 0.7, dtype=np.float32)
            self.height += (np.sin(np.arange(RENDER_W) / 90.0) * 60.0).astype(np.float32)
            self.material = np.zeros(RENDER_W, dtype=np.uint8)

        def ground_y(self, x: float) -> float:
            return float(self.height[int(max(0, min(RENDER_W - 1, x)))])

    class _A:
        def __init__(self, i: int, x: float, y: float) -> None:
            self.id, self.x, self.y = i, x, y
            self.alive, self.warmth, self.morale = True, 0.0, 0.6
            self.name = f"agent{i}"

    class _P:
        def __init__(self, i: int, x: float, y: float) -> None:
            self.id, self.x, self.y, self.kind = i, x, y, "tree"
            self.burning = False

    class _W:
        def __init__(self) -> None:
            self.terrain = _T()
            self.lighting = Lighting()
            self.world_time = 0.0
            self.agents = [_A(i, 100.0 + i * 150.0, 0.0) for i in range(6)]
            for a in self.agents:
                a.y = self.terrain.ground_y(a.x)
            self.props = [_P(100 + i, 80.0 + i * 200.0, 0.0) for i in range(6)]
            self.chronicle: list[str] = []

    w = _W()
    ev = EventSystem()
    for scene in SCENES:
        ev.request_scene(scene)
        for _ in range(int(30 * 25)):        # 25 sim-seconds per scene
            ev.tick(w, 1 / 30)
            w.lighting.tick(1 / 30)
            w.world_time += 1 / 30
        alive = sum(1 for a in w.agents if a.alive)
        print(f"{scene:12s} rain={ev.rain:.2f} snow={ev.snow:.2f} ash={ev.ash:.2f} "
              f"wind={ev.wind:+.2f} water={ev.water_level} alive={alive} "
              f"shake={ev.shake_offset()[0]:+.2f}")
    blob = ev.to_dict()
    back = EventSystem.from_dict(blob)
    print("roundtrip scene:", back.scene, "keys:", len(blob))
    print("degenerate from_dict:", EventSystem.from_dict({"scene": "nope"}).scene)
    ev.tick(None, 1 / 30)                    # must not raise with no world
    print("null-world tick ok; errors:", ev._errors)
