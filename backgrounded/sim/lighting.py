"""Lighting: the day/night ambient curve, dynamic light sources, flashes.

Pure data + numpy. **No pygame** — this module must stay importable headless.

The renderer consumes it read-only::

    amb  = lighting.ambient(scene, world.world_time)      # 0..1 global level
    rgb  = lighting.ambient_rgb(scene, world.world_time)  # tinted lightmap fill
    for s in lighting.sources:                            # radial gradient blits
        ...
    lit  = lighting.light_at(ag.x, ag.y)                  # silhouette test

``sources`` is rebuilt every sim tick by ``world.py`` from the things that
actually emit light (candle bearers, firepits, burning props).  Drive it with
``begin_frame()`` + ``add_source(...)``, or call the duck-typed one-liner
``rebuild_from_world(world)`` which does the usual job.

Flashes are the global transient term: lightning and meteors call
``add_flash(intensity, decay_sec, color)``.  A flash raises *ambient* for
everyone — that reveal-then-dark rhythm is the point of the whole scene.
"""
from __future__ import annotations

import logging
import math
import zlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from ..constants import (
    DAY_LENGTH_SEC,
    SCENE_ASHFALL,
    SCENE_BLIZZARD,
    SCENE_CLEAR,
    SCENE_FLOOD,
    SCENE_METEOR,
    SCENE_MUDSLIDE,
    SCENE_NIGHT_STORM,
    SCENE_WILDFIRE,
)

log = logging.getLogger(__name__)

__all__ = [
    "LightSource", "Flash", "Lighting",
    "sun_phase", "daylight_factor", "is_night",
    "CANDLE_RADIUS", "CANDLE_COLOR", "FIRE_RADIUS", "FIRE_COLOR",
    "BURNING_RADIUS", "BURNING_COLOR", "LIGHTNING_COLOR", "METEOR_COLOR",
    "SILHOUETTE_THRESHOLD",
]

# ------------------------------------------------------------------ tuning --
CANDLE_RADIUS = 118.0
CANDLE_COLOR: tuple[int, int, int] = (255, 186, 96)
FIRE_RADIUS = 224.0
FIRE_COLOR: tuple[int, int, int] = (255, 152, 62)
BURNING_RADIUS = 152.0
BURNING_COLOR: tuple[int, int, int] = (255, 120, 44)
LIGHTNING_COLOR: tuple[int, int, int] = (208, 224, 255)
METEOR_COLOR: tuple[int, int, int] = (255, 226, 186)

#: Below this total illumination the renderer should draw an agent as a
#: near-black silhouette (feature 36).
SILHOUETTE_THRESHOLD = 0.18

#: (night level, day level) of the ambient curve, per scene.
#
# The night floors are set from measured output, not taste. At 0.06 the storm
# scene rendered at mean luminance 2.3/255 with 98% of pixels below 10 - on a
# desktop that reads as a black screen with an orange dot, and the lightning
# reveal has nothing to reveal *from*. At ~0.13 the landscape silhouette is
# faintly legible, so a strike adds detail to a world you can already sense
# rather than briefly proving one exists. Contrast against a 1.0 flash is
# still nearly 8x, which is what sells the effect.
_SCENE_AMBIENT: dict[str, tuple[float, float]] = {
    SCENE_CLEAR:       (0.34, 0.95),
    SCENE_NIGHT_STORM: (0.42, 0.60),
    SCENE_WILDFIRE:    (0.34, 0.74),
    SCENE_MUDSLIDE:    (0.30, 0.70),
    SCENE_BLIZZARD:    (0.36, 0.66),
    SCENE_FLOOD:       (0.30, 0.68),
    SCENE_METEOR:      (0.30, 0.82),
    SCENE_ASHFALL:     (0.26, 0.48),
}
_DEFAULT_AMBIENT: tuple[float, float] = (0.22, 0.88)

# Colour multipliers (0..1 per channel) applied to the ambient grey.
#
# These stack multiplicatively with the scene ambient AND with _SCENE_RGB, so
# they compound fast: at ambient 0.13 the old night red channel came out at
# 0.13 * 0.44 * 0.74 = 0.042, i.e. 4% - which is why raising ambient alone
# never made the storm visible. Keep the blue cast (night should read cool)
# but do not crush red and green to nothing.
_NIGHT_RGB: tuple[float, float, float] = (0.66, 0.76, 1.00)
_DAY_RGB: tuple[float, float, float] = (1.0, 0.98, 0.94)
_SCENE_RGB: dict[str, tuple[float, float, float]] = {
    SCENE_CLEAR:       (1.00, 1.00, 1.00),
    SCENE_NIGHT_STORM: (0.86, 0.92, 1.00),
    SCENE_WILDFIRE:    (1.00, 0.72, 0.48),
    SCENE_MUDSLIDE:    (0.92, 0.86, 0.78),
    SCENE_BLIZZARD:    (0.90, 0.95, 1.00),
    SCENE_FLOOD:       (0.80, 0.90, 1.00),
    SCENE_METEOR:      (0.86, 0.86, 1.00),
    SCENE_ASHFALL:     (0.98, 0.62, 0.50),
}

_MAX_FLASHES = 8
_MAX_SOURCES = 96


# ----------------------------------------------------------------- helpers --
def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 0.0 if x < edge0 else 1.0
    u = _clamp((x - edge0) / (edge1 - edge0))
    return u * u * (3.0 - 2.0 * u)


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):   # NaN / inf
        return default
    return f


def _as_color(v: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    try:
        r, g, b = (int(c) for c in tuple(v)[:3])
    except (TypeError, ValueError):
        return default
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def _as_owner(v: Any) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _stable_phase(kind: str, owner_id: int | None) -> float:
    """Deterministic 0..2pi flicker offset, stable across rebuilds and runs."""
    key = f"{kind}:{owner_id}".encode("utf-8", "replace")
    return (zlib.crc32(key) % 100003) / 100003.0 * math.tau


# ------------------------------------------------------- day / night curve --
def sun_phase(world_time: float) -> float:
    """Position in the day, 0..1. 0=midnight, .25=sunrise, .5=noon, .75=sunset."""
    day = DAY_LENGTH_SEC if DAY_LENGTH_SEC > 0 else 1.0
    return (_as_float(world_time) % day) / day


def daylight_factor(world_time: float) -> float:
    """0 at deep night, 1 at midday, with a smooth dawn/dusk ramp."""
    elev = -math.cos(math.tau * sun_phase(world_time))   # -1 midnight .. +1 noon
    return _smoothstep(-0.20, 0.28, elev)


def is_night(world_time: float) -> bool:
    """True while the sky is dark enough for stars, fireflies and candles."""
    return daylight_factor(world_time) < 0.25


# ------------------------------------------------------------ light source --
@dataclass
class LightSource:
    """One emitter. Cheap enough to rebuild a hundred of these every tick."""

    x: float = 0.0
    y: float = 0.0
    radius: float = 100.0
    color: tuple[int, int, int] = (255, 190, 110)
    intensity: float = 1.0           # 0..1
    flicker: float = 0.0             # 0..1 amplitude of per-frame noise
    kind: str = "light"              # candle | fire | burning | glow | fx
    owner_id: int | None = None      # agent / prop / structure id, if any
    phase: float = 0.0               # deterministic flicker offset (radians)
    fmul: float = 1.0                # derived flicker multiplier (not persisted)

    # ------------------------------------------------------------------ io --
    def to_dict(self) -> dict[str, Any]:
        return {
            "x": float(self.x), "y": float(self.y),
            "radius": float(self.radius),
            "color": list(self.color),
            "intensity": float(self.intensity),
            "flicker": float(self.flicker),
            "kind": str(self.kind),
            "owner_id": self.owner_id,
            "phase": float(self.phase),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "LightSource":
        s = cls()
        if not isinstance(d, dict):
            return s
        s.x = _as_float(d.get("x"), 0.0)
        s.y = _as_float(d.get("y"), 0.0)
        s.radius = max(0.0, _as_float(d.get("radius"), 100.0))
        s.color = _as_color(d.get("color"), (255, 190, 110))
        s.intensity = _clamp(_as_float(d.get("intensity"), 1.0), 0.0, 4.0)
        s.flicker = _clamp(_as_float(d.get("flicker"), 0.0))
        s.kind = str(d.get("kind", "light"))
        s.owner_id = _as_owner(d.get("owner_id"))
        s.phase = _as_float(d.get("phase"), _stable_phase(s.kind, s.owner_id))
        s.fmul = 1.0
        return s


# ------------------------------------------------------------------ flash --
@dataclass
class Flash:
    """A global additive light burst with a fast attack and a soft decay."""

    intensity: float = 1.0
    decay: float = 0.4               # seconds from peak to nothing
    color: tuple[int, int, int] = LIGHTNING_COLOR
    attack: float = 0.035            # seconds to reach peak
    t: float = 0.0                   # age

    def value(self) -> float:
        """Current 0..intensity contribution."""
        if self.t < 0.0:
            return 0.0
        if self.t < self.attack:
            a = self.t / self.attack if self.attack > 0.0 else 1.0
            return self.intensity * _clamp(a)
        if self.decay <= 0.0:
            return 0.0
        u = (self.t - self.attack) / self.decay
        if u >= 1.0:
            return 0.0
        rem = 1.0 - u
        return self.intensity * rem * rem      # quadratic falloff, snappy

    @property
    def expired(self) -> bool:
        return self.t >= self.attack + self.decay

    def to_dict(self) -> dict[str, Any]:
        return {
            "intensity": float(self.intensity),
            "decay": float(self.decay),
            "color": list(self.color),
            "attack": float(self.attack),
            "t": float(self.t),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Flash":
        f = cls()
        if not isinstance(d, dict):
            return f
        f.intensity = _clamp(_as_float(d.get("intensity"), 1.0), 0.0, 4.0)
        f.decay = max(0.0, _as_float(d.get("decay"), 0.4))
        f.color = _as_color(d.get("color"), LIGHTNING_COLOR)
        f.attack = max(0.0, _as_float(d.get("attack"), 0.035))
        f.t = max(0.0, _as_float(d.get("t"), 0.0))
        return f


# --------------------------------------------------------------- lighting --
class Lighting:
    """Owns the light source registry, the ambient curve and flash state.

    Nothing here mutates the world; ``EventSystem`` and ``World`` push into it.
    """

    def __init__(self, sources: Iterable[LightSource] | None = None) -> None:
        self.sources: list[LightSource] = list(sources) if sources else []
        self.flashes: list[Flash] = []
        self.t: float = 0.0              # local clock, drives flicker
        self.wind_gust: float = 0.0      # 0..1, set by EventSystem; guttering
        self.last_ambient: float = 0.5   # cached result of ambient()
        self._flash_total: float = 0.0
        self._flash_color: tuple[int, int, int] = (0, 0, 0)

    # ------------------------------------------------------ source registry --
    def begin_frame(self) -> None:
        """Drop last tick's sources. Call before re-adding them."""
        self.sources.clear()

    def add_source(
        self,
        x: float,
        y: float,
        radius: float,
        color: tuple[int, int, int],
        intensity: float = 1.0,
        flicker: float = 0.0,
        kind: str = "light",
        owner_id: int | None = None,
    ) -> LightSource | None:
        """Register one emitter for this tick. Returns it (or None if full)."""
        try:
            if len(self.sources) >= _MAX_SOURCES:
                return None
            src = LightSource(
                x=_as_float(x), y=_as_float(y),
                radius=max(0.0, _as_float(radius, 100.0)),
                color=_as_color(color, (255, 190, 110)),
                intensity=_clamp(_as_float(intensity, 1.0), 0.0, 4.0),
                flicker=_clamp(_as_float(flicker, 0.0)),
                kind=str(kind),
                owner_id=_as_owner(owner_id),
            )
            src.phase = _stable_phase(src.kind, src.owner_id)
            src.fmul = self._flicker_mul(src)
            self.sources.append(src)
            return src
        except Exception:                                   # pragma: no cover
            return None

    def set_sources(self, sources: Iterable[LightSource]) -> None:
        """Replace the whole registry in one go."""
        try:
            self.sources = [s for s in sources if isinstance(s, LightSource)][:_MAX_SOURCES]
            self._refresh_flicker()
        except Exception:                                   # pragma: no cover
            self.sources = []

    def rebuild_from_world(self, world: Any) -> int:
        """Duck-typed convenience rebuild: candle bearers + fires + burning props.

        Reads only. ``world.py`` may call this once per tick instead of hand
        rolling the loop; anything it misses can be appended with
        ``add_source``. Returns the number of sources registered.
        """
        self.begin_frame()
        try:
            for ag in _iter_attr(world, ("agents", "stickmen", "people")):
                if not getattr(ag, "alive", True):
                    continue
                if not getattr(ag, "holds_candle", False):
                    continue
                self.add_source(
                    _as_float(getattr(ag, "x", 0.0)),
                    _as_float(getattr(ag, "y", 0.0)) - 12.0,
                    CANDLE_RADIUS, CANDLE_COLOR, 0.92, 0.42,
                    "candle", _as_owner(getattr(ag, "id", None)),
                )
            for group in ("props", "structures", "fires"):
                for ob in _iter_attr(world, (group,)):
                    self._add_object_light(ob)
        except Exception:                                   # pragma: no cover
            log.debug("lighting rebuild failed", exc_info=True)
        return len(self.sources)

    def _add_object_light(self, ob: Any) -> None:
        try:
            if getattr(ob, "destroyed", False) or not getattr(ob, "alive", True):
                return
            oid = _as_owner(getattr(ob, "id", None))
            x = _as_float(getattr(ob, "x", 0.0))
            y = _as_float(getattr(ob, "y", 0.0))
            if getattr(ob, "burning", False):
                heat = _clamp(_as_float(getattr(ob, "burn", 1.0), 1.0), 0.15, 1.0)
                self.add_source(x, y - 8.0, BURNING_RADIUS * (0.6 + 0.4 * heat),
                                BURNING_COLOR, 0.85 * heat, 0.55, "burning", oid)
                return
            kind = ""
            for attr in ("kind", "type", "name"):
                v = getattr(ob, attr, None)
                if isinstance(v, str) and v:
                    kind = v.lower()
                    break
            if kind in ("firepit", "fire", "campfire", "hearth", "brazier"):
                if not getattr(ob, "lit", True):
                    return
                stage_ok = getattr(ob, "complete", True)
                if not stage_ok:
                    return
                self.add_source(x, y - 6.0, FIRE_RADIUS, FIRE_COLOR,
                                0.95, 0.30, "fire", oid)
            elif kind in ("lava", "vent"):
                self.add_source(x, y, FIRE_RADIUS * 0.7, (255, 96, 30),
                                0.7, 0.18, "glow", oid)
        except Exception:                                   # pragma: no cover
            return

    # --------------------------------------------------------------- flash --
    def add_flash(
        self,
        intensity: float = 1.0,
        decay_sec: float = 0.4,
        color: tuple[int, int, int] = LIGHTNING_COLOR,
    ) -> None:
        """Fire a transient global additive light (lightning, meteor impact)."""
        try:
            f = Flash(
                intensity=_clamp(_as_float(intensity, 1.0), 0.0, 2.0),
                decay=max(0.02, _as_float(decay_sec, 0.4)),
                color=_as_color(color, LIGHTNING_COLOR),
            )
            self.flashes.append(f)
            if len(self.flashes) > _MAX_FLASHES:
                del self.flashes[: len(self.flashes) - _MAX_FLASHES]
            self._recompute_flash()
        except Exception:                                   # pragma: no cover
            return

    def flash_value(self) -> float:
        """Current summed flash contribution, 0..1."""
        return self._flash_total

    def flash_color(self) -> tuple[int, int, int]:
        """Intensity-weighted blend of the active flash colours."""
        return self._flash_color

    def _recompute_flash(self) -> None:
        total = 0.0
        acc = np.zeros(3, dtype=np.float64)
        for f in self.flashes:
            v = f.value()
            if v <= 0.0:
                continue
            total += v
            acc += np.asarray(f.color, dtype=np.float64) * v
        self._flash_total = _clamp(total)
        if total > 0.0:
            self._flash_color = tuple(int(max(0, min(255, c))) for c in (acc / total))  # type: ignore[assignment]
        else:
            self._flash_color = (0, 0, 0)

    # -------------------------------------------------------------- ticking --
    def tick(self, dt: float) -> None:
        """Advance flicker phases and decay flashes. Never raises."""
        try:
            step = _as_float(dt, 0.0)
            if step <= 0.0:
                return
            step = min(step, 0.25)
            self.t += step
            if self.flashes:
                for f in self.flashes:
                    f.t += step
                self.flashes = [f for f in self.flashes if not f.expired]
            self._recompute_flash()
            self._refresh_flicker()
        except Exception:                                   # pragma: no cover
            log.debug("lighting tick failed", exc_info=True)

    def _flicker_mul(self, s: LightSource) -> float:
        if s.flicker <= 0.0:
            return 1.0
        amp = s.flicker * (0.55 + 0.85 * _clamp(self.wind_gust))
        w = (0.62 * math.sin(self.t * 7.9 + s.phase)
             + 0.38 * math.sin(self.t * 17.3 + s.phase * 2.3))
        return _clamp(1.0 + amp * w, 0.12, 1.65)

    def _refresh_flicker(self) -> None:
        for s in self.sources:
            s.fmul = self._flicker_mul(s)

    # -------------------------------------------------------------- ambient --
    def ambient_base(self, scene: str, world_time: float) -> float:
        """The day/night curve for this scene, 0..1, before any flash."""
        lo, hi = _SCENE_AMBIENT.get(scene, _DEFAULT_AMBIENT)
        return _clamp(lo + (hi - lo) * daylight_factor(world_time))

    def ambient(self, scene: str, world_time: float) -> float:
        """Ambient light level including the active flash term, 0..1."""
        amb = _clamp(self.ambient_base(scene, world_time) + self._flash_total)
        self.last_ambient = amb
        return amb

    def ambient_rgb(self, scene: str, world_time: float) -> tuple[int, int, int]:
        """Ambient as a colour, for the lightmap fill. Tinted by time and scene."""
        base = self.ambient_base(scene, world_time)
        d = daylight_factor(world_time)
        night = np.asarray(_NIGHT_RGB, dtype=np.float64)
        day = np.asarray(_DAY_RGB, dtype=np.float64)
        tint = night + (day - night) * d
        tint = tint * np.asarray(_SCENE_RGB.get(scene, (1.0, 1.0, 1.0)), dtype=np.float64)
        rgb = tint * (base * 255.0)
        if self._flash_total > 0.0:
            rgb = rgb + np.asarray(self._flash_color, dtype=np.float64) * self._flash_total
        rgb = np.clip(rgb, 0.0, 255.0)
        return (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    # ---------------------------------------------------------- point light --
    def light_at(self, x: float, y: float) -> float:
        """Light reaching (x, y) from *sources and flashes only*, 0..1.

        Deliberately excludes ambient so the renderer can ask "is this agent in
        somebody's light?" — see ``total_light_at`` for source+ambient, and
        ``is_lit`` for the silhouette decision.
        """
        total = self._flash_total
        if total >= 1.0:
            return 1.0
        for s in self.sources:
            r = s.radius
            if r <= 0.0:
                continue
            dx = x - s.x
            if dx > r or dx < -r:
                continue
            dy = y - s.y
            if dy > r or dy < -r:
                continue
            d2 = dx * dx + dy * dy
            rr = r * r
            if d2 >= rr:
                continue
            f = 1.0 - math.sqrt(d2) / r
            total += s.intensity * s.fmul * f * f
            if total >= 1.0:
                return 1.0
        return total

    def total_light_at(self, x: float, y: float, scene: str, world_time: float) -> float:
        """Ambient + local light at a point, 0..1."""
        return _clamp(self.ambient_base(scene, world_time) + self.light_at(x, y))

    def is_lit(self, x: float, y: float, threshold: float = SILHOUETTE_THRESHOLD) -> bool:
        """True if a point catches enough source light to render in colour."""
        return self.light_at(x, y) >= threshold

    def light_at_many(self, xs: Sequence[float], ys: Sequence[float]) -> np.ndarray:
        """Vectorised ``light_at`` for a batch of points (float32 array 0..1)."""
        try:
            x = np.asarray(xs, dtype=np.float32)
            y = np.asarray(ys, dtype=np.float32)
            out = np.full(x.shape, float(self._flash_total), dtype=np.float32)
            for s in self.sources:
                r = float(s.radius)
                if r <= 0.0:
                    continue
                d = np.sqrt((x - s.x) ** 2 + (y - s.y) ** 2) / r
                f = np.clip(1.0 - d, 0.0, 1.0)
                out += (f * f) * np.float32(s.intensity * s.fmul)
            return np.clip(out, 0.0, 1.0)
        except Exception:                                   # pragma: no cover
            return np.zeros(np.shape(xs), dtype=np.float32)

    # ------------------------------------------------------------------ io --
    def to_dict(self) -> dict[str, Any]:
        return {
            "t": float(self.t),
            "wind_gust": float(self.wind_gust),
            "last_ambient": float(self.last_ambient),
            "sources": [s.to_dict() for s in self.sources],
            "flashes": [f.to_dict() for f in self.flashes],
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Lighting":
        lig = cls()
        if not isinstance(d, dict):
            return lig
        lig.t = max(0.0, _as_float(d.get("t"), 0.0))
        lig.wind_gust = _clamp(_as_float(d.get("wind_gust"), 0.0))
        lig.last_ambient = _clamp(_as_float(d.get("last_ambient"), 0.5))
        raw_sources = d.get("sources")
        if isinstance(raw_sources, list):
            lig.sources = [LightSource.from_dict(s) for s in raw_sources[:_MAX_SOURCES]]
        raw_flashes = d.get("flashes")
        if isinstance(raw_flashes, list):
            lig.flashes = [Flash.from_dict(f) for f in raw_flashes[:_MAX_FLASHES]]
            lig.flashes = [f for f in lig.flashes if not f.expired]
        lig._recompute_flash()
        lig._refresh_flicker()
        return lig


def _iter_attr(obj: Any, names: tuple[str, ...]) -> list[Any]:
    """First present attribute among ``names``, coerced to a list. Never raises."""
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


# ------------------------------------------------------------ smoke test --
if __name__ == "__main__":                                  # pragma: no cover
    lig = Lighting()
    for scene in (SCENE_NIGHT_STORM, SCENE_CLEAR, SCENE_ASHFALL):
        night = lig.ambient_base(scene, 0.0)
        noon = lig.ambient_base(scene, DAY_LENGTH_SEC * 0.5)
        print(f"{scene:12s} night={night:.3f} noon={noon:.3f}")
    lig.add_source(200.0, 400.0, 120.0, CANDLE_COLOR, 0.9, 0.4, "candle", 1)
    print("at candle  :", round(lig.light_at(200.0, 400.0), 3))
    print("40px away  :", round(lig.light_at(240.0, 400.0), 3))
    print("far away   :", round(lig.light_at(900.0, 400.0), 3))
    lig.add_flash(1.0, 0.45, LIGHTNING_COLOR)
    lig.tick(0.05)
    print("flash amb  :", round(lig.ambient(SCENE_NIGHT_STORM, 0.0), 3),
          lig.ambient_rgb(SCENE_NIGHT_STORM, 0.0))
    for _ in range(20):
        lig.tick(1 / 30)
    print("after 0.7s :", round(lig.ambient(SCENE_NIGHT_STORM, 0.0), 3),
          "flashes:", len(lig.flashes))
    round_trip = Lighting.from_dict(lig.to_dict())
    print("roundtrip  :", len(round_trip.sources), "sources ok")
