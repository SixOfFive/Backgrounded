"""Sky and parallax backdrop.

Layer 1 and 2 of the render pipeline. Everything here is read-only with respect
to the simulation: it consumes ``scene`` (a ``SCENE_*`` string), ``world_time``
(seconds), the event system (for wind) and the lighting object (for ambient),
and paints a backdrop.

Performance rule of the module: never build a 1280x800 image in Python. The
vertical gradient is computed as a 1xH numpy strip, turned into a Surface and
scaled up, and the whole thing is cached per ``(scene, coarse time bucket)``.
Stars, clouds and ridge silhouettes are baked once and re-blitted.

Time convention: ``day_phase`` 0.0 = midnight, 0.25 = sunrise, 0.5 = noon,
0.75 = sunset. If the ``lighting`` object exposes its own ``day_phase`` that is
preferred, so the sky can never disagree with the ambient curve.
"""
from __future__ import annotations

import math
import random
from itertools import repeat as _repeat
from typing import Any

import numpy as np
import pygame

from ..constants import (
    DAY_LENGTH_SEC,
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
)
from .fx import (
    attr_num,
    blit_batch,
    clamp,
    lerp_color,
    radial_light_surface,
    smoothstep,
)

Color = tuple[int, int, int]

# ------------------------------------------------------------------ palette --

_NIGHT_TOP = (7, 9, 20)
_NIGHT_BOT = (26, 32, 54)
_DAY_TOP = (58, 116, 190)
_DAY_BOT = (162, 198, 224)
_HORIZON_GLOW = (208, 108, 52)      # added near the horizon at dawn/dusk

STAR_SEED = 20260725
CLOUD_SEED = 5150
HORIZON_FRAC = 0.53                 # where the sky "meets" the ridges, as h frac

# scene -> (tint, tint_strength, per-channel gain, cloud_density, star_mul, aurora_ok)
# The per-channel gain is what gives each scene its identity: a scalar gain over
# a blue day sky just makes everything mauve.
_SceneSky = tuple[Color, float, tuple[float, float, float], float, float, bool]
_SCENE_SKY: dict[str, _SceneSky] = {
    SCENE_CLEAR:       ((0, 0, 0),       0.00, (1.00, 1.00, 1.00), 0.30, 1.00, True),
    SCENE_NIGHT_STORM: ((10, 13, 24),    0.72, (0.46, 0.50, 0.62), 1.00, 0.06, False),
    SCENE_WILDFIRE:    ((172, 72, 26),   0.62, (1.05, 0.80, 0.52), 0.70, 0.22, False),
    SCENE_MUDSLIDE:    ((70, 60, 46),    0.55, (0.74, 0.66, 0.52), 0.90, 0.18, False),
    SCENE_BLIZZARD:    ((182, 190, 202), 0.66, (0.80, 0.84, 0.90), 1.00, 0.08, False),
    SCENE_FLOOD:       ((44, 66, 82),    0.55, (0.60, 0.72, 0.80), 0.90, 0.18, False),
    SCENE_METEOR:      ((18, 14, 36),    0.58, (0.68, 0.64, 0.86), 0.28, 0.95, True),
    SCENE_ASHFALL:     ((84, 38, 28),    0.66, (0.86, 0.58, 0.46), 0.85, 0.12, False),
}
_DEFAULT_SKY = _SCENE_SKY[SCENE_CLEAR]

# scene -> ridge silhouette base colour
_SCENE_RIDGE: dict[str, Color] = {
    SCENE_CLEAR:       (46, 58, 62),
    SCENE_NIGHT_STORM: (16, 20, 32),
    SCENE_WILDFIRE:    (54, 30, 22),
    SCENE_MUDSLIDE:    (44, 38, 30),
    SCENE_BLIZZARD:    (94, 104, 118),
    SCENE_FLOOD:       (34, 46, 54),
    SCENE_METEOR:      (22, 20, 38),
    SCENE_ASHFALL:     (44, 32, 30),
}

# ------------------------------------------------------------------- caches --

_GRAD_CACHE: dict[tuple, pygame.Surface] = {}
_GRAD_CACHE_MAX = 72
_STAR_DATA: dict[tuple, tuple[np.ndarray, ...]] = {}
_STAR_SPRITES: dict[tuple[int, int, int], pygame.Surface] = {}
_CLOUD_SPRITES: dict[tuple, list[pygame.Surface]] = {}
_CLOUD_TINTED: dict[tuple, pygame.Surface] = {}
_CLOUD_LAYOUT: dict[tuple, list[tuple[int, float, float, float]]] = {}
_DISC_CACHE: dict[tuple, pygame.Surface] = {}
_RIDGE_CACHE: dict[tuple, tuple[pygame.Surface, int]] = {}
_AURORA_CACHE: dict[str, Any] = {"bucket": None, "surf": None}

_cloud_drift: float = 0.0
_last_world_time: float | None = None


# --------------------------------------------------------------------------
# time of day
# --------------------------------------------------------------------------


def day_phase(world_time: float, lighting: Any = None) -> float:
    """Normalised time of day in ``[0, 1)``; 0 = midnight, 0.5 = noon.

    Defers to ``lighting.day_phase(world_time)`` / ``lighting.phase`` when the
    lighting module offers one, so sky and ambient stay in lockstep.
    """
    if lighting is not None:
        for attr in ("day_phase", "sun_phase"):
            fn = getattr(lighting, attr, None)
            if not callable(fn):
                continue
            for args in ((world_time,), ()):
                try:
                    val = float(fn(*args))
                except Exception:
                    continue
                if math.isfinite(val):
                    return val % 1.0
        for name in ("phase", "day_frac", "time_of_day"):
            try:
                val = getattr(lighting, name)
            except Exception:
                continue
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                return float(val) % 1.0
    try:
        wt = float(world_time)
    except (TypeError, ValueError):
        wt = 0.0
    if not math.isfinite(wt):
        wt = 0.0
    return (wt / float(DAY_LENGTH_SEC)) % 1.0


def sun_elevation(phase: float) -> float:
    """-1 at midnight, +1 at noon. Cheap stand-in for a real solar altitude."""
    return -math.cos(2.0 * math.pi * phase)


def daylight(phase: float) -> float:
    """0 at night, 1 in full day, smooth across dawn and dusk."""
    return smoothstep(sun_elevation(phase), -0.24, 0.30)


def night_factor(phase: float) -> float:
    """1 at deep night, 0 in daylight - the star/aurora/firefly gate."""
    return 1.0 - smoothstep(sun_elevation(phase), -0.34, 0.02)


def _scene_sky(scene: str) -> _SceneSky:
    """Sky styling for a scene id; unknown scenes fall back to clear skies."""
    return _SCENE_SKY.get(scene, _DEFAULT_SKY)


def _ambient(lighting: Any, scene: str | None = None,
             world_time: float | None = None) -> float:
    """Best-effort ambient level 0..1 from whatever the lighting object offers.

    Tries the *pure* accessors first (``ambient_base(scene, world_time)`` and the
    cached ``last_ambient``) before any call that might update lighting state -
    render code must never mutate the simulation, not even a memo field.
    """
    if lighting is None:
        return 0.5

    fn = getattr(lighting, "ambient_base", None)
    if callable(fn) and scene is not None and world_time is not None:
        try:
            val = float(fn(scene, world_time))
            if math.isfinite(val):
                return clamp(val, 0.0, 1.0)
        except Exception:
            pass

    val = attr_num(lighting, "last_ambient", "ambient_level", "level",
                   "current_ambient", default=-1.0)
    if val >= 0.0:
        return clamp(val, 0.0, 1.0)

    fn = getattr(lighting, "ambient", None)
    if callable(fn):
        arg_sets: list[tuple] = []
        if scene is not None and world_time is not None:
            arg_sets.append((scene, world_time))
        if world_time is not None:
            arg_sets.append((world_time,))
        arg_sets.append(())
        for args in arg_sets:
            try:
                out = float(fn(*args))
            except Exception:
                continue
            if math.isfinite(out):
                return clamp(out, 0.0, 1.0)
    return 0.5


# --------------------------------------------------------------------------
# gradient
# --------------------------------------------------------------------------


def _gradient_surface(scene: str, phase: float, w: int, h: int) -> pygame.Surface:
    """Vertical sky gradient as a Surface. Built from a 1xH numpy strip."""
    tint, strength, gain, _cd, _sm, _aur = _scene_sky(scene)
    elev = sun_elevation(phase)
    dl = smoothstep(elev, -0.24, 0.30)

    top = np.array(_NIGHT_TOP, dtype=np.float32) * (1.0 - dl) + np.array(_DAY_TOP, dtype=np.float32) * dl
    bot = np.array(_NIGHT_BOT, dtype=np.float32) * (1.0 - dl) + np.array(_DAY_BOT, dtype=np.float32) * dl

    t = (np.arange(h, dtype=np.float32) / max(1.0, h - 1.0))[:, None]
    col = top[None, :] * (1.0 - t) + bot[None, :] * t

    # dawn / dusk band, ramping up to the horizon line and holding below it
    twilight = math.exp(-((elev - 0.02) / 0.26) ** 2)
    if twilight > 0.01:
        band = np.clip(t / max(0.05, HORIZON_FRAC), 0.0, 1.0) ** 2.4
        col = col + np.array(_HORIZON_GLOW, dtype=np.float32)[None, :] * band * twilight

    if strength > 0.0:
        col = col * (1.0 - strength) + np.array(tint, dtype=np.float32)[None, :] * strength
    col = col * np.array(gain, dtype=np.float32)[None, :]

    arr = np.clip(col, 0.0, 255.0).astype(np.uint8).reshape(1, h, 3)
    strip = pygame.surfarray.make_surface(arr)
    return pygame.transform.scale(strip, (w, h))


def _gradient(scene: str, phase: float, w: int, h: int) -> pygame.Surface:
    """Cached gradient. Phase is bucketed so it rebuilds a few times a minute."""
    bucket = int(phase * 240.0) % 240
    key = (scene, bucket, w, h)
    hit = _GRAD_CACHE.get(key)
    if hit is not None:
        return hit
    surf = _gradient_surface(scene, bucket / 240.0, w, h)
    if len(_GRAD_CACHE) >= _GRAD_CACHE_MAX:
        for k in list(_GRAD_CACHE)[: _GRAD_CACHE_MAX // 3]:
            _GRAD_CACHE.pop(k, None)
    _GRAD_CACHE[key] = surf
    return surf


# --------------------------------------------------------------------------
# stars
# --------------------------------------------------------------------------

_STAR_TINTS: tuple[Color, ...] = (
    (236, 242, 255), (198, 214, 255), (255, 236, 208), (222, 236, 240),
)
_STAR_COUNT = 340
_STAR_SIZES = 3
_STAR_LEVELS = 6                    # alpha quantisation buckets
_STAR_OFFSET = (0, 1, 3)            # sprite half-size per size class


def _star_field(w: int, h: int, seed: int = STAR_SEED) -> tuple[np.ndarray, ...]:
    """Fixed star positions and twinkle parameters, generated once per seed."""
    key = (w, h, seed)
    hit = _STAR_DATA.get(key)
    if hit is not None:
        return hit

    rng = np.random.default_rng(seed)
    n = _STAR_COUNT
    x = rng.integers(0, max(1, w), n).astype(np.int32)
    # crowd the upper sky, thin out towards the horizon
    y = (np.power(rng.random(n), 0.62) * (h * 0.80)).astype(np.int32)
    mag = rng.random(n).astype(np.float32)
    size = np.where(mag > 0.94, 2, np.where(mag > 0.62, 1, 0)).astype(np.int32)
    tint = rng.integers(0, len(_STAR_TINTS), n).astype(np.int32)
    base = (0.30 + 0.70 * np.power(mag, 1.4)).astype(np.float32)
    rate = (0.45 + 2.7 * rng.random(n)).astype(np.float32)
    phase = (rng.random(n) * math.tau).astype(np.float32)
    off = np.asarray(_STAR_OFFSET, dtype=np.int32)[size]
    dest = np.stack([x - off, y - off], axis=1).astype(np.int32)
    sprite_key = (tint * _STAR_SIZES + size).astype(np.int32)

    data = (dest, sprite_key, base, rate, phase)
    if len(_STAR_DATA) > 4:
        _STAR_DATA.clear()
    _STAR_DATA[key] = data
    return data


def _star_sprite(tint: int, size: int, level: int) -> pygame.Surface:
    """Tiny pre-rendered star dot, keyed by tint / size class / alpha bucket."""
    key = (tint, size, level)
    hit = _STAR_SPRITES.get(key)
    if hit is not None:
        return hit
    col = _STAR_TINTS[tint % len(_STAR_TINTS)]
    a = int(255 * (level + 1) / _STAR_LEVELS)
    if size >= 2:
        spr = pygame.Surface((7, 7), pygame.SRCALPHA)
        pygame.draw.line(spr, (*col, a // 5), (0, 3), (6, 3), 1)
        pygame.draw.line(spr, (*col, a // 5), (3, 0), (3, 6), 1)
        pygame.draw.circle(spr, (*col, max(1, a // 3)), (3, 3), 3)
        pygame.draw.circle(spr, (*col, a), (3, 3), 2)
    elif size == 1:
        spr = pygame.Surface((3, 3), pygame.SRCALPHA)
        spr.fill((*col, max(1, a // 4)), pygame.Rect(1, 0, 1, 1))
        spr.fill((*col, max(1, a // 4)), pygame.Rect(1, 2, 1, 1))
        spr.fill((*col, max(1, a // 3)), pygame.Rect(0, 1, 1, 1))
        spr.fill((*col, max(1, a // 3)), pygame.Rect(2, 1, 1, 1))
        spr.fill((*col, a), pygame.Rect(1, 1, 1, 1))
    else:
        spr = pygame.Surface((1, 1), pygame.SRCALPHA)
        spr.fill((*col, a))
    _STAR_SPRITES[key] = spr
    return spr


def _draw_stars(surf: pygame.Surface, night: float, star_mul: float,
                world_time: float) -> None:
    """Twinkling star field: one vectorised pass, one batched fblits call."""
    amp = night * star_mul
    if amp <= 0.02:
        return
    w, h = surf.get_size()
    dest, sprite_key, base, rate, phase = _star_field(w, h)

    alpha = base * amp * (0.62 + 0.38 * np.sin(world_time * rate + phase))
    # level L renders at alpha (L+1)/_STAR_LEVELS, so the -1 lets the faintest
    # stars fade out entirely at dawn instead of popping off at 1/6 alpha
    lvl = np.clip((alpha * _STAR_LEVELS).astype(np.int32) - 1, -1, _STAR_LEVELS - 1)
    vis = lvl >= 0
    if not vis.any():
        return
    keys = sprite_key[vis] * _STAR_LEVELS + lvl[vis]
    pos = dest[vis]

    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    pts = pos[order].tolist()
    uniq, starts = np.unique(ks, return_counts=False, return_index=True)

    seq: list[tuple[pygame.Surface, list[int]]] = []
    ustarts = starts.tolist()
    ukeys = uniq.tolist()
    total = len(pts)
    for i, k in enumerate(ukeys):
        a0 = ustarts[i]
        a1 = ustarts[i + 1] if i + 1 < len(ustarts) else total
        level = k % _STAR_LEVELS
        rest = k // _STAR_LEVELS
        spr = _star_sprite(rest // _STAR_SIZES, rest % _STAR_SIZES, level)
        seq.extend(zip(_repeat(spr), pts[a0:a1]))
    blit_batch(surf, seq)


# --------------------------------------------------------------------------
# sun and moon
# --------------------------------------------------------------------------


def _body_track(u: float, w: int, horizon_y: float, arc: float) -> tuple[float, float]:
    """Position along a rising/setting arc; *u* 0 = rise (left), 1 = set (right)."""
    x = w * (0.07 + 0.86 * u)
    y = horizon_y - arc * math.sin(math.pi * clamp(u, -0.12, 1.12))
    return (x, y)


def sun_position(phase: float, w: int = RENDER_W, h: int = RENDER_H) -> tuple[float, float, bool]:
    """``(x, y, visible)`` of the sun for a given day phase."""
    u = (phase - 0.25) / 0.5
    horizon = h * HORIZON_FRAC
    x, y = _body_track(u, w, horizon, h * 0.44)
    return (x, y, -0.06 <= u <= 1.06)


def moon_position(phase: float, w: int = RENDER_W, h: int = RENDER_H) -> tuple[float, float, bool]:
    """``(x, y, visible)`` of the moon - opposite the sun."""
    u = ((phase + 0.5) % 1.0 - 0.25) / 0.5
    horizon = h * HORIZON_FRAC
    x, y = _body_track(u, w, horizon, h * 0.41)
    return (x, y, -0.06 <= u <= 1.06)


def _sun_disc(r: int) -> pygame.Surface:
    key = ("sun", r)
    hit = _DISC_CACHE.get(key)
    if hit is not None:
        return hit
    size = r * 2 + 4
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size // 2
    pygame.draw.circle(s, (255, 248, 226, 255), (c, c), r)
    pygame.draw.circle(s, (255, 226, 168, 120), (c, c), r + 2, 2)
    _DISC_CACHE[key] = s
    return s


def _moon_disc(r: int, seed: int = 11) -> pygame.Surface:
    key = ("moon", r, seed)
    hit = _DISC_CACHE.get(key)
    if hit is not None:
        return hit
    size = r * 2 + 4
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size // 2
    pygame.draw.circle(s, (226, 232, 244, 255), (c, c), r)
    rng = random.Random(seed)
    for _ in range(5):
        cr = max(1, int(r * rng.uniform(0.10, 0.24)))
        ang = rng.uniform(0, math.tau)
        rad = rng.uniform(0.0, r * 0.62)
        cx = int(c + math.cos(ang) * rad)
        cy = int(c + math.sin(ang) * rad)
        pygame.draw.circle(s, (200, 208, 224, 255), (cx, cy), cr)
    # terminator shading on the lower right
    pygame.draw.circle(s, (206, 214, 230, 90), (c + max(1, r // 4), c + max(1, r // 5)), r)
    _DISC_CACHE[key] = s
    return s


_MAX_SKY_GLOW = 300          # px; keeps cached glow sprites off the heap ceiling
_GLOW_LEVELS = 6             # intensity quantisation for sky glows


def _glow(surf: pygame.Surface, x: float, y: float, radius: float,
          color: Color, intensity: float, falloff: float = 1.8) -> None:
    """Additive sky glow with coarsely bucketed radius/intensity.

    Sky glows are large, so they get a much coarser cache key than the lightmap
    lights do - otherwise a smoothly changing sunset would rebuild a megapixel
    gradient every frame.
    """
    inten = clamp(intensity, 0.0, 1.0)
    if inten <= 0.02:
        return
    level = max(1, int(round(inten * _GLOW_LEVELS)))
    r = int(clamp(radius, 8.0, _MAX_SKY_GLOW))
    r = max(8, (r // 12) * 12)
    spr = radial_light_surface(r, color, level / _GLOW_LEVELS, falloff)
    half = spr.get_width() // 2
    surf.blit(spr, (int(x) - half, int(y) - half), special_flags=pygame.BLEND_RGB_ADD)


def _draw_celestial(surf: pygame.Surface, phase: float, scene: str,
                    dl: float, night: float) -> None:
    w, h = surf.get_size()
    _tint, _st, _gain, _cd, _sm, _aur = _scene_sky(scene)
    storm = scene in (SCENE_NIGHT_STORM, SCENE_BLIZZARD, SCENE_ASHFALL, SCENE_MUDSLIDE)
    veil = 0.30 if storm else 1.0

    mx, my, mvis = moon_position(phase, w, h)
    if mvis and night > 0.05:
        r = max(6, int(h * 0.026))
        _glow(surf, mx, my, r * 5.5, (150, 176, 226), 0.45 * night * veil, falloff=2.4)
        disc = _moon_disc(r)
        disc.set_alpha(int(clamp(255 * night * (0.45 + 0.55 * veil), 0, 255)))
        surf.blit(disc, (int(mx) - disc.get_width() // 2, int(my) - disc.get_height() // 2))

    sx, sy, svis = sun_position(phase, w, h)
    if svis and dl > 0.02:
        r = max(7, int(h * 0.030))
        low = 1.0 - abs(sun_elevation(phase))
        _glow(surf, sx, sy, r * (7.0 + 4.0 * low), (255, 196, 120),
              (0.35 + 0.4 * low) * dl * veil, falloff=2.0)
        disc = _sun_disc(r)
        disc.set_alpha(int(clamp(255 * dl * (0.35 + 0.65 * veil), 0, 255)))
        surf.blit(disc, (int(sx) - disc.get_width() // 2, int(sy) - disc.get_height() // 2))

    # horizon glow band at dawn / dusk, anchored under whichever body is low
    twilight = math.exp(-((sun_elevation(phase) - 0.02) / 0.22) ** 2)
    if twilight > 0.03 and not storm:
        gx = sx if svis else w * 0.5
        _glow(surf, gx, h * HORIZON_FRAC, _MAX_SKY_GLOW, (210, 118, 58),
              0.55 * twilight, falloff=1.4)


# --------------------------------------------------------------------------
# clouds
# --------------------------------------------------------------------------

_CLOUD_COUNT = 7


def _cloud_sprites(seed: int = CLOUD_SEED) -> list[pygame.Surface]:
    """Bake soft elliptical cloud masks (white RGB, gaussian alpha)."""
    key = (seed,)
    hit = _CLOUD_SPRITES.get(key)
    if hit is not None:
        return hit

    rng = np.random.default_rng(seed)
    sprites: list[pygame.Surface] = []
    for i in range(_CLOUD_COUNT):
        w = int(180 + rng.integers(0, 190))
        h = int(48 + rng.integers(0, 46))
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        alpha = np.zeros((w, h), dtype=np.float32)
        for _ in range(int(rng.integers(4, 8))):
            cx = float(rng.uniform(w * 0.18, w * 0.82))
            cy = float(rng.uniform(h * 0.34, h * 0.68))
            rx = float(rng.uniform(w * 0.16, w * 0.34))
            ry = float(rng.uniform(h * 0.22, h * 0.42))
            d = ((xs[:, None] - cx) / rx) ** 2 + ((ys[None, :] - cy) / ry) ** 2
            alpha += np.exp(-d * 1.9)
        alpha = np.clip(alpha, 0.0, 1.0) ** 1.25
        # soften the very bottom so clouds do not look like cut-outs
        fade = np.clip(1.0 - (ys / max(1.0, h - 1.0) - 0.62) / 0.38, 0.0, 1.0)
        alpha *= fade[None, :] * 0.55 + 0.45
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        try:
            rgb = pygame.surfarray.pixels3d(surf)
            rgb[:, :, :] = 255
            del rgb
            av = pygame.surfarray.pixels_alpha(surf)
            av[:, :] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
            del av
        except Exception:
            surf.fill((255, 255, 255, 90))
        sprites.append(surf)
    _CLOUD_SPRITES[key] = sprites
    return sprites


def _cloud_layout(w: int, h: int, seed: int = CLOUD_SEED) -> list[tuple[int, float, float, float]]:
    """``(sprite_idx, y, speed, x0)`` for each drifting cloud, baked once."""
    key = (w, h, seed)
    hit = _CLOUD_LAYOUT.get(key)
    if hit is not None:
        return hit
    rng = random.Random(seed)
    layout: list[tuple[int, float, float, float]] = []
    for i in range(12):
        band = i % 3
        y = h * (0.05 + 0.16 * band) + rng.uniform(-h * 0.03, h * 0.03)
        speed = (0.35 + 0.45 * band) * rng.uniform(0.8, 1.25)
        layout.append((rng.randrange(_CLOUD_COUNT), y, speed, rng.uniform(0, w * 1.6)))
    _CLOUD_LAYOUT[key] = layout
    return layout


def _tinted_cloud(idx: int, color: Color) -> pygame.Surface:
    """Cloud mask multiplied by a quantised tint. Keyed by the exact tint used."""
    tint = (color[0] & 0xF0, color[1] & 0xF0, color[2] & 0xF0)
    key = (idx, *tint)
    hit = _CLOUD_TINTED.get(key)
    if hit is not None:
        return hit
    base = _cloud_sprites()[idx % _CLOUD_COUNT]
    spr = base.copy()
    spr.fill((*tint, 255), special_flags=pygame.BLEND_RGBA_MULT)
    if len(_CLOUD_TINTED) > 96:
        _CLOUD_TINTED.clear()
    _CLOUD_TINTED[key] = spr
    return spr


def _draw_clouds(surf: pygame.Surface, scene: str, dl: float, drift: float,
                 wind: float) -> None:
    w, h = surf.get_size()
    tint, strength, _gain, density, _sm, _aur = _scene_sky(scene)
    if density <= 0.02:
        return
    night_col = (26, 30, 44)
    day_col = (206, 210, 220)
    col = lerp_color(night_col, day_col, dl)
    if strength > 0.0:
        col = lerp_color(col, tint, strength * 0.55)

    layout = _cloud_layout(w, h)
    count = max(2, int(len(layout) * clamp(density, 0.0, 1.0)))
    alpha = int(clamp(90 + 130 * density, 0, 255))
    for i in range(count):
        idx, y, speed, x0 = layout[i]
        spr = _tinted_cloud(idx, col)
        span = w + spr.get_width()
        x = (x0 + drift * speed * (1.0 if wind >= 0.0 else 1.0)) % span - spr.get_width()
        spr.set_alpha(alpha)
        surf.blit(spr, (int(x), int(y)))


# --------------------------------------------------------------------------
# aurora
# --------------------------------------------------------------------------

_AURORA_W = 176
_AURORA_H = 88
_AURORA_COLORS = ((44, 210, 130), (60, 150, 220), (150, 90, 210))


def _aurora_band(t: float, strength: float, w: int, h: int) -> pygame.Surface:
    """Build the (expensive) upscaled aurora ribbon surface."""
    xs = np.linspace(0.0, 1.0, _AURORA_W, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, _AURORA_H, dtype=np.float32)
    rgb = np.zeros((_AURORA_W, _AURORA_H, 3), dtype=np.float32)
    for b in range(3):
        f1 = 1.4 + 0.7 * b
        f2 = 3.1 + 1.3 * b
        yc = (0.30 + 0.11 * b
              + 0.075 * np.sin(xs * f1 * math.tau + t * (0.13 + 0.05 * b))
              + 0.040 * np.sin(xs * f2 * math.tau - t * 0.19 + b))
        thick = 0.042 + 0.016 * b
        d = (ys[None, :] - yc[:, None]) / thick
        val = np.exp(-d * d)
        # vertical curtain striations, drifting sideways
        val *= 0.55 + 0.45 * np.sin(xs * 46.0 + t * 0.9 + b * 2.0)[:, None] ** 2
        # fade the ribbon out towards the edges of the sky
        val *= np.clip(np.sin(xs * math.pi) * 1.4, 0.0, 1.0)[:, None]
        rgb += val[:, :, None] * np.asarray(_AURORA_COLORS[b], dtype=np.float32)
    rgb *= clamp(strength, 0.0, 1.0) * 0.62
    small = pygame.surfarray.make_surface(np.clip(rgb, 0, 255).astype(np.uint8))
    return pygame.transform.smoothscale(small, (w, int(h * 0.56)))


def _draw_aurora(surf: pygame.Surface, t: float, strength: float) -> None:
    """Additive aurora ribbons, rebuilt at ~8 Hz and cached in between."""
    if strength <= 0.03:
        return
    w, h = surf.get_size()
    bucket = (int(t * 8.0), int(strength * 12.0), w, h)
    band = _AURORA_CACHE.get("surf")
    if band is None or _AURORA_CACHE.get("bucket") != bucket:
        band = _aurora_band(bucket[0] / 8.0, bucket[1] / 12.0, w, h)
        _AURORA_CACHE["surf"] = band
        _AURORA_CACHE["bucket"] = bucket
    surf.blit(band, (0, 0), special_flags=pygame.BLEND_RGB_ADD)


# --------------------------------------------------------------------------
# public: sky
# --------------------------------------------------------------------------


def draw_sky(surf: pygame.Surface, scene: str, world_time: float,
             events: Any = None, lighting: Any = None) -> None:
    """Paint layer 1: gradient, stars, sun/moon, clouds, aurora, horizon glow.

    Read-only. *events* is consulted (defensively) for ``wind``; *lighting* for
    the day phase and ambient level. Never raises.
    """
    global _cloud_drift, _last_world_time
    try:
        w, h = surf.get_size()
        phase = day_phase(world_time, lighting)
        dl = daylight(phase)
        night = night_factor(phase)
        _tint, _st, _gain, _density, star_mul, aurora_ok = _scene_sky(scene)

        wind = attr_num(events, "wind", "wind_x", "wind_speed", default=0.0)
        wind = clamp(wind, -3.0, 3.0)

        # advance the cloud drift from the delta of world_time so a pause or a
        # save/load jump cannot make the clouds teleport
        try:
            wt = float(world_time)
        except (TypeError, ValueError):
            wt = 0.0
        if not math.isfinite(wt):
            wt = 0.0
        if _last_world_time is None:
            dt = 0.0
        else:
            dt = clamp(wt - _last_world_time, 0.0, 1.0)
        _last_world_time = wt
        _cloud_drift += dt * (6.0 + wind * 22.0)
        if not math.isfinite(_cloud_drift):
            _cloud_drift = 0.0

        surf.blit(_gradient(scene, phase, w, h), (0, 0))
        _draw_stars(surf, night, star_mul, wt)
        if aurora_ok and night > 0.55:
            _draw_aurora(surf, wt * 0.35, (night - 0.55) / 0.45)
        _draw_celestial(surf, phase, scene, dl, night)
        _draw_clouds(surf, scene, dl, _cloud_drift, wind)
    except Exception:
        try:
            surf.fill((10, 12, 20))
        except Exception:
            pass


# --------------------------------------------------------------------------
# public: parallax ridges
# --------------------------------------------------------------------------


def _ridge_profile(seed: int, layer: int, w: int) -> np.ndarray:
    """Deterministic 0..1 ridge profile via summed sines (no noise library)."""
    rng = np.random.default_rng((int(seed) & 0xFFFFFF) * 977 + layer * 7919)
    xs = np.arange(w, dtype=np.float32) / float(max(1, w))
    prof = np.zeros(w, dtype=np.float32)
    amp = 1.0
    total = 0.0
    freq = 0.7 + 0.5 * layer
    for _ in range(5):
        prof += amp * np.sin((xs * freq + float(rng.random())) * math.tau)
        total += amp
        amp *= 0.56
        freq *= 2.07
    prof /= max(1e-6, total)
    return (prof * 0.5 + 0.5).astype(np.float32)


RIDGE_LAYERS = 3


def _ridge_composite(seed: int, scene: str, tone: int, w: int,
                     h: int) -> tuple[pygame.Surface, int]:
    """Bake all ridge layers into one cropped surface plus its y offset.

    Compositing at bake time means the per-frame cost is a single blit of a
    half-height surface instead of three full-screen alpha blits.
    """
    key = (int(seed) & 0xFFFFFF, scene, tone, w, h)
    hit = _RIDGE_CACHE.get(key)
    if hit is not None:
        return hit

    base = _SCENE_RIDGE.get(scene, _SCENE_RIDGE[SCENE_CLEAR])
    shade = 0.25 + 0.75 * (tone / 8.0)
    comp = pygame.Surface((w, h), pygame.SRCALPHA)
    top_y = h
    step = max(4, w // 180)

    for layer in range(RIDGE_LAYERS):
        prof = _ridge_profile(seed, layer, w)
        top = h * (0.50 + 0.055 * layer)
        amp = h * (0.13 - 0.030 * layer)
        haze = 1.0 - 0.30 * (2 - layer)           # far layers pick up sky haze
        far = 0.42 + 0.29 * layer                 # near layers darker
        col = (
            int(clamp(base[0] * shade * far + 8 * haze, 0, 255)),
            int(clamp(base[1] * shade * far + 9 * haze, 0, 255)),
            int(clamp(base[2] * shade * far + 12 * haze, 0, 255)),
        )
        crest: list[tuple[int, int]] = []
        for x in range(0, w, step):
            crest.append((x, int(top - amp * prof[x])))
        crest.append((w - 1, int(top - amp * prof[w - 1])))
        top_y = min(top_y, min(p[1] for p in crest))
        poly = crest + [(w - 1, h), (0, h)]
        try:
            pygame.draw.polygon(comp, (*col, 255), poly)
        except Exception:
            comp.fill((*col, 255), pygame.Rect(0, int(top), w, h - int(top)))
        # subtle rim light along the crest so ridges read in near-darkness
        rim = (
            min(255, col[0] + 26 + 10 * layer),
            min(255, col[1] + 30 + 10 * layer),
            min(255, col[2] + 38 + 12 * layer),
        )
        try:
            pygame.draw.lines(comp, (*rim, 150), False, crest, 1)
        except Exception:
            pass

    top_y = max(0, min(h - 1, top_y - 2))
    cropped = comp.subsurface(pygame.Rect(0, top_y, w, h - top_y)).copy()
    result = (cropped, top_y)
    if len(_RIDGE_CACHE) > 24:
        _RIDGE_CACHE.clear()
    _RIDGE_CACHE[key] = result
    return result


def draw_parallax(surf: pygame.Surface, terrain_seed: int, scene: str,
                  lighting: Any = None, world_time: float | None = None) -> None:
    """Paint layer 2: three ridge silhouettes, darker and lower with depth.

    Shapes are generated once from *terrain_seed* and cached; the ambient level
    only selects one of 8 tone buckets, so this stays a single blit per frame.
    Never raises.
    """
    try:
        w, h = surf.get_size()
        amb = _ambient(lighting, scene, world_time)
        tone = int(clamp(amb, 0.0, 1.0) * 7.0 + 0.5)
        try:
            seed = int(terrain_seed)
        except (TypeError, ValueError):
            seed = 0
        ridges, y = _ridge_composite(seed, scene, tone, w, h)
        surf.blit(ridges, (0, y))
    except Exception:
        return


# --------------------------------------------------------------------------
# maintenance
# --------------------------------------------------------------------------


def clear_caches() -> None:
    """Drop every cached surface (call if the render resolution changes)."""
    _GRAD_CACHE.clear()
    _STAR_DATA.clear()
    _STAR_SPRITES.clear()
    _CLOUD_SPRITES.clear()
    _CLOUD_TINTED.clear()
    _CLOUD_LAYOUT.clear()
    _DISC_CACHE.clear()
    _RIDGE_CACHE.clear()
    _AURORA_CACHE["surf"] = None
    _AURORA_CACHE["bucket"] = None


def cache_stats() -> dict[str, int]:
    """Cache occupancy, for the diagnostics overlay."""
    return {
        "gradients": len(_GRAD_CACHE),
        "star_sprites": len(_STAR_SPRITES),
        "clouds": len(_CLOUD_TINTED),
        "discs": len(_DISC_CACHE),
        "ridges": len(_RIDGE_CACHE),
    }


if __name__ == "__main__":  # pragma: no cover - smoke test
    import os
    import time

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    from ..paths import CAPTURE_DIR, ensure_dirs

    scene_surf = pygame.Surface((RENDER_W, RENDER_H))

    class _Ev:
        wind = 0.6

    class _Li:
        def ambient(self, wt: float = 0.0) -> float:
            return 0.35

    ev, li = _Ev(), _Li()
    for sc in (SCENE_NIGHT_STORM, SCENE_CLEAR, SCENE_WILDFIRE, SCENE_BLIZZARD):
        draw_sky(scene_surf, sc, 300.0, ev, li)
        draw_parallax(scene_surf, 12345, sc, li, 300.0)

    for label, base in (("night", 60.0), ("dawn", DAY_LENGTH_SEC * 0.235),
                        ("noon", DAY_LENGTH_SEC * 0.5), ("dusk", DAY_LENGTH_SEC * 0.775)):
        draw_sky(scene_surf, SCENE_CLEAR, base, ev, li)          # warm caches
        t0 = time.perf_counter()
        frames = 120
        for i in range(frames):
            wt = base + i / 60.0
            draw_sky(scene_surf, SCENE_CLEAR, wt, ev, li)
            draw_parallax(scene_surf, 12345, SCENE_CLEAR, li, wt)
        dt = (time.perf_counter() - t0) / frames
        print(f"sky+parallax {label:5s}: {dt * 1000:.2f} ms/frame")
    print("caches:", cache_stats())

    try:
        ensure_dirs()
        for name, wt in (("night", 0.0), ("dawn", DAY_LENGTH_SEC * 0.25),
                         ("noon", DAY_LENGTH_SEC * 0.5), ("dusk", DAY_LENGTH_SEC * 0.78)):
            draw_sky(scene_surf, SCENE_CLEAR, wt, ev, li)
            draw_parallax(scene_surf, 12345, SCENE_CLEAR, li, wt)
            pygame.image.save(scene_surf, str(CAPTURE_DIR / f"sky_{name}.png"))
        print("wrote sky_*.png to", CAPTURE_DIR)
    except Exception as exc:
        print("capture skipped:", exc)
