"""Screen-space effects: lightning, vignette, rain overlay, shake, light sprites.

Everything in this module is *presentation only*. Nothing here reads or mutates
simulation state - callers hand in plain geometry and scalars. The expensive
pieces (radial light sprites, the vignette, the rain tile) are built once with
numpy and cached; the per-frame path only blits.

The single most performance-critical entry point is :func:`radial_light_surface`
because the renderer calls it for *every* light *every* frame while building the
lightmap. It is aggressively bucketed and cached.
"""
from __future__ import annotations

import math
import random
from typing import Any, Sequence

import numpy as np
import pygame

from ..constants import RENDER_H, RENDER_W

Color = tuple[int, int, int]
Polyline = list[tuple[float, float]]

# --------------------------------------------------------------------------
# small defensive helpers shared by the other render modules
# --------------------------------------------------------------------------


def attr_num(obj: Any, *names: str, default: float = 0.0) -> float:
    """Read the first numeric attribute in *names* off *obj*, failing soft.

    Zero-argument callables are invoked. Anything that is missing, raises, or
    is not convertible to float yields *default*. This exists so the render
    layer can consume sim objects owned by other modules without hard-coupling
    to their exact attribute names.
    """
    if obj is None:
        return float(default)
    for name in names:
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if val is None:
            continue
        if callable(val):
            try:
                val = val()
            except Exception:
                continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            return f
    return float(default)


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp *v* into ``[lo, hi]``. NaN maps to *lo*.

    The NaN rule matters: a single bad frame (a NaN world_time, a NaN dt) would
    otherwise propagate into an accumulator and poison the visuals permanently.
    Callers rely on ``clamp(nan, 0.0, x) == 0.0`` to mean "skip this frame".
    """
    if v != v:
        return lo
    return lo if v < lo else (hi if v > hi else v)


def smoothstep(x: float, edge0: float = 0.0, edge1: float = 1.0) -> float:
    """Hermite smoothstep of *x* between the two edges, clamped to 0..1."""
    if edge1 == edge0:
        return 0.0 if x < edge0 else 1.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp_color(a: Sequence[float], b: Sequence[float], t: float) -> Color:
    """Linear blend between two colours, returning an int ``(r, g, b)``."""
    t = clamp(t, 0.0, 1.0)
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


_HAS_FBLITS = hasattr(pygame.Surface, "fblits")


def blit_batch(surf: pygame.Surface, seq: Sequence[Any], special_flags: int = 0) -> None:
    """Blit a ``[(source, dest), ...]`` sequence in one C-level call.

    Uses ``Surface.fblits`` when available (pygame-ce >= 2.1.4) and falls back to
    a plain loop otherwise. Fails soft.
    """
    if not seq:
        return
    try:
        if _HAS_FBLITS:
            surf.fblits(seq, special_flags)
            return
        for src, dest in seq:
            surf.blit(src, dest, special_flags=special_flags)
    except Exception:
        return


def _surface_from_rgb(arr: np.ndarray) -> pygame.Surface:
    """Build an opaque Surface from a ``(w, h, 3)`` uint8 array."""
    return pygame.surfarray.make_surface(np.ascontiguousarray(arr, dtype=np.uint8))


# --------------------------------------------------------------------------
# radial light sprites  (the lightmap workhorse)
# --------------------------------------------------------------------------

_LIGHT_CACHE: dict[tuple[int, int, int, int, int, int], pygame.Surface] = {}
_LIGHT_CACHE_MAX = 320
_LIGHT_CACHE_BYTES = 0
_LIGHT_CACHE_BYTE_MAX = 96 * 1024 * 1024   # hard ceiling for an unattended run

MAX_LIGHT_RADIUS = 1400.0


def _radius_bucket(radius: float) -> int:
    """Quantise a radius so slightly different lights share one sprite."""
    r = clamp(float(radius), 2.0, MAX_LIGHT_RADIUS)
    if r < 48.0:
        return max(2, int(round(r / 2.0)) * 2)
    if r < 160.0:
        return int(round(r / 6.0)) * 6
    if r < 480.0:
        return int(round(r / 16.0)) * 16
    return int(round(r / 48.0)) * 48


def radial_light_surface(
    radius: float,
    color: Sequence[int] = (255, 220, 160),
    intensity: float = 1.0,
    falloff: float = 1.6,
) -> pygame.Surface:
    """Return a cached radial-gradient sprite for additive light compositing.

    The result is an opaque Surface that is black at the edges, so blitting it
    with ``special_flags=pygame.BLEND_RGB_ADD`` adds nothing outside the light's
    reach. Blit it centred on the light:

    ``dst.blit(spr, (x - spr.get_width() // 2, y - spr.get_height() // 2),
    special_flags=pygame.BLEND_RGB_ADD)``

    Sprites are keyed by ``(radius_bucket, quantised colour, intensity_bucket,
    falloff_bucket)`` so calling this once per light per frame is cheap.
    """
    try:
        rb = _radius_bucket(radius)
        cr = int(clamp(float(color[0]), 0, 255)) & 0xF8
        cg = int(clamp(float(color[1]), 0, 255)) & 0xF8
        cb = int(clamp(float(color[2]), 0, 255)) & 0xF8
        ib = int(round(clamp(float(intensity), 0.0, 4.0) * 24.0))
        fb = int(round(clamp(float(falloff), 0.4, 6.0) * 4.0))
    except Exception:
        rb, cr, cg, cb, ib, fb = 32, 248, 216, 160, 24, 6

    key = (rb, cr, cg, cb, ib, fb)
    hit = _LIGHT_CACHE.get(key)
    if hit is not None:
        return hit

    inten = ib / 24.0
    fall = fb / 4.0
    size = rb * 2 + 1
    axis = np.arange(size, dtype=np.float32) - float(rb)
    dist = np.sqrt(axis[:, None] ** 2 + axis[None, :] ** 2) / float(rb)
    t = np.clip(1.0 - dist, 0.0, 1.0)
    # smoothstep body plus a small hot core so lights have a readable centre
    s = t * t * (3.0 - 2.0 * t)
    grad = 0.82 * np.power(s, fall, dtype=np.float32) + 0.18 * np.power(s, 6.0, dtype=np.float32)
    grad *= inten
    rgb = grad[:, :, None] * np.asarray((cr, cg, cb), dtype=np.float32)[None, None, :]
    surf = _surface_from_rgb(np.clip(rgb, 0.0, 255.0))

    global _LIGHT_CACHE_BYTES
    if len(_LIGHT_CACHE) >= _LIGHT_CACHE_MAX or _LIGHT_CACHE_BYTES > _LIGHT_CACHE_BYTE_MAX:
        # cheap eviction: drop roughly a quarter of the cache, oldest first
        for k in list(_LIGHT_CACHE)[: max(1, _LIGHT_CACHE_MAX // 4)]:
            dead = _LIGHT_CACHE.pop(k, None)
            if dead is not None:
                _LIGHT_CACHE_BYTES -= dead.get_width() * dead.get_height() * 4
        _LIGHT_CACHE_BYTES = max(0, _LIGHT_CACHE_BYTES)
    _LIGHT_CACHE[key] = surf
    _LIGHT_CACHE_BYTES += size * size * 4
    return surf


def blit_light(
    surf: pygame.Surface,
    x: float,
    y: float,
    radius: float,
    color: Sequence[int] = (255, 220, 160),
    intensity: float = 1.0,
    falloff: float = 1.6,
) -> None:
    """Additively blit a cached radial light centred on ``(x, y)``. Fails soft."""
    try:
        spr = radial_light_surface(radius, color, intensity, falloff)
        half = spr.get_width() // 2
        surf.blit(spr, (int(x) - half, int(y) - half), special_flags=pygame.BLEND_RGB_ADD)
    except Exception:
        return


# --------------------------------------------------------------------------
# lightning
# --------------------------------------------------------------------------


def _displace(points: Polyline, offset: float, rng: random.Random, iters: int,
              decay: float = 0.55) -> Polyline:
    """Midpoint-displacement subdivision of a polyline, perpendicular to it."""
    for _ in range(iters):
        out: Polyline = [points[0]]
        for i in range(len(points) - 1):
            ax, ay = points[i]
            bx, by = points[i + 1]
            dx = bx - ax
            dy = by - ay
            length = math.hypot(dx, dy)
            if length < 1e-6:
                out.append((bx, by))
                continue
            nx = -dy / length
            ny = dx / length
            d = rng.uniform(-1.0, 1.0) * offset
            out.append(((ax + bx) * 0.5 + nx * d, (ay + by) * 0.5 + ny * d))
            out.append((bx, by))
        points = out
        offset *= decay
    return points


def make_lightning_bolt(
    x_top: float,
    y_top: float,
    x_bottom: float,
    y_bottom: float,
    seed: int,
    branch_depth: int = 3,
) -> list[Polyline]:
    """Build a branching lightning bolt as pure geometry.

    Returns a list of polylines; element 0 is the main channel and the rest are
    progressively thinner forks. The result is fully determined by *seed*, so a
    strike can be regenerated (or simply cached) and redrawn identically for
    every frame of its ~0.5 s life.
    """
    try:
        rng = random.Random(int(seed) & 0x7FFFFFFF)
    except Exception:
        rng = random.Random(0)

    span = math.hypot(float(x_bottom) - float(x_top), float(y_bottom) - float(y_top))
    span = max(span, 8.0)
    main = _displace(
        [(float(x_top), float(y_top)), (float(x_bottom), float(y_bottom))],
        span * 0.11,
        rng,
        6,
    )
    bolts: list[Polyline] = [main]

    def fork(parent: Polyline, depth: int, scale: float) -> None:
        if depth <= 0 or len(parent) < 6:
            return
        count = rng.randint(1, 3) if depth == branch_depth else rng.randint(0, 2)
        for _ in range(count):
            i = rng.randrange(len(parent) // 6, max(len(parent) // 6 + 1, len(parent) - 2))
            px, py = parent[i]
            qx, qy = parent[min(i + 1, len(parent) - 1)]
            base_ang = math.atan2(qy - py, qx - px)
            ang = base_ang + rng.uniform(-1.05, 1.05)
            seg = span * scale * rng.uniform(0.24, 0.52)
            ex = px + math.cos(ang) * seg
            ey = py + math.sin(ang) * seg
            child = _displace([(px, py), (ex, ey)], seg * 0.16, rng, 4)
            bolts.append(child)
            fork(child, depth - 1, scale * 0.6)

    fork(main, max(0, int(branch_depth)), 1.0)
    return bolts


_SCRATCH: dict[tuple[int, int], pygame.Surface] = {}


def _scratch(size: tuple[int, int]) -> pygame.Surface:
    """A reusable black scratch surface used for additive passes."""
    surf = _SCRATCH.get(size)
    if surf is None:
        surf = pygame.Surface(size)
        if len(_SCRATCH) > 4:
            _SCRATCH.clear()
        _SCRATCH[size] = surf
    return surf


def lightning_envelope(t: float, life: float) -> float:
    """Brightness 0..1 of a strike *t* seconds into its *life* second lifetime.

    Hot initial flash, a couple of restrikes, then an exponential decay.
    """
    if life <= 0.0:
        return 0.0
    u = clamp(t / life, 0.0, 1.0)
    if u >= 1.0:
        return 0.0
    base = math.exp(-3.4 * u)
    flicker = 0.62 + 0.38 * math.sin(u * 41.0)
    spike = 1.0 if u < 0.045 else 0.0
    return clamp(base * flicker + spike * 0.45 * (1.0 - u / 0.045), 0.0, 1.0)


def draw_lightning(
    surf: pygame.Surface,
    bolt: Sequence[Polyline],
    t: float,
    life: float,
    color: Color = (206, 224, 255),
) -> None:
    """Draw a bolt: wide dim glow pass, then a thin bright core, additively.

    *bolt* is the geometry from :func:`make_lightning_bolt`, *t* is seconds
    since the strike began and *life* its total duration. Fails soft.
    """
    if not bolt:
        return
    try:
        k = lightning_envelope(float(t), float(life))
        if k <= 0.004:
            return
        scratch = _scratch(surf.get_size())
        scratch.fill((0, 0, 0))

        passes = (
            (max(7.0, 15.0 * k), 0.16),   # broad halo
            (max(3.0, 7.0 * k), 0.42),    # inner glow
            (max(1.0, 2.0 + 1.6 * k), 1.0),  # core
        )
        for idx, line in enumerate(bolt):
            if len(line) < 2:
                continue
            main = idx == 0
            pts = [(int(px), int(py)) for px, py in line]
            for width, gain in passes:
                w = max(1, int(round(width * (1.0 if main else 0.45))))
                amp = gain * k * (1.0 if main else 0.55)
                col = (
                    min(255, int(color[0] * amp)),
                    min(255, int(color[1] * amp)),
                    min(255, int(color[2] * amp)),
                )
                if col == (0, 0, 0):
                    continue
                pygame.draw.lines(scratch, col, False, pts, w)
            if main:
                # white-hot centreline
                amp = min(255, int(255 * k))
                if amp > 0:
                    pygame.draw.lines(scratch, (amp, amp, amp), False, pts, 1)

        surf.blit(scratch, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    except Exception:
        return


def lightning_flash_level(t: float, life: float, peak: float = 1.0) -> float:
    """Global ambient boost contributed by a strike - the reveal-then-dark beat."""
    return clamp(lightning_envelope(t, life) * peak, 0.0, 1.0)


# --------------------------------------------------------------------------
# vignette
# --------------------------------------------------------------------------

_VIGNETTE_CACHE: dict[tuple[int, int, int], pygame.Surface] = {}


def vignette_surface(size: tuple[int, int], strength: float = 0.55,
                     radius: float = 0.78) -> pygame.Surface:
    """Cached multiply-surface that darkens the frame edges."""
    w, h = int(size[0]), int(size[1])
    sb = int(round(clamp(strength, 0.0, 1.0) * 40.0))
    key = (w, h, sb)
    hit = _VIGNETTE_CACHE.get(key)
    if hit is not None:
        return hit

    st = sb / 40.0
    xs = (np.arange(w, dtype=np.float32) / max(1.0, w - 1.0)) * 2.0 - 1.0
    ys = (np.arange(h, dtype=np.float32) / max(1.0, h - 1.0)) * 2.0 - 1.0
    # elliptical distance, slightly wider than tall so the corners bite first
    d = np.sqrt((xs[:, None] * 1.0) ** 2 + (ys[None, :] * 1.12) ** 2)
    f = np.clip((d - radius) / max(1e-3, (1.45 - radius)), 0.0, 1.0)
    f = f * f * (3.0 - 2.0 * f)
    mul = np.clip(1.0 - f * st, 0.0, 1.0) * 255.0
    surf = _surface_from_rgb(np.repeat(mul[:, :, None], 3, axis=2))

    if len(_VIGNETTE_CACHE) > 8:
        _VIGNETTE_CACHE.clear()
    _VIGNETTE_CACHE[key] = surf
    return surf


def draw_vignette(surf: pygame.Surface, strength: float = 0.55) -> None:
    """Darken the edges of *surf* with a cached radial mask. Fails soft."""
    try:
        surf.blit(vignette_surface(surf.get_size(), strength), (0, 0),
                  special_flags=pygame.BLEND_RGB_MULT)
    except Exception:
        return


# --------------------------------------------------------------------------
# rain overlay (cheap full-screen haze; the real drops live in particles.py)
# --------------------------------------------------------------------------

_RAIN_TILE: dict[tuple[int, int, int], pygame.Surface] = {}
_RAIN_TILE_SIZE = 256


def _rain_tile(shear: int, density: int, seed: int = 91) -> pygame.Surface:
    """A seamless tile of faint diagonal rain streaks, built once."""
    key = (shear, density, seed)
    hit = _RAIN_TILE.get(key)
    if hit is not None:
        return hit
    n = _RAIN_TILE_SIZE
    tile = pygame.Surface((n, n))
    tile.fill((0, 0, 0))
    rng = random.Random(seed * 7919 + shear * 31 + density)
    dx = shear / 8.0
    for _ in range(density):
        x = rng.uniform(0, n)
        y = rng.uniform(0, n)
        length = rng.uniform(9.0, 20.0)
        v = rng.randint(16, 40)
        col = (int(v * 0.72), int(v * 0.86), v)
        for ox in (-n, 0, n):
            for oy in (-n, 0, n):
                pygame.draw.line(
                    tile, col,
                    (int(x + ox), int(y + oy)),
                    (int(x + ox + dx * length), int(y + oy + length)),
                    1,
                )
    if len(_RAIN_TILE) > 24:
        _RAIN_TILE.clear()
    _RAIN_TILE[key] = tile
    return tile


def draw_rain_overlay(surf: pygame.Surface, intensity: float, wind: float = 0.0,
                      t: float = 0.0) -> None:
    """Scroll a tiled streak texture over the frame for distant rain haze.

    *intensity* 0..1, *wind* in the same units the event system uses (roughly
    -1..1). Additive, so it only ever lightens. Fails soft.
    """
    try:
        inten = clamp(float(intensity), 0.0, 1.0)
        if inten <= 0.01:
            return
        shear = int(round(clamp(float(wind), -2.0, 2.0) * 8.0))
        density = 90 + int(110 * inten)
        density = (density // 20) * 20
        tile = _rain_tile(shear, density)
        tile.set_alpha(int(60 + 140 * inten))
        n = _RAIN_TILE_SIZE
        speed = 620.0 * (0.55 + 0.45 * inten)
        oy = int((t * speed) % n)
        ox = int((t * speed * (shear / 8.0)) % n)
        w, h = surf.get_size()
        for gy in range(-n + oy, h, n):
            for gx in range(-n + ox, w, n):
                surf.blit(tile, (gx, gy), special_flags=pygame.BLEND_RGB_ADD)
        tile.set_alpha(255)
    except Exception:
        return


# --------------------------------------------------------------------------
# haze overlay (a coloured wash that swallows the distance)
# --------------------------------------------------------------------------

_FOG_CACHE: dict[tuple, pygame.Surface] = {}
#: Default wash colour - fog, and the value ``events.HAZE_GREY`` must match.
#: sim/ may not import render/, so the two are kept in step by hand.
_FOG_GREY = (198, 202, 208)


def _fog_layer(w: int, h: int, inten: float, color: Color) -> pygame.Surface:
    """A full-frame wash whose opacity rises toward the ground, built from a
    tiny gradient strip and cached per (colour, intensity bucket) - haze ramps
    smoothly, so ~8 buckets are indistinguishable from continuous.

    The gradient itself is deliberately colour-independent: every scene that
    uses this is claiming the same physical thing (suspended particles you are
    standing *inside*, densest where they have furthest to settle), so only the
    tint changes between fog and a sandstorm.
    """
    bucket = max(0, min(8, int(round(inten * 8))))
    key = (w, h, bucket, color)
    hit = _FOG_CACHE.get(key)
    if hit is not None:
        return hit
    g = bucket / 8.0
    strip = pygame.Surface((1, 256), pygame.SRCALPHA)
    for y in range(256):
        f = y / 255.0
        a = int(255 * g * (0.30 + 0.55 * f))       # a haze everywhere, thick low down
        strip.set_at((0, y), (color[0], color[1], color[2],
                              max(0, min(255, a))))
    surf = pygame.transform.smoothscale(strip, (max(1, w), max(1, h)))
    # Nine buckets per colour, so the cap has to clear a couple of scenes' worth
    # or a rotation between two hazed scenes would thrash the cache every frame.
    if len(_FOG_CACHE) > 40:
        _FOG_CACHE.clear()
    _FOG_CACHE[key] = surf
    return surf


def _as_rgb(value: Any, default: Color) -> Color:
    """Coerce anything the sim might publish into an (r, g, b) byte triple."""
    try:
        r, g, b = (int(c) for c in tuple(value)[:3])
    except (TypeError, ValueError):
        return default
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def draw_fog_overlay(surf: pygame.Surface, intensity: float, t: float = 0.0,
                     color: Any = None, daylight: float = 1.0) -> None:
    """Wash the whole frame toward a flat colour - the look of standing inside
    weather you cannot see through.

    *intensity* 0..1; *color* is the haze tint, defaulting to fog's grey.
    Drawn late (after the light composite) so it dims the lit scene rather than
    being multiplied away by the ambient.

    *daylight* (0..1) is what keeps the haze honest about the time of day. The
    wash is applied over the finished frame, so without it the overlay is the
    one thing in the scene the day/night cycle cannot touch: measured, a fog at
    midnight came out at 113 mean against 128 at noon - a 12% swing, while every
    other daylight-sensitive scene swings several-fold - so fog rotating in at
    2am read as broad daylight. Fog at night is a dark murk, not a white sheet.
    Quantised to eight steps before it reaches the tint so a continuously moving
    sun cannot thrash the layer cache. Fails soft.
    """
    try:
        inten = clamp(float(intensity), 0.0, 1.0)
        if inten <= 0.02:
            return
        tint = _FOG_GREY if color is None else _as_rgb(color, _FOG_GREY)
        dl = round(clamp(float(daylight), 0.0, 1.0) * 8.0) / 8.0
        # Floor rather than a straight multiply: haze is lit by whatever light
        # there is, and a colony's torches under fog should still show it as a
        # glow rather than leaving the night looking merely clear.
        k = 0.26 + 0.74 * dl
        tint = (int(tint[0] * k), int(tint[1] * k), int(tint[2] * k))
        surf.blit(_fog_layer(*surf.get_size(), inten, tint), (0, 0))
    except Exception:
        return


# --------------------------------------------------------------------------
# eclipse twilight ring
# --------------------------------------------------------------------------

_ECLIPSE_BAND_CACHE: dict[tuple, pygame.Surface] = {}
_ECLIPSE_BAND_COLOR = (216, 124, 62)
#: Where the sky meets the ridges, as a fraction of frame height. Mirrors
#: sky.HORIZON_FRAC; duplicated rather than imported because fx sits *below*
#: sky in the render layering and must not depend on it.
_ECLIPSE_BAND_Y = 0.53
#: Darkness below which there is no ring at all. The 360-degree sunset is a
#: totality effect - it appears when the shadow cone is overhead and there is
#: still lit atmosphere on every horizon, not while the day is merely dimmer.
_ECLIPSE_BAND_ONSET = 0.45
_ECLIPSE_BAND_PEAK = 0.42        # scales the additive punch; see below


def _eclipse_band(w: int, h: int, level: int) -> pygame.Surface:
    """Warm horizon band, cached per intensity bucket.

    Built as an RGB surface with the intensity baked into the *colour* rather
    than an alpha, because it is blitted additively - BLEND_RGB_ADD ignores the
    source alpha, so encoding strength there would silently do nothing.
    """
    key = (w, h, level)
    hit = _ECLIPSE_BAND_CACHE.get(key)
    if hit is not None:
        return hit

    g = level / 8.0
    ys = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    # Gaussian centred on the horizon, a little wider below it than above so the
    # glow sits *on* the land rather than floating over it.
    d = (ys - _ECLIPSE_BAND_Y) / np.where(ys < _ECLIPSE_BAND_Y, 0.070, 0.105).astype(np.float32)
    prof = np.exp(-d * d) * (g * _ECLIPSE_BAND_PEAK)
    rgb = prof[:, None] * np.asarray(_ECLIPSE_BAND_COLOR, dtype=np.float32)[None, :]
    strip = pygame.surfarray.make_surface(
        np.clip(rgb, 0.0, 255.0).astype(np.uint8).reshape(1, 256, 3))
    surf = pygame.transform.smoothscale(strip, (max(1, w), max(1, h)))
    if len(_ECLIPSE_BAND_CACHE) > 16:
        _ECLIPSE_BAND_CACHE.clear()
    _ECLIPSE_BAND_CACHE[key] = surf
    return surf


def draw_eclipse_twilight(surf: pygame.Surface, strength: float) -> None:
    """Add the ring of sunset that circles the horizon during totality.

    *strength* is the scene's published eclipse darkness, 0..1; the onset ramp is
    applied here so callers stay a one-liner. Additive, and drawn *after* the
    light composite: this is light arriving from beyond the shadow, not part of
    the lit world, so multiplying it down with everything else would delete the
    one cue that tells a viewer this darkness is not just night. Fails soft.
    """
    try:
        u = clamp(float(strength), 0.0, 1.0)
        band = clamp((u - _ECLIPSE_BAND_ONSET) / (1.0 - _ECLIPSE_BAND_ONSET), 0.0, 1.0)
        if band <= 0.02:
            return
        level = max(1, min(8, int(round(band * 8))))
        w, h = surf.get_size()
        surf.blit(_eclipse_band(w, h, level), (0, 0),
                  special_flags=pygame.BLEND_RGB_ADD)
    except Exception:
        return


# --------------------------------------------------------------------------
# earthquake fissures
# --------------------------------------------------------------------------

_FISSURE_CACHE: dict[tuple, pygame.Surface] = {}
_FISSURE_DARK = (11, 9, 8)          # the inside of the crack: near-black
_FISSURE_LIP = (108, 98, 84)        # freshly broken rock along the rim


def _fissure_sprite(width: int, depth: int, seed: int) -> pygame.Surface:
    """A jagged black crack ``width`` px across at the rim, ``depth`` px deep.

    The heightmap already carries the dip the sim carved, so this is only the
    part a heightmap cannot express: that the dip is *open*. A tapered dark
    wedge with a broken-rock lip is enough to turn a smooth notch into a hole.

    Cached per (width, depth, seed). The sim opens a handful of scars and their
    geometry stops changing once they finish opening, so all but the first
    second and a half of a scar's life is a cache hit and a blit.
    """
    key = (width, depth, seed)
    hit = _FISSURE_CACHE.get(key)
    if hit is not None:
        return hit

    surf = pygame.Surface((max(1, width), max(1, depth)), pygame.SRCALPHA)
    rng = random.Random(seed)
    steps = max(3, depth // 3)
    left: Polyline = []
    right: Polyline = []
    mid = width * 0.5
    for i in range(steps + 1):
        u = i / steps
        # ``** 0.75`` rather than a straight taper: a linear wedge reads as a
        # drawn triangle, whereas a crack stays wide most of the way down and
        # then closes quickly, which is what rock actually does.
        half = mid * (1.0 - u) ** 0.75
        wob = (rng.random() - 0.5) * width * 0.12
        y = u * max(1, depth - 1)
        left.append((mid - half + wob, y))
        right.append((mid + half + wob, y))

    poly = left + right[::-1]
    if len(poly) >= 3:
        pygame.draw.polygon(surf, (*_FISSURE_DARK, 238), poly)
        pygame.draw.lines(surf, (*_FISSURE_LIP, 140), False, left, 1)
        pygame.draw.lines(surf, (*_FISSURE_LIP, 140), False, right, 1)

    if len(_FISSURE_CACHE) > 24:
        _FISSURE_CACHE.clear()
    _FISSURE_CACHE[key] = surf
    return surf


def draw_fissures(surf: pygame.Surface, cracks: Sequence[Any]) -> None:
    """Darken the ground scars an earthquake tore open.

    *cracks* is a plain ``(x, ground_y, width, depth, seed)`` sequence - this
    module never reads sim state, so the caller resolves the terrain height
    itself. Each crack hangs from its own ground point, which is what keeps it
    registered with the notch once the light composite has flattened the
    contour. Fails soft: a scar that does not draw is a cosmetic loss.
    """
    if not cracks:
        return
    batch: list[tuple[pygame.Surface, tuple[int, int]]] = []
    for crack in cracks:
        try:
            x, gy, width, depth, seed = crack
            w = int(clamp(float(width), 4.0, 160.0))
            d = int(clamp(float(depth), 2.0, 120.0))
            spr = _fissure_sprite(w, d, int(seed) & 0xFFFFF)
            batch.append((spr, (int(float(x) - w * 0.5), int(float(gy)) - 1)))
        except Exception:
            continue
    blit_batch(surf, batch)


# --------------------------------------------------------------------------
# lava flows
# --------------------------------------------------------------------------

#: The three tones of a flow, from the crusted rim inward to the incandescent
#: channel. Warmer and much brighter than MATERIAL_COLORS[MAT_LAVA] (200,78,24),
#: which is the flat band the terrain pass already painted underneath this: that
#: one has been through the light composite and is sitting at whatever the ash
#: cloud left of the ambient, so the whole job here is to put back the light the
#: flow is producing *itself*.
_LAVA_CRUST = (128, 26, 10)
_LAVA_BODY = (238, 84, 20)
_LAVA_CORE = (255, 214, 128)
#: Colour and reach of the glow it throws onto everything around it.
_LAVA_GLOW = (255, 92, 26)
_LAVA_GLOW_R = 150.0
#: px between glow sprites along the flow. Each is a cached radial blit, so this
#: is the cost dial: at 96 a full-width flow costs four of them.
_LAVA_GLOW_STEP = 96.0


def draw_lava_flow(surf: pygame.Surface, flows: Sequence[Any], t: float = 0.0) -> None:
    """Draw molten flows as emissive bands following the ground.

    *flows* is a plain ``[(points, heat), ...]`` sequence - this module never
    reads sim state, so the caller samples the terrain and hands over the
    contour. ``points`` is a list of ``(x, y)`` along the surface and ``heat``
    is 0..1, the flow's own temperature.

    Drawn *after* the light composite, for the reason the UFO beam is: this is
    light the world is emitting, not light falling on it, and multiplying it
    down by an ambient the same eruption has just crushed would leave the one
    bright thing on screen as dark as everything else.

    The bright core is dashed and the dashes travel, which is the only thing
    here that says the rock is *moving*: a static band reads as painted ground
    no matter how orange it is. Fails soft - a frame without lava is still a
    frame.
    """
    try:
        for flow in flows:
            pts, heat = flow[0], clamp(float(flow[1]), 0.0, 1.0)
            if heat <= 0.02 or len(pts) < 2:
                continue
            # Glow first, so the band itself is drawn over its own halo rather
            # than being washed out by it.
            span = abs(float(pts[-1][0]) - float(pts[0][0]))
            steps = max(1, int(span / _LAVA_GLOW_STEP) + 1)
            for i in range(steps):
                u = i / max(1, steps - 1) if steps > 1 else 0.5
                idx = int(u * (len(pts) - 1))
                gx, gy = pts[idx]
                blit_light(surf, gx, gy, _LAVA_GLOW_R * (0.55 + 0.45 * heat),
                           _LAVA_GLOW, 0.42 * heat)

            line = [(int(px), int(py)) for px, py in pts]
            pygame.draw.lines(surf, _LAVA_CRUST, False, line, max(3, int(9 * heat)))
            pygame.draw.lines(surf, _LAVA_BODY, False, line, max(1, int(5 * heat)))

            # Travelling bright dashes along the channel. One line call per lit
            # run rather than per segment, so a full-width flow is a handful of
            # calls however finely the caller sampled it.
            phase = float(t) * 0.55
            run: list[tuple[int, int]] = []
            for i, p in enumerate(line):
                lit = ((i * 0.11 - phase) % 1.0) < 0.42
                if lit:
                    run.append(p)
                    continue
                if len(run) >= 2:
                    pygame.draw.lines(surf, _LAVA_CORE, False, run, 2)
                run = []
            if len(run) >= 2:
                pygame.draw.lines(surf, _LAVA_CORE, False, run, 2)
    except Exception:
        return


# --------------------------------------------------------------------------
# heat shimmer
# --------------------------------------------------------------------------


def draw_heat_shimmer(surf: pygame.Surface, rect: Sequence[int], t: float,
                      amount: float = 3.0, band: int = 4) -> None:
    """Horizontally wobble a rectangular region of *surf* to fake rising heat.

    Cheap row-shift distortion done with numpy on a snapshot of the region.
    Fails soft (a shimmer that does not draw is better than a crash).
    """
    try:
        amt = clamp(float(amount), 0.0, 24.0)
        if amt < 0.4:
            return
        clip_rect = pygame.Rect(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        clip_rect = clip_rect.clip(surf.get_rect())
        if clip_rect.width < 8 or clip_rect.height < 8:
            return
        region = surf.subsurface(clip_rect).copy()
        src = pygame.surfarray.pixels3d(region).copy()
        w, h = clip_rect.width, clip_rect.height
        rows = np.arange(h, dtype=np.float32)
        phase = rows / max(1.0, float(band))
        shift = (np.sin(phase * 0.9 + t * 7.0) * amt
                 + np.sin(phase * 0.31 - t * 4.3) * amt * 0.45)
        # stronger near the bottom of the region (close to the heat source)
        shift *= np.clip(rows / max(1.0, h - 1.0), 0.0, 1.0) ** 0.6

        # Whole-pixel shifts applied as run-length slice copies. Every column in
        # a row moves by the same amount, so the obvious ``src[idx, rows]``
        # gather is doing per-pixel work to express a per-row fact: it
        # materialises two int32 arrays the size of the region and reads through
        # them, which measured at ~6 ms for the 1600x220 ground band the
        # heatwave scene asks for, against ~0.5 ms for the slices below. That is
        # the difference between an effect that fits inside a 60 Hz frame and
        # one that eats a third of it, and the output is identical: the same
        # rounded shift, and the same clamp-to-edge on the columns the shift
        # pulls in from outside the region.
        k = np.rint(shift).astype(np.int32)
        np.clip(k, -(w - 1), w - 1, out=k)
        out = np.empty_like(src)
        start = 0
        for i in range(1, h + 1):
            if i < h and k[i] == k[start]:
                continue                    # same shift; keep growing the run
            s = int(k[start])
            dst, col = out[:, start:i], src[:, start:i]
            if s == 0:
                dst[:] = col
            elif s > 0:                     # sample from further right
                dst[:w - s] = col[s:]
                dst[w - s:] = col[w - 1:w]
            else:                           # ...or further left
                n = -s
                dst[n:] = col[:w - n]
                dst[:n] = col[0:1]
            start = i
        surf.blit(_surface_from_rgb(out), clip_rect.topleft)
    except Exception:
        return


# --------------------------------------------------------------------------
# screen shake
# --------------------------------------------------------------------------

MAX_SHAKE = 26
_shake_phase = 0.0


def shake_offset(magnitude: float, t: float, seed: int = 1337) -> tuple[float, float]:
    """Smooth pseudo-random 2D offset of the given *magnitude* at time *t*.

    Deterministic in ``(magnitude, t, seed)``: no hidden state, so the same
    quake replays identically.
    """
    m = clamp(float(magnitude), 0.0, float(MAX_SHAKE))
    if m <= 0.0:
        return (0.0, 0.0)
    s = (int(seed) & 0xFFFF) * 0.000431
    dx = (math.sin(t * 41.3 + s) * 0.6 + math.sin(t * 97.7 + s * 3.1) * 0.4) * m
    dy = (math.sin(t * 53.9 + s * 1.7) * 0.6 + math.sin(t * 89.1 + s * 2.3) * 0.4) * m * 0.7
    return (dx, dy)


def apply_shake(offset: Any) -> tuple[int, int]:
    """Turn a shake request into a whole-pixel blit offset for the renderer.

    Accepts either an ``(dx, dy)`` pair (typically from :func:`shake_offset`) or
    a bare magnitude, in which case an offset is generated from a free-running
    internal phase. The result is rounded and clamped to ``+-MAX_SHAKE`` so a
    runaway magnitude can never push the scene off screen.

    Usage: ``dst.blit(scene, apply_shake(mag))``.
    """
    global _shake_phase
    try:
        if isinstance(offset, (int, float)):
            _shake_phase += 1.0 / 60.0
            dx, dy = shake_offset(float(offset), _shake_phase)
        else:
            dx = float(offset[0])
            dy = float(offset[1])
    except Exception:
        return (0, 0)
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return (0, 0)
    return (
        int(round(clamp(dx, -MAX_SHAKE, MAX_SHAKE))),
        int(round(clamp(dy, -MAX_SHAKE, MAX_SHAKE))),
    )


def shake_blit(dst: pygame.Surface, src: pygame.Surface, offset: Any,
               fill: Color = (0, 0, 0)) -> None:
    """Blit *src* onto *dst* with a shake offset, filling the exposed edges."""
    try:
        ox, oy = apply_shake(offset)
        if ox or oy:
            dst.fill(fill)
        dst.blit(src, (ox, oy))
    except Exception:
        try:
            dst.blit(src, (0, 0))
        except Exception:
            return


# --------------------------------------------------------------------------
# crossings: bridges and ladders
#
# These two are drawn as geometry rather than baked into the atlas because
# their size is not a property of the *kind*, it is a property of the gap. A
# bridge spans whatever chasm the terrain generated (60-190 px on the maps
# measured) and a ladder is as long as the face is tall; a fixed 108x32 sprite
# stretched to fit would put planking where the walkable deck is not.
#
# Both take plain geometry - the caller samples the terrain and hands over
# points - so what gets drawn is the same line the physics is standing on
# rather than an independent guess at where it ought to be.
#
# Palette runs brighter than the atlas earth tones on purpose. The scene is
# multiplied down by the ambient before any light is added back, and a crossing
# is usually out over a gap with no fire near it, so it has nothing but its own
# luminance and its rim highlight to stay legible with at night.
# --------------------------------------------------------------------------

CROSSING_WOOD: Color = (96, 70, 44)
CROSSING_WOOD_LIGHT: Color = (140, 108, 68)
CROSSING_WOOD_DARK: Color = (54, 40, 27)
CROSSING_ROPE: Color = (150, 134, 100)
CROSSING_RIM: Color = (198, 208, 224)

#: px between planks / rungs. Sized to read at 1600x1000 without turning a
#: 160 px span into a solid bar - about 22 planks across a typical chasm.
CROSSING_PLANK_PITCH = 7.0
CROSSING_RUNG_PITCH = 9.0
#: px between the posts that carry the deck down onto whatever is below it.
CROSSING_POST_PITCH = 34.0
#: how far a post is allowed to reach down before it is simply not drawn -
#: a bridge over a 300 px chasm should not sprout 300 px legs.
CROSSING_POST_MAX = 62.0
CROSSING_RAIL_H = 13.0


def _crossing_alpha(ruined: bool) -> int:
    return 120 if ruined else 255


def draw_bridge(
    surf: pygame.Surface,
    deck: Sequence[Sequence[float]],
    ground: Sequence[float] | None = None,
    *,
    progress: float = 1.0,
    ruined: bool = False,
) -> None:
    """Draw a plank bridge along *deck*, a polyline of ``(x, y)`` deck points.

    *ground* is the raw ground y under each deck point, used to stand the
    support posts on something; omit it and the posts are skipped. *progress*
    0..1 plants only that fraction of the planking, so a half-built bridge
    reads as a half-built bridge. Fails soft and silently - a frame without a
    bridge on it is still a frame.
    """
    try:
        pts = [(float(p[0]), float(p[1])) for p in deck
               if len(p) >= 2 and math.isfinite(float(p[0])) and math.isfinite(float(p[1]))]
        if len(pts) < 2:
            return
        prog = clamp(float(progress), 0.0, 1.0)
        alpha = _crossing_alpha(ruined)
        x_lo, x_hi = pts[0][0], pts[-1][0]
        span = max(1.0, x_hi - x_lo)

        def y_at(x: float) -> float:
            """Deck y at x, by linear search over the (small) sample list."""
            if x <= pts[0][0]:
                return pts[0][1]
            if x >= pts[-1][0]:
                return pts[-1][1]
            for i in range(1, len(pts)):
                if pts[i][0] >= x:
                    x0, y0 = pts[i - 1]
                    x1, y1 = pts[i]
                    t = (x - x0) / max(1e-6, x1 - x0)
                    return y0 + (y1 - y0) * t
            return pts[-1][1]

        # 1. support posts, drawn first so the deck sits on top of them.
        if ground is not None and not ruined:
            gl = [float(g) for g in ground]
            if len(gl) == len(pts):
                x = x_lo + CROSSING_POST_PITCH * 0.5
                while x < x_hi:
                    i = min(len(pts) - 1, int((x - x_lo) / span * (len(pts) - 1)))
                    top = y_at(x)
                    bot = min(gl[i], top + CROSSING_POST_MAX)
                    if bot - top > 5.0:
                        pygame.draw.line(surf, (*CROSSING_WOOD_DARK, alpha),
                                         (int(x), int(top)), (int(x), int(bot)), 3)
                        pygame.draw.line(surf, (*CROSSING_WOOD, alpha),
                                         (int(x) - 1, int(top)), (int(x) - 1, int(bot)), 1)
                    x += CROSSING_POST_PITCH

        # 2. the deck: a dark bed, then alternating planks over it.
        bed = [(int(px), int(py) + 2) for px, py in pts]
        if len(bed) >= 2:
            pygame.draw.lines(surf, (*CROSSING_WOOD_DARK, alpha), False, bed, 5)
        planted = x_lo + span * prog
        x = x_lo
        i = 0
        while x < planted:
            py = y_at(x)
            col = CROSSING_WOOD_LIGHT if i % 2 else CROSSING_WOOD
            pygame.draw.line(surf, (*col, alpha),
                             (int(x), int(py) - 2), (int(x), int(py) + 3), 3)
            x += CROSSING_PLANK_PITCH
            i += 1

        # 3. the highlight along the walking line. This is the part that keeps
        #    a bridge readable against a black sky at 3am, so it goes on last
        #    and it is the brightest thing in the drawing.
        lit = [(int(px), int(py) - 2) for px, py in pts if px <= planted]
        if len(lit) >= 2:
            pygame.draw.lines(surf, (*CROSSING_RIM, 150 if ruined else 210),
                              False, lit, 1)

        # 4. rope handrail, once the thing is far enough along to have one.
        if prog > 0.55 and not ruined:
            rail = [(int(px), int(py - CROSSING_RAIL_H)) for px, py in pts if px <= planted]
            if len(rail) >= 2:
                pygame.draw.lines(surf, (*CROSSING_ROPE, alpha), False, rail, 2)
                pygame.draw.lines(surf, (*CROSSING_RIM, 90), False,
                                  [(px, py - 1) for px, py in rail], 1)
                step = max(1, len(rail) // 7)
                for k in range(0, len(rail), step):
                    rx, ry = rail[k]
                    pygame.draw.line(surf, (*CROSSING_WOOD, alpha),
                                     (rx, ry), (rx, int(y_at(rx))), 1)
    except Exception:
        return


def draw_ladder(
    surf: pygame.Surface,
    foot: Sequence[float],
    top: Sequence[float],
    *,
    progress: float = 1.0,
    ruined: bool = False,
) -> None:
    """Draw a ladder leaning from *foot* ``(x, y)`` up to *top* ``(x, y)``.

    Two rails with rungs between them, laid along the line the climb overlay
    actually put there. *progress* 0..1 builds it from the foot upward.
    """
    try:
        fx0, fy0 = float(foot[0]), float(foot[1])
        tx0, ty0 = float(top[0]), float(top[1])
        if not all(math.isfinite(v) for v in (fx0, fy0, tx0, ty0)):
            return
        dx, dy = tx0 - fx0, ty0 - fy0
        length = math.hypot(dx, dy)
        if length < 6.0:
            return
        prog = clamp(float(progress), 0.0, 1.0)
        alpha = _crossing_alpha(ruined)
        ux, uy = dx / length, dy / length            # along the ladder
        # Perpendicular, normalised: the rails sit either side of the climb line.
        px, py = -uy, ux
        half = 5.0
        end_t = length * prog

        def at(t: float, off: float) -> tuple[int, int]:
            return (int(fx0 + ux * t + px * off), int(fy0 + uy * t + py * off))

        for off in (-half, half):
            a, b = at(0.0, off), at(end_t, off)
            pygame.draw.line(surf, (*CROSSING_WOOD, alpha), a, b, 3)
            pygame.draw.line(surf, (*CROSSING_RIM, 120 if ruined else 170),
                             at(0.0, off - 1.0), at(end_t, off - 1.0), 1)

        t = CROSSING_RUNG_PITCH * 0.5
        while t < end_t:
            a, b = at(t, -half), at(t, half)
            pygame.draw.line(surf, (*CROSSING_WOOD_LIGHT, alpha), a, b, 2)
            t += CROSSING_RUNG_PITCH

        # A stub at the foot so the ladder reads as resting on the ground
        # rather than floating an inch above it.
        if not ruined:
            pygame.draw.line(surf, (*CROSSING_WOOD_DARK, alpha),
                             at(-3.0, -half), at(-3.0, half), 4)
    except Exception:
        return


# --------------------------------------------------------------------------
# litter and bonfires
# --------------------------------------------------------------------------

#: Rubbish is drab on purpose. At wallpaper scale a hundred specks of anything
#: saturated reads as confetti or, worse, as berries somebody should be picking.
#: These sit close to dirt and dry grass so a piece is legible when you look at
#: it and disappears into the ground when you are not - which is exactly what a
#: dense pile has to do to be atmosphere rather than noise.
LITTER_DARK: Color = (46, 42, 34)
LITTER_BODY: Color = (104, 95, 76)
LITTER_LIGHT: Color = (146, 136, 112)

#: One sprite per silhouette, baked once. A piece of litter is 9x6 px, so the
#: whole cache is four tiny surfaces and the per-frame cost of a hundred of them
#: is a hundred blits of a 54 px sprite.
_LITTER_CACHE: dict[int, pygame.Surface] = {}
_LITTER_W, _LITTER_H = 9, 6


def _litter_sprite(shape: int) -> pygame.Surface:
    """Baked debris sprite for *shape* 0..3, bottom-centred on the ground."""
    key = int(shape) & 3
    hit = _LITTER_CACHE.get(key)
    if hit is not None:
        return hit
    s = pygame.Surface((_LITTER_W, _LITTER_H), pygame.SRCALPHA)
    if key == 0:                      # a crumpled scrap
        pygame.draw.polygon(s, LITTER_BODY, [(1, 5), (3, 2), (6, 1), (8, 4), (5, 5)])
        pygame.draw.line(s, LITTER_LIGHT, (3, 3), (6, 2))
        pygame.draw.line(s, LITTER_DARK, (1, 5), (8, 5))
    elif key == 1:                    # a gnawed bone / a snapped stick
        pygame.draw.line(s, LITTER_BODY, (1, 4), (7, 3), 2)
        pygame.draw.line(s, LITTER_LIGHT, (2, 3), (6, 3))
        s.set_at((0, 4), LITTER_DARK)
        s.set_at((8, 3), LITTER_DARK)
    elif key == 2:                    # scattered shards
        for px, py in ((1, 5), (4, 4), (7, 5), (5, 2)):
            s.set_at((px, py), LITTER_BODY)
            s.set_at((px, py - 1), LITTER_LIGHT)
    else:                             # a small heap
        pygame.draw.polygon(s, LITTER_BODY, [(1, 5), (4, 1), (8, 5)])
        pygame.draw.line(s, LITTER_LIGHT, (4, 2), (6, 4))
        pygame.draw.line(s, LITTER_DARK, (1, 5), (8, 5))
    if len(_LITTER_CACHE) < 8:
        _LITTER_CACHE[key] = s
    return s


def draw_litter(surf: pygame.Surface, x: float, y: float,
                shape: int = 0, flip: bool = False) -> None:
    """Blit one piece of litter with its base sitting on ``(x, y)``.

    *flip* mirrors the sprite so a pile of the same shape does not read as a
    row of stamps. Fails soft: a frame missing a speck of rubbish is still a
    frame.
    """
    try:
        spr = _litter_sprite(shape)
        if flip:
            spr = pygame.transform.flip(spr, True, False)
        surf.blit(spr, (int(x) - _LITTER_W // 2, int(y) - _LITTER_H + 1))
    except Exception:
        return


#: Flame palette, outside in. The core is deliberately near-white: a bonfire has
#: to be the brightest thing on a night frame by a clear margin, and the light
#: composite multiplies this down before the lightmap adds it back.
BONFIRE_FLAME: tuple[Color, ...] = (
    (176, 54, 16), (238, 116, 28), (255, 190, 66), (255, 244, 206),
)


def draw_bonfire(surf: pygame.Surface, x: float, y: float, width: float,
                 height: float, t: float, strength: float = 1.0) -> None:
    """A heaped, roaring fire centred on ``(x, y)`` (``y`` = the ground line).

    Four nested teardrops, each narrower, shorter and hotter than the one
    outside it, with the inner ones leaning on their own phase so the whole
    thing writhes. *strength* 0..1 scales height and lick, so a bonfire visibly
    builds and dies back rather than popping in and out.

    Presentation only - no cache, because the shape is a function of ``t`` and
    there are never more than a handful of firepits in a colony. Four polygons
    each is cheaper than the cache lookup would be.
    """
    try:
        k = clamp(float(strength), 0.0, 1.0)
        if k <= 0.01:
            return
        w = max(6.0, float(width))
        h = max(8.0, float(height)) * (0.55 + 0.45 * k)
        cx, base = float(x), float(y)
        # A dark bed of embers so the flames have something to stand on and the
        # pit does not look like fire hovering over grass.
        pygame.draw.ellipse(
            surf, (58, 30, 18),
            pygame.Rect(int(cx - w * 0.6), int(base - 4), int(w * 1.2), 8))
        for i, col in enumerate(BONFIRE_FLAME):
            f = 1.0 - i * 0.24                      # 1.00 .. 0.28
            hw = w * 0.5 * f
            top = base - h * f
            # Each layer sways on its own phase and frequency; the inner ones
            # faster, so the core flickers inside a steadier outer envelope.
            sway = math.sin(t * (3.1 + i * 1.7) + i * 1.9) * w * 0.10 * (i + 1) * 0.4
            lean = math.sin(t * (1.7 + i * 0.6)) * w * 0.06
            pts = [
                (cx - hw + lean, base),
                (cx - hw * 0.55 + lean, base - h * f * 0.45),
                (cx + sway, top),
                (cx + hw * 0.55 + lean, base - h * f * 0.45),
                (cx + hw + lean, base),
            ]
            pygame.draw.polygon(surf, col, [(int(px), int(py)) for px, py in pts])
        # A few sparks lifting off the top, seeded off t so they climb and reset.
        for i in range(3):
            ph = (t * 0.8 + i * 0.37) % 1.0
            sx = cx + math.sin(t * 2.3 + i * 2.1) * w * 0.35
            sy = base - h * (0.9 + 0.6 * ph)
            a = int(220 * (1.0 - ph) * k)
            if a > 8:
                surf.fill((255, 200, 120), pygame.Rect(int(sx), int(sy), 2, 2))
    except Exception:
        return


# --------------------------------------------------------------------------
# maintenance
# --------------------------------------------------------------------------


def clear_caches() -> None:
    """Drop every cached surface. Only needed if the render size changes."""
    global _LIGHT_CACHE_BYTES
    _LIGHT_CACHE.clear()
    _LIGHT_CACHE_BYTES = 0
    _VIGNETTE_CACHE.clear()
    _ECLIPSE_BAND_CACHE.clear()
    _FISSURE_CACHE.clear()
    _RAIN_TILE.clear()
    _SCRATCH.clear()
    _LITTER_CACHE.clear()


def cache_stats() -> dict[str, int]:
    """Cache occupancy, for the diagnostics overlay."""
    return {
        "lights": len(_LIGHT_CACHE),
        "vignettes": len(_VIGNETTE_CACHE),
        "rain_tiles": len(_RAIN_TILE),
        "fissures": len(_FISSURE_CACHE),
        "scratch": len(_SCRATCH),
    }


if __name__ == "__main__":  # pragma: no cover - smoke test
    import os
    import time

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    scene = pygame.Surface((RENDER_W, RENDER_H))
    scene.fill((14, 16, 26))

    bolt = make_lightning_bolt(620, -20, 540, 560, seed=42, branch_depth=3)
    print(f"bolt segments: {sum(len(p) for p in bolt)} across {len(bolt)} polylines")

    t0 = time.perf_counter()
    for i in range(60):
        draw_rain_overlay(scene, 0.7, wind=0.6, t=i / 60.0)
        blit_light(scene, 400 + i, 620, 90, (255, 190, 110), 0.9)
        blit_light(scene, 900, 600, 150, (255, 150, 60), 0.8)
        draw_lightning(scene, bolt, t=(i % 30) / 60.0, life=0.5)
        draw_vignette(scene)
    dt = (time.perf_counter() - t0) / 60.0
    print(f"full fx pass: {dt * 1000:.2f} ms/frame  caches={cache_stats()}")
    print("shake:", apply_shake(shake_offset(8.0, 1.23)), apply_shake(6.0))
    draw_heat_shimmer(scene, (200, 400, 300, 200), 1.0, 4.0)
    print("ok")
