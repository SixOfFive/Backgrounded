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
# fog overlay (a grey wash that swallows the distance)
# --------------------------------------------------------------------------

_FOG_CACHE: dict[tuple, pygame.Surface] = {}
_FOG_GREY = (198, 202, 208)


def _fog_layer(w: int, h: int, inten: float) -> pygame.Surface:
    """A full-frame grey wash whose opacity rises toward the ground, built from
    a tiny gradient strip and cached per intensity bucket (fog ramps smoothly,
    so ~8 buckets are indistinguishable from continuous)."""
    bucket = max(0, min(8, int(round(inten * 8))))
    key = (w, h, bucket)
    hit = _FOG_CACHE.get(key)
    if hit is not None:
        return hit
    g = bucket / 8.0
    strip = pygame.Surface((1, 256), pygame.SRCALPHA)
    for y in range(256):
        f = y / 255.0
        a = int(255 * g * (0.30 + 0.55 * f))       # a haze everywhere, thick low down
        strip.set_at((0, y), (_FOG_GREY[0], _FOG_GREY[1], _FOG_GREY[2],
                              max(0, min(255, a))))
    surf = pygame.transform.smoothscale(strip, (max(1, w), max(1, h)))
    if len(_FOG_CACHE) > 16:
        _FOG_CACHE.clear()
    _FOG_CACHE[key] = surf
    return surf


def draw_fog_overlay(surf: pygame.Surface, intensity: float, t: float = 0.0) -> None:
    """Wash the whole frame toward a flat grey - the look of standing in fog.

    *intensity* 0..1. Drawn late (after the light composite) so it dims the lit
    scene rather than being multiplied away by the ambient. Fails soft."""
    try:
        inten = clamp(float(intensity), 0.0, 1.0)
        if inten <= 0.02:
            return
        surf.blit(_fog_layer(*surf.get_size(), inten), (0, 0))
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
        cols = np.arange(w, dtype=np.float32)[:, None]
        idx = np.clip(np.rint(cols + shift[None, :]), 0, w - 1).astype(np.int32)
        rows_i = np.broadcast_to(np.arange(h, dtype=np.int32)[None, :], (w, h))
        out = src[idx, rows_i]
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
# maintenance
# --------------------------------------------------------------------------


def clear_caches() -> None:
    """Drop every cached surface. Only needed if the render size changes."""
    global _LIGHT_CACHE_BYTES
    _LIGHT_CACHE.clear()
    _LIGHT_CACHE_BYTES = 0
    _VIGNETTE_CACHE.clear()
    _RAIN_TILE.clear()
    _SCRATCH.clear()


def cache_stats() -> dict[str, int]:
    """Cache occupancy, for the diagnostics overlay."""
    return {
        "lights": len(_LIGHT_CACHE),
        "vignettes": len(_VIGNETTE_CACHE),
        "rain_tiles": len(_RAIN_TILE),
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
