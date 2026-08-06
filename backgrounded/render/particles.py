"""Numpy-backed particle system: rain, snow, embers, smoke, dust, ash, fireflies.

Layers 3, 8 and 9 of the render pipeline all come out of this one system.

State lives in preallocated numpy arrays, not a list of objects: every physics
step is a handful of vector ops over at most ``CAPACITY`` particles, and drawing
groups particles into pre-rendered sprite batches blitted with a single
``fblits`` call per blend mode. There is no per-particle Python attribute access
anywhere in ``update`` or ``draw``.

Read-only with respect to the simulation: ``update`` samples ``terrain`` for
ground collisions and reads wind off the event system, and writes to neither.

COORDINATE SPACE - the one thing to get right in this file, because both
answers are correct and they are correct about different lines:

* particle ``x`` is WORLD space. ``_ground_heights`` indexes the terrain
  profile by it, so embers, dust, splash and settled snow have to scroll with
  the hillside they landed on or rain falls through hills.
* every SPAWN RANGE and the CULL are VIEW space. Weather seeded across all
  ``WORLD_W`` would put three quarters of the pool off camera and thin the
  visible rain to a quarter of what it is now, at four times the cost.

So the pool is world-space and the *window* it is kept inside follows the
camera. ``view_x`` carries that window; ``draw`` subtracts it from the whole x
array in one vectorised op, never per particle - a ``cam.sx()`` call across a
3000-element pool is a Python-level loop on the frame path.
"""
from __future__ import annotations

from itertools import repeat as _repeat
from typing import Any, Sequence

import numpy as np
import pygame

from ..constants import RENDER_H, RENDER_W, WORLD_W
from .camera import IDENTITY, Camera
from .fx import attr_num, blit_batch, clamp, radial_light_surface

Color = tuple[int, int, int]

# ------------------------------------------------------------------- kinds --

K_RAIN = 0
K_SNOW = 1
K_EMBER = 2
K_SMOKE = 3
K_DUST = 4
K_ASH = 5
K_FIREFLY = 6
K_SPLASH = 7
K_SPARK = 8
KIND_COUNT = 9

KIND_NAMES: tuple[str, ...] = (
    "rain", "snow", "ember", "smoke", "dust", "ash", "firefly", "splash", "spark",
)
KIND_IDS: dict[str, int] = {name: i for i, name in enumerate(KIND_NAMES)}

# per-kind physics, indexed by kind id
_GRAVITY = np.array([680.0, 26.0, -78.0, -46.0, 210.0, 15.0, 0.0, 520.0, 460.0], np.float32)
_DRAG = np.array([0.02, 0.85, 1.25, 0.95, 1.55, 0.90, 2.40, 0.35, 1.05], np.float32)
_WINDF = np.array([26.0, 34.0, 20.0, 26.0, 12.0, 30.0, 6.0, 8.0, 6.0], np.float32)
_SWAY = np.array([0.0, 16.0, 24.0, 16.0, 0.0, 12.0, 0.0, 0.0, 0.0], np.float32)
# kinds that stop at the ground rather than passing through it
_COLLIDES = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1], bool)

CAPACITY = 3000

# --------------------------------------------------------------- palette --
# color_idx into this table; kept small so the sprite cache stays bounded.
PALETTE: np.ndarray = np.array([
    (150, 178, 214),   # 0  rain
    (108, 132, 168),   # 1  rain, dim
    (236, 242, 250),   # 2  snow
    (198, 210, 228),   # 3  snow, dim
    (255, 220, 130),   # 4  ember, hot
    (255, 138, 46),    # 5  ember, mid
    (188, 50, 18),     # 6  ember, cold
    (118, 118, 124),   # 7  smoke, light
    (56, 56, 60),      # 8  smoke, dark
    (132, 110, 84),    # 9  dust
    (88, 72, 54),      # 10 dust, dark
    (130, 128, 126),   # 11 ash
    (76, 74, 72),      # 12 ash, dark
    (224, 255, 138),   # 13 firefly
    (176, 200, 232),   # 14 splash
    (255, 238, 176),   # 15 spark
    (110, 150, 200),   # 16 water
    (255, 120, 40),    # 17 lava
], dtype=np.uint8)

_DEFAULT_COLOR = np.array([0, 2, 5, 8, 9, 11, 13, 14, 15], np.int32)

# kinds drawn additively (they emit light rather than block it)
_ADDITIVE = np.array([0, 0, 1, 0, 0, 0, 1, 0, 1], bool)
# kinds drawn as motion streaks instead of dots
_STREAK = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1], bool)

_ALPHA_LEVELS = 6
_SIZE_LEVELS = 7

# Per-kind spawn defaults used when emit() is called without x/y/vx/vy, so the
# renderer can say ``emit("rain", 40, wind=ev.wind)`` and get a full curtain of
# rain instead of a column at screen centre. Each entry is (x, y, vx, vy) as a
# scalar or an inclusive (lo, hi) range.
#
# The x entries are VIEW-RELATIVE and stay expressed in RENDER_W: they are
# offsets from the left edge of the frame, and emit() adds ``view_x`` to turn
# them into the world coordinates the pool actually stores. Scaling them to
# WORLD_W is the reflex and is wrong - it emits four times the weather and
# shows you a quarter of it.
_SPAWN_DEFAULTS: tuple[tuple[Any, Any, Any, Any], ...] = (
    ((-60.0, RENDER_W + 60.0), (-40.0, -4.0), (-30.0, 30.0), (520.0, 760.0)),   # rain
    ((-40.0, RENDER_W + 40.0), (-30.0, -2.0), (-12.0, 12.0), (26.0, 62.0)),     # snow
    ((0.0, float(RENDER_W)), (RENDER_H * 0.7, RENDER_H * 0.8),
     (-22.0, 22.0), (-95.0, -40.0)),                                            # ember
    ((0.0, float(RENDER_W)), (RENDER_H * 0.6, RENDER_H * 0.75),
     (-14.0, 14.0), (-42.0, -18.0)),                                            # smoke
    ((0.0, float(RENDER_W)), (RENDER_H * 0.7, RENDER_H * 0.8),
     (-60.0, 60.0), (-120.0, -20.0)),                                           # dust
    ((-40.0, RENDER_W + 40.0), (-30.0, -2.0), (-8.0, 8.0), (14.0, 38.0)),       # ash
    ((30.0, RENDER_W - 30.0), (RENDER_H * 0.42, RENDER_H * 0.74), 0.0, 0.0),    # firefly
    ((0.0, float(RENDER_W)), (RENDER_H * 0.7, RENDER_H * 0.8),
     (-32.0, 32.0), (-150.0, -46.0)),                                           # splash
    ((0.0, float(RENDER_W)), (RENDER_H * 0.5, RENDER_H * 0.7),
     (-180.0, 180.0), (-220.0, 40.0)),                                          # spark
)

# how strongly a `wind=` kwarg biases the spawn velocity, per kind
_WIND_SPAWN = np.array([130.0, 34.0, 18.0, 24.0, 10.0, 26.0, 0.0, 6.0, 4.0], np.float32)

# kinds that get split across the distant/near weather layers
_LAYERED = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0], bool)


# Per-kind population ceilings. These sum to well over CAPACITY, so one kind can
# still fill most of the pool - they only stop a long-lived kind emitted at a
# constant rate (fireflies live ~25 s, snow ~15 s) from crowding out everything
# else over an unattended multi-hour run.
_KIND_CAP = np.array([1400, 1200, 500, 320, 400, 800, 140, 200, 300], np.int32)


def _pick(kw: dict[str, Any], name: str, default: Any) -> Any:
    """Keyword value if the caller supplied a real one, else the spawn default."""
    val = kw.get(name)
    return default if val is None else val

_FLAG_SETTLED = 1
_FLAG_BACK = 2                   # belongs to the distant weather layer

# fraction of weather particles assigned to the distant ("back") layer
_BACK_SHARE = 0.42
_BACK_DIM = 0.55                 # brightness multiplier for distant particles

_SPRITES: dict[tuple[int, int, int, int], pygame.Surface] = {}


def _add_radius(size_level: int) -> int:
    """Glow radius for an additive particle.

    Deliberately even and below 48 px so it survives ``radial_light_surface``'s
    radius bucketing unchanged - that keeps the vectorised blit offsets exact.
    """
    return max(4, (((size_level + 1) * 3) // 2) * 2)


def _dot_sprite(color_idx: int, size_level: int, alpha_level: int,
                additive: int) -> pygame.Surface:
    """Cached particle sprite: soft disc (alpha) or radial glow (additive)."""
    key = (color_idx, size_level, alpha_level, additive)
    hit = _SPRITES.get(key)
    if hit is not None:
        return hit

    col = tuple(int(c) for c in PALETTE[color_idx % len(PALETTE)])
    a = (alpha_level + 1) / _ALPHA_LEVELS
    r = size_level + 1
    if additive:
        spr = radial_light_surface(_add_radius(size_level), col, a * 0.9, falloff=1.7)
    else:
        size = r * 2 + 1
        spr = pygame.Surface((size, size), pygame.SRCALPHA)
        alpha = int(clamp(255 * a, 0, 255))
        if r <= 1:
            spr.fill((*col, alpha))
        else:
            pygame.draw.circle(spr, (*col, max(1, alpha // 3)), (r, r), r)
            pygame.draw.circle(spr, (*col, alpha), (r, r), max(1, r - 1))
    if len(_SPRITES) > 900:
        _SPRITES.clear()
    _SPRITES[key] = spr
    return spr


# half-extent of an additive sprite per size level, so blits land centred
_ADD_OFFSETS = np.array([_add_radius(i) for i in range(_SIZE_LEVELS)], np.int32)


class ParticleSystem:
    """Fixed-capacity, numpy-backed particle pool.

    Typical use per frame::

        ps.view_x = cam.x
        ps.emit("rain", 40, x=(cam.x, cam.x + RENDER_W), y=-8, vy=(520, 700))
        ps.update(dt, terrain, events, cam=cam)
        ps.draw(surface, cam=cam)

    ``cam`` is required on both calls and has no default. Particle x is WORLD
    space and the surface is one 1600 px frame of a 6400 px world, so a call
    that omitted it would emit into, and draw, the leftmost quarter of the map
    wherever the camera actually was - a plausible-looking empty sky rather than
    an error. See camera._IdentityCamera for why that class of default was
    removed everywhere.

    Nothing here touches simulation state. ``update`` reads ``terrain`` (for
    ``ground_y`` / ``height``) and ``events`` (for ``wind``) and mutates only
    its own arrays.
    """

    __slots__ = (
        "capacity", "n", "t", "x", "y", "vx", "vy", "life", "life0",
        "size", "kind", "color_idx", "phase", "flags", "_rng", "_ground",
        "_ground_key", "view_x",
    )

    def __init__(self, capacity: int = CAPACITY, seed: int = 7) -> None:
        cap = max(16, int(capacity))
        self.capacity: int = cap
        self.n: int = 0
        self.t: float = 0.0
        #: World x of the left edge of the frame - the window weather is seeded
        #: into and culled against. A plain attribute rather than a `cam`
        #: parameter on emit() because emit() is called from a dozen places
        #: with a dozen signatures and its x is, and stays, WORLD space; only
        #: the *defaults* need to know where the camera is. update() keeps it
        #: in step, and the renderer sets it before its emit pass because that
        #: pass runs first.
        self.view_x: float = 0.0
        self.x = np.zeros(cap, np.float32)
        self.y = np.zeros(cap, np.float32)
        self.vx = np.zeros(cap, np.float32)
        self.vy = np.zeros(cap, np.float32)
        self.life = np.zeros(cap, np.float32)
        self.life0 = np.ones(cap, np.float32)
        self.size = np.ones(cap, np.float32)
        self.kind = np.zeros(cap, np.uint8)
        self.color_idx = np.zeros(cap, np.uint8)
        self.phase = np.zeros(cap, np.float32)
        self.flags = np.zeros(cap, np.uint8)
        self._rng = np.random.default_rng(seed)
        self._ground: np.ndarray | None = None
        self._ground_key: Any = None

    # ------------------------------------------------------------ helpers --

    def __len__(self) -> int:
        return self.n

    @property
    def count(self) -> int:
        """Number of live particles."""
        return self.n

    def clear(self) -> None:
        """Drop every particle."""
        self.n = 0

    def _spread(self, value: Any, count: int, default: float) -> np.ndarray:
        """Broadcast a scalar / ``(lo, hi)`` range / sequence into ``count`` floats."""
        if value is None:
            return np.full(count, float(default), np.float32)
        if isinstance(value, (int, float)):
            return np.full(count, float(value), np.float32)
        if isinstance(value, tuple) and len(value) == 2 and all(
            isinstance(v, (int, float)) for v in value
        ):
            lo, hi = float(value[0]), float(value[1])
            return self._rng.uniform(lo, hi, count).astype(np.float32)
        try:
            arr = np.asarray(value, dtype=np.float32).ravel()
        except Exception:
            return np.full(count, float(default), np.float32)
        if arr.size == 0:
            return np.full(count, float(default), np.float32)
        if arr.size >= count:
            return arr[:count]
        return np.resize(arr, count).astype(np.float32)

    def _make_room(self, want: int) -> tuple[int, int]:
        """Ensure *want* slots; recycles the oldest particles when full."""
        want = min(int(want), self.capacity)
        free = self.capacity - self.n
        if want > free:
            drop = want - free
            keep = self.n - drop
            if keep > 0:
                for arr in (self.x, self.y, self.vx, self.vy, self.life, self.life0,
                            self.size, self.kind, self.color_idx, self.phase, self.flags):
                    arr[:keep] = arr[drop:self.n]
            self.n = max(0, keep)
        return (self.n, self.n + want)

    # ------------------------------------------------------------- emitting --

    def emit(self, kind: str | int, n: int, **kw: Any) -> int:
        """Spawn *n* particles of *kind*; returns how many were actually created.

        Accepted keywords (each a scalar, a ``(lo, hi)`` range, or a sequence):
        ``x``, ``y``, ``vx``, ``vy``, ``speed``, ``angle``, ``life``, ``size``,
        ``color`` (palette index or ``(r, g, b)``), plus ``wind`` (a signed -1..1
        bias applied to the spawn velocity) and ``back`` (force the near/distant
        layer). Unknown keywords are ignored, so callers can pass extra context
        safely.

        Weather kinds have sensible spawn defaults: omitting ``x``/``y`` for
        rain, snow or ash seeds them across the top edge, and fireflies seed
        across the lower half. Never raises.
        """
        try:
            kid = KIND_IDS.get(kind, -1) if isinstance(kind, str) else int(kind)
            if kid < 0 or kid >= KIND_COUNT:
                return 0
            count = int(n)
            if count <= 0:
                return 0
            cap = int(_KIND_CAP[kid])
            if cap > 0 and self.n > 0:
                live = int(np.count_nonzero(self.kind[:self.n] == kid))
                if live >= cap:
                    return 0
                count = min(count, cap - live)
            a, b = self._make_room(count)
            count = b - a
            if count <= 0:
                return 0
            sl = slice(a, b)

            wind = 0.0
            if "wind" in kw:
                try:
                    w = float(kw["wind"])
                    wind = clamp(w, -3.0, 3.0)
                except (TypeError, ValueError):
                    wind = 0.0

            dx, dy, dvx, dvy = _SPAWN_DEFAULTS[kid]
            # x stays WORLD space whoever supplies it. A caller that names an x
            # is naming a thing in the world - a firepit, a burning tree, a
            # lava front - and gets it through untouched. A caller that omits
            # one wanted "across the frame", which is now a window somewhere in
            # a 6400 px world rather than the whole map, so the view-relative
            # default is slid onto the camera here.
            xk = kw.get("x")
            if xk is None:
                self.x[sl] = self._spread(dx, count, 0.0) + np.float32(self.view_x)
            else:
                self.x[sl] = self._spread(xk, count, 0.0)
            self.y[sl] = self._spread(_pick(kw, "y", dy), count, 0.0)
            vx = self._spread(_pick(kw, "vx", dvx), count, 0.0)
            vy = self._spread(_pick(kw, "vy", dvy), count, 0.0)
            if wind:
                vx = vx + wind * _WIND_SPAWN[kid]
            if "speed" in kw:
                speed = self._spread(kw.get("speed"), count, 0.0)
                ang = self._spread(kw.get("angle"), count, 0.0)
                if "angle" not in kw:
                    ang = self._rng.uniform(0.0, 2.0 * np.pi, count).astype(np.float32)
                vx = vx + np.cos(ang) * speed
                vy = vy + np.sin(ang) * speed
            self.vx[sl] = vx
            self.vy[sl] = vy

            life = self._spread(kw.get("life"), count, _DEFAULT_LIFE[kid])
            life = np.maximum(life, 0.05)
            self.life[sl] = life
            self.life0[sl] = life
            self.size[sl] = np.clip(
                self._spread(kw.get("size"), count, _DEFAULT_SIZE[kid]), 0.4, 12.0)

            color = kw.get("color")
            if color is None:
                cidx = np.full(count, int(_DEFAULT_COLOR[kid]), np.int32)
            elif isinstance(color, (int, np.integer)):
                cidx = np.full(count, int(color) % len(PALETTE), np.int32)
            elif isinstance(color, (tuple, list)) and len(color) == 3:
                cidx = np.full(count, _nearest_palette(color), np.int32)
            else:
                cidx = np.asarray(self._spread(color, count, _DEFAULT_COLOR[kid]),
                                  dtype=np.int32) % len(PALETTE)
            self.color_idx[sl] = cidx.astype(np.uint8)

            self.kind[sl] = kid
            self.phase[sl] = self._rng.uniform(0.0, 6.2831853, count).astype(np.float32)

            back = kw.get("back")
            if back is None:
                depth = (self._rng.random(count) < _BACK_SHARE) if _LAYERED[kid] \
                    else np.zeros(count, bool)
            else:
                depth = np.full(count, bool(back))
            self.flags[sl] = np.where(depth, _FLAG_BACK, 0).astype(np.uint8)

            self.n = b
            return count
        except Exception:
            return 0

    # ------------------------------------------------------------ terrain --

    def _ground_heights(self, terrain: Any, xs: np.ndarray) -> np.ndarray:
        """Vectorised ground y for each x. Works off ``terrain.surface()`` when
        present, otherwise samples ``ground_y`` at 96 points and interpolates.

        WORLD_W wide and indexed by world x, not by screen x. That is 25 KB for
        the whole map, which is cheaper than any windowed alternative and means
        a settled snowflake keeps sitting on the hill it landed on when the
        camera moves - rebuilding a view-sized profile every time cam.x changed
        would re-key this cache on every frame the camera is easing.
        """
        if terrain is None:
            return np.full(xs.shape, RENDER_H * 0.78, np.float32)
        prof = self._ground
        # Keyed on the terrain's epoch as well as its identity. A bridge deck
        # gets stamped into the same Terrain object mid-session, so identity
        # alone would pin the profile to whatever the world looked like on the
        # first frame - and rain that keeps falling through a finished deck is
        # exactly what that stale cache looks like on screen.
        key = (id(terrain), int(getattr(terrain, "epoch", 0)))
        if prof is None or self._ground_key != key or prof.shape[0] != WORLD_W:
            prof = None
            # surface(), not height: a finished bridge or ladder is an overlay
            # on top of the raw heightmap, and weather has to land on the deck
            # the walkers are standing on rather than on the chasm floor under
            # it. surface() hands back `height` itself when nothing is stamped,
            # so the common case costs an identity check and no array work.
            h = None
            surf_fn = getattr(terrain, "surface", None)
            if callable(surf_fn):
                try:
                    h = surf_fn()
                except Exception:
                    h = None
            if h is None:
                h = getattr(terrain, "height", None)
            if isinstance(h, np.ndarray) and h.ndim == 1 and h.size >= 2:
                if h.size == WORLD_W:
                    prof = h.astype(np.float32, copy=False)
                else:
                    # A heightmap of some other width gets stretched across the
                    # world rather than left short. This is the path an old
                    # 1600-wide save takes if it ever reaches here before
                    # Terrain.from_dict has regenerated it: a stretched map is
                    # wrong, but weather landing on the right *shape* four
                    # times too wide reads far better than every drop falling
                    # through the ground beyond x=1600.
                    src = np.linspace(0.0, WORLD_W - 1.0, h.size, dtype=np.float32)
                    dst = np.arange(WORLD_W, dtype=np.float32)
                    prof = np.interp(dst, src, h.astype(np.float32)).astype(np.float32)
            else:
                fn = getattr(terrain, "ground_y", None)
                if callable(fn):
                    # 96 samples spanned 1600 px; over WORLD_W that is one
                    # every 67 px, which turns a cliff into a ramp. Kept
                    # proportional to the map instead - it is a one-off cost
                    # paid on a cache miss, not per frame.
                    n_s = int(96 * max(1, WORLD_W // RENDER_W))
                    sx = np.linspace(0.0, WORLD_W - 1.0, n_s, dtype=np.float32)
                    try:
                        sy = np.array([float(fn(float(v))) for v in sx], np.float32)
                        prof = np.interp(np.arange(WORLD_W, dtype=np.float32),
                                         sx, sy).astype(np.float32)
                    except Exception:
                        prof = None
            if prof is None:
                prof = np.full(WORLD_W, RENDER_H * 0.78, np.float32)
            self._ground = prof
            self._ground_key = key
        idx = np.clip(xs.astype(np.int32), 0, prof.shape[0] - 1)
        return prof[idx]

    def invalidate_terrain(self) -> None:
        """Force the cached ground profile to be re-read (after a deformation)."""
        self._ground = None
        self._ground_key = None

    # ------------------------------------------------------------- update --

    def update(self, dt: float, terrain: Any = None, events: Any = None,
               *, cam: Camera) -> None:
        """Advance every particle. Never raises.

        *cam* is where the frame is, and it does exactly two things here: it
        sets ``view_x`` (so the next emit pass seeds into the right window) and
        it moves the cull. Physics is untouched by it - a particle's x is world
        space and stays so.

        Required, no IDENTITY default: an omission would silently pin ``view_x``
        to 0 and cull five sixths of the map's weather out of existence, which
        looks like a clear day rather than like a bug. Pass ``cam=IDENTITY``
        explicitly on a harness where world really is frame.
        """
        try:
            self.view_x = float(cam.x)
            step = clamp(float(dt), 0.0, 0.1)
            if step <= 0.0 or self.n <= 0:
                self.t += max(0.0, float(dt) if np.isfinite(dt) else 0.0)
                return
            self.t += step
            n = self.n
            sl = slice(0, n)

            kind = self.kind[sl].astype(np.int32)
            x = self.x[sl]
            y = self.y[sl]
            vx = self.vx[sl]
            vy = self.vy[sl]
            settled = (self.flags[sl] & _FLAG_SETTLED) != 0
            moving = ~settled

            wind = clamp(attr_num(events, "wind", "wind_x", "wind_speed",
                                  default=0.0), -3.0, 3.0)

            self.life[sl] -= step
            self.phase[sl] += step * (1.4 + 0.6 * (kind == K_FIREFLY))

            grav = _GRAVITY[kind]
            drag = _DRAG[kind]
            windf = _WINDF[kind]
            sway = _SWAY[kind]

            # integrate velocity: gravity, wind coupling, linear drag, sway
            vy_new = vy + grav * step
            vx_new = vx + (wind * windf - vx * drag) * step
            vy_new -= vy_new * (drag * 0.35) * step
            has_sway = sway > 0.0
            if has_sway.any():
                osc = np.sin(self.phase[sl] * 2.3) * sway
                vx_new = np.where(has_sway, vx_new + osc * step * 3.0, vx_new)

            # fireflies steer rather than fall
            ff = kind == K_FIREFLY
            if ff.any():
                ph = self.phase[sl]
                vx_new = np.where(ff, np.cos(ph * 0.9) * 26.0 + np.sin(ph * 2.7) * 9.0, vx_new)
                vy_new = np.where(ff, np.sin(ph * 0.7) * 18.0 + np.cos(ph * 3.1) * 7.0, vy_new)

            # smoke slows as it expands
            sm = kind == K_SMOKE
            if sm.any():
                self.size[sl] = np.where(sm, np.minimum(self.size[sl] + step * 5.5, 14.0),
                                         self.size[sl])

            vx_new = np.where(moving, vx_new, 0.0)
            vy_new = np.where(moving, vy_new, 0.0)
            self.vx[sl] = vx_new
            self.vy[sl] = vy_new
            self.x[sl] = x + vx_new * step
            self.y[sl] = y + vy_new * step

            # ------------------------------------------------ ground contact --
            collide = _COLLIDES[kind] & moving
            splash_x: np.ndarray | None = None
            splash_y: np.ndarray | None = None
            if collide.any():
                gy = self._ground_heights(terrain, self.x[sl])
                hit = collide & (self.y[sl] >= gy)
                if hit.any():
                    rain_hit = hit & (kind == K_RAIN)
                    if rain_hit.any():
                        splash_x = self.x[sl][rain_hit].copy()
                        splash_y = gy[rain_hit].copy()
                        self.life[sl] = np.where(rain_hit, -1.0, self.life[sl])
                    # snow and ash settle for a moment, everything else dies
                    rest = hit & ((kind == K_SNOW) | (kind == K_ASH))
                    if rest.any():
                        self.y[sl] = np.where(rest, gy - 1.0, self.y[sl])
                        self.flags[sl] = np.where(rest, self.flags[sl] | _FLAG_SETTLED,
                                                  self.flags[sl])
                        self.life[sl] = np.where(rest, np.minimum(self.life[sl], 5.0),
                                                 self.life[sl])
                    gone = hit & ~rest & ~rain_hit
                    if gone.any():
                        self.life[sl] = np.where(gone, -1.0, self.life[sl])

            # fireflies keep clear of the ground
            if ff.any():
                gy = self._ground_heights(terrain, self.x[sl])
                self.y[sl] = np.where(ff, np.minimum(self.y[sl], gy - 6.0), self.y[sl])

            # --------------------------------------------------- cull & pack --
            # The cull is VIEW space, not world space: a particle that has
            # scrolled out of frame is gone even though it is still well inside
            # the map. That is what keeps the pool spent on what is on screen -
            # culling at 0/WORLD_W instead would let a curtain of rain sit in
            # the far east of the map holding slots the visible weather needs.
            vlo = self.view_x - 60.0
            vhi = self.view_x + RENDER_W + 60.0
            alive = (self.life[sl] > 0.0)
            alive &= (self.x[sl] > vlo) & (self.x[sl] < vhi)
            alive &= (self.y[sl] > -200.0) & (self.y[sl] < RENDER_H + 60.0)
            keep = int(np.count_nonzero(alive))
            if keep != n:
                if keep == 0:
                    self.n = 0
                else:
                    idx = np.flatnonzero(alive)
                    for arr in (self.x, self.y, self.vx, self.vy, self.life,
                                self.life0, self.size, self.kind, self.color_idx,
                                self.phase, self.flags):
                        arr[:keep] = arr[:n][idx]
                    self.n = keep

            if splash_x is not None and splash_x.size:
                m = min(int(splash_x.size), 60)
                self.emit(K_SPLASH, m, x=splash_x[:m], y=splash_y[:m] - 2.0,
                          vx=(-32.0, 32.0), vy=(-150.0, -46.0),
                          life=(0.12, 0.30), size=1.0)
        except Exception:
            return

    # --------------------------------------------------------------- draw --

    def draw(self, surf: pygame.Surface, layer: str | None = None,
             *, cam: Camera) -> None:
        """Render particles. Two batched blit calls plus streak lines.

        *layer* selects a depth band so the renderer can straddle the terrain:
        ``"back"`` draws only distant weather (pipeline layer 3, behind the
        terrain), ``"front"`` draws everything else (layers 8-9), and ``None``
        draws the lot. Distant particles are dimmed for aerial perspective.

        The world -> screen conversion is the single ``- cam.x`` below, applied
        to the whole array at once. Everything after it is screen space, which
        is why ``_draw_streaks`` needs no camera of its own. Deliberately NOT
        ``cam.sx()`` per particle: this runs over up to CAPACITY elements every
        frame and a method call each would be the most expensive line in the
        renderer.

        Read-only; never raises.
        """
        try:
            n = self.n
            if n <= 0:
                return
            sl = slice(0, n)
            kind = self.kind[sl].astype(np.int32)
            x = self.x[sl] - np.float32(cam.x)
            y = self.y[sl]
            life = self.life[sl]
            life0 = np.maximum(self.life0[sl], 1e-3)
            ratio = np.clip(life / life0, 0.0, 1.0)

            is_back = (self.flags[sl] & _FLAG_BACK) != 0
            if layer == "back":
                band = is_back
            elif layer == "front":
                band = ~is_back
            else:
                band = None
            if band is not None and not band.any():
                return

            # ---------------------------------------------- colour by age --
            cidx = self.color_idx[sl].astype(np.int32)
            em = kind == K_EMBER
            if em.any():
                cidx = np.where(em, np.where(ratio > 0.62, 4, np.where(ratio > 0.30, 5, 6)),
                                cidx)
            sm = kind == K_SMOKE
            if sm.any():
                cidx = np.where(sm, np.where(ratio > 0.55, 8, 7), cidx)

            # ----------------------------------------------------- alpha --
            fade = np.where(sm, ratio * 0.75,
                            np.where(em | (kind == K_SPARK), np.sqrt(ratio),
                                     np.clip(ratio * 3.0, 0.0, 1.0)))
            ffm = kind == K_FIREFLY
            if ffm.any():
                pulse = 0.45 + 0.55 * np.abs(np.sin(self.phase[sl] * 1.9))
                fade = np.where(ffm, np.clip(ratio * 4.0, 0.0, 1.0) * pulse, fade)

            # distant particles read as haze, not as objects
            fade = np.where(is_back, fade * _BACK_DIM, fade)

            additive = _ADDITIVE[kind]
            streak = _STREAK[kind]
            if band is not None:
                streak = streak & band

            # ---------------------------------------------------- streaks --
            if streak.any():
                self._draw_streaks(surf, streak, x, y, cidx, fade)

            dots = ~_STREAK[kind]
            if band is not None:
                dots = dots & band
            if not dots.any():
                return

            size_lvl = np.clip(self.size[sl].astype(np.int32) - 1, 0, _SIZE_LEVELS - 1)
            size_lvl = np.where(is_back, np.maximum(size_lvl - 1, 0), size_lvl)
            # level L renders at alpha (L+1)/_ALPHA_LEVELS, so the -1 makes the
            # dimmest bucket drop out entirely instead of popping at 1/6 alpha
            alpha_lvl = np.clip((fade * _ALPHA_LEVELS).astype(np.int32) - 1, -1,
                                _ALPHA_LEVELS - 1)
            vis = dots & (alpha_lvl >= 0)
            if not vis.any():
                return

            add = additive[vis]
            keys = ((cidx[vis] * _SIZE_LEVELS + size_lvl[vis]) * _ALPHA_LEVELS
                    + alpha_lvl[vis]) * 2 + add.astype(np.int32)
            slv = size_lvl[vis]
            off = np.where(add, _ADD_OFFSETS[slv], slv + 1)
            pos = np.stack([x[vis] - off, y[vis] - off], axis=1).astype(np.int32)

            order = np.argsort(keys, kind="stable")
            ks = keys[order]
            pts = pos[order].tolist()
            uniq, starts = np.unique(ks, return_index=True)
            ukeys = uniq.tolist()
            ustarts = starts.tolist()
            total = len(pts)

            normal_seq: list[Any] = []
            add_seq: list[Any] = []
            for i, k in enumerate(ukeys):
                a0 = ustarts[i]
                a1 = ustarts[i + 1] if i + 1 < len(ustarts) else total
                is_add = k & 1
                rest = k >> 1
                alvl = rest % _ALPHA_LEVELS
                rest //= _ALPHA_LEVELS
                slvl = rest % _SIZE_LEVELS
                col = rest // _SIZE_LEVELS
                spr = _dot_sprite(col, slvl, alvl, is_add)
                target = add_seq if is_add else normal_seq
                target.extend(zip(_repeat(spr), pts[a0:a1]))

            blit_batch(surf, normal_seq)
            blit_batch(surf, add_seq, pygame.BLEND_RGB_ADD)
        except Exception:
            return

    def _draw_streaks(self, surf: pygame.Surface, mask: np.ndarray, x: np.ndarray,
                      y: np.ndarray, cidx: np.ndarray, fade: np.ndarray) -> None:
        """Rain and sparks as short motion-blur lines scaled by velocity."""
        sl = slice(0, self.n)
        vx = self.vx[sl][mask]
        vy = self.vy[sl][mask]
        sx = x[mask]
        sy = y[mask]
        f = np.clip(fade[mask], 0.0, 1.0)
        cols = PALETTE[cidx[mask] % len(PALETTE)].astype(np.float32)
        # colours are pre-multiplied by alpha: streaks sit on a dark sky, so
        # scaling towards black reads the same as blending and costs nothing
        cols = np.clip(cols * f[:, None], 0, 255).astype(np.int32)
        k = 0.028
        ex = sx - vx * k
        ey = sy - vy * k

        starts = np.stack([sx, sy], axis=1).astype(np.int32).tolist()
        ends = np.stack([ex, ey], axis=1).astype(np.int32).tolist()
        colors = cols.tolist()
        line = pygame.draw.line
        for i in range(len(starts)):
            c = colors[i]
            if c[0] < 6 and c[1] < 6 and c[2] < 6:
                continue
            line(surf, c, starts[i], ends[i], 1)

    # ---------------------------------------------------------- persistence --

    def to_dict(self) -> dict[str, Any]:
        """Serialise the system.

        Live particles are deliberately *not* persisted: they are cosmetic,
        sub-second, and would bloat the save by megabytes. Only the pool
        configuration survives a restart; the weather refills the pool within a
        frame or two of loading.
        """
        return {"capacity": int(self.capacity), "t": float(self.t)}

    @classmethod
    def from_dict(cls, d: Any) -> "ParticleSystem":
        """Rebuild from :meth:`to_dict`. Missing keys take defaults, unknown
        keys are ignored, and a malformed payload yields a default system."""
        cap = CAPACITY
        t = 0.0
        if isinstance(d, dict):
            try:
                cap = int(d.get("capacity", CAPACITY))
            except (TypeError, ValueError):
                cap = CAPACITY
            try:
                t = float(d.get("t", 0.0))
            except (TypeError, ValueError):
                t = 0.0
        ps = cls(capacity=cap if 16 <= cap <= 20000 else CAPACITY)
        ps.t = t if np.isfinite(t) else 0.0
        return ps


# per-kind defaults used when emit() is not given life/size
_DEFAULT_LIFE = np.array([2.4, 14.0, 1.5, 3.2, 1.1, 20.0, 26.0, 0.30, 0.55], np.float32)
_DEFAULT_SIZE = np.array([1.0, 1.6, 1.4, 4.0, 1.6, 1.4, 1.6, 1.0, 1.0], np.float32)


def _nearest_palette(color: Sequence[int]) -> int:
    """Index of the palette entry closest to an ``(r, g, b)`` request."""
    try:
        c = np.asarray(color, dtype=np.float32)[:3]
    except Exception:
        return 0
    d = np.sum((PALETTE.astype(np.float32) - c[None, :]) ** 2, axis=1)
    return int(np.argmin(d))


# --------------------------------------------------------------------------
# convenience emitters used by the renderer for the standard scenes
# --------------------------------------------------------------------------


# ``width`` stays RENDER_W in all three: it is how wide the *curtain* is, which
# is a property of the frame and not of the map. ``origin`` is the world x of
# the frame's left edge - pass ``cam.x`` (or leave it at 0.0, which is the
# harness case where the world and the frame are still the same thing).


def emit_rainfall(ps: ParticleSystem, n: int, wind: float = 0.0,
                  width: int = RENDER_W, origin: float = 0.0) -> int:
    """A frame's worth of rain entering from the top edge."""
    return ps.emit(K_RAIN, n, x=(origin - 60.0, origin + float(width) + 60.0),
                   y=(-40.0, -4.0),
                   vx=(wind * 120.0 - 30.0, wind * 120.0 + 30.0), vy=(520.0, 760.0),
                   size=1.0, life=(1.4, 2.6))


def emit_snowfall(ps: ParticleSystem, n: int, wind: float = 0.0,
                  width: int = RENDER_W, origin: float = 0.0) -> int:
    """A frame's worth of snow."""
    return ps.emit(K_SNOW, n, x=(origin - 40.0, origin + float(width) + 40.0),
                   y=(-30.0, -2.0),
                   vx=(wind * 30.0 - 12.0, wind * 30.0 + 12.0), vy=(26.0, 62.0),
                   size=(1.0, 2.8), life=(9.0, 18.0))


def emit_ashfall(ps: ParticleSystem, n: int, wind: float = 0.0,
                 width: int = RENDER_W, origin: float = 0.0) -> int:
    """Slow grey volcanic ash."""
    return ps.emit(K_ASH, n, x=(origin - 40.0, origin + float(width) + 40.0),
                   y=(-30.0, -2.0),
                   vx=(wind * 24.0 - 8.0, wind * 24.0 + 8.0), vy=(14.0, 38.0),
                   size=(1.0, 2.2), life=(12.0, 24.0))


def emit_fire(ps: ParticleSystem, x: float, y: float, strength: float = 1.0) -> None:
    """Embers plus smoke from a fire at ``(x, y)``."""
    s = clamp(strength, 0.0, 3.0)
    ps.emit(K_EMBER, max(1, int(4 * s)), x=(x - 9.0 * s, x + 9.0 * s), y=(y - 5.0, y),
            vx=(-38.0, 38.0), vy=(-105.0 * s, -40.0 * s), size=(1.0, 3.0),
            life=(0.7, 1.9))
    ps.emit(K_SMOKE, max(1, int(1 * s)), x=(x - 7.0, x + 7.0), y=(y - 12.0, y - 2.0),
            vx=(-16.0, 16.0), vy=(-44.0, -18.0), size=(3.0, 5.0), life=(2.0, 4.0))


def emit_impact_dust(ps: ParticleSystem, x: float, y: float, power: float = 1.0) -> None:
    """Dust burst for mudslides, quakes, meteor impacts and falls."""
    p = clamp(power, 0.1, 4.0)
    ps.emit(K_DUST, int(10 * p), x=(x - 8.0 * p, x + 8.0 * p), y=(y - 6.0, y),
            speed=(40.0 * p, 150.0 * p), angle=(-3.14159, 0.0),
            size=(1.0, 3.0), life=(0.5, 1.4))


def emit_sparks(ps: ParticleSystem, x: float, y: float, n: int = 12) -> None:
    """Bright short-lived sparks (lightning strike, tool impact)."""
    ps.emit(K_SPARK, n, x=x, y=y, speed=(90.0, 320.0), size=1.0, life=(0.18, 0.5))


def emit_fireflies(ps: ParticleSystem, n: int, x_range: tuple[float, float],
                   y_range: tuple[float, float]) -> int:
    """Wandering fireflies - clear nights only, gated by the caller."""
    return ps.emit(K_FIREFLY, n, x=x_range, y=y_range, size=(1.5, 3.0),
                   life=(14.0, 34.0))


def clear_caches() -> None:
    """Drop every cached particle sprite."""
    _SPRITES.clear()


def cache_stats() -> dict[str, int]:
    """Sprite cache occupancy, for the diagnostics overlay."""
    return {"sprites": len(_SPRITES)}


if __name__ == "__main__":  # pragma: no cover - smoke test
    import os
    import time

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    class _Terrain:
        # WORLD_W wide, like the real thing: the profile is indexed by world x.
        height = (np.full(WORLD_W, RENDER_H * 0.74, np.float32)
                  + 40.0 * np.sin(np.arange(WORLD_W) / 180.0).astype(np.float32))

        def ground_y(self, x: float) -> float:
            return float(self.height[int(clamp(x, 0, WORLD_W - 1))])

    class _Events:
        wind = 0.7

    ps = ParticleSystem()
    terrain, events = _Terrain(), _Events()
    # ...but the surface is RENDER_W, because that is the frame. The whole
    # point of this harness is that those two numbers now differ.
    surf = pygame.Surface((RENDER_W, RENDER_H))
    cam = Camera()

    frames = 300
    t0 = time.perf_counter()
    for i in range(frames):
        # Park the camera deep in the map for the second half of the run: if
        # the view-relative spawn or the view-space cull were wrong, the pool
        # would empty out the moment cam.x left zero.
        cam.nudge(14.0) if i > 120 else None
        ps.view_x = cam.x
        emit_rainfall(ps, 26, wind=0.7, origin=cam.x)
        emit_snowfall(ps, 4, wind=0.7, origin=cam.x)
        emit_ashfall(ps, 3, origin=cam.x)
        emit_fire(ps, cam.x + 420.0, RENDER_H * 0.72, 1.4)
        if i % 30 == 0:
            emit_impact_dust(ps, cam.x + 800.0, RENDER_H * 0.72, 1.5)
            emit_sparks(ps, cam.x + 640.0, RENDER_H * 0.5, 14)
        if i == 0:
            emit_fireflies(ps, 40, (100.0, 1100.0), (400.0, 560.0))
        ps.update(1.0 / 60.0, terrain, events, cam=cam)
        surf.fill((10, 12, 20))
        ps.draw(surf, cam=cam)
    dt = (time.perf_counter() - t0) / frames
    print(f"particles: {len(ps)} live, {dt * 1000:.2f} ms/frame (update+draw), "
          f"caches={cache_stats()}")
    if len(ps):
        wx = ps.x[:len(ps)]
        print(f"cam.x={cam.x:.0f}  world x span {wx.min():.0f}..{wx.max():.0f} "
              f"-> screen {wx.min() - cam.x:.0f}..{wx.max() - cam.x:.0f} "
              f"(must sit inside -60..{RENDER_W + 60})")

    round_trip = ParticleSystem.from_dict(ps.to_dict())
    print("round trip capacity:", round_trip.capacity,
          "| defensive:", ParticleSystem.from_dict(None).capacity,
          ParticleSystem.from_dict({"bogus": 1}).capacity)

    try:
        from ..paths import CAPTURE_DIR, ensure_dirs
        ensure_dirs()
        pygame.image.save(surf, str(CAPTURE_DIR / "particles.png"))
        print("wrote particles.png to", CAPTURE_DIR)
    except Exception as exc:
        print("capture skipped:", exc)
