"""Terrain: a per-column heightmap plus a per-column material id.

Pure python + numpy.  **No pygame** - this module has to import and run
headless so the simulation can be tested without a display.

Coordinate system (see docs/ARCHITECTURE.md section 3):

    * world space *is* render space, origin top-left
    * ``x`` runs 0 .. RENDER_W-1, one array column per screen pixel column
    * ``y`` grows **downward**, so a *smaller* height value is *higher* ground

Every mutator is written to fail soft: bad indices are clamped, NaNs are
dropped, and nothing in here raises on a per-frame path.

Interfaces other modules rely on
--------------------------------
``Terrain.chasm``            ``(x0, x1)`` of the cut gap or ``None`` - the
                             bridge buildable (feature 9) needs this.
``Terrain.find_flat_span``   structure placement.
``Terrain.find_basins``      water/pond placement and the flood event.
``Terrain.deform``/``crater``/``add_layer``  mudslide, meteor, blizzard, ashfall.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

import numpy as np

from ..constants import (
    MAT_ASH,
    MAT_DIRT,
    MAT_GRASS,
    MAT_SAND,
    MAT_STONE,
    MAX_SLOPE_CLIMB,
    MAX_SLOPE_WALK,
    RENDER_H,
    RENDER_W,
)

__all__ = [
    "Terrain",
    "STYLES",
    "HEIGHT_MIN",
    "HEIGHT_MAX",
    "HARD_MIN",
    "HARD_MAX",
    "CHASM_FLOOR",
]

# ------------------------------------------------------------------ tuning --

STYLES: tuple[str, ...] = ("hills", "cliffs", "plateau", "chasm", "valley")

#: Generation clamp.  The freshly generated surface always lives in this band
#: (the deliberately cut chasm floor is the one documented exception).
HEIGHT_MIN: float = RENDER_H * 0.35        # 280 - highest ground allowed
HEIGHT_MAX: float = RENDER_H * 0.92        # 736 - lowest ground allowed

#: Runtime clamp.  Craters, digging and snow accumulation may leave the
#: generation band but never the screen.
HARD_MIN: float = RENDER_H * 0.10          # 80
HARD_MAX: float = RENDER_H * 0.985         # 788

CHASM_FLOOR: float = RENDER_H * 0.95       # 760

#: ``slope()`` is a central difference over this many pixels total.
SLOPE_SPAN: float = 5.0
_SLOPE_HALF: float = SLOPE_SPAN * 0.5

#: |slope| at or below this counts as "flat enough to build on".
FLAT_SLOPE: float = 0.28

#: Cosine falloff width used by :meth:`Terrain.deform` with ``blend='smooth'``.
DEFORM_EDGE: int = 24

_BASE_CELLS: dict[str, float] = {
    "hills": 3.0,
    "cliffs": 4.0,
    "plateau": 3.0,
    "chasm": 3.5,
    "valley": 2.5,
}


# ------------------------------------------------------------------ helpers --


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Classic 3t^2-2t^3 ease, input clipped to 0..1."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _value_noise(rng: np.random.Generator, w: int, cells: float) -> np.ndarray:
    """One octave of 1-D value noise in 0..1.

    Random control points every ``w/cells`` pixels, smoothstep-interpolated
    between them.  Deliberately hand rolled - no noise library dependency.
    """
    n = int(round(cells))
    n = max(2, min(n, max(2, w - 1)))
    pts = rng.random(n + 1)
    xs = np.linspace(0.0, float(w - 1), n + 1)
    x = np.arange(w, dtype=np.float64)
    idx = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, n - 1)
    x0 = xs[idx]
    x1 = xs[idx + 1]
    t = _smoothstep((x - x0) / np.maximum(x1 - x0, 1e-9))
    return pts[idx] * (1.0 - t) + pts[idx + 1] * t


def _fbm(
    rng: np.random.Generator,
    w: int,
    octaves: int = 5,
    base_cells: float = 3.0,
    gain: float = 0.5,
    lacunarity: float = 2.0,
) -> np.ndarray:
    """Sum of ``octaves`` value-noise layers, normalised to roughly 0..1."""
    total = np.zeros(w, dtype=np.float64)
    amp = 1.0
    norm = 0.0
    cells = float(base_cells)
    for _ in range(max(1, octaves)):
        total += amp * _value_noise(rng, w, cells)
        norm += amp
        amp *= gain
        cells *= lacunarity
    return total / max(norm, 1e-9)


def _normalise(f: np.ndarray) -> np.ndarray:
    lo = float(np.min(f))
    hi = float(np.max(f))
    if hi - lo < 1e-6:
        return np.full_like(f, 0.5)
    return (f - lo) / (hi - lo)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs of a 1-D bool mask as half-open ``(a, b)``."""
    if mask.size == 0:
        return []
    m = mask.astype(np.int8)
    d = np.diff(m)
    starts = (np.flatnonzero(d == 1) + 1).tolist()
    ends = (np.flatnonzero(d == -1) + 1).tolist()
    if m[0]:
        starts.insert(0, 0)
    if m[-1]:
        ends.append(int(mask.size))
    return list(zip(starts, ends))


def _overlaps(a: tuple[float, float], b: tuple[float, float] | None, pad: float = 0.0) -> bool:
    if b is None:
        return False
    return not (a[1] + pad < b[0] or a[0] - pad > b[1])


def _b64_array(arr: np.ndarray) -> dict:
    return {
        "b64": base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def _unb64_array(d: object, dtype: str, length: int) -> np.ndarray | None:
    """Inverse of :func:`_b64_array`.  Returns ``None`` if anything is off."""
    if not isinstance(d, dict):
        return None
    raw = d.get("b64")
    if not isinstance(raw, str):
        return None
    try:
        buf = base64.b64decode(raw.encode("ascii"), validate=False)
        arr = np.frombuffer(buf, dtype=np.dtype(str(d.get("dtype", dtype))))
    except Exception:
        return None
    if arr.size == 0:
        return None
    arr = arr.astype(dtype, copy=True)
    if arr.size != length:
        # Saved at a different render width - resample rather than give up.
        src = np.linspace(0.0, 1.0, arr.size)
        dst = np.linspace(0.0, 1.0, length)
        if np.issubdtype(np.dtype(dtype), np.integer):
            idx = np.clip(np.round(np.interp(dst, src, np.arange(arr.size))), 0, arr.size - 1)
            arr = arr[idx.astype(np.int64)]
        else:
            arr = np.interp(dst, src, arr.astype(np.float64)).astype(dtype)
    return np.ascontiguousarray(arr)


# -------------------------------------------------------------------- class --


@dataclass
class Terrain:
    """A one-dimensional landscape: ground height and material per column."""

    W: int = RENDER_W
    height: np.ndarray = field(
        default_factory=lambda: np.full(RENDER_W, RENDER_H * 0.72, dtype=np.float32)
    )
    material: np.ndarray = field(
        default_factory=lambda: np.full(RENDER_W, MAT_GRASS, dtype=np.uint8)
    )
    seed: int = 0
    style: str = "hills"
    #: ``(x0, x1)`` of the cut gap for the 'chasm' style, else ``None``.
    chasm: tuple[int, int] | None = None

    # ---------------------------------------------------------------- setup --

    def __post_init__(self) -> None:
        self.W = int(self.W) if int(self.W) > 8 else RENDER_W
        self.height = np.asarray(self.height, dtype=np.float32).reshape(-1).copy()
        self.material = np.asarray(self.material, dtype=np.uint8).reshape(-1).copy()
        if self.height.size != self.W:
            self.height = np.full(self.W, RENDER_H * 0.72, dtype=np.float32)
        if self.material.size != self.W:
            self.material = np.full(self.W, MAT_GRASS, dtype=np.uint8)
        bad = ~np.isfinite(self.height)
        if bad.any():
            self.height[bad] = RENDER_H * 0.72
        np.clip(self.height, HARD_MIN, HARD_MAX, out=self.height)

    # ----------------------------------------------------------- generation --

    @classmethod
    def generate(cls, seed: int, style: str = "hills") -> "Terrain":
        """Build a fresh landscape.

        Guarantees, verified before returning:

        * never flat - 4-5 octaves of value noise drive the base profile;
        * at least one cliff face with ``|slope| > MAX_SLOPE_CLIMB`` (2.6);
        * at least one buildable shelf at least 120 px wide;
        * height clamped to ``[HEIGHT_MIN, HEIGHT_MAX]`` (the chasm floor is
          the single deliberate exception).
        """
        try:
            return cls._generate_impl(int(seed), str(style))
        except Exception:
            return cls._fallback(int(seed), str(style))

    @classmethod
    def _generate_impl(cls, seed: int, style: str) -> "Terrain":
        seed_i = int(seed) & 0xFFFFFFFF
        style = style if style in STYLES else "hills"
        rng = np.random.default_rng(seed_i)
        w = RENDER_W

        f = _normalise(_fbm(rng, w, octaves=5, base_cells=_BASE_CELLS.get(style, 3.0)))
        f = _shape_style(f, style, rng, w)
        h = HEIGHT_MAX - (HEIGHT_MAX - HEIGHT_MIN) * np.clip(f, 0.0, 1.0)

        t = cls(
            W=w,
            height=h.astype(np.float32),
            material=np.full(w, MAT_GRASS, dtype=np.uint8),
            seed=seed_i,
            style=style,
        )
        t._clip_gen()

        if style == "chasm":
            t._cut_chasm(rng)

        shelf = t._carve_shelf(rng)
        t._carve_cliff(rng, avoid=(shelf, t.chasm))
        t._clip_gen()

        # -- guarantee sweep ------------------------------------------------
        for _ in range(3):
            if t._has_cliff():
                break
            t._carve_cliff(rng, avoid=(shelf, t.chasm), force=True)
            t._clip_gen()
        for _ in range(3):
            if t._has_shelf(120):
                break
            shelf = t._carve_shelf(rng)
            t._clip_gen()

        t.retexture()
        return t

    @classmethod
    def _fallback(cls, seed: int, style: str) -> "Terrain":
        """Dumb but valid terrain, used only if generation somehow explodes."""
        w = RENDER_W
        x = np.arange(w, dtype=np.float64)
        h = (
            RENDER_H * 0.70
            - 70.0 * np.sin(x / 190.0 + (seed % 97) * 0.11)
            - 26.0 * np.sin(x / 61.0)
        )
        t = cls(
            W=w,
            height=np.clip(h, HEIGHT_MIN, HEIGHT_MAX).astype(np.float32),
            material=np.full(w, MAT_GRASS, dtype=np.uint8),
            seed=int(seed) & 0xFFFFFFFF,
            style=style if style in STYLES else "hills",
        )
        # A guaranteed shelf and a guaranteed cliff, hand placed.
        t.height[300:460] = t.height[380]
        step = np.arange(w) >= 900
        t.height[step] -= 150.0
        edge = slice(894, 907)
        t.height[edge] = np.linspace(t.height[893], t.height[907], 13)
        np.clip(t.height, HEIGHT_MIN, HEIGHT_MAX, out=t.height)
        t.retexture()
        return t

    def _clip_gen(self) -> None:
        """Clip to the generation band, sparing the chasm floor."""
        lo = np.full(self.W, HEIGHT_MIN, dtype=np.float32)
        hi = np.full(self.W, HEIGHT_MAX, dtype=np.float32)
        if self.chasm is not None:
            c0, c1 = self.chasm
            a = max(0, int(c0) - 14)
            b = min(self.W, int(c1) + 14)
            if b > a:
                hi[a:b] = np.float32(CHASM_FLOOR + 8.0)
        np.clip(self.height, lo, hi, out=self.height)

    def _cut_chasm(self, rng: np.random.Generator) -> None:
        """Cut a 60-120 px gap most of the way down the screen, for a bridge."""
        w = self.W
        gw = int(rng.integers(60, 121))
        cx = int(rng.integers(int(w * 0.28), int(w * 0.72)))
        x0 = int(np.clip(cx - gw // 2, 30, w - 30 - gw))
        x1 = x0 + gw
        wall = 9
        floor = float(CHASM_FLOOR) + float(rng.uniform(-5.0, 5.0))

        a = max(0, x0 - wall)
        b = min(w, x1 + wall)
        idx = np.arange(a, b)
        left = _smoothstep((idx - (x0 - wall)) / float(2 * wall))
        right = 1.0 - _smoothstep((idx - (x1 - wall)) / float(2 * wall))
        mask = np.clip(np.minimum(left, right), 0.0, 1.0)
        wobble = floor + rng.uniform(-2.0, 2.0, size=idx.size)
        cur = self.height[a:b].astype(np.float64)
        self.height[a:b] = (cur * (1.0 - mask) + wobble * mask).astype(np.float32)
        self.chasm = (x0, x1)

    def _carve_shelf(self, rng: np.random.Generator) -> tuple[int, int] | None:
        """Flatten the already-flattest window into a genuinely level shelf.

        Returns the ``(x0, x1)`` of the perfectly flat core (>= 120 px wide).
        """
        w = self.W
        edge = 26
        span = int(rng.integers(190, 246))
        span = int(min(span, max(2 * edge + 130, w // 4)))
        n = w - span + 1
        if n < 2:
            return None

        sl = np.abs(self.column_slope())
        win = np.lib.stride_tricks.sliding_window_view(sl, span)
        score = win.max(axis=1) + 0.25 * win.mean(axis=1)

        bad = np.zeros(n, dtype=bool)
        bad[: min(40, n)] = True
        bad[max(0, n - 40) :] = True
        if self.chasm is not None:
            c0, c1 = self.chasm
            starts = np.arange(n)
            bad |= (starts + span > c0 - 40) & (starts < c1 + 40)
        if not bad.all():
            score = np.where(bad, np.inf, score)
        i0 = int(np.argmin(score))
        if not np.isfinite(score[i0]):
            i0 = max(0, w // 2 - span // 2)

        seg = self.height[i0 : i0 + span].astype(np.float64)
        core = seg[edge : span - edge]
        target = float(np.median(core)) if core.size else float(np.median(seg))
        target = float(np.clip(target, HEIGHT_MIN + 10.0, HEIGHT_MAX - 10.0))

        prof = np.ones(span, dtype=np.float64)
        ramp = _smoothstep((np.arange(edge) + 0.5) / float(edge))
        prof[:edge] = ramp
        prof[span - edge :] = ramp[::-1]
        self.height[i0 : i0 + span] = (seg * (1.0 - prof) + target * prof).astype(np.float32)
        return (i0 + edge, i0 + span - edge)

    def _carve_cliff(
        self,
        rng: np.random.Generator,
        avoid: tuple[tuple[int, int] | None, ...] = (),
        force: bool = False,
    ) -> bool:
        """Cut an escarpment: a steep face, a short terrace, a long relaxation."""
        w = self.W
        for attempt in range(6):
            wid = int(rng.integers(9, 17)) if not force else 7
            terrace = int(rng.integers(30, 90))
            relax = int(rng.integers(120, 210))
            want = float(rng.uniform(130.0, 200.0))
            reach = 2 * wid + terrace + relax

            cands: list[int] = []
            for cx in range(70, max(71, w - 60 - reach), 6):
                box = (float(cx - wid), float(cx + wid + terrace + relax))
                if any(_overlaps(box, a, pad=36.0) for a in avoid):
                    continue
                cands.append(cx)
            if not cands:
                cands = list(range(70, max(71, w - 60 - reach), 6)) or [w // 3]
            cx = int(cands[int(rng.integers(0, len(cands)))])

            here = float(self.height[min(max(cx, 0), w - 1)])
            room_down = HEIGHT_MAX - here
            room_up = here - HEIGHT_MIN
            sign = 1.0 if room_down >= room_up else -1.0
            amount = min(want, (room_down if sign > 0 else room_up) - 8.0)
            if amount < 70.0 and attempt < 5:
                continue
            amount = max(amount, 70.0) * sign

            delta = np.zeros(w, dtype=np.float64)
            a = max(0, cx - wid)
            b = min(w, cx + wid + 1)
            if b > a:
                face = np.arange(a, b)
                delta[a:b] = amount * _smoothstep((face - (cx - wid)) / float(2 * wid))
            c = min(w, b + terrace)
            delta[b:c] = amount
            d = min(w, c + relax)
            if d > c:
                tail = np.arange(c, d)
                delta[c:d] = amount * (1.0 - _smoothstep((tail - c) / float(max(1, relax))))

            self.height += delta.astype(np.float32)
            self._clip_gen()
            if self._has_cliff() or attempt >= 5:
                return True
        return False

    # --------------------------------------------------------- measurements --

    def column_slope(self) -> np.ndarray:
        """Per-column ``dy/dx``, identical to :meth:`slope` at integer x.

        Central difference over ``SLOPE_SPAN`` px, edge-padded so the borders
        do not report phantom cliffs.
        """
        h = self.height.astype(np.float64)
        w = self.W
        hp = np.pad(h, 3, mode="edge")
        ahead = 0.5 * (hp[5 : 5 + w] + hp[6 : 6 + w])      # ground_y(x + 2.5)
        behind = 0.5 * (hp[0:w] + hp[1 : 1 + w])           # ground_y(x - 2.5)
        return (ahead - behind) / SLOPE_SPAN

    def _has_cliff(self) -> bool:
        return bool(np.max(np.abs(self.column_slope())) > MAX_SLOPE_CLIMB)

    def _has_shelf(self, width: int) -> bool:
        flat = np.abs(self.column_slope()) <= FLAT_SLOPE
        if self.chasm is not None:
            c0, c1 = self.chasm
            flat[max(0, c0 - 12) : min(self.W, c1 + 12)] = False
        return any((b - a) >= width for a, b in _runs(flat))

    def ground_y(self, x: float) -> float:
        """Ground surface ``y`` at fractional ``x``, linearly interpolated."""
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return float(self.height[0])
        if not np.isfinite(xf):
            return float(self.height[0])
        if xf <= 0.0:
            return float(self.height[0])
        last = self.W - 1
        if xf >= last:
            return float(self.height[last])
        i = int(xf)
        t = xf - i
        h = self.height
        return float(h[i]) * (1.0 - t) + float(h[i + 1]) * t

    def slope(self, x: float) -> float:
        """``dy/dx`` at ``x``, central difference smoothed over ~5 px."""
        return (self.ground_y(x + _SLOPE_HALF) - self.ground_y(x - _SLOPE_HALF)) / SLOPE_SPAN

    def is_walkable(self, x: float) -> bool:
        """Gentle enough to simply walk across."""
        return abs(self.slope(x)) <= MAX_SLOPE_WALK

    def is_climbable(self, x: float) -> bool:
        """Steep, maybe needs climbing, but not a fall-risk cliff."""
        return abs(self.slope(x)) <= MAX_SLOPE_CLIMB

    def is_cliff(self, x: float) -> bool:
        """Beyond climbing: agents here can slip and fall."""
        return abs(self.slope(x)) > MAX_SLOPE_CLIMB

    def material_at(self, x: float) -> int:
        i = int(np.clip(int(x) if np.isfinite(x) else 0, 0, self.W - 1))
        return int(self.material[i])

    def highest_point(self) -> int:
        """``x`` of the highest ground (smallest y)."""
        return int(np.argmin(self.height))

    def lowest_point(self) -> int:
        """``x`` of the lowest ground (largest y)."""
        return int(np.argmax(self.height))

    def columns_below(self, y: float) -> np.ndarray:
        """Column indices whose surface sits below ``y`` - i.e. submerged by a
        water line at ``y``.  Used by the flood scene."""
        try:
            yv = float(y)
        except (TypeError, ValueError):
            return np.empty(0, dtype=np.int64)
        if not np.isfinite(yv):
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(self.height > yv).astype(np.int64)

    def is_submerged(self, x: float, water_y: float) -> bool:
        return self.ground_y(x) > float(water_y)

    def find_flat_span(self, width: int, rng: np.random.Generator) -> int | None:
        """An ``x`` where ``|slope|`` stays small for ``width`` px, or ``None``.

        Deterministic for a given ``rng`` state.  Structure placement uses this;
        the chasm floor is excluded so nobody builds a hut down the hole.
        """
        try:
            need = int(width)
        except (TypeError, ValueError):
            return None
        if need <= 0:
            return 0
        if need >= self.W:
            return None

        sl = np.abs(self.column_slope())
        blocked = np.zeros(self.W, dtype=bool)
        if self.chasm is not None:
            c0, c1 = self.chasm
            blocked[max(0, c0 - 16) : min(self.W, c1 + 16)] = True

        for thresh in (FLAT_SLOPE, 0.45, 0.7):
            ok = (sl <= thresh) & ~blocked
            runs = [(a, b) for a, b in _runs(ok) if (b - a) >= need]
            if not runs:
                continue
            slots = np.array([(b - a) - need + 1 for a, b in runs], dtype=np.float64)
            cw = np.cumsum(slots)
            u = float(rng.random()) * float(cw[-1])
            k = int(np.searchsorted(cw, u, side="right"))
            k = int(np.clip(k, 0, len(runs) - 1))
            off = int(rng.integers(0, int(slots[k])))
            return int(runs[k][0] + off)
        return None

    def find_basins(
        self,
        min_width: int = 30,
        min_depth: float = 6.0,
        max_width: int = 420,
    ) -> list[tuple[int, int, float, float]]:
        """Local depressions that would hold water.

        Classic trapped-water sweep on the inverted heightmap.  Returns a list
        of ``(x0, x1, surface_y, floor_y)`` sorted deepest first, where ``x1``
        is exclusive.  Ridiculously wide basins (the whole map sagging between
        two tall edges) are filtered out via ``max_width``.
        """
        try:
            elev = -self.height.astype(np.float64)
            lmax = np.maximum.accumulate(elev)
            rmax = np.maximum.accumulate(elev[::-1])[::-1]
            level = np.minimum(lmax, rmax)
            depth = np.clip(level - elev, 0.0, None)
            out: list[tuple[int, int, float, float]] = []
            for a, b in _runs(depth > float(min_depth)):
                if (b - a) < int(min_width) or (b - a) > int(max_width):
                    continue
                floor_y = float(np.max(self.height[a:b]))
                surface_y = float(-np.max(level[a:b]))
                out.append((int(a), int(b), surface_y, floor_y))
            out.sort(key=lambda r: r[3] - r[2], reverse=True)
            return out
        except Exception:
            return []

    # ------------------------------------------------------------ mutators --

    def deform(self, x0: int, x1: int, dy: float, blend: str = "smooth") -> None:
        """Raise or lower the span ``[x0, x1]`` by ``dy``.

        ``dy`` is *added to the heightmap*, and y grows downward, so a positive
        ``dy`` digs the ground **down** and a negative ``dy`` pushes it **up**.
        ``blend`` shapes the edges so slides and diggings do not look blocky:

        ``'smooth'`` cosine falloff over ``DEFORM_EDGE`` px at each end (default),
        ``'bowl'``   one cosine bell across the whole span,
        ``'linear'`` triangular falloff,
        ``'hard'``   no falloff at all.
        """
        try:
            a = int(x0)
            b = int(x1)
            amount = float(dy)
        except (TypeError, ValueError):
            return
        if not np.isfinite(amount) or amount == 0.0:
            return
        if a > b:
            a, b = b, a
        a = max(0, min(a, self.W - 1))
        b = max(0, min(b, self.W - 1))
        n = b - a + 1
        if n < 1:
            return

        mode = blend if blend in ("smooth", "bowl", "linear", "hard") else "smooth"
        prof = np.ones(n, dtype=np.float64)
        if mode == "bowl":
            t = (np.arange(n) + 0.5) / n
            prof = 0.5 - 0.5 * np.cos(2.0 * np.pi * t)
        elif mode == "linear":
            e = max(1, min(n // 2, DEFORM_EDGE))
            ramp = (np.arange(e) + 0.5) / e
            prof[:e] = ramp
            prof[n - e :] = ramp[::-1]
        elif mode == "smooth":
            e = max(1, min(n // 2, DEFORM_EDGE))
            ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(e) + 0.5) / e)
            prof[:e] = ramp
            prof[n - e :] = ramp[::-1]

        self.height[a : b + 1] += (amount * prof).astype(np.float32)
        np.clip(self.height, HARD_MIN, HARD_MAX, out=self.height)

    def crater(self, cx: int, radius: int, depth: float) -> None:
        """Punch a meteor crater: parabolic bowl, raised rim, scorched floor."""
        try:
            c = int(cx)
            r = int(radius)
            d = float(depth)
        except (TypeError, ValueError):
            return
        if r < 1 or not np.isfinite(d) or d == 0.0:
            return

        a = max(0, c - int(r * 1.45))
        b = min(self.W, c + int(r * 1.45) + 1)
        if b <= a:
            return
        x = np.arange(a, b, dtype=np.float64)
        u = np.abs(x - c) / float(r)

        bowl = np.clip(1.0 - u * u, 0.0, 1.0)                     # dig down
        rim = np.clip(1.0 - np.abs(u - 1.06) / 0.42, 0.0, 1.0)    # ejecta lip
        rim = rim * rim * (3.0 - 2.0 * rim)
        delta = d * bowl - 0.22 * d * rim
        self.height[a:b] += delta.astype(np.float32)
        np.clip(self.height, HARD_MIN, HARD_MAX, out=self.height)

        inner_a = max(0, c - int(r * 0.9))
        inner_b = min(self.W - 1, c + int(r * 0.9))
        self.set_material_span(inner_a, inner_b, MAT_ASH)

    def set_material_span(self, x0: int, x1: int, mat: int) -> None:
        """Repaint columns ``[x0, x1]`` (inclusive) with material ``mat``."""
        try:
            a = int(x0)
            b = int(x1)
            m = int(mat)
        except (TypeError, ValueError):
            return
        if a > b:
            a, b = b, a
        a = max(0, min(a, self.W - 1))
        b = max(0, min(b, self.W - 1))
        if b < a:
            return
        self.material[a : b + 1] = np.uint8(m & 0xFF)

    def add_layer(
        self,
        mat: int,
        thickness: float,
        x0: int | None = None,
        x1: int | None = None,
    ) -> None:
        """Accumulate a layer of ``mat`` on top of the ground (snow, ash).

        Raises the surface by up to ``thickness`` and repaints the columns that
        actually accumulated.  Steep faces shed the layer, so a cliff keeps its
        rock face while the ledges above and below go white.
        """
        try:
            m = int(mat)
            th = float(thickness)
        except (TypeError, ValueError):
            return
        if not np.isfinite(th) or th <= 0.0:
            return

        a = 0 if x0 is None else max(0, min(int(x0), self.W - 1))
        b = self.W - 1 if x1 is None else max(0, min(int(x1), self.W - 1))
        if a > b:
            a, b = b, a
        sl = np.abs(self.column_slope()[a : b + 1])
        keep = np.clip(1.0 - sl / 2.0, 0.0, 1.0)
        eff = th * keep

        self.height[a : b + 1] -= eff.astype(np.float32)
        np.clip(self.height, HARD_MIN, HARD_MAX, out=self.height)

        paint = eff >= max(0.12, th * 0.35)
        if paint.any():
            seg = self.material[a : b + 1]
            seg[paint] = np.uint8(m & 0xFF)
            self.material[a : b + 1] = seg

    def retexture(self) -> None:
        """Recompute materials from height and slope.

        Called once after generation.  Safe to call again after a large
        reshaping event, but note it erases painted ash/snow.
        """
        try:
            sl = np.abs(self.column_slope())
            h = self.height.astype(np.float64)
            rng = np.random.default_rng((int(self.seed) ^ 0x5EED) & 0xFFFFFFFF)
            patch = _value_noise(rng, self.W, 11.0)

            mat = np.full(self.W, MAT_GRASS, dtype=np.uint8)
            mat[patch > 0.66] = MAT_DIRT
            mat[h > RENDER_H * 0.885] = MAT_SAND
            mat[h < RENDER_H * 0.415] = MAT_STONE
            mat[sl > 1.15] = MAT_STONE
            if self.chasm is not None:
                c0, c1 = self.chasm
                a = max(0, c0 - 12)
                b = min(self.W, c1 + 12)
                mat[a:b] = MAT_STONE
            self.material = mat
        except Exception:
            self.material = np.full(self.W, MAT_GRASS, dtype=np.uint8)

    # ------------------------------------------------------------ persistence --

    def to_dict(self) -> dict:
        """Exact landscape snapshot - raw numpy bytes, base64'd."""
        return {
            "w": int(self.W),
            "seed": int(self.seed),
            "style": str(self.style),
            "chasm": [int(self.chasm[0]), int(self.chasm[1])] if self.chasm else None,
            "height": _b64_array(self.height.astype(np.float32)),
            "material": _b64_array(self.material.astype(np.uint8)),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Terrain":
        """Restore a saved landscape.  Never raises.

        Missing or unreadable arrays fall back to regenerating from the saved
        seed/style, and finally to a fresh default world.
        """
        if not isinstance(d, dict):
            return cls.generate(0, "hills")

        seed = d.get("seed", 0)
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = 0
        style = d.get("style", "hills")
        style = style if isinstance(style, str) and style in STYLES else "hills"

        try:
            w = int(d.get("w", RENDER_W))
        except (TypeError, ValueError):
            w = RENDER_W
        if w != RENDER_W:
            w = RENDER_W

        height = _unb64_array(d.get("height"), "float32", w)
        material = _unb64_array(d.get("material"), "uint8", w)
        if height is None:
            return cls.generate(seed, style)
        if material is None:
            material = np.full(w, MAT_GRASS, dtype=np.uint8)

        chasm: tuple[int, int] | None = None
        raw = d.get("chasm")
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            try:
                chasm = (int(raw[0]), int(raw[1]))
            except (TypeError, ValueError):
                chasm = None

        try:
            return cls(
                W=w,
                height=height.astype(np.float32),
                material=material.astype(np.uint8),
                seed=seed,
                style=style,
                chasm=chasm,
            )
        except Exception:
            return cls.generate(seed, style)


def _shape_style(
    f: np.ndarray, style: str, rng: np.random.Generator, w: int
) -> np.ndarray:
    """Bend a 0..1 fbm profile into the requested landscape character."""
    x = np.arange(w, dtype=np.float64)

    if style == "cliffs":
        terraced = np.floor(f * 5.0) / 5.0
        g = 0.42 * f + 0.58 * terraced
        return np.clip(0.10 + 0.80 * g, 0.0, 1.0)

    if style == "plateau":
        cx = float(rng.integers(int(w * 0.30), int(w * 0.70)))
        hw = float(rng.integers(150, 300))
        edge = float(rng.uniform(15.0, 26.0))
        d = np.abs(x - cx)
        mask = _smoothstep((hw + edge - d) / (2.0 * edge))
        base = 0.10 + 0.46 * f
        top = 0.62 + 0.10 * f
        return np.clip(base * (1.0 - mask) + top * mask, 0.0, 1.0)

    if style == "chasm":
        return np.clip(0.12 + 0.70 * f, 0.0, 1.0)

    if style == "valley":
        cx = w * 0.5 + float(rng.uniform(-120.0, 120.0))
        wid = float(rng.uniform(w * 0.30, w * 0.42))
        d = np.clip(np.abs(x - cx) / wid, 0.0, 1.0)
        bowl = _smoothstep(d)
        return np.clip(0.06 + 0.34 * f + 0.56 * bowl, 0.0, 1.0)

    # 'hills' and anything unexpected
    return np.clip(0.12 + 0.76 * _smoothstep(f), 0.0, 1.0)


if __name__ == "__main__":  # pragma: no cover - run with: python -m backgrounded.sim.terrain
    for _style in STYLES:
        for _seed in range(6):
            _t = Terrain.generate(_seed, _style)
            _cs = np.abs(_t.column_slope())
            _flat = max((b - a) for a, b in _runs(_cs <= FLAT_SLOPE)) if _t._has_shelf(1) else 0
            _lo = float(_t.height.min())
            _hi = float(_t.height.max())
            _rt = Terrain.from_dict(_t.to_dict())
            assert np.array_equal(_rt.height, _t.height), "height round-trip"
            assert np.array_equal(_rt.material, _t.material), "material round-trip"
            assert _t._has_cliff(), f"{_style}/{_seed}: no cliff (max {_cs.max():.2f})"
            assert _t._has_shelf(120), f"{_style}/{_seed}: no 120px shelf"
            assert _lo >= HEIGHT_MIN - 0.5, f"{_style}/{_seed}: too high {_lo}"
            assert _hi <= (CHASM_FLOOR + 9.0 if _style == "chasm" else HEIGHT_MAX + 0.5)
            print(
                f"{_style:8s} seed={_seed}  slope_max={_cs.max():5.2f}  "
                f"flat={_flat:4d}px  y=[{_lo:6.1f},{_hi:6.1f}]  basins={len(_t.find_basins())}"
            )
    print("terrain smoke test OK")
