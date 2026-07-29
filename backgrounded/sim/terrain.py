"""Terrain: a per-column heightmap plus a per-column material id.

Pure python + numpy.  **No pygame** - this module has to import and run
headless so the simulation can be tested without a display.

Coordinate system (see docs/ARCHITECTURE.md section 3):

    * world space *is* render space, origin top-left
    * ``x`` runs 0 .. RENDER_W-1, one array column per screen pixel column
    * ``y`` grows **downward**, so a *smaller* height value is *higher* ground

Every mutator is written to fail soft: bad indices are clamped, NaNs are
dropped, and nothing in here raises on a per-frame path.

The effective surface
---------------------
``height`` is the *raw ground*.  What agents actually stand on is the
**effective surface**: the raw ground with any finished crossing laid over it.
Two overlays feed it, both one float per column with ``NaN`` meaning "nothing
here", and both combined with the ground by "whichever surface is higher"
(``fmin``, because y grows downward):

``deck``   a finished bridge's planking, spanning a gap at rim height.
``climb``  the ramp a finished ladder puts against a vertical face.  Columns
           carrying a climb aid are additionally exempt from the cliff test -
           making an unclimbable face climbable is what a ladder *is*.

``ground_y``, ``slope``, ``column_slope`` and ``is_cliff`` all read the
effective surface, which is why a stamped deck starts working for movement,
cliff avoidance and prop/structure siting without any of those callers knowing
that structures exist.  Generation, texturing, water and snow deliberately keep
reading the raw ``height`` - a bridge is not geology, does not hold a lake and
does not change what the rock under it is made of.

``epoch`` counts every change to the surface (height or either overlay), so a
caller that caches something derived from it can invalidate on an integer
compare instead of diffing a 1600-column array.

Interfaces other modules rely on
--------------------------------
``Terrain.chasm``            ``(x0, x1)`` of the cut gap or ``None`` - the
                             bridge buildable (feature 9) needs this.  These are
                             the *midpoints* of the two blended walls, not the
                             rims: each wall reaches up to ``CHASM_WALL_MAX``
                             further out again.
``Terrain.find_flat_span``   structure placement.
``Terrain.find_basins``      water/pond placement and the flood event.
``Terrain.deform``/``crater``/``add_layer``  mudslide, meteor, blizzard, ashfall.
``Terrain.stamp_deck``/``stamp_climb``/``clear_crossing``  finished bridges and
                             ladders, driven from structures.py.
``Terrain.find_climb_face``  where a ladder would be worth building.
``Terrain.regions``/``barriers``  which stretches of the map agents can move
                             between, and the walls between them.  Cached
                             against ``epoch``; behaviour asks this to notice
                             the colony has been cut off from something.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..constants import (
    BARRIER_MIN_RELIEF,
    CHASM_SAFE_GRADIENT,
    CROSSING_RAMP_PX,
    LADDER_MAX_W,
    LADDER_MIN_RISE,
    LADDER_MIN_W,
    LADDER_SLOPE,
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

# -- chasm walls -------------------------------------------------------------
# The hole is not what kills at a chasm; the *walls* are.  Each wall is a
# smoothstep blend from the rim down to the floor, ``2 * wall`` px wide and
# straddling the declared span's edge, so half of it eats into the gap and half
# spills outside.  A smoothstep's gradient peaks at 1.5x its average, which
# gives the whole design in one line:
#
#     peak |dy/dx|  =  depth * 1.5 / (2 * wall)
#
# (predicted 12.8 against 12.9 measured, on a 341 px cut at wall 20).  Depth is
# not a constant - it is the floor minus whatever the generated rim happened to
# be, and across seeds that runs 168 to 495 px - so no single wall value can
# hold every cut under CHASM_SAFE_GRADIENT.  At a flat wall of 20 the shallow
# cuts come out over-blended and the deep ones still peak near 18.  The wall is
# therefore derived per cut, per *side*, from the depth that side actually has.

#: Lower bound on the derived wall.  A shallow cut needs nothing wider, and a
#: crack with sharp edges is worth keeping where it is safe: this is the value
#: the chasm was cut with before the wall became adaptive.
CHASM_WALL_MIN: int = 9

#: Ceiling on it.  The deepest cut the generation band allows is
#: ``CHASM_FLOOR + 8 - HEIGHT_MIN`` = 488 px, which by the formula above wants a
#: 34 px wall; 36 leaves the verification loop below a round-up of headroom on
#: top of that.  It does bind occasionally (2 of 32 measured seeds) and the
#: failure is soft when it does: the wall stops widening and comes out at
#: whatever gradient the geometry gives, which is still nowhere near the ~14:1
#: where a slip turns fatal - worst measured over those 32 seeds is 10.97.
CHASM_WALL_MAX: int = 36

#: Flat floor left between the two walls, px.  The gap still has to *read* as a
#: gap and still has to be worth bridging, so the span is widened when the walls
#: would otherwise eat it - 40 px is the narrowest floor the old fixed-wall cut
#: ever produced (a 60 px gap at wall 9), so nothing gets narrower than what
#: already shipped.
CHASM_FLOOR_MIN: int = 40

#: Re-cuts allowed while verifying the measured gradient against the predicted
#: one.  Two is normally enough; the extra pair is for a rim whose own slope is
#: fighting the blend.
CHASM_WALL_TRIES: int = 4

#: ``slope()`` is a central difference over this many pixels total.
SLOPE_SPAN: float = 5.0
_SLOPE_HALF: float = SLOPE_SPAN * 0.5

#: |slope| at or below this counts as "flat enough to build on".
FLAT_SLOPE: float = 0.28

#: Cosine falloff width used by :meth:`Terrain.deform` with ``blend='smooth'``.
DEFORM_EDGE: int = 24

#: px the reachability sweep looks ahead when judging a step.  Deliberately the
#: same number as ``entities.STEP_PROBE``: that is the span the physics itself
#: measures a step over, so the graph refuses exactly the steps the mover does.
#: Imported by value rather than from entities - sim/terrain.py sits *under*
#: entities in the import order and must not reach back up.
BARRIER_PROBE: int = 3

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


def _wall_for_depth(depth: float) -> int:
    """Blend half-width a ``depth`` px drop needs to stay under the limit.

    Straight inversion of ``peak = depth * 1.5 / (2 * wall)``.  A first guess
    only - :meth:`Terrain._cut_chasm` measures what it actually got and widens
    again if the ground the cut landed in was tilting into the hole.
    """
    try:
        d = float(depth)
    except (TypeError, ValueError):
        return CHASM_WALL_MIN
    if not np.isfinite(d) or d <= 0.0:
        return CHASM_WALL_MIN
    need = d * 1.5 / (2.0 * max(1.0, float(CHASM_SAFE_GRADIENT)))
    return int(np.clip(np.ceil(need), CHASM_WALL_MIN, CHASM_WALL_MAX))


def _wall_for_peak(wall: int, peak: float) -> int:
    """The wall a *measured* peak gradient says the cut should have had.

    Gradient scales as 1/wall, so the correction is one multiply.  The extra
    pixel is not superstition: the peak is read off a 5 px central difference of
    a wobbled surface, and landing exactly on the limit leaves it free to round
    back over on the next re-cut.
    """
    try:
        p = float(peak)
    except (TypeError, ValueError):
        return int(wall)
    if not np.isfinite(p) or p <= float(CHASM_SAFE_GRADIENT):
        return int(wall)
    scaled = np.ceil(float(wall) * p / max(1.0, float(CHASM_SAFE_GRADIENT))) + 1.0
    return int(np.clip(scaled, CHASM_WALL_MIN, CHASM_WALL_MAX))


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
    #: Bridge planking laid over the raw ground, one y per column, ``NaN`` where
    #: there is none.  ``None`` while nothing is stamped - which is the case in
    #: almost every world, and is what keeps ``ground_y`` on its fast path.
    deck: np.ndarray | None = None
    #: Ladder ramps, same shape and same "NaN means absent" convention.  A
    #: non-NaN column here is *climb-aided*: ``is_cliff`` is False on it and the
    #: slope helpers clamp to ``MAX_SLOPE_CLIMB`` there.
    climb: np.ndarray | None = None

    # ---------------------------------------------------------------- setup --

    def __post_init__(self) -> None:
        # These are runtime caches, not state: they are rebuilt on demand and
        # deliberately never serialised.  Assigned before anything else so that
        # a mutator called during construction cannot find them missing.
        self._epoch: int = 0
        self._surf: np.ndarray | None = None
        self._cmask: np.ndarray | None = None
        self._surf_epoch: int = -1
        self._reach: tuple[np.ndarray, tuple[tuple[int, int, float], ...]] | None = None
        self._reach_epoch: int = -1

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

        # An overlay that is the wrong length, unreadable, or entirely NaN is
        # indistinguishable from "no crossings", and that is how a malformed or
        # older save has to land: without one, not refusing to load.
        self.deck = self._coerce_overlay(self.deck)
        self.climb = self._coerce_overlay(self.climb)

    def _coerce_overlay(self, arr: Any) -> np.ndarray | None:
        """A contiguous float32 overlay of exactly ``W`` columns, or ``None``."""
        if arr is None:
            return None
        try:
            a = np.asarray(arr, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if a.size != self.W or not bool(np.isfinite(a).any()):
            return None
        # An overlay is a *surface*, so it lives under the same hard clamp the
        # heightmap does; NaNs survive the clip untouched and stay "absent".
        return np.ascontiguousarray(np.clip(a, HARD_MIN, HARD_MAX), dtype=np.float32)

    # ------------------------------------------------- effective surface --

    @property
    def epoch(self) -> int:
        """Bumped by every change to the walkable surface.

        Height *or* either overlay.  A caller memoising something expensive off
        the surface - phase 2's reachability graph is the motivating one - can
        hold ``(epoch, result)`` and invalidate on an int compare rather than
        rescanning 1600 columns every tick to find out nothing moved.
        """
        return int(getattr(self, "_epoch", 0))

    def touch(self) -> None:
        """Declare the surface changed.  Cheap, never raises, safe to overcall.

        Every mutator in here calls it.  It is public so that anything that
        writes ``terrain.height`` in place from outside this module can keep the
        invariant true without reaching into private state.
        """
        try:
            self._epoch = int(getattr(self, "_epoch", 0)) + 1
        except Exception:
            self._epoch = 1
        self._surf = None
        self._cmask = None
        self._reach = None

    def _refresh(self) -> None:
        """Rebuild the cached effective surface and climb-aid mask."""
        s = self.height.copy()
        for ov in (self.deck, self.climb):
            if ov is not None and ov.size == s.size:
                # fmin, not minimum: NaN means "no overlay here", and fmin is
                # the one that yields the other operand rather than NaN.
                np.fmin(s, ov, out=s)
        self._surf = s
        self._cmask = np.isfinite(self.climb) if self.climb is not None else None
        self._surf_epoch = self._epoch

    def surface(self) -> np.ndarray:
        """The walkable surface: raw ground with any finished crossing on top.

        Returns ``height`` itself - no copy, no work - in the overwhelmingly
        common case where nothing is stamped, so the callers that hit this every
        tick pay two identity checks rather than an array merge.  Do not write
        to the result: it may be the live heightmap.
        """
        # `_epoch` rather than the `epoch` property: this is on the per-agent
        # per-tick path and the property costs a getattr-with-default plus an
        # int() every call for no benefit here.
        if self.deck is None and self.climb is None:
            return self.height
        if self._surf is None or self._surf_epoch != self._epoch:
            self._refresh()
        return self._surf if self._surf is not None else self.height

    def climb_aided(self, x: float) -> bool:
        """True where a finished ladder makes the column passable."""
        if self.climb is None:
            return False
        if self._cmask is None or self._surf_epoch != self._epoch:
            self._refresh()
        m = self._cmask
        if m is None or m.size != self.W:
            return False
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return False
        # NaN test written as a self-comparison: np.isfinite on a python float
        # is a numpy call, and this runs several times per agent per tick. The
        # two range tests below catch both infinities, so int() only ever sees
        # a finite in-range value and cannot raise OverflowError.
        if xf != xf:
            return False
        last = self.W - 1
        i = 0 if xf < 0.0 else (last if xf >= last else int(xf))
        # A ladder occupies whole columns but agents stand at fractional x, so
        # either side of the boundary counts as on it.  Without this an agent at
        # x.9 one column short of the ramp reads the unaided cliff.
        return bool(m[i]) or (i < last and bool(m[i + 1]))

    def has_deck(self, x: float) -> bool:
        """True where a finished bridge's planking covers the column."""
        d = self.deck
        if d is None:
            return False
        try:
            i = int(np.clip(int(float(x)) if np.isfinite(float(x)) else 0, 0, self.W - 1))
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(d[i]))

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

        # The *zone*, not the declared span: an escarpment landing on the outer
        # half of a chasm wall would re-steepen the face the cut just softened.
        shelf = t._carve_shelf(rng)
        t._carve_cliff(rng, avoid=(shelf, t._chasm_zone()))
        t._clip_gen()

        # -- guarantee sweep ------------------------------------------------
        for _ in range(3):
            if t._has_cliff():
                break
            t._carve_cliff(rng, avoid=(shelf, t._chasm_zone()), force=True)
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
        t.touch()
        t.retexture()
        return t

    def _clip_gen(self) -> None:
        """Clip to the generation band, sparing the chasm floor."""
        lo = np.full(self.W, HEIGHT_MIN, dtype=np.float32)
        hi = np.full(self.W, HEIGHT_MAX, dtype=np.float32)
        # The whole wall, not just the declared span: a wall crosses HEIGHT_MAX
        # somewhere down its face, and clipping the columns past the window back
        # up to 736 would put a hard step exactly where the blend was smooth.
        zone = self._chasm_zone(4)
        if zone is not None:
            a = max(0, zone[0])
            b = min(self.W, zone[1])
            if b > a:
                hi[a:b] = np.float32(CHASM_FLOOR + 8.0)
        np.clip(self.height, lo, hi, out=self.height)
        # Every generation carver funnels through here, so this one call keeps
        # the epoch honest across the whole build even though nothing can be
        # caching against it yet (no overlay exists until a structure finishes).
        self.touch()

    def _chasm_zone(self, pad: int = 0) -> tuple[int, int] | None:
        """The cut *including its walls*, widened by ``pad``.  Columns, unclipped.

        ``chasm`` is the declared span, and its edges are the blend *midpoints*,
        not the rims: each wall reaches up to ``CHASM_WALL_MAX`` outside it.
        Anything that has to stay off the cut has to stay off the walls too -
        flattening or clipping the outer half of a wall puts a step back in the
        one place this whole exercise exists to keep smooth.

        Sized off the cap rather than off the wall this particular cut used, so
        the answer does not depend on state that a terrain restored from a save
        never had: ``_cut_chasm`` does not run on load, only the span is stored.
        The cost is a slightly generous exclusion around a shallow cut, which
        every caller here can afford.
        """
        if self.chasm is None:
            return None
        try:
            c0, c1 = int(self.chasm[0]), int(self.chasm[1])
        except (TypeError, ValueError, IndexError):
            return None
        m = CHASM_WALL_MAX + max(0, int(pad))
        return c0 - m, c1 + m

    def _stamp_chasm(
        self,
        x0: int,
        x1: int,
        wl: int,
        wr: int,
        floor: float,
        rng: np.random.Generator,
    ) -> None:
        """Blend the floor in across ``[x0, x1]`` with per-side wall widths."""
        w = self.W
        # A zero wall would divide to inf rather than raise, and a clipped
        # smoothstep of inf is a step function - i.e. the sheerest possible
        # wall, arrived at silently, which is the exact opposite of the point.
        wl = max(1, int(wl))
        wr = max(1, int(wr))
        a = max(0, x0 - wl)
        b = min(w, x1 + wr)
        if b <= a:
            return
        idx = np.arange(a, b)
        left = _smoothstep((idx - (x0 - wl)) / float(2 * wl))
        right = 1.0 - _smoothstep((idx - (x1 - wr)) / float(2 * wr))
        mask = np.clip(np.minimum(left, right), 0.0, 1.0)
        wobble = floor + rng.uniform(-2.0, 2.0, size=idx.size)
        cur = self.height[a:b].astype(np.float64)
        self.height[a:b] = (cur * (1.0 - mask) + wobble * mask).astype(np.float32)

    def _cut_chasm(self, rng: np.random.Generator) -> None:
        """Cut a 60-120 px gap most of the way down the screen, for a bridge.

        The gap is the feature; the *walls* are the hazard, and they are sized
        rather than picked.  Each side gets the blend width its own depth needs
        to keep the face under ``CHASM_SAFE_GRADIENT`` - see the notes on
        ``CHASM_WALL_MIN`` above for why one number cannot serve both a 168 px
        cut and a 495 px one.  Per side, not per cut, because the two rims are
        rarely the same height: sizing both walls off the deeper one would spend
        the gap's whole width blending a side that never needed it.

        The prediction is then *verified*, because it is a prediction about a
        blend in isolation and the ground it lands in has a slope of its own
        that adds to it.  Cutting onto a snapshot rather than onto the running
        heightmap is what makes a re-cut possible: measure the face an agent
        would actually feel, and if it came out steep, throw the cut away and do
        it again wider.

        The span itself grows if the walls would otherwise eat the floor, so a
        deep cut is a wide cut rather than a V.  It never shrinks below the
        60-120 px it always drew.
        """
        w = self.W
        gw = int(rng.integers(60, 121))
        cx = int(rng.integers(int(w * 0.28), int(w * 0.72)))
        floor = float(CHASM_FLOOR) + float(rng.uniform(-5.0, 5.0))
        pre = self.height.copy()

        # First guess from the untouched ground at the nominal rims.
        probe0 = int(np.clip(cx - gw // 2, 30, w - 30 - gw))
        wl = _wall_for_depth(floor - float(pre[probe0]))
        wr = _wall_for_depth(floor - float(pre[min(w - 1, probe0 + gw)]))

        x0, x1 = probe0, probe0 + gw
        for _ in range(max(1, CHASM_WALL_TRIES)):
            span = max(gw, wl + wr + CHASM_FLOOR_MIN)
            lo = 30 + wl
            hi = max(lo, w - 30 - wr - span)
            x0 = int(np.clip(cx - span // 2, lo, hi))
            x1 = min(w - 1, x0 + span)

            self.height[:] = pre
            self._stamp_chasm(x0, x1, wl, wr, floor, rng)

            sl = np.abs(self.column_slope(raw=True))
            # +-3 px past each blend: that is the reach of the central
            # difference, so the toe of the wall is included rather than
            # measured half in and half out of the flat above it.
            nl = _wall_for_peak(wl, self._span_peak(sl, x0 - wl - 3, x0 + wl + 3))
            nr = _wall_for_peak(wr, self._span_peak(sl, x1 - wr - 3, x1 + wr + 3))
            # Also how the loop terminates against the cap: _wall_for_peak
            # clamps to CHASM_WALL_MAX, so a side already there asks for no
            # more and reads as converged.
            if nl <= wl and nr <= wr:
                break
            # Monotone: a re-cut only ever widens, so the loop cannot oscillate
            # between two shapes and burn its whole budget doing it.
            wl = max(wl, nl)
            wr = max(wr, nr)

        self.chasm = (x0, x1)
        self.touch()

    def _span_peak(self, absslope: np.ndarray, a: int, b: int) -> float:
        """Worst ``|dy/dx|`` between columns ``a`` and ``b`` inclusive."""
        a = int(max(0, min(int(a), self.W - 1)))
        b = int(max(0, min(int(b), self.W - 1)))
        if b < a or absslope.size < self.W:
            return 0.0
        return float(np.max(absslope[a : b + 1]))

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

        sl = np.abs(self.column_slope(raw=True))
        win = np.lib.stride_tricks.sliding_window_view(sl, span)
        score = win.max(axis=1) + 0.25 * win.mean(axis=1)

        bad = np.zeros(n, dtype=bool)
        bad[: min(40, n)] = True
        bad[max(0, n - 40) :] = True
        # Keep clear of the walls, not just the gap: a shelf is *perfectly*
        # flat, and one whose ramp overlapped a wall would leave a step where
        # the two profiles met.
        zone = self._chasm_zone(12)
        if zone is not None:
            starts = np.arange(n)
            bad |= (starts + span > zone[0]) & (starts < zone[1])
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
        self.touch()
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

    def column_slope(self, raw: bool = False) -> np.ndarray:
        """Per-column ``dy/dx``, identical to :meth:`slope` at integer x.

        Central difference over ``SLOPE_SPAN`` px, edge-padded so the borders
        do not report phantom cliffs.

        Reads the *effective* surface by default, so a finished bridge reads as
        the flat deck it is.  ``raw=True`` asks about the bare geology instead,
        which is what generation, texturing and snow accumulation want: a deck
        is not a reason to call the rock under it flat.
        """
        h = (self.height if raw else self.surface()).astype(np.float64)
        w = self.W
        hp = np.pad(h, 3, mode="edge")
        ahead = 0.5 * (hp[5 : 5 + w] + hp[6 : 6 + w])      # ground_y(x + 2.5)
        behind = 0.5 * (hp[0:w] + hp[1 : 1 + w])           # ground_y(x - 2.5)
        sl = (ahead - behind) / SLOPE_SPAN
        if not raw and self._cmask is not None and self._cmask.size == w:
            # A ladder's ramp is climbable by construction, but the two columns
            # where it meets the face still show the join in a 5 px central
            # difference.  Clamping there keeps every consumer's "is this a
            # cliff" answer consistent with is_cliff's.
            sl = np.where(self._cmask, np.clip(sl, -MAX_SLOPE_CLIMB, MAX_SLOPE_CLIMB), sl)
        return sl

    def _has_cliff(self) -> bool:
        return bool(np.max(np.abs(self.column_slope(raw=True))) > MAX_SLOPE_CLIMB)

    def _has_shelf(self, width: int) -> bool:
        flat = np.abs(self.column_slope(raw=True)) <= FLAT_SLOPE
        if self.chasm is not None:
            c0, c1 = self.chasm
            flat[max(0, c0 - 12) : min(self.W, c1 + 12)] = False
        return any((b - a) >= width for a, b in _runs(flat))

    def ground_y(self, x: float) -> float:
        """Walkable surface ``y`` at fractional ``x``, linearly interpolated.

        This is the effective surface, not the raw heightmap: over a finished
        bridge it reports the deck, and on a ladder's columns it reports the
        ramp.  That is deliberate and is the whole mechanism - movement, cliff
        avoidance, prop placement and build siting all come through here, so
        they all start honouring a crossing the moment one is stamped.
        """
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return float(self.height[0])
        if not np.isfinite(xf):
            return float(self.height[0])
        h = self.surface()
        if xf <= 0.0:
            return float(h[0])
        last = self.W - 1
        if xf >= last:
            return float(h[last])
        i = int(xf)
        t = xf - i
        return float(h[i]) * (1.0 - t) + float(h[i + 1]) * t

    def slope(self, x: float) -> float:
        """``dy/dx`` at ``x``, central difference smoothed over ~5 px."""
        s = (self.ground_y(x + _SLOPE_HALF) - self.ground_y(x - _SLOPE_HALF)) / SLOPE_SPAN
        # Identity check first: this runs several times per agent per tick and
        # in almost every world there is no ladder to ask about.
        if self.climb is not None and self.climb_aided(x):
            if s > MAX_SLOPE_CLIMB:
                return MAX_SLOPE_CLIMB
            if s < -MAX_SLOPE_CLIMB:
                return -MAX_SLOPE_CLIMB
        return s

    def is_walkable(self, x: float) -> bool:
        """Gentle enough to simply walk across."""
        return abs(self.slope(x)) <= MAX_SLOPE_WALK

    def is_climbable(self, x: float) -> bool:
        """Steep, maybe needs climbing, but not a fall-risk cliff."""
        return abs(self.slope(x)) <= MAX_SLOPE_CLIMB

    def is_cliff(self, x: float) -> bool:
        """Beyond climbing: agents here can slip and fall.

        A finished ladder is exactly the thing that makes a face not a cliff, so
        its columns answer False however steep the rock behind them is.
        """
        if self.climb is not None and self.climb_aided(x):
            return False
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

        sl = np.abs(self.column_slope(raw=True))
        blocked = np.zeros(self.W, dtype=bool)
        if self.chasm is not None:
            c0, c1 = self.chasm
            blocked[max(0, c0 - 16) : min(self.W, c1 + 16)] = True
        # A deck is perfectly flat and would otherwise score as the best
        # building site on the map.  Nobody puts a hut on the bridge.
        for ov in (self.deck, self.climb):
            if ov is not None and ov.size == self.W:
                blocked |= np.isfinite(ov)

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

    def find_climb_face(
        self,
        near: float | None = None,
        min_rise: float = LADDER_MIN_RISE,
    ) -> tuple[int, int, float, float] | None:
        """Where a ladder would earn its keep.

        Finds a face too steep to climb (``|slope| > MAX_SLOPE_CLIMB``) and
        returns the footprint of a ramp that would defeat it, as
        ``(x0, x1, y0, y1)`` with ``x0 < x1``, ``y0`` the surface y at ``x0``
        and ``y1`` the surface y at ``x1`` - exactly the arguments
        :meth:`stamp_climb` takes.

        The ramp always lies on the **low** side of the face: a straight line
        back from the lip, down to where it meets the low ground.  Putting it
        anywhere else would bury it - the overlay only wins where it is above
        the rock, so a ramp running out over the high side is simply ignored and
        leaves the face exactly as impassable as it was.

        Every candidate is *verified* before it is returned, by merging the
        proposed ramp onto the real heightmap and measuring the worst gradient
        an agent would actually meet.  Sizing the run from the face's height
        alone is not enough and was measured failing: where the low ground keeps
        falling away past the lip, the foot lands far below where it was aimed
        and the "ramp" comes out steeper than the face it was meant to replace
        (3.5 and 4.6 against a 2.6 limit on cliffs/1 and cliffs/3).

        ``near`` biases the choice toward an x (the colony centre, typically);
        without it the tallest qualifying face wins.  Returns ``None`` when
        nothing qualifies, which is the normal answer on gentle terrain.
        """
        try:
            return self._find_climb_face_impl(near, float(min_rise))
        except Exception:
            return None

    def _ramp_gradient(self, a: int, b: int, ya: float, yb: float) -> float:
        """Worst ``|dy/dx|`` on a ramp once merged onto the real ground.

        Probed over 3 px because that is ``entities.STEP_PROBE`` - the span the
        physics itself judges a step over.  Measuring the ramp in isolation
        would miss the only thing that can go wrong, which is the ground poking
        back up through it.
        """
        lo = max(0, a - 4)
        hi = min(self.W - 1, b + 4)
        prof = self.height[lo : hi + 1].astype(np.float64)
        n = b - a + 1
        i = a - lo
        prof[i : i + n] = np.fmin(prof[i : i + n], np.linspace(ya, yb, n))
        if prof.size <= 3:
            return 0.0
        return float(np.max(np.abs(prof[3:] - prof[:-3]))) / 3.0

    def _find_climb_face_impl(
        self, near: float | None, min_rise: float
    ) -> tuple[int, int, float, float] | None:
        w = self.W
        h = self.height.astype(np.float64)
        steep = np.abs(self.column_slope(raw=True)) > MAX_SLOPE_CLIMB

        # A ladder into the chasm is not the crossing the colony needs, and the
        # bridge already owns that span.  Keep the two structures off each other
        # - over the whole wall, since a softened wall is a longer steep run and
        # its outer end now reaches well past the declared span.
        zone = self._chasm_zone(8)
        if zone is not None:
            steep[max(0, zone[0]) : min(w, zone[1])] = False
        # Nor onto something already bridged or laddered.
        for ov in (self.deck, self.climb):
            if ov is not None and ov.size == w:
                steep &= ~np.isfinite(ov)

        # Verify against a little under the limit: the physics also samples the
        # slope helper at fractional x, and landing exactly on 2.6 leaves no
        # room for the interpolation between two columns to round the wrong way.
        limit = MAX_SLOPE_CLIMB * 0.92

        best: tuple[int, int, float, float] | None = None
        best_score = -1e18
        for a, b in _runs(steep):
            lo_i = max(0, a - 2)
            hi_i = min(w - 1, b + 1)
            rise = abs(float(h[lo_i]) - float(h[hi_i]))
            if rise < min_rise:
                continue

            # Which end is the top (smaller y is higher ground)?
            if float(h[hi_i]) < float(h[lo_i]):
                x_top, direction = hi_i, -1
            else:
                x_top, direction = lo_i, +1
            y_top = float(h[x_top])

            # A run shorter than rise/LADDER_SLOPE cannot possibly come out
            # gentle enough - its average gradient is already over the design
            # slope before the ground under it is taken into account - so the
            # search starts there rather than crawling up from the minimum.
            first = int(max(LADDER_MIN_W, round(rise / max(0.5, LADDER_SLOPE))))
            found: tuple[int, int, float, float] | None = None
            for run in range(min(first, LADDER_MAX_W), LADDER_MAX_W + 1, 2):
                x_foot = x_top + direction * run
                if not (0 <= x_foot < w):
                    break                      # ran off the screen edge
                y_foot = float(h[x_foot])
                if abs(y_foot - y_top) < min_rise * 0.8:
                    continue                   # has not cleared the face yet
                lo_x, hi_x = (x_foot, x_top) if x_foot < x_top else (x_top, x_foot)
                y_a, y_b = (y_foot, y_top) if x_foot < x_top else (y_top, y_foot)
                if self._ramp_gradient(lo_x, hi_x, y_a, y_b) <= limit:
                    found = (lo_x, hi_x, y_a, y_b)
                    break
            if found is None:
                continue

            score = rise
            if near is not None:
                try:
                    score -= abs(0.5 * (found[0] + found[1]) - float(near)) * 0.35
                except (TypeError, ValueError):
                    pass
            if score > best_score:
                best_score = score
                best = found
        return best

    # --------------------------------------------------------- reachability --

    def _build_reach(self) -> tuple[np.ndarray, tuple[tuple[int, int, float], ...]]:
        """Label every column with the stretch of map it belongs to.

        One vectorised sweep of the *effective* surface, so a finished bridge or
        ladder rejoins the two sides it spans the moment it is stamped.

        **What counts as a barrier.**  Connectivity here is *mutual*: two
        columns are in the same region when an agent can get from either to the
        other.  In one dimension that collapses the asymmetry the physics
        genuinely has - ``entities._ground_step`` refuses a face steeper than
        ``MAX_SLOPE_CLIMB`` going *up* but lets you walk off any drop going
        *down* - into a single undirected test, because a one-way drop into a
        hole is not a route a colony can use.  Modelling the descent as
        connectivity would report the chasm floor as "reachable" (it is, once)
        and so hide the exact problem the crossing exists to solve.  So the
        first half of the test is ``|dy/dx| > MAX_SLOPE_CLIMB``, measured over
        ``BARRIER_PROBE`` px, in whichever direction is uphill.

        The second half is ``BARRIER_MIN_RELIEF`` - see the constant for the
        measurements.  Slope alone is not a barrier: generation guarantees a
        cliff on every map and agents climb the small ones routinely, so the
        slope test on its own splits an ordinary hills map into 150 pieces.
        Only a face that is both unclimbable *and* tall enough for the descent
        to kill is something the colony has to build its way past.

        Returns ``(labels, barriers)``:  ``labels[i]`` is the region id of
        column ``i`` (ids increase left to right, regions are contiguous), and
        ``barriers`` is ``(rim_left, rim_right, relief)`` per wall, where the
        two rims are the last walkable column on each side.
        """
        w = self.W
        h = self.surface().astype(np.float64)
        p = int(BARRIER_PROBE)
        if w <= p + 2:
            return np.zeros(w, dtype=np.int32), ()

        # g[j] > 0: the ground p px right of column j is *higher* (y smaller),
        # so travelling right out of j is a climb; g[j] < 0 is a climb going
        # left out of column j+p.  Either way |g| over the limit is a wall.
        g = (h[: w - p] - h[p:]) / float(p)
        bad = np.abs(g) > MAX_SLOPE_CLIMB

        # A finished ladder is precisely the thing that makes a face passable,
        # so a window touching its ramp is not a wall however steep the rock
        # under it is.  Same exemption is_cliff() makes, in bulk.
        if self.climb is not None and self.climb.size == w:
            aided = np.isfinite(self.climb)
            bad &= ~(aided[: w - p] | aided[p:])

        # A window spanning columns j..j+p blocks every boundary inside it.
        cut = np.zeros(w - 1, dtype=bool)
        for k in range(p):
            cut[k : k + bad.size] |= bad

        barriers: list[tuple[int, int, float]] = []
        # One label step per *wall*, placed at the wall's midpoint - not one per
        # cut boundary.  A wall is 20-60 columns wide and none of it is ground
        # anybody stands on, so counting each of those columns as its own region
        # would bury the two regions that matter (seed 5: 62 "regions" for what
        # is plainly three pieces of map).  Splitting the face down the middle
        # keeps regions contiguous and leaves each rim in the region it
        # overlooks, which is all any caller asks of the face itself.
        step = np.zeros(w - 1, dtype=np.int32)
        for a, b in _runs(cut):
            # Boundaries a..b-1 are cut, so columns a..b are the wall and the
            # rims are a (left) and b (right) - the last standable column on
            # each side.
            face = h[a : b + 1]
            relief = float(face.max() - face.min())
            if relief < BARRIER_MIN_RELIEF:
                continue                  # rough ground, not a wall
            barriers.append((int(a), int(b), relief))
            step[(a + b) // 2] = 1

        labels = np.concatenate(([0], np.cumsum(step))).astype(np.int32)
        return labels, tuple(barriers)

    def _reach_cache(self) -> tuple[np.ndarray, tuple[tuple[int, int, float], ...]]:
        """The memoised reachability sweep.  Rebuilt only when ``epoch`` moves."""
        if self._reach is None or self._reach_epoch != self._epoch:
            try:
                self._reach = self._build_reach()
            except Exception:
                # Fail soft: one region, no barriers, i.e. "nothing is cut off".
                # A wrong answer here must never be an exception on a director
                # pass that also has a colony to run.
                self._reach = (np.zeros(self.W, dtype=np.int32), ())
            self._reach_epoch = self._epoch
        return self._reach

    def regions(self) -> np.ndarray:
        """Per-column region id.  Do not write to the result - it is cached."""
        return self._reach_cache()[0]

    def barriers(self) -> tuple[tuple[int, int, float], ...]:
        """``(rim_left, rim_right, relief)`` for every wall, left to right."""
        return self._reach_cache()[1]

    def region_at(self, x: float) -> int:
        """Region id of the column under ``x``.  ``0`` for a nonsense ``x``."""
        lab = self._reach_cache()[0]
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return 0
        if xf != xf:                       # NaN; see climb_aided for the idiom
            return 0
        last = self.W - 1
        i = 0 if xf < 0.0 else (last if xf >= last else int(xf))
        return int(lab[i])

    def region_bounds(self, region: int) -> tuple[int, int] | None:
        """``(first_column, last_column)`` of a region, or ``None`` if unknown.

        Regions are contiguous by construction, so this is two searches rather
        than a scan.
        """
        lab = self._reach_cache()[0]
        try:
            r = int(region)
        except (TypeError, ValueError):
            return None
        a = int(np.searchsorted(lab, r, side="left"))
        b = int(np.searchsorted(lab, r, side="right")) - 1
        if a > b or a < 0 or b >= self.W:
            return None
        return a, b

    # ------------------------------------------------ crossings (overlays) --

    def _blank_overlay(self) -> np.ndarray:
        return np.full(self.W, np.nan, dtype=np.float32)

    def _span_indices(self, x0: float, x1: float) -> tuple[int, int] | None:
        """``(a, b)`` inclusive column range for a crossing, or ``None``."""
        try:
            a = int(np.floor(float(x0)))
            b = int(np.ceil(float(x1)))
        except (TypeError, ValueError, OverflowError):
            return None
        if a > b:
            a, b = b, a
        a = max(0, min(a, self.W - 1))
        b = max(0, min(b, self.W - 1))
        # Two columns is not a crossing, and something claiming to span most of
        # the map is a bug upstream rather than an unusually ambitious bridge.
        if (b - a) < 3 or (b - a) > int(self.W * 0.6):
            return None
        return a, b

    def stamp_deck(
        self, x0: float, x1: float, y: float, ramp: float = CROSSING_RAMP_PX
    ) -> bool:
        """Lay a bridge deck across ``[x0, x1]`` at height ``y``.  Fails soft.

        The deck is flat at ``y`` across the middle and eases onto the raw
        ground over a ramp at each end, so the join is continuous wherever the
        rim actually is - a hard step there reads to the walk code as a ledge
        and gets refused, which is precisely the bug this whole overlay exists
        to fix.  Returns True if anything was stamped.

        ``ramp`` is a *floor*, not the length: the ramp is sized to whatever
        height it actually has to shed.  A fixed 10 px ramp measured 2.89 on
        seed 5, where the two rims of the chasm differ by 21 px - over
        MAX_SLOPE_CLIMB, so the finished bridge still ended in a cliff and
        agents still refused the last five columns of it.
        """
        try:
            span = self._span_indices(x0, x1)
            if span is None:
                return False
            a, b = span
            level = float(y)
            if not np.isfinite(level):
                return False
            level = float(np.clip(level, HARD_MIN, HARD_MAX))

            n = b - a + 1
            # Cap how far the deck may sit off the ground it has to come down
            # onto. Each end ramp gets at most half the span, and a smoothstep's
            # gradient peaks at 1.5x its average, so a ramp r px long can shed
            # MAX_SLOPE_WALK * r / 1.5 and no more. Asking for more than that
            # does not produce a higher bridge, it produces a bridge that ends
            # in a cliff: on seeds 25 and 36 behaviour surveyed a rim 150 px
            # above the ends of the span it handed over, and the resulting deck
            # joined the ground at 3.94 against a 2.6 limit - a fresh fall
            # hazard created by the very structure meant to remove one.
            ha, hb = float(self.height[a]), float(self.height[b])
            shed = MAX_SLOPE_WALK * max(1, n // 2) / 1.5
            lo, hi = max(ha, hb) - shed, min(ha, hb) + shed
            # lo > hi means the two ends differ by more than two ramps can
            # reconcile - the span is simply too short for the step across it.
            level = float(np.clip(level, lo, hi)) if lo <= hi else 0.5 * (ha + hb)

            prof = np.full(n, level, dtype=np.float64)
            # Same peak-vs-average reasoning sizes the ramp actually used:
            # averaging leaves the steepest 3 px of the join over the limit,
            # which is the only part that matters.
            drop = max(abs(ha - level), abs(hb - level))
            need = 1.5 * drop / max(0.05, MAX_SLOPE_WALK)
            r = int(min(max(1.0, float(ramp), need), n // 2))
            if r >= 1:
                t = _smoothstep((np.arange(r, dtype=np.float64) + 0.5) / float(r))
                prof[:r] = float(self.height[a]) * (1.0 - t) + level * t
                prof[n - r :] = (float(self.height[b]) * (1.0 - t) + level * t)[::-1]

            if self.deck is None:
                self.deck = self._blank_overlay()
            self.deck[a : b + 1] = prof.astype(np.float32)
            self.touch()
            return True
        except Exception:
            return False

    def stamp_climb(self, x0: float, x1: float, y0: float, y1: float) -> bool:
        """Put a ladder ramp from ``(x0, y0)`` to ``(x1, y1)``.  Fails soft.

        A straight line between the two anchors.  Both ends are meant to sit on
        the raw ground, so the "higher surface wins" merge joins them without a
        step and the ramp only shows where it is genuinely above the rock.
        Returns True if anything was stamped.
        """
        try:
            span = self._span_indices(x0, x1)
            if span is None:
                return False
            a, b = span
            ya, yb = float(y0), float(y1)
            if not (np.isfinite(ya) and np.isfinite(yb)):
                return False
            if float(x1) < float(x0):
                ya, yb = yb, ya
            ya = float(np.clip(ya, HARD_MIN, HARD_MAX))
            yb = float(np.clip(yb, HARD_MIN, HARD_MAX))

            n = b - a + 1
            prof = np.linspace(ya, yb, n, dtype=np.float64)
            if self.climb is None:
                self.climb = self._blank_overlay()
            self.climb[a : b + 1] = prof.astype(np.float32)
            self.touch()
            return True
        except Exception:
            return False

    def clear_crossing(self, x0: float, x1: float) -> bool:
        """Remove any deck or climb aid over ``[x0, x1]``.  Fails soft.

        An overlay left holding nothing but NaN is dropped entirely, which puts
        ``ground_y`` back on its no-overlay fast path once the last bridge in
        the world falls down.
        """
        try:
            span = self._span_indices(x0, x1)
            if span is None:
                return False
            a, b = span
            hit = False
            for name in ("deck", "climb"):
                ov = getattr(self, name)
                if ov is None or ov.size != self.W:
                    continue
                if bool(np.isfinite(ov[a : b + 1]).any()):
                    hit = True
                ov[a : b + 1] = np.nan
                if not bool(np.isfinite(ov).any()):
                    setattr(self, name, None)
            if hit:
                self.touch()
            return hit
        except Exception:
            return False

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
        self.touch()

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
        self.touch()

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
        sl = np.abs(self.column_slope(raw=True)[a : b + 1])
        keep = np.clip(1.0 - sl / 2.0, 0.0, 1.0)
        eff = th * keep

        self.height[a : b + 1] -= eff.astype(np.float32)
        np.clip(self.height, HARD_MIN, HARD_MAX, out=self.height)
        self.touch()

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
            sl = np.abs(self.column_slope(raw=True))
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
        """Exact landscape snapshot - raw numpy bytes, base64'd.

        The crossing overlays are written only when they hold something, so a
        world that never built a bridge saves exactly the bytes it always did
        and an older loader that has never heard of them is unaffected.
        """
        d: dict = {
            "w": int(self.W),
            "seed": int(self.seed),
            "style": str(self.style),
            "chasm": [int(self.chasm[0]), int(self.chasm[1])] if self.chasm else None,
            "height": _b64_array(self.height.astype(np.float32)),
            "material": _b64_array(self.material.astype(np.uint8)),
        }
        if self.deck is not None:
            d["deck"] = _b64_array(self.deck.astype(np.float32))
        if self.climb is not None:
            d["climb"] = _b64_array(self.climb.astype(np.float32))
        return d

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

        # A crossing overlay is pure convenience: it can always be re-stamped by
        # the structure that owns it on its next tick.  So anything unreadable
        # here loads as "no crossing" rather than failing the whole terrain -
        # __post_init__ then throws out anything the wrong length or all-NaN.
        deck = _unb64_array(d.get("deck"), "float32", w) if "deck" in d else None
        climb = _unb64_array(d.get("climb"), "float32", w) if "climb" in d else None

        try:
            return cls(
                W=w,
                height=height.astype(np.float32),
                material=material.astype(np.uint8),
                seed=seed,
                style=style,
                chasm=chasm,
                deck=deck,
                climb=climb,
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
