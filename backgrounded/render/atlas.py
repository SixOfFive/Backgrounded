"""Procedurally baked sprite atlas for static props and structures.

Every sprite is drawn once at startup into its own ``SRCALPHA`` Surface and
cached by ``(kind, variant, stage, scale_bucket)``. Nothing is re-baked per
frame; ``get()`` on a warm atlas is a dict lookup.

Art direction: flat, slightly hand-drawn shapes with dark-friendly silhouettes
and a thin cool rim highlight along the upper-left edge. The whole scene is
routinely lit by nothing but one candle, so a sprite has to read as a shape
first and a texture second - the rim is what keeps a hut from dissolving into
the night.

Anchoring: every sprite's ground contact point is its **bottom centre**. Use
:meth:`Atlas.anchor` (or :meth:`Atlas.blit`) rather than assuming, so tweaks to
sprite padding cannot desync the renderer. The renderer sets
``rect.midbottom = (st.x, st.y + 2)``, so a taller sprite grows *upward* from the
ground line - every baker keeps its ground contact at the canvas bottom.

Huts are the one kind with real growth: :func:`hut_dims` recomputes the geometry
per scale instead of upscaling the small drawing, and the result is cached in
:data:`HUT_GROWTH_BUCKETS` buckets across ``HUT_SCALE_MIN..HUT_SCALE_MAX`` so a
hut creeping up by 0.001 per tick still gets a cache hit every frame.
"""
from __future__ import annotations

import math
import random
import time
import zlib
from typing import Any, Sequence

import pygame

try:                                            # pragma: no cover - import guard
    from ..constants import HUT_SCALE_MAX, HUT_SCALE_MIN
except Exception:                               # standalone / partial checkout
    HUT_SCALE_MIN, HUT_SCALE_MAX = 1.0, 1.85

Color = tuple[int, int, int]
RGBA = tuple[int, int, int, int]

# ------------------------------------------------------------------ palette --

WOOD_DARK: Color = (52, 38, 26)
WOOD: Color = (86, 62, 40)
WOOD_LIGHT: Color = (122, 92, 56)
STONE_DARK: Color = (46, 48, 52)
STONE: Color = (76, 78, 84)
STONE_LIGHT: Color = (112, 114, 122)
LEAF_DARK: Color = (28, 48, 30)
LEAF: Color = (44, 74, 42)
LEAF_LIGHT: Color = (66, 104, 58)
THATCH: Color = (110, 88, 50)
THATCH_LIGHT: Color = (148, 122, 70)
ROPE: Color = (132, 116, 84)
BERRY: Color = (192, 58, 62)
EARTH: Color = (60, 46, 32)
RIM: Color = (178, 190, 208)
RIM_A = 110                      # rim highlight alpha

# Crops: a tilled bed, green shoots that grow taller, then heavy golden ears.
# Kept in the same muted register as the leaf/wood palette so a field reads as a
# shape under a single torch rather than glowing on its own.
SOIL: Color = (74, 56, 38)
SOIL_DARK: Color = (46, 34, 24)
STALK: Color = (78, 112, 54)
STALK_DARK: Color = (46, 70, 34)
GRAIN_DARK: Color = (150, 116, 44)
GRAIN: Color = (198, 158, 66)
GRAIN_LIGHT: Color = (230, 200, 112)

# ------------------------------------------------------------------ registry --

# kind -> number of visual variants. Prop kinds match ``sim.props.KIND_*``;
# structure kinds match ``sim.structures.STRUCTURE_SPECS``. Callers may pass any
# integer variant (props use large random ints) - it wraps.
KIND_VARIANTS: dict[str, int] = {
    "tree": 4,
    "sapling": 2,
    "rock": 3,
    "boulder": 3,
    "bush": 3,
    "bush_berry": 3,        # alias: a bush that always carries berries
    "grave": 3,
    "grave_cross": 2,       # alias: forces the cross form
    "grave_stone": 2,       # alias: forces the headstone form
    "scorch": 3,
    "water": 1,             # intentionally blank; the renderer draws water itself
    "firepit": 2,
    "hut": 4,
    "wall": 3,
    "bridge": 2,
    "watchtower": 2,
    "totem": 3,
    "stockpile": 2,
    "crop": 4,              # four field looks; growth lives on the stage axis
    "barricade": 4,         # spiked palisades raised near the world edges
    "hermit_hut": 2,        # one man's lean-to, facing either way
    "hermit_fire": 2,       # his own scrape fire, two stone layouts
}

#: Hut sprite kind -> the material its walls are drawn in.
#:
#: The colony's building tech tier is a ``material`` key on ``Structure.state``,
#: not a second structure kind - a new kind would mean chasing eight literal
#: ``"hut"`` comparisons across five sim files and four hardcoded tables, all to
#: say "same building, different walls". So the material becomes a *picture*
#: here and nowhere else, exactly the way ``resolve`` already turns a bush
#: carrying berries into ``bush_berry``. Everything past ``resolve`` - the cache
#: key, the growth ladder, ``size``/``anchor`` - just sees another kind string.
#: Add a material by adding a row; :func:`_mirror_hut_aliases` wires the rest.
HUT_KINDS: dict[str, str] = {
    "hut": "wood",
    "hut_stone": "stone",
}
_HUT_KIND_FOR_MATERIAL: dict[str, str] = {m: k for k, m in HUT_KINDS.items()}

#: A crop is a prop, not a buildable, but its growth reads on the same *stage*
#: axis the atlas already caches by: stage 0 is a freshly tilled seedling bed,
#: the stages between are green shoots getting taller, and the last is heavy
#: golden ears. :meth:`Atlas.crop_stage` maps a prop's ``state`` dict onto this.
CROP_ART_STAGES = 5

# How many distinct *drawings* exist per structure kind. This is independent of
# how many build stages the sim declares: ``get`` maps the sim's stage index
# proportionally onto this range, so the two can disagree without the finished
# look ever becoming unreachable.
ART_STAGES: dict[str, int] = {
    "firepit": 3,
    "hut": 4,
    "wall": 3,
    "bridge": 3,
    "watchtower": 4,
    "totem": 3,
    "stockpile": 3,
    "grave": 2,
    "barricade": 3,
    "hermit_hut": 3,
    "hermit_fire": 2,
}

# kind -> number of build stages the simulation uses (stage 0 is stakes /
# foundation, the last is finished). Synced from sim.structures when importable.
STRUCTURE_STAGES: dict[str, int] = {
    "firepit": 3,
    "hut": 5,
    "wall": 3,
    "bridge": 4,
    "watchtower": 4,
    "totem": 3,
    "stockpile": 2,
    "grave": 2,
    "crop": CROP_ART_STAGES,   # crop growth is baked as staged art
    "barricade": 3,            # stakes -> framed -> full spiked palisade
}


def _sync_with_sim() -> None:
    """Adopt the sim's stage and variant counts when they are importable.

    Purely additive and fully guarded: if ``sim.structures`` is missing or has
    changed shape, the hardcoded tables above stand. This is what stops a
    finished stockpile from being unreachable when the sim declares fewer build
    stages than the art has drawings.
    """
    try:
        from ..sim.structures import STRUCTURE_SPECS  # noqa: PLC0415
    except Exception:
        return
    try:
        for kind, spec in STRUCTURE_SPECS.items():
            stages = int(getattr(spec, "max_stage", 0)) + 1
            if 1 <= stages <= 12:
                STRUCTURE_STAGES[kind] = stages
            variants = int(getattr(spec, "variants", 0))
            if 1 <= variants <= 16:
                KIND_VARIANTS[kind] = max(KIND_VARIANTS.get(kind, 1), variants)
            ART_STAGES.setdefault(kind, min(4, stages))
            KIND_SIZE.setdefault(kind, (48, 48))
    except Exception:
        return


def _mirror_hut_aliases() -> None:
    """Give every non-wood hut kind the same registry numbers as ``hut``.

    They are one building drawn in different walls, so they must share the
    variant count, the stage ladder, the art-stage count and the canvas. Copied
    rather than written out per material because ``_sync_with_sim`` may have
    just moved ``hut``'s numbers to whatever the sim declares: a hardcoded
    stone-hut row would silently leave the stone version one stage short of
    finished the next time somebody edits ``StructureSpec("hut", ...)``.
    """
    for kind in HUT_KINDS:
        if kind == "hut":
            continue
        KIND_VARIANTS[kind] = KIND_VARIANTS.get("hut", 4)
        STRUCTURE_STAGES[kind] = STRUCTURE_STAGES.get("hut", 5)
        ART_STAGES[kind] = ART_STAGES.get("hut", 4)
        KIND_SIZE[kind] = KIND_SIZE.get("hut", (78, 64))


# kind -> base sprite size in pixels
KIND_SIZE: dict[str, tuple[int, int]] = {
    "tree": (58, 98),
    "sapling": (20, 32),
    "rock": (28, 18),
    "boulder": (50, 34),
    "bush": (36, 26),
    "bush_berry": (36, 26),
    "crop": (26, 30),
    "grave": (24, 32),
    "grave_cross": (22, 32),
    "grave_stone": (24, 30),
    "scorch": (54, 18),
    "water": (2, 2),
    "firepit": (46, 28),
    "hut": (78, 64),
    "wall": (38, 44),
    "bridge": (108, 32),
    "watchtower": (58, 116),
    "totem": (30, 90),
    "stockpile": (52, 38),
    "barricade": (60, 58),
    # The hermit's hut. Deliberately much smaller than a VILLAGE hut (78x64)
    # and a different shape: see _bake_hermit_hut for why a lone lean-to must
    # not resolve to village housing.
    "hermit_hut": (46, 40),
    # His fire, against the colony pit's (46, 28). Two thirds the width and a
    # good deal lower: see _bake_hermit_fire for why a scrape must not resolve
    # to the settlement's hearth.
    "hermit_fire": (30, 20),
}

SCALE_BUCKET = 20.0              # scales quantise to 1/20 = 0.05 steps

# --------------------------------------------------------------- hut growth --
#: A hut's scale is a continuous sim value, so it must be quantised or the
#: renderer re-bakes a sprite every frame. Eight buckets across the documented
#: growth range is ~0.12 of scale per step: below the eye's ability to see a
#: single step, well above the rate a hut actually grows.
HUT_GROWTH_BUCKETS = 8
#: Cache-key namespace for grown huts. Well clear of the generic scale buckets
#: (capped at 160 in :meth:`Atlas.get`) so the two can never collide - a grown
#: hut and a naively upscaled one must not share a key.
HUT_BUCKET_KEY = 1000
#: Requests past ``HUT_SCALE_MAX`` keep stepping on the same ladder rather than
#: being clamped (which would silently lie about the sprite's size). The art
#: detail saturates; only the geometry keeps growing.
HUT_BUCKET_MAX = 23
#: A grown hut gains a little more height than width - taller walls read as a
#: more substantial building, where a uniform upscale just reads as "closer".
HUT_HEIGHT_GAIN = 0.10


def hut_bucket_step() -> float:
    """Scale increment between adjacent hut growth buckets."""
    span = float(HUT_SCALE_MAX) - float(HUT_SCALE_MIN)
    if span <= 0.0:
        return 0.0
    return span / max(1, HUT_GROWTH_BUCKETS - 1)


def hut_growth_bucket(scale: float) -> int:
    """Quantise a hut *scale* to its bucket index (0 == ``HUT_SCALE_MIN``)."""
    step = hut_bucket_step()
    if step <= 0.0:
        return 0
    try:
        s = float(scale)
        if not math.isfinite(s):
            return 0
        idx = int(round((s - float(HUT_SCALE_MIN)) / step))
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(idx, HUT_BUCKET_MAX))


def hut_bucket_scale(idx: int) -> float:
    """The representative scale actually drawn for bucket *idx*."""
    return float(HUT_SCALE_MIN) + hut_bucket_step() * max(0, int(idx))


def hut_growth(scale: float) -> float:
    """0..1 growth for a hut *scale*, saturating at ``HUT_SCALE_MAX``.

    This is the knob the art reads: it decides how many windows, whether there
    is a stone footing, and whether the hut has gained a second doorway.
    """
    span = float(HUT_SCALE_MAX) - float(HUT_SCALE_MIN)
    if span <= 0.0:
        return 0.0
    try:
        s = float(scale)
        if not math.isfinite(s):
            return 0.0
        return max(0.0, min(1.0, (s - float(HUT_SCALE_MIN)) / span))
    except (TypeError, ValueError, OverflowError):
        return 0.0


def hut_scale_for(growth: float) -> float:
    """Inverse of :func:`hut_growth`: map a 0..1 growth value onto a scale.

    Offered so the call site can hand the atlas a growth fraction and get the
    same mapping the art uses, instead of two places agreeing by accident.
    """
    g = 0.0
    try:
        g = max(0.0, min(1.0, float(growth)))
    except (TypeError, ValueError):
        g = 0.0
    return float(HUT_SCALE_MIN) + (float(HUT_SCALE_MAX) - float(HUT_SCALE_MIN)) * g


#: Fraction of the canvas width the walls occupy in the base drawing
#: (``0.88 - 0.12``). Held fixed so growth widens the *building*, and the extra
#: canvas width a grown hut gets is spent on roof overhang, not on the footprint.
HUT_WALL_SPAN = 0.76


def _hut_eave(s: float, g: float) -> int:
    """Roof overhang past the wall, in pixels. Grows faster than the hut does."""
    return max(1, int(round(6.0 * s + 5.0 * g * s)))


def hut_dims(scale: float) -> tuple[int, int]:
    """Canvas size of a hut at *scale*.

    Height gains a little on width (:data:`HUT_HEIGHT_GAIN`) and width gains a
    margin for the longer eaves, so a big roof is never clipped by its own
    canvas. At ``scale == 1.0`` this is exactly ``KIND_SIZE["hut"]``.
    """
    s = float(scale)
    s = max(0.05, s if math.isfinite(s) else 1.0)
    # Capped at the top of the bucket ladder: get() can never serve anything
    # bigger, so neither should the size helper the renderer lays out against.
    s = min(s, hut_bucket_scale(HUT_BUCKET_MAX))
    g = hut_growth(s)
    bw, bh = KIND_SIZE.get("hut", (78, 64))
    w = max(4, int(round(bw * s)) + int(round(4.0 * g * s)))
    h = max(4, int(round(bh * s * (1.0 + HUT_HEIGHT_GAIN * g))))
    return (w, h)


# ------------------------------------------------------------------- helpers --


def _rgba(color: Sequence[int], alpha: int = 255) -> RGBA:
    return (int(color[0]), int(color[1]), int(color[2]), int(alpha))


def _jitter(points: Sequence[tuple[float, float]], rng: random.Random,
            amount: float) -> list[tuple[int, int]]:
    """Nudge polygon vertices for a hand-drawn, non-CAD silhouette."""
    return [(int(px + rng.uniform(-amount, amount)),
             int(py + rng.uniform(-amount, amount))) for px, py in points]


def _poly(surf: pygame.Surface, color: Sequence[int], points: Sequence[Any],
          alpha: int = 255) -> None:
    if len(points) >= 3:
        pygame.draw.polygon(surf, _rgba(color, alpha), [(int(a), int(b)) for a, b in points])


def _ellipse(surf: pygame.Surface, color: Sequence[int], cx: float, cy: float,
             rx: float, ry: float, alpha: int = 255) -> None:
    rect = pygame.Rect(int(cx - rx), int(cy - ry), max(1, int(rx * 2)), max(1, int(ry * 2)))
    pygame.draw.ellipse(surf, _rgba(color, alpha), rect)


def _line(surf: pygame.Surface, color: Sequence[int], p0: Sequence[float],
          p1: Sequence[float], width: int = 1, alpha: int = 255) -> None:
    pygame.draw.line(surf, _rgba(color, alpha),
                     (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), max(1, width))


def _rim_line(surf: pygame.Surface, p0: Sequence[float], p1: Sequence[float],
              alpha: int = RIM_A, width: int = 1) -> None:
    """Thin cool highlight - the thing that keeps sprites legible in the dark."""
    _line(surf, RIM, p0, p1, width, alpha)


def _rim_arc(surf: pygame.Surface, cx: float, cy: float, rx: float, ry: float,
             start: float, end: float, alpha: int = RIM_A) -> None:
    rect = pygame.Rect(int(cx - rx), int(cy - ry), max(2, int(rx * 2)), max(2, int(ry * 2)))
    try:
        pygame.draw.arc(surf, _rgba(RIM, alpha), rect, start, end, 1)
    except Exception:
        return


def _canvas(kind: str) -> pygame.Surface:
    w, h = KIND_SIZE.get(kind, (32, 32))
    return pygame.Surface((w, h), pygame.SRCALPHA)


# --------------------------------------------------------------------------
# prop bakers
# --------------------------------------------------------------------------


def _bake_tree(variant: int, rng: random.Random) -> pygame.Surface:
    surf = _canvas("tree")
    w, h = surf.get_size()
    cx = w // 2

    if variant == 3:
        # conifer: stacked triangles on a straight bole
        _poly(surf, WOOD_DARK, [(cx - 3, h), (cx - 2, h - 26), (cx + 2, h - 26), (cx + 3, h)])
        _rim_line(surf, (cx - 3, h - 2), (cx - 2, h - 24))
        tiers = 5
        for i in range(tiers):
            t = i / (tiers - 1)
            y = h - 24 - t * (h - 40)
            half = 24 * (1.0 - t * 0.62)
            col = LEAF_DARK if i % 2 == 0 else LEAF
            pts = _jitter([(cx - half, y), (cx, y - 20), (cx + half, y)], rng, 1.6)
            _poly(surf, col, pts)
            if i == tiers - 1:
                _rim_line(surf, (cx - half * 0.5, y - 8), (cx, y - 19))
        _poly(surf, LEAF_LIGHT, [(cx - 5, h - 74), (cx, h - 88), (cx + 4, h - 74)])
        return surf

    # broadleaf: tapered trunk, a couple of limbs, layered canopy blobs
    lean = rng.uniform(-3.0, 3.0)
    trunk_h = h * rng.uniform(0.42, 0.52)
    top = h - trunk_h
    _poly(surf, WOOD_DARK, [
        (cx - 4, h), (cx - 2 + lean * 0.5, top + 6), (cx + 2 + lean * 0.5, top + 6), (cx + 4, h),
    ])
    _line(surf, WOOD, (cx + lean * 0.5, top + 8), (cx - 12 + lean, top - 4), 2)
    _line(surf, WOOD, (cx + lean * 0.5, top + 14), (cx + 12 + lean, top + 2), 2)
    _rim_line(surf, (cx - 4, h - 3), (cx - 2 + lean * 0.5, top + 8))

    blobs = 3 + variant
    for i in range(blobs):
        ang = math.tau * (i / blobs) + rng.uniform(-0.3, 0.3)
        rad = rng.uniform(6.0, 13.0)
        bx = cx + lean + math.cos(ang) * rad
        by = top - 6 + math.sin(ang) * rad * 0.62
        _ellipse(surf, LEAF_DARK, bx, by, rng.uniform(13, 19), rng.uniform(10, 15))
    for i in range(blobs - 1):
        bx = cx + lean + rng.uniform(-10, 8)
        by = top - 12 + rng.uniform(-8, 4)
        _ellipse(surf, LEAF, bx, by, rng.uniform(9, 14), rng.uniform(7, 11))
    for _ in range(2):
        bx = cx + lean + rng.uniform(-12, -2)
        by = top - 16 + rng.uniform(-8, 0)
        _ellipse(surf, LEAF_LIGHT, bx, by, rng.uniform(4, 7), rng.uniform(3, 5))
    _rim_arc(surf, cx + lean - 4, top - 12, 18, 14, 2.2, 4.3)
    return surf


def _bake_sapling(variant: int, rng: random.Random) -> pygame.Surface:
    surf = _canvas("sapling")
    w, h = surf.get_size()
    cx = w // 2
    lean = rng.uniform(-2.0, 2.0)
    _line(surf, WOOD, (cx, h - 1), (cx + lean, h * 0.30), 2)
    _rim_line(surf, (cx - 1, h - 2), (cx + lean - 1, h * 0.34), 80)
    leaves = 3 + variant
    for i in range(leaves):
        t = 0.25 + 0.7 * (i / max(1, leaves - 1))
        lx = cx + lean * t
        ly = h - t * h * 0.72
        side = -1 if i % 2 == 0 else 1
        _ellipse(surf, LEAF if i % 2 else LEAF_DARK, lx + side * 4, ly, 4.2, 2.6)
    _ellipse(surf, LEAF_LIGHT, cx + lean, h * 0.30, 3.2, 2.4)
    return surf


def _bake_rock(variant: int, rng: random.Random) -> pygame.Surface:
    surf = _canvas("rock")
    w, h = surf.get_size()
    pts = _jitter([(2, h - 1), (5, h * 0.42), (w * 0.42, 2), (w - 6, h * 0.38),
                   (w - 2, h - 1)], rng, 1.4)
    _poly(surf, STONE, pts)
    _poly(surf, STONE_LIGHT, _jitter([(6, h * 0.5), (w * 0.42, 4), (w * 0.58, h * 0.46)],
                                     rng, 1.0))
    _rim_line(surf, pts[1], pts[2], 120)
    _rim_line(surf, pts[2], pts[3], 70)
    return surf


def _bake_boulder(variant: int, rng: random.Random) -> pygame.Surface:
    surf = _canvas("boulder")
    w, h = surf.get_size()
    pts = _jitter([(1, h - 1), (3, h * 0.52), (w * 0.22, h * 0.14), (w * 0.56, 1),
                   (w - 5, h * 0.34), (w - 1, h - 1)], rng, 2.0)
    _poly(surf, STONE_DARK, pts)
    _poly(surf, STONE, _jitter([(4, h * 0.62), (w * 0.24, h * 0.18), (w * 0.56, 3),
                                (w * 0.62, h * 0.48), (w * 0.3, h - 2)], rng, 1.6))
    _poly(surf, STONE_LIGHT, _jitter([(w * 0.26, h * 0.28), (w * 0.5, h * 0.10),
                                      (w * 0.46, h * 0.44)], rng, 1.2))
    for _ in range(2 + variant):
        x0 = rng.uniform(w * 0.2, w * 0.8)
        y0 = rng.uniform(h * 0.3, h * 0.8)
        _line(surf, STONE_DARK, (x0, y0), (x0 + rng.uniform(-7, 7), y0 + rng.uniform(3, 9)), 1)
    _rim_line(surf, pts[2], pts[3], 130)
    _rim_line(surf, pts[1], pts[2], 90)
    return surf


def _bake_bush(variant: int, rng: random.Random, berries: bool) -> pygame.Surface:
    surf = _canvas("bush_berry" if berries else "bush")
    w, h = surf.get_size()
    for i in range(3 + variant % 2):
        bx = w * (0.24 + 0.24 * i) + rng.uniform(-2, 2)
        by = h * rng.uniform(0.52, 0.72)
        _ellipse(surf, LEAF_DARK, bx, by, rng.uniform(7, 10), rng.uniform(6, 9))
    for i in range(2):
        _ellipse(surf, LEAF, w * (0.34 + 0.3 * i), h * 0.48 + rng.uniform(-2, 2),
                 rng.uniform(5, 8), rng.uniform(4, 6))
    _ellipse(surf, LEAF_LIGHT, w * 0.34, h * 0.40, 4.0, 2.8)
    _rim_arc(surf, w * 0.42, h * 0.52, 12, 9, 3.0, 4.9)
    if berries:
        for _ in range(5 + variant):
            bx = rng.uniform(w * 0.16, w * 0.84)
            by = rng.uniform(h * 0.36, h * 0.80)
            pygame.draw.circle(surf, _rgba(BERRY), (int(bx), int(by)), 2)
            surf.fill(_rgba((240, 150, 150), 160), pygame.Rect(int(bx) - 1, int(by) - 1, 1, 1))
    return surf


def _crop_shoot(surf: pygame.Surface, root: tuple[float, float], sh: float,
                lean: float, grow: float, rng: random.Random) -> None:
    """One young plant: a green stem with paired blades and a bright tip.

    More blades and a longer stem as *grow* rises, so a field visibly fills in
    on its way to ripe."""
    top = (root[0] + lean, root[1] - sh)
    _line(surf, STALK_DARK, root, top, 2 if sh > 11 else 1)
    _line(surf, STALK, (root[0] - 0.6, root[1]), (top[0] - 0.6, top[1]), 1, 200)
    blades = 1 + int(round(grow * 3))
    for j in range(blades):
        u = 0.30 + 0.55 * (j / max(1, blades))
        lx = root[0] + (top[0] - root[0]) * u
        ly = root[1] + (top[1] - root[1]) * u
        side = -1.0 if j % 2 == 0 else 1.0
        blen = 2.6 + 2.2 * grow
        col = LEAF if j % 2 else LEAF_DARK
        _ellipse(surf, col, lx + side * blen * 0.55, ly, blen * 0.55, 1.4)
    _ellipse(surf, LEAF_LIGHT, top[0], top[1], 1.7, 2.3)
    _rim_line(surf, root, top, 70)


def _crop_ear(surf: pygame.Surface, root: tuple[float, float], sh: float,
              lean: float, rng: random.Random) -> None:
    """One ripe plant: a dry stem carrying a heavy, drooping golden ear."""
    shoulder = (root[0] + lean * 0.5, root[1] - sh * 0.60)
    _line(surf, GRAIN_DARK, root, shoulder, 2 if sh > 12 else 1)
    _line(surf, GRAIN, (root[0] - 0.6, root[1]), (shoulder[0] - 0.6, shoulder[1]), 1, 180)
    ear_len = sh * 0.44
    droop = rng.uniform(-2.4, 2.4)
    tip = (shoulder[0] + droop, shoulder[1] - ear_len)
    grains = 4 + int(ear_len // 3)
    for k in range(grains):
        u = k / max(1, grains - 1)
        gx = shoulder[0] + (tip[0] - shoulder[0]) * u
        gy = shoulder[1] + (tip[1] - shoulder[1]) * u
        col = GRAIN if k % 2 else GRAIN_DARK
        _ellipse(surf, col, gx, gy, 2.0, 1.5)
        if k > grains // 2:                 # awns flicking off the upper grains
            _line(surf, GRAIN_LIGHT, (gx, gy), (gx + 1.6, gy - 2.2), 1, 120)
    _ellipse(surf, GRAIN_LIGHT, tip[0], tip[1], 1.5, 1.9)
    _rim_line(surf, shoulder, tip, 90)


def _bake_crop(variant: int, stage: int, rng: random.Random) -> pygame.Surface:
    """A farmed food plant, keyed by growth stage so a field visibly ripens.

    Stage 0 is a freshly tilled mound with seedling nubs; the middle stages are
    green shoots growing taller; the final stage is heavy golden ears. Four
    variants plus per-stalk jitter keep a planted row from looking stamped, and
    it uses the same flat, rim-lit idiom as the other props so a crop still
    reads when the only light on it is a passing torch.
    """
    surf = _canvas("crop")
    w, h = surf.get_size()
    cx = w // 2
    base = h - 1
    ns = max(1, CROP_ART_STAGES)
    st = max(0, min(int(stage), ns - 1))
    ripe = st >= ns - 1
    # 0..1 how grown the greenery is; the ripe stage is as tall as the last
    # green one but spends its extra height on the ear rather than the stalk.
    grow = 1.0 if ns <= 2 else (st / float(ns - 2) if not ripe else 1.0)
    grow = max(0.0, min(1.0, grow))

    # -- tilled mound -----------------------------------------------------
    mound_h = 4.0
    _ellipse(surf, SOIL_DARK, cx, base, w * 0.46, mound_h + 1.0)
    _ellipse(surf, SOIL, cx, base - 1.0, w * 0.40, mound_h)
    for i in range(3):                       # a couple of furrow ticks
        fxp = w * (0.28 + 0.22 * i)
        _line(surf, SOIL_DARK, (fxp, base - 1), (fxp + 2, base - 3), 1, 150)
    _rim_arc(surf, cx, base - 1.0, w * 0.40, mound_h, 3.3, 6.1, 80)

    # -- the stand of plants ---------------------------------------------
    bed = base - mound_h + 1.0
    max_stalk = (h - 5) * (0.30 + 0.66 * grow)
    n = 3 + (int(variant) % 3)               # 3..5 stalks, per field
    seedling = (not ripe) and st <= 0
    for i in range(n):
        t = (i + 0.5) / n
        sx = w * 0.20 + (w * 0.60) * t + rng.uniform(-1.4, 1.4)
        lean = rng.uniform(-2.0, 2.0)
        sh = max(2.0, max_stalk * rng.uniform(0.82, 1.08))
        root = (sx, bed)
        if seedling:                          # just-planted nubs
            _ellipse(surf, LEAF_DARK, sx, bed - 1.6, 1.6, 1.8)
            _ellipse(surf, LEAF, sx + rng.uniform(-1.0, 1.0), bed - 2.6, 1.2, 1.4)
        elif ripe:
            _crop_ear(surf, root, sh, lean, rng)
        else:
            _crop_shoot(surf, root, sh, lean, grow, rng)
    return surf


def _bake_grave_cross(variant: int, rng: random.Random) -> pygame.Surface:
    surf = _canvas("grave_cross")
    w, h = surf.get_size()
    cx = w // 2
    tilt = rng.uniform(-1.6, 1.6) + (2.0 if variant else -2.0)
    _line(surf, WOOD_DARK, (cx, h - 1), (cx + tilt, 4), 4)
    _line(surf, WOOD_DARK, (cx - 7 + tilt * 0.6, 11), (cx + 7 + tilt * 0.7, 10), 3)
    _rim_line(surf, (cx - 1, h - 3), (cx + tilt - 1, 5), 90)
    _rim_line(surf, (cx - 6 + tilt * 0.6, 10), (cx + 6 + tilt * 0.7, 9), 70)
    _ellipse(surf, EARTH, cx, h - 2, w * 0.34, 2.4)
    return surf


def _bake_grave_stone(variant: int, rng: random.Random) -> pygame.Surface:
    surf = _canvas("grave_stone")
    w, h = surf.get_size()
    cx = w // 2
    top = 6 + variant * 2
    body = pygame.Rect(int(w * 0.22), top, int(w * 0.56), h - top - 2)
    pygame.draw.rect(surf, _rgba(STONE), body)
    _ellipse(surf, STONE, cx, top, w * 0.28, 5.0)
    _poly(surf, STONE_DARK, [(cx + 2, top + 2), (int(w * 0.78), top + 4),
                             (int(w * 0.78), h - 3), (cx + 2, h - 3)])
    for i in range(2):
        _line(surf, STONE_DARK, (w * 0.34, top + 10 + i * 5), (w * 0.62, top + 10 + i * 5), 1)
    _rim_line(surf, (w * 0.22, top + 2), (w * 0.22, h - 3), 110)
    _rim_arc(surf, cx, top, w * 0.28, 5.0, 3.2, 6.1, 120)
    _ellipse(surf, EARTH, cx, h - 2, w * 0.38, 2.4)
    return surf


def _bake_scorch(variant: int, rng: random.Random) -> pygame.Surface:
    surf = _canvas("scorch")
    w, h = surf.get_size()
    cx, cy = w // 2, h - 5
    _ellipse(surf, (18, 14, 12), cx, cy, w * 0.44, h * 0.34, 190)
    for _ in range(4 + variant * 2):
        bx = cx + rng.uniform(-w * 0.42, w * 0.42)
        by = cy + rng.uniform(-h * 0.28, h * 0.24)
        _ellipse(surf, (26, 20, 18), bx, by, rng.uniform(3, 8), rng.uniform(2, 4), 150)
    for _ in range(6):
        bx = cx + rng.uniform(-w * 0.48, w * 0.48)
        by = cy + rng.uniform(-h * 0.3, h * 0.3)
        surf.fill(_rgba((72, 66, 62), 120), pygame.Rect(int(bx), int(by), 1, 1))
    return surf


# --------------------------------------------------------------------------
# structure bakers - stage 0 is stakes/foundation, the last stage is finished
# --------------------------------------------------------------------------


def _bake_firepit(stage: int, rng: random.Random) -> pygame.Surface:
    surf = _canvas("firepit")
    w, h = surf.get_size()
    cx, cy = w // 2, h - 6
    _ellipse(surf, EARTH, cx, cy + 2, w * 0.40, 5.0)
    ring = 3 if stage == 0 else 8
    for i in range(ring):
        a = math.tau * i / max(1, ring) + 0.2
        sx = cx + math.cos(a) * w * 0.36
        sy = cy + math.sin(a) * 6.0
        _ellipse(surf, STONE, sx, sy, 4.0, 3.0)
        _rim_arc(surf, sx, sy, 4.0, 3.0, 3.1, 6.0, 90)
    if stage >= 1:
        for i in range(3):
            x0 = cx - 8 + i * 8
            _line(surf, WOOD, (x0, cy + 1), (x0 + rng.uniform(-4, 4), cy - 9), 2)
    if stage >= 2:
        for i in range(3):
            _line(surf, WOOD_DARK, (cx - 10 + i * 3, cy - 2 - i * 3),
                  (cx + 10 - i * 2, cy - 4 - i * 3), 3)
        _line(surf, WOOD_LIGHT, (cx - 9, cy - 3), (cx + 9, cy - 5), 1)
        _rim_line(surf, (cx - 9, cy - 4), (cx + 8, cy - 6), 90)
    return surf


def _hut_window(surf: pygame.Surface, cx: float, cy: float, ww: float, wh: float,
                s: float, mullion: bool = False,
                lintel: Color = WOOD, sill: Color = WOOD_DARK) -> None:
    """A dark opening with a lintel and sill, rim-lit on the upper left.

    *lintel* / *sill* default to timber; a masonry wall passes stone, because a
    wood-brown lintel sitting in grey courses reads as a mistake rather than as
    a different material.
    """
    unit = max(1, int(round(s)))
    rect = pygame.Rect(int(cx - ww * 0.5), int(cy - wh * 0.5),
                       max(2, int(round(ww))), max(2, int(round(wh))))
    pygame.draw.rect(surf, _rgba((16, 12, 10), 230), rect)
    _line(surf, lintel, (rect.left - unit, rect.top - unit),
          (rect.right + unit, rect.top - unit), max(1, int(round(1.6 * s))))
    _line(surf, sill, (rect.left - unit, rect.bottom),
          (rect.right + unit, rect.bottom), max(1, int(round(1.4 * s))))
    _rim_line(surf, rect.topleft, rect.bottomleft, 95)
    _rim_line(surf, (rect.left, rect.top - 1), (rect.right, rect.top - 1), 70)
    if mullion:
        _line(surf, lintel, (rect.centerx, rect.top), (rect.centerx, rect.bottom), unit, 220)


def _hut_masonry(surf: pygame.Surface, left: int, right: int, top: int,
                 bottom: int, s: float, jog: float) -> None:
    """Fill a wall rectangle with staggered stone courses.

    Deliberately the same vocabulary as :func:`_bake_wall` - alternating
    ``STONE``/``STONE_DARK`` blocks, a dark mortar outline, every other course
    offset, a faint rim along each course line - but sized from the wall it is
    filling rather than from that baker's fixed canvas. A stone hut and a stone
    wall standing beside each other have to read as the same masonry, or the
    hut looks like it wandered in from another game.

    *jog* is the per-variant stagger, a fraction of a block: two stone huts
    side by side with identical coursing look like one building stamped twice.
    """
    if right - left < 2 or bottom - top < 2:
        return
    pygame.draw.rect(surf, _rgba(STONE_DARK),
                     pygame.Rect(left, top, right - left, bottom - top))
    ch = max(3, int(round(7.0 * s)))            # course height
    bw = max(4, int(round(11.0 * s)))           # block width
    course = 0
    y = bottom
    while y > top:
        ty = max(top, y - ch + 1)
        bx = left - int(round(bw * jog)) * (course % 2)
        col = 0
        while bx < right:
            x0, x1 = max(left, bx), min(right, bx + bw - 1)
            if x1 - x0 > 1 and y - ty > 0:
                rect = pygame.Rect(x0, ty, x1 - x0, y - ty)
                pygame.draw.rect(surf, _rgba(STONE if (course + col) % 2 else STONE_DARK), rect)
                pygame.draw.rect(surf, _rgba(STONE_DARK, 200), rect, 1)
            bx += bw
            col += 1
        _rim_line(surf, (left, ty), (right, ty), 55)
        y -= ch
        course += 1


def _bake_hut(stage: int, rng: random.Random, scale: float = 1.0,
              material: str = "wood") -> pygame.Surface:
    """Bake a hut at *scale*, geometry recomputed rather than upscaled.

    Growth is not zoom. As ``scale`` climbs the walls take a bigger share of a
    taller canvas, the eaves reach further past them, and the building earns
    structure the small hut does not have: a stone footing, then a window, then a
    second window and a smoke hole, and at the top of the range a second doorway
    - two dwellings under one roof.

    *material* swaps the walls, not the building. Stone gets coursed masonry
    (:func:`_hut_masonry`), stone quoins, stone lintels and shorter eaves -
    masonry needs less overhang to keep the rain off than daub does, and that
    shorter roof is what tells the two apart at 30 px across in the dark. The
    roof frame and the thatch stay timber in both: you cannot make a rafter out
    of stone, and keeping the roof shared means every growth gate above
    (footing, windows, smoke hole, second doorway) composes with either wall
    without a second copy of the ladder.

    The caller seeds *rng* per ``(kind, variant, stage)``, so a hut keeps its own
    jitter as it grows; details must not pop when it crosses a bucket. Vertical
    geometry is a fraction of the canvas, horizontal geometry a fraction of the
    wall span (see ``fx``), and the ground contact stays at the canvas bottom,
    because the renderer anchors ``midbottom`` - a taller hut has to grow upward,
    not straddle the ground line.
    """
    s = max(0.05, float(scale))
    g = hut_growth(s)
    w, h = hut_dims(s)
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    stone = material == "stone"

    # Per-variant character, drawn up front and unconditionally. The variant
    # axis existed and was wired all the way through the sim, saves and the
    # cache key, but nothing in here ever touched `rng` past the scaffolding
    # stages - all four finished huts baked to the same bytes (md5 3e131ce4).
    # Drawing the knobs before any branch is what makes them safe: consume the
    # stream inside a `if g > ...` and the same hut re-rolls its character the
    # moment it crosses a growth bucket, which is the popping the docstring
    # above promises not to do. Fractions throughout, never pixel counts, for
    # the same reason.
    pitch = rng.uniform(-0.026, 0.028)      # ridge height, fraction of canvas
    overhang = rng.uniform(0.78, 1.12)      # eave length multiplier
    door_f = rng.uniform(0.21, 0.28)        # doorway width, fraction of the wall
    win_t = rng.uniform(0.15, 0.27)         # where the near window sits
    jog = rng.uniform(0.28, 0.62)           # masonry course stagger
    lean = rng.uniform(-1.0, 1.0)           # how badly the stakes were driven
    extra = rng.randint(0, 1)               # one more wall band / thatch course
    far_t = 1.0 - win_t                     # the far window, and later the far door

    def lw(n: float) -> int:
        """A base-art line width in pixels, thickened with the sprite."""
        return max(1, int(round(n * s)))

    # Walls span a fixed fraction of the *base* width and sit centred, so the
    # extra canvas width from hut_dims lands under the eaves instead of inside
    # the building. At scale 1.0 this reproduces the original left=9, right=68.
    bw = KIND_SIZE.get("hut", (78, 64))[0]
    wall_span = max(4, int(round(HUT_WALL_SPAN * bw * s)))
    left = max(1, (w - wall_span) // 2)
    right = min(w - 2, left + wall_span)
    base = h - lw(2)
    # Walls claim more of the height as the hut grows (0.54h -> 0.60h) and the
    # ridge lifts slightly, so the roof reads long and low instead of pointy.
    wall_top = int(h * (0.46 - 0.06 * g))
    ridge = max(1, int(h * (0.12 - 0.02 * g + pitch)))
    wall_h = max(2, base - wall_top)
    cx = w // 2
    # Clamped to the canvas: a roof that overhangs past its own surface gets its
    # corners sliced off, which is exactly the detail the rim highlight sells.
    # Stone carries a shorter eave - see the material note in the docstring.
    reach = int(round(_hut_eave(s, g) * overhang * (0.78 if stone else 1.0)))
    eave = max(1, min(reach, left - 1, w - right - 2))
    # Quoins on a stone hut, corner posts on a timber one.
    post = STONE_DARK if stone else WOOD_DARK
    lintel = STONE_LIGHT if stone else WOOD
    sill = STONE if stone else WOOD_DARK

    def fx(t: float) -> int:
        """Horizontal position as a fraction *across the walls*, not the canvas.

        Wall-relative so the composition is identical at every scale even though
        a grown canvas carries extra padding for its eaves.
        """
        return int(left + wall_span * t)

    if stage == 0:
        # A timber hut starts as driven stakes; a stone one starts as squat set
        # footing blocks, which is the only cue the material gets this early.
        post_h = int(h * (0.13 if stone else 0.22))
        for x in (left, fx(0.32), fx(0.68), right):
            # Per-post scruff on top of the variant's own slant. Safe to draw
            # from the stream here: nothing in this branch is gated on growth,
            # so the four tilts come out the same at every bucket.
            tilt = (lean + rng.uniform(-0.8, 0.8)) * s
            _line(surf, post, (x, base), (x + tilt, base - post_h), lw(5 if stone else 3))
            _rim_line(surf, (x - 1, base - lw(2)), (x - 1, base - post_h + lw(1)), 80)
        _ellipse(surf, EARTH, cx, base, w * 0.42, 3.0 * s)
        return surf

    # posts. The wall plate across the top stays timber whatever the walls are
    # made of - it is what the rafters land on.
    for x in (left, right):
        _line(surf, post, (x, base), (x, wall_top), lw(4))
    _line(surf, WOOD_DARK, (left, wall_top), (right, wall_top), lw(3))

    if stage == 1:
        # frame only: rafters and a ridge beam
        _line(surf, WOOD, (left, wall_top), (cx, ridge), lw(3))
        _line(surf, WOOD, (right, wall_top), (cx, ridge), lw(3))
        _line(surf, WOOD, (fx(0.25), wall_top), (cx, ridge + lw(6)), lw(2))
        _line(surf, WOOD, (fx(0.75), wall_top), (cx, ridge + lw(6)), lw(2))
        _rim_line(surf, (left, wall_top - 1), (cx, ridge - 1), 100)
        _ellipse(surf, EARTH, cx, base, w * 0.42, 3.0 * s)
        return surf

    # walls
    if stone:
        _hut_masonry(surf, left, right, wall_top, base, s, jog)
    else:
        _poly(surf, WOOD_DARK,
              [(left, base), (left, wall_top), (right, wall_top), (right, base)])
        bands = 4 + int(round(2.0 * g)) + extra
        span = max(2, wall_h // (bands + 1))
        for i in range(bands):
            y = wall_top + lw(5) + i * span
            if y >= base - 1:
                break
            _line(surf, WOOD, (left + lw(2), y), (right - lw(2), y), 1, 150)

    # A grown hut sits on stone. Drawn before the openings so the doorway cuts
    # through it rather than standing on a plinth. On masonry the plinth has to
    # run a shade *lighter* than the wall: the STONE_DARK footing that grounds a
    # timber hut just disappears into the courses above it.
    if g > 0.15:
        fh = lw(4) + int(round(3.0 * g * s))
        plinth, cobble = (STONE, STONE_LIGHT) if stone else (STONE_DARK, STONE)
        pygame.draw.rect(surf, _rgba(plinth),
                         pygame.Rect(left, base - fh, max(2, right - left), fh))
        step = max(4, int(round(9.0 * s)))
        for bx in range(left, max(left + 1, right - 2), step):
            _ellipse(surf, cobble, bx + step * 0.5, base - fh * 0.55, step * 0.34, fh * 0.34)
        _rim_line(surf, (left, base - fh), (right, base - fh), 90)

    if stage == 2:
        # partially thatched roof: rafters showing through
        _poly(surf, THATCH, [(left - eave, wall_top + lw(2)), (cx, ridge),
                             (cx + lw(2), ridge + lw(3)), (fx(0.12), wall_top + lw(2))])
        _line(surf, WOOD, (right + lw(4), wall_top + lw(2)), (cx, ridge), lw(3))
        _line(surf, WOOD, (fx(0.80), wall_top + lw(2)), (cx, ridge + lw(5)), lw(2))
        _rim_line(surf, (left - eave + lw(1), wall_top + 1), (cx, ridge - 1), 110)
    else:
        # finished: full thatch roof, ridge cap, doorway
        lspan = float(wall_top + lw(3) - ridge)
        _poly(surf, THATCH, [(left - eave, wall_top + lw(3)), (cx, ridge),
                             (right + eave, wall_top + lw(3))])
        courses = 5 + int(round(3.0 * g)) + extra
        for i in range(courses):
            t = 0.18 + (0.80 / courses) * i
            ly = wall_top + lw(3) - lspan * t
            _line(surf, THATCH_LIGHT,
                  (left - eave + (cx - left + eave) * t, ly),
                  (right + eave - (right + eave - cx) * t, ly), lw(1), 90)
        _line(surf, WOOD_LIGHT, (cx - lw(4), ridge + 1), (cx + lw(4), ridge + 1), lw(2))
        _rim_line(surf, (left - eave, wall_top + lw(2)), (cx, ridge - 1), 130)
        _rim_line(surf, (cx, ridge - 1), (right + eave, wall_top + lw(2)), 60)

        if g > 0.55:
            # a hearth big enough to need venting
            hx = cx + int(wall_span * 0.18)
            hy = wall_top + lw(3) - lspan * 0.42
            _poly(surf, (18, 14, 12), [(hx - lw(4), hy), (hx + lw(4), hy - lw(2)),
                                       (hx + lw(4), hy + lw(4)), (hx - lw(4), hy + lw(5))], 220)
            _line(surf, THATCH_LIGHT, (hx - lw(5), hy - lw(3)),
                  (hx + lw(5), hy - lw(5)), lw(2), 200)
            _rim_line(surf, (hx - lw(5), hy - lw(3)), (hx + lw(5), hy - lw(5)), 100)

        # Wall furniture. The main doorway stays centred at every size: the
        # renderer puts its occupancy glow at rect.centerx.
        win_y = wall_top + wall_h * 0.34
        win_w, win_h = wall_span * 0.17, wall_h * 0.30
        if g > 0.30:
            _hut_window(surf, fx(win_t), win_y, win_w, win_h, s, g > 0.70, lintel, sill)
        if 0.55 < g <= 0.78:
            _hut_window(surf, fx(far_t), win_y, win_w, win_h, s, g > 0.70, lintel, sill)

        dw = max(6, int(round(wall_span * door_f)))
        dh = max(8, int(round(wall_h * 0.58)))
        door = pygame.Rect(cx - dw // 2, base - dh - lw(1), dw, dh)
        pygame.draw.rect(surf, _rgba((16, 12, 10), 235), door)
        if stone:
            # A masonry opening cannot span itself; the timber hut's doorway is
            # a gap between studs and needs nothing.
            _line(surf, lintel, (door.left - lw(2), door.top - lw(1)),
                  (door.right + lw(2), door.top - lw(1)), lw(2))
        _rim_line(surf, (door.left - 1, door.top - 1), (door.left - 1, base - lw(2)), 90)
        _rim_line(surf, (door.left - 1, door.top - 1), (door.right - 1, door.top - 1), 70)

        if g > 0.78:
            # top of the range: a second doorway, plus a gable window above the
            # main one - the hut has become a longhouse for two families. The
            # second door lands exactly where the second window was, so the
            # crossover at g == 0.78 reads as the opening being enlarged.
            d2w = max(5, int(round(wall_span * 0.17)))
            d2h = max(7, int(round(wall_h * 0.50)))
            d2 = pygame.Rect(fx(far_t) - d2w // 2, base - d2h - lw(1), d2w, d2h)
            pygame.draw.rect(surf, _rgba((16, 12, 10), 235), d2)
            _line(surf, lintel, (d2.left - lw(1), d2.top - lw(1)),
                  (d2.right + lw(1), d2.top - lw(1)), lw(2))
            _rim_line(surf, (d2.left - 1, d2.top - 1), (d2.left - 1, base - lw(2)), 85)
            _rim_line(surf, (d2.left - 1, d2.top - 1), (d2.right - 1, d2.top - 1), 65)
            _hut_window(surf, cx, wall_top + wall_h * 0.20, wall_span * 0.13,
                        wall_h * 0.16, s, False, lintel, sill)

    _ellipse(surf, EARTH, cx, base, w * 0.44, 3.0 * s)
    return surf


def _bake_wall(stage: int, rng: random.Random) -> pygame.Surface:
    surf = _canvas("wall")
    w, h = surf.get_size()
    base = h - 2
    if stage == 0:
        for x in (int(w * 0.2), int(w * 0.8)):
            _line(surf, WOOD_DARK, (x, base), (x, base - 12), 3)
            _rim_line(surf, (x - 1, base - 2), (x - 1, base - 11), 80)
        _line(surf, ROPE, (int(w * 0.2), base - 11), (int(w * 0.8), base - 12), 1, 180)
        _ellipse(surf, EARTH, w // 2, base, w * 0.42, 2.6)
        return surf

    courses = 2 if stage == 1 else 4
    ch = 8
    for c in range(courses):
        y = base - 2 - c * ch
        offset = (c % 2) * 5
        for i in range(3):
            bx = int(w * 0.14) + offset + i * 10
            rect = pygame.Rect(bx, y - ch + 1, 9, ch - 1)
            pygame.draw.rect(surf, _rgba(STONE if (c + i) % 2 else STONE_DARK), rect)
            pygame.draw.rect(surf, _rgba(STONE_DARK, 200), rect, 1)
        _rim_line(surf, (int(w * 0.14) + offset, y - ch + 1),
                  (int(w * 0.14) + offset + 28, y - ch + 1), 70)
    if stage >= 2:
        cap_y = base - 2 - courses * ch
        pygame.draw.rect(surf, _rgba(STONE_LIGHT),
                         pygame.Rect(int(w * 0.10), cap_y - 4, int(w * 0.80), 5))
        _rim_line(surf, (int(w * 0.10), cap_y - 4), (int(w * 0.90), cap_y - 4), 130)
    _ellipse(surf, EARTH, w // 2, base, w * 0.44, 2.6)
    return surf


def _bake_bridge(stage: int, rng: random.Random) -> pygame.Surface:
    surf = _canvas("bridge")
    w, h = surf.get_size()
    deck = int(h * 0.46)
    base = h - 2
    for x in (4, w - 5):
        _line(surf, WOOD_DARK, (x, base), (x, deck - 8), 4)
        _rim_line(surf, (x - 1, base - 2), (x - 1, deck - 7), 90)

    def sag(t: float, amount: float) -> float:
        return deck + amount * math.sin(math.pi * t)

    if stage == 0:
        pts = [(4 + (w - 9) * (i / 12), sag(i / 12, 9.0)) for i in range(13)]
        pygame.draw.lines(surf, _rgba(ROPE, 220), False,
                          [(int(a), int(b)) for a, b in pts], 2)
        return surf

    planks = 5 if stage == 1 else 13
    top = [(4 + (w - 9) * (i / 12), sag(i / 12, 6.0)) for i in range(13)]
    pygame.draw.lines(surf, _rgba(ROPE, 230), False,
                      [(int(a), int(b)) for a, b in top], 2)
    for i in range(planks):
        t = (i + 0.5) / planks
        px = 4 + (w - 9) * t
        py = sag(t, 6.0)
        rect = pygame.Rect(int(px - 3), int(py - 2), 6, 5)
        pygame.draw.rect(surf, _rgba(WOOD if i % 2 else WOOD_DARK), rect)
    if stage >= 2:
        rail = [(4 + (w - 9) * (i / 12), sag(i / 12, 6.0) - 12) for i in range(13)]
        pygame.draw.lines(surf, _rgba(WOOD, 240), False,
                          [(int(a), int(b)) for a, b in rail], 2)
        pygame.draw.lines(surf, _rgba(RIM, 90), False,
                          [(int(a), int(b - 1)) for a, b in rail], 1)
        for i in range(0, 13, 3):
            _line(surf, WOOD_DARK, rail[i], top[i], 1)
    return surf


def _bake_watchtower(stage: int, rng: random.Random) -> pygame.Surface:
    surf = _canvas("watchtower")
    w, h = surf.get_size()
    base = h - 2
    lx, rx = int(w * 0.16), int(w * 0.84)
    plat = int(h * 0.34)

    if stage == 0:
        for x in (lx, rx, int(w * 0.34), int(w * 0.66)):
            _ellipse(surf, STONE_DARK, x, base - 2, 5.0, 3.5)
            _line(surf, WOOD_DARK, (x, base), (x, base - 8), 3)
        _ellipse(surf, EARTH, w // 2, base, w * 0.44, 3.0)
        return surf

    top_y = plat if stage >= 2 else int(h * 0.52)
    _line(surf, WOOD_DARK, (lx, base), (int(w * 0.34), top_y), 4)
    _line(surf, WOOD_DARK, (rx, base), (int(w * 0.66), top_y), 4)
    _rim_line(surf, (lx - 1, base - 2), (int(w * 0.34) - 1, top_y), 100)

    if stage >= 2:
        for i in range(3):
            y = base - (base - top_y) * (i + 1) / 4
            _line(surf, WOOD, (lx + (int(w * 0.34) - lx) * (i / 3), y),
                  (rx - (rx - int(w * 0.66)) * (i / 3), y), 2)
        # ladder
        for i in range(6):
            y = base - 6 - i * ((base - top_y) / 7)
            _line(surf, WOOD_LIGHT, (w * 0.42, y), (w * 0.58, y), 1, 200)
        _line(surf, WOOD, (w * 0.42, base), (w * 0.44, top_y), 1)
        _line(surf, WOOD, (w * 0.58, base), (w * 0.56, top_y), 1)

    if stage >= 2:
        pygame.draw.rect(surf, _rgba(WOOD_DARK),
                         pygame.Rect(int(w * 0.14), top_y - 5, int(w * 0.72), 6))
        _rim_line(surf, (int(w * 0.14), top_y - 5), (int(w * 0.86), top_y - 5), 130)
    if stage >= 3:
        for x in (int(w * 0.18), int(w * 0.5), int(w * 0.82)):
            _line(surf, WOOD, (x, top_y - 5), (x, top_y - 18), 2)
        _line(surf, WOOD, (int(w * 0.16), top_y - 18), (int(w * 0.84), top_y - 18), 2)
        roof = [(int(w * 0.08), top_y - 20), (w // 2, int(h * 0.06)),
                (int(w * 0.92), top_y - 20)]
        _poly(surf, THATCH, roof)
        _rim_line(surf, roof[0], roof[1], 130)
        _line(surf, WOOD_LIGHT, (w // 2 - 4, int(h * 0.06) + 2),
              (w // 2 + 4, int(h * 0.06) + 2), 2)
    _ellipse(surf, EARTH, w // 2, base, w * 0.44, 3.0)
    return surf


def _bake_totem(stage: int, rng: random.Random) -> pygame.Surface:
    surf = _canvas("totem")
    w, h = surf.get_size()
    cx = w // 2
    base = h - 2
    top = int(h * (0.30 if stage == 0 else 0.10))
    pygame.draw.rect(surf, _rgba(WOOD_DARK),
                     pygame.Rect(cx - 6, top, 12, base - top))
    _rim_line(surf, (cx - 6, top + 1), (cx - 6, base - 1), 110)
    _ellipse(surf, EARTH, cx, base, w * 0.42, 3.0)
    if stage == 0:
        return surf

    faces = 2 if stage == 1 else 3
    span = (base - top) / (faces + 0.4)
    for i in range(faces):
        fy = top + 6 + i * span
        pygame.draw.rect(surf, _rgba(WOOD), pygame.Rect(cx - 6, int(fy), 12, int(span * 0.7)))
        surf.fill(_rgba((14, 12, 10), 230), pygame.Rect(cx - 4, int(fy + 4), 3, 3))
        surf.fill(_rgba((14, 12, 10), 230), pygame.Rect(cx + 1, int(fy + 4), 3, 3))
        _line(surf, WOOD_DARK, (cx - 4, fy + 11), (cx + 4, fy + 11), 2)
        _rim_line(surf, (cx - 6, fy), (cx + 6, fy), 80)
    if stage >= 2:
        _poly(surf, WOOD, [(cx - 6, top + 4), (cx - 15, top - 2), (cx - 6, top + 12)])
        _poly(surf, WOOD, [(cx + 6, top + 4), (cx + 15, top - 2), (cx + 6, top + 12)])
        _rim_line(surf, (cx - 6, top + 4), (cx - 14, top - 1), 110)
        _poly(surf, BERRY, [(cx - 7, top), (cx, top - 7), (cx + 7, top)])
        _rim_line(surf, (cx - 6, top - 1), (cx, top - 7), 130)
    return surf


def _bake_grave_structure(stage: int, rng: random.Random) -> pygame.Surface:
    """A grave marker. Stage 0 already reads as a finished grave, because props
    place completed graves at stage 0 while the build director walks 0 -> 1."""
    surf = _canvas("grave")
    w, h = surf.get_size()
    cx = w // 2
    base = h - 2
    _ellipse(surf, EARTH, cx, base, w * 0.42, 3.4)
    # the rng is seeded per variant, so half the markers are crosses
    if stage <= 0 and rng.random() < 0.5:
        tilt = rng.uniform(-1.5, 1.5)
        _line(surf, WOOD_DARK, (cx, base - 1), (cx + tilt, h * 0.28), 4)
        _line(surf, WOOD_DARK, (cx - 6 + tilt * 0.5, h * 0.40),
              (cx + 6 + tilt * 0.6, h * 0.39), 3)
        _rim_line(surf, (cx - 1, base - 3), (cx + tilt - 1, h * 0.29), 95)
        _rim_line(surf, (cx - 5 + tilt * 0.5, h * 0.39), (cx + 5 + tilt * 0.6, h * 0.38), 70)
        return surf

    top = int(h * (0.30 if stage <= 0 else 0.22))
    body = pygame.Rect(int(w * 0.24), top, int(w * 0.52), base - top - 1)
    pygame.draw.rect(surf, _rgba(STONE), body)
    _ellipse(surf, STONE, cx, top, w * 0.26, 5.0)
    _poly(surf, STONE_DARK, [(cx + 1, top + 2), (int(w * 0.76), top + 4),
                             (int(w * 0.76), base - 2), (cx + 1, base - 2)])
    for i in range(2):
        _line(surf, STONE_DARK, (w * 0.34, top + 11 + i * 5), (w * 0.62, top + 11 + i * 5), 1)
    if stage >= 1:
        for i in range(3):
            _ellipse(surf, STONE_DARK, cx - 8 + i * 8, base - 2, 4.0, 2.4)
    _rim_line(surf, (w * 0.24, top + 2), (w * 0.24, base - 2), 115)
    _rim_arc(surf, cx, top, w * 0.26, 5.0, 3.2, 6.1, 125)
    return surf


def _bake_stockpile(stage: int, rng: random.Random) -> pygame.Surface:
    surf = _canvas("stockpile")
    w, h = surf.get_size()
    base = h - 2
    _ellipse(surf, EARTH, w // 2, base, w * 0.44, 3.5)
    for x in (4, w - 5):
        _line(surf, WOOD_DARK, (x, base), (x, base - 9), 2)
    _line(surf, ROPE, (4, base - 9), (w - 5, base - 9), 1, 170)
    if stage == 0:
        return surf

    logs = 3 if stage == 1 else 6
    for i in range(logs):
        row = i // 3
        col = i % 3
        lx = 10 + col * 11
        ly = base - 4 - row * 7
        pygame.draw.rect(surf, _rgba(WOOD if i % 2 else WOOD_DARK),
                         pygame.Rect(lx, int(ly - 5), 10, 6))
        _ellipse(surf, WOOD_LIGHT, lx + 10, ly - 2, 2.0, 3.0)
        _rim_line(surf, (lx, ly - 5), (lx + 10, ly - 5), 70)
    if stage >= 2:
        for i in range(3):
            sx = w - 20 + (i % 2) * 8
            sy = base - 5 - (i // 2) * 8
            _ellipse(surf, STONE, sx, sy, 5.0, 4.0)
            _rim_arc(surf, sx, sy, 5.0, 4.0, 3.1, 5.9, 90)
        crate = pygame.Rect(int(w * 0.36), base - 22, 14, 12)
        pygame.draw.rect(surf, _rgba(WOOD), crate)
        pygame.draw.rect(surf, _rgba(WOOD_DARK, 220), crate, 1)
        _line(surf, WOOD_LIGHT, crate.topleft, crate.bottomright, 1, 140)
        _rim_line(surf, crate.topleft, crate.topright, 120)
    return surf


def _stake(surf: pygame.Surface, x: float, base: float, top_y: float,
           half: float, lean: float = 0.0, point: float = 0.26) -> tuple[float, float]:
    """One driven stake, drawn as a filled post with a sharpened tip.

    ``point`` is the fraction of the stake's height taken up by the sharpened
    head: ~0.26 is a wicked spike, ~0.10 a stake only starting to be worked to a
    point, and <=0.02 a blunt, freshly-sawn top. ``lean`` tilts the head sideways
    so a row of these can fan outward. The lit upper-left edge and the near face
    of the point carry the cool rim highlight the whole atlas leans on to stay
    legible under a single torch. Returns the tip point.
    """
    tx = x + lean
    span = max(1.0, base - top_y)
    pf = max(0.0, min(0.5, point))
    shoulder_y = top_y + span * pf
    band = max(1, int(round(half * 0.7)))
    if pf <= 0.02:                                  # blunt: a plain driven post
        _poly(surf, WOOD_DARK, [(x - half, base), (x - half, top_y),
                                (x + half, top_y), (x + half, base)])
        _line(surf, WOOD, (x - half * 0.15, base - 1), (x - half * 0.15, top_y), band, 150)
        _rim_line(surf, (x - half, base - 2), (x - half, top_y + 1), 90)
        _ellipse(surf, WOOD, x, top_y, max(1.2, half * 0.9), 1.4)
        return (x, top_y)
    _poly(surf, WOOD_DARK, [
        (x - half, base), (x - half * 0.72, shoulder_y), (tx, top_y),
        (x + half * 0.72, shoulder_y), (x + half, base),
    ])
    _line(surf, WOOD, (x - half * 0.15, base - 1),
          (tx - half * 0.15, shoulder_y), band, 150)
    _rim_line(surf, (x - half, base - 2), (x - half * 0.72, shoulder_y), 90)
    _rim_line(surf, (x - half * 0.72, shoulder_y), (tx, top_y), 120)
    return (tx, top_y)


def _bake_barricade(stage: int, rng: random.Random) -> pygame.Surface:
    """A spiked defensive palisade the colony raises near a world edge.

    Stage 0 is a scraped trench with a couple of short driven stakes; the middle
    stage frames a row of posts and lashes a crossbar across them, their tips
    only starting to be worked to a point; the finished stage is a dense row of
    sharpened posts lashed with crossbars, sharp points at the top and the
    flanking stakes splayed *outward* on both sides so the barricade reads as a
    spiked barrier from whichever edge it guards - the renderer draws it without
    flipping, so "angled outward" has to hold for the left edge and the right
    edge at once.

    Same flat, rim-lit idiom as the other structures so it holds its silhouette
    when a passing torch is the only light on it, and it grows upward from the
    canvas bottom, because the renderer anchors ``midbottom`` on the ground line.
    """
    surf = _canvas("barricade")
    w, h = surf.get_size()
    cx = w // 2
    base = h - 2
    st = max(0, min(int(stage), 2))
    inner_l = int(w * 0.18)
    inner_r = int(w * 0.82)

    # turned earth heaped at the foot, common to every stage
    _ellipse(surf, EARTH, cx, base, w * 0.48, 3.0)

    if st == 0:
        # foundation: a scraped trench line and a few short, blunt driven stakes
        _line(surf, SOIL_DARK, (inner_l, base - 1), (inner_r, base - 1), 2, 170)
        for t in (0.28, 0.5, 0.74):
            sx = int(inner_l + (inner_r - inner_l) * t) + int(rng.uniform(-1, 1))
            ph = base - int(h * (0.20 + 0.12 * rng.random()))
            _stake(surf, sx, base, ph, 2.4, lean=rng.uniform(-1.0, 1.0), point=0.0)
        # a spare stake laid ready beside the works, plus a mallet chip
        _line(surf, WOOD, (int(w * 0.30), base - 1), (int(w * 0.56), base - 2), 2, 170)
        _ellipse(surf, WOOD_LIGHT, int(w * 0.62), base - 1, 1.8, 1.2)
        return surf

    full = st >= 2
    top_full = int(h * (0.14 if full else 0.42))
    n = (6 if full else 4) + (1 if rng.random() < 0.5 else 0)
    n = max(3, n)
    half = 2.5 if full else 2.2
    point = 0.28 if full else 0.11
    fan = 6.5 if full else 2.4

    posts: list[tuple[int, float]] = []
    for i in range(n):
        t = i / (n - 1)
        px = int(inner_l + (inner_r - inner_l) * t)
        spread = (t - 0.5) * 2.0                     # -1 at left, +1 at right
        lean = spread * (fan + 2.2 * rng.random())   # outer stakes splay outward
        top_y = top_full + rng.uniform(0.0, h * 0.05)
        _stake(surf, px, base, top_y, half, lean=lean, point=point)
        posts.append((px, top_y))

    # lashed crossbar(s): a horizontal timber with a rope wrap at each post
    bars = 2 if (full and rng.random() < 0.6) else 1
    for k in range(bars):
        yy = base - int(h * (0.30 + 0.26 * k))
        _line(surf, WOOD, (inner_l - 3, yy + 1), (inner_r + 3, yy), 2)
        _rim_line(surf, (inner_l - 3, yy), (inner_r + 3, yy - 1), 70)
        for px, _ in posts:
            _line(surf, ROPE, (px - 2, yy - 2), (px + 2, yy + 2), 1, 190)
            _line(surf, ROPE, (px - 2, yy + 2), (px + 2, yy - 2), 1, 190)

    # the finished palisade earns a pair of low stakes jammed outward at the
    # foot - points fanning left and right, menacing from either edge
    if full and rng.random() < 0.7:
        for sx, sdir in ((int(w * 0.32), -1), (int(w * 0.68), 1)):
            tipx = sx + sdir * int(w * 0.15)
            tipy = base - int(h * 0.30)
            _line(surf, WOOD_DARK, (sx, base), (tipx, tipy), 3)
            _poly(surf, WOOD, [(tipx - sdir * 2, tipy + 3),
                               (tipx + sdir * 3, tipy - 2),
                               (tipx - sdir, tipy + 5)])
            _rim_line(surf, (sx, base - 2), (tipx, tipy), 90)
    return surf


def _bake_hermit_fire(stage: int, rng: random.Random) -> pygame.Surface:
    """The hermit's fire: a scrape with stones round it, and NOT a firepit.

    Same argument as ``_bake_hermit_hut``, one building along. This is drawn a
    few px from his hut at the far edge of the frame, and if it reads as the
    colony's hearth then what the user sees out there is a second settlement
    rather than one man living alone - which is worse than drawing nothing,
    because it is wrong rather than missing.

    So it is the colony pit's silhouette taken apart:

      * (30, 20) against the pit's (46, 28) - two thirds the width, and lower;
      * a BROKEN ring. The pit lays a closed ring of eight stones, which is a
        built thing. This is four or five, uneven, with a gap at the front where
        he sits, which is a fire somebody scraped out with his hands;
      * no crossed woodpile. The pit stacks three logs in a lattice over itself
        even unlit - the colony has firewood banked. He has two or three sticks
        laid in, ends out, because he is burning what he cut this afternoon.

    Two stages against the pit's three, because the sim gives it two: a scrape
    and the stones set, then the sticks laid in it. There are no flames at any
    stage in either building - fire is drawn by the renderer off ``fire_active``
    and lit by ``Structure.light_source``, so a lit fire baked into the sprite
    would be a fire that is always burning even when it is out.

    ``rng`` is seeded per ``(kind, variant, stage)`` by ``Atlas._rng``, so the
    two variants keep their own stone layout across both stages.
    """
    surf = _canvas("hermit_fire")
    w, h = surf.get_size()
    cx, cy = w // 2, h - 5
    st = max(0, min(int(stage), 1))

    # The scrape: bare turned earth, wider than the stones, the mark of a fire
    # somebody dug out rather than built up.
    _ellipse(surf, EARTH, cx, cy + 2, w * 0.42, 4.0)

    # A BROKEN ring, and the gap is the character. Five stones over ~250 degrees
    # with the opening facing the viewer, each one nudged off the circle so no
    # two sit at the same height - the pit's ring is regular on purpose and this
    # one must not be.
    stones = 4 if st == 0 else 5
    span = 4.35                                  # radians, ~250 deg - not a ring
    start = 3.60 + rng.uniform(-0.25, 0.25)
    for i in range(stones):
        a = start + span * (i / max(1, stones - 1))
        jx = rng.uniform(-1.2, 1.2)
        jy = rng.uniform(-0.8, 0.8)
        sx = cx + math.cos(a) * w * 0.31 + jx
        sy = cy + math.sin(a) * 4.2 + jy
        rx = rng.uniform(2.6, 3.8)
        _ellipse(surf, STONE, sx, sy, rx, rx * 0.72)
        _rim_arc(surf, sx, sy, rx, rx * 0.72, 3.1, 6.0, 78)

    if st >= 1:
        # Sticks laid IN, ends sticking out over the stones. Never a stacked
        # lattice - that is the colony's banked firewood and it is the one shape
        # this must not borrow.
        for i in range(3):
            x0 = cx - 6 + i * 6 + rng.uniform(-1.0, 1.0)
            y0 = cy - 1
            x1 = x0 + rng.uniform(-7.0, 7.0)
            y1 = cy - rng.uniform(4.0, 7.0)
            _line(surf, WOOD_DARK if i % 2 else WOOD, (x0, y0), (x1, y1), 2)
        _rim_line(surf, (cx - 6, cy - 4), (cx + 6, cy - 5), 78)
    return surf


def _bake_hermit_hut(stage: int, rng: random.Random) -> pygame.Surface:
    """The hermit's hut: a single-pitch lean-to, and NOT a small village hut.

    THE SILHOUETTE IS THE WHOLE POINT, for the same reason his staff is. This
    building is drawn at the far edge of the frame with the settlement a long way
    off to one side, and the one thing it must never do is read as "the colony
    put up another hut over there". A village hut is symmetric: two slopes at a
    ridge over the middle of its walls. This is deliberately the opposite -
    ONE slope, running from a tall open mouth down to a back wall barely off the
    ground, with no gable and no ridge line anywhere. At thirty px across in the
    dark the asymmetry is legible when nothing else is, which is the only test
    that matters here.

    The rest of the character is scarcity. It is 46x40 against a hut's 78x64 and
    it never grows (``Structure.scale`` is 1.0 for everything but a hut, and
    ``growth_target`` returns 0.0 for this kind), so a settlement of grown huts
    at 1.85x makes it look smaller as the colony prospers, which is exactly
    right. The frame is unsawn poles rather than posts, the thatch is laid in
    uneven courses, and the whole thing leans a little.

    Stages, on the sim's three:
      0  two poles driven into a scrape of turned earth, and a spare laid beside
         them - the frame of a thing, and it is meant to look like almost
         nothing, because the interesting part is watching one man walk twelve
         loads of wood to it;
      1  the poles carry a head beam and bare rafters. A skeleton with the sky
         still coming through it;
      2  thatched, with a dark mouth under the high end and a small stack of
         firewood at the back. Finished.

    ``rng`` is seeded per ``(kind, variant, stage)`` by ``Atlas._rng``, so the
    two variants keep their own jitter and their own facing across every stage,
    and nothing here re-rolls between stages. The one draw taken up front and
    unconditionally is the facing, for the reason ``_bake_hut`` spells out: a
    draw taken inside a stage branch makes the building change character as it
    is built.
    """
    surf = _canvas("hermit_hut")
    w, h = surf.get_size()
    base = h - 2
    st = max(0, min(int(stage), 2))

    # Drawn before any branch: which way the mouth faces, how badly the poles
    # were driven, how ragged the thatch is.
    face_right = rng.random() < 0.5
    lean = rng.uniform(-1.1, 1.1)
    ragged = rng.uniform(0.0, 1.0)
    extra = rng.randint(0, 1)

    # The tall (open) end and the low (back) end, in canvas x.
    tall_x = int(w * (0.74 if face_right else 0.26))
    back_x = int(w * (0.20 if face_right else 0.80))
    tall_y = int(h * 0.20)
    back_y = int(h * 0.62)

    _ellipse(surf, EARTH, (tall_x + back_x) // 2, base, w * 0.40, 2.6)

    if st == 0:
        # Two driven poles and a spare on the ground. Nothing else exists yet.
        _stake(surf, tall_x, base, tall_y + h * 0.18, 2.0, lean=lean, point=0.0)
        _stake(surf, back_x, base, back_y + h * 0.10, 1.8, lean=-lean * 0.6, point=0.0)
        _line(surf, WOOD, (int(w * 0.30), base - 1), (int(w * 0.62), base - 2), 2, 180)
        _rim_line(surf, (tall_x, tall_y + h * 0.18), (tall_x, base), 90)
        return surf

    # Stage 1 and 2 share the frame: two uprights and the single sloping beam.
    _stake(surf, tall_x, base, tall_y, 2.2, lean=lean, point=0.0)
    _stake(surf, back_x, base, back_y, 2.0, lean=-lean * 0.5, point=0.0)
    _line(surf, WOOD_LIGHT, (tall_x, tall_y), (back_x, back_y), 2)

    # Rafters, hanging off the slope toward the ground on the closed side.
    n = 3 + extra
    eave_x = int(w * (0.06 if face_right else 0.94))
    for i in range(n):
        t = (i + 1) / (n + 1)
        rx = int(tall_x + (back_x - tall_x) * t)
        ry = int(tall_y + (back_y - tall_y) * t)
        fx = int(rx + (eave_x - rx) * 0.34)
        _line(surf, WOOD, (rx, ry), (fx, base - int(h * 0.06 * (1.0 - t))), 1, 210)

    if st == 1:
        _rim_line(surf, (tall_x, tall_y), (back_x, back_y), 120)
        return surf

    # Finished: thatch laid over the slope in uneven courses.
    roof = [
        (tall_x, tall_y - 1),
        (back_x, back_y - 1),
        (int(back_x + (eave_x - back_x) * 0.55), base - int(h * 0.04)),
        (int(tall_x + (eave_x - tall_x) * 0.30), base - int(h * 0.22)),
    ]
    _poly(surf, THATCH, roof)
    courses = 3 + int(ragged * 2.0)
    for i in range(courses):
        t = (i + 0.5) / courses
        ax = int(tall_x + (back_x - tall_x) * t)
        ay = int(tall_y + (back_y - tall_y) * t)
        bx = int(ax + (eave_x - ax) * (0.30 + 0.26 * t))
        by = base - int(h * (0.22 - 0.18 * t))
        _line(surf, THATCH_LIGHT if i % 2 else WOOD_DARK, (ax, ay), (bx, by), 1, 200)

    # The mouth: a dark opening under the high end. This is the one hole in the
    # silhouette and it is what says "somebody lives in here".
    mw = max(4, int(w * 0.17))
    mh = max(6, int(h * 0.36))
    mx = tall_x - (mw if face_right else 0)
    _poly(surf, (18, 14, 12), [
        (mx, base), (mx + mw, base),
        (mx + mw, base - mh), (mx, base - mh + int(h * 0.06)),
    ])

    # A little stacked firewood against the closed end - his whole economy.
    sx = int(back_x + (eave_x - back_x) * 0.22)
    for i in range(2 + extra):
        yy = base - 1 - i * 2
        _line(surf, WOOD_LIGHT if i % 2 else WOOD, (sx - 4, yy), (sx + 4, yy), 2, 230)

    # Rim along the one slope, which is the line that has to survive the dark.
    _rim_line(surf, (tall_x, tall_y - 1), (back_x, back_y - 1), RIM_A)
    _rim_line(surf, (tall_x, tall_y - 1),
              (int(tall_x + (eave_x - tall_x) * 0.30), base - int(h * 0.22)), 80)
    return surf


_STRUCTURE_BAKERS = {
    "firepit": _bake_firepit,
    "hut": _bake_hut,
    "wall": _bake_wall,
    "bridge": _bake_bridge,
    "watchtower": _bake_watchtower,
    "totem": _bake_totem,
    "stockpile": _bake_stockpile,
    "grave": _bake_grave_structure,
    "barricade": _bake_barricade,
    "hermit_hut": _bake_hermit_hut,
    "hermit_fire": _bake_hermit_fire,
}

_sync_with_sim()
_mirror_hut_aliases()      # after the sync, so the aliases adopt the sim's numbers


# --------------------------------------------------------------------------
# the atlas
# --------------------------------------------------------------------------


class Atlas:
    """Baked sprite cache for props and structures.

    ``Atlas()`` bakes every base sprite immediately (well under 400 ms); after
    that ``get`` is a dict lookup, with off-scale requests transformed once and
    cached under their scale bucket.
    """

    __slots__ = ("seed", "_cache", "bake_ms", "_missing")

    def __init__(self, seed: int = 20260725, bake: bool = True) -> None:
        self.seed = int(seed)
        self._cache: dict[tuple[str, int, int, int], pygame.Surface] = {}
        self.bake_ms: float = 0.0
        self._missing = pygame.Surface((2, 2), pygame.SRCALPHA)
        if bake:
            self.bake_all()

    # ------------------------------------------------------------- baking --

    def _rng(self, kind: str, variant: int, stage: int) -> random.Random:
        """The per-sprite jitter stream, seeded so it survives a restart.

        crc32 rather than ``hash()``: CPython salts str hashing per process, so
        the tuple hash this used to key on changed on every launch and *every*
        rng-driven sprite was redrawn differently each run - measured tree
        variant-0 at md5 be386bcf then 5b4b2fa0, one grave at d67d0a83 then
        0539a9bb across two fresh processes. Nothing crashed, which is why it
        survived: a forest that quietly reshuffles itself overnight just looks
        like a forest. Same trap and same fix as ``sim.props._name_variant``.
        """
        tag = f"{kind}\x1f{int(variant)}\x1f{int(stage)}".encode("utf-8", "replace")
        return random.Random((self.seed * 31 + zlib.crc32(tag)) & 0x7FFFFFFF)

    @staticmethod
    def _art_stage(kind: str, stage: int) -> int:
        """Map a simulation build stage onto this kind's drawing index.

        The sim owns how many stages a structure takes; the art owns how many
        distinct pictures exist. Mapping proportionally means neither can make
        the other's final state unreachable.
        """
        sim_stages = max(1, STRUCTURE_STAGES.get(kind, 1))
        art = max(1, ART_STAGES.get(kind, sim_stages))
        st = max(0, min(int(stage), sim_stages - 1))
        if sim_stages == 1 or art == 1:
            return art - 1 if sim_stages == 1 and art > 1 else 0
        return int(round(st / (sim_stages - 1) * (art - 1)))

    def _bake_one(self, kind: str, variant: int, stage: int,
                  scale: float = 1.0) -> pygame.Surface:
        rng = self._rng(kind, variant, stage)
        material = HUT_KINDS.get(kind)
        if material is not None:
            # the only kind whose art is a function of scale, not a resample
            return _bake_hut(self._art_stage(kind, stage), rng, scale, material)
        baker = _STRUCTURE_BAKERS.get(kind)
        if baker is not None:
            return baker(self._art_stage(kind, stage), rng)
        if kind == "water":
            return pygame.Surface(KIND_SIZE.get("water", (2, 2)), pygame.SRCALPHA)
        if kind == "tree":
            return _bake_tree(variant, rng)
        if kind == "sapling":
            return _bake_sapling(variant, rng)
        if kind == "rock":
            return _bake_rock(variant, rng)
        if kind == "boulder":
            return _bake_boulder(variant, rng)
        if kind == "bush":
            return _bake_bush(variant, rng, False)
        if kind == "bush_berry":
            return _bake_bush(variant, rng, True)
        if kind == "grave_cross":
            return _bake_grave_cross(variant, rng)
        if kind == "grave_stone":
            return _bake_grave_stone(variant, rng)
        if kind == "scorch":
            return _bake_scorch(variant, rng)
        if kind == "crop":
            return _bake_crop(variant, stage, rng)
        return self._missing

    def bake_all(self) -> float:
        """Bake every base sprite. Returns elapsed milliseconds. Fails soft."""
        t0 = time.perf_counter()
        for kind, variants in KIND_VARIANTS.items():
            stages = STRUCTURE_STAGES.get(kind, 1)
            for variant in range(variants):
                for stage in range(stages):
                    key = (kind, variant, stage, 20)
                    if key in self._cache:
                        continue
                    try:
                        self._cache[key] = self._bake_one(kind, variant, stage)
                    except Exception:
                        self._cache[key] = self._missing
        self._prebake_hut_growth()
        self.bake_ms = (time.perf_counter() - t0) * 1000.0
        return self.bake_ms

    def _prebake_hut_growth(self) -> int:
        """Bake the grown forms of a *finished* hut. Returns how many.

        Only the finished stage: growth is earned by standing, so nothing ever
        asks for a big hut still in scaffolding, and baking every stage would
        multiply startup cost for sprites no frame requests. Done at startup so
        the first hut to cross a bucket does not bake mid-frame.
        """
        n = 0
        for kind in HUT_KINDS:
            st = self.stages(kind) - 1
            for v in range(self.variants(kind)):
                for idx in range(1, HUT_GROWTH_BUCKETS):
                    try:
                        self._hut_sprite(kind, v, st, hut_bucket_scale(idx))
                        n += 1
                    except Exception:
                        continue
        return n

    # ------------------------------------------------------------ queries --

    @staticmethod
    def kinds() -> tuple[str, ...]:
        """Every sprite kind the atlas can produce."""
        return tuple(KIND_VARIANTS)

    @staticmethod
    def variants(kind: str) -> int:
        """How many visual variants *kind* has (>= 1)."""
        return max(1, KIND_VARIANTS.get(kind, 1))

    @staticmethod
    def stages(kind: str) -> int:
        """How many build stages *kind* has, as the simulation counts them
        (1 for plain props). See :data:`ART_STAGES` for how many drawings back
        those stages."""
        return max(1, STRUCTURE_STAGES.get(kind, 1))

    @staticmethod
    def is_structure(kind: str) -> bool:
        """True if *kind* is a multi-stage buildable.

        The hut aliases are in here too: they have the same stage ladder as
        ``hut``, they just are not reached through ``_STRUCTURE_BAKERS``.
        """
        return kind in _STRUCTURE_BAKERS or kind in HUT_KINDS

    def size(self, kind: str, scale: float = 1.0) -> tuple[int, int]:
        """Sprite size in pixels at *scale*.

        Reports what :meth:`get` will actually hand back, bucket quantisation and
        the hut's extra height gain included - not the naive ``base * scale``.
        """
        if kind in HUT_KINDS and float(scale) > float(HUT_SCALE_MIN) - 1e-6:
            return hut_dims(hut_bucket_scale(hut_growth_bucket(scale)))
        w, h = KIND_SIZE.get(kind, (2, 2))
        s = max(0.05, float(scale))
        return (max(1, int(round(w * s))), max(1, int(round(h * s))))

    def anchor(self, kind: str, scale: float = 1.0) -> tuple[int, int]:
        """Offset from the sprite's top-left to its ground contact point.

        Always bottom-centre; use it rather than assuming, so a change of
        padding here cannot desync the renderer.
        """
        w, h = self.size(kind, scale)
        return (w // 2, h)

    # --------------------------------------------------------------- get --

    @staticmethod
    def resolve(kind: str, state: Any = None) -> str:
        """Pick the sprite kind for a prop given its ``state`` dict.

        ``sim.props`` models a berry bush as ``kind="bush"`` with a
        ``berries_left`` counter rather than as a separate kind, so the visual
        choice has to happen here.
        """
        if isinstance(state, dict) and kind == "bush":
            try:
                if float(state.get("berries_left", 0) or 0) > 0:
                    return "bush_berry"
            except (TypeError, ValueError):
                return kind
        if isinstance(state, dict) and kind == "hut":
            # The building tech tier is ``state["material"]`` on the Structure,
            # not a second structure kind - see :data:`HUT_KINDS`. Anything
            # missing or unrecognised stays a timber hut, so every save written
            # before the tier existed renders exactly as it did before.
            try:
                mat = str(state.get("material", "") or "").strip().lower()
            except (TypeError, ValueError):
                return kind
            return _HUT_KIND_FOR_MATERIAL.get(mat, kind)
        return kind

    @staticmethod
    def crop_stage(state: Any) -> int:
        """Growth-stage index for a crop from its ``sim.props`` state dict.

        ``state['growth']`` runs 0..1 as the crop ripens and ``state['ripe']``
        latches True once it is harvest-ready; the top art stage is the golden
        ear, the ones below it are green shoots getting taller. A field that
        regrows after harvest (``ripe`` back to False, ``growth`` to 0) walks
        back down these stages on its own. Fails soft to the seedling stage.
        """
        ns = max(1, CROP_ART_STAGES)
        if not isinstance(state, dict):
            return 0
        try:
            if bool(state.get("ripe", False)):
                return ns - 1
            g = float(state.get("growth", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0
        if not math.isfinite(g):
            return 0
        g = 0.0 if g < 0.0 else (1.0 if g > 1.0 else g)
        green = max(1, ns - 1)          # stages 0 .. ns-2 are the green ones
        return max(0, min(int(g * green), green - 1))

    def get(self, kind: str, variant: int = 0, stage: int = 0,
            scale: float = 1.0, state: Any = None) -> pygame.Surface:
        """Return the baked sprite for ``(kind, variant, stage)`` at *scale*.

        Out-of-range variants wrap and out-of-range stages clamp, so callers
        never have to guard. Unknown kinds return a 2x2 transparent surface
        rather than raising - a missing prop must not kill the frame.

        *state* accepts a ``sim.props.Prop.state`` dict (used to choose between
        visual forms of one kind, e.g. a bush with or without berries) or a bare
        int, which is treated as a stage index.

        A hut at ``scale >= HUT_SCALE_MIN`` is served from the growth ladder: a
        purpose-drawn bigger building, quantised into
        :data:`HUT_GROWTH_BUCKETS` buckets. Below that it falls through to the
        generic resample, so shrunken previews still work.
        """
        try:
            if isinstance(state, (int, float)) and not isinstance(state, bool):
                stage = max(stage, int(state))
            kind = self.resolve(kind, state)
            if kind == "crop" and isinstance(state, dict):
                stage = self.crop_stage(state)
            nv = self.variants(kind)
            ns = self.stages(kind)
            v = int(variant) % nv
            st = max(0, min(int(stage), ns - 1))
            if kind in HUT_KINDS and float(scale) > float(HUT_SCALE_MIN) - 1e-6:
                return self._hut_sprite(kind, v, st, float(scale))
            bucket = int(round(float(scale) * SCALE_BUCKET))
            bucket = max(2, min(bucket, 160))
            key = (kind, v, st, bucket)
            hit = self._cache.get(key)
            if hit is not None:
                return hit

            base_key = (kind, v, st, int(SCALE_BUCKET))
            base = self._cache.get(base_key)
            if base is None:
                if kind not in KIND_VARIANTS:
                    return self._missing
                base = self._bake_one(kind, v, st)
                self._cache[base_key] = base
            if bucket == int(SCALE_BUCKET):
                return base

            s = bucket / SCALE_BUCKET
            w = max(1, int(round(base.get_width() * s)))
            h = max(1, int(round(base.get_height() * s)))
            try:
                scaled = pygame.transform.smoothscale(base, (w, h))
            except Exception:
                scaled = pygame.transform.scale(base, (w, h))
            if len(self._cache) > 1200:
                self._trim()
            self._cache[key] = scaled
            return scaled
        except Exception:
            return self._missing

    def _hut_sprite(self, kind: str, variant: int, stage: int,
                    scale: float) -> pygame.Surface:
        """Grown hut of *kind* for *scale*, baked once per growth bucket.

        A hut's scale drifts by a fraction of a percent per tick, so the whole
        point is that the bucket - not the raw scale - is the cache key. *kind*
        is one of :data:`HUT_KINDS` and goes into the key unchanged, so a stone
        hut and a timber one at the same size can never be served each other's
        sprite.
        """
        idx = hut_growth_bucket(scale)
        if idx == 0 and abs(float(HUT_SCALE_MIN) - 1.0) < 1e-6:
            # bucket 0 *is* the base bake; don't hold a second copy of it
            key = (kind, variant, stage, int(SCALE_BUCKET))
        else:
            key = (kind, variant, stage, HUT_BUCKET_KEY + idx)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        try:
            spr = self._bake_one(kind, variant, stage, hut_bucket_scale(idx))
        except Exception:
            spr = self._missing
        self._cache[key] = spr
        return spr

    def hut_bucket(self, scale: float) -> tuple[int, float, tuple[int, int]]:
        """``(bucket index, drawn scale, drawn size)`` for a hut *scale*.

        Exposed for the diagnostics overlay and for tests that need to prove the
        cache is actually bucketing rather than re-baking.
        """
        idx = hut_growth_bucket(scale)
        drawn = hut_bucket_scale(idx)
        return (idx, drawn, hut_dims(drawn))

    def blit(self, surf: pygame.Surface, kind: str, x: float, ground_y: float,
             variant: int = 0, stage: int = 0, scale: float = 1.0,
             flip: bool = False, state: Any = None) -> None:
        """Draw a sprite standing on the ground at ``(x, ground_y)``. Fails soft."""
        try:
            spr = self.get(kind, variant, stage, scale, state)
            if flip:
                spr = pygame.transform.flip(spr, True, False)
            rect = spr.get_rect()
            rect.midbottom = (int(x), int(ground_y))
            surf.blit(spr, rect)
        except Exception:
            return

    def _trim(self) -> None:
        """Drop resampled variants, keeping the base bake and the hut ladder.

        The hut ladder is bounded (8 buckets x variants x stages) and is hit
        every frame; the generic resample cache is the one that grows without
        limit. Evicting the huts first would guarantee a re-bake next frame.
        """
        base = int(SCALE_BUCKET)
        for key in [k for k in self._cache if k[3] != base and k[3] < HUT_BUCKET_KEY]:
            self._cache.pop(key, None)
        if len(self._cache) > 2000:                      # pathological; give it all back
            for key in [k for k in self._cache if k[3] >= HUT_BUCKET_KEY]:
                self._cache.pop(key, None)

    def cache_stats(self) -> dict[str, Any]:
        """Sprite count and bake time, for the diagnostics overlay."""
        grown = sum(1 for k in self._cache if k[3] >= HUT_BUCKET_KEY)
        return {"sprites": len(self._cache), "bake_ms": round(self.bake_ms, 2),
                "hut_grown": grown}


# --------------------------------------------------------------------------
# shared instance
# --------------------------------------------------------------------------

_ATLAS: Atlas | None = None


def get_atlas(seed: int = 20260725) -> Atlas:
    """Process-wide shared atlas, baked on first use."""
    global _ATLAS
    if _ATLAS is None:
        _ATLAS = Atlas(seed)
    return _ATLAS


def clear_caches() -> None:
    """Discard the shared atlas so the next call re-bakes it."""
    global _ATLAS
    _ATLAS = None


if __name__ == "__main__":  # pragma: no cover - smoke test
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    atlas = Atlas()
    print(f"baked {len(atlas._cache)} sprites in {atlas.bake_ms:.1f} ms")
    assert atlas.bake_ms < 400.0, "atlas bake budget blown"

    # A golden number, not a re-run: two Atlases inside one process agree even
    # when the seeding is salted, so the only way to catch a slide back to
    # hash() of a str from a single process is to pin the value it must produce.
    assert abs(atlas._rng("tree", 0, 0).random() - 0.32432691957352955) < 1e-12, \
        "sprite rng is no longer reproducible across processes"

    # contact sheet of everything, on a dark background
    cell = 128
    cols = 8
    items: list[tuple[str, int, int]] = []
    for kind, nv in KIND_VARIANTS.items():
        for v in range(nv):
            for st in range(STRUCTURE_STAGES.get(kind, 1)):
                items.append((kind, v, st))
    rows = (len(items) + cols - 1) // cols
    sheet = pygame.Surface((cols * cell, rows * cell))
    sheet.fill((18, 20, 26))
    for i, (kind, v, st) in enumerate(items):
        cx = (i % cols) * cell + cell // 2
        cy = (i // cols) * cell + cell - 12
        pygame.draw.line(sheet, (40, 44, 50), (cx - 50, cy), (cx + 50, cy), 1)
        atlas.blit(sheet, kind, cx, cy, v, st)
    try:
        from ..paths import CAPTURE_DIR, ensure_dirs
        ensure_dirs()
        pygame.image.save(sheet, str(CAPTURE_DIR / "atlas.png"))
        print("wrote atlas.png to", CAPTURE_DIR)
    except Exception as exc:
        print("capture skipped:", exc)

    # hut growth ladder: every bucket, on one strip, with the ground line drawn
    # so it is obvious the sprite grows upward from it
    st_final = STRUCTURE_STAGES.get("hut", 5) - 1
    dims = [atlas.hut_bucket(hut_bucket_scale(i)) for i in range(HUT_GROWTH_BUCKETS)]
    strip_w = sum(d[2][0] + 16 for d in dims) + 16
    row_h = max(d[2][1] for d in dims) + 40
    strip = pygame.Surface((strip_w, row_h * len(HUT_KINDS)))
    strip.fill((18, 20, 26))
    # One row per material: the point of the strip is that both walls climb the
    # same ladder and pick up the same growth details at the same buckets.
    for row, hkind in enumerate(HUT_KINDS):
        gy = row * row_h + row_h - 16
        pygame.draw.line(strip, (44, 48, 54), (0, gy), (strip_w, gy), 1)
        px = 16
        for idx, drawn, (sw, sh) in dims:
            atlas.blit(strip, hkind, px + sw // 2, gy, 0, st_final, drawn)
            px += sw + 16
    try:
        pygame.image.save(strip, str(CAPTURE_DIR / "hut_growth.png"))
        print("wrote hut_growth.png to", CAPTURE_DIR)
    except Exception as exc:
        print("hut strip capture skipped:", exc)

    print(f"hut ladder ({HUT_GROWTH_BUCKETS} buckets, step {hut_bucket_step():.4f}):")
    for idx, drawn, (sw, sh) in dims:
        spr = atlas.get("hut", 0, st_final, drawn)
        print(f"  bucket {idx}  scale {drawn:.3f}  growth {hut_growth(drawn):.2f}  "
              f"dims {sw}x{sh}  actual {spr.get_size()}")

    # the whole point of bucketing: a drifting scale must not add sprites
    before = len(atlas._cache)
    t0 = time.perf_counter()
    n = 0
    for i in range(4000):
        sc = HUT_SCALE_MIN + (HUT_SCALE_MAX - HUT_SCALE_MIN) * (i / 3999.0)
        for hkind in HUT_KINDS:
            atlas.get(hkind, i % 4, st_final, sc)
            n += 1
    dt = (time.perf_counter() - t0) / n * 1e6
    print(f"drifting-scale get(): {dt:.2f} us/call, cache {before} -> {len(atlas._cache)}")
    assert len(atlas._cache) == before, "drifting hut scale is not hitting the bucket cache"

    # The variant axis is wired through the sim, the saves and the cache key, so
    # it has to actually change the picture; and a stone hut must never be
    # served a timber one's sprite at the same size.
    for sc in (1.0, 1.4, HUT_SCALE_MAX):
        for hkind in HUT_KINDS:
            seen = {pygame.image.tobytes(atlas.get(hkind, v, st_final, sc), "RGBA")
                    for v in range(atlas.variants(hkind))}
            assert len(seen) == atlas.variants(hkind), f"{hkind} variants collapsed at {sc}"
        assert (pygame.image.tobytes(atlas.get("hut", 0, st_final, sc), "RGBA")
                != pygame.image.tobytes(atlas.get("hut_stone", 0, st_final, sc), "RGBA")), \
            "stone hut is baking the timber drawing"
    assert Atlas.resolve("hut", {"material": "stone"}) == "hut_stone"
    assert Atlas.resolve("hut", {"material": "sponge"}) == "hut"   # unknown -> timber
    assert Atlas.resolve("hut", None) == "hut"                     # pre-tier save

    t0 = time.perf_counter()
    for _ in range(2000):
        atlas.get("tree", 1, 0)
        atlas.get("hut", 0, 3)
    print(f"warm get(): {(time.perf_counter() - t0) / 4000 * 1e6:.2f} us/call")
    print("scaled:", atlas.get("tree", 0, 0, 0.5).get_size(),
          "| unknown kind:", atlas.get("nope").get_size(),
          "| stage clamp:", atlas.get("hut", 0, 99).get_size(),
          "| shrunk hut:", atlas.get("hut", 0, st_final, 0.5).get_size(),
          "| over-max hut:", atlas.get("hut", 0, st_final, 2.6).get_size())
    print("size() agrees with get():",
          all(atlas.size("hut", d[1]) == atlas.get("hut", 0, st_final, d[1]).get_size()
              for d in dims))
    print(atlas.cache_stats())
