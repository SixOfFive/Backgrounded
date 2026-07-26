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
sprite padding cannot desync the renderer.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Sequence

import pygame

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
}

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


# kind -> base sprite size in pixels
KIND_SIZE: dict[str, tuple[int, int]] = {
    "tree": (58, 98),
    "sapling": (20, 32),
    "rock": (28, 18),
    "boulder": (50, 34),
    "bush": (36, 26),
    "bush_berry": (36, 26),
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
}

SCALE_BUCKET = 20.0              # scales quantise to 1/20 = 0.05 steps


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


def _bake_hut(stage: int, rng: random.Random) -> pygame.Surface:
    surf = _canvas("hut")
    w, h = surf.get_size()
    left, right = int(w * 0.12), int(w * 0.88)
    base = h - 2
    wall_top = int(h * 0.46)
    ridge = int(h * 0.12)
    cx = w // 2

    if stage == 0:
        for x in (left, int(w * 0.36), int(w * 0.64), right):
            _line(surf, WOOD_DARK, (x, base), (x + rng.uniform(-1, 1), base - 14), 3)
            _rim_line(surf, (x - 1, base - 2), (x - 1, base - 13), 80)
        _ellipse(surf, EARTH, cx, base, w * 0.42, 3.0)
        return surf

    # posts
    for x in (left, right):
        _line(surf, WOOD_DARK, (x, base), (x, wall_top), 4)
    _line(surf, WOOD_DARK, (left, wall_top), (right, wall_top), 3)

    if stage == 1:
        # frame only: rafters and a ridge beam
        _line(surf, WOOD, (left, wall_top), (cx, ridge), 3)
        _line(surf, WOOD, (right, wall_top), (cx, ridge), 3)
        _line(surf, WOOD, (int(w * 0.3), wall_top), (cx, ridge + 6), 2)
        _line(surf, WOOD, (int(w * 0.7), wall_top), (cx, ridge + 6), 2)
        _rim_line(surf, (left, wall_top - 1), (cx, ridge - 1), 100)
        _ellipse(surf, EARTH, cx, base, w * 0.42, 3.0)
        return surf

    # walls
    _poly(surf, WOOD_DARK, [(left, base), (left, wall_top), (right, wall_top), (right, base)])
    for i in range(4):
        y = wall_top + 5 + i * ((base - wall_top) // 5)
        _line(surf, WOOD, (left + 2, y), (right - 2, y), 1, 150)

    if stage == 2:
        # partially thatched roof: rafters showing through
        _poly(surf, THATCH, [(left - 5, wall_top + 2), (cx, ridge), (cx + 2, ridge + 3),
                             (int(w * 0.2), wall_top + 2)])
        _line(surf, WOOD, (right + 4, wall_top + 2), (cx, ridge), 3)
        _line(surf, WOOD, (int(w * 0.72), wall_top + 2), (cx, ridge + 5), 2)
        _rim_line(surf, (left - 4, wall_top + 1), (cx, ridge - 1), 110)
    else:
        # finished: full thatch roof, ridge cap, doorway
        _poly(surf, THATCH, [(left - 6, wall_top + 3), (cx, ridge),
                             (right + 6, wall_top + 3)])
        for i in range(5):
            t = 0.18 + 0.16 * i
            _line(surf, THATCH_LIGHT,
                  (left - 6 + (cx - left + 6) * t, wall_top + 3 - (wall_top + 3 - ridge) * t),
                  (right + 6 - (right + 6 - cx) * t, wall_top + 3 - (wall_top + 3 - ridge) * t),
                  1, 90)
        _line(surf, WOOD_LIGHT, (cx - 4, ridge + 1), (cx + 4, ridge + 1), 2)
        _rim_line(surf, (left - 6, wall_top + 2), (cx, ridge - 1), 130)
        _rim_line(surf, (cx, ridge - 1), (right + 6, wall_top + 2), 60)
        door = pygame.Rect(cx - 7, base - 20, 14, 19)
        pygame.draw.rect(surf, _rgba((16, 12, 10), 235), door)
        _rim_line(surf, (cx - 8, base - 21), (cx - 8, base - 2), 90)
        _rim_line(surf, (cx - 8, base - 21), (cx + 7, base - 21), 70)

    _ellipse(surf, EARTH, cx, base, w * 0.44, 3.0)
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


_STRUCTURE_BAKERS = {
    "firepit": _bake_firepit,
    "hut": _bake_hut,
    "wall": _bake_wall,
    "bridge": _bake_bridge,
    "watchtower": _bake_watchtower,
    "totem": _bake_totem,
    "stockpile": _bake_stockpile,
    "grave": _bake_grave_structure,
}

_sync_with_sim()


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
        return random.Random((self.seed * 31 + hash((kind, variant, stage))) & 0x7FFFFFFF)

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

    def _bake_one(self, kind: str, variant: int, stage: int) -> pygame.Surface:
        rng = self._rng(kind, variant, stage)
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
        self.bake_ms = (time.perf_counter() - t0) * 1000.0
        return self.bake_ms

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
        """True if *kind* is a multi-stage buildable."""
        return kind in _STRUCTURE_BAKERS

    def size(self, kind: str, scale: float = 1.0) -> tuple[int, int]:
        """Sprite size in pixels at *scale*."""
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
        return kind

    def get(self, kind: str, variant: int = 0, stage: int = 0,
            scale: float = 1.0, state: Any = None) -> pygame.Surface:
        """Return the baked sprite for ``(kind, variant, stage)`` at *scale*.

        Out-of-range variants wrap and out-of-range stages clamp, so callers
        never have to guard. Unknown kinds return a 2x2 transparent surface
        rather than raising - a missing prop must not kill the frame.

        *state* accepts a ``sim.props.Prop.state`` dict (used to choose between
        visual forms of one kind, e.g. a bush with or without berries) or a bare
        int, which is treated as a stage index.
        """
        try:
            if isinstance(state, (int, float)) and not isinstance(state, bool):
                stage = max(stage, int(state))
            kind = self.resolve(kind, state)
            nv = self.variants(kind)
            ns = self.stages(kind)
            v = int(variant) % nv
            st = max(0, min(int(stage), ns - 1))
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
        """Drop scaled variants, keeping the base bake."""
        base = int(SCALE_BUCKET)
        for key in [k for k in self._cache if k[3] != base]:
            self._cache.pop(key, None)

    def cache_stats(self) -> dict[str, Any]:
        """Sprite count and bake time, for the diagnostics overlay."""
        return {"sprites": len(self._cache), "bake_ms": round(self.bake_ms, 2)}


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

    t0 = time.perf_counter()
    for _ in range(2000):
        atlas.get("tree", 1, 0)
        atlas.get("hut", 0, 3)
    print(f"warm get(): {(time.perf_counter() - t0) / 4000 * 1e6:.2f} us/call")
    print("scaled:", atlas.get("tree", 0, 0, 0.5).get_size(),
          "| unknown kind:", atlas.get("nope").get_size(),
          "| stage clamp:", atlas.get("hut", 0, 99).get_size())
    print(atlas.cache_stats())
