"""Procedural skeletal stick figures, drawn with antialiased lines.

No sprite sheets and no baked frames: every pose is a set of joint angles
evaluated per frame, so a stickman can walk, carry, chop, climb and flail
mid-fall without a single asset on disk.

**This module never mutates sim state.** It reads a :class:`Stickman` and
draws it. Every value it needs that is not on the agent (breathing phase,
candle flicker, panic jitter) is derived from the render clock ``t`` and the
agent's id.

Layout
------
Local space has its origin at the agent's *feet*, with -y up. Each limb is a
two-bone chain of angles measured from straight-down; the torso is measured
from straight-up. The whole skeleton is mirrored by ``facing``, optionally
rotated about the hip (falling, lying down), then translated to world space.
Limbs on the far side are drawn first and dimmed, which is what gives a
two-line figure any sense of depth at 21 pixels tall.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import pygame

from ..constants import (
    RENDER_H,
    RENDER_W,
    RES_COOKED,
    RES_FIBRE,
    RES_FOOD,
    RES_STONE,
    RES_WOOD,
)
from ..sim.entities import AGENT_HEIGHT, GRAVE_DELAY, ROLE_CHILD, ROLE_ELDER, Stickman

log = logging.getLogger(__name__)

__all__ = [
    "POSES", "Skeleton", "build_skeleton", "pose_of",
    "draw_stickman", "draw_grave", "SILHOUETTE",
]

TAU = math.tau

# ------------------------------------------------------------------- poses --
#: Every pose the poser draws. ``'dead'`` is derived, not requested.
POSES: tuple[str, ...] = (
    "idle", "walk", "carry", "chop", "build", "climb", "fall", "sleep",
    "dance", "mourn", "panic",
)
_KNOWN: frozenset[str] = frozenset(POSES) | {"dead"}

#: sim/actions.py has a wider pose vocabulary than the poser draws (it names 21
#: activities, we draw 11 bodies). Rather than dropping the ones we lack and
#: falling back to "standing still", each maps onto its nearest body.
POSE_ALIASES: dict[str, str] = {
    "run": "walk",
    "haul": "carry",
    "eat": "carry",            # food held up to the face
    "mine": "chop",            # same overhead swing, different tool
    "forage": "build",         # crouched, hands working
    "cook": "build",
    "plant": "build",
    "warm": "idle",            # huddled at the fire
    "talk": "idle",
    "lookout": "idle",
    "flee": "panic",
}


def _resolve_pose(name: Any) -> str | None:
    """Canonical pose for a name from anywhere, or None if unrecognised."""
    if not isinstance(name, str) or not name:
        return None
    if name in _KNOWN:
        return name
    return POSE_ALIASES.get(name)

# ------------------------------------------------------------ proportions --
# All fractions of the agent's drawn height.
_HIP = 0.47          # hip height above the feet
_TORSO = 0.30        # hip -> neck
_SHOULDER = 0.86     # arms hang from this fraction along the torso
_THIGH = 0.245
_SHIN = 0.245
_UARM = 0.155
_FARM = 0.145
_HEADR = 0.115

#: Default silhouette colour for agents outside every light source.
SILHOUETTE: tuple[int, int, int] = (9, 10, 14)

_FAR_DIM = 0.62      # far-side limbs are drawn this much darker
_WALK_HZ = 1.55      # stride cycles per unit of anim_t

_FLAME_CORE = (255, 240, 186)
_FLAME_OUTER = (255, 162, 58)
_CANDLE_WAX = (232, 224, 200)
_TORCH_SHAFT = (120, 82, 44)      # a longer wooden haft than the candle's stub
_TORCH_HEAD = (60, 44, 30)        # the pitch-soaked wrap at the top
_TORCH_FLAME = (255, 138, 40)

_BUBBLE_BG = (233, 236, 244)
_BUBBLE_EDGE = (44, 48, 58)
_BUBBLE_INK = (32, 34, 42)

_GRAVE_STONE = (126, 128, 136)
_GRAVE_SHADE = (74, 76, 84)
_GRAVE_MOUND = (58, 46, 36)

RESOURCE_COLORS: dict[str, tuple[int, int, int]] = {
    RES_WOOD: (142, 98, 58),
    RES_STONE: (132, 136, 144),
    RES_FOOD: (206, 74, 74),
    RES_COOKED: (214, 150, 70),
    RES_FIBRE: (178, 166, 112),
}

# pygame-ce takes a width on aaline; older pygame does not. Probe once at
# import so the per-frame path has no try/except and no version check.
try:
    _probe = pygame.Surface((4, 4))
    pygame.draw.aaline(_probe, (1, 2, 3), (0, 0), (3, 3), 2)
    _AA_WIDTH = True
except Exception:                                     # pragma: no cover
    _AA_WIDTH = False
finally:
    _probe = None

try:
    _AACIRCLE: Callable[..., Any] | None = pygame.draw.aacircle
except AttributeError:                                # pragma: no cover
    _AACIRCLE = None


# ------------------------------------------------------------- primitives --
def _dim(col: Sequence[int], f: float) -> tuple[int, int, int]:
    return (int(col[0] * f), int(col[1] * f), int(col[2] * f))


def _line(surf: pygame.Surface, col: Sequence[int],
          a: tuple[float, float], b: tuple[float, float], w: int = 1) -> None:
    if _AA_WIDTH:
        pygame.draw.aaline(surf, col, a, b, w)
    else:                                             # pragma: no cover
        pygame.draw.aaline(surf, col, a, b)
        if w > 1:
            pygame.draw.line(surf, col, (int(a[0]), int(a[1])),
                             (int(b[0]), int(b[1])), w)


def _disc(surf: pygame.Surface, col: Sequence[int],
          c: tuple[float, float], r: float, width: int = 0) -> None:
    # aacircle draws *nothing* below r=1.0, which silently swallowed every
    # sub-pixel dot (speech glyphs, grave chips, the candle core). A dot that
    # small should still be one pixel, so clamp rather than skip.
    if r <= 0.0:
        return
    if _AACIRCLE is not None:
        _AACIRCLE(surf, col, c, max(1.0, r), width)
    else:                                             # pragma: no cover
        pygame.draw.circle(surf, col, (int(c[0]), int(c[1])), max(1, int(r)), width)


# ------------------------------------------------------------------- pose --
@dataclass(slots=True)
class _Pose:
    """Joint angles for one frame. Angles in radians, offsets in pixels."""

    lean: float = 0.0                       # torso from vertical, + = forward
    head: float = 0.0                       # extra head tilt on top of lean
    bob: float = 0.0                        # vertical offset, - = up
    dx: float = 0.0                         # horizontal offset (jitter)
    rot: float = 0.0                        # whole-body rotation about the hip
    crouch: float = 0.0                     # 0..1, lowers the hip
    lie: bool = False                       # lay the body along the ground
    # (front, back) pairs - index 0 is the limb on the side we face
    thigh: tuple[float, float] = (0.0, 0.0)
    knee: tuple[float, float] = (0.0, 0.0)
    shoulder: tuple[float, float] = (0.0, 0.0)
    elbow: tuple[float, float] = (0.0, 0.0)


@dataclass(slots=True)
class Skeleton:
    """Solved joint positions in world space."""

    hip: tuple[float, float] = (0.0, 0.0)
    neck: tuple[float, float] = (0.0, 0.0)
    shoulder: tuple[float, float] = (0.0, 0.0)
    head: tuple[float, float] = (0.0, 0.0)
    head_r: float = 2.0
    knees: list[tuple[float, float]] = field(default_factory=list)
    feet: list[tuple[float, float]] = field(default_factory=list)
    elbows: list[tuple[float, float]] = field(default_factory=list)
    hands: list[tuple[float, float]] = field(default_factory=list)
    height: float = AGENT_HEIGHT
    facing: int = 1
    pose: str = "idle"

    @property
    def front_hand(self) -> tuple[float, float]:
        """Where a carried item, tool or candle sits."""
        return self.hands[0] if self.hands else self.shoulder


def pose_of(s: Stickman) -> str:
    """Pick the pose for an agent: action first, then override, then motion."""
    act = getattr(s, "action", None)
    if act is not None:
        p = _resolve_pose(getattr(act, "pose", None))
        if p is not None:
            return p
    p = _resolve_pose(getattr(s, "pose_override", None))
    if p is not None:
        return p
    if not getattr(s, "alive", True):
        return "dead"
    if getattr(s, "climbing", False):
        return "climb"
    if not getattr(s, "on_ground", True) or getattr(s, "fall_t", 0.0) > 0.05:
        return "fall"
    if abs(getattr(s, "vx", 0.0)) > 1.5:
        return "carry" if getattr(s, "carrying", None) else "walk"
    return "idle"


def _height_of(s: Stickman) -> float:
    fn = getattr(s, "height", None)
    if callable(fn):
        try:
            h = float(fn())
            if 4.0 <= h <= 200.0:
                return h
        except Exception:
            pass
    role = getattr(s, "role", "")
    if role == ROLE_CHILD:
        return AGENT_HEIGHT * 0.68
    if role == ROLE_ELDER:
        return AGENT_HEIGHT * 0.94
    return AGENT_HEIGHT


# --------------------------------------------------------- pose functions --
def _p_idle(s: Stickman, t: float, ph: float, idp: float) -> _Pose:
    breath = math.sin(t * 1.55 + idp)
    return _Pose(
        lean=0.03,
        head=-0.02 + 0.02 * breath,
        bob=-0.35 - 0.30 * breath,
        # A closed stance fuses into one thick line; keep the limbs standing off
        # the body even at rest.
        thigh=(0.20, -0.17),
        knee=(0.06, 0.05),
        shoulder=(0.30 + 0.04 * breath, -0.27 - 0.04 * breath),
        elbow=(0.16, 0.14),
    )


def _p_walk(s: Stickman, t: float, ph: float, idp: float) -> _Pose:
    sw = math.sin(ph)
    sw2 = math.sin(ph + math.pi)
    return _Pose(
        lean=0.10,
        head=-0.05,
        bob=-abs(math.sin(ph)) * 0.95,
        thigh=(sw * 0.62, sw2 * 0.62),
        # The knee only bends backwards, and only on the recovery half-stride.
        knee=(max(0.0, -math.sin(ph + 0.85)) * 0.95,
              max(0.0, -math.sin(ph + 0.85 + math.pi)) * 0.95),
        shoulder=(sw2 * 0.46, sw * 0.46),
        elbow=(0.26 + 0.16 * sw2, 0.26 + 0.16 * sw),
    )


def _p_carry(s: Stickman, t: float, ph: float, idp: float) -> _Pose:
    moving = abs(getattr(s, "vx", 0.0)) > 1.5
    amp = 0.44 if moving else 0.06
    sw = math.sin(ph)
    sw2 = math.sin(ph + math.pi)
    return _Pose(
        lean=-0.09,                          # lean back against the load
        head=0.06,
        bob=(-abs(math.sin(ph)) * 0.75) if moving else (-0.3 - 0.2 * math.sin(t * 1.4)),
        thigh=(sw * amp, sw2 * amp),
        knee=(max(0.0, -math.sin(ph + 0.85)) * 0.7,
              max(0.0, -math.sin(ph + 0.85 + math.pi)) * 0.7),
        shoulder=(1.30, 1.16),               # both arms out in front
        elbow=(0.16, 0.24),
    )


def _p_chop(s: Stickman, t: float, ph: float, idp: float) -> _Pose:
    # Slow overhead raise, fast strike down. k: 0 at the strike, 1 overhead.
    u = ((ph / TAU) * 0.62) % 1.0
    if u < 0.68:
        e = u / 0.68
        k = e * e * (3.0 - 2.0 * e)          # smoothstep up
    else:
        k = 1.0 - ((u - 0.68) / 0.32) ** 0.65
    arm = 1.28 + (2.92 - 1.28) * k
    impact = (1.0 - k) ** 2
    return _Pose(
        lean=0.14 + 0.26 * (1.0 - k),
        head=-0.10 * k,
        bob=1.1 * impact,
        crouch=0.22 * impact,
        thigh=(0.34, -0.30),
        knee=(0.24 + 0.2 * impact, 0.22 + 0.2 * impact),
        shoulder=(arm + 0.10, arm - 0.10),   # both hands on the haft
        elbow=(-0.30 * k, -0.26 * k),
    )


def _p_build(s: Stickman, t: float, ph: float, idp: float) -> _Pose:
    w = math.sin(ph * 2.1 + idp)
    w2 = math.sin(ph * 2.1 + idp + 0.9)
    return _Pose(
        lean=0.44,
        head=0.10,
        bob=0.55 * abs(w),
        crouch=0.42,
        thigh=(0.62, -0.34),
        knee=(0.95, 0.88),
        shoulder=(1.42 + 0.30 * w, 1.30 + 0.30 * w2),
        elbow=(0.55 + 0.32 * w2, 0.48 + 0.32 * w),
    )


def _p_climb(s: Stickman, t: float, ph: float, idp: float) -> _Pose:
    # Near-upright and pressed to the rock, reaching hand over hand. Leaning the
    # torso far forward here reads as "doubled over", not "climbing".
    r = math.sin(ph * 0.85 + idp)
    return _Pose(
        lean=0.15,
        head=-0.26,                          # looking up the face
        bob=-0.45 + 0.40 * r,
        # one leg stepped high with a folded knee, the other taking the weight
        thigh=(0.30 + 0.45 * r, 0.30 - 0.45 * r),
        knee=(0.45 + 0.35 * r, 0.45 - 0.35 * r),
        shoulder=(2.80 + 0.30 * r, 2.60 - 0.30 * r),
        elbow=(-0.20 - 0.30 * r, -0.20 + 0.30 * r),
    )


def _p_fall(s: Stickman, t: float, ph: float, idp: float) -> _Pose:
    f = float(getattr(s, "fall_t", 0.0)) + idp
    spin = float(getattr(s, "fall_t", 0.0)) * 4.6
    return _Pose(
        lean=0.12 * math.sin(f * 7.0),
        head=-0.22,
        rot=spin,
        thigh=(0.85 + 0.55 * math.sin(f * 10.0), -0.65 + 0.55 * math.sin(f * 9.0 + 1.0)),
        knee=(0.55 + 0.45 * math.sin(f * 15.0), 0.45 + 0.45 * math.sin(f * 13.0)),
        shoulder=(2.25 + 0.60 * math.sin(f * 12.0),
                  -2.25 + 0.60 * math.sin(f * 14.0 + 2.0)),
        elbow=(0.50 * math.sin(f * 17.0), 0.50 * math.sin(f * 19.0)),
    )


def _p_sleep(s: Stickman, t: float, ph: float, idp: float) -> _Pose:
    breath = math.sin(t * 0.85 + idp)
    return _Pose(
        lean=-0.10,
        head=0.30,
        rot=math.pi * 0.5,
        lie=True,
        bob=0.20 * breath,
        thigh=(0.40, 0.55),
        knee=(0.72 + 0.04 * breath, 0.62),
        shoulder=(0.52, 0.36),
        elbow=(0.62, 0.50 + 0.05 * breath),
    )


def _p_dead(s: Stickman, t: float, ph: float, idp: float) -> _Pose:
    return _Pose(
        lean=-0.22,
        head=0.55,
        rot=math.pi * 0.5,
        lie=True,
        thigh=(0.52, -0.38),
        knee=(0.30, 0.22),
        shoulder=(0.85, -0.95),
        elbow=(0.18, -0.22),
    )


def _p_dance(s: Stickman, t: float, ph: float, idp: float) -> _Pose:
    d = t * TAU * 2.1 + idp
    hop = abs(math.sin(d))
    return _Pose(
        lean=0.14 * math.sin(d),
        head=-0.12 * math.cos(d),
        bob=-hop * 2.7,
        thigh=(0.40 * math.sin(d), -0.40 * math.sin(d)),
        knee=(0.55 * hop, 0.50),
        shoulder=(2.72 + 0.32 * math.sin(d * 2.0), -2.72 - 0.32 * math.sin(d * 2.0)),
        elbow=(0.38 * math.sin(d * 2.0), -0.38 * math.sin(d * 2.0)),
    )


def _p_mourn(s: Stickman, t: float, ph: float, idp: float) -> _Pose:
    sway = math.sin(t * 0.7 + idp)
    return _Pose(
        lean=0.30,
        head=0.46,                           # head bowed
        bob=0.22 * sway,
        thigh=(0.10, -0.10),
        knee=(0.16, 0.16),
        shoulder=(0.16, -0.12),
        elbow=(0.30, 0.28),
    )


def _p_panic(s: Stickman, t: float, ph: float, idp: float) -> _Pose:
    j = math.sin(t * 23.0 + idp * 1.7)
    j2 = math.sin(t * 19.0 + idp * 2.3)
    return _Pose(
        lean=0.12 * j2,
        head=-0.20 + 0.12 * j,
        bob=-abs(math.sin(t * 13.0 + idp)) * 1.3,
        dx=0.55 * j,
        thigh=(0.55 * j, -0.55 * j2),
        knee=(0.45 + 0.30 * j, 0.42 + 0.30 * j2),
        shoulder=(2.88 + 0.34 * j, -2.88 - 0.34 * j2),
        elbow=(0.42 * j2, -0.42 * j),
    )


_POSE_FN: dict[str, Callable[[Stickman, float, float, float], _Pose]] = {
    "idle": _p_idle, "walk": _p_walk, "carry": _p_carry, "chop": _p_chop,
    "build": _p_build, "climb": _p_climb, "fall": _p_fall, "sleep": _p_sleep,
    "dance": _p_dance, "mourn": _p_mourn, "panic": _p_panic, "dead": _p_dead,
}


# ------------------------------------------------------------- skeleton --
def build_skeleton(s: Stickman, t: float, pose: str | None = None) -> Skeleton:
    """Solve the joint chain for *s* at render time *t*.

    Public because other render modules need real joint positions - the hand
    for a carried prop, the head for a speech bubble. Read-only w.r.t. *s*.
    """
    name = _resolve_pose(pose) or pose_of(s)
    height = _height_of(s)
    idp = (float(getattr(s, "id", 0)) * 0.7137) % TAU
    ph = float(getattr(s, "anim_t", 0.0)) * TAU * _WALK_HZ + idp
    fn = _POSE_FN.get(name, _p_idle)
    try:
        p = fn(s, t, ph, idp)
    except Exception:
        log.debug("pose %s failed", name, exc_info=True)
        p = _Pose()
    return _solve(s, p, height, name)


def _solve(s: Stickman, p: _Pose, h: float, name: str) -> Skeleton:
    hip_h = h * _HIP * (1.0 - 0.30 * p.crouch)
    l_thigh, l_shin = h * _THIGH, h * _SHIN
    l_torso = h * _TORSO
    l_ua, l_fa = h * _UARM, h * _FARM
    head_r = h * _HEADR

    sin, cos = math.sin, math.cos
    hip = (0.0, -hip_h)
    sl, cl = sin(p.lean), cos(p.lean)
    neck = (hip[0] + sl * l_torso, hip[1] - cl * l_torso)
    shoulder = (hip[0] + sl * l_torso * _SHOULDER, hip[1] - cl * l_torso * _SHOULDER)
    ha = p.lean + p.head
    # Sit the head clear of the shoulders. Anatomically the skull rests on the
    # neck, but at ~20px a touching circle fuses with the torso into one blob,
    # so the head gets a pixel of daylight underneath it.
    neck_gap = head_r * 1.15 + 1.1
    head = (neck[0] + sin(ha) * neck_gap, neck[1] - cos(ha) * neck_gap)

    knees: list[tuple[float, float]] = []
    feet: list[tuple[float, float]] = []
    for i in (0, 1):
        a1 = p.thigh[i]
        knee = (hip[0] + sin(a1) * l_thigh, hip[1] + cos(a1) * l_thigh)
        a2 = a1 + p.knee[i]
        knees.append(knee)
        feet.append((knee[0] + sin(a2) * l_shin, knee[1] + cos(a2) * l_shin))

    elbows: list[tuple[float, float]] = []
    hands: list[tuple[float, float]] = []
    for i in (0, 1):
        b1 = p.shoulder[i]
        elbow = (shoulder[0] + sin(b1) * l_ua, shoulder[1] + cos(b1) * l_ua)
        b2 = b1 + p.elbow[i]
        elbows.append(elbow)
        hands.append((elbow[0] + sin(b2) * l_fa, elbow[1] + cos(b2) * l_fa))

    # -- transform: mirror by facing, rotate about the hip, translate --------
    fx = -1.0 if int(getattr(s, "facing", 1)) < 0 else 1.0
    rot = p.rot * fx
    cr, sr = (cos(rot), sin(rot)) if rot else (1.0, 0.0)
    piv_y = hip[1]
    lie_dy = (hip_h - head_r - 0.4) if p.lie else 0.0
    ox = float(getattr(s, "x", 0.0)) + p.dx * fx
    oy = float(getattr(s, "y", 0.0)) + p.bob + lie_dy

    def xf(pt: tuple[float, float]) -> tuple[float, float]:
        px, py = pt[0] * fx, pt[1]
        if rot:
            dy0 = py - piv_y
            px, py = px * cr - dy0 * sr, piv_y + px * sr + dy0 * cr
        return (px + ox, py + oy)

    return Skeleton(
        hip=xf(hip), neck=xf(neck), shoulder=xf(shoulder), head=xf(head),
        head_r=head_r,
        knees=[xf(k) for k in knees], feet=[xf(f) for f in feet],
        elbows=[xf(e) for e in elbows], hands=[xf(hh) for hh in hands],
        height=h, facing=int(fx), pose=name,
    )


# ------------------------------------------------------------- main draw --
def draw_stickman(
    surf: pygame.Surface,
    s: Stickman,
    t: float,
    alpha_color: tuple[int, int, int] | None = None,
) -> None:
    """Draw one stickman, its held items and its speech bubble.

    *t* is the render clock in seconds (breathing, flicker, jitter). Pass
    *alpha_color* to force silhouette mode - a flat near-black outline for an
    agent standing outside every light source.
    """
    try:
        _draw(surf, s, float(t), alpha_color)
    except Exception:
        log.exception("draw_stickman failed for %s", getattr(s, "name", "?"))


def _draw(surf: pygame.Surface, s: Stickman, t: float,
          alpha_color: tuple[int, int, int] | None) -> None:
    x = float(getattr(s, "x", 0.0))
    y = float(getattr(s, "y", 0.0))
    if not (math.isfinite(x) and math.isfinite(y)):
        return
    h = _height_of(s)
    # Cheap reject: bubbles and flung limbs stay inside this margin.
    if x < -h * 2.0 or x > RENDER_W + h * 2.0 or y < -h * 3.0 or y > RENDER_H + h * 2.0:
        return

    sk = build_skeleton(s, t)
    silhouette = alpha_color is not None
    if silhouette:
        base = tuple(int(c) for c in alpha_color[:3])   # type: ignore[index]
    else:
        col = getattr(s, "color", (206, 214, 226))
        try:
            base = (int(col[0]), int(col[1]), int(col[2]))
        except Exception:
            base = (206, 214, 226)

    if not getattr(s, "alive", True):
        # Corpses sink toward the dark as the grave clock runs out.
        fade = 1.0 - 0.5 * min(1.0, float(getattr(s, "dead_t", 0.0)) / GRAVE_DELAY)
        base = _dim(base, fade)

    far = base if silhouette else _dim(base, _FAR_DIM)
    # Thin limbs, one step heavier for the torso. At this scale a uniform 2px
    # figure fuses into a blob - the weight difference is what keeps the arms
    # and legs readable against the body.
    w = 1 if h < 30.0 else 2
    tw = w + 1

    # far limbs -> torso -> near limbs -> head: cheapest possible depth cue
    _line(surf, far, sk.hip, sk.knees[1], w)
    _line(surf, far, sk.knees[1], sk.feet[1], w)
    _line(surf, far, sk.shoulder, sk.elbows[1], w)
    _line(surf, far, sk.elbows[1], sk.hands[1], w)

    _line(surf, base, sk.hip, sk.neck, tw)

    _line(surf, base, sk.hip, sk.knees[0], w)
    _line(surf, base, sk.knees[0], sk.feet[0], w)
    _line(surf, base, sk.shoulder, sk.elbows[0], w)
    _line(surf, base, sk.elbows[0], sk.hands[0], w)

    _disc(surf, base, sk.head, sk.head_r)

    if getattr(s, "carrying", None) and int(getattr(s, "carry_qty", 0) or 0) > 0:
        _draw_carried(surf, s, sk, base, silhouette)
    if getattr(s, "holds_candle", False) and getattr(s, "alive", True):
        _draw_candle(surf, s, sk, t, base)
    elif getattr(s, "holds_torch", False) and getattr(s, "alive", True):
        _draw_torch(surf, s, sk, t, base)
    speech = getattr(s, "speech", None)
    if isinstance(speech, str) and speech and getattr(s, "alive", True):
        _draw_bubble(surf, sk, speech, silhouette)


# ------------------------------------------------------------ held items --
def _draw_carried(surf: pygame.Surface, s: Stickman, sk: Skeleton,
                  base: Sequence[int], silhouette: bool) -> None:
    kind = str(getattr(s, "carrying", "") or "")
    col = base if silhouette else RESOURCE_COLORS.get(kind, (170, 170, 170))
    hx, hy = sk.front_hand
    # Bias the icon a little further out so it clears the hand joint.
    r = max(1.4, sk.height * 0.085)
    _icon(surf, kind, hx + sk.facing * r * 0.5, hy, r, col, base if silhouette else None)


def _icon(surf: pygame.Surface, kind: str, cx: float, cy: float, r: float,
          col: Sequence[int], flat: Sequence[int] | None = None) -> None:
    """Tiny in-hand resource icon. *flat* forces one colour (silhouette)."""
    edge = flat if flat is not None else _dim(col, 0.55)
    if kind == RES_WOOD:
        pts = [(cx - r, cy - r * 0.45), (cx + r, cy - r * 0.75),
               (cx + r, cy + r * 0.15), (cx - r, cy + r * 0.45)]
        pygame.draw.polygon(surf, col, pts)
        pygame.draw.lines(surf, edge, True, [(int(p[0]), int(p[1])) for p in pts], 1)
    elif kind == RES_STONE:
        pts = [(cx - r, cy), (cx - r * 0.4, cy - r), (cx + r * 0.6, cy - r * 0.8),
               (cx + r, cy + r * 0.2), (cx, cy + r)]
        pygame.draw.polygon(surf, col, pts)
    elif kind == RES_FIBRE:
        for k in (-1, 0, 1):
            _line(surf, col, (cx - r * 0.8 + k * 0.6, cy + r),
                  (cx + r * 0.6 + k * 0.6, cy - r), 1)
    elif kind == RES_COOKED:
        _disc(surf, col, (cx, cy), r)
        _disc(surf, edge, (cx, cy), r * 0.45)
    else:                                     # food / berries / anything else
        _disc(surf, col, (cx, cy), r * 0.9)


def _draw_candle(surf: pygame.Surface, s: Stickman, sk: Skeleton,
                 t: float, base: Sequence[int]) -> None:
    """The candle and its flame. Always drawn lit - a flame is its own light."""
    hx, hy = sk.front_hand
    stick = max(2.2, sk.height * 0.17)
    tip = (hx + sk.facing * stick * 0.16, hy - stick)
    _line(surf, _CANDLE_WAX, (hx, hy), tip, 2 if sk.height >= 17 else 1)

    idp = float(getattr(s, "id", 0)) * 1.31
    flick = (0.55 * math.sin(t * 17.3 + idp)
             + 0.30 * math.sin(t * 29.7 + idp * 2.1)
             + 0.15 * math.sin(t * 43.1 + idp * 0.7))
    r = max(0.9, sk.height * 0.075 * (1.0 + 0.28 * flick))
    lean = sk.facing * 0.35 * flick
    fc = (tip[0] + lean, tip[1] - r * 0.75)
    _disc(surf, _FLAME_OUTER, fc, r)
    _disc(surf, _FLAME_CORE, (fc[0] + lean * 0.3, fc[1] + r * 0.18), r * 0.5)


def _draw_torch(surf: pygame.Surface, s: Stickman, sk: Skeleton,
                t: float, base: Sequence[int]) -> None:
    """A torch: a longer haft with a wrapped, pitch-soaked head and a broad
    flame. Bigger and rowdier than the elder's candle, so a whole colony of
    them reads as a scatter of firelight rather than a single steady point."""
    hx, hy = sk.front_hand
    shaft = max(3.0, sk.height * 0.34)
    tip = (hx + sk.facing * shaft * 0.22, hy - shaft)
    w = 2 if sk.height >= 17 else 1
    _line(surf, _TORCH_SHAFT, (hx, hy), tip, w)
    # the soaked head sits just below the flame
    head = (tip[0], tip[1] + max(1.0, sk.height * 0.05))
    _disc(surf, _TORCH_HEAD, head, max(1.0, sk.height * 0.06))

    idp = float(getattr(s, "id", 0)) * 1.71
    flick = (0.55 * math.sin(t * 15.1 + idp)
             + 0.30 * math.sin(t * 26.3 + idp * 2.3)
             + 0.15 * math.sin(t * 39.7 + idp * 0.5))
    r = max(1.1, sk.height * 0.11 * (1.0 + 0.32 * flick))
    lean = sk.facing * 0.4 * flick
    fc = (tip[0] + lean, tip[1] - r * 0.7)
    _disc(surf, _TORCH_FLAME, fc, r)
    _disc(surf, _FLAME_CORE, (fc[0] + lean * 0.3, fc[1] + r * 0.2), r * 0.45)


# --------------------------------------------------------- speech bubble --
def _draw_bubble(surf: pygame.Surface, sk: Skeleton, symbol: str,
                 silhouette: bool) -> None:
    bg = _dim(_BUBBLE_BG, 0.28) if silhouette else _BUBBLE_BG
    ink = (18, 20, 26) if silhouette else _BUBBLE_INK
    edge = (12, 13, 18) if silhouette else _BUBBLE_EDGE

    bw, bh = 14, 11
    hx, hy = sk.head
    side = sk.facing
    cx = hx + side * (sk.head_r + 7.0)
    cy = hy - sk.head_r - 6.0
    if cx - bw * 0.5 < 1 or cx + bw * 0.5 > RENDER_W - 1:      # flip if clipped
        side = -side
        cx = hx + side * (sk.head_r + 7.0)
    cx = min(max(cx, bw * 0.5 + 1), RENDER_W - bw * 0.5 - 1)
    cy = max(cy, bh * 0.5 + 1)

    rect = pygame.Rect(int(cx - bw / 2), int(cy - bh / 2), bw, bh)
    tail = [(hx + side * sk.head_r * 0.6, hy - sk.head_r * 0.5),
            (rect.centerx - 2, rect.bottom - 1),
            (rect.centerx + 2, rect.bottom - 1)]
    pygame.draw.polygon(surf, bg, [(int(p[0]), int(p[1])) for p in tail])
    pygame.draw.rect(surf, bg, rect, border_radius=3)
    pygame.draw.rect(surf, edge, rect, 1, border_radius=3)
    _glyph(surf, symbol, rect.centerx + 0.5, rect.centery + 0.5, 3.2, ink, silhouette)


#: Speech symbols understood by :func:`_glyph`, plus their aliases.
_GLYPH_ALIASES: dict[str, str] = {
    "berry": "food", RES_FOOD: "food", "eat": "food",
    "log": "wood", RES_WOOD: "wood", "tree": "wood",
    "rock": "stone", RES_STONE: "stone",
    RES_COOKED: "cooked", "meal": "cooked",
    RES_FIBRE: "fibre",
    "rain": "water", "drink": "water",
    "flame": "fire", "burn": "fire",
    "home": "hut", "house": "hut", "build": "hut",
    "zzz": "sleep", "tired": "sleep",
    "love": "heart", "friend": "heart",
    "song": "music", "talk": "music",
    "skull": "death", "grave": "death", "mourn": "death",
    "danger": "warn", "alarm": "warn", "!": "warn",
    "hi": "wave", "hello": "wave",
    "?": "question", "huh": "question",
    "…": "dots", "...": "dots", "think": "dots",
    # sim/actions.py emits terse single-character symbols; give each a shape
    # so a bubble is never an empty box.
    "~": "dots",        # chatter
    "*": "star",        # something notable
    "+": "plus",        # good / more
    "o": "dot",
    "^": "up",          # up there / higher ground
    "#": "hut",         # structure
}

_GLYPH_TINT: dict[str, tuple[int, int, int]] = {
    "food": (198, 62, 62), "wood": (140, 94, 54), "stone": (120, 124, 132),
    "cooked": (206, 142, 66), "fibre": (168, 156, 104), "water": (92, 152, 214),
    "fire": (226, 118, 44), "heart": (214, 76, 104), "warn": (226, 176, 60),
}


def _glyph(surf: pygame.Surface, key: str, cx: float, cy: float, r: float,
           ink: Sequence[int], silhouette: bool) -> None:
    name = _GLYPH_ALIASES.get(key, key if key in _GLYPH_DRAW else "dot")
    col = ink
    if not silhouette:
        tint = _GLYPH_TINT.get(name)
        if tint is not None:
            col = tint
    fn = _GLYPH_DRAW.get(name, _g_dot)
    try:
        fn(surf, col, cx, cy, r)
    except Exception:
        log.debug("glyph %r failed", key, exc_info=True)


def _g_dot(surf, col, cx, cy, r):
    _disc(surf, col, (cx, cy), r * 0.4)


def _g_question(surf, col, cx, cy, r):
    pts = [(cx - r * 0.55, cy - r * 0.5), (cx - r * 0.1, cy - r),
           (cx + r * 0.55, cy - r * 0.55), (cx + r * 0.1, cy),
           (cx, cy + r * 0.3)]
    pygame.draw.aalines(surf, col, False, pts)
    _disc(surf, col, (cx, cy + r * 0.85), 0.8)


def _g_warn(surf, col, cx, cy, r):
    _line(surf, col, (cx, cy - r), (cx, cy + r * 0.25), 1)
    _disc(surf, col, (cx, cy + r * 0.8), 0.8)


def _g_dots(surf, col, cx, cy, r):
    for k in (-1, 0, 1):
        _disc(surf, col, (cx + k * r * 0.7, cy + r * 0.3), 0.75)


def _g_food(surf, col, cx, cy, r):
    _disc(surf, col, (cx, cy + r * 0.2), r * 0.72)
    _line(surf, col, (cx, cy - r * 0.45), (cx + r * 0.5, cy - r), 1)


def _g_wood(surf, col, cx, cy, r):
    pts = [(cx - r, cy - r * 0.35), (cx + r, cy - r * 0.7),
           (cx + r, cy + r * 0.35), (cx - r, cy + r * 0.7)]
    pygame.draw.polygon(surf, col, [(int(p[0]), int(p[1])) for p in pts])


def _g_stone(surf, col, cx, cy, r):
    pts = [(cx - r, cy + r * 0.3), (cx - r * 0.45, cy - r * 0.8),
           (cx + r * 0.5, cy - r * 0.85), (cx + r, cy + r * 0.15),
           (cx + r * 0.2, cy + r * 0.9)]
    pygame.draw.polygon(surf, col, [(int(p[0]), int(p[1])) for p in pts])


def _g_cooked(surf, col, cx, cy, r):
    _disc(surf, col, (cx, cy + r * 0.35), r * 0.75)
    _line(surf, col, (cx - r * 0.4, cy - r * 0.45), (cx - r * 0.15, cy - r), 1)
    _line(surf, col, (cx + r * 0.35, cy - r * 0.45), (cx + r * 0.6, cy - r), 1)


def _g_fibre(surf, col, cx, cy, r):
    for k in (-1, 0, 1):
        _line(surf, col, (cx - r * 0.7 + k * r * 0.45, cy + r),
              (cx + r * 0.3 + k * r * 0.45, cy - r), 1)


def _g_water(surf, col, cx, cy, r):
    pygame.draw.polygon(surf, col, [(int(cx), int(cy - r)),
                                    (int(cx - r * 0.75), int(cy + r * 0.2)),
                                    (int(cx + r * 0.75), int(cy + r * 0.2))])
    _disc(surf, col, (cx, cy + r * 0.25), r * 0.7)


def _g_fire(surf, col, cx, cy, r):
    pygame.draw.polygon(surf, col, [(int(cx), int(cy - r)),
                                    (int(cx + r * 0.8), int(cy + r * 0.8)),
                                    (int(cx - r * 0.8), int(cy + r * 0.8))])


def _g_hut(surf, col, cx, cy, r):
    pygame.draw.polygon(surf, col, [(int(cx), int(cy - r)),
                                    (int(cx + r), int(cy - r * 0.1)),
                                    (int(cx - r), int(cy - r * 0.1))])
    pygame.draw.rect(surf, col, pygame.Rect(int(cx - r * 0.6), int(cy - r * 0.1),
                                            max(1, int(r * 1.2)), max(1, int(r))))


def _g_sleep(surf, col, cx, cy, r):
    _line(surf, col, (cx - r * 0.7, cy - r * 0.7), (cx + r * 0.7, cy - r * 0.7), 1)
    _line(surf, col, (cx + r * 0.7, cy - r * 0.7), (cx - r * 0.7, cy + r * 0.75), 1)
    _line(surf, col, (cx - r * 0.7, cy + r * 0.75), (cx + r * 0.7, cy + r * 0.75), 1)


def _g_heart(surf, col, cx, cy, r):
    _disc(surf, col, (cx - r * 0.42, cy - r * 0.3), r * 0.5)
    _disc(surf, col, (cx + r * 0.42, cy - r * 0.3), r * 0.5)
    pygame.draw.polygon(surf, col, [(int(cx - r * 0.9), int(cy - r * 0.1)),
                                    (int(cx + r * 0.9), int(cy - r * 0.1)),
                                    (int(cx), int(cy + r))])


def _g_music(surf, col, cx, cy, r):
    _disc(surf, col, (cx - r * 0.35, cy + r * 0.55), r * 0.45)
    _line(surf, col, (cx + r * 0.05, cy + r * 0.55), (cx + r * 0.05, cy - r), 1)
    _line(surf, col, (cx + r * 0.05, cy - r), (cx + r * 0.7, cy - r * 0.55), 1)


def _g_death(surf, col, cx, cy, r):
    _disc(surf, col, (cx, cy - r * 0.15), r * 0.8)
    _line(surf, col, (cx - r * 0.5, cy + r * 0.7), (cx + r * 0.5, cy + r * 0.7), 1)


def _g_wave(surf, col, cx, cy, r):
    _line(surf, col, (cx, cy + r), (cx, cy - r * 0.2), 1)
    for a in (-0.7, 0.0, 0.7):
        _line(surf, col, (cx, cy - r * 0.2),
              (cx + math.sin(a) * r, cy - r * 0.2 - math.cos(a) * r * 0.8), 1)


def _g_star(surf, col, cx, cy, r):
    pygame.draw.polygon(surf, col, [
        (int(cx), int(cy - r)), (int(cx + r * 0.3), int(cy - r * 0.3)),
        (int(cx + r), int(cy)), (int(cx + r * 0.3), int(cy + r * 0.3)),
        (int(cx), int(cy + r)), (int(cx - r * 0.3), int(cy + r * 0.3)),
        (int(cx - r), int(cy)), (int(cx - r * 0.3), int(cy - r * 0.3)),
    ])


def _g_plus(surf, col, cx, cy, r):
    _line(surf, col, (cx - r * 0.85, cy), (cx + r * 0.85, cy), 1)
    _line(surf, col, (cx, cy - r * 0.85), (cx, cy + r * 0.85), 1)


def _g_up(surf, col, cx, cy, r):
    _line(surf, col, (cx - r * 0.8, cy + r * 0.25), (cx, cy - r * 0.7), 1)
    _line(surf, col, (cx, cy - r * 0.7), (cx + r * 0.8, cy + r * 0.25), 1)


_GLYPH_DRAW: dict[str, Callable[..., None]] = {
    "dot": _g_dot, "question": _g_question, "warn": _g_warn, "dots": _g_dots,
    "food": _g_food, "wood": _g_wood, "stone": _g_stone, "cooked": _g_cooked,
    "fibre": _g_fibre, "water": _g_water, "fire": _g_fire, "hut": _g_hut,
    "sleep": _g_sleep, "heart": _g_heart, "music": _g_music, "death": _g_death,
    "wave": _g_wave, "star": _g_star, "plus": _g_plus, "up": _g_up,
}


# ------------------------------------------------------------------ grave --
def draw_grave(surf: pygame.Surface, prop: Any,
               alpha_color: tuple[int, int, int] | None = None) -> None:
    """Draw a gravestone.

    *prop* is duck-typed so props.py owns the real class: anything with ``x``
    and ``y`` attributes (or a dict with those keys) works. An optional
    ``color`` is used as an identity chip on the stone, and an optional
    ``name`` seeds the stone's lean so a row of graves is not uniform.
    """
    try:
        _draw_grave(surf, prop, alpha_color)
    except Exception:
        log.exception("draw_grave failed")


def _get(prop: Any, key: str, default: Any = None) -> Any:
    if isinstance(prop, dict):
        return prop.get(key, default)
    return getattr(prop, key, default)


def _draw_grave(surf: pygame.Surface, prop: Any,
                alpha_color: tuple[int, int, int] | None) -> None:
    try:
        x = float(_get(prop, "x", 0.0))
        y = float(_get(prop, "y", 0.0))
    except Exception:
        return
    if not (math.isfinite(x) and math.isfinite(y)):
        return
    if x < -24 or x > RENDER_W + 24 or y < -24 or y > RENDER_H + 24:
        return

    silhouette = alpha_color is not None
    stone = tuple(int(c) for c in alpha_color[:3]) if silhouette else _GRAVE_STONE
    shade = stone if silhouette else _GRAVE_SHADE
    mound = stone if silhouette else _GRAVE_MOUND

    h = AGENT_HEIGHT * 0.62
    w = h * 0.62
    # Deterministic lean from position: no rng, no state, stable across frames.
    tilt = math.sin(x * 0.37 + y * 0.11) * 0.13
    st, ct = math.sin(tilt), math.cos(tilt)

    def pt(lx: float, ly: float) -> tuple[int, int]:
        return (int(round(x + lx * ct - ly * st)), int(round(y + lx * st + ly * ct)))

    # earth mound first
    pygame.draw.ellipse(
        surf, mound,
        pygame.Rect(int(x - w * 1.15), int(y - 2), max(2, int(w * 2.3)), 5),
    )

    body = [pt(-w * 0.5, 0.0), pt(-w * 0.5, -h * 0.72),
            pt(-w * 0.28, -h), pt(w * 0.28, -h),
            pt(w * 0.5, -h * 0.72), pt(w * 0.5, 0.0)]
    pygame.draw.polygon(surf, stone, body)
    pygame.draw.aalines(surf, shade, True, body)
    # incised line, and the identity chip of whoever lies here
    _line(surf, shade, pt(-w * 0.26, -h * 0.34), pt(w * 0.26, -h * 0.34), 1)
    chip = _get(prop, "color", None)
    if chip is not None and not silhouette:
        try:
            c = (int(chip[0]), int(chip[1]), int(chip[2]))
            _disc(surf, c, pt(0.0, -h * 0.62), max(1.0, w * 0.16))
        except Exception:
            pass


if __name__ == "__main__":   # pragma: no cover - visual smoke test
    import os
    import random

    from ..sim.entities import Population

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    CELL_W, CELL_H, COLS = 96, 92, 6
    rows = (len(POSES) + 2 + COLS - 1) // COLS
    sheet = pygame.Surface((CELL_W * COLS, CELL_H * rows))
    sheet.fill((16, 18, 24))

    rng = random.Random(4)
    pop = Population()
    cells = list(POSES) + ["dead", "silhouette"]
    for i, pose in enumerate(cells):
        cx = (i % COLS) * CELL_W + CELL_W // 2
        cy = (i // COLS) * CELL_H + CELL_H - 22
        a = pop.spawn(cx, cy, rng=rng)
        a.anim_t = 1.7
        a.vx = 20.0
        a.fall_t = 0.42
        a.carrying, a.carry_qty = RES_WOOD, 2
        a.holds_candle = (i % 3 == 0)
        a.say(("food", "?", "sleep", "heart", "fire", "hut")[i % 6], 9.9)
        if pose == "dead":
            a.alive = False
        elif pose != "silhouette":
            a.pose_override = pose
        draw_stickman(sheet, a, 1.35,
                      SILHOUETTE if pose == "silhouette" else None)
        pygame.draw.line(sheet, (40, 44, 52), (cx - 34, cy), (cx + 34, cy))

    for k in range(3):
        draw_grave(sheet, {"x": 26 + k * 16, "y": CELL_H - 22,
                           "color": (232, 106, 106), "name": "Vessa"})

    import tempfile
    out = os.path.join(tempfile.gettempdir(), "backgrounded_poses.png")
    pygame.image.save(sheet, out)
    print("wrote", out, sheet.get_size(), "poses:", ", ".join(cells))
    pygame.quit()
