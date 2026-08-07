"""Animals, the saucer, and the gear a stickman carries.

Everything the *combat* layer put into the world needs a body. This module
draws three of them:

* :func:`draw_animals` - procedural quadrupeds (wolf / bear / boar) posed from
  ``anim_t`` with two-bone IK legs, in the same thin-AA-line idiom as
  ``stickfigure.py``. Wolves are lean and long-snouted, bears are humped and
  slow, boars are low and wide with tusks.
* :func:`draw_ufo` - a saucer with a dark dome, cycling rim lights and an
  additive abduction beam. The abductee rises inside the beam.
* :func:`draw_gear` - the spear, the leather armour and whichever of the five
  dragon relics the agent is wearing, drawn onto an existing stick figure.
  Called from the stickman draw path. The relic *drawings* live in
  ``render/items.py``; what lives here is the decision of when to ask for one
  and how it stacks with the jerkin, because this is the function that already
  owns the torso.

**Read-only w.r.t. sim state.** Nothing here writes to a World, an agent, an
animal or the saucer - ``lift_t`` in particular is *read* to place the abductee
and never advanced. Every value that is animation-only (rim-light cycle, saucer
wobble, tail wag) is derived from the render clock ``t`` and the entity id.

Layer wiring
------------
``draw_animals`` belongs with the agents at **step 7**, before the light
composite: an animal outside every light source should fall to a silhouette
exactly like a stickman does (feature 36), and this module does that itself by
querying ``world.lighting``.

``draw_ufo`` belongs at **step 11**, *after* the light composite, next to the
lightning geometry. Its beam is self-lit; drawn before the composite it would
be multiplied down to nothing on a night-storm ambient of ~0.06.

If the renderer's agent pass also draws abducted agents, skip agents with
``taken`` set there - the beam draws them at the height ``lift_t`` implies.

Duck typing
-----------
The animal and saucer sim objects are owned by ``sim/``, which this module is
not allowed to import from beyond ``entities``. Every field is read through
:func:`fx.attr_num` or ``getattr`` with a fallback, and several plausible names
are tried for each (``kind``/``species``, ``state``/``mode``/``phase``), so a
rename on the sim side degrades to a default pose rather than a traceback.

Caching
-------
The beam cone and the saucer body are numpy/blit-baked once and cached by
bucketed size, so the per-frame path is a handful of blits and ~40 AA lines per
animal. There are no per-pixel Python loops anywhere in here.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pygame

from ..constants import (
    ANIMAL_BEAR,
    ANIMAL_BOAR,
    ANIMAL_STATS,
    ANIMAL_WOLF,
    ARMOUR_LEATHER,
    RELIC_KINDS,
    RENDER_H,
    UFO_BEAM_SEC,
    WEAPON_SPEAR,
)
from . import fx
from .camera import IDENTITY, Camera

log = logging.getLogger(__name__)

__all__ = [
    "draw_animals", "draw_ufo", "draw_gear", "draw_mining_dust",
    "animal_kind_of", "clear_caches", "cache_stats",
]

TAU = math.tau
Color = tuple[int, int, int]

# Mirrors renderer.SILHOUETTE_CUTOFF / SILHOUETTE_COLOR. Deliberately a copy
# rather than an import: renderer imports this module, and a cycle in the
# render package would be paid at startup for two scalars.
_SIL_CUTOFF = 0.30
_SIL_COLOR: Color = (14, 15, 22)

_DEFAULT_GROUND_Y = RENDER_H * 0.72

# pygame-ce takes a width on aaline; older pygame does not. Probed once at
# import so the per-frame path carries no version check. Same probe as
# stickfigure.py - duplicated rather than imported to keep this module free of
# a render-package import cycle.
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


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def _dim(col: Sequence[int], f: float) -> Color:
    return (int(col[0] * f), int(col[1] * f), int(col[2] * f))


def _lit(col: Sequence[int], f: float) -> Color:
    """Brighten toward white by *f*."""
    return (
        int(min(255.0, col[0] + (255 - col[0]) * f)),
        int(min(255.0, col[1] + (255 - col[1]) * f)),
        int(min(255.0, col[2] + (255 - col[2]) * f)),
    )


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
    if r <= 0.0:
        return
    if _AACIRCLE is not None:
        _AACIRCLE(surf, col, c, max(1.0, r), width)
    else:                                             # pragma: no cover
        pygame.draw.circle(surf, col, (int(c[0]), int(c[1])), max(1, int(r)), width)


def _poly(surf: pygame.Surface, col: Sequence[int],
          pts: Sequence[tuple[float, float]]) -> None:
    if len(pts) >= 3:
        pygame.draw.polygon(surf, col, [(int(round(p[0])), int(round(p[1]))) for p in pts])


def _ik2(root: tuple[float, float], target: tuple[float, float],
         l1: float, l2: float, sign: float
         ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Two-bone IK. Returns ``(joint, end)``.

    *sign* picks which side the joint bends to: with the chain hanging
    downward, ``+1`` bends it toward -x (backward) and ``-1`` toward +x. A
    quadruped's front elbow bends backward and its hind stifle forward, which
    is the entire difference between "dog" and "table".

    The target is pulled in if it is out of reach, so the returned *end* is not
    always the requested one - callers draw the returned value.
    """
    rx, ry = root
    dx, dy = target[0] - rx, target[1] - ry
    d = math.hypot(dx, dy)
    if d < 1e-4:
        return ((rx, ry + l1), (rx, ry + l1 + l2))
    lo, hi = abs(l1 - l2) + 1e-3, l1 + l2 - 1e-3
    if d < lo or d > hi:
        d2 = min(max(d, lo), hi)
        sc = d2 / d
        dx, dy, d = dx * sc, dy * sc, d2
    ux, uy = dx / d, dy / d
    a = (d * d + l1 * l1 - l2 * l2) / (2.0 * d)
    h = math.sqrt(max(0.0, l1 * l1 - a * a))
    px, py = -uy, ux
    joint = (rx + ux * a + px * h * sign, ry + uy * a + py * h * sign)
    return (joint, (rx + dx, ry + dy))


def _ground_y(world: Any, x: float) -> float:
    try:
        gy = float(world.terrain.ground_y(x))
        if math.isfinite(gy):
            return gy
    except Exception:
        pass
    return _DEFAULT_GROUND_Y


def _light_at(world: Any, x: float, y: float) -> float:
    try:
        return float(world.lighting.light_at(x, y))
    except Exception:
        return 1.0


# --------------------------------------------------------------------------
# animal shape table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Build:
    """Everything that makes one kind look like itself.

    Lengths are pixels at the scale a stickman is ``AGENT_HEIGHT`` (26 px)
    tall, so a bear's shoulder comes up to a person's chest and a boar's to
    their knee. ``height`` is the shoulder height above the ground; legs are
    derived from it so the feet always reach.
    """

    length: float          # spine length, chest to rump
    height: float          # shoulder height above the ground
    hump: float            # extra rise over the shoulders (bear)
    rump: float            # hip height as a fraction of shoulder height
    chest: float           # belly depth under the chest, fraction of height
    tuck: float            # belly height at the loin - larger = more tucked up
    head_r: float
    neck_x: float          # where the neck leaves the body, fraction of length
    neck_y: float          # ... and its height, fraction of shoulder height
    neck_len: float
    neck_ang: float        # radians from horizontal, + = head down
    snout: float
    ear: float             # 0 = none
    ear_round: bool
    tail_len: float
    tail_ang: float        # radians above straight-back at rest
    stride: float          # foot travel per cycle
    lift: float            # peak foot lift
    gait_hz: float         # stride cycles per second of anim_t
    phase: tuple[float, float, float, float]   # front-near, front-far, rear-near, rear-far
    bend: float            # leg bone length as a multiple of the reach needed
    leg_w: int             # leg line weight
    haunch: float          # shoulder/haunch mass radius, fraction of height
    fur: Color
    rim: Color
    tusks: bool = False


BUILDS: dict[str, _Build] = {
    # Lean, low, deep chest and a tucked-up loin, long snout carried ahead of
    # the shoulders, brush tail, light diagonal trot.
    ANIMAL_WOLF: _Build(
        length=22.0, height=13.5, hump=0.6, rump=0.93,
        chest=0.46, tuck=0.62,
        head_r=2.7, neck_x=0.40, neck_y=0.90, neck_len=6.8, neck_ang=0.24, snout=5.4,
        ear=2.6, ear_round=False,
        tail_len=9.5, tail_ang=0.30,
        stride=6.2, lift=2.6, gait_hz=2.55,
        phase=(0.0, math.pi, math.pi, 0.0),
        bend=1.13, leg_w=2, haunch=0.30,
        fur=(126, 128, 140), rim=(190, 196, 210),
    ),
    # Bulky and barrel-bellied, humped shoulders, head slung low in front of
    # them, stub tail, a slow heavy amble.
    ANIMAL_BEAR: _Build(
        length=27.0, height=19.0, hump=4.2, rump=0.78,
        chest=0.42, tuck=0.46,
        head_r=3.4, neck_x=0.44, neck_y=0.80, neck_len=7.6, neck_ang=0.30, snout=4.8,
        ear=2.4, ear_round=True,
        tail_len=3.0, tail_ang=-0.20,
        stride=6.4, lift=2.0, gait_hz=1.35,
        phase=(0.0, math.pi, math.pi * 1.25, 0.25 * math.pi),
        bend=1.10, leg_w=3, haunch=0.34,
        fur=(104, 76, 52), rim=(164, 128, 88),
    ),
    # Low, wide and front-heavy: big shoulders, small hindquarters, head down
    # at the ground, tusks, a scuttling little gait.
    ANIMAL_BOAR: _Build(
        length=20.0, height=12.0, hump=2.4, rump=0.72,
        chest=0.40, tuck=0.44,
        head_r=2.9, neck_x=0.44, neck_y=0.86, neck_len=5.2, neck_ang=0.66, snout=5.6,
        ear=2.0, ear_round=False,
        tail_len=4.5, tail_ang=0.55,
        stride=5.4, lift=1.8, gait_hz=3.05,
        phase=(0.0, math.pi, math.pi, 0.0),
        bend=1.15, leg_w=2, haunch=0.30,
        fur=(98, 86, 74), rim=(152, 138, 120),
        tusks=True,
    ),
}

_TUSK: Color = (196, 190, 172)
_PIP_BACK: Color = (16, 17, 22)
_PIP_FILL: Color = (214, 78, 70)
_PIP_EDGE: Color = (70, 24, 22)

#: Words that mean "this animal is running away", checked against whatever
#: state string the sim exposes. Kept generous on purpose - the sim side owns
#: that vocabulary and this module must not dictate it.
_FLEE_WORDS = ("flee", "leav", "retreat", "scare", "panic", "rout", "wander_off")
_CHARGE_WORDS = ("charge", "attack", "hunt", "chase", "strike", "maul")

_STATE_ATTRS = ("state", "mode", "phase", "behaviour", "behavior", "activity", "stance")


def _state_of(obj: Any) -> str:
    """Best-effort lowercase state word for an animal or the saucer."""
    for name in _STATE_ATTRS:
        try:
            val = getattr(obj, name, None)
        except Exception:
            continue
        if isinstance(val, str) and val:
            return val.lower()
    # An action-object style state, e.g. `animal.action.kind`.
    for name in ("action", "goal", "task"):
        try:
            kind = getattr(getattr(obj, name, None), "kind", None)
        except Exception:
            continue
        if isinstance(kind, str) and kind:
            return kind.lower()
    return ""


def animal_kind_of(a: Any) -> str:
    """Which :data:`BUILDS` entry to draw *a* with. Always returns a valid key."""
    for name in ("kind", "species", "animal", "type", "name"):
        try:
            val = getattr(a, name, None)
        except Exception:
            continue
        if isinstance(val, str):
            k = val.strip().lower()
            if k in BUILDS:
                return k
    return ANIMAL_WOLF


def _iter_animals(world: Any) -> list[Any]:
    """Find the animal collection wherever the sim decided to hang it."""
    src: Any = None
    events = getattr(world, "events", None)
    for holder, name in ((world, "animals"), (events, "animals"),
                         (world, "wildlife"), (events, "wildlife"),
                         (world, "fauna")):
        if holder is None:
            continue
        try:
            val = getattr(holder, name, None)
        except Exception:
            continue
        if val is not None:
            src = val
            break
    if src is None:
        return []
    # Registry-shaped containers (like PropRegistry) expose their contents
    # through a method rather than by being iterable.
    if not isinstance(src, (list, tuple)):
        for meth in ("all", "alive", "values"):
            fn = getattr(src, meth, None)
            if callable(fn):
                try:
                    src = fn()
                    break
                except Exception:
                    continue
    try:
        return [a for a in src if a is not None]
    except TypeError:
        return []


# --------------------------------------------------------------------------
# animals
# --------------------------------------------------------------------------


def draw_animals(surf: pygame.Surface, world: Any, t: float, *,
                 cam: Camera) -> None:
    """Draw every animal in *world*. Fails soft, per animal and overall.

    *cam* is REQUIRED and has no default: an animal's ``x`` is a world x on a
    6400 px map and *surf* is a 1600 px frame, so a caller that forgets it puts
    three quarters of the wildlife off the edge and misplaces the rest. It used
    to be optional, and every wolf in the game was invisible for it.
    """
    try:
        animals = _iter_animals(world)
    except Exception:
        log.exception("could not read the animal list")
        return
    if not animals:
        return
    try:
        animals = sorted(animals, key=lambda a: fx.attr_num(a, "y", default=0.0))
    except Exception:
        pass
    for a in animals:
        try:
            _draw_animal(surf, a, world, float(t), cam)
        except Exception:
            log.debug("animal draw failed", exc_info=True)


def _draw_animal(surf: pygame.Surface, a: Any, world: Any, t: float,
                 cam: Camera) -> None:
    wx = fx.attr_num(a, "x", default=math.nan)
    y = fx.attr_num(a, "y", default=math.nan)
    if not (math.isfinite(wx) and math.isfinite(y)):
        return
    # Cull in SCREEN space. The old test compared a world x against RENDER_W,
    # which on a 6400 px map means "anything past the first quarter of the land
    # is gone" - and the quarter that survived was drawn at cam.x px off.
    if not cam.visible(wx, 80.0):
        return
    if y < -80.0 or y > RENDER_H + 80.0:
        return
    # The one conversion. `wx` stays world below, because lighting.light_at
    # samples the sim's own sources and those live in the world; `x` is screen
    # and is what every draw call gets.
    x = cam.sx(wx)

    kind = animal_kind_of(a)
    b = BUILDS[kind]
    alive = bool(getattr(a, "alive", True))
    state = _state_of(a)
    fleeing = bool(getattr(a, "fleeing", False)) or any(w in state for w in _FLEE_WORDS)
    charging = (not fleeing) and any(w in state for w in _CHARGE_WORDS)

    # No usable `facing` (missing, zero, or something that is not a number)
    # falls back to the direction of travel, so an animal never moons the
    # colony while walking at it.
    face = fx.attr_num(a, "facing", default=0.0)
    if face == 0.0:
        face = fx.attr_num(a, "vx", "dx", default=1.0)
    facing = -1.0 if face < 0.0 else 1.0

    # Silhouette rule, same as the stickmen: outside every light pool an animal
    # is a shape, not a species. That is the point of feature 36 and it is what
    # makes a wolf at the edge of the candle frightening.
    lit = _light_at(world, wx, y - b.height * 0.5)
    silhouette = lit < _SIL_CUTOFF
    fur: Color = _SIL_COLOR if silhouette else b.fur
    rim: Color = _dim(_SIL_COLOR, 1.8) if silhouette else b.rim
    if not alive:
        fur = _dim(fur, 0.55)
        rim = _dim(rim, 0.45)
    far = fur if silhouette else _dim(fur, 0.72)

    # Animation phase. anim_t is the sim's accumulator; if the animal has none
    # we fabricate one from the clock so it still moves rather than freezing
    # mid-stride.
    anim = fx.attr_num(a, "anim_t", default=-1.0)
    idp = (fx.attr_num(a, "id", default=0.0) * 0.9137) % TAU
    if anim < 0.0:
        anim = t
    rate = b.gait_hz * (1.55 if fleeing else 1.0)
    ph = anim * TAU * rate + idp

    if not alive:
        _draw_animal_dead(surf, x, y, b, facing, fur, far)
        return

    _draw_animal_body(surf, x, y, b, facing, ph, fur, far, rim,
                      fleeing=fleeing, charging=charging, t=t, idp=idp)

    maxh = fx.attr_num(a, "max_health", "maxhp", "health_max", default=0.0)
    if maxh <= 0.0:
        try:
            maxh = float(ANIMAL_STATS[kind][0])
        except Exception:
            maxh = 100.0
    health = fx.attr_num(a, "health", "hp", default=maxh)
    if health < maxh * 0.995:
        top = y - (b.height + b.hump) - b.head_r - 5.0
        # The pip obeys the light too. Drawn at full strength it survived the
        # light composite when the animal under it did not, so a night scene
        # grew little red bars floating over nothing at all.
        _health_pip(surf, x, top, health / maxh if maxh > 0 else 0.0,
                    0.30 if silhouette else 1.0)


def _draw_animal_body(surf: pygame.Surface, x: float, y: float, b: _Build,
                      facing: float, ph: float, fur: Color, far: Color,
                      rim: Color, *, fleeing: bool, charging: bool,
                      t: float, idp: float) -> None:
    """The whole quadruped, in local space then transformed once.

    Local space has its origin at the ground under the animal, +x pointing the
    way it faces and -y up, exactly like ``stickfigure``'s figure space. The
    single ``xf`` at the end is what makes mirroring free.
    """
    L, H = b.length, b.height
    crouch = 0.93 if fleeing else 1.0          # a running animal flattens out
    H = H * crouch
    hump = b.hump
    bob = math.sin(ph * 2.0) * H * 0.022       # body rises on the double beat

    def xf(lx: float, ly: float) -> tuple[float, float]:
        return (x + lx * facing, y + ly + bob)

    # -- torso ------------------------------------------------------------
    # Nine points, front-top round to front-bottom. The two that do the work
    # are the chest (deep, at the front) and the tuck (high, at the loin):
    # without that diagonal the outline is a rectangle and the animal reads as
    # a table. Everything else is rounding.
    outline = [
        (L * 0.38, -(H + hump)),                          # 0 top of shoulder
        (L * 0.06, -(H * 0.97 + hump * 0.30)),            # 1 mid back
        (-L * 0.24, -(H * b.rump + H * 0.03)),            # 2 loin
        (-L * 0.46, -(H * b.rump * 0.86)),                # 3 rump
        (-L * 0.41, -(H * b.tuck * 0.78)),                # 4 back of the thigh
        (-L * 0.14, -(H * b.tuck)),                       # 5 belly, tucked up
        (L * 0.14, -(H * (b.tuck + b.chest) * 0.48)),     # 6 belly
        (L * 0.38, -(H * b.chest)),                       # 7 chest, deep
        (L * 0.50, -(H * (b.chest + 0.62) * 0.60)),       # 8 breast
    ]
    body = [xf(*p) for p in outline]

    # -- legs -------------------------------------------------------------
    # Roots sit inside the torso outline so the shoulder joint is hidden by the
    # body rather than poking out of it. The far pair is offset backward and
    # drawn slightly shorter, which is enough parallax at this scale.
    roots = (
        (L * 0.34, -(H * 0.86 + hump * 0.30), +1.0, 1.00),   # front near, elbow back
        (L * 0.27, -(H * 0.84 + hump * 0.30), +1.0, 0.96),   # front far
        (-L * 0.32, -(H * b.rump * 0.90), -1.0, 1.00),       # rear near, stifle fwd
        (-L * 0.26, -(H * b.rump * 0.88), -1.0, 0.96),       # rear far
    )
    stride = b.stride * (1.18 if fleeing else 1.0)
    legs: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    for i, (rx, ry, sign, scale) in enumerate(roots):
        p = ph + b.phase[i]
        foot_x = rx + stride * 0.5 * math.cos(p)
        foot_y = -max(0.0, math.sin(p)) * b.lift - (1.0 - scale) * H * 0.7
        # Bones are sized from the reach this leg actually needs at full
        # stride, not from a flat fraction of the height. Fixed fractions left
        # the bear's legs fractionally too short, the IK clamped the target,
        # and a 140 kg animal walked around 2 px above the ground.
        need = math.hypot(ry - foot_y, stride * 0.5)
        bone = need * b.bend * 0.5 * scale
        joint, foot = _ik2((rx, ry), (foot_x, foot_y), bone, bone, sign)
        legs.append((xf(rx, ry), xf(*joint), xf(*foot)))

    w = b.leg_w
    # far legs -> body -> near legs: the same cheap depth cue the stickmen use
    for root, joint, foot in (legs[1], legs[3]):
        _line(surf, far, root, joint, w)
        _line(surf, far, joint, foot, max(1, w - 1))

    _poly(surf, fur, body)
    # Shoulder and haunch masses. Two discs, and suddenly the legs grow out of
    # something instead of being pinned to a slab. They sit low enough not to
    # break the topline - a disc poking above the back reads as a lump, not as
    # muscle.
    _disc(surf, fur, xf(L * 0.27, -(H * 0.70 + hump * 0.30)), H * b.haunch)
    _disc(surf, fur, xf(-L * 0.30, -(H * b.rump * 0.70)), H * b.haunch * 1.05)
    # Rim along the spine, shade along the belly. One lit edge and one dark one
    # is the whole trick that stops a flat fill reading as a paper cutout - and
    # in a scene lit by a single candle the rim is what separates a bear from
    # the hillside behind it.
    prev = body[8]
    for pt in (body[0], body[1], body[2], body[3]):
        _line(surf, rim, prev, pt, 1)
        prev = pt
    shade = _dim(fur, 0.62)
    prev = body[4]
    for pt in (body[5], body[6], body[7]):
        _line(surf, shade, prev, pt, 1)
        prev = pt

    for root, joint, foot in (legs[0], legs[2]):
        _line(surf, fur, root, joint, w)
        _line(surf, fur, joint, foot, max(1, w - 1))

    # -- tail -------------------------------------------------------------
    # A fleeing animal tucks its tail. It is one number and it changes the
    # whole read of the silhouette.
    if b.tail_len > 0.5:
        ta = (-0.95 if fleeing else b.tail_ang) + math.sin(t * 3.1 + idp) * 0.14
        base = (-L * 0.45, -(H * b.rump * 0.90))
        tip = (base[0] - b.tail_len * math.cos(ta),
               base[1] - b.tail_len * math.sin(ta))
        mid = ((base[0] + tip[0]) * 0.5, (base[1] + tip[1]) * 0.5 + b.tail_len * 0.16)
        _line(surf, fur, xf(*base), xf(*mid), w)
        _line(surf, fur, xf(*mid), xf(*tip), max(1, w - 1))

    # -- neck and head ----------------------------------------------------
    # The neck is a tapered wedge, not a line: at 13 px tall a one-pixel neck
    # makes the head look like a balloon on a string.
    na = b.neck_ang + (0.30 if charging else 0.0) - (0.16 if fleeing else 0.0)
    neck_root = (L * b.neck_x, -(H * b.neck_y + hump * 0.30))
    head_c = (neck_root[0] + math.cos(na) * b.neck_len,
              neck_root[1] + math.sin(na) * b.neck_len)
    nperp = (-math.sin(na), math.cos(na))          # points *below* the neck
    nr = b.head_r * 1.05
    _poly(surf, fur, [
        xf(neck_root[0] + nperp[0] * nr, neck_root[1] + nperp[1] * nr),
        xf(head_c[0] + nperp[0] * b.head_r * 0.72, head_c[1] + nperp[1] * b.head_r * 0.72),
        xf(head_c[0] - nperp[0] * b.head_r * 0.72, head_c[1] - nperp[1] * b.head_r * 0.72),
        xf(neck_root[0] - nperp[0] * nr, neck_root[1] - nperp[1] * nr),
    ])
    _disc(surf, fur, xf(*head_c), b.head_r)

    # Muzzle: a wedge, so the wolf's long snout and the bear's blunt one are
    # the same three points with different numbers.
    ma = na + 0.16
    nose = (head_c[0] + math.cos(ma) * b.snout, head_c[1] + math.sin(ma) * b.snout)
    perp = (-math.sin(ma), math.cos(ma))
    _poly(surf, fur, [
        xf(head_c[0] + perp[0] * b.head_r * 0.62, head_c[1] + perp[1] * b.head_r * 0.62),
        xf(*nose),
        xf(head_c[0] - perp[0] * b.head_r * 0.72, head_c[1] - perp[1] * b.head_r * 0.72),
    ])

    # Carry the rim over the neck, skull and muzzle. Without it the head is the
    # same flat tone as the shoulders it sits against and the whole front end
    # merges into one brick - which is exactly what a bear looked like before
    # this line existed.
    _line(surf, rim, body[0],
          xf(neck_root[0] - nperp[0] * nr * 0.9, neck_root[1] - nperp[1] * nr * 0.9), 1)
    _line(surf, rim,
          xf(neck_root[0] - nperp[0] * nr * 0.9, neck_root[1] - nperp[1] * nr * 0.9),
          xf(head_c[0] - nperp[0] * b.head_r * 0.80, head_c[1] - nperp[1] * b.head_r * 0.80), 1)
    _line(surf, rim,
          xf(head_c[0] - nperp[0] * b.head_r * 0.80, head_c[1] - nperp[1] * b.head_r * 0.80),
          xf(*nose), 1)

    # One dark pixel for the eye. It costs nothing and it is the difference
    # between a head and a lump on the end of a neck. Drawn as a rect, not a
    # disc: aacircle rounds a sub-pixel radius up to a 3x3 plus sign, which on
    # a 5 px skull looks like a bullet hole.
    if fur is not _SIL_COLOR:
        eye = xf(head_c[0] + math.cos(ma) * b.head_r * 0.42 - nperp[0] * b.head_r * 0.34,
                 head_c[1] + math.sin(ma) * b.head_r * 0.42 - nperp[1] * b.head_r * 0.34)
        er = 2 if b.head_r >= 3.2 else 1
        pygame.draw.rect(surf, _dim(fur, 0.40),
                         pygame.Rect(int(eye[0]), int(eye[1]), er, er))

    if b.ear > 0.2:
        # Ears go back when running - the same trick as the tail.
        eb = (head_c[0] - b.head_r * 0.55, head_c[1] - b.head_r * 0.55)
        et = (eb[0] - (b.ear * 0.75 if fleeing else b.ear * 0.20),
              eb[1] - (b.ear * 0.45 if fleeing else b.ear))
        if b.ear_round:
            _disc(surf, fur, xf(eb[0] - b.ear * 0.2, eb[1] - b.ear * 0.45), b.ear * 0.55)
        else:
            _line(surf, fur, xf(*eb), xf(*et), max(1, w))

    if b.tusks:
        tusk = fur if fur is _SIL_COLOR else _TUSK
        for s, ln in ((0.50, 2.2), (-0.30, 1.6)):
            root = (nose[0] - math.cos(ma) * 1.1 + perp[0] * b.head_r * s,
                    nose[1] - math.sin(ma) * 1.1 + perp[1] * b.head_r * s)
            tip = (root[0] + math.cos(ma - 1.25) * ln,
                   root[1] + math.sin(ma - 1.25) * ln)
            _line(surf, tusk, xf(*root), xf(*tip), 1)


def _draw_animal_dead(surf: pygame.Surface, x: float, y: float, b: _Build,
                      facing: float, fur: Color, far: Color) -> None:
    """A carcass: the same torso laid flat, legs folded under it."""
    L = b.length
    h = b.height * 0.30

    def xf(lx: float, ly: float) -> tuple[float, float]:
        return (x + lx * facing, y + ly)

    body = [
        xf(L * 0.46, -h * 1.35), xf(L * 0.10, -h * 1.5),
        xf(-L * 0.46, -h * 1.2), xf(-L * 0.44, -h * 0.1),
        xf(L * 0.42, -h * 0.1),
    ]
    _poly(surf, fur, body)
    for sx in (L * 0.30, -L * 0.30):
        _line(surf, far, xf(sx, -h * 1.1), xf(sx + b.height * 0.35 * facing, -h * 0.1), b.leg_w)
    _disc(surf, fur, xf(L * 0.54, -h * 0.55), b.head_r * 0.95)


def _health_pip(surf: pygame.Surface, cx: float, cy: float, frac: float,
                bright: float = 1.0) -> None:
    """A 14x3 wound pip. Only ever drawn for something already hurt, so its
    presence alone is the signal - you notice it before you read it."""
    frac = fx.clamp(frac, 0.0, 1.0)
    w, h = 14, 3
    x0, y0 = int(cx - w * 0.5), int(cy)
    pygame.draw.rect(surf, _dim(_PIP_BACK, bright), pygame.Rect(x0 - 1, y0 - 1, w + 2, h + 2))
    pygame.draw.rect(surf, _dim(_PIP_EDGE, bright), pygame.Rect(x0, y0, w, h))
    fill = int(round(w * frac))
    if fill > 0:
        pygame.draw.rect(surf, _dim(_PIP_FILL, bright), pygame.Rect(x0, y0, fill, h))


# --------------------------------------------------------------------------
# the saucer
# --------------------------------------------------------------------------

_HULL_DARK: Color = (30, 33, 42)
_HULL: Color = (56, 61, 76)
_HULL_LIGHT: Color = (96, 104, 124)
_DOME: Color = (44, 52, 70)
_DOME_GLINT: Color = (150, 176, 208)
_LAMPS: tuple[Color, ...] = ((120, 236, 255), (255, 196, 96), (208, 140, 255), (140, 255, 176))
_BEAM_COLOR: Color = (108, 206, 255)

_SAUCER_CACHE: dict[tuple[int, int], pygame.Surface] = {}
_BEAM_CACHE: dict[tuple[int, int, int], pygame.Surface] = {}
_CACHE_MAX = 64


def _saucer_body(width: int, tilt_bucket: int) -> pygame.Surface:
    """Baked saucer, keyed by width and a quantised tilt.

    The hull never changes shape, only its lean as it wobbles, so a handful of
    pre-rotated copies covers every frame the thing is ever on screen.
    """
    key = (int(width), int(tilt_bucket))
    hit = _SAUCER_CACHE.get(key)
    if hit is not None:
        return hit

    w = max(16, int(width))
    hull_h = max(5, int(w * 0.26))
    dome_w = int(w * 0.46)
    dome_h = max(4, int(w * 0.24))
    pad = 6
    sw, sh = w + pad * 2, hull_h + dome_h + pad * 2
    surf = pygame.Surface((sw, sh), pygame.SRCALPHA)

    cx = sw * 0.5
    hull_top = pad + dome_h

    # Dome first, then the hull ellipse over its lower half: two ellipses and
    # no clipping maths.
    pygame.draw.ellipse(surf, _DOME,
                        pygame.Rect(int(cx - dome_w * 0.5), pad, dome_w, dome_h * 2))
    pygame.draw.ellipse(surf, _HULL, pygame.Rect(pad, hull_top, w, hull_h))
    # Underside: a slightly smaller, darker ellipse sunk into the hull, which
    # is what gives the disc a lip instead of reading as a flat pill.
    pygame.draw.ellipse(surf, _HULL_DARK,
                        pygame.Rect(pad + int(w * 0.10), hull_top + int(hull_h * 0.34),
                                    int(w * 0.80), max(3, int(hull_h * 0.72))))
    # Cool rim highlight along the top edge of the hull and one glint on the
    # dome. Same trick as atlas.py: a shape in the dark needs one lit edge.
    try:
        pygame.draw.arc(surf, _HULL_LIGHT, pygame.Rect(pad, hull_top, w, hull_h),
                        0.15, math.pi - 0.15, 2)
        pygame.draw.arc(surf, _DOME_GLINT,
                        pygame.Rect(int(cx - dome_w * 0.34), pad + int(dome_h * 0.22),
                                    int(dome_w * 0.5), dome_h),
                        0.6, 2.0, 1)
    except Exception:                                  # pragma: no cover
        pass

    tilt = tilt_bucket * 1.5
    if tilt:
        try:
            surf = pygame.transform.rotate(surf, tilt)
        except Exception:                              # pragma: no cover
            pass

    if len(_SAUCER_CACHE) > _CACHE_MAX:
        _SAUCER_CACHE.clear()
    _SAUCER_CACHE[key] = surf
    return surf


def _beam_surface(w: int, h: int, intensity: float) -> pygame.Surface | None:
    """Cached additive cone.

    Built with numpy in one vectorised pass and returned as an *opaque* surface
    that is black outside the cone, so ``BLEND_RGB_ADD`` adds nothing where the
    beam is not. Bucketed hard: a 4 px change in either dimension is invisible
    at this scale and a cache miss costs a megapixel of numpy.
    """
    wb = max(8, int(round(w / 4.0)) * 4)
    hb = max(8, int(round(h / 8.0)) * 8)
    ib = int(fx.clamp(intensity, 0.0, 1.0) * 6.0 + 0.5)
    if ib <= 0:
        return None
    key = (wb, hb, ib)
    hit = _BEAM_CACHE.get(key)
    if hit is not None:
        return hit

    inten = ib / 6.0
    xs = (np.arange(wb, dtype=np.float32) - (wb - 1) * 0.5)[:, None]
    v = (np.arange(hb, dtype=np.float32) / float(hb - 1))[None, :]
    # Cone: narrow at the saucer, wide at the ground.
    half = (wb * 0.5) * (0.30 + 0.70 * v)
    d = np.abs(xs) / np.maximum(half, 1.0)
    core = np.clip(1.0 - d * d, 0.0, 1.0)
    # Bright at the mouth, thinning toward the ground, plus a soft fade-in over
    # the first few rows so the cone does not start with a hard bright line.
    body = 0.42 * core + 0.58 * core ** 3
    body *= (1.0 - 0.42 * v)
    body *= np.clip(v * 6.0, 0.0, 1.0)
    grad = (body * inten).astype(np.float32)
    rgb = grad[:, :, None] * np.asarray(_BEAM_COLOR, dtype=np.float32)[None, None, :]
    try:
        surf = pygame.surfarray.make_surface(
            np.ascontiguousarray(np.clip(rgb, 0.0, 255.0), dtype=np.uint8))
    except Exception:                                  # pragma: no cover
        log.debug("beam surface build failed", exc_info=True)
        return None

    if len(_BEAM_CACHE) > _CACHE_MAX:
        _BEAM_CACHE.clear()
    _BEAM_CACHE[key] = surf
    return surf


def _find_ufo(world: Any) -> Any:
    """The saucer object, wherever it lives. ``None`` when nothing is up there."""
    events = getattr(world, "events", None)
    for holder, name in ((world, "ufo"), (events, "ufo"),
                         (world, "saucer"), (events, "saucer")):
        if holder is None:
            continue
        try:
            val = getattr(holder, name, None)
        except Exception:
            continue
        if val:
            return val
    for holder, name in ((world, "ufos"), (events, "ufos")):
        if holder is None:
            continue
        try:
            seq = list(getattr(holder, name, ()) or ())
        except Exception:
            continue
        for u in seq:
            if u is not None:
                return u
    return None


def _abductee(world: Any, ufo: Any) -> Any:
    """The agent the beam has hold of, by explicit id if the saucer names one
    and by the ``taken`` flag otherwise."""
    for name in ("target_id", "victim_id", "agent_id", "target", "victim"):
        try:
            val = getattr(ufo, name, None)
        except Exception:
            continue
        if val is None:
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            try:
                got = world.agent_by_id(int(val))
            except Exception:
                got = None
            if got is not None:
                return got
        elif hasattr(val, "x"):
            return val
    try:
        agents = list(world.population.agents)
    except Exception:
        return None
    for a in agents:
        if getattr(a, "taken", False):
            return a
    return None


def draw_ufo(surf: pygame.Surface, world: Any, t: float, *,
             cam: Camera) -> None:
    """Draw the saucer, its beam and whoever is inside the beam.

    Call this *after* the light composite - the beam is its own light source
    and multiplying it by a night ambient would erase it. Fails soft.

    *cam* is REQUIRED and has no default; the saucer's ``x`` is a world x.
    """
    try:
        ufo = _find_ufo(world)
        if ufo is None:
            return
        _draw_ufo(surf, ufo, world, float(t), cam)
    except Exception:
        log.debug("ufo draw failed", exc_info=True)


def _draw_ufo(surf: pygame.Surface, ufo: Any, world: Any, t: float,
              cam: Camera) -> None:
    if getattr(ufo, "alive", True) is False or getattr(ufo, "active", True) is False:
        return
    state = _state_of(ufo)
    if state in ("gone", "done", "idle", "off", "none", "waiting"):
        return

    wx = fx.attr_num(ufo, "x", default=math.nan)
    if not math.isfinite(wx):
        return
    # World x: ground_y indexes the WORLD_W-long heightmap.
    ground = _ground_y(world, wx)
    y = fx.attr_num(ufo, "y", default=math.nan)
    if not math.isfinite(y):
        y = ground - 190.0

    width = fx.attr_num(ufo, "width", "w", default=0.0)
    if width <= 8.0:
        r = fx.attr_num(ufo, "radius", "r", "size", default=0.0)
        width = r * 2.0 if r > 4.0 else 54.0
    width = fx.clamp(width, 24.0, 160.0)
    if not cam.visible(wx, width * 2.0):
        return

    # Wobble. Two incommensurate sines so it never visibly repeats, small
    # enough to read as "hovering" rather than "bouncing".
    idp = fx.attr_num(ufo, "id", default=0.0) * 1.7
    wob_y = math.sin(t * 1.9 + idp) * 1.5 + math.sin(t * 0.71 + idp * 0.3) * 0.9
    wob_x = math.sin(t * 0.53 + idp) * 1.8
    # The conversion: cx is screen from here down. Only _draw_beam's abductee
    # needs the world back, and it takes the camera for exactly that.
    cx, cy = cam.sx(wx) + wob_x, y + wob_y
    tilt_bucket = int(round(math.sin(t * 0.83 + idp) * 2.0))

    beam_amt = _beam_strength(ufo, world, state)
    if beam_amt > 0.01:
        _draw_beam(surf, world, ufo, cx, cy, width, ground, beam_amt, t, cam)

    body = _saucer_body(int(width), tilt_bucket)
    rect = body.get_rect()
    rect.center = (int(cx), int(cy))
    surf.blit(body, rect)

    _draw_rim_lights(surf, cx, cy, width, t, idp)


def _beam_strength(ufo: Any, world: Any, state: str) -> float:
    """0..1 beam brightness. Ramps in over the first 0.5 s of the beam so it
    switches on rather than appearing."""
    if "beam" in state or "abduct" in state or "lift" in state:
        on = True
    else:
        on = bool(getattr(ufo, "beaming", False)) or bool(getattr(ufo, "beam_on", False))
    bt = fx.attr_num(ufo, "beam_t", "beam_time", default=-1.0)
    if bt < 0.0 and on:
        # sim/ufo.py has no beam_t: it keeps `t`, seconds spent in the current
        # phase, which during the beam phase is the same clock.
        bt = fx.attr_num(ufo, "t", default=-1.0)
    if not on and bt > 0.0:
        on = True
    if not on:
        # Last resort: somebody is being lifted, so there is a beam whatever
        # the saucer's state string happens to be called.
        try:
            on = any(getattr(a, "taken", False) and fx.attr_num(a, "lift_t") > 0.0
                     for a in world.population.agents)
        except Exception:
            on = False
    if not on:
        return 0.0
    if bt < 0.0:
        return 1.0
    ramp = fx.smoothstep(bt, 0.0, 0.5)
    tail = 1.0 - fx.smoothstep(bt, UFO_BEAM_SEC + 0.4, UFO_BEAM_SEC + 1.2)
    return fx.clamp(ramp * tail, 0.0, 1.0)


def _draw_beam(surf: pygame.Surface, world: Any, ufo: Any, cx: float, cy: float,
               width: float, ground: float, amt: float, t: float,
               cam: Camera) -> None:
    # *cx* arrives in SCREEN space; *cam* is here only so _draw_abductee can
    # convert the abductee's own world x, which it has to read off the agent.
    # Clamp the mouth to just off the top of the screen. A saucer that has not
    # finished descending can sit hundreds of pixels above the world, and the
    # cone is baked with numpy - an unclamped height would allocate a surface
    # taller than the screen to draw the fraction of it that is visible.
    top = max(cy + width * 0.10, -60.0)
    height = min(ground - top, RENDER_H + 60.0)
    if height < 8.0:
        return
    bw = int(width * 1.30)
    cone = _beam_surface(bw, int(height), 0.55 + 0.45 * amt)
    if cone is None:
        return
    cw, ch = cone.get_size()
    surf.blit(cone, (int(cx - cw * 0.5), int(top)), special_flags=pygame.BLEND_RGB_ADD)

    # Scan rings sliding down the cone. Drawn per frame (they move, so they
    # cannot be baked) but they are three AA lines, not a surface. Their width
    # tracks the *baked* cone, not the requested one, or they poke out of the
    # sides wherever the size bucket rounded.
    for k in range(3):
        v = ((t * 0.42 + k / 3.0) % 1.0)
        ry = top + height * v
        half = (cw * 0.5) * (0.30 + 0.70 * v) * 0.72
        fade = (1.0 - v * 0.7) * amt
        col = _dim((214, 250, 255), 0.30 + 0.70 * fade)
        # An ellipse, not a bar: the beam is a cone seen from the side, so the
        # ring that slides down it has to have a near side and a far side.
        rh = max(2, int(half * 0.34))
        try:
            pygame.draw.ellipse(surf, col,
                                pygame.Rect(int(cx - half), int(ry - rh * 0.5),
                                            int(half * 2.0), rh), 1)
        except Exception:                              # pragma: no cover
            _line(surf, col, (cx - half, ry), (cx + half, ry), 1)

    # Pool of light where the beam lands, and a soft halo at the mouth.
    fx.blit_light(surf, cx, ground - 2.0, width * 0.85,
                  _BEAM_COLOR, 0.42 * amt, falloff=2.2)
    fx.blit_light(surf, cx, top + 2.0, width * 0.42,
                  (170, 235, 255), 0.55 * amt, falloff=1.4)

    _draw_abductee(surf, world, ufo, cx, top, ground, t, cam)


def _draw_abductee(surf: pygame.Surface, world: Any, ufo: Any,
                   cx: float, top: float, ground: float, t: float,
                   cam: Camera) -> None:
    """Draw whoever the beam has, at the height ``lift_t`` implies.

    ``lift_t`` is *read* here and nowhere written - the sim owns it. If it is
    absent or zero the figure simply stands at the bottom of the beam.
    """
    a = _abductee(world, ufo)
    if a is None:
        return
    lift = fx.attr_num(a, "lift_t", default=0.0)
    p = fx.clamp(lift / max(0.1, UFO_BEAM_SEC), 0.0, 1.0)
    p = fx.smoothstep(p, 0.0, 1.0)

    # World x for the heightmap sample, screen x for the drawing. `cx` is
    # already screen, so mixing the two in the drift below would slide the
    # figure by cam.x as it rose.
    wax = fx.attr_num(a, "x", default=cam.wx(cx))
    gy = _ground_y(world, wax)
    if not math.isfinite(gy):
        gy = ground
    ax = cam.sx(wax)
    # Drift toward the beam's axis as they rise, so they end up centred under
    # the saucer rather than clipping through its rim.
    px = ax + (cx - ax) * p
    py = gy + (top + 4.0 - gy) * p

    col = getattr(a, "color", (206, 214, 226))
    try:
        base: Color = (int(col[0]), int(col[1]), int(col[2]))
    except Exception:
        base = (206, 214, 226)
    # They are standing in a spotlight: wash the identity colour toward white
    # so it still reads as *them* while obviously being lit from above.
    base = _lit(base, 0.30 + 0.35 * p)

    # They hang further over as they rise: upright at the ground, most of the
    # way onto their side by the time the hull swallows them.
    ang = math.sin(t * 1.15) * 0.50 * p + p * 0.95
    _limp_figure(surf, px, py, 24.0, base, ang)


def _limp_figure(surf: pygame.Surface, cx: float, cy: float, h: float,
                 col: Color, ang: float) -> None:
    """A stick figure hanging in the beam: limbs trailing, slowly turning."""
    ca, sa = math.cos(ang), math.sin(ang)

    def pt(lx: float, ly: float) -> tuple[float, float]:
        return (cx + lx * ca - ly * sa, cy + lx * sa + ly * ca)

    hip = pt(0.0, -h * 0.10)
    neck = pt(0.0, -h * 0.40)
    head = pt(0.0, -h * 0.52)
    _line(surf, col, hip, neck, 2)
    # Arms and legs trail below - they are being lifted, so everything hangs.
    for s in (-1.0, 1.0):
        _line(surf, col, neck, pt(s * h * 0.10, -h * 0.24), 1)
        _line(surf, col, pt(s * h * 0.10, -h * 0.24), pt(s * h * 0.16, -h * 0.06), 1)
        _line(surf, col, hip, pt(s * h * 0.07, h * 0.14), 1)
        _line(surf, col, pt(s * h * 0.07, h * 0.14), pt(s * h * 0.12, h * 0.32), 1)
    _disc(surf, col, head, h * 0.115)


def _draw_rim_lights(surf: pygame.Surface, cx: float, cy: float,
                     width: float, t: float, idp: float) -> None:
    """Seven lamps chasing round the lower rim."""
    n = 7
    rx = width * 0.42
    ry = width * 0.26 * 0.42
    for i in range(n):
        th = math.pi * (i + 0.5) / n
        lx = cx - math.cos(th) * rx
        ly = cy + math.sin(th) * ry + width * 0.055
        b = 0.28 + 0.72 * (0.5 + 0.5 * math.sin(t * 3.6 - i * 0.92 + idp))
        col = _LAMPS[i % len(_LAMPS)]
        # Never below ~1.5 px: aacircle turns a radius of 1 into a 3x3 plus
        # sign, and a row of plus signs reads as text, not as lamps.
        _disc(surf, _dim(col, 0.34 + 0.66 * b), (lx, ly), 1.5 + b * 0.8)
        if b > 0.55:
            fx.blit_light(surf, lx, ly, 7.0 + 5.0 * b, col, 0.45 * b, falloff=1.2)


# --------------------------------------------------------------------------
# worn gear
# --------------------------------------------------------------------------

_SHAFT: Color = (146, 110, 70)
_SHAFT_DARK: Color = (78, 56, 34)
_SPEAR_HEAD: Color = (162, 166, 176)
_LEATHER: Color = (92, 60, 38)
_LEATHER_RIM: Color = (138, 100, 62)

#: The five wire strings a ``Stickman.relic`` may hold, as a set for the hot
#: path. Read here rather than through ``items.relic_of`` on purpose - see
#: :func:`_relic_of`.
_RELIC_SET: frozenset[str] = frozenset(RELIC_KINDS)

#: ``render.items``, resolved on the first relic that is actually drawn.
#:
#: It cannot be a module-level import and never will be: ``items.py`` imports
#: this module's primitives (``_line``, ``_disc``, ``_poly``, ``_dim``, ``_lit``,
#: ``_SIL_COLOR``, ``_state_of``, ``_find_ufo``) at *its* import time and
#: deliberately so - a relic drawn with a different line helper than the spear in
#: the same fist antialiases differently and reads as a different object. An
#: import back the other way at module level makes the pair genuinely circular,
#: and whichever of the two the renderer reaches first hands the other a
#: half-built module. ``stickfigure.py`` holds ``creatures`` behind exactly this
#: idiom for the same reason, and records the cost it is avoiding: a relative
#: import in the hot path measured +1.9 us per figure against +0.8 us cached.
#:
#: Deferred to the first *relic* rather than the first frame because 62.5% of
#: colonies never kill a dragon: :func:`_relic_of` answers "no relic" with a
#: getattr and a set lookup and never touches this at all.
#:
#: Three states, not two: ``None`` is "not looked yet", a module is the answer,
#: and ``False`` is "the import failed, stop trying". Without the third the
#: retry happens once per relic wearer per frame forever, and an ImportError is
#: not cheap - a colony with a BFG would pay for a stack unwind sixty times a
#: second to be told the same thing.
_items: Any = None


def _relic_of(agent: Any) -> str:
    """The relic *agent* is wearing, or ``""``. Never raises.

    A deliberate local twin of ``items.relic_of``. This runs for every agent on
    every frame, and the common answer is "" - resolving a module through a
    cache to be told that is work the fast path should not do, and importing
    ``render.items`` (numpy beam bakes and all) into a process whose colony has
    never seen a dragon is work it should not do either.

    Duck-typed rather than typed against ``Stickman.relic`` for the reason
    ``items.relic_of`` records: ``stickfigure._Disarmed`` hands this function a
    three-field stand-in for the poses that put a weapon down, and an unknown
    string (a save from a future kind, a half-written field) must degrade to
    "no relic" rather than to a traceback in the figure draw.
    """
    try:
        val = getattr(agent, "relic", "")
    except Exception:
        return ""
    return val if isinstance(val, str) and val in _RELIC_SET else ""


def draw_gear(surf: pygame.Surface, agent: Any, sk: Any = None, t: float = 0.0,
              base: Sequence[int] | None = None, silhouette: bool = False) -> None:
    """Draw an agent's spear, armour and relic.

    Meant to be called from ``stickfigure._draw`` right after the limbs, where
    the skeleton and colours are already in hand::

        from .creatures import draw_gear
        draw_gear(surf, s, sk, t, base, silhouette)

    Every argument after *agent* is optional: given only an agent, the
    skeleton is solved here. Read-only, fails soft, and draws nothing at all
    for an unarmed, unarmoured, relicless agent - which is the common case, so
    that path is three ``getattr`` calls and a return.

    The relic branch is the fix for a feature that shipped fully drawn and
    entirely unreachable: ``render/items.py`` has had five worn drawings in it
    since the relics landed, and the string "relic" appeared nowhere in this
    file, so four of the five were invisible on their wearer and the fifth
    (dragonscale, which mirrors itself into ``agent.armour``) showed up as a
    generic leather jerkin.
    """
    try:
        weapon = str(getattr(agent, "weapon", "") or "")
        armour = fx.attr_num(agent, "armour", default=0.0)
        relic = _relic_of(agent)
        has_spear = weapon == WEAPON_SPEAR
        has_armour = armour >= ARMOUR_LEATHER * 0.5
        # The relic HAS to be in this test. A BFG carrier's gun replaces their
        # spear, so `weapon` is "bfg9000" and `has_spear` is False, and they
        # usually have no jerkin either - without the third term the one
        # villager holding the loudest object in the game returns right here
        # with nothing drawn.
        if not (has_spear or has_armour or relic):
            return
        if sk is None:
            from .stickfigure import build_skeleton      # local: avoids a cycle
            sk = build_skeleton(agent, float(t))
        if sk is None or not getattr(sk, "hands", None):
            return

        alive = bool(getattr(agent, "alive", True))
        fade = 1.0 if alive else 0.55
        if silhouette and base is not None:
            flat: Color | None = (int(base[0]), int(base[1]), int(base[2]))
        else:
            flat = None

        # Resolved inside its own guard, not inside the outer one. items.py
        # pulls in numpy for the beam bakes and reaches back into this module
        # for primitives, so it has strictly more ways to fail on import than
        # anything else here - and if it took the outer handler with it, a
        # broken relic module would strip the jerkin off every armoured
        # villager and the shaft out of every fist. A relic that cannot be
        # drawn must cost exactly the relic.
        art: Any = None
        if relic:
            global _items
            if _items is None:
                try:
                    from . import items as _i           # local: see _items above
                    _items = _i
                except Exception:
                    log.warning("render.items would not import; relics are off "
                                "for this session", exc_info=True)
                    _items = False                      # sentinel: do not retry
            art = _items or None

        # The dragonscale mirrors itself into `agent.armour` - that is what
        # makes it absorb damage with no edit to the combat path - so its
        # wearer passes the jerkin test too and would get leather drawn under
        # plate on one torso. Which relics own the chest is asked of the module
        # that draws them rather than decided here, so a sixth relic that also
        # covers the body needs no edit in this file.
        if has_armour and not (art is not None and art.covers_torso(relic)):
            _draw_armour(surf, sk, flat, fade)
        if has_spear:
            _draw_spear(surf, sk, flat, fade)
        if art is not None:
            # Last, and over the spear. Four of the five sit on the body
            # (throat, chest, hip, feet) where nothing else competes; the fifth
            # is a two-handed gun that has to occlude the arms holding it, the
            # way the spear occludes the fist.
            art.draw_worn(surf, agent, sk, float(t), flat, fade, kind=relic)
            # ...and then the head comes back, for the one relic big enough to
            # take it. `stickfigure` draws worn gear in front of the whole
            # figure, head included, and is right to: an overhead chop should
            # swing a spear past the face. That rule was written against a 2 px
            # shaft. The BFG is a ~10 px block held in both fists, and in the
            # poses that raise the hands to head height (chop, panic, dance) it
            # covered 91-97% of the head disc - the wielder read as headless.
            # Re-stamping the head in the colour it was already painted with is
            # the whole fix: the gun ends up behind the face and in front of
            # everything else, which is where a carried object belongs anyway.
            #
            # `base` and not a colour of our own. It is the exact value
            # stickfigure._draw used, corpse fade and silhouette verdict
            # included, so the disc lands pixel-identical to the one underneath
            # rather than as a slightly wrong head. Skipped when there is none
            # (a contact sheet calling draw_gear with only an agent), because a
            # guessed head colour is worse than the occlusion.
            if base is not None and art.hides_head(relic, sk):
                head = getattr(sk, "head", None)
                if head is not None:
                    _disc(surf, (int(base[0]), int(base[1]), int(base[2])),
                          head, float(getattr(sk, "head_r", 2.0)))
    except Exception:
        # No getattr on `agent` in here. The handler used to log the agent's
        # name, which meant a sim object whose attribute access raised got its
        # exception re-raised *out of the except block* - the one path in this
        # module that could still take down a frame.
        log.debug("draw_gear failed", exc_info=True)


def _draw_armour(surf: pygame.Surface, sk: Any, flat: Color | None, fade: float) -> None:
    """A leather jerkin over the torso.

    Drawn as a quad tapering from the shoulders to the hips rather than as a
    fatter line: a uniform band reads as a thick torso, a tapered one reads as
    something *worn*, and that is the whole job at 26 px tall.
    """
    hip = sk.hip
    neck = sk.neck
    dx, dy = neck[0] - hip[0], neck[1] - hip[1]
    d = math.hypot(dx, dy)
    if d < 1.0:
        return
    px, py = -dy / d, dx / d

    def at(u: float, half: float, s: float) -> tuple[float, float]:
        return (hip[0] + dx * u + px * half * s, hip[1] + dy * u + py * half * s)

    scale = max(0.55, float(getattr(sk, "height", 26.0)) / 26.0)
    w_hip, w_sh = 1.5 * scale, 2.2 * scale
    body = flat if flat is not None else _dim(_LEATHER, fade)
    rim = flat if flat is not None else _dim(_LEATHER_RIM, fade)
    _poly(surf, body, [
        at(0.08, w_hip, +1.0), at(0.86, w_sh, +1.0),
        at(0.86, w_sh, -1.0), at(0.08, w_hip, -1.0),
    ])
    # A collar tick at the shoulders: without it the jerkin is a brown blob
    # rather than a garment with an edge.
    if flat is None:
        _line(surf, rim, at(0.84, w_sh, +1.0), at(0.84, w_sh, -1.0), 1)


#: How much of the full shaft has to survive the ground clamp for the spear to
#: still read as a weapon rather than as a floating arrowhead.
#:
#: Not a taste knob: the knapped head is 0.16 body-heights long, so a shaft that
#: reads as "haft plus point" rather than as "point" needs at least twice that,
#: 0.32h, and the full shaft is 0.98h - which is 0.327 of it. Rounded to 0.34.
#: Below it the stub is shorter than two heads and there is nothing left to see.
# TODO(backlog): belongs in constants.py
_SPEAR_MIN_VISIBLE: float = 0.34

#: How far ABOVE the foot line the clamp actually stops, in px. A geometric
#: clamp puts the tip exactly on the ground line and then the STROKE straddles
#: it: the shaft is 2 px wide and the knapped head is 1.5 px off the axis either
#: way, so a near-horizontal shaft ending on the line still painted a 19 px
#: smear of dirt-coloured spear below it (ink-diff, ``flip``). This is half a
#: stroke plus the head's half-width, and it is why the measured floor is 0 and
#: not "a couple of pixels". At 26 px tall the shift is invisible.
# TODO(backlog): belongs in constants.py
_SPEAR_GROUND_PAD: float = 2.6


def spear_span(sk: Any) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Butt, tip and the surviving fraction of a held spear, CLAMPED AT THE DIRT.

    The shaft is aimed down the elbow->hand vector, which is what makes it obey
    every pose for free - it hangs like a staff when idle, swings overhead on a
    chop, points at whatever is being stabbed - but it is 0.98 body-heights long
    and the renderer draws agents at step 7 and terrain at step 4. So anything
    the shaft pushes below the ground paints *over* the dirt: a spear tip
    floating on the hillside in front of a standing man. Measured by ink-diff
    (an armed and an unarmed agent into the same surface, gear-only pixels below
    the lowest foot counted), the worst sample was 121 px on ``handstand`` and
    28 px on plain ``idle``.

    The ground line is the lowest foot, which is the only ground reference a
    skeleton carries - and the right one, because it is where this figure is
    standing rather than where the terrain is somewhere else.

    Both ends clamp, independently: the tip when it points down and the butt
    when the shaft is inverted and the *butt* is the low end. What is returned
    is the surviving length as a fraction of the full shaft, which is what makes
    :func:`spear_grounded` a derived test instead of a list of pose names that
    rots the moment somebody adds a pose that plants the hands.

    A fist at or below the ground line (a handstand, a corpse lying on its face)
    has no direction left in which the shaft stays above the dirt, so the
    fraction is 0.0 and the caller is expected to drop the weapon entirely.

    Never raises; a skeleton with no feet is treated as having no ground at all
    and comes back unclamped, which is the right answer for a contact sheet.
    """
    hx, hy = sk.front_hand
    ex, ey = (sk.elbows[0] if getattr(sk, "elbows", None) else (hx, hy + 1.0))
    dx, dy = hx - ex, hy - ey
    d = math.hypot(dx, dy)
    if d < 0.35:                       # arm folded flat: fall back to horizontal
        dx, dy, d = float(getattr(sk, "facing", 1)), 0.15, 1.0
    ux, uy = dx / d, dy / d

    h = float(getattr(sk, "height", 26.0))
    ln = h * 0.98
    fwd, back = ln * 0.72, ln * 0.26
    feet = getattr(sk, "feet", None)
    gy = (max(p[1] for p in feet) - _SPEAR_GROUND_PAD) if feet else float("inf")
    if hy >= gy:
        # The fist itself is in the dirt. Any length in either direction is
        # buried, so there is nothing to clamp to - report a zero-length span
        # at the hand and let the caller decide (it drops the spear). This test
        # carries the pad too, which is what keeps the 2 px GRIP - centred on
        # the fist, so no endpoint clamp can move it - out of the ground.
        return (hx, hy), (hx, hy), 0.0
    if uy > 1e-3:
        fwd = min(fwd, (gy - hy) / uy)
    elif uy < -1e-3:
        back = min(back, (gy - hy) / -uy)
    butt = (hx - ux * back, hy - uy * back)
    tip = (hx + ux * fwd, hy + uy * fwd)
    return butt, tip, (fwd + back) / ln if ln > 1e-6 else 0.0


def spear_grounded(sk: Any) -> bool:
    """True when a held spear would be too buried to be worth drawing.

    Replaces ``stickfigure._SPEAR_GROUNDED``, which was a static list of pose
    names and therefore already wrong: it named the five acrobatic and lying
    poses somebody thought of, and ``mourn`` - which kneels, and buries 29 px -
    was never on it. This asks the geometry instead, so a pose invented next
    year is covered on the frame it first draws.
    """
    try:
        return spear_span(sk)[2] < _SPEAR_MIN_VISIBLE
    except Exception:
        return False


def _draw_spear(surf: pygame.Surface, sk: Any, flat: Color | None, fade: float) -> None:
    """A shaft through the fist, aligned with the forearm and clamped at the dirt.

    Geometry is :func:`spear_span`; this only paints it. A clamped spear becomes
    a stub, which is the price of the other half of the read: the hermit's
    walking staff is a vertical line planted at his feet and rising past his
    head, and it is his silhouette marker. Flipping every idle spear butt-down
    (the other way to keep it out of the ground) would hand that exact
    silhouette to every armed colonist and destroy the tell.
    """
    butt, tip, frac = spear_span(sk)
    if frac < _SPEAR_MIN_VISIBLE:
        return
    hx, hy = sk.front_hand
    d = math.hypot(tip[0] - butt[0], tip[1] - butt[1])
    if d < 1e-6:
        return
    ux, uy = (tip[0] - butt[0]) / d, (tip[1] - butt[1]) / d
    h = float(getattr(sk, "height", 26.0))

    shaft = flat if flat is not None else _dim(_SHAFT, fade)
    head = flat if flat is not None else _dim(_SPEAR_HEAD, fade)
    grip = flat if flat is not None else _dim(_SHAFT_DARK, fade)

    _line(surf, shaft, butt, tip, 2 if h >= 22.0 else 1)
    # Knapped head: a small triangle rather than a blob, so the pointy end is
    # obvious even when the whole figure is 26 px tall.
    # The head is bounded by the span, not just by the body height: a shaft the
    # ground clamp has cut to a stub must not grow a head longer than the stub,
    # which would put the knapped point out behind the butt.
    px, py = -uy, ux
    hl = min(h * 0.16, d)
    base_pt = (tip[0] - ux * hl, tip[1] - uy * hl)
    _poly(surf, head, [
        (base_pt[0] + px * 1.5, base_pt[1] + py * 1.5),
        tip,
        (base_pt[0] - px * 1.5, base_pt[1] - py * 1.5),
    ])
    if flat is None:
        _line(surf, grip, (hx - ux * 1.6, hy - uy * 1.6), (hx + ux * 1.6, hy + uy * 1.6), 2)


# --------------------------------------------------------------------------
# mining dust
# --------------------------------------------------------------------------

_DUST: Color = (152, 148, 140)
_DUST_DARK: Color = (110, 106, 100)
_CHIP: Color = (182, 184, 190)


def _iter_mining_agents(world: Any) -> list[Any]:
    """Alive agents with the sim's soft ``mining`` flag set. Fails soft to []."""
    pop = getattr(world, "population", None)
    agents = getattr(pop, "agents", None) if pop is not None else None
    if agents is None:
        agents = getattr(world, "agents", None)
    if agents is None:
        return []
    out: list[Any] = []
    try:
        for a in agents:
            if a is None:
                continue
            if not bool(getattr(a, "alive", True)):
                continue
            if not bool(getattr(a, "mining", False)):
                continue
            out.append(a)
    except TypeError:
        return []
    return out


def draw_mining_dust(surf: pygame.Surface, world: Any, t: float, *,
                     cam: Camera) -> None:
    """Kick up grey dust and stone chips at the feet of every mining agent.

    Reads the sim's soft ``agent.mining`` flag and never writes it. Meant to be
    called from ``renderer.draw`` right after the agents layer and *before* the
    light composite, so the motes obey the same rule the figures do: a miner
    working under a torch throws lit dust, one in the dark throws dust the
    composite swallows to near-black. Cheap - a handful of AA discs per miner,
    no per-pixel work - and fails soft, per miner and overall.

    *cam* is REQUIRED and has no default; a miner's ``x`` is a world x.
    """
    try:
        miners = _iter_mining_agents(world)
    except Exception:
        log.debug("could not read the miner list", exc_info=True)
        return
    if not miners:
        return
    try:
        tt = float(t)
    except (TypeError, ValueError):
        return
    if not math.isfinite(tt):
        return
    for a in miners:
        try:
            _draw_one_miner_dust(surf, a, tt, cam)
        except Exception:
            log.debug("mining dust draw failed", exc_info=True)


def _draw_one_miner_dust(surf: pygame.Surface, a: Any, t: float,
                         cam: Camera) -> None:
    wx = fx.attr_num(a, "x", default=math.nan)
    y = fx.attr_num(a, "y", default=math.nan)
    if not (math.isfinite(wx) and math.isfinite(y)):
        return
    if not cam.visible(wx, 40.0):
        return
    x = cam.sx(wx)          # the one conversion; everything below is screen
    face = fx.attr_num(a, "facing", "vx", default=1.0)
    facing = -1.0 if face < 0.0 else 1.0
    idp = (fx.attr_num(a, "id", default=0.0) * 1.37) % TAU

    # The working point: a little in front of the feet, where the tool bites.
    wx = x + facing * 5.0
    wy = y - 1.0

    # A short-lived puff of grey motes rising and drifting from the strike.
    n = 5
    for i in range(n):
        life = (t * 1.7 + i * 0.41 + idp) % 1.0        # 0..1 loop per mote
        rise = life * 9.0
        drift = facing * (1.6 + i * 0.9) * (0.4 + life)
        mx = wx + drift + math.sin(t * 2.3 + i + idp) * 1.1
        my = wy - rise - (i % 2) * 1.4
        fade = 1.0 - life
        r = max(0.8, 2.2 - 1.3 * life)
        col = _DUST if i % 2 else _DUST_DARK
        _disc(surf, _dim(col, 0.30 + 0.55 * fade), (mx, my), r)

    # Chip sparks flicked out low and forward on the swing's contact beat.
    swing = 0.5 + 0.5 * math.sin(t * 6.4 + idp)        # ~one strike per cycle
    if swing > 0.74:
        for k in range(2):
            u = 0.5 + 0.5 * k
            cxp = wx + facing * (2.5 + 6.5 * u)
            cyp = wy - 1.0 - math.sin(u * math.pi) * 3.2
            _disc(surf, _CHIP, (cxp, cyp), 1.0)


# --------------------------------------------------------------------------
# cache management
# --------------------------------------------------------------------------


def clear_caches() -> None:
    """Drop the baked saucer and beam surfaces. Used by tests and by the
    display-reset path, where old Surfaces belong to a dead display."""
    _SAUCER_CACHE.clear()
    _BEAM_CACHE.clear()


def cache_stats() -> dict[str, int]:
    return {"saucer": len(_SAUCER_CACHE), "beam": len(_BEAM_CACHE)}


# --------------------------------------------------------------------------
# smoke test
# --------------------------------------------------------------------------

if __name__ == "__main__":                             # pragma: no cover
    import os
    import time as _time
    import types

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode((320, 200))

    from ..constants import WEAPON_SPEAR as _WS
    from ..sim.world import World
    from .renderer import Renderer
    from .stickfigure import build_skeleton as _build_sk
    from .stickfigure import to_screen as _to_screen

    world = World(seed=7)
    for _ in range(240):
        world.tick(1.0 / 30.0)

    # Everything below is sited relative to the COLONY, not at absolute
    # 300-980. The world is 6400 px wide and the camera is wherever the
    # villagers are, so literal coordinates put the whole cast several
    # thousand px off the left of the frame and this smoke test measured the
    # cost of drawing nothing at all.
    from ..sim.actions import colony_center

    HOME = float(colony_center(world))

    # Prefer the real sim objects; fall back to the duck-typed shapes this
    # module actually contracts on, so the smoke test still runs on a tree
    # where sim/animals.py or sim/ufo.py is absent or renamed.
    def _fake(kind, x, **kw):
        st = ANIMAL_STATS[kind]
        o = types.SimpleNamespace(id=hash(kind) & 255, kind=kind, x=x,
                                  y=world.terrain.ground_y(x), facing=1,
                                  anim_t=0.0, health=st[0], max_health=st[0],
                                  alive=True, state="hunt", vx=0.0)
        o.__dict__.update(kw)
        return o

    spec = [(ANIMAL_WOLF, HOME - 340.0, {}),
            (ANIMAL_WOLF, HOME - 280.0,
             {"state": "flee", "fleeing": True, "health": 21.0}),
            (ANIMAL_BEAR, HOME - 120.0, {"state": "attack", "health": 90.0}),
            (ANIMAL_BOAR, HOME + 60.0, {"state": "attack"}),
            (ANIMAL_BOAR, HOME + 140.0, {"health": 0.0})]
    try:
        from ..sim.animals import Animal, AnimalRegistry

        reg = AnimalRegistry(seed=4)
        for kind, ax, kw in spec:
            st = ANIMAL_STATS[kind]
            reg.add(Animal(kind=kind, x=ax, y=world.terrain.ground_y(ax),
                           health=float(kw.get("health", st[0])), max_health=st[0],
                           state=str(kw.get("state", "approach")),
                           fleeing=bool(kw.get("fleeing", False)),
                           facing=1, anim_t=0.0))
        world.animals = reg
        src = "sim.animals.AnimalRegistry"
    except Exception as exc:                          # pragma: no cover
        world.animals = [_fake(k, ax, **kw) for k, ax, kw in spec]
        src = f"stand-ins ({exc})"

    try:
        from ..sim.ufo import Ufo

        saucer = Ufo(seed=2)
        saucer.active = True
        saucer.phase = "beam"
        saucer.x = saucer.hold_x = HOME + 340.0
        saucer.y = world.terrain.ground_y(HOME + 340.0) - 210.0
        saucer.t = 1.6
        world.ufo = saucer
        src += " + sim.ufo.Ufo"
    except Exception as exc:                          # pragma: no cover
        world.ufo = types.SimpleNamespace(
            id=1, x=HOME + 340.0,
            y=world.terrain.ground_y(HOME + 340.0) - 210.0,
            state="beam", beam_t=1.6, radius=28.0)
        src += f" + ufo stand-in ({exc})"

    agents = world.population.alive_agents()
    for i, ag in enumerate(agents):
        if i == 0:
            ag.weapon = _WS
            ag.armour = ARMOUR_LEATHER
        elif i == 1:
            ag.weapon = _WS
        elif i == 2:
            ag.armour = ARMOUR_LEATHER
        elif i == 3:
            ag.mining = True            # exercise the mining-dust path
    if agents:
        victim = agents[-1]
        victim.x = HOME + 340.0
        victim.taken = True
        victim.lift_t = 2.2
        world.ufo.target_id = victim.id

    drawn = list(_iter_animals(world))
    print(f"animals from {src}: {len(drawn)} "
          f"({', '.join(animal_kind_of(a) for a in drawn)})")

    ren = Renderer()
    ren.camera.snap_to(HOME)
    frames = 300
    t_creatures = 0.0
    t0 = _time.perf_counter()
    for f in range(frames):
        wt = world.world_time + f / 60.0
        for a in drawn:
            a.anim_t = wt
        scene = ren.draw(world, 1.0 / 60.0)
        base = pygame.surfarray.array3d(scene).copy() if f == frames - 1 else None
        c0 = _time.perf_counter()
        # The renderer's own camera, so this second pass lands on top of the
        # one draw() just did rather than 4000 px to the left of it.
        cam = ren.camera
        draw_animals(scene, world, wt, cam=cam)
        for ag in world.population.agents:
            # draw_gear takes a solved skeleton and is coordinate-space
            # agnostic; the sk=None branch solves in WORLD space, which is only
            # right under the identity camera. Convert explicitly here so this
            # sheet does not quietly become the thing this lane exists to end.
            draw_gear(scene, ag, _to_screen(_build_sk(ag, wt), -cam.x), t=wt)
        draw_mining_dust(scene, world, wt, cam=cam)
        draw_ufo(scene, world, wt, cam=cam)
        t_creatures += _time.perf_counter() - c0
        if base is not None:
            _changed = int((base != pygame.surfarray.array3d(scene))
                           .any(axis=2).sum())
    total = _time.perf_counter() - t0
    print(f"second pass put {_changed} px on the frame "
          f"(0 would mean the cast is off camera)")
    assert _changed > 0, "nothing this module drew landed inside the frame"

    try:
        draw_animals(scene, world, 0.0)          # type: ignore[call-arg]
    except TypeError as exc:
        print("cam is required:", exc)
    else:
        raise SystemExit("draw_animals accepted a call with no camera")
    print("identity camera still available for sheets:", IDENTITY)

    out = os.path.join(os.environ.get("TEMP", "."), "creatures_smoke.png")
    pygame.image.save(scene, out)
    print(f"{frames} frames, full renderer {total * 1000 / frames:.2f} ms/frame, "
          f"creatures {t_creatures * 1000 / frames:.3f} ms/frame")
    print("caches:", cache_stats())
    print("saved", out)
