"""The five dragons, drawn.

One module, five silhouettes, one entry point. :func:`draw_dragons` walks
``world.dragons`` and draws each one; :func:`draw_one` draws a single sim
dragon; :func:`draw_dragon` draws a kind at a point for anyone who has no sim
object at all (the contact sheet, a debug tray).

The five
--------
=========== ============ ======================================= ===== ========
kind        anchor       what it is                              scale drawn px
=========== ============ ======================================= ===== ========
flyover     body centre  from-below plan view, crossing the sky   1.35  107x123
wyvern      wing root    heraldic raider, side body, splayed      1.00  178x96
quadruped   **feet**     the siege dragon, standing in the huts   1.80  268x183
serpent     skull base   eastern wingless serpent, body trails    1.00  174x108
skeletal    wing root    the bone-wyrm, flight-only armature      1.20  134x102
=========== ============ ======================================= ===== ========

"drawn px" is the union bounding box over a full animation cycle, measured off
the surface rather than read off the proportions - a villager is 26 px tall and
a hut is 92x64.

The anchors disagree because the *drawings* disagree, and normalising them to
one anchor would mean re-cutting five sets of proportions. :data:`ANCHOR` is
the record of which point ``(x, y)`` is for each kind, and :func:`draw_dragons`
is the only place that has to care: everything but the quadruped is placed at
``ground_y(x) - alt``, and the quadruped is placed at ``ground_y(x)`` because
its origin is already the ground under its feet.

:data:`SCALE` is load-bearing and was tuned against a contact sheet, not
guessed - and the first cut of it was wrong in an instructive way. The five
prototypes quote nominal figures that look wildly inconsistent ("66 px
wingspan" against "SHOULDER_H 34.0" against "172 px spine"), so the table
started out correcting for them hard. But those figures measure different
things; the *drawn* extents at scale 1.0 turn out to be within a factor of two
of each other and already cut against the 26 px villager. See :data:`SCALE`
for what is left, which is two real decisions and a nudge.

Layering
--------
Called *after* the light composite, next to ``creatures.draw_ufo``: a dragon at
altitude is above every light source, and multiplying it down into a night
ambient is what would make it a black smear on black. It therefore does its own
silhouette decision, exactly like ``creatures._draw_animal``: below
``_SIL_CUTOFF`` of *source* light the whole animal collapses to one flat
colour. Since ``lighting.light_at`` excludes ambient, anything more than a
torch-radius off the ground is flat by definition - up there the flat path is
the normal case, not the exception, which is what the flyover was designed
around.

**Read-only w.r.t. sim state.** Nothing here writes to a World or a Dragon.
Every field is read through :func:`fx.attr_num` or ``getattr`` with a fallback,
the way ``creatures.py`` reads animals, so a rename on the sim side degrades to
a default pose rather than a traceback. Nothing from ``sim/`` is imported -
same rule ``creatures.py`` keeps, and it is also why this module loads whether
or not ``sim/dragons.py`` exists yet.

Where the art came from
-----------------------
The five bodies below are ports of five standalone prototypes, moved in rather
than imported: the prototypes are scratch files outside the game tree and the
game cannot depend on them. The drawing code is *copied*, not rewritten - every
one of those files carries a comment trail of what was tried and why it failed
(the wyvern's humerus that must never swing through horizontal, the serpent's
head at ~3.3 body-thicknesses, the scalloped trailing edges that separate a
dragon from a bird), and rewriting for tidiness would have thrown that away
silently. What was normalised is only the seams:

* one primitive layer instead of five copies of ``_line``/``_disc``/``_poly``;
* one signature. Three prototypes took ``facing`` and two did not - the
  skeletal hard-coded ``facing = 1.0`` and the flyover always flew right - so
  mirroring was added to both;
* one fail-soft boundary, in :func:`draw_dragon`, instead of five;
* per-kind name prefixes (``_fo_`` ``_wy_`` ``_qd_`` ``_sp_`` ``_sk_``),
  because the five disagreed on what ``_BODY``, ``_HZ`` and ``_HIDE`` mean.

A pixel-equality harness compared this module against the prototypes over a
full beat at 1.0x and 0.5x, lit and flat: identical for four of the five. The
flyover differs on purpose - see below.

What changed in the flyover, and why
------------------------------------
The flyover was judged the weakest of the five: near-invisible against a dark
sky. That is not a taste call, it is arithmetic. The sim flies it at
alt 300-420 px, so on a 1000 px frame it sits around y=300-420, where the night
sky gradient is ``(13, 17, 32)`` - and the silhouette colour every unlit
creature is forced to is ``(14, 15, 22)``. Rec.601 luma 17.5 against 15.5: a
difference of **2 of 255**. It was not badly drawn, it was drawn in the sky's
own colour.

Four changes, because none of them is enough alone. They are one tuple,
:data:`_fo_TONE`, plus one entry in :data:`SCALE`:

1. **Bigger** - ``SCALE["flyover"] = 1.35``, and that number is not taste:
   1.35 is where ``_fo_draw``'s own line-weight test steps the bone lines from
   1 px to 2, i.e. the size at which this drawing stops being thin enough to
   lose. It grows the union bbox from 79x108 to 107x123.
2. **The mass goes darker than the flat it was handed** - ``_dim(flat, 0.35)``
   for bone, body, legs and tail. There is nowhere to go but down for the
   structure, and down is where the prototype's own palette already was.
3. **The membrane goes lighter than the sky** - ``_lit(flat, 0.10)``. This is
   the one that actually recovers the drawing. A wing skin thin enough to see
   sky through is *backlit*, and lifting it above the sky rather than sinking
   it below is what makes the scalloped trailing edge - the cue that separates
   a dragon from a bird - exist at all in flat mode. The result straddles the
   sky instead of matching it: dark structure inside a pale sheet.
4. **A rim** at ``_lit(flat, 0.20)`` on the wing leading edges, the tail's
   centreline and a U round the snout. One lit side and no more, the rule
   ``creatures.py`` states and the wyvern already follows.

Measured against that night sky: mean |luma - sky| over the drawn pixels goes
from **1.9 to 21.2**, peak from 2.6 to 47.5. Swept as a set - at
``_lit(flat, 0.06)`` the membrane is still subtle, at ``0.16`` it starts to
read as a pale kite rather than as a creature, and with the membrane *dimmed*
instead the leading-edge rim becomes the only thing visible and the animal
reads as an archer's bow.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Sequence

import pygame

from ..constants import RENDER_H, RENDER_W
from . import fx

log = logging.getLogger(__name__)

__all__ = ["draw_dragons", "draw_one", "draw_dragon", "dragon_kind_of",
           "KINDS", "ANCHOR", "SCALE"]

TAU = math.tau
Color = tuple[int, int, int]
Pt = tuple[float, float]

# Mirrors renderer.SILHOUETTE_CUTOFF / SILHOUETTE_COLOR, copied rather than
# imported for the same reason creatures.py copies them: renderer imports this
# package and a cycle at startup is not worth two scalars.
_SIL_CUTOFF = 0.30
_SIL_COLOR: Color = (14, 15, 22)

_DEFAULT_GROUND_Y = RENDER_H * 0.72

#: How far off the frame a dragon may be before it is skipped. Generous: the
#: serpent is ~156 px long and the quadruped's tail runs a body length behind
#: its anchor, so a tight cull would clip the back half off a creature whose
#: anchor has only just left the screen.
_CULL = 260.0

# pygame-ce takes a width on aaline; older pygame does not. Probed once at
# import so the per-frame path carries no version check - the same probe
# stickfigure.py and creatures.py both make, duplicated for the same reason
# (no render-package import cycle).
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
# primitives - one copy, shared by all five bodies below
# --------------------------------------------------------------------------


def _dim(col: Sequence[int], f: float) -> Color:
    return (int(col[0] * f), int(col[1] * f), int(col[2] * f))


def _lit(col: Sequence[int], f: float) -> Color:
    """Brighten toward white by *f*. ``creatures._lit``, verbatim."""
    return (
        int(min(255.0, col[0] + (255 - col[0]) * f)),
        int(min(255.0, col[1] + (255 - col[1]) * f)),
        int(min(255.0, col[2] + (255 - col[2]) * f)),
    )


def _line(surf: pygame.Surface, col: Sequence[int], a: Pt, b: Pt,
          w: int = 1) -> None:
    if _AA_WIDTH:
        pygame.draw.aaline(surf, col, a, b, w)
    else:                                             # pragma: no cover
        pygame.draw.aaline(surf, col, a, b)
        if w > 1:
            pygame.draw.line(surf, col, (int(a[0]), int(a[1])),
                             (int(b[0]), int(b[1])), w)


def _disc(surf: pygame.Surface, col: Sequence[int], c: Pt, r: float,
          width: int = 0) -> None:
    if r <= 0.0:
        return
    if _AACIRCLE is not None:
        _AACIRCLE(surf, col, c, max(1.0, r), width)
    else:                                             # pragma: no cover
        pygame.draw.circle(surf, col, (int(c[0]), int(c[1])),
                           max(1, int(r)), width)


def _poly(surf: pygame.Surface, col: Sequence[int], pts: Sequence[Pt]) -> None:
    if len(pts) >= 3:
        pygame.draw.polygon(
            surf, col, [(int(round(p[0])), int(round(p[1]))) for p in pts])


def _smooth(e: float) -> float:
    e = 0.0 if e < 0.0 else (1.0 if e > 1.0 else e)
    return e * e * (3.0 - 2.0 * e)


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------

KIND_FLYOVER = "flyover"
KIND_WYVERN = "wyvern"
KIND_QUADRUPED = "quadruped"
KIND_SERPENT = "serpent"
KIND_SKELETAL = "skeletal"

#: Draw order is not this order - see :func:`draw_dragons`, which sorts by y.
KINDS: tuple[str, ...] = (KIND_FLYOVER, KIND_WYVERN, KIND_QUADRUPED,
                          KIND_SERPENT, KIND_SKELETAL)

#: What ``(x, y)`` means for each kind. The quadruped is the only ground
#: anchor and the only one whose ``y`` is not offset by ``alt``.
ANCHOR: dict[str, str] = {
    KIND_FLYOVER: "body",           # (x, y) is the body centre
    KIND_WYVERN: "wing_root",       # the withers
    KIND_QUADRUPED: "feet",         # GROUND anchor - the only one
    KIND_SERPENT: "skull_base",     # the body trails BEHIND
    KIND_SKELETAL: "wing_root",
}

#: Prototype scale 1.0 is not automatically game scale. The nominal figures in
#: the five prototypes look wildly inconsistent - "66 px wingspan", "~124 px
#: span", "SHOULDER_H 34.0", "172 px spine" - and the first cut of this table
#: corrected for them hard. That was wrong: those figures measure different
#: things, and the *drawn* extents, swept over a full cycle at scale 1.0 and
#: measured off the surface, are already close to each other and already close
#: to the villager the prototypes were each cut against:
#:
#:     flyover 79x108   wyvern 179x96   quadruped 148x101
#:     serpent 174x108  skeletal 112x85       (a hut is 92x64, a villager 26)
#:
#: So four of the five sit at or near 1.0 and the table only carries the two
#: real decisions:
#:
#: * **flyover 1.35** - part of the invisibility fix (module docstring). 1.35
#:   is not a taste number: it is the threshold at which the prototype's own
#:   code steps the bone lines from 1 px to 2, i.e. the size at which that
#:   drawing stops being thin enough to lose.
#: * **quadruped 1.80** - the siege dragon, and the one kind whose role is a
#:   size. 34.0 x 1.8 puts its shoulder at 61 px, level with a hut's eaves at
#:   64, with the head and the mantled wings well over the roof. At 1.0 it
#:   stands lower than the roofline and reads as a large dog among the
#:   buildings, which is the one thing this dragon must not look like. 2.5,
#:   the top of the band the design asked for, makes it three huts long and it
#:   stops being *in* the colony at all.
#: * **skeletal 1.20** - a nudge, not a correction. It is the smallest of the
#:   five and it is the throw-only dragon, so it has to read as a target from
#:   the ground at alt 60-110.
SCALE: dict[str, float] = {
    KIND_FLYOVER: 1.35,
    KIND_WYVERN: 1.00,
    KIND_QUADRUPED: 1.80,
    KIND_SERPENT: 1.00,
    KIND_SKELETAL: 1.20,
}

# ==========================================================================
# the five bodies
#
# Ported from the prototypes verbatim; see the module docstring. Names are
# prefixed per kind because the five disagreed on what _BODY, _HZ and
# _HIDE mean, and a merge that silently picked one meaning would have
# been a bug you could only find by looking at the picture.
# ==========================================================================

# --------------------------------------------------------------------------
# flyover - THE PASSAGE. A from-below plan view: the wings do not swing up and
# down the screen, they foreshorten, and a viewer tilt _fo_PHI stops the two
# from being mirror images. Anchored on the body centre; no legs to land on.
# --------------------------------------------------------------------------
# Two flat tones, both essentially black. The membrane is a hair lighter than
# the bone because a stretched skin thin enough to see sky through is the one
# piece of information a pure-black blob loses.
#
# It was three times this far apart to begin with, and the flat-silhouette
# cell of the test strip was visibly the *stronger* drawing - against a night
# sky at 40-70 px a membrane you can see through is a membrane you cannot see.
# What survives is a difference small enough to be a texture rather than a
# tone, and the shape stays a hole in the sky either way.
_fo_BONE: tuple[int, int, int] = (7, 8, 12)
_fo_MEMBRANE: tuple[int, int, int] = (13, 15, 21)

#: How far off straight-overhead the viewer is, in radians. 0 gives a
#: symmetric jellyfish pulse; anything past ~0.5 starts to read as a side view.
_fo_PHI = 0.30

#: Wingbeats per second. Slow on purpose - a big thing high up.
_fo_BEAT_HZ = 0.55

# Amplitude is a compromise, and it is the number this whole thing lives or
# dies on. Too little and the wings are a rigid cross; too much and the span
# collapses to nothing twice a second and the animal reads as a shutter - at
# 0.85 it did exactly that. At 0.62 the projected span never drops below ~80%
# of full while the *asymmetry* between the two wings still swings hard.
_fo_TH_MID = 0.05        # wing elevation at the middle of the stroke
_fo_TH_AMP = 0.62        # ... and how far either side of it
_fo_WARP = 0.42          # phase warp: hangs at the top, snaps through the bottom

#: The three numbers that decide whether this thing can be seen at all:
#: ``(bone, membrane, rim)`` as factors applied to the flat colour the caller
#: handed us - ``_dim`` for the first, ``_lit`` for the other two. They are
#: here as one tuple rather than inline because they were swept as a set
#: against the real night sky, and they only make sense as a set: see the
#: module docstring for the arithmetic and :func:`_fo_draw` for the read.
_fo_TONE = (0.35, 0.10, 0.20)


# ---------------------------------------------------------------- planform --
# Everything below is in "body units": +x is the way it is flying, and the
# second number is *lateral distance from the spine*, always positive - the
# side is applied by the fold. One body unit is one pixel at scale 1.0.
#
# Lateral distances on the wing are fractions of the half-span S so the
# planform stays itself at any size; distances on the body are absolute,
# because a body that scaled with the span would swallow the wing roots.
_fo_S = 33.0                      # half-span at scale 1.0 -> 66 px wingspan

# The leading edge sweeps back the *whole* way out, not just over the last
# third. With a straight inner arm the wing stands out sideways from the body
# and the animal reads as an open umbrella on a stick; with the arm raked, the
# outline becomes a swept V and the same shape reads as flying.
_fo_SHOULDER = (5.6, 2.2)
_fo_ELBOW = (4.0, 0.30)
_fo_WRIST = (-0.4, 0.58)
_fo_THUMB = (2.8, 0.62)           # claw hooking forward off the leading edge
_fo_TIP = (-16.0, 1.00)
_fo_F2 = (-20.0, 0.72)
_fo_F3 = (-19.5, 0.44)
_fo_F4 = (-15.0, 0.20)
_fo_WING_HIP = (-7.5, 2.0)
#: How far each membrane panel is pulled in toward the wrist, outermost
#: first. The outer panel is the long one on a real wing, so it gets the
#: deepest scoop - equal scallops all the way in read as a paper doily.
_fo_SCOOP = (0.40, 0.34, 0.28)


def _fo_wing_outline(S: float, deepen: float = 1.0
                     ) -> tuple[list[tuple[float, float]],
                                list[tuple[float, float]]]:
    """(polygon, strut endpoints) for one wing, in unfolded (lx, d) space.

    The lateral figure is a fraction of the half-span everywhere out on the
    wing, but the two points where the wing meets the body - the shoulder and
    the trailing edge's anchor at the hip - are absolute, because they sit on
    a torso whose width does not grow with the span.
    """
    def sp(p: tuple[float, float]) -> tuple[float, float]:
        return (p[0], p[1] * S)

    sh, hip = _fo_SHOULDER, _fo_WING_HIP
    el, wr = sp(_fo_ELBOW), sp(_fo_WRIST)
    th, tp = sp(_fo_THUMB), sp(_fo_TIP)
    f2, f3, f4 = sp(_fo_F2), sp(_fo_F3), sp(_fo_F4)

    def scoop(a: tuple[float, float], b: tuple[float, float], k: float
              ) -> tuple[float, float]:
        """Pull the midpoint of a membrane panel back toward the wrist."""
        mx, my = (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5
        k = min(0.56, k * deepen)
        return (mx + (wr[0] - mx) * k, my + (wr[1] - my) * k)

    poly = [sh, el, wr, th, tp,
            scoop(tp, f2, _fo_SCOOP[0]), f2,
            scoop(f2, f3, _fo_SCOOP[1]), f3,
            scoop(f3, f4, _fo_SCOOP[2]), f4, hip]
    struts = [sh, el, wr, tp, f2, f3, f4]
    return poly, struts


# Body outline, one side, nose-ward first. Deliberately narrow: a spine that
# is the same weight as the wing root turns the whole animal into one lozenge,
# and then the wings have nothing to be attached *to*.
_fo_BODY_SIDE = [(10.5, 1.1), (6.5, 2.4), (1.0, 2.9), (-3.5, 2.5), (-7.0, 1.8)]
# The head has to be obviously heavier than the tail spade or the animal has
# two arrowheads and no direction of travel. Blunter, wider, and horned.
# Neck: three stations, so it can taper *and* bend. Two points made a rigid
# bar, and a rigid bar in front of a pair of swept horns reads as an arrow
# glyph rather than as an animal - which is exactly what it did.
_fo_NECK = [(6.5, 2.2), (12.6, 1.4), (18.4, 0.9)]
_fo_NECK_TIP = 18.4
# Head in its own frame, hinged at the neck tip: blunt snout, cheeks flaring
# wider than the snout, narrow again where it meets the neck. Widest at the
# back of the skull is what makes a shape read as a reptile head from above.
# The whole skull, one side, nose first: pointed snout, a hard cheek flare,
# then a horn raked back past the jaw, then in to the neck.
#
# Three goes at this. A blunt lozenge read as a club on a stick. Adding horns
# as two separate 1 px lines angled out at 25 degrees turned the front end
# into a literal "->" glyph, and pulling them in until they no longer did
# that left them hanging beside the neck like insect antennae. Cutting them
# into the skull outline is what finally worked: one filled shape, so at 1x
# they thicken the head instead of becoming two stray pixels, and the notch
# between horn and cheek survives all the way down to 0.5x.
_fo_HEAD = [(27.4, 1.0), (24.2, 1.7), (22.2, 3.6), (19.4, 3.0),
            (16.2, 4.5), (18.6, 2.2)]
_fo_LEG = ((-6.5, 1.6), (-10.8, 3.9), (-15.0, 2.6))
# The tail is longer than the neck and the body put together, which is the
# proportion that separates "dragon" from "bat" once the wings stop moving.
_fo_TAIL_FROM, _fo_TAIL_TO = -6.5, -42.0
_fo_TAIL_W0, _fo_TAIL_W1 = 2.1, 0.45
_fo_SPADE_LEN, _fo_SPADE_W = 8.0, 2.0    # a leaf, not a barbed arrowhead


def _fo_beat(t: float) -> float:
    """Wing elevation in radians. + is up, 0 is level with the shoulders."""
    p = t * TAU * _fo_BEAT_HZ
    # A sine beats symmetrically, which reads as flapping rather than as
    # flying. Warping the phase makes the wing hang at the top of the recovery
    # and snap through the power stroke.
    return _fo_TH_MID + _fo_TH_AMP * math.cos(p - _fo_WARP * math.sin(p))


def _fo_draw(surf: pygame.Surface, x: float, y: float, t: float, scale: float,
             flat: Sequence[int] | None, facing: int) -> None:
    if scale <= 0.0 or not (math.isfinite(x) and math.isfinite(y)):
        return
    S = _fo_S * scale

    # Mirroring, which the prototype did not have: it always flew right.
    #
    # It is mirrored in **x only**, and that is a decision rather than a free
    # transform. The camera model here is deliberately asymmetric - the near
    # wing projects as +d*cos(th - PHI) and the far one as -d*cos(th + PHI),
    # with a viewer tilt _fo_PHI of 0.30 - so negating x alone flips the
    # direction of travel while leaving the viewer where they were. Negating
    # the fold as well would move the *viewer* to the other side of the animal
    # every time it turned around, which is not a thing viewers do.
    mir = -1.0 if facing < 0 else 1.0

    # Colour, and this is the fix for the "near-invisible" verdict. The sim
    # flies this thing at alt 300-420, so on a 1000 px frame it sits where the
    # night sky gradient is about (14, 17, 32) - and _SIL_COLOR, which is what
    # every unlit creature is forced to, is (14, 15, 22). Luma 17.8 against
    # 15.5: a difference of 2.3 of 255. It was not drawn badly, it was drawn in
    # the sky's own colour.
    #
    # So the drawing straddles the sky instead of matching it: the structure
    # goes darker than the flat it was handed and the membrane goes *lighter*
    # than the sky, because a wing skin thin enough to see sky through is
    # backlit and that is the only way the scalloped trailing edge exists at
    # all in flat mode. Measured mean |luma - sky| over the drawn pixels: 1.9
    # before, 21.2 after. See the module docstring for the sweep.
    if flat is not None:
        bone = _dim(flat, _fo_TONE[0])
        memb = _lit(flat, _fo_TONE[1])
        rim = _lit(flat, _fo_TONE[2])
    else:
        bone, memb = _fo_BONE, _fo_MEMBRANE
        rim = _lit(_fo_MEMBRANE, _fo_TONE[2] + 0.04)
    w = 2 if scale >= 1.35 else 1

    # Detail floor. Below about 0.8x the three things that say "dragon" - the
    # tail, the skull and the scallops on the trailing edge - all fall under a
    # pixel at once and what is left is a dark smear with wings. Rather than
    # let that happen, a shrinking dragon gets a fatter tail, a bigger head and
    # deeper scoops, so the *features* hold their pixel size while the animal
    # gets smaller. Same bargain creatures.py makes when it refuses to draw a
    # rim light below 1.5 px.
    fat = min(1.85, max(1.0, 0.80 / scale))

    th0 = _fo_beat(t)
    # Finite difference rather than the analytic derivative: the phase warp
    # makes that ugly and this is two cosines a frame.
    dth = (_fo_beat(t + 0.02) - th0) / 0.02
    # The tip lags the root, so a wing driven downward is cupped upward. One
    # term, and the wing stops being a rigid plank.
    curl = max(-0.8, min(0.8, -0.115 * dth))
    ph = (t * TAU * _fo_BEAT_HZ) % TAU

    # The body sinks through the recovery and is pushed up on the power
    # stroke. Tiny - a couple of pixels - but a shape that only changes width
    # reads as a shutter opening and closing.
    bob = 0.9 * scale * math.cos(ph)
    sweep = 0.09 * S

    def fold(pt: tuple[float, float], s: float) -> tuple[float, float]:
        """(lx, d) in planform space -> world space, for side *s*."""
        lx, d = pt
        u = min(1.0, d / S) if S > 0.0 else 0.0
        th = th0 + curl * (u ** 1.5)
        fac = math.cos(th - _fo_PHI) if s > 0 else -math.cos(th + _fo_PHI)
        return (x + (lx - sweep * u * th) * scale * mir,
                y + d * fac * scale + bob)

    def at(pt: tuple[float, float], s: float = 1.0) -> tuple[float, float]:
        """Body-space point (no fold - the body is on the roll axis)."""
        return (x + pt[0] * scale * mir, y + pt[1] * s * scale + bob)

    # A slow head turn, on its own clock rather than the beat's - it is
    # looking at things down there. The neck bends into it and the skull
    # swings with the neck's tip, so the front end articulates instead of the
    # whole animal sliding sideways.
    swing = math.sin(t * 0.41)
    yaw = 2.4 * swing                                  # body units, at the tip
    ang = 0.16 * swing                                 # skull, radians
    ca, sa = math.cos(ang), math.sin(ang)

    def bend(lx: float) -> float:
        k = min(1.0, max(0.0, (lx - 6.5) / (_fo_NECK_TIP - 6.5)))
        return yaw * (k ** 1.6)

    def atn(pt: tuple[float, float], s: float = 1.0) -> tuple[float, float]:
        """A point on the neck."""
        return (x + pt[0] * scale * mir,
                y + (pt[1] * s * fat + bend(pt[0])) * scale + bob)

    def ath(pt: tuple[float, float], s: float = 1.0) -> tuple[float, float]:
        """A point on the skull, hinged at the neck's tip."""
        lx, ly = pt[0] - _fo_NECK_TIP, pt[1] * s * fat
        return (x + (_fo_NECK_TIP + lx * ca - ly * sa) * scale * mir,
                y + (yaw + lx * sa + ly * ca) * scale + bob)

    # -- wings ------------------------------------------------------------
    # Drawn before everything else so the spine, neck and tail lie on top of
    # the membrane rather than being cut in half by it. Which wing is "in
    # front" barely matters when both are the same one tone.
    poly, struts = _fo_wing_outline(S, 1.0 + 0.75 * (fat - 1.0))
    for s in (-1.0, 1.0):
        pts = [fold(p, s) for p in poly]
        _poly(surf, memb, pts)
        sh, el, wr, tp, _f2, f3, _f4 = [fold(p, s) for p in struts]
        # Arm bones along the leading edge, and *one* finger raking back off
        # the wrist. Drawing all four fingers turned the wing into a folding
        # fan - four radiating lines plus a sawtooth trailing edge is more
        # internal structure than a 30 px shape can carry, and at wallpaper
        # scale it read as texture noise rather than as anatomy. The scallops
        # on the outline already say "membrane"; the struts only have to say
        # "there is an arm in here".
        #
        # The arm chain is drawn in *rim* rather than in bone, and that is the
        # second half of the visibility fix. It is the leading edge, and the
        # leading edge is where sky light wraps a thing you are looking up at,
        # so one lifted swept-V per wing is both the physics and the read the
        # planform was built for. Both wings get it: in a from-below view
        # neither is nearer the sky than the other, so the wyvern's near/far
        # dimming would be a lie here.
        _line(surf, rim, sh, el, w)
        _line(surf, rim, el, wr, w)
        _line(surf, rim, wr, tp, w)
        _line(surf, bone, wr, f3, 1)

    # -- tail -------------------------------------------------------------
    # Longer than the body and whipping. This is the single strongest cue
    # that the thing overhead is not a bird.
    span = _fo_TAIL_FROM - _fo_TAIL_TO
    spine: list[tuple[float, float]] = []
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for i in range(11):
        k = i / 10.0
        lx = _fo_TAIL_FROM - span * k
        amp = 5.6 * (k ** 1.5)
        ly = amp * math.sin(k * 2.5 - ph)
        hw = (_fo_TAIL_W0 + (_fo_TAIL_W1 - _fo_TAIL_W0) * k) * fat
        spine.append((lx, ly))
        left.append((lx, ly - hw))
        right.append((lx, ly + hw))
    _poly(surf, bone, [at(p) for p in left] + [at(p) for p in reversed(right)])
    sw = _fo_SPADE_W * fat

    # Leaf-shaped spade on the tip, aimed down the tail's own direction. It
    # was a wide barbed arrowhead first, which made the tail end read as a
    # *pointer* - the heaviest, most graphic thing in the whole silhouette,
    # aimed the wrong way. Long and narrow, and it reads as a tail again.
    ax, ay = spine[-1]
    bx, by = spine[-2]
    dx, dy = ax - bx, ay - by
    d = math.hypot(dx, dy) or 1.0
    ux, uy = dx / d, dy / d
    px, py = -uy, ux
    _poly(surf, bone, [
        at((ax + ux * _fo_SPADE_LEN, ay + uy * _fo_SPADE_LEN)),
        at((ax + ux * _fo_SPADE_LEN * 0.46 + px * sw,
            ay + uy * _fo_SPADE_LEN * 0.46 + py * sw)),
        at((ax - ux * 2.5, ay - uy * 2.5)),
        at((ax + ux * _fo_SPADE_LEN * 0.46 - px * sw,
            ay + uy * _fo_SPADE_LEN * 0.46 - py * sw)),
    ])

    # A rim down the tail's own centreline, out to the spade's point. The tail
    # is the strongest "not a bird" cue in the drawing and it is also the
    # thinnest thing in it, so it is the first thing that dies when the mass
    # goes to one flat tone - a 2 px ribbon of near-black on near-black.
    # Lifting the centreline keeps the whip legible without touching its
    # outline, which is what carries the taper.
    rimline = [at(p) for p in spine]
    rimline.append(at((ax + ux * _fo_SPADE_LEN, ay + uy * _fo_SPADE_LEN)))
    for a, b in zip(rimline, rimline[1:]):
        _line(surf, rim, a, b, 1)

    # -- hind legs --------------------------------------------------------
    # Trailing and splayed, not tucked: a tucked leg is a bird's.
    for s in (-1.0, 1.0):
        a, b, c = (at(p, s) for p in _fo_LEG)
        _line(surf, bone, a, b, 1)
        _line(surf, bone, b, c, 1)

    # -- body, neck, head -------------------------------------------------
    _poly(surf, bone,
          [at(p, 1.0) for p in _fo_BODY_SIDE] +
          [at(p, -1.0) for p in reversed(_fo_BODY_SIDE)])
    _poly(surf, bone,
          [atn(p, 1.0) for p in _fo_NECK] +
          [atn(p, -1.0) for p in reversed(_fo_NECK)])
    _poly(surf, bone,
          [ath(p, 1.0) for p in _fo_HEAD] +
          [ath(p, -1.0) for p in reversed(_fo_HEAD)])
    # One dot at the shoulders keeps the wing roots from pinching off the
    # torso when the wings fold across the body.
    _disc(surf, bone, at((4.0, 0.0)), 3.0 * scale)

    # Last of the rim: a U round the front of the skull, from one cheek flare
    # over the snout to the other. The head is what says which end is the
    # front, and cutting the horns into the skull outline - which is what made
    # the head work in the first place - is exactly what makes it read as one
    # undifferentiated blob once the whole animal is a single tone. The three
    # forward points only; carrying the rim round the horns as well put a
    # bright arrowhead on the front and the "->" glyph the prototype spent
    # three passes killing came straight back.
    nose = ([ath(_fo_HEAD[i], -1.0) for i in (2, 1, 0)]
            + [ath(_fo_HEAD[i], 1.0) for i in (0, 1, 2)])
    for a, b in zip(nose, nose[1:]):
        _line(surf, rim, a, b, 1)


# --------------------------------------------------------------------------
# wyvern - THE RAIDER. Side-on body in the stickman projection, wings splayed
# pseudo-frontally the way heraldry does it. Anchored at the wing root. Every
# pose in here is a flight pose: it has two legs, but no standing.
# --------------------------------------------------------------------------
def _wy_pol(a: float, r: float) -> tuple[float, float]:
    """Unit-ish vector at *elevation* a (radians above horizontal), scaled."""
    return (math.cos(a) * r, -math.sin(a) * r)


# ------------------------------------------------------------------ palette
# Everything is darker than the sky it flies against. A creature between the
# viewer and the sky is backlit, and the game already commits to that read -
# ``stickfigure.SILHOUETTE`` is near-black. Values here sit low enough that
# the shape survives even before the flat mode takes over.
_wy_HIDE: Color = (38, 52, 40)          # dark scale green
_wy_HIDE_RIM: Color = (116, 144, 102)   # one lit edge along the spine
_wy_HIDE_SHADE: Color = (22, 30, 24)
# The membrane is lighter than the hide - it is a skin two cells thick with the
# sky behind it - and every colour here is starved of blue on purpose. The
# thing flies against a blue sky, so separating by hue as well as by value
# means the shape still holds at the one moment value alone would fail: dusk,
# when the sky comes down to meet it.
_wy_MEMB: Color = (60, 72, 46)          # wing membrane, near side
_wy_MEMB_RIB: Color = (34, 42, 28)      # finger bones, read as ribs *in* the sheet
_wy_MEMB_LEAD: Color = (104, 122, 74)   # lit leading edge - the near wing's only
_wy_HORN: Color = (100, 100, 84)        # horns and claws. Not white, barely even
_wy_EYE: Color = (250, 178, 52)         # pale - the eye is the one hot pixel here,
                                    # and it only works if nothing competes.

_wy_FAR = 0.55                         # the far wing and far leg are drawn this dark

# ---------------------------------------------------------------- proportions
# Pixels at scale 1.0, where a stickman is 26 px tall.
_wy_HUM = 21.0            # shoulder -> elbow
_wy_FORE = 24.0           # elbow -> wrist
_wy_HAND = 34.0           # wrist -> longest fingertip
#: (sweep back from the hand's base direction, length as a fraction of _HAND).
#: The inner fingers are long on purpose: they are what gives the wing a
#: *chord*. With a short fan the membrane is a sliver on a stick and the
#: creature reads as a heron the moment the wings pass through horizontal.
_wy_FINGERS = ((0.10, 1.00), (0.55, 0.94), (1.05, 0.84), (1.58, 0.70))
_wy_WING_ROOT = (0.0, 0.0)
#: Where each membrane closes back onto the body. Kept short and high on the
#: back rather than run down to the hip: a hip anchor buys chord and costs the
#: one thing that matters more, a notch of sky between the wing root and the
#: body. At the bottom of the stroke a hip-anchored membrane sweeps across the
#: torso and the whole animal fuses into one wide slab - a manta ray, not a
#: dragon. The chord comes from the finger fan instead.
#:
#: The *near* wing is the one splayed toward the face, and the near/far reach
#: difference is the 3/4 view: two exactly mirrored wings on a side-on body is
#: a moth. Which side is near is not arbitrary - see the layer order in
#: :func:`draw_dragon`.
_wy_TRAIL_NEAR = (3.0, 0.5)
_wy_TRAIL_FAR = (-6.0, 0.0)
_wy_REACH_FAR = 0.92

#: The arm never goes below the body. This is the whole animation decision and
#: it took three passes to arrive at: an honest flap swings the humerus through
#: horizontal, and for the several frames it spends there the wings, the spine,
#: the neck and the tail are all one horizontal line. The creature stops being
#: a dragon and becomes a smear. So the humerus stays in a shallow V above the
#: back for the entire cycle and the *hand* does the flapping - which is where
#: most of a wing's travel lives anyway.
_wy_F_TOP = 1.00          # humerus elevation at the top of the beat
_wy_F_BOT = 0.24          # ... and at the bottom: still above the back
#: How hard the hand cups down at the bottom of the stroke. Between this, the
#: lag and the fold, the fingertips swing through ~60 degrees while the arm
#: only moves 44, and the silhouette is a V at every frame of the cycle. Held
#: down deliberately: cup this hard enough and the hand points at the floor,
#: the whole outer wing foreshortens, and the bottom of the beat - the frame
#: that should look most powerful - is the frame with the *narrowest* span.
_wy_CUP = 0.30
_wy_HZ = 0.62             # wingbeats per second: heavy and slow
#: The far wing runs this fraction of a cycle behind the near one. Two exactly
#: mirrored wings read as a moth; a few frames of skew read as perspective.
_wy_WING_SKEW = 0.06

#: Torso, breast round to brisket. The deep chest at the front and the tucked
#: loin behind it are the two points doing the work - without that diagonal
#: this is a sausage.
_wy_BODY = (
    (12.0, 3.0),      # 0 breast, base of the neck
    (5.0, -1.6),      # 1 withers (the wing root hides just under here)
    (-7.0, -1.0),     # 2 back
    (-16.0, 1.6),     # 3 loin
    (-21.0, 4.2),     # 4 rump
    (-18.0, 8.4),     # 5 back of the thigh
    (-8.0, 11.0),     # 6 belly, tucked up
    (3.0, 12.6),      # 7 chest, deep
    (11.0, 9.6),      # 8 brisket
)

#: Root, mid, skull base. Long on purpose: with a short neck the head sits
#: inside the wing mass and the whole animal reads as a moth. The head has to
#: get clear of the wings, and low, while the wings go high.
_wy_NECK = ((11.5, 4.5), (21.0, 0.5), (31.0, 0.0))
_wy_NECK_W = (5.2, 3.6, 2.8)


# --------------------------------------------------------------- the wingbeat
def _wy_beat(t: float) -> tuple[float, float, float, float]:
    """``(elevation, hand lag, fold, cup)`` for the wings at time *t*.

    The downstroke takes 38% of the cycle and the recovery the rest, so the
    wing snaps down and drifts up. *lag* is the hand trailing the arm - up on
    the power stroke as the membrane cups air, down on the recovery - and it
    is the single term that makes a flap read as a flap rather than as a
    scissor. *fold* shortens the outer wing on the way up.
    """
    u = (t * _wy_HZ) % 1.0
    k = _wy_k_of(u)
    du = 0.004
    dn = (_wy_k_of(u + du) - k) / du / 2.7           # ~ -1 down, ~ +0.6 up
    dn = -1.0 if dn < -1.0 else (1.0 if dn > 1.0 else dn)
    f = _wy_F_BOT + (_wy_F_TOP - _wy_F_BOT) * k
    lag = -0.44 * dn
    fold = 1.0 - 0.24 * max(0.0, dn)
    return f, lag, fold, _wy_CUP * (1.0 - k)


def _wy_k_of(u: float) -> float:
    """1 at the top of the stroke, 0 at the bottom."""
    u = u % 1.0
    if u < 0.38:
        return 1.0 - _smooth(u / 0.38)
    return _smooth((u - 0.38) / 0.62)


def _wy_wing_pts(ox: float, f: float, lag: float, fold: float, cup: float,
                 anchor: tuple[float, float], reach: float
                 ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Membrane outline and the leading-edge chain for one wing.

    *ox* is +1 for the wing splayed toward the face and -1 for the one splayed
    toward the tail. Everything is in local space; the caller transforms.
    """
    rx, ry = _wy_WING_ROOT
    e1 = f
    e2 = f - 0.16 + lag * 0.5 - cup * 0.35
    eh = e2 - 0.10 + lag - cup

    dx, dy = _wy_pol(e1, _wy_HUM * reach)
    elbow = (rx + dx * ox, ry + dy)
    dx, dy = _wy_pol(e2, _wy_FORE * fold * reach)
    wrist = (elbow[0] + dx * ox, elbow[1] + dy)

    tips: list[tuple[float, float]] = []
    for sweep, frac in _wy_FINGERS:
        # The fingers gather as the wing folds on the recovery.
        sw = sweep * (1.0 + 0.22 * (1.0 - fold) / 0.24)
        dx, dy = _wy_pol(eh - sw, _wy_HAND * frac * fold * reach)
        tips.append((wrist[0] + dx * ox, wrist[1] + dy))

    anc = anchor

    # Trailing edge: scalloped between the fingertips. The scallops are what
    # separate a dragon from a bird at 30 px - a bird's trailing edge is a
    # smooth curve, a membrane's is a row of bites.
    outline: list[tuple[float, float]] = [(rx, ry), elbow, wrist, tips[0]]
    for a, b in zip(tips, tips[1:]):
        mx, my = (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5
        outline.append((mx + (wrist[0] - mx) * 0.26, my + (wrist[1] - my) * 0.26))
        outline.append(b)
    mx, my = (tips[-1][0] + anc[0]) * 0.5, (tips[-1][1] + anc[1]) * 0.5
    outline.append((mx + (wrist[0] - mx) * 0.12, my + (wrist[1] - my) * 0.12))
    outline.append(anc)

    chain = [(rx, ry), elbow, wrist] + tips
    return outline, chain


def _wy_draw_wing(surf: pygame.Surface, xf: Callable[..., tuple[float, float]],
                  ox: float, beat: tuple[float, float, float, float],
                  memb: Color, rib: Color, lead: Color, horn: Color,
                  anchor: tuple[float, float], reach: float, s: float) -> None:
    outline, chain = _wy_wing_pts(ox, *beat, anchor=anchor, reach=reach)
    _poly(surf, memb, [xf(*p) for p in outline])

    root, elbow, wrist = chain[0], chain[1], chain[2]
    w1 = max(1, int(round(1.0 * s)))
    w2 = max(1, int(round(2.0 * s)))

    # Ribs, darker than the membrane, so the sheet reads as skin stretched over
    # bone rather than as a paper kite.
    for tip in chain[4:]:
        _line(surf, rib, xf(*wrist), xf(*tip), w1)

    # One lit edge and no more. creatures.py is explicit about this and the
    # first pass here ignored it: a lit leading edge on both wings *plus* a lit
    # spine *plus* pale claws on eight fingertips is not rim lighting, it is
    # confetti, and it shatters the mass the membrane exists to make. *lead*
    # arrives already dimmed for the far wing, which is the depth cue.
    _line(surf, lead, xf(*root), xf(*elbow), w2)
    _line(surf, lead, xf(*elbow), xf(*wrist), w2)
    _line(surf, lead, xf(*wrist), xf(*chain[3]), w1)

    # Spurs off each fingertip, so the wing edge is spiky rather than
    # scalloped-smooth. Membrane-dark: these are for the *outline*, and the
    # outline is all that survives at wallpaper scale anyway.
    for tip in chain[3:]:
        dx, dy = tip[0] - wrist[0], tip[1] - wrist[1]
        d = math.hypot(dx, dy) or 1.0
        _line(surf, memb, xf(*tip),
              xf(tip[0] + dx / d * 3.0, tip[1] + dy / d * 3.0), w1)
    # The wrist claw. Heraldry always draws it and it is the one place a hard
    # highlight earns its keep: it puts a point on the leading edge.
    dx, dy = wrist[0] - elbow[0], wrist[1] - elbow[1]
    d = math.hypot(dx, dy) or 1.0
    ux, uy = dx / d, dy / d
    _poly(surf, horn, [
        xf(wrist[0] - ux * 1.6 - uy * 0.8, wrist[1] - uy * 1.6 + ux * 0.8),
        xf(wrist[0] + ux * 3.4 - uy * 2.4, wrist[1] + uy * 3.4 + ux * 2.4),
        xf(wrist[0] - ux * 0.4 + uy * 0.7, wrist[1] - uy * 0.4 - ux * 0.7),
    ])


# --------------------------------------------------------------------- tail
def _wy_tail_spine(t: float, n: int = 15) -> list[tuple[float, float, float]]:
    """``(x, y, half-width)`` down the tail, rump to the base of the spade."""
    seg = 4.6
    x, y = -19.5, 5.0
    out: list[tuple[float, float, float]] = []
    for i in range(n):
        u = i / (n - 1.0)
        # Long and mostly straight, with a sag out of the rump and a lazy curl
        # at the end. An earlier version curled hard through 90 degrees and the
        # tail coiled up into a paddle beside the body - which reads as a third
        # small wing, not as a whip. Length along the ground is the cue; the
        # curl is only there to stop it being a stick.
        a = math.pi - 0.40 + 0.72 * u * u
        a += 0.30 * math.sin(u * 3.4 - t * _wy_HZ * TAU + 0.9) * (0.2 + 0.8 * u)
        # Tapers hard and early. A tail that leaves the hip as thick as the
        # hip is not a tail, it is more body - the torso and the tail fuse into
        # one green tube and the animal loses its waist.
        # The 0.9 floor is not decoration: at 0.5x a tail that tapers to
        # nothing is one antialiased grey pixel and the whip is gone.
        out.append((x, y, 3.0 * (1.0 - u) ** 0.95 + 0.9))
        x += math.cos(a) * seg
        y += math.sin(a) * seg
    return out


def _wy_draw_tail(surf: pygame.Surface, xf: Callable[..., tuple[float, float]],
                  t: float, hide: Color, rim: Color, s: float) -> None:
    sp = _wy_tail_spine(t)
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    norms: list[tuple[float, float]] = []
    for i, (x, y, w) in enumerate(sp):
        j = min(i + 1, len(sp) - 1)
        dx, dy = sp[j][0] - x, sp[j][1] - y
        if i == len(sp) - 1:
            dx, dy = x - sp[i - 1][0], y - sp[i - 1][1]
        d = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / d, dx / d
        norms.append((nx, ny))
        left.append((x + nx * w, y + ny * w))
        right.append((x - nx * w, y - ny * w))
    _poly(surf, hide, [xf(*p) for p in left] + [xf(*p) for p in reversed(right)])

    # A serrated topline. At wallpaper scale a smooth back is a lizard and a
    # sawtooth one is a dragon, and it costs four triangles.
    for i in range(1, 10, 2):
        x, y, w = sp[i]
        nx, ny = norms[i]
        j = min(i + 1, len(sp) - 1)
        dx, dy = sp[j][0] - x, sp[j][1] - y
        d = math.hypot(dx, dy) or 1.0
        h = 3.4 * (1.0 - i / 11.0) + 1.0
        _poly(surf, hide, [
            xf(x - nx * w - dx / d * 1.8, y - ny * w - dy / d * 1.8),
            xf(x - nx * (w + h), y - ny * (w + h)),
            xf(x - nx * w + dx / d * 1.8, y - ny * w + dy / d * 1.8),
        ])

    # The spade. A whip tail that just tapers to nothing reads as a rope; the
    # arrowhead is the cheapest dragon cue on the whole animal.
    tx, ty, _ = sp[-1]
    dx, dy = tx - sp[-2][0], ty - sp[-2][1]
    d = math.hypot(dx, dy) or 1.0
    ux, uy = dx / d, dy / d
    px, py = -uy, ux
    # Longer than it is wide, or it is a canoe paddle rather than an arrowhead.
    spade = [
        (tx - ux * 2.0, ty - uy * 2.0),
        (tx + ux * 4.5 + px * 4.2, ty + uy * 4.5 + py * 4.2),
        (tx + ux * 13.0, ty + uy * 13.0),
        (tx + ux * 4.5 - px * 4.2, ty + uy * 4.5 - py * 4.2),
    ]
    _poly(surf, hide, [xf(*p) for p in spade])
    _line(surf, rim, xf(*spade[1]), xf(*spade[2]), max(1, int(round(1.0 * s))))


# ---------------------------------------------------------------- head, legs
def _wy_draw_head(surf: pygame.Surface, xf: Callable[..., tuple[float, float]],
                  hide: Color, rim: Color, horn: Color, eye: Color | None,
                  s: float) -> None:
    bx, by = _wy_NECK[2]
    # Skull as a wedge, not a disc: at this size a circle is a bead on a
    # string and a wedge is a reptile.
    _poly(surf, hide, [xf(*p) for p in (
        (bx - 1.5, by - 4.0), (bx + 5.5, by - 3.4), (bx + 12.2, by + 1.6),
        (bx + 11.6, by + 3.2), (bx + 3.0, by + 4.0), (bx - 1.8, by + 2.6),
    )])
    # Lower jaw, cracked open a hair. The gap of sky between skull and jaw is
    # what makes it a mouth instead of a snout.
    _poly(surf, hide, [xf(*p) for p in (
        (bx + 1.0, by + 3.6), (bx + 10.8, by + 3.8),
        (bx + 10.4, by + 5.4), (bx + 1.2, by + 5.4),
    )])
    # Horns, swept back *and up* over the neck. Thin spikes with a gap of sky
    # between them - drawn as two fat wedges off the same root they merge into
    # one blunt shape and the head reads as a helmet. The pair, and the sky
    # between the pair, is the cue.
    dx, dy = -0.72, -0.69
    px, py = -dy, dx
    for rx, ry, ln, hw in ((bx + 2.6, by - 3.0, 8.4, 1.7),
                           (bx - 0.4, by - 2.2, 6.2, 1.4)):
        _poly(surf, horn, [xf(*p) for p in (
            (rx + px * hw, ry + py * hw),
            (rx + dx * ln, ry + dy * ln),
            (rx - px * hw, ry - py * hw),
        )])
    _line(surf, rim, xf(bx - 1.4, by - 3.8), xf(bx + 11.8, by + 1.6),
          max(1, int(round(1.0 * s))))
    if eye is not None:
        p = xf(bx + 5.4, by - 0.4)
        r = max(1, int(round(1.9 * s)))
        pygame.draw.rect(surf, eye, pygame.Rect(int(p[0]), int(p[1]), r, r))


def _wy_draw_leg(surf: pygame.Surface, xf: Callable[..., tuple[float, float]],
                 dx: float, dy: float, col: Color, horn: Color, s: float) -> None:
    """One hind leg, hanging under the belly with the foot cocked forward.

    A wyvern has two of these and no forelimbs, and letting them hang clear of
    the body instead of tucking them away is the cheapest thing that separates
    this from a bat.
    """
    hip = (-11.0 + dx, 8.4 + dy)
    knee = (-17.0 + dx, 16.4 + dy)
    ankle = (-9.4 + dx, 20.2 + dy)
    toe = (-4.2 + dx, 19.0 + dy)
    # A haunch, the way creatures.py hangs its quadruped legs off a disc: two
    # bare lines under a torso read as wire, and the mass is what makes the leg
    # grow out of something.
    _disc(surf, col, xf(*hip), 4.2)
    w = max(1, int(round(3.0 * s)))
    _line(surf, col, xf(*hip), xf(*knee), w)
    _line(surf, col, xf(*knee), xf(*ankle), max(1, w - 1))
    # The foot as a filled wedge rather than a line, so there is a fist of
    # talons under the belly instead of a fraying thread.
    _poly(surf, col, [xf(*p) for p in (
        (ankle[0] - 1.6, ankle[1] - 1.4), (toe[0] + 1.0, toe[1] - 1.2),
        (toe[0], toe[1] + 1.8), (ankle[0] - 1.2, ankle[1] + 1.4))])
    for a in (-0.62, 0.02, 0.66):
        _line(surf, horn, xf(*toe),
              xf(toe[0] + math.cos(a) * 4.0, toe[1] + math.sin(a) * 4.0),
              max(1, int(round(1.0 * s))))


# ---------------------------------------------------------------- main draw
def _wy_draw(surf: pygame.Surface, x: float, y: float, t: float,
             scale: float, flat: Sequence[int] | None, facing: int) -> None:
    """Draw a wyvern in low flight, wings beating, at render time *t*.

    ``(x, y)`` is the wing root - the withers, and the creature's visual
    centre of mass. *flat*, if given, forces every colour to that one colour,
    the way ``creatures.py`` collapses an unlit animal to a silhouette.
    """
    if not (math.isfinite(x) and math.isfinite(y)) or scale <= 0.0:
        return
    s = float(scale)
    beat = _wy_beat(t)
    k = _wy_k_of((t * _wy_HZ) % 1.0)

    hide = flat or _wy_HIDE
    rim = flat or _wy_HIDE_RIM
    shade = flat or _wy_HIDE_SHADE
    memb = flat or _wy_MEMB
    rib = flat or _wy_MEMB_RIB
    lead = flat or _wy_MEMB_LEAD
    horn = flat or _wy_HORN
    eye: Color | None = None if flat else _wy_EYE

    def far(c: Color) -> Color:
        return c if flat else _dim(c, _wy_FAR)

    hide_far = far(hide)

    # The body sinks as the wings come up and is thrown up on the power
    # stroke, and pitches a little with it.
    bob = 2.8 * (k - 0.5)
    pitch = 0.22 * (k - 0.5)     # nose up on the power stroke, and it is climbing
    mir = -1.0 if facing < 0 else 1.0
    cp, sp_ = math.cos(pitch), math.sin(pitch)

    def xf(lx: float, ly: float) -> tuple[float, float]:
        rx = lx * cp - ly * sp_
        ry = lx * sp_ + ly * cp + bob
        return (x + rx * s * mir, y + ry * s)

    # Layer order, and it is load-bearing. The two features that say "wyvern"
    # rather than "bat" are the spade tail and the horned head, and each of
    # them shares a side of the body with a wing that would otherwise paint
    # over it. So the far wing is the one splayed back over the *tail* and the
    # near wing the one splayed forward over the *head*, and the tail is drawn
    # after the first while the neck and head are drawn after the second:
    #
    #   far wing (back) -> tail -> body -> legs -> near wing (front)
    #   -> neck -> head
    #
    # Putting the head in front of the near wing is a small cheat against the
    # depth order. It buys a crisp horned wedge against a big dark membrane,
    # every frame of the beat, and nothing in the drawing gives it away.
    _wy_draw_wing(surf, xf, -1.0, _wy_beat(t - _wy_WING_SKEW / _wy_HZ),
                  far(memb), far(rib), far(lead), far(horn),
                  _wy_TRAIL_FAR, _wy_REACH_FAR, s)

    _wy_draw_tail(surf, xf, t, hide, rim, s)

    body = [xf(*p) for p in _wy_BODY]
    _poly(surf, hide, body)

    # Dorsal crest over the loin - the stretch of back the near wing does not
    # cover. Between this, the neck crest and the tail's sawtooth the whole
    # topline is serrated from horn to spade, which at wallpaper scale is most
    # of the difference between a dragon and a large bat.
    crest = flat or _dim(_wy_HIDE_RIM, 0.58)
    for u in (0.16, 0.46, 0.76):
        ax = -7.0 + (-19.5 - -7.0) * u
        ay = -1.0 + (3.4 - -1.0) * u
        h = 4.0 - 1.6 * u
        _poly(surf, crest, [xf(*p) for p in (
            (ax + 2.2, ay - 0.2), (ax - 0.2, ay - h), (ax - 2.2, ay + 0.6))])

    _wy_draw_leg(surf, xf, 2.2, -1.6, hide_far, hide_far, s)
    _wy_draw_leg(surf, xf, 0.0, 0.0, hide, horn, s)

    # One lit edge along the spine and one dark one under the belly: the trick
    # from creatures.py that stops a flat fill reading as a paper cutout.
    lw = max(1, int(round(1.0 * s)))
    for a, b in zip(body[0:4], body[1:5]):
        _line(surf, rim, a, b, lw)
    for a, b in zip(body[5:8], body[6:9]):
        _line(surf, shade, a, b, lw)

    _wy_draw_wing(surf, xf, +1.0, beat, memb, rib, lead, horn, _wy_TRAIL_NEAR, 1.0, s)

    # Neck as a tapered wedge, plus a crest. A one-pixel neck makes the head a
    # balloon on a string; the crest is what stops the wedge reading as a wing.
    nk: list[tuple[float, float]] = []
    nk2: list[tuple[float, float]] = []
    for i, (nx, ny) in enumerate(_wy_NECK):
        j = min(i + 1, len(_wy_NECK) - 1)
        dx, dy = _wy_NECK[j][0] - nx, _wy_NECK[j][1] - ny
        if i == len(_wy_NECK) - 1:
            dx, dy = nx - _wy_NECK[i - 1][0], ny - _wy_NECK[i - 1][1]
        d = math.hypot(dx, dy) or 1.0
        w = _wy_NECK_W[i]
        px, py = -dy / d * w, dx / d * w
        nk.append((nx + px, ny + py))
        nk2.append((nx - px, ny - py))
    _poly(surf, hide, [xf(*p) for p in nk + list(reversed(nk2))])
    for i in range(4):
        u = 0.12 + i * 0.26
        a = _wy_NECK[0][0] + (_wy_NECK[2][0] - _wy_NECK[0][0]) * u
        b = _wy_NECK[0][1] + (_wy_NECK[2][1] - _wy_NECK[0][1]) * u - 3.4
        _poly(surf, flat or _dim(_wy_HIDE_RIM, 0.58), [xf(*p) for p in (
            (a - 2.2, b + 1.2), (a + 0.4, b - 3.2), (a + 2.2, b + 1.0))])

    _wy_draw_head(surf, xf, hide, rim, horn, eye, s)


# --------------------------------------------------------------------------
# quadruped - THE SIEGE. The only GROUND-anchored body in the module: (x, y)
# is the feet. It has no flight pose at all - t drives a grounded mantle cycle
# - which is why draw_one pins it to the terrain whatever `alt` says.
# --------------------------------------------------------------------------
def _qd_add(a: Pt, b: Pt) -> Pt:
    return (a[0] + b[0], a[1] + b[1])


def _qd_mix(a: Pt, b: Pt, u: float) -> Pt:
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)


def _qd_unit(a: Pt, b: Pt) -> Pt:
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = math.hypot(dx, dy)
    return (1.0, 0.0) if d < 1e-6 else (dx / d, dy / d)


def _qd_bez(p0: Pt, p1: Pt, p2: Pt, p3: Pt, n: int) -> list[Pt]:
    """Cubic Bezier, sampled n+1 times. The spine of every curved part."""
    out: list[Pt] = []
    for i in range(n + 1):
        u = i / n
        v = 1.0 - u
        a, b, c, d = v * v * v, 3 * v * v * u, 3 * v * u * u, u * u * u
        out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                    a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return out


def _qd_qbez(p0: Pt, p1: Pt, p2: Pt, n: int) -> list[Pt]:
    out: list[Pt] = []
    for i in range(n + 1):
        u = i / n
        v = 1.0 - u
        a, b, c = v * v, 2 * v * u, u * u
        out.append((a * p0[0] + b * p1[0] + c * p2[0],
                    a * p0[1] + b * p1[1] + c * p2[1]))
    return out


def _qd_norms(pts: Sequence[Pt]) -> list[Pt]:
    """Per-point normals, averaged across the joint."""
    n = len(pts)
    out: list[Pt] = []
    for i in range(n):
        a = pts[max(0, i - 1)]
        b = pts[min(n - 1, i + 1)]
        u = _qd_unit(a, b)
        out.append((-u[1], u[0]))
    return out


def _qd_strip(pts: Sequence[Pt], w0: float, w1: float, gamma: float = 1.0) -> list[Pt]:
    """A tapered ribbon around a polyline: the neck, the tail, a limb.

    *gamma* below 1 front-loads the taper. A linear taper down a long curve
    stays fat far too long, which is what made the first neck read as a
    sauropod's rather than a dragon's.
    """
    nm = _qd_norms(pts)
    n = len(pts)
    left: list[Pt] = []
    right: list[Pt] = []
    for i, p in enumerate(pts):
        h = (w0 + (w1 - w0) * (i / (n - 1)) ** gamma) * 0.5
        left.append((p[0] + nm[i][0] * h, p[1] + nm[i][1] * h))
        right.append((p[0] - nm[i][0] * h, p[1] - nm[i][1] * h))
    right.reverse()
    return left + right


def _qd_up_perp(u: Pt) -> Pt:
    """Whichever perpendicular points more nearly *up* (crest side)."""
    a = (u[1], -u[0])
    b = (-u[1], u[0])
    return a if a[1] < b[1] else b


def _qd_spines(surf: pygame.Surface, col: Sequence[int], pts: Sequence[Pt],
               h0: float, h1: float, every: int, back: float = 0.45) -> None:
    """Sawtooth crest along a spine. Each tooth leans back along the body."""
    n = len(pts)
    for i in range(1, n - 1, every):
        u = _qd_unit(pts[i - 1], pts[i + 1])
        p = _qd_up_perp(u)
        f = i / (n - 1)
        h = h0 + (h1 - h0) * f
        if h < 0.8:
            continue
        base = pts[i]
        b0 = (base[0] - u[0] * h * 0.55, base[1] - u[1] * h * 0.55)
        b1 = (base[0] + u[0] * h * 0.55, base[1] + u[1] * h * 0.55)
        tip = (base[0] + p[0] * h - u[0] * h * back,
               base[1] + p[1] * h - u[1] * h * back)
        _poly(surf, col, [b0, tip, b1])


# ------------------------------------------------------------ proportions --
#: Shoulder height in pixels at ``scale=1.0``. A stickman is 26 px, so the
#: dragon's shoulder is chin-height on a person and the furled wings top out
#: at nearly three times a person - the size of a hut, which is the point.
_qd_SHOULDER_H = 34.0
_qd_BODY_L = 46.0

# Flat palette, same discipline as BUILDS: a body tone, a rim, a shade, and one
# accent. The membrane is deliberately its own *lighter* tone - a wing painted
# the body colour welds itself to the shoulder and the sail stops reading, which
# is exactly what the first pass of this file did.
# Values matter more than hues here. A hide much darker than the sky made the
# body vanish at wallpaper distance and left a pink sail floating on its own -
# the wolf's fur is (126,128,140) for exactly this reason, and the dragon has to
# hold the same kind of value against the sky or it is not a silhouette, it is
# a stain.
_qd_HIDE: Color = (92, 104, 88)
_qd_RIM: Color = (176, 196, 162)
_qd_MEMBRANE: Color = (150, 96, 90)
_qd_HORN: Color = (146, 144, 128)
_qd_EYE: Color = (255, 186, 66)


# ---------------------------------------------------------------- the beat --
def _qd_wing_phase(t: float) -> tuple[float, float]:
    """``(raise, beat)`` for the mantle cycle.

    Furled for most of it, then thrown up, two beats at the top, then settled.
    A grounded dragon that flaps continuously reads as a chicken.
    """
    c = (t * 0.42) % 1.0
    if c < 0.14:
        r = 0.0
    elif c < 0.34:
        r = _smooth((c - 0.14) / 0.20)
    elif c < 0.74:
        r = 1.0
    elif c < 0.94:
        r = 1.0 - _smooth((c - 0.74) / 0.20)
    else:
        r = 0.0
    beat = math.sin((c - 0.24) * TAU * 2.4) * r
    return r, beat


# ------------------------------------------------------------------- wing --
def _qd_wing_poly(root: Pt, H: float, rise: float, beat: float, s: float,
                  anchor: Pt, back: Sequence[Pt]) -> tuple[list[Pt], Pt, Pt, list[Pt]]:
    """The whole wing as one filled polygon, plus the spars to overdraw.

    Built trailing-edge-first, because the trailing edge *is* the silhouette:
    a curve from the wrist apex round to the anchor on the back, with the finger
    tips pushed out of it and the membrane sagging in between. Building it as a
    fan of digits instead - the anatomically honest way - kept producing
    self-crossing outlines the moment the wing furled, because furled digits
    stack on top of each other.
    """
    def L(a: Pt, b: Pt, u: float) -> Pt:
        return _qd_mix(a, b, u)

    # furled -> raised, in units of shoulder height.
    #
    # The elbow has to stay *inside* the apex-to-anchor line or the wing has no
    # area at all: slung far back, the leading and trailing edges ran down the
    # same corridor and the sail collapsed to a sliver with a flap hanging off
    # it. Forward of that line the membrane is a real shape, and the bow only
    # has to add to it rather than rescue it.
    elbow = L(_qd_add(root, (-0.10 * H, -0.52 * H)),
              _qd_add(root, (-0.05 * H, -0.92 * H)), rise)
    apex = L(_qd_add(root, (0.20 * H, -0.96 * H)),
             _qd_add(root, (0.26 * H, -1.62 * H)), rise)
    bulge = (0.22 + 0.72 * rise) * H
    # Long tips, shallow sags. Shallow tips sanded off at wallpaper scale and
    # left a smooth triangle (a shark fin); sags as deep as the tips ate the
    # membrane away and left a comb of spikes, which is worse - in flat
    # silhouette it read as a broken umbrella. The membrane has to stay a solid
    # mass with fingers poking *through* its trailing edge.
    tip = (0.19 + 0.10 * rise) * H
    sag = (0.09 + 0.05 * rise) * H

    # the beat rotates the arm about the shoulder root
    if abs(beat) > 1e-3:
        a = beat * 0.30
        ca, sa = math.cos(a), math.sin(a)

        def rot(p: Pt) -> Pt:
            dx, dy = p[0] - root[0], p[1] - root[1]
            return (root[0] + dx * ca - dy * sa, root[1] + dx * sa + dy * ca)

        elbow, apex = rot(elbow), rot(apex)

    # trailing edge: apex -> anchor, bulging backward
    mid = _qd_mix(apex, anchor, 0.5)
    u = _qd_unit(apex, anchor)
    out = _qd_up_perp((-u[0], -u[1]))          # points back-and-up off the chord
    ctrl = (mid[0] + out[0] * bulge, mid[1] + out[1] * bulge)
    arc = _qd_qbez(apex, ctrl, anchor, 8)

    # finger tips poke out of the arc; the membrane sags between them
    # Digits get different lengths. Four equal teeth on an even arc read as a
    # staircase - stegosaur plates, not fingers - and the eye picks up the
    # regularity before it picks up the shape.
    dig = (1.18, 1.00, 0.82, 0.55)
    edge: list[Pt] = [apex]
    tips: list[Pt] = []
    for i in range(1, 8):
        p = arc[i]
        nu = _qd_unit(arc[i - 1], arc[i + 1])
        nrm = _qd_up_perp((-nu[0], -nu[1]))
        if i % 2 == 1:                       # a digit
            k = dig[min(len(dig) - 1, i // 2)]
            q = (p[0] + nrm[0] * tip * k, p[1] + nrm[1] * tip * k)
            tips.append(q)
        else:                                # the sag between two digits
            q = (p[0] - nrm[0] * sag, p[1] - nrm[1] * sag)
        edge.append(q)
    edge.append(anchor)

    # ...and home along the topline. Closing straight from the anchor to the
    # root cut a chord across the animal's back, and a membrane sitting *in* the
    # body reads as a saddle. Routing the closing edge through the spine points
    # buries it under the outline instead.
    poly = [root, elbow] + edge + list(back)
    return poly, elbow, apex, tips


def _qd_draw_wing(surf: pygame.Surface, xf, root: Pt, anchor: Pt, back: Sequence[Pt],
                  H: float, rise: float, beat: float, s: float,
                  mem: Color, spar: Color, hide: Color, flat: bool) -> None:
    poly, elbow, apex, tips = _qd_wing_poly(root, H, rise, beat, s, anchor, back)
    _poly(surf, mem, [xf(*p) for p in poly])
    w = max(1, int(round(2.2 * s)))
    # spars: the arm along the leading edge, then a digit to each tip. Dark ribs
    # on a lighter membrane - drawn lighter than the membrane they vanished, and
    # a sail with no ribs in it reads as a fin.
    _line(surf, hide, xf(*root), xf(*elbow), w + 1)
    _line(surf, hide, xf(*elbow), xf(*apex), w)
    for q in tips:
        _line(surf, spar, xf(*apex), xf(*q), max(1, w - 1))
    # the wrist claw: a hook at the top of the sail, and the one detail that
    # says "wing" rather than "fin" when the thing is furled
    u = _qd_unit(elbow, apex)
    claw = (apex[0] + u[0] * 5.2 * s - u[1] * 3.0 * s,
            apex[1] + u[1] * 5.2 * s + u[0] * 3.0 * s)
    _line(surf, hide, xf(*apex), xf(*claw), max(1, w))


def _qd_draw(surf: pygame.Surface, x: float, y: float, t: float, s: float,
             flat: Sequence[int] | None, facing: int) -> None:
    # Same first line as `creatures._draw_animal`: junk in, nothing out. The
    # fail-soft wrapper would catch a NaN anyway, but it would catch it once per
    # animal per frame with a traceback, which is not free.
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(t)):
        return
    if not math.isfinite(s) or s <= 0.02:
        return
    H = _qd_SHOULDER_H * s
    L = _qd_BODY_L * s
    mir = -1.0 if facing < 0 else 1.0

    isflat = flat is not None
    F: Color | None = (int(flat[0]), int(flat[1]), int(flat[2])) if isflat else None

    def C(c: Color) -> Color:
        return F if F is not None else c

    hide = C(_qd_HIDE)
    far = C(_dim(_qd_HIDE, 0.66))
    rim = C(_qd_RIM)
    shade = C(_dim(_qd_HIDE, 0.58))
    mem = C(_qd_MEMBRANE)
    # ...only a little darker. At 0.58 the far membrane was darker than the hide
    # and the second wing read as a hole punched through the first one.
    mem_far = C(_dim(_qd_MEMBRANE, 0.78))
    mem_spar = C(_dim(_qd_MEMBRANE, 0.50))
    horn = C(_qd_HORN)

    rise, beat = _qd_wing_phase(t)
    breath = math.sin(t * 1.1) * 0.4 * s
    bob = breath - rise * 0.9 * s

    def xf(lx: float, ly: float) -> Pt:
        return (x + lx * mir, y + ly + bob)

    # ------------------------------------------------------------- far side --
    # far legs, far wing, far horn: drawn first and dimmed, the same cheap
    # depth cue the stickmen and the wolves use.
    _qd_legs(surf, xf, H, L, s, far, far, t, near=False)
    # The far wing goes *forward* of the near one, not behind it. Tucked behind
    # it was swallowed whole by the near sail's bulge and the animal had one
    # wing; forward it clears the leading edge and the silhouette gets its
    # second peak, which is what says "a pair of wings" rather than "a fin".
    # The far wing is *implied* while furled and only becomes a wing as the
    # animal mantles. Drawn at full size alongside the near one it produced two
    # overlapping sails of similar height, and their merged outline was a crown
    # of spikes - unreadable flat. Small, it clears the near sail's leading edge
    # as a single spur and the eye supplies the rest.
    _qd_draw_wing(surf, xf, (0.36 * L, -0.90 * H), (-0.12 * L, -0.98 * H),
                  ((0.14 * L, -0.92 * H),), H * (0.42 + 0.50 * rise),
                  rise, -beat * 0.85, s, mem_far, mem_far, far, isflat)

    # ---------------------------------------------------------------- tail --
    tail = _qd_bez((-0.52 * L, -0.82 * H), (-1.02 * L, -0.78 * H),
                   (-1.44 * L, -0.12 * H), (-1.62 * L, -0.07 * H), 14)
    sway = math.sin(t * 0.9) * 0.05
    if abs(sway) > 1e-3:                     # the tail drifts, it is not a stick
        for i, p in enumerate(tail):
            f = (i / (len(tail) - 1)) ** 2
            tail[i] = (p[0], p[1] - f * sway * 26.0 * s)
    _poly(surf, hide, [xf(*p) for p in _qd_strip(tail, 0.24 * H, 0.035 * H)])

    # tail spade
    tu = _qd_unit(tail[-3], tail[-1])
    tp = (-tu[1], tu[0])
    tipp = tail[-1]
    _poly(surf, hide, [
        xf(tipp[0] + tu[0] * 5.0 * s, tipp[1] + tu[1] * 5.0 * s),
        xf(tipp[0] - tu[0] * 3.0 * s + tp[0] * 4.6 * s,
           tipp[1] - tu[1] * 3.0 * s + tp[1] * 4.6 * s),
        xf(tipp[0] - tu[0] * 8.0 * s, tipp[1] - tu[1] * 8.0 * s),
        xf(tipp[0] - tu[0] * 3.0 * s - tp[0] * 4.6 * s,
           tipp[1] - tu[1] * 3.0 * s - tp[1] * 4.6 * s),
    ])

    # ---------------------------------------------------------------- body --
    # Rump higher than the shoulder: a cat about to spring, not a horse. The
    # topline is what carries the crouch, because the leg joints are only a few
    # pixels apart at this size and their angles do not survive.
    outline = [
        (0.50 * L, -0.86 * H),               # 0 breast, top
        (0.34 * L, -1.02 * H),               # 1 shoulder / wing root
        (0.06 * L, -0.98 * H),               # 2 dip in the back
        (-0.24 * L, -1.12 * H),              # 3 croup, high
        (-0.50 * L, -0.96 * H),              # 4 rump
        (-0.60 * L, -0.74 * H),              # 5 tail base
        (-0.34 * L, -0.60 * H),              # 6 behind the thigh
        (0.00 * L, -0.55 * H),               # 7 belly, tucked
        (0.30 * L, -0.50 * H),               # 8 chest, deep
        (0.50 * L, -0.64 * H),               # 9 breast, low
    ]
    body = [xf(*p) for p in outline]
    _poly(surf, hide, body)
    _disc(surf, hide, xf(0.29 * L, -0.78 * H), 0.30 * H)      # shoulder mass
    _disc(surf, hide, xf(-0.30 * L, -0.78 * H), 0.34 * H)     # haunch

    # ---------------------------------------------------------------- neck --
    # Short and thick. The first pass gave this animal a neck 40 px long and it
    # read as a sauropod - a dragon's head belongs close over its shoulders.
    neck = _qd_bez((0.44 * L, -0.92 * H), (0.60 * L, -1.54 * H),
                   (0.86 * L, -1.52 * H), (0.98 * L, -1.28 * H), 12)
    look = math.sin(t * 0.73 + 1.1) * 0.030
    for i, p in enumerate(neck):
        f = (i / (len(neck) - 1)) ** 2
        neck[i] = (p[0] + f * look * 30.0 * s, p[1] - f * look * 18.0 * s
                   - f * rise * 5.0 * s)
    _poly(surf, hide, [xf(*p) for p in _qd_strip(neck, 0.44 * H, 0.13 * H, 0.55)])

    # ---------------------------------------------------------------- head --
    # The head gets its own angle rather than inheriting the neck's end tangent.
    # Inheriting it pitched the skull 45 degrees nose-down - the horns came out
    # of the top of a bird's head and the whole animal read as pecking.
    hang = 0.20 + look * 1.6 - rise * 0.10
    hu = (math.cos(hang), math.sin(hang))
    hp = (-hu[1], hu[0])                     # "down" in head space
    hc = (neck[-1][0] + hu[0] * 3.0 * s, neck[-1][1] + hu[1] * 3.0 * s)

    def hpt(a: float, b: float) -> Pt:
        return (hc[0] + hu[0] * a * s + hp[0] * b * s,
                hc[1] + hu[1] * a * s + hp[1] * b * s)

    # Skull and lower jaw as two polygons with daylight between them. The gap is
    # under 2 px at 0.5x and closes up, but at full size an agape jaw is the
    # single cheapest "this is a predator with a mouth" cue there is.
    gape = 1.6 + 1.4 * rise
    _poly(surf, hide, [xf(*hpt(*p)) for p in (
        (-8.0, -4.6), (2.0, -7.2), (12.0, -5.4), (20.0, -1.2),
        (16.5, 1.4), (-7.0, 2.0))])
    _poly(surf, hide, [xf(*hpt(a, b + gape)) for a, b in (
        (-7.0, 0.6), (3.0, 2.0), (15.0, 3.0), (13.5, 5.2), (-6.5, 4.0))])
    # teeth: two chips off the upper jaw, only worth it at full size
    if s > 0.7:
        for a in (15.0, 9.0):
            _poly(surf, horn, [xf(*hpt(a - 1.3, 0.4)), xf(*hpt(a, 3.2)),
                               xf(*hpt(a + 1.3, 0.4))])
    # horns, swept back off the skull, near and far
    # ...short, dull and *tapered*. Drawn as a uniform near-white line this was a
    # sword balanced on the skull: a bright object floating over the neck rather
    # than part of the head. A triangle rooted in the skull extends the head's
    # silhouette instead of competing with it.
    for col, dx, dy, ln, w in (
            (_qd_dim_or(horn, 0.62, isflat), -4.0, -1.4, 8.5, 2.0),
            (horn, -5.5, -5.0, 12.0, 2.8)):
        tip = hpt(dx - ln * 0.52, dy - ln * 0.88)
        _poly(surf, col, [xf(*hpt(dx + w, dy + w * 0.4)), xf(*tip),
                          xf(*hpt(dx - w, dy - w * 0.4))])
    # jaw spikes under the chin and at the cheek
    for a, b, ln in ((7.0, 5.0 + gape, 5.5), (-5.0, 2.2, 6.0)):
        _line(surf, hide, xf(*hpt(a, b)),
              xf(*hpt(a - ln * 0.7, b + ln * 0.7)), max(1, int(round(1.8 * s))))

    # ------------------------------------------------------------ near side --
    _qd_legs(surf, xf, H, L, s, hide, shade, t, near=True)

    # crest: horns -> neck -> croup -> tail, one continuous sawtooth. Nothing
    # between the shoulder and the croup: the furled wing covers that stretch of
    # back, so spines there are drawn and then painted over.
    _qd_spines(surf, hide, neck, 0.21 * H, 0.15 * H, 2)
    _qd_spines(surf, hide, [outline[3], outline[4], outline[5]], 0.15 * H, 0.13 * H, 1)
    _qd_spines(surf, hide, tail, 0.15 * H, 0.02 * H, 2)

    if not isflat:
        # one lit edge along the top, one dark one under the belly - the whole
        # trick that stops a flat fill reading as a paper cutout
        prev = body[9]
        for p in (body[0], body[1], body[2], body[3], body[4]):
            _line(surf, rim, prev, p, 1)
            prev = p
        prev = body[5]
        for p in (body[6], body[7], body[8], body[9]):
            _line(surf, shade, prev, p, 1)
            prev = p
        _line(surf, rim, xf(*hpt(-5.0, -3.6)), xf(*hpt(3.0, -4.0)), 1)
        _line(surf, rim, xf(*hpt(3.0, -4.0)), xf(*hpt(12.5, -0.8)), 1)
        # mouth line and the eye
        _line(surf, shade, xf(*hpt(11.0, 0.9)), xf(*hpt(1.0, 2.2)), 1)
        er = max(1, int(round(1.8 * s)))
        ep = xf(*hpt(2.2, -1.0))
        pygame.draw.rect(surf, _qd_EYE, pygame.Rect(int(ep[0]), int(ep[1]), er, er))

    # near wing last: it is the read, so nothing may cross it
    # The anchor goes all the way back to the rump. Anchored short, over the
    # croup, the trailing edge was a 28 px chord under a fat bulge and all four
    # digit tips bunched around the top of it - a cockscomb, not a wing. Run
    # back to the hip the same four tips spread down the whole trailing edge and
    # the sail reads as something with fingers in it.
    _qd_draw_wing(surf, xf, (0.32 * L, -1.00 * H), (-0.42 * L, -1.00 * H),
                  ((0.06 * L, -0.95 * H),), H,
                  rise, beat, s, mem, mem_spar, hide, isflat)


def _qd_dim_or(c: Color, f: float, isflat: bool) -> Color:
    return c if isflat else _dim(c, f)


def _qd_legs(surf: pygame.Surface, xf, H: float, L: float, s: float,
             col: Color, low: Color, t: float, near: bool) -> None:
    """Four legs. Front two-bone and columnar, hind a three-segment Z.

    The Z is the whole difference between a dragon crouched to spring and a
    lizard lying on its belly, and it survives to 0.5x when almost nothing
    else on the animal does.
    """
    off = 0.0 if near else -0.09 * L
    lift = 0.0 if near else -2.0 * s        # far feet a touch off the ground
    sh = 1.0 if near else 0.94
    w2 = max(1, int(round((2.6 if near else 2.0) * s)))
    w3 = max(1, int(round((2.2 if near else 1.7) * s)))
    shift = math.sin(t * 1.1) * 0.5 * s     # weight shift with the breath

    def limb(a: Pt, b: Pt, wa: float, wb: float) -> None:
        """An upper bone with real mass. Two lines and the leg is a wire."""
        u = _qd_unit(a, b)
        p = (-u[1], u[0])
        _poly(surf, col, [
            xf(a[0] + p[0] * wa, a[1] + p[1] * wa),
            xf(b[0] + p[0] * wb, b[1] + p[1] * wb),
            xf(b[0] - p[0] * wb, b[1] - p[1] * wb),
            xf(a[0] - p[0] * wa, a[1] - p[1] * wa)])

    # ---- front: shoulder, elbow back, wrist forward, foot planted ----------
    p0 = (0.31 * L + off, -0.86 * H * sh)
    p1 = (p0[0] - 0.09 * H, p0[1] + 0.40 * H * sh)
    p2 = (p1[0] + 0.16 * H, p1[1] + 0.31 * H * sh)
    p3 = (p2[0] + 0.11 * H + shift, lift)
    limb(p0, p1, 0.100 * H, 0.055 * H)
    _line(surf, col, xf(*p1), xf(*p2), w2)
    _line(surf, col, xf(*p2), xf(*p3), w3)
    _qd_claws(surf, xf, p3, s, col, +1.0)

    # ---- hind: femur forward, hock back, metatarsus forward ----------------
    q0 = (-0.28 * L + off, -0.92 * H * sh)
    q1 = (q0[0] + 0.32 * H, q0[1] + 0.30 * H * sh)
    q2 = (q1[0] - 0.30 * H, q1[1] + 0.32 * H * sh)
    q3 = (q2[0] + 0.26 * H - shift, lift)
    limb(q0, q1, 0.115 * H, 0.060 * H)
    _line(surf, col, xf(*q1), xf(*q2), w2 + 1)
    _line(surf, col, xf(*q2), xf(*q3), w3)
    _qd_claws(surf, xf, q3, s, col, +1.0)


def _qd_claws(surf: pygame.Surface, xf, foot: Pt, s: float, col: Color,
              face: float) -> None:
    w = max(1, int(round(1.6 * s)))
    for dx, dy in ((4.4, -0.6), (3.0, 0.4), (-2.6, -0.8)):
        _line(surf, col, xf(*foot),
              xf(foot[0] + dx * face * s, foot[1] + dy * s), w)


# --------------------------------------------------------------------------
# serpent - THE DEEP ONE. Wingless, so the head carries the whole read: it is
# ~3.3 body-thicknesses long on an arched neck, and nothing in here can be
# posed as flight. Anchored at the BASE OF THE SKULL; the body trails behind.
# --------------------------------------------------------------------------
def _sp_taper(surf: pygame.Surface, col: Sequence[int],
              a: tuple[float, float], b: tuple[float, float],
              w0: float, w1: float) -> None:
    """A tapered quad from *a* to *b*. Bones and horns, so they have mass."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = math.hypot(dx, dy)
    if d < 1e-4:
        return
    px, py = -dy / d, dx / d
    w0 = max(0.7, w0)
    w1 = max(0.5, w1)
    _poly(surf, col, ((a[0] + px * w0, a[1] + py * w0),
                      (b[0] + px * w1, b[1] + py * w1),
                      (b[0] - px * w1, b[1] - py * w1),
                      (a[0] - px * w0, a[1] - py * w0)))


def _sp_rot(v: tuple[float, float], a: float) -> tuple[float, float]:
    c, s = math.cos(a), math.sin(a)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


# ---------------------------------------------------------------- palette --
_sp_BODY: Color = (72, 128, 94)          # jade
_sp_RIM: Color = (132, 194, 138)         # lit edge along the back
_sp_SHADE: Color = (38, 74, 56)          # belly
_sp_LIMB: Color = (94, 152, 110)        # a step lighter, or the legs vanish
                                      # into the body they grow out of
_sp_CREST: Color = (222, 172, 74)        # mane, spines, horns, beard, tail flame
_sp_CREST_DK: Color = (152, 106, 40)
_sp_CLAW: Color = (204, 200, 182)
_sp_EYE: Color = (255, 226, 132)
_sp_MAW: Color = (108, 40, 44)           # inside the mouth
_sp_FACE: Color = (92, 152, 110)         # skull a shade off the neck, so the
                                      # head does not melt into the body

# ------------------------------------------------------------ proportions --
_sp_LEN = 172.0        # skull-to-tail spine length at scale 1.0 (a stickman is 26)
_sp_RAD = 5.0          # body half-thickness at the shoulders
_sp_HEADR = 11.4       # skull radius - deliberately >> _RAD
_sp_N = 34             # spine samples
_sp_HZ = 2.75          # rad/s of the travelling body wave

#: Half-thickness down the body as a fraction of ``_RAD``. A pinched neck, a
#: swell behind the shoulders, then a long taper to a whip.
_sp_PROFILE = ((0.00, 0.50), (0.05, 0.80), (0.14, 1.00), (0.32, 0.93),
               (0.50, 0.78), (0.68, 0.56), (0.82, 0.34), (0.92, 0.17),
               (1.00, 0.04))


def _sp_profile(s: float) -> float:
    if s <= 0.0:
        return _sp_PROFILE[0][1]
    for i in range(len(_sp_PROFILE) - 1):
        s0, v0 = _sp_PROFILE[i]
        s1, v1 = _sp_PROFILE[i + 1]
        if s <= s1:
            return v0 + (v1 - v0) * (s - s0) / (s1 - s0)
    return _sp_PROFILE[-1][1]


# -------------------------------------------------------------- the spine --
def _sp_spine(t: float, scale: float, facing: float):
    """Body-local centreline, tangents, back-normals and half-widths.

    Local space has the **skull base at the origin** and the body trailing
    toward -x, with -y up, exactly like the figure space in ``stickfigure.py``.
    The curve is built by integrating a heading angle: a heading wave coils
    into real serpent loops, where a displacement wave only ripples a straight
    line. The ``arch`` term drops the neck away from a raised head.
    """
    L = _sp_LEN * scale
    ds = L / _sp_N
    tilt = -0.16
    waves = 1.55
    om = t * _sp_HZ
    px, py = 0.0, 0.0
    pts = [(0.0, 0.0)]
    for i in range(_sp_N):
        s = (i + 0.5) / _sp_N
        # A *bump* of extra turn over the first third, zero at both ends, so
        # the neck arches down away from a level head and then straightens.
        # A ramp that is non-zero at s=0 does not arch the neck, it aims the
        # whole head at the sky.
        arch = -0.78 * math.sin(math.pi * min(1.0, s / 0.40))
        amp = 0.30 + 1.30 * s                          # tail whips, head steady
        th = math.pi + tilt + arch + amp * math.sin(s * waves * TAU - om)
        px += math.cos(th) * ds
        py += math.sin(th) * ds
        pts.append((px, py))

    tans: list[tuple[float, float]] = []
    for i in range(_sp_N + 1):
        a = pts[max(0, i - 1)]
        b = pts[min(_sp_N, i + 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy) or 1.0
        tans.append((dx / d, dy / d))
    # Normal points to the *back*, which for a body running to -x is up the
    # screen. It follows the coils, so the dorsal ridge rolls over the top of
    # one loop and under the next - free three-dimensionality.
    nrms = [(-ty, tx) for (tx, ty) in tans]
    rads = [max(0.4, _sp_RAD * scale * _sp_profile(i / _sp_N)) for i in range(_sp_N + 1)]

    if facing < 0:
        pts = [(-p[0], p[1]) for p in pts]
        tans = [(-v[0], v[1]) for v in tans]
        nrms = [(v[0], -v[1]) for v in nrms]
    return pts, tans, nrms, rads


# ----------------------------------------------------------------- pieces --
def _sp_spike(surf, col, p, tan, nrm, r, h, sweep, half):
    """One backswept triangular spine sitting on the dorsal edge."""
    bx = p[0] + nrm[0] * r * 0.80
    by = p[1] + nrm[1] * r * 0.80
    _poly(surf, col, (
        (bx - tan[0] * half, by - tan[1] * half),
        (bx + nrm[0] * h + tan[0] * h * sweep, by + nrm[1] * h + tan[1] * h * sweep),
        (bx + tan[0] * half, by + tan[1] * half),
    ))


def _sp_mane(surf, col, pts, tans, nrms, rads, i0, i1, hfn, sweep):
    """The neck mane as ONE filled, saw-edged band.

    A row of separate thin spikes reads as straw at wallpaper scale - the eye
    resolves each stroke and none of them add up. A solid mass with a notched
    outer edge keeps its shape all the way down to a handful of pixels, which
    is the whole point of the mane: it is the collar that says "head ends
    here, body starts here" when nothing else survives.
    """
    inner: list[tuple[float, float]] = []
    outer: list[tuple[float, float]] = []
    for i in range(i0, i1 + 1):
        p, tn, n, r = pts[i], tans[i], nrms[i], rads[i]
        base = (p[0] + n[0] * r * 0.55, p[1] + n[1] * r * 0.55)
        inner.append(base)
        h = hfn(i) * (0.66 if (i - i0) % 2 else 1.0)   # notch every other rib
        outer.append((base[0] + n[0] * h + tn[0] * h * sweep,
                      base[1] + n[1] * h + tn[1] * h * sweep))
    _poly(surf, col, outer + inner[::-1])


def _sp_limb(surf, p, tan, nrm, r, ph, body, claw, w):
    """A short clawed limb hanging off the belly.

    Both bones are filled tapers, not lines: at wallpaper scale a limb made of
    two 1-px strokes is not there at all, and the limbs are the entire
    "not a snake" argument.
    """
    bel = (-nrm[0], -nrm[1])
    sw = math.sin(ph)
    hip = (p[0] + bel[0] * r * 0.40, p[1] + bel[1] * r * 0.40)
    l1, l2 = r * 2.45, r * 2.00
    a1 = 0.62 + 0.40 * sw                  # from the belly normal, + = back
    knee = (hip[0] + (bel[0] * math.cos(a1) + tan[0] * math.sin(a1)) * l1,
            hip[1] + (bel[1] * math.cos(a1) + tan[1] * math.sin(a1)) * l1)
    a2 = a1 - 1.05 - 0.35 * sw
    foot = (knee[0] + (bel[0] * math.cos(a2) + tan[0] * math.sin(a2)) * l2,
            knee[1] + (bel[1] * math.cos(a2) + tan[1] * math.sin(a2)) * l2)

    _sp_taper(surf, body, hip, knee, r * 0.56, r * 0.34)
    _sp_taper(surf, body, knee, foot, r * 0.34, r * 0.22)
    dx, dy = foot[0] - knee[0], foot[1] - knee[1]
    d = math.hypot(dx, dy) or 1.0
    u = (dx / d, dy / d)
    for a in (-0.80, -0.05, 0.75):
        c = _sp_rot(u, a)
        _sp_taper(surf, claw, foot, (foot[0] + c[0] * r * 1.15,
                                     foot[1] + c[1] * r * 1.15), w * 0.36, w * 0.16)


def _sp_horn(surf, col, h, off, w):
    """One antler: a swept beam with two tines. ``off`` fakes the far one.

    Swept *steeply* up rather than back, so it clears the mane instead of
    fusing with it - two gold masses at the same angle read as one gold mass.
    """
    def q(a, b):
        return h(a + off, b)

    root, mid, tip = q(-0.35, 0.95), q(-0.95, 1.60), q(-1.55, 2.35)
    _sp_taper(surf, col, root, mid, w * 1.20, w * 0.80)
    _sp_taper(surf, col, mid, tip, w * 0.80, w * 0.30)
    _sp_taper(surf, col, q(-0.80, 1.40), q(-0.35, 2.10), w * 0.68, w * 0.26)


def _sp_head(surf, p, fwd, up, hr, t, body, rim, crest, crest_dk, claw, flat, w):
    """Cranium, muzzle, hinged jaw, brow horn, beard, antlers, whiskers, eye.

    Laid out in head-local units of the skull radius: ``a`` forward along the
    muzzle, ``b`` up toward the crest. Everything that carries the outline is
    a filled polygon, because a head assembled from strokes is the first thing
    to disappear when the creature gets small.

    The proportion that matters is depth-to-length. A profile 2.5x longer than
    it is deep is a newt whatever you hang off it; this one is ~1.7x, with a
    tall cranium at the back, a *stop* where the brow drops to the muzzle, and
    a blunt nose. That silhouette is doing more work than the antlers.
    """
    def h(a: float, b: float) -> tuple[float, float]:
        return (p[0] + fwd[0] * a * hr + up[0] * b * hr,
                p[1] + fwd[1] * a * hr + up[1] * b * hr)

    gape = 0.21 + 0.15 * (0.5 + 0.5 * math.sin(t * 1.6))
    hw = max(1.0, hr * 0.24)

    # Far antler, behind everything - but only when there is colour to tell
    # it from the near one. Flat, a second horn at a 3 px offset is not depth,
    # it is just a thicker blob where the head is supposed to be.
    if flat is None:
        _sp_horn(surf, _dim(crest, 0.58), h, -0.42, hw)

    # The maw. Lit, it is a dark wedge that turns the gape into a mouth.
    # Flat, it is deliberately NOT drawn: the sky between the jaws is worth
    # more to the silhouette than any shape could be. Same trick as the
    # stickmen, where the daylight under the head is what stops the figure
    # fusing into one blob.
    if flat is None:
        _poly(surf, _sp_MAW, (h(-0.80, -0.05), h(1.80, -0.24), h(1.55, -1.00)))

    # -- lower jaw, hinged at the back of the skull ------------------------
    jc, js = math.cos(gape), math.sin(gape)
    HA, HB = -0.95, -0.22

    def j(a: float, b: float) -> tuple[float, float]:
        la, lb = a - HA, b - HB
        return h(HA + la * jc + lb * js, HB + lb * jc - la * js)

    jawc = crest_dk if flat is not None else _dim(body, 0.80)
    _poly(surf, jawc, (j(-0.95, -0.12), j(1.62, -0.32), j(1.70, -0.76),
                       j(-0.05, -1.00), j(-1.05, -0.72)))
    if flat is None:                       # one fang on the lower jaw
        _poly(surf, claw, (j(1.32, -0.36), j(1.52, -0.36), j(1.44, -0.68)))
    # beard under the chin: gold, swept back and down. The single clearest
    # "not a lizard" mark on the animal, and it survives to a few pixels.
    _poly(surf, crest, (j(0.30, -0.82), j(-0.45, -0.98), j(-1.00, -1.62),
                        j(-0.15, -1.35), j(0.55, -1.05)))

    # -- cranium and muzzle ------------------------------------------------
    _disc(surf, body, h(-0.40, 0.34), hr * 0.78)
    _poly(surf, body, (
        h(-1.10, 0.20), h(-1.00, 0.80), h(-0.45, 1.12), h(0.35, 1.00),
        h(0.72, 0.62), h(1.45, 0.56), h(2.00, 0.40), h(2.16, 0.00),
        h(1.70, -0.26), h(0.35, -0.38), h(-0.60, -0.44), h(-1.08, -0.24),
    ))
    if flat is None:                       # two fangs on the upper lip
        for a0 in (0.72, 1.42):
            _poly(surf, claw, (h(a0, -0.28), h(a0 + 0.20, -0.28),
                               h(a0 + 0.10, 0.06)))
    # brow horn over the eye - the profile's whole attitude, one wedge
    _poly(surf, crest, (h(-0.05, 1.02), h(0.68, 0.78), h(0.30, 1.85)))

    if flat is None:
        _line(surf, rim, h(-1.02, 0.78), h(-0.45, 1.10), 1)
        _line(surf, rim, h(-0.45, 1.10), h(0.35, 0.98), 1)
        _line(surf, rim, h(0.72, 0.62), h(1.98, 0.40), 1)

    # near antler
    _sp_horn(surf, crest, h, 0.0, hw)

    # -- whiskers: long, trailing, one pixel. Flavour, not structure -------
    wc = crest if flat is not None else _sp_CREST
    for b0, off in ((-0.02, 0.0), (-0.26, 1.2)):
        prev = h(2.05, b0)
        for k in range(1, 8):
            u = k / 7.0
            nxt = h(2.05 - u * 4.4,
                    b0 - u * 0.85 + math.sin(u * 3.6 + t * 3.0 + off) * 1.15 * u)
            _line(surf, wc, prev, nxt, 1)
            prev = nxt

    if flat is None:
        e = h(-0.18, 0.42)
        er = max(1, int(round(hr * 0.26)))
        pygame.draw.rect(surf, _sp_EYE, pygame.Rect(int(e[0]), int(e[1]), er, er))
        n = h(1.86, 0.16)
        pygame.draw.rect(surf, _sp_SHADE,
                         pygame.Rect(int(n[0]), int(n[1]),
                                     max(1, er - 1), max(1, er - 1)))


def _sp_draw(surf: pygame.Surface, x: float, y: float, t: float, scale: float,
             flat: Sequence[int] | None, facing: int) -> None:
    if not (math.isfinite(x) and math.isfinite(y)) or scale <= 0.02:
        return
    # The spine wants a signed float, not a flag: it mirrors points, tangents
    # and normals by multiplying through.
    face = -1.0 if facing < 0 else 1.0
    pts, tans, nrms, rads = _sp_spine(t, scale, face)

    if flat is not None:
        f = (int(flat[0]), int(flat[1]), int(flat[2]))
        body = rim = shade = crest = crest_dk = claw = limb = f
    else:
        body, rim, shade, limb = _sp_BODY, _sp_RIM, _sp_SHADE, _sp_LIMB
        crest, crest_dk, claw = _sp_CREST, _sp_CREST_DK, _sp_CLAW
    far_body = body if flat is not None else _dim(body, 0.58)
    far_claw = claw if flat is not None else _dim(claw, 0.58)

    def W(p):
        return (x + p[0], y + p[1])

    w = max(1, int(round(1.6 * scale)))
    om = t * _sp_HZ

    # -- far limbs, behind the body ---------------------------------------
    for si, lag in ((0.25, 0.0), (0.57, 2.1)):
        i = int(round(si * _sp_N))
        _sp_limb(surf, W(pts[i]), tans[i], nrms[i], rads[i] * 0.86,
                 om + lag + 1.0, far_body, far_claw, max(1, w - 1))

    # -- tail flame: a solid notched fan, not a spray of needles -----------
    ti = _sp_N - 2
    tp, tt, tn = W(pts[ti]), tans[ti], nrms[ti]
    fl = _sp_RAD * scale * 3.6
    base = max(1.2, rads[ti] * 2.2)
    fan = [(tp[0] + tn[0] * base, tp[1] + tn[1] * base)]
    for a, ln in ((0.52, 0.62), (0.22, 1.25), (-0.06, 0.60), (-0.42, 1.00)):
        d = _sp_rot(tt, a)
        fan.append((tp[0] + d[0] * fl * ln, tp[1] + d[1] * fl * ln))
    fan.append((tp[0] - tn[0] * base, tp[1] - tn[1] * base))
    _poly(surf, crest, fan)

    # -- dorsal crest. Discrete and gappy: a continuous fin is an eel ------
    for i in range(14, _sp_N - 3, 4):
        r = rads[i]
        _sp_spike(surf, crest, W(pts[i]), tans[i], nrms[i], r,
                  r * 1.75, 0.50, r * 0.60)

    # -- the mane, rooted on the neck and sized off the *head* ------------
    hr = _sp_HEADR * scale
    wpts = [W(p) for p in pts]
    _sp_mane(surf, crest, wpts, tans, nrms, rads, 4, 13,
             lambda i: hr * (0.72 - 0.048 * (i - 4)), 0.40)

    # -- the body ribbon ---------------------------------------------------
    top, bot = [], []
    for i in range(_sp_N + 1):
        p, n, r = W(pts[i]), nrms[i], rads[i]
        top.append((p[0] + n[0] * r, p[1] + n[1] * r))
        bot.append((p[0] - n[0] * r, p[1] - n[1] * r))
    _poly(surf, body, top + bot[::-1])
    if flat is None:
        pygame.draw.aalines(surf, rim, False, top)
        pygame.draw.aalines(surf, shade, False, bot)
        for i in range(5, _sp_N - 5, 3):     # belly banding
            p, n, r = W(pts[i]), nrms[i], rads[i]
            _line(surf, shade, (p[0] - n[0] * r * 0.10, p[1] - n[1] * r * 0.10),
                  (p[0] - n[0] * r * 0.95, p[1] - n[1] * r * 0.95), 1)

    # -- near limbs, in front of the body ---------------------------------
    for si, lag in ((0.22, 0.0), (0.54, 2.1)):
        i = int(round(si * _sp_N))
        _sp_limb(surf, W(pts[i]), tans[i], nrms[i], rads[i], om + lag,
                 limb, claw, w)

    # -- head --------------------------------------------------------------
    # Nose carried a little above the line of the neck: a head aimed straight
    # down its own spine reads as a hose fitting, not as an animal looking
    # where it is going.
    k = 0.20 * face
    f0, u0 = (-tans[0][0], -tans[0][1]), nrms[0]
    ck, sk = math.cos(k), math.sin(k)
    fwd = (f0[0] * ck + u0[0] * sk, f0[1] * ck + u0[1] * sk)
    up = (u0[0] * ck - f0[0] * sk, u0[1] * ck - f0[1] * sk)
    hp = (W(pts[0])[0] + fwd[0] * hr * 0.42, W(pts[0])[1] + fwd[1] * hr * 0.42)
    face = body if flat is not None else _sp_FACE
    _sp_head(surf, hp, fwd, up, hr, t, face, rim, crest, crest_dk, claw, flat, w)


# --------------------------------------------------------------------------
# skeletal - THE REVENANT. A jointed armature in the stickfigure.py idiom:
# lines with discs at the joints, polygons only where a line cannot do the job
# (membrane, ridge, jaws, horns, fin). Anchored at the wing root, flight only.
# --------------------------------------------------------------------------
def _sk_pt(p: tuple[float, float], a: float, d: float) -> tuple[float, float]:
    """*d* along angle *a* from *p*. Angles are from +x, positive is **up**."""
    return (p[0] + math.cos(a) * d, p[1] - math.sin(a) * d)


# ------------------------------------------------------------- proportions --
# Every length is a fraction of S, the unit, which is one stickman height
# (26 px) times the caller's scale. A dragon is therefore always described in
# villagers, which is the only scale that matters in this game.
_sk_PITCH = 0.16         # the whole animal carries its nose up in flight
_sk_BODY = 0.66          # chest -> hip along the spine
_sk_BODY_DROP = 0.08     # ... and how much lower the hip sits
_sk_HAUNCH = 0.14        # shoulder / hip mass radius

# A long neck is not decoration: it is the only thing that gets the head clear
# of the wing mass, and a head that is inside the wing is a bat.
_sk_NECK_SEG = 0.30      # x4
_sk_NECK_A = (1.35, 1.05, 0.50, 0.10)    # rising arc, radians above horizontal
_sk_SKULL_R = 0.175
_sk_SNOUT = 0.44
_sk_JAW = 0.36
_sk_GAPE = 0.46          # how far the lower jaw hangs off the muzzle
_sk_HORN = 0.24          # the longer of two back-swept horns
# Ridge height over the neck / back / tail base. These have to beat the haunch
# discs and the spine line or the ridge is drawn and then buried: the first
# numbers here peaked 1.5 px above a 4.4 px disc and were simply invisible.
_sk_CREST = (0.085, 0.20, 0.17)
_sk_BACK_N = 3           # teeth between the shoulder and the hip

_sk_TAIL = 1.35          # total length over _TAIL_N segments
_sk_TAIL_N = 6
_sk_TAIL_CURL = 0.55     # how hard the tail hooks down before levelling out
_sk_FIN = 0.46

_sk_LEG_UP = 0.20        # hind: femur / tibia, tucked in flight
_sk_LEG_LO = 0.17
_sk_ARM_UP = 0.16        # fore: humerus / radius
_sk_ARM_LO = 0.15
_sk_CLAW = 0.055

# -- wing ------------------------------------------------------------------
# The membrane is the animal. Fingers stay long and the fan stays narrow, so
# the wing is a broad scalloped blade rather than a spread of umbrella spokes -
# the first version fanned to 82 degrees and read as a broken umbrella.
_sk_WING = 1.95          # scales the whole wing; fractions below are of _WING*S
_sk_W_HUM = 0.20         # short arm, long fingers: the arm is what swings the
_sk_W_FORE = 0.19        # wing out in FRONT of the head when it points forward
_sk_W_TIP = 0.68         # the leading finger, from the wrist
_sk_W_CLAW = 0.11

# The shoulder swings from level (bottom of the downstroke) to high and raked
# back (top of the upstroke), on a curve that keeps it low for most of the
# beat: high at mid-stroke put the wingtip straight through the head.
_sk_W_A1_LO = 0.05
_sk_W_A1_HI = 1.95
_sk_W_A1_SHAPE = 1.5
_sk_W_DWELL = 0.55       # <1 makes the beat linger at full up and full down
_sk_W_ELBOW = 0.42       # standing bend at the elbow
_sk_W_WRIST = 0.30       # ... and at the wrist
# Folding rotates the hand *backward*, toward the body. Subtracting it (the
# obvious sign, and the one v5 used) tucks the wing forward instead, which
# swings the folded wingtip out over the dragon's own face.
_sk_W_TUCK = (0.85, 0.55)
_sk_W_SPLAY = 0.34       # how far the last finger swings clear of the flank
_sk_W_REACH = 0.94       # ... and how far along the wrist->hip line it lands
_sk_W_LAG = 0.30         # how far the fingertips trail the beat
_sk_W_SAG = 0.17         # trailing-edge scallop depth
_sk_W_FAR = 0.84         # the far wing is smaller and folds a touch harder
_sk_W_HZ = 0.62          # beats per second

# ------------------------------------------------------------------ colour --
_sk_HIDE: Color = (112, 120, 110)
_sk_MEMB: Color = (66, 58, 62)
# Struts sit *just* off the membrane. Drawn in a bright bone colour they beat
# the membrane in the read and the wing turns into a fan of spokes.
_sk_BONE: Color = (92, 84, 86)
# Horn sits only a little off the hide. Near-white (198,192,174) it read as a
# feather crest no matter what shape it was cut to.
_sk_HORN_C: Color = (158, 152, 132)
_sk_EYE: Color = (232, 146, 52)
_sk_FAR_DIM = 0.62       # stickfigure._FAR_DIM, verbatim


# ------------------------------------------------------------------- rig --
def _sk_wing_rig(root: tuple[float, float], anchor: tuple[float, float],
                 S: float, p: float, size: float, fold_bias: float):
    """Solve one wing. Returns (elbow, wrist, tips, claw).

    The fan is **not** a fixed set of angles off the forearm. That was the
    first construction and it crumpled: half way through the beat the forearm
    pointed forward, the fingers fanned forward with it, and the membrane
    folded across the animal's own chest and throat. Here the leading finger
    takes the arm's direction and the *trailing* one is aimed at the hip, with
    the rest interpolated - so the membrane is always a clean sheet running
    from the wingtip home to the body, at every phase of the beat.
    """
    s0 = math.sin(p)                  # +1 top of the upstroke, -1 bottom
    # Linger at full up and full down, pass quickly through the middle. The
    # extremes are where a side-on wing has area; the middle is where it is
    # edge-on and the animal briefly stops being a dragon.
    sweep = math.copysign(abs(s0) ** _sk_W_DWELL, s0)
    vel = math.cos(p)
    fold = max(0.0, sweep) ** 1.4      # a wing folds on the *up* stroke only
    WL = _sk_WING * S * size

    a1 = _sk_W_A1_LO + (_sk_W_A1_HI - _sk_W_A1_LO) * (0.5 + 0.5 * sweep) ** _sk_W_A1_SHAPE
    elbow = _sk_pt(root, a1, WL * _sk_W_HUM)
    a2 = a1 - _sk_W_ELBOW + _sk_W_TUCK[0] * fold - fold_bias
    wrist = _sk_pt(elbow, a2, WL * _sk_W_FORE * (1.0 - 0.18 * fold))
    a_tip = a2 - _sk_W_WRIST + _sk_W_TUCK[1] * fold

    # Where the trailing edge has to come home, and the short way round to it.
    dx, dy = anchor[0] - wrist[0], anchor[1] - wrist[1]
    d_home = math.hypot(dx, dy)
    a_home = math.atan2(-dy, dx) - _sk_W_SPLAY
    while a_home - a_tip > math.pi:
        a_home -= TAU
    while a_tip - a_home > math.pi:
        a_home += TAU

    lag = -_sk_W_LAG * vel
    tip_len = WL * _sk_W_TIP * (1.0 - 0.20 * fold)
    tips = []
    for i in range(4):
        u = i / 3.0
        a = a_tip + (a_home - a_tip) * (u ** 0.85) + lag * (0.35 + 0.65 * u)
        ln = tip_len + (d_home * _sk_W_REACH - tip_len) * (u ** 1.35)
        tips.append(_sk_pt(wrist, a, max(ln, WL * 0.10)))
    claw = _sk_pt(wrist, a_tip + 0.85, WL * _sk_W_CLAW)
    return elbow, wrist, tips, claw


def _sk_scallop(a: tuple[float, float], b: tuple[float, float],
                toward: tuple[float, float], k: float) -> tuple[float, float]:
    mx, my = (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5
    return (mx + (toward[0] - mx) * k, my + (toward[1] - my) * k)


def _sk_draw_wing(surf, xf, root, S, p, size, fold_bias, hide, memb, bone, w):
    anchor = (-_sk_BODY * S * 0.86, _sk_BODY_DROP * S * 1.4)
    elbow, wrist, tips, claw = _sk_wing_rig(root, anchor, S, p, size, fold_bias)

    # membrane: leading edge out to the long finger, then the scalloped
    # trailing edge back down the fingers to the hip.
    pts = [root, elbow, wrist, tips[0]]
    for i in range(3):
        pts.append(_sk_scallop(tips[i], tips[i + 1], wrist, _sk_W_SAG))
        pts.append(tips[i + 1])
    pts.append(_sk_scallop(tips[3], anchor, wrist, _sk_W_SAG * 0.55))
    pts.append(anchor)
    _poly(surf, memb, [xf(*q) for q in pts])

    # Struts over the membrane, then the leading edge - tapering outward, the
    # same way a stickman's torso outweighs its arms. Drawn at a flat weight in
    # full hide colour the wingtip spar became a bright kite batten that beat
    # the membrane it was supposed to be holding up.
    for tp in tips[1:]:
        _line(surf, bone, xf(*wrist), xf(*tp), max(1, w - 1))
    _line(surf, bone, xf(*tips[3]), xf(*anchor), max(1, w - 1))
    spar = hide if hide is memb else _dim(hide, 0.82)
    _line(surf, spar, xf(*root), xf(*elbow), w + 1)
    _line(surf, spar, xf(*elbow), xf(*wrist), w)
    _line(surf, spar, xf(*wrist), xf(*tips[0]), max(1, w - 1))
    _line(surf, spar, xf(*wrist), xf(*claw), max(1, w - 1))
    _disc(surf, spar, xf(*elbow), max(1.0, S * 0.045))
    _disc(surf, spar, xf(*wrist), max(1.0, S * 0.040))


def _sk_crest(chain, heights):
    """A sawtooth ridge lying on *chain*, as one closed polygon.

    Eight separate 2 px spines vanish the moment the animal is scaled down;
    one polygon with a serrated top edge survives, and a serrated topline is
    most of the difference between a dragon and a large bat. *chain* runs from
    the head end backward, so the perpendicular below points up.
    """
    top = []
    for i in range(len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy)
        if n < 1e-6:
            continue
        px, py = -dy / n, dx / n
        h = heights[i]
        top.append(a)
        top.append(((a[0] + b[0]) * 0.5 + px * h, (a[1] + b[1]) * 0.5 + py * h))
    top.append(chain[-1])
    return top + list(reversed(chain))


def _sk_draw_leg(surf, xf, root, a_up, a_lo, l_up, l_lo, col, w, S):
    knee = _sk_pt(root, a_up, l_up)
    foot = _sk_pt(knee, a_lo, l_lo)
    _line(surf, col, xf(*root), xf(*knee), w)
    _line(surf, col, xf(*knee), xf(*foot), max(1, w - 1))
    for k in (-0.55, 0.0, 0.55):
        _line(surf, col, xf(*foot), xf(*_sk_pt(foot, a_lo + k, S * _sk_CLAW)), 1)


# ------------------------------------------------------------- main draw --
def _sk_draw(surf: pygame.Surface, x: float, y: float, t: float, scale: float,
             flat: Sequence[int] | None, facing: int) -> None:
    """Draw a flying dragon centred on its wing root at *(x, y)*.

    *t* is the render clock in seconds - the wings beat off it, and so do the
    tail wave, the neck bob and the body's rise and fall. *scale* is in
    villagers: 1.0 makes a dragon whose body is about the length of a stickman
    is tall. *flat*, if given, forces every colour to that one colour, which is
    how ``creatures.py`` draws anything standing outside every light source.

    The prototype had no *facing*: it hard-coded ``1.0`` and always flew right.
    It is a strict side view built in the ``stickfigure.py`` idiom, where
    mirroring is one multiply in ``xf`` and nothing else in the drawing has an
    opinion about it, so the parameter is a free addition here - which is not
    true of the flyover, whose whole camera model is asymmetric.
    """
    if not (math.isfinite(x) and math.isfinite(y)) or scale <= 0.0:
        return
    S = 26.0 * float(scale)
    mir = -1.0 if facing < 0 else 1.0
    p = float(t) * TAU * _sk_W_HZ

    if flat is not None:
        f = (int(flat[0]), int(flat[1]), int(flat[2]))
        hide = memb = bone = horn = f
        hide_far = memb_far = bone_far = f
    else:
        hide, memb, bone, horn = _sk_HIDE, _sk_MEMB, _sk_BONE, _sk_HORN_C
        hide_far = _dim(_sk_HIDE, _sk_FAR_DIM)
        memb_far = _dim(_sk_MEMB, _sk_FAR_DIM)
        bone_far = _dim(_sk_BONE, _sk_FAR_DIM)

    # The body rises through the downstroke and sinks through the upstroke, and
    # pitches nose-up as it does. Two numbers, and the animal stops looking
    # like a paper cut-out being waggled.
    bob = math.sin(p + 1.35) * S * 0.075
    pitch = _sk_PITCH + math.sin(p + 1.35) * 0.07
    cp, sp = math.cos(-pitch), math.sin(-pitch)

    def xf(lx: float, ly: float) -> tuple[float, float]:
        rx, ry = lx * cp - ly * sp, lx * sp + ly * cp
        return (x + rx * mir, y + ry + bob)

    w = max(1, int(round(S / 13.0)))          # limb weight: 2 at scale 1
    tw = w + 1                                # spine, one step heavier

    chest = (0.0, 0.0)
    hip = (-_sk_BODY * S, _sk_BODY_DROP * S)

    # -- solve the chains first, draw second ---------------------------------
    # Same split as stickfigure's build_skeleton / _draw, and it is what lets
    # the dorsal ridge be one polygon over joints the body has not drawn yet.
    nod = math.sin(p * 0.85 + 0.6) * 0.06
    neck = [chest]
    na = 0.0
    for i, a0 in enumerate(_sk_NECK_A):
        na = a0 + nod * (i + 1)
        neck.append(_sk_pt(neck[-1], na, _sk_NECK_SEG * S))
    skull = neck[-1]
    ha = na - 0.62                            # the head levels off

    seg = _sk_TAIL * S / _sk_TAIL_N
    tail = [hip]
    ta = math.pi
    for i in range(_sk_TAIL_N):
        u = (i + 0.5) / _sk_TAIL_N
        ta = (math.pi + _sk_TAIL_CURL * math.sin(math.pi * u)
              + 0.26 * math.sin(p * 0.85 - i * 0.62))
        tail.append(_sk_pt(tail[-1], ta, seg))

    # -- far wing, far legs: drawn first and dimmed, exactly as a villager's --
    _sk_draw_wing(surf, xf, (-S * 0.26, S * 0.10), S, p - 0.16, _sk_W_FAR, 0.20,
                  hide_far, memb_far, bone_far, w)
    _sk_draw_leg(surf, xf, (hip[0] + S * 0.06, hip[1] - S * 0.02),
                 math.pi + 0.90, math.pi + 0.12, _sk_LEG_UP * S * 0.92,
                 _sk_LEG_LO * S * 0.92, hide_far, w, S)
    _sk_draw_leg(surf, xf, (S * 0.06, S * 0.14), -1.30, -0.22,
                 _sk_ARM_UP * S * 0.92, _sk_ARM_LO * S * 0.92, hide_far, 1, S)

    # -- tail: a hooking, tapering chain, with a wave running down it --------
    # A straight tail is an arrow, and an arrow on the back of a flying thing
    # points harder than the head does - version one read as a spear with wings
    # bolted on. The curl is what turns it back into an animal.
    for i in range(_sk_TAIL_N):
        _line(surf, hide, xf(*tail[i]), xf(*tail[i + 1]),
              max(1, int(round((tw + 2) * (1.0 - 0.78 * i / (_sk_TAIL_N - 1))))))
    # Tail fin: a leaf lying *along* the last segment, widest near its root, so
    # it tapers to a point instead of reading as an arrowhead on a shaft.
    fin_l, fq = _sk_FIN * S, tail[-1]
    _poly(surf, memb if flat is None else hide, [
        xf(*_sk_pt(fq, ta + math.pi, fin_l * 0.34)),
        xf(*_sk_pt(_sk_pt(fq, ta, fin_l * 0.10), ta + 1.57, fin_l * 0.32)),
        xf(*_sk_pt(fq, ta, fin_l)),
        xf(*_sk_pt(_sk_pt(fq, ta, fin_l * 0.10), ta - 1.57, fin_l * 0.32)),
    ])

    # -- spine, and the two masses the limbs grow out of ---------------------
    _disc(surf, hide, xf(*hip), S * _sk_HAUNCH)
    _line(surf, hide, xf(*chest), xf(*hip), tw + 1)
    _disc(surf, hide, xf(*chest), S * _sk_HAUNCH * 1.10)

    # -- dorsal ridge: shoulders, back, and all the way down the tail --------
    # Drawn *after* the spine and the haunches, because it has to sit on top of
    # them to exist at all, and subdivided along the back so there is a row of
    # teeth rather than one hump.
    ridge = [(chest[0] + (hip[0] - chest[0]) * i / _sk_BACK_N,
              chest[1] + (hip[1] - chest[1]) * i / _sk_BACK_N)
             for i in range(_sk_BACK_N)] + tail
    hts = [_sk_CREST[1] * S] * _sk_BACK_N
    for i in range(_sk_TAIL_N):
        hts.append(_sk_CREST[2] * S * (1.0 - 0.95 * (i / (_sk_TAIL_N - 1)) ** 0.65))
    _poly(surf, hide, [xf(*q) for q in _sk_crest(ridge, hts)])

    # -- near legs: short and tucked. Long ones read as an insect's ----------
    _sk_draw_leg(surf, xf, hip, math.pi + 0.85, math.pi + 0.05,
                 _sk_LEG_UP * S, _sk_LEG_LO * S, hide, w, S)
    _sk_draw_leg(surf, xf, (S * 0.10, S * 0.12), -1.22, -0.14,
                 _sk_ARM_UP * S, _sk_ARM_LO * S, hide, max(1, w - 1), S)

    # -- near wing: the mass that carries the whole read ---------------------
    _sk_draw_wing(surf, xf, (-S * 0.10, -S * 0.04), S, p, 1.0, 0.0,
                  hide, memb, bone, w)

    # -- neck and head, last. -------------------------------------------------
    # Drawn *over* the near wing on purpose: strictly the near wing is closer,
    # but a head that disappears behind the membrane for half of every beat
    # costs more than the depth cue is worth, and the neck really is in front
    # of the wing along the body. This was the single worst read in v2.
    ridge_n = list(reversed(neck))
    _poly(surf, hide, [xf(*q) for q in
                       _sk_crest(ridge_n, [_sk_CREST[0] * S] * len(ridge_n))])
    for i, lw in enumerate((tw + 1, tw, tw, max(1, w))):
        _line(surf, hide, xf(*neck[i]), xf(*neck[i + 1]), lw)
    _disc(surf, hide, xf(*skull), S * _sk_SKULL_R)

    r = S * _sk_SKULL_R
    perp = (-math.sin(ha), -math.cos(ha))          # points *below* the muzzle
    nose = _sk_pt(skull, ha, _sk_SNOUT * S)
    # Upper jaw, blunt-ended. A muzzle that tapers to a single point is a beak;
    # squaring off the end by a quarter of a skull is what makes it a snout.
    _poly(surf, hide, [xf(*q) for q in [
        (skull[0] - perp[0] * r * 0.95, skull[1] - perp[1] * r * 0.95),
        (nose[0] - perp[0] * r * 0.30, nose[1] - perp[1] * r * 0.30),
        (nose[0] + perp[0] * r * 0.16, nose[1] + perp[1] * r * 0.16),
        (skull[0] + perp[0] * r * 0.25, skull[1] + perp[1] * r * 0.25),
    ]])
    # Lower jaw as its own wedge, hinged and hanging open. The V of daylight
    # between the two is what makes a 5 px head read as a head with a mouth.
    jaw_t = _sk_pt(skull, ha - _sk_GAPE, _sk_JAW * S)
    _poly(surf, hide, [xf(*q) for q in [
        (skull[0] + perp[0] * r * 0.45, skull[1] + perp[1] * r * 0.45),
        jaw_t,
        (skull[0] + perp[0] * r * 1.10, skull[1] + perp[1] * r * 1.10),
    ]])

    # Two horns swept off the back of the skull, as solid tapered wedges.
    # Drawn as pale *lines* (v2, three of them, 0.44 S long) they read as an
    # insect's antennae, and then as a cockatoo's crest when they were
    # shortened - a thin bright line on a skull is a feather. A wedge is horn.
    for k, ln in ((0.10, 1.0), (0.52, 0.70)):
        hb = _sk_pt(skull, ha + 2.15 + k, r * 0.72)
        ht = _sk_pt(hb, ha + 2.86 + k * 0.5, _sk_HORN * S * ln)
        hp = (-math.sin(ha + 2.86), -math.cos(ha + 2.86))
        _poly(surf, horn, [xf(*q) for q in [
            (hb[0] + hp[0] * r * 0.30, hb[1] + hp[1] * r * 0.30), ht,
            (hb[0] - hp[0] * r * 0.30, hb[1] - hp[1] * r * 0.30)]])
    if flat is None:
        e = xf(*_sk_pt(skull, ha + 0.42, r * 0.50))
        er = max(1, int(S * 0.05))
        pygame.draw.rect(surf, _sk_EYE, pygame.Rect(int(e[0]), int(e[1]), er, er))


# --------------------------------------------------------------------------
# the public face
#
# Five bodies, one door. Everything above this line is prototype art with a
# namespace on it; everything below is the part the renderer talks to.
# --------------------------------------------------------------------------

_DRAW: dict[str, Callable[..., None]] = {
    KIND_FLYOVER: _fo_draw,
    KIND_WYVERN: _wy_draw,
    KIND_QUADRUPED: _qd_draw,
    KIND_SERPENT: _sp_draw,
    KIND_SKELETAL: _sk_draw,
}

#: Roughly how far above the anchor the top of each drawing reaches, in
#: prototype units. Only two things use it and neither needs it to be exact:
#: where to hang the health pip, and where to sample the light (a creature's
#: mass is above its anchor, so sampling at the anchor puts a grounded dragon's
#: light test in the dirt).
_TOP: dict[str, float] = {
    KIND_FLYOVER: 14.0,
    KIND_WYVERN: 56.0,
    KIND_QUADRUPED: 92.0,
    KIND_SERPENT: 34.0,
    KIND_SKELETAL: 54.0,
}

#: States that mean "not in the world". Same list ``creatures._draw_ufo``
#: refuses on, for the same reason: a registry keeps its entry between visits.
_OFF_STATES = frozenset(("gone", "done", "idle", "off", "none", "waiting"))

#: Below this altitude the serpent's skull is under the dirt and there is
#: nothing to see. It is a skip, not a clip: clipping would want a scratch
#: surface every frame, and the honest failure - a serpent cresting a chasm
#: mouth overdraws a few px of the far wall on its way out - costs nothing and
#: happens inside a hole in the ground anyway.
_SUBMERGED = -6.0

_PIP_BACK: Color = (10, 11, 14)
_PIP_EDGE: Color = (54, 58, 68)
_PIP_FILL: Color = (206, 74, 64)
_PIP_HOLLOW: Color = (120, 126, 140)


def dragon_kind_of(d: Any) -> str:
    """The drawing to use for *d*. Anything unrecognised is a wyvern.

    A wyvern rather than a raise: an unknown kind means the sim grew a sixth
    dragon or renamed one, and a frame with the wrong dragon in it is a better
    failure than a frame with a traceback in it.
    """
    for name in ("kind", "species", "type"):
        try:
            val = getattr(d, name, None)
        except Exception:
            continue
        if isinstance(val, str) and val in _DRAW:
            return val
    return KIND_WYVERN


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


def draw_dragon(surf: pygame.Surface, x: float, y: float, t: float, *,
                kind: str = KIND_WYVERN, scale: float = 1.0,
                flat: Sequence[int] | None = None, facing: int = 1) -> None:
    """Draw one dragon of *kind* with its anchor at ``(x, y)``.

    ``(x, y)`` is whatever :data:`ANCHOR` says it is for that kind - the feet
    for the quadruped, the wing root for the wyvern and the skeletal, the base
    of the skull for the serpent, the body centre for the flyover.

    *scale* is the **prototype** scale, not the game scale: 1.0 draws the
    creature at the size its own prototype called 1.0, which is a different
    size for each of the five. The sim-facing path applies :data:`SCALE` on
    top; a caller that wants a dragon the size the game draws it wants
    ``scale=SCALE[kind]``.

    *t* is the render clock in seconds. Every cycle in every body - wingbeats,
    the mantle, the body wave, jaws, whiskers - comes off it, so two dragons
    drawn at the same *t* are in step and the caller offsets *t* to break that
    up.

    *flat*, if given, forces every colour to that one colour: the silhouette
    path ``creatures._draw_animal`` uses for anything outside a light source.

    Fails soft. This is the one try/except for the whole module - the bodies
    below do not each carry their own, so there is exactly one place a dragon
    can swallow a frame and it is this one.
    """
    fn = _DRAW.get(kind)
    if fn is None:
        fn = _DRAW[KIND_WYVERN]
    try:
        xf, yf, sf = float(x), float(y), float(scale)
        if not (math.isfinite(xf) and math.isfinite(yf) and math.isfinite(sf)):
            return
        if sf <= 0.02:
            return
        tf = float(t)
        if not math.isfinite(tf):
            tf = 0.0
        fn(surf, xf, yf, tf, sf, flat, -1 if facing < 0 else 1)
    except Exception:
        log.debug("dragon draw failed (%s)", kind, exc_info=True)


def draw_one(surf: pygame.Surface, world: Any, d: Any, t: float) -> None:
    """Draw one *sim* dragon: the entry point per dragon.

    Reads ``kind``, ``x``, ``alt``, ``facing``, ``state``, ``id``, ``health``,
    ``max_health`` and ``sated`` off *d*, all duck-typed and all optional.
    Nothing is written back.

    Placement is the whole of the altitude contract as far as render is
    concerned: ``y = ground_y(x) - alt`` for four of the five, and
    ``y = ground_y(x)`` for the quadruped, whose origin is already the ground
    under its feet.
    """
    kind = dragon_kind_of(d)
    state = str(getattr(d, "state", "") or "").lower()
    if state in _OFF_STATES:
        return

    x = fx.attr_num(d, "x", default=math.nan)
    if not math.isfinite(x) or x < -_CULL or x > RENDER_W + _CULL:
        return
    alt = fx.attr_num(d, "alt", default=0.0)

    # The quadruped has no flight pose at all - `t` drives a *grounded* mantle
    # cycle - so there is nothing correct to draw for it in the air. Render
    # pins it to the ground for its whole visit and the sim is expected to keep
    # the arrive and leave legs off the frame edge, which is what the animals
    # already do. The alternative was a visibly wrong pose 100 px up.
    ground = _ground_y(world, x)
    y = ground if kind == KIND_QUADRUPED else ground - alt
    if y < -_CULL or y > RENDER_H + _CULL:
        return
    if kind == KIND_SERPENT and alt < _SUBMERGED:
        return

    scale = SCALE.get(kind, 1.0)
    top = _TOP.get(kind, 40.0) * scale

    # No usable `facing` falls back to the direction of travel, the same
    # fallback creatures._draw_animal makes so nothing ever moons the colony
    # while walking at it.
    face = fx.attr_num(d, "facing", default=0.0)
    if face == 0.0:
        face = fx.attr_num(d, "vx", "dx", default=1.0)
    facing = -1 if face < 0.0 else 1

    # Silhouette rule, same as the animals: below _SIL_CUTOFF of *source*
    # light the whole creature is one flat colour. light_at excludes ambient,
    # so anything more than a torch radius off the ground is flat by
    # definition - which for the flyover at alt 300-420 is every frame it is
    # ever on screen.
    lit = _light_at(world, x, y - top * 0.45)
    flat: Color | None = _SIL_COLOR if lit < _SIL_CUTOFF else None

    # Break the lockstep. Two dragons on screen at once is not a thing the sim
    # allows today (MAX_DRAGONS_ALIVE is 1), but a debug tray can summon a row
    # of them and five identical wingbeats is a chorus line.
    anim = fx.attr_num(d, "anim_t", default=-1.0)
    if anim < 0.0:
        anim = t
    anim += (fx.attr_num(d, "id", default=0.0) * 0.6180339) % 4.0

    draw_dragon(surf, x, y, anim, kind=kind, scale=scale, flat=flat,
                facing=facing)
    _health_pip(surf, d, kind, x, y - top - 7.0, 0.35 if flat else 1.0)


def _health_pip(surf: pygame.Surface, d: Any, kind: str, cx: float, cy: float,
                bright: float) -> None:
    """The wound bar, and the one place the *feeding rule* is drawn.

    An unsated dragon discards all damage, so its health never moves, so the
    ``health < max`` test every animal uses draws nothing at all - which reads
    as "the bar is missing" rather than as "it cannot be hurt". So while it is
    unsated the bar is drawn **hollow at full width**: an untouched bar you can
    see is what makes the refusal read as a property of the monster rather than
    as a rendering bug. Presentation only; the rule is complete without it.

    The flyover is exempt. It is above THROW_MAX_RANGE and cannot be hurt by
    anything, so a bar over it is a promise the game does not keep.
    """
    if kind == KIND_FLYOVER:
        return
    maxh = fx.attr_num(d, "max_health", "maxhp", default=0.0)
    if maxh <= 0.0:
        return
    health = fx.attr_num(d, "health", "hp", default=maxh)
    sated = bool(getattr(d, "sated", True))
    hurt = health < maxh * 0.995
    if sated and not hurt:
        return

    w, h = 18, 3
    x0, y0 = int(cx - w * 0.5), int(cy)
    pygame.draw.rect(surf, _dim(_PIP_BACK, bright),
                     pygame.Rect(x0 - 1, y0 - 1, w + 2, h + 2))
    pygame.draw.rect(surf, _dim(_PIP_EDGE, bright), pygame.Rect(x0, y0, w, h))
    if not sated:
        pygame.draw.rect(surf, _dim(_PIP_HOLLOW, bright),
                         pygame.Rect(x0, y0, w, h), 1)
        return
    fill = int(round(w * fx.clamp(health / maxh, 0.0, 1.0)))
    if fill > 0:
        pygame.draw.rect(surf, _dim(_PIP_FILL, bright),
                         pygame.Rect(x0, y0, fill, h))


def _iter_dragons(world: Any) -> list[Any]:
    """The dragon list, or an empty one.

    The fast path is the only one that matters: there is no dragon on screen
    for the overwhelming majority of a run, and this is a per-frame call. One
    ``getattr`` and one ``len`` and we are out.
    """
    reg = getattr(world, "dragons", None)
    if reg is None:
        return []
    try:
        if len(reg) == 0:
            return []
    except TypeError:
        pass
    if not isinstance(reg, (list, tuple)):
        for meth in ("alive", "all", "values"):
            fn = getattr(reg, meth, None)
            if callable(fn):
                try:
                    reg = fn()
                    break
                except Exception:
                    continue
    try:
        return [d for d in reg if d is not None]
    except TypeError:
        return []


def draw_dragons(surf: pygame.Surface, world: Any, t: float) -> None:
    """Draw every dragon in *world*. Fails soft, per dragon and overall.

    Call this **after** the light composite, beside ``creatures.draw_ufo``: a
    dragon at altitude is above every light source, and a night composite would
    multiply it into a black smear on black.
    """
    try:
        dragons = _iter_dragons(world)
    except Exception:
        log.exception("could not read the dragon list")
        return
    if not dragons:
        return
    if len(dragons) > 1:
        try:
            dragons = sorted(dragons,
                             key=lambda d: fx.attr_num(d, "y", default=0.0))
        except Exception:
            pass
    # The clock is coerced once, here, rather than five times inside the
    # bodies: renderer passes world.world_time and a world that has just been
    # loaded from a junk save can hand over anything at all.
    try:
        tf = float(t)
    except (TypeError, ValueError):
        tf = 0.0
    if not math.isfinite(tf):
        tf = 0.0
    for d in dragons:
        try:
            draw_one(surf, world, d, tf)
        except Exception:
            log.debug("dragon draw failed", exc_info=True)
