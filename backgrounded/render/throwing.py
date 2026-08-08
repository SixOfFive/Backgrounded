"""Thrown spears, drawn.

``sim/throwing.py`` has modelled a real projectile since it landed - x, y, vx,
vy, its own gravity, a landed state that can be walked to and picked up - and
none of it was ever on screen. A hostile proof over 16 colonies measured **230
throws that drew literally nothing**: no shaft in the air, no shaft in the dirt,
no aftermath. The mechanic existed, was tuned, wrote chronicle lines about
wolves it had killed, and was invisible. This module is the missing half.

Two states, two drawings
------------------------
* **in flight** - the shaft is aimed down the velocity vector, so it noses over
  at the top of the arc for free. No render-clock animation is involved: the
  pose is a pure function of sim state, which is why a paused world's spears
  hang still instead of twitching.
* **on the ground** - planted, tip in the dirt, shaft leaning back. The sim
  zeroes ``vx``/``vy`` on landing (``SpearRegistry._land``), so the impact
  heading is *not recoverable* and the lean has to come from somewhere else -
  see :func:`_ground_pose`.

The first cut laid the grounded spear flat along the terrain, which is what
"lands in the dirt" sounds like, and it lost the A/B (seed 7, noon, five spears
at 6x): the two variants cost the same ink - 381 changed pixels lying against
363 planted - but a 25 px horizontal bar runs *along* the ground contour and
reads as a fallen branch, of which this world already draws plenty. Planted, it
crosses the ground line at 62 degrees and is unmistakably a weapon somebody
threw. Same pixels, one of them legible.

Anchor: **(x, y) is the TIP**, not the centre
---------------------------------------------
Because that is the point the sim tests. ``SpearRegistry._hit_check`` asks
whether ``(s.x, s.y)`` is inside the target box, so drawing the tip anywhere
else would put the spear through a wolf's shoulder on a frame the sim scored as
a hit through its chest. It also makes the grounded pose fall out for free:
``_land`` writes ``y = ground_y(x)``, so the tip is already exactly on the
ground and the shaft is the only thing that has to be placed.

The one cost is at release. ``launch_point`` puts the spear at the thrower's
shoulder, so for the first frame or two the butt trails back *through* the
thrower's torso. At THROW_SPEED that is ~10 px a frame, so it is clear of a
26 px body in two, and it reads as a release rather than as an error. The
obvious fix - grow the shaft over the first 0.1 s - is not worth it: it trades
a two-frame overlap for a spear that visibly stretches, on every single throw.

House idiom
-----------
The art is ``creatures._draw_spear`` with the arm taken out: same length, same
proportions, same knapped-triangle head, same palette, same grip band at the
same fraction of the shaft - a spear in the air has to be recognisably the
object that just left a hand. Those five numbers and three colours are
*imported* from ``creatures`` rather than copied, so one repaint moves both;
the literal fallback below exists only so a rename over there degrades this
module to a colour drift instead of an ImportError that takes the renderer
down with it.

Flat-silhouette support is the same rule every creature in this package keeps
(``creatures._draw_animal``, ``dragons.draw_one``): below ``_SIL_CUTOFF`` of
*source* light the whole object collapses to one flat ``_SIL_COLOR`` shape. A
spear thrown across a dark field is a shape, not a woodgrain.

Layering and cost
-----------------
Belongs at **step 7**, with the animals and *before* the light composite -
unlike the dragons. A thrown spear lives in the same 0-140 px band as the
wolves it is thrown at, so it must be lit by the same torches; drawn after the
composite it would be the one object in the colony that glows at 3am, which is
the artefact ``renderer._draw_dragons`` measured at 5x on the siege dragon.
``renderer.draw`` already carries the call, immediately after the agents and
the animals - a 2 px shaft drawn *behind* a body is simply eaten by it.

**Read-only w.r.t. sim state** - nothing here writes to a World or a Spear -
and fail-soft twice over, per spear and overall, because this is a frame path.
Measured on seed 7 against a ~7.3 ms full frame, over several runs: **0.2 us**
with an empty registry - two ``getattr`` calls and a return - ~50 us with eight
in the air (~6 us each) and ~85-95 us with eight in the dirt (~11 us each).
The dirt costs more than the air because a grounded spear re-reads
``terrain.ground_y``; see :func:`_ground_y` for why it has to. Roughly 0.7% of
a frame for a full sky of spears, and it is the *blit* that dominates (~43 us
of the 50), not the pose maths - there is nothing here worth caching.

Note which case is which. Spears *in flight* are genuinely rare - a throw is
one action with a 1.6 s cooldown - but the registry is **not** usually empty,
because a spear nobody went back for lies there for SPEAR_LIFE_SEC (600 s).
The steady-state cost of this module is therefore the grounded one, and
MAX_SPEARS = 12 caps the whole thing at ~120 us, or 1.6% of a frame.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import pygame

from ..constants import RENDER_H
from ..sim.entities import AGENT_HEIGHT
from . import fx
from .camera import IDENTITY, Camera

log = logging.getLogger(__name__)

__all__ = ["draw_spears", "draw_one", "draw_spear", "spear_pose"]

TAU = math.tau
Color = tuple[int, int, int]

# Mirrors renderer.SILHOUETTE_CUTOFF / SILHOUETTE_COLOR, copied for the same
# reason creatures.py copies them: renderer imports this package and a cycle
# would be paid at startup for two scalars.
_SIL_CUTOFF = 0.30
_SIL_COLOR: Color = (14, 15, 22)

# The palette and the primitives, from the module that draws the carried spear.
# NOT a copy: a thrown spear that is a different brown from the one in the
# thrower's hand reads as a different object, and that is exactly the bug a
# duplicated constant produces six months later.
try:
    from .creatures import (            # noqa: PLC2701 - deliberate, see above
        _SHAFT,
        _SHAFT_DARK,
        _SPEAR_HEAD,
        _dim,
        _line,
        _lit,
        _poly,
    )
except Exception:                                     # pragma: no cover
    # Renames over there must not take the renderer down at import time.
    log.warning("creatures.py did not yield the spear palette; using a copy",
                exc_info=True)
    _SHAFT = (146, 110, 70)
    _SHAFT_DARK = (78, 56, 34)
    _SPEAR_HEAD = (162, 166, 176)

    def _dim(col: Sequence[int], f: float) -> Color:
        return (int(col[0] * f), int(col[1] * f), int(col[2] * f))

    def _lit(col: Sequence[int], f: float) -> Color:
        return (
            int(min(255.0, col[0] + (255 - col[0]) * f)),
            int(min(255.0, col[1] + (255 - col[1]) * f)),
            int(min(255.0, col[2] + (255 - col[2]) * f)),
        )

    def _line(surf: pygame.Surface, col: Sequence[int],
              a: tuple[float, float], b: tuple[float, float], w: int = 1) -> None:
        try:
            pygame.draw.aaline(surf, col, a, b, w)
        except TypeError:                             # pragma: no cover
            pygame.draw.aaline(surf, col, a, b)

    def _poly(surf: pygame.Surface, col: Sequence[int],
              pts: Sequence[tuple[float, float]]) -> None:
        if len(pts) >= 3:
            pygame.draw.polygon(
                surf, col, [(int(round(p[0])), int(round(p[1]))) for p in pts])

# ---------------------------------------------------------------- geometry --
# All of it derived from creatures._draw_spear so the two cannot drift:
#   ln   = height * 0.98            the shaft it lays out
#   butt = hand - u * ln * 0.26     ... measured from the fist
#   tip  = hand + u * ln * 0.72
# so a whole spear is ln * 0.98 long and the grip sits 0.26/0.98 of the way up
# from the butt. A thrown spear is not attached to a skeleton, so `height` is
# the nominal AGENT_HEIGHT rather than a per-agent one - a child's spear is the
# same stick as an adult's once it has left the hand.
_LEN = AGENT_HEIGHT * 0.98 * 0.98         # 24.97 px
_HEAD_LEN = AGENT_HEIGHT * 0.16           # 4.16 px of knapped head
_HEAD_HALF_W = 1.5
_GRIP_AT = 0.26 / 0.98                    # fraction of _LEN, from the butt
_GRIP_HALF = 1.6
_SHAFT_W = 2                              # `2 if h >= 22.0 else 1`, and h is 26

#: How far the silhouette rim is lifted toward white. Same figure the dragon
#: flyover uses (`dragons._lit(flat, 0.20)`) and for the same reason.
#:
#: ``_SIL_COLOR`` is tuned to read against the SKY, and it does - 148 luma of
#: contrast there. The parallax ridge is dark by design, and over the three
#: darkest scenes the shaft was DARKER THAN THE BAND IT CROSSED: measured
#: against the flat ridge tones, night_storm gave 4.7 luma of contrast, aurora
#: 5.5 and meteor 7.2, where about 10 is the floor for reading a 25 px shape.
#: So the spear was drawn in full - 80 px of ink at every angle sampled - and
#: could not be seen. A value shift on ``_SIL_COLOR`` itself was the other
#: option and was not taken: it would lift every silhouette in the game to fix
#: one object over one band.
_SIL_RIM = 0.20

#: Grown by one pixel each side, so the rim survives only as an edge once the
#: flat shape is painted over it.
_RIM_GROW = 1.0

#: How far a spear may sit off-screen before it stops being drawn.
#: creatures._draw_animal uses 80 px for animals; a spear is 25 px long, so 60
#: is already more than twice what it takes for the shaft to be fully outside
#: the frame, and the check is two comparisons.
_CULL = 60.0

#: Angle below horizontal of a planted spear, rad. 62 degrees: shallow enough
#: to read as "thrown and stuck" rather than "ceremonially placed", steep enough
#: that the butt clears the terrain noise by ~22 px.
_PLANT_ANG = math.radians(62.0)
#: Per-spear lean spread, rad, so a cluster of misses is debris rather than a
#: comb of parallel strokes. Small on purpose: past ~20 degrees the shallow end
#: starts reading as a stick lying down, which is the pose the A/B rejected.
_PLANT_SPREAD = math.radians(13.0)

#: A grounded spear is dulled slightly - it is half in the dirt and it is
#: *not* the thing the player should be watching. In flight it is.
_GROUND_FADE = 0.90


# ------------------------------------------------------------- adapters --
def _ground_y(world: Any, x: float, fallback: float) -> float:
    """Terrain height under *x*, or *fallback*.

    A grounded spear already carries the y the sim wrote when it landed, so
    this looks redundant - it is not. Terrain *deforms* after the fact
    (earthquake fissures, the chasm crossings), and the sim never revisits a
    spear once it has come to rest, so the stored y goes stale and a spear ends
    up planted in mid-air over a new fissure. Re-reading the ground each frame
    is one call per grounded spear (<=12) and is the only way render can fix
    this without writing back into sim state, which it is not allowed to do.
    """
    terrain = getattr(world, "terrain", None)
    fn = getattr(terrain, "ground_y", None)
    if not callable(fn):
        return fallback
    try:
        gy = float(fn(x))
    except Exception:
        return fallback
    return gy if math.isfinite(gy) else fallback


def _light_at(world: Any, x: float, y: float) -> float:
    try:
        return float(world.lighting.light_at(x, y))
    except Exception:
        return 1.0


def _iter_spears(world: Any) -> list[Any]:
    """Every spear the world is holding, however it is holding it.

    Fast path first and on purpose: ``SpearRegistry.spears`` is the live list,
    so the common "nothing has been thrown" case is two attribute reads and a
    falsy test - no method call, no list copy, no allocation. ``.all()`` would
    be the polite API and it builds a list every frame forever to tell us it is
    empty.
    """
    reg = getattr(world, "spears", None)
    if reg is None:
        return []
    items = getattr(reg, "spears", None)
    if items is None:
        if isinstance(reg, (list, tuple)):
            items = reg
        else:
            for meth in ("all", "values"):
                fn = getattr(reg, meth, None)
                if callable(fn):
                    try:
                        items = fn()
                        break
                    except Exception:
                        continue
    if not items:
        return []
    try:
        return [s for s in items if s is not None]
    except TypeError:
        return []


# -------------------------------------------------------------- one spear --
def spear_pose(spear: Any, world: Any = None) -> tuple[float, float, float] | None:
    """``(tip_x, tip_y, angle)`` for *spear* in WORLD space, or None.

    Split out from the drawing so the geometry can be asserted in a test
    without a Surface, and so the flight/ground decision lives in exactly one
    place. Duck-typed throughout - ``state`` may be missing entirely, in which
    case a spear with velocity is flying and one without is lying down, which
    is the same answer the sim would give.

    The horizontal cull used to live here and it was wrong twice over: it
    compared a WORLD x against RENDER_W, which discards everything past the
    first quarter of a 6400 px map, and it made a geometry function depend on
    where the camera happened to be looking. It now lives in :func:`draw_one`,
    in screen space, which is the only place that knows about a frame.
    """
    x = fx.attr_num(spear, "x", default=math.nan)
    y = fx.attr_num(spear, "y", default=math.nan)
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if y < -_CULL or y > RENDER_H + _CULL:
        return None

    vx = fx.attr_num(spear, "vx", default=0.0)
    vy = fx.attr_num(spear, "vy", default=0.0)
    state = str(getattr(spear, "state", "") or "")
    flying = (state == "flight") if state else (vx or vy)

    if flying and (vx or vy):
        return (x, y, math.atan2(vy, vx))
    return _ground_pose(spear, x, y, world)


def _ground_pose(spear: Any, x: float, y: float,
                 world: Any) -> tuple[float, float, float]:
    """A planted spear: tip on the terrain, shaft leaning one way or the other.

    The lean is derived from ``spear.id`` - deliberately *not* drawn from a
    random stream. Two reasons, and the first is a hard rule: every random
    number in this game comes from a seeded stream, and render has no stream of
    its own to draw from. The second is that it has to be *stable*. A lean
    re-rolled per frame is a spear vibrating in the dirt, and one re-rolled per
    load is a spear that moves when you reopen the game; a pure function of the
    persisted id is identical on every frame and across every reload, for free.

    ``id * 0.9137`` is the same phase-decorrelating trick ``creatures``
    uses for animal gait (``idp``), reused so consecutive ids do not land on
    consecutive leans.
    """
    idp = (fx.attr_num(spear, "id", default=0.0) * 0.9137) % TAU
    side = 1.0 if math.sin(idp) >= 0.0 else -1.0
    ang = _PLANT_ANG + math.cos(idp) * _PLANT_SPREAD
    gy = _ground_y(world, x, y) if world is not None else y
    # +y is down, so the heading (butt -> tip) points down into the ground.
    return (x, gy, math.atan2(math.sin(ang), math.cos(ang) * side))


def draw_spear(surf: pygame.Surface, x: float, y: float, angle: float, *,
               flat: Color | None = None, fade: float = 1.0,
               length: float = _LEN) -> None:
    """One spear, tip at ``(x, y)``, shaft trailing back along *angle*.

    *flat* is the silhouette colour: given one, every part is drawn in it and
    the interior detail (the grip band) is skipped, exactly as
    ``creatures._draw_spear`` does it. Public because a contact sheet and a
    debug tray both want to draw a spear that no sim object exists for.
    """
    ux, uy = math.cos(angle), math.sin(angle)
    tip = (x, y)
    butt = (x - ux * length, y - uy * length)

    shaft = flat if flat is not None else _dim(_SHAFT, fade)
    head = flat if flat is not None else _dim(_SPEAR_HEAD, fade)
    grip = flat if flat is not None else _dim(_SHAFT_DARK, fade)

    px, py = -uy, ux
    base = (x - ux * _HEAD_LEN, y - uy * _HEAD_LEN)

    if flat is not None:
        # Rim first, one pixel proud all round, then the silhouette over it -
        # so what is left is an edge rather than a lighter spear. See _SIL_RIM:
        # without this the shaft is invisible against the three darkest ridges,
        # which is where a spear thrown at a dragon usually is.
        rim = _lit(flat, _SIL_RIM)
        _line(surf, rim, butt, tip, _SHAFT_W + int(_RIM_GROW * 2))
        _poly(surf, rim, [
            (base[0] + px * (_HEAD_HALF_W + _RIM_GROW),
             base[1] + py * (_HEAD_HALF_W + _RIM_GROW)),
            (tip[0] + ux * _RIM_GROW, tip[1] + uy * _RIM_GROW),
            (base[0] - px * (_HEAD_HALF_W + _RIM_GROW),
             base[1] - py * (_HEAD_HALF_W + _RIM_GROW)),
        ])

    _line(surf, shaft, butt, tip, _SHAFT_W)
    # Knapped head: a small triangle rather than a blob, so the pointy end is
    # obvious at 25 px - and on a thrown spear it is the only cue for which end
    # is which, since there is no hand to read it from.
    _poly(surf, head, [
        (base[0] + px * _HEAD_HALF_W, base[1] + py * _HEAD_HALF_W),
        tip,
        (base[0] - px * _HEAD_HALF_W, base[1] - py * _HEAD_HALF_W),
    ])
    if flat is None:
        gx = butt[0] + ux * length * _GRIP_AT
        gy = butt[1] + uy * length * _GRIP_AT
        _line(surf, grip, (gx - ux * _GRIP_HALF, gy - uy * _GRIP_HALF),
              (gx + ux * _GRIP_HALF, gy + uy * _GRIP_HALF), _SHAFT_W)


def draw_one(surf: pygame.Surface, spear: Any, world: Any, *,
             cam: Camera) -> None:
    """Draw a single sim spear, lit or flat. Raises nothing the caller minds.

    *cam* is REQUIRED and has no default: a spear's ``x`` is a world x on a
    6400 px map against a 1600 px frame. sim/throwing.py modelled a real
    projectile for 230 throws before anything drew it at all; drawing it at
    world coordinates is the same failure wearing a call site.
    """
    pose = spear_pose(spear, world)
    if pose is None:
        return
    wx, y, ang = pose
    # The cull the geometry function used to do, now in screen space and with
    # the shaft's own length as the pad so a spear does not pop in at the edge.
    if not cam.visible(wx, _CULL):
        return
    grounded = str(getattr(spear, "state", "") or "") != "flight"
    # Sample the light at the middle of the shaft, not at the tip. The object
    # is 25 px long and the silhouette test is a single point, so the sample
    # has to be the least wrong one available: a planted spear at the edge of a
    # torch pool has its tip in the dark and its butt in the light, and picking
    # the tip would flip the whole thing flat while most of it is lit.
    # creatures._draw_animal does exactly this and for exactly this reason - it
    # samples `y - height * 0.5`, the middle of the body, not the feet.
    # WORLD x for the light sample - lighting.light_at reads the sim's own
    # sources - and screen x for the drawing. The one conversion is on the
    # draw_spear line.
    lit = _light_at(world, wx - math.cos(ang) * _LEN * 0.5,
                    y - math.sin(ang) * _LEN * 0.5)
    flat: Color | None = _SIL_COLOR if lit < _SIL_CUTOFF else None
    draw_spear(surf, cam.sx(wx), y, ang, flat=flat,
               fade=_GROUND_FADE if grounded else 1.0)


def draw_spears(surf: pygame.Surface, world: Any, t: float = 0.0, *,
                cam: Camera) -> None:
    """Draw every spear in *world*. Fails soft, per spear and overall.

    *t* is accepted and ignored, so the call site matches
    ``draw_animals(s, world, world.world_time)`` and its neighbours. There is
    nothing for a clock to drive here: both poses are pure functions of sim
    state, which is the reason a spear does not jitter while the game is
    paused.

    *cam* is REQUIRED and has no default - see :func:`draw_one`.
    """
    try:
        spears = _iter_spears(world)
    except Exception:
        log.debug("could not read the spear list", exc_info=True)
        return
    if not spears:
        return
    for s in spears:
        try:
            draw_one(surf, s, world, cam=cam)
        except Exception:
            log.debug("spear draw failed", exc_info=True)


# ============================================================== smoke test ==
if __name__ == "__main__":                             # pragma: no cover
    import os
    import time as _time
    import types

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode((320, 200))

    from ..sim.world import World
    from .renderer import Renderer

    world = World(seed=7)
    for _ in range(240):
        world.tick(1.0 / 30.0)

    reg = getattr(world, "spears", None)
    src = "world.spears"
    if reg is None:                                    # a tree without the sim half
        reg = types.SimpleNamespace(spears=[])
        world.spears = reg
        src = "stand-in registry"

    # A spear in every interesting state: rising, at apex, falling, and four
    # planted where they missed.
    from ..sim.throwing import Spear, STATE_FLIGHT, STATE_GROUND

    gy = world.terrain.ground_y
    # Sited relative to the COLONY, not at absolute 300-820. The world is 6400
    # px wide and the camera is wherever the villagers are, so the literal
    # coordinates this used put every spear ~3500 px off the left of the frame
    # and the "does this module draw anything" assertion below failed for the
    # only reason it must never fail for: the test was looking the wrong way.
    from ..sim.actions import colony_center

    home = float(colony_center(world))
    ren = Renderer()
    ren.camera.snap_to(home)

    for i, (off, ang) in enumerate(((-260.0, -1.05), (-140.0, -0.35), (-20.0, 0.45))):
        sx = home + off
        reg.spears.append(Spear(id=90 + i, x=sx, y=gy(sx) - 70.0,
                                vx=math.cos(ang) * 300.0,
                                vy=math.sin(ang) * 300.0, state=STATE_FLIGHT))
    for i, off in enumerate((80.0, 140.0, 200.0, 260.0)):
        sx = home + off
        reg.spears.append(Spear(id=80 + i, x=sx, y=gy(sx), state=STATE_GROUND))

    # The base frame has to be rendered with this module STUBBED OUT, because
    # renderer.draw already calls draw_spears at step 7. Copying the finished
    # frame instead measures "what does drawing the same AA line twice change",
    # which is a different and much less interesting number.
    from . import throwing as _self

    _real, _self.draw_spears = _self.draw_spears, lambda *a, **k: None
    try:
        scene = ren.draw(world, 1.0 / 60.0)
    finally:
        _self.draw_spears = _real
    before = pygame.surfarray.array3d(scene).copy()
    # ren.camera, not IDENTITY: this second pass has to land on top of the
    # frame ren.draw just produced, and that frame is wherever the colony is.
    draw_spears(scene, world, world.world_time, cam=ren.camera)
    after = pygame.surfarray.array3d(scene)
    changed = int((before != after).any(axis=2).sum())
    print(f"{len(reg.spears)} spears from {src}: {changed} px changed "
          f"vs a stubbed no-op draw")
    assert changed > 0, "the whole point of this module is that it draws something"

    # Cost with an empty registry (the ~99% case) and with eight in the air.
    def _cost(n_frames: int = 400) -> float:
        t0 = _time.perf_counter()
        for _ in range(n_frames):
            draw_spears(scene, world, 0.0, cam=ren.camera)
        return (_time.perf_counter() - t0) * 1e6 / n_frames

    busy = _cost()
    held, reg.spears = reg.spears, []
    empty = _cost()
    reg.spears = held
    print(f"cost: {empty:.2f} us/frame empty, {busy:.2f} us/frame with "
          f"{len(reg.spears)}")

    # Fail-soft: junk in the list, a world with no terrain and no lighting, a
    # registry that is a bare list. None of it may raise.
    class _Bare:
        pass

    draw_spears(scene, _Bare(), 0.0, cam=IDENTITY)
    bare = _Bare()
    bare.spears = [None, 7, "spear", types.SimpleNamespace(x="?", y=1.0)]
    draw_spears(scene, bare, 0.0, cam=IDENTITY)
    bare2 = _Bare()
    bare2.spears = list(reg.spears)                    # registry as a plain list
    draw_spears(scene, bare2, 0.0, cam=IDENTITY)
    # spear_pose answers in WORLD space and no longer culls on x: a spear at
    # 1e9 is a real position on no map we ship, but it is the CAMERA that
    # decides it is off frame, and it does that in draw_one.
    assert spear_pose(types.SimpleNamespace(x=1e9, y=0.0)) is not None
    _probe = types.SimpleNamespace(id=1, x=1e9, y=0.0, state="ground")
    _b = pygame.surfarray.array3d(scene).copy()
    draw_one(scene, _probe, None, cam=IDENTITY)
    assert not (_b != pygame.surfarray.array3d(scene)).any(), \
        "draw_one drew a spear a billion px off screen"
    try:
        draw_spears(scene, bare2, 0.0)                  # type: ignore[call-arg]
    except TypeError as _exc:
        print("cam is required:", _exc)
    else:
        raise SystemExit("draw_spears accepted a call with no camera")
    print("fail-soft OK")

    out = os.path.join(os.environ.get("TEMP", "."), "throwing_smoke.png")
    pygame.image.save(scene, out)
    print("saved", out)
