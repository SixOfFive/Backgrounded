"""Renderer - composites the world into a single surface.

Strictly read-only with respect to simulation state. The layer order here is
the one defined in docs/ARCHITECTURE.md section 4 and the light composite at
step 10 is the whole visual thesis: draw the world lit, then multiply it down
to near-black, then add light back from candles, fires and lightning.

THE CAMERA LIVES HERE. The world is WORLD_W (6400) px of land and the frame is
RENDER_W (1600) px of picture, so a sim x is no longer a screen x. ``draw``
calls ``self.camera.follow`` as its first statement and every pass below reads
``self.camera``; the private ``_draw_*`` methods gained no parameters for it.

The rule for each pass is: convert exactly ONCE, at the line where a sim x is
first read, and treat everything downstream as screen space. Where a method
also has to *index the heightmap* it does that in world space first (the array
is WORLD_W long) and converts afterwards - ``_lava_flows`` and ``_draw_crossing``
are the two places both halves appear in the same function.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pygame

from ..constants import (
    BONFIRE_DRAW_SCALE, MATERIAL_COLORS, MAT_GRASS, MAT_LAVA, RENDER_H,
    RENDER_SIZE, RENDER_W, SCENE_BLIZZARD, SCENE_FLOOD, SCENE_SANDSTORM,
    SCENE_WILDFIRE, WORLD_W,
)
from ..sim.structures import (
    BONFIRE_FADE_UNITS, BONFIRE_RAMP_SEC, CROSSING_KINDS,
)
from . import creatures, dragons as dragon_art, fx, hud, sky
from . import items as relic_art
from . import throwing as spear_art
from .atlas import Atlas
from .camera import Camera
from .particles import ParticleSystem
from .stickfigure import draw_stickman

log = logging.getLogger(__name__)

# There used to be a `_takes_cam` probe and a `Renderer._art` wrapper here that
# passed `cam=` only to the entry points whose signature already accepted one,
# and called the rest exactly as before. It was written so the lanes of the
# wide-world change could land in any order, on the theory that a module drawn
# at world coordinates is a visible bug rather than a lost frame.
#
# That theory was wrong, and it is the single most expensive line of reasoning
# in this project's history. Ten of the eleven sim-entity entry points never
# grew the parameter; the shim called every one of them the old way, silently;
# they drew every stickman, animal, dragon, spear and relic off the right-hand
# edge of a 1600 px frame; and six independent lanes then measured their own
# patched trees and reported that the wide world worked. The frame that came
# back had name plates floating over empty ground - the plates being the one
# thing that went through camera.sx - and nothing whatsoever underneath them.
#
# So the shim is gone and `cam` is a REQUIRED keyword-only parameter on all
# eleven. A caller that forgets now raises TypeError on the first frame it is
# on, which is the loudest and cheapest possible way to be told. Nothing in
# this file may reintroduce a "call it the old way if it does not take one"
# branch; if a future entry point genuinely has no camera, it passes
# camera.IDENTITY and says so at the call site.

#: Largest magnitude, in px, that this file will hand to a pygame Rect.
#:
#: A Rect holds a C int. ``int(1e308)`` is a perfectly good Python int and a
#: TypeError the moment it is assigned to one - so a save carrying ONE huge but
#: finite coordinate used to kill the frame loop, and app.py's shutdown then
#: wrote the same world back, which made it permanent: every launch after that
#: died the same way, with no console and no window under run.pyw. persist.py
#: cannot catch this; ``_f`` there tests FINITENESS, and 1e308 is finite.
#:
#: 2**20 is 1,048,576 px: 164 times the width of the world and 1049 times the
#: height of the frame, so no coordinate any part of this program can legitimately
#: produce is anywhere near it and the clamp is invisible in every real frame.
#: It is also small enough that Rect's own internal arithmetic (a midbottom
#: assignment subtracts half a sprite width; a blit computes a clip) stays far
#: inside a 32-bit int. Anything past it is off screen by six orders of
#: magnitude, so pinning it to the limit and letting pygame clip draws exactly
#: what drawing it "properly" would have: nothing.
RECT_LIMIT: int = 1 << 20


def _ri(v: float) -> int:
    """A coordinate off sim state, as an int a pygame Rect can actually hold.

    THE conversion point. Every ``int(...)`` in this file that feeds a Rect, a
    blit destination or a surface size goes through here instead, which is why
    there is no per-entity try/except on the hot path: the clamp costs a
    comparison pair where the bare ``int()`` already was.

    Deliberately branch-free via min/max rather than an ``if``, because the
    NaN case falls out for free: ``max(-L, nan)`` is ``-L`` (the comparison is
    False, so max keeps what it has), and the result is then a fixed number
    rather than the ValueError ``int(nan)`` raises. inf clamps to the limit,
    -inf to its negative, and every real coordinate passes through untouched.
    """
    return int(min(RECT_LIMIT, max(-RECT_LIMIT, v)))


# Below this local light level an agent is drawn as a silhouette rather than
# in its identity colour. Feature 36.
SILHOUETTE_CUTOFF = 0.30
SILHOUETTE_COLOR = (14, 15, 22)

# Heat shimmer geometry, in px. The band is measured off the live heightmap
# rather than fixed, so it follows a map the disasters have reshaped. It reaches
# SHIMMER_RISE above the highest ground (rising air) and SHIMMER_BELOW under the
# lowest (so the ground line itself is inside the distorted region and boils),
# and SHIMMER_MAX_H caps the whole thing: fx.draw_heat_shimmer snapshots and
# rebuilds the rect it is handed, so its cost is linear in area and a map with
# 400 px of relief would otherwise turn a cheap effect into a full-frame one.
# Measured at 1600x220: ~5 ms a frame on top of a ~5 ms base draw, which fits
# inside the 60 Hz budget with room to spare and is only paid in this scene.
SHIMMER_RISE = 92
SHIMMER_BELOW = 24
SHIMMER_MAX_H = 220
#: px of horizontal displacement at zero and at full drought. The low end is
#: kept just above fx.draw_heat_shimmer's own 0.4 px cutoff so the effect fades
#: in continuously instead of popping on.
SHIMMER_MIN_PX, SHIMMER_MAX_PX = 0.5, 3.2

# Lava geometry. The flow is sampled onto the live heightmap every LAVA_SAMPLE_PX
# so the band follows a contour the disasters keep rewriting, and LAVA_SAMPLES
# caps the work however wide the front gets (a full-width flow at 8 px is ~42
# points, which is already finer than the terrain the ridge pass draws).
LAVA_SAMPLE_PX = 8.0
LAVA_SAMPLES = 64
#: ...and how many of those points get a light in the lightmap. Each one is a
#: full radial blit the size of a firepit's, so this is the cost dial rather
#: than a look dial: six across a flow overlap into one continuous wash.
LAVA_LIGHTS = 6
LAVA_LIGHT_R = 190.0
LAVA_LIGHT_COLOR = (255, 104, 34)
#: Embers per second per flow at full heat, split over three points along it.
#: The particle pool caps "ember" at 500 and burning props draw on the same
#: budget, so this is set to read as a constant drizzle of sparks rather than to
#: saturate: at 26/s a flow holds roughly 40 in the air at once.
LAVA_EMBERS_PER_SEC = 26.0

#: px between samples when tracing a bridge deck off the terrain. Finer than
#: the deck's end ramps (which are >= 10 px) so the join is drawn where it
#: actually is, coarse enough that a 190 px chasm costs ~24 points, not 190.
CROSSING_SAMPLE_PX = 8.0

#: How much slack a cull gets, in px. Half a sprite's width, near enough: the
#: widest thing drawn from a single anchor point is a grown hut at ~110 px and
#: a mature tree at ~120, so 160 keeps every one of them from popping in at the
#: frame edge while still throwing away the ~75% of a 6400 px world that is not
#: being looked at.
CULL_PAD = 160.0
#: Agents get less, because a stickman is 21 px wide - but its name plate is
#: not, and the plate is anchored to the body.
AGENT_CULL_PAD = 120.0


class Renderer:
    #: Width of one cached terrain strip, in px.
    #:
    #: Re-measured on this tree rather than inherited, by forcing one strip to
    #: rebuild 40 times at each width (mean / p95, ms):
    #:
    #:     800 px  3.04 / 4.01      1600 px  5.06 / 6.09
    #:    3200 px 12.05 / 12.60     6400 px 26.65 / 30.21
    #:
    #: So a full-width rebuild is 26.65 ms - 1.6 frames at 60 Hz - on a path
    #: that fires on EVERY quarry divot and every quake scar. The 26.65 that
    #: stood here for the full rebuild reproduces exactly; the 2.56 quoted for
    #: an 800 px chunk does not, and is really 3.04, so the per-chunk figures
    #: below are the measured ones and the old 5.31-at-1600 is now 5.06.
    #:
    #: Blitting was never the problem (a 6400-wide alpha surface onto the 1600
    #: scene measured 0.634 ms/frame against 0.571 for a 1600-wide one; pygame
    #: clips). So the fix is to rebuild less, not to blit less: 8 chunks, a
    #: per-chunk damage test, and only the 2-3 that intersect the frame are
    #: ever built or blitted. 800 keeps the worst hitch inside a 60 Hz frame,
    #: which 1600 (5.06 ms, p95 6.09) does not.
    TERRAIN_CHUNK: int = 800

    #: How far a column of ground may move before its strip is redrawn, in px.
    #:
    #: This number is the whole of the cache's second life. The test used to be
    #: an exact float sum over the strip, and three scenes - blizzard, volcano
    #: and sandstorm - settle snow or ash onto EVERY column at ~0.0018 px a
    #: tick. That is invisible (a fiftieth of a pixel a second) and it moved an
    #: 800-column sum by ~1.4 every frame, so the fingerprint never matched,
    #: every visible strip was rebuilt every frame, and the cache had a 0% hit
    #: rate in exactly the scenes that cost the most to draw. Measured before
    #: this: 3.00 rebuilds/frame in all three, 0.00 in clear.
    #:
    #: Rounding the sum does not fix it - the drift outruns any rounding you
    #: would accept - and neither does rounding per column: with 800 columns
    #: drifting together, ~1.4 of them cross a rounding boundary every frame
    #: whatever the quantum. The test has to be *per column against the ground
    #: as it was drawn*, which is what this is: keep the height slice the strip
    #: was built from and rebuild when any column has moved half a pixel from
    #: it. A settling scene then redraws about every 280 ticks (~9 s), which is
    #: precisely when the accumulated drift becomes a pixel you could see, and
    #: a 30 px quarry divot still trips it on the very next frame.
    #:
    #: Half a pixel and not one: the rim highlight is a 2 px line drawn on the
    #: contour, and at a 1.0 px threshold a slope that crept just under it
    #: showed a visible step against the strip next door when that one did
    #: rebuild for its own reasons.
    TERRAIN_TOL: float = 0.5

    #: How long a strip may hold a stale MATERIAL trickle, in seconds.
    #:
    #: The height tolerance above kills the settling drift outright (measured
    #: after it: 0 height-dirty strip-frames in all three scenes). What is left
    #: is the other half of the same weather: snow, ash and sand are *painted*
    #: a column or two at a time, and a changed material is a changed colour,
    #: so it cannot be given a tolerance the way a sub-pixel height can. It can
    #: be given a clock. Measured over 120 frames of each scene, the trickle is
    #: 1-3 columns per dirty strip - i.e. paying 3.04 ms to recolour 16 px of
    #: band - and that is worth batching.
    #:
    #: So material-only damage waits: a strip refreshes at most every half
    #: second, which is 15 frames on the preview and two on the 4 Hz wallpaper
    #: path. Anything that is not a trickle jumps the queue - see
    #: :meth:`_chunk_damage`, where moved ground, lava and a wide change are
    #: all urgent - so a quarry divot, a quake scar and a lava front are drawn
    #: on the very next frame exactly as before.
    #:
    #: Half a second and not two: this is the delay before a newly white column
    #: of ground turns white, and at 0.5 s the snow line reads as advancing
    #: while at 2 s it reads as stepping.
    CHUNK_REFRESH_SEC: float = 0.5

    #: Columns of changed material that make a repaint urgent rather than
    #: deferrable. One or two is weather; sixteen is something that happened.
    CHUNK_MAT_URGENT: int = 16

    #: Chunks kept either side of the visible span. Everything else is dropped.
    #:
    #: The cache never evicted: pan across the map once and all 8 strips stay
    #: resident forever, measured at 13.31 MiB of Surfaces in a process that is
    #: meant to sit behind a wallpaper all day. One chunk of margin is what the
    #: camera can actually reach before the next frame - MAX_SPEED is 220 px/s,
    #: so at 4 Hz on the wallpaper path a strip one chunk out is at most one
    #: frame away from being needed, and evicting it would trade 3.04 ms of
    #: rebuild for ~1.6 MiB. Two chunks of margin saves nothing more; zero
    #: margin makes every pan direction cost a rebuild at the frame edge.
    #: Measured after a full pan across the map: 8/8 strips and 13.31 MiB
    #: resident before this, 3/8 and 4.89 MiB after.
    CHUNK_KEEP: int = 1

    def __init__(self) -> None:
        self.scene = pygame.Surface(RENDER_SIZE).convert()
        self.out = pygame.Surface(RENDER_SIZE).convert()
        self.lightmap = pygame.Surface(RENDER_SIZE).convert()
        self.atlas = Atlas()
        self.particles = ParticleSystem()

        self.show_stats: bool = True
        self.show_roster: bool = True
        self.show_names: bool = True
        self.show_log: bool = True
        # Read straight off the stored config rather than waiting to be told.
        # app.py sets the other three flags from its own Config instance and
        # will overwrite this one with the same value the moment it learns to;
        # self-loading in the meantime is what stops the flag being decorative.
        try:
            from ..config import Config
            self.show_activity: bool = bool(
                getattr(Config.load(), "show_activity", True))
        except Exception:
            log.debug("could not read show_activity from config", exc_info=True)
            self.show_activity = True
        #: Where the frame sits on the world. The Renderer owns the only real
        #: Camera in the process; nothing in sim/ can see it.
        self.camera = Camera()
        #: cam.x baked into the frame draw() last returned. app.py needs this
        #: and not self.camera.x to turn a click back into a world x: the user
        #: clicked the picture they were looking at, which is last frame's.
        self.camera_x_presented: float = 0.0
        # One drawn strip per TERRAIN_CHUNK px of world, each with a snapshot
        # of the ground it was drawn from, so a divot dug at x=5100 rebuilds
        # 800 px and not 6400 - and so a scene that settles snow onto every
        # column does not rebuild anything at all until it has settled half a
        # pixel of it. See TERRAIN_TOL.
        self._chunk_cache: list[tuple[pygame.Surface, int] | None] = []
        #: Per chunk: ``(height_slice_copy, material_slice_copy)`` as of the
        #: last build, or None. Copies, not views - a view aliases the live
        #: heightmap and compares equal to itself forever.
        self._chunk_fp: list[tuple[np.ndarray, np.ndarray] | None] = []
        #: Value of ``self._clock`` when each strip was last built, so a
        #: deferred repaint can be given a deadline. See CHUNK_REFRESH_SEC.
        self._chunk_t: list[float] = []
        #: Seconds of drawn time. Accumulated from ``draw``'s own dt rather
        #: than read off a wall clock, so a paused or stepped renderer defers
        #: repaints by the time the *picture* has been up for.
        self._clock: float = 0.0
        # Label plates already placed this frame, so two villagers standing on
        # top of each other do not stack their status text into a smear.
        self._label_rects: list[pygame.Rect] = []
        #: Frames drawn. A DIAGNOSTIC, and nothing in this file may animate off
        #: it: `self._frame % 3` was the lava smoke puff's rate until it was
        #: found to be a frame-rate dial rather than a time one. Anything that
        #: moves takes dt (or world_time); see :meth:`_emit_weather`.
        self._frame = 0
        # Last screen-shake offset baked into the returned frame, so draw_hud
        # can match it on the wallpaper path.
        self._shake: tuple[int, int] = (0, 0)
        # (disabled subsystem names, hud scale, rendered plate) - see
        # _fault_banner. The scale is part of the key, not just the content.
        self._fault_plate: tuple[tuple[str, ...], float, pygame.Surface] | None = None
        #: pass name -> how many times it has raised. See :meth:`_pass`.
        self._pass_faults: dict[str, int] = {}
        #: Per-emitter leftover fraction of a particle. See :meth:`_emit_weather`.
        #: Rebuilt every frame from the emitters that actually ran, so a firepit
        #: that goes out or a prop that finishes burning takes its entry with it.
        self._emit_acc: dict[object, float] = {}

    # --------------------------------------------------------------- guard --
    def _pass(self, name: str, fn, *a, **kw) -> None:
        """Run one draw pass. A pass that raises costs the pass, not the frame.

        Half the passes in :meth:`draw` were already wrapped like this and half
        were bare, and the bare ones are how a single poisoned coordinate in a
        save file used to end the process: the exception left draw(), left
        App._frame, ended App.run's loop, and _shutdown then wrote the same
        world back to disk, so the next launch did it again - permanently, with
        no console and no window under run.pyw.

        The clamp at :func:`_ri` is the real fix for coordinates and this is the
        floor underneath it: whatever else a pass finds to raise on, the other
        eleven still draw and the app is still running to be told about it.

        It is a per-PASS guard and deliberately not a per-entity one - there are
        a dozen calls a frame here against ~150 props and ~20 structures, so the
        cost is a function call per pass and nothing at all per entity.

        Logged loudly once and then counted, because these run at 60 Hz for
        hours: a pass that fails every frame would otherwise write 216,000 stack
        traces an hour into a log file in %LOCALAPPDATA%.
        """
        try:
            fn(*a, **kw)
        except Exception:
            n = self._pass_faults.get(name, 0) + 1
            self._pass_faults[name] = n
            if n == 1:
                log.exception("render pass %r failed; the rest of the frame is "
                              "still being drawn", name)
            elif n % 1800 == 0:
                log.warning("render pass %r has now failed %d times", name, n)

    # ------------------------------------------------------------- public --
    def draw(self, world, dt: float) -> pygame.Surface:
        # First statement, before anything reads a coordinate: every pass below
        # converts through self.camera, and they must all agree on where it is.
        self.camera.follow(world, dt)
        self._frame += 1
        # Clamped the way Camera.follow clamps its own step: a 20-second stall
        # (a laptop coming out of sleep) must not retire every deferred repaint
        # at once and hand back a frame with three chunk rebuilds in it.
        try:
            self._clock += min(0.25, max(0.0, float(dt)))
        except (TypeError, ValueError):
            pass
        s = self.scene
        ev = world.events

        # Sampled once, up front, because three different passes want it: the
        # ember emitter below, the lightmap (a flow lights what is around it)
        # and the emissive band drawn after the composite. Empty in every scene
        # but the eruption - and for the minute afterwards, while the flow the
        # eruption left behind is still cooling.
        lava = self._lava_flows(world)

        try:
            # The emit pass runs before update(), so the particle system is
            # told where the frame is first: weather with no explicit x is
            # seeded across the VIEW, and at cam.x = 3200 that is a different
            # 1600 px of world than it was at startup.
            self.particles.view_x = self.camera.x
            self._emit_weather(world, dt, lava)
            self.particles.update(dt, world.terrain, ev, cam=self.camera)
        except Exception:
            log.exception("particle update failed")

        # 1-2. sky, then the parallax ridges. Both view space - the sky is not
        # in the world at all, and the ridges scroll at a fraction of the
        # camera rather than with it, which is what "parallax" means and what
        # stops a 4x world reading as a backdrop sliding past on rails.
        self._pass("sky", sky.draw_sky, s, ev.scene, world.world_time, ev,
                   world.lighting)
        self._pass("parallax", sky.draw_parallax, s, world.seed, ev.scene,
                   world.lighting, cam=self.camera)

        # 3. distant weather
        self._pass("weather_back", self.particles.draw, s, layer="back",
                   cam=self.camera)

        # 4. terrain, then the scars an earthquake has torn in it. The scars go
        # here and not later because they are *ground*: they must sit under
        # everything that stands on the terrain, and the light composite must
        # dim them along with the rest of the world.
        self._pass("terrain", self._blit_terrain, s, world.terrain)
        self._pass("fissures", self._draw_fissures, s, world)

        # 5-6. props then structures, painter-ordered by x for a little depth
        self._pass("props", self._draw_props, s, world)
        self._pass("structures", self._draw_structures, s, world)

        # 6b. the one dragon that stands in the colony rather than over it.
        # Inside the light composite, and before the agents - see
        # _draw_dragons for both halves of why.
        self._draw_dragons(s, world, grounded=True)

        # 7. agents, then the wildlife among them
        self._pass("agents", self._draw_agents, s, world)
        try:
            creatures.draw_animals(s, world, world.world_time, cam=self.camera)
        except Exception:
            log.exception("animal draw failed")
        # Thrown spears, immediately after the bodies. sim/throwing.py has
        # modelled a real projectile since it landed and there was no call here
        # at all, so a hostile proof measured 230 throws that drew literally
        # nothing - no shaft in the air, none in the dirt. This line is the
        # whole of that fix; the art is render/throwing.py.
        #
        # At this layer and not with the dragons: a spear lives in the same
        # 0-140 px band as the wolves it is thrown at, so the composite below
        # must light it off the same torches, or it becomes the one object in
        # the colony that glows at 3am - the very thing _draw_dragons had to
        # unpick for the quadruped. After the agents and the animals rather
        # than before, because a shaft is 25 px long and 2 px wide and any body
        # it were drawn behind would simply eat it.
        try:
            spear_art.draw_spears(s, world, world.world_time, cam=self.camera)
        except Exception:
            log.exception("spear draw failed")
        # Relics lying in the dirt, immediately after the spears and for the
        # same reasons: a dropped relic is 13 px of ground clutter in the same
        # band as the wolves, so it has to be INSIDE the light composite or it
        # is the one object in the colony that glows at 3am. After the bodies,
        # because anything drawn behind a stickman is simply eaten.
        #
        # Its self-lit half - the halo that is how you actually find one at
        # night - is a separate call after the composite; see draw_relic_fx
        # below. Two calls rather than one is the whole reason the art reads:
        # the body drawn here is a dark smudge in the middle of its own pool of
        # light until that second pass puts the glow back on top.
        try:
            relic_art.draw_relics(s, world, world.world_time, cam=self.camera)
        except Exception:
            log.exception("relic draw failed")
        try:
            creatures.draw_mining_dust(s, world, world.world_time,
                                       cam=self.camera)
        except Exception:
            log.exception("mining dust draw failed")

        # 8-9. near weather and particles
        self._pass("weather_front", self.particles.draw, s, layer="front",
                   cam=self.camera)

        # flood water sits above the terrain but below the light pass
        if getattr(ev, "water_level", None):
            self._pass("water", self._draw_water, s, world, ev.water_level)

        # 10. the light composite
        self._pass("light", self._composite_light, s, world, lava)

        # Lava, immediately after it. The terrain pass has already painted the
        # flat MAT_LAVA band under everything, which is right - it is ground -
        # but the composite has just multiplied it down by an ambient this same
        # scene crushed to a quarter. This puts the flow's own light back on top
        # of the lit world, which is what makes it the brightest thing on screen
        # instead of a dark orange stripe.
        if lava:
            self._pass("lava", fx.draw_lava_flow, s, lava, world.world_time)

        # Heat haze over the ground, after the composite because it displaces
        # what you can *see* - lit pixels, not the world's own geometry - and
        # before the UFO, the lightning and the HUD, none of which are things
        # the air between you and the ground is in front of.
        try:
            heat = float(getattr(ev, "heat", 0.0) or 0.0)
        except (TypeError, ValueError):
            heat = 0.0
        if heat > 0.02:
            self._draw_heat_shimmer(s, world, heat)

        # The saucer sits above the light pass: its beam is its own light and
        # should not be multiplied down into the night.
        try:
            creatures.draw_ufo(s, world, world.world_time, cam=self.camera)
        except Exception:
            log.exception("ufo draw failed")

        # Everything about a relic that makes its own light: the halo over each
        # one on the ground, a BFG discharge, and the saucer's detonation when an
        # amulet-wearer refuses to be taken.
        #
        # After the composite, next to the saucer, and specifically AFTER it: the
        # amulet's flash and the falling wreck are drawn over a hull that
        # creatures.draw_ufo has already put down, so this call going first would
        # paint the explosion and then paint the intact saucer on top of it.
        #
        # This is also why the two relic calls are split. A ground relic needs
        # both: the body at step 7 so the scene's own light lands on it, and the
        # glow here so it survives the multiply-down. Measured on a night frame,
        # the halo is 25x the ink of the object it sits on - at 13 px across, a
        # relic is found by its light and not by its silhouette.
        try:
            relic_art.draw_relic_fx(s, world, world.world_time, cam=self.camera)
        except Exception:
            log.exception("relic fx draw failed")

        # The four dragons that fly, at the same layer and for the same reason.
        # A dragon at cruise altitude (150-420 px) is above every light source
        # in the world, so putting it INSIDE the composite would multiply it
        # down by the scene's ambient and leave a black smear on black - it
        # would not merely be dark, it would be gone. Drawing it here instead
        # means the module does its own silhouette handling: render/dragons.py
        # applies exactly creatures._draw_animal's rule (light_at below
        # _SIL_CUTOFF -> draw the whole creature in _SIL_COLOR), so a dragon
        # low enough to be in torchlight is lit like a wolf and one at altitude
        # is a flat cut-out against the sky, which is what the high-altitude
        # art was designed around.
        #
        # The one accepted cost of drawing after the composite: a serpent
        # cresting a chasm mouth overdraws a few px of the far wall. The module
        # skips a serpent entirely below alt -6 rather than clipping, because
        # clipping wants a scratch surface every frame; if it ever reads badly
        # on a real chasm map the fix is a clip rect here, not in the art.
        self._draw_dragons(s, world, grounded=False)

        # 11. lightning geometry, then vignette
        self._pass("lightning", self._draw_lightning, s, world)
        self._pass("vignette", fx.draw_vignette, s)

        # The eclipse's horizon ring, for the same reason and at the same layer
        # as the fog wash: it is light coming in under the shadow from outside
        # it, not part of the lit world, so the light composite must not have a
        # say in it. Above the vignette so the corners keep their glow.
        try:
            ecl = float(getattr(ev, "eclipse", 0.0) or 0.0)
        except (TypeError, ValueError):
            ecl = 0.0
        if ecl > 0.02:
            self._pass("eclipse", fx.draw_eclipse_twilight, s, min(1.0, ecl))

        # The haze wash sits above the whole lit world (so it dims it rather
        # than being multiplied away), but below the HUD, which must stay
        # readable. Density and colour both come off the sim: the scene decides
        # whether this is a grey fog or a tan sandstorm, not the renderer.
        try:
            fog = float(getattr(ev, "fog", 0.0) or 0.0)
        except (TypeError, ValueError):
            fog = 0.0
        if fog > 0.02:
            # Hand it the daylight level too, or the wash - being painted over
            # the finished frame - becomes the one part of the scene the
            # day/night cycle cannot reach, and a 2am fog reads as noon.
            try:
                dl = sky.daylight(sky.day_phase(world.world_time, world.lighting))
            except Exception:
                dl = 1.0
            self._pass("fog", fx.draw_fog_overlay, s, min(1.0, fog),
                       world.world_time, getattr(ev, "fog_color", None), dl)

        # 12. the HUD used to be blitted here, after the light composite so it
        # stays readable in a black scene. It is drawn by :meth:`draw_hud`
        # instead, because *where* it belongs depends on who is looking: the
        # wallpaper is a bare image and needs it baked in, while the preview
        # needs it on the window surface or zooming drags the panels off with
        # the picture. app.py calls draw_hud once per output; see it for the
        # ordering that keeps this to a single render.

        # screen shake: blit the composed scene at an offset
        try:
            dx, dy = ev.shake_offset() if hasattr(ev, "shake_offset") else (0, 0)
        except Exception:
            log.debug("shake offset failed", exc_info=True)
            dx, dy = 0, 0
        # Remembered so draw_hud can re-apply it on the wallpaper path. The HUD
        # was drawn before this blit and therefore shook with the world; that is
        # the wallpaper's long-standing look and is preserved deliberately.
        #
        # Clamped through _ri like every other coordinate: shake_offset is
        # events.shake_amp times a sine, and a save carrying shake_amp = 1e308
        # made the blit two lines down raise "invalid destination position for
        # blit" - the same permanent-brick as a poisoned prop, reached by a
        # channel that is not a position at all.
        self._shake = (_ri(dx), _ri(dy))
        dx, dy = self._shake
        # Last statement, after everything that could still have moved the
        # camera: this is the cam.x that is baked into the surface being
        # returned, and app.py turns a click on THAT picture back into a world
        # x with it. Reading self.camera.x at click time would be a frame
        # ahead of what the user was looking at.
        self.camera_x_presented = self.camera.x
        if dx or dy:
            self.out.fill((0, 0, 0))
            self.out.blit(s, (int(dx), int(dy)))
            return self.out
        return s

    def draw_hud(self, surf: "pygame.Surface | None", world,
                 shake: bool = False,
                 mouse: "tuple[int, int] | None" = None) -> None:
        """Paint the stats panel and the chronicle log onto *surf*.

        Both anchor to *surf*'s own corners, so this lands top-right /
        bottom-left of whatever it is handed - the 1600x1000 world frame on the
        wallpaper and capture paths, or the real preview window surface, where
        the panels then hold the window corner however the view is zoomed and
        panned.

        ``shake`` re-applies the earthquake offset :meth:`draw` baked into the
        frame it returned. The wallpaper wants it (the HUD has always shaken
        with the world there); the window overlay must not, because the window
        itself is not what is shaking.

        ``mouse`` is the pointer in *surf*'s coordinates, and enables the stats
        panel's hover tooltips. Only the window path passes it - the wallpaper
        is an image with no pointer over it, and a tooltip baked into it would
        be a permanent grey box on the desktop.

        Never raises: both hud functions already swallow their own errors, and
        the flags are read defensively because this sits on the frame loop.
        """
        if surf is None or world is None:
            return
        try:
            off = self._shake if shake else (0, 0)
            if self.show_stats:
                hud.draw_stats(surf, world, show_roster=self.show_roster,
                               offset=off, mouse=mouse)
            if self.show_log:
                hud.draw_log(surf, world, offset=off)
            # Deliberately not behind show_stats/show_log: this is a fault
            # report, not a statistic, and hiding the panels must not hide it.
            plate = self._fault_banner(world)
            if plate is not None:
                r = plate.get_rect()
                r.midtop = (surf.get_width() // 2 + _ri(off[0]),
                            12 + _ri(off[1]))
                surf.blit(plate, r)
        except Exception:
            log.debug("hud draw failed", exc_info=True)

    def _fault_banner(self, world) -> "pygame.Surface | None":
        """Plate naming any subsystem the world has switched off, or None.

        ``World._guarded`` disables a subsystem for the rest of the session the
        first time it raises, and logs it - but the app runs under pythonw with
        no console, so the StreamHandler app.py attaches writes those records to
        a ``None`` stdout and they are simply gone. That is exactly how a dead
        scene rotation once shipped with every smoke check passing (see
        tools/smoke.py, "no subsystem disabled itself after a reload"). A colony
        running for hours with no weather, no population or no lighting is the
        worst failure this thing has and the one the user is least able to spot,
        so it gets said on the frame rather than only in a log nobody reads.

        Cached on the disabled set *and the HUD scale*: this is on the frame loop
        and the set only changes when something dies, which is at most a handful
        of times ever. The scale belongs in the key because it is baked into the
        two font sizes below, and it is a live user dial - [ and ] move it at
        runtime - so keying on the names alone left the banner frozen at whatever
        size it happened to be built at while every other panel resized around it.
        """
        try:
            names = tuple(sorted(str(n) for n in (getattr(world, "_disabled", ()) or ())))
        except Exception:
            return None
        if not names:
            return None
        scale = float(getattr(hud, "HUD_SCALE", 1.0) or 1.0)
        hit = self._fault_plate
        if hit is not None and hit[0] == names and abs(hit[1] - scale) < 1e-6:
            return hit[2]
        try:
            font = pygame.font.SysFont("consolas,dejavusansmono,couriernew",
                                       max(8, int(round(11 * scale))), bold=True)
            sub = pygame.font.SysFont("consolas,dejavusansmono,couriernew",
                                      max(7, int(round(9 * scale))))
            head = font.render("SUBSYSTEM OFFLINE: " + ", ".join(names),
                               True, (235, 112, 112))
            tail = sub.render("this colony is running degraded - restart to recover",
                              True, (196, 202, 214))
            pad = max(4, int(round(6 * scale)))
            w = max(head.get_width(), tail.get_width()) + pad * 2
            h = head.get_height() + tail.get_height() + pad * 2 + 2
            plate = pygame.Surface((w, h), pygame.SRCALPHA)
            plate.fill((18, 10, 12, 214))
            pygame.draw.rect(plate, (168, 62, 62), plate.get_rect(),
                             max(1, int(round(scale))))
            plate.blit(head, ((w - head.get_width()) // 2, pad))
            plate.blit(tail, ((w - tail.get_width()) // 2,
                              pad + head.get_height() + 2))
        except Exception:
            log.debug("fault banner build failed", exc_info=True)
            return None
        self._fault_plate = (names, scale, plate)
        return plate

    # ------------------------------------------------------------ terrain --
    def _blit_terrain(self, s: pygame.Surface, terrain) -> None:
        """Paint the 2-3 cached terrain strips that intersect the frame.

        This used to be one full-width surface, rebuilt whole whenever a cheap
        fingerprint of the heightmap moved. At WORLD_W that rebuild measured
        26.65 ms - 1.6 frames at 60 Hz - and it fires on every quarry divot,
        every quake scar and every meteor crater, i.e. constantly and always
        during the busiest moment on screen. Chunking makes the rebuild local
        to the damage: digging a hole at x=5100 redraws 800 px, not 6400.

        The strips are blitted at ``-cam.x``, which is why ``Camera.follow``
        rounds: a fractional offset makes pygame resample the entire background
        every frame, and on the 4 Hz wallpaper path that crawl is the single
        most visible artefact this renderer can produce.
        """
        try:
            h = terrain.height
            n = int(h.shape[0])
            if n <= 0:
                return
            chunk = max(64, int(self.TERRAIN_CHUNK))
            count = (n + chunk - 1) // chunk
            if (len(self._chunk_cache) != count or len(self._chunk_fp) != count
                    or len(self._chunk_t) != count):
                # Width changed under us - a regenerated world, or a save from
                # before the map got wider. Drop the lot rather than trying to
                # reconcile strips against a different heightmap.
                #
                # All three lengths are tested, not just the cache's: they are
                # parallel arrays and the two that are not the cache are read
                # by index off the cache's bounds, so a desync would be an
                # IndexError on the frame path rather than a stale strip.
                self._chunk_cache = [None] * count
                self._chunk_fp = [None] * count
                self._chunk_t = [0.0] * count
            cx = int(self.camera.x)
            i0 = max(0, cx // chunk)
            i1 = min(count - 1, (cx + RENDER_W - 1) // chunk)
            for i in range(i0, i1 + 1):
                hit = self._terrain_chunk(terrain, i)
                if hit is not None:
                    strip, top = hit
                    s.blit(strip, (i * chunk - cx, top))
            self._evict_chunks(i0, i1, count)
        except Exception:
            log.exception("terrain blit failed")

    def _evict_chunks(self, i0: int, i1: int, count: int) -> None:
        """Drop every cached strip more than CHUNK_KEEP chunks off screen.

        Without this the cache is write-only: pan the length of the map once
        and all 8 strips stay resident for the rest of the session, measured at
        13.31 MiB of Surfaces. That is not a leak in the sense of growing
        without bound - it is bounded by the map - but this process is a
        wallpaper that runs all day, and holding 13 MiB of pictures of ground
        nobody is looking at is exactly the kind of thing that gets an ambient
        app killed by a memory-pressure reaper.

        Cheap enough to run unconditionally: `count` is 8.
        """
        lo = max(0, i0 - self.CHUNK_KEEP)
        hi = min(count - 1, i1 + self.CHUNK_KEEP)
        for j in range(count):
            if lo <= j <= hi:
                continue
            if j < len(self._chunk_cache) and self._chunk_cache[j] is not None:
                self._chunk_cache[j] = None
                if j < len(self._chunk_fp):
                    self._chunk_fp[j] = None
                if j < len(self._chunk_t):
                    self._chunk_t[j] = 0.0

    def _terrain_chunk(self, terrain,
                       i: int) -> "tuple[pygame.Surface, int] | None":
        """The drawn strip for chunk *i*, rebuilt only if its own ground moved.

        "Moved" is :meth:`_chunk_damage`'s verdict and not a fingerprint
        comparison. It was a fingerprint - first a ``[::64]`` stride, which was
        25 samples across the whole map and stepped straight over a 30 px
        quarry divot, then an exact float sum over the strip, which caught the
        divot and also caught the ~0.0018 px/tick of settling that blizzard,
        volcano and sandstorm apply to every column, so those three scenes
        rebuilt every visible strip every frame forever. Both failures are the
        same mistake: one number cannot answer both "did anything change" and
        "did anything change enough to see".

        Geometry is drawn one column PAST each end and left to pygame's clip.
        The alternative - a padded surface with overlapping blits - double-
        blends the rim highlight at every seam, which shows up as a brighter
        vertical tick every 800 px. Abutting strips with clipped overdraw have
        no seam at all.

        Returns ``(surface, top_y)``: the strip is cropped to its own skyline
        rather than being full height. A full-height strip is ~65% empty sky,
        and the blit is per-frame while the crop is per-rebuild - measured, the
        three visible strips cost 0.94 ms/frame uncropped against 0.31 cropped,
        which is most of the difference between this and the single full-width
        blit it replaced.
        """
        try:
            h = terrain.height
            mats = terrain.material
            n = int(h.shape[0])
            chunk = max(64, int(self.TERRAIN_CHUNK))
            x0 = i * chunk
            x1 = min(n, x0 + chunk)
            if x0 >= x1:
                return None
            if (i < len(self._chunk_cache) and i < len(self._chunk_t)
                    and self._chunk_cache[i] is not None):
                dmg = self._chunk_damage(i, h, mats, x0, x1)
                if dmg == 0:
                    return self._chunk_cache[i]
                if (dmg == 1 and
                        self._clock - self._chunk_t[i] < self.CHUNK_REFRESH_SEC):
                    # A column or two of settled weather, and the strip was
                    # redrawn recently. Keep showing the one we have.
                    return self._chunk_cache[i]

            lo = max(0, x0 - 1)
            hi = min(n, x1 + 1)
            # 4 px of headroom for the 2 px rim line and its own rounding.
            top = int(max(0, min(RENDER_H - 1, math.floor(float(h[lo:hi].min())) - 4)))
            surf = pygame.Surface((x1 - x0, RENDER_H - top), pygame.SRCALPHA)
            bottom = RENDER_H - top

            # Solid body: one polygon down to the bottom of the screen.
            pts = [(lo - x0, bottom)]
            pts += [(x - x0, float(h[x]) - top) for x in range(lo, hi)]
            pts += [(hi - 1 - x0, bottom)]
            pygame.draw.polygon(surf, (38, 34, 32), pts)

            # Surface band coloured per material, drawn as a run-length set of
            # rects rather than one line per column.
            band = 16
            x = lo
            while x < hi:
                m = int(mats[x])
                x2 = x
                while x2 < hi and int(mats[x2]) == m:
                    x2 += 1
                col = MATERIAL_COLORS.get(m, MATERIAL_COLORS[MAT_GRASS])
                # Both edges of the band follow the contour. Closing the polygon
                # on two corner points instead - which is what this did - only
                # draws a band where the run is flat: over a hill it fills the
                # whole wedge between the skyline and a straight line joining
                # the run's ends. It was invisible while every material was a
                # muted earth tone against a near-black body, and became a lit
                # orange tent the first time a volcanic flow painted a 300 px
                # run of MAT_LAVA across a ridge. Two points per column rather
                # than one, on a strip that is cached until this strip moves.
                #
                # A run cut by the strip edge is fine: the geometry is per
                # column on both edges, so the visible half of a run is drawn
                # exactly where it would have been as part of the whole.
                seg = [(xi - x0, float(h[xi]) - top) for xi in range(x, x2)]
                seg += [(xi - x0, float(h[xi]) + band - top)
                        for xi in range(x2 - 1, x - 1, -1)]
                if len(seg) > 2:
                    pygame.draw.polygon(surf, col, seg)
                x = x2

            # Rim highlight so the silhouette still reads in near-darkness.
            rim = [(xi - x0, float(h[xi]) - top) for xi in range(lo, hi)]
            if len(rim) > 1:
                pygame.draw.lines(surf, (96, 104, 96), False, rim, 2)

            out = (surf.convert_alpha(), top)
            if i < len(self._chunk_cache):
                self._chunk_cache[i] = out
                # Copies. A view would alias the live heightmap, compare equal
                # to itself forever, and the strip would never be rebuilt - the
                # exact opposite failure to the one being fixed here, and a far
                # quieter one.
                self._chunk_fp[i] = (h[x0:x1].copy(), mats[x0:x1].copy())
                if i < len(self._chunk_t):
                    self._chunk_t[i] = self._clock
            return out
        except Exception:
            log.exception("terrain chunk %s build failed", i)
            return None

    def _chunk_damage(self, i: int, h, mats, x0: int, x1: int) -> int:
        """How badly chunk *i*'s ground differs from the strip drawn for it.

        ``0`` nothing worth drawing, ``1`` a deferrable trickle, ``2`` urgent.

        The whole point of TERRAIN_TOL, and the reason this is a per-column
        test and not a sum: one number cannot tell 800 columns drifting 0.0018
        px apart from one column dropping 1.4 px, and the first of those is
        three scenes' worth of settling snow while the second is a shovel.

        Materials get no *tolerance* - they are discrete and a change is always
        a change of colour - but they do get a queue. Three cases jump it:

        * moved ground, because that is a divot or a scar and it arrives with a
          villager standing in it;
        * MAT_LAVA appearing or leaving, because a flow paints colour across a
          ridge while moving no height at all, and a late lava front is the one
          stale repaint anybody would actually see;
        * a wide change (CHUNK_MAT_URGENT columns), because that is not weather.

        Answers ``2`` on anything it cannot make sense of, so a shape mismatch
        or a junk array costs a rebuild rather than a stale picture.
        """
        snap = self._chunk_fp[i] if i < len(self._chunk_fp) else None
        if snap is None:
            return 2
        try:
            hs, ms = snap
            if hs.shape[0] != x1 - x0 or ms.shape[0] != x1 - x0:
                return 2
            # float subtraction over 800 elements: ~3 us, against the 3.04 ms
            # rebuild it is deciding about.
            if float(np.abs(h[x0:x1] - hs).max()) >= self.TERRAIN_TOL:
                return 2
            now = mats[x0:x1]
            diff = now != ms
            n = int(diff.sum())
            if n == 0:
                return 0
            if n >= self.CHUNK_MAT_URGENT:
                return 2
            if bool(((now == MAT_LAVA) | (ms == MAT_LAVA))[diff].any()):
                return 2
            return 1
        except Exception:
            log.debug("chunk %s damage test failed", i, exc_info=True)
            return 2

    def _draw_fissures(self, s: pygame.Surface, world) -> None:
        """Darken the gaps the earthquake tore open. A no-op in every other
        scene, because nothing else ever fills ``events.fissures``.

        The ground y is resolved here rather than being taken from the scar's
        stored ``y``: that one records where the surface was when the crack
        started, and the sim then digs the surface out from under it, so using
        it would leave the crack floating a couple of dozen px above its hole.
        """
        fissures = getattr(world.events, "fissures", None)
        if not fissures:
            return
        cam = self.camera
        cracks: list[tuple[float, float, float, float, int]] = []
        for f in list(fissures):
            try:
                # Bounded on the way in, for the same reason every coordinate
                # in this file is: half and depth are not positions but they
                # become geometry, and fx builds a polygon out of them.
                depth = float(f.get("depth", 0.0))
                if not depth >= 2.0:    # not open enough to read as a hole yet
                    continue            # (and `not >=` so NaN is skipped too)
                depth = min(float(RENDER_H), depth)
                half = min(float(WORLD_W), max(0.0, float(f.get("half", 12.0))))
                x = float(f["x"])
                # Sampled in world space (ground_y indexes the heightmap), then
                # handed to fx in screen space - fx primitives take screen
                # coords and always did.
                if not cam.visible(x, half + 8.0):
                    continue
                cracks.append((cam.sx(x), world.terrain.ground_y(x),
                               half * 1.05, depth, int(f.get("seed", 1))))
            except Exception:
                continue
        try:
            fx.draw_fissures(s, cracks)
        except Exception:
            log.debug("fissure draw failed", exc_info=True)

    # -------------------------------------------------------------- props --
    def _draw_props(self, s: pygame.Surface, world) -> None:
        # Cull BEFORE the sort, not inside the loop. The scatter is 4x denser
        # than it was (143 props on a fresh map) and litter caps at 240, so on
        # a bad frame this was sorting ~400 objects to draw the ~100 that are
        # actually in the 1600 px frame. pygame would have clipped the rest
        # correctly - this is purely the cost of sorting and blitting a world's
        # worth of scenery to show a screen's worth.
        cam = self.camera
        visible = [p for p in world.props.all()
                   if p.alive and cam.visible(p.x, CULL_PAD)]
        for p in sorted(visible, key=lambda p: p.y):
            px = cam.sx(p.x)
            # Litter is drawn here rather than baked into the atlas: it is four
            # 9x6 px silhouettes with no stages, no variants worth baking and no
            # growth ladder, and the atlas would answer with its 2x2 transparent
            # "missing" surface for an unknown kind - i.e. the feature would be
            # invisible while looking perfectly wired up.
            if p.kind == "litter":
                try:
                    st = p.state if isinstance(p.state, dict) else {}
                    fx.draw_litter(s, px, p.y, int(st.get("shape", 0) or 0),
                                   bool(p.variant & 1))
                except Exception:
                    pass
                continue
            try:
                spr = self.atlas.get(p.kind, p.variant, state=p.state)
            except Exception:
                continue
            if spr is None:
                continue
            if p.kind == "tree" and p.state.get("fallen"):
                ang = float(p.state.get("fall_angle", 0.0))
                spr = pygame.transform.rotate(spr, -ang * 57.2958)
            r = spr.get_rect()
            r.midbottom = (_ri(px), _ri(p.y) + 2)
            s.blit(spr, r)

    def _draw_structures(self, s: pygame.Surface, world) -> None:
        cam = self.camera
        # Crossings are span-shaped and can be far wider than CULL_PAD, so they
        # are exempted from the point cull and clipped by _draw_crossing's own
        # span test instead. Everything else is a sprite on a single anchor.
        visible = [st for st in world.structures.all()
                   if st.kind in CROSSING_KINDS or cam.visible(st.x, CULL_PAD)]
        for st in sorted(visible, key=lambda st: st.y):
            # Bridges and ladders are span-shaped, not sprite-shaped: how big
            # they are is a property of the gap, not of the kind. They are
            # drawn from the terrain's own surface so the planking lands
            # exactly on the deck the physics is using.
            if st.kind in CROSSING_KINDS:
                self._draw_crossing(s, world, st)
                continue
            # A firepit running on swept-up rubbish is drawn as a bonfire:
            # the same pit, scaled up, under a much taller flame. The sim owns
            # the flag (Structure.bonfire_active); this only reads it, like
            # everything else in here.
            blaze = False
            try:
                blaze = bool(getattr(st, "bonfire_active", False))
            except Exception:
                blaze = False
            # Structure.scale() is 1.0 for everything but a finished hut, where
            # it rides `growth` up to HUT_SCALE_MAX. It used to be dropped on the
            # floor here, so a hut the sim had already grown - wider footprint,
            # taller silhouette, more sleepers - still drew as the 78x64 box it
            # was founded as, and the growth ladder the atlas bakes at startup
            # was unreachable. Passed undamped: width_now/height_now/top_y/
            # contains_point are all already this number, so drawing anything
            # else puts the sprite out of step with the sim's own footprint.
            try:
                scale = BONFIRE_DRAW_SCALE if blaze else st.scale()
            except Exception:
                scale = 1.0
            try:
                # `state` picks the wall material for a hut (Atlas.resolve);
                # absent or unknown is timber, which is every save so far.
                spr = self.atlas.get(st.kind, st.variant, stage=st.stage,
                                     scale=scale, state=getattr(st, "state", None))
            except Exception:
                continue
            if spr is None:
                continue
            r = spr.get_rect()
            r.midbottom = (_ri(cam.sx(st.x)), _ri(st.y) + 2)
            s.blit(spr, r)
            if blaze:
                self._draw_bonfire(s, world, st)
            # occupied buildings glow: someone is asleep in there
            occ = self._occupants(world, st)
            if occ:
                self._draw_occupancy(s, st, r, occ, world)

            # in-progress structures get a faint scaffold hint
            if not st.built and st.progress > 0:
                w = int(28 * st.progress)
                pygame.draw.line(s, (120, 110, 80),
                                 (r.centerx - 14, r.top - 6),
                                 (r.centerx - 14 + w, r.top - 6), 2)

    def _draw_bonfire(self, s: pygame.Surface, world, st) -> None:
        """The blaze over a firepit burning garbage, plus its ember drizzle.

        Strength rides the same two curves the sim's light source uses (a ramp
        in, a fade as the heap runs down), so the flame and the pool of light it
        throws grow and die together instead of one popping while the other
        eases. Read-only, and swallowed whole on failure: a missing bonfire is a
        cosmetic loss, a raised exception is a lost frame.
        """
        try:
            state = getattr(st, "state", {}) or {}
            age = float(state.get("bonfire_t", 0.0) or 0.0)
            left = float(getattr(st, "garbage_left", 0.0) or 0.0)
            k = max(0.2, min(1.0, age / BONFIRE_RAMP_SEC)
                    * max(0.3, min(1.0, left / BONFIRE_FADE_UNITS)))
            w = float(st.spec.width) * BONFIRE_DRAW_SCALE
            fx.draw_bonfire(s, self.camera.sx(st.x), float(st.y) + 1.0,
                            w * 0.7, w * 1.15, float(world.world_time), k)
        except Exception:
            log.debug("bonfire draw failed for #%s", getattr(st, "id", "?"),
                      exc_info=True)

    def _draw_crossing(self, s: pygame.Surface, world, st) -> None:
        """Draw a bridge or a ladder, sampled off the live terrain surface.

        Read-only with respect to the sim, like everything else in here. For a
        *finished* crossing the geometry comes from ``terrain.ground_y`` across
        the structure's own span - which over a stamped crossing is the deck or
        ramp itself - so the art and the walkable surface cannot drift apart.

        An unfinished bridge is drawn along the line it is *aiming* at instead,
        because sampling the ground there traces the wall of the chasm and
        paints planking 300 px down the inside of it. Ropes strung across the
        gap is both what building a bridge looks like and the honest signal
        that this is not something to step onto yet.
        """
        try:
            terrain = world.terrain
            span = st.crossing_span()
            if span is None:
                return
            cam = self.camera
            x0, x1 = span
            # WORLD clamps: these index the heightmap. Clamping to RENDER_W
            # here - which is what this line said when the world was 1600 px
            # wide - would drag every crossing east of x=1600 back to the map's
            # left edge and paint it there.
            x0 = max(0.0, min(float(x0), WORLD_W - 1.0))
            x1 = max(0.0, min(float(x1), WORLD_W - 1.0))
            if x1 - x0 < 4.0:
                return
            # Off-frame entirely: the span cull the point cull in
            # _draw_structures deliberately skipped for this kind.
            if cam.sx(x1) < -CULL_PAD or cam.sx(x0) > RENDER_W + CULL_PAD:
                return
            ruined = bool(st.is_ruined)
            prog = 1.0 if st.built else float(st.completion())

            if st.kind == "ladder":
                fx.draw_ladder(
                    s,
                    (cam.sx(x0), terrain.ground_y(x0)),
                    (cam.sx(x1), terrain.ground_y(x1)),
                    progress=prog,
                    ruined=ruined,
                )
                return

            # A sample every CROSSING_SAMPLE_PX is finer than the deck's own
            # end ramps and far coarser than per-column, which keeps this at a
            # couple of dozen points however wide the chasm turned out to be.
            n = max(2, min(64, int((x1 - x0) / CROSSING_SAMPLE_PX) + 1))
            xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
            standing = bool(st.built) and not ruined
            # xs are world; the deck's y comes from the world heightmap and its
            # x goes to fx in screen space.
            if standing:
                deck = [(cam.sx(x), terrain.ground_y(x)) for x in xs]
            else:
                deck = [(cam.sx(x), float(st.y)) for x in xs]
            h = terrain.height
            n_h = int(h.shape[0]) - 1
            ground = [float(h[int(max(0, min(n_h, x)))]) for x in xs]
            fx.draw_bridge(s, deck, ground, progress=prog, ruined=ruined)
        except Exception:
            log.debug("crossing draw failed for %s#%s", st.kind, st.id, exc_info=True)

    @staticmethod
    def _occupants(world, st) -> list:
        try:
            return [a for a in world.population.alive_agents()
                    if getattr(a, "inside", None) == st.id]
        except Exception:
            return []

    def _draw_occupancy(self, s: pygame.Surface, st, rect: pygame.Rect,
                        occ: list, world) -> None:
        """A warm doorway and a sleep mark, so an occupied hut reads as lived in.

        This is the whole point of putting sleepers inside: you should still be
        able to tell the difference between an empty hut and one with someone in
        it, without seeing a body draped over the threshold.
        """
        try:
            # Warm light spilling from the doorway. Brighter at night.
            night = bool(getattr(world, "is_night", False))
            glow = 150 if night else 70
            dw, dh = max(4, rect.width // 5), max(5, rect.height // 3)
            door = pygame.Rect(0, 0, dw, dh)
            door.midbottom = (rect.centerx, rect.bottom - 1)
            light = pygame.Surface(door.size, pygame.SRCALPHA)
            light.fill((255, 186, 96, glow))
            s.blit(light, door)

            # A small "z" drifting up, one per sleeper, phase-offset by id.
            t = float(getattr(world, "world_time", 0.0))
            for i, a in enumerate(occ[:3]):
                ph = (t * 0.6 + i * 0.37 + (a.id % 7) * 0.11) % 1.0
                zy = rect.top - 4 - int(ph * 12)
                zx = rect.centerx + 6 + i * 5 + int(3 * math.sin(ph * 6.28))
                alpha = int(200 * (1.0 - ph))
                if alpha <= 0:
                    continue
                col = (*tuple(a.color), alpha)
                mark = pygame.Surface((5, 5), pygame.SRCALPHA)
                pygame.draw.line(mark, col, (0, 0), (4, 0))
                pygame.draw.line(mark, col, (4, 0), (0, 4))
                pygame.draw.line(mark, col, (0, 4), (4, 4))
                s.blit(mark, (zx, zy))
        except Exception:
            pass

    # ------------------------------------------------------------ dragons --
    def _draw_dragons(self, s: pygame.Surface, world, *, grounded: bool) -> None:
        """Draw the dragons that belong at this layer, and only those.

        render/dragons.py ships one entry point and documents it as "call after
        the light composite". That is right for the four kinds that fly and
        wrong for the fifth, and the module's own reasoning says why without
        quite noticing: it argues from *cruise altitude*, and the quadruped
        never has one. The quadruped is the siege dragon, and ``draw_one`` pins
        it to ``ground_y`` for its whole visit precisely because it has no
        flight pose - so drawing it after the composite made it the one object
        in the colony that glows at 3am.

        Measured, seed 4242, scene clear, deep night (ambient 0.34), mean
        luminance of the body's own ink against the world in the same box:

        ==================== ======= ============= =============
        where it stands      world   after (was)   inside (now)
        ==================== ======= ============= =============
        240 px from any fire    10.1   17.3 (1.7x)     7.3 (0.7x)
        in a firepit's pool     33.2   96.7 (2.9x)    64.6 (1.9x)
        ==================== ======= ============= =============

        Unlit it was brighter than the ground it stood on; by a fire it was
        three times the scene. Inside the composite it tracks the world in both
        - still the brightest thing near the fire, which is correct, because it
        is standing in firelight. render/throwing.py names the identical
        failure as its reason for putting thrown spears at step 7; same call.

        Split by KIND and not by altitude, which is the tempting version and is
        wrong: a wyvern stooping to seize somebody crosses the ground band
        twice a visit, so an altitude threshold would carry it across the
        composite mid-dive and pop its brightness on the way through. That is a
        worse artefact than the bug being fixed. The quadruped never leaves the
        ground and nothing else ever settles on it, so this partition never
        moves a dragon between layers while it is on screen. An unrecognised
        kind draws as a wyvern (``dragon_kind_of``) and so lands in the flying
        half, which is exactly where every dragon used to be - a sim that grows
        a sixth kind degrades to the old behaviour rather than to a new one.

        What this deliberately does *not* buy is occlusion by buildings. The
        grounded pass runs after the structures, so a dragon standing behind a
        roof still overdraws it, the same way a villager does. A 268 px siege
        dragon half-hidden by a 92 px hut would read as a clipping bug rather
        than as depth. It runs *before* the agents for the opposite reason:
        that body is wide enough to swallow a dozen people, and the villagers
        it is hunting have to stay visible in front of it.

        Fails soft, like every other pass in here: no dragons, an unreadable
        registry or a raise inside the art costs this frame's dragon, not the
        frame.
        """
        reg = getattr(world, "dragons", None)
        if reg is None:
            return
        try:
            if len(reg) == 0:
                return                  # the case ~every frame of every run
        except TypeError:
            pass                        # no __len__; fall through and walk it
        try:
            # ``alive()`` is the first name render/dragons._iter_dragons probes
            # for, so this and the module's own walk select the same set.
            fn = getattr(reg, "alive", None)
            dragons = fn() if callable(fn) else list(reg)
        except Exception:
            log.debug("could not read the dragon list", exc_info=True)
            return
        # Coerced here rather than five times inside the art, and separately
        # from the list read: a world reloaded from a junk save can hand over
        # anything at all as a clock, and losing the animation phase is not a
        # reason to lose the dragon.
        try:
            t = float(world.world_time)
        except (AttributeError, TypeError, ValueError):
            t = 0.0
        if not math.isfinite(t):
            t = 0.0
        want = dragon_art.KIND_QUADRUPED
        for d in dragons:
            if d is None:
                continue
            try:
                if (dragon_art.dragon_kind_of(d) == want) is not grounded:
                    continue
                dragon_art.draw_one(s, world, d, t, cam=self.camera)
            except Exception:
                log.debug("dragon draw failed", exc_info=True)

    # ------------------------------------------------------------- agents --
    def _draw_agents(self, s: pygame.Surface, world) -> None:
        lighting = world.lighting
        cam = self.camera
        self._label_rects.clear()
        for a in world.population.agents:
            if not a.alive and a.dead_t > 2.0:
                continue
            # Inside a building: they are genuinely in there, not lying across
            # the doorstep. _draw_occupancy marks the hut instead.
            if getattr(a, "inside", None) is not None and a.alive:
                continue
            # The camera follows the densest cluster, so on a split colony this
            # is a real saving rather than a theoretical one: the half across
            # the chasm costs nothing to not draw.
            if not cam.visible(a.x, AGENT_CULL_PAD):
                continue
            try:
                # WORLD coords: lighting.light_at samples the sim's own light
                # sources, which live in the world like everything else in sim/.
                lit = lighting.light_at(a.x, a.y)
            except Exception:
                lit = 1.0
            silhouette = SILHOUETTE_COLOR if lit < SILHOUETTE_CUTOFF else None
            try:
                draw_stickman(s, a, world.world_time, silhouette,
                              cam=self.camera)
            except Exception:
                log.exception("stickman draw failed for %s", getattr(a, "name", "?"))
            if (self.show_names or self.show_activity) and a.alive:
                self._draw_labels(s, a, lit)

    def _draw_labels(self, s: pygame.Surface, a, lit: float) -> None:
        """The plates above a stickman's head: what it is doing, then its name.

        Both plates are placed as one block so the pair move together, and the
        activity sits *above* the name rather than below it. Below reads more
        naturally, but the name's bottom edge is already only ~11 px clear of a
        21 px stickman's scalp, so a second line there would sit on the head.
        Stacking upward keeps the figure clean at every scale.

        Faded with local light so labels do not flatten the night scene - an
        unlit figure gets a dim tag, matching the silhouette it is drawn as.
        """
        try:
            dim = 0.42 + 0.58 * max(0.0, min(1.0, float(lit)))
            plates: list[pygame.Surface] = []
            if self.show_activity:
                act = hud.activity_tag(hud.activity_of(a), dim)
                if act is not None:
                    plates.append(act)
            if self.show_names:
                plates.append(hud.name_tag(str(a.name)[:12], tuple(a.color), dim))
            if not plates:
                return

            gap = 1
            width = max(p.get_width() for p in plates)
            height = sum(p.get_height() for p in plates) + gap * (len(plates) - 1)
            block = pygame.Rect(0, 0, width, height)
            # AGENT_HEIGHT above the feet, plus clearance for the speech bubble.
            # The ONE conversion in this method: everything below - the two
            # RENDER_W clamps, _avoid_overlap, the stored rect - is screen
            # space, and the clamps only keep a plate on screen because the
            # block was anchored off cam.sx(a.x) rather than off a.x.
            block.midbottom = (_ri(self.camera.sx(a.x)), _ri(a.y) - 32)
            if a.speech:
                block.top -= 10
            block.left = max(1, min(block.left, RENDER_W - block.width - 1))
            block.top = max(1, block.top)
            self._avoid_overlap(block)

            y = block.top
            for p in plates:
                r = p.get_rect()
                r.centerx = block.centerx
                r.top = y
                # Re-clamp per plate: they differ in width, so the widest one
                # setting the block's edge does not keep the others on screen.
                r.left = max(1, min(r.left, RENDER_W - r.width - 1))
                s.blit(p, r)
                y += p.get_height() + gap
            self._label_rects.append(block)
        except Exception:
            pass

    def _avoid_overlap(self, block: pygame.Rect) -> None:
        """Nudge *block* upward until it clears the labels already drawn.

        Villagers cluster - around a fire, a build site, a body - and two name
        plates at the same height overlap into an unreadable smear. Lifting the
        later one keeps both legible and still visibly anchored to its owner.
        Bounded: a handful of passes, and never off the top of the screen.
        """
        for _ in range(6):
            hit = block.collidelist(self._label_rects)
            if hit < 0:
                return
            lift = block.bottom - self._label_rects[hit].top + 2
            if lift <= 0 or block.top - lift < 1:
                return
            block.top -= lift

    # --------------------------------------------------------------- lava --
    def _lava_flows(self, world) -> list[tuple[list[tuple[float, float]], float]]:
        """Sample each published flow onto the live ground contour.

        ``events.lava`` carries a vent x, how far the front has advanced and how
        hot it still is; the *shape* has to be read off the heightmap here,
        because the flow lies on the ground and the ground is what the rest of
        the scene keeps changing (ash settles on it, a tremor digs a notch in
        it). Sampling per frame rather than caching for the same reason.

        The live half-width is ``half * hot``, exactly as the sim computes it in
        ``_lava_paint`` - so what is drawn is the span that is actually painted
        MAT_LAVA, and the band cannot drift off the material it belongs to as
        the flow congeals. Fails soft to no lava at all.
        """
        out: list[tuple[list[tuple[float, float]], float]] = []
        try:
            flows = getattr(world.events, "lava", None)
            if not flows:
                return out
            cam = self.camera
            ground = world.terrain.ground_y
            for f in list(flows):
                try:
                    heat = max(0.0, min(1.0, float(f.get("hot", 0.0))))
                    live = max(0.0, float(f.get("half", 0.0)) * heat)
                    if heat <= 0.02 or live < 2.0:
                        continue
                    cx = float(f["x"])
                    n = max(2, min(LAVA_SAMPLES, int(live * 2.0 / LAVA_SAMPLE_PX) + 1))
                    pts = []
                    for i in range(n):
                        x = cx - live + 2.0 * live * (i / (n - 1))
                        # WORLD clamp - x is about to index the heightmap
                        # through ground_y - then converted, because both
                        # consumers of this list (fx.draw_lava_flow and the
                        # lightmap in _composite_light) are screen space.
                        x = max(0.0, min(float(WORLD_W - 1), x))
                        pts.append((cam.sx(x), ground(x) + 1.0))
                    out.append((pts, heat))
                except Exception:
                    continue
        except Exception:
            log.debug("lava sampling failed", exc_info=True)
        return out

    # -------------------------------------------------------------- light --
    def _composite_light(self, s: pygame.Surface, world,
                         lava: list[tuple[list[tuple[float, float]], float]] | None = None) -> None:
        lighting = world.lighting
        try:
            amb = float(lighting.ambient(world.events.scene, world.world_time))
        except Exception:
            amb = 0.6
        amb = max(0.0, min(1.0, amb))

        lm = self.lightmap
        base = int(amb * 255)
        lm.fill((base, base, base))

        for src in getattr(lighting, "sources", ()):
            try:
                # Flicker is folded into intensity here rather than passed
                # through, so fx keeps a small cache key and the sprite cache
                # still hits. Quantised to 12 steps for the same reason.
                flick = 1.0
                if src.flicker:
                    phase = world.world_time * 9.7 + (src.owner_id or 0) * 1.7
                    wob = (math.sin(phase) * 0.6 + math.sin(phase * 2.3) * 0.4)
                    flick = 1.0 + src.flicker * wob * 0.5
                inten = round(max(0.05, min(1.6, src.intensity * flick)), 2)
                spr = fx.radial_light_surface(src.radius, src.color, inten)
            except Exception:
                continue
            if spr is None:
                continue
            r = spr.get_rect()
            # The lightmap is a frame-sized surface, so a light source's world
            # x has to be converted like everything else. Not culled: a source
            # just off the left edge still spills 190 px of glow into the
            # frame, and pygame clips the blit for free.
            r.center = (_ri(self.camera.sx(src.x)), _ri(src.y))
            lm.blit(spr, r, special_flags=pygame.BLEND_RGB_ADD)

        # A flow is a light source the size of a street, and it belongs in the
        # lightmap rather than only in the emissive pass: what sells an eruption
        # at night is not the orange band, it is the colony, the trees and the
        # hillside behind them all picked out in red from below. Added here and
        # not through lighting.sources because those are simulation state, and
        # the sim has no business carrying a hundred px of derived geometry the
        # renderer can sample off the heightmap for free.
        for pts, heat in (lava or ()):
            step = max(1, len(pts) // LAVA_LIGHTS)
            for i in range(0, len(pts), step):
                try:
                    spr = fx.radial_light_surface(
                        LAVA_LIGHT_R, LAVA_LIGHT_COLOR, 0.85 * heat)
                except Exception:
                    break
                r = spr.get_rect()
                r.center = (_ri(pts[i][0]), _ri(pts[i][1]))
                lm.blit(spr, r, special_flags=pygame.BLEND_RGB_ADD)

        s.blit(lm, (0, 0), special_flags=pygame.BLEND_RGB_MULT)

    def _draw_lightning(self, s: pygame.Surface, world) -> None:
        strikes = getattr(world.events, "strikes", None)
        if not strikes:
            return
        cam = self.camera
        for st in list(strikes):
            try:
                # The bolt is generated straight in screen space: it is a
                # jagged line between two points and has nothing to sample off
                # the world. ground_y came off the heightmap in sim/ and is a
                # y, which the camera does not touch.
                bx = cam.sx(float(st["x"]))
                bolt = fx.make_lightning_bolt(
                    bx, -20.0, bx, st["ground_y"], st["seed"])
                fx.draw_lightning(s, bolt, st["t"], 0.5)
            except Exception:
                continue

    def _draw_heat_shimmer(self, s: pygame.Surface, world, heat: float) -> None:
        """Boil the ground band, scaled by how deep the drought has got.

        Driven off the published ``events.heat`` channel rather than a scene
        name test, for the same reason the eclipse ring and the haze wash are:
        the sim decides how hot it is, the renderer only decides what that looks
        like, and the two cannot drift apart. Every other scene leaves the
        channel at zero, so this is a no-op everywhere else.

        Fails soft and silently - a frame without shimmer is a frame that still
        gets drawn, which on a wallpaper that runs unattended for hours is worth
        far more than the effect is.
        """
        try:
            # Only the ground that is ON SCREEN sets the band. Taking the min
            # and max over the whole WORLD_W heightmap would size the effect
            # off a cliff 4000 px away that nobody can see, and on a map with
            # real relief that pins the band at SHIMMER_MAX_H permanently -
            # i.e. the height of the shimmer would stop tracking the ground it
            # is supposed to be boiling.
            h = world.terrain.height
            n = int(h.shape[0])
            lo = max(0, min(n - 1, int(self.camera.x)))
            hi = max(lo + 1, min(n, lo + RENDER_W))
            win = h[lo:hi]
            y0 = int(float(np.min(win))) - SHIMMER_RISE
            y1 = int(float(np.max(win))) + SHIMMER_BELOW
            y0 = max(0, y0)
            y1 = min(RENDER_H, y1)
            if y1 - y0 > SHIMMER_MAX_H:
                # Trim from the top: the interesting half is the ground line.
                y0 = y1 - SHIMMER_MAX_H
            if y1 - y0 < 16:
                return
            amount = SHIMMER_MIN_PX + (SHIMMER_MAX_PX - SHIMMER_MIN_PX) * min(1.0, heat)
            fx.draw_heat_shimmer(s, (0, y0, RENDER_W, y1 - y0),
                                 float(world.world_time), amount, band=5)
        except Exception:
            log.debug("heat shimmer failed", exc_info=True)

    def _draw_water(self, s: pygame.Surface, world, level: float) -> None:
        """The flood surface, from ``events.water_level`` down to the frame edge.

        ``level`` is a sim number that is about to become a SURFACE SIZE, which
        is the one thing here worse than a bad Rect: ``RENDER_H - level`` at
        level = -1e308 asks pygame for a 1600 x 10^308 surface. The clamp is the
        same one every coordinate in this file gets, and then the level is
        pinned into the frame - water above the top edge is simply a full frame
        of water, which is what an unclamped draw would have shown anyway.
        """
        lv = _ri(level)
        if lv >= RENDER_H:
            return
        lv = max(0, lv)
        water = pygame.Surface((RENDER_W, RENDER_H - lv), pygame.SRCALPHA)
        water.fill((30, 70, 120, 150))
        s.blit(water, (0, lv))
        pygame.draw.line(s, (120, 170, 220), (0, lv), (RENDER_W, lv), 2)

    # ------------------------------------------------------------ weather --
    def _emit_weather(self, world, dt: float,
                      lava: list[tuple[list[tuple[float, float]], float]] | None = None) -> None:
        """Seed this frame's weather and fire particles, at a rate PER SECOND.

        Every count in here used to be per FRAME, and that is the whole of a
        measured bug: freeze the world and call ``draw(world, 0.0)`` sixty times
        and all sixty frames differ from the first, because a firepit spawns one
        ember and the night spawns one firefly on every call whether or not any
        time has passed. The pool grew by exactly one particle per frame with
        dt = 0 - the signature this was found by - and with it suppressed the
        residual is 0 differing frames out of 60, at cam.x 0 and at 3200.

        The visible consequence is not the frozen case, it is the moving one:
        the density of fire and snow was a function of how fast the machine
        happened to be compositing. A firepit threw 60 embers a second at 60 fps
        and 25 at 25 fps, so a busy desktop quietly put the fires out. The two
        emitters below that already knew this - ``_emit_grit`` and
        ``_emit_lava_sparks``, whose docstrings both spell out that a per-frame
        count would put sixteen times as many particles in the preview as on the
        wallpaper path - were right, and the rest of this method was not.

        The rates are the old per-frame counts times 60, so the look at 60 fps
        is the one that was there before, to the particle; anywhere else it is
        now the same look rather than a thinner one.

        ``_rate`` carries the leftover fraction between frames rather than
        rounding it away, which is what makes a 60/s emitter still emit at 240
        fps (0.25 of a particle a frame, one every fourth) instead of truncating
        to nothing.

        ``rain`` was the one emitter left out of that when the rest moved over,
        on the grounds that it was already dt-driven and that changing the look
        of light rain was not that change's business. It was: being dt-driven is
        not enough on its own, and the truncation made LIGHT RAIN INVISIBLE.
        ``int(90 * ev.rain * step)`` at 60 fps is ``int(1.5 * ev.rain)``, which
        is zero for every ``ev.rain`` below 0.67 - so two thirds of the rain
        range emitted not one drop, forever, while the sky darkened and the
        ground got wet around it. Below 0.014 it was empty even on the 4 Hz
        wallpaper path. With the carry it emits at 90/s at any framerate and any
        intensity, and heavy rain at 60 fps gets 50% denser than it used to be
        because 90/s is what it always asked for and 60/s is what truncation
        gave it - the wallpaper path was already showing the honest density, so
        this makes the preview agree with it rather than the other way round.

        The accumulator dict is rebuilt each frame from the keys that were
        actually used, so it holds exactly the live emitters - it cannot grow
        with the number of firepits that have ever existed.
        """
        ev = world.events
        scene = ev.scene
        try:
            step = min(0.25, max(0.0, float(dt)))
        except (TypeError, ValueError):
            step = 0.0
        prev, acc = self._emit_acc, {}

        def rate(key, per_sec: float) -> int:
            """How many particles *key* has earned this frame.

            The bound is not tuning, it is the same lesson as _ri: ``ev.snow``
            comes off a save, ``2400.0 * 1e308`` is ``inf``, and ``int(inf)``
            raises OverflowError - which the per-frame counts this replaced did
            not, because they went straight to emit() and were capped there. A
            rate this file computes must not be a new way to lose the frame.
            Everything real is under 3000/s and the pool's own per-kind cap is
            1400, so the ceiling is unreachable by anything legitimate.
            """
            v = prev.get(key, 0.0) + per_sec * step
            if not 0.0 <= v <= 100000.0:          # NaN, inf, or a poisoned channel
                v = 100000.0 if v > 0.0 else 0.0
            n = int(v)
            acc[key] = v - n
            return n

        try:
            if lava:
                self._emit_lava_sparks(lava, step, rate)
            if getattr(ev, "rain", 0) > 0.01:
                self.particles.emit("rain", rate("rain", 90.0 * ev.rain),
                                    wind=ev.wind)
            if getattr(ev, "snow", 0) > 0.01:
                self.particles.emit("snow", rate("snow", 2400.0 * ev.snow),
                                    wind=ev.wind)
            if getattr(ev, "ash", 0) > 0.01:
                self.particles.emit("ash", rate("ash", 1500.0 * ev.ash),
                                    wind=ev.wind)
            if scene == SCENE_SANDSTORM:
                self._emit_grit(ev, step)
            if scene == SCENE_WILDFIRE or world.props.burning():
                for p in world.props.burning():
                    self.particles.emit("ember", rate(("burn_e", p.id), 120.0),
                                        x=p.x, y=p.y - 10)
                    self.particles.emit("smoke", rate(("burn_s", p.id), 60.0),
                                        x=p.x, y=p.y - 24)
            for st in world.structures.all():
                if not st.state.get("lit"):
                    continue
                # A bonfire throws a column of sparks rather than a firepit's
                # single ember. It shares the pool's "ember" budget with burning
                # props on purpose - a wildfire and a bonfire at once should
                # compete for the same 500 particles, not double them.
                if getattr(st, "bonfire_active", False):
                    self.particles.emit("ember", rate(("fire_e", st.id), 300.0),
                                        x=(st.x - 13.0, st.x + 13.0),
                                        y=(st.y - 30.0, st.y - 12.0),
                                        vy=(-150.0, -55.0), life=(0.9, 2.4))
                    self.particles.emit("smoke", rate(("fire_s", st.id), 120.0),
                                        x=(st.x - 9.0, st.x + 9.0),
                                        y=st.y - 44.0)
                else:
                    self.particles.emit("ember", rate(("fire_e", st.id), 60.0),
                                        x=st.x, y=st.y - 8)
            if not world.is_night:
                return
            if scene not in (SCENE_BLIZZARD, SCENE_FLOOD) and getattr(ev, "rain", 0) < 0.1:
                self.particles.emit("firefly", rate("firefly", 60.0))
        finally:
            self._emit_acc = acc

    def _emit_lava_sparks(self, lava, dt: float, rate) -> None:
        """Throw sparks and smoke off the flow.

        Deliberately dt-scaled: this runs at whatever rate the app happens to be
        compositing at, and a per-frame count would put sixteen times as many
        sparks in a 60 fps preview as on a machine drawing at four. When this was
        written it and the sandstorm's grit were the only two emitters that knew
        that; :meth:`_emit_weather` now applies the same rule to all of them.

        Only three launch points per flow, spread across it, rather than one per
        sampled column: an emit() call is a numpy round trip and the particles
        spread themselves out anyway once the ember kind's negative gravity has
        carried them up. Read-only and fails soft - a full pool or a missing
        channel just means no sparks this frame.

        The smoke puff below used to be the one literal frame counter in this
        file - ``self._frame % 3`` - which is 20 puffs a second at 60 fps and
        ten at 30. It now goes through the same per-second *rate* accumulator
        as everything else in _emit_weather, at the 20/s that ``% 3`` meant on
        the machine it was tuned on.
        """
        try:
            step = max(0.0, min(0.25, float(dt)))
            if step <= 0.0:
                return
            for f_i, (pts, heat) in enumerate(lava):
                n = int(LAVA_EMBERS_PER_SEC * heat * step)
                if n <= 0 or len(pts) < 2:
                    continue
                for i in (0, len(pts) // 2, len(pts) - 1):
                    x, y = pts[i]
                    # Palette 17 is the lava tone; embers are drawn additively,
                    # so they read as sparks rather than as orange confetti.
                    self.particles.emit("ember", max(1, n // 3),
                                        x=(x - 14.0, x + 14.0), y=(y - 6.0, y),
                                        vx=(-26.0, 26.0), vy=(-120.0, -40.0),
                                        life=(0.7, 2.1), size=(0.7, 2.0), color=17)
                puffs = rate(("lava_s", f_i), 20.0)
                if puffs > 0:
                    mid = pts[len(pts) // 2]
                    self.particles.emit("smoke", puffs,
                                        x=(mid[0] - 30.0, mid[0] + 30.0),
                                        y=(mid[1] - 20.0, mid[1]))
        except Exception:
            log.debug("lava spark emit failed", exc_info=True)

    def _emit_grit(self, ev, dt: float) -> None:
        """Drive sand across the frame along the sandstorm's signed wind.

        The "dust" kind is reused rather than a new particle type being added:
        it already has the physics grit wants - heavy, high drag, and it dies on
        contact with the ground instead of settling into the permanent litter
        that snow and ash leave. So all this has to supply is a launch and a
        horizontal speed the drag can eat.

        Seeded across the *whole* width rather than off the upwind edge, which
        is the obvious way to write it and is wrong: the drag bleeds a grain's
        speed off in about a second, so an edge spawn only ever puts grit in the
        upwind third of the frame and the rest of the map stays suspiciously
        calm. Spawning everywhere and letting each grain streak a few hundred px
        gives a field that is moving uniformly, which is what a storm looks like.

        Deliberately dt-scaled, unlike the snow and ash emitters above it: this
        runs at whatever rate the app is compositing, and a per-frame count
        would put sixteen times as much grit in the preview window as on the
        4 Hz wallpaper push. Read-only and fails soft - a missing channel or a
        full particle pool just means no grit this frame.
        """
        try:
            wind = float(getattr(ev, "wind", 0.0) or 0.0)
            haze = float(getattr(ev, "fog", 0.0) or 0.0)
            if haze <= 0.05 or abs(wind) < 0.05:
                return                  # still ramping in; nothing to blow yet
            # Grains per second at full haze. The pool caps "dust" at 400 and
            # mining/impact dust draws on the same budget, so this is set to
            # settle around 330 rather than to saturate: measured over 240
            # frames, 620/s holds 247 grains, 900/s holds ~330 and 1100/s pins
            # the cap at 387, from which point the extra emit calls buy nothing
            # and only starve the diggers of their dust.
            n = int(900.0 * haze * max(0.0, min(0.25, float(dt))))
            if n <= 0:
                return
            sign = 1.0 if wind >= 0.0 else -1.0
            # Fast enough to read as driven, and _DRAG bleeds it off over ~1 s
            # so the grains slow into the murk rather than crossing in a
            # straight line. sorted() because ParticleSystem._spread hands a
            # (lo, hi) pair to Generator.uniform, which *raises* on hi < lo -
            # and emit() swallows that, so a leftward wind silently emitted
            # nothing at all until the ordering was forced.
            speed = 520.0 + 640.0 * min(1.0, abs(wind))
            vx = tuple(sorted((sign * speed * 0.55, sign * speed)))
            # Split across the two weather layers by hand: "dust" is not in the
            # particle system's own _LAYERED set, so without this every grain
            # would draw in front of the colony and the depth would be flat.
            # Palette 9 is the pale dust and 10 the dark: the distant grains get
            # the pale one so they sink into the tan wash, the near ones the
            # dark so they read as individual grains crossing the frame.
            back = max(1, int(n * 0.45))
            for count, is_back, tone in ((back, True, 9), (n - back, False, 10)):
                if count <= 0:
                    continue
                # VIEW-relative, and slid onto the camera: a sandstorm is a
                # thing you see, and grit seeded across all of WORLD_W would
                # put three quarters of the pool in empty desert and thin the
                # visible storm to a quarter at four times the cost.
                vx0 = self.camera.x
                self.particles.emit(
                    "dust", count, x=(vx0 - 40.0, vx0 + float(RENDER_W) + 40.0),
                    y=(RENDER_H * 0.12, RENDER_H * 0.80),
                    vx=vx, vy=(-45.0, 45.0), life=(0.65, 1.7),
                    size=(0.5, 1.9), color=tone, back=is_back)
        except Exception:
            log.debug("grit emit failed", exc_info=True)
