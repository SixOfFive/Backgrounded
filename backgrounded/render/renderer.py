"""Renderer - composites the world into a single surface.

Strictly read-only with respect to simulation state. The layer order here is
the one defined in docs/ARCHITECTURE.md section 4 and the light composite at
step 10 is the whole visual thesis: draw the world lit, then multiply it down
to near-black, then add light back from candles, fires and lightning.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pygame

from ..constants import (
    BONFIRE_DRAW_SCALE, MATERIAL_COLORS, MAT_GRASS, RENDER_H, RENDER_SIZE,
    RENDER_W, SCENE_BLIZZARD, SCENE_FLOOD, SCENE_SANDSTORM, SCENE_WILDFIRE,
)
from ..sim.structures import (
    BONFIRE_FADE_UNITS, BONFIRE_RAMP_SEC, CROSSING_KINDS,
)
from . import creatures, fx, hud, sky
from .atlas import Atlas
from .particles import ParticleSystem
from .stickfigure import draw_stickman

log = logging.getLogger(__name__)

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


class Renderer:
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
        self._terrain_cache: pygame.Surface | None = None
        self._terrain_fingerprint: tuple | None = None
        # Label plates already placed this frame, so two villagers standing on
        # top of each other do not stack their status text into a smear.
        self._label_rects: list[pygame.Rect] = []
        self._frame = 0
        # Last screen-shake offset baked into the returned frame, so draw_hud
        # can match it on the wallpaper path.
        self._shake: tuple[int, int] = (0, 0)
        # (disabled subsystem names, hud scale, rendered plate) - see
        # _fault_banner. The scale is part of the key, not just the content.
        self._fault_plate: tuple[tuple[str, ...], float, pygame.Surface] | None = None

    # ------------------------------------------------------------- public --
    def draw(self, world, dt: float) -> pygame.Surface:
        self._frame += 1
        s = self.scene
        ev = world.events

        # Sampled once, up front, because three different passes want it: the
        # ember emitter below, the lightmap (a flow lights what is around it)
        # and the emissive band drawn after the composite. Empty in every scene
        # but the eruption - and for the minute afterwards, while the flow the
        # eruption left behind is still cooling.
        lava = self._lava_flows(world)

        try:
            self._emit_weather(world, dt, lava)
            self.particles.update(dt, world.terrain, ev)
        except Exception:
            log.exception("particle update failed")

        # 1-2. sky and parallax ridges
        sky.draw_sky(s, ev.scene, world.world_time, ev, world.lighting)
        sky.draw_parallax(s, world.seed, ev.scene, world.lighting)

        # 3. distant weather
        self.particles.draw(s, layer="back")

        # 4. terrain, then the scars an earthquake has torn in it. The scars go
        # here and not later because they are *ground*: they must sit under
        # everything that stands on the terrain, and the light composite must
        # dim them along with the rest of the world.
        s.blit(self._terrain_surface(world.terrain), (0, 0))
        self._draw_fissures(s, world)

        # 5-6. props then structures, painter-ordered by x for a little depth
        self._draw_props(s, world)
        self._draw_structures(s, world)

        # 7. agents, then the wildlife among them
        self._draw_agents(s, world)
        try:
            creatures.draw_animals(s, world, world.world_time)
        except Exception:
            log.exception("animal draw failed")
        try:
            creatures.draw_mining_dust(s, world, world.world_time)
        except Exception:
            log.exception("mining dust draw failed")

        # 8-9. near weather and particles
        self.particles.draw(s, layer="front")

        # flood water sits above the terrain but below the light pass
        if getattr(ev, "water_level", None):
            self._draw_water(s, world, ev.water_level)

        # 10. the light composite
        self._composite_light(s, world, lava)

        # Lava, immediately after it. The terrain pass has already painted the
        # flat MAT_LAVA band under everything, which is right - it is ground -
        # but the composite has just multiplied it down by an ambient this same
        # scene crushed to a quarter. This puts the flow's own light back on top
        # of the lit world, which is what makes it the brightest thing on screen
        # instead of a dark orange stripe.
        if lava:
            fx.draw_lava_flow(s, lava, world.world_time)

        # Heat haze over the ground, after the composite because it displaces
        # what you can *see* - lit pixels, not the world's own geometry - and
        # before the UFO, the lightning and the HUD, none of which are things
        # the air between you and the ground is in front of.
        heat = float(getattr(ev, "heat", 0.0) or 0.0)
        if heat > 0.02:
            self._draw_heat_shimmer(s, world, heat)

        # The saucer sits above the light pass: its beam is its own light and
        # should not be multiplied down into the night.
        try:
            creatures.draw_ufo(s, world, world.world_time)
        except Exception:
            log.exception("ufo draw failed")

        # 11. lightning geometry, then vignette
        self._draw_lightning(s, world)
        fx.draw_vignette(s)

        # The eclipse's horizon ring, for the same reason and at the same layer
        # as the fog wash: it is light coming in under the shadow from outside
        # it, not part of the lit world, so the light composite must not have a
        # say in it. Above the vignette so the corners keep their glow.
        ecl = float(getattr(ev, "eclipse", 0.0) or 0.0)
        if ecl > 0.02:
            fx.draw_eclipse_twilight(s, ecl)

        # The haze wash sits above the whole lit world (so it dims it rather
        # than being multiplied away), but below the HUD, which must stay
        # readable. Density and colour both come off the sim: the scene decides
        # whether this is a grey fog or a tan sandstorm, not the renderer.
        fog = float(getattr(ev, "fog", 0.0) or 0.0)
        if fog > 0.02:
            # Hand it the daylight level too, or the wash - being painted over
            # the finished frame - becomes the one part of the scene the
            # day/night cycle cannot reach, and a 2am fog reads as noon.
            try:
                dl = sky.daylight(sky.day_phase(world.world_time, world.lighting))
            except Exception:
                dl = 1.0
            fx.draw_fog_overlay(s, fog, world.world_time,
                                getattr(ev, "fog_color", None), dl)

        # 12. the HUD used to be blitted here, after the light composite so it
        # stays readable in a black scene. It is drawn by :meth:`draw_hud`
        # instead, because *where* it belongs depends on who is looking: the
        # wallpaper is a bare image and needs it baked in, while the preview
        # needs it on the window surface or zooming drags the panels off with
        # the picture. app.py calls draw_hud once per output; see it for the
        # ordering that keeps this to a single render.

        # screen shake: blit the composed scene at an offset
        dx, dy = ev.shake_offset() if hasattr(ev, "shake_offset") else (0, 0)
        # Remembered so draw_hud can re-apply it on the wallpaper path. The HUD
        # was drawn before this blit and therefore shook with the world; that is
        # the wallpaper's long-standing look and is preserved deliberately.
        self._shake = (int(dx), int(dy))
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
                r.midtop = (surf.get_width() // 2 + int(off[0]), 12 + int(off[1]))
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
    def _terrain_surface(self, terrain) -> pygame.Surface:
        """Terrain only changes on deformation, so cache the drawn surface and
        rebuild only when a cheap fingerprint of the heightmap moves."""
        fp = (float(terrain.height[::64].sum()), int(terrain.material[::64].sum()))
        if self._terrain_cache is not None and fp == self._terrain_fingerprint:
            return self._terrain_cache

        surf = pygame.Surface(RENDER_SIZE, pygame.SRCALPHA)
        h = terrain.height
        mats = terrain.material

        # Solid body: one polygon down to the bottom of the screen.
        pts = [(0, RENDER_H)]
        pts += [(x, float(h[x])) for x in range(RENDER_W)]
        pts += [(RENDER_W - 1, RENDER_H)]
        pygame.draw.polygon(surf, (38, 34, 32), pts)

        # Surface band coloured per material, drawn as a run-length set of
        # rects rather than 1280 individual lines.
        band = 16
        x = 0
        while x < RENDER_W:
            m = int(mats[x])
            x2 = x
            while x2 < RENDER_W and int(mats[x2]) == m:
                x2 += 1
            col = MATERIAL_COLORS.get(m, MATERIAL_COLORS[MAT_GRASS])
            # Both edges of the band follow the contour. Closing the polygon on
            # two corner points instead - which is what this did - only draws a
            # band where the run is flat: over a hill it fills the whole wedge
            # between the skyline and a straight line joining the run's ends.
            # It was invisible while every material was a muted earth tone
            # against a near-black body, and became a lit orange tent the first
            # time a volcanic flow painted a 300 px run of MAT_LAVA across a
            # ridge. Two points per column rather than one, on a surface that is
            # cached until the heightmap moves.
            seg = [(xi, float(h[xi])) for xi in range(x, x2)]
            seg += [(xi, float(h[xi]) + band) for xi in range(x2 - 1, x - 1, -1)]
            if len(seg) > 2:
                pygame.draw.polygon(surf, col, seg)
            x = x2

        # Rim highlight so the silhouette still reads in near-darkness.
        rim = [(xi, float(h[xi])) for xi in range(RENDER_W)]
        pygame.draw.lines(surf, (96, 104, 96), False, rim, 2)

        self._terrain_cache = surf.convert_alpha()
        self._terrain_fingerprint = fp
        return self._terrain_cache

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
        cracks: list[tuple[float, float, float, float, int]] = []
        for f in list(fissures):
            try:
                depth = float(f.get("depth", 0.0))
                if depth < 2.0:
                    continue            # not open enough to read as a hole yet
                x = float(f["x"])
                cracks.append((x, world.terrain.ground_y(x),
                               float(f.get("half", 12.0)) * 1.05, depth,
                               int(f.get("seed", 1))))
            except Exception:
                continue
        fx.draw_fissures(s, cracks)

    # -------------------------------------------------------------- props --
    def _draw_props(self, s: pygame.Surface, world) -> None:
        for p in sorted(world.props.all(), key=lambda p: p.y):
            if not p.alive:
                continue
            # Litter is drawn here rather than baked into the atlas: it is four
            # 9x6 px silhouettes with no stages, no variants worth baking and no
            # growth ladder, and the atlas would answer with its 2x2 transparent
            # "missing" surface for an unknown kind - i.e. the feature would be
            # invisible while looking perfectly wired up.
            if p.kind == "litter":
                try:
                    st = p.state if isinstance(p.state, dict) else {}
                    fx.draw_litter(s, p.x, p.y, int(st.get("shape", 0) or 0),
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
            r.midbottom = (int(p.x), int(p.y) + 2)
            s.blit(spr, r)

    def _draw_structures(self, s: pygame.Surface, world) -> None:
        for st in sorted(world.structures.all(), key=lambda st: st.y):
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
            r.midbottom = (int(st.x), int(st.y) + 2)
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
            fx.draw_bonfire(s, float(st.x), float(st.y) + 1.0,
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
            x0, x1 = span
            x0 = max(0.0, min(float(x0), RENDER_W - 1.0))
            x1 = max(0.0, min(float(x1), RENDER_W - 1.0))
            if x1 - x0 < 4.0:
                return
            ruined = bool(st.is_ruined)
            prog = 1.0 if st.built else float(st.completion())

            if st.kind == "ladder":
                fx.draw_ladder(
                    s,
                    (x0, terrain.ground_y(x0)),
                    (x1, terrain.ground_y(x1)),
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
            if standing:
                deck = [(x, terrain.ground_y(x)) for x in xs]
            else:
                deck = [(x, float(st.y)) for x in xs]
            h = terrain.height
            ground = [float(h[int(max(0, min(RENDER_W - 1, x)))]) for x in xs]
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

    # ------------------------------------------------------------- agents --
    def _draw_agents(self, s: pygame.Surface, world) -> None:
        lighting = world.lighting
        self._label_rects.clear()
        for a in world.population.agents:
            if not a.alive and a.dead_t > 2.0:
                continue
            # Inside a building: they are genuinely in there, not lying across
            # the doorstep. _draw_occupancy marks the hut instead.
            if getattr(a, "inside", None) is not None and a.alive:
                continue
            try:
                lit = lighting.light_at(a.x, a.y)
            except Exception:
                lit = 1.0
            silhouette = SILHOUETTE_COLOR if lit < SILHOUETTE_CUTOFF else None
            try:
                draw_stickman(s, a, world.world_time, alpha_color=silhouette)
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
            # AGENT_HEIGHT above the feet, plus clearance for the speech bubble
            block.midbottom = (int(a.x), int(a.y) - 32)
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
                        x = max(0.0, min(float(RENDER_W - 1), x))
                        pts.append((x, ground(x) + 1.0))
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
            r.center = (int(src.x), int(src.y))
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
                r.center = (int(pts[i][0]), int(pts[i][1]))
                lm.blit(spr, r, special_flags=pygame.BLEND_RGB_ADD)

        s.blit(lm, (0, 0), special_flags=pygame.BLEND_RGB_MULT)

    def _draw_lightning(self, s: pygame.Surface, world) -> None:
        strikes = getattr(world.events, "strikes", None)
        if not strikes:
            return
        for st in list(strikes):
            try:
                bolt = fx.make_lightning_bolt(
                    st["x"], -20.0, st["x"], st["ground_y"], st["seed"])
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
            h = world.terrain.height
            y0 = int(float(np.min(h))) - SHIMMER_RISE
            y1 = int(float(np.max(h))) + SHIMMER_BELOW
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
        if level >= RENDER_H:
            return
        h = int(RENDER_H - level)
        water = pygame.Surface((RENDER_W, h), pygame.SRCALPHA)
        water.fill((30, 70, 120, 150))
        s.blit(water, (0, int(level)))
        pygame.draw.line(s, (120, 170, 220), (0, int(level)), (RENDER_W, int(level)), 2)

    # ------------------------------------------------------------ weather --
    def _emit_weather(self, world, dt: float,
                      lava: list[tuple[list[tuple[float, float]], float]] | None = None) -> None:
        ev = world.events
        scene = ev.scene
        if lava:
            self._emit_lava_sparks(lava, dt)
        if getattr(ev, "rain", 0) > 0.01:
            self.particles.emit("rain", int(90 * ev.rain * dt * 60 / 60),
                                wind=ev.wind)
        if getattr(ev, "snow", 0) > 0.01:
            self.particles.emit("snow", int(40 * ev.snow), wind=ev.wind)
        if getattr(ev, "ash", 0) > 0.01:
            self.particles.emit("ash", int(25 * ev.ash), wind=ev.wind)
        if scene == SCENE_SANDSTORM:
            self._emit_grit(ev, dt)
        if scene == SCENE_WILDFIRE or world.props.burning():
            for p in world.props.burning():
                self.particles.emit("ember", 2, x=p.x, y=p.y - 10)
                self.particles.emit("smoke", 1, x=p.x, y=p.y - 24)
        for st in world.structures.all():
            if not st.state.get("lit"):
                continue
            # A bonfire throws a column of sparks rather than a firepit's single
            # ember. It shares the pool's "ember" budget with burning props on
            # purpose - a wildfire and a bonfire at once should compete for the
            # same 500 particles, not double them.
            if getattr(st, "bonfire_active", False):
                self.particles.emit("ember", 5, x=(st.x - 13.0, st.x + 13.0),
                                    y=(st.y - 30.0, st.y - 12.0),
                                    vy=(-150.0, -55.0), life=(0.9, 2.4))
                self.particles.emit("smoke", 2, x=(st.x - 9.0, st.x + 9.0),
                                    y=st.y - 44.0)
            else:
                self.particles.emit("ember", 1, x=st.x, y=st.y - 8)
        if not world.is_night:
            return
        if scene not in (SCENE_BLIZZARD, SCENE_FLOOD) and getattr(ev, "rain", 0) < 0.1:
            self.particles.emit("firefly", 1)

    def _emit_lava_sparks(self, lava, dt: float) -> None:
        """Throw sparks and smoke off the flow.

        Deliberately dt-scaled, like the sandstorm's grit and unlike the rain
        and snow emitters above: this runs at whatever rate the app happens to
        be compositing at, and a per-frame count would put sixteen times as many
        sparks in the preview window as on the 4 Hz wallpaper push.

        Only three launch points per flow, spread across it, rather than one per
        sampled column: an emit() call is a numpy round trip and the particles
        spread themselves out anyway once the ember kind's negative gravity has
        carried them up. Read-only and fails soft - a full pool or a missing
        channel just means no sparks this frame.
        """
        try:
            step = max(0.0, min(0.25, float(dt)))
            if step <= 0.0:
                return
            for pts, heat in lava:
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
                if self._frame % 3 == 0:
                    mid = pts[len(pts) // 2]
                    self.particles.emit("smoke", 1, x=(mid[0] - 30.0, mid[0] + 30.0),
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
                self.particles.emit(
                    "dust", count, x=(-40.0, float(RENDER_W) + 40.0),
                    y=(RENDER_H * 0.12, RENDER_H * 0.80),
                    vx=vx, vy=(-45.0, 45.0), life=(0.65, 1.7),
                    size=(0.5, 1.9), color=tone, back=is_back)
        except Exception:
            log.debug("grit emit failed", exc_info=True)
