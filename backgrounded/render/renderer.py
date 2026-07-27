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
    MATERIAL_COLORS, MAT_GRASS, RENDER_H, RENDER_SIZE, RENDER_W,
    SCENE_BLIZZARD, SCENE_FLOOD, SCENE_WILDFIRE,
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
        self._terrain_cache: pygame.Surface | None = None
        self._terrain_fingerprint: tuple | None = None
        self._frame = 0

    # ------------------------------------------------------------- public --
    def draw(self, world, dt: float) -> pygame.Surface:
        self._frame += 1
        s = self.scene
        ev = world.events

        try:
            self._emit_weather(world, dt)
            self.particles.update(dt, world.terrain, ev)
        except Exception:
            log.exception("particle update failed")

        # 1-2. sky and parallax ridges
        sky.draw_sky(s, ev.scene, world.world_time, ev, world.lighting)
        sky.draw_parallax(s, world.seed, ev.scene, world.lighting)

        # 3. distant weather
        self.particles.draw(s, layer="back")

        # 4. terrain
        s.blit(self._terrain_surface(world.terrain), (0, 0))

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
        self._composite_light(s, world)

        # The saucer sits above the light pass: its beam is its own light and
        # should not be multiplied down into the night.
        try:
            creatures.draw_ufo(s, world, world.world_time)
        except Exception:
            log.exception("ufo draw failed")

        # 11. lightning geometry, then vignette
        self._draw_lightning(s, world)
        fx.draw_vignette(s)

        # 12. stats panel. Deliberately after the light composite: it is UI,
        # not part of the world, so it must stay readable in a black scene.
        if self.show_stats:
            hud.draw_stats(s, world, show_roster=self.show_roster)
        if self.show_log:
            hud.draw_log(s, world)

        # screen shake: blit the composed scene at an offset
        dx, dy = ev.shake_offset() if hasattr(ev, "shake_offset") else (0, 0)
        if dx or dy:
            self.out.fill((0, 0, 0))
            self.out.blit(s, (int(dx), int(dy)))
            return self.out
        return s

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
            seg = [(x, float(h[x]))]
            seg += [(xi, float(h[xi])) for xi in range(x, x2)]
            seg += [(x2 - 1, float(h[x2 - 1]) + band), (x, float(h[x]) + band)]
            if len(seg) > 2:
                pygame.draw.polygon(surf, col, seg)
            x = x2

        # Rim highlight so the silhouette still reads in near-darkness.
        rim = [(xi, float(h[xi])) for xi in range(RENDER_W)]
        pygame.draw.lines(surf, (96, 104, 96), False, rim, 2)

        self._terrain_cache = surf.convert_alpha()
        self._terrain_fingerprint = fp
        return self._terrain_cache

    # -------------------------------------------------------------- props --
    def _draw_props(self, s: pygame.Surface, world) -> None:
        for p in sorted(world.props.all(), key=lambda p: p.y):
            if not p.alive:
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
            try:
                spr = self.atlas.get(st.kind, st.variant, stage=st.stage)
            except Exception:
                continue
            if spr is None:
                continue
            r = spr.get_rect()
            r.midbottom = (int(st.x), int(st.y) + 2)
            s.blit(spr, r)
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
            if self.show_names and a.alive:
                self._draw_name(s, a, lit)

    def _draw_name(self, s: pygame.Surface, a, lit: float) -> None:
        """Name plate above the head, in that stickman's identity colour.

        Faded with local light so labels do not flatten the night scene - an
        unlit figure gets a dim tag, matching the silhouette it is drawn as.
        """
        try:
            dim = 0.42 + 0.58 * max(0.0, min(1.0, float(lit)))
            tag = hud.name_tag(str(a.name)[:12], tuple(a.color), dim)
            r = tag.get_rect()
            # AGENT_HEIGHT above the feet, plus clearance for the speech bubble
            r.midbottom = (int(a.x), int(a.y) - 32)
            if a.speech:
                r.top -= 10
            r.left = max(1, min(r.left, RENDER_W - r.width - 1))
            r.top = max(1, r.top)
            s.blit(tag, r)
        except Exception:
            pass

    # -------------------------------------------------------------- light --
    def _composite_light(self, s: pygame.Surface, world) -> None:
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

    def _draw_water(self, s: pygame.Surface, world, level: float) -> None:
        if level >= RENDER_H:
            return
        h = int(RENDER_H - level)
        water = pygame.Surface((RENDER_W, h), pygame.SRCALPHA)
        water.fill((30, 70, 120, 150))
        s.blit(water, (0, int(level)))
        pygame.draw.line(s, (120, 170, 220), (0, int(level)), (RENDER_W, int(level)), 2)

    # ------------------------------------------------------------ weather --
    def _emit_weather(self, world, dt: float) -> None:
        ev = world.events
        scene = ev.scene
        if getattr(ev, "rain", 0) > 0.01:
            self.particles.emit("rain", int(90 * ev.rain * dt * 60 / 60),
                                wind=ev.wind)
        if getattr(ev, "snow", 0) > 0.01:
            self.particles.emit("snow", int(40 * ev.snow), wind=ev.wind)
        if getattr(ev, "ash", 0) > 0.01:
            self.particles.emit("ash", int(25 * ev.ash), wind=ev.wind)
        if scene == SCENE_WILDFIRE or world.props.burning():
            for p in world.props.burning():
                self.particles.emit("ember", 2, x=p.x, y=p.y - 10)
                self.particles.emit("smoke", 1, x=p.x, y=p.y - 24)
        for st in world.structures.all():
            if st.state.get("lit"):
                self.particles.emit("ember", 1, x=st.x, y=st.y - 8)
        if not world.is_night:
            return
        if scene not in (SCENE_BLIZZARD, SCENE_FLOOD) and getattr(ev, "rain", 0) < 0.1:
            self.particles.emit("firefly", 1)
