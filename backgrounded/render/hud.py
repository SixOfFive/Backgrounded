"""On-screen stats panel, drawn top-right.

Read-only with respect to sim state, like everything else under render/.

Text rendering is the expensive part here - a roster of ten agents is ~40
`font.render` calls, and at 60 fps that would dominate the frame. Every glyph
run is therefore cached by (text, colour, size); the cache is bounded and the
strings repeat almost every frame, so in practice it renders nothing at all
after the first pass and simply blits.

The panel is drawn *after* the light composite on purpose. It is UI, not part
of the world, so it must stay legible when the scene is nearly black.
"""
from __future__ import annotations

import logging

import pygame

from ..constants import (
    DAY_LENGTH_SEC, RENDER_W, RES_COOKED, RES_FIBRE, RES_FOOD, RES_STONE,
    RES_WOOD, SCENE_LABELS,
)

log = logging.getLogger(__name__)

PANEL_W = 268
PAD = 9
LINE = 14

BG = (10, 12, 18)
BG_ALPHA = 178
EDGE = (78, 88, 108)
TITLE = (238, 242, 250)
DIM = (150, 160, 176)
GOOD = (126, 208, 132)
WARN = (232, 194, 104)
BAD = (230, 108, 108)

_font_cache: dict[int, pygame.font.Font] = {}
_text_cache: dict[tuple, pygame.Surface] = {}
_TEXT_CACHE_MAX = 900


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    key = size * 2 + int(bold)
    f = _font_cache.get(key)
    if f is None:
        if not pygame.font.get_init():
            pygame.font.init()
        try:
            f = pygame.font.SysFont("consolas,dejavusansmono,couriernew",
                                    size, bold=bold)
        except Exception:
            f = pygame.font.Font(None, size + 2)
        _font_cache[key] = f
    return f


def _text(s: str, color: tuple[int, int, int], size: int = 11,
          bold: bool = False) -> pygame.Surface:
    key = (s, color, size, bold)
    surf = _text_cache.get(key)
    if surf is None:
        if len(_text_cache) > _TEXT_CACHE_MAX:
            _text_cache.clear()
        surf = _font(size, bold).render(s, True, color)
        _text_cache[key] = surf
    return surf


def name_tag(name: str, color: tuple[int, int, int],
             dim: float = 1.0) -> pygame.Surface:
    """A small name plate for drawing above a stickman.

    Reuses this module's glyph cache, so a colony of ten costs ten cache hits
    per frame rather than ten font renders. `dim` fades the whole tag for
    agents standing outside any light, so the labels do not undo the darkness
    the night scene is built on.
    """
    d = 0.30 if dim < 0.30 else (1.0 if dim > 1.0 else dim)
    col = (int(color[0] * d), int(color[1] * d), int(color[2] * d))
    key = ("__tag__", name, col)
    surf = _text_cache.get(key)
    if surf is None:
        if len(_text_cache) > _TEXT_CACHE_MAX:
            _text_cache.clear()
        glyphs = _font(10, bold=True).render(name, True, col)
        pad = 2
        surf = pygame.Surface((glyphs.get_width() + pad * 2,
                               glyphs.get_height() + pad), pygame.SRCALPHA)
        # A faint plate behind the text so a pale name still reads against
        # snow, and a dark one still reads against a lightning flash.
        surf.fill((0, 0, 0, int(110 * d)))
        surf.blit(glyphs, (pad, 0))
        _text_cache[key] = surf
    return surf


def _need_color(v: float) -> tuple[int, int, int]:
    return GOOD if v < 0.45 else (WARN if v < 0.75 else BAD)


def _bar(surf: pygame.Surface, x: int, y: int, w: int, h: int,
         frac: float, color: tuple[int, int, int]) -> None:
    frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)
    pygame.draw.rect(surf, (44, 48, 58), (x, y, w, h))
    if frac > 0:
        pygame.draw.rect(surf, color, (x, y, max(1, int(w * frac)), h))


def _clock(world) -> str:
    frac = (world.world_time % DAY_LENGTH_SEC) / DAY_LENGTH_SEC
    mins = int(frac * 24 * 60)
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _doing(agent) -> str:
    """What to show in the 'doing' column - the vignette reads better than the
    action kind when one is running, because it is written as a phrase."""
    act = getattr(agent, "action", None)
    if act is None:
        return "idle"
    vig = getattr(act, "vignette", None) or (act.data.get("vignette")
                                             if getattr(act, "data", None) else None)
    label = getattr(vig, "label", None)
    if isinstance(vig, str):
        label = vig
    if label:
        return str(label)
    kind = str(getattr(act, "kind", "") or "idle")
    # BuildStructure -> Build Structure, then trim to fit the column
    out = "".join((" " + c if c.isupper() and i else c)
                  for i, c in enumerate(kind)).strip()
    phase = getattr(act, "phase", None)
    if phase and phase not in ("start", "work"):
        out = f"{out} ({phase})"
    return out


_panel_cache: pygame.Surface | None = None
_panel_key: tuple | None = None

#: The panel is rebuilt at most this often. Everything on it - needs, counts,
#: the chronicle - moves on a human timescale, so redrawing it at 60 fps is
#: pure waste: building it costs ~5 ms against a ~5 ms scene, i.e. it doubled
#: frame time. Rebuilding at 6 Hz makes it effectively free while still
#: looking live.
_REBUILD_HZ = 6.0


def _content_key(world, show_roster: bool) -> tuple:
    """Cheap fingerprint of everything the panel displays."""
    agents = world.population.alive_agents()
    return (
        world.events.scene,
        int(world.world_time * _REBUILD_HZ),
        len(agents),
        world.population.generation,
        world.structures.count(),
        world.stats.get("died", 0),
        tuple(sorted(world.stockpile.items())),
        show_roster,
    )


def draw_stats(surf: pygame.Surface, world, show_roster: bool = True) -> None:
    """Draw the panel. Never raises - a broken HUD must not kill the frame."""
    global _panel_cache, _panel_key
    try:
        key = _content_key(world, show_roster)
        if _panel_cache is None or key != _panel_key:
            _panel_cache = _build(world, show_roster)
            _panel_key = key
        if _panel_cache is not None:
            surf.blit(_panel_cache, (RENDER_W - PANEL_W - 12, 12))
    except Exception:
        log.exception("hud draw failed")


def _build(world, show_roster: bool) -> pygame.Surface:
    agents = [a for a in world.population.alive_agents()]
    agents.sort(key=lambda a: a.id)

    rows = len(agents) if show_roster else 0
    height = (PAD * 2 + LINE * 3 + 6 + (rows * (LINE + 9)) + LINE * 2 + 8
              + (LINE - 2 if show_roster else 0))
    panel = pygame.Surface((PANEL_W, height), pygame.SRCALPHA)
    panel.fill((*BG, BG_ALPHA))
    pygame.draw.rect(panel, EDGE, panel.get_rect(), 1)

    y = PAD

    # ---- header -----------------------------------------------------------
    scene = SCENE_LABELS.get(world.events.scene, world.events.scene)
    panel.blit(_text(scene, TITLE, 13, True), (PAD, y))
    day = int(world.world_time // DAY_LENGTH_SEC) + 1
    stamp = f"day {day}  {_clock(world)}"
    ts = _text(stamp, DIM, 11)
    panel.blit(ts, (PANEL_W - PAD - ts.get_width(), y + 2))
    y += LINE + 3

    pop = len(agents)
    gen = world.population.generation
    panel.blit(_text(f"pop {pop}   gen {gen}   built {world.structures.count()}"
                     f"   lost {world.stats.get('died', 0)}", DIM, 11), (PAD, y))
    y += LINE

    sp = world.stockpile
    panel.blit(_text(
        f"wd {sp.get(RES_WOOD,0):<3} st {sp.get(RES_STONE,0):<3} "
        f"fd {sp.get(RES_FOOD,0):<3} ck {sp.get(RES_COOKED,0):<3} "
        f"fb {sp.get(RES_FIBRE,0)}", DIM, 11), (PAD, y))
    y += LINE
    if show_roster:
        panel.blit(_text("needs: hun=hungry tir=tired cld=cold (full=bad)",
                         (108, 116, 132), 9), (PAD, y))
        y += LINE - 2
    y += 4
    pygame.draw.line(panel, EDGE, (PAD, y), (PANEL_W - PAD, y))
    y += 5

    # ---- roster -----------------------------------------------------------
    if show_roster:
        for a in agents:
            col = tuple(a.color)
            pygame.draw.circle(panel, col, (PAD + 4, y + 5), 4)
            if getattr(a, "holds_candle", False):
                pygame.draw.circle(panel, (255, 210, 120), (PAD + 4, y + 5), 6, 1)

            name = str(a.name)[:11]
            panel.blit(_text(name, TITLE, 11, True), (PAD + 13, y - 1))

            doing = _doing(a)
            maxw = PANEL_W - PAD - 78
            ds = _text(doing, DIM, 10)
            if ds.get_width() > maxw:
                while len(doing) > 4 and _text(doing + "..", DIM, 10).get_width() > maxw:
                    doing = doing[:-1]
                ds = _text(doing + "..", DIM, 10)
            panel.blit(ds, (PAD + 76, y))
            y += LINE - 2

            # need bars: hunger / fatigue / cold. Full bar == bad.
            bw = 20
            bx = PAD + 13
            # Spelled out rather than h/f/c: those read as health/food, which
            # is neither what they measure nor which way round they run.
            for lbl, val in (("hun", a.hunger), ("tir", a.fatigue),
                             ("cld", a.warmth)):
                panel.blit(_text(lbl, DIM, 9), (bx, y - 2))
                _bar(panel, bx + 16, y + 1, bw, 4, float(val),
                     _need_color(float(val)))
                bx += 16 + bw + 6
            role = str(getattr(a, "role", ""))[:9]
            rs = _text(role, DIM, 9)
            panel.blit(rs, (PANEL_W - PAD - rs.get_width(), y - 2))
            y += 11

    # ---- footer: most recent chronicle line -------------------------------
    pygame.draw.line(panel, EDGE, (PAD, y), (PANEL_W - PAD, y))
    y += 5
    last = world.chronicle[-1] if world.chronicle else ""
    if "] " in last:
        last = last.split("] ", 1)[1]
    ls = _text(last[:44], DIM, 10)
    panel.blit(ls, (PAD, y))

    return panel
