"""On-screen stats panel, drawn top-right.

Read-only with respect to sim state, like everything else under render/.

Both panels anchor to the corners of whatever surface they are handed, never to
``RENDER_W``/``RENDER_H``. That is deliberate and load-bearing: the wallpaper is
a bare 1600x1000 image with no window to anchor to, so its copy is baked into
the world surface, while the preview draws the *same* panels onto the real
window surface after the world has been scaled and cropped in - which is what
keeps them in the window's corner while the view zooms and pans underneath.

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
from collections import Counter

import pygame

from ..constants import (
    DAY_LENGTH_SEC, RES_COOKED, RES_FIBRE, RES_FOOD, RES_STONE,
    RES_WOOD, SCENE_LABELS, SCENE_ROTATE_SEC,
)
from ..sim.names import role_label

log = logging.getLogger(__name__)

PANEL_W = 268
PAD = 9

#: Multiplier applied to the finished stats panel and chronicle log.
#:
#: These panels used to be drawn into the 1600x1000 world surface and then
#: scaled to the window along with everything else, so on a big display they
#: grew with the picture. Screen-anchoring them (so they stay in the window
#: corner through a zoom) means they are now blitted at their authored pixel
#: size instead - which on a large or fullscreen window leaves 9-11 px text
#: looking tiny, because it is no longer being magnified by the letterbox fit.
#:
#: Scaling the finished surface rather than re-authoring every font size and
#: layout offset keeps the panel's internal geometry exactly as measured (the
#: alternative touches ~20 hardcoded offsets and reflows the roster), at the
#: cost of some softness in the glyphs. The panels are rebuilt at ~6 Hz and the
#: scaled copy is cached alongside, so this costs one smoothscale per rebuild.
HUD_SCALE = 1.6

#: slot -> (source surface, scale it was built at, scaled result). The source is
#: held by reference deliberately: comparing by id() alone would be unsound,
#: since a freed surface's id can be handed to its replacement and we would
#: serve stale pixels at the right size.
_SCALED: dict[str, tuple[pygame.Surface, float, pygame.Surface]] = {}


def _scaled(src: pygame.Surface, slot: str) -> pygame.Surface:
    """*src* enlarged by :data:`HUD_SCALE`, cached until *src* is rebuilt."""
    try:
        if HUD_SCALE <= 1.001:
            return src
        hit = _SCALED.get(slot)
        if hit is not None and hit[0] is src and abs(hit[1] - HUD_SCALE) < 1e-6:
            return hit[2]
        w = max(1, int(round(src.get_width() * HUD_SCALE)))
        h = max(1, int(round(src.get_height() * HUD_SCALE)))
        out = pygame.transform.smoothscale(src, (w, h))
        _SCALED[slot] = (src, HUD_SCALE, out)
        return out
    except Exception:
        log.debug("hud scale failed", exc_info=True)
        return src
LINE = 14

#: Gap between a panel and the two edges of the target surface it hugs.
MARGIN = 12

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


# =========================================================== activity label ==
# What a stickman is *doing*, as a phrase you can read at wallpaper scale from
# across the room. Two lookups: (kind, phase) first, then kind alone. The phase
# pass matters because nearly every action walks somewhere before it works, and
# "chopping wood" floating over someone still crossing the map reads as a bug.
#
# Phrases are lower case, present tense and short on purpose. They sit above the
# head at 9 px and two neighbours standing a body-width apart must not merge
# into a wall of text, so anything over ~18 characters is the wrong answer here
# however good the copy is.

_ACTIVITY_BY_PHASE: dict[tuple[str, str], str] = {
    ("GatherWood", "approach"): "finding a tree",
    ("GatherWood", "work"): "chopping wood",
    ("GatherWood", "deliver"): "hauling wood",
    ("GatherStone", "approach"): "finding stone",
    ("GatherStone", "work"): "breaking rock",
    ("GatherStone", "deliver"): "hauling stone",
    ("ForageBerries", "approach"): "seeking berries",
    ("ForageBerries", "work"): "picking berries",
    ("ForageBerries", "deliver"): "hauling food",
    ("BuildStructure", "fetch"): "fetching timber",
    ("BuildStructure", "approach"): "hauling to site",
    ("BuildStructure", "work"): "building",
    ("RepairStructure", "approach"): "off to repair",
    ("RepairStructure", "work"): "repairing",
    ("Eat", "approach"): "off to eat",
    ("Eat", "eat"): "eating",
    ("Sleep", "approach"): "heading to bed",
    ("Sleep", "sleep"): "sleeping",
    ("WarmAtFire", "approach"): "seeking warmth",
    ("WarmAtFire", "warm"): "warming up",
    ("CookFood", "fetch"): "fetching food",
    ("CookFood", "approach"): "off to the fire",
    ("CookFood", "cook"): "cooking",
    ("CookFood", "deliver"): "storing a meal",
    ("PlantSapling", "approach"): "finding a spot",
    ("PlantSapling", "plant"): "planting a tree",
    ("Farm", "approach"): "off to the field",
    ("Farm", "till"): "tilling a field",
    ("Farm", "harvest"): "harvesting",
    ("Farm", "deliver"): "hauling crops",
    ("Mine", "approach"): "off to the quarry",
    ("Mine", "dig"): "mining",
    ("Converse", "approach"): "going to chat",
    ("Converse", "talk"): "chatting",
    ("Celebrate", "approach"): "joining in",
    ("Celebrate", "dance"): "dancing",
    ("Mourn", "approach"): "paying respects",
    ("Mourn", "bow"): "mourning",
    ("Lookout", "approach"): "off to the tower",
    ("Lookout", "ascend"): "climbing up",
    ("Lookout", "watch"): "keeping watch",
    ("CraftSpear", "approach"): "fetching wood",
    ("CraftSpear", "work"): "carving a spear",
    ("CraftArmour", "approach"): "fetching hide",
    ("CraftArmour", "work"): "making armour",
    ("FightAnimal", "approach"): "closing in",
    ("FightAnimal", "fight"): "fighting",
}

#: Fallback per kind, used whenever the phase is one the table above does not
#: name (every machine starts in "start", and several never leave it).
_ACTIVITY: dict[str, str] = {
    "Wander": "wandering",
    "GatherWood": "chopping wood",
    "GatherStone": "quarrying",
    "ForageBerries": "foraging",
    "HaulToStockpile": "hauling",
    "BuildStructure": "building",
    "RepairStructure": "repairing",
    "Eat": "eating",
    "Sleep": "sleeping",
    "WarmAtFire": "warming up",
    "CookFood": "cooking",
    "PlantSapling": "planting a tree",
    "Farm": "farming",
    "Mine": "mining",
    "Converse": "chatting",
    "Celebrate": "celebrating",
    "Mourn": "mourning",
    "FleeFrom": "fleeing",
    "ClimbTo": "climbing",
    "Lookout": "keeping watch",
    "FollowParent": "tagging along",
    "Panic": "panicking",
    "CraftSpear": "carving a spear",
    "CraftArmour": "making armour",
    "FightAnimal": "fighting",
    "FleeAnimal": "running away",
    "Vignette": "idling",          # only reached if the vignette lost its label
}


def _humanise(kind: str) -> str:
    """Last-ditch phrasing for a kind no table above knows.

    A new behaviour module, or a save written by a later build, must never put
    a bare identifier like ``TendBeehive`` on the user's wallpaper - it looks
    like a leaked internal, because it is one. Splitting the camel case and
    lower-casing is wrong-but-readable, which is the correct failure here.
    """
    out = "".join((" " + c.lower() if c.isupper() and i else c.lower())
                  for i, c in enumerate(str(kind)))
    return out.strip() or "idle"


def _vignette_label(act) -> str:
    """The hand-written phrase off a cosmetic vignette, or "" if this is not one.

    Vignettes carry copy an author wrote ("throws a cartwheel"), which beats
    anything generated from a kind, so it wins outright wherever it exists.
    Both shapes are accepted: the live VignetteAction exposes ``.label``, while
    a rehydrated one may only have stashed the string in ``data``.
    """
    try:
        label = getattr(act, "label", None)
        if isinstance(label, str) and label.strip():
            return label.strip()
        vig = getattr(act, "vignette", None)
        if isinstance(vig, str) and vig.strip():
            return vig.strip()
        label = getattr(vig, "label", None)
        if isinstance(label, str) and label.strip():
            return label.strip()
        data = getattr(act, "data", None)
        if isinstance(data, dict):
            label = data.get("label")
            if isinstance(label, str) and label.strip():
                return label.strip()
    except Exception:
        pass
    return ""


def activity_of(agent) -> str:
    """A short present-tense phrase for what *agent* is doing right now.

    Returns "" for the dead and for anything unreadable, so callers can simply
    skip drawing. Never raises: this is on the per-frame path for every agent.
    """
    try:
        if not getattr(agent, "alive", True):
            return ""
        act = getattr(agent, "action", None)
        if act is None:
            return "idle"
        label = _vignette_label(act)
        if label:
            return label
        kind = str(getattr(act, "kind", "") or "")
        if not kind:
            return "idle"
        phase = str(getattr(act, "phase", "") or "")
        phrase = _ACTIVITY_BY_PHASE.get((kind, phase))
        if phrase is None:
            phrase = _ACTIVITY.get(kind)
        if phrase is None:
            phrase = _humanise(kind)
        return phrase
    except Exception:
        return ""


#: Plate and text colours for the activity tag. Deliberately *not* the agent's
#: identity colour - the name plate above already carries identity, and ten
#: hues of tiny text is noise. Pale grey on a near-opaque near-black plate is
#: the one pairing that survives both the black night-storm and the white-out
#: fog scene, because it reads against the plate rather than against the sky.
ACT_TEXT = (208, 216, 230)
ACT_PLATE = (5, 7, 12, 178)

#: Longest phrase the plate will show. Sized off the real content, not the
#: table above: the vignette engine writes whole sentences ("cups both hands
#: and drinks what the sky gives it"), and cutting those at the ~18 characters
#: the action phrases need turns good copy into rubble. 26 is about 140 px at
#: 9 px - wide next to a 21 px stickman, but the overlap nudge in the renderer
#: is what actually keeps a crowd readable, not a tighter cap here.
ACT_MAX_CHARS = 26


def _shorten(s: str, limit: int) -> str:
    """Trim to *limit* characters on a word boundary, marking the cut.

    Mid-word cuts ("cups both hands a..") read as a rendering fault rather than
    as elision, which is the one thing a status label must never look like."""
    if len(s) <= limit:
        return s
    cut = s[:limit - 2]
    space = cut.rfind(" ")
    if space >= limit // 2:              # only if a word boundary is near
        cut = cut[:space]
    return cut.rstrip(" ,;:") + ".."


def activity_tag(text: str, dim: float = 1.0) -> "pygame.Surface | None":
    """A small rounded plate reading what a stickman is doing.

    Returns None for empty text so the caller can skip the blit outright.

    `dim` only ever fades to 0.55, not to the 0.30 the name plate allows: the
    name is bold and coloured and survives being crushed, a 9 px grey line does
    not. It is also quantised to 8 steps before it reaches the cache key -
    it comes from a continuous light sample, so passing it through raw would
    mint a fresh surface per agent per frame and turn this module's glyph cache
    into a churn machine, which is the one cost it exists to avoid.
    """
    s = str(text or "").strip()
    if not s:
        return None
    s = _shorten(s, ACT_MAX_CHARS)
    d = 0.55 if dim < 0.55 else (1.0 if dim > 1.0 else dim)
    q = round(d * 8.0) / 8.0
    key = ("__act__", s, q)
    surf = _text_cache.get(key)
    if surf is not None:
        return surf
    if len(_text_cache) > _TEXT_CACHE_MAX:
        _text_cache.clear()
    col = (int(ACT_TEXT[0] * q), int(ACT_TEXT[1] * q), int(ACT_TEXT[2] * q))
    glyphs = _font(9).render(s, True, col)
    padx, pady = 3, 1
    surf = pygame.Surface((glyphs.get_width() + padx * 2,
                           glyphs.get_height() + pady * 2), pygame.SRCALPHA)
    try:
        pygame.draw.rect(surf, ACT_PLATE, surf.get_rect(), border_radius=3)
    except TypeError:                    # a pygame too old for rounded corners
        surf.fill(ACT_PLATE)
    surf.blit(glyphs, (padx, pady))
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
    """What to show in the roster's 'doing' column.

    Shares :func:`activity_of` with the label drawn above the stickman's head,
    so the panel and the world can never disagree about what someone is up to -
    they used to, because this column spelled out the raw kind ("Gather Wood")
    while the label read a phrase.
    """
    return activity_of(agent) or "idle"


_panel_cache: pygame.Surface | None = None
_panel_key: tuple | None = None

# ================================================================== tooltips ==
# Hover detail for the stats panel: "wd 12" is compact enough to read at a
# glance and opaque enough to be useless until you know what it means, so the
# panel now explains itself under the pointer.
#
# The hit regions are built *with* the panel, in the panel's own unscaled
# coordinates, and transformed at draw time by the blit origin and HUD_SCALE.
# Two things fall out of that which matter: a window resize moves the panel and
# needs no cache invalidation (only the origin changes), and a [ / ] resize
# needs none either (only the divisor changes). Storing them in window
# coordinates instead would silently rot on both.
#
# Tooltip text is captured when the panel is rebuilt, so it is at most one
# _REBUILD_HZ period (~0.17 s) stale. Rebuilding zones per frame to close that
# would cost a font-metrics pass at 60 Hz for a number nobody can read moving
# that fast.

#: (rect in panel-local unscaled px, title, body lines).
_panel_zones: list[tuple[pygame.Rect, str, tuple[str, ...]]] = []

TIP_BG = (16, 19, 26)
TIP_BG_ALPHA = 242
TIP_EDGE = (98, 110, 132)
#: Wrap width for tooltip body text, in authored px before HUD_SCALE.
TIP_W_MAX = 262


def _span(font: pygame.font.Font, text: str, i: int, j: int,
          x: int, y: int, h: int) -> pygame.Rect:
    """Rect covering ``text[i:j]`` of *text* rendered at (*x*, *y*).

    Measured with the same font the line is rendered with rather than assumed
    monospace: the panel asks for consolas and takes whatever the system has,
    and on a machine with none of the three the fallback is proportional.
    """
    x0 = font.size(text[:i])[0]
    x1 = font.size(text[:j])[0]
    return pygame.Rect(int(x + x0), int(y), max(2, int(x1 - x0)), int(h))


def _tip_surface(title: str, body: tuple[str, ...],
                 avail: int = 1 << 20) -> pygame.Surface:
    """The tooltip plate, cached by content, scale and the width available.

    Rendered at scaled font sizes rather than built small and enlarged: this
    is the one thing on screen the user is actively squinting at, and a
    smoothscaled 10 px glyph is exactly what they were complaining about.

    ``avail`` is how much room the target surface has. It has to be a factor
    because the wrap width is scaled by :data:`HUD_SCALE`, and scale is a user
    dial while window size is a separate one: at 3.5x the authored 262 px wraps
    at ~917 px, which is wider than a 640 px window, and no amount of clamping
    the *position* rescues a plate that does not fit. It is quantised to 32 px
    so that dragging a resize does not mint a fresh plate per pixel.
    """
    s = max(1.0, float(HUD_SCALE))
    pad = max(4, int(round(6 * s)))
    cap = max(120, int(avail) // 32 * 32)
    key = ("__tip__", title, body, round(s, 3), cap)
    surf = _text_cache.get(key)
    if surf is not None:
        return surf
    if len(_text_cache) > _TEXT_CACHE_MAX:
        _text_cache.clear()

    ts = max(9, int(round(12 * s)))
    bs = max(8, int(round(10 * s)))
    maxw = max(60, min(int(TIP_W_MAX * s), cap - pad * 2))
    rows: list[pygame.Surface] = []
    if title:
        rows.append(_font(ts, True).render(title, True, TITLE))
    for entry in body:
        text = str(entry)
        if not text:                      # deliberate blank spacer row
            rows.append(_font(bs).render(" ", True, DIM))
            continue
        for line in _wrap(text, maxw, bs):
            rows.append(_font(bs).render(line, True, DIM))

    # A single unbreakable word longer than the wrap width still comes back at
    # full length from _wrap, so the plate is capped here as well and that row
    # is clipped by the blit. Losing the tail of one long word beats a plate
    # hanging off the side of the window.
    w = min(cap, max((r.get_width() for r in rows), default=1) + pad * 2)
    h = sum(r.get_height() for r in rows) + pad * 2
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill((*TIP_BG, TIP_BG_ALPHA))
    pygame.draw.rect(surf, TIP_EDGE, surf.get_rect(), 1)
    y = pad
    for r in rows:
        surf.blit(r, (pad, y))
        y += r.get_height()
    _text_cache[key] = surf
    return surf


def _draw_tip(surf: pygame.Surface, mouse: tuple[int, int],
              title: str, body: tuple[str, ...]) -> None:
    """Place the tooltip beside the pointer, kept inside *surf*.

    Left of the cursor by preference. The panel hugs the right edge of the
    window, so anything hovering it is already near that edge and a tooltip
    opening rightwards would be clamped against the frame on every single
    hover - which reads as the tooltip being stuck rather than as placement.
    """
    tip = _tip_surface(title, body, max(160, surf.get_width() - MARGIN * 2))
    s = max(1.0, float(HUD_SCALE))
    gap = int(12 * s)
    x = mouse[0] - tip.get_width() - gap
    if x < MARGIN:
        x = mouse[0] + gap
    y = mouse[1] + int(8 * s)
    x = max(0, min(x, surf.get_width() - tip.get_width()))
    y = max(0, min(y, surf.get_height() - tip.get_height()))
    surf.blit(tip, (x, y))


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
        world.stats.get("abducted", 0),
        world.stats.get("returned", 0),
        tuple(sorted(world.stockpile.items())),
        show_roster,
    )


def draw_stats(surf: pygame.Surface, world, show_roster: bool = True,
               offset: tuple[int, int] = (0, 0),
               mouse: "tuple[int, int] | None" = None) -> None:
    """Draw the panel in *surf*'s own top-right corner. Never raises.

    The anchor is measured off the target surface rather than off ``RENDER_W``,
    which is what lets one function serve both outputs: handed the 1600x1000
    world surface it lands exactly where it always has (the wallpaper has no
    window to anchor to, so its panel has to be baked into the image), and
    handed a preview *window* surface of any size it lands in that window's
    corner - which is the whole point, because a panel baked into the image
    slides off screen the moment you zoom or pan.

    Only the blit *position* depends on the target; the cached panel does not.
    So a resized window costs nothing, needs no cache invalidation, and can
    never draw at a stale position.

    ``offset`` shifts the whole panel, and exists for one caller: the wallpaper
    path re-applies the screen shake that :meth:`Renderer.draw` baked into the
    frame, so the panel keeps riding the earthquake there exactly as before.

    ``mouse`` is the pointer in *surf*'s coordinates, and is what turns the
    hover tooltips on. It is a parameter rather than something read off
    ``pygame.mouse`` here because this function is called twice per frame - once
    onto the window and once into the wallpaper image - and only one of those
    has a pointer over it. Reading the mouse internally would bake a tooltip
    into the desktop wallpaper, anchored to a window the wallpaper cannot see.
    """
    global _panel_cache, _panel_key, _panel_zones
    try:
        key = _content_key(world, show_roster)
        if _panel_cache is None or key != _panel_key:
            _panel_cache, _panel_zones = _build(world, show_roster)
            _panel_key = key
        if _panel_cache is not None:
            panel = _scaled(_panel_cache, "stats")
            # max(0, ...) so a window narrower than the panel still shows its
            # left edge rather than pushing it off screen entirely. Measured off
            # the *scaled* width, or the anchor would drift by the scale factor.
            x = max(0, surf.get_width() - panel.get_width() - MARGIN)
            ox = x + int(offset[0])
            oy = MARGIN + int(offset[1])
            surf.blit(panel, (ox, oy))
            if mouse is not None and _panel_zones:
                _hover(surf, mouse, ox, oy)
    except Exception:
        log.exception("hud draw failed")


def _hover(surf: pygame.Surface, mouse: tuple[int, int],
           ox: int, oy: int) -> None:
    """Draw the tooltip for whatever zone the pointer is inside, if any."""
    s = max(0.01, float(HUD_SCALE))
    lx = (mouse[0] - ox) / s
    ly = (mouse[1] - oy) / s
    for rect, title, body in _panel_zones:
        if rect.collidepoint(lx, ly):
            _draw_tip(surf, mouse, title, body)
            return


def _is_mutant(a) -> bool:
    try:
        fn = getattr(a, "is_mutant", None)
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


def _dur(sec: float) -> str:
    """Coarse elapsed time - "2h 05m", or "7m 30s" under the hour."""
    total = max(0, int(sec))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def _scene_tip(world) -> tuple[str, ...]:
    try:
        if getattr(world, "auto_scene_rotate", True):
            left = max(0.0, SCENE_ROTATE_SEC
                       - float(getattr(world.events, "scene_t", 0.0) or 0.0))
            return (f"changes by itself in {_dur(left)}",
                    "turn that off in Tray > Scene")
        return ("automatic scene changes are off",
                "pick another in Tray > Scene")
    except Exception:
        return ()


def _pop_tip(world, agents) -> tuple[str, ...]:
    try:
        tally = Counter(role_label(getattr(a, "role", "")) for a in agents)
        lines = [f"{len(agents)} alive right now"]
        lines += [f"   {n} x {role}" for role, n in sorted(tally.items())]
        mutants = sum(1 for a in agents if _is_mutant(a))
        if mutants:
            lines.append(f"{mutants} of them came out strange")
        return tuple(lines)
    except Exception:
        return ()


def _built_tip(world, finished: int) -> tuple[str, ...]:
    try:
        total = world.structures.count(built_only=False)
        wip = max(0, int(total) - int(finished))
        lines = [f"{finished} finished and standing"]
        if wip:
            lines.append(f"{wip} still going up")
        return tuple(lines)
    except Exception:
        return ()


def _lost_tip(world) -> tuple[str, ...]:
    try:
        st = world.stats
        lines = [f"{int(st.get('died', 0) or 0)} have died here"]
        taken = int(st.get("abducted", 0) or 0)
        if taken:
            back = int(st.get("returned", 0) or 0)
            # Kept apart from the death toll on purpose: an abduction takes
            # somebody off the roster alive and leaves no grave.
            lines.append(f"{taken} taken by the lights, {back} put back")
        born = int(st.get("born", 0) or 0)
        lines.append(f"{born} born here since the colony started")
        return tuple(lines)
    except Exception:
        return ()


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def _agent_tip(a) -> tuple[str, tuple[str, ...]]:
    """Title and body for one roster row."""
    try:
        title = str(getattr(a, "name", "?"))
        body = [f"{role_label(getattr(a, 'role', ''))}, "
                f"generation {int(getattr(a, 'generation', 0) or 0)}"]
        if _is_mutant(a):
            morph = str(getattr(a, "morph", "") or "").replace("_", " ")
            if morph:
                body.append(f"and {morph}")
        body.append("")
        # The roster column truncates this to fit; the whole point of the
        # tooltip is that here it does not.
        body.append(activity_of(a) or "idle")
        carrying = getattr(a, "carrying", None)
        qty = int(getattr(a, "carry_qty", 0) or 0)
        if carrying and qty > 0:
            body.append(f"carrying {qty} x {carrying}")
        if getattr(a, "holds_candle", False):
            body.append("holding a candle")
        body.append("")
        for label, val in (("hungry", getattr(a, "hunger", 0.0)),
                           ("tired", getattr(a, "fatigue", 0.0)),
                           ("cold", getattr(a, "warmth", 0.0))):
            body.append(f"{label + ':':<8}{int(round(float(val) * 100)):3d}%")
        return title, tuple(body)
    except Exception:
        return str(getattr(a, "name", "?")), ()


#: (key, resource, title, why it matters). The second line of each is the part
#: the panel cannot say: "wd 12" tells you the number and nothing about what
#: spends it.
_RES_FIELDS = (
    ("wd", RES_WOOD, "Wood",
     ("chopped from trees", "every build needs it, and the firepit burns it")),
    ("st", RES_STONE, "Stone",
     ("broken off the rock face", "walls, hearths and the heavier builds")),
    ("fd", RES_FOOD, "Food",
     ("raw - foraged berries and harvested crops",
      "edible as it is, but worth cooking first")),
    ("ck", RES_COOKED, "Cooked food",
     ("meals made at the fire", "eaten before anyone touches the raw food")),
    ("fb", RES_FIBRE, "Fibre",
     ("stripped while gathering",
      "huts, bridges, ladders and totems all want it")),
)


def _build(world, show_roster: bool) -> tuple[pygame.Surface, list]:
    agents = [a for a in world.population.alive_agents()]
    agents.sort(key=lambda a: a.id)

    # Hit regions for the hover tooltips, in this panel's own coordinates. Built
    # here rather than in a second pass so a zone cannot drift from the text it
    # names: every rect below is measured from the same y the blit above it used.
    zones: list[tuple[pygame.Rect, str, tuple[str, ...]]] = []
    f11 = _font(11)

    rows = len(agents) if show_roster else 0

    # The ufo's books, kept apart from "lost": an abduction takes somebody off
    # the roster alive and leaves no grave, so counting it as a death would be
    # a lie, and leaving it out entirely means the headcount just drops with
    # nothing on the panel to explain it. .get() with defaults so a save from
    # before these counters existed still renders.
    taken = int(world.stats.get("abducted", 0) or 0)
    back = int(world.stats.get("returned", 0) or 0)
    # Only once the ufo has actually taken someone - most colonies never meet
    # it, and a permanently-zero row is pure noise. It needs its own line
    # because appended to "pop/gen/built/lost" it runs off the panel edge.
    ufo_line = taken > 0

    # Footer now wraps to two lines, so reserve LINE * 3 for it (divider +
    # two text lines) instead of LINE * 2.
    height = (PAD * 2 + LINE * 3 + 6 + (rows * (LINE + 9)) + LINE * 3 + 8
              + (LINE - 2 if show_roster else 0) + (LINE if ufo_line else 0))
    panel = pygame.Surface((PANEL_W, height), pygame.SRCALPHA)
    panel.fill((*BG, BG_ALPHA))
    pygame.draw.rect(panel, EDGE, panel.get_rect(), 1)

    y = PAD

    # ---- header -----------------------------------------------------------
    scene = SCENE_LABELS.get(world.events.scene, world.events.scene)
    st = _text(scene, TITLE, 13, True)
    panel.blit(st, (PAD, y))
    zones.append((pygame.Rect(PAD, y, st.get_width(), LINE),
                  f"Scene: {scene}", _scene_tip(world)))
    day = int(world.world_time // DAY_LENGTH_SEC) + 1
    stamp = f"day {day}  {_clock(world)}"
    ts = _text(stamp, DIM, 11)
    tx = PANEL_W - PAD - ts.get_width()
    panel.blit(ts, (tx, y + 2))
    zones.append((pygame.Rect(tx, y, ts.get_width(), LINE),
                  f"Day {day}, {_clock(world)}",
                  (f"a full day and night is {int(DAY_LENGTH_SEC)}s of sim time",
                   f"this world has been running {_dur(world.world_time)}")))
    y += LINE + 3

    pop = len(agents)
    gen = world.population.generation
    finished = world.structures.count()
    # Built as segments rather than one f-string so each field's character span
    # is known, which is what the hover zones are measured from. The joined
    # result is byte-identical to the single string this replaced.
    counters = (
        (f"pop {pop}", "Population", _pop_tip(world, agents)),
        (f"gen {gen}", "Generation",
         (f"{gen} {_plural(gen, 'generation has', 'generations have')} lived here",
          "it goes up when a child of this colony has a child")),
        (f"built {finished}", "Structures", _built_tip(world, finished)),
        (f"lost {int(world.stats.get('died', 0) or 0)}", "Lost", _lost_tip(world)),
    )
    line = "   ".join(text for text, _, _ in counters)
    panel.blit(_text(line, DIM, 11), (PAD, y))
    at = 0
    for text, title, body in counters:
        zones.append((_span(f11, line, at, at + len(text), PAD, y, LINE),
                      title, body))
        at += len(text) + 3                      # the three-space separator
    y += LINE

    if ufo_line:
        panel.blit(_text(f"taken {taken}" + (f", back {back}" if back else ""),
                         DIM, 11), (PAD, y))
        y += LINE

    sp = world.stockpile
    parts: list[str] = []
    tips: list[tuple[str, tuple[str, ...]]] = []
    last = len(_RES_FIELDS) - 1
    for i, (key, res, title, why) in enumerate(_RES_FIELDS):
        qty = int(sp.get(res, 0) or 0)
        # The final field carries no padding, exactly as the original did -
        # trailing spaces on the last column would widen the panel's text for
        # no reason and shift nothing else.
        parts.append(f"{key} {qty}" if i == last else f"{key} {qty:<3}")
        tips.append((f"{title} - {qty}", why))
    line = " ".join(parts)
    panel.blit(_text(line, DIM, 11), (PAD, y))
    at = 0
    for part, (title, body) in zip(parts, tips):
        zones.append((_span(f11, line, at, at + len(part), PAD, y, LINE),
                      title, body))
        at += len(part) + 1
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
            row_top = y
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

            # One zone for the whole two-line block rather than four small ones
            # per agent. Splitting it would mean the tooltip changed under the
            # pointer as it drifted a few pixels between a name and the bar
            # beside it, and everything it would say is on the one plate anyway.
            title, body = _agent_tip(a)
            zones.append((pygame.Rect(PAD, row_top, PANEL_W - PAD * 2,
                                      max(1, y - row_top)), title, body))

    # ---- footer: most recent chronicle line -------------------------------
    pygame.draw.line(panel, EDGE, (PAD, y), (PANEL_W - PAD, y))
    y += 5
    foot_top = y
    raw = str(world.chronicle[-1] if world.chronicle else "")
    last = raw.split("] ", 1)[1] if "] " in raw else raw
    stamp = raw.split("] ", 1)[0].lstrip("[") if "] " in raw else ""
    # Wrap onto up to two lines rather than clipping mid-word. The height
    # calc below reserves two footer lines, so this never overruns the panel.
    wrapped = _wrap(last, PANEL_W - PAD * 2, 10)
    for line in wrapped[:2]:
        panel.blit(_text(line, DIM, 10), (PAD, y))
        y += LINE - 3
    if last:
        # Worth a tooltip even though the text is right there: a long event is
        # cut at two lines, and the day stamp is stripped to save width.
        zones.append((pygame.Rect(PAD, foot_top, PANEL_W - PAD * 2,
                                  max(LINE, y - foot_top)),
                      "Latest event",
                      ((last,) if not stamp else (last, "", stamp))))

    return panel, zones


def _wrap(text: str, max_w: int, size: int) -> list[str]:
    """Greedy word-wrap to a pixel width, using the cached font metrics."""
    words = text.split()
    if not words:
        return [""]
    font = _font(size)
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if font.size(trial)[0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ============================================================ chronicle log ==
# The last N chronicle lines, lower-left. Where the top-right footer shows only
# the newest event, this is the running story: births, deaths, buildings, the
# spikes taking a wolf. Anchored to the target surface like the stats panel, so
# it reads on the wallpaper and holds the preview window's corner alike.
# Newest sits at the bottom; older lines fade up.

LOG_W = 360
LOG_LINES = 10

_log_cache: pygame.Surface | None = None
_log_key: tuple | None = None


def draw_log(surf: pygame.Surface, world, n: int = LOG_LINES,
             offset: tuple[int, int] = (0, 0)) -> None:
    """Draw the last `n` chronicle entries in *surf*'s lower-left. Never raises.

    Anchored to the target surface's own height for the same reason
    :func:`draw_stats` is anchored to its width - see that docstring.
    """
    global _log_cache, _log_key
    try:
        chron = list(getattr(world, "chronicle", ()) or ())
        tail = chron[-n:]
        key = (tuple(tail),)
        if _log_cache is None or key != _log_key:
            _log_cache = _build_log(tail)
            _log_key = key
        if _log_cache is not None:
            panel = _scaled(_log_cache, "log")
            y = max(0, surf.get_height() - panel.get_height() - MARGIN)
            surf.blit(panel, (MARGIN + int(offset[0]), y + int(offset[1])))
    except Exception:
        log.exception("chronicle log draw failed")


def _build_log(tail: list) -> "pygame.Surface | None":
    # Strip the "[day N] " stamp the chronicle prepends - the log is tight on
    # width and the day is already implied by order.
    lines: list[str] = []
    for entry in tail:
        text = str(entry)
        if "] " in text:
            text = text.split("] ", 1)[1]
        wrapped = _wrap(text, LOG_W - PAD * 2, 11)
        lines.extend(wrapped[:2])          # a single event never eats the log

    lines = lines[-(LOG_LINES + 4):]       # cap total rows after wrapping
    if not lines:
        return None

    row = LINE - 2
    height = PAD * 2 + row * len(lines)
    panel = pygame.Surface((LOG_W, height), pygame.SRCALPHA)
    panel.fill((*BG, 150))
    pygame.draw.rect(panel, EDGE, panel.get_rect(), 1)

    y = PAD
    total = len(lines)
    for i, ln in enumerate(lines):
        # Oldest at the top, dimmest; newest at the bottom, brightest.
        f = 0.45 + 0.55 * (i + 1) / total
        col = (int(DIM[0] * f), int(DIM[1] * f), int(DIM[2] * f))
        panel.blit(_text(ln, col, 11), (PAD, y))
        y += row
    return panel
