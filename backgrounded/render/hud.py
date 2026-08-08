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
import re                      # also imported locally in two helpers below;
                               # the module-level one is what their return
                               # annotations resolve against.
from collections import Counter

import pygame

from ..constants import (
    DAY_LENGTH_SEC, MAX_HEALTH, RES_COOKED, RES_FIBRE, RES_FOOD, RES_STONE,
    RES_WOOD, SCENE_LABELS, SCENE_ROTATE_SEC,
)
from ..sim.names import ROLE_HERMIT, role_label

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

#: Stickman id the camera is locked onto, or None. Set by App once a frame.
#:
#: A MODULE DIAL, exactly like HUD_SCALE above, and for the same reason: the
#: window overlay and the wallpaper bake go through this one draw function and
#: must agree, and the wallpaper path has no window, no pointer and no caption
#: to carry the information any other way. If the ring were a draw_stats
#: argument the wallpaper - which is where this app is actually watched - would
#: be the copy that never said who was being followed.
#:
#: It is a plain id and not the Camera: hud.py must not learn what a camera is,
#: and a stale id simply rings nobody. App writes it from ``camera.lock_id``;
#: nothing here writes it. It is part of :func:`_content_key`, or the panel
#: cache would keep serving a ring for the man you stopped following.
FOLLOW_ID: int | None = None
#: Colour of that ring. Deliberately cool and deliberately not the candle's
#: amber (255, 210, 120), which rings the same dot one pixel further in: the
#: two marks mean completely different things - "he carries the light" against
#: "the view is on him" - and a colonist can easily be both at once.
FOLLOW_RING = (150, 214, 255)

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


def _health_color(v: float) -> tuple[int, int, int]:
    """Colour for a health fraction, where FULL IS GOOD.

    The inverse of :func:`_need_color`, and that inversion is the whole reason
    this exists rather than reusing it. Every other bar on the roster is a
    *need*: hunger, fatigue and cold all run empty-is-fine to full-is-dying, and
    the panel's own legend says "(full=bad)". Health runs the other way. Sharing
    a colour ramp between the two would mean a dying agent's health bar went
    green at the moment it should be screaming, which is worse than not drawing
    it at all - a glanceable panel that lies is not glanceable.
    """
    return BAD if v < 0.30 else (WARN if v < 0.65 else GOOD)


def _health_frac(agent) -> float:
    """0..1 health, tolerant of an agent that has none.

    Read through MAX_HEALTH rather than assumed to be a fraction already, and
    clamped, because relics push raw damage well past the nominal range -
    dragons._devour deals MAX_HEALTH * 10 - and a bar drawn from an unclamped
    ratio would run off the end of the panel.
    """
    try:
        hp = float(getattr(agent, "health", MAX_HEALTH))
    except (TypeError, ValueError):
        return 1.0
    if hp != hp:                                  # nan
        return 1.0
    return 0.0 if hp <= 0.0 else min(1.0, hp / float(MAX_HEALTH or 1.0))


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

#: budget -> (content key, panel, zones). The panel is now built against the
#: height it has to fit into (see :func:`draw_stats`), and this function is
#: called TWICE PER FRAME with two different targets on the wallpaper path - the
#: window and the 1600x1000 world surface - so a single slot would thrash and
#: pay the ~5 ms build on every one of those calls. Two entries is all that is
#: ever live; four is room for a resize in flight. Bounded by construction, so
#: it cannot grow with the number of windows that have ever existed.
_panels: dict[int, tuple[tuple, pygame.Surface, list]] = {}
_PANELS_MAX = 4

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
#: Roster rows paired with their agent id, for click-to-follow. Same rects as the
#: agent zones above, in the same panel-local UNSCALED space.
_panel_agents: list[tuple[pygame.Rect, int]] = []
#: Where the panel was last blitted ON THE INTERACTIVE PASS. draw_stats runs
#: twice a frame and only the window overlay can be clicked, so this is written
#: under `mouse is not None` and never by the wallpaper bake.
_panel_origin: tuple[int, int] | None = None


def agent_at(mouse: "tuple[int, int] | None") -> "int | None":
    """Which colonist's roster row is under *mouse*, if any. Never raises.

    Screen coords in, agent id out. The rects are stored in the panel's own
    authored coordinates and scaled at lookup time by HUD_SCALE - the same trick
    the tooltips use - so this keeps working across a window resize and the
    [ / ] scale keys with no cache to invalidate.
    """
    if mouse is None or _panel_origin is None or not _panel_agents:
        return None
    try:
        s = max(0.01, float(HUD_SCALE))
        lx = (mouse[0] - _panel_origin[0]) / s
        ly = (mouse[1] - _panel_origin[1]) / s
        for rect, aid in _panel_agents:
            if rect.collidepoint(lx, ly):
                return int(aid)
    except Exception:
        log.debug("hud agent_at failed", exc_info=True)
    return None

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
        # His pile moves independently of the colony's, so it needs its own term
        # or the panel would show a stale row for up to a rebuild period after
        # every stoke - and, worse, would never grow the panel on the tick a
        # hermit is first appointed.
        tuple(sorted(_stash_of(world).items())),
        sum(1 for a in agents if _is_hermit(a)),
        show_roster,
        # The follow ring is drawn into the cached panel, so the id it was
        # drawn for has to be part of the fingerprint. Without this the ring
        # would stay on the man you stopped following - and never appear on the
        # one you started - until some unrelated change happened to invalidate
        # the cache, which on a settled colony is up to a whole rebuild period.
        FOLLOW_ID,
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
    global _panel_cache, _panel_key, _panel_zones, _panel_agents, _panel_origin
    try:
        # THE HEIGHT THE PANEL HAS TO FIT INTO, in the panel's own authored px.
        # `_build` lays out at authored size and `_scaled` then multiplies the
        # finished surface by HUD_SCALE, so anything past the bottom of the
        # target is simply blitted off the edge and lost. Handing the budget
        # down is what lets `_build` decide whether the hermit's extra line can
        # be afforded - see the stash-row note there. Quantised to whole px so
        # a sub-pixel wobble in the window height cannot miss the cache.
        budget = int((surf.get_height() - MARGIN * 2) / max(0.01, float(HUD_SCALE)))
        key = _content_key(world, show_roster)
        hit = _panels.get(budget)
        if hit is None or hit[0] != key:
            panel_src, zones, agents = _build(world, show_roster, budget)
            if len(_panels) >= _PANELS_MAX:
                for stale in _panels:
                    _SCALED.pop(f"stats:{stale}", None)
                _panels.clear()
            _panels[budget] = (key, panel_src, zones, agents)
        else:
            panel_src, zones, agents = hit[1], hit[2], hit[3]
        # The module globals stay live: `_hover` reads `_panel_zones`, and the
        # probes and tools/smoke.py read both. They track the panel that was
        # drawn most recently, which is the one the pointer is over.
        _panel_cache, _panel_zones, _panel_key = panel_src, zones, key
        _panel_agents = agents
        if _panel_cache is not None:
            panel = _scaled(_panel_cache, f"stats:{budget}")
            # max(0, ...) so a window narrower than the panel still shows its
            # left edge rather than pushing it off screen entirely. Measured off
            # the *scaled* width, or the anchor would drift by the scale factor.
            x = max(0, surf.get_width() - panel.get_width() - MARGIN)
            ox = x + int(offset[0])
            oy = MARGIN + int(offset[1])
            surf.blit(panel, (ox, oy))
            if mouse is not None:
                # THE INTERACTIVE PASS, AND ONLY IT. draw_stats runs twice a
                # frame - once as this screen-anchored window overlay and once
                # baked world-anchored into the wallpaper frame - and the second
                # one would otherwise leave `_panel_origin` pointing at geometry
                # nobody can click. `mouse is not None` is exactly the window
                # pass (app.py hands it a position only there), which is the
                # same discriminator `_hover` has always relied on.
                _panel_origin = (ox, oy)
                if _panel_zones:
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
        # Health in points rather than a percentage, unlike the needs below it:
        # the damage numbers in this game are authored against MAX_HEALTH (a
        # spear does SPEAR_DAMAGE, a dragon's maw deals ten times the bar), so
        # "62/100" tells you how many more hits they have in them and "62%"
        # does not.
        hp = _health_frac(a)
        body.append(f"{'health:':<8}{int(round(hp * float(MAX_HEALTH))):3d}"
                    f"/{int(MAX_HEALTH)}")
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


def _is_hermit(a) -> bool:
    try:
        return str(getattr(a, "role", "")) == ROLE_HERMIT
    except Exception:
        return False


def _stash_of(world) -> dict:
    """The hermit's camp store, read-only and never assumed to exist.

    A save written before the stash existed loads with an all-zero dict from
    `World._init_colony_state`, but this panel is also handed stub worlds by the
    probe and by `tools/smoke.py`, and a HUD that raises is a lost frame.
    """
    try:
        st = getattr(world, "hermit_stash", None)
        return st if isinstance(st, dict) else {}
    except Exception:
        return {}


def _stash_line(stash: dict) -> tuple[str, list[str]]:
    """``("stash wd 4 fd 8", [...detail...])`` for the row under his name.

    THE SAME IDIOM AS THE STOCKPILE and deliberately so: the panel already
    teaches `wd`/`st`/`fd`/`ck`/`fb` two lines further up, so his row costs the
    reader no new vocabulary. Zeroes are DROPPED rather than padded, which is
    the one departure: the stockpile line is a fixed set of columns whose widths
    must not jitter, while this one sits inside a roster row that is already
    ragged, and "stash wd 4 fd 8" reads better than five fields of mostly zero.
    """
    parts: list[str] = []
    detail: list[str] = []
    for key, res, title, _why in _RES_FIELDS:
        qty = int(stash.get(res, 0) or 0)
        if qty > 0:
            parts.append(f"{key} {qty}")
            detail.append(f"   {qty} x {title.lower()}")
    if not parts:
        return "stash empty", []
    return "stash " + " ".join(parts), detail


def _stash_tip(name: str, stash: dict) -> tuple[str, tuple[str, ...]]:
    _line, detail = _stash_line(stash)
    body = ["what he keeps at his own camp"]
    body += detail if detail else ["   nothing yet"]
    body += ["",
             "his, not the colony's: the stockpile above",
             "cannot draw on this and he never draws on it",
             "",
             "he builds his hut from it, feeds his fire",
             "from it and eats out of it - and whoever",
             "takes the title after him inherits the pile"]
    return f"{name}'s stash", tuple(body)


def _build(world, show_roster: bool,
           budget: int | None = None) -> tuple[pygame.Surface, list, list]:
    agents = [a for a in world.population.alive_agents()]
    agents.sort(key=lambda a: a.id)

    # Hit regions for the hover tooltips, in this panel's own coordinates. Built
    # here rather than in a second pass so a zone cannot drift from the text it
    # names: every rect below is measured from the same y the blit above it used.
    zones: list[tuple[pygame.Rect, str, tuple[str, ...]]] = []
    #: Roster rows only, paired with the agent id they belong to. Same rects,
    #: same coordinate space, different question: `zones` answers "what do I say
    #: about this?", `agent_zones` answers "who is this?".
    agent_zones: list[tuple[pygame.Rect, int]] = []
    f11 = _font(11)

    rows = len(agents) if show_roster else 0
    # The hermit's row is one line taller than everybody else's - it carries his
    # stash under his name - and the panel's height is computed up front, so the
    # extra line has to be counted here or the roster runs off the bottom of the
    # surface. At most one hermit is ever appointed; `min(1, ..)` is belt and
    # braces against a mangled save carrying two, which `assign_roles` demotes on
    # its next pass but which must not clip the panel in the meantime.
    stash_rows = min(1, sum(1 for a in agents if _is_hermit(a))) if show_roster else 0

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
    def _height_for(extra_rows: int) -> int:
        return (PAD * 2 + LINE * 3 + 6 + (rows * (LINE + 9)) + LINE * 3 + 8
                + (LINE - 2 if show_roster else 0) + (LINE if ufo_line else 0)
                + extra_rows * (LINE - 3))

    # THE STASH LINE IS AFFORDABLE OR IT IS NOT, and this is the fix to a claim
    # that was exactly inverted. It was said to cost a roster slot only where
    # capacity already exceeded MAX_POP; measured against the real panel it did
    # the opposite - 36 rows with and without at HUD_SCALE 1.0, 20 and 20 at
    # 1.6, but 11 without and 10 WITH at 2.5, and 6 against 5 at 3.5. MAX_POP is
    # 20, so the one extra line cost a colonist precisely at the scales where
    # rows were already scarce and never at the scales with room to spare.
    #
    # The cause is that a taller panel is not clipped, it is blitted off the
    # bottom of the target: `_build` lays out at authored size and `_scaled`
    # multiplies the finished surface by HUD_SCALE, so 11 authored px become 27
    # at 2.5 and push the last colonist past the frame edge. `draw_stats` now
    # measures the target and hands the budget down, so the extra line is taken
    # only when the panel still fits with it. When it does not, the row is not
    # lost - the stash zone moves onto his NAME, so the pointer still explains
    # his pile, and the panel keeps the colonist instead.
    height = _height_for(stash_rows)
    if stash_rows and budget is not None and height > int(budget):
        stash_rows = 0
        height = _height_for(0)
    show_stash_line = bool(stash_rows)
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
        ufo_text = f"taken {taken}" + (f", back {back}" if back else "")
        ut = _text(ufo_text, DIM, 11)
        panel.blit(ut, (PAD, y))
        # The one row that arrived without a hover zone, because the line and
        # the zones were written on separate branches. It needs one more than
        # most: "taken 2" is the panel's only counter that is neither a death
        # nor a birth, and the tooltip is where that can actually be said.
        zones.append((pygame.Rect(PAD, y, max(ut.get_width(), 40), LINE),
                      "Taken by the lights",
                      (f"{taken} lifted away by the ufo",
                       f"{back} of them put back",
                       "",
                       "not counted among the dead - they leave no grave, "
                       "and some come home")))
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
        # hp is called out separately from the three needs because it is the one
        # bar on the row where a full bar is good news.
        panel.blit(_text("hp=health (full=good) | hun tir cld (full=bad)",
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
            # ...and, outside that, the camera lock. Same idiom as the candle
            # ring one radius further out, so a candle-carrying hermit who is
            # also being followed reads as two rings rather than as one mark
            # that changed colour. r=8 from a centre at x=PAD+4=13 leaves 5 px
            # of panel to its left, so nothing clips.
            if FOLLOW_ID is not None and getattr(a, "id", None) == FOLLOW_ID:
                pygame.draw.circle(panel, FOLLOW_RING, (PAD + 4, y + 5), 8, 1)

            name = str(a.name)[:11]
            name_s = _text(name, TITLE, 11, True)
            panel.blit(name_s, (PAD + 13, y - 1))
            name_rect = pygame.Rect(PAD + 13, y - 2, name_s.get_width(), LINE - 2)

            doing = _doing(a)
            maxw = PANEL_W - PAD - 78
            ds = _text(doing, DIM, 10)
            if ds.get_width() > maxw:
                while len(doing) > 4 and _text(doing + "..", DIM, 10).get_width() > maxw:
                    doing = doing[:-1]
                ds = _text(doing + "..", DIM, 10)
            panel.blit(ds, (PAD + 76, y))
            y += LINE - 2

            # Health, then the three needs. hp leads deliberately: it is the one
            # that means someone is about to die, and it is also the one that
            # runs the other way (full is good), so putting it first and giving
            # it its own colour ramp stops the eye reading the row as four bars
            # of the same thing.
            #
            # bw drops 20 -> 16 and the gap 6 -> 5 to buy the fourth column. Four
            # at the old width ran to x=190 and the right-aligned role label
            # starts around 219 on a nine-character role, which is a 29 px margin
            # - too close to trust with a font the system chooses at runtime.
            # At 16 the row ends at 162 and the margin is 57.
            bw = 16
            bx = PAD + 13
            # Spelled out rather than h/f/c: those read as health/food, which
            # is neither what they measure nor which way round they run.
            hp = _health_frac(a)
            for lbl, val, col in (
                ("hp", hp, _health_color(hp)),
                ("hun", float(a.hunger), _need_color(float(a.hunger))),
                ("tir", float(a.fatigue), _need_color(float(a.fatigue))),
                ("cld", float(a.warmth), _need_color(float(a.warmth))),
            ):
                panel.blit(_text(lbl, DIM, 9), (bx, y - 2))
                _bar(panel, bx + 14, y + 1, bw, 4, val, col)
                bx += 14 + bw + 5
            role = str(getattr(a, "role", ""))[:9]
            rs = _text(role, DIM, 9)
            panel.blit(rs, (PANEL_W - PAD - rs.get_width(), y - 2))
            y += 11

            # ---- and, for the one man who has one, his stash ---------------
            # A third line on a two-line row, and only on his, and only when the
            # frame has the height for it (`show_stash_line`, decided up in the
            # height block). It is the only private store in the game and the
            # panel has nowhere else to say so: the stockpile line at the top is
            # the COLONY's and cannot include his without lying about who can
            # spend it. When the line cannot be afforded the tooltip still can,
            # so the information is never lost, only moved under the pointer.
            if _is_hermit(a):
                stash = _stash_of(world)
                # APPENDED BEFORE THE AGENT ZONE, WHICH IS LOAD-BEARING.
                # `_hover` returns the FIRST zone the pointer is inside, and the
                # agent zone below covers this whole block - so a stash zone
                # added after it could never be reached and the one row on the
                # panel that most needs explaining would be the one row that
                # does not explain itself.
                if show_stash_line:
                    line, _detail = _stash_line(stash)
                    sl = _text(line, DIM, 9)
                    panel.blit(sl, (PAD + 13, y - 1))
                    zones.append((pygame.Rect(PAD + 13, y - 2,
                                              max(sl.get_width(), 60), LINE - 2),
                                  *_stash_tip(str(getattr(a, "name", "?")),
                                              stash)))
                    y += LINE - 3
                else:
                    # No room for the line at this HUD_SCALE - see the height
                    # block above. The pile is still one hover away: the zone
                    # goes over his NAME, and only his name, so the rest of his
                    # row still shows the ordinary agent tooltip. It costs the
                    # panel nothing.
                    zones.append((name_rect,
                                  *_stash_tip(str(getattr(a, "name", "?")),
                                              stash)))

            # One zone for the whole two-line block rather than four small ones
            # per agent. Splitting it would mean the tooltip changed under the
            # pointer as it drifted a few pixels between a name and the bar
            # beside it, and everything it would say is on the one plate anyway.
            title, body = _agent_tip(a)
            row_rect = pygame.Rect(PAD, row_top, PANEL_W - PAD * 2,
                                   max(1, y - row_top))
            zones.append((row_rect, title, body))
            # ...and the same rect again, on its own list, carrying WHO it is.
            # Clicking a name follows that colonist, and the hit test wants an
            # id rather than a tooltip. It is a second list instead of a fourth
            # tuple element because `_hover` and tools/smoke.py both unpack
            # `_panel_zones` as a 3-tuple, and widening it to teach one caller
            # about ids would have made every other caller learn it too.
            aid = getattr(a, "id", None)
            if aid is not None:
                agent_zones.append((row_rect, int(aid)))

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

    return panel, zones, agent_zones


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

#: Colour for a chronicle entry that is a death or a taking, and one that is a
#: birth or a homecoming. Everything else stays the ordinary grey - if half the
#: log is coloured, none of it is emphasis.
LOG_LOSS = BAD
LOG_GAIN = GOOD


def _template_res(keys) -> "list[re.Pattern[str]]":
    """Regexes matching any rendered form of the named event templates.

    Built from ``sim.names.EVENT_TEMPLATES``, which is the actual source of
    these lines, rather than from a hand-written keyword list. Keywords looked
    tempting and are a trap here: "The cold took {name}" is a death and "The
    lights take hold of {name}" is an abduction, and a matcher keyed on "took"
    or "take" cannot tell a colonist freezing from the ufo arriving. Deriving
    from the templates also means a new death line written next year is coloured
    without anyone remembering this file exists.

    ``_tidy`` capitalises the first letter and adds a full stop, so matching is
    case-insensitive and unanchored at the end.
    """
    import re
    out = []
    try:
        from ..sim.names import EVENT_TEMPLATES
    except Exception:
        log.debug("chronicle templates unavailable; log stays monochrome",
                  exc_info=True)
        return out
    for key in keys:
        for tmpl in EVENT_TEMPLATES.get(key, ()) or ():
            # {name}/{where}/{cause}/{parent}/{other} -> "something non-empty".
            pat = re.sub(r"\\\{[a-z_]+\\\}", ".+?", re.escape(str(tmpl)))
            try:
                out.append(re.compile(r"^\s*" + pat.rstrip("\\.") , re.I))
            except re.error:
                continue
    return out


def _extra_res(patterns) -> "list[re.Pattern[str]]":
    import re
    return [re.compile(p, re.I) for p in patterns]


#: Deaths come from the died_* family. The taking does NOT - ufo.py writes it as
#: a plain f-string - so it is listed explicitly, and so are the events on the
#: other side that no template covers: the founding party, the ufo handing
#: somebody back, and the cairnstone raising one of the dead.
_LOSS_RES = _template_res([k for k in
                           ("died", "died_age", "died_cold", "died_disintegrated",
                            "died_fall", "died_fire", "died_flood", "died_hunger",
                            "died_lightning", "died_meteor", "died_mudslide",
                            "died_quake")]) + _extra_res([
    r"^\s*.+? is taken by the lights",
])
_GAIN_RES = _template_res(["born", "arrived"]) + _extra_res([
    r"^\s*A group of \d+ arrives",
    r"^\s*The lights give .+? back",
    r"sets the wyrm's stone on the grave",
    r"^\s*The wyrm's stone goes down on a grave",
])


def _log_tone(text: str) -> "tuple[int, int, int] | None":
    """LOG_LOSS, LOG_GAIN, or None for an ordinary line. Never raises."""
    try:
        s = str(text)
        for rx in _LOSS_RES:
            if rx.search(s):
                return LOG_LOSS
        for rx in _GAIN_RES:
            if rx.search(s):
                return LOG_GAIN
    except Exception:
        pass
    return None


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
    # (text, tone) per ROW, not per entry: the tone has to survive the wrap or a
    # death long enough to take two lines would be red on the first and grey on
    # the second. Classified once per entry, before wrapping, because half a
    # sentence does not match a template.
    lines: list[tuple[str, tuple[int, int, int] | None]] = []
    for entry in tail:
        text = str(entry)
        if "] " in text:
            text = text.split("] ", 1)[1]
        tone = _log_tone(text)
        wrapped = _wrap(text, LOG_W - PAD * 2, 11)
        lines.extend((w, tone) for w in wrapped[:2])   # one event never eats the log

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
    for i, (ln, tone) in enumerate(lines):
        # Oldest at the top, dimmest; newest at the bottom, brightest. The age
        # fade multiplies the tone rather than replacing it, so an old death is
        # a dim red and not a grey line - the colour is what the entry IS, and
        # the brightness is how recent it is. Those are two different facts and
        # the log shows both at once.
        f = 0.45 + 0.55 * (i + 1) / total
        base = tone if tone is not None else DIM
        col = (int(base[0] * f), int(base[1] * f), int(base[2] * f))
        panel.blit(_text(ln, col, 11), (PAD, y))
        y += row
    return panel
