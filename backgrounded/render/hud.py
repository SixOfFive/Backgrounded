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

import pygame

from ..constants import (
    DAY_LENGTH_SEC, RES_COOKED, RES_FIBRE, RES_FOOD, RES_STONE,
    RES_WOOD, SCENE_LABELS,
)

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
               offset: tuple[int, int] = (0, 0)) -> None:
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
    """
    global _panel_cache, _panel_key
    try:
        key = _content_key(world, show_roster)
        if _panel_cache is None or key != _panel_key:
            _panel_cache = _build(world, show_roster)
            _panel_key = key
        if _panel_cache is not None:
            panel = _scaled(_panel_cache, "stats")
            # max(0, ...) so a window narrower than the panel still shows its
            # left edge rather than pushing it off screen entirely. Measured off
            # the *scaled* width, or the anchor would drift by the scale factor.
            x = max(0, surf.get_width() - panel.get_width() - MARGIN)
            surf.blit(panel, (x + int(offset[0]), MARGIN + int(offset[1])))
    except Exception:
        log.exception("hud draw failed")


def _build(world, show_roster: bool) -> pygame.Surface:
    agents = [a for a in world.population.alive_agents()]
    agents.sort(key=lambda a: a.id)

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

    if ufo_line:
        panel.blit(_text(f"taken {taken}" + (f", back {back}" if back else ""),
                         DIM, 11), (PAD, y))
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
    # Wrap onto up to two lines rather than clipping mid-word. The height
    # calc below reserves two footer lines, so this never overruns the panel.
    for line in _wrap(str(last), PANEL_W - PAD * 2, 10)[:2]:
        panel.blit(_text(line, DIM, 10), (PAD, y))
        y += LINE - 3

    return panel


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
