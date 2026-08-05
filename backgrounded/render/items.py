"""Relics: the five dragon drops, in the dirt and on the body.

Five items fall off dead dragons and nothing else in this world looks like
them. A spear is a stick, a jerkin is a hide, a hut is timber - the whole
palette is earth, and it has to stay that way for the relics to be *readable as
magic* rather than as one more brown thing on a brown hill. So every drawing
here carries a colour the rest of the game does not own:

    dragonscale   sage-green plate + an emerald rim   (the quadruped's hide)
    amulet        mint-white bead                     (the wyvern's ward)
    bfg9000       jade barrel + acid-green core       (the serpent's gun)
    ironshod      iron grey + a cold cyan spark       (the skeletal's boots)
    cairnstone    bone-pale stone + a violet rune     (the skeletal's stone)

The hue *is* the tell. At 26 px there is no room for filigree, and the first
cut proved it: a "rune-etched" cairnstone with three scratches on it and a
gold-inlaid barrel both landed as noise at 1.0x and as a single grey pixel at
0.5x. What survives the shrink is a **flat silhouette in a colour, plus one hot
core pixel** - which is the same lesson ``creatures._draw_spear`` records about
the knapped head, applied to five objects instead of one.

Four drawings, three layers
---------------------------
* :func:`draw_relics` - relics lying in the dirt. Belongs at **step 7**, with
  the thrown spears and *inside* the light composite: a relic on the ground
  lives in the same 0-140 px band as the wolves and must be lit by the same
  torches, or it is the one object in the colony that glows at 3am. That is the
  artefact ``renderer._draw_dragons`` had to unpick for the quadruped and
  ``render/throwing.py`` records for the spear.
* :func:`draw_relic_fx` - everything **self-lit**: the halo around a relic on
  the ground, the BFG beam, and the saucer coming apart. Belongs *after* the
  light composite, next to ``creatures.draw_ufo``, for the same reason the beam
  and the lava flow are drawn there - a multiply by a night-storm ambient of
  ~0.06 does not dim a glow, it deletes it. Splitting the ground relic across
  the two passes is deliberate and is exactly what ``renderer.draw`` already
  does for lava (flat band under the composite, flow light over it): the
  *object* is lit like a spear, the *magic* is not.
* :func:`draw_worn` - the relic on a body, called from
  ``creatures.draw_gear`` with the skeleton it has already solved. Inside the
  composite, so a worn relic dims with its wearer. A bead that stayed bright
  through a silhouetted figure detaches from the body and floats.

The two spectacle moments
-------------------------
The BFG firing and the saucer exploding are the only two things in this game
that happen *at* the player rather than in front of them, and both are over in
under a second, so both are drawn hot:

* :func:`draw_beam` - a corridor-wide additive bar with a white core, a muzzle
  bloom and a much bigger bloom where it stopped. The bar is baked with numpy
  and cached by (length, width, intensity, angle) bucket, so the ~8 frames a
  shot is on screen cost one rotate and two blits after the first.

  **The beam's ``t`` is not one unit.** The two producers disagree - the
  registry ages its records, ``combat_actions`` stamps its with the world clock
  - and reading the second as the first is what kept this drawing off the screen
  entirely in any world older than BEAM_LIFE. :func:`_iter_beams` tags each
  record with which convention it follows and :func:`_draw_beam_record` converts;
  neither of them guesses from the value, because near t = 0 the two readings
  agree and the guess would be right for exactly as long as it did not matter.
* :func:`draw_misfire` - **the wielder alone**. Not a blast radius. The gun
  comes apart in his hands and takes him and nobody else; the corridor shot is
  the one that can wipe a colony. :data:`_MISFIRE_R` carries the reasoning and
  the number.
* :func:`draw_wreck` - the ward detonation: a mint flash, an expanding shock
  ring, arcs crawling over the hull, and a spark-and-smoke trail behind it all
  the way down. The falling hull itself is *not* drawn here -
  ``creatures._draw_ufo`` already draws the saucer wherever ``y`` puts it, and
  PHASE_WRECK is not in its skip list, so the fall comes for free.

Wiring
------
Three call sites, none of them in this file (``renderer.py``,
``creatures.py`` and ``stickfigure.py`` are owned elsewhere)::

    # renderer.draw, step 7, right after spear_art.draw_spears:
    items.draw_relics(s, world, world.world_time)

    # renderer.draw, after the light composite, beside creatures.draw_ufo:
    items.draw_relic_fx(s, world, world.world_time)

    # creatures.draw_gear, where it already resolves sk / flat / fade - and
    # note its early-return has to learn about relics, or a BFG carrier (no
    # spear, no armour) returns before anything is drawn:
    relic = items.relic_of(agent)
    if not (has_spear or has_armour or relic):
        return
    if has_armour and not items.covers_torso(relic):
        _draw_armour(surf, sk, flat, fade)
    ...
    if relic:
        items.draw_worn(surf, agent, sk, t, flat, fade, kind=relic)
        # ...and the head back on top, for the one relic big enough to take
        # it. See items.hides_head - the BFG covers 91-97% of the head disc
        # in the poses that raise the fists to head height.
        if base is not None and items.hides_head(relic, sk):
            _disc(surf, tuple(int(c) for c in base[:3]), sk.head, sk.head_r)

``stickfigure._Disarmed`` - the stand-in it hands ``draw_gear`` for the poses
that put a weapon down - needs a ``relic`` field too, or a sleeping, dead or
cartwheeling villager loses their coat and their boots along with their spear.
Passing the relic through *except* a BFG (which is a held weapon and should be
set down with the spear) is the whole of that change.

Read-only w.r.t. sim state
--------------------------
Nothing here writes to a World, a registry, a relic or the saucer. Every field
is read through :func:`fx.attr_num` or ``getattr`` with a fallback and several
plausible names, so a rename on the sim side degrades to a default rather than
a traceback - the rule the rest of this package keeps.

There is **no random number anywhere in the drawing path** (the ``__main__``
contact sheet keeps a ``random.Random(4)`` because ``Population.spawn`` wants
one to build test villagers with). Every scatter (the lean
of a dropped relic, the spark trail off the wreck, the mote rising off a halo)
is a pure function of a persisted id and the render clock, using the same
``id * 0.9137`` phase-decorrelating trick ``creatures`` uses for animal gait.
Render has no seeded stream to draw from and is not allowed to invent one; a
lean re-rolled per frame is a relic vibrating in the dirt, and one re-rolled
per load is a relic that moves when you reopen the game.

Cost
----
The common case is a colony with no relics at all - 62.5% of them never earn a
dragon kill inside 90 minutes - and :func:`draw_relics` costs **two attribute
reads and a return** for it. Measured against a real seed-7 world at 1600x1000
with a genuine ``sim.items.RelicRegistry`` hung on it, 300 frames each:

    no registry / empty     draw_relics 0.32 us   draw_relic_fx  2.2 us
    five relics down        draw_relics   68 us   draw_relic_fx   75 us
    ... + a beam + misfire  draw_relics   80 us   draw_relic_fx  169 us

``draw_relic_fx``'s empty case is not free (2.2 us) because it still has to ask
the saucer whether it is exploding; that is 0.03% of a 7.3 ms frame and buys the
wreck the right to need no state of its own. Five relics plus their halos is
2.0% of a frame against the MAX_RELICS = 6 cap, and a beam and a misfire
together add another 1.3% on the ~9 frames a discharge is visible, a few times
a wielder-hour.

The five relics are 458 changed pixels on that frame and their halos 11405 -
the halo is 25x the ink of the object, which is the whole point: at 13 px a
relic in the dirt is found by its light, not by its shape.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Sequence

import numpy as np
import pygame

from ..constants import RENDER_H, RENDER_W, UFO_WRECK_FALL_SEC
from ..sim.entities import AGENT_HEIGHT
from . import fx

log = logging.getLogger(__name__)

__all__ = [
    "draw_relics", "draw_relic_fx", "draw_worn", "draw_relic",
    "draw_beam", "draw_misfire", "draw_wreck", "covers_torso", "hides_head",
    "relic_of",
    "KINDS", "BEAM_LIFE", "clear_caches", "cache_stats",
]

TAU = math.tau
Color = tuple[int, int, int]

# The relic vocabulary. Imported when it exists and mirrored as literals when
# it does not, because this module and the constants land in the same change
# from different lanes and a render package that will not *import* takes the
# whole app down at startup - a far worse failure than a colour drift.
# render/throwing.py sets the precedent (it does this with the spear palette)
# and the fallback is safe for a different reason here: these five strings are
# the **wire format**. They are what a save file contains, so a literal copy is
# not an approximation of the constant, it is the same datum.
# BFG_BLAST_R is deliberately NOT imported. It used to size the misfire, and
# the misfire is now wielder-only by spec, so importing it here would leave a
# 70 px number one edit away from the drawing that must never use it again.
try:
    from ..constants import (
        BFG_CORRIDOR,
        RELIC_AMULET,
        RELIC_BFG,
        RELIC_BOOTS,
        RELIC_CAIRN,
        RELIC_DECAY_SEC,
        RELIC_SCALE,
    )
except ImportError:                                    # pragma: no cover
    log.info("constants.py has no relic block yet; using the wire literals")
    RELIC_SCALE = "dragonscale"
    RELIC_AMULET = "amulet"
    RELIC_BFG = "bfg9000"
    RELIC_BOOTS = "ironshod"
    RELIC_CAIRN = "cairnstone"
    BFG_CORRIDOR = 22.0
    RELIC_DECAY_SEC = 1800.0

#: Draw order is not this order - :func:`draw_relics` paints in list order, and
#: relics do not overlap enough for it to matter.
KINDS: tuple[str, ...] = (RELIC_SCALE, RELIC_AMULET, RELIC_BFG,
                          RELIC_BOOTS, RELIC_CAIRN)

# The primitives and the silhouette rule, from the module that draws the gear
# these sit beside. NOT a copy: a relic drawn with a different line helper than
# the spear in the same fist antialiases differently and reads as a different
# object, which is exactly the bug a duplicated primitive produces six months
# later. The fallback exists so a rename over there costs this module its
# polish rather than costing the renderer its import.
try:
    from .creatures import (          # noqa: PLC2701 - deliberate, see above
        _SIL_COLOR,
        _SIL_CUTOFF,
        _dim,
        _disc,
        _find_ufo,
        _line,
        _lit,
        _poly,
        _state_of,
    )
except Exception:                                      # pragma: no cover
    log.warning("creatures.py did not yield its primitives; using copies",
                exc_info=True)
    _SIL_CUTOFF = 0.30
    _SIL_COLOR = (14, 15, 22)

    def _dim(col: Sequence[int], f: float) -> Color:
        return (int(col[0] * f), int(col[1] * f), int(col[2] * f))

    def _lit(col: Sequence[int], f: float) -> Color:
        return (int(min(255.0, col[0] + (255 - col[0]) * f)),
                int(min(255.0, col[1] + (255 - col[1]) * f)),
                int(min(255.0, col[2] + (255 - col[2]) * f)))

    def _line(surf: pygame.Surface, col: Sequence[int],
              a: tuple[float, float], b: tuple[float, float], w: int = 1) -> None:
        try:
            pygame.draw.aaline(surf, col, a, b, w)
        except TypeError:
            pygame.draw.aaline(surf, col, a, b)

    def _disc(surf: pygame.Surface, col: Sequence[int],
              c: tuple[float, float], r: float, width: int = 0) -> None:
        if r > 0.0:
            pygame.draw.circle(surf, col, (int(c[0]), int(c[1])),
                               max(1, int(r)), width)

    def _poly(surf: pygame.Surface, col: Sequence[int],
              pts: Sequence[tuple[float, float]]) -> None:
        if len(pts) >= 3:
            pygame.draw.polygon(
                surf, col, [(int(round(p[0])), int(round(p[1]))) for p in pts])

    def _state_of(obj: Any) -> str:
        for name in ("state", "mode", "phase"):
            val = getattr(obj, name, None)
            if isinstance(val, str) and val:
                return val.lower()
        return ""

    def _find_ufo(world: Any) -> Any:
        for holder, name in ((world, "ufo"),
                             (getattr(world, "events", None), "ufo")):
            if holder is not None and getattr(holder, name, None):
                return getattr(holder, name)
        return None


# --------------------------------------------------------------------------
# palette
#
# Every one of these is outside the earth palette on purpose, and they were
# picked to be distinguishable *from each other* at 6 px, which is what a
# ground relic occupies at 0.5x. The three greens (scale sage, bfg jade,
# amulet mint) are the risk in that set and they are separated by form, not by
# hue: a coat is a mound, a gun is a bar, an amulet is a dot on a loop.
# --------------------------------------------------------------------------

# dragonscale - the quadruped's hide, so it borrows that dragon's own two
# tones (dragons._qd_HIDE / _qd_RIM) rather than inventing a green. A player
# who watched the thing die should recognise what is on the wearer's back.
_SCALE_PLATE: Color = (92, 104, 88)
_SCALE_RIM: Color = (176, 196, 162)
_SCALE_DEEP: Color = (48, 60, 50)
_SCALE_HOT: Color = (140, 255, 176)      # the magic: an emerald sheen

# amulet - the ward. Mint-white, and deliberately the same hue as the flash the
# detonation throws (180, 255, 210), so the bead and the explosion it causes
# are visibly the same magic.
_AMULET_CORD: Color = (74, 82, 92)
_AMULET_BEAD: Color = (180, 255, 210)
_AMULET_HOT: Color = (236, 255, 246)

# bfg9000 - the serpent's jade with its crest gold for the fittings, plus an
# acid core that is the same green as the beam. Nothing else in the game is
# this colour.
_BFG_BODY: Color = (72, 128, 94)
_BFG_DARK: Color = (30, 54, 42)
_BFG_TRIM: Color = (222, 172, 74)
_BFG_CORE: Color = (168, 255, 96)
_BEAM_GREEN: Color = (150, 255, 110)
_BEAM_WHITE: Color = (232, 255, 220)

# ironshod - iron, which is *not* in this game's palette either: every tool
# here is stone and wood. The cyan spark is what stops a grey boot reading as
# a pebble.
_IRON: Color = (128, 136, 150)
_IRON_DARK: Color = (58, 64, 76)
_IRON_HOT: Color = (150, 226, 255)

# cairnstone - bone from the skeletal dragon, lit by a violet nothing else in
# the world owns. Grave magic should not look like any other magic here.
_CAIRN_STONE: Color = (198, 192, 178)
_CAIRN_SHADE: Color = (112, 106, 98)
_CAIRN_RUNE: Color = (196, 140, 255)

#: Halo colour and relative strength per kind, for the self-lit pass.
_HALO: dict[str, tuple[Color, float]] = {
    RELIC_SCALE: (_SCALE_HOT, 0.62),
    RELIC_AMULET: (_AMULET_BEAD, 0.78),
    RELIC_BFG: (_BFG_CORE, 0.70),
    RELIC_BOOTS: (_IRON_HOT, 0.50),
    RELIC_CAIRN: (_CAIRN_RUNE, 0.72),
}

#: Nominal size of a relic on the ground, in pixels, before the per-kind
#: geometry below scales it. A villager is 26 px, a spear 25 and a gravestone
#: 16; 0.42 was the first cut and the contact sheet said it read as gravel -
#: at 0.50 (13 px) a relic is unmistakably an *object somebody could pick up*
#: and is still 6-7 px at the 0.5x wallpaper zoom, where the five have to stay
#: distinguishable from each other.
_UNIT = AGENT_HEIGHT * 0.50

#: How far off-screen a relic may sit before it stops being drawn. The widest
#: drawing (the barrel) is ~16 px at 1.0x; 40 is twice clear.
_CULL = 40.0

#: Seconds of RELIC_DECAY_SEC over which an unclaimed relic dims out. Long
#: enough that a fading relic reads as *going*, short enough that it is not
#: dim for most of its life: a relic at full brightness for 28 minutes and then
#: gone in 90 s is the shape of an object with a clock on it.
_DECAY_FADE = 90.0

#: Seconds a freshly dropped relic takes to come up to full. Short - this is a
#: landing, not an entrance.
_DROP_FADE = 0.8

#: Seconds a BFG discharge stays on screen, and it is the *sim's* number: the
#: registry drops a beam record from ``beams`` at BEAM_LIFE_SEC, so a render
#: envelope that ran longer would draw a beam that no longer exists and one
#: that ran shorter would leave a live record drawing nothing. Imported rather
#: than copied for exactly that reason, with a literal fallback so a rename
#: over there costs the effect its timing instead of costing the renderer its
#: import.
try:
    from ..sim.items import BEAM_LIFE_SEC as BEAM_LIFE
except Exception:                                      # pragma: no cover
    BEAM_LIFE = 0.30


# --------------------------------------------------------------------------
# adapters - everything that reads sim state lives here
# --------------------------------------------------------------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute or dict key, whichever this object turns out to be.

    The BFG shot record is a dict (``events.strikes`` is the precedent - the
    lightning geometry render consumes is a list of plain dicts) and the relics
    themselves are dataclasses. One reader covers both, and neither lane has to
    care which the other picked.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _num(obj: Any, *names: str, default: float = 0.0) -> float:
    """``fx.attr_num`` that also looks inside a dict."""
    if isinstance(obj, dict):
        for name in names:
            val = obj.get(name)
            if val is None:
                continue
            try:
                f = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                return f
        return float(default)
    return fx.attr_num(obj, *names, default=default)


def _ground_y(world: Any, x: float, fallback: float) -> float:
    """Terrain height under *x*, or *fallback*.

    A relic carries the y the sim wrote when it dropped, so re-reading this
    looks redundant - it is not, and render/throwing.py records the same bug for
    grounded spears. Terrain *deforms* after the fact (earthquake fissures, the
    chasm crossings) and the sim never revisits a relic once it is lying still,
    so the stored y goes stale and the relic ends up floating over a new
    fissure. One call per relic, capped at MAX_RELICS = 6.
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


def relic_of(agent: Any) -> str:
    """The relic *agent* is wearing, or ``""``. Never raises.

    Duck-typed rather than typed against ``Stickman.relic`` because
    ``stickfigure._Disarmed`` hands ``creatures.draw_gear`` a stand-in object
    for the poses that set a weapon down, and this must degrade to "no relic"
    for anything that does not carry the field rather than blow up the figure.
    """
    try:
        val = getattr(agent, "relic", "")
    except Exception:
        return ""
    if not isinstance(val, str):
        return ""
    val = val.strip()
    return val if val in KINDS else ""


def covers_torso(relic: str) -> bool:
    """True when this relic replaces the leather jerkin on the chest.

    ``creatures.draw_gear`` asks this before drawing ``_draw_armour``: the
    dragonscale mirrors itself into ``agent.armour`` (that is what makes the
    damage path work with no edit), so an agent wearing it passes the armour
    test and would get *both* a jerkin and a scale coat stacked on one torso.
    """
    return relic == RELIC_SCALE


def hides_head(relic: str, sk: Any = None) -> bool:
    """True when this relic's worn drawing lands on the wearer's face.

    ``creatures.draw_gear`` asks this straight after :func:`draw_worn` and
    re-stamps ``sk.head`` in the colour it already has when the answer is yes,
    so the gun ends up *behind* the wearer's face and in front of everything
    else. Asked rather than assumed for the same reason :func:`covers_torso` is:
    the stacking rule belongs to the module that knows how big each drawing is,
    and a sixth relic that also needs it should need no edit over there.

    ``stickfigure`` puts worn gear in front of the whole figure, head included,
    and says why - an overhead chop swings the spear past the face, and it
    should. That decision was made against a 2 px shaft. Measured across the
    pose set with the BFG at its drawn size, the gun covered **91% of the head
    disc in chop, 93% in panic and 97% in dance**: poses that put both fists at
    or above head height, where a 10 px block centred on them lands exactly on
    the face. The spear crosses the same three heads at 36 / 67 / 7% and gets
    away with it because a scratch across a face is still a face. Bulk does not
    - the villager reads as headless, and a stickman with no head is not
    carrying something impressive, he is gone.

    The sag in :func:`_bfg_frame` is the other half of this and was tried as the
    whole of it first: on its own it took chop to 55% and climb to 56% but
    pushed *panic* to 100%, because those hands are above the head rather than
    beside it and any offset that clears one lands on the other.

    **Given a skeleton this is a geometry test, not a constant**, and that is
    worth the dozen lines. Answering an unconditional yes re-stamps the head on
    every frame of every pose, including the ten where the gun is nowhere near
    it, and ``_disc`` antialiases - a disc drawn twice on itself blends its
    fringe twice, so a BFG carrier ends up with a visibly harder head than
    everyone else. Measured: an unconditional stamp changed 33-38% of the head's
    pixels in poses where the gun does not come within 10 px of it.
    """
    if relic != RELIC_BFG:
        return False
    if sk is None:
        return True                   # no geometry offered: assume the worst
    try:
        frame = _bfg_frame(sk)
        head = getattr(sk, "head", None)
        if frame is None or head is None:
            return False
        breech, ux, uy, _px, _py, ln, hw, _s, _f, _r = frame
        # Distance from the head centre to the barrel's axis *segment*. The hull
        # is not a capsule, but its widest half-section is 1.95 hw and its
        # narrowest 0.95, so testing against the widest is the conservative
        # answer - it re-stamps a little more often than strictly needed and
        # never misses a face.
        dx, dy = head[0] - breech[0], head[1] - breech[1]
        along = fx.clamp(dx * ux + dy * uy, 0.0, ln)
        ox = dx - ux * along
        oy = dy - uy * along
        reach = float(getattr(sk, "head_r", 2.0)) + hw * 1.95
        return (ox * ox + oy * oy) < reach * reach
    except Exception:
        log.debug("hides_head geometry failed", exc_info=True)
        return False


def _iter_relics(world: Any) -> list[Any]:
    """Every relic the world is holding, however it is holding it.

    Fast path first and on purpose: the common case is "this colony has never
    killed a dragon", and it must cost two attribute reads and a falsy test -
    no method call, no list copy, no allocation. ``.all()`` is the polite API
    and it builds a list every frame forever to tell us it is empty.
    """
    reg = getattr(world, "relics", None)
    if reg is None:
        # ``sim/items.py`` looks under both names for its own registry (the
        # module is items.py and the attribute is relics), so a reader that
        # knew only one of them would go silently inert on the other.
        reg = getattr(world, "items", None)
        if reg is None:
            return []
    items = getattr(reg, "relics", None)
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
        return [r for r in items if r is not None]
    except TypeError:
        return []


def _world_now(world: Any, fallback: float) -> float:
    """The sim's clock, in seconds since the world was made.

    Needed because one of the two beam producers stamps its records against it
    rather than ageing them - see :func:`_iter_beams`. *fallback* is the render
    clock the caller was handed, which ``renderer.draw`` sets to exactly
    ``world.world_time``; it only has to answer for a synthetic world (a contact
    sheet, a test) that carries no clock at all.
    """
    for name in ("world_time", "time", "t"):
        val = getattr(world, name, None)
        if val is None:
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            return f
    return float(fallback)


def _iter_beams(world: Any) -> list[tuple[Any, bool]]:
    """Live BFG discharges, each tagged with what its ``t`` field *means*.

    The tag is the whole reason this returns pairs, and it exists because there
    are **two producers in this game and they do not agree about ``t``**:

    ``world.relics.beams`` - ``sim.items.RelicRegistry.add_beam``
        writes ``"t": 0.0`` and ``_age_beams`` adds ``dt`` to it every tick,
        dropping the record once it passes ``BEAM_LIFE_SEC``. ``t`` is an
        **age in seconds**. This is the API the contract named, and it currently
        has no callers anywhere in the repo.
    ``world.bfg_beams`` - ``sim.combat_actions._record_beam``
        writes ``"t": world_now(world)`` - the absolute world time of the shot -
        and never touches the record again; the list is only capped at eight.
        ``t`` is an **absolute timestamp**. This is the producer that actually
        fires in a real game.

    Reading the second with the first's convention is what kept the beam off the
    screen: a shot fired at world_time 6000 resolves to an age of 6000 s, which
    is 20000x BEAM_LIFE, so the envelope returned 0 and the single most
    spectacular thing this game draws had never been seen in any world older
    than a third of a second. The hostile proof measured bfg_shots 0 so it could
    not catch this; the beam is invisible for every shot that *does* fire.

    So the tag is not a guess about the data - it is the documented contract of
    whichever attribute the record was found on, and :func:`_draw_beam_record`
    converts one to the other. Interpreting the absolute list correctly is also
    what makes it safe to read a list nothing ever prunes: a stale stamp
    resolves to ``u >= 1`` and draws nothing.

    Anything missing still degrades rather than raising: no ``t`` draws the shot
    at full brightness for one frame, and a misfire ignores the endpoint.
    """
    # `is None` and not `or`: RelicRegistry defines __len__, so a registry
    # holding no relics is FALSY, and an `or` here would skip the beams of a
    # colony whose one relic is being carried - which is every colony that has
    # a BFG, because the man firing it is holding it.
    reg = getattr(world, "relics", None)
    if reg is None:
        reg = getattr(world, "items", None)
    for holder, name, absolute in ((reg, "beams", False),
                                   (reg, "shots", False),
                                   (world, "bfg_beams", True)):
        if holder is None:
            continue
        try:
            seq = getattr(holder, name, None)
        except Exception:
            continue
        if seq:
            try:
                return [(b, absolute) for b in seq if b is not None]
            except TypeError:
                continue
    return []


# --------------------------------------------------------------------------
# one relic, lying in the dirt
#
# All five are drawn in a local frame whose origin is the CONTACT POINT with
# the ground and whose unit is _UNIT * scale. Anchoring at the contact point
# rather than the centre is what makes a relic sit on a slope correctly with no
# per-kind offsets, and it matches every other ground object in the package
# (draw_grave, the planted spear).
# --------------------------------------------------------------------------


def _r_scale(surf: pygame.Surface, x: float, y: float, u: float,
             flat: Color | None, fade: float, lean: float) -> None:
    """A folded coat of plate. A mound, not a rectangle - a folded garment has
    no straight edges and the mound is what stops it reading as a crate."""
    w, h = u * 0.62, u * 0.42
    body = flat or _dim(_SCALE_PLATE, fade)
    rim = flat or _dim(_SCALE_RIM, fade)
    deep = flat or _dim(_SCALE_DEEP, fade)
    sx = 1.0 if lean >= 0.0 else -1.0

    def pt(lx: float, ly: float) -> tuple[float, float]:
        return (x + lx * w * sx, y + ly * h)

    _poly(surf, body, [pt(-1.0, 0.0), pt(-0.88, -0.62), pt(-0.30, -1.0),
                       pt(0.42, -0.92), pt(1.0, -0.34), pt(1.0, 0.0)])
    # Two scale courses across the fold. Arcs would be truer and cost a
    # Rect + an angle pair each; at 11 px a straight tick is the same three
    # pixels and never rounds to nothing.
    if flat is None:
        _line(surf, rim, pt(-0.72, -0.50), pt(0.10, -0.74), 1)
        _line(surf, rim, pt(-0.40, -0.20), pt(0.62, -0.42), 1)
        _line(surf, deep, pt(-1.0, 0.0), pt(1.0, 0.0), 1)


def _r_amulet(surf: pygame.Surface, x: float, y: float, u: float,
              flat: Color | None, fade: float, lean: float) -> None:
    """A bead on a slack cord. The cord is the whole drawing: a bead on its own
    is a berry, and this world already draws berries."""
    w = u * 0.52
    cord = flat or _dim(_AMULET_CORD, fade)
    bead = flat or _dim(_AMULET_BEAD, fade)
    sx = 1.0 if lean >= 0.0 else -1.0
    # Cord: two straight runs meeting at the bead, which at this size reads as
    # a dropped loop and costs two lines instead of an arc. Drawn 2 px wide -
    # at 1 px the first cut lost the cord entirely to the AA blend and the
    # relic became an anonymous sparkle on the ground.
    _line(surf, cord, (x - w * sx, y - u * 0.40), (x, y - u * 0.08), 2)
    _line(surf, cord, (x + w * 0.86 * sx, y - u * 0.36), (x, y - u * 0.08), 2)
    _disc(surf, bead, (x, y - u * 0.13), max(1.5, u * 0.20))
    if flat is None:
        _disc(surf, _lit(_AMULET_HOT, 0.35), (x, y - u * 0.15), max(1.0, u * 0.09))


def _r_bfg(surf: pygame.Surface, x: float, y: float, u: float,
           flat: Color | None, fade: float, lean: float) -> None:
    """A fat barrel lying on its side, muzzle toward the lean."""
    w, h = u * 0.86, u * 0.26
    body = flat or _dim(_BFG_BODY, fade)
    dark = flat or _dim(_BFG_DARK, fade)
    trim = flat or _dim(_BFG_TRIM, fade)
    sx = 1.0 if lean >= 0.0 else -1.0

    def pt(lx: float, ly: float) -> tuple[float, float]:
        return (x + lx * w * sx, y + ly * h)

    # Barrel as a flat-bottomed capsule: two tones split along the axis, which
    # is the only way a 6 px bar reads as round rather than as a plank.
    _poly(surf, body, [pt(-1.0, -0.30), pt(-0.86, -1.0), pt(0.86, -1.0),
                       pt(1.0, -0.30), pt(1.0, 0.0), pt(-1.0, 0.0)])
    _poly(surf, dark, [pt(-1.0, -0.28), pt(1.0, -0.28), pt(1.0, 0.0), pt(-1.0, 0.0)])
    if flat is None:
        _line(surf, trim, pt(0.80, -1.0), pt(0.96, 0.0), 2)      # muzzle ring
        _line(surf, dark, pt(-0.30, 0.0), pt(-0.18, 0.44), 2)    # the grip
        _disc(surf, _BFG_CORE, pt(-0.52, -0.52), max(1.0, u * 0.10))


def _r_boots(surf: pygame.Surface, x: float, y: float, u: float,
             flat: Color | None, fade: float, lean: float) -> None:
    """A pair. One boot is a boot; two side by side is *ironshod boots*, and
    the pair is what separates this from a lump of ore at 6 px."""
    iron = flat or _dim(_IRON, fade)
    dark = flat or _dim(_IRON_DARK, fade)
    w, h = u * 0.30, u * 0.46
    sx = 1.0 if lean >= 0.0 else -1.0
    for k in (-1, 1):
        ox = x + k * u * 0.34
        oy = y - (0.0 if k * sx > 0 else u * 0.04)   # one slightly askew

        def pt(lx: float, ly: float, ox: float = ox, oy: float = oy) -> tuple[float, float]:
            return (ox + lx * w * sx, oy + ly * h)

        # L-shape: sole forward, shaft up. Filled, because at 0.5x an outline
        # boot loses its interior and becomes a bracket.
        _poly(surf, iron, [pt(-0.6, 0.0), pt(-0.6, -1.0), pt(0.15, -1.0),
                           pt(0.15, -0.34), pt(1.0, -0.34), pt(1.0, 0.0)])
        if flat is None:
            _line(surf, dark, pt(-0.6, 0.0), pt(1.0, 0.0), 1)     # the iron sole
            # Cold spark on the cuff of the near boot only. On both boots it
            # read as a puddle of light with two grey lumps in it - the pair is
            # already the silhouette cue, so the colour only has to say
            # "not stone" once.
            if k > 0:
                _line(surf, _dim(_IRON_HOT, 0.75), pt(-0.6, -1.0), pt(0.15, -1.0), 1)


def _r_cairn(surf: pygame.Surface, x: float, y: float, u: float,
             flat: Color | None, fade: float, lean: float) -> None:
    """Three stones stacked, the top one cut with a rune."""
    stone = flat or _dim(_CAIRN_STONE, fade)
    shade = flat or _dim(_CAIRN_SHADE, fade)
    sx = 1.0 if lean >= 0.0 else -1.0
    for k, (ry, rw) in enumerate(((0.0, 0.52), (-0.34, 0.40), (-0.62, 0.26))):
        cy = y + ry * u
        hw = rw * u
        hh = max(1.2, u * 0.16)
        off = (k - 1) * u * 0.05 * sx
        _poly(surf, stone if k != 1 else shade, [
            (x + off - hw, cy), (x + off - hw * 0.72, cy - hh * 2.0),
            (x + off + hw * 0.72, cy - hh * 2.0), (x + off + hw, cy),
        ])
    if flat is None:
        # The rune: a hooked tick, not a glyph. A glyph at 3 px is a smudge -
        # the first cut drew three scratches and they merged into one pixel.
        top = (x + u * 0.05 * sx, y - u * 0.70)
        _line(surf, _CAIRN_RUNE, (top[0] - u * 0.10, top[1]),
              (top[0] + u * 0.10, top[1] - u * 0.10), 1)
        _line(surf, _CAIRN_RUNE, (top[0] + u * 0.10, top[1] - u * 0.10),
              (top[0] + u * 0.02, top[1] - u * 0.22), 1)


_GROUND: dict[str, Callable[..., None]] = {
    RELIC_SCALE: _r_scale,
    RELIC_AMULET: _r_amulet,
    RELIC_BFG: _r_bfg,
    RELIC_BOOTS: _r_boots,
    RELIC_CAIRN: _r_cairn,
}


def draw_relic(surf: pygame.Surface, kind: str, x: float, y: float, *,
               scale: float = 1.0, flat: Color | None = None,
               fade: float = 1.0, lean: float = 1.0) -> None:
    """One relic of *kind*, sitting on the ground at ``(x, y)``.

    Public because a contact sheet, a HUD tray and a debug overlay all want to
    draw a relic that no sim object exists for. *flat* is the silhouette
    colour: given one, every part is drawn in it and the interior detail is
    skipped, exactly as ``creatures._draw_spear`` and ``draw_grave`` do it.
    """
    fn = _GROUND.get(kind)
    if fn is None:
        return
    u = _UNIT * max(0.2, float(scale))
    fn(surf, float(x), float(y), u, flat, fx.clamp(fade, 0.0, 1.0), lean)


def _relic_fade(relic: Any) -> float:
    """Brightness for a relic: fades in when it lands, out when it is dying.

    The two clocks are not interchangeable and ``sim/items.py`` says which is
    which: ``t`` is seconds lying here and is what RELIC_DECAY_SEC is read
    against, while ``age`` is seconds since the object was created and is never
    reset - the module docstring over there points at *this* use for it, "so
    render/ can fade a fresh drop in off ``age`` without reading the decay
    clock". Reading them the wrong way round (which the first cut did) makes a
    relic that has just dropped look like one that is about to expire.

    Neither fade ever reaches zero. A relic at 0% for two frames reads as a
    dragon that dropped nothing, which is the one thing this drawing must never
    say.
    """
    out = 1.0
    lying = _num(relic, "t", default=0.0)
    if lying > 0.0 and RELIC_DECAY_SEC > _DECAY_FADE:
        left = RELIC_DECAY_SEC - lying
        if left < _DECAY_FADE:
            out = 0.35 + 0.65 * fx.clamp(left / _DECAY_FADE, 0.0, 1.0)
    age = _num(relic, "age", default=-1.0)
    if 0.0 <= age < _DROP_FADE:
        out *= 0.45 + 0.55 * fx.smoothstep(age, 0.0, _DROP_FADE)
    return out


def _relic_place(relic: Any, world: Any) -> tuple[str, float, float, float] | None:
    """``(kind, x, ground_y, lean)`` for a relic, or None if it is not drawable.

    The lean is derived from the persisted id - deliberately not from a random
    stream, for the two reasons render/throwing.py records for planted spears:
    render has no seeded stream of its own, and the value has to be stable
    across frames *and* across a reload.
    """
    kind = _get(relic, "kind", "")
    if not isinstance(kind, str) or kind not in _GROUND:
        return None
    x = _num(relic, "x", default=math.nan)
    y = _num(relic, "y", default=math.nan)
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if x < -_CULL or x > RENDER_W + _CULL or y < -_CULL or y > RENDER_H + _CULL:
        return None
    gy = _ground_y(world, x, y) if world is not None else y
    idp = (_num(relic, "id", default=0.0) * 0.9137) % TAU
    return (kind, x, gy, math.sin(idp))


def draw_relics(surf: pygame.Surface, world: Any, t: float = 0.0) -> None:
    """Every relic lying in *world*, lit by the scene like a spear is.

    Draw at step 7, before the light composite. *t* is accepted and ignored:
    the body of a relic is a pure function of sim state, so a paused world's
    relics hold still. The animated half is in :func:`draw_relic_fx`.

    Fails soft, per relic and overall - this is a frame path.
    """
    try:
        relics = _iter_relics(world)
    except Exception:
        log.debug("could not read the relic list", exc_info=True)
        return
    if not relics:
        return
    for r in relics:
        try:
            place = _relic_place(r, world)
            if place is None:
                continue
            kind, x, gy, lean = place
            # Sample the light where the object is, not at the contact point:
            # a relic is 11 px tall and half of it can be inside a torch pool
            # while its footprint is not. creatures._draw_animal samples the
            # middle of the body for the same reason.
            flat = (_SIL_COLOR if _light_at(world, x, gy - _UNIT * 0.3) < _SIL_CUTOFF
                    else None)
            draw_relic(surf, kind, x, gy, flat=flat, fade=_relic_fade(r), lean=lean)
        except Exception:
            log.debug("relic draw failed", exc_info=True)


# --------------------------------------------------------------------------
# the self-lit pass: halos, beams, the wreck
# --------------------------------------------------------------------------


def _draw_halo(surf: pygame.Surface, kind: str, x: float, y: float,
               t: float, idp: float, fade: float) -> None:
    """The magic. An additive bloom plus one rising mote.

    This is the only reason relics are drawn across two passes. Inside the
    light composite the bloom is multiplied by the scene ambient, which on a
    night storm is ~0.06 - the halo would not be dim, it would be *gone*, on
    exactly the nights it is the only way to find a relic in the dirt.

    The pulse is slow (0.55 Hz) and shallow. A fast pulse reads as an alarm and
    a deep one reads as a fire; a relic should look like it is *waiting*.
    """
    col, strength = _HALO.get(kind, (_AMULET_BEAD, 0.6))
    pulse = 0.78 + 0.22 * math.sin(t * 3.4 + idp)
    amt = strength * pulse * fade
    if amt <= 0.02:
        return
    # Radius 2.6 units, not 1.55. The first cut was sized to the object and it
    # was invisible on the sheet - a halo that ends at the edge of the thing it
    # is around is a rim light, and a rim light on a 13 px object at night is
    # three pixels. It has to be a pool the eye catches from across the map.
    fx.blit_light(surf, x, y - _UNIT * 0.35, _UNIT * 2.6, col, amt * 0.85,
                  falloff=2.1)
    # A hot pip on the object itself, because the *body* of a relic is drawn
    # inside the light composite and at a night ambient of ~0.1 it is a dark
    # smudge in the middle of its own pool - measured on seed 7 at 240 ticks,
    # where the halo was legible from across the map and the thing making it
    # was not. Two or three pixels of the relic's own colour, on top of the
    # composite, is the whole fix: the object is now a lit thing in a pool
    # rather than a hole in one.
    _disc(surf, _dim(col, (0.55 + 0.35 * pulse) * fade),
          (x, y - _UNIT * 0.30), max(1.0, _UNIT * 0.13))
    # One mote drifting up out of it, on a 2.4 s cycle. Two motes read as a
    # fire and none reads as a light source somebody left on; one is the
    # difference between "glowing" and "alive".
    v = ((t * 0.42 + idp * 0.16) % 1.0)
    my = y - _UNIT * (0.45 + v * 1.35)
    mx = x + math.sin(idp + v * 3.1) * _UNIT * 0.22
    mote = _dim(col, (1.0 - v) * (0.55 + 0.45 * pulse) * fade)
    _disc(surf, mote, (mx, my), max(1.0, _UNIT * 0.09 * (1.0 - v * 0.5)))


def draw_relic_fx(surf: pygame.Surface, world: Any, t: float = 0.0) -> None:
    """Everything self-lit: relic halos, BFG discharges, the wrecked saucer.

    Draw *after* the light composite, next to ``creatures.draw_ufo``. One entry
    point for the three because they share a layer and a reason to be on it,
    and because a renderer that has to remember three calls will one day
    remember two - which is the failure ``tools/smoke.py`` records about spears
    being modelled for 230 throws and never drawn.

    Each of the three guards itself, so a junk beam record cannot cost the
    halos and a saucer mid-explosion cannot cost the relics.
    """
    t = float(t)
    try:
        relics = _iter_relics(world)
    except Exception:
        relics = []
    for r in relics:
        try:
            place = _relic_place(r, world)
            if place is None:
                continue
            kind, x, gy, lean = place
            idp = (_num(r, "id", default=0.0) * 0.9137) % TAU
            _draw_halo(surf, kind, x, gy, t, idp, _relic_fade(r))
        except Exception:
            log.debug("relic halo failed", exc_info=True)

    try:
        beams = _iter_beams(world)
    except Exception:
        beams = []
    if beams:
        # Resolved once, not per record: it is a getattr chain and there can be
        # eight records, but more importantly every record in one frame must be
        # aged against the *same* instant or two simultaneous shots fade at
        # different rates.
        now = _world_now(world, t)
        for b, absolute in beams:
            try:
                _draw_beam_record(surf, world, b, now, absolute)
            except Exception:
                log.debug("bfg beam draw failed", exc_info=True)

    try:
        draw_wreck(surf, world, t)
    except Exception:
        log.debug("wreck fx failed", exc_info=True)


# ----------------------------------------------------------------- the BFG --

_BAKE_CACHE: dict[tuple[int, int, int], pygame.Surface] = {}
_BAR_CACHE: dict[tuple[int, int, int, int], pygame.Surface] = {}
_CACHE_MAX = 48


def _bake_bar(lb: int, hb: int, inten: int) -> pygame.Surface | None:
    """The unrotated beam bar, along +x.

    Built with numpy in one vectorised pass and returned *opaque* and black
    outside the corridor, so ``BLEND_RGB_ADD`` adds nothing beside the beam -
    the same construction as ``creatures._beam_surface`` for the abduction
    cone, and for the same reason.
    """
    if inten <= 0:
        return None
    key = (lb, hb, int(inten))
    hit = _BAKE_CACHE.get(key)
    if hit is not None:
        return hit

    amt = inten / 8.0
    w, h = lb, hb * 2 + 1
    u = (np.arange(w, dtype=np.float32) / max(1.0, w - 1.0))[:, None]
    v = (np.abs(np.arange(h, dtype=np.float32) - hb) / float(hb))[None, :]
    # Across the corridor: a soft body with a hard bright core. The core is
    # what makes it read as a beam rather than as a green smear - a pure
    # gaussian at this width looked like fog.
    body = np.clip(1.0 - v * v, 0.0, 1.0) ** 2
    core = np.clip(1.0 - v * 5.5, 0.0, 1.0) ** 2
    # Along it: full at the muzzle, a little hotter again at the far end where
    # it stopped, because that is where the player should look.
    along = 0.86 + 0.14 * np.sin(u * math.pi)
    # ...and feather both ends over 7% of the length. A rectangle rotated off
    # axis ends in a hard diagonal cut, which on the sheet read as the beam
    # being clipped by something invisible. The muzzle end is hidden by its
    # bloom anyway; the far end is the one that matters, because that is the
    # end the player is being told the beam *stopped* at.
    along = along * np.clip(np.minimum(u, 1.0 - u) / 0.07, 0.0, 1.0)
    # 0.72, not 1.0. At full strength the middle 40% of the corridor clipped to
    # pure white over the beam's whole length (measured: 255 green for 320
    # consecutive pixels) and the core stopped being visible *as* a core - the
    # thing read as a solid slab. Backing the bake off leaves the additive
    # blit room to blow out where it overlaps the two blooms instead, which is
    # where the eye should go.
    g_body = (body * along * amt * 0.72).astype(np.float32)
    g_core = (core * along * amt * 0.82).astype(np.float32)
    rgb = (g_body[:, :, None] * np.asarray(_BEAM_GREEN, dtype=np.float32)
           + g_core[:, :, None] * np.asarray(_BEAM_WHITE, dtype=np.float32) * 0.85)
    try:
        surf = pygame.surfarray.make_surface(
            np.ascontiguousarray(np.clip(rgb, 0.0, 255.0), dtype=np.uint8))
    except Exception:                                  # pragma: no cover
        log.debug("beam bar build failed", exc_info=True)
        return None

    if len(_BAKE_CACHE) > _CACHE_MAX:
        _BAKE_CACHE.clear()
    _BAKE_CACHE[key] = surf
    return surf


def _beam_bar(length: int, half: int, inten: int, deg: float
              ) -> pygame.Surface | None:
    """The beam bar, baked and rotated onto the shot's own heading.

    Cached on the rotation as well as the bake, at **one degree**. That number
    is the whole story of this function. The first cut used 32 buckets (11.25
    degrees) so the rotate could be lifted out of the frame path entirely, and
    the contact sheet showed it immediately: half a bucket is 5.6 degrees,
    which over a 420 px beam walks the far end 41 px off the line the shot
    actually took, and the baked bar crossed the exact white core stroke in an
    X. At 1 degree the worst case is 0.5 degrees and 3.7 px of lateral drift on
    a bar 44 px wide with a soft edge - invisible, and the exact stroke and the
    two blooms are all still drawn on the true geometry.

    Measured on a real 1600x1000 seed-7 frame with six relics on the ground,
    one 400 px beam and one misfire all up at once, netting out the halos the
    same pass draws either way: the two shot records cost **~930 us/frame**
    rotating exactly every frame and **~105 us/frame** through this cache. A
    discharge is ~8 frames, so it is the difference between 7 ms and 0.8 ms of
    extra work per shot fired.

    rotozoom and not rotate: rotate samples nearest-neighbour, and on a 45 px
    soft-edged bar that turned both long edges into a visible comb of ~4 px
    teeth. rotozoom interpolates and the fringe stays a fringe.
    """
    lb = max(8, int(round(length / 16.0)) * 16)
    hb = max(3, int(round(half / 2.0)) * 2)
    db = int(round(deg)) % 360
    key = (lb, hb, int(inten), db)
    hit = _BAR_CACHE.get(key)
    if hit is not None:
        return hit
    flat = _bake_bar(lb, hb, int(inten))
    if flat is None:
        return None
    try:
        surf = pygame.transform.rotozoom(flat, float(db), 1.0)
    except Exception:                                  # pragma: no cover
        try:
            surf = pygame.transform.rotate(flat, float(db))
        except Exception:
            return flat
    if len(_BAR_CACHE) > _CACHE_MAX:
        _BAR_CACHE.clear()
    _BAR_CACHE[key] = surf
    return surf


def _beam_envelope(u: float) -> float:
    """Brightness of a discharge *u* of the way through its life.

    Attack in 3 frames, hold, then a long tail. A symmetric envelope read as a
    flashbulb; the tail is what makes it read as something that *fired*.
    """
    if u <= 0.0:
        return 1.0
    if u >= 1.0:
        return 0.0
    return fx.smoothstep(u, 0.0, 0.10) * (1.0 - fx.smoothstep(u, 0.34, 1.0))


def draw_beam(surf: pygame.Surface, x0: float, y0: float, x1: float, y1: float,
              u: float = 0.0, *, corridor: float = BFG_CORRIDOR) -> None:
    """One BFG discharge, from the muzzle ``(x0, y0)`` to where it stopped.

    *u* is 0..1 through :data:`BEAM_LIFE`. Public so a contact sheet can draw a
    beam with no sim object behind it.
    """
    amt = _beam_envelope(u)
    if amt <= 0.01:
        return
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if not math.isfinite(length) or length < 2.0:
        return
    ang = math.atan2(dy, dx)
    # The corridor narrows as it fades. A bar that only dims reads as a
    # dissolve; one that also pinches reads as a discharge collapsing back into
    # the muzzle, which is the read the whole effect is for.
    half = max(2.0, corridor * (1.0 - 0.42 * u))
    # Screen y is down, so a beam heading up-and-right has a NEGATIVE atan2;
    # pygame rotates counter-clockwise on screen for a positive angle.
    # -degrees(ang) is therefore the identity that puts the bar on the same
    # line as the core stroke below, and getting that sign wrong is what
    # crossed the two in an X on the first contact sheet.
    bar = _beam_bar(int(length), int(half), int(round(amt * 8.0)),
                    -math.degrees(ang))
    if bar is not None:
        bw, bh = bar.get_size()
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        surf.blit(bar, (int(cx - bw * 0.5), int(cy - bh * 0.5)),
                  special_flags=pygame.BLEND_RGB_ADD)
    # A hard white centre line on top of the baked bar. The bar is bucketed to
    # 16 px of length and 2 px of width, so on its own the beam visibly snaps
    # between sizes on consecutive frames; the line is exact and hides it.
    _line(surf, _dim(_BEAM_WHITE, amt), (x0, y0), (x1, y1),
          max(1, int(round(2.0 + 2.0 * amt))))
    # Muzzle bloom, and a bigger one where it stopped.
    fx.blit_light(surf, x0, y0, corridor * 1.6, _BEAM_GREEN, 0.85 * amt, falloff=1.5)
    fx.blit_light(surf, x1, y1, corridor * 2.6, _BEAM_WHITE, 0.95 * amt, falloff=1.9)


#: How wide a misfire is drawn, in pixels, and this number is a *spec* and not
#: a look choice. The user's correction: '"randomly explodes taking out the
#: stickmen" .. "stickman", just himself .. so its an or situation'. A misfire
#: kills the man holding the gun and **nobody else**; the heedless corridor
#: shot is the one that can wipe most of a colony and leave a survivor.
#:
#: The old drawing was ``BFG_BLAST_R`` = 70 px, which is 2.7 villagers wide, and
#: it told the viewer that everybody standing near the wielder had just died. A
#: drawing that overstates a kill zone is worse than no drawing at all: it
#: arrives in the same frame as the corpse, so the eye believes it and reads
#: every *unrelated* death for the next few seconds as part of the blast.
#:
#: 0.62 * 26 = 16 px, so the ring lands ~32 px across against a 26 px figure -
#: him and the air he is standing in. A neighbour one body-width away is
#: visibly outside it, which is the whole assertion the drawing has to make.
#:
#: Derived from AGENT_HEIGHT and **not** from any BFG_* constant, deliberately.
#: While this lane was open, ``constants.BFG_BLAST_R`` went 70.0 -> 0.0 (with a
#: separate drawn-radius constant beside it) -> 70.0 again as the combat lane
#: worked out where the spec correction belonged. A drawing keyed to a number
#: mid-argument is a drawing that silently becomes a 2 px dot or a 140 px disc
#: depending on which day it is read; keyed to the wielder it is right in every
#: version of that argument, because the wielder is what the spec is about.
_MISFIRE_R = AGENT_HEIGHT * 0.62

#: How far the *light* of a misfire reaches, as a multiple of :data:`_MISFIRE_R`.
#:
#: Split from the ink on purpose, and this is the line the two readings of the
#: correction meet on. The combat lane's note argues "a gun coming apart should
#: still throw light; it just should not kill the bystanders it lights up" and
#: that is right - a bloom is not a hitbox, nobody has ever read a glow as a
#: casualty list, and a misfire that flashes no harder than the shot that did
#: not go off is a damp read of the loudest failure in the game. What must stay
#: small is every *opaque* mark: the ring, the shards, the core. Those are the
#: ones the eye measures a blast by.
_MISFIRE_GLOW = 2.4


def draw_misfire(surf: pygame.Surface, x: float, y: float, u: float = 0.0, *,
                 radius: float = _MISFIRE_R) -> None:
    """The wyrm-gun coming apart in the wielder's hands, and taking only him.

    Deliberately *not* a beam with a short range: a misfire has no line, so the
    drawing has to be radial. Equally deliberately not a big one - see
    :data:`_MISFIRE_R` for why the radius is one body and not BFG_BLAST_R.

    Three parts, and the shards are the load-bearing one. A ring plus a flash
    is the vocabulary of an area attack no matter how small you draw it (the
    contact sheet's first cut at 16 px still read as "a small bomb"); six pieces
    of *gun* going outward is the read that this was a thing that broke rather
    than a thing that detonated, and it is what makes a viewer look at the man
    instead of at the ground around him.
    """
    amt = 1.0 - fx.smoothstep(u, 0.15, 1.0)
    if amt <= 0.01:
        return
    r = max(2.0, float(radius))
    grow = fx.smoothstep(u, 0.0, 0.55)
    # The ring reaches one body width and *stops there*. The old version grew
    # for 65% of the life and ran off the edge of the drawing, which is most of
    # why it read as an area: an expanding ring the eye never sees arrive
    # anywhere is a shockwave still travelling.
    rr = max(2.0, r * (0.15 + 0.85 * grow))
    try:
        pygame.draw.circle(surf, _dim(_BEAM_GREEN, amt), (int(x), int(y)),
                           int(rr), max(1, int(round(3.0 * amt))))
    except Exception:                                  # pragma: no cover
        pass
    # Six fittings thrown clear. Fixed angles phased off the muzzle x, so this
    # is a pure function of the record: render owns no seeded stream, and a
    # shard set re-rolled per frame is a sparkler rather than one event. Gold,
    # because that is the gun's own trim colour - the pieces have to be
    # recognisable as having come off the object in his hands.
    ph = (float(x) * 0.137) % TAU
    for k in range(6):
        a = ph + k * TAU / 6.0 + 0.19 * math.sin(ph * 3.0 + k)
        r0 = r * (0.22 + 0.48 * grow)
        r1 = r0 + r * 0.30 * (0.40 + grow)
        ca, sa = math.cos(a), math.sin(a) * 0.82      # squashed: he is standing
        _line(surf, _dim(_BFG_TRIM, amt * 0.95),
              (x + ca * r0, y + sa * r0), (x + ca * r1, y + sa * r1), 2)
    # The core collapsing, not expanding. A core that grows is a second ring.
    _disc(surf, _dim(_BEAM_WHITE, amt), (x, y),
          max(1.0, r * 0.42 * (1.0 - 0.68 * grow)))
    # The glow, and it is allowed well past the ring - see _MISFIRE_GLOW. A
    # steep falloff so it stays a flash centred on one man rather than an even
    # wash over his neighbours: at falloff 1.7 the pool is at a third of its
    # peak by the time it reaches the nearest hut.
    fx.blit_light(surf, x, y, r * _MISFIRE_GLOW, _BEAM_GREEN, 1.05 * amt,
                  falloff=1.7)


def _draw_beam_record(surf: pygame.Surface, world: Any, rec: Any,
                      now: float = 0.0, absolute: bool = False) -> None:
    """Draw one shot record from the sim.

    *absolute* is :func:`_iter_beams`'s verdict on what ``rec["t"]`` means and
    *now* the world clock to measure it against; see that function for why the
    two producers disagree and why this cannot be decided from the value.
    """
    x0 = _num(rec, "x0", "x", default=math.nan)
    y0 = _num(rec, "y0", "y", default=math.nan)
    if not (math.isfinite(x0) and math.isfinite(y0)):
        return
    if absolute:
        stamp = _num(rec, "t", "age", default=math.nan)
        if not math.isfinite(stamp):
            # No stamp on a record that should carry one: draw it once at full
            # strength, which is what the age branch has always done for the
            # same omission. Resolving it to `now - 0` instead would silently
            # hide every shot in a world older than a third of a second, which
            # is the exact failure this argument exists to end.
            age = 0.0
        else:
            # max(0) and not abs(): a stamp in the future means the clock moved
            # under us (a load, a debug scrub), and a fresh beam is the honest
            # reading of that - a negative age would drive u below 0 and the
            # envelope's `u <= 0` branch would hold the shot on screen forever.
            age = max(0.0, now - stamp)
    else:
        age = _num(rec, "t", "age", default=0.0)
    u = fx.clamp(age / max(1e-3, BEAM_LIFE), 0.0, 1.0)
    if bool(_get(rec, "misfire", False)):
        # `radius` is NOT read off the record. A misfire is wielder-only now
        # (see _MISFIRE_R), so its drawn size is set by the man and not by a
        # blast radius - and a producer that still writes BFG_BLAST_R = 70 into
        # the record would otherwise put the old lie straight back on screen.
        draw_misfire(surf, x0, y0, u)
        return
    x1 = _num(rec, "x1", "tx", default=math.nan)
    y1 = _num(rec, "y1", "ty", default=math.nan)
    if not (math.isfinite(x1) and math.isfinite(y1)):
        return
    draw_beam(surf, x0, y0, x1, y1, u,
              corridor=_num(rec, "corridor", default=BFG_CORRIDOR))


# ----------------------------------------------------------- the ward wreck --

_WRECK_FLASH: Color = (180, 255, 210)      # lighting.add_flash's own colour
_WRECK_FIRE: Color = (255, 196, 120)
_WRECK_SMOKE: Color = (58, 62, 70)


def draw_wreck(surf: pygame.Surface, world: Any, t: float = 0.0) -> None:
    """The saucer coming apart over an amulet bearer.

    Drives entirely off the saucer's own state - ``phase == "wreck"`` (or a
    ``wrecked`` flag) and ``t``, seconds in that phase - so it is a pure
    function of sim state plus the render clock and needs nothing new stored
    anywhere. The falling hull is *not* drawn here: ``creatures._draw_ufo``
    already draws the saucer at whatever ``y`` says, and "wreck" is not in its
    list of phases to skip, so the fall arrives for free and this module only
    has to set it on fire.
    """
    ufo = _find_ufo(world)
    if ufo is None:
        return
    state = _state_of(ufo)
    if "wreck" not in state and not bool(getattr(ufo, "wrecked", False)):
        return
    x = fx.attr_num(ufo, "x", default=math.nan)
    y = fx.attr_num(ufo, "y", default=math.nan)
    if not (math.isfinite(x) and math.isfinite(y)):
        return
    width = fx.clamp(fx.attr_num(ufo, "width", "w", default=54.0), 24.0, 160.0)
    ph = fx.attr_num(ufo, "t", default=0.0)
    fall = fx.clamp(ph / max(0.2, UFO_WRECK_FALL_SEC), 0.0, 1.0)
    idp = (fx.attr_num(ufo, "id", default=0.0) * 1.7) % TAU
    ground = _ground_y(world, x, y + 100.0)

    # 1. the detonation. Front-loaded and short: this is the frame the player
    # has to catch out of the corner of their eye, so it is the brightest thing
    # the game ever draws, for a fifth of a second.
    burst = 1.0 - fx.smoothstep(ph, 0.0, 0.75)
    if burst > 0.01:
        fx.blit_light(surf, x, y, width * (1.1 + 2.6 * (1.0 - burst)),
                      _WRECK_FLASH, 1.8 * burst, falloff=1.3)
        _disc(surf, _dim((255, 255, 255), burst), (x, y),
              max(1.0, width * 0.34 * burst))
        # Shock ring. An outline, not a disc: a filled expanding disc reads as
        # a light turning on, a ring reads as something leaving.
        rr = width * (0.5 + 2.1 * (1.0 - burst))
        try:
            pygame.draw.circle(surf, _dim(_WRECK_FLASH, burst * 0.9),
                               (int(x), int(y)), int(rr),
                               max(1, int(round(3.0 * burst))))
        except Exception:                              # pragma: no cover
            pass

    # 2. the hull burning on the way down: a fire core under the dome and three
    # arcs crawling over it, phase-locked to the render clock so they crackle.
    flick = 0.72 + 0.28 * math.sin(t * 27.0 + idp)
    # A burning glow over the whole hull, not just a hot dot on it. At
    # width * 0.13 the fire was a 7 px pip that the saucer's own rim lights
    # out-shouted, and the late-fall cell read as an intact saucer with a
    # spark on it rather than as a wreck.
    fx.blit_light(surf, x, y + width * 0.04, width * 0.95, _WRECK_FIRE,
                  0.45 + 0.25 * flick, falloff=1.8)
    _disc(surf, _dim(_WRECK_FIRE, 0.9 * flick), (x, y + width * 0.06),
          max(2.0, width * 0.20 * flick))
    _disc(surf, _lit(_WRECK_FIRE, 0.55 * flick), (x, y + width * 0.05),
          max(1.0, width * 0.09 * flick))
    for k in range(3):
        a = idp + t * 5.1 + k * TAU / 3.0
        r0, r1 = width * 0.20, width * 0.46
        p0 = (x + math.cos(a) * r0, y + math.sin(a) * r0 * 0.42)
        p1 = (x + math.cos(a + 0.9) * r1, y + math.sin(a + 0.9) * r1 * 0.42)
        _line(surf, _dim(_WRECK_FLASH, 0.55 + 0.45 * flick), p0, p1, 2)

    # 3. the trail. Sparks and smoke lag *above* the hull because it is falling
    # - the trail is the path it has already taken, and the sim never stores
    # that path, so it is reconstructed from how far through the fall we are.
    #
    # Sparks carry it and smoke only punctuates: the first cut was seven fat
    # smoke discs and it drew a column of black balls stacked over the saucer
    # like a caterpillar, because a dark grey disc *dimmed as it disperses* is
    # backwards - dispersing smoke goes toward the sky, not toward black.
    drop = width * 3.0 * fall
    for k in range(9):
        f = (k + 1) / 10.0
        sy = y - drop * f
        sx = x + math.sin(idp * 1.7 + k * 2.31) * width * 0.34 * f
        if k % 3 == 0:
            _disc(surf, _lit(_WRECK_SMOKE, 0.10 + 0.22 * f),
                  (sx, sy), max(1.0, width * 0.055 * (0.6 + f)))
        else:
            _disc(surf, _dim(_WRECK_FIRE if k % 2 else _WRECK_FLASH,
                             (1.0 - f * 0.85) * flick),
                  (sx, sy), max(1.0, width * 0.035 * (1.2 - f)))

    # 4. the ground taking it, over the last quarter of the fall.
    land = fx.smoothstep(fall, 0.72, 1.0)
    if land > 0.01:
        fx.blit_light(surf, x, ground - 2.0, width * 1.25, _WRECK_FLASH,
                      0.8 * land, falloff=2.2)


# --------------------------------------------------------------------------
# worn gear
#
# Called from creatures.draw_gear with the skeleton it has already solved. The
# contract with that function is exactly the one _draw_armour and _draw_spear
# keep: read the skeleton, honour `flat` (silhouette) and `fade` (corpse), draw
# nothing else, and never raise.
# --------------------------------------------------------------------------


def _sk_scale(sk: Any) -> float:
    """Drawn size relative to a nominal 26 px villager.

    Floored at 0.55 and not at 0.4, which is the exact clamp
    ``creatures._draw_armour`` already carries for the leather jerkin. That
    number is not arbitrary over there and it is not here either: the limbs of a
    13 px figure are still drawn 1 px wide and the torso 2, because a line
    cannot be thinner than a pixel, so gear that keeps scaling linearly past
    ~0.55 shrinks *underneath* the body it is supposed to be worn over and
    disappears into it. Measured before this floor, the dragonscale coat changed
    3 pixels of a 0.5x silhouette; the jerkin it is meant to out-read changed 2.
    """
    return max(0.55, fx.attr_num(sk, "height", default=AGENT_HEIGHT) / AGENT_HEIGHT)


def _w_scale(surf: pygame.Surface, sk: Any, flat: Color | None, fade: float,
             t: float) -> None:
    """The dragonscale coat: the jerkin's quad, plated.

    Same tapered shoulder-to-hip quad ``creatures._draw_armour`` uses, because
    the silhouette of *something worn* is the part that survives at 26 px and
    that shape is already proven. What is different is the fill: two tones
    split down the middle plus two chevron courses, so the coat reads as plate
    rather than as hide. The plate is a full step lighter than leather and
    carries an emerald rim - across a dark field the wearer of this thing is
    supposed to be identifiable at a glance, because they are the one who walks
    out of the wolf fight.
    """
    hip = getattr(sk, "hip", None)
    neck = getattr(sk, "neck", None)
    if hip is None or neck is None:
        return
    dx, dy = neck[0] - hip[0], neck[1] - hip[1]
    d = math.hypot(dx, dy)
    if d < 1.0:
        return
    px, py = -dy / d, dx / d
    s = _sk_scale(sk)

    def at(u: float, half: float, side: float) -> tuple[float, float]:
        return (hip[0] + dx * u + px * half * side, hip[1] + dy * u + py * half * side)

    # Wider than the jerkin's 1.5 / 2.2, and the gap is the point: plate has
    # bulk and leather does not, and the two are drawn on the same quad in the
    # same place, so *width* is the only channel left that works in silhouette.
    w_hip, w_sh = 2.05 * s, 3.0 * s
    body = flat or _dim(_SCALE_PLATE, fade)
    deep = flat or _dim(_SCALE_DEEP, fade)
    rim = flat or _dim(_SCALE_RIM, fade)

    _poly(surf, body, [at(0.04, w_hip, +1.0), at(0.88, w_sh, +1.0),
                       at(0.88, w_sh, -1.0), at(0.04, w_hip, -1.0)])
    # Pauldrons. Two discs proud of the shoulder line, and they are what makes
    # this read as *armour* rather than as a slightly fatter jerkin - a
    # silhouette with lumps on its shoulders is armoured in every game ever
    # drawn, and it is the one cue that survives the flat path intact because
    # it changes the outline instead of the fill. Drawn before the interior
    # detail so the courses below can cross them.
    pd = max(1.3, 1.55 * s)
    _disc(surf, body, at(0.90, w_sh * 0.92, +1.0), pd)
    _disc(surf, body, at(0.90, w_sh * 0.92, -1.0), pd)
    if flat is not None:
        return
    _disc(surf, rim, at(0.93, w_sh * 0.92, -1.0), max(1.0, pd * 0.5))
    # Shaded far half. A single flat quad at this size is a slab; the split is
    # the cheapest thing that makes it a body wearing something.
    _poly(surf, deep, [at(0.04, w_hip * 0.1, +1.0), at(0.88, w_sh * 0.1, +1.0),
                       at(0.88, w_sh, +1.0), at(0.04, w_hip, +1.0)])
    _line(surf, rim, at(0.86, w_sh, +1.0), at(0.86, w_sh, -1.0), 1)
    if s >= 0.75:
        # Two courses of scale. Below 0.75 they merge into the rim and cost a
        # pixel of contrast for nothing, so they are dropped rather than drawn
        # into mud - the 0.5x cell of the contact sheet is what set this gate.
        for u in (0.34, 0.60):
            _line(surf, rim, at(u, w_sh * 0.86, -1.0), at(u - 0.06, w_sh * 0.86, +1.0), 1)
    # The one hot pixel: an emerald catch-light at the shoulder.
    _disc(surf, _lit(_SCALE_HOT, 0.15), at(0.80, w_sh * 0.55, -1.0), max(1.0, 0.9 * s))


def _w_amulet(surf: pygame.Surface, sk: Any, flat: Color | None, fade: float,
              t: float) -> None:
    """A bead on a V of cord, hung at the throat and worn *proud* of the chest.

    At the neck rather than in a hand, because a hand-held amulet is occluded
    by every carried icon and every tool, and this is the one relic whose whole
    job happens while the wearer is standing still in a beam.

    Two things here were measured wrong the first time and both are about the
    same mistake - drawing it *on* the spine instead of *over* the chest:

    * The bead sat on the torso line, which is 2-3 px wide, so most of it was
      inside the body. Worn vs not-worn changed **7 pixels lit and 4 flat**,
      against 77 and 60 for the spear the player is supposed to read this
      beside. It was a smudge on the chest, not an object.
    * There was one hair of cord, which at 1 px and mostly behind the head
      contributed nothing at all.

    So: pushed out to the facing side until it breaks the outline, sized up,
    and hung on a V from both shoulders. The V is the half that survives 0.5x,
    where the bead is one disc and could be anything; two lines converging on
    it are unambiguously a necklace.

    The whole thing is built in the spine's own frame (down = neck->hip, across
    = its normal) rather than in screen axes, so it hangs correctly through a
    bow, a crouch and a corpse lying on its side instead of pointing at the
    bottom of the monitor.
    """
    neck = getattr(sk, "neck", None)
    hip = getattr(sk, "hip", None)
    if neck is None:
        return
    s = _sk_scale(sk)
    facing = float(getattr(sk, "facing", 1) or 1)
    if hip is not None:
        dx, dy = hip[0] - neck[0], hip[1] - neck[1]
        d = math.hypot(dx, dy)
        ux, uy = (dx / d, dy / d) if d >= 1.0 else (0.0, 1.0)
    else:
        ux, uy = 0.0, 1.0
    px, py = -uy, ux
    # Floored in *pixels* and not only in units, and this is the same argument
    # _sk_scale makes one level up: a shrinking figure's torso stops shrinking
    # at 1 px per limb line, so an offset that keeps scaling walks the bead
    # back inside a body that is not getting any thinner. At 1.0x every floor
    # here is inert; at 0.5x they are the difference between a bead over the
    # chest and a bead in it.
    #
    # `drop` is floored HARDER than `spread` on purpose. With the cord spread
    # scaled down and the drop scaled down equally, the V flattens into a
    # horizontal bar and the 0.5x sheet cell read as a plus sign across the
    # chest rather than as something hung round a neck. Holding the drop at
    # full size while the spread shrinks keeps the two arms steep, which is the
    # part of the shape that says "necklace".
    spread = max(2.1, 2.2 * s)
    drop = max(2.9, 2.9 * s)
    off = max(1.9, 1.95 * s) * (1.0 if facing >= 0.0 else -1.0)
    cx = neck[0] + ux * drop + px * off
    cy = neck[1] + uy * drop + py * off
    cord = flat or _dim(_AMULET_CORD, fade)
    # The V is struck wider than the torso, so both arms of it cross open air
    # and land in the silhouette. Struck from the spine they would run *inside*
    # the body line and cost two draws for nothing, which is exactly what the
    # single hair of cord in the first cut did.
    for k in (-1.0, 1.0):
        _line(surf, cord, (neck[0] + px * spread * k, neck[1] + py * spread * k),
              (cx, cy), max(1, int(round(1.5 * s))))
    _disc(surf, flat or _dim(_AMULET_BEAD, fade), (cx, cy), max(1.7, 2.25 * s))
    if flat is None:
        # A slow twinkle. The amulet is a ward against something that has not
        # happened yet, so it is the one relic that should look *awake*.
        tw = 0.55 + 0.45 * math.sin(t * 2.1 + cx * 0.11)
        _disc(surf, _dim(_AMULET_HOT, tw * fade),
              (cx - ux * 0.55 * s, cy - uy * 0.55 * s), max(1.0, 0.95 * s))


#: The gun's outline, as ``(f, off)`` in its own frame: *f* runs 0 at the back
#: of the receiver to 1 at the muzzle, *off* is across the axis in multiples of
#: :data:`_BFG_HW` with **positive = the grip side**. Walked once as a single
#: filled polygon, top edge first and then back along the bottom.
#:
#: This table *is* the bulk. The read the drawing has to win is "machine" in a
#: world where every other tool is one piece of one material, and at 26 px there
#: is no room to win it with detail - the only channel that survives both the
#: flat silhouette pass and the 0.5x shrink is the **outline changing width**.
#: So over a 22 px run it steps four times: a 9.7 px receiver, a hard drop onto
#: a 5.8 px waist, a hard step back up, and an 11.5 px muzzle housing. Fill
#: either step back in and the whole thing is a log again, which is what the cut
#: before this one was.
#:
#: Two blocks and a waist, and deliberately **not** a cone. A barrel that opens
#: out smoothly into a bell is a trumpet, which is exactly what the third cut
#: looked like on the sheet - a villager carrying a horn, with the gold along
#: the flare the loudest thing in the frame. A gun this size is a *casting*:
#: parallel sides, hard steps, and mass at both ends rather than a taper to a
#: point.
_BFG_OUT: tuple[tuple[float, float], ...] = (
    (0.00, -1.40), (0.44, -1.48), (0.50, -0.92),       # receiver, then the step
    (0.72, -0.95), (0.78, -1.90), (1.00, -1.95),       # waist, then the housing
    (1.00, +1.95), (0.78, +1.90), (0.72, +1.02),       # front face, coming back
    (0.52, +1.02), (0.46, +1.80), (0.00, +1.70),       # waist and the receiver
)

#: The bottom edge of :data:`_BFG_OUT` alone, front-to-back, so the dark
#: underside can be laid along the profile instead of straight across it.
_BFG_UNDER: tuple[tuple[float, float], ...] = (
    (0.00, 1.70), (0.46, 1.80), (0.52, 1.02),
    (0.72, 1.02), (0.78, 1.90), (1.00, 1.95),
)

#: The unit every offset above is quoted in, in pixels. 2.95 puts the receiver
#: at 9.7 px through, the muzzle housing at 11.5 px and the waist between them
#: at 5.8 px - against a 26 px body, a 6 px head, a 2 px spear and a jerkin
#: 4-6 px wide. That is the whole answer to "bulky": the gun is over a third of
#: a body height thick and the single widest object anybody in this colony
#: carries.
#:
#: Ranged against the pose set at 2.6 and 2.95 and the bigger one won on both
#: counts that matter: it is 12% more ink at 1.0x, and at **0.5x** - where the
#: shape is 5-8 px and every interior line has already gone - it takes the gun
#: from 2.7x the spear's footprint to 3.0x. Past this the head starts losing.
#: Floored at 1.9 for the reason ``_sk_scale`` floors at 0.55: below that a
#: shape with four steps in it stops being a shape and becomes a smear with a
#: bright pixel in it.
_BFG_HW = 2.95


_BfgFrame = tuple[
    tuple[float, float],        # breech, the origin of the gun's own frame
    float, float,               # ux, uy - along the barrel toward the muzzle
    float, float,               # px, py - across it, positive = the grip side
    float, float,               # ln (barrel run), hw (half-thickness unit)
    float,                      # the figure's gear scale
    tuple[float, float],        # the front fist
    tuple[float, float] | None,  # the rear fist, when the pose gave us two
]


def _bfg_frame(sk: Any) -> _BfgFrame | None:
    """Where the gun is, in its own coordinates. ``None`` if it cannot be placed.

    Split out from the drawing because :func:`hides_head` has to answer a
    question about the *same* geometry - does this thing land on the wearer's
    face - and a second copy of the solve is a second copy that drifts.
    """
    hands = list(getattr(sk, "hands", None) or ())
    if not hands:
        return None
    s = _sk_scale(sk)
    facing = float(getattr(sk, "facing", 1) or 1)
    front = hands[0]
    rear = hands[1] if len(hands) > 1 else None

    # The axis, in order of preference. Two hands far enough apart is the good
    # case and it is what makes the gun two-handed; the forearm is the fallback,
    # and it is the *spear's* fallback for the same reason - it is the one
    # vector that is always defined and always points where the arm is going.
    #
    # Snapping straight to horizontal here (which an earlier cut did) is what
    # laid a 10 px block across the wielder's face in `chop` and `climb`: those
    # poses put both fists together beside the head, so the two-hand test fails,
    # and a horizontal gun centred on a hand at head height is a gun where the
    # head used to be.
    ux = uy = 0.0
    if rear is not None:
        dx, dy = front[0] - rear[0], front[1] - rear[1]
        d = math.hypot(dx, dy)
        if d >= 1.2:
            ux, uy = dx / d, dy / d
    if ux == 0.0 and uy == 0.0:
        elbows = list(getattr(sk, "elbows", None) or ())
        ex, ey = elbows[0] if elbows else (front[0] - facing, front[1])
        dx, dy = front[0] - ex, front[1] - ey
        d = math.hypot(dx, dy)
        if d < 0.35:                  # arm folded flat: last-resort horizontal
            dx, dy, d = facing, 0.12, 1.0
        ux, uy = dx / d, dy / d
        rear = None                   # one fist on it, so only draw one grip
    # Muzzle forward, elevation kept. Forcing the whole vector horizontal threw
    # away the elevation as well as the direction, which is the other half of
    # why a raised weapon ended up lying across a face.
    if ux * facing < 0.0:
        ux = -ux
    # Grip side. `sgn` is what keeps the underside *under* the gun: (-uy, ux) is
    # the left normal, which for a gun pointing left is the side above it, and
    # an unflipped normal hangs the grips off the top of the barrel for every
    # left-facing villager in the colony.
    sgn = 1.0 if facing >= 0.0 else -1.0
    px, py = -uy * sgn, ux * sgn

    h = fx.attr_num(sk, "height", default=AGENT_HEIGHT)
    # Shorter than the spear's 0.98 and than this drawing's own previous 0.92.
    # Length is the one axis where a gun cannot beat a spear - a longer barrel
    # just reads as a worse spear - and every pixel taken off the run makes the
    # same mass read squatter, which is the whole Doom-BFG silhouette.
    ln = h * 0.84
    hw = max(1.9, _BFG_HW * s)
    # The rear fist sits on the grip, so the receiver hangs back off it. With
    # only one fist resolved that same fist *is* the grip, so the anchor is the
    # front hand and the overhang is unchanged.
    #
    # `sag` is a constant offset in **screen** space and not in the gun's frame,
    # and it is the cheapest thing in this drawing that says *weight*: a weapon
    # that hangs a little below the fists holding it is heavy, and one balanced
    # exactly on them is light. It also buys back most of the head in the poses
    # that raise the hands beside it - chop went from 91% of the head disc
    # covered to 55%, climb 79% to 56% - though not in the ones that raise them
    # *above* it, which is what :func:`hides_head` exists for.
    #
    # Screen-down and not grip-side, deliberately: the grip normal rotates with
    # the barrel, so on an overhead swing it points *forward*, and the offset
    # would stop hanging the gun at exactly the pose that needs it most.
    anchor = rear if rear is not None else front
    breech = (anchor[0] - ux * ln * 0.18,
              anchor[1] - uy * ln * 0.18 + hw * 1.15)
    return (breech, ux, uy, px, py, ln, hw, s, front, rear)


def _w_bfg(surf: pygame.Surface, sk: Any, flat: Color | None, fade: float,
           t: float) -> None:
    """A squat, wide-mouthed cannon carried in both fists.

    Placed by :func:`_bfg_frame`, which aims it along the two fists the poser
    solved. Taking the angle from the hands is what makes the gun obey every
    pose for free, the same trick ``creatures._draw_spear`` uses on the forearm -
    and it matters more here, because a barrel that ignores the arms reads as a
    plank glued to the chest.

    **Sized for mass, not for reach**, and shaped for a *profile* rather than a
    thickness. Three cuts got there, and the two dead ends are worth keeping
    because "make it bigger" is the obvious fix and it is not the one that works:

    * cut 1 - 0.62 of the figure's height and 4 px thick. 84 changed pixels
      against the plain spear's 77: the most powerful object in the game drawn
      the same size as a sharpened stick.
    * cut 2 - 0.92 of height and 6 px thick, with a drum disc over the breech
      and a bell that flared from 1.05 to 1.35 half-widths. Three times the
      spear's ink and it still did not read: on the sheet it was **a green
      plank** and in silhouette a plain dark brick. Both shape cues had been
      swallowed - the drum sat inside an outline as wide as itself, and a 29%
      flare over the last quarter of a 22 px run is two pixels, which the
      antialias eats.

    What reads is :data:`_BFG_OUT`: an outline whose width *steps* rather than
    tapers, with a muzzle housing twice the waist it is on the end of. Plus the
    three things that say somebody is *holding* something heavy:

    * **handles at the real hand positions**, struck from the fists the poser
      solved down through the hull and a little way out the far side. The gun is
      two-handed for free, without the poser knowing guns exist - the same deal
      the spear strikes with the forearm - and the stubs break the underside
      outline, which on a horizontal object has nothing else happening on it.
    * **the sag** in :func:`_bfg_frame`: the hull hangs below the fists rather
      than balancing on them.
    * **a housing over the receiver**, so the top edge is not one straight run
      either.

    The grip side is taken off ``facing`` rather than off screen-down, so a gun
    swung overhead in a chop keeps its handles toward the wielder instead of
    flipping them through the barrel halfway round the swing.
    """
    frame = _bfg_frame(sk)
    if frame is None:
        return
    breech, ux, uy, px, py, ln, hw, s, front, rear = frame

    def at(f: float, off: float = 0.0) -> tuple[float, float]:
        return (breech[0] + ux * ln * f + px * off * hw,
                breech[1] + uy * ln * f + py * off * hw)

    body = flat or _dim(_BFG_BODY, fade)
    dark = flat or _dim(_BFG_DARK, fade)

    # 1. the hull.
    _poly(surf, body, [at(f, off) for f, off in _BFG_OUT])
    # 2. the housing over the receiver - a slabbed box, not a disc. A disc on a
    #    flat-sided machine reads as a wheel; the slant on its front face is
    #    what makes it a part bolted on rather than a bump in the casting.
    _poly(surf, body, [at(0.08, -1.15), at(0.08, -2.05),
                       at(0.26, -2.05), at(0.32, -1.15)])
    # 3. the handles, one per fist. Struck from the hand the poser actually
    #    solved, down through the hull and a little way out the far side, so
    #    each one reads as a fist closed round something *and* breaks the
    #    underside outline - which is the edge a horizontal object has nothing
    #    else happening on. Drawn at the hands rather than at fixed fractions of
    #    the barrel, so a pose that spreads the arms stretches the gun's contact
    #    with the body instead of detaching from it; clamped along the run so a
    #    hand thrown right out front does not put a handle on the muzzle.
    grip_w = max(2, int(round(2.6 * s)))
    for hand in ((front, rear) if rear is not None else (front,)):
        hx, hy = hand
        f = ((hx - breech[0]) * ux + (hy - breech[1]) * uy) / max(1.0, ln)
        _line(surf, body, hand, at(min(max(f, 0.04), 0.70), 1.35), grip_w)

    if flat is not None:
        return

    # Underside in the dark tone, as a band that **follows the hull's own bottom
    # edge** rather than as a straight quad across the whole run. That is not
    # tidiness: a straight dark quad from breech to muzzle is drawn at receiver
    # width, so it paints green back into the notch the outline just cut, and
    # the first render of this cut was a machine in silhouette and a solid slab
    # with the lights on. Whatever cuts the profile has to cut every layer.
    _poly(surf, dark, [at(f, off * 0.34) for f, off in _BFG_UNDER]
                      + [at(f, off) for f, off in reversed(_BFG_UNDER)])
    trim = _dim(_BFG_TRIM, fade)
    # **The aperture.** A dark port sunk into the side of the muzzle housing
    # with the acid core burning inside it, and it is the single strongest cue
    # in the drawing: a big glowing green eye on the front block is what a BFG
    # is, and unlike every line here it is made of area and colour, so it is the
    # one detail that survives intact at 0.5x where the 1 px trim does not.
    #
    # On the *side* of the housing rather than across the end of it, which is
    # the correction two earlier cuts needed from opposite directions: a ring
    # closed over the muzzle read as a cap on the end of a log, and the same
    # ring left open at the centre read as an arrowhead pointing forward.
    _poly(surf, dark, [at(0.77, -1.44), at(0.97, -1.50),
                       at(0.97, 1.50), at(0.77, 1.52)])
    # Front lip and the collar at the receiver step - gold at the two places the
    # outline changes, and nowhere else. The trim's job is to *point at* the
    # profile, not to decorate the middle of it, and every extra coloured pixel
    # is noise once the wallpaper is being seen at half size.
    _line(surf, trim, at(0.995, 1.95), at(0.995, -1.95),
          max(1, int(round(1.3 * s))))
    _line(surf, trim, at(0.47, 1.80), at(0.47, -1.48),
          max(1, int(round(1.3 * s))))
    # Lit edge along the top of the receiver and the housing, so the block has a
    # corner instead of dissolving into the barrel behind it.
    hi = _lit(_BFG_BODY, 0.30)
    _line(surf, hi, at(0.08, -2.05), at(0.26, -2.05), 1)
    _line(surf, hi, at(0.02, -1.40), at(0.42, -1.47), 1)
    # The acid green, in the aperture and again at the breech. It is the only
    # green of its kind on any body in this game and it is the same green the
    # beam will be a second later, so a wielder about to fire is already lit by
    # the thing that is going to do it.
    glow = 0.62 + 0.38 * math.sin(t * 3.7 + breech[0] * 0.07)
    # Framed by the housing on all four sides, not flooding it: an aperture that
    # reaches the block's own outline stops being a port sunk into a machine and
    # becomes a green lamp with a handle, which is what the first pass at this
    # looked like.
    core = _dim(_BFG_CORE, (0.48 + 0.44 * glow) * fade)
    _poly(surf, core, [at(0.81, -1.06), at(0.94, -1.10),
                       at(0.94, 1.12), at(0.81, 1.14)])
    _disc(surf, _lit(core, 0.30), at(0.875, 0.0), max(1.0, 0.95 * s))
    _disc(surf, _dim(_BFG_CORE, (0.45 + 0.40 * glow) * fade),
          at(0.16, -0.25), max(1.2, 1.35 * s))


def _w_boots(surf: pygame.Surface, sk: Any, flat: Color | None, fade: float,
             t: float) -> None:
    """A heavier ankle on each foot, and an iron sole under it.

    The feet are the hardest place on this figure to add anything - they are
    two line ends 1 px wide, and the ground is right there. A cuff *above* the
    foot joint plus a short sole *through* it is what came out of the contact
    sheet: drawn as a blob on the foot the boots read as mud, and drawn as an
    outline they vanished at 0.5x.
    """
    feet = list(getattr(sk, "feet", None) or ())
    if not feet:
        return
    knees = list(getattr(sk, "knees", None) or ())
    s = _sk_scale(sk)
    iron = flat or _dim(_IRON, fade)
    dark = flat or _dim(_IRON_DARK, fade)
    facing = float(getattr(sk, "facing", 1) or 1)
    for i, ft in enumerate(feet[:2]):
        col = iron if i == 0 else _dim(iron, 0.72)     # far boot dimmed, as limbs are
        kn = knees[i] if i < len(knees) else (ft[0], ft[1] - 4.0 * s)
        # Cuff along the shin, so the boot leans with the leg instead of
        # sitting on the ground at a fixed angle while the leg swings.
        dx, dy = ft[0] - kn[0], ft[1] - kn[1]
        d = math.hypot(dx, dy) or 1.0
        ux, uy = dx / d, dy / d
        cuff = (ft[0] - ux * 2.6 * s, ft[1] - uy * 2.6 * s)
        _line(surf, col, cuff, (ft[0] - ux * 0.4 * s, ft[1] - uy * 0.4 * s),
              max(2, int(round(2.4 * s))))
        _line(surf, dark if flat is None else col,
              (ft[0] - facing * 0.8 * s, ft[1] + 0.5 * s),
              (ft[0] + facing * 2.2 * s, ft[1] + 0.5 * s), max(1, int(round(1.6 * s))))
        if flat is None and i == 0:
            # One cold pixel at the toe cap, at 0.6 strength. Full-strength
            # _IRON_HOT on both feet lit up as a puddle under the figure and
            # read as standing in water, which is a thing this world actually
            # draws - the spark has to say "iron", not "shining".
            _disc(surf, _dim(_IRON_HOT, 0.6 * fade),
                  (ft[0] + facing * 1.9 * s, ft[1] + 0.2 * s), max(1.0, 0.55 * s))


def _w_cairn(surf: pygame.Surface, sk: Any, flat: Color | None, fade: float,
             t: float) -> None:
    """The cairnstone, slung at the hip behind the wearer.

    Not in the fist: the fist already holds the carried-resource icon, the
    candle and the torch, and the one action this relic exists for
    (RaiseTheDead) is a kneel with both hands on a headstone. At the hip it is
    never occluded and never fights the pose.

    Two corrections from the sheet, and both of them are "it was inside the
    body". At an offset of 1.6 px with a 1.5 px radius the whole stone sat
    within a 3 px torso, and worn vs not-worn changed **10 pixels lit and 2
    flat** - the single weakest cell of the five, and 2 px of silhouette is not
    a relic, it is dither. It is pushed clear of the hip and drawn bigger.

    A **tapered stone** and not a disc, which is the second correction:
    ``stickfigure._draw_carried`` paints a ~4 px disc for a load of berries or
    stone, so a disc at the belt is the game's existing word for "inventory",
    and this relic must not read as somebody's shopping. Four corners with a
    narrow top and a heavy base is a menhir at any size, and it is the shape
    the graves in this world already use.
    """
    hip = getattr(sk, "hip", None)
    neck = getattr(sk, "neck", None)
    if hip is None:
        return
    s = _sk_scale(sk)
    facing = 1.0 if float(getattr(sk, "facing", 1) or 1) >= 0.0 else -1.0
    # Positioned in the spine's frame so it stays at the hip through a bow or a
    # crouch, but *shaped* in screen axes: it hangs off a belt, so gravity and
    # not the wearer decides which way is up.
    if neck is not None:
        dx, dy = hip[0] - neck[0], hip[1] - neck[1]
        d = math.hypot(dx, dy)
        ux, uy = (dx / d, dy / d) if d >= 1.0 else (0.0, 1.0)
    else:
        ux, uy = 0.0, 1.0
    px, py = -uy, ux
    off = -3.1 * s * facing               # behind them, away from the facing
    cx = hip[0] + px * off + ux * 0.5 * s
    cy = hip[1] + py * off + uy * 0.5 * s
    r = max(1.9, 2.35 * s)
    stone = flat or _dim(_CAIRN_STONE, fade)
    _poly(surf, stone, [(cx - r * 0.86, cy + r * 0.92),
                        (cx - r * 0.60, cy - r * 0.86),
                        (cx + r * 0.58, cy - r * 1.00),
                        (cx + r * 0.88, cy + r * 0.86)])
    if flat is not None:
        return
    _poly(surf, _dim(_CAIRN_SHADE, fade),
          [(cx + r * 0.20, cy + r * 0.90), (cx + r * 0.24, cy - r * 0.92),
           (cx + r * 0.58, cy - r * 1.00), (cx + r * 0.88, cy + r * 0.86)])
    # The rune, breathing. Violet on bone is the strongest contrast pair in the
    # whole relic set and it is spent here on purpose: the cairnstone is the
    # only relic whose effect the player has to *anticipate* to watch happen.
    pulse = 0.5 + 0.5 * math.sin(t * 1.9 + cx * 0.09)
    _line(surf, _dim(_CAIRN_RUNE, (0.45 + 0.55 * pulse) * fade),
          (cx - r * 0.30, cy - r * 0.30), (cx + r * 0.18, cy - r * 0.34),
          max(1, int(round(1.3 * s))))
    _line(surf, _dim(_CAIRN_RUNE, (0.45 + 0.55 * pulse) * fade),
          (cx + r * 0.18, cy - r * 0.34), (cx - r * 0.06, cy + r * 0.34),
          max(1, int(round(1.3 * s))))


_WORN: dict[str, Callable[..., None]] = {
    RELIC_SCALE: _w_scale,
    RELIC_AMULET: _w_amulet,
    RELIC_BFG: _w_bfg,
    RELIC_BOOTS: _w_boots,
    RELIC_CAIRN: _w_cairn,
}


def draw_worn(surf: pygame.Surface, agent: Any, sk: Any = None, t: float = 0.0,
              flat: Color | None = None, fade: float = 1.0,
              kind: str | None = None) -> None:
    """Draw whatever relic *agent* is wearing, onto an existing skeleton.

    Meant to be called from ``creatures.draw_gear`` where the skeleton and the
    colours are already in hand::

        from .items import covers_torso, draw_worn, relic_of
        relic = relic_of(agent)
        if has_armour and not covers_torso(relic):
            _draw_armour(surf, sk, flat, fade)
        if has_spear:
            _draw_spear(surf, sk, flat, fade)
        if relic:
            draw_worn(surf, agent, sk, t, flat, fade, kind=relic)

    *flat* is the silhouette colour and *fade* the corpse fade, both already
    resolved by the caller - identical to ``_draw_armour``'s contract, so the
    relic dims with the limbs and never applies a second light term.

    Given no skeleton, one is solved here, so a caller with only an agent (a
    contact sheet, a debug tray) still works. Fails soft: this is a frame path
    and a relic is not worth a dropped frame.
    """
    try:
        relic = kind if kind is not None else relic_of(agent)
        fn = _WORN.get(relic or "")
        if fn is None:
            return
        if sk is None:
            from .stickfigure import build_skeleton    # local: avoids a cycle
            sk = build_skeleton(agent, float(t))
        if sk is None:
            return
        fn(surf, sk, flat, fx.clamp(fade, 0.0, 1.0), float(t))
    except Exception:
        # No getattr on `agent` in here: creatures.draw_gear records why - a sim
        # object whose attribute access raises would get its exception
        # re-raised out of the except block and take down the frame.
        log.debug("draw_worn failed", exc_info=True)


# --------------------------------------------------------------------------
# caches
# --------------------------------------------------------------------------


def clear_caches() -> None:
    """Drop the baked beam bars. Mirrors ``creatures.clear_caches``."""
    _BAKE_CACHE.clear()
    _BAR_CACHE.clear()


def cache_stats() -> dict[str, int]:
    return {"beam_bakes": len(_BAKE_CACHE), "beam_bars": len(_BAR_CACHE)}


# ============================================================== smoke test ==
if __name__ == "__main__":                             # pragma: no cover
    import os
    import random
    import time as _time
    import types

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode((320, 200))

    from ..sim.entities import Population
    from .stickfigure import Skeleton, build_skeleton, draw_stickman

    BG = (26, 28, 34)
    DIRT = (58, 48, 40)

    def _scaled(sk: Skeleton, k: float, ax: float, ay: float) -> Skeleton:
        """A skeleton at *k* times the size, about the feet.

        The 0.5x cell has to be a genuine small drawing, not a big one shrunk
        with transform.smoothscale - that would measure the scaler, not the
        art, and the whole question is whether the shapes survive being drawn
        at half size.
        """
        def p(q):
            return (ax + (q[0] - ax) * k, ay + (q[1] - ay) * k)
        return Skeleton(
            hip=p(sk.hip), neck=p(sk.neck), shoulder=p(sk.shoulder),
            head=p(sk.head), head_r=sk.head_r * k,
            knees=[p(q) for q in sk.knees], feet=[p(q) for q in sk.feet],
            elbows=[p(q) for q in sk.elbows], hands=[p(q) for q in sk.hands],
            height=sk.height * k, facing=sk.facing, pose=sk.pose, girth=sk.girth,
        )

    def _bones(surf: pygame.Surface, sk: Skeleton, col) -> None:
        """stickfigure._draw's limb pass, minus everything it hangs off."""
        w = max(1, int(round(sk.height / 26.0)))
        far = _dim(col, 0.62)
        _line(surf, far, sk.hip, sk.knees[1], w)
        _line(surf, far, sk.knees[1], sk.feet[1], w)
        _line(surf, far, sk.shoulder, sk.elbows[1], w)
        _line(surf, far, sk.elbows[1], sk.hands[1], w)
        _line(surf, col, sk.hip, sk.neck, w + 1)
        _line(surf, col, sk.hip, sk.knees[0], w)
        _line(surf, col, sk.knees[0], sk.feet[0], w)
        _line(surf, col, sk.shoulder, sk.elbows[0], w)
        _line(surf, col, sk.elbows[0], sk.hands[0], w)
        _disc(surf, col, sk.head, sk.head_r)

    # ---------------------------------------------------------------- sheet --
    CW, CH = 104, 116
    COLS = len(KINDS)
    ROWS = 7
    SW = max(CW * COLS, 1040)         # the spectacle strip is wider than the grid
    sheet = pygame.Surface((SW, CH * ROWS + 286))
    sheet.fill(BG)
    font = pygame.font.Font(None, 15)

    def label(text: str, x: int, y: int, col=(150, 156, 168)) -> None:
        sheet.blit(font.render(text, True, col), (x, y))

    rng = random.Random(4)          # spawn only; nothing drawn here is random
    pop = Population()

    rows = [
        ("ground 1.0x lit", 1.0, False, False),
        ("ground 1.0x flat", 1.0, True, False),
        ("ground 0.5x lit", 0.5, False, False),
        ("ground 0.5x flat", 0.5, True, False),
        ("worn 1.0x lit", 1.0, False, True),
        ("worn 1.0x flat", 1.0, True, True),
        ("worn 0.5x lit", 0.5, False, True),
    ]
    for r, (name, k, is_flat, worn) in enumerate(rows):
        gy = r * CH + CH - 26
        pygame.draw.rect(sheet, DIRT, pygame.Rect(0, gy, SW, 3))
        label(name, 4, r * CH + 4)
        for c, kind in enumerate(KINDS):
            cx = c * CW + CW // 2
            flat = _SIL_COLOR if is_flat else None
            if worn:
                a = pop.spawn(cx, gy, rng=rng)
                a.pose_override = "idle" if c % 2 else "walk"
                a.anim_t, a.vx = 1.7, 20.0
                # Strip the loadout the spawner may have handed out. A carried
                # resource icon is a 4 px disc at the fist and a torch is a
                # flame at head height; both sit exactly where two of the five
                # relics are drawn, and the first sheet was unreadable because
                # of it.
                a.carrying, a.carry_qty = None, 0
                a.holds_candle = a.holds_torch = False
                a.speech = ""
                sk = build_skeleton(a, 1.35)
                if k == 1.0:
                    draw_stickman(sheet, a, 1.35, _SIL_COLOR if is_flat else None)
                else:
                    # A real half-size figure, drawn bone by bone. Shrinking a
                    # 1.0x cell with smoothscale would measure the scaler and
                    # not the art, and the whole question in this row is
                    # whether the shapes survive being *drawn* small.
                    sk = _scaled(sk, k, cx, gy)
                    body = _SIL_COLOR if is_flat else tuple(int(v) for v in a.color)
                    _bones(sheet, sk, body)
                draw_worn(sheet, a, sk, 1.35, flat, 1.0, kind=kind)
            else:
                draw_relic(sheet, kind, cx, gy, scale=k, flat=flat)
                if not is_flat:
                    draw_relic_fx(sheet, types.SimpleNamespace(
                        relics=types.SimpleNamespace(relics=[types.SimpleNamespace(
                            id=c * 3 + 1, kind=kind, x=cx, y=gy, age=0.0)])), 1.35)
            if r == 0:
                label(kind, c * CW + 4, CH - 16, (120, 126, 138))

    # spectacle strip. A stickman is drawn at the muzzle of the middle beam and
    # beside the misfire, because the only question worth asking about either
    # is *how big is this next to the man holding it* - the corridor is 22 px
    # against a 26 px villager and the blast radius is 70.
    base = ROWS * CH + 8
    label("bfg beam u=0.05 / 0.35 / 0.75", 4, base + 2)
    for i, u in enumerate((0.05, 0.35, 0.75)):
        ox = i * 175 + 24
        if i == 1:
            shooter = pop.spawn(ox, base + 96, rng=rng)
            shooter.carrying, shooter.carry_qty = None, 0
            shooter.holds_candle = shooter.holds_torch = False
            shooter.speech = ""
            draw_stickman(sheet, shooter, 1.35)
            draw_worn(sheet, shooter, None, 1.35, kind=RELIC_BFG)
        draw_beam(sheet, ox, base + 76, ox + 150, base + 44, u)
    # The misfire, with a NEIGHBOUR standing one body-width away who must
    # visibly survive it. That second figure is the whole assertion of the
    # corrected spec - the gun takes the man holding it and nobody else - and
    # it is on the sheet rather than in a comment because the failure mode is a
    # drawing that quietly grows back into an area attack.
    label("misfire u=0.45 - wielder only", 560, base + 2)
    for mx, mu in ((628, True), (662, False)):
        who = pop.spawn(mx, base + 96, rng=rng)
        who.carrying, who.carry_qty = None, 0
        who.holds_candle = who.holds_torch = False
        who.speech = ""
        if mu:
            who.weapon, who.relic = RELIC_BFG, RELIC_BFG
        draw_stickman(sheet, who, 1.35)
        if mu:
            draw_worn(sheet, who, None, 1.35, kind=RELIC_BFG)
    draw_misfire(sheet, 628, base + 76, 0.45)
    # The wreck at two points in its 1.4 s fall, hull included: creatures.py
    # draws the saucer wherever y puts it and PHASE_WRECK is not in its skip
    # list, so this cell is also the *proof* of that claim - if a later change
    # adds "wreck" to that list, this cell loses its saucer and says so.
    from .creatures import draw_ufo
    for i, (wx, wt) in enumerate(((790.0, 0.18), (940.0, 1.05))):
        label(f"wreck t={wt:.2f}", int(wx) - 46, base + 2)
        _wr = types.SimpleNamespace(
            x=wx, y=base + 44.0 + 46.0 * (wt / UFO_WRECK_FALL_SEC),
            width=54.0, t=wt, phase="wreck", id=3 + i, wrecked=True)
        _wworld = types.SimpleNamespace(
            ufo=_wr,
            terrain=types.SimpleNamespace(ground_y=lambda _x: base + 90.0))
        draw_ufo(sheet, _wworld, 1.35)
        draw_wreck(sheet, _wworld, 1.35)

    # ---------------------------------------------- the beam's two clocks --
    # The regression cell. Every one of these goes through draw_relic_fx and a
    # world object, not through draw_beam, because the bug was never in the
    # drawing - it was in what `t` was taken to mean, and only the full path
    # exercises that. Cells 2 and 3 are the ones that were blank before:
    # combat_actions stamps `t` with world_now(), so every shot in a world older
    # than BEAM_LIFE resolved to an age of hours and drew nothing at all.
    b2 = ROWS * CH + 116
    label("draw_relic_fx via a WORLD: absolute stamps (combat_actions) vs aged "
          "records (RelicRegistry)", 4, b2 + 2)
    _cells = (
        ("bfg_beams t=0.05 @world 0.05", 0.05, 0.05, True),
        ("bfg_beams t=6000 @world 6000", 6000.0, 6000.0, True),
        ("...0.14 s old", 6000.0, 5999.86, True),
        ("...0.40 s old: gone", 6000.0, 5999.60, True),
        ("relics.beams age=0.05", 6000.0, 0.05, False),
    )
    for i, (name, wt, stamp, absolute) in enumerate(_cells):
        ox = i * 205 + 20
        label(name, ox, b2 + 18, (120, 126, 138))
        _rec = {"x0": float(ox), "y0": b2 + 86.0,
                "x1": ox + 160.0, "y1": b2 + 44.0,
                "corridor": BFG_CORRIDOR, "t": stamp}
        if absolute:
            _bw = types.SimpleNamespace(world_time=wt, bfg_beams=[_rec])
        else:
            _bw = types.SimpleNamespace(
                world_time=wt, relics=types.SimpleNamespace(beams=[_rec]))
        draw_relic_fx(sheet, _bw, wt)

    out = os.path.join(os.environ.get("TEMP", "."), "relics_sheet.png")
    pygame.image.save(sheet, out)
    print("wrote", out, sheet.get_size())

    # ----------------------------------------------------------- fail-soft --
    class _Bare:
        pass

    for junk in (_Bare(), None, types.SimpleNamespace(relics=None),
                 types.SimpleNamespace(relics=types.SimpleNamespace(
                     relics=[None, 7, "relic", types.SimpleNamespace(x="?", y=1.0),
                             types.SimpleNamespace(kind="nope", x=1.0, y=1.0)]))):
        draw_relics(sheet, junk, 0.0)
        draw_relic_fx(sheet, junk, 0.0)
    draw_worn(sheet, _Bare(), None, 0.0)
    draw_worn(sheet, _Bare(), types.SimpleNamespace(hip=(1, 1)), 0.0, kind=RELIC_SCALE)
    draw_beam(sheet, 0.0, 0.0, 0.0, 0.0, 0.5)
    draw_relic(sheet, "not-a-relic", 10, 10)
    print("fail-soft OK")

    # ----------------------------------------------------------------- cost --
    empty = _Bare()
    six = types.SimpleNamespace(relics=types.SimpleNamespace(relics=[
        types.SimpleNamespace(id=i, kind=KINDS[i % len(KINDS)],
                              x=100.0 + i * 60.0, y=300.0, age=10.0)
        for i in range(6)]))

    def cost(world, n: int = 400) -> tuple[float, float]:
        t0 = _time.perf_counter()
        for i in range(n):
            draw_relics(sheet, world, i * 0.01)
        a = (_time.perf_counter() - t0) * 1e6 / n
        t0 = _time.perf_counter()
        for i in range(n):
            draw_relic_fx(sheet, world, i * 0.01)
        return a, (_time.perf_counter() - t0) * 1e6 / n

    e1, e2 = cost(empty)
    s1, s2 = cost(six)
    print(f"cost: empty {e1:.2f} + {e2:.2f} us/frame; "
          f"six relics {s1:.2f} + {s2:.2f} us/frame")

    beam_world = types.SimpleNamespace(relics=types.SimpleNamespace(
        beams=[{"x0": 200.0, "y0": 300.0, "x1": 600.0, "y1": 260.0, "t": 0.1}]))
    t0 = _time.perf_counter()
    for i in range(200):
        draw_relic_fx(sheet, beam_world, i * 0.01)
    print(f"      one beam {((_time.perf_counter() - t0) * 1e6 / 200):.2f} us/frame, "
          f"caches {cache_stats()}")
    pygame.quit()
