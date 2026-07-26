"""Vignettes - the small useless things a stickman does when nobody needs it.

A vignette is *cosmetic filler*: scratching an ear, skipping a stone, staring
at the moon, warming its hands, tripping over its own feet. It is deliberately
**not** part of the utility AI. Nothing here gathers, builds, consumes or
changes the world; the colony's economy is identical whether vignettes fire or
not. What changes is whether the colony reads as alive or as a shuttle of
workers, which is the entire point.

Contract every vignette obeys
-----------------------------
* It touches only the acting agent's cosmetic fields - ``pose``, ``facing``,
  ``speech``, ``anim_t`` - plus a horizontal nudge of at most
  :data:`MAX_DRIFT` px from where it started.
* It terminates. Durations are clamped into ``[MIN_DUR, MAX_DUR]`` and the
  state machine additionally hard-stops past its own duration.
* It never raises. :meth:`VignetteAction.update` is a per-frame path; a broken
  vignette marks itself failed and the agent gets on with its life.

This module is the *engine*. The content lives in ``vignettes_a.py`` ...
``vignettes_e.py``, each exposing a module-level list called ``VIGNETTES``;
:func:`load_all` imports them lazily (one bad file cannot break the others)
and is called once at import time. If none of them exist, a small built-in set
is registered so the colony still has something to do.

No pygame. Pure python; numpy is not needed here at all.
"""
from __future__ import annotations

import importlib
import pathlib
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..constants import RENDER_W, SCENE_BLIZZARD, SCENE_NIGHT_STORM

log = logging.getLogger(__name__)

__all__ = [
    "Vignette",
    "VignetteAction",
    "REGISTRY",
    "MOTIONS",
    "KNOWN_TAGS",
    "MIN_DUR",
    "MAX_DUR",
    "MAX_DRIFT",
    "register",
    "all_vignettes",
    "by_tag",
    "get",
    "context_tags",
    "pick",
    "make_vignette_action",
    "vignette_action_from_dict",
    "load_all",
    "describe",
]

# ------------------------------------------------------------------ tuning --
MIN_DUR = 1.5               # s - shorter than this and it reads as a twitch
MAX_DUR = 9.0               # s - longer than this and it reads as being stuck
MAX_DRIFT = 40.0            # px - hard cap on how far a vignette may wander
RECENT_MEMORY = 2           # how many of an agent's last vignettes are barred

#: Motion styles the state machine knows how to apply.
MOTIONS: tuple[str, ...] = (
    "still", "pace", "hop", "crouch", "spin", "sit", "lie",
)

# Context thresholds. Needs are 0..1 with 1 = worst (morale is inverted).
TIRED_AT = 0.55
HUNGRY_AT = 0.55
COLD_AT = 0.55
HAPPY_AT = 0.72
SAD_AT = 0.30

# Context distances, in world px.
FIRE_DIST = 120.0
WATER_DIST = 90.0
PROP_DIST = 90.0
SOCIAL_DIST = 80.0
ALONE_DIST = 160.0

WEATHER_AT = 0.25           # events.rain/.snow above this counts as weather
GUST_AT = 0.60              # events.gust above this counts as a storm

HOP_RATE = 2.6              # extra anim_t per second while hopping
SPIN_PERIOD = 0.75          # s between facing flips while spinning
PACE_MIN_PERIOD = 1.6       # s for one there-and-back pace sweep

#: Tags :func:`context_tags` knows how to evaluate. A vignette tagged with
#: anything *outside* this set is not gated on it - unknown tags are treated as
#: decorative labels ("silly", "quiet") so a content author cannot accidentally
#: write a vignette that can never fire.
KNOWN_TAGS: frozenset[str] = frozenset({
    "night", "day",
    "raining", "snowing", "storm",
    "near_fire", "near_water", "near_tree", "near_rock",
    "high_up", "alone", "social",
    "tired", "hungry", "cold", "happy", "sad",
    "child", "elder", "builder", "gatherer", "lookout", "adult",
    "candle", "carrying", "windy",
})

_TREE_KINDS = frozenset({"tree", "sapling", "pine", "oak", "deadtree"})
_ROCK_KINDS = frozenset({"rock", "boulder", "stone", "outcrop"})
_WATER_KINDS = frozenset({"water", "pond", "pool"})

#: Content modules the registry is populated from, in load order.
def _discover_content_modules() -> list[str]:
    """Find every vignettes_*.py sitting next to this file.

    Content arrives in bands authored independently, so hard-coding the list
    means each new band needs an edit here and silently contributes nothing if
    that is forgotten. Globbing the directory makes a band live the moment it
    lands.
    """
    try:
        here = pathlib.Path(__file__).parent
        names = sorted(p.stem for p in here.glob("vignettes_*.py")
                       if p.stem != "vignettes")
        return names
    except Exception:
        return []


CONTENT_MODULES = _discover_content_modules()

_FALLBACK_RNG = random.Random(0x5CA1AB1E)


# ===========================================================================
#  Accessor layer
#
#  actions.py already owns a hardened set of world accessors; reuse them when
#  they are importable and fall back to local equivalents when they are not,
#  so this module stays usable on its own (and testable in isolation).
# ===========================================================================
def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _clamp_x(x: float) -> float:
    return _clamp(_f(x, 0.0), 2.0, float(RENDER_W - 2))


def _fallback_rng_of(world: Any) -> random.Random:
    for name in ("pyrng", "rng"):
        r = getattr(world, name, None)
        if isinstance(r, random.Random):
            return r
    return _FALLBACK_RNG


def _fallback_is_night(world: Any) -> bool:
    v = getattr(world, "is_night", None)
    if isinstance(v, bool):
        return v
    if callable(v):
        try:
            return bool(v())
        except Exception:
            return False
    return False


def _fallback_props_of(world: Any) -> list[Any]:
    v = getattr(world, "props", None)
    if isinstance(v, (list, tuple)):
        return list(v)
    fn = getattr(v, "all", None)
    if callable(fn):
        try:
            return list(fn())
        except Exception:
            return []
    return []


def _fallback_prop_alive(prop: Any) -> bool:
    return bool(prop is not None and getattr(prop, "alive", True)
                and not getattr(prop, "removed", False))


def _fallback_structures_of(world: Any) -> Any:
    return getattr(world, "structures", None)


_rng_of = _fallback_rng_of
_is_night = _fallback_is_night
_props_of = _fallback_props_of
_prop_alive = _fallback_prop_alive
_structures_of = _fallback_structures_of

try:  # pragma: no cover - exercised implicitly by the app
    from .actions import (            # noqa: PLC0415
        is_night as _is_night,
        prop_alive as _prop_alive,
        props_of as _props_of,
        rng_of as _rng_of,
        structures_of as _structures_of,
    )
except Exception:                     # pragma: no cover - standalone use
    log.debug("actions.py accessors unavailable; using local fallbacks",
              exc_info=True)


def _agents_of(world: Any) -> list[Any]:
    """Every living agent, however this world spells its roster.

    ``World`` keeps its roster on ``world.population`` rather than
    ``world.agents``, so this checks both rather than silently returning an
    empty list (which would make "alone" always true and "social" never).
    """
    for name in ("agents", "stickmen", "people"):
        v = getattr(world, name, None)
        if isinstance(v, (list, tuple)):
            return [a for a in v if getattr(a, "alive", True)]
    pop = getattr(world, "population", None)
    if pop is not None:
        fn = getattr(pop, "alive_agents", None)
        if callable(fn):
            try:
                return list(fn())
            except Exception:
                pass
        v = getattr(pop, "agents", None)
        if isinstance(v, (list, tuple)):
            return [a for a in v if getattr(a, "alive", True)]
        try:
            return [a for a in pop if getattr(a, "alive", True)]
        except Exception:
            return []
    return []


def _events_of(world: Any) -> Any:
    return getattr(world, "events", None)


def _dist(agent: Any, other: Any) -> float:
    return math.hypot(_f(getattr(other, "x", 0.0)) - _f(getattr(agent, "x", 0.0)),
                      _f(getattr(other, "y", 0.0)) - _f(getattr(agent, "y", 0.0)))


def _prop_kind(prop: Any) -> str:
    k = getattr(prop, "kind", None) or getattr(prop, "type", None) or ""
    return str(k).lower()


def _terrain_range(world: Any) -> tuple[float, float] | None:
    """(highest, lowest) ground y across the world, or None if unknowable."""
    terrain = getattr(world, "terrain", None)
    if terrain is None:
        return None
    h = getattr(terrain, "height", None)
    if h is not None:
        try:
            lo = float(h.min()) if hasattr(h, "min") else float(min(h))
            hi = float(h.max()) if hasattr(h, "max") else float(max(h))
            if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                return (lo, hi)
        except Exception:
            pass
    fn = getattr(terrain, "ground_y", None)
    if not callable(fn):
        return None
    try:
        ys = [_f(fn(x), 0.0) for x in range(0, RENDER_W, max(1, RENDER_W // 24))]
    except Exception:
        return None
    ys = [y for y in ys if math.isfinite(y)]
    if len(ys) < 2:
        return None
    lo, hi = min(ys), max(ys)
    return (lo, hi) if hi > lo else None


# ===========================================================================
#  Vignette
# ===========================================================================
@dataclass(frozen=True)
class Vignette:
    """One cosmetic beat. Immutable, so the registry can be shared freely."""

    key: str                            # unique snake_case id
    label: str                          # human phrase, e.g. "scratches an ear"
    pose: str                           # base pose for the renderer
    dur: tuple[float, float]            # (min, max) seconds
    weight: float = 1.0                 # relative pick weight
    tags: tuple[str, ...] = ()          # e.g. ("night", "near_fire")
    speech: str | None = None           # optional bubble symbol
    motion: str = "still"               # see MOTIONS
    drift: float = 0.0                  # max px of horizontal drift, total

    def __post_init__(self) -> None:
        # Normalise rather than reject: content authors should not be able to
        # take the colony down with a typo, and a clamped vignette still plays.
        s = object.__setattr__
        s(self, "key", str(self.key).strip().lower())
        s(self, "label", str(self.label).strip() or self.key.replace("_", " "))
        s(self, "pose", str(self.pose).strip() or "idle")

        lo, hi = MIN_DUR, MAX_DUR
        try:
            a, b = (float(self.dur[0]), float(self.dur[1]))
            if not (math.isfinite(a) and math.isfinite(b)):
                raise ValueError
            if b < a:
                a, b = b, a
            lo = _clamp(a, MIN_DUR, MAX_DUR)
            hi = _clamp(b, MIN_DUR, MAX_DUR)
            if hi < lo:
                hi = lo
        except Exception:
            log.warning("vignette %r has an unusable duration %r; using %.1f-%.1fs",
                        self.key, self.dur, MIN_DUR, MAX_DUR)
        s(self, "dur", (lo, hi))

        s(self, "weight", max(0.0, _f(self.weight, 1.0)))
        s(self, "tags", tuple(
            str(t).strip().lower() for t in (self.tags or ()) if str(t).strip()))
        s(self, "speech", str(self.speech) if self.speech else None)
        motion = str(self.motion or "still").strip().lower()
        s(self, "motion", motion if motion in MOTIONS else "still")
        s(self, "drift", _clamp(_f(self.drift, 0.0), 0.0, MAX_DRIFT))

    # ------------------------------------------------------------ helpers --
    @property
    def gates(self) -> frozenset[str]:
        """The subset of ``tags`` that :func:`context_tags` actually gates on."""
        return frozenset(t for t in self.tags if t in KNOWN_TAGS)

    def duration(self, rng: Any = None) -> float:
        """A concrete duration for one playing, always inside the legal band."""
        lo, hi = self.dur
        if hi <= lo:
            return _clamp(lo, MIN_DUR, MAX_DUR)
        try:
            v = float(rng.uniform(lo, hi))       # type: ignore[union-attr]
        except Exception:
            v = _FALLBACK_RNG.uniform(lo, hi)
        return _clamp(v, MIN_DUR, MAX_DUR)

    def matches(self, ctx: Iterable[str]) -> bool:
        """True when every gating tag is satisfied by *ctx*."""
        return self.gates <= set(ctx)


# ===========================================================================
#  Registry
# ===========================================================================
REGISTRY: dict[str, Vignette] = {}


def register(*vs: Vignette) -> None:
    """Add vignettes to the registry. Duplicate keys are rejected loudly.

    Loudly on purpose: two content modules quietly clobbering each other's
    ``scratch_head`` is exactly the bug that is impossible to see from the
    wallpaper. :func:`load_all` catches this per item so one collision costs
    one vignette, not a whole file.
    """
    for v in vs:
        if not isinstance(v, Vignette):
            raise TypeError(f"register() takes Vignette, got {type(v).__name__}")
        if not v.key:
            raise ValueError("vignette has an empty key")
        existing = REGISTRY.get(v.key)
        if existing is not None:
            if existing == v:
                continue                       # idempotent re-registration
            raise ValueError(f"duplicate vignette key {v.key!r}")
        REGISTRY[v.key] = v


def all_vignettes() -> list[Vignette]:
    """Every registered vignette, in a stable (key-sorted) order."""
    return [REGISTRY[k] for k in sorted(REGISTRY)]


def by_tag(tag: str) -> list[Vignette]:
    """Every vignette carrying *tag*."""
    t = str(tag).strip().lower()
    return [v for v in all_vignettes() if t in v.tags]


def get(key: str) -> Vignette | None:
    """Look a vignette up by key."""
    return REGISTRY.get(str(key).strip().lower())


def describe(agent: Any, v: Vignette) -> str:
    """A chronicle-ready sentence. Nothing here writes it - callers decide."""
    name = str(getattr(agent, "name", None) or "Someone")
    return f"{name} {v.label}."


# ===========================================================================
#  Context matching
# ===========================================================================
def context_tags(agent: Any, world: Any) -> set[str]:
    """Which :data:`KNOWN_TAGS` currently apply to *agent*.

    Cheap and defensive: every probe is individually guarded, so a missing or
    half-built subsystem costs one tag rather than the whole set. Called once
    per vignette pick (roughly once every few seconds per agent), never per
    frame.
    """
    tags: set[str] = set()
    try:
        _time_tags(agent, world, tags)
        _weather_tags(agent, world, tags)
        _place_tags(agent, world, tags)
        _company_tags(agent, world, tags)
        _need_tags(agent, world, tags)
        _identity_tags(agent, world, tags)
    except Exception:                       # pragma: no cover - belt and braces
        log.debug("context_tags failed", exc_info=True)
    return tags


def _time_tags(agent: Any, world: Any, tags: set[str]) -> None:
    try:
        tags.add("night" if _is_night(world) else "day")
    except Exception:
        tags.add("day")


def _weather_tags(agent: Any, world: Any, tags: set[str]) -> None:
    ev = _events_of(world)
    if ev is None:
        return
    scene = str(getattr(ev, "scene", "") or "")
    rain = _f(getattr(ev, "rain", 0.0))
    snow = _f(getattr(ev, "snow", 0.0))
    gust = _f(getattr(ev, "gust", 0.0))
    wind = abs(_f(getattr(ev, "wind", 0.0)))
    if rain > WEATHER_AT:
        tags.add("raining")
    if snow > WEATHER_AT or scene == SCENE_BLIZZARD:
        tags.add("snowing")
    if scene == SCENE_NIGHT_STORM or gust > GUST_AT:
        tags.add("storm")
    if gust > 0.4 or wind > 0.5:
        tags.add("windy")


def _place_tags(agent: Any, world: Any, tags: set[str]) -> None:
    ax = _f(getattr(agent, "x", 0.0))
    ay = _f(getattr(agent, "y", 0.0))

    # --- fire: a lit firepit, a burning structure, or a burning prop --------
    reg = _structures_of(world)
    if reg is None:
        # actions.structures_of only recognises a full StructureRegistry; any
        # object that can list its structures is good enough for a proximity
        # check, so try the raw attribute before giving up on fire entirely.
        reg = getattr(world, "structures", None)
    if reg is not None:
        try:
            for s in (reg.all() if hasattr(reg, "all") else reg):
                if abs(_f(getattr(s, "x", 0.0)) - ax) > FIRE_DIST:
                    continue
                if getattr(s, "fire_active", False) or getattr(s, "is_burning", False):
                    tags.add("near_fire")
                    break
        except Exception:
            log.debug("structure fire probe failed", exc_info=True)

    # --- props: water / trees / rocks, and burning ones are fire too --------
    try:
        props = _props_of(world)
    except Exception:
        props = []
    water_seen = False
    for p in props:
        try:
            dx = abs(_f(getattr(p, "x", 0.0)) - ax)
            if dx > FIRE_DIST:
                continue
            if getattr(p, "burning", False) and dx <= FIRE_DIST:
                tags.add("near_fire")
            if dx > PROP_DIST:
                continue
            kind = _prop_kind(p)
            if kind in _WATER_KINDS:
                water_seen = True
                tags.add("near_water")
                continue
            if not _prop_alive(p):
                continue
            if kind in _TREE_KINDS:
                tags.add("near_tree")
            elif kind in _ROCK_KINDS:
                tags.add("near_rock")
        except Exception:
            continue

    # --- flood water counts as water even where no water prop exists --------
    if not water_seen:
        ev = _events_of(world)
        level = getattr(ev, "water_level", None) if ev is not None else None
        if isinstance(level, (int, float)) and math.isfinite(float(level)):
            if ay > float(level) - WATER_DIST:
                tags.add("near_water")

    # --- height: the top third of the terrain's vertical range --------------
    rng_ = _terrain_range(world)
    if rng_ is not None:
        lo, hi = rng_
        if ay <= lo + (hi - lo) / 3.0:
            tags.add("high_up")


def _company_tags(agent: Any, world: Any, tags: set[str]) -> None:
    nearest = float("inf")
    my_id = getattr(agent, "id", None)
    for other in _agents_of(world):
        if other is agent:
            continue
        if my_id is not None and getattr(other, "id", None) == my_id:
            continue
        d = _dist(agent, other)
        if d < nearest:
            nearest = d
            if nearest <= SOCIAL_DIST:
                break
    if nearest <= SOCIAL_DIST:
        tags.add("social")
    elif nearest > ALONE_DIST:
        tags.add("alone")


def _need_tags(agent: Any, world: Any, tags: set[str]) -> None:
    if _f(getattr(agent, "fatigue", 0.0)) >= TIRED_AT:
        tags.add("tired")
    if _f(getattr(agent, "hunger", 0.0)) >= HUNGRY_AT:
        tags.add("hungry")
    if _f(getattr(agent, "warmth", 0.0)) >= COLD_AT:
        tags.add("cold")
    morale = _f(getattr(agent, "morale", 0.6), 0.6)
    if morale >= HAPPY_AT:
        tags.add("happy")
    elif morale <= SAD_AT:
        tags.add("sad")


def _identity_tags(agent: Any, world: Any, tags: set[str]) -> None:
    role = str(getattr(agent, "role", "") or "").strip().lower()
    if role in ("child", "elder", "builder", "gatherer", "lookout"):
        tags.add(role)
    if role and role != "child":
        tags.add("adult")
    if getattr(agent, "holds_candle", False):
        tags.add("candle")
    if getattr(agent, "carrying", None) and int(_f(getattr(agent, "carry_qty", 0))) > 0:
        tags.add("carrying")


# ===========================================================================
#  Picking
# ===========================================================================
def _coerce_rng(rng: Any, world: Any) -> Any:
    if rng is not None and all(hasattr(rng, n) for n in ("random", "uniform")):
        return rng
    try:
        return _rng_of(world)
    except Exception:
        return _FALLBACK_RNG


def recent_of(agent: Any) -> list[str]:
    """The agent's ``_recent_vignettes`` list, created on demand."""
    r = getattr(agent, "_recent_vignettes", None)
    if not isinstance(r, list):
        r = []
        try:
            setattr(agent, "_recent_vignettes", r)
        except Exception:
            return []
    if len(r) > RECENT_MEMORY:
        del r[:-RECENT_MEMORY]
    return r


def _remember(agent: Any, key: str) -> None:
    r = recent_of(agent)
    try:
        while key in r:
            r.remove(key)
        r.append(key)
        if len(r) > RECENT_MEMORY:
            del r[:-RECENT_MEMORY]
    except Exception:
        log.debug("could not record recent vignette", exc_info=True)


def candidates(agent: Any, world: Any, ctx: set[str] | None = None) -> list[Vignette]:
    """Every vignette that fits the context and is not one of the last two."""
    if not REGISTRY:
        return []
    if ctx is None:
        ctx = context_tags(agent, world)
    barred = set(recent_of(agent))
    return [v for v in REGISTRY.values()
            if v.weight > 0.0 and v.key not in barred and v.gates <= ctx]


def pick(agent: Any, world: Any, rng: Any = None) -> Vignette | None:
    """Weighted choice over the vignettes whose tags the context satisfies.

    Never returns either of the agent's previous two vignettes - the repeat is
    what makes a colony read as a looping animation instead of a life. Returns
    None when nothing fits, and the caller should fall back to whatever it
    would otherwise have done.
    """
    try:
        pool = candidates(agent, world)
        if not pool:
            return None
        r = _coerce_rng(rng, world)
        total = math.fsum(v.weight for v in pool)
        if total <= 0.0:
            chosen = pool[0]
        else:
            try:
                roll = float(r.random()) * total
            except Exception:
                roll = _FALLBACK_RNG.random() * total
            acc = 0.0
            chosen = pool[-1]
            for v in pool:
                acc += v.weight
                if roll < acc:
                    chosen = v
                    break
        _remember(agent, chosen.key)
        return chosen
    except Exception:
        log.debug("vignette pick failed", exc_info=True)
        return None


# ===========================================================================
#  VignetteAction - the resumable state machine
# ===========================================================================
def _jsonable(v: Any, depth: int = 0) -> Any:
    if depth > 3:
        return None
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else 0.0
    if isinstance(v, (list, tuple)):
        return [_jsonable(x, depth + 1) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x, depth + 1) for k, x in v.items()}
    return None


@dataclass
class VignetteAction:
    """A vignette in flight. Same shape as :class:`actions.Action` on purpose.

    ``kind``/``pose``/``phase``/``t``/``data``/``done``/``failed`` plus
    ``update``/``to_dict``/``from_dict``/``abandon``/``finished``, so world.py
    and behavior.py can drive it without knowing it is not a real Action.
    """

    kind: str = "Vignette"
    pose: str = "idle"
    target: Any = None
    phase: str = "start"
    t: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    failed: bool = False

    # ------------------------------------------------------------ queries --
    @property
    def finished(self) -> bool:
        return bool(self.done or self.failed)

    @property
    def vignette(self) -> Vignette | None:
        return get(str(self.data.get("key", "")))

    @property
    def label(self) -> str:
        v = self.vignette
        return v.label if v is not None else str(self.data.get("label", ""))

    def describe(self, agent: Any) -> str:
        v = self.vignette
        if v is None:
            return ""
        return describe(agent, v)

    def __repr__(self) -> str:      # pragma: no cover - debug sugar
        flag = "done" if self.done else ("failed" if self.failed else self.phase)
        return (f"<Vignette:{self.data.get('key', '?')} {flag} "
                f"t={self.t:.1f}/{_f(self.data.get('dur'), 0.0):.1f}>")

    # ---------------------------------------------------------------- run --
    def update(self, agent: Any, world: Any, dt: float) -> None:
        """Advance by *dt*. Never raises - this runs inside the frame loop."""
        if self.done or self.failed:
            return
        step = _clamp(_f(dt, 0.0), 0.0, 0.25)   # a stalled frame must not warp
        self.t += step
        try:
            self._step(agent, world, step)
        except Exception:
            log.debug("vignette %r failed", self.data.get("key"), exc_info=True)
            self.failed = True
            self._release(agent)

    def abandon(self, agent: Any, world: Any) -> None:
        """Give up early (something more important came along)."""
        self._release(agent)
        self.done = True

    # ----------------------------------------------------------- internals --
    def _step(self, agent: Any, world: Any, dt: float) -> None:
        v = self.vignette
        if v is None:
            if self.phase != "start":
                # The vignette vanished under us (content module removed
                # between save and load). Stop cleanly rather than hanging.
                self.done = True
                self._release(agent)
                return
            v = pick(agent, world, _coerce_rng(None, world))
            if v is None:
                self.failed = True
                return

        if self.phase == "start":
            # Covers both entry points: make_vignette_action() hands us a key
            # but no timings, and a bare VignetteAction picks its own above.
            self._begin(agent, world, v)

        self.pose = v.pose
        _set_pose(agent, v.pose)

        dur = _f(self.data.get("dur"), v.dur[1])
        if dur <= 0.0:
            dur = v.dur[1]

        self._motion(agent, world, v, dt, dur)

        # Two independent stops: the chosen duration, and a hard ceiling that
        # holds even if `dur` was corrupted by a bad save.
        if self.t >= dur or self.t >= MAX_DUR + 1.0:
            self.done = True
            self._release(agent)

    def _begin(self, agent: Any, world: Any, v: Vignette) -> None:
        rng = _coerce_rng(None, world)
        self.data["key"] = v.key
        self.data["label"] = v.label
        self.data["dur"] = v.duration(rng)
        self.data["x0"] = _f(getattr(agent, "x", 0.0))
        self.data["spin_t"] = 0.0
        try:
            self.data["dir"] = 1.0 if rng.random() < 0.5 else -1.0
        except Exception:
            self.data["dir"] = 1.0
        self.phase = "play"
        # Stop whatever walk was in flight; a vignette is a pause, not a trip.
        _halt(agent)
        if v.speech:
            _say(agent, v.speech, min(MAX_DUR, _f(self.data["dur"], 2.5)))

    def _motion(self, agent: Any, world: Any, v: Vignette, dt: float,
                dur: float) -> None:
        motion = v.motion
        if motion == "pace":
            self._pace(agent, v, dur)
            return

        _halt(agent)
        if motion == "hop":
            # The bob itself is the renderer's business; the sim only drives
            # the phase it bobs on.
            try:
                agent.anim_t = _f(getattr(agent, "anim_t", 0.0)) + dt * HOP_RATE
            except Exception:
                pass
        elif motion == "spin":
            self.data["spin_t"] = _f(self.data.get("spin_t"), 0.0) + dt
            if self.data["spin_t"] >= SPIN_PERIOD:
                self.data["spin_t"] = 0.0
                try:
                    agent.facing = -1 if int(getattr(agent, "facing", 1)) > 0 else 1
                except Exception:
                    pass
        # "still", "crouch", "sit", "lie" are pose-only: nothing else to do.

    def _pace(self, agent: Any, v: Vignette, dur: float) -> None:
        """A small there-and-back shuffle, strictly inside the drift budget."""
        budget = _clamp(v.drift, 0.0, MAX_DRIFT)
        if budget < 1.0:
            _halt(agent)
            return
        x0 = _f(self.data.get("x0"), _f(getattr(agent, "x", 0.0)))
        period = max(PACE_MIN_PERIOD, dur / 2.0)
        half = budget * 0.5
        prev = _f(getattr(agent, "x", x0))
        want = x0 + half * math.sin(2.0 * math.pi * self.t / period)
        want = _clamp(want, x0 - budget, x0 + budget)
        nx = _clamp_x(want)
        try:
            agent.x = nx
            agent.vx = (nx - prev) / max(1e-3, 0.033)
        except Exception:
            return
        d = nx - prev
        if abs(d) > 0.05:
            try:
                agent.facing = 1 if d > 0.0 else -1
            except Exception:
                pass

    def _release(self, agent: Any) -> None:
        _halt(agent)

    # ------------------------------------------------------------ persist --
    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "Vignette",
            "pose": str(self.pose),
            "target": None,
            "phase": str(self.phase),
            "t": _f(self.t, 0.0),
            "data": _jsonable(self.data) or {},
            "done": bool(self.done),
            "failed": bool(self.failed),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "VignetteAction":
        """Rebuild from a save. Junk in, a harmless finished action out."""
        a = cls()
        if not isinstance(d, dict):
            a.done = True
            return a
        pose = d.get("pose")
        a.pose = pose if isinstance(pose, str) and pose else "idle"
        a.phase = str(d.get("phase", "start")) or "start"
        a.t = max(0.0, _f(d.get("t"), 0.0))
        data = d.get("data")
        a.data = dict(data) if isinstance(data, dict) else {}
        a.done = bool(d.get("done", False))
        a.failed = bool(d.get("failed", False))
        key = str(a.data.get("key", "") or "")
        if key and key not in REGISTRY:
            # Content changed under the save; do not resurrect a ghost.
            a.data.pop("key", None)
            a.done = True
        a.data["dur"] = _clamp(_f(a.data.get("dur"), MAX_DUR), 0.0, MAX_DUR)
        return a


def _halt(agent: Any) -> None:
    try:
        agent.vx = 0.0
    except Exception:
        pass
    try:
        # Clear any walk intent so entities.apply_physics does not drag the
        # agent out from under its own vignette.
        if getattr(agent, "target_x", None) is not None:
            agent.target_x = None
    except Exception:
        pass


def _say(agent: Any, symbol: str, ttl: float) -> None:
    """Show a symbol bubble. Uses Stickman.say when it exists, fields if not."""
    if not symbol:
        return
    try:
        fn = getattr(agent, "say", None)
        if callable(fn):
            fn(str(symbol), _clamp(_f(ttl, 2.5), 0.5, MAX_DUR))
            return
        agent.speech = str(symbol)
        agent.speech_t = _clamp(_f(ttl, 2.5), 0.5, MAX_DUR)
    except Exception:
        log.debug("could not set speech", exc_info=True)


def _set_pose(agent: Any, pose: str) -> None:
    """Publish the pose.

    ``render.stickfigure.pose_of`` reads ``agent.action.pose`` first, which is
    the real channel; ``agent.pose`` is set too for anything that asks the
    agent directly. ``pose_override`` is deliberately *not* touched - it is
    persisted, and a vignette must not leave a sticky pose behind in a save.
    """
    try:
        agent.pose = pose
    except Exception:
        pass


def make_vignette_action(agent: Any, world: Any, rng: Any = None) -> VignetteAction | None:
    """Pick a vignette for *agent* and wrap it in an action. None if none fit.

    This is what behavior.py calls when an agent is idle or between jobs.
    """
    try:
        v = pick(agent, world, rng)
    except Exception:
        log.debug("make_vignette_action failed", exc_info=True)
        return None
    if v is None:
        return None
    a = VignetteAction(pose=v.pose)
    a.data["key"] = v.key
    a.data["label"] = v.label
    return a


def vignette_action_from_dict(d: Any) -> VignetteAction | None:
    """Rehydrate a saved action if (and only if) it is one of ours.

    ``actions.Action.from_dict`` coerces unknown kinds to ``Wander``, so the
    loader should try this first: ``vignette_action_from_dict(raw) or
    Action.from_dict(raw)``.
    """
    if isinstance(d, dict) and str(d.get("kind", "")) == "Vignette":
        return VignetteAction.from_dict(d)
    return None


# ===========================================================================
#  Content loading
# ===========================================================================
_loaded = False


def _register_one(v: Any, source: str) -> bool:
    """Tolerant single register used by the loader (never raises)."""
    try:
        if not isinstance(v, Vignette):
            log.warning("%s: ignoring a non-Vignette entry (%r)", source, type(v).__name__)
            return False
        register(v)
        return True
    except ValueError:
        log.warning("%s: duplicate vignette key %r ignored", source,
                    getattr(v, "key", "?"))
    except Exception:
        log.warning("%s: could not register a vignette", source, exc_info=True)
    return False


def load_all(force: bool = False) -> int:
    """Import every content module and register what it exposes.

    Each module is imported inside its own try/except so one broken file
    cannot cost the colony the other four. Returns the registry size.
    """
    global _loaded
    if _loaded and not force:
        return len(REGISTRY)
    _loaded = True

    found = 0
    for name in CONTENT_MODULES:
        try:
            mod = importlib.import_module(f".{name}", __package__)
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", None)
            if missing in (name, f"{__package__}.{name}"):
                continue                   # not written yet - entirely normal
            # The module exists but one of *its* imports does not; that is a
            # bug in the content file and should not vanish silently.
            log.warning("vignette module %s could not import %r", name, missing)
            continue
        except Exception:
            log.warning("vignette module %s failed to import", name, exc_info=True)
            continue
        try:
            items = getattr(mod, "VIGNETTES", None)
            if items is None:
                log.warning("vignette module %s exposes no VIGNETTES list", name)
                continue
            for v in list(items):
                if _register_one(v, name):
                    found += 1
        except Exception:
            log.warning("vignette module %s produced an unusable VIGNETTES",
                        name, exc_info=True)

    if not REGISTRY:
        # No content on disk. A colony with zero vignettes is the failure mode
        # this whole module exists to prevent, so ship a floor.
        for v in _BUILTIN:
            _register_one(v, "builtin")
        log.info("no vignette content modules found; registered %d built-ins",
                 len(REGISTRY))
    return len(REGISTRY)


#: A deliberately small, deliberately generic floor set. Only used when no
#: content module could be loaded, and prefixed so it cannot collide with a
#: content author's keys.
_BUILTIN: tuple[Vignette, ...] = (
    Vignette("core_scratch", "scratches an ear", "idle", (1.8, 3.2), 1.4),
    Vignette("core_stretch", "stretches", "idle", (2.0, 3.5), 1.2),
    Vignette("core_look_around", "looks around", "idle", (2.0, 4.0), 1.3,
             motion="spin"),
    Vignette("core_shuffle", "shuffles about", "walk", (3.0, 6.0), 1.0,
             motion="pace", drift=28.0),
    Vignette("core_sit", "sits down for a moment", "sleep", (4.0, 8.0), 0.8,
             motion="sit"),
    Vignette("core_ponder", "stands thinking", "idle", (2.5, 5.0), 1.0,
             speech="~"),
    Vignette("core_hop", "hops on the spot", "walk", (1.6, 3.0), 0.7,
             motion="hop"),
    Vignette("core_crouch", "crouches to look at something", "build",
             (2.2, 4.5), 0.9, motion="crouch"),
    Vignette("core_stargaze", "stares up at the sky", "idle", (3.0, 7.0), 1.1,
             tags=("night",), speech="*"),
    Vignette("core_warm_hands", "warms its hands", "idle", (3.0, 6.0), 1.3,
             tags=("near_fire",)),
    Vignette("core_skip_stone", "skips a stone", "chop", (2.4, 4.5), 1.1,
             tags=("near_water",), speech="o"),
    Vignette("core_yawn", "yawns", "idle", (1.8, 3.0), 1.2, tags=("tired",),
             speech="~"),
    Vignette("core_wave", "waves at someone", "idle", (1.6, 3.0), 1.2,
             tags=("social",), speech="!"),
    Vignette("core_sigh", "sighs at the horizon", "mourn", (2.5, 5.0), 1.0,
             tags=("alone",)),
    Vignette("core_caper", "capers about", "dance", (2.5, 5.0), 1.0,
             tags=("happy",), motion="hop"),
    Vignette("core_shiver", "shivers", "idle", (2.0, 4.0), 1.3, tags=("cold",)),
)


try:                                    # populate on import, fail soft
    load_all()
except Exception:                       # pragma: no cover
    log.warning("vignette content failed to load", exc_info=True)


# ===========================================================================
#  Smoke test
# ===========================================================================
if __name__ == "__main__":   # pragma: no cover - headless smoke test
    class _Terrain:
        def __init__(self) -> None:
            self.height = [600.0 + 70.0 * math.sin(x / 190.0)
                           for x in range(RENDER_W)]

        def ground_y(self, x: float) -> float:
            i = int(_clamp(x, 0.0, float(RENDER_W - 1)))
            return self.height[i]

        def slope(self, x: float) -> float:
            return 0.0

    class _Prop:
        def __init__(self, kind: str, x: float) -> None:
            self.kind = kind
            self.x = x
            self.y = 600.0
            self.alive = True
            self.burning = False

    class _Struct:
        kind = "firepit"

        def __init__(self, x: float) -> None:
            self.x = x
            self.y = 600.0
            self.fire_active = True
            self.is_burning = False

    class _Structures:
        def __init__(self, items: list[Any]) -> None:
            self._items = items

        def all(self) -> list[Any]:
            return list(self._items)

    class _Events:
        scene = SCENE_NIGHT_STORM
        rain = 0.6
        snow = 0.0
        wind = 0.4
        gust = 0.7
        water_level = None

    class _Agent:
        def __init__(self, aid: int, x: float) -> None:
            self.id = aid
            self.name = f"Agent{aid}"
            self.x = x
            self.y = 600.0
            self.vx = 0.0
            self.vy = 0.0
            self.facing = 1
            self.hunger = self.fatigue = self.warmth = 0.2
            self.morale = 0.6
            self.carrying = None
            self.carry_qty = 0
            self.role = "gatherer"
            self.alive = True
            self.anim_t = 0.0
            self.speech = None
            self.speech_t = 0.0
            self.target_x = None
            self.holds_candle = False
            self.action = None

        def say(self, symbol: str, duration: float = 3.0) -> None:
            self.speech = symbol
            self.speech_t = duration

    class _World:
        def __init__(self, seed: int = 3) -> None:
            self.terrain = _Terrain()
            self.props = [_Prop("tree", 500.0), _Prop("rock", 640.0),
                          _Prop("water", 900.0)]
            self.structures = _Structures([_Struct(560.0)])
            self.events = _Events()
            self.agents = [_Agent(1, 520.0), _Agent(2, 580.0)]
            self.pyrng = random.Random(seed)
            self.rng = random.Random(seed ^ 7)
            self.world_time = 0.0
            self.is_night = True

    print(f"registry: {len(REGISTRY)} vignettes")
    assert REGISTRY, "load_all() left the registry empty"
    assert all(MIN_DUR <= v.dur[0] <= v.dur[1] <= MAX_DUR for v in all_vignettes())
    assert all(v.motion in MOTIONS for v in all_vignettes())
    assert all(0.0 <= v.drift <= MAX_DRIFT for v in all_vignettes())

    # --- registry behaviour ------------------------------------------------
    dup = all_vignettes()[0]
    register(dup)                                   # identical: idempotent
    try:
        register(Vignette(dup.key, "something else", "idle", (2.0, 3.0)))
    except ValueError as exc:
        print("duplicate rejected:", exc)
    else:                                           # pragma: no cover
        raise AssertionError("duplicate key was not rejected")
    try:
        register("not a vignette")                  # type: ignore[arg-type]
    except TypeError:
        pass
    else:                                           # pragma: no cover
        raise AssertionError("non-Vignette was not rejected")

    # --- normalisation -----------------------------------------------------
    weird = Vignette("  WEIRD_Key ", "  ", "", (99.0, -3.0), -2.0,
                     tags=("Night", " social "), motion="moonwalk", drift=999.0)
    assert weird.key == "weird_key" and weird.pose == "idle", weird
    assert weird.dur == (MIN_DUR, MAX_DUR) and weird.weight == 0.0, weird
    assert weird.tags == ("night", "social") and weird.motion == "still", weird
    assert weird.drift == MAX_DRIFT and weird.label == "weird key", weird
    assert Vignette("x", "y", "idle", ("bad", None)).dur == (MIN_DUR, MAX_DUR)

    # --- context -----------------------------------------------------------
    w = _World()
    a = w.agents[0]
    ctx = context_tags(a, w)
    print("context:", sorted(ctx))
    assert "night" in ctx and "raining" in ctx and "storm" in ctx, ctx
    assert "near_fire" in ctx and "social" in ctx and "gatherer" in ctx, ctx
    assert ctx <= KNOWN_TAGS, ctx - KNOWN_TAGS
    assert context_tags(a, None) is not None            # a world-less world
    assert context_tags(None, w) is not None
    assert isinstance(by_tag("night"), list)

    # --- 500 random vignettes, start to finish -----------------------------
    rng = random.Random(1234)
    ran = 0
    skipped = 0
    poses: set[str] = set()
    for i in range(500):
        agent = w.agents[i % len(w.agents)]
        agent.x = rng.uniform(60.0, RENDER_W - 60.0)
        agent.y = w.terrain.ground_y(agent.x)
        agent.fatigue = rng.random()
        agent.hunger = rng.random()
        agent.warmth = rng.random()
        agent.morale = rng.random()
        agent.role = rng.choice(("gatherer", "builder", "elder", "child", "lookout"))
        agent.holds_candle = rng.random() < 0.2
        w.is_night = rng.random() < 0.5
        w.events.rain = rng.random()
        w.events.snow = rng.random()

        act = make_vignette_action(agent, w, rng)
        if act is None:
            skipped += 1
            recent_of(agent).clear()
            continue

        x0 = agent.x
        agent.action = act
        for step in range(2000):                    # 2000 * 1/30 s ~= 66 s
            act.update(agent, w, 1.0 / 30.0)
            if step == 5:                           # mid-vignette save/load
                act = VignetteAction.from_dict(act.to_dict())
                agent.action = act
            if act.finished:
                break
        assert act.done and not act.failed, (act, act.data)
        assert act.t <= MAX_DUR + 1.5, (act.t, act.data)
        assert abs(agent.x - x0) <= MAX_DRIFT + 0.5, (agent.x - x0, act.data)
        assert 0.0 <= agent.x <= RENDER_W
        poses.add(act.pose)
        ran += 1

    print(f"ran {ran} vignettes to completion, {skipped} skipped for lack of fit")
    print("poses exercised:", sorted(poses))
    assert ran >= 400, ran

    # --- recency: never the same twice, never three in a row ---------------
    solo = _Agent(9, 400.0)
    seen: list[str] = []
    for _ in range(40):
        v = pick(solo, w, rng)
        if v is None:
            break
        seen.append(v.key)
    for i in range(2, len(seen)):
        assert seen[i] not in seen[i - 2:i], (i, seen[i - 3:i + 1])
    assert len(recent_of(solo)) <= RECENT_MEMORY
    print("recency ok over", len(seen), "picks")

    # --- persistence -------------------------------------------------------
    live = make_vignette_action(w.agents[0], w, rng)
    assert live is not None
    live.update(w.agents[0], w, 0.5)
    blob = live.to_dict()
    assert VignetteAction.from_dict(blob).to_dict()["data"]["key"] == blob["data"]["key"]
    assert VignetteAction.from_dict(None).finished
    assert VignetteAction.from_dict({"data": {"key": "no_such_vignette"}}).done
    assert vignette_action_from_dict({"kind": "Wander"}) is None
    assert vignette_action_from_dict(blob) is not None

    # --- a hostile world must not raise ------------------------------------
    class _Hostile:
        @property
        def terrain(self) -> Any:
            raise RuntimeError("boom")

        @property
        def props(self) -> Any:
            raise RuntimeError("boom")

        @property
        def events(self) -> Any:
            raise RuntimeError("boom")

    hostile = _Hostile()
    print("hostile context:", sorted(context_tags(_Agent(1, 100.0), hostile)))
    bad = VignetteAction()
    for _ in range(400):
        bad.update(object(), hostile, 1.0 / 30.0)
    assert bad.finished, bad
    print("smoke test passed")
