"""Concrete stickman behaviours as resumable state machines.

One :class:`Action` class, one handler function per ``kind``. Keeping it a
single dataclass (rather than a subclass per behaviour) is what makes
``to_dict``/``from_dict`` round-tripping trivial: a save taken mid-build
reloads mid-build, because everything the machine needs lives in
``phase``/``t``/``data`` plus the target structure's own persisted progress.

No pygame in here. The only thing an action tells the renderer is ``pose``.

Everything an action touches on ``world`` is reached through the small
accessor layer at the top of this module, so a missing or renamed subsystem
degrades to a sensible default instead of raising inside the per-frame path.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..constants import (
    CLIMB_SPEED,
    DAY_LENGTH_SEC,
    MAX_SLOPE_WALK,
    RENDER_H,
    RENDER_W,
    RES_COOKED,
    RES_FIBRE,
    RES_FOOD,
    RES_STONE,
    RES_WOOD,
    WALK_SPEED,
)
from .entities import GROUND_SNAP

#: Ground drop, in px, that counts as a ledge rather than a slope. Roughly a
#: body height: below this a stickman steps down, above it they would have to
#: jump, and only a panicking one does.
CLIFF_DROP = 46.0

#: How far ahead to probe the ground for a ledge, in px. Must be several
#: times the per-tick step or a cliff edge is invisible until you are over it.
CLIFF_LOOKAHEAD = 14.0
from .structures import Structure, StructureRegistry

log = logging.getLogger(__name__)

__all__ = [
    "Action",
    "ACTION_KINDS",
    "POSES",
    "make_action",
    "CARRY_CAP",
    # accessor layer, reused by behavior.py
    "world_now", "is_night", "rng_of", "ground_y", "slope_at",
    "stock_qty", "stock_add", "stock_take", "stockpile_of",
    "agents_of", "agent_by_id", "structures_of", "props_of", "find_prop",
    "prop_alive", "hazards_of", "chronicle", "emit_speech",
    "celebrations_of", "push_celebration", "colony_center",
]

# ------------------------------------------------------------------ tuning --
CARRY_CAP = 8               # units a stickman can shoulder at once
REACH = 14.0                # how close counts as "there"
FIRE_REACH = 26.0
TALK_REACH = 30.0
EAT_TIME = 2.6
COOK_TIME = 6.0
CONVERSE_TIME = 4.0
CELEBRATE_TIME = 6.0
MOURN_TIME = 5.0
PLANT_TIME = 2.5
CLIMB_TIME = 2.5
CHOP_YIELD = 6
STONE_YIELD = 5
BERRY_YIELD = 4
SPEECH_SYMBOLS = ("?", "!", "~", "*", "+", "o", "^", "#")

POSES: tuple[str, ...] = (
    "idle", "walk", "run", "carry", "chop", "mine", "forage", "build",
    "eat", "sleep", "warm", "cook", "plant", "talk", "dance", "mourn",
    "flee", "climb", "lookout", "panic", "haul",
)

ACTION_KINDS: tuple[str, ...] = (
    "Wander", "GatherWood", "GatherStone", "ForageBerries", "HaulToStockpile",
    "BuildStructure", "RepairStructure", "Eat", "Sleep", "WarmAtFire",
    "CookFood", "PlantSapling", "Converse", "Celebrate", "Mourn", "FleeFrom",
    "ClimbTo", "Lookout", "FollowParent", "Panic",
)

_TREE_KINDS = ("tree", "pine", "oak", "deadtree")
_ROCK_KINDS = ("rock", "boulder", "stone", "outcrop")
_BUSH_KINDS = ("bush", "berry", "berrybush", "shrub")

_FALLBACK_RNG = random.Random(0xB16B00B5)


# ===========================================================================
#  World accessor layer - every one of these tolerates a missing subsystem
# ===========================================================================
def _clamp(v: float, lo: float, hi: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return lo
    if math.isnan(f):
        return lo
    return lo if f < lo else (hi if f > hi else f)


def _clamp01(v: float) -> float:
    return _clamp(v, 0.0, 1.0)


def _clamp_x(x: float) -> float:
    return _clamp(x, 4.0, float(RENDER_W - 4))


def rng_of(world: Any) -> random.Random:
    """The world's RNG so replays stay deterministic; a module RNG otherwise.

    World carries two generators: `rng` is a numpy Generator (for array work)
    and `pyrng` is a stdlib Random. Callers here want the stdlib surface -
    randrange/choice/randint - which numpy's Generator does *not* provide.
    Duck-typing on .random/.uniform alone matches numpy and then blows up later
    at the first .randrange, so prefer pyrng and demand the full surface.
    """
    pr = getattr(world, "pyrng", None)
    if isinstance(pr, random.Random):
        return pr
    r = getattr(world, "rng", None)
    if isinstance(r, random.Random):
        return r
    if r is not None and all(hasattr(r, n) for n in
                             ("random", "uniform", "randrange", "choice")):
        return r  # type: ignore[return-value]
    return _FALLBACK_RNG


def world_now(world: Any) -> float:
    for name in ("world_time", "time_s", "elapsed", "t"):
        v = getattr(world, name, None)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def day_phase(world: Any) -> float:
    """0..1 through the day. 0 == dawn, 0.5 == dusk."""
    v = getattr(world, "day_phase", None)
    if isinstance(v, (int, float)):
        return float(v) % 1.0
    if callable(v):
        try:
            return float(v()) % 1.0
        except Exception:
            pass
    day = float(getattr(world, "day_length", DAY_LENGTH_SEC) or DAY_LENGTH_SEC)
    if day <= 0.0:
        day = float(DAY_LENGTH_SEC)
    return (world_now(world) % day) / day


def is_night(world: Any) -> bool:
    v = getattr(world, "is_night", None)
    if isinstance(v, bool):
        return v
    if callable(v):
        try:
            return bool(v())
        except Exception:
            pass
    return day_phase(world) >= 0.5


def ground_y(world: Any, x: float) -> float:
    t = getattr(world, "terrain", None)
    if t is not None:
        fn = getattr(t, "ground_y", None)
        if callable(fn):
            try:
                y = float(fn(_clamp_x(x)))
                if math.isfinite(y):
                    return y
            except Exception:
                pass
    return RENDER_H * 0.75


def slope_at(world: Any, x: float) -> float:
    t = getattr(world, "terrain", None)
    if t is not None:
        fn = getattr(t, "slope", None)
        if callable(fn):
            try:
                s = float(fn(_clamp_x(x)))
                if math.isfinite(s):
                    return s
            except Exception:
                pass
    return 0.0


# ------------------------------------------------------------- stockpile ----
def stockpile_of(world: Any) -> Any:
    sp = getattr(world, "stockpile", None)
    if sp is None:
        sp = {}
        try:
            setattr(world, "stockpile", sp)
        except Exception:
            pass
    return sp


def stock_qty(world: Any, res: str) -> int:
    sp = stockpile_of(world)
    try:
        if isinstance(sp, dict):
            return max(0, int(sp.get(res, 0)))
        fn = getattr(sp, "get", None) or getattr(sp, "qty", None)
        if callable(fn):
            return max(0, int(fn(res) or 0))
    except Exception:
        pass
    return 0


def stock_add(world: Any, res: str, qty: int) -> int:
    if qty <= 0 or not res:
        return 0
    sp = stockpile_of(world)
    try:
        if isinstance(sp, dict):
            sp[res] = int(sp.get(res, 0)) + int(qty)
            return int(qty)
        for name in ("add", "deposit", "put"):
            fn = getattr(sp, name, None)
            if callable(fn):
                fn(res, int(qty))
                return int(qty)
    except Exception:
        log.debug("stock_add failed", exc_info=True)
    return 0


def stock_take(world: Any, res: str, qty: int) -> int:
    """Remove up to `qty`. Returns how much was actually taken."""
    if qty <= 0 or not res:
        return 0
    sp = stockpile_of(world)
    try:
        if isinstance(sp, dict):
            have = int(sp.get(res, 0))
            take = min(have, int(qty))
            if take > 0:
                sp[res] = have - take
            return max(0, take)
        for name in ("take", "withdraw", "remove"):
            fn = getattr(sp, name, None)
            if callable(fn):
                return max(0, int(fn(res, int(qty)) or 0))
    except Exception:
        log.debug("stock_take failed", exc_info=True)
    return 0


def food_in_store(world: Any) -> tuple[str | None, int]:
    """Best available food resource and its quantity - cooked wins."""
    ck = stock_qty(world, RES_COOKED)
    if ck > 0:
        return RES_COOKED, ck
    raw = stock_qty(world, RES_FOOD)
    if raw > 0:
        return RES_FOOD, raw
    return None, 0


# ----------------------------------------------------------------- agents ---
def agents_of(world: Any) -> list[Any]:
    for name in ("agents", "stickmen", "people"):
        v = getattr(world, name, None)
        if isinstance(v, (list, tuple)):
            return list(v)
        if v is not None and hasattr(v, "__iter__"):
            try:
                return list(v)
            except Exception:
                continue
    return []


def alive_agents(world: Any) -> list[Any]:
    return [a for a in agents_of(world) if getattr(a, "alive", True)]


def agent_by_id(world: Any, aid: Any) -> Any | None:
    if aid is None:
        return None
    fn = getattr(world, "agent_by_id", None)
    if callable(fn):
        try:
            return fn(aid)
        except Exception:
            pass
    try:
        aid_i = int(aid)
    except (TypeError, ValueError):
        return None
    for a in agents_of(world):
        if getattr(a, "id", None) == aid_i:
            return a
    return None


# ------------------------------------------------------------- structures ---
def structures_of(world: Any) -> StructureRegistry | None:
    reg = getattr(world, "structures", None)
    if isinstance(reg, StructureRegistry):
        return reg
    if reg is not None and hasattr(reg, "nearest") and hasattr(reg, "get"):
        return reg  # type: ignore[return-value]
    return None


def structure_by_id(world: Any, sid: Any) -> Structure | None:
    reg = structures_of(world)
    if reg is None:
        return None
    try:
        s = reg.get(sid)
    except Exception:
        return None
    return s if isinstance(s, Structure) else s


def nearest_structure(
    world: Any,
    kind: str | None,
    x: float,
    *,
    built_only: bool = True,
    predicate: Callable[[Any], bool] | None = None,
) -> Structure | None:
    reg = structures_of(world)
    if reg is None:
        return None
    try:
        return reg.nearest(kind, x, built_only=built_only, predicate=predicate)
    except TypeError:
        try:
            return reg.nearest(kind, x)
        except Exception:
            return None
    except Exception:
        return None


def colony_center(world: Any) -> float:
    reg = structures_of(world)
    if reg is not None:
        try:
            c = reg.colony_center()
            if isinstance(c, (int, float)) and math.isfinite(float(c)):
                return float(c)
        except Exception:
            pass
    ags = alive_agents(world)
    if ags:
        return float(sum(float(getattr(a, "x", 0.0)) for a in ags) / len(ags))
    return RENDER_W * 0.5


# ------------------------------------------------------------------ props ---
def props_of(world: Any) -> list[Any]:
    for name in ("props", "prop_registry", "scenery"):
        v = getattr(world, name, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            return list(v)
        for meth in ("all", "items", "values"):
            fn = getattr(v, meth, None)
            if callable(fn):
                try:
                    out = list(fn())
                    if out and isinstance(out[0], tuple):
                        out = [o[1] for o in out]
                    return out
                except Exception:
                    continue
        if hasattr(v, "__iter__"):
            try:
                return list(v)
            except Exception:
                continue
    return []


def prop_alive(prop: Any) -> bool:
    if prop is None:
        return False
    if getattr(prop, "removed", False) or getattr(prop, "dead", False):
        return False
    if not getattr(prop, "alive", True):
        return False
    if getattr(prop, "depleted", False):
        return False
    hp = getattr(prop, "hp", None)
    if isinstance(hp, (int, float)) and hp <= 0:
        return False
    return True


def _prop_kind(prop: Any) -> str:
    k = getattr(prop, "kind", None) or getattr(prop, "type", None) or ""
    return str(k).lower()


def find_prop(
    world: Any,
    kinds: Iterable[str],
    x: float,
    *,
    max_dist: float = 720.0,
    exclude: Iterable[Any] = (),
) -> Any | None:
    """Nearest living prop whose kind is in `kinds`."""
    want = tuple(k.lower() for k in kinds)
    skip = {id(e) for e in exclude}
    best = None
    best_d = float("inf")
    for p in props_of(world):
        if id(p) in skip or not prop_alive(p):
            continue
        if _prop_kind(p) not in want:
            continue
        try:
            d = abs(float(getattr(p, "x", 0.0)) - float(x))
        except (TypeError, ValueError):
            continue
        if d < best_d:
            best_d = d
            best = p
    if best is not None and best_d <= max_dist:
        return best
    return None


def prop_by_id(world: Any, pid: Any) -> Any | None:
    if pid is None:
        return None
    src = getattr(world, "props", None)
    fn = getattr(src, "get", None)
    if callable(fn):
        try:
            p = fn(pid)
            if p is not None:
                return p
        except Exception:
            pass
    for p in props_of(world):
        if getattr(p, "id", None) == pid:
            return p
    return None


def _kill_prop(world: Any, prop: Any) -> None:
    src = getattr(world, "props", None)
    for name in ("remove", "kill", "destroy"):
        fn = getattr(src, name, None)
        if callable(fn):
            try:
                fn(prop)
                return
            except Exception:
                break
    for attr, val in (("alive", False), ("removed", True), ("depleted", True)):
        if hasattr(prop, attr):
            try:
                setattr(prop, attr, val)
            except Exception:
                pass
    if not hasattr(prop, "alive") and not hasattr(prop, "removed"):
        try:
            prop.alive = False
        except Exception:
            pass


def hit_prop(world: Any, prop: Any, amount: float, default_yield: int) -> int:
    """Deal one work-hit to a prop. Returns resource units produced (often 0)."""
    if prop is None:
        return 0
    for name in ("chop", "mine", "harvest", "hit"):
        fn = getattr(prop, name, None)
        if callable(fn):
            try:
                out = fn(amount)
            except TypeError:
                try:
                    out = fn()
                except Exception:
                    return 0
            except Exception:
                return 0
            if isinstance(out, tuple) and out:
                out = out[0]
            if isinstance(out, bool):
                return default_yield if out else 0
            if isinstance(out, (int, float)):
                return max(0, int(out))
            return 0
    for attr in ("hp", "hits", "integrity", "amount", "qty"):
        if hasattr(prop, attr):
            try:
                v = float(getattr(prop, attr)) - float(amount)
            except (TypeError, ValueError):
                break
            try:
                setattr(prop, attr, max(0.0, v))
            except Exception:
                break
            if v <= 0.0:
                _kill_prop(world, prop)
                return default_yield
            return 0
    _kill_prop(world, prop)
    return default_yield


def spawn_sapling(world: Any, x: float) -> bool:
    """Ask props.py to plant a sapling at `x`. Queues it if there is no hook."""
    y = ground_y(world, x)
    src = getattr(world, "props", None)
    for name in ("plant_sapling", "add_sapling", "spawn_sapling"):
        fn = getattr(src, name, None) or getattr(world, name, None)
        if callable(fn):
            try:
                fn(float(x))
                return True
            except TypeError:
                try:
                    fn(float(x), float(y))
                    return True
                except Exception:
                    pass
            except Exception:
                pass
    for name in ("spawn", "create", "add"):
        fn = getattr(src, name, None)
        if callable(fn):
            try:
                fn("sapling", float(x), float(y))
                return True
            except Exception:
                continue
    q = getattr(world, "pending_props", None)
    if not isinstance(q, list):
        q = []
        try:
            setattr(world, "pending_props", q)
        except Exception:
            return False
    q.append({"kind": "sapling", "x": float(x), "y": float(y)})
    return True


# ----------------------------------------------------------------- hazards --
def hazards_of(world: Any) -> list[dict[str, Any]]:
    """Normalised danger zones: {kind, x, y, radius, water_y?}."""
    out: list[dict[str, Any]] = []
    raw: list[Any] = []
    for name in ("hazards", "danger_zones"):
        v = getattr(world, name, None)
        if callable(v):
            try:
                v = v()
            except Exception:
                v = None
        if isinstance(v, (list, tuple)):
            raw.extend(v)
    ev = getattr(world, "events", None)
    if ev is not None:
        v = getattr(ev, "hazards", None)
        if callable(v):
            try:
                v = v()
            except Exception:
                v = None
        if isinstance(v, (list, tuple)):
            raw.extend(v)
    for h in raw:
        d = _normalise_hazard(h)
        if d is not None:
            out.append(d)
    # Burning structures are hazards even if events.py never says so.
    reg = structures_of(world)
    if reg is not None:
        try:
            for s in reg:
                if getattr(s, "is_burning", False) and not getattr(s, "is_ruined", False):
                    out.append({
                        "kind": "fire", "x": float(s.x), "y": float(s.y),
                        "radius": 62.0,
                    })
        except Exception:
            pass
    return out


def _normalise_hazard(h: Any) -> dict[str, Any] | None:
    try:
        if isinstance(h, dict):
            get = h.get
        else:
            def get(k, default=None):  # type: ignore[misc]
                return getattr(h, k, default)
        kind = str(get("kind", None) or get("type", None) or "danger")
        water_y = get("water_y", None)
        x = get("x", None)
        radius = get("radius", None) or get("r", None)
        if x is None and water_y is None:
            return None
        d: dict[str, Any] = {
            "kind": kind,
            "x": float(x) if isinstance(x, (int, float)) else RENDER_W * 0.5,
            "y": float(get("y", RENDER_H * 0.75) or RENDER_H * 0.75),
            "radius": float(radius) if isinstance(radius, (int, float)) else 70.0,
        }
        if isinstance(water_y, (int, float)):
            d["water_y"] = float(water_y)
            d["kind"] = kind or "flood"
        return d
    except Exception:
        return None


# ------------------------------------------------------------ world events --
def chronicle(world: Any, text: str) -> None:
    if not text:
        return
    try:
        fn = getattr(world, "chronicle", None)
        if callable(fn):
            fn(text)
            return
        if fn is not None:
            add = getattr(fn, "add", None) or getattr(fn, "append", None)
            if callable(add):
                add(text)
                return
        fn = getattr(world, "log_event", None)
        if callable(fn):
            fn(text)
    except Exception:
        log.debug("chronicle failed: %s", text, exc_info=True)


def emit_speech(world: Any, agent: Any, symbol: str, ttl: float = 2.2) -> None:
    """Push a symbol speech bubble for the renderer to pick up."""
    try:
        fn = getattr(world, "emit_speech", None)
        if callable(fn):
            fn(agent, symbol)
            return
        q = getattr(world, "speech", None)
        if not isinstance(q, list):
            q = []
            setattr(world, "speech", q)
        q.append({
            "agent": int(getattr(agent, "id", 0)),
            "x": float(getattr(agent, "x", 0.0)),
            "y": float(getattr(agent, "y", 0.0)) - 26.0,
            "symbol": str(symbol),
            "t": 0.0,
            "ttl": float(ttl),
            "color": tuple(getattr(agent, "color", (230, 230, 230))),
        })
        if len(q) > 64:
            del q[: len(q) - 64]
    except Exception:
        log.debug("emit_speech failed", exc_info=True)


def celebrations_of(world: Any) -> list[dict[str, Any]]:
    q = getattr(world, "celebrations", None)
    return q if isinstance(q, list) else []


def push_celebration(world: Any, x: float, y: float, kind: str) -> None:
    try:
        q = getattr(world, "celebrations", None)
        if not isinstance(q, list):
            q = []
            setattr(world, "celebrations", q)
        q.append({"x": float(x), "y": float(y), "kind": str(kind),
                  "t": world_now(world)})
        if len(q) > 8:
            del q[: len(q) - 8]
    except Exception:
        log.debug("push_celebration failed", exc_info=True)


# ===========================================================================
#  Movement primitives
# ===========================================================================
def _face(agent: Any, d: float) -> None:
    if d > 0.01:
        agent.facing = 1
    elif d < -0.01:
        agent.facing = -1



def _plant(agent: Any, gy: float) -> None:
    """Put the agent on the ground and make the rest of its state agree.

    Writing y alone is not enough. entities.apply_physics decides an agent is
    airborne from on_ground/vy, so pinning y here while leaving vy alone let
    gravity accumulate against a body that never moved: measured vy reaching
    780 px/s over 0.87s of "falling" that covered 0.3px. The landing check then
    read that as a fatal impact, which is why every agent died of a fall while
    standing still on flat ground. Position and state must be asserted together.
    """
    agent.y = float(gy)
    agent.vy = 0.0
    agent.on_ground = True
    agent.fall_t = 0.0


def step_toward(
    agent: Any,
    world: Any,
    tx: float,
    dt: float,
    *,
    speed: float = WALK_SPEED,
    arrive: float = REACH,
) -> float:
    """Walk/climb one tick toward `tx`. Returns remaining |dx| afterwards."""
    try:
        tx = _clamp_x(tx)
        dx = tx - float(agent.x)
        dist = abs(dx)
        if dist <= arrive:
            agent.vx = 0.0
            return dist
        d = 1.0 if dx > 0.0 else -1.0
        _face(agent, d)
        sl = slope_at(world, agent.x)
        climbing = abs(sl) > MAX_SLOPE_WALK
        spd = CLIMB_SPEED if climbing else float(speed)
        if not climbing and sl * d < 0.0:
            spd *= 0.78          # uphill is slower
        elif not climbing and sl * d > 0.0:
            spd *= 1.08          # downhill is quicker
        step = spd * max(0.0, dt)
        if step > dist:
            step = dist
        nx = _clamp_x(float(agent.x) + d * step)

        # Look before stepping. Walking is terrain-following: if the ground
        # ahead drops by more than a body height this is a ledge, not a slope,
        # and a calm stickman does not stroll off it. Letting gravity work
        # without this check turned every cliff into a meat grinder - 30 to 42
        # deaths per 260s, every single one a fall, even in clear weather.
        # Panicking agents DO go over, which is what makes a stampede lethal.
        gy_now = ground_y(world, agent.x)
        gy_next = ground_y(world, nx)
        # Probe a body-width ahead, not one footstep. A step is ~1px, and even
        # a sheer cliff only drops 4-11px across 1px of travel, so a per-step
        # test never fires: agents walked straight off a 146px drop measuring
        # it 8px at a time. LOOKAHEAD is what makes the edge visible.
        gy_ahead = ground_y(world, _clamp_x(float(agent.x) + d * CLIFF_LOOKAHEAD))
        drop = max(gy_next - gy_now, gy_ahead - gy_now)
        panicking = float(getattr(agent, "panic", 0.0) or 0.0) > 0.0
        if drop > CLIFF_DROP and not panicking:
            agent.vx = 0.0
            _plant(agent, gy_now)
            agent._blocked_t = getattr(agent, "_blocked_t", 0.0) + dt
            return abs(tx - float(agent.x))
        agent._blocked_t = 0.0

        agent.x = nx
        agent.vx = d * spd
        # Stick to the surface ONLY when actually on it. Snapping y whenever
        # vy happened to be ~0 teleported falling agents straight to the
        # ground before entities.apply_physics could see them airborne, so
        # gravity never accumulated and nobody could ever fall to their death.
        # Actions own x; entities owns y whenever the agent is off the ground.
        if float(agent.y) >= gy_next - GROUND_SNAP:
            _plant(agent, gy_next)
        return abs(tx - float(agent.x))
    except Exception:
        log.debug("step_toward failed", exc_info=True)
        return 0.0


def _halt(agent: Any) -> None:
    try:
        agent.vx = 0.0
    except Exception:
        pass


def _uphill_dir(world: Any, x: float, probe: float = 60.0) -> float:
    """+1 or -1, whichever direction climbs (lower y == higher ground)."""
    left = ground_y(world, x - probe)
    right = ground_y(world, x + probe)
    if left < right - 1.0:
        return -1.0
    if right < left - 1.0:
        return 1.0
    return 1.0 if x < RENDER_W * 0.5 else -1.0


def _work_rate(agent: Any) -> float:
    role = str(getattr(agent, "role", "gatherer"))
    return {
        "builder": 1.35, "gatherer": 1.0, "elder": 0.8,
        "lookout": 0.9, "child": 0.5,
    }.get(role, 1.0)


def _need(agent: Any, name: str) -> float:
    return _clamp01(getattr(agent, name, 0.0))


def _adjust(agent: Any, name: str, delta: float) -> None:
    try:
        setattr(agent, name, _clamp01(float(getattr(agent, name, 0.0)) + delta))
    except Exception:
        pass


def _carry_add(agent: Any, action: "Action", res: str, qty: int) -> None:
    """Shoulder `qty` of `res`, stashing anything that does not fit the slot."""
    if qty <= 0:
        return
    cur = getattr(agent, "carrying", None)
    have = int(getattr(agent, "carry_qty", 0) or 0)
    if not cur or have <= 0:
        agent.carrying = res
        agent.carry_qty = int(qty)
    elif cur == res:
        agent.carry_qty = have + int(qty)
    else:
        stash = action.data.setdefault("stash", {})
        stash[res] = int(stash.get(res, 0)) + int(qty)


def _drop_all(agent: Any, action: "Action", world: Any) -> int:
    """Move everything the agent holds into the stockpile. Returns units moved."""
    moved = 0
    res = getattr(agent, "carrying", None)
    qty = int(getattr(agent, "carry_qty", 0) or 0)
    if res and qty > 0:
        moved += stock_add(world, res, qty)
    try:
        agent.carrying = None
        agent.carry_qty = 0
    except Exception:
        pass
    stash = action.data.get("stash")
    if isinstance(stash, dict):
        for r, q in list(stash.items()):
            try:
                moved += stock_add(world, str(r), int(q))
            except Exception:
                continue
        action.data["stash"] = {}
    return moved



def _load_is_useful(ag: Any, a: "Action", need: dict) -> bool:
    """True if anything the agent is carrying is something the site still wants.

    Guards the build state machine against walking a useless load to the site
    forever. Checks both the hand and any stashed load the action tracks.
    """
    if not need:
        return False
    carrying = getattr(ag, "carrying", None)
    if carrying in need and int(getattr(ag, "carry_qty", 0) or 0) > 0:
        return True
    stash = a.data.get("stash") or {}
    try:
        return any(need.get(res, 0) > 0 and int(qty) > 0
                   for res, qty in stash.items())
    except Exception:
        return False

def _has_load(agent: Any, action: "Action") -> bool:
    if int(getattr(agent, "carry_qty", 0) or 0) > 0 and getattr(agent, "carrying", None):
        return True
    stash = action.data.get("stash")
    return bool(isinstance(stash, dict) and any(int(v) > 0 for v in stash.values()))


def _deposit_step(action: "Action", agent: Any, world: Any, dt: float) -> bool:
    """Walk to the stockpile and unload. Returns True when finished."""
    if not _has_load(agent, action):
        return True
    sp = nearest_structure(world, "stockpile", agent.x, built_only=True)
    tx = sp.x if sp is not None else colony_center(world)
    action.pose = "carry"
    rem = step_toward(agent, world, tx, dt, speed=WALK_SPEED * 0.92)
    if rem <= REACH:
        _drop_all(agent, action, world)
        _halt(agent)
        return True
    if action.t > 90.0:            # never get stuck hauling forever
        _drop_all(agent, action, world)
        return True
    return False


# ===========================================================================
#  Action
# ===========================================================================
@dataclass
class Action:
    """A resumable behaviour. `kind` selects the handler, everything else is
    the machine's persisted state."""

    kind: str = "Wander"
    pose: str = "idle"
    target: Any = None            # structure id / agent id / prop id / x
    phase: str = "start"
    t: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    failed: bool = False

    # ------------------------------------------------------------------ run --
    def update(self, agent: Any, world: Any, dt: float) -> None:
        """Advance the machine by `dt`. Never raises - this is a frame path."""
        if self.done or self.failed:
            return
        try:
            self.t += float(dt)
        except (TypeError, ValueError):
            dt = 0.0
        handler = _HANDLERS.get(self.kind)
        if handler is None:
            log.debug("unknown action kind %r", self.kind)
            self.failed = True
            return
        try:
            handler(self, agent, world, float(dt))
        except Exception:
            log.warning("action %s/%s failed", self.kind, self.phase, exc_info=True)
            self.failed = True
            try:
                _halt(agent)
            except Exception:
                pass

    def abandon(self, agent: Any, world: Any) -> None:
        """Release anything held (hut slot, tower slot) before being replaced."""
        cleanup = _CLEANUP.get(self.kind)
        if cleanup is not None:
            try:
                cleanup(self, agent, world)
            except Exception:
                log.debug("cleanup for %s failed", self.kind, exc_info=True)
        self.done = True

    @property
    def finished(self) -> bool:
        return bool(self.done or self.failed)

    # ------------------------------------------------------------ serialise --
    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "pose": str(self.pose),
            "target": _safe(self.target),
            "phase": str(self.phase),
            "t": float(self.t),
            "data": _safe(self.data) or {},
            "done": bool(self.done),
            "failed": bool(self.failed),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Action":
        if not isinstance(d, dict):
            d = {}
        kind = d.get("kind")
        if not isinstance(kind, str) or kind not in _HANDLERS:
            kind = "Wander"
        pose = d.get("pose")
        data = d.get("data")
        try:
            t = float(d.get("t", 0.0))
            if not math.isfinite(t):
                t = 0.0
        except (TypeError, ValueError):
            t = 0.0
        return cls(
            kind=kind,
            pose=pose if isinstance(pose, str) and pose else "idle",
            target=d.get("target"),
            phase=str(d.get("phase", "start")) or "start",
            t=t,
            data=dict(data) if isinstance(data, dict) else {},
            done=bool(d.get("done", False)),
            failed=bool(d.get("failed", False)),
        )

    def __repr__(self) -> str:  # pragma: no cover - debug sugar
        flag = "done" if self.done else ("failed" if self.failed else self.phase)
        return f"<{self.kind}:{flag} t={self.t:.1f} target={self.target!r}>"


def _safe(v: Any, depth: int = 0) -> Any:
    if depth > 4:
        return None
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else 0.0
    if isinstance(v, (list, tuple)):
        return [_safe(x, depth + 1) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe(x, depth + 1) for k, x in v.items()}
    return None


def make_action(kind: str, **kw: Any) -> Action:
    """Build an Action, folding unknown kwargs into `data`."""
    a = Action(kind=kind)
    for field_name in ("pose", "target", "phase"):
        if field_name in kw:
            setattr(a, field_name, kw.pop(field_name))
    data = kw.pop("data", None)
    if isinstance(data, dict):
        a.data.update(data)
    a.data.update(kw)
    return a


# ===========================================================================
#  Handlers
# ===========================================================================
def _h_wander(a: Action, ag: Any, w: Any, dt: float) -> None:
    if a.phase == "start":
        rng = rng_of(w)
        span = rng.uniform(45.0, 230.0) * (1.0 if rng.random() < 0.5 else -1.0)
        a.data["tx"] = _clamp_x(float(ag.x) + span)
        a.data["pause"] = rng.uniform(1.2, 4.5)
        a.data["pt"] = 0.0
        a.phase = "walk"
    if a.phase == "walk":
        a.pose = "walk"
        rem = step_toward(ag, w, float(a.data.get("tx", ag.x)), dt,
                          speed=WALK_SPEED * 0.72)
        if rem <= REACH or a.t > 30.0:
            a.phase = "pause"
        return
    a.pose = "idle"
    _halt(ag)
    a.data["pt"] = float(a.data.get("pt", 0.0)) + dt
    if a.data["pt"] >= float(a.data.get("pause", 2.0)):
        a.done = True


_GATHER_SPEC: dict[str, dict[str, Any]] = {
    "GatherWood": {
        "kinds": _TREE_KINDS, "res": RES_WOOD, "pose": "chop",
        "hit_time": 0.55, "dmg": 1.0, "yield": CHOP_YIELD, "verb": "felled a tree",
    },
    "GatherStone": {
        "kinds": _ROCK_KINDS, "res": RES_STONE, "pose": "mine",
        "hit_time": 0.70, "dmg": 1.0, "yield": STONE_YIELD, "verb": "broke open a rock",
    },
    "ForageBerries": {
        "kinds": _BUSH_KINDS, "res": RES_FOOD, "pose": "forage",
        "hit_time": 0.90, "dmg": 1.0, "yield": BERRY_YIELD, "verb": "stripped a bush",
        # bushes are also where the colony's cordage comes from
        "bonus_res": RES_FIBRE, "bonus_ratio": 0.5,
    },
}


def _h_gather(a: Action, ag: Any, w: Any, dt: float) -> None:
    spec = _GATHER_SPEC.get(a.kind)
    if spec is None:
        a.failed = True
        return

    if a.phase == "start":
        p = find_prop(w, spec["kinds"], ag.x)
        if p is None:
            a.failed = True
            return
        a.target = getattr(p, "id", None)
        a.data["px"] = float(getattr(p, "x", ag.x))
        a.data["hits"] = 0
        a.data["hit_t"] = 0.0
        a.phase = "approach"

    if a.phase in ("approach", "work"):
        p = prop_by_id(w, a.target) if a.target is not None else None
        if p is None:
            # id lookup can fail if props.py has no id field - fall back to
            # position, and give up gracefully if the prop is really gone.
            p = find_prop(w, spec["kinds"], float(a.data.get("px", ag.x)), max_dist=40.0)
        if p is None or not prop_alive(p):
            if _has_load(ag, a):
                a.phase = "deliver"
            else:
                a.failed = True
            return
        a.data["px"] = float(getattr(p, "x", a.data.get("px", ag.x)))

    if a.phase == "approach":
        a.pose = "walk"
        rem = step_toward(ag, w, float(a.data["px"]), dt, arrive=REACH)
        if rem <= REACH:
            a.phase = "work"
            a.data["hit_t"] = 0.0
        elif a.t > 75.0:
            a.failed = True
        return

    if a.phase == "work":
        a.pose = str(spec["pose"])
        _halt(ag)
        _face(ag, float(a.data["px"]) - float(ag.x))
        a.data["hit_t"] = float(a.data.get("hit_t", 0.0)) + dt
        if a.data["hit_t"] < float(spec["hit_time"]) / max(0.3, _work_rate(ag)):
            return
        a.data["hit_t"] = 0.0
        a.data["hits"] = int(a.data.get("hits", 0)) + 1
        p = prop_by_id(w, a.target) or find_prop(
            w, spec["kinds"], float(a.data["px"]), max_dist=40.0)
        got = hit_prop(w, p, float(spec["dmg"]), int(spec["yield"]))
        if a.data["hits"] > 24 and got <= 0:
            # Something odd about this prop - take the yield and move on.
            _kill_prop(w, p)
            got = int(spec["yield"])
        if got > 0:
            _carry_add(ag, a, str(spec["res"]), got)
            bonus_res = spec.get("bonus_res")
            if bonus_res:
                bonus = max(1, int(got * float(spec.get("bonus_ratio", 0.5))))
                _carry_add(ag, a, str(bonus_res), bonus)
            _adjust(ag, "fatigue", 0.02)
            if not prop_alive(p):
                chronicle(w, f"{getattr(ag, 'name', 'Someone')} {spec['verb']}.")
            if int(getattr(ag, "carry_qty", 0) or 0) >= CARRY_CAP or not prop_alive(p):
                a.phase = "deliver"
        return

    if a.phase == "deliver":
        if _deposit_step(a, ag, w, dt):
            a.done = True
        return

    a.phase = "start"


def _h_haul(a: Action, ag: Any, w: Any, dt: float) -> None:
    if not _has_load(ag, a):
        a.done = True
        return
    a.phase = "deliver"
    if _deposit_step(a, ag, w, dt):
        a.done = True


def _h_build(a: Action, ag: Any, w: Any, dt: float) -> None:
    s = structure_by_id(w, a.target)
    if s is None and a.target is None:
        reg = structures_of(w)
        s = reg.find_incomplete(None, float(ag.x)) if reg is not None else None
        if s is not None:
            a.target = s.id
    if s is None or getattr(s, "is_ruined", False):
        a.failed = True
        return
    if getattr(s, "built", False):
        a.done = True
        return

    need = s.missing_for_stage()
    carrying = getattr(ag, "carrying", None)
    carry_qty = int(getattr(ag, "carry_qty", 0) or 0)

    if not need:
        a.phase = "work"
    elif carrying in need and carry_qty > 0:
        a.phase = "approach"
    elif _has_load(ag, a) and _load_is_useful(ag, a, need):
        a.phase = "approach"      # we are carrying something the site wants
    else:
        # Carrying a surplus the site has no use for. Routing to "approach"
        # here (as this did unconditionally) deadlocks: the builder walks to
        # the site, cannot deliver, and the haul action wins the next re-score,
        # so the pair alternate every tick and the agent paces on the spot.
        # Go and fetch what is actually needed instead; the surplus gets
        # dropped off on the way.
        a.phase = "fetch"

    if a.phase == "fetch":
        want_res, want_qty = None, 0
        for res, qty in sorted(need.items(), key=lambda kv: -kv[1]):
            have = stock_qty(w, res)
            if have > 0:
                want_res, want_qty = res, min(qty, have, CARRY_CAP)
                break
        if want_res is None:
            a.data["short"] = {k: int(v) for k, v in need.items()}
            a.failed = True
            return
        sp = nearest_structure(w, "stockpile", ag.x, built_only=True)
        tx = sp.x if sp is not None else colony_center(w)
        a.pose = "walk"
        rem = step_toward(ag, w, tx, dt)
        if rem <= REACH:
            got = stock_take(w, want_res, want_qty)
            if got <= 0:
                a.failed = True
                return
            _carry_add(ag, a, want_res, got)
            a.phase = "approach"
        elif a.t > 90.0:
            a.failed = True
        return

    if a.phase == "approach":
        a.pose = "carry"
        rem = step_toward(ag, w, float(s.x), dt, speed=WALK_SPEED * 0.92)
        if rem <= REACH * 1.6:
            res = getattr(ag, "carrying", None)
            qty = int(getattr(ag, "carry_qty", 0) or 0)
            if res and qty > 0:
                took = s.deliver(res, qty)
                left = qty - took
                ag.carry_qty = left
                if left <= 0:
                    ag.carrying = None
            stash = a.data.get("stash")
            if isinstance(stash, dict):
                for r, q in list(stash.items()):
                    took = s.deliver(str(r), int(q))
                    stash[r] = int(q) - took
                    if stash[r] <= 0:
                        stash.pop(r, None)
            a.phase = "work" if not s.missing_for_stage() else "fetch"
        elif a.t > 120.0:
            a.failed = True
        return

    if a.phase == "work":
        a.pose = "build"
        _halt(ag)
        _face(ag, float(s.x) - float(ag.x))
        _adjust(ag, "fatigue", 0.012 * dt)
        completed = s.advance(dt, _work_rate(ag))
        if completed:
            name = getattr(ag, "name", "Someone")
            chronicle(w, f"{name} finished the {s.kind}.")
            push_celebration(w, float(s.x), float(s.y), str(s.kind))
            _adjust(ag, "morale", 0.18)
            a.done = True
        elif s.missing_for_stage():
            a.phase = "fetch"
        return

    a.phase = "fetch"


def _h_repair(a: Action, ag: Any, w: Any, dt: float) -> None:
    s = structure_by_id(w, a.target)
    if s is None:
        reg = structures_of(w)
        if reg is not None:
            try:
                dmg = reg.damaged()
            except Exception:
                dmg = []
            if dmg:
                s = min(dmg, key=lambda st: abs(st.x - float(ag.x)))
                a.target = s.id
    if s is None or getattr(s, "is_ruined", False):
        a.failed = True
        return
    if s.hp >= s.max_hp:
        a.done = True
        return

    if a.phase in ("start", "approach"):
        a.phase = "approach"
        a.pose = "walk"
        rem = step_toward(ag, w, float(s.x), dt)
        if rem <= REACH * 1.6:
            a.phase = "work"
            a.data["mat_t"] = 0.0
        elif a.t > 75.0:
            a.failed = True
        return

    a.pose = "build"
    _halt(ag)
    _face(ag, float(s.x) - float(ag.x))
    a.data["mat_t"] = float(a.data.get("mat_t", 0.0)) + dt
    rate = 12.0 * _work_rate(ag)
    if a.data["mat_t"] >= 2.0:
        a.data["mat_t"] = 0.0
        primary = RES_STONE if s.kind in ("wall", "grave") else RES_WOOD
        if stock_take(w, primary, 1) <= 0:
            rate *= 0.35        # patching with what is to hand: slow
    s.repair(rate * dt)
    s.extinguish()
    _adjust(ag, "fatigue", 0.01 * dt)
    if s.hp >= s.max_hp:
        chronicle(w, f"{getattr(ag, 'name', 'Someone')} repaired the {s.kind}.")
        a.done = True


def _h_eat(a: Action, ag: Any, w: Any, dt: float) -> None:
    if a.phase == "start":
        held = getattr(ag, "carrying", None)
        if held in (RES_FOOD, RES_COOKED) and int(getattr(ag, "carry_qty", 0) or 0) > 0:
            ag.carry_qty = int(ag.carry_qty) - 1
            if ag.carry_qty <= 0:
                ag.carrying = None
            a.data["res"] = held
            a.phase = "eat"
            a.data["et"] = 0.0
        else:
            res, qty = food_in_store(w)
            if res is None or qty <= 0:
                a.failed = True
                return
            a.data["res"] = res
            a.phase = "approach"

    if a.phase == "approach":
        sp = nearest_structure(w, "stockpile", ag.x, built_only=True)
        tx = sp.x if sp is not None else colony_center(w)
        a.pose = "walk"
        rem = step_toward(ag, w, tx, dt, speed=WALK_SPEED * 1.05)
        if rem <= REACH:
            res, qty = food_in_store(w)
            if res is None or stock_take(w, res, 1) <= 0:
                a.failed = True
                return
            a.data["res"] = res
            a.data["et"] = 0.0
            a.phase = "eat"
        elif a.t > 75.0:
            a.failed = True
        return

    if a.phase == "eat":
        a.pose = "eat"
        _halt(ag)
        a.data["et"] = float(a.data.get("et", 0.0)) + dt
        if a.data["et"] >= EAT_TIME:
            cooked = a.data.get("res") == RES_COOKED
            _adjust(ag, "hunger", -0.78 if cooked else -0.52)
            _adjust(ag, "morale", 0.09 if cooked else 0.04)
            a.done = True
        return

    a.phase = "start"


def _pick_hut(w: Any, ag: Any) -> Structure | None:
    return nearest_structure(
        w, "hut", ag.x, built_only=True,
        predicate=lambda s: s.has_room() or int(getattr(ag, "id", -1)) in s.occupants,
    )


def _h_sleep(a: Action, ag: Any, w: Any, dt: float) -> None:
    hut = structure_by_id(w, a.target)
    if hut is None or getattr(hut, "is_ruined", False) or not getattr(hut, "built", False):
        if a.phase == "sleep":
            a.done = True           # bed collapsed under them; just get up
            return
        hut = _pick_hut(w, ag)
        if hut is None:
            a.failed = True
            return
        a.target = hut.id

    if a.phase in ("start", "approach"):
        a.phase = "approach"
        a.pose = "walk"
        rem = step_toward(ag, w, float(hut.x), dt)
        if rem <= REACH * 1.4:
            if not hut.enter(int(getattr(ag, "id", 0))):
                a.failed = True
                return
            a.data["in"] = 1
            a.phase = "sleep"
        elif a.t > 75.0:
            a.failed = True
        return

    a.pose = "sleep"
    _halt(ag)
    try:
        ag.x = float(ag.x) + (float(hut.x) - float(ag.x)) * min(1.0, dt * 4.0)
        ag.y = float(hut.y) - 2.0
    except Exception:
        pass
    _adjust(ag, "fatigue", -0.055 * dt)
    _adjust(ag, "warmth", -0.030 * dt)
    _adjust(ag, "hunger", 0.004 * dt)
    _adjust(ag, "morale", 0.006 * dt)
    fatigue = _need(ag, "fatigue")
    if fatigue <= 0.02 or (not is_night(w) and fatigue < 0.22) or a.t > 400.0:
        _c_sleep(a, ag, w)          # free the bed here, not just via abandon()
        a.done = True


def _c_sleep(a: Action, ag: Any, w: Any) -> None:
    hut = structure_by_id(w, a.target)
    if hut is not None:
        hut.leave(int(getattr(ag, "id", 0)))
    if a.data.get("in"):
        try:
            ag.y = ground_y(w, ag.x)
        except Exception:
            pass
    a.data["in"] = 0


def _h_warm(a: Action, ag: Any, w: Any, dt: float) -> None:
    fire = structure_by_id(w, a.target)
    if fire is None or fire.kind != "firepit" or getattr(fire, "is_ruined", False):
        fire = nearest_structure(w, "firepit", ag.x, built_only=True)
        if fire is None:
            a.failed = True
            return
        a.target = fire.id

    if a.phase in ("start", "approach"):
        a.phase = "approach"
        a.pose = "walk"
        rem = step_toward(ag, w, float(fire.x), dt, arrive=FIRE_REACH)
        if rem <= FIRE_REACH:
            a.phase = "warm"
        elif a.t > 75.0:
            a.failed = True
        return

    a.pose = "warm"
    _halt(ag)
    _face(ag, float(fire.x) - float(ag.x))
    if not fire.fire_active:
        wood = 0
        if getattr(ag, "carrying", None) == RES_WOOD:
            wood = min(2, int(getattr(ag, "carry_qty", 0) or 0))
            if wood:
                ag.carry_qty = int(ag.carry_qty) - wood
                if ag.carry_qty <= 0:
                    ag.carrying = None
        if wood <= 0:
            wood = stock_take(w, RES_WOOD, 2)
        if wood <= 0:
            a.failed = True
            return
        fire.stoke(wood)
        fire.light_fire(0.5)
        chronicle(w, f"{getattr(ag, 'name', 'Someone')} coaxed the fire back to life.")
    _adjust(ag, "warmth", -0.11 * dt)
    _adjust(ag, "morale", 0.012 * dt)
    _adjust(ag, "fatigue", -0.004 * dt)
    if _need(ag, "warmth") <= 0.04 or a.t > 60.0:
        a.done = True


def _h_cook(a: Action, ag: Any, w: Any, dt: float) -> None:
    fire = structure_by_id(w, a.target)
    if fire is None or getattr(fire, "is_ruined", False):
        fire = nearest_structure(w, "firepit", ag.x, built_only=True)
        if fire is None:
            a.failed = True
            return
        a.target = fire.id

    if a.phase == "start":
        if stock_qty(w, RES_FOOD) <= 0 and getattr(ag, "carrying", None) != RES_FOOD:
            a.failed = True
            return
        a.phase = "fetch"

    if a.phase == "fetch":
        if getattr(ag, "carrying", None) == RES_FOOD and int(getattr(ag, "carry_qty", 0) or 0) > 0:
            a.phase = "approach"
        else:
            sp = nearest_structure(w, "stockpile", ag.x, built_only=True)
            tx = sp.x if sp is not None else colony_center(w)
            a.pose = "walk"
            rem = step_toward(ag, w, tx, dt)
            if rem <= REACH:
                got = stock_take(w, RES_FOOD, 3)
                if got <= 0:
                    a.failed = True
                    return
                _carry_add(ag, a, RES_FOOD, got)
                a.phase = "approach"
            elif a.t > 90.0:
                a.failed = True
            return

    if a.phase == "approach":
        a.pose = "carry"
        rem = step_toward(ag, w, float(fire.x), dt, arrive=FIRE_REACH)
        if rem <= FIRE_REACH:
            a.phase = "cook"
            a.data["ct"] = 0.0
        elif a.t > 120.0:
            a.failed = True
        return

    if a.phase == "cook":
        a.pose = "cook"
        _halt(ag)
        _face(ag, float(fire.x) - float(ag.x))
        if not fire.fire_active:
            got = stock_take(w, RES_WOOD, 1)
            if got > 0:
                fire.stoke(got)
                fire.light_fire(0.4)
            else:
                a.phase = "deliver"
                return
        a.data["ct"] = float(a.data.get("ct", 0.0)) + dt
        _adjust(ag, "warmth", -0.03 * dt)
        if a.data["ct"] >= COOK_TIME:
            qty = int(getattr(ag, "carry_qty", 0) or 0)
            if getattr(ag, "carrying", None) == RES_FOOD and qty > 0:
                ag.carrying = RES_COOKED
                chronicle(w, f"{getattr(ag, 'name', 'Someone')} cooked a meal.")
            a.phase = "deliver"
        return

    if _deposit_step(a, ag, w, dt):
        a.done = True


def _h_plant(a: Action, ag: Any, w: Any, dt: float) -> None:
    if a.phase == "start":
        rng = rng_of(w)
        base = colony_center(w)
        tx = None
        for _ in range(12):
            cand = _clamp_x(base + rng.uniform(-380.0, 380.0))
            if abs(slope_at(w, cand)) > MAX_SLOPE_WALK:
                continue
            near = find_prop(w, _TREE_KINDS + ("sapling",), cand, max_dist=44.0)
            if near is not None:
                continue
            st = nearest_structure(w, None, cand, built_only=False)
            if st is not None and abs(float(st.x) - cand) < 34.0:
                continue
            tx = cand
            break
        if tx is None:
            a.failed = True
            return
        a.data["tx"] = float(tx)
        a.phase = "approach"

    if a.phase == "approach":
        a.pose = "walk"
        rem = step_toward(ag, w, float(a.data.get("tx", ag.x)), dt)
        if rem <= REACH:
            a.phase = "plant"
            a.data["pt"] = 0.0
        elif a.t > 75.0:
            a.failed = True
        return

    a.pose = "plant"
    _halt(ag)
    a.data["pt"] = float(a.data.get("pt", 0.0)) + dt
    if a.data["pt"] >= PLANT_TIME:
        if spawn_sapling(w, float(a.data.get("tx", ag.x))):
            chronicle(w, f"{getattr(ag, 'name', 'Someone')} planted a sapling.")
            _adjust(ag, "morale", 0.06)
            a.done = True
        else:
            a.failed = True


def _h_converse(a: Action, ag: Any, w: Any, dt: float) -> None:
    other = agent_by_id(w, a.target)
    if other is None or not getattr(other, "alive", True):
        other = _nearest_other(w, ag, 240.0)
        if other is None:
            a.done = True
            return
        a.target = getattr(other, "id", None)

    if a.phase in ("start", "approach"):
        a.phase = "approach"
        a.pose = "walk"
        gap = float(other.x) - float(ag.x)
        want = float(other.x) - (TALK_REACH * 0.6) * (1.0 if gap > 0 else -1.0)
        rem = step_toward(ag, w, want, dt)
        if abs(float(other.x) - float(ag.x)) <= TALK_REACH:
            a.phase = "talk"
            a.data["st"] = 0.0
            _pair_up(w, ag, other)
        elif a.t > 30.0:
            a.done = True
        return

    a.pose = "talk"
    _halt(ag)
    _face(ag, float(other.x) - float(ag.x))
    if abs(float(other.x) - float(ag.x)) > TALK_REACH * 2.2:
        a.done = True
        return
    a.data["st"] = float(a.data.get("st", 0.0)) + dt
    if a.data["st"] >= 1.15:
        a.data["st"] = 0.0
        rng = rng_of(w)
        emit_speech(w, ag, rng.choice(SPEECH_SYMBOLS))
    _adjust(ag, "morale", 0.030 * dt)
    _adjust(ag, "fatigue", -0.002 * dt)
    if a.t > CONVERSE_TIME + 1.0:
        a.done = True


def _nearest_other(w: Any, ag: Any, max_dist: float) -> Any | None:
    best, best_d = None, float("inf")
    my_id = getattr(ag, "id", None)
    for o in alive_agents(w):
        if getattr(o, "id", None) == my_id:
            continue
        act = getattr(o, "action", None)
        if act is not None and getattr(act, "kind", "") in ("Sleep", "Lookout", "Panic", "FleeFrom"):
            continue
        d = abs(float(getattr(o, "x", 0.0)) - float(ag.x))
        if d < best_d:
            best_d, best = d, o
    return best if best is not None and best_d <= max_dist else None


def _pair_up(w: Any, ag: Any, other: Any) -> None:
    """Give the partner a matching Converse so the pause is mutual."""
    try:
        cur = getattr(other, "action", None)
        if cur is not None and getattr(cur, "kind", "") not in ("Wander", "Converse"):
            return
        if cur is not None and getattr(cur, "kind", "") == "Converse":
            return
        if cur is not None:
            cur.abandon(other, w)
        other.action = make_action(
            "Converse", target=getattr(ag, "id", None), phase="talk", pose="talk")
    except Exception:
        log.debug("pair_up failed", exc_info=True)


def _h_celebrate(a: Action, ag: Any, w: Any, dt: float) -> None:
    if a.phase == "start":
        tx = a.data.get("tx")
        if tx is None:
            s = structure_by_id(w, a.target)
            if s is not None:
                tx = float(s.x)
            else:
                cel = celebrations_of(w)
                tx = float(cel[-1].get("x", ag.x)) if cel else float(ag.x)
        a.data["tx"] = _clamp_x(float(tx))
        a.phase = "approach"

    if a.phase == "approach":
        a.pose = "walk"
        rem = step_toward(ag, w, float(a.data["tx"]), dt, speed=WALK_SPEED * 1.15,
                          arrive=FIRE_REACH)
        if rem <= FIRE_REACH or a.t > 9.0:   # was 25s: a long failed walk reads as pacing
            a.phase = "dance"
            a.data["dt"] = 0.0
        return

    a.pose = "dance"
    _halt(ag)
    a.data["dt"] = float(a.data.get("dt", 0.0)) + dt
    if int(a.data["dt"] * 3.0) % 2 == 0:
        ag.facing = 1
    else:
        ag.facing = -1
    _adjust(ag, "morale", 0.055 * dt)
    _adjust(ag, "fatigue", 0.008 * dt)
    if a.data["dt"] >= CELEBRATE_TIME:
        a.done = True


def _h_mourn(a: Action, ag: Any, w: Any, dt: float) -> None:
    grave = structure_by_id(w, a.target)
    if grave is None or grave.kind != "grave":
        grave = nearest_structure(w, "grave", ag.x, built_only=False)
        if grave is None:
            a.failed = True
            return
        a.target = grave.id

    if a.phase in ("start", "approach"):
        a.phase = "approach"
        a.pose = "walk"
        rem = step_toward(ag, w, float(grave.x) + 10.0, dt, speed=WALK_SPEED * 0.7)
        if rem <= REACH * 1.5:
            a.phase = "bow"
            a.data["bt"] = 0.0
        elif a.t > 90.0:
            a.failed = True
        return

    a.pose = "mourn"
    _halt(ag)
    _face(ag, float(grave.x) - float(ag.x))
    a.data["bt"] = float(a.data.get("bt", 0.0)) + dt
    _adjust(ag, "morale", -0.020 * dt)
    if a.data["bt"] >= MOURN_TIME:
        _adjust(ag, "morale", 0.10)        # grief spent, they carry on
        try:
            mourners = grave.state.get("mourners")
            if not isinstance(mourners, list):
                mourners = []
                grave.state["mourners"] = mourners
            aid = int(getattr(ag, "id", 0))
            if aid not in mourners:
                mourners.append(aid)
            grave.state["mourn_done"] = float(grave.state.get("mourn_done", 0.0)) + 1.0
        except Exception:
            pass
        a.done = True


def _h_flee(a: Action, ag: Any, w: Any, dt: float) -> None:
    fx = a.data.get("fx")
    if not isinstance(fx, (int, float)):
        fx = float(a.target) if isinstance(a.target, (int, float)) else float(ag.x)
        a.data["fx"] = float(fx)
    radius = float(a.data.get("radius", 90.0) or 90.0)

    if a.phase == "start":
        if a.data.get("uphill"):
            d = _uphill_dir(w, float(ag.x))
        else:
            d = 1.0 if float(ag.x) >= float(fx) else -1.0
            edge = _clamp_x(float(ag.x) + d * (radius + 110.0))
            if abs(edge - float(ag.x)) < radius * 0.5:
                d = -d          # backed against a wall, break the other way
        a.data["dir"] = float(d)
        a.data["tx"] = _clamp_x(float(ag.x) + d * (radius + 130.0))
        a.phase = "run"

    a.pose = "flee"
    d = float(a.data.get("dir", 1.0))
    tx = float(a.data.get("tx", ag.x))
    rem = step_toward(ag, w, tx, dt, speed=WALK_SPEED * 2.1, arrive=6.0)
    _adjust(ag, "fatigue", 0.035 * dt)
    _adjust(ag, "morale", -0.012 * dt)
    away = abs(float(ag.x) - float(fx))
    if rem <= 6.0 and away < radius * 1.3:
        a.data["tx"] = _clamp_x(float(ag.x) + d * 150.0)
        return
    if away > radius * 1.3 and a.t > 1.0:
        a.done = True
    elif a.t > 22.0:
        a.done = True


def _h_climb(a: Action, ag: Any, w: Any, dt: float) -> None:
    ty = a.data.get("ty")
    if not isinstance(ty, (int, float)):
        ty = ground_y(w, ag.x) - 90.0
        a.data["ty"] = float(ty)
    if a.phase == "start":
        a.data["dir"] = _uphill_dir(w, float(ag.x))
        a.data["lastx"] = float(ag.x)
        a.data["stuck"] = 0.0
        a.phase = "climb"

    d = float(a.data.get("dir", 1.0))
    sl = slope_at(w, ag.x)
    a.pose = "climb" if abs(sl) > MAX_SLOPE_WALK else "walk"
    step_toward(ag, w, _clamp_x(float(ag.x) + d * 60.0), dt, arrive=1.0)
    _adjust(ag, "fatigue", 0.018 * dt)

    moved = abs(float(ag.x) - float(a.data.get("lastx", ag.x)))
    a.data["lastx"] = float(ag.x)
    a.data["stuck"] = 0.0 if moved > 0.05 else float(a.data.get("stuck", 0.0)) + dt
    if ground_y(w, ag.x) <= float(ty):
        a.done = True
    elif a.data["stuck"] > 2.0 or a.t > 30.0:
        a.done = True


def _h_lookout(a: Action, ag: Any, w: Any, dt: float) -> None:
    tower = structure_by_id(w, a.target)
    if tower is None or tower.kind != "watchtower" or getattr(tower, "is_ruined", False):
        if a.phase in ("ascend", "watch"):
            # The tower went out from under them - put them back on the ground
            # rather than leaving a stickman hanging in mid-air.
            _c_lookout(a, ag, w)
            a.done = True
            return
        tower = nearest_structure(
            w, "watchtower", ag.x, built_only=True,
            predicate=lambda s: s.has_room() or int(getattr(ag, "id", -1)) in s.occupants,
        )
        if tower is None:
            a.failed = True
            return
        a.target = tower.id

    if a.phase in ("start", "approach"):
        a.phase = "approach"
        a.pose = "walk"
        rem = step_toward(ag, w, float(tower.x), dt)
        if rem <= REACH:
            if not tower.enter(int(getattr(ag, "id", 0))):
                a.failed = True
                return
            a.data["in"] = 1
            a.data["y0"] = float(ag.y)
            a.data["at"] = 0.0
            a.data["dur"] = rng_of(w).uniform(70.0, 190.0)
            a.phase = "ascend"
        elif a.t > 90.0:
            a.failed = True
        return

    top = float(tower.top_y()) + 6.0

    if a.phase == "ascend":
        a.pose = "climb"
        _halt(ag)
        a.data["at"] = float(a.data.get("at", 0.0)) + dt
        k = _clamp01(a.data["at"] / CLIMB_TIME)
        y0 = float(a.data.get("y0", ag.y))
        try:
            ag.x = float(tower.x)
            ag.y = y0 + (top - y0) * k
        except Exception:
            pass
        _adjust(ag, "fatigue", 0.02 * dt)
        if k >= 1.0:
            a.phase = "watch"
            a.data["wt"] = 0.0
        return

    a.pose = "lookout"
    _halt(ag)
    try:
        ag.x = float(tower.x)
        ag.y = top
    except Exception:
        pass
    a.data["wt"] = float(a.data.get("wt", 0.0)) + dt
    if int(a.data["wt"] / 4.0) % 2 == 0:
        ag.facing = 1
    else:
        ag.facing = -1
    _adjust(ag, "fatigue", 0.004 * dt)
    _adjust(ag, "warmth", 0.006 * dt)
    haz = hazards_of(w)
    if haz:
        h = min(haz, key=lambda z: abs(float(z["x"]) - float(ag.x)))
        try:
            setattr(w, "alarm", {"x": float(h["x"]), "kind": str(h["kind"]),
                                 "t": world_now(w)})
        except Exception:
            pass
        emit_speech(w, ag, "!")
        chronicle(w, f"{getattr(ag, 'name', 'The lookout')} spotted danger.")
        _c_lookout(a, ag, w)
        a.done = True
        return
    if a.data["wt"] >= float(a.data.get("dur", 120.0)):
        _c_lookout(a, ag, w)
        a.done = True


def _c_lookout(a: Action, ag: Any, w: Any) -> None:
    tower = structure_by_id(w, a.target)
    if tower is not None:
        tower.leave(int(getattr(ag, "id", 0)))
    if a.data.get("in"):
        try:
            ag.y = ground_y(w, ag.x)
        except Exception:
            pass
    a.data["in"] = 0


def _h_follow(a: Action, ag: Any, w: Any, dt: float) -> None:
    parent = agent_by_id(w, a.target)
    if parent is None or not getattr(parent, "alive", True):
        parent = _nearest_adult(w, ag)
        if parent is None:
            a.failed = True
            return
        a.target = getattr(parent, "id", None)
    if a.phase == "start":
        a.data["dur"] = rng_of(w).uniform(22.0, 50.0)
        a.phase = "follow"

    gap = float(parent.x) - float(ag.x)
    if abs(gap) > 34.0:
        a.pose = "walk"
        step_toward(ag, w, float(parent.x) - 22.0 * (1.0 if gap > 0 else -1.0), dt,
                    speed=WALK_SPEED * 1.12)
    else:
        a.pose = "idle"
        _halt(ag)
        _face(ag, gap)
        _adjust(ag, "morale", 0.008 * dt)
    if a.t >= float(a.data.get("dur", 30.0)):
        a.done = True


def _nearest_adult(w: Any, ag: Any) -> Any | None:
    best, best_d = None, float("inf")
    my_id = getattr(ag, "id", None)
    for o in alive_agents(w):
        if getattr(o, "id", None) == my_id:
            continue
        if str(getattr(o, "role", "")) == "child":
            continue
        d = abs(float(getattr(o, "x", 0.0)) - float(ag.x))
        if d < best_d:
            best_d, best = d, o
    return best


def _h_panic(a: Action, ag: Any, w: Any, dt: float) -> None:
    if a.phase == "start":
        rng = rng_of(w)
        a.data["dir"] = 1.0 if rng.random() < 0.5 else -1.0
        a.data["dur"] = rng.uniform(3.0, 7.0)
        a.data["flip"] = 0.0
        a.phase = "run"
    a.pose = "panic"
    a.data["flip"] = float(a.data.get("flip", 0.0)) + dt
    if a.data["flip"] > 0.9:
        a.data["flip"] = 0.0
        a.data["dir"] = -float(a.data.get("dir", 1.0))
        emit_speech(w, ag, "!")
    d = float(a.data.get("dir", 1.0))
    step_toward(ag, w, _clamp_x(float(ag.x) + d * 90.0), dt,
                speed=WALK_SPEED * 1.8, arrive=2.0)
    _adjust(ag, "morale", -0.045 * dt)
    _adjust(ag, "fatigue", 0.030 * dt)
    if a.t >= float(a.data.get("dur", 4.0)):
        a.done = True


_HANDLERS: dict[str, Callable[[Action, Any, Any, float], None]] = {
    "Wander": _h_wander,
    "GatherWood": _h_gather,
    "GatherStone": _h_gather,
    "ForageBerries": _h_gather,
    "HaulToStockpile": _h_haul,
    "BuildStructure": _h_build,
    "RepairStructure": _h_repair,
    "Eat": _h_eat,
    "Sleep": _h_sleep,
    "WarmAtFire": _h_warm,
    "CookFood": _h_cook,
    "PlantSapling": _h_plant,
    "Converse": _h_converse,
    "Celebrate": _h_celebrate,
    "Mourn": _h_mourn,
    "FleeFrom": _h_flee,
    "ClimbTo": _h_climb,
    "Lookout": _h_lookout,
    "FollowParent": _h_follow,
    "Panic": _h_panic,
}

_CLEANUP: dict[str, Callable[[Action, Any, Any], None]] = {
    "Sleep": _c_sleep,
    "Lookout": _c_lookout,
}

# Sanity: every advertised kind must have a real handler.
assert set(ACTION_KINDS) == set(_HANDLERS), "ACTION_KINDS/_HANDLERS mismatch"


if __name__ == "__main__":  # pragma: no cover - headless smoke test
    from dataclasses import dataclass as _dc

    @_dc
    class _Prop:
        id: int
        kind: str
        x: float
        hp: float = 4.0
        alive: bool = True

    class _Terrain:
        def ground_y(self, x: float) -> float:
            return 600.0 + 40.0 * math.sin(float(x) / 180.0)

        def slope(self, x: float) -> float:
            return (40.0 / 180.0) * math.cos(float(x) / 180.0)

    class _Agent:
        def __init__(self, aid: int, x: float) -> None:
            self.id = aid
            self.name = f"Agent{aid}"
            self.color = (200, 200, 200)
            self.x = x
            self.y = 600.0
            self.vx = self.vy = 0.0
            self.facing = 1
            self.hunger = self.fatigue = self.warmth = 0.2
            self.morale = 0.6
            self.carrying = None
            self.carry_qty = 0
            self.role = "gatherer"
            self.generation = 0
            self.action = None
            self.alive = True
            self.anim_t = 0.0

    class _World:
        def __init__(self) -> None:
            self.terrain = _Terrain()
            self.structures = StructureRegistry()
            self.props = [_Prop(1, "tree", 500.0), _Prop(2, "rock", 700.0),
                          _Prop(3, "bush", 620.0)]
            self.stockpile: dict[str, int] = {}
            self.agents: list[Any] = []
            self.world_time = 0.0
            self.rng = random.Random(7)
            self.lines: list[str] = []

        def chronicle(self, text: str) -> None:
            self.lines.append(text)

    w = _World()
    a1 = _Agent(1, 400.0)
    w.agents.append(a1)
    w.structures.create("stockpile", 460.0, w.terrain.ground_y(460.0), built=True)

    act = make_action("GatherWood")
    for _ in range(4000):
        act.update(a1, w, 1 / 30)
        if act.finished:
            break
    print("gather:", act, "stock:", w.stockpile)

    site = w.structures.create("firepit", 480.0, w.terrain.ground_y(480.0))
    w.stockpile[RES_WOOD] = 20
    w.stockpile[RES_STONE] = 20
    b = make_action("BuildStructure", target=site.id)
    for i in range(6000):
        b.update(a1, w, 1 / 30)
        if i == 300:                      # mid-build save/load round trip
            b = Action.from_dict(b.to_dict())
        if b.finished:
            break
    print("build:", b, "built:", site.built, "chronicle:", w.lines)

    for kind in ACTION_KINDS:
        act = make_action(kind, ty=520.0, fx=400.0, radius=80.0)
        for _ in range(600):
            act.update(a1, w, 1 / 30)
            if act.finished:
                break
        rt = Action.from_dict(act.to_dict())
        assert rt.kind == act.kind
        print(f"  {kind:16s} -> {act.phase:9s} done={act.done} failed={act.failed}")
    print("junk action:", Action.from_dict({"kind": "Nope"}))
