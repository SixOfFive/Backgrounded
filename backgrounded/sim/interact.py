"""Player interaction with the world - the sim side of the left-mouse tools.

Pure sim, **no pygame**: the app translates mouse input into world coordinates
and calls these; render/ draws the toolbar and cursor. Keeping it here means the
whole thing is testable headless, which is how the grab/toss physics get
verified without a window.

Held objects are tracked by id in two *transient* sets on the World
(``held_agent_ids`` / ``held_prop_ids``) that are deliberately NOT persisted:
a save taken mid-drag then reloads with nothing held, so a crash or quit while
carrying a stickman can never leave one frozen in the air forever.
"""
from __future__ import annotations

import logging
from typing import Any

from ..constants import (
    FEED_AMOUNT, GRAB_RADIUS, RENDER_W, RES_FOOD, TOSS_SPEED,
    TOOL_FEED, TOOL_HAND, TOOL_LIGHTNING, TOOL_METEOR, TOOL_PLANT,
    TOOL_ROCK, TOOL_SPAWN,
)

log = logging.getLogger(__name__)


# ------------------------------------------------------------------- grab --
class Grab:
    """A handle on whatever the hand is currently holding."""

    __slots__ = ("kind", "id", "trail")

    def __init__(self, kind: str, oid: int) -> None:
        self.kind = kind          # "agent" | "prop"
        self.id = int(oid)
        self.trail: list[tuple[float, float, float]] = []   # (t, x, y) for toss velocity

    def to_dict(self) -> dict:
        return {"kind": self.kind, "id": self.id}


def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = ax - bx, ay - by
    return dx * dx + dy * dy


def grab(world: Any, wx: float, wy: float) -> Grab | None:
    """Pick up the nearest grabbable thing to (wx, wy), if any is close enough.

    Agents win ties with props at the same spot - you almost always mean the
    person, not the rock they are standing on.
    """
    best: Grab | None = None
    best_d = GRAB_RADIUS * GRAB_RADIUS

    try:
        for a in world.population.alive_agents():
            d = _dist2(a.x, a.y - 12.0, wx, wy)
            if d <= best_d:
                best_d = d
                best = Grab("agent", a.id)
    except Exception:
        pass

    if best is not None:
        _mark_held(world, best, True)
        # Whatever they were standing on, they are not standing on it now. A
        # lookout plucked off his tower keeps that platform otherwise, and the
        # physics would haul him back up to it the moment he was let go.
        obj = _resolve(world, best)
        if obj is not None and hasattr(obj, "perch_y"):
            obj.perch_y = None
        return best

    # No person nearby: try a prop. Skip water and graves - you cannot pocket a
    # pond, and disturbing the dead is a step too far even here.
    try:
        for p in world.props.all():
            if p.kind in ("water", "grave", "scorch"):
                continue
            d = _dist2(p.x, p.y - 8.0, wx, wy)
            if d <= best_d:
                best_d = d
                best = Grab("prop", p.id)
    except Exception:
        pass

    if best is not None:
        _mark_held(world, best, True)
    return best


def move_held(world: Any, g: Grab, wx: float, wy: float, t: float) -> None:
    """Drag the held object to the cursor and remember the path for toss speed."""
    obj = _resolve(world, g)
    if obj is None:
        return
    wx = max(2.0, min(float(RENDER_W - 2.0), float(wx)))
    obj.x = wx
    obj.y = float(wy)
    # A short trail is enough to estimate release velocity; keep ~0.15s of it.
    g.trail.append((t, wx, float(wy)))
    if len(g.trail) > 6:
        del g.trail[0]
    # Held things do not fall or act while carried (see world tick skips), but
    # zero their velocity so they do not lurch when picked up mid-stride.
    for attr in ("vx", "vy"):
        if hasattr(obj, attr):
            setattr(obj, attr, 0.0)


def release(world: Any, g: Grab, terrain: Any = None) -> None:
    """Let go. A quick flick throws; a slow move sets down gently.

    A thrown stickman is launched with the drag's velocity and left airborne,
    so the ordinary fall/landing code applies - fling one off a cliff and it
    can die on impact, exactly as if it had walked off. Set one down softly and
    it simply carries on.
    """
    obj = _resolve(world, g)
    _mark_held(world, g, False)
    if obj is None:
        return

    vx, vy = _toss_velocity(g)
    speed = (vx * vx + vy * vy) ** 0.5

    if g.kind == "agent":
        obj.vx = vx
        obj.vy = vy
        if speed > TOSS_SPEED:
            obj.on_ground = False       # a real throw: let physics fly it
            obj.action = None           # drop whatever it was mid-task
            _say(obj, "!")
        else:
            _reground(obj, world, terrain)
        return

    # Props: boulders can roll from a throw; everything else just lands.
    if g.kind == "prop":
        state = getattr(obj, "state", None)
        if isinstance(state, dict) and speed > TOSS_SPEED and obj.kind == "boulder":
            state["rolling"] = True
            state["vx"] = vx
        _reground(obj, world, terrain)


# ------------------------------------------------------------ click tools --
def use_tool(world: Any, tool: str, wx: float, wy: float) -> str | None:
    """Fire a single-click tool at (wx, wy). Returns a chronicle line, or None.

    The hand is not here - it is a press/drag/release gesture handled by
    grab/move_held/release. These are the instantaneous ones.
    """
    try:
        if tool == TOOL_LIGHTNING:
            world.events.strike_at(world, wx, direct=True)
            return "A bolt answers from a clear hand."
        if tool == TOOL_METEOR:
            world.events.meteor_at(world, wx)
            return "A star falls where a finger pointed."
        if tool == TOOL_PLANT:
            return _plant(world, wx)
        if tool == TOOL_ROCK:
            return _drop_rock(world, wx)
        if tool == TOOL_FEED:
            world.give(RES_FOOD, FEED_AMOUNT)
            return f"A gift of food appears in the stores (+{FEED_AMOUNT})."
        if tool == TOOL_SPAWN:
            return _spawn_person(world, wx)
    except Exception:
        log.exception("tool %r failed", tool)
    return None


def _plant(world: Any, wx: float) -> str | None:
    t = world.terrain
    x = max(6.0, min(float(RENDER_W - 6.0), float(wx)))
    p = world.props.spawn("sapling", x, t.ground_y(x), scale=0.3,
                          state={"growth": 0.0})
    return "A sapling is pressed into the earth." if p else None


def _drop_rock(world: Any, wx: float) -> str | None:
    t = world.terrain
    x = max(6.0, min(float(RENDER_W - 6.0), float(wx)))
    p = world.props.spawn("rock", x, t.ground_y(x), scale=0.8)
    return "A rock settles onto the ground." if p else None


def _spawn_person(world: Any, wx: float) -> str | None:
    fn = getattr(world, "spawn_visitor", None)
    if callable(fn):
        return fn(wx)
    # Fall back to the colony's own replacement path if no dedicated hook.
    try:
        world._spawn_replacement()
        return "A newcomer wanders in."
    except Exception:
        return None


# ---------------------------------------------------------------- helpers --
def _resolve(world: Any, g: Grab) -> Any:
    try:
        if g.kind == "agent":
            for a in world.population.agents:
                if a.id == g.id:
                    return a
        else:
            for p in world.props.all():
                if p.id == g.id:
                    return p
    except Exception:
        pass
    return None


def _mark_held(world: Any, g: Grab, held: bool) -> None:
    which = "held_agent_ids" if g.kind == "agent" else "held_prop_ids"
    ids = getattr(world, which, None)
    if ids is None:
        ids = set()
        setattr(world, which, ids)
    if held:
        ids.add(g.id)
    else:
        ids.discard(g.id)


def _toss_velocity(g: Grab) -> tuple[float, float]:
    if len(g.trail) < 2:
        return (0.0, 0.0)
    t0, x0, y0 = g.trail[0]
    t1, x1, y1 = g.trail[-1]
    dt = t1 - t0
    if dt <= 1e-4:
        return (0.0, 0.0)
    return ((x1 - x0) / dt, (y1 - y0) / dt)


def _reground(obj: Any, world: Any, terrain: Any) -> None:
    t = terrain if terrain is not None else getattr(world, "terrain", None)
    if t is None:
        return
    try:
        obj.y = t.ground_y(obj.x)
        if hasattr(obj, "vx"):
            obj.vx = 0.0
        if hasattr(obj, "vy"):
            obj.vy = 0.0
        if hasattr(obj, "on_ground"):
            obj.on_ground = True
        if hasattr(obj, "perch_y"):
            obj.perch_y = None
    except Exception:
        pass


def _say(obj: Any, symbol: str) -> None:
    try:
        obj.speech = symbol
        obj.speech_t = 1.5
    except Exception:
        pass
