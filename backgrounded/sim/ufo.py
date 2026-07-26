"""Ufo - the abduction event.

Pure python. **No pygame** - like everything else in ``sim/`` this module must
stay importable and runnable headless.

One :class:`Ufo` instance is owned by :class:`~.world.World`. It is a small
state machine that spends nearly all of its life in ``idle`` counting down a
long random timer, then descends, holds somebody in a beam, and takes them.

    idle -> arrive -> beam   -> depart -> idle        (an abduction)
    idle -> arrive -> return -> depart -> idle        (a delivery)

**Abduction is not death.** The victim is *removed from the roster while still
alive*: ``die()`` is never called, no death hook fires, ``death_cause`` stays
empty and therefore ``World._reap_dead`` never sees them and never raises a
gravestone. They are remembered in :attr:`Ufo.abducted`, and a later cycle may
bring one back (``UFO_RETURN_CHANCE``) with their original name, colour and
generation intact - dazed, but home.

Holding the victim
------------------
An agent under the beam must not walk away. Every tick of ``beam``/``return``
this module:

* clears ``action``/``action_raw`` and ``target_x``/``path_goal``,
* zeroes ``vx`` and re-pins ``x`` to the beam column,
* sets ``agent.beamed = True`` - the flag behaviour may respect if it wants to
  score nothing at all for a held agent, and
* sets ``pose_override`` so the figure reads as limp rather than mid-stride.

Re-pinning ``x`` is what makes the hold robust regardless of where world.py
puts this tick: even if behaviour hands the agent a fresh action first, the
agent ends the tick back under the beam. **Ticking the ufo after the agent
loop is still preferable** - it makes the freeze exact and leaves
``light_source()`` fresh for the light rebuild that follows.

Render contract (render/ reads, never writes)
---------------------------------------------
``active``/``phase``/``x``/``y`` place the craft, :meth:`beam_poly` gives the
beam's ground trapezoid, and :meth:`light_source` gives world.py a
``LightSource(**d)``-shaped dict so a beam at night actually lights the ground
under it. The victim's own ``lift_t`` (0..1) is how far up the beam they are
drawn; sim never moves their ``y``.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any

from ..constants import (
    MAX_POP as _MAX_POP,
    MAX_HEALTH,
    RENDER_H,
    RENDER_W,
    UFO_BEAM_SEC,
    UFO_INTERVAL_MAX,
    UFO_INTERVAL_MIN,
    UFO_RETURN_CHANCE,
)
from .entities import Stickman

log = logging.getLogger(__name__)

__all__ = [
    "Ufo",
    "PHASE_IDLE", "PHASE_ARRIVE", "PHASE_BEAM", "PHASE_DEPART", "PHASE_RETURN",
    "PHASES",
]

# ------------------------------------------------------------------ phases --
PHASE_IDLE = "idle"
PHASE_ARRIVE = "arrive"
PHASE_BEAM = "beam"
PHASE_DEPART = "depart"
PHASE_RETURN = "return"
PHASES: tuple[str, ...] = (
    PHASE_IDLE, PHASE_ARRIVE, PHASE_BEAM, PHASE_DEPART, PHASE_RETURN,
)

# ------------------------------------------------------------------ tuning --
#: Hover height above the victim's feet. The beam has to be long enough to read
#: as a beam once the 1280x800 frame is upscaled to 2560.
HOVER_HEIGHT = 120.0
HOVER_Y_MIN = 44.0                   # never hover off the top of the screen
HOVER_Y_MAX = RENDER_H * 0.62        # ...nor down inside the terrain

SPAWN_Y = -90.0                      # comes in from above the top edge
DEPART_END_Y = -150.0                # gone once it is this far past it again

DESCENT_SPEED = 132.0                # px/s while arriving
DRIFT_SPEED = 108.0                  # px/s of horizontal search while arriving
DEPART_SPEED = 150.0                 # px/s, and it accelerates away
DEPART_ACCEL = 2.4                   # multiplier growth per second of departure
ARRIVE_EPS = 3.0                     # px, close enough to start the beam
ARRIVE_TIMEOUT = 16.0                # s before it beams from wherever it got to
DEPART_TIMEOUT = 20.0                # s before it is declared gone regardless

#: Craft geometry, in world px. render/ may read these for the hull.
HULL_W = 58.0
HULL_H = 19.0

BEAM_TOP_HALF = 7.0                  # half-width where the beam leaves the hull
BEAM_BOTTOM_MIN = 11.0               # half-width at the ground, beam opening
BEAM_BOTTOM_MAX = 26.0               # ...and at full strength
BEAM_FADE = 0.7                      # s the beam takes to open and to close

BEAM_COLOR: tuple[int, int, int] = (170, 255, 206)
BEAM_LIGHT_RADIUS = 176.0
BEAM_LIGHT_INTENSITY = 0.92
BEAM_LIGHT_FLICKER = 0.24
BEAM_LIGHT_LIFT = 12.0               # px above the ground the pool is centred

#: The ufo never takes the last stickman. A colony of one that gets abducted is
#: an empty world, which is not an event - it is a bug that looks like one.
MIN_LEFT_BEHIND = 1
#: ...and if taking somebody leaves the colony below this, world's ordinary
#: replacement path is asked for a newcomer. Never a grave; just someone new
#: walking in out of the dark.
MIN_COLONY = 2

#: How many abductees are remembered. Each record carries a full agent snapshot
#: so a returnee comes back exactly as they left; keep the list short so an
#: all-night run cannot grow the save without bound.
MAX_ABDUCTED = 6

#: What being taken and put down again does to you.
DAZED_FATIGUE = 0.92
DAZED_MORALE = 0.06
DAZED_HUNGER = 0.35                  # minimum hunger on return

_EDGE_PAD = 26.0
_MAX_DT = 0.25


# ----------------------------------------------------------------- helpers --
def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _b(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return default


def _color(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    try:
        r, g, b = (int(c) for c in tuple(value)[:3])
    except Exception:
        return default
    return (int(_clamp(r, 0, 255)), int(_clamp(g, 0, 255)), int(_clamp(b, 0, 255)))


def _approach(cur: float, target: float, rate: float, dt: float) -> float:
    step = max(0.0, rate) * dt
    if cur < target:
        return min(target, cur + step)
    if cur > target:
        return max(target, cur - step)
    return cur


# ------------------------------------------------------------ world access --
# world.py is another agent's file and may evolve. Every read goes through one
# of these so a partial World degrades instead of taking the app down.
def _population(world: Any) -> Any:
    return getattr(world, "population", None)


def _alive(world: Any) -> list[Any]:
    pop = _population(world)
    if pop is None:
        return []
    fn = getattr(pop, "alive_agents", None)
    if callable(fn):
        try:
            return [a for a in fn() if getattr(a, "alive", False)]
        except Exception:
            return []
    try:
        return [a for a in pop if getattr(a, "alive", False)]
    except Exception:
        return []


def _agent_by_id(world: Any, aid: Any) -> Any:
    if aid is None:
        return None
    pop = _population(world)
    fn = getattr(pop, "by_id", None)
    if callable(fn):
        try:
            return fn(_i(aid, -1))
        except Exception:
            return None
    for a in _alive(world):
        if _i(getattr(a, "id", -1), -1) == _i(aid, -2):
            return a
    return None


def _ground_y(world: Any, x: float) -> float:
    terrain = getattr(world, "terrain", None)
    fn = getattr(terrain, "ground_y", None)
    if callable(fn):
        try:
            return _f(fn(float(x)), RENDER_H * 0.72)
        except Exception:
            pass
    return RENDER_H * 0.72


def _chronicle(world: Any, text: str) -> None:
    """Best-effort chronicle line. World has ``log_event``; be tolerant anyway."""
    if not text:
        return
    fn = getattr(world, "log_event", None)
    if callable(fn):
        try:
            fn(text)
            return
        except Exception:
            pass
    for name in ("chronicle", "history", "log"):
        obj = getattr(world, name, None)
        if obj is None:
            continue
        for meth in ("add", "record", "append"):
            call = getattr(obj, meth, None)
            if callable(call):
                try:
                    call(text)
                    return
                except Exception:
                    continue


def _bump(world: Any, key: str) -> None:
    try:
        stats = getattr(world, "stats", None)
        if isinstance(stats, dict):
            stats[key] = _i(stats.get(key), 0) + 1
    except Exception:
        pass


def _rand_x(rng: random.Random) -> float:
    return rng.uniform(_EDGE_PAD, RENDER_W - _EDGE_PAD)


# =================================================================== Ufo ====
class Ufo:
    """The lights over the ridge. One instance, owned by World."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed: int = random.randrange(1 << 30) if seed is None else _i(seed, 0)
        self._rng: random.Random = random.Random(self.seed ^ 0x0F0)

        self.active: bool = False
        self.phase: str = PHASE_IDLE
        self.x: float = RENDER_W * 0.5
        self.y: float = SPAWN_Y
        self.t: float = 0.0
        self.target_id: int | None = None
        self.next_in: float = self._interval()
        self.returning: bool = False
        self.abducted: list[dict[str, Any]] = []

        # Derived geometry, persisted so a reload looks identical mid-event.
        self.hover_y: float = HOVER_Y_MIN
        self.hold_x: float = self.x          # the beam column
        self.ground_y: float = RENDER_H * 0.72   # ground under the beam, cached
        self.taken_count: int = 0
        self.returned_count: int = 0
        self._sweep_t: float = 0.0           # derived; see _release_strays

    # ------------------------------------------------------------ schedule --
    def _interval(self) -> float:
        lo = min(UFO_INTERVAL_MIN, UFO_INTERVAL_MAX)
        hi = max(UFO_INTERVAL_MIN, UFO_INTERVAL_MAX)
        return float(self._rng.uniform(lo, hi))

    def summon(self, delay: float = 0.0) -> None:
        """Bring the next visit forward (debug hook / tray command)."""
        self.next_in = max(0.0, _f(delay, 0.0))

    # ---------------------------------------------------------------- tick --
    def tick(self, world: Any, dt: float) -> None:
        """Advance the state machine. Never raises."""
        try:
            step = _clamp(_f(dt, 0.0), 0.0, _MAX_DT)
            if step <= 0.0:
                return
            handler = {
                PHASE_IDLE: self._tick_idle,
                PHASE_ARRIVE: self._tick_arrive,
                PHASE_BEAM: self._tick_beam,
                PHASE_RETURN: self._tick_deliver,
                PHASE_DEPART: self._tick_depart,
            }.get(self.phase)
            if handler is None:
                self._go_idle()
                return
            handler(world, step)
        except Exception:
            log.exception("ufo tick failed in phase %r; going home", self.phase)
            try:
                self._panic_reset(world)
            except Exception:
                pass

    # -- idle --------------------------------------------------------------
    def _tick_idle(self, world: Any, dt: float) -> None:
        self.active = False
        self._sweep_t += dt
        if self._sweep_t >= 1.0:
            self._sweep_t = 0.0
            self._release_strays(world)
        self.next_in -= dt
        if self.next_in > 0.0:
            return
        if not self._begin(world):
            # Nothing to do this cycle - try again sooner than a full interval
            # so a colony that was momentarily too small is not skipped for
            # another twenty minutes.
            self.next_in = self._interval() * 0.25

    def _begin(self, world: Any) -> bool:
        """Start a visit. Returns False if there is nobody worth visiting."""
        want_return = bool(self.abducted) and self._rng.random() < UFO_RETURN_CHANCE
        victim = None if want_return else self._pick_victim(world)
        if victim is None and not want_return:
            # No legal victim (colony too small). A delivery is still possible.
            want_return = bool(self.abducted)
        if want_return:
            rec = self.abducted[0]
            self.returning = True
            self.target_id = _i(rec.get("id"), 0) or None
            self.hold_x = self._delivery_x(world)
        else:
            if victim is None:
                return False
            self.returning = False
            self.target_id = _i(getattr(victim, "id", 0), 0)
            self.hold_x = _clamp(_f(getattr(victim, "x", RENDER_W * 0.5)),
                                 _EDGE_PAD, RENDER_W - _EDGE_PAD)

        self.ground_y = _ground_y(world, self.hold_x)
        self.hover_y = self._hover_for(self.ground_y)
        # Come in from off to one side so it crosses the sky rather than
        # dropping out of nowhere.
        side = -1.0 if self._rng.random() < 0.5 else 1.0
        self.x = _clamp(self.hold_x + side * self._rng.uniform(160.0, 380.0),
                        _EDGE_PAD, RENDER_W - _EDGE_PAD)
        self.y = SPAWN_Y
        self.t = 0.0
        self.phase = PHASE_ARRIVE
        self.active = True
        return True

    def _hover_for(self, ground_y: float) -> float:
        return _clamp(ground_y - HOVER_HEIGHT, HOVER_Y_MIN, HOVER_Y_MAX)

    def _pick_victim(self, world: Any) -> Any:
        crowd = [a for a in _alive(world) if not getattr(a, "taken", False)]
        if len(crowd) <= MIN_LEFT_BEHIND:
            return None
        return crowd[self._rng.randrange(len(crowd))]

    def _delivery_x(self, world: Any) -> float:
        """Set them down near the settlement if there is one, else anywhere."""
        crowd = _alive(world)
        if crowd:
            who = crowd[self._rng.randrange(len(crowd))]
            base = _f(getattr(who, "x", RENDER_W * 0.5), RENDER_W * 0.5)
            base += self._rng.uniform(-90.0, 90.0)
        else:
            base = _rand_x(self._rng)
        return _clamp(base, _EDGE_PAD, RENDER_W - _EDGE_PAD)

    # -- arrive ------------------------------------------------------------
    def _tick_arrive(self, world: Any, dt: float) -> None:
        self.active = True
        self.t += dt

        if not self.returning:
            victim = _agent_by_id(world, self.target_id)
            if victim is None or not getattr(victim, "alive", False):
                # They died, or were taken by something else, on the way in.
                self._go_depart()
                return
            self.hold_x = _clamp(_f(getattr(victim, "x", self.hold_x), self.hold_x),
                                 _EDGE_PAD, RENDER_W - _EDGE_PAD)

        self.ground_y = _ground_y(world, self.hold_x)
        self.hover_y = self._hover_for(self.ground_y)

        self.x = _approach(self.x, self.hold_x, DRIFT_SPEED, dt)
        self.y = _approach(self.y, self.hover_y, DESCENT_SPEED, dt)

        in_place = (abs(self.x - self.hold_x) <= ARRIVE_EPS
                    and abs(self.y - self.hover_y) <= ARRIVE_EPS)
        if in_place or self.t >= ARRIVE_TIMEOUT:
            self.x = self.hold_x
            self.y = self.hover_y
            self.t = 0.0
            if self.returning:
                if not self._materialise(world):
                    self._release_victim(world)
                    self._go_depart()
                    return
                self.phase = PHASE_RETURN
            else:
                self.phase = PHASE_BEAM

    # -- beam (taking somebody) -------------------------------------------
    def _tick_beam(self, world: Any, dt: float) -> None:
        self.active = True
        victim = _agent_by_id(world, self.target_id)
        if victim is None or not getattr(victim, "alive", False):
            self._go_depart()
            return

        self.t += dt
        progress = _clamp(self.t / max(0.1, UFO_BEAM_SEC), 0.0, 1.0)
        self._hold(victim, progress)
        self.ground_y = _ground_y(world, self.hold_x)

        if self.t >= UFO_BEAM_SEC:
            self._complete_abduction(world, victim)

    def _complete_abduction(self, world: Any, victim: Any) -> None:
        """Lift completes: off the roster, into :attr:`abducted`. No grave."""
        name = str(getattr(victim, "name", "Someone"))
        try:
            victim.taken = True
            victim.lift_t = 1.0
        except Exception:
            pass

        self._pass_the_candle(world, victim)
        rec = self._record(world, victim)

        pop = _population(world)
        removed = False
        try:
            fn = getattr(pop, "remove", None)
            if callable(fn):
                removed = bool(fn(victim))
        except Exception:
            removed = False
        if not removed:
            # Could not take them off the roster - do NOT leave a live agent
            # flagged as taken, or render draws a permanently levitating
            # stickman that behaviour still tries to send to work.
            log.warning("ufo could not remove %s from the roster; releasing", name)
            try:
                victim.taken = False
            except Exception:
                pass
            self._release(victim)
            self._go_depart()
            return

        self.abducted.append(rec)
        if len(self.abducted) > MAX_ABDUCTED:
            del self.abducted[: len(self.abducted) - MAX_ABDUCTED]
        self.taken_count += 1
        _bump(world, "abducted")
        _chronicle(world, f"{name} is taken by the lights.")
        self._keep_colony_up(world)
        self._go_depart()

    def _record(self, world: Any, victim: Any) -> dict[str, Any]:
        """{name, colour, generation, id} - plus a full snapshot to come back with."""
        state: dict[str, Any] | None = None
        try:
            fn = getattr(victim, "to_dict", None)
            if callable(fn):
                got = fn()
                if isinstance(got, dict):
                    state = got
                    # A returnee must not arrive carrying a second candle, a
                    # stale action, or the "taken" flag itself.
                    state["holds_candle"] = False
                    state["action"] = None
                    state["taken"] = False
                    state["lift_t"] = 0.0
                    state["target_animal"] = None
        except Exception:
            state = None
        return {
            "id": _i(getattr(victim, "id", 0), 0),
            "name": str(getattr(victim, "name", "Someone")),
            "color": [int(c) for c in _color(getattr(victim, "color", None),
                                             (206, 214, 226))],
            "generation": _i(getattr(victim, "generation", 0), 0),
            "at": _f(getattr(world, "world_time", 0.0), 0.0),
            "state": state,
        }

    def _pass_the_candle(self, world: Any, victim: Any) -> None:
        """The world keeps exactly one candle; being taken must not blow it out."""
        try:
            if not getattr(victim, "holds_candle", False):
                return
            victim.holds_candle = False
            vid = _i(getattr(victim, "id", -1), -1)
            best, best_d = None, float("inf")
            for a in _alive(world):
                if _i(getattr(a, "id", -2), -2) == vid:
                    continue
                d = abs(_f(getattr(a, "x", 1e9), 1e9) - _f(getattr(victim, "x", 0.0)))
                if d < best_d:
                    best, best_d = a, d
            if best is not None:
                best.holds_candle = True
        except Exception:
            log.debug("candle hand-off failed", exc_info=True)

    def _keep_colony_up(self, world: Any) -> None:
        """A short colony gets a newcomer through the *existing* path.

        Deliberately world's own ``_spawn_replacement``: it owns generation
        numbering, the candle invariant and the "arrived" chronicle line. This
        never runs the death path, so no gravestone is ever raised for someone
        the lights took.
        """
        try:
            if len(_alive(world)) >= MIN_COLONY:
                return
            fn = getattr(world, "_spawn_replacement", None)
            if callable(fn):
                fn()
        except Exception:
            log.debug("replacement after abduction failed", exc_info=True)

    # -- return (putting somebody back) -----------------------------------
    def _materialise(self, world: Any) -> bool:
        """Re-add the abductee, held at the top of the beam. True on success."""
        rec = self._pending_record()
        if rec is None:
            return False
        pop = _population(world)
        if pop is None:
            return False
        # Respect the population ceiling. This path adds directly via pop.add,
        # bypassing World._spawn_replacement, so without this check a return at
        # a full colony pushes it to MAX_POP + 1. Keep the record and try again
        # on a later cycle, once somebody has left.
        try:
            alive = len(pop.alive_agents())
            if alive >= _MAX_POP:
                return False
        except Exception:
            pass

        agent: Any = None
        state = rec.get("state")
        if isinstance(state, dict):
            try:
                agent = Stickman.from_dict(state)
            except Exception:
                agent = None
        if agent is None:
            try:
                agent = Stickman()
            except Exception:
                return False
            agent.id = _i(rec.get("id"), 0)
            agent.name = str(rec.get("name") or "Someone")
            agent.color = _color(rec.get("color"), (206, 214, 226))
            agent.generation = _i(rec.get("generation"), 0)

        # Identity is the whole point of a return: keep the name and colour the
        # settlement knew them by, whatever the snapshot said.
        agent.name = str(rec.get("name") or agent.name)
        agent.color = _color(rec.get("color"), agent.color)
        agent.generation = _i(rec.get("generation"), agent.generation)

        # An id that has since been handed to somebody else would make
        # by_id/agent_by_id ambiguous forever. Take a fresh one in that case.
        try:
            aid = _i(rec.get("id"), 0)
            clash = pop.by_id(aid) is not None if callable(
                getattr(pop, "by_id", None)) else False
            if aid <= 0 or clash:
                aid = max(1, _i(getattr(pop, "next_id", 1), 1))
            agent.id = aid
        except Exception:
            pass

        self.ground_y = _ground_y(world, self.hold_x)
        agent.x = _clamp(self.hold_x, _EDGE_PAD, RENDER_W - _EDGE_PAD)
        agent.y = self.ground_y
        agent.vx = agent.vy = 0.0
        agent.on_ground = True
        agent.climbing = False
        agent.climb_t = 0.0
        agent.fall_t = 0.0
        agent.dead_t = 0.0
        agent.alive = True
        agent.death_cause = ""
        agent.action = None
        agent.action_raw = None
        agent.holds_candle = False
        agent.health = MAX_HEALTH
        agent.target_animal = None
        agent.taken = True            # still in the beam; cleared when set down
        agent.lift_t = 1.0

        try:
            add = getattr(pop, "add", None)
            if not callable(add):
                return False
            add(agent)
        except Exception:
            log.debug("could not re-add %s", agent.name, exc_info=True)
            return False

        # Keep the record and the live agent on the same id: _forget() matches
        # by it, and a re-issued id would otherwise drop the wrong record.
        try:
            rec["id"] = _i(agent.id, 0)
        except Exception:
            pass
        self.target_id = _i(agent.id, 0)
        return True

    def _pending_record(self) -> dict[str, Any] | None:
        """The record this delivery is for - matched by id, else the oldest."""
        if not self.abducted:
            return None
        if self.target_id is not None:
            for rec in self.abducted:
                if _i(rec.get("id"), -1) == _i(self.target_id, -2):
                    return rec
        return self.abducted[0]

    def _tick_deliver(self, world: Any, dt: float) -> None:
        self.active = True
        agent = _agent_by_id(world, self.target_id)
        if agent is None or not getattr(agent, "alive", False):
            self._forget(self.target_id)
            self._go_depart()
            return

        self.t += dt
        progress = _clamp(self.t / max(0.1, UFO_BEAM_SEC), 0.0, 1.0)
        self._hold(agent, 1.0 - progress)      # lowered, not lifted
        self.ground_y = _ground_y(world, self.hold_x)

        if self.t >= UFO_BEAM_SEC:
            self._complete_return(world, agent)

    def _complete_return(self, world: Any, agent: Any) -> None:
        name = str(getattr(agent, "name", "Someone"))
        self._release(agent)
        try:
            agent.taken = False
            agent.lift_t = 0.0
            agent.y = _ground_y(world, agent.x)
            agent.on_ground = True
            # Dazed: wherever they have been, it was not restful.
            agent.fatigue = max(_f(getattr(agent, "fatigue", 0.0)), DAZED_FATIGUE)
            agent.morale = min(_f(getattr(agent, "morale", 1.0), 1.0), DAZED_MORALE)
            agent.hunger = max(_f(getattr(agent, "hunger", 0.0)), DAZED_HUNGER)
            agent.speech = None
            agent.speech_t = 0.0
        except Exception:
            log.debug("could not daze %s", name, exc_info=True)

        self._forget(self.target_id)
        self.returned_count += 1
        _bump(world, "returned")
        _chronicle(world, f"{name} is set down again, dazed.")
        self._go_depart()

    def _forget(self, aid: Any) -> None:
        try:
            if aid is not None:
                for i, rec in enumerate(self.abducted):
                    if _i(rec.get("id"), -1) == _i(aid, -2):
                        del self.abducted[i]
                        return
            if self.abducted:
                del self.abducted[0]
        except Exception:
            pass

    # -- depart ------------------------------------------------------------
    def _tick_depart(self, world: Any, dt: float) -> None:
        self.active = True
        self.t += dt
        self.y -= DEPART_SPEED * (1.0 + DEPART_ACCEL * self.t) * dt
        self.x = _clamp(self.x + math.sin(self.t * 1.7) * 26.0 * dt,
                        _EDGE_PAD, RENDER_W - _EDGE_PAD)
        if self.y <= DEPART_END_Y or self.t >= DEPART_TIMEOUT:
            self._go_idle()

    # -- transitions -------------------------------------------------------
    def _release_victim(self, world: Any) -> None:
        """Let go of whoever we are holding, alive or not."""
        try:
            vid = self.target_id
            if vid is None:
                return
            for agent in (getattr(getattr(world, "population", None),
                                  "agents", None) or []):
                if getattr(agent, "id", None) == vid:
                    self._release(agent)
                    try:
                        agent.taken = False
                    except Exception:
                        pass
                    break
        except Exception:
            log.debug("victim release failed", exc_info=True)

    def _go_depart(self) -> None:
        self.phase = PHASE_DEPART
        self.t = 0.0
        self.target_id = None
        self.active = True

    def _go_idle(self) -> None:
        self.phase = PHASE_IDLE
        self.active = False
        self.returning = False
        self.target_id = None
        self.t = 0.0
        self.y = SPAWN_Y
        self.next_in = self._interval()

    def _panic_reset(self, world: Any) -> None:
        """Something threw. Let go of whoever we were holding and go home."""
        # Anyone still findable on the roster is by definition not abducted -
        # a completed abduction is off it - so whoever this is, let go of them.
        agent = _agent_by_id(world, self.target_id)
        if agent is not None:
            self._release(agent)
            try:
                agent.taken = False
            except Exception:
                pass
        self._go_idle()

    # -- holding an agent --------------------------------------------------
    def _hold(self, agent: Any, lift: float) -> None:
        """Pin an agent under the beam for this tick. Never raises."""
        try:
            agent.action = None
            agent.action_raw = None
            agent.target_x = None
            agent.path_goal = None
            agent.vx = 0.0
            agent.vy = 0.0
            agent.climbing = False
            agent.climb_t = 0.0
            agent.x = _clamp(self.hold_x, 0.0, float(RENDER_W - 1))
            agent.beamed = True
            agent.pose_override = "fall"
            agent.lift_t = _clamp(_f(lift, 0.0), 0.0, 1.0)
        except Exception:
            log.debug("could not hold agent under the beam", exc_info=True)

    def _release_strays(self, world: Any) -> None:
        """While idle, nobody may still be held. Cheap insurance, once a second.

        A save written mid-beam whose ufo state did not survive the reload - or
        a tick that threw between the hold and the lift - would otherwise leave
        a living agent frozen, posed limp and drawn hanging in mid-air forever,
        with nothing left in the world to let go of them.
        """
        try:
            # Sweep the whole roster, not just the living. A victim who died
            # mid-beam was skipped forever: the corpse kept lift_t and stayed
            # 'beamed', so render drew the body hanging in mid-air above its
            # own gravestone until the roster retired it.
            roster = list(getattr(getattr(world, "population", None),
                                  "agents", None) or _alive(world))
            for agent in roster:
                if getattr(agent, "beamed", False) or _f(
                        getattr(agent, "lift_t", 0.0), 0.0) > 0.0:
                    self._release(agent)
                    try:
                        agent.taken = False
                    except Exception:
                        pass
        except Exception:
            log.debug("stray release failed", exc_info=True)

    @staticmethod
    def _release(agent: Any) -> None:
        try:
            agent.beamed = False
            if getattr(agent, "pose_override", None) == "fall":
                agent.pose_override = None
            agent.lift_t = 0.0
        except Exception:
            pass

    # ------------------------------------------------------------ readouts --
    def is_beaming(self) -> bool:
        """True while the beam is down (either direction)."""
        return self.active and self.phase in (PHASE_BEAM, PHASE_RETURN)

    def beam_progress(self) -> float:
        """0..1 through the current beam, 0 when there is no beam."""
        if not self.is_beaming():
            return 0.0
        return _clamp(self.t / max(0.1, UFO_BEAM_SEC), 0.0, 1.0)

    def _beam_envelope(self) -> float:
        """0..1 opening/closing envelope, so the beam does not snap on and off."""
        if not self.is_beaming():
            return 0.0
        span = max(0.1, UFO_BEAM_SEC)
        open_u = _clamp(self.t / BEAM_FADE, 0.0, 1.0)
        close_u = _clamp((span - self.t) / BEAM_FADE, 0.0, 1.0)
        return _clamp(open_u * close_u, 0.0, 1.0)

    def beam_poly(self) -> list[tuple[float, float]]:
        """The beam as a ground-anchored trapezoid, or ``[]`` when there is none.

        Corners run hull-left, hull-right, ground-right, ground-left, so the
        renderer can fill it directly.
        """
        try:
            env = self._beam_envelope()
            if env <= 0.0:
                return []
            top_y = self.y + HULL_H * 0.5
            bot_y = max(top_y + 4.0, self.ground_y)
            half = BEAM_BOTTOM_MIN + (BEAM_BOTTOM_MAX - BEAM_BOTTOM_MIN) * env
            return [
                (self.x - BEAM_TOP_HALF, top_y),
                (self.x + BEAM_TOP_HALF, top_y),
                (self.x + half, bot_y),
                (self.x - half, bot_y),
            ]
        except Exception:
            return []

    def light_source(self) -> dict[str, Any] | None:
        """LightSource kwargs for the beam pool, or None when it is dark.

        Shaped for ``LightSource(**ufo.light_source())`` exactly as
        ``Structure.light_source()`` is, so world.py can append it to the light
        rebuild without a special case. The pool sits *on the ground* under the
        beam - that is the whole point: a beam at night should light the dirt.
        """
        try:
            env = self._beam_envelope()
            if env <= 0.0:
                return None
            return {
                "x": float(self.x),
                "y": float(self.ground_y - BEAM_LIGHT_LIFT),
                "radius": float(BEAM_LIGHT_RADIUS * (0.55 + 0.45 * env)),
                "color": tuple(BEAM_COLOR),
                "intensity": float(BEAM_LIGHT_INTENSITY * env),
                "flicker": float(BEAM_LIGHT_FLICKER),
            }
        except Exception:
            return None

    def summary(self) -> str:
        """One-line human description, for debug and the smoke test."""
        bits = [f"ufo {self.phase}"]
        if self.active:
            bits.append(f"at ({self.x:.0f},{self.y:.0f})")
        if self.target_id is not None:
            bits.append(f"target={self.target_id}")
        if self.phase == PHASE_IDLE:
            bits.append(f"next in {self.next_in:.0f}s")
        bits.append(f"taken={self.taken_count} returned={self.returned_count}")
        if self.abducted:
            bits.append("held: " + ", ".join(str(r.get("name")) for r in self.abducted))
        return " ".join(bits)

    # -------------------------------------------------------- persistence --
    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "active": bool(self.active),
            "phase": str(self.phase),
            "x": float(self.x),
            "y": float(self.y),
            "t": float(self.t),
            "target_id": None if self.target_id is None else int(self.target_id),
            "next_in": float(self.next_in),
            "returning": bool(self.returning),
            "abducted": [dict(r) for r in self.abducted],
            "hover_y": float(self.hover_y),
            "hold_x": float(self.hold_x),
            "ground_y": float(self.ground_y),
            "taken_count": int(self.taken_count),
            "returned_count": int(self.returned_count),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Ufo":
        """Defensive load: junk gives an idle ufo, never an exception."""
        u = cls()
        if not isinstance(d, dict):
            return u
        try:
            u.seed = _i(d.get("seed"), u.seed)
            u._rng = random.Random(u.seed ^ 0x0F0)

            phase = d.get("phase")
            u.phase = phase if isinstance(phase, str) and phase in PHASES else PHASE_IDLE
            u.x = _clamp(_f(d.get("x"), RENDER_W * 0.5), -400.0, RENDER_W + 400.0)
            u.y = _clamp(_f(d.get("y"), SPAWN_Y), -600.0, float(RENDER_H))
            u.t = max(0.0, _f(d.get("t"), 0.0))
            u.next_in = max(0.0, _f(d.get("next_in"), u.next_in))
            u.returning = _b(d.get("returning"), False)
            u.hover_y = _clamp(_f(d.get("hover_y"), HOVER_Y_MIN),
                               HOVER_Y_MIN, HOVER_Y_MAX)
            u.hold_x = _clamp(_f(d.get("hold_x"), RENDER_W * 0.5),
                              0.0, float(RENDER_W - 1))
            u.ground_y = _clamp(_f(d.get("ground_y"), RENDER_H * 0.72),
                                0.0, float(RENDER_H))
            u.taken_count = max(0, _i(d.get("taken_count"), 0))
            u.returned_count = max(0, _i(d.get("returned_count"), 0))

            tid = d.get("target_id")
            u.target_id = _i(tid, 0) if isinstance(tid, (int, float)) and not \
                isinstance(tid, bool) else None

            raw = d.get("abducted")
            if isinstance(raw, (list, tuple)):
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    rec = {
                        "id": _i(item.get("id"), 0),
                        "name": str(item.get("name") or "Someone"),
                        "color": [int(c) for c in _color(item.get("color"),
                                                         (206, 214, 226))],
                        "generation": _i(item.get("generation"), 0),
                        "at": _f(item.get("at"), 0.0),
                        "state": item.get("state") if isinstance(
                            item.get("state"), dict) else None,
                    }
                    u.abducted.append(rec)
                if len(u.abducted) > MAX_ABDUCTED:
                    del u.abducted[: len(u.abducted) - MAX_ABDUCTED]

            u.active = _b(d.get("active"), u.phase != PHASE_IDLE)
            # A save caught mid-beam whose victim no longer exists would hold a
            # phase that can never complete; the phase handlers all check, but
            # normalise the obvious contradictions here anyway.
            if u.phase == PHASE_IDLE:
                u.active = False
                u.target_id = None
            elif u.phase in (PHASE_BEAM, PHASE_RETURN) and u.target_id is None:
                u.phase = PHASE_DEPART
                u.t = 0.0
        except Exception:
            log.warning("ufo save unreadable; starting idle", exc_info=True)
            return cls()
        return u


# ------------------------------------------------------------- smoke test --
if __name__ == "__main__":   # pragma: no cover - headless, runs a real World
    import sys

    from .world import World

    SIM_DT = 1.0 / 30.0

    def run(seed: int, seconds: float, visits: int = 3) -> dict[str, Any]:
        w = World(seed=seed)
        u = Ufo(seed=seed)
        u.summon(2.0)
        graves_before = len(w.props.all_of("grave"))
        beam_lights = 0
        polys = 0
        events = 0
        steps = int(seconds / SIM_DT)
        for i in range(steps):
            w.tick(SIM_DT)
            u.tick(w, SIM_DT)
            if u.light_source() is not None:
                beam_lights += 1
            if u.beam_poly():
                polys += 1
            if u.phase == PHASE_IDLE and events < visits and u.next_in > 60.0:
                events += 1
                u.summon(1.5)          # force the next visit for the test
        return {
            "alive": len(w.population.alive_agents()),
            "taken": u.taken_count,
            "returned": u.returned_count,
            "held": len(u.abducted),
            "graves": len(w.props.all_of("grave")) - graves_before,
            "deaths": w.stats["died"],
            "beam_ticks": beam_lights,
            "poly_ticks": polys,
            "chronicle": [c for c in w.chronicle
                          if "lights" in c or "set down" in c],
            "world": w,
            "ufo": u,
        }

    print("-- forced visits ------------------------------------------------")
    total_taken = total_ret = 0
    for seed in (1, 2, 3, 4, 5):
        r = run(seed, 240.0, visits=4)
        total_taken += r["taken"]
        total_ret += r["returned"]
        print(f"seed {seed}: taken={r['taken']} returned={r['returned']} "
              f"alive={r['alive']} deaths={r['deaths']} "
              f"new graves={r['graves']} beam ticks={r['beam_ticks']}")
        for line in r["chronicle"]:
            print("   ", line)
        assert r["graves"] <= r["deaths"], "an abduction raised a gravestone"
        assert r["alive"] >= 1, "the colony was emptied"
        assert r["beam_ticks"] > 0, "the beam never lit anything"
        assert r["poly_ticks"] > 0, "the beam never had geometry"
    print(f"totals: taken={total_taken} returned={total_ret}")
    assert total_taken > 0, "nobody was ever taken"

    print("-- light + geometry --------------------------------------------")
    w = World(seed=9)
    u = Ufo(seed=9)
    u.summon(0.0)
    seen_light: dict[str, Any] | None = None
    for _ in range(int(40.0 / SIM_DT)):
        w.tick(SIM_DT)
        u.tick(w, SIM_DT)
        ls = u.light_source()
        if ls is not None and (seen_light is None
                               or ls["intensity"] > seen_light["intensity"]):
            seen_light = ls
            poly = u.beam_poly()
    assert seen_light is not None, "no beam light in 40s of forced visit"
    print("light:", {k: (round(v, 2) if isinstance(v, float) else v)
                     for k, v in seen_light.items()})
    print("poly :", [(round(p[0], 1), round(p[1], 1)) for p in poly])
    from .lighting import LightSource
    LightSource(**seen_light)          # must be constructible as-is
    assert len(poly) == 4 and poly[2][1] > poly[0][1]

    print("-- persistence --------------------------------------------------")
    blob = u.to_dict()
    clone = Ufo.from_dict(blob)
    assert clone.to_dict() == blob, "round-trip mismatch"
    assert Ufo.from_dict(None).phase == PHASE_IDLE
    assert Ufo.from_dict({"phase": "nonsense", "abducted": [1, "x"]}).phase == PHASE_IDLE
    assert Ufo.from_dict({"phase": PHASE_BEAM, "target_id": None}).phase == PHASE_DEPART
    junk = Ufo.from_dict({"x": float("nan"), "t": "boom", "next_in": None})
    assert junk.phase == PHASE_IDLE and math.isfinite(junk.x)
    print("round-trip ok;", clone.summary())

    print("-- never raises -------------------------------------------------")
    lone = Ufo(seed=3)
    lone.tick(None, 0.05)              # no world at all
    lone.tick(object(), 0.05)          # a world with nothing on it
    lone.summon(0.0)
    for _ in range(200):
        lone.tick(object(), 0.05)
    assert lone.phase == PHASE_IDLE, lone.phase
    print("survived a worldless tick:", lone.summary())
    print("ok")
    sys.exit(0)
