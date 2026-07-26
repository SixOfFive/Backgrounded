"""World - the simulation aggregate root.

Owns every piece of sim state and drives the fixed-timestep tick. Contains no
pygame and no rendering: this module must stay importable and runnable headless
so the simulation can be tested without a display.

Fail-soft policy: each subsystem tick is individually guarded. A subsystem that
raises is logged once and disabled for the rest of the session rather than
taking the whole program down - this app runs unattended for hours.
"""
from __future__ import annotations

import logging
import random
from collections import deque
from typing import Any

import numpy as np

from ..constants import (
    AI_TICKS, ALL_RESOURCES, DAY_LENGTH_SEC, RENDER_H, RENDER_W,
    RES_FOOD, RES_STONE, RES_WOOD, SCENE_NIGHT_STORM, SAVE_VERSION,
)
from . import behavior, names
from .entities import GRAVE_DELAY, Population, Stickman
from .events import EventSystem
from .lighting import Lighting, LightSource
from .props import PropRegistry, scatter
from .structures import StructureRegistry
from .terrain import Terrain

log = logging.getLogger(__name__)

CHRONICLE_MAX = 400


class World:
    """Everything that persists across a restart."""

    def __init__(self, seed: int | None = None, scene: str = SCENE_NIGHT_STORM) -> None:
        self.seed: int = int(seed if seed is not None else random.randrange(1 << 30))
        self.rng: np.random.Generator = np.random.default_rng(self.seed)
        self.pyrng: random.Random = random.Random(self.seed ^ 0x5EED)

        self.world_time: float = 0.0      # seconds since world creation
        self.tick_count: int = 0

        self.terrain: Terrain = Terrain.generate(self.seed, style=self._pick_style())
        self.props: PropRegistry = scatter(
            self.terrain, self.rng,
            {"tree": 14, "rock": 8, "bush": 10, "boulder": 3},
        )
        self.structures: StructureRegistry = StructureRegistry()
        self.population: Population = Population()
        self.lighting: Lighting = Lighting()
        self.events: EventSystem = EventSystem(scene=scene)

        # Colony-level state
        self.stockpile: dict[str, int] = {r: 0 for r in ALL_RESOURCES}
        self.build_queue: list[str] = []
        self.chronicle: deque[str] = deque(maxlen=CHRONICLE_MAX)
        self.stats: dict[str, int] = {
            "born": 0, "died": 0, "built": 0, "trees_felled": 0,
            "lightning_strikes": 0, "generations": 1,
        }

        # Subsystems that have failed and been disabled this session.
        self._disabled: set[str] = set()

        self._seed_population()

    # ------------------------------------------------------------ startup --
    def _pick_style(self) -> str:
        return str(self.rng.choice(["hills", "cliffs", "plateau", "chasm", "valley"]))

    def _seed_population(self, n: int = 4) -> None:
        used: set[str] = set()
        for i in range(n):
            x = float(self.rng.uniform(RENDER_W * 0.2, RENDER_W * 0.8))
            s = self.population.spawn(
                x=x,
                y=self.terrain.ground_y(x),
                name=names.new_name(self.pyrng, used),
                generation=1,
                birth_time=0.0,
            )
            used.add(s.name)
            s.role = "gatherer" if i % 2 else "builder"
        # Exactly one candle-bearer, per the spec's night-storm scene.
        alive = self.population.alive_agents()
        if alive:
            alive[0].holds_candle = True
            alive[0].role = "elder"
        self.log_event(f"A group of {n} arrives in a new land.")

    # --------------------------------------------------------------- time --
    @property
    def day_fraction(self) -> float:
        """0.0 = midnight, 0.5 = noon."""
        return (self.world_time % DAY_LENGTH_SEC) / DAY_LENGTH_SEC

    @property
    def is_night(self) -> bool:
        f = self.day_fraction
        return f < 0.25 or f > 0.75

    # --------------------------------------------------------------- tick --
    def tick(self, dt: float) -> None:
        self.world_time += dt
        self.tick_count += 1

        self._guarded("events", lambda: self.events.tick(self, dt))
        self._guarded("props", lambda: self._tick_props(dt))
        self._guarded("structures", lambda: self.structures.update(dt, self))
        self._guarded("agents", lambda: self._tick_agents(dt))
        self._guarded("lights", self._rebuild_lights)
        self._guarded("lighting", lambda: self.lighting.tick(dt))
        self._guarded("colony", lambda: self._tick_colony(dt))

    def _guarded(self, name: str, fn) -> None:
        """Run a subsystem step, disabling it if it ever raises.

        `fn` is a zero-arg callable rather than (method, *args) deliberately:
        resolving `self.foo.bar` at the call site would happen *outside* this
        try block, so a missing attribute would escape the guard entirely.
        """
        if name in self._disabled:
            return
        try:
            fn()
        except Exception:
            log.exception("subsystem %r failed; disabling for this session", name)
            self._disabled.add(name)

    def _tick_agents(self, dt: float) -> None:
        ai_due = (self.tick_count % AI_TICKS) == 0
        for agent in self.population.alive_agents():
            agent.update_needs(dt, self)
            if not agent.alive:
                # Starved or froze this tick. The roster snapshot still lists
                # them, and running the rest of the loop would hand a corpse a
                # fresh action (die() clears agent.action, which reads as
                # "needs a new one") and then walk it around.
                continue

            act = agent.action
            need_new = act is None or act.done or act.failed
            if not need_new and ai_due:
                # An action in flight is left alone. Re-scoring on a timer made
                # agents abandon half-finished jobs: most tasks take 13-20s to
                # complete, so re-deciding every few seconds meant walking
                # toward a tree, changing its mind, walking back, and never
                # arriving. Only a real emergency interrupts now.
                try:
                    need_new = behavior.emergency_override(agent, self)
                except Exception:
                    need_new = False
            if need_new:
                try:
                    agent.action = behavior.choose_action(agent, self)
                except Exception:
                    log.exception("choose_action failed for %s", agent.name)
                    agent.action = None

            if agent.action is not None:
                try:
                    agent.action.update(agent, self, dt)
                except Exception:
                    log.exception("action %r failed for %s",
                                  agent.action.kind, agent.name)
                    agent.action.failed = True

            agent.apply_physics(dt, self.terrain)

        self._reap_dead(dt)

    def _reap_dead(self, dt: float) -> None:
        """Turn fresh corpses into graves and bring in a replacement."""
        for agent in list(self.population.agents):
            if agent.alive:
                continue
            # This is the ONLY place a corpse's clock advances. _tick_agents
            # iterates alive_agents(), so the dead never reach apply_physics
            # and Stickman._physics never runs for them. Dropping this (it
            # looked like a double-count against entities.py) stopped dead_t
            # ever reaching the burial threshold: no graves, no replacements,
            # and the colony went extinct with stats["died"] still reading 0.
            agent.dead_t += dt
            if agent.dead_t < 2.0 or agent.__dict__.get("_buried"):
                continue
            agent.__dict__["_buried"] = True
            self.props.add_grave(agent.x, self.terrain.ground_y(agent.x), agent.name)
            self.stats["died"] += 1
            # One chronicle line per death, here, because every cause funnels
            # through this reaper. "death" is not a template kind - it fell
            # through to the generic "did something worth noting", so a colony
            # could lose half its people without the log ever saying so.
            # death_event_kind() maps the cause onto died_fall / died_hunger /
            # died_lightning / ... so the line names who died and of what.
            self.log_event(names.describe_event(
                agent.death_event_kind(),
                name=agent.name,
                cause=agent.death_cause or "the dark",
                x=agent.x,
            ))
            self._spawn_replacement()

        # Retire buried corpses off the roster. Leaving them on it meant
        # Population.agents only ever grew - measured 11 -> 41 entries over
        # 1800s while the living count stayed at 4 - and every tick's
        # alive_agents(), every hazard's agent scan and every save walked the
        # whole thing. This program is meant to run unattended for hours, so an
        # unbounded roster is a real leak, not a tidiness issue. The grave prop
        # is what remembers them now.
        stale = [a for a in self.population.agents
                 if not a.alive and a.__dict__.get("_buried")
                 and a.dead_t > GRAVE_DELAY]
        for a in stale:
            try:
                self.population.agents.remove(a)
            except ValueError:
                pass

    def _spawn_replacement(self) -> None:
        used = {a.name for a in self.population.agents}
        gen = self.population.generation + 1
        self.population.generation = gen
        self.stats["generations"] = max(self.stats["generations"], gen)
        x = float(self.rng.uniform(RENDER_W * 0.1, RENDER_W * 0.9))
        s = self.population.spawn(
            x=x, y=self.terrain.ground_y(x),
            name=names.new_name(self.pyrng, used),
            generation=gen, birth_time=self.world_time,
        )
        self.stats["born"] += 1
        # Keep exactly one candle in the world.
        if not any(a.holds_candle for a in self.population.alive_agents()):
            s.holds_candle = True
        # "arrived", not "arrival" - the latter is not a template key and fell
        # through to the generic line, so every replacement was announced as
        # "Something happened." rather than by name.
        self.log_event(names.describe_event("arrived", name=s.name,
                                            generation=gen))

    def _rebuild_lights(self) -> None:
        """Lights are derived state - rebuilt from scratch every tick."""
        srcs: list[LightSource] = []
        for agent in self.population.alive_agents():
            if agent.holds_candle and agent.__dict__.get("candle_lit", True):
                srcs.append(LightSource(
                    x=agent.x, y=agent.y - 14.0, radius=118.0,
                    color=(255, 186, 92), intensity=0.85, flicker=0.30,
                    kind="candle", owner_id=agent.id,
                ))
        for st in self.structures.all():
            ls = st.light_source()
            if ls:
                srcs.append(LightSource(**ls))
        for prop in self.props.burning():
            srcs.append(LightSource(
                x=prop.x, y=prop.y - 18.0, radius=150.0,
                color=(255, 130, 40), intensity=0.95, flicker=0.45,
                kind="fire", owner_id=None,
            ))
        self.lighting.sources = srcs

    def _tick_props(self, dt: float) -> None:
        """props.tick reports what happened; route it into the chronicle and
        stats rather than discarding it."""
        events = self.props.tick(self, dt) or ()
        for ev in events:
            if not isinstance(ev, dict):
                continue
            # The event NAME lives under "type"; "kind" is the prop's kind
            # ("tree", "bush"). Reading "kind" here matched nothing, so
            # trees_felled sat at 0 while trees fell all around it.
            #
            # Do not splat ev into describe_event either: it carries its own
            # "kind" key, which collides with describe_event's first parameter
            # and raises TypeError. That raise happens inside _guarded("props"),
            # which would disable the entire props subsystem for the session -
            # a logging nicety silently stopping trees from growing.
            etype = ev.get("type")
            if etype == "tree_felled":
                self.stats["trees_felled"] += 1
            elif etype == "prop_burned_out" and ev.get("kind") == "tree":
                self.log_event(f"A tree burned to nothing.")

    def _tick_colony(self, dt: float) -> None:
        # stats["built"] was declared but never written by anything, so it read
        # 0 while 157 structures completed across a test sweep. Count the
        # finished ones directly rather than trusting an increment somewhere.
        self.stats["built"] = sum(1 for st in self.structures.all()
                                  if getattr(st, "built", False))
        behavior.update_director(self, dt)
        behavior.assign_roles(self, dt)

    # ------------------------------------------------------------ helpers --
    def log_event(self, text: str) -> None:
        stamp = f"[day {int(self.world_time // DAY_LENGTH_SEC) + 1}]"
        self.chronicle.append(f"{stamp} {text}")

    def take(self, resource: str, qty: int) -> bool:
        if self.stockpile.get(resource, 0) >= qty:
            self.stockpile[resource] -= qty
            return True
        return False

    def give(self, resource: str, qty: int) -> None:
        self.stockpile[resource] = self.stockpile.get(resource, 0) + qty

    def agent_by_id(self, aid: int) -> Stickman | None:
        for a in self.population.agents:
            if a.id == aid:
                return a
        return None

    # -------------------------------------------------------- persistence --
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SAVE_VERSION,
            "seed": self.seed,
            "world_time": self.world_time,
            "tick_count": self.tick_count,
            "terrain": self.terrain.to_dict(),
            "props": self.props.to_dict(),
            "structures": self.structures.to_dict(),
            "population": self.population.to_dict(),
            "lighting": self.lighting.to_dict(),
            "events": self.events.to_dict(),
            "stockpile": dict(self.stockpile),
            "build_queue": list(self.build_queue),
            "chronicle": list(self.chronicle),
            "stats": dict(self.stats),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "World":
        """Defensive load: any missing or malformed section falls back to a
        freshly generated equivalent rather than raising."""
        w = cls.__new__(cls)
        w.seed = int(d.get("seed", random.randrange(1 << 30)))
        w.rng = np.random.default_rng(w.seed)
        w.pyrng = random.Random(w.seed ^ 0x5EED)
        w.world_time = float(d.get("world_time", 0.0))
        w.tick_count = int(d.get("tick_count", 0))
        w._disabled = set()

        def _sub(key, loader, fallback):
            try:
                return loader(d[key])
            except Exception:
                log.warning("save section %r unusable; regenerating", key)
                return fallback()

        w.terrain = _sub("terrain", Terrain.from_dict,
                         lambda: Terrain.generate(w.seed))
        w.props = _sub("props", PropRegistry.from_dict, PropRegistry)
        w.structures = _sub("structures", StructureRegistry.from_dict,
                            StructureRegistry)
        w.population = _sub("population", Population.from_dict, Population)
        w.lighting = _sub("lighting", Lighting.from_dict, Lighting)
        w.events = _sub("events", EventSystem.from_dict, EventSystem)

        w.stockpile = {r: 0 for r in ALL_RESOURCES}
        w.stockpile.update({k: int(v) for k, v in
                            (d.get("stockpile") or {}).items() if k in ALL_RESOURCES})
        w.build_queue = list(d.get("build_queue") or [])
        w.chronicle = deque(d.get("chronicle") or [], maxlen=CHRONICLE_MAX)
        w.stats = {"born": 0, "died": 0, "built": 0, "trees_felled": 0,
                   "lightning_strikes": 0, "generations": 1}
        w.stats.update({k: int(v) for k, v in (d.get("stats") or {}).items()})

        if not w.population.alive_agents():
            log.info("loaded world had no survivors; seeding a new group")
            w._seed_population()
        return w
