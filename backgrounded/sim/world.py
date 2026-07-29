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
    AI_TICKS, ALL_RESOURCES, DAY_LENGTH_SEC, FOOD_PER_HEAD_TO_GROW,
    MAX_POP, MIN_POP, MORALE_TO_GROW, MORPH_LABELS,
    POP_BIRTH_COOLDOWN, POP_PER_HUT,
    REGROW_MAX, REGROW_PER_HEAD, RENDER_H, RENDER_W,
    RES_COOKED, RES_FOOD, RES_STONE, RES_WOOD, SAVE_VERSION,
    SCENE_LABELS, SCENE_NIGHT_STORM, SCENE_ROTATE_SEC, SCENES,
    TORCH_COLOR, TORCH_FLICKER, TORCH_INTENSITY, TORCH_RADIUS,
)
from . import behavior, names
from .actions import sweep_claims
from .animals import AnimalRegistry
from .ufo import Ufo
from .entities import GRAVE_DELAY, Population, Stickman
from .events import EventSystem
from .lighting import Lighting, LightSource
from .props import PropRegistry, scatter
from .structures import StructureRegistry
from .terrain import Terrain

log = logging.getLogger(__name__)

CHRONICLE_MAX = 400

#: How many headstones stay standing. Older ones weather away. Graves were
#: permanent, so a long run turned the map into a cemetery - 35 of them after
#: 25 minutes, on a world 1280px wide.
MAX_GRAVES = 10


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
        # Seed both, or they fall back to unseeded module RNGs and the whole
        # sim stops being reproducible from a seed - two runs of the same world
        # in one process gave 1 vs 0 deaths and different terrain wear, so no
        # balance number could be trusted. Offset the seeds so the two streams
        # do not march in lockstep.
        self.events: EventSystem = EventSystem(scene=scene, seed=self.seed ^ 0x513)
        self.animals: AnimalRegistry = AnimalRegistry(seed=self.seed ^ 0xA17)
        # The ufo was the one subsystem left unseeded: Ufo() with no seed falls
        # back to an unseeded module RNG, so two runs of the same world seed
        # differed in ufo.seed and next_in from the first tick - the abduction
        # landed at a different minute each time and no seeded comparison of a
        # long run could be trusted. Same offset trick as above.
        self.ufo: Ufo = Ufo(seed=self.seed ^ 0x71F)

        # Colony-level state
        self.auto_scene_rotate: bool = True     # flip scene every SCENE_ROTATE_SEC
        self.stockpile: dict[str, int] = {r: 0 for r in ALL_RESOURCES}
        self.build_queue: list[str] = []
        self.chronicle: deque[str] = deque(maxlen=CHRONICLE_MAX)
        self.stats: dict[str, int] = {
            "born": 0, "died": 0, "built": 0, "trees_felled": 0,
            "lightning_strikes": 0, "generations": 1,
        }

        # Subsystems that have failed and been disabled this session.
        self._disabled: set[str] = set()

        # Things the player's Hand tool is currently carrying, by id. Transient
        # by design: never serialised, so a save mid-drag reloads with nothing
        # held and no one can end up frozen in the air across a restart.
        self.held_agent_ids: set[int] = set()
        self.held_prop_ids: set[int] = set()

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
                rng=self.pyrng,
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
        # After the arrival line, so the chronicle reads in the order it
        # happened: the group lands, then you notice one of them is odd.
        for founder in alive:
            self._log_morph(founder)

    @property
    def agents(self) -> list:
        """The roster, under the name duck-typed helpers look for first."""
        return self.population.agents

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
        self._guarded("scene", lambda: self._tick_scene(dt))
        self._guarded("props", lambda: self._tick_props(dt))
        self._guarded("structures", lambda: self.structures.update(dt, self))
        self._guarded("agents", lambda: self._tick_agents(dt))
        self._guarded("animals", lambda: self.animals.tick(self, dt))
        self._guarded("ufo", lambda: self.ufo.tick(self, dt))
        self._guarded("lights", self._rebuild_lights)
        self._guarded("lighting", lambda: self.lighting.tick(dt))
        self._guarded("colony", lambda: self._tick_colony(dt))
        self._guarded("population", lambda: self._tick_population(dt))

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

    def _tick_scene(self, dt: float) -> None:
        """Rotate to a fresh, random scene every SCENE_ROTATE_SEC, and sweep the
        harvest-claim table while we are here (both are cheap, per-tick chores
        that key off the world clock). Keying the flip off the event system's own
        ``scene_t`` means a manual scene pick from the tray resets the countdown,
        so a chosen scene holds for a full interval before the weather moves on."""
        ev = self.events
        # getattr, not attribute access: this method is fail-soft-guarded, so a
        # missing attribute here does not raise loudly - it silently disables the
        # whole subsystem for the session. Defaulting rather than trusting the
        # instance is what keeps a partially-built world (an old save, a future
        # from_dict that forgets a field) rotating instead of quietly freezing.
        if getattr(self, "auto_scene_rotate", True) and \
                float(getattr(ev, "scene_t", 0.0)) >= SCENE_ROTATE_SEC:
            choices = [s for s in SCENES if s != ev.scene]
            if choices:
                nxt = choices[self.pyrng.randrange(len(choices))]
                if ev.request_scene(nxt):
                    self.log_event(f"The weather turns to {SCENE_LABELS.get(nxt, nxt)}.")
        sweep_claims(self)

    def _tick_agents(self, dt: float) -> None:
        ai_due = (self.tick_count % AI_TICKS) == 0
        for agent in self.population.alive_agents():
            if agent.id in self.held_agent_ids:
                continue          # held by the Hand tool: frozen until released
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

            # heal() had no caller in production: an agent hurt to 40hp still
            # read exactly 40hp after 600s. In a program that runs for hours
            # that makes every wound permanent and cumulative, so the whole
            # colony ratchets monotonically toward death.
            agent.heal(dt)
            # Pass the world RNG explicitly. apply_physics falls back to an
            # unseeded module-global otherwise, which made the sim
            # irreproducible: the same seed gave 8 deaths in one process and 1
            # in another, so balance changes could not be compared.
            agent.apply_physics(dt, self.terrain, rng=self.pyrng)

        self._free_orphaned_sleepers()
        self._reap_dead(dt)

    def _free_orphaned_sleepers(self) -> None:
        """Nobody stays inside a building that no longer exists.

        render/ skips anyone with .inside set, so an agent still flagged as
        inside a hut that burned down or was destroyed would be invisible for
        the rest of the run.
        """
        for agent in self.population.alive_agents():
            sid = getattr(agent, "inside", None)
            if sid is None:
                continue
            st = None
            try:
                for cand in self.structures.all():
                    if cand.id == sid:
                        st = cand
                        break
            except Exception:
                st = None
            if st is None or getattr(st, "is_ruined", False):
                agent.inside = None
                try:
                    agent.y = self.terrain.ground_y(agent.x)
                except Exception:
                    pass

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
                rng=self.pyrng,
            ))
            # Deaths are NOT replaced one-for-one any more. A fixed headcount
            # made every disaster free: lose four to a wildfire and four walk
            # in. The colony is allowed to shrink, and grows back on its own
            # when it has food, shelter and morale - see _tick_population.
            if len(self.population.alive_agents()) < MIN_POP:
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

        self._weather_graves(dt)

    def _weather_graves(self, dt: float) -> None:
        """Let old headstones fall back into the ground.

        Graves are permanent otherwise, and a long-running world becomes a
        graveyard: measured 35 headstones after 25 minutes of mudslide, on a
        world only 1280px wide. Keeping the most recent MAX_GRAVES means the
        colony's losses stay legible - you can see they have buried people
        recently - without the map silently turning into a cemetery.
        """
        try:
            graves = self.props.all_of("grave")
        except Exception:
            return
        if len(graves) <= MAX_GRAVES:
            return
        # Ascending by age, so the OLDEST land at the tail and get dropped.
        # This had reverse=True, which sorted oldest-first and therefore
        # deleted the newest headstone within a tick or two of it appearing:
        # past ten deaths the graveyard froze with the ten original stones and
        # no burial ever showed again - the exact opposite of the intent.
        # Verified after the fix: surviving ages are the small ones.
        #
        # props.tick_props already advances state["age"] for every grave each
        # tick, so ageing them here as well ran the clock at ~2x - graves ended
        # up 1629s old in a 900s world. props.py owns that field.
        graves.sort(key=lambda g: float(g.state.get("age", 0.0)))
        for g in graves[MAX_GRAVES:]:
            try:
                self.props.remove(g)
            except Exception:
                pass

    def clear_graves(self) -> int:
        """Remove every headstone. Exposed to the tray as 'Clear Graves'."""
        removed = 0
        try:
            for g in list(self.props.all_of("grave")):
                try:
                    self.props.remove(g)
                    removed += 1
                except Exception:
                    pass
        except Exception:
            log.exception("clear_graves failed")
        if removed:
            self.log_event(f"The old graves are lost to the weather.")
        return removed

    def randomise_terrain(self) -> None:
        """New landscape and scenery, same colony.

        Distinct from a full reset: the people, their names, generations and
        the chronicle all carry over - they wake up somewhere new. Structures
        go, because a hut half-way up the old cliff has nowhere to stand.
        """
        try:
            self.seed = int(self.pyrng.randrange(1 << 30))
            self.rng = np.random.default_rng(self.seed)
            self.terrain = Terrain.generate(self.seed, style=self._pick_style())
            self.props = scatter(self.terrain, self.rng,
                                 {"tree": 14, "rock": 8, "bush": 10, "boulder": 3})
            self.structures = StructureRegistry()
            self.build_queue = []
            for a in self.population.agents:
                a.x = float(self.rng.uniform(RENDER_W * 0.15, RENDER_W * 0.85))
                a.y = self.terrain.ground_y(a.x)
                a.vx = a.vy = 0.0
                a.on_ground = True
                a.action = None
            self.log_event("The land changes shape beneath them.")
            log.info("terrain randomised (seed=%d)", self.seed)
        except Exception:
            log.exception("randomise_terrain failed")

    def _spawn_replacement(self, born: bool = False) -> None:
        # Hard cap at the single choke point. Every growth path funnels through
        # here, but the UFO also hands abductees back, which bypassed the check
        # in _tick_population - a long run peaked at 11 against a MAX_POP of 10.
        # The below-MIN_POP rescue is the one exemption and cannot collide with
        # this, since MIN_POP < MAX_POP.
        if len(self.population.alive_agents()) >= MAX_POP:
            return
        used = {a.name for a in self.population.agents}
        gen = self.population.generation + 1
        self.population.generation = gen
        self.stats["generations"] = max(self.stats["generations"], gen)
        x = float(self.rng.uniform(RENDER_W * 0.1, RENDER_W * 0.9))
        s = self.population.spawn(
            x=x, y=self.terrain.ground_y(x),
            rng=self.pyrng,
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
        self.log_event(names.describe_event("born" if born else "arrived",
                                            name=s.name, generation=gen,
                                            rng=self.pyrng))
        self._log_morph(s)

    def _log_morph(self, s: Stickman) -> None:
        """Chronicle a rare odd-bodied villager, right after their arrival line.

        names.describe_event has no template kind for a body, so the sentence is
        built straight from MORPH_LABELS. Guarded because this hangs off three
        spawn paths - including the one that rescues a colony below MIN_POP -
        and none of them may fail over a log line.
        """
        try:
            if not s.is_mutant():
                return
            phrase = MORPH_LABELS.get(s.morph)
            if phrase:
                self.log_event(f"{s.name} is {phrase}.")
        except Exception:
            log.debug("morph chronicle failed for %r", getattr(s, "name", "?"),
                      exc_info=True)

    def spawn_visitor(self, x: float) -> str | None:
        """Player's Spawn tool: a newcomer appears at x, if there is room."""
        from ..constants import MAX_POP
        if len(self.population.alive_agents()) >= MAX_POP:
            return "The land is already full."
        used = {a.name for a in self.population.agents}
        gx = float(max(RENDER_W * 0.03, min(RENDER_W * 0.97, x)))
        s = self.population.spawn(
            x=gx, y=self.terrain.ground_y(gx),
            rng=self.pyrng,
            name=names.new_name(self.pyrng, used),
            generation=self.population.generation,
            birth_time=self.world_time,
        )
        self.stats["born"] += 1
        self.log_event(names.describe_event("arrived", name=s.name,
                                            generation=s.generation,
                                            rng=self.pyrng))
        self._log_morph(s)
        return None

    def _rebuild_lights(self) -> None:
        """Lights are derived state - rebuilt from scratch every tick."""
        srcs: list[LightSource] = []
        for agent in self.population.alive_agents():
            if getattr(agent, "inside", None) is not None:
                continue          # indoors: their light does not spill outside
            if agent.holds_candle and agent.__dict__.get("candle_lit", True):
                srcs.append(LightSource(
                    x=agent.x, y=agent.y - 14.0, radius=118.0,
                    color=(255, 186, 92), intensity=0.85, flicker=0.30,
                    kind="candle", owner_id=agent.id,
                ))
            elif getattr(agent, "holds_torch", True):
                srcs.append(LightSource(
                    x=agent.x, y=agent.y - 16.0, radius=TORCH_RADIUS,
                    color=TORCH_COLOR, intensity=TORCH_INTENSITY,
                    flicker=TORCH_FLICKER, kind="torch", owner_id=agent.id,
                ))
        for st in self.structures.all():
            ls = st.light_source()
            if ls:
                srcs.append(LightSource(**ls))
        try:
            beam = self.ufo.light_source()
            if beam:
                srcs.append(LightSource(**beam))
        except Exception:
            pass
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

    # ---------------------------------------------------------- population --
    def regrowth_factor(self) -> float:
        """How fast the wild recovers, scaled by headcount.

        Deliberately the inverse of the usual depletion curve: more hands means
        faster regrowth. Without it a colony of ten strips the map and starves,
        which collapses the population back down and makes the whole range
        pointless.
        """
        n = len(self.population.alive_agents())
        return float(min(REGROW_MAX, 1.0 + max(0, n - MIN_POP) * REGROW_PER_HEAD))

    def _tick_population(self, dt: float) -> None:
        """Let the colony grow when it is genuinely thriving."""
        self._birth_cd = getattr(self, "_birth_cd", POP_BIRTH_COOLDOWN) - dt
        alive = self.population.alive_agents()
        n = len(alive)

        # Never let the line die out entirely.
        if n < MIN_POP:
            self._spawn_replacement()
            self._birth_cd = POP_BIRTH_COOLDOWN
            return

        if n >= MAX_POP or self._birth_cd > 0.0:
            return

        food = self.stockpile.get(RES_FOOD, 0) + self.stockpile.get(RES_COOKED, 0)
        if food < FOOD_PER_HEAD_TO_GROW * n:
            return
        try:
            morale = sum(float(a.morale) for a in alive) / max(1, n)
        except Exception:
            morale = 0.0
        if morale < MORALE_TO_GROW:
            return
        # Somewhere to sleep, or nobody wants to bring anyone into it.
        huts = self.structures.count("hut")
        if huts <= 0 or n >= huts * POP_PER_HUT:
            return

        self._birth_cd = POP_BIRTH_COOLDOWN
        self._spawn_replacement(born=True)

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
            "animals": self.animals.to_dict(),
            "ufo": self.ufo.to_dict(),
            "auto_scene_rotate": bool(getattr(self, "auto_scene_rotate", True)),
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
        w.held_agent_ids = set()
        w.held_prop_ids = set()
        # from_dict builds the instance with __new__, so anything set only in
        # __init__ is simply absent here. Leaving this one out meant every
        # *reloaded* world raised AttributeError on its first _tick_scene and -
        # because subsystem ticks are fail-soft - silently disabled the "scene"
        # subsystem for the rest of the session, killing both the 10-minute
        # scene rotation and the harvest-claim sweep. Persisted so a user who
        # turns rotation off keeps it off across a restart.
        w.auto_scene_rotate = bool(d.get("auto_scene_rotate", True))

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
        w.animals = _sub("animals", AnimalRegistry.from_dict, AnimalRegistry)
        w.ufo = _sub("ufo", Ufo.from_dict, lambda: Ufo(seed=w.seed ^ 0x71F))

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
