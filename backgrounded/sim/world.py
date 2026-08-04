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
    AI_TICKS, ALL_RESOURCES, DAY_LENGTH_SEC, DRAGON_GATE_STONE_HUTS,
    FOOD_PER_HEAD_TO_GROW,
    HUT_TIER_DWELL_SEC, HUT_TIER_MORALE,
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
from .dragons import DragonRegistry
from .throwing import SpearRegistry
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

#: Seconds a corpse lies there before _reap_dead raises its headstone and books
#: the death. Named because from_dict has to reason about the same threshold:
#: the "already buried" flag does not survive a save, so a body loaded from
#: inside this window has to be re-flagged or it is buried, and counted, twice.
BURIAL_DELAY = 2.0

#: The declared shape of :attr:`World.stats`. Defined once because __init__ and
#: from_dict both build it, and a key present in only one of them is a stat that
#: silently vanishes across a reload.
#:
#: "abducted" and "returned" are declared here at zero rather than being
#: conjured by ufo.py's first increment. A key that only exists once the lights
#: have actually been makes ``world.stats["abducted"]`` a KeyError on every
#: world where nothing has happened yet, so every reader has to remember
#: ``.get()`` - and the readers that did not simply left the abductee out of the
#: books entirely. That is how "survivors out of 9" came to under-report by
#: exactly one on every seed where the ufo took somebody, and why a scene or a
#: terrain change measured against survivor counts looked more lethal than it
#: was. Being taken is an exit from the roster that is NOT a death; being handed
#: back is an entry that is NOT a birth. Both now have a line in the ledger.
STAT_DEFAULTS: dict[str, int] = {
    "born": 0, "died": 0, "built": 0, "trees_felled": 0,
    "lightning_strikes": 0, "generations": 1,
    "abducted": 0, "returned": 0,
}


def _finite(value: Any, default: float, lo: float, hi: float) -> float:
    """A real number in [lo, hi] from anything at all, never raising.

    For fields read by :meth:`World.from_dict`, which is the one function here
    that must not raise: persist.load_world reads an exception as a corrupt save
    and quarantines it, so a junk value in a hand-edited file would cost the
    whole colony rather than the one field.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):   # nan / +-inf
        return default
    return lo if f < lo else (hi if f > hi else f)


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
        # Dragons and thrown spears, each on its own seeded stream and each
        # offset like every registry above it. Two separate streams rather than
        # one shared with pyrng, deliberately: behavior.choose_action draws from
        # pyrng once per scored candidate, so a dragon scheduler sharing it
        # would re-phase the stream wolves spawn from every time it rolled - the
        # measured re-phase confound that moved mean deaths 16% on its own.
        self.dragons: DragonRegistry = DragonRegistry(seed=self.seed ^ 0xD4A)
        self.spears: SpearRegistry = SpearRegistry(seed=self.seed ^ 0x59EA)

        self._init_colony_state()
        self._seed_population()

    def _init_colony_state(self) -> None:
        """Colony-level defaults, for a fresh world *and* a loaded one.

        Called by ``__init__`` and by ``from_dict``, and that is the entire
        reason it exists as a method. ``from_dict`` builds with
        ``cls.__new__(cls)``, so it never runs ``__init__`` and a field set only
        there is simply absent on every loaded world. The failure is close to
        silent: subsystem ticks run behind ``_guarded``, so the AttributeError
        is caught, logged once to a file nobody is watching, and the subsystem
        is switched off for the rest of the session.

        ``auto_scene_rotate`` went exactly that way - added to ``__init__``
        only, which killed the scene subsystem on every reloaded world and took
        the 10-minute rotation and the harvest-claim sweep down with it. That
        instance was fixed by adding one line to ``from_dict``; this is the fix
        for the next one, because a default added here now reaches both paths
        whether or not anyone remembers the second.

        ``from_dict`` calls this BEFORE reading the save, then overwrites what
        the save actually carries - so these are defaults, not a reset.
        """
        self.auto_scene_rotate: bool = True     # flip scene every SCENE_ROTATE_SEC
        self.stockpile: dict[str, int] = {r: 0 for r in ALL_RESOURCES}
        self.build_queue: list[str] = []
        self.chronicle: deque[str] = deque(maxlen=CHRONICLE_MAX)
        self.stats: dict[str, int] = dict(STAT_DEFAULTS)

        # Subsystems that have failed and been disabled this session. Never
        # carried across a load: a subsystem that died in a previous session
        # deserves a fresh start in this one.
        self._disabled: set[str] = set()

        # The stone-hut tier: a latch, and the dwell timer that earns it.
        # Declared here rather than in __init__ for exactly the reason this
        # method exists. _tick_hut_tier runs inside _guarded("colony"), so on a
        # loaded world a missing hut_tier_unlocked would not crash - it would
        # raise once, disable the "colony" subsystem for the session, and take
        # update_director and assign_roles down with it. Nothing visible would
        # say why; the colony would simply stop building and stop assigning
        # roles. from_dict writes the saved values over these defaults right
        # after calling this, and a save written before the tier existed keeps
        # them and re-earns the tier, which costs at most HUT_TIER_DWELL_SEC.
        self.hut_tier_unlocked: bool = False
        self.hut_tier_dwell: float = 0.0

        # The dragon gate: a latch, set the first tick DRAGON_GATE_STONE_HUTS
        # stone huts are standing. Declared HERE and not in __init__ for exactly
        # the reason above - _tick_dragon_gate rides inside _guarded("colony"),
        # so on a loaded world a missing attribute would not crash, it would
        # disable the colony subsystem for the session and quietly take
        # update_director and assign_roles with it.
        #
        # Chosen over built>=4 / pop>=6 / generation counts because it is the
        # only candidate that requires all three of the masonry TECH, a real
        # stone SURPLUS hauled and applied twice, and enough SHELTER that a
        # second hut was worth upgrading - it is a thing standing in the world
        # rather than a flag. Measured over 20 seeds x 75 sim-min: 16/20 reach
        # it, earliest 21.9 min, median 45.3, and four colonies never do. A
        # sustained DWELL was measured and REJECTED: across all 16 seeds that
        # reach the gate there were ZERO drops back below two stone huts for the
        # entire remainder of the run, so a dwell buys nothing and costs 1-2 min
        # of delay. The latch alone is the mechanism.
        self.dragons_unlocked: bool = False
        # Republished every director pass from the chosen hut's own
        # state["upgrade"], and deliberately NOT persisted: it is derived, and a
        # saved copy is a second source of truth that can disagree with the
        # structure it claims to describe. The truth lives on the Structure.
        self.upgrade_job: dict[str, Any] | None = None

        # Things the player's Hand tool is currently carrying, by id. Transient
        # by design: never serialised, so a save mid-drag reloads with nothing
        # held and no one can end up frozen in the air across a restart.
        self.held_agent_ids: set[int] = set()
        self.held_prop_ids: set[int] = set()

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
        # After "animals" and before "ufo". The spear registry is ALSO driven
        # from AnimalRegistry.tick, so that the mechanic is not silently inert
        # on a world built before this line existed; its tick is
        # frame-deduplicated on world.tick_count, so whichever of the two runs
        # second this frame is a no-op rather than a second helping of gravity.
        self._guarded("spears", lambda: self.spears.tick(self, dt))
        self._guarded("dragons", lambda: self.dragons.tick(self, dt))
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
            if agent.dead_t < BURIAL_DELAY or agent.__dict__.get("_buried"):
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
                # Every structure went with the old landscape, including the
                # watchtower somebody was standing on: leaving the perch set
                # would hold him at the height of a tower that no longer exists.
                a.perch_y = None
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

    def colony_morale(self) -> float:
        """Mean morale over the living - the colony's one definition of "happy".

        Two gates read this: the birth gate below (MORALE_TO_GROW) and the
        stone-hut tier gate (HUT_TIER_MORALE). One helper, so "how is the colony
        doing" cannot come to mean two slightly different numbers depending on
        who is asking.

        The body is the expression that was inlined in ``_tick_population``,
        moved here verbatim. It is deliberately NOT ``Population.stats()
        ["morale"]``, which computes the same mean in numpy float32 and so
        differs in the last bits: swapping that in here would silently shift the
        already-shipped birth gate, and nobody would ever trace a colony that
        grew one birth differently back to a change of dtype.

        Never raises. An empty colony reads 0.0, and so does any failure - a
        gate that cannot measure morale has to stay shut, not fall open.
        """
        try:
            alive = self.population.alive_agents()
            return sum(float(a.morale) for a in alive) / max(1, len(alive))
        except Exception:
            return 0.0

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
        # Shared with the stone-hut tier gate. The try/except that used to wrap
        # this sum now lives inside colony_morale(), which returns 0.0 on any
        # failure - the same "no birth" answer this code took before.
        if self.colony_morale() < MORALE_TO_GROW:
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
        # Rides inside the existing "colony" guard rather than taking a
        # _guarded name of its own. A new name is a new thing that can switch
        # itself off for the session on one bad tick, and this is two floats and
        # a comparison - it does not earn its own failure mode.
        self._tick_hut_tier(dt)
        # Same reasoning as _tick_hut_tier: rides inside the existing "colony"
        # guard rather than taking a _guarded name of its own. It is a counter
        # and a comparison; it does not earn its own failure mode.
        self._tick_dragon_gate(dt)

    def _stone_huts_standing(self) -> int:
        """How many built, unruined STONE huts are up right now.

        ``Structure.material()`` rather than reading ``state["material"]``
        directly: that accessor already strips and lower-cases, so ``"STONE "``
        from a hand-edited save reads as stone here exactly as it does in
        render, and an unknown material falls back to the default in one place
        instead of two. Never raises - it is called from a per-tick path and a
        malformed structure must cost its own row, not the gate.
        """
        n = 0
        for st in self.structures.all():
            try:
                if st.kind != "hut" or not st.built or st.is_ruined:
                    continue
                if st.material() == "stone":
                    n += 1
            except Exception:
                continue
        return n

    def _tick_dragon_gate(self, dt: float) -> None:
        """Latch the dragon gate once two stone huts are standing.

        LATCHED, like ``hut_tier_unlocked`` and for the same reason: a fire that
        ruins a stone hut does not un-notice the colony. An unlatched gate would
        also make the schedule flicker - the registry's own next-visit timer is
        armed on the latching edge, so a gate that closed again would keep
        re-arming DRAGON_FIRST_DELAY and no dragon would ever actually arrive.
        See ``_init_colony_state`` for why this signal and not another, and for
        the dwell that was measured and rejected.

        The registry ALSO counts stone huts for itself and prefers this flag
        when it is present, so the world-side latch is a mirror rather than a
        prerequisite - the feature is whole either way. What this adds is
        persistence: the registry's own count is recomputed from the standing
        structures, so a colony whose stone huts later burn down would look
        un-gated again to the registry alone. Once this is True it stays True
        across the fire and across the save.

        ``dt`` is unused - the gate is a comparison, not a dwell - and is taken
        anyway so the signature matches every other ``_tick_*`` here and a dwell
        could be reinstated without touching the call site.
        """
        if self.dragons_unlocked:
            return
        standing = self._stone_huts_standing()
        if standing >= DRAGON_GATE_STONE_HUTS:
            self.dragons_unlocked = True
            # No chronicle line here on purpose. The gate is not the event - the
            # HERALD is, and the registry announces that itself
            # DRAGON_FIRST_DELAY later with "Something crosses the sky, high
            # up." Announcing the unlock too would tell the player about a
            # mechanic before anything has happened, and turn a thing they
            # notice in the sky into a reward popping in a log.
            log.info("dragon gate latched at t=%.1fs (%d stone huts)",
                     self.world_time, self._stone_huts_standing())

    def _tick_hut_tier(self, dt: float) -> None:
        """Unlock stone huts once the colony has been content for long enough.

        The comparison is ``>=`` and that is load-bearing. morale is 1 = good,
        while hunger, fatigue and warmth are 1 = bad (entities.py:429-432), and
        both conventions live in this package. A ``<=`` here produces a gate
        that opens on a miserable colony and never on a happy one - and it would
        still fire, still latch, and still read perfectly plausibly in the
        chronicle, because there is no crash to catch it. The acceptance test is
        the clock: measured over 26 seeds, the earliest honest unlock is 15.0
        sim-minutes, so a tier that appears in the first few minutes of a fresh
        seed means the comparison is backwards.

        The dwell must be UNBROKEN - one tick below the threshold puts it back
        to zero. That is what keeps the opening honeymoon from handing the tier
        over for free: agents spawn at morale 0.6 with every need at zero, so
        colony mean sits above 0.55 for the first 47-132 s of every seed
        measured, and any dwell at or under ~140 s unlocks during it.

        Latched: once True it is never set False again. A later famine does not
        un-teach masonry. An unlatched gate would also be actively worse than no
        tier at all - colony morale is noisy at these headcounts (2-10 agents)
        and crosses 0.55 constantly, so huts would flicker between timber and
        stone as it wobbled.

        Accumulating ``dt`` rather than counting whole seconds is deliberate: the
        measurement that chose 180 s sampled once per sim-second, so a per-tick
        accumulator is strictly finer-grained and cannot unlock any earlier than
        the sampled model predicted.
        """
        if self.hut_tier_unlocked:
            return
        m = self.colony_morale()
        self.hut_tier_dwell = (self.hut_tier_dwell + dt) if m >= HUT_TIER_MORALE else 0.0
        if self.hut_tier_dwell >= HUT_TIER_DWELL_SEC:
            self.hut_tier_unlocked = True
            self.log_event("They have learned to raise walls of stone.")

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

    def deaths_ever(self) -> int:
        """Everyone who has ever died, including bodies not yet buried.

        ``stats["died"]`` is NOT that number: it is only incremented once a
        corpse has lain BURIAL_DELAY seconds and been given a headstone, so for
        those first two seconds the dead are missing from ``died`` *and* from
        ``alive_agents()`` at the same time. Any sum built on stats["died"]
        alone therefore reads one short whenever it is taken in that window -
        which for a measurement harness that stops on an arbitrary tick is a
        coin flip. The unburied corpses still on the roster close the gap
        exactly, because ``_buried`` is set on precisely the same line that
        increments the stat.
        """
        try:
            pending = sum(1 for a in self.population.agents
                          if not a.alive and not a.__dict__.get("_buried"))
        except Exception:
            pending = 0
        return int(self.stats.get("died", 0)) + pending

    def reconcile(self) -> dict[str, int]:
        """The population books and the residual, which must always be zero.

            births + returned - deaths - abducted - alive == 0

        Every way onto the roster and every way off it, as counters, so the
        colony's headcount is fully explained at any instant:

        * ``births``   - Population.births, incremented by every ``spawn()``,
                         which includes the founding group. Nothing external
                         has to remember a "starting population".
        * ``returned`` - agents the ufo put *back* on the roster, alive. Not a
                         birth: they already existed and are already in
                         ``births`` from the first time round.
        * ``deaths``   - :meth:`deaths_ever`, not stats["died"]; see there.
        * ``abducted`` - agents the ufo took *off* the roster, alive. Not a
                         death: no grave, no death hook, and a quarter of them
                         come back. This is the term that did not exist, and
                         its absence is what made every abduction read as an
                         unexplained loss and every balance measurement taken
                         through it read as one death too many.
        * ``alive``    - who is actually standing there now.

        ``ok`` is 1 when the figures were actually readable. It exists because
        this is guarded - the tray may call it on a half-built world - and a
        bare ``except`` that returned a residual of 0 would report perfect books
        for a world it could not read at all, which is precisely the kind of
        silent pass this method was written to stop. Assert on ``ok`` as well as
        ``residual``.

        Side-effect free and not on any per-tick path.
        """
        try:
            births = int(getattr(self.population, "births", 0))
            alive = len(self.population.alive_agents())
            deaths = self.deaths_ever()
            abducted = int(self.stats.get("abducted", 0))
            returned = int(self.stats.get("returned", 0))
            held = len(getattr(self.ufo, "abducted", None) or ())
        except Exception:
            log.debug("reconcile failed", exc_info=True)
            return {"ok": 0, "residual": 0}
        return {
            "ok": 1,
            "births": births,
            "returned": returned,
            "deaths": deaths,
            "abducted": abducted,
            "alive": alive,
            "held_by_ufo": held,
            "residual": births + returned - deaths - abducted - alive,
        }

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
            "dragons": self.dragons.to_dict(),
            "spears": self.spears.to_dict(),
            "ufo": self.ufo.to_dict(),
            "auto_scene_rotate": bool(getattr(self, "auto_scene_rotate", True)),
            "stockpile": dict(self.stockpile),
            "build_queue": list(self.build_queue),
            "chronicle": list(self.chronicle),
            "stats": dict(self.stats),
            # Additive, so SAVE_VERSION stays 1 and persist.py never quarantines
            # anything: from_dict reads both with .get(key, default), and a build
            # old enough not to know them ignores them. The partial dwell is
            # saved alongside the latch so a reload never costs more than the
            # tick it happened on - saving only the latch would silently reset a
            # colony 179 seconds into earning it, every single save.
            "hut_tier_unlocked": bool(self.hut_tier_unlocked),
            "hut_tier_dwell": float(self.hut_tier_dwell),
            # Additive for the same reason, and with no partner field: the
            # dragon gate is a pure latch with no dwell to lose, so there is
            # nothing here that a reload could reset half-way through earning.
            "dragons_unlocked": bool(self.dragons_unlocked),
            # upgrade_job is NOT here. It is derived from the chosen hut's
            # state["upgrade"] and republished on the first director pass after
            # a load; persisting it would create a second copy that can disagree
            # with the structure it describes.
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
        # Every colony-level default, in one call shared with __init__, so a
        # field can no longer exist on a fresh world and be missing on a loaded
        # one. See _init_colony_state for what that cost last time. Saved values
        # go over the top of these below.
        w._init_colony_state()
        # The stone-hut tier, straight after the defaults it overwrites. A save
        # written before the tier existed carries neither key and simply loads
        # locked with an empty accumulator, then re-earns it - the "growth"
        # precedent at structures.py:1174-1177.
        #
        # Everything here is coerced defensively rather than trusted, because
        # from_dict is the one function in this file that must not raise:
        # persist.load_world treats an exception as a corrupt save and
        # QUARANTINES it, so a single junk value in a hand-edited file does not
        # lose the tier, it loses the colony. `or 0.0` was enough for null and
        # nothing else - "banana" raised ValueError and a list or dict raised
        # TypeError. nan/inf are rejected too: the accumulator does self-heal
        # (the next sub-threshold tick resets it to 0.0), but an inf sitting
        # above HUT_TIER_DWELL_SEC unlocks the tier on the first tick, which is
        # not what a corrupt file should be able to buy.
        w.hut_tier_unlocked = bool(d.get("hut_tier_unlocked", False))
        w.hut_tier_dwell = _finite(d.get("hut_tier_dwell"), 0.0,
                                   lo=0.0, hi=HUT_TIER_DWELL_SEC)
        # The dragon gate, on the same terms. A save written before dragons
        # existed carries no key at all and simply loads locked - and then
        # re-latches on the first tick if its stone huts are already up, which
        # costs nothing because the registry's own DRAGON_FIRST_DELAY starts
        # from that moment. bool() rather than a truthiness test on the raw
        # value so "false", 0 and [] all coerce the way a hand-edited file would
        # be read anywhere else in this method.
        w.dragons_unlocked = bool(d.get("dragons_unlocked", False))
        # Persisted so a user who turns rotation off keeps it off across a
        # restart; the default above is what an older save falls back to.
        w.auto_scene_rotate = bool(d.get("auto_scene_rotate", True))

        def _sub(key, loader, fallback):
            # An ABSENT section and an UNUSABLE one are different events and used
            # to log the same alarming line. Every save written before dragons
            # existed lacks "dragons" and "spears", so loading a perfectly good
            # older colony printed "save section 'dragons' unusable" - which
            # reads as corruption, is the first thing anyone would investigate
            # after a crash, and is wrong. Missing is the ordinary case for any
            # subsystem added after a save was written; only a section that is
            # PRESENT and unreadable deserves a warning.
            if key not in d:
                log.debug("save section %r absent; regenerating", key)
                return fallback()
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
        # Regenerated subsystems are seeded from the loaded world, with the same
        # offsets __init__ uses, so a section that had to be rebuilt behaves like
        # a fresh one instead of like an unseeded one. A bare EventSystem() or
        # AnimalRegistry() draws its seed from the *global* random module, so a
        # reload whose section was missing or corrupt came back permanently
        # non-reproducible - and silently, because _sub logs the regeneration but
        # nothing downstream can tell a seeded stream from an unseeded one. The
        # ufo already had this; events and animals did not.
        w.events = _sub("events", EventSystem.from_dict,
                        lambda: EventSystem(seed=w.seed ^ 0x513))
        w.animals = _sub("animals", AnimalRegistry.from_dict,
                         lambda: AnimalRegistry(seed=w.seed ^ 0xA17))
        w.ufo = _sub("ufo", Ufo.from_dict, lambda: Ufo(seed=w.seed ^ 0x71F))
        # Both new registries go through _sub with a SEEDED fallback, and the
        # seeding is load-bearing rather than tidy: a bare DragonRegistry() or
        # SpearRegistry() draws its seed from the global random module, so a
        # reload whose section was missing or corrupt would come back
        # permanently non-reproducible - silently, because _sub logs the
        # regeneration but nothing downstream can tell a seeded stream from an
        # unseeded one. That is the exact bug already fixed here for events and
        # animals; this is the same shape, so it gets the same treatment on the
        # way in rather than after somebody spends a day on it.
        #
        # A save written before either existed has neither key, so _sub's
        # KeyError sends both down the fallback and the world loads with no
        # dragons and no spears in flight - which is precisely right.
        w.dragons = _sub("dragons", DragonRegistry.from_dict,
                         lambda: DragonRegistry(seed=w.seed ^ 0xD4A))
        w.spears = _sub("spears", SpearRegistry.from_dict,
                        lambda: SpearRegistry(seed=w.seed ^ 0x59EA))

        w.stockpile = {r: 0 for r in ALL_RESOURCES}
        w.stockpile.update({k: int(v) for k, v in
                            (d.get("stockpile") or {}).items() if k in ALL_RESOURCES})
        w.build_queue = list(d.get("build_queue") or [])
        w.chronicle = deque(d.get("chronicle") or [], maxlen=CHRONICLE_MAX)
        # Defaults first, saved values over the top: a save written before
        # "abducted"/"returned" existed simply keeps them at zero instead of
        # loading a stats dict that KeyErrors the moment anything reads them.
        saved_stats = d.get("stats") or {}
        w.stats = dict(STAT_DEFAULTS)
        w.stats.update({k: int(v) for k, v in saved_stats.items()})

        # ...but zero is only right if the lights had never been. A save from
        # before the counter existed carries no history, yet the ufo section
        # still remembers the abductees it is holding - and those are exactly
        # the people who are alive, off the roster, and subtracted nowhere. Seed
        # from them so an old save reconciles instead of reading one head over
        # for the rest of its life.
        #
        # Two caveats, both unavoidable and both harmless: this is a NET figure
        # rather than a history (a legacy save cannot say how many were ever
        # taken, only how many are still up there), and anyone who fell off the
        # end of Ufo.abducted at MAX_ABDUCTED is gone from the record entirely.
        # A held record whose agent IS on the roster is a delivery caught
        # mid-beam - already handed back, so not off the books - and must not
        # be counted, or the save reconciles one short in the other direction.
        if "abducted" not in saved_stats:
            try:
                held = getattr(w.ufo, "abducted", None) or ()
                w.stats["abducted"] = sum(
                    1 for r in held
                    if w.population.by_id(int(r.get("id", 0))) is None)
            except Exception:
                log.debug("could not seed abducted from a legacy save",
                          exc_info=True)

        # The "already buried" flag lives in the corpse's __dict__ and is not
        # part of Stickman.to_dict, so it does not survive a save - but dead_t
        # does. Without this, any body saved between BURIAL_DELAY and its
        # retirement off the roster is buried a *second* time on load: a
        # duplicate headstone, and stats["died"] incremented twice for one
        # person. The saved stats["died"] already counts them, so re-flag.
        for a in w.population.agents:
            try:
                if not a.alive and a.dead_t >= BURIAL_DELAY:
                    a.__dict__["_buried"] = True
            except Exception:
                continue

        if not w.population.alive_agents():
            log.info("loaded world had no survivors; seeding a new group")
            w._seed_population()
        return w
