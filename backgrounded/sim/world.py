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
    HERMIT_STASH_CAP,
    HUT_TIER_DWELL_SEC, HUT_TIER_MORALE,
    MAX_POP, MIN_POP, MORALE_TO_GROW, MORPH_LABELS,
    POP_BIRTH_COOLDOWN, POP_PER_HUT,
    REGROW_MAX, REGROW_PER_HEAD, RENDER_H,
    RES_COOKED, RES_FOOD, RES_STONE, RES_WOOD, SAVE_VERSION,
    SCENE_LABELS, SCENE_NIGHT_STORM, SCENE_ROTATE_SEC, SCENES,
    STAGE_HALF,
    TORCH_COLOR, TORCH_FLICKER, TORCH_INTENSITY, TORCH_RADIUS,
    WORLD_SCALE, WORLD_W,
)
from . import behavior, names
from .actions import stage_bounds, sweep_claims
from .animals import AnimalRegistry
from .dragons import DragonRegistry
from .items import RelicRegistry
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
#:
#: 10 -> 20 with MAX_POP 10 -> 20. This is driven by the ROSTER, not by the map:
#: twice the colonists is twice the funerals over the same run, and a cemetery
#: that recycles twice as fast erases names the player still remembers. The 4x
#: map is a secondary argument (they spread further so they read as less of a
#: cemetery) but not the reason. Knock-on recorded at constants.DRAGON_SCOUR_MAX,
#: which is sized as a fraction of this.
MAX_GRAVES = 20

#: Scenery per map, and therefore a count that MUST scale with the map: on a
#: 6400 px world the old {tree 14, rock 8, bush 10, boulder 3} is a quarter of
#: the density it was tuned at, and since foraging is the colony's food supply
#: that is not "a sparser look", it is a starving colony.
#:
#: One constant because there used to be two identical literals - __init__ and
#: randomise_terrain - and a duplicated tuning number is a number that drifts.
#: (props.DEFAULT_COUNTS is a THIRD copy, used only when scatter() is called
#: with counts=None, which no gameplay path does; it wants the same x4 and is
#: flagged for whoever owns props.py.)
SCATTER_COUNTS: dict[str, int] = {
    kind: int(round(n * WORLD_SCALE))
    for kind, n in (("tree", 14), ("rock", 8), ("bush", 10), ("boulder", 3))
}

#: How far founders scatter around their landing seat, px. The colony arrives as
#: a GROUP - it always did, but on a 1600 px world "uniform(0.2W, 0.8W)" was a
#: 960 px spread that a walker crosses in half a minute, so they converged
#: anyway and nobody noticed the spread was doing the work. At WORLD_W that same
#: expression is 3840 px: four strangers who never meet, no colony, no
#: structures, and colony_center() a mean over noise - which then presents as a
#: broken camera and gets debugged in the wrong place. Pick one seat, stand them
#: around it, and the old emergent behaviour is now the explicit rule.
FOUNDER_SCATTER = 120.0

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
    # The dragon relics, declared here at zero for exactly the reason the two
    # lines above are. items.py bumps these through a `.get()`-style helper that
    # only writes a key that already exists, so a counter conjured on first use
    # would be missing on every world where no dragon has died yet - and the HUD
    # readers would each have to remember .get() or KeyError on a fresh colony.
    #
    # "raised" is a separate line from "born" ON PURPOSE. A cairnstone raising is
    # not a birth: reconcile() reads population.births (which Population.spawn
    # bumps for us, cancelling against the roster entry so the residual stays
    # zero by construction), while stats["born"] is a display counter the player
    # reads as "children". Adding a raising to it would say the colony had a baby
    # every time it dug one back up.
    "relics_found": 0, "relics_used": 0, "raised": 0,
    # Visits paid to the hermit. Declared at zero for the third-time-paid-for
    # reason spelled out below: actions._arrive_at_hermit bumps it with a
    # .get()-style write that would CONJURE the key on the first visit, so on
    # every colony that has never paid one it would be a KeyError rather than a
    # zero - and a counter that does not exist until the event happens cannot be
    # used to prove the event never happened, which is exactly what an A/B run
    # of this feature needs it for. Not a display counter; nothing in the HUD
    # reads it.
    "hermit_visits": 0,
    # The wyrm-gun's two counters, and they are here for the third time this
    # lesson has been paid for. combat_actions._bump_stat writes them with
    # ``st[key] = int(st.get(key, 0) or 0) + 1``, which CONJURES the key on the
    # first shot - so on every world where nobody has ever pulled the trigger,
    # ``stats["bfg_shots"]`` is a KeyError rather than a zero. That is exactly
    # the shape of the "abducted" bug two lines above: the readers that forgot
    # .get() did not crash loudly, they quietly left the number out, and a
    # balance measurement taken through them was wrong for hours before anyone
    # noticed. A counter that does not exist until the event happens cannot be
    # used to prove the event never happened - which is the single most useful
    # thing these two are for, since a hostile proof of this feature measured
    # bfg_shots == 0 across 16 seeds and that zero had to be read as "inert",
    # not "unread".
    #
    # NOT display counters. Nothing in the HUD reads them; they exist so a
    # A/B run can assert the gun fired at all.
    "bfg_shots": 0, "bfg_misfires": 0,
    # ...and the other two the same trigger writes. These were missed rather
    # than deferred - they are the same family, bumped by the same function in
    # the same file as the two above, and they are the pair that answers the
    # question the user actually asked ("can one shot leave a single stickman
    # standing?"). A hostile proof reading stats["bfg_heedless"] on a colony
    # where nobody had fired would have got a KeyError, not a zero, which is the
    # third time this exact shape has cost this project a measurement.
    "bfg_heedless": 0, "bfg_friendly_deaths": 0,
    # DELIBERATELY NOT DECLARED, and recorded here so the next hand does not
    # think they were forgotten: "animals_killed", "spears_thrown",
    # "spears_thrown_child", "spears_hit", "spears_missed", "spears_recovered",
    # "throw_damage", "animals_seen", "last_animal_t". All are conjured on first
    # use by their own subsystems and all have the same latent bug. They are not
    # added because (a) they belong to measurements this lane does not own and
    # declaring them changes what a fresh world's stats dict contains under
    # somebody else's test, and (b) "last_animal_t" is a TIMESTAMP, not a
    # counter - a default of 0 there would mean "an animal was seen at world
    # time zero", which is worse than absent. Whoever owns throwing.py and
    # combat_actions.py should close the rest; this dict is the right home for
    # them when they do.
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


def _int(value: Any, default: int, lo: int, hi: int) -> int:
    """A whole number in ``[lo, hi]`` from anything at all, never raising.

    :func:`_finite`'s companion, for the fields :meth:`World.from_dict` needs
    as ints, and it exists because the bare ``int()`` calls it replaces were
    every bit as fragile as the bare ``float()`` calls above them. ``int()``
    raises on a string, on ``None``, on a list, on a dict, on NaN (ValueError)
    and on both infinities (OverflowError) - and each of those was reachable
    from one hand-edited scalar in a save file, which persist.load_world reads
    as corruption and QUARANTINES. Routing through :func:`_finite` first makes
    the conversion total, and the clamp then keeps the value inside the range
    the field is actually allowed to hold: ``seed`` in particular must be a
    NON-NEGATIVE int or ``np.random.default_rng`` raises ValueError, which is
    the one case here that junk-typing alone would not have caught.
    """
    return int(_finite(value, float(default), lo=float(lo), hi=float(hi)))


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
            dict(SCATTER_COUNTS),
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
        # What the dragons leave behind. Its own seeded stream, offset like every
        # registry above it, and the single source of randomness for the WHOLE
        # relic feature - the skeletal's 50/50 drop, BFG_MISFIRE, BFG_HEEDLESS.
        # Sharing pyrng would be the expensive mistake here rather than a tidy
        # one: behavior.choose_action draws from pyrng once per scored candidate,
        # and adding a single inert candidate to that stream moved mean deaths
        # 11.33 -> 13.17 (+16%) with 0/24 seeds matching on any metric. A misfire
        # roll landing in the middle of it would re-phase every wolf spawn in the
        # colony, and only on the worlds that had earned a relic.
        #
        # A registry and not a plain field, so it does NOT go in
        # _init_colony_state: from_dict loads it through _sub with a seeded
        # fallback, exactly as dragons and spears are loaded, which is what makes
        # it present on both construction paths. See from_dict.
        self.relics: RelicRegistry = RelicRegistry(seed=self.seed ^ 0x2E11)

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
        # list[dict], NOT list[str]. behavior._publish writes whole job records
        # here - see _queue_of, which types it list[dict[str, Any]] - and this
        # annotation said ``list[str]`` for long enough to mislead a defensive
        # filter in from_dict into dropping every real entry on load.
        self.build_queue: list[dict[str, Any]] = []
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

        # The hermit's visit slot and the clock that opens it. Declared HERE for
        # the reason this whole method exists: behavior._tick_hermit_visit rides
        # inside _guarded("colony"), so on a loaded world a missing attribute
        # would not crash - it would raise once, disable the colony subsystem for
        # the session, and quietly take update_director and assign_roles down
        # with it. Nothing visible would say why.
        #
        # Neither is serialised, and both are transients rather than state:
        # `hermit_visit` describes an action that is already saved on the
        # visitor himself (a save taken mid-visit reloads with him simply
        # walking home), and `_hermit_visit_due` is a countdown that costs at
        # most one extra visit to restart. A saved copy of either would be a
        # second source of truth, which is the argument `upgrade_job` above
        # makes for itself.
        self.hermit_visit: dict[str, Any] | None = None
        self._hermit_visit_due: float = 0.0

        # THE HERMIT'S PRIVATE STASH - his camp's store, not the colony's.
        # Declared HERE and not in __init__ for exactly the reason this method
        # exists: `actions.hermit_stash_of` reads it on the gather/build/stoke
        # paths, all of which ride inside guarded ticks, so a field that existed
        # only on the __init__ path would not crash a loaded world - it would
        # raise once, disable a subsystem for the session, and quietly leave the
        # hermit unable to bank or spend anything for the rest of the run, with
        # nothing visible saying why. That is the exact failure `auto_scene_rotate`
        # shipped with.
        #
        # WORLD-LEVEL, KEYED TO THE CAMP, rather than hung off the hut. The hut
        # is the more dramatic owner - a dragon burning it would take the winter
        # stores too - but he has no hut for the first minutes of his life, none
        # while a burned one is rubble, and none across the tick he strikes camp
        # and re-stakes. A store that vanishes at the three moments he most needs
        # one is not a store. Here the successor inherits it exactly as he
        # inherits the hut, because the camp outlives the man.
        #
        # It is NOT `stockpile` and it must never be summed with it: the colony
        # cannot see it and he cannot draw on theirs. `World.reconcile` conserves
        # PEOPLE and not resources, so there is no accounting hazard - the gate
        # is about the role, not the arithmetic.
        self.hermit_stash: dict[str, int] = {r: 0 for r in ALL_RESOURCES}

        # Things the player's Hand tool is currently carrying, by id. Transient
        # by design: never serialised, so a save mid-drag reloads with nothing
        # held and no one can end up frozen in the air across a restart.
        self.held_agent_ids: set[int] = set()
        self.held_prop_ids: set[int] = set()

    # ------------------------------------------------------------ startup --
    def _pick_style(self) -> str:
        return str(self.rng.choice(["hills", "cliffs", "plateau", "chasm", "valley"]))

    def _landing_seat(self) -> float:
        """One x for a whole group to arrive on.

        Kept a full STAGE_HALF in from each rim so the camera, which is
        RENDER_W wide and centred on them, has world on both sides and does not
        open the run already clamped against an edge.

        Rejects seats in the chasm. This did not need saying while the group
        spread over 960 px of a 1600 px world - somebody always landed on a rim.
        A single seat plus FOUNDER_SCATTER is 240 px wide, narrower than a cut
        plus its walls, so the whole colony can now start on the chasm floor
        with unclimbable rock on both sides: not a bad opening, a dead one. Rare
        (a cut is ~120 px of 4800 eligible, on the one style in five that has
        one) which is exactly why it would ship.
        """
        lo = float(STAGE_HALF)
        hi = float(WORLD_W) - float(STAGE_HALF)
        if hi <= lo:                       # only if WORLD_W ever shrank to <= RENDER_W
            return float(WORLD_W) * 0.5
        seat = float(self.rng.uniform(lo, hi))
        try:
            gap = self.terrain.chasm
            if gap:
                # Half the walls plus the scatter plus a walker's room. Bounded
                # retries, then take what we have: a guaranteed-terminating loop
                # matters more here than a perfect seat, and the fallback is only
                # as bad as the behaviour this method replaced.
                pad = 200.0 + FOUNDER_SCATTER
                for _ in range(8):
                    if not (gap[0] - pad < seat < gap[1] + pad):
                        break
                    seat = float(self.rng.uniform(lo, hi))
        except Exception:
            pass
        return seat

    def _place(self, agent: Any, x: float) -> None:
        """Stand an agent on the ground at x, with nothing left over from before."""
        agent.x = float(min(max(x, 8.0), float(WORLD_W) - 8.0))
        agent.y = self.terrain.ground_y(agent.x)
        agent.vx = agent.vy = 0.0
        agent.on_ground = True

    def _seed_population(self, n: int = 4) -> None:
        used: set[str] = set()
        seat = self._landing_seat()
        for i in range(n):
            x = float(seat + self.rng.uniform(-FOUNDER_SCATTER, FOUNDER_SCATTER))
            x = min(max(x, 8.0), float(WORLD_W) - 8.0)
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
        # Immediately after "dragons" and before "ufo", and the order is
        # load-bearing rather than aesthetic. DragonRegistry.slain is a transient
        # list raised by _reap on the frame a dragon dies and drained here; put
        # this line above "dragons" instead and every drop is served a frame
        # late, which is survivable, but a save taken in that gap carries a
        # phantom kill that spawns its relic a second time on load. Draining in
        # the same frame it is raised is what makes `slain` safe to leave out of
        # to_dict. Before "ufo" because the amulet is a relic the saucer has to
        # be able to see this frame.
        self._guarded("relics", lambda: self.relics.tick(self, dt))
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
            # The generation goes onto the headstone. It was being dropped on the
            # floor here - add_grave has taken one since it was written and this
            # call never passed it, so every grave in the world read generation 0
            # - and a cairnstone raising reads it straight back off the stone to
            # put the same person back with the same number. Without this, anyone
            # raised is a first-generation founder no matter who they were.
            self.props.add_grave(agent.x, self.terrain.ground_y(agent.x),
                                 agent.name, getattr(agent, "generation", 0))
            # Whatever they were carrying falls where they fell. Here rather than
            # in Stickman.die() because this is the ONE funnel every death goes
            # through whatever killed them, and it is already one-shot per corpse
            # behind the _buried flag above - so a relic cannot be dropped twice
            # and duplicated. drop_from() never raises and returns None for the
            # overwhelmingly common case of a colonist who owned nothing.
            #
            # The one relic this does NOT recover is a BFG lost to a misfire:
            # combat_actions destroys the weapon BEFORE it deals the blast damage
            # precisely so a wielder who kills themselves cannot re-drop the gun
            # here. That asymmetry is the BFG's balance valve - it is the only
            # item in the set that can leave the world.
            #
            # getattr rather than self.relics, and this is not defensive
            # decoration: _reap_dead runs inside _guarded("agents"), the single
            # most load-bearing guard in the file. An AttributeError here would
            # not lose a relic, it would disable ageing, hunger, burial and
            # replacement for the rest of the session - the exact failure
            # _init_colony_state's docstring was written about. Both construction
            # paths set self.relics, so this branch should be unreachable; it
            # costs one lookup a death to make sure a wrong assumption about that
            # stays cheap.
            _relics = getattr(self, "relics", None)
            if _relics is not None:
                _relics.drop_from(self, agent)
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
            self.props = scatter(self.terrain, self.rng, dict(SCATTER_COUNTS))
            self.structures = StructureRegistry()
            self.build_queue = []
            # One seat for the whole colony, same rule as the founders. They
            # wake up somewhere new *together*: scattering them over 0.15-0.85
            # of a 6400 px world is 5440 px of separation and the colony that
            # was supposed to carry over does not survive the move.
            #
            # This is also the single case Camera.SNAP_DIST exists for - every
            # agent relocates in one frame, and easing a camera across 5000 px
            # at 220 px/s would crawl for 25 seconds.
            seat = self._landing_seat()
            for a in self.population.agents:
                self._place(a, seat + self.rng.uniform(-FOUNDER_SCATTER, FOUNDER_SCATTER))
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
        # A newcomer joins a colony that already exists, so they arrive where it
        # can be seen happening: uniform(0.1W, 0.9W) meant "somewhere on screen"
        # when the world was one screen, and would now drop a replacement up to
        # 3000 px from anyone, walking in unwatched for a minute and a half.
        x = float(self.rng.uniform(*stage_bounds(self)))
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

    def raise_the_dead(self, grave: Any, by: str = "") -> bool:
        """The cairnstone: put the person under *grave* back on the roster.

        *by* is the bearer's name and is optional purely so the one-argument call
        in the contract still works. Pass it: with it the chronicle gets the line
        that names both people, which is the line worth screenshotting, and
        without it all the world can honestly say is that somebody came back.

        Returns True only if somebody actually stood up. **The caller consumes
        the relic on True and keeps it on False** - a refusal must not eat the
        item, or a bearer who walks to a headstone while the colony happens to be
        full loses the rarest half of the skeletal's drop table for nothing.
        Consuming here instead would hide that decision inside a method that has
        four separate ways to decline.

        THE LEDGER. This needs no new term in :meth:`reconcile`. That residual is
        ``births + returned - deaths - abducted - alive``; ``Population.spawn``
        bumps ``population.births`` and the roster gains one entry, so the two
        cancel and the residual stays zero by construction. What it must NOT do
        is touch ``stats["born"]`` - that is the display counter the player reads
        as children, and a raising is not a birth. ``stats["raised"]`` is its own
        line in STAT_DEFAULTS.

        THE CAP. ``len(alive_agents()) >= MAX_POP`` is checked FIRST and refuses.
        Not optional and not theoretical: the ufo's return path shipped exactly
        this bug once by adding straight to the roster and peaked a long run at
        11 against a MAX_POP of 10. Every growth path in this file funnels
        through a cap check; this is one more of them.
        """
        try:
            if grave is None or getattr(grave, "kind", "") != "grave":
                return False
            if not getattr(grave, "alive", True):
                return False          # already weathered away, or already raised
            if len(self.population.alive_agents()) >= MAX_POP:
                return False

            state = getattr(grave, "state", None) or {}
            if not isinstance(state, dict):
                return False
            dead_name = str(state.get("name") or "").strip()
            # Coerced through _finite rather than int()/float() directly, and not
            # out of tidiness: a headstone whose state was hand-edited to
            # {"generation": "old"} would raise inside the try below, and the
            # except branch logs a full traceback. An action that re-scores this
            # grave every couple of seconds would then write one traceback per
            # attempt into a log file nobody is watching, forever. Junk should
            # decline quietly, not shout.
            gen = int(_finite(state.get("generation"), 1.0, lo=0.0, hi=1e6)) or 1
            # The grave's own x, wherever on the LAND it stands - a raising puts
            # the colonist back where they were buried, which is the point of
            # the item. Bounds check only.
            x = _finite(getattr(grave, "x", None), -1.0,
                        lo=0.0, hi=float(WORLD_W))
            if x < 0.0:
                return False
            if not dead_name:
                # A headstone with nobody's name on it. Older saves have these -
                # _reap_dead only started passing the generation through with
                # this change, and a hand-edited prop can carry anything. Nothing
                # to raise, so refuse rather than inventing a stranger: the whole
                # point of the item is that the player recognises the name.
                return False

            # Population.spawn only honours a requested name if it is NOT already
            # in used_names, and the dead keep their names reserved forever so no
            # two colonists are ever confused for each other. Discard it for the
            # length of this one call, or the raising comes back under a brand
            # new name and the item silently stops being what it says it is.
            # spawn() re-adds it on the way through.
            self.population.used_names.discard(dead_name)
            s = self.population.spawn(
                x=x, y=self.terrain.ground_y(x),
                rng=self.pyrng,
                name=dead_name,
                generation=gen,
                birth_time=self.world_time,
            )
            if s.name != dead_name:
                # spawn() fell back to a fresh name anyway (a live colonist is
                # somehow already called this). Put the reservation back so the
                # buried name stays spoken for, and let the new arrival stand -
                # rolling the spawn back would be a second, worse surgery on the
                # roster than living with a stranger.
                self.population.used_names.add(dead_name)

            self.props.remove(grave)
            self.stats["raised"] += 1
            bearer = str(by or "").strip()
            if bearer and bearer != s.name:
                self.log_event(f"{bearer} sets the wyrm's stone on the grave, "
                               f"and {s.name} draws breath.")
            else:
                self.log_event(f"The wyrm's stone goes down on a grave, and "
                               f"{s.name} draws breath.")
            return True
        except Exception:
            # Never fatal: this is reached from an action handler, which runs
            # inside _guarded("agents"). Losing a raising is a bad afternoon;
            # losing the agents subsystem is the colony.
            log.exception("raise_the_dead failed")
            return False

    def spawn_visitor(self, x: float) -> str | None:
        """Player's Spawn tool: a newcomer appears at x, if there is room."""
        from ..constants import MAX_POP
        if len(self.population.alive_agents()) >= MAX_POP:
            return "The land is already full."
        used = {a.name for a in self.population.agents}
        # The player clicked here, so this is a plain WORLD clamp and stays one:
        # narrowing it to the stage would move a deliberate spawn away from the
        # spot that was pointed at.
        gx = float(max(WORLD_W * 0.03, min(WORLD_W * 0.97, x)))
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
            elif (getattr(agent, "holds_torch", True)
                  and getattr(agent, "role", "") != names.ROLE_HERMIT):
                # The hermit carries no light, and it is decided HERE rather than
                # by clearing holds_torch on appointment. render/stickfigure.py
                # already declines to draw his flame, so the two halves have to
                # agree; a flag toggled at appointment has to be un-toggled on
                # every demotion path (death, the population floor, a UFO return
                # handing back a stale role) and one missed path leaves a light
                # with no source - which is exactly the bug this replaces, a
                # moving pool of torchlight lighting nobody.
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
        #
        # "hut" IS THE EXACT KIND STRING AND IT MUST STAY THAT WAY. This is the
        # birth gate: it turns roofs into a population cap, and it is the site
        # the note on `structures.hermit_hut` calls out as the one that fails
        # SILENTLY if a building nobody in the colony lives in is counted here.
        # The hermit's hut (`hermit_hut`) and his fire (`hermit_fire`) are
        # separate kinds precisely so this line cannot see them - a hut
        # counted here would raise the colony's cap by POP_PER_HUT on a house
        # with one recluse in it who does not breed, and nothing would error.
        # Do not widen this to a kinds tuple or a "has capacity" test.
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
            # Ground relics and the relic stream's draw count. What a colonist is
            # WEARING is not here: that rides on Stickman.to_dict, in entities.py.
            "relics": self.relics.to_dict(),
            "ufo": self.ufo.to_dict(),
            "auto_scene_rotate": bool(getattr(self, "auto_scene_rotate", True)),
            "stockpile": dict(self.stockpile),
            # His camp's store, kept apart from the colony's above it. Additive
            # like the tier latches below, so SAVE_VERSION stays 1 and a build
            # old enough not to know the key simply ignores it.
            "hermit_stash": dict(getattr(self, "hermit_stash", {}) or {}),
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
        # THE PLAIN SCALARS ARE COERCED, NOT TRUSTED - see _int. These four
        # lines used to be bare int()/float() calls sitting right beside the
        # carefully guarded sections below them, and they were the whole of the
        # docstring's promise being broken: eight distinct junk values across
        # these fields raised out of from_dict, and persist.load_world reads any
        # exception as a corrupt save and QUARANTINES it. One junk scalar in a
        # hand-edited file therefore did not cost the field, it cost the colony.
        #
        # The ranges are the ones each field is actually allowed to hold rather
        # than decoration. `seed` must land in [0, 2**32) because
        # np.random.default_rng REJECTS a negative seed - `"seed": -1` raised
        # ValueError("expected non-negative integer") straight out of the line
        # below, which is a failure no amount of type-checking would have found.
        # world_time and tick_count are monotone counters, so a negative one is
        # meaningless and clamps to zero; their ceilings sit well inside the
        # range a float can hold an integer exactly (2**53), because _finite
        # goes through float on the way.
        w.seed = _int(d.get("seed"), random.randrange(1 << 30),
                      lo=0, hi=(1 << 32) - 1)
        w.rng = np.random.default_rng(w.seed)
        w.pyrng = random.Random(w.seed ^ 0x5EED)
        w.world_time = _finite(d.get("world_time"), 0.0, lo=0.0, hi=1e12)
        w.tick_count = _int(d.get("tick_count"), 0, lo=0, hi=1_000_000_000_000)
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
        # Relics, on the same terms and for the same reasons. Every save written
        # before this feature existed has no "relics" key at all, and _sub treats
        # ABSENT as the ordinary case: it logs at DEBUG and regenerates, without
        # the "save section unusable" WARNING that reads as corruption. That
        # distinction was added to _sub precisely because loading a perfectly good
        # pre-dragon colony used to print an alarming line; do not undo it by
        # reaching for d["relics"] here.
        #
        # The fallback is SEEDED, which is the load-bearing half: a bare
        # RelicRegistry() draws its seed from the global random module, so a world
        # whose section was missing or unreadable would come back permanently
        # non-reproducible and nothing downstream could tell. That bug has been
        # fixed in this file three times now (events, animals, then dragons and
        # spears); this is the same shape, so it gets the same treatment on the
        # way in. Same 0x2E11 offset as __init__.
        #
        # ...and the LOADER is seeded too, which is where this differs from the
        # four lines above it and is worth the extra lambda. RelicRegistry
        # .from_dict is defensive by contract - it never raises, it returns an
        # empty registry - so _sub's except branch is unreachable for this
        # section and the seeded fallback never actually runs. A hand-edited
        # `"relics": "banana"` would therefore load a registry seeded by crc32 of
        # the junk instead of by the world, and be silently on a different stream
        # for the rest of its life. Threading w.seed ^ 0x2E11 through makes the
        # seed come from the world in every case; it is the same value the save
        # carries on any honest load, so nothing changes for a real one.
        w.relics = _sub("relics",
                        lambda raw: RelicRegistry.from_dict(
                            raw, seed=w.seed ^ 0x2E11),
                        lambda: RelicRegistry(seed=w.seed ^ 0x2E11))

        # The three plain containers, on the same terms as the scalars above and
        # for the same reason. ``or {}`` / ``or []`` only ever rescued the FALSY
        # junk - None, 0, "" - and left every truthy shape to raise: a
        # "stockpile" that is a string or a number died on .items()
        # (AttributeError), and a "build_queue" or "chronicle" that is a number
        # died on the iteration (TypeError). A container that is the right shape
        # but carries junk INSIDE it was unguarded too, and that is the half the
        # top-level junk sweep cannot see: ``{"wood": "banana"}`` is a perfectly
        # ordinary-looking dict whose int() raised eleven different ways.
        #
        # So each one is normalised to its container type, then filtered element
        # by element. A bad element is DROPPED rather than defaulted: an unknown
        # resource is not a resource, a non-string build order names no
        # structure, and a non-string chronicle line would raise again the first
        # time the HUD tried to draw it.
        w.stockpile = {r: 0 for r in ALL_RESOURCES}
        raw_stock = d.get("stockpile")
        if not isinstance(raw_stock, dict):
            if raw_stock:
                log.warning("save section 'stockpile' was %s, not a mapping; "
                            "resources reset to zero", type(raw_stock).__name__)
            raw_stock = {}
        for k, v in raw_stock.items():
            # A non-string key cannot match a resource name, so the membership
            # test also does the type check.
            if k in ALL_RESOURCES:
                w.stockpile[k] = _int(v, 0, lo=0, hi=1_000_000_000)

        # The hermit's camp store, on exactly the same terms as the colony's
        # above it and deliberately in a separate dict: the two books never meet.
        # A save written before the stash existed carries no key at all, loads
        # empty, and the hermit simply starts banking from that moment - the same
        # "growth" precedent every additive field here follows. `_init_colony_state`
        # already put an all-zero dict on `w`; this overwrites the entries a save
        # actually carries and drops anything that is not a resource.
        w.hermit_stash = {r: 0 for r in ALL_RESOURCES}
        raw_stash = d.get("hermit_stash")
        if not isinstance(raw_stash, dict):
            if raw_stash:
                log.warning("save section 'hermit_stash' was %s, not a mapping; "
                            "the camp store resets to zero",
                            type(raw_stash).__name__)
            raw_stash = {}
        for k, v in raw_stash.items():
            if k in ALL_RESOURCES:
                w.hermit_stash[k] = _int(v, 0, lo=0, hi=HERMIT_STASH_CAP)

        raw_queue = d.get("build_queue")
        if not isinstance(raw_queue, (list, tuple)):
            if raw_queue:
                log.warning("save section 'build_queue' was %s, not a list; "
                            "the queue starts empty",
                            type(raw_queue).__name__)
            raw_queue = ()
        # dict, not str. The queue holds whole job records - behavior._publish
        # writes them and _queue_of reads them back as list[dict[str, Any]] -
        # and the ``list[str]`` annotation this field carried until now is
        # exactly how a first cut of this filter came to drop every genuine
        # entry on load while the junk sweep stayed green, because junk is what
        # it was looking for. The round-trip check is what caught it.
        #
        # Filtering at all is still right: the old ``list(...)`` accepted a bare
        # string by iterating it, so ``"hut"`` quietly became three
        # one-character build orders, and any non-record element reaches
        # behavior.py as something it will try to read keys off.
        w.build_queue = [k for k in raw_queue if isinstance(k, dict)]

        raw_chron = d.get("chronicle")
        if not isinstance(raw_chron, (list, tuple)):
            if raw_chron:
                log.warning("save section 'chronicle' was %s, not a list; "
                            "the log starts empty", type(raw_chron).__name__)
            raw_chron = ()
        w.chronicle = deque((s for s in raw_chron if isinstance(s, str)),
                            maxlen=CHRONICLE_MAX)
        # Defaults first, saved values over the top: a save written before
        # "abducted"/"returned" existed simply keeps them at zero instead of
        # loading a stats dict that KeyErrors the moment anything reads them.
        #
        # Coerced key by key rather than in one comprehension, and that is a fix
        # rather than a style change. This was
        # ``w.stats.update({k: int(v) for k, v in saved_stats.items()})``, and
        # every hostile value in a hand-edited save came straight out of it as an
        # exception: "banana" raises ValueError, [] and None raise TypeError,
        # float("nan") raises ValueError, float("inf") raises OverflowError, and
        # a "stats" section that is a LIST rather than a dict raises
        # AttributeError on .items(). from_dict is the one method in this file
        # that must not raise - persist.load_world reads any exception as a
        # corrupt save and QUARANTINES it - so a single junk counter did not cost
        # the counter, it cost the colony. Same reasoning, and the same fix
        # shape, as hut_tier_dwell forty lines up.
        #
        # A bad value falls back to the DEFAULT for that key (or 0 for a key this
        # build has never heard of), because a stat is a tally: losing one is a
        # cosmetic wrong number, and there is nothing better to guess.
        #
        # Unknown keys are still carried through, deliberately. A save written by
        # a newer build carries counters this one has no name for, and dropping
        # them would silently reset them on the first load by an older binary.
        # NORMALISED TO A DICT ONCE, HERE, and the name is reused below by the
        # legacy-"abducted" block's ``"abducted" not in saved_stats``. That
        # membership test is why this is not simply inlined: the old code leaned
        # on ``d.get("stats") or {}`` to guarantee a container, and dropping the
        # ``or {}`` while adding the isinstance check moved the crash rather than
        # fixing it - ``"abducted" not in 42`` is a TypeError, and a save with
        # ``"stats": 42`` was quarantined by the very method meant to rescue it.
        # Caught by the junk-load proof; do not un-normalise this.
        raw_stats = d.get("stats")
        if not isinstance(raw_stats, dict):
            if raw_stats:
                log.warning("save section 'stats' was %s, not a mapping; "
                            "counters reset to defaults",
                            type(raw_stats).__name__)
            raw_stats = {}
        saved_stats = raw_stats
        w.stats = dict(STAT_DEFAULTS)
        for k, v in saved_stats.items():
            # A non-string key cannot be one of ours and cannot survive JSON as
            # anything but a string anyway, so it is a hand-edit. Skip it rather
            # than letting it into a dict[str, int].
            if not isinstance(k, str):
                continue
            w.stats[k] = int(_finite(v, float(STAT_DEFAULTS.get(k, 0)),
                                     lo=-1e12, hi=1e12))

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

        # The roster cap, enforced on the way IN as well as at spawn. Before
        # _rebase_books, so the books are re-derived from the roster this leaves
        # behind rather than from the one the file claimed.
        w._clamp_roster()

        # LAST, because it needs the final roster: after any _seed_population
        # above, and after the burial re-flag loop that decides which corpses
        # are still pending. See _rebase_books for why a load is the one moment
        # the books can legitimately be rewritten.
        w._rebase_books()
        return w

    def _clamp_roster(self) -> None:
        """Hold the loaded roster to MAX_POP living, once, at load time.

        MAX_POP was enforced at the spawn choke point and nowhere else, so it
        bounded everything the SIMULATION can do and nothing a FILE can claim.
        Measured on the shipped tree: a save whose ``population.agents`` list was
        hand-inflated to 2000 records loaded 2000 living colonists against a
        MAX_POP of 20, reconcile ok=1 residual 0, no exception and no warning -
        an impossible world, imported without a murmur, by the one method in
        this file that is the project's trust boundary.

        Nothing in the running sim can reach here (``_spawn_replacement``
        returns early at the cap and the ufo re-checks it before handing anyone
        back), so on every honest save this is a no-op that logs nothing. It is
        for the hand-edited file, the half-written one, and the save from some
        future build with a different cap.

        WHAT HAPPENS TO THE SURPLUS, and why:

        * They are removed from the roster, not killed. A death raises a grave,
          fires the death hooks and lands in ``stats["died"]`` forever - so
          "clamping" by killing 1980 fabricated people would write 1980 deaths
          into a colony's permanent record for a headcount it never had. The
          surplus did not die here; as far as this build is concerned they were
          never here at all.
        * ``births`` comes down by exactly as many as were dropped, which is
          what keeps :meth:`reconcile` balanced WITHOUT inventing anything.
          Leaving births alone would hand _rebase_books a slack of 1980 with
          nowhere to put it but ``died``, which is the same fabrication by a
          longer route. Floored at zero; if the file's births cannot cover the
          drop, _rebase_books credits the remainder as returns, as it does for
          any other under-counted roster.
        * The survivors are the FIRST MAX_POP in save order. ``Population``
          appends on spawn and pops on removal, so the saved order is arrival
          order and the head of it is the longest-standing colonists - the only
          evidence a file offers about who belongs. It is also deterministic,
          which matters more than the choice itself: two processes loading the
          same save must keep the same people.
        * ``peak`` is clamped too. It is a high-water mark of ``alive_count``
          and the running sim cannot push it past MAX_POP either, so a peak read
          off an impossible roster is impossible in the same way.

        The DEAD are left alone. MAX_POP caps the living; corpses awaiting
        burial are already counted as deaths, and dropping them here would
        delete graves the books have already paid for.

        Never raises: it is called from ``from_dict``.
        """
        try:
            alive = [a for a in self.population.agents if a.alive]
            surplus = len(alive) - MAX_POP
            if surplus <= 0:
                return
            doomed = {id(a) for a in alive[MAX_POP:]}
            # Slice assignment, not rebinding: the roster is a list somebody
            # else may already be holding a reference to, and swapping the
            # attribute would leave them looking at the impossible one.
            self.population.agents[:] = [a for a in self.population.agents
                                         if id(a) not in doomed]
            births = int(getattr(self.population, "births", 0))
            self.population.births = max(0, births - surplus)
            self.population.peak = max(
                min(int(getattr(self.population, "peak", 0)), MAX_POP),
                len(self.population.alive_agents()),
            )
            log.warning("save carried %d living colonists against MAX_POP %d; "
                        "dropped the %d most recent and rebased births %d -> %d",
                        len(alive), MAX_POP, surplus, births,
                        self.population.births)
        except Exception:
            log.debug("could not clamp the loaded roster", exc_info=True)

    def _rebase_books(self) -> None:
        """Make the population books add up again, once, at load time.

        :meth:`reconcile`'s identity - ``births + returned - deaths - abducted
        - alive == 0`` - holds continuously in a running world, but a LOAD can
        hand it two halves that describe different histories, and then it is
        non-zero from the first tick through no fault of the simulation.

        The two halves are not equally trustworthy, and that asymmetry is the
        whole fix. ``births`` and ``alive`` are STRUCTURAL: they are read off
        the roster, which is a thing you can count. ``died``, ``abducted`` and
        ``returned`` are bare tallies carried in ``stats``. So when the halves
        disagree, the roster wins and the tallies are re-derived from it.

        Both directions were measured on a genuine seed-314 save:

        * an unreadable or absent ``population`` section falls back to a fresh
          ``Population``, which then seeds a new group - but ``stats`` still
          remembers the 5 deaths and 2 abductions of the roster that is now
          gone. Residual -7.
        * an unreadable or absent ``stats`` section resets the tallies to their
          defaults while the roster keeps all 19 births and 12 survivors.
          Residual +5 - and here the rebase does better than damage control,
          because ``died`` is the only free variable in the identity and
          solving for it recovers the true 5.

        ``slack`` is what the roster says must be accounted for; ``abducted`` is
        honoured as far as it will fit (the ufo's own record is the better
        evidence for it) and everything left is death, which is the only term
        with nowhere else to go.

        THIS RUNS ON THE LOAD PATH AND NOWHERE ELSE, and it warns when it
        changes anything. That distinction is load-bearing: reconcile exists to
        catch population drift during a session, and a rebase on the tick path
        would silently absorb exactly the drift it is watching for. On an honest
        save the arithmetic is a no-op - the identity already held when the save
        was written, so it re-derives the values the save carries and logs
        nothing.

        Never raises: it is called from ``from_dict``.
        """
        try:
            alive = len(self.population.alive_agents())
            births = int(getattr(self.population, "births", 0))
            # The same "dead but not yet buried" term deaths_ever adds, counted
            # the same way, so the two agree by construction.
            pending = sum(1 for a in self.population.agents
                          if not a.alive and not a.__dict__.get("_buried"))
            returned = max(0, int(self.stats.get("returned", 0)))
            abducted = max(0, int(self.stats.get("abducted", 0)))
        except Exception:
            log.debug("could not rebase the population books", exc_info=True)
            return

        slack = births + returned - alive - pending
        if slack < 0:
            # More people standing there than the counters can explain. The
            # roster is the half that can be counted, so the shortfall is
            # credited as returns rather than argued with - and `returned` is
            # the right home for it because it is the only term that ADDS
            # people who were not born here.
            returned -= slack
            slack = 0
        abducted = min(abducted, slack)
        died = slack - abducted

        before = (self.stats.get("died"), self.stats.get("abducted"),
                  self.stats.get("returned"))
        after = (died, abducted, returned)
        if before != after:
            log.warning("population books did not balance on load; rebased "
                        "(died, abducted, returned) %r -> %r", before, after)
        self.stats["died"] = died
        self.stats["abducted"] = abducted
        self.stats["returned"] = returned
