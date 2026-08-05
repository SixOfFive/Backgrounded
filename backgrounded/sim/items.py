"""Relics - what a dead dragon leaves behind, and the one slot that holds it.

Pure python - **no pygame**, no render import, same rule as every other module
under ``sim/``.

This module owns five things and deliberately nothing else: the definitions, the
drop table, the on-ground object, the pickup/carry/drop transitions, and the
seeded random stream every relic decision in the feature draws from. It does not
own a scored action (that is :mod:`.combat_actions`), it does not own the beam
geometry (:mod:`.throwing`), and it does not own the ufo's ward
(:mod:`.ufo`). Those three read this module; this module reads none of them.

Why this is not a prop
----------------------
The obvious home was ``props.py``: a relic is prop-SHAPED - id, kind, x, y, a
state dict that round-trips - and ``PropRegistry`` already does all of that. It
was rejected on inspection, not on taste. Adding a kind to ``props.KINDS`` drags
the new object into ``_regrow``'s target bookkeeping, ``_usable_count``, the
harvest-claim system and ``find_prop``'s gather targets, and ``tick_props`` runs
fire spread and regrowth machinery over it every frame. A relic has nothing to do
with any of that, and the failure mode of getting it wrong is a colonist deciding
to *harvest* the dragonscale.

``throwing.SpearRegistry`` is the closer precedent and is what is copied here: a
droppable object with a pickup radius, a ``take()`` that returns False when
somebody else got there first, a ``nearest()``, a cap that recycles the oldest,
and a matching fetch action in combat_actions. The double-claim problem that cost
props.py 691 collision events is already solved over there, and it is solved the
same way here.

What a dragon leaves, and why it is not one item per kind
--------------------------------------------------------
The first table paired each dragon kind with exactly one relic. Over 32 seeds x
120 sim-min of ordinary play that produced dragonscale 18, amulet 5, ironshod 2,
cairnstone 2 and **bfg9000 zero** - the headline item of the feature never
entered play at all, because a 1:1 table multiplies a relic's rarity by its
dragon's and the serpent is both the rarest kind and the least-killed. See
:data:`DROP_WEIGHTS` for the replacement - weighted rows, a drought, and a pity
rule for the first wyrm-gun and the first pair of boots a colony ever earns -
and for the measured before/after mix over 48 seeds x 120 sim-min per arm. The
number of relics per dead dragon is still exactly one, so the total drop rate
did not move; only the mix did.

Randomness
----------
**Every relic decision in the whole feature comes off** :meth:`RelicRegistry.roll`
- the drop weighting, ``BFG_MISFIRE``, ``BFG_HEEDLESS``, and any tiebreak
a later lane needs. Not ``world.pyrng``: ``behavior.choose_action`` draws from
that once per scored candidate, and adding ONE inert candidate to it moved mean
deaths 11.33 -> 13.17 (+16%) over 24 seeds with 0/24 matching on any metric. Not
``world.spears.rng``, or a misfire re-phases every spear throw. Not bare
``random``. Not ``hash()`` of a string - CPython salts str hashing per process,
which moves irreproducibility from per-load to per-launch instead of fixing it;
``zlib.crc32`` where a string has to become a number. That bug has shipped seven
times in this project.

:meth:`RelicRegistry.from_dict` **replays** the stream by ``draws``
(``dragons.py``'s pattern) rather than restarting it (``ufo.py``'s), so a reload
CONTINUES the sequence instead of starting a fresh one.

One caveat worth writing down, because it looks like a violation of the
neighbouring rule and is not. ``ufo._pick_victim`` must draw its victim
UNCONDITIONALLY and only then override for an amulet bearer, because that stream
also schedules abductions and a conditional draw there would silently re-phase
every world that has earned an amulet. :meth:`RelicRegistry.roll_drop` below
takes exactly ONE draw per dead dragon whatever path it goes down - the pity
rule spends a draw it does not use, precisely so the count cannot depend on the
colony's history or on a tuning constant. The sequence therefore stays a pure
function of the world seed and the dragons that died.

Failure policy
--------------
:meth:`RelicRegistry.tick` is a frame path and is documented never to raise, for
the same reason ``AnimalRegistry.tick`` and ``SpearRegistry.tick`` are.
:meth:`RelicRegistry.from_dict` never raises either: ``persist.load_world`` reads
an exception out of a loader as a corrupt save and QUARANTINES it, which destroys
a live colony over one bad field. Junk takes defaults.

Nothing here bumps ``SAVE_VERSION``. A save written before relics existed simply
has no ``"relics"`` key, ``world.py``'s ``_sub`` helper treats an absent key as
the ordinary case and hands back the seeded fallback, and the colony loads clean
with no relics and no warning.
"""
from __future__ import annotations

import logging
import math
import random
import zlib
from dataclasses import dataclass
from typing import Any, Iterator

from ..constants import (
    ARMOUR_DRAGONSCALE, ARMOUR_NONE, DRAGON_QUADRUPED, DRAGON_SERPENT,
    DRAGON_SKELETAL, DRAGON_WYVERN, MAX_RELICS, RELIC_AMULET, RELIC_BFG,
    RELIC_BOOTS, RELIC_CAIRN, RELIC_DECAY_SEC, RELIC_FETCH_RANGE,
    RELIC_KINDS, RELIC_NONE, RELIC_SCALE, RENDER_W, WEAPON_BFG, WEAPON_NONE,
)

log = logging.getLogger(__name__)

__all__ = [
    "Relic", "RelicRegistry", "KINDS", "DROP_TABLE", "DROP_WEIGHTS",
    "DROUGHT_GAIN", "DROUGHT_MAX", "PITY_DROPS", "PITY_FLOOR", "PITY_KINDS",
    "NOUN", "PICKUP_LINE", "BEAM_LIFE_SEC", "MAX_BEAMS",
    "registry_of", "ensure_registry", "relics_of",
    "worn", "has", "equip", "unequip", "consume", "drop_for_kind", "noun_for",
]

#: == constants.RELIC_KINDS. Re-exported so a caller that already imports this
#: module does not have to reach into constants for the tuple as well.
KINDS: tuple[str, ...] = RELIC_KINDS

# ------------------------------------------------------------- drop table --
#: dragon kind -> relative weight per relic. **``DRAGON_FLYOVER`` is
#: deliberately absent**: it cruises at 360 px, outside ``THROW_MAX_RANGE`` and
#: well outside the 141 px ballistic ceiling, so it cannot die by design and a
#: drop entry for it would be code that can never run.
#:
#: WHY THIS IS NOT FOUR 1:1 PAIRINGS ANY MORE. It was, and the pairings read
#: beautifully - quadruped -> hide, wyvern -> the ward against being snatched,
#: serpent -> the game-ender, skeletal -> what is made of graves. Then it was
#: measured twice, unforced, and it did not deliver the two items the design
#: leans hardest on:
#:
#:   16 seeds x 120 sim-min x 2 arms (32 colony-hours), 27 drops:
#:       dragonscale 18, amulet 5, ironshod 2, cairnstone 2, **bfg9000 0**
#:   32 seeds x 120 sim-min (64 colony-hours), 34 drops:
#:       amulet 12, dragonscale 10, cairnstone 5, ironshod 4, **bfg9000 3**
#:       - and the gun reached 3 colonies out of 32.
#:
#: A 1:1 table multiplies the relic's rarity by its dragon's, and the serpent is
#: both the rarest kind in ``dragons._KIND_WEIGHTS`` and the least-killed of the
#: five (32% killed against the quadruped's 82%), so the compound probability
#: put the headline item of the whole feature at 0.047 drops per colony-hour -
#: one per 21 hours, in a game whose colonies mostly do not live that long. The
#: boots - the only answer in the game to falling, which kills more colonists
#: than anything else - came out second rarest for the same reason, and caught
#: exactly ZERO lethal landings across all 64 of those colony-hours.
#:
#: So the table is now weighted rather than paired. Every kind can leave any of
#: the five; the signature relic keeps the largest weight, so the flavour of
#: "the quadruped drops its hide" survives as a tendency instead of a law, and
#: the gun is no longer hostage to one dragon that hardly ever dies. THE NUMBER
#: OF DROPS DID NOT CHANGE: still exactly one relic per dead dragon, no second
#: roll, so the total drop rate is untouched and only the MIX moves. That was
#: the constraint - the player should not be tripping over loot.
#:
#: Rows are read through :meth:`RelicRegistry.roll_drop`, which ages each weight
#: by that relic's DROUGHT (:data:`DROUGHT_GAIN`), floors the two headline items
#: while the colony has never seen them (:data:`PITY_FLOOR`) and applies the
#: backstop (:data:`PITY_DROPS`) before drawing. The weights below are therefore
#: the steady-state tendency, not the shipped frequencies.
#:
#: THE SHIPPED FREQUENCIES, A/B, 48 seeds x 120 sim-min per arm on the SAME
#: seeds, unforced, real build, 96 colony-hours each (the "before" arm is this
#: file with ``roll_drop`` patched back to the paired table, so the only
#: difference between the two columns is the table):
#:
#:   relic          drops per colony-hour     colonies that saw one, of 48
#:                   paired    weighted          paired    weighted
#:   dragonscale      0.135       0.083            13          8
#:   amulet           0.125       0.042            11          4
#:   bfg9000          0.073       0.125             6         12
#:   ironshod         0.031       0.073             3          7
#:   cairnstone       0.083       0.083             8          8
#:   ALL              0.448       0.406            29         29
#:
#: The gun and the boots roughly double, the total rate does not move (-9%, one
#: standard deviation on 43 vs 39 drops - the player is not tripping over loot),
#: and the same 29 colonies get relics either way because that is the stone-hut
#: gate's business and not this table's. The number that matters most is not in
#: the table: LETHAL LANDINGS THE BOOTS CAUGHT went 0 -> 4 across those 96
#: colony-hours. The item with the best answer to the biggest killer in the game
#: had never once been worn by somebody who then fell.
#:
#: The cost is the amulet, which is now the rarest drop instead of the second
#: commonest (0.125 -> 0.042). That is the floor crowding it out of the wyvern's
#: row while a colony is still looking for its first gun, and it is a SHORT
#: horizon effect - once both headline items are seen the floor lifts and the
#: amulet is back to 40% of that row. Watch it if the drop rate ever rises.
#:
#: AND OVER "A FEW HOURS", which is what the brief actually asked for - 16 seeds
#: x 300 sim-min per arm, 80 colony-hours each, where a colony has matured and
#: earns 1-8 relics instead of 0-2:
#:
#:   relic          drops per colony-hour     colonies that saw one, of 16
#:                   paired    weighted          paired    weighted
#:   dragonscale      0.212       0.188             8         10
#:   amulet           0.263       0.150            14         10
#:   bfg9000          0.150       0.225             9         14
#:   ironshod         0.125       0.225             7         13
#:   cairnstone       0.113       0.125             6          7
#:   ALL              0.863       0.912            16         16
#:
#: 14 colonies of 16 see a wyrm-gun, median first sighting 209 sim-min. The
#: mix settles to bfg9000 24.7% / ironshod 24.7% / dragonscale 20.5% / amulet
#: 16.4% / cairnstone 13.7%, which is the table's own tendency showing through
#: once the floor has lifted - the gun does NOT keep climbing. Lethal landings
#: the boots caught: 75 -> 138. Deaths per colony-hour: 19.36 -> 19.59, i.e.
#: unmoved, even though the colony fired the gun 47 times instead of 20.
DROP_WEIGHTS: dict[str, dict[str, float]] = {
    # The landed, 320 hp, heaviest-hided one, and the kind that actually dies:
    # 82% of kills, so this row is most of the shipped mix on its own.
    DRAGON_QUADRUPED: {RELIC_SCALE: 0.42, RELIC_BOOTS: 0.24,
                       RELIC_AMULET: 0.12, RELIC_CAIRN: 0.12,
                       RELIC_BFG: 0.10},
    # The thing that snatches you out of the air still mostly drops the ward
    # against being snatched.
    DRAGON_WYVERN:    {RELIC_AMULET: 0.40, RELIC_BOOTS: 0.20,
                       RELIC_SCALE: 0.15, RELIC_BFG: 0.15,
                       RELIC_CAIRN: 0.10},
    # The rarest kill still favours the game-ender by a mile - killing a serpent
    # remains the fast way to a wyrm-gun. It is simply no longer the only way.
    DRAGON_SERPENT:   {RELIC_BFG: 0.50, RELIC_AMULET: 0.18,
                       RELIC_BOOTS: 0.16, RELIC_SCALE: 0.10,
                       RELIC_CAIRN: 0.06},
    # The grave-robber, and the home of the two grave-made items.
    DRAGON_SKELETAL:  {RELIC_BOOTS: 0.34, RELIC_CAIRN: 0.30,
                       RELIC_SCALE: 0.14, RELIC_AMULET: 0.12,
                       RELIC_BFG: 0.10},
}

#: Legacy view of the table: dragon kind -> the relics it can leave, commonest
#: first. Derived, never edited by hand. Kept because it is the shape the rest
#: of the project was told the drop table had, and because "which kinds can drop
#: X" is a question worth being able to ask cheaply.
DROP_TABLE: dict[str, tuple[str, ...]] = {
    kind: tuple(k for k, w in sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))
                if w > 0.0)
    for kind, row in DROP_WEIGHTS.items()
}

#: How much a relic's weight grows for each drop that has passed without it.
#: ``weight x (1 + DROUGHT_GAIN * dry)``, ``dry`` capped at :data:`DROUGHT_MAX`.
#:
#: This is the soft half of the reachability fix and it is deliberately soft: it
#: does not change what the FIRST drop of a colony can be, it only stops a
#: colony that has had six dragonscales from getting a seventh. Long-run it
#: compresses the mix toward the table's tendencies rather than its extremes,
#: which is what makes "the boots are not the rarest thing" true over a long
#: session as well as a short one.
DROUGHT_GAIN = 0.5
#: Cap on the drought counter, so a hundred-hour colony that has somehow never
#: seen a cairnstone does not carry a 50x weight - and so a hand-edited save
#: cannot either.
DROUGHT_MAX = 8

#: The pity rule, part one: the weight FLOOR a member of :data:`PITY_KINDS`
#: carries while this colony has never seen one. Not a multiplier - a floor, so
#: a relic that is rare on the table (the gun, at 0.10 of the quadruped's row)
#: and one that is common on it (the boots, at 0.24) get the same shot at being
#: the thing a colony finally finds, and neither drowns the other.
#:
#: WHY A FLOOR AND NOT A GUARANTEE, MEASURED. A pity that simply HANDS OVER an
#: unseen headline item on the colony's second drop was tried first, in two
#: variants, and both broke the mix - because an ordinary colony earns 0, 1 or 2
#: relics in two hours (eleven of 32 got none, twelve got one, nine got two, and
#: NOT ONE got three), so anything promised "by the second drop" becomes most of
#: the loot in the game:
#:
#:   gun first when both were owed:  bfg9000 14 drops of 31 - 45%, commonest
#:                                   relic in the game, boots still 5
#:   a coin between the two:         bfg9000 13 of 30 - 43%. The coin did not
#:                                   help, because the BOOTS are common enough
#:                                   to arrive first on their own, and every
#:                                   colony that found boots first then had
#:                                   only the gun left to be owed.
#:
#: A floor is the same idea without the cliff: while a colony has never seen the
#: gun it is roughly a one-in-four drop rather than a one-in-ten, and the colony
#: that has had four dragons and no gun is helped four times instead of once.
#:
#: 0.35 AND NOT 0.5, ALSO MEASURED. At 0.5 the floor sat above the boots' own
#: weight in every row, which made the gun beat the boots on the quadruped (31%
#: against 23% of a colony's first drop) and shipped 11 guns against 2 pairs of
#: boots over 32 seeds x 120 sim-min. That is the brief's complaint about the
#: boots restated, so the floor was dropped until the two headline items came
#: out level: 26% each on the quadruped's row while both are unseen, with the
#: dragonscale still the commonest thing that row can produce.
PITY_FLOOR = 0.35
#: The pity rule, part two: the backstop. A member of :data:`PITY_KINDS` this
#: colony has never seen and has now sat out this many drops IS the next drop.
#:
#: With PITY_FLOOR doing the work this only catches the unlucky tail - P(three
#: drops, no gun) is about a third - and it is deliberately set past the two
#: drops an ordinary two-hour colony ever gets, so it costs nothing at that
#: horizon and only speaks up for the long-running wallpaper colony that has
#: been robbed by the dice. Kill four dragons and you have seen a wyrm-gun.
PITY_DROPS = 3
#: The kinds the pity rule speaks for, in the order the backstop hands them over.
#:
#: Two of five, on purpose. These are the items the player asked for by name -
#: the gun, and the boots that counter falls, the largest single cause of death
#: in the game - and they were the two the paired table made least reachable
#: (bfg9000 0 drops and ironshod 2 in 32 colony-hours). The other three are
#: common enough on the weights alone and are left to luck, which also keeps the
#: rule from turning a colony's first five drops into a scripted sequence.
PITY_KINDS: tuple[str, ...] = (RELIC_BFG, RELIC_BOOTS)

#: How the chronicle and the HUD name each relic. Lower case, no article - the
#: callers supply "a"/"the" so a line can read either way.
NOUN: dict[str, str] = {
    RELIC_SCALE:  "dragonscale coat",
    RELIC_AMULET: "wyrm-bead amulet",
    RELIC_BFG:    "wyrm-gun",
    RELIC_BOOTS:  "pair of ironshod boots",
    RELIC_CAIRN:  "cairnstone",
}

#: What the chronicle says when somebody claims one. The dragonscale line is the
#: one the design specified; the other four are written to match its register -
#: plain, physical, no exclamation - so the set reads as one voice.
PICKUP_LINE: dict[str, str] = {
    RELIC_SCALE:  "{name} straps the wyrm's hide across their shoulders.",
    RELIC_AMULET: "{name} hangs the wyrm's bead at their throat.",
    RELIC_BFG:    "{name} takes up the wyrm-gun, and puts the spear down.",
    RELIC_BOOTS:  "{name} laces on boots shod with the wyrm's own iron.",
    RELIC_CAIRN:  "{name} pockets a stone that hums against the palm.",
}

#: Kinds whose effect is a mirror into an existing ``Stickman`` field rather than
#: a new code path. Written down as data because :func:`equip` and
#: :func:`unequip` must agree about it exactly, and they were two functions
#: before they were one table.
_MIRROR_ARMOUR = RELIC_SCALE
_MIRROR_WEAPON = RELIC_BFG

# ---------------------------------------------------------------- limits ----
_EDGE_MIN = 0.0
_EDGE_MAX = float(RENDER_W - 1)
_MAX_DT = 0.25
#: Cap on the replay loop in :meth:`RelicRegistry.from_dict`. A hand-edited
#: ``"draws": 1e12`` must cost a fraction of a second, not the session. Same
#: number and same reason as ``dragons._MAX_REPLAY``.
_MAX_REPLAY = 200_000
#: How many entries of ``DragonRegistry.slain`` one tick will turn into drops.
#: The list is transient and is drained in the same frame it is raised (world.py
#: runs "relics" immediately after "dragons"), so this only ever bites if that
#: ordering is broken or if the "relics" guard has been disabled for the session
#: - in which case a bounded backlog is exactly what we want.
_MAX_DRAIN = 8

#: Seconds a BFG discharge record stays in :attr:`RelicRegistry.beams`.
#:
#: Must be a little LONGER than ``render/items.BEAM_LIFE`` (0.28), which is what
#: actually fades the beam out: if the sim dropped the record first the shot
#: would blink out mid-fade, and the failure would only show on a frame the
#: profiler was not looking at. render is read-only w.r.t. sim state and cannot
#: age this list itself, so the ownership has to sit here.
BEAM_LIFE_SEC = 0.30
#: Hard cap on that list. It is transient and short-lived, but BFG_COOLDOWN is
#: 9.0 s against a 0.30 s life, so anything above a handful means something is
#: appending in a loop - bound it rather than let a frame path grow without one.
MAX_BEAMS = 8


# ------------------------------------------------------------- coercions --
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


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _clamp_x(x: Any) -> float:
    return _clamp(_f(x, 0.0), _EDGE_MIN, _EDGE_MAX)


def _s(value: Any, default: str = "") -> str:
    try:
        return str(value)
    except Exception:
        return default


# --------------------------------------------------------- world adapters --
# All duck-typed. A bare object with none of these is a legal argument
# everywhere in this module: the mechanic degrades to "no relics" rather than to
# an exception, which is the only safe direction for a frame path.
def _ground_y(world: Any, x: float) -> float:
    terrain = getattr(world, "terrain", None)
    fn = getattr(terrain, "ground_y", None)
    if not callable(fn):
        return 0.0
    try:
        return _f(fn(x), 0.0)
    except Exception:
        return 0.0


def _log(world: Any, text: str) -> None:
    """Chronicle line, if the world has a chronicle. Never raises."""
    if not text:
        return
    for name in ("log_event", "chronicle"):
        fn = getattr(world, name, None)
        if callable(fn):
            try:
                fn(text)
            except Exception:
                log.debug("chronicle write failed", exc_info=True)
            return


def _bump(world: Any, key: str, n: int = 1) -> None:
    st = getattr(world, "stats", None)
    if not isinstance(st, dict):
        return
    try:
        st[key] = _i(st.get(key, 0), 0) + int(n)
    except Exception:
        pass


# ================================================================== Relic ==
@dataclass
class Relic:
    """One relic lying in the dirt, waiting for somebody to walk over and take it.

    A relic only ever exists as a *ground* object. The carried side is not a
    Relic at all - it is the single string field ``Stickman.relic``, which rides
    on the agent's own to_dict/from_dict. One slot per agent, so nobody is both
    impenetrable and holding the gun, and no inventory list to reconcile.
    """

    id: int = 0
    kind: str = RELIC_SCALE
    x: float = 0.0
    y: float = 0.0
    #: Seconds lying here. This is the clock ``RELIC_DECAY_SEC`` is read
    #: against, and it resets if a relic is ever re-landed.
    t: float = 0.0
    #: Seconds since this ground object was created, never reset. Only diverges
    #: from ``t`` if something re-lands an existing Relic (nothing does today);
    #: kept separate so render/ can fade a fresh drop in off ``age`` without
    #: reading the decay clock, exactly as ``Spear`` splits the same two.
    age: float = 0.0

    @property
    def noun(self) -> str:
        return NOUN.get(self.kind, "relic")

    def summary(self) -> str:
        return (f"relic#{self.id} {self.kind} @{self.x:.0f},{self.y:.0f} "
                f"t={self.t:.1f}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "kind": str(self.kind),
            "x": float(self.x),
            "y": float(self.y),
            "t": float(self.t),
            "age": float(self.age),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Relic":
        """Rebuild from a save. Junk takes defaults; **never raises**."""
        r = cls()
        if not isinstance(d, dict):
            return r
        r.id = max(0, _i(d.get("id"), 0))
        kind = _s(d.get("kind"), "")
        # An unreadable kind becomes the dragonscale rather than the empty
        # string: RELIC_NONE is "no relic", and a ground object claiming to be
        # nothing is a thing nobody can ever pick up and nothing ever removes.
        r.kind = kind if kind in RELIC_KINDS else RELIC_SCALE
        r.x = _clamp_x(d.get("x"))
        r.y = _f(d.get("y"), 0.0)
        r.t = _clamp(_f(d.get("t"), 0.0), 0.0, RELIC_DECAY_SEC)
        r.age = max(0.0, _f(d.get("age"), 0.0))
        return r


# =========================================================== RelicRegistry ==
class RelicRegistry:
    """Every relic on the ground, the drop table's dice, and the decay clock.

    Lives on ``world.relics`` and is ticked once a frame from world.py's
    ``_guarded("relics", ...)``, placed AFTER the "dragons" guard and BEFORE the
    "ufo" one so that ``DragonRegistry.slain`` is raised and drained inside the
    same frame and can never cross a save boundary.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed: int = random.randrange(1 << 30) if seed is None else _i(seed, 0)
        self._rng: random.Random = random.Random(self.seed)
        #: Values taken off ``_rng`` so far. REPLAYED by from_dict, never
        #: restarted - see the module docstring.
        self.draws: int = 0

        self.relics: list[Relic] = []
        self.next_id: int = 1
        self.t: float = 0.0
        #: How many relics have been picked up over this world's life. A
        #: display counter; the ledger does not read it.
        self.found: int = 0

        #: Per relic kind, how many this colony has ever DROPPED (not found).
        #: The pity rule reads it as a boolean - "has this colony ever seen
        #: one" - and it is a count rather than a flag because the count is
        #: what a HUD or a chronicle would want and it costs the same to save.
        self.seen: dict[str, int] = {k: 0 for k in RELIC_KINDS}
        #: Per relic kind, how many drops have happened since the last one of
        #: that kind (since the colony began, for a kind never dropped).
        #: Drives both halves of the reachability fix - the drought weighting
        #: and the pity rule - and is capped at ``DROUGHT_MAX``.
        self.dry: dict[str, int] = {k: 0 for k in RELIC_KINDS}
        #: Total drops this colony has made. Not derived from ``seen`` on
        #: purpose: a save written by a build that had neither can restore
        #: sensible defaults for all three independently.
        self.drops: int = 0

        #: Live BFG discharges, for render to draw. TRANSIENT - deliberately not
        #: in to_dict/from_dict, exactly like ``events.strikes`` and
        #: ``DragonRegistry.slain``: a beam is a quarter of a second of light,
        #: and a save caught mid-shot must not reload into one.
        #:
        #: This list lives here rather than on World because ``render/items.py``
        #: already looks for ``world.relics.beams`` first, and because it is the
        #: registry that owns the clock that ages it out - render may not write
        #: sim state, so nothing else can. Each entry is a plain dict
        #: ``{"x0","y0","x1","y1","t","misfire"}`` (+ optional "corridor"), the
        #: same shape ``events.strikes`` uses, so the renderer needs no new type
        #: and a missing key degrades rather than raising.
        self.beams: list[dict[str, Any]] = []

    # ---------------------------------------------------------- container --
    def __len__(self) -> int:
        return len(self.relics)

    def __iter__(self) -> Iterator[Relic]:
        return iter(list(self.relics))

    def all(self) -> list[Relic]:
        return list(self.relics)

    def count(self) -> int:
        return len(self.relics)

    def get(self, rid: Any) -> Relic | None:
        if rid is None:
            return None
        want = _i(rid, -1)
        for r in self.relics:
            if r.id == want:
                return r
        return None

    def add(self, relic: Relic) -> Relic:
        if not isinstance(relic, Relic):
            return relic
        if relic.id <= 0 or any(r.id == relic.id for r in self.relics):
            relic.id = self.next_id
        self.next_id = max(self.next_id, relic.id + 1)
        self.relics.append(relic)
        return relic

    def remove(self, relic: Relic | int) -> bool:
        target = relic if isinstance(relic, Relic) else self.get(relic)
        if target is None:
            return False
        try:
            self.relics.remove(target)
        except ValueError:
            return False
        return True

    def clear(self) -> int:
        n = len(self.relics)
        self.relics.clear()
        return n

    def nearest(self, x: float, *,
                max_dist: float = RELIC_FETCH_RANGE) -> Relic | None:
        """Nearest relic to world-x *x* within *max_dist*, or None.

        Ties break on id, not on list order, so two relics dropped at the same
        spot pick the same one before and after a reload.
        """
        best: Relic | None = None
        best_d = float("inf")
        px = _f(x, 0.0)
        limit = max(0.0, _f(max_dist, RELIC_FETCH_RANGE))
        for r in self.relics:
            d = abs(r.x - px)
            if d > limit:
                continue
            if d < best_d or (d == best_d and best is not None and r.id < best.id):
                best, best_d = r, d
        return best

    def of_kind(self, kind: str) -> list[Relic]:
        return [r for r in self.relics if r.kind == kind]

    # ------------------------------------------------------------- random --
    def roll(self, lo: float = 0.0, hi: float = 1.0) -> float:
        """THE relic random stream. Every relic decision draws from here.

        The skeletal's 50/50, ``BFG_MISFIRE``, ``BFG_HEEDLESS`` and anything a
        later lane adds. Bumps ``draws`` so from_dict can replay to exactly this
        point. See the module docstring for the four streams this must not be.
        """
        self.draws += 1
        r = self._rng.random()
        a = _f(lo, 0.0)
        b = _f(hi, 1.0)
        return a + (b - a) * r

    def chance(self, p: float) -> bool:
        """``roll() < p``. One draw, always, whatever *p* is.

        Written as its own method precisely so a caller cannot "optimise" the
        draw away for p == 0.0 or p == 1.0: the number of draws taken must not
        depend on a tuning constant, or re-tuning ``BFG_MISFIRE`` to 0 would
        silently re-phase everything downstream of it.
        """
        return self.roll() < _f(p, 0.0)

    def pick_weighted(self, weights: dict[str, float]) -> str | None:
        """Weighted choice off :meth:`roll`. **Exactly one draw, always.**

        Iterates ``RELIC_KINDS`` rather than the dict, so two dicts with the
        same contents in a different insertion order cannot pick differently -
        a save/reload must not be able to change which relic a given draw
        selects.

        One draw even when only one kind has a weight. This replaced a uniform
        ``pick(options)`` that short-circuited a single-entry table without
        drawing, which was safe while the table was four fixed tuples and would
        not be now: the number of draws a DROP costs must not depend on how
        many kinds happen to carry a weight, or the stream would re-phase
        whenever the table is retuned. Same reasoning as :meth:`chance`.
        """
        total = 0.0
        for k in RELIC_KINDS:
            w = _f(weights.get(k, 0.0), 0.0)
            if w > 0.0:
                total += w
        if total <= 0.0:
            self.roll()                      # spend it anyway; see above
            return None
        want = self.roll(0.0, total)
        acc = 0.0
        last: str | None = None
        for k in RELIC_KINDS:
            w = _f(weights.get(k, 0.0), 0.0)
            if w <= 0.0:
                continue
            last = k
            acc += w
            if want < acc:
                return k
        return last                          # float slop at the top end

    # -------------------------------------------------------------- drops --
    def overdue(self) -> list[str]:
        """What the BACKSTOP owes this colony, in ``PITY_KINDS`` order.

        The soft half of the pity rule is the ``PITY_FLOOR`` in
        :meth:`weights_for`; this is the hard half, and it only speaks when the
        floor has been unlucky ``PITY_DROPS`` times running. Takes no draw and
        mutates nothing; when it owes both, ``PITY_KINDS`` order decides which
        lands first - the gun, then the boots on the drop after.

        "Unseen" is the whole reason none of this can inflate the long-run
        rate: a kind drops out of both halves the moment the colony has one, so
        the backstop fires at most ``len(PITY_KINDS)`` times in a world's life
        and only ever moves the FIRST of each kind earlier.
        """
        out: list[str] = []
        for k in PITY_KINDS:
            if _i(self.seen.get(k), 0) > 0:
                continue
            if _i(self.dry.get(k), 0) >= PITY_DROPS:
                out.append(k)
        return out

    def weights_for(self, dragon_kind: str) -> dict[str, float]:
        """The weights a *dragon_kind* death rolls against, for THIS colony.

        The table's row, aged by each relic's drought, and floored at
        ``PITY_FLOOR`` for a headline item this colony has never seen. Pure:
        takes no draw and mutates nothing, so a test (or a HUD) can ask what
        the odds are without moving the stream.
        """
        row = DROP_WEIGHTS.get(_s(dragon_kind, ""))
        if not row:
            return {}
        out: dict[str, float] = {}
        for k in RELIC_KINDS:
            base = _f(row.get(k, 0.0), 0.0)
            if base <= 0.0:
                continue
            dry = _clamp(_i(self.dry.get(k), 0), 0, DROUGHT_MAX)
            w = base * (1.0 + DROUGHT_GAIN * dry)
            if k in PITY_KINDS and _i(self.seen.get(k), 0) <= 0:
                w = max(w, PITY_FLOOR)
            out[k] = w
        return out

    def roll_drop(self, dragon_kind: str) -> str | None:
        """What a dead *dragon_kind* leaves, and the bookkeeping behind it.

        None for the flyover and for junk. Exactly ONE relic per death and
        exactly ONE draw per death - the rarity is still the dragon, and the
        total drop rate is untouched by everything in this method.

        Three steps, in order:

        1. the backstop (:meth:`overdue`) - a headline item this colony has
           never seen and has waited ``PITY_DROPS`` drops for is handed over,
           the gun before the boots;
        2. otherwise a weighted draw over :meth:`weights_for`, which is where
           ``PITY_FLOOR`` does the ordinary day's work;
        3. either way, :meth:`_record_drop` resets that kind's drought and ages
           everybody else's.

        Both paths spend exactly one draw, which is the same rule
        :meth:`chance` is written to protect: the number of draws a drop costs
        must not depend on the colony's history or on a tuning constant, or
        ``PITY_DROPS`` could not be retuned without re-phasing every world that
        had ever earned a relic.
        """
        row = DROP_WEIGHTS.get(_s(dragon_kind, ""))
        if not row:
            return None
        try:
            owed = self.overdue()
            if owed:
                kind = owed[0]       # PITY_KINDS order: the gun before the boots
                self.roll()          # spend the drop's draw anyway; see below
            else:
                kind = self.pick_weighted(self.weights_for(dragon_kind))
            if kind not in RELIC_KINDS:
                # Every row is non-empty, so this is unreachable short of a
                # hand-edited table; take the row's own commonest rather than
                # dropping the relic the dragon owed the colony.
                kind = DROP_TABLE.get(_s(dragon_kind, ""), (RELIC_SCALE,))[0]
            self._record_drop(kind)
            return kind
        except Exception:
            log.exception("drop roll failed")
            return DROP_TABLE.get(_s(dragon_kind, ""), (RELIC_SCALE,))[0]

    def _record_drop(self, kind: str) -> None:
        """Age every drought, zero the one that just landed. Never raises."""
        try:
            self.drops += 1
            for k in RELIC_KINDS:
                if k == kind:
                    self.seen[k] = _i(self.seen.get(k), 0) + 1
                    self.dry[k] = 0
                else:
                    self.dry[k] = min(DROUGHT_MAX, _i(self.dry.get(k), 0) + 1)
        except Exception:
            log.debug("drop bookkeeping failed", exc_info=True)

    def spawn(self, world: Any, kind: str, x: float, *,
              announce: bool = True) -> Relic | None:
        """Put a relic of *kind* on the ground at world-x *x*.

        None when *kind* is not a relic. Never raises.
        """
        try:
            k = _s(kind, "")
            if k not in RELIC_KINDS:
                return None
            px = _clamp_x(x)
            self._make_room()
            r = Relic(kind=k, x=px, y=_ground_y(world, px))
            self.add(r)
            if announce:
                _log(world, f"The wyrm leaves behind a {r.noun}.")
            return r
        except Exception:
            log.exception("relic spawn failed")
            return None

    def _make_room(self) -> None:
        """Recycle the oldest relic when the ground is full.

        Forgetting the oldest rather than refusing the newest, exactly as
        ``SpearRegistry._make_room`` does: a cap that swallows the drop the
        player just earned is a bug they can see, and a cap that quietly forgets
        a relic nobody walked to in half an hour is not.
        """
        guard = 0
        while len(self.relics) >= MAX_RELICS and guard < MAX_RELICS + 4:
            guard += 1
            oldest = max(self.relics, key=lambda r: (r.t, r.id))
            self.remove(oldest)

    def drain_slain(self, world: Any) -> list[Relic]:
        """Turn ``world.dragons.slain`` into drops. Drains it either way.

        ``DragonRegistry.slain`` is a TRANSIENT list of ``(kind, x)`` raised by
        ``_reap`` on the frame a dragon dies. It is deliberately not persisted,
        and it is cleared here BEFORE anything can fail, so a spawn that refuses
        (unknown kind, flyover, a stub world) cannot leave a phantom drop sitting
        in the list to be re-served next frame or - worse - across a save.
        """
        out: list[Relic] = []
        reg = getattr(world, "dragons", None)
        raw = getattr(reg, "slain", None)
        if not isinstance(raw, list) or not raw:
            return out
        pending = list(raw[-_MAX_DRAIN:])
        try:
            raw.clear()
        except Exception:
            log.debug("could not drain dragons.slain", exc_info=True)
        for item in pending:
            try:
                kind, x = item[0], item[1]
            except Exception:
                continue
            want = drop_for_kind(self, _s(kind, ""))
            if want is None:
                # The flyover, or a kind this table does not know. Both are
                # "no drop", and neither is an error.
                continue
            r = self.spawn(world, want, _f(x, 0.0))
            if r is not None:
                out.append(r)
        return out

    def add_beam(self, x0: float, y0: float, x1: float, y1: float, *,
                 misfire: bool = False, corridor: float | None = None) -> None:
        """Record one BFG discharge for render. Never raises.

        Called by whoever fires the gun; this module does not fire it. On a
        misfire the endpoint is ignored by the renderer, so passing the
        wielder's own position twice is the honest call.
        """
        try:
            rec: dict[str, Any] = {
                "x0": _f(x0, 0.0), "y0": _f(y0, 0.0),
                "x1": _f(x1, 0.0), "y1": _f(y1, 0.0),
                "t": 0.0, "misfire": bool(misfire),
            }
            if corridor is not None:
                rec["corridor"] = _f(corridor, 0.0)
            self.beams.append(rec)
            del self.beams[:-MAX_BEAMS]
        except Exception:
            log.debug("beam record failed", exc_info=True)

    def _age_beams(self, dt: float) -> None:
        keep: list[dict[str, Any]] = []
        for b in self.beams:
            try:
                b["t"] = _f(b.get("t"), 0.0) + dt
                if b["t"] < BEAM_LIFE_SEC:
                    keep.append(b)
            except Exception:
                continue
        self.beams = keep

    # ----------------------------------------------------- carry and wear --
    def take(self, relic: Relic | int, agent: Any, *,
             world: Any = None) -> bool:
        """Ground -> agent. False when somebody beat you to it.

        False for four reasons, and the first is the one that matters: the relic
        is no longer in the registry, because another colonist walking the same
        way arrived a frame earlier. That is the whole double-claim answer, and
        it is the one ``SpearRegistry.take`` already proves works - props.py
        solved it a different way and paid 691 collision events for it.

        *world* is optional so the contract's two-argument call still works; pass
        it and the pickup gets its chronicle line and its stat.
        """
        try:
            target = relic if isinstance(relic, Relic) else self.get(relic)
            if target is None or agent is None:
                return False
            if not bool(getattr(agent, "alive", True)):
                return False
            if worn(agent):
                return False              # one slot per agent, and it is full
            if not self.remove(target):
                return False              # gone; somebody beat you to it
            if not equip(agent, target.kind):
                # A stub that will not take attributes. Put it back rather than
                # deleting the item - the same "put it back on a failed arm"
                # rule _h_retrieve uses for a spear.
                self.add(target)
                return False
            self.found += 1
            if world is not None:
                _bump(world, "relics_found")
                name = _s(getattr(agent, "name", ""), "")
                line = PICKUP_LINE.get(target.kind, "")
                if name and line:
                    _log(world, line.format(name=name))
            return True
        except Exception:
            log.exception("relic take failed")
            return False

    def drop_from(self, world: Any, agent: Any) -> Relic | None:
        """Agent -> ground, at their feet. None when they carried nothing.

        The death hook. ``World._reap_dead`` is the ONE funnel every death goes
        through and it is already one-shot per corpse behind
        ``agent.__dict__["_buried"]``, so this needs no guard of its own.

        The mirrored slots are cleared back to ARMOUR_NONE / WEAPON_NONE by
        :func:`unequip`, so a corpse is not still wearing the coat it just
        dropped. The one loss that costs: a dragonscale wearer who also had
        leather comes back to no armour rather than to leather. It is very nearly
        unreachable - ``combat_actions._armoured`` reads 1.0 >= ARMOUR_LEATHER,
        so a scale-wearer never crafts leather in the first place - and the
        alternative is remembering a second armour value on the agent for a case
        that does not arise.
        """
        try:
            kind = unequip(agent)
            if not kind:
                return None
            x = _f(getattr(agent, "x", 0.0), 0.0)
            r = self.spawn(world, kind, x, announce=False)
            if r is not None:
                name = _s(getattr(agent, "name", ""), "")
                if name:
                    _log(world, f"{name}'s {r.noun} lies where they fell.")
            return r
        except Exception:
            log.exception("relic drop failed")
            return None

    # --------------------------------------------------------------- tick --
    def tick(self, world: Any, dt: float) -> None:
        """Drain the dragons' kills, age what is lying about. **Never raises.**

        Deliberately NOT frame-deduplicated on ``world.tick_count`` the way
        ``SpearRegistry.tick`` is. That guard exists over there because spears
        are driven from two places; relics are driven from exactly one
        (``world.tick``'s ``_guarded("relics", ...)``), and a dedupe would
        silently turn every harness that forgets to bump ``tick_count`` into a
        world where nothing ever decays.
        """
        try:
            # Before the dt gate: the slain list is transient and must not
            # survive a paused frame.
            self.drain_slain(world)

            step = _clamp(_f(dt, 0.0), 0.0, _MAX_DT)
            if step <= 0.0:
                return
            self.t += step
            self._age_beams(step)
            for r in list(self.relics):
                try:
                    r.t += step
                    r.age += step
                    if r.t >= RELIC_DECAY_SEC:
                        self.remove(r)
                except Exception:
                    log.exception("relic %s failed; removing it", r.id)
                    self.remove(r)
        except Exception:
            log.exception("relic tick failed")

    # ------------------------------------------------------------ persist --
    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "draws": int(self.draws),
            "relics": [r.to_dict() for r in self.relics],
            "next_id": int(self.next_id),
            "t": float(self.t),
            "found": int(self.found),
            # The pity rule's memory. Three new keys under the existing
            # "relics" sub-dict, which is why nothing here touches
            # SAVE_VERSION: an older save simply has none of them, from_dict
            # defaults them to a fresh colony's zeros, and the colony loads
            # clean - it just gets its pity clock restarted, which is the
            # generous direction to be wrong in.
            "seen": {k: _i(self.seen.get(k), 0) for k in RELIC_KINDS},
            "dry": {k: _i(self.dry.get(k), 0) for k in RELIC_KINDS},
            "drops": int(self.drops),
        }

    @classmethod
    def from_dict(cls, d: Any, seed: int | None = None) -> "RelicRegistry":
        """Defensive load: an unreadable record is dropped, never fatal.

        ``seed`` is optional so a caller holding the world seed can thread it
        through. ``World.from_dict`` loads every subsystem through a
        one-argument helper and cannot, so the saved seed is used, and failing
        that a stable one is derived from the record itself with
        :func:`zlib.crc32` - never ``hash()``.
        """
        base = _i((d or {}).get("seed"), 0) if isinstance(d, dict) else 0
        if seed is not None:
            base = _i(seed, 0)
        elif not base:
            base = _record_seed(d)
        reg = cls(seed=base)
        if not isinstance(d, dict):
            return reg
        try:
            raw = d.get("relics")
            if isinstance(raw, (list, tuple)):
                for item in raw:
                    try:
                        reg.relics.append(Relic.from_dict(item))
                    except Exception:
                        log.debug("skipped an unreadable relic", exc_info=True)
                        continue
            del reg.relics[MAX_RELICS:]
            reg.next_id = max(1, _i(d.get("next_id"), 1))
            for r in reg.relics:
                reg.next_id = max(reg.next_id, r.id + 1)
            reg.t = max(0.0, _f(d.get("t"), 0.0))
            reg.found = max(0, _i(d.get("found"), 0))
            reg.seen = _counts(d.get("seen"), 0, 1 << 30)
            reg.dry = _counts(d.get("dry"), 0, DROUGHT_MAX)
            reg.drops = max(0, _i(d.get("drops"), 0))
            # Replay the stream to where it was, so a reload CONTINUES the
            # sequence instead of restarting it. Capped, because a hand-edited
            # "draws": 1e12 must cost a fraction of a second, not the session.
            want = int(_clamp(_i(d.get("draws"), 0), 0, _MAX_REPLAY))
            for _ in range(want):
                reg._rng.random()
            reg.draws = want
        except Exception:
            log.warning("relic registry save unreadable; starting empty",
                        exc_info=True)
            return cls(seed=base)
        return reg


def _counts(raw: Any, lo: int, hi: int) -> dict[str, int]:
    """A per-relic-kind counter dict out of whatever the save actually holds.

    Always returns one entry per ``RELIC_KINDS`` and nothing else, so a save
    saying ``{"dry": {"excalibur": 9e9}}`` reads as a fresh colony rather than
    as a table the pity rule can trip over. Clamped, because a hand-edited
    drought of a billion would otherwise become a billion-fold weight.
    """
    out = {k: lo for k in RELIC_KINDS}
    if not isinstance(raw, dict):
        return out
    for k in RELIC_KINDS:
        out[k] = int(_clamp(_i(raw.get(k), lo), lo, hi))
    return out


def _record_seed(d: Any) -> int:
    """A stable seed derived from a save record that carries none.

    crc32 of what the record *does* hold, never ``hash()`` of a string: CPython
    salts str hashing per process, which would move the irreproducibility from
    per-load to per-launch rather than fixing it. Seven of those have shipped in
    this project.
    """
    parts = ["relics"]
    if isinstance(d, dict):
        parts.append(str(_i(d.get("next_id"), 0)))
        parts.append("%.4f" % _f(d.get("t"), 0.0))
        parts.append(str(_i(d.get("found"), 0)))
        raw = d.get("relics")
        if isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, dict):
                    parts.append("%s:%d" % (item.get("kind"),
                                            _i(item.get("id"), 0)))
                else:
                    parts.append("?")
    return zlib.crc32("|".join(parts).encode("utf-8", "replace"))


# ------------------------------------------------------ module-level API --
def registry_of(world: Any) -> RelicRegistry | None:
    """The world's relic registry, or None if it has not got one.

    Checks ``world.relics`` first and ``world.items`` second. The two names are
    the module's own history - this file is ``sim/items.py`` and the registry
    attribute is ``relics`` - and a reader that only knew one of them would go
    silently inert on a world that used the other. Cheap insurance, one getattr.
    """
    for name in ("relics", "items"):
        reg = getattr(world, name, None)
        if isinstance(reg, RelicRegistry):
            return reg
    return None


def ensure_registry(world: Any) -> RelicRegistry | None:
    """The world's relic registry, creating and attaching one if absent.

    world.py declares ``self.relics`` itself; this exists so the mechanic still
    works on a world built before it did - an old save loaded by a build without
    the world-side wiring, a bare test stub - rather than being silently dead.
    Seeded off ``world.seed`` where there is one, with the same 0x2E11 offset
    world.py uses, so a lazily-created registry replays identically to a
    declared one.
    """
    reg = registry_of(world)
    if reg is not None:
        return reg
    seed = getattr(world, "seed", None)
    try:
        s = (int(seed) ^ 0x2E11) if isinstance(seed, (int, float)) else None
    except Exception:
        s = None
    reg = RelicRegistry(seed=s)
    try:
        setattr(world, "relics", reg)
    except Exception:
        return None            # a stub that will not take attributes
    return reg


def relics_of(world: Any) -> list[Relic]:
    reg = registry_of(world)
    return [] if reg is None else reg.all()


def noun_for(kind: Any) -> str:
    return NOUN.get(_s(kind, ""), "relic")


def worn(agent: Any) -> str:
    """What *agent* is carrying, "" for nothing. Duck-typed; never raises.

    Validated against RELIC_KINDS on the way out, so a hand-edited save saying
    ``"relic": "excalibur"`` reads as no relic rather than as an item with no
    behaviour attached and no way to drop it. ``Stickman.from_dict`` validates
    the same thing on the way IN; both ends check because either end can be the
    one a future caller decides to trust.

    THE ONE SANCTIONED READER THAT DOES NOT COME THROUGH HERE is
    ``entities._boots_catch``, which compares ``self.relic`` to ``RELIC_BOOTS``
    directly. entities.py sits underneath almost everything in sim/ and importing
    a dependent of itself to test a string would be a worse trade than the
    duplication. Any OTHER module reading the slot should call this.
    """
    try:
        k = _s(getattr(agent, "relic", RELIC_NONE) or RELIC_NONE, RELIC_NONE)
    except Exception:
        return RELIC_NONE
    return k if k in RELIC_KINDS else RELIC_NONE


def has(agent: Any, kind: str) -> bool:
    return bool(kind) and worn(agent) == kind


def equip(agent: Any, kind: str) -> bool:
    """Put *kind* in *agent*'s one relic slot and mirror it into the old fields.

    The mirroring is the point. Dragonscale writes ``agent.armour`` and the
    BFG writes ``agent.weapon``, so every existing reader - the damage path in
    ``Stickman.hurt``, ``combat_actions._armoured``, ``throwing.can_throw``,
    ``render.draw_gear`` - keeps working with zero edits, and the two "you cannot
    do that any more" rules the BFG needs fall out for free: ``can_throw`` tests
    ``weapon != WEAPON_SPEAR`` exactly, and so does ``_armed``.

    Taking the gun therefore DESTROYS the spear the agent was holding. That is
    the trade, stated once here: nobody carries both.

    ``agent.relic`` IS A REAL FIELD ON ``entities.Stickman``, and it has to stay
    one. For a release it was not: this assignment created a bare instance
    attribute on a dataclass that had never declared the slot, so ``to_dict``
    never wrote it and every relic a colony had earned vanished on the next
    save/load. Nothing raised, nothing warned, and the docstring above claimed
    the opposite. The smoke block at the bottom of this file now round-trips a
    real ``Stickman`` for exactly that reason - do not replace it with a stub.
    """
    try:
        k = _s(kind, "")
        if k not in RELIC_KINDS:
            return False
        agent.relic = k
        if k == _MIRROR_ARMOUR:
            agent.armour = ARMOUR_DRAGONSCALE
        elif k == _MIRROR_WEAPON:
            agent.weapon = WEAPON_BFG
        return True
    except Exception:
        log.debug("equip failed", exc_info=True)
        return False


def unequip(agent: Any) -> str:
    """Empty the slot and clear the mirrors. Returns what was removed, "" if none.

    Idempotent, and safe on an agent that never had a relic field at all.
    """
    kind = worn(agent)
    try:
        agent.relic = RELIC_NONE
    except Exception:
        log.debug("unequip failed", exc_info=True)
        return RELIC_NONE
    try:
        if kind == _MIRROR_ARMOUR:
            agent.armour = ARMOUR_NONE
        elif kind == _MIRROR_WEAPON:
            agent.weapon = WEAPON_NONE
    except Exception:
        log.debug("unequip mirror failed", exc_info=True)
    return kind


def consume(world: Any, agent: Any) -> str:
    """Spend the relic *agent* is carrying: it is gone, not dropped.

    The amulet's ward, the cairnstone's raising and a BFG lost to a misfire all
    end here. Kept separate from :func:`unequip` so that "the item left the
    world" and "the item changed hands" are two different calls and
    ``stats["relics_used"]`` can only be bumped by the first.
    """
    kind = unequip(agent)
    if kind:
        _bump(world, "relics_used")
    return kind


def drop_for_kind(reg: RelicRegistry, dragon_kind: str) -> str | None:
    """What a dead *dragon_kind* leaves. None for the flyover and for junk.

    Thin front door onto :meth:`RelicRegistry.roll_drop`; the weighting, the
    drought and the pity rule all live on the registry because they are colony
    state and colony state has to be saved. Without a registry there is no
    stream and no memory, so this degrades to the kind's commonest relic rather
    than reaching for ``random``.
    """
    options = DROP_TABLE.get(_s(dragon_kind, ""))
    if not options:
        return None
    if reg is None:
        return options[0]
    try:
        return reg.roll_drop(dragon_kind)
    except Exception:
        log.debug("drop roll failed", exc_info=True)
        return options[0]


# ============================================================== smoke test ==
if __name__ == "__main__":   # pragma: no cover - headless
    import json as _json

    DT = 1.0 / 30.0

    class _Terrain:
        def ground_y(self, x: float) -> float:
            return 600.0

    class _Dragons:
        def __init__(self) -> None:
            self.slain: list[tuple[str, float]] = []

    class _Agent:
        def __init__(self, x: float = 400.0, name: str = "Ash") -> None:
            self.id, self.name = 7, name
            self.x, self.y = float(x), 600.0
            self.alive = True
            self.relic = RELIC_NONE
            self.weapon = "spear"
            self.armour = ARMOUR_NONE

    class _World:
        def __init__(self, seed: int = 11) -> None:
            self.seed = seed
            self.terrain = _Terrain()
            self.dragons = _Dragons()
            self.stats: dict[str, int] = {}
            self.lines: list[str] = []

        def log_event(self, text: str) -> None:
            self.lines.append(text)

    # --- the table is complete and the flyover is absent -------------------
    assert set(DROP_WEIGHTS) == {DRAGON_QUADRUPED, DRAGON_WYVERN,
                                 DRAGON_SERPENT, DRAGON_SKELETAL}
    assert set(DROP_TABLE) == set(DROP_WEIGHTS)
    # EVERY kind can leave EVERY relic. This is the whole reachability fix in
    # one assertion: the previous table paired one relic to one dragon, and the
    # bfg9000's dragon was the rarest and least-killed of the five, so the
    # headline item of the feature dropped ZERO times in 32 colony-hours.
    for _k, _row in DROP_WEIGHTS.items():
        assert set(_row) == set(RELIC_KINDS), (_k, sorted(_row))
        assert all(w > 0.0 for w in _row.values()), _k
        # ...and the signature relic is still the likeliest, so "the quadruped
        # drops its hide" survives as a tendency.
        assert DROP_TABLE[_k][0] == max(_row, key=lambda r: _row[r])
    assert DROP_TABLE[DRAGON_QUADRUPED][0] == RELIC_SCALE
    assert DROP_TABLE[DRAGON_WYVERN][0] == RELIC_AMULET
    assert DROP_TABLE[DRAGON_SERPENT][0] == RELIC_BFG
    assert DROP_TABLE[DRAGON_SKELETAL][0] == RELIC_BOOTS
    assert set(PITY_KINDS) <= set(RELIC_KINDS) and PITY_KINDS[0] == RELIC_BFG
    assert all(k in NOUN and k in PICKUP_LINE for k in RELIC_KINDS)
    assert drop_for_kind(RelicRegistry(seed=1), "flyover") is None
    assert RelicRegistry(seed=1).weights_for("flyover") == {}

    # --- every item drops, is picked up, is worn, and mirrors correctly ----
    for kind in RELIC_KINDS:
        w = _World()
        reg = RelicRegistry(seed=3)
        w.relics = reg
        r = reg.spawn(w, kind, 400.0)
        assert r is not None and r.y == 600.0, kind
        ag = _Agent()
        assert reg.take(r, ag, world=w), kind
        assert worn(ag) == kind and has(ag, kind)
        assert reg.count() == 0, "the relic is still on the ground"
        assert not reg.take(r, ag, world=w), "picked the same relic up twice"
        if kind == RELIC_SCALE:
            assert ag.armour == ARMOUR_DRAGONSCALE
        elif kind == RELIC_BFG:
            assert ag.weapon == WEAPON_BFG, "the gun did not replace the spear"
        # ...and dropping it puts it back and clears the mirror.
        back = reg.drop_from(w, ag)
        assert back is not None and back.kind == kind
        assert worn(ag) == RELIC_NONE
        assert ag.armour == ARMOUR_NONE and ag.weapon in ("", "spear")
    print("all five items: drop -> pick up -> wear -> drop, mirrors OK")

    # --- one slot per agent ------------------------------------------------
    w = _World()
    reg = RelicRegistry(seed=4)
    w.relics = reg
    ag = _Agent()
    a = reg.spawn(w, RELIC_SCALE, 100.0)
    b = reg.spawn(w, RELIC_BFG, 120.0)
    assert reg.take(a, ag, world=w)
    assert not reg.take(b, ag, world=w), "carried two relics at once"
    assert reg.count() == 1

    # --- the drain: a dead dragon drops exactly one relic ------------------
    w = _World()
    reg = RelicRegistry(seed=5)
    w.relics = reg
    w.dragons.slain.extend([("quadruped", 200.0), ("flyover", 300.0),
                            ("skeletal", 400.0)])
    got = reg.drain_slain(w)
    assert len(got) == 2, [g.kind for g in got]      # the flyover drops nothing
    assert all(g.kind in RELIC_KINDS for g in got), [g.kind for g in got]
    assert reg.drops == 2 and w.dragons.slain == [], "slain was not drained"
    print("drain:", [g.summary() for g in got])

    # --- one draw per drop, whatever path it goes down ---------------------
    # The pity path spends a draw it does not use, on purpose: retuning
    # PITY_DROPS must not re-phase a world that has already earned relics.
    for _kind in (DRAGON_QUADRUPED, DRAGON_WYVERN, DRAGON_SERPENT,
                  DRAGON_SKELETAL):
        reg = RelicRegistry(seed=99)
        for i in range(20):
            before = reg.draws
            got = drop_for_kind(reg, _kind)
            assert got in RELIC_KINDS, got
            assert reg.draws == before + 1, (_kind, i, reg.draws - before)
        assert reg.drops == 20 and sum(reg.seen.values()) == 20
    assert drop_for_kind(RelicRegistry(seed=1), DRAGON_QUADRUPED) in RELIC_KINDS

    # --- THE REACHABILITY CLAIM ---------------------------------------------
    # This is the assertion the whole fix exists for, and it has two halves.
    #
    # The FLOOR: a colony that has never seen the gun rolls it at ~30% instead
    # of the table's 10%, whatever it is killing. Checked against the
    # quadruped, which is the kind that actually dies (82% of kills) and whose
    # row is the least generous to the gun.
    _n = 3000
    _fresh = sum(1 for s in range(_n)
                 if RelicRegistry(seed=s).roll_drop(DRAGON_QUADRUPED)
                 == RELIC_BFG)
    assert 0.20 * _n < _fresh < 0.42 * _n, _fresh
    # ...and once it HAS one, the floor lifts and the gun goes back to being a
    # rare drop. This is what stops the fix inflating the long-run rate.
    _after = 0
    for s in range(_n):
        r = RelicRegistry(seed=s)
        r.seen[RELIC_BFG] = 1
        if r.roll_drop(DRAGON_QUADRUPED) == RELIC_BFG:
            _after += 1
    assert _after < 0.5 * _fresh, (_fresh, _after)
    print(f"floor: bfg9000 is {100.0*_fresh/_n:.0f}% of a colony's first drop "
          f"and {100.0*_after/_n:.0f}% once it has one")

    # The BACKSTOP: however the dice fall, four dragons and you have seen the
    # gun, five and you have the boots as well.
    _late = 0
    for _seed in range(60):
        for _kind in (DRAGON_QUADRUPED, DRAGON_WYVERN, DRAGON_SKELETAL):
            reg = RelicRegistry(seed=_seed)
            seq = [drop_for_kind(reg, _kind) for _ in range(PITY_DROPS + 1)]
            assert RELIC_BFG in seq, (_seed, _kind, seq)
            if seq[-1] == RELIC_BFG and RELIC_BFG not in seq[:-1]:
                _late += 1           # the backstop, not the floor, delivered it
            seq.append(drop_for_kind(reg, _kind))
            assert RELIC_BOOTS in seq, (_seed, _kind, seq)
    print(f"backstop: caught {_late} of 180 colonies the floor had missed "
          f"three drops running")
    # Once a colony has seen a kind, the pity rule never speaks for it again:
    # it can only move the FIRST one earlier, never inflate the rate.
    reg = RelicRegistry(seed=5)
    for _ in range(200):
        drop_for_kind(reg, DRAGON_QUADRUPED)
    assert reg.overdue() == [], reg.dry
    assert all(v > 0 for v in reg.seen.values()), reg.seen
    assert all(v <= DROUGHT_MAX for v in reg.dry.values()), reg.dry
    # The rule speaks for two kinds and no others - a colony that has never
    # seen a cairnstone is never handed one.
    starved = RelicRegistry(seed=6)
    starved.seen = {k: (0 if k == RELIC_CAIRN else 1) for k in RELIC_KINDS}
    starved.dry = {k: DROUGHT_MAX for k in RELIC_KINDS}
    assert starved.overdue() == [], starved.seen
    print(f"pity: the gun by drop {PITY_DROPS + 1} and the boots by drop "
          f"{PITY_DROPS + 2}, in 180/180 colonies")

    # --- the long-run mix: rare stays rarer, nothing is unreachable --------
    reg = RelicRegistry(seed=1234)
    mix = {k: 0 for k in RELIC_KINDS}
    N = 4000
    for _ in range(N):
        mix[drop_for_kind(reg, DRAGON_QUADRUPED)] += 1
    assert all(c > N * 0.05 for c in mix.values()), mix
    assert mix[RELIC_SCALE] == max(mix.values()), mix
    assert mix[RELIC_BFG] < mix[RELIC_SCALE], mix
    assert mix[RELIC_BOOTS] > mix[RELIC_BFG], mix
    print("quadruped mix over", N, "drops:",
          " ".join(f"{k} {100.0*c/N:.1f}%" for k, c in mix.items()))
    # The drought is what does that: skip a kind and its weight climbs. Probed
    # on the cairnstone, which no floor touches - PITY_FLOOR would mask the
    # effect on the two headline kinds, and masking it is exactly its job.
    probe = RelicRegistry(seed=7)
    lo = probe.weights_for(DRAGON_QUADRUPED)[RELIC_CAIRN]
    for _ in range(DROUGHT_MAX):
        probe._record_drop(RELIC_SCALE)
    hi = probe.weights_for(DRAGON_QUADRUPED)[RELIC_CAIRN]
    assert hi > lo * 1.5, (lo, hi)
    # ...and the floor is a floor, not a bonus: it lifts a weight to
    # PITY_FLOOR and never adds to one already above it, whether the weight got
    # there from the table (the skeletal's boots) or from a long drought.
    fresh = RelicRegistry(seed=8)
    for _kind, _row in DROP_WEIGHTS.items():
        _w = fresh.weights_for(_kind)
        for _k, _base in _row.items():
            _want = max(_base, PITY_FLOOR) if _k in PITY_KINDS else _base
            assert abs(_w[_k] - _want) < 1e-9, (_kind, _k, _w[_k], _want)
    heavy = RelicRegistry(seed=9)
    for _ in range(DROUGHT_MAX):
        heavy._record_drop(RELIC_SCALE)
    assert heavy.weights_for(DRAGON_QUADRUPED)[RELIC_BFG] > PITY_FLOOR

    # --- the cap recycles rather than refusing -----------------------------
    w = _World()
    reg = RelicRegistry(seed=6)
    w.relics = reg
    for i in range(MAX_RELICS * 3):
        reg.tick(w, 1.0)               # so the ones already down are older
        assert reg.spawn(w, RELIC_BOOTS, 50.0 + i * 10.0) is not None
    assert reg.count() <= MAX_RELICS, reg.count()
    # The survivors must be the NEWEST ones - a cap that ate the drop the player
    # just earned would be a bug they can see.
    assert max(r.t for r in reg.relics) <= float(MAX_RELICS), \
        [r.summary() for r in reg.relics]

    # --- decay ------------------------------------------------------------
    w = _World()
    reg = RelicRegistry(seed=7)
    w.relics = reg
    reg.spawn(w, RELIC_AMULET, 700.0)
    steps = int(RELIC_DECAY_SEC / 0.25) + 8
    for _ in range(steps):
        reg.tick(w, 0.25)
    assert reg.count() == 0, "an unclaimed relic did not rot"
    print(f"decay: gone after {RELIC_DECAY_SEC:.0f} s on the ground")

    # --- nearest ----------------------------------------------------------
    w = _World()
    reg = RelicRegistry(seed=8)
    w.relics = reg
    near = reg.spawn(w, RELIC_BOOTS, 410.0)
    reg.spawn(w, RELIC_CAIRN, 900.0)
    assert reg.nearest(400.0) is near
    assert reg.nearest(400.0, max_dist=5.0) is None
    assert reg.nearest(900.0, max_dist=RELIC_FETCH_RANGE) is not None

    # --- persistence, and the stream REPLAYS rather than restarting -------
    w = _World()
    reg = RelicRegistry(seed=1234)
    w.relics = reg
    reg.spawn(w, RELIC_SCALE, 120.0)
    reg.spawn(w, RELIC_BFG, 640.0)
    for _ in range(40):
        reg.tick(w, DT)
    for _ in range(17):
        reg.roll()
    for _ in range(5):
        drop_for_kind(reg, DRAGON_SKELETAL)
    blob = _json.loads(_json.dumps(reg.to_dict()))
    clone = RelicRegistry.from_dict(blob)
    assert clone.to_dict()["relics"] == blob["relics"]
    assert clone.draws == reg.draws == 22
    # The pity clock is colony state and has to survive a reload, or a colony
    # that saves and loads every evening never reaches PITY_DROPS.
    assert clone.seen == reg.seen and clone.dry == reg.dry, (clone.seen,
                                                             clone.dry)
    assert clone.drops == reg.drops == 5
    assert clone.overdue() == reg.overdue()
    live = [reg.roll() for _ in range(6)]
    reloaded = [clone.roll() for _ in range(6)]
    assert live == reloaded, "a reloaded registry did not continue the stream"
    # Two loads of one record must agree, or a reloaded world stops replaying.
    x = RelicRegistry.from_dict(blob)
    y = RelicRegistry.from_dict(blob)
    assert [x.roll() for _ in range(4)] == [y.roll() for _ in range(4)]
    print("persistence + replay OK; draws =", clone.draws)

    # --- beams are transient and are NOT persisted -------------------------
    w = _World()
    reg = RelicRegistry(seed=9)
    w.relics = reg
    for _ in range(MAX_BEAMS * 3):
        reg.add_beam(100.0, 580.0, 500.0, 560.0)
    assert len(reg.beams) == MAX_BEAMS, len(reg.beams)
    assert "beams" not in reg.to_dict(), "a quarter-second of light got saved"
    # Two ticks, not one: tick() clamps dt to _MAX_DT (0.25) and BEAM_LIFE_SEC
    # is 0.30, so a single huge step cannot age a beam out. That clamp is the
    # frame-hitch guard every registry here has and it is not being worked
    # around - the beam simply lives for two frames of a stalled world.
    reg.tick(w, 0.25)
    reg.tick(w, 0.25)
    assert reg.beams == [], "a beam outlived its fade"
    reg.add_beam(100.0, 580.0, 100.0, 580.0, misfire=True)
    assert RelicRegistry.from_dict(reg.to_dict()).beams == []

    # --- junk never raises -------------------------------------------------
    assert RelicRegistry.from_dict("nonsense").count() == 0
    assert RelicRegistry.from_dict(None).count() == 0
    assert RelicRegistry.from_dict({"relics": [None, 7, {"kind": "sword"}]}
                                   ).relics[2].kind in RELIC_KINDS
    assert RelicRegistry.from_dict({"draws": 10 ** 12}).draws == _MAX_REPLAY
    # A hand-edited pity clock must not become a billion-fold weight, and a
    # counter dict full of nonsense must read as a fresh colony.
    # A save written before the pity clock existed has none of its three keys.
    # It must load as a fresh clock rather than as junk - the colony simply
    # gets its grace period back, which is the generous direction to be wrong.
    legacy = RelicRegistry.from_dict({"seed": 3, "draws": 4, "relics": [],
                                      "next_id": 1, "t": 9.0, "found": 2})
    assert legacy.drops == 0 and legacy.overdue() == []
    assert set(legacy.seen) == set(legacy.dry) == set(RELIC_KINDS)
    assert legacy.roll_drop(DRAGON_QUADRUPED) in RELIC_KINDS
    junk = RelicRegistry.from_dict({"dry": {"excalibur": 10 ** 12,
                                            RELIC_BFG: 10 ** 12},
                                    "seen": "nope", "drops": -4})
    assert set(junk.dry) == set(RELIC_KINDS) and set(junk.seen) == set(RELIC_KINDS)
    assert junk.dry[RELIC_BFG] == DROUGHT_MAX and junk.drops == 0
    assert junk.weights_for(DRAGON_QUADRUPED)[RELIC_BFG] <= 1.0
    assert junk.roll_drop(DRAGON_QUADRUPED) in RELIC_KINDS
    assert Relic.from_dict(None).kind in RELIC_KINDS
    assert Relic.from_dict({"x": "nope", "t": float("nan")}).t == 0.0
    assert worn(object()) == RELIC_NONE
    assert unequip(object()) == RELIC_NONE
    assert not equip(object(), RELIC_SCALE) or True

    # --- the carried slot survives a save, on a REAL Stickman --------------
    # Every other agent in this file is a stub that will accept any attribute
    # you write to it, which is precisely why the shipped bug was invisible
    # here: equip() set `agent.relic` on a dataclass that had never declared the
    # field, python made a bare instance attribute, and to_dict dropped it. A
    # colony's entire loot evaporated on reload with nothing raising. The stubs
    # above still earn their keep (they prove the module survives a duck-typed
    # world), but the persistence claim in Relic's docstring can only be checked
    # against the real class, so it is checked against the real class.
    from .entities import Stickman as _Stickman        # noqa: PLC0415

    w = _World()
    reg = RelicRegistry(seed=12)
    w.relics = reg
    for kind in RELIC_KINDS:
        body = _Stickman(id=3, name="Ash", x=400.0, y=600.0)
        got = reg.spawn(w, kind, 400.0)
        assert got is not None and reg.take(got, body, world=w), kind
        assert worn(body) == kind, kind
        twin = _Stickman.from_dict(_json.loads(_json.dumps(body.to_dict())))
        assert worn(twin) == kind, (kind, "the relic did not survive the save")
        assert twin.armour == body.armour and twin.weapon == body.weapon, kind
        # ...and it comes off again on the far side of the reload.
        assert unequip(twin) == kind and worn(twin) == RELIC_NONE, kind
    print("carried slot round-trips on a real Stickman, all",
          len(RELIC_KINDS), "kinds")

    # --- a bare world is not an error -------------------------------------
    class _Bare:
        pass

    bare = _Bare()
    assert registry_of(bare) is None
    assert relics_of(bare) == []
    r = RelicRegistry(seed=1)
    r.tick(bare, DT)                    # must not raise
    assert r.spawn(bare, RELIC_CAIRN, 10.0) is not None
    assert ensure_registry(bare) is not None
    print("bare world OK")

    print("\nall relic checks passed")
