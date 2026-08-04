"""Shared constants. Imported by sim, render and platform layers alike.

Nothing in here may import pygame - `sim/` must stay headless-importable.
"""
from __future__ import annotations

# ---------------------------------------------------------------- geometry --
RENDER_W = 1600
RENDER_H = 1000
RENDER_SIZE = (RENDER_W, RENDER_H)

# ------------------------------------------------------------------ timing --
SIM_HZ = 30                 # fixed sim timestep
SIM_DT = 1.0 / SIM_HZ
AI_TICKS = 15               # re-score utility AI every N sim ticks (~2 Hz)
TARGET_FPS = 60             # preview window
WALLPAPER_FPS = 4           # desktop wallpaper push rate (I/O bound, ~200ms each)
AUTOSAVE_SEC = 60.0
DAY_LENGTH_SEC = 20 * 60    # one full day/night cycle in real seconds

# --------------------------------------------------------------- materials --
MAT_GRASS = 0
MAT_DIRT = 1
MAT_STONE = 2
MAT_SAND = 3
MAT_SNOW = 4
MAT_ASH = 5
MAT_MUD = 6
MAT_LAVA = 7
MAT_COUNT = 8

MATERIAL_COLORS: dict[int, tuple[int, int, int]] = {
    MAT_GRASS: (58, 92, 48),
    MAT_DIRT:  (74, 56, 40),
    MAT_STONE: (86, 88, 94),
    MAT_SAND:  (156, 140, 96),
    MAT_SNOW:  (222, 230, 240),
    MAT_ASH:   (52, 48, 46),
    MAT_MUD:   (62, 48, 34),
    MAT_LAVA:  (200, 78, 24),
}

# ------------------------------------------------------------------ agents --
MAX_SLOPE_WALK = 0.9        # |dy/dx| above this must be climbed, not walked
MAX_SLOPE_CLIMB = 2.6       # above this is a cliff: fall risk
FALL_LETHAL_SPEED = 340.0   # px/s downward impact that kills
WALK_SPEED = 34.0           # px/s baseline
CLIMB_SPEED = 15.0

# --------------------------------------------------------------- crossings --
# Geometry shared by Terrain's effective-surface overlay (which decides what an
# agent can actually walk on), structures.py (which stamps a finished bridge or
# ladder into it) and the renderer (which has to draw the same shape the
# physics is using). One source of truth for all three, because a deck drawn
# somewhere other than where it is walkable is worse than no deck at all.

#: px of blend at each end of a stamped deck. The overlay is combined with the
#: raw ground by "whichever surface is higher", so without a blend a deck
#: sitting a few px proud of its rim is a step the walk code reads as a ledge.
#: Ramping the last CROSSING_RAMP_PX down onto the raw ground guarantees the
#: join is continuous wherever the rim actually turned out to be.
CROSSING_RAMP_PX = 10.0

#: Design slope of a ladder's ramp, |dy/dx|. Must stay under MAX_SLOPE_CLIMB or
#: the physics refuses the face the ladder was built to make passable - that is
#: the entire point of the structure. The margin below 2.6 is not cosmetic: the
#: walk code also refuses a ledge when the ground drops more than CLIFF_DROP
#: (46 px) within its 14 px lookahead, and 2.2 * 14 = 31 px clears that with
#: room for the terrain under the ramp to be uneven.
LADDER_SLOPE = 2.2
#: Vertical face, in px, below which a ladder is not worth building - agents
#: can already climb anything up to MAX_SLOPE_CLIMB unaided.
LADDER_MIN_RISE = 55.0
#: Bounds on the resulting footprint. The lower bound keeps a ladder from being
#: a two-column curiosity; the upper stops one enormous escarpment from
#: producing a 400 px ramp that reads as a landslide rather than a ladder.
LADDER_MIN_W = 22
LADDER_MAX_W = 130

# ------------------------------------------------------------ reachability --
# Terrain.regions()/barriers() split the map into the stretches agents can
# actually move between. The thresholds below decide what counts as a wall.

#: Vertical relief a slope-impassable stretch must have before it is treated as
#: a barrier rather than as ordinary rough ground.
#:
#: Slope alone is not enough and was measured being useless on its own: cutting
#: at |dy/dx| > MAX_SLOPE_CLIMB shatters an ordinary hills map into 150 pieces
#: and a cliffs map into 141, because generation *guarantees* at least one such
#: face on every map and agents climb the small ones all day. What actually
#: stops a colony is a face it cannot climb (that is the slope test) which is
#: also tall enough that the one route that does exist - descending it attached
#: via Stickman.descend_step, then climbing the far side - is a coin flip on a
#: life: a slip more than FALL_LETHAL_SPEED**2 / (2 * GRAVITY) = 64 px above the
#: floor is fatal, and the descent takes relief/DESCENT_DROP_RATE seconds of
#: rolling against SLIP_CHANCE_PER_SEC the whole way.
#:
#: 150 px is a bit over twice that lethal height, and is where the measurements
#: land: across a 12-minute trace of ten seeds, every stretch that actually
#: killed anybody had relief between 162 and 392 px, and the tallest stretch on
#: the seeds that killed nobody at all was 144.
#:
#: It is deliberately not a clean separation, and cannot be: seeds 7 and 7777
#: carry 172-263 px faces and lost nobody, because those colonies never had a
#: reason to cross them. Relief is only half the test - behaviour.find_cutoff
#: still has to find the colony *losing* something to the split before anything
#: gets built.
BARRIER_MIN_RELIEF = 150.0

#: Widest crossing the colony will attempt as a single structure, px. Past this
#: the far side is not "just over there" any more; the colony crosses the first
#: barrier and re-surveys rather than staking one enormous bridge.
CROSSING_MAX_SPAN = 260.0

#: How far a gap has to sink below its *lower* rim before it counts as a hole to
#: be bridged rather than a step to be laddered. A chasm sinks 250-320 px below
#: its rims; a plateau edge sinks nothing at all - it is simply a face, and the
#: thing that defeats a face is a ladder.
CROSSING_MIN_DEPTH = 30.0

#: Steepest |dy/dx| a deliberately *cut* chasm wall is allowed to reach.
#:
#: Not a look, a survival threshold. What kills at a chasm is a slip partway
#: down a wall somebody is deliberately descending, and whether that slip kills
#: depends on the gradient of the face and on nothing else - not its height, not
#: how far the body still has to fall. _slip pitches you forward at a fixed
#: 16 px/s while gravity pulls at GRAVITY, so the fall's own trajectory descends
#: at a fixed rate; a face steeper than that falls away faster than the body
#: does and you re-contact it much further down, or not at all. Driving the real
#: fall integrator down synthetic faces: impact 282 px/s at gradient 12-13, 312
#: at 13.5-14.5, and 342 at 15 - past FALL_LETHAL_SPEED (340). So a slip is
#: self-arresting below about 14:1 and fatal above it, at any depth.
#:
#: 11 is that knee with margin for the two things that add to the wall's own
#: gradient: the +-2 px wobble on the chasm floor, and the pre-existing slope of
#: the ground the cut lands in (the blend interpolates *between* them, so a rim
#: already tilting into the hole steepens the join).
CHASM_SAFE_GRADIENT = 11.0

# Palette used to assign identity colours to stickmen. Deliberately high
# chroma so a single candle or lightning flash still reads them apart.
STICK_PALETTE: list[tuple[int, int, int]] = [
    (232, 106, 106),   # red
    (106, 176, 232),   # blue
    (128, 216, 132),   # green
    (238, 196, 104),   # amber
    (196, 132, 232),   # violet
    (104, 226, 214),   # teal
    (240, 152, 200),   # pink
    (206, 214, 226),   # bone
    (232, 150, 96),    # orange
    (160, 200, 110),   # lime
]

# --------------------------------------------------------------- resources --
RES_WOOD = "wood"
RES_STONE = "stone"
RES_FOOD = "food"
RES_COOKED = "cooked"
RES_FIBRE = "fibre"
RES_HIDE = "hide"                 # taken from a killed animal; makes armour
ALL_RESOURCES = (RES_WOOD, RES_STONE, RES_FOOD, RES_COOKED, RES_FIBRE, RES_HIDE)

#: What a villager shoulders when he sweeps litter up. Deliberately **not** in
#: ALL_RESOURCES: that tuple is what seeds ``World.stockpile``, and every
#: colony-value reading in the sim (hut growth's store fraction, the population
#: gate, the build director's availability maths) sums that dict. Garbage is a
#: chore, not wealth - it must never make a settlement look richer, and it must
#: never satisfy a build cost. ``actions.stock_add`` refuses it outright, so the
#: guarantee holds even if some future haul path tries to bank it by accident.
RES_GARBAGE = "garbage"

# ------------------------------------------------------------------ litter --
#: Mean seconds between one living villager dropping one piece of litter.
#:
#: This number is set against the pit's BURN RATE, not by taste, because getting
#: it wrong breaks the feature in two visible ways at once. A firepit consumes a
#: unit every BONFIRE_SEC_PER_GARBAGE seconds, i.e. about 4.3 a minute. At the
#: original 75 s, nine villagers dropped 7.2 a minute - a 1.68x oversupply - and
#: it was measured doing exactly what oversupply must do:
#:
#:   * the pit sat pinned at BONFIRE_GARBAGE_CAP and blazed for 81.5% of a
#:     45-minute run (99% of it after the first load), so the bonfire stopped
#:     being an event and became the permanent state of the world; and
#:   * villagers swept up 1181 pieces to have 1012 of them refused at the pit and
#:     tipped straight back onto the ground - a treadmill that looked like work.
#:
#: The corrected value was then measured too, because the failure is symmetric:
#: at 420 s (~1.3 a minute) litter was so sparse that clusters never reached
#: LITTER_CLUSTER_MIN, cleanup essentially never ran, and the bonfire fired 0.0%
#: of a 25-minute run - the feature went silent, which is worse than it being
#: too loud.
#:
#: 200 s puts nine villagers at ~2.7 a minute against the 4.3 a minute the pit
#: can burn: about 63% utilisation, so clusters still build up where people work
#: and get swept, while the pit has the headroom to burn down between loads
#: instead of sitting pinned at its cap.
LITTER_DROP_SEC = 200.0

#: Hard ceiling on litter props alive at once. The prop loop is O(n) per tick and
#: this thing runs unattended overnight - at the rate above, eight hours of nine
#: villagers would be ~3400 props without a cap.
#:
#: Enforced by recycling the OLDEST piece rather than by refusing to drop a new
#: one. That matters: a colony that has walled itself off from a pile it can no
#: longer reach (the far side of a chasm) would otherwise sit at the cap forever
#: and stop littering where it actually lives, and the feature would go quiet on
#: exactly the maps that are most interesting.
LITTER_MAX = 120

#: Litter does **not** decay. A timer that quietly deleted it would be doing the
#: colony's job for it: the whole point of the feature is that mess accumulates
#: until somebody deals with it, and a decay fast enough to matter would make the
#: cleanup job unnecessary while a decay slow enough not to would be invisible.
#: The only sinks are a villager carrying it to a fire and the recycling cap.
LITTER_DECAYS = False

#: Half-width of the window a cluster is measured over, and how many pieces have
#: to fall inside it before the colony treats it as *dense* rather than as one
#: stray item. The user was explicit that a single dropped thing is not a job.
LITTER_CLUSTER_R = 70.0
LITTER_CLUSTER_MIN = 5
#: Density at which the cleanup score saturates - past this it is not any more
#: urgent, it is just more walking.
LITTER_CLUSTER_FULL = 16

#: Hard ceiling on the cleanup score, whatever the density. Sweeping up is the
#: lowest-value job in the colony: it must sit under gathering (0.10-1.00),
#: farming (0.18+), mining (0.30+), building, eating, sleeping, warming and
#: everything combat_actions scores. Capping it absolutely - rather than trusting
#: the formula to stay small - is what makes "never outranks food, shelter or
#: defence" a property of the code instead of a property of today's tuning.
#:
#: 0.42 did not deliver that and was measured failing it: BuildStructure for a
#: gatherer with a half-built hut and every material in store scores 0.362, so
#: dense litter beat SHELTER outright, and Eat (hunger squared) only overtook at
#: hunger 0.648, so a mildly hungry villager swept instead of eating beside a
#: full larder. 0.30 restores the ordering it claims: under that same build at
#: 0.362, and under Eat from hunger 0.548.
CLEANUP_SCORE_MAX = 0.30

# ----------------------------------------------------------------- bonfire --
#: Seconds of fire one unit of garbage buys. Wood, for comparison, is
#: FIRE_STOKE_PER_WOOD / FIRE_FUEL_BURN = 0.34 * 240 = 81.6 s per unit, so
#: garbage burns nearly six times faster - it flares and is gone, which is what
#: makes a bonfire an event rather than a new steady state.
BONFIRE_SEC_PER_GARBAGE = 14.0
#: Most garbage a single firepit will hold. Bounds the bonfire at 16 * 14 = 224 s
#: however much a keen colony shovels in.
BONFIRE_GARBAGE_CAP = 16.0
#: The payoff. A well-fed firepit lights 96 + 96 px; a bonfire throws this far,
#: which visibly lifts a whole night scene rather than a doorstep.
BONFIRE_LIGHT_RADIUS = 330.0
BONFIRE_LIGHT_INTENSITY = 1.05
BONFIRE_LIGHT_COLOR = (255, 196, 120)
#: How much bigger the pit itself is drawn while it roars.
BONFIRE_DRAW_SCALE = 1.75

# ------------------------------------------------------------------ combat --
#: Stickmen have health only because something can now hurt them. Everything
#: else that kills is instant (a fall, a strike, drowning); animals are the
#: first threat you can survive, fight, and be wounded by.
MAX_HEALTH = 100.0
HEALTH_REGEN_PER_SEC = 1.0 / 6.0   # a scratch heals in a minute or so

WEAPON_NONE = ""
WEAPON_SPEAR = "spear"
#: A spear is the difference between prey and a hunting party. Unarmed agents
#: flee; armed ones will stand and fight.
SPEAR_DAMAGE = 26.0
SPEAR_REACH = 26.0
SPEAR_COOLDOWN = 1.1
SPEAR_COST = {RES_WOOD: 2, RES_STONE: 1}

#: Every stickman carries a torch, so a colony at night is a scatter of moving
#: light rather than one candle in the black. Kept a touch smaller and oranger
#: than the elder's candle so a cluster reads as many pools, not one floodlight,
#: and the lightning reveal still has darkness to cut through between them.
TORCH_RADIUS = 96.0
TORCH_COLOR = (255, 146, 54)
TORCH_INTENSITY = 0.70
TORCH_FLICKER = 0.38

ARMOUR_NONE = 0.0
ARMOUR_LEATHER = 0.45              # fraction of incoming damage absorbed
ARMOUR_COST = {RES_HIDE: 2, RES_FIBRE: 1}

# ----------------------------------------------------------------- animals --
ANIMAL_WOLF = "wolf"
ANIMAL_BEAR = "bear"
ANIMAL_BOAR = "boar"
ANIMAL_KINDS = (ANIMAL_WOLF, ANIMAL_BEAR, ANIMAL_BOAR)

#: (max_health, damage/s, speed px/s, hides dropped, pack size)
ANIMAL_STATS: dict[str, tuple[float, float, float, int, int]] = {
    ANIMAL_WOLF: (58.0,  8.0, 46.0, 1, 3),
    ANIMAL_BEAR: (140.0, 15.0, 30.0, 3, 1),
    ANIMAL_BOAR: (78.0, 11.0, 54.0, 2, 1),
}

#: Seconds between attempted incursions. A pack that arrives every couple of
#: minutes is a threat; one that arrives constantly is just a death clock.
ANIMAL_SPAWN_MIN = 150.0
ANIMAL_SPAWN_MAX = 420.0
ANIMAL_MAX_ALIVE = 4
ANIMAL_LEAVE_SEC = 90.0            # give up and wander off after this long

# --------------------------------------------------------------------- ufo --
#: Rare enough to be an event you tell someone about.
UFO_INTERVAL_MIN = 420.0
UFO_INTERVAL_MAX = 1500.0
UFO_BEAM_SEC = 4.5                 # hover + beam before the lift completes
UFO_RETURN_CHANCE = 0.25           # sometimes they get dropped back, dazed

# -------------------------------------------------------------- population --
#: The colony breathes between these. Deaths are no longer replaced one-for-one
#: - the number is allowed to fall, and a colony with food, shelter and morale
#: grows back on its own. A fixed headcount made every disaster consequence-free.
MIN_POP = 2                     # below this, newcomers arrive to save the line
MAX_POP = 10
POP_BIRTH_COOLDOWN = 95.0       # min seconds between new arrivals
#: Stored food per existing colonist needed before they will take on another
#: mouth. Scales with headcount, so growth slows as the colony gets big.
FOOD_PER_HEAD_TO_GROW = 5
MORALE_TO_GROW = 0.45
#: Sleeping space gates growth too: roughly this many people per finished hut.
POP_PER_HUT = 3

# ----------------------------------------------------------- hut tier ------
#: The colony earns masonry by being *content*, not by hitting a resource
#: target - see World._tick_hut_tier. Colony morale here is the mean over the
#: living, the same number MORALE_TO_GROW above is compared against, so the two
#: gates cannot drift apart.
#:
#: Both values were measured over 26 seeds x 50 sim-min, sampling colony mean
#: morale once per sim-second. Pooled mean morale is 0.496, so 0.55 sits a
#: little above typical: 21/26 colonies unlock, earliest at 15.0 min, median
#: 24.2 min, and the five that never do miss by only 7-65 s of dwell. 0.60 is a
#: stretch nobody sustains (the longest run above it has a median of 143 s,
#: shorter than any usable dwell) and 0.65 unlocked 1 seed in 16 - dead code.
HUT_TIER_MORALE = 0.55          # colony-mean morale that counts as "content"
#: ...and it must be SUSTAINED, unbroken, for this long. The dwell is what
#: stops the opening honeymoon from handing the tier over for free: agents spawn
#: at morale 0.6 with every need at zero, so colony mean stays above 0.55 for the
#: first 47-132 s of every seed measured. 180 s clears the worst observed
#: honeymoon by 48 s; 150 s clears it by 18 s, which is too thin a margin to
#: trust on a seed nobody sampled. Anything at or under ~140 s fires in the first
#: two minutes regardless of threshold - that was measured too, and rejected.
HUT_TIER_DWELL_SEC = 180.0      # seconds it must be sustained, unbroken

# ---------------------------------------------------------- hut growth ------
#: A hut is not a fixed object. It grows with the time it has stood and with
#: how well stocked the colony is, so a long-lived, well-fed camp visibly
#: becomes a bigger settlement rather than staying four identical boxes.
HUT_GROWTH_AGE_SEC = 900.0      # seconds of standing to reach full age credit
HUT_GROWTH_STORE_REF = 60       # stockpile total that counts as "well stocked"
HUT_SCALE_MIN = 1.0
HUT_SCALE_MAX = 1.85
#: Age and storage each contribute this share of the final growth value.
HUT_AGE_WEIGHT = 0.55
HUT_STORE_WEIGHT = 0.45

# ------------------------------------------------------- resource regrowth --
#: More hands, faster the wild recovers. Deliberately the opposite of the usual
#: depletion curve: it is what keeps a colony of ten from stripping the map
#: bare and starving, so population can actually range instead of collapsing.
REGROW_BASE = 1.0               # multiplier at MIN_POP
REGROW_PER_HEAD = 0.22          # added per colonist above MIN_POP
REGROW_MAX = 3.0

# --------------------------------------------------------- player tools ----
#: The left-mouse tool palette, shown top-left in the preview. The player picks
#: one, then left-clicks the world to use it. Right-mouse always pans; left is
#: never pan. Order here is the on-screen order.
TOOL_NONE = "none"          # just watch; clicks do nothing
TOOL_HAND = "hand"          # grab a prop or stickman, drag, then place or toss
TOOL_LIGHTNING = "lightning"  # smite the clicked spot
TOOL_METEOR = "meteor"      # crater the clicked spot
TOOL_PLANT = "plant"        # grow a tree where clicked
TOOL_ROCK = "rock"          # drop a boulder/rock where clicked
TOOL_FEED = "feed"          # bless the colony with food at the spot
TOOL_SPAWN = "spawn"        # a newcomer arrives where clicked
TOOLS = (TOOL_HAND, TOOL_LIGHTNING, TOOL_METEOR, TOOL_PLANT, TOOL_ROCK,
         TOOL_FEED, TOOL_SPAWN)
TOOL_LABELS = {
    TOOL_HAND: "Hand - grab and toss", TOOL_LIGHTNING: "Lightning",
    TOOL_METEOR: "Meteor", TOOL_PLANT: "Plant a tree", TOOL_ROCK: "Drop a rock",
    TOOL_FEED: "Feed the colony", TOOL_SPAWN: "New arrival",
}
#: How close to the click a grab looks for something to pick up, in px.
GRAB_RADIUS = 46.0
#: A release faster than this (px/s) is a throw, not a placement. A hard throw
#: can hurt: a stickman flung off a cliff still takes fall damage on landing.
TOSS_SPEED = 240.0
FEED_AMOUNT = 8             # food added per Feed click

# ------------------------------------------------------- farming & mining --
#: Farming grows food (distinct from foraging wild berries); mining digs stone
#: from boulders and stone ground (distinct from the quick GatherStone off loose
#: rocks). Both are sustained, visible jobs meant to read as work.
FARM_FIELD_SIZE = 8         # crops a colony tends toward
FARM_HARVEST_FOOD = 4       # food per ripe crop
FARM_TILL_SEC = 6.0         # planting a new crop
MINE_YIELD_STONE = 2        # stone per mining yield tick
MINE_YIELD_SEC = 4.0        # seconds of digging per yield
MINE_SESSION_SEC = 24.0     # how long a miner works one spot before moving on

# ------------------------------------------------------------- defense -----
#: Spiked barricades the colony raises near the world's edges - where animals
#: come in from - so invaders are hurt (and often killed) before they reach the
#: settlement. Passive: they damage any wild animal that lingers in range, never
#: a stickman.
BARRICADE_DAMAGE = 22.0     # hp/s dealt to an animal within range
BARRICADE_RANGE = 48.0      # px
BARRICADE_EDGE_FRAC = 0.16  # a barricade sits within this fraction of an edge
BARRICADE_MIN_POP = 4       # colony builds them once it is this many strong

# ------------------------------------------------------------------ scenes --
SCENE_NIGHT_STORM = "night_storm"
SCENE_CLEAR = "clear"
SCENE_WILDFIRE = "wildfire"
SCENE_MUDSLIDE = "mudslide"
SCENE_BLIZZARD = "blizzard"
SCENE_FLOOD = "flood"
SCENE_METEOR = "meteor"
SCENE_ASHFALL = "ashfall"
SCENE_AURORA = "aurora"          # calm deep-night sky lit by aurora ribbons
SCENE_FOG = "fog"                # a still grey mist that swallows the distance
SCENE_ECLIPSE = "eclipse"        # the sun is eaten, then given back
SCENE_EARTHQUAKE = "earthquake"  # quiet, then the ground tears itself open
SCENE_SANDSTORM = "sandstorm"    # a howling wall of grit; the arid blizzard
SCENE_HEATWAVE = "heatwave"      # glaring, still drought; bites the economy
SCENE_VOLCANO = "volcano"        # a vent opens and lava crawls out of it

SCENES = (
    SCENE_NIGHT_STORM, SCENE_CLEAR, SCENE_WILDFIRE, SCENE_MUDSLIDE,
    SCENE_BLIZZARD, SCENE_FLOOD, SCENE_METEOR, SCENE_ASHFALL,
    SCENE_AURORA, SCENE_FOG, SCENE_ECLIPSE, SCENE_EARTHQUAKE,
    SCENE_SANDSTORM, SCENE_HEATWAVE, SCENE_VOLCANO,
)

#: The world flips to a fresh, randomly chosen scene this often (world-seconds).
#: A manual scene pick from the tray resets the countdown, so a deliberately
#: chosen scene still gets a full interval before the weather moves on.
SCENE_ROTATE_SEC = 600.0        # 10 minutes

SCENE_LABELS = {
    SCENE_NIGHT_STORM: "Night Storm",
    SCENE_CLEAR:       "Clear Skies",
    SCENE_WILDFIRE:    "Wildfire",
    SCENE_MUDSLIDE:    "Mudslide",
    SCENE_BLIZZARD:    "Blizzard",
    SCENE_FLOOD:       "Flood",
    SCENE_METEOR:      "Meteor Shower",
    SCENE_ASHFALL:     "Volcanic Ashfall",
    SCENE_AURORA:      "Aurora Night",
    SCENE_FOG:         "Fog",
    SCENE_ECLIPSE:     "Solar Eclipse",
    SCENE_EARTHQUAKE:  "Earthquake",
    SCENE_SANDSTORM:   "Sandstorm",
    SCENE_HEATWAVE:    "Heatwave",
    SCENE_VOLCANO:     "Volcanic Eruption",
}

# ------------------------------------------------------------------ saving --
SAVE_VERSION = 1

# ------------------------------------------------------------- body morphs --
#: Almost every stickman is built the same. A rare few are not: a giant, a
#: dwarf, a barrel of a man, a spindly one. They are a novelty you notice once
#: in a while - never the norm - so the roll below is deliberately mean.
MORPH_NONE = ""
MORPH_GIANT = "giant"
MORPH_TINY = "tiny"
MORPH_STOUT = "stout"
MORPH_LANKY = "lanky"

#: name -> (height scale, girth). Only the *name* is stored on an agent; these
#: two numbers are what actually drive the drawing, so a morph can be retuned
#: here without touching a single save.
#:
#: The height scale multiplies the *role* height rather than replacing it, so a
#: giant child is still visibly a child. Girth is the width factor: it broadens
#: the stance, spreads the limbs, thickens the drawn lines and grows the head
#: against the body. Without it "fat" and "big" are the same silhouette, since
#: a stick figure scaled uniformly reads only as distance.
#:
#: ``tiny`` is deliberately given girth > 1: a small figure with ordinary
#: proportions is indistinguishable from a child at this scale, and a squat one
#: reads as a dwarf instead.
MORPH_TABLE: dict[str, tuple[float, float]] = {
    MORPH_GIANT: (1.45, 1.25),
    MORPH_TINY:  (0.62, 1.20),
    MORPH_STOUT: (0.96, 1.85),
    MORPH_LANKY: (1.30, 0.62),
}

#: Hard bounds on whatever comes out of that table. Height is not purely
#: cosmetic - it positions the candle light and the head - and the physics
#: (GROUND_SNAP, STEP_DROP_MAX, the climb probe) is written for a body roughly
#: AGENT_HEIGHT tall, so a table hand-edited to 5.0 must produce a strange
#: villager rather than one that clips through terrain.
MORPH_SCALE_MIN, MORPH_SCALE_MAX = 0.60, 1.60
MORPH_GIRTH_MIN, MORPH_GIRTH_MAX = 0.55, 2.10

#: Chance of each morph, rolled once per newcomer at spawn. Sums to 0.07: 93 of
#: every 100 arrivals are ordinary, which is the point - a mutant should be
#: something you notice and point at, not a fifth of the colony.
MORPH_SPAWN_CHANCE: dict[str, float] = {
    MORPH_GIANT: 0.015,
    MORPH_TINY:  0.015,
    MORPH_STOUT: 0.020,
    MORPH_LANKY: 0.020,
}

#: Chronicle phrasing. names.describe_event has no template kind for a body, so
#: world.py builds the line straight from these.
MORPH_LABELS: dict[str, str] = {
    MORPH_GIANT: "a giant, head and shoulders above the rest",
    MORPH_TINY:  "no bigger than a stump, and quick with it",
    MORPH_STOUT: "broad as a boulder",
    MORPH_LANKY: "all elbows and shins",
}

# ---------------------------------------------------------------- altitude --
# Two numbers the SIM reads that describe things RENDER draws. Nothing in this
# world had an altitude until dragons, so every reach test in the package is
# one-dimensional (x only) and was *correct* while that was true - animals.py's
# own header says "Nothing here flies or falls". These are the ceilings that
# make those tests two-dimensional for the handful of callers that must care.

#: Px. How tall a barricade's spikes are, i.e. the highest thing they can bite.
#:
#: This is not a taste number: it is the atlas canvas height for the "barricade"
#: sprite, ``(60, 58)`` at render/atlas.py:220. Sim and render MUST agree on how
#: tall the spikes are or the hitbox stops matching the picture - and nothing
#: would fail if they drifted, which is exactly why it is written down here with
#: its source. **If that canvas changes, THIS CHANGES TOO.**
#:
#: The bug this exists to close was measured before it was fixed
#: (scratchpad/barricade_altitude_probe.py, driven against the real
#: Structure.update with a duck-typed flying stub): a dragon 400 px above a
#: barricade took the full BARRICADE_DAMAGE of 22.0 hp/s, byte-for-byte
#: identical to a wolf standing in the spikes, because the predicate at
#: structures.py:1292 tests ``abs(a.x - self.x)`` and nothing else.
BARRICADE_SPIKE_H = 58.0

#: Px. How high a stickman on the ground can reach with a spear held overhead -
#: AGENT_HEIGHT (26) plus the thrust. Above this a melee attacker is swinging at
#: air and must throw instead; combat_actions breaks off rather than standing
#: underneath something 200 px up jabbing at it.
MELEE_CEILING = 44.0

# ----------------------------------------------------------------- dragons --
#: The rarest thing in the world, and the only one that has to be *earned*.
#:
#: THE GATE is two standing stone huts - ``kind == "hut"``, built, not ruined,
#: ``state["material"] == "stone"`` - latched the first tick it is true, and
#: never unlatched (a fire that ruins a stone hut does not un-notice the
#: colony), exactly like ``hut_tier_unlocked``.
#:
#: Chosen by measurement, 20 seeds x 75 sim-min sampled once per sim-second
#: (scratchpad/gate_measure.py). Every cheaper candidate fires on a colony that
#: merely *started* rather than one that prospered: built>=4 fired on 20/20 at a
#: median of 7.8 min, pop>=6 on 20/20 at 19.1 min, and hut_tier_unlocked alone
#: is a latch on a MORALE dwell, so it fires on a happy colony that has built
#: nothing (earliest 9.6 min). GENERATION was disqualified outright: it counts
#: ARRIVALS, not lineage depth - end-of-run generation was 15-33 on every seed -
#: so it is a churn counter wearing a tech counter's name.
#:
#: stone_hut>=2 is the only candidate that requires all three of the masonry
#: TECH (hut_tier_unlocked is a hard prerequisite: tier+stone>=1 measured
#: identically to stone>=1, proving stone huts cannot precede the tier), a real
#: stone SURPLUS hauled and applied twice, and enough SHELTER that a second hut
#: was worth upgrading. 16/20 colonies reach it inside 75 min - earliest 21.9,
#: median 45.3, latest 67.8 - and four never do (two failed colonies, two merely
#: slow). Rare, late, and some colonies never see a dragon at all.
#:
#: A DWELL WAS MEASURED AND REJECTED, not skipped: requiring 60/120/240 s of
#: sustained >=2 stone huts (scratchpad/gate_measure2.py) found ZERO drops back
#: below 2 across all 16 seeds that reach the gate, over 430-3184 s of post-gate
#: time each. A dwell buys nothing measurable and costs 1.0-2.3 min of median
#: delay, so the latch alone is the mechanism.
DRAGON_GATE_STONE_HUTS = 2

#: Seconds of quiet after the latch before the first dragon. The gate says the
#: colony is ready; this says the world is not going to say so in the same
#: breath.
DRAGON_FIRST_DELAY = 480.0
#: Seconds between visits thereafter. Compare UFO_INTERVAL_MIN/MAX (420/1500):
#: dragons are deliberately rarer than the saucer. The measured post-gate window
#: is a median of 29.4 sim-min in a 75-min run; minus the 8-minute grace that
#: leaves ~21 min against a mean interval of 27.5 min, so roughly half of
#: unlocked colonies see one dragon and the rest see none - about 40% of all
#: colonies meet a dragon in 75 sim-min.
DRAGON_INTERVAL_MIN = 900.0
DRAGON_INTERVAL_MAX = 2400.0

#: A dragon cannot be hurt until it has fed (see dragons.Dragon.hurt). That rule
#: creates one failure - an unsated dragon with nobody reachable is an immortal
#: monster parked on the map forever - and this is the hatch that closes it.
#: After this many seconds with NO eligible victim (everyone indoors, everyone
#: dead but the last one, everyone already taken) it latches sated anyway and
#: becomes killable on ordinary terms. Hiding indoors stays a real defence; it
#: just costs ninety seconds and turns the monster mortal.
DRAGON_STARVE_SEC = 90.0

#: Seconds a wyvern holds a colonist before the lift completes, and the same
#: window the serpent needs for a take. This is the ONE stretch of a wyvern's
#: visit spent under BARRICADE_SPIKE_H and MELEE_CEILING - it flies over the
#: traps for everything else - so this number is how long the fight lasts.
DRAGON_SEIZE_SEC = 3.0

#: Seconds a fed dragon spends GORGING - low, heavy and killable - before it
#: starts the climb out. This is the payoff half of the feeding rule, and it
#: exists because the rule shipped with only its first half working.
#:
#: MEASURED, and this is the defect it closes: the chronicle wrote "The wyrm has
#: fed. It can be hurt now." and then took the dragon out of reach on the very
#: next state transition. Over 16 seeds x 90 sim-min a fed dragon was under a
#: thrower for a median of 1.2 s and inside melee for 0.3 s, and every wyvern
#: that ever fed left with its health bar untouched - 0 of 3, not one point of
#: damage. Two independent reasons, both of them the same mistake:
#:
#:  1. ``_devour`` set ``sated`` and moved straight to PHASE_LEAVE, whose first
#:     act is a 96 px/s climb. A wyvern was inside MELEE_CEILING (44) for about
#:     0.35 s after the sentence promising it could be hurt.
#:  2. ``combat_actions.animal_leaving`` treats ANY target whose ``state``
#:     reads "leave" as walking off the map and refuses to spend a spear on it.
#:     So the instant a dragon fed, every thrower in the colony stopped
#:     considering it - for the whole rest of its life, at any altitude.
#:
#: The gorge is therefore a phase of its own and not a slower leave: the name
#: is load-bearing, because "leave" is a word another module reads.
#:
#: MEASURED after: 17.7 s inside a thrower's ceiling and 13.4 s inside melee per
#: fed wyvern (median over six seeds), against its 180 hp - one armed adult who
#: commits, or five spears. A serpent gets 17.2 s of both, because it crawls
#: home along the surface instead of climbing, and 200 hp to spend it on.
#:
#: What that buys, end to end: over 24 seeds an ARMED colony brought down 6 of
#: the 24 wyverns that fed on it, and 6 of the 18 that got away left at 24 hp of
#: 180 - one more spear. Over 18 seeds of a forced regime (203 visits, 168 of
#: them killable) colonies killed 93, about half - quadruped 24 of 31, skeletal
#: 38 of 69, wyvern 23 of 49, serpent 8 of 19. On the unforced 16 x 90 the
#: window went from 1.2 s to 17.7 s and the kills from 2 of 7 to 4 of 6.
#: See sim/dragons.py for the two legs.
DRAGON_GORGE_SEC = 12.0

#: Seconds a quadruped stays on the ground once it has landed. It is grounded
#: for its entire dangerous phase, which is the payoff for having an altitude at
#: all: barricades, melee and thrown spears all reach it, so a colony that built
#: defences finally has a trap that does what the player asked it to do.
DRAGON_SIEGE_SEC = 150.0
#: Seconds between headstones a skeletal takes, and how many it takes before it
#: goes. MAX_GRAVES is 10 (world.py), so four is a visible bite out of a
#: resource the player has watched accumulate all run - and it costs no lives
#: and no buildings, which is the whole point of that kind.
DRAGON_SCOUR_SEC = 12.0
DRAGON_SCOUR_MAX = 4

#: Px. Anything a grounded quadruped comes this close to is eaten.
DRAGON_MAW_REACH = 30.0
#: Hp/s a roosting quadruped does to the nearest built structure in reach. Over
#: a full DRAGON_SIEGE_SEC that is 900 hp of damage if it is never interrupted -
#: real structural loss, and the reason this is the expensive dragon.
DRAGON_RAZE_DPS = 6.0

#: One at a time, ever. Two dragons is not twice the event, it is a different
#: game - and MAX_DRAGONS_ALIVE is also what keeps the visit rare rather than
#: the interval, which only spaces the *starts*.
MAX_DRAGONS_ALIVE = 1

DRAGON_FLYOVER = "flyover"
DRAGON_WYVERN = "wyvern"
DRAGON_QUADRUPED = "quadruped"
DRAGON_SERPENT = "serpent"
DRAGON_SKELETAL = "skeletal"
#: Five genuinely different machines, not one body with five skins. See
#: sim/dragons.py for what each one actually does.
DRAGON_KINDS: tuple[str, ...] = (
    DRAGON_FLYOVER, DRAGON_WYVERN, DRAGON_QUADRUPED, DRAGON_SERPENT,
    DRAGON_SKELETAL,
)

#: ``(max_health, speed px/s, cruise altitude px)``.
#:
#: The altitudes are the load-bearing column, not the health: they are what
#: decides which weapon can reach which dragon, and they were chosen against
#: MELEE_CEILING (44), BARRICADE_SPIKE_H (58) and throwing.THROW_MAX_RANGE
#: (240) rather than by eye:
#:
#:   flyover    360  - above THROW_MAX_RANGE. Unreachable BY DESIGN; the only
#:                     one that is. It is atmosphere, not an event.
#:   wyvern     180  - out of reach on the cruise, inside every reach during
#:                     the seize (alt 6-14). That window is the fight.
#:   quadruped  110  - only while off-screen. It spends its whole visit at
#:                     alt 0, where everything reaches it.
#:   serpent      9  - inside BARRICADE_SPIKE_H and MELEE_CEILING for its
#:                     entire surfaced phase, but barricades are sited within
#:                     BARRICADE_EDGE_FRAC of a screen EDGE and a chasm is not
#:                     an edge, so the traps are in the wrong place for it by
#:                     construction.
#:   skeletal    85  - ABOVE melee and spikes, well inside throwing range. The
#:                     throw-only dragon, killable from the first second
#:                     (sated at birth), which makes it the throw's tutorial.
#:
#: Health is a first pass and is NOT measured. It cannot honestly be: adding a
#: scored action re-phases world.pyrng hard enough to move mean deaths by 16%
#: over 24 seeds, so any balance number diffed against a pre-change baseline is
#: inside the noise floor before the mechanic does anything. These are set so
#: the arithmetic is legible - a wyvern is 6 thrown spears (THROW_DAMAGE 34), a
#: quadruped is 15 s of barricade (BARRICADE_DAMAGE 22 hp/s) - and are to be
#: retuned only behind the phase-locked A/B protocol.
DRAGON_STATS: dict[str, tuple[float, float, float]] = {
    DRAGON_FLYOVER:   (120.0, 44.0, 360.0),
    DRAGON_WYVERN:    (180.0, 96.0, 180.0),
    DRAGON_QUADRUPED: (320.0, 54.0, 110.0),
    DRAGON_SERPENT:   (200.0, 62.0,   9.0),
    DRAGON_SKELETAL:  (150.0, 70.0,  85.0),
}

#: How many colonists one visit may eat, per kind. Lifted from animals.py's "one
#: kill each, then they go" rule, which was measured: without a cap a wolf pack
#: that could not be fought took 15 people in 300 s, because the replacement
#: path kept feeding it new arrivals. A dragon has the same failure available to
#: it and the same answer. The quadruped's 3 is deliberately the exception - it
#: STAYS, so a colony that throws bodies at it loses three and no more.
DRAGON_MAX_FED: dict[str, int] = {
    DRAGON_FLYOVER: 0,
    DRAGON_WYVERN: 1,
    DRAGON_QUADRUPED: 3,
    DRAGON_SERPENT: 1,
    DRAGON_SKELETAL: 0,
}
