"""Shared constants. Imported by sim, render and platform layers alike.

Nothing in here may import pygame - `sim/` must stay headless-importable.
"""
from __future__ import annotations

# ---------------------------------------------------------------- geometry --
#: How much land exists, in px. Until now this WAS ``RENDER_W``: the world and
#: the frame were the same object because they had always been the same number,
#: and 201 uses of ``RENDER_W`` across the codebase silently meant one or the
#: other. They are now separate. Anything that clamps a position, sites a spawn,
#: sizes the heightmap or scatters scenery means THIS one.
WORLD_W = 6400

#: The camera, and the wallpaper image. Deliberately UNCHANGED at 1600x1000:
#: shell/wallpaper.py resizes the render surface onto the desktop with no aspect
#: correction, and the user picked this shape over a larger frame for exactly
#: that reason. Anything drawn - parallax, sky, HUD anchors, screen-space
#: effects, culling, the letterbox - means THIS one.
RENDER_W = 1600
RENDER_H = 1000
RENDER_SIZE = (RENDER_W, RENDER_H)

#: World px per frame px, i.e. how much more land there is than there used to
#: be. The single multiplier for anything authored as a count-PER-MAP: prop
#: scatter, terrain feature cells, landmark carving. NOT for anything authored
#: per-colonist, per-second or per-frame - multiplying a rate by this puts four
#: times the events on the map and still shows you the same number of them,
#: because three quarters of them are off-camera.
WORLD_SCALE = WORLD_W / RENDER_W          # 4.0

#: Half the camera, in world px. The world is wider than the view, but the view
#: is always centred on the colony, so ``|x - colony_center()| <= STAGE_HALF``
#: is "on screen" without sim/ ever learning that a camera exists. This is the
#: third meaning RENDER_W used to carry and the easiest one to lose: code that
#: meant "somewhere the player can see" reads identically to code that meant
#: "somewhere on the map", and on a 1600 px world both were true.
STAGE_HALF = RENDER_W * 0.5               # 800.0

#: ...and its complement: far enough from the colony that nothing can be seen
#: there, whatever the camera is doing. Where animals walk in from and leave by,
#: where a dragon's airborne leg lives, where barricades stand. The 160 px of
#: slack over STAGE_HALF is the camera's own DEADZONE, so a spawn placed here
#: cannot pop into view because the camera happened to be mid-ease.
OFFSTAGE = STAGE_HALF + 160.0             # 960.0

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
#:
#: 120 -> 240 with MAX_POP 10 -> 20. Litter is dropped per-person-per-time, so
#: twice the roster is twice the drop rate and the old cap would be hit twice as
#: fast and then pin there - the failure mode the recycling note above describes,
#: made permanent. Doubled and NOT quadrupled: the wider map does not make
#: anybody drop more, and the number the cap is defending is the O(n) prop tick,
#: which is a headcount cost rather than an area one.
LITTER_MAX = 240

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

#: Seconds into the beam at which an amulet-bearing victim wards it off and the
#: saucer comes apart. **Must stay strictly below UFO_BEAM_SEC**, and the check
#: must sit ABOVE the completion branch in ``Ufo._tick_beam``, because that is
#: the whole accounting argument for the item:
#:
#:   * not 0.0 - BEAM_FADE is 0.7 s, so at t=0 there is no beam drawn yet and the
#:     saucer appears to detonate for nothing;
#:   * not at completion - ``_complete_abduction`` is the line that calls
#:     ``pop.remove()`` and bumps ``stats["abducted"]``. Firing after it means
#:     either double-booking the roster or un-booking it, and ``reconcile()``'s
#:     residual (births + returned - deaths - abducted - alive) stops being zero.
#:
#: Firing strictly before it means ``_complete_abduction`` is NEVER CALLED on a
#: warded beam: the roster never changes, no stat moves, and the ledger needs no
#: new term. That corruption has already been found and fixed once in this
#: project (see world.reconcile) - do not reintroduce it by moving this number
#: above UFO_BEAM_SEC or by moving the check below the completion branch.
UFO_WARD_AT = 2.4
#: Seconds the wrecked hull takes to fall from hover altitude to the ground.
UFO_WRECK_FALL_SEC = 1.4
#: Seconds before a wrecked saucer is replaced. Deliberately long (40 min against
#: a measured 3.04 abductions per colony-hour): one amulet must not switch the
#: abduction system off for the rest of the colony's life. The ward is an EVENT,
#: not a state.
UFO_WRECK_RESPAWN = 2400.0

# -------------------------------------------------------------- population --
#: The colony breathes between these. Deaths are no longer replaced one-for-one
#: - the number is allowed to fall, and a colony with food, shelter and morale
#: grows back on its own. A fixed headcount made every disaster consequence-free.
MIN_POP = 2                     # below this, newcomers arrive to save the line
#: Raised 10 -> 20. Things that key off this and are worth knowing: the HUD
#: roster draws one two-line block per living colonist, so a full house is a
#: 588 px panel (941 px at the default 1.6 HUD scale) against a 1000 px frame -
#: it fits, with less room than before. POP_PER_HUT means a full colony now
#: wants roughly twice the huts, and FOOD_PER_HEAD_TO_GROW scales per head, so
#: the food bar to keep growing rises with the cap rather than staying flat.
MAX_POP = 20
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

# ------------------------------------------------------ surplus economy -----
#: What a colony wants standing in the store, per head, for every resource a
#: scored job can actually produce. ``behavior._stock_state`` turns each into
#: two numbers - how far below target ("short") and how far past it ("glut") -
#: and every material job in ``score_actions`` is driven off that one pair.
#:
#: Before this existed the ONLY demand signal for wood, stone and fibre was
#: "the head of the build queue cannot afford it". Measured over 8 seeds x 45
#: sim-min: stone urgency read exactly 0.0 in 93.5-95.7% of snapshots, wood in
#: 58.7-85%, and mean head-of-queue availability was 0.878-0.934. Nothing was
#: ever blocked, so GatherStone sat at 0.05 - below Wander - and the colony
#: quarried nothing on spec. Stone gross ran 25.1/colonist-hour against food's
#: 169.4. A standing per-head target is what gives those jobs a reason to exist
#: between builds.
#:
#: RES_STONE 3.0 / floor 6.0 is today's ``max(6, pop * 3)`` from the old inline
#: Mine expression, kept to the digit so mining is unchanged where it already
#: worked. RES_WOOD floor 12.0 and RES_FIBRE floor 6.0 are the scales
#: ``_gather_urgency`` already passes for those resources.
#:
#: RES_FOOD 8.0 is NOT a taste number and must not be lowered casually. The
#: birth gate in ``World._tick_population`` refuses a child while stored food is
#: under ``FOOD_PER_HEAD_TO_GROW * n`` = 5/head. The old scoring target was
#: ``max(8, pop * 4)`` = 4/head, i.e. BELOW the gate, so a colony that damped
#: its own farming down to its own target would have parked itself permanently
#: under its own reproduction threshold - a silent stall with no crash, no
#: chronicle line and no visible cause. 8/head is 1.6x the gate. Anyone editing
#: FOOD_PER_HEAD_TO_GROW must move this with it.
STOCK_PER_HEAD = {
    RES_FOOD: 8.0,              # raw + cooked, counted together
    #: Meals standing ready, on top of the raw pile - and THE lever on cooking.
    #: It was 1.5, and at 1.5 no score change could have made cooking visible:
    #: a colony of eight wanted 12 meals, a cook delivers 3 at a time, so four
    #: trips satisfied the entire standing demand and CookFood correctly shut
    #: itself off. The measured symptom was a colony sitting on a mean 175.6 raw
    #: food against 6.7 cooked. Raising the score alone (COOK_PILE_BONUS, with
    #: this still at 1.5) moved CookFood 1.87% -> 3.42% of colonist-time and
    #: stopped there, because the demand ceiling and not the score was binding.
    #:
    #: 4.0 is safe in a way no other per-head target here is, and the reason is
    #: worth stating: cooking is a CONVERSION. `_h_cook` returns exactly the 3
    #: units it took, and RES_FOOD above counts raw and cooked together, so this
    #: number cannot change how much food the colony has - only which bin it
    #: sits in. It says "hold about a third of the larder as prepared meals",
    #: which is also the better bin: eating cooked is hunger -0.78 / morale
    #: +0.09 against raw's -0.52 / +0.04.
    #:
    #: Measured at 4.0 over 5 seeds x 30 sim-min against an exact baseline:
    #: CookFood 1.87% -> 6.61% of colonist-time, cooked gross 14.5 -> 43.8 per
    #: colonist-hour (+203%), mean cooked stock 6.7 -> 39.9 against a mean pop
    #: whose target is ~32 - i.e. the pantry settles just past target and the
    #: glut term takes over, which is the loop closing on its own rather than a
    #: number that happens to look right.
    RES_COOKED: 4.0,
    RES_WOOD: 3.0,
    RES_STONE: 3.0,
    RES_FIBRE: 1.5,
}
#: Floors, for the first few colonists - a target of "3 stone" for a colony of
#: one is not a target. RES_HIDE is deliberately absent from both maps: no job
#: in score_actions produces it, so a target for it could never be acted on.
#:
#: RES_FOOD 60.0 is the one floor that is not a convenience, and it was measured
#: the hard way. At 16.0 a colony of three with 58 food read as fully glutted -
#: 2.4x its own target - and spent its entire opening on the quarry: seed 42
#: reached fatigue 1.000 and colony morale 0.004 by sim-minute five, then lost
#: seven people to cold. A young colony's larder is not a surplus, it is the
#: whole of its capital, and the store it needs is sized by the colony it is
#: about to become rather than by the three mouths it has now. 60 is roughly
#: what MAX_POP 20's own target would be at four food a head, so the feature
#: simply does not engage until a settlement is genuinely established.
STOCK_FLOOR = {
    RES_FOOD: 60.0,
    RES_COOKED: 4.0,
    RES_WOOD: 12.0,
    RES_STONE: 6.0,
    RES_FIBRE: 6.0,
}
#: How many multiples of the target count as a full glut. At 1.5 a store is
#: "glutted" at 2.5x its target, and the appetite it drives dies there.
#:
#: This is the width of the hysteresis band, and widening it is the fix if the
#: colony oscillates between farming and quarrying at the COLONY level - a much
#: slower and harder-to-see thrash than the per-agent kind MIN_COMMIT_SEC
#: handles. Narrowing it to save food reintroduces exactly that.
STOCK_GLUT_SPAN = 1.5
#: The food glut, slewed. Rise slow, fall fast, and the asymmetry is the whole
#: safety story: it takes two minutes of plenty before the colony believes it is
#: rich, and twelve seconds of famine to put everyone back in the fields.
#:
#: A naked ratio was tried in design and rejected for the reason
#: ``_tick_hut_tier`` gives about morale: at 2-10 agents the store crosses any
#: line constantly, and a naked ratio would flip the colony's whole labour
#: allocation on one birth or one cooked batch. Unlike the hut tier this is a
#: SLEW and not a LATCH, and that difference is deliberate - a famine does not
#: un-teach masonry, but a famine absolutely must put people back on the crops.
SURPLUS_RISE_SEC = 120.0
SURPLUS_FALL_SEC = 12.0
#: How often the slew advances, in sim-seconds. score_actions runs per agent per
#: AI tick, so the slew is stepped on the first call in each window and every
#: other agent that window reads the same value - which is what keeps it
#: deterministic and independent of how many people are alive.
SURPLUS_STEP_SEC = 1.0
#: The surplus is scaled to this after dark. Not a safety bolt-on: the extra
#: work a surplus buys is OUTDOOR work, up to 720 px from camp, and
#: ``animals._spawn_chance`` goes 0.25 -> 0.70 at night with bears weighted 1.6x.
#: Folding the night factor into the slew's TARGET rather than its output means
#: dusk ramps down over SURPLUS_FALL_SEC and dawn ramps back up over
#: SURPLUS_RISE_SEC, so there is no step at either edge for the action system to
#: oscillate across. The colony drifts back toward the fields and the fire after
#: dark, which are the jobs that happen inside the settlement.
SURPLUS_NIGHT = 0.35
#: A colony only spends its dividend when it is content. Morale here is
#: ``World.colony_morale()`` - the same mean the birth gate and the stone-hut
#: tier read, so "how is the colony doing" cannot come to mean three slightly
#: different numbers.
#:
#: This gate is load-bearing and not garnish. Un-gated, the extra work raises
#: fatigue, fatigue drops colony morale under MORALE_TO_GROW, and the BIRTH gate
#: closes - measured in design as morale blocking 27.8% -> 59.7% of sampled
#: ticks and a population going 11 -> 3. It is negative feedback: work makes
#: them tired, tired makes them rest.
#:
#: The floor is what stops it from being an off switch. Pooled mean morale runs
#: ~0.50, so a hard ramp from MORALE_TO_GROW would leave a typical colony
#: spending under a third of its dividend and the freed labour would land on
#: Wander instead of the quarry - which is not what was asked for.
SURPLUS_MORALE_MIN = MORALE_TO_GROW
SURPLUS_MORALE_FULL = 0.62
SURPLUS_THRIFT_FLOOR = 0.35
#: ...and the same idea per PERSON rather than per colony. `thrift` is a mean
#: over the roster and it lags: an individual can be at fatigue 1.0 while the
#: colony still reads content, and the dividend would send him to the quarry
#: anyway. That is not a hypothetical - it is what the first measured arm did.
#:
#: Daytime Sleep scores ``fatigue^2 * 0.55``, so a man at fatigue 0.8 rates rest
#: at 0.352; an undamped dividend put GatherWood at 0.48 and beat it, and the
#: colony worked through its own exhaustion until cold and mauling took it. At
#: 0.80 the same man's material appetite is cut to 0.36 of full and Sleep wins,
#: while a rested one at fatigue 0.2 keeps 84% of it.
#:
#: This is the term that lets the dividend stay generous. Turning the whole
#: feature down would have bought the same safety by giving up the effect the
#: user actually asked to see; this gives it up only for the people who cannot
#: afford it, at the moment they cannot afford it.
SURPLUS_FATIGUE_DAMP = 0.80
#: Appetite a full surplus buys for a resource sitting exactly at its target.
#: This is the number that decides whether the colony visibly branches out or
#: merely nudges: at 0.75 a gatherer's GatherWood goes 0.100 -> 0.483 at a full
#: larder, which puts it above BuildStructure's measured mean of 0.376 and far
#: above the damped Farm.
SURPLUS_DIVIDEND = 0.75
#: ...and this much of it survives even when the store is already at target, so
#: the dividend tapers with how short the resource actually is rather than
#: paying flat. Two jobs to do: it keeps a topped-up resource from holding the
#: same appetite as an empty one (negative feedback - as wood fills, stone
#: overtakes it, so the three gathers separate instead of landing on one number
#: inside TIEBREAK of each other), and the residual is what keeps a quarry
#: ticking over in a rich colony at all.
SURPLUS_RESIDUAL = 0.55
#: The standing per-head target also pulls WITHOUT a surplus, at this strength.
#: Mine already did exactly this at full strength (its old ``0.70 * stone_low``)
#: and it is why mining was the only material job leaking any stone into the
#: colony at all. Extending it to wood and fibre at 0.45 generalises the one
#: idiom in the file that was working. It is what stops the wood store hitting
#: literal zero, which it did at three of eight 5-minute marks on the measured
#: baseline.
MATERIAL_BASELINE = 0.45
#: Ceiling on any appetite. Exactly ``_gather_urgency``'s own maximum
#: (0.42 + 0.48), so the reachable maxima of GatherWood (0.91) and GatherStone
#: (0.905) are unchanged and neither can newly cross OVERRIDE_FLOOR.
APPETITE_MAX = 0.90
#: How hard a full surplus damps the food jobs. Farm keeps 40% of its appetite
#: at full glut rather than 0: the point is that a rich colony farms LESS, not
#: that it stops - a damp of 1.0 would abolish farming, take the score to 0, and
#: (because choose_action drops zero-scored candidates before drawing a
#: tiebreak) re-phase the whole RNG stream as a side effect.
FARM_SURPLUS_DAMP = 0.60
FORAGE_SURPLUS_DAMP = 0.85
#: Berry bushes are the colony's ONLY source of fibre (``actions`` gives them
#: ``bonus_res: RES_FIBRE`` at half yield), and every hut costs 6 of it. Damping
#: the food half of ForageBerries without this arm would quietly strangle
#: cordage. A rich colony strips bushes FOR the fibre and takes the berries as a
#: by-product, which is the honest reading of what it is doing.
FIBRE_VIA_BERRIES = 0.70
#: What a surplus adds to BuildStructure, scaled by how much of the site's cost
#: is already on the ground. Scaled by availability on purpose: this is "the
#: stuff is lying there, go and put it up", never "walk to a site with nothing
#: at it".
BUILD_SURPLUS_BONUS = 0.12

# ------------------------------------------------------------- cooking ------
#: Cooking is the one job in this economy that is a CONVERSION rather than a
#: production, and the scoring never said so. ``_h_cook`` takes 3 raw off the
#: pile and puts 3 cooked back (``ag.carrying = RES_COOKED``, same qty), and
#: ``_stock_state`` counts raw and cooked together as RES_FOOD - so cooking can
#: never lower the larder by a single unit. What it buys is strictly better
#: food: eating cooked is ``hunger -0.78 / morale +0.09`` against raw's
#: ``-0.52 / +0.04`` (see ``actions._h_eat``).
#:
#: Despite that, CookFood was scored purely on how short the PANTRY was
#: (``_appetite(RES_COOKED, ...)``), and the raw pile it draws from entered only
#: as a binary ``raw >= 3`` gate. Measured over 5 seeds x 15 sim-min, 827 scored
#: samples: the colony sat on a mean 105.2 raw food against a mean 2.7 cooked,
#: and CookFood was the top-scored candidate in 7 of 807 live samples - 0.87%.
#: It was not losing to one rival, it was losing to all of them: BuildStructure
#: outscored it in 75.5% of samples, Farm 56.0%, GatherWood 49.4%,
#: ForageBerries 45.8%, Mine 45.2%. A mean of 0.280 against a board where five
#: other jobs sit between 0.28 and 0.44.
#:
#: So the lever is not "beat BuildStructure". It is that a big raw pile plus an
#: idle fire should be a reason to cook in its own right, which is what this
#: bonus is. It is added on TOP of the pantry appetite, not folded into it, so
#: the two demands stay separate: "we have no meals ready" and "we are sitting
#: on a mountain of raw food".
#:
#: 0.24 was chosen against measured samples: it lifts CookFood's mean score from
#: 0.280 to 0.509 and makes it top the board in 21.3% of live samples against
#: 0.87%.
#:
#: On its own that was NOT enough, and the honest reading of this constant is
#: that it is the smaller half of the fix. With ``STOCK_PER_HEAD[RES_COOKED]``
#: still at its old 1.5 this bonus moved CookFood only 1.87% -> 3.42% of
#: colonist-time, because the binding constraint was never the score - it was
#: that the colony only wanted 1.5 meals a head and stopped. Raising the target
#: is what actually moved cooking; this term is what makes the colony reach the
#: target off the back of its raw pile rather than only when the pantry is bare.
#: Both were measured separately before either was kept.
COOK_PILE_BONUS = 0.24
#: Raw food per head, past the gate, that counts as a full pile. At pop 8 that
#: is 48 raw above the gate for the bonus to reach full strength.
COOK_PILE_SPAN = 6.0
#: The floor of the raw gate - today's flat ``raw >= 3``, which is exactly one
#: cook-load (``stock_take(w, RES_FOOD, 3)``).
COOK_RAW_GATE = 3.0
#: ...and this much raw per head ON TOP of it, scaled by how short of food the
#: colony actually is. This is the anti-starvation guard and it is the reason
#: the gate is no longer a flat 3: a cook takes his 3 raw OUT of the store and
#: holds them for COOK_TIME, so with the old gate a colony of eight with 3 food
#: left and everyone hungry could put its entire larder in one man's arms and
#: leave the store empty. At full shortfall and pop 8 the gate is now 35 raw;
#: at a comfortable larder (food_short 0) it is 3, i.e. unchanged. The gate can
#: therefore only ever be stricter than it was, never looser.
COOK_RAW_RESERVE = 4.0

# ------------------------------------------------- cracking loose rock ------
#: GatherStone is a colonist breaking up a loose rock where it lies; Mine is a
#: sustained dig at a quarry face. They are different things to look at and both
#: are supposed to be visible. They were not: over the same 827 samples
#: GatherStone topped the board in 13 (1.57%) at a mean score of 0.160, against
#: Mine's 0.283.
#:
#: The cause is structural rather than a matter of tuning, and all of it sits in
#: the two expressions. Mine reads
#: ``0.30 + 0.70 * max(stone_low, stone_want) + 0.20 if no loose rock``;
#: GatherStone read ``0.05 + 0.95 * stone_want`` - the lowest base on the board,
#: no scarcity term, and crucially ``stone_want`` alone rather than
#: ``max(stone_low, stone_want)``. Since ``_appetite`` only carries the standing
#: target through at ``MATERIAL_BASELINE`` (0.45) strength, Mine sees the full
#: per-head shortfall and GatherStone saw 45% of it. Both then read the SAME
#: ``stone_want``, so the surplus dividend that lifted stone demand lifted Mine
#: far more than GatherStone - which is exactly what the round measured
#: (Mine 1.55% -> 3.75% of colonist-time, GatherStone 2.11% -> 2.28%).
#:
#: These numbers raise GatherStone's floor and add a proximity term in place of
#: Mine's scarcity one. Reachable maximum is
#: ``0.20 + 0.62 * APPETITE_MAX + 0.12 = 0.878``, which is BELOW the old
#: expression's 0.905 - the ceiling goes down, not up, and nothing here can
#: newly cross OVERRIDE_FLOOR (0.95).
#:
#: Measured over 5 seeds x 30 sim-min against an exact baseline: GatherStone
#: 2.65% -> 3.59% of colonist-time (4.16% -> 6.39% in the rich-larder segment)
#: with Mine held at 2.89% -> 2.81%, which is the point - the quarry keeps its
#: work. A stronger variant (floor 0.28, span 0.52) was measured and rejected:
#: it bought almost nothing on GatherStone (3.71%), pulled Mine down to 1.78%,
#: raised thrash from 3.46 to 3.91 switches per colonist-minute, took mean peak
#: from 0.682 to 0.710, and produced the run's only hunger death.
GATHER_STONE_FLOOR = 0.20
GATHER_STONE_SPAN = 0.62
#: How much of the standing per-head stone shortfall the LOOSE-ROCK job sees,
#: as a fraction of what the quarry sees. Mine reads `max(stone_low, ...)` at
#: full strength; this is the same arm for GatherStone, weighted.
#:
#: It exists because setting it to 1.0 was measured and was wrong. At parity the
#: two expressions come out nearly identical for the same colonist
#: (0.308 + 0.62x against Mine's 0.30 + 0.70x), and since far more of the roster
#: carries a `gather` affinity than a `mine` one, the loose rock simply took the
#: quarry's work: over 5 seeds x 30 sim-min GatherStone went to 4.98% of
#: colonist-time while Mine fell to 0.80% against a 3.75% baseline. Total stone
#: labour was conserved - it just all moved to one of the two activities, which
#: is the same failure as the one being fixed, pointing the other way. The user
#: named cracking rocks AND the quarry; both have to stay on screen.
#:
#: At 0.0 the arm is off and GatherStone is driven by `stone_want` alone, as it
#: always was - the lift then comes only from the floor and the proximity bonus,
#: which are terms Mine does not share and so cannot be taken from it.
GATHER_STONE_LOW_ARM = 0.0
#: What a rock lying at your feet is worth over one at the edge of range. This
#: is the mirror image of Mine's ``scarce_bonus`` (which pays +0.20 when there
#: is NO loose rock to crack) and it is also the walk guard: ``find_prop``
#: already refuses anything past 720 px, but a flat score would make a rock at
#: 700 px as attractive as one at 20 px, and fall deaths have already drifted
#: 2 -> 5 on a plausible walk-distance mechanism. Measured rock distances are
#: short - median 52 px, p90 220 px - so a 400 px reference is a real gradient
#: over the range colonists actually see rather than a rounding error: it pays
#: 0.87 of full at the median, 0.45 at p90, and nothing at all past 400 px.
GATHER_STONE_NEAR_BONUS = 0.12
GATHER_STONE_NEAR_REF = 400.0

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
#: The cap, and it was sized to the OLD roster and nothing else.
#: ``world.regrowth_factor`` is ``min(REGROW_MAX, 1 + (n - MIN_POP) * 0.22)``,
#: which saturates at n = 2 + 3.0/0.22 = 11.09 - i.e. from the eleventh colonist
#: upward the wild recovers no faster however many mouths there are. At the old
#: MAX_POP of 10 that cap was never reached and the curve was linear over the
#: whole range; at MAX_POP 20 the top half of the roster is flat while the
#: foraging load still rises 82%, which is exactly the starvation spiral the
#: comment above says this exists to prevent.
#: 5.0 = 1 + 0.22*18, i.e. the same straight line extended to the new cap rather
#: than a new number. The 4x map hands the colony 4x the standing stock and
#: partially masks the shortfall, which is why this would otherwise be missed.
#: NOTE: props.py:77 carries a duplicate fallback ``_REGROW_MAX = 3.0`` used only
#: if this import fails; it wants the same value.
REGROW_MAX = 5.0

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

#: Where a quarry is allowed to be. Mining is not a neutral act: every yield tick
#: calls Terrain.deform through _dig_divot and lowers the ground it happens on,
#: and the site used to be simply the NEAREST stone column to whoever decided to
#: dig - which is to say, under the settlement, because that is where everybody
#: stands. A colony therefore spent the evening excavating its own floor, and
#: this is the same mechanism that once cut agents off from ground they needed
#: (see the crossing director, which exists to build a way back).
#:
#: So a quarry is sited AWAY from home and TOWARD an edge. The keep-out is
#: measured from the colony centre; the edge preference reuses the band the
#: barricades already think in, so "the edge of the map" means one thing in this
#: codebase rather than two.
MINE_KEEP_OUT = 320.0       # px from colony_center() a quarry may not open
#: How far a miner will walk to reach a legal site. Generous, because the whole
#: point is to send them out of town, and bounded, because a colony whose stone
#: is all on the far side of the map should still mine rather than march.
MINE_MAX_WALK = 900.0
#: Relative pull of "near an edge" against "close to where I am standing" when
#: ranking legal columns. Edge dominates - that is the request - but the walk
#: term is not zero, or two equally good edges would send miners across the whole
#: map on a coin flip and the colony would watch its quarrymen commute.
MINE_EDGE_WEIGHT = 1.0
MINE_WALK_WEIGHT = 0.35

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
#: goes. Sized as "a visible bite out of a resource the player has watched
#: accumulate all run" - it costs no lives and no buildings, which is the whole
#: point of that kind. That bite was 4 of MAX_GRAVES 10 = 40%; MAX_GRAVES is now
#: 20 (twice the roster means twice the funerals), which would have quietly
#: halved the event to a 20% nibble. 8 keeps the fraction, and the SEC stays 12
#: so the visit is longer rather than faster - the scouring is meant to be
#: something you watch happen and can interrupt, not a burst.
DRAGON_SCOUR_SEC = 12.0
DRAGON_SCOUR_MAX = 8

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

# ------------------------------------------------------------------ relics --
# What a dead dragon leaves behind. One slot per colonist (``Stickman.relic``),
# so nobody is ever both impenetrable and holding the gun.
#
# THE RARITY IS THE DRAGON, NOT A DICE THROW ON TOP OF IT. Every non-flyover
# dragon death drops exactly one relic and there is no second roll. Measured, 16
# seeds x 90 sim-min of an unmodified World: 19 visits and 6 kills over 24
# colony-hours, i.e. 0.250 kills per RAW colony-hour and 1.14 per GATED one (the
# stone-hut gate latched in 12 of 16 seeds at a median of ~58 min, and the four
# that never latched could never see a relic at all). A 30% drop roll on top of
# that would be one item per 13 colony-hours, which is dead content by the
# brief's own standard; 6 of 16 seeds saw a relic inside 90 minutes from cold,
# which already satisfies "not everybody has one".
#
# THE RARITY IS STILL THE DRAGON. WHICH RELIC IS NO LONGER THE DRAGON'S NAME.
# The first table paired one relic to one dragon kind, which multiplied the
# item's rarity by its dragon's: the bfg9000 hung off the serpent, the rarest
# kind and the least-killed of the five, and it dropped ZERO times in one 32
# colony-hour sample and 0.073 times per colony-hour in a 96 colony-hour one -
# reaching 6 colonies out of 48. Ironshod boots, off the skeletal's coin flip,
# were rarer still at 0.031. Both are items the design leans on: the gun is the
# headline, and the boots are the only answer in the game to falling, which
# kills more colonists than anything else. Over those same 96 colony-hours the
# boots caught exactly ZERO lethal landings.
#
# ``sim/items.DROP_WEIGHTS`` is therefore weighted rather than paired (every
# kind can leave any relic, its signature one likeliest) and carries a pity rule
# for the first gun and the first pair of boots a colony ever earns. A/B on the
# same 48 seeds: bfg9000 0.073 -> 0.125 per colony-hour (6 -> 12 colonies of
# 48), ironshod 0.031 -> 0.073 (3 -> 7), lethal landings caught 0 -> 4. The
# number of relics per dead dragon is unchanged at exactly one, so nothing above
# moves: the total drop RATE is the same 0.4-0.5 (0.448 -> 0.406, inside the
# noise) and only the mix is different.
#
# The drop table itself lives in ``sim/items.py`` beside the code that reads it,
# following the precedent of ``throwing.py``'s tuning block: numbers only one
# mechanic reads live with the mechanic, and it keeps this file single-owner so
# two workstreams cannot collide in it.
RELIC_NONE   = ""
RELIC_SCALE  = "dragonscale"
RELIC_AMULET = "amulet"
RELIC_BFG    = "bfg9000"
RELIC_BOOTS  = "ironshod"
RELIC_CAIRN  = "cairnstone"
RELIC_KINDS: tuple[str, ...] = (RELIC_SCALE, RELIC_AMULET, RELIC_BFG,
                                RELIC_BOOTS, RELIC_CAIRN)

#: Dragonscale, mirrored into ``Stickman.armour`` so every existing reader - the
#: damage path, ``combat_actions._armoured``, ``render.draw_gear`` - keeps
#: working with no edit at all.
#:
#: "1.0" does NOT mean invulnerable, and that is deliberate rather than a
#: rounding accident. ``Stickman.hurt`` reads
#: ``taken = amount * (1.0 - _clamp(self.armour, 0.0, 0.9))``, so the existing
#: 0.9 clamp turns this into 90% absorbed: impenetrable is EFFECTIVE, not
#: literal. That clamp is the whole design - nothing has to be special-cased, a
#: hand-edited save claiming ``armour: 9.9`` still lands on 0.9, and no agent can
#: ever be immortal.
#:
#: MEASURED, and this is the honest ceiling on the item: over 16 seeds x 90
#: sim-min, 293 deaths, only TWO causes route through ``Stickman.hurt`` at all -
#: mauled (73, 24.9%) and devoured (10, 3.4%). The other 200 (lightning 54, fire
#: 46, fall 27, drown 22, cold 20, meteor 18, mudslide 11, hunger 2) go through
#: ``events._kill -> Population.kill -> Stickman.die`` and NEVER TOUCH ARMOUR.
#: The strongest armour conceivable addresses about a quarter of what kills this
#: colony. The dragon's own maw is already immune: ``_devour`` deals
#: ``MAX_HEALTH * 10.0`` = 1000, of which 10% = exactly MAX_HEALTH, so a
#: dragonscale wearer is still eaten. That 10x predates this item and anticipates
#: it exactly - leave it alone.
ARMOUR_DRAGONSCALE = 1.0

#: The wyrm-gun. Held in ``Stickman.weapon`` instead of a spear, which is meant
#: to be the entire cost of carrying it.
#:
#: WHAT THIS NOTE USED TO CLAIM, AND WHY IT WAS WRONG. It said the cost "costs
#: no code to enforce: ``throwing.can_throw`` tests ``weapon != WEAPON_SPEAR``
#: exactly and ``combat_actions._armed`` tests the same, so a BFG carrier
#: automatically cannot throw and cannot melee." Both halves of that are true
#: and the conclusion still does not follow, because it only looks at the READS
#: and never at the WRITES. The slot is not a lock. Three live paths assign to
#: it (``combat_actions._craft_apply`` and ``_h_retrieve_spear``, both to
#: WEAPON_SPEAR, and ``throwing`` to WEAPON_NONE on release), and the first is
#: on the daily round of every idle adult:
#:
#:   ``score_craft`` gates CraftSpear on ``not armed``, where ``armed`` is
#:   ``_armed(agent)`` - deliberately spear-only. A BFG carrier therefore reads
#:   as UNARMED to the crafting goal, scores CraftSpear at 0.55-0.87, and
#:   ``_craft_apply`` then does ``agent.weapon = WEAPON_SPEAR`` with no check on
#:   what was already in the slot. The wyrm-gun is overwritten - not dropped,
#:   not chronicled, gone. ``Stickman.relic`` still reads ``bfg9000``, so the
#:   colonist is still "carrying" it: ``_bfg()`` is False forever after, the gun
#:   can never fire again, and because the one relic slot is still occupied that
#:   colonist can never pick up another relic either.
#:
#:   combat_actions already owns the predicate that fixes this. It wrote
#:   ``_combat_capable`` (``_armed or _bfg``) for exactly this class of mistake
#:   and then used it in ONE place, the flee gate. The craft gate is the second
#:   place it belongs.
#:
#: MEASURED, real World, a carrier handed the gun at t=0 on level ground with a
#: hostile always in range: the slot was overwritten by CraftSpear at t=53s
#: (seed 62) and t=31s (seed 63); the third carrier died at t=7s still holding
#: it. From there it cycles CraftSpear -> ThrowSpear -> RetrieveSpear forever.
#: Over 3 seeds x 90s with a target permanently available, ``score_bfg``
#: returned 0.0 on 2700 of 2700 ticks and the dominant reason - 1736 to 2001 of
#: them, 64-74% - was "not holding the gun (weapon='')". Over 4 seeds x 30
#: sim-min of GUARANTEED carry (re-armed the instant the colony had no BFG):
#: 0 shots, 0 misfires, 0.00 shots/colony-hour.
#:
#: THE FIX IS NOT IN THIS FILE and was deliberately not attempted from here.
#: Retuning a number cannot stop an assignment; CraftSpear must decline for a
#: relic-weapon holder (or route through ``items``), and combat_actions.py
#: belongs to another workstream. Recorded here because this is where the false
#: "costs no code to enforce" reasoning was written down, and the next hand will
#: read it before they read the handler.
#:
#: Read BFG_MISFIRE's carry-life arithmetic with this in mind: it is derived per
#: SHOT, and a gun that is overwritten before it fires has no misfire rate at
#: all.
WEAPON_BFG = "bfg9000"
BFG_RANGE      = 420.0
#: == throwing.THROW_MIN_RANGE. Inside this you are shooting your own boots.
BFG_MIN_RANGE  = 34.0
#: Perpendicular half-width of the beam. Everything whose ``throwing._target_box``
#: overlaps the corridor goes down; TERRAIN stops the beam and a body does not,
#: which is why "wipes out animals" and "kills everything in line of sight" are
#: one mechanic and not two.
BFG_CORRIDOR   = 22.0
#: How big a misfire LOOKS. **No longer a damage radius** - read this before
#: retuning it, because the name is now a half-truth kept on purpose.
#:
#: THE SPEC WAS CORRECTED, in the author's words: '"randomly explodes taking out
#: the stickmen" .. "stickman", just himself .. so its an or situation .. could
#: possibly leave a single stickman'. The gun has two failure modes and they are
#: an OR, not one blast that scales up:
#:
#:   MISFIRE  -> the gun comes apart and kills THE WIELDER ALONE. Not his
#:               neighbours, not the man who walked over to talk to him.
#:   HEEDLESS -> the shot goes down the corridor and kills every stickman on the
#:               line as well as the animals. THAT is the one that can wipe most
#:               of a colony and leave a single survivor, and it stays fully
#:               lethal - see BFG_HEEDLESS and BFG_CORRIDOR, both untouched.
#:
#: At 70 px of lethal blast (roughly a hut's width, and a measured mean of 1.915
#: colonists inside it) the misfire reliably caught the neighbours, which made
#: the two outcomes the same outcome at a different radius and left the *failure*
#: of the weapon doing the mass casualties instead of the shot the wielder chose
#: to take. ``combat_actions._bfg_misfire`` now hurts ``agent`` and nobody else,
#: so no damage path reads this value at all any more.
#:
#: STILL 70.0, AND DELIBERATELY NOT ZEROED. It was briefly set to 0.0 from this
#: file as a way of enforcing the new spec through the old distance filter, which
#: was the wrong lever twice over: combat_actions had already removed that filter
#: (so it bought nothing), and ``render/items.draw_misfire`` defaults its ring
#: radius to this constant, so zero would have silently shrunk the explosion to a
#: 2 px dot. The flash is the whole reason a viewer knows what killed the
#: colonist. The light says "70 px of that went up", and it did - it just no
#: longer kills the man standing in it. Do not add a second constant for the
#: visual either; one number, one meaning, no drift.
BFG_BLAST_R    = 70.0
#: DELIBERATELY 9x AND NOT ``_devour``'s 10x, and the two must be checked
#: together by anyone tuning either. Through the armour clamp: unarmoured takes
#: 900 and dies, ARMOUR_LEATHER takes 495 and dies, ARMOUR_DRAGONSCALE takes 90
#: against 100 health and SURVIVES AT 10 HP. One constant buys the interaction
#: the scale exists for - the impenetrable armour walks out of a BFG beam - and
#: costs nothing anywhere else. At 10x the scale takes exactly 100, dies, and
#: that interaction silently disappears.
BFG_DAMAGE     = MAX_HEALTH * 9.0
BFG_COOLDOWN   = 9.0
BFG_WINDUP     = 1.1
#: Chance a shot comes apart instead of firing: no ray, no blast, the WIELDER
#: takes BFG_DAMAGE and NOBODY ELSE DOES (see BFG_BLAST_R above for the spec
#: correction), and the relic is DESTROYED rather than dropped. The only item in
#: the set that can leave the world, and that asymmetry is its balance valve -
#: the mean carried life of a BFG is 1/0.256 = 3.9 colony-hours.
#:
#: THAT 3.9 HOURS IS AN ESTIMATE PER SHOT, AND IT DOES NOT HOLD IN THE BUILD AS
#: IT STANDS. It is 1/(BFG_MISFIRE x shots-per-hour) and assumed 3.2 shots per
#: colony-hour; the measured rate is 0.00 over 4 seeds x 30 sim-min of
#: guaranteed carry, and 2 shots over 16 seeds x 20 sim-min of ordinary play.
#: The gun is not surviving longer because it is safe - it is being overwritten
#: in the weapon slot by CraftSpear long before it ever fires. See WEAPON_BFG
#: above for the measurement and for why the fix is not in this file. Do not
#: re-tune this constant against the shipped rate: tuning a misfire chance
#: against a weapon that never pulls the trigger is fitting noise. Re-measure
#: once the slot is defended, then decide.
#:
#: Deliberately NOT re-tuned upward now that the misfire is wielder-only. It is
#: tempting - the misfire kills ~1.9 fewer people than it used to, so 0.08
#: "buys less" than it did. But the 3.9-colony-hour carry life is the number the
#: whole item is balanced around (it is what stops one lucky drop from arming a
#: colony forever) and that depends on the misfire RATE, not on the body count.
#: Raise this and the gun stops existing. The other way to make a misfire hurt
#: more is not available: hurting the bystanders is precisely what the corrected
#: spec rules out.
BFG_MISFIRE    = 0.08
#: Chance the wielder fires anyway with a friendly standing in the corridor.
#:
#: NOT squeamishness - this is what makes the item shippable. Measured geometry,
#: 8 seeds x 60 sim-min, 5083 shot opportunities: mean 1.129 friendlies in the
#: corridor, P(line clear) only 0.445, so MORE THAN HALF of all shots have a
#: colonist in the beam. At heedless = 1.0 the wielder kills 6.12 of its own
#: colony per hour against a measured baseline of 11.75 deaths/hour, which ends
#: every colony inside sixty minutes. At 0.20 it is 1.64/hour (+14%), one
#: single-shot colony wipe per 59.7 colony-hours of carry, so P(a given BFG ends
#: the colony that found it) ~= 6.5% - roughly one in fifteen.
#:
#: Both numbers come out of a SHADOW Monte Carlo (500 trials over the measured
#: geometry, calibrated to the measured engagement rate of 262 throws / 6210
#: opportunities = 4.22%). It does not feed deaths back into the world, so it
#: biases single-shot wipes DOWN and per-shot friendly deaths UP, and it does not
#: model collapse by attrition at all. Re-measure both in a live A/B over >= 16
#: seeds before trusting them.
BFG_HEEDLESS   = 0.20
#: Cause string for a BFG kill on a colonist. NOW REGISTERED in
#: ``names.DEATH_KINDS`` -> ``died_disintegrated``, with templates, which is what
#: the previous note here asked the next hand to do ("add the template first,
#: then the mapping" - both landed together).
#:
#: What it read like before: an unregistered cause falls through to the generic
#: ``died`` line, so the chronicle said "X was lost to disintegrated." - true,
#: but the one death in the game that should read like an event read like a
#: database field. The half-fix to avoid is registering the mapping WITHOUT the
#: template, because a kind with no template renders as "Something happened."
CAUSE_DISINTEGRATED = "disintegrated"

#: Seconds a relic lies unclaimed before it rots away, so an unreachable drop at
#: the far edge of the map does not litter the world forever. Same idea as
#: ``throwing.SPEAR_LIFE_SEC`` (600) but three times longer, because a relic
#: arrives once per 4 raw colony-hours and a spear arrives every fight.
RELIC_DECAY_SEC   = 1800.0
#: How far a colonist will walk to fetch one. Deliberately wider than
#: ``throwing.THROW_MAX_RANGE``: the colony clusters near its stockpile, and a
#: radius that only covered the camp would mean most drops were never claimed.
RELIC_FETCH_RANGE = 520.0
#: == throwing.RETRIEVE_RADIUS. Close enough to stoop and pick it up.
RELIC_PICKUP_R    = 18.0
#: Cap on relics lying on the ground at once, in the style of ANIMAL_MAX_ALIVE
#: and MAX_SPEARS. The oldest is recycled rather than the drop being refused - a
#: cap that swallows loot silently is worse than one that forgets old loot.
MAX_RELICS        = 6

#: What ironshod boots cost you instead of your life. Still routed through
#: ``Stickman.hurt``, so it goes through armour (a scale-wearer in boots takes 3)
#: and a boot-wearer already at low health can still die of the fall - the item
#: removes the instant-death gate, it does not remove falling.
#:
#: MEASURED: falls are the 4th cause of death at 27 of 293 (9.2%), 1.7 per
#: colony-90min. One wearer out of a mean 7.04 alive prevents ~0.24 fall deaths
#: per colony-90min - small, safe, and exactly right for a common drop.
#:
#: "THE MOST COMMON DROP IN THE TABLE", WHICH IS WHAT THIS USED TO SAY, WAS
#: NEVER TRUE. Under the paired table the boots came off the skeletal on a coin
#: flip and measured 2 drops in 32 colony-hours - tied for the rarest thing in
#: the game bar the bfg9000, which managed none at all. They are now the second
#: heaviest weight in every row of ``sim/items.DROP_WEIGHTS`` and the second
#: item its pity rule speaks for. Every row now weights them first or second
#: (third only on the serpent, whose row belongs to the gun), which is what the
#: sentence above always assumed and never had.
#:
#: RE-MEASURED, and 9.2% is the LOW end of the range rather than the number.
#: Two fresh passes on the current build, causes latched at the tick of death
#: (an end-of-run scan of the roster reads near zero, because ``_reap_dead``
#: retires buried corpses - that mistake produced an empty histogram first
#: time):
#:
#:   * 6 seeds x 30 sim-min, all SCENE_NIGHT_STORM: 33 deaths, 11.0/colony-hour,
#:     falls 6 (18.2%) behind lightning 10 and level with mauled 8 / mudslide 8.
#:   * one seed per scene x 12 sim-min across all of SCENES: 25 deaths, falls 4
#:     (16.0%), 2nd behind mauled 13.
#:
#: So falls are 16-18% and 2nd-3rd, not 9.2% and 4th, and the item is worth
#: correspondingly more than the paragraph above says. The brief that prompted
#: this work quoted a third figure again - 191 of 565, 33.8%, the LARGEST cause
#: - which neither pass here reproduced; scene mix moves this number hard (0 of
#: 4 in flood, 1 of 1 in ashfall / earthquake / volcano) and run length moves it
#: too, so a fall share quoted without both is not comparable. Nothing was
#: retuned on the strength of any of them: 30.0 is defensible across the whole
#: 9-34% range, and re-tuning against a share that swings 4x with the weather
#: would be fitting the sample. What this note now protects against is the
#: opposite mistake - somebody reading "9.2%, 4th cause" and deciding the boots
#: are not worth their slot.
FALL_SURVIVED_DAMAGE = 30.0
