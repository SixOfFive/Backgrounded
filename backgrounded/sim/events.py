"""EventSystem: weather scenes, disasters and the things they do to the world.

Pure data + numpy. **No pygame** — this module must stay importable headless.

The system owns the current scene, its intensity envelope, the weather
channels (``wind``/``rain``/``snow``/``ash``), scheduled one-shot events and
the transient geometry the renderer draws (``strikes``, ``meteors``,
``water_level``, ``shake_offset()``).

``tick(world, dt)`` is the only entry point that mutates anything. It applies
scene consequences to the world: terrain deformation and repainting, prop
ignition, agent panic and death, and flashes pushed into ``world.lighting``.
Every phase is individually guarded — a broken subsystem disables itself for
that tick rather than taking the whole app down.

World coupling is deliberately duck-typed so the modules other agents own can
evolve. What is actually used, all optional:

    world.terrain      .height (np.float32[W])  .material (np.uint8[W])
                       .ground_y(x) .deform(x0,x1,dy) .paint(x0,x1,m)
                       .crater(cx, radius, depth)   <- column based, takes no y
    world.lighting     .add_flash(i, decay, color) ; .wind_gust attribute
    world.agents       iterable of Stickman (.x .y .alive .warmth .morale .id)
                       - the real World calls this ``world.population``
    world.props        iterable of Prop (.x .y .kind .burning .ignite())
    world.structures   iterable, same shape as props
    world.cfg/.config  Config (auto_scene_change, scene_min_sec)
    world.kill_agent(agent, cause)      preferred death path if present
    world.chronicle.add(text)           preferred history path if present

Sign convention used for terrain: ``height[x]`` is the *y* of the surface, so
**+dy lowers the ground (digs) and -dy raises it**.

Panic protocol (honoured by behavior.py if it wants to): events set
``agent.panic`` (seconds), ``agent.panic_target_x`` and ``agent.panic_reason``,
and count the timer back down again in ``_expire_panic`` - nothing else does,
and a stuck flag makes an agent reckless near drops for the rest of its life.
Evacuation proper goes through :meth:`EventSystem.hazards`, which behavior.py
reads (via ``actions.hazards_of``) to score FleeFrom.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any

import numpy as np

from ..constants import (
    MAT_ASH,
    MAT_DIRT,
    MAT_LAVA,
    MAT_MUD,
    MAT_SAND,
    MAT_SNOW,
    MAT_STONE,
    RENDER_H,
    SCENE_ASHFALL,
    SCENE_AURORA,
    SCENE_BLIZZARD,
    SCENE_CLEAR,
    SCENE_EARTHQUAKE,
    SCENE_ECLIPSE,
    SCENE_FLOOD,
    SCENE_FOG,
    SCENE_HEATWAVE,
    SCENE_METEOR,
    SCENE_MUDSLIDE,
    SCENE_NIGHT_STORM,
    SCENE_SANDSTORM,
    SCENE_VOLCANO,
    SCENE_WILDFIRE,
    SCENES,
    STAGE_HALF,
    WORLD_W,
)
# stage_bounds is the whole reason this module can keep its per-second rates on
# a map four times wider than the camera: see _stage() below. actions imports
# only constants/entities/structures, none of which import events, so this
# cannot close a cycle - and world.py already imports events, not the reverse.
from .actions import stage_bounds as _stage_bounds
from .lighting import LIGHTNING_COLOR, METEOR_COLOR, daylight_factor, is_night
# props and structures own what burns; see _FLAMMABLE below. Both are leaves -
# neither imports events (nor lighting), so this cannot close a cycle.
from .props import FLAMMABLE as _PROP_FLAMMABLE, KINDS as _PROP_KINDS
from .structures import STRUCTURE_SPECS

log = logging.getLogger(__name__)

__all__ = ["EventSystem"]

# ------------------------------------------------------------------ tuning --
STRIKE_TTL = 0.5                 # seconds a bolt stays drawable
STRIKE_MIN, STRIKE_MAX = 4.0, 12.0
#: Share of bolts that aim at somebody. A storm throws ~37 bolts per 300 s, and
#: an aimed one lands inside its own kill radius by construction, so this is
#: very nearly the number of funerals per storm: at 0.08 a four-person colony
#: was losing 4.3 people per 300 s to lightning alone. 0.02 gives ~0.9 aimed
#: bolts per storm. Measured over 20 seeds x 300 s: 0.08 -> 4.3 lightning
#: deaths per storm, 0.02 -> 1.25, 0.016 -> 0.85. The night-storm budget is the
#: tightest in the spec (0-2 deaths *including* falls), so this sits low: a
#: stickman is struck roughly every second storm, which is what "rare" means.
DIRECT_HIT_CHANCE = 0.016
#: A bolt kills everything in this column, and stickmen cluster round the fire,
#: so a wide radius quietly turns one strike into a double funeral (measured:
#: 2.1 deaths per aimed bolt at 20 px, 1.15 at 13 px). Keep the jitter under it
#: or "direct" strikes go wide and feature 29 stops firing at all.
DIRECT_KILL_RADIUS = 13.0
DIRECT_AIM_JITTER = 7.0
SCORCH_HALF_WIDTH = 14.0
#: px either side of a bolt that can catch. props.py runs its own fire spread,
#: so one ignition is not one burnt tree - it is a fire front. Every ignition
#: here is therefore paid for in wood the colony will not get to build with.
STRIKE_IGNITE_REACH = 45.0

METEOR_MIN, METEOR_MAX = 1.6, 5.0
#: At 0.20 roughly 18 rocks per 300 s actually landed. That is a crater every
#: 17 s and an ignition with each one, which stripped the map from 24 trees to
#: 2 and halved what got built - all for 0.7 deaths, because most of them
#: landed nowhere near anybody. Fewer, bigger, more dangerous impacts instead.
METEOR_IMPACT_CHANCE = 0.09
METEOR_TTL = 0.45                # afterglow once the streak lands
METEOR_BLAST = 1.5               # share of the crater radius that actually kills
METEOR_WARN_R = 130.0            # how far out an inbound rock reads as a hazard
METEOR_PANIC = 1.2               # panic seconds for the near-miss ring
#: multiples of the crater radius a rock can set alight. At 2.5 nearly every
#: impact started a fire and props.py's own spread did the rest - the map went
#: from 24 trees to 2 in a single meteor scene.
METEOR_IGNITE_REACH = 1.0

MUDSLIDE_WARN = 2.8
MUDSLIDE_SLIDE = 3.0
MUDSLIDE_SETTLE = 2.5
MUDSLIDE_REST = 38.0             # quiet spell before the slope fails again
#: Panic seconds granted per tick in the span. Re-applied every tick while the
#: ground is moving, so this is not "how long the fright lasts" - it is how
#: long an agent stays reckless *after* getting clear, which is the part that
#: kills. Short.
MUDSLIDE_PANIC = 0.15
#: Share of slides that come down where people are. Fewer slides (see REST)
#: means each one has to matter, or the scene rumbles all evening and buries
#: nobody: at 0.5 only 13% of slides caught anyone at all.
MUDSLIDE_AIM_CHANCE = 0.85
#: px the upper slope is scoured over one slide. Nine slides fit in a 300 s
#: scene and the scour is cumulative, so this is really "how much fresh cliff
#: the scene manufactures per run". At 26 px the map grew drops that killed
#: more people by falling than the mud ever buried (42 falls vs 24 burials).
MUDSLIDE_DROP = 9.0
MUDSLIDE_BURY_SEC = 1.1          # seconds inside the moving span before it kills
#: px of hillside that goes at once. Width is mostly a *panic* dial rather than
#: a burial one: the aimed slides centre on somebody either way, but a wide
#: span also panics the bystanders, and a panicking stickman walks off ledges a
#: calm one refuses. Wide spans made the mudslide scene kill more people by
#: falling than by burial.
MUDSLIDE_SPAN_MIN, MUDSLIDE_SPAN_MAX = 90.0, 170.0
#: px past the edge of the span a panicking stickman runs for. Measured to be
#: inert as a safety dial (60 and 170 give identical death counts) - what makes
#: a mudslide flight dangerous is the panic flag itself, not how far it aims.
MUDSLIDE_FLEE_X = 170.0

#: The surge envelope, and the most important build-rate dial in the file.
#: behavior.py makes anyone whose ground is under the waterline flee uphill, so
#: while the water is up most of the colony is running rather than working:
#: the old 96 s wet / 24 s dry cycle spent 36% of all agent-time on FleeFrom
#: and finished 2.3 structures per 300 s. 64 s wet / 95 s dry finishes 3.25 and
#: still drowns people, because a surge that is *shorter* is not gentler.
FLOOD_RISE = 28.0
FLOOD_HOLD = 12.0
FLOOD_FALL = 24.0
FLOOD_DRY = 95.0                 # dry spell before the next surge
FLOOD_DEPTH_FRAC = 0.45          # share of the terrain's relief the water covers
DROWN_SEC = 5.5
DROWN_DEPTH = 8.0                # px below the line before you are actually under
FLOOD_PANIC = 2.0                # panic seconds while under the water

SNOW_MAX_DEPTH = 3.2             # px the ground rises under a full snow layer
SNOW_RATE = 0.055                # px/s
ASH_MAX_DEPTH = 2.4
ASH_RATE = 0.030
SAND_MAX_DEPTH = 2.8
#: Slower than snow on purpose. Sand is the only accumulation the colony has to
#: watch *arrive*: a blizzard is over in ten minutes and the snow melts back in
#: the next clear scene, but nothing in the sim ever lifts sand again, so the
#: drift creeping across the map is the permanent mark this scene leaves. At
#: 0.032 px/s a full layer takes ~88 s of storm - a seventh of the slot - which
#: is long enough that you can see the desert winning.
SAND_RATE = 0.032

#: The colour the haze wash is drawn in for the fog scene. Duplicated from
#: ``fx._FOG_GREY`` rather than imported, because sim/ may never import render/:
#: the two must agree by hand, and a test asserts they do (a mismatch would
#: silently retint the fog scene, which is exactly what generalising the wash
#: was not allowed to do).
HAZE_GREY: tuple[int, int, int] = (198, 202, 208)
#: ...and the sandstorm's. Warm, dirty and light enough to wash a night scene up
#: to a brown murk rather than smearing the frame with mud.
HAZE_SAND: tuple[int, int, int] = (196, 158, 96)
#: ...and the eruption's: ash lit from underneath. Darker and far redder than
#: either of the others, because this haze is not being lit by the sky - the sky
#: has gone - it is being lit by what is on the ground.
HAZE_EMBER: tuple[int, int, int] = (150, 74, 54)

#: Peak haze density for the sandstorm. Below the fog scene's 0.85 deliberately:
#: fog is a *still* wash you look through, whereas this one has grit streaking
#: across it and a hard colour cast behind it, and stacking all three at fog's
#: strength buries the colony instead of obscuring it.
SAND_HAZE = 0.72
#: Wind amplitude and gust multiplier. Both are the highest in the file - the
#: night storm sits at 0.85/1.0 and the blizzard at 0.95/0.9 - because the gust
#: channel is the only part of this scene the *lighting* can see: it feeds
#: Lighting.wind_gust, which is what makes every torch in the colony gutter
#: together as a squall goes through.
SAND_WIND = 0.95
SAND_GUST = 1.30
#: Grit only bites while it is actually blowing hard, so the harm arrives in
#: squalls you can watch rather than as a constant tax.
SAND_STING_GUST = 0.55
SAND_STING_RATE = 0.55           # hp/s of scouring while a squall is up
#: ...and the health no amount of sandstorm will take anyone below. This is a
#: hard floor, not a tuning target: the scene is meant to be an atmosphere piece,
#: and the surrounding disasters have taught this file repeatedly that "usually
#: survivable" becomes "wiped the colony" on the one seed that stacks two
#: hazards. A scene that *cannot* reduce anyone below it needs no luck, and the
#: 1/6 hp/s regen puts the difference back within a couple of minutes of the
#: storm passing.
#:
#: The exact number is set by what *else* can hurt a stickman, not by how hard
#: the sand ought to feel. Wounds are one pool, so a floor is a standing
#: discount on everyone's margin against a wolf pack for the whole scene, and
#: that indirect cost is the only one this scene can actually levy. Measured
#: over 40 seeds x 600 s of sandstorm with nothing but this dial moved: deaths
#: by mauling were 22 with the scour off entirely, 27 at 75, and 37 at 62, and
#: final population 3.73 / 3.77 / 3.55. At 62 the storm was killing people it
#: never touched. At 75 the population is indistinguishable from not scouring at
#: all, and the cost is a quarter of a health bar you can watch in the HUD.
SAND_HEALTH_FLOOR = 75.0

#: --- heatwave / drought ---------------------------------------------------
#: Seconds for the drought to reach full depth. Deliberately an order of
#: magnitude slower than the 6 s ``intensity`` envelope every other scene rides
#: on: this is the only scene whose cost is *economic*, and an economy that
#: halts the instant the weather turns is a switch, not a drought. Ramping over
#: three quarters of a minute means the colony's food curve bends rather than
#: breaking, which is the difference between hardship and a cliff.
HEAT_RAMP_SEC = 45.0
#: Share of natural recovery the drought takes away at full heat. Everything
#: that grows back on its own funnels through ``props.growth_factor`` - sapling
#: growth, berry regrow, crop ripening and the reseeding that repopulates a
#: felled map - so this one number is the whole economic bite. 0.80 leaves a
#: fifth of normal: berries still come back, just far too slowly to feed anyone,
#: which is what makes the colony eat into the stockpile instead of living hand
#: to mouth off the bushes. It is a multiplier applied *live* and never a
#: subtraction, so nothing about the economy is permanently damaged - the moment
#: the scene ends the factor is 1.0 again (see _drought_hook).
#:
#: Measured over 12 seeds, each warmed up for 900 s and then forked into a full
#: 600 s heatwave slot and an identical 600 s of clear skies from the same save.
#: Food gained over the slot: +102 with the drought against +235 without, so the
#: colony's larder fills at 44% of its normal rate. Berries left standing on the
#: map at the end: 10.1 against 26.8 - the wild is stripped and does not come
#: back, which is the thing you can actually watch happen. Population 6.4
#: against 5.8 and deaths 3.00 against 2.58: within noise, which is the point.
#: This scene is meant to cost the colony its momentum, not its people.
#:
#: The rotation picks its next scene at random, so three heatwave slots can land
#: in a row. Stressed at 1800 s of unbroken drought over 6 seeds: no wipes, the
#: smallest colony still standing at 2, and the mean going 3.8 -> 6.2 with food
#: 177 -> 380. A drought slows the colony down; it never runs it out.
HEAT_DROUGHT_DEPTH = 0.80
#: Snow left by a previous blizzard goes this many times faster than the normal
#: thaw. Free texture: ``_melt_snow`` takes dt as a plain scale, so a heatwave
#: arriving on a white map visibly strips it back inside a minute.
HEAT_THAW_MUL = 3.5
#: Extra fatigue per second of full heat, on top of the baseline
#: ``entities.FATIGUE_PER_SEC`` (1/600). At 1/900 that is a two-thirds surcharge
#: on staying awake, which the colony pays in *sleep* - and sleep is time not
#: spent gathering. That indirect cost is the intended one; fatigue itself is
#: not a lethal need (only hunger and cold are), which is exactly why this scene
#: is allowed to touch it and is not allowed to touch hunger.
HEAT_FATIGUE_RATE = 1.0 / 900.0
#: ...and the fatigue no amount of heat will push anyone past. A hard ceiling
#: rather than a tuning target, in the shape SAND_HEALTH_FLOOR uses: an agent at
#: 1.0 fatigue drops everything to sleep, so a scene that could pin the whole
#: colony there would stop the economy dead rather than slowing it. Below the
#: cap the heat merely makes people tired sooner; the last stretch to exhaustion
#: is always the agent's own working day.
HEAT_FATIGUE_CAP = 0.80
#: Warmth (1 = freezing) bled off per second of full heat. The needs model has
#: no heatstroke - warmth is a one-sided cold clock - so the honest translation
#: of "sweltering" is that nobody is cold, day or night. It is the one thing the
#: scene gives back, and it is genuinely worth something: a heatwave night is the
#: only night in the game where the colony does not have to break off work and
#: crowd the firepit.
HEAT_RELIEF_RATE = 1.0 / 200.0
#: Gap between attempts at a brush fire, seconds. The wildfire scene relights
#: every 70-140 s; this is two to four times as slow *and* gated on the drought
#: being deep (below), so a ten-minute heatwave sees two or three ignitions at
#: the very most and often none at all. Fire is the scene's only body count and
#: it has to stay an event you notice, not a background hazard. Measured over
#: the 12-seed pairs described under HEAT_DROUGHT_DEPTH: 3.00 deaths per slot
#: against the control's 2.58, i.e. the brush costs the colony roughly one extra
#: life every second heatwave.
HEAT_IGNITE_MIN, HEAT_IGNITE_MAX = 190.0, 380.0
#: ...and how dry it has to get first. Below this the brush simply will not take,
#: so the first minute of the scene is always safe and an ignition reads as a
#: consequence of the drought rather than of the weather changing.
HEAT_IGNITE_HEAT = 0.55

#: Fire is tuned as "rare but nasty" rather than "constant and survivable".
#: props.py spreads fire on its own and a burnt tree is wood the colony never
#: gets, so the number of fires has to stay low (FIRE_RELIGHT_*) - which means
#: each one has to earn its 1-3 deaths. Measured over 10 seeds x 300 s of
#: wildfire: rare + gentle (6.0 s / 15 px) gives 0.9 deaths, rare + nasty
#: (3.5 s / 22 px) gives 2.6, and both leave 3.2 structures standing where the
#: old constant fire left 1.7.
BURN_DEATH_SEC = 3.5             # seconds of unbroken contact before it kills
BURN_TOUCH_DIST = 22.0           # horizontal reach of the flames
BURN_TOUCH_HEIGHT = 45.0         # ...but not onto a ledge far above/below
BURN_COOL_RATE = 1.2             # burn timer bleeds off this much faster once clear
#: How near a fire you have to be to bolt. This is the single most lethal
#: number in the file, because a panicking agent is allowed over ledges a calm
#: one refuses (see actions.step_toward): at 92 px a wildfire kept most of the
#: colony permanently panicking and killed 27 people per 10 runs by fall
#: against 13 by fire. At 30 px only the people genuinely in danger run.
FLEE_DIST = 30.0
BURN_PANIC = 1.5                 # panic seconds granted per tick near flames
FIRE_SPREAD_RATE = 0.15          # jumps/s from a lit prop to its neighbour
FIRE_SPREAD_REACH = 80.0         # px the fire can reach for its next victim
#: quiet spell before the hillside catches again once the last fire is out.
#: Too short and the map is stripped bare inside a scene: every tree gone is
#: wood the colony cannot build with (24 trees -> 1, and the build rate halved).
FIRE_RELIGHT_MIN, FIRE_RELIGHT_MAX = 70.0, 140.0
STRIKE_PANIC = 1.5               # panic seconds for everyone near a bolt
STRIKE_PANIC_R = 130.0           # ...and how near that is

#: The eclipse is the one scene that is an *event on a clock* rather than a
#: weather state: it happens once, near the top of the scene, and is over. The
#: scene rotation gives every scene 600 s, so these three add up to well under
#: that on purpose - the sun goes out, it comes back, and the colony gets the
#: rest of the slot to recover in ordinary daylight. Driven off ``scene_t``, so
#: the whole thing is a pure function of scene age: deterministic, seed-free,
#: and identical after a save/load.
ECLIPSE_INGRESS = 60.0           # first bite to totality
ECLIPSE_TOTALITY = 26.0          # the hold, with the sky dark
ECLIPSE_EGRESS = 55.0            # light coming back
#: Fraction of the ambient the shadow can take at full totality. 0.85 against
#: the scene's flat 0.90 daylight leaves ambient at 0.135 - within a whisker of
#: SCENE_AURORA's 0.15 night, which is the darkest the project already ships and
#: is known to still read on a desktop. It is applied to *ambient*, never to the
#: finished frame, which is what makes the torches the brightest thing on screen
#: instead of dimming with everything else (see Lighting.ambient_dim).
ECLIPSE_MAX_DIM = 0.85

#: The earthquake is a scene of *punctuation*, not of weather. A continuous
#: rumble reads as a broken renderer rather than a disaster, so the quiet is an
#: order of magnitude longer than the shaking: at these numbers a 600 s slot
#: runs 14-22 tremors and spends about 6% of its life actually moving.
QUAKE_LEAD_IN = 9.0              # calm before the first tremor of the scene
QUAKE_REST_MIN, QUAKE_REST_MAX = 15.0, 44.0
QUAKE_TREMOR_MIN, QUAKE_TREMOR_MAX = 2.0, 5.5
#: Peak screen-shake amplitude of the weakest and strongest tremor. add_shake
#: clamps at 18 and the renderer blits the *whole frame* at this offset, so
#: anything near that ceiling is unreadable rather than dramatic.
QUAKE_MAG_MIN, QUAKE_MAG_MAX = 5.0, 13.0
#: At or above this peak a tremor is a "big one": it tears the ground open,
#: cracks the buildings and earns a line in the chronicle. Below it the world
#: merely shakes. Roughly a third of tremors clear the bar.
QUAKE_BIG_MAG = 9.0

#: Fissures are capped hard on *both* axes, because this map kills by falling.
#: Terrain.deform's 'smooth' blend ramps over min(span // 2, DEFORM_EDGE = 24)
#: px, so a scar's steepest slope is about 1.57 * depth / that ramp. Tying depth
#: to half-width via QUAKE_FISSURE_ASPECT pins every scar at ~1.5 whatever size
#: it rolls: above MAX_SLOPE_WALK (0.9), so it is an obstacle that has to be
#: climbed round, and comfortably under MAX_SLOPE_CLIMB (2.6), so it is never a
#: cliff. Depth matters independently: at GRAVITY 900 a free fall needs ~64 px
#: to reach FALL_LETHAL_SPEED (340 px/s), so a 26 px ceiling cannot kill even if
#: somebody does drop straight in. Measured over 10 seeds x 600 s: zero deaths
#: attributable to the terrain this scene carves.
QUAKE_FISSURE_HALF_MIN, QUAKE_FISSURE_HALF_MAX = 15.0, 26.0
QUAKE_FISSURE_ASPECT = 0.95
QUAKE_FISSURE_DEPTH_MAX = 26.0
QUAKE_FISSURE_OPEN_SEC = 1.4     # seconds of tearing before a scar is fully open
QUAKE_FISSURE_MAX = 5            # most scars one run of the scene will leave
#: px between scar centres. Without it a run of big tremors walks a line of
#: overlapping notches across the map and they merge into one unbounded trench -
#: which is exactly the mass grave the depth cap exists to prevent.
QUAKE_FISSURE_GAP = 110.0
#: Refuse to open a scar where the ground already sits this far below the
#: shoulders either side of it. Measured against the local shoulders rather than
#: an absolute height so it is slope-neutral: it blocks digging an old crater or
#: last scene's fissure deeper without blocking honest hillsides.
QUAKE_FISSURE_SINK = 34.0
#: px either side of an open scar that reads as dangerous while the ground moves.
#: behaviour widens whatever it is given by 1.45x when it picks a flee target,
#: and up to QUAKE_FISSURE_MAX scars can be live at once, so this number is
#: really "what share of the map is off limits during a tremor": at 78 it was
#: over half of it, the colony spent the shaking running instead of working, and
#: the scene finished with fewer people than it started even though nothing it
#: did was lethal. 46 keeps each scar a place you walk around.
QUAKE_HAZARD_R = 46.0
#: Deliberately absent: this scene sets no ``agent.panic``. The flag lets an
#: agent step off drops it would normally refuse, and measured over 10 seeds x
#: 600 s it was the entire cost of the scene - falls went 15 (clear-sky control)
#: -> 39 with the flag -> 20 with only the hazard, for identical shaking,
#: identical scars and identical building damage. Deaths per run went 4.90 ->
#: 2.60 against a 2.40 control, and the colony went back to *growing* through
#: the scene (3.4 -> 4.0) instead of shrinking to the MIN_POP floor (3.4 ->
#: 2.5). Nothing here can kill, so nothing here needs the reckless kind of
#: urgency: hazards() moves people out of the way perfectly well without it.
#: See _warn_slide, which learned the same lesson the same way.

#: Share of a structure's max hp a tremor takes at the epicentre, tapering to
#: nothing at QUAKE_DAMAGE_R. A fraction rather than a flat number so a 120 hp
#: bridge and a 40 hp firepit both take a crack rather than a demolition.
QUAKE_STRUCT_FRAC = 0.14
#: ...and the floor no tremor will damage a building below. This is the single
#: dial that keeps the scene survivable rather than merely non-lethal: huts gate
#: population growth and firepits gate warmth, so a scene that flattens them
#: costs the colony far more than the bodies it never took. Buildings come out
#: of a bad quake cracked and get repaired between tremors (structures.damaged()
#: is already a standing repair job) instead of coming out as rubble.
QUAKE_STRUCT_FLOOR = 0.20
QUAKE_DAMAGE_R = 240.0
QUAKE_BOULDER_R = 150.0          # boulders this near the epicentre shake loose
QUAKE_BOULDER_MAX = 2            # ...and at most this many per tremor

#: --- volcanic eruption ----------------------------------------------------
#: Seconds of scene before the ground opens. The sky has already turned and the
#: ash is already falling by then, so the eruption reads as the arrival of
#: something that was visibly on its way rather than as a switch being thrown.
#: A threshold crossing of ``scene_t``, in the shape the eclipse and the
#: sandstorm use: nothing to initialise, nothing to clear, nothing to
#: round-trip, and a save loaded past the moment simply never fires it.
VOLCANO_ERUPT_AT = 5.0
#: ...and when a second vent tears open, well into the slot. One vent is a
#: disaster you walk away from in one direction; two is a decision.
VOLCANO_SECOND_VENT_AT = 250.0
#: Most flows alive at once, counting any left cooling by a previous eruption.
#: A hard cap rather than a tuning number: the rotation is free to hand this
#: scene two slots in a row, and an unbounded list of flows is an unbounded
#: share of the map under lava.
LAVA_MAX_FLOWS = 3
#: px the front advances per second. Deliberately a thirtieth of WALK_SPEED
#: (34 px/s): the flow must never be able to *catch* anybody, only to arrive
#: where they are standing. It is the ground going away underneath a job, not a
#: monster - which is what makes "lethal and avoidable" true by construction
#: rather than by tuning.
LAVA_SPREAD = 1.1
#: ...and how far one vent will ever reach, either side. This is the balance
#: dial of the scene. Two vents at the top of the range is 660 px of a 1600 px
#: map, and only at the very end of a ten-minute slot; the roll usually gives
#: less, and half of it has crusted to stone within a minute of the weather
#: moving on.
LAVA_HALF_MIN, LAVA_HALF_MAX = 100.0, 165.0
#: Seconds for an abandoned flow to congeal once the vent stops feeding it.
#:
#: This is the answer to "does cooled lava revert?", and the answer is *yes, to
#: stone*, for a reason that is nothing to do with geology: the material map is
#: permanent state. Nothing in the sim ever repaints a column back, so a scene
#: that left MAT_LAVA behind would leave a live, lethal, permanently growing
#: lava field for every future rotation to inherit and would own the whole map
#: inside an evening. Congealing to MAT_STONE keeps the scar - the eruption
#: really happened, and it really did change the ground - while bounding the
#: damage to a material the colony can walk over, build beside and mine.
LAVA_COOL_SEC = 75.0
#: Below this heat the crust has closed over: the flow still glows, but it no
#: longer burns and the AI stops giving it a wide berth. Without the cutoff the
#: colony would keep its distance from a field of cold stone for a minute after
#: the danger had passed.
LAVA_LETHAL_HEAT = 0.35
#: px between the points a flow is sampled at when it is handed to the fire-harm
#: path, and the most points one flow may contribute. The step sits comfortably
#: inside BURN_TOUCH_DIST (22 px) so the front is a continuous hot line rather
#: than stepping stones with cold gaps between them; the cap keeps the per-agent
#: distance scan bounded however wide the front gets.
LAVA_SAMPLE_STEP = 18.0
LAVA_SAMPLE_MAX = 24
#: px beyond the edge of the flow that reads as dangerous to the AI. Kept small
#: for the reason QUAKE_HAZARD_R spells out at length: behaviour widens whatever
#: it is handed by 1.45x when it picks a flee target, and a hazard covering half
#: the map makes the colony spend the scene running instead of working - which
#: on this terrain kills more people by falling than the hazard itself ever
#: does. 40 px is about one panicked stride past the edge of the heat.
LAVA_HAZARD_PAD = 40.0
#: How far from the colony the vent tries to open, in px, and how many places it
#: looks before settling. A vent under the firepit is not a disaster anybody can
#: respond to - it is a coin flip that ends the settlement - and the entire
#: claim of this scene is that it is survivable by moving.
LAVA_VENT_CLEAR = 520.0
LAVA_VENT_TRIES = 8
#: ...and how far the second vent keeps off the first. Two fronts that overlap
#: are not twice the disaster, they are a *trap*: behaviour flees the nearest
#: hazard, so a villager caught between two vents 170 px apart runs out of one
#: flow and straight into the other. Measured on the seed that produced that
#: layout, eight of its thirty-five deaths were people who died mid-flee.
LAVA_VENT_SPLIT = 420.0
#: How hard the picker avoids opening a vent that *severs the map*.
#:
#: The failure this exists to stop is quieter than being burned and much worse to
#: watch: a flow that opens between the colony and a map edge walls off the strip
#: beyond it, and anyone standing there is cut off from the stockpile. They do not
#: burn - they starve, in a flee/eat/sleep livelock, while hundreds of food sit in
#: a store they can no longer reach. Measured on seed 20260728: five of seven
#: deaths were hunger, four of them inside nineteen seconds, with ~300 food in the
#: stockpile and the victims pinned at x = 17..43 behind a flow spanning 191..417.
#:
#: The fix is geometric rather than behavioural: score down any vent that would
#: leave a walkable pocket behind it, so the picker prefers a front backed onto
#: the map edge (which strands nobody) over an island in the middle. The weight is
#: set so a severed pocket cannot be paid for by the distance term - that term
#: caps at LAVA_VENT_CLEAR, and it was precisely "get as far from home as
#: possible" that was steering vents into the middle of the far half of the map.
LAVA_ORPHAN_WEIGHT = 3.2
#: Seconds of contact with the flow before it kills, against BURN_DEATH_SEC's
#: 3.5 for a burning tree. Less than half, because molten rock and a campfire
#: should not share a fuse - and because with the turn-back ring in place the
#: 3.5 s version was survivable *by construction*: at flee speed (2.1x walk, so
#: 71 px/s) an agent clears 245 px in that time, which is wider than any flow
#: this scene builds, so nobody who ran ever died and the lava was lethal only
#: on paper. At 1.6 s the margin is 114 px: walking into the front and turning
#: straight round is survivable, being caught in the middle of one is not.
#: Measured over 16 seeds x 600 s: lava deaths went from 1 to 10, i.e. from
#: "never" to roughly one every other eruption, while total deaths went 3.75 ->
#: 4.31 against a clear-sky control's 3.56. The people it kills are the ones who
#: were already in trouble, and the colony still comes out of the slot with the
#: headcount it went in with.
LAVA_DEATH_SEC = 1.6
#: px from the flow at which a villager drops whatever job it was walking to and
#: re-decides (see _lava_interrupts). Bounded on both sides, and both bounds are
#: load-bearing:
#:
#: * above BURN_TOUCH_DIST (22 px), so the ordinary case is somebody stopping at
#:   the edge unhurt rather than stopping because they are already alight;
#: * strictly *below* LAVA_HAZARD_PAD (40 px), so anyone this interrupts is
#:   guaranteed to be inside the ring ``hazards`` publishes and will therefore
#:   score FleeFrom on the re-score it just forced. At 46 - i.e. outside the
#:   hazard ring - the band between the two rings was a livelock: the villager
#:   dropped the job, found no danger to flee, picked the same job again, took
#:   one step and dropped it again. Measured at 1048 abandonments of the same
#:   GatherWood in a single ten-minute slot.
LAVA_TURN_BACK = 34.0
#: Seconds for a building standing in the flow to go from whole to rubble.
#: Slow enough to watch happen, and slow enough that a sleeper evicted by the
#: collapse still has a long way to run before the front reaches him.
LAVA_RUINS_SEC = 15.0
#: Peak screen shake of the eruption itself, and of the swells that follow it.
#: The eruption is allowed to be violent (add_shake clamps at 18 and the
#: earthquake's biggest tremor is 13); the swells are not, because they run for
#: the rest of the slot and a wallpaper that never stops moving is a wallpaper
#: nobody keeps.
VOLCANO_ERUPT_SHAKE = 11.0
VOLCANO_SWELL_SHAKE = 3.2
#: Peak haze density. Below the sandstorm's 0.72 and well below the fog's 0.85:
#: this wash is dark rather than bright, so it *removes* contrast much faster
#: per unit than either of those, and the lava has to keep punching through it.
VOLCANO_HAZE = 0.42
#: Ash falls harder here than in the ashfall scene - that scene is the aftermath,
#: this one still has the vent open - but onto the same ASH_MAX_DEPTH ceiling, so
#: the drift arrives sooner and stops in the same place.
VOLCANO_ASH_MUL = 1.7
#: Snow goes faster than it would in a clear scene, for the same free-texture
#: reason the heatwave thaws it: a red-hot map that keeps last week's snowfield
#: reads as two scenes fighting over the same ground.
VOLCANO_THAW_MUL = 2.5

SHAKE_DECAY = 3.4                # exponential decay rate of screen shake

#: scene -> handler method name. Kept as strings so the instance holds no
#: bound-method cycles and stays trivially serialisable.
_HANDLERS: dict[str, str] = {
    SCENE_NIGHT_STORM: "_scene_night_storm",
    SCENE_CLEAR: "_scene_clear",
    SCENE_WILDFIRE: "_scene_wildfire",
    SCENE_MUDSLIDE: "_scene_mudslide",
    SCENE_BLIZZARD: "_scene_blizzard",
    SCENE_FLOOD: "_scene_flood",
    SCENE_METEOR: "_scene_meteor",
    SCENE_ASHFALL: "_scene_ashfall",
    SCENE_AURORA: "_scene_aurora",
    SCENE_FOG: "_scene_fog",
    SCENE_ECLIPSE: "_scene_eclipse",
    SCENE_EARTHQUAKE: "_scene_earthquake",
    SCENE_SANDSTORM: "_scene_sandstorm",
    SCENE_HEATWAVE: "_scene_heatwave",
    SCENE_VOLCANO: "_scene_volcano",
}

#: plausible successors, weighted. Storms clear, fires leave ash, etc.
_TRANSITIONS: dict[str, dict[str, float]] = {
    SCENE_NIGHT_STORM: {SCENE_CLEAR: 5, SCENE_FLOOD: 2, SCENE_MUDSLIDE: 2,
                        SCENE_WILDFIRE: 1, SCENE_ASHFALL: 1},
    SCENE_CLEAR:       {SCENE_NIGHT_STORM: 4, SCENE_WILDFIRE: 3, SCENE_BLIZZARD: 2,
                        SCENE_METEOR: 2, SCENE_ASHFALL: 1},
    SCENE_WILDFIRE:    {SCENE_ASHFALL: 3, SCENE_CLEAR: 3, SCENE_NIGHT_STORM: 2},
    SCENE_MUDSLIDE:    {SCENE_CLEAR: 4, SCENE_NIGHT_STORM: 2, SCENE_FLOOD: 2},
    SCENE_BLIZZARD:    {SCENE_CLEAR: 5, SCENE_NIGHT_STORM: 2},
    SCENE_FLOOD:       {SCENE_CLEAR: 4, SCENE_NIGHT_STORM: 2, SCENE_MUDSLIDE: 1},
    SCENE_METEOR:      {SCENE_WILDFIRE: 3, SCENE_CLEAR: 3, SCENE_ASHFALL: 2,
                        SCENE_NIGHT_STORM: 2},
    SCENE_ASHFALL:     {SCENE_CLEAR: 4, SCENE_NIGHT_STORM: 2, SCENE_BLIZZARD: 1},
    # An eruption ends in its own fallout far more often than it ends in blue
    # sky, and the ashfall scene is exactly that aftermath: the vent has closed,
    # the column is still coming down.
    SCENE_VOLCANO:     {SCENE_ASHFALL: 5, SCENE_WILDFIRE: 2, SCENE_CLEAR: 2},
}

#: Extra names for things this module may be handed that are not registered
#: kinds at all - duck-typed stand-ins from other modules and from the smoke
#: tests. Filtered against the registries below so one can never contradict a
#: spec: only names nothing else claims survive.
_FLAMMABLE_ALIASES = frozenset({"shrub", "log", "grass", "scaffold"})

#: Kind names that catch fire, for objects with no ``flammable`` flag of their
#: own to read (Prop has the property; Structure only has ``spec.flammable``).
#:
#: Derived, never written down. This was a hand-kept tuple and it drifted out of
#: agreement with STRUCTURE_SPECS: it listed 'wall' (spec says flammable=False)
#: and omitted 'stockpile', 'ladder' and the 'crop' prop. A bolt landing beside
#: a wall therefore picked the wall as its nearest fuel, Structure.ignite()
#: refused it, and _ignite() reported success anyway - so the strike consumed
#: its one ignition on something that cannot burn and the hut 20 px behind it
#: never caught. Measured: wall+hut, bolt on the wall, nothing alight.
_FLAMMABLE: frozenset[str] = (
    (frozenset(_PROP_FLAMMABLE)
     | frozenset(k for k, s in STRUCTURE_SPECS.items() if s.flammable)
     | _FLAMMABLE_ALIASES)
    - frozenset(k for k in _PROP_KINDS if k not in _PROP_FLAMMABLE)
    - frozenset(k for k, s in STRUCTURE_SPECS.items() if not s.flammable)
)


# ------------------------------------------------------- generic utilities --
def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _fnum(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):
        return default
    return f


def _stage(world: Any) -> tuple[float, float]:
    """``(lo, hi)`` of the strip a viewer can actually see, in world px.

    This is the single most important line in the whole 1600 -> 6400 migration
    as far as this file is concerned. Every scene here fires on a *rate*:
    STRIKE_MIN/MAX (4-12 s), METEOR_MIN/MAX (1.6-5 s), FIRE_RELIGHT (70-140 s),
    HEAT_IGNITE (190-380 s). Those rates were tuned against a world that was
    exactly one screen wide, so "one bolt every 8 seconds" and "one bolt every 8
    seconds *that you see*" were the same sentence. They are not any more.

    Tried and rejected: scaling the rates by WORLD_SCALE. That puts four times
    the disasters on the map, costs four times as much, and shows the viewer the
    same number of them, because three quarters land on hillside nobody is
    looking at. Keep the rate; narrow the siting. The file already argues for
    exactly this - DIRECT_HIT_CHANCE aims bolts at a victim, MUDSLIDE_AIM_CHANCE
    is 0.85, the quake epicentre picks a person 70% of the time, and every one
    of those was added because "a scene whose every event lands on empty
    hillside is a screen saver" (see _start_tremor). A 4x map made that failure
    four times as likely; this is the same fix applied to the other 30%.

    Guarded like everything else in this module: a stub world with no colony and
    no roster still gets a real span back rather than an exception on the tick
    path. The fallback matches actions.colony_center()'s own default so the two
    agree on where "nowhere in particular" is.
    """
    try:
        lo, hi = _stage_bounds(world)
        lo, hi = float(lo), float(hi)
        if hi - lo >= 1.0 and lo == lo and hi == hi:
            return lo, hi
    except Exception:
        log.debug("stage_bounds failed", exc_info=True)
    return WORLD_W * 0.5 - STAGE_HALF, WORLD_W * 0.5 + STAGE_HALF


def _stage_inset(world: Any, pad: float) -> tuple[float, float]:
    """:func:`_stage` pulled *pad* px in from both ends, never inverted.

    The call sites this replaces all read ``uniform(40.0, RENDER_W - 40.0)``:
    the inset kept an event off the very rim of the world so its blast radius
    had somewhere to land. That reason survives the migration unchanged, it just
    now applies to the edge of the *stage* rather than the edge of the map.
    """
    lo, hi = _stage(world)
    p = max(0.0, _fnum(pad, 0.0))
    if hi - lo <= 2.0 * p:              # stage narrower than the inset wants
        mid = (lo + hi) * 0.5
        return mid, mid
    return lo + p, hi - p


def _approach(cur: float, target: float, rate: float, dt: float) -> float:
    step = rate * dt
    if cur < target:
        return min(target, cur + step)
    if cur > target:
        return max(target, cur - step)
    return cur


def _iter(obj: Any, names: tuple[str, ...]) -> list[Any]:
    if obj is None:
        return []
    for name in names:
        v = getattr(obj, name, None)
        if v is None:
            continue
        if isinstance(v, dict):
            return list(v.values())
        try:
            return list(v)
        except TypeError:
            continue
    return []


def _agents(world: Any) -> list[Any]:
    # World keeps its roster in ``world.population`` (a Population, which is
    # iterable) - it has no ``.agents`` attribute. Leaving that name out of the
    # lookup made this return [] against the real World, which is why every
    # hazard that iterates agents killed nobody: the loop body never ran.
    return [a for a in _iter(world, ("agents", "population", "stickmen", "people"))
            if getattr(a, "alive", True)]


def _props(world: Any) -> list[Any]:
    out = _iter(world, ("props",))
    out.extend(_iter(world, ("structures",)))
    return out


def _aid(obj: Any) -> int | None:
    v = getattr(obj, "id", None)
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------ world access --
def _terrain(world: Any) -> Any:
    return getattr(world, "terrain", None)


def _height(world: Any) -> np.ndarray | None:
    h = getattr(_terrain(world), "height", None)
    if isinstance(h, np.ndarray) and h.ndim == 1 and h.size > 0:
        return h
    return None


def _material(world: Any) -> np.ndarray | None:
    m = getattr(_terrain(world), "material", None)
    if isinstance(m, np.ndarray) and m.ndim == 1 and m.size > 0:
        return m
    return None


def _ground_y(world: Any, x: float) -> float:
    t = _terrain(world)
    if t is not None:
        fn = getattr(t, "ground_y", None)
        if callable(fn):
            try:
                v = float(fn(float(x)))
                if v == v:
                    return v
            except Exception:
                pass
        h = _height(world)
        if h is not None:
            i = int(max(0, min(h.size - 1, int(x))))
            return float(h[i])
    return RENDER_H * 0.72


def _deform(world: Any, x0: float, x1: float, dy: float) -> None:
    """+dy lowers the surface (digs), -dy raises it. Never raises."""
    t = _terrain(world)
    if t is None:
        return
    fn = getattr(t, "deform", None)
    if callable(fn):
        try:
            fn(int(x0), int(x1), float(dy))
            return
        except Exception:
            pass
    h = _height(world)
    if h is None:
        return
    a = max(0, min(h.size, int(x0)))
    b = max(0, min(h.size, int(x1)))
    if b > a:
        h[a:b] = np.clip(h[a:b] + np.float32(dy), 12.0, float(RENDER_H - 4))


def _paint(world: Any, x0: float, x1: float, mat: int) -> None:
    t = _terrain(world)
    if t is None:
        return
    fn = getattr(t, "paint", None)
    if callable(fn):
        try:
            fn(int(x0), int(x1), int(mat))
            return
        except Exception:
            pass
    m = _material(world)
    if m is None:
        return
    a = max(0, min(m.size, int(x0)))
    b = max(0, min(m.size, int(x1)))
    if b > a:
        m[a:b] = np.uint8(mat)


def _crater(world: Any, x: float, y: float, radius: float) -> None:
    t = _terrain(world)
    if t is None:
        return
    fn = getattr(t, "crater", None)
    if callable(fn):
        # Terrain.crater is heightmap-column based: crater(cx, radius, depth).
        # It takes no y - the impact height is implicit in the surface. Passing
        # the screen y here as `radius` is arity-valid, so it raises nothing and
        # silently craters the entire 1280-column map instead of a small bowl.
        try:
            fn(int(x), int(max(3.0, radius)), float(max(3.0, radius) * 0.55))
            return
        except Exception:
            pass
    h = _height(world)
    if h is None:
        return
    r = max(3.0, float(radius))
    depth = r * 0.55
    a = max(0, int(x - r))
    b = min(h.size, int(x + r) + 1)
    if b <= a:
        return
    u = (np.arange(a, b, dtype=np.float32) - np.float32(x)) / np.float32(r)
    bowl = np.clip(1.0 - u * u, 0.0, 1.0)
    h[a:b] = np.clip(h[a:b] + bowl * np.float32(depth), 12.0, float(RENDER_H - 4))


def _lighting(world: Any) -> Any:
    return getattr(world, "lighting", None)


def _flash(world: Any, intensity: float, decay: float,
           color: tuple[int, int, int]) -> None:
    lig = _lighting(world)
    fn = getattr(lig, "add_flash", None)
    if callable(fn):
        try:
            fn(float(intensity), float(decay), tuple(color))
        except Exception:
            pass


def _chronicle(world: Any, text: str) -> None:
    if world is None or not text:
        return
    for name in ("chronicle", "history", "log"):
        obj = getattr(world, name, None)
        if obj is None:
            continue
        for meth in ("add", "record", "append", "log"):
            fn = getattr(obj, meth, None)
            if callable(fn):
                try:
                    fn(text)
                    return
                except Exception:
                    continue
    fn = getattr(world, "log_event", None)
    if callable(fn):
        try:
            fn(text)
        except Exception:
            pass


def _kill(world: Any, agent: Any, cause: str) -> bool:
    """Best-effort death. Prefers world.kill_agent, falls back to alive=False."""
    if agent is None or not getattr(agent, "alive", True):
        return False
    fn = getattr(world, "kill_agent", None)
    if callable(fn):
        try:
            fn(agent, cause)
            return True
        except TypeError:
            try:
                fn(agent)
                return True
            except Exception:
                pass
        except Exception:
            pass
    for meth in ("kill", "die"):
        fn = getattr(agent, meth, None)
        if callable(fn):
            try:
                fn(cause)
                return True
            except TypeError:
                try:
                    fn()
                    return True
                except Exception:
                    pass
            except Exception:
                pass
    try:
        agent.alive = False
    except Exception:
        return False
    try:
        agent.death_cause = cause
    except Exception:
        pass
    _chronicle(world, f"{getattr(agent, 'name', 'A stickman')} {cause}.")
    return True


def _panic(world: Any, agent: Any, target_x: float, duration: float,
           reason: str) -> None:
    fn = getattr(world, "panic", None)
    if callable(fn):
        try:
            fn(agent, float(target_x), float(duration), reason)
            return
        except TypeError:
            try:
                fn(agent, float(target_x), float(duration))
                return
            except Exception:
                pass
        except Exception:
            pass
    try:
        cur = _fnum(getattr(agent, "panic", 0.0), 0.0)
        agent.panic = max(cur, float(duration))
        # WORLD, not stage: a panic target is somewhere a person runs *to*, and
        # running off the edge of the camera is fine - running off the edge of
        # the land is not. Callers already pick targets relative to the agent.
        agent.panic_target_x = _clamp(float(target_x), 6.0, WORLD_W - 6.0)
        agent.panic_reason = reason
    except Exception:
        pass
    try:
        agent.morale = _clamp(_fnum(getattr(agent, "morale", 0.5), 0.5) - 0.02)
    except Exception:
        pass


def _is_flammable(prop: Any) -> bool:
    if getattr(prop, "burning", False) or getattr(prop, "burnt", False):
        return False
    if getattr(prop, "is_burning", False) or getattr(prop, "is_ruined", False):
        return False        # Structure spells both of those differently
    if getattr(prop, "destroyed", False) or not getattr(prop, "alive", True):
        return False
    flag = getattr(prop, "flammable", None)
    if isinstance(flag, bool):
        return flag
    for attr in ("kind", "type", "name"):
        v = getattr(prop, attr, None)
        if isinstance(v, str) and v:
            return v.lower() in _FLAMMABLE
    return False


def _ignite(world: Any, prop: Any) -> bool:
    """Set something alight. Returns whether it actually caught.

    The return value is load-bearing: three callers write a chronicle line off
    it ("Flames take hold in the brush."). Discarding what ``ignite()`` said and
    returning True regardless meant a refusal - ruined, already burning, or a
    kind whose spec says it does not burn - still printed the line with nothing
    alight anywhere on the map. Only ``False`` counts as a refusal; ``set_fire``
    and older stubs return None and are still taken at their word.
    """
    if prop is None:
        return False
    for meth in ("ignite", "set_fire"):
        fn = getattr(prop, meth, None)
        if callable(fn):
            try:
                return fn() is not False
            except TypeError:
                try:
                    return fn(1.0) is not False
                except Exception:
                    pass
            except Exception:
                pass
    try:
        if getattr(prop, "burning", False):
            return False
        prop.burning = True
        prop.burn_t = 0.0
        return True
    except Exception:
        return False


def _nudge_boulder(prop: Any, vx: float) -> bool:
    """Start a boulder rolling. Returns True if it is now in motion.

    props.nudge() is a module function rather than a method, so importing it
    here would be the first hard coupling from events to props in the file.
    Duck-typed instead, exactly as _ignite is: try a ``nudge`` method if some
    future prop grows one, otherwise write the two state keys props.nudge sets.
    """
    if prop is None:
        return False
    fn = getattr(prop, "nudge", None)
    if callable(fn):
        try:
            fn(float(vx))
            return True
        except Exception:
            pass
    state = getattr(prop, "state", None)
    if not isinstance(state, dict):
        return False
    try:
        state["rolling"] = True
        state["vx"] = float(vx)
        return True
    except Exception:
        return False


def _clean_rgb(v: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    """Coerce a loaded colour back to three bytes, or fall back to *default*.

    JSON round-trips a tuple as a list, and a hand-edited save can hold anything
    at all - so this is the same defensive shape ``from_dict`` already uses for
    ``tint``, factored out because there are now two colours to load.
    """
    try:
        r, g, b = (int(c) for c in tuple(v)[:3])
    except (TypeError, ValueError):
        return default
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def _clean_fissure(d: dict[str, Any]) -> dict[str, Any]:
    """Re-clamp one loaded scar back inside the caps that created it."""
    half = _clamp(_fnum(d.get("half"), QUAKE_FISSURE_HALF_MIN),
                  QUAKE_FISSURE_HALF_MIN, QUAKE_FISSURE_HALF_MAX)
    target = _clamp(_fnum(d.get("target"), half * QUAKE_FISSURE_ASPECT),
                    0.0, QUAKE_FISSURE_DEPTH_MAX)
    return {
        "x": _clamp(_fnum(d.get("x"), WORLD_W * 0.5), 0.0, float(WORLD_W)),
        "half": half,
        "depth": _clamp(_fnum(d.get("depth"), 0.0), 0.0, target),
        "target": target,
        "y": _clamp(_fnum(d.get("y"), RENDER_H * 0.7), 0.0, float(RENDER_H)),
        "seed": int(_fnum(d.get("seed"), 1.0)) & 0xFFFFF,
    }


def _clean_lava(d: dict[str, Any]) -> dict[str, Any]:
    """Re-clamp one loaded flow back inside the caps that created it.

    Every field is bounded by another: ``half`` cannot exceed the reach the vent
    rolled, and ``paint`` - the debt this flow owes the material map - cannot
    exceed the reach it has actually advanced to. A hand-edited save asking for
    a 900 px flow would otherwise repaint most of the world to stone the moment
    it started cooling.
    """
    cap = _clamp(_fnum(d.get("max"), LAVA_HALF_MAX), 0.0, LAVA_HALF_MAX)
    half = _clamp(_fnum(d.get("half"), 0.0), 0.0, cap)
    return {
        "x": _clamp(_fnum(d.get("x"), WORLD_W * 0.5), 0.0, float(WORLD_W)),
        "half": half,
        "max": cap,
        "hot": _clamp(_fnum(d.get("hot"), 0.0)),
        "paint": _clamp(_fnum(d.get("paint"), 0.0), 0.0, half),
        "y": _clamp(_fnum(d.get("y"), RENDER_H * 0.7), 0.0, float(RENDER_H)),
        "seed": int(_fnum(d.get("seed"), 1.0)) & 0xFFFFF,
    }


def _extinguish(prop: Any) -> None:
    fn = getattr(prop, "extinguish", None)
    if callable(fn):
        try:
            fn()
            return
        except Exception:
            pass
    try:
        prop.burning = False
    except Exception:
        pass


def _nearest_flammable(world: Any, x: float, max_dist: float) -> Any:
    best, best_d = None, max_dist
    for p in _props(world):
        if not _is_flammable(p):
            continue
        d = abs(_fnum(getattr(p, "x", 1e9), 1e9) - x)
        if d < best_d:
            best, best_d = p, d
    return best


# ================================================================= system ==
class EventSystem:
    """Drives weather and disasters, and applies their consequences."""

    def __init__(self, scene: str = SCENE_NIGHT_STORM, seed: int | None = None) -> None:
        self.scene: str = scene if scene in SCENES else SCENE_NIGHT_STORM
        self.scene_t: float = 0.0
        self.intensity: float = 0.0            # 0..1 envelope, ramps in on entry

        # weather channels ------------------------------------------------
        self.wind: float = 0.0                 # -1..1, signed (screen x)
        self.rain: float = 0.0                 # 0..1
        self.snow: float = 0.0                 # 0..1
        self.ash: float = 0.0                  # 0..1
        self.fog: float = 0.0                  # 0..1 mist density (renderer wash)
        #: Colour of that wash. The haze is a *density* channel plus a colour so
        #: one renderer path can serve every low-visibility scene: fog keeps the
        #: grey it has always had, the sandstorm asks for tan, and anything added
        #: later only has to name a colour. Published rather than derived from
        #: ``scene`` in the renderer so the two cannot drift, and round-tripped
        #: through to_dict so a save taken mid-storm reloads the same murk.
        self.fog_color: tuple[int, int, int] = HAZE_GREY
        #: 0..1 depth of the solar eclipse. Read by the sky (dark gradient,
        #: stars, the occluded disc), by fx (the horizon twilight ring) and, via
        #: _publish, by Lighting.ambient_dim - which is where the actual
        #: darkness lives. Zero in every scene but SCENE_ECLIPSE.
        self.eclipse: float = 0.0
        #: 0..1 drought depth. One channel serves both halves of SCENE_HEATWAVE:
        #: the renderer scales its ground shimmer by it, and ``_drought_hook``
        #: turns it into the multiplier on ``world.regrowth_factor`` that stalls
        #: every natural-recovery rate in props.py. Deliberately *one* field and
        #: not two - the look and the economics are the same envelope, and a
        #: second field would be a second thing to initialise here, clear in
        #: _reset_scene_state and round-trip through to_dict, which is where this
        #: file's renderer-facing channels have historically gone wrong. Zero in
        #: every scene but the heatwave.
        self.heat: float = 0.0
        self.gust: float = 0.0                 # 0..1 candle guttering strength
        self.water_level: float | None = None  # y of the flood line, or None
        self.quake_t: float = 0.0              # seconds of quake left
        #: Open ground scars, as ``{x, half, depth, target, y, seed}``. Read by
        #: the renderer (which darkens the notch the sim carved) and by
        #: ``hazards`` (which makes villagers give them a wide berth while the
        #: ground is moving). Only SCENE_EARTHQUAKE ever fills this; it is
        #: cleared on every scene change, so the *terrain* keeps the scar but the
        #: crack overlay leaves with the weather that made it.
        self.fissures: list[dict[str, Any]] = []
        #: Molten flows, as ``{x, half, max, hot, paint, y, seed}``. ``half`` is
        #: how far the front has advanced either side of the vent, ``hot`` is
        #: 1.0 while the vent feeds it and falls to zero as it congeals, and
        #: ``paint`` is how much of the material map is currently claimed (so
        #: the crust can be handed back to MAT_STONE from the outside in).
        #: Read by the renderer (the emissive band and its light), by
        #: ``hazards`` (villagers give the front a wide berth) and by
        #: ``_lava_step`` (which burns whoever is standing in one).
        #:
        #: Unlike ``fissures`` this is *deliberately not cleared* on a scene
        #: change - see _reset_scene_state. The flow is painted into the shared
        #: material map, and dropping the bookkeeping would strand a permanent
        #: lethal lava field there with nothing left alive to cool it.
        self.lava: list[dict[str, Any]] = []

        # renderer-facing transients --------------------------------------
        self.pending: list[dict[str, Any]] = []
        self.strikes: list[dict[str, Any]] = []
        self.meteors: list[dict[str, Any]] = []
        self.ember_rate: float = 0.0           # 0..1 particle budget hint
        self.tint: tuple[int, int, int] = (255, 255, 255)
        self.rumble: float = 0.0               # 0..1 mudslide warning
        self.fireflies: bool = False
        self.shake_amp: float = 0.0
        self.shake_t: float = 0.0

        # internal --------------------------------------------------------
        self.t: float = 0.0
        self.seed: int = int(seed) if seed is not None else random.randrange(1 << 30)
        self._rng = random.Random(self.seed)
        self.next_strike: float = self._rng.uniform(2.0, 6.0)
        self.next_meteor: float = self._rng.uniform(1.0, 3.0)
        self.next_ignite: float = 0.5
        #: Earthquake pacing. Seeded off a plain constant rather than a draw from
        #: ``_rng`` on purpose: an extra draw here (and in _reset_scene_state)
        #: would shift the seeded stream every other scene rides, so identical
        #: seeds would stop replaying identically the moment this scene existed.
        self.next_tremor: float = QUAKE_LEAD_IN
        self.tremor_mag: float = 0.0           # peak shake of the current tremor
        self.tremor_len: float = 0.0           # ...and how long it was given
        self.snow_depth: float = 0.0
        self.ash_depth: float = 0.0
        #: Like snow_depth and ash_depth this is *not* cleared on a scene change:
        #: the drift is in the heightmap and the material map, and the ground
        #: does not un-bury itself because the weather moved on.
        self.sand_depth: float = 0.0
        self._snow_prev: list[int] | None = None
        self.slide_phase: str = "idle"
        self.slide_x0: float = 0.0
        self.slide_x1: float = 0.0
        self.slide_t: float = 0.0
        self.flood_y0: float | None = None
        self.flood_y1: float | None = None
        self.flood_t: float = 0.0              # surge clock, resets per surge
        self._submerged: dict[int, float] = {}
        self._buried: dict[int, float] = {}    # agent id -> s inside the slide
        self._burning: dict[int, float] = {}
        self._want_advance: bool = False
        self._cov_order: np.ndarray | None = None
        self._errors: int = 0
        self._last_err_log: float = -1e9

    # --------------------------------------------------------------- tick --
    def tick(self, world: Any, dt: float) -> None:
        """Advance the scene and apply its consequences. Never raises."""
        step = _fnum(dt, 0.0)
        if step <= 0.0:
            return
        step = min(step, 0.25)
        self.t += step
        self.scene_t += step
        self.intensity = _approach(self.intensity, 1.0, 1.0 / 6.0, step)

        self._safe(self._tick_transients, world, step)
        handler = getattr(self, _HANDLERS.get(self.scene, "_scene_clear"), None)
        if handler is not None:
            self._safe(handler, world, step)
        self._safe(self._tick_pending, world, step)
        self._safe(self._tick_quake, world, step)
        self._safe(self._tick_panic, world, step)
        self._safe(self._publish, world, step)

        cfg = getattr(world, "cfg", None) or getattr(world, "config", None)
        if cfg is not None:
            self._safe(self.auto_advance, cfg)

    def _safe(self, fn: Any, *args: Any) -> None:
        try:
            fn(*args)
        except Exception as exc:                            # pragma: no cover
            self._note_error(getattr(fn, "__name__", "?"), exc)

    def _note_error(self, where: str, exc: BaseException) -> None:
        self._errors += 1
        if self.t - self._last_err_log > 30.0:
            self._last_err_log = self.t
            log.warning("events: %s failed (%s: %s) [%d total]",
                        where, type(exc).__name__, exc, self._errors)

    # ------------------------------------------------------ shared per-tick --
    def _tick_transients(self, world: Any, dt: float) -> None:
        for s in self.strikes:
            s["t"] = _fnum(s.get("t"), 0.0) + dt
        if self.strikes:
            self.strikes = [s for s in self.strikes if s["t"] < STRIKE_TTL]
        self.shake_t += dt
        if self.shake_amp > 0.0:
            self.shake_amp *= math.exp(-SHAKE_DECAY * dt)
            if self.shake_amp < 0.05:
                self.shake_amp = 0.0
        self.rumble = max(0.0, self.rumble - dt * 0.8)
        self.fireflies = False
        # Lava is a transient too - a slow one. It lives here rather than in the
        # volcano handler on purpose: the flow is painted into the shared
        # material map, so *something* has to keep cooling it after the rotation
        # has moved on to a clear afternoon. See _lava_step.
        self._lava_step(world, dt)

    def _tick_panic(self, world: Any, dt: float) -> None:
        """Run the panic clock down.

        The module docstring calls ``agent.panic`` a countdown in seconds and
        events.py is its only writer - but nothing anywhere was decrementing
        it. One hazard call therefore pinned an agent above zero forever, so
        ``behavior.emergency_override`` kept returning True, the agent
        re-decided every AI tick and stayed at flee speed for the rest of the
        session. On cliff terrain that is a death sentence: measured over
        5x300s of wildfire, fall deaths went 5 (no panic at all) -> 67 (panic,
        no countdown) -> 32 (panic with this countdown), with burn deaths
        unchanged at 7-8. Fleeing should be a spike, not a personality.
        """
        for ag in _agents(world):
            left = _fnum(getattr(ag, "panic", 0.0), 0.0)
            if left <= 0.0:
                continue
            try:
                ag.panic = max(0.0, left - dt)
            except Exception:
                break

    def _publish(self, world: Any, dt: float) -> None:
        """Push the few values other subsystems read off us."""
        lig = _lighting(world)
        if lig is not None:
            try:
                lig.wind_gust = _clamp(self.gust)
            except Exception:
                pass
            try:
                # Unconditional, every tick, in every scene - not inside the
                # eclipse handler. That is what makes the eclipse impossible to
                # leak: ``self.eclipse`` is zero everywhere else, so this writes
                # 1.0 back the moment the shadow passes or the rotation moves on,
                # and a save loaded mid-eclipse recomputes the right factor on its
                # first tick without anyone re-deriving it.
                lig.ambient_dim = 1.0 - ECLIPSE_MAX_DIM * _clamp(self.eclipse)
            except Exception:
                pass
        # Unconditional, every tick, in every scene - for the reason spelled out
        # above the eclipse line. The hook lifts itself when self.heat is zero,
        # so a save loaded mid-drought re-installs it and any scene that follows
        # one is guaranteed to get its economy back without having to know the
        # heatwave ever existed.
        self._drought_hook(world)

    def _drought_hook(self, world: Any) -> None:
        """Install (or lift) the drought multiplier on ``world.regrowth_factor``.

        ``props.growth_factor(world)`` is the single funnel every natural
        recovery rate in the sim goes through - sapling growth, berry regrow,
        crop ripening and the reseeding that repopulates a stripped map - and it
        resolves that rate by looking up ``regrowth_factor`` on the world and
        calling it. Wrapping that one callable is therefore the entire economic
        effect of the scene, with no edit to props.py or world.py and no second
        channel for the two to disagree about.

        Three details are load-bearing:

        * The wrapper is a **callable**, not a plain number. ``growth_factor``
          would happily accept a float attribute, but shadowing a method with a
          float is a landmine for any future caller that writes
          ``world.regrowth_factor()`` - it would raise TypeError inside
          World._guarded and disable a whole subsystem for the session.
        * It reads ``self.heat`` **live** rather than capturing a multiplier, so
          the drought lifts the instant the field is zeroed, including on the
          tick where _reset_scene_state runs after we have already published.
        * It records the callable it replaced and refuses to stack, so a reload
          (which builds a fresh EventSystem against the same World) replaces the
          old wrapper instead of nesting a second drought inside it.

        Fails soft in every direction: a world with no ``__dict__`` (slots, or a
        test stub) simply gets no drought, which costs the scene its economics
        and nothing else.
        """
        d = getattr(world, "__dict__", None)
        if not isinstance(d, dict):
            return
        cur = d.get("regrowth_factor")
        if getattr(cur, "_drought_owner", None) is self and self.heat > 0.0:
            return                              # already installed, still wanted

        # Peel off any wrapper - ours and no longer wanted, or a stale one left
        # behind by the EventSystem a reload replaced - so that `cur` below is
        # whatever the World genuinely had. Popping is the normal case: the real
        # World defines regrowth_factor on the class, so an empty instance dict
        # is what "no drought" looks like.
        if getattr(cur, "_drought_owner", None) is not None:
            prev = getattr(cur, "_drought_prev", None)
            if prev is None:
                d.pop("regrowth_factor", None)
            else:
                d["regrowth_factor"] = prev
            cur = prev
        if self.heat <= 0.0:
            return

        # Resolved only now the instance dict is clean, so this cannot pick up
        # the wrapper we were about to replace and recurse into itself.
        base = cur if cur is not None else getattr(world, "regrowth_factor", None)

        def _drought(_base: Any = base, _ev: "EventSystem" = self) -> float:
            raw = 1.0
            if callable(_base):
                try:
                    raw = _fnum(_base(), 1.0)
                except Exception:
                    raw = 1.0
            elif isinstance(_base, (int, float)) and not isinstance(_base, bool):
                raw = float(_base)
            return raw * max(0.0, 1.0 - HEAT_DROUGHT_DEPTH * _clamp(_ev.heat))

        _drought._drought_owner = self          # type: ignore[attr-defined]
        _drought._drought_prev = cur            # type: ignore[attr-defined]
        d["regrowth_factor"] = _drought

    def _tick_pending(self, world: Any, dt: float) -> None:
        if not self.pending:
            return
        due: list[dict[str, Any]] = []
        keep: list[dict[str, Any]] = []
        for ev in self.pending:
            if not isinstance(ev, dict):
                continue
            ev["t"] = _fnum(ev.get("t"), 0.0) - dt
            (due if ev["t"] <= 0.0 else keep).append(ev)
        self.pending = keep[:32]
        for ev in due:
            self._safe(self._fire, world, str(ev.get("kind", "")), ev)

    def schedule(self, delay: float, kind: str, **data: Any) -> None:
        """Queue a one-shot event ``delay`` seconds from now."""
        if len(self.pending) >= 32:
            return
        ev: dict[str, Any] = {"t": max(0.0, _fnum(delay, 0.0)), "kind": str(kind)}
        ev.update(data)
        self.pending.append(ev)

    def _fire(self, world: Any, kind: str, ev: dict[str, Any]) -> None:
        if kind == "ignite":
            # An "ignite" with no x is "something catches, somewhere". STAGE:
            # the whole point of the event is that a viewer watches it take
            # hold, and _nearest_flammable's 400 px reach is short enough that
            # a world-wide draw would usually find nothing and silently no-op.
            x = _fnum(ev.get("x"), self._rng.uniform(*_stage(world)))
            p = _nearest_flammable(world, x, 400.0)
            if p is not None and _ignite(world, p):
                _chronicle(world, "Flames take hold in the brush.")
        elif kind == "strike":
            self._lightning_strike(world)
        elif kind == "quake":
            self.trigger_quake(world, _fnum(ev.get("dur"), 2.5),
                               _fnum(ev.get("mag"), 6.0))
        elif kind == "slide":
            self.slide_phase = "slide"
            self.slide_t = 0.0
        elif kind == "advance":
            self._want_advance = True

    # -------------------------------------------------------------- weather --
    def _noise(self, slow: float = 0.31, fast: float = 1.13, off: float = 0.0) -> float:
        """Cheap smooth -1..1 noise from two incommensurate sines."""
        return (0.6 * math.sin(self.t * slow + off)
                + 0.4 * math.sin(self.t * fast + off * 1.7 + 1.3))

    def _approach_env(self, dt: float, rain: float = 0.0, snow: float = 0.0,
                      ash: float = 0.0, rate: float = 0.5) -> None:
        k = self.intensity
        self.rain = _approach(self.rain, _clamp(rain) * k, rate, dt)
        self.snow = _approach(self.snow, _clamp(snow) * k, rate, dt)
        self.ash = _approach(self.ash, _clamp(ash) * k, rate, dt)

    def _set_wind(self, dt: float, amplitude: float, gustiness: float,
                  slow: float = 0.27, fast: float = 1.07) -> None:
        target = _clamp(amplitude * self._noise(slow, fast), -1.0, 1.0) * self.intensity
        self.wind = _approach(self.wind, target, 0.9, dt)
        g = 0.5 + 0.5 * math.sin(self.t * 2.3 + 0.7)
        self.gust = _clamp(abs(self.wind) * (0.5 + 0.5 * g) * gustiness)

    # ------------------------------------------------------ accumulation fx --
    def _coverage_order(self, width: int) -> np.ndarray:
        if (self._cov_order is None or self._cov_order.size != width):
            rng = np.random.default_rng(self.seed & 0xFFFFFFFF)
            self._cov_order = rng.permutation(width).astype(np.int32)
        return self._cov_order

    def _accumulate(self, world: Any, dt: float, mat: int, depth: float,
                    max_depth: float, rate: float, remember: bool) -> float:
        """Raise the ground a little and repaint a growing share of columns."""
        h = _height(world)
        m = _material(world)
        if h is None:
            return depth
        new_depth = min(max_depth, depth + rate * dt * self.intensity)
        delta = new_depth - depth
        if delta <= 1e-4:
            return depth
        h -= np.float32(delta)                    # ground surface rises (y down)
        np.clip(h, 12.0, float(RENDER_H - 4), out=h)
        if m is not None:
            order = self._coverage_order(int(m.size))
            n = int(_clamp(new_depth / max_depth) * m.size)
            if n > 0:
                cols = order[:n]
                if remember and self._snow_prev is None:
                    self._snow_prev = [int(v) for v in m.tolist()]
                m[cols] = np.uint8(mat)
        return new_depth

    def _melt_snow(self, world: Any, dt: float) -> None:
        if self.snow_depth <= 0.0:
            return
        h = _height(world)
        m = _material(world)
        back = min(self.snow_depth, 0.09 * dt)
        if h is not None and back > 0.0:
            h += np.float32(back)
            np.clip(h, 12.0, float(RENDER_H - 4), out=h)
        self.snow_depth = max(0.0, self.snow_depth - back)
        if m is not None and self._snow_prev is not None:
            prev = np.asarray(self._snow_prev[:m.size], dtype=np.uint8)
            if prev.size == m.size:
                frac = _clamp(self.snow_depth / SNOW_MAX_DEPTH)
                order = self._coverage_order(int(m.size))
                keep = order[:int(frac * m.size)]
                mask = np.zeros(m.size, dtype=bool)
                if keep.size:
                    mask[keep] = True
                thawed = (m == np.uint8(MAT_SNOW)) & (~mask)
                m[thawed] = prev[thawed]
        if self.snow_depth <= 0.0:
            self._snow_prev = None

    # ================================================== scene: night storm ==
    def _scene_night_storm(self, world: Any, dt: float) -> None:
        swell = 0.5 + 0.5 * (0.5 + 0.5 * math.sin(self.t * 0.11))
        self._approach_env(dt, rain=_clamp(swell, 0.5, 1.0), rate=0.35)
        self._set_wind(dt, 0.85, 1.0, slow=0.23, fast=0.97)
        self.tint = (206, 218, 255)
        self.ember_rate = 0.0
        self.next_strike -= dt * (0.4 + 0.6 * self.intensity)
        if self.next_strike <= 0.0:
            self.next_strike = self._rng.uniform(STRIKE_MIN, STRIKE_MAX)
            self._lightning_strike(world)
        # A bolt can set a tree alight mid-storm (see _strike_damage), so the
        # burn path has to run here too - it used to live only in the wildfire
        # handler, which meant a storm fire was pure decoration. Rain still
        # douses it below, so storm fires are short and usually survivable.
        storm_fires = self._burning_props(world)
        if storm_fires:
            self._fire_harms_agents(world, dt, storm_fires)
        self._rain_douses(world, dt)

    def strike_at(self, world: Any, x: float, direct: bool = True) -> None:
        """A lightning bolt at a chosen x. The player's Lightning tool calls
        this; it is the aimed sibling of the scene's random _lightning_strike."""
        try:
            # WORLD: the player clicked somewhere. app.py has already mapped the
            # click back through the camera into a world x, so this clamp only
            # has to keep the bolt on the land.
            x = max(24.0, min(float(WORLD_W - 24.0), float(x)))
            gy = _ground_y(world, x)
            self.strikes.append({
                "x": float(x), "t": 0.0,
                "seed": int(self._rng.randrange(1 << 20)),
                "ground_y": float(gy), "direct": bool(direct),
            })
            if len(self.strikes) > 6:
                del self.strikes[0]
            self._bump_strike_stat(world)
            _flash(world, 1.0 if direct else 0.7, 0.45, LIGHTNING_COLOR)
            self.add_shake(7.5 if direct else 3.0)
            if direct:
                self._strike_damage(world, x, gy)
        except Exception:
            log.debug("strike_at failed", exc_info=True)

    def meteor_at(self, world: Any, x: float) -> None:
        """A meteor impact at a chosen x. The player's Meteor tool calls this."""
        try:
            x = max(8.0, min(float(WORLD_W - 8.0), float(x)))   # aimed: WORLD
            self._meteor_impact(world, x, _ground_y(world, x))
        except Exception:
            log.debug("meteor_at failed", exc_info=True)

    def _lightning_strike(self, world: Any) -> None:
        direct = self._rng.random() < DIRECT_HIT_CHANCE
        # STAGE. STRIKE_MIN/MAX is a per-second rate, so widening this draw to
        # WORLD_W would not give the viewer more lightning - it would give them
        # a quarter as much, at the same cost, with the rest flashing over
        # hillside 3 km away. The rate stays; the siting narrows.
        x = self._rng.uniform(*_stage_inset(world, 24.0))

        # A "direct" strike has to actually aim. Rolling a uniform x across
        # 1280px and then killing only within 25px meant a direct hit landed on
        # somebody with probability ~4%, so across a whole storm nobody was ever
        # struck and feature 29 never once fired. So: pick a victim and strike
        # near them - but the jitter has to stay inside DIRECT_KILL_RADIUS or
        # the aim is a lie. At the old +/-34px against a 25px radius roughly one
        # aimed bolt in three still landed too wide to hurt anyone.
        #
        # Both re-clamps below are WORLD, deliberately, and this is the one
        # place in the file where stage-clamping would be an outright bug: the
        # victim's x is wherever the victim is, and pulling the bolt back onto
        # the stage would shove it out of its own kill radius - exactly the
        # failure the second clamp was written to prevent. A colonist standing
        # past the stage edge is rare (they are inside SITE_RANGE nearly always)
        # but "rare" is how this class of bug survives.
        if direct:
            crowd = [a for a in _agents(world) if getattr(a, "alive", False)]
            if crowd:
                victim = crowd[self._rng.randrange(len(crowd))]
                vx = _fnum(getattr(victim, "x", x), x)
                x = vx + self._rng.uniform(-DIRECT_AIM_JITTER, DIRECT_AIM_JITTER)
                x = max(24.0, min(WORLD_W - 24.0, x))
                if abs(x - vx) > DIRECT_KILL_RADIUS:
                    x = vx + math.copysign(DIRECT_KILL_RADIUS * 0.5, x - vx)
                    x = max(0.0, min(float(WORLD_W - 1), x))
        gy = _ground_y(world, x)
        self.strikes.append({
            "x": float(x), "t": 0.0,
            "seed": int(self._rng.randrange(1 << 20)),
            "ground_y": float(gy), "direct": bool(direct),
        })
        if len(self.strikes) > 6:
            del self.strikes[0]
        self._bump_strike_stat(world)
        if direct:
            _flash(world, 1.0, 0.45, LIGHTNING_COLOR)
            self.add_shake(7.5)
            self._strike_damage(world, x, gy)
        else:
            _flash(world, self._rng.uniform(0.5, 0.85),
                   self._rng.uniform(0.22, 0.38), LIGHTNING_COLOR)

    @staticmethod
    def _bump_strike_stat(world: Any) -> None:
        """World keeps a ``lightning_strikes`` counter that nothing was ever
        incrementing. Best-effort; a world without stats is fine."""
        try:
            stats = getattr(world, "stats", None)
            if isinstance(stats, dict):
                stats["lightning_strikes"] = int(stats.get("lightning_strikes", 0)) + 1
        except Exception:
            pass

    def _strike_damage(self, world: Any, x: float, gy: float) -> None:
        # A bolt is a vertical column, so the hit test is a *column* test.
        # The old `hypot(ax - x, ay - gy)` compared the agent's own y against
        # the ground under the bolt: on any slope that vertical term alone blew
        # past the 25px radius, so even a bolt landing 8px away killed nobody.
        # Guard the vertical only against being a whole terrace away.
        for ag in _agents(world):
            ax = _fnum(getattr(ag, "x", 1e9), 1e9)
            ay = _fnum(getattr(ag, "y", 1e9), 1e9)
            if abs(ax - x) > DIRECT_KILL_RADIUS:
                continue
            if abs(ay - _ground_y(world, ax)) > 120.0:
                continue                       # airborne / on a tower, spared
            if abs(ay - gy) > 120.0:
                continue                       # different terrace entirely
            if _kill(world, ag, "lightning"):
                _chronicle(world, f"{getattr(ag, 'name', 'A stickman')} "
                                  f"was struck by lightning.")
        _paint(world, x - SCORCH_HALF_WIDTH, x + SCORCH_HALF_WIDTH, MAT_ASH)
        p = _nearest_flammable(world, x, STRIKE_IGNITE_REACH)
        if p is not None:
            _ignite(world, p)
        for ag in _agents(world):
            ax = _fnum(getattr(ag, "x", 1e9), 1e9)
            if abs(ax - x) < STRIKE_PANIC_R:
                away = -STRIKE_PANIC_R if ax < x else STRIKE_PANIC_R
                _panic(world, ag, ax + away, STRIKE_PANIC, "lightning")

    def _rain_douses(self, world: Any, dt: float) -> None:
        if self.rain < 0.55:
            return
        if self._rng.random() > dt * 0.25 * self.rain:
            return
        for p in _props(world):
            if self._is_alight(p):        # structures included - they burn too
                _extinguish(p)
                break

    # ======================================================== scene: clear ==
    def _scene_clear(self, world: Any, dt: float) -> None:
        self._approach_env(dt, rate=0.25)
        self._set_wind(dt, 0.28, 0.35, slow=0.19, fast=0.61)
        self.tint = (255, 255, 255)
        self.ember_rate = 0.0
        self.water_level = None
        wt = _fnum(getattr(world, "world_time", self.t), self.t)
        self.fireflies = is_night(wt)
        self._melt_snow(world, dt)

    # ======================================================= scene: aurora ==
    def _scene_aurora(self, world: Any, dt: float) -> None:
        """A calm, clear deep-night sky lit by aurora ribbons. No hazards - the
        whole payoff is the light, and a scatter of torches under the ribbons.
        The renderer pins this scene to midnight, so it always reads as night
        whatever the world clock says, and the ambient curve is pinned to match."""
        self._approach_env(dt, rate=0.30)          # let any lingering rain/snow clear
        self._set_wind(dt, 0.20, 0.28, slow=0.15, fast=0.50)
        self.tint = (150, 200, 220)                # a faint cool cast
        self.ember_rate = 0.0
        self.water_level = None
        self.fireflies = True                      # a still night draws them out
        self.fog = _approach(self.fog, 0.0, 0.40, dt)
        self._melt_snow(world, dt)

    # ========================================================== scene: fog ==
    def _scene_fog(self, world: Any, dt: float) -> None:
        """A thick, still grey mist. Nothing dangerous: it just swallows the
        distance and damps the colour. Depth is published as ``fog`` (0..1) and
        drawn as a full-frame wash by the renderer, densest low to the ground."""
        self._approach_env(dt, rate=0.25)          # no rain, no snow
        self._set_wind(dt, 0.10, 0.14, slow=0.08, fast=0.30)   # barely a breath
        self.tint = (206, 210, 214)                # desaturated toward grey
        self.ember_rate = 0.0
        self.water_level = None
        self.fireflies = False
        self.fog = _approach(self.fog, 0.85 * self.intensity, 0.35, dt)
        self._melt_snow(world, dt)

    # ====================================================== scene: eclipse ==
    def _scene_eclipse(self, world: Any, dt: float) -> None:
        """A bright day the sun is taken out of, and then given back.

        Unlike every other scene in this file the eclipse is *timed*, not held:
        it runs a slow ingress, a short totality and an egress off ``scene_t``
        (see ``eclipse_strength``) and is finished inside two and a half minutes,
        after which the slot plays out as ordinary daylight. Deliberately no
        hazards, no accumulation, no panic and no deaths - the colony is meant to
        walk out of this with the same headcount it walked in with. The only
        thing that changes is the light, and the point of the scene is what that
        does to the torches: ``_publish`` turns this channel into
        ``Lighting.ambient_dim``, so the ambient the light composite runs against
        collapses while every flame keeps its full brightness.

        Note ``self.intensity`` is *not* used to gate the strength. The 6 s entry
        envelope every other scene rides is invisible against a 60 s ingress, and
        folding it in would mean the timeline no longer matched ``scene_t`` - the
        one property that keeps this deterministic across a save/load.
        """
        self._approach_env(dt, rate=0.35)          # clear out any lingering rain
        self._set_wind(dt, 0.18, 0.22, slow=0.13, fast=0.44)
        self.ember_rate = 0.0
        self.water_level = None
        self.fog = _approach(self.fog, 0.0, 0.40, dt)
        self._melt_snow(world, dt)

        t1 = self.scene_t
        t0 = max(0.0, t1 - dt)
        self.eclipse = _clamp(eclipse_strength(t1))

        # The world goes cold and steel-coloured as the light goes; back to
        # neutral white as it returns.
        k = self.eclipse
        self.tint = (int(255 - 105 * k), int(255 - 87 * k), int(255 - 55 * k))
        # False dusk: the fireflies are fooled, which is a real and very cheap
        # detail - they come out at totality and go back in when the sun does.
        self.fireflies = k > 0.75

        # Beats are fired on *threshold crossings* rather than from a stored
        # "have I said this yet" flag, so they need no extra field in __init__,
        # no clearing in _reset_scene_state and no round-trip in to_dict - and a
        # save loaded mid-eclipse silently skips the lines it already printed
        # instead of repeating them.
        hold_end = ECLIPSE_INGRESS + ECLIPSE_TOTALITY
        if t0 < 2.0 <= t1:
            _chronicle(world, "A shadow begins to eat the sun.")
        if t0 < ECLIPSE_INGRESS <= t1:
            _chronicle(world, "The sun goes dark.")
        if t0 < hold_end <= t1:
            _chronicle(world, "A sliver of light returns.")
        if t0 < hold_end + ECLIPSE_EGRESS <= t1:
            _chronicle(world, "The sun is whole again.")
            # It has happened; there is nothing left for this scene to do. If
            # auto-advance is on the weather may move along early, and if it is
            # not, the 600 s rotation collects it in its own time.
            self._want_advance = True

    # =================================================== scene: earthquake ==
    def _scene_earthquake(self, world: Any, dt: float) -> None:
        """Long quiet, then a few violent seconds. The ground is the hazard.

        The shape of the scene is a metronome with a very long gap: a tremor
        arrives, swells, tears something and rolls off, and then nothing happens
        for half a minute. Everything that makes it dangerous is bounded at the
        point it is created rather than moderated afterwards - the scars have a
        capped depth *and* a capped steepness (see QUAKE_FISSURE_*), buildings
        have a damage floor they cannot be taken below, and boulders are the only
        thing let loose that can move on its own.

        The reason it is written that way: this map's fall system is lethal and
        the panic flag makes agents walk off ledges they would otherwise refuse,
        so a terrain-deforming scene that tunes for "usually survivable" ends up
        killing by accident on the one seed that stacks two scars. A scene that
        *cannot* carve a lethal drop needs no luck.
        """
        # Dust, not weather: the ash channel is borrowed purely as a particle
        # budget (no _accumulate call, so nothing settles on the ground). It
        # kicks up fast when the ground moves and hangs in the air afterwards,
        # which is why the two rates are so far apart.
        shaking = self.quake_t > 0.0
        dust = 0.60 * _clamp(self.tremor_mag / QUAKE_MAG_MAX) if shaking else 0.0
        self._approach_env(dt, ash=dust, rate=1.2 if shaking else 0.16)
        self._set_wind(dt, 0.16, 0.20, slow=0.11, fast=0.37)
        self.tint = (232, 208, 172)
        self.ember_rate = 0.0
        self.water_level = None
        self.fireflies = False
        self.fog = _approach(self.fog, 0.0, 0.40, dt)
        self._melt_snow(world, dt)
        # No _expire_panic call, because nothing here raises the flag. The
        # scenes that do have to lower it again; a scene that does not would
        # only be draining somebody else's fright (an animal's, the saucer's)
        # at double rate for no reason - tick() already runs _tick_panic.

        if shaking:
            self._tremor_shake()
        else:
            self.tremor_mag = 0.0
            self.next_tremor -= dt * (0.4 + 0.6 * self.intensity)
            if self.next_tremor <= 0.0:
                self._start_tremor(world)
        self._open_fissures(world, dt)

    def _start_tremor(self, world: Any) -> None:
        """Kick off one tremor and apply everything it does in the same instant.

        The damage is one-shot at the leading edge rather than spread over the
        shaking, because that is what a shock *is*: the buildings crack when it
        arrives, and the rest of the tremor is the world ringing afterwards.
        """
        mag = self._rng.uniform(QUAKE_MAG_MIN, QUAKE_MAG_MAX)
        self.tremor_mag = mag
        self.tremor_len = self._rng.uniform(QUAKE_TREMOR_MIN, QUAKE_TREMOR_MAX)
        self.next_tremor = self._rng.uniform(QUAKE_REST_MIN, QUAKE_REST_MAX)
        self.trigger_quake(world, self.tremor_len, mag)

        # Epicentres favour where people are, for the same reason the mudslide
        # aims: a scene whose every event lands on empty hillside is a screen
        # saver. It is safe to aim here in a way it is not there, because
        # nothing this method does can kill - the worst outcome is a cracked hut
        # and a scar somebody has to walk round.
        # STAGE for the unaimed 30%, WORLD for the clamp on the aimed 70%: the
        # first is "somewhere in shot", the second must not drag an epicentre
        # off the person it was just aimed at.
        epi = self._rng.uniform(*_stage_inset(world, 60.0))
        crowd = _agents(world)
        if crowd and self._rng.random() < 0.7:
            who = crowd[self._rng.randrange(len(crowd))]
            epi = _clamp(_fnum(getattr(who, "x", epi), epi)
                         + self._rng.uniform(-120.0, 120.0), 60.0, WORLD_W - 60.0)

        if mag < QUAKE_BIG_MAG:
            self._shake_props(world, epi, mag * 0.45)
            return

        opened = self._open_fissure(world, epi)
        cracked = self._shake_structures(world, epi, mag)
        self._shake_props(world, epi, mag)
        # One line per tremor would flush the 10-entry chronicle in three
        # minutes, so only the big ones speak, and they say which kind of big.
        if opened:
            _chronicle(world, "The ground tears open with a crack like thunder.")
        elif cracked:
            _chronicle(world, "A violent tremor shakes the settlement.")
        else:
            _chronicle(world, "The whole valley shudders.")

    def _tremor_shake(self) -> None:
        """The shake envelope, per tick, while the ground is moving. That is the
        whole of it: everything this scene does to people goes through
        hazards(), and everything it does to the world happened at the shock."""
        # quake_t counts *down*, so this runs 1 -> 0 across the tremor. A flat
        # amplitude reads as a stuck renderer; a half-sine arrives, peaks and
        # rolls off the way a real shock does. The 0.25 floor keeps the ground
        # alive at both ends instead of fading to a dead stop.
        left = _clamp(self.quake_t / max(0.2, self.tremor_len))
        env = 0.25 + 0.75 * math.sin(math.pi * _clamp(1.0 - left))
        self.add_shake(self.tremor_mag * env)

    def _open_fissure(self, world: Any, x: float) -> bool:
        """Register a new scar near *x*. Returns True if one actually opened.

        Nothing is deformed here - the scar is recorded with ``depth`` 0 and a
        capped ``target``, and ``_open_fissures`` tears it down to that over
        QUAKE_FISSURE_OPEN_SEC. Splitting it that way is what lets the ground be
        seen opening rather than blinking into its final shape.
        """
        if len(self.fissures) >= QUAKE_FISSURE_MAX:
            return False
        half = self._rng.uniform(QUAKE_FISSURE_HALF_MIN, QUAKE_FISSURE_HALF_MAX)
        # WORLD: *x* arrives already sited (stage-drawn or aimed at a person);
        # all this clamp does is keep the scar's full width on the land.
        cx = _clamp(x + self._rng.uniform(-90.0, 90.0),
                    half + 10.0, WORLD_W - half - 10.0)
        for f in self.fissures:
            if abs(_fnum(f.get("x"), -1e9) - cx) < QUAKE_FISSURE_GAP:
                return False                    # would merge into a trench
        if self._already_hollow(world, cx, half):
            return False                        # do not dig an old hole deeper
        self.fissures.append({
            "x": float(cx), "half": float(half), "depth": 0.0,
            "target": float(min(QUAKE_FISSURE_DEPTH_MAX, half * QUAKE_FISSURE_ASPECT)),
            "y": float(_ground_y(world, cx)),
            "seed": int(self._rng.randrange(1 << 20)),
        })
        return True

    @staticmethod
    def _already_hollow(world: Any, cx: float, half: float) -> bool:
        """True if the ground at *cx* already sits in a dip of its own.

        Measured against the two shoulders just outside the span rather than
        against any absolute height, so it is slope-neutral: an honest hillside
        passes, an old crater or a scar from a previous run of this scene does
        not. Without it a long session pockmarks the same low ground over and
        over until the cap that bounds a *single* fissure stops meaning anything.
        """
        gy = _ground_y(world, cx)
        shoulder = 0.5 * (_ground_y(world, cx - half * 2.5)
                          + _ground_y(world, cx + half * 2.5))
        return (gy - shoulder) > QUAKE_FISSURE_SINK

    def _open_fissures(self, world: Any, dt: float) -> None:
        """Deepen every scar that has not finished opening. Idempotent once they
        have: a fissure at its target is never deformed again, which is what
        keeps a 600 s scene from grinding a 26 px notch into a canyon."""
        if not self.fissures:
            return
        rate = QUAKE_FISSURE_DEPTH_MAX / max(0.2, QUAKE_FISSURE_OPEN_SEC)
        for f in self.fissures:
            depth = _fnum(f.get("depth"), 0.0)
            target = min(QUAKE_FISSURE_DEPTH_MAX, _fnum(f.get("target"), 0.0))
            if depth >= target - 1e-3:
                continue
            step = min(target - depth, rate * dt)
            cx = _fnum(f.get("x"), 0.0)
            half = max(4.0, _fnum(f.get("half"), 8.0))
            # +dy digs. The 'smooth' blend Terrain.deform defaults to is doing
            # the real safety work here: it ramps the walls over half the span
            # instead of cutting a vertical step an agent can fall down.
            _deform(world, cx - half, cx + half, step)
            _paint(world, cx - half * 0.6, cx + half * 0.6, MAT_STONE)
            f["depth"] = depth + step

    def _shake_structures(self, world: Any, x: float, mag: float) -> int:
        """Crack the buildings near the epicentre. Returns how many took damage.

        Never demolishes: QUAKE_STRUCT_FLOOR is the share of max hp below which
        this refuses to push a structure, so an unlucky run of tremors leaves a
        settlement of cracked huts to repair rather than a field of rubble.
        """
        span = max(1.0, QUAKE_MAG_MAX - QUAKE_BIG_MAG)
        scale = _clamp((mag - QUAKE_BIG_MAG) / span, 0.35, 1.0)
        hurt = 0
        for st in _iter(world, ("structures",)):
            if not getattr(st, "built", False) or getattr(st, "is_ruined", False):
                continue
            d = abs(_fnum(getattr(st, "x", 1e9), 1e9) - x)
            if d > QUAKE_DAMAGE_R:
                continue
            max_hp = max(1.0, _fnum(getattr(st, "max_hp", 60.0), 60.0))
            hp = _fnum(getattr(st, "hp", max_hp), max_hp)
            amount = max_hp * QUAKE_STRUCT_FRAC * scale * (1.0 - d / QUAKE_DAMAGE_R)
            amount = min(amount, hp - max_hp * QUAKE_STRUCT_FLOOR)
            if amount < 0.5:
                continue
            fn = getattr(st, "damage", None)
            if not callable(fn):
                continue
            try:
                fn(float(amount), "earthquake")
            except TypeError:
                try:
                    fn(float(amount))
                except Exception:
                    continue
            except Exception:
                continue
            hurt += 1
        return hurt

    def _shake_props(self, world: Any, x: float, mag: float) -> None:
        """Knock the loose furniture about: boulders roll, everything else whips.

        Boulders are the one thing here that keeps moving after the tremor -
        props.py already runs them downhill and lets them smash what they hit -
        so they are rationed hard (QUAKE_BOULDER_MAX per tremor) and only the
        ones genuinely near the epicentre go.
        """
        strength = _clamp(mag / QUAKE_MAG_MAX)
        rolled = 0
        for p in _iter(world, ("props",)):
            if not getattr(p, "alive", True):
                continue
            px = _fnum(getattr(p, "x", 1e9), 1e9)
            d = abs(px - x)
            if d > QUAKE_BOULDER_R:
                continue
            near = 1.0 - d / QUAKE_BOULDER_R
            if str(getattr(p, "kind", "")) == "boulder":
                if rolled >= QUAKE_BOULDER_MAX or self._rng.random() > near * strength:
                    continue
                push = self._rng.uniform(45.0, 110.0) * strength
                if _nudge_boulder(p, push if px >= x else -push):
                    rolled += 1
                continue
            # Trees and bushes just thrash. props.tick bleeds ``sway`` off on
            # its own, so this needs no cleanup and cannot leak into the next
            # scene the way a deformation would.
            state = getattr(p, "state", None)
            if isinstance(state, dict):
                try:
                    state["sway"] = min(1.0, _fnum(state.get("sway"), 0.0)
                                        + 0.7 * near * strength)
                except Exception:
                    continue

    # ===================================================== scene: wildfire ==
    def _scene_wildfire(self, world: Any, dt: float) -> None:
        self._approach_env(dt, rate=0.6)
        self._set_wind(dt, 0.55, 0.7, slow=0.33, fast=1.31)
        self.tint = (255, 176, 118)
        self.ember_rate = 0.55 + 0.45 * self.intensity
        burning = self._burning_props(world)
        self.next_ignite -= dt
        if not burning and self.next_ignite <= 0.0:
            self.next_ignite = self._rng.uniform(FIRE_RELIGHT_MIN, FIRE_RELIGHT_MAX)
            x = self._rng.uniform(*_stage_inset(world, 40.0))
            # The reach keeps its old NUMBER (1600 px) and loses its old
            # meaning. It used to be spelled RENDER_W and read "search the whole
            # map"; written as the stage width it reads "search anywhere in
            # shot", which is what the contrast with the drought's short 240 px
            # reach was always about. Taking it to WORLD_W instead would let a
            # relight pick a tree 3 km away and burn it where nobody is looking,
            # which is the same scene as no wildfire at all.
            p = _nearest_flammable(world, x, STAGE_HALF * 2.0)
            if p is not None and _ignite(world, p):
                _chronicle(world, "A wildfire catches in the dry brush.")
        if burning:
            self._fire_harms_agents(world, dt, burning)
            if self._rng.random() < dt * FIRE_SPREAD_RATE * (0.3 + 0.7 * abs(self.wind)):
                lead = burning[self._rng.randrange(len(burning))]
                fx = _fnum(getattr(lead, "x", 0.0))
                nxt = _nearest_flammable(world, fx + 40.0 * (1 if self.wind >= 0 else -1),
                                         FIRE_SPREAD_REACH)
                if nxt is not None:
                    _ignite(world, nxt)

    @staticmethod
    def _is_alight(obj: Any) -> bool:
        """Props spell it ``burning``; Structure spells it ``is_burning``.

        Only checking ``burning`` meant a hut fully ablaze was not fire as far
        as this module was concerned - it neither burned anyone standing in it
        nor spread. A ruined structure has stopped burning, so skip those.
        """
        if getattr(obj, "burning", False):
            return True
        return bool(getattr(obj, "is_burning", False)) and not getattr(obj, "is_ruined", False)

    def _burning_props(self, world: Any) -> list[Any]:
        """Everything currently alight, prop or structure, in any scene."""
        return [p for p in _props(world) if self._is_alight(p)]

    def _fire_harms_agents(self, world: Any, dt: float, burning: list[Any],
                           extra: list[tuple[float, float]] | None = None,
                           phrase: str = "burned in the wildfire.",
                           panic: bool = True, shelters: bool = False,
                           lethal_sec: float = BURN_DEATH_SEC) -> None:
        """Burn anyone standing in the flames; scare anyone merely near them.

        Contact is measured as a *horizontal* reach with a loose vertical sanity
        check. The old 2D ``hypot`` against the prop's anchor y meant that on a
        slope - which is most of this map - an agent sharing a tree's column was
        already 20-30px "away" and never accumulated a single burn tick, so
        ``BURN_DEATH_SEC`` was unreachable and feature 30 never fired.

        *extra* is a list of bare ``(x, y)`` hot spots with no object behind
        them, and *phrase* is what the chronicle says when one of them kills.
        Together they are what lets the volcano's lava front reuse this path
        rather than grow a second damage model beside it: a flow is sampled into
        a row of points (see ``_lava_points``) and burns at exactly the rate,
        reach and forgiveness a wildfire does.

        *panic* exists for the same caller and is the difference between a
        dangerous scene and a lethal one. A wildfire is a handful of *points*, so
        the FLEE_DIST ring around it is small and the reckless flag it raises
        costs a life now and then. A lava front is a continuous *line* up to
        300 px long, so the same ring is a band across a fifth of the map with
        the whole colony inside it, panicking every tick - and a panicking
        stickman is allowed over ledges a calm one refuses. Measured over four
        seeds x 600 s of eruption: falls came out at 5, 22, 4 and 17 against a
        clear-sky control's 1, 5, 1 and 6, with only 14 lava deaths between them.
        The lava was killing three people by cliff for every one it burned. It
        passes ``panic=False`` and leans on ``hazards`` instead, which walks
        people out of the way without making them careless - the lesson
        SCENE_EARTHQUAKE and _warn_slide both wrote down before this.

        *shelters* says a roof counts, and exists because a sleeper cannot save
        himself: behaviour explicitly refuses to interrupt Sleep for anything,
        so an agent in a hut the flow reaches would burn in his bed with no
        decision available to him - which is the one death this scene is not
        allowed to hand out. The lava damages the *building* instead (see
        ``_lava_consumes``); its collapse evicts the occupants onto the ground,
        awake, and from there they burn like everybody else. Off by default,
        because a wildfire burning a hut down around its sleepers is exactly
        what a wildfire is for.

        *lethal_sec* is the contact time before it kills. Molten rock is not a
        campfire and the two should not share a fuse: BURN_DEATH_SEC is tuned so
        that brushing past a burning tree is survivable, which is right for a
        tree and absurd for a lava flow.
        """
        fires = [(_fnum(getattr(p, "x", 1e9), 1e9), _fnum(getattr(p, "y", 1e9), 1e9))
                 for p in burning]
        if extra:
            fires.extend(extra)
        if not fires:
            return
        for ag in _agents(world):
            if shelters and getattr(ag, "inside", None) is not None:
                continue                        # under a roof; see the docstring
            aid = _aid(ag)
            ax = _fnum(getattr(ag, "x", 1e9), 1e9)
            ay = _fnum(getattr(ag, "y", 1e9), 1e9)
            fx, fy = min(fires, key=lambda f: abs(ax - f[0]))
            gap = abs(ax - fx)
            if panic and gap < FLEE_DIST:
                away = ax + (140.0 if ax >= fx else -140.0)
                _panic(world, ag, away, BURN_PANIC, "fire")
            if aid is None:
                continue
            in_flames = gap <= BURN_TOUCH_DIST and abs(ay - fy) <= BURN_TOUCH_HEIGHT
            if in_flames:
                burnt = self._burning.get(aid, 0.0) + dt
                self._burning[aid] = burnt
                try:
                    ag.warmth = _clamp(_fnum(getattr(ag, "warmth", 0.5), 0.5) - dt * 0.4)
                except Exception:
                    pass
                try:
                    ag.morale = _clamp(_fnum(getattr(ag, "morale", 0.5), 0.5) - dt * 0.25)
                except Exception:
                    pass
                if burnt >= lethal_sec:
                    self._burning.pop(aid, None)
                    if _kill(world, ag, "fire"):
                        _chronicle(world, f"{getattr(ag, 'name', 'A stickman')} "
                                          f"{phrase}")
            elif aid in self._burning:
                # Getting clear is survivable: the timer bleeds off faster than
                # it filled, so anyone who runs within a second or two lives.
                self._burning[aid] = max(0.0, self._burning[aid] - dt * BURN_COOL_RATE)
                if self._burning[aid] <= 0.0:
                    self._burning.pop(aid, None)
        if len(self._burning) > 64:
            # Someone who was mid-burn died of something else; drop the orphans
            # rather than let the dict grow over an all-night run.
            live = {i for i in (_aid(a) for a in _agents(world)) if i is not None}
            self._burning = {k: v for k, v in self._burning.items() if k in live}

    # ===================================================== scene: mudslide ==
    def _scene_mudslide(self, world: Any, dt: float) -> None:
        self._approach_env(dt, rain=0.55, rate=0.4)
        self._set_wind(dt, 0.4, 0.5)
        self.tint = (226, 214, 198)
        self.ember_rate = 0.0
        self._expire_panic(world, dt)
        if self.slide_phase == "idle":
            self._pick_slide_span(world)
            self.slide_phase = "warn"
            self.slide_t = 0.0
            self._buried.clear()
            _chronicle(world, "The hillside groans; loose earth begins to shift.")
        elif self.slide_phase == "warn":
            self.slide_t += dt
            self.rumble = _clamp(self.slide_t / MUDSLIDE_WARN)
            self.add_shake(1.4 * self.rumble)
            self._warn_slide(world)
            if self.slide_t >= MUDSLIDE_WARN:
                self.slide_phase = "slide"
                self.slide_t = 0.0
                self.trigger_quake(world, 1.6, 5.0)
        elif self.slide_phase == "slide":
            self.slide_t += dt
            self.rumble = 1.0
            self.add_shake(5.0)
            self._slide_step(world, dt)
            if self.slide_t >= MUDSLIDE_SLIDE:
                self.slide_phase = "settle"
                self.slide_t = 0.0
                self._buried.clear()
                _paint(world, self.slide_x0, self.slide_x1, MAT_MUD)
                _chronicle(world, "A mudslide reshapes the slope.")
        elif self.slide_phase == "settle":
            self.slide_t += dt
            self.rumble = max(0.0, 1.0 - self.slide_t / MUDSLIDE_SETTLE)
            if self.slide_t >= MUDSLIDE_SETTLE:
                self.slide_phase = "done"
                self.slide_t = 0.0
                self._want_advance = True
        else:
            # "done" doubles as the cooldown. It used to be terminal: one slide
            # and then an inert hillside for as long as the scene ran, because
            # auto_scene_change is off by default so nothing ever reset the
            # phase. Re-arm on a new span so the slope keeps failing.
            self.slide_t += dt
            if self.slide_t >= MUDSLIDE_REST:
                self.slide_phase = "idle"
                self.slide_t = 0.0

    def _pick_slide_span(self, world: Any) -> None:
        width = self._rng.uniform(MUDSLIDE_SPAN_MIN, MUDSLIDE_SPAN_MAX)
        h = _height(world)
        lo, hi = _stage(world)
        x0 = self._rng.uniform(lo, max(lo + 1.0, hi - width))
        crowd = _agents(world)
        if crowd and self._rng.random() < MUDSLIDE_AIM_CHANCE:
            # Always sliding the steepest face means mostly sliding empty
            # hillside: whole scenes went by burying nobody. Half the time the
            # slope that goes is one somebody is standing on. They still get
            # the full MUDSLIDE_WARN of rumble to walk off it.
            who = crowd[self._rng.randrange(len(crowd))]
            cx = _fnum(getattr(who, "x", None), (lo + hi) * 0.5)
            cx += self._rng.uniform(-width * 0.25, width * 0.25)
            # WORLD clamps: the span is already anchored on a person, so these
            # only have to keep it on the land.
            self.slide_x0 = float(_clamp(cx - width * 0.5, 0.0,
                                         max(0.0, WORLD_W - width)))
            self.slide_x1 = float(min(WORLD_W, self.slide_x0 + width))
            return
        if h is not None and h.size > 16:
            # Pure argmax picks the same face every cycle, so a long scene slid
            # the one slope over and over. Wander off it a little. Drawn here
            # rather than after the scan so the seeded stream does not depend on
            # whether the stage window turned out wide enough to scan.
            jitter = int(self._rng.uniform(-width, width))
            # STAGE-WINDOWED SCAN. This used to run over the whole heightmap,
            # because the whole heightmap was the stage. Left alone on a 6400 px
            # map it finds the single steepest face in the *world*, which is off
            # camera roughly three times in four: MUDSLIDE_WARN seconds of
            # rumble, a quake, a chronicle line, and a buried hillside nobody
            # ever saw. The aimed branch above escapes this because it anchors
            # on a person; the unaimed 1 - MUDSLIDE_AIM_CHANCE does not.
            # Slicing also drops the convolve from 6400 columns to ~1600.
            a = int(_clamp(lo, 0.0, float(h.size - 2)))
            b = int(_clamp(hi, float(a + 2), float(h.size)))
            win_h = h[a:b]
            if win_h.size > 16:
                grad = np.gradient(win_h.astype(np.float32))
                win = max(8, int(width) // 8)
                kern = np.ones(win, dtype=np.float32) / float(win)
                smooth = np.convolve(np.abs(grad), kern, mode="same")
                j0 = int(min(width * 0.5, float(smooth.size - 1)))
                j1 = max(j0 + 1, int(smooth.size - width * 0.5))
                centre = a + j0 + int(np.argmax(smooth[j0:j1])) + jitter
                x0 = _clamp(centre - width * 0.5, 0.0, max(0.0, WORLD_W - width))
        self.slide_x0 = float(x0)
        self.slide_x1 = float(min(WORLD_W, x0 + width))

    def _warn_slide(self, world: Any) -> None:
        """Rumble phase: get anyone standing *on* the span moving.

        Only agents actually on the doomed ground are panicked. The flag is
        expensive - actions.py lets a panicking agent step off a drop it would
        normally refuse - so scaring the whole neighbourhood traded a few
        burials for a lot of broken necks. Everyone else is served by
        ``hazards()``, which makes them flee without making them reckless.
        """
        x0, x1 = self.slide_x0, self.slide_x1
        if x1 - x0 < 8.0:
            return
        if self.slide_t < MUDSLIDE_WARN * 0.45:
            return                      # the ground only starts to go late on
        mid = (x0 + x1) * 0.5
        for ag in _agents(world):
            ax = _fnum(getattr(ag, "x", 1e9), 1e9)
            if x0 - 20.0 <= ax <= x1 + 20.0:
                away = (x0 - MUDSLIDE_FLEE_X) if ax < mid else (x1 + MUDSLIDE_FLEE_X)
                _panic(world, ag, away, MUDSLIDE_PANIC, "mudslide")

    def _slide_step(self, world: Any, dt: float) -> None:
        x0, x1 = self.slide_x0, self.slide_x1
        if x1 - x0 < 8.0:
            return
        # Terrain.deform(x0:int, x1:int, dy:float): +dy digs, -dy heaps, and the
        # default 'smooth' blend tapers both ends, so scour and toe meet without
        # a knife-edge step (a hard step here just made agents fall to death).
        mid = x0 + (x1 - x0) * 0.5
        amount = MUDSLIDE_DROP * dt / max(0.1, MUDSLIDE_SLIDE)
        # Earth moves *downhill*. Splitting the span down the middle and always
        # scouring the left half dug a pit beside a mound whenever the slope ran
        # the other way: local relief grew every cycle and the map sprouted
        # fresh cliffs to fall off. Scour whichever half is higher (smaller y).
        h = _height(world)
        high_first = True
        if h is not None:
            a, b, c = int(max(0.0, x0)), int(mid), int(min(float(h.size), x1))
            if b > a and c > b:
                high_first = float(h[a:b].mean()) <= float(h[b:c].mean())
        if high_first:
            _deform(world, x0, mid, amount)          # scour the upper slope
            _deform(world, mid, x1, -amount * 0.75)  # pile it at the toe
        else:
            _deform(world, mid, x1, amount)
            _deform(world, x0, mid, -amount * 0.75)
        _paint(world, x0, x1, MAT_MUD)
        for ag in _agents(world):
            ax = _fnum(getattr(ag, "x", 1e9), 1e9)
            aid = _aid(ag)
            if x0 <= ax <= x1 and _fnum(getattr(ag, "y", 0.0)) >= _ground_y(world, ax) - 30.0:
                # Inside the moving span and on the ground (not up a tower).
                # A short grace period, so the 4s of rumble is a real warning:
                # anyone still standing here when the earth arrives goes under.
                if aid is None:
                    continue
                under = self._buried.get(aid, 0.0) + dt
                self._buried[aid] = under
                _panic(world, ag,
                       (x0 - MUDSLIDE_FLEE_X) if ax < mid else (x1 + MUDSLIDE_FLEE_X),
                       MUDSLIDE_PANIC, "mudslide")
                if under >= MUDSLIDE_BURY_SEC:
                    self._buried.pop(aid, None)
                    if _kill(world, ag, "mudslide"):
                        _chronicle(world, f"{getattr(ag, 'name', 'A stickman')} "
                                          f"was buried by the mudslide.")
                continue
            if aid is not None and aid in self._buried:
                self._buried.pop(aid, None)      # got clear in time

    # ===================================================== scene: blizzard ==
    def _scene_blizzard(self, world: Any, dt: float) -> None:
        self._approach_env(dt, snow=1.0, rate=0.35)
        self._set_wind(dt, 0.95, 0.9, slow=0.29, fast=1.51)
        self.tint = (214, 226, 244)
        self.ember_rate = 0.0
        self.snow_depth = self._accumulate(world, dt, MAT_SNOW, self.snow_depth,
                                           SNOW_MAX_DEPTH, SNOW_RATE, True)
        chill = self.exposure_rate() * dt
        if chill > 0.0:
            for ag in _agents(world):
                try:
                    ag.warmth = _clamp(_fnum(getattr(ag, "warmth", 0.0), 0.0) + chill)
                except Exception:
                    break

    # ==================================================== scene: sandstorm ==
    def _scene_sandstorm(self, world: Any, dt: float) -> None:
        """The arid sibling of the blizzard: no visibility, hard driving wind,
        and sand piling up against everything.

        Structurally it is the blizzard - the same ``_accumulate`` drift, with
        MAT_SAND instead of MAT_SNOW - crossed with the fog scene's haze wash,
        now carrying a colour so this one comes out tan. What is deliberately
        *not* borrowed is the blizzard's lethality. A blizzard kills through the
        ``warmth`` need, which is a real clock with a real death at the end of
        it; this scene has no equivalent, and it should not grow one. It is an
        atmosphere piece: the cost of a sandstorm is that the colony spends ten
        minutes working blind, at reduced light, with the drift swallowing the
        ground - not a pile of bodies.

        Nothing here sets ``agent.panic`` and nothing publishes a hazard, both
        for the reason SCENE_EARTHQUAKE writes out at length: the panic flag
        lets an agent walk off drops it would otherwise refuse, and this map
        kills by falling. A hazard would be no better - the storm is the whole
        map, so there is nowhere for ``hazards_of`` to send anybody.
        """
        # No precipitation channel at all: the grit is airborne, not falling, so
        # rain/snow/ash all bleed to zero and the renderer draws the blowing
        # particles off the wind and the haze instead.
        self._approach_env(dt, rate=0.40)
        self._sand_wind(dt)
        self.tint = (228, 190, 128)
        self.ember_rate = 0.0
        self.water_level = None
        self.fireflies = False          # nothing flies in this

        # Visibility. Same wash the fog scene uses, asked for in tan.
        self.fog = _approach(self.fog, SAND_HAZE * self.intensity, 0.30, dt)
        self.fog_color = HAZE_SAND

        # Melt first, then drift. _melt_snow only repaints columns that are
        # still MAT_SNOW, so a column the sand has already claimed is left
        # alone: the desert buries last week's blizzard rather than the thaw
        # undoing this scene's own work a tick after it lands.
        self._melt_snow(world, dt)
        self.sand_depth = self._accumulate(world, dt, MAT_SAND, self.sand_depth,
                                           SAND_MAX_DEPTH, SAND_RATE, False)

        self._sand_scours(world, dt)
        self._sand_smothers(world, dt)

        # Beats on threshold crossings of scene_t, exactly as the eclipse does
        # them: no "have I said this" flag to add to __init__, to clear in
        # _reset_scene_state and to round-trip, and a save loaded mid-storm
        # silently skips the lines it has already printed instead of repeating
        # them into a 10-entry chronicle.
        t1 = self.scene_t
        t0 = max(0.0, t1 - dt)
        if t0 < 3.0 <= t1:
            _chronicle(world, "The horizon turns brown. Sand is coming.")
        if t0 < 75.0 <= t1:
            _chronicle(world, "Grit hisses against everything that stands.")

    def _sand_wind(self, dt: float) -> None:
        """Drive the wind hard, and drive it *one way*.

        ``_set_wind`` builds its target from ``_noise``, which is signed and
        crosses zero every few seconds - right for weather that merely blows
        about, wrong for a storm front, which reads as a broken desk fan if it
        reverses twice a minute. So the direction is fixed for the whole scene
        and only the strength is modulated.

        The direction comes off ``self.seed`` rather than a draw from ``_rng``.
        Two reasons, both learned elsewhere in this file: a draw would shift the
        seeded stream every other scene rides, and a stored field would need
        clearing and round-tripping. A bit of the seed costs nothing, replays
        identically, and survives a save/load with no new state - at the price
        that one world always has the same prevailing wind, which is a feature.
        """
        sign = 1.0 if (self.seed >> 11) & 1 else -1.0
        # 0.62..1.0 of full strength: it surges and eases, but never lets go.
        surge = 0.62 + 0.38 * (0.5 + 0.5 * self._noise(0.33, 1.61))
        target = sign * _clamp(SAND_WIND * surge) * self.intensity
        self.wind = _approach(self.wind, target, 1.2, dt)
        # Gust rides a faster beat than the wind so the torches gutter *within*
        # a squall rather than only between them.
        g = 0.5 + 0.5 * math.sin(self.t * 2.9 + 0.4)
        self.gust = _clamp(abs(self.wind) * (0.45 + 0.55 * g) * SAND_GUST)

    def _sand_scours(self, world: Any, dt: float) -> None:
        """Sting anyone caught in the open while a squall is up.

        Hard-bounded rather than tuned: the damage taken is capped at whatever
        would bring the agent down to SAND_HEALTH_FLOOR, so the arithmetic makes
        a sandstorm death impossible instead of merely unlikely. That matters
        more than the rate does - the rate only decides how quickly somebody
        reaches a floor they cannot pass.

        Being indoors is complete cover, which is the one behavioural texture
        the scene has: the colony does not need new AI to look like it is
        sheltering, because the villagers who happen to be inside visibly come
        out unmarked.
        """
        if self.gust < SAND_STING_GUST:
            return
        bite = SAND_STING_RATE * dt * self.intensity
        if bite <= 0.0:
            return
        for ag in _agents(world):
            try:
                if getattr(ag, "inside", None) is not None:
                    continue            # under a roof; the grit cannot reach
                hp = _fnum(getattr(ag, "health", 0.0), 0.0)
                take = min(bite, hp - SAND_HEALTH_FLOOR)
                if take <= 0.0:
                    continue
                hurt = getattr(ag, "hurt", None)
                if callable(hurt):
                    # Armour applies, so a hide cloak takes even less than the
                    # cap - which can only ever push health further above it.
                    hurt(take, "sandstorm")
                else:
                    ag.health = hp - take
            except Exception:
                break                   # an unfamiliar agent shape: stop, quietly

    def _sand_smothers(self, world: Any, dt: float) -> None:
        """Sand puts fires out, the way rain does in the storm scene.

        Same shape as ``_rain_douses`` - one prop per successful roll, so a
        hillside of burning trees goes out over a minute rather than all at once
        - and it is the only thing in this scene that helps the colony. A
        wildfire rolling into a sandstorm should end.
        """
        if self.intensity < 0.35:
            return
        if self._rng.random() > dt * 0.30 * self.intensity:
            return
        for p in _props(world):
            if self._is_alight(p):      # structures included - they burn too
                _extinguish(p)
                break

    # ===================================================== scene: heatwave ==
    def _scene_heatwave(self, world: Any, dt: float) -> None:
        """A glaring, motionless drought. The scene that bites the economy.

        Every other disaster in this file is paid for in bodies. This one is
        paid for in *time*: nothing explodes, nobody drowns, and the colony
        simply stops being able to live off the land. The drought multiplier
        (see ``_drought_hook``) throttles every natural recovery rate in
        props.py, so berries stop coming back, crops stop ripening and a felled
        map stops reseeding itself; meanwhile ``_heat_saps`` makes everyone tired
        sooner, which they answer by sleeping, which is more time not gathering.
        The colony has to eat its stockpile, and the interesting question for ten
        minutes is whether it built one.

        Two deliberate softenings keep that from becoming a wipe. Hunger is not
        touched at all - it is one of the two needs that actually kills, and a
        scene that both stops the food growing *and* speeds the clock down to
        starvation would compound into an extinction on any seed that started
        lean. And warmth is pushed the other way: it is sweltering, so nobody is
        cold, day or night, which is a genuine gift on a map where a clear night
        pulls the whole colony off work and around the firepit.

        Nothing here sets ``agent.panic`` directly. The only thing that can is a
        brush fire, which ``_heat_ignites`` allows at a third of the wildfire
        scene's rate and only once the drought is deep.
        """
        wt = _fnum(getattr(world, "world_time", self.t), self.t)
        # Bone dry: every precipitation channel is driven to zero, faster than
        # any other scene does it, because a rain shower bleeding out over the
        # first few seconds of a drought reads as a mistake.
        self._approach_env(dt, rate=0.50)
        # Barely moving air. The stillness is half the look: it is the only
        # scene in the file with no particles, no haze and no wind to watch, so
        # the flat calm is what tells you something is wrong.
        self._set_wind(dt, 0.08, 0.10, slow=0.06, fast=0.23)
        self.tint = (255, 238, 196)
        self.ember_rate = 0.0
        self.water_level = None
        self.fog = _approach(self.fog, 0.0, 0.40, dt)
        self.fireflies = is_night(wt)      # a hot still night is exactly their weather
        # Whatever the last blizzard left goes fast. _melt_snow takes dt as a
        # plain scale, so the multiplier costs nothing and a heatwave landing on
        # a white map strips it inside a minute, which is a better transition
        # than watching snow sit under a white-hot sky for ten of them.
        self._melt_snow(world, dt * HEAT_THAW_MUL)

        # The drought envelope. Chases ``intensity`` (itself a 6 s ramp) but at
        # a 45 s rate, so the economic effect arrives slowly enough that the
        # colony's food curve bends rather than snapping. Everything else in the
        # scene - the shimmer, the fatigue, the ignition gate - hangs off this
        # one number, so they all ramp together.
        self.heat = _approach(self.heat, self.intensity, 1.0 / HEAT_RAMP_SEC, dt)

        self._heat_saps(world, dt)
        self._heat_ignites(world, dt)
        burning = self._burning_props(world)
        if burning:
            # Same contact model the wildfire uses, reached far less often. A
            # fire nobody can be hurt by is scenery, and this scene has exactly
            # one danger in it.
            self._fire_harms_agents(world, dt, burning)

        # Beats on threshold crossings of scene_t, as the eclipse and sandstorm
        # do them: no "have I said this" flag to add to __init__, clear in
        # _reset_scene_state and round-trip, and a save loaded mid-drought skips
        # the lines it has already printed rather than repeating them.
        t1 = self.scene_t
        t0 = max(0.0, t1 - dt)
        if t0 < 4.0 <= t1:
            _chronicle(world, "The wind dies and the light turns white.")
        if t0 < 100.0 <= t1:
            _chronicle(world, "The ground is cracking. Nothing is growing.")
        if t0 < 340.0 <= t1:
            _chronicle(world, "The colony is eating into its stores.")

    def _heat_saps(self, world: Any, dt: float) -> None:
        """Tire everyone out in the sun, and take the cold away from everyone.

        The fatigue is hard-capped rather than tuned, in the shape
        ``_sand_scours`` uses for health: the heat can carry an agent up to
        HEAT_FATIGUE_CAP and no further, so it can never be the thing that pins
        a villager at 1.0 and drops them where they stand. That matters more
        than the rate does - a colony asleep in a field is a colony that has
        stopped gathering, which would turn an economic scene into a starvation
        one by the back door.

        Shade is complete cover, exactly as it is for the sandstorm's grit, and
        for the same reason: it gives the scene a behavioural texture for free.
        The villagers who happen to be indoors come out rested, so the huts
        visibly earn their keep without a line of new AI.
        """
        k = self.heat
        if k <= 0.0:
            return
        tire = HEAT_FATIGUE_RATE * k * dt
        # Negative in this scene: exposure_rate is signed, and the heatwave is
        # the only entry that gives warmth back rather than taking it.
        chill = self.exposure_rate() * dt
        for ag in _agents(world):
            try:
                if tire > 0.0 and getattr(ag, "inside", None) is None:
                    fat = _fnum(getattr(ag, "fatigue", 0.0), 0.0)
                    if fat < HEAT_FATIGUE_CAP:
                        ag.fatigue = min(HEAT_FATIGUE_CAP, fat + tire)
                if chill != 0.0:
                    ag.warmth = _clamp(_fnum(getattr(ag, "warmth", 0.0), 0.0) + chill)
            except Exception:
                break                   # an unfamiliar agent shape: stop, quietly

    def _heat_ignites(self, world: Any, dt: float) -> None:
        """Occasionally set the dry brush alight.

        Reuses ``next_ignite``, which already exists, is already cleared in
        _reset_scene_state and is already round-tripped - so the scene's only
        stochastic element needs no new state. The timer is rolled on every
        attempt whether or not the fire actually starts, which is what keeps the
        draws on ``_rng`` at a fixed cadence: a gate that skipped the draw would
        make the seeded stream depend on the drought depth.

        One fire at a time. props.py runs its own spread, so a single ignition
        is a fire *front*, not a burnt bush, and the wood it eats is wood the
        colony never gets to build with - which is this scene's currency anyway.
        """
        self.next_ignite -= dt
        if self.next_ignite > 0.0:
            return
        self.next_ignite = self._rng.uniform(HEAT_IGNITE_MIN, HEAT_IGNITE_MAX)
        if self.heat < HEAT_IGNITE_HEAT:
            return                      # not dry enough yet; the brush will not take
        if self._burning_props(world):
            return                      # something is already alight; let it run
        x = self._rng.uniform(*_stage_inset(world, 40.0))
        # A short reach, unlike the wildfire's stage-wide search: a drought fire
        # starts where the brush happens to be, not wherever the last flammable
        # thing in shot is standing. HEAT_IGNITE_MIN/MAX is per-second, so the
        # draw above narrows and the interval does not.
        p = _nearest_flammable(world, x, 240.0)
        if p is not None and _ignite(world, p):
            _chronicle(world, "Dry brush catches alight in the heat.")

    def exposure_rate(self) -> float:
        """Extra ``warmth`` (0..1, 1 = freezing) per second from the weather.

        Signed: negative means the weather is *warming* people, which only the
        heatwave does. Already applied by ``tick``; exposed for UI/telemetry so
        entities.py does not need to apply it a second time.
        """
        if self.scene == SCENE_BLIZZARD:
            return 0.012 * self.intensity
        if self.scene == SCENE_NIGHT_STORM:
            return 0.004 * self.rain
        if self.scene == SCENE_HEATWAVE:
            return -HEAT_RELIEF_RATE * self.heat
        return 0.0

    # ======================================================== scene: flood ==
    def _scene_flood(self, world: Any, dt: float) -> None:
        self._approach_env(dt, rain=0.35, rate=0.3)
        self._set_wind(dt, 0.35, 0.4)
        self.tint = (206, 224, 246)
        self.ember_rate = 0.0
        self._expire_panic(world, dt)
        # The envelope runs off its own clock, not scene_t. scene_t only ever
        # grows, so once the single surge had drained (96s in) the scene sat at
        # u=0 with water_level None for the rest of its life - which is what
        # "200s in SCENE_FLOOD and water_level is still None" was measuring.
        self.flood_t += dt
        if self.flood_y0 is None or self.flood_y1 is None:
            self._arm_flood(world)
        t = self.flood_t
        cycle = FLOOD_RISE + FLOOD_HOLD + FLOOD_FALL
        if t < FLOOD_RISE:
            u = _smooth(t / FLOOD_RISE)              # ~40s to full height
        elif t < FLOOD_RISE + FLOOD_HOLD:
            u = 1.0
        elif t < cycle:
            u = 1.0 - _smooth((t - FLOOD_RISE - FLOOD_HOLD) / FLOOD_FALL)
        else:
            u = 0.0
            if self.water_level is not None:
                _chronicle(world, "The floodwater drains away.")
            self._want_advance = True
            if t >= cycle + FLOOD_DRY:              # nobody moved us on: again
                self.flood_t = 0.0
                self.flood_y0 = None
                self.flood_y1 = None
        if u <= 0.0:
            self.water_level = None
            self._submerged.clear()
            return
        self.water_level = self.flood_y0 + (self.flood_y1 - self.flood_y0) * u
        self._flood_effects(world, dt, float(self.water_level))

    def _arm_flood(self, world: Any) -> None:
        """Fix the still-water line for one surge: y0 dry, y1 at full height."""
        h = _height(world)
        if h is not None:
            low = float(np.max(h))                  # y grows downward: lowest ground
            high = float(np.min(h))
        else:
            low, high = RENDER_H * 0.8, RENDER_H * 0.4
        depth = _clamp(FLOOD_DEPTH_FRAC * (low - high), 40.0, 170.0)
        self.flood_y0 = low + 4.0
        self.flood_y1 = low - depth
        self._submerged.clear()
        _chronicle(world, "Water begins to pool in the low ground.")

    def _flood_effects(self, world: Any, dt: float, level: float) -> None:
        for p in _props(world):
            if getattr(p, "burning", False) and _fnum(getattr(p, "y", -1e9), -1e9) > level:
                _extinguish(p)
        h = _height(world)
        dry: np.ndarray | None = None
        if h is not None:
            found = np.flatnonzero(h < np.float32(level - 12.0))
            if found.size:
                dry = found
        for ag in _agents(world):
            ax = _fnum(getattr(ag, "x", 0.0))
            ay = _fnum(getattr(ag, "y", 0.0))
            aid = _aid(ag)
            if ay <= level + DROWN_DEPTH:
                # Dry, or no worse than ankle deep. The clock unwinds faster
                # than it filled: reaching the shore is meant to save you.
                if aid is not None and aid in self._submerged:
                    self._submerged[aid] = max(0.0, self._submerged[aid] - dt * 1.5)
                    if self._submerged[aid] <= 0.0:
                        self._submerged.pop(aid, None)
                continue
            # Under the line. Run for the nearest ground above the water; the
            # drowning timer only pays out for whoever does not make it. The
            # panic flag is what interrupts a job mid-swing - hazards() handles
            # the ones who are merely near the water, without the recklessness.
            _panic(world, ag, self._high_ground(world, ax, dry), FLOOD_PANIC, "flood")
            if aid is None:
                continue
            sub = self._submerged.get(aid, 0.0) + dt
            self._submerged[aid] = sub
            try:
                ag.morale = _clamp(_fnum(getattr(ag, "morale", 0.5), 0.5) - dt * 0.08)
            except Exception:
                pass
            if sub >= DROWN_SEC:
                self._submerged.pop(aid, None)
                if _kill(world, ag, "drown"):
                    _chronicle(world, f"{getattr(ag, 'name', 'A stickman')} "
                                      f"drowned in the floodwater.")

    def _high_ground(self, world: Any, ax: float, dry: np.ndarray | None) -> float:
        """Nearest column standing clear of the water; failing that, the peak."""
        if dry is not None and dry.size:
            j = int(np.argmin(np.abs(dry - int(ax))))
            return float(dry[j])
        h = _height(world)
        if h is not None:
            return float(int(np.argmin(h)))
        # Last resort, and only reachable on a world with no heightmap at all
        # (a stub in a test): "run to the far end". WORLD, because with no
        # terrain there is no high ground to prefer and no reason to stop at the
        # stage edge - and in the real game this line never executes.
        return 20.0 if ax > WORLD_W * 0.5 else WORLD_W - 20.0

    # ======================================================= scene: meteor ==
    def _scene_meteor(self, world: Any, dt: float) -> None:
        self._approach_env(dt, rate=0.4)
        self._set_wind(dt, 0.3, 0.35)
        self.tint = (226, 226, 255)
        self.ember_rate = 0.25
        self._expire_panic(world, dt)
        wt = _fnum(getattr(world, "world_time", self.t), self.t)
        self.fireflies = is_night(wt) and daylight_factor(wt) < 0.1
        self.next_meteor -= dt * (0.5 + 0.5 * self.intensity)
        if self.next_meteor <= 0.0:
            self.next_meteor = self._rng.uniform(METEOR_MIN, METEOR_MAX)
            self._spawn_meteor(world)
        self._advance_meteors(world, dt)

    def _spawn_meteor(self, world: Any) -> None:
        impact = self._rng.random() < METEOR_IMPACT_CHANCE
        # STAGE. METEOR_MIN/MAX is 1.6-5 s - the densest rate in the file - so
        # this is where a world-wide draw would hurt most: four times the rocks
        # simulated, the same one or two on screen.
        x1 = self._rng.uniform(*_stage_inset(world, 40.0))
        drift = self._rng.uniform(180.0, 420.0) * (1 if self._rng.random() < 0.5 else -1)
        x0 = x1 - drift
        y0 = -self._rng.uniform(40.0, 220.0)
        y1 = _ground_y(world, x1) if impact else self._rng.uniform(0.25, 0.5) * RENDER_H
        self.meteors.append({
            "x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1),
            "t": 0.0, "ttl": float(self._rng.uniform(0.7, 1.3)),
            "seed": int(self._rng.randrange(1 << 20)),
            "impact": bool(impact), "done": False,
        })
        if len(self.meteors) > 12:
            del self.meteors[0]

    def _advance_meteors(self, world: Any, dt: float) -> None:
        if not self.meteors:
            return
        keep: list[dict[str, Any]] = []
        for m in self.meteors:
            m["t"] = _fnum(m.get("t"), 0.0) + dt
            ttl = _fnum(m.get("ttl"), 1.0)
            if m["t"] >= ttl and not m.get("done"):
                m["done"] = True
                if m.get("impact"):
                    self._meteor_impact(world, _fnum(m.get("x1")), _fnum(m.get("y1")))
            if m["t"] < ttl + METEOR_TTL:
                keep.append(m)
        self.meteors = keep

    def _meteor_impact(self, world: Any, x: float, y: float) -> None:
        r = self._rng.uniform(16.0, 34.0)
        # Terrain.crater is (cx:int, radius:int, depth:float) - column based,
        # no y at all. _crater() adapts to that; the y here is only used to
        # decide who was standing close enough to be under it.
        _crater(world, x, y, r)
        _paint(world, x - r * 0.9, x + r * 0.9, MAT_ASH)
        _flash(world, 0.85, 0.35, METEOR_COLOR)
        self.add_shake(9.0)
        self.trigger_quake(world, 1.2, 4.0)
        p = _nearest_flammable(world, x, r * METEOR_IGNITE_REACH)
        if p is not None:
            _ignite(world, p)
        blast = max(9.0, r * METEOR_BLAST)
        killed = 0
        for ag in _agents(world):
            ax = _fnum(getattr(ag, "x", 1e9), 1e9)
            ay = _fnum(getattr(ag, "y", 1e9), 1e9)
            d = math.hypot(ax - x, ay - y)
            if d <= blast:
                if _kill(world, ag, "meteor"):
                    killed += 1
                    _chronicle(world, f"{getattr(ag, 'name', 'A stickman')} "
                                      f"was crushed by a falling star.")
            elif d < r * 3.0:
                _panic(world, ag, ax + (150.0 if ax >= x else -150.0), METEOR_PANIC, "meteor")
        if killed == 0 and self._rng.random() < 0.3:
            # One line per rock would drown the chronicle - impacts are common,
            # a stickman dying under one is not.
            _chronicle(world, "A meteor slams into the hillside.")

    # ================================================== hazards / panic aid ==
    def hazards(self) -> list[dict[str, Any]]:
        """Live danger zones, read by ``actions.hazards_of`` (behavior.py).

        Nothing used to publish these, so ``_danger_for`` always came back
        empty: FleeFrom scored 0 and no stickman ever ran from a flood, a slope
        about to go, or an incoming rock. ``water_y`` is the shape behavior.py
        expects for a waterline (it makes the flee uphill).
        """
        out: list[dict[str, Any]] = []
        level = self.water_level
        if level is not None:
            # WORLD, both of them. A flood is a waterline, not a place: every
            # column below `level` is under it, so the hazard has to cover the
            # map or agents outside the disc simply would not flee. Narrowing
            # this to the stage would be the one hazard bug that kills people -
            # a colonist who walks off-stage into deep water gets no FleeFrom
            # score at all and drowns without ever trying to move.
            out.append({"kind": "flood", "x": WORLD_W * 0.5, "y": float(level),
                        "water_y": float(level), "radius": float(WORLD_W)})
        if self.scene == SCENE_MUDSLIDE and self.slide_phase in ("warn", "slide"):
            x0, x1 = self.slide_x0, self.slide_x1
            if x1 - x0 >= 8.0:
                out.append({"kind": "mudslide", "x": (x0 + x1) * 0.5,
                            "y": RENDER_H * 0.6, "radius": (x1 - x0) * 0.5 + 70.0})
        for m in self.meteors:
            if not isinstance(m, dict) or not m.get("impact") or m.get("done"):
                continue
            out.append({"kind": "meteor", "x": _fnum(m.get("x1"), WORLD_W * 0.5),
                        "y": _fnum(m.get("y1"), RENDER_H * 0.7),
                        "radius": METEOR_WARN_R})
        # A scar is only frightening while the ground is actually moving. Once
        # the tremor passes it is just difficult terrain, and villagers have to
        # be allowed to walk back over it or half the map goes permanently
        # out of bounds for the rest of the scene.
        if self.quake_t > 0.0:
            for f in self.fissures:
                if not isinstance(f, dict):
                    continue
                out.append({"kind": "quake", "x": _fnum(f.get("x"), WORLD_W * 0.5),
                            "y": _fnum(f.get("y"), RENDER_H * 0.7),
                            "radius": QUAKE_HAZARD_R + _fnum(f.get("half"), 0.0)})
        # Lava, in *any* scene - not just the volcano. A flow left cooling by the
        # scene before this one is still hot enough to kill for the best part of
        # a minute, and a hazard that vanished with the weather would leave the
        # colony walking cheerfully back into it. The ring shrinks with the flow
        # as it congeals and stops being published at all once the crust closes
        # (LAVA_LETHAL_HEAT), which is the same test the burn path uses - the
        # thing they run from and the thing that hurts them cannot disagree.
        for f in self.lava:
            if not isinstance(f, dict):
                continue
            hot = _clamp(_fnum(f.get("hot"), 0.0))
            if hot < LAVA_LETHAL_HEAT:
                continue
            reach = max(0.0, _fnum(f.get("half"), 0.0) * hot)
            out.append({"kind": "lava", "x": _fnum(f.get("x"), WORLD_W * 0.5),
                        "y": _fnum(f.get("y"), RENDER_H * 0.7),
                        "radius": reach + LAVA_HAZARD_PAD})
        return out

    def _expire_panic(self, world: Any, dt: float) -> None:
        """Count the panic timer down for everyone.

        ``_panic`` sets ``agent.panic`` in seconds but nothing in the sim ever
        decremented it, so a single scare left an agent permanently "in an
        emergency": behavior.emergency_override kept returning True and every
        in-flight job was re-decided for the rest of the session. The scenes
        that raise the flag lower it again.
        """
        for ag in _agents(world):
            try:
                p = _fnum(getattr(ag, "panic", 0.0), 0.0)
                if p > 0.0:
                    ag.panic = max(0.0, p - dt)
            except Exception:
                return

    # ====================================================== scene: ashfall ==
    def _scene_ashfall(self, world: Any, dt: float) -> None:
        self._approach_env(dt, ash=1.0, rate=0.3)
        self._set_wind(dt, 0.45, 0.5, slow=0.17, fast=0.73)
        self.tint = (214, 128, 104)
        self.ember_rate = 0.18
        self.ash_depth = self._accumulate(world, dt, MAT_ASH, self.ash_depth,
                                          ASH_MAX_DEPTH, ASH_RATE, False)

    # ====================================================== scene: volcano ==
    def _scene_volcano(self, world: Any, dt: float) -> None:
        """The ashfall scene with the vent still open, and lava coming out of it.

        Two things happen at once and they are deliberately separate. The *air*
        is an escalation of SCENE_ASHFALL - heavier fall, a red-black sky, a dark
        haze, embers - and costs the colony nothing but visibility. The *ground*
        is the hazard: a vent opens away from the settlement and crawls outward
        at a thirtieth of walking pace, painting MAT_LAVA as it goes.

        That asymmetry is the design. The spectacle is free and constant, so the
        scene reads from across the room for its whole ten minutes; the danger is
        slow, local and telegraphed by a chronicle beat and a quake, so the cost
        of it is measured in the jobs the colony abandons rather than in bodies.
        Anyone who stands in it dies - through the wildfire's own burn path, at
        the wildfire's own contact time - and ``hazards`` tells the AI where it
        is, so anyone who is merely near it leaves.

        Everything to do with the flow itself lives in ``_lava_step``, which runs
        from the shared per-tick path in every scene. This handler only decides
        *when* the ground opens; the cleanup is not its business, and that is
        what makes the cleanup impossible to forget.
        """
        # Air. Ash on the ashfall scene's own channel and ceiling, just faster.
        self._approach_env(dt, ash=1.0, rate=0.35)
        self._set_wind(dt, 0.50, 0.62, slow=0.19, fast=0.83)
        self.tint = (255, 104, 62)
        self.ember_rate = 0.60 + 0.40 * self.intensity
        self.water_level = None
        self.fireflies = False          # nothing comes out in this
        self.fog = _approach(self.fog, VOLCANO_HAZE * self.intensity, 0.25, dt)
        self.fog_color = HAZE_EMBER
        self._melt_snow(world, dt * VOLCANO_THAW_MUL)
        self.ash_depth = self._accumulate(world, dt, MAT_ASH, self.ash_depth,
                                          ASH_MAX_DEPTH, ASH_RATE * VOLCANO_ASH_MUL,
                                          False)
        # _accumulate repaints a growing share of *random* columns, and some of
        # them are columns the flow already owns, so the lava is re-asserted
        # after it. Without this the front comes out speckled with ash the tick
        # it is drawn - the two are writing to the same array.
        for f in self.lava:
            self._lava_paint(world, f)

        # Ground. Beats on threshold crossings of scene_t, the shape the eclipse
        # and the sandstorm use: no "have I erupted yet" flag to add to
        # __init__, to clear in _reset_scene_state and to round-trip, and a save
        # loaded past a beat quietly skips it instead of firing it twice.
        t1 = self.scene_t
        t0 = max(0.0, t1 - dt)
        if t0 < 2.0 <= t1:
            _chronicle(world, "The sky goes the colour of a burn. Ash begins to fall.")
        if t0 < VOLCANO_ERUPT_AT <= t1:
            self._erupt(world, "The ground splits open and lava comes out of it.")
        if t0 < VOLCANO_SECOND_VENT_AT <= t1:
            self._erupt(world, "A second vent tears open. The flow has them from two sides.")

        # ...and the rumble between them. Pulsed rather than constant: the
        # earthquake scene learned that continuous shaking reads as a broken
        # renderer, and this one runs for the full slot. ``** 3`` on a slow sine
        # gives a swell every seventy seconds that is only really felt for about
        # ten of them, so most of the scene is still. Off ``self.t`` rather than
        # ``scene_t`` so two eruptions in a row do not beat in lockstep.
        if self.lava:
            swell = math.sin(self.t * 0.09)
            if swell > 0.0:
                self.add_shake(VOLCANO_SWELL_SHAKE * swell ** 3 * self.intensity)

    def _erupt(self, world: Any, line: str) -> None:
        """Open a vent: pick the ground, tear it, shake the world, say so."""
        if len(self.lava) >= LAVA_MAX_FLOWS:
            return                      # already as much lava as this map gets
        x = self._pick_vent(world)
        self.lava.append({
            "x": float(x),
            "half": 8.0,                # it starts as a crack, not a lake
            "max": self._rng.uniform(LAVA_HALF_MIN, LAVA_HALF_MAX),
            "hot": 1.0,
            "paint": 0.0,
            "y": _ground_y(world, x),
            "seed": self._rng.randrange(1 << 20),
        })
        self.trigger_quake(world, 3.4, VOLCANO_ERUPT_SHAKE)
        # A long, warm flash rather than the lightning's hard white one: this is
        # a glow welling up out of the ground, not a strike.
        _flash(world, 0.50, 2.4, (255, 132, 52))
        _chronicle(world, line)

    def _pick_vent(self, world: Any) -> float:
        """Where the ground opens: away from the colony, and high up if it can be.

        Several candidates scored rather than one draw, because both terms are
        load-bearing. Distance from what the colony has built is the balance
        decision - a vent that opens under the firepit is not a disaster anyone
        can respond to - and height is the flavour one: lava that wells out of
        the high ground reads as a mountain, lava that appears in the valley the
        huts are standing in reads as a bug.

        Falls back to the middle of the map if the world has nothing to measure
        against, which is what a stub world in a test looks like.
        """
        h = _height(world)
        # STAGE, and this one is not a rename - it is a repair. The scoring
        # below only makes sense over a span comparable to LAVA_VENT_CLEAR
        # (520 px) and LAVA_HALF_MAX + LAVA_HAZARD_PAD (205 px). Handed
        # 140..6260 instead of 140..1460, both terms invert:
        #   * the distance credit saturates at 520 px, so every candidate past
        #     that scores identically on the term that was supposed to choose;
        #   * _orphan_width is measured to the far edge, and on a 6400 px map it
        #     is only zero within 205 px of the rim - so the -3.2/px penalty
        #     dominates everything and pins the vent to whichever map edge the
        #     colony is furthest from, ~2600 px away and permanently off camera.
        # Measured, colony at x=3200, 400 draws of the 8-candidate search:
        #   naive WORLD_W rename : 400/400 vents off camera, mean 2717 px from
        #                          home, worst 3059 px
        #   stage-scoped (this)  :   0/400 off camera, mean  582 px, worst 660
        # 582 px is also the shape the scene was tuned for - just past
        # LAVA_VENT_CLEAR, so the distance term is satisfied and the height and
        # orphan terms decide, which is what they were written to do.
        lo, hi = _stage_inset(world, 140.0)
        home = self._colony_x(world)
        best_x, best_score = (lo + hi) * 0.5, -1e18
        for _ in range(LAVA_VENT_TRIES):
            x = self._rng.uniform(lo, hi)
            # Distance credit stops accruing past LAVA_VENT_CLEAR: beyond that
            # the vent is already out of the colony's way, and letting the term
            # keep growing would pin every eruption to whichever map edge the
            # settlement is furthest from.
            score = min(abs(x - home), LAVA_VENT_CLEAR)
            if h is not None:
                gy = float(h[int(_clamp(x, 0.0, float(h.size - 1)))])
                score += (RENDER_H - gy) * 0.35     # y grows downward: high = small y
            # Crowding an existing flow is penalised hard enough that no other
            # term can pay for it: two overlapping fronts are a trap rather than
            # a bigger disaster, because behaviour flees the nearest hazard and
            # will happily run out of one flow into the other.
            for other in self.lava:
                gap = abs(x - _fnum(other.get("x"), x))
                if gap < LAVA_VENT_SPLIT:
                    score -= (LAVA_VENT_SPLIT - gap) * 4.0
            score -= self._orphan_width(x, home, lo, hi) * LAVA_ORPHAN_WEIGHT
            if score > best_score:
                best_x, best_score = x, score
        return best_x

    @staticmethod
    def _orphan_width(x: float, home: float, lo: float, hi: float) -> float:
        """Width of the strip a flow at *x* would cut off from *home*, in px.

        Measured at the front's full extent (half-width plus the hazard pad the
        AI actually flees), because the question is not how wide the lava is now
        but how wide the wall gets before it stops growing. Zero means the front
        reaches the end of the span on its far side, so there is nowhere behind
        it to be stranded - which is the shape this scene wants.

        *lo*/*hi* used to be implicit: they were 0 and RENDER_W, because the
        world was one screen wide and "the edge of the map" and "the edge of
        what anyone will ever walk to" were the same number. They are now the
        stage edges, passed in. Against the world rim on a 6400 px map this
        term stops discriminating - see _pick_vent - and it is the strongest
        term in the score, so it takes the whole decision with it.
        """
        reach = LAVA_HALF_MAX + LAVA_HAZARD_PAD
        if x <= home:                       # colony to the right: pocket is left
            return max(0.0, (x - reach) - float(lo))
        return max(0.0, float(hi) - (x + reach))

    @staticmethod
    def _colony_x(world: Any) -> float:
        """Mean x of what the colony has built, falling back to where it stands."""
        for group in (_iter(world, ("structures",)), _agents(world)):
            xs = [_fnum(getattr(o, "x", None), float("nan")) for o in group]
            xs = [x for x in xs if x == x]
            if xs:
                return sum(xs) / len(xs)
        return WORLD_W * 0.5

    def _lava_step(self, world: Any, dt: float) -> None:
        """Advance, cool and repaint every flow, then burn whoever is in one.

        Called from ``_tick_transients``, i.e. **every tick in every scene**, and
        that is the entire cleanup story. The growth term is gated on the volcano
        scene being the current one, so the front stops dead the moment the
        rotation moves on; the cooling term is not, so the crust closes over,
        the columns are handed back to MAT_STONE from the outside in, and the
        entry drops off this list when there is nothing left of it. Nothing has
        to remember to tidy up: a save loaded three scenes later still finishes
        cooling the flow it was carrying, and a handler that somehow stops
        running leaves lava that goes cold rather than lava that stays hot for
        the rest of the session.
        """
        if not self.lava:
            return
        erupting = self.scene == SCENE_VOLCANO
        live: list[dict[str, Any]] = []
        for f in self.lava:
            if not isinstance(f, dict):
                continue
            cap = _clamp(_fnum(f.get("max"), LAVA_HALF_MAX), 0.0, LAVA_HALF_MAX)
            half = _clamp(_fnum(f.get("half"), 0.0), 0.0, cap)
            hot = _clamp(_fnum(f.get("hot"), 0.0))
            if erupting:
                half = min(cap, half + LAVA_SPREAD * dt * self.intensity)
                hot = 1.0
            else:
                hot = max(0.0, hot - dt / LAVA_COOL_SEC)
            f["max"], f["half"], f["hot"] = cap, half, hot
            self._lava_paint(world, f)
            if hot > 0.0:
                live.append(f)
        self.lava = live[:LAVA_MAX_FLOWS]

        self._lava_consumes(world, dt)
        points = self._lava_points(world)
        if points:
            self._lava_interrupts(world, points)
            # Only the flow, with no burning props folded in. Tempting as it is
            # to do both in one pass, this call runs in *every* scene - including
            # a wildfire that follows an eruption while the flow is still cooling
            # - and that scene is already running its own pass over the same
            # props, with the same per-agent burn timer. Standing in a burning
            # tree that is itself inside a lava flow does then charge two ticks a
            # frame and kills in half the time, which is the right answer anyway.
            self._fire_harms_agents(world, dt, [], extra=points, panic=False,
                                    shelters=True, lethal_sec=LAVA_DEATH_SEC,
                                    phrase="was caught by the lava.")

    def _lava_interrupts(self, world: Any, points: list[tuple[float, float]]) -> None:
        """Make anyone who reaches the edge of the flow drop what they are doing.

        This is the other half of ``_lava_consumes``, and it exists for the same
        gap in the AI. Behaviour re-decides only when the action it is running
        ends, and ``emergency_override`` knows about floods and burning props and
        nothing else - so a villager whose job is on the *far side* of the flow
        walks straight into it, and the 1-D pathing means "the far side" is a
        third of the colony's map. Measured before this existed, twenty-seven of
        one seed's deaths were people crossing, all within fifty px of the same
        edge, none of them with any business being there.

        Abandoning the action is the whole fix: on the next tick behaviour has
        to choose again, FleeFrom is top-scoring by construction (they are
        standing inside the hazard ring this same system published), and they
        turn round on their own. ``Action.abandon`` rather than a poke at
        ``failed`` because it is the sanctioned path - it releases the hut slot,
        the tower slot and the harvest claim the job was holding.

        Deliberately *not* the panic flag, which would also force the re-score:
        panic lets an agent take drops it would otherwise refuse, and on this
        terrain that trades burns for falls roughly one for one (measured on one
        seed: 13 burns + 22 falls with the flag, 35 burns + 4 falls without it).
        The turn-back ring is a little wider than the flames reach, so the
        ordinary case is somebody stopping at the edge unhurt rather than
        somebody stopping because they are already alight.
        """
        for ag in _agents(world):
            try:
                if getattr(ag, "inside", None) is not None:
                    continue            # sheltered; the building is what burns
                act = getattr(ag, "action", None)
                kind = getattr(act, "kind", "") if act is not None else ""
                if not kind or kind in ("FleeFrom", "Panic"):
                    continue            # already leaving
                ax = _fnum(getattr(ag, "x", 1e9), 1e9)
                if min(abs(ax - px) for px, _ in points) > LAVA_TURN_BACK:
                    continue
                act.abandon(ag, world)
            except Exception:
                break                   # an unfamiliar agent shape: stop, quietly

    def _lava_consumes(self, world: Any, dt: float) -> None:
        """Swallow what the flow has reached: props outright, buildings slowly.

        This is not decoration. It is the single most important safety mechanism
        in the scene, and it is here because of what the AI does *not* do:
        ``behavior.emergency_override`` re-decides an in-flight job for a flood,
        for a burning prop or for a raised panic flag, and for nothing else - so
        a villager walking to a job inside the flow keeps walking, and a
        villager working inside the flow keeps working, for the thirteen to
        twenty seconds the job takes. Contact kills in BURN_DEATH_SEC (3.5 s).

        Measured before this existed, one seed lost *eighteen people to the same
        tree*: the front rolled over a wood target, the tree stayed the best job
        on the map, and gatherer after gatherer walked into the lava for it.
        Taking the prop away ends the action - its target stops existing, so the
        handler fails it - and *that* forces the re-score that lets the hazard
        ring do its job. Fixing it in the AI would be the other option, but
        behaviour is not this scene's to change, and "the lava ate the tree" is
        the truer statement anyway.

        Buildings are damaged rather than removed, because ``collapse`` evicts
        the sleepers inside (and world._free_orphaned_sleepers then puts them
        back on the ground, awake, where they can run). Slow enough to watch:
        a hut in the flow is rubble in about fifteen seconds.
        """
        spans = [(_fnum(f.get("x"), 0.0),
                  max(0.0, _fnum(f.get("half"), 0.0) * _clamp(_fnum(f.get("hot"), 0.0))))
                 for f in self.lava
                 if isinstance(f, dict) and _fnum(f.get("hot"), 0.0) >= LAVA_LETHAL_HEAT]
        if not spans:
            return
        reg = getattr(world, "props", None)
        drop = getattr(reg, "remove", None)
        for p in _iter(world, ("props",)):
            try:
                px = _fnum(getattr(p, "x", 1e9), 1e9)
                if not any(abs(px - cx) <= half for cx, half in spans):
                    continue
                if callable(drop):
                    drop(p)
                else:
                    p.alive = False
            except Exception:
                break               # an unfamiliar prop registry: stop, quietly
        for st in _iter(world, ("structures",)):
            try:
                sx = _fnum(getattr(st, "x", 1e9), 1e9)
                if not any(abs(sx - cx) <= half for cx, half in spans):
                    continue
                hurt = getattr(st, "damage", None)
                if callable(hurt):
                    hurt(_fnum(getattr(st, "max_hp", 100.0), 100.0)
                         * dt / LAVA_RUINS_SEC, "lava")
            except Exception:
                break

    def _lava_paint(self, world: Any, f: dict[str, Any]) -> None:
        """Claim the hot span for MAT_LAVA and hand the cooled shoulders back.

        ``paint`` records how much of the material map this flow currently owns,
        which is what makes the reversion exact: as ``hot`` falls the live span
        shrinks, and the difference between the two - and only that difference -
        is repainted MAT_STONE. The flow therefore congeals from its edges
        inward and leaves a stone scar the width it reached, rather than either
        vanishing all at once or leaving live lava behind.

        The 1 px threshold on that debt is load-bearing, not an optimisation.
        Cooling retreats about a twentieth of a pixel per tick, so a version that
        advanced ``paint`` on every tick regardless never accumulated a whole
        column to hand back and quietly congealed *nothing*: the flow went cold,
        dropped off the list, and left its full width painted MAT_LAVA for good.
        ``paint`` may only move when the map has actually been written.
        """
        try:
            x = _fnum(f.get("x"), WORLD_W * 0.5)
            live = max(0.0, _fnum(f.get("half"), 0.0) * _clamp(_fnum(f.get("hot"), 0.0)))
            painted = max(0.0, _fnum(f.get("paint"), 0.0))
            if live > 0.5:
                _paint(world, x - live, x + live, MAT_LAVA)
            if painted - live > 1.0 or (live <= 0.0 < painted):
                _paint(world, x - painted, x - live, MAT_STONE)
                _paint(world, x + live, x + painted, MAT_STONE)
                f["paint"] = live
            elif live > painted:
                f["paint"] = live       # the front took new ground this tick
        except Exception:
            return

    def _lava_points(self, world: Any) -> list[tuple[float, float]]:
        """The hot part of every flow, as (x, y) points sitting on the ground.

        The fire-harm path measures contact horizontally against a list of
        burning *things*, so a flow is handed to it as a row of hot points a
        little closer together than BURN_TOUCH_DIST. That is what lets the lava
        reuse the wildfire's damage path outright instead of growing a second
        one: the same contact time before it kills, the same faster cool-off for
        anyone who runs, and the same short panic radius that keeps people from
        bolting off a ledge they were nowhere near.
        """
        out: list[tuple[float, float]] = []
        for f in self.lava:
            if not isinstance(f, dict):
                continue
            hot = _clamp(_fnum(f.get("hot"), 0.0))
            if hot < LAVA_LETHAL_HEAT:
                continue                # crusted over; it glows but it cannot burn
            x = _fnum(f.get("x"), WORLD_W * 0.5)
            live = max(0.0, _fnum(f.get("half"), 0.0) * hot)
            n = int(min(float(LAVA_SAMPLE_MAX), live * 2.0 / LAVA_SAMPLE_STEP + 1.0))
            if n <= 1:
                out.append((x, _ground_y(world, x)))
                continue
            for i in range(n):
                px = x - live + 2.0 * live * (i / (n - 1))
                out.append((px, _ground_y(world, px)))
        return out

    # ================================================= quakes / shake / fx ==
    def trigger_quake(self, world: Any, duration: float = 2.5,
                      magnitude: float = 6.0) -> None:
        """Start (or extend) an earthquake: shake plus a chance of a fissure."""
        self.quake_t = max(self.quake_t, max(0.2, _fnum(duration, 2.5)))
        self.add_shake(max(1.0, _fnum(magnitude, 6.0)))

    def _tick_quake(self, world: Any, dt: float) -> None:
        if self.quake_t <= 0.0:
            if self.scene in (SCENE_METEOR, SCENE_MUDSLIDE):
                if self._rng.random() < dt / 150.0:
                    self.trigger_quake(world, self._rng.uniform(2.0, 4.5), 6.0)
            return
        self.quake_t = max(0.0, self.quake_t - dt)
        # SCENE_EARTHQUAKE owns all three of these for itself: it drives its own
        # shake envelope (a flat 6.0 here would flatten the swell), it carves
        # capped, spaced scars that this ad-hoc gap would stack an unbounded
        # trench on top of, and it chronicles its own beats - a line per tremor
        # would flush the 10-entry log every three minutes. The rng draw still
        # happens either way so the seeded stream is identical in every other
        # scene, which is the whole reason the scene test is not written first.
        own = self.scene == SCENE_EARTHQUAKE
        if not own:
            self.add_shake(6.0 if self.quake_t > 0.6 else 3.0)
        if self._rng.random() < dt * 0.35 and not own:
            # STAGE: a 6-16 px divot is far too small to notice unless it lands
            # in shot, and this fires on a per-second roll during any borrowed
            # quake. Scattering them across 6400 px would leave the ground in
            # front of the viewer visibly untouched by an earthquake.
            x = self._rng.uniform(*_stage_inset(world, 30.0))
            w = self._rng.uniform(6.0, 16.0)
            _deform(world, x - w, x + w, self._rng.uniform(4.0, 11.0))
            _paint(world, x - w, x + w, MAT_DIRT)
        if self.quake_t <= 0.0 and not own:
            _chronicle(world, "The ground stops shaking.")

    def add_shake(self, amplitude: float) -> None:
        """Request screen shake. Takes the max of current and requested."""
        a = _fnum(amplitude, 0.0)
        if a > self.shake_amp:
            self.shake_amp = min(18.0, a)

    def shake_offset(self) -> tuple[float, float]:
        """Renderer-side camera offset in px. Pure: safe to call many times."""
        if self.shake_amp <= 0.05:
            return (0.0, 0.0)
        a = self.shake_amp
        dx = math.sin(self.shake_t * 47.0) * a
        dy = math.cos(self.shake_t * 39.3) * a * 0.6
        return (dx, dy)

    # ============================================================ control ==
    def request_scene(self, name: str) -> bool:
        """Switch scene (tray menu / commands). Returns False for a bad name."""
        if name not in SCENES:
            return False
        self.scene = name
        self._reset_scene_state()
        return True

    def _reset_scene_state(self) -> None:
        self.scene_t = 0.0
        self.intensity = 0.0
        self._want_advance = False
        self.pending.clear()
        self.strikes.clear()
        self.meteors.clear()
        self.water_level = None
        self.flood_y0 = None
        self.flood_y1 = None
        self.flood_t = 0.0
        self.slide_phase = "idle"
        self.slide_t = 0.0
        self.rumble = 0.0
        self.ember_rate = 0.0
        self.fog = 0.0
        # Back to grey with it. The density going to zero already hides the
        # wash, so a stale tan here is invisible *until* the next low-visibility
        # scene forgets to set a colour - at which point fog would quietly come
        # back as a sandstorm. Reset the colour with the channel it belongs to.
        self.fog_color = HAZE_GREY
        # Cleared here as well as in __init__: _publish reads it every tick in
        # every scene to drive Lighting.ambient_dim, so a stale non-zero value
        # would leave the *next* scene sitting under an eclipse that has no
        # handler to lift it.
        self.eclipse = 0.0
        # Zeroing this is what *lifts the drought*: the wrapper installed on
        # world.regrowth_factor reads this field live rather than capturing a
        # number, so the multiplier is back to 1.0 on the very same tick the
        # scene changes - which matters, because _reset_scene_state runs from
        # World._tick_scene, i.e. after events.tick has already published for
        # this frame but before props.tick reads the factor. _drought_hook then
        # removes the wrapper outright on the next tick.
        self.heat = 0.0
        # The heightmap keeps every scar the quake tore - that is the point of
        # the scene - but the crack overlay and the hazard ring must not survive
        # into the next weather, or a clear afternoon inherits a set of no-go
        # zones nothing is left to explain.
        self.fissures.clear()
        # ``self.lava`` is deliberately NOT cleared here, and it is the one piece
        # of scene state in this file that must not be. A flow is painted into
        # the shared material map, and this list is the only record of which
        # columns it took and how to give them back: dropping it would strand a
        # lethal, unowned lava field on the map for the rest of the session and
        # every session after it, growing by one field per eruption. What stops
        # the flow instead is _lava_step's growth gate - the front only advances
        # while SCENE_VOLCANO is the current scene, so a scene change freezes it
        # on the spot and the cooling path repaints it to stone over the next
        # LAVA_COOL_SEC.
        self.next_tremor = QUAKE_LEAD_IN
        self.tremor_mag = 0.0
        self.tremor_len = 0.0
        self._submerged.clear()
        self._buried.clear()
        self._burning.clear()
        self.next_strike = self._rng.uniform(2.0, 6.0)
        self.next_meteor = self._rng.uniform(1.0, 3.0)
        self.next_ignite = 0.5

    def auto_advance(self, cfg: Any) -> bool:
        """Pick a plausible next scene once the current one has had its run."""
        if cfg is None or not getattr(cfg, "auto_scene_change", False):
            return False
        min_sec = _fnum(getattr(cfg, "scene_min_sec", 180.0), 180.0)
        if not self._want_advance and self.scene_t < max(20.0, min_sec):
            return False
        nxt = self._pick_next()
        if nxt is None:
            self._want_advance = False
            return False
        prev = self.scene
        ok = self.request_scene(nxt)
        if ok:
            log.info("scene %s -> %s", prev, nxt)
        return ok

    def _pick_next(self) -> str | None:
        table = _TRANSITIONS.get(self.scene) or {}
        options = [(s, w) for s, w in table.items() if s in SCENES and s != self.scene]
        if not options:
            options = [(s, 1.0) for s in SCENES if s != self.scene]
        total = sum(w for _, w in options)
        if total <= 0.0:
            return None
        r = self._rng.uniform(0.0, total)
        for name, w in options:
            r -= w
            if r <= 0.0:
                return name
        return options[-1][0]

    # ================================================================= io ==
    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "scene_t": float(self.scene_t),
            "intensity": float(self.intensity),
            "t": float(self.t),
            "seed": int(self.seed),
            "wind": float(self.wind),
            "rain": float(self.rain),
            "snow": float(self.snow),
            "ash": float(self.ash),
            "fog": float(self.fog),
            "fog_color": list(self.fog_color),
            "eclipse": float(self.eclipse),
            "heat": float(self.heat),
            "gust": float(self.gust),
            "water_level": None if self.water_level is None else float(self.water_level),
            "quake_t": float(self.quake_t),
            "fissures": [dict(f) for f in self.fissures if isinstance(f, dict)],
            "lava": [dict(f) for f in self.lava if isinstance(f, dict)],
            "next_tremor": float(self.next_tremor),
            "tremor_mag": float(self.tremor_mag),
            "tremor_len": float(self.tremor_len),
            "pending": [dict(p) for p in self.pending if isinstance(p, dict)],
            "strikes": [dict(s) for s in self.strikes],
            "meteors": [dict(m) for m in self.meteors],
            "ember_rate": float(self.ember_rate),
            "tint": list(self.tint),
            "rumble": float(self.rumble),
            "shake_amp": float(self.shake_amp),
            "shake_t": float(self.shake_t),
            "snow_depth": float(self.snow_depth),
            "ash_depth": float(self.ash_depth),
            "sand_depth": float(self.sand_depth),
            "snow_prev": self._snow_prev,
            "slide_phase": self.slide_phase,
            "slide_x0": float(self.slide_x0),
            "slide_x1": float(self.slide_x1),
            "slide_t": float(self.slide_t),
            "flood_y0": None if self.flood_y0 is None else float(self.flood_y0),
            "flood_y1": None if self.flood_y1 is None else float(self.flood_y1),
            "flood_t": float(self.flood_t),
            "buried": {str(k): float(v) for k, v in self._buried.items()},
            "next_strike": float(self.next_strike),
            "next_meteor": float(self.next_meteor),
            "next_ignite": float(self.next_ignite),
            "submerged": {str(k): float(v) for k, v in self._submerged.items()},
            "burning": {str(k): float(v) for k, v in self._burning.items()},
            "want_advance": bool(self._want_advance),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "EventSystem":
        ev = cls()
        if not isinstance(d, dict):
            return ev
        scene = d.get("scene")
        ev.scene = scene if isinstance(scene, str) and scene in SCENES else SCENE_NIGHT_STORM
        ev.scene_t = max(0.0, _fnum(d.get("scene_t"), 0.0))
        ev.intensity = _clamp(_fnum(d.get("intensity"), 1.0))
        ev.t = max(0.0, _fnum(d.get("t"), 0.0))
        seed = d.get("seed")
        if isinstance(seed, (int, float)) and not isinstance(seed, bool):
            ev.seed = int(seed)
            ev._rng = random.Random(ev.seed)
        ev.wind = _clamp(_fnum(d.get("wind"), 0.0), -1.0, 1.0)
        ev.rain = _clamp(_fnum(d.get("rain"), 0.0))
        ev.snow = _clamp(_fnum(d.get("snow"), 0.0))
        ev.ash = _clamp(_fnum(d.get("ash"), 0.0))
        ev.fog = _clamp(_fnum(d.get("fog"), 0.0))
        ev.fog_color = _clean_rgb(d.get("fog_color"), HAZE_GREY)
        ev.eclipse = _clamp(_fnum(d.get("eclipse"), 0.0))
        ev.heat = _clamp(_fnum(d.get("heat"), 0.0))
        ev.gust = _clamp(_fnum(d.get("gust"), 0.0))
        wl = d.get("water_level")
        ev.water_level = None if wl is None else _fnum(wl, 0.0)
        ev.quake_t = max(0.0, _fnum(d.get("quake_t"), 0.0))
        # Re-clamped rather than trusted: a hand-edited save that asks for a
        # 400 px fissure would otherwise carve a lethal chasm on the first tick
        # after load, which is precisely what the caps upstream exist to stop.
        ev.fissures = [_clean_fissure(f) for f in _listof(d.get("fissures"))
                       if isinstance(f, dict)][:QUAKE_FISSURE_MAX]
        # Same treatment, and for a sharper reason: ``paint`` is what the flow
        # owes the material map back, so a save that arrived with a 900 px one
        # would repaint most of the world to stone as it cooled.
        ev.lava = [_clean_lava(f) for f in _listof(d.get("lava"))
                   if isinstance(f, dict)][:LAVA_MAX_FLOWS]
        ev.next_tremor = max(0.0, _fnum(d.get("next_tremor"), QUAKE_LEAD_IN))
        ev.tremor_mag = _clamp(_fnum(d.get("tremor_mag"), 0.0), 0.0, QUAKE_MAG_MAX)
        ev.tremor_len = _clamp(_fnum(d.get("tremor_len"), 0.0), 0.0, QUAKE_TREMOR_MAX)
        ev.pending = [dict(p) for p in _listof(d.get("pending")) if isinstance(p, dict)][:32]
        ev.strikes = [dict(s) for s in _listof(d.get("strikes")) if isinstance(s, dict)][:6]
        ev.meteors = [dict(m) for m in _listof(d.get("meteors")) if isinstance(m, dict)][:12]
        ev.ember_rate = _clamp(_fnum(d.get("ember_rate"), 0.0))
        tint = d.get("tint")
        try:
            r, g, b = (int(c) for c in tuple(tint)[:3])
            ev.tint = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
        except (TypeError, ValueError):
            ev.tint = (255, 255, 255)
        ev.rumble = _clamp(_fnum(d.get("rumble"), 0.0))
        ev.shake_amp = max(0.0, _fnum(d.get("shake_amp"), 0.0))
        ev.shake_t = max(0.0, _fnum(d.get("shake_t"), 0.0))
        ev.snow_depth = _clamp(_fnum(d.get("snow_depth"), 0.0), 0.0, SNOW_MAX_DEPTH)
        ev.ash_depth = _clamp(_fnum(d.get("ash_depth"), 0.0), 0.0, ASH_MAX_DEPTH)
        ev.sand_depth = _clamp(_fnum(d.get("sand_depth"), 0.0), 0.0, SAND_MAX_DEPTH)
        prev = d.get("snow_prev")
        if isinstance(prev, list) and prev:
            try:
                ev._snow_prev = [int(v) & 0xFF for v in prev]
            except (TypeError, ValueError):
                ev._snow_prev = None
        phase = d.get("slide_phase")
        ev.slide_phase = phase if phase in ("idle", "warn", "slide", "settle", "done") else "idle"
        ev.slide_x0 = _fnum(d.get("slide_x0"), 0.0)
        ev.slide_x1 = _fnum(d.get("slide_x1"), 0.0)
        ev.slide_t = max(0.0, _fnum(d.get("slide_t"), 0.0))
        f0, f1 = d.get("flood_y0"), d.get("flood_y1")
        ev.flood_y0 = None if f0 is None else _fnum(f0, 0.0)
        ev.flood_y1 = None if f1 is None else _fnum(f1, 0.0)
        ev.flood_t = max(0.0, _fnum(d.get("flood_t"), 0.0))
        ev._buried = _intmap(d.get("buried"))
        ev.next_strike = max(0.0, _fnum(d.get("next_strike"), 5.0))
        ev.next_meteor = max(0.0, _fnum(d.get("next_meteor"), 2.0))
        ev.next_ignite = max(0.0, _fnum(d.get("next_ignite"), 0.5))
        ev._submerged = _intmap(d.get("submerged"))
        ev._burning = _intmap(d.get("burning"))
        ev._want_advance = bool(d.get("want_advance", False))
        return ev


def _smooth(u: float) -> float:
    u = _clamp(u)
    return u * u * (3.0 - 2.0 * u)


def eclipse_strength(scene_t: float) -> float:
    """Eclipse darkness, 0..1, as a pure function of the scene's age in seconds.

    Public and side-effect free so a test (or a future HUD) can ask "what will
    this look like at t=70?" without building a world.

    The ``** 2.2`` is the whole character of the scene. A linear ramp reads as
    somebody sliding a brightness fader for a minute; the real thing keeps
    looking like an ordinary bright day until the sun is most of the way gone
    and then drops away in the last fifteen seconds. Squaring the ingress buys
    exactly that: at the halfway point of the ingress this returns 0.22, i.e.
    the world has lost a fifth of its light while the disc is already 60% eaten
    (the renderer derives disc coverage back out of this value - see
    sky._eclipse_cover - so the two can never drift apart).
    """
    t = _fnum(scene_t, 0.0)
    if t <= 0.0:
        return 0.0
    if t < ECLIPSE_INGRESS:
        return _smooth(t / ECLIPSE_INGRESS) ** 2.2
    hold_end = ECLIPSE_INGRESS + ECLIPSE_TOTALITY
    if t < hold_end:
        return 1.0
    if t < hold_end + ECLIPSE_EGRESS:
        return _smooth(1.0 - (t - hold_end) / ECLIPSE_EGRESS) ** 2.2
    return 0.0                                  # passed; ordinary daylight again


def _listof(v: Any) -> list[Any]:
    return list(v) if isinstance(v, list) else []


def _intmap(v: Any) -> dict[int, float]:
    out: dict[int, float] = {}
    if not isinstance(v, dict):
        return out
    for k, val in v.items():
        try:
            out[int(k)] = float(val)
        except (TypeError, ValueError):
            continue
    return out


# ------------------------------------------------------------ smoke test --
if __name__ == "__main__":                                  # pragma: no cover
    from .lighting import Lighting

    class _T:
        def __init__(self) -> None:
            # WORLD: this stub stands in for sim.terrain.Terrain, whose arrays
            # are WORLD_W wide. A RENDER_W-wide stub would index out of range
            # the moment a stage-sited event landed past x=1600.
            self.height = np.full(WORLD_W, RENDER_H * 0.7, dtype=np.float32)
            self.height += (np.sin(np.arange(WORLD_W) / 90.0) * 60.0).astype(np.float32)
            self.material = np.zeros(WORLD_W, dtype=np.uint8)

        def ground_y(self, x: float) -> float:
            return float(self.height[int(max(0, min(WORLD_W - 1, x)))])

    class _A:
        def __init__(self, i: int, x: float, y: float) -> None:
            self.id, self.x, self.y = i, x, y
            self.alive, self.warmth, self.morale = True, 0.0, 0.6
            self.name = f"agent{i}"

    class _P:
        def __init__(self, i: int, x: float, y: float) -> None:
            self.id, self.x, self.y, self.kind = i, x, y, "tree"
            self.burning = False

    class _W:
        def __init__(self) -> None:
            self.terrain = _T()
            self.lighting = Lighting()
            self.world_time = 0.0
            self.agents = [_A(i, 100.0 + i * 150.0, 0.0) for i in range(6)]
            for a in self.agents:
                a.y = self.terrain.ground_y(a.x)
            self.props = [_P(100 + i, 80.0 + i * 200.0, 0.0) for i in range(6)]
            self.chronicle: list[str] = []

    w = _W()
    ev = EventSystem()
    for scene in SCENES:
        ev.request_scene(scene)
        for _ in range(int(30 * 25)):        # 25 sim-seconds per scene
            ev.tick(w, 1 / 30)
            w.lighting.tick(1 / 30)
            w.world_time += 1 / 30
        alive = sum(1 for a in w.agents if a.alive)
        print(f"{scene:12s} rain={ev.rain:.2f} snow={ev.snow:.2f} ash={ev.ash:.2f} "
              f"wind={ev.wind:+.2f} water={ev.water_level} alive={alive} "
              f"shake={ev.shake_offset()[0]:+.2f}")
    blob = ev.to_dict()
    back = EventSystem.from_dict(blob)
    print("roundtrip scene:", back.scene, "keys:", len(blob))
    print("degenerate from_dict:", EventSystem.from_dict({"scene": "nope"}).scene)
    ev.tick(None, 1 / 30)                    # must not raise with no world
    print("null-world tick ok; errors:", ev._errors)
