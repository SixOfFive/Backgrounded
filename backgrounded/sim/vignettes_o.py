"""Vignette content, band O - wonder and discovery.

Thirty cosmetic beats about *noticing*. A stickman in this band has stopped
because something caught it: a glint in the dirt, a stone with a hole through
it, its own face looking back out of the water, the moon still up at noon. It
turns the thing over, holds it to the light, buries it, digs it up again, and
waits for somebody to come and be impressed.

Nothing here changes anything. A label that says "digs up something it buried"
does not touch the terrain and the treasure does not exist; the find is in the
viewer's head, which is the whole trick. The engine may only move this agent's
pose, facing, speech and a few px of x - see :mod:`.vignettes` for the contract
and the normalisation rules.

House rules for this file
-------------------------
* Every key is ``o_``-prefixed so the bands cannot collide.
* Poses stay inside the renderer's own vocabulary (``idle``/``walk``/``carry``/
  ``chop``/``build``/``climb``/``fall``/``sleep``/``dance``/``mourn``/
  ``panic``), so nothing here falls back to a blank stand.
* Gating tags are ones :func:`~.vignettes.context_tags` really evaluates, so
  context *means* something: a reflection wants water, a first star wants
  night, and the elder is the one who already knows what it is holding. Two
  entries carry a purely decorative tag instead - those never gate, so the
  pool cannot empty out.
* Short beats outnumber long ones. Most discoveries are over in three seconds;
  only a couple are worth sitting down for.
* The neighbouring bands already cover shadows (B, D, E), star-tracing (B),
  catching snowflakes (B, D) and turning a rock over (B). This band goes
  elsewhere on purpose: reflections, holes in stones, the daytime moon, the
  sky in a puddle, and the long comic business of losing what you just found.
"""
from __future__ import annotations

from .vignettes import Vignette

__all__ = ["VIGNETTES"]


VIGNETTES: list[Vignette] = [
    # ==================================================================
    #  Ungated - the plain act of finding a thing and studying it. These
    #  fit any context, so they are what carries the band on a grey
    #  afternoon with nobody about.
    # ==================================================================
    Vignette(
        "o_shiny_find", "stops dead over something shiny", "idle",
        (1.8, 3.0), 1.3, speech="*", motion="still",
    ),
    Vignette(
        # climb reads as an arm stretched up - exactly the gesture.
        "o_hold_to_light", "holds a find up to the light", "climb",
        (2.0, 3.4), 1.2, motion="still",
    ),
    Vignette(
        "o_turn_it_over", "turns something over and over, unconvinced", "idle",
        (2.4, 4.0), 1.1, speech="?", motion="still",
    ),
    Vignette(
        "o_thumb_rub", "rubs a thumb over a find to see its real colour", "idle",
        (1.6, 2.8), 1.0, motion="still",
    ),
    Vignette(
        "o_listen_to_it", "holds something to its ear and listens", "idle",
        (2.0, 3.0), 0.9, speech="~", motion="still",
    ),
    Vignette(
        "o_knock_stones", "knocks two stones together to see what is inside",
        "chop", (1.8, 2.8), 0.9, motion="still",
    ),
    Vignette(
        "o_cupped_hands", "carries something in cupped hands, checking on it",
        "carry", (3.0, 5.0), 0.9, motion="pace", drift=24.0,
    ),
    Vignette(
        # The burying itself is one beat; the counting is what sells it as
        # a treasure rather than a hole.
        "o_bury_and_count", "buries a treasure and counts the steps back to it",
        "walk", (3.5, 6.0), 0.7, motion="pace", drift=34.0,
    ),
    Vignette(
        "o_dig_it_up", "digs up something it buried and is delighted", "build",
        (2.6, 4.5), 0.9, speech="*", motion="crouch",
    ),
    Vignette(
        "o_sort_finds", "sorts its finds into an order only it understands",
        "sleep", (4.0, 7.0), 0.6, motion="sit",
    ),

    # ------------------------------------------------------------------
    #  Decorative tags only: they never gate, and the weights keep them
    #  rare enough to stay funny when they do land.
    # ------------------------------------------------------------------
    Vignette(
        "o_drops_it", "drops its find and hunts through the grass for it",
        "build", (2.4, 4.2), 0.45, tags=("silly",), motion="crouch",
    ),
    Vignette(
        "o_tastes_it", "tastes a find and regrets it at once", "panic",
        (1.5, 2.4), 0.15, tags=("silly",), speech="!", motion="still",
    ),

    # ==================================================================
    #  Stone - the nearest interesting thing in any colony.
    # ==================================================================
    Vignette(
        "o_hole_stone", "looks through a stone with a hole in it", "idle",
        (2.2, 3.8), 0.9, tags=("near_rock",), speech="o", motion="still",
    ),
    Vignette(
        "o_rock_nose_close", "examines a rock from very close up", "build",
        (2.0, 3.6), 1.1, tags=("near_rock",), motion="crouch",
    ),
    Vignette(
        "o_under_rock", "lies flat to see what is under a rock", "sleep",
        (2.8, 4.6), 0.5, tags=("near_rock",), motion="lie",
    ),

    # ==================================================================
    #  Water - the only mirror the colony owns.
    # ==================================================================
    Vignette(
        "o_reflection_meet", "meets itself in the water and stops dead", "idle",
        (1.8, 3.0), 1.0, tags=("near_water",), speech="!", motion="still",
    ),
    Vignette(
        "o_reflection_wave",
        "waves at its reflection and waits for it to wave first", "build",
        (2.6, 4.4), 0.15, tags=("near_water",), motion="crouch",
    ),

    # ==================================================================
    #  Fire and candle - light as a tool for looking closer.
    # ==================================================================
    Vignette(
        "o_two_shadows", "discovers it has two shadows and tries to lose one",
        "idle", (2.6, 4.5), 0.5, tags=("near_fire",), motion="spin",
    ),
    Vignette(
        "o_candle_close", "brings the candle close to see something better",
        "build", (2.2, 3.8), 0.9, tags=("candle",), motion="crouch",
    ),

    # ==================================================================
    #  Sky - and the small print in it that nobody else looked up for.
    # ==================================================================
    Vignette(
        "o_first_star", "points out the first star", "idle",
        (1.8, 3.0), 1.1, tags=("night",), motion="still",
    ),
    Vignette(
        # Two gates and a low weight: this is meant to be a rarity, one
        # figure alone under a whole sky naming things after itself.
        "o_name_stars", "gives a run of stars a name of its own", "idle",
        (3.2, 5.5), 0.3, tags=("night", "alone"), motion="still",
    ),
    Vignette(
        "o_moon_in_daylight", "notices the moon is out in the daytime", "idle",
        (1.8, 3.0), 0.9, tags=("day",), speech="?", motion="still",
    ),
    Vignette(
        "o_one_cloud", "follows one cloud all the way across", "idle",
        (3.5, 6.0), 0.7, tags=("day",), motion="spin",
    ),

    # ==================================================================
    #  Weather that hands you something to look at.
    # ==================================================================
    Vignette(
        "o_puddle_sky", "finds the whole sky in a puddle", "build",
        (2.4, 4.0), 0.6, tags=("raining",), motion="crouch",
    ),
    Vignette(
        "o_snow_crystal", "studies a snowflake on its sleeve until it is gone",
        "idle", (1.8, 3.0), 1.0, tags=("snowing",), motion="still",
    ),

    # ==================================================================
    #  Company - a find is only half a find until somebody else sees it.
    # ==================================================================
    Vignette(
        "o_show_and_wait", "holds a find out and waits to be impressed", "carry",
        (2.2, 3.8), 1.1, tags=("social",), motion="still",
    ),
    Vignette(
        "o_dance_over_find", "hops about over something it found", "dance",
        (2.0, 3.4), 0.35, tags=("happy",), motion="hop",
    ),

    # ==================================================================
    #  Who is doing the looking.
    # ==================================================================
    Vignette(
        "o_child_run_show", "runs to show somebody what it found", "walk",
        (2.4, 4.0), 0.9, tags=("child",), motion="pace", drift=34.0,
    ),
    Vignette(
        "o_elder_knows", "turns a find over and knows at once what it is", "idle",
        (2.6, 4.2), 0.8, tags=("elder",), motion="still",
    ),
    Vignette(
        "o_lookout_speck", "watches a speck on the horizon, refusing to blink",
        "idle", (3.0, 5.0), 0.8, tags=("lookout",), motion="still",
    ),
]


# ===========================================================================
#  Smoke test - content-side invariants only; the engine tests itself.
# ===========================================================================
if __name__ == "__main__":   # pragma: no cover
    from .vignettes import MAX_DRIFT, MAX_DUR, MIN_DUR, MOTIONS

    assert len(VIGNETTES) == 30, len(VIGNETTES)
    keys = [v.key for v in VIGNETTES]
    assert len(set(keys)) == 30, "duplicate key"
    assert all(k.startswith("o_") for k in keys), "unprefixed key"
    assert all(MIN_DUR <= v.dur[0] <= v.dur[1] <= MAX_DUR for v in VIGNETTES)
    assert all(v.motion in MOTIONS for v in VIGNETTES)
    assert all(0.0 <= v.drift <= MAX_DRIFT for v in VIGNETTES)
    assert all(v.weight > 0.0 for v in VIGNETTES)
    assert all(v.speech is None or len(v.speech) == 1 for v in VIGNETTES)
    assert all(v.drift == 0.0 or v.motion == "pace" for v in VIGNETTES)

    poses = {"idle", "walk", "carry", "chop", "build", "climb", "fall",
             "sleep", "dance", "mourn", "panic"}
    assert all(v.pose in poses for v in VIGNETTES), \
        {v.pose for v in VIGNETTES} - poses

    tagged = [v for v in VIGNETTES if v.tags]
    ungated = [v for v in VIGNETTES if not v.gates]
    short = [v for v in VIGNETTES if v.dur[1] <= 3.0]
    long_ = [v for v in VIGNETTES if v.dur[1] > 4.5]
    spoken = [v for v in VIGNETTES if v.speech]
    assert len(tagged) >= 15, len(tagged)
    assert len(ungated) >= 12, len(ungated)
    assert len(short) > len(long_), (len(short), len(long_))
    assert 6 <= len(spoken) <= 9, len(spoken)

    used = {v.motion for v in VIGNETTES}
    assert used == set(MOTIONS), set(MOTIONS) - used
    print(f"{len(VIGNETTES)} vignettes, {len(tagged)} tagged, "
          f"{len(ungated)} always-available, {len(spoken)} with speech")
    print("motions:", {m: sum(1 for v in VIGNETTES if v.motion == m)
                       for m in MOTIONS})
