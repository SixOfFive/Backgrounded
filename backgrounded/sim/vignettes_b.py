"""Vignette content, band B - the world and everything in it.

Forty cosmetic beats about *place*: dirt, stone, water, weather, trees, the
edge of a drop. A stickman in this band is never working - it is poking the
ground, turning a rock over to see what lives under it, holding a hand up to
find the wind, or throwing a pebble off a cliff because the cliff is there.

Nothing here touches the world. Every entry is inert data; the state machine
in :mod:`.vignettes` is what plays it, and that machine may only move the
acting agent's pose, facing, speech and a few px of x. A label that *says*
"digs a hole" does not dig a hole - the terrain is untouched and the hole is
in the viewer's head, which is exactly the deal.

Poses are drawn from ``render.stickfigure.POSES`` and its alias table
(``forage``/``plant``/``lookout``/``mine`` all resolve onto a drawn body), so
nothing in this list falls back to a blank stand.
"""
from __future__ import annotations

import logging

from .vignettes import Vignette

log = logging.getLogger(__name__)

__all__ = ["VIGNETTES"]


#: Band B content. Every key is ``b_``-prefixed so it cannot collide with the
#: other four content modules.
VIGNETTES: list[Vignette] = [
    # ---------------------------------------------------------- the ground --
    Vignette(
        key="b_poke_ground",
        label="pokes at the ground",
        pose="build",
        dur=(1.8, 3.0),
        weight=1.3,
        motion="crouch",
    ),
    Vignette(
        key="b_pebble_walk",
        label="carries a pebble a few steps and puts it down",
        pose="carry",
        dur=(3.0, 5.0),
        weight=1.0,
        motion="pace",
        drift=30.0,
    ),
    Vignette(
        key="b_dirt_drawing",
        label="draws something in the dirt",
        pose="build",
        dur=(3.0, 5.5),
        weight=1.0,
        motion="crouch",
    ),
    Vignette(
        key="b_heel_scuff",
        label="scuffs a long line in the dirt with one heel",
        pose="walk",
        dur=(2.4, 4.0),
        weight=1.1,
        motion="pace",
        drift=22.0,
    ),
    Vignette(
        key="b_dig_and_fill",
        label="digs a small hole and fills it back in",
        pose="build",
        dur=(4.0, 7.0),
        weight=0.7,
        motion="crouch",
    ),
    Vignette(
        key="b_listen_ground",
        label="presses an ear to the ground and listens",
        pose="sleep",
        dur=(2.5, 4.5),
        weight=0.5,
        speech="?",
        motion="lie",
    ),
    Vignette(
        key="b_follow_ant",
        label="follows something very small along the ground",
        pose="walk",
        dur=(3.0, 5.5),
        weight=0.9,
        speech="?",
        motion="pace",
        drift=26.0,
    ),
    Vignette(
        key="b_chew_grass",
        label="plucks a blade of grass and chews it",
        pose="forage",
        dur=(2.5, 4.5),
        weight=1.0,
        motion="sit",
    ),
    Vignette(
        key="b_toss_dust",
        label="throws a handful of dust into the air",
        pose="chop",
        dur=(1.6, 2.8),
        weight=0.45,
        motion="still",
    ),
    Vignette(
        key="b_bury_a_secret",
        label="buries something small and marks the spot",
        pose="plant",
        dur=(4.5, 7.5),
        weight=0.15,
        speech="*",
        motion="crouch",
    ),

    # ------------------------------------------------------------- stone ---
    Vignette(
        key="b_stack_pebbles",
        label="stacks three pebbles on top of each other",
        pose="build",
        dur=(3.5, 6.0),
        weight=0.8,
        speech="*",
        motion="crouch",
    ),
    Vignette(
        key="b_stone_ring",
        label="arranges loose stones into a small ring",
        pose="build",
        dur=(4.0, 7.0),
        weight=0.5,
        motion="crouch",
    ),
    Vignette(
        key="b_kick_rock",
        label="kicks a loose rock along the ground",
        pose="walk",
        dur=(2.6, 4.2),
        weight=1.1,
        tags=("near_rock",),
        motion="pace",
        drift=30.0,
    ),
    Vignette(
        key="b_turn_over_stone",
        label="turns a stone over to see what is under it",
        pose="forage",
        dur=(2.2, 3.6),
        weight=1.0,
        tags=("near_rock",),
        motion="crouch",
    ),
    Vignette(
        key="b_heave_at_boulder",
        label="tries to lift a boulder and thinks better of it",
        pose="build",
        dur=(2.0, 3.4),
        weight=0.8,
        tags=("near_rock",),
        speech="!",
        motion="crouch",
    ),
    Vignette(
        key="b_show_a_stone",
        label="shows someone a stone it found",
        pose="carry",
        dur=(2.0, 3.5),
        weight=0.8,
        tags=("social",),
        speech="!",
        motion="still",
    ),

    # ------------------------------------------------------------- water ---
    Vignette(
        key="b_skip_stone",
        label="skips a stone across the water",
        pose="chop",
        dur=(2.4, 4.2),
        weight=1.2,
        tags=("near_water",),
        speech="o",
        motion="still",
    ),
    Vignette(
        key="b_splash_water",
        label="splashes at the water with one hand",
        pose="build",
        dur=(2.0, 3.6),
        weight=1.0,
        tags=("near_water",),
        motion="crouch",
    ),
    Vignette(
        key="b_water_through_fingers",
        label="lets water run through its fingers",
        pose="carry",
        dur=(2.5, 4.0),
        weight=0.9,
        tags=("near_water",),
        motion="crouch",
    ),
    Vignette(
        key="b_watch_ripples",
        label="watches the ripples spread out and go flat",
        pose="idle",
        dur=(3.0, 6.0),
        weight=0.9,
        tags=("near_water",),
        motion="sit",
    ),

    # -------------------------------------------------------------- trees ---
    Vignette(
        key="b_pat_tree",
        label="pats a tree trunk",
        pose="idle",
        dur=(1.6, 2.8),
        weight=1.0,
        tags=("near_tree",),
        motion="still",
    ),
    Vignette(
        key="b_sniff_flower",
        label="sniffs at a flower",
        pose="forage",
        dur=(2.0, 3.2),
        weight=0.9,
        tags=("day",),
        motion="crouch",
    ),
    Vignette(
        key="b_count_rings",
        label="counts the rings on an old stump",
        pose="build",
        dur=(3.0, 5.0),
        weight=0.4,
        tags=("near_tree",),
        speech="?",
        motion="crouch",
    ),
    Vignette(
        key="b_watch_leaf",
        label="watches a leaf come all the way down",
        pose="idle",
        dur=(2.0, 3.6),
        weight=0.9,
        tags=("near_tree",),
        motion="still",
    ),
    Vignette(
        key="b_shelter_under_tree",
        label="shelters under a tree until it eases off",
        pose="idle",
        dur=(3.5, 6.5),
        weight=1.0,
        tags=("raining", "near_tree"),
        motion="sit",
    ),

    # ------------------------------------------------------------ weather ---
    Vignette(
        key="b_test_wind",
        label="holds a hand up to find the wind",
        pose="idle",
        dur=(1.8, 3.0),
        weight=1.1,
        tags=("windy",),
        motion="spin",
    ),
    Vignette(
        key="b_lean_into_wind",
        label="leans into the wind and holds there",
        pose="walk",
        dur=(2.0, 3.5),
        weight=1.0,
        tags=("windy",),
        motion="still",
    ),
    Vignette(
        key="b_watch_rain",
        label="watches the rain come down",
        pose="idle",
        dur=(3.0, 6.0),
        weight=1.2,
        tags=("raining",),
        motion="sit",
    ),
    Vignette(
        key="b_stamp_puddle",
        label="stamps in a puddle just to hear it",
        pose="walk",
        dur=(1.6, 2.8),
        weight=1.0,
        tags=("raining",),
        motion="hop",
    ),
    Vignette(
        key="b_slip_in_mud",
        label="slips in the mud and gets up again",
        pose="fall",
        dur=(1.6, 2.6),
        weight=0.3,
        tags=("raining",),
        speech="!",
        motion="still",
    ),
    Vignette(
        key="b_catch_snowflake",
        label="tries to catch a single snowflake",
        pose="idle",
        dur=(2.0, 3.6),
        weight=1.0,
        tags=("snowing",),
        speech="*",
        motion="hop",
    ),
    Vignette(
        key="b_stamp_snow",
        label="stamps a footprint into fresh snow",
        pose="walk",
        dur=(1.8, 3.0),
        weight=0.9,
        tags=("snowing",),
        motion="hop",
    ),
    Vignette(
        key="b_smell_the_air",
        label="turns about, smelling the air",
        pose="idle",
        dur=(1.6, 2.6),
        weight=1.2,
        motion="spin",
    ),

    # ----------------------------------------------------------- high up ---
    Vignette(
        key="b_peer_over_ledge",
        label="peers over the ledge",
        pose="lookout",
        dur=(2.5, 4.5),
        weight=1.0,
        tags=("high_up",),
        motion="crouch",
    ),
    Vignette(
        key="b_stone_off_cliff",
        label="throws a stone off the cliff and waits",
        pose="chop",
        dur=(2.0, 3.5),
        weight=0.9,
        tags=("high_up",),
        speech="o",
        motion="still",
    ),
    Vignette(
        key="b_shout_at_the_valley",
        label="shouts once, to hear it come back",
        pose="idle",
        dur=(2.0, 3.5),
        weight=0.3,
        tags=("high_up", "alone"),
        speech="!",
        motion="still",
    ),

    # ------------------------------------------------------ sky and light ---
    Vignette(
        key="b_trace_stars",
        label="traces a shape in the stars with one finger",
        pose="idle",
        dur=(3.5, 6.5),
        weight=0.8,
        tags=("night",),
        speech="*",
        motion="still",
    ),
    Vignette(
        key="b_lie_and_look_up",
        label="lies down and looks straight up",
        pose="sleep",
        dur=(4.0, 8.0),
        weight=0.7,
        tags=("night",),
        motion="lie",
    ),
    Vignette(
        key="b_measure_shadow",
        label="measures its own shadow",
        pose="idle",
        dur=(2.4, 3.6),
        weight=0.5,
        tags=("day",),
        motion="spin",
    ),
    Vignette(
        key="b_watch_sparks",
        label="watches the sparks climb into the dark",
        pose="warm",
        dur=(3.0, 6.0),
        weight=1.0,
        tags=("near_fire",),
        motion="sit",
    ),
]


# ---------------------------------------------------------------------------
# Self-registration safety net.
#
# The engine's load_all() imports this module and reads VIGNETTES. That works
# for the normal path (vignettes.py first), but if *this* module is imported
# first - `from backgrounded.sim.vignettes_b import VIGNETTES`, which is how it
# gets tested - then the engine imports us back mid-execution, finds no
# VIGNETTES yet (it is defined below that import), and logs a warning while the
# other bands register fine. Registering here closes that window. It is
# idempotent: register() skips a key already bound to an identical vignette,
# and on the normal path these are the very same objects.
# ---------------------------------------------------------------------------
def _self_register() -> None:
    try:
        from .vignettes import register
    except Exception:                       # pragma: no cover - engine gone
        return
    for v in VIGNETTES:
        try:
            register(v)
        except Exception:
            log.debug("vignettes_b: %r already registered elsewhere",
                      getattr(v, "key", "?"), exc_info=True)


_self_register()


# ===========================================================================
#  Smoke test
# ===========================================================================
if __name__ == "__main__":   # pragma: no cover - headless content check
    from collections import Counter

    from .vignettes import MAX_DRIFT, MAX_DUR, MIN_DUR, MOTIONS

    assert len(VIGNETTES) == 40, len(VIGNETTES)
    keys = [v.key for v in VIGNETTES]
    assert len(set(keys)) == 40, "duplicate key"
    assert all(k.startswith("b_") for k in keys), [k for k in keys
                                                  if not k.startswith("b_")]
    assert all(MIN_DUR <= v.dur[0] <= v.dur[1] <= MAX_DUR for v in VIGNETTES)
    assert all(v.motion in MOTIONS for v in VIGNETTES)
    assert all(0.0 <= v.drift <= MAX_DRIFT for v in VIGNETTES)
    assert all(v.weight > 0.0 for v in VIGNETTES)
    assert all(len(v.speech or "") <= 1 for v in VIGNETTES)
    # Only pacing vignettes are allowed a drift budget.
    assert all(v.drift == 0.0 or v.motion == "pace" for v in VIGNETTES)

    tagged = [v for v in VIGNETTES if v.tags]
    untagged = [v for v in VIGNETTES if not v.gates]
    short = [v for v in VIGNETTES if v.dur[1] <= 3.6]
    long_ = [v for v in VIGNETTES if v.dur[1] > 4.5]
    speaking = [v for v in VIGNETTES if v.speech]

    assert len(tagged) >= 20, len(tagged)
    assert len(untagged) >= 8, len(untagged)
    assert len(short) > len(long_), (len(short), len(long_))

    print(f"{len(VIGNETTES)} vignettes, {len(tagged)} tagged, "
          f"{len(untagged)} always-eligible, {len(speaking)} with speech")
    print("motions:", dict(Counter(v.motion for v in VIGNETTES)))
    print("poses:  ", dict(Counter(v.pose for v in VIGNETTES)))
    print("tags:   ", dict(Counter(t for v in VIGNETTES for t in v.tags)))
    print(f"durations: {len(short)} short, {len(long_)} long")
    print("smoke test passed")
