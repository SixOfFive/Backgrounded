"""Vignette content, band L - maintenance and tidying.

Thirty cosmetic beats about *keeping things straight*: sweeping a patch of
ground, righting a leaning post, patting loose dirt flat, coiling a rope,
squaring off a pile until its edges line up. None of it does anything. The
stockpile is not touched, no structure is repaired, no prop moves; a colony
that has just been swept is byte-for-byte the colony it was before.

That is the point. Work in this colony is a shuttle - fetch, carry, drop -
and a shuttle looks like a shuttle. Someone who stops to straighten a post
nobody asked about, or who brushes the dust off the same rock twice, is
somebody who *lives* here rather than somebody routed through here. Tidying
is the cheapest possible signal of ownership.

Content only - the state machine, the picker and the contract all live in
``vignettes.py``.

House rules for this file
-------------------------
* Keys are all prefixed ``l_``.
* Gating tags are ones :func:`~.vignettes.context_tags` really evaluates, so
  they *mean* something: you sweep snow when it is snowing, you rake embers
  next to a fire, you feel a wall for gaps if you are the builder. Two entries
  carry only decorative tags ("silly", "quiet"), which never gate, so the pool
  is never empty however mild the weather.
* Short beats outnumber long ones roughly four to one. Tidying is mostly a
  two-second correction on the way past something, not a project.
* The crouch is over-represented on purpose - almost all of this work happens
  at ground level, and the silhouette of a bent back is the band's signature.
"""
from __future__ import annotations

from .vignettes import Vignette

__all__ = ["VIGNETTES"]


VIGNETTES: list[Vignette] = [
    # ==================================================================
    #  Ungated - the everyday corrections. These fit any context and are
    #  the band's bread and butter: a hand put out to fix one small thing
    #  in passing.
    # ==================================================================
    Vignette(
        # The long one of the ungated set: an actual sweep, edge to edge.
        "l_sweep_patch", "sweeps a patch of ground clear with a bundle of twigs",
        "chop", (2.4, 4.0), 1.3, motion="pace", drift=18.0,
    ),
    Vignette(
        "l_pat_dirt_flat", "pats loose dirt down with both palms", "build",
        (1.6, 2.8), 1.3, motion="crouch",
    ),
    Vignette(
        "l_tamp_heel", "tamps a soft patch down with its heel", "walk",
        (1.8, 3.0), 1.2, motion="hop",
    ),
    Vignette(
        "l_clear_twigs", "clears the twigs off the path, one at a time", "carry",
        (2.4, 4.0), 1.2, motion="pace", drift=24.0,
    ),
    Vignette(
        "l_pluck_weed", "pulls a weed up and flicks it away", "build",
        (1.5, 2.6), 1.2, motion="crouch",
    ),
    Vignette(
        "l_scrape_feet", "scrapes the mud off its feet against a stone", "idle",
        (1.6, 2.8), 1.2, motion="hop",
    ),
    Vignette(
        "l_wipe_surface", "wipes a surface down with a handful of grass", "build",
        (1.8, 3.0), 1.1, motion="still",
    ),
    Vignette(
        # Arms overhead to walk the post upright - "climb" is that shape.
        "l_straighten_post", "straightens a leaning post", "climb",
        (1.8, 3.0), 1.2, motion="still",
    ),
    Vignette(
        "l_prop_it_up", "props up something that was about to go over", "build",
        (1.8, 3.0), 1.1, speech="!", motion="still",
    ),
    Vignette(
        "l_coil_rope", "coils a rope around its elbow", "carry",
        (2.4, 4.0), 1.1, motion="sit",
    ),
    Vignette(
        "l_line_pebbles", "sets pebbles in a line along the edge of the path",
        "build", (3.0, 5.0), 0.9, motion="crouch",
    ),
    Vignette(
        "l_shake_and_fold", "shakes something out and folds it over its arm",
        "carry", (1.8, 3.0), 1.0, motion="spin",
    ),

    # ==================================================================
    #  Decorative tags only - the two that never gate. One is quietly
    #  obsessive, one is plainly futile; both are rare on weight alone.
    # ==================================================================
    Vignette(
        # Cheek to the dirt, one eye shut, checking the ground is true.
        "l_sight_level", "lies down to sight along the ground for a dip",
        "sleep", (2.8, 4.5), 0.2, tags=("alone", "quiet"), speech="?",
        motion="lie",
    ),
    Vignette(
        "l_half_tidy", "starts tidying, then gives up halfway", "mourn",
        (2.4, 4.0), 0.9, tags=("tired",), speech="~", motion="crouch",
    ),

    # ==================================================================
    #  Place - what is to hand decides what gets straightened.
    # ==================================================================
    Vignette(
        "l_brush_rock", "brushes the dust off a rock", "build",
        (1.6, 2.8), 1.2, tags=("near_rock",), motion="crouch",
    ),
    Vignette(
        "l_rinse_grit", "rinses the grit off its hands", "build",
        (1.8, 3.0), 1.1, tags=("near_water",), motion="crouch",
    ),
    Vignette(
        # Sweeping water. It does not get less wet. It sweeps anyway.
        "l_sweep_puddle", "sweeps at a puddle, achieving nothing", "chop",
        (2.6, 4.2), 0.15, tags=("near_water", "silly"), speech="o",
        motion="still",
    ),
    Vignette(
        "l_rake_embers", "rakes the embers into a neater heap", "build",
        (2.4, 4.0), 1.0, tags=("near_fire", "night"), motion="crouch",
    ),
    Vignette(
        "l_clear_under_tree", "clears the fallen wood out from under a tree",
        "carry", (1.6, 2.8), 1.0, tags=("near_tree",), motion="spin",
    ),
    Vignette(
        "l_clear_ledge", "sweeps the loose grit off the ledge", "build",
        (1.8, 3.0), 0.9, tags=("high_up",), motion="crouch",
    ),

    # ==================================================================
    #  Weather - tidying as a small argument with the sky.
    # ==================================================================
    Vignette(
        "l_sweep_snow", "sweeps the snow off a flat stone", "chop",
        (1.8, 3.0), 1.1, tags=("snowing", "cold"), motion="still",
    ),
    Vignette(
        "l_rain_channel", "scrapes a channel to send the rain somewhere else",
        "build", (2.6, 4.2), 1.0, tags=("raining",), motion="crouch",
    ),
    Vignette(
        # Hurried, because the gust is already here.
        "l_weigh_down", "weighs something down before the wind takes it",
        "panic", (1.8, 2.9), 1.0, tags=("storm",), speech="!", motion="crouch",
    ),
    Vignette(
        "l_chase_leaves", "picks up one blown leaf, then another, then another",
        "carry", (3.0, 5.0), 0.15, tags=("windy", "silly"), speech="*",
        motion="pace", drift=30.0,
    ),

    # ==================================================================
    #  Trade - the same instinct, pointed at whatever the job is.
    # ==================================================================
    Vignette(
        "l_plumb_string", "holds a string up to see whether it hangs straight",
        "build", (1.8, 3.0), 1.0, tags=("builder",), speech="?", motion="still",
    ),
    Vignette(
        "l_walk_the_wall", "walks the wall, feeling along it for gaps", "build",
        (3.0, 5.0), 0.9, tags=("builder",), motion="pace", drift=34.0,
    ),
    Vignette(
        "l_square_stockpile", "squares the stockpile off so it sits straight",
        "carry", (3.0, 5.0), 1.0, tags=("gatherer",), motion="crouch",
    ),
    Vignette(
        "l_clear_sightline", "clears the brush out of its own sightline", "chop",
        (2.2, 3.6), 0.9, tags=("lookout",), motion="still",
    ),

    # ==================================================================
    #  Age - the two ends of the colony, tidying very differently.
    # ==================================================================
    Vignette(
        "l_child_one_twig", "carries one twig very carefully to the pile",
        "carry", (2.4, 4.0), 0.8, tags=("child",), speech="*", motion="pace",
        drift=20.0,
    ),
    Vignette(
        "l_elder_inspection", "walks the length of the work, nodding at it",
        "walk", (3.5, 5.5), 0.8, tags=("elder", "day"), motion="pace",
        drift=32.0,
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
    assert all(k.startswith("l_") for k in keys), "unprefixed key"
    assert all(MIN_DUR <= v.dur[0] <= v.dur[1] <= MAX_DUR for v in VIGNETTES)
    assert all(v.motion in MOTIONS for v in VIGNETTES)
    assert all(0.0 <= v.drift <= MAX_DRIFT for v in VIGNETTES)
    assert all(v.weight > 0.0 for v in VIGNETTES)
    assert all(v.speech is None or len(v.speech) == 1 for v in VIGNETTES)

    poses = {"idle", "walk", "carry", "chop", "build", "climb", "fall",
             "sleep", "dance", "mourn", "panic"}
    assert all(v.pose in poses for v in VIGNETTES), \
        sorted({v.pose for v in VIGNETTES} - poses)

    tagged = [v for v in VIGNETTES if v.tags]
    ungated = [v for v in VIGNETTES if not v.gates]
    short = [v for v in VIGNETTES if v.dur[1] <= 3.0]
    long_ = [v for v in VIGNETTES if v.dur[1] > 4.5]
    assert len(tagged) >= 14, len(tagged)
    assert len(ungated) >= 12, len(ungated)
    assert len(short) > len(long_), (len(short), len(long_))

    used = {v.motion for v in VIGNETTES}
    assert used == set(MOTIONS), set(MOTIONS) - used
    print(f"{len(VIGNETTES)} vignettes, {len(tagged)} tagged, "
          f"{len(ungated)} always-available, "
          f"{sum(1 for v in VIGNETTES if v.speech)} with speech")
    print("motions:", {m: sum(1 for v in VIGNETTES if v.motion == m)
                       for m in MOTIONS})
