"""Vignette content, band I - exploration and movement.

Thirty cosmetic beats about *going somewhere*, or thinking about it, or
failing at it: peering off at a ridge, scrambling up a slope and sliding back,
toeing an edge to see what will take a boot, pacing a distance out heel to
toe, doubling back halfway because something felt wrong. Nothing here scouts
for the AI or moves the colony's plans one inch - the utility system decides
where people actually go. This band is the *body language* of travelling,
which is what makes a walk read as a walk rather than as a position update.

Content only - the state machine, the picker and the contract all live in
``vignettes.py``. See :class:`~.vignettes.Vignette` for the field meanings and
the normalisation rules; everything below is written to sit comfortably inside
them rather than to be clamped into them.

House rules for this file
-------------------------
* Keys are all prefixed ``i_`` so the content bands cannot collide.
* Poses stay inside the set the renderer draws directly: ``idle``, ``walk``,
  ``carry``, ``chop``, ``build``, ``climb``, ``fall``, ``sleep``, ``dance``,
  ``mourn``, ``panic``. Movement leans on ``walk`` and ``climb``; the crouches
  borrow ``build`` because that is the renderer's folded-over body.
* ``drift`` is only meaningful under ``motion="pace"`` - that is the one motion
  the engine actually translates. Beats that bob in place (``hop``) or turn on
  the spot (``spin``) leave it at zero rather than declaring a budget the
  engine will never spend.
* Gating tags are ones :func:`~.vignettes.context_tags` really evaluates, so a
  ledge-edge beat needs an agent that is actually high up and a track-following
  beat needs snow to have left tracks in. Three entries carry the decorative
  tag "silly" instead: decorative tags never gate, so those stay available in
  any weather while their 0.15 weights keep them rare enough to still land.
* Short beats outnumber long ones. Most travelling hesitations are a second or
  two; only the surveying and the long way round earn five seconds.
"""
from __future__ import annotations

from .vignettes import Vignette

__all__ = ["VIGNETTES"]


VIGNETTES: list[Vignette] = [
    # ==================================================================
    #  Ungated - the everyday business of getting about. These fit any
    #  context, and are what the picker leans on when nothing special
    #  applies to where the agent is standing.
    # ==================================================================
    Vignette(
        # The whole band in one beat: look one way, look the other, commit
        # to neither. The spin is the vignette.
        "i_scout_left_right", "checks left, then right, then left again",
        "idle", (1.6, 2.6), 1.3, speech="?", motion="spin",
    ),
    Vignette(
        "i_up_on_toes", "goes up on its toes to see over something",
        "idle", (1.5, 2.4), 1.2, motion="still",
    ),
    Vignette(
        "i_leap_gap", "takes a short run-up and leaps a gap",
        "walk", (1.6, 2.6), 1.0, speech="!", motion="hop",
    ),
    Vignette(
        "i_run_then_stop", "breaks into a run and stops for no clear reason",
        "walk", (1.8, 2.8), 0.9, motion="pace", drift=28.0,
    ),
    Vignette(
        "i_scramble_slide", "scrambles a few steps up the slope and slides back",
        "climb", (1.9, 3.0), 1.0, motion="pace", drift=22.0,
    ),
    Vignette(
        "i_squeeze_through", "turns sideways to squeeze through a gap",
        "idle", (1.8, 2.8), 0.8, motion="still",
    ),
    Vignette(
        "i_sight_along_arm", "sights along an outstretched arm at something far off",
        "idle", (2.0, 3.0), 1.1, motion="still",
    ),
    Vignette(
        "i_double_back", "gets halfway, changes its mind, and doubles back",
        "walk", (2.2, 3.6), 1.0, motion="pace", drift=30.0,
    ),
    Vignette(
        "i_turn_full_circle", "turns a full circle working out which way is which",
        "idle", (2.2, 3.6), 1.0, speech="o", motion="spin",
    ),
    Vignette(
        "i_retrace_steps", "retraces three steps, hunting for what it dropped",
        # Bent-over shuffle: the build pose is the folded body, the pace is
        # the three steps back it is retracing.
        "build", (2.4, 3.8), 0.9, motion="pace", drift=20.0,
    ),
    Vignette(
        "i_two_ways_at_once", "sets off one way, then the other, then stands still",
        "walk", (2.4, 4.0), 0.7, motion="pace", drift=26.0,
    ),
    Vignette(
        "i_jog_short_lap", "jogs a short lap and comes back",
        "walk", (3.0, 5.0), 0.9, motion="pace", drift=36.0,
    ),
    Vignette(
        "i_zigzag_up", "zigzags up the steep part instead of taking it head on",
        "climb", (3.0, 5.0), 0.8, motion="pace", drift=34.0,
    ),
    Vignette(
        # Rare and slow on purpose - the joke only works if it is a while
        # before anyone sees it again.
        "i_shortcut_fails", "tries a shortcut and comes back the long way",
        "walk", (3.4, 5.5), 0.4, motion="pace", drift=38.0,
    ),

    # ==================================================================
    #  Decorative tags only - the ways travelling goes daft. They never
    #  gate, so they are always in the pool; the weights keep them rare.
    # ==================================================================
    Vignette(
        "i_measure_gap", "looks at a gap, then at its own legs, then at the gap",
        "idle", (2.2, 3.6), 0.15, tags=("silly",), speech="?", motion="still",
    ),
    Vignette(
        "i_slide_sit_down", "sits down and slides the last of the slope",
        "sleep", (1.8, 3.0), 0.15, tags=("silly",), speech="~", motion="sit",
    ),
    Vignette(
        "i_wrong_direction", "sets off confidently in entirely the wrong direction",
        "walk", (2.6, 4.2), 0.15, tags=("silly",), motion="pace", drift=36.0,
    ),

    # ==================================================================
    #  High up - where a wrong step actually costs something. Everything
    #  here is slower and more careful than the flat-ground equivalent.
    # ==================================================================
    Vignette(
        "i_test_footing_edge", "toes the edge, testing what will take its weight",
        "build", (2.0, 3.2), 0.9, tags=("high_up",), motion="crouch",
    ),
    Vignette(
        "i_edge_along_ledge", "edges along the ledge with its back to the rock",
        "climb", (3.0, 4.8), 0.7, tags=("high_up",), motion="pace", drift=24.0,
    ),

    # ==================================================================
    #  Terrain - water, snow, rain, trees, scree. What the ground makes
    #  a body do differently.
    # ==================================================================
    Vignette(
        "i_hop_stepping_stones", "hops from one stone to the next across the shallows",
        "walk", (2.2, 3.6), 0.9, tags=("near_water",), motion="hop",
    ),
    Vignette(
        "i_follow_own_tracks", "follows its own tracks back a little way",
        "walk", (2.6, 4.2), 0.8, tags=("snowing",), motion="pace", drift=28.0,
    ),
    Vignette(
        "i_dash_between_cover", "dashes from one patch of cover to the next",
        "walk", (1.8, 3.0), 0.9, tags=("raining",), motion="pace", drift=34.0,
    ),
    Vignette(
        "i_crawl_under_trunk", "flattens out and crawls under a fallen trunk",
        "sleep", (2.6, 4.2), 0.6, tags=("near_tree",), motion="lie",
    ),
    Vignette(
        "i_scree_slide", "rides a slide of loose scree a little way down",
        "fall", (1.8, 2.8), 0.5, tags=("near_rock",), speech="!", motion="pace",
        drift=30.0,
    ),

    # ==================================================================
    #  Dark - distance stops being a thing you can see and becomes a
    #  thing you have to remember.
    # ==================================================================
    Vignette(
        "i_peer_into_dark", "leans out into the dark to see how far the light reaches",
        "idle", (2.4, 4.0), 0.9, tags=("night", "candle"), speech="?",
        motion="still",
    ),
    Vignette(
        "i_backtrack_to_marker", "walks back to the last thing it recognised",
        "walk", (2.6, 4.2), 0.6, tags=("night", "alone"), motion="pace",
        drift=32.0,
    ),

    # ==================================================================
    #  Who is doing the moving - role and condition change the gait far
    #  more than the ground does.
    # ==================================================================
    Vignette(
        "i_lookout_sweep", "sweeps the treeline slowly, left to right",
        "idle", (3.0, 5.0), 1.0, tags=("lookout",), motion="spin",
    ),
    Vignette(
        "i_pace_out_builder", "paces a distance out, heel to toe",
        "walk", (3.0, 5.0), 0.9, tags=("builder",), motion="pace", drift=34.0,
    ),
    Vignette(
        "i_child_race", "races an imaginary rival to a rock, and wins",
        "walk", (2.2, 3.6), 0.6, tags=("child",), speech="*", motion="pace",
        drift=38.0,
    ),
    Vignette(
        "i_tired_trudge", "trudges a few steps and gives up on going further",
        "walk", (2.4, 4.0), 0.9, tags=("tired",), motion="pace", drift=20.0,
    ),
]


# ===========================================================================
#  Smoke test - content-side invariants only; the engine tests itself.
# ===========================================================================
if __name__ == "__main__":   # pragma: no cover
    from .vignettes import MAX_DRIFT, MAX_DUR, MIN_DUR, MOTIONS

    POSES = {"idle", "walk", "carry", "chop", "build", "climb", "fall",
             "sleep", "dance", "mourn", "panic"}

    assert len(VIGNETTES) == 30, len(VIGNETTES)
    keys = [v.key for v in VIGNETTES]
    assert len(set(keys)) == 30, "duplicate key"
    assert all(k.startswith("i_") for k in keys), "unprefixed key"
    assert all(MIN_DUR <= v.dur[0] <= v.dur[1] <= MAX_DUR for v in VIGNETTES)
    assert all(v.motion in MOTIONS for v in VIGNETTES)
    assert all(v.pose in POSES for v in VIGNETTES), "unknown pose"
    assert all(0.0 <= v.drift <= MAX_DRIFT for v in VIGNETTES)
    assert all(v.weight > 0.0 for v in VIGNETTES)
    assert all(v.speech is None or len(v.speech) == 1 for v in VIGNETTES)
    # drift is only spent by the pace motion; anything else declaring one is
    # a lie to whoever reads this file next.
    assert all(v.drift == 0.0 or v.motion == "pace" for v in VIGNETTES)

    tagged = [v for v in VIGNETTES if v.tags]
    ungated = [v for v in VIGNETTES if not v.gates]
    short = [v for v in VIGNETTES if v.dur[1] <= 3.0]
    long_ = [v for v in VIGNETTES if v.dur[1] > 4.5]
    spoken = [v for v in VIGNETTES if v.speech]
    assert 12 <= len(tagged) <= 18, len(tagged)
    assert len(ungated) >= 15, len(ungated)
    assert len(short) > len(long_), (len(short), len(long_))
    assert 6 <= len(spoken) <= 9, len(spoken)

    used = {v.motion for v in VIGNETTES}
    assert used == set(MOTIONS), set(MOTIONS) - used
    print(f"{len(VIGNETTES)} vignettes, {len(tagged)} tagged, "
          f"{len(ungated)} always-available, {len(spoken)} with speech")
    print("motions:", {m: sum(1 for v in VIGNETTES if v.motion == m)
                       for m in MOTIONS})
