"""Vignette content, band F - work and craft mimicry.

Thirty cosmetic beats about *hands that know a trade*: sharpening, whittling,
plaiting, knapping, threading, measuring a gap with two arms and believing the
answer. None of it produces anything. No stone is really sharpened, no cord is
really plaited, the stockpile is exactly as large after the count as before it -
the point is that a colony whose people only ever swing at a tree looks like a
machine, and a colony where someone stops to hold a tool up to the light and
turn it over does not.

The distinction this band lives on: *real* work is an ``Action`` and belongs to
``actions.py``. What is here is the shape of work performed for nobody, in the
gaps between jobs - the fidget of a competent pair of hands with nothing urgent
to do. If a beat below would change a single number in the world, it is written
wrong.

Content only; the state machine, the picker and the contract live in
``vignettes.py``. See :class:`~.vignettes.Vignette` for field meanings.

House rules for this file
-------------------------
* Keys are all prefixed ``f_`` so content bands cannot collide.
* Poses come from the eleven bodies ``render.stickfigure.POSES`` actually
  draws: fine handwork reads as ``build``, edge work as ``chop``, anything
  held out and studied as ``carry``.
* Gating tags are ones :func:`~.vignettes.context_tags` evaluates, and they are
  chosen to *mean* something: flint gets knapped near rock, tools get dried by
  fire, fine work happens by candlelight at night. Two entries carry a purely
  decorative tag ("quiet", "silly") instead - those never gate, so the
  ungated pool cannot run dry.
* Short beats outnumber long ones. A craftsman checks an edge far more often
  than it lies down under a frame to squint at a joint.
"""
from __future__ import annotations

from .vignettes import Vignette

__all__ = ["VIGNETTES"]


VIGNETTES: list[Vignette] = [
    # ==================================================================
    #  Ungated - the trade done anywhere, on nothing, for no reason.
    #  These are what the picker has left when the context is bare.
    # ==================================================================
    Vignette(
        "f_sharpen_stone", "sharpens a stone on another stone", "chop",
        (1.8, 3.0), 1.3, motion="still",
    ),
    Vignette(
        # The hop is the whole gag: it is tapping with its heel because both
        # hands are busy holding the peg it is pretending to hold.
        "f_hammer_peg", "hammers a peg flat", "build",
        (1.8, 3.0), 1.1, motion="hop",
    ),
    Vignette(
        "f_measure_gap", "measures a gap with both arms", "idle",
        (1.6, 2.8), 1.2, speech="?", motion="still",
    ),
    Vignette(
        "f_inspect_tool", "turns a tool over, hunting for cracks", "carry",
        (1.8, 3.0), 1.2, motion="spin",
    ),
    Vignette(
        "f_test_rope", "leans back on a rope to see if it holds", "climb",
        (1.6, 2.8), 1.0, motion="still",
    ),
    Vignette(
        "f_notch_tally", "cuts a fresh notch and counts the old ones", "chop",
        (2.0, 3.4), 0.9, motion="still",
    ),
    Vignette(
        "f_bundle_sticks", "bundles sticks and ties the bundle off", "carry",
        (2.4, 4.0), 1.1, motion="crouch",
    ),
    Vignette(
        "f_pace_out_length", "paces out a length, heel to toe", "walk",
        (3.0, 5.0), 1.0, motion="pace", drift=34.0,
    ),
    Vignette(
        "f_plait_cord", "plaits three strands into one cord", "build",
        (3.4, 5.6), 0.9, motion="sit",
    ),
    Vignette(
        "f_whittle_stick", "whittles a stick down to nothing much", "build",
        (3.5, 6.0), 0.8, tags=("quiet",), motion="sit",
    ),
    Vignette(
        "f_sand_it_smooth", "rubs a rough edge smooth", "build",
        (2.6, 4.4), 0.9, motion="crouch",
    ),
    Vignette(
        "f_thread_fibre", "threads fibre through a hole too small for it",
        "idle", (2.0, 3.4), 0.8, motion="still",
    ),
    Vignette(
        # Rare on purpose. Once in a long while is funny; often is a colony
        # of people who cannot hold a hammer.
        "f_hammer_own_thumb", "hammers its own thumb", "panic",
        (1.5, 2.4), 0.15, tags=("silly",), speech="!", motion="hop",
    ),

    # ==================================================================
    #  Material to hand - the job the surroundings suggest.
    # ==================================================================
    Vignette(
        "f_knap_flint", "knaps a flake off a flint", "chop",
        (1.8, 3.0), 1.1, tags=("near_rock",), speech="!", motion="crouch",
    ),
    Vignette(
        "f_sort_pebbles", "sorts pebbles by size, then starts again by colour",
        "build", (3.2, 5.4), 0.8, tags=("near_rock", "alone"), motion="crouch",
    ),
    Vignette(
        "f_wet_the_stone", "wets a stone before working it", "build",
        (1.8, 3.0), 1.0, tags=("near_water",), motion="crouch",
    ),
    Vignette(
        "f_bark_cordage", "peels a strip of bark and tests how it bends",
        "chop", (2.4, 4.0), 1.0, tags=("near_tree",), motion="still",
    ),
    Vignette(
        "f_split_kindling", "splits kindling into smaller kindling", "chop",
        (2.4, 4.2), 1.0, tags=("near_fire",), motion="crouch",
    ),
    Vignette(
        # Wet weather plus a fire is the only time drying a tool reads as
        # sense rather than fussing.
        "f_dry_the_tool", "turns a damp tool over by the fire", "idle",
        (2.4, 4.0), 1.0, tags=("near_fire", "raining"), motion="still",
    ),

    # ==================================================================
    #  Conditions - light, weather and cold hands.
    # ==================================================================
    Vignette(
        "f_candle_fine_work", "squints at fine work by candlelight", "build",
        (3.5, 6.0), 0.8, tags=("night", "candle"), motion="sit",
    ),
    Vignette(
        "f_pack_up_the_work", "gathers its work up before the storm takes it",
        "carry", (1.8, 3.0), 1.0, tags=("storm",), speech="!", motion="still",
    ),
    Vignette(
        "f_frozen_fingers", "picks at a knot with fingers that will not bend",
        "idle", (1.8, 3.0), 1.0, tags=("cold",), speech="~", motion="still",
    ),
    Vignette(
        "f_sharpen_to_nothing", "sharpens a point until there is no point left",
        "chop", (2.6, 4.2), 0.15, tags=("tired",), speech="o", motion="still",
    ),

    # ==================================================================
    #  Trades - the same hands, told apart by what they were taught.
    # ==================================================================
    Vignette(
        "f_check_the_joint", "wiggles a joint to see whether it holds", "build",
        (1.8, 3.0), 1.1, tags=("builder",), motion="crouch",
    ),
    Vignette(
        # The longest beat in the band, and the only one flat on its back:
        # the only way to see the underside of a joint is from underneath it.
        "f_under_the_frame", "lies down to look up at a joint from underneath",
        "sleep", (4.0, 7.0), 0.5, tags=("builder",), motion="lie",
    ),
    Vignette(
        "f_count_the_stockpile", "counts the stockpile and gets a different number",
        "idle", (2.5, 4.2), 0.9, tags=("gatherer",), speech="?", motion="pace",
        drift=22.0,
    ),
    Vignette(
        "f_hone_on_watch", "hones a point without taking its eyes off the ridge",
        "chop", (2.6, 4.4), 0.9, tags=("lookout",), motion="still",
    ),
    Vignette(
        "f_copy_the_grip", "copies the way a grown-up holds a tool", "chop",
        (1.8, 3.0), 0.8, tags=("child",), motion="still",
    ),
    Vignette(
        "f_redo_the_knot", "unties a knot that was fine and ties it again",
        "build", (2.6, 4.4), 0.7, tags=("elder",), motion="crouch",
    ),
    Vignette(
        "f_show_the_knot", "shows someone how the knot goes", "build",
        (2.4, 4.0), 1.0, tags=("social",), speech="*", motion="still",
    ),
]


# ===========================================================================
#  Smoke test - content-side invariants only; the engine tests itself.
# ===========================================================================
if __name__ == "__main__":   # pragma: no cover
    from .vignettes import KNOWN_TAGS, MAX_DRIFT, MAX_DUR, MIN_DUR, MOTIONS

    assert len(VIGNETTES) == 30, len(VIGNETTES)
    keys = [v.key for v in VIGNETTES]
    assert len(set(keys)) == 30, "duplicate key"
    assert all(k.startswith("f_") for k in keys), "unprefixed key"
    assert all(MIN_DUR <= v.dur[0] <= v.dur[1] <= MAX_DUR for v in VIGNETTES)
    assert all(v.motion in MOTIONS for v in VIGNETTES)
    assert all(0.0 <= v.drift <= MAX_DRIFT for v in VIGNETTES)
    assert all(v.weight > 0.0 for v in VIGNETTES)
    assert all(v.speech is None or len(v.speech) == 1 for v in VIGNETTES)
    assert all(v.drift == 0.0 or v.motion == "pace" for v in VIGNETTES)
    assert all(v.pose in ("idle", "walk", "carry", "chop", "build", "climb",
                          "fall", "sleep", "dance", "mourn", "panic")
               for v in VIGNETTES)

    tagged = [v for v in VIGNETTES if v.tags]
    ungated = [v for v in VIGNETTES if not v.gates]
    short = [v for v in VIGNETTES if v.dur[1] <= 3.0]
    long_ = [v for v in VIGNETTES if v.dur[1] > 4.5]
    spoken = [v for v in VIGNETTES if v.speech]
    assert 12 <= len(tagged) <= 20, len(tagged)
    assert len(ungated) >= 12, len(ungated)
    assert len(short) > len(long_), (len(short), len(long_))
    assert 5 <= len(spoken) <= 10, len(spoken)

    gates = {t for v in VIGNETTES for t in v.gates}
    assert gates <= KNOWN_TAGS, gates - KNOWN_TAGS
    used = {v.motion for v in VIGNETTES}
    assert used == set(MOTIONS), set(MOTIONS) - used

    print(f"{len(VIGNETTES)} vignettes, {len(tagged)} tagged, "
          f"{len(ungated)} always-available, {len(spoken)} with speech")
    print("gates:", sorted(gates))
    print("motions:", {m: sum(1 for v in VIGNETTES if v.motion == m)
                       for m in MOTIONS})
