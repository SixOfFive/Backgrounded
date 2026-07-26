"""Vignette content, band C - the social ones.

Waves, shrugs, pointing at something nobody else can see, a brief and
pointless argument, a staring contest, somebody copying a gesture badly. This
is the band that makes a colony read as a group of people rather than a set of
independently scheduled workers.

Everything here is pure data: 40 frozen :class:`~.vignettes.Vignette` records
in a module-level ``VIGNETTES`` list, which ``vignettes.load_all()`` picks up
and registers. Nothing in this file touches the world, and nothing in it runs
per frame - the engine owns all of that.

House rules this band follows
-----------------------------
* Keys are all prefixed ``c_`` so they cannot collide with bands A, B, D or E.
* Labels are third-person present phrases with no name in them; the chronicle
  renders them as "<Name> waves at a neighbour."
* Tags are gates, not decoration - anything in ``vignettes.KNOWN_TAGS``
  ("social", "night", "near_fire", ...) means the beat only fires when that is
  true. ``"silly"`` and ``"quiet"`` are *not* known tags, so they are free
  labels for grouping and never narrow when a vignette can play. Twelve
  entries carry no tag at all, so there is always something pickable for a
  stickman standing alone in the dark.
* Poses are chosen for what the poser actually draws. ``sleep`` genuinely lies
  the figure down (``lie=True``), so it is used only for the ``lie`` beat;
  sitting and crouching both borrow ``build``, which is the low hunched body.
  Poses whose animation phase comes from ``anim_t`` (``walk``, ``chop``) are
  paired only with ``pace``/``hop``, the two motions that keep that phase
  moving - a "walk" pose on a motionless agent would freeze mid-stride.
* Weights: an ordinary beat sits near 1.0; the odd or once-in-a-while ones
  (a missed high five, a wobbled bow, an interminable lecture) sit well below,
  down to 0.15, so they stay a surprise.

No pygame, no world mutation, no imports beyond the engine's own dataclass.
"""
from __future__ import annotations

from .vignettes import MAX_DRIFT, MAX_DUR, MIN_DUR, MOTIONS, Vignette

__all__ = ["VIGNETTES"]


VIGNETTES: list[Vignette] = [
    # ================================================================== #
    #  Ungated - gestures that need no audience to be worth doing.
    #  These carry no tags at all and are the band's always-available floor.
    # ================================================================== #
    Vignette(
        key="c_point_yonder",
        label="points at something over there",
        pose="idle",
        dur=(1.8, 3.0),
        weight=1.2,
        speech="!",
        motion="still",
    ),
    Vignette(
        key="c_nod_along",
        label="nods along to nothing in particular",
        pose="idle",
        dur=(1.5, 2.4),
        weight=1.3,
        motion="still",
    ),
    Vignette(
        key="c_shrug",
        label="shrugs at the whole business",
        pose="idle",
        dur=(1.5, 2.4),
        weight=1.4,
        motion="still",
    ),
    Vignette(
        key="c_laughs",
        label="laughs at something nobody said",
        pose="dance",
        dur=(1.6, 2.6),
        weight=1.1,
        motion="hop",
    ),
    Vignette(
        key="c_applauds",
        label="applauds something very small",
        pose="idle",
        dur=(1.6, 2.6),
        weight=0.9,
        motion="still",
    ),
    Vignette(
        key="c_bow",
        label="bows to no one in particular",
        pose="build",
        dur=(1.7, 2.8),
        weight=0.5,
        motion="crouch",
    ),
    Vignette(
        key="c_beckons",
        label="beckons at somebody who is not looking",
        pose="idle",
        dur=(1.6, 2.8),
        weight=1.0,
        motion="still",
    ),
    Vignette(
        key="c_gesture_horizon",
        label="turns about, gesturing at the whole horizon",
        pose="idle",
        dur=(2.2, 3.6),
        weight=0.6,
        motion="spin",
    ),
    Vignette(
        key="c_small_speech",
        label="makes a small speech to the middle distance",
        pose="talk",
        dur=(3.0, 5.5),
        weight=0.45,
        speech="~",
        motion="still",
    ),
    Vignette(
        key="c_thumbs_up",
        label="gives somebody a small thumbs up",
        pose="idle",
        dur=(1.5, 2.2),
        weight=1.0,
        motion="still",
    ),
    Vignette(
        key="c_pace_and_explain",
        label="paces about explaining something",
        pose="walk",
        dur=(3.0, 5.0),
        weight=0.7,
        motion="pace",
        drift=30.0,
    ),
    Vignette(
        key="c_sits_to_watch",
        label="sits down to watch the others",
        pose="build",
        dur=(4.0, 7.0),
        weight=0.55,
        motion="sit",
    ),

    # ================================================================== #
    #  Free labels only ("silly", "quiet" are not KNOWN_TAGS) - these are
    #  grouped for the author's benefit and still fire anywhere.
    # ================================================================== #
    Vignette(
        key="c_nod_twice",
        label="nods twice and says nothing",
        pose="idle",
        dur=(1.5, 2.2),
        weight=1.1,
        tags=("quiet",),
        motion="still",
    ),
    Vignette(
        key="c_double_shrug",
        label="shrugs twice, for emphasis",
        pose="idle",
        dur=(2.0, 3.0),
        weight=0.45,
        tags=("silly",),
        speech="?",
        motion="still",
    ),
    Vignette(
        key="c_bow_wobble",
        label="attempts a deep bow and wobbles",
        pose="build",
        dur=(1.8, 3.0),
        weight=0.15,
        tags=("silly",),
        motion="crouch",
    ),

    # ================================================================== #
    #  Greetings - the cheapest way to say two stickmen know each other.
    # ================================================================== #
    Vignette(
        key="c_wave",
        label="waves at a neighbour",
        pose="idle",
        dur=(1.6, 2.6),
        weight=1.4,
        tags=("social",),
        speech="!",
        motion="still",
    ),
    Vignette(
        key="c_wave_at_nobody",
        label="waves at nobody at all",
        pose="idle",
        dur=(1.8, 3.0),
        weight=0.3,
        tags=("alone",),
        speech="?",
        motion="still",
    ),
    Vignette(
        key="c_high_five",
        label="goes in for a high five",
        pose="idle",
        dur=(1.5, 2.4),
        weight=1.0,
        tags=("social",),
        speech="!",
        motion="hop",
    ),
    Vignette(
        key="c_pat_on_back",
        label="pats somebody on the back",
        pose="idle",
        dur=(1.5, 2.4),
        weight=1.0,
        tags=("social",),
        motion="still",
    ),

    # ================================================================== #
    #  Showing and telling.
    # ================================================================== #
    Vignette(
        key="c_point_and_check",
        label="points, then checks whether anybody looked",
        pose="idle",
        dur=(2.2, 3.6),
        weight=0.7,
        tags=("social",),
        motion="spin",
    ),
    Vignette(
        key="c_point_at_moon",
        label="points up at the moon until somebody looks",
        pose="idle",
        dur=(2.6, 4.4),
        weight=0.45,
        tags=("night", "social"),
        speech="*",
        motion="still",
    ),
    Vignette(
        key="c_talks_with_hands",
        label="talks with its hands",
        pose="talk",
        dur=(2.4, 4.0),
        weight=1.2,
        tags=("social",),
        motion="still",
    ),
    Vignette(
        key="c_holds_forth",
        label="holds forth at considerable length",
        pose="walk",
        dur=(5.0, 8.0),
        weight=0.25,
        tags=("social",),
        speech="~",
        motion="pace",
        drift=24.0,
    ),
    Vignette(
        key="c_share_nothing",
        label="shares something invisible",
        pose="carry",
        dur=(2.0, 3.2),
        weight=0.75,
        tags=("social",),
        motion="still",
    ),
    Vignette(
        key="c_teach_gesture",
        label="teaches somebody a gesture",
        pose="talk",
        dur=(3.0, 5.0),
        weight=0.55,
        tags=("social", "elder"),
        motion="still",
    ),

    # ================================================================== #
    #  Copying, following, playing.
    # ================================================================== #
    Vignette(
        key="c_mimic_walk",
        label="mimics somebody else's walk",
        pose="walk",
        dur=(2.6, 4.2),
        weight=0.5,
        tags=("social", "silly"),
        motion="pace",
        drift=26.0,
    ),
    Vignette(
        key="c_copy_gesture_badly",
        label="copies a gesture badly",
        pose="idle",
        dur=(2.0, 3.2),
        weight=0.4,
        tags=("social", "child"),
        speech="?",
        motion="hop",
    ),
    Vignette(
        key="c_play_fight",
        label="play-fights with a neighbour",
        pose="chop",
        dur=(2.2, 3.6),
        weight=0.8,
        tags=("social",),
        motion="hop",
    ),
    Vignette(
        key="c_follow_a_few_steps",
        label="follows somebody a few steps, then thinks better of it",
        pose="walk",
        dur=(2.6, 4.2),
        weight=0.9,
        tags=("social",),
        motion="pace",
        drift=34.0,
    ),
    Vignette(
        key="c_dance_it_out",
        label="starts a dance and waits for somebody to join in",
        pose="dance",
        dur=(3.0, 5.0),
        weight=0.5,
        tags=("social", "happy"),
        motion="hop",
    ),
    Vignette(
        key="c_missed_five",
        label="misses a high five completely",
        pose="idle",
        dur=(1.8, 2.8),
        weight=0.15,
        tags=("social", "silly"),
        motion="still",
    ),

    # ================================================================== #
    #  Friction - the colony is not all waving.
    # ================================================================== #
    Vignette(
        key="c_brief_argument",
        label="has a brief and pointless argument",
        pose="talk",
        dur=(2.4, 4.0),
        weight=0.7,
        tags=("social",),
        speech="!",
        motion="pace",
        drift=20.0,
    ),
    Vignette(
        key="c_lost_argument",
        label="loses an argument and mutters about it",
        pose="mourn",
        dur=(2.4, 3.8),
        weight=0.3,
        tags=("social", "sad"),
        motion="still",
    ),
    Vignette(
        key="c_turn_away_huff",
        label="turns away in a huff",
        pose="idle",
        dur=(1.8, 3.0),
        weight=0.45,
        tags=("social", "sad"),
        motion="spin",
    ),
    Vignette(
        key="c_staring_contest",
        label="holds a staring contest",
        pose="idle",
        dur=(3.0, 6.0),
        weight=0.3,
        tags=("social",),
        motion="still",
    ),
    Vignette(
        key="c_queue_politely",
        label="queues politely behind somebody",
        pose="idle",
        dur=(2.6, 4.4),
        weight=0.6,
        tags=("social",),
        motion="still",
    ),

    # ================================================================== #
    #  Kindness, and sitting near people.
    # ================================================================== #
    Vignette(
        key="c_offer_hand",
        label="offers a hand up to somebody sitting down",
        pose="build",
        dur=(1.8, 2.8),
        weight=0.8,
        tags=("social",),
        motion="crouch",
    ),
    Vignette(
        key="c_comforts_neighbour",
        label="comforts a neighbour",
        pose="build",
        dur=(3.0, 5.0),
        weight=0.85,
        tags=("social",),
        motion="crouch",
    ),
    Vignette(
        key="c_sits_beside",
        label="sits down beside somebody and says nothing",
        pose="build",
        dur=(4.0, 7.0),
        weight=0.6,
        tags=("social", "quiet"),
        motion="sit",
    ),
    Vignette(
        key="c_fireside_shuffle",
        label="shuffles up to share the fire",
        pose="idle",
        dur=(2.4, 4.0),
        weight=0.9,
        tags=("near_fire", "social"),
        motion="pace",
        drift=18.0,
    ),
    Vignette(
        key="c_lie_and_talk",
        label="lies back and carries on talking",
        pose="sleep",
        dur=(4.5, 7.5),
        weight=0.2,
        tags=("social", "night", "tired"),
        speech="~",
        motion="lie",
    ),
]


# ===========================================================================
#  Smoke test - the band's own contract, checkable without the rest of the app.
# ===========================================================================
if __name__ == "__main__":   # pragma: no cover - headless smoke test
    from collections import Counter

    assert len(VIGNETTES) == 40, len(VIGNETTES)
    keys = [v.key for v in VIGNETTES]
    assert len(set(keys)) == 40, "duplicate key in band C"
    assert all(k.startswith("c_") for k in keys), [k for k in keys
                                                  if not k.startswith("c_")]
    assert all(MIN_DUR <= v.dur[0] <= v.dur[1] <= MAX_DUR for v in VIGNETTES)
    assert all(v.motion in MOTIONS for v in VIGNETTES)
    assert all(0.0 <= v.drift <= MAX_DRIFT for v in VIGNETTES)
    assert all(v.weight > 0.0 for v in VIGNETTES)
    # A pace beat with no drift budget just stands there; a drifting beat that
    # never paces never spends it. Either is a content bug, not an engine one.
    assert all((v.drift > 0.0) == (v.motion == "pace") for v in VIGNETTES), [
        v.key for v in VIGNETTES if (v.drift > 0.0) != (v.motion == "pace")]

    tagged = [v for v in VIGNETTES if v.tags]
    ungated = [v for v in VIGNETTES if not v.gates]
    short = [v for v in VIGNETTES if v.dur[1] <= 3.0]
    long_ = [v for v in VIGNETTES if v.dur[0] >= 3.0]
    speaks = [v for v in VIGNETTES if v.speech]
    rare = [v for v in VIGNETTES if v.weight < 0.4]

    assert len(tagged) >= 20, len(tagged)
    assert len(ungated) >= 10, len(ungated)          # always something to pick
    assert len(short) > len(long_), (len(short), len(long_))
    assert 8 <= len(speaks) <= 13, len(speaks)
    assert len(set(v.motion for v in VIGNETTES)) == len(MOTIONS)

    print(f"band C: {len(VIGNETTES)} vignettes")
    print(f"  tagged {len(tagged)}, ungated {len(ungated)}, "
          f"speaking {len(speaks)}, rare {len(rare)}")
    print(f"  short {len(short)}, long {len(long_)}")
    print("  motions:", dict(Counter(v.motion for v in VIGNETTES)))
    print("  poses:  ", dict(Counter(v.pose for v in VIGNETTES)))
    print("  tags:   ", dict(Counter(t for v in VIGNETTES for t in v.tags)))
    print("smoke test passed")
