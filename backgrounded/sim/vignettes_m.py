"""Vignette content, band M - signals, and the trouble with sending them.

Thirty cosmetic beats about *trying to be understood*: both arms over the head
because a shout will not carry that far, a candle swung in a slow arc because
it is dark, an arrow of stones because the person it is meant for has not
arrived yet. Where band A is a body and band B is a place, this band is the gap
between two stickmen and the arm-waving that tries to cross it.

Nothing here changes the world, and - importantly - nothing here changes the
*other* stickman either. A vignette may only touch the acting agent's pose,
facing, speech and a few px of x, so a beckon does not summon anybody and a
"stop there" does not stop anybody. Half of these signals are therefore
unanswered by construction, which is fine: an unanswered signal is still the
most alive thing a figure on a wallpaper can do.

Content only - the dataclass, the picker and the state machine all live in
``vignettes.py``.

House rules for this file
-------------------------
* Keys are all prefixed ``m_`` so the content bands cannot collide.
* Gating tags are ones :func:`~.vignettes.context_tags` really evaluates, and
  they are chosen so the *context* is what makes the gesture legible: the
  candle signal wants night and a candle, the all-clear wants a lookout, the
  "stay back" wants a storm to stay back from.
* Roughly half the band carries no tags at all. Those are the plain signals -
  wave, point, whistle, beckon - and they keep the pool healthy in weather and
  moods the tagged half cannot reach.
* Short beats outnumber long ones. Most signals are one gesture; only the mimed
  explanations and the two lying-down ones run past four seconds.
"""
from __future__ import annotations

from .vignettes import Vignette

__all__ = ["VIGNETTES"]


VIGNETTES: list[Vignette] = [
    # ==================================================================
    #  Ungated - the plain signals. No context makes these strange, so
    #  they are what the picker reaches for when nothing else fits.
    # ==================================================================
    Vignette(
        # Both arms up *and* bouncing: the universal "over here, look at me".
        key="m_wave_both_arms",
        label="waves both arms over its head",
        pose="climb",
        dur=(1.6, 2.8),
        weight=1.3,
        motion="hop",
    ),
    Vignette(
        key="m_point_at_sky",
        label="points at the sky and holds the point",
        pose="climb",
        dur=(1.8, 3.0),
        weight=1.1,
        motion="still",
    ),
    Vignette(
        key="m_cup_hands_and_call",
        label="cups both hands and calls somebody's name",
        pose="idle",
        dur=(2.0, 3.4),
        weight=1.2,
        speech="!",
        motion="still",
    ),
    Vignette(
        key="m_whistle_two_notes",
        label="whistles two notes and waits for the answer",
        pose="idle",
        dur=(2.0, 3.2),
        weight=1.2,
        speech="~",
        motion="still",
    ),
    Vignette(
        key="m_beckon_over",
        label="beckons somebody over with one arm",
        pose="idle",
        dur=(1.6, 2.8),
        weight=1.3,
        motion="still",
    ),
    Vignette(
        key="m_flat_hand_stop",
        label="holds a flat hand up: stop there",
        pose="idle",
        dur=(1.5, 2.6),
        weight=1.1,
        motion="still",
    ),
    Vignette(
        # The spin is the head turning from one refusal to the next.
        key="m_shake_head_no",
        label="shakes its head, no, and then no again",
        pose="idle",
        dur=(1.8, 3.0),
        weight=1.0,
        motion="spin",
    ),
    Vignette(
        key="m_cup_an_ear",
        label="cups an ear toward something far away",
        pose="idle",
        dur=(1.8, 3.0),
        weight=1.2,
        speech="?",
        motion="still",
    ),
    Vignette(
        key="m_count_down_hand",
        label="counts down from five on one hand",
        pose="idle",
        dur=(1.8, 3.0),
        weight=0.9,
        motion="still",
    ),
    Vignette(
        key="m_semaphore_arms",
        label="works both arms through a flat semaphore",
        pose="dance",
        dur=(2.6, 4.5),
        weight=0.7,
        motion="still",
    ),
    Vignette(
        key="m_talks_with_hands",
        label="explains something entirely with its hands",
        pose="dance",
        dur=(3.0, 5.0),
        weight=0.9,
        motion="still",
    ),
    Vignette(
        # Pacing while explaining: the drift is the argument getting away.
        key="m_pace_and_explain",
        label="paces up and down, explaining as it goes",
        pose="walk",
        dur=(3.0, 5.0),
        weight=0.8,
        motion="pace",
        drift=26.0,
    ),
    Vignette(
        # A message left for a reader who is not born yet: signalling across
        # time rather than distance, which is the same arm-waving really.
        key="m_stone_arrow",
        label="lays stones in an arrow for whoever comes next",
        pose="build",
        dur=(3.5, 6.0),
        weight=0.5,
        motion="crouch",
    ),
    Vignette(
        key="m_failed_whistle",
        label="tries to whistle and manages only air",
        pose="idle",
        dur=(1.5, 2.4),
        weight=0.15,
        motion="still",
    ),

    # ==================================================================
    #  Signals that need something to signal *with* - a rock, a fire, a
    #  candle, a hill. The context is half the sentence.
    # ==================================================================
    Vignette(
        key="m_drum_on_rock",
        label="drums a rhythm out on a rock",
        pose="chop",
        dur=(2.4, 4.0),
        weight=1.1,
        tags=("near_rock",),
        motion="crouch",
    ),
    Vignette(
        key="m_smoke_column",
        label="fans the smoke into a column with both arms",
        pose="idle",
        dur=(2.6, 4.5),
        weight=0.7,
        tags=("near_fire",),
        motion="still",
    ),
    Vignette(
        # Rare and slightly sad: the only reply it is ever going to get.
        key="m_wave_at_reflection",
        label="waves at its own reflection until it stops",
        pose="idle",
        dur=(2.2, 3.8),
        weight=0.2,
        tags=("near_water",),
        speech="?",
        motion="still",
    ),
    Vignette(
        key="m_candle_signal",
        label="swings its candle in a slow arc, three times",
        pose="carry",
        dur=(2.4, 4.0),
        weight=1.0,
        tags=("night", "candle"),
        speech="*",
        motion="still",
    ),
    Vignette(
        key="m_hail_the_camp",
        label="hails the camp far below",
        pose="climb",
        dur=(2.0, 3.4),
        weight=1.0,
        tags=("high_up",),
        speech="!",
        motion="still",
    ),
    Vignette(
        key="m_all_clear_sweep",
        label="sweeps an arm along the horizon: all clear",
        pose="idle",
        dur=(2.4, 4.0),
        weight=1.0,
        tags=("lookout",),
        motion="still",
    ),

    # ==================================================================
    #  Company - signals with an actual audience, and one without.
    # ==================================================================
    Vignette(
        # Sidles the few px over to the shoulder it is tapping.
        key="m_tap_and_point",
        label="taps a shoulder and points somewhere else",
        pose="walk",
        dur=(1.8, 3.0),
        weight=1.1,
        tags=("social",),
        motion="pace",
        drift=16.0,
    ),
    Vignette(
        key="m_silent_hand_signs",
        label="makes small hand-signs nobody was meant to hear",
        pose="idle",
        dur=(2.2, 3.8),
        weight=0.8,
        tags=("night", "social"),
        motion="still",
    ),
    Vignette(
        key="m_elder_instruction",
        label="gives a long instruction entirely in gestures",
        pose="idle",
        dur=(3.5, 6.0),
        weight=0.6,
        tags=("elder",),
        motion="still",
    ),
    Vignette(
        # The longest, rarest thing in the band: flat on its back, working
        # the arms at whatever might be up there.
        key="m_signal_at_the_sky",
        label="lies flat and signals at the sky, just in case",
        pose="sleep",
        dur=(4.0, 6.5),
        weight=0.15,
        tags=("night", "alone"),
        motion="lie",
    ),

    # ==================================================================
    #  Weather - the conditions that make signalling necessary and also
    #  make it fail.
    # ==================================================================
    Vignette(
        key="m_jab_toward_shelter",
        label="stamps twice and jabs a thumb toward shelter",
        pose="walk",
        dur=(1.6, 2.8),
        weight=0.9,
        tags=("raining",),
        motion="hop",
    ),
    Vignette(
        key="m_wave_off_the_snow",
        label="waves off a shout it cannot hear through the snow",
        pose="idle",
        dur=(1.8, 3.0),
        weight=0.8,
        tags=("snowing",),
        motion="still",
    ),
    Vignette(
        key="m_arms_crossed_warning",
        label="crosses both arms overhead: stay back",
        pose="panic",
        dur=(1.8, 3.0),
        weight=0.9,
        tags=("storm",),
        speech="!",
        motion="still",
    ),

    # ==================================================================
    #  Need and mood - what a body signals when it wants something, or
    #  when it has stopped wanting to signal at all.
    # ==================================================================
    Vignette(
        key="m_mime_a_meal",
        label="mimes a whole meal at nobody in particular",
        pose="idle",
        dur=(1.8, 3.0),
        weight=0.9,
        tags=("hungry",),
        speech="o",
        motion="still",
    ),
    Vignette(
        key="m_flap_later",
        label="flaps a hand from where it sits: later, later",
        pose="sleep",
        dur=(1.5, 2.6),
        weight=0.9,
        tags=("tired",),
        motion="sit",
    ),
    Vignette(
        key="m_wave_that_falls",
        label="starts a wave and lets the arm fall",
        pose="mourn",
        dur=(1.8, 3.0),
        weight=0.7,
        tags=("sad",),
        motion="still",
    ),
]


# ===========================================================================
#  Smoke test - content-side invariants only; the engine tests itself.
# ===========================================================================
if __name__ == "__main__":   # pragma: no cover
    from .vignettes import MAX_DRIFT, MAX_DUR, MIN_DUR, MOTIONS

    POSES = {"idle", "walk", "carry", "chop", "build", "climb",
             "fall", "sleep", "dance", "mourn", "panic"}

    assert len(VIGNETTES) == 30, len(VIGNETTES)
    keys = [v.key for v in VIGNETTES]
    assert len(set(keys)) == 30, "duplicate key"
    assert all(k.startswith("m_") for k in keys), "unprefixed key"
    assert all(MIN_DUR <= v.dur[0] <= v.dur[1] <= MAX_DUR for v in VIGNETTES)
    assert all(v.motion in MOTIONS for v in VIGNETTES)
    assert all(v.pose in POSES for v in VIGNETTES), "unknown pose"
    assert all(0.0 <= v.drift <= MAX_DRIFT for v in VIGNETTES)
    assert all(v.weight > 0.0 for v in VIGNETTES)
    assert all(v.speech is None or len(v.speech) == 1 for v in VIGNETTES)
    assert all(v.drift > 0.0 for v in VIGNETTES if v.motion == "pace")

    tagged = [v for v in VIGNETTES if v.tags]
    ungated = [v for v in VIGNETTES if not v.gates]
    short = [v for v in VIGNETTES if v.dur[1] <= 3.0]
    long_ = [v for v in VIGNETTES if v.dur[1] > 4.5]
    speech = [v for v in VIGNETTES if v.speech]
    assert 12 <= len(tagged) <= 18, len(tagged)
    assert len(ungated) >= 14, len(ungated)
    assert len(short) > len(long_), (len(short), len(long_))
    assert 6 <= len(speech) <= 9, len(speech)

    used = {v.motion for v in VIGNETTES}
    assert used == set(MOTIONS), set(MOTIONS) - used
    print(f"{len(VIGNETTES)} vignettes, {len(tagged)} tagged, "
          f"{len(ungated)} always-available, {len(speech)} with speech")
    print("motions:", {m: sum(1 for v in VIGNETTES if v.motion == m)
                       for m in MOTIONS})
