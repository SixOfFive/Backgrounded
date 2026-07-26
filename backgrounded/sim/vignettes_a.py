"""Vignette content, band A - the body and the idle moment.

Forty cosmetic beats about *having a body*: stretching, yawning, scratching,
cracking, shivering, sneezing, sitting down for no reason at all. Nothing here
gathers, builds, consumes or changes a single thing about the world. A colony
whose people only ever walk to a tree and back is a shuttle service; a colony
where someone stops halfway to rub the back of its neck is alive.

Content only - the state machine, the picker and the contract all live in
``vignettes.py``. See :class:`~.vignettes.Vignette` for the field meanings and
the normalisation rules; everything below is written to sit comfortably inside
them rather than to be clamped into them.

House rules for this file
-------------------------
* Keys are all prefixed ``a_`` so the five content bands cannot collide.
* Poses are drawn from the sim's pose vocabulary (``actions.POSES``); the
  renderer aliases the ones it does not draw directly (``eat``/``warm``/
  ``lookout`` and friends) onto their nearest body.
* Gating tags are ones :func:`~.vignettes.context_tags` actually evaluates, so
  they *mean* something - a yawn wants a tired agent, not a random one. A
  handful of entries carry a decorative tag ("silly", "quiet", "sudden")
  instead: those never gate, so the pool is never empty however grim the
  weather or however fresh the agent.
* Short beats outnumber long ones. An idle body twitches far more often than
  it lies down.
"""
from __future__ import annotations

from .vignettes import Vignette

__all__ = ["VIGNETTES"]


VIGNETTES: list[Vignette] = [
    # ==================================================================
    #  Ungated - the everyday twitches. These fit any context at all and
    #  are what the picker falls back on when nothing special applies.
    # ==================================================================
    Vignette(
        "a_stretch_tall", "stretches until its back pops", "climb",
        (2.0, 3.4), 1.3, motion="still",
    ),
    Vignette(
        "a_scratch_ear", "scratches an ear", "idle",
        (1.5, 2.6), 1.4, motion="still",
    ),
    Vignette(
        "a_scratch_head", "scratches its head", "idle",
        (1.6, 2.8), 1.3, speech="?", motion="still",
    ),
    Vignette(
        "a_crack_knuckles", "cracks its knuckles one at a time", "idle",
        (1.5, 2.4), 1.1, motion="still",
    ),
    Vignette(
        "a_crack_neck", "cracks its neck, twice", "idle",
        (1.5, 2.4), 1.0, speech="!", motion="still",
    ),
    Vignette(
        "a_roll_shoulders", "rolls its shoulders", "idle",
        (1.6, 2.8), 1.2, motion="still",
    ),
    Vignette(
        # Turning to beat the dust off its own back - the spin sells it.
        "a_dust_off", "dusts itself off", "idle",
        (1.8, 3.0), 1.2, motion="spin",
    ),
    Vignette(
        "a_wipe_brow", "wipes its brow", "idle",
        (1.6, 2.8), 1.2, motion="still",
    ),
    Vignette(
        "a_straighten_up", "remembers its posture and straightens up", "idle",
        (1.5, 2.6), 1.1, motion="still",
    ),
    Vignette(
        "a_check_hands", "turns its hands over and studies them", "idle",
        (2.0, 3.6), 1.0, speech="?", motion="still",
    ),
    Vignette(
        "a_pick_teeth", "picks at its teeth", "eat",
        (1.8, 3.0), 0.8, motion="still",
    ),
    Vignette(
        "a_tie_something", "crouches to tie something", "build",
        (2.2, 3.8), 1.1, motion="crouch",
    ),
    Vignette(
        "a_balance_one_leg", "balances on one leg for no reason at all", "idle",
        (2.6, 4.5), 0.6, motion="still",
    ),
    Vignette(
        "a_sit_down", "sits down for no particular reason", "sleep",
        (4.0, 7.0), 0.9, motion="sit",
    ),
    Vignette(
        "a_pace_stiff", "walks off a stiff leg", "walk",
        (3.0, 5.0), 0.9, motion="pace", drift=26.0,
    ),

    # ==================================================================
    #  Decorative tags only - odd little accidents. They never gate, but
    #  their weights keep them rare enough to stay funny.
    # ==================================================================
    Vignette(
        "a_sneeze", "sneezes", "mourn",
        (1.5, 2.2), 0.9, tags=("sudden",), speech="!", motion="still",
    ),
    Vignette(
        "a_hiccups", "gets the hiccups", "idle",
        (2.4, 4.0), 0.4, tags=("silly",), speech="o", motion="hop",
    ),
    Vignette(
        "a_trip_over", "trips over its own feet", "panic",
        (1.5, 2.4), 0.15, tags=("silly",), speech="!", motion="hop",
    ),
    Vignette(
        "a_scratch_back", "twists around to scratch its own back", "idle",
        (2.0, 3.6), 0.8, tags=("silly",), motion="spin",
    ),
    Vignette(
        "a_twist_waist", "twists side to side, loosening up", "idle",
        (2.0, 3.4), 0.9, tags=("quiet",), motion="spin",
    ),

    # ==================================================================
    #  Tired - the body asking to stop.
    # ==================================================================
    Vignette(
        "a_yawn", "yawns hugely", "idle",
        (1.6, 2.8), 1.4, tags=("tired",), speech="~", motion="still",
    ),
    Vignette(
        "a_rub_eyes", "rubs its eyes", "idle",
        (1.8, 3.0), 1.3, tags=("tired",), motion="still",
    ),
    Vignette(
        "a_catch_breath", "doubles over to catch its breath", "build",
        (2.2, 4.0), 1.1, tags=("tired",), motion="crouch",
    ),
    Vignette(
        "a_slow_circuit", "paces a slow circle to wake its legs up", "walk",
        (3.5, 6.0), 0.7, tags=("tired",), motion="pace", drift=34.0,
    ),
    Vignette(
        # The longest, rarest thing in the band: flat out, under the sky.
        "a_lie_flat", "flops flat on its back", "sleep",
        (5.0, 9.0), 0.35, tags=("night", "tired"), speech="z", motion="lie",
    ),

    # ==================================================================
    #  Mood - what a body does with nobody watching, or with too many.
    # ==================================================================
    Vignette(
        "a_lie_back", "lies back and does absolutely nothing", "sleep",
        (4.5, 8.0), 0.5, tags=("alone",), motion="lie",
    ),
    Vignette(
        "a_sit_hug_knees", "sits and hugs its knees", "sleep",
        (4.0, 7.5), 0.55, tags=("sad",), motion="sit",
    ),
    Vignette(
        "a_wring_hands", "wrings its hands", "mourn",
        (2.0, 3.4), 0.6, tags=("sad",), motion="still",
    ),
    Vignette(
        "a_flex_arm", "flexes an arm and admires it", "idle",
        (2.0, 3.4), 0.5, tags=("happy", "social"), speech="*", motion="still",
    ),

    # ==================================================================
    #  Cold and fire - thermoregulation as theatre.
    # ==================================================================
    Vignette(
        "a_shiver", "shivers and hugs itself", "idle",
        (2.0, 3.6), 1.4, tags=("cold",), motion="still",
    ),
    Vignette(
        "a_stamp_feet", "stamps its feet to get the blood moving", "walk",
        (2.0, 3.4), 1.1, tags=("cold",), motion="hop",
    ),
    Vignette(
        "a_blow_fingers", "blows on its fingers", "eat",
        (2.0, 3.4), 1.1, tags=("cold",), motion="still",
    ),
    Vignette(
        "a_toast_back", "turns around to warm its back", "warm",
        (3.0, 5.0), 1.0, tags=("near_fire",), motion="spin",
    ),

    # ==================================================================
    #  Weather on skin.
    # ==================================================================
    Vignette(
        "a_hunch_rain", "hunches its shoulders against the rain", "idle",
        (2.0, 3.6), 1.2, tags=("raining",), motion="still",
    ),
    Vignette(
        "a_shake_off_rain", "shakes the water out of its hair", "idle",
        (1.5, 2.6), 0.9, tags=("raining",), motion="spin",
    ),
    Vignette(
        "a_brace_wind", "leans into the wind", "idle",
        (2.0, 3.6), 1.0, tags=("windy",), motion="still",
    ),
    Vignette(
        "a_squint_sun", "squints and shields its eyes", "lookout",
        (1.6, 2.8), 1.1, tags=("day",), motion="still",
    ),

    # ==================================================================
    #  Appetite and age - bodies that are specifically *this* body.
    # ==================================================================
    Vignette(
        "a_belly_rumble", "presses a hand to its stomach", "idle",
        (1.8, 3.0), 1.0, tags=("hungry",), speech="o", motion="still",
    ),
    Vignette(
        "a_creak_knees", "gets up slowly, knees complaining", "build",
        (2.5, 4.2), 0.8, tags=("elder",), motion="crouch",
    ),
    Vignette(
        "a_child_wobble", "wobbles about testing its balance", "walk",
        (2.0, 3.6), 0.7, tags=("child",), motion="hop",
    ),
]


# ===========================================================================
#  Smoke test - content-side invariants only; the engine tests itself.
# ===========================================================================
if __name__ == "__main__":   # pragma: no cover
    from .vignettes import MAX_DRIFT, MAX_DUR, MIN_DUR, MOTIONS

    assert len(VIGNETTES) == 40, len(VIGNETTES)
    keys = [v.key for v in VIGNETTES]
    assert len(set(keys)) == 40, "duplicate key"
    assert all(k.startswith("a_") for k in keys), "unprefixed key"
    assert all(MIN_DUR <= v.dur[0] <= v.dur[1] <= MAX_DUR for v in VIGNETTES)
    assert all(v.motion in MOTIONS for v in VIGNETTES)
    assert all(0.0 <= v.drift <= MAX_DRIFT for v in VIGNETTES)
    assert all(v.weight > 0.0 for v in VIGNETTES)
    assert all(v.speech is None or len(v.speech) == 1 for v in VIGNETTES)

    tagged = [v for v in VIGNETTES if v.tags]
    ungated = [v for v in VIGNETTES if not v.gates]
    short = [v for v in VIGNETTES if v.dur[1] <= 3.0]
    long_ = [v for v in VIGNETTES if v.dur[1] > 4.5]
    assert len(tagged) >= 20, len(tagged)
    assert len(ungated) >= 15, len(ungated)
    assert len(short) > len(long_), (len(short), len(long_))

    used = {v.motion for v in VIGNETTES}
    assert used == set(MOTIONS), set(MOTIONS) - used
    print(f"{len(VIGNETTES)} vignettes, {len(tagged)} tagged, "
          f"{len(ungated)} always-available, "
          f"{sum(1 for v in VIGNETTES if v.speech)} with speech")
    print("motions:", {m: sum(1 for v in VIGNETTES if v.motion == m)
                       for m in MOTIONS})
