"""Vignette content, band N - discomfort and ailment.

Thirty cosmetic beats about a body that is *slightly wrong today*: a stitch, a
splinter, a stone in a shoe, a back that will not straighten on the first try.
Band A covers the neutral idle body - stretching, scratching, yawning. This
band covers the same body complaining. The distinction matters: a stretch says
"nothing is happening", a limp says "something happened, earlier, off-screen",
and a colony full of small unexplained injuries reads as one that has been
working rather than one that has been idling.

Nothing here touches the world. No stockpile moves, no prop changes, no agent
is actually hurt - the sim's health model, such as it is, lives elsewhere and
is not consulted. An agent that limps is not injured; it is *acting* injured,
for two and a half seconds, and then it gets on with its day. That is the whole
contract, and it is deliberate: the moment a vignette could hurt someone, the
picker would need to care about consequences, and it must not.

Content only - the state machine, the picker and the field rules all live in
``vignettes.py``.

House rules for this file
-------------------------
* Keys are all prefixed ``n_`` so the content bands cannot collide.
* Poses stay inside the set the renderer actually draws: ``idle``, ``walk``,
  ``carry``, ``chop``, ``build``, ``climb``, ``fall``, ``sleep``, ``dance``,
  ``mourn``, ``panic``. Crouching to look at a shin is ``build``; hauling
  yourself upright against a bad back is ``climb``.
* Gating tags earn their place. A cramped hand belongs to a gatherer who has
  been gripping things all day; a hit thumb belongs to a builder; a wheeze
  belongs to cold air. Roughly half the band is ungated so the pool never
  empties for a warm, fed, well-rested agent standing alone at noon.
* Short beats outnumber long ones by a wide margin. Pain is mostly a flinch.
  Only two entries run past 4.5s, and both of them are the body giving up.
"""
from __future__ import annotations

from .vignettes import Vignette

__all__ = ["VIGNETTES"]


VIGNETTES: list[Vignette] = [
    # ==================================================================
    #  Legs and feet - the parts that carried it here and resent it.
    # ==================================================================
    Vignette(
        # The drift is the point: it walks the limp off rather than posing.
        "n_limp_it_off", "limps a few steps and walks it off", "walk",
        (2.6, 4.2), 1.1, motion="pace", drift=24.0,
    ),
    Vignette(
        "n_favour_leg", "puts all its weight on one leg", "idle",
        (2.0, 3.6), 1.0, motion="still",
    ),
    Vignette(
        "n_calf_cramp", "shakes a cramp out of its calf", "walk",
        (1.8, 3.0), 1.0, motion="hop",
    ),
    Vignette(
        "n_stone_in_shoe", "hops on one foot, shaking something out", "walk",
        (2.0, 3.4), 0.8, speech="?", motion="hop",
    ),
    Vignette(
        "n_blister_heel", "sits down and inspects a blistered heel", "sleep",
        (2.6, 4.2), 0.7, motion="sit",
    ),
    Vignette(
        "n_pins_and_needles", "stamps a foot that has gone to sleep", "walk",
        (1.8, 3.0), 0.7, speech="o", motion="hop",
    ),
    Vignette(
        "n_test_ankle", "puts careful weight on one ankle, testing it", "idle",
        (2.0, 3.4), 0.8, tags=("high_up",), motion="still",
    ),

    # ==================================================================
    #  Back, neck and shoulders - the freight of a working day.
    # ==================================================================
    Vignette(
        "n_rub_lower_back", "rubs the small of its back", "idle",
        (1.8, 3.0), 1.2, tags=("builder",), motion="still",
    ),
    Vignette(
        # `climb` reads as hauling itself upright, which is exactly the beat.
        "n_back_seized", "straightens up in stages, and thinks about it", "climb",
        (2.4, 4.0), 0.7, tags=("elder",), speech="!", motion="still",
    ),
    Vignette(
        "n_stiff_neck", "works a stiff neck slowly side to side", "idle",
        (1.8, 3.0), 1.2, motion="spin",
    ),
    Vignette(
        "n_sore_shoulder", "rolls one shoulder, only one", "idle",
        (1.6, 2.8), 1.0, tags=("gatherer",), motion="still",
    ),

    # ==================================================================
    #  Lungs and gut.
    # ==================================================================
    Vignette(
        "n_cough_fit", "coughs into its elbow, twice", "idle",
        (1.8, 3.0), 1.1, speech="!", motion="still",
    ),
    Vignette(
        "n_wheeze_cold_air", "wheezes at a lungful of cold air", "idle",
        (2.0, 3.4), 0.9, tags=("cold",), motion="still",
    ),
    Vignette(
        "n_stitch_side", "walks off a stitch with a hand on its ribs", "walk",
        (2.4, 4.0), 1.0, tags=("tired",), motion="pace", drift=18.0,
    ),
    Vignette(
        "n_stomach_cramp", "folds over, holding its stomach", "mourn",
        (2.0, 3.4), 1.0, tags=("hungry",), motion="crouch",
    ),
    Vignette(
        "n_breath_after_climb", "hangs its head, still breathing hard", "build",
        (2.6, 4.4), 1.0, tags=("high_up",), motion="crouch",
    ),

    # ==================================================================
    #  Hands - the tools it did not get issued.
    # ==================================================================
    Vignette(
        "n_hand_cramp", "opens and closes a hand that will not unclench", "idle",
        (1.6, 2.8), 1.0, tags=("gatherer",), motion="still",
    ),
    Vignette(
        "n_pick_splinter", "digs a splinter out of its palm", "idle",
        (2.4, 4.0), 0.9, tags=("builder",), speech="!", motion="still",
    ),
    Vignette(
        "n_rub_cold_hands", "rubs its hands together hard, and harder", "idle",
        (1.6, 2.8), 1.2, tags=("cold",), motion="still",
    ),
    Vignette(
        "n_raw_palms", "turns its palms up and looks at the raw skin", "idle",
        (2.2, 3.6), 0.6, tags=("gatherer",), motion="still",
    ),
    Vignette(
        "n_hit_thumb", "shakes a hand hard and glares at it", "panic",
        (1.5, 2.4), 0.5, tags=("builder",), speech="!", motion="still",
    ),

    # ==================================================================
    #  Head, face and the general small betrayals.
    # ==================================================================
    Vignette(
        "n_wince_at_nothing", "winces at nothing in particular", "mourn",
        (1.5, 2.4), 1.1, motion="still",
    ),
    Vignette(
        "n_grit_in_eye", "blinks hard at something in its eye", "idle",
        (1.6, 2.8), 0.9, tags=("windy",), motion="still",
    ),
    Vignette(
        "n_press_temples", "presses two fingers to its temples", "mourn",
        (2.0, 3.4), 0.8, tags=("storm",), motion="still",
    ),
    Vignette(
        "n_split_lip", "worries at a split lip", "idle",
        (1.5, 2.6), 0.6, tags=("cold",), motion="still",
    ),
    Vignette(
        # Decorative tag only: an ear that will not clear is not a mountain
        # problem, and two gated high_up entries in the band are plenty.
        "n_ears_popping", "works its jaw until its ears pop", "idle",
        (1.8, 3.0), 0.5, tags=("quiet",), speech="o", motion="still",
    ),

    # ==================================================================
    #  Giving up, briefly - and the two rare ones.
    # ==================================================================
    Vignette(
        "n_check_scrape", "crouches to inspect a scrape on its shin", "build",
        (2.2, 3.8), 1.0, motion="crouch",
    ),
    Vignette(
        "n_sit_down_heavy", "sits down heavily and stays down a while", "sleep",
        (4.5, 7.5), 0.6, tags=("tired",), motion="sit",
    ),
    Vignette(
        # Rare on purpose. Standing up too fast, and the world tilting.
        "n_dizzy_spell", "sways, and waits for the world to settle", "fall",
        (1.8, 3.0), 0.15, speech="~", motion="spin",
    ),
    Vignette(
        # The rarest thing in the band: shivering when it is not cold out.
        "n_fever_chill", "shivers although the night is mild", "sleep",
        (4.0, 6.5), 0.15, tags=("night", "alone"), motion="lie",
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
    assert all(k.startswith("n_") for k in keys), "unprefixed key"
    assert all(MIN_DUR <= v.dur[0] <= v.dur[1] <= MAX_DUR for v in VIGNETTES)
    assert all(v.motion in MOTIONS for v in VIGNETTES)
    assert all(v.pose in POSES for v in VIGNETTES), "unknown pose"
    assert all(0.0 <= v.drift <= MAX_DRIFT for v in VIGNETTES)
    assert all(v.weight > 0.0 for v in VIGNETTES)
    assert all(v.speech is None or len(v.speech) == 1 for v in VIGNETTES)

    tagged = [v for v in VIGNETTES if v.tags]
    ungated = [v for v in VIGNETTES if not v.gates]
    short = [v for v in VIGNETTES if v.dur[1] <= 3.0]
    long_ = [v for v in VIGNETTES if v.dur[1] > 4.5]
    assert 12 <= len(tagged) <= 20, len(tagged)
    assert len(ungated) >= 12, len(ungated)
    assert len(short) > 2 * len(long_), (len(short), len(long_))

    used = {v.motion for v in VIGNETTES}
    assert used == set(MOTIONS), set(MOTIONS) - used
    print(f"{len(VIGNETTES)} vignettes, {len(tagged)} tagged, "
          f"{len(ungated)} always-available, "
          f"{sum(1 for v in VIGNETTES if v.speech)} with speech")
    print("motions:", {m: sum(1 for v in VIGNETTES if v.motion == m)
                       for m in MOTIONS})
