"""Vignette content, band E - the odd and the rare.

Band A-D cover the everyday beats: scratching, stretching, staring at the fire.
This file is the other end of the distribution - the things a stickman does
maybe once an hour, that make a watcher lean in and go "did it just...?".
Cartwheels that do not land. Two steps of a moonwalk. A serious conversation
with a rock. An interpretive dance for an audience of nobody.

Because they are meant to be *rare*, every weight here sits well under 1.0 -
the showiest ones sit near 0.12, so on any given idle beat the colony almost
always does something ordinary instead, and the strange thing lands as a
surprise rather than a tic.

Contract (enforced by ``vignettes.Vignette``, restated so it is visible here):
nothing in this file touches world state. A vignette moves the acting agent's
pose, facing, speech bubble and animation phase, plus at most ``MAX_DRIFT`` px
of horizontal shuffle. No stockpile, no structures, no props, no resources.

Poses are limited to the eleven ``render.stickfigure.POSES`` bodies; "fall" is
doing a lot of comedic work below because it is the only body that reads as
"this has gone wrong".

Tags: the ones inside ``vignettes.KNOWN_TAGS`` gate when the beat may fire
("night", "near_fire", "social", ...). Anything else ("silly", "solemn",
"showing_off") is a decorative label the engine ignores - useful for a human
reading the file, harmless to the picker. Roughly half of these are ungated so
there is always something available to pick.

No pygame. Pure python.
"""
from __future__ import annotations

from .vignettes import Vignette

__all__ = ["VIGNETTES"]


VIGNETTES: list[Vignette] = [
    # ---------------------------------------------------------- acrobatics --
    Vignette(
        key="e_cartwheel_fail",
        label="attempts a cartwheel and lands flat",
        pose="fall",
        dur=(1.8, 3.0),
        weight=0.12,
        tags=("silly", "showing_off"),
        speech="!",
        motion="hop",
    ),
    Vignette(
        key="e_handstand_brief",
        label="holds a handstand for one whole second",
        pose="fall",
        dur=(1.7, 2.8),
        weight=0.12,
        tags=("showing_off",),
        motion="still",
    ),
    Vignette(
        key="e_headstand_flop",
        label="tries a headstand and flops over",
        pose="fall",
        dur=(1.9, 3.0),
        weight=0.12,
        tags=("silly",),
        speech="!",
        motion="lie",
    ),
    Vignette(
        key="e_roll_downhill",
        label="lies down and rolls a little way",
        pose="fall",
        dur=(2.2, 3.8),
        weight=0.14,
        tags=("silly",),
        motion="lie",
    ),
    Vignette(
        key="e_trip_flat_ground",
        label="trips over perfectly flat ground",
        pose="fall",
        dur=(1.5, 2.2),
        weight=0.30,
        tags=("silly",),
        speech="!",
        motion="still",
    ),

    # ------------------------------------------------------------- dancing --
    Vignette(
        key="e_moonwalk",
        label="moonwalks two steps",
        pose="walk",
        dur=(2.0, 3.2),
        weight=0.15,
        tags=("silly", "showing_off"),
        motion="pace",
        drift=26.0,
    ),
    Vignette(
        key="e_dance_badly",
        label="dances badly",
        pose="dance",
        dur=(2.6, 4.6),
        weight=0.45,
        tags=("silly",),
        motion="hop",
    ),
    Vignette(
        key="e_interpretive_dance",
        label="performs an interpretive dance",
        pose="dance",
        dur=(4.5, 7.5),
        weight=0.12,
        tags=("alone", "silly"),
        motion="spin",
    ),
    Vignette(
        key="e_air_drums",
        label="air-drums a solo nobody asked for",
        pose="dance",
        dur=(2.2, 4.0),
        weight=0.32,
        tags=("silly",),
        motion="still",
    ),
    Vignette(
        key="e_spin_till_dizzy",
        label="spins until it is dizzy",
        pose="dance",
        dur=(2.4, 4.4),
        weight=0.22,
        tags=("silly",),
        speech="*",
        motion="spin",
    ),
    Vignette(
        key="e_moon_conduct",
        label="conducts the moon",
        pose="dance",
        dur=(3.0, 5.5),
        weight=0.12,
        tags=("night", "solemn"),
        speech="*",
        motion="still",
    ),

    # -------------------------------------------------------------- drills --
    Vignette(
        key="e_pushups",
        label="does a set of push-ups",
        pose="build",
        dur=(2.6, 5.0),
        weight=0.30,
        tags=("physical",),
        motion="crouch",
    ),
    Vignette(
        key="e_march",
        label="marches on the spot",
        pose="walk",
        dur=(2.0, 3.4),
        weight=0.40,
        motion="hop",
    ),
    Vignette(
        key="e_shadow_box",
        label="shadow-boxes an invisible opponent",
        pose="chop",
        dur=(2.2, 4.0),
        weight=0.28,
        tags=("physical",),
        motion="hop",
    ),
    Vignette(
        key="e_salute_nothing",
        label="salutes nothing in particular",
        pose="idle",
        dur=(1.5, 2.4),
        weight=0.25,
        tags=("lookout", "solemn"),
        motion="still",
    ),
    Vignette(
        key="e_tiptoe",
        label="tiptoes for no reason at all",
        pose="walk",
        dur=(2.2, 3.8),
        weight=0.30,
        tags=("silly",),
        motion="pace",
        drift=30.0,
    ),
    Vignette(
        key="e_walk_backwards",
        label="walks backwards to see how it feels",
        pose="walk",
        dur=(2.4, 4.0),
        weight=0.24,
        tags=("silly",),
        motion="pace",
        drift=36.0,
    ),
    Vignette(
        key="e_kick_pretend_ball",
        label="kicks an imaginary ball",
        pose="walk",
        dur=(1.8, 3.0),
        weight=0.35,
        tags=("silly",),
        motion="hop",
    ),
    Vignette(
        key="e_pretend_fly",
        label="runs about pretending to fly",
        pose="climb",
        dur=(2.4, 4.6),
        weight=0.18,
        tags=("silly",),
        motion="pace",
        drift=34.0,
    ),

    # ---------------------------------------------------- talking to things --
    Vignette(
        key="e_talk_to_rock",
        label="has a long word with a rock",
        pose="build",
        dur=(2.5, 5.0),
        weight=0.20,
        tags=("near_rock", "silly"),
        speech="?",
        motion="crouch",
    ),
    Vignette(
        key="e_talks_to_fire",
        label="explains something complicated to the fire",
        pose="idle",
        dur=(3.5, 6.5),
        weight=0.22,
        tags=("near_fire",),
        motion="sit",
    ),
    Vignette(
        key="e_bow_to_tree",
        label="bows solemnly to a tree",
        pose="build",
        dur=(1.8, 3.0),
        weight=0.22,
        tags=("near_tree", "solemn"),
        motion="crouch",
    ),
    Vignette(
        key="e_argue_with_self",
        label="argues with itself and loses",
        pose="idle",
        dur=(2.6, 4.4),
        weight=0.18,
        tags=("alone", "silly"),
        speech="?",
        motion="spin",
    ),
    Vignette(
        key="e_high_five_air",
        label="high-fives the air where a friend should be",
        pose="idle",
        dur=(1.6, 2.6),
        weight=0.16,
        tags=("alone", "silly"),
        motion="still",
    ),

    # ----------------------------------------------------------- stillness --
    Vignette(
        key="e_meditate",
        label="sits and meditates",
        pose="sleep",
        dur=(5.0, 9.0),
        weight=0.30,
        tags=("alone", "quiet"),
        speech="~",
        motion="sit",
    ),
    Vignette(
        key="e_pretend_statue",
        label="holds still, pretending to be a statue",
        pose="idle",
        dur=(4.0, 8.0),
        weight=0.16,
        tags=("silly",),
        motion="still",
    ),
    Vignette(
        key="e_sleep_standing",
        label="falls asleep standing up",
        pose="sleep",
        dur=(3.2, 6.0),
        weight=0.28,
        tags=("tired",),
        speech="z",
        motion="still",
    ),
    Vignette(
        key="e_freeze_mid_step",
        label="freezes mid-step for no reason",
        pose="walk",
        dur=(1.5, 2.6),
        weight=0.26,
        tags=("silly",),
        speech="?",
        motion="still",
    ),
    Vignette(
        key="e_lie_and_stare",
        label="lies down and stares at nothing",
        pose="sleep",
        dur=(5.0, 9.0),
        weight=0.26,
        tags=("quiet",),
        motion="lie",
    ),
    Vignette(
        key="e_heroic_pose",
        label="strikes a heroic pose",
        pose="idle",
        dur=(2.0, 3.2),
        weight=0.28,
        tags=("high_up", "showing_off"),
        motion="still",
    ),

    # ------------------------------------------------------ small business --
    Vignette(
        key="e_juggle_pebbles",
        label="juggles pebbles and drops every one",
        pose="carry",
        dur=(2.4, 4.2),
        weight=0.20,
        tags=("near_rock", "silly"),
        speech="o",
        motion="still",
    ),
    Vignette(
        key="e_count_fingers",
        label="counts its fingers twice",
        pose="idle",
        dur=(2.0, 3.2),
        weight=0.30,
        tags=("silly",),
        motion="still",
    ),
    Vignette(
        key="e_bug_parade",
        label="follows a beetle across the ground",
        pose="build",
        dur=(3.0, 5.5),
        weight=0.28,
        tags=("day", "quiet"),
        motion="pace",
        drift=22.0,
    ),
    Vignette(
        key="e_chase_shadow",
        label="chases its own shadow",
        pose="walk",
        dur=(2.0, 3.4),
        weight=0.18,
        tags=("day", "silly"),
        motion="spin",
    ),
    Vignette(
        key="e_solemn_hat_removal",
        label="removes an imaginary hat",
        pose="mourn",
        dur=(1.5, 2.4),
        weight=0.18,
        tags=("solemn", "silly"),
        motion="still",
    ),

    # ----------------------------------------------------------- an audience --
    Vignette(
        key="e_practise_bow",
        label="practises a formal bow, repeatedly",
        pose="build",
        dur=(2.0, 3.4),
        weight=0.24,
        tags=("social", "solemn"),
        motion="crouch",
    ),
    Vignette(
        key="e_impersonate_elder",
        label="does an impression of the elder",
        pose="walk",
        dur=(2.6, 4.4),
        weight=0.14,
        tags=("social", "silly"),
        motion="pace",
        drift=24.0,
    ),

    # ------------------------------------------------------------- weather --
    Vignette(
        key="e_pretend_lightning",
        label="re-enacts the lightning, badly",
        pose="panic",
        dur=(2.0, 3.4),
        weight=0.18,
        tags=("storm", "silly"),
        speech="!",
        motion="hop",
    ),
    Vignette(
        key="e_puddle_stomp",
        label="stomps a puddle with unreasonable joy",
        pose="walk",
        dur=(2.0, 3.5),
        weight=0.38,
        tags=("raining", "silly"),
        motion="hop",
    ),
    Vignette(
        key="e_snow_angel",
        label="makes a snow angel",
        pose="sleep",
        dur=(3.0, 6.0),
        weight=0.24,
        tags=("snowing",),
        motion="lie",
    ),
]


# ===========================================================================
#  Smoke test - run with:  python -m backgrounded.sim.vignettes_e
# ===========================================================================
if __name__ == "__main__":   # pragma: no cover - headless smoke test
    from .vignettes import MAX_DRIFT, MAX_DUR, MIN_DUR, MOTIONS

    assert len(VIGNETTES) == 40, len(VIGNETTES)
    keys = [v.key for v in VIGNETTES]
    assert len(set(keys)) == 40, "duplicate key"
    assert all(k.startswith("e_") for k in keys), "unprefixed key"
    assert all(MIN_DUR <= v.dur[0] <= v.dur[1] <= MAX_DUR for v in VIGNETTES)
    assert all(v.motion in MOTIONS for v in VIGNETTES)
    assert all(0.0 < v.weight < 1.0 for v in VIGNETTES), "band E must stay rare"
    assert all(0.0 <= v.drift <= MAX_DRIFT for v in VIGNETTES)
    assert all(v.drift == 0.0 or v.motion == "pace" for v in VIGNETTES)

    motions = {m: sum(1 for v in VIGNETTES if v.motion == m) for m in MOTIONS}
    assert all(motions[m] for m in MOTIONS), motions
    short = sum(1 for v in VIGNETTES if v.dur[1] <= 3.5)
    long_ = sum(1 for v in VIGNETTES if v.dur[0] >= 3.0)
    assert short > long_, (short, long_)
    speech = sum(1 for v in VIGNETTES if v.speech)
    tagged = sum(1 for v in VIGNETTES if v.tags)
    gated = sum(1 for v in VIGNETTES if v.gates)
    assert tagged >= 20 and gated <= 20, (tagged, gated)

    print(f"vignettes_e: {len(VIGNETTES)} entries")
    print("  motions :", motions)
    print(f"  duration: {short} short, {long_} long")
    print(f"  speech  : {speech}, tagged: {tagged}, gated: {gated}")
    print(f"  ungated : {len(VIGNETTES) - gated} always available")
    print("smoke test passed")
