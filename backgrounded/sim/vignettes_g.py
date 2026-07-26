"""Vignette content, band G - weather on a body that has to stand in it.

Thirty cosmetic beats about being *rained on, blown at, frozen, glared at and
soaked through*. Band A already owns the body's own twitches and band B owns
the ground and the sky as scenery; this band is the argument between the two -
what a stickman does with its hands, its shirt and its footing while the
weather happens to it.

Nothing here changes the world. No puddle is drained, no fire is fed, no snow
is cleared; the labels describe, the state machine in :mod:`.vignettes` plays
poses and a few px of x, and the weather carries on exactly as it would have.

House rules for this file
-------------------------
* Keys are all ``g_``-prefixed so the bands cannot collide.
* Poses stay inside the set the renderer draws directly - ``idle``, ``walk``,
  ``carry``, ``chop``, ``build``, ``climb``, ``fall``, ``sleep``, ``dance``,
  ``mourn``, ``panic`` - so nothing falls back to a blank stand.
* Gating tags name a real condition. A snow beat wants ``snowing``; a beat
  about steaming dry wants ``raining`` *and* ``near_fire``, because that
  particular picture only exists where those two overlap.
* Twelve entries carry no tags at all. They are the ones that read true in any
  weather - waiting for the first drop, walking round a puddle, bracing for a
  gust that does not arrive - and they keep the pool non-empty on a clear
  afternoon when every gated entry in the band is locked out.
* Short beats outnumber long ones. Weather is mostly reacted to in a second and
  a half; only shelter and sunbathing earn real time.
"""
from __future__ import annotations

from .vignettes import Vignette

__all__ = ["VIGNETTES"]


VIGNETTES: list[Vignette] = [
    # ==================================================================
    #  Untagged - weather as a permanent condition of being outdoors.
    #  These fit a clear noon as readily as a downpour, which is the
    #  point: the picker always has somewhere to go.
    # ==================================================================
    Vignette(
        # Arm up, palm flat: the climb pose already draws a raised arm.
        "g_palm_out", "holds a palm out flat, waiting for the first drop",
        "climb", (2.0, 3.4), 1.2, motion="still",
    ),
    Vignette(
        "g_read_clouds", "reads the clouds and decides it has time", "idle",
        (2.2, 3.6), 1.2, motion="still",
    ),
    Vignette(
        "g_wipe_face", "wipes the weather off its face", "idle",
        (1.5, 2.6), 1.3, motion="still",
    ),
    Vignette(
        "g_stamp_dry", "stamps the wet off its feet", "walk",
        (1.6, 2.6), 1.1, motion="hop",
    ),
    Vignette(
        "g_wring_hem", "wrings out the hem of its shirt", "build",
        (2.0, 3.4), 1.1, motion="crouch",
    ),
    Vignette(
        "g_flap_shirt", "flaps its shirt for a bit of air", "idle",
        (1.8, 3.0), 1.0, motion="still",
    ),
    Vignette(
        "g_boot_tip", "tips something wet out of one boot", "build",
        (2.2, 3.6), 0.6, motion="crouch",
    ),
    Vignette(
        "g_far_weather", "watches a grey smudge of weather away on the horizon",
        "idle", (2.5, 4.5), 0.9, motion="still",
    ),
    Vignette(
        "g_puddle_detour", "walks the long way round a puddle", "walk",
        (3.0, 5.0), 0.9, motion="pace", drift=30.0,
    ),
    Vignette(
        "g_pace_shelter", "paces about looking for somewhere out of it", "walk",
        (3.0, 5.0), 0.8, motion="pace", drift=34.0,
    ),
    Vignette(
        # Rare and daft: the whole-body shudder, no water involved.
        "g_dog_shake", "shakes itself dry all over, like a dog", "idle",
        (1.6, 2.8), 0.35, motion="spin",
    ),
    Vignette(
        # The best joke in the band, so the rarest thing in it.
        "g_brace_nothing", "braces for a gust that never comes", "idle",
        (1.8, 3.0), 0.15, speech="?", motion="still",
    ),

    # ==================================================================
    #  Storm - loud weather, and the small human arithmetic of it.
    # ==================================================================
    Vignette(
        "g_count_thunder", "counts under its breath between the flash and the bang",
        "idle", (3.0, 5.0), 1.1, tags=("storm",), speech="?", motion="still",
    ),
    Vignette(
        "g_lean_gust", "leans back on the gust and lets it take its weight",
        "idle", (2.0, 3.2), 0.15, tags=("storm", "silly"), motion="still",
    ),
    Vignette(
        "g_high_ground", "thinks better of standing on the high ground", "panic",
        (2.0, 3.4), 0.5, tags=("storm", "high_up"), motion="crouch",
    ),
    Vignette(
        # The band's one long lie-down: too tired to fight the weather.
        "g_curl_lee", "curls up small with its back to the weather", "sleep",
        (5.0, 8.0), 0.35, tags=("storm", "tired"), speech="z", motion="lie",
    ),

    # ==================================================================
    #  Rain - what the hands do when the sky will not stop.
    # ==================================================================
    Vignette(
        "g_cup_rain", "cups both hands and drinks what the sky gives it", "idle",
        (2.5, 4.0), 1.1, tags=("raining",), speech="~", motion="still",
    ),
    Vignette(
        "g_shield_load", "hunches over what it carries to keep the rain off it",
        "carry", (2.0, 3.4), 1.0, tags=("raining", "carrying"), motion="still",
    ),
    Vignette(
        # Soaked, then stood by the fire: the only place this picture exists.
        "g_steam_off", "stands close enough to the fire to steam", "idle",
        (4.0, 6.5), 0.9, tags=("raining", "near_fire"), motion="still",
    ),
    Vignette(
        "g_leaf_umbrella", "holds a broad leaf over its head, hopefully", "carry",
        (2.5, 4.0), 0.15, tags=("raining", "near_tree"), motion="still",
    ),
    Vignette(
        "g_child_rain", "runs about with its face turned up into the rain",
        "dance", (2.5, 4.0), 0.8, tags=("raining", "child"), speech="*",
        motion="hop",
    ),

    # ==================================================================
    #  Snow and cold - footing, shoulders, and visible breath.
    # ==================================================================
    Vignette(
        "g_ice_slip", "slips on the ice and stays upright by luck alone", "fall",
        (1.5, 2.4), 0.15, tags=("snowing",), speech="!", motion="hop",
    ),
    Vignette(
        "g_brush_snow", "brushes the snow off its own shoulders", "idle",
        (1.6, 2.8), 1.1, tags=("snowing",), motion="spin",
    ),
    Vignette(
        "g_breath_cloud", "breathes out and watches it hang there", "idle",
        (2.0, 3.2), 1.2, tags=("cold",), speech="o", motion="still",
    ),

    # ==================================================================
    #  Wind - everything it takes an interest in.
    # ==================================================================
    Vignette(
        "g_wind_grit", "turns its back on the grit the wind is throwing", "idle",
        (1.8, 3.0), 1.1, tags=("windy",), motion="spin",
    ),
    Vignette(
        "g_shield_candle", "curls a hand round the candle flame", "carry",
        (2.0, 3.5), 0.9, tags=("windy", "candle"), motion="still",
    ),
    Vignette(
        "g_rock_lee", "gets in behind a rock until the wind gives up", "sleep",
        (3.5, 6.0), 0.7, tags=("windy", "near_rock"), motion="crouch",
    ),

    # ==================================================================
    #  Sun - the other weather, the one nobody complains about.
    # ==================================================================
    Vignette(
        "g_feet_dip", "sits and hangs its feet in the water", "sleep",
        (4.5, 7.5), 0.5, tags=("day", "near_water"), motion="sit",
    ),
    Vignette(
        "g_rainbow", "points at something bright in the sky and will not stop",
        "idle", (2.5, 4.0), 0.15, tags=("day", "social"), speech="!",
        motion="still",
    ),
    Vignette(
        "g_elder_weather", "declares it has seen far worse than this", "idle",
        (2.5, 4.5), 0.6, tags=("elder", "social"), motion="still",
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
    assert all(k.startswith("g_") for k in keys), "unprefixed key"
    assert all(MIN_DUR <= v.dur[0] <= v.dur[1] <= MAX_DUR for v in VIGNETTES)
    assert all(v.motion in MOTIONS for v in VIGNETTES)
    assert all(0.0 <= v.drift <= MAX_DRIFT for v in VIGNETTES)
    assert all(v.weight > 0.0 for v in VIGNETTES)
    assert all(v.speech is None or len(v.speech) == 1 for v in VIGNETTES)

    POSES = {"idle", "walk", "carry", "chop", "build", "climb", "fall",
             "sleep", "dance", "mourn", "panic"}
    assert all(v.pose in POSES for v in VIGNETTES), {v.pose for v in VIGNETTES} - POSES

    tagged = [v for v in VIGNETTES if v.tags]
    ungated = [v for v in VIGNETTES if not v.gates]
    short = [v for v in VIGNETTES if v.dur[1] <= 3.0]
    long_ = [v for v in VIGNETTES if v.dur[1] > 4.5]
    assert len(tagged) >= 14, len(tagged)
    assert len(ungated) >= 10, len(ungated)
    assert len(short) > len(long_), (len(short), len(long_))

    used = {v.motion for v in VIGNETTES}
    assert used == set(MOTIONS), set(MOTIONS) - used
    print(f"{len(VIGNETTES)} vignettes, {len(tagged)} tagged, "
          f"{len(ungated)} always-available, "
          f"{sum(1 for v in VIGNETTES if v.speech)} with speech")
    print("motions:", {m: sum(1 for v in VIGNETTES if v.motion == m)
                       for m in MOTIONS})
