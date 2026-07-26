"""Vignette content, band H - animals and small nature.

Thirty cosmetic beats about *the other things that live here*: the ant, the
beetle, the spider, the bird that will not stay still long enough to be
identified, the fly that has decided on this particular stickman. The colony
does not simulate any of them. There is no ant. What there is, is a stickman
crouched over nothing for four seconds with its head tilted, and a viewer who
supplies the ant for free - which is cheaper than an ant and works better.

Everything here is inert data. The state machine in :mod:`.vignettes` plays
it, and that machine may only touch the acting agent's pose, facing, speech,
``anim_t`` and a few px of x. A label that says "moves a worm off the path"
moves no worm; the world is byte-identical before and after.

House rules for this file
-------------------------
* Every key is ``h_``-prefixed so the content bands cannot collide.
* Poses come from the renderer's vocabulary - ``idle``, ``walk``, ``carry``,
  ``build``, ``climb``, ``sleep``, ``panic`` are the ones this band needs.
  ``build`` is the crouch-over-something body and does most of the work here,
  because most small nature happens at ankle height.
* Gating tags are ones :func:`~.vignettes.context_tags` really evaluates, so
  they *mean* something: moths come to a fire, worms come out in the rain,
  prints only survive in snow. A few entries carry a decorative tag ("silly",
  "quiet") which never gates - those keep the pool from ever emptying.
* Short beats outnumber long ones. Noticing a thing is quick; watching a
  spider finish a web is the exception, not the rule.
"""
from __future__ import annotations

from .vignettes import Vignette

__all__ = ["VIGNETTES"]


VIGNETTES: list[Vignette] = [
    # ==================================================================
    #  Something is on you - the fastest beats in the band. Ungated, so
    #  these are always available whatever the weather or the hour.
    # ==================================================================
    Vignette(
        "h_swat_fly", "swats at a fly that will not give up", "idle",
        (1.5, 2.4), 1.3, motion="hop",
    ),
    Vignette(
        # The spin is the point: it is tracking something it cannot see.
        "h_buzz_circle", "turns in circles after something buzzing round its head",
        "idle", (1.6, 2.8), 1.1, speech="?", motion="spin",
    ),
    Vignette(
        "h_grass_startle", "startles at something moving in the grass", "panic",
        (1.5, 2.4), 1.0, speech="!", motion="hop",
    ),
    Vignette(
        "h_freeze_listen", "goes completely still to hear what the grass is doing",
        "idle", (1.8, 3.0), 1.1, motion="still",
    ),

    # ==================================================================
    #  Ankle height - insects, and the small courtesies paid to them.
    # ==================================================================
    Vignette(
        "h_ant_burden", "watches an ant carry something twice its size", "build",
        (2.2, 3.6), 1.1, motion="crouch",
    ),
    Vignette(
        "h_beetle_cup", "cups a beetle in both hands, then opens them", "build",
        (2.6, 4.4), 1.0, speech="o", motion="crouch",
    ),
    Vignette(
        "h_beetle_right", "rights a beetle that has got onto its back", "build",
        (1.9, 3.2), 0.9, motion="crouch",
    ),
    Vignette(
        # Long-ish and rare: the whole joke is the round trip for a spider.
        "h_spider_carry", "carries a spider a few steps away and lets it off",
        "carry", (3.4, 5.4), 0.7, motion="pace", drift=36.0,
    ),
    Vignette(
        "h_track_hand", "sets a hand beside a track to see how big it is",
        "build", (2.4, 4.0), 1.0, motion="crouch",
    ),
    Vignette(
        "h_feather_turn", "picks up a feather and turns it over twice", "idle",
        (2.0, 3.4), 1.1, motion="still",
    ),

    # ==================================================================
    #  Birds - the only thing in the colony that gets to leave.
    # ==================================================================
    Vignette(
        "h_bird_overhead", "follows one bird all the way across the sky", "idle",
        (2.6, 4.4), 1.2, motion="spin",
    ),
    Vignette(
        "h_birdcall_try", "whistles back at a bird and gets it wrong", "idle",
        (2.0, 3.4), 0.8, tags=("day",), speech="~", motion="still",
    ),
    Vignette(
        "h_crumbs_nobody", "scatters crumbs for birds that never come", "build",
        (2.8, 4.4), 0.6, tags=("alone",), motion="crouch",
    ),
    Vignette(
        "h_high_birds", "watches the birds go past below it", "idle",
        (2.4, 4.0), 0.7, tags=("high_up",), motion="still",
    ),
    Vignette(
        "h_nest_crane", "cranes up at a nest it has no way of reaching", "climb",
        (2.4, 4.0), 0.9, tags=("near_tree",), motion="still",
    ),

    # ==================================================================
    #  Decorative tags only - these never gate, they are just rare. The
    #  0.15s are the ones that should feel like a gift when they land.
    # ==================================================================
    Vignette(
        "h_pigeon_walk", "walks a few steps with its head bobbing like a bird's",
        "walk", (1.8, 3.0), 0.15, tags=("silly",), motion="pace", drift=24.0,
    ),
    Vignette(
        "h_snail_cheer", "quietly encourages a snail", "build",
        (2.6, 4.2), 0.15, tags=("silly",), speech="~", motion="crouch",
    ),
    Vignette(
        "h_crow_stare", "loses a staring contest with a crow", "idle",
        (2.2, 3.4), 0.2, tags=("quiet",), motion="still",
    ),

    # ==================================================================
    #  Place - water, tree, rock. Each one only fires where it makes
    #  sense, which is what stops the band reading as generic pottering.
    # ==================================================================
    Vignette(
        "h_water_stalk", "creeps up on something at the water's edge", "build",
        (2.6, 4.2), 0.9, tags=("near_water",), motion="crouch",
    ),
    Vignette(
        "h_water_shadow", "watches a shadow move under the water", "idle",
        (2.8, 4.4), 0.8, tags=("near_water",), motion="still",
    ),
    Vignette(
        # The longest thing here, and deliberately so - a web takes a while.
        "h_web_finish", "watches a spider finish a web", "idle",
        (4.0, 7.0), 0.7, tags=("near_tree",), motion="still",
    ),
    Vignette(
        "h_lizard_rock", "outwaits a lizard sunning itself on a rock", "idle",
        (2.6, 4.5), 0.9, tags=("near_rock",), motion="still",
    ),
    Vignette(
        "h_bee_flower", "watches a bee work its way over one flower", "build",
        (2.4, 4.0), 1.0, tags=("day",), motion="crouch",
    ),
    Vignette(
        "h_grass_lie", "lies in the grass watching something climb a stem",
        "sleep", (4.5, 7.5), 0.35, tags=("happy",), motion="lie",
    ),

    # ==================================================================
    #  Weather brings its own animals out.
    # ==================================================================
    Vignette(
        "h_worm_path", "moves a worm off the path", "build",
        (2.0, 3.4), 0.9, tags=("raining",), motion="crouch",
    ),
    Vignette(
        "h_snow_prints", "follows a line of small prints until they stop", "walk",
        (2.8, 4.4), 0.8, tags=("snowing",), motion="pace", drift=32.0,
    ),

    # ==================================================================
    #  After dark - moths, crickets, and whatever is out there.
    # ==================================================================
    Vignette(
        "h_moth_chase", "chases a moth and keeps losing it", "walk",
        (3.0, 5.0), 0.8, tags=("night",), speech="!", motion="pace", drift=34.0,
    ),
    Vignette(
        "h_fire_moths", "watches the moths come to the fire", "idle",
        (3.0, 5.0), 0.9, tags=("near_fire", "night"), speech="*", motion="still",
    ),
    Vignette(
        "h_cricket_count", "sits and counts along with the crickets", "sleep",
        (3.6, 6.0), 0.6, tags=("night",), speech="~", motion="sit",
    ),

    # ==================================================================
    #  Shown to somebody else - the only social beat in the band, and the
    #  only one where the imaginary animal is imaginary for two people.
    # ==================================================================
    Vignette(
        "h_bug_show", "shows someone a beetle, very carefully", "carry",
        (2.0, 3.4), 0.8, tags=("child", "social"), speech="!", motion="still",
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
    assert all(k.startswith("h_") for k in keys), "unprefixed key"
    assert all(MIN_DUR <= v.dur[0] <= v.dur[1] <= MAX_DUR for v in VIGNETTES)
    assert all(v.motion in MOTIONS for v in VIGNETTES)
    assert all(0.0 <= v.drift <= MAX_DRIFT for v in VIGNETTES)
    assert all(v.weight > 0.0 for v in VIGNETTES)
    assert all(v.speech is None or len(v.speech) == 1 for v in VIGNETTES)
    assert all(v.pose in {"idle", "walk", "carry", "chop", "build", "climb",
                          "fall", "sleep", "dance", "mourn", "panic"}
               for v in VIGNETTES)

    tagged = [v for v in VIGNETTES if v.tags]
    ungated = [v for v in VIGNETTES if not v.gates]
    short = [v for v in VIGNETTES if v.dur[1] <= 3.4]
    long_ = [v for v in VIGNETTES if v.dur[1] > 4.5]
    assert len(tagged) >= 12, len(tagged)
    assert len(ungated) >= 12, len(ungated)
    assert len(short) > len(long_), (len(short), len(long_))

    used = {v.motion for v in VIGNETTES}
    assert used == set(MOTIONS), set(MOTIONS) - used
    print(f"{len(VIGNETTES)} vignettes, {len(tagged)} tagged, "
          f"{len(ungated)} always-available, "
          f"{sum(1 for v in VIGNETTES if v.speech)} with speech")
    print("motions:", {m: sum(1 for v in VIGNETTES if v.motion == m)
                       for m in MOTIONS})
