"""Vignette content, band J - ritual, custom and the things done *properly*.

Thirty cosmetic beats about a colony that has started to believe in something.
Nobody here is working. Somebody is laying a stone on the cairn, cutting the
day's tally into a post, walking the old boundary heel to toe, murmuring over
the fire, or touching the hut post on the way past because you always touch the
hut post on the way past. None of it does anything. That is what makes it a
ritual rather than a job.

The other bands cover the body (A), the ground and the weather (B), the
weather-as-feeling and the inner life (D) and outright clowning (E). This band
is deliberately the *formal* register: gestures with a right way and a wrong
way to do them, repeated because they were repeated before. Where a beat here
could be confused with one over there it is pushed further into custom - band B
draws something in the dirt, this band scratches a specific ward-sign and then
steps over it.

Contract, restated because content is where it gets broken
----------------------------------------------------------
* Inert data. Nothing below touches the stockpile, structures, props or the
  population; the state machine in :mod:`.vignettes` may move only the acting
  agent's pose, facing, speech and a few px of x. A label that says "lays a
  stone on the cairn" does not add a stone - the cairn is in the viewer's head.
* Every key is ``j_``-prefixed so the ten content bands cannot collide.
* Gating tags are ones :func:`~.vignettes.context_tags` actually evaluates, so
  they mean something: the vigil wants night, the ash wants a fire to have made
  it. Decorative tags ("superstition", "silly", "quiet") never gate, which
  keeps the pool non-empty in any weather.
* Short beats outnumber long ones. A rite is mostly a small gesture; the long
  ones - the vigil, the circling, the boundary walk - are rare on purpose.
"""
from __future__ import annotations

from .vignettes import Vignette

__all__ = ["VIGNETTES"]


VIGNETTES: list[Vignette] = [
    # ==================================================================
    #  Ungated custom - habits with no conditions attached. These are
    #  the band's floor: they fit any hour, any weather, any role.
    # ==================================================================
    Vignette(
        "j_tally_mark", "cuts the day's tally into the post", "chop",
        (1.7, 2.8), 1.2, motion="still",
    ),
    Vignette(
        # The smallest superstition in the colony, and the most common.
        "j_touch_post_luck", "touches the hut post for luck as it passes", "idle",
        (1.5, 2.4), 1.3, motion="still",
    ),
    Vignette(
        "j_palm_to_chest", "lays a palm flat on its chest and holds it there",
        "idle", (1.6, 2.6), 1.1, motion="still",
    ),
    Vignette(
        "j_moment_silence", "holds a moment of silence", "mourn",
        (2.4, 3.8), 1.1, motion="still",
    ),
    Vignette(
        "j_ward_sign", "scratches the ward-sign into the dirt and steps over it",
        "build", (2.6, 4.2), 0.9, motion="crouch",
    ),
    Vignette(
        "j_count_knots", "counts the knots on a cord, one for each", "idle",
        (2.4, 4.0), 0.9, tags=("quiet",), motion="still",
    ),
    Vignette(
        "j_practice_the_words",
        "mouths the words of the rite until they come out right", "idle",
        (2.6, 4.2), 0.8, tags=("quiet",), speech="~", motion="still",
    ),
    Vignette(
        # Facing each quarter in turn - the spin is the whole gesture.
        "j_greet_directions", "turns a slow full circle, greeting each direction",
        "idle", (3.0, 4.6), 0.8, motion="spin",
    ),
    Vignette(
        "j_three_turns", "turns about three times before it will go inside",
        "walk", (2.2, 3.4), 0.7, tags=("superstition",), motion="spin",
    ),
    Vignette(
        # The longest ungated beat: pacing a line only the colony can see.
        "j_boundary_pace", "walks the old boundary line, heel to toe", "walk",
        (4.0, 6.5), 0.9, tags=("superstition",), motion="pace", drift=38.0,
    ),

    # ==================================================================
    #  Rare and faintly ridiculous. Weight 0.15 keeps them special -
    #  a rite you see every ten minutes stops being funny.
    # ==================================================================
    Vignette(
        "j_hop_boundary_line", "hops the boundary line twice without touching it",
        "walk", (1.8, 2.8), 0.15, tags=("silly", "superstition"), speech="o",
        motion="hop",
    ),
    Vignette(
        "j_spit_luck", "spits over one shoulder, for luck", "idle",
        (1.5, 2.2), 0.15, tags=("silly", "superstition"), motion="still",
    ),
    Vignette(
        "j_wrong_rite", "performs entirely the wrong rite, with great confidence",
        "dance", (2.4, 4.0), 0.15, tags=("silly",), speech="*", motion="spin",
    ),

    # ==================================================================
    #  Fire - the colony's altar, whether or not anyone calls it that.
    # ==================================================================
    Vignette(
        "j_fire_murmur", "murmurs something over the fire", "idle",
        (3.0, 5.0), 1.1, tags=("near_fire",), speech="~", motion="still",
    ),
    Vignette(
        "j_ash_thumb", "marks its own brow with ash", "idle",
        (1.9, 3.0), 0.8, tags=("near_fire",), motion="still",
    ),
    Vignette(
        # Widdershins is wrong; the fire stays on the right hand.
        "j_circle_fire", "circles the fire slowly, keeping it on its right",
        "walk", (4.0, 7.0), 0.7, tags=("near_fire", "night"), motion="pace",
        drift=36.0,
    ),
    Vignette(
        "j_back_to_fire", "sits with its back to the fire, facing the dark",
        "sleep", (4.5, 7.5), 0.55, tags=("near_fire", "night"), motion="sit",
    ),

    # ==================================================================
    #  Stone, tree, water - the fixed points a custom attaches itself to.
    # ==================================================================
    Vignette(
        "j_stone_on_cairn", "lays a stone on the cairn and steps back", "build",
        (2.8, 4.4), 1.0, tags=("near_rock",), motion="crouch",
    ),
    Vignette(
        "j_tie_token", "reaches up to tie a token onto a low branch", "climb",
        (2.6, 4.2), 0.9, tags=("near_tree",), motion="still",
    ),
    Vignette(
        "j_wash_hands_rite", "washes its hands in the proper order, three times",
        "build", (2.6, 4.0), 0.9, tags=("near_water",), motion="crouch",
    ),

    # ==================================================================
    #  Grief and observance - the rites nobody had to be taught.
    # ==================================================================
    Vignette(
        "j_grave_stone", "carries a stone to the grave and sets it down", "carry",
        (3.0, 4.4), 0.7, tags=("sad",), motion="pace", drift=24.0,
    ),
    Vignette(
        "j_sit_vigil", "sits the vigil and does not move", "sleep",
        (5.0, 9.0), 0.5, tags=("night",), motion="sit",
    ),
    Vignette(
        # The rarest and longest thing in the band, and the strangest.
        "j_lie_in_ring", "lies down inside the stone ring and waits", "sleep",
        (5.0, 8.0), 0.2, tags=("night", "alone"), motion="lie",
    ),
    Vignette(
        "j_candle_murmur", "shields the candle with one hand and murmurs over it",
        "idle", (2.6, 4.2), 0.9, tags=("candle",), speech="~", motion="still",
    ),

    # ==================================================================
    #  The calendar - sunrise, first snow, the days that get marked.
    # ==================================================================
    Vignette(
        "j_sunrise_salute", "salutes the sunrise", "idle",
        (1.9, 3.0), 1.0, tags=("day",), motion="still",
    ),
    Vignette(
        "j_first_snow_greeting", "greets the first snow with both palms up",
        "idle", (2.0, 3.0), 0.8, tags=("snowing",), speech="*", motion="still",
    ),

    # ==================================================================
    #  Who does it - the same culture, performed differently by each.
    # ==================================================================
    Vignette(
        "j_elder_recites", "recites something long that only it still remembers",
        "idle", (3.6, 5.6), 0.7, tags=("elder",), motion="still",
    ),
    Vignette(
        "j_child_copies", "copies the rite a beat behind and gets it wrong",
        "dance", (2.2, 3.6), 0.8, tags=("child",), speech="?", motion="hop",
    ),
    Vignette(
        "j_watch_salute", "salutes the horizon before taking the watch", "idle",
        (1.8, 2.8), 0.9, tags=("lookout",), speech="!", motion="still",
    ),
    Vignette(
        "j_clasp_forearms", "clasps forearms with someone, formally", "idle",
        (1.8, 2.8), 1.0, tags=("social",), motion="still",
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
    assert all(k.startswith("j_") for k in keys), "unprefixed key"
    assert all(MIN_DUR <= v.dur[0] <= v.dur[1] <= MAX_DUR for v in VIGNETTES)
    assert all(v.motion in MOTIONS for v in VIGNETTES)
    assert all(0.0 <= v.drift <= MAX_DRIFT for v in VIGNETTES)
    assert all(v.weight > 0.0 for v in VIGNETTES)
    assert all(v.speech is None or len(v.speech) == 1 for v in VIGNETTES)

    gated = [v for v in VIGNETTES if v.gates]
    ungated = [v for v in VIGNETTES if not v.gates]
    short = [v for v in VIGNETTES if v.dur[1] <= 3.0]
    long_ = [v for v in VIGNETTES if v.dur[1] > 4.5]
    assert len(gated) >= 12 and len(ungated) >= 12, (len(gated), len(ungated))
    assert len(short) > len(long_), (len(short), len(long_))

    used = {v.motion for v in VIGNETTES}
    assert used == set(MOTIONS), set(MOTIONS) - used
    print(f"{len(VIGNETTES)} vignettes, {len(gated)} context-gated, "
          f"{len(ungated)} always-available, "
          f"{sum(1 for v in VIGNETTES if v.speech)} with speech")
    print("motions:", {m: sum(1 for v in VIGNETTES if v.motion == m)
                       for m in MOTIONS})
