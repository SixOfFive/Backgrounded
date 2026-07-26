"""Vignette content, band K - play and games.

Thirty cosmetic beats about *not working*: hopscotch scratched into the dirt,
a pebble thrown up and caught without looking, a rock leapfrogged without its
consent, a tug of war against nobody at all. Where band A is about having a
body and band B is about standing somewhere, this band is about the oldest
thing a person does with an idle afternoon - inventing a rule and then trying
to beat it.

Content only. Nothing here touches the world: no stockpile, no structure, no
prop, no other agent's state. A label that says "vaults over a friend's back"
does not move the friend - the friend is busy being a stickman two metres away
and the vault happens entirely in the viewer's head, which is the whole deal.
The state machine, the picker and the contract all live in ``vignettes.py``.

House rules for this file
-------------------------
* Keys are all prefixed ``k_`` so the content bands cannot collide.
* Poses are drawn from the renderer's vocabulary only (idle, walk, carry,
  chop, build, climb, fall, sleep, dance, mourn, panic). All eleven get used;
  a game is the one thing that legitimately needs the whole body.
* Gating tags mean something. Hide-and-seek wants a tree to hide behind,
  king-of-the-hill wants high ground, splash games want water. Two entries
  carry the decorative tag "silly" instead - that never gates, so the pool is
  never empty however dull the weather.
* Games are mostly *quick*: short beats outnumber long ones roughly two to
  one, because a round of anything is over before a stickman remembers it was
  supposed to be having fun. The long entries are the ones with an audience.
* The rare and the ridiculous sit at weight 0.15 so they stay a surprise.
"""
from __future__ import annotations

from .vignettes import Vignette

__all__ = ["VIGNETTES"]


VIGNETTES: list[Vignette] = [
    # ==================================================================
    #  Ungated - games that need nothing but ground and boredom. These
    #  fit any context and carry the band when the world is featureless.
    # ==================================================================
    Vignette(
        "k_hopscotch_grid", "hops a hopscotch grid scratched into the dirt",
        "walk", (2.2, 3.6), 1.1, motion="hop",
    ),
    Vignette(
        "k_throw_catch_blind",
        "throws a pebble up and catches it without looking", "chop",
        (1.8, 2.8), 1.2, motion="still",
    ),
    Vignette(
        "k_pebble_dribble", "dribbles a pinecone along between its feet",
        "walk", (2.6, 4.2), 1.0, motion="pace", drift=32.0,
    ),
    Vignette(
        "k_stick_palm", "balances a stick upright on one palm", "idle",
        (2.6, 4.4), 0.9, motion="still",
    ),
    Vignette(
        "k_flick_target", "flicks pebbles at a target scratched in the dirt",
        "build", (2.0, 3.0), 1.0, motion="crouch",
    ),
    Vignette(
        # The rival does not exist. The rival is comprehensively beaten.
        "k_race_self", "races an imaginary rival three strides and wins",
        "walk", (1.6, 2.6), 0.9, motion="pace", drift=34.0,
    ),
    Vignette(
        "k_hoop_stick", "rolls a bent withy hoop along with a stick", "walk",
        (3.2, 5.4), 0.6, motion="pace", drift=38.0,
    ),
    Vignette(
        "k_skip_rope", "skips a rope of twisted fibre", "walk",
        (2.0, 3.0), 0.8, motion="hop",
    ),
    Vignette(
        "k_heel_spin_record", "spins on one heel, counting the turns", "dance",
        (2.0, 3.0), 0.6, speech="o", motion="spin",
    ),
    Vignette(
        "k_stick_duel_bush", "fences a bush that never agreed to the duel",
        "chop", (1.8, 2.8), 0.5, motion="hop",
    ),
    Vignette(
        # Ambition outruns the arms. Rare, so it stays funny.
        "k_catch_miss", "throws a pebble higher than it can catch", "panic",
        (1.5, 2.4), 0.15, speech="!", motion="hop",
    ),
    Vignette(
        "k_bench_score", "sits out a round and keeps the score in the dirt",
        "sleep", (3.0, 5.0), 0.6, motion="sit",
    ),

    # ==================================================================
    #  Company - games that need somebody else in the frame, even if the
    #  somebody else is entirely uninvolved.
    # ==================================================================
    Vignette(
        "k_tag_it", "plays tag, and is very much It", "walk",
        (2.0, 3.0), 0.9, tags=("social",), motion="pace", drift=36.0,
    ),
    Vignette(
        "k_count_to_twenty", "covers its eyes and counts to twenty", "idle",
        (2.8, 4.4), 0.9, tags=("social",), motion="still",
    ),
    Vignette(
        "k_leapfrog_friend", "vaults over a friend's back", "climb",
        (1.5, 2.6), 0.8, tags=("social",), speech="!", motion="hop",
    ),
    Vignette(
        # Longest social beat in the band: it only works if watched.
        "k_play_dead", "plays dead until somebody notices", "sleep",
        (3.5, 6.0), 0.4, tags=("social",), speech="z", motion="lie",
    ),
    Vignette(
        "k_teach_the_rules",
        "teaches a child the rules and gets two of them wrong", "idle",
        (3.5, 6.0), 0.5, tags=("elder", "social"), motion="still",
    ),

    # ==================================================================
    #  Terrain as equipment - a tree to hide behind, a rock to lose
    #  behind, a mound worth defending.
    # ==================================================================
    Vignette(
        "k_hide_tree", "hides behind a tree and peers around it", "build",
        (2.2, 3.8), 1.0, tags=("near_tree",), speech="?", motion="crouch",
    ),
    Vignette(
        "k_search_rock",
        "searches behind a rock for somebody who is not there", "build",
        (2.4, 4.0), 0.5, tags=("near_rock", "alone"), motion="crouch",
    ),
    Vignette(
        "k_king_of_mound", "holds the top of the mound against all comers",
        "idle", (3.0, 5.5), 0.7, tags=("high_up",), motion="still",
    ),
    Vignette(
        "k_king_toppled", "is shoved off the top of the mound, laughing",
        "fall", (1.5, 2.4), 0.15, tags=("high_up", "social"), motion="hop",
    ),
    Vignette(
        "k_twig_race", "races two twigs down the water", "build",
        (2.6, 4.4), 0.8, tags=("near_water",), motion="crouch",
    ),

    # ==================================================================
    #  Weather as equipment - snow to pack, rain to dodge, firelight to
    #  make shapes in.
    # ==================================================================
    Vignette(
        "k_snowball_tree", "packs a snowball and lobs it at a tree", "chop",
        (1.8, 3.0), 0.9, tags=("snowing",), motion="still",
    ),
    Vignette(
        "k_dry_patch_hop", "hops between the dry patches, avoiding the wet",
        "walk", (2.0, 3.0), 0.8, tags=("raining",), motion="hop",
    ),
    Vignette(
        "k_shadow_puppets", "throws shadow shapes onto the firelit ground",
        "idle", (3.0, 5.2), 0.6, tags=("near_fire", "night"), speech="~",
        motion="still",
    ),

    # ==================================================================
    #  Winning, losing, and officiating.
    # ==================================================================
    Vignette(
        "k_victory_lap", "runs a small victory lap for nobody", "walk",
        (2.0, 3.0), 0.8, tags=("happy",), speech="*", motion="pace",
        drift=40.0,
    ),
    Vignette(
        "k_sore_loser", "loses three rounds running and takes it hard",
        "mourn", (2.4, 4.0), 0.4, tags=("sad",), motion="still",
    ),
    Vignette(
        "k_spot_game", "plays spot-the-thing on the horizon, keeping score",
        "idle", (2.6, 4.2), 0.6, tags=("lookout", "day"), motion="still",
    ),
    Vignette(
        # Decorative tag only: never gates, always ridiculous.
        "k_tug_nobody", "loses a tug of war against nobody at all", "carry",
        (2.4, 4.0), 0.15, tags=("silly",), motion="pace", drift=20.0,
    ),
    Vignette(
        "k_invisible_referee",
        "referees a bout between two invisible champions", "dance",
        (2.8, 4.6), 0.15, tags=("silly",), speech="!", motion="spin",
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
    assert all(k.startswith("k_") for k in keys), "unprefixed key"
    assert all(MIN_DUR <= v.dur[0] <= v.dur[1] <= MAX_DUR for v in VIGNETTES)
    assert all(v.motion in MOTIONS for v in VIGNETTES)
    assert all(0.0 <= v.drift <= MAX_DRIFT for v in VIGNETTES)
    assert all(v.weight > 0.0 for v in VIGNETTES)
    assert all(v.speech is None or len(v.speech) == 1 for v in VIGNETTES)

    poses = {"idle", "walk", "carry", "chop", "build", "climb", "fall",
             "sleep", "dance", "mourn", "panic"}
    assert all(v.pose in poses for v in VIGNETTES), \
        {v.pose for v in VIGNETTES} - poses

    tagged = [v for v in VIGNETTES if v.tags]
    ungated = [v for v in VIGNETTES if not v.gates]
    short = [v for v in VIGNETTES if v.dur[1] <= 3.0]
    long_ = [v for v in VIGNETTES if v.dur[1] > 4.5]
    spoken = [v for v in VIGNETTES if v.speech]
    assert 12 <= len(tagged) <= 20, len(tagged)
    assert len(ungated) >= 12, len(ungated)
    assert len(short) > len(long_), (len(short), len(long_))
    assert 6 <= len(spoken) <= 10, len(spoken)

    used = {v.motion for v in VIGNETTES}
    assert used == set(MOTIONS), set(MOTIONS) - used
    print(f"{len(VIGNETTES)} vignettes, {len(tagged)} tagged, "
          f"{len(ungated)} always-available, {len(spoken)} with speech")
    print("motions:", {m: sum(1 for v in VIGNETTES if v.motion == m)
                       for m in MOTIONS})
    print("poses:", sorted({v.pose for v in VIGNETTES}))
