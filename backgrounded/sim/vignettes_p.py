"""Vignette content, band P - acrobatics and showing off.

The band the whole vignette engine was quietly waiting for: a villager with
nothing to do and morale to spare throws a cartwheel, walks on its hands, tries
a backflip and lands it (or does not). Nothing here touches the world - a
handstand gathers no wood - but it is the difference between a colony of workers
and a colony of *people* who mess about when the work is done.

These lean on the acrobatic poses added to ``render.stickfigure`` (``cartwheel``,
``handstand``, ``flip``, ``backflip``, ``highkick``, ``split``), which spin the
whole body about the hip rather than flailing limbs - far more legible at 21px.

House rules (same as bands A-O)
-------------------------------
* Keys are all prefixed ``p_`` so no band can collide.
* Poses come from the sim's vocabulary; the acrobatic ones are drawn directly.
* Real gating tags (``happy``, ``child``, ``social``, ``day``) *mean* something -
  you show off when you feel good, when there is someone to show, in the light.
  Plenty carry only a decorative tag ("silly", "showoff") so the pool is never
  empty however grim the mood.
* Motion stays ``still``: the pose already carries all the movement, and a
  ``pace`` drift would fight the spin.
"""
from __future__ import annotations

from .vignettes import Vignette

__all__ = ["VIGNETTES"]


VIGNETTES: list[Vignette] = [
    # ==================================================================
    #  Cartwheels - the signature move. Ungated ones keep the pool alive.
    # ==================================================================
    Vignette(
        "p_cartwheel", "throws a cartwheel", "cartwheel",
        (1.8, 2.8), 1.4, tags=("showoff",), motion="still",
    ),
    Vignette(
        "p_cartwheel_glee", "cartwheels for the sheer joy of it", "cartwheel",
        (2.2, 3.4), 1.5, tags=("happy",), speech="+", motion="still",
    ),
    Vignette(
        "p_cartwheel_show", "cartwheels past, hoping someone saw", "cartwheel",
        (2.0, 3.0), 1.3, tags=("social",), speech="*", motion="still",
    ),
    Vignette(
        "p_cartwheel_play", "practises cartwheels, wobbling on each landing",
        "cartwheel", (2.4, 3.6), 1.2, tags=("child",), motion="still",
    ),

    # ==================================================================
    #  Handstands - held, trembling, occasionally in daylight to be seen.
    # ==================================================================
    Vignette(
        "p_handstand", "kicks up into a handstand", "handstand",
        (2.0, 3.6), 1.3, tags=("showoff",), motion="still",
    ),
    Vignette(
        "p_handstand_hold", "holds a handstand far too long", "handstand",
        (3.4, 5.0), 1.0, tags=("happy",), speech="!", motion="still",
    ),
    Vignette(
        "p_handstand_day", "walks a few steps on its hands", "handstand",
        (2.4, 3.8), 1.1, tags=("day",), motion="still",
    ),
    Vignette(
        "p_handstand_kid", "tips upside down and giggles", "handstand",
        (2.0, 3.2), 1.2, tags=("child",), speech="+", motion="still",
    ),

    # ==================================================================
    #  Flips - front and back. Showy, quick, sometimes for an audience.
    # ==================================================================
    Vignette(
        "p_frontflip", "tucks into a front flip", "flip",
        (1.6, 2.4), 1.2, tags=("showoff",), motion="still",
    ),
    Vignette(
        "p_backflip", "throws a backflip and sticks the landing", "backflip",
        (1.6, 2.4), 1.3, tags=("happy",), speech="*", motion="still",
    ),
    Vignette(
        "p_backflip_try", "attempts a backflip, thinks better of it", "backflip",
        (1.8, 2.6), 1.0, tags=("social",), speech="?", motion="still",
    ),
    Vignette(
        "p_flip_combo", "flips forward, then back, showing off", "flip",
        (2.2, 3.2), 0.9, tags=("showoff",), speech="+", motion="still",
    ),

    # ==================================================================
    #  Kicks and stretches - the warm-up, and the limber ones.
    # ==================================================================
    Vignette(
        "p_highkick", "throws a high kick at nothing", "highkick",
        (1.8, 2.8), 1.2, tags=("showoff",), motion="still",
    ),
    Vignette(
        "p_highkick_drill", "drills high kicks, one after another", "highkick",
        (2.6, 3.8), 1.0, tags=("happy",), motion="still",
    ),
    Vignette(
        "p_shadowbox", "shadow-boxes an invisible foe", "highkick",
        (2.4, 3.6), 1.1, tags=("silly",), speech="!", motion="still",
    ),
    Vignette(
        "p_splits", "drops into the splits, wincing", "split",
        (2.2, 3.4), 1.0, tags=("showoff",), speech="…", motion="still",
    ),
    Vignette(
        "p_splits_stretch", "eases into the splits to limber up", "split",
        (2.6, 4.0), 0.9, tags=("day",), motion="still",
    ),

    # ==================================================================
    #  Dance-adjacent flourishes - reuse the dance body, framed as a bit.
    # ==================================================================
    Vignette(
        "p_victory_jig", "breaks into a little victory jig", "dance",
        (2.0, 3.2), 1.2, tags=("happy",), speech="+", motion="hop",
    ),
    Vignette(
        "p_show_dance", "busts out a few moves for an audience", "dance",
        (2.4, 3.6), 1.1, tags=("social",), speech="*", motion="hop",
    ),
    Vignette(
        "p_kid_spin", "spins in circles until dizzy", "dance",
        (2.2, 3.4), 1.2, tags=("child",), speech="~", motion="spin",
    ),
    Vignette(
        "p_freestyle", "freestyles a routine nobody asked for", "dance",
        (2.4, 3.8), 0.9, tags=("silly",), motion="hop",
    ),

    # ==================================================================
    #  Bows and flourishes - the finish. Reuse simple bodies, big energy.
    # ==================================================================
    Vignette(
        "p_take_a_bow", "takes an elaborate bow", "mourn",   # deep bow body
        (1.6, 2.6), 1.0, tags=("social",), speech="*", motion="still",
    ),
    Vignette(
        "p_flex", "flexes, admiring its own arms", "idle",
        (1.6, 2.6), 1.1, tags=("showoff",), speech="+", motion="still",
    ),
    Vignette(
        "p_balance_beam", "walks an imaginary tightrope, arms out", "climb",
        (2.2, 3.6), 1.0, tags=("silly",), motion="pace", drift=26.0,
    ),
]
