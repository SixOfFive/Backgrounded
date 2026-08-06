"""Name, role and chronicle-phrase generation.

Pure python: no pygame, no numpy, no I/O. Safe to import headless.

Three things live here:

* **Roles** - the canonical ``ROLE_*`` strings every other module compares against.
* **Names** - :func:`new_name` builds short pronounceable consonant/vowel names
  ("Ka" + "ren" + "do" -> "Karendo") and guarantees the result is not already in
  the ``used`` set it is handed.
* **Chronicle phrasing** - :func:`describe_event` turns an event kind plus a few
  keyword facts into one human-readable sentence for the chronicle log.

Every function here is total: bad arguments produce a dull-but-valid result
rather than an exception, because the chronicle runs unattended for hours.
"""
from __future__ import annotations

import random
from typing import Any, Iterable

from ..constants import CAUSE_DISINTEGRATED, WORLD_W

# --------------------------------------------------------------------- roles --
ROLE_GATHERER = "gatherer"
ROLE_BUILDER = "builder"
ROLE_ELDER = "elder"
ROLE_CHILD = "child"
ROLE_LOOKOUT = "lookout"

ROLES: tuple[str, ...] = (
    ROLE_GATHERER, ROLE_BUILDER, ROLE_ELDER, ROLE_CHILD, ROLE_LOOKOUT,
)

#: Roles a grown stickman may hold (a child matures into one of these).
ADULT_ROLES: tuple[str, ...] = (ROLE_GATHERER, ROLE_BUILDER, ROLE_LOOKOUT, ROLE_ELDER)

#: Roles a freshly matured child picks from, weighted by listing frequency.
WORKING_ROLES: tuple[str, ...] = (
    ROLE_GATHERER, ROLE_GATHERER, ROLE_GATHERER, ROLE_BUILDER, ROLE_BUILDER,
    ROLE_LOOKOUT,
)

ROLE_LABELS: dict[str, str] = {
    ROLE_GATHERER: "Gatherer",
    ROLE_BUILDER: "Builder",
    ROLE_ELDER: "Elder",
    ROLE_CHILD: "Child",
    ROLE_LOOKOUT: "Lookout",
}


def is_role(value: Any) -> bool:
    """True if *value* is one of the canonical role strings."""
    return isinstance(value, str) and value in ROLES


def role_label(role: Any) -> str:
    """Human-readable name for a role, tolerant of junk input."""
    if isinstance(role, str):
        return ROLE_LABELS.get(role, role.replace("_", " ").title() or "Wanderer")
    return "Wanderer"


# ------------------------------------------------------------ rng plumbing --
# Callers pass either a ``random.Random`` or a ``numpy.random.Generator``. Both
# expose ``.random() -> float`` and nothing else in common, so everything here
# is built on that one call.
_FALLBACK = random.Random()


def _rand(rng: Any) -> float:
    """Uniform float in [0,1) from any rng-ish object, never raising."""
    if rng is not None:
        try:
            v = float(rng.random())
            if 0.0 <= v < 1.0:
                return v
            # numpy can hand back an array for some call shapes; ignore it.
        except Exception:
            pass
    return _FALLBACK.random()


def _pick(rng: Any, seq: Iterable[Any]) -> Any:
    items = tuple(seq)
    if not items:
        return ""
    return items[int(_rand(rng) * len(items)) % len(items)]


def _randint(rng: Any, lo: int, hi: int) -> int:
    """Inclusive integer in [lo, hi]."""
    if hi <= lo:
        return lo
    return lo + int(_rand(rng) * (hi - lo + 1)) % (hi - lo + 1)


# --------------------------------------------------------------- syllables --
# Onsets usable anywhere. Clusters only follow an open syllable, so the joins
# stay pronounceable ("Ka|ren|do", never "Karnvtro").
_ONSETS: tuple[str, ...] = (
    "b", "br", "d", "dr", "f", "g", "gr", "h", "k", "kr", "l", "m", "n", "p",
    "r", "s", "sh", "st", "t", "th", "tr", "v", "vr", "w", "y", "z", "kh",
    "gl", "sk", "pl", "ny",
)
_ONSETS_SIMPLE: tuple[str, ...] = (
    "b", "d", "f", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
)
_NUCLEI: tuple[str, ...] = (
    "a", "a", "a", "e", "e", "e", "i", "i", "i", "o", "o", "o", "u", "u",
    "ae", "ai", "ea", "ei", "ia", "oa", "ou", "y",
)
_CODAS: tuple[str, ...] = (
    "n", "n", "l", "l", "r", "r", "s", "m", "k", "th", "nd", "st", "rn", "sh",
)

_MIN_LEN = 4
_MAX_LEN = 9


def _syllable(rng: Any, *, allow_cluster: bool, coda_chance: float) -> tuple[str, bool]:
    onset = _pick(rng, _ONSETS if allow_cluster else _ONSETS_SIMPLE)
    nucleus = _pick(rng, _NUCLEI)
    coda = ""
    if _rand(rng) < coda_chance:
        coda = _pick(rng, _CODAS)
    return f"{onset}{nucleus}{coda}", bool(coda)


def _has_triple(word: str) -> bool:
    return any(word[i] == word[i + 1] == word[i + 2] for i in range(len(word) - 2))


def make_name(rng: Any = None, syllables: int = 0) -> str:
    """Build one pronounceable name. Not checked for uniqueness.

    ``syllables`` of 0 (the default) picks 2 or 3 at random.
    """
    try:
        count = int(syllables)
    except Exception:
        count = 0
    if count < 2 or count > 4:
        count = 2 if _rand(rng) < 0.62 else 3

    parts: list[str] = []
    prev_coda = False
    for i in range(count):
        last = i == count - 1
        syl, prev_coda = _syllable(
            rng,
            allow_cluster=(not prev_coda) and (i == 0 or not last),
            coda_chance=0.45 if last else 0.16,
        )
        if parts and syl == parts[-1]:          # no "Kaka"
            syl = syl[::-1] if len(syl) > 2 else syl + "n"
        parts.append(syl)

    word = "".join(parts)
    if _has_triple(word):
        word = word.replace(word[0] * 3, word[0] * 2)
    if len(word) < _MIN_LEN:
        word += _pick(rng, ("n", "th", "ra", "el"))
    if len(word) > _MAX_LEN:
        word = word[:_MAX_LEN].rstrip("aeiouy") or word[:_MAX_LEN]
    return word[:1].upper() + word[1:]


def syllables_of(name: str) -> list[str]:
    """Rough syllable split, used by :func:`child_name` for lineage echoes."""
    if not isinstance(name, str) or not name:
        return []
    out: list[str] = []
    cur = ""
    seen_vowel = False
    for ch in name.lower():
        low = ch in "aeiouy"
        if low:
            if seen_vowel and cur and cur[-1] not in "aeiouy":
                out.append(cur)
                cur = ""
                seen_vowel = False
            seen_vowel = True
        cur += ch
        if not low and seen_vowel and len(cur) >= 3:
            out.append(cur)
            cur = ""
            seen_vowel = False
    if cur:
        if out:
            out[-1] += cur
        else:
            out.append(cur)
    return out


_ROMAN = (
    "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
    "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
)


def _ordinal_suffix(n: int) -> str:
    idx = n - 2
    if 0 <= idx < len(_ROMAN):
        return _ROMAN[idx]
    return str(n)


def _remember(used: Any, name: str) -> None:
    try:
        used.add(name)
    except Exception:
        pass


def new_name(rng: Any = None, used: set[str] | None = None) -> str:
    """Return a name that is **not** in *used*, and record it there.

    The returned name is added to *used* (when *used* supports ``.add``) so a
    caller that keeps handing the same set back can never collide. If the
    random space is exhausted the name gains a lineage numeral - "Karendo II" -
    so uniqueness is guaranteed, not merely likely.
    """
    pool: set[str] = used if used is not None else set()
    for _ in range(256):
        name = make_name(rng)
        if name and name not in pool:
            _remember(pool, name)
            return name

    base = make_name(rng) or "Ren"
    n = 2
    while n < 10_000:
        cand = f"{base} {_ordinal_suffix(n)}"
        if cand not in pool:
            _remember(pool, cand)
            return cand
        n += 1
    cand = f"{base} {len(pool) + 1}"
    _remember(pool, cand)
    return cand


def child_name(
    rng: Any = None,
    used: set[str] | None = None,
    parents: Iterable[str] = (),
) -> str:
    """A new unique name that echoes a parent's first syllable when it can."""
    pool: set[str] = used if used is not None else set()
    stems = [s for s in (syllables_of(p) for p in parents if isinstance(p, str)) if s]
    if stems and _rand(rng) < 0.7:
        stem = _pick(rng, [s[0] for s in stems])
        for _ in range(96):
            tail_count = 1 if _rand(rng) < 0.6 else 2
            tail = make_name(rng, syllables=max(2, tail_count + 1))[1:].lower()
            cand = str(stem) + tail
            cand = cand[:_MAX_LEN] if len(cand) > _MAX_LEN else cand
            cand = cand[:1].upper() + cand[1:]
            if len(cand) >= _MIN_LEN and cand not in pool:
                _remember(pool, cand)
                return cand
    return new_name(rng, pool)


# ------------------------------------------------------------- place words --
def place_word(x: float | None = None) -> str:
    """Coarse compass-ish word for a world x coordinate (0..WORLD_W).

    WORLD, not stage: chronicle lines outlive the camera. "The eastern cliff
    took Karendo" has to mean the same place when it is read back an hour later
    from a colony that has since wandered somewhere else, so this normalises
    against the land and never against where anyone is standing.

    Seven buckets, up from five. On a 1600 px world five words gave a 350 px
    band and every event in a session got a genuinely different one. At 6400 px
    the same five give 1400 px bands, and since a settlement lives inside
    SITE_RANGE - under a tenth of the map - virtually every line it ever
    generates lands in the *same* bucket and the word stops carrying
    information. Seven does not fix that (nothing that reads only x can), but it
    roughly restores the old band width, which is what the phrasing was written
    against. "westward"/"eastward" rather than "near-western"/"mid-western"
    because every one of these has to survive being dropped into "the {where}
    cliff" and "the {where} camp" and still read as English.
    """
    try:
        v = float(x) / float(WORLD_W) if x is not None else 0.5
    except Exception:
        v = 0.5
    if v != v:                                   # NaN
        v = 0.5
    if v < 0.11:
        return "far western"
    if v < 0.27:
        return "western"
    if v < 0.43:
        return "westward"
    if v < 0.57:
        return "central"
    if v < 0.73:
        return "eastward"
    if v < 0.89:
        return "eastern"
    return "far eastern"


def height_word(y: float | None, ground: float | None = None) -> str:
    """"high"/"low" flavour word from a y coordinate against a reference."""
    try:
        if y is None or ground is None:
            return "high"
        return "high" if float(ground) - float(y) > 40.0 else "low"
    except Exception:
        return "high"


# ------------------------------------------------------------- chronicle --
#: Every event kind :func:`describe_event` knows. Unknown kinds still produce a
#: sentence, they just fall back to a generic phrasing.
EVENT_TEMPLATES: dict[str, tuple[str, ...]] = {
    # --- life
    "born": (
        "{name} was born to {parent}.",
        "{parent} welcomed {name} into the settlement.",
        "{name} drew first breath in the {where} camp.",
    ),
    "arrived": (
        "{name} arrived from beyond the ridge.",
        "{name} walked in out of the dark and stayed.",
        "A stranger named {name} joined the settlement.",
    ),
    "matured": (
        "{name} came of age and took up work as a {role}.",
        "{name} grew up. The settlement gained a {role}.",
    ),
    "role_change": (
        "{name} became a {role}.",
        "{name} set aside the old work and became a {role}.",
    ),
    # --- death
    "died_fall": (
        "{name} fell from the {where} cliff.",
        "{name} lost their grip on the {where} rock face and fell.",
        "The {where} cliff took {name}.",
    ),
    "died_lightning": (
        "Lightning found {name} in the open.",
        "{name} was struck by lightning and did not rise.",
    ),
    "died_fire": (
        "{name} was caught by the fire.",
        "The flames closed over {name}.",
    ),
    "died_mudslide": (
        "{name} was buried by the {where} slide.",
        "The hillside came down and {name} was under it.",
    ),
    "died_flood": (
        "The water rose faster than {name} could climb.",
        "{name} was carried off by the flood.",
    ),
    "died_cold": (
        "{name} froze in the night.",
        "The cold took {name} before dawn.",
    ),
    "died_hunger": (
        "{name} starved.",
        "There was nothing left to eat, and {name} died of it.",
    ),
    "died_meteor": (
        "A falling stone struck {name}.",
        "{name} was lost where the sky fell.",
    ),
    "died_quake": (
        "The ground opened and {name} went into it.",
    ),
    "died_age": (
        "{name} died old, by the fire.",
        "{name} lived long and died quietly.",
    ),
    # The wyrm-gun's kill, on a colonist. Registered here AND in DEATH_KINDS
    # below, and the pair is the whole point: a cause with a mapping but no
    # template renders as "Something happened.", and a cause with neither falls
    # through to the generic "{name} was lost to {cause}." - which is how this
    # one read until now ("X was lost to disintegrated.", a database field in the
    # middle of a chronicle).
    #
    # No {where} and no {other}: the beam kills at up to BFG_RANGE and through
    # anything but terrain, so the victim's own ground tells you nothing about
    # what happened, and naming the shooter here would be wrong on a misfire -
    # where the only person the gun kills is the one holding it. combat_actions
    # already chronicles the shot itself with the wielder's name, so the two
    # lines land together and the reader joins them up.
    "died_disintegrated": (
        "{name} was caught by the wyrm-gun's light, and there was nothing left.",
        "The green fire went through {name}.",
        "{name} came apart where the beam found them.",
    ),
    "died": (
        "{name} died.",
        "{name} was lost to {cause}.",
    ),
    "grave": (
        "A grave was raised for {name} on the {where} slope.",
        "{name} was buried where they fell.",
    ),
    "mourn": (
        "{name} stood a while at {other}'s grave.",
        "The settlement mourned {other}.",
    ),
    # --- work
    "build_start": (
        "{name} began the {structure}.",
        "{name} laid the first timbers of the {structure}.",
    ),
    "build_done": (
        "{name} finished the {structure}.",
        "The {structure} was finished by {name}.",
    ),
    "repair": (
        "{name} repaired the {structure}.",
    ),
    "chop": (
        "{name} felled a tree on the {where} slope.",
        "{name} brought a tree down.",
    ),
    "gather": (
        "{name} gathered {resource}.",
        "{name} carried {resource} back to the pile.",
    ),
    "haul": (
        "{name} hauled {resource} to the stockpile.",
    ),
    "plant": (
        "{name} planted a sapling on the {where} slope.",
    ),
    "harvest": (
        "{name} harvested the {where} bushes.",
    ),
    # --- needs
    "eat": (
        "{name} ate.",
        "{name} took a share of the {resource}.",
    ),
    "cook": (
        "{name} cooked at the firepit.",
    ),
    "sleep": (
        "{name} slept.",
        "{name} lay down in the hut.",
    ),
    "wake": (
        "{name} woke.",
    ),
    "huddle": (
        "{name} huddled at the fire.",
    ),
    "famine": (
        "The stores ran empty.",
        "There was no food left in the settlement.",
    ),
    # --- social
    "talk": (
        "{name} and {other} talked a while.",
        "{name} spoke with {other} in the dark.",
    ),
    "dance": (
        "The settlement danced for the new {structure}.",
        "{name} danced by the fire.",
    ),
    "milestone": (
        "The settlement grew to {count}.",
        "{count} now live here.",
    ),
    # --- the candle
    "candle_lit": (
        "{name} lit the candle.",
        "{name} took up the candle and carried it out.",
    ),
    "candle_out": (
        "The wind took {name}'s candle.",
        "{name}'s candle guttered out.",
    ),
    # --- weather and disaster
    "storm": (
        "A storm came in over the ridge.",
        "The sky closed and the rain started.",
    ),
    "rain_start": (
        "It began to rain.",
    ),
    "rain_stop": (
        "The rain thinned out and stopped.",
    ),
    "wildfire": (
        "Fire took the {where} treeline.",
        "A wildfire started on the {where} slope.",
    ),
    "mudslide": (
        "The {where} hillside gave way.",
    ),
    "blizzard": (
        "Snow came down hard and did not stop.",
    ),
    "flood": (
        "The water line rose over the low ground.",
    ),
    "meteor": (
        "Stones fell out of the sky.",
    ),
    "quake": (
        "The ground shook and split open.",
    ),
    "lightning": (
        "Lightning lit the whole valley for a moment.",
    ),
    "dawn": (
        "The sun came up.",
    ),
    "dusk": (
        "The light went out of the sky.",
    ),
    # --- housekeeping
    "world_new": (
        "The first stickmen woke on an empty hillside.",
    ),
    "loaded": (
        "The settlement carried on where it left off.",
    ),
}

EVENT_KINDS: tuple[str, ...] = tuple(EVENT_TEMPLATES)

_GENERIC = ("Something happened.", "{name} did something worth noting.")

_DEFAULTS: dict[str, str] = {
    "name": "Someone",
    "other": "someone",
    "parent": "the settlement",
    "structure": "structure",
    "resource": "supplies",
    "role": "wanderer",
    "where": "high",
    "count": "several",
    "cause": "the dark",
    "qty": "some",
    "scene": "the weather",
}


class _SafeFields(dict):
    """Format mapping that substitutes a dull default instead of raising."""

    def __missing__(self, key: str) -> str:      # pragma: no cover - trivial
        return _DEFAULTS.get(key, "something")


#: cause string (as stored on ``Stickman.death_cause``) -> chronicle kind.
DEATH_KINDS: dict[str, str] = {
    "fall": "died_fall",
    "fell": "died_fall",
    "lightning": "died_lightning",
    "fire": "died_fire",
    "burn": "died_fire",
    "mudslide": "died_mudslide",
    "buried": "died_mudslide",
    "flood": "died_flood",
    "drown": "died_flood",
    "cold": "died_cold",
    "freeze": "died_cold",
    "hunger": "died_hunger",
    "starve": "died_hunger",
    "meteor": "died_meteor",
    "quake": "died_quake",
    "age": "died_age",
    "old": "died_age",
    # The wyrm-gun. THE KEY IS THE IMPORTED CONSTANT ("disintegrated"), not a
    # literal like every other line above it, and the exception is deliberate.
    # Every key here is a cause string some other module writes onto
    # Stickman.death_cause, and the rest are written as literals in the module
    # that raises them, so a literal here can only drift if two literals drift
    # together. This one has a NAME - combat_actions imports
    # CAUSE_DISINTEGRATED - so spelling it out here would be the one place the
    # table could silently fall out of step with the code that fills it, and the
    # symptom would be soft: the death still happens, the chronicle just quietly
    # goes back to "X was lost to disintegrated." Importing costs nothing, since
    # this module already imports WORLD_W from the same file.
    CAUSE_DISINTEGRATED: "died_disintegrated",
}


def death_kind(cause: Any) -> str:
    """Map a free-form death cause onto a chronicle event kind."""
    if isinstance(cause, str):
        return DEATH_KINDS.get(cause.strip().lower(), "died")
    return "died"


def _tidy(line: str) -> str:
    line = " ".join(line.split())
    if not line:
        return ""
    if line[0].islower():
        line = line[0].upper() + line[1:]
    if line[-1] not in ".!?":
        line += "."
    return line


def describe_event(kind: str, **kw: Any) -> str:
    """One chronicle-ready sentence for *kind*.

    Facts arrive as keywords - ``name``, ``other``, ``structure``, ``role``,
    ``resource``, ``where``, ``count``, ``cause``. Anything the template wants
    but you did not supply gets a bland default, so this never raises.

    Pass ``x=<world x>`` and the ``{where}`` slot fills itself in from
    :func:`place_word`. Pass ``rng=`` to make the phrasing choice reproducible.

    ``describe_event("died_fall", name="Karendo", where="north")`` gives
    "Karendo fell from the north cliff."; ``describe_event("build_done",
    name="Vessa", structure="hut")`` gives "Vessa finished the hut."
    """
    try:
        rng = kw.pop("rng", None)
        if "where" not in kw and "x" in kw:
            kw["where"] = place_word(kw.get("x"))
        kw.pop("x", None)

        key = kind if isinstance(kind, str) else ""
        options = EVENT_TEMPLATES.get(key)
        if not options:
            options = _GENERIC
        template = str(_pick(rng, options))

        fields = _SafeFields()
        for k, v in kw.items():
            if v is None:
                continue
            fields[k] = v if isinstance(v, str) else str(v)
        return _tidy(template.format_map(fields))
    except Exception:
        # The chronicle must never be the thing that kills the app.
        try:
            who = str(kw.get("name", "Someone"))
        except Exception:
            who = "Someone"
        return f"{who} did something worth noting."


if __name__ == "__main__":   # pragma: no cover - smoke test
    r = random.Random(7)
    seen: set[str] = set()
    print("names:", ", ".join(new_name(r, seen) for _ in range(12)))
    print("child:", child_name(r, seen, ["Karendo", "Vessa"]))
    print(describe_event("died_fall", name="Karendo", x=120.0))
    print(describe_event("build_done", name="Vessa", structure="hut"))
    print(describe_event("milestone", count=9))
    print(describe_event("no_such_kind", name="Vessa"))
    print(describe_event("died", name="Vessa", cause="the cold"))
    assert len(seen) == 13, seen
