"""Headless integration smoke test.

Runs the simulation with no display at all, which is the point of the rule that
sim/ must not import pygame. Checks the invariants that matter for something
meant to run unattended for hours:

  * every module imports
  * sim/ is genuinely headless
  * a long run produces no exceptions and no NaNs
  * agents stay inside the world and on the ground
  * the world really is WORLD_W wide and the colonies really use it
  * a save round-trips without losing state, at the threshold as well as below it
  * no save, however mangled, can raise out of from_dict, disable a subsystem,
    leave the population books unbalanced, or put an impossible roster on the map
  * a save's declared terrain width never overrules the payload it carries
  * something actually *happens* (the colony is not inert)

A NOTE ON WHAT A CHECK IS FOR, because this file has been burned three times by
the same mistake. ``0 <= a.x <= RENDER_W`` passed for a whole round while every
agent on 13 of 14 seeds was pinned against an invisible wall. A junk sweep passed
while eight scalar fields raised straight out of from_dict. A round-trip check
passed because its fixture sat below the one threshold that breaks it. In each
case the assertion was true and told us nothing. So the bar for anything added
here is not "does it pass" - it is *would it still pass if the thing it claims to
verify were removed*. Where the answer is uncomfortable, the check carries a
GUARD alongside it: a second assertion that the fixture really is in the state
that makes the first one load-bearing. There are four of those now, all labelled.

Usage:  python tools/smoke.py [--ticks 3600] [--scene night_storm]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAILURES: list[str] = []
CHECKS = 0


def check(cond: bool, label: str, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  PASS  {label}")
        return True
    msg = f"{label}" + (f" -- {detail}" if detail else "")
    print(f"  FAIL  {msg}")
    FAILURES.append(msg)
    return False


def section(name: str) -> None:
    print(f"\n=== {name} ===")


# ------------------------------------------------------- save comparison --
def _canon(d) -> str:
    """A save as one canonical string, so two of them can be compared at all."""
    return json.dumps(d, separators=(",", ":"), sort_keys=True)


def _fields(obj, prefix: str = "") -> dict[str, object]:
    """Flatten a save to ``{dotted.path: leaf}``.

    Field level, not section level. The version of the round-trip check this
    replaces printed ``section 'population' differs`` and stopped there, which
    is the difference between "something in a 180 KiB blob changed" and "one
    named float changed by 0.03". Every diff below is reported as a path, a
    before and an after, because a comparison you cannot read is a comparison
    somebody will eventually delete.
    """
    out: dict[str, object] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_fields(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_fields(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


_MISSING = object()


def _field_diff(a, b) -> list[tuple[str, object, object]]:
    fa, fb = _fields(a), _fields(b)
    return [(k, fa.get(k, _MISSING), fb.get(k, _MISSING))
            for k in sorted(set(fa) | set(fb))
            if fa.get(k, _MISSING) != fb.get(k, _MISSING)]


#: Dotted paths whose value on the way OUT may legitimately differ from the
#: value that went IN, with the reason. Anything else that moves is a loss.
#:
#: This exists because "save -> load -> save is byte-identical" is NOT TRUE in
#: general, and the check that asserted it passed only because its fixture never
#: reached the one place it breaks. ``World.from_dict`` reads the stone-hut dwell
#: accumulator through ``_finite(..., hi=HUT_TIER_DWELL_SEC)``, and the
#: accumulator overshoots: it adds a whole ``dt`` per tick and latches on the
#: first value at or above 180.0, so a world that has earned the tier carries
#: 180.0333333333257 and loads back as 180.0. Measured on seed 7 at 45 sim-min,
#: reproduced identically by the cheap fixture in test_roundtrip_at_clamp.
#:
#: The clamp is right and stays: an infinity in a hand-edited file must not be
#: able to buy the tier. So the CHECK is what changes - to the two things that
#: are actually true, which are that nothing outside this table moves, and that
#: the value settles from the second round on.
#:
#: Add to this only with a measurement and a reason. An entry here is a licence
#: to lose a field on load, and the licence is exactly as wide as the table.
CLAMPED_ON_LOAD: dict[str, str] = {
    "hut_tier_dwell": "clamped to HUT_TIER_DWELL_SEC by _finite() on load; "
                      "the per-tick accumulator overshoots the threshold",
}


def _roundtrip(world, loader=None) -> dict:
    """``to_dict -> load -> to_dict -> load -> to_dict``, as canonical strings.

    ``loader`` is injectable for one reason: a check nobody has ever seen fail
    is not a check. The negative control runs this same function with a
    deliberately lossy loader and watches the assertions below go red; without
    the seam that proof would have to patch the repo, which is exactly what this
    project has agreed never to do to make a measurement come out.
    """
    from backgrounded.sim.world import World
    load = loader or World.from_dict

    d0 = world.to_dict()
    s0 = _canon(d0)
    d1 = load(json.loads(s0)).to_dict()
    s1 = _canon(d1)
    d2 = load(json.loads(s1)).to_dict()
    s2 = _canon(d2)
    return {"d0": d0, "d1": d1, "d2": d2, "s0": s0, "s1": s1, "s2": s2,
            "diff1": _field_diff(d0, d1), "diff2": _field_diff(d1, d2)}


def _show_diff(diff, limit: int = 6) -> None:
    for path, before, after in diff[:limit]:
        b = "<absent>" if before is _MISSING else repr(before)
        a = "<absent>" if after is _MISSING else repr(after)
        print(f"        {path}: {b} -> {a}")
    if len(diff) > limit:
        print(f"        ... and {len(diff) - limit} more")


# --------------------------------------------------------------- imports --
def test_imports() -> bool:
    section("imports")
    # This list was a coverage hole and the hole had consequences. It named no
    # module under render/ at all, and lagged sim/ by whatever was added last -
    # so a render module that did not exist (thrown spears were modelled and
    # never drawn, for 230 throws across 16 colonies) and a sim module nobody
    # imported both sat behind a green 26/26. An import test is the cheapest
    # check here and the only one that catches "the file is not there yet".
    #
    # sim/ first and alone, because test_sim_is_headless asserts below that
    # nothing imported so far has pulled in pygame - importing render/ before
    # that would be importing pygame and the assertion would be meaningless.
    mods = [
        "backgrounded.constants", "backgrounded.paths", "backgrounded.config",
        "backgrounded.sim.terrain", "backgrounded.sim.props",
        "backgrounded.sim.entities", "backgrounded.sim.names",
        "backgrounded.sim.lighting", "backgrounded.sim.events",
        "backgrounded.sim.structures", "backgrounded.sim.actions",
        "backgrounded.sim.behavior", "backgrounded.sim.animals",
        "backgrounded.sim.combat_actions", "backgrounded.sim.dragons",
        "backgrounded.sim.throwing", "backgrounded.sim.ufo",
        "backgrounded.sim.interact", "backgrounded.sim.world",
        "backgrounded.persist",
    ]
    ok = True
    for m in mods:
        try:
            __import__(m)
            print(f"  PASS  import {m}")
        except Exception as exc:
            print(f"  FAIL  import {m}: {type(exc).__name__}: {exc}")
            FAILURES.append(f"import {m}: {exc}")
            ok = False
    return ok


#: An import statement, at the start of a line, naming a module. Deliberately
#: anchored to the line so the prose does not match: half a dozen files in sim/
#: mention render/ in a comment explaining WHY they may not import it, and a
#: substring search calls every one of them an offender.
_IMPORTS_RE = r"^\s*(?:from\s+[.\w]*\b{0}\b|import\s+[.\w]*\b{0}\b)"


def test_sim_is_headless() -> None:
    """sim/ imports neither pygame nor render/ - BOTH halves of the rule.

    The rule in CLAUDE.md is "backgrounded/sim/** must never import pygame or
    anything from render/", and only the first half was ever checked. The second
    half is the one with teeth in the other direction: pygame announces itself
    (sim/ stops being runnable headless the moment it arrives), while a stray
    ``from backgrounded.render import fx`` costs nothing at import time and
    quietly makes the whole simulation depend on a display. Three files in sim/
    carry a comment saying they copy a constant out of render/ BY HAND because
    of this rule - which is the maintenance cost being paid for an invariant
    nothing was enforcing.

    Two forms of each, because they fail differently. The runtime check
    (sys.modules) catches any route in, including a lazy import inside a
    function, but only for code that actually ran. The source check catches an
    import in a module the smoke run never touches, but only if it is spelled
    the ordinary way.
    """
    section("sim/ is headless")
    # pygame must not have been pulled in by any sim import above.
    check("pygame" not in sys.modules,
          "no sim module imported pygame",
          f"loaded by: {[m for m in sys.modules if 'pygame' in m][:3]}")
    # ...and neither must render/. Same argument, same moment: this runs before
    # test_render_imports, so anything from render/ in sys.modules here can only
    # have been dragged in by sim/.
    pulled = [m for m in sys.modules if m.startswith("backgrounded.render")]
    check(not pulled, "no sim module imported anything from render/",
          f"loaded: {sorted(pulled)[:3]}")

    root = Path(__file__).resolve().parent.parent / "backgrounded" / "sim"
    pg = re.compile(_IMPORTS_RE.format("pygame"), re.M)
    rn = re.compile(_IMPORTS_RE.format("render"), re.M)
    off_pg, off_rn = [], []
    for p in sorted(root.glob("*.py")):
        src = p.read_text("utf-8", errors="ignore")
        if pg.search(src):
            off_pg.append(p.name)
        if rn.search(src):
            off_rn.append(p.name)
    check(not off_pg, "no module in sim/ imports pygame", str(off_pg))
    check(not off_rn, "no module in sim/ imports render/", str(off_rn))


def test_render_imports() -> None:
    """Every render module imports, under a dummy video driver.

    Runs strictly AFTER test_sim_is_headless, which asserts pygame has not been
    loaded yet - importing render/ is importing pygame, so doing it any earlier
    would make that assertion vacuous.

    This exists because a render module that was never written stayed invisible
    behind a green suite: spears were modelled and thrown 230 times across 16
    colonies with nothing on screen, because render/throwing.py did not exist and
    nothing here would have noticed. Discovering every module on disk rather than
    listing them means the next one is covered without anyone remembering.
    """
    section("render/ imports")
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    root = Path(__file__).resolve().parent.parent / "backgrounded" / "render"
    names = sorted(p.stem for p in root.glob("*.py") if p.stem != "__init__")
    check(bool(names), "found render modules to import", str(names))
    for name in names:
        mod = f"backgrounded.render.{name}"
        try:
            __import__(mod)
            check(True, f"import {mod}")
        except Exception as exc:
            check(False, f"import {mod}", f"{type(exc).__name__}: {exc}")


# ------------------------------------------------- the guard's own control --
def test_disable_guard() -> None:
    """A positive control for ``World._guarded``, which six checks lean on.

    Six assertions in this file are of the form ``not world._disabled`` - after
    the long run, after a reload, after a reload at the clamp, and over 24,000
    malformed saves. Every one of them is worth exactly as much as the mechanism
    that fills that set, and NOT ONE of them exercises it. If ``_guarded`` ever
    stopped recording - a rename, a swallowed exception, an early return - all
    six would go green forever and the suite would report a perfectly healthy
    world with its subsystems switched off one by one. That is the same shape as
    every other failure this file has been burned by, one level down.

    So: make a subsystem raise, on a real World, through the real guard, and
    watch the set fill. Then check the second half of the contract too - a
    disabled subsystem is never called again - because "records the name" and
    "actually stops running it" are separate claims and the checks above depend
    on both.
    """
    section("the disable guard itself")
    from backgrounded.sim.world import World

    w = World(seed=4242)
    check(not w._disabled, "a fresh world starts with nothing disabled",
          str(sorted(w._disabled)))

    def _boom():
        raise RuntimeError("smoke: deliberate subsystem failure")

    w._guarded("smoke_probe", _boom)
    check("smoke_probe" in w._disabled,
          "_guarded records a subsystem that raises",
          f"_disabled={sorted(w._disabled)} - the six 'nothing disabled' "
          f"checks in this file cannot fail while this is broken")

    calls: list[int] = []
    w._guarded("smoke_probe", lambda: calls.append(1))
    check(not calls, "...and never calls it again",
          f"called {len(calls)} more time(s)")

    # ...and it disables only what failed. A guard that emptied or filled the
    # set wholesale would pass both checks above.
    w2 = World(seed=4242)
    w2._guarded("smoke_probe", _boom)
    w2._guarded("smoke_other", lambda: None)
    check(w2._disabled == {"smoke_probe"},
          "...and disables only the subsystem that raised",
          str(sorted(w2._disabled)))


# ------------------------------------------------------------------- run --
def test_long_run(ticks: int, scene: str):
    section(f"simulate {ticks} ticks ({ticks / 30:.0f}s of world time)")
    from backgrounded.constants import RENDER_H, RENDER_W, SIM_DT, WORLD_W
    from backgrounded.sim.world import World

    world = World(seed=12345, scene=scene)
    # Was ``check(True, "world generated ...")`` - a line that could not fail,
    # counted in the total as if it could. It is a real claim now, and the two
    # halves are chosen so the wrong answer is not also a passing one: the
    # heightmap is WORLD_W columns (RENDER_W would fail it) and the land is not
    # flat (a generator that returned zeros would fail it, and a flat world
    # passes an alarming number of the checks further down).
    cols = int(world.terrain.height.size)
    relief = float(world.terrain.height.max() - world.terrain.height.min())
    check(cols == WORLD_W and relief > 1.0,
          f"world generated (seed={world.seed}, {cols} terrain columns, "
          f"relief {relief:.0f}px)",
          f"cols={cols} (WORLD_W={WORLD_W}, RENDER_W={RENDER_W}) relief={relief:.2f}")

    start_pop = len(world.population.alive_agents())
    errors = 0
    for i in range(ticks):
        try:
            world.tick(SIM_DT)
        except Exception:
            errors += 1
            if errors <= 3:
                traceback.print_exc()
    check(errors == 0, "no exceptions escaped world.tick()", f"{errors} raised")
    check(not world._disabled, "no subsystem disabled itself",
          str(sorted(world._disabled)))

    # invariants. WORLD_W, not RENDER_W: the bound is how much LAND there is,
    # and the two stopped being the same number when the world went to 6400 px.
    # See test_world_width below for why this line on its own is not enough.
    bad_pos, bad_num = [], []
    for a in world.population.agents:
        if not (0 <= a.x <= WORLD_W and -50 <= a.y <= RENDER_H + 50):
            bad_pos.append((a.name, round(a.x, 1), round(a.y, 1)))
        if any(math.isnan(v) or math.isinf(v) for v in (a.x, a.y, a.vx, a.vy)):
            bad_num.append(a.name)
    check(not bad_pos, "all agents inside the world bounds", str(bad_pos[:4]))
    check(not bad_num, "no NaN/inf in agent state", str(bad_num[:4]))

    grounded = 0
    for a in world.population.alive_agents():
        if abs(a.y - world.terrain.ground_y(a.x)) < 40:
            grounded += 1
    alive = len(world.population.alive_agents())
    check(alive > 0, "somebody is still alive", f"pop={alive}")
    if alive:
        check(grounded >= alive * 0.6,
              "most agents are on the ground",
              f"{grounded}/{alive} grounded")

    check(all(v >= 0 for v in world.stockpile.values()),
          "no negative stockpile", str(world.stockpile))

    # did anything actually happen?
    gathered = sum(world.stockpile.values())
    activity = (gathered > 0 or world.structures.count() > 0
                or world.stats["died"] > 0 or len(world.chronicle) > 1)
    check(activity, "the colony did something",
          f"stock={world.stockpile} structures={world.structures.count()} "
          f"chronicle={len(world.chronicle)}")

    print(f"\n  world_time  {world.world_time:.0f}s")
    print(f"  population  {alive} (started {start_pop}, gen {world.population.generation})")
    print(f"  stockpile   {world.stockpile}")
    print(f"  structures  {[(s.kind, s.stage, round(s.progress,2)) for s in world.structures.all()][:8]}")
    print(f"  stats       {world.stats}")
    print(f"  scene       {world.events.scene}")
    if world.chronicle:
        print("  chronicle tail:")
        for line in list(world.chronicle)[-6:]:
            print(f"     {line}")
    return world


# ------------------------------------------------------------ world width --
#: Seeds the width sweep runs. Fixed, so a failure is reproducible, and more
#: than one, because ONE seed is what let the old bounds check pass while the
#: bug was live: it ran World(seed=12345) only, that colony happens to seat low,
#: and ``0 <= a.x <= RENDER_W`` was therefore satisfied by a tree in which every
#: agent on 13 of 14 seeds was pinned against an invisible wall at x = 1599.
#: These six were picked to spread their seats across the map - measured, they
#: seat at x = 5445, 4951, 1823, 2402, 2059 and 2253, and over the two minutes
#: below their agents range x = 1126 .. 5830 - rather than for any property of
#: their colonies.
WIDTH_SWEEP_SEEDS = (1, 42, 314, 2718, 90210, 8675309)
#: Sim-minutes each sweep seed runs. Short on purpose - this is a smoke test,
#: not a balance run. Two minutes is enough for everyone to leave the spawn
#: huddle and start walking, which is all the check needs.
WIDTH_SWEEP_MIN = 2.0


def test_world_width() -> None:
    """The world is WORLD_W wide and the sim must actually use it.

    Three separate claims, because the shipped tree passed the weakest one:

      1. the module-level edge clamps really are the world's edges. Three files
         each carried their own ``_EDGE_MAX = float(RENDER_W - 1)`` - agents,
         dropped relics and thrown spears - and any one of them left behind
         re-pins that class of object at x = 1599 on a 6400 px map while every
         behavioural check still goes green;
      2. across several seeds nobody ever leaves the land, checked every tick
         rather than at the end - a body hauled back by the next clamp is still
         a body that left; and
      3. across several seeds somebody actually STANDS past the old frame
         width, and nobody is pinned to the old frame edge. That is the claim
         the single-seed version could not make and the one that fails on a
         tree with the bug in it: with ``_EDGE_MAX = RENDER_W - 1`` no agent can
         hold an x above 1599 at all, so it fails by a factor of three rather
         than by a hair.
    """
    section("world width")
    from backgrounded.constants import RENDER_W, SIM_DT, SIM_HZ, WORLD_W
    from backgrounded.sim import entities, items, throwing
    from backgrounded.sim.world import World

    check(WORLD_W > RENDER_W,
          "WORLD_W is wider than the camera",
          f"WORLD_W={WORLD_W} RENDER_W={RENDER_W}")

    want = float(WORLD_W - 1)
    for mod in (entities, items, throwing):
        got = float(getattr(mod, "_EDGE_MAX"))
        check(got == want,
              f"{mod.__name__.split('.')[-1]}._EDGE_MAX is the edge of the LAND",
              f"{got} != {want} (RENDER_W-1 is {float(RENDER_W - 1)})")

    ticks = int(WIDTH_SWEEP_MIN * 60 * SIM_HZ)
    lo = float("inf")
    hi = float("-inf")
    seats: list[float] = []
    out_of_bounds: list[tuple] = []
    nan: list[tuple] = []
    pinned: list[tuple] = []
    old_edge = float(RENDER_W - 1)

    for seed in WIDTH_SWEEP_SEEDS:
        w = World(seed=seed)
        agents = w.population.alive_agents()
        if agents:
            seats.append(sum(a.x for a in agents) / len(agents))
        for i in range(ticks):
            try:
                w.tick(SIM_DT)
            except Exception:
                traceback.print_exc()
                break
            # Every tick, not just the last one: a body that leaves the world
            # and is hauled back by the next clamp is still a body that left.
            for a in w.population.agents:
                x = a.x
                if math.isnan(x) or math.isinf(x):
                    nan.append((seed, a.name))
                    continue
                if not (0.0 <= x <= float(WORLD_W)):
                    out_of_bounds.append((seed, a.name, round(x, 1)))
                if x < lo:
                    lo = x
                if x > hi:
                    hi = x
            if i % SIM_HZ == 0:
                for a in w.population.alive_agents():
                    # The fingerprint of the bug, and it is a fingerprint rather
                    # than a coincidence: x = RENDER_W - 1 exactly, repeatedly,
                    # is what a clamp produces and what free walking never does.
                    if abs(a.x - old_edge) < 1e-6:
                        pinned.append((seed, a.name))

    check(not out_of_bounds,
          f"no agent ever leaves 0..WORLD_W on any of {len(WIDTH_SWEEP_SEEDS)} seeds",
          str(out_of_bounds[:4]))
    check(not nan, "no NaN/inf agent x across the sweep", str(nan[:4]))
    check(not pinned,
          "nobody is pinned to the old frame edge (x == RENDER_W-1)",
          str(sorted(set(pinned))[:6]))

    # These two are what give the bounds check above its teeth. Written as
    # "past the OLD edge" rather than as a span, because a span check would
    # have passed by 1 px on a buggy tree (max span 1599 against RENDER_W 1600)
    # while these fail against a measured 5445 and 5830.
    span = (hi - lo) if seats else 0.0
    seat_span = (max(seats) - min(seats)) if seats else 0.0
    check(bool(seats) and max(seats) > RENDER_W,
          "the sweep seeds a colony past the old frame width",
          f"seats {[round(s) for s in seats]} - none above RENDER_W {RENDER_W}")
    check(hi > RENDER_W,
          "somebody actually stands past the old frame width",
          f"furthest agent x {hi:.1f} <= RENDER_W {RENDER_W}")
    print(f"\n  swept {len(WIDTH_SWEEP_SEEDS)} seeds x {WIDTH_SWEEP_MIN:.0f} sim-min")
    print(f"  agent x range  {lo:.1f} .. {hi:.1f}  "
          f"({span / WORLD_W * 100:.1f}% of WORLD_W)")
    print(f"  colony seats   {[round(s) for s in seats]} "
          f"(span {seat_span:.0f})")


# ----------------------------------------------------------- persistence --
def test_roundtrip(world, loader=None) -> None:
    section("save / load round-trip")
    from backgrounded.sim.world import World
    if loader is None:
        loader = World.from_dict

    try:
        blob = json.dumps(world.to_dict())
    except Exception as exc:
        check(False, "world serialises to JSON", f"{type(exc).__name__}: {exc}")
        return
    check(True, f"world serialises to JSON ({len(blob)/1024:.0f} KiB)")

    try:
        clone = loader(json.loads(blob))
    except Exception as exc:
        check(False, "world deserialises", f"{type(exc).__name__}: {exc}")
        return
    check(True, "world deserialises")

    check(abs(clone.world_time - world.world_time) < 1e-6, "world_time preserved")
    check(len(clone.population.agents) == len(world.population.agents),
          "population size preserved",
          f"{len(clone.population.agents)} != {len(world.population.agents)}")
    check(clone.stockpile == world.stockpile, "stockpile preserved",
          f"{clone.stockpile} != {world.stockpile}")
    check(clone.structures.count() == world.structures.count(),
          "structure count preserved")

    import numpy as np
    same_terrain = np.allclose(clone.terrain.height, world.terrain.height)
    check(same_terrain, "terrain heightmap preserved exactly")

    # The whole blob, not a list of fields somebody remembered to name. Every
    # check above is a field an author thought of; this one is the only thing
    # here that notices a field NOBODY thought of, and it earns its place - a
    # defensive filter added to from_dict for the junk sweep silently dropped
    # every entry in build_queue on load (the queue holds job records, and a
    # stale ``list[str]`` annotation said otherwise), and not one of the named
    # checks above moved. A save that does not re-serialise to itself has lost
    # something, whatever it was.
    #
    # It used to say "byte-identical", which is FALSE in general and passed only
    # because this fixture is two sim-minutes old and never reaches the clamp
    # that breaks it. See CLAMPED_ON_LOAD, and test_roundtrip_at_clamp for the
    # fixture that does reach it. Two true claims replace the one false one:
    # nothing outside the declared table moves, and the save settles from the
    # second round on.
    rt = _roundtrip(world, loader)
    stray = [d for d in rt["diff1"] if d[0] not in CLAMPED_ON_LOAD]
    if not check(not stray,
                 "save -> load -> save changes nothing outside the fields "
                 "from_dict is documented to clamp",
                 f"{len(stray)} field(s) moved, "
                 f"{len(rt['s0'])} vs {len(rt['s1'])} chars"):
        _show_diff(stray)
    if not check(rt["s1"] == rt["s2"],
                 "save -> load -> save is stable from the second round",
                 f"{len(rt['diff2'])} field(s) still moving on round 2"):
        _show_diff(rt["diff2"])

    names_a = sorted(a.name for a in world.population.agents)
    names_b = sorted(a.name for a in clone.population.agents)
    check(names_a == names_b, "agent names preserved")

    cols_a = sorted(tuple(a.color) for a in world.population.agents)
    cols_b = sorted(tuple(a.color) for a in clone.population.agents)
    check(cols_a == cols_b, "agent identity colours preserved")

    # A mid-build action must survive, per the spec. GUARD FIRST: this was
    # wrapped in ``if acting:`` and so could disappear without trace, taking the
    # check count down by one where nobody would read it - and an empty ``acting``
    # is not a neutral fact, it means the fixture stopped exercising the thing.
    # Measured: the long run leaves 4 agents mid-action (Celebrate, Farm,
    # GatherWood, Mine), so requiring at least one costs nothing and the day it
    # does cost something is the day somebody needs to know.
    acting = [a for a in world.population.alive_agents() if a.action]
    check(bool(acting),
          "the round-trip fixture has in-flight actions to preserve",
          "no agent was mid-action, so the next check verifies nothing")
    kinds_a = sorted(a.action.kind for a in acting)
    kinds_b = sorted(a.action.kind for a in clone.population.alive_agents()
                     if a.action)
    check(kinds_a == kinds_b, "in-flight actions preserved",
          f"{kinds_a} != {kinds_b}")

    # and the clone must keep running
    from backgrounded.constants import SIM_DT
    try:
        for _ in range(60):
            clone.tick(SIM_DT)
        check(True, "loaded world keeps ticking")
    except Exception as exc:
        check(False, "loaded world keeps ticking", f"{type(exc).__name__}: {exc}")

    # ...and keep running *intact*. "Keeps ticking" is not enough on its own:
    # World.tick guards every subsystem and disables any that raises, so a
    # reloaded world missing an attribute set only in __init__ ticks along
    # perfectly happily with a subsystem silently switched off for the rest of
    # the session. That is exactly how a dead scene rotation shipped while all
    # 25 checks passed, so the clone is now held to the same bar as a fresh one.
    check(not clone._disabled, "no subsystem disabled itself after a reload",
          str(sorted(clone._disabled)))


def _world_at_hut_tier_clamp(world):
    """An independent copy of ``world``, run forward until the stone-hut gate
    latches - the one state in which a save does not reload to itself.

    HOW THE FIXTURE IS BUILT, and why this is not cheating. The only thing
    asserted about the colony is that it is HAPPY - every living agent's morale
    is set to 1.0 before each tick, which is exactly the condition the tier is
    gated on and a state colonies reach unaided. Everything else is the shipped
    simulation: ``World.tick`` runs unpatched at the real ``SIM_DT``, and the
    accumulator, the 180 s threshold and the latch are all ``_tick_hut_tier``'s
    own code. Nothing here writes ``hut_tier_dwell``.

    The proof that this is a faithful fixture rather than a convenient one is
    that it lands on the same value the sim reaches on its own. Seed 7 on the
    real scene rotation crosses at tick 73,337 - 40.7 sim-minutes, ~40 s of wall
    clock - and ends on ``hut_tier_dwell == 180.0333333333257``. This fixture
    crosses in 5,401 ticks (~3.5 s) and ends on 180.0333333333257, the same
    float. Seed 12345, which the long run uses, never crosses at all: measured
    to 20,000 ticks its dwell is still 0.0, because that colony's morale sits
    under HUT_TIER_MORALE. That is precisely why the round-trip check passed for
    the wrong reason for as long as it did, and why the fixture cannot simply be
    "run the long-run world for longer".
    """
    from backgrounded.constants import HUT_TIER_DWELL_SEC, SIM_DT, SIM_HZ
    from backgrounded.sim.world import World

    # Through a save, so the long-run world this test shares with everything
    # after it is not mutated by 5,401 extra ticks. Cheap, and the round-trip is
    # exact for a world this young (the check above just said so).
    fixture = World.from_dict(json.loads(_canon(world.to_dict())))
    budget = int(HUT_TIER_DWELL_SEC * SIM_HZ) + 2 * SIM_HZ
    ticks = 0
    while not fixture.hut_tier_unlocked and ticks < budget:
        for a in fixture.population.alive_agents():
            a.morale = 1.0
        fixture.tick(SIM_DT)
        ticks += 1
    return fixture, ticks


def test_roundtrip_at_clamp(world, loader=None) -> None:
    """The round-trip check, at the threshold that breaks the naive version.

    ``to_dict -> from_dict -> to_dict`` is not byte-identical in general, and
    the check that claimed it was passed only because its fixture sat below
    HUT_TIER_DWELL_SEC. This is the same claim made where it is hardest, and it
    is deliberately the STRICTEST form available: not "some tolerance", but
    *exactly one named field moved, and it moved to exactly the clamp*. A second
    field going lossy shows up here as a failure, with its path printed.
    """
    section("save / load round-trip at the clamp")
    from backgrounded.constants import HUT_TIER_DWELL_SEC
    from backgrounded.sim.world import World
    if loader is None:
        loader = World.from_dict

    fixture, ticks = _world_at_hut_tier_clamp(world)

    # GUARD. Everything below is vacuous if the fixture did not actually reach
    # the threshold - it would go green on a world with dwell 0.0, which is the
    # exact failure this whole test exists to retire. So the fixture asserts its
    # own state first, and a future change that makes the colony unhappy or
    # moves the constant fails HERE, loudly, instead of quietly downgrading the
    # three checks after it into tautologies.
    dwell = float(fixture.hut_tier_dwell)
    check(fixture.hut_tier_unlocked and dwell > HUT_TIER_DWELL_SEC,
          f"the fixture really is past HUT_TIER_DWELL_SEC "
          f"(dwell {dwell!r} after {ticks} ticks)",
          f"unlocked={fixture.hut_tier_unlocked} dwell={dwell!r} "
          f"vs HUT_TIER_DWELL_SEC {HUT_TIER_DWELL_SEC} after {ticks} ticks")

    rt = _roundtrip(fixture, loader)
    paths = [p for p, _, _ in rt["diff1"]]
    if not check(paths == ["hut_tier_dwell"],
                 "at the threshold a load changes the clamped accumulator "
                 "and nothing else",
                 f"{len(paths)} field(s) moved: {paths[:6]}"):
        _show_diff(rt["diff1"])
    before = rt["diff1"][0][1] if rt["diff1"] else None
    after = rt["diff1"][0][2] if rt["diff1"] else None
    check(before is not None and after == HUT_TIER_DWELL_SEC
          and isinstance(before, float) and before > HUT_TIER_DWELL_SEC,
          "...and it lands on exactly HUT_TIER_DWELL_SEC",
          f"{before!r} -> {after!r}, want > {HUT_TIER_DWELL_SEC} -> "
          f"{HUT_TIER_DWELL_SEC}")
    if not check(rt["s1"] == rt["s2"],
                 "a second round-trip changes nothing at all (idempotent)",
                 f"{len(rt['diff2'])} field(s) still moving on round 2"):
        _show_diff(rt["diff2"])

    # And the clamped world is still a working world, not just a matching blob.
    from backgrounded.constants import SIM_DT
    reloaded = loader(json.loads(rt["s1"]))
    # BEFORE the ticks, and that ordering is the whole check. Written the other
    # way round it passed against a loader that deliberately threw the latch
    # away: the reloaded dwell sits AT 180.0, so the very first tick of a happy
    # colony pushes it over again and re-latches the tier. The world healed
    # itself inside the measurement, and the measurement reported the loader as
    # correct. Read the flag off the load, not off the world sixty ticks later.
    check(bool(reloaded.hut_tier_unlocked),
          "the earned stone-hut tier survives the clamp",
          f"hut_tier_unlocked={reloaded.hut_tier_unlocked} straight off the load")
    try:
        for _ in range(60):
            reloaded.tick(SIM_DT)
        check(not reloaded._disabled,
              "a world reloaded at the threshold keeps ticking intact",
              str(sorted(reloaded._disabled)))
    except Exception as exc:
        check(False, "a world reloaded at the threshold keeps ticking intact",
              f"{type(exc).__name__}: {exc}")


#: How many living colonists the roster-cap probe stuffs into a save. Two
#: thousand, i.e. a hundred times MAX_POP, because the point is a number no
#: amount of simulation could produce - a clamp that merely trimmed a couple of
#: strays would not be distinguishable from one that worked.
ROSTER_CAP_PROBE = 2000


def test_roster_cap(world) -> None:
    """MAX_POP is the simulation's cap; a FILE must not be able to beat it.

    Before this, MAX_POP was enforced at ``_spawn_replacement`` and nowhere
    else, which bounded everything the sim can do and nothing a save can claim.
    Measured on the tree as shipped: a ``population.agents`` list hand-inflated
    to 2000 records loaded 2000 living colonists against a MAX_POP of 20 -
    ``reconcile`` ok=1, residual 0, no exception, no warning, no subsystem
    disabled. An impossible world, imported in silence by the method that is the
    project's trust boundary.

    Only reachable by hand-edit, which is why this is the low-priority end of
    the file - but from_dict's whole job is to survive files it did not write.

    Four claims, and the first is the one that keeps the rest honest:

      1. an honest, under-cap save loads COMPLETELY UNCHANGED. A clamp that
         trims the roster of every save is not a clamp, it is a bug, and the
         other three checks here would pass just as green while it ate people;
      2. the inflated save does not raise and does not come back over MAX_POP;
      3. it comes back AT MAX_POP, not empty. "Clamped to zero and reseeded"
         also satisfies "not over the cap", and is data loss dressed as a fix;
      4. nobody was killed to make room - ``stats["died"]`` stays 0 and the
         books still reconcile. The surplus is dropped from the roster and
         ``births`` comes down to match; see World._clamp_roster for why a death
         would be the wrong answer (it raises a grave and lands in the colony's
         permanent record for a headcount it never had).
    """
    section("roster cap on load")
    from backgrounded.constants import MAX_POP, SIM_DT
    from backgrounded.sim.world import World

    base = json.loads(json.dumps(world.to_dict()))
    recs = list(base.get("population", {}).get("agents") or ())
    check(bool(recs) and len(recs) <= MAX_POP,
          "the probe starts from an honest under-cap roster",
          f"{len(recs)} records vs MAX_POP {MAX_POP}")

    # 1. the honest save, untouched.
    honest = World.from_dict(copy.deepcopy(base))
    check(len(honest.population.alive_agents())
          == len(world.population.alive_agents())
          and int(honest.population.births) == int(world.population.births),
          "an under-cap roster loads unchanged (the clamp is a no-op)",
          f"alive {len(honest.population.alive_agents())} vs "
          f"{len(world.population.alive_agents())}, births "
          f"{honest.population.births} vs {world.population.births}")

    # 2-4. the inflated save. Real agent records, cycled, with fresh ids and
    # names so nothing is rejected as a duplicate - the roster is impossible in
    # size and in nothing else, which is the hardest version for the clamp.
    blob = copy.deepcopy(base)
    ghosts = []
    while len(ghosts) < ROSTER_CAP_PROBE:
        r = copy.deepcopy(recs[len(ghosts) % len(recs)])
        r["id"] = 10_000 + len(ghosts)
        r["name"] = f"Ghost{len(ghosts)}"
        r["alive"] = True
        ghosts.append(r)
    blob["population"]["agents"] = ghosts
    blob["population"]["births"] = len(ghosts)

    try:
        w = World.from_dict(blob)
    except Exception as exc:
        check(False, f"a save claiming {ROSTER_CAP_PROBE} colonists does not raise",
              f"{type(exc).__name__}: {exc}")
        return
    check(True, f"a save claiming {ROSTER_CAP_PROBE} colonists does not raise")

    alive = len(w.population.alive_agents())
    check(alive == MAX_POP,
          f"...and lands at exactly MAX_POP, not {ROSTER_CAP_PROBE} and not 0",
          f"alive={alive} MAX_POP={MAX_POP}")
    r = w.reconcile()
    check(r.get("ok") == 1 and r.get("residual") == 0,
          "...with the population books still balanced", str(r))
    check(int(w.stats.get("died", 0)) == 0,
          "...and nobody was killed to make room",
          f"stats[died]={w.stats.get('died')} - the surplus must be dropped "
          f"from the roster, not buried")

    try:
        for _ in range(JUNK_TICKS):
            w.tick(SIM_DT)
        check(not w._disabled, "...and the clamped world ticks intact",
              str(sorted(w._disabled)))
    except Exception as exc:
        check(False, "...and the clamped world ticks intact",
              f"{type(exc).__name__}: {exc}")

    # The dead are not the living, and the cap is on the living. A corpse
    # awaiting burial is already paid for in the books; dropping it here would
    # delete a grave the colony has already accounted for.
    blob2 = copy.deepcopy(base)
    corpses = []
    for i in range(4):
        c = copy.deepcopy(recs[i % len(recs)])
        c["id"] = 90_000 + i
        c["name"] = f"Fallen{i}"
        c["alive"] = False
        c["dead_t"] = 0.0
        corpses.append(c)
    blob2["population"]["agents"] = ghosts + corpses
    blob2["population"]["births"] = len(ghosts) + len(corpses)
    blob2.setdefault("stats", {})["died"] = 0
    w2 = World.from_dict(blob2)
    kept_dead = sum(1 for a in w2.population.agents if not a.alive)
    check(len(w2.population.alive_agents()) == MAX_POP
          and kept_dead == len(corpses),
          "the clamp takes the surplus off the LIVING roster and keeps the dead",
          f"alive={len(w2.population.alive_agents())} (want {MAX_POP}) "
          f"dead={kept_dead} (want {len(corpses)})")
    r2 = w2.reconcile()
    check(r2.get("ok") == 1 and r2.get("residual") == 0,
          "...and those books balance too", str(r2))


#: Every shape a hand-edited or half-written save file can put where a value
#: belongs.
#:
#: ``World.from_dict`` is the one method in the codebase that MUST NOT RAISE:
#: persist.load_world reads any exception as corruption and QUARANTINES the
#: save, so a junk value does not cost the field it sits in, it costs the whole
#: colony. The only way to have any confidence in that is to try everything
#: everywhere, which is what this list is for.
#:
#: The two entries that look like padding are the two that matter most. NaN and
#: both infinities are reachable from a REAL file, not hypothetical - json.loads
#: accepts ``NaN``, ``Infinity`` and ``-Infinity`` by default - and they are the
#: values that get past a type check and then raise in the conversion: NaN
#: raises ValueError out of int(), infinity raises OverflowError. ``negative``
#: is the other one: -12345 is a perfectly good int that passes every type test
#: there is, and ``np.random.default_rng`` rejects it outright.
JUNK_VALUES: list[tuple[str, object]] = [
    ("str", "banana"),
    ("empty-str", ""),
    ("none", None),
    ("list", [1]),
    ("list2", [1, 2]),
    ("empty-list", []),
    ("dict", {"a": 1}),
    ("empty-dict", {}),
    ("int", 42),
    ("float", 1.5),
    ("bool", True),
    ("nan", float("nan")),
    ("inf", float("inf")),
    ("-inf", float("-inf")),
    ("huge", 1e308),
    ("negative", -12345),
    ("nested-junk", {"x": [None, {"y": float("nan")}], "z": "q"}),
]

#: Sim ticks each junk save is run for after loading. A subsystem that a bad
#: save has broken disables itself on the tick that first touches it, so this
#: does not need to be long - but it does need to be more than zero, because
#: "from_dict returned an object" is not the same claim as "the world it
#: returned still works".
JUNK_TICKS = 3


def _history_save(world) -> dict:
    """A save whose books carry a HISTORY, and the reason the sweep needs one.

    The long-run world above is two sim-minutes old: nobody has died, nobody has
    been abducted, and ``births == alive == 4``. Sweeping junk over THAT save
    can never catch an accounting bug, because every way of losing a section
    lands on the same trivial answer - a fresh Population seeds four agents,
    stats has nothing to contradict them, and the residual is zero whether the
    code is right or wrong. That is precisely the shape of green tick this file
    has been burned by twice, so it is worth being explicit about.

    So the sweep also runs against a save that HAS lived: four survivors of
    eleven births, five dead and two still up in the ufo. Nothing is fabricated
    about the format - ``births`` is a persisted field of the population section
    and the three tallies are persisted fields of ``stats`` - and the books are
    deliberately made to BALANCE (11 + 0 - 5 - 2 - 4 == 0), so the honest load
    is a clean control and only the injected junk can break them.

    Against this base the old code fails loudly: an unreadable ``population``
    section leaves stats remembering five deaths the new roster never had
    (residual -7), and an unreadable ``stats`` section leaves eleven births with
    nothing to explain the seven missing people (residual +7).
    """
    d = json.loads(json.dumps(world.to_dict()))
    alive = len(world.population.alive_agents())
    died, abducted = 5, 2
    if isinstance(d.get("population"), dict):
        d["population"]["births"] = alive + died + abducted
    stats = d.get("stats")
    if isinstance(stats, dict):
        stats["died"] = died
        stats["abducted"] = abducted
        stats["returned"] = 0
    return d


def _junk_cases(base: dict, history: dict):
    """Every ``(label, blob)`` the defensive-load sweep runs.

    Three layers, and the second and third are there because the version of
    this test that shipped had only a weak form of the first: it tried three
    hand-written blobs that between them mutated four keys, never gave ``seed``
    or ``world_time`` a non-numeric value, and never touched ``stockpile``,
    ``build_queue``, ``chronicle`` or ``tick_count`` at all. Eight scalar fields
    were raising straight out of from_dict the whole time, behind 49 green
    checks.

      1. EVERY top-level key against the whole battery, plus deleting it, on a
         real save - the layer that catches a field nobody remembered to coerce.
      2. The same over a save that carries a history, so the population books
         are actually load-bearing. See :func:`_history_save`.
      3. Junk INSIDE the containers rather than replacing them. This is the
         layer a top-level sweep structurally cannot reach: ``{"wood":
         "banana"}`` is an ordinary-looking dict that passes every check made of
         the container itself, and its int() raised eleven different ways.
    """
    from backgrounded.constants import ALL_RESOURCES

    for tag, src in (("", base), ("history:", history)):
        for key in sorted(src.keys()):
            for jname, jval in JUNK_VALUES:
                blob = copy.deepcopy(src)
                blob[key] = jval
                yield (f"{tag}{key}={jname}", blob)
            blob = copy.deepcopy(src)
            blob.pop(key, None)
            yield (f"{tag}{key}=DELETED", blob)

    res = sorted(ALL_RESOURCES)[0]
    for jname, jval in JUNK_VALUES:
        def _case(label, mutate):
            blob = copy.deepcopy(base)
            try:
                mutate(blob)
            except Exception:
                return None
            return (label, blob)

        for label, mutate in (
            (f"stockpile[{res}]={jname}",
             lambda b: b["stockpile"].__setitem__(res, jval)),
            (f"stockpile[unknown]={jname}",
             lambda b: b["stockpile"].__setitem__("bogus", jval)),
            (f"build_queue[0]={jname}",
             lambda b: b.__setitem__("build_queue", [jval] + list(b["build_queue"]))),
            (f"chronicle[0]={jname}",
             lambda b: b.__setitem__("chronicle", [jval] + list(b["chronicle"]))),
            (f"stats[died]={jname}",
             lambda b: b["stats"].__setitem__("died", jval)),
            (f"stats[unknown]={jname}",
             lambda b: b["stats"].__setitem__("bogus", jval)),
            (f"terrain[w]={jname}", lambda b: b["terrain"].__setitem__("w", jval)),
            (f"terrain[height]={jname}",
             lambda b: b["terrain"].__setitem__("height", jval)),
            (f"terrain.height[shape]={jname}",
             lambda b: b["terrain"]["height"].__setitem__("shape", jval)),
            (f"terrain.height[dtype]={jname}",
             lambda b: b["terrain"]["height"].__setitem__("dtype", jval)),
            (f"terrain.height[b64]={jname}",
             lambda b: b["terrain"]["height"].__setitem__("b64", jval)),
            (f"ufo[abducted]={jname}",
             lambda b: b["ufo"].__setitem__("abducted", jval)),
            # The population section, field by field, and it was the hole in
            # this layer. Every other section here had its internals swept while
            # population - the one section reconcile reads its books off, and the
            # only one with a per-record list inside it - was only ever replaced
            # wholesale by the top-level sweep. "population is a string" and
            # "population.births is a string" are different loads and only the
            # first was tried; likewise a roster whose FIRST RECORD is junk,
            # which is a shape a truncated write produces on its own.
            (f"population[births]={jname}",
             lambda b: b["population"].__setitem__("births", jval)),
            (f"population[next_id]={jname}",
             lambda b: b["population"].__setitem__("next_id", jval)),
            (f"population[peak]={jname}",
             lambda b: b["population"].__setitem__("peak", jval)),
            (f"population[graves_raised]={jname}",
             lambda b: b["population"].__setitem__("graves_raised", jval)),
            (f"population[used_names]={jname}",
             lambda b: b["population"].__setitem__("used_names", jval)),
            (f"population[agents]={jname}",
             lambda b: b["population"].__setitem__("agents", jval)),
            (f"population.agents[0]={jname}",
             lambda b: b["population"]["agents"].__setitem__(0, jval)),
            (f"population.agents[0][id]={jname}",
             lambda b: b["population"]["agents"][0].__setitem__("id", jval)),
            (f"population.agents[0][alive]={jname}",
             lambda b: b["population"]["agents"][0].__setitem__("alive", jval)),
            (f"population.agents[0][x]={jname}",
             lambda b: b["population"]["agents"][0].__setitem__("x", jval)),
            (f"population.agents[0][generation]={jname}",
             lambda b: b["population"]["agents"][0].__setitem__("generation", jval)),
        ):
            got = _case(label, mutate)
            if got is not None:
                yield got

    # The three the old test had, kept verbatim: a save can also be absent,
    # truncated to nothing, or written by a build that knew fewer sections.
    yield ("empty dict", {})
    yield ("junk sections",
           {"terrain": "nonsense", "population": 42, "props": None})
    yield ("partial", {"world_time": 10.0, "seed": 7})


def test_defensive_load(world) -> None:
    """No save file, however mangled, may raise, silently disable a subsystem,
    or leave the population books unbalanced."""
    section("defensive load")
    from backgrounded.constants import SIM_DT
    from backgrounded.sim.world import World

    base = json.loads(json.dumps(world.to_dict()))
    history = _history_save(world)

    # The sweep is only worth running if its history base really does carry
    # one; this is the check that stops it quietly rotting back into the
    # trivial case the moment somebody shortens the long run.
    hb = history.get("population", {}).get("births")
    hs = history.get("stats", {})
    check(bool(hb) and hb > len(world.population.alive_agents())
          and hs.get("died", 0) > 0 and hs.get("abducted", 0) > 0,
          "the junk sweep has a base save that carries a population history",
          f"births={hb} stats={{died: {hs.get('died')}, "
          f"abducted: {hs.get('abducted')}}}")

    raised: list[str] = []
    disabled: list[str] = []
    unbalanced: list[str] = []
    unreadable: list[str] = []
    total = 0

    for label, blob in _junk_cases(base, history):
        total += 1
        try:
            w = World.from_dict(blob)
        except Exception as exc:
            raised.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        try:
            for _ in range(JUNK_TICKS):
                w.tick(SIM_DT)
        except Exception as exc:
            raised.append(f"{label} (tick): {type(exc).__name__}: {exc}")
            continue
        if w._disabled:
            disabled.append(f"{label}: {sorted(w._disabled)}")
        r = w.reconcile()
        if not r.get("ok"):
            unreadable.append(label)
        elif r.get("residual") != 0:
            unbalanced.append(f"{label}: residual {r.get('residual')}")

    check(not raised,
          f"World.from_dict never raises, over {total} malformed saves",
          f"{len(raised)} raised, e.g. " + " | ".join(raised[:4]))
    check(not disabled,
          "no subsystem disables itself after loading a malformed save",
          f"{len(disabled)} did, e.g. " + " | ".join(disabled[:4]))
    check(not unreadable,
          "reconcile can still read the books after a malformed save",
          f"{len(unreadable)} unreadable, e.g. " + ", ".join(unreadable[:4]))
    check(not unbalanced,
          "reconcile residual is 0 after every malformed save",
          f"{len(unbalanced)} unbalanced, e.g. " + " | ".join(unbalanced[:4]))
    print(f"\n  swept {total} malformed saves "
          f"({len(JUNK_VALUES)} junk values x every top-level key, twice, "
          f"plus junk inside the containers)")


def test_terrain_width_claims(world) -> None:
    """A save's declared width is a CLAIM; the payload is the evidence.

    ``Terrain.from_dict`` decides between "these are the saved columns, keep
    them and generate the new land beside them" and "resample these to the
    current width", and getting that decision wrong the second way is the
    failure that killed a colony: a 1600 px heightmap stretched 4x loads
    perfectly cleanly, so persist.py never quarantines it, and then the huts are
    174 px in the air and the population falls to its death inside three
    sim-minutes.

    ``w`` was already understood to be an unreliable claim. ``shape`` was not,
    and was trusted ahead of the payload - so a save declaring 6400 columns
    while carrying 1600 sailed past the migration and into the resampler, 152 to
    429 px of stretch depending on the seed. Both fields are checked here, in
    every combination of honest and lying, because there is no reason to think
    the next one will be found any faster than this one was.
    """
    section("terrain width claims")
    import numpy as np
    from backgrounded.constants import WORLD_W
    from backgrounded.sim.terrain import Terrain, _b64_array

    n = 1600
    band_h = np.asarray(world.terrain.height, dtype=np.float32)[:n].copy()
    band_m = np.asarray(world.terrain.material, dtype=np.uint8)[:n].copy()

    for label, w_field, shape_field in (
        ("both honest", n, [n]),
        ("w lies (says 6400)", WORLD_W, [n]),
        ("shape lies (says 6400)", n, [WORLD_W]),
        ("w and shape both lie", WORLD_W, [WORLD_W]),
        ("no shape field at all", n, None),
    ):
        d = {
            "seed": int(world.terrain.seed),
            "style": str(world.terrain.style),
            "w": w_field,
            "height": _b64_array(band_h),
            "material": _b64_array(band_m),
        }
        for arr in (d["height"], d["material"]):
            if shape_field is None:
                arr.pop("shape", None)
            else:
                arr["shape"] = list(shape_field)

        t = Terrain.from_dict(d)
        live = np.asarray(t.height, dtype=np.float64)
        wide = live.size == WORLD_W
        delta = float(np.abs(live[:n] - band_h).max()) if live.size >= n else -1.0
        check(wide and delta == 0.0,
              f"a 1600px payload restores byte-exact when {label}",
              f"cols={live.size} max|restored-saved|={delta:.2f}px")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=3600)
    ap.add_argument("--scene", default="night_storm")
    args = ap.parse_args()

    if not test_imports():
        print("\nimports failed; stopping here")
        return 1
    test_sim_is_headless()
    test_render_imports()          # after the headless check - it loads pygame
    test_disable_guard()           # before anything that asserts _disabled is empty
    test_world_width()
    world = test_long_run(args.ticks, args.scene)
    test_roundtrip(world)
    test_roundtrip_at_clamp(world)
    test_roster_cap(world)
    test_defensive_load(world)
    test_terrain_width_claims(world)

    print(f"\n{'=' * 58}")
    if FAILURES:
        print(f"FAILED  {len(FAILURES)}/{CHECKS} checks")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"OK      all {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
