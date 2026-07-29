"""Headless integration smoke test.

Runs the simulation with no display at all, which is the point of the rule that
sim/ must not import pygame. Checks the invariants that matter for something
meant to run unattended for hours:

  * every module imports
  * sim/ is genuinely headless
  * a long run produces no exceptions and no NaNs
  * agents stay inside the world and on the ground
  * a save round-trips without losing state
  * something actually *happens* (the colony is not inert)

Usage:  python tools/smoke.py [--ticks 3600] [--scene night_storm]
"""
from __future__ import annotations

import argparse
import json
import math
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


# --------------------------------------------------------------- imports --
def test_imports() -> bool:
    section("imports")
    mods = [
        "backgrounded.constants", "backgrounded.paths", "backgrounded.config",
        "backgrounded.sim.terrain", "backgrounded.sim.props",
        "backgrounded.sim.entities", "backgrounded.sim.names",
        "backgrounded.sim.lighting", "backgrounded.sim.events",
        "backgrounded.sim.structures", "backgrounded.sim.actions",
        "backgrounded.sim.behavior", "backgrounded.sim.world",
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


def test_sim_is_headless() -> None:
    section("sim/ is headless")
    # pygame must not have been pulled in by any sim import above.
    check("pygame" not in sys.modules,
          "no sim module imported pygame",
          f"loaded by: {[m for m in sys.modules if 'pygame' in m][:3]}")

    root = Path(__file__).resolve().parent.parent / "backgrounded" / "sim"
    offenders = [p.name for p in root.glob("*.py")
                 if "import pygame" in p.read_text("utf-8", errors="ignore")]
    check(not offenders, "no 'import pygame' in sim/", str(offenders))


# ------------------------------------------------------------------- run --
def test_long_run(ticks: int, scene: str):
    section(f"simulate {ticks} ticks ({ticks / 30:.0f}s of world time)")
    from backgrounded.constants import RENDER_H, RENDER_W, SIM_DT
    from backgrounded.sim.world import World

    world = World(seed=12345, scene=scene)
    check(True, f"world generated (seed={world.seed}, "
                f"style terrain {world.terrain.height.min():.0f}..{world.terrain.height.max():.0f})")

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

    # invariants
    bad_pos, bad_num = [], []
    for a in world.population.agents:
        if not (0 <= a.x <= RENDER_W and -50 <= a.y <= RENDER_H + 50):
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


# ----------------------------------------------------------- persistence --
def test_roundtrip(world) -> None:
    section("save / load round-trip")
    from backgrounded.sim.world import World

    try:
        blob = json.dumps(world.to_dict())
    except Exception as exc:
        check(False, "world serialises to JSON", f"{type(exc).__name__}: {exc}")
        return
    check(True, f"world serialises to JSON ({len(blob)/1024:.0f} KiB)")

    try:
        clone = World.from_dict(json.loads(blob))
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

    names_a = sorted(a.name for a in world.population.agents)
    names_b = sorted(a.name for a in clone.population.agents)
    check(names_a == names_b, "agent names preserved")

    cols_a = sorted(tuple(a.color) for a in world.population.agents)
    cols_b = sorted(tuple(a.color) for a in clone.population.agents)
    check(cols_a == cols_b, "agent identity colours preserved")

    # a mid-build action must survive, per the spec
    acting = [a for a in world.population.alive_agents() if a.action]
    if acting:
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


def test_defensive_load() -> None:
    section("defensive load")
    from backgrounded.sim.world import World
    for label, blob in [
        ("empty dict", {}),
        ("junk sections", {"terrain": "nonsense", "population": 42, "props": None}),
        ("partial", {"world_time": 10.0, "seed": 7}),
    ]:
        try:
            w = World.from_dict(blob)
            w.tick(1 / 30)
            check(True, f"survives a {label} save")
        except Exception as exc:
            check(False, f"survives a {label} save", f"{type(exc).__name__}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=3600)
    ap.add_argument("--scene", default="night_storm")
    args = ap.parse_args()

    if not test_imports():
        print("\nimports failed; stopping here")
        return 1
    test_sim_is_headless()
    world = test_long_run(args.ticks, args.scene)
    test_roundtrip(world)
    test_defensive_load()

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
