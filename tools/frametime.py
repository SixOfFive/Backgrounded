"""Render frame timing per scene, against the 60 fps preview budget.

  python tools/frametime.py
  python tools/frametime.py --scenes volcano --frames 1200 --runs 5

WHY THIS EXISTS. "Volcano is over the frame budget" was carried as a defect for
a long time on the strength of one measurement taken while ~20 concurrent
python processes from other sweep lanes were on the box. Re-measured on a quiet
machine it was not true - volcano's p95 came in at 10.5 ms against a 16.7 ms
budget, with 0.0% of frames over. The mean had barely moved; the tail, which is
the part that actually breaks a budget, had been almost entirely other people's
CPU. A frame-timing claim is only worth as much as the machine it was taken on,
so this reports the run-to-run spread alongside the numbers: if that spread is
not small, nothing else on the line means anything.

p95 rather than the mean, because a 60 fps budget is broken by the tail. An
8 ms mean is comfortable and tells you nothing about the one frame in twenty
that takes 20 ms and shows up as a visible hitch.

This is a PREVIEW-WINDOW concern only. The wallpaper path runs at 4 Hz, a
250 ms budget, which none of this comes close to.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows only - see the same guard in tools/probe.py. Frame timings need a
# real driver rather than "dummy", and off Windows SDL picks that itself.
if sys.platform == "win32":
    os.environ.setdefault("SDL_VIDEODRIVER", "windib")

import pygame  # noqa: E402

from backgrounded.constants import SIM_DT  # noqa: E402
from backgrounded.render.renderer import Renderer  # noqa: E402
from backgrounded.sim.world import World  # noqa: E402

FRAME_DT = 1.0 / 60.0
BUDGET_MS = 1000.0 / 60.0


def one_run(scene: str, frames: int, warm_s: float, seed: int) -> list[float]:
    world = World(seed=seed, scene=scene)
    world.events.request_scene(scene)
    for _ in range(int(warm_s / SIM_DT)):
        world.tick(SIM_DT)
    r = Renderer()

    # Twenty frames before the clock starts. The first draw builds every lazy
    # cache there is - terrain strips, gradients, fonts - and timing it would
    # put a one-off in the tail, which is precisely the statistic this tool is
    # for. Steady state is the question.
    for _ in range(20):
        world.tick(SIM_DT)
        r.draw(world, FRAME_DT)

    out = []
    for _ in range(frames):
        world.tick(SIM_DT)
        t0 = time.perf_counter()
        r.draw(world, FRAME_DT)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
    return s[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="clear,blizzard,volcano,sandstorm")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--runs", type=int, default=3,
                    help="best-of-N; the spread between runs is the noise floor")
    ap.add_argument("--warm", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    pygame.init()
    pygame.display.set_mode((64, 64), pygame.HIDDEN)

    print(f"budget {BUDGET_MS:.1f} ms/frame at 60 fps   "
          f"{args.frames} frames x {args.runs} runs, seed {args.seed}\n")
    print(f"{'scene':<11} {'mean':>7} {'p50':>7} {'p95':>7} {'p99':>7} "
          f"{'max':>7}   {'over':>6}   spread")
    print("-" * 74)

    over_budget = []
    for scene in args.scenes.split(","):
        runs = [one_run(scene, args.frames, args.warm, args.seed)
                for _ in range(args.runs)]
        best = min(runs, key=lambda xs: statistics.mean(xs))
        means = [statistics.mean(r) for r in runs]
        spread = max(means) - min(means)
        over = 100.0 * sum(1 for v in best if v > BUDGET_MS) / len(best)
        if pct(best, 95) > BUDGET_MS:
            over_budget.append(scene)
        print(f"{scene:<11} {statistics.mean(best):7.2f} {pct(best, 50):7.2f} "
              f"{pct(best, 95):7.2f} {pct(best, 99):7.2f} {max(best):7.2f}   "
              f"{over:5.1f}%   {spread:.2f} ms")

    print()
    if over_budget:
        print(f"OVER BUDGET at p95: {', '.join(over_budget)}")
        print("Check the spread column first - a noisy box fakes this.")
        return 1
    print("every scene is inside the 16.7 ms budget at p95")
    return 0


if __name__ == "__main__":
    sys.exit(main())
