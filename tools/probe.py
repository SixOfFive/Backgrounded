"""Render probe - deterministic visual contact sheets.

The interesting moments in this program are rare and short (a lightning flash
lasts ~0.45s and fires every 4-12s), so catching them by sampling a live run is
mostly luck. This forces them instead: it warms a world up, then renders named
moments on demand and writes a labelled contact sheet.

  python tools/probe.py --mode storm     # dark / flash / afterglow
  python tools/probe.py --mode scenes    # every scene side by side
  python tools/probe.py --mode ambient   # ambient sweep for tuning
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SDL_VIDEODRIVER", "windib")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from backgrounded.constants import (  # noqa: E402
    RENDER_H, RENDER_SIZE, RENDER_W, SCENES, SIM_DT,
)
from backgrounded.render.renderer import Renderer  # noqa: E402
from backgrounded.sim.world import World  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "shots"


def stats(surf: pygame.Surface) -> str:
    a = pygame.surfarray.array3d(surf).astype(np.float32)
    lum = a.mean(axis=2)
    return (f"mean={lum.mean():5.1f} p99={np.percentile(lum,99):5.1f} "
            f"max={lum.max():5.0f} lit={100*(lum>40).mean():4.1f}%")


def warm(world: World, seconds: float, renderer: Renderer) -> None:
    for _ in range(int(seconds / SIM_DT)):
        world.tick(SIM_DT)


def shot(r: Renderer, world: World) -> pygame.Surface:
    """One frame, with the HUD baked in the way the wallpaper sees it.

    Renderer.draw deliberately stopped blitting the HUD when the preview gained
    a screen-anchored one: the window overlays it at window coordinates and the
    wallpaper path bakes it at world coordinates. A contact sheet is a picture of
    the *wallpaper*, so it has to bake it too - otherwise these shots silently
    stop showing the stats panel and the chronicle log, which is exactly what
    happened until this helper existed.
    """
    f = r.draw(world, SIM_DT)
    r.draw_hud(f, world)
    return f


def label(surf: pygame.Surface, text: str) -> pygame.Surface:
    out = surf.copy()
    font = pygame.font.SysFont("consolas", 17)
    t = font.render(text, True, (255, 255, 120))
    bg = pygame.Surface((t.get_width() + 10, t.get_height() + 6))
    bg.set_alpha(190)
    bg.fill((0, 0, 0))
    out.blit(bg, (6, 6))
    out.blit(t, (11, 9))
    return out


def sheet(frames: list[tuple[str, pygame.Surface]], path: Path,
          cols: int = 2, scale: float = 0.5) -> None:
    if not frames:
        return
    w = int(RENDER_W * scale)
    h = int(RENDER_H * scale)
    rows = (len(frames) + cols - 1) // cols
    sheet = pygame.Surface((w * cols, h * rows))
    sheet.fill((25, 25, 30))
    for i, (name, s) in enumerate(frames):
        small = pygame.transform.smoothscale(label(s, name), (w, h))
        sheet.blit(small, ((i % cols) * w, (i // cols) * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(sheet, str(path))
    print(f"\nwrote {path}  ({path.stat().st_size // 1024} KiB)")


def mode_storm(args) -> None:
    world = World(seed=4242, scene="night_storm")
    r = Renderer()
    warm(world, args.warm, r)

    frames = []
    f = shot(r, world)
    print("ambient dark ", stats(f))
    frames.append(("1. night storm - ambient only", f.copy()))

    # Force the reveal rather than waiting for it. A flash contributes nothing
    # until Lighting.tick integrates it (the attack ramp starts at zero), so
    # advance one tick before drawing or the "peak" frame is just the dark one.
    world.lighting.add_flash(1.0, 0.45, (200, 220, 255))
    ev = world.events
    if hasattr(ev, "strikes"):
        gy = float(world.terrain.ground_y(RENDER_W * 0.62))
        ev.strikes.append({"x": RENDER_W * 0.62, "t": 0.0, "seed": 7,
                           "ground_y": gy, "direct": False})
    world.lighting.tick(SIM_DT)
    f = shot(r, world)
    print("flash peak   ", stats(f))
    frames.append(("2. lightning strike - the reveal", f.copy()))

    for _ in range(6):
        world.tick(SIM_DT)
    f = shot(r, world)
    print("afterglow    ", stats(f))
    frames.append(("3. 0.2s later - fading", f.copy()))

    for _ in range(20):
        world.tick(SIM_DT)
    f = shot(r, world)
    print("back to dark ", stats(f))
    frames.append(("4. dark again - candle only", f.copy()))

    sheet(frames, OUT / "storm.png", cols=2, scale=0.5)


def mode_scenes(args) -> None:
    frames = []
    for sc in SCENES:
        world = World(seed=99, scene=sc)
        r = Renderer()
        world.events.request_scene(sc)
        warm(world, args.warm, r)
        f = shot(r, world)
        print(f"{sc:14} {stats(f)}")
        frames.append((sc, f.copy()))
    sheet(frames, OUT / "scenes.png", cols=2, scale=0.42)


def mode_ambient(args) -> None:
    """Sweep the night-storm ambient floor so the value is chosen from
    evidence rather than taste."""
    frames = []
    for amb in (0.06, 0.12, 0.16, 0.22):
        world = World(seed=4242, scene="night_storm")
        r = Renderer()
        warm(world, args.warm, r)
        world.lighting.night_floor = amb          # honoured if present
        world.lighting._probe_ambient = amb
        f = shot(r, world)
        print(f"ambient {amb:.2f}  {stats(f)}")
        frames.append((f"ambient {amb:.2f}", f.copy()))
    sheet(frames, OUT / "ambient.png", cols=2, scale=0.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="storm",
                    choices=["storm", "scenes", "ambient"])
    ap.add_argument("--warm", type=float, default=45.0,
                    help="sim seconds to run before capturing")
    args = ap.parse_args()

    pygame.init()
    pygame.display.set_mode((64, 64), pygame.HIDDEN)
    pygame.font.init()

    {"storm": mode_storm, "scenes": mode_scenes, "ambient": mode_ambient}[args.mode](args)
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
