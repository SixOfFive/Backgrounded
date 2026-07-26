"""App - owns the main loop and the two worker threads.

Thread ownership (see docs/ARCHITECTURE.md section 1):
  main       pygame render loop, fixed-timestep sim, command dispatch
  tray       Win32 message pump, pushes commands onto cmd_q
  wallpaper  blocking disk + SystemParametersInfoW writes
"""
from __future__ import annotations

import argparse
import ctypes
import logging
import queue
import sys
import time

import pygame

from . import paths, persist
from .config import Config
from .constants import (
    AUTOSAVE_SEC, RENDER_H, RENDER_SIZE, RENDER_W, SCENES, SIM_DT, TARGET_FPS,
)
from .render.renderer import Renderer
from .shell.preview import Preview
from .shell.tray import Tray
from .shell.wallpaper import WallpaperWriter
from .sim.world import World

log = logging.getLogger(__name__)


def _acquire_single_instance() -> bool:
    """Take a named mutex so only one copy ever runs.

    Two instances both write the wallpaper A/B pair and both call
    SystemParametersInfoW, so they overwrite each other's files mid-write and
    the desktop flickers between two different worlds. Worse, closing the
    preview window only *hides* it (the app lives in the tray), so it is easy
    to leave orphans running without noticing. The handle is deliberately
    leaked: it lives as long as the process and Windows frees it on exit.
    """
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.restype = ctypes.c_void_p
        handle = k32.CreateMutexW(None, False, "Local\\BackgroundedSingleInstance")
        if not handle:
            return True                      # cannot tell; allow startup
        ERROR_ALREADY_EXISTS = 183
        if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
            return False
        globals()["_INSTANCE_MUTEX"] = handle
        return True
    except Exception:
        return True


def _screen_size() -> tuple[int, int]:
    try:
        u = ctypes.windll.user32
        u.SetProcessDPIAware()
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        return (1920, 1080)


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        paths.ensure_dirs()
        self.args = args
        self.cfg = Config.load()

        # Command-line flags are *session* overrides, not preferences. Without
        # this split a single `--hide --no-wallpaper` test run would write both
        # into config.json and permanently disable the window and the wallpaper
        # for every later run. _cli_keys records what the CLI touched so
        # _save_config can put the stored value back; _user_keys records what
        # the user changed via the tray during the session, which does win.
        self._pristine = {k: getattr(self.cfg, k) for k in vars(self.cfg)}
        self._cli_keys: set[str] = set()
        self._user_keys: set[str] = set()

        if args.scene:
            self.cfg.scene = args.scene
            self._cli_keys.add("scene")
        if args.hide:
            self.cfg.show_window = False
            self._cli_keys.add("show_window")
        if args.no_wallpaper:
            self.cfg.wallpaper_enabled = False
            self._cli_keys.add("wallpaper_enabled")

        self.running = True
        self.cmd_q: queue.Queue = queue.Queue()

        # --- world -------------------------------------------------------
        self.world: World | None = None
        if not args.fresh:
            self.world = persist.load_world()
        if self.world is None:
            self.world = World(scene=self.cfg.scene)
            log.info("generated a new world (seed=%d)", self.world.seed)
        else:
            self.world.events.request_scene(self.cfg.scene)

        # --- pygame ------------------------------------------------------
        pygame.init()
        pygame.display.set_caption("Backgrounded - Live View")
        self.preview = Preview(RENDER_SIZE, scale=self.cfg.window_scale)
        self.preview.ensure_window(self.cfg.show_window)
        self.renderer = Renderer()
        self.renderer.show_stats = getattr(self.cfg, "show_stats", True)

        # --- threads -----------------------------------------------------
        self.wallpaper = WallpaperWriter(_screen_size())
        self.wallpaper.capture_original()
        if self.cfg.wallpaper_enabled:
            self.wallpaper.start()

        self.tray = Tray(self.cmd_q, self._tray_state)
        self.tray.start()

        self.clock = pygame.time.Clock()
        self._sim_accum = 0.0
        self._last_save = time.monotonic()
        self._last_wall = 0.0
        self._captures = 0

    # ---------------------------------------------------------- tray glue --
    def _tray_state(self) -> dict:
        return {
            "show_window": self.cfg.show_window,
            "paused": self.cfg.paused,
            "scene": self.world.events.scene if self.world else self.cfg.scene,
            "sim_speed": self.cfg.sim_speed,
            "wallpaper_enabled": self.cfg.wallpaper_enabled,
            "show_stats": getattr(self.cfg, "show_stats", True),
        }

    def _drain_commands(self) -> None:
        while True:
            try:
                kind, payload = self.cmd_q.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle(kind, payload)
            except Exception:
                log.exception("command %r failed", kind)

    def _save_config(self, *changed: str) -> None:
        """Persist preferences, without letting this session's CLI flags leak
        into the stored config."""
        self._user_keys.update(changed)
        revert = {k: getattr(self.cfg, k)
                  for k in self._cli_keys - self._user_keys}
        for k in revert:
            setattr(self.cfg, k, self._pristine[k])
        try:
            self.cfg.save()
        finally:
            for k, v in revert.items():       # restore the live session value
                setattr(self.cfg, k, v)

    def _handle(self, kind: str, payload) -> None:
        cfg = self.cfg
        if kind == "quit":
            self.running = False
        elif kind == "toggle_window":
            cfg.show_window = not cfg.show_window
            self.preview.ensure_window(cfg.show_window)
            self._save_config("show_window")
        elif kind == "toggle_wallpaper":
            cfg.wallpaper_enabled = not cfg.wallpaper_enabled
            if cfg.wallpaper_enabled:
                self.wallpaper.start()
            else:
                self.wallpaper.stop()
                self.wallpaper.restore()
            self._save_config("wallpaper_enabled")
        elif kind == "toggle_pause":
            cfg.paused = not cfg.paused
            self._save_config("paused")
        elif kind == "speed":
            cfg.sim_speed = float(payload)
            self._save_config("sim_speed")
        elif kind == "scene" and payload in SCENES:
            cfg.scene = payload
            self.world.events.request_scene(payload)
            self._save_config("scene")
        elif kind == "toggle_stats":
            self.renderer.show_stats = not self.renderer.show_stats
            cfg.show_stats = self.renderer.show_stats
            self._save_config("show_stats")
        elif kind == "save":
            persist.save_world(self.world)
        elif kind == "reset":
            persist.save_world(self.world)
            self.world = World(scene=cfg.scene)
            log.info("world reset (seed=%d)", self.world.seed)

    # --------------------------------------------------------------- loop --
    def run(self) -> int:
        log.info("running: pop=%d scene=%s",
                 len(self.world.population.alive_agents()), self.world.events.scene)
        try:
            while self.running:
                self._frame()
        except KeyboardInterrupt:
            log.info("interrupted")
        except Exception:
            log.exception("fatal error in main loop")
        finally:
            self._shutdown()
        return 0

    def _frame(self) -> None:
        dt = self.clock.tick(TARGET_FPS) / 1000.0
        dt = min(dt, 0.1)                     # never let a stall bomb the sim

        self._drain_commands()
        if self.preview.handle_close():
            self.cfg.show_window = False
            self.preview.ensure_window(False)
            self._save_config("show_window")

        # fixed-timestep sim, scaled by the speed setting
        if not self.cfg.paused:
            self._sim_accum += dt * self.cfg.sim_speed
            steps = 0
            while self._sim_accum >= SIM_DT and steps < 8:
                self.world.tick(SIM_DT)
                self._sim_accum -= SIM_DT
                steps += 1

        frame = self.renderer.draw(self.world, dt)

        if self.cfg.show_window:
            self.preview.present(frame)
            self.preview.set_caption(self._hud())

        now = time.monotonic()
        if self.cfg.wallpaper_enabled and (now - self._last_wall) >= 1.0 / max(1, self.cfg.wallpaper_fps):
            self._last_wall = now
            try:
                raw = pygame.image.tobytes(frame, "RGB")
                self.wallpaper.submit(raw, RENDER_SIZE)
            except Exception:
                log.exception("wallpaper submit failed")

        if self.args.capture and self._captures < self.args.capture:
            if self.world.tick_count % self.args.capture_every == 0:
                self._save_capture(frame)

        if now - self._last_save >= AUTOSAVE_SEC:
            self._last_save = now
            persist.save_world(self.world)

        if self.args.exit_after and self.world.world_time >= self.args.exit_after:
            log.info("exit-after reached")
            self.running = False

    def _hud(self) -> str:
        w = self.world
        pop = len(w.population.alive_agents())
        z = getattr(self.preview, "zoom", 1.0)
        zoom = f" - {z:.1f}x" if z > 1.01 else ""
        return (f"Backgrounded - {w.events.scene} - pop {pop} "
                f"gen {w.population.generation} - "
                f"wood {w.stockpile.get('wood',0)} stone {w.stockpile.get('stone',0)} "
                f"food {w.stockpile.get('food',0)}{zoom} - "
                f"{self.clock.get_fps():.0f} fps  [wheel=zoom drag=pan 0=reset F11=full]")

    def _save_capture(self, frame: pygame.Surface) -> None:
        self._captures += 1
        p = paths.CAPTURE_DIR / f"frame_{self._captures:03d}_{self.world.events.scene}.png"
        try:
            pygame.image.save(frame, str(p))
            log.info("capture -> %s", p)
        except Exception:
            log.exception("capture failed")

    def _shutdown(self) -> None:
        log.info("shutting down")
        try:
            persist.save_world(self.world)
        except Exception:
            log.exception("final save failed")
        try:
            self.wallpaper.stop()
            if self.cfg.restore_wallpaper_on_exit:
                self.wallpaper.restore()
        except Exception:
            log.exception("wallpaper shutdown failed")
        try:
            self.tray.stop()
        except Exception:
            log.exception("tray shutdown failed")
        try:
            pygame.quit()
        except Exception:
            pass
        self._save_config()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="backgrounded")
    p.add_argument("--fresh", action="store_true", help="ignore any saved world")
    p.add_argument("--scene", choices=list(SCENES), help="opening scene")
    p.add_argument("--hide", action="store_true", help="start with the window hidden")
    p.add_argument("--no-wallpaper", action="store_true",
                   help="do not touch the desktop wallpaper")
    p.add_argument("--capture", type=int, default=0,
                   help="save N png frames to the captures dir then keep running")
    p.add_argument("--capture-every", type=int, default=60,
                   help="sim ticks between captures")
    p.add_argument("--exit-after", type=float, default=0.0,
                   help="exit after N sim seconds (testing)")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--allow-multi", action="store_true",
                   help="permit a second instance (they will fight over the "
                        "wallpaper; for debugging only)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths.ensure_dirs()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.FileHandler(paths.LOG_PATH, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    if not args.allow_multi and not _acquire_single_instance():
        log.error("Backgrounded is already running - look for the tray icon. "
                  "Use --allow-multi to override.")
        return 2
    return App(args).run()
