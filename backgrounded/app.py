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
    AUTOSAVE_SEC, RENDER_SIZE, SCENES, SIM_DT, TARGET_FPS,
)
from .render.renderer import Renderer
from .shell.preview import Preview
from .shell.tools import ToolController
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
        # Honour the config's auto-scene switch (default on): the world flips to
        # a random new scene every SCENE_ROTATE_SEC unless the user turned it off.
        self.world.auto_scene_rotate = bool(getattr(self.cfg, "auto_scene_change", True))

        # --- pygame ------------------------------------------------------
        pygame.init()
        pygame.display.set_caption("Backgrounded - Live View")
        self.preview = Preview(RENDER_SIZE, scale=self.cfg.window_scale)
        self.preview.ensure_window(self.cfg.show_window)
        self.renderer = Renderer()
        self.renderer.show_stats = getattr(self.cfg, "show_stats", True)
        self.renderer.show_names = getattr(self.cfg, "show_names", True)
        self.renderer.show_log = getattr(self.cfg, "show_log", True)
        # HUD size is a module-level dial rather than a Renderer flag because the
        # wallpaper bake and the window overlay go through the same two draw
        # functions and must agree; applying the stored value here is what makes
        # the [ and ] adjustment survive a restart.
        try:
            from .render import hud as _hud
            _hud.HUD_SCALE = float(getattr(self.cfg, "hud_scale", _hud.HUD_SCALE))
        except Exception:
            log.debug("could not apply stored hud scale", exc_info=True)
        self.tools = ToolController()

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
        # Frame counter, used only to bake the HUD into the world frame at most
        # once per frame - see _bake_hud.
        self._frame_no = 0
        self._hud_frame = -1

    # ---------------------------------------------------------- tray glue --
    def _tray_state(self) -> dict:
        return {
            "show_window": self.cfg.show_window,
            "paused": self.cfg.paused,
            "scene": self.world.events.scene if self.world else self.cfg.scene,
            "sim_speed": self.cfg.sim_speed,
            "wallpaper_enabled": self.cfg.wallpaper_enabled,
            "show_stats": getattr(self.cfg, "show_stats", True),
            "show_names": getattr(self.cfg, "show_names", True),
            "show_log": getattr(self.cfg, "show_log", True),
            "auto_scene_change": getattr(self.cfg, "auto_scene_change", True),
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

    def _nudge_hud_scale(self, step: int) -> None:
        """Grow or shrink the stats panel and chronicle log, and remember it.

        Screen-anchoring the HUD means it is drawn at its authored pixel size
        rather than being magnified by the letterbox fit, so how big it *should*
        be depends on the monitor it is being read on. That is a per-user
        judgement, not something a constant can be right about, hence a live
        control - and hence persisting it, since being asked to re-tune it on
        every launch would be worse than the original hard-coded number.
        """
        try:
            from .render import hud
            lo, hi = 1.0, 3.5
            new = round(min(hi, max(lo, hud.HUD_SCALE + 0.15 * step)), 2)
            if abs(new - hud.HUD_SCALE) < 1e-6:
                return
            hud.HUD_SCALE = new
            self.cfg.hud_scale = new
            self._save_config("hud_scale")
            log.info("hud scale %.2f", new)
        except Exception:
            log.debug("could not change hud scale", exc_info=True)

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
        elif kind == "toggle_names":
            self.renderer.show_names = not self.renderer.show_names
            cfg.show_names = self.renderer.show_names
            self._save_config("show_names")
        elif kind == "toggle_log":
            self.renderer.show_log = not self.renderer.show_log
            cfg.show_log = self.renderer.show_log
            self._save_config("show_log")
        elif kind == "toggle_auto_scene":
            # The setting lives on the config; the world holds the live copy the
            # tick actually reads, so both move together or the menu would show
            # one thing while the weather did another.
            cfg.auto_scene_change = not getattr(cfg, "auto_scene_change", True)
            if self.world is not None:
                self.world.auto_scene_rotate = cfg.auto_scene_change
            self._save_config("auto_scene_change")
            log.info("auto scene rotation %s",
                     "on" if cfg.auto_scene_change else "off")
        elif kind == "new_terrain":
            self.world.randomise_terrain()
            persist.save_world(self.world)
        elif kind == "clear_graves":
            n = self.world.clear_graves()
            log.info("cleared %d graves", n)
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
        self._frame_no += 1

        self._drain_commands()
        pointer = self.preview.pump_events()
        # Held WASD / arrows, polled after the event pump so a key released this
        # frame has already been seen. Self-guarding: a hidden, unfocused or
        # un-zoomed preview does nothing here.
        self.preview.pan_keys(dt)
        self.tools.handle(pointer.get("pointer"), self.world, self.world.world_time)
        step = pointer.get("hud_scale")
        if step:
            self._nudge_hud_scale(int(step))
        if pointer.get("closed"):
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

        # One render, two destinations, and the HUD has to land differently on
        # each - so the ordering below is load-bearing. The window gets it as a
        # *screen-anchored* overlay painted straight onto the window surface,
        # which is what keeps the panels in the window's corners while the world
        # is zoomed and panned beneath them. Only afterwards does the wallpaper
        # path bake the HUD into the shared frame, world-anchored, because a
        # wallpaper is a bare image with no corner of its own to hold. Present
        # first, bake second: the other order would put both copies on screen.
        if self.cfg.show_window:
            self.preview.present(frame, overlay=self._draw_window_overlay)
            self.preview.set_caption(self._hud())

        now = time.monotonic()
        if self.cfg.wallpaper_enabled and (now - self._last_wall) >= 1.0 / max(1, self.cfg.wallpaper_fps):
            self._last_wall = now
            try:
                self._bake_hud(frame)
                raw = pygame.image.tobytes(frame, "RGB")
                self.wallpaper.submit(raw, RENDER_SIZE)
            except Exception:
                log.exception("wallpaper submit failed")

        if self.args.capture and self._captures < self.args.capture:
            if self.world.tick_count % self.args.capture_every == 0:
                # Captures are meant to look like the wallpaper, so they need
                # the baked-in HUD too. _bake_hud is a no-op if the wallpaper
                # push above already did it this frame.
                self._bake_hud(frame)
                self._save_capture(frame)

        if now - self._last_save >= AUTOSAVE_SEC:
            self._last_save = now
            persist.save_world(self.world)

        if self.args.exit_after and self.world.world_time >= self.args.exit_after:
            log.info("exit-after reached")
            self.running = False

    def _draw_window_overlay(self, screen: pygame.Surface) -> None:
        """Everything drawn in *window* pixels, after the world is scaled in.

        The HUD goes down first and the tool palette on top, so a tooltip is
        never buried under the stats panel. Both halves fail soft - an overlay
        that raises must not cost the frame.
        """
        try:
            self.renderer.draw_hud(screen, self.world)
        except Exception:
            log.debug("window hud draw failed", exc_info=True)
        self.tools.draw_overlay(screen)

    def _bake_hud(self, frame: pygame.Surface) -> None:
        """Bake the HUD into the world frame, at most once per frame.

        The wallpaper has no window for the panels to anchor to, so they are
        drawn into the 1600x1000 image at exactly the position they have always
        held, screen shake included. The once-per-frame guard is not an
        optimisation: the wallpaper push and --capture consume the *same*
        surface, and the panel is translucent, so a second blit would visibly
        darken it.
        """
        if self._hud_frame == self._frame_no:
            return
        self._hud_frame = self._frame_no
        try:
            self.renderer.draw_hud(frame, self.world, shake=True)
        except Exception:
            log.debug("wallpaper hud draw failed", exc_info=True)

    def _hud(self) -> str:
        w = self.world
        pop = len(w.population.alive_agents())
        z = getattr(self.preview, "zoom", 1.0)
        zoom = f" - {z:.1f}x" if z > 1.01 else ""
        return (f"Backgrounded - {w.events.scene} - pop {pop} "
                f"gen {w.population.generation} - "
                f"wood {w.stockpile.get('wood',0)} stone {w.stockpile.get('stone',0)} "
                f"food {w.stockpile.get('food',0)}{zoom} - "
                f"{self.clock.get_fps():.0f} fps  "
                f"[wheel=zoom wasd/arrows=pan []=text 0=reset F11=full]")

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
            self.tools.release_all(self.world)
        except Exception:
            pass
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
