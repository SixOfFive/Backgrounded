"""User-facing settings, persisted as JSON. Load is total: a broken or
partial config never prevents startup."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, fields

from . import paths
from .constants import SCENE_NIGHT_STORM, WALLPAPER_FPS

log = logging.getLogger(__name__)


@dataclass
class Config:
    show_window: bool = True          # preview visible by default, per spec
    show_stats: bool = True           # top-right colony/roster panel
    show_names: bool = True           # name plates above each stickman
    wallpaper_enabled: bool = True
    scene: str = SCENE_NIGHT_STORM    # opening scene, per spec
    wallpaper_fps: int = WALLPAPER_FPS
    sim_speed: float = 1.0            # 0.5 / 1 / 2 / 4 from the tray menu
    paused: bool = False
    population_target: int = 6
    auto_scene_change: bool = True    # let scenes evolve on their own
    scene_min_sec: float = 180.0      # dwell time before an auto scene switch
    window_scale: float = 1.0         # preview window size multiplier
    restore_wallpaper_on_exit: bool = True
    log_level: str = "INFO"

    # ---------------------------------------------------------------- io --
    @classmethod
    def load(cls) -> "Config":
        try:
            raw = json.loads(paths.CONFIG_PATH.read_text("utf-8"))
        except FileNotFoundError:
            return cls()
        except Exception as exc:                      # corrupt / unreadable
            log.warning("config unreadable (%s); using defaults", exc)
            return cls()
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in raw.items() if k in known}
        try:
            return cls(**clean)
        except Exception as exc:
            log.warning("config had bad values (%s); using defaults", exc)
            return cls()

    def save(self) -> None:
        try:
            paths.ensure_dirs()
            tmp = paths.CONFIG_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(self), indent=2), "utf-8")
            tmp.replace(paths.CONFIG_PATH)
        except Exception as exc:
            log.warning("could not save config: %s", exc)
