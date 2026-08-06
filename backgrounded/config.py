"""User-facing settings, persisted as JSON. Load is total: a broken or
partial config never prevents startup."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, fields

from . import paths
from .constants import SCENE_NIGHT_STORM, WALLPAPER_FPS

log = logging.getLogger(__name__)

#: Bounds on the tray's Speed setting. The menu offers 0.5/1/2/4; these are
#: wider than that on purpose - the point is not to police the menu but to keep
#: a hand-edited or corrupted config.json from installing a value that stops the
#: world ticking. See :meth:`Config._clamp`.
SIM_SPEED_MIN = 0.05
SIM_SPEED_MAX = 16.0


@dataclass
class Config:
    show_window: bool = True          # preview visible by default, per spec
    show_stats: bool = True           # top-right colony/roster panel
    show_names: bool = True           # name plates above each stickman
    show_activity: bool = True        # "chopping wood" plate above each stickman
    show_log: bool = True             # last-10 chronicle log, lower-left
    hud_scale: float = 1.6            # size of the stats panel / chronicle log
    wallpaper_enabled: bool = True
    scene: str = SCENE_NIGHT_STORM    # opening scene, per spec
    wallpaper_fps: int = WALLPAPER_FPS
    sim_speed: float = 1.0            # 0.5 / 1 / 2 / 4 from the tray menu
    paused: bool = False
    population_target: int = 6
    auto_scene_change: bool = True    # let scenes evolve on their own
    scene_min_sec: float = 180.0      # dwell time before an auto scene switch
    window_scale: float = 1.0         # preview size, from Tray > Window Size
    restore_wallpaper_on_exit: bool = True
    log_level: str = "INFO"

    # ---------------------------------------------------------------- io --
    @staticmethod
    def _coerce(value, default):
        """Force *value* to the type of *default*, or give up and return it.

        A dataclass does NOT coerce: ``Config(**{"sim_speed": "fast"})`` builds
        happily and the string only detonates later, in app.py's
        ``self._sim_accum += dt * self.cfg.sim_speed``, on frame one of every
        launch. That crash then looks exactly like a poisoned SAVE - the world
        never ticked - so the recovery path blamed save.json and moved a
        perfectly good colony aside. One bad character in config.json, and the
        colony is evicted for it.

        Coercing off ``type(default)`` rather than the annotation because
        ``from __future__ import annotations`` makes every ``f.type`` a string;
        the default is a real object and cannot drift from the field it belongs
        to. bool is checked before int, being a subclass of it.
        """
        want = type(default)
        if want is bool:
            return bool(value)
        if isinstance(value, bool) and want in (int, float):
            return default                            # True is not a speed
        if want is float:
            v = float(value)
            if v != v or v in (float("inf"), float("-inf")):
                return default                        # NaN/inf are not settings
            return v
        if want is int:
            return int(value)
        if want is str:
            return value if isinstance(value, str) else default
        return value

    @classmethod
    def load(cls) -> "Config":
        try:
            raw = json.loads(paths.CONFIG_PATH.read_text("utf-8"))
        except FileNotFoundError:
            return cls()
        except Exception as exc:                      # corrupt / unreadable
            log.warning("config unreadable (%s); using defaults", exc)
            return cls()
        # json.loads happily returns a list, a bare int, or None. `.items()` on
        # any of those is an AttributeError, and it used to be raised OUT of
        # here into App.__init__, which nothing catches - so a config.json
        # holding `[]` stopped the app starting at all, on every launch, for
        # ever, and under run.pyw there is no console to say why. The docstring
        # at the top of this module promises load is total; this is what makes
        # that true.
        if not isinstance(raw, dict):
            log.warning("config.json is %s, not an object; using defaults",
                        type(raw).__name__)
            return cls()
        defaults = {f.name: f.default for f in fields(cls)}
        clean = {}
        for k, v in raw.items():
            if k not in defaults:
                continue
            try:
                clean[k] = cls._coerce(v, defaults[k])
            except (TypeError, ValueError):
                log.warning("config field %r had %r; using the default", k, v)
        try:
            cfg = cls(**clean)
        except Exception as exc:
            log.warning("config had bad values (%s); using defaults", exc)
            return cls()
        cfg._clamp()
        return cfg

    def _clamp(self) -> None:
        """Pull the few fields with a live blast radius back into range.

        sim_speed is the one that matters: NaN, 0 or a negative value leaves
        ``while self._sim_accum >= SIM_DT`` permanently false, so the window
        keeps drawing and the wallpaper keeps updating while the world never
        advances another tick. Nothing anywhere notices, and to the user the
        colony has simply frozen for ever.
        """
        try:
            if not (SIM_SPEED_MIN <= self.sim_speed <= SIM_SPEED_MAX):
                log.warning("sim_speed %r out of range; using 1.0",
                            self.sim_speed)
                self.sim_speed = 1.0
        except TypeError:
            self.sim_speed = 1.0
        self.hud_scale = min(max(float(self.hud_scale), 0.5), 6.0)
        self.window_scale = min(max(float(self.window_scale), 0.1), 8.0)
        self.wallpaper_fps = min(max(int(self.wallpaper_fps), 1), 60)

    def save(self) -> None:
        try:
            paths.ensure_dirs()
            tmp = paths.CONFIG_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(self), indent=2), "utf-8")
            tmp.replace(paths.CONFIG_PATH)
        except Exception as exc:
            log.warning("could not save config: %s", exc)
