"""Versioned save/load.

A save file must never be able to prevent the program from starting. Anything
unreadable is moved aside and a fresh world is generated in its place.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from typing import Any

from . import paths
from .constants import SAVE_VERSION
from .sim.world import World

log = logging.getLogger(__name__)


def save_world(world: World) -> bool:
    """Atomically write the world to disk. Returns success."""
    try:
        paths.ensure_dirs()
        data = world.to_dict()
        tmp = paths.SAVE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")), "utf-8")
        tmp.replace(paths.SAVE_PATH)
    except Exception:
        log.exception("save failed")
        return False

    # The chronicle is also mirrored as plain text - it is the one part of the
    # save a human might actually want to read.
    try:
        paths.CHRONICLE_PATH.write_text("\n".join(world.chronicle), "utf-8")
    except Exception:
        pass
    return True


def load_world() -> World | None:
    """Load the saved world, or None if there isn't a usable one."""
    if not paths.SAVE_PATH.exists():
        return None
    try:
        raw = paths.SAVE_PATH.read_text("utf-8")
        data: dict[str, Any] = json.loads(raw)
    except Exception:
        log.exception("save file unreadable")
        _quarantine()
        return None

    version = data.get("version")
    if version != SAVE_VERSION:
        log.warning("save version %r != %r; starting fresh", version, SAVE_VERSION)
        _quarantine()
        return None

    try:
        world = World.from_dict(data)
    except Exception:
        log.exception("save file could not be reconstructed")
        _quarantine()
        return None

    log.info("loaded world: t=%.0fs pop=%d gen=%d",
             world.world_time, len(world.population.alive_agents()),
             world.population.generation)
    return world


def _quarantine() -> None:
    """Move a bad save aside so the next run starts clean but the evidence
    is still there to look at."""
    try:
        if paths.SAVE_PATH.exists():
            stamped = paths.CORRUPT_SAVE_PATH.with_name(
                f"save.corrupt.{int(time.time())}.json")
            shutil.move(str(paths.SAVE_PATH), str(stamped))
            log.warning("moved unusable save to %s", stamped)
    except Exception:
        log.exception("could not quarantine save file")
