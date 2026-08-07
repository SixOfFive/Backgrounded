"""ToolController - the glue between preview pointer events and the sim tools.

Owns which tool is selected and any in-progress Hand drag. It reads the pointer
events Preview collects each frame (already mapped to world coordinates), and
drives backgrounded.sim.interact. The palette itself is drawn as a preview
overlay via :meth:`draw_overlay`.

Left-mouse only: a click on an icon selects a tool; a click on the world uses
it; a press-drag-release with the Hand grabs, carries and sets down or tosses.
Right-mouse pans (handled in Preview) and never reaches here.
"""
from __future__ import annotations

import logging

import pygame

from ..constants import TOOL_HAND, TOOL_NONE, TOOL_LABELS, TOOLS
from ..render import toolbar
from ..sim import interact

log = logging.getLogger(__name__)


class ToolController:
    def __init__(self) -> None:
        self.tool: str = TOOL_NONE          # start inert; pick a tool to act
        self._grab: interact.Grab | None = None

    # ------------------------------------------------------------- input --
    def handle(self, pointer_events: list, world, world_time: float) -> None:
        if world is None:
            return
        for ev in pointer_events or ():
            try:
                self._one(ev, world, world_time)
            except Exception:
                log.exception("tool event failed")

    def _one(self, ev: dict, world, world_time: float) -> None:
        kind = ev.get("type")
        sx, sy = ev.get("sx", 0), ev.get("sy", 0)
        wx, wy = ev.get("wx"), ev.get("wy")

        if kind == "down":
            hit = toolbar.hit_test((sx, sy), len(TOOLS))
            if hit is not None:
                picked = TOOLS[hit]
                # Click the selected tool again to put it away.
                self.tool = TOOL_NONE if picked == self.tool else picked
                # Putting the hand away while it is still holding someone would
                # strand them exactly as above: nothing else in this class can
                # reach a Grab once self.tool has moved on.
                if self.tool != TOOL_HAND:
                    self.release_all(world)
                return
            if wx is None:                       # clicked a letterbox bar
                return
            if self.tool == TOOL_HAND:
                # A press while something is ALREADY held has to let go of it
                # first. It should not be reachable - a press follows a release
                # - but it was, and it left a colonist frozen for the rest of
                # the session. interact.grab marks its target held and ONLY
                # interact.release clears that flag, while world.tick skips the
                # physics and the behaviour tree of anything held; so
                # overwriting self._grab dropped the only reference to the last
                # one and stranded them mid-air, out of their own life, with
                # nothing left that could ever put them down.
                #
                # The way in was the lost-button-up bug in Preview: with no
                # release event the grab persisted, and the "it self-heals on
                # the next click" consolation healed the CURSOR, not the person
                # it had been carrying. Preview now reconciles button 1 against
                # the device so that release arrives on its own, which closes
                # the ordinary road here - but this is the last line before an
                # entity is unreachable, and it costs one branch on a mouse
                # press.
                if self._grab is not None:
                    interact.release(world, self._grab,
                                     getattr(world, "terrain", None))
                    self._grab = None
                self._grab = interact.grab(world, wx, wy)
            elif self.tool != TOOL_NONE:
                msg = interact.use_tool(world, self.tool, wx, wy)
                if msg:
                    world.log_event(msg)

        elif kind == "move":
            if self._grab is not None and wx is not None:
                interact.move_held(world, self._grab, wx, wy, world_time)

        elif kind == "up":
            if self._grab is not None:
                interact.release(world, self._grab, getattr(world, "terrain", None))
                self._grab = None

    # -------------------------------------------------------------- draw --
    def draw_overlay(self, win_surface: "pygame.Surface") -> None:
        try:
            toolbar.draw(win_surface, TOOLS, self.tool, TOOL_LABELS,
                         pygame.mouse.get_pos())
        except Exception:
            log.debug("toolbar draw failed", exc_info=True)

    # ------------------------------------------------------------ status --
    def release_all(self, world) -> None:
        """Drop anything held - called on shutdown so nothing is left frozen."""
        if self._grab is not None and world is not None:
            try:
                interact.release(world, self._grab, getattr(world, "terrain", None))
            except Exception:
                pass
        self._grab = None
