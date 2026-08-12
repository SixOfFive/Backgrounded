"""MenuController - the in-window settings menu's open/closed state and clicks.

The same shape as :class:`~backgrounded.shell.tools.ToolController`: it reads
the pointer events Preview collected this frame, acts on the ones that landed
on its own overlay, and hands the rest back. It takes the same
``(cmd_q, get_state)`` pair :class:`~backgrounded.shell.tray.Tray` takes, and
emits the same ``(verb, payload)`` tuples onto the same queue.

That symmetry is the whole design. ``App._handle`` already implements every
verb the tray can send; this is a second producer for that queue, not a second
implementation of anything. Adding the menu required no change to _handle at
all, and a verb added there is reachable from both menus at once.

Clicks that land on the menu are **consumed**, for the same reason
``App._roster_click`` consumes: the panel is drawn on top of the scene, so
every pixel of it also has a perfectly good world position underneath, and
letting a click through would change the speed AND strike lightning wherever
the chip happened to be sitting.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import pygame

from ..render import menu

log = logging.getLogger(__name__)

#: Verbs that take two clicks: the first arms, the second fires. Only ``reset``
#: is here, and only because it is the one entry that destroys a colony that
#: cannot be got back. The tray never armed it - it did not need to, since
#: reaching it meant walking a popup menu, whereas here it is one click sitting
#: a few pixels from "Save now".
_ARMS = frozenset({"reset"})


class MenuController:
    def __init__(self, cmd_q: "Any", get_state: Callable[[], dict],
                 show_gear: bool = True) -> None:
        self.cmd_q = cmd_q
        self.get_state = get_state
        #: Whether the button is painted. The key binding works regardless, so
        #: this only governs whether the menu advertises itself - see App.
        self.show_gear = bool(show_gear)
        self._open = False
        self._armed: str | None = None
        #: A consumed press must consume its own release too, or the tools see
        #: an orphan button-up. Same lost-half-a-gesture problem Preview's
        #: _lmb latch exists for, one layer up.
        self._swallow_up = False

    # --------------------------------------------------------------- state --
    @property
    def open(self) -> bool:
        return self._open

    @property
    def armed(self) -> str | None:
        return self._armed

    def toggle(self) -> bool:
        self._open = not self._open
        if not self._open:
            self._armed = None
        return self._open

    def close(self) -> None:
        self._open = False
        self._armed = None

    # -------------------------------------------------------------- events --
    def handle(self, events, win_size: tuple[int, int]):
        """Act on clicks that hit the menu; return the events that did not.

        Order inside a frame matters and is set at the call site: this runs
        before the roster and before the tools, because the panel is painted
        over both.
        """
        if not events:
            return events
        kept = []
        for ev in events:
            try:
                if self._consume(ev, win_size):
                    continue
            except Exception:
                log.exception("menu event failed")
            kept.append(ev)
        return kept

    def _consume(self, ev: dict, win_size: tuple[int, int]) -> bool:
        kind = ev.get("type")
        button = int(ev.get("button") or 0)
        if kind == "up" and button == 1 and self._swallow_up:
            self._swallow_up = False
            return True
        if kind != "down" or button != 1:
            return False

        pos = (ev.get("sx"), ev.get("sy"))
        if pos[0] is None or pos[1] is None:
            return False

        if self.show_gear and menu.gear_rect(win_size).collidepoint(pos):
            self.toggle()
            self._swallow_up = True
            return True

        if not self._open:
            return False

        item = menu.hit_test(pos, win_size, self.get_state(), self._armed)
        if item is None:
            # A click anywhere else closes the panel, and is spent doing it.
            # Spent rather than passed through because the alternative is that
            # dismissing a menu also plants a tree.
            self.close()
            self._swallow_up = True
            return True

        self._activate(item)
        self._swallow_up = True
        return True

    def _activate(self, item: dict) -> None:
        verb = str(item.get("verb") or "")
        if not verb:
            return
        if verb in _ARMS and self._armed != verb:
            self._armed = verb
            return
        # Any other choice disarms, but still does its own job - a click that
        # silently did nothing except cancel something invisible is worse than
        # one that acts.
        self._armed = None
        self._emit(verb, item.get("payload"))
        # One-shots and destructive verbs close the panel behind them; the
        # toggles and radios do not, because setting the speed and then the
        # scene is one visit, not two.
        if item.get("kind") == menu.KIND_ACTION:
            self.close()

    def _emit(self, verb: str, payload) -> None:
        try:
            self.cmd_q.put((verb, payload))
        except Exception:
            log.exception("menu could not queue %r", verb)

    # ------------------------------------------------------------- drawing --
    def draw_overlay(self, win_surface: "pygame.Surface") -> None:
        try:
            mouse = pygame.mouse.get_pos() if pygame.mouse.get_focused() else None
        except Exception:
            mouse = None
        try:
            if self.show_gear:
                menu.draw_gear(win_surface, self._open, mouse)
            if self._open:
                menu.draw(win_surface, self.get_state(), self._armed, mouse)
        except Exception:
            log.debug("menu draw failed", exc_info=True)
