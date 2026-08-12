"""The in-window settings menu - a preview-only overlay, bottom-right.

What the tray menu is on Windows, for every platform that has no tray. It
offers the same verbs and emits the same commands; only the presentation
differs, and it differs deliberately:

* **Flat, not nested.** The tray builds four popup submenus because that is
  what a Win32 popup is good at. Here the whole thing is one panel of chips,
  because a submenu that opens on hover is miserable to use over a moving
  scene, and because the flat form answers "what is the speed right now?"
  without opening anything.
* **Window pixels, like the tool palette.** Drawn after the world is scaled
  in, so it is a constant size at any zoom, and never baked into the wallpaper
  frame - the desktop cannot be clicked, so a menu there would be a picture of
  a menu.

:func:`layout` is the single source of truth for what is drawn and what is
clickable, the same arrangement :mod:`.toolbar` uses. Nothing here mutates
anything: it turns a state dict into rectangles, and :mod:`..shell.menu` turns
a click on a rectangle into a command.
"""
from __future__ import annotations

import pygame

from .. import host
from ..constants import (
    SCENES, SCENE_LABELS, SCENE_ROTATE_SEC, SPEEDS, WINDOW_SCALES,
)

GEAR = 34
MARGIN = 10
PAD = 12
ROW_H = 26
ROW_GAP = 5
CHIP_GAP = 5
CHIP_PAD = 9
LABEL_W = 62
PANEL_MAX_W = 580

_BG = (18, 20, 28)
_PANEL = (14, 16, 23, 238)
_EDGE = (90, 100, 122)
_EDGE_HI = (150, 200, 255)
_CHIP = (34, 38, 50)
_CHIP_ON = (60, 78, 120)
_CHIP_HOVER = (48, 54, 70)
_INK = (226, 232, 244)
_INK_DIM = (150, 158, 176)
_DANGER = (150, 60, 58)
_DANGER_ARMED = (200, 70, 64)

#: Verbs whose chip is a radio within its group, so the group needs the live
#: value to know which one is lit. Everything else is a checkbox or a one-shot.
KIND_RADIO = "radio"
KIND_CHECK = "check"
KIND_ACTION = "action"

_font: pygame.font.Font | None = None
_font_small: pygame.font.Font | None = None


def _fonts() -> tuple[pygame.font.Font, pygame.font.Font]:
    global _font, _font_small
    if _font is None:
        if not pygame.font.get_init():
            pygame.font.init()
        fam = "segoeui,arial,dejavusans,liberationsans,notosans"
        _font = pygame.font.SysFont(fam, 13)
        _font_small = pygame.font.SysFont(fam, 11)
    return _font, _font_small                     # type: ignore[return-value]


def gear_rect(win_size: tuple[int, int]) -> pygame.Rect:
    """The always-visible button that opens the panel, bottom-right.

    Bottom-right because it is the one corner the HUD leaves alone: the stats
    panel is top-right, the chronicle bottom-left, the tool palette top-left.
    """
    w, h = win_size
    return pygame.Rect(w - MARGIN - GEAR, h - MARGIN - GEAR, GEAR, GEAR)


def _groups(state: dict) -> list[tuple[str, list[dict]]]:
    """The menu's contents, as (heading, rows) with no geometry yet.

    Two entries the tray has are deliberately absent.

    ``toggle_window`` - "Show Window" - because hiding the window from a button
    inside that window is a trap: on a platform with no tray there is nothing
    left to click to bring it back, and on Windows the tray already offers it.

    ``toggle_wallpaper`` appears only where there is a wallpaper to write, so
    the panel never offers a switch that would do nothing.
    """
    scene = str(state.get("scene") or "")
    try:
        speed = float(state.get("sim_speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0
    try:
        scale = float(state.get("window_scale", 1.0))
    except (TypeError, ValueError):
        scale = 1.0
    mins = max(1, int(round(SCENE_ROTATE_SEC / 60.0)))

    scene_rows = [
        {"label": SCENE_LABELS.get(key, key), "verb": "scene", "payload": key,
         "kind": KIND_RADIO, "on": key == scene}
        for key in SCENES
    ]
    scene_rows.append(
        {"label": f"Auto, every {mins} min", "verb": "toggle_auto_scene",
         "payload": None, "kind": KIND_CHECK,
         "on": bool(state.get("auto_scene_change", True))})

    sim_rows = [{"label": "Pause", "verb": "toggle_pause", "payload": None,
                 "kind": KIND_CHECK, "on": bool(state.get("paused"))}]
    if host.WALLPAPER_SUPPORTED:
        sim_rows.append({"label": "Wallpaper output", "verb": "toggle_wallpaper",
                         "payload": None, "kind": KIND_CHECK,
                         "on": bool(state.get("wallpaper_enabled"))})

    return [
        ("Scene", scene_rows),
        ("Speed", [
            {"label": label, "verb": "speed", "payload": value,
             "kind": KIND_RADIO, "on": abs(value - speed) < 1e-6}
            for label, value in SPEEDS
        ]),
        ("Window", [
            {"label": f"{value * 100:.0f}%", "verb": "window_scale",
             "payload": value, "kind": KIND_RADIO,
             "on": abs(value - scale) < 1e-6}
            for value in WINDOW_SCALES
        ]),
        ("Show", [
            {"label": "Stats", "verb": "toggle_stats", "payload": None,
             "kind": KIND_CHECK, "on": bool(state.get("show_stats", True))},
            {"label": "Names", "verb": "toggle_names", "payload": None,
             "kind": KIND_CHECK, "on": bool(state.get("show_names", True))},
            {"label": "Activity", "verb": "toggle_activity", "payload": None,
             "kind": KIND_CHECK, "on": bool(state.get("show_activity", True))},
            {"label": "Log", "verb": "toggle_log", "payload": None,
             "kind": KIND_CHECK, "on": bool(state.get("show_log", True))},
        ]),
        ("Sim", sim_rows),
        ("World", [
            {"label": "New landscape", "verb": "new_terrain", "payload": None,
             "kind": KIND_ACTION, "on": False},
            {"label": "Clear graves", "verb": "clear_graves", "payload": None,
             "kind": KIND_ACTION, "on": False},
            {"label": "Save now", "verb": "save", "payload": None,
             "kind": KIND_ACTION, "on": False},
        ]),
        # Its own group, and "Start over" alone is marked dangerous. Quit is
        # not: the world is saved on the way out, so the worst it costs is a
        # relaunch. "Start over" is the only entry in the whole panel that
        # destroys a colony, and it sits four chips from "Save now".
        ("", [
            {"label": "Start over", "verb": "reset", "payload": None,
             "kind": KIND_ACTION, "on": False, "danger": True},
            {"label": "Quit", "verb": "quit", "payload": None,
             "kind": KIND_ACTION, "on": False},
        ]),
    ]


def layout(win_size: tuple[int, int], state: dict, armed: str | None = None):
    """Geometry for the open panel: ``(panel_rect, headings, items)``.

    ``items`` each carry their own ``rect`` in window pixels and the verb and
    payload to emit, so the hit-test and the paint cannot disagree - there is
    only one set of rectangles and both read it.

    ``armed`` is the verb currently awaiting a confirming second click; its
    chip is widened to hold the longer label, which is why arming has to be
    known here rather than only at paint time.
    """
    font, _ = _fonts()
    w, h = win_size
    inner_max = max(160, min(PANEL_MAX_W, w - 2 * MARGIN) - 2 * PAD)
    chip_max = max(80, inner_max - LABEL_W - CHIP_GAP)

    rows: list[list[dict]] = []                   # laid-out rows of chips
    headings: list[tuple[str, int]] = []          # (text, row index it starts)
    for title, entries in _groups(state):
        first = len(rows)
        line: list[dict] = []
        used = 0
        for entry in entries:
            text = _chip_text(entry, armed)
            cw = font.size(text)[0] + 2 * CHIP_PAD
            cw = min(cw, chip_max)
            need = cw if not line else cw + CHIP_GAP
            if line and used + need > chip_max:
                rows.append(line)
                line, used = [], 0
                need = cw
            line.append(dict(entry, _w=cw, _text=text))
            used += need
        if line:
            rows.append(line)
        headings.append((title, first))

    inner_h = len(rows) * ROW_H + max(0, len(rows) - 1) * ROW_GAP
    panel_w = LABEL_W + CHIP_GAP + chip_max + 2 * PAD
    panel_h = inner_h + 2 * PAD

    gear = gear_rect(win_size)
    x = max(MARGIN, gear.right - panel_w)
    y = max(MARGIN, gear.top - ROW_GAP - panel_h)
    panel = pygame.Rect(x, y, panel_w, panel_h)

    items: list[dict] = []
    heading_at = {idx: text for text, idx in headings}
    out_headings: list[tuple[str, int, int]] = []
    for r, line in enumerate(rows):
        top = panel.y + PAD + r * (ROW_H + ROW_GAP)
        if r in heading_at and heading_at[r]:
            out_headings.append((heading_at[r], panel.x + PAD, top))
        cx = panel.x + PAD + LABEL_W + CHIP_GAP
        for entry in line:
            rect = pygame.Rect(cx, top, entry["_w"], ROW_H)
            items.append(dict(entry, rect=rect))
            cx += entry["_w"] + CHIP_GAP
    return panel, out_headings, items


def _chip_text(entry: dict, armed: str | None) -> str:
    if armed and entry["verb"] == armed:
        return "Sure? " + entry["label"]
    return str(entry["label"])


def hit_test(pos, win_size: tuple[int, int], state: dict,
             armed: str | None = None) -> dict | None:
    """The open panel's item under a window pixel, or None."""
    _panel, _headings, items = layout(win_size, state, armed)
    for item in items:
        if item["rect"].collidepoint(pos):
            return item
    return None


# ----------------------------------------------------------------- drawing --
def draw_gear(surf: pygame.Surface, open_: bool, mouse_pos=None) -> None:
    r = gear_rect(surf.get_size())
    hot = open_ or bool(mouse_pos and r.collidepoint(mouse_pos))
    pygame.draw.rect(surf, _CHIP_ON if open_ else _BG, r, border_radius=6)
    pygame.draw.rect(surf, _EDGE_HI if hot else _EDGE, r, 2, border_radius=6)
    cx, cy = r.center
    # A cog: a ring plus eight teeth. Drawn rather than blitted so it costs no
    # asset and scales with GEAR if that is ever retuned.
    pygame.draw.circle(surf, _INK, (cx, cy), 7, 2)
    pygame.draw.circle(surf, _INK, (cx, cy), 2)
    for i in range(8):
        a = i * 3.14159 / 4.0
        dx, dy = pygame.math.Vector2(1, 0).rotate_rad(a)
        pygame.draw.line(surf, _INK,
                         (cx + dx * 8, cy + dy * 8),
                         (cx + dx * 11, cy + dy * 11), 2)


def draw(surf: pygame.Surface, state: dict, armed: str | None = None,
         mouse_pos=None) -> None:
    """Paint the open panel. Call after the HUD and the palette - it is modal
    in spirit and must not be drawn under anything."""
    font, small = _fonts()
    panel, headings, items = layout(surf.get_size(), state, armed)

    bg = pygame.Surface(panel.size, pygame.SRCALPHA)
    bg.fill(_PANEL)
    surf.blit(bg, panel.topleft)
    pygame.draw.rect(surf, _EDGE, panel, 1, border_radius=8)

    for text, x, y in headings:
        label = small.render(text.upper(), True, _INK_DIM)
        surf.blit(label, (x, y + (ROW_H - label.get_height()) // 2))

    for item in items:
        rect = item["rect"]
        hover = bool(mouse_pos and rect.collidepoint(mouse_pos))
        danger = bool(item.get("danger"))
        is_armed = armed is not None and item["verb"] == armed
        if is_armed:
            fill = _DANGER_ARMED
        elif item["on"]:
            fill = _CHIP_ON
        elif danger:
            fill = _DANGER if hover else _CHIP
        elif hover:
            fill = _CHIP_HOVER
        else:
            fill = _CHIP
        # The dangerous chip carries its red edge ALWAYS, not only under the
        # cursor. A colour that appears on hover tells you what you are about
        # to click; the point here is to be recognisable before you go near it.
        if item["on"] or is_armed:
            edge = _EDGE_HI
        elif danger:
            edge = _DANGER_ARMED
        else:
            edge = _EDGE
        pygame.draw.rect(surf, fill, rect, border_radius=5)
        pygame.draw.rect(surf, edge, rect, 1, border_radius=5)
        label = font.render(item["_text"], True, _INK)
        if label.get_width() > rect.w - 6:
            label = label.subsurface((0, 0, max(1, rect.w - 6),
                                      label.get_height()))
        surf.blit(label, (rect.centerx - label.get_width() // 2,
                          rect.centery - label.get_height() // 2))
