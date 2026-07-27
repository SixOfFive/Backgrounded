"""The left-mouse tool palette - a preview-only overlay, top-left.

Drawn in *window* pixels (after the world has been scaled in), so it stays a
constant size whatever the zoom, and never appears on the desktop wallpaper -
the wallpaper cannot be clicked, so a palette there would be pointless.

Geometry lives here alongside the drawing so the controller hit-tests exactly
what the player sees: :func:`icon_rects` is the single source of truth for
both.
"""
from __future__ import annotations

import math

import pygame

from ..constants import (
    TOOL_FEED, TOOL_HAND, TOOL_LIGHTNING, TOOL_METEOR, TOOL_PLANT,
    TOOL_ROCK, TOOL_SPAWN,
)

ICON = 34
GAP = 6
MARGIN = 10

_BG = (18, 20, 28)
_BG_SEL = (60, 78, 120)
_EDGE = (90, 100, 122)
_EDGE_SEL = (150, 200, 255)
_INK = (226, 232, 244)


def icon_rects(n: int) -> list[pygame.Rect]:
    """The clickable box for each of `n` tools, top-left, stacked vertically."""
    return [pygame.Rect(MARGIN, MARGIN + i * (ICON + GAP), ICON, ICON)
            for i in range(n)]


def hit_test(pos, n: int) -> int | None:
    """Index of the tool box under a window pixel, or None."""
    for i, r in enumerate(icon_rects(n)):
        if r.collidepoint(pos):
            return i
    return None


def draw(surf: pygame.Surface, tools: tuple, selected: str,
         labels: dict, mouse_pos=None) -> None:
    """Paint the palette and, for whatever the cursor is over, its label."""
    rects = icon_rects(len(tools))
    hover = None
    for tool, r in zip(tools, rects):
        sel = tool == selected
        pygame.draw.rect(surf, _BG_SEL if sel else _BG, r, border_radius=6)
        pygame.draw.rect(surf, _EDGE_SEL if sel else _EDGE, r, 2, border_radius=6)
        _glyph(surf, tool, r)
        if mouse_pos and r.collidepoint(mouse_pos):
            hover = (tool, r)

    if hover is not None:
        tool, r = hover
        _tooltip(surf, labels.get(tool, tool), r)


# ------------------------------------------------------------------ glyphs --
def _glyph(surf: pygame.Surface, tool: str, r: pygame.Rect) -> None:
    cx, cy = r.center
    ink = _INK
    if tool == TOOL_HAND:
        # a simple mitten: palm block + thumb
        pygame.draw.rect(surf, ink, (cx - 6, cy - 4, 10, 11), border_radius=3)
        pygame.draw.rect(surf, ink, (cx - 9, cy - 1, 4, 6), border_radius=2)
        for k in range(4):
            pygame.draw.line(surf, ink, (cx - 5 + k * 3, cy - 4),
                             (cx - 5 + k * 3, cy - 9), 2)
    elif tool == TOOL_LIGHTNING:
        pts = [(cx + 3, cy - 10), (cx - 5, cy + 1), (cx, cy + 1),
               (cx - 3, cy + 10), (cx + 6, cy - 3), (cx + 1, cy - 3)]
        pygame.draw.polygon(surf, (255, 230, 120), pts)
    elif tool == TOOL_METEOR:
        pygame.draw.circle(surf, (170, 110, 70), (cx + 3, cy + 3), 6)
        pygame.draw.circle(surf, (255, 180, 90), (cx + 3, cy + 3), 6, 1)
        for k in range(3):
            pygame.draw.line(surf, (255, 150, 60),
                             (cx - 10 + k * 3, cy - 10 + k * 3),
                             (cx - 3 + k * 3, cy - 3 + k * 3), 2)
    elif tool == TOOL_PLANT:
        pygame.draw.line(surf, (140, 96, 54), (cx, cy + 9), (cx, cy - 1), 3)
        pygame.draw.circle(surf, (110, 190, 110), (cx, cy - 4), 6)
        pygame.draw.circle(surf, (90, 165, 90), (cx - 4, cy), 4)
        pygame.draw.circle(surf, (90, 165, 90), (cx + 4, cy), 4)
    elif tool == TOOL_ROCK:
        pts = [(cx - 8, cy + 6), (cx - 4, cy - 4), (cx + 3, cy - 6),
               (cx + 8, cy + 2), (cx + 5, cy + 6)]
        pygame.draw.polygon(surf, (150, 152, 158), pts)
        pygame.draw.polygon(surf, (100, 102, 110), pts, 1)
    elif tool == TOOL_FEED:
        pygame.draw.circle(surf, (210, 70, 60), (cx, cy + 2), 7)
        pygame.draw.circle(surf, (240, 120, 90), (cx - 2, cy), 2)
        pygame.draw.line(surf, (120, 90, 50), (cx, cy - 5), (cx, cy - 9), 2)
        pygame.draw.circle(surf, (110, 190, 110), (cx + 3, cy - 8), 2)
    elif tool == TOOL_SPAWN:
        pygame.draw.circle(surf, ink, (cx, cy - 6), 3)          # head
        pygame.draw.line(surf, ink, (cx, cy - 3), (cx, cy + 4), 2)   # body
        pygame.draw.line(surf, ink, (cx, cy - 1), (cx - 4, cy + 3), 2)
        pygame.draw.line(surf, ink, (cx, cy - 1), (cx + 4, cy + 3), 2)
        pygame.draw.line(surf, ink, (cx, cy + 4), (cx - 4, cy + 9), 2)
        pygame.draw.line(surf, ink, (cx, cy + 4), (cx + 4, cy + 9), 2)
        pygame.draw.line(surf, (140, 220, 140), (cx + 7, cy - 6),
                         (cx + 7, cy - 1), 2)                    # a little +
        pygame.draw.line(surf, (140, 220, 140), (cx + 5, cy - 4),
                         (cx + 9, cy - 4), 2)


_font: pygame.font.Font | None = None


def _tooltip(surf: pygame.Surface, text: str, r: pygame.Rect) -> None:
    global _font
    if _font is None:
        if not pygame.font.get_init():
            pygame.font.init()
        _font = pygame.font.SysFont("segoeui,arial", 13)
    label = _font.render(text, True, _INK)
    pad = 5
    box = pygame.Rect(r.right + 8, r.centery - label.get_height() // 2 - pad,
                      label.get_width() + pad * 2, label.get_height() + pad * 2)
    bg = pygame.Surface(box.size, pygame.SRCALPHA)
    bg.fill((12, 14, 20, 220))
    surf.blit(bg, box.topleft)
    pygame.draw.rect(surf, _EDGE, box, 1, border_radius=4)
    surf.blit(label, (box.x + pad, box.y + pad))
