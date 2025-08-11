import pygame as pg
from . import config as C


class UI:
    def __init__(self):
        pg.font.init()
        self.font = pg.font.SysFont("consolas", 18)

    def draw(self, surf, info: dict):
        best = info.get("best_time")
        best_str = "—" if best is None else f"{best:.2f}s"
        reason = info.get("reason")
        reason_str = "" if not reason else f"  Last reset: {reason}"
        lines = [
            f"Mode: {'AI' if info['ai_control'] else 'HUMAN'}",
            f"Episode: {info['episodes']}",
            f"Reward: {info['reward']:.2f}{reason_str}",
            f"Time: {info.get('time', 0.0):.2f}s  Best: {best_str}",
        ]
        x, y = 12, 10
        for ln in lines:
            self._text(surf, ln, x + 1, y + 1, (0, 0, 0))
            self._text(surf, ln, x, y, C.TEXT_COLOR)
            y += 20

        # Optional: draw AI WASD overlay if provided and AI control is active
        if info.get("ai_control") and info.get("ai_wasd"):
            self._draw_wasd(surf, info["ai_wasd"])

    def _text(self, surf, txt, x, y, color):
        img = self.font.render(txt, True, color)
        surf.blit(img, (x, y))

    def _draw_wasd(self, surf, pressed: dict):
        # Layout near top-right
        sw = surf.get_width()
        box = 32
        gap = 6
        total_w = box * 3 + gap * 2
        x0 = sw - total_w - 16
        y0 = 16 + 20 * 4  # position below the text block

        # Top row: W
        self._draw_key(surf, "W", x0 + box + gap, y0, bool(pressed.get("w")), box)
        # Bottom row: A S D
        y1 = y0 + box + gap
        self._draw_key(surf, "A", x0, y1, bool(pressed.get("a")), box)
        self._draw_key(surf, "S", x0 + box + gap, y1, bool(pressed.get("s")), box)
        self._draw_key(surf, "D", x0 + (box + gap) * 2, y1, bool(pressed.get("d")), box)
        # Small caption
        cap = "AI inputs"
        cap_img = self.font.render(cap, True, C.TEXT_COLOR)
        surf.blit(cap_img, (x0, y1 + box + gap))

    def _draw_key(self, surf, label: str, px: int, py: int, is_down: bool, box: int):
        rect = pg.Rect(px, py, box, box)
        base = (30, 34, 42)
        on = (90, 200, 255)
        off = (70, 80, 95)
        fill = on if is_down else off
        pg.draw.rect(surf, base, rect.inflate(6, 6), border_radius=6)
        pg.draw.rect(surf, fill, rect, border_radius=6)
        pg.draw.rect(surf, (15, 18, 22), rect, width=2, border_radius=6)
        # Label
        label_img = self.font.render(label, True, (15, 18, 22))
        lw, lh = label_img.get_size()
        surf.blit(label_img, (px + (box - lw) // 2, py + (box - lh) // 2))
