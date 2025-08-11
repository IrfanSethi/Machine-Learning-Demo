import random
import pygame as pg
from . import config as C


class Level:
    def __init__(self):
        # Theming
        self.themes = [
            {
                "name": "Dawn",
                "BG_TOP": (32, 36, 64),
                "BG_BOTTOM": (246, 214, 189),
                "PLATFORM_COLOR": (54, 96, 170),
                "PLATFORM_EDGE": (240, 245, 255),
                "EXIT_COLOR": (120, 240, 220),
            },
            {
                "name": "Forest",
                "BG_TOP": (24, 46, 44),
                "BG_BOTTOM": (10, 18, 16),
                "PLATFORM_COLOR": (50, 140, 110),
                "PLATFORM_EDGE": (210, 245, 220),
                "EXIT_COLOR": (180, 255, 200),
            },
            {
                "name": "Sunset",
                "BG_TOP": (68, 24, 136),
                "BG_BOTTOM": (250, 120, 110),
                "PLATFORM_COLOR": (86, 72, 160),
                "PLATFORM_EDGE": (255, 240, 210),
                "EXIT_COLOR": (255, 220, 150),
            },
        ]
        self.theme_index = 0
        self.colors = self.themes[self.theme_index]

        # Layouts and geometry
        self.layout_index = 0
        self.platforms: list[pg.Rect] = []
        self.hazards: list[pg.Rect] = []
        self.spawn_x = 40
        self.spawn_y = C.HEIGHT - C.TILE - C.PLAYER_H
        self.exit_rect = pg.Rect(C.LEVEL_WIDTH - 120, C.HEIGHT - C.TILE * 5 - 48, 48, 96)
        self.exit_trigger = self.exit_rect.inflate(80, 80)
        self._apply_layout(self.layout_index)

        # Cached visuals
        self.bg_surface = self._make_gradient_surface()
        self.cloud_base = self._make_cloud_base()
        self.clouds = self._generate_clouds()
        self.portal_frames = self._make_portal_frames()
        self.level_surface = self._build_platform_surface()

    def _apply_layout(self, idx: int):
        self.platforms.clear()
        self.hazards.clear()
        # Ground
        ground_h = C.HEIGHT - C.TILE
        self.platforms.append(pg.Rect(0, ground_h, C.LEVEL_WIDTH, C.TILE))

        # Three curated layouts with different rhythms, fewer mid-air platforms
        if idx % 3 == 0:
            rng = random.Random(42)
            x = 260
            for i in range(6):
                y = ground_h - (i % 3) * C.TILE * 2 - rng.randint(0, 1) * C.TILE
                w = rng.randint(3, 5) * C.TILE
                self.platforms.append(pg.Rect(x, y, w, C.TILE // 2))
                x += rng.randint(260, 420)
            self.platforms.append(pg.Rect(1550, ground_h - C.TILE * 4, C.TILE * 3, C.TILE // 2))
            self.platforms.append(pg.Rect(2200, ground_h - C.TILE * 3, C.TILE * 5, C.TILE // 2))
        elif idx % 3 == 1:
            x = 200
            for i in range(5):
                y = ground_h - (i + 1) * (C.TILE * 1.2)
                self.platforms.append(pg.Rect(x + i * 180, int(y), C.TILE * 3, C.TILE // 2))
            self.platforms.append(pg.Rect(1400, ground_h - C.TILE * 5, C.TILE * 5, C.TILE // 2))
            self.platforms.append(pg.Rect(1800, ground_h - C.TILE * 2, C.TILE * 3, C.TILE // 2))
            self.platforms.append(pg.Rect(2050, ground_h - C.TILE * 3, C.TILE * 2, C.TILE // 2))
            self.platforms.append(pg.Rect(2300, ground_h - C.TILE * 4, C.TILE * 3, C.TILE // 2))
            self.platforms.append(pg.Rect(2550, ground_h - C.TILE * 5, C.TILE * 3, C.TILE // 2))
        else:
            x = 240
            for i in range(4):
                self.platforms.append(pg.Rect(x, ground_h - C.TILE * (2 + (i % 2)), C.TILE * 4, C.TILE // 2))
                x += 420
            self.platforms.append(pg.Rect(2200, ground_h - C.TILE * 4, C.TILE * 4, C.TILE // 2))
            self.platforms.append(pg.Rect(2500, ground_h - C.TILE * 3, C.TILE * 3, C.TILE // 2))
            self.platforms.append(pg.Rect(2800, ground_h - C.TILE * 2, C.TILE * 3, C.TILE // 2))

        # Exit and spawn placement
        self.spawn_x = 40
        self.spawn_y = C.HEIGHT - C.TILE - C.PLAYER_H
        self.exit_rect = pg.Rect(C.LEVEL_WIDTH - 120, ground_h - C.TILE * 4 - 48, 48, 96)
        self.exit_trigger = self.exit_rect.inflate(80, 80)

        # Hazards along ground: small spikes to jump over
        spike_w = 28
        spike_h = 22
        gaps = [
            (520, spike_w), (2000, spike_w), (2420, spike_w)
        ]
        for gx, gw in gaps:
            hx = gx
            hy = ground_h - spike_h + 2
            self.hazards.append(pg.Rect(hx, hy, gw, spike_h))

    def _generate_clouds(self):
        rng = random.Random(7)
        clouds = []
        for _ in range(16):  # double the clouds for more atmosphere
            cx = rng.randint(0, C.LEVEL_WIDTH)
            cy = rng.randint(10, 260)
            speed = rng.uniform(8, 36)
            scale = rng.uniform(0.5, 1.5)
            w, h = int(180 * scale), int(80 * scale)
            img = pg.transform.smoothscale(self.cloud_base, (w, h)).convert_alpha()
            clouds.append({"x": cx, "y": cy, "speed": speed, "img": img, "alpha": int(120 + 80 * scale)})
        return clouds

    def update_clouds(self, dt: float):
        for c in self.clouds:
            c["x"] += c["speed"] * dt
            if c["x"] > C.LEVEL_WIDTH + 100:
                c["x"] = -200

    def draw_background(self, surf: pg.Surface, cam_x: float):
        # Blit cached vertical gradient background
        surf.blit(self.bg_surface, (0, 0))

        # Add subtle atmospheric haze
        haze = pg.Surface((C.WIDTH, C.HEIGHT), pg.SRCALPHA)
        for y in range(0, C.HEIGHT, 8):
            alpha = int(18 + 22 * (y / C.HEIGHT))
            pg.draw.line(haze, (180, 200, 255, alpha), (0, y), (C.WIDTH, y))
        surf.blit(haze, (0, 0))

        # Parallax clouds with alpha for depth
        for c in self.clouds:
            px = int(c["x"] - cam_x * 0.4)
            py = int(c["y"])
            cloud_img = c["img"].copy()
            cloud_img.set_alpha(c["alpha"])
            surf.blit(cloud_img, (px, py))

    def _make_cloud_base(self) -> pg.Surface:
        color = (220, 235, 255, 180)
        cloud = pg.Surface((180, 80), pg.SRCALPHA)
        pg.draw.ellipse(cloud, color, pg.Rect(0, 20, 160, 50))
        pg.draw.ellipse(cloud, color, pg.Rect(40, 0, 90, 60))
        pg.draw.ellipse(cloud, color, pg.Rect(90, 10, 70, 55))
        return cloud.convert_alpha()

    def draw_platforms(self, surf: pg.Surface, cam_x: float):
        # Blit cached platforms; negative x offset scrolls with camera
        surf.blit(self.level_surface, (-int(cam_x), 0))
        # Draw hazards on top
        for h in self.hazards:
            r = pg.Rect(h.x - cam_x, h.y, h.w, h.h)
            pg.draw.polygon(
                surf, C.HAZARD_COLOR,
                [(r.left, r.bottom), (r.centerx, r.top), (r.right, r.bottom)]
            )
            pg.draw.polygon(
                surf, C.HAZARD_EDGE,
                [(r.left, r.bottom), (r.centerx, r.top), (r.right, r.bottom)], 2
            )

    def draw_exit(self, surf: pg.Surface, cam_x: float, t: float):
        # Solid beacon behind portal for visibility
        portal_center = (self.exit_rect.centerx - cam_x, self.exit_rect.centery)
        core_radius = 22
        pg.draw.circle(
            surf, (255, 250, 200), (int(portal_center[0]), int(portal_center[1])), core_radius
        )
        # Pulsing portal (cached frames)
        idx = int((t * 12) % len(self.portal_frames))
        frame = self.portal_frames[idx]
        surf.blit(
            frame,
            (
                portal_center[0] - frame.get_width() // 2,
                portal_center[1] - frame.get_height() // 2,
            ),
        )
        # Subtle base/door frame
        base_rect = pg.Rect(
            self.exit_rect.left - cam_x - 6,
            self.exit_rect.bottom - 8,
            self.exit_rect.width + 12,
            10,
        )
        pg.draw.rect(surf, (230, 230, 230), base_rect)

    def _make_gradient_surface(self) -> pg.Surface:
        surf = pg.Surface((C.WIDTH, C.HEIGHT)).convert()
        for y in range(C.HEIGHT):
            t = y / max(1, C.HEIGHT - 1)
            top = self.colors["BG_TOP"]
            bot = self.colors["BG_BOTTOM"]
            r = int(top[0] + (bot[0] - top[0]) * t)
            g = int(top[1] + (bot[1] - top[1]) * t)
            b = int(top[2] + (bot[2] - top[2]) * t)
            pg.draw.line(surf, (r, g, b), (0, y), (C.WIDTH, y))
        return surf.convert()

    def _make_portal_frames(self) -> list[pg.Surface]:
        frames: list[pg.Surface] = []
        steps = 24
        for i in range(steps):
            pulse = 0.5 + 0.5 * C.ease_in_out_sine(i / steps)
            radius = int(28 + 8 * pulse)
            size = radius * 4
            surf = pg.Surface((size, size), pg.SRCALPHA)
            center = (size // 2, size // 2)
            # Glow rings
            for j in range(6):
                alpha = int(50 - j * 8)
                rr = radius + j * 5
                col = self.colors["EXIT_COLOR"]
                pg.draw.circle(surf, (*col, max(alpha, 0)), center, rr)
            # Outline
            pg.draw.circle(surf, self.colors["EXIT_COLOR"], center, radius, 3)
            frames.append(surf.convert_alpha())
        return frames

    def _build_platform_surface(self) -> pg.Surface:
        surf = pg.Surface((C.LEVEL_WIDTH, C.HEIGHT), pg.SRCALPHA).convert_alpha()
        for r in self.platforms:
            pg.draw.rect(surf, self.colors["PLATFORM_COLOR"], r, border_radius=6)
            pg.draw.line(surf, self.colors["PLATFORM_EDGE"], (r.left, r.top), (r.right, r.top), 2)
        return surf

    # Public API
    def next_layout(self):
        self.layout_index = (self.layout_index + 1) % 3
        self._apply_layout(self.layout_index)
        self.level_surface = self._build_platform_surface()

    def next_theme(self):
        self.theme_index = (self.theme_index + 1) % len(self.themes)
        self.colors = self.themes[self.theme_index]
        self.bg_surface = self._make_gradient_surface()
        self.portal_frames = self._make_portal_frames()
        self.level_surface = self._build_platform_surface()

    def reset(self, rotate_layout: bool = False, rotate_theme: bool = False):
        if rotate_layout:
            self.next_layout()
        if rotate_theme:
            self.next_theme()

    def intersects_hazard(self, rect: pg.Rect) -> bool:
        for h in self.hazards:
            if rect.colliderect(h):
                return True
        return False
