import pygame as pg
from dataclasses import dataclass
from . import config as C


@dataclass
class InputState:
    left: bool = False
    right: bool = False
    jump: bool = False


class Player:
    def __init__(self, spawn_x: int, spawn_y: int):
        self.rect = pg.Rect(spawn_x, spawn_y, C.PLAYER_W, C.PLAYER_H)
        # Subpixel position accumulators to avoid truncation-induced stickiness
        self._fx = float(self.rect.x)
        self._fy = float(self.rect.y)
        self.vel = pg.Vector2(0, 0)
        self.on_ground = False
        self.time_since_ground = 0.0
        self.jump_buffer = 0.0
        self.facing = 1
        self.alive = True
        self.particles = []
        # Sprite
        try:
            img = pg.image.load("Sprites/Cube.png").convert_alpha()
            self._base_sprite = img
            scaled = pg.transform.scale(self._base_sprite, (self.rect.w, self.rect.h))
            self.sprite = self._apply_rect_mask(scaled)
        except Exception:
            self._base_sprite = None
            self.sprite = None
        self.last_input = InputState()

    def reset(self, spawn_x: int, spawn_y: int):
        self.rect.x, self.rect.y = spawn_x, spawn_y
        self._fx, self._fy = float(self.rect.x), float(self.rect.y)
        self.vel.update(0, 0)
        self.on_ground = False
        self.time_since_ground = 0.0
        self.jump_buffer = 0.0
        self.facing = 1
        self.alive = True
        self.particles.clear()
        # Ensure sprite matches rect size on reset
        if self._base_sprite:
            scaled = pg.transform.scale(self._base_sprite, (self.rect.w, self.rect.h))
            self.sprite = self._apply_rect_mask(scaled)

    def update(self, dt: float, level, inp: InputState):
        self.last_input = inp
        # Horizontal movement
        ax = 0.0
        if inp.left:
            ax -= C.MOVE_ACCEL
            self.facing = -1
        if inp.right:
            ax += C.MOVE_ACCEL
            self.facing = 1

        # Apply friction when no input
        if ax == 0.0:
            self.vel.x -= self.vel.x * min(C.FRICTION * dt, 1.0)
        else:
            self.vel.x += ax * dt

        # Clamp horizontal speed
        if self.vel.x > C.MAX_SPEED_X:
            self.vel.x = C.MAX_SPEED_X
        if self.vel.x < -C.MAX_SPEED_X:
            self.vel.x = -C.MAX_SPEED_X

        # Jump buffering and coyote time
        self.time_since_ground += dt
        if inp.jump:
            self.jump_buffer = 0.12
        else:
            self.jump_buffer = max(0.0, self.jump_buffer - dt)

        if (self.on_ground or self.time_since_ground < 0.12) and self.jump_buffer > 0.0:
            self.vel.y = C.JUMP_VELOCITY
            self.on_ground = False
            self.time_since_ground = 0.5  # prevent double-coyote
            self.jump_buffer = 0.0
            self._emit_jump_particles()

        # Gravity
        self.vel.y += C.GRAVITY * dt
        if self.vel.y > 2000:
            self.vel.y = 2000

        # Move and collide: X then Y
        self._move_axis(level.platforms, self.vel.x * dt, 0.0)
        self._move_axis(level.platforms, 0.0, self.vel.y * dt)

        # Death condition
        if self.rect.top > C.HEIGHT + 200:
            self.alive = False

        # Particles update
        self._update_particles(dt)

    def _move_axis(self, platforms, dx: float, dy: float):
        if dx != 0.0:
            self._fx += dx
            self.rect.x = int(self._fx)
        if dy != 0.0:
            self._fy += dy
            self.rect.y = int(self._fy)

        # Ground check reset each Y movement
        if dy != 0.0:
            self.on_ground = False

        for p in platforms:
            if self.rect.colliderect(p):
                if dx > 0:
                    self.rect.right = p.left
                    self.vel.x = 0
                elif dx < 0:
                    self.rect.left = p.right
                    self.vel.x = 0
                if dy > 0:
                    self.rect.bottom = p.top
                    self.vel.y = 0
                    if not getattr(self, "_landed_this_frame", False):
                        self._emit_land_particles(abs(self.vel.x))
                    self.on_ground = True
                    self.time_since_ground = 0.0
                elif dy < 0:
                    self.rect.top = p.bottom
                    self.vel.y = 0

        self._landed_this_frame = self.on_ground and dy > 0.0
        # Keep float positions in sync after any collision corrections
        self._fx = float(self.rect.x)
        self._fy = float(self.rect.y)

    def draw(self, surf: pg.Surface, cam_x: float, t: float):
        # Shadow
        shadow = pg.Rect(self.rect.x - cam_x + 4, self.rect.bottom - 8, self.rect.w, 8)
        pg.draw.ellipse(surf, (0, 0, 0, 120), shadow)

        # Body (sprite if available, fallback to rect)
        body = pg.Rect(self.rect.x - cam_x, self.rect.y, self.rect.w, self.rect.h)
        if self.sprite:
            surf.blit(self.sprite, body)
        else:
            pg.draw.rect(surf, C.PLAYER_OUTLINE, body.inflate(6, 6), border_radius=8)
            pg.draw.rect(surf, C.PLAYER_COLOR, body, border_radius=8)

        # Face
        if not self.sprite:
            eye_y = body.y + 16
            eye_x = body.centerx + 6 * self.facing
            pg.draw.circle(surf, (40, 40, 40), (eye_x - cam_x, eye_y), 4)

        # Particles
        for part in self.particles:
            alpha = max(0, int(255 * part["life"]))
            col = (*part["color"], alpha)
            pg.draw.circle(surf, col, (int(part["x"] - cam_x), int(part["y"])), int(part["r"]))

    def _emit_jump_particles(self):
        for i in range(4):
            self.particles.append({
                "x": self.rect.centerx,
                "y": self.rect.bottom,
                "vx": (i - 4) * 30,
                "vy": -120 - i * 10,
                "r": 3,
                "life": 1.0,
                "color": (255, 255, 255)
            })

    def _emit_land_particles(self, speed_x: float):
        if speed_x < 60:
            return
        for i in range(5):
            self.particles.append({
                "x": self.rect.centerx + (i - 5) * 2,
                "y": self.rect.bottom + 1,
                "vx": (i - 5) * 40,
                "vy": -80 - abs(i - 5) * 10,
                "r": 3,
                "life": 1.0,
                "color": (200, 220, 240)
            })

    def _update_particles(self, dt: float):
        alive = []
        for p in self.particles:
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["vy"] += C.GRAVITY * 0.6 * dt
            p["life"] -= dt * 1.6
            p["r"] = max(1, p["r"] - dt * 4)
            if p["life"] > 0:
                alive.append(p)
        self.particles = alive

    def _apply_rect_mask(self, surf: pg.Surface) -> pg.Surface:
        # Clip sprite strictly to a rounded rectangle to hide any stray edge pixels
        w, h = surf.get_width(), surf.get_height()
        out = surf.copy()
        mask = pg.Surface((w, h), pg.SRCALPHA)
        mask.fill((255, 255, 255, 0))
        # Slight rounding for a softer look; set to 0 for perfect square edges
        rr = 4
        pg.draw.rect(mask, (255, 255, 255, 255), pg.Rect(0, 0, w, h), border_radius=rr)
        out.blit(mask, (0, 0), special_flags=pg.BLEND_RGBA_MULT)
        return out
