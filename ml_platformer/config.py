import math

# Window
WIDTH = 960
HEIGHT = 540
FPS = 60
TIME_SCALE = 0.85  # < 1.0 to run simulation a bit slower
VSYNC = 0  # 0 to rely on pygame clock; set to 1 to try vsync (may vary by GPU/driver)

# World
TILE = 48
GRAVITY = 2000.0  # px/s^2
MOVE_ACCEL = 8000.0  # px/s^2
MAX_SPEED_X = 360.0  # px/s
FRICTION = 8.0
JUMP_VELOCITY = -750.0  # px/s
PLAYER_W = 40
PLAYER_H = 40

# AI
ACTIONS = [
    "idle",         # 0
    "left",         # 1
    "right",        # 2
    "jump",         # 3
    "left_jump",    # 4
    "right_jump",   # 5
]
AI_UPDATE_EVERY = 1  # frames between AI action decisions (more responsive)
MIN_ACTION_HOLD_FRAMES = 4  # smooth AI inputs to reduce flicker
EPISODE_MAX_STEPS = 2000
EPISODE_MAX_TIME_SEC = 120.0  # episode time limit to avoid endless runs

REWARD_REACH_EXIT = 100.0
REWARD_FALL_DEATH = -50.0
# Time and progress shaping (optimize for fastest time)
# Per-second penalty to encourage shorter times (frame-rate independent)
REWARD_TIME_PENALTY_PER_SEC = 4.5
# Additional shaping on horizontal progress (towards exit along X)
REWARD_PROGRESS_X_SCALE = 0.22
REWARD_PROGRESS_SCALE = 0.25  # multiplied by (old_dist - new_dist)
# Bonus inversely proportional to episode time upon reaching exit (seconds)
REWARD_TIME_BONUS = 800.0
# Extra penalty for idling on ground (per second)
IDLE_PENALTY_PER_SEC = 2.8
# Small penalty when the agent presses jump (per second held)
JUMP_PENALTY_PER_SEC = 0.8
# New: reward for pushing the furthest progress rightwards (only when exceeding prior best this episode)
REWARD_FURTHEST_X_PER_PX = 0.018
# New: small penalty for moving left (away from exit) per pixel
LEFT_MOVE_PENALTY_PER_PX = 0.04
# New: penalty when hitting the episode time limit without finishing
TIMEOUT_PENALTY = -30.0

# Penalty for dying to a hazard (spike)
HAZARD_DEATH_PENALTY = -80.0

# Visuals
BG_TOP = (18, 31, 56)
BG_BOTTOM = (6, 8, 12)
PLATFORM_COLOR = (40, 160, 200)
PLATFORM_EDGE = (255, 255, 255)
PLAYER_COLOR = (255, 210, 90)
PLAYER_OUTLINE = (30, 25, 20)
EXIT_COLOR = (120, 240, 220)
TEXT_COLOR = (240, 240, 240)
HAZARD_COLOR = (230, 70, 70)
HAZARD_EDGE = (255, 230, 230)

# Camera
CAMERA_LERP = 0.1

# Level size
LEVEL_WIDTH = 3200
LEVEL_HEIGHT = HEIGHT

# Sprites
# If False, the player is drawn programmatically (no external PNG), which avoids
# any stray edge pixels/artifacts from image scaling.
USE_IMAGE_SPRITE = True
DRAW_FACE = False
DRAW_PLAYER_OUTLINE = False
DRAW_SHADOW = False
DRAW_PARTICLES = False

def ease_in_out_sine(t: float) -> float:
    return -(math.cos(math.pi * t) - 1) / 2
