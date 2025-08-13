# ML Platformer — Q-learning Demo

## Introduction
This project is a small 2D platformer built with Pygame that trains a tabular Q-learning agent to reach a portal (the exit) as fast as possible. The environment features simple physics, curated platform layouts, hazards (spikes), and a lightweight UI that visualizes training stats and the agent’s current inputs (WASD overlay). The goal is to showcase practical reward shaping and a compact state representation that enables an agent to quickly learn a viable policy.

## Features
- Playable platformer with smooth movement, jump buffering, coyote time, particles, clouds, and animated portal.
- AI and Human modes: you can let the AI play, or take over and also let the agent learn from your actions.
- Epsilon-greedy tabular Q-learning with per-step epsilon decay and persistent Q-table save/load.
- Reward shaping optimized for fastest time-to-exit with additional shaping for forward progress, idling, jumping, and hazard interactions.
- Multiple themed backgrounds and multiple curated level layouts (rotate at runtime).
- Episode timing, best-time tracking, and AI completion time logging to `ml_platformer/completion_times.txt`.

### Agent state representation (discrete)
The agent observes a compact, discretized state:
- sdx: binned horizontal distance to exit (64px bins, clamped to [-30, 30])
- sdy: binned vertical distance to exit (48px bins, clamped to [-20, 20])
- vx sign: {-1, 0, 1} with a deadzone around |vx| ≤ 40
- vy sign: {-1, 0, 1} with thresholds at |vy| > 50
- on_ground: {0, 1}
- under: {0, 1} a nearby-ledge hint if a platform is under the player within ~64px

### Action space
`[ "idle", "left", "right", "jump", "left_jump", "right_jump" ]`

### Q-learning hyperparameters
- alpha (learning rate): 0.2
- gamma (discount): 0.98
- epsilon (initial exploration): 0.25
- min_epsilon: 0.02
- epsilon decay per step: 0.9985

### Episode settings
- Frame rate: 60 FPS (simulation can be sped up)
- AI update stride: every 1 frame; min action hold: 4 frames
- Episode max time: 120.0s (timeout penalty applied)
- Optional step cap: 2000 steps (config constant)

### Reward shaping (per `ml_platformer/config.py` and `ml_platformer/main.py`)
- Reach exit: +100.0 plus a time bonus `REWARD_TIME_BONUS / episode_time` with `REWARD_TIME_BONUS = 800.0`
- Fall death: −50.0
- Hazard death (spikes): −80.0 additional
- Timeout (exceeding 120s): −30.0
- Time penalty per second: −4.5
- Progress toward exit (Euclidean): `+0.25 * (old_dist − new_dist)`
- Forward horizontal motion (to the right): `+0.22 per pixel`
- Backward motion (to the left): `−0.04 per pixel`
- Idle while grounded and nearly stopped: `−2.8 per second`
- Pressing jump (held): `−0.8 per second`
- New furthest-right progress in episode: `+0.018 per pixel` beyond previous best this episode
- Clearing a spike gap and landing: +80.0 (granted after landing)

## Screenshots
- Place screenshots in `docs/images/` and reference them here. Suggested captures:
	- Gameplay with AI overlay and stats
	- Themed backgrounds and different layouts
	- Agent clearing spikes and reaching the portal

Example (add your PNGs to make these links work):
![Gameplay](docs/images/gameplay.png)
![Portal](docs/images/portal.png)

## Installation (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage
Run the game:
```powershell
python -m ml_platformer.main
```

Controls (always available):
- Esc: quit (auto-saves Q-table)
- Space: hold to fast-forward simulation (when AI is in control). In human mode Space is also jump.

Toggles and helpers:
- H: toggle AI control on/off (human vs AI)
- T: toggle training on/off (Q-updates)
- R: reset the episode
- S: save Q-table to `ml_platformer/qtable.pkl`
- L: load Q-table from `ml_platformer/qtable.pkl`
- F1: rotate level layout
- F2: rotate theme
- F12: capture screenshot to `docs/images/`

Data and logs:
- Q-table: `ml_platformer/qtable.pkl`
- AI completion times: `ml_platformer/completion_times.txt` (CSV: episode_index,seconds)
- Episode CSV log: `ml_platformer/episode_log.csv` with columns: `episode,time,reward,epsilon,steps,reason`

Tuning:
- Adjust physics, visuals, and reward weights in `ml_platformer/config.py`.
- Modify discretization, epsilon schedule, and learning rates in `ml_platformer/ai_agent.py`.

CLI examples:
```powershell
# Headless 50 episodes, fast sim, save on exit
python -m ml_platformer.main --headless --episodes 50 --speedup 3 --save-on-exit

# Human play, no training, custom seed
python -m ml_platformer.main --human --no-train --seed 7

# Start with layout 2 and theme 1 at 90 FPS
python -m ml_platformer.main --layout 2 --theme 1 --fps 90
```

## Technologies
- Python 3.x
- Pygame for rendering, input, and timing
- NumPy for RNG and Q-table value ops

## Why this project?
This demo aims to present a compact, approachable example of reinforcement learning in a custom 2D environment. It highlights:
- How reward shaping can drive a specific objective (minimizing time-to-exit) while discouraging unhelpful behavior (idling, backtracking, jump-spam).
- How a small, engineered state space and tabular Q-learning can learn useful policies quickly—without neural networks or complex frameworks.
- Practical game-loop integration, with clear controls to pause, toggle training, and visualize AI decisions.

If you need screenshots or want a short video/gif added, capture a few runs (AI mode on) and drop them into `docs/images/`—the links above will pick them up.