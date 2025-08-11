# Machine-Learning-Demo

# ML Platformer Demo

A simple platformer game built with Pygame where an AI agent learns with Q-learning and is rewarded for getting closer to the exit. Includes a gradient sky, parallax clouds, and a pulsing portal.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ml_platformer.main
```

## Controls
- T: toggle training
- H: toggle AI vs human control
- R: reset episode
- S: save Q-table
- L: load Q-table
- Esc: quit (auto-saves)

## Notes
- The AI uses a discretized state and tabular Q-learning with epsilon-greedy exploration.
- Rewards are shaped based on change in distance to the exit, with bonuses/penalties for success/fall.
- Tweak constants in `ml_platformer/config.py` to adjust difficulty and training speed.