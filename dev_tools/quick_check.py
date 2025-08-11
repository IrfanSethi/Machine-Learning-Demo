import importlib, sys, os
# Add repo root to sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

mods = [
    "ml_platformer.config",
    "ml_platformer.level",
    "ml_platformer.player",
    "ml_platformer.ai_agent",
    "ml_platformer.ui",
    "ml_platformer.main",
]
errs = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"OK {m}")
    except Exception as e:
        print(f"ERR {m}: {e}")
        errs.append((m, e))
if errs:
    sys.exit(1)
