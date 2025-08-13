"""dry_run.py – synthesize bands, pass through adapter/merge, render GR‑ERG frames"""
import numpy as np
from band_adapter import adapt
from band_merge import merge
from band_conflict import conflicts
from renderer import render_glyph  # iter + compositor implemented below

np.random.seed(42)
BANDS = ["valence", "arousal", "dominance", "curiosity", "selfref"]


def random_raw():
    return {
        "valence": np.random.uniform(-0.35, 0.40),
        "arousal": np.random.uniform(0.10, 0.85),
        "dominance": np.random.uniform(-0.30, 0.50),
        "curiosity": np.random.uniform(0.05, 0.60),
        "selfref": np.random.uniform(0.00, 0.75),
    }

history = []
for turn in range(10):
    o4_raw = random_raw()
    o3_raw = random_raw()
    o4_scaled = adapt(o4_raw)
    o3_scaled = adapt(o3_raw)  # using same adapter for demo
    bands = merge(o3_scaled, o4_scaled)
    deltas = conflicts(o3_scaled, o4_scaled)
    history.append(bands)
    print(f"Turn {turn}: conflicts {deltas}")

# Render final glyph with ghosts of previous 9 states
img = render_glyph(history)
img.save("dry_run.png")
print("Saved dry_run.png")