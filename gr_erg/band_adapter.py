"""band_adapter.py  – maps GPT‑4o raw scores to o3 ∈ [‑1,1] scale"""
import numpy as np

# Calibrated mins/maxes from 1000‑sample baseline (can refine via EWMA later)
CALIB = {
    "valence":   {"min": -0.35, "max": 0.40},  # VADER compound
    "arousal":   {"min":  0.10, "max": 0.85},  # Normed energy score
    "dominance": {"min": -0.30, "max": 0.50},  # Imperative density z‑score
    "curiosity": {"min":  0.05, "max": 0.60},  # Q‑ratio
    "selfref":   {"min":  0.00, "max": 0.75},  # 1st‑person freq
}

def linear_scale(x, lo, hi):
    if hi == lo:
        return 0.0
    y = (x - lo) / (hi - lo)
    return float(np.clip(y * 2 - 1, -1, 1))  # map to [‑1,1]

def adapt(raw: dict[str, float]) -> dict[str, float]:
    """Convert raw 4o band dict to o3‑scaled bands."""
    return {k: linear_scale(raw[k], CALIB[k]["min"], CALIB[k]["max"]) for k in CALIB}

if __name__ == "__main__":
    demo = {"valence": 0.1, "arousal": 0.3, "dominance": 0.05,
            "curiosity": 0.4, "selfref": 0.2}
    print(adapt(demo))