"""band_conflict.py – detect disagreement beyond τ between o3 & 4o"""
from typing import Dict

TAU = 0.3  # tolerance before flag (post‑scaling units)
BANDS = ["valence", "arousal", "dominance", "curiosity", "selfref"]

def band_deltas(o3: Dict[str, float], o4: Dict[str, float]) -> Dict[str, float]:
    return {k: abs(o3[k] - o4[k]) for k in BANDS}

def conflicts(o3: Dict[str, float], o4: Dict[str, float], tau: float = TAU):
    """Yield bands where |o3 - o4| > tau."""
    return [k for k, d in band_deltas(o3, o4).items() if d > tau]

if __name__ == "__main__":
    o3_demo = {"valence": 0.7, "arousal": 0.1, "dominance": 0.0,
               "curiosity": 0.3, "selfref": 0.1}
    o4_demo = {"valence": -0.4, "arousal": 0.2, "dominance": 0.05,
               "curiosity": 0.35, "selfref": 0.1}
    print(conflicts(o3_demo, o4_demo))  # → ['valence']