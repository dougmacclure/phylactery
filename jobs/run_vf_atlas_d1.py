#!/usr/bin/env python
"""
Vector-flow atlas D1
Maps dominant-eigenvector angle for a coarse c-grid.
Outputs numpy tiles + a quick PNG preview.
"""

import numpy as np, cmath, os, pathlib, imageio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gr_erg.phylactery_core.coeffs import gen_coeffs
from gr_erg.phylactery_core.pd_eval import P_ext
from gr_erg.phylactery_core.metrics import dominant_eig_angle

# --- grid spec -------------------------------------------------------------
REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX = -3.5,  3.5,  -3.5, 3.5
N  = 401                # 401×401 ≈ 160 k points (quick)
DEG = 40                # coefficients depth
OUT_DIR = pathlib.Path("data/vf_atlas_d1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- main ------------------------------------------------------------------
angles = np.full((N, N), np.nan, dtype=np.float32)
real_vals = np.linspace(REAL_MIN, REAL_MAX, N)
imag_vals = np.linspace(IMAG_MIN, IMAG_MAX, N)
def compute_fixed_points(c):
    a = 1
    b = -1
    d = c
    disc = cmath.sqrt(b**2 - 4 * a * d)
    z1 = (-b + disc) / (2 * a)
    z2 = (-b - disc) / (2 * a)
    return z1, z2

def compute_coefficients(c, a0, z0, deg=100):
    coeffs = []
    
    if a0 == z0:
        return None
    coeffs.append(a0)
    a1 = -c / (2.0 * a0 - (2.0 * z0) ** 1)
    coeffs.append(a1)

    for n in range(2, deg + 1):
        acc = -c
        for k in range(1, n):
            if n % 2 == 0 and k == n // 2:
                acc -= coeffs[k] * coeffs[n - k]
            else:
                acc -= 2.0 * coeffs[k] * coeffs[n - k]
        denom = 2.0 * a0 - (2.0 * z0) ** n
        coeffs.append(0.0 if abs(denom) < 1e-12 else acc / denom)
    return coeffs


for iy, Im in enumerate(imag_vals):
    for ix, Re in enumerate(real_vals):
        c = Re + 1j*Im
        z1, z2 = (0.5*(1 + sgn*cmath.sqrt(1-4*c)) for sgn in (1,-1))  # fixed pts
        z0 = max((z1, z2), key=lambda z: abs(2*z))              # dominant
        coeffs = compute_coefficients(c, a0=z0, z0=z0, deg=DEG)
        if coeffs is None:
            continue
        try:
            angle = dominant_eig_angle(coeffs[:DEG])
            angles[iy, ix] = angle
        except Exception:
            pass

# save npy + png preview
np.save(OUT_DIR / "vf_angles.npy", angles)
norm = (angles + np.pi) / (2*np.pi)  # map to [0,1] for hue
rgb  = np.stack([norm, np.ones_like(norm), np.ones_like(norm)], axis=-1)
rgb  = np.uint8(np.nan_to_num(rgb) * 255)
imageio.imwrite(OUT_DIR / "vf_preview.png", rgb)

print("Atlas tile saved to", OUT_DIR)
