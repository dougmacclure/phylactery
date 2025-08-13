#!/usr/bin/env python3
"""
Residue-loop test around c = 1/4
--------------------------------
Tracks (a1, a2) coefficients of the Phylactery map while circling
c = 0.25 with radius eps = 5e-3.  Saves a CSV and a polar plot.
"""

import numpy as np
import mpmath as mp
import csv, matplotlib.pyplot as plt
from pathlib import Path

# ---------- parameters ----------
eps   = 0.0001
steps = 720                 # 0.5° per step
out   = Path("data/coeff_loops")
out.mkdir(parents=True, exist_ok=True)
csv_path = out / "loop_n1.csv"
png_path = out / "loop_n1_polar.png"
# ---------------------------------



def a2(c): return (-c - a1(c)**2) / (2*a0(c) - (2*z0(c))**2)
# --- add just above the sweep loop --------------------------
def sqrt_continuous(z, prev):
    """
    Return a square root of z that is C¹-close to the previous value `prev`.
    On the first call pass prev=None to take the principal value.
    """
    r = mp.sqrt(z)            # principal branch
    if prev is None:
        return r
    return r if abs(r - prev) < abs(-r - prev) else -r
# ------------------------------------------------------------
rows, a1_vals = [], []
angles = []
prev_s = None                          # running square-root
theta  = np.linspace(0, 2*np.pi, steps, endpoint=False)
for t in theta:
    c = 0.25 + eps * mp.e**(1j*t)
    s  = sqrt_continuous(1 - 4*c, prev_s)   # <- branch-tracked √
    prev_s = s

    a0 = (1 + s) / 2
    z0 = (1 - s) / 2
    a1c = -c / (2*a0 - (2*z0))          # recursion as before
    a2c = (-c - a1c**2) / (2*a0 - (2*z0**2))
    angles.append(mp.arg(a1c))
    rows.append((t, c.real, c.imag, a1c, a2c))
    a1_vals.append(a1c)

c_vals = 0.25 + eps*np.exp(1j*theta)

angles_unw = np.unwrap(np.array([float(a) for a in angles]))
print("Δ arg(a1) =", angles_unw[-1] - angles_unw[0])

print('max(angles_unw)=',max(angles_unw),'min(angles_unw=)' ,min(angles_unw))
print('angles_unw[719]=',angles_unw[719],'angles_unw[0]=' ,angles_unw[0])
print("Δ arg unwrapped =", angles_unw[719]-angles_unw[0])  # same magnitude

angles = np.unwrap([mp.arg(x) for x in a1_vals])

# CSV
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["theta", "Re(c)", "Im(c)", "a1", "a2"])
    w.writerows(rows)

# quick polar plot
plt.figure(figsize=(5,5))
#plt.plot(theta, [mp.arg(x) for x in a1_vals], ".-")
plt.plot(theta, angles, ".-")
plt.ylabel("unwrapped arg(a1)")

plt.xlabel("loop parameter θ"); plt.ylabel("arg(a1)")
plt.title("Phase gain of a₁ around c = ¼"); plt.tight_layout()
plt.savefig(png_path, dpi=150)
print(f"Wrote {csv_path} and {png_path}")
import numpy as np, mpmath as mp
