# atlas_builder_fast.py  – vectorised & cached   runtime: ~2 min @ 600²
import numpy as np, argparse
from PIL import Image
from functools import lru_cache
from phylactery_core.coeffs import gen_coeffs
from phylactery_core.metrics import rgb_mod
from phylactery_core.pd_eval import P_ext

# ---------------- CLI ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--bounds", nargs=4, type=float, default=[-1,1,-1,1])
ap.add_argument("--grid", type=int, default=600)
ap.add_argument("--max_iter", type=int, default=800)
ap.add_argument("--escape", type=float, default=1e3)
ap.add_argument("--out_png", default="atlas_vec.png")
cfg = ap.parse_args()
xmin,xmax,ymin,ymax = cfg.bounds
N = cfg.grid

# -------------- grid -----------------
xs = np.linspace(xmin, xmax, N)
ys = np.linspace(ymin, ymax, N)
X, Y = np.meshgrid(xs, ys)
D = X + 1j*Y                 # (N,N) complex parameters
I = np.zeros((N,N), dtype=np.int16)
Z = np.zeros_like(D)
active = np.ones_like(D, bool)

# -------- coeffs cache --------------
@lru_cache(maxsize=50_000)
def coeff_blob(d: complex):
    coeffs, z0 = gen_coeffs(d, 40)
    return coeffs, z0
# ------------------------------------

print(f"[Atlas] {N}×{N}  max_iter={cfg.max_iter}")
for i in range(1, cfg.max_iter+1):
    iy, ix = np.where(active)
    if iy.size == 0:
        break
    for y,x in zip(iy, ix):
        d = D[y, x]
        coeffs, z0 = coeff_blob(d)
        try:
            Z[y, x] = P_ext(Z[y, x], d, coeffs=coeffs, z0=z0)
        except Exception:
            I[y, x] = cfg.max_iter
            active[y, x] = False
            continue
    diff = np.abs(Z)
    esc = i * diff > cfg.escape
    hit = esc & active
    I[hit] = i
    active[hit] = False
    if i % 50 == 0:
        print(f" it {i:<4} remaining {active.sum():>6}")

I[active] = cfg.max_iter
# ------------- render ----------------
img = Image.new("RGB", (N,N))
pix = img.load()
for y in range(N):
    for x in range(N):
        pix[x,y] = rgb_mod(int(I[y,x]), cfg.max_iter)
img.save(cfg.out_png)
print(f"[Atlas] saved → {cfg.out_png}")
