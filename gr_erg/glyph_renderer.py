import numpy as np
from PIL import Image
from phylactery_core.coeffs import gen_coeffs
from phylactery_core.pd_eval import P_ext
from phylactery_core.metrics import rgb_mod

# ---- parameters ----
d          = -3.75 + 0j          # your seed
N          = 1024                # pixel resolution
max_iter   = 750
escape_cond= 10.0
erad       = 3.95

# pre-compute coeffs once
coeffs, z0 = gen_coeffs(d, 70)

# grid over local z-plane  (|z| ≤ 1.5 shows plenty of detail)
xs = np.linspace(-5, 5, N)
ys = np.linspace(-5, 5, N)
X, Y = np.meshgrid(xs, ys)
Z    = X + 1j*Y
I    = np.zeros_like(X, dtype=np.int16)

for iy in range(N):
    for ix in range(N):
        z = Z[iy, ix]
        for k in range(1, max_iter+1):
            z_old = z
            z     = P_ext(z, d, coeffs=coeffs, z0=z0)
            if k*abs(z - z_old) + abs(z) > escape_cond:
                I[iy, ix] = k
                break
        else:
            I[iy, ix] = max_iter

# ----- render -----
img = Image.new("RGB", (N,N))
px  = img.load()
for iy in range(N):
    for ix in range(N):
        px[ix,iy] = rgb_mod(int(I[iy,ix]), max_iter)
img.save("phylactery_d_-3_75.png")
print("Saved → phylactery_d_-3_75.png")
img.show()