import numpy as np
import numba as nb
import matplotlib.pyplot as plt

@nb.njit
def fixed_points(c):
    """Return the two fixed points of z²+c and the larger |2z0| multiplier."""
    disc = np.sqrt(1 - 4*c)           # branch via principal sqrt
    z1 = (1 + disc)/2
    z2 = (1 - disc)/2
    return z1, z2, max(abs(2*z1), abs(2*z2))

@nb.njit
def coeffs_up_to_k(c, a0,z0, k):
    """a₀..a_k for the phylactery series (no caching, JIT loops)."""
    
    arr = np.empty(k+1, dtype=np.complex128)
    arr[0] = a0
    for n in range(1, k+1):
        num = -c - np.sum(arr[1:n] * arr[n-1:0:-1])   # convolution sum
        denom = 2*a0 - (2*z0)**n
        arr[n] = num / denom
    return arr

@nb.njit
def leading_ratio_band(c, k0=120, k1=160, tol=1e-2):
    """Return (R₁, e_mid) or (-1,-1) if tail not flat."""
    z1,z2,mul = fixed_points(c)
    if z1 == z2:
        return 0,0
    z0 = z1 if abs(2*z1) > abs(2*z2) else z2
    a0 = z2 if abs(2*z1) > abs(2*z2) else z1
    a = coeffs_up_to_k(c, a0, z0, k1+1)

    # ratio plateau test
    r0 = abs(a[k0+1]/a[k0])
    r1 = abs(a[k1+1]/a[k1])
    if np.abs(r1 - r0) > tol:          # tail not yet stable
        return -1.0, -1.0

    # Perron–Frobenius approx: use ratio at k0
    e_mid = r0 / abs(2*z0)
    return abs(a[1]/a[0]), e_mid       # R₁, stable multiplier

# --- parameter box
res   = 350        # 350×350 ~ 1-2 min on laptop
re_lo,re_hi = -2.5, 2.5
im_lo,im_hi = -2.5, 2.5

grid_R1   = np.full((res,res), np.nan)
grid_band = np.full((res,res), np.nan)

for ix,x in enumerate(np.linspace(re_lo,re_hi,res)):
    for iy,y in enumerate(np.linspace(im_lo,im_hi,res)):
        c = x + 1j*y
        R1, e = leading_ratio_band(c)
        if e>0:
            grid_R1[iy,ix]   = R1
            grid_band[iy,ix] = e             # or 1/e for allowed erad range

# --- quick heat-map
fig,ax = plt.subplots(figsize=(6,6))
im = ax.imshow(grid_band, extent=[re_lo,re_hi,im_lo,im_hi],
               origin='lower', cmap='viridis', vmin=0, vmax=2)
ax.set_xlabel("Re(c)"); ax.set_ylabel("Im(c)")
ax.set_title("erad_band mid-point  (plateaued spectra)")
fig.colorbar(im,label="e_mid = ρ(Cₖ)/|2 z₀|")
plt.tight_layout(); plt.show()
