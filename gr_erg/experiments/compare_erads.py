# -*- coding: utf-8 -*-
import numpy as np, mpmath as mp, scipy.linalg as la
from PIL import Image
from phylactery_core.pd_eval import gen_coeffs
from legacy.phylactery_modern import generate_bifurcation_image

W, H = 1200, 1200        # same res for all modes
c = -3.75 + 0j
erad = 3.182            # pick your favourite “good” radius
lim  = 1e4
mp.mp.dps = 80
# ---------- helpers ----------
def companion(a):
    k = len(a)
    C = np.zeros((k,k), dtype=np.complex128)
    C[1:,:-1] = np.eye(k-1)
    C[0,:k-1] = -np.asarray(a[1:], np.complex128)/a[0]
    return C

def modern_double(z):
    """pure-python double-precision evaluation"""
    coeffs, z0 = gen_coeffs(c, N=30)
    #   annular pull-back (modulus only) + series
    #   === your pd_eval.P_ext condensed ===
    j   = int(max(0, np.floor(np.log(abs(z)/erad)/np.log(abs(2*z0)))))
    w   = z / (abs(2*z0)**(j+1))
    s   = coeffs[-1]
    for a in reversed(coeffs[:-1]):
        s = s*w + a
    for _ in range(j):
        s = s**2 + c
    return s


def high_prec(z):
    """mpmath 80-bit equivalent"""
    z_mp = mp.mpc(z.real, z.imag)
    coeffs, z0 = gen_coeffs(c, N=30)
    z0 = complex(z0)                     # we only need |2 z0|
    j   = int(mp.floor(mp.log(abs(z_mp)/erad)/mp.log(abs(2*z0))))
    w   = z_mp / (abs(2*z0)**(j+1))
    s   = mp.mpc(coeffs[-1])
    for a in reversed(coeffs[:-1]):
        s = s*w + a
    for _ in range(j):
        s = s**2 + c
    return complex(s)

# ---------- render ----------
def buddha(eval_fn, tag):
    img = np.zeros((H,W), np.float64)
    xs  = np.linspace(-erad, erad, W)
    ys  = np.linspace(-erad, erad, H)
    for iy,y in enumerate(ys):
        for ix,x in enumerate(xs):
            z = x + 1j*y
            z = eval_fn(z)
            img[iy,ix] = abs(z)
    img = np.log(img + 1e-12)            # compress dynamic range
    # min-max → 0…255
    img -= img.min();  img *= 255/img.max()
    Image.fromarray(img.astype(np.uint8)
        ).save(f"forward_{tag}.png")
    return img


# ---------- difference heat-maps ----------
def diff(a,b,name):
    d = np.abs(a.astype(float)-b.astype(float))
    d -= d.min(); d *= 255/d.max()
    Image.fromarray(d.astype(np.uint8)
        ).save(f"diff_{name}.png")
    
def main():

    modern  = buddha(modern_double, "modern")
    mp80    = buddha(high_prec,      "mp80")

    # legacy png already produced outside:
    legacy  = np.asarray(Image.open("/legacy/forward_x87.png"), np.uint8)
    diff(legacy, modern, "legacy_vs_modern")
    diff(legacy, mp80,   "legacy_vs_mp80")
    diff(modern, mp80,   "modern_vs_mp80")
    print("Images saved in experiments/ .  Open the three *diff_*.png files")

if __name__=='__main__':
    main()