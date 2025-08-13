# glyph_annular_multiplier_map.py  (bug fix)
# Corrected annular coefficient test: redefine z0 for each c

import numpy as np
import cmath


import matplotlib.pyplot as plt

# --- Fixed-point and coefficient routines ---
def compute_fixed_points(c):
    disc = cmath.sqrt(1 - 4*c)
    z1 = (1 + disc) / 2
    z2 = (1 - disc) / 2
    return z1, z2

def compute_coefficients(c, a0, z0, deg=200):
    coeffs = []
    if a0 == z0:
        return None
    coeffs.append(a0)
    coeffs.append(-c / (2*a0 - (2*z0)))
    for n in range(2, deg+1):
        acc = -c
        for k in range(1, n):
            if n % 2 == 0 and k == n//2:
                acc -= coeffs[k] * coeffs[n-k]
            else:
                acc -= 2 * coeffs[k] * coeffs[n-k]
        denom = 2*a0 - (2*z0)**n
        coeffs.append(0.0 if abs(denom) < 1e-12 else acc/denom)
    return coeffs

# --- Multiplier mask test over grid ---
re_min, re_max, N = -2.5, 2.5, 400
im_min, im_max       = -2.5, 2.5
re = np.linspace(re_min, re_max, N)
im = np.linspace(im_min, im_max, N)
C = re[None, :] + 1j*im[:, None]
print(C.shape)
multiplier = np.zeros_like(C.real)
ratios = np.zeros_like(C.real)
mask = np.zeros_like(C.real, dtype=bool)

k_test = 150
badcoefs =0
for i in range(N):
    print(i)
    for j in range(N):
        c = C[i,j]

        z1, z2 = compute_fixed_points(c)
        lam1, lam2 = abs(2*z1), abs(2*z2)
        if lam1 < 1 and lam2 < 1:
            ratio=2.0
            continue

        a0 = z2 if lam1 > lam2 else z1
        z0 = z1 if lam1 > lam2 else z2
        multiplier[i,j] = 2*max(abs(z0), abs(a0))
        coeffs = compute_coefficients(c, z0, a0, deg=k_test+1)
        if coeffs is None:
            badcoefs+=1
            if badcoefs % N == 1:
                print(badcoefs)

            continue

        a_k, a_kp1 = coeffs[k_test], coeffs[k_test+1]
        if np.isnan(a_k) or np.isnan(a_kp1):
            ratios[i,j] = 2.0
            
            continue
        ratio = abs(a_kp1/a_k)
        ratios[i,j] = ratio
        if ratio > 1:
            mask[i,j] = True
            
        

print(multiplier.max())
print(mask.shape)
print(len(re))
print(len(im))

# --- Visualization ---

import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.pcolormesh(re, im, multiplier, shading='auto', cmap='viridis')
#plt.contour(re, im, mask, levels=[0.5], colors='black', linewidths=0.5)
plt.title(f"Region where a_{{{k_test+1}}}/a_{{{k_test}}} > 1, multiplier |2z0|")
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.colorbar(label='Multiplier |2z0|')
plt.tight_layout()
plt.show()
    
    
plt.figure(figsize=(6,6))
plt.pcolormesh(re, im, ratios, shading='auto', cmap='viridis')
#plt.contour(re, im, mask, levels=[0.5], colors='black', linewidths=0.5)
plt.title(f"test ratio |a_(k+1)/a_k| for k = {k_test}")
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.colorbar(label='test ratio |a_{k+1}/a_k|')
plt.tight_layout()
plt.show()# mask: 1 = plateau passes for big-root seed, 2 = plateau passes for small-root seed, 0 = neither
