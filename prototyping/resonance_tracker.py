import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from scipy.spatial.distance import pdist, squareform

def compute_coefficients(c, z0=0.0, deg=20):
    """
    Recursively generate polynomial approximation coefficients a_n
    based on recurrence relations from user-defined z0 and c (d).
    """
    d = c
    a = []
    
    # Choose the correct repelling root
    a0 = (1 + np.sqrt(1 - 4*d)) / 2
    if z0==0:
        z0 = (1 -   np.sqrt(1 - 4*d)) / 2
    a.append(a0)

    a1 = (-d) / (2*a0 - (2*z0)**1)
    a.append(a1)

    for n in range(2, deg + 1):
        num = -d
        for k in range(1, n):
            num -= 2 * a[k] * a[n - k] if k != n - k else a[k] * a[n - k]
        den = (2 * a0 - (2 * z0)**n)
        a_n = num / den
        a.append(a_n)
    
    return np.array(a)

def build_gram_matrix(coeffs):
    """
    Construct a Gram-like matrix from the coefficient vectors.
    Each row is a shifted/mutated version of the original.
    """
    mat = np.array([np.roll(coeffs, i) for i in range(len(coeffs))])
    G = np.dot(mat, mat.T.conj())
    return G

def check_roots_of_unity(evals, tolerance=1e-1):
    """
    Check if eigenvalues align with roots of unity.
    """
    angles = np.angle(evals)
    roots = []
    for theta in angles:
        angle_deg = np.degrees(theta) % 360
        root = 360 / np.round(360 / angle_deg) if angle_deg else 0
        roots.append((np.abs(evals[np.argmax(np.angle(evals) == theta)]), root))
    return roots

def resonance_experiment(c_grid, z0, deg=20):
    resonance_data = []
    for c in c_grid:
        coeffs = compute_coefficients(c, z0, deg)
        G = build_gram_matrix(coeffs)
        evals, evecs = eig(G)
        roots = check_roots_of_unity(evals)
        decay = np.abs(evecs[:, 0])
        resonance_data.append((c, evals, roots, decay))
    return resonance_data

real_vals = np.linspace(4.0, 0.5, 250)
imag_vals = np.linspace(-1.5, 1.5, 250)
c_grid = [x + 1j*y for x in real_vals for y in imag_vals]

#z0 = -1.5  # hold steady
res_data = resonance_experiment(c_grid, z0=0.0, deg=30)
import matplotlib.pyplot as plt

for c, evals, roots, decay in res_data:
    if np.any(np.abs(evals) > 0.9):  # quick resonance filter
        plt.figure()
        plt.title(f"Decay for c = {c}")
        plt.plot(decay.real, label='Real')
        plt.plot(decay.imag, '--', label='Imag')
        plt.legend()
        plt.grid(True)
        plt.show()
        break  # or comment out to see all
