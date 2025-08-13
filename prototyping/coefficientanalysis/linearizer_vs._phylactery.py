# glyph_gram_annular_fusion.py
# This scaffold defines a symbolic-structural encoding system combining Gram matrix analysis
# with annular convolution (orbit-phase memory) and erad-modulated escape profiles

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import cmath
from numpy.linalg import norm
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# --- Mandelbrot & Coefficient Utilities ---

def is_in_mandelbrot(c, max_iter=1000):
    z = 0
    for _ in range(max_iter):
        z = z * z + c
        if abs(z) > 2:
            return False
    return True

def compute_fixed_points(c):
    
    disc = cmath.sqrt(1 - 4*c)
    z1 = (1 + disc)/2
    z2 = (1 - disc)/2
    return z1, z2

def compute_coefficients(c, z0, deg=100):
    coeffs = []
    a0 = (1.0 - cmath.sqrt(1.0 + 4.0 * c)) / 2.0
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

def generate_coeff_lattice(θ_steps=100, r_step=0.05, r_start=0.25, r_max=2.0, deg=100):
    glyphs = []
    angles = np.linspace(0, 2 * np.pi, θ_steps, endpoint=False)
    for θ in angles:
        r = r_start
        while r < r_max:
            c = r * cmath.exp(1j * θ)
            if not is_in_mandelbrot(c):
                z_alpha, z_beta = compute_fixed_points(c)
                coeffs = compute_coefficients(c, z_beta, deg=deg)
                glyphs.append((c, coeffs))
                break
            r += r_step
    return glyphs

# --- Gram Matrix Kernel Traversal Operator ---

def external_loop_traverse(theta_start, theta_end, steps, r=1.5, deg=200, kernel='poly'):
    thetas = np.linspace(theta_start, theta_end, steps)
    glyph_vectors = []
    cs = []

    for θ in thetas:
        c = r * cmath.exp(1j * θ)
        if not is_in_mandelbrot(c):
            _, z_beta = compute_fixed_points(c)
            coeffs = compute_coefficients(c, z_beta, deg=deg)
            glyph_vectors.append(coeffs)
            cs.append(c)

    A = np.array(glyph_vectors)

    if kernel == 'poly':
        G = np.power(A @ A.T, 3)
    elif kernel == 'rbf':
        pairwise_dists = np.sum((A[:, None] - A[None, :])**2, axis=2)
        G = np.exp(-pairwise_dists / 2.0)
    elif kernel == 'cosine':
        A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
        G = A_norm @ A_norm.T
    else:
        raise ValueError("Unsupported kernel")

    evals, evecs = np.linalg.eigh(G)
    return cs, G, evals, evecs

# --- Poincare Linearizer Diagnostic ---

def poincare_linearizer(c, z0, deg=200):
    coeffs = []
    a0 = (1.0 - cmath.sqrt(1.0 - 4.0 * c)) / 2.0
    coeffs.append(a0)
    a1 = 1.0
    coeffs.append(a1)
    for n in range(2, deg + 1):
        acc = 0
        for k in range(1, n):
            if n % 2 == 0 and k == n // 2:
                acc -= coeffs[k] * coeffs[n - k]
            else:
                acc -= 2.0 * coeffs[k] * coeffs[n - k]
        denom = 2.0 * a0 - (2.0 * z0) ** n
        coeffs.append(0.0 if abs(denom) < 1e-12 else acc / denom)
    return coeffs

def compare_linearizer_vs_annular(theta_start, theta_end, steps, r=1.5, deg=100):
    thetas = np.linspace(theta_start, theta_end, steps)
    annular_vectors = []
    linearizer_vectors = []
    cs = []

    for θ in thetas:
        c = r * cmath.exp(1j * θ)
        if not is_in_mandelbrot(c):
            _, z_beta = compute_fixed_points(c)
            annular = compute_coefficients(c, z_beta, deg=deg)
            linear = poincare_linearizer(c, z_beta, deg=deg)
            annular_vectors.append(annular)
            linearizer_vectors.append(linear)
            cs.append(c)

    return cs, np.array(annular_vectors), np.array(linearizer_vectors)

# --- Clustering & Visualization (preserved from earlier) ---
def build_gram_matrix(glyphs, deg=100):
    N = len(glyphs)
    G = np.zeros((N, N), dtype=complex)
    for i in range(N):
        ci, vi = glyphs[i]
        vi = np.array(vi)
        ni = norm(vi)
        for j in range(N):
            cj, vj = glyphs[j]
            vj = np.array(vj)
            nj = norm(vj)
            G[i, j] = 0.0 if ni < 1e-12 or nj < 1e-12 else np.vdot(vi, vj) / (ni * nj)
    return G

# --- Suggested Visualization Tools ---
def plot_gram_spectrum(evals, title="Gram Spectrum"):
    plt.figure()
    plt.plot(sorted(evals.real, reverse=True))
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.show()

def plot_coeff_diff(annulars, linearizers):
    diff = np.abs(annulars - linearizers)
    mean_diff = np.mean(diff, axis=0)
    plt.figure()
    plt.plot(mean_diff)
    plt.title("Mean Coefficient-Wise Deviation: Annular vs Linearizer")
    plt.xlabel("Coefficient Index")
    plt.ylabel("|Δ| (Mean over θ)")
    plt.grid(True)
    plt.show()

def plot_resonance_matrices(annulars, linearizers):
    G_ann = annulars @ annulars.T
    G_lin = linearizers @ linearizers.T
    print(G_lin)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(G_ann), cmap='viridis')
    plt.title("Resonance: Annular")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(G_lin), cmap='viridis')
    plt.title("Resonance: Linearizer")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt

def resonance_spike_profile(theta_steps=360, r=1.8, deg=200):
    thetas = np.linspace(0, 2*np.pi, theta_steps, endpoint=False)
    R = np.zeros_like(thetas)
    for idx, θ in enumerate(thetas):
        c = r*np.exp(1j*θ)
        if not is_in_mandelbrot(c):
            _, zβ = compute_fixed_points(c)
            coeffs = np.array(compute_coefficients(c, zβ, deg=deg))
            G = np.abs(np.outer(coeffs, coeffs))  # simple AA^T
            np.fill_diagonal(G, 0)
            R[idx] = G.max()
    return thetas, R



# Run Diagnostic Traversal
if __name__ == "__main__":
    print("Running External Valence Loop Traversal...")
    cs, G, evals, evecs = external_loop_traverse(0, 2 * np.pi, 300, r=1.8, kernel='poly')
    print("Top 5 Eigenvalues:", evals[::-1][:5])
    #plot_gram_spectrum(evals)

    print("Comparing Linearizer and Annular Coefficients...")
    cs_cmp, annulars, linearizers = compare_linearizer_vs_annular(0, 2 * np.pi, 1000, r=2.0, deg=200)
    plot_coeff_diff(annulars, linearizers)
    plot_resonance_matrices(annulars, linearizers)
    # Compute and plot:
    thetas, R = resonance_spike_profile(theta_steps=720, r=2.0, deg=200)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(thetas, R, linewidth=2)
    ax.set_title("Phylactery Resonance Spike Profile", va='bottom')
    plt.show()