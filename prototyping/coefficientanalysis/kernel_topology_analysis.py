# glyph_gram_annular_fusion.py (modified for PaCMAP on KPCA output with NaN/Inf handling)

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import cmath
from numpy.linalg import norm
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.decomposition import KernelPCA
import pacmap
from sklearn.impute import SimpleImputer

# --- Mandelbrot and Coefficient Functions ---

def is_in_mandelbrot(c, max_iter=1000):
    z = 0
    for _ in range(max_iter):
        z = z * z + c
        if abs(z) > 2:
            return False
    return True

def compute_fixed_points(c):
    a = 1
    b = 2 * c - 1
    d = c**2
    disc = cmath.sqrt(b**2 - 4 * a * d)
    z1 = (-b + disc) / (2 * a)
    z2 = (-b - disc) / (2 * a)
    return z1, z2

def compute_coefficients(c, z0, deg=100):
    coeffs = []
    a0 = (1.0 - cmath.sqrt(1.0 - 4.0 * c)) / 2.0
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

def generate_coeff_lattice(θ_steps=1000, r_step=0.05, r_start=0.25, r_max=2.0, deg=100):
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

def construct_annular_embedding(glyphs, erad_fn=lambda c: 1.0, num_annuli=20):
    N = len(glyphs)
    deg = len(glyphs[0][1])
    A_coeff = np.zeros((N, deg))
    A_annular = np.zeros((N, num_annuli))
    A_erad = np.zeros((N, 1))

    for i, (c, coeffs) in enumerate(glyphs):
        A_coeff[i, :] = np.real(coeffs[:deg])
        radius = abs(c)
        bin_idx = int(min(num_annuli - 1, (radius / 2.0) * num_annuli))
        A_annular[i, bin_idx] = 1.0
        A_erad[i, 0] = erad_fn(c)

    A_annular /= np.maximum(A_annular.sum(axis=1, keepdims=True), 1e-6)
    return np.hstack([A_coeff, A_annular, A_erad])

def cluster_glyphs(G, num_clusters=5):
    G_sym = np.abs(0.5 * (G + G.T))
    G_norm = normalize(G_sym.real)
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )
    return clustering.fit_predict(np.abs(G_norm))

def plot_pacmap_on_kpca(X_kpca, labels):
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
    X_embedded = embedding.fit_transform(X_kpca)
    plt.figure(figsize=(10, 6))
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], label=f"Cluster {label}", alpha=0.7)
    plt.title("PaCMAP Projection on Kernel PCA Output")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()
def external_loop_traverse(theta_start, theta_end, steps, r=1.5, deg=100, kernel='poly'):
    """
    Traverse around ∂M in angle space, collecting coefficient vectors.
    """
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

    # Kernelized Gram
    if kernel == 'poly':
        G = A @ A.T
        G = np.power(G, 3)  # or another degree if needed

    elif kernel == 'rbf':
        pairwise_dists = np.sum((A[:, None] - A[None, :])**2, axis=2)
        G = np.exp(-pairwise_dists / 2.0)

    elif kernel == 'cosine':
        A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
        G = A_norm @ A_norm.T

    # Eigenvalue projection
    evals, evecs = np.linalg.eigh(G)

    return cs, G, evals, evecs

if __name__ == "__main__":
    glyphs = generate_coeff_lattice()
    A_aug = construct_annular_embedding(glyphs)

    # Handle NaNs and Infs
    A_aug = np.nan_to_num(A_aug, nan=0.0, posinf=1e6, neginf=-1e6)
    A_aug = np.clip(A_aug, -1e10, 1e10)  # Clip extreme values to avoid overflow
    A_aug = np.where(np.isinf(A_aug), 0, A_aug)  # Replace infinities with 0
    imputer = SimpleImputer(strategy='mean')
    A_aug = imputer.fit_transform(A_aug)

    kpca = KernelPCA(n_components=10, kernel='poly', degree=3)
    X_kpca = kpca.fit_transform(A_aug)

    G_aug = A_aug @ A_aug.T
    labels = cluster_glyphs(G_aug)
    plot_pacmap_on_kpca(X_kpca, labels)
