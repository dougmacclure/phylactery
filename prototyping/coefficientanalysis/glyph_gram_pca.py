# glyph_gram_annular_fusion.py
# This scaffold defines a symbolic-structural encoding system combining Gram matrix analysis
# with annular convolution (orbit-phase memory) and erad-modulated escape profiles

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import cmath
from numpy.linalg import norm
from scipy.ndimage import gaussian_filter, sobel
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

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
    b = -1
    d = c
    disc = cmath.sqrt(b**2 - 4 * a * d)
    z1 = (-b + disc) / (2 * a)
    z2 = (-b - disc) / (2 * a)
    return z1, z2

def compute_coefficients(c, z0, deg=100):
    coeffs = []
    a0 = (1.0 + cmath.sqrt(1.0 - 4.0 * c)) / 2.0
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

def build_gram_matrix(glyphs, deg=100):
    N = len(glyphs)
    G = np.zeros((N, N), dtype=complex)
    for i in range(N):
        _, vi = glyphs[i]
        vi = np.array(vi)
        ni = norm(vi)
        for j in range(N):
            _, vj = glyphs[j]
            vj = np.array(vj)
            nj = norm(vj)
            G[i, j] = 0.0 if ni < 1e-12 or nj < 1e-12 else np.vdot(vi, vj) / (ni * nj)
    return G

# --- Annular Embedding + Erad Fusion ---

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

# --- Spectral Clustering and PCA Visualization ---

def plot_pca_with_clusters(G, labels):
    coords = PCA(n_components=2).fit_transform(np.real(G))
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(coords[mask, 0], coords[mask, 1], label=f"Cluster {label}", alpha=0.6)
    plt.title("PCA Projection with Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.show()

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

def visualize_resonance_field1(G, title="Resonance Field (|G|)", recurrence_threshold=0.9):
    log_G = np.log(np.abs(G) + 1e-10)
    magnitude = np.abs(log_G)
    plt.figure(figsize=(10, 8))
    plt.imshow(magnitude, cmap='cividis', interpolation='nearest', vmin=0, vmax=np.percentile(magnitude, 99.9))
    plt.colorbar(label='Resonance Magnitude')
    plt.title(title)
    plt.xlabel("j (glyph index)")
    plt.ylabel("i (glyph index)")
    recur_i, recur_j = np.where(magnitude > recurrence_threshold)
    if recur_i.size > 0:
        plt.scatter(recur_j, recur_i, color='white', edgecolor='black', s=20, alpha=0.9, label=f'Recurrence (|Gᵢⱼ| > {recurrence_threshold})')
        plt.legend()
    Y, X = np.mgrid[0:magnitude.shape[0], 0:magnitude.shape[1]]
    dy, dx = np.gradient(magnitude)
    #plt.quiver(X, Y, dx, dy, color='red', alpha=0.3, scale=50)
    #plt.tight_layout()
    plt.show()

from sklearn.decomposition import KernelPCA

from sklearn.decomposition import KernelPCA
from sklearn.impute import SimpleImputer

def kernel_pca_with_clusters(G, labels, degree=3):
    # Convert to real-valued matrix
    G_real = np.real(G)

    # Step 1: Replace infs and NaNs with finite values
    G_real[np.isnan(G_real)] = 0.0
    G_real[np.isinf(G_real)] = np.sign(G_real[np.isinf(G_real)]) * 1e10  # Large but finite

    # Step 2: Optional normalization (can stabilize kernel space)
    G_real = np.clip(G_real, -1e10, 1e10)

    # Step 3: Impute if necessary (safe fallback)
    imputer = SimpleImputer(strategy='mean')
    G_clean = imputer.fit_transform(G_real)

    # Step 4: Kernel PCA
    kpca = KernelPCA(n_components=2, kernel='poly', degree=degree, coef0=1)
    coords = kpca.fit_transform(G_clean)

    # Step 5: Plot
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(coords[mask, 0], coords[mask, 1], label=f"Cluster {label}", alpha=0.6)
    plt.title(f"Polynomial Kernel PCA (degree={degree}) with Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_resonance_field(G, title="Resonance Field (|G|)"):
    magnitude = np.abs(G)
    plt.figure(figsize=(10, 8))
    plt.imshow(magnitude, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Resonance Magnitude')
    plt.title(title)
    plt.xlabel("j (glyph index)")
    plt.ylabel("i (glyph index)")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
# --- Execution ---
if __name__ == "__main__":
    glyphs = generate_coeff_lattice()
    A_aug = construct_annular_embedding(glyphs)
    G_aug = A_aug @ A_aug.T
    G_diff = expm(G_aug)
    
    evals, evecs = np.linalg.eigh(G_diff)
    labels = cluster_glyphs(G_aug)
    kernel_pca_with_clusters(G_aug, labels, degree=3)
    #plot_pca_with_clusters(G_aug, labels)
    visualize_resonance_field(G_aug)
    print("Augmented Gram Matrix shape:", G_aug.shape)
    print("Top 5 eigenvalues:", evals[::-1][:5])