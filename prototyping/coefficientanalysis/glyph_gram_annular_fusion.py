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
import umap
import pacmap

    
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

def compute_coefficients(c, a0, z0, deg=100):
    coeffs = []
    
    if a0 == z0:
        return None
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

def generate_coeff_lattice(θ_steps=1000, r_step=0.01, r_start=0.251, r_max=2.0, deg=100):
    glyphs = []
    angles = np.linspace(0, 2 * np.pi, θ_steps, endpoint=False)
    for θ in angles:
        r = r_start
        while r < r_max:
            c = r * cmath.exp(1j * θ)
            if not is_in_mandelbrot(c):
                z_alpha, z_beta = compute_fixed_points(c)
                if z_alpha == z_beta:
                    continue
                print(c)
                coeffs = compute_coefficients(c, z_alpha,z_beta, deg=deg)
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

import numpy as np
from sklearn.preprocessing import normalize
from scipy.ndimage import gaussian_filter1d

def construct_annular_embedding(glyphs,
                                erad_values,    # precomputed array shape (N,)
                                num_annuli=20,
                                r_min=None, r_max=None,
                                smooth_sigma=1.0):
    """
    glyphs: list of (c, coeffs)
    erad_values: np.array of length N with the L2 norm (or your chosen measure)
    """
    N = len(glyphs)
    deg = len(glyphs[0][1])

    # 1) Collect radii
    radii = np.array([abs(c) for c, _ in glyphs])
    if r_min is None: r_min = radii.min()
    if r_max is None: r_max = radii.max()

    # 2) Build soft annular histogram
    edges = np.linspace(r_min, r_max, num_annuli + 1)
    A_ann = np.zeros((N, num_annuli), dtype=float)
    for i, r in enumerate(radii):
        # find bin index
        idx = np.searchsorted(edges, r, side='right') - 1
        idx = np.clip(idx, 0, num_annuli - 1)
        A_ann[i, idx] = 1.0

    # 3) Smooth each row so membership is soft
    A_ann = gaussian_filter1d(A_ann, sigma=smooth_sigma, axis=1, mode='constant')

    # 4) Normalize rows to sum to 1
    A_ann = normalize(A_ann, norm='l1', axis=1)

    # 5) Coefficient block
    A_coeff = np.real(np.vstack([coeffs for _, coeffs in glyphs]))

    # 6) Erad block (precomputed)
    A_erad = erad_values.reshape(N, 1)

    # 7) Stack everything
    return np.hstack([A_coeff, A_ann, A_erad])

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
    # 1) Symmetrize and clean up
    G_sym = np.real(G)
    G_sym = (G_sym + G_sym.T) / 2
    G_sym = np.clip(G_sym, 0, None)

    # 2) SpectralClustering with LOBPCG
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        eigen_solver='lobpcg',
        random_state=42,
        n_jobs=-1
    )
    return clustering.fit_predict(G_sym)
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
from sklearn.decomposition import KernelPCA
from sklearn.impute import SimpleImputer

def kernel_pca_with_clusters(G, labels, degree=3):
    # 1) Force to real
    G_real = np.real(G)

    # 2) Clean up any NaNs or Infs
    G_real[np.isnan(G_real)] = 0.0
    G_real[np.isinf(G_real)] = np.sign(G_real[np.isinf(G_real)]) * 1e6

    # 3) Impute any remaining missing values
    imputer = SimpleImputer(strategy='mean')
    G_imputed = imputer.fit_transform(G_real)

    # 4) Standardize to zero mean, unit variance
    scaler = StandardScaler()
    G_scaled = scaler.fit_transform(G_imputed)

    # 5) Kernel PCA
    kpca = KernelPCA(
        n_components=2,
        kernel='poly',
        degree=degree,
        coef0=1,
        eigen_solver='auto'
    )
    coords = kpca.fit_transform(G_scaled)

    # 6) Plot clusters
    plt.figure(figsize=(8, 6))
    for lbl in np.unique(labels):
        mask = (labels == lbl)
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    label=f"Cluster {lbl}", alpha=0.6, s=30)
    plt.title(f"Polynomial Kernel PCA (degree={degree}) with Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import SpectralClustering


def apply_kernel_pca_with_clusters(X, kernel='poly', degree=3, gamma=1e-9, coef0=1.0, clusters=5):
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

def pacMAP(X_scaled):    
    # Initialize and fit PaCMAP
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
    X_transformed = embedding.fit_transform(X_scaled, init="pca")

    # Plot the results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1],  cmap='Spectral', s=5)
    plt.legend(*scatter.legend_elements(), title="Digits")
    plt.title("PaCMAP Embedding of Digits Dataset")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


    return X_kpca, labels   

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



# --- Gram Matrix Kernel Traversal Operator ---

def external_loop_traverse(theta_start, theta_end, steps, r=1.5, deg=100, kernel='poly'):
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

# Example use with A_aug or your glyph annular-convolved structure
# Ensure A_aug is your input matrix
# X_kpca, labels = apply_kernel_pca_with_clusters(A_aug, degree=7)
import numpy as np

def spectral_embedding(G, n_components=2):
    # G should be Hermitian, size N×N
    evals, evecs = np.linalg.eigh(G.real)      # force real
    idx = np.argsort(evals)[::-1]              # descending
    top_idx = idx[:n_components]
    # scale each eigenvector by sqrt(eigenvalue)
    coords = evecs[:, top_idx] * np.sqrt(evals[top_idx])
    return coords


# --- Execution ---
if __name__ == "__main__":
    glyphs = generate_coeff_lattice(θ_steps=1000, r_step=0.05, r_start=1.0, r_max=2.1, deg=100)
    G = build_gram_matrix(glyphs)
    coords = spectral_embedding(G, n_components=2)  # shape (N,2)
    plt.scatter(coords[:,0], coords[:,1], c=labels, cmap='tab10')
    plt.title("Spectral Embedding of Gram Matrix")
    plt.xlabel("Mode 1"); plt.ylabel("Mode 2"); plt.show()
    eigs = np.linalg.eigvals(G)
    erad_values = np.linalg.norm(eigs)   # L2 norm for the whole Gram
    erad_values = np.array([erad_values] * len(glyphs))
    A_aug = construct_annular_embedding(
    glyphs,
    erad_values,
    num_annuli=20,
    r_min=1.0, r_max=2.1,
    smooth_sigma=1.0
)

    G_aug = A_aug @ A_aug.T
    #G_diff = expm(G_aug)
    #evals, evecs = np.linalg.eigh(G_diff)
    labels = cluster_glyphs(G_aug)
    print(G_aug.shape)
    
    #plot_pca_with_clusters(G_aug, labels)
    #visualize_resonance_field(G_aug)
    #print("Augmented Gram Matrix shape:", G_aug.shape)
    #print("Top 5 eigenvalues:", evals[::-1][:5])
    #kernel_pca_with_clusters(G_aug, labels, degree=12)
    #apply_kernel_pca_with_clusters(G_aug, kernel='poly', degree=3, gamma=1e-9, coef0=1.0, clusters=5)
    for i in range(15):
        kernel_pca_with_clusters(A_aug, labels, i+1)