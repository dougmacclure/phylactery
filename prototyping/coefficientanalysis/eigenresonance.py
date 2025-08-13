import numpy as np
import matplotlib.pyplot as plt
import cmath
from numpy.linalg import norm
#test
# --- Gram Matrix Construction Framework ---

def is_in_mandelbrot(c, max_iter=1000):
    z = 0
    for _ in range(max_iter):
        z = z * z + c
        if abs(z) > 2:
            return False
    return True

def compute_fixed_points(c):
    # Solves z = (z + c)^2 => z^2 + 2cz + c^2 - z = 0 => z^2 + (2c - 1)z + c^2 = 0
    a = 1
    b = 2 * c - 1
    d = c**2
    disc = cmath.sqrt(b**2 - 4 * a * d)
    z1 = (-b + disc) / (2 * a)
    z2 = (-b - disc) / (2 * a)
    return z1, z2  # Return both alpha and beta fixed points

def compute_coefficients(c, z0, deg=100):
    coeffs = []
    a0 = (1.0 - cmath.sqrt(1.0 - 4.0 * c)) / 2.0  # Choose negative root
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
        if abs(denom) < 1e-12:
            coeffs.append(0.0)
        else:
            coeffs.append(acc / denom)
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
                coeffs = compute_coefficients(c, z_beta, deg=deg)  # Use beta fixed point
                glyphs.append((c, coeffs))
                break
            r += r_step
    return glyphs

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
            if ni < 1e-12 or nj < 1e-12:
                G[i, j] = 0.0
            else:
                G[i, j] = np.vdot(vi, vj) / (ni * nj)
    return G

# --- Clustering and PCA Coloring ---
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


def plot_pca_with_clusters(G, labels):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(np.real(G))
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

def overlay_z0_norms(z0_norms):
    plt.figure(figsize=(10, 4))
    plt.plot(z0_norms, '.', markersize=1, alpha=0.7)
    plt.title("Overlay of |z₀| Norms per Glyph")
    plt.xlabel("Glyph Index")
    plt.ylabel("|z₀|")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pca_colored_by_recurrence(G):
    recurrence_strength = np.abs(G).mean(axis=1)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(np.real(G))
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=recurrence_strength, cmap='viridis')
    plt.title("PCA Projection Colored by Recurrence Strength")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label='Mean |G[i,j]|')
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
    labels = clustering.fit_predict(np.abs(G_norm))
    return labels

def visualize_resonance_field(G, title="Resonance Field (|G|)", recurrence_threshold=0.9):
    magnitude = np.abs(G)
    plt.figure(figsize=(10, 8))
    plt.imshow(magnitude, cmap='cividis', interpolation='nearest', vmin=0, vmax=np.percentile(magnitude, 95))
    plt.colorbar(label='Resonance Magnitude')
    plt.title(title)
    plt.xlabel("j (glyph index)")
    plt.ylabel("i (glyph index)")
    plt.grid(False)

    recur_i, recur_j = np.where(magnitude > recurrence_threshold)
    if recur_i.size > 0:
        plt.scatter(recur_j, recur_i, color='white', edgecolor='black', s=20, alpha=0.9, label=f'Recurrence (|Gᵢⱼ| > {recurrence_threshold})')
        plt.legend()
    else:
        print("[info] No recurrence points found above threshold.")

    # Overlay gradient vector field as directional flow
    Y, X = np.mgrid[0:magnitude.shape[0], 0:magnitude.shape[1]]
    dy, dx = np.gradient(magnitude)
    plt.quiver(X, Y, dx, dy, color='red', alpha=0.3, scale=50)

    plt.tight_layout()
    plt.show()

# Run the full procedure
glyphs = generate_coeff_lattice()
G = build_gram_matrix(glyphs)
visualize_resonance_field(G)
# Usage example:
#glyphs = generate_exotic_glyphs()
#G = compute_finite_gram_matrix(glyphs, weight_fn=lambda n: np.exp(-0.15 * n), N=20)
labels = cluster_glyphs(G)
plot_pca_with_clusters(G, labels)
plot_pca_colored_by_recurrence(G)

# Display a summary of matrix shape
G.shape
