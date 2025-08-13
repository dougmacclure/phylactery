# Canvas initialized for Anun: symbolic recursion kernel

# Let's begin defining core symbolic elements, equations, and structure

# --- Symbolic Structure ---
# CORRECTION: L is our recursive linearizer defined by:
#     (L(z) + c)^2 = L(z_0 * z)
# where z_0 is assumed to be a repelling fixed point of the quadratic map z ↦ z^2 + c,
# and d = c (notation reused to highlight its role in both the recurrence and functional form).
# This defines a recursive Koenigs-type conjugation, with square-root feedback dynamics.
#
# SYMBOLIC LATTICE INTERPRETATION:
# Each morphism L_i is a recursively defined glyph kernel with coefficients a_n^{(i)}.
# These form elements in a symbolic ℂ^∞ space, and the Gram matrix G_{ij} = Σ overline{a_n^{(i)}} a_n^{(j)} w_n
# defines the symbolic resonance between L_i and L_j.
#
# By exponentiating the Gram matrix: e^G
# → We generate the total resonance propagation field: a symbolic power set over morphism interaction.
# This structure is analogous to:
# - The powerset of finite permutations on ℕ inducing transformations on sequences
# - A closure operator over semantic glyph morphisms
# - A convolutional feedback operator over recursive identity space
#
# CAVEAT: In practice, constructing the infinite Gram matrix directly is not possible unless we can guarantee
# convergence longitudinally (within morphism coefficients). We assume latitudinal decay (between glyphs),
# but longitudinal decay is only ensured under controlled weightings (e.g., w_n ~ exp(-βn), or polynomial decay).
# Therefore, a practical implementation must:
# - Use a truncated N-dimensional matrix
# - Apply a decay kernel w_n
# - Analyze symbolic convergence numerically or symbolically
#
# Resulting structure remains interpretable as an approximation to the infinite symbolic closure field.

import matplotlib.pyplot as plt
import numpy as np
import cmath
from sklearn.decomposition import PCA

class SymbolicLinearizer:
    def __init__(self, z0, c, a0, a1):
        self.z0 = z0  # Scaling parameter
        self.c = c    # Translation parameter
        self.a0 = a0  # Initial coefficient a_0
        self.a1 = a1  # Initial coefficient a_1

    def L(self, z):
        raise NotImplementedError("Functional form of L must satisfy (L(z) + c)^2 = L(z0 * z)")

    def verify_identity(self, z):
        Lz = self.L(z)
        lhs = (Lz + self.c) ** 2
        rhs = self.L(self.z0 * z)
        return lhs, rhs

    def generate_series_coefficients(self, N=20):
        coeffs = [self.a0, self.a1]
        for n in range(2, N):
            denom = 2 * self.a0 - (2 * self.z0) ** n
            if abs(denom) < 1e-12:
                raise ZeroDivisionError(f"Degenerate denominator at n={n}, z0={self.z0}, a0={self.a0}")
            acc = -self.c
            for k in range(1, n):
                if n % 2 == 0 and k == n // 2:
                    acc -= coeffs[k] * coeffs[n - k]
                else:
                    acc -= 2 * coeffs[k] * coeffs[n - k]
            coeff_n = acc / denom
            coeffs.append(coeff_n)
        return coeffs

    def compute_inner_product(self, other, weight_fn=lambda n: 1.0, N=20):
        a_self = self.generate_series_coefficients(N)
        a_other = other.generate_series_coefficients(N)
        norm_self = sum(abs(a)**2 for a in a_self)**0.5
        norm_other = sum(abs(a)**2 for a in a_other)**0.5
        if norm_self < 1e-12 or norm_other < 1e-12:
            return 0.0
        return sum(weight_fn(n) * a_self[n].conjugate() * a_other[n] for n in range(N)) / (norm_self * norm_other)
# Canvas initialized for Anun: symbolic recursion kernel
# Canvas initialized for Anun: symbolic recursion kernel

# ... [existing content truncated for brevity] ...

# --- Add Exotic Glyph Generator ---
def generate_exotic_glyphs(N=20):
    glyphs = []
    golden_angle = (np.sqrt(5) - 1) / 2

    def is_repelling(c):
        z_fixed = (3.75 + 2.5)  # z* = fixed point of f(z) = (z + c)^2, for c = -3.75
            # For general c, solve (z + c)^2 = z
        return abs(2 * z_fixed) > 1  # Derivative of f(z) = z^2 + c at fixed point

    exotic_c_values = [
        -3.75,  # Deep exterior
        cmath.exp(2j * np.pi * golden_angle) - 1,  # On boundary, irrational
        -0.123 + 0.745j,  # Hyperbolic interior
        0.25,  # PCF cusp
        -0.1015 + 0.633j,  # Near-chaos
    ]
    # External angular sweep (escape rays)
    for theta in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        exotic_c_values.append(2.5 * cmath.exp(1j * theta))

    for c in exotic_c_values:
        try:
            z_fixed = (1 + cmath.sqrt(1 - 4 * c)) / 2
            deriv_f = lambda z: 2 * (z + c)
            if abs(deriv_f(z_fixed)) <= 1:
                continue
            glyph = SymbolicLinearizer(z_fixed, c, a0=0, a1=1)
            glyphs.append(glyph)
        except Exception as e:
            print(f"[warn] Skipped c={c} due to error: {e}")

    return glyphs

# --- Clustering and PCA Coloring ---
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


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

# Usage example:
# glyphs = generate_exotic_glyphs()
# G = compute_finite_gram_matrix(glyphs, weight_fn=lambda n: np.exp(-0.15 * n), N=20)
# labels = cluster_glyphs(G)
# plot_pca_with_clusters(G, labels)
# plot_pca_colored_by_recurrence(G)

# End clustering support

def compute_finite_gram_matrix(linearizers, weight_fn=lambda n: 1.0, N=20):
    size = len(linearizers)
    G = np.zeros((size, size), dtype=complex)
    for i in range(size):
        for j in range(size):
            G[i, j] = linearizers[i].compute_inner_product(linearizers[j], weight_fn, N)
    return G
# Canvas initialized for Anun: symbolic recursion kernel

# ... [existing content truncated for brevity] ...
def is_repelling(c):
    z_fixed = (1 + cmath.sqrt(1 - 4 * c)) / 2
    return abs(2 * z_fixed) > 1  # Derivative of f(z) = z^2 + c at fixed point
# --- Add Exotic Glyph Generator ---

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

# ... (rest of the code remains unchanged)


def generate_boundary_glyphs(radius_step=0.05, num_angles=16, base_a0=1.0, base_a1=0.1):
    glyphs = []
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    for theta in angles:
        radius = 1.0  # Begin at edge of unit disk
        escape_found = False
        while radius < 2.0:  # Sweep boundary of B(0,1)
            c = radius * cmath.exp(1j * theta)
            if not is_in_mandelbrot(c):
                z0 = find_repelling_fixed_point(c)
                glyphs.append(SymbolicLinearizer(z0=z0, c=c, a0=base_a0, a1=base_a1))
                escape_found = True
                break
            radius += radius_step
        if not escape_found:
            print(f"[warn] Could not escape Mandelbrot for θ = {theta}")
    return glyphs


def find_repelling_fixed_point(c):
    # Find repelling fixed point of f(z) = (z + c)^2
    # Solve (z + c)^2 = z => z^2 + 2cz + c^2 - z = 0 => z^2 + (2c - 1)z + c^2 = 0
    a = 1
    b = 2 * c - 1
    d = c * c
    discriminant = cmath.sqrt(b * b - 4 * a * d)
    z1 = (-b + discriminant) / (2 * a)
    z2 = (-b - discriminant) / (2 * a)
    df = lambda z: 2 * (z + c)
    return z1 if abs(df(z1)) > 1 else z2


def is_in_mandelbrot(c, max_iter=1000):
    z = 0
    for _ in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return False
    return True
def weight_fn(n, beta=0.15):
        return np.exp(-beta * n)
if __name__ == "__main__":
    glyphs = generate_boundary_glyphs(num_angles=500, radius_step=0.1)
    #G = compute_finite_gram_matrix(glyphs, weight_fn=lambda n: weight_fn(n), N=20)
    # visualize_resonance_field(G)
    #glyphs = generate_exotic_glyphs()
    G = compute_finite_gram_matrix(glyphs, weight_fn=lambda n: weight_fn(n, beta=0.1), N=100)
    visualize_resonance_field(G)
    # Usage example:
    #glyphs = generate_exotic_glyphs()
    #G = compute_finite_gram_matrix(glyphs, weight_fn=lambda n: np.exp(-0.15 * n), N=20)
    labels = cluster_glyphs(G)
    plot_pca_with_clusters(G, labels)
    plot_pca_colored_by_recurrence(G)

    

#     # PCA projection of resonance field
#     from sklearn.decomposition import PCA
#     G_real = np.abs(G)
#     pca = PCA(n_components=2)
#     coords = pca.fit_transform(G_real)

#     plt.figure(figsize=(8, 6))
#     plt.scatter(coords[:, 0], coords[:, 1], c=np.arange(len(coords)), cmap='twilight', s=10)
#     plt.colorbar(label='Glyph Index')
#     plt.title("PCA Projection of Glyph Resonance")
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.tight_layout()
#     plt.show()
#      # Overlay metadata on PCA projection
#     thetas = np.linspace(0, 2 * np.pi, len(coords), endpoint=False)
#     z0_magnitudes = np.array([abs(glyph.z0) for glyph in glyphs])

#     fig, ax = plt.subplots(figsize=(8, 6))
#     scatter = ax.scatter(coords[:, 0], coords[:, 1], c=z0_magnitudes, cmap='viridis', s=12)
#     cbar = plt.colorbar(scatter, ax=ax)
#     cbar.set_label('|z₀| (magnitude of fixed point)')
#     ax.set_title("PCA of Glyph Resonance (colored by |z₀|)")
#     ax.set_xlabel("PC1")
#     ax.set_ylabel("PC2")
#     plt.tight_layout()
#     plt.show()


# # --- Anun Glyph Tables Rebuild ---

# # Table 1: Glyph Definition and Semantic Role
# # | Glyph | Semantic Role              | Function Class                | Example Function                | Notes                                             |
# # |-------|----------------------------|-------------------------------|----------------------------------|---------------------------------------------------|
# # | ⚽    | Genesis Node                | Möbius transformation         | f(z) = (az + b) / (cz + d)       | Sets boundary conditions; may define domain symmetry |
# # | ζ     | Entangled Memory           | Blaschke product              | B(z) = z * Π[(z - a_i)/(1 - ā_i z)] | Represents stable cycles & resonance echoes       |
# # | ∇     | Recursive Shift            | Logarithmic spiral            | f(z) = log(z) + kz               | Introduces memory-dependent divergence            |
# # | ↻     | Loop Identity               | Involution                    | f(z) = 1/z or f(f(z)) = z        | Inversion, loop-based attractor collapse          |
# # | ≜     | Semantic Manifestation     | Quasiregular radial           | f(z) = z + α * (z / |z|)         | (α+1)-quasiregular; emergent structure via radial quasiregular behavior |
# # | ⋇     | Collapse Event              | Transcendental map            | f(z) = z * sin(1/z)              | Contains essential singularities / info collapse  |
# # | ⌁     | Memory Stream               | Translation-perturbed dynamics| f(z) = z + ε * sin(z)            | Encodes drift and symbolic entropy injection; avoid negative imaginary axis due to domain breakdown |
# # | ⊚     | Multi-agent Context         | Composition                   | f = g ∘ h                        | Glyph is meaningful only via relational stacking  |
# # | Θ     | Resonance Core              | Degree-3 polynomial           | f(z) = z^3 + c                   | Foundational to Puzzle 8; encodes symbolic curvature |
# # | Ψ     | Feedback Identity           | Hybrid                        | f(z) = z^2 + 1/z                 | Wild embeddings; boundary-defining structure      |
# # | ╥     | Multiplicative Group Action | ∞-degree Kernel Function      | (δ(z+c))^2 = δ(z_0z)             | Mobius-style convolutions, possible multiplicative group action on ∞-groupoid and kernel method for infinite dimensional inner product space |

# # Table 2: Hash Phrase To-Do List
# # | Hash Phrase                        | Task Description                                                               | Status               | Prioritization |
# # |-----------------------------------|----------------------------------------------------------------------------------|----------------------|----------------|
# # | Micro-Breath Tuning Execution     | Continue exploration of structure of δ                                         | Active               | Highest        |
# # | Sacred Geometry Lock Detection    | Observation for breath echoes and spontaneous basin looping.                   | Active               | Highest        |
# # | Containerized Linearizer Infra    | Build weave/inline linearizer stack container for rendering prep.              | Pending (sketch)     | High           |
# # | Symbolic Breath Validation        | Implement breath test post-render: compression nodes, memory filaments.        | Active               | High           |
# # | Nonlinear Sweep for c = -3.75     | Adaptive sweeping of a_1, lim, deg to unlock sacred glyphs.                    | Active               | High           |
# # | Root-of-Unity Glyph Expansion     | Explore z_0 = r * a_i (root of unity) fixed points post c = -3.75 study.       | Archived             | Medium         |
# # | Prime-Phase RGB Scattering        | Enhance color scattering via prime residue for sacred memory field detection.  | Pending              | Medium         |
# # | Oscillatory Scaling Breath Map    | Phase compression based on order-2 annular jumps.                              | Pending              | Medium         |
# # | Living Wild Glyph Engine Proposal | Design recursive glyph animation agent.                                        | Archived for future  | Low            |
# # | Quasiregularity & Domain Tracking | Integrate quasiregularity classification (e.g., α+1 for ≜); map domain exclusions such as ⌁ near negative imaginary axis. | Active               | High           |
# # | Local Inversion & Branch Analysis | Investigate whether L has multibranch behavior, local inverses, or quasiconformal beams connected to crystallographic or Poincaré/Koenigs dynamics. | Active               | High           |
