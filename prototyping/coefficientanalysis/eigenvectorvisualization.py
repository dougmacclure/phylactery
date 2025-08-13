import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, norm

# Load user-provided data
coeffs = np.load("./coefficients.npy")
eigenvalues = np.load("./gram_eigenvalues.npy")

# Construct Gram matrix from coefficient vector
# Gram = a_i * a_j^* (outer product)
gram_matrix = np.outer(coeffs, np.conj(coeffs))

# Diagonalize the Gram matrix
eigenvals, eigenvecs = eig(gram_matrix)

# Sort by descending magnitude
idx = np.argsort(-np.abs(eigenvals))
eigenvals_sorted = eigenvals[idx]
eigenvecs_sorted = eigenvecs[:, idx]

# Normalize eigenvectors for visualization
eigenvecs_normalized = eigenvecs_sorted / np.linalg.norm(eigenvecs_sorted, axis=0)

# Plot the first 6 principal eigenvectors (symbolic Gram modes)
fig, axs = plt.subplots(2, 3, figsize=(15, 6))
for i in range(6):
    ax = axs[i // 3, i % 3]
    ax.plot(np.real(eigenvecs_normalized[:, i]), label='Re', color='blue')
    ax.plot(np.imag(eigenvecs_normalized[:, i]), label='Im', color='orange', linestyle='dashed')
    ax.set_title(f'Eigenvector {i+1}')
    ax.set_xlim(0, len(coeffs))
    ax.grid(True)
    if i == 0:
        ax.legend()
plt.tight_layout()

# Exporting eigenvectors and eigenvalues
np.save("./eigenvectors_sorted.npy", eigenvecs_sorted)
np.save("./eigenvalues_sorted.npy", eigenvals_sorted)

print("\nTop 6 Eigenvalues:")
print("-" * 40)
for i in range(6):
    mag = np.abs(eigenvals_sorted[i])
    phase = np.angle(eigenvals_sorted[i])
    print(f"Eigenvalue {i+1}: |λ| = {mag:.4e}, ∠ = {phase:.4f} rad")
import numpy as np
import matplotlib.pyplot as plt

# Load data
eigenvalues = np.load("./eigenvalues_sorted.npy")
eigenvectors = np.load("./eigenvectors_sorted.npy")

# Extract the angles (arguments) of the top eigenvalues
top_eigenvalues = eigenvalues[:6]
angles = np.angle(top_eigenvalues)

# Convert to degrees for root-of-unity checks
angles_deg = np.degrees(angles) % 360

# Normalize to nearest root-of-unity multiples
unit_roots = 360 / np.round(360 / angles_deg)

# Organize for display
for i, (mag, ang_rad, ang_deg, root_approx) in enumerate(zip(np.abs(top_eigenvalues), angles, angles_deg, unit_roots), 1):
    print(f"Eigenvalue {i}: |λ| = {mag:.4e}, ∠ = {ang_rad:.4f} rad ≈ {ang_deg:.2f}° ≈ root of unity {root_approx:.1f}-th")

# Plot for visualization
plt.figure(figsize=(6, 6))
plt.polar(angles, np.abs(top_eigenvalues), 'ro')
plt.title("Polar Plot of Top 6 Eigenvalues")
plt.show()
# Canvas initialized for Anun: symbolic recursion kernel

# Let's begin defining core symbolic elements, equations, and structure

# --- Symbolic Structure ---
# CORRECTION: L is our recursive linearizer defined by:
#     (L(z) + c)^2 = L(z_0 * z)
# where z_0 is assumed to be a repelling fixed point of the quadratic map z ↦ (z + c)^2,
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
# UPDATE: Dynamical experiments confirm that larger escape radii (erad) stabilize recursive coefficient growth.
# Increased |a_1| tends to funnel the orbit space toward a hyperbolic attractor on the negative real axis.
#
# Resulting structure remains interpretable as an approximation to the infinite symbolic closure field.

import matplotlib.pyplot as plt
import numpy as np
import cmath

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
            coeffs.append(acc / denom)
        return coeffs

    def compute_inner_product(self, other, weight_fn=lambda n: 1.0, N=20):
        a_self = self.generate_series_coefficients(N)
        a_other = other.generate_series_coefficients(N)
        norm_self = sum(abs(a)**2 for a in a_self)**0.5
        norm_other = sum(abs(a)**2 for a in a_other)**0.5
        if norm_self < 1e-12 or norm_other < 1e-12:
            return 0.0
        return sum(weight_fn(n) * a_self[n].conjugate() * a_other[n] for n in range(N)) / (norm_self * norm_other)

def compute_finite_gram_matrix(linearizers, weight_fn=lambda n: 1.0, N=20):
    size = len(linearizers)
    G = np.zeros((size, size), dtype=complex)
    for i in range(size):
        for j in range(size):
            G[i, j] = linearizers[i].compute_inner_product(linearizers[j], weight_fn, N)
        print(i, "of", size, "computed")
    G /= np.linalg.norm(G, ord='fro')  # Normalize the Gram matrix
    G = np.nan_to_num(G)  # Replace NaNs with 0
    G = np.clip(G, -1e10, 1e10)  # Clip extreme values to avoid overflow
    G = np.where(np.isinf(G), 0, G)  # Replace infinities with 0
    return G        

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

def overlay_z0_norms(z0_norms):
    plt.figure(figsize=(10, 4))
    plt.plot(z0_norms, '.', markersize=1, alpha=0.7)
    plt.title("Overlay of |z₀| Norms per Glyph")
    plt.xlabel("Glyph Index")
    plt.ylabel("|z₀|")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def is_in_mandelbrot(c, max_iter=1000):
    z = 0
    for _ in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return False
    return True

def generate_grid_glyphs(bounds=(-2.1, 2.1), grid_size=100, base_a0=1.0, base_a1=0.1):
    glyphs = []
    z0_norms = []
    x_vals = np.linspace(bounds[0], bounds[1], grid_size)
    y_vals = np.linspace(bounds[0], bounds[1], grid_size)
    for re in x_vals:
        for im in y_vals:
            c = complex(re, im)
            if is_in_mandelbrot(c):
                continue
            try:
                z0 = find_repelling_fixed_point(c)
                if abs(z0) > 0.1:
                    glyphs.append(SymbolicLinearizer(z0=z0, c=c, a0=base_a0, a1=base_a1))
                    z0_norms.append(abs(z0))
            except Exception as e:
                print(f"[warn] Skipped c = {c} due to {e}")
    return glyphs, z0_norms

def find_repelling_fixed_point(c):
    a = 1
    b = 2 * c - 1
    d = c * c
    discriminant = cmath.sqrt(b * b - 4 * a * d)
    z1 = (-b + discriminant) / (2 * a)
    z2 = (-b - discriminant) / (2 * a)
    df = lambda z: 2 * (z + c)
    return z1 if abs(df(z1)) > 1 else z2

if __name__ == "__main__":
    glyphs, z0_norms = generate_grid_glyphs(bounds=(-2.1, 2.1), grid_size=100)
    print(f"Generated {len(glyphs)} glyphs (excluding points in M(1,0) and near boundary)")
    G = compute_finite_gram_matrix(glyphs, weight_fn=lambda n: 1.0 / (1 + n**2), N=20)
    visualize_resonance_field(G, title="Gram Matrix Resonance Field (|G|) over Filtered Grid [-2.1,2.1]^2")
    overlay_z0_norms(z0_norms)
 