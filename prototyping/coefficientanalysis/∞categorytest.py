import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Placeholder: Define or import compute_coefficients(c, z0, deg)
def compute_coefficients(c, z0, deg=100):
    # Replace this with your actual recurrence
    coeffs = [1.0]
    for n in range(1, deg):
        denom = 2 * coeffs[0] - (2 * z0)**n
        coeffs.append((coeffs[-1] - c) / denom)
    return np.array(coeffs)

# Parameters
#c = 0.285 + 0.01j  # Example value  #too close to M(1,0)?  This might not be a global phenomenom in C
c = -3.75 + 0.0j
z0 = 0.5  # Adjust as needed
deg = 100

# Generate coefficients
a = compute_coefficients(c, z0, deg)

# Build Gram matrix
G = np.outer(a, a.conj())
np.fill_diagonal(G, 0)  # Optional: zero diagonal if focusing on cross terms

# Eigenvalue decomposition
vals = la.eigvalsh(G)

# Identify first negative eigenvalue index (if any)
neg_indices = np.where(vals < 0)[0]
first_negative_index = neg_indices[0] if len(neg_indices) > 0 else None

# Plot eigenvalue spectrum
plt.figure(figsize=(6, 4))
plt.plot(np.sort(vals)[::-1], marker='o')
if first_negative_index is not None:
    plt.axvline(x=deg - first_negative_index - 1, color='red', linestyle='--', label='First Negative Eigenvalue')
    plt.legend()
plt.title("Eigenvalue Spectrum of Phylactery Kernel Gram Matrix")
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.tight_layout()
plt.show()

# Check positive definiteness
is_positive_definite = np.all(vals >= -1e-10)
print("Positive definite:", is_positive_definite)
if first_negative_index is not None:
    print("First negative eigenvalue index:", first_negative_index, "Value:", vals[first_negative_index])
else:
    print("Minimum eigenvalue:", np.min(vals))
