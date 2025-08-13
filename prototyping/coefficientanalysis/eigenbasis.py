# Re-import necessary libraries due to kernel reset
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
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

# Reload uploaded coefficient data
coeffs = np.load('./coefficients.npy')  # shape: (n,)

# Construct Gram matrix
G = np.outer(coeffs, np.conj(coeffs))  # complex Gram matrix

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(G)

# Sort by eigenvalue magnitude (descending)
idx = np.argsort(np.abs(eigenvalues))[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Normalize eigenvectors for visualization
normed_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

# Plot top 6 eigenvectors (real and imaginary parts)
fig, axes = plt.subplots(2, 3, figsize=(15, 6))
for i, ax in enumerate(axes.flatten()):
    ev_real = normed_eigenvectors[:, i].real
    ev_imag = normed_eigenvectors[:, i].imag
    ax.plot(ev_real, label='Re')
    ax.plot(ev_imag, linestyle='--', label='Im')
    ax.set_title(f"Eigenvector {i+1}")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
