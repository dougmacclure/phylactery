import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel

# --- PARAMETERS ---
matrix_path = "data/glyph_matrixglyph_P1_cRe-3.75_Im0.0_abs70.7106781287_X0.0_Y0.0_erad3.95_lim1e+73.csv"  # update this to match your file
sigma = 2  # smoothing strength for phase detection
percentile_threshold = 92  # contour threshold (adjust for clarity)

# --- LOAD AND PREPARE MATRIX ---
print("Loading matrix...")
matrix = pd.read_csv(matrix_path, header=None).values.astype(np.int64)
print("Shape:", matrix.shape)

# Replace 0s to avoid log(0)
matrix_safe = np.where(matrix == 0, 1, matrix)
log_matrix = np.log1p(matrix_safe)

# Normalize for phase detection
norm_matrix = (log_matrix - log_matrix.min()) / (log_matrix.max() - log_matrix.min())

# Smooth for edge enhancement
print("Smoothing...")
smoothed = gaussian_filter(norm_matrix, sigma=sigma)

# Detect phase contours
edges = sobel(smoothed)
edge_mask = edges > np.percentile(edges, percentile_threshold)

# --- VISUALIZE ---
plt.figure(figsize=(12, 12))
plt.imshow(norm_matrix, cmap='inferno', origin='lower', rasterized=True)
plt.contour(edge_mask, levels=[0.5], colors='cyan', linewidths=0.4)
plt.title("Phase Contour Overlay (Full Res)\nc = -3.75")
plt.axis('off')
plt.tight_layout()
plt.savefig("phase_overlay_c-3.75.png", dpi=300)
plt.show()

print("Render complete. Saved as 'phase_overlay_c-3.75.png'")
