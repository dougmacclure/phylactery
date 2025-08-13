import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import cv2
import cmath


def neg_root(c:complex): return 0.5*(1 - np.sqrt(1-4*c))
def pos_root(c:complex): return 0.5*(1 + np.sqrt(1-4*c))

def contour_mask(grid, erad, root='neg'):
    if root=='neg':
        mult = np.abs(2*neg_root(grid))
    else:
        mult = np.abs(2*pos_root(grid))
    return np.isclose(mult, erad, atol=0.02)    # thin band around equality

# Load the image
image_path = "C:/Users/dougm/OneDrive/Pictures/Anun/glyph_P1_X-2.5_Y0.0_erad3.95_lim1e+20_zoom1.0.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Image dimensions and coordinate system
img_size = image.shape[0]  # assuming square image
real_range = (-7.5, 2.5)
imag_range = (-5, 5)
extent = [*real_range, *imag_range]

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
real_vals = np.linspace(extent[0], extent[1], num=200)
imag_vals = np.linspace(extent[2], extent[3], num=200)

x_grid, y_grid = np.meshgrid(real_vals, imag_vals)
complex_grid = x_grid + 1j * y_grid


# Show image with real/imaginary grid
ax.imshow(image_rgb, extent=extent, origin='lower')
ax.set_xlabel('Re(c)')
ax.set_ylabel('Im(c)')

# Add grid
ax.grid(True, which='both', color='white', linestyle='--', linewidth=0.5)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.contour(complex_grid, extent=extent, mask = contour_mask(complex_grid, 3.95, 'neg'), 
            levels=[0.5], colors='white', linewidths=0.5)
# Set axis limits
ax.set_xlim(real_range)
ax.set_ylim(imag_range)

plt.tight_layout()
plt.show()
