# Kernel reset cleared prior file access — re-import and re-load everything
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from scipy.ndimage import gaussian_laplace
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
import os
import cv2
import matplotlib.patches as mpatches

def get_density_differential_field():
    # Re-upload density matrix file
    #density_path = "C:/Users/dougm/OneDrive/Documents/code/anun/prototyping/glyph_matrix_glyph_buddhaP1_cRe-3.75_Im0.0_abs35.3553390643_X0.0_Y0.0_erad3.95_lim1e+74.csv"
    #density = np.loadtxt(density_path, delimiter=',')
    data_flat = np.load("c:\\Users\\dougm\\OneDrive\\Documents\\code\\anun\\data/tempbuffer_glyph_parameter_buddha_array.npy")
    density = data_flat.reshape((5000, 5000))
    

    # Apply smoothing and compute gradient
    smoothed = gaussian_filter(np.log1p(density), sigma=1.0)
    gx = sobel(smoothed, axis=0)
    gy = sobel(smoothed, axis=1)
    gradient_magnitude = np.hypot(gx, gy)
    grad_normalized = gradient_magnitude / np.max(gradient_magnitude)

    # Plot maximum density differential field
    plt.figure(figsize=(10, 10))
    plt.imshow(grad_normalized, cmap='magma', origin='lower')
    plt.axis('off')
    plt.title("Max Density Differential Field – ΣΞ₁-emergence::NestedAnnularDream", fontsize=14)
    plt.show()
    plt.imsave("c:\\Users\\dougm\\OneDrive\\Documents\\code\\anun\\plots/density_differential_field.png", arr=grad_normalized)
    plt.close()

    image_path = "c:\\Users\\dougm\\OneDrive\\Documents\\code\\anun\\plots/density_differential_field.png"
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image_np = np.array(image)
    log_image = np.log1p(image_np.astype(np.float32))
    smoothed = gaussian_filter(log_image, sigma=1.0)

    # Compute gradient magnitude
    gx = sobel(smoothed, axis=0)
    gy = sobel(smoothed, axis=1)
    gradient_magnitude = np.hypot(gx, gy)
    grad_normalized = gradient_magnitude / np.max(gradient_magnitude)

    # Define thresholds for segmentation
    echo_threshold = 0.1
    pulse_threshold = 0.3
    silent_threshold = 0.02

    # Create overlay
    overlay = np.zeros((*grad_normalized.shape, 3), dtype=np.uint8)

    # Color mapping: red = echo zone, green = pulse root, blue = silent zone
    overlay[(grad_normalized >= echo_threshold) & (grad_normalized < pulse_threshold)] = [255, 0, 0]  # Echo
    overlay[grad_normalized >= pulse_threshold] = [0, 255, 0]  # Pulse root
    overlay[grad_normalized < silent_threshold] = [0, 0, 255]  # Silent glyph

    # Convert original grayscale image to RGB
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # Blend overlay with original
    blended = cv2.addWeighted(image_rgb, 0.6, overlay, 0.4, 0)

    # Save and return
    output_path = "c:\\Users\\dougm\\OneDrive\\Documents\\code\\anun\\plots/resonance_sheaf_overlay.png"
    Image.fromarray(blended).save(output_path)

    image_path = "c:\\Users\\dougm\\OneDrive\\Documents\\code\\anun\\plots/resonance_sheaf_overlay.png"
    image = Image.open(image_path)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)
    ax.axis('off')  # Hide axes

    # Add legend patches
    legend_elements = [
        mpatches.Patch(color='red', label='Echo Zones'),
        mpatches.Patch(color='green', label='Pulse Roots'),
        mpatches.Patch(color='blue', label='Silent Glyphs'),
    ]

    # Place legend on the image
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12, frameon=True, facecolor='white')

    # Save the labeled image
    labeled_path = "c:\\Users\\dougm\\OneDrive\\Documents\\code\\anun\\plots/resonance_sheaf_overlay_labeled.png"
    plt.savefig(labeled_path, bbox_inches='tight')
    plt.close()


def get_density_differential_field_vector():
    # Re-import necessary libraries after reset


    # Reload the numpy array after kernel reset
    data_flat = np.load("c:\\Users\\dougm\\OneDrive\\Documents\\code\\anun\\data/tempbuffer_glyph_parameter_buddha_array.npy")
    data_matrix = data_flat.reshape((5000, 5000))

    # Apply Laplacian filter to compute differential density
    diff_density = gaussian_laplace(data_matrix.astype(float), sigma=1.5)

    # Plot the result
    plt.figure(figsize=(10, 10))
    plt.imshow(np.abs(diff_density), cmap='inferno', norm=LogNorm())
    plt.title("Differential Density Field (Laplacian Filtered)")
    plt.axis('off')
    plt.colorbar()
    plt.show()
    plt.imsave("density_differential_field_parameterspace.png", arr=data_matrix, cmap='inferno')
    plt.close()

def overlay_eigenvector_angles():
    # Re-import libraries after kernel reset
    import numpy as np
    import matplotlib.pyplot as plt

    # Reload glyph matrix CSV
    density = np.loadtxt("C:/Users/dougm/OneDrive/Documents/code/anun/prototyping/glyph_matrix_glyph_buddhaP1_cRe-3.75_Im0.0_abs35.3553390643_X0.0_Y0.0_erad3.95_lim1e+74.csv", delimiter=',')

    # Recalculate angles from earlier session (re-declare since kernel reset)
    from numpy import sqrt, angle, conj

    # Parameters
    z0 = complex(-1.5, 0.0)
    d = complex(-3.75, 0.0)
    plmin = 1
    a0 = (1.0 + sqrt(1.0 - 4.0 * d)) / 2.0 if plmin == 1 else (1.0 - sqrt(1.0 - 4.0 * d)) / 2.0
    a0 = complex(a0, 0.0)
    a1 = -d / (2.0 * a0 - (2.0 * z0))

    # Generate coefficient list
    coefficients = [a0, a1]
    for n in range(2, 100):
        sum_term = 0
        for i in range(1, n):
            mult = 2.0 if i != n - i else 1.0
            sum_term += mult * coefficients[i] * coefficients[n - i]
        denom = 2.0 * a0 - (2.0 * z0) ** n
        an = (-d - sum_term) / denom
        coefficients.append(an)

    # Gram matrix and eigenvector angle calculation
    coeff_array = np.array(coefficients)
    gram_matrix = np.outer(coeff_array, conj(coeff_array))
    _, eigenvectors = np.linalg.eigh(gram_matrix)
    dominant_vector = eigenvectors[:, -1]
    angles = angle(dominant_vector)

    # Prepare density visualization
    log_density = np.log1p(density)
    log_density /= log_density.max()
    dim = density.shape[0]
    center = dim // 2
    line_len = dim * 0.4

    # Generate overlay lines
    lines = [
        (
            (center, center),
            (
                center + line_len * np.cos(theta),
                center + line_len * np.sin(theta)
            )
        )
        for theta in angles
    ]

    # Plot result
    plt.figure(figsize=(10, 10))
    plt.imshow(log_density, cmap='inferno', origin='lower')
    for (x0, y0), (x1, y1) in lines:
        plt.plot([x0, x1], [y0, y1], color='cyan', linewidth=0.5, alpha=0.6)
    plt.axis('off')
    plt.title("Eigenvector Angle Overlay – Ξ₁-spine::Eigenvector Vein", fontsize=14)
    plt.show()


if __name__ == "__main__":
    get_density_differential_field()
    get_density_differential_field_vector()
    #overlay_eigenvector_angles()