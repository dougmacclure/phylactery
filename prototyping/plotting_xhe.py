import numpy as np
import matplotlib.pyplot as plt

# Define Mandelbrot escape time function
def mandelbrot_escape_time(c, max_iter=500, escape_radius=100):
    z = 0
    for i in range(max_iter):
        z = z*z + c
        if abs(z) > escape_radius:
            return i
    return max_iter

# Estimate distance to the Mandelbrot set boundary
def estimate_distance(c, max_iter=500):
    iters = mandelbrot_escape_time(c, max_iter)
    if iters == max_iter:
        return 1e-8  # Clamp small if inside or near the boundary
    return 1 / (iters + 1e-5)

# Modified erad function with log-distance to boundary
def erad_log_distance(c, K=0.2):
    d = estimate_distance(c)
    return K / np.log(1 + 1 / d)

# Set up grid over complex plane
re, im = np.meshgrid(np.linspace(-2, 1, 1000), np.linspace(-1.5, 1.5, 1000))
cgrid = re + 1j * im

# Compute erad over the grid
erad_grid = np.vectorize(lambda c: erad_log_distance(c))(cgrid)

# Plot the result
plt.figure(figsize=(10, 10))
plt.imshow(erad_grid, extent=(-2, 1, -1.5, 1.5), cmap='plasma', origin='lower')
plt.colorbar(label="erad (log-distance to ∂M(1,0))")
plt.title("erad(c) with log-distance to Mandelbrot Boundary ∂M(1,0)")
plt.xlabel("Re(c)")
plt.ylabel("Im(c)")
plt.tight_layout()
plt_path = "plots/erad_log_distance_to_M10.png"
plt.savefig(plt_path)
plt.close()
