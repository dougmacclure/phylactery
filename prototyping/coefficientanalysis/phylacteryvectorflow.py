import numpy as np

def flow_field(c, coeffs, pts):
    """Return unit eigen-vector & growth for an array of points."""
    # derivative of L_c evaluated via series
    deriv = np.zeros_like(pts, dtype=complex)
    zpow = np.ones_like(pts, dtype=complex)
    for n, a_n in enumerate(coeffs[1:], start=1):
        zpow *= pts            # z^{n}
        deriv += n * a_n * zpow
    v = 2*pts * deriv         # Jacobian in ℂ
    g = np.log(np.abs(v)+1e-14)
    hat = v / (np.abs(v)+1e-14)
    return hat, g

hat, g = flow_field(c, coeffs, sample_grid)
plt.quiver(xg, yg, hat.real, hat.imag, g, cmap='plasma')
