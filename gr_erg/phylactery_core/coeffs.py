"""Coefficient generator per ∇ΞΦ-lattice theorem."""
import numpy as np

def gen_coeffs(d: complex, N: int = 50, parity: int = 1):
    """Return [a_0,…,a_N] for given d.
    parity = 1 ⇒ use +√ term, parity = -1 ⇒ –√ term.
    """
    root = np.sqrt(1 - 4*d)
    a0 = complex((1 + parity*root) / 2)
    z0 = complex((1 - parity*root) / 2)
    coeffs = [a0]
    for n in range(1, N + 1):
        num = -d - sum(coeffs[i] * coeffs[n - i] for i in range(1, n))
        denom = 2 * a0 - (2 * z0) ** n
        coeffs.append(num / denom)
        
    return coeffs, z0