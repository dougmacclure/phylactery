# phylactery_core/pd_eval.py
from functools import lru_cache
from .coeffs import gen_coeffs
from .annular import f_iter

@lru_cache(maxsize=20_000)
def _horner(w: complex, coeff_key: tuple, erad: float):
    """Cached Horner evaluation of Σ a_n wⁿ."""
    val = 0j
    for a in coeff_key[::-1]:
        val = val * w + a
    return val

def eval_series(w, coeffs, erad):
    if abs(w) > erad:
        raise ValueError("|w| outside disk")
        return np.inf
    #w_round = complex(round(w.real, 6), round(w.imag, 6))
    w_round = w
    return _horner(w_round, tuple(coeffs), erad)

def P_ext(z, d, *, coeffs, z0, erad=3.95, lim=1e6,
          k_cap=60, eps=1.001):
    factor = 2 * z0
    k = 0
    if abs(factor) > eps:
        while (abs(z) > erad * (abs(factor) ** k)
               and abs(z) < lim and k < k_cap-1):
            z /= factor
            k += 1
    # attracting (|factor|≤eps) or hit k_cap
    w = eval_series(z, coeffs, erad)
    return f_iter(w, d, k)

