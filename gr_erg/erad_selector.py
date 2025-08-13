from phylactery_core.coeffs import gen_coeffs         # your helper
import numpy as np, scipy.linalg as la, cmath
def _companion(coeffs):
    """
    Build the k×k companion matrix for coeff list [a0 … a_{k-1}]
    representing  a0 + a1 z + ... + a_{k-1} z^{k-1}.
    """
    k = len(coeffs)
    C = np.zeros((k, k), dtype=np.complex128)
    C[1:, :-1] = np.eye(k-1)
    C[0, :k-1] = -np.asarray(coeffs[1:], dtype=np.complex128) / coeffs[0]
    return C
def erad_bandwidth(c, N=8, band=0.1):
    a0 = (1 - cmath.sqrt(1 - 4*c)) / 2
    z0 = (1 + cmath.sqrt(1 - 4*c)) / 2
    candidates = []
    for erad in np.linspace(0.05, 800.0, 5000):
        count = 0
        for n in range(1, N):
            Dn = 2*a0 - (2*z0)**n
            if abs(abs(Dn) - erad) < band:
                count += 1
        if count >= 2:
            candidates.append((erad, count))
    print(count)
    return candidates

def companion(a):
    k = len(a)
    C = np.zeros((k, k), dtype=np.complex128)
    C[1:, :-1] = np.eye(k-1)
    C[0, :k-1] = -np.asarray(a[1:], np.complex128) / a[0]
    return C

def spec_l2_weighted(coeffs):
    vals = np.abs(np.linalg.eigvals(_companion(coeffs)))
    weights = np.arange(1, len(vals)+1)
    return np.sqrt(((weights * vals) ** 2).sum())

#I got my 

def spec_l2(coeffs):
    vals = np.abs(np.linalg.eigvals(_companion(coeffs)))
    return np.sqrt((vals ** 2).sum())

def _spec_sum(coeffs):
    vals = np.abs(la.eigvals(_companion(coeffs)))
    return vals.sum()

def _spec_geom(coeffs):
    vals = np.abs(la.eigvals(_companion(coeffs)))
    print(vals)
    return vals.prod() ** (1.0/len(vals))

def _leading_eig_mag(coeffs):
    k = len(coeffs)
    coeffs = coeffs[::-1]

    
    C = np.zeros((k, k), dtype=np.complex128)
    C[1:, :-1] = np.eye(k-1)
    C[0, :k-1] = -np.asarray(coeffs[1:], dtype=np.complex128) / coeffs[0]
    return np.abs(la.eigvals(C)).max()

def l2_norm(coeffs):
    vals = np.abs(la.eigvals(companion(coeffs)))
    return np.sqrt((vals * vals).sum())               # Euclidean norm

def erad_from_spectrum(c, k=100):
    coeffs, z0 = gen_coeffs(c, N=k)            # UNPACK here
    print(l2_norm(coeffs))
    print(abs(2*z0))
    return l2_norm(coeffs)
def goldilocks_band(c, k=8, step=0.05):
    e_list, neutral = [], []
    for e in np.arange(0.5, 8.0, step):
        a, z0 = gen_coeffs(c, N=k)
        if abs(l2_norm(a) - e) < 0.05:    # 5 % band
            neutral.append(e)
        elif neutral:
            e_list.append((neutral[0], neutral[-1]))
            neutral = []
    return e_list

def ratio(e, c, k, parity=1, spec='geom'):
    coeffs, _ = gen_coeffs(c, N=k, parity=1)
    if spec == 'leading':   
        lam = _leading_eig_mag(coeffs)
    elif spec == 'sum':
        lam = _spec_sum(coeffs)
    elif spec == 'geom':
        lam = _spec_geom(coeffs)
    elif spec == 'l2':
        lam = spec_l2(coeffs)
    print('leading eigenvalue:', lam)
    if lam == 0:
        raise RuntimeError("Leading eigenvalue is zero. "
                           "Check coefficients for c={}".format(c))
    return lam / e          # “pressure” ratio

def find_erad_ratio(c, k=8, tol=1e-3,
                    bracket=(0.5, 6.0), max_iter=30, spec='geom'):
    """Return erad s.t. |λ1| / erad ≈ 1 (within tol)."""
    e_lo, e_hi = bracket
    f_lo = ratio(e_lo, c, k, spec=spec) - 1.0
    f_hi = ratio(e_hi, c, k,spec=spec) - 1.0
    #if f_lo * f_hi > 0:
    #    raise RuntimeError("Bracket does not straddle root. "
    #                       "Adjust bracket for c={}".format(c))

    for _ in range(max_iter):
        e_mid = 0.5 * (e_lo + e_hi)
        f_mid = ratio(e_mid, c, k, spec=spec) - 1.0
        if abs(f_mid) < tol:
            return e_mid, f_mid + 1.0   # return erad & ratio
        if f_mid * f_lo < 0:
            e_hi, f_hi = e_mid, f_mid
        else:
            e_lo, f_lo = e_mid, f_mid
    return e_mid, f_mid + 1.0           # after max_iter
def get_erad():
    import sys
    c = complex(sys.argv[1]) if len(sys.argv) > 1 else -3.75+0j
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    bracket = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.5, 6.0)
    tol = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
    maxiter = int(sys.argv[6]) if len(sys.argv) > 6 else 30
    spec = sys.argv[7] if len(sys.argv) > 7 else 'geom'
    if len(sys.argv) > 8:
        print("Usage: {} [c] [k] [bracket] [tol] [maxiter]".format(sys.argv[0]))
        print("Default: c=-3.75+0j, k=8, bracket=(0.5, 6.0), tol=1e-3, maxiter=30")
        sys.exit(1)
    if len(sys.argv) > 9:
        print("Usage: {} [c] [k] [bracket] [tol] [maxiter]".format(sys.argv[0]))
        print("Default: c=-3.75+0j, k=8, bracket=(0.5, 6.0), tol=1e-3, maxiter=30")
        sys.exit(1)
    e, r = find_erad_ratio(c, k=k, bracket=bracket, tol=tol, max_iter=maxiter, spec=spec)
    print("c =", c)
    print("erad =", e, "   |λ1|/erad =", r)
    erad = erad_from_spectrum(c, k=k)
    print("erad from spectrum =", erad)
    e_list = goldilocks_band(c, k, 0.01)
    candidates = erad_bandwidth(c, k, band=0.2)
    print('goldilocks band: ', e_list)
    print('Remns candidates: ', candidates)
    
def goldilocks_band(c, k=8, step=0.05, tol=0.05):
    e_list, neutral = [], []
    a, z0 = gen_coeffs(c, N=k)
    cntr = 0
    for e in np.arange(0.05, 800.0, step):
        cntr += 1
        if cntr % 100 == 1:
            print(100*cntr/((800-0.05)/step), ' percent remaining')
        if abs(l2_norm(a) - e) < 0.05:    # 5 % band
            neutral.append(e)
        elif neutral:
            e_list.append((neutral[0], neutral[-1]))
            neutral = []
        
    return e_list

# quick CLI sanity
if __name__ == "__main__":
    get_erad()
    
    
