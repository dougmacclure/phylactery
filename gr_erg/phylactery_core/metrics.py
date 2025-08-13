import numpy as np
from numpy import sqrt, angle, conj
def escape_iter(d, max_iter=1500, escape_cond=1e3):
    z = 0j
    for i in range(1, max_iter + 1):
        old = z
        z = z * z + d
        check = i * abs(z - old) + abs(z)
        if check > escape_cond:
            return i
    return max_iter

def rgb_mod(i, max_iter):
    checkset = max_iter - i
    r = (checkset % 139) + (checkset % 109) + (checkset % 7)
    g = (checkset % 23) + (checkset % 31) + (checkset % 53) + (checkset % 113) \
        + (checkset % 2) + (checkset % 3) + (checkset % 17) + (checkset % 13)
    b = (checkset % 131) + (checkset % 107) + (checkset % 17)
    return r % 256, g % 256, b % 256
def dominant_eig_angle(coefficients):
    coeff_array = np.array(coefficients)
    gram_matrix = np.outer(coeff_array, conj(coeff_array))
    _, eigenvectors = np.linalg.eigh(gram_matrix)
    dominant_vector = eigenvectors[:, -1]
    return angle(dominant_vector)
