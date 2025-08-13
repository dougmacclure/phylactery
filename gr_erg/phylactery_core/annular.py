import numpy as np

def T_inv(z: complex, k: int, z0: complex):
    return z / (2 * z0) ** (k + 1)

def f_iter(z: complex, d: complex, k: int):
    if k == 0:
        return z
    for _ in range(k):
        z = z * z + d
    return z