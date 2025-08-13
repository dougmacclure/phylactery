import numpy as np
from pathlib import Path
from phylactery_core.metrics import escape_iter

LEGACY = Path(__file__).parent.parent.parent / "glyph_matrix_glyph_P1_X0.0_Y0.0_erad4.0_lim9999999.9_zoom5.0.csv"
M = np.loadtxt(LEGACY, delimiter=",")
H, W = M.shape
xmin, xmax = -1, 1

def test_diff():
    dx = (xmax - xmin) / (W - 1)
    for ix in range(0, W, 100):      # sample grid for speed
        for iy in range(0, H, 100):
            x = xmin + ix * dx
            y = xmin + iy * dx
            d = complex(x, y)
            new_i = escape_iter(d)
            assert new_i == M[iy, ix]