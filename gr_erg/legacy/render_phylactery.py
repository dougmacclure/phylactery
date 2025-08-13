#!/usr/bin/env python2
import argparse, os, sys
from legacy.phylactery_modern import render_single  # ← your legacy function

ap = argparse.ArgumentParser()
ap.add_argument("--d_real", type=float, required=True)
ap.add_argument("--d_imag", type=float, required=True)
ap.add_argument("--out",     default="/shared/out.png")
cfg = ap.parse_args()

render_single(complex(cfg.d_real, cfg.d_imag), cfg.out)
print("Wrote", cfg.out)
