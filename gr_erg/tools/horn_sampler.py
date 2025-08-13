"""
horn_sampler.py  –  pick a parameter d on |2 z0|=1, render it in the
legacy VM, return (png_path, band_dict).

Usage example
-------------
from horn_sampler import sample_horn

png, bands = sample_horn(t=0.35)   # t ∈ [0,1] sweeps the horn once
update_gerg_with_band(bands)       # your existing call
show_inline(png)                   # optional preview
"""
import numpy as np, subprocess, time, pathlib, math, cmath
from vm_glyph_client import png_to_bands          # we wrote this earlier
from vm_glyph_client import render_d               # + render_d(d) helper

VM_NAME    = "PhylacteryVM"
SHARED_DIR = pathlib.Path("~/vmshare").expanduser()
PNG_FILE   = SHARED_DIR / "out.png"
ERAD       = 3.95            # keep in sync with VM script

def horn_param(t: float) -> complex:
    """
    Map t ∈ [0,1] to a point on |2 z0| = 1 with Re d > 0.
    We solve |2 z0(d)| = 1 numerically for each angle φ.
    """
    phi = 2 * math.pi * t          # full loop; adjust if you want half-horn
    # initial radius guess based on large-|d| asymptotics
    r = 2.0
    for _ in range(15):            # Newton iterations
        d = r * math.exp(1j*phi)
        z0 = (1 - cmath.sqrt(1 - 4*d)) / 2
        f  = abs(2*z0) - 1
        if abs(f) < 1e-6:
            return d
        # d|f|/dr ≈ sign(f)   crude slope
        r -= f * 0.5
    return d                       # fallback

def sample_horn(t: float):
    """Render horn point for fraction t, return (png, bands)."""
    d = horn_param(t)
    png = render_d(d)              # VM call from vm_glyph_client.py
    bands = png_to_bands(png)
    return png, bands
