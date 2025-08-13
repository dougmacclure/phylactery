#!/usr/bin/env python3
"""Call the legacy VM, fetch out.png, convert to GR-ERG band vector."""
import subprocess, pathlib, time, math
from PIL import Image
import numpy as np
#VMX   = r"C:\VMs\PhylacteryBeetle\Phylactery.vmx"   # your .vmx path
VMX = r"C:\Users\dougm\OneDrive\Desktop\LinuxMint_variant\LinuxMint_variant.vmx"
SHARE = r"C:\vmshare"   
VM_NAME    = "PhylacteryVM"          # VirtualBox machine name
SHARED_DIR = pathlib.Path("~/vmshare").expanduser()
OUT_FILE   = SHARED_DIR / "out.png"

def render_d(d: complex):
    # Clean previous output
    if OUT_FILE.exists(): OUT_FILE.unlink()
    cmd = [
        "vboxmanage", "guestcontrol", VM_NAME, "run",
        "--username", "legacy", "--password", "secret",
        "--exe", "/home/legacy/render_phylactery.py",
        "--",
        "--d_real", str(d.real),
        "--d_imag", str(d.imag),
        "--out", "/shared/out.png",
    ]
    subprocess.run(cmd, check=True)
    # Wait for PNG
    for _ in range(120):
        if OUT_FILE.exists():
            return OUT_FILE
        time.sleep(0.5)
    raise RuntimeError("VM did not produce out.png in time")

def png_to_bands(png_path) -> dict:
    """Toy mapping: angle→Valence, radius→Arousal, rest neutral."""
    im = Image.open(png_path).convert("L")   # grayscale
    arr = np.asarray(im, dtype=np.float32) / 255.0
    valence  = (np.mean(arr) - 0.5) * 2      # -1 … 1
    arousal  = np.std(arr) * 2               # 0 … 1.4 approx
    return dict(valence=valence,
                arousal=arousal,
                dominance=0.0,
                curiosity=0.0,
                selfref=0.0)

def get_bands_for_d(d: complex):
    png = render_d(d)
    return png, png_to_bands(png)

# quick demo
if __name__ == "__main__":
    d = -3.75 + 0j
    png, bands = get_bands_for_d(d)
    print("PNG saved →", png)
    print("Bands:", bands)
