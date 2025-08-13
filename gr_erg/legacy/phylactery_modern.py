
# phylactery_modern.py (Python 3.12 conversion of the original bifurcation glyph generator)

import numpy as np
from PIL import Image, ImageDraw
import ctypes
import os
from importlib import import_module

_phyl = import_module("gr_erg.legacy._phylactery_cpp")

def iterate_point(real, imag, maxit, c_real, c_imag, plmin, erad, lim1):
    # call C++ core
    return _phyl.iterate(real, imag, maxit,
                         c_real, c_imag,
                         plmin, erad, lim1)
# Par
# ameters
d = 1

cutoff = 1e3
iters = 1500

# Load compiled C function (assuming it’s compiled to phylactery.so)
# lib = ctypes.CDLL("gr_erg/legacy/phylactery.so")
# lib.iterate.restype = ctypes.c_double
# lib.iterate.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double]

def dist_from_mandelbrot(c):
    z = 0
    for n in range(1000):
        z = z*z + c
        if abs(z) > 2:
            return n  # escape time
    return 1000  # likely inside

def generate_bifurcation_image(c_real=-3.75, c_imag = 0.0, plmin = 1, erad = 3.95, lim1 = 1.0e+20, width = 4000, height = 4000, scale = 5.0, center_x = 0.0, center_y = 0.0):
    img = Image.new("RGB", (width, height), (0,0,0))
    
    #pixels = img.load()
    draw = ImageDraw.Draw(img)
    blkcntr = 0
    cntr = 0
    for x in range(width):
        for y in range(height):
            cntr +=1
            # Map pixel to complex plane
            re = (x - width / 2) * (scale / width) + center_x
            im = (y - height / 2) * (scale / height) + center_y

            # Call compiled C function
            checkset = iterate_point(re, im, 1500, c_real, c_imag, plmin, erad, lim1)

            # Simple coloring (can be enhanced for glyphic aesthetic)
            #color_val = int(255 * min(1, iters / cutoff))
            if checkset == 0:
                draw.point((x,y), fill=(0,0,0))
                blkcntr +=1	


            elif checkset == -iters:
                
                draw.point((x,y), fill=(0,0,255))
                
            else:
            
                r, g, b, = 0, 0, 0
                
                r = checkset%139 + checkset%109 + checkset%7
                b = checkset%131 + checkset%107 + checkset%17
                g = checkset%23 + checkset%31 + checkset%53 + checkset%113 + checkset%2 + checkset%3 + checkset%17 + checkset%13
                draw.point((x, y), fill=(r,g,b)) 
            if blkcntr % 1000 == 1:
                print(str(blkcntr) + " number of 'stable' points.  Percent of stable points = " + str(100*blkcntr/cntr))
                blkcntr += 1
            #pixels[x, y] = (color_val, color_val, 255 - color_val)

        print(x)

    signature = "glyph_P" + str(plmin) + "_cRe" + str(c_real) + "_Im" + str(c_imag) + str(erad) + "_lim" + str(lim1)
    filename = signature + ".png"
    img.save(filename, "PNG")	

if __name__ == "__main__":
    generate_bifurcation_image(-10.0, 0.01, 1, 3.95, 1.0e+14)