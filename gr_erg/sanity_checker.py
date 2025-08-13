from phylactery_core.coeffs import gen_coeffs
from phylactery_core.pd_eval import P_ext
d = -3.75+0j
coeffs, z0 = gen_coeffs(-3.75+0j, 40)
z  = 1+0j          # clearly outside erad
out = P_ext(z, -3.75+0j, coeffs=coeffs, z0=z0)
print(out)         # should differ from z