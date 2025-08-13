import numpy as np, cmath

def rotation_number(c):
    # pick the repelling root z0(c)
    z1, z2 = compute_fixed_points(c)
    # decide which one has |f'(z)|>1
    z0 = z1 if abs(2*z1) > 1 else z2
    θ = np.angle(2*z0)
    return (θ / (2*np.pi)) % 1

# grid of angles
thetas = np.linspace(0,2*np.pi,400,endpoint=False)
rhos = [rotation_number(2.0*cmath.exp(1j*t)) for t in thetas]

import matplotlib.pyplot as plt
plt.plot(np.degrees(thetas), rhos, '.')
plt.xlabel("θ (deg) of c=r·e^{iθ}")
plt.ylabel("rotation number ρ")
plt.axhline(0.5, color='red', ls='--', label='ρ=1/2')
plt.legend()
plt.show()
