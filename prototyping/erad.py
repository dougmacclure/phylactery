import numpy as np
import matplotlib.pyplot as plt

# Function: escape_time or custom P_c orbital test goes here
def test_function(c, erad):
    # Placeholder: Replace with your P_c logic
    z = 0
    for i in range(1000):
        z = z**2 + c
        if abs(z) > erad:
            return i
    return 1000

# Sweep Parameters
center = 0.25
r_vals = np.linspace(0.015, 0.025, 50)   # Delta sweep
theta = 0  # Aligned with positive real axis (adjustable)
erad = 10.0  # Base escape radius (tweakable)

results = []

for r in r_vals:
    delta_c = r * np.exp(1j * theta)
    c = center + delta_c
    val = test_function(c, erad)
    results.append(val)

# Visualization
plt.figure(figsize=(8, 4))
plt.plot(r_vals, results, marker='o', lw=2)
plt.xlabel('Delta r (radial offset from 0.25)')
plt.ylabel('Escape Value or Resonance Proxy')
plt.title('δ-Sweep Resonance Map near c = 0.25')
plt.grid(True)
plt.tight_layout()
plt.show()