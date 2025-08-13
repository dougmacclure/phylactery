import numpy as np
import matplotlib.pyplot as plt
# Parameters
z0 = complex(-1.5, 0.0)
# Re-import required modules due to execution state reset
import numpy as np
import pandas as pd

# Generate grid of c-values around Mandelbrot boundary
#num_radii = 100
#num_angles = 48
numpts = 200
#radii = np.linspace(1.2, 2.5, num_radii)
#radii = np.linspace(0.0, 2.5, num_radii)
#angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

linear_range = np.linspace(-2.5, 2.5, numpts)

# Create complex grid
#c_values = [r * np.exp(1j * theta) for r in radii for theta in angles]
c_values = [complex(r, i) for r in linear_range for i in linear_range]
c_coords = [(np.real(c), np.imag(c)) for c in c_values]

# Package into DataFrame for tracking
c_grid_df = pd.DataFrame(c_coords, columns=["Re(c)", "Im(c)"])
#c_grid_df["c"] = [f"{c.real:.4f}+{c.imag:.4f}i" for c in c_values]

# Show a preview of the sampling scheme
#import ace_tools as tools
#tools.display_dataframe_to_user(name="Phylactery Sampling Grid (Initial Sweep)", dataframe=c_grid_df.head(20))

#darray = c_grid_df["c"].values
eigenvaluelist = []
egenveclist = []
darray = c_values
eigendf = pd.DataFrame()
cntr = 0
for d in darray:    
    #d = -3.75
    plmin = 1

    # Define a_0
    if plmin == 1:
        a0 = (1.0 + np.sqrt(1.0 - 4.0 * d)) / 2.0
    else:
        a0 = (1.0 - np.sqrt(1.0 - 4.0 * d)) / 2.0
    a0 = complex(a0, 0.0)

    # Define a_1
    a1 = -d / (2.0 * a0 - (2.0 * z0))

    # Compute coefficients up to a_25
    coefficients = [a0, a1]
    for n in range(2, 100):
        sum_term = 0
        for i in range(1, n):
            mult = 2.0 if i != n - i else 1.0
            sum_term += mult * coefficients[i] * coefficients[n - i]
        denom = 2.0 * a0 - (2.0 * z0) ** n
        an = (-d - sum_term) / denom
        coefficients.append(an)
    
    # Convert to numpy array
    coeff_array = np.array(coefficients)

    # Compute Gram matrix and eigenvalues
    gram_matrix = np.outer(coeff_array, np.conj(coeff_array))
    
    eigenvalues = np.linalg.eigvalsh(gram_matrix)
    
    dstr = str(np.real(d)) + '_' + str(np.imag(d))
    # Save outputs
    #np.save("./data/gram_eigenvalues"+dstr + ".npy", eigenvalues)
    #np.save("./data/coefficients"+dstr + ".npy", coeff_array)
    #print("Eigenvalues and coefficients saved.")

    #coeff_array = np.load("./data/coefficients"+dstr + ".npy")
    #eigenvalues = np.load("./data/gram_eigenvalues"+dstr + ".npy")
    sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

    eps = 1e-16
    normalized = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    spectral_entropy = -np.sum(normalized * np.log(normalized + eps))
    condition_number = sorted_eigenvalues[0] / (sorted_eigenvalues[-1] + eps)
    frobenius_norm = np.sqrt(np.sum(eigenvalues**2))
    tempdf = pd.DataFrame(data = [coeff_array, eigenvalues]).T
    tempdf.columns = ["coefficients", "eigenvalues"]
    tempdf["c"] = d
    tempdf["frobenius_norm"] = frobenius_norm  
    #eigenvaluelist.append([d, coeff_array, eigenvalues, frobenius_norm])
    eigendf = pd.concat([eigendf, tempdf], ignore_index=True)
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(sorted_eigenvalues)+1), sorted_eigenvalues, 'ro-')
    # plt.yscale('log')
    # plt.xlabel('Index')
    # plt.ylabel('Eigenvalue Magnitude (log scale)')
    # plt.title('Resonance Spectrum of Gram Matrix (a₀ to a₂₅) for c = ' + dstr)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    cntr += 1
    if cntr % 100 == 0:
        print("Completed", cntr, "of", len(darray), "c-values.")
    #print("Spectral Entropy:", spectral_entropy)
    #print("Condition Number:", condition_number)
    #print("Frobenius Norm:", frobenius_norm)
    #print("Eigenvalue Count:", len(sorted_eigenvalues))
#eigendf = pd.DataFrame(eigenvaluelist, columns=["c", "coefficients", "eigenvalues", "frobenius_norm"])
list_df = [eigendf[i:(i+1)int(len(eigendf)/10)] for i in range(10)]
for i in range(10):

    list_df[i].to_csv('data/eigendf' + str(i) +'.csv',index=False)