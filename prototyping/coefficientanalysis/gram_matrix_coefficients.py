import numpy as np
import matplotlib.pyplot as plt
# Parameters
z0 = complex(-1.5, 0.0)
darray = [complex(-3.75,0), complex(-4.0,-0.15), complex(4.0, 0.15), complex(0.0, 2.5), complex(0.0, -2.5)]
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
    np.save("./data/gram_eigenvalues"+dstr + ".npy", eigenvalues)
    np.save("./data/coefficients"+dstr + ".npy", coeff_array)
    print("Eigenvalues and coefficients saved.")

    coeff_array = np.load("./data/coefficients"+dstr + ".npy")
    eigenvalues = np.load("./data/gram_eigenvalues"+dstr + ".npy")
    sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

    eps = 1e-16
    normalized = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    spectral_entropy = -np.sum(normalized * np.log(normalized + eps))
    condition_number = sorted_eigenvalues[0] / (sorted_eigenvalues[-1] + eps)
    frobenius_norm = np.sqrt(np.sum(eigenvalues**2))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sorted_eigenvalues)+1), sorted_eigenvalues, 'ro-')
    plt.yscale('log')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue Magnitude (log scale)')
    plt.title('Resonance Spectrum of Gram Matrix (a₀ to a₂₅) for c = ' + dstr)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Spectral Entropy:", spectral_entropy)
    print("Condition Number:", condition_number)
    print("Frobenius Norm:", frobenius_norm)
    print("Eigenvalue Count:", len(sorted_eigenvalues))