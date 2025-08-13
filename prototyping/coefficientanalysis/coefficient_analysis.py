import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import seaborn as sns
import ast
from numpy.linalg import eig


def plot_eigenvalue_field():
    # Load the data
    df = pd.read_csv("c:\\Users\\dougm\\OneDrive\\Documents\\code\\anun_git\\data\\eigendf.csv")

    # Parse eigenvalues column into complex numbers
    eigenvals = df['eigenvalues'].apply(lambda z: complex(z.strip("()"))).values

    # Real & Imaginary parts
    real_vals = eigenvals.real
    imag_vals = eigenvals.imag

    # Histogram settings
    res = 600
    lim = 4  # Adjust based on your λ spread

    hist, xedges, yedges = np.histogram2d(
        real_vals, imag_vals,
        bins=res,
        range=[[-lim, lim], [-lim, lim]]
    )

    # Smooth with Gaussian
    smoothed = gaussian_filter(hist, sigma=2.5)

    # Plot
    plt.figure(figsize=(10, 8))
    # plt.imshow(smoothed.T, origin="lower",
    #            extent=[-lim, lim, -lim, lim],
    #            cmap="magma", aspect="equal")
    plt.imshow(
        np.log1p(smoothed.T),  # log1p = log(1 + x), safe for zero
        origin="lower",
        extent=[-lim, lim, -lim, lim],
        cmap="magma",
        aspect="equal"
    )
    plt.colorbar(label="Eigenvalue Density")
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title("Smoothed Eigenvalue Field of Phylactery Map")
    plt.tight_layout()
    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def psychoanalyze(df):
    results = []

    for i, row in df.iterrows():
        try:
            # Parse eigenvalues
            lam = complex(row['eigenvalues'].strip("()"))
            re = lam.real
            im = lam.imag
            mag = abs(lam)
            norm = row['frobenius_norm']
            
            # VALENCE: positive vs negative growth
            valence = np.tanh(re) * sigmoid(norm)

            # AROUSAL: how jittery
            arousal = np.tanh(abs(im)) * np.log1p(norm)

            # DOMINANCE: controlled or dispersed
            dominance = (abs(re) / (norm + 1e-6)) * sigmoid(norm)

            # CURIOSITY: indirect measure of eigenvector wandering
            # crude: if |λ| ≠ 1, it's exploring
            curiosity = np.abs(mag - 1) * sigmoid(abs(im))

            # SELF-REFERENCE: how close to fixed-point recursion
            selfref = sigmoid(1 - abs(mag - 1)) * (1 / (1 + np.abs(im)))

            results.append({
                "valence": valence,
                "arousal": arousal,
                "dominance": dominance,
                "curiosity": curiosity,
                "self_reference": selfref
            })
        except Exception as e:
            results.append({
                "valence": np.nan,
                "arousal": np.nan,
                "dominance": np.nan,
                "curiosity": np.nan,
                "self_reference": np.nan
            })

    return pd.DataFrame(results)


def plot_emotional_metrics_by_eigenvalue(dfall):
    df = dfall.sample(4000)
    df_emotions = psychoanalyze(df)
    # Merge emotional metrics back with your original df
    df_combined = pd.concat([df, df_emotions], axis=1)

    # === 1. Visualize distributions of each ER metric ===
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(["valence", "arousal", "dominance", "curiosity", "self_reference"]):
        plt.subplot(2, 3, i+1)
        sns.histplot(df_combined[col], kde=True, bins=60, color="orchid")
        plt.title(col)
    plt.tight_layout()
    plt.show()

    # === 2. Pairwise scatterplots for correlation ===
    sns.pairplot(
        df_combined[["valence", "arousal", "dominance", "curiosity", "self_reference"]],
        corner=True,
        plot_kws={'alpha': 0.1, 's': 5}
    )
    plt.suptitle("Pairwise ER Metric Relationships", y=1.02)
    plt.show()

    # === 3. Heatmap of correlations ===
    plt.figure(figsize=(8, 6))
    corr = df_combined[["valence", "arousal", "dominance", "curiosity", "self_reference"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Emotional Metrics")
    plt.show()

    # === 4. Emotional landscape over parameter space Re(c), Im(c) ===
    # Unpack 'c' if it's stored as a string like "(a+bj)"
    df_combined['c_real'] = df_combined['c'].apply(lambda z: complex(z).real)
    df_combined['c_imag'] = df_combined['c'].apply(lambda z: complex(z).imag)

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        df_combined['c_real'], df_combined['c_imag'],
        c=df_combined['valence'],
        cmap="viridis", s=1, alpha=0.5
    )
    plt.colorbar(sc, label="Valence")
    plt.title("Valence Field over Complex c-Plane")

    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

def flatten_and_analyze(df):
    records = []
    cntr = 0
    for i, row in df.iterrows():
        
        try:
            eigen_list = eval(row['eigenvalues'])  # list of eigenvalues per glyph
            norm = row['frobenius_norm']
            c = row['c']
            for lam in eigen_list:
                lam = complex(lam)
                re, im, mag = lam.real, lam.imag, abs(lam)
                
                valence = np.tanh(re) * sigmoid(norm)
                arousal = np.tanh(abs(im)) * np.log1p(norm)
                dominance = (abs(re) / (norm + 1e-6)) * sigmoid(norm)
                curiosity = np.abs(mag - 1) * sigmoid(abs(im))
                selfref = sigmoid(1 - abs(mag - 1)) * (1 / (1 + np.abs(im)))

                records.append({
                    "Re_lambda": re,
                    "Im_lambda": im,
                    "mag_lambda": mag,
                    "valence": valence,
                    "arousal": arousal,
                    "dominance": dominance,
                    "curiosity": curiosity,
                    "self_reference": selfref,
                    "c": c
                })
        except:
            continue
        if cntr % 1000 == 0:
            print(f"Processed {100*cntr/df.shape[1]} % of rows" )
    cntr += 1
    return pd.DataFrame(records)


def parse_eigenvalues(val):
    try:
        return [complex(val)]
    except:
        return []


def density_by_metric(df_eigs, metric='valence'):
    plt.figure(figsize=(10, 8))
    sns.kdeplot(
        data=df_eigs,
        x="Re_lambda",
        y="Im_lambda",
        fill=True,
        cmap="viridis",
        levels=100,
        thresh=0.01
    )
    sc = plt.scatter(
        df_eigs["Re_lambda"],
        df_eigs["Im_lambda"],
        c=df_eigs[metric],
        cmap="coolwarm",
        s=1,
        alpha=0.4
    )
    plt.colorbar(sc, label=metric)
    plt.title(f"Eigenvalue field colored by {metric}")
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def main():
    # Load the data
    df = pd.read_csv("c:\\Users\\dougm\\OneDrive\\Documents\\code\\anun_git\\data\\eigendf.csv")
    
    # Ensure 'c' is parsed as a complex number
    df['c'] = df['c'].apply(lambda x: complex(x.strip("()")))

    # Group by 'c' (each group is one glyph)
    grouped = df.groupby('c')

    # Sample N glyphs
    sampled_c = np.random.choice(list(grouped.groups.keys()), size=1000, replace=False)
    # Build coefficient matrix: each row is a vector of coefficients for one c
    coeff_matrix = np.stack([
        grouped.get_group(c_val).sort_index()['coefficients'].apply(lambda z: complex(z.strip("()"))).values
        for c_val in sampled_c
    ])
    
    #coeffs = df_sample['coefficients'].apply(lambda z: np.array(ast.literal_eval(z))).values
    # Construct Gram matrix
    #G = np.outer(coeffs, np.conj(coeffs))  # complex Gram matrix
    # Construct complex Gram matrix
    G = coeff_matrix @ coeff_matrix.conj().T  # (N x N) matrix
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(G)

    df["eigen_list"] = df["eigenvalues"].apply(parse_eigenvalues)
    # Flatten the eigenvalues and analyze
    flat_records = []
    cntr = 0
    for i, row in df.iterrows():
        c_val = complex(row['c'])
        norm = row['frobenius_norm']
        for lam in row['eigen_list']:
            re, im, mag = lam.real, lam.imag, abs(lam)
            
            valence = np.tanh(re) * (1 / (1 + np.exp(-norm)))
            arousal = np.tanh(abs(im)) * np.log1p(norm)
            dominance = (abs(re) / (norm + 1e-6)) * (1 / (1 + np.exp(-norm)))
            curiosity = np.abs(mag - 1) * (1 / (1 + np.exp(-abs(im))))
            selfref = (1 / (1 + np.exp(-(1 - abs(mag - 1))))) * (1 / (1 + np.abs(im)))

            flat_records.append({
                "Re_lambda": re,
                "Im_lambda": im,
                "valence": valence,
                "arousal": arousal,
                "dominance": dominance,
                "curiosity": curiosity,
                "self_reference": selfref,
                "c_real": c_val.real,
                "c_imag": c_val.imag
            })
        if cntr % 1000 == 0:
                print(f"Processed {100*cntr/df.shape[0]} % of rows" )
        cntr += 1
    df_eigs = pd.DataFrame(flat_records)
    emo_metrics = ["valence", "arousal", "dominance", "curiosity", "self_reference"]
    for metric in emo_metrics:
        density_by_metric(df_eigs, metric=metric)
    