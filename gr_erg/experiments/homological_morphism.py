
import gudhi.euclidean_witness_complex
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

import gudhi
from gudhi import rips_complex, witness_complex

from gudhi import euclidean_witness_complex as EuclideanWitnessComplex
from gudhi import pick_n_random_points
from gudhi.persistence_graphical_tools import plot_persistence_diagram
# --- Parameters ---
THRESHOLD       = 1.0
MAX_EDGE_LENGTH = 0.1
MAX_DIM         = 2
NUMPTS          = 10000
LANDMARK_FACTOR = 50   # nb_points = NUMPTS // LANDMARK_FACTOR
# --- Step 1: Load and Threshold Image ---
def extract_point_cloud(img_path, threshold=THRESHOLD):
    img = io.imread(img_path)
    gray = color.rgb2gray(img)
    points = np.column_stack(np.where(gray > threshold))
    if len(points) > NUMPTS:
        points = points[np.random.choice(len(points), NUMPTS, replace=False)]
    return points / np.max(points)

# --- Step 2: Compute Persistence Diagram ---
# def compute_betti_diagram_deprecated(points, max_dim=MAX_DIM):
#     #rips = rips_complex.RipsComplex(points=points, max_edge_length=MAX_EDGE_LENGTH)
#     witnesses = points
#     landmarks = gudhi.pick_n_random_points(points=witnesses, nb_points=int(NUMPTS/50))

#     rips = gudhi.EuclideanWitnessComplex(witnesses=witnesses, landmarks=landmarks)
    
#     try:
#         simplex_tree = rips.create_simplex_tree(max_alpha_square = MAX_EDGE_LENGTH**2, limit_dimension=max_dim)  
#         print('simplex tree computed')
#         diag = simplex_tree.persistence()
#         print('persistence computed')
#     except Exception as e:
#         print("Error during persistence:", e)
#         return
#     return diag, simplex_tree
# 2) Build witness complex and compute persistence
def compute_betti_diagram(points, max_dim=MAX_DIM):
    # pick landmarks
    witnesses = points
    print(points)
    landmarks = gudhi.pick_n_random_points(points=witnesses, nb_points=int(NUMPTS/LANDMARK_FACTOR))
    
    # build the Euclidean witness complex
    wc = gudhi.euclidean_witness_complex.EuclideanWitnessComplex(witnesses=points, landmarks=landmarks)
    # use squared radius here:
    st = wc.create_simplex_tree(
        max_alpha_square=MAX_EDGE_LENGTH**2,
        limit_dimension=max_dim
    )

    print(f"  • Landmarks: {len(landmarks)}")
    print(f"  • Simplex count: {st.num_simplices()}")

    # compute persistence
    diag = st.persistence()
    return diag, st

# 3) Plot
def plot_diagram(diag, title="Persistence Diagram"):
    plt.figure(figsize=(6,6))
    plot_persistence_diagram(diag)
    plt.title(title)
    plt.savefig("alpha_persistenceDiagram.png")
    plt.show()


# --- Main Execution ---
if __name__ == '__main__':
    image_path = "C:/Users/dougm/OneDrive/Documents/code/anun_git/preprints/figure1.png"  # Change this to your path
    print('image extracted')
    #points = extract_point_cloud(image_path)
    points = np.load('C:/Users/dougm/OneDrive/Documents/code/anun/data/alpha_glyph_parameter_buddha_array.npy')
    points = points.reshape(5000,5000)
    print('point cloud extracted')
    diag, simplex_tree = compute_betti_diagram(points)
    
    plot_diagram(diag, title="PhylacteryBuddha Persistence Diagram")
    print('saving data')
    np.save("./alpha_persistence.npy", diag)
    #np.save("./persistence.npy", diag)