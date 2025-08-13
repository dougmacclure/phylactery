import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt

# ---------- 1.  LOAD DENSITY GRID ----------
data_flat        = np.load("C:/Users/dougm/OneDrive/Documents/code/anun/parameter_plane_clustering/beta_glyph_parameter_buddha_array.npy")    # shape (H, W)
D = data_flat.reshape((5000, 5000))

D = np.genfromtxt('C:/Users/dougm/OneDrive/Documents/code/anun/parameter_plane_clustering/glyph_matrix_glyph_P1_X-2.5_Y0.0_erad3.95_lim1e+20_zoom1.0.csv', delimiter=',')


#D = pd.read_csv('C:/Users/dougm/OneDrive/Documents/code/anun/data/parameter_plane/')

#scaler = StandardScaler()
scaler = RobustScaler()
H, W     = D.shape

# --------- 1. load + sparsify -----------

ys, xs   = np.nonzero(D)
w  = D[ys, xs].astype(float)
coords   = np.column_stack([xs, ys]).astype(float)   # pixel coords (x, y)
coords = scaler.fit_transform(coords)


# --------- 2. reservoir-style sampling -----------
# keep ≤ 500k points, but sample proportionally to weight
keep   = 200_000
p      = (w / w.sum())
idx    = np.random.choice(len(coords), size=keep, replace=False, p=p)
coords_sample = coords[idx]
w_sample      = w[idx]
trainingdata = []
for K in range(180):

    if K < 2:
        continue
    # --------- 3. Mini-batch KMeans -----------
    
    mb = MiniBatchKMeans(
            n_clusters   = K,
            batch_size   = 20_000,
            n_init       = "auto",
            max_iter     = 500,
            random_state = 42
        )
    mb.fit(coords_sample, sample_weight=w_sample)
    print('computing inertia')
    inertia = mb.inertia_
    labels = mb.fit_predict(coords_sample)
    print('computing avg silhouette coefficient')
    # Calculate the average    silhouette coefficient
    avg_silhouette = silhouette_score(coords_sample, labels, sample_size=20_000)
    print(f"Average Silhouette Coefficient: {avg_silhouette:.2f} for k = {K}")
    trainingdata.append([K, inertia, avg_silhouette])
trainingdf = pd.DataFrame(trainingdata)
trainingdf.columns =['K', 'Inertia','Avg Silhouette Coeff']
scalermm = MinMaxScaler()


trainingdf['min_max_inertia'] = scalermm.fit_transform(trainingdf[['Inertia']])
plt.figure(figsize=(10, 10))
fig = plt.scatter(x=trainingdf['K'], y = trainingdf['min_max_inertia'])
plt.title("K vs. minmax inertia")
plt.yscale('log')
#plt.imsave("data/minmaxinertia.png")
plt.show()
plt.close()

plt.figure(figsize=(10, 10))
fig = plt.scatter(x=trainingdf['K'], y = trainingdf['Avg Silhouette Coeff'])
plt.title("K vs. Average Silhouette Coefficient")

#plt.imsave("data/minmaxinertia.png")
plt.show()
plt.close()

#Use optimal K = 25 and retrain

keep   = 1_000_000
p      = (w / w.sum())
idx    = np.random.choice(len(coords), size=keep, replace=False, p=p)

km = KMeans(n_clusters=25,
            n_init='auto',
            max_iter=10000,
            random_state=952380)
km.fit(coords, sample_weight= w)

inertia = km.inertia_
labels = km.fit_predict(coords_sample)

centroids = km.cluster_centers_.astype(int)   # K × 2  (x,y)
np.save('centroids2.npy', centroids)

########
# overlay centroids in image

import numpy as np
from skimage.draw import disk
from PIL import Image
import matplotlib.pyplot as plt

# ------------------ USER INPUT ------------------
centroids = np.load('C:/Users/dougm/OneDrive/Documents/code/anun/parameter_plane_clustering/centroids2.npy')  # shape (25, 2)  ints (x_pix, y_pix)
radius    = 20                             # 10-pixel diameter → r = 5
img_path2  = 'C:/Users/dougm/OneDrive/Documents/code/anun/parameter_plane_clustering/glyph_P1_X-2.5_Y0.0_erad3.95_lim1e+20_zoom1.0.png'

img_path1  = 'C:/Users/dougm/OneDrive/Documents/code/anun/parameter_plane_clustering/glyph_buddhaP10.0_Y0.0_erad3.95_lim1e20.png.png'
out_png   = 'C:/Users/dougm/OneDrive/Documents/code/anun/parameter_plane_clustering/overlay2.png'
# -----------------------------------------------

# 0) load base image (assumed 5000×5000 PNG)
base_img = Image.open(img_path2).convert('RGBA')
H, W     = base_img.size[1], base_img.size[0]

# 1) prepare mask canvas (uint8, 0 = background)
mask = np.zeros((H, W), dtype=np.uint8)

# 2) paint each centroid as a filled disk
for idx, (x, y) in enumerate(centroids):
    rr, cc = disk((y, x), radius, shape=mask.shape)
    mask[rr, cc] = idx + 1            # avoid 0 so each ID > 0

# --- OPTIONAL: export raw mask for later use ----
np.save('cluster_mask2.npy', mask)

# 3) build an RGBA overlay for pretty plotting
colours = plt.cm.tab20(np.linspace(0, 1, len(centroids)))  # 25 distinct hues
palette = plt.cm.Set1(np.linspace(0, 1, len(centroids)))
overlay = np.zeros((H, W, 4), dtype=np.uint8)

for idx in range(1, len(centroids)+1):
    rgb  = (palette[idx-1, :3] * 255).astype(np.uint8)
    alpha = 255                                  # 0 transparent … 255 solid
    overlay[mask == idx] = np.concatenate([rgb, [alpha]])

# 4) composite overlay on top of base
overlay_img = Image.fromarray(overlay, mode='RGBA')
composited  = Image.alpha_composite(base_img, overlay_img)
composited.save(out_png)
print(f'Wrote {out_png}, and mask saved to cluster_mask.npy')

