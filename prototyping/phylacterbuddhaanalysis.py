import numpy as np, matplotlib.pyplot as plt, pathlib

raw  = np.load('C:/Users/dougm/OneDrive/Documents/code/anun_git/data/tempbuffer_glyph_parameter_buddha_array.npy')
data = raw.reshape(5000, 5000).astype(np.float32)

lims = 3.333
N    = data.shape[0]
xs   = np.linspace(-lims,  lims,  N)
ys   = np.linspace(-lims,  lims,  N)
X, Y = np.meshgrid(xs, ys)
D    = X + 1j*Y
z0   = (1 - np.sqrt(1 - 4*D)) / 2
lam  = np.abs(2*z0)

fig, ax = plt.subplots(figsize=(8,8))
im = ax.imshow(np.log10(data+1), cmap='viridis', origin='lower',
               extent=[-lims,lims,-lims,lims])
cs = ax.contour(X, Y, lam, levels=[1.0], colors='white')
#cs.collections[0].set_label('|2 z₀|=1')
ax.legend(loc='upper left')
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('log10(density)')
fig.tight_layout()

out = pathlib.Path('C:/Users/dougm/OneDrive/Documents/code/anun_git/plots/alphaphylactery_parameter_density_overlay.png')
fig.savefig(out, dpi=300)
print("saved →", out)
