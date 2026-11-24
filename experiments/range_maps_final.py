"""Before/after range maps for the report, zoomed to each species' region.

Columns: expert range + training sightings | nearest sighting |
         network (location only) | network (location + land flag).
Rows: the same three species as the error analysis (chosen automatically
there: network's biggest gain, biggest loss, and a typical rare species).
Needs results/metrics/per_species_ap.csv and the saved predictions.
"""
import sys
sys.path.append('.')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from global_land_mask import globe
from src.data import load_all
from src.evaluate import per_species_ap

INK, MUTED, SURFACE = '#0b0b0b', '#898781', '#fcfcfb'
LAND, SEA = '#e6e5e0', '#fcfcfb'
PANELS = [('Expert range + training sightings', None),
          ('Nearest sighting', 'nearest_sighting'),
          ('Network: location only', 'shared_mlp_full'),
          ('Network: location + land flag', 'shared_mlp_without_extra_species')]

d = load_all()
Y, locs = d['Y_test'], d['test_locs']
names = [d['names'][sp] for sp in d['species']]
preds = {m: np.load(f'results/predictions/{m}.npy', mmap_mode='r')
         for _, m in PANELS if m}

# Same species as the error analysis
t = pd.read_csv('results/metrics/per_species_ap.csv')
t['gap'] = t['shared_mlp_full'] - t['nearest_sighting']
land_sp = t[t['ocean_share'] < 0.5]
rows = {'Network gains most': int(t['gap'].idxmax()),
        'Network loses most': int(t['gap'].idxmin()),
        'Rare land species': int(land_sp[land_sp['n_train'] < 100]['shared_mlp_full']
                                 .sub(land_sp['shared_mlp_full'].median()).abs().idxmin())}


def region(j, margin=12.0, min_size=30.0):
    """Lat/lon box around a species' expert range and training sightings."""
    pts = np.concatenate([locs[Y[:, j]], d['train_locs'][d['train_cols'] == j]])
    lat0, lat1 = pts[:, 0].min() - margin, pts[:, 0].max() + margin
    lon0, lon1 = pts[:, 1].min() - margin, pts[:, 1].max() + margin
    # make the box at least min_size wide/tall, then roughly square
    size = max(min_size, lat1 - lat0, lon1 - lon0)
    clat, clon = (lat0 + lat1) / 2, (lon0 + lon1) / 2
    return (max(clat - size / 2, -90), min(clat + size / 2, 90),
            max(clon - size / 2, -180), min(clon + size / 2, 180))


def draw_land(ax, box):
    lat0, lat1, lon0, lon1 = box
    la = np.linspace(lat1, lat0, 300)
    lo = np.linspace(lon0, lon1, 300)
    LO, LA = np.meshgrid(lo, la)
    ax.imshow(globe.is_land(LA, LO), extent=[lon0, lon1, lat0, lat1],
              cmap=ListedColormap([SEA, LAND]), interpolation='nearest', zorder=0)


fig, axes = plt.subplots(3, 4, figsize=(12, 9), dpi=200)
fig.patch.set_facecolor(SURFACE)
for r, (row_title, j) in enumerate(rows.items()):
    box = region(j)
    lat0, lat1, lon0, lon1 = box
    inside = ((locs[:, 0] >= lat0) & (locs[:, 0] <= lat1) &
              (locs[:, 1] >= lon0) & (locs[:, 1] <= lon1))
    sub = locs[inside]
    for c, (title, m) in enumerate(PANELS):
        ax = axes[r, c]
        draw_land(ax, box)
        if m is None:
            pos = Y[inside, j]
            ax.scatter(sub[pos, 1], sub[pos, 0], s=7, c='#86b6ef', marker='s',
                       lw=0, zorder=2, label='Expert range (test)')
            tr = d['train_locs'][d['train_cols'] == j]
            ax.scatter(tr[:, 1], tr[:, 0], s=4, c='#0d366b', lw=0, zorder=3,
                       label='Training sightings')
            ax.set_ylabel(f'{row_title}\n{names[j]}\n({int(t.loc[j, "n_train"])} sightings)',
                          fontsize=7, color=INK)
        else:
            sc = np.asarray(preds[m][:, j])
            ap = per_species_ap(np.asarray(preds[m][:, j:j + 1]), Y[:, j:j + 1])[0]
            s = sc[inside]
            show = s > 0.05 * max(s.max(), 1e-6)      # hide near-zero scores so land shows
            order = np.argsort(s[show])
            pts, vals = sub[show][order], s[show][order]
            ax.scatter(pts[:, 1], pts[:, 0], s=7, c=vals, cmap='Blues',
                       vmin=0, vmax=max(s.max(), 1e-6), marker='s', lw=0, zorder=2)
            
            ax.text(0.02, 0.03, f'AP {ap:.2f}', transform=ax.transAxes, fontsize=8,
                    color=INK, zorder=4,
                    bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.5, alpha=0.9))
        if r == 0:
            ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(LAND)
axes[0, 0].legend(frameon=False, fontsize=6, loc='upper left', markerscale=2)
fig.text(0.5, 0.005, 'Darker blue = higher predicted score. Grey = land. '
         'Each panel is zoomed to the species\' region.', ha='center',
         fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
Path('results/figures').mkdir(parents=True, exist_ok=True)
fig.savefig('results/figures/range_maps_final.png', facecolor=SURFACE)
print('Saved results/figures/range_maps_final.png')
for row_title, j in rows.items():
    print(f'{row_title}: {names[j]}')