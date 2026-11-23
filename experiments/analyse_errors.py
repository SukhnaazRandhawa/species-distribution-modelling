"""Error analysis: which species and places do the models get wrong?

Needs the saved predictions from experiments/make_predictions.py.
Writes results/metrics/per_species_ap.csv and two figures.
"""
import sys
sys.path.append('.')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import load_all
from src.features import land_mask
from src.evaluate import per_species_ap

MODELS = {'nearest_sighting': 'Nearest sighting',
          'random_forest': 'Random forest',
          'shared_mlp_full': 'Shared network'}
COLORS = {'nearest_sighting': '#2a78d6', 'random_forest': '#eb6834',
          'shared_mlp_full': '#1baf7a'}
INK, MUTED, SURFACE = '#0b0b0b', '#898781', '#fcfcfb'

d = load_all()
Y, locs = d['Y_test'], d['test_locs']
land = land_mask(locs)
names = [d['names'][sp] for sp in d['species']]
preds = {m: np.load(f'results/predictions/{m}.npy', mmap_mode='r') for m in MODELS}

# ---------------- 1. Per-species table ----------------
n_train = np.bincount(d['train_cols'], minlength=Y.shape[1])
n_pos = Y.sum(0)
ocean_share = (Y & ~land[:, None]).sum(0) / n_pos   # share of range in the ocean
table = pd.DataFrame({'species': names, 'n_train': n_train,
                      'range_size': n_pos, 'ocean_share': ocean_share})
for m in MODELS:
    table[m] = per_species_ap(preds[m], Y)
Path('results/metrics').mkdir(parents=True, exist_ok=True)
table.to_csv('results/metrics/per_species_ap.csv', index=False)

pd.set_option('display.width', 140)
print('Per-species AP spread (shared network):')
print(table['shared_mlp_full'].describe().round(3).to_string(), '\n')

# ---------------- 2. Rare vs common ----------------
bins = [50, 100, 250, 1000, 2001]
labels = ['50-99', '100-249', '250-999', '1000-2000']
table['rarity'] = pd.cut(table['n_train'], bins, right=False, labels=labels)
by_rarity = table.groupby('rarity', observed=True)[list(MODELS)].mean()
by_rarity.insert(0, 'n_species', table.groupby('rarity', observed=True).size())
print('Mean AP by number of training sightings:')
print(by_rarity.round(3).to_string(), '\n')

# ---------------- 3. Marine vs land species ----------------
table['habitat'] = np.where(table['ocean_share'] > 0.5, 'mostly ocean', 'mostly land')
by_habitat = table.groupby('habitat')[list(MODELS)].mean()
by_habitat.insert(0, 'n_species', table.groupby('habitat').size())
print('Mean AP by where the species lives:')
print(by_habitat.round(3).to_string(), '\n')

# ---------------- 4. Where does the network win / lose? ----------------
table['net_minus_nearest'] = table['shared_mlp_full'] - table['nearest_sighting']
cols = ['species', 'n_train', 'ocean_share', 'nearest_sighting',
        'shared_mlp_full', 'net_minus_nearest']
print('Network loses most vs nearest sighting:')
print(table.nsmallest(10, 'net_minus_nearest')[cols].round(3).to_string(index=False), '\n')
print('Network wins most vs nearest sighting:')
print(table.nlargest(10, 'net_minus_nearest')[cols].round(3).to_string(index=False), '\n')
print(f"Network better for {(table['net_minus_nearest'] > 0).sum()} "
      f"of {len(table)} species\n")

# ---------------- Figure 1: AP by rarity ----------------
Path('results/figures').mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
x = np.arange(len(by_rarity))
w = 0.26
for i, m in enumerate(MODELS):
    vals = by_rarity[m].values
    bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=COLORS[m],
                  label=MODELS[m], zorder=3)
    for b, v in zip(bars, vals):                     # value labels (relief rule)
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.2f}',
                ha='center', va='bottom', fontsize=6.5, color=INK)
ax.set_xticks(x, [f'{l}\n({n} species)' for l, n in
                  zip(by_rarity.index, by_rarity['n_species'])], fontsize=8, color=INK)
ax.set_ylabel('Mean average precision', fontsize=8, color=INK)
ax.set_xlabel('Training sightings per species', fontsize=8, color=INK)
ax.set_ylim(0, 1)
ax.tick_params(colors=MUTED, labelsize=8)
ax.grid(axis='y', color='#e6e5e0', linewidth=0.6, zorder=0)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
for s in ['left', 'bottom']: ax.spines[s].set_color(MUTED)
ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper left')
ax.set_title('Test AP by species rarity', fontsize=10, color=INK, loc='left')
fig.tight_layout()
fig.savefig('results/figures/ap_by_rarity.png', facecolor=SURFACE)
plt.close(fig)

# ---------------- Figure 2: range maps ----------------
mid = table[(table['ocean_share'] < 0.5)]
examples = {
    'Network gains most': table['net_minus_nearest'].idxmax(),
    'Network loses most': table['net_minus_nearest'].idxmin(),
    'Rare land species': mid[mid['n_train'] < 100]['shared_mlp_full']
                         .sub(mid['shared_mlp_full'].median()).abs().idxmin(),
}
train_locs, train_cols = d['train_locs'], d['train_cols']
fig, axes = plt.subplots(3, 3, figsize=(12, 7.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
for r, (title, j) in enumerate(examples.items()):
    panels = [('Expert range (test) + training sightings', None),
              ('Nearest sighting', 'nearest_sighting'),
              ('Shared network', 'shared_mlp_full')]
    for c, (ptitle, m) in enumerate(panels):
        ax = axes[r, c]
        ax.set_facecolor(SURFACE)
        if m is None:
            ax.scatter(locs[:, 1], locs[:, 0], s=0.05, c='#e6e5e0', rasterized=True)
            pos = Y[:, j]
            ax.scatter(locs[pos, 1], locs[pos, 0], s=0.3, c='#86b6ef',
                       rasterized=True, label='expert range')
            tr = train_cols == j
            ax.scatter(train_locs[tr, 1], train_locs[tr, 0], s=1.5, c='#0d366b',
                       rasterized=True, label='training sightings')
            ax.set_ylabel(f'{title}\n{names[j]}\n(n={n_train[j]})',
                          fontsize=7, color=INK)
        else:
            sc = np.asarray(preds[m][:, j])
            order = np.argsort(sc)                   # draw high scores on top
            ax.scatter(locs[order, 1], locs[order, 0], s=0.05, c=sc[order],
                       cmap='Blues', vmin=0, vmax=sc.max(), rasterized=True)
            ax.text(0.01, 0.02, f'AP {table.loc[j, m]:.2f}', transform=ax.transAxes,
                    fontsize=7, color=INK)
        if r == 0:
            ax.set_title(ptitle, fontsize=8, color=INK)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color('#e6e5e0')
fig.tight_layout()
fig.savefig('results/figures/range_maps.png', facecolor=SURFACE)
plt.close(fig)
print('Saved results/metrics/per_species_ap.csv, '
      'results/figures/ap_by_rarity.png and results/figures/range_maps.png')