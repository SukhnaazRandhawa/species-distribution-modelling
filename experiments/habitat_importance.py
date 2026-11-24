"""Step 10c: which climate variables matter? (habitat preferences)

Train the climate-only network, then measure permutation importance on LAND
test locations: shuffle one climate variable at a time and see how much
each species' test AP drops. A big drop means the model relies on that
variable to decide where the species lives.
"""
import sys, json
sys.path.append('.')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import load_all
from src.features import land_mask
from src.inputs import make_featurizer
from src.climate import BIO_NAMES
from src.evaluate import per_species_ap
from src.models.shared_mlp import fit_predict_shared

N_REPEATS = 3                                 # shuffles per variable
TEMP, PRECIP = range(0, 11), range(11, 19)    # column groups (BIO1-11, BIO12-19)

d = load_all()
n_species = len(d['species'])
land = land_mask(d['test_locs'])
locs, Y = d['test_locs'][land], d['Y_test'][land]

with open('results/metrics/climate_network_climate_tuned.json') as f:
    settings = json.load(f)['settings']
feat = make_featurizer('climate', d['train_locs'])
_, predict = fit_predict_shared(
    d['train_locs'], d['train_cols'], locs, n_species, featurize=feat,
    full_loss=True, pos_weight=settings['pos_weight'], epochs=30,
    verbose=False, return_model=True)

X = feat(locs)
base_ap = per_species_ap(predict(X), Y)
print(f'Climate-only network, land test locations: mean AP = {np.nanmean(base_ap):.3f}\n')
rng = np.random.default_rng(0)


def importance(cols):
    """Mean drop in each species' AP when the given columns are shuffled
    (rows shuffled together, so the variables keep their relationship)."""
    drops = []
    for _ in range(N_REPEATS):
        Xp = X.copy()
        Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
        drops.append(base_ap - per_species_ap(predict(Xp), Y))
    return np.mean(drops, axis=0)


# ---- 1. Each variable on its own ----
names = [f'BIO{i}: {BIO_NAMES[i]}' for i in range(1, 20)]
imp = np.stack([importance([i]) for i in range(19)])       # (19, n_species)
overall = pd.Series(np.nanmean(imp, axis=1), index=names).sort_values(ascending=False)
print('Mean drop in AP when each variable is shuffled:')
print(overall.round(4).to_string(), '\n')

# ---- 2. Temperature vs precipitation as groups ----
group_imp = {'temperature (BIO1-11)': importance(list(TEMP)),
             'precipitation (BIO12-19)': importance(list(PRECIP))}

# ---- 3. Tropical vs temperate species ----
mean_lat = np.array([np.abs(d['train_locs'][d['train_cols'] == j, 0]).mean()
                     for j in range(n_species)])
zone = np.where(mean_lat < 23.4, 'tropical', 'temperate')
rows = []
for z in ['tropical', 'temperate']:
    m = zone == z
    rows.append({'zone': z, 'n_species': int(m.sum()),
                 **{g: float(np.nanmean(v[m])) for g, v in group_imp.items()}})
by_zone = pd.DataFrame(rows).set_index('zone')
print('Mean AP drop when a whole group of variables is shuffled:')
print(by_zone.round(4).to_string(), '\n')

# ---- 4. Most important variable per species ----
top = pd.Series([names[i] for i in np.nanargmax(np.nan_to_num(imp, nan=-1), axis=0)])
print('Most important variable, counted over species:')
print(top.value_counts().head(8).to_string(), '\n')

# ---- Save ----
Path('results/metrics').mkdir(parents=True, exist_ok=True)
per_species = pd.DataFrame(imp.T, columns=[f'BIO{i}' for i in range(1, 20)])
per_species.insert(0, 'species', [d['names'][sp] for sp in d['species']])
per_species.insert(1, 'zone', zone)
per_species.insert(2, 'base_ap_land', base_ap)
per_species.to_csv('results/metrics/habitat_importance_per_species.csv', index=False)
by_zone.to_csv('results/metrics/habitat_importance_by_zone.csv')

# ---- Figure: importance of each variable ----
INK, MUTED, SURFACE = '#0b0b0b', '#898781', '#fcfcfb'
COL = {'Temperature': '#2a78d6', 'Precipitation': '#eb6834'}
order = overall.index[::-1]                     # largest at the top
fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for y, n in enumerate(order):
    g = 'Temperature' if int(n.split(':')[0][3:]) <= 11 else 'Precipitation'
    ax.barh(y, overall[n], color=COL[g], height=0.7, zorder=3,
            label=g if g not in ax.get_legend_handles_labels()[1] else None)
    ax.text(overall[n] + overall.max() * 0.01, y, f'{overall[n]:.3f}',
            va='center', fontsize=6.5, color=INK)
ax.set_yticks(range(len(order)), order, fontsize=7, color=INK)
ax.set_xlabel('Mean drop in AP when shuffled (land test locations)', fontsize=8, color=INK)
ax.tick_params(axis='x', colors=MUTED, labelsize=7)
ax.tick_params(axis='y', colors=MUTED, labelcolor=INK)
ax.grid(axis='x', color='#e6e5e0', linewidth=0.6, zorder=0)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
for s in ['left', 'bottom']: ax.spines[s].set_color(MUTED)
ax.legend(frameon=False, fontsize=8, loc='lower right')
ax.set_title('Which climate variables does the model rely on?', fontsize=10,
             color=INK, loc='left')
fig.tight_layout()
Path('results/figures').mkdir(parents=True, exist_ok=True)
fig.savefig('results/figures/habitat_importance.png', facecolor=SURFACE)
print('Saved results/metrics/habitat_importance_*.csv and '
      'results/figures/habitat_importance.png')