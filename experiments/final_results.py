"""Gather every saved result into one table, and make the report figures.

Reads only files in results/metrics (no models are trained here).
Writes:
  results/metrics/final_results.csv      the full table
  results/metrics/final_results.tex      LaTeX table for the report
  results/figures/climate_experiment.png main research-question figure
  results/figures/main_results.png       headline model comparison
  results/figures/habitat_top10.png      compact habitat-importance figure
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

M = Path('results/metrics')
F = Path('results/figures'); F.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#898781', '#e6e5e0', '#fcfcfb'
BLUE, ORANGE = '#2a78d6', '#eb6834'


def load(name):
    with open(M / f'{name}.json') as f:
        return json.load(f)


def row(group, model, r, n_seeds=1, std=None):
    """One table row from a result dict with 'all' and 'land' sections."""
    a, l = r['all'], r['land']
    return dict(group=group, model=model,
                species_ap=a['mean_ap_species'], species_ap_land=l['mean_ap_species'],
                location_ap=a['mean_ap_location'],
                richness_mae=a.get('richness_mae', np.nan),
                n_seeds=n_seeds, **(std or {}))


rows = []
b = load('baselines')
rows += [row('Baselines', 'Frequency (≈ chance)', b['frequency']),
         row('Baselines', 'Nearest sighting', b['nearest_sighting'])]

rows += [row('Per-species models (location)', 'Logistic regression', load('logreg_tuned')),
         row('Per-species models (location)', 'k-NN', load('knn_tuned')),
         row('Per-species models (location)', 'Random forest', load('random_forest_tuned'))]

rows += [row('Shared network (location)', 'Random-only loss', load('shared_mlp_random_only_tuned')),
         row('Shared network (location)', 'Full loss', load('shared_mlp_full_tuned'))]

names = {'location': 'Location', 'climate': 'Climate', 'both': 'Location + climate',
         'location_land': 'Location + land flag'}
for model, label in [('network', 'Shared network'), ('forest', 'Random forest')]:
    for inp, nice in names.items():
        rows.append(row(f'Inputs experiment: {label}', nice, load(f'climate_{model}_{inp}_tuned')))
        
# Extra species: mean ± std over 3 seeds
seeds = pd.read_csv(M / 'extra_species_seeds.csv')
for use_extra, nice in [(False, 'Location + land flag'), (True, '+ 1,918 extra species')]:
    s = seeds[seeds.use_extra == use_extra]
    rows.append(dict(group='Extra species (3 seeds)', model=nice,
                     species_ap=s.species_ap.mean(), species_ap_land=s.species_ap_land.mean(),
                     location_ap=s.location_ap.mean(), richness_mae=s.richness_mae.mean(),
                     n_seeds=len(s), species_ap_std=s.species_ap.std(),
                     species_ap_land_std=s.species_ap_land.std(),
                     location_ap_std=s.location_ap.std(),
                     richness_mae_std=s.richness_mae.std()))

ens = load('ensemble')['test']
for key, nice in [('network_extra=False + baseline (weight by spread)',
                   'Network + baseline (by spread)'),
                  ('network_extra=True + baseline (weight by spread)',
                   'Network (extra species) + baseline (by spread)')]:
    rows.append(row('Ensembles', nice, ens[key]))
    rows[-1]['location_ap'] = np.nan     # rank-normalised scores aren't comparable across species
    
table = pd.DataFrame(rows)
table.to_csv(M / 'final_results.csv', index=False)

# ---------------- Readable print ----------------
def fmt(v, s=None, d=3):
    if pd.isna(v):
        return '–'
    return f'{v:.{d}f}' + (f' ± {s:.{d}f}' if s is not None and not pd.isna(s) else '')

cols = [('species_ap', 'Species AP'), ('species_ap_land', 'Species AP (land)'),
        ('location_ap', 'Location AP'), ('richness_mae', 'Richness MAE')]
show = pd.DataFrame({'Group': table.group, 'Model': table.model})
for c, nice in cols:
    std = table.get(f'{c}_std', pd.Series([np.nan] * len(table)))
    show[nice] = [fmt(v, s, 2 if c == 'richness_mae' else 3) for v, s in zip(table[c], std)]
pd.set_option('display.width', 200, 'display.max_colwidth', 50)
print(show.to_string(index=False))

# ---------------- LaTeX table (compact: the rows that go in the report) ----------------
report_rows = [('Baselines', 'Nearest sighting'),
               ('Per-species models (location)', 'Logistic regression'),
               ('Per-species models (location)', 'k-NN'),
               ('Per-species models (location)', 'Random forest'),
               ('Inputs experiment: Shared network', 'Location'),
               ('Inputs experiment: Shared network', 'Climate'),
               ('Inputs experiment: Shared network', 'Location + climate'),
               ('Inputs experiment: Shared network', 'Location + land flag'),
               ('Extra species (3 seeds)', '+ 1,918 extra species'),
               ('Ensembles', 'Network (extra species) + baseline (by spread)')]
lines = [r'\begin{tabular}{llcccc}', r'\toprule',
         r'Group & Model & Species AP & Species AP (land) & Location AP & Richness MAE \\',
         r'\midrule']
for g, m in report_rows:
    r = table[(table.group == g) & (table.model == m)].iloc[0]
    vals = [fmt(r[c], r.get(f'{c}_std'), 2 if c == 'richness_mae' else 3)
            .replace(' ± ', r' $\pm$ ') for c, _ in cols]
    lines.append(f'{g.split(":")[-1].strip()} & {m} & ' + ' & '.join(vals) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
(M / 'final_results.tex').write_text('\n'.join(lines).replace('≈', r'$\approx$'))


# ---------------- Figure helpers ----------------
def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9, color=INK, loc='left')
    ax.tick_params(colors=MUTED, labelsize=7, labelcolor=INK)
    ax.grid(axis='x' if ax.get_xlim()[1] <= 1.05 else 'y', color=GRID, lw=0.6, zorder=0)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(MUTED)


def bars_with_labels(ax, x, h, width, color, label):
    bs = ax.bar(x, h, width, color=color, label=label, zorder=3)
    for b_, v in zip(bs, h):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.01, f'{v:.2f}',
                ha='center', va='bottom', fontsize=6.5, color=INK)


# ---------------- Figure 1: the inputs experiment ----------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), dpi=200, sharey=True)
fig.patch.set_facecolor(SURFACE)
x = np.arange(len(names)); w = 0.38
for ax, col, title in [(axes[0], 'species_ap', 'All test locations'),
                       (axes[1], 'species_ap_land', 'Land test locations only')]:
    for i, (label, color) in enumerate([('Shared network', BLUE), ('Random forest', ORANGE)]):
        vals = [table[(table.group == f'Inputs experiment: {label}') &
                      (table.model == m)][col].iloc[0] for m in names.values()]
        bars_with_labels(ax, x + (i - 0.5) * w, vals, w * 0.92, color, label)
    base = table[table.model == 'Nearest sighting'][col].iloc[0]
    ax.axhline(base, color=MUTED, ls='--', lw=1, zorder=2,
               label=f'Nearest-sighting baseline ({base:.2f})')
    ax.set_xticks(x, [v.replace(' + ', '\n+ ') for v in names.values()], fontsize=7)
    ax.set_ylim(0, 1)
    style(ax, title)
axes[0].set_ylabel('Mean species AP', fontsize=8, color=INK)
for ax in axes:
    ax.legend(frameon=False, fontsize=6.5, loc='upper left', ncol=3)
    ax.set_ylim(0, 1.08)
fig.tight_layout()
fig.savefig(F / 'climate_experiment.png', facecolor=SURFACE)
plt.close(fig)

# ---------------- Figure 2: headline comparison ----------------
key = [('Baselines', 'Nearest sighting', 'Nearest sighting'),
       ('Per-species models (location)', 'Logistic regression', 'Logistic regression'),
       ('Per-species models (location)', 'k-NN', 'k-NN'),
       ('Per-species models (location)', 'Random forest', 'Random forest'),
       ('Inputs experiment: Shared network', 'Location', 'Network: location'),
       ('Inputs experiment: Shared network', 'Location + land flag', 'Network: location + land'),
       ('Extra species (3 seeds)', '+ 1,918 extra species', '+ extra species'),
       ('Ensembles', 'Network (extra species) + baseline (by spread)', '+ baseline ensemble')]
sel = pd.DataFrame([table[(table.group == g) & (table.model == m)].iloc[0].to_dict() | {'label': l}
                    for g, m, l in key])
fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
y = np.arange(len(sel))[::-1]; h = 0.38
for i, (col, label, color) in enumerate([('species_ap', 'All locations', BLUE),
                                         ('species_ap_land', 'Land only', ORANGE)]):
    bs = ax.barh(y + (0.5 - i) * h, sel[col], h * 0.92, color=color, label=label, zorder=3)
    for b_, v in zip(bs, sel[col]):
        ax.text(v + 0.005, b_.get_y() + b_.get_height() / 2, f'{v:.3f}',
                va='center', fontsize=6, color=INK)
ax.set_yticks(y, sel.label, fontsize=7)
ax.set_xlim(0, 1)
ax.set_xlabel('Mean species AP (test)', fontsize=8, color=INK)
ax.legend(frameon=False, fontsize=7, loc='upper center',
          bbox_to_anchor=(0.5, -0.17), ncol=2)
style(ax, 'Model comparison (Q1)')
fig.tight_layout()
fig.savefig(F / 'main_results.png', facecolor=SURFACE)
plt.close(fig)

# ---------------- Figure 3: top-10 habitat variables ----------------
BIO = {1: 'Annual mean temperature', 2: 'Mean diurnal range', 3: 'Isothermality',
       4: 'Temperature seasonality', 5: 'Max temp. of warmest month',
       6: 'Min temp. of coldest month', 7: 'Temperature annual range',
       8: 'Mean temp. of wettest quarter', 9: 'Mean temp. of driest quarter',
       10: 'Mean temp. of warmest quarter', 11: 'Mean temp. of coldest quarter',
       12: 'Annual precipitation', 13: 'Precip. of wettest month',
       14: 'Precip. of driest month', 15: 'Precipitation seasonality',
       16: 'Precip. of wettest quarter', 17: 'Precip. of driest quarter',
       18: 'Precip. of warmest quarter', 19: 'Precip. of coldest quarter'}
hab_file = M / 'habitat_importance_per_species.csv'
if hab_file.exists():
    hab = pd.read_csv(hab_file)
    imp = hab[[f'BIO{i}' for i in range(1, 20)]].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(5.5, 3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    yy = np.arange(len(imp))[::-1]
    for yi, (k, v) in zip(yy, imp.items()):
        i = int(k[3:])
        color, lab = (BLUE, 'Temperature') if i <= 11 else (ORANGE, 'Precipitation')
        ax.barh(yi, v, 0.7, color=color, zorder=3,
                label=lab if lab not in ax.get_legend_handles_labels()[1] else None)
        ax.text(v + imp.max() * 0.01, yi, f'{v:.3f}', va='center', fontsize=6.5, color=INK)
    ax.set_yticks(yy, [f'BIO{int(k[3:])}: {BIO[int(k[3:])]}' for k in imp.index], fontsize=7)
    ax.set_xlabel('Mean drop in species AP when shuffled (land)', fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=7, loc='upper center',
              bbox_to_anchor=(0.4, -0.2), ncol=2)
    style(ax, 'Top 10 climate variables (climate-only network)')
    ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(F / 'habitat_top10.png', facecolor=SURFACE)
    plt.close(fig)

print('\nSaved results/metrics/final_results.csv, final_results.tex and figures: '
      'climate_experiment.png, main_results.png, habitat_top10.png')