"""Is the extra-species gain reliable, or partly luck?

Re-train both versions (without / with the 1,918 extra species) with
different random seeds, using the SAME settings chosen earlier
(pos_weight 64, 30 epochs). Seed 0 is already in extra_species_tuned.json.
This measures run-to-run variation; it does not choose a model.
"""
import sys, json, time
sys.path.append('.')
import numpy as np
import pandas as pd
from src.data import load_all, load_extra_species
from src.features import land_mask
from src.evaluate import summarise
from src.inputs import make_featurizer
from src.models.shared_mlp import fit_predict_shared

SEEDS = [1, 2]

d = load_all()
n_main = len(d['species'])
ex_locs, ex_cols, n_extra = load_extra_species(d['species'])
land = land_mask(d['test_locs'])
feat = make_featurizer('location_land', d['train_locs'], n_freq=4)

# Seed 0 results from the earlier run
with open('results/metrics/extra_species_tuned.json') as f:
    prev = json.load(f)
rows = []
for key, use_extra in [('without_extra_species', False), ('with_extra_species', True)]:
    r = prev[key]
    rows.append(dict(seed=0, use_extra=use_extra,
                     species_ap=r['all']['mean_ap_species'],
                     species_ap_land=r['land']['mean_ap_species'],
                     location_ap=r['all']['mean_ap_location'],
                     richness_mae=r['all']['richness_mae']))

for seed in SEEDS:
    for use_extra in [False, True]:
        t = time.time()
        if use_extra:
            locs = np.concatenate([d['train_locs'], ex_locs])
            cols = np.concatenate([d['train_cols'], ex_cols])
            n_out = n_main + n_extra
        else:
            locs, cols, n_out = d['train_locs'], d['train_cols'], n_main
        sc = fit_predict_shared(locs, cols, d['test_locs'], n_out, featurize=feat,
                                full_loss=True, pos_weight=64.0, epochs=30,
                                seed=seed, verbose=False)[:, :n_main]
        a, l = summarise(sc, d['Y_test']), summarise(sc, d['Y_test'], mask=land)
        rows.append(dict(seed=seed, use_extra=use_extra,
                         species_ap=a['mean_ap_species'],
                         species_ap_land=l['mean_ap_species'],
                         location_ap=a['mean_ap_location'],
                         richness_mae=a['richness_mae']))
        print(f"seed {seed}, use_extra={use_extra}: species AP {a['mean_ap_species']:.4f} "
              f"(land {l['mean_ap_species']:.4f})  ({time.time() - t:.0f}s)")

table = pd.DataFrame(rows)
table.to_csv('results/metrics/extra_species_seeds.csv', index=False)
print('\nAll runs:\n', table.round(4).to_string(index=False))

metrics = ['species_ap', 'species_ap_land', 'location_ap', 'richness_mae']
summary = table.groupby('use_extra')[metrics].agg(['mean', 'std']).round(4)
print('\nMean and standard deviation over 3 seeds:\n', summary.to_string())

# Paired comparison: same seed, with minus without
w = table[table.use_extra].set_index('seed')[metrics]
wo = table[~table.use_extra].set_index('seed')[metrics]
diff = (w - wo).round(4)
print('\nWith minus without, per seed:\n', diff.to_string())
print('\nExtra species better on species AP in',
      int((diff['species_ap'] > 0).sum()), 'of', len(diff), 'seeds')