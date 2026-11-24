"""Step 11b: does learning from 1,918 EXTRA species help the shared network
predict our 500 species?

The extra file has no sightings of our 500 species, so only the shared
network can use it: it learns all 2,418 species together, and we score
only the 500 test species. Per-species models and the baseline cannot use it.

Validation: extra sightings inside the held-out validation blocks are also
removed, so no information from those areas reaches the model.
"""
import sys, json, time, itertools
sys.path.append('.')
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import load_all, load_extra_species
from src.features import land_mask
from src.evaluate import per_species_ap, summarise
from src.validation import make_validation_split, spatial_blocks, val_ap_vs_background
from src.inputs import make_featurizer
from src.models.shared_mlp import fit_predict_shared

d = load_all()
n_main = len(d['species'])
ex_locs, ex_cols, n_extra = load_extra_species(d['species'])
print(f'Main species: {n_main}, extra species: {n_extra}, '
      f'extra sightings: {len(ex_locs):,}')


def train_predict(main_locs, main_cols, ex_locs, ex_cols, query_locs,
                  use_extra, pos_weight, epochs):
    if use_extra:
        locs = np.concatenate([main_locs, ex_locs])
        cols = np.concatenate([main_cols, ex_cols])
        n_out = n_main + n_extra
    else:
        locs, cols, n_out = main_locs, main_cols, n_main
    feat = make_featurizer('location_land', main_locs, n_freq=4)
    scores = fit_predict_shared(locs, cols, query_locs, n_out, featurize=feat,
                                full_loss=True, pos_weight=pos_weight,
                                epochs=epochs, verbose=False)
    return scores[:, :n_main]              # only our 500 species are scored


# ---------- 1. Tune on the spatial validation split ----------
s = make_validation_split(d['train_locs'], d['train_cols'], n_main)
keep = ~np.isin(spatial_blocks(ex_locs, s['block_deg']), s['val_blocks'])
print(f'Extra sightings kept for tuning (outside validation blocks): '
      f'{keep.sum():,} of {len(keep):,}\n')

# One epoch WITH extra species is ~5x more training steps than one epoch
# without, so the 'without' arm is also allowed many more epochs. Otherwise
# any gain could just come from training longer, not from the extra species.
grid = ([dict(use_extra=False, pos_weight=64.0, epochs=e) for e in [30, 60, 150]] +
        [dict(use_extra=True, pos_weight=64.0, epochs=e) for e in [10, 30]])
rows = []
for g in grid:
    t = time.time()
    sc = train_predict(s['train_locs'], s['train_cols'], ex_locs[keep], ex_cols[keep],
                       s['val_locs'], **g)
    rows.append(dict(**g, val_ap=np.nanmean(per_species_ap(sc, s['Y_val'])),
                     val_ap_bg=np.nanmean(val_ap_vs_background(sc, s['Y_val'])),
                     seconds=round(time.time() - t)))
    print(f"{g}  val AP={rows[-1]['val_ap']:.4f}  "
          f"val AP (bg)={rows[-1]['val_ap_bg']:.4f}  ({rows[-1]['seconds']}s)")

table = pd.DataFrame(rows).sort_values('val_ap', ascending=False)
Path('results/metrics').mkdir(parents=True, exist_ok=True)
table.to_csv('results/metrics/extra_species_tuning.csv', index=False)
print('\nRanking:\n', table.to_string(index=False))

# ---------- 2. Best WITH and best WITHOUT extra species, each tested once ----------
land = land_mask(d['test_locs'])
results = {}
for use_extra in [False, True]:
    sub = table[table['use_extra'] == use_extra]
    best = grid[sub.index[0]]
    print(f'\nBest with use_extra={use_extra}: {best}. Training on all data...')
    sc = train_predict(d['train_locs'], d['train_cols'], ex_locs, ex_cols,
                       d['test_locs'], **best)
    key = 'with_extra_species' if use_extra else 'without_extra_species'
    results[key] = {'settings': best,
                    'all': summarise(sc, d['Y_test']),
                    'land': summarise(sc, d['Y_test'], mask=land)}
    print(json.dumps(results[key], indent=2))
    np.save(f'results/predictions/shared_mlp_{key}.npy', sc.astype(np.float32))

with open('results/metrics/extra_species_tuned.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved results/metrics/extra_species_tuning.csv and extra_species_tuned.json')