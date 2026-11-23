"""Shared tune-then-test procedure used by every model.

1. Train each candidate setting on the train part of the spatial split
   and score it on validation.
2. Pick the best setting by validation AP (all negatives).
3. Retrain that setting on ALL training data and evaluate on test once.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import load_all
from src.features import land_mask
from src.evaluate import per_species_ap, summarise
from src.validation import make_validation_split, val_ap_vs_background


def tune_and_test(name, settings_list, predict):
    """predict(train_locs, train_cols, query_locs, n_species, **settings)
    must return an (n_query, n_species) score array."""
    d = load_all()
    n_species = len(d['species'])
    s = make_validation_split(d['train_locs'], d['train_cols'], n_species)

    rows = []
    for settings in settings_list:
        t = time.time()
        scores = predict(s['train_locs'], s['train_cols'], s['val_locs'],
                         n_species, **settings)
        row = dict(**settings,
                   val_ap=np.nanmean(per_species_ap(scores, s['Y_val'])),
                   val_ap_bg=np.nanmean(val_ap_vs_background(scores, s['Y_val'])),
                   seconds=round(time.time() - t))
        rows.append(row)
        print(f"{settings}  val AP={row['val_ap']:.4f}  "
              f"val AP (bg)={row['val_ap_bg']:.4f}  ({row['seconds']}s)")

    table = pd.DataFrame(rows).sort_values('val_ap', ascending=False)
    Path('results/metrics').mkdir(parents=True, exist_ok=True)
    table.to_csv(f'results/metrics/{name}_tuning.csv', index=False)
    print('\nRanking:\n', table.to_string(index=False))

    # Take the original settings dict (pandas would turn ints into floats).
    best = settings_list[table.index[0]]
    print(f'\nBest: {best}. Retraining on all training data...')
    t = time.time()
    scores = predict(d['train_locs'], d['train_cols'], d['test_locs'],
                     n_species, **best)
    land = land_mask(d['test_locs'])
    result = {'settings': best,
              'all': summarise(scores, d['Y_test']),
              'land': summarise(scores, d['Y_test'], mask=land),
              'seconds': round(time.time() - t)}
    print(json.dumps(result, indent=2))
    with open(f'results/metrics/{name}_tuned.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved results/metrics/{name}_tuning.csv and {name}_tuned.json')
    return scores, result