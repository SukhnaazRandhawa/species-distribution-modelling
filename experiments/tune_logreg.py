"""Tune per-species logistic regression on the spatial validation split,
then retrain the best setting on all training data and evaluate on test once."""
import sys, json, time, itertools
sys.path.append('.')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.data import load_all
from src.features import land_mask
from src.evaluate import per_species_ap, summarise
from src.validation import make_validation_split, val_ap_vs_background
from src.models.per_species import fit_predict_per_species

d = load_all()
n_species = len(d['species'])
s = make_validation_split(d['train_locs'], d['train_cols'], n_species)

def make_lr(C):
    return lambda: LogisticRegression(C=C, class_weight='balanced', max_iter=2000)

# ---------- 1. Grid search on validation ----------
grid = list(itertools.product([2, 4, 8, 16], [0.01, 0.1, 1.0, 10.0]))
rows = []
for n_freq, C in grid:
    t = time.time()
    scores = fit_predict_per_species(make_lr(C), s['train_locs'], s['train_cols'],
                                     s['val_locs'], n_species, background='uniform',
                                     n_freq=n_freq, verbose=False)
    row = dict(n_freq=n_freq, C=C,
               val_ap=np.nanmean(per_species_ap(scores, s['Y_val'])),
               val_ap_bg=np.nanmean(val_ap_vs_background(scores, s['Y_val'])),
               seconds=round(time.time() - t))
    rows.append(row)
    print(f"n_freq={n_freq:2d}  C={C:<5}  val AP={row['val_ap']:.4f}  "
          f"val AP (bg)={row['val_ap_bg']:.4f}  ({row['seconds']}s)")

table = pd.DataFrame(rows).sort_values('val_ap', ascending=False)
Path('results/metrics').mkdir(parents=True, exist_ok=True)
table.to_csv('results/metrics/logreg_tuning.csv', index=False)
print('\nTop 5 settings:\n', table.head().to_string(index=False))

# ---------- 2. Retrain best on ALL training data, test once ----------
best = table.iloc[0]
n_freq, C = int(best['n_freq']), float(best['C'])
print(f'\nBest: n_freq={n_freq}, C={C}. Retraining on all training data...')
scores = fit_predict_per_species(make_lr(C), d['train_locs'], d['train_cols'],
                                 d['test_locs'], n_species, background='uniform',
                                 n_freq=n_freq, verbose=False)
land = land_mask(d['test_locs'])
result = {'settings': {'n_freq': n_freq, 'C': C},
          'all': summarise(scores, d['Y_test']),
          'land': summarise(scores, d['Y_test'], mask=land)}
print(json.dumps(result, indent=2))
with open('results/metrics/logreg_tuned.json', 'w') as f:
    json.dump(result, f, indent=2)
print('Saved results/metrics/logreg_tuning.csv and logreg_tuned.json')