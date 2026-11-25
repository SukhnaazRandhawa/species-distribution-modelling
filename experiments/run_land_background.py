"""Combined experiment: our shared network + Tej's land-only pseudo-absences.

Question: does climate help the network ON LAND when its random background
points come from land only (so it must tell land habitats apart), as it did
for Tej's random forest (land AP 0.780 -> 0.806)?

Setup (as Tej): background points only on land; ocean predictions set to 0.
Setup (as ours): settings chosen on the spatial validation split, test once.
Inputs: location, climate (19 WorldClim variables), location + climate.

Writes results/metrics/landbg_network_{inputs}_tuning.csv / _tuned.json,
saves test predictions, and adds per-species land AP columns to
per_species_land_ap.csv for the significance test.

Usage:  caffeinate -i python experiments/run_land_background.py
Takes roughly 20-30 minutes.
"""
import sys, json
sys.path.append('.')
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import load_all
from src.features import land_mask
from src.evaluate import per_species_ap
from src.tuning import tune_and_test
from src.inputs import make_featurizer
from src.models.shared_mlp import fit_predict_shared

P, M = Path('results/predictions'), Path('results/metrics')
INPUTS = sys.argv[1:] or ['location', 'climate', 'both']


def network(train_locs, train_cols, query_locs, n_species, inputs, pos_weight):
    feat = make_featurizer(inputs, train_locs, n_freq=4)
    sc = fit_predict_shared(train_locs, train_cols, query_locs, n_species,
                            featurize=feat, full_loss=True, pos_weight=pos_weight,
                            epochs=30, bg_land_only=True, verbose=False)
    return sc * land_mask(query_locs)[:, None]        # ocean -> 0, as Tej


results = {}
for inputs in INPUTS:
    print(f'\n========== network, land-only background: {inputs} ==========')
    settings = [dict(inputs=inputs, pos_weight=w) for w in [64.0, 128.0]]
    scores, res = tune_and_test(f'landbg_network_{inputs}', settings, network)
    np.save(P / f'landbg_network_{inputs}.npy', scores.astype(np.float32))
    results[inputs] = scores

# ---- Per-species land AP for the significance test ----
d = load_all()
land = land_mask(d['test_locs'])
ps = pd.read_csv(M / 'per_species_land_ap.csv')
nice = {'location': 'location', 'climate': 'climate', 'both': 'location + climate'}
for inputs, sc in results.items():
    ps[f'Network, land-only background: {nice[inputs]}'] = np.round(
        per_species_ap(sc, d['Y_test'], mask=land), 5)
ps.to_csv(M / 'per_species_land_ap.csv', index=False)

# ---- Summary: land species AP, side by side ----
print('\nLand species AP (test):')
print(f'  {"":38s} {"location":>9s} {"climate":>9s} {"both":>9s}')
row = []
for inputs in ['location', 'climate', 'both']:
    f = M / f'landbg_network_{inputs}_tuned.json'
    row.append(json.load(open(f))['land']['mean_ap_species'] if f.exists() else np.nan)
print(f'  {"Network, land-only background (new)":38s} ' + ' '.join(f'{v:9.3f}' for v in row))
print(f'  {"Network, global background (ours)":38s} {0.805:9.3f} {0.682:9.3f} {0.803:9.3f}')
print(f'  {"Random forest, land-only bg (Tej)":38s} {0.780:9.3f} {0.537:9.3f} {0.806:9.3f}')
print(f'  {"Nearest-sighting baseline":38s} {0.817:9.3f}')
print('\nAdded columns to per_species_land_ap.csv')