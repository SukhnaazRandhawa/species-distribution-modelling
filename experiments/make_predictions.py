"""Re-create the test-set predictions of the main models (with their tuned
settings from results/metrics) and save them for analysis.

Saved to results/predictions/ (ignored by Git: each file is ~576 MB).
"""
import sys, json, time
sys.path.append('.')
from pathlib import Path
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from src.data import load_all
from src.models.baselines import nearest_sighting_baseline
from src.models.per_species import fit_predict_per_species
from src.models.shared_mlp import fit_predict_shared

d = load_all()
n_species = len(d['species'])
args = (d['train_locs'], d['train_cols'], d['test_locs'], n_species)
out_dir = Path('results/predictions')
out_dir.mkdir(parents=True, exist_ok=True)


def settings(name):
    with open(f'results/metrics/{name}_tuned.json') as f:
        return json.load(f)['settings']


def save(name, make):
    t = time.time()
    print(f'{name}...')
    np.save(out_dir / f'{name}.npy', make().astype(np.float32))
    print(f'  saved ({time.time() - t:.0f}s)')


save('nearest_sighting', lambda: nearest_sighting_baseline(*args))

rf = settings('random_forest')
save('random_forest', lambda: fit_predict_per_species(
    lambda: BalancedRandomForestClassifier(
        n_estimators=100, min_samples_leaf=rf['min_samples_leaf'],
        sampling_strategy='all', replacement=True, bootstrap=False,
        n_jobs=-1, random_state=0),
    *args, background='uniform', n_freq=rf['n_freq'], verbose=False))

nn = settings('shared_mlp_full')
save('shared_mlp_full', lambda: fit_predict_shared(*args, **nn, verbose=False))