"""Run and evaluate the two baselines."""
import sys, json, time
sys.path.append('.')
from pathlib import Path
from src.data import load_all
from src.features import land_mask
from src.evaluate import summarise
from src.models.baselines import frequency_baseline, nearest_sighting_baseline

d = load_all()
Y, n_species = d['Y_test'], len(d['species'])
land = land_mask(d['test_locs'])
results = {}

def evaluate(name, scores):
    results[name] = {
        'all':  summarise(scores, Y),
        'land': summarise(scores, Y, mask=land),
    }
    print(name, json.dumps(results[name], indent=2))

t = time.time()
evaluate('frequency', frequency_baseline(d['train_cols'], n_species, len(Y)))
print(f'({time.time() - t:.0f}s)\n')

t = time.time()
evaluate('nearest_sighting', nearest_sighting_baseline(
    d['train_locs'], d['train_cols'], d['test_locs'], n_species))
print(f'({time.time() - t:.0f}s)')

Path('results/metrics').mkdir(parents=True, exist_ok=True)
with open('results/metrics/baselines.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved results/metrics/baselines.json')