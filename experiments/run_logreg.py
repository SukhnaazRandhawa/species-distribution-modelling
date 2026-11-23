"""Per-species logistic regression with two pseudo-absence strategies."""
import sys, json, time
sys.path.append('.')
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from src.data import load_all
from src.features import land_mask
from src.evaluate import summarise
from src.models.per_species import fit_predict_per_species

d = load_all()
Y, n_species = d['Y_test'], len(d['species'])
land = land_mask(d['test_locs'])
make_lr = lambda: LogisticRegression(class_weight='balanced', max_iter=1000)
results = {}

for bg in ['uniform', 'target_group']:
    t = time.time()
    print(f'Logistic regression, {bg} pseudo-absences')
    scores = fit_predict_per_species(make_lr, d['train_locs'], d['train_cols'],
                                     d['test_locs'], n_species, background=bg)
    results[f'logreg_{bg}'] = {'all': summarise(scores, Y),
                               'land': summarise(scores, Y, mask=land)}
    print(json.dumps(results[f'logreg_{bg}'], indent=2))
    print(f'({time.time() - t:.0f}s)\n')

Path('results/metrics').mkdir(parents=True, exist_ok=True)
with open('results/metrics/logreg.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved results/metrics/logreg.json')