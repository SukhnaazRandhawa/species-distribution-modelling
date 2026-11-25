"""Re-score every model on the group's shared WorldClim land mask.

Why: our land-only numbers used the global_land_mask package, but the group
(and our own models' land flag) define land as "WorldClim has data here".
This script puts every row of the results table on that one definition.

What it does (no tuning, test set used only for scoring, as before):
  1. Builds the WorldClim land mask and compares it with the old mask
     (and with data/land_mask_test.npy from Adwvit, if that file exists;
     if it does, his file is used so the group matches exactly).
  2. Re-creates any test predictions we did not save, using the settings
     already stored in results/metrics/*_tuned.json.
  3. Re-scores every model (all locations + land) and rewrites the metric
     files that final_results.py reads. Old files are copied to
     results/metrics/old_mask/ first.
  4. Re-runs the extra-species seeds (1 and 2) and re-scores the ensembles
     with the weights already chosen on validation.
  5. Writes per_species_land_ap.csv and land_summary.csv for the group.

Predictions are cached in results/predictions/, so if the script stops you
can run it again and it continues where it left off.
Needs ~8 GB of free disk space. Takes roughly 1.5-2 hours.

Usage:  caffeinate -i python experiments/rescore_land.py
"""
import sys, json, time, shutil
sys.path.append('.')
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import load_all, load_extra_species
from src.features import land_mask
from src.climate import ClimateGrid
from src.evaluate import per_species_ap, summarise

P, M = Path('results/predictions'), Path('results/metrics')
OLD = M / 'old_mask'
OLD.mkdir(exist_ok=True)

d = load_all()
Y, n = d['Y_test'], len(d['species'])
args = (d['train_locs'], d['train_cols'], d['test_locs'], n)

# =====================================================================
# 1. The land mask
# =====================================================================
from global_land_mask import globe
old = globe.is_land(d['test_locs'][:, 0], d['test_locs'][:, 1])
raw = ClimateGrid().raw(d['test_locs'])
has = ~np.isnan(raw)
land = land_mask(d['test_locs'])            # all 19 variables present

print('LAND MASK CHECK (288,122 test locations)')
print(f'  Old mask (global_land_mask package):   {old.sum():>7,}')
print(f'  WorldClim, all 19 variables present:   {has.all(1).sum():>7,}   <- our rule')
print(f'  WorldClim, BIO1 present:               {has[:, 0].sum():>7,}')
print(f'  WorldClim, any variable present:       {has.any(1).sum():>7,}')
print(f'  Adwvit\'s mask (from his message):      {85661:>7,}')
print(f'  Land in both old and new:              {(old & land).sum():>7,}')
print(f'  Land only in the new mask:             {(~old & land).sum():>7,}')
print(f'  Land only in the old mask:             {(old & ~land).sum():>7,}')

official = Path('data/land_mask_test.npy')
if official.exists():
    adw = np.load(official).astype(bool).ravel()
    assert adw.shape == land.shape, f'his mask has {adw.shape}, expected {land.shape}'
    print(f'\n  Found {official}: {adw.sum():,} land points; '
          f'agrees with ours at {(adw == land).mean():.4%} of locations. Using his.')
    land = adw
np.save(M / 'land_mask_test.npy', land)
print(f'\nUsing {land.sum():,} land locations '
      f'({land.mean():.1%} of the test set). Saved results/metrics/land_mask_test.npy\n')


# =====================================================================
# 2. Recipes for re-creating predictions (settings from *_tuned.json)
# =====================================================================
def tuned(name):
    with open(M / f'{name}_tuned.json') as f:
        return json.load(f)


def network(inputs, pos_weight, train_locs=None, train_cols=None, n_out=None, seed=0):
    from src.inputs import make_featurizer
    from src.models.shared_mlp import fit_predict_shared
    tl = d['train_locs'] if train_locs is None else train_locs
    tc = d['train_cols'] if train_cols is None else train_cols
    feat = make_featurizer(inputs, d['train_locs'], n_freq=4)  # scaling from main data
    return fit_predict_shared(tl, tc, d['test_locs'], n_out or n, featurize=feat,
                              full_loss=True, pos_weight=pos_weight, epochs=30,
                              seed=seed, verbose=False)[:, :n]


def forest(min_samples_leaf, inputs=None, n_freq=1):
    from imblearn.ensemble import BalancedRandomForestClassifier
    from src.models.per_species import fit_predict_per_species
    from src.inputs import make_featurizer
    feat = None if inputs is None else make_featurizer(inputs, d['train_locs'], n_freq=1)
    make_rf = lambda: BalancedRandomForestClassifier(
        n_estimators=100, min_samples_leaf=min_samples_leaf,
        sampling_strategy='all', replacement=True, bootstrap=False,
        n_jobs=-1, random_state=0)
    return fit_predict_per_species(make_rf, *args, background='uniform',
                                   n_freq=n_freq, featurize=feat, verbose=False)


def logreg():
    from sklearn.linear_model import LogisticRegression
    from src.models.per_species import fit_predict_per_species
    s = tuned('logreg')['settings']
    print(f'    logistic regression settings: {s}')
    return fit_predict_per_species(
        lambda: LogisticRegression(C=s['C'], class_weight='balanced', max_iter=1000),
        *args, background=s.get('background', 'uniform'),
        n_freq=s.get('n_freq', 4), verbose=False)


def knn():
    from src.models.knn import knn_per_species
    return knn_per_species(*args, k=tuned('knn')['settings']['k'], verbose=False)


def mlp(name):
    from src.models.shared_mlp import fit_predict_shared
    return fit_predict_shared(*args, **tuned(name)['settings'], verbose=False)


def baseline(kind):
    from src.models.baselines import frequency_baseline, nearest_sighting_baseline
    if kind == 'frequency':
        return frequency_baseline(d['train_cols'], n, len(d['test_locs']))
    return nearest_sighting_baseline(*args)


def climate(model, inp):
    s = tuned(f'climate_{model}_{inp}')['settings']
    if model == 'network':
        return network(inp, s['pos_weight'])
    return forest(s['min_samples_leaf'], inputs=inp)


# (metric file, key inside it or None, prediction file, recipe, nice name)
MODELS = [
    ('baselines', 'frequency', None, lambda: baseline('frequency'), 'Frequency'),
    ('baselines', 'nearest_sighting', 'nearest_sighting', lambda: baseline('nearest'),
     'Nearest sighting'),
    ('logreg_tuned', None, 'logreg', logreg, 'Logistic regression'),
    ('knn_tuned', None, 'knn', knn, 'k-NN'),
    ('random_forest_tuned', None, 'random_forest',
     lambda: forest(tuned('random_forest')['settings']['min_samples_leaf'],
                    n_freq=tuned('random_forest')['settings']['n_freq']), 'Random forest'),
    ('shared_mlp_random_only_tuned', None, 'shared_mlp_random_only',
     lambda: mlp('shared_mlp_random_only'), 'Network: random-only loss'),
    ('shared_mlp_full_tuned', None, 'shared_mlp_full',
     lambda: mlp('shared_mlp_full'), 'Network: full loss'),
]
NICE = {'location': 'location', 'climate': 'climate', 'both': 'location + climate',
        'location_land': 'location + land flag'}
for model in ['network', 'forest']:
    for inp, nice in NICE.items():
        pred = f'climate_{model}_{inp}'
        if (model, inp) == ('forest', 'location'):
            pred = 'random_forest'            # identical model and settings
        MODELS.append((f'climate_{model}_{inp}_tuned', None, pred,
                       lambda m=model, i=inp: climate(m, i),
                       f'Inputs: {model}, {nice}'))


def get_scores(pred_file, recipe):
    """Load cached predictions, or create them once and cache them."""
    if pred_file is None:
        return recipe()
    f = P / f'{pred_file}.npy'
    if not f.exists():
        t = time.time()
        print(f'    re-creating {f.name} ...', flush=True)
        np.save(f, recipe().astype(np.float32))
        print(f'    saved ({(time.time() - t) / 60:.1f} min)', flush=True)
    return np.load(f, mmap_mode='r')


def backup(name):
    src = M / f'{name}.json'
    if src.exists() and not (OLD / src.name).exists():
        shutil.copy(src, OLD / src.name)


# =====================================================================
# 3. Re-score every model
# =====================================================================
per_species = pd.DataFrame({'species': [d['names'][sp] for sp in d['species']],
                            'n_train': np.bincount(d['train_cols'], minlength=n)})
summary = []
print(f'{"Model":34s} {"old all":>8s} {"new all":>8s} {"old land":>9s} {"new land":>9s}')
for metric_file, key, pred_file, recipe, nice in MODELS:
    backup(metric_file)
    with open(M / f'{metric_file}.json') as f:
        res = json.load(f)
    entry = res[key] if key else res
    old_all = entry['all']['mean_ap_species']
    old_land = entry['land']['mean_ap_species']

    sc = get_scores(pred_file, recipe)
    entry['all'] = summarise(sc, Y)
    entry['land'] = summarise(sc, Y, mask=land)
    entry['land_mask'] = f'WorldClim ({land.sum()} test locations)'
    with open(M / f'{metric_file}.json', 'w') as f:
        json.dump(res, f, indent=2)

    ap_land = per_species_ap(sc, Y, mask=land)
    per_species[nice] = np.round(ap_land, 5)
    new_all, new_land = entry['all']['mean_ap_species'], entry['land']['mean_ap_species']
    summary.append(dict(model=nice, species_ap=new_all, species_ap_land=new_land,
                        location_ap_land=entry['land']['mean_ap_location']))
    flag = '' if abs(new_all - old_all) < 0.002 else '   <- re-run differs'
    print(f'{nice:34s} {old_all:8.4f} {new_all:8.4f} {old_land:9.4f} {new_land:9.4f}{flag}',
          flush=True)
    del sc

# =====================================================================
# 4a. Extra species: 3 seeds (seed 0 = saved predictions)
# =====================================================================
print('\nExtra species, 3 seeds:')
ex_locs, ex_cols, n_extra = load_extra_species(d['species'])
if not (OLD / 'extra_species_seeds.csv').exists():
    shutil.copy(M / 'extra_species_seeds.csv', OLD / 'extra_species_seeds.csv')
rows = []
for seed in [0, 1, 2]:
    for use_extra in [False, True]:
        tag = 'with' if use_extra else 'without'
        pred = f'shared_mlp_{tag}_extra_species' if seed == 0 else f'extra_seed{seed}_{tag}'
        if use_extra:
            recipe = lambda s=seed: network(
                'location_land', 64.0,
                np.concatenate([d['train_locs'], ex_locs]),
                np.concatenate([d['train_cols'], ex_cols]), n + n_extra, seed=s)
        else:
            recipe = lambda s=seed: network('location_land', 64.0, seed=s)
        sc = get_scores(pred, recipe)
        a, l = summarise(sc, Y), summarise(sc, Y, mask=land)
        rows.append(dict(seed=seed, use_extra=use_extra,
                         species_ap=a['mean_ap_species'],
                         species_ap_land=l['mean_ap_species'],
                         location_ap=a['mean_ap_location'],
                         richness_mae=a['richness_mae']))
        if seed == 0:
            per_species[f'Network ({tag} extra species, seed 0)'] = np.round(
                per_species_ap(sc, Y, mask=land), 5)
        print(f"  seed {seed}, {tag:7s} extra: species AP {a['mean_ap_species']:.4f}"
              f"  land {l['mean_ap_species']:.4f}", flush=True)
        del sc
seeds = pd.DataFrame(rows)
seeds.to_csv(M / 'extra_species_seeds.csv', index=False)
print(seeds.groupby('use_extra')[['species_ap', 'species_ap_land', 'location_ap',
                                  'richness_mae']].agg(['mean', 'std']).round(4).to_string())

# =====================================================================
# 4b. Ensembles: same weights as before (chosen on validation)
# =====================================================================
from src.ensemble import rank_normalise, combine, species_spread_km
print('\nEnsembles (weights from validation, unchanged):')
backup('ensemble')
with open(M / 'ensemble.json') as f:
    ens = json.load(f)
GROUPS = ['clustered', 'medium', 'spread out']
group = np.digitize(species_spread_km(d['train_locs'], d['train_cols'], n),
                    ens['spread_cuts_km'])
base = rank_normalise(np.load(P / 'nearest_sighting.npy', mmap_mode='r'))
for net, file in [('network_extra=False', 'shared_mlp_without_extra_species'),
                  ('network_extra=True', 'shared_mlp_with_extra_species')]:
    w = ens['weights'][net]
    w_sp = np.array([w['group_weights'][GROUPS[g]] for g in group], dtype=np.float32)
    sc = combine(rank_normalise(np.load(P / f'{file}.npy', mmap_mode='r')), base, w_sp)
    name = f'{net} + baseline (weight by spread)'
    ens['test'][name] = {'all': summarise(sc, Y, probs=False),
                         'land': summarise(sc, Y, mask=land, probs=False)}
    tag = 'with' if net.endswith('True') else 'without'
    per_species[f'Ensemble ({tag} extra species)'] = np.round(
        per_species_ap(sc, Y, mask=land), 5)
    print(f"  {name:48s} species AP {ens['test'][name]['all']['mean_ap_species']:.4f}"
          f"  land {ens['test'][name]['land']['mean_ap_species']:.4f}", flush=True)
    del sc
ens['land_mask'] = f'WorldClim ({land.sum()} test locations)'
with open(M / 'ensemble.json', 'w') as f:
    json.dump(ens, f, indent=2)

# =====================================================================
# 5. Files for the group
# =====================================================================
per_species.to_csv(M / 'per_species_land_ap.csv', index=False)
pd.DataFrame(summary).round(4).to_csv(M / 'land_summary.csv', index=False)
print('\nSaved results/metrics/per_species_land_ap.csv and land_summary.csv.')
print('Old metric files are in results/metrics/old_mask/.')
print('Next: python experiments/habitat_importance.py && python experiments/final_results.py')