"""Step 12: ensemble of the shared network and the nearest-sighting baseline.

The error analysis showed the baseline wins on small ranges and the network
on large ranges. So we combine their (rank-normalised) scores:
    score = w * network + (1 - w) * baseline
with (a) one weight for all species, and (b) a separate weight for species
with tightly clustered / medium / widely spread training sightings.
All weights are chosen on the spatial validation split; test is used once.
"""
import sys, json, time
sys.path.append('.')
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import load_all, load_extra_species
from src.features import land_mask
from src.evaluate import per_species_ap, summarise
from src.validation import make_validation_split, spatial_blocks
from src.inputs import make_featurizer
from src.models.baselines import nearest_sighting_baseline
from src.models.shared_mlp import fit_predict_shared
from src.ensemble import rank_normalise, combine, species_spread_km

WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)     # 0 = baseline only, 1 = network only
GROUPS = ['clustered', 'medium', 'spread out']

d = load_all()
n = len(d['species'])
ex_locs, ex_cols, n_extra = load_extra_species(d['species'])
s = make_validation_split(d['train_locs'], d['train_cols'], n)


def network(train_locs, train_cols, query_locs, use_extra, ex_keep=None):
    feat = make_featurizer('location_land', train_locs, n_freq=4)
    if use_extra:
        el, ec = (ex_locs, ex_cols) if ex_keep is None else (ex_locs[ex_keep], ex_cols[ex_keep])
        train_locs = np.concatenate([train_locs, el])
        train_cols = np.concatenate([train_cols, ec])
    n_out = n + n_extra if use_extra else n
    return fit_predict_shared(train_locs, train_cols, query_locs, n_out, featurize=feat,
                              full_loss=True, pos_weight=64.0, epochs=30,
                              verbose=False)[:, :n]


# ---------------- 1. Validation predictions ----------------
t = time.time()
print('Validation predictions...')
val = {'baseline': nearest_sighting_baseline(s['train_locs'], s['train_cols'],
                                             s['val_locs'], n)}
ex_keep = ~np.isin(spatial_blocks(ex_locs, s['block_deg']), s['val_blocks'])
for use_extra in [False, True]:
    val[f'network_extra={use_extra}'] = network(
        s['train_locs'], s['train_cols'], s['val_locs'], use_extra, ex_keep)
val = {k: rank_normalise(v) for k, v in val.items()}
print(f'  done ({time.time() - t:.0f}s)')

# Spread groups: thresholds from the validation-split training data
spread_tr = species_spread_km(s['train_locs'], s['train_cols'], n)
cuts = np.quantile(spread_tr, [1 / 3, 2 / 3])
group_val = np.digitize(spread_tr, cuts)            # 0, 1, 2
print(f'Spread thresholds (km): {cuts.round(0)}\n')

# ---------------- 2. Choose weights on validation ----------------
chosen = {}
for net in ['network_extra=False', 'network_extra=True']:
    ap = np.stack([per_species_ap(combine(val[net], val['baseline'], w), s['Y_val'])
                   for w in WEIGHTS])                  # (n_weights, n_species)
    mean_ap = np.nanmean(ap, axis=1)
    w_global = WEIGHTS[np.argmax(mean_ap)]
    w_group = [float(WEIGHTS[np.argmax(np.nanmean(ap[:, group_val == g], axis=1))])
               for g in range(3)]
    chosen[net] = dict(global_weight=float(w_global),
                       group_weights={GROUPS[g]: float(w) for g, w in enumerate(w_group)})
    print(f'{net}: validation AP by weight '
          + ', '.join(f'{w}:{m:.4f}' for w, m in zip(WEIGHTS, mean_ap)))
    print(f'  best single weight: {w_global}; per spread group: '
          f'{dict(zip(GROUPS, w_group))}\n')

# ---------------- 3. Test, once ----------------
land = land_mask(d['test_locs'])
Y = d['Y_test']
spread_all = species_spread_km(d['train_locs'], d['train_cols'], n)
group_test = np.digitize(spread_all, cuts)
base = rank_normalise(np.load('results/predictions/nearest_sighting.npy', mmap_mode='r'))
results, per_group = {}, []


def evaluate(name, scores):
    sp = per_species_ap(scores, Y)
    results[name] = {'all': summarise(scores, Y, probs=False),
                     'land': summarise(scores, Y, mask=land, probs=False)}
    per_group.append({'model': name, **{GROUPS[g]: float(np.nanmean(sp[group_test == g]))
                                        for g in range(3)}})
    print(f"{name:42s} species AP {results[name]['all']['mean_ap_species']:.4f} "
          f"(land {results[name]['land']['mean_ap_species']:.4f})")


evaluate('baseline (nearest sighting)', base)
for net, file in [('network_extra=False', 'shared_mlp_without_extra_species'),
                  ('network_extra=True', 'shared_mlp_with_extra_species')]:
    netr = rank_normalise(np.load(f'results/predictions/{file}.npy', mmap_mode='r'))
    evaluate(net, netr)
    c = chosen[net]
    evaluate(f"{net} + baseline (w={c['global_weight']})",
             combine(netr, base, c['global_weight']))
    w_sp = np.array([c['group_weights'][GROUPS[g]] for g in group_test])
    evaluate(f'{net} + baseline (weight by spread)', combine(netr, base, w_sp))
    del netr

print('\nTest species AP by how spread out the training sightings are:')
print(pd.DataFrame(per_group).set_index('model').round(4).to_string())

Path('results/metrics').mkdir(parents=True, exist_ok=True)
with open('results/metrics/ensemble.json', 'w') as f:
    json.dump({'weights': chosen, 'spread_cuts_km': cuts.tolist(),
               'test': results, 'test_by_spread': per_group}, f, indent=2)
print('\nSaved results/metrics/ensemble.json')