"""Main experiment: does climate help? Location only vs climate only vs both.

Each (model, inputs) pair is tuned on the spatial validation split and
evaluated once on the test set, exactly like the earlier models.
Usage:  python experiments/run_climate.py network
        python experiments/run_climate.py forest
"""
import sys
sys.path.append('.')
from imblearn.ensemble import BalancedRandomForestClassifier
from src.tuning import tune_and_test
from src.inputs import make_featurizer
from src.models.shared_mlp import fit_predict_shared
from src.models.per_species import fit_predict_per_species

INPUTS = sys.argv[2:] or ['location', 'climate', 'both']


def network(train_locs, train_cols, query_locs, n_species, inputs, pos_weight):
    feat = make_featurizer(inputs, train_locs, n_freq=4)
    return fit_predict_shared(train_locs, train_cols, query_locs, n_species,
                              featurize=feat, full_loss=True,
                              pos_weight=pos_weight, epochs=30, verbose=False)


def forest(train_locs, train_cols, query_locs, n_species, inputs, min_samples_leaf):
    feat = make_featurizer(inputs, train_locs, n_freq=1)
    make_rf = lambda: BalancedRandomForestClassifier(
        n_estimators=100, min_samples_leaf=min_samples_leaf,
        sampling_strategy='all', replacement=True, bootstrap=False,
        n_jobs=-1, random_state=0)
    return fit_predict_per_species(make_rf, train_locs, train_cols, query_locs,
                                   n_species, background='uniform',
                                   featurize=feat, verbose=False)


which = sys.argv[1] if len(sys.argv) > 1 else 'network'
for inputs in INPUTS:
    print(f'\n========== {which}: {inputs} ==========')
    if which == 'network':
        settings = [dict(inputs=inputs, pos_weight=w) for w in [64.0, 128.0]]
        tune_and_test(f'climate_network_{inputs}', settings, network)
    else:
        settings = [dict(inputs=inputs, min_samples_leaf=m) for m in [1, 5]]
        tune_and_test(f'climate_forest_{inputs}', settings, forest)