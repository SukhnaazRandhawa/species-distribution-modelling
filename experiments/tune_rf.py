"""Tune a down-sampled (balanced) random forest per species."""
import sys
sys.path.append('.')
from imblearn.ensemble import BalancedRandomForestClassifier
from src.tuning import tune_and_test
from src.models.per_species import fit_predict_per_species


def predict(train_locs, train_cols, query_locs, n_species, min_samples_leaf, n_freq):
    make_rf = lambda: BalancedRandomForestClassifier(
        n_estimators=100,               # number of trees
        min_samples_leaf=min_samples_leaf,
        sampling_strategy='all',        # each tree: equal presences and background
        replacement=True,               # bootstrap presences too
        bootstrap=False,                # sampling is done by the balancing step
        n_jobs=-1,                      # use all CPU cores
        random_state=0)
    return fit_predict_per_species(make_rf, train_locs, train_cols, query_locs,
                                   n_species, background='uniform',
                                   n_freq=n_freq, verbose=False)


settings = [dict(min_samples_leaf=m, n_freq=f) for f in [1, 2] for m in [1, 5, 20]]
tune_and_test('random_forest', settings, predict)