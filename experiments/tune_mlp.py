"""Tune the shared neural network (SINR-style) on the spatial validation split."""
import sys
sys.path.append('.')
from src.tuning import tune_and_test
from src.models.shared_mlp import fit_predict_shared


def predict(train_locs, train_cols, query_locs, n_species, full_loss, pos_weight):
    return fit_predict_shared(train_locs, train_cols, query_locs, n_species,
                              full_loss=full_loss, pos_weight=pos_weight,
                              epochs=10, verbose=True)


settings = [dict(full_loss=f, pos_weight=w)
            for f in [True, False] for w in [128.0, 512.0, 2048.0]]
tune_and_test('shared_mlp', settings, predict)