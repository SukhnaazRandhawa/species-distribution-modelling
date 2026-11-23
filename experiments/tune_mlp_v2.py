"""Second round of tuning for the shared network.

Round 1 showed (a) the loss was still falling after 10 epochs and
(b) the two validation scores disagreed on which loss is better.
So here we tune each loss type SEPARATELY (more epochs, more input
frequencies) and report both on test as an ablation of the loss.
"""
import sys
sys.path.append('.')
from src.tuning import tune_and_test
from src.models.shared_mlp import fit_predict_shared


def predict(train_locs, train_cols, query_locs, n_species,
            full_loss, pos_weight, n_freq, epochs):
    return fit_predict_shared(train_locs, train_cols, query_locs, n_species,
                              full_loss=full_loss, pos_weight=pos_weight,
                              n_freq=n_freq, epochs=epochs, verbose=False)


for full, weights in [(True, [64.0, 128.0]), (False, [256.0, 512.0])]:
    settings = [dict(full_loss=full, pos_weight=w, n_freq=f, epochs=e)
                for w in weights for f in [1, 2, 4, 8] for e in [10, 30]]
    name = 'shared_mlp_full' if full else 'shared_mlp_random_only'
    print(f'\n========== {name} ==========')
    tune_and_test(name, settings, predict)