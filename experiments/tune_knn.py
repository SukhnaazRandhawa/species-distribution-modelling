"""Tune per-species k-nearest-neighbours (number of neighbours k)."""
import sys
sys.path.append('.')
from src.tuning import tune_and_test
from src.models.knn import knn_per_species


def predict(train_locs, train_cols, query_locs, n_species, k):
    return knn_per_species(train_locs, train_cols, query_locs, n_species,
                           k=k, verbose=False)


tune_and_test('knn', [dict(k=k) for k in [5, 15, 50]], predict)