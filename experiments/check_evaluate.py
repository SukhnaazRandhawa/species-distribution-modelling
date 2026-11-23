"""Sanity checks for the evaluation code, on the real test set."""
import sys
sys.path.append('.')
import numpy as np
from sklearn.metrics import average_precision_score
from src.data import load_all
from src.evaluate import per_species_ap, summarise

d = load_all()
Y = d['Y_test']
rng = np.random.default_rng(0)

# 1. Perfect model: scores = the answers themselves. Should give 1.0.
perfect = Y.astype(np.float32)
print('Perfect model:', summarise(perfect, Y, probs=False))

# 2. Random model: should give mean AP close to the presence rate (~0.012).
random = rng.random(Y.shape, dtype=np.float32)
print('Random model: ', summarise(random, Y, probs=False))
print(f'Presence rate: {Y.mean():.4f}')

# 3. Our AP vs scikit-learn on 5 real species.
ours = per_species_ap(random[:, :5], Y[:, :5])
sk = [average_precision_score(Y[:, j], random[:, j]) for j in range(5)]
print('Ours:   ', np.round(ours, 5))
print('sklearn:', np.round(sk, 5))