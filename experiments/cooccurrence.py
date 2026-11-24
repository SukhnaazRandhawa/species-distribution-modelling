"""Step 13 (Q3): does the presence of one species tell us about others?

The shared network gives every species a 256-number 'signature': its row of
weights in the final layer. Species with similar signatures are predicted
in similar places. We test whether the model's similarities match REAL
co-occurrence in the expert (IUCN) test maps, and compare with a simple
reference: how often the two species' TRAINING sightings fall in the same
1-degree grid cells.
"""
import sys
sys.path.append('.')
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import load_all
from src.inputs import make_featurizer
from src.models.shared_mlp import fit_predict_shared

d = load_all()
n = len(d['species'])
names = np.array([d['names'][sp] for sp in d['species']])
Y = d['Y_test']


def jaccard(A):
    """A: (items, species) 0/1 matrix -> (species, species) Jaccard overlap:
    shared items / items in either."""
    A = sparse.csr_matrix(A, dtype=np.float32)
    inter = (A.T @ A).toarray()
    size = np.diag(inter)
    union = size[:, None] + size[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


# ---- 1. TRUE co-occurrence: overlap of expert ranges ----
true_j = jaccard(Y)

# ---- 2. Reference: overlap of training sightings in 1-degree cells ----
cell = (np.floor(d['train_locs'][:, 0] + 90).astype(int) * 360
        + np.floor(d['train_locs'][:, 1] + 180).astype(int))
_, cell_idx = np.unique(cell, return_inverse=True)
train_inc = sparse.csr_matrix((np.ones(len(cell_idx)), (cell_idx, d['train_cols'])))
train_inc.data[:] = 1                                  # presence, not counts
train_j = jaccard(train_inc)

# ---- 3. The model's species signatures ----
feat = make_featurizer('location_land', d['train_locs'], n_freq=4)
_, predict = fit_predict_shared(d['train_locs'], d['train_cols'], d['test_locs'][:1],
                                n, featurize=feat, full_loss=True, pos_weight=64.0,
                                epochs=30, verbose=False, return_model=True)
W = predict.model.classifier.weight.detach().cpu().numpy()     # (500, 256)
W = W / np.linalg.norm(W, axis=1, keepdims=True)
model_sim = W @ W.T                                             # cosine similarity

# ---- 4. Do they agree with real co-occurrence? ----
iu = np.triu_indices(n, k=1)                         # every pair once
pairs = pd.DataFrame({'a': names[iu[0]], 'b': names[iu[1]],
                      'true_overlap': true_j[iu], 'train_overlap': train_j[iu],
                      'model_similarity': model_sim[iu]})
print(f'{len(pairs):,} species pairs; '
      f'{(pairs.true_overlap > 0).mean() * 100:.1f}% overlap at all in the expert maps\n')
for col in ['model_similarity', 'train_overlap']:
    rho = spearmanr(pairs[col], pairs['true_overlap']).correlation
    print(f'Spearman correlation of {col:17s} with true overlap: {rho:.3f}')

# For each species: are its 5 most similar species really found with it?
def top_k_overlap(sim, k=5):
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    top = np.argsort(-s, axis=1)[:, :k]
    hits = true_j[np.arange(n)[:, None], top]
    return (hits > 0).mean(), hits.mean()

print('\nFor each species, its 5 most similar species:')
for name, sim in [('model signatures', model_sim), ('training overlap', train_j)]:
    share, mean_j = top_k_overlap(sim)
    print(f'  by {name:17s}: {share * 100:5.1f}% truly overlap, mean overlap {mean_j:.3f}')
print(f'  random pairs:              {(pairs.true_overlap > 0).mean() * 100:5.1f}% truly '
      f'overlap, mean overlap {pairs.true_overlap.mean():.3f}')

# Pairs the model links that were NEVER seen together in training data
unseen = pairs[pairs.train_overlap == 0]
top_unseen = unseen.nlargest(1000, 'model_similarity')
print(f'\nPairs never recorded together in training ({len(unseen):,}): the model\'s '
      f'1,000 most similar of these truly overlap in '
      f'{(top_unseen.true_overlap > 0).mean() * 100:.1f}% of cases '
      f'(vs {(unseen.true_overlap > 0).mean() * 100:.1f}% for all such pairs)')

print('\nMost similar pairs according to the model:')
print(pairs.nlargest(8, 'model_similarity')[['a', 'b', 'model_similarity', 'true_overlap']]
      .round(3).to_string(index=False))

Path('results/metrics').mkdir(parents=True, exist_ok=True)
pairs.to_csv('results/metrics/cooccurrence_pairs.csv', index=False)

# ---- Figure: model similarity vs true overlap ----
INK, MUTED, SURFACE = '#0b0b0b', '#898781', '#fcfcfb'
fig, ax = plt.subplots(figsize=(5.5, 4), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
hb = ax.hexbin(pairs.model_similarity, pairs.true_overlap, gridsize=45, bins='log',
               cmap='Blues', mincnt=1)
cb = fig.colorbar(hb, ax=ax); cb.set_label('Number of species pairs (log)', fontsize=7)
cb.ax.tick_params(labelsize=7)
ax.set_xlabel('Similarity of model signatures (cosine)', fontsize=8, color=INK)
ax.set_ylabel('True range overlap (Jaccard, expert maps)', fontsize=8, color=INK)
ax.tick_params(colors=MUTED, labelsize=7, labelcolor=INK)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
for s in ['left', 'bottom']: ax.spines[s].set_color(MUTED)
ax.set_title('Do species the model links really co-occur?', fontsize=9, color=INK, loc='left')
fig.tight_layout()
Path('results/figures').mkdir(parents=True, exist_ok=True)
fig.savefig('results/figures/cooccurrence.png', facecolor=SURFACE)
print('\nSaved results/metrics/cooccurrence_pairs.csv and results/figures/cooccurrence.png')