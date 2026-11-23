"""Per-species k-nearest-neighbours with uniform pseudo-absences.

Mathematically the same as training scikit-learn's KNeighborsClassifier
(haversine distance) on [species sightings + shared background points]
for every species, but much faster: the distances from every query point
to the shared background are computed once and reused for all species.
"""
import numpy as np
from sklearn.neighbors import BallTree
from src.pseudo_absences import uniform_sphere


def knn_per_species(train_locs, train_cols, query_locs, n_species, k=15,
                    n_bg=10000, seed=0, verbose=True):
    """Score = fraction of the k nearest training points that are sightings
    of this species (the rest are background pseudo-absences).
    Returns an (n_query, n_species) float32 array in [0, 1].
    """
    rng = np.random.default_rng(seed)
    q = np.radians(query_locs)
    bg = np.radians(uniform_sphere(n_bg, rng))

    # Distances to the k nearest background points: computed ONCE.
    d_bg, _ = BallTree(bg, metric='haversine').query(q, k=k)

    scores = np.zeros((len(q), n_species), dtype=np.float32)
    for j in range(n_species):
        pos = np.radians(train_locs[train_cols == j])
        k_pos = min(k, len(pos))
        d_pos, _ = BallTree(pos, metric='haversine').query(q, k=k_pos)

        # Merge the two candidate lists and keep the k closest overall.
        d_all = np.concatenate([d_pos, d_bg], axis=1)
        is_pos = np.zeros(d_all.shape[1], dtype=bool)
        is_pos[:k_pos] = True
        nearest = np.argsort(d_all, axis=1, kind='stable')[:, :k]
        scores[:, j] = is_pos[nearest].mean(axis=1)

        if verbose and (j + 1) % 100 == 0:
            print(f'  {j + 1}/{n_species} species done')
    return scores