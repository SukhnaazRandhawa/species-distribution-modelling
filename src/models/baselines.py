"""Simple non-learned baselines that every ML model must beat."""
import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0


def frequency_baseline(train_cols, n_species, n_locs):
    """Score every species by how often it appears in training data.

    The score is the same at every location, so this model knows
    WHICH species are common, but nothing about WHERE they live.
    """
    counts = np.bincount(train_cols, minlength=n_species).astype(np.float32)
    probs = counts / counts.sum()
    return np.tile(probs, (n_locs, 1))          # (n_locs, n_species)


def nearest_sighting_baseline(train_locs, train_cols, test_locs, n_species,
                              scale_km=500.0):
    """Score = how close the nearest training sighting of each species is.

    score = exp(-distance / scale_km): 1 at a sighting, falling towards 0
    far away. Distances are great-circle (haversine) distances on Earth.
    """
    test_rad = np.radians(test_locs)            # BallTree haversine needs radians
    scores = np.zeros((len(test_locs), n_species), dtype=np.float32)
    for j in range(n_species):
        sp_locs = np.radians(train_locs[train_cols == j])
        tree = BallTree(sp_locs, metric='haversine')
        dist, _ = tree.query(test_rad, k=1)     # distance in radians
        dist_km = dist[:, 0] * EARTH_RADIUS_KM
        scores[:, j] = np.exp(-dist_km / scale_km)
        if (j + 1) % 50 == 0:
            print(f'  {j + 1}/{n_species} species done')
    return scores