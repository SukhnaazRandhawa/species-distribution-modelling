"""Combining two models' scores (ensembles)."""
import numpy as np
from scipy.stats import rankdata


def rank_normalise(scores, chunk=25):
    """Replace each species' scores by their rank among locations, scaled to
    0..1. Ranks make two models comparable even if their scores are on
    different scales (e.g. a probability vs exp(-distance))."""
    out = np.empty(scores.shape, dtype=np.float32)
    n = scores.shape[0]
    for start in range(0, scores.shape[1], chunk):
        block = np.asarray(scores[:, start:start + chunk], dtype=np.float64)
        out[:, start:start + chunk] = rankdata(block, method='average', axis=0) / n
    return out


def combine(rank_a, rank_b, weights):
    """Weighted average per species: w * a + (1 - w) * b.
    weights: one number, or one weight per species (column)."""
    w = np.broadcast_to(np.asarray(weights, dtype=np.float32), (rank_a.shape[1],))
    return w * rank_a + (1 - w) * rank_b


def species_spread_km(locs, cols, n_species):
    """How spread out each species' training sightings are: median
    great-circle distance (km) from the sightings to their centre.
    A rough, training-data-only estimate of range size."""
    lat, lon = np.radians(locs[:, 0]), np.radians(locs[:, 1])
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                    np.sin(lat)], axis=1)                 # points on a unit sphere
    spread = np.zeros(n_species)
    for j in range(n_species):
        p = xyz[cols == j]
        centre = p.mean(axis=0)
        centre /= np.linalg.norm(centre)
        angle = np.arccos(np.clip(p @ centre, -1, 1))     # angle in radians
        spread[j] = np.median(angle) * 6371.0
    return spread