"""Generating pseudo-absence ("probably not here") points."""
import numpy as np


def uniform_sphere(n, rng):
    """n locations spread evenly over the Earth's surface (land and ocean).

    Longitude is uniform. Latitude is NOT uniform: there is much less
    area near the poles, so we sample sin(latitude) uniformly instead.
    """
    lon = rng.uniform(-180, 180, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    return np.stack([lat, lon], axis=1)


def target_group(train_locs, train_cols, species, n, rng):
    """n locations sampled from sightings of OTHER species.

    These are places people actually visited and recorded wildlife,
    so they share the same sampling bias as the positives.
    """
    others = np.flatnonzero(train_cols != species)
    idx = rng.choice(others, size=n, replace=len(others) < n)
    return train_locs[idx]