"""Features derived from locations."""
import numpy as np


def land_mask(locs):
    """True where a (lat, lon) location is on land."""
    from global_land_mask import globe
    return globe.is_land(locs[:, 0], locs[:, 1])


def encode_coords(locs, n_freq=4):
    """Turn (lat, lon) in degrees into sine/cosine features.

    For each frequency k = 1..n_freq we add sin(k*lat), cos(k*lat),
    sin(k*lon), cos(k*lon) (angles in radians). Low k captures broad
    patterns, higher k captures finer detail. Longitude wraps around
    correctly: -180 and 180 degrees give identical features.
    Returns an (N, 4 * n_freq) float32 array.
    """
    lat = np.radians(locs[:, 0])[:, None]
    lon = np.radians(locs[:, 1])[:, None]
    k = np.arange(1, n_freq + 1)[None, :]
    feats = [np.sin(k * lat), np.cos(k * lat), np.sin(k * lon), np.cos(k * lon)]
    return np.concatenate(feats, axis=1).astype(np.float32)