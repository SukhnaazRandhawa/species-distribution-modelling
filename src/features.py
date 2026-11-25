"""Features derived from locations."""
import numpy as np


_LAND_GRID = None


def land_mask(locs, file='data/climate_10m.npz'):
    """True where a (lat, lon) location is on land.

    Land = WorldClim has data for all 19 variables at that location. This is
    the group's shared definition of land, and the same rule as the land flag
    the models use (ClimateGrid.features), so models are scored on the same
    land they were given. (Earlier versions used the global_land_mask package.)
    """
    global _LAND_GRID
    from src.climate import ClimateGrid
    if _LAND_GRID is None:                       # load the climate file once
        _LAND_GRID = ClimateGrid(file)
    return ~np.isnan(_LAND_GRID.raw(locs)).any(axis=1)


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


def to_radians(locs):
    """(lat, lon) in degrees -> radians, as needed by haversine distance."""
    return np.radians(locs).astype(np.float64)