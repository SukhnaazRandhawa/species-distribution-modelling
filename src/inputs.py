"""Build the input features for the three settings of the main experiment."""
import numpy as np
from src.features import encode_coords
from src.climate import ClimateGrid

_GRID = None


def _climate_grid(file='data/climate_10m.npz'):
    global _GRID                      # load the climate file only once
    if _GRID is None:
        _GRID = ClimateGrid(file)
    return _GRID


def make_featurizer(inputs, train_locs, n_freq=4):
    """Return a function locs -> feature matrix.

    inputs : 'location' -> sine/cosine of lat/lon
             'climate'  -> 19 standardised WorldClim variables + land flag
             'both'     -> both, side by side
    The climate scaling is fitted on train_locs only (no test information).
    """
    if inputs == 'location':
        return lambda locs: encode_coords(locs, n_freq)

    grid = _climate_grid()
    grid.fit_scaling(train_locs)
    mean, std = grid.mean.copy(), grid.std.copy()   # freeze this run's scaling

    def climate(locs):
        grid.mean, grid.std = mean, std
        return grid.features(locs)

    if inputs == 'climate':
        return climate
    if inputs == 'location_land':
        return lambda locs: np.concatenate(
        [encode_coords(locs, n_freq), climate(locs)[:, -1:]], axis=1)
    if inputs == 'both':
        return lambda locs: np.concatenate(
            [encode_coords(locs, n_freq), climate(locs)], axis=1)
    raise ValueError(f'unknown inputs: {inputs}')