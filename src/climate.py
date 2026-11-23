"""WorldClim 2.1 bioclimatic variables (Fick & Hijmans, 2017).

prepare_climate_grid() reads the 19 GeoTIFF files once and saves one compact
array; ClimateGrid then looks up the climate at any (lat, lon) instantly.
"""
from pathlib import Path
import numpy as np

BIO_NAMES = {
    1: 'Annual mean temperature', 2: 'Mean diurnal range',
    3: 'Isothermality', 4: 'Temperature seasonality',
    5: 'Max temperature of warmest month', 6: 'Min temperature of coldest month',
    7: 'Temperature annual range', 8: 'Mean temperature of wettest quarter',
    9: 'Mean temperature of driest quarter', 10: 'Mean temperature of warmest quarter',
    11: 'Mean temperature of coldest quarter', 12: 'Annual precipitation',
    13: 'Precipitation of wettest month', 14: 'Precipitation of driest month',
    15: 'Precipitation seasonality', 16: 'Precipitation of wettest quarter',
    17: 'Precipitation of driest quarter', 18: 'Precipitation of warmest quarter',
    19: 'Precipitation of coldest quarter',
}


def prepare_climate_grid(tif_dir, out_file):
    """Read bio_1..bio_19 GeoTIFFs and save them as one array.

    Saves raw values (float32, NaN over the ocean) with shape
    (19, n_rows, n_cols), plus the grid's geographic layout.
    """
    import rasterio
    layers, transform = [], None
    for i in range(1, 20):
        path = next(Path(tif_dir).glob(f'*bio_{i}.tif'))
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            if src.nodata is not None:
                band[band == src.nodata] = np.nan
            band[band < -1e30] = np.nan               # safety for float nodata
            transform = src.transform
        layers.append(band)
    grid = np.stack(layers)
    np.savez_compressed(out_file, grid=grid,
                        lon0=transform.c, lat0=transform.f,   # top-left corner
                        cell=transform.a)                       # cell size (deg)
    return grid.shape


class ClimateGrid:
    """Fast lookup of standardised climate features at (lat, lon) locations."""

    def __init__(self, file='data/climate_10m.npz'):
        d = np.load(file)
        self.grid = d['grid']
        self.lon0, self.lat0, self.cell = float(d['lon0']), float(d['lat0']), float(d['cell'])
        self.n_rows, self.n_cols = self.grid.shape[1:]
        self.mean = None

    def raw(self, locs):
        """(N, 19) raw climate values; NaN where there is no land."""
        row = np.floor((self.lat0 - locs[:, 0]) / self.cell).astype(int)
        col = np.floor((locs[:, 1] - self.lon0) / self.cell).astype(int)
        row = np.clip(row, 0, self.n_rows - 1)
        col = np.clip(col, 0, self.n_cols - 1)
        return self.grid[:, row, col].T

    def fit_scaling(self, locs):
        """Learn mean/std of each variable from TRAINING locations only."""
        x = self.raw(locs)
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0) + 1e-6

    def features(self, locs):
        """(N, 20): 19 standardised climate variables + a land indicator.

        Ocean locations (no climate data) get 0 for every variable, i.e. the
        training average, and 0 for the land indicator.
        """
        assert self.mean is not None, 'call fit_scaling(train_locs) first'
        x = (self.raw(locs) - self.mean) / self.std
        is_land = ~np.isnan(x).any(axis=1)
        x = np.nan_to_num(x, nan=0.0)
        return np.concatenate([x, is_land[:, None]], axis=1).astype(np.float32)