"""Build data/climate_10m.npz from the WorldClim GeoTIFFs and sanity-check it."""
import sys
sys.path.append('.')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm
from src.climate import prepare_climate_grid, ClimateGrid
from src.data import load_all

TIF_DIR = 'data/wc2.1_10m_bio'          # folder with the 19 unzipped .tif files
OUT = 'data/climate_10m.npz'

if not Path(OUT).exists():
    print('Reading GeoTIFFs...')
    print('Grid shape (variables, rows, cols):', prepare_climate_grid(TIF_DIR, OUT))
g = ClimateGrid(OUT)

# ---- 1. Known places: do the values make sense? ----
places = {'Edinburgh': (55.95, -3.19), 'Sahara (Algeria)': (25.0, 5.0),
          'Amazon (Manaus)': (-3.1, -60.0), 'Antarctica': (-80.0, 0.0),
          'Mid-Pacific (ocean)': (0.0, -150.0)}
raw = g.raw(np.array(list(places.values()), dtype=float))
print(f"\n{'Place':22s} {'Mean temp (BIO1, °C)':>22s} {'Annual rain (BIO12, mm)':>25s}")
for name, (t, p) in zip(places, raw[:, [0, 11]]):
    print(f'{name:22s} {t:22.1f} {p:25.0f}')

# ---- 2. Coverage: how many locations have climate data? ----
d = load_all()
has = lambda locs: (~np.isnan(g.raw(locs)).any(axis=1)).mean() * 100
print(f"\nTraining sightings with climate data: {has(d['train_locs']):.1f}%")
print(f"Test locations with climate data:     {has(d['test_locs']):.1f}%")

# ---- 3. Maps of two variables ----
Path('results/figures').mkdir(parents=True, exist_ok=True)
INK, SURFACE = '#0b0b0b', '#fcfcfb'
temp_cmap = LinearSegmentedColormap.from_list(
    'cold_warm', ['#104281', '#5598e7', '#f0efec', '#ec835a', '#9b2424'])
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
extent = [g.lon0, g.lon0 + g.n_cols * g.cell, g.lat0 - g.n_rows * g.cell, g.lat0]
for ax, idx, title, cmap, norm, unit in [
        (axes[0], 0, 'BIO1: annual mean temperature', temp_cmap,
         TwoSlopeNorm(vcenter=0, vmin=np.nanmin(g.grid[0]), vmax=np.nanmax(g.grid[0])), '°C'),
        (axes[1], 11, 'BIO12: annual precipitation', 'Blues',
         LogNorm(vmin=10, vmax=np.nanmax(g.grid[11])), 'mm')]:
    ax.set_facecolor('#e6e5e0')                  # ocean = no data
    im = ax.imshow(g.grid[idx], extent=extent, cmap=cmap, norm=norm,
                   interpolation='nearest')
    ax.set_title(title, fontsize=9, color=INK, loc='left')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(unit, fontsize=8, color=INK); cb.ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig('results/figures/climate_maps.png', facecolor=SURFACE)
print('\nSaved results/figures/climate_maps.png')