import numpy as np

locs = np.load('species_train.npz')['train_locs']
lat, lon = locs[:, 0], locs[:, 1]

regions = {
    'North America':   (lat > 15) & (lon < -50),
    'South/Central America': (lat <= 15) & (lon < -30),
    'Europe':          (lat > 35) & (lon >= -30) & (lon < 45),
    'Africa':          (lat <= 35) & (lon >= -30) & (lon < 55),
    'Asia':            (lat > 0) & (lon >= 45),
    'Oceania/SE Asia': (lat <= 0) & (lon >= 55),
}
for name, mask in regions.items():
    print(f'{name:25s} {mask.mean()*100:5.1f}%')