import numpy as np
from global_land_mask import globe

locs = np.load('species_test.npz', allow_pickle=True)['test_locs']
on_land = globe.is_land(locs[:, 0], locs[:, 1])
print(f'Test locations on land:  {on_land.mean()*100:.1f}%')
print(f'Test locations in ocean: {(~on_land).mean()*100:.1f}%')
print('Latitude range: ', locs[:, 0].min(), locs[:, 0].max())
print('Longitude range:', locs[:, 1].min(), locs[:, 1].max())