"""Step 11a: what is in the extra training data, and how does it relate
to the main training data?"""
import sys
sys.path.append('.')
import numpy as np
from src.data import load_train

main_locs, main_ids, main_species, _ = load_train(extra=False)
extra_locs, extra_ids, extra_species, _ = load_train(extra=True)

print(f'Main  training sightings: {len(main_locs):,}')
print(f'Extra training sightings: {len(extra_locs):,}')
print('Same species list (same order):', np.array_equal(main_species, extra_species))
print('Same set of species:           ', set(main_species) == set(extra_species))

# ---- Sightings per species ----
for name, ids in [('Main', main_ids), ('Extra', extra_ids)]:
    _, c = np.unique(ids, return_counts=True)
    print(f'{name:5s} per species: min {c.min()}, median {int(np.median(c))}, '
          f'mean {c.mean():.0f}, max {c.max()}, at max: {(c == c.max()).sum()}')

# ---- Overlap: are the main sightings also inside the extra file? ----
def keys(locs, ids):
    return set(zip(np.round(locs[:, 0], 5), np.round(locs[:, 1], 5), ids))

main_keys, extra_keys = keys(main_locs, main_ids), keys(extra_locs, extra_ids)
shared = len(main_keys & extra_keys)
print(f'\nUnique (lat, lon, species) records: main {len(main_keys):,}, '
      f'extra {len(extra_keys):,}')
print(f'Main records also in extra: {shared:,} '
      f'({shared / len(main_keys) * 100:.1f}% of main)')

# ---- Where are the sightings? (same rough regions as before) ----
def regions(locs):
    lat, lon = locs[:, 0], locs[:, 1]
    r = {'North America': (lat > 15) & (lon < -50),
         'South/Central America': (lat <= 15) & (lon < -30),
         'Europe': (lat > 35) & (lon >= -30) & (lon < 45),
         'Africa': (lat <= 35) & (lon >= -30) & (lon < 55),
         'Asia': (lat > 0) & (lon >= 45),
         'Oceania/SE Asia': (lat <= 0) & (lon >= 55)}
    return {k: v.mean() * 100 for k, v in r.items()}

print(f"\n{'Region':24s} {'Main %':>8s} {'Extra %':>8s}")
rm, re = regions(main_locs), regions(extra_locs)
for k in rm:
    print(f'{k:24s} {rm[k]:8.1f} {re[k]:8.1f}')

# ---- Which of OUR 500 species are in the extra file? ----
main_set, extra_set = set(main_species), set(extra_species)
shared_sp = main_set & extra_set
print(f'\nSpecies in main: {len(main_set)}, in extra: {len(extra_set)}')
print(f'Species in both: {len(shared_sp)}')
print(f'Extra-only species (not in our test set): {len(extra_set - main_set)}')

is_ours = np.isin(extra_ids, list(main_set))
print(f'Extra sightings of OUR 500 species: {is_ours.sum():,} '
      f'({is_ours.mean() * 100:.1f}% of the extra file)')
print(f'Extra sightings of other species:    {(~is_ours).sum():,}')

col = {sp: i for i, sp in enumerate(main_species)}
cm = np.bincount([col[s] for s in main_ids], minlength=len(main_species))
ce = np.bincount([col[s] for s in extra_ids[is_ours]], minlength=len(main_species))
print(f'\nOf our 500 species, {int((ce > 0).sum())} get extra sightings')
if (ce > 0).any():
    gain = ce[ce > 0]
    print(f'Extra sightings per species (those that get some): '
          f'min {gain.min()}, median {int(np.median(gain))}, max {gain.max()}')
    for lo, hi in [(50, 100), (100, 250), (250, 1000), (1000, 2001)]:
        m = (cm >= lo) & (cm < hi)
        print(f'  species with {lo}-{hi - 1} main sightings: '
              f'{int((ce[m] > 0).sum())}/{int(m.sum())} get extra, '
              f'median extra = {int(np.median(ce[m]))}')