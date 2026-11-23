"""Is the network's loss vs the baseline linked to small or coastal ranges?"""
import pandas as pd
from scipy.stats import spearmanr

t = pd.read_csv('results/metrics/per_species_ap.csv')
t['gap'] = t['shared_mlp_full'] - t['nearest_sighting']

for col in ['range_size', 'ocean_share', 'n_train']:
    rho, p = spearmanr(t[col], t['gap'])
    print(f'{col:12s} vs network-minus-baseline: Spearman rho = {rho:+.2f} (p = {p:.1g})')

t['range_group'] = pd.qcut(t['range_size'], 4,
                           labels=['smallest 25%', 'small', 'large', 'largest 25%'])
print('\nMean AP by expert range size:')
print(t.groupby('range_group', observed=True)[
    ['nearest_sighting', 'random_forest', 'shared_mlp_full']].mean().round(3))