"""Check the per-species land file and add the extra-species and ensemble
rows to land_summary.csv (their numbers were already computed by
rescore_land.py; they were just not written to the summary).

Usage:  python experiments/finish_land_summary.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

M = Path('results/metrics')
ps = pd.read_csv(M / 'per_species_land_ap.csv')
summary = pd.read_csv(M / 'land_summary.csv')

# ---- 1. Is per_species_land_ap.csv really land AP? ----
print('Check: mean of each column in per_species_land_ap.csv vs land_summary.csv')
means = ps.drop(columns=['species', 'n_train']).mean()
land = dict(zip(summary.model, summary.species_ap_land))
alls = dict(zip(summary.model, summary.species_ap))
for col, m in means.items():
    if col in land:
        verdict = 'LAND' if abs(m - land[col]) < 1e-3 else (
            'ALL LOCATIONS' if abs(m - alls[col]) < 1e-3 else '???')
        print(f'  {col:40s} {m:.4f}  land {land[col]:.4f}  all {alls[col]:.4f}  -> {verdict}')
    else:
        print(f'  {col:40s} {m:.4f}')

# ---- 2. Add extra-species (3 seeds) and ensemble rows ----
summary = summary[~summary.model.str.contains('seeds|Ensemble')]   # safe to re-run
seeds = pd.read_csv(M / 'extra_species_seeds.csv')
rows = []
for use_extra, name in [(False, 'Network: location + land flag (3 seeds)'),
                        (True, 'Network: + 1,918 extra species (3 seeds)')]:
    s = seeds[seeds.use_extra == use_extra]
    rows.append(dict(model=name, species_ap=s.species_ap.mean(),
                     species_ap_land=s.species_ap_land.mean(),
                     species_ap_land_std=s.species_ap_land.std()))
ens = json.load(open(M / 'ensemble.json'))['test']
for key, name in [('network_extra=False + baseline (weight by spread)',
                   'Ensemble: network + baseline'),
                  ('network_extra=True + baseline (weight by spread)',
                   'Ensemble: network (extra species) + baseline')]:
    rows.append(dict(model=name, species_ap=ens[key]['all']['mean_ap_species'],
                     species_ap_land=ens[key]['land']['mean_ap_species']))
# location AP is left empty for these: not stored for seeds 1-2, and not
# valid for ensembles (rank-normalised scores aren't comparable across species)
summary = pd.concat([summary, pd.DataFrame(rows)], ignore_index=True).round(4)
summary.to_csv(M / 'land_summary.csv', index=False)
print('\nUpdated land_summary.csv:\n', summary.to_string(index=False))