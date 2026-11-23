"""Build the validation split and check it ranks models like the test set."""
import sys, time
sys.path.append('.')
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.data import load_all
from src.validation import make_validation_split
from src.evaluate import per_species_ap
from src.models.baselines import nearest_sighting_baseline
from src.models.per_species import fit_predict_per_species
from src.validation import make_validation_split, val_ap_vs_background

d = load_all()
n_species = len(d['species'])
s = make_validation_split(d['train_locs'], d['train_cols'], n_species)
Y_val = s['Y_val']

# ---- Describe the split ----
n_train, n_val_pos = len(s['train_locs']), int(Y_val.sum())
print(f'Train sightings:       {n_train}')
print(f'Validation sightings:  {n_val_pos} '
      f'({n_val_pos / len(d["train_locs"]) * 100:.1f}% of all)')
print(f'Validation locations:  {len(s["val_locs"])} (incl. 10,000 random)')
print('Species with no validation sightings:', int((Y_val.sum(0) == 0).sum()))
print('Fewest training sightings for a species:',
      np.bincount(s['train_cols'], minlength=n_species).min())

# ---- Score the three known models on validation ----
def val_score(name, scores):
    ap_all = np.nanmean(per_species_ap(scores, Y_val))
    ap_bg = np.nanmean(val_ap_vs_background(scores, Y_val))
    print(f'{name:22s} val AP (all negatives) = {ap_all:.3f}   '
          f'val AP (vs background) = {ap_bg:.3f}')

make_lr = lambda: LogisticRegression(class_weight='balanced', max_iter=1000)
t = time.time()
val_score('Nearest sighting', nearest_sighting_baseline(
    s['train_locs'], s['train_cols'], s['val_locs'], n_species))
for bg in ['uniform', 'target_group']:
    val_score(f'LogReg {bg}', fit_predict_per_species(
        make_lr, s['train_locs'], s['train_cols'], s['val_locs'],
        n_species, background=bg, verbose=False))
print(f'({time.time() - t:.0f}s)')