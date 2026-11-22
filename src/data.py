"""Loading and preparing the species datasets."""
from pathlib import Path
import numpy as np

# Folder containing the .npz files, found relative to this file,
# so the code works no matter where you run it from.
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


def load_train(extra: bool = False):
    """Load training sightings.

    Returns
    -------
    locs    : (N, 2) float array of (latitude, longitude)
    ids     : (N,) int array of species IDs (raw iNaturalist taxon IDs)
    species : (S,) array of all species IDs; defines the column order
    names   : dict mapping species ID -> latin name
    """
    fname = 'species_train_extra.npz' if extra else 'species_train.npz'
    d = np.load(DATA_DIR / fname)
    names = dict(zip(d['taxon_ids'], d['taxon_names']))
    return d['train_locs'], d['train_ids'], d['taxon_ids'], names


def load_test():
    """Load test locations and, per species, the indices where it is present."""
    d = np.load(DATA_DIR / 'species_test.npz', allow_pickle=True)
    pos = dict(zip(d['taxon_ids'], d['test_pos_inds']))
    return d['test_locs'], pos


def species_to_columns(species):
    """Fixed mapping from species ID to column index 0..S-1."""
    return {sp: i for i, sp in enumerate(species)}


def build_test_matrix(n_locs, pos, sp_to_col):
    """Answer key: Y[i, j] = True if species j is present at test location i."""
    Y = np.zeros((n_locs, len(sp_to_col)), dtype=bool)
    for sp, inds in pos.items():
        Y[inds, sp_to_col[sp]] = True
    return Y


def load_all(extra: bool = False):
    """Load everything later steps need, in one call."""
    train_locs, train_ids, species, names = load_train(extra)
    test_locs, pos = load_test()

    # Stop immediately if train and test don't cover the same species.
    assert set(species) == set(pos), 'Train and test species differ!'
    sp_to_col = species_to_columns(species)

    train_cols = np.array([sp_to_col[s] for s in train_ids])
    Y_test = build_test_matrix(len(test_locs), pos, sp_to_col)

    return dict(train_locs=train_locs, train_cols=train_cols,
                test_locs=test_locs, Y_test=Y_test,
                species=species, names=names)