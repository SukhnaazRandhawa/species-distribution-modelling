"""Spatial validation split built from the training data only."""
import numpy as np
from src.pseudo_absences import uniform_sphere


def spatial_blocks(locs, block_deg):
    """ID of the block_deg x block_deg grid cell each location falls in."""
    row = np.floor((locs[:, 0] + 90) / block_deg).astype(int)
    col = np.floor((locs[:, 1] + 180) / block_deg).astype(int)
    return row * 1000 + col


def make_validation_split(train_locs, train_cols, n_species, block_deg=5.0,
                          val_frac=0.2, n_val_bg=10000, min_train=10, seed=0):
    """Hold out whole grid cells of training sightings for validation.

    1. Split the world into block_deg x block_deg cells and put a random
       val_frac of the cells (with all their sightings) in validation.
    2. If a species would keep fewer than min_train training sightings,
       give its sightings back to training (it is then not validated).
    3. Validation labels: each held-out sighting marks its own species as
       present; every other species counts as absent there (presence-only
       proxy). n_val_bg uniform random points are added with no species
       present, so oceans and unvisited areas are also checked.
    """
    rng = np.random.default_rng(seed)
    blocks = spatial_blocks(train_locs, block_deg)
    unique_blocks = np.unique(blocks)
    val_blocks = rng.choice(unique_blocks, int(val_frac * len(unique_blocks)),
                            replace=False)
    is_val = np.isin(blocks, val_blocks)

    # Make sure every species keeps enough training data.
    for j in range(n_species):
        sp = train_cols == j
        if (sp & ~is_val).sum() < min_train:
            is_val[sp] = False

    val_pos_locs, val_pos_cols = train_locs[is_val], train_cols[is_val]
    bg = uniform_sphere(n_val_bg, rng)
    val_locs = np.concatenate([val_pos_locs, bg])

    Y_val = np.zeros((len(val_locs), n_species), dtype=bool)
    Y_val[np.arange(len(val_pos_cols)), val_pos_cols] = True

    return dict(train_locs=train_locs[~is_val], train_cols=train_cols[~is_val],
                val_locs=val_locs, Y_val=Y_val)
    
def val_ap_vs_background(scores, Y_val, n_val_bg=10000):
    """Per-species validation AP using ONLY random background as negatives.

    For species j, rank its own held-out sightings against the random
    background points. Other species' sightings are ignored, because the
    species may well live there unrecorded (false 'absences').
    Assumes the background points are the last n_val_bg rows (as built by
    make_validation_split). Species with no held-out sightings get NaN.
    """
    from src.evaluate import average_precision_rows
    n_species = Y_val.shape[1]
    bg_rows = np.arange(len(Y_val) - n_val_bg, len(Y_val))
    ap = np.full(n_species, np.nan)
    for j in range(n_species):
        pos_rows = np.flatnonzero(Y_val[:, j])
        if len(pos_rows) == 0:
            continue
        rows = np.concatenate([pos_rows, bg_rows])
        ap[j] = average_precision_rows(scores[rows, j][None, :],
                                       Y_val[rows, j][None, :])[0]
    return ap