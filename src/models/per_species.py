"""One independent binary classifier per species."""
import numpy as np
from src.features import encode_coords
from src.pseudo_absences import uniform_sphere, target_group


def fit_predict_per_species(make_model, train_locs, train_cols, test_locs,
                            n_species, background='uniform', n_bg=10000,
                            n_freq=4, featurize=None, seed=0, verbose=True):
    """Train one classifier per species and predict at the test locations.

    make_model : function returning a fresh scikit-learn classifier
    background : 'uniform' or 'target_group' pseudo-absence strategy
    n_bg       : number of pseudo-absences per species
    featurize  : function turning (N, 2) lat/lon into model inputs;
                 default is sine/cosine encoding with n_freq frequencies
    Returns an (n_test, n_species) float32 array of probabilities.
    """
    if featurize is None:
        featurize = lambda locs: encode_coords(locs, n_freq)
    rng = np.random.default_rng(seed)
    X_test = featurize(test_locs)
    scores = np.zeros((len(test_locs), n_species), dtype=np.float32)

    if background == 'uniform':            # same background for every species
        bg = uniform_sphere(n_bg, rng)

    for j in range(n_species):
        pos = train_locs[train_cols == j]
        if background == 'target_group':
            bg = target_group(train_locs, train_cols, j, n_bg, rng)

        X = featurize(np.concatenate([pos, bg]))
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(bg))])

        model = make_model()
        model.fit(X, y)
        scores[:, j] = model.predict_proba(X_test)[:, 1]

        if verbose and (j + 1) % 100 == 0:
            print(f'  {j + 1}/{n_species} species done')
    return scores