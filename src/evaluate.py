"""Evaluation metrics for species distribution models.

All functions take:
    scores : (L, S) float array, predicted score for species s at location l
    Y      : (L, S) bool array, true presence (the answer key)
Higher score = model thinks the species is more likely present.
"""
import numpy as np


def average_precision_rows(scores, labels, seed=0):
    """Average precision for each row of a 2D array.

    For each row: rank items by score (highest first), then average the
    precision at the rank of every true positive.
    Rows with no positives get NaN (AP is undefined for them).

    Ties (equal scores) are broken in a fixed random order, so a model
    cannot gain or lose AP from the order the data happens to be stored in.
    """
    perm = np.random.default_rng(seed).permutation(scores.shape[1])
    order = perm[np.argsort(-scores[:, perm], axis=1, kind='stable')]
    hits = np.take_along_axis(labels, order, axis=1)          # labels in ranked order
    cum_hits = np.cumsum(hits, axis=1, dtype=np.float32)       # positives found so far
    ranks = np.arange(1, labels.shape[1] + 1, dtype=np.float32)
    precision_at_k = cum_hits / ranks                          # precision at every rank
    n_pos = hits.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        ap = (precision_at_k * hits).sum(axis=1) / n_pos
    return ap


def per_species_ap(scores, Y, mask=None, chunk=25):
    """Q1: AP for each species, ranking all (or masked) test locations.

    mask : optional (L,) bool array to evaluate on a subset of locations,
           e.g. land only.
    Returns an (S,) array.
    """
    if mask is not None:
        scores, Y = scores[mask], Y[mask]
    S = Y.shape[1]
    ap = np.empty(S, dtype=np.float64)
    for start in range(0, S, chunk):                 # chunks keep memory low
        end = min(start + chunk, S)
        ap[start:end] = average_precision_rows(
            scores[:, start:end].T, Y[:, start:end].T)
    return ap


def per_location_ap(scores, Y, mask=None, chunk=20000):
    """Q2: AP for each location, ranking the 500 species.

    Locations with no true species are skipped (NaN).
    Returns an (L,) array.
    """
    if mask is not None:
        scores, Y = scores[mask], Y[mask]
    L = Y.shape[0]
    ap = np.empty(L, dtype=np.float64)
    for start in range(0, L, chunk):
        end = min(start + chunk, L)
        ap[start:end] = average_precision_rows(scores[start:end], Y[start:end])
    return ap


def richness_error(probs, Y, mask=None):
    """Q2: how well predicted species counts match the true counts.

    probs must be probabilities (0..1); their row sum is the expected
    number of species at each location.
    """
    if mask is not None:
        probs, Y = probs[mask], Y[mask]
    pred = probs.sum(axis=1)
    true = Y.sum(axis=1)
    # Correlation is undefined if the model predicts the same count everywhere.
    corr = np.nan if np.allclose(pred, pred[0]) else float(np.corrcoef(pred, true)[0, 1])
    return dict(mae=float(np.abs(pred - true).mean()), corr=corr)


def summarise(scores, Y, mask=None, probs=True):
    """All headline numbers in one dictionary."""
    sp = per_species_ap(scores, Y, mask)
    loc = per_location_ap(scores, Y, mask)
    out = dict(mean_ap_species=float(np.nanmean(sp)),
               mean_ap_location=float(np.nanmean(loc)))
    if probs:
        out.update({f'richness_{k}': v
                    for k, v in richness_error(scores, Y, mask).items()})
    return out