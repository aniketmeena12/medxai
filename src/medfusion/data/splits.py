"""Grouped, stratified splits. A patient (or lesion) never appears on both sides of a split."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def grouped_folds(labels, groups, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Return a fold id (0..n_folds-1) per sample."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    folds = np.full(len(labels), -1, dtype=np.int64)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(labels)), labels, groups)):
        folds[test_idx] = k
    return folds


def inner_val_split(labels, groups, val_fraction: float = 0.15, seed: int = 0) -> np.ndarray:
    """Boolean mask selecting a grouped, stratified validation subset."""
    n_splits = max(2, round(1 / val_fraction))
    folds = grouped_folds(labels, groups, n_folds=n_splits, seed=seed)
    return folds == 0


def assert_disjoint_groups(groups_a, groups_b) -> None:
    overlap = set(np.asarray(groups_a)) & set(np.asarray(groups_b))
    if overlap:
        raise ValueError(f"{len(overlap)} groups leak across the split, e.g. {sorted(overlap)[:3]}")
