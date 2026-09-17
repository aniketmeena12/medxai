"""Statistics fixed in docs/04-preregistration.md §7. Every random draw is seeded."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats as sps


def mean_ci(values, alpha: float = 0.05, n: int = 10_000, seed: int = 0):
    """Percentile-bootstrap CI of the mean. Returns (mean, lo, hi); NaNs dropped."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(boots, alpha / 2)), float(
        np.quantile(boots, 1 - alpha / 2))


def t_ci(values, alpha: float = 0.05):
    """t-interval across a small number of independent runs (trap-set splits x seeds)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    h = se * sps.t.ppf(1 - alpha / 2, len(v) - 1)
    return float(m), float(m - h), float(m + h)


def cluster_bootstrap_diff(
    groups,
    metric: Callable[[np.ndarray], float],
    n: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
):
    """Paired cluster bootstrap for a difference statistic.

    ``metric(idx)`` receives row indices (clusters resampled with replacement, all rows of a
    cluster kept) and returns model B minus model A on those rows. Returns
    (estimate, lo, hi, p_two_sided).
    """
    groups = np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    rows_by_cluster = [np.flatnonzero(inverse == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    estimate = metric(np.arange(len(groups)))
    boots = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        boots[b] = metric(np.concatenate([rows_by_cluster[k] for k in pick]))
    lo, hi = np.nanquantile(boots, [alpha / 2, 1 - alpha / 2])
    # Two-sided p from the bootstrap distribution (shifted to the null of zero difference).
    p = 2 * min(np.nanmean(boots - estimate >= estimate), np.nanmean(boots - estimate <= estimate))
    return float(estimate), float(lo), float(hi), float(min(1.0, p))


def holm(pvalues, alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni. Returns a boolean array: reject H0 for each test."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    reject = np.zeros(len(p), dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (len(p) - rank):
            reject[i] = True
        else:
            break
    return reject


def tost_equivalent(lo_90: float, hi_90: float, margin: float) -> bool:
    """TOST via the 90% CI of the difference: equivalent if it lies inside (-margin, margin)."""
    return bool(-margin < lo_90 and hi_90 < margin)
