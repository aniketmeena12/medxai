"""Saliency-vs-annotation metrics.

Protocol notes (verified 2026-09-17, docs/research/01-dataset-verification.md):
- CheXlocalize's "hit rate" is the pointing game: the saliency argmax pixel must lie inside the
  ground-truth mask, with no tolerance.
- CheXlocalize binarizes maps with Otsu's threshold by default.
Primary endpoint P1 is ``energy_in_mask`` (docs/04-preregistration.md §5).
"""

from __future__ import annotations

import numpy as np


def _check(saliency: np.ndarray, mask: np.ndarray) -> None:
    if saliency.shape != mask.shape:
        raise ValueError(f"shape mismatch: saliency {saliency.shape} vs mask {mask.shape}")


def energy_in_mask(saliency: np.ndarray, mask: np.ndarray) -> float:
    """Share of positive attribution inside the mask (1 = all inside). NaN if no positive mass."""
    _check(saliency, mask)
    pos = np.clip(saliency, 0, None)
    total = pos.sum()
    return float(pos[mask.astype(bool)].sum() / total) if total > 0 else float("nan")


def mask_area_fraction(mask: np.ndarray) -> float:
    """Chance level for ``energy_in_mask`` under a uniform map; report alongside P1."""
    return float(mask.astype(bool).mean())


def pointing_game(saliency: np.ndarray, mask: np.ndarray) -> bool:
    _check(saliency, mask)
    y, x = np.unravel_index(np.argmax(saliency), saliency.shape)
    return bool(mask.astype(bool)[y, x])


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold on the value histogram (maximizes between-class variance)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    lo, hi = v.min(), v.max()
    if hi <= lo:
        return float(lo)
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    between = (w0 * w1 * (mu0 - mu1) ** 2)[:-1]
    # Empty bins between well-separated modes tie; take the middle of the tied run, not its edge.
    best = np.flatnonzero(between >= between.max() * (1 - 1e-9))
    return float(edges[best[len(best) // 2] + 1])


def binarize(saliency: np.ndarray, method: str = "otsu") -> np.ndarray:
    """``otsu`` (primary, CheXlocalize default) or ``mean_std`` (sensitivity analysis)."""
    if method == "otsu":
        thr = otsu_threshold(saliency)
    elif method == "mean_std":
        thr = saliency.mean() + saliency.std()
    else:
        raise ValueError(f"unknown method {method!r}")
    return saliency > thr


def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    _check(pred_mask, gt_mask)
    p, g = pred_mask.astype(bool), gt_mask.astype(bool)
    union = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / union) if union else float("nan")


def miou(saliencies, gts, method: str = "otsu") -> float:
    pairs = zip(saliencies, gts, strict=True)
    return float(np.nanmean([iou(binarize(s, method), g) for s, g in pairs]))
