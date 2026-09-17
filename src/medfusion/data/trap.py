"""Shortcut trap sets.

Image-artifact traps: published CSVs from Bissoto et al. (ISIC@ECCV 2022),
``trap_sets/{train,val,test}_bias_{level}_{split}.csv``.

Metadata traps (contribution N1, docs/04-preregistration.md §6): resample so that a binary
metadata attribute agrees with the label with probability ``0.5 + rho/2`` in training and
``0.5 - rho/2`` at test (reversed), keeping each class's size as large as possible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _level_str(level: float) -> str:
    return "1" if float(level) == 1 else ("0" if float(level) == 0 else f"{float(level):g}")


def load_image_trap(trap_dir: str | Path, level: float, split: int) -> dict[str, pd.DataFrame]:
    trap_dir = Path(trap_dir)
    out = {}
    for part in ("train", "val", "test"):
        df = pd.read_csv(trap_dir / f"{part}_bias_{_level_str(level)}_{split}.csv")
        out[part] = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    return out


def agreement_fraction(attr: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean(np.asarray(attr).astype(bool) == np.asarray(label).astype(bool)))


def resample_metadata_trap(
    df: pd.DataFrame,
    attr: np.ndarray,
    label_col: str,
    rho: float,
    reverse: bool,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Subsample ``df`` so that P(attr == label) = 0.5 + rho/2 (or 0.5 - rho/2 if reversed).

    ``attr`` is a 0/1 array aligned with ``df``; rows with NaN attr are dropped. Class proportions
    are preserved; each class is kept as large as the four (label, attr) cells allow.
    """
    if not 0 <= rho <= 1:
        raise ValueError("rho must be in [0, 1]")
    attr = np.asarray(attr, dtype=float)
    keep = ~np.isnan(attr)
    df, attr = df[keep].reset_index(drop=True), attr[keep].astype(int)
    y = df[label_col].to_numpy().astype(int)
    p_agree = 0.5 - rho / 2 if reverse else 0.5 + rho / 2

    cells = {(c, a): np.flatnonzero((y == c) & (attr == a)) for c in (0, 1) for a in (0, 1)}
    # Largest per-class size n_c with round(p*n_c) agree rows and the rest disagree rows available.
    scale = np.inf
    for c in (0, 1):
        n_class = (y == c).sum()
        n_agree_avail, n_dis_avail = len(cells[(c, c)]), len(cells[(c, 1 - c)])
        limits = [n_class]
        if p_agree > 0:
            limits.append(n_agree_avail / p_agree)
        if p_agree < 1:
            limits.append(n_dis_avail / (1 - p_agree))
        scale = min(scale, min(limits) / n_class)
    chosen = []
    for c in (0, 1):
        n_c = int(np.floor((y == c).sum() * scale))
        n_agree = int(round(p_agree * n_c))
        n_agree = min(n_agree, len(cells[(c, c)]))
        n_dis = min(n_c - n_agree, len(cells[(c, 1 - c)]))
        chosen.append(rng.choice(cells[(c, c)], n_agree, replace=False))
        chosen.append(rng.choice(cells[(c, 1 - c)], n_dis, replace=False))
    idx = np.sort(np.concatenate(chosen))
    return df.iloc[idx].reset_index(drop=True)


def synthetic_attribute(
    label: np.ndarray, rho: float, reverse: bool, rng: np.random.Generator
) -> np.ndarray:
    """Binary token agreeing with the label with probability 0.5 +/- rho/2 (no subsampling)."""
    label = np.asarray(label).astype(int)
    p_agree = 0.5 - rho / 2 if reverse else 0.5 + rho / 2
    agree = rng.random(len(label)) < p_agree
    return np.where(agree, label, 1 - label)
