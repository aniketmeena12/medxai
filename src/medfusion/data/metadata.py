"""Tabular metadata encoding shared by every fusion model.

Categorical fields -> integer index; index ``cardinality`` means MISSING (a real category, not
imputed). Continuous fields -> standardized float plus a boolean "present" mask.

Vocabularies are built from all rows (label-free, so no leakage). Continuous mean/std are fitted
on training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

MISSING_TOKENS = {"", "nan", "none", "unk", "unknown", "null", "na", "n/a"}


def is_missing(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)


@dataclass
class MetadataSchema:
    categorical: list[str]
    continuous: list[str]
    vocabs: dict[str, list[str]] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.vocabs[c]) for c in self.categorical]

    @property
    def n_fields(self) -> int:
        return len(self.categorical) + len(self.continuous)

    @property
    def field_names(self) -> list[str]:
        return list(self.categorical) + list(self.continuous)

    def fit(self, df: pd.DataFrame, train_mask: np.ndarray | None = None) -> MetadataSchema:
        for col in self.categorical:
            vals = df.loc[~is_missing(df[col]), col].astype(str).str.strip()
            self.vocabs[col] = sorted(vals.unique().tolist())
        train = df if train_mask is None else df[train_mask]
        for col in self.continuous:
            x = pd.to_numeric(train[col], errors="coerce")
            self.means[col] = float(x.mean()) if x.notna().any() else 0.0
            std = float(x.std()) if x.notna().sum() > 1 else 1.0
            self.stds[col] = std if std > 0 else 1.0
        return self

    def encode(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(df)
        cat = np.zeros((n, len(self.categorical)), dtype=np.int64)
        for j, col in enumerate(self.categorical):
            lookup = {v: i for i, v in enumerate(self.vocabs[col])}
            miss = is_missing(df[col]).to_numpy()
            vals = df[col].astype(str).str.strip().to_numpy()
            # Values unseen at fit time are treated as MISSING.
            cat[:, j] = [
                len(lookup) if m else lookup.get(v, len(lookup))
                for v, m in zip(vals, miss, strict=True)
            ]
        cont = np.zeros((n, len(self.continuous)), dtype=np.float32)
        mask = np.zeros((n, len(self.continuous)), dtype=bool)
        for j, col in enumerate(self.continuous):
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            present = ~np.isnan(x)
            cont[present, j] = (x[present] - self.means[col]) / self.stds[col]
            mask[:, j] = present
        return cat, cont, mask

    def to_dict(self) -> dict:
        return {
            "categorical": self.categorical,
            "continuous": self.continuous,
            "vocabs": self.vocabs,
            "means": self.means,
            "stds": self.stds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MetadataSchema:
        return cls(d["categorical"], d["continuous"], d["vocabs"], d["means"], d["stds"])


def field_dropout(
    cat: torch.Tensor,
    cont_mask: torch.Tensor,
    cardinalities: list[int],
    p: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly set whole fields to MISSING with probability ``p`` (training-time regularizer).

    Done at the data level so every fusion model (concat, FiLM, MetaBlock, cross-attention) gets
    exactly the same corruption. Fixes handbook §15 item 7.
    """
    if p <= 0:
        return cat, cont_mask
    drop_cat = torch.rand(cat.shape, generator=generator, device=cat.device) < p
    missing = torch.tensor(cardinalities, device=cat.device).expand_as(cat)
    cat = torch.where(drop_cat, missing, cat)
    drop_cont = torch.rand(cont_mask.shape, generator=generator, device=cont_mask.device) < p
    cont_mask = cont_mask & ~drop_cont
    return cat, cont_mask
