"""Shortcut metrics: degradation over bias levels (P2) and the offloading index (P3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Published levels of the ISIC 2019 trap sets (Bissoto et al., ISIC@ECCV 2022).
BIAS_LEVELS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]


@dataclass
class DegradationCurve:
    model: str
    scores: dict[float, float]  # bias level -> test AUROC

    def normalized_area(self) -> float:
        """P2: trapezoid area under score-vs-level, divided by the level range (1 = no drop at
        perfect score). Higher is more shortcut-robust."""
        levels = np.array(sorted(self.scores))
        values = np.array([self.scores[lv] for lv in levels])
        if len(levels) < 2:
            raise ValueError("need at least two bias levels")
        return float(np.trapezoid(values, levels) / (levels[-1] - levels[0]))

    def drop(self) -> float:
        levels = sorted(self.scores)
        return float(self.scores[levels[0]] - self.scores[levels[-1]])

    def as_row(self) -> dict:
        row = {"model": self.model, "norm_area": self.normalized_area(), "drop": self.drop()}
        row.update({f"bias_{lv:g}": s for lv, s in sorted(self.scores.items())})
        return row


def offloading_index(energy_no_bias: np.ndarray, energy_with_bias: np.ndarray) -> float:
    """P3: mean change in lesion energy when the metadata shortcut is introduced.
    Negative = the model looks at the lesion less (offloading)."""
    return float(np.nanmean(energy_with_bias) - np.nanmean(energy_no_bias))


def stratified_auc(y_true, y_score, stratum) -> dict[str, float]:
    """AUROC per binary stratum and the gap (e.g. artifact present vs absent)."""
    from sklearn.metrics import roc_auc_score

    y_true, y_score, stratum = map(np.asarray, (y_true, y_score, stratum))
    out = {}
    for name, sel in (("present", stratum.astype(bool)), ("absent", ~stratum.astype(bool))):
        ok = sel.any() and len(np.unique(y_true[sel])) == 2
        out[name] = float(roc_auc_score(y_true[sel], y_score[sel])) if ok else float("nan")
    out["gap"] = out["present"] - out["absent"]
    return out
