"""P2 and P3 from the block-C trap-set runs (docs/04-preregistration.md §5).

    python scripts/evaluate_trap_endpoints.py --endpoints P2 P3

P2  Normalized area under the degradation curve: test AUROC across image-bias levels
    {0, 0.3, 0.5, 0.7, 0.9, 1}. Higher = more shortcut-robust. One curve per published split;
    CIs are t-intervals across the 3 splits (§7).

P3  Grounding loss under a metadata shortcut: the change in energy-inside-lesion-mask between the
    models trained at metadata bias 0 and 0.9, both at image bias 0. Negative = the model looks at
    the lesion less once metadata predicts the label, i.e. offloading.

    The two cells have different test sets (the metadata trap resamples the test set to the
    reversed correlation), so both checkpoints are scored on the **intersection** of their test
    images that carry a HAM10000 lesion mask. Otherwise the "change" would confound a change of
    evidence with a change of evaluation set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import ttest_1samp

from medfusion.eval.endpoints import grounding, load_checkpoint
from medfusion.eval.shortcut import BIAS_LEVELS, DegradationCurve
from medfusion.eval.stats import t_ci
from medfusion.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
MODELS = ["B0", "B3", "B4", "B5", "M1"]


def run_dir(model: str, mode: str) -> Path | None:
    dirs = [d for d in sorted(Path("data/experiments").glob(f"isic2019_trap_{model}_{mode}*"),
                              reverse=True) if "debug" not in d.name]
    return dirs[0] if dirs else None


def p2(splits: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, per_split = [], {}
    for model in MODELS:
        d = run_dir(model, "curve")
        if d is None:
            continue
        cells = [json.loads(f.read_text()) for f in d.glob("img*_nometa_seed*.json")]
        areas = []
        for split in splits:
            scores = {c["image_level"]: c["auroc"] for c in cells if c["split"] == split
                      and not c.get("skipped")}
            if len(scores) < len(BIAS_LEVELS):
                continue
            curve = DegradationCurve(model, scores)
            areas.append(curve.normalized_area())
            rows.append({"model": model, "split": split, **curve.as_row()})
        if areas:
            per_split[model] = np.array(areas)
    summary = []
    for model, a in per_split.items():
        m, lo, hi = t_ci(a)
        summary.append({"endpoint": "P2", "model": model, "n_splits": len(a),
                        "norm_area": m, "ci_lo": lo, "ci_hi": hi})
    if "B0" in per_split:
        for model in [m for m in MODELS if m != "B0" and m in per_split]:
            diff = per_split[model] - per_split["B0"]
            m, lo, hi = t_ci(diff)
            summary.append({"endpoint": "P2", "model": f"{model} - B0", "n_splits": len(diff),
                            "norm_area": m, "ci_lo": lo, "ci_hi": hi,
                            "p": float(ttest_1samp(diff, 0.0).pvalue)})
    return pd.DataFrame(rows), pd.DataFrame(summary)


def _energy_on(run: Path, tag: str, images: set[str], samples: pd.DataFrame,
               device) -> pd.DataFrame | None:
    ck = run / "checkpoints" / f"{tag}.pt"
    if not ck.exists():
        return None
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    model, schema, _ = load_checkpoint(ck, cfg, device)
    df = samples[samples["sample_id"].isin(images)].reset_index(drop=True)
    out = grounding(model, df, schema, cfg, device, "lesion_id", batch_size=32)
    del model
    torch.cuda.empty_cache()
    return out


def p3(splits: list[int]) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(REPO / "configs" / "data_isic2019_trap.yaml")
    samples = pd.read_csv(cfg["data"]["processed_csv"])
    samples = samples[samples["mask_path"].notna()].reset_index(drop=True)
    rows, per_model = [], {}
    for model in MODELS:
        run = run_dir(model, "grid")
        if run is None:
            continue
        shifts = []
        for split in splits:
            tags = {rho: f"img0_split{split}_meta-sex{rho:g}_seed0" for rho in (0, 0.9)}
            preds = {rho: run / f"{t}_test_predictions.csv" for rho, t in tags.items()}
            if not all(p.exists() for p in preds.values()):
                continue
            # Same images for both cells, and only those with a HAM10000 lesion mask.
            sets = []
            for p in preds.values():
                f = pd.read_csv(p)
                sets.append(set(f.loc[f["mask_path"].notna(), "image"]))
            shared = sets[0] & sets[1]
            if not shared:
                continue
            energies = {}
            for rho, tag in tags.items():
                e = _energy_on(run, tag, shared, samples, device)
                if e is None or e.empty:
                    energies = {}
                    break
                energies[rho] = float(np.nanmean(e["energy_in_mask"]))
            if len(energies) != 2:
                continue
            shift = energies[0.9] - energies[0]
            shifts.append(shift)
            rows.append({"model": model, "split": split, "n_shared_masked_images": len(shared),
                         "energy_meta0": energies[0], "energy_meta0.9": energies[0.9],
                         "grounding_loss": shift})
        if shifts:
            per_model[model] = np.array(shifts)
    summary = []
    for model, s in per_model.items():
        m, lo, hi = t_ci(s)
        summary.append({"endpoint": "P3", "model": model, "n_splits": len(s),
                        "grounding_loss": m, "ci_lo": lo, "ci_hi": hi})
    if "B0" in per_model:
        for model in [m for m in MODELS if m != "B0" and m in per_model]:
            diff = per_model[model] - per_model["B0"]
            m, lo, hi = t_ci(diff)
            summary.append({"endpoint": "P3", "model": f"{model} - B0", "n_splits": len(diff),
                            "grounding_loss": m, "ci_lo": lo, "ci_hi": hi,
                            "p": float(ttest_1samp(diff, 0.0).pvalue)})
    return pd.DataFrame(rows), pd.DataFrame(summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", nargs="*", default=["P2", "P3"])
    args = ap.parse_args()
    cfg = load_config(REPO / "configs" / "data_isic2019_trap.yaml")
    splits = cfg["data"]["splits"]
    out = REPO / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)

    if "P2" in args.endpoints:
        curves, summary = p2(splits)
        curves.to_csv(out / "trap_P2_curves.csv", index=False)
        summary.to_csv(out / "trap_P2_summary.csv", index=False)
        print("=== P2: degradation curves (AUROC by image bias level) ===")
        print(curves.round(4).to_string(index=False))
        print()
        print(summary.round(4).to_string(index=False))
    if "P3" in args.endpoints:
        rows, summary = p3(splits)
        rows.to_csv(out / "trap_P3_per_split.csv", index=False)
        summary.to_csv(out / "trap_P3_summary.csv", index=False)
        print("\n=== P3: grounding loss under a metadata shortcut ===")
        print(rows.round(4).to_string(index=False))
        print()
        print(summary.round(4).to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
