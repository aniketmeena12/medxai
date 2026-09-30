"""Assemble the pre-registered family of tests, apply Holm-Bonferroni, and run the TOST
equivalence check for the "same accuracy" claim (docs/04-preregistration.md §7).

    python scripts/finalize_family.py

**Which dataset each endpoint comes from.** §5 lists P4 and P5 against more than one dataset while
§7 counts each endpoint once, so the primary dataset per endpoint is fixed here as the one the
endpoint was designed for, and the others are reported as secondary, uncorrected:

    P1 grounding      -> B (HAM10000; the only set with lesion masks)
    P2 image shortcut -> C (ISIC 2019 trap sets)
    P3 offloading     -> C
    P4 counterfactual -> A (PAD-UFES-20; 21 fields incl. the pre-specified irrelevant ones)
    P5 reliance       -> A (21 real metadata fields; HAM10000 has only age/sex/site)

That gives 4 models x 4 endpoints (P1, P2, P3, P5) + 3 P4 comparisons = the 19 tests of §7.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from medfusion.eval.stats import holm, mean_ci, tost_equivalent

TABLES = "results/tables"
FUSION = ["B3", "B4", "B5", "M1"]
EXPECTED = 19


def _rows(path: str, endpoint: str, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["endpoint"] == endpoint].copy()
    df["dataset"] = dataset
    return df[["endpoint", "dataset", "comparison", "estimate", "ci_lo", "ci_hi", "p"]]


def _trap_rows(path: str, endpoint: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["model"].str.contains(" - B0", na=False)].copy()
    df["endpoint"] = endpoint
    df["dataset"] = "C"
    df = df.rename(columns={"model": "comparison", value_col: "estimate"})
    return df[["endpoint", "dataset", "comparison", "estimate", "ci_lo", "ci_hi", "p"]]


def family() -> pd.DataFrame:
    parts = [
        _rows(f"{TABLES}/endpoints_ham10000_gradcam_tests.csv", "P1", "B"),
        _trap_rows(f"{TABLES}/trap_P2_summary.csv", "P2", "norm_area"),
        _trap_rows(f"{TABLES}/trap_P3_summary.csv", "P3", "grounding_loss"),
        _rows(f"{TABLES}/endpoints_padufes20_gradcam_tests.csv", "P5", "A"),
        _rows(f"{TABLES}/endpoints_padufes20_gradcam_tests.csv", "P4", "A"),
    ]
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["endpoint", "comparison"]).reset_index(drop=True)
    if len(df) != EXPECTED:
        raise SystemExit(
            f"family has {len(df)} tests, expected {EXPECTED}:\n"
            + df.groupby("endpoint").size().to_string()
            + "\nRun the missing endpoint before finalizing.")
    df["reject_holm"] = holm(df["p"].to_numpy(), alpha=0.05)
    return df


def tost_accuracy() -> pd.DataFrame:
    """§7: the 'same accuracy' claim, TOST on balanced accuracy with a +/-2 point margin."""
    import glob
    import json

    rows = []
    for dataset in ("padufes20", "ham10000"):
        runs = {}
        for model in ["B0", *FUSION]:
            fs = sorted(glob.glob(f"data/experiments/{dataset}_{model}_*/metrics.json"))
            d = next((json.load(open(f)) for f in reversed(fs)
                      if json.load(open(f))["complete_protocol"]), None)
            if d:
                runs[model] = {(r["seed"], r["fold"]): r["balanced_accuracy"]
                               for r in d["per_fold_seed"]}
        if "B0" not in runs:
            continue
        for model in FUSION:
            if model not in runs:
                continue
            keys = sorted(runs["B0"])
            diff = np.array([runs[model][k] - runs["B0"][k] for k in keys])
            # 90% CI of the difference is what TOST at alpha=0.05 requires.
            mean, lo90, hi90 = mean_ci(diff, alpha=0.10)
            rows.append({"dataset": dataset, "comparison": f"{model} - B0",
                         "mean_bacc_diff": mean, "ci90_lo": lo90, "ci90_hi": hi90,
                         "equivalent_within_2pts": tost_equivalent(lo90, hi90, 0.02)})
    return pd.DataFrame(rows)


def main() -> None:
    df = family()
    df.to_csv(f"{TABLES}/family_holm.csv", index=False)
    print("=== Pre-registered family of 19 tests, Holm-Bonferroni at FWER 0.05 ===")
    print(df.round(4).to_string(index=False))
    print(f"\nrejected: {int(df['reject_holm'].sum())}/{len(df)}")

    tost = tost_accuracy()
    tost.to_csv(f"{TABLES}/tost_accuracy.csv", index=False)
    print("\n=== 'Same accuracy' claim: TOST on balanced accuracy, margin +/-2 points ===")
    print(tost.round(4).to_string(index=False))
    print(f"\nwritten to {TABLES}/")


if __name__ == "__main__":
    main()
