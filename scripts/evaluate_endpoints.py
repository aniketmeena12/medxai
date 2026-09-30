"""Compute the pre-registered endpoints from finished protocol runs.

    # per-image endpoint values (resumable; skips (seed, fold) pairs already computed)
    python scripts/evaluate_endpoints.py --data configs/data_ham10000.yaml  --endpoints P1 P5
    python scripts/evaluate_endpoints.py --data configs/data_padufes20.yaml --endpoints P4 P5
    # aggregate what exists into results/tables/ with cluster-bootstrap CIs
    python scripts/evaluate_endpoints.py --data configs/data_ham10000.yaml --aggregate

Per-image values are written to ``<run_dir>/endpoints/<seed><fold>_<endpoint>.csv``; the aggregate
step pools them per seed and averages over seeds (docs/04-preregistration.md §7), with 95%
patient-cluster bootstrap CIs and two-sided bootstrap p-values.

**Holm correction:** §7 fixes one family of 19 tests spanning P1-P3 and P5 (fusion vs B0) and P4
(M1 vs the other fusion models). P2/P3 come from the trap-set runs, so until those exist this
script reports **uncorrected** p-values and says so; pass ``--family-complete`` once every test in
the family is available to apply Holm across all of them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score

from medfusion.eval import endpoints as E
from medfusion.eval.stats import cluster_bootstrap_diff, holm
from medfusion.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
MODELS = ["B0", "B3", "B4", "B5", "M1"]  # B1 has no image branch, so no attribution endpoints


def complete_run(dataset: str, model: str) -> Path | None:
    """Newest run of this model that finished the whole protocol."""
    for d in sorted(Path("data/experiments").glob(f"{dataset}_{model}_[0-9]*"), reverse=True):
        m = d / "metrics.json"
        if m.exists() and json.loads(m.read_text()).get("complete_protocol"):
            return d
    return None


def compute(dataset: str, models: list[str], endpoints: list[str], method: str,
            target: str, limit: int | None, batch_size: int = 32) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_id in models:
        run = complete_run(dataset, model_id)
        if run is None:
            print(f"{model_id}: no complete run, skipping")
            continue
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        d = cfg["data"]
        df = pd.read_csv(d["processed_csv"])
        out_dir = run / "endpoints"
        out_dir.mkdir(exist_ok=True)
        for ck in sorted((run / "checkpoints").glob("seed*_fold*.pt")):
            seed = int(ck.stem.split("_")[0][4:])
            fold = int(ck.stem.split("_")[1][4:])
            suffix = f"_limit{limit}" if limit else ""
            todo = [e for e in endpoints
                    if not (out_dir / f"seed{seed}_fold{fold}_{e}_{method}{suffix}.csv").exists()]
            if not todo:
                continue
            test = df[df["fold"] == fold].reset_index(drop=True)
            if limit:
                test = test.head(limit)
            model, schema, _ = E.load_checkpoint(ck, cfg, device)
            for endpoint in todo:
                if endpoint == "P1":
                    if "mask_path" not in test.columns:
                        print(f"{model_id}: no masks in {d['name']}, skipping P1")
                        continue
                    frame = E.grounding(model, test, schema, cfg, device, d["group_col"],
                                        method=method, target=target, batch_size=batch_size)
                elif endpoint == "P4":
                    cf = d.get("counterfactual", {})
                    fields = {k: v for k, v in cf.items() if v}
                    if not fields:
                        print(f"{model_id}: no counterfactual fields for {d['name']}")
                        continue
                    frame = E.counterfactual(model, test, schema, cfg, device, d["group_col"],
                                             fields, method=method, target=target,
                                             batch_size=batch_size)
                elif endpoint == "P5":
                    frame = E.reliance(model, test, schema, cfg, device,
                                       batch_size=batch_size)
                    frame[d["group_col"]] = test[d["group_col"]].to_numpy()[frame["index"]]
                else:
                    raise ValueError(f"{endpoint} is not computed here (P2/P3: trap-set runs)")
                frame.insert(0, "seed", seed)
                frame.insert(1, "fold", fold)
                frame.to_csv(
                    out_dir / f"seed{seed}_fold{fold}_{endpoint}_{method}{suffix}.csv",
                    index=False)
                print(f"{model_id} seed{seed} fold{fold} {endpoint}: {len(frame)} rows")
            del model
            torch.cuda.empty_cache()


def load_endpoint(dataset: str, model: str, endpoint: str, method: str) -> pd.DataFrame | None:
    run = complete_run(dataset, model)
    if run is None:
        return None
    files = [f for f in sorted((run / "endpoints").glob(f"seed*_fold*_{endpoint}_{method}*.csv"))
             if "_limit" not in f.name]  # debug subsets never enter an aggregate
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _per_seed_mean(frame: pd.DataFrame, rows: np.ndarray, column: str) -> float:
    """Pool rows within each seed, then average the per-seed means (§7's unit of analysis)."""
    sub = frame.iloc[rows]
    return float(sub.groupby("seed")[column].mean().mean())


def _per_seed_drop(frame: pd.DataFrame, rows: np.ndarray, condition: str) -> float:
    sub = frame.iloc[rows]
    drops = []
    for _, g in sub.groupby("seed"):
        if g["y"].nunique() < 2:
            continue
        drops.append(balanced_accuracy_score(g["y"], g["pred_full"])
                     - balanced_accuracy_score(g["y"], g[condition]))
    return float(np.mean(drops)) if drops else float("nan")


def paired(a: pd.DataFrame, b: pd.DataFrame, group_col: str, statistic) -> dict:
    """Align two models' per-image rows and cluster-bootstrap the difference (b - a)."""
    keys = ["seed", "fold", "sample_id"] + (["field"] if "field" in a.columns else [])
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"))
    if merged.empty:
        return {}
    groups = merged[f"{group_col}_a"] if f"{group_col}_a" in merged else merged[group_col]
    est, lo, hi, p = cluster_bootstrap_diff(groups.to_numpy(),
                                            lambda idx: statistic(merged, idx))
    return {"estimate": est, "ci_lo": lo, "ci_hi": hi, "p": p, "n_rows": len(merged),
            "n_clusters": int(groups.nunique())}


def aggregate(dataset: str, method: str, family_complete: bool) -> None:
    cfg = load_config(REPO / "configs" / f"data_{dataset}.yaml")
    group_col = cfg["data"]["group_col"]
    out = REPO / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    descriptive, tests = [], []

    # --- P1 grounding: energy in the lesion mask, each fusion model vs B0 ------------------
    frames = {m: load_endpoint(dataset, m, "P1", method) for m in MODELS}
    frames = {m: f for m, f in frames.items() if f is not None}
    for m, f in frames.items():
        for col in ("energy_in_mask", "mask_area_fraction", "pointing_game", "iou_otsu"):
            descriptive.append({"endpoint": "P1", "model": m, "metric": col,
                                "value": _per_seed_mean(f, np.arange(len(f)), col),
                                "n_rows": len(f), "n_missing": int(f[col].isna().sum())})
        # A map with no positive mass yields NaN energy and is skipped by every mean below;
        # report how often that happened so it is never silently dropped.
        descriptive.append({"endpoint": "P1", "model": m, "metric": "maps_without_attribution",
                            "value": float(f["energy_in_mask"].isna().mean()),
                            "n_rows": len(f), "n_missing": int(f["energy_in_mask"].isna().sum())})
    if "B0" in frames:
        for m in [x for x in MODELS if x != "B0" and x in frames]:
            r = paired(frames["B0"], frames[m], group_col,
                       lambda fr, idx: _per_seed_mean(fr, idx, "energy_in_mask_b")
                       - _per_seed_mean(fr, idx, "energy_in_mask_a"))
            if r:
                tests.append({"endpoint": "P1", "comparison": f"{m} - B0", **r})

    # --- P5 reliance: BACC drop under metadata shuffle / mean image ------------------------
    frames = {m: load_endpoint(dataset, m, "P5", method) for m in MODELS}
    frames = {m: f for m, f in frames.items() if f is not None}
    for m, f in frames.items():
        for cond in ("pred_meta_shuffled", "pred_mean_image"):
            descriptive.append({"endpoint": "P5", "model": m,
                                "metric": f"drop_{cond.replace('pred_', '')}",
                                "value": _per_seed_drop(f, np.arange(len(f)), cond),
                                "n_rows": len(f)})
    if "B0" in frames:
        for m in [x for x in MODELS if x != "B0" and x in frames]:
            r = paired(frames["B0"], frames[m], group_col,
                       lambda fr, idx: _per_seed_drop(
                           fr.rename(columns={"y_b": "y", "pred_full_b": "pred_full",
                                              "pred_meta_shuffled_b": "pred_meta_shuffled"}),
                           idx, "pred_meta_shuffled")
                       - _per_seed_drop(
                           fr.rename(columns={"y_a": "y", "pred_full_a": "pred_full",
                                              "pred_meta_shuffled_a": "pred_meta_shuffled"}),
                           idx, "pred_meta_shuffled"))
            if r:
                tests.append({"endpoint": "P5", "comparison": f"{m} - B0", **r})

    # --- P4 counterfactual: map shift on irrelevant fields, M1 vs the other fusion models ---
    frames = {m: load_endpoint(dataset, m, "P4", method) for m in MODELS}
    frames = {m: f for m, f in frames.items() if f is not None}
    for m, f in frames.items():
        for kind, g in f.groupby("field_kind"):
            descriptive.append({"endpoint": "P4", "model": m, "metric": f"map_shift_{kind}",
                                "value": _per_seed_mean(g.reset_index(drop=True),
                                                        np.arange(len(g)), "map_shift"),
                                "n_rows": len(g)})
    if "M1" in frames:
        for m in [x for x in ("B3", "B4", "B5") if x in frames]:
            a = frames[m][frames[m]["field_kind"] == "irrelevant"].reset_index(drop=True)
            b = frames["M1"][frames["M1"]["field_kind"] == "irrelevant"].reset_index(drop=True)
            if a.empty or b.empty:
                continue
            r = paired(a, b, group_col,
                       lambda fr, idx: _per_seed_mean(fr, idx, "map_shift_b")
                       - _per_seed_mean(fr, idx, "map_shift_a"))
            if r:
                tests.append({"endpoint": "P4", "comparison": f"M1 - {m}", **r})

    desc = pd.DataFrame(descriptive)
    desc.to_csv(out / f"endpoints_{dataset}_{method}_descriptive.csv", index=False)
    tf = pd.DataFrame(tests)
    if not tf.empty:
        expected = 19
        if family_complete:
            if len(tf) != expected:
                raise SystemExit(f"--family-complete needs all {expected} tests, found {len(tf)}")
            tf["reject_holm"] = holm(tf["p"].to_numpy(), alpha=0.05)
            tf["correction"] = "holm_family_of_19"
        else:
            tf["reject_holm"] = pd.NA
            tf["correction"] = f"UNCORRECTED ({len(tf)}/{expected} of the family available)"
        tf.to_csv(out / f"endpoints_{dataset}_{method}_tests.csv", index=False)
    print(desc.to_string(index=False))
    if not tf.empty:
        print()
        print(tf.to_string(index=False))
    print(f"\nwritten to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--endpoints", nargs="*", default=["P1", "P4", "P5"])
    ap.add_argument("--method", default="gradcam", choices=["gradcam", "rise"])
    ap.add_argument("--target", default="pred", choices=["pred", "true"])
    ap.add_argument("--limit", type=int, help="first N test rows per fold (debug only)")
    ap.add_argument("--batch-size", type=int, default=32, help="attribution batch size")
    ap.add_argument("--aggregate", action="store_true", help="only aggregate what already exists")
    ap.add_argument("--family-complete", action="store_true",
                    help="apply Holm across the full pre-registered family of 19 tests")
    args = ap.parse_args()

    dataset = load_config(args.data)["data"]["name"]
    if not args.aggregate:
        compute(dataset, args.models, args.endpoints, args.method, args.target, args.limit,
                args.batch_size)
    aggregate(dataset, args.method, args.family_complete)


if __name__ == "__main__":
    main()
