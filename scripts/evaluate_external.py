"""Zero-shot external evaluation, and the in-domain binary number it must be read against.

The label mapping and the all-MISSING metadata condition are frozen in ``docs/label_mapping.md``.

    # in-domain binary reference (no GPU: reuses the saved out-of-fold predictions)
    python scripts/evaluate_external.py --in-domain --train-data configs/data_padufes20.yaml
    # external test once DDI is downloaded and prepared
    python scripts/evaluate_external.py --train-data configs/data_padufes20.yaml \
        --external configs/data_ddi.yaml

Handbook rule: when the external evaluation is binary, the in-domain binary number is reported
beside it, so a drop cannot be mistaken for the cost of collapsing six classes into two.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from medfusion.data.metadata import MetadataSchema
from medfusion.eval.endpoints import load_checkpoint
from medfusion.eval.stats import mean_ci
from medfusion.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
MODELS = ["B0", "B3", "B4", "B5", "M1"]
# docs/label_mapping.md: which training classes count as malignant.
MALIGNANT = {
    "padufes20": {"BCC", "SCC", "MEL", "ACK"},
    "ham10000": {"mel", "bcc", "akiec"},
}
AK_CLASS = {"padufes20": "ACK", "ham10000": "akiec"}  # the pre-specified sensitivity analysis


def complete_run(dataset: str, model: str) -> Path | None:
    for d in sorted(Path("data/experiments").glob(f"{dataset}_{model}_[0-9]*"), reverse=True):
        m = d / "metrics.json"
        if m.exists() and json.loads(m.read_text()).get("complete_protocol"):
            return d
    return None


def binary_scores(probs: np.ndarray, classes: list[str], malignant: set[str]) -> np.ndarray:
    """Malignant score = total probability mass on the classes mapped to malignant."""
    cols = [i for i, c in enumerate(classes) if c in malignant]
    return probs[:, cols].sum(axis=1)


def in_domain(train_cfg: dict, drop_ak: bool) -> pd.DataFrame:
    """Binary AUROC/BACC from the saved out-of-fold predictions - the external test's reference."""
    dataset, classes = train_cfg["data"]["name"], train_cfg["data"]["classes"]
    malignant = MALIGNANT[dataset]
    rows = []
    for model in MODELS:
        run = complete_run(dataset, model)
        if run is None:
            continue
        oof = pd.read_csv(run / "oof_predictions.csv")
        if drop_ak:
            oof = oof[oof["y"] != classes.index(AK_CLASS[dataset])].reset_index(drop=True)
        probs = oof[[f"p{i}" for i in range(len(classes))]].to_numpy()
        y_bin = np.isin(np.array(classes)[oof["y"].to_numpy()], list(malignant)).astype(int)
        score = binary_scores(probs, classes, malignant)
        per_seed = []
        for _seed, g in oof.groupby("seed"):
            idx = g.index.to_numpy()
            if len(np.unique(y_bin[idx])) < 2:
                continue
            per_seed.append((roc_auc_score(y_bin[idx], score[idx]),
                             balanced_accuracy_score(y_bin[idx], (score[idx] >= 0.5).astype(int))))
        auc, bacc = (np.mean([p[0] for p in per_seed]), np.mean([p[1] for p in per_seed]))
        lo, hi = mean_ci(np.array([p[0] for p in per_seed]))[1:]
        rows.append({"model": model, "setting": "in-domain binary",
                     "ak_excluded": drop_ak, "n": len(oof), "auroc": auc,
                     "auroc_lo": lo, "auroc_hi": hi, "balanced_accuracy": bacc})
    return pd.DataFrame(rows)


def external(train_cfg: dict, ext_cfg: dict, drop_ak: bool, batch_size: int = 32) -> pd.DataFrame:
    """Zero-shot evaluation with every metadata field MISSING (docs/label_mapping.md)."""
    from torch.utils.data import DataLoader

    from medfusion.data.dataset import FusionDataset, build_transform

    dataset, classes = train_cfg["data"]["name"], train_cfg["data"]["classes"]
    malignant = MALIGNANT[dataset]
    df = pd.read_csv(ext_cfg["data"]["processed_csv"])
    strat = ext_cfg["data"].get("stratify_by")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for model_id in MODELS:
        run = complete_run(dataset, model_id)
        if run is None:
            continue
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        for ck in sorted((run / "checkpoints").glob("seed*_fold*.pt")):
            model, schema, _ = load_checkpoint(ck, cfg, device)
            # All-MISSING metadata: the external set carries none of the training fields, so
            # FusionDataset encodes every categorical as MISSING and every continuous as absent.
            blank = MetadataSchema(schema.categorical, schema.continuous, schema.vocabs,
                                   schema.means, schema.stds)
            ds = FusionDataset(df.assign(**{c: np.nan for c in blank.field_names}), blank,
                               build_transform(cfg, train=False))
            probs = []
            with torch.no_grad():
                for b in DataLoader(ds, batch_size=batch_size, shuffle=False):
                    b = {k: v.to(device) for k, v in b.items()}
                    probs.append(model(b["image"], b["cat"], b["cont"],
                                       b["cont_mask"]).softmax(1).cpu().numpy())
            score = binary_scores(np.concatenate(probs), classes, malignant)
            out = df.copy()
            out["score"] = score
            groups = [("all", out)] + ([(str(v), g) for v, g in out.groupby(strat)]
                                       if strat and strat in out else [])
            for name, g in groups:
                if g["y"].nunique() < 2:
                    continue
                rows.append({"model": model_id, "checkpoint": ck.stem, "stratum": name,
                             "n": len(g), "auroc": roc_auc_score(g["y"], g["score"]),
                             "balanced_accuracy": balanced_accuracy_score(
                                 g["y"], (g["score"] >= 0.5).astype(int))})
            del model
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True,
                    help="config of the dataset the models were trained on")
    ap.add_argument("--external", help="external dataset config (e.g. configs/data_ddi.yaml)")
    ap.add_argument("--in-domain", action="store_true", help="only the in-domain binary reference")
    ap.add_argument("--ak-excluded", action="store_true",
                    help="pre-specified sensitivity analysis: drop AK/ACK/akiec rows")
    args = ap.parse_args()

    train_cfg = load_config(args.train_data)
    out_dir = REPO / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_ak_excluded" if args.ak_excluded else ""
    name = train_cfg["data"]["name"]

    ref = in_domain(train_cfg, args.ak_excluded)
    if not ref.empty:
        ref.to_csv(out_dir / f"external_{name}_in_domain_binary{tag}.csv", index=False)
        print(ref.round(4).to_string(index=False))
    if args.in_domain or not args.external:
        return

    ext_cfg = load_config(args.external)
    res = external(train_cfg, ext_cfg, args.ak_excluded)
    if res.empty:
        print("no external results (is the external set prepared?)")
        return
    agg = (res.groupby(["model", "stratum"])
              .agg(n=("n", "first"), auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
                   balanced_accuracy=("balanced_accuracy", "mean")).reset_index())
    res.to_csv(out_dir / f"external_{name}_to_{ext_cfg['data']['name']}_per_checkpoint{tag}.csv",
               index=False)
    agg.to_csv(out_dir / f"external_{name}_to_{ext_cfg['data']['name']}{tag}.csv", index=False)
    print()
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
