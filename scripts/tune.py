"""Hyperparameter tuning, fixed in docs/04-preregistration.md §3 and change log.

Every model gets the same 6-trial grid (lr x head_dropout) on fold 0, seed 0, scored by balanced
accuracy on the inner validation split. The fold-0 test images are never loaded. The winner is
written to configs/tuned/<dataset>_<model>.yaml, which train.py applies automatically.
The trap-set experiments reuse the HAM10000 settings (same imaging modality).

    python scripts/tune.py --data configs/data_padufes20.yaml --model M1
Resumable: finished trials are read back from the tuning run directory given with --resume.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
import yaml

from medfusion.data.metadata import MetadataSchema
from medfusion.data.splits import inner_val_split
from medfusion.train.engine import train_one
from medfusion.utils.config import deep_merge, load_config, parse_overrides
from medfusion.utils.run import create_run_dir, write_json

REPO = Path(__file__).resolve().parents[1]
GRID = {"train.lr": [1e-4, 3e-4, 1e-3], "model.head_dropout": [0.1, 0.3]}
FOLD, SEED = 0, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, choices=["B0", "B3", "B4", "B5", "M1"])
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--set", nargs="*", default=[], help="overrides (debug only)")
    args = ap.parse_args()

    base = deep_merge(load_config(REPO / "configs" / "base.yaml"), load_config(args.data))
    base = deep_merge(base, parse_overrides(args.set))
    d, t = base["data"], base["train"]
    run_dir = args.resume or create_run_dir(base["paths"]["runs_root"],
                                            f"tune_{d['name']}_{args.model}", base)
    print(f"run dir: {run_dir}")

    df = pd.read_csv(d["processed_csv"])
    train_all = df[df["fold"] != FOLD].reset_index(drop=True)  # fold-0 test rows never used
    val_mask = inner_val_split(train_all["y"], train_all[d["group_col"]], t["val_fraction"],
                               seed=SEED)
    train_df, val_df = train_all[~val_mask], train_all[val_mask]
    schema = MetadataSchema(d["categorical"], d["continuous"]).fit(
        df, train_mask=(df["fold"] != FOLD).to_numpy())

    trials = []
    for i, values in enumerate(itertools.product(*GRID.values())):
        params = dict(zip(GRID.keys(), values, strict=True))
        out = run_dir / f"trial{i}.json"
        if out.exists():
            trials.append(json.loads(out.read_text(encoding="utf-8")))
            print(f"trial {i} {params}: done")
            continue
        cfg = deep_merge(base, parse_overrides([f"{k}={v}" for k, v in params.items()]))
        print(f"trial {i} {params}")
        # Validation set passed as "test" so nothing outside train/val is touched.
        res = train_one(cfg, args.model, schema, train_df, val_df, val_df,
                        n_classes=len(d["classes"]), seed=SEED)
        record = {"trial": i, "params": params, "val_balanced_accuracy":
                  res["val_metrics"]["balanced_accuracy"], "val_metrics": res["val_metrics"],
                  "epochs_run": res["epochs_run"], "seconds": res["seconds"]}
        write_json(out, record)
        trials.append(record)

    best = max(trials, key=lambda r: r["val_balanced_accuracy"])
    write_json(run_dir / "tuning.json", {"grid": GRID, "trials": trials, "best": best})
    tuned: dict = {}
    for key, value in best["params"].items():
        section, name = key.split(".")
        tuned.setdefault(section, {})[name] = value
    dest = REPO / "configs" / "tuned" / f"{d['name']}_{args.model}.yaml"
    dest.parent.mkdir(exist_ok=True)
    header = (f"# Selected by scripts/tune.py ({run_dir.name}); val balanced accuracy "
              f"{best['val_balanced_accuracy']:.4f}. Do not edit by hand.\n")
    dest.write_text(header + yaml.safe_dump(tuned, sort_keys=False), encoding="utf-8")
    print(f"best {best['params']} -> {dest}")


if __name__ == "__main__":
    main()
