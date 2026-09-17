"""Train one model under the full protocol (all folds x all seeds).

Examples:
    python scripts/train.py --data configs/data_padufes20.yaml --model M1
    python scripts/train.py --data configs/data_ham10000.yaml --model B0 --set train.batch_size=16
    # timing / debug only (recorded as incomplete in metrics.json):
    python scripts/train.py --data configs/data_padufes20.yaml --model B0 --folds 0 --seeds 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from medfusion.train.engine import run_cv
from medfusion.utils.config import deep_merge, load_config, parse_overrides
from medfusion.utils.run import create_run_dir

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset config, e.g. configs/data_ham10000.yaml")
    ap.add_argument("--model", required=True, choices=["B0", "B1", "B3", "B4", "B5", "M1"])
    ap.add_argument("--folds", type=int, nargs="*", help="subset of folds (debug/timing only)")
    ap.add_argument("--seeds", type=int, nargs="*", help="subset of seeds (debug/timing only)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.epochs=2")
    ap.add_argument("--resume", type=Path,
                    help="existing run directory; finished folds are skipped")
    args = ap.parse_args()

    cfg = deep_merge(load_config(REPO / "configs" / "base.yaml"), load_config(args.data))
    debug = args.folds is not None or args.seeds is not None or bool(args.set)
    tuned = REPO / "configs" / "tuned" / f"{cfg['data']['name']}_{args.model}.yaml"
    if tuned.exists():
        cfg = deep_merge(cfg, load_config(tuned))
        print(f"applied tuned hyperparameters: {tuned.name}")
    elif args.model != "B1" and not debug:
        raise SystemExit(f"{tuned} missing: run scripts/tune.py first (pre-registration §3)")
    cfg = deep_merge(cfg, parse_overrides(args.set))
    cfg["run"] = {"model": args.model, "folds": args.folds, "seeds": args.seeds,
                  "tuned": tuned.name if tuned.exists() else None}
    name = f"{cfg['data']['name']}_{args.model}"
    if debug:
        name += "_debug"
    run_dir = args.resume or create_run_dir(cfg["paths"]["runs_root"], name, cfg)
    print(f"run dir: {run_dir}")
    run_cv(cfg, args.model, run_dir, folds=args.folds, seeds=args.seeds)


if __name__ == "__main__":
    main()
