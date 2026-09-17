"""Run ISIC 2019 trap-set cells. Every finished cell is written immediately, and cells whose
result JSON already exists in --resume are skipped, so the grid can run over many sessions.

Examples:
    # P2 image-artifact curve for one model (all levels x splits 1-3 x seeds from config)
    python scripts/train_trap.py --model M1 --curve
    # P3 2-D grid (image {0,.5,.9} x metadata sex {0,.5,.9})
    python scripts/train_trap.py --model B3 --grid --meta-mode sex
    # continue an interrupted run
    python scripts/train_trap.py --model M1 --curve \\
        --resume data/experiments/isic2019_trap_M1_curve_001
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from medfusion.train.trap import cell_tag, run_trap_cell
from medfusion.utils.config import deep_merge, load_config, parse_overrides
from medfusion.utils.run import create_run_dir

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["B0", "B3", "B4", "B5", "M1"])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--curve", action="store_true", help="image bias levels, no metadata trap")
    mode.add_argument("--grid", action="store_true", help="image x metadata bias grid")
    ap.add_argument("--meta-mode", default="sex", choices=["sex", "synthetic"])
    ap.add_argument("--published", action="store_true",
                    help="secondary analysis: published splits without the lesion-leak filter")
    ap.add_argument("--resume", type=Path, help="existing run directory to continue")
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.epochs=2")
    args = ap.parse_args()

    cfg = deep_merge(load_config(REPO / "configs" / "base.yaml"),
                     load_config(REPO / "configs" / "data_isic2019_trap.yaml"))
    # Trap sets reuse the HAM10000-tuned hyperparameters (same modality; pre-registration log).
    tuned = REPO / "configs" / "tuned" / f"ham10000_{args.model}.yaml"
    if tuned.exists():
        cfg = deep_merge(cfg, load_config(tuned))
    elif not args.set:
        raise SystemExit(f"{tuned} missing: tune on HAM10000 first")
    cfg = deep_merge(cfg, parse_overrides(args.set))
    cfg["data"]["lesion_leak_filter"] = not args.published
    d = cfg["data"]
    if args.curve:
        cells = [(lv, None) for lv in d["image_bias_levels"]]
        name = f"isic2019_trap_{args.model}_curve"
    else:
        mt = d["metadata_trap"]
        cells = list(itertools.product(mt["grid_image_levels"], mt["levels"]))
        name = f"isic2019_trap_{args.model}_grid_{args.meta_mode}"

    if args.published:
        name += "_published"
    run_dir = args.resume or create_run_dir(cfg["paths"]["runs_root"], name, cfg)
    print(f"run dir: {run_dir}")
    for (level, rho), split, seed in itertools.product(cells, d["splits"], d["seeds"]):
        tag = cell_tag(level, split, rho, args.meta_mode, seed, not args.published)
        if (run_dir / f"{tag}.json").exists():
            print(f"skip {tag} (done)")
            continue
        print(f"== {tag}")
        result = run_trap_cell(cfg, args.model, level, split, rho, args.meta_mode, seed, run_dir)
        print({k: result.get(k) for k in ("tag", "auroc", "balanced_accuracy", "skipped")})


if __name__ == "__main__":
    main()
