"""Seeding and run-directory bookkeeping.

Handbook §14: config snapshot, git SHA, never overwrite a run.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def create_run_dir(runs_root: str | Path, name: str, config: dict) -> Path:
    """Create ``<runs_root>/<name>_<NNN>`` with the next free number and snapshot the config."""
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    taken = [
        int(p.name.rsplit("_", 1)[1])
        for p in root.glob(f"{name}_*")
        if p.is_dir() and p.name.rsplit("_", 1)[1].isdigit()
    ]
    run_dir = root / f"{name}_{max(taken, default=0) + 1:03d}"
    run_dir.mkdir()
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    sha = _git("rev-parse", "HEAD").strip() or "NO_COMMIT"
    diff = _git("diff", "HEAD")
    (run_dir / "git_sha.txt").write_text(sha + ("\n\n" + diff if diff else "\n"), encoding="utf-8")
    return run_dir


def write_json(path: str | Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
