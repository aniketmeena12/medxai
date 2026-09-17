"""Trap-set experiments on ISIC 2019 (P2 image-artifact curve, P3 metadata trap / 2-D grid).

One call = one cell: (image bias level, published split, metadata rho or None, seed).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from medfusion.data.metadata import MetadataSchema
from medfusion.data.trap import (
    agreement_fraction,
    load_image_trap,
    resample_metadata_trap,
    synthetic_attribute,
)
from medfusion.train.engine import classification_metrics, train_one
from medfusion.utils.run import write_json

META_TOKEN = "meta_trap_token"


def _sex_attr(df: pd.DataFrame) -> np.ndarray:
    sex = df["sex"].astype(str).str.lower()
    return np.where(sex == "male", 1.0, np.where(sex == "female", 0.0, np.nan))


ARTIFACTS = ["dark_corner", "hair", "gel_border", "gel_bubble", "ruler", "ink", "patches"]
ARTIFACT_THRESHOLD = 0.6  # as in Bissoto et al.'s code


def cell_tag(image_level: float, split: int, meta_rho: float | None, meta_mode: str, seed: int,
             leak_filter: bool) -> str:
    meta = f"meta-{meta_mode}{meta_rho:g}" if meta_rho is not None else "nometa"
    return (f"img{image_level:g}_split{split}_{meta}_seed{seed}"
            + ("" if leak_filter else "_published"))


def artifact_rates(frame: pd.DataFrame) -> dict[str, float]:
    return {a: float((frame[a] > ARTIFACT_THRESHOLD).mean()) for a in ARTIFACTS if a in frame}


def build_trap_cell(cfg: dict, image_level: float, split: int, meta_rho: float | None,
                    meta_mode: str, seed: int) -> tuple[dict[str, pd.DataFrame], dict, dict]:
    """Return ({train,val,test} frames with column y), the effective config, and bookkeeping.

    With ``data.lesion_leak_filter`` (default true, the primary analysis), validation and test
    images whose lesion also appears in training are removed: the published splits are per image
    and share ~2,000 lesions between train and test (checked 2026-09-17). Filtering happens before
    any metadata resampling.
    """
    d = cfg["data"]
    leak_filter = bool(d.get("lesion_leak_filter", True))
    samples = pd.read_csv(d["processed_csv"]).drop(columns=["y"])
    parts = load_image_trap(d["trap_dir"], image_level, split)
    frames, info = {}, {"lesion_leak_filter": leak_filter}
    for name, part in parts.items():
        merged = part.merge(samples, on="image", how="inner")
        info[f"{name}_unmatched"] = int(len(part) - len(merged))
        frames[name] = merged.rename(columns={"label": "y"})

    group = d["group_col"]
    info["test_artifact_rates_published"] = artifact_rates(frames["test"])
    train_lesions = set(frames["train"][group])
    for name in ("val", "test"):
        leaked = frames[name][group].isin(train_lesions)
        info[f"{name}_leaked_images_published"] = int(leaked.sum())
        if leak_filter:
            frames[name] = frames[name][~leaked].reset_index(drop=True)
    info["test_artifact_rates"] = artifact_rates(frames["test"])

    cfg = copy.deepcopy(cfg)
    rng = np.random.default_rng(seed)
    if meta_rho is not None:
        if meta_mode == "sex":
            for name in frames:
                frames[name] = resample_metadata_trap(
                    frames[name], _sex_attr(frames[name]), "y", meta_rho,
                    reverse=(name == "test"), rng=rng)
            attr_of = _sex_attr
        elif meta_mode == "synthetic":
            for name in frames:
                frames[name][META_TOKEN] = synthetic_attribute(
                    frames[name]["y"].to_numpy(), meta_rho, reverse=(name == "test"), rng=rng
                ).astype(str)
            cfg["data"]["categorical"] = [*d["categorical"], META_TOKEN]
            attr_of = lambda f: f[META_TOKEN].astype(int).to_numpy()  # noqa: E731
        else:
            raise ValueError(f"meta_mode must be 'sex' or 'synthetic', got {meta_mode!r}")
        for name, f in frames.items():
            a = attr_of(f)
            ok = ~np.isnan(a.astype(float))
            info[f"{name}_meta_agreement"] = agreement_fraction(a[ok], f["y"].to_numpy()[ok])

    for name, f in frames.items():
        info[f"{name}_n"] = len(f)
        info[f"{name}_positive_rate"] = float(f["y"].mean())
    info["train_test_lesion_overlap"] = len(set(frames["train"][group])
                                            & set(frames["test"][group]))
    return frames, cfg, info


def run_trap_cell(cfg: dict, model_id: str, image_level: float, split: int,
                  meta_rho: float | None, meta_mode: str, seed: int, run_dir: Path) -> dict:
    frames, cfg, info = build_trap_cell(cfg, image_level, split, meta_rho, meta_mode, seed)
    tag = cell_tag(image_level, split, meta_rho, meta_mode, seed, info["lesion_leak_filter"])
    min_train = cfg["data"].get("metadata_trap", {}).get("min_train_images", 0)
    if meta_rho is not None and info["train_n"] < min_train:
        result = {"tag": tag, "skipped": f"train_n {info['train_n']} < {min_train}", **info}
        write_json(run_dir / f"{tag}.json", result)
        return result

    d = cfg["data"]
    schema = MetadataSchema(d["categorical"], d["continuous"]).fit(
        pd.concat(frames.values()), train_mask=np.r_[np.ones(len(frames["train"]), bool),
                                                     np.zeros(len(frames["val"])
                                                              + len(frames["test"]), bool)])
    res = train_one(cfg, model_id, schema, frames["train"], frames["val"], frames["test"],
                    n_classes=2, seed=seed,
                    ckpt_path=run_dir / "checkpoints" / f"{tag}.pt")
    test = frames["test"]
    metrics = classification_metrics(test["y"].to_numpy(), res["test_probs"])
    keep = ["image", "y", d["group_col"], *[a for a in ARTIFACTS if a in test], "mask_path"]
    preds = test[[c for c in keep if c in test]].copy()
    preds["p1"] = res["test_probs"][:, 1]
    preds.to_csv(run_dir / f"{tag}_test_predictions.csv", index=False)
    result = {"tag": tag, "model_id": model_id, "image_level": image_level, "split": split,
              "meta_rho": meta_rho, "meta_mode": meta_mode, "seed": seed, **metrics, **info,
              "epochs_run": res["epochs_run"], "seconds": res["seconds"]}
    write_json(run_dir / f"{tag}.json", result)
    return result
