"""Compute the pre-registered endpoints P1, P4 and P5 from finished runs.

``docs/04-preregistration.md`` §5. Each function takes one trained model and one evaluation frame
and returns **per-image rows** (P1, P4) or one row per run (P5), carrying the group column so §7's
patient-cluster bootstrap can resample clusters. P2/P3 come from the trap-set runs instead
(``scripts/train_trap.py``) and are aggregated separately.

Attribution is Grad-CAM by default (§4 primary), RISE as the secondary channel. Maps are computed
for the **predicted** class; pass ``target="true"`` for the sensitivity analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from medfusion.data.dataset import FusionDataset, build_transform
from medfusion.data.metadata import MetadataSchema
from medfusion.eval.counterfactual import map_shift, perturb_field
from medfusion.eval.localization import (
    binarize,
    energy_in_mask,
    iou,
    mask_area_fraction,
    pointing_game,
)
from medfusion.models.factory import build_model
from medfusion.xai.attribution import gradcam, passes_sanity_gate, randomized_copy, rise


def load_checkpoint(path, cfg: dict, device) -> tuple[torch.nn.Module, MetadataSchema, int]:
    """Rebuild a model from a checkpoint written by ``train_one`` (model id, state, schema)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    schema = MetadataSchema.from_dict(ck["schema"])
    model = build_model(ck["model_id"], cfg, schema, ck["n_classes"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), schema, ck["n_classes"]


def _loader(df: pd.DataFrame, schema: MetadataSchema, cfg: dict, with_mask: bool,
            batch_size: int) -> DataLoader:
    ds = FusionDataset(df, schema, build_transform(cfg, train=False), with_mask=with_mask,
                       mask_size=cfg["image"]["input_size"])
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def _to_device(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def attribute(model, batch: dict, method: str = "gradcam", target=None, **kw) -> torch.Tensor:
    if method == "gradcam":
        return gradcam(model, batch, target)
    if method == "rise":
        return rise(model, batch, target, **kw)
    raise ValueError(f"unknown attribution method {method!r}")


def _target_class(model, batch: dict, target: str) -> torch.Tensor:
    if target == "true":
        return batch["y"]
    with torch.no_grad():
        logits = model(batch["image"], batch.get("cat"), batch.get("cont"),
                       batch.get("cont_mask"))
    return logits.argmax(1)


def grounding(model, df: pd.DataFrame, schema: MetadataSchema, cfg: dict, device,
              group_col: str, method: str = "gradcam", target: str = "pred",
              batch_size: int = 16) -> pd.DataFrame:
    """**P1**: share of positive attribution inside the lesion mask, one row per masked image.

    Also returns the chance level (mask area fraction, §5 says report it alongside) and the
    secondary localization metrics: pointing game and Otsu-thresholded IoU.
    """
    df = df[df["mask_path"].notna()].reset_index(drop=True)
    rows = []
    for batch in _loader(df, schema, cfg, with_mask=True, batch_size=batch_size):
        idx = batch["index"].numpy()
        batch = _to_device(batch, device)
        cls = _target_class(model, batch, target)
        maps = attribute(model, batch, method, cls).cpu().numpy()
        masks = batch["lesion_mask"].cpu().numpy()
        has = batch["has_mask"].cpu().numpy()
        for j, i in enumerate(idx):
            if not has[j]:
                continue
            sal, gt = maps[j], masks[j]
            rows.append({
                "sample_id": df.at[int(i), "sample_id"],
                group_col: df.at[int(i), group_col],
                "energy_in_mask": energy_in_mask(sal, gt),
                "mask_area_fraction": mask_area_fraction(gt),
                "pointing_game": float(pointing_game(sal, gt)),
                "iou_otsu": iou(binarize(sal, "otsu"), gt),
            })
    return pd.DataFrame(rows)


def counterfactual(model, df: pd.DataFrame, schema: MetadataSchema, cfg: dict, device,
                   group_col: str, fields: dict[str, list[str]], method: str = "gradcam",
                   target: str = "pred", batch_size: int = 16, seed: int = 0) -> pd.DataFrame:
    """**P4**: map shift (1 - Spearman rho) when one metadata field is changed, image fixed.

    Both maps use the class predicted from the **original** metadata, so the shift measures how the
    evidence for one fixed explanandum moves rather than a change of explained class; whether the
    prediction itself flipped is recorded in ``pred_changed``.
    """
    rows = []
    for batch in _loader(df, schema, cfg, with_mask=False, batch_size=batch_size):
        idx = batch["index"].numpy()
        batch = _to_device(batch, device)
        cls = _target_class(model, batch, target)
        before = attribute(model, batch, method, cls).cpu().numpy()
        for kind, names in fields.items():
            for field in names:
                gen = torch.Generator(device="cpu").manual_seed(seed)
                cat, cont, cont_mask = perturb_field(
                    schema, field, batch["cat"].cpu(), batch["cont"].cpu(),
                    batch["cont_mask"].cpu(), generator=gen)
                changed = dict(batch)
                changed["cat"] = cat.to(device)
                changed["cont"] = cont.to(device)
                changed["cont_mask"] = cont_mask.to(device)
                after = attribute(model, changed, method, cls).cpu().numpy()
                with torch.no_grad():
                    new_cls = model(changed["image"], changed["cat"], changed["cont"],
                                    changed["cont_mask"]).argmax(1)
                for j, i in enumerate(idx):
                    rows.append({
                        "sample_id": df.at[int(i), "sample_id"],
                        group_col: df.at[int(i), group_col],
                        "field": field,
                        "field_kind": kind,
                        "map_shift": map_shift(before[j], after[j]),
                        "pred_changed": float(new_cls[j].item() != cls[j].item()),
                    })
    return pd.DataFrame(rows)


def reliance(model, df: pd.DataFrame, schema: MetadataSchema, cfg: dict, device,
             batch_size: int = 32, seed: int = 0) -> pd.DataFrame:
    """**P5**: modality reliance - balanced-accuracy drop when the metadata is shuffled across
    patients, versus when every image is replaced by the dataset mean image.

    Returns per-image predictions under all three conditions so the drops can be bootstrapped over
    patient clusters like every other endpoint.
    """

    loader = _loader(df, schema, cfg, with_mask=False, batch_size=batch_size)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))

    mean_image, n_seen = None, 0
    for batch in loader:  # dataset mean image, in normalized input space
        s = batch["image"].sum(0)
        mean_image = s if mean_image is None else mean_image + s
        n_seen += len(batch["image"])
    mean_image = (mean_image / n_seen).to(device)

    cat_all = torch.from_numpy(np.asarray(loader.dataset.cat)[perm])
    cont_all = torch.from_numpy(np.asarray(loader.dataset.cont)[perm])
    mask_all = torch.from_numpy(np.asarray(loader.dataset.cont_mask)[perm])

    rows = []
    with torch.no_grad():
        for batch in loader:
            idx = batch["index"].numpy()
            batch = _to_device(batch, device)
            full = model(batch["image"], batch["cat"], batch["cont"],
                         batch["cont_mask"]).argmax(1)
            shuffled = model(batch["image"], cat_all[idx].to(device), cont_all[idx].to(device),
                             mask_all[idx].to(device)).argmax(1)
            blank = mean_image.unsqueeze(0).expand(len(idx), *mean_image.shape)
            no_image = model(blank, batch["cat"], batch["cont"], batch["cont_mask"]).argmax(1)
            for j, i in enumerate(idx):
                rows.append({"index": int(i), "y": int(batch["y"][j].item()),
                             "pred_full": int(full[j].item()),
                             "pred_meta_shuffled": int(shuffled[j].item()),
                             "pred_mean_image": int(no_image[j].item())})
    out = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    out["sample_id"] = df["sample_id"].to_numpy()[out["index"]]
    return out


def reliance_drops(preds: pd.DataFrame, rows: np.ndarray | None = None) -> dict[str, float]:
    """Balanced-accuracy drops from the per-image frame returned by :func:`reliance`."""
    from sklearn.metrics import balanced_accuracy_score

    p = preds if rows is None else preds.iloc[rows]
    full = balanced_accuracy_score(p["y"], p["pred_full"])
    return {
        "bacc_full": float(full),
        "bacc_meta_shuffled": float(balanced_accuracy_score(p["y"], p["pred_meta_shuffled"])),
        "bacc_mean_image": float(balanced_accuracy_score(p["y"], p["pred_mean_image"])),
        "drop_metadata": float(full - balanced_accuracy_score(p["y"], p["pred_meta_shuffled"])),
        "drop_image": float(full - balanced_accuracy_score(p["y"], p["pred_mean_image"])),
    }


def sanity_gate(model, df: pd.DataFrame, schema: MetadataSchema, cfg: dict, device,
                method: str = "gradcam", batch_size: int = 16, seed: int = 0,
                threshold: float = 0.5) -> dict:
    """Pre-registered sanity gate (§4; Adebayo et al., 2018).

    Compare each image's map from the trained model with the map from a fully parameter-randomized
    copy. A method whose maps barely change (mean SSIM >= ``threshold``) is not reading the learned
    weights and is excluded, with the exclusion reported.
    """
    random_model = randomized_copy(model, seed=seed).to(device).eval()
    trained_maps, random_maps = [], []
    for batch in _loader(df, schema, cfg, with_mask=False, batch_size=batch_size):
        batch = _to_device(batch, device)
        cls = _target_class(model, batch, "pred")
        trained_maps.append(attribute(model, batch, method, cls).cpu())
        random_maps.append(attribute(random_model, batch, method, cls).cpu())
    trained = torch.cat(trained_maps)
    randomized = torch.cat(random_maps)
    passed, score = passes_sanity_gate(trained, randomized, threshold=threshold)
    return {"method": method, "n_images": int(len(trained)), "mean_ssim": score,
            "threshold": threshold, "passed": bool(passed)}
