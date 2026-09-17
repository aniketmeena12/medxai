"""Training and cross-validation.

``train_one`` fits one model on (train, val) and predicts on test. ``run_cv`` repeats it over the
fixed folds in ``samples.csv`` x seeds and writes per-fold, per-seed raw results (never only the
aggregate; handbook §14). There is deliberately no single-split shortcut.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from medfusion.data.dataset import FusionDataset, build_transform
from medfusion.data.metadata import MetadataSchema, field_dropout
from medfusion.data.splits import assert_disjoint_groups, inner_val_split
from medfusion.models.factory import build_model
from medfusion.models.tabular import fit_b1, metadata_matrix
from medfusion.utils.run import seed_everything, write_json


def classification_metrics(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(1)
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }
    try:
        if probs.shape[1] == 2:
            out["auroc"] = float(roc_auc_score(y, probs[:, 1]))
        else:
            out["auroc"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro",
                                               labels=list(range(probs.shape[1]))))
    except ValueError:
        out["auroc"] = float("nan")
    return out


def _loader(ds, cfg, train: bool) -> DataLoader:
    t = cfg["train"]
    # Worker processes are slow to spawn on Windows (~5 s each), so only the training loader gets
    # them; evaluation images are small cached JPEGs and load fast in the main process.
    workers = t["num_workers"] if train else 0
    return DataLoader(ds, batch_size=t["batch_size"], shuffle=train, drop_last=train,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0)


def _to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, index = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(batch["image"], batch["cat"], batch["cont"], batch["cont_mask"])
        probs.append(logits.float().softmax(1).cpu())
        index.append(batch["index"].cpu())
    return torch.cat(probs).numpy(), torch.cat(index).numpy()


def train_one(
    cfg: dict,
    model_id: str,
    schema: MetadataSchema,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_classes: int,
    seed: int,
    ckpt_path: Path | None = None,
    log=print,
) -> dict:
    """Returns {"test_probs", "val_metrics", "epochs_run", "seconds"}; rows align with test_df."""
    seed_everything(seed)
    t = cfg["train"]

    if model_id == "B1":
        start = time.time()
        clf = fit_b1(schema, pd.concat([train_df, val_df]),
                     np.concatenate([train_df["y"], val_df["y"]]), seed=seed)
        x_test, _ = metadata_matrix(schema, test_df)
        return {"test_probs": clf.predict_proba(x_test), "val_metrics": {}, "epochs_run": 0,
                "seconds": time.time() - start}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(t["amp"]) and device.type == "cuda"
    train_ds = FusionDataset(train_df, schema, build_transform(cfg, train=True))
    val_ds = FusionDataset(val_df, schema, build_transform(cfg, train=False))
    test_ds = FusionDataset(test_df, schema, build_transform(cfg, train=False))
    train_dl, val_dl, test_dl = (_loader(train_ds, cfg, True), _loader(val_ds, cfg, False),
                                 _loader(test_ds, cfg, False))

    model = build_model(model_id, cfg, schema, n_classes).to(device)
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in enc_params}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": t["lr"] * t["backbone_lr_mult"]},
         {"params": other, "lr": t["lr"]}],
        weight_decay=t["weight_decay"],
    )
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = max(1, len(train_dl) // t["accumulate"])
    total_steps, warmup = t["epochs"] * steps_per_epoch, t["warmup_epochs"] * steps_per_epoch

    def set_lr(step: int) -> None:
        if step < warmup:
            scale = (step + 1) / max(1, warmup)
        else:
            progress = (step - warmup) / max(1, total_steps - warmup)
            scale = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        for g, lr in zip(opt.param_groups, base_lrs, strict=True):
            g["lr"] = lr * scale

    weights = None
    if t.get("class_weights") == "balanced":
        present = np.unique(train_df["y"])
        w = np.ones(n_classes, dtype=np.float32)
        w[present] = compute_class_weight("balanced", classes=present, y=train_df["y"])
        weights = torch.tensor(w, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler(enabled=amp)
    p_drop = cfg["model"]["meta_field_dropout"] if model.uses_metadata else 0.0

    best, best_state, bad, step, start = -1.0, None, 0, 0, time.time()
    epochs_run = 0
    for epoch in range(t["epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_dl):
            batch = _to_device(batch, device)
            cat, cont_mask = field_dropout(batch["cat"], batch["cont_mask"],
                                           schema.cardinalities, p_drop)
            with torch.autocast(device.type, enabled=amp):
                logits = model(batch["image"], cat, batch["cont"], cont_mask)
                loss = loss_fn(logits.float(), batch["y"]) / t["accumulate"]
            scaler.scale(loss).backward()
            if (i + 1) % t["accumulate"] == 0:
                set_lr(step)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
        val_probs, _ = predict(model, val_dl, device, amp)
        val_bacc = balanced_accuracy_score(val_df["y"], val_probs.argmax(1))
        epochs_run = epoch + 1
        log(f"  epoch {epoch + 1:3d} loss {loss.item() * t['accumulate']:.4f} "
            f"val_bacc {val_bacc:.4f} ({time.time() - start:.0f}s)")
        if val_bacc > best:
            best, bad = val_bacc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= t["patience"]:
                break

    model.load_state_dict(best_state)
    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_id": model_id, "state_dict": best_state,
                    "schema": schema.to_dict(), "n_classes": n_classes}, ckpt_path)
    test_probs, _ = predict(model, test_dl, device, amp)
    val_probs, _ = predict(model, val_dl, device, amp)
    return {"test_probs": test_probs,
            "val_metrics": classification_metrics(val_df["y"].to_numpy(), val_probs),
            "epochs_run": epochs_run, "seconds": time.time() - start}


def run_cv(cfg: dict, model_id: str, run_dir: Path, folds: list[int] | None = None,
           seeds: list[int] | None = None) -> dict:
    """Cross-validation over the fixed ``fold`` column x seeds. ``folds``/``seeds`` subsets are for
    timing and debugging only and are recorded in metrics.json."""
    d, t = cfg["data"], cfg["train"]
    df = pd.read_csv(d["processed_csv"])
    n_classes = len(d["classes"])
    folds = folds if folds is not None else list(range(t["n_folds"]))
    seeds = seeds if seeds is not None else list(t["seeds"])
    log_file = open(run_dir / "log.txt", "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    fold_dir = run_dir / "folds"
    fold_dir.mkdir(exist_ok=True)
    for seed in seeds:
        for k in folds:
            tag = f"seed{seed}_fold{k}"
            if (fold_dir / f"{tag}.json").exists():  # finished in an earlier session
                log(f"[{model_id}] {tag}: already done, skipping")
                continue
            train_all = df[df["fold"] != k].reset_index(drop=True)
            test_df = df[df["fold"] == k].reset_index(drop=True)
            val_mask = inner_val_split(train_all["y"], train_all[d["group_col"]],
                                       t["val_fraction"], seed=seed)
            train_df, val_df = train_all[~val_mask], train_all[val_mask]
            assert_disjoint_groups(train_df[d["group_col"]], test_df[d["group_col"]])
            assert_disjoint_groups(train_df[d["group_col"]], val_df[d["group_col"]])
            schema = MetadataSchema(d["categorical"], d["continuous"]).fit(
                df, train_mask=(df["fold"] != k).to_numpy())
            log(f"[{model_id}] seed {seed} fold {k}: train {len(train_df)} val {len(val_df)} "
                f"test {len(test_df)}")
            res = train_one(cfg, model_id, schema, train_df, val_df, test_df, n_classes, seed,
                            ckpt_path=run_dir / "checkpoints" / f"seed{seed}_fold{k}.pt", log=log)
            m = classification_metrics(test_df["y"].to_numpy(), res["test_probs"])
            log(f"[{model_id}] seed {seed} fold {k}: {json.dumps(m)} ({res['seconds']:.0f}s)")
            out = test_df[["sample_id", "y", d["group_col"]]].copy()
            out["seed"], out["fold"] = seed, k
            for c in range(n_classes):
                out[f"p{c}"] = res["test_probs"][:, c]
            out.to_csv(fold_dir / f"{tag}_predictions.csv", index=False)
            # The JSON is written last: its presence marks the fold as complete.
            write_json(fold_dir / f"{tag}.json",
                       {"seed": seed, "fold": k, **m, "epochs_run": res["epochs_run"],
                        "seconds": res["seconds"], "val_metrics": res["val_metrics"]})

    records, oof = [], []
    for seed in seeds:
        for k in folds:
            tag = f"seed{seed}_fold{k}"
            with open(fold_dir / f"{tag}.json", encoding="utf-8") as f:
                records.append(json.load(f))
            oof.append(pd.read_csv(fold_dir / f"{tag}_predictions.csv"))
    pd.concat(oof).to_csv(run_dir / "oof_predictions.csv", index=False)
    summary = {"model_id": model_id, "dataset": d["name"], "folds": folds, "seeds": seeds,
               "complete_protocol": sorted(folds) == list(range(t["n_folds"]))
               and sorted(seeds) == sorted(t["seeds"]),
               "per_fold_seed": records}
    write_json(run_dir / "metrics.json", summary)
    log_file.close()
    return summary
