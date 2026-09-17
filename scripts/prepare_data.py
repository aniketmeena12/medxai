"""Unpack raw downloads, cache square-resized images/masks, and write processed samples.csv files.

    python scripts/prepare_data.py --dataset padufes20
    python scripts/prepare_data.py --dataset ham10000
    python scripts/prepare_data.py --dataset isic2019      # needs ham10000 prepared first for masks

Outputs under <data_root>/processed/<dataset>/:
    samples.csv   sample_id, image_path, y, <group>, metadata columns, fold[, mask_path]
    SUMMARY.txt   class counts, groups, missingness
Folds: StratifiedGroupKFold(5, shuffle, seed 0), fixed for all seeds and models.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# medfusion (and therefore torch) is imported inside functions: on Windows every worker process
# re-imports this module, and 15 workers each loading torch exhausted 16 GB of RAM.

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("MEDFUSION_DATA", REPO / "data"))
SIZE = 256
IMG_EXT = {".png", ".jpg", ".jpeg"}


def extract_all(src_dir: Path, dest: Path) -> None:
    """Extract every zip under src_dir (and zips found inside them) into dest, once."""
    marker = dest / ".extracted"
    if marker.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for zp in sorted(src_dir.glob("*.zip")):
        print(f"extracting {zp.name}")
        with zipfile.ZipFile(zp) as z:
            z.extractall(dest)
    while True:  # nested zips (e.g. PAD-UFES-20 image parts inside the Mendeley bundle)
        inner = [p for p in dest.rglob("*.zip") if not p.with_name(p.name + ".done").exists()]
        if not inner:
            break
        for zp in inner:
            print(f"extracting {zp.relative_to(dest)}")
            with zipfile.ZipFile(zp) as z:
                z.extractall(zp.parent)
            zp.with_name(zp.name + ".done").touch()
    marker.touch()


def index_files(root: Path, exts=IMG_EXT) -> dict[str, Path]:
    return {p.stem: p for p in root.rglob("*") if p.suffix.lower() in exts}


def _resize_one(job: tuple[str, str, bool]) -> str | None:
    src, dst, is_mask = job
    if Path(dst).exists():
        return None
    try:
        im = Image.open(src)
        if not is_mask:
            im.draft("RGB", (2 * SIZE, 2 * SIZE))  # JPEG: decode at reduced scale
        im = im.convert("L") if is_mask else im.convert("RGB")
        im = im.resize((SIZE, SIZE), Image.NEAREST if is_mask else Image.BICUBIC)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        if is_mask:
            im.save(dst)
        else:
            im.save(dst, quality=95)
    except Exception as e:  # noqa: BLE001 - report and continue; missing files are checked later
        return f"{src}: {e}"
    return None


def cache(jobs: list[tuple[str, str, bool]], desc: str) -> None:
    workers = min(6, max(1, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        errors = [e for e in tqdm(ex.map(_resize_one, jobs, chunksize=32), total=len(jobs),
                                  desc=desc) if e]
    if errors:
        raise RuntimeError(f"{len(errors)} files failed, first: {errors[0]}")


def add_folds(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    from medfusion.data.splits import grouped_folds

    df["fold"] = grouped_folds(df["y"], df[group_col], n_folds=5, seed=0)
    return df


def write(df: pd.DataFrame, name: str, group_col: str, meta_cols: list[str]) -> None:
    from medfusion.data.metadata import is_missing

    out = DATA_ROOT / "processed" / name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "samples.csv", index=False)
    lines = [f"rows {len(df)}", f"groups ({group_col}) {df[group_col].nunique()}",
             "class counts:", df["y_name"].value_counts().to_string(), "missing fraction:"]
    lines += [f"  {c}: {is_missing(df[c]).mean():.3f}" for c in meta_cols]
    if "fold" in df:
        lines += ["fold sizes:", df["fold"].value_counts().sort_index().to_string()]
    if "mask_path" in df:
        lines.append(f"with mask: {df['mask_path'].notna().sum()}")
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def prepare_padufes20() -> None:
    raw = DATA_ROOT / "raw" / "pad-ufes-20"
    ext = DATA_ROOT / "interim" / "pad-ufes-20" / "extracted"
    extract_all(raw, ext)
    # Official Mendeley metadata (SHA256-verified). The Kaggle copy holds the same values but was
    # re-saved with different formatting (TRUE vs True, 1.0 vs 1), so it is not used.
    meta = pd.read_csv(DATA_ROOT / "raw" / "pad-ufes-20-official" / "metadata.csv")
    files = index_files(ext)
    classes = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
    meta["diagnostic"] = meta["diagnostic"].str.upper().str.strip()
    meta = meta[meta["diagnostic"].isin(classes)].reset_index(drop=True)
    meta["stem"] = meta["img_id"].str.rsplit(".", n=1).str[0]
    missing = sorted(set(meta["stem"]) - set(files))
    if missing:
        raise FileNotFoundError(f"{len(missing)} images not found, e.g. {missing[:3]}")
    cache_dir = DATA_ROOT / "interim" / "pad-ufes-20" / f"images_{SIZE}"
    meta["image_path"] = [str(cache_dir / f"{s}.jpg") for s in meta["stem"]]
    pairs = zip(meta["stem"], meta["image_path"], strict=True)
    cache([(str(files[s]), p, False) for s, p in pairs], "PAD")
    meta["sample_id"] = meta["stem"]
    meta["y_name"] = meta["diagnostic"]
    meta["y"] = meta["diagnostic"].map({c: i for i, c in enumerate(classes)})
    add_folds(meta, "patient_id")
    cols = [c for c in meta.columns if c not in ("stem",)]
    write(meta[cols], "padufes20", "patient_id",
          ["region", "fitspatrick", "gender", "smoke", "age", "diameter_1", "bleed", "itch"])


def prepare_ham10000() -> None:
    ext = DATA_ROOT / "interim" / "ham10000" / "extracted"
    extract_all(DATA_ROOT / "raw" / "ham10000", ext / "images")
    extract_all(DATA_ROOT / "raw" / "ham10000-masks", ext / "masks")
    meta = pd.read_csv(next(ext.rglob("HAM10000_metadata.csv")))
    images = index_files(ext, {".jpg"})
    masks = {k.replace("_segmentation", ""): v
             for k, v in index_files(ext, {".png"}).items() if k.endswith("_segmentation")}
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    missing = sorted(set(meta["image_id"]) - set(images))
    if missing:
        raise FileNotFoundError(f"{len(missing)} images not found, e.g. {missing[:3]}")
    base = DATA_ROOT / "interim" / "ham10000"
    meta["image_path"] = [str(base / f"images_{SIZE}" / f"{i}.jpg") for i in meta["image_id"]]
    meta["mask_path"] = [str(base / f"masks_{SIZE}" / f"{i}.png") if i in masks else None
                         for i in meta["image_id"]]
    ids = meta["image_id"]
    jobs = [(str(images[i]), p, False) for i, p in zip(ids, meta["image_path"], strict=True)]
    jobs += [(str(masks[i]), p, True) for i, p in zip(ids, meta["mask_path"], strict=True) if p]
    cache(jobs, "HAM10000")
    meta["sample_id"] = meta["image_id"]
    meta["y_name"] = meta["dx"]
    meta["y"] = meta["dx"].map({c: i for i, c in enumerate(classes)})
    # "unknown" is how HAM10000 spells missing for sex/localization.
    for col in ("sex", "localization"):
        meta.loc[meta[col].astype(str).str.lower() == "unknown", col] = np.nan
    add_folds(meta, "lesion_id")
    write(meta, "ham10000", "lesion_id", ["age", "sex", "localization"])


def prepare_isic2019() -> None:
    raw = DATA_ROOT / "raw" / "isic2019"
    ext = DATA_ROOT / "interim" / "isic2019" / "extracted"
    extract_all(raw, ext)
    meta = pd.read_csv(raw / "ISIC_2019_Training_Metadata.csv")
    gt = pd.read_csv(raw / "ISIC_2019_Training_GroundTruth.csv")
    df = meta.merge(gt, on="image", how="inner")
    dx_cols = [c for c in gt.columns if c != "image"]
    df["y_name"] = df[dx_cols].idxmax(axis=1)
    malignant = {"MEL", "BCC", "SCC", "AK"}
    df["y"] = df["y_name"].isin(malignant).astype(int)  # reference only; trap CSVs carry labels
    images = index_files(ext, {".jpg"})
    # The official CSVs name 2,074 images "ISIC_xxx_downsampled"; the trap sets use "ISIC_xxx".
    # Keep the file stem for lookup and use the plain id everywhere else.
    df["stem"] = df["image"]
    df["image"] = df["image"].str.replace("_downsampled", "", regex=False)
    missing = sorted(set(df["stem"]) - set(images))
    if missing:
        raise FileNotFoundError(f"{len(missing)} images not found, e.g. {missing[:3]}")
    base = DATA_ROOT / "interim" / "isic2019"
    df["image_path"] = [str(base / f"images_{SIZE}" / f"{i}.jpg") for i in df["image"]]
    pairs = zip(df["stem"], df["image_path"], strict=True)
    cache([(str(images[s]), p, False) for s, p in pairs], "ISIC2019")
    # Lesion masks come from HAM10000 for the images the two datasets share.
    ham = DATA_ROOT / "processed" / "ham10000" / "samples.csv"
    if ham.exists():
        hm = pd.read_csv(ham, usecols=["image_id", "mask_path"]).rename(columns={"image_id":
                                                                                 "image"})
        df = df.merge(hm, on="image", how="left")
    else:
        print("WARNING: ham10000 not prepared; no masks attached")
    df["lesion_id"] = df["lesion_id"].fillna(df["image"])
    df["sample_id"] = df["image"]
    write(df.drop(columns=["stem"]), "isic2019", "lesion_id",
          ["age_approx", "sex", "anatom_site_general"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["padufes20", "ham10000", "isic2019"])
    args = ap.parse_args()
    {"padufes20": prepare_padufes20, "ham10000": prepare_ham10000,
     "isic2019": prepare_isic2019}[args.dataset]()


if __name__ == "__main__":
    main()
