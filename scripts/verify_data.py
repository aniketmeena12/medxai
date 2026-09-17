"""Check downloaded (Kaggle mirror) data against facts from the official sources
(docs/research/01-dataset-verification.md). Run after download, before prepare_data.py.

    python scripts/verify_data.py
Exit code 1 if any check fails.
"""

from __future__ import annotations

import hashlib
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("MEDFUSION_DATA", REPO / "data"))
RAW = DATA_ROOT / "raw"
# SHA256 of PAD-UFES-20 metadata.csv from the Mendeley Data API (checked 2026-09-17).
PAD_METADATA_SHA256 = "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527"
PAD_COLUMNS = {"patient_id", "lesion_id", "img_id", "smoke", "drink", "background_father",
               "background_mother", "age", "pesticide", "gender", "skin_cancer_history",
               "cancer_history", "has_piped_water", "has_sewage_system", "fitspatrick", "region",
               "diameter_1", "diameter_2", "diagnostic", "itch", "grew", "hurt", "changed",
               "bleed", "elevation", "biopsed"}

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))
    print(("PASS " if ok else "FAIL ") + msg)


def zip_names(folder: Path) -> list[str]:
    names = []
    for zp in folder.glob("*.zip"):
        with zipfile.ZipFile(zp) as z:
            names += z.namelist()
    return names


def read_from_zip(folder: Path, suffix: str) -> bytes | None:
    for zp in folder.glob("*.zip"):
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if n.endswith(suffix):
                    return z.read(n)
    return None


def stems(names: list[str], ext: str) -> set[str]:
    return {Path(n).stem for n in names if n.lower().endswith(ext)}


def verify_pad() -> None:
    folder = RAW / "pad-ufes-20"
    names = zip_names(folder)
    pngs = stems(names, ".png")
    check(len(pngs) == 2298, f"PAD-UFES-20: {len(pngs)} unique .png images (expected 2298)")
    official = RAW / "pad-ufes-20-official" / "metadata.csv"
    check(official.exists(), "PAD-UFES-20: official Mendeley metadata.csv present")
    if not official.exists():
        return
    raw = official.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    check(sha == PAD_METADATA_SHA256,
          f"PAD-UFES-20: metadata.csv SHA256 matches Mendeley ({sha[:12]})")
    meta = pd.read_csv(pd.io.common.BytesIO(raw))
    check(set(meta.columns) == PAD_COLUMNS, f"PAD-UFES-20: 26 columns as in the paper "
          f"(missing {PAD_COLUMNS - set(meta.columns)}, extra {set(meta.columns) - PAD_COLUMNS})")
    check(len(meta) == 2298 and meta["patient_id"].nunique() == 1373
          and meta["lesion_id"].nunique() == 1641,
          f"PAD-UFES-20: rows {len(meta)}, patients {meta['patient_id'].nunique()}, "
          f"lesions {meta['lesion_id'].nunique()} (expected 2298 / 1373 / 1641)")
    ids = {Path(i).stem for i in meta["img_id"]}
    check(ids <= pngs,
          f"PAD-UFES-20: every metadata img_id has an image ({len(ids - pngs)} missing)")


def verify_ham() -> None:
    names = zip_names(RAW / "ham10000")
    jpgs = stems(names, ".jpg")
    check(len(jpgs) == 10015, f"HAM10000: {len(jpgs)} unique .jpg images (expected 10015)")
    raw = read_from_zip(RAW / "ham10000", "HAM10000_metadata.csv")
    check(raw is not None, "HAM10000: HAM10000_metadata.csv present")
    if raw is not None:
        meta = pd.read_csv(pd.io.common.BytesIO(raw))
        check(len(meta) == 10015 and set(meta["image_id"]) <= jpgs,
              f"HAM10000: metadata rows {len(meta)}, all image_ids have images")
        check(sorted(meta["dx"].unique()) == ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
              "HAM10000: 7 diagnosis classes")
    masks = {s.replace("_segmentation", "") for s in stems(zip_names(RAW / "ham10000-masks"),
                                                            ".png")}
    check(len(masks) == 10015 and masks == jpgs,
          f"HAM10000 masks: {len(masks)} masks, one per image")


def verify_isic() -> None:
    folder = RAW / "isic2019"
    jpgs = {s.replace("_downsampled", "") for s in stems(zip_names(folder), ".jpg")}
    check(len(jpgs) == 25331, f"ISIC 2019: {len(jpgs)} unique images (expected 25331)")
    # Official CSVs call 2,074 images "ISIC_xxx_downsampled"; trap sets use the plain id.
    gt = pd.read_csv(folder / "ISIC_2019_Training_GroundTruth.csv")
    meta = pd.read_csv(folder / "ISIC_2019_Training_Metadata.csv")
    for frame in (gt, meta):
        frame["image"] = frame["image"].str.replace("_downsampled", "", regex=False)
    check(set(gt["image"]) <= jpgs, f"ISIC 2019: all {len(gt)} ground-truth images present")
    trap_dir = DATA_ROOT / "annotations" / "artifact-generalization-skin" / "trap_sets"
    trap_ids = set()
    for csv in trap_dir.glob("*.csv"):
        trap_ids |= set(pd.read_csv(csv, usecols=["image"])["image"])
    check(bool(trap_ids) and trap_ids <= jpgs,
          f"Trap sets: {len(trap_ids)} image ids, {len(trap_ids - jpgs)} missing from ISIC 2019")
    covered = meta[meta["image"].isin(trap_ids)]
    sex_known = covered["sex"].isin(["male", "female"]).mean() if len(covered) else 0
    check(len(covered) == len(trap_ids),
          f"Trap sets: metadata rows for {len(covered)}/{len(trap_ids)} images; "
          f"sex known for {sex_known:.1%}")


def main() -> None:
    for fn in (verify_pad, verify_ham, verify_isic):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - one missing dataset should not hide the others
            check(False, f"{fn.__name__}: {type(e).__name__}: {e}")
    failed = [m for ok, m in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
