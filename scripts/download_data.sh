#!/usr/bin/env bash
# Download the dermatology-track datasets. Safe to re-run: finished datasets are skipped.
# Usage: bash scripts/download_data.sh [DATA_ROOT]   (default: the project data/ folder)
#
# Sources (switched 2026-09-17: Mendeley ~22 KB/s and Harvard Dataverse ~37 KB/s from this
# network vs Kaggle ~3.5 MB/s). Kaggle copies are checked against official facts after download
# (scripts/verify_data.py) and recorded in each dataset's SOURCE.md.
#   PAD-UFES-20       kaggle mahdavi1202/skin-cancer            (official: Mendeley 10.17632/zr7vgbcyr2.1)
#   HAM10000          kaggle kmader/skin-cancer-mnist-ham10000  (official: Dataverse 10.7910/DVN/DBW86T)
#   HAM10000 masks    kaggle tschandl/ham10000-lesion-segmentations (uploaded by the dataset author)
#   ISIC 2019 images  kaggle kioriaanthony/isic-2019-training-input (official: ISIC challenge S3)
#   ISIC 2019 CSVs    official ISIC challenge S3 (small)
#   Bissoto repos     GitHub
# Not included: DDI (needs the Stanford AIMI research-use agreement; download manually).
# Kaggle auth: ~/.kaggle/access_token.
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)/data}"
RAW="$ROOT/raw"
KAGGLE="${KAGGLE:-python -m kaggle}"
mkdir -p "$RAW"

kaggle_get() {  # kaggle_get OWNER/DATASET DEST_DIR
  local ref="$1" dest="$2"
  if [ -f "$dest/.complete" ]; then echo "skip $ref (done)"; return; fi
  mkdir -p "$dest"
  echo "[$(date +%H:%M:%S)] kaggle $ref -> $dest"
  $KAGGLE datasets download -d "$ref" -p "$dest"
  echo "$ref" > "$dest/.complete"
}

fetch() {  # fetch URL DEST
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then echo "skip $dest (exists)"; return; fi
  mkdir -p "$(dirname "$dest")"
  echo "[$(date +%H:%M:%S)] $dest"
  curl -L --fail --retry 5 --retry-delay 10 -o "$dest.part" "$url" && mv "$dest.part" "$dest"
}

kaggle_get mahdavi1202/skin-cancer                        "$RAW/pad-ufes-20"
kaggle_get tschandl/ham10000-lesion-segmentations         "$RAW/ham10000-masks"
kaggle_get kmader/skin-cancer-mnist-ham10000              "$RAW/ham10000"

# Official PAD-UFES-20 metadata (small; the Kaggle copy was re-saved with different formatting).
fetch "https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/fa850265-57da-48f0-ba3e-998b3e44b1f6/file_downloaded" \
      "$RAW/pad-ufes-20-official/metadata.csv"
echo "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527  $RAW/pad-ufes-20-official/metadata.csv" \
  | sha256sum -c -

ISIC="https://isic-challenge-data.s3.amazonaws.com/2019"
fetch "$ISIC/ISIC_2019_Training_Metadata.csv"    "$RAW/isic2019/ISIC_2019_Training_Metadata.csv"
fetch "$ISIC/ISIC_2019_Training_GroundTruth.csv" "$RAW/isic2019/ISIC_2019_Training_GroundTruth.csv"
kaggle_get kioriaanthony/isic-2019-training-input         "$RAW/isic2019"

for repo in artifact-generalization-skin debiasing-skin; do
  dest="$ROOT/annotations/$repo"
  if [ -d "$dest/.git" ]; then git -C "$dest" pull --ff-only
  else git clone --depth 1 "https://github.com/alceubissoto/$repo.git" "$dest"; fi
  git -C "$dest" rev-parse HEAD > "$dest.commit.txt"
done

( cd "$RAW" && find . -type f \( -name '*.zip' -o -name '*.csv' \) -print0 \
    | xargs -0 sha256sum ) > "$ROOT/SHA256SUMS.txt"
echo "Done. Checksums in $ROOT/SHA256SUMS.txt"
