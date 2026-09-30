#!/usr/bin/env bash
# Download the dermatology-track datasets from their OFFICIAL sources (no Kaggle account needed).
# Companion to download_data.sh, which uses Kaggle mirrors. Safe to re-run: finished files are
# skipped and partial files resume (curl -C -). Usage: bash scripts/download_data_official.sh [DATA_ROOT]
#
#   PAD-UFES-20 metadata  Mendeley Data V1, DOI 10.17632/zr7vgbcyr2.1
#   HAM10000 imgs+masks   Harvard Dataverse v4, DOI 10.7910/DVN/DBW86T (file ids below)
#   ISIC 2019 imgs+CSVs   ISIC challenge S3
#   Bissoto repos         GitHub
# Not included: PAD-UFES-20 images (Mendeley exposes no direct link for the image folders; see
# download_padufes_images.sh), DDI (Stanford AIMI agreement).
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)/data}"
RAW="$ROOT/raw"
mkdir -p "$RAW"

fetch() {  # fetch URL DEST [EXPECTED_BYTES]
  local url="$1" dest="$2" want="${3:-}"
  if [ -s "$dest" ] && { [ -z "$want" ] || [ "$(stat -c%s "$dest")" = "$want" ]; }; then
    echo "skip $(basename "$dest") (complete)"; return
  fi
  mkdir -p "$(dirname "$dest")"
  echo "[$(date +%H:%M:%S)] $(basename "$dest")"
  curl -L --fail --retry 10 --retry-delay 15 --retry-all-errors -C - -o "$dest" "$url"
}

DV="https://dataverse.harvard.edu/api/access/datafile"
ISIC="https://isic-challenge-data.s3.amazonaws.com/2019"

# TARGET=ham|isic|small|all selects a subset so transfers can run in parallel.
TARGET="${TARGET:-all}"

if [ "$TARGET" = all ] || [ "$TARGET" = ham ]; then
  fetch "$DV/3172585" "$RAW/ham10000/HAM10000_images_part_1.zip" 1366522108
  fetch "$DV/3172584" "$RAW/ham10000/HAM10000_images_part_2.zip" 1403566547
  fetch "$DV/4338392?format=original" "$RAW/ham10000/HAM10000_metadata.csv"
  fetch "$DV/3838943" "$RAW/ham10000-masks/HAM10000_segmentations_lesion_tschandl.zip" 10808743
fi

if [ "$TARGET" = all ] || [ "$TARGET" = isic ]; then
  fetch "$ISIC/ISIC_2019_Training_Metadata.csv"    "$RAW/isic2019/ISIC_2019_Training_Metadata.csv"
  fetch "$ISIC/ISIC_2019_Training_GroundTruth.csv" "$RAW/isic2019/ISIC_2019_Training_GroundTruth.csv"
  fetch "$ISIC/ISIC_2019_Training_Input.zip"       "$RAW/isic2019/ISIC_2019_Training_Input.zip"
fi

if [ "$TARGET" = all ] || [ "$TARGET" = small ]; then
  fetch "https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/fa850265-57da-48f0-ba3e-998b3e44b1f6/file_downloaded" \
        "$RAW/pad-ufes-20-official/metadata.csv" 316209
  echo "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527  $RAW/pad-ufes-20-official/metadata.csv" | sha256sum -c -
  for repo in artifact-generalization-skin debiasing-skin; do
    dest="$ROOT/annotations/$repo"
    if [ -d "$dest/.git" ]; then git -C "$dest" pull --ff-only
    else git clone --depth 1 "https://github.com/alceubissoto/$repo.git" "$dest"; fi
    git -C "$dest" rev-parse HEAD > "$dest.commit.txt"
  done
fi

echo "[$(date +%H:%M:%S)] TARGET=$TARGET done"
