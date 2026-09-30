#!/usr/bin/env bash
# PAD-UFES-20 images (2,298 PNGs). The official Mendeley record (DOI 10.17632/zr7vgbcyr2.1) exposes
# no direct link for its image folders and its browser download runs at ~22 KB/s, so the images come
# from a mirror. Two routes, in order of preference:
#
#   1. Kaggle mirror `mahdavi1202/skin-cancer` (route used 2026-09-17; needs ~/.kaggle/kaggle.json):
#        TARGET=kaggle bash scripts/download_padufes_images.sh
#   2. Hugging Face mirror `SalmaneExploring/pad-ufes-20` (no credentials; keeps the original
#      filenames and imgs_part_1/2/3 layout, and its metadata.csv is identical to the
#      SHA256-verified official Mendeley CSV):
#        bash scripts/download_padufes_images.sh
#
# Either way `scripts/verify_data.py` checks 2,298 PNGs against every img_id in the official
# metadata, and only the official CSV is ever used for metadata.
set -euo pipefail

ROOT="${DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data}"
DEST="$ROOT/raw/pad-ufes-20"
PY="${PY:-$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python}"
mkdir -p "$DEST"

if [ "${TARGET:-hf}" = kaggle ]; then
  ${KAGGLE:-python -m kaggle} datasets download -d mahdavi1202/skin-cancer -p "$DEST"
  echo "mahdavi1202/skin-cancer" > "$DEST/.complete"
  exit 0
fi

HF_HUB_ENABLE_HF_TRANSFER=1 "$PY" - "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download

dest = sys.argv[1]
path = snapshot_download(
    repo_id="SalmaneExploring/pad-ufes-20",
    repo_type="dataset",
    local_dir=f"{dest}/hf-SalmaneExploring-pad-ufes-20",
    max_workers=8,
)
print("downloaded to", path)
PY
echo "SalmaneExploring/pad-ufes-20 (HF mirror)" > "$DEST/.complete"
