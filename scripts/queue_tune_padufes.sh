#!/usr/bin/env bash
# Tune all image models on PAD-UFES-20, one after another. Safe to re-run:
#   - models whose configs/tuned/padufes20_<model>.yaml exists are skipped;
#   - a model with an unfinished tuning run resumes it (finished trials are kept).
# Log: data/logs/tune_padufes20.log
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/c/medfusion-venv/Scripts/python.exe
LOG=data/logs/tune_padufes20.log
mkdir -p "$(dirname "$LOG")"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1
for model in B0 B3 B4 B5 M1; do
  if [ -f "configs/tuned/padufes20_${model}.yaml" ]; then
    echo "[$(date '+%F %T')] $model already tuned, skipping" >> "$LOG"; continue
  fi
  # Resume the newest run for this model that has at least one finished trial.
  resume=""
  for dir in $(ls -d data/experiments/tune_padufes20_"${model}"_* 2>/dev/null | sort -r); do
    if ls "$dir"/trial*.json >/dev/null 2>&1; then resume="--resume $dir"; break; fi
  done
  echo "[$(date '+%F %T')] tuning $model ${resume}" >> "$LOG"
  if ! "$PY" scripts/tune.py --data configs/data_padufes20.yaml --model "$model" $resume \
      >> "$LOG" 2>&1; then
    echo "[$(date '+%F %T')] $model FAILED" >> "$LOG"
  fi
done
echo "[$(date '+%F %T')] queue finished" >> "$LOG"
