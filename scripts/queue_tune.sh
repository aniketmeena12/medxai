#!/usr/bin/env bash
# Tune every image model on one dataset, one after another (pre-registration §3: the same 6-trial
# grid for each model, fold 0 / seed 0). Safe to re-run:
#   - models whose configs/tuned/<dataset>_<model>.yaml exists are skipped;
#   - a model with an unfinished tuning run resumes it (finished trials are kept).
#
#   bash scripts/queue_tune.sh padufes20            # configs/data_padufes20.yaml
#   bash scripts/queue_tune.sh ham10000 B0 M1       # only these models
# Log: data/logs/tune_<dataset>.log
set -uo pipefail
cd "$(dirname "$0")/.."
DATASET="${1:?usage: queue_tune.sh <dataset> [models...]}"
shift || true
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(B0 B3 B4 B5 M1)
PY="${PY:-.venv/bin/python}"
LOG="data/logs/tune_${DATASET}.log"
mkdir -p "$(dirname "$LOG")"
export PYTHONUNBUFFERED=1
for model in "${MODELS[@]}"; do
  if [ -f "configs/tuned/${DATASET}_${model}.yaml" ]; then
    echo "[$(date '+%F %T')] $model already tuned, skipping" >> "$LOG"; continue
  fi
  # Resume the newest run for this model that has at least one finished trial.
  resume=""
  for dir in $(ls -d data/experiments/tune_"${DATASET}"_"${model}"_* 2>/dev/null | sort -r); do
    if ls "$dir"/trial*.json >/dev/null 2>&1; then resume="--resume $dir"; break; fi
  done
  echo "[$(date '+%F %T')] tuning $model ${resume}" >> "$LOG"
  if ! "$PY" scripts/tune.py --data "configs/data_${DATASET}.yaml" --model "$model" $resume \
      >> "$LOG" 2>&1; then
    echo "[$(date '+%F %T')] $model FAILED" >> "$LOG"
  fi
done
echo "[$(date '+%F %T')] queue finished for $DATASET" >> "$LOG"
