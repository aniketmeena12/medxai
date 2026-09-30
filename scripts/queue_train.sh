#!/usr/bin/env bash
# Run the full pre-registered protocol (all folds x all seeds) for every model on one dataset,
# one model at a time. Safe to re-run and to interrupt:
#   - a model whose newest run has metrics.json with "complete_protocol": true is skipped;
#   - a model with an unfinished run resumes it (run_cv skips folds that already have a JSON).
#
#   bash scripts/queue_train.sh padufes20             # B0 B1 B3 B4 B5 M1
#   bash scripts/queue_train.sh ham10000 M1           # only this model
# Log: data/logs/train_<dataset>.log
set -uo pipefail
cd "$(dirname "$0")/.."
DATASET="${1:?usage: queue_train.sh <dataset> [models...]}"
shift || true
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(B0 B1 B3 B4 B5 M1)
PY="${PY:-.venv/bin/python}"
LOG="data/logs/train_${DATASET}.log"
mkdir -p "$(dirname "$LOG")"
export PYTHONUNBUFFERED=1

complete() {  # complete RUN_DIR -> 0 if the run finished the whole protocol
  "$PY" - "$1" <<'PY'
import json, sys, pathlib
m = pathlib.Path(sys.argv[1]) / "metrics.json"
sys.exit(0 if m.exists() and json.loads(m.read_text()).get("complete_protocol") else 1)
PY
}

for model in "${MODELS[@]}"; do
  done_run="" resume=""
  for dir in $(ls -d data/experiments/"${DATASET}"_"${model}"_[0-9]* 2>/dev/null | sort -r); do
    if complete "$dir"; then done_run="$dir"; break; fi
    if ls "$dir"/folds/*.json >/dev/null 2>&1; then resume="--resume $dir"; break; fi
  done
  if [ -n "$done_run" ]; then
    echo "[$(date '+%F %T')] $model already complete ($done_run), skipping" >> "$LOG"; continue
  fi
  echo "[$(date '+%F %T')] training $model ${resume}" >> "$LOG"
  if ! "$PY" scripts/train.py --data "configs/data_${DATASET}.yaml" --model "$model" $resume \
      >> "$LOG" 2>&1; then
    echo "[$(date '+%F %T')] $model FAILED" >> "$LOG"
  else
    echo "[$(date '+%F %T')] $model done" >> "$LOG"
  fi
done
echo "[$(date '+%F %T')] protocol queue finished for $DATASET" >> "$LOG"
