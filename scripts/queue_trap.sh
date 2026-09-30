#!/usr/bin/env bash
# Block C: ISIC 2019 trap-set cells (P2 curve, P3 metadata grid). Built to share the GPU with
# other workloads: before each attempt it waits for enough free memory, and because every finished
# cell is written to its own JSON, a run killed by an out-of-memory error resumes where it stopped.
#
#   bash scripts/queue_trap.sh curve                # B0 B3 M1, image bias levels (P2)
#   bash scripts/queue_trap.sh grid                 # 2-D image x metadata grid (P3)
#   bash scripts/queue_trap.sh curve B0             # one model
#   NEED_MIB=2500 RETRIES=200 bash scripts/queue_trap.sh curve
# Log: data/logs/trap_<mode>.log
set -uo pipefail
cd "$(dirname "$0")/.."
MODE="${1:?usage: queue_trap.sh <curve|grid> [models...]}"
shift || true
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(B0 B3 M1)
PY="${PY:-.venv/bin/python}"
NEED_MIB="${NEED_MIB:-2500}"     # headroom one cell needs
RETRIES="${RETRIES:-100}"        # attempts per model before giving up
WAIT_S="${WAIT_S:-120}"          # pause between attempts / memory polls
EXTRA="${EXTRA:-}"               # e.g. --published for the secondary analysis
LOG="data/logs/trap_${MODE}.log"
mkdir -p "$(dirname "$LOG")"
export PYTHONUNBUFFERED=1

wait_for_gpu() {
  local waited=0
  while :; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$free" ] && return 0                       # no GPU query: just try
    [ "$free" -ge "$NEED_MIB" ] && return 0
    if [ $((waited % 600)) -eq 0 ]; then
      echo "[$(date '+%F %T')] waiting for GPU: ${free} MiB free, need ${NEED_MIB}" >> "$LOG"
    fi
    sleep "$WAIT_S"; waited=$((waited + WAIT_S))
  done
}

for model in "${MODELS[@]}"; do
  for attempt in $(seq 1 "$RETRIES"); do
    # Resume the newest real run for this model and mode (never a *_debug_* directory).
    resume=""
    for dir in $(ls -d data/experiments/isic2019_trap_"${model}"_"${MODE}"* 2>/dev/null \
                 | grep -v debug | sort -r); do
      if ls "$dir"/*.json >/dev/null 2>&1; then resume="--resume $dir"; break; fi
    done
    wait_for_gpu
    echo "[$(date '+%F %T')] trap $MODE $model attempt $attempt ${resume}" >> "$LOG"
    if "$PY" scripts/train_trap.py --model "$model" --"$MODE" $EXTRA $resume \
         >> "$LOG" 2>&1; then
      echo "[$(date '+%F %T')] $model $MODE done" >> "$LOG"
      break
    fi
    echo "[$(date '+%F %T')] $model $MODE attempt $attempt failed, retrying" >> "$LOG"
    sleep "$WAIT_S"
  done
done
echo "[$(date '+%F %T')] trap queue finished: $MODE" >> "$LOG"
