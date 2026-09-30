#!/usr/bin/env bash
# Compute the pre-registered endpoints for every complete run of a dataset, then aggregate into
# results/tables/. Resumable: (seed, fold) pairs already computed are skipped.
#   bash scripts/queue_endpoints.sh padufes20 P4 P5
#   bash scripts/queue_endpoints.sh ham10000            # P1 P4 P5
# Log: data/logs/endpoints_<dataset>.log
set -uo pipefail
cd "$(dirname "$0")/.."
DATASET="${1:?usage: queue_endpoints.sh <dataset> [endpoints...]}"
shift || true
ENDPOINTS=("$@")
[ ${#ENDPOINTS[@]} -eq 0 ] && ENDPOINTS=(P1 P4 P5)
PY="${PY:-.venv/bin/python}"
LOG="data/logs/endpoints_${DATASET}.log"
mkdir -p "$(dirname "$LOG")"
export PYTHONUNBUFFERED=1
echo "[$(date '+%F %T')] endpoints ${ENDPOINTS[*]} for $DATASET" >> "$LOG"
if "$PY" scripts/evaluate_endpoints.py --data "configs/data_${DATASET}.yaml" \
     --endpoints "${ENDPOINTS[@]}" >> "$LOG" 2>&1; then
  echo "[$(date '+%F %T')] endpoints finished for $DATASET" >> "$LOG"
else
  echo "[$(date '+%F %T')] endpoints FAILED for $DATASET" >> "$LOG"
fi
