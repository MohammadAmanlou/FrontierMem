#!/usr/bin/env bash
# Read-only consolidated status, safe after closing/reopening the laptop.
set -euo pipefail
cd "$HOME/research/FrontierMem"
STAMP=data/local/hpc_pipeline/queued_first_stages.txt
if [[ -f "$STAMP" ]]; then
  cat "$STAMP"
else
  echo "No chained job IDs have been registered."
fi
echo "=== QUEUE ==="
squeue -u "$USER" || true
echo "=== STAGE RESULTS ==="
for flag in bootstrap_PASS smoke_PASS; do
  if [[ -f "data/local/hpc_pipeline/$flag" ]]; then
    echo "$flag: PASS"
  else
    echo "$flag: not yet completed"
  fi
done
echo "=== LOGS ==="
for f in fm-gpu-probe-4290248.out fm-gpu-probe-4290248.err fm-qwen-env-*.out fm-qwen-env-*.err fm-qwen-smoke-*.out fm-qwen-smoke-*.err; do
  if [[ -f "$f" ]]; then
    echo "--- $f ---"
    tail -n 30 "$f"
  fi
done
echo "=== SAVED RUN SUMMARIES ==="
find data/local/hpc_pipeline -maxdepth 3 -name training_summary.json -print 2>/dev/null || true
