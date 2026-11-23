#!/usr/bin/env bash
# One-time, unattended dependency chain:
# Existing RTX probe -> CPU package bootstrap -> RTX Qwen3.5-9B real smoke.
# Existing A100 probe remains untouched. No unattended full GPU sweeps.
set -Eeuo pipefail
cd "$HOME/research/FrontierMem"
PROBE=4290248
if (( $# > 0 )); then PROBE="$1"; fi
if [[ ! "$PROBE" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/hpc_queue_first_stages.sh [EXISTING_RTX_PROBE_JOB_ID]" >&2
  exit 2
fi
scontrol show job "$PROBE" >/dev/null || {
  echo "Probe $PROBE is not in current Slurm job history; stop." >&2
  exit 3
}
STATUS_DIR="$PWD/data/local/hpc_pipeline"
mkdir -p "$STATUS_DIR"
STAMP="$STATUS_DIR/queued_first_stages.txt"
if [[ -e "$STAMP" ]]; then
  echo "Existing submission found; refusing duplicate jobs:"
  cat "$STAMP"
  exit 4
fi
echo "Queueing under existing GPU probe job $PROBE (afterok dependencies)."
ENV_ID="$(sbatch --parsable --dependency=afterok:"$PROBE" \
  scripts/hpc_bootstrap_qwen_env.sbatch)"
printf 'gpu_probe_id=%s\nbootstrap_job_id=%s\n' "$PROBE" "$ENV_ID" > "$STAMP"
echo "CPU environment job submitted: $ENV_ID"
if ! SMOKE_ID="$(sbatch --parsable --dependency=afterok:"$ENV_ID" \
  scripts/hpc_smoke_qwen35.sbatch)"; then
  echo "GPU smoke submission failed; environment job exists: $ENV_ID" >&2
  echo "Check QOS/time limits. Do not launch duplicates." >&2
  exit 5
fi
printf 'gpu_smoke_job_id=%s\n' "$SMOKE_ID" >> "$STAMP"
echo "GPU smoke job submitted: $SMOKE_ID"
echo "Submission record: $STAMP"
echo "The old A100 job is unchanged."
squeue -u "$USER"
