#!/usr/bin/env bash
# One-time, fail-closed Slurm submission; 4 primary runs then 8 replication runs.
# Exact dependencies prevent GPU usage before the existing Qwen smoke passes.
set -Eeuo pipefail
cd "$HOME/research/FrontierMem"
SMOKE_ID="${1:-4290280}"
[[ "$SMOKE_ID" =~ ^[0-9]+$ ]] || { echo "Usage: bash scripts/hpc_queue_weekly.sh [SMOKE_JOB_ID]" >&2; exit 2; }
scontrol show job "$SMOKE_ID" >/dev/null || {
  echo "Smoke job not known to Slurm; refusing to enqueue." >&2
  exit 3
}
test -s data/local/dual_axis_plan/experiment_plan.json || {
  echo "Missing local experiment plan." >&2
  exit 4
}
test -s data/local/s2pref_candidates/s2pref_context_train_CANDIDATES.jsonl || {
  echo "Missing local S2Pref candidate train data." >&2
  exit 5
}
test -s data/external_processed/caid_pairwise_train.jsonl
source "$HOME/research/.venvs/frontiermem311/bin/activate"
python -m scripts.train_dual_axis_qwen --arm dual_axis --model Qwen/Qwen3.5-9B --dry-run \
  > /dev/null
ROOT="$PWD/data/local/weekly_2026w48"
mkdir -p "$ROOT"
RECORD="$ROOT/queued_jobs.tsv"
if [[ -e "$RECORD" ]]; then
  echo "Weekly pipeline was already submitted (or partially submitted). Do not duplicate:"
  cat "$RECORD"
  exit 6
fi
# Test the two accounts/partitions without occupying resources.
sbatch --test-only --account=cuda --array=0-3%1 \
  --dependency=afterok:"$SMOKE_ID" scripts/hpc_weekly_train.sbatch || {
    echo "CUDA/GPU submission test failed; no actual weekly jobs submitted" >&2
    exit 7
  }
sbatch --test-only --account=mdbf scripts/hpc_weekly_report.sbatch || {
  echo "MDBF/CPU submission test failed; no actual weekly jobs submitted" >&2
  exit 8
}
{
  echo -e "stage$tjob_id$tdependency"
  echo -e "existing_smoke$t$SMOKE_ID$tpreexisting"
} > "$RECORD"
# Write each submitted job immediately so failures cannot silently duplicate submissions.
FIRST="$(sbatch --parsable --account=cuda --array=0-3%1 \
  --dependency=afterok:"$SMOKE_ID" scripts/hpc_weekly_train.sbatch)"
echo -e "primary_seed17_4arms$t$FIRST$tafterok:$SMOKE_ID" >> "$RECORD"
echo "Primary 4-arm array: $FIRST"

EARLY="$(sbatch --parsable --account=mdbf \
  --dependency=afterany:"$FIRST" scripts/hpc_weekly_report.sbatch)"
echo -e "early_report$t$EARLY$tafterany:$FIRST" >> "$RECORD"
echo "Early meeting report: $EARLY"

SECOND="$(sbatch --parsable --account=cuda --array=4-11%1 \
  --dependency=afterok:"$FIRST" scripts/hpc_weekly_train.sbatch)"
echo -e "replication_8arms$t$SECOND$tafterok:$FIRST" >> "$RECORD"
echo "Replication array: $SECOND"

EVAL="$(sbatch --parsable --account=cuda \
  --dependency=afterok:"$SECOND" scripts/hpc_weekly_eval.sbatch)"
echo -e "heldout_eval$t$EVAL$tafterok:$SECOND" >> "$RECORD"
echo "Held-out final evaluation: $EVAL"

FINAL="$(sbatch --parsable --account=mdbf \
  --dependency=afterany:"$EVAL" scripts/hpc_weekly_report.sbatch)"
echo -e "final_report$t$FINAL$tafterany:$EVAL" >> "$RECORD"
echo "Final meeting report: $FINAL"
echo "==== Registered pipeline ===="
cat "$RECORD"
squeue -u "$USER"
echo "Submission complete. Jobs run independently of VPN/laptop connection."
