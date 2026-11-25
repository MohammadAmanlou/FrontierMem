#!/usr/bin/env bash
# Add full 27B and 30B experiments WITHOUT changing any existing 9B job.
# From a detached, immutable worktree: bash scripts/hpc_submit_multibackbone.sh submit /absolute/original/repo
set -Eeuo pipefail
umask 077
MODE="${1:-}"
shift || true

case "$MODE" in
train)
  SRC="$1"; WT="$2"; ROOT="$3"; MODEL="$4"; COMMIT="$5"
  cd "$WT"
  test "$(git rev-parse HEAD)" = "$COMMIT"
  git diff --quiet HEAD -- scripts frontiermem
  source "$HOME/research/.venvs/frontiermem-cu128/bin/activate"
  export HF_HOME="$HOME/research/.cache/huggingface"
  export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
  ARMS=(pointwise caid caid_s2pref dual_axis)
  SEEDS=(17 23 42)
  IDX="${SLURM_ARRAY_TASK_ID:?missing array index}"
  (( IDX >= 0 && IDX <= 11 ))
  ARM="${ARMS[$((IDX % 4))]}"
  SEED="${SEEDS[$((IDX / 4))]}"
  OUT="$ROOT/seed_$SEED/$ARM"
  mkdir -p "$OUT"
  trap 'rc=$?; printf "exit=%s line=%s\n" "$rc" "$LINENO" > "$OUT/FAILED"; exit "$rc"' ERR
  # All arms share the same immutable Hugging Face revision.
  # Model metadata lookup happens in the allocation, never on login.
  if [[ ! -s "$ROOT/model_revision.txt" ]]; then
    REV=$(python -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$MODEL")
    [[ "$REV" =~ ^[0-9a-f]{40}$ ]]
    printf '%s\n' "$REV" > "$ROOT/model_revision.tmp.$SLURM_JOB_ID"
    # Atomic first-writer-wins publication; also safe on shared storage.
    ln "$ROOT/model_revision.tmp.$SLURM_JOB_ID" "$ROOT/model_revision.txt" 2>/dev/null || test -s "$ROOT/model_revision.txt"
    rm -f "$ROOT/model_revision.tmp.$SLURM_JOB_ID"
  fi
  REV=$(cat "$ROOT/model_revision.txt")
  [[ "$REV" =~ ^[0-9a-f]{40}$ ]]
  printf '%s\n' "$COMMIT" > "$OUT/code_commit.txt"
  printf '%s\n' "$REV" > "$OUT/model_revision.txt"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu.csv"
  python -m pip freeze > "$OUT/environment.txt"
  sha256sum "$SRC/data/local/dual_axis_plan/experiment_plan.json" "$SRC/data/external_processed/caid_pairwise_train.jsonl" "$SRC/data/external_processed/all_applicability_examples.jsonl" "$SRC/data/local/s2pref_candidates/"s2pref_context_{train,dev}_CANDIDATES.jsonl > "$OUT/data_sha256.txt"
  echo "FULL TRAINING: $MODEL $REV $ARM seed=$SEED"
  python - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA and BF16 required"
p=torch.cuda.get_device_properties(0)
assert p.total_memory >= 75*1024**3, p.total_memory
print(p.name, p.total_memory/1024**3)
PY
  if [[ ! -s "$OUT/training_summary.json" ]]; then
    python -m scripts.train_dual_axis_qwen \
      --model "$MODEL" --revision "$REV" --arm "$ARM" --seed "$SEED" \
      --manifest "$SRC/data/local/dual_axis_plan/experiment_plan.json" \
      --rpeval-pairs "$SRC/data/external_processed/caid_pairwise_train.jsonl" \
      --s2pref-dir "$SRC/data/local/s2pref_candidates" \
      --allow-silver-exploratory --epochs 1 --pairs-per-batch 1 --grad-accum 8 \
      --max-train-steps 0 --max-eval-pairs 0 --max-length 768 \
      --lora-r 16 --lora-alpha 32 --save-adapter --output-dir "$OUT"
  fi
  test -s "$OUT/training_summary.json"
  test -s "$OUT/best_adapter/adapter_config.json"
  python -m scripts.eval_weekly_heldout \
    --runs-root "$ROOT" --arm "$ARM" --seed "$SEED" --model "$MODEL" --revision "$REV" \
    --examples "$SRC/data/external_processed/all_applicability_examples.jsonl" \
    --batch-size 2 --max-length 768
  test -s "$OUT/official_eval/metrics.json"
  date -Is > "$OUT/PASS"
  rm -f "$OUT/FAILED"
  echo "FULL TRAINING AND HELDOUT PASS: $MODEL $ARM $SEED"
  ;;
report)
  WT="$1"; ROOT="$2"; LABEL="$3"; COMMIT="$4"
  cd "$WT"
  test "$(git rev-parse HEAD)" = "$COMMIT"
  source "$HOME/research/.venvs/frontiermem311/bin/activate"
  python -m scripts.aggregate_weekly_results --runs-root "$ROOT" --output-dir "$ROOT/$LABEL"
  ;;
submit)
  SRC=$(realpath "${1:?Original FrontierMem repository path required}")
  WT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
  test "$SRC" != "$WT"
  COMMIT=$(git -C "$WT" rev-parse HEAD)
  git -C "$WT" diff --quiet HEAD -- scripts frontiermem
  SELF="$WT/scripts/hpc_submit_multibackbone.sh"
  bash -n "$SELF"
  test -x "$HOME/research/.venvs/frontiermem-cu128/bin/python"
  for file in data/local/hpc_pipeline/bootstrap_PASS data/local/dual_axis_plan/experiment_plan.json data/external_processed/caid_pairwise_train.jsonl data/external_processed/all_applicability_examples.jsonl data/local/s2pref_candidates/s2pref_context_train_CANDIDATES.jsonl data/local/s2pref_candidates/s2pref_context_dev_CANDIDATES.jsonl; do
    test -s "$SRC/$file"
  done
  # Earlier replacement script would cancel 9B. Stop if already used.
  if [[ -e "$SRC/data/local/final_27b_2026w48/queued_jobs.tsv" ]]; then
    echo "Earlier 27B submission record exists; inspect it before adding jobs."
    exit 1
  fi
  ROOT="$SRC/data/local/multibackbone_2026w48"
  if [[ -e "$ROOT" ]]; then
    echo "Submission directory already exists; do NOT submit duplicates."
    cat "$ROOT/queued_jobs.tsv" 2>/dev/null || true
    exit 1
  fi
  if [[ -n "$(squeue -u "$USER" -h -n fm-27b,fm-27b-report,fm-q38,fm-g42,fm-q38-report,fm-g42-report)" ]]; then
    echo "Potential duplicate model jobs; inspect current queue."
    exit 1
  fi
  # At most one GPU per new family, plus existing 9B allocations.
  mkdir "$ROOT"
  mkdir "$ROOT/qwen38_27b" "$ROOT/granite42_30b" "$ROOT/logs"
  printf '%s\n' "$COMMIT" > "$ROOT/code_commit.txt"
  RECORD="$ROOT/queued_jobs.tsv"
  printf 'model\tstage\tjob_id\n' > "$RECORD"
  trap 'echo "Submission stopped; recorded jobs may still be held. Inspect $RECORD. No old jobs were cancelled." >&2' ERR
  submit() {
    local family="$1" stage="$2" id
    shift 2
    id=$(sbatch --parsable --hold "$@") || return 1
    id=$(cut -d ';' -f1 <<< "$id")
    [[ "$id" =~ ^[0-9]+$ ]] || return 1
    printf '%s\t%s\t%s\n' "$family" "$stage" "$id" >> "$RECORD" || return 1
    printf '%s\n' "$id"
  }
  train_opts=(--partition=cuda --account=cuda --qos=cuda --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=96G --gres=gpu:1 --exclude=cn04,cn05,cn06,cn07,cn09,cn10,cn11,cn14,gpu01 --time=2-00:00:00 --output="$ROOT/logs/%x-%A_%a.out" --error="$ROOT/logs/%x-%A_%a.err")
  report_opts=(--partition=mid_mdbf --account=mdbf --qos=mid_mdbf --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:20:00 --output="$ROOT/logs/%x-%j.out" --error="$ROOT/logs/%x-%j.err")
  for family in qwen38_27b granite42_30b; do
    if [[ "$family" == qwen38_27b ]]; then
      model=Qwen/Qwen3.8-27B; name=fm-q38
    else
      model=ibm-granite/granite-4.2-30b; name=fm-g42
    fi
    out="$ROOT/$family"
    first=$(submit "$family" primary "${train_opts[@]}" --job-name="$name" --array=0-3%1 "$SELF" train "$SRC" "$WT" "$out" "$model" "$COMMIT")
    second=$(submit "$family" replications "${train_opts[@]}" --job-name="$name" --array=4-11%1 --dependency=afterok:"$first" --kill-on-invalid-dep=yes "$SELF" train "$SRC" "$WT" "$out" "$model" "$COMMIT")
    early=$(submit "$family" primary_report "${report_opts[@]}" --job-name="$name-report" --dependency=afterany:"$first" "$SELF" report "$WT" "$out" meeting_primary "$COMMIT")
    final=$(submit "$family" final_report "${report_opts[@]}" --job-name="$name-report" --dependency=afterany:"$first":"$second" "$SELF" report "$WT" "$out" meeting "$COMMIT")
  done
  cat "$RECORD"
  # Release only our own newly submitted IDs. Never cancel or update 9B.
  while IFS=$'\t' read -r family stage id; do
    [[ "$id" =~ ^[0-9]+$ ]] || continue
    scontrol release "$id"
  done < "$RECORD"
  echo "ALL 24 ADDITIONAL FULL EXPERIMENTS QUEUED"
  squeue -u "$USER"
  ;;
*)
  echo "Usage: $0 submit /absolute/original/FrontierMem" >&2
  exit 2
  ;;
esac
