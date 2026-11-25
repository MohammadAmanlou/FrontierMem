#!/usr/bin/env bash
# Submit additional full baselines without changing any existing job.
set -Eeuo pipefail
umask 077
MODE="${1:-}"; shift || true
case "$MODE" in
gpu)
 SRC="$1"; WT="$2"; ROOT="$3"; COMMIT="$4"
 cd "$WT"; test "$(git rev-parse HEAD)" = "$COMMIT"
 git diff --quiet HEAD -- scripts frontiermem
 source "$HOME/research/.venvs/frontiermem-cu128/bin/activate"
 export HF_HOME="$HOME/research/.cache/huggingface" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
 ARMS=(pointwise caid caid_s2pref dual_axis); SEEDS=(17 23 42)
 IDX="${SLURM_ARRAY_TASK_ID:?}"
 ARM="${ARMS[$((IDX%4))]}"; SEED="${SEEDS[$((IDX/4))]}"
 OUT="$ROOT/seed_$SEED/$ARM"; mkdir -p "$OUT"
 trap 'rc=$?; printf "exit=%s line=%s\n" "$rc" "$LINENO" > "$OUT/FAILED"; exit "$rc"' ERR
 nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu.csv"
 python -m pip freeze > "$OUT/environment.txt"
 if [[ ! -s "$ROOT/revision.txt" ]]; then
   REV=$(python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("answerdotai/ModernBERT-base").sha)')
   [[ "$REV" =~ ^[0-9a-f]{40}$ ]]
   printf '%s\n' "$REV" > "$ROOT/revision.$SLURM_JOB_ID"
   ln "$ROOT/revision.$SLURM_JOB_ID" "$ROOT/revision.txt" 2>/dev/null || test -s "$ROOT/revision.txt"
   rm -f "$ROOT/revision.$SLURM_JOB_ID"
 fi
 REV=$(cat "$ROOT/revision.txt")
 [[ "$REV" =~ ^[0-9a-f]{40}$ ]]
 python -m scripts.train_modernbert_full --source "$SRC" --out "$OUT" --arm "$ARM" --seed "$SEED" --revision "$REV"
 test -s "$OUT/official_eval/metrics.json"
 rm -f "$OUT/FAILED"
 ;;
cpu)
 SRC="$1"; WT="$2"; ROOT="$3"; COMMIT="$4"
 cd "$WT"; test "$(git rev-parse HEAD)" = "$COMMIT"
 git diff --quiet HEAD -- scripts frontiermem
 source "$HOME/research/.venvs/frontiermem-cu128/bin/activate"
 export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
 MODES=(full no_context); CONTEXT="${MODES[$SLURM_ARRAY_TASK_ID]}"
 OUT="$ROOT/$CONTEXT"; mkdir -p "$OUT"
 trap 'rc=$?; printf "exit=%s line=%s\n" "$rc" "$LINENO" > "$OUT/FAILED"; exit "$rc"' ERR
 python -m scripts.train_lexical_baselines --source "$SRC" --out "$OUT" --context "$CONTEXT"
 date -Is > "$OUT/PASS"
 ;;
report)
 WT="$1"; ROOT="$2"; COMMIT="$3"
 cd "$WT"; test "$(git rev-parse HEAD)" = "$COMMIT"
 source "$HOME/research/.venvs/frontiermem311/bin/activate"
 python -m scripts.aggregate_weekly_results --runs-root "$ROOT" --output-dir "$ROOT/meeting"
 ;;
submit)
 SRC=$(realpath "${1:?source repo}")
 WT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
 test "$WT" != "$SRC"
 COMMIT=$(git -C "$WT" rev-parse HEAD)
 git -C "$WT" diff --quiet HEAD -- scripts frontiermem
 SELF="$WT/scripts/hpc_submit_small_baselines.sh"
 ROOT="$SRC/data/local/small_baselines_2026w48"
 for file in data/local/dual_axis_plan/experiment_plan.json data/external_processed/caid_pairwise_train.jsonl data/external_processed/all_applicability_examples.jsonl data/local/s2pref_candidates/s2pref_context_train_CANDIDATES.jsonl data/local/s2pref_candidates/s2pref_context_dev_CANDIDATES.jsonl; do test -s "$SRC/$file"; done
 if [[ -e "$ROOT" ]]; then echo "Already registered; inspect existing record"; cat "$ROOT/queued_jobs.tsv"; exit 1; fi
 mkdir -p "$ROOT/logs" "$ROOT/modernbert" "$ROOT/lexical"
 printf '%s\n' "$COMMIT" > "$ROOT/code_commit.txt"
 RECORD="$ROOT/queued_jobs.tsv"; printf 'stage\tjob_id\n' > "$RECORD"
 trap 'echo "Submission stopped. See $RECORD; existing main jobs untouched." >&2' ERR
 submit() {
   local stage="$1" id; shift
   id=$(sbatch --parsable --hold "$@") || return 1
   id="${id%%;*}"
   [[ "$id" =~ ^[0-9]+$ ]] || return 1
   printf '%s\t%s\n' "$stage" "$id" >> "$RECORD" || return 1
   printf '%s\n' "$id"
 }
 gpu=(--partition=cuda --account=cuda --qos=cuda --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=1-00:00:00 --output="$ROOT/logs/%x-%A_%a.out" --error="$ROOT/logs/%x-%A_%a.err")
 cpu=(--partition=mid_mdbf --account=mdbf --qos=mid_mdbf --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=04:00:00 --output="$ROOT/logs/%x-%A_%a.out" --error="$ROOT/logs/%x-%A_%a.err")
 # Submit CPU first, then release everything together. No duplicate large-model runs.
 C=$(submit lexical "${cpu[@]}" --job-name=fm-lex --array=0-1%1 "$SELF" cpu "$SRC" "$WT" "$ROOT/lexical" "$COMMIT")
 A=$(submit modernbert_seed17 "${gpu[@]}" --gres=gpu:tesla_v100s:1 --job-name=fm-mbert-s --array=0-3%1 "$SELF" gpu "$SRC" "$WT" "$ROOT/modernbert" "$COMMIT")
 B=$(submit modernbert_seed23_42 "${gpu[@]}" --gres=gpu:tesla_v100:1 --job-name=fm-mbert-v --array=4-11%1 "$SELF" gpu "$SRC" "$WT" "$ROOT/modernbert" "$COMMIT")
 R=$(submit modernbert_report --partition=mid_mdbf --account=mdbf --qos=mid_mdbf --cpus-per-task=1 --mem=4G --time=00:15:00 --job-name=fm-mbert-r --dependency=afterany:"$A":"$B" --output="$ROOT/logs/%x-%j.out" --error="$ROOT/logs/%x-%j.err" "$SELF" report "$WT" "$ROOT/modernbert" "$COMMIT")
 cat "$RECORD"
 for id in "$C" "$A" "$B" "$R"; do scontrol release "$id"; done
 echo "ALL 20 ADDITIONAL FULL BASELINES QUEUED"
 squeue -u "$USER"
 ;;
*) echo "Usage: $0 submit /path/to/original/repo" >&2; exit 2;;
esac
