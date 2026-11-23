#!/usr/bin/env bash
#SBATCH --job-name=fm-week-train
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --account=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=fm-week-train-%A_%a.out
#SBATCH --error=fm-week-train-%A_%a.err
# Submit as two sequential stages: 0-3%1, 4-11%1.
set -Eeuo pipefail
umask 077
cd "$HOME/research/FrontierMem"
source "$HOME/research/.venvs/frontiermem-cu128/bin/activate"
STATUS="$PWD/data/local/weekly_2026w48"
mkdir -p "$STATUS/status"
n="${SLURM_ARRAY_TASK_ID:?Requires Slurm array task ID}"
(( n >= 0 && n <= 11 )) || { echo "Invalid task index $n" >&2; exit 10; }
if [[ -f "$STATUS/status/FAILED" ]]; then
  echo "Upstream task failed; refusing to expend GPU time" >&2
  exit 11
fi
trap 'code=$?; echo "FAILED array_index=$n job=$SLURM_JOB_ID rc=$code" > "$STATUS/status/FAILED"; exit "$code"' ERR
test -f data/local/hpc_pipeline/smoke_PASS
test -f data/local/dual_axis_plan/experiment_plan.json
arms=(pointwise caid caid_s2pref dual_axis)
seeds=(17 23 42)
arm="${arms[$(( n % 4 ))]}"
seed="${seeds[$(( n / 4 ))]}"
out="$STATUS/seed_${seed}/${arm}"
mkdir -p "$out"
if [[ -f "$out/training_summary.json" && -f "$out/best_adapter/adapter_config.json" ]]; then
  echo "Already completed: $out"
  exit 0
fi
if [[ -e "$out/training_summary.json" ]]; then
  echo "Partial/inconsistent existing run: $out" >&2
  exit 12
fi
{
  echo "timestamp=$(date -Is)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "job_id=$SLURM_JOB_ID"
  echo "array_index=$n"
  echo "arm=$arm"
  echo "seed=$seed"
  echo "hostname=$(hostname)"
} > "$out/run_provenance.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
export HF_HOME="$HOME/research/.cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Full train split (5694 pairs), exactly one epoch; --max-train-steps=0.
python -m scripts.train_dual_axis_qwen \
  --arm "$arm" --seed "$seed" --model Qwen/Qwen3.5-9B \
  --allow-silver-exploratory --epochs 1 --pairs-per-batch 1 \
  --grad-accum 8 --max-train-steps 0 --max-eval-pairs 0 \
  --max-length 768 --lora-r 16 --lora-alpha 32 \
  --save-adapter --output-dir "$out"
test -s "$out/training_summary.json"
test -f "$out/best_adapter/adapter_config.json"
date -Is > "$STATUS/status/PASS_${n}"
echo "WEEKLY_TRAIN_PASS index=$n arm=$arm seed=$seed"
