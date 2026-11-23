#!/usr/bin/env bash
#SBATCH --job-name=fm-week-eval
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --account=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=fm-week-eval-%j.out
#SBATCH --error=fm-week-eval-%j.err
set -Eeuo pipefail
umask 077
cd "$HOME/research/FrontierMem"
source "$HOME/research/.venvs/frontiermem-cu128/bin/activate"
test -f data/local/hpc_pipeline/smoke_PASS
export HF_HOME="$HOME/research/.cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
python -m scripts.eval_weekly_heldout \
  --runs-root data/local/weekly_2026w48 \
  --require-all \
  --batch-size 2 \
  --max-length 768
