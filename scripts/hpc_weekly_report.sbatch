#!/usr/bin/env bash
#SBATCH --job-name=fm-week-report
#SBATCH --partition=short_mdbf
#SBATCH --qos=short_mdbf
#SBATCH --account=mdbf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=fm-week-report-%j.out
#SBATCH --error=fm-week-report-%j.err
set -Eeuo pipefail
cd "$HOME/research/FrontierMem"
source "$HOME/research/.venvs/frontiermem311/bin/activate"
python -m scripts.aggregate_weekly_results \
  --runs-root data/local/weekly_2026w48 \
  --output-dir data/local/weekly_2026w48/meeting
