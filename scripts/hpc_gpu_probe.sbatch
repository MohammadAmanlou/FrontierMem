#!/usr/bin/env bash
#SBATCH --job-name=fm-gpu-probe
#SBATCH --partition=short_mdbf
#SBATCH --qos=short_mdbf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=fm-gpu-probe-%j.out
#SBATCH --error=fm-gpu-probe-%j.err
# Request an A100 80GB specifically instead of accepting arbitrary/K80 GPUs.
# The allocation may stay pending if A100 is busy.
# If the scheduler complains about AccountNotAllowed/InvalidAccount,
# consult your cluster association; do not guess another user's account.
set -euo pipefail
echo '=== job identity ==='
date -Is
hostname
printf 'JOB_ID=%s\n' "${SLURM_JOB_ID:-none}"
printf 'VISIBLE_GPU=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
echo '=== GPU hardware ==='
if command -v nvidia-smi >/dev/null; then
  nvidia-smi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
else
  echo 'nvidia-smi not available in job environment'
fi
echo '=== Python/CUDA availability ==='
python3 - <<'PY'
import sys
print('Python', sys.version.replace('\n',' '))
try:
    import torch
    print('Torch', torch.__version__)
    print('CUDA available', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA devices', torch.cuda.device_count())
        print('CUDA GPU 0', torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print('GPU VRAM GiB', round(props.total_memory / 1024**3, 2))
except ImportError:
    print('PyTorch not installed in this environment; nvidia-smi is authoritative for hardware.')
PY
