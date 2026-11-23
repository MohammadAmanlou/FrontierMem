#!/usr/bin/env bash
#SBATCH --job-name=fm-qwen35-smoke
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --account=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=fm-qwen-smoke-%j.out
#SBATCH --error=fm-qwen-smoke-%j.err
# Only starts after successful CPU environment job via afterok dependency.
# No unrestricted full training. Does real model forward/backward on two batches.
set -Eeuo pipefail
umask 077
cd "$HOME/research/FrontierMem"
STATUS_DIR="$PWD/data/local/hpc_pipeline"
mkdir -p "$STATUS_DIR"
rm -f "$STATUS_DIR/smoke_PASS"
test -f "$STATUS_DIR/bootstrap_PASS" || {
  echo "Missing bootstrap_PASS: stop before model download" >&2
  exit 22
}
test -n "$SLURM_JOB_ID"
command -v nvidia-smi >/dev/null || {
  echo "GPU driver tools not visible in this allocation" >&2
  exit 23
}
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
  --format=csv,noheader | tee "$STATUS_DIR/gpu_smoke_hardware.txt"
source "$HOME/research/.venvs/frontiermem-cu128/bin/activate"
python scripts/hpc_model_env_preflight.py --gpu | tee "$STATUS_DIR/gpu_smoke_preflight.json"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable despite Slurm GPU allocation"
major, minor = torch.cuda.get_device_capability(0)
assert major >= 8, "CUDA architecture unsupported for selected QLoRA smoke"
print("CUDA SMOKE DEVICE:",torch.cuda.get_device_name(0),"capability",(major,minor))
x=torch.ones(8,device="cuda"); assert float(x.sum().item()) == 8.0
PY

# Put data and model caches under the account's persistent storage, not /tmp.
export HF_HOME="$HOME/research/.cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"
OUT="$STATUS_DIR/qwen35_gpu_smoke_$SLURM_JOB_ID"
python -m scripts.train_dual_axis_qwen \
  --arm dual_axis \
  --model Qwen/Qwen3.5-9B \
  --allow-silver-exploratory \
  --epochs 1 \
  --pairs-per-batch 1 \
  --max-train-steps 2 \
  --max-eval-pairs 2 \
  --max-length 384 \
  --lora-r 16 \
  --lora-alpha 32 \
  --grad-accum 2 \
  --save-adapter \
  --output-dir "$OUT"
python - "$OUT" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
m=json.loads((root/"training_summary.json").read_text())
assert m["history"] and m["history"][0]["rpeval_batches"] == 2
assert m["history"][0]["partial_training"] is True
assert m["history"][0]["s2pref_silver_dev"]["metrics_are_derived_silver_not_gold"]
assert (root/"best_adapter"/"adapter_config.json").is_file()
assert any((root/"best_adapter").glob("*.safetensors"))
print("SMOKE_PASS: forward, backward, pair losses, adapter files saved")
PY
date -Is > "$STATUS_DIR/smoke_PASS"
echo "SMOKE_SUCCESS; results under $OUT"
