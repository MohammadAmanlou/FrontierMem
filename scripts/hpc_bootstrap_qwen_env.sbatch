#!/usr/bin/env bash
#SBATCH --job-name=fm-qwen-env
#SBATCH --partition=short_mdbf
#SBATCH --qos=short_mdbf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:45:00
#SBATCH --output=fm-qwen-env-%j.out
#SBATCH --error=fm-qwen-env-%j.err
# CPU job; no GPU reserved while downloading packages.
set -Eeuo pipefail
umask 077
cd "$HOME/research/FrontierMem"
ROOT="$HOME/research"
ENV_DIR="$ROOT/.venvs/frontiermem-cu128"
STATUS_DIR="$PWD/data/local/hpc_pipeline"
mkdir -p "$STATUS_DIR"
rm -f "$STATUS_DIR/bootstrap_PASS"
trap 'echo "BOOTSTRAP_FAILED; check fm-qwen-env logs" >&2' ERR
echo "bootstrap job: $SLURM_JOB_ID"
date -Is

# Keep historical environment immutable; create a dedicated reproducible venv.
BASE_PYTHON="$ROOT/.venvs/frontiermem311/bin/python"
"$BASE_PYTHON" --version
if [[ ! -f "$ENV_DIR/pyvenv.cfg" ]]; then
  "$BASE_PYTHON" -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
python -m pip install --disable-pip-version-check --no-cache-dir \
  --retries 2 --timeout 60 --upgrade "pip==25.2"

# Blackwell GPU requires CUDA 12.8-capable PyTorch wheels. CPU job installs;
# compute GPU job verifies driver before importing/using CUDA.
python -m pip install --disable-pip-version-check --no-cache-dir \
  --retries 2 --timeout 60 "torch==2.9.1" \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install --disable-pip-version-check --no-cache-dir \
  --retries 2 --timeout 60 \
  "transformers==5.20.2" "peft==0.22.0" \
  "accelerate==1.16.0" "bitsandbytes==0.50.2" \
  "safetensors" "huggingface_hub" "scikit-learn"

python - <<'PY'
import importlib.metadata as md, json
import torch, transformers, peft, bitsandbytes
versions = {p: md.version(p) for p in (
    "torch","transformers","peft","accelerate","bitsandbytes",
    "huggingface_hub","safetensors")}
assert "+cu128" in torch.__version__, (
    "Expected torch cu128 wheel, got " + torch.__version__)
print(json.dumps({"status":"BOOTSTRAP_IMPORTS_PASS","versions":versions}, indent=2))
PY
python scripts/hpc_model_env_preflight.py > "$STATUS_DIR/bootstrap_packages.json"
python -m pip freeze > "$STATUS_DIR/bootstrap_freeze.txt"
date -Is > "$STATUS_DIR/bootstrap_PASS"
echo "BOOTSTRAP_SUCCESS (package install only; no GPU or model loaded)"
