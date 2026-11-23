#!/usr/bin/env bash
#SBATCH --job-name=fm-qwen-reload
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --account=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=fm-qwen-reload-%j.out
#SBATCH --error=fm-qwen-reload-%j.err
# Explicit reload check: the existing smoke job only checked adapter files.
set -Eeuo pipefail
umask 077
cd "$HOME/research/FrontierMem"
source "$HOME/research/.venvs/frontiermem-cu128/bin/activate"
STATUS="$PWD/data/local/hpc_pipeline"
test -f "$STATUS/smoke_PASS"
test -f "$STATUS/bootstrap_PASS"
export HF_HOME="$HOME/research/.cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
python - <<'PY'
from pathlib import Path
import math
import torch
from transformers import AutoTokenizer
from scripts.eval_weekly_heldout import load_adapter
path=Path("data/local/hpc_pipeline/qwen35_gpu_smoke_4290280/best_adapter")
assert (path/"adapter_config.json").is_file(), "Missing saved smoke adapter"
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
if tok.pad_token is None:
    tok.pad_token=tok.eos_token
tok.padding_side="right"
model=load_adapter("Qwen/Qwen3.5-9B", path, tok)
toks=tok(["[PREFERENCE]\nI like tea.\n[QUERY]\nWhat drink should I choose?\n[TASK]\nClassify how the preference should affect this request: IGNORE, SUPPORT, or DOMINATE.\n[DECISION]"],
         padding=True,truncation=True,max_length=256,return_tensors="pt")
with torch.inference_mode():
    z=model(**{k:v.to("cuda:0") for k,v in toks.items()}).logits.float()
assert tuple(z.shape)==(1,3), f"Unexpected logits shape {tuple(z.shape)}"
assert torch.isfinite(z).all().item(), "Non-finite logits"
print("RELOAD_PASS adapter restored + real forward CUDA:",z.cpu().tolist(),flush=True)
PY
date -Is > "$STATUS/reload_PASS"
