#!/usr/bin/env python3
"""Read-only Sabanci HPC model-environment report; no model download or GPU work.

Use on the LOGIN node without --gpu, or inside an allocated Slurm job with --gpu.
Only prints package versions, CUDA metadata, model ID recommendations, scheduler
inventory. Never installs packages or loads Qwen weights.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys

MODELS = {
    "primary_ablation": "Qwen/Qwen3.5-9B",
    "flagship_large": "Qwen/Qwen3.8-27B",
    "independent_family": "google/gemma-4-12B-it",
    "historical_only": "Qwen/Qwen3-4B-Instruct-2507",
}
PACKAGES = (
    "torch", "transformers", "accelerate", "peft", "bitsandbytes",
    "huggingface_hub", "safetensors", "tokenizers",
    "flash-linear-attention", "causal-conv1d", "triton", "numpy",
)


def version_of(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def inspect(with_gpu: bool = False):
    report = {
        "status": "READ_ONLY_NO_MODEL_DOWNLOAD",
        "host": platform.node(),
        "python_version": sys.version.split()[0],
        "in_slurm_job": bool(os.getenv("SLURM_JOB_ID")),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "models_to_evaluate": MODELS,
        "installed_packages": {p: version_of(p) for p in PACKAGES},
        "nvidia_smi_available": bool(shutil.which("nvidia-smi")),
    }
    if with_gpu:
        if not os.getenv("SLURM_JOB_ID"):
            raise RuntimeError("--gpu is forbidden outside an allocated Slurm job")
        if shutil.which("nvidia-smi"):
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                 "--format=csv,noheader"],
                capture_output=True, text=True, check=False, timeout=20,
            )
            report["nvidia_smi"] = {
                "exit_code": proc.returncode, "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        if version_of("torch"):
            import torch  # import only when explicitly running in a Slurm job
            report["pytorch"] = {
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
            if torch.cuda.is_available():
                report["pytorch"]["gpu0"] = torch.cuda.get_device_name(0)
                report["pytorch"]["gpu0_total_memory_GiB"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
                )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", action="store_true", help="Only inside Slurm compute GPU allocation")
    args = p.parse_args()
    print(json.dumps(inspect(args.gpu), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
