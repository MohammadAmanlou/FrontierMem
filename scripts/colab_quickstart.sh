#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements-hf.txt
python run_demo.py
python run_tiny_llm.py
python run_qwen_small_llm.py --max-examples 24
