#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements-external.txt

python scripts/prepare_external_benchmarks.py \
  --download-rpeval \
  --benchpres \
  --output-dir data/external_processed

python scripts/export_pairwise_training_data.py \
  --pairs data/external_processed/query_preference_contrast_pairs.jsonl \
  --pairs data/external_processed/context_decision_flip_pairs.jsonl \
  --output data/external_processed/caid_pairwise_train.jsonl

python scripts/run_external_caid_baseline.py \
  --examples data/external_processed/all_applicability_examples.jsonl \
  --flip-pairs data/external_processed/context_decision_flip_pairs.jsonl \
  --output-dir results/external_caid_v0 \
  --alpha 0.10 \
  --mode full

echo
echo "CPU data/baseline pipeline completed."
echo "For fair GPU comparison on exactly the same pair texts:"
echo
echo "  pip install -r requirements-train.txt"
echo "  python scripts/train_pointwise_baseline.py --lora"
echo "  python scripts/train_caid_pairwise.py --lora"
