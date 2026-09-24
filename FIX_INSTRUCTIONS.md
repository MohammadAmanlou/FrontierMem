# FrontierMem v2.4 — GPU protocol fix

This patch addresses two issues discovered after the v2.3 CPU ablations.

## A. Empty-section formatting artifact

After v2.3, RPEval rows have empty `history` and `context`. The old `full`
formatter still inserted literal `[HISTORY]` and `[CONTEXT]` tokens. TF-IDF
normalization made `full` differ from `preference_query`, and conformal
coverage changed sharply for purely formatting reasons.

v2.4 omits empty optional sections. Therefore, on v2.3 RPEval data:
`full == preference_query` at the text level.

## B. GPU train/dev pair leakage

The old GPU scripts randomly split pairs. The same RPEval query/group can
produce many pairs, so random pair splitting can place the same query group in
both train and dev.

v2.4:
- preserves `group_id` in the exported training-pair file
- splits train/dev by `group_id`
- uses the same group-safe split for Pointwise and CAID
- adds held-out pair-direction evaluation to checkpoint evaluation

## Apply

```powershell
$zip="$HOME\Downloads\FrontierMem_v2_4_gpu_protocol_fix.zip"
Test-Path $zip
Remove-Item "$env:TEMP\frontiermem-v2-4" -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -Path $zip -DestinationPath "$env:TEMP\frontiermem-v2-4" -Force
Copy-Item "$env:TEMP\frontiermem-v2-4\*" "." -Recurse -Force
```

## Tests

```powershell
python -m pytest -q tests/test_identifiability.py tests/test_external_data.py tests/test_rpeval_generation.py tests/test_rpeval_protocol.py tests/test_rpeval_label_variants.py tests/test_text_format_protocol.py
```

## Re-export pairs (required because v2.4 adds group_id)

```powershell
python scripts/export_pairwise_training_data.py --pairs data/external_processed/query_preference_contrast_pairs.jsonl --pairs data/external_processed/context_decision_flip_pairs.jsonl --output data/external_processed/caid_pairwise_train.jsonl
```

## Optional CPU sanity check

`full` and `preference_query` should now match on RPEval because optional fields
are empty after v2.3.

## GPU comparison

Use exactly the same model, seed, pair file, epochs, and group-safe split.

Pointwise:
```powershell
python scripts/train_pointwise_baseline.py --pairs data/external_processed/caid_pairwise_train.jsonl --model Qwen/Qwen2.5-3B-Instruct --output-dir results/pointwise_qwen3b_v24 --epochs 2 --batch-size 4 --grad-accum 8 --lr 2e-5 --max-length 1024 --seed 17 --lora
```

CAID:
```powershell
python scripts/train_caid_pairwise.py --pairs data/external_processed/caid_pairwise_train.jsonl --model Qwen/Qwen2.5-3B-Instruct --output-dir results/caid_qwen3b_v24 --epochs 2 --batch-size 2 --grad-accum 8 --lr 2e-5 --max-length 1024 --lambda-rank 1.0 --margin 1.0 --seed 17 --lora
```

Evaluate both:
```powershell
python scripts/evaluate_caid_checkpoint.py --checkpoint results/pointwise_qwen3b_v24 --examples data/external_processed/all_applicability_examples.jsonl --pairs data/external_processed/query_preference_contrast_pairs.jsonl --output-dir results/pointwise_qwen3b_v24_eval --alpha 0.10 --batch-size 4 --max-length 1024

python scripts/evaluate_caid_checkpoint.py --checkpoint results/caid_qwen3b_v24 --examples data/external_processed/all_applicability_examples.jsonl --pairs data/external_processed/query_preference_contrast_pairs.jsonl --output-dir results/caid_qwen3b_v24_eval --alpha 0.10 --batch-size 4 --max-length 1024
```

Important reporting rule:
- report RPEval explicit, RPEval implicit, and BenchPreS separately
- do not headline one pooled pair-direction number across datasets
- conformal results on implicit/BenchPreS are shift diagnostics, not guaranteed
  source-conditional coverage
