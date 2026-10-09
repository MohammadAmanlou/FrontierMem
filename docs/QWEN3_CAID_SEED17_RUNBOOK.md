# Qwen3-4B CAID vs. Pointwise: Seed 17 reproducibility record

**Status as of 2026-10-09:** CAID training completed; evaluation of both saved Qwen3-4B checkpoints is running in Kaggle. **No held-out Qwen3 test scores are claimed in this record.**

## What this experiment measures
This experiment isolates **preference-to-query applicability classification**, *not* the entire history-to-memory-to-response pipeline. The sequence classifier predicts native RPEval labels `IGNORE`, `SUPPORT`, or `DOMINATE`. Binary APPLY combines SUPPORT and DOMINATE; binary IGNORE remains IGNORE.

**Important distinction:** training contrast pairs hold the **query fixed and change the preference**. They do not constitute verified "same preference / same query / context flip" counterfactuals. Do not describe the available pairs as real contextual flips.

## Exact data and split
- Pinned input repository commit: `fca62bf316fabd629edf78ea46a36620d331ffe2`
- `data/external_processed/all_applicability_examples.jsonl`: 12,423 atomic records total in the audited snapshot.
- RPEval-generated source: 8,567 records; RPEval Explicit test: 953; RPEval Implicit test: 953; BenchPreS test-only: 1,950.
- `data/external_processed/caid_pairwise_train.jsonl`: 6,336 pair records.
- Seed-17 group-safe 90/10 train-dev partition: 5,694 training pairs / 642 development pairs, 510 training groups / 57 development groups, zero group overlap.
- Test/eval-only sets are **not used to select the checkpoint**, fit hyperparameters, or train the adapter. Evaluation pair file contains held-out contrast pairs only.

## Model, objective, checkpoint selection
- Backbone: `Qwen/Qwen3-4B-Instruct-2507`; Tesla T4; QLoRA NF4 double-quantization, rank 32, alpha 64, dropout 0.05, rsLoRA all-linear adapters.
- LoRA+ style differential learning rates, cosine schedule with warmup; maximum 5 epochs; patience 2.
- Shared training objective for both methods: native 3-way cross-entropy (label smoothing 0.05) + 0.5 × binary applicability BCE.
- CAID additionally uses 0.5 × smooth pairwise rank loss; margin 0.5 and temperature 1.0. Pair scoring: `logsumexp(SUPPORT, DOMINATE) - IGNORE`.
- Checkpoint selection: dev binary macro-F1; tie-break by native 3-way macro-F1, then pair-direction accuracy.
- **CAID best checkpoint: epoch 1**; training ran for three epochs and stopped after patience=2. Detailed actual dev metrics are in `results/qwen3_seed17/caid_dev_epochs.csv`.
- Earlier Pointwise seed-17 training ended after epoch 2, before its planned stop criterion, but its best epoch-1 adapter is preserved. The run used microbatch 1 / accumulation 16; CAID used 2 pairs (4 sequences) / accumulation 4. This is **not** a fully budget/microbatch-matched multi-seed ablation.

## Versioned executable artifacts
- `notebooks/FrontierMem_Seed17_STAGE_A_Train_CAID_Only.ipynb`: **historical** Kaggle Stage A; train CAID only, retain best adapter, and package its checkpoint.
- `notebooks/FrontierMem_Seed17_STAGE_B_Evaluate_Both_NoTraining.ipynb`: historical Kaggle Stage B; read the **existing** Pointwise and CAID checkpoint outputs as Kaggle Notebook Outputs (under `/kaggle/input`), then run inference and calibration only.
- `scripts/v4_1_train_fast_strong.py`: exact Python script embedded by Stage A.
- `scripts/v4_1_evaluate_disjoint_cal.py`: exact Python evaluation script embedded by Stage B.

Both notebooks **explicitly checkout the original pinned commit** then write the embedded script into `scripts/` locally. This preserves the code used by the historical run, even after newer source files are added to `main`. If you want to run scripts directly from current `main`, use the standalone script copies instead. Do not rerun Stage A if the CAID checkpoint already exists.

## Stage B input/output workflow: no repeated large-file upload
In Kaggle Notebook B, use **Add Input → Notebook Output Files → Your Work** to mount:
1. `caid-vs-pointwise` (earlier Pointwise training output)
2. `train-caid` (CAID Stage A output)

The observed input layout includes:
```
/kaggle/input/notebooks/mohammadamanlou/caid-vs-pointwise/frontiermem_v4_final/seed_17/pointwise/train/best/
/kaggle/input/notebooks/mohammadamanlou/train-caid/FrontierMem_Caid_Seed17_Complete_Training_Checkpoint.zip
/kaggle/input/notebooks/mohammadamanlou/train-caid/frontiermem_seed17_no_repeat/caid/train/best/
```
The original Stage B restores from the matching ZIP or extracted directory. Set GPU=T4 and Internet=On in Kaggle. Its final artifact is `FrontierMem_Seed17_Complete_Evaluation_Results.zip`, which excludes `adapter_model.safetensors`.

Publish only the small validated derived metrics after Stage B completes, e.g. `seed17_metrics_both_models.csv`, `seed17_heldout_pair_metrics_both_models.csv`, `seed17_provenance.json`, and an accompanying interpretation note. Do **not** commit training adapters, full Kaggle output archives, the raw prediction table if it contains user-like text, access tokens, or unreviewed third-party data.

## Scientific boundaries of interpretation
- RPEval source text is Chinese; BenchPreS is English and differs in domain/task construction, so cross-benchmark differences do **not** identify a pure language effect.
- Dev performance near 100% should not be treated as evidence of out-of-distribution generalization.
- `raw_AAR` and `raw_MR` in the classifier evaluator are **decision-level proxies**, not BenchPreS's original response-generation/judge metrics.
- Calibration/conformal fitted on RPEval generation-calibration groups does not provide a distribution-free coverage guarantee under RPEval Implicit or BenchPreS shift.
- Evaluate repeated seeds and strong independent benchmarks before writing definitive CAID superiority claims.
