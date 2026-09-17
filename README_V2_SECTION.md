## FrontierMem v2: existing-data-first CAID direction

The current repository snapshot contains the original controlled FrontierMem
prototype and Qwen ablations. The next research phase changes the empirical
strategy:

- **No new synthetic benchmark as the main contribution.**
- Use released personalization datasets as the primary empirical substrate.
- Normalize only released labels and preserve native benchmark IDs/metrics.
- Mine natural same-preference context pairs from existing records.
- Train an applicability scorer with pointwise + pairwise decision-flip loss.
- Use calibrated selective prediction: APPLY / IGNORE / CLARIFY.

Primary direct-applicability sources:
- **RPEval:** training/development pool + held-out benchmark.
- **BenchPreS:** external evaluation only; never training.

Transfer/clarification sources:
- S2Pref, RealPref, PersonaMem-v2, AlpsBench.

### Run the CPU preparation/baseline pipeline

```bash
bash run_external_pipeline.sh
```

### Run pairwise CAID training on GPU

```bash
pip install -r requirements-train.txt

python scripts/train_caid_pairwise.py \
  --pairs data/external_processed/caid_pairwise_train.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output-dir results/caid_pairwise_qwen3b \
  --lora
```

### Evaluate the trained checkpoint

```bash
python scripts/evaluate_caid_checkpoint.py \
  --checkpoint results/caid_pairwise_qwen3b \
  --examples data/external_processed/all_applicability_examples.jsonl \
  --output-dir results/caid_pairwise_qwen3b_eval
```

See `docs/CAID_EXISTING_DATA_PLAN.md` for the experimental design and leakage
rules.


### Fair same-data baseline vs CAID

```bash
python scripts/train_pointwise_baseline.py \
  --pairs data/external_processed/caid_pairwise_train.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output-dir results/pointwise_qwen3b \
  --lora

python scripts/train_caid_pairwise.py \
  --pairs data/external_processed/caid_pairwise_train.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output-dir results/caid_pairwise_qwen3b \
  --lora
```

These two trainers use the same pair file and deterministic split seed. The
difference is the CAID ranking term, which isolates the contribution of
pairwise applicability supervision.
