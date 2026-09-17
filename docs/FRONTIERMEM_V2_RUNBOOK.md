# FrontierMem v2 — Execution Runbook

## Goal

Move FrontierMem away from a synthetic-dataset-centered paper and toward an
**existing-data-first Preference Decision Identifiability / CAID paper**.

The immediate empirical question becomes:

> With the same released personalization examples and the same base model,
> does pairwise applicability-boundary supervision improve rational preference
> use over ordinary pointwise training, especially on held-out and external
> context shifts?

No new natural-language benchmark is required for the first main experiment.

---

## Data roles

| Source | Role | Training? | Main use |
|---|---|---:|---|
| RPEval data-generation pool | primary supervised pool | yes | native IGNORE/SUPPORT/DOMINATE supervision |
| RPEval benchmark | held-out in-domain evaluation | no | direct comparison on rational memory utilization |
| BenchPreS | external selectivity challenge | **never** | APPLY vs suppress under communication contexts |
| S2Pref | context/clarification evaluation | not yet | stable/situational ambiguity and clarification |
| RealPref | transfer | no | unseen-scenario / long-horizon generalization |
| PersonaMem-v2 | transfer | no | implicit preference / long-context personalization |
| AlpsBench | real-dialogue transfer | no | preference/constraint utilization on real dialogues |

BenchPreS is hard-coded as `eval_only`.

---

## What this patch adds

### Data layer
`frontiermem/external_data.py`

- Normalizes RPEval and BenchPreS into one atomic applicability schema.
- Preserves original source labels and IDs.
- Keeps RPEval native three-way labels.
- Marks benchmark/test data so it cannot silently enter training.

### Contrast layer
`frontiermem/contrast_sets.py`

Two distinct existing-data pair types:

1. `query_preference_contrast`
   - same released query
   - different released preferences
   - one applicable, one ignored
   - naturally available in RPEval multi-preference examples
   - usable for training

2. `context_decision_flip`
   - same released preference
   - different released contexts
   - one APPLY, one IGNORE
   - especially strong on BenchPreS
   - used as held-out boundary evaluation when source is evaluation-only

No text is generated in either case.

### Identifiability layer
`frontiermem/identifiability.py`

- Exact hypothesis-consensus decision rule.
- Split-conformal abstention layer:
  singleton APPLY -> APPLY
  singleton IGNORE -> IGNORE
  ambiguous/empty set -> CLARIFY

### Fair model comparison

`train_pointwise_baseline.py`

- Qwen/Llama/Gemma-style sequence classifier
- native RPEval 3-way CE:
  IGNORE / SUPPORT / DOMINATE
- exact same pair-derived texts and deterministic split as CAID
- no pairwise ranking objective

`train_caid_pairwise.py`

- same model
- same native 3-way CE
- same text budget
- adds only:
  `lambda * max(0, margin - (score_apply - score_ignore))`

where:

`score_apply = logsumexp(SUPPORT, DOMINATE) - IGNORE`

This is the cleanest first ablation of the method.

---

## Phase A — local preparation

From repository root:

```bash
unzip FrontierMem_existing_data_v2_patch.zip -d /tmp/frontiermem-v2
cp -r /tmp/frontiermem-v2/* .
```

Run unit tests:

```bash
PYTHONPATH=. pytest -q \
  tests/test_identifiability.py \
  tests/test_external_data.py
```

Expected from this delivered patch: **6 tests pass**.

Install external-data dependencies:

```bash
pip install -r requirements-external.txt
```

---

## Phase B — obtain released data

Recommended for RPEval because its generation pool may use Git LFS:

```bash
git lfs install
git clone https://github.com/XueyangFeng/RPEval \
  data/external_raw/RPEval
git -C data/external_raw/RPEval lfs pull
```

Then prepare RPEval + BenchPreS:

```bash
python scripts/prepare_external_benchmarks.py \
  --rpeval-dir data/external_raw/RPEval \
  --benchpres \
  --output-dir data/external_processed
```

The script creates a group-safe train/calibration partition **before**
pair construction.

Important outputs:

```text
data/external_processed/
  all_applicability_examples.jsonl
  rpeval_explicit.jsonl
  rpeval_implicit.jsonl
  benchpres.jsonl
  query_preference_contrast_pairs.jsonl
  context_decision_flip_pairs.jsonl
  natural_invariance_pairs.jsonl
  dataset_registry.json
  summary.json
```

Exact source filenames can differ slightly because source-specific normalized
files use their source key.

---

## Phase C — create training pairs

```bash
python scripts/export_pairwise_training_data.py \
  --pairs data/external_processed/query_preference_contrast_pairs.jsonl \
  --pairs data/external_processed/context_decision_flip_pairs.jsonl \
  --output data/external_processed/caid_pairwise_train.jsonl
```

Because pair usage is source-aware:

- RPEval training-pool pairs can enter training.
- BenchPreS pairs stay evaluation-only and are automatically excluded.

---

## Phase D — CPU plumbing baseline

```bash
python scripts/run_external_caid_baseline.py \
  --examples data/external_processed/all_applicability_examples.jsonl \
  --flip-pairs data/external_processed/context_decision_flip_pairs.jsonl \
  --output-dir results/external_caid_v0 \
  --alpha 0.10 \
  --mode full
```

This is **not** intended to be a paper-level model.
It verifies:

- split correctness
- source mapping
- direct applicability metrics
- conformal coverage/clarification
- pair-direction metrics

---

## Phase E — first fair GPU experiment

Install:

```bash
pip install -r requirements-train.txt
```

### Pointwise baseline

```bash
python scripts/train_pointwise_baseline.py \
  --pairs data/external_processed/caid_pairwise_train.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output-dir results/pointwise_qwen3b \
  --epochs 2 \
  --lora
```

### CAID pairwise

```bash
python scripts/train_caid_pairwise.py \
  --pairs data/external_processed/caid_pairwise_train.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output-dir results/caid_pairwise_qwen3b \
  --epochs 2 \
  --lambda-rank 1.0 \
  --margin 1.0 \
  --lora
```

These runs use:

- same base model
- same training pair records
- same native RPEval labels
- same deterministic train/dev pair split

The difference is the ranking term.

---

## Phase F — evaluate both checkpoints

```bash
python scripts/evaluate_caid_checkpoint.py \
  --checkpoint results/pointwise_qwen3b \
  --examples data/external_processed/all_applicability_examples.jsonl \
  --output-dir results/pointwise_qwen3b_eval \
  --alpha 0.10

python scripts/evaluate_caid_checkpoint.py \
  --checkpoint results/caid_pairwise_qwen3b \
  --examples data/external_processed/all_applicability_examples.jsonl \
  --output-dir results/caid_pairwise_qwen3b_eval \
  --alpha 0.10
```

The evaluator reports:

### RPEval-native
- 3-way accuracy
- 3-way Macro-F1

### Unified applicability
- binary APPLY/IGNORE accuracy
- binary Macro-F1
- conformal coverage
- selective accuracy
- clarification rate
- appropriate application rate
- misapplication rate

This allows direct RPEval-native reporting and cross-dataset applicability
reporting in the same experiment.

---

## Phase G — S2Pref

Do not guess its heterogeneous nested labels.

First:

```bash
python scripts/inspect_s2pref.py \
  --output-dir data/external_raw/S2Pref
```

This writes a schema report.

For the paper, use S2Pref only after mapping its exact published tasks to:
- context alignment
- conflict/ambiguity
- clarification

The intended use is external clarification validation, not silently relabeling
the whole corpus.

---

## Phase H — next model families

Only after the Qwen3B pipeline is stable:

```text
Qwen2.5-3B
Qwen2.5-7B
Llama-3.x-8B
Gemma-scale comparable model
```

Run pointwise and CAID with the same data/split/metrics.

The paper-level claim should require the direction of the gain to reproduce
across model families.

---

## Main paper tables we should target

### Table 1 — Native RPEval
Rows:
- published / reproduced RPEval baselines
- pointwise same-model baseline
- CAID pairwise

Columns:
- IGNORE / SUPPORT / DOMINATE metrics
- native accuracy / Macro-F1

### Table 2 — External applicability selectivity
Datasets:
- RPEval held-out
- BenchPreS

Metrics:
- AAR
- MR
- binary Macro-F1
- pair direction accuracy

### Table 3 — Selective personalization
For several alpha values:
- coverage
- selective accuracy
- clarification rate
- misapplication rate

### Table 4 — Transfer
- S2Pref
- RealPref / PersonaMem-v2
- AlpsBench

Keep each benchmark's native evaluation protocol.

---

## What counts as a successful first result?

Before making a top-tier claim, we want:

1. CAID > pointwise on RPEval native or applicability metrics without using more
   underlying texts.
2. Larger improvement on held-out contrast/context tests than on IID-like tests.
3. Lower BenchPreS misapplication rate without collapsing appropriate
   application rate.
4. Stable selective-risk curve: lower misapplication as clarification increases.
5. Reproduction on at least two model families before finalizing the method.

If CAID does not beat pointwise on the external boundary tests, do not add more
architecture. Diagnose the pair construction / representation first.

---

## What to push now

Push the new pipeline code and docs, but **do not delete the old prototype**.

The old 4,608-example synthetic suite remains useful as:
- historical motivation
- smoke/regression test
- controlled illustration

It should not be the main empirical evidence of the final paper.
