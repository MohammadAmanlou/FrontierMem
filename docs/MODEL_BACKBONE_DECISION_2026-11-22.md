# Backbone decision for 2027 FrontierMem research — 22 Nov 2026

**Decision status:** research-design choice; not a completed GPU compatibility check and not an evidence-based claim that any model has superior preference-applicability performance.

## Recommended roles (exact public model IDs)

| Role | Model ID | Release / reason | Training scope |
|---|---|---|---|
| Full ablation backbone | `Qwen/Qwen3.5-9B` | March 2, 2026; 9B dense multimodal/hybrid text core, strong multilingual coverage; affordable enough for four-arm matched runs | 4 arms, >=3 seeds if cluster allocation allows |
| Large-backbone flagship replication | `Qwen/Qwen3.8-27B` | August 14, 2026; newer 27B dense with same Qwen3.5 hybrid architecture family | ideally 4 matched arms, at minimum 2 most diagnostic arms, paired compute report |
| Independent architecture replication | `google/gemma-4-12B-it` | Gemma 4 (2026), distinct architecture | matched baseline vs proposed method; check model access/license |
| Historical baseline | `Qwen/Qwen3-4B-Instruct-2507` | existing historical Oct 2026 checkpoints and trainer | report historical evidence separately; do not imply matched experimental comparisons |

Primary evidence should be **within-backbone, matched-data, matched-budget** ablations, not larger-vs-smaller model comparisons. The large flagship is *conditional* on a verified GPU/software allocation and a complete paired ablation; do not call a model intrinsically best on preference applicability without direct validation.

Official links:
- https://github.com/QwenLM/Qwen3.8
- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/Qwen/Qwen3.5-9B
- https://huggingface.co/google/gemma-4-12B-it
- https://huggingface.co/docs/transformers/model_doc/qwen3_5

## Why these models, not blindly changing a string

- Native RPEval records are Chinese; S2Pref silver candidates are English. The chosen Qwen family has multilingual pretraining and long-context architecture, but neither proves the new method generalizes to novel users or languages.
- Qwen3.5/Qwen3.8 are native multimodal configurations with a hybrid Gated DeltaNet. Existing `AutoModelForSequenceClassification` + QLoRA was originally tested on **Qwen3-4B**; compatibility, classification-head initialization, adapter save/reload, kernel speed and any missing packages **must be probed before training**.
- Hugging Face has `Qwen3_5TextForSequenceClassification`, but the top-level multimodal `Qwen3_5Config`, text-only extraction, and the `num_labels` propagation are version-sensitive. Evaluate a text-only wrapper or a generative verbalized classifier if the sequence head cannot safely load weights. Do not silently change objective when comparing arms.
- 27B QLoRA may work on A100 80GB/H200/RTX PRO 6000 (subject to actual VRAM) but must pass max-length=512, one-pair GPU smoke and a checkpoint reload; V100/K80 have compatibility and/or VRAM limitations, so do not guess readiness from `sinfo` alone.
- Store HF cache and checkpoints under persistent per-user storage and check quota. Never install CUDA stacks or run a model on the login node.

## Paper-relevant data/split design

- Baseline A: native 3-way Pointwise (CE3+BCE2) on RPEval generation train.
- Baseline B: CAID = same RPEval input exposure plus PreferenceRank.
- Augmented C: CAID+S2Pref = CAID plus *derived silver* BCE2 (no invented native SUPPORT/DOMINATE label).
- Proposed D: Dual-Axis = same data and optimizer updates as C, plus ContextRank.
- Explicitly compare A vs B and C vs D as controlled ablations. Compare B vs C as **additional-silver-data exposure** rather than proof of architecture gain.
- Keep S2Pref train/dev source groups disjoint, report additional lexical overlap and an unseen-preference subset.
- Hold RPEval Explicit/Implicit and BenchPreS (test-only) outside optimization. S2Pref held-out source groups must remain untouched. Exclude semantically questionable pairs only via logged manifest.
- Calibrate APPLY/IGNORE/CLARIFY from disjoint calibration groups. Evaluate native 3-way macro-F1, binary macro-F1, wrong-APPLY, flip direction, selective risk-coverage and source-group bootstrap intervals.
- Whole-memory research **still requires new source-grounded user histories**, evidence spans, scope/exception/provenance supervision and token-budget-matched selection baselines. Existing S2Pref candidates have empty histories and one shared generic query. Do not claim long-horizon memory ability from that setup.
- The paper's novel contribution must be **decision-aware preference applicability and memory selection**, not that a new model checkpoint exists.

## Execution gates

0. **Already complete:** data provenance/split audit and four-arm CPU dry-runs, not GPU readiness.
1. Run read-only environment audit: `python scripts/hpc_model_env_preflight.py` on login. Inventory exact GRES and `AllocTRES` for RTX/A100/H200; maintain A100 probe job 4289553 unless valid alternative is confirmed.
2. After getting a GPU allocation, load recommended Python/CUDA environment, install user-local **compatible** PyTorch/Transformers/PEFT/bitsandbytes in an allocated compute session as HPC policy permits. No sensitive credentials in logs.
3. **Architecture gate:** from official model config, instantiate/load exactly one 9B text sequence classifier with three labels; test gradient flow to a trainable head and LoRA, save/reload, then repeat on 27B.
4. **Smoke gate:** one pair / max 512 tokens / 2 steps, RPEval CE3+BCE2 and S2Pref silver BCE+rank, verify finite gradients, actual CUDA use, saved adapter, zero test-set exposure.
5. **Research gate:** matched A/B/C/D at 9B with 3+ random seeds and controlled exposure; validate all eval splits, then replicate paired ablations at 27B on large GPU.
6. **Cross-family gate:** perform matched Gemma 4 12B experiments before asserting backbone-independent improvements.
7. Freeze versions/model revision hashes, prompt formats, seeds, GPU type, VRAM peaks, wall time, training-token budgets, test exposure and statistical inference in paper.

**DO NOT launch new 27B full training until gates 1–4 pass.**
