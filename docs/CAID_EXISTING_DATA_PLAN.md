# FrontierMem v2: Existing-Data-First CAID Plan

## Decision

We should **not make a new synthetic benchmark the center of the paper**.

The current FrontierMem generator remains useful only as a historical diagnostic
and regression test. The new paper should train/evaluate on released
personalization benchmarks wherever possible, and create *contrast sets by
re-pairing existing benchmark records* rather than generating new language.

## Primary data stack

### 1. RPEval — primary supervised applicability source

Why it fits:
- Atomic preference-query utilization labels.
- `ignore` is a direct non-application decision.
- `support` / `dominate` are direct application decisions.
- Explicit and implicit preference variants are released.
- A separate `data_generation` pool and benchmark test sets are released.

Mapping:
- `ignore` / `A` -> `IGNORE`
- `support`, `supportive`, `dominate`, `B`, `C` -> `APPLY`

Use:
- `data_generation/data.json`: training/development pool if the actual LFS
  object is available.
- `benchmark_dataset/*`: held-out evaluation only.

Do not collapse support vs dominate in analysis: preserve the upstream label,
even though the first applicability head maps both to APPLY.

### 2. BenchPreS — external application/suppression challenge

Why it fits:
- Same stored preferences are evaluated across different recipient/task
  contexts.
- Each preference has a boolean should-apply / should-suppress label.
- This is almost exactly an external test of applicability selectivity.

Critical rule:
- **Evaluation only.**
- The upstream dataset card explicitly says the benchmark should never appear
  in training corpora.
- Our loader hard-codes `usage=eval_only`.

Mapping:
- `preference_label=True` -> `APPLY`
- `preference_label=False` -> `IGNORE`

The strongest natural counterfactual-like pairs come from this dataset:
the exact same preference can be paired across two existing contexts where the
gold decision flips. No new text is generated.

### 3. S2Pref — context/clarification evaluation

Why it fits:
- Stable vs situational preferences.
- Preference conflicts and proactive ambiguity resolution.
- 10K entries and explicit context-aware tasks.

Current handling:
- Download and inspect only.
- The public HF JSON has heterogeneous nested fields; the dataset viewer itself
  reports schema inconsistency.
- We deliberately do **not** guess a CLARIFY label mapping.
- `scripts/inspect_s2pref.py` creates a schema report; then a task-specific
  adapter should be added from the exact released task structure.

This is safer than silently manufacturing labels.

### 4. RealPref / PersonaMem-v2 — long-horizon transfer

These benchmarks are not clean direct APPLY/IGNORE supervision. Preserve their
native MCQ/TF/open-ended evaluation and use them as downstream transfer tests:
does a CAID-trained applicability model improve long-horizon preference
following/generalization without hurting useful personalization?

### 5. AlpsBench — real-dialogue transfer

AlpsBench is especially valuable because it is derived from real human-LLM
dialogues. Relevant tracks:
- Task 4 ability2: preference following.
- Task 4 ability4: constraint following.

Keep the native benchmark protocol. Do not convert its hidden test into a new
training set.

---

## What is new in our method if we use existing data?

The contribution is not dataset creation.

The paper becomes:

1. **Formal object:** Preference Decision Identifiability.
2. **Existing-data transformation:** benchmark-native decision contrast sets.
3. **Learning:** pairwise applicability training on existing opposite-decision
   contexts.
4. **Risk control:** apply/ignore only when the calibrated decision is a
   singleton; otherwise clarify.
5. **Evaluation:** in-domain RPEval + external BenchPreS + S2Pref clarification
   + long-horizon/real-dialogue transfer.

This is cleaner than claiming a synthetic benchmark as the primary novelty.

---

## Natural contrast-set construction

Given released atomic records with fields:

`(preference, context/query, gold applicability)`

we group by exact normalized preference.

A **decision-flip pair** is:

- same released preference,
- two *existing* contexts,
- one gold `APPLY`,
- one gold `IGNORE`.

The pairwise objective is:

`L = L_pointwise + lambda * max(0, margin - (s_apply - s_ignore))`

This is not synthetic augmentation. It directly learns the boundary from
benchmark-observed context variation.

An **invariance pair** is:

- same preference,
- different existing contexts,
- same gold action.

Later we can add an invariance penalty without generating new text.

---

## Immediate experiment ladder

### E0 — data audit
- RPEval counts and label distributions.
- BenchPreS counts and label distributions.
- Number of natural decision-flip pairs.
- Number of invariance pairs.
- Preference overlap and context diversity.

### E1 — classical plumbing baseline
- query-only TF-IDF
- preference+query TF-IDF
- full evidence TF-IDF
- forced binary prediction
- conformal selective prediction

Purpose: verify data/metrics/splits. Not a paper result.

### E2 — standard pretrained pointwise model
Same 3B/7B model, same data budget, pointwise APPLY/IGNORE objective.

### E3 — CAID pairwise model
Same model and same atomic examples plus pairwise ranking loss mined from the
released training pool.

### E4 — selective decision layer
Calibrate on held-out training-pool groups.
- singleton APPLY -> APPLY
- singleton IGNORE -> IGNORE
- ambiguous set -> CLARIFY

Report:
- coverage
- selective accuracy
- misapplication rate
- appropriate application rate
- pair direction accuracy
- clarification rate

### E5 — external evaluation
- RPEval held-out benchmark.
- BenchPreS (strict external eval only).
- S2Pref ambiguity/clarification task.
- RealPref/PersonaMem-v2 transfer.
- AlpsBench real-dialogue utilization/constraint tracks.

---

## Leakage rules

These are non-negotiable for a top-tier submission.

1. Never train on BenchPreS.
2. Never mix RPEval benchmark test files into the training pool.
3. Split RPEval generation-pool calibration by preference/group, not by row.
4. Build pairs only *inside* a usage split.
5. Keep exact source IDs and original labels in all processed files.
6. Never overwrite upstream data.
7. Report every label mapping explicitly.
8. Preserve upstream benchmark-native metrics alongside our unified metrics.

---

## What stays from the old FrontierMem code?

Keep:
- repository structure,
- old controlled generator as `legacy diagnostic`,
- APPLY / IGNORE / CLARIFY interface,
- Qwen ablation results as motivation,
- modular memory-vs-decision finding,
- evaluation utilities.

Move out of the main claim:
- synthetic 4,608-example suite,
- rule-based 97% result,
- hierarchy itself,
- tree/graph storage,
- RL.

The old results motivate the problem; they should not be the final evidence.

---

## Meeting-level claim

> We no longer plan to make a synthetic dataset the contribution. We use
> existing preference benchmarks as the empirical substrate, normalize only
> their released labels, mine contrastive context changes without generating
> new text, and make the contribution the identifiability-aware learning and
> selective decision mechanism.

That is a substantially cleaner experimental story.


## Important refinement: two kinds of benchmark-native pairs

We should distinguish them in the paper and code.

### Query-preference contrast (training-friendly, RPEval)
Same released query, different stored preferences:
- one preference should APPLY;
- one should IGNORE.

This is available naturally in RPEval multi-preference records and gives a
pairwise learning signal without creating any new language.

### Context-decision flip (strong boundary test, BenchPreS and similar sources)
Same released preference, different released contexts:
- APPLY in one context;
- IGNORE in another.

This is the closer analogue of a counterfactual applicability boundary.
BenchPreS is especially valuable here, but it remains evaluation-only.

The first fair model comparison is therefore:
- Pointwise model trained on the flattened exact pair texts.
- CAID model trained on the same exact pair texts + pairwise ranking term.
- Both calibrated on groups held out *before* pair mining.


## Native RPEval comparability

To compare fairly with RPEval/RP-Reasoner, we should not report only a collapsed
binary metric.

The GPU trainers therefore use RPEval's native three labels:

- IGNORE
- SUPPORT
- DOMINATE

The pointwise baseline minimizes ordinary 3-way cross-entropy.

CAID uses the **same model, same texts, same native 3-way cross-entropy**, plus
one extra applicability ranking term:

`app_score = logsumexp(SUPPORT, DOMINATE) - IGNORE`

`L_CAID = L_native_CE + lambda * max(0, margin - (s_positive - s_negative))`

This gives a clean ablation:
the only methodological difference is the pairwise applicability-boundary
supervision.

At evaluation time we report both:
1. native RPEval 3-way accuracy/Macro-F1;
2. unified APPLY-vs-IGNORE + selective/CLARIFY metrics.

BenchPreS remains binary external evaluation.
