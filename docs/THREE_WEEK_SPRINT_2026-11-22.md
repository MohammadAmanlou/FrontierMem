# FrontierMem — 21-day research and engineering sprint
**Dates:** 2026-11-22 to 2026-12-12. **Branch:** `research/decision-sufficient-memory-2026-11`.

## Objective and acceptance criteria
Build and fairly evaluate Pointwise, existing CAID and Dual-Axis CAID, then compare decision-aware evidence selection to memory baselines at identical token budgets. Deliver reproducible data manifests, code, tests, checkpoint provenance, per-source metric tables, uncertainty, figures, error analysis and a paper-ready report. **A positive result or paper acceptance is not guaranteed.**

## Current audited state
- 8,567 RPEval training-pool atoms; 953 each Explicit/Implicit held-out tests; BenchPreS 1,950 eval-only atoms.
- Existing `context_decision_flip_pairs.jsonl` has **0** records. Same-query preference contrast pairs are NOT contextual flips.
- Historical Qwen3-4B Pointwise/CAID checkpoints (seed 17) are already saved in Kaggle, but training exposure differs, so they are not a strict causal comparison.
- Archived S2Pref Phase-01: 400 silver decisions from 20 held-out source groups (200 pairs); CAID pair direction 86%, strict flip 21.5%; Pointwise 80%, 25%. This historical result motivates the redesign; it does not validate it.
- Day-1 model-agnostic memory selection scaffold is demonstrable by `python scripts/demo_decision_memory.py`. **Its outputs are explicitly toy results**.

## Day-by-day deliverables
| Day | Date | Task | Exit gate |
|---|---|---|---|
| 1 | Nov 22 | Audit repo, open research branch, add evidence-selection code/tests/protocol | PR and passing toy tests |
| 2 | Nov 23 | Freeze RPEval/BenchPreS/S2Pref registry, splits, source IDs, license and leakage rules | no train/eval overlap |
| 3 | Nov 24 | Audit S2Pref existing heldout groups; design disjoint context-training pool | source-group manifest |
| 4 | Nov 25 | Build/validate opposite-decision same-preference context pairs from NEW train groups | nonempty verified context pairs |
| 5 | Nov 26 | Implement one shared Qwen3-4B CE3+BCE2+PrefRank+ContextRank trainer | zero-lambda = Pointwise tests |
| 6 | Nov 27 | GPU smoke on Kaggle T4, 32–64 examples, tokenizer/truncation/gradient audits | saves/loads valid checkpoint |
| 7 | Nov 28 | Validate silver labels and freeze all heldout splits | Week-1 GO/NO-GO |
| 8 | Nov 29 | Matched Pointwise seed 17 training | checkpoint, exposure/compute log |
| 9 | Nov 30 | Matched CAID seed 17 training | checkpoint, identical texts/updates |
| 10 | Dec 1 | Dual-Axis CAID seed 17 training | ContextRank isolated |
| 11 | Dec 2 | Frozen RPEval E/I, S2Pref silver heldout, BenchPreS eval-only | per-group prediction CSVs |
| 12 | Dec 3 | Selective/CLARIFY calibration on disjoint development groups | risk–coverage curves |
| 13 | Dec 4 | Evidence spans + provenance adapter from real histories; CAID callback | real-memory smoke |
| 14 | Dec 5 | Selection and heldout probes, fixed budgets, freeze experiment | Week-2 GO/NO-GO |
| 15 | Dec 6 | Greedy versus random, semantic top-k, summary, full-history baselines | budget controlled |
| 16 | Dec 7 | Whole-tree and per-leaf heldout decision quality | accuracy, wrong-APPLY, strict flip |
| 17 | Dec 8 | Remove ContextRank, evidence scope, exceptions, selection, CLARIFY individually | ablation table |
| 18 | Dec 9 | Group-aware bootstrap and, if GPU allows, seeds 23/42 | intervals, seed variability |
| 19 | Dec 10 | Audited downstream response-quality sample | genuine generation-level analysis |
| 20 | Dec 11 | Figures, compute costs, failure cases, latest related work | Results/Method/Limitations |
| 21 | Dec 12 | Freeze hashes, scripts, README and report draft | reproducible release candidate |

## Primary data sources and immutable roles
1. **RPEval:** only `data_generation/data.json` may be training source. Official `benchmark_dataset` is never for training, threshold calibration or prompt engineering.
2. **BenchPreS:** always evaluation-only; published decision-level proxies must not be confused with native response-generation metrics.
3. **S2Pref:** existing 20 source-record heldout groups stay strictly held out. Any new training pairs must be from disjoint source IDs with independent label audit, clearly marked derived silver.
4. **New source-grounded histories:** required to evaluate *whole-memory* selection and sufficiency. Annotate preference spans, scope, exceptions, evidence provenance, and APPLY/IGNORE/CLARIFY probes; double-review ambiguous cases.
5. Split by original user/source/paraphrase group, not row; labels and reference decisions cannot be generated from CAID predictions.

## Main ablation
- E1: shared CE3 + BCE2 (Pointwise).
- E2: E1 + preference ranking (original CAID).
- E3: E2 + context ranking with absolute supervision (Dual-Axis).
- E4: E3 + decision-aware evidence selection.
- E5: E3 + fixed-budget random/semantic/summary selection.
- E6: E4 + dev-calibrated CLARIFY, at matched coverage.

Use `Qwen/Qwen3-4B-Instruct-2507`, pinned tokenizer, identical inputs, groups, seed and optimization budget for E1–E3. Existing Kaggle seed-17 checkpoints are diagnostic, not matched final ablation evidence.

## Metrics and reporting
- Native 3-class and binary Macro-F1, APPLY recall, wrong-APPLY rate, accuracy, confusion matrices.
- S2Pref pair direction **and** strict flip (both sides must be correct).
- Selective risk vs coverage, clarification rate, calibration and Brier score.
- Decision preservation versus full history **separately from correctness against human/reference gold**.
- Token budget (64/128/256/512 as feasible), budget–quality curve, GPU cost, truncation and seeds.
- Bootstrap by original source record/user history; no treating 200 related pairs as 200 independent users.

## Gating / stop-loss
If no validated training context flips by Day 7, **do not fabricate them**: publish honest CAID+memory-selection results, or record blocked context arm. If GPU access fails, finish data, baseline, tests and methods but explicitly flag missing model results. If strict flip does not improve, retain negative results, check calibration/shortcut learning and avoid tuning on test data.

## Reproduce available Day-1 artifact
```bash
python -m pytest -q tests/test_decision_memory.py
python scripts/demo_decision_memory.py
```
The demonstration uses a deterministic toy predictor; it is **not** a Qwen or benchmark result. Future day gates require actual commits, trained artifacts and independently checkable logs.
