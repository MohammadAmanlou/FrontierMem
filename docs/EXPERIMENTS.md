# Experiment inventory

## Controlled local prototype
The repository includes the synthetic FrontierMem dataset generator, classical baselines, FrontierMem-Lite rules, a tiny Transformer, and a hybrid Tiny-Transformer-memory + rule-policy system.

## Qwen2.5-0.5B zero-shot baseline
The original small-LLM experiment is preserved in the root `results/` files (`metrics_qwen_small.csv`, `test_predictions_qwen_small.csv`, and extraction metrics). It collapsed to a conservative action in the early run and motivated stronger prompting and modular ablations.

## Qwen2.5-1.5B ablations (72 examples each)
Four modes are preserved under `results/qwen_1_5b/`:
1. `gold_memory_llm`: gold structured memory + LLM decision
2. `rule_memory_llm`: rule-extracted memory + LLM decision
3. `llm_memory_rule`: LLM-extracted memory + deterministic rule policy
4. `full_llm`: LLM memory extraction + LLM decision

Key accuracies:
- Gold Memory + LLM Decision: 62.50%
- Rule Memory + LLM Decision: 70.83%
- LLM Memory + Rule Policy: 66.67%
- Full LLM: 33.33% (collapsed to IGNORE on this run)

The complete short report is in `reports/qwen_1_5b/`.

## Qwen2.5-3B run
A 72-example `gold_memory_llm` run was completed in Kaggle. Aggregate metrics:
- Accuracy / balanced accuracy: 75.00%
- Macro-F1: 73.55%
- Inside / outside / near accuracy: 45.83% / 79.17% / 100.00%
- Overgeneralization: 4.17%
- Unnecessary clarification: 27.08%
- Predictions: APPLY=12, IGNORE=23, CLARIFY=37

The raw per-example CSV was created in Kaggle but was not provided as a standalone upload. The executed log is preserved in the final Kaggle notebook; no per-example 3B rows have been fabricated in this repository snapshot.
