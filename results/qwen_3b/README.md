# Qwen2.5-3B-Instruct results

The uploaded Kaggle notebook contains a completed 72-example `gold_memory_llm` run for `Qwen/Qwen2.5-3B-Instruct`.

Aggregate results preserved here:
- Accuracy: **0.7500**
- Balanced accuracy: **0.7500**
- Macro-F1: **0.7355**
- Inside accuracy: **0.4583**
- Outside accuracy: **0.7917**
- Near accuracy: **1.0000**
- Overgeneralization rate: **0.0417**
- Unnecessary clarification rate: **0.2708**
- Prediction counts: APPLY=12, IGNORE=23, CLARIFY=37

Important provenance note: the raw per-example CSV was created in the Kaggle runtime, but it was not uploaded as a standalone file with this repository snapshot. Therefore this package does **not** invent or reconstruct per-example predictions. The complete execution log and aggregate outputs are preserved in `notebooks/FrontierMem_Kaggle_Experiment_Log.ipynb`. Re-running the command in that notebook recreates the exact raw CSV.
