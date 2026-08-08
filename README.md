# FrontierMem

**Beyond “The User Likes X”: Learning Counterfactual Preference Applicability Boundaries for Personalized LLMs**

FrontierMem studies a simple but under-tested personalization problem: a stored user preference should not only encode *what* the user likes, but also *when* that preference applies, when it should be ignored, and when the assistant should ask for clarification.

The decision space is **APPLY / IGNORE / CLARIFY**. The prototype uses controlled counterfactual twin histories to expose preference-overgeneralization and includes rule-based, classical ML, tiny-Transformer, and Qwen baselines/ablations.

## Final snapshot contents

- 4,608-example controlled synthetic suite across 12 families and 4 domains.
- Classical baselines and FrontierMem-Lite rule system.
- Tiny Transformer and learned-memory + rule-policy hybrid.
- Qwen2.5-0.5B initial zero-shot experiment.
- Complete Qwen2.5-1.5B 72-example modular ablation suite.
- Qwen2.5-3B 72-example gold-memory decision run.
- Clean Kaggle experiment notebook + exact raw experiment-log notebook.
- Reusable model-comparison script.
- Research concept report and compact 1.5B experiment report (PDF + LaTeX + figures).

## Key results

| System / mode | Accuracy | Macro-F1 | Inside | Outside | Near |
|---|---:|---:|---:|---:|---:|
| Flat Fact Memory | 66.67% | 26.67% | 100.00% | 0.00% | 0.00% |
| Raw Long-Context TF-IDF | 79.51% | 78.02% | 75.00% | 97.92% | 79.17% |
| FrontierMem-Lite (rules) | 97.14% | 95.48% | 98.57% | 94.27% | 94.27% |
| Tiny Transformer Memory + Rule | 93.66% | 90.93% | 97.53% | 85.94% | 85.94% |
| Qwen2.5-1.5B — Gold Memory + LLM | 62.50% | 60.04% | 100.00% | 29.17% | 58.33% |
| Qwen2.5-1.5B — Rule Memory + LLM | 70.83% | 68.45% | 100.00% | 33.33% | 79.17% |
| Qwen2.5-1.5B — LLM Memory + Rule | 66.67% | 65.35% | 83.33% | 37.50% | 79.17% |
| Qwen2.5-1.5B — Full LLM | 33.33% | 16.67% | 0.00% | 100.00% | 0.00% |
| **Qwen2.5-3B — Gold Memory + LLM** | **75.00%** | **73.55%** | **45.83%** | **79.17%** | **100.00%** |

The 3B result is directly comparable to the 1.5B **gold-memory** mode, not to full end-to-end modes. It improves overall accuracy by 12.5 percentage points, while shifting behavior from strong APPLY bias toward much better outside/near-boundary handling.

## Reproduce

Offline/local baselines:
```bash
python -m pip install -r requirements.txt
python run_all.py
```

Qwen ablations (GPU recommended):
```bash
python -m pip install -r requirements-hf.txt
python run_qwen_fast_ablation.py --model Qwen/Qwen2.5-1.5B-Instruct --mode gold_memory_llm --max-examples 72
```

Cross-model aggregation:
```bash
python scripts/compare_qwen_models.py --results results --output results/comparisons
```

See `docs/REPOSITORY_GUIDE.md` and `docs/EXPERIMENTS.md` for the complete artifact map and experiment provenance.

## Interpretation caveat

This remains a controlled synthetic proof of concept. The rule-based system is aligned with scenario templates, and the current Qwen runs are small-sample prompt-based evaluations rather than final fine-tuned results. The next scientific step is structured supervised fine-tuning / counterfactual training on more natural histories and transfer evaluation.
