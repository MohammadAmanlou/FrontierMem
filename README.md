# FrontierMem: Updated Prototype with a Small Transformer and Qwen Backend

This package is a complete, reproducible proof of concept for learning **when a stored user preference should be applied, ignored, or clarified**.

## What is included

1. **Controlled counterfactual dataset generator**
   - 12 scenario families across writing, education, travel, and food.
   - 4,608 examples: 3,456 train and 1,152 held-out test.
   - Each group contains stable/scoped twin histories and inside/outside/near-boundary queries.

2. **Classical baselines**
   - Flat Fact Memory (always applies a stored preference).
   - Query-Only TF-IDF + Logistic Regression.
   - Raw Long-Context word/character TF-IDF + Logistic Regression.

3. **Transparent FrontierMem-Lite**
   - Rule-based family/profile extraction.
   - Explicit `applies_when` and `does_not_apply_when` memory.
   - Apply/Clarify/Ignore policy.

4. **Small learned Transformer**
   - A compact PyTorch Transformer trained from scratch on the controlled data.
   - Shared encoder with heads for family, stable/scoped status, owner, information type, temporal validity, relation, and action.
   - Two evaluations are included:
     - fully learned end-to-end heads;
     - **Tiny Transformer Memory + Rule Policy**, which follows the practical paper design: learned memory typing plus a cheap auditable controller.

5. **Genuine small-LLM backend**
   - Optional runner for `Qwen/Qwen2.5-0.5B-Instruct`.
   - The Qwen model performs structured memory extraction, Apply/Clarify/Ignore selection, and natural-language response generation.
   - It requires `transformers` and model download access, so it is separated from the offline core package.

## Quick start: fully offline local experiment

```bash
python -m pip install -r requirements.txt
python run_all.py
```

The pre-trained local checkpoint is included under:

```text
results/checkpoints/tiny_multitask_lm.pt
```

To retrain it, delete that checkpoint and run:

```bash
python run_tiny_llm.py
```

## Run only the baselines and rule-based prototype

```bash
python run_demo.py
```

## Run the small learned Transformer

```bash
python run_tiny_llm.py
```

## Run Qwen2.5-0.5B-Instruct in Colab or a machine with internet

```bash
python -m pip install -r requirements-hf.txt
python run_qwen_small_llm.py --max-examples 96
```

For a faster smoke test:

```bash
python run_qwen_small_llm.py --max-examples 12
```

The Qwen runner uses JSON-constrained prompting and robust parsing. It saves predictions and metrics in `results/`.

## Current local results

The included run produced the following held-out test results:

| Method | Accuracy | Balanced Acc. | Macro-F1 | Outside Acc. | Near Acc. | Overgeneralization |
|---|---:|---:|---:|---:|---:|---:|
| Flat Fact Memory | 0.667 | 0.333 | 0.267 | 0.000 | 0.000 | 1.000 |
| Query-Only TF-IDF | 0.635 | 0.527 | 0.510 | 0.177 | 0.662 | 0.823 |
| Raw Long-Context TF-IDF | 0.795 | 0.840 | 0.780 | 0.979 | 0.792 | 0.000 |
| FrontierMem-Lite (Rules) | 0.971 | 0.957 | 0.955 | 0.943 | 0.943 | 0.000 |
| Tiny Transformer end-to-end heads | 0.665 | 0.344 | 0.296 | 0.031 | 0.016 | 0.964 |
| **Tiny Transformer Memory + Rule Policy** | **0.937** | **0.898** | **0.909** | **0.859** | **0.859** | **0.083** |

The very small end-to-end model is deliberately underpowered and often collapses toward the majority `APPLY` action. The hybrid version is more meaningful for the research design: it shows that a learned stable/scoped memory representation can be combined with an explicit boundary controller to obtain strong, inspectable decisions.

## Important interpretation

This is still a **controlled synthetic proof of concept**, not a final paper result.

- The dataset is procedurally generated from manually designed scenario schemas.
- The rule-based model knows the scenario vocabulary.
- The local tiny Transformer is trained from scratch, not pretrained on natural language at LLM scale.
- The Qwen backend is the proper small pretrained LLM experiment and should be run on Colab/GPU before making claims about LLM performance.
- The next research step is structured SFT and counterfactual contrastive training on more natural histories, then transfer evaluation on PersonaMem-v2/RealPref-style data.

## Project layout

```text
frontiermem_llm_prototype/
├── frontiermem/
│   ├── data.py             # dataset and counterfactual twins
│   ├── baselines.py        # TF-IDF baselines
│   ├── memory.py           # rule-based FrontierMem-Lite
│   ├── tiny_lm.py          # learned small Transformer
│   ├── hf_small_llm.py     # optional Qwen backend
│   ├── evaluation.py       # boundary and twin metrics
│   └── visualization.py
├── run_demo.py
├── run_tiny_llm.py
├── run_qwen_small_llm.py
├── run_all.py
├── notebooks/
├── data/
├── results/
└── tests/
```
