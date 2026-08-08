# FrontierMem Qwen2.5-1.5B Experimental Report

## Setup

We evaluated Qwen2.5-1.5B-Instruct on 72 balanced FrontierMem
examples using four modular experimental settings:

1. Gold memory with LLM decision
2. Rule-extracted memory with LLM decision
3. LLM-extracted memory with rule-based decision
4. Full LLM memory and decision pipeline

## Main Results

| Mode | Accuracy |
|---|---:|
| Gold Memory + LLM Decision | 62.50% |
| Rule Memory + LLM Decision | 70.83% |
| LLM Memory + Rule Policy | 66.67% |
| Full LLM Pipeline | 33.33% |

The full pipeline achieved a Macro-F1 of 16.67%
and a twin exact-match score of 0.00%.

## Interpretation

The results suggest a mixed bottleneck: both memory construction and final applicability decisions contribute to the remaining errors.

Unlike the previous 0.5B zero-shot run, this modular evaluation
allows us to separate errors caused by memory extraction from
errors caused by the final APPLY, IGNORE, or CLARIFY decision.

## Important Caveat

The current dataset is controlled and synthetic. The rule-based
memory extractor is aligned with the scenario templates, so its
performance should not be interpreted as evidence of real-world
generalization.

## Next Step

The next research step is structured supervised fine-tuning on
counterfactual twin histories, followed by evaluation on more
natural and lexically diverse interaction histories.
