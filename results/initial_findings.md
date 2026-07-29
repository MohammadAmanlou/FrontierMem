# FrontierMem Initial Prototype: Preliminary Findings

## Experimental setup

- Controlled suite: **4,608 examples** from **768 counterfactual twin history groups**.
- Domains: writing, education, travel, and food.
- Main evaluation: a **lexical holdout challenge**, where one query paraphrase per region and scenario family is unseen during baseline training.
- Decisions: `APPLY`, `IGNORE`, or `CLARIFY`.
- Test examples: **1,568**; training examples: **3,040**.

## Main preliminary result

The strongest system is **FrontierMem-Lite**:

- Accuracy: **0.947**
- Balanced accuracy: **0.894**
- Macro-F1: **0.920**
- Counterfactual twin discrimination: **0.844**
- Overgeneralization rate: **0.091**

For comparison:

- Raw long-context TF-IDF accuracy: **0.816**
- Raw long-context TF-IDF twin discrimination: **0.853**
- Flat fact-memory overgeneralization: **1.000**

## Interpretation

The flat fact-memory baseline succeeds only when the stored preference happens to apply and overgeneralizes on every outside-boundary example. The raw text classifier learns many surface cues, but its performance drops under held-out paraphrases. FrontierMem-Lite explicitly separates stable preferences from temporary constraints, context-dependent preferences, and preferences owned by another person. It then checks the current context and chooses whether to apply, ignore, or clarify.

## What this result demonstrates

1. Preference overgeneralization is measurable with controlled counterfactual twins.
2. A structured applicability representation supports interpretable decisions.
3. The Apply/Clarify/Ignore policy can reduce both outside-boundary misuse and unnecessary questions.
4. The complete experimental pipeline is feasible on a laptop and is ready to be upgraded to an SFT-trained LLM extractor.

## Important limitation

This is a **controlled offline proof-of-concept**, not the final LLM-based system. FrontierMem-Lite currently uses a transparent rule-based extractor, while the comparison classifiers are lightweight TF-IDF models. The initial experiment validates the problem formulation, data structure, metrics, and end-to-end pipeline. The next phase should replace the extractor and response generator with an instruction-tuned 3B--8B open-source LLM trained using structured SFT and counterfactual contrastive examples.
