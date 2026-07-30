# Preliminary Results Summary

## Dataset

- 4,608 controlled examples
- 3,456 training examples
- 1,152 held-out test examples
- 768 counterfactual history groups
- 2,304 twin query pairs
- 12 scenario families in four domains

## Main observation

Flat fact memory applies preferences outside their valid context. Explicit applicability boundaries reduce this overgeneralization. A tiny learned memory typer combined with a transparent boundary policy reaches 93.7% test accuracy and 90.9% Macro-F1 in the controlled experiment.

## Limitation

The pure tiny end-to-end Transformer is too small and is trained from scratch; it collapses toward the majority action. This negative result is useful: language pretraining or stronger counterfactual supervision is needed for the fully learned version. The included Qwen2.5-0.5B runner provides the next experiment.
