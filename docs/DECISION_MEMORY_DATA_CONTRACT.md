# Decision-memory data contract (v0.1)

The Day-1 selector is a **model-independent scaffold**, not a trained Qwen system or a proof of future sufficiency.

## Evidence schema
Each retained item maps to one source-backed preference leaf:
```json
{"evidence_id":"e7","leaf_id":"style","text":"Only during debugging, keep answers concise.","cost_tokens":12,"source_id":"history_01:turn_7"}
```
`cost_tokens` is measured with a single pinned tokenizer. `source_id` identifies the original evidence span. Preserve conditions/exceptions and provenance across memory compression.

## Probe schema
```json
{"probe_id":"q1","group_id":"history_01_flip_1","preference_id":"style","context":"debugging","query":"Fix this Python exception","gold_action":"APPLY","split":"selection","source_id":"human_review_01","pair_id":"flip_1","source_usage":"human_validated"}
```
- Gold decisions must be independently validated, not copied from CAID or an unreviewed LLM.
- `CLARIFY` is the existing repo's name for the proposed `UNRESOLVED` action; not an upstream RPEval or S2Pref native label.
- `selection` probes can influence evidence choice; `heldout` probes must never influence selection. Groups must be disjoint, including paraphrase/related source histories.
- The constructor rejects `source_usage=eval_only/test/unverified` for selection probes. The data loader must additionally enforce the immutable upstream source registry; metadata supplied by the caller alone cannot prove provenance.
- A `pair_id` denotes a validated two-sided pair with one APPLY and one IGNORE. Report the number of valid pairs separately.

## API example
```python
from frontiermem.decision_memory import greedy_backward_select, heldout_comparison

selected = greedy_backward_select(
    evidence, selection_probes, predict_fixed_checkpoint, budget_tokens=128
)
heldout = heldout_comparison(
    evidence, selected["selected_evidence_ids"],
    selection_probes, heldout_probes, predict_fixed_checkpoint
)
```
The `predict_fixed_checkpoint` callable must accept `(tuple[Evidence,...], DecisionProbe)` and return APPLY/IGNORE/CLARIFY. Implementing this real Qwen3 adapter is a subsequent sprint gate.

## Selection algorithm
Backward greedy: compute full-memory reference-label loss; for each removable evidence item, predict all selection probes without it; calculate the loss change per saved token; remove the smallest-cost candidate, and repeat until under budget. Cache by retained evidence IDs. This heuristic does **not** guarantee a minimal sufficient subset; it can be expensive for large memories.

Weights are configurable; by default an incorrect APPLY on IGNORE costs more than a missed APPLY. Report the resulting coverage and false-APPLY rate in addition to weighted loss. No single weighting is universal.

## Critical distinction
- **Decision preservation**: chosen memory agrees with the full-history model.
- **Decision correctness**: chosen memory agrees with independently validated gold.
Agreement alone is not evidence of correctness. Any universal sufficiency claim is explicitly out of scope. Heldout tests estimate generalization only to their defined query distribution.
