"""Decision-aware evidence selection for preference memory.

A model-agnostic reference implementation. It requires independently labeled
probe decisions; model agreement is not treated as ground-truth sufficiency.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

ACTIONS = {"APPLY", "IGNORE", "CLARIFY"}


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    leaf_id: str
    text: str
    cost_tokens: int
    source_id: str

    def __post_init__(self):
        if not all((self.evidence_id, self.leaf_id, self.text, self.source_id)):
            raise ValueError("Evidence must have id, leaf, text and source provenance")
        if not isinstance(self.cost_tokens, int) or self.cost_tokens <= 0:
            raise ValueError("cost_tokens must be a positive integer from a fixed tokenizer")


@dataclass(frozen=True)
class DecisionProbe:
    probe_id: str
    group_id: str
    preference_id: str
    context: str
    query: str
    gold_action: str
    split: str
    source_id: str
    pair_id: str = ""
    source_usage: str = "unverified"

    def __post_init__(self):
        if not all((self.probe_id, self.group_id, self.preference_id, self.query, self.source_id)):
            raise ValueError("Probe id, group, preference, query and source provenance required")
        if self.gold_action not in ACTIONS:
            raise ValueError("Requires independently validated APPLY, IGNORE or CLARIFY")
        if self.split not in {"selection", "heldout"}:
            raise ValueError("Probe split must be selection or heldout")
        if self.split == "selection" and self.source_usage not in {"train", "selection", "human_validated"}:
            raise ValueError("Selection prohibits eval/test/unverified source_usage")


@dataclass(frozen=True)
class DecisionCosts:
    wrong_apply: float = 3.0
    wrong_ignore: float = 1.0
    incorrect_clarify: float = 0.5
    force_on_ambiguous: float = 3.0

    def __post_init__(self):
        if min(asdict(self).values()) < 0:
            raise ValueError("Decision costs must be non-negative")


Predictor = Callable[[tuple[Evidence, ...], DecisionProbe], str]


def decision_loss(gold: str, pred: str, costs: DecisionCosts) -> float:
    if gold == pred:
        return 0.0
    if gold == "CLARIFY":
        return costs.force_on_ambiguous
    if pred == "CLARIFY":
        return costs.incorrect_clarify
    if pred == "APPLY" and gold == "IGNORE":
        return costs.wrong_apply
    return costs.wrong_ignore


def evaluate_memory(
    evidence: Sequence[Evidence],
    probes: Sequence[DecisionProbe],
    predictor: Predictor,
    costs: DecisionCosts = DecisionCosts(),
) -> dict:
    """Evaluate decisions against validated gold; None denotes no denominator."""
    if not probes:
        raise ValueError("At least one independently labeled probe is required")
    gold = [p.gold_action for p in probes]
    predicted = [predictor(tuple(evidence), p) for p in probes]
    if any(y not in ACTIONS for y in predicted):
        raise ValueError("Predictor returned an unsupported decision")
    binary = [i for i, y in enumerate(gold) if y in {"APPLY", "IGNORE"}]
    accepted = [i for i in binary if predicted[i] != "CLARIFY"]
    neg = [i for i, y in enumerate(gold) if y == "IGNORE"]
    pos = [i for i, y in enumerate(gold) if y == "APPLY"]
    pairs: dict[str, list[int]] = {}
    for i, p in enumerate(probes):
        if p.pair_id:
            pairs.setdefault(p.pair_id, []).append(i)
    flip_pairs = [inds for inds in pairs.values()
                  if len(inds) == 2 and {gold[i] for i in inds} == {"APPLY", "IGNORE"}]

    def rate(inds, pred):
        return sum(bool(pred(i)) for i in inds) / len(inds) if inds else None

    return {
        "n": len(probes),
        "mean_weighted_loss": sum(decision_loss(g, y, costs) for g, y in zip(gold, predicted)) / len(probes),
        "exact_decision_accuracy": sum(g == y for g, y in zip(gold, predicted)) / len(probes),
        "coverage_on_binary_gold": rate(binary, lambda i: predicted[i] != "CLARIFY"),
        "selective_risk": rate(accepted, lambda i: predicted[i] != gold[i]),
        "wrong_apply_rate": rate(neg, lambda i: predicted[i] == "APPLY"),
        "apply_recall": rate(pos, lambda i: predicted[i] == "APPLY"),
        "strict_flip_accuracy": rate(list(range(len(flip_pairs))),
                                     lambda j: all(predicted[i] == gold[i] for i in flip_pairs[j])),
        "n_valid_flip_pairs": len(flip_pairs),
    }


def greedy_backward_select(
    evidence: Sequence[Evidence],
    selection_probes: Sequence[DecisionProbe],
    predictor: Predictor,
    budget_tokens: int,
    costs: DecisionCosts = DecisionCosts(),
) -> dict:
    """Remove the least decision-harmful evidence per saved token, iteratively.

    This is a baseline heuristic, not an optimality or sufficiency proof.
    Crucially, selection probes cannot be official benchmark eval/test rows.
    """
    if not evidence or len({e.evidence_id for e in evidence}) != len(evidence):
        raise ValueError("Nonempty evidence with unique IDs required")
    if not selection_probes or any(p.split != "selection" for p in selection_probes):
        raise ValueError("Selection requires selection-only probes")
    if len({p.probe_id for p in selection_probes}) != len(selection_probes):
        raise ValueError("Duplicate selection probe IDs")
    if not isinstance(budget_tokens, int) or budget_tokens < 0:
        raise ValueError("budget_tokens must be a non-negative integer")

    current = sorted(evidence, key=lambda e: e.evidence_id)
    cache: dict[tuple[str, ...], dict] = {}
    audit = []

    def scored(items):
        key = tuple(sorted(e.evidence_id for e in items))
        if key not in cache:
            cache[key] = evaluate_memory(items, selection_probes, predictor, costs)
        return cache[key]

    full_metrics = scored(current)
    while sum(e.cost_tokens for e in current) > budget_tokens and current:
        before = scored(current)["mean_weighted_loss"]
        candidates = []
        for evidence_item in current:
            remaining = [e for e in current if e.evidence_id != evidence_item.evidence_id]
            after = scored(remaining)["mean_weighted_loss"]
            delta = after - before
            candidates.append((delta / evidence_item.cost_tokens, delta,
                               evidence_item.evidence_id, evidence_item, remaining, after))
        _, delta, _, removed, remaining, after = min(candidates, key=lambda row: row[:3])
        audit.append({
            "removed_evidence_id": removed.evidence_id,
            "removed_leaf_id": removed.leaf_id,
            "saved_tokens": removed.cost_tokens,
            "loss_before": before,
            "loss_after": after,
            "delta_loss": delta,
        })
        current = remaining

    return {
        "algorithm": "backward_greedy_decision_loss_per_token_v1",
        "budget_tokens": budget_tokens,
        "full_tokens": sum(e.cost_tokens for e in evidence),
        "selected_tokens": sum(e.cost_tokens for e in current),
        "selected_evidence_ids": [e.evidence_id for e in current],
        "full_selection_metrics": full_metrics,
        "selected_selection_metrics": scored(current),
        "removal_audit": audit,
        "unique_model_subset_evaluations": len(cache),
        "limitations": "Selection scores are not held-out generalization or universal sufficiency.",
    }


def heldout_comparison(
    evidence: Sequence[Evidence],
    selected_evidence_ids: Sequence[str],
    selection_probes: Sequence[DecisionProbe],
    heldout_probes: Sequence[DecisionProbe],
    predictor: Predictor,
    costs: DecisionCosts = DecisionCosts(),
) -> dict:
    """Evaluate full and selected evidence on independently labeled held-out groups."""
    if not selection_probes or any(p.split != "selection" for p in selection_probes):
        raise ValueError("Expected selection probes")
    if not heldout_probes or any(p.split != "heldout" for p in heldout_probes):
        raise ValueError("Expected nonempty heldout probes")
    a = {p.group_id for p in selection_probes}
    b = {p.group_id for p in heldout_probes}
    if a & b:
        raise ValueError("Leakage: selection and heldout group overlap")
    combined = tuple(selection_probes) + tuple(heldout_probes)
    if len({p.probe_id for p in combined}) != len(combined):
        raise ValueError("Selection and heldout share probe IDs")
    chosen = set(selected_evidence_ids)
    if not chosen.issubset({e.evidence_id for e in evidence}):
        raise ValueError("Unknown selected evidence ID")
    selected = [e for e in evidence if e.evidence_id in chosen]
    return {
        "n_heldout_groups": len(b),
        "full": evaluate_memory(evidence, heldout_probes, predictor, costs),
        "selected": evaluate_memory(selected, heldout_probes, predictor, costs),
    }
