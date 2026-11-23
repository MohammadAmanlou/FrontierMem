"""Tests for the model-independent decision-memory selection scaffold."""
import pytest
from frontiermem.decision_memory import (
    Evidence, DecisionProbe, evaluate_memory, greedy_backward_select, heldout_comparison,
)

def evidence():
    return [
        Evidence("pref", "style", "prefers short explanations", 6, "hist:1"),
        Evidence("scope", "style", "only when debugging", 4, "hist:2"),
        Evidence("redundant", "style", "thanks for the short explanation", 7, "hist:3"),
    ]

def probes(split="selection", prefix="s"):
    return [
        DecisionProbe(prefix+"1", prefix+"1", "style", "debug", "debug this", "APPLY", split, "annotated:1", "pair", "human_validated"),
        DecisionProbe(prefix+"2", prefix+"2", "style", "exam", "explain theorem", "IGNORE", split, "annotated:2", "pair", "human_validated"),
    ]

def predict(items, probe):
    ids = {e.evidence_id for e in items}
    if "pref" not in ids:
        return "IGNORE"
    if probe.context == "exam" and "scope" in ids:
        return "IGNORE"
    return "APPLY"

def test_preserves_critical_scope():
    result = greedy_backward_select(evidence(), probes(), predict, 10)
    assert result["selected_evidence_ids"] == ["pref", "scope"]
    assert result["selected_tokens"] == 10
    assert result["removal_audit"][0]["removed_evidence_id"] == "redundant"
    assert result["selected_selection_metrics"]["strict_flip_accuracy"] == 1

def test_heldout_scoring_and_strict_flip():
    result = greedy_backward_select(evidence(), probes(), predict, 10)
    scores = heldout_comparison(evidence(), result["selected_evidence_ids"], probes(), probes("heldout","h"), predict)
    assert scores["selected"]["wrong_apply_rate"] == 0
    assert scores["selected"]["strict_flip_accuracy"] == 1

def test_rejects_group_leakage():
    with pytest.raises(ValueError, match="overlap"):
        heldout_comparison(evidence(), ["pref"], probes(), probes("heldout","s"), predict)

def test_blocks_eval_only_for_selection():
    with pytest.raises(ValueError, match="Selection prohibits"):
        DecisionProbe("x","g","p","c","q","APPLY","selection","benchpres:test",source_usage="eval_only")
    with pytest.raises(ValueError, match="selection"):
        greedy_backward_select(evidence(), probes("heldout"), predict, 10)

def test_rejects_invalid_source_and_label():
    with pytest.raises(ValueError, match="provenance"):
        Evidence("id","p","text",1,"")
    with pytest.raises(ValueError, match="validated"):
        DecisionProbe("x","g","p","c","q","OTHER","selection","source",source_usage="train")

def test_wrong_apply_rate_and_asymmetric_loss():
    scores = evaluate_memory(evidence()[:1], probes(), predict)
    assert scores["wrong_apply_rate"] == 1
    assert scores["apply_recall"] == 1
    assert scores["mean_weighted_loss"] == 1.5

def test_zero_budget_and_duplicate_ids():
    scores = greedy_backward_select(evidence(), probes(), predict, 0)
    assert scores["selected_evidence_ids"] == []
    with pytest.raises(ValueError, match="unique IDs"):
        greedy_backward_select([evidence()[0], evidence()[0]], probes(), predict, 0)
