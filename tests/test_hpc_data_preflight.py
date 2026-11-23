import json
from pathlib import Path

import pytest

from scripts.hpc_data_preflight import audit_atomic, audit_training_pairs, inspect


def write_rows(path: Path, entries):
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_preflight_minimal_valid(tmp_path):
    write_rows(tmp_path / "all_applicability_examples.jsonl", [
        {"source": "rpeval_generation", "usage": "train", "action": "APPLY", "example_id": "a", "group_id": "g1"},
        {"source": "rpeval_generation", "usage": "calibration", "action": "IGNORE", "example_id": "b", "group_id": "g2"},
        {"source": "rpeval_explicit", "usage": "eval", "action": "IGNORE", "example_id": "c"},
        {"source": "rpeval_implicit", "usage": "eval", "action": "IGNORE", "example_id": "d"},
        {"source": "benchpres", "usage": "eval_only", "action": "APPLY", "example_id": "e"},
    ])
    write_rows(tmp_path / "caid_pairwise_train.jsonl", [
        {"source": "rpeval_generation", "pair_id": "p1", "group_id": "g1",
         "pair_type": "query_preference_contrast",
         "positive": {"action": "APPLY", "native_label": "SUPPORT"},
         "negative": {"action": "IGNORE", "native_label": "IGNORE"}}
    ])
    (tmp_path / "context_decision_flip_pairs.jsonl").write_text("", encoding="utf-8")
    summary = inspect(tmp_path)
    assert summary["status"] == "PASS_STRUCTURAL_CHECKS_ONLY"
    assert summary["atomic"]["n"] == 5
    assert summary["rpeval_training_pairs"]["n"] == 1
    assert summary["legacy_context_flip_records"] == 0


def test_preflight_rejects_training_test_leakage(tmp_path):
    write_rows(tmp_path / "all_applicability_examples.jsonl", [
        {"source": "benchpres", "usage": "train", "action": "APPLY"}
    ])
    with pytest.raises(ValueError, match="policy violation"):
        audit_atomic(tmp_path / "all_applicability_examples.jsonl")


def test_preflight_rejects_group_overlap(tmp_path):
    write_rows(tmp_path / "all_applicability_examples.jsonl", [
        {"source": "rpeval_generation", "usage": "train", "action": "APPLY", "group_id": "same"},
        {"source": "rpeval_generation", "usage": "calibration", "action": "IGNORE", "group_id": "same"},
    ])
    with pytest.raises(ValueError, match="spanning train/calibration"):
        audit_atomic(tmp_path / "all_applicability_examples.jsonl")


def test_preflight_rejects_eval_pairs(tmp_path):
    write_rows(tmp_path / "pairs.jsonl", [
        {"source": "benchpres", "pair_id": "bad",
         "positive": {"action": "APPLY", "native_label": "SUPPORT"},
         "negative": {"action": "IGNORE", "native_label": "IGNORE"}}
    ])
    with pytest.raises(ValueError, match="Non-training source"):
        audit_training_pairs(tmp_path / "pairs.jsonl")
