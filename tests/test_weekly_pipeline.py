from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_weekly_results import collect, render_markdown
from scripts.eval_weekly_heldout import f1_macro, normalize_eval_text, measure_part


def test_binary_f1_and_duplicate_prompt_elimination():
    assert f1_macro(["APPLY", "IGNORE"], ["APPLY", "IGNORE"],
                    ("IGNORE", "APPLY")) == 1.0
    example = {"preference": "I like tea", "query": "Write a letter",
               "history": "Write a letter", "context": "Recipient: a colleague"}
    txt = normalize_eval_text(example)
    assert txt.count("Write a letter") == 1
    assert "[HISTORY]" not in txt
    assert "[CONTEXT]" in txt


def test_within_group_direction_and_wrong_apply_rate():
    rows = [
        {"example_id": "a", "group_id": "g1", "action": "APPLY"},
        {"example_id": "b", "group_id": "g1", "action": "IGNORE"},
    ]
    predictions = {
        "a": {"prediction_binary": "APPLY", "score_apply": 2.0},
        "b": {"prediction_binary": "APPLY", "score_apply": 1.0},
    }
    result = measure_part(rows, predictions, "benchpres", "benchpres")
    assert result["within_query_contrast_n"] == 1
    assert result["within_query_all_pairs_direction_accuracy"] == 1.0
    assert result["false_apply_rate_on_ignore"] == 1.0
    assert result["full_query_binary_exact_match"] == 0.0
    assert result["evaluation_is_decision_level_proxy"]


def test_partial_report_never_fabricates_missing_experiments(tmp_path: Path):
    rows = collect(tmp_path)
    assert len(rows) == 12
    assert all(r["train_status"] == "missing" for r in rows)
    md = render_markdown(rows)
    assert "0/12" in md
    assert "derived silver" in md
    assert "—" in md


def test_completed_dev_summary_is_read_without_heldout(tmp_path: Path):
    path=tmp_path/"seed_17"/"caid"
    path.mkdir(parents=True)
    doc={
        "arm":"caid","seed":17,"best_epoch":1,
        "history":[{
            "epoch":1,"rpeval_batches":5694,"seconds":123.0,
            "partial_training":False,"partial_evaluation":False,
            "rpeval_dev":{"binary_macro_f1":0.61,
                          "native_macro_f1":0.53,
                          "pair_direction_accuracy":0.72},
            "s2pref_silver_dev":{"strict_unseen_binary_macro_f1":0.58,
                                  "strict_unseen_pair_direction_accuracy":0.70},
        }]
    }
    (path/"training_summary.json").write_text(json.dumps(doc))
    rows=collect(tmp_path)
    hit=next(r for r in rows if r["seed"]==17 and r["arm"]=="caid")
    assert hit["train_status"] == "completed"
    assert hit["rpeval_dev_binary_macro_f1"] == 0.61
    assert hit["official_eval_status"] == "missing"
    assert hit.get("benchpres_binary_macro_f1") is None
