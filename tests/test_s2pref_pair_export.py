import pytest
from scripts.export_s2pref_context_pairs import validate_and_pair

def pair(split="train", group="g1", pair_id="p1"):
    base = dict(pair_id=pair_id, group_id=group, preference="pref", query="query",
                history="history", source="s2pref", split=split,
                gold_status="S2Pref_derived_silver", original_sha="upstream:1")
    return [
        dict(base, example_id=pair_id+"a", context="at work", action="APPLY"),
        dict(base, example_id=pair_id+"b", context="at home", action="IGNORE"),
    ]

def test_silver_split_roles():
    rows = pair() + pair("dev","g2","p2") + pair("test","g3","p3")
    result, groups = validate_and_pair(rows)
    assert groups == {"dev":1,"test":1,"train":1}
    assert [x["usage"] for x in result] == ["train_candidate","dev_candidate","eval_only"]

def test_rejects_group_leakage():
    with pytest.raises(ValueError, match="leaked"):
        validate_and_pair(pair()+pair("test","g1","p2"))

def test_rejects_preference_mismatch():
    rows = pair()
    rows[1]["preference"] = "different"
    with pytest.raises(ValueError, match="share preference"):
        validate_and_pair(rows)

def test_rejects_same_context_and_fake_gold():
    rows = pair()
    rows[1]["context"] = "at work"
    with pytest.raises(ValueError, match="must flip"):
        validate_and_pair(rows)
    rows = pair()
    rows[0]["gold_status"] = "gold"
    with pytest.raises(ValueError, match="silver"):
        validate_and_pair(rows)
