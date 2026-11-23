import json

import pytest

from scripts.prepare_dual_axis_run import (
    prepare_plan, load_rpeval_pairs, split_rpeval_groups,
    METHODS, SUSPECT_PILOT_TRAIN_IDS,
)


def write_lines(path, rows):
    with path.open("w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def rpair(idx, group):
    return {"source":"rpeval_generation", "group_id":group, "pair_id":"r"+str(idx),
            "positive":{"native_label":"SUPPORT", "action":"APPLY"},
            "negative":{"native_label":"IGNORE", "action":"IGNORE"}}


def spair(split, idx, sha):
    return {
        "pair_id": "s"+str(idx), "source": "s2pref",
        "group_id": "s2pref_source_"+sha[:20],
        "source_original_sha": sha, "split": split,
        "usage": split+"_candidate",
        "pair_type": "same_preference_same_query_opposite_context",
        "label_status": "S2Pref_derived_silver_unreviewed",
        "positive": {"example_id":"spos"+str(idx), "preference":"Prefer peace "+str(idx),
                     "context":"Quiet group "+str(idx), "history":"",
                     "query":"What to do?", "action":"APPLY"},
        "negative": {"example_id":"sneg"+str(idx), "preference":"Prefer peace "+str(idx),
                     "context":"Noisy group "+str(idx), "history":"",
                     "query":"What to do?", "action":"IGNORE"},
    }


def fixtures(tmp_path):
    rp = tmp_path/"rpeval.jsonl"
    write_lines(rp,[rpair(i, "group"+str(i)) for i in range(10)])
    sp = tmp_path/"silver"
    sp.mkdir()
    write_lines(sp/"s2pref_context_train_CANDIDATES.jsonl",
                [spair("train", 1, "a"*64)])
    write_lines(sp/"s2pref_context_dev_CANDIDATES.jsonl",
                [spair("dev", 2, "b"*64), spair("dev",3,"c"*64)])
    return rp,sp


def test_four_arm_plan_is_group_safe_and_binary_only_silver(tmp_path):
    r,s=fixtures(tmp_path)
    plan=prepare_plan(r,s)
    assert plan["status"].startswith("CPU_PREFLIGHT")
    assert plan["rpeval"]["train_pair_count"] == 9
    assert plan["rpeval"]["dev_pair_count"] == 1
    assert plan["s2pref"]["train_pair_count"] == 1
    assert plan["s2pref"]["dev_pair_count"] == 2
    assert len(plan["methods"])==4
    assert METHODS["dual_axis"]["s2pref_context_rank"]
    assert not METHODS["caid_s2pref"]["s2pref_context_rank"]
    assert all("SUPPORT" not in json.dumps(plan["methods"][m]) for m in plan["methods"])
    assert plan["s2pref"]["train_dev_group_overlap"] == 0


def test_explicit_silver_exclusion(tmp_path):
    r,s=fixtures(tmp_path)
    plan=prepare_plan(r,s,exclude_s2pref_dev=["s2"])
    assert plan["s2pref"]["dev_pair_count"]==1
    assert plan["s2pref"]["excluded_pilot_dev_pair_ids"]==["s2"]


def test_no_false_native_silver_labels(tmp_path):
    r,s=fixtures(tmp_path)
    with pytest.raises(ValueError, match="forbidden RPEval"):
        write_lines(r,[{**rpair(1,"group1"),"source":"benchpres"}])
        prepare_plan(r,s)


def test_rejects_overlap_rpeval(tmp_path):
    p=[rpair(1,"same"),rpair(2,"same")]
    with pytest.raises(ValueError,match="groups"):
        split_rpeval_groups(p, .1, 17)


def test_rejects_duplicate_rpeval_ids(tmp_path):
    r=tmp_path/"r.jsonl"
    write_lines(r,[rpair(1,"g1"),rpair(1,"g2")])
    with pytest.raises(ValueError,match="duplicate"):
        load_rpeval_pairs(r)


def test_no_training_on_silver_dev(tmp_path):
    r,s=fixtures(tmp_path)
    plan=prepare_plan(r,s)
    assert set(plan["s2pref"]["train_pair_ids"]).isdisjoint(plan["s2pref"]["dev_pair_ids"])
    assert all(plan["methods"][a]["s2pref_train_pair_ids"] is None
               for a in ("pointwise","caid"))
