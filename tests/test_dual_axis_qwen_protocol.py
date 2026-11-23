import json

import pytest

from scripts.prepare_dual_axis_run import prepare_plan
from frontiermem.dual_axis_protocol import load_and_verify, format_side, summary
from scripts.train_dual_axis_qwen import (
    macro_f1, parse_args, planned_silver_steps, silver_index_for_step,
)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf8")


def rpair(i):
    return {"pair_id":f"r{i}","group_id":f"rg{i}","source":"rpeval_generation",
            "positive":{"preference":"like apples","query":"food?","context":"at home",
                        "action":"APPLY","native_label":"SUPPORT"},
            "negative":{"preference":"like pears","query":"food?","context":"at work",
                        "action":"IGNORE","native_label":"IGNORE"}}


def spair(split,i,sha):
    preference = "I prefer something "+str(i)
    return {
        "pair_id":f"s{i}", "group_id":"s2pref_source_"+sha[:20],
        "source_original_sha":sha, "split":split, "source":"s2pref",
        "usage":split+"_candidate",
        "pair_type":"same_preference_same_query_opposite_context",
        "label_status":"S2Pref_derived_silver_unreviewed",
        "positive":{"example_id":f"sp{i}", "preference":preference, "query":"What now?",
                    "context":"At home "+str(i),"history":"","action":"APPLY"},
        "negative":{"example_id":f"sn{i}", "preference":preference, "query":"What now?",
                    "context":"At work "+str(i),"history":"","action":"IGNORE"},
    }


def fixtures(tmp_path):
    rp = tmp_path/"r.jsonl"
    write_jsonl(rp,[rpair(i) for i in range(10)])
    sp = tmp_path/"silver"
    sp.mkdir()
    write_jsonl(sp/"s2pref_context_train_CANDIDATES.jsonl",
                [spair("train",100,"a"*64),spair("train",101,"b"*64)])
    write_jsonl(sp/"s2pref_context_dev_CANDIDATES.jsonl",
                [spair("dev",200,"c"*64),spair("dev",201,"d"*64)])
    plan = prepare_plan(rp,sp)
    man=tmp_path/"plan.json"
    man.write_text(json.dumps(plan),encoding="utf8")
    return rp,sp,man


def test_four_arms_and_immutable_splits(tmp_path):
    rp,sp,man=fixtures(tmp_path)
    data = load_and_verify(man,rp,sp)
    assert len(data["rpeval_train"])==9
    assert len(data["rpeval_dev"])==1
    assert len(data["s2pref_train"])==2
    assert len(data["s2pref_dev"])==2
    assert len(data["s2pref_strict_dev"])==2
    assert summary(data,"dual_axis")["s2pref_context_rank_enabled"]
    assert summary(data,"caid_s2pref")["s2pref_context_rank_enabled"] is False
    assert summary(data,"pointwise")["s2pref_train_pairs"] == 0
    serialized=format_side(data["s2pref_train"][0]["positive"])
    assert "[CONTEXT]" in serialized
    assert serialized.startswith("[QUERY]")
    assert serialized.index("[QUERY]") < serialized.index("[PREFERENCE]")


def test_tampering_after_freeze_is_rejected(tmp_path):
    rp,sp,man=fixtures(tmp_path)
    rp.write_text(rp.read_text(encoding="utf8")+"\n",encoding="utf8")
    with pytest.raises(ValueError,match="RPEval input changed"):
        load_and_verify(man,rp,sp)


def test_no_fabricated_three_way_s2pref_labels(tmp_path):
    rp,sp,man=fixtures(tmp_path)
    bad=json.loads((sp/"s2pref_context_train_CANDIDATES.jsonl").read_text().splitlines()[0])
    bad["positive"]["native_label"]="SUPPORT"
    records=[bad]+[json.loads(x) for x in (sp/"s2pref_context_train_CANDIDATES.jsonl").read_text().splitlines()[1:]]
    write_jsonl(sp/"s2pref_context_train_CANDIDATES.jsonl",records)
    plan=prepare_plan(rp,sp)
    man.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match="fabricated native labels"):
        load_and_verify(man,rp,sp)


def test_silver_schedule_visits_each_pair_once():
    scheduled=[silver_index_for_step(i,10,3) for i in range(10)]
    assert [x for x in scheduled if x is not None] == [0,1,2]
    assert planned_silver_steps(10,3,1)==3
    assert planned_silver_steps(10,3,1,2)==1


def test_silver_schedule_cannot_oversample():
    with pytest.raises(ValueError,match="Silver train pairs exceed"):
        planned_silver_steps(3,4,1)


def test_macro_f1_smoke():
    assert macro_f1([0,1,2],[0,1,2],(0,1,2))==1
    assert macro_f1([0,1],[1,0],(0,1))==0


def test_cpu_dry_run_cli_does_not_require_torch():
    args=parse_args(["--arm","dual_axis","--dry-run"])
    assert args.dry_run
    assert not args.allow_silver_exploratory


def test_unseen_dev_remains_group_disjoint(tmp_path):
    rp,sp,man=fixtures(tmp_path)
    data=load_and_verify(man,rp,sp)
    assert {x["group_id"] for x in data["s2pref_train"]}.isdisjoint(
        {x["group_id"] for x in data["s2pref_strict_dev"]})
