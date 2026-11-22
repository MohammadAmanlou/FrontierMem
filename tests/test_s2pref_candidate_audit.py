import json

import pytest

from scripts.audit_s2pref_candidates import validate_candidate_splits, annotation_sample


def pair(split: str, suffix: str, sha: str):
    return {
        "pair_id": "pair-" + suffix, "source": "s2pref",
        "group_id": "s2pref_source_" + sha[:20],
        "source_original_sha": sha, "split": split,
        "pair_type": "same_preference_same_query_opposite_context",
        "label_status": "S2Pref_derived_silver_unreviewed",
        "usage": split + "_candidate",
        "positive": {
            "example_id": "ex-pos-" + suffix, "preference": "Prefer simple meals",
            "history": "", "context": "At home " + suffix,
            "query": "What should I do?", "action": "APPLY",
        },
        "negative": {
            "example_id": "ex-neg-" + suffix, "preference": "Prefer simple meals",
            "history": "", "context": "At work " + suffix,
            "query": "What should I do?", "action": "IGNORE",
        },
    }


def save(directory, train, dev):
    directory.mkdir(parents=True, exist_ok=True)
    for split, items in (("train", train), ("dev", dev)):
        (directory / ("s2pref_context_" + split + "_CANDIDATES.jsonl")).write_text(
            "".join(json.dumps(x) + "\n" for x in items), encoding="utf8"
        )


def test_valid_review_sample_and_summary(tmp_path):
    save(tmp_path, [pair("train", "t1", "a" * 64)],
         [pair("dev", "d1", "b" * 64)])
    summary, pairs = validate_candidate_splits(tmp_path)
    assert summary["status"].startswith("PASS_STRUCTURAL")
    assert summary["splits"]["train"]["pairs"] == 1
    assert summary["train_dev_group_overlap"] == 0
    sample = annotation_sample(pairs, 4, 17)
    assert len(sample) == 2
    assert all(not x["reviewer_apply_correct"] for x in sample)


def test_fails_if_source_is_shared(tmp_path):
    save(tmp_path, [pair("train", "t1", "a" * 64)],
         [pair("dev", "d1", "a" * 64)])
    with pytest.raises(ValueError, match="leaks across splits"):
        validate_candidate_splits(tmp_path)


def test_fails_if_silver_is_relabelled_gold(tmp_path):
    entry = pair("train", "t1", "a" * 64)
    entry["label_status"] = "gold"
    save(tmp_path, [entry], [pair("dev", "d1", "b" * 64)])
    with pytest.raises(ValueError, match="unreviewed silver"):
        validate_candidate_splits(tmp_path)


def test_fails_if_split_mismatch(tmp_path):
    entry = pair("train", "t1", "a" * 64)
    entry["split"] = "test"
    save(tmp_path, [entry], [pair("dev", "d1", "b" * 64)])
    with pytest.raises(ValueError, match="source/split mismatch"):
        validate_candidate_splits(tmp_path)


def test_fails_if_pair_does_not_flip(tmp_path):
    entry = pair("train", "t1", "a" * 64)
    entry["negative"]["context"] = entry["positive"]["context"]
    save(tmp_path, [entry], [pair("dev", "d1", "b" * 64)])
    with pytest.raises(ValueError, match="unchanged context"):
        validate_candidate_splits(tmp_path)


def test_fails_duplicate_across_splits(tmp_path):
    entry = pair("train", "t1", "a" * 64)
    duplicate = pair("dev", "t1", "b" * 64)
    save(tmp_path, [entry], [duplicate])
    with pytest.raises(ValueError, match="duplicate pair_id"):
        validate_candidate_splits(tmp_path)
