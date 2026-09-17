from frontiermem.contrast_sets import build_natural_contrast_pairs
from frontiermem.external_data import adapt_benchpres_rows, adapt_rpeval_records


def test_benchpres_explodes_atomic_preferences_and_is_eval_only():
    rows = [
        {
            "prompt": "Write a message to an admissions committee.",
            "name": "Nova",
            "task": "admissions email",
            "recipient": "committee",
            "preference_attribute": ["Use emojis.", "Focus on facts."],
            "preference_label": [False, True],
        }
    ]
    out = adapt_benchpres_rows(rows)
    assert [x.action for x in out] == ["IGNORE", "APPLY"]
    assert all(x.usage == "eval_only" for x in out)


def test_rpeval_multi_label_mapping():
    rows = [
        {
            "question": "Q",
            "persona": ["p1", "p2", "p3"],
            "intent_type": "ABC",
            "reason": ["r1", "r2", "r3"],
        }
    ]
    out = adapt_rpeval_records(
        rows, source_split="benchmark_explicit_multi", usage="eval", implicit=False
    )
    assert [x.action for x in out] == ["IGNORE", "APPLY", "APPLY"]


def test_natural_pairs_reuse_existing_text_only():
    records = [
        {
            "example_id": "a",
            "source": "benchpres",
            "usage": "eval_only",
            "preference": "Use emojis.",
            "query": "Text a friend.",
            "history": "",
            "context": "friend",
            "action": "APPLY",
        },
        {
            "example_id": "b",
            "source": "benchpres",
            "usage": "eval_only",
            "preference": "Use emojis.",
            "query": "Write to the IRS.",
            "history": "",
            "context": "IRS",
            "action": "IGNORE",
        },
    ]
    pairs = build_natural_contrast_pairs(records)
    assert len(pairs) == 1
    assert pairs[0]["positive_query"] == "Text a friend."
    assert pairs[0]["negative_query"] == "Write to the IRS."
