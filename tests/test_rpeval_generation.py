from frontiermem.external_data import adapt_rpeval_generation_pool


def test_generation_pool_adapter_preserves_native_three_way_labels():
    rows = [
        {
            "question": "Q1",
            "ignore": ["p-ignore-1", "p-ignore-2"],
            "supportive": ["p-support"],
            "dominant": ["p-dominate"],
        }
    ]
    out = adapt_rpeval_generation_pool(rows)
    assert len(out) == 4
    assert [x.source_label for x in out] == [
        "IGNORE", "IGNORE", "SUPPORT", "DOMINATE"
    ]
    assert [x.action for x in out] == [
        "IGNORE", "IGNORE", "APPLY", "APPLY"
    ]
    assert len({x.group_id for x in out}) == 1
    assert all(x.usage == "train_pool" for x in out)
