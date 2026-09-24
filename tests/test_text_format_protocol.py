from frontiermem.caid_baseline import format_example


def test_full_equals_preference_query_when_optional_fields_empty():
    row = {
        "preference": "P",
        "history": "",
        "context": "",
        "query": "Q",
    }
    assert format_example(row, "full") == format_example(row, "preference_query")


def test_full_includes_only_nonempty_optional_fields():
    row = {
        "preference": "P",
        "history": "H",
        "context": "",
        "query": "Q",
    }
    text = format_example(row, "full")
    assert "[HISTORY]" in text
    assert "[CONTEXT]" not in text
