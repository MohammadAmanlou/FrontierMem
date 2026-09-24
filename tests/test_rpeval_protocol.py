from frontiermem.external_data import adapt_rpeval_records


def test_rpeval_implicit_hides_explicit_persona_and_gold_reason():
    rows = [
        {
            "persona": "EXPLICIT_CANONICAL_PREFERENCE",
            "implicit_persona": "OBSERVED_DIALOGUE_MEMORY",
            "question": "Q",
            "intent_type": "支持性偏好",
            "reason": "GOLD_EXPLANATION",
        }
    ]
    out = adapt_rpeval_records(
        rows,
        source_split="benchmark_implicit_single",
        usage="eval",
        implicit=True,
    )
    assert len(out) == 1
    ex = out[0]
    assert ex.preference == "OBSERVED_DIALOGUE_MEMORY"
    assert ex.history == ""
    assert ex.context == ""
    assert ex.metadata["canonical_preference"] == "EXPLICIT_CANONICAL_PREFERENCE"
    assert ex.metadata["reason"] == "GOLD_EXPLANATION"
    assert ex.metadata["memory_setting"] == "implicit"


def test_rpeval_explicit_uses_explicit_persona_only():
    rows = [
        {
            "persona": "EXPLICIT_PREFERENCE",
            "question": "Q",
            "intent_type": "ignore",
            "reason": "GOLD_EXPLANATION",
        }
    ]
    out = adapt_rpeval_records(
        rows,
        source_split="benchmark_explicit_single",
        usage="eval",
        implicit=False,
    )
    ex = out[0]
    assert ex.preference == "EXPLICIT_PREFERENCE"
    assert ex.history == ""
    assert ex.context == ""
    assert ex.metadata["reason"] == "GOLD_EXPLANATION"
