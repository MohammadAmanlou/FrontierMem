from frontiermem.identifiability import (
    DecisionHypothesis,
    SplitConformalAbstainer,
    identify_from_hypotheses,
)


def test_identifiable_apply():
    action, identified, decisions = identify_from_hypotheses(
        [
            DecisionHypothesis("stable", "APPLY"),
            DecisionHypothesis("another-compatible-world", "APPLY"),
        ]
    )
    assert action == "APPLY"
    assert identified
    assert decisions == ("APPLY",)


def test_nonidentifiable_clarifies():
    action, identified, decisions = identify_from_hypotheses(
        [
            DecisionHypothesis("stable-preference", "APPLY"),
            DecisionHypothesis("temporary-constraint", "IGNORE"),
        ]
    )
    assert action == "CLARIFY"
    assert not identified
    assert set(decisions) == {"APPLY", "IGNORE"}


def test_conformal_abstainer_runs():
    abstainer = SplitConformalAbstainer(alpha=0.2).fit(
        [0.95, 0.90, 0.10, 0.05, 0.80, 0.20],
        [1, 1, 0, 0, 1, 0],
    )
    assert abstainer.predict_action(0.99) in {"APPLY", "CLARIFY"}
    assert abstainer.predict_action(0.01) in {"IGNORE", "CLARIFY"}
