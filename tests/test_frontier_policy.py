from frontiermem.data import generate_frontier_suite
from frontiermem.memory import FrontierPolicy


def test_scoped_inside_outside_near_decisions():
    df = generate_frontier_suite(groups_per_family=1, seed=7)
    family = "travel_budget"
    scoped = df[(df["family"] == family) & (df["profile_variant"] == "scoped")]
    policy = FrontierPolicy()
    predictions = {
        row["zone"]: policy.predict_one(row["history"], row["query"])["prediction"]
        for _, row in scoped.iterrows()
    }
    assert predictions["inside"] == "APPLY"
    assert predictions["outside"] == "IGNORE"
    assert predictions["near"] == "CLARIFY"


def test_stable_preferences_are_applied():
    df = generate_frontier_suite(groups_per_family=1, seed=11)
    stable = df[df["profile_variant"] == "stable"].head(20)
    policy = FrontierPolicy()
    preds = policy.predict(stable["history"].tolist(), stable["query"].tolist())
    assert set(preds) == {"APPLY"}
