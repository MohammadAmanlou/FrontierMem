from pathlib import Path

import pandas as pd

from frontiermem.data import generate_frontier_suite
from frontiermem.memory import RuleBasedFrontierMem
from frontiermem.tiny_lm import TinyLLMFrontierMem


def test_dataset_shape_and_split():
    df = generate_frontier_suite()
    assert len(df) == 4608
    assert set(df["split"]) == {"train", "test"}
    train_groups = set(df.loc[df["split"] == "train", "group_id"])
    test_groups = set(df.loc[df["split"] == "test", "group_id"])
    assert train_groups.isdisjoint(test_groups)
    assert set(df["action"]) == {"APPLY", "IGNORE", "CLARIFY"}


def test_rule_pipeline_returns_valid_action():
    df = generate_frontier_suite(test_groups_per_family=1, train_groups_per_family=1)
    row = df.iloc[0]
    output = RuleBasedFrontierMem().predict_one(row.history, row.query)
    assert output["action"] in {"APPLY", "IGNORE", "CLARIFY"}
    assert output["memory"]["family"]


def test_checkpoint_loads_if_present():
    checkpoint = Path(__file__).parents[1] / "results" / "checkpoints" / "tiny_multitask_lm.pt"
    if not checkpoint.exists():
        return
    model = TinyLLMFrontierMem.load(checkpoint, device="cpu")
    df = generate_frontier_suite(test_groups_per_family=1, train_groups_per_family=1)
    pred = model.predict_dataframe(df.head(2), generate_responses_for=1)
    assert len(pred) == 2
    assert set(pred["prediction"]).issubset({"APPLY", "IGNORE", "CLARIFY"})
