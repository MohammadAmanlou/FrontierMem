from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from frontiermem.baselines import (
    FlatFactMemoryBaseline,
    make_long_context_baseline,
    make_query_only_baseline,
)
from frontiermem.data import generate_frontier_suite, save_frontier_suite
from frontiermem.evaluation import evaluate_predictions, extraction_metrics
from frontiermem.memory import RuleBasedFrontierMem
from frontiermem.visualization import plot_boundary_metrics, plot_error_rates, plot_main_metrics


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = generate_frontier_suite()
    save_frontier_suite(df, DATA_DIR / "frontier_suite_mini.csv")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    metrics: list[dict] = []
    predictions: list[pd.DataFrame] = []

    flat = FlatFactMemoryBaseline()
    flat_df = test_df.copy()
    flat_df["prediction"] = flat.predict(flat_df["query"])
    metrics.append(evaluate_predictions(flat_df, flat.name))
    flat_df["model"] = flat.name
    predictions.append(flat_df)

    query_only = make_query_only_baseline().fit(train_df)
    q_df = test_df.copy()
    q_df["prediction"] = query_only.predict(q_df)
    metrics.append(evaluate_predictions(q_df, query_only.name))
    q_df["model"] = query_only.name
    predictions.append(q_df)

    long_context = make_long_context_baseline().fit(train_df)
    lc_df = test_df.copy()
    lc_df["prediction"] = long_context.predict(lc_df)
    metrics.append(evaluate_predictions(lc_df, long_context.name))
    lc_df["model"] = long_context.name
    predictions.append(lc_df)

    frontier = RuleBasedFrontierMem()
    rb_df = test_df.copy()
    outputs = frontier.predict(rb_df["history"], rb_df["query"])
    rb_df["prediction"] = [x["action"] for x in outputs]
    rb_df["pred_relation"] = [x["relation"] for x in outputs]
    rb_df["decision_confidence"] = [x["decision_confidence"] for x in outputs]
    rb_df["structured_memory"] = [json.dumps(x["memory"], ensure_ascii=False) for x in outputs]
    rb_df["generated_response"] = [x["response"] for x in outputs]
    rb_df["pred_family"] = [x["memory"]["family"] for x in outputs]
    rb_df["pred_profile_variant"] = [x["memory"]["profile_variant"] for x in outputs]
    rb_df["pred_owner"] = [x["memory"]["owner"] for x in outputs]
    rb_df["pred_information_type"] = [x["memory"]["information_type"] for x in outputs]
    rb_df["pred_temporal_validity"] = [x["memory"]["temporal_validity"] for x in outputs]
    metrics.append(evaluate_predictions(rb_df, frontier.name))
    rb_df["model"] = frontier.name
    predictions.append(rb_df)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(RESULTS_DIR / "metrics_rules_and_baselines.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        RESULTS_DIR / "test_predictions_rules_and_baselines.csv", index=False
    )
    extraction_metrics(rb_df).to_csv(
        RESULTS_DIR / "rule_memory_extraction_metrics.csv", index=False
    )

    plot_main_metrics(metrics_df, RESULTS_DIR / "main_metrics_rules.png")
    plot_boundary_metrics(metrics_df, RESULTS_DIR / "boundary_metrics_rules.png")
    plot_error_rates(metrics_df, RESULTS_DIR / "error_rates_rules.png")

    examples = rb_df.groupby(["family", "zone"], group_keys=False).head(1)
    examples[
        [
            "example_id",
            "family",
            "zone",
            "history",
            "query",
            "action",
            "prediction",
            "structured_memory",
            "generated_response",
        ]
    ].to_json(
        RESULTS_DIR / "example_rule_memories.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )

    print("DATASET")
    print(f"  total examples: {len(df)}")
    print(f"  train examples: {len(train_df)}")
    print(f"  test examples: {len(test_df)}")
    print(f"  scenario families: {df['family'].nunique()}")
    print("\nRULES + CLASSICAL BASELINES")
    print(metrics_df.round(4).to_string(index=False))
    print(f"\nArtifacts saved under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
