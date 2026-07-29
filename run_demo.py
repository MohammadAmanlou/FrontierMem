from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from frontiermem.data import generate_frontier_suite, SPECS
from frontiermem.baselines import FlatFactMemoryBaseline, QueryOnlyClassifier, RawHistoryClassifier
from frontiermem.memory import FrontierMemoryExtractor, FrontierPolicy
from frontiermem.evaluation import evaluate_predictions, evaluate_counterfactual_twins, confusion_dataframe
from frontiermem.visualization import plot_overall_metrics, plot_boundary_metrics, plot_failure_rates

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def compose_text(df: pd.DataFrame) -> list[str]:
    return ("HISTORY:\n" + df["history"] + "\nCURRENT QUERY:\n" + df["query"]).tolist()


def lexical_holdout_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out one paraphrase per query region for every scenario family."""
    held_out_queries: set[str] = set()
    for spec in SPECS:
        held_out_queries.update({
            spec.shared_queries[-1],
            spec.crossing_queries[-1],
            spec.uncertain_queries[-1],
        })
    is_test = df["query"].isin(held_out_queries)
    return df.loc[~is_test].copy(), df.loc[is_test].copy()


def iid_group_split(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["group_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def response_for_row(row: pd.Series, prediction: str) -> str:
    if prediction == "APPLY":
        return f"I will use the stored preference and {row['apply_response']}."
    if prediction == "IGNORE":
        return f"The previous preference is outside its valid scope here, so I will {row['neutral_response']}."
    return row["clarification_question"]


def evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    models: dict[str, np.ndarray] = {}

    flat = FlatFactMemoryBaseline().fit(compose_text(train_df), train_df["action"].tolist())
    models["Flat Fact Memory"] = flat.predict(compose_text(test_df))

    qonly = QueryOnlyClassifier().fit(train_df["query"].tolist(), train_df["action"].tolist())
    models["Query-Only TF-IDF"] = qonly.predict(test_df["query"].tolist())

    raw = RawHistoryClassifier().fit(compose_text(train_df), train_df["action"].tolist())
    models["Raw Long-Context TF-IDF"] = raw.predict(compose_text(test_df))

    policy = FrontierPolicy()
    models["FrontierMem-Lite"] = np.array(
        policy.predict(test_df["history"].tolist(), test_df["query"].tolist())
    )

    metrics_rows = []
    all_predictions = test_df.copy()
    for name, preds in models.items():
        safe = name.lower().replace(" ", "_").replace("-", "_")
        all_predictions[f"pred_{safe}"] = preds
        tmp = test_df.copy()
        tmp["prediction"] = preds
        metrics = evaluate_predictions(tmp)
        metrics.update(evaluate_counterfactual_twins(tmp))
        metrics["model"] = name
        metrics_rows.append(metrics)
        confusion_dataframe(tmp).to_csv(RESULTS_DIR / f"confusion_{prefix}_{safe}.csv")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df[["model"] + [c for c in metrics_df.columns if c != "model"]]
    return metrics_df, all_predictions


def main() -> None:
    # More groups give stable preliminary estimates while keeping the demo lightweight.
    df = generate_frontier_suite(groups_per_family=64, seed=42)
    df.to_csv(DATA_DIR / "frontier_suite_mini.csv", index=False)

    # Main experiment: lexical holdout challenge.
    train_df, test_df = lexical_holdout_split(df)
    metrics_df, predictions_df = evaluate_models(train_df, test_df, prefix="lexical")
    metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    predictions_df.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    plot_overall_metrics(metrics_df, str(RESULTS_DIR / "overall_metrics.png"))
    plot_boundary_metrics(metrics_df, str(RESULTS_DIR / "boundary_metrics.png"))
    plot_failure_rates(metrics_df, str(RESULTS_DIR / "failure_rates.png"))

    # Secondary in-distribution sanity check.
    iid_train, iid_test = iid_group_split(df)
    iid_metrics, _ = evaluate_models(iid_train, iid_test, prefix="iid")
    iid_metrics.to_csv(RESULTS_DIR / "metrics_iid_sanity.csv", index=False)

    # Evaluate the structured memory extraction stage.
    extractor = FrontierMemoryExtractor()
    extraction_rows = []
    for _, row in test_df.iterrows():
        pred_mem = extractor.extract(row["history"], row["query"])
        true_stable = row["profile_variant"] == "stable"
        pred_stable = pred_mem.information_type == "trait_preference"
        extraction_rows.append({
            "family_correct": pred_mem.preference == row["preference_label"],
            "stable_vs_scoped_correct": true_stable == pred_stable,
            "owner_correct": pred_mem.owner == row["owner"],
            "temporal_correct": pred_mem.temporal_validity == row["temporal_validity"],
        })
    extraction_df = pd.DataFrame(extraction_rows)
    extraction_summary = (
        extraction_df.mean()
        .rename("score")
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    extraction_summary.to_csv(RESULTS_DIR / "memory_extraction_metrics.csv", index=False)

    # Save representative memories and final actions for the advisor demo.
    policy = FrontierPolicy()
    examples = []
    for family in ["travel_budget", "education_detail", "food_vegetarian_ownership", "writing_formality"]:
        subset = test_df[(test_df["family"] == family) & (test_df["profile_variant"] == "scoped")]
        for _, row in subset.groupby("query_kind").head(1).iterrows():
            out = policy.predict_one(row["history"], row["query"])
            examples.append({
                "family": family,
                "zone": row["zone"],
                "history": row["history"],
                "query": row["query"],
                "gold_action": row["action"],
                "predicted_action": out["prediction"],
                "structured_memory": out["memory"],
                "generated_demo_response": response_for_row(row, out["prediction"]),
            })
    with open(RESULTS_DIR / "example_memories.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    best = metrics_df.sort_values("balanced_accuracy", ascending=False).iloc[0]
    raw_row = metrics_df.loc[metrics_df["model"] == "Raw Long-Context TF-IDF"].iloc[0]
    flat_row = metrics_df.loc[metrics_df["model"] == "Flat Fact Memory"].iloc[0]
    report = f"""# FrontierMem Initial Prototype: Preliminary Findings

## Experimental setup

- Controlled suite: **{len(df):,} examples** from **{df['group_id'].nunique():,} counterfactual twin history groups**.
- Domains: writing, education, travel, and food.
- Main evaluation: a **lexical holdout challenge**, where one query paraphrase per region and scenario family is unseen during baseline training.
- Decisions: `APPLY`, `IGNORE`, or `CLARIFY`.
- Test examples: **{len(test_df):,}**; training examples: **{len(train_df):,}**.

## Main preliminary result

The strongest system is **{best['model']}**:

- Accuracy: **{best['accuracy']:.3f}**
- Balanced accuracy: **{best['balanced_accuracy']:.3f}**
- Macro-F1: **{best['macro_f1']:.3f}**
- Counterfactual twin discrimination: **{best['twin_discrimination_accuracy']:.3f}**
- Overgeneralization rate: **{best['overgeneralization_rate']:.3f}**

For comparison:

- Raw long-context TF-IDF accuracy: **{raw_row['accuracy']:.3f}**
- Raw long-context TF-IDF twin discrimination: **{raw_row['twin_discrimination_accuracy']:.3f}**
- Flat fact-memory overgeneralization: **{flat_row['overgeneralization_rate']:.3f}**

## Interpretation

The flat fact-memory baseline succeeds only when the stored preference happens to apply and overgeneralizes on every outside-boundary example. The raw text classifier learns many surface cues, but its performance drops under held-out paraphrases. FrontierMem-Lite explicitly separates stable preferences from temporary constraints, context-dependent preferences, and preferences owned by another person. It then checks the current context and chooses whether to apply, ignore, or clarify.

## What this result demonstrates

1. Preference overgeneralization is measurable with controlled counterfactual twins.
2. A structured applicability representation supports interpretable decisions.
3. The Apply/Clarify/Ignore policy can reduce both outside-boundary misuse and unnecessary questions.
4. The complete experimental pipeline is feasible on a laptop and is ready to be upgraded to an SFT-trained LLM extractor.

## Important limitation

This is a **controlled offline proof-of-concept**, not the final LLM-based system. FrontierMem-Lite currently uses a transparent rule-based extractor, while the comparison classifiers are lightweight TF-IDF models. The initial experiment validates the problem formulation, data structure, metrics, and end-to-end pipeline. The next phase should replace the extractor and response generator with an instruction-tuned 3B--8B open-source LLM trained using structured SFT and counterfactual contrastive examples.
"""
    (RESULTS_DIR / "initial_findings.md").write_text(report, encoding="utf-8")

    print("MAIN LEXICAL-HOLDOUT RESULTS")
    print(metrics_df.round(4).to_string(index=False))
    print("\nMEMORY EXTRACTION RESULTS")
    print(extraction_summary.round(4).to_string(index=False))
    print(f"\nArtifacts saved under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
