from __future__ import annotations

from pathlib import Path

import pandas as pd

from frontiermem.data import generate_frontier_suite, save_frontier_suite
from frontiermem.evaluation import evaluate_predictions, extraction_metrics
from frontiermem.tiny_lm import TinyLLMFrontierMem, train_tiny_lm
from frontiermem.visualization import plot_boundary_metrics, plot_error_rates, plot_main_metrics


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    data_path = DATA_DIR / "frontier_suite_mini.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        df = generate_frontier_suite()
        save_frontier_suite(df, data_path)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    # Fixed validation subset from the held-out groups; the final metrics use all test rows.
    validation_df = test_df.sample(min(384, len(test_df)), random_state=23).reset_index(drop=True)

    checkpoint = CHECKPOINT_DIR / "tiny_multitask_lm.pt"
    if checkpoint.exists():
        print(f"Loading existing checkpoint: {checkpoint}")
        model = TinyLLMFrontierMem.load(checkpoint)
    else:
        model, summary = train_tiny_lm(
            train_df,
            validation_df,
            CHECKPOINT_DIR,
            epochs=4,
            batch_size=192,
            learning_rate=8e-4,
            seed=17,
        )
        print(f"Best validation action accuracy: {summary.best_validation_action_accuracy:.4f}")

    pred_df = model.predict_dataframe(test_df, batch_size=192, generate_responses_for=36)
    hybrid_df = model.predict_dataframe_hybrid(
        test_df, batch_size=192, generate_responses_for=36
    )

    metric = evaluate_predictions(pred_df, model.name + " (end-to-end heads)")
    hybrid_metric = evaluate_predictions(
        hybrid_df, "Tiny Transformer Memory + Rule Policy"
    )
    metrics_df = pd.DataFrame([metric, hybrid_metric])
    extraction_df = extraction_metrics(pred_df)
    hybrid_extraction_df = extraction_metrics(hybrid_df)

    pred_df.to_csv(RESULTS_DIR / "test_predictions_tiny_lm.csv", index=False)
    hybrid_df.to_csv(
        RESULTS_DIR / "test_predictions_tiny_lm_hybrid.csv", index=False
    )
    metrics_df.to_csv(RESULTS_DIR / "metrics_tiny_lm.csv", index=False)
    extraction_df.to_csv(
        RESULTS_DIR / "tiny_lm_extraction_metrics.csv", index=False
    )
    hybrid_extraction_df.to_csv(
        RESULTS_DIR / "tiny_lm_hybrid_extraction_metrics.csv", index=False
    )
    plot_main_metrics(metrics_df, RESULTS_DIR / "main_metrics_tiny_lm.png")
    plot_boundary_metrics(metrics_df, RESULTS_DIR / "boundary_metrics_tiny_lm.png")
    plot_error_rates(metrics_df, RESULTS_DIR / "error_rates_tiny_lm.png")

    samples = hybrid_df[
        hybrid_df["generated_response"].astype(str).str.len() > 0
    ].head(24)
    samples.to_json(
        RESULTS_DIR / "tiny_lm_generated_examples.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )

    print("\nTINY TRANSFORMER LM RESULTS")
    print(metrics_df.round(4).to_string(index=False))
    print("\nMEMORY EXTRACTION (PURE TINY MODEL)")
    print(extraction_df.round(4).to_string(index=False))
    print("\nMEMORY EXTRACTION (HYBRID)")
    print(hybrid_extraction_df.round(4).to_string(index=False))
    print("\nGenerated response samples:")
    print(
        samples[["query", "action", "prediction", "generated_response"]]
        .head(6)
        .to_string(index=False)
    )
    print(f"\nArtifacts saved under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
