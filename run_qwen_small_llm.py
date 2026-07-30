from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from frontiermem.data import generate_frontier_suite, save_frontier_suite
from frontiermem.evaluation import evaluate_predictions, extraction_metrics
from frontiermem.hf_small_llm import HFGenerationConfig, HFSmallLLMFrontierMem


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-examples", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu"],
        default="auto",
        help="Use cpu to avoid CUDA initialization when the local NVIDIA driver is too old.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa"],
        default="eager",
        help="eager avoids sliding-window SDPA warnings for this small demo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = ROOT / "data" / "frontier_suite_mini.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        df = generate_frontier_suite()
        save_frontier_suite(df, data_path)

    test_df = df[df["split"] == "test"].reset_index(drop=True)
    subset = (
        test_df.groupby(["family", "action"], group_keys=False)
        .head(max(1, args.max_examples // (test_df["family"].nunique() * 3)))
        .head(args.max_examples)
        .reset_index(drop=True)
    )

    backend = HFSmallLLMFrontierMem(
        HFGenerationConfig(
            model_name=args.model,
            temperature=args.temperature,
            attn_implementation=args.attn_implementation,
        ),
        device_map=args.device,
    )
    pred = backend.predict_dataframe(subset)
    metrics = pd.DataFrame([evaluate_predictions(pred, backend.name)])
    extraction = extraction_metrics(pred)

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    pred.to_csv(results / "test_predictions_qwen_small.csv", index=False)
    metrics.to_csv(results / "metrics_qwen_small.csv", index=False)
    extraction.to_csv(results / "qwen_small_extraction_metrics.csv", index=False)

    print("\nDecision metrics")
    print(metrics.round(4).to_string(index=False))
    print("\nMemory extraction metrics")
    print(extraction.round(4).to_string(index=False))
    print("\nParser/fallback usage")
    print(pred[["memory_parse_mode", "decision_parse_mode"]].value_counts().to_string())
    print(f"\nSaved to {results}")


if __name__ == "__main__":
    main()
