#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontiermem.caid_baseline import ApplicabilityClassifier
from frontiermem.contrast_sets import load_pairs
from frontiermem.external_data import load_jsonl
from frontiermem.identifiability import selective_metrics


def group_train_calibration_split(rows: list[dict], calibration_size: float, seed: int):
    groups = np.asarray([r.get("preference_key") or r.get("group_id") for r in rows])
    idx = np.arange(len(rows))
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=calibration_size, random_state=seed
    )
    train_idx, cal_idx = next(splitter.split(idx, groups=groups))
    return [rows[i] for i in train_idx], [rows[i] for i in cal_idx]


def binary_metrics(gold: list[str], pred: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, labels=["APPLY", "IGNORE"], average="macro")),
        "n": len(gold),
    }


def evaluate_contrast_pairs(
    model: ApplicabilityClassifier,
    pairs: list[dict],
) -> dict[str, float]:
    if not pairs:
        return {"n_pairs": 0, "direction_accuracy": float("nan"), "mean_margin": float("nan")}

    apply_rows, ignore_rows = [], []
    for p in pairs:
        apply_rows.append(
            {
                "preference": p.get("positive_preference", p.get("preference", "")),
                "query": p["positive_query"],
                "history": p.get("positive_history", ""),
                "context": p.get("positive_context", ""),
            }
        )
        ignore_rows.append(
            {
                "preference": p.get("negative_preference", p.get("preference", "")),
                "query": p["negative_query"],
                "history": p.get("negative_history", ""),
                "context": p.get("negative_context", ""),
            }
        )
    pa = model.predict_proba_apply(apply_rows)
    pi = model.predict_proba_apply(ignore_rows)
    margin = pa - pi
    return {
        "n_pairs": len(pairs),
        "direction_accuracy": float((margin > 0).mean()),
        "mean_margin": float(margin.mean()),
        "median_margin": float(np.median(margin)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a leakage-aware pointwise + conformal CAID-v0 baseline."
    )
    parser.add_argument(
        "--examples",
        default="data/external_processed/all_applicability_examples.jsonl",
    )
    parser.add_argument(
        "--flip-pairs",
        default="data/external_processed/context_decision_flip_pairs.jsonl",
    )
    parser.add_argument("--output-dir", default="results/external_caid_v0")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--calibration-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--mode",
        choices=["full", "preference_query", "query_only"],
        default="full",
    )
    parser.add_argument(
        "--allow-eval-training",
        action="store_true",
        help="Emergency/debug only. Never use this for reported benchmark results.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.examples)
    binary_rows = [
        r for r in rows if str(r.get("action", "")).upper() in {"APPLY", "IGNORE"}
    ]
    train_rows_fixed = [r for r in binary_rows if r.get("usage") == "train"]
    calibration_rows_fixed = [r for r in binary_rows if r.get("usage") == "calibration"]
    train_pool = [r for r in binary_rows if r.get("usage") == "train_pool"]

    if not train_rows_fixed and not train_pool and not args.allow_eval_training:
        raise SystemExit(
            "No train_pool examples found. For valid experiments, obtain the "
            "RPEval data_generation pool and rerun preparation. Do NOT train on "
            "BenchPreS or RPEval benchmark test data."
        )
    if train_rows_fixed and calibration_rows_fixed:
        train_rows, calibration_rows = train_rows_fixed, calibration_rows_fixed
    elif train_pool:
        train_rows, calibration_rows = group_train_calibration_split(
            train_pool, args.calibration_size, args.seed
        )
    else:
        train_pool = [r for r in binary_rows if r.get("usage") != "eval_only"]
        print("[WARN] --allow-eval-training enabled; results are debug-only.")
        train_rows, calibration_rows = group_train_calibration_split(
            train_pool, args.calibration_size, args.seed
        )
    model = ApplicabilityClassifier(mode=args.mode).fit(train_rows)
    model.fit_conformal(calibration_rows, alpha=args.alpha)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source_groups: dict[str, list[dict]] = defaultdict(list)
    for row in binary_rows:
        if row.get("usage") in {"eval", "eval_only"}:
            source_groups[row["source"]].append(row)

    metric_rows = []
    prediction_rows = []
    for source, eval_rows in sorted(source_groups.items()):
        gold = [r["action"] for r in eval_rows]
        p_apply = model.predict_proba_apply(eval_rows)
        forced = model.predict_binary(eval_rows)
        selective = model.predict_selective(eval_rows)

        forced_m = binary_metrics(gold, forced)
        select_m = selective_metrics(gold, selective)
        record = {
            "source": source,
            "mode": args.mode,
            "alpha": args.alpha,
            **{f"forced_{k}": v for k, v in forced_m.items()},
            **{f"selective_{k}": v for k, v in select_m.items()},
        }
        metric_rows.append(record)

        for row, prob, f_pred, s_pred in zip(eval_rows, p_apply, forced, selective):
            prediction_rows.append(
                {
                    "example_id": row["example_id"],
                    "source": source,
                    "gold": row["action"],
                    "p_apply": float(prob),
                    "forced_prediction": f_pred,
                    "selective_prediction": s_pred,
                    "preference": row["preference"],
                    "query": row["query"],
                }
            )

    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out / "predictions.csv", index=False)

    flip_path = Path(args.flip_pairs)
    flip_pairs = load_pairs(flip_path) if flip_path.exists() else []
    eval_pairs = [p for p in flip_pairs if p.get("usage") != "train_pool"]
    pair_metrics = evaluate_contrast_pairs(model, eval_pairs)
    (out / "contrast_metrics.json").write_text(
        json.dumps(pair_metrics, indent=2), encoding="utf-8"
    )

    with (out / "model.pkl").open("wb") as f:
        pickle.dump(model, f)

    run_info = {
        "n_train": len(train_rows),
        "n_calibration": len(calibration_rows),
        "alpha": args.alpha,
        "mode": args.mode,
        "qhat": model.conformal.qhat if model.conformal else None,
        "sources_evaluated": {k: len(v) for k, v in source_groups.items()},
        "contrast_metrics": pair_metrics,
    }
    (out / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )

    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nContrast metrics:")
    print(json.dumps(pair_metrics, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
