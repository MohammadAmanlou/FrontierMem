from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


LABELS = ("APPLY", "IGNORE", "CLARIFY")


def _zone_accuracy(df: pd.DataFrame, zone: str) -> float:
    part = df[df["zone"] == zone]
    if part.empty:
        return float("nan")
    return float(accuracy_score(part["action"], part["prediction"]))


def twin_metrics(df: pd.DataFrame) -> tuple[float, float, int]:
    exact_scores: list[float] = []
    discrimination_scores: list[float] = []
    differing_gold_pairs = 0
    for _, group in df.groupby("twin_pair_id", sort=False):
        if len(group) != 2:
            continue
        exact_scores.append(float((group["prediction"].values == group["action"].values).all()))
        if group["action"].nunique() > 1:
            differing_gold_pairs += 1
            discrimination_scores.append(float(group["prediction"].nunique() > 1))
    exact = float(np.mean(exact_scores)) if exact_scores else float("nan")
    discrimination = (
        float(np.mean(discrimination_scores)) if discrimination_scores else float("nan")
    )
    return exact, discrimination, differing_gold_pairs


def evaluate_predictions(df: pd.DataFrame, model_name: str) -> dict:
    required = {"action", "prediction", "zone", "twin_pair_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    y_true = df["action"].astype(str)
    y_pred = df["prediction"].astype(str)
    outside = df["zone"].eq("outside")
    non_near = ~df["zone"].eq("near")
    twin_exact, twin_discrimination, differing_pairs = twin_metrics(df)

    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)),
        "inside_accuracy": _zone_accuracy(df, "inside"),
        "outside_accuracy": _zone_accuracy(df, "outside"),
        "near_accuracy": _zone_accuracy(df, "near"),
        "overgeneralization_rate": (
            float((y_pred[outside] == "APPLY").mean()) if outside.any() else float("nan")
        ),
        "unnecessary_clarification_rate": (
            float((y_pred[non_near] == "CLARIFY").mean()) if non_near.any() else float("nan")
        ),
        "twin_exact_match": twin_exact,
        "twin_discrimination_accuracy": twin_discrimination,
        "different_gold_twin_pairs": differing_pairs,
        "n_examples": int(len(df)),
    }


def extraction_metrics(df: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "family_correct": (df["pred_family"] == df["family"]).mean(),
        "profile_variant_correct": (
            df["pred_profile_variant"] == df["profile_variant"]
        ).mean(),
        "owner_correct": (df["pred_owner"] == df["owner"]).mean(),
        "information_type_correct": (
            df["pred_information_type"] == df["information_type"]
        ).mean(),
        "temporal_correct": (
            df["pred_temporal_validity"] == df["temporal_validity"]
        ).mean(),
    }
    return pd.DataFrame(
        [{"metric": name, "score": float(score)} for name, score in columns.items()]
    )


__all__ = ["LABELS", "evaluate_predictions", "extraction_metrics", "twin_metrics"]
