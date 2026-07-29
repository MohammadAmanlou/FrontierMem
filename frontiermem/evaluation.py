from __future__ import annotations

from typing import Dict, Iterable
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

ACTIONS = ["APPLY", "CLARIFY", "IGNORE"]


def evaluate_predictions(df: pd.DataFrame, pred_col: str = "prediction") -> Dict[str, float]:
    y_true = df["action"].to_numpy()
    y_pred = df[pred_col].to_numpy()
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=ACTIONS, average="macro", zero_division=0),
    }
    for zone in ["inside", "outside", "near"]:
        mask = df["zone"].eq(zone)
        metrics[f"{zone}_accuracy"] = accuracy_score(y_true[mask], y_pred[mask]) if mask.any() else np.nan
    outside = df["zone"].eq("outside")
    non_near = ~df["zone"].eq("near")
    metrics["overgeneralization_rate"] = float(np.mean(y_pred[outside] == "APPLY")) if outside.any() else np.nan
    metrics["unnecessary_clarification_rate"] = float(np.mean(y_pred[non_near] == "CLARIFY")) if non_near.any() else np.nan
    return metrics


def evaluate_counterfactual_twins(df: pd.DataFrame, pred_col: str = "prediction") -> Dict[str, float]:
    correct_groups = []
    discriminative_groups = []
    for _, group in df.groupby("twin_pair_id"):
        if len(group) != 2:
            continue
        gt = group["action"].tolist()
        pred = group[pred_col].tolist()
        if gt[0] != gt[1]:
            discriminative_groups.append(int(pred == gt))
        correct_groups.append(int(pred == gt))
    return {
        "twin_exact_match": float(np.mean(correct_groups)) if correct_groups else np.nan,
        "twin_discrimination_accuracy": float(np.mean(discriminative_groups)) if discriminative_groups else np.nan,
        "num_twin_pairs": len(correct_groups),
        "num_discriminative_pairs": len(discriminative_groups),
    }


def confusion_dataframe(df: pd.DataFrame, pred_col: str = "prediction") -> pd.DataFrame:
    cm = confusion_matrix(df["action"], df[pred_col], labels=ACTIONS)
    return pd.DataFrame(cm, index=[f"true_{a}" for a in ACTIONS], columns=[f"pred_{a}" for a in ACTIONS])
