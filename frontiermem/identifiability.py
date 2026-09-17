from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class DecisionHypothesis:
    name: str
    decision: str
    weight: float = 1.0


def identify_from_hypotheses(
    hypotheses: Iterable[DecisionHypothesis],
) -> tuple[str, bool, tuple[str, ...]]:
    """Return decision, identifiability flag, and supported decision set.

    If all plausible hypotheses imply the same decision, that decision is
    identified. If plausible hypotheses disagree, the rational action is
    CLARIFY. This is the executable core of Preference Decision Identifiability.
    """
    decisions = sorted(
        {
            h.decision.upper()
            for h in hypotheses
            if h.weight > 0 and h.decision.upper() in {"APPLY", "IGNORE"}
        }
    )
    if not decisions:
        return "CLARIFY", False, tuple()
    if len(decisions) == 1:
        return decisions[0], True, tuple(decisions)
    return "CLARIFY", False, tuple(decisions)


class SplitConformalAbstainer:
    """Binary split-conformal prediction sets with CLARIFY as abstention.

    The underlying model predicts P(APPLY). Calibration labels use
    IGNORE=0/APPLY=1. At inference:
      singleton {APPLY}  -> APPLY
      singleton {IGNORE} -> IGNORE
      {APPLY, IGNORE} or empty -> CLARIFY

    This is an operational risk-control layer; it is not claimed as the
    theoretical novelty by itself.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.qhat: float | None = None

    def fit(self, p_apply: Sequence[float], labels: Sequence[int]) -> "SplitConformalAbstainer":
        probs = np.asarray(p_apply, dtype=float)
        y = np.asarray(labels, dtype=int)
        if len(probs) != len(y) or len(y) == 0:
            raise ValueError("Calibration probabilities and labels must be non-empty and aligned")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("labels must be binary: IGNORE=0, APPLY=1")

        p_true = np.where(y == 1, probs, 1.0 - probs)
        scores = 1.0 - np.clip(p_true, 0.0, 1.0)
        n = len(scores)
        quantile_level = min(1.0, ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.qhat = float(np.quantile(scores, quantile_level, method="higher"))
        return self

    def prediction_set(self, p_apply: float) -> tuple[str, ...]:
        if self.qhat is None:
            raise RuntimeError("Call fit() before prediction_set()")
        p_apply = float(np.clip(p_apply, 0.0, 1.0))
        probs = {"IGNORE": 1.0 - p_apply, "APPLY": p_apply}
        labels = tuple(
            label for label, prob in probs.items()
            if (1.0 - prob) <= self.qhat
        )
        return labels

    def predict_action(self, p_apply: float) -> str:
        labels = self.prediction_set(p_apply)
        if len(labels) == 1:
            return labels[0]
        return "CLARIFY"

    def predict(self, p_apply: Sequence[float]) -> list[str]:
        return [self.predict_action(float(p)) for p in p_apply]


def selective_metrics(
    gold_actions: Sequence[str],
    predicted_actions: Sequence[str],
) -> dict[str, float]:
    gold = np.asarray([str(x).upper() for x in gold_actions], dtype=object)
    pred = np.asarray([str(x).upper() for x in predicted_actions], dtype=object)
    if len(gold) != len(pred):
        raise ValueError("gold and predictions must be aligned")

    covered = pred != "CLARIFY"
    apply_gold = gold == "APPLY"
    ignore_gold = gold == "IGNORE"

    def safe_mean(mask: np.ndarray) -> float:
        return float(mask.mean()) if len(mask) else float("nan")

    return {
        "coverage": safe_mean(covered),
        "clarification_rate": safe_mean(~covered),
        "selective_accuracy": (
            float((gold[covered] == pred[covered]).mean())
            if covered.any()
            else float("nan")
        ),
        "overall_binary_correct_or_abstain": safe_mean(
            (gold == pred) | (pred == "CLARIFY")
        ),
        "appropriate_application_rate": (
            float((pred[apply_gold] == "APPLY").mean())
            if apply_gold.any()
            else float("nan")
        ),
        "misapplication_rate": (
            float((pred[ignore_gold] == "APPLY").mean())
            if ignore_gold.any()
            else float("nan")
        ),
        "n": float(len(gold)),
    }


__all__ = [
    "DecisionHypothesis",
    "SplitConformalAbstainer",
    "identify_from_hypotheses",
    "selective_metrics",
]
