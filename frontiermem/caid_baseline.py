from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from .identifiability import SplitConformalAbstainer


def format_example(row: dict, mode: str = "full") -> str:
    pref = str(row.get("preference", ""))
    query = str(row.get("query", ""))
    history = str(row.get("history", ""))
    context = str(row.get("context", ""))

    if mode == "query_only":
        return f"[QUERY]\n{query}"
    if mode == "preference_query":
        return f"[PREFERENCE]\n{pref}\n[QUERY]\n{query}"
    if mode == "full":
        return (
            f"[PREFERENCE]\n{pref}\n"
            f"[HISTORY]\n{history}\n"
            f"[CONTEXT]\n{context}\n"
            f"[QUERY]\n{query}"
        )
    raise ValueError(f"Unknown text mode: {mode}")


@dataclass
class ApplicabilityClassifier:
    mode: str = "full"
    c: float = 2.0
    max_features_word: int = 80000
    max_features_char: int = 120000

    def __post_init__(self) -> None:
        # Word features help English; character n-grams are essential for the
        # released Chinese RPEval data and cross-script robustness.
        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=self.max_features_word,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(2, 5),
                        min_df=2,
                        max_features=self.max_features_char,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        self.pipeline = Pipeline(
            [
                ("features", features),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.c,
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=17,
                    ),
                ),
            ]
        )
        self.conformal: SplitConformalAbstainer | None = None

    @staticmethod
    def _label(action: str) -> int:
        action = str(action).upper()
        if action == "APPLY":
            return 1
        if action == "IGNORE":
            return 0
        raise ValueError("Pointwise applicability fitting accepts APPLY/IGNORE only")

    def fit(self, rows: Sequence[dict]) -> "ApplicabilityClassifier":
        usable = [r for r in rows if str(r.get("action", "")).upper() in {"APPLY", "IGNORE"}]
        if not usable:
            raise ValueError("No APPLY/IGNORE rows available for fitting")
        x = [format_example(r, self.mode) for r in usable]
        y = [self._label(r["action"]) for r in usable]
        self.pipeline.fit(x, y)
        return self

    def predict_proba_apply(self, rows: Sequence[dict]) -> np.ndarray:
        x = [format_example(r, self.mode) for r in rows]
        proba = self.pipeline.predict_proba(x)
        classes = list(self.pipeline.named_steps["classifier"].classes_)
        apply_col = classes.index(1)
        return proba[:, apply_col]

    def fit_conformal(self, calibration_rows: Sequence[dict], alpha: float = 0.10) -> "ApplicabilityClassifier":
        usable = [
            r for r in calibration_rows
            if str(r.get("action", "")).upper() in {"APPLY", "IGNORE"}
        ]
        probs = self.predict_proba_apply(usable)
        labels = [self._label(r["action"]) for r in usable]
        self.conformal = SplitConformalAbstainer(alpha=alpha).fit(probs, labels)
        return self

    def predict_binary(self, rows: Sequence[dict], threshold: float = 0.5) -> list[str]:
        probs = self.predict_proba_apply(rows)
        return ["APPLY" if p >= threshold else "IGNORE" for p in probs]

    def predict_selective(self, rows: Sequence[dict]) -> list[str]:
        if self.conformal is None:
            raise RuntimeError("Call fit_conformal() before predict_selective()")
        return self.conformal.predict(self.predict_proba_apply(rows))


__all__ = ["ApplicabilityClassifier", "format_example"]
