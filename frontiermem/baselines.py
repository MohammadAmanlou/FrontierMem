from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


LABELS = ("APPLY", "IGNORE", "CLARIFY")


def compose_long_context(df: pd.DataFrame) -> pd.Series:
    return "HISTORY:\n" + df["history"].astype(str) + "\nCURRENT QUERY:\n" + df["query"].astype(str)


class FlatFactMemoryBaseline:
    name = "Flat Fact Memory"

    def fit(self, *_args, **_kwargs):
        return self

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        return np.asarray(["APPLY"] * len(texts), dtype=object)


@dataclass
class TfidfActionClassifier:
    name: str
    use_history: bool
    random_state: int = 17

    def __post_init__(self) -> None:
        if self.use_history:
            word = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=20_000,
                sublinear_tf=True,
            )
            char = TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=20_000,
                sublinear_tf=True,
            )
            features = FeatureUnion([("word", word), ("char", char)])
        else:
            features = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=15_000,
                sublinear_tf=True,
            )

        self.pipeline = Pipeline(
            [
                ("features", features),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def _input(self, df: pd.DataFrame) -> pd.Series:
        return compose_long_context(df) if self.use_history else df["query"].astype(str)

    def fit(self, df: pd.DataFrame) -> "TfidfActionClassifier":
        self.pipeline.fit(self._input(df), df["action"].astype(str))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(self._input(df))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(self._input(df))


def make_query_only_baseline(seed: int = 17) -> TfidfActionClassifier:
    return TfidfActionClassifier("Query-Only TF-IDF", use_history=False, random_state=seed)


def make_long_context_baseline(seed: int = 17) -> TfidfActionClassifier:
    return TfidfActionClassifier("Raw Long-Context TF-IDF", use_history=True, random_state=seed)


__all__ = [
    "LABELS",
    "compose_long_context",
    "FlatFactMemoryBaseline",
    "TfidfActionClassifier",
    "make_query_only_baseline",
    "make_long_context_baseline",
]
