from __future__ import annotations

from typing import List
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class FlatFactMemoryBaseline:
    """Collapses every observed preference into a global fact and always applies it."""

    def fit(self, texts: List[str], labels: List[str] | None = None) -> "FlatFactMemoryBaseline":
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        return np.array(["APPLY"] * len(texts), dtype=object)


class QueryOnlyClassifier:
    """Predicts the action from the current query without using user history."""

    def __init__(self, random_state: int = 42) -> None:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=15000)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ])

    def fit(self, queries: List[str], labels: List[str]) -> "QueryOnlyClassifier":
        self.pipeline.fit(queries, labels)
        return self

    def predict(self, queries: List[str]) -> np.ndarray:
        return self.pipeline.predict(queries)


class RawHistoryClassifier:
    """Directly classifies Apply/Clarify/Ignore from raw history + query."""

    def __init__(self, random_state: int = 42) -> None:
        word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=25000, sublinear_tf=True)
        self.pipeline = Pipeline([
            ("features", FeatureUnion([("word", word), ("char", char)])),
            ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", C=3.0, random_state=random_state)),
        ])

    def fit(self, texts: List[str], labels: List[str]) -> "RawHistoryClassifier":
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict(texts)
