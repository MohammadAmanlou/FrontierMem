from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .external_data import preference_key, stable_hash


def _context_signature(row: dict[str, Any]) -> str:
    return stable_hash(
        row.get("query", ""),
        row.get("history", ""),
        row.get("context", ""),
        prefix="ctx-",
    )


def _pair_usage(left: dict[str, Any], right: dict[str, Any]) -> str:
    l, r = str(left.get("usage", "")), str(right.get("usage", ""))
    if l == r:
        return l
    # Never let a mixed-source pair silently enter training.
    return "eval"


def build_natural_contrast_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    max_pairs_per_preference: int = 20,
    seed: int = 17,
    allowed_usage: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Same-preference, different-context APPLY-vs-IGNORE pairs.

    This is the cleanest benchmark-native analogue of an applicability boundary:
    the released preference text is unchanged, while existing contexts imply
    opposite utilization decisions. No text is generated or rewritten.
    """
    if allowed_usage is None:
        allowed_usage = {
            "train_pool", "train", "calibration", "dev", "eval", "eval_only"
        }

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        usage = str(row.get("usage", ""))
        if usage not in allowed_usage:
            continue
        action = str(row.get("action", "")).upper()
        if action not in {"APPLY", "IGNORE"}:
            continue
        pref = preference_key(row.get("preference", ""))
        if not pref:
            continue
        grouped[(str(row.get("source", "")), usage, pref)].append(row)

    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []

    for (source, usage, pref_key), items in grouped.items():
        apply_rows = [r for r in items if str(r["action"]).upper() == "APPLY"]
        ignore_rows = [r for r in items if str(r["action"]).upper() == "IGNORE"]
        candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for a in apply_rows:
            for i in ignore_rows:
                if _context_signature(a) != _context_signature(i):
                    candidates.append((a, i))
        rng.shuffle(candidates)

        for a, i in candidates[:max_pairs_per_preference]:
            pairs.append(
                {
                    "pair_id": stable_hash(
                        source, usage, pref_key,
                        a.get("example_id"), i.get("example_id"),
                        prefix="ctxflip-",
                    ),
                    "pair_type": "context_decision_flip",
                    "source": source,
                    "usage": usage,
                    "preference_key": pref_key,
                    "positive_preference": a.get("preference", ""),
                    "positive_example_id": a.get("example_id", ""),
                    "positive_query": a.get("query", ""),
                    "positive_history": a.get("history", ""),
                    "positive_context": a.get("context", ""),
                    "positive_source_label": a.get("source_label", ""),
                    "negative_preference": i.get("preference", ""),
                    "negative_example_id": i.get("example_id", ""),
                    "negative_query": i.get("query", ""),
                    "negative_history": i.get("history", ""),
                    "negative_context": i.get("context", ""),
                    "negative_source_label": i.get("source_label", ""),
                }
            )
    return pairs


def build_query_preference_contrast_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    max_pairs_per_group: int = 12,
    seed: int = 17,
    allowed_usage: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Same-query, different-preference positive/negative utilization pairs.

    RPEval's multi-preference records naturally supply these comparisons:
    under one fixed user query, some stored preferences should be used while
    others should be ignored. This gives a strong existing-data pairwise
    training signal even when the exact same preference does not recur across
    multiple contexts.
    """
    if allowed_usage is None:
        allowed_usage = {"train_pool", "train", "calibration", "dev"}

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        usage = str(row.get("usage", ""))
        if usage not in allowed_usage:
            continue
        action = str(row.get("action", "")).upper()
        if action not in {"APPLY", "IGNORE"}:
            continue
        grouped[
            (str(row.get("source", "")), usage, str(row.get("group_id", "")))
        ].append(row)

    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    for (source, usage, group_id), items in grouped.items():
        pos = [r for r in items if str(r["action"]).upper() == "APPLY"]
        neg = [r for r in items if str(r["action"]).upper() == "IGNORE"]
        candidates = [(p, n) for p in pos for n in neg]
        rng.shuffle(candidates)
        for p, n in candidates[:max_pairs_per_group]:
            pairs.append(
                {
                    "pair_id": stable_hash(
                        source, usage, group_id,
                        p.get("example_id"), n.get("example_id"),
                        prefix="qcontrast-",
                    ),
                    "pair_type": "query_preference_contrast",
                    "source": source,
                    "usage": usage,
                    "group_id": group_id,
                    "positive_preference": p.get("preference", ""),
                    "positive_example_id": p.get("example_id", ""),
                    "positive_query": p.get("query", ""),
                    "positive_history": p.get("history", ""),
                    "positive_context": p.get("context", ""),
                    "positive_source_label": p.get("source_label", ""),
                    "negative_preference": n.get("preference", ""),
                    "negative_example_id": n.get("example_id", ""),
                    "negative_query": n.get("query", ""),
                    "negative_history": n.get("history", ""),
                    "negative_context": n.get("context", ""),
                    "negative_source_label": n.get("source_label", ""),
                }
            )
    return pairs


def build_invariance_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    max_pairs_per_preference_action: int = 10,
    seed: int = 17,
    allowed_usage: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Same preference, different existing contexts, same utilization decision."""
    if allowed_usage is None:
        allowed_usage = {
            "train_pool", "train", "calibration", "dev", "eval", "eval_only"
        }

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        usage = str(row.get("usage", ""))
        if usage not in allowed_usage:
            continue
        action = str(row.get("action", "")).upper()
        if action not in {"APPLY", "IGNORE"}:
            continue
        pref = preference_key(row.get("preference", ""))
        if pref:
            grouped[(str(row.get("source", "")), usage, pref, action)].append(row)

    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    for (source, usage, pref_key, action), items in grouped.items():
        candidates = []
        for x_idx in range(len(items)):
            for y_idx in range(x_idx + 1, len(items)):
                x, y = items[x_idx], items[y_idx]
                if _context_signature(x) != _context_signature(y):
                    candidates.append((x, y))
        rng.shuffle(candidates)
        for x, y in candidates[:max_pairs_per_preference_action]:
            pairs.append(
                {
                    "pair_id": stable_hash(
                        source, usage, pref_key, action,
                        x.get("example_id"), y.get("example_id"),
                        prefix="inv-",
                    ),
                    "pair_type": "decision_invariant",
                    "source": source,
                    "usage": usage,
                    "preference_key": pref_key,
                    "action": action,
                    "left_preference": x.get("preference", ""),
                    "left_example_id": x.get("example_id", ""),
                    "left_query": x.get("query", ""),
                    "left_history": x.get("history", ""),
                    "left_context": x.get("context", ""),
                    "right_preference": y.get("preference", ""),
                    "right_example_id": y.get("example_id", ""),
                    "right_query": y.get("query", ""),
                    "right_history": y.get("history", ""),
                    "right_context": y.get("context", ""),
                }
            )
    return pairs


def save_pairs(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_pairs(path: str | Path) -> list[dict[str, Any]]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


__all__ = [
    "build_invariance_pairs",
    "build_natural_contrast_pairs",
    "build_query_preference_contrast_pairs",
    "load_pairs",
    "save_pairs",
]
