#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def native_label(raw: str, fallback_action: str) -> str:
    text = str(raw).strip().lower()
    mapping = {
        "a": "IGNORE",
        "ignore": "IGNORE",
        "b": "SUPPORT",
        "support": "SUPPORT",
        "supportive": "SUPPORT",
        "c": "DOMINATE",
        "dominate": "DOMINATE",
        "dominant": "DOMINATE",
    }
    return mapping.get(text, fallback_action)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export benchmark-native positive/negative applicability pairs. "
            "No new text is generated. RPEval's SUPPORT/DOMINATE distinction "
            "is preserved for native 3-way evaluation."
        )
    )
    parser.add_argument("--pairs", action="append", default=None)
    parser.add_argument(
        "--output",
        default="data/external_processed/caid_pairwise_train.jsonl",
    )
    args = parser.parse_args()

    pair_files = args.pairs or [
        "data/external_processed/query_preference_contrast_pairs.jsonl",
        "data/external_processed/context_decision_flip_pairs.jsonl",
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    seen = set()
    with out.open("w", encoding="utf-8") as f:
        for pair_file in pair_files:
            path = Path(pair_file)
            if not path.exists():
                continue
            for p in read_jsonl(path):
                if p.get("usage") not in {"train", "train_pool"}:
                    continue
                if p["pair_id"] in seen:
                    continue
                seen.add(p["pair_id"])
                pos_native = native_label(
                    p.get("positive_source_label", ""), "SUPPORT"
                )
                neg_native = native_label(
                    p.get("negative_source_label", ""), "IGNORE"
                )
                # Training pairs require a positive applicable side and an ignored side.
                if pos_native == "IGNORE" or neg_native != "IGNORE":
                    continue
                row = {
                    "pair_id": p["pair_id"],
                    "pair_type": p.get("pair_type", "unknown"),
                    "source": p["source"],
                    "positive": {
                        "preference": p.get("positive_preference", ""),
                        "query": p["positive_query"],
                        "history": p.get("positive_history", ""),
                        "context": p.get("positive_context", ""),
                        "action": "APPLY",
                        "native_label": pos_native,
                    },
                    "negative": {
                        "preference": p.get("negative_preference", ""),
                        "query": p["negative_query"],
                        "history": p.get("negative_history", ""),
                        "context": p.get("negative_context", ""),
                        "action": "IGNORE",
                        "native_label": "IGNORE",
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} benchmark-native applicability pairs to {out}")


if __name__ == "__main__":
    main()
