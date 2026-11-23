#!/usr/bin/env python3
"""Structural + provenance audit for exported S2Pref TRAIN/DEV *silver candidates*.

Only reads two local candidate files. NEVER reads or exports the held-out test set.
Produces a JSON audit and an annotation CSV for independent semantic review.
A structural PASS does not certify that any silver label is correct.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re

NAMES = {
    "train": "s2pref_context_train_CANDIDATES.jsonl",
    "dev": "s2pref_context_dev_CANDIDATES.jsonl",
}
EXPECTED_LABEL_STATUS = "S2Pref_derived_silver_unreviewed"
EXPECTED_PAIR_TYPE = "same_preference_same_query_opposite_context"


def _read_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{lineno}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{lineno}: expected JSON object")
            yield value


def validate_candidate_splits(input_dir: Path):
    """Return summary and records (for local annotation only). Raise on violation."""
    all_pairs = {}
    seen_pairs, seen_examples, seen_source_shas = set(), set(), {}
    seen_contexts = {}
    seen_preference_queries = {}
    split_groups = {}
    for split, filename in NAMES.items():
        path = input_dir / filename
        pairs = []
        groups, original_shas = set(), set()
        source_group_map = {}
        blank_history = 0
        queries = Counter()
        pair_contexts = set()
        for i, p in enumerate(_read_jsonl(path), 1):
            loc = f"{filename}:{i}"
            for key in ("pair_id", "group_id", "source_original_sha", "positive", "negative"):
                if key not in p:
                    raise ValueError(f"{loc}: missing {key}")
            pid = p["pair_id"]
            gid = p["group_id"]
            source_sha = p["source_original_sha"]
            if not all(isinstance(x, str) and x.strip() for x in (pid, gid, source_sha)):
                raise ValueError(f"{loc}: missing identifiers")
            if pid in seen_pairs:
                raise ValueError(f"{loc}: duplicate pair_id {pid}")
            seen_pairs.add(pid)
            if not re.fullmatch(r"[0-9a-f]{64}", source_sha):
                raise ValueError(f"{loc}: malformed original source SHA256")
            if gid != "s2pref_source_" + source_sha[:20]:
                raise ValueError(f"{loc}: source/group identity inconsistent")
            if p.get("source") != "s2pref" or p.get("split") != split:
                raise ValueError(f"{loc}: source/split mismatch")
            if p.get("usage") != f"{split}_candidate":
                raise ValueError(f"{loc}: unexpected usage")
            if p.get("label_status") != EXPECTED_LABEL_STATUS:
                raise ValueError(f"{loc}: unreviewed silver status required")
            if p.get("pair_type") != EXPECTED_PAIR_TYPE:
                raise ValueError(f"{loc}: not a same-preference contextual contrast")
            if gid in source_group_map and source_group_map[gid] != source_sha:
                raise ValueError(f"{loc}: inconsistent group->source mapping")
            source_group_map[gid] = source_sha
            if source_sha in seen_source_shas and seen_source_shas[source_sha] != split:
                raise ValueError(f"{loc}: original source SHA leaks across splits")
            seen_source_shas[source_sha] = split
            for side, action in (("positive", "APPLY"), ("negative", "IGNORE")):
                obj = p[side]
                if not isinstance(obj, dict):
                    raise ValueError(f"{loc}: {side} must be object")
                if obj.get("action") != action:
                    raise ValueError(f"{loc}: expected {side} action {action}")
                for key in ("example_id", "preference", "context", "query"):
                    if not isinstance(obj.get(key), str) or not obj[key].strip():
                        raise ValueError(f"{loc}: empty {side}.{key}")
                exid = obj["example_id"]
                if exid in seen_examples:
                    raise ValueError(f"{loc}: duplicate atomic example ID")
                seen_examples.add(exid)
                context_hash = hashlib.sha256(obj["context"].strip().encode("utf8")).hexdigest()
                if context_hash in seen_contexts and seen_contexts[context_hash] != split:
                    raise ValueError(f"{loc}: identical context text crosses train/dev")
                seen_contexts[context_hash] = split
            pos, neg = p["positive"], p["negative"]
            for key in ("preference", "query", "history"):
                if pos.get(key, "") != neg.get(key, ""):
                    raise ValueError(f"{loc}: pair sides differ in {key}")
            if pos["context"].strip() == neg["context"].strip():
                raise ValueError(f"{loc}: unchanged context in contrast pair")
            if (pos["context"], neg["context"], pos["preference"]) in pair_contexts:
                raise ValueError(f"{loc}: repeated content-identical pair")
            pair_contexts.add((pos["context"], neg["context"], pos["preference"]))
            if not str(pos.get("history") or "").strip():
                blank_history += 1
            groups.add(gid)
            original_shas.add(source_sha)
            queries[pos["query"]] += 1
            pairs.append(p)
        if not pairs:
            raise ValueError(f"{filename} is empty")
        split_groups[split] = groups
        all_pairs[split] = pairs
        seen_preference_queries[split] = {(p["positive"]["preference"], p["positive"]["query"]) for p in pairs}
    group_overlap = split_groups["train"] & split_groups["dev"]
    if group_overlap:
        raise ValueError(f"Train/dev group overlap: {len(group_overlap)}")

    summary = {
        "status": "PASS_STRUCTURAL_CHECKS_ONLY_NOT_SEMANTIC_GOLD",
        "splits": {
            split: {
                "pairs": len(pairs),
                "original_source_groups": len(split_groups[split]),
                "empty_history_pairs": sum(not str(p["positive"].get("history") or "").strip() for p in pairs),
                "unique_queries": len({p["positive"]["query"] for p in pairs}),
                "unique_preferences": len({p["positive"]["preference"] for p in pairs}),
                "sha256_file": hashlib.sha256((input_dir / NAMES[split]).read_bytes()).hexdigest(),
            } for split, pairs in all_pairs.items()
        },
        "train_dev_group_overlap": len(group_overlap),
        "train_dev_preference_query_overlap": len(
            seen_preference_queries["train"] & seen_preference_queries["dev"]
        ),
        "audit_limitations": [
            "Silver binary actions are derived, not native S2Pref gold.",
            "No semantic truth, causal validity, or history coverage verified.",
            "Matching preference text across distinct groups can still permit lexical leakage.",
            "Annotation CSV contains source text and must remain local/private until license reviewed.",
            "Never train on dev or held-out test records.",
        ],
    }
    return summary, all_pairs


def annotation_sample(pairs_by_split, per_split: int, seed: int):
    """Group-diverse, deterministic sample for an independent human audit."""
    rng = random.Random(seed)
    output = []
    for split, pairs in pairs_by_split.items():
        by_group = defaultdict(list)
        for p in pairs:
            by_group[p["group_id"]].append(p)
        groups = sorted(by_group)
        rng.shuffle(groups)
        # One per source group first; next rounds cover repeats only if requested.
        selected = []
        while len(selected) < min(per_split, len(pairs)):
            candidates = []
            for group in groups:
                options = [p for p in by_group[group] if p not in selected]
                if options:
                    candidates.append(rng.choice(options))
            if not candidates:
                break
            selected.extend(candidates[: max(0, per_split - len(selected))])
        for p in selected:
            output.append({
                "split": split, "pair_id": p["pair_id"],
                "group_id": p["group_id"], "preference": p["positive"]["preference"],
                "query": p["positive"]["query"],
                "history": p["positive"].get("history", ""),
                "context_apply_silver": p["positive"]["context"],
                "context_ignore_silver": p["negative"]["context"],
                "reviewer_apply_correct": "", "reviewer_ignore_correct": "",
                "clear_context_flip": "", "label_ambiguous": "",
                "reviewer_notes": "",
            })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/local/s2pref_candidates"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/local/s2pref_audit"))
    parser.add_argument("--sample-per-split", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.sample_per_split < 1:
        parser.error("--sample-per-split must be >=1")
    summary, pairs = validate_candidate_splits(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "structural_audit.json"
    audit_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    sample = annotation_sample(pairs, args.sample_per_split, args.seed)
    sample_path = args.output_dir / "human_review_sample.csv"
    with sample_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample[0]))
        writer.writeheader()
        writer.writerows(sample)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"LOCAL HUMAN REVIEW SAMPLE: {sample_path} ({len(sample)} rows)")
    print("REVIEW NOT COMPLETE: do not claim these are independently verified gold labels.")


if __name__ == "__main__":
    main()
