#!/usr/bin/env python3
"""Read-only, CPU-only data preflight for the FrontierMem HPC experiments.

Never prints raw dataset text or opens BenchPreS/S2Pref test data for training.
It checks the on-disk RPEval pair source and source-usage policy; source-group
checks are conditional on available group identifiers. Not a semantic label audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

ALLOWED = {
    "rpeval_generation": {"train", "calibration"},
    "rpeval_explicit": {"eval"},
    "rpeval_implicit": {"eval"},
    "benchpres": {"eval_only"},
}
ACTIONS = {"APPLY", "IGNORE"}


def rows(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        for index, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path.name}:{index}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path.name}:{index}")
            yield value


def audit_atomic(path: Path):
    by_source = Counter()
    by_usage = Counter()
    by_action = Counter()
    group_usage: dict[str, set[str]] = defaultdict(set)
    group_missing = 0
    duplicate_ids = 0
    example_ids = set()
    for i, item in enumerate(rows(path), 1):
        source, usage, action = item.get("source"), item.get("usage"), item.get("action")
        if source not in ALLOWED:
            raise ValueError(f"Unknown source at row {i}: {source!r}")
        if usage not in ALLOWED[source]:
            raise ValueError(f"Source/usage policy violation at row {i}: {source!r}/{usage!r}")
        if action not in ACTIONS:
            raise ValueError(f"Invalid action at row {i}: {action!r}")
        by_source[source] += 1
        by_usage[usage] += 1
        by_action[action] += 1
        eid = item.get("example_id")
        if eid:
            scoped_id = (source, str(eid))
            if scoped_id in example_ids:
                duplicate_ids += 1
            example_ids.add(scoped_id)
        if source == "rpeval_generation":
            gid = item.get("group_id")
            if not gid:
                group_missing += 1
            else:
                group_usage[str(gid)].add(usage)
    if duplicate_ids:
        raise ValueError(f"{duplicate_ids} duplicated (source, example_id) entries")
    overlaps = sum(len(s) > 1 for s in group_usage.values())
    if overlaps:
        raise ValueError(f"RPEval generation groups spanning train/calibration: {overlaps}")
    if not by_source:
        raise ValueError("Empty atomic examples")
    return {
        "n": sum(by_source.values()),
        "by_source": dict(sorted(by_source.items())),
        "by_usage": dict(sorted(by_usage.items())),
        "by_action": dict(sorted(by_action.items())),
        "rpeval_generation_groups_checked": len(group_usage),
        "rpeval_generation_missing_group_ids": group_missing,
        "rpeval_train_calibration_group_overlap": overlaps,
    }


def audit_training_pairs(path: Path):
    seen = set()
    pair_types = Counter()
    groups = set()
    for i, item in enumerate(rows(path), 1):
        if item.get("source") != "rpeval_generation":
            raise ValueError(f"Non-training source in pair row {i}")
        pid = item.get("pair_id")
        if not pid or pid in seen:
            raise ValueError(f"Empty/duplicate pair_id in row {i}")
        seen.add(pid)
        pos, neg = item.get("positive", {}), item.get("negative", {})
        if not isinstance(pos, dict) or not isinstance(neg, dict):
            raise ValueError(f"Malformed pair sides at row {i}")
        if pos.get("action") != "APPLY" or neg.get("action") != "IGNORE":
            raise ValueError(f"Incorrect positive/negative actions in row {i}")
        if pos.get("native_label") not in {"SUPPORT", "DOMINATE"} or neg.get("native_label") != "IGNORE":
            raise ValueError(f"Invalid RPEval-native labels in row {i}")
        pair_types[str(item.get("pair_type", "unspecified"))] += 1
        gid = item.get("group_id")
        if gid:
            groups.add(str(gid))
    if not seen:
        raise ValueError("Empty RPEval training pair file")
    return {"n": len(seen), "n_groups_with_id": len(groups), "by_pair_type": dict(sorted(pair_types.items()))}


def inspect(data_dir: Path):
    main = audit_atomic(data_dir / "all_applicability_examples.jsonl")
    training = audit_training_pairs(data_dir / "caid_pairwise_train.jsonl")
    context_file = data_dir / "context_decision_flip_pairs.jsonl"
    context_lines = sum(1 for _ in rows(context_file)) if context_file.is_file() else None
    return {
        "status": "PASS_STRUCTURAL_CHECKS_ONLY",
        "atomic": main,
        "rpeval_training_pairs": training,
        "legacy_context_flip_records": context_lines,
        "notes": [
            "BenchPreS and official RPEval tests are evaluation-only.",
            "Existing RPEval query-preference pairs are not same-preference context flips.",
            "S2Pref Phase-01 training candidates need a separate upload and semantic silver-label review.",
            "Group identifiers in third-party sources are not a substitute for human validation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/external_processed"))
    args = parser.parse_args()
    print(json.dumps(inspect(args.data_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
