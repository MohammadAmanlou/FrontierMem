#!/usr/bin/env python3
"""Build a CPU-only, text-free, group-safe four-arm Qwen experiment plan.

Nothing is trained, and no private S2Pref text is exported. Outputs contain
pair IDs and provenance summaries only; they stay under data/local/.
All S2Pref labels are DERIVED SILVER, not benchmark gold.

Run: python -m scripts.prepare_dual_axis_run
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from scripts.audit_s2pref_candidates import validate_candidate_splits

SUSPECT_PILOT_TRAIN_IDS = frozenset({
    "s2pref_705387d828076935002b_2_mem0",  # date text says Monday, role says Saturday
})
SUSPECT_PILOT_DEV_IDS = frozenset({
    "s2pref_54f73710da24875056c9_1_mem1",  # book club != necessarily close friends
})
THREE_WAY = frozenset({"IGNORE", "SUPPORT", "DOMINATE"})
METHODS = {
    "pointwise": {"rpeval_ce3": True, "rpeval_pref_rank": False,
                  "s2pref_binary": False, "s2pref_context_rank": False},
    "caid": {"rpeval_ce3": True, "rpeval_pref_rank": True,
             "s2pref_binary": False, "s2pref_context_rank": False},
    "caid_s2pref": {"rpeval_ce3": True, "rpeval_pref_rank": True,
                   "s2pref_binary": True, "s2pref_context_rank": False},
    "dual_axis": {"rpeval_ce3": True, "rpeval_pref_rank": True,
                  "s2pref_binary": True, "s2pref_context_rank": True},
}


def load_rpeval_pairs(path: Path):
    seen_ids = set()
    pairs = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != "rpeval_generation":
                raise ValueError(f"row {line_no}: forbidden RPEval training source")
            pid, gid = row.get("pair_id"), row.get("group_id")
            if not pid or not gid or pid in seen_ids:
                raise ValueError(f"row {line_no}: missing or duplicate pair/group ID")
            seen_ids.add(pid)
            for side, expected in (("positive", "APPLY"), ("negative", "IGNORE")):
                s = row.get(side)
                if not isinstance(s, dict) or s.get("action") != expected:
                    raise ValueError(f"row {line_no}: invalid {side} action")
                if s.get("native_label") not in THREE_WAY:
                    raise ValueError(f"row {line_no}: invalid native 3-way label")
                if expected == "APPLY" and s["native_label"] == "IGNORE":
                    raise ValueError(f"row {line_no}: APPLY contradicts native label")
                if expected == "IGNORE" and s["native_label"] != "IGNORE":
                    raise ValueError(f"row {line_no}: IGNORE contradicts native label")
            pairs.append(row)
    if len(pairs) < 2:
        raise ValueError("Need at least two RPEval training pairs")
    return pairs


def split_rpeval_groups(pairs, fraction: float, seed: int):
    if not 0 < fraction < 0.5:
        raise ValueError("dev_fraction must be in (0, 0.5)")
    gids = sorted({str(p["group_id"]) for p in pairs})
    if len(gids) < 2:
        raise ValueError("Need >=2 distinct groups")
    random.Random(seed).shuffle(gids)
    n = max(1, min(len(gids) - 1, round(fraction * len(gids))))
    dev_groups = set(gids[:n])
    train = [p for p in pairs if p["group_id"] not in dev_groups]
    dev = [p for p in pairs if p["group_id"] in dev_groups]
    assert not ({p["group_id"] for p in train} & {p["group_id"] for p in dev})
    return train, dev


def prepare_plan(rpeval_path: Path, s2pref_dir: Path, seed=17, dev_fraction=0.10,
                 exclude_s2pref_train=(), exclude_s2pref_dev=()):
    rpeval = load_rpeval_pairs(rpeval_path)
    rtrain, rdev = split_rpeval_groups(rpeval, dev_fraction, seed)
    audit, s2pref = validate_candidate_splits(s2pref_dir)
    withheld_train = set(SUSPECT_PILOT_TRAIN_IDS) | set(exclude_s2pref_train)
    withheld_dev = set(SUSPECT_PILOT_DEV_IDS) | set(exclude_s2pref_dev)
    strained = [p for p in s2pref["train"] if p["pair_id"] not in withheld_train]
    sdev = [p for p in s2pref["dev"] if p["pair_id"] not in withheld_dev]
    if not strained or not sdev:
        raise ValueError("Filtering removed all train or dev silver pairs")
    train_terms = {(p["positive"]["preference"], p["positive"]["query"]) for p in strained}
    strict_dev = [p for p in sdev
                  if (p["positive"]["preference"], p["positive"]["query"]) not in train_terms]
    if not strict_dev:
        raise ValueError("No lexical-unseen silver dev pairs")
    train_gids = {p["group_id"] for p in strained}
    assert not train_gids.intersection({p["group_id"] for p in sdev})
    s_train_ids = [p["pair_id"] for p in strained]
    # Never export any raw preference, query, context or history to manifest.
    plan = {
        "status": "CPU_PREFLIGHT_ONLY_NO_TRAINING",
        "seed": seed, "rpeval_dev_fraction": dev_fraction,
        "rpeval_input_sha256": hashlib.sha256(rpeval_path.read_bytes()).hexdigest(),
        "s2pref_input_sha256": {
            x: hashlib.sha256((s2pref_dir / f"s2pref_context_{x}_CANDIDATES.jsonl").read_bytes()).hexdigest()
            for x in ("train", "dev")
        },
        "source_and_holdout_policy": {
            "RPEval": "only data_generation native 3-way labels for training",
            "S2Pref": "derived silver binary only; never fabricate SUPPORT/DOMINATE",
            "BenchPreS": "evaluation_only_and_absent",
            "S2Pref_test": "untouched_and_absent",
            "status": "human semantic label review pending; exploratory use only",
        },
        "rpeval": {
            "train_pair_count": len(rtrain), "dev_pair_count": len(rdev),
            "train_group_count": len({p["group_id"] for p in rtrain}),
            "dev_group_count": len({p["group_id"] for p in rdev}),
            "train_pair_ids": [p["pair_id"] for p in rtrain],
            "dev_pair_ids": [p["pair_id"] for p in rdev],
        },
        "s2pref": {
            "original_train_pair_count": len(s2pref["train"]),
            "original_dev_pair_count": len(s2pref["dev"]),
            "train_pair_count": len(strained),
            "dev_pair_count": len(sdev),
            "strict_unseen_dev_pair_count": len(strict_dev),
            "train_pair_ids": s_train_ids,
            "dev_pair_ids": [p["pair_id"] for p in sdev],
            "strict_unseen_dev_pair_ids": [p["pair_id"] for p in strict_dev],
            "excluded_pilot_train_pair_ids": sorted(withheld_train & {p["pair_id"] for p in s2pref["train"]}),
            "excluded_pilot_dev_pair_ids": sorted(withheld_dev & {p["pair_id"] for p in s2pref["dev"]}),
            "pilot_exclusions_are_provisional": True,
            "original_structural_status": audit["status"],
            "train_dev_group_overlap": audit["train_dev_group_overlap"],
            "train_dev_preference_query_overlap": audit["train_dev_preference_query_overlap"],
        },
        "methods": {name: {**losses,
                            "rpeval_train_pair_ids": "rpeval.train_pair_ids",
                            "s2pref_train_pair_ids": ("s2pref.train_pair_ids" if losses["s2pref_binary"] else None)}
                    for name, losses in METHODS.items()},
        "comparability": [
            "Pointwise vs CAID: match native RPEval sides and optimization steps.",
            "CAID+S2Pref vs Dual-Axis: match both datasets and optimization steps.",
            "Different S2Pref exposure makes baseline-vs-augmented comparisons noncausal without controls.",
            "No held-out benchmark data is in this plan.",
        ],
    }
    assert (set(plan["rpeval"]["train_pair_ids"]) &
            set(plan["rpeval"]["dev_pair_ids"])) == set()
    assert (set(s_train_ids) &
            set(plan["s2pref"]["dev_pair_ids"])) == set()
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpeval-pairs", type=Path,
                        default=Path("data/external_processed/caid_pairwise_train.jsonl"))
    parser.add_argument("--s2pref-dir", type=Path,
                        default=Path("data/local/s2pref_candidates"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/local/dual_axis_plan/experiment_plan.json"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--exclude-s2pref-train", action="append", default=[])
    parser.add_argument("--exclude-s2pref-dev", action="append", default=[])
    args = parser.parse_args()
    plan = prepare_plan(args.rpeval_pairs, args.s2pref_dir, args.seed,
                        args.dev_fraction, args.exclude_s2pref_train, args.exclude_s2pref_dev)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    # Print only aggregate counts. Individual sample IDs live in local/ and are not pushed.
    summary = {
        "status": plan["status"],
        "rpeval": {k:v for k,v in plan["rpeval"].items() if not k.endswith("_ids")},
        "s2pref": {k:v for k,v in plan["s2pref"].items() if not k.endswith("_ids")},
        "experiment_arms": list(plan["methods"]),
        "local_manifest_path": str(args.output),
        "important": plan["source_and_holdout_policy"]["status"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
