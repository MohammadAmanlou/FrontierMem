#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontiermem.contrast_sets import (
    build_invariance_pairs,
    build_natural_contrast_pairs,
    build_query_preference_contrast_pairs,
    save_pairs,
)
from frontiermem.external_data import (
    download_rpeval_public,
    load_benchpres_hf,
    load_rpeval_directory,
    save_jsonl,
    write_registry,
)


def assign_train_calibration_usage(
    rows: list[dict],
    *,
    calibration_fraction: float,
    seed: int,
) -> None:
    """Split the upstream train_pool by source/group before pair construction.

    We split by group_id, never row-wise, so all atomic preferences from one
    RPEval query stay together. The calibration groups are never used in
    pairwise training.
    """
    by_source: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, row in enumerate(rows):
        if row.get("usage") == "train_pool":
            by_source[row["source"]][row["group_id"]].append(idx)

    rng = random.Random(seed)
    for source, groups in by_source.items():
        keys = list(groups)
        rng.shuffle(keys)
        n_cal = max(1, int(round(calibration_fraction * len(keys)))) if len(keys) > 1 else 0
        cal_keys = set(keys[:n_cal])
        for key, indices in groups.items():
            target = "calibration" if key in cal_keys else "train"
            for idx in indices:
                rows[idx]["metadata"] = dict(rows[idx].get("metadata") or {})
                rows[idx]["metadata"]["upstream_usage"] = "train_pool"
                rows[idx]["usage"] = target


def summarize(rows: list[dict]) -> dict:
    return {
        "n_atomic_examples": len(rows),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_usage": dict(Counter(r["usage"] for r in rows)),
        "by_action": dict(Counter(r["action"] for r in rows)),
        "by_source_action": {
            source: dict(Counter(r["action"] for r in rows if r["source"] == source))
            for source in sorted({r["source"] for r in rows})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize existing personalization benchmarks without generating new text."
    )
    parser.add_argument("--output-dir", default="data/external_processed")
    parser.add_argument("--rpeval-dir", default="data/external_raw/RPEval")
    parser.add_argument("--download-rpeval", action="store_true")
    parser.add_argument("--skip-rpeval", action="store_true")
    parser.add_argument(
        "--benchpres",
        action="store_true",
        help="Download and normalize BenchPreS from Hugging Face (evaluation only).",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-context-flip-pairs", type=int, default=20)
    parser.add_argument("--max-query-contrast-pairs", type=int, default=12)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_examples: list[dict] = []

    if not args.skip_rpeval:
        rpeval_dir = Path(args.rpeval_dir)
        if args.download_rpeval:
            download_rpeval_public(rpeval_dir)
        if rpeval_dir.exists():
            rpeval_examples = load_rpeval_directory(rpeval_dir)
            all_examples.extend(ex.to_record() for ex in rpeval_examples)
        else:
            print(
                f"[WARN] RPEval directory not found: {rpeval_dir}. "
                "Use --download-rpeval or provide --rpeval-dir."
            )

    if args.benchpres:
        benchpres = load_benchpres_hf()
        all_examples.extend(ex.to_record() for ex in benchpres)

    if not all_examples:
        raise SystemExit("No benchmark examples were prepared.")

    assign_train_calibration_usage(
        all_examples,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )

    # Save source-specific normalized views after the split assignment.
    for source in sorted({r["source"] for r in all_examples}):
        save_jsonl(
            [r for r in all_examples if r["source"] == source],
            out / f"{source}.jsonl",
        )
    save_jsonl(all_examples, out / "all_applicability_examples.jsonl")
    write_registry(out / "dataset_registry.json")

    context_flips = build_natural_contrast_pairs(
        all_examples,
        max_pairs_per_preference=args.max_context_flip_pairs,
        seed=args.seed,
    )
    query_contrasts = build_query_preference_contrast_pairs(
        all_examples,
        max_pairs_per_group=args.max_query_contrast_pairs,
        seed=args.seed,
    )
    invariance_pairs = build_invariance_pairs(all_examples, seed=args.seed)

    save_pairs(context_flips, out / "context_decision_flip_pairs.jsonl")
    save_pairs(query_contrasts, out / "query_preference_contrast_pairs.jsonl")
    save_pairs(invariance_pairs, out / "natural_invariance_pairs.jsonl")

    summary = summarize(all_examples)
    summary.update(
        {
            "n_context_decision_flip_pairs": len(context_flips),
            "n_query_preference_contrast_pairs": len(query_contrasts),
            "n_invariance_pairs": len(invariance_pairs),
        }
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nPrepared files in: {out}")


if __name__ == "__main__":
    main()
