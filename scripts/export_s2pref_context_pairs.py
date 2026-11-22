#!/usr/bin/env python3
"""Audit and export S2Pref silver context contrasts from the Phase-01 archive.

Outputs TRAIN/DEV CANDIDATES only (NOT training-approved or native gold).
Never writes held-out test text. The independent semantic audit is pending.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

ZIP_MEMBER = "phase_01/s2pref_manifest_with_splits.csv"
REQUIRED = {"example_id", "pair_id", "group_id", "preference", "history", "context",
            "query", "action", "split", "gold_status", "source", "original_sha"}
SPLITS = {"train", "dev", "test"}

def load_manifest(path: Path):
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            with archive.open(ZIP_MEMBER) as fh:
                contents = fh.read().decode("utf-8-sig")
    else:
        contents = path.read_text(encoding="utf-8-sig")
    return list(csv.DictReader(io.StringIO(contents)))

def validate_and_pair(rows):
    if not rows:
        raise ValueError("Manifest is empty")
    if REQUIRED - set(rows[0]):
        raise ValueError("Missing columns: " + str(sorted(REQUIRED-set(rows[0]))))
    split_groups = defaultdict(set)
    grouped = defaultdict(list)
    seen_ids = set()
    for r in rows:
        split = r["split"].strip()
        if split not in SPLITS:
            raise ValueError("Invalid split")
        if r["action"] not in {"APPLY", "IGNORE"}:
            raise ValueError("Invalid action")
        if r["gold_status"] != "S2Pref_derived_silver":
            raise ValueError("Expected explicitly silver source labels")
        if r["source"] != "s2pref":
            raise ValueError("Unexpected dataset source")
        if any(not r[k].strip() for k in
               ("example_id", "pair_id", "group_id", "preference", "query", "context", "original_sha")):
            raise ValueError("Required record field is empty")
        if r["example_id"] in seen_ids:
            raise ValueError("Duplicate example ID")
        seen_ids.add(r["example_id"])
        split_groups[split].add(r["group_id"])
        grouped[r["pair_id"]].append(r)
    for s in SPLITS:
        for other in SPLITS - {s}:
            if split_groups[s] & split_groups[other]:
                raise ValueError("Source group leaked across splits")
    pairs = []
    for pair_id, examples in sorted(grouped.items()):
        if len(examples) != 2:
            raise ValueError(f"{pair_id}: expected two sides")
        p, n = sorted(examples, key=lambda r: r["action"] != "APPLY")
        if (p["action"], n["action"]) != ("APPLY", "IGNORE"):
            raise ValueError(f"{pair_id}: expected opposite labels")
        for key in ("preference", "query", "history", "group_id", "split", "source", "original_sha"):
            if p[key] != n[key]:
                raise ValueError(f"{pair_id}: sides must share {key}")
        if p["context"].strip() == n["context"].strip():
            raise ValueError(f"{pair_id}: context must flip")
        pairs.append({
            "pair_id": pair_id,
            "pair_type": "same_preference_same_query_opposite_context",
            "group_id": p["group_id"], "source": "s2pref", "split": p["split"],
            "source_original_sha": p["original_sha"],
            "label_status": "S2Pref_derived_silver_unreviewed",
            "usage": {"train":"train_candidate", "dev":"dev_candidate", "test":"eval_only"}[p["split"]],
            "positive": {k:p[k] for k in ("example_id","preference","history","context","query","action")},
            "negative": {k:n[k] for k in ("example_id","preference","history","context","query","action")},
        })
    return pairs, {k:len(v) for k,v in sorted(split_groups.items())}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    pairs, group_counts = validate_and_pair(load_manifest(args.manifest))
    counts = Counter(p["split"] for p in pairs)
    audit = {
        "source_path_name":args.manifest.name,
        "source_archive_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "upstream_label_type":"derived_silver_NOT_gold",
        "pairs_by_split":{s:counts.get(s,0) for s in ("train","dev","test")},
        "source_groups_by_split":group_counts,
        "all_pairs_same_preference_query_history_different_context":True,
        "source_group_overlap":0,
        "test_exported":False,
        "caveat":"Silver labels need independent semantic review before training."
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"s2pref_context_pair_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    if not args.audit_only:
        for split in ("train","dev"):
            output = args.output_dir / f"s2pref_context_{split}_CANDIDATES.jsonl"
            with output.open("w", encoding="utf-8") as f:
                for p in pairs:
                    if p["split"] == split:
                        f.write(json.dumps(p, ensure_ascii=False)+"\n")
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
