#!/usr/bin/env python3
"""Reproducible, text-safe loader for the four-arm Qwen decision experiment.

Importable on CPU without torch. The manifest is the sole authority for group
splits and pilot exclusions. BenchPreS and benchmark test text are never read.
S2Pref supplies binary *derived silver* labels only (NOT three-way gold).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ARMS = ("pointwise", "caid", "caid_s2pref", "dual_axis")
BINARY_ARMS = frozenset(("caid_s2pref", "dual_axis"))
S2PREF_NAMES = {
    "train": "s2pref_context_train_CANDIDATES.jsonl",
    "dev": "s2pref_context_dev_CANDIDATES.jsonl",
}
RPEVAL_NATIVE = frozenset(("IGNORE", "SUPPORT", "DOMINATE"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_pairs(rows, ids, label):
    by_id = {}
    for p in rows:
        pid = p.get("pair_id")
        if not pid or pid in by_id:
            raise ValueError(f"{label}: empty/duplicated pair_id")
        by_id[pid] = p
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: duplicated manifest pair ids")
    missing = set(ids) - set(by_id)
    if missing:
        raise ValueError(f"{label}: missing manifest pair ids ({len(missing)})")
    return [by_id[pid] for pid in ids]


def load_and_verify(manifest_path: Path, rpeval_path: Path, s2pref_dir: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "CPU_PREFLIGHT_ONLY_NO_TRAINING":
        raise ValueError("Unknown experiment manifest status")
    if manifest.get("rpeval_input_sha256") != file_sha256(rpeval_path):
        raise ValueError("RPEval input changed since experiment plan")
    for split, name in S2PREF_NAMES.items():
        expected = manifest.get("s2pref_input_sha256", {}).get(split)
        if expected != file_sha256(s2pref_dir / name):
            raise ValueError(f"S2Pref {split} file changed since experiment plan")
    rpeval = read_jsonl(rpeval_path)
    r_tr = select_pairs(rpeval, manifest["rpeval"]["train_pair_ids"], "rpeval train")
    r_de = select_pairs(rpeval, manifest["rpeval"]["dev_pair_ids"], "rpeval dev")
    for p in r_tr + r_de:
        if p.get("source") != "rpeval_generation":
            raise ValueError("Forbidden source in RPEval train/dev")
        for k, action in (("positive", "APPLY"), ("negative", "IGNORE")):
            s = p[k]
            if s.get("action") != action or s.get("native_label") not in RPEVAL_NATIVE:
                raise ValueError("Invalid RPEval gold side")
    if {p["group_id"] for p in r_tr} & {p["group_id"] for p in r_de}:
        raise ValueError("RPEval group leakage")

    silver_tr_all = read_jsonl(s2pref_dir / S2PREF_NAMES["train"])
    silver_de_all = read_jsonl(s2pref_dir / S2PREF_NAMES["dev"])
    s_tr = select_pairs(silver_tr_all, manifest["s2pref"]["train_pair_ids"], "silver train")
    s_de = select_pairs(silver_de_all, manifest["s2pref"]["dev_pair_ids"], "silver dev")
    strict = select_pairs(s_de, manifest["s2pref"]["strict_unseen_dev_pair_ids"], "silver unseen dev")
    if {p["group_id"] for p in s_tr} & {p["group_id"] for p in s_de}:
        raise ValueError("S2Pref group leakage")
    if ({p["positive"]["preference"] for p in s_tr}
            & {p["positive"]["preference"] for p in strict}):
        raise ValueError("Unseen dev contains preference seen in train")
    for p in s_tr + s_de:
        if p.get("source") != "s2pref" or p.get("label_status") != "S2Pref_derived_silver_unreviewed":
            raise ValueError("S2Pref must remain explicitly derived silver")
        if p.get("usage") not in ("train_candidate", "dev_candidate"):
            raise ValueError("Invalid S2Pref usage")
        if p["positive"].get("action") != "APPLY" or p["negative"].get("action") != "IGNORE":
            raise ValueError("Invalid S2Pref binary actions")
        if any("native_label" in p[side] for side in ("positive", "negative")):
            raise ValueError("S2Pref contains fabricated native labels")
        if p["positive"].get("preference") != p["negative"].get("preference"):
            raise ValueError("S2Pref contrast must preserve preference")
        if p["positive"].get("query") != p["negative"].get("query"):
            raise ValueError("S2Pref contrast must preserve query")
    return {
        "manifest": manifest,
        "rpeval_train": r_tr, "rpeval_dev": r_de,
        "s2pref_train": s_tr, "s2pref_dev": s_de,
        "s2pref_strict_dev": strict,
    }


def format_side(side: dict) -> str:
    """One identical input template for RPEval/S2Pref and every training arm."""
    segments = ["[PREFERENCE]\n" + str(side.get("preference", ""))]
    history = str(side.get("history") or "").strip()
    context = str(side.get("context") or "").strip()
    if history:
        segments.append("[HISTORY]\n" + history)
    if context:
        segments.append("[CONTEXT]\n" + context)
    segments.extend((
        "[QUERY]\n" + str(side.get("query", "")),
        "[TASK]\nClassify how the preference should affect this request: "
        "IGNORE, SUPPORT, or DOMINATE. The binary applicability event is "
        "SUPPORT or DOMINATE versus IGNORE.",
        "[DECISION]",
    ))
    return "\n".join(segments)


def summary(data: dict, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Invalid training arm")
    return {
        "arm": arm,
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "rpeval_train_pairs": len(data["rpeval_train"]),
        "rpeval_dev_pairs": len(data["rpeval_dev"]),
        "s2pref_train_pairs": len(data["s2pref_train"]) if arm in BINARY_ARMS else 0,
        "s2pref_dev_pairs": len(data["s2pref_dev"]),
        "s2pref_strict_unseen_dev_pairs": len(data["s2pref_strict_dev"]),
        "s2pref_label_status": "derived_silver_unreviewed",
        "native_three_class_loss_on_s2pref": False,
        "s2pref_context_rank_enabled": arm == "dual_axis",
        "rpeval_pref_rank_enabled": arm != "pointwise",
        "benchmark_test_sets_loaded": False,
    }
