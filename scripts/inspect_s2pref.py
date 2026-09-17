#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def walk(obj, path="$", out=None, depth=0, max_depth=8):
    if out is None:
        out = Counter()
    if depth > max_depth:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out[f"{path}.{k}"] += 1
            walk(v, f"{path}.{k}", out, depth + 1, max_depth)
    elif isinstance(obj, list):
        out[f"{path}[]"] += len(obj)
        for item in obj[:20]:
            walk(item, f"{path}[]", out, depth + 1, max_depth)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download/inspect S2Pref without guessing task labels. The HF dataset "
            "contains heterogeneous nested JSON, so this produces a schema report "
            "for a later exact task-specific adapter."
        )
    )
    parser.add_argument("--output-dir", default="data/external_raw/S2Pref")
    parser.add_argument("--max-files", type=int, default=20)
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("pip install -r requirements-external.txt") from exc

    local = Path(
        snapshot_download(
            repo_id="cyhfut/s2pref",
            repo_type="dataset",
            local_dir=args.output_dir,
        )
    )
    json_files = list(local.rglob("*.json")) + list(local.rglob("*.jsonl"))
    report = {"root": str(local), "files": []}

    for path in json_files[: args.max_files]:
        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".jsonl":
                first = next((line for line in text.splitlines() if line.strip()), "")
                obj = json.loads(first) if first else {}
            else:
                obj = json.loads(text)
            counts = walk(obj)
            report["files"].append(
                {
                    "path": str(path.relative_to(local)),
                    "top_type": type(obj).__name__,
                    "frequent_paths": counts.most_common(80),
                }
            )
        except Exception as exc:
            report["files"].append(
                {"path": str(path.relative_to(local)), "error": repr(exc)}
            )

    report_path = local / "frontiermem_s2pref_schema_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote schema report to {report_path}")
    print(
        "No S2Pref action labels were guessed. Inspect the report and map the exact "
        "published tasks before using S2Pref in a reported experiment."
    )


if __name__ == "__main__":
    main()
