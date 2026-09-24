#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("version https://git-lfs.github.com/spec/"):
        raise RuntimeError(f"{path} is still a Git-LFS pointer")
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpeval-dir", default="data/external_raw/RPEval")
    args = ap.parse_args()
    root = Path(args.rpeval_dir)

    gen_path = root / "data_generation" / "data.json"
    print(f"generation_file={gen_path}")
    print(f"exists={gen_path.exists()}")
    if gen_path.exists():
        print(f"bytes={gen_path.stat().st_size}")
        gen = load(gen_path)
        print(f"generation_query_records={len(gen)}")
        counts = Counter()
        for row in gen:
            for key in ("ignore", "supportive", "dominant"):
                value = row.get(key, [])
                if not isinstance(value, list):
                    value = [value] if value else []
                counts[key] += len(value)
        print("generation_atomic_counts=" + json.dumps(dict(counts), ensure_ascii=False))
        print(f"generation_atomic_total={sum(counts.values())}")
        if gen:
            print("generation_first_keys=" + json.dumps(sorted(gen[0].keys()), ensure_ascii=False))

    files = {
        "explicit_single": root / "benchmark_dataset/explicit_preference/single_testset.json",
        "explicit_multi": root / "benchmark_dataset/explicit_preference/multi_testset.json",
        "implicit_single": root / "benchmark_dataset/implicit_preference/single_testset.json",
        "implicit_multi": root / "benchmark_dataset/implicit_preference/multi_testset.json",
    }
    for name, path in files.items():
        data = load(path)
        atomic = 0
        malformed = 0
        for row in data:
            personas = row.get("persona", [])
            if isinstance(personas, list):
                atomic += len(personas)
                labels = row.get("intent_type", "")
                compact = [c for c in str(labels).upper() if c in {"A", "B", "C"}]
                if len(compact) != len(personas):
                    malformed += 1
            else:
                atomic += 1
        print(
            f"{name}: query_records={len(data)}, "
            f"atomic_from_persona={atomic}, malformed_multi_label_rows={malformed}"
        )


if __name__ == "__main__":
    main()
