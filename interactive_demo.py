from __future__ import annotations

import argparse
import json
import pandas as pd
from pathlib import Path

from frontiermem.memory import FrontierPolicy

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "frontier_suite_mini.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a FrontierMem decision and structured memory.")
    parser.add_argument("--family", default="travel_budget")
    parser.add_argument("--zone", choices=["inside", "outside", "near"], default="near")
    parser.add_argument("--variant", choices=["stable", "scoped"], default="scoped")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise SystemExit("Run `python run_demo.py` first to generate the dataset.")

    df = pd.read_csv(DATA_PATH)
    row = df[
        (df["family"] == args.family)
        & (df["zone"] == args.zone)
        & (df["profile_variant"] == args.variant)
    ].iloc[0]

    output = FrontierPolicy().predict_one(row["history"], row["query"])
    print("\n=== HISTORY ===\n")
    print(row["history"])
    print("\n=== CURRENT QUERY ===\n")
    print(row["query"])
    print("\n=== STRUCTURED MEMORY ===\n")
    print(json.dumps(output["memory"], indent=2, ensure_ascii=False))
    print("\n=== DECISION ===\n")
    print("Gold:", row["action"])
    print("Predicted:", output["prediction"])
    print("Relation:", output["relation"])


if __name__ == "__main__":
    main()
