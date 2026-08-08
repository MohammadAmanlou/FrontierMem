#!/usr/bin/env python3
"""Aggregate FrontierMem result CSVs across local baselines and Qwen model sizes.

Usage:
    python scripts/compare_qwen_models.py --results results --output results/comparisons

The script is tolerant of missing runs. It never fabricates missing per-example
predictions. When only aggregate prediction counts are available (as in the
3B snapshot), it uses the explicit aggregate distribution CSV.
"""
from pathlib import Path
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt

MODES = ["gold_memory_llm", "rule_memory_llm", "llm_memory_rule", "full_llm"]
LABELS = ["APPLY", "IGNORE", "CLARIFY"]
MODEL_NAME_MAP = {
    "Qwen2_5-0_5B-Instruct": "Qwen2.5-0.5B-Instruct",
    "Qwen2_5-1_5B-Instruct": "Qwen2.5-1.5B-Instruct",
    "Qwen2_5-3B-Instruct": "Qwen2.5-3B-Instruct",
}


def pretty(slug: str) -> str:
    return MODEL_NAME_MAP.get(slug, slug.replace("_", "."))


def parse_name(filename: str, prefix: str):
    modes = "|".join(re.escape(m) for m in MODES)
    m = re.match(rf"^{re.escape(prefix)}_(.+)_({modes})\.csv$", filename)
    return m.groups() if m else None


def normalize_metric(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"metric", "score"}.issubset(df.columns):
        return (
            df[["metric", "score"]]
            .dropna(subset=["metric"])
            .drop_duplicates("metric")
            .set_index("metric")["score"]
            .to_frame()
            .T
            .reset_index(drop=True)
        )
    return df.iloc[[0]].copy() if len(df) else pd.DataFrame()


def build_metrics(results_dir: Path) -> pd.DataFrame:
    frames = []

    # New modular Qwen results kept at the root of results/.
    for p in sorted(results_dir.glob("metrics_Qwen*.csv")):
        parsed = parse_name(p.name, "metrics")
        if not parsed:
            continue
        slug, mode = parsed
        d = normalize_metric(p)
        if d.empty:
            continue
        d["model"] = pretty(slug)
        d["mode"] = mode
        d["source_file"] = p.name
        frames.append(d)

    # Legacy 0.5B result.
    legacy = results_dir / "metrics_qwen_small.csv"
    if legacy.exists():
        d = normalize_metric(legacy)
        if not d.empty:
            d["model"] = "Qwen2.5-0.5B-Instruct"
            d["mode"] = "legacy_full_llm"
            d["source_file"] = legacy.name
            frames.append(d)

    # Local classical/rule/tiny baselines.
    local = results_dir / "metrics_all_local.csv"
    if local.exists():
        d = pd.read_csv(local).copy()
        d["mode"] = "local_baseline"
        d["source_file"] = local.name
        frames.append(d)

    if not frames:
        return pd.DataFrame()

    allm = pd.concat(frames, ignore_index=True, sort=False)
    cols = [
        c
        for c in [
            "model", "mode", "accuracy", "balanced_accuracy", "macro_f1",
            "inside_accuracy", "outside_accuracy", "near_accuracy",
            "overgeneralization_rate", "unnecessary_clarification_rate",
            "twin_exact_match", "twin_discrimination_accuracy", "n_examples",
        ]
        if c in allm.columns
    ]
    return allm[cols].sort_values(["model", "mode"]).reset_index(drop=True)


def build_distributions(results_dir: Path) -> pd.DataFrame:
    rows = []
    represented = set()

    # New per-example Qwen prediction files at results/ root.
    for p in sorted(results_dir.glob("predictions_Qwen*.csv")):
        parsed = parse_name(p.name, "predictions")
        if not parsed:
            continue
        slug, mode = parsed
        d = pd.read_csv(p)
        if not {"action", "prediction"}.issubset(d.columns):
            continue
        model = pretty(slug)
        represented.add((model, mode))
        gold = d["action"].astype(str).str.strip().str.upper()
        pred = d["prediction"].astype(str).str.strip().str.upper()
        for label in LABELS:
            rows.append({
                "model": model,
                "mode": mode,
                "label": label,
                "gold_count": int((gold == label).sum()),
                "prediction_count": int((pred == label).sum()),
                "prediction_share": float((pred == label).mean()) if len(d) else 0.0,
            })

    # Legacy 0.5B per-example result.
    legacy = results_dir / "test_predictions_qwen_small.csv"
    if legacy.exists():
        d = pd.read_csv(legacy)
        if {"action", "prediction"}.issubset(d.columns):
            model, mode = "Qwen2.5-0.5B-Instruct", "legacy_full_llm"
            represented.add((model, mode))
            gold = d["action"].astype(str).str.strip().str.upper()
            pred = d["prediction"].astype(str).str.strip().str.upper()
            for label in LABELS:
                rows.append({
                    "model": model,
                    "mode": mode,
                    "label": label,
                    "gold_count": int((gold == label).sum()),
                    "prediction_count": int((pred == label).sum()),
                    "prediction_share": float((pred == label).mean()) if len(d) else 0.0,
                })

    # Aggregate-only prediction distributions (e.g. 3B snapshot).
    for p in sorted(results_dir.rglob("prediction_distribution_Qwen*.csv")):
        d = pd.read_csv(p)
        required = {"model", "mode", "label", "gold_count", "prediction_count", "prediction_share"}
        if not required.issubset(d.columns):
            continue
        for _, r in d.iterrows():
            key = (str(r["model"]), str(r["mode"]))
            if key in represented:
                continue
            rows.append({k: r[k] for k in required})
        for key in d[["model", "mode"]].drop_duplicates().itertuples(index=False, name=None):
            represented.add(tuple(map(str, key)))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)[
        ["model", "mode", "label", "gold_count", "prediction_count", "prediction_share"]
    ].sort_values(["model", "mode", "label"]).reset_index(drop=True)


def main(results_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = build_metrics(results_dir)
    if not metrics.empty:
        metrics.to_csv(out_dir / "all_model_metrics_raw.csv", index=False)
        pct = metrics.copy()
        for c in pct.columns:
            if c not in {"model", "mode", "n_examples"}:
                pct[c] = pd.to_numeric(pct[c], errors="coerce") * 100
        pct.to_csv(out_dir / "all_model_metrics_percent.csv", index=False)

        # Direct 3B-vs-1.5B comparison only on shared modes.
        delta_rows = []
        for mode in MODES:
            a = metrics[(metrics["model"] == "Qwen2.5-1.5B-Instruct") & (metrics["mode"] == mode)]
            b = metrics[(metrics["model"] == "Qwen2.5-3B-Instruct") & (metrics["mode"] == mode)]
            if a.empty or b.empty:
                continue
            a, b = a.iloc[0], b.iloc[0]
            row = {"mode": mode}
            for c in metrics.columns:
                if c in {"model", "mode", "n_examples"}:
                    continue
                av, bv = pd.to_numeric(a[c], errors="coerce"), pd.to_numeric(b[c], errors="coerce")
                row[f"1.5B_{c}"] = av
                row[f"3B_{c}"] = bv
                row[f"delta_pp_{c}"] = (bv - av) * 100
            delta_rows.append(row)
        if delta_rows:
            pd.DataFrame(delta_rows).to_csv(out_dir / "qwen_3b_vs_1_5b_deltas.csv", index=False)

        q = metrics[metrics["model"].astype(str).str.contains("Qwen", case=False, na=False)].copy()
        if len(q):
            q["run"] = q["model"] + "\n" + q["mode"]
            plotcols = [c for c in ["accuracy", "balanced_accuracy", "macro_f1"] if c in q.columns]
            if plotcols:
                ax = q.set_index("run")[plotcols].plot(kind="bar", figsize=(14, 7))
                ax.set_ylim(0, 1)
                ax.set_ylabel("Score")
                ax.set_title("FrontierMem: Qwen model comparison")
                ax.tick_params(axis="x", rotation=30)
                plt.tight_layout()
                plt.savefig(out_dir / "qwen_all_models_main_metrics.png", dpi=220, bbox_inches="tight")
                plt.close()
            boundary = [c for c in ["inside_accuracy", "outside_accuracy", "near_accuracy"] if c in q.columns]
            if boundary:
                ax = q.set_index("run")[boundary].plot(kind="bar", figsize=(14, 7))
                ax.set_ylim(0, 1)
                ax.set_ylabel("Accuracy")
                ax.set_title("FrontierMem boundary accuracy")
                ax.tick_params(axis="x", rotation=30)
                plt.tight_layout()
                plt.savefig(out_dir / "qwen_all_models_boundary_metrics.png", dpi=220, bbox_inches="tight")
                plt.close()

    dist = build_distributions(results_dir)
    if not dist.empty:
        dist.to_csv(out_dir / "all_prediction_distributions.csv", index=False)
        collapse = []
        for (model, mode), g in dist.groupby(["model", "mode"]):
            top = g.loc[g["prediction_count"].astype(float).idxmax()]
            n = int(pd.to_numeric(g["gold_count"], errors="coerce").sum())
            share = float(top["prediction_share"])
            collapse.append({
                "model": model,
                "mode": mode,
                "n_examples": n,
                "dominant_prediction": top["label"],
                "dominant_prediction_count": int(top["prediction_count"]),
                "dominant_prediction_share": share,
                "possible_collapse": share >= 0.90,
            })
        pd.DataFrame(collapse).to_csv(out_dir / "all_collapse_analysis.csv", index=False)

        piv = (
            dist.pivot_table(index=["model", "mode"], columns="label", values="prediction_share", fill_value=0)
            .reindex(columns=LABELS, fill_value=0)
        )
        ax = piv.plot(kind="bar", stacked=True, figsize=(14, 7))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Prediction share")
        ax.set_title("Prediction distribution by model and mode")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(out_dir / "all_prediction_distributions.png", dpi=220, bbox_inches="tight")
        plt.close()

    print(f"Wrote comparison artifacts to: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--output", type=Path, default=Path("results/comparisons"))
    args = ap.parse_args()
    main(args.results, args.output)
