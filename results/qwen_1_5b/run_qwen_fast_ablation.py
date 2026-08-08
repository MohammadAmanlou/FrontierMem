from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from frontiermem.data import generate_frontier_suite, save_frontier_suite
from frontiermem.evaluation import evaluate_predictions, extraction_metrics
from frontiermem.hf_small_llm import HFGenerationConfig, HFSmallLLMFrontierMem
from frontiermem.memory import MemoryItem, RuleBasedMemoryExtractor, RuleBasedApplicabilityPolicy, render_response

ROOT = Path(__file__).resolve().parent
LABELS = ("APPLY", "IGNORE", "CLARIFY")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--mode", choices=[
        "gold_memory_llm",
        "rule_memory_llm",
        "llm_memory_rule",
        "full_llm",
    ], default="gold_memory_llm")
    p.add_argument("--max-examples", type=int, default=72)
    p.add_argument("--device", choices=["auto", "cpu"], default="auto")
    return p.parse_args()


def balanced_subset(df: pd.DataFrame, max_examples: int) -> pd.DataFrame:
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    per_cell = max(1, max_examples // (test_df["family"].nunique() * 3))
    return (
        test_df.groupby(["family", "action"], group_keys=False)
        .head(per_cell)
        .head(max_examples)
        .reset_index(drop=True)
    )


def history_evidence(history: str) -> list[str]:
    lines = [re.sub(r"^Session\s+\d+:\s*", "", x).strip() for x in history.splitlines() if x.strip()]
    return lines[:3]


def gold_memory(row) -> MemoryItem:
    return MemoryItem(
        preference=str(row.preference_label),
        family=str(row.family),
        owner=str(row.owner),
        information_type=str(row.information_type),
        applies_when=str(row.applies_when),
        does_not_apply_when=str(row.does_not_apply_when),
        temporal_validity=str(row.temporal_validity),
        confidence=1.0,
        evidence=history_evidence(str(row.history)),
        permission="response_personalization",
        profile_variant=str(row.profile_variant),
    )


def direct_action(backend: HFSmallLLMFrontierMem, memory: MemoryItem, query: str):
    system = """You are a deterministic classifier for preference applicability.
Choose exactly one label and output only that label: APPLY, IGNORE, or CLARIFY.
Rules:
1. A stable preference is APPLY unless the current query explicitly overrides or corrects it.
2. A scoped preference is APPLY when the query satisfies applies_when.
3. A scoped preference is IGNORE when the query satisfies does_not_apply_when.
4. Use CLARIFY only when a material fact needed to distinguish APPLY from IGNORE is genuinely missing.
Do not explain your answer."""

    user = f"""Examples:
Memory: profile_variant=stable; applies_when=broadly; does_not_apply_when=explicit override only.
Query: The company is paying, but the user consistently prefers inexpensive hotels.
Answer: APPLY

Memory: profile_variant=scoped; applies_when=self-funded trip; does_not_apply_when=fully reimbursed without a cap.
Query: The company covers the full hotel cost with no cap.
Answer: IGNORE

Memory: profile_variant=scoped; applies_when=self-funded trip; does_not_apply_when=fully reimbursed without a cap.
Query: I have not checked who will pay.
Answer: CLARIFY

Now classify this case.
MEMORY:
{memory.to_json()}

CURRENT QUERY:
{query}

Answer:"""

    old_max = backend.config.max_new_tokens
    backend.config.max_new_tokens = 8
    raw = backend._chat(system, user).strip().upper()
    backend.config.max_new_tokens = old_max

    match = re.search(r"\b(APPLY|IGNORE|CLARIFY)\b", raw)
    if match:
        return match.group(1), raw, "llm_label"
    return "CLARIFY", raw, "parse_fallback"


def main():
    args = parse_args()
    data_path = ROOT / "data" / "frontier_suite_mini.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        df = generate_frontier_suite()
        save_frontier_suite(df, data_path)
    subset = balanced_subset(df, args.max_examples)

    backend = HFSmallLLMFrontierMem(
        HFGenerationConfig(model_name=args.model, max_new_tokens=280, temperature=0.0),
        device_map=args.device,
    )
    rule_extractor = RuleBasedMemoryExtractor()
    rule_policy = RuleBasedApplicabilityPolicy()

    rows = []
    for i, row in enumerate(subset.itertuples(), start=1):
        print(f"[{i}/{len(subset)}] mode={args.mode} family={row.family} gold={row.action}", flush=True)

        if args.mode == "gold_memory_llm":
            memory = gold_memory(row)
            action, raw, parse_mode = direct_action(backend, memory, row.query)
            relation = "not_evaluated"
            confidence = 1.0
            response = ""
            memory_parse = "gold"
            memory_raw = ""

        elif args.mode == "rule_memory_llm":
            memory = rule_extractor.extract(row.history, row.query)
            action, raw, parse_mode = direct_action(backend, memory, row.query)
            relation = "not_evaluated"
            confidence = memory.confidence
            response = ""
            memory_parse = "rule"
            memory_raw = ""

        elif args.mode == "llm_memory_rule":
            memory, memory_parse, memory_raw = backend.extract_memory(row.history, row.query)
            action, relation, confidence = rule_policy.decide(memory, row.query)
            response = render_response(action, memory, row.query)
            raw = ""
            parse_mode = "rule_policy"

        else:
            output = backend.predict_one(row.history, row.query)
            memory = MemoryItem(**output["memory"])
            action = output["action"]
            relation = output["relation"]
            confidence = output["confidence"]
            response = output["response"]
            raw = output["raw_output"]
            parse_mode = output["decision_parse_mode"]
            memory_parse = output["memory_parse_mode"]
            memory_raw = output["memory_raw_output"]

        item = row._asdict()
        item.update({
            "prediction": action,
            "pred_relation": relation,
            "decision_confidence": confidence,
            "generated_response": response,
            "decision_raw_output": raw,
            "decision_parse_mode": parse_mode,
            "structured_memory": json.dumps(memory.to_dict(), ensure_ascii=False),
            "memory_parse_mode": memory_parse,
            "memory_raw_output": memory_raw,
            "pred_family": memory.family,
            "pred_profile_variant": memory.profile_variant,
            "pred_owner": memory.owner,
            "pred_information_type": memory.information_type,
            "pred_temporal_validity": memory.temporal_validity,
        })
        rows.append(item)

    pred = pd.DataFrame(rows)
    model_label = f"{args.model.split('/')[-1]}::{args.mode}"
    metrics = pd.DataFrame([evaluate_predictions(pred, model_label)])

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    slug = args.model.split("/")[-1].replace(".", "_")
    pred_path = results / f"predictions_{slug}_{args.mode}.csv"
    met_path = results / f"metrics_{slug}_{args.mode}.csv"
    pred.to_csv(pred_path, index=False)
    metrics.to_csv(met_path, index=False)

    print("\nMetrics")
    print(metrics.round(4).to_string(index=False))
    print("\nPrediction distribution")
    print(pred["prediction"].value_counts().to_string())
    print(f"\nSaved: {pred_path}\nSaved: {met_path}")

    if args.mode in {"llm_memory_rule", "full_llm"}:
        ext = extraction_metrics(pred)
        ext_path = results / f"extraction_{slug}_{args.mode}.csv"
        ext.to_csv(ext_path, index=False)
        print("\nExtraction metrics")
        print(ext.round(4).to_string(index=False))
        print(f"Saved: {ext_path}")


if __name__ == "__main__":
    main()
