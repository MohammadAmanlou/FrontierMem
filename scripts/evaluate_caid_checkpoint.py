#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontiermem.external_data import load_jsonl, normalize_rpeval_native_label
from frontiermem.identifiability import SplitConformalAbstainer, selective_metrics

LABELS = ["IGNORE", "SUPPORT", "DOMINATE"]


def format_row(row: dict) -> str:
    return (
        f"[PREFERENCE]\n{row.get('preference', '')}\n"
        f"[HISTORY]\n{row.get('history', '')}\n"
        f"[CONTEXT]\n{row.get('context', '')}\n"
        f"[QUERY]\n{row.get('query', '')}\n"
        "[QUESTION]\nHow strongly should this preference influence the response?"
    )


def logits_rows(model, tokenizer, rows, device, batch_size, max_length):
    logits_all = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            toks = tokenizer(
                [format_row(r) for r in chunk],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            logits_all.append(model(**toks).logits.float().cpu().numpy())
    return np.concatenate(logits_all, axis=0)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def native_label(row):
    if not str(row.get("source", "")).startswith("rpeval"):
        return None
    try:
        return normalize_rpeval_native_label(row.get("source_label", ""))
    except ValueError:
        return None


def load_model(checkpoint, dtype):
    from transformers import AutoModelForSequenceClassification
    adapter_cfg = Path(checkpoint) / "adapter_config.json"
    if adapter_cfg.exists():
        from peft import PeftModel
        peft_cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        base_name = peft_cfg["base_model_name_or_path"]
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_name,
            num_labels=3,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        return PeftModel.from_pretrained(base_model, checkpoint)
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint, torch_dtype=dtype, trust_remote_code=True
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a 3-way RPEval applicability checkpoint. Reports native "
            "IGNORE/SUPPORT/DOMINATE metrics plus binary/selective applicability "
            "metrics for cross-benchmark comparison."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--examples",
        default="data/external_processed/all_applicability_examples.jsonl",
    )
    parser.add_argument("--output-dir", default="results/caid_checkpoint_eval")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    rows = [r for r in load_jsonl(args.examples) if r.get("action") in {"APPLY", "IGNORE"}]
    calibration = [r for r in rows if r.get("usage") == "calibration"]
    if not calibration:
        raise SystemExit(
            "No leakage-safe calibration split found. Rerun prepare_external_benchmarks.py."
        )
    evaluation = [r for r in rows if r.get("usage") in {"eval", "eval_only"}]

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = load_model(args.checkpoint, dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cal_logits = logits_rows(model, tokenizer, calibration, device, args.batch_size, args.max_length)
    cal_probs = softmax(cal_logits)
    p_apply_cal = cal_probs[:, 1] + cal_probs[:, 2]
    y_cal = [1 if r["action"] == "APPLY" else 0 for r in calibration]
    abstainer = SplitConformalAbstainer(alpha=args.alpha).fit(p_apply_cal, y_cal)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metric_rows, pred_rows = [], []

    for source in sorted({r["source"] for r in evaluation}):
        part = [r for r in evaluation if r["source"] == source]
        logits = logits_rows(model, tokenizer, part, device, args.batch_size, args.max_length)
        probs = softmax(logits)
        p_apply = probs[:, 1] + probs[:, 2]
        selective = abstainer.predict(p_apply)
        forced_binary = ["APPLY" if p >= 0.5 else "IGNORE" for p in p_apply]
        gold_binary = [r["action"] for r in part]

        record = {
            "source": source,
            "alpha": args.alpha,
            "qhat": abstainer.qhat,
            "forced_binary_accuracy": float(accuracy_score(gold_binary, forced_binary)),
            "forced_binary_macro_f1": float(
                f1_score(gold_binary, forced_binary, labels=["APPLY", "IGNORE"], average="macro")
            ),
            **{f"selective_{k}": v for k, v in selective_metrics(gold_binary, selective).items()},
        }

        native_gold = [native_label(r) for r in part]
        native_idx = [i for i, y in enumerate(native_gold) if y is not None]
        native_pred = [LABELS[int(logits[i].argmax())] for i in native_idx]
        native_true = [native_gold[i] for i in native_idx]
        if native_idx:
            record["native_3way_accuracy"] = float(accuracy_score(native_true, native_pred))
            record["native_3way_macro_f1"] = float(
                f1_score(native_true, native_pred, labels=LABELS, average="macro")
            )
            record["native_n"] = len(native_idx)

        metric_rows.append(record)

        for i, (r, s_pred, b_pred) in enumerate(zip(part, selective, forced_binary)):
            pred_rows.append(
                {
                    "example_id": r["example_id"],
                    "source": source,
                    "gold_binary": r["action"],
                    "gold_native": native_label(r) or "",
                    "p_ignore": float(probs[i, 0]),
                    "p_support": float(probs[i, 1]),
                    "p_dominate": float(probs[i, 2]),
                    "p_apply": float(p_apply[i]),
                    "forced_binary_prediction": b_pred,
                    "selective_prediction": s_pred,
                    "native_prediction": LABELS[int(logits[i].argmax())],
                    "preference": r["preference"],
                    "query": r["query"],
                }
            )

    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "predictions.csv", index=False)
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
