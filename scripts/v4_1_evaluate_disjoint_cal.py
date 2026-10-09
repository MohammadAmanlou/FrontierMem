#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontiermem.external_data import load_jsonl, normalize_rpeval_native_label
from frontiermem.contrast_sets import load_pairs
from frontiermem.identifiability import SplitConformalAbstainer, selective_metrics

LABELS = ["IGNORE", "SUPPORT", "DOMINATE"]
LABEL2ID = {"IGNORE": 0, "SUPPORT": 1, "DOMINATE": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def format_row(row: dict) -> str:
    parts = [f"[PREFERENCE]\n{row.get('preference', '')}"]
    history = str(row.get("history", ""))
    context = str(row.get("context", ""))
    if history.strip():
        parts.append(f"[HISTORY]\n{history}")
    if context.strip():
        parts.append(f"[CONTEXT]\n{context}")
    parts.append(f"[QUERY]\n{row.get('query', '')}")
    parts.append(
        "[TASK]\nJudge the preference-query relation using the native RPEval "
        "labels IGNORE, SUPPORT, or DOMINATE."
    )
    parts.append("[DECISION]")
    return "\n".join(parts)


def app_score_np(z):
    m = np.maximum(z[:, 1], z[:, 2])
    lse = m + np.log(np.exp(z[:, 1] - m) + np.exp(z[:, 2] - m))
    return lse - z[:, 0]


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


def load_model(checkpoint):
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
    from peft import PeftConfig, PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA in this v4 notebook.")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    peft_cfg = PeftConfig.from_pretrained(checkpoint)
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        peft_cfg.base_model_name_or_path,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        quantization_config=qconfig,
        device_map={"": 0},
        dtype=compute_dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    model.config.use_cache = False
    return model


@torch.no_grad()
def logits_rows(model, tokenizer, rows, batch_size, max_length):
    logits_all = []
    model.eval()
    device = torch.device("cuda:0")
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


def aar_mr(gold, pred):
    gold = np.asarray(gold)
    pred = np.asarray(pred)
    pos = gold == "APPLY"
    neg = gold == "IGNORE"
    aar = float(np.mean(pred[pos] == "APPLY")) if pos.any() else float("nan")
    mr = float(np.mean(pred[neg] == "APPLY")) if neg.any() else float("nan")
    return aar, mr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--examples", default="data/external_processed/all_applicability_examples.jsonl")
    p.add_argument("--pairs", default="data/external_processed/query_preference_contrast_pairs.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    args = p.parse_args()

    from transformers import AutoTokenizer

    rows = [r for r in load_jsonl(args.examples) if r.get("action") in {"APPLY", "IGNORE"}]
    calibration = [r for r in rows if r.get("usage") == "calibration"]
    evaluation = [r for r in rows if r.get("usage") in {"eval", "eval_only"}]
    if not calibration:
        raise SystemExit("No leakage-safe calibration split found.")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = load_model(args.checkpoint)
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.pad_token_id = tokenizer.pad_token_id

    cal_logits = logits_rows(model, tokenizer, calibration, args.batch_size, args.max_length)
    cal_probs = softmax(cal_logits)
    cal_score = app_score_np(cal_logits)
    y_cal = np.asarray([1 if r["action"] == "APPLY" else 0 for r in calibration])

    # IMPORTANT: Platt and conformal MUST use disjoint calibration groups.
    # This avoids reusing the same labels to fit the calibrator and estimate qhat.
    import random
    group_keys = [str(r.get("group_id") or r.get("example_id")) for r in calibration]
    all_groups = sorted(set(group_keys))
    if len(all_groups) < 4:
        raise RuntimeError("Insufficient calibration groups for group-disjoint split")
    random.Random(1729).shuffle(all_groups)
    platt_groups = set(all_groups[:max(1, len(all_groups) // 2)])
    ix_platt = np.array([i for i, g in enumerate(group_keys) if g in platt_groups], dtype=int)
    ix_conformal = np.array([i for i, g in enumerate(group_keys) if g not in platt_groups], dtype=int)
    if min(len(ix_platt), len(ix_conformal)) == 0:
        raise RuntimeError("Calibration split empty")
    if len(set(y_cal[ix_platt].tolist())) < 2 or len(set(y_cal[ix_conformal].tolist())) < 2:
        raise RuntimeError("Each calibration partition must contain both classes")
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=17)
    platt.fit(cal_score[ix_platt].reshape(-1, 1), y_cal[ix_platt])
    p_apply_conformal = platt.predict_proba(cal_score[ix_conformal].reshape(-1, 1))[:, 1]
    abstainer = SplitConformalAbstainer(alpha=args.alpha).fit(
        p_apply_conformal, y_cal[ix_conformal]
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(
        json.dumps({
            "platt_coef": float(platt.coef_[0, 0]),
            "platt_intercept": float(platt.intercept_[0]),
            "conformal_alpha": args.alpha,
            "conformal_qhat": float(abstainer.qhat),
            "calibration_n": len(calibration),
            "platt_fit_n": len(ix_platt),
            "conformal_fit_n": len(ix_conformal),
            "calibration_split": "group-disjoint, randomized seed 1729",
            "calibration_brier": float(brier_score_loss(y_cal[ix_conformal], p_apply_conformal)),
        }, indent=2),
        encoding="utf-8",
    )

    metric_rows, pred_rows = [], []

    for source in sorted({r["source"] for r in evaluation}):
        part = [r for r in evaluation if r["source"] == source]
        logits = logits_rows(model, tokenizer, part, args.batch_size, args.max_length)
        probs = softmax(logits)
        raw_p_apply = probs[:, 1] + probs[:, 2]
        score = app_score_np(logits)
        calibrated_p_apply = platt.predict_proba(score.reshape(-1, 1))[:, 1]

        raw_forced = np.where(raw_p_apply >= 0.5, "APPLY", "IGNORE")
        cal_forced = np.where(calibrated_p_apply >= 0.5, "APPLY", "IGNORE")
        selective = np.asarray(abstainer.predict(calibrated_p_apply))
        gold_binary = np.asarray([r["action"] for r in part])
        y_binary = (gold_binary == "APPLY").astype(int)

        raw_aar, raw_mr = aar_mr(gold_binary, raw_forced)
        cal_aar, cal_mr = aar_mr(gold_binary, cal_forced)

        record = {
            "source": source,
            "n": len(part),
            "alpha": args.alpha,
            "qhat": float(abstainer.qhat),
            "raw_binary_accuracy": float(accuracy_score(gold_binary, raw_forced)),
            "raw_binary_macro_f1": float(
                f1_score(gold_binary, raw_forced, labels=["APPLY", "IGNORE"], average="macro", zero_division=0)
            ),
            "raw_AAR": raw_aar,
            "raw_MR": raw_mr,
            "raw_brier": float(brier_score_loss(y_binary, raw_p_apply)),
            "cal_binary_accuracy": float(accuracy_score(gold_binary, cal_forced)),
            "cal_binary_macro_f1": float(
                f1_score(gold_binary, cal_forced, labels=["APPLY", "IGNORE"], average="macro", zero_division=0)
            ),
            "cal_AAR": cal_aar,
            "cal_MR": cal_mr,
            "cal_brier": float(brier_score_loss(y_binary, calibrated_p_apply)),
            **{f"selective_{k}": v for k, v in selective_metrics(gold_binary, selective).items()},
        }

        native_gold = [native_label(r) for r in part]
        native_idx = [i for i, y in enumerate(native_gold) if y is not None]
        if native_idx:
            native_true = [native_gold[i] for i in native_idx]
            native_pred = [LABELS[int(logits[i].argmax())] for i in native_idx]
            record["native_3way_accuracy"] = float(accuracy_score(native_true, native_pred))
            record["native_3way_macro_f1"] = float(
                f1_score(native_true, native_pred, labels=LABELS, average="macro", zero_division=0)
            )
            record["native_n"] = len(native_idx)

        metric_rows.append(record)

        for i, r in enumerate(part):
            pred_rows.append({
                "example_id": r["example_id"],
                "source": source,
                "gold_binary": r["action"],
                "gold_native": native_label(r) or "",
                "applicability_score": float(score[i]),
                "p_ignore": float(probs[i, 0]),
                "p_support": float(probs[i, 1]),
                "p_dominate": float(probs[i, 2]),
                "raw_p_apply": float(raw_p_apply[i]),
                "calibrated_p_apply": float(calibrated_p_apply[i]),
                "raw_forced_prediction": str(raw_forced[i]),
                "cal_forced_prediction": str(cal_forced[i]),
                "selective_prediction": str(selective[i]),
                "native_prediction": LABELS[int(logits[i].argmax())],
                "preference": r["preference"],
                "query": r["query"],
            })

    pair_metric_rows = []
    pair_path = Path(args.pairs)
    if pair_path.exists():
        pairs = load_pairs(pair_path)

        def side_rows(part, prefix):
            return [{
                "preference": p.get(f"{prefix}_preference", ""),
                "history": p.get(f"{prefix}_history", ""),
                "context": p.get(f"{prefix}_context", ""),
                "query": p.get(f"{prefix}_query", ""),
            } for p in part]

        for usage in sorted({str(p.get("usage", "")) for p in pairs}):
            usage_rows = [p for p in pairs if str(p.get("usage", "")) == usage]
            for source in sorted({str(p.get("source", "")) for p in usage_rows}):
                part = [p for p in usage_rows if str(p.get("source", "")) == source]
                if not part:
                    continue
                lp = logits_rows(model, tokenizer, side_rows(part, "positive"), args.batch_size, args.max_length)
                ln = logits_rows(model, tokenizer, side_rows(part, "negative"), args.batch_size, args.max_length)
                margin = app_score_np(lp) - app_score_np(ln)
                pair_metric_rows.append({
                    "usage": usage,
                    "source": source,
                    "n_pairs": len(part),
                    "direction_accuracy": float((margin > 0).mean()),
                    "mean_margin": float(margin.mean()),
                    "median_margin": float(np.median(margin)),
                })

    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(pair_metric_rows).to_csv(out / "pair_metrics_by_split_source.csv", index=False)

    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nPAIR METRICS")
    print(pd.DataFrame(pair_metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
