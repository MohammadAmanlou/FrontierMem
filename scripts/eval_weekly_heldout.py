#!/usr/bin/env python3
"""Held-out decision-level evaluation for the matched four-arm FrontierMem run.

Only execute on a scheduled GPU. No held-out rows are used in training or
checkpoint selection. Preserve source-specific metrics; S2Pref silver
development scores are never described as benchmark gold.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import os

from frontiermem.dual_axis_protocol import ARMS, format_side
from frontiermem.external_data import normalize_rpeval_native_label

LABELS = ("IGNORE", "SUPPORT", "DOMINATE")
SOURCES = ("rpeval_explicit", "rpeval_implicit", "benchpres")
SEEDS = (17, 23, 42)


def f1_macro(gold, predictions, classes):
    if len(gold) != len(predictions) or not gold:
        raise ValueError("No data or unequal gold/prediction lengths")
    scores = []
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, predictions))
        fp = sum(g != c and p == c for g, p in zip(gold, predictions))
        fn = sum(g == c and p != c for g, p in zip(gold, predictions))
        scores.append((2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0)
    return sum(scores) / len(scores)


def safe_accuracy(gold, predicted):
    return sum(g == p for g, p in zip(gold, predicted)) / len(gold) if gold else None


def normalize_eval_text(row):
    # The processed BenchPreS adapter stores prompt twice as history and query.
    # Correct this duplication uniformly across all arms; do not inject gold metadata.
    return format_side({
        "preference": row.get("preference") or "",
        "query": row.get("query") or "",
        "history": "" if row.get("history") == row.get("query") else row.get("history") or "",
        "context": row.get("context") or "",
    })


def load_eval_rows(path):
    groups = {s: [] for s in SOURCES}
    with path.open(encoding="utf8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source")
            if source not in groups:
                continue
            expected_usage = "eval_only" if source == "benchpres" else "eval"
            if row.get("usage") != expected_usage:
                raise ValueError(f"Non-held-out row in {source}")
            if row.get("action") not in {"IGNORE", "APPLY"}:
                raise ValueError(f"Unknown action in {source}")
            if source.startswith("rpeval"):
                native = normalize_rpeval_native_label(row.get("source_label", ""))
                if (native == "IGNORE") != (row["action"] == "IGNORE"):
                    raise ValueError("Native label vs binary label mismatch")
            groups[source].append(row)
    expected = {"rpeval_explicit": 953, "rpeval_implicit": 953, "benchpres": 1950}
    observed = {source: len(v) for source, v in groups.items()}
    if observed != expected:
        raise ValueError(f"Unexpected protected evaluation sizes: {observed} expected {expected}")
    return groups


def sections_for_source(source, rows):
    if source == "benchpres":
        return {"benchpres": rows}
    # RPEval official source has 150 single-query and 150 multi-query records.
    single, multi = [], []
    for row in rows:
        n = (row.get("metadata") or {}).get("num_preferences", 1)
        (single if n == 1 else multi).append(row)
    if not single or not multi:
        raise ValueError(f"RPEval single/multi split missing: {source}")
    return {f"{source}_single": single, f"{source}_multi": multi}


def measure_part(rows, predictions, source, section):
    # Predictions keyed by example_id; one-to-one alignment is enforced.
    pred_rows = [predictions[r["example_id"]] for r in rows]
    gold_b = [r["action"] for r in rows]
    pred_b = [p["prediction_binary"] for p in pred_rows]
    ignored = [i for i, g in enumerate(gold_b) if g == "IGNORE"]
    by_group = defaultdict(list)
    for row, p in zip(rows, pred_rows):
        by_group[row["group_id"]].append((row, p))
    exact = sum(all(row["action"] == p["prediction_binary"] for row, p in g)
                for g in by_group.values()) / len(by_group)
    margins = []
    for group_rows in by_group.values():
        applied = [p["score_apply"] for r,p in group_rows if r["action"] == "APPLY"]
        ignored_scores = [p["score_apply"] for r,p in group_rows if r["action"] == "IGNORE"]
        margins.extend(a-b for a in applied for b in ignored_scores)
    out = {
        "source": source,
        "section": section,
        "n": len(rows),
        "binary_accuracy": safe_accuracy(gold_b, pred_b),
        "binary_macro_f1": f1_macro(gold_b, pred_b, ("IGNORE","APPLY")),
        "false_apply_rate_on_ignore": (
            sum(pred_b[i] == "APPLY" for i in ignored) / len(ignored)
            if ignored else None
        ),
        "full_query_binary_exact_match": exact,
        "within_query_contrast_n": len(margins),
        "within_query_all_pairs_direction_accuracy": (
            sum(m > 0 for m in margins) / len(margins) if margins else None
        ),
        "within_query_mean_margin": sum(margins)/len(margins) if margins else None,
        "binary_threshold": "apply_log_odds >= 0 (uncalibrated, fixed)",
        "evaluation_is_decision_level_proxy": source == "benchpres",
    }
    if source.startswith("rpeval"):
        gold = [normalize_rpeval_native_label(r["source_label"]) for r in rows]
        pred = [p["prediction_native"] for p in pred_rows]
        out.update(native_3way_accuracy=safe_accuracy(gold,pred),
                   native_3way_macro_f1=f1_macro(gold,pred,LABELS))
    return out


def load_adapter(base, adapter, tokenizer, revision=None):
    import torch
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
    from peft import PeftModel
    from scripts.train_dual_axis_qwen import ensure_classification_backend
    ensure_classification_backend(base)
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base, revision=revision, num_labels=3, quantization_config=quant, device_map={"":0},
        dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.get_text_config().pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    return model


def evaluate_one(model, tokenizer, groups, batch_size, max_length):
    import torch
    results = {}
    with torch.inference_mode():
        for source, rows in groups.items():
            predictions = {}
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start+batch_size]
                encoded = tokenizer(
                    [normalize_eval_text(r) for r in batch],
                    padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {k:v.to("cuda:0") for k,v in encoded.items()}
                logits = model(**encoded).logits.float()
                logodds = torch.logsumexp(logits[:,1:3],dim=-1)-logits[:,0]
                probs = torch.softmax(logits,dim=-1).cpu().tolist()
                scores = logodds.cpu().tolist()
                native = logits.argmax(-1).cpu().tolist()
                for i, row in enumerate(batch):
                    eid=row["example_id"]
                    if eid in predictions:
                        raise ValueError("Duplicate held-out example ID")
                    predictions[eid] = {
                        "example_id":eid, "group_id":row["group_id"],
                        "source":source, "gold_binary":row["action"],
                        "score_apply":scores[i],
                        "p_apply":probs[i][1]+probs[i][2],
                        "prediction_binary":"APPLY" if scores[i]>=0 else "IGNORE",
                        "prediction_native":LABELS[native[i]],
                    }
            source_metrics=[]
            for section, secrows in sections_for_source(source,rows).items():
                source_metrics.append(measure_part(secrows,predictions,source,section))
            # Also include source total, never pool unrelated sources together.
            source_metrics.append(measure_part(rows,predictions,source,source+"_all"))
            results[source] = {
                "metrics":source_metrics,
                "predictions":[predictions[row["example_id"]] for row in rows],
            }
    return results


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Official evaluation requires scheduled Slurm GPU")
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA + BF16 GPU required")
    groups=load_eval_rows(args.examples)
    expected=[(arm,seed) for seed in SEEDS for arm in ARMS]
    available=[]
    for arm,seed in expected:
        root=args.runs_root / f"seed_{seed}" / arm
        if (root/"training_summary.json").is_file() and (root/"best_adapter"/"adapter_config.json").is_file():
            available.append((arm,seed,root))
    if args.arm is not None:
        available = [v for v in available if v[0] == args.arm and v[1] == args.seed]
        if len(available) != 1:
            raise RuntimeError("Requested completed run not found")
    if args.require_all and len(available) != len(expected):
        raise RuntimeError(f"Only {len(available)}/{len(expected)} completed; refusing final held-out evaluation")
    for arm,seed,root in available:
        summary=json.loads((root/"training_summary.json").read_text())
        base = summary.get("model")
        revision = summary.get("model_revision")
        if not base or (args.model is not None and base != args.model) or summary.get("seed")!=seed or summary.get("arm")!=arm:
            raise ValueError(f"Run metadata mismatch: {root}")
        if args.revision is not None and revision != args.revision:
            raise ValueError(f"Revision mismatch: {root}")
        adapter_meta = json.loads((root/"best_adapter"/"adapter_config.json").read_text())
        if adapter_meta.get("base_model_name_or_path") != base:
            raise ValueError(f"Run metadata mismatch: {root}")
        if any(h.get("partial_training") or h.get("partial_evaluation") for h in summary.get("history",[])):
            raise ValueError(f"Partial training/evaluation is not the full experiment: {root}")
        out=root/"official_eval"
        if (out/"metrics.json").exists():
            print(f"Skip completed eval {arm} seed {seed}",flush=True)
            continue
        out.mkdir(parents=True,exist_ok=True)
        tok=AutoTokenizer.from_pretrained(str(root/"best_adapter"))
        if tok.pad_token is None:
            tok.pad_token=tok.eos_token
        tok.padding_side="right"
        model=load_adapter(base,root/"best_adapter",tok,revision)
        parts=evaluate_one(model,tok,groups,args.batch_size,args.max_length)
        metrics={"arm":arm,"seed":seed,"model":base,"model_revision":revision,
                 "evaluation_status":"HELD_OUT_DECISION_LEVEL_UNCALIBRATED",
                 "official_test_was_used_for_training_or_selection":False,
                 "note":"Published benchmark test examples were previously examined in the broader project; not a fresh blind confirmation.",
                 "sources":{k:v["metrics"] for k,v in parts.items()}}
        # Write predictions without raw texts. File may be large but remains on HPC.
        with (out/"predictions.jsonl").open("w",encoding="utf8") as file:
            for part in parts.values():
                for row in part["predictions"]:
                    file.write(json.dumps(row,ensure_ascii=False)+"\n")
        (out/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding="utf8")
        print(f"HELDOUT_EVAL_PASS arm={arm} seed={seed} {out}",flush=True)
        del model,tok,parts
        gc.collect()
        torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root",type=Path,default=Path("data/local/weekly_2026w48"))
    p.add_argument("--examples",type=Path,default=Path("data/external_processed/all_applicability_examples.jsonl"))
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--max-length",type=int,default=768)
    p.add_argument("--require-all",action="store_true")
    p.add_argument("--arm",choices=ARMS)
    p.add_argument("--seed",type=int,choices=SEEDS)
    p.add_argument("--model")
    p.add_argument("--revision")
    args=p.parse_args()
    if (args.arm is None) != (args.seed is None) or (args.require_all and args.arm is not None):
        p.error("--arm and --seed must be used together, not with --require-all")
    if args.batch_size<1 or args.max_length<128:
        p.error("batch-size must be positive; max-length >=128")
    run(args)


if __name__=="__main__":
    main()
