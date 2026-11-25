#!/usr/bin/env python3
"""Summarize available weekly training/held-out metrics without inventing results.

Runs on CPU and works with incomplete stages. Does not touch datasets or GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ARMS = ("pointwise","caid","caid_s2pref","dual_axis")
SEEDS = (17,23,42)


def metric_value(record, key):
    return record.get(key) if isinstance(record, dict) else None


def collect(root):
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            run = root / f"seed_{seed}" / arm
            item = {"seed":seed,"arm":arm,"model":None,"model_revision":None,"train_status":"missing",
                    "rpeval_dev_binary_macro_f1":None,
                    "rpeval_dev_native_macro_f1":None,
                    "rpeval_dev_pair_direction_accuracy":None,
                    "s2pref_silver_strict_unseen_binary_macro_f1":None,
                    "s2pref_silver_strict_unseen_pair_direction_accuracy":None,
                    "step_count":None,"training_seconds":None,
                    "partial_training":None,"official_eval_status":"missing"}
            summary=run/"training_summary.json"
            if summary.is_file():
                r=json.loads(summary.read_text(encoding="utf8"))
                if r.get("arm")!=arm or r.get("seed")!=seed:
                    raise ValueError(f"Run metadata mismatch in {summary}")
                history=r.get("history",[])
                if not history:
                    raise ValueError(f"Missing history in {summary}")
                chosen=next((h for h in history if h["epoch"]==r.get("best_epoch")),history[-1])
                rv=chosen.get("rpeval_dev",{})
                sv=chosen.get("s2pref_silver_dev",{})
                item.update(
                    train_status="completed",
                    model=r.get("model"), model_revision=r.get("model_revision"),
                    rpeval_dev_binary_macro_f1=rv.get("binary_macro_f1"),
                    rpeval_dev_native_macro_f1=rv.get("native_macro_f1"),
                    rpeval_dev_pair_direction_accuracy=rv.get("pair_direction_accuracy"),
                    s2pref_silver_strict_unseen_binary_macro_f1=sv.get("strict_unseen_binary_macro_f1"),
                    s2pref_silver_strict_unseen_pair_direction_accuracy=sv.get("strict_unseen_pair_direction_accuracy"),
                    step_count=sum(h.get("rpeval_batches",0) for h in history),
                    training_seconds=sum(h.get("seconds",0) for h in history),
                    partial_training=any(h.get("partial_training",False) or h.get("partial_evaluation",False) for h in history),
                )
            held=run/"official_eval"/"metrics.json"
            if held.is_file():
                e=json.loads(held.read_text(encoding="utf8"))
                if e.get("arm")!=arm or e.get("seed")!=seed or e.get("model")!=item["model"] or e.get("model_revision")!=item["model_revision"]:
                    raise ValueError(f"Official eval mismatch {held}")
                item["official_eval_status"]="completed"
                for source,section in (
                    ("rpeval_explicit","rpeval_explicit_all"),
                    ("rpeval_implicit","rpeval_implicit_all"),
                    ("benchpres","benchpres")
                ):
                    sections=e.get("sources",{}).get(source,[])
                    d=next((x for x in sections if x.get("section")==section),{})
                    item[f"{source}_binary_macro_f1"]=d.get("binary_macro_f1")
                    item[f"{source}_native_3way_macro_f1"]=d.get("native_3way_macro_f1")
                    item[f"{source}_wrong_apply_rate"]=d.get("false_apply_rate_on_ignore")
                for key in ("rpeval_explicit_single","rpeval_explicit_multi",
                            "rpeval_implicit_single","rpeval_implicit_multi"):
                    src=key.rsplit("_",1)[0]
                    sections=e.get("sources",{}).get(src,[])
                    d=next((x for x in sections if x.get("section")==key),{})
                    item[key+"_binary_macro_f1"]=d.get("binary_macro_f1")
                    item[key+"_native_macro_f1"]=d.get("native_3way_macro_f1")
            rows.append(item)
    return rows


def render_markdown(rows):
    lines=[
        "# FrontierMem — Weekly experimental results (generated)",
        "",
        "This is an automatically generated status report, **not** a complete research paper.",
        "Training: one full RPEval epoch, 4 methods × 3 seeds per model. Model: " + ", ".join(sorted({r["model"] for r in rows if r.get("model")})) + ".",
        "S2Pref candidate training labels are **derived silver, not human-adjudicated gold**.",
        "Native RPEval test sets and BenchPreS are evaluation-only; BenchPreS metrics are decision-level proxies.",
        "The official benchmark sets were examined in earlier research, so they are not fresh blind confirmation.",
        "No full end-to-end structured memory, real-user history sufficiency, or response generation result is implied.",
        "",
    ]
    train=sum(r["train_status"]=="completed" for r in rows)
    test=sum(r["official_eval_status"]=="completed" for r in rows)
    lines += [f"**Training runs complete:** {train}/12. **Held-out evaluations complete:** {test}/12.", ""]
    lines += ["| Arm | Seed | Training | RPEval dev binary Macro-F1 | RPEval dev 3-way Macro-F1 | Silver strict-unseen Macro-F1 | Official BenchPreS binary Macro-F1 |",
              "|---|---:|---|---:|---:|---:|---:|"]
    def fmt(x):
        return f"{x:.4f}" if isinstance(x,(int,float)) else "—"
    for r in rows:
        lines.append("| "+ " | ".join([
            r["arm"],str(r["seed"]),r["train_status"],
            fmt(r["rpeval_dev_binary_macro_f1"]),
            fmt(r["rpeval_dev_native_macro_f1"]),
            fmt(r["s2pref_silver_strict_unseen_binary_macro_f1"]),
            fmt(r.get("benchpres_binary_macro_f1")),
        ])+" |")
    lines.extend(["","## Scope and limitations",
                  "Dev comparisons are valid only as exploratory model-selection diagnostics. "
                  "Full benchmark claims require validating source-specific error modes, "
                  "independent semantic review of silver labels, and replicated significance checks.",
                  "Missing metrics are shown as dashes; never interpret as zero or a completed run.",
                  "Additional source-specific and single/multi-stratum metrics are in weekly_metrics.csv.",
                  ""])
    return "\n".join(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root",type=Path,default=Path("data/local/weekly_2026w48"))
    ap.add_argument("--output-dir",type=Path,default=Path("data/local/weekly_2026w48/meeting"))
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=collect(args.runs_root)
    fields=list(dict.fromkeys(k for item in rows for k in item))
    with (args.output_dir/"weekly_metrics.csv").open("w",newline="",encoding="utf8") as fp:
        writer=csv.DictWriter(fp,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir/"meeting_summary.md").write_text(render_markdown(rows),encoding="utf8")
    state={
        "training_completed":sum(r["train_status"]=="completed" for r in rows),
        "heldout_completed":sum(r["official_eval_status"]=="completed" for r in rows),
        "expected":12,
        "meeting_ready_full":all(r["official_eval_status"]=="completed" for r in rows),
        "silver_label_status":"unreviewed derived silver",
    }
    (args.output_dir/"status.json").write_text(json.dumps(state,indent=2),encoding="utf8")
    print(json.dumps(state,indent=2),flush=True)
    print("Saved: "+str(args.output_dir/"meeting_summary.md"),flush=True)


if __name__=="__main__":
    main()
