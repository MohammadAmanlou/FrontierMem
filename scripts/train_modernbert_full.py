#!/usr/bin/env python3
"""ModernBERT full parameter FP32 four-arm baseline, for Volta."""
from __future__ import annotations
import gc,json,sys
from pathlib import Path
import scripts.train_dual_axis_qwen as trainer
from scripts.eval_weekly_heldout import load_eval_rows,evaluate_one

MODEL="answerdotai/ModernBERT-base"

def setup(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if args.model!=MODEL or not torch.cuda.is_available(): raise RuntimeError("ModernBERT CUDA only")
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    tokenizer.padding_side="right"
    model=AutoModelForSequenceClassification.from_pretrained(
        args.model,revision=args.revision,num_labels=3,dtype=torch.float32,
        id2label={v:k for k,v in trainer.LABEL2ID.items()},label2id=trainer.LABEL2ID,
        attn_implementation="eager").to("cuda:0")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    return torch,model,tokenizer

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--arm",choices=trainer.ARMS,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--revision",required=True)
    a=p.parse_args()
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    assert torch.cuda.is_available()
    assert torch.isfinite(torch.ones(2,device="cuda").sum()).item()
    src=a.source; out=a.out; out.mkdir(parents=True,exist_ok=True)
    if not (out/"training_summary.json").is_file():
        trainer.setup_torch_model=setup
        trainer.main(["--model",MODEL,"--revision",a.revision,"--arm",a.arm,"--seed",str(a.seed),
            "--manifest",str(src/"data/local/dual_axis_plan/experiment_plan.json"),
            "--rpeval-pairs",str(src/"data/external_processed/caid_pairwise_train.jsonl"),
            "--s2pref-dir",str(src/"data/local/s2pref_candidates"),"--allow-silver-exploratory",
            "--epochs","1","--pairs-per-batch","2","--grad-accum","4",
            "--lambda-silver-binary","0.25","--lambda-context-rank","0.25",
            "--max-train-steps","0","--max-eval-pairs","0","--max-length","768","--save-adapter",
            "--output-dir",str(out)])
    summary=json.loads((out/"training_summary.json").read_text())
    if summary.get("model")!=MODEL or summary.get("model_revision")!=a.revision or summary.get("arm")!=a.arm or summary.get("seed")!=a.seed:
        raise RuntimeError("Checkpoint metadata mismatch")
    gc.collect();torch.cuda.empty_cache()
    # Shared trainer calls the saved directory best_adapter; here it contains
    # a FULL ModernBERT checkpoint, not a PEFT adapter.
    checkpoint=out/"best_adapter"
    tok=AutoTokenizer.from_pretrained(str(checkpoint))
    tok.padding_side="right"
    model=AutoModelForSequenceClassification.from_pretrained(str(checkpoint),dtype=torch.float32,attn_implementation="eager").to("cuda:0").eval()
    groups=load_eval_rows(src/"data/external_processed/all_applicability_examples.jsonl")
    parts=evaluate_one(model,tok,groups,2,768)
    dest=out/"official_eval";dest.mkdir(exist_ok=True)
    with (dest/"predictions.jsonl").open("w") as f:
        for part in parts.values():
            for row in part["predictions"]: f.write(json.dumps(row)+"\n")
    (dest/"metrics.json").write_text(json.dumps({"model":MODEL,"model_revision":a.revision,"arm":a.arm,"seed":a.seed,
        "training_method":"full_parameter_fp32","evaluation_status":"HELD_OUT_DECISION_LEVEL_UNCALIBRATED",
        "note":"Previously examined public test sets, BenchPreS decision proxy, S2Pref derived silver.",
        "sources":{k:v["metrics"] for k,v in parts.items()}},indent=2))
    (out/"PASS").write_text("training and evaluation complete\n")
if __name__=="__main__": main()
