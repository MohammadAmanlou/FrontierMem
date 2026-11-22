#!/usr/bin/env python3
"""GPU-only matched four-arm Qwen3-4B applicability trainer (experimental).

A CPU --dry-run verifies inputs and never imports torch or loads a model.
Real training MUST run in an allocated GPU Slurm job. Never train on login.
S2Pref is derived silver, never three-way RPEval gold; use --allow-silver-
exploratory to explicitly acknowledge that it is NOT human-approved evidence.

Training:
  - All arms see the same RPEval train pairs and sequence order.
  - Pointwise, CAID share CE3+BCE2; only CAID adds preference ranking.
  - CAID+S2Pref and Dual-Axis see exactly the same additional silver pairs;
    only Dual-Axis adds context ranking; no invented native S2Pref labels.
  - One uniformly distributed S2Pref pair is consumed at selected RPEval
    steps, exactly once per full epoch. The paired enhanced arms have identical
    sample exposure and optimizer-step budgets.
  - Selection uses RPEval dev macro-F1 for ALL arms. S2Pref dev including its
    strictly unseen subset is diagnostic, never native benchmark gold.

This is a research prototype; not validated full training until a Slurm GPU
smoke and checkpoint reload both complete.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from frontiermem.dual_axis_protocol import (
    ARMS, BINARY_ARMS, format_side, load_and_verify, summary,
)

LABEL2ID = {"IGNORE": 0, "SUPPORT": 1, "DOMINATE": 2}


def make_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def silver_index_for_step(step: int, total_rpeval_steps: int, total_silver: int):
    """Spread each silver pair over the RPEval epoch without oversampling."""
    if total_silver <= 0 or total_rpeval_steps <= 0:
        return None
    # Can spread without repeating only if enough RPEval batches exist.
    if total_silver > total_rpeval_steps:
        raise ValueError("Silver train pairs exceed RPEval batches; increase RPEval steps or revise schedule")
    index = (step * total_silver) // total_rpeval_steps
    previous = ((step - 1) * total_silver) // total_rpeval_steps if step else -1
    return index if index != previous else None


def macro_f1(y_true, y_pred, classes):
    if not y_true:
        return 0.0
    total = 0.0
    for c in classes:
        tp = sum(a == c and b == c for a, b in zip(y_true, y_pred))
        fp = sum(a != c and b == c for a, b in zip(y_true, y_pred))
        fn = sum(a == c and b != c for a, b in zip(y_true, y_pred))
        denom = 2 * tp + fp + fn
        total += (2 * tp / denom) if denom else 0.0
    return total / len(classes)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--manifest", type=Path,
                   default=Path("data/local/dual_axis_plan/experiment_plan.json"))
    p.add_argument("--rpeval-pairs", type=Path,
                   default=Path("data/external_processed/caid_pairwise_train.jsonl"))
    p.add_argument("--s2pref-dir", type=Path,
                   default=Path("data/local/s2pref_candidates"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-silver-exploratory", action="store_true",
                   help="Permit experimental silver training; NOT a gold-label approval")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--pairs-per-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-train-steps", type=int, default=0,
                   help="0=full epoch, positive value caps steps for GPU smoke")
    p.add_argument("--max-eval-pairs", type=int, default=0,
                   help="0=full dev, positive value caps dev pairs for GPU smoke")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lambda-binary", type=float, default=0.5)
    p.add_argument("--lambda-pref-rank", type=float, default=0.5)
    p.add_argument("--lambda-silver-binary", type=float, default=0.5)
    p.add_argument("--lambda-context-rank", type=float, default=0.5)
    p.add_argument("--rank-margin", type=float, default=0.5)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--save-adapter", action="store_true",
                   help="Save PEFT model/tokenizer after training")
    args = p.parse_args(argv)
    if args.epochs <= 0 or args.pairs_per_batch <= 0 or args.grad_accum <= 0:
        p.error("epochs, pairs-per-batch and grad-accum must be positive")
    if args.max_length < 128 or args.max_train_steps < 0 or args.max_eval_pairs < 0:
        p.error("max-length must be >=128; caps must be nonnegative")
    if args.output_dir is None:
        args.output_dir = Path("data/local/train_runs") / args.arm
    return args


def planned_silver_steps(rpeval_count, silver_count, pair_batch_size, cap=0):
    total = math.ceil(rpeval_count / pair_batch_size)
    used = min(total, cap) if cap else total
    if silver_count > total:
        raise ValueError("Increase pairs-per-batch? No: reduce it; silver pairs must fit in RPEval steps")
    return sum(silver_index_for_step(i, total, silver_count) is not None for i in range(used))


def encode_pair_batch(tokenizer, pairs, max_length):
    left = [format_side(p["positive"]) for p in pairs]
    right = [format_side(p["negative"]) for p in pairs]
    encoded = tokenizer(left + right, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
    return encoded, len(pairs)


def score(logits, torch):
    return torch.logsumexp(logits.float()[:, 1:3], dim=-1) - logits.float()[:, 0]


def rank_loss(pos, neg, torch, margin):
    import torch.nn.functional as F
    return F.softplus(margin - (pos - neg)).mean()


def setup_torch_model(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA missing from Slurm allocation")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label={v: k for k, v in LABEL2ID.items()},
        label2id=LABEL2ID,
        quantization_config=config,
        device_map={"": 0},
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear", modules_to_save=["score"],
        bias="none", use_rslora=True,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return torch, model, tokenizer


def evaluate_rpeval(model, tokenizer, rows, torch, limit, max_length):
    import torch.nn.functional as F
    model.eval()
    chosen = rows[:limit] if limit else rows
    y3, pred3, ybin, pbin, margins = [], [], [], [], []
    with torch.no_grad():
        for batch in make_batches(chosen, 2):
            enc, n = encode_pair_batch(tokenizer, batch, max_length)
            inputs = {k: v.to("cuda:0") for k, v in enc.items()}
            logits = model(**inputs).logits.float()
            truth = [LABEL2ID[p["positive"]["native_label"]] for p in batch] + [
                LABEL2ID[p["negative"]["native_label"]] for p in batch]
            pred = logits.argmax(-1).cpu().tolist()
            scores = score(logits, torch).cpu().tolist()
            y3.extend(truth)
            pred3.extend(pred)
            ybin.extend([int(x != 0) for x in truth])
            pbin.extend([int(v >= 0) for v in scores])
            margins.extend([scores[i] - scores[i + n] for i in range(n)])
    return {
        "n_pairs": len(chosen),
        "native_macro_f1": macro_f1(y3, pred3, (0,1,2)),
        "binary_macro_f1": macro_f1(ybin, pbin, (0,1)),
        "pair_direction_accuracy": sum(m > 0 for m in margins) / max(1, len(margins)),
    }


def evaluate_silver(model, tokenizer, rows, strict_ids, torch, limit, max_length):
    model.eval()
    chosen = rows[:limit] if limit else rows
    truth, pred, directions = [], [], []
    strict_set = set(strict_ids)
    unseen = []
    with torch.no_grad():
        for batch in make_batches(chosen, 2):
            enc, n = encode_pair_batch(tokenizer, batch, max_length)
            inp = {k: v.to("cuda:0") for k, v in enc.items()}
            s = score(model(**inp).logits, torch).cpu().tolist()
            for i, p in enumerate(batch):
                truth.extend((1, 0))
                pred.extend((int(s[i] >= 0), int(s[n+i] >= 0)))
                correct = int(s[i] > s[n+i])
                directions.append(correct)
                if p["pair_id"] in strict_set:
                    unseen.append((p["pair_id"], correct, int(s[i]>=0), int(s[n+i]>=0)))
    strict_true = [t for _pid, _d, ap, ig in unseen for t in (1, 0)]
    strict_pred = [v for _pid, _d, ap, ig in unseen for v in (ap, ig)]
    return {
        "n_pairs": len(chosen), "binary_macro_f1": macro_f1(truth, pred, (0,1)),
        "pair_direction_accuracy": sum(directions)/max(1,len(directions)),
        "strict_unseen_n_pairs": len(unseen),
        "strict_unseen_binary_macro_f1": macro_f1(strict_true,strict_pred,(0,1)),
        "strict_unseen_pair_direction_accuracy": sum(x[1] for x in unseen)/max(1,len(unseen)),
        "metrics_are_derived_silver_not_gold": True,
    }


def train(args, data, info):
    if args.arm in BINARY_ARMS and not args.allow_silver_exploratory:
        raise RuntimeError("S2Pref silver is unreviewed; use --allow-silver-exploratory only for research smoke")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU training is restricted to an allocated Slurm job, not the login node")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Job has no CUDA GPU; do not train on CPU/login")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "training_summary.json").exists():
        raise RuntimeError("This run directory already has a completed result; specify a new --output-dir")
    torch, model, tokenizer = setup_torch_model(args)
    import torch.nn.functional as F

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    rtrain, strain = data["rpeval_train"], data["s2pref_train"]
    steps_per_epoch = math.ceil(len(rtrain) / args.pairs_per_batch)
    used_per_epoch = min(steps_per_epoch, args.max_train_steps) if args.max_train_steps else steps_per_epoch
    total_updates = math.ceil(used_per_epoch / args.grad_accum) * args.epochs
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=max(1, round(0.05 * total_updates)),
        num_training_steps=total_updates,
    )

    history = []
    best_f1, best_epoch = -1.0, None
    if args.arm in BINARY_ARMS:
        print("WARNING: UNREVIEWED S2PREF SILVER EXPERIMENTAL TRAINING", flush=True)
    for epoch in range(args.epochs):
        model.train()
        order = list(rtrain)
        random.Random(args.seed + epoch).shuffle(order)
        silver = list(strain)
        random.Random(args.seed + 1000 + epoch).shuffle(silver)
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        num_batches, silver_batches = 0, 0
        start = time.monotonic()
        for step, pairs in enumerate(make_batches(order, args.pairs_per_batch)):
            if step >= used_per_epoch:
                break
            enc, n = encode_pair_batch(tokenizer, pairs, args.max_length)
            inp = {k: v.to("cuda:0") for k,v in enc.items()}
            logits = model(**inp).logits.float()
            gold = torch.tensor(
                [LABEL2ID[p["positive"]["native_label"]] for p in pairs]
                + [LABEL2ID[p["negative"]["native_label"]] for p in pairs],
                device="cuda:0",
            )
            ce3 = F.cross_entropy(logits, gold, label_smoothing=0.05)
            raw_score = score(logits, torch)
            target = (gold != 0).float()
            bce = F.binary_cross_entropy_with_logits(raw_score, target)
            loss_r = ce3 + args.lambda_binary * bce
            if args.arm != "pointwise":
                preference_rank = rank_loss(raw_score[:n], raw_score[n:], torch, args.rank_margin)
                loss_r = loss_r + args.lambda_pref_rank * preference_rank
            if not torch.isfinite(loss_r):
                raise RuntimeError("non-finite RPEval loss")
            (loss_r / args.grad_accum).backward()
            scalar_loss = float(loss_r.detach().cpu())
            del logits, loss_r, enc, inp

            if args.arm in BINARY_ARMS:
                si = silver_index_for_step(step, steps_per_epoch, len(silver))
                if si is not None:
                    sb = [silver[si]]
                    senc, ns = encode_pair_batch(tokenizer, sb, args.max_length)
                    sinp = {k: v.to("cuda:0") for k,v in senc.items()}
                    slogits = model(**sinp).logits.float()
                    sc = score(slogits, torch)
                    silver_gold = torch.cat((torch.ones(ns,device="cuda:0"),
                                              torch.zeros(ns,device="cuda:0")))
                    # Binary ONLY. No native RPEval SUP/DOM labels are inferred.
                    silver_bce = F.binary_cross_entropy_with_logits(sc, silver_gold)
                    loss_s = args.lambda_silver_binary * silver_bce
                    if args.arm == "dual_axis":
                        context_rank = rank_loss(sc[:ns], sc[ns:], torch, args.rank_margin)
                        loss_s = loss_s + args.lambda_context_rank * context_rank
                    if not torch.isfinite(loss_s):
                        raise RuntimeError("non-finite S2Pref loss")
                    (loss_s / args.grad_accum).backward()
                    scalar_loss += float(loss_s.detach().cpu())
                    silver_batches += 1
                    del senc, sinp, slogits, loss_s

            is_last_step = (step + 1 == used_per_epoch)
            if (step+1) % args.grad_accum == 0 or is_last_step:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
            num_batches += 1
            loss_total += scalar_loss
            if (step+1) % 25 == 0 or is_last_step:
                print(json.dumps({"epoch":epoch+1, "step":step+1,
                                  "total_steps":used_per_epoch,
                                  "silver_batches":silver_batches,
                                  "mean_loss":round(loss_total/num_batches,5)}),flush=True)

        rv = evaluate_rpeval(model, tokenizer, data["rpeval_dev"],
                             torch, args.max_eval_pairs, args.max_length)
        sv = evaluate_silver(model, tokenizer, data["s2pref_dev"],
                             data["manifest"]["s2pref"]["strict_unseen_dev_pair_ids"],
                             torch, args.max_eval_pairs, args.max_length)
        record = {
            "epoch": epoch+1, "rpeval_dev": rv,
            "s2pref_silver_dev": sv, "rpeval_batches":num_batches,
            "silver_batches":silver_batches,
            "seconds":round(time.monotonic()-start,2),
            "partial_training": bool(args.max_train_steps),
            "partial_evaluation": bool(args.max_eval_pairs),
        }
        history.append(record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
        if rv["binary_macro_f1"] > best_f1:
            best_f1, best_epoch = rv["binary_macro_f1"], epoch+1
            if args.save_adapter:
                adapter_path = args.output_dir / "best_adapter"
                model.save_pretrained(adapter_path)
                tokenizer.save_pretrained(adapter_path)
        model.train()
    manifest = {
        **info,
        "status": "GPU_TRAINING_COMPLETED_EXPLORATORY",
        "seed": args.seed, "model": args.model,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_node": os.getenv("SLURMD_NODENAME"),
        "cuda_name": torch.cuda.get_device_name(0),
        "rpeval_selection_only": "binary_macro_f1",
        "best_epoch": best_epoch, "best_rpeval_dev_binary_macro_f1": best_f1,
        "S2Pref_test_seen": False, "BenchPreS_test_seen": False,
        "s2pref_labels_not_gold": True,
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k,v in vars(args).items()},
        "history":history,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf8")
    print(f"Training summary: {args.output_dir / 'training_summary.json'}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    data = load_and_verify(args.manifest, args.rpeval_pairs, args.s2pref_dir)
    info = summary(data, args.arm)
    steps = math.ceil(len(data["rpeval_train"]) / args.pairs_per_batch)
    info["rpeval_batches_full_epoch"] = steps
    info["rpeval_batches_this_epoch"] = min(args.max_train_steps, steps) if args.max_train_steps else steps
    info["s2pref_batches_this_epoch"] = (
        planned_silver_steps(len(data["rpeval_train"]),len(data["s2pref_train"]),
                             args.pairs_per_batch,args.max_train_steps)
        if args.arm in BINARY_ARMS else 0
    )
    info["dry_run"] = args.dry_run
    print(json.dumps(info,indent=2), flush=True)
    if args.dry_run:
        print("DRY RUN ONLY: no GPU, model, or optimizer used.")
        return
    train(args, data, info)


if __name__ == "__main__":
    main()
