#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

LABEL2ID = {"IGNORE": 0, "SUPPORT": 1, "DOMINATE": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_side(side: dict) -> str:
    parts = [f"[PREFERENCE]\n{side.get('preference', '')}"]
    history = str(side.get("history", ""))
    context = str(side.get("context", ""))
    if history.strip():
        parts.append(f"[HISTORY]\n{history}")
    if context.strip():
        parts.append(f"[CONTEXT]\n{context}")
    parts.append(f"[QUERY]\n{side.get('query', '')}")
    parts.append(
        "[TASK]\nJudge the preference-query relation using the native RPEval "
        "labels IGNORE, SUPPORT, or DOMINATE."
    )
    parts.append("[DECISION]")
    return "\n".join(parts)


def group_safe_pair_split(pairs, dev_fraction: float, seed: int):
    groups = sorted({str(p.get("group_id", "")) for p in pairs})
    if not groups or "" in groups:
        raise SystemExit("Pair file is missing group_id.")
    rng = random.Random(seed)
    rng.shuffle(groups)
    dev_n = max(1, int(round(dev_fraction * len(groups))))
    dev_groups = set(groups[:dev_n])
    train = [p for p in pairs if str(p["group_id"]) not in dev_groups]
    dev = [p for p in pairs if str(p["group_id"]) in dev_groups]
    if {p["group_id"] for p in train} & {p["group_id"] for p in dev}:
        raise RuntimeError("Group leakage detected.")
    return train, dev


class FlatPairDataset(Dataset):
    def __init__(self, pairs):
        self.items = []
        for pair in pairs:
            for name in ("positive", "negative"):
                side = pair[name]
                label = str(side.get("native_label", "")).upper()
                if label in LABEL2ID:
                    self.items.append((format_side(side), LABEL2ID[label]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class PairDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return {
            "positive_text": format_side(row["positive"]),
            "positive_label": LABEL2ID[str(row["positive"]["native_label"]).upper()],
            "negative_text": format_side(row["negative"]),
            "negative_label": LABEL2ID[str(row["negative"]["native_label"]).upper()],
        }


class FlatCollator:
    def __init__(self, tokenizer, max_length):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts, labels = zip(*batch)
        toks = self.tok(
            list(texts), padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        return toks, torch.tensor(labels, dtype=torch.long)


class PairConcatCollator:
    """Tokenize positive and negative sides together so CAID uses one forward call."""
    def __init__(self, tokenizer, max_length):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        pos_text = [x["positive_text"] for x in batch]
        neg_text = [x["negative_text"] for x in batch]
        texts = pos_text + neg_text
        labels = (
            [x["positive_label"] for x in batch]
            + [x["negative_label"] for x in batch]
        )
        toks = self.tok(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        return toks, torch.tensor(labels, dtype=torch.long), len(batch)


def applicability_score(logits):
    z = logits.float()
    return torch.logsumexp(z[:, 1:3], dim=-1) - z[:, 0]


def binary_anchor_loss(logits, labels):
    y = (labels != LABEL2ID["IGNORE"]).float()
    return F.binary_cross_entropy_with_logits(applicability_score(logits), y)


def native_loss(logits, labels, smoothing):
    return F.cross_entropy(logits.float(), labels, label_smoothing=smoothing)


def build_model(args, tokenizer):
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    compute_dtype = torch.float16  # T4
    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        quantization_config=qcfg,
        device_map={"": 0},
        dtype=compute_dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        modules_to_save=["score"],
        bias="none",
        use_rslora=True,
    )
    model = get_peft_model(model, lora)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    return model, compute_dtype


def build_optimizer(model, args):
    import bitsandbytes as bnb
    groups = {"A": [], "B": [], "head": [], "other": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        low = name.lower()
        if "lora_a" in low:
            groups["A"].append(p)
        elif "lora_b" in low:
            groups["B"].append(p)
        elif "score" in low:
            groups["head"].append(p)
        else:
            groups["other"].append(p)

    pgs = []
    if groups["A"]:
        pgs.append({"params": groups["A"], "lr": args.lr_a, "weight_decay": args.weight_decay})
    if groups["B"]:
        pgs.append({"params": groups["B"], "lr": args.lr_a * args.loraplus_ratio, "weight_decay": args.weight_decay})
    if groups["head"]:
        pgs.append({"params": groups["head"], "lr": args.head_lr, "weight_decay": args.weight_decay})
    if groups["other"]:
        pgs.append({"params": groups["other"], "lr": args.lr_a, "weight_decay": args.weight_decay})
    return bnb.optim.PagedAdamW8bit(pgs, betas=(0.9, 0.999), eps=1e-8)


@torch.no_grad()
def evaluate_dev(model, pair_loader, device):
    model.eval()
    nt, npred, bt, bpred, margins = [], [], [], [], []
    for toks, labels, n_pairs in pair_loader:
        toks = {k: v.to(device, non_blocking=True) for k, v in toks.items()}
        labels = labels.to(device, non_blocking=True)
        logits = model(**toks).logits.float()
        lp, ln = logits[:n_pairs], logits[n_pairs:]
        yp, yn = labels[:n_pairs], labels[n_pairs:]

        nt.extend(yp.cpu().tolist()); nt.extend(yn.cpu().tolist())
        npred.extend(lp.argmax(-1).cpu().tolist()); npred.extend(ln.argmax(-1).cpu().tolist())

        sp, sn = applicability_score(lp), applicability_score(ln)
        bt.extend([1] * n_pairs + [0] * n_pairs)
        bpred.extend((sp >= 0).long().cpu().tolist() + (sn >= 0).long().cpu().tolist())
        margins.extend((sp - sn).cpu().tolist())

    return {
        "dev_native_3way_accuracy": float(accuracy_score(nt, npred)),
        "dev_native_3way_macro_f1": float(f1_score(nt, npred, labels=[0,1,2], average="macro", zero_division=0)),
        "dev_binary_accuracy": float(accuracy_score(bt, bpred)),
        "dev_binary_macro_f1": float(f1_score(bt, bpred, labels=[1,0], average="macro", zero_division=0)),
        "dev_pair_direction_accuracy": float(np.mean(np.asarray(margins) > 0)),
        "dev_mean_applicability_margin": float(np.mean(margins)),
        "dev_median_applicability_margin": float(np.median(margins)),
    }


def save_best(model, tokenizer, best_dir, record, args):
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    (best_dir / "best_meta.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (best_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["pointwise", "caid"], required=True)
    p.add_argument("--pairs", default="data/external_processed/caid_pairwise_train.jsonl")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--pointwise-batch", type=int, default=4)
    p.add_argument("--caid-pair-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr-a", type=float, default=1e-5)
    p.add_argument("--loraplus-ratio", type=float, default=16.0)
    p.add_argument("--head-lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lambda-binary", type=float, default=0.5)
    p.add_argument("--lambda-rank", type=float, default=0.5)
    p.add_argument("--rank-margin", type=float, default=0.5)
    p.add_argument("--rank-temperature", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--dev-fraction", type=float, default=0.10)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--progress-every", type=int, default=100)
    args = p.parse_args()

    seed_all(args.seed)
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    rows = load_jsonl(args.pairs)
    train_rows, dev_rows = group_safe_pair_split(rows, args.dev_fraction, args.seed)
    print(json.dumps({
        "seed": args.seed, "method": args.method,
        "train_pairs": len(train_rows), "dev_pairs": len(dev_rows),
        "train_groups": len({p["group_id"] for p in train_rows}),
        "dev_groups": len({p["group_id"] for p in dev_rows}),
        "group_overlap": 0,
    }, indent=2), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model, compute_dtype = build_model(args, tok)
    model.print_trainable_parameters()
    device = torch.device("cuda:0")

    pair_collator = PairConcatCollator(tok, args.max_length)
    dev_loader = DataLoader(
        PairDataset(dev_rows),
        batch_size=args.caid_pair_batch,
        shuffle=False,
        collate_fn=pair_collator,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    if args.method == "pointwise":
        train_loader = DataLoader(
            FlatPairDataset(train_rows),
            batch_size=args.pointwise_batch,
            shuffle=True,
            collate_fn=FlatCollator(tok, args.max_length),
            pin_memory=True,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
        sequences_per_microbatch = args.pointwise_batch
    else:
        train_loader = DataLoader(
            PairDataset(train_rows),
            batch_size=args.caid_pair_batch,
            shuffle=True,
            collate_fn=pair_collator,
            pin_memory=True,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
        sequences_per_microbatch = 2 * args.caid_pair_batch

    opt = build_optimizer(model, args)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = updates_per_epoch * args.max_epochs
    warmup = max(1, int(round(args.warmup_ratio * total_updates)))
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_updates)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_dir = out / "best"

    (out / "run_manifest.json").write_text(json.dumps({
        "method": args.method,
        "seed": args.seed,
        "model": args.model,
        "single_forward_for_caid_pos_neg": True,
        "sequences_per_microbatch": sequences_per_microbatch,
        "effective_sequences_per_update": sequences_per_microbatch * args.grad_accum,
        "gradient_checkpointing_use_reentrant": False,
        "selection_metric": "dev_binary_macro_f1",
        "tie_breakers": ["dev_native_3way_macro_f1", "dev_pair_direction_accuracy"],
        "max_epochs": args.max_epochs,
        "patience": args.patience,
    }, indent=2), encoding="utf-8")

    history = []
    best_tuple = None
    best_epoch = None
    bad = 0
    global_update = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        start = time.time()
        running = {"loss":0.0, "native":0.0, "binary":0.0, "rank":0.0}
        nb = 0

        for step, batch in enumerate(train_loader, start=1):
            if args.method == "pointwise":
                toks, labels = batch
                toks = {k:v.to(device, non_blocking=True) for k,v in toks.items()}
                labels = labels.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=compute_dtype):
                    logits = model(**toks).logits
                nloss = native_loss(logits, labels, args.label_smoothing)
                bloss = binary_anchor_loss(logits, labels)
                rloss = torch.zeros((), device=device)
                loss = nloss + args.lambda_binary * bloss
            else:
                toks, labels, n_pairs = batch
                toks = {k:v.to(device, non_blocking=True) for k,v in toks.items()}
                labels = labels.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=compute_dtype):
                    logits = model(**toks).logits
                lp, ln = logits[:n_pairs], logits[n_pairs:]
                yp, yn = labels[:n_pairs], labels[n_pairs:]
                nloss = (native_loss(lp, yp, args.label_smoothing) + native_loss(ln, yn, args.label_smoothing))/2
                bloss = (binary_anchor_loss(lp, yp) + binary_anchor_loss(ln, yn))/2
                delta = applicability_score(lp) - applicability_score(ln)
                rloss = F.softplus((args.rank_margin-delta)/args.rank_temperature).mean()*args.rank_temperature
                loss = nloss + args.lambda_binary*bloss + args.lambda_rank*rloss

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}, step={step}")

            (loss / args.grad_accum).backward()

            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_update += 1

            running["loss"] += float(loss.detach().cpu())
            running["native"] += float(nloss.detach().cpu())
            running["binary"] += float(bloss.detach().cpu())
            running["rank"] += float(rloss.detach().cpu())
            nb += 1

            if step % args.progress_every == 0 or step == len(train_loader):
                elapsed = time.time() - start
                rate = elapsed / step
                eta = rate * (len(train_loader) - step)
                print(json.dumps({
                    "epoch": epoch,
                    "step": step,
                    "steps": len(train_loader),
                    "pct": round(100*step/len(train_loader), 1),
                    "elapsed_min": round(elapsed/60, 1),
                    "eta_min": round(eta/60, 1),
                    "avg_loss_so_far": round(running["loss"]/nb, 5),
                }), flush=True)

        dev = evaluate_dev(model, dev_loader, device)
        rec = {
            "epoch": epoch,
            "global_updates": global_update,
            "epoch_minutes": round((time.time()-start)/60, 2),
            "train_loss": running["loss"]/max(1,nb),
            "train_native_loss": running["native"]/max(1,nb),
            "train_binary_anchor_loss": running["binary"]/max(1,nb),
            "train_rank_loss": running["rank"]/max(1,nb),
            **dev,
        }
        history.append(rec)
        print(json.dumps(rec, indent=2), flush=True)

        candidate = (
            rec["dev_binary_macro_f1"],
            rec["dev_native_3way_macro_f1"],
            rec["dev_pair_direction_accuracy"],
        )
        if best_tuple is None or candidate > best_tuple:
            best_tuple = candidate
            best_epoch = epoch
            bad = 0
            save_best(model, tok, best_dir, rec, args)
            print(f"NEW BEST epoch={epoch}: {best_tuple}", flush=True)
        else:
            bad += 1
            print(f"No improvement: patience {bad}/{args.patience}", flush=True)

        (out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if bad >= args.patience:
            print(f"EARLY STOP. Best epoch={best_epoch}.", flush=True)
            break

    summary = {
        "method": args.method,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_selection_tuple": best_tuple,
        "epochs_ran": len(history),
        "best_checkpoint": str(best_dir),
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
