#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LABEL2ID = {"IGNORE": 0, "SUPPORT": 1, "DOMINATE": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_side(side: dict) -> str:
    return (
        f"[PREFERENCE]\n{side.get('preference', '')}\n"
        f"[HISTORY]\n{side.get('history', '')}\n"
        f"[CONTEXT]\n{side.get('context', '')}\n"
        f"[QUERY]\n{side.get('query', '')}\n"
        "[QUESTION]\nHow strongly should this preference influence the response?"
    )


class PairDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "pair_id": row["pair_id"],
            "positive_text": format_side(row["positive"]),
            "positive_label": LABEL2ID[row["positive"]["native_label"]],
            "negative_text": format_side(row["negative"]),
            "negative_label": LABEL2ID[row["negative"]["native_label"]],
        }


class PairCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        pos = self.tokenizer(
            [x["positive_text"] for x in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        neg = self.tokenizer(
            [x["negative_text"] for x in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "positive": pos,
            "positive_label": torch.tensor(
                [x["positive_label"] for x in batch], dtype=torch.long
            ),
            "negative": neg,
            "negative_label": torch.tensor(
                [x["negative_label"] for x in batch], dtype=torch.long
            ),
        }


def maybe_apply_lora(model, enabled):
    if not enabled:
        return model
    from peft import LoraConfig, TaskType, get_peft_model
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )
    return get_peft_model(model, config)


def applicability_score(logits):
    """Log-odds-like applicability score from native RPEval logits.

    Higher means SUPPORT/DOMINATE collectively outrank IGNORE.
    """
    return torch.logsumexp(logits[:, 1:3], dim=-1) - logits[:, 0]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "CAID pairwise training: same native RPEval 3-way supervision as the "
            "pointwise baseline plus a pairwise applicability-boundary ranking loss."
        )
    )
    parser.add_argument("--pairs", default="data/external_processed/caid_pairwise_train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", default="results/caid_pairwise_qwen3b")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lora", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = load_jsonl(args.pairs)
    if len(rows) < 2:
        raise SystemExit("Need at least two pair records.")
    random.shuffle(rows)
    dev_n = max(1, int(0.1 * len(rows)))
    dev_rows, train_rows = rows[:dev_n], rows[dev_n:]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = maybe_apply_lora(model, args.lora)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collator = PairCollator(tokenizer, args.max_length)
    loader = DataLoader(
        PairDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        PairDataset(dev_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(loader, start=1):
            pos = {k: v.to(device) for k, v in batch["positive"].items()}
            neg = {k: v.to(device) for k, v in batch["negative"].items()}
            y_pos = batch["positive_label"].to(device)
            y_neg = batch["negative_label"].to(device)

            logits_pos = model(**pos).logits
            logits_neg = model(**neg).logits

            native_ce = (
                F.cross_entropy(logits_pos, y_pos)
                + F.cross_entropy(logits_neg, y_neg)
            ) / 2.0
            s_pos = applicability_score(logits_pos)
            s_neg = applicability_score(logits_neg)
            rank = F.relu(args.margin - (s_pos - s_neg)).mean()
            loss = native_ce + args.lambda_rank * rank

            (loss / args.grad_accum).backward()
            if step % args.grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().cpu())

        model.eval()
        direction, native_correct, native_total, margins = 0, 0, 0, []
        with torch.no_grad():
            for batch in dev_loader:
                pos = {k: v.to(device) for k, v in batch["positive"].items()}
                neg = {k: v.to(device) for k, v in batch["negative"].items()}
                y_pos = batch["positive_label"].to(device)
                y_neg = batch["negative_label"].to(device)
                lp = model(**pos).logits
                ln = model(**neg).logits
                sp, sn = applicability_score(lp), applicability_score(ln)
                m = sp - sn
                margins.extend(m.float().cpu().tolist())
                direction += int((m > 0).sum().item())
                native_correct += int((lp.argmax(-1) == y_pos).sum().item())
                native_correct += int((ln.argmax(-1) == y_neg).sum().item())
                native_total += 2 * len(y_pos)

        rec = {
            "epoch": epoch + 1,
            "train_loss": running / max(1, len(loader)),
            "dev_pair_direction_accuracy": direction / max(1, len(dev_rows)),
            "dev_native_3way_accuracy": native_correct / max(1, native_total),
            "dev_mean_applicability_margin": sum(margins) / max(1, len(margins)),
        }
        history.append(rec)
        print(json.dumps(rec, indent=2))

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (out / "training_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    print(f"Saved checkpoint to {out}")


if __name__ == "__main__":
    main()
