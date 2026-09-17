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


class FlatPairDataset(Dataset):
    """Same texts as CAID, but standard pointwise 3-way RPEval supervision."""

    def __init__(self, pairs):
        self.items = []
        for pair in pairs:
            for side_name in ("positive", "negative"):
                side = pair[side_name]
                label = str(side.get("native_label", "")).upper()
                if label in LABEL2ID:
                    self.items.append((format_side(side), LABEL2ID[label]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class Collator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        text, labels = zip(*batch)
        toks = self.tokenizer(
            list(text),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        toks["labels_native"] = torch.tensor(labels, dtype=torch.long)
        return toks


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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fair pointwise baseline: same model, same benchmark-native pair texts, "
            "same pair split as CAID, but standard 3-way cross-entropy only."
        )
    )
    parser.add_argument("--pairs", default="data/external_processed/caid_pairwise_train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", default="results/pointwise_qwen3b")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lora", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    pairs = load_jsonl(args.pairs)
    if len(pairs) < 2:
        raise SystemExit("Need at least two pair records.")
    random.shuffle(pairs)
    dev_n = max(1, int(0.1 * len(pairs)))
    dev_pairs, train_pairs = pairs[:dev_n], pairs[dev_n:]

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

    collator = Collator(tokenizer, args.max_length)
    train_loader = DataLoader(
        FlatPairDataset(train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        FlatPairDataset(dev_pairs),
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
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels_native").to(device)
            toks = {k: v.to(device) for k, v in batch.items()}
            logits = model(**toks).logits
            loss = F.cross_entropy(logits, labels)
            (loss / args.grad_accum).backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().cpu())

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in dev_loader:
                labels = batch.pop("labels_native").to(device)
                toks = {k: v.to(device) for k, v in batch.items()}
                pred = model(**toks).logits.argmax(-1)
                correct += int((pred == labels).sum().item())
                total += len(labels)
        rec = {
            "epoch": epoch + 1,
            "train_loss": running / max(1, len(train_loader)),
            "dev_native_3way_accuracy": correct / max(1, total),
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
