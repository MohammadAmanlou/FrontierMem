from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import SPEC_BY_FAMILY, SPECS
from .memory import MemoryItem, RuleBasedMemoryExtractor, RuleBasedApplicabilityPolicy


FAMILIES = tuple(spec.family for spec in SPECS)
PROFILE_VARIANTS = ("stable", "scoped")
OWNERS = ("user", "sister")
INFO_TYPES = (
    "trait_preference",
    "temporary_constraint",
    "temporary_state",
    "current_goal",
    "contextual_preference",
    "other_owner",
)
TEMPORAL = ("persistent", "context-dependent")
RELATIONS = ("match", "mismatch", "unknown")
ACTIONS = ("APPLY", "IGNORE", "CLARIFY")


INDEX = {
    "family": {v: i for i, v in enumerate(FAMILIES)},
    "profile": {v: i for i, v in enumerate(PROFILE_VARIANTS)},
    "owner": {v: i for i, v in enumerate(OWNERS)},
    "type": {v: i for i, v in enumerate(INFO_TYPES)},
    "temporal": {v: i for i, v in enumerate(TEMPORAL)},
    "relation": {v: i for i, v in enumerate(RELATIONS)},
    "action": {v: i for i, v in enumerate(ACTIONS)},
}


class SimpleTokenizer:
    SPECIALS = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        self.vocab = vocab or {token: i for i, token in enumerate(self.SPECIALS)}
        self.inverse = {i: token for token, i in self.vocab.items()}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_'-]+|[^\w\s]", text.lower())

    def fit(self, texts: Iterable[str], min_freq: int = 1, max_vocab: int = 6000) -> None:
        counts: dict[str, int] = {}
        for text in texts:
            for token in self.tokenize(str(text)):
                counts[token] = counts.get(token, 0) + 1
        items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        for token, count in items:
            if count < min_freq or token in self.vocab:
                continue
            if len(self.vocab) >= max_vocab:
                break
            self.vocab[token] = len(self.vocab)
        self.inverse = {i: token for token, i in self.vocab.items()}

    @property
    def pad_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.vocab["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.vocab["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.vocab["<unk>"]

    def encode(self, text: str, max_length: int, add_special: bool = True) -> list[int]:
        ids = [self.vocab.get(tok, self.unk_id) for tok in self.tokenize(text)]
        if add_special:
            ids = [self.bos_id, *ids, self.eos_id]
        return ids[:max_length]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        tokens: list[str] = []
        for idx in ids:
            token = self.inverse.get(int(idx), "<unk>")
            if skip_special and token in self.SPECIALS:
                continue
            tokens.append(token)
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = text.replace(" n't", "n't").replace(" 's", "'s")
        return text.strip()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.vocab, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class TinyLMConfig:
    vocab_size: int
    d_model: int = 32
    nhead: int = 4
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    dim_feedforward: int = 64
    dropout: float = 0.1
    max_input_length: int = 96
    max_output_length: int = 48


class TinyMultiTaskLM(nn.Module):
    """A small Transformer language model with structured classification heads.

    The encoder is shared by memory extraction and boundary decision tasks. A small
    autoregressive decoder generates the final answer or clarification question.
    """

    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            max(config.max_input_length, config.max_output_length) + 4,
            config.d_model,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=config.num_decoder_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.heads = nn.ModuleDict(
            {
                "family": nn.Linear(config.d_model, len(FAMILIES)),
                "profile": nn.Linear(config.d_model, len(PROFILE_VARIANTS)),
                "owner": nn.Linear(config.d_model, len(OWNERS)),
                "info_type": nn.Linear(config.d_model, len(INFO_TYPES)),
                "temporal": nn.Linear(config.d_model, len(TEMPORAL)),
                "relation": nn.Linear(config.d_model, len(RELATIONS)),
                "action": nn.Linear(config.d_model, len(ACTIONS)),
            }
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        return self.token_embedding(input_ids) * math.sqrt(self.config.d_model) + self.position_embedding(positions)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._embed(input_ids)
        key_padding_mask = ~attention_mask.bool()
        memory = self.encoder(x, src_key_padding_mask=key_padding_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (memory * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.norm(memory), self.norm(pooled)

    def classify(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        _, pooled = self.encode(input_ids, attention_mask)
        return {name: head(pooled) for name, head in self.heads.items()}

    def decode_response(
        self,
        encoder_memory: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        y = self._embed(decoder_input_ids)
        length = decoder_input_ids.size(1)
        causal = torch.triu(
            torch.ones(length, length, device=decoder_input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        decoded = self.decoder(
            y,
            encoder_memory,
            tgt_mask=causal,
            memory_key_padding_mask=~encoder_attention_mask.bool(),
        )
        return self.lm_head(self.norm(decoded))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 40,
    ) -> torch.Tensor:
        self.eval()
        memory, _ = self.encode(input_ids, attention_mask)
        output = torch.full(
            (input_ids.size(0), 1), bos_id, dtype=torch.long, device=input_ids.device
        )
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            logits = self.decode_response(memory, attention_mask, output)
            next_token = logits[:, -1].argmax(dim=-1)
            output = torch.cat([output, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(eos_id)
            if bool(finished.all()):
                break
        return output


class ClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: SimpleTokenizer, task: str, max_length: int) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        if self.task == "extract":
            text = extraction_prompt(row["history"])
        elif self.task == "decision":
            text = decision_prompt_from_row(row)
        else:
            raise ValueError(self.task)
        return {
            "input_ids": self.tokenizer.encode(text, self.max_length),
            "family": INDEX["family"][row["family"]],
            "profile": INDEX["profile"][row["profile_variant"]],
            "owner": INDEX["owner"][row["owner"]],
            "type": INDEX["type"][row["information_type"]],
            "temporal": INDEX["temporal"][row["temporal_validity"]],
            "relation": INDEX["relation"][gold_relation(row)],
            "action": INDEX["action"][row["action"]],
        }


class ResponseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: SimpleTokenizer, config: TinyLMConfig) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        source = response_prompt_from_row(row)
        target = response_target_from_row(row)
        target_ids = self.tokenizer.encode(target, self.config.max_output_length)
        return {
            "input_ids": self.tokenizer.encode(source, self.config.max_input_length),
            "target_ids": target_ids,
        }


def extraction_prompt(history: str) -> str:
    return "task extract memory from user history : " + history


def memory_text(
    family: str,
    profile_variant: str,
    owner: str,
    information_type: str,
    temporal_validity: str,
) -> str:
    spec = SPEC_BY_FAMILY[family]
    if profile_variant == "stable":
        applies = "broadly across relevant situations"
        not_applies = "only when explicitly overridden or corrected"
    else:
        applies = spec.applies_when
        not_applies = spec.does_not_apply_when
    return (
        f"family {family} ; preference {spec.preference_label} ; profile {profile_variant} ; "
        f"owner {owner} ; type {information_type} ; time {temporal_validity} ; "
        f"applies {applies} ; does not apply {not_applies}"
    )


def decision_prompt_from_row(row: pd.Series | dict) -> str:
    mem = memory_text(
        row["family"],
        row["profile_variant"],
        row["owner"],
        row["information_type"],
        row["temporal_validity"],
    )
    return f"task decide applicability : {mem} ; current query {row['query']}"


def response_prompt_from_row(row: pd.Series | dict) -> str:
    return (
        f"task write final response : action {row['action']} ; preference {row['preference_label']} ; "
        f"current query {row['query']} ; clarification option {row['clarification_question']}"
    )


def response_target_from_row(row: pd.Series | dict) -> str:
    if row["action"] == "APPLY":
        return row["apply_response"]
    if row["action"] == "IGNORE":
        return row["neutral_response"]
    return row["clarification_question"]


def gold_relation(row: pd.Series | dict) -> str:
    if row["profile_variant"] == "stable" or row["zone"] == "inside":
        return "match"
    if row["zone"] == "outside":
        return "mismatch"
    return "unknown"


def _pad_sequences(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, : len(seq)] = True
    return ids, mask


def make_classification_collate(tokenizer: SimpleTokenizer):
    def collate(batch: list[dict]) -> dict:
        input_ids, attention_mask = _pad_sequences(
            [item["input_ids"] for item in batch], tokenizer.pad_id
        )
        result = {"input_ids": input_ids, "attention_mask": attention_mask}
        for key in ("family", "profile", "owner", "type", "temporal", "relation", "action"):
            result[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
        return result

    return collate


def make_response_collate(tokenizer: SimpleTokenizer):
    def collate(batch: list[dict]) -> dict:
        input_ids, attention_mask = _pad_sequences(
            [item["input_ids"] for item in batch], tokenizer.pad_id
        )
        target_ids, target_mask = _pad_sequences(
            [item["target_ids"] for item in batch], tokenizer.pad_id
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }

    return collate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainingSummary:
    epochs: int
    best_validation_action_accuracy: float
    training_history: list[dict]


@torch.no_grad()
def evaluate_action_accuracy(
    model: TinyMultiTaskLM,
    df: pd.DataFrame,
    tokenizer: SimpleTokenizer,
    config: TinyLMConfig,
    device: torch.device,
    batch_size: int = 128,
) -> float:
    model.eval()
    dataset = ClassificationDataset(df, tokenizer, "decision", config.max_input_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_classification_collate(tokenizer),
    )
    correct = 0
    total = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model.classify(ids, mask)["action"]
        pred = logits.argmax(dim=-1).cpu()
        correct += int((pred == batch["action"]).sum())
        total += len(pred)
    return correct / max(total, 1)


def train_tiny_lm(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_dir: str | Path,
    epochs: int = 5,
    batch_size: int = 96,
    learning_rate: float = 8e-4,
    seed: int = 17,
    device: str | None = None,
) -> tuple["TinyLLMFrontierMem", TrainingSummary]:
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep the local CPU demo fast while preserving every family/profile/query regime.
    train_df = pd.concat(
        [
            part.sample(min(len(part), 12), random_state=seed)
            for _, part in train_df.groupby(["family", "profile_variant", "query_kind"], sort=False)
        ],
        ignore_index=True,
    )

    texts: list[str] = []
    for _, row in train_df.iterrows():
        texts.extend(
            [
                extraction_prompt(row["history"]),
                decision_prompt_from_row(row),
                response_prompt_from_row(row),
                response_target_from_row(row),
            ]
        )
    tokenizer = SimpleTokenizer()
    tokenizer.fit(texts, min_freq=1, max_vocab=6000)
    config = TinyLMConfig(vocab_size=len(tokenizer.vocab))
    model = TinyMultiTaskLM(config)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)

    extract_ds = ClassificationDataset(train_df, tokenizer, "extract", config.max_input_length)
    decision_ds = ClassificationDataset(train_df, tokenizer, "decision", config.max_input_length)
    extract_loader = DataLoader(
        extract_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_classification_collate(tokenizer),
    )
    decision_loader = DataLoader(
        decision_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_classification_collate(tokenizer),
    )
    # The local demo uses deterministic response templates after the learned decision.
    # A Hugging Face Qwen backend is included separately for fully generative responses.
    response_loader = []

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()
    token_ce = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    history: list[dict] = []
    best_val = -1.0
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        extract_loss_sum = 0.0
        decision_loss_sum = 0.0
        response_loss_sum = 0.0

        for batch in extract_loader:
            optimizer.zero_grad(set_to_none=True)
            ids = batch["input_ids"].to(target_device)
            mask = batch["attention_mask"].to(target_device)
            logits = model.classify(ids, mask)
            loss = (
                ce(logits["family"], batch["family"].to(target_device))
                + ce(logits["profile"], batch["profile"].to(target_device))
                + 0.6 * ce(logits["owner"], batch["owner"].to(target_device))
                + ce(logits["info_type"], batch["type"].to(target_device))
                + 0.7 * ce(logits["temporal"], batch["temporal"].to(target_device))
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            extract_loss_sum += float(loss.detach())

        for batch in decision_loader:
            optimizer.zero_grad(set_to_none=True)
            ids = batch["input_ids"].to(target_device)
            mask = batch["attention_mask"].to(target_device)
            logits = model.classify(ids, mask)
            loss = ce(logits["relation"], batch["relation"].to(target_device)) + ce(
                logits["action"], batch["action"].to(target_device)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            decision_loss_sum += float(loss.detach())

        # Response generation is intentionally skipped in the CPU-fast local training.
        response_loss_sum = 0.0

        val_acc = evaluate_action_accuracy(
            model, validation_df, tokenizer, config, target_device, batch_size=batch_size
        )
        row = {
            "epoch": epoch,
            "extract_loss": extract_loss_sum / max(len(extract_loader), 1),
            "decision_loss": decision_loss_sum / max(len(decision_loader), 1),
            "response_loss": response_loss_sum / max(len(response_loader), 1),
            "validation_action_accuracy": val_acc,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{epochs} | extract={row['extract_loss']:.3f} "
            f"decision={row['decision_loss']:.3f} response={row['response_loss']:.3f} "
            f"val_action={val_acc:.3f}"
        )
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(target_device)

    checkpoint_path = output_dir / "tiny_multitask_lm.pt"
    torch.save(
        {
            "config": asdict(config),
            "state_dict": model.state_dict(),
            "vocab": tokenizer.vocab,
            "training_history": history,
        },
        checkpoint_path,
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    wrapper = TinyLLMFrontierMem(model, tokenizer, target_device)
    return wrapper, TrainingSummary(epochs, best_val, history)


class TinyLLMFrontierMem:
    name = "Tiny Transformer LM"

    def __init__(
        self,
        model: TinyMultiTaskLM,
        tokenizer: SimpleTokenizer,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load(cls, checkpoint_path: str | Path, device: str | None = None) -> "TinyLLMFrontierMem":
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = TinyLMConfig(**payload["config"])
        model = TinyMultiTaskLM(config)
        model.load_state_dict(payload["state_dict"])
        tokenizer = SimpleTokenizer(payload["vocab"])
        target = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model, tokenizer, target)

    def _batch_encode(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = [
            self.tokenizer.encode(text, self.model.config.max_input_length) for text in texts
        ]
        ids, mask = _pad_sequences(sequences, self.tokenizer.pad_id)
        return ids.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def extract_batch(self, histories: Sequence[str], batch_size: int = 128) -> list[dict]:
        outputs: list[dict] = []
        for start in range(0, len(histories), batch_size):
            texts = [extraction_prompt(h) for h in histories[start : start + batch_size]]
            ids, mask = self._batch_encode(texts)
            logits = self.model.classify(ids, mask)
            probs = {k: torch.softmax(v, dim=-1).cpu() for k, v in logits.items()}
            for i in range(len(texts)):
                family = FAMILIES[int(probs["family"][i].argmax())]
                profile = PROFILE_VARIANTS[int(probs["profile"][i].argmax())]
                owner = OWNERS[int(probs["owner"][i].argmax())]
                info_type = INFO_TYPES[int(probs["info_type"][i].argmax())]
                temporal = TEMPORAL[int(probs["temporal"][i].argmax())]
                confidence = float(
                    np.mean(
                        [
                            probs["family"][i].max().item(),
                            probs["profile"][i].max().item(),
                            probs["info_type"][i].max().item(),
                            probs["temporal"][i].max().item(),
                        ]
                    )
                )
                outputs.append(
                    {
                        "family": family,
                        "profile_variant": profile,
                        "owner": owner,
                        "information_type": info_type,
                        "temporal_validity": temporal,
                        "confidence": confidence,
                    }
                )
        return outputs

    @staticmethod
    def _memory_from_prediction(pred: dict, history: str) -> MemoryItem:
        spec = SPEC_BY_FAMILY[pred["family"]]
        stable = pred["profile_variant"] == "stable"
        lines = [re.sub(r"^Session\s+\d+:\s*", "", x) for x in history.splitlines() if x.strip()]
        if stable:
            applies = "broadly across relevant situations"
            not_applies = "only when explicitly overridden or corrected"
        else:
            applies = spec.applies_when
            not_applies = spec.does_not_apply_when
        return MemoryItem(
            preference=spec.preference_label,
            family=pred["family"],
            owner=pred["owner"],
            information_type=pred["information_type"],
            applies_when=applies,
            does_not_apply_when=not_applies,
            temporal_validity=pred["temporal_validity"],
            confidence=round(pred["confidence"], 4),
            evidence=lines[:3],
            profile_variant=pred["profile_variant"],
        )

    @torch.no_grad()
    def decide_batch(
        self,
        memories: Sequence[MemoryItem],
        queries: Sequence[str],
        batch_size: int = 128,
    ) -> list[dict]:
        outputs: list[dict] = []
        texts = []
        for memory, query in zip(memories, queries):
            mem = memory_text(
                memory.family,
                memory.profile_variant,
                memory.owner,
                memory.information_type,
                memory.temporal_validity,
            )
            texts.append(f"task decide applicability : {mem} ; current query {query}")
        for start in range(0, len(texts), batch_size):
            ids, mask = self._batch_encode(texts[start : start + batch_size])
            logits = self.model.classify(ids, mask)
            action_prob = torch.softmax(logits["action"], dim=-1).cpu()
            relation_prob = torch.softmax(logits["relation"], dim=-1).cpu()
            for i in range(len(action_prob)):
                outputs.append(
                    {
                        "action": ACTIONS[int(action_prob[i].argmax())],
                        "relation": RELATIONS[int(relation_prob[i].argmax())],
                        "confidence": float(action_prob[i].max()),
                    }
                )
        return outputs

    @torch.no_grad()
    def generate_response(self, action: str, memory: MemoryItem, query: str) -> str:
        spec = SPEC_BY_FAMILY[memory.family]
        if action == "APPLY":
            return spec.apply_response
        if action == "IGNORE":
            return spec.neutral_response
        return spec.clarification_question

    def predict_dataframe_hybrid(
        self,
        df: pd.DataFrame,
        batch_size: int = 128,
        generate_responses_for: int = 24,
    ) -> pd.DataFrame:
        """Use the tiny Transformer for memory typing and a transparent boundary policy.

        This mirrors the intended modular design: a learned extractor plus a cheap,
        auditable applicability controller. Family detection is kept rule-based in this
        CPU demo because the from-scratch model is intentionally tiny.
        """
        result = df.copy().reset_index(drop=True)
        extracts = self.extract_batch(result["history"].tolist(), batch_size=batch_size)
        rule_extractor = RuleBasedMemoryExtractor()
        rule_policy = RuleBasedApplicabilityPolicy()
        memories: list[MemoryItem] = []
        actions: list[str] = []
        relations: list[str] = []
        confidences: list[float] = []
        for pred, history, query in zip(extracts, result["history"], result["query"]):
            family, family_conf = rule_extractor.detect_family(history, query)
            pred = dict(pred)
            pred["family"] = family
            pred["confidence"] = float((pred["confidence"] + family_conf) / 2.0)
            memory = self._memory_from_prediction(pred, history)
            action, relation, confidence = rule_policy.decide(memory, query)
            memories.append(memory)
            actions.append(action)
            relations.append(relation)
            confidences.append(confidence)
        result["prediction"] = actions
        result["pred_relation"] = relations
        result["decision_confidence"] = confidences
        result["pred_family"] = [m.family for m in memories]
        result["pred_profile_variant"] = [m.profile_variant for m in memories]
        result["pred_owner"] = [m.owner for m in memories]
        result["pred_information_type"] = [m.information_type for m in memories]
        result["pred_temporal_validity"] = [m.temporal_validity for m in memories]
        result["structured_memory"] = [json.dumps(m.to_dict(), ensure_ascii=False) for m in memories]
        responses = [""] * len(result)
        for i in range(min(generate_responses_for, len(result))):
            responses[i] = self.generate_response(actions[i], memories[i], result.loc[i, "query"])
        result["generated_response"] = responses
        return result

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = 128,
        generate_responses_for: int = 24,
    ) -> pd.DataFrame:
        result = df.copy().reset_index(drop=True)
        extracts = self.extract_batch(result["history"].tolist(), batch_size=batch_size)
        memories = [
            self._memory_from_prediction(pred, history)
            for pred, history in zip(extracts, result["history"])
        ]
        decisions = self.decide_batch(memories, result["query"].tolist(), batch_size=batch_size)
        result["prediction"] = [d["action"] for d in decisions]
        result["pred_relation"] = [d["relation"] for d in decisions]
        result["decision_confidence"] = [d["confidence"] for d in decisions]
        result["pred_family"] = [p["family"] for p in extracts]
        result["pred_profile_variant"] = [p["profile_variant"] for p in extracts]
        result["pred_owner"] = [p["owner"] for p in extracts]
        result["pred_information_type"] = [p["information_type"] for p in extracts]
        result["pred_temporal_validity"] = [p["temporal_validity"] for p in extracts]
        result["structured_memory"] = [json.dumps(m.to_dict(), ensure_ascii=False) for m in memories]
        responses = [""] * len(result)
        for i in range(min(generate_responses_for, len(result))):
            responses[i] = self.generate_response(
                decisions[i]["action"], memories[i], result.loc[i, "query"]
            )
        result["generated_response"] = responses
        return result


__all__ = [
    "TinyLMConfig",
    "TinyMultiTaskLM",
    "TinyLLMFrontierMem",
    "SimpleTokenizer",
    "TrainingSummary",
    "train_tiny_lm",
    "evaluate_action_accuracy",
    "extraction_prompt",
    "decision_prompt_from_row",
    "response_prompt_from_row",
]
