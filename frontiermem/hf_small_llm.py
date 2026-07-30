from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .data import SPEC_BY_FAMILY, SPECS
from .memory import MemoryItem


ALLOWED_ACTIONS = {"APPLY", "IGNORE", "CLARIFY"}
ALLOWED_RELATIONS = {"match", "mismatch", "unknown"}
ALLOWED_PROFILES = {"stable", "scoped"}


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")
    return json.loads(candidate)


@dataclass
class HFGenerationConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 280
    temperature: float = 0.0
    torch_dtype: str = "auto"


class HFSmallLLMFrontierMem:
    """Zero/few-shot FrontierMem pipeline using a small Hugging Face chat model.

    This backend is optional so the package remains runnable without transformers.
    It uses the LLM for both structured memory extraction and the final
    Apply/Clarify/Ignore decision with a natural-language response.
    """

    name = "Qwen2.5-0.5B-Instruct"

    def __init__(self, config: HFGenerationConfig | None = None, device_map: str = "auto") -> None:
        config = config or HFGenerationConfig()
        self.config = config
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The Hugging Face backend needs transformers. Install requirements-hf.txt first."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.torch_dtype,
            device_map=device_map,
        )
        self.model.eval()

    def _chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.temperature > 0:
            kwargs["temperature"] = self.config.temperature
        output = self.model.generate(**inputs, **kwargs)
        generated = output[:, inputs.input_ids.shape[1] :]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def extract_memory(self, history: str) -> MemoryItem:
        families = ", ".join(spec.family for spec in SPECS)
        system = (
            "You extract an evidence-grounded user preference boundary. Return JSON only. "
            "Do not infer a global preference from a temporary state, external constraint, "
            "hypothetical statement, quotation, or another person's preference."
        )
        user = f"""
Read the interaction history and return exactly one JSON object with these keys:
preference, family, profile_variant, owner, information_type, applies_when,
does_not_apply_when, temporal_validity, confidence, evidence, permission.

Allowed family values: {families}
Allowed profile_variant: stable or scoped
Allowed temporal_validity: persistent or context-dependent
Evidence must be a list of up to three exact sentences copied from the history.
Confidence must be between 0 and 1.

HISTORY:
{history}
""".strip()
        raw = self._chat(system, user)
        data = _extract_json(raw)
        family = str(data.get("family", "")).strip()
        if family not in SPEC_BY_FAMILY:
            # Conservative family fallback based on the generated preference text.
            pref = str(data.get("preference", "")).lower()
            family = max(
                SPEC_BY_FAMILY,
                key=lambda f: sum(w in pref for w in SPEC_BY_FAMILY[f].preference_label.split()),
            )
        profile = str(data.get("profile_variant", "scoped")).lower()
        if profile not in ALLOWED_PROFILES:
            profile = "scoped"
        spec = SPEC_BY_FAMILY[family]
        evidence = data.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return MemoryItem(
            preference=str(data.get("preference") or spec.preference_label),
            family=family,
            owner=str(data.get("owner", "user")),
            information_type=str(
                data.get(
                    "information_type",
                    "trait_preference" if profile == "stable" else spec.scoped_type,
                )
            ),
            applies_when=str(
                data.get(
                    "applies_when",
                    "broadly across relevant situations" if profile == "stable" else spec.applies_when,
                )
            ),
            does_not_apply_when=str(
                data.get(
                    "does_not_apply_when",
                    "only when explicitly overridden or corrected"
                    if profile == "stable"
                    else spec.does_not_apply_when,
                )
            ),
            temporal_validity=str(
                data.get(
                    "temporal_validity",
                    "persistent" if profile == "stable" else "context-dependent",
                )
            ),
            confidence=max(0.0, min(1.0, confidence)),
            evidence=[str(x) for x in evidence[:3]],
            permission=str(data.get("permission", "response_personalization")),
            profile_variant=profile,
        )

    def decide_and_respond(self, memory: MemoryItem, query: str) -> dict[str, Any]:
        system = (
            "You are a boundary-aware personalized assistant. Decide whether the stored "
            "preference applies to the current query. Return JSON only. Use APPLY when the "
            "scope clearly matches, IGNORE when it clearly does not, and CLARIFY when a "
            "material missing fact prevents a safe decision. Ask at most one minimal question."
        )
        user = f"""
MEMORY:
{memory.to_json()}

CURRENT QUERY:
{query}

Return a JSON object with:
action: APPLY, IGNORE, or CLARIFY
relation: match, mismatch, or unknown
confidence: number from 0 to 1
response: the final personalized response, neutral response, or one clarification question
reason: one short sentence grounded in the memory scope
""".strip()
        raw = self._chat(system, user)
        data = _extract_json(raw)
        action = str(data.get("action", "CLARIFY")).upper()
        if action not in ALLOWED_ACTIONS:
            action = "CLARIFY"
        relation = str(data.get("relation", "unknown")).lower()
        if relation not in ALLOWED_RELATIONS:
            relation = "unknown"
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return {
            "action": action,
            "relation": relation,
            "confidence": max(0.0, min(1.0, confidence)),
            "response": str(data.get("response", "Could you clarify the relevant context?")),
            "reason": str(data.get("reason", "")),
            "raw_output": raw,
        }

    def predict_one(self, history: str, query: str) -> dict[str, Any]:
        memory = self.extract_memory(history)
        decision = self.decide_and_respond(memory, query)
        return {"memory": memory.to_dict(), **decision}

    def predict_dataframe(self, df: pd.DataFrame, max_examples: int | None = None) -> pd.DataFrame:
        work = df.head(max_examples).copy().reset_index(drop=True) if max_examples else df.copy().reset_index(drop=True)
        outputs = [self.predict_one(row.history, row.query) for row in work.itertuples()]
        work["prediction"] = [x["action"] for x in outputs]
        work["pred_relation"] = [x["relation"] for x in outputs]
        work["decision_confidence"] = [x["confidence"] for x in outputs]
        work["generated_response"] = [x["response"] for x in outputs]
        work["structured_memory"] = [json.dumps(x["memory"], ensure_ascii=False) for x in outputs]
        work["pred_family"] = [x["memory"]["family"] for x in outputs]
        work["pred_profile_variant"] = [x["memory"]["profile_variant"] for x in outputs]
        work["pred_owner"] = [x["memory"]["owner"] for x in outputs]
        work["pred_information_type"] = [x["memory"]["information_type"] for x in outputs]
        work["pred_temporal_validity"] = [x["memory"]["temporal_validity"] for x in outputs]
        return work


__all__ = ["HFGenerationConfig", "HFSmallLLMFrontierMem"]
