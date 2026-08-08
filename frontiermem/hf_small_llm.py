from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .data import SPEC_BY_FAMILY, SPECS
from .memory import (
    MemoryItem,
    RuleBasedApplicabilityPolicy,
    RuleBasedMemoryExtractor,
    render_response,
)


ALLOWED_ACTIONS = {"APPLY", "IGNORE", "CLARIFY"}
ALLOWED_RELATIONS = {"match", "mismatch", "unknown"}
ALLOWED_PROFILES = {"stable", "scoped"}

KNOWN_KEYS = (
    "preference",
    "family",
    "profile_variant",
    "owner",
    "information_type",
    "applies_when",
    "does_not_apply_when",
    "temporal_validity",
    "confidence",
    "evidence",
    "permission",
    "action",
    "relation",
    "relationship",
    "response",
    "reason",
)


def _clean_scalar(value: str) -> str:
    value = value.strip().strip(",").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value.strip()


def _coerce_value(key: str, value: str) -> Any:
    value = _clean_scalar(value)
    if key == "confidence":
        match = re.search(r"[-+]?\d*\.?\d+", value)
        return float(match.group(0)) if match else 0.5
    if key == "evidence":
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except Exception:
            pass
        return [x.strip(" -\t\"'") for x in re.split(r"\s*[;|]\s*", value) if x.strip()]
    return value


def _normalize_fields(data: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in data.items():
        key = str(raw_key).strip().lower().replace(" ", "_").replace("-", "_")
        if key == "relationship":
            key = "relation"
        normalized[key] = value

    if "action" in normalized:
        action_text = str(normalized["action"]).upper()
        normalized["action"] = next(
            (action for action in ALLOWED_ACTIONS if action in action_text),
            action_text.strip(),
        )

    if "relation" in normalized:
        relation_text = str(normalized["relation"]).strip().lower()
        if any(token in relation_text for token in ("mismatch", "not match", "outside", "conflict")):
            relation_text = "mismatch"
        elif any(token in relation_text for token in ("unknown", "uncertain", "unclear", "ambiguous")):
            relation_text = "unknown"
        elif any(token in relation_text for token in ("match", "inside", "applies", "relevant")):
            relation_text = "match"
        normalized["relation"] = relation_text

    if "profile_variant" in normalized:
        profile_text = str(normalized["profile_variant"]).strip().lower()
        normalized["profile_variant"] = "stable" if "stable" in profile_text else "scoped"

    if "confidence" in normalized:
        try:
            normalized["confidence"] = float(normalized["confidence"])
        except (TypeError, ValueError):
            match = re.search(r"[-+]?\d*\.?\d+", str(normalized["confidence"]))
            normalized["confidence"] = float(match.group(0)) if match else 0.5

    return normalized


def _parse_loose_key_values(text: str) -> dict[str, Any]:
    """Parse Qwen's common `key: value, key: value` pseudo-JSON output.

    Values may contain commas; boundaries are detected only when a known next key appears.
    """
    key_pattern = "|".join(re.escape(key) for key in sorted(KNOWN_KEYS, key=len, reverse=True))
    pattern = re.compile(rf"(?i)(?<![A-Za-z0-9_])({key_pattern})\s*:\s*")
    matches = list(pattern.finditer(text))
    if not matches:
        return {}

    data: dict[str, Any] = {}
    for index, match in enumerate(matches):
        key = match.group(1).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_value = text[start:end].strip().rstrip(",;\n ")
        data[key] = _coerce_value(key, raw_value)
    return _normalize_fields(data)


def _extract_json(text: str) -> tuple[dict[str, Any], str]:
    """Return a structured object plus the parser mode used.

    Modes: json, python_dict, loose_kv. This intentionally tolerates malformed
    output from very small language models instead of aborting an entire run.
    """
    text = text.strip()

    candidates: list[str] = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        candidates.append(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return _normalize_fields(parsed), "json"
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(cleaned)
            if isinstance(parsed, dict):
                return _normalize_fields(parsed), "python_dict"
        except (ValueError, SyntaxError):
            pass

    loose = _parse_loose_key_values(text)
    if loose:
        return loose, "loose_kv"

    raise ValueError(f"No structured object found in model output: {text[:300]}")


@dataclass
class HFGenerationConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 280
    temperature: float = 0.0
    torch_dtype: str = "auto"
    attn_implementation: str = "eager"


class HFSmallLLMFrontierMem:
    """Zero/few-shot FrontierMem pipeline using a small Hugging Face chat model.

    The implementation accepts strict JSON, Python-style dictionaries, and
    Qwen's common `key: value` output. If extraction is still impossible, it
    uses an explicit rule fallback and records that fallback in the CSV.
    """

    name = "Qwen2.5-1.5B-Instruct"

    def __init__(self, config: HFGenerationConfig | None = None, device_map: str = "auto") -> None:
        config = config or HFGenerationConfig()
        self.config = config
        self.name = config.model_name.split("/")[-1]
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The Hugging Face backend needs transformers. Install requirements-hf.txt first."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=config.torch_dtype,
            device_map=device_map,
            attn_implementation=config.attn_implementation,
        )
        self.model.eval()

        # Qwen's saved generation config may contain sampling parameters. Clear
        # them for deterministic generation to avoid misleading warnings.
        if config.temperature <= 0:
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None

        self.rule_extractor = RuleBasedMemoryExtractor()
        self.rule_policy = RuleBasedApplicabilityPolicy()

    def _chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.temperature > 0:
            kwargs["temperature"] = self.config.temperature
        output = self.model.generate(**inputs, **kwargs)
        generated = output[:, inputs.input_ids.shape[1] :]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def extract_memory(self, history: str, query: str = "") -> tuple[MemoryItem, str, str]:
        families = ", ".join(spec.family for spec in SPECS)
        system = (
            "You extract an evidence-grounded user preference boundary. "
            "Return one strict JSON object only. The first character must be { and the last } . "
            "Use double quotes around every key and string value. Do not use markdown, YAML, "
            "comments, or prose outside the JSON. Do not infer a global preference from a "
            "temporary state, external constraint, hypothetical statement, quotation, or "
            "another person's preference."
        )
        user = f"""
Read the interaction history and return exactly one JSON object with this shape:
{{"preference":"","family":"","profile_variant":"","owner":"","information_type":"","applies_when":"","does_not_apply_when":"","temporal_validity":"","confidence":0.0,"evidence":[],"permission":"response_personalization"}}

Allowed family values: {families}
Allowed profile_variant values: stable, scoped. Choose exactly one.
Allowed temporal_validity values: persistent, context-dependent. Choose exactly one.
Use stable only when the preference generalizes across relevant contexts.
Use scoped when the behavior is caused by a temporary state, goal, external constraint, role, audience, or local context.
Evidence must contain at most three exact sentences copied from the history.
Confidence must be a number from 0 to 1.

Stable example:
History: "I consistently prefer direct answers across tasks, even when I have extra time."
Expected profile_variant: stable. Expected temporal_validity: persistent.

Scoped example:
History: "For this trip only, choose a cheap hotel because the university reimbursement cap is strict. This is not my general preference."
Expected profile_variant: scoped. Expected information_type: temporary_constraint. Expected temporal_validity: context-dependent.

HISTORY:
{history}
""".strip()
        raw = self._chat(system, user)
        try:
            data, parse_mode = _extract_json(raw)
        except ValueError:
            fallback = self.rule_extractor.extract(history, query)
            return fallback, "rule_fallback", raw

        family = str(data.get("family", "")).strip()
        if family not in SPEC_BY_FAMILY:
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
        memory = MemoryItem(
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
        return memory, parse_mode, raw

    def decide_and_respond(self, memory: MemoryItem, query: str) -> dict[str, Any]:
        system = (
            "You are a boundary-aware personalized assistant. Return one strict JSON object only. "
            "The first character must be { and the last } . Use double quotes around all keys and "
            "string values. Never use the key relationship; use relation. Do not use markdown, YAML, "
            "or prose outside the JSON. Use APPLY when the scope clearly matches, IGNORE when it "
            "clearly does not, and CLARIFY only when a material missing fact prevents a safe decision."
        )
        user = f"""
Decision rules:
1. If profile_variant is stable, choose APPLY unless the current query explicitly overrides or corrects the preference.
2. If profile_variant is scoped and the query satisfies applies_when, choose relation=match and action=APPLY.
3. If profile_variant is scoped and the query satisfies does_not_apply_when, choose relation=mismatch and action=IGNORE.
4. Choose relation=unknown and action=CLARIFY only when a material fact needed to distinguish APPLY from IGNORE is genuinely missing.
5. Never choose CLARIFY merely because confidence is below 1.0.

Examples:
- scoped; applies_when=self-funded; does_not_apply_when=fully reimbursed; query=company pays fully -> action=IGNORE, relation=mismatch.
- scoped; same boundary; query=I have not checked who pays -> action=CLARIFY, relation=unknown.
- stable preference; query changes payment source but does not override the preference -> action=APPLY, relation=match.

MEMORY:
{memory.to_json()}

CURRENT QUERY:
{query}

Return exactly one JSON object with this shape:
{{"action":"","relation":"","confidence":0.0,"response":"","reason":""}}
Allowed action values: APPLY, IGNORE, CLARIFY. Choose exactly one.
Allowed relation values: match, mismatch, unknown. Choose exactly one.
""".strip()
        raw = self._chat(system, user)
        try:
            data, parse_mode = _extract_json(raw)
        except ValueError:
            # Keep the run alive while making the fallback explicit in the output.
            action, relation, confidence = self.rule_policy.decide(memory, query)
            return {
                "action": action,
                "relation": relation,
                "confidence": confidence,
                "response": render_response(action, memory, query),
                "reason": "Rule fallback used because the small LLM output was not parseable.",
                "raw_output": raw,
                "decision_parse_mode": "rule_fallback",
            }

        action = str(data.get("action", "CLARIFY")).upper()
        if action not in ALLOWED_ACTIONS:
            action = next((x for x in ALLOWED_ACTIONS if x in action), "CLARIFY")
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
            "decision_parse_mode": parse_mode,
        }

    def predict_one(self, history: str, query: str) -> dict[str, Any]:
        memory, memory_parse_mode, memory_raw = self.extract_memory(history, query)
        decision = self.decide_and_respond(memory, query)
        return {
            "memory": memory.to_dict(),
            "memory_parse_mode": memory_parse_mode,
            "memory_raw_output": memory_raw,
            **decision,
        }

    def predict_dataframe(self, df: pd.DataFrame, max_examples: int | None = None) -> pd.DataFrame:
        work = (
            df.head(max_examples).copy().reset_index(drop=True)
            if max_examples
            else df.copy().reset_index(drop=True)
        )
        outputs: list[dict[str, Any]] = []
        total = len(work)
        for index, row in enumerate(work.itertuples(), start=1):
            result = self.predict_one(row.history, row.query)
            outputs.append(result)
            print(
                f"[{index}/{total}] family={row.family} | gold={row.action} | prediction={result['action']}",
                flush=True,
            )

        work["prediction"] = [x["action"] for x in outputs]
        work["pred_relation"] = [x["relation"] for x in outputs]
        work["decision_confidence"] = [x["confidence"] for x in outputs]
        work["generated_response"] = [x["response"] for x in outputs]
        work["decision_reason"] = [x["reason"] for x in outputs]
        work["structured_memory"] = [json.dumps(x["memory"], ensure_ascii=False) for x in outputs]
        work["memory_parse_mode"] = [x["memory_parse_mode"] for x in outputs]
        work["decision_parse_mode"] = [x["decision_parse_mode"] for x in outputs]
        work["memory_raw_output"] = [x["memory_raw_output"] for x in outputs]
        work["decision_raw_output"] = [x["raw_output"] for x in outputs]
        work["pred_family"] = [x["memory"]["family"] for x in outputs]
        work["pred_profile_variant"] = [x["memory"]["profile_variant"] for x in outputs]
        work["pred_owner"] = [x["memory"]["owner"] for x in outputs]
        work["pred_information_type"] = [x["memory"]["information_type"] for x in outputs]
        work["pred_temporal_validity"] = [x["memory"]["temporal_validity"] for x in outputs]
        return work


__all__ = ["HFGenerationConfig", "HFSmallLLMFrontierMem"]
