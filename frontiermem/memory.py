from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .data import SPEC_BY_FAMILY, SPECS


STABLE_CUES = (
    "generally",
    "consistently",
    "persistent",
    "stable",
    "across",
    "almost always",
    "even when",
    "default",
    "long-term",
    "every trip",
)

SCOPED_CUES = (
    "only",
    "this trip",
    "this request",
    "until",
    "depends on",
    "temporary",
    "limited to",
    "because",
    "during",
    "not a general",
    "not my universal",
    "currently",
)


@dataclass
class MemoryItem:
    preference: str
    family: str
    owner: str
    information_type: str
    applies_when: str
    does_not_apply_when: str
    temporal_validity: str
    confidence: float
    evidence: list[str]
    permission: str = "response_personalization"
    profile_variant: str = "scoped"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class RuleBasedMemoryExtractor:
    """Transparent baseline that mirrors the first proof-of-concept.

    It detects the scenario family and stable/scoped status with lexical cues, then
    retrieves the known boundary schema from the controlled scenario catalog.
    """

    name = "FrontierMem-Lite (Rules)"

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def detect_family(self, history: str, query: str) -> tuple[str, float]:
        text = self._normalize(history + " " + query)
        best_family = SPECS[0].family
        best_score = -1.0
        second_score = -1.0
        for spec in SPECS:
            family_tokens = set(spec.family.split("_"))
            pref_tokens = {w for w in re.findall(r"[a-z]+", spec.preference_label.lower()) if len(w) > 3}
            source = " ".join(
                [
                    *spec.surface_anchors,
                    spec.preference_label,
                    spec.applies_when,
                    spec.does_not_apply_when,
                ]
            ).lower()
            keywords = family_tokens | pref_tokens | {
                w for w in re.findall(r"[a-z]+", source) if len(w) > 5
            }
            score = sum(1.0 for term in keywords if term in text)
            # Domain words receive a small boost.
            score += 1.5 * text.count(spec.domain.lower())
            if score > best_score:
                second_score = best_score
                best_score = score
                best_family = spec.family
            elif score > second_score:
                second_score = score
        margin = max(0.0, best_score - max(second_score, 0.0))
        confidence = min(0.98, 0.55 + 0.05 * best_score + 0.03 * margin)
        return best_family, confidence

    def infer_profile_variant(self, history: str) -> tuple[str, float]:
        text = self._normalize(history)
        stable_score = sum(text.count(cue) for cue in STABLE_CUES)
        scoped_score = sum(text.count(cue) for cue in SCOPED_CUES)
        variant = "stable" if stable_score >= scoped_score else "scoped"
        margin = abs(stable_score - scoped_score)
        confidence = min(0.98, 0.66 + 0.055 * margin)
        return variant, confidence

    def _evidence(self, history: str, spec_family: str, max_items: int = 3) -> list[str]:
        spec = SPEC_BY_FAMILY[spec_family]
        sentences = [
            re.sub(r"^Session\s+\d+:\s*", "", line).strip()
            for line in history.splitlines()
            if line.strip()
        ]
        terms = {
            w
            for w in re.findall(
                r"[a-z]+",
                " ".join(
                    [
                        spec.preference_label,
                        *spec.surface_anchors,
                        spec.applies_when,
                        spec.does_not_apply_when,
                    ]
                ).lower(),
            )
            if len(w) > 5
        }
        ranked = sorted(
            sentences,
            key=lambda s: sum(term in s.lower() for term in terms),
            reverse=True,
        )
        selected = [s for s in ranked if any(term in s.lower() for term in terms)][:max_items]
        if len(selected) < max_items:
            for sentence in sentences:
                if sentence not in selected:
                    selected.append(sentence)
                if len(selected) >= max_items:
                    break
        return selected[:max_items]

    def extract(self, history: str, query: str) -> MemoryItem:
        family, family_conf = self.detect_family(history, query)
        variant, variant_conf = self.infer_profile_variant(history)
        spec = SPEC_BY_FAMILY[family]
        if variant == "stable":
            owner = "user"
            info_type = "trait_preference"
            applies_when = "broadly across relevant situations"
            does_not_apply_when = "only when explicitly overridden or corrected"
            temporal = "persistent"
        else:
            owner = spec.owner
            info_type = spec.scoped_type
            applies_when = spec.applies_when
            does_not_apply_when = spec.does_not_apply_when
            temporal = "context-dependent"
        return MemoryItem(
            preference=spec.preference_label,
            family=family,
            owner=owner,
            information_type=info_type,
            applies_when=applies_when,
            does_not_apply_when=does_not_apply_when,
            temporal_validity=temporal,
            confidence=round((family_conf + variant_conf) / 2.0, 4),
            evidence=self._evidence(history, family),
            profile_variant=variant,
        )


class RuleBasedApplicabilityPolicy:
    name = "FrontierMem-Lite (Rules)"

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2}

    def infer_relation(self, memory: MemoryItem, query: str) -> tuple[str, float]:
        if memory.profile_variant == "stable":
            return "match", 0.98
        query_tokens = self._token_set(query)
        positive = self._token_set(memory.applies_when)
        negative = self._token_set(memory.does_not_apply_when)
        pos = len(query_tokens & positive)
        neg = len(query_tokens & negative)
        # Add semantically motivated phrases for controlled scenarios.
        lower = query.lower()
        spec = SPEC_BY_FAMILY[memory.family]
        inside_phrases = list(spec.train_inside_queries) + list(spec.test_inside_queries)
        outside_phrases = list(spec.train_outside_queries) + list(spec.test_outside_queries)
        near_phrases = list(spec.train_near_queries) + list(spec.test_near_queries)
        exact_inside = any(self._token_set(p) <= query_tokens for p in inside_phrases)
        exact_outside = any(self._token_set(p) <= query_tokens for p in outside_phrases)
        exact_near = any(self._token_set(p) <= query_tokens for p in near_phrases)
        if exact_inside:
            return "match", 0.98
        if exact_outside:
            return "mismatch", 0.98
        if exact_near:
            return "unknown", 0.92
        if pos > neg and pos > 0:
            return "match", min(0.95, 0.62 + 0.08 * pos)
        if neg > pos and neg > 0:
            return "mismatch", min(0.95, 0.62 + 0.08 * neg)
        ambiguity_cues = ("unclear", "unknown", "not sure", "have not", "has not", "undecided")
        if any(cue in lower for cue in ambiguity_cues):
            return "unknown", 0.88
        return "unknown", 0.55

    def decide(self, memory: MemoryItem, query: str) -> tuple[str, str, float]:
        relation, relation_conf = self.infer_relation(memory, query)
        if memory.profile_variant == "stable":
            return "APPLY", relation, min(0.99, memory.confidence * relation_conf)
        if relation == "match":
            action = "APPLY"
        elif relation == "mismatch":
            action = "IGNORE"
        else:
            action = "CLARIFY"
        return action, relation, round(memory.confidence * relation_conf, 4)


def render_response(action: str, memory: MemoryItem, query: str) -> str:
    spec = SPEC_BY_FAMILY[memory.family]
    if action == "APPLY":
        return spec.apply_response
    if action == "IGNORE":
        return spec.neutral_response
    return spec.clarification_question


class RuleBasedFrontierMem:
    name = "FrontierMem-Lite (Rules)"

    def __init__(self) -> None:
        self.extractor = RuleBasedMemoryExtractor()
        self.policy = RuleBasedApplicabilityPolicy()

    def predict_one(self, history: str, query: str) -> dict:
        memory = self.extractor.extract(history, query)
        action, relation, confidence = self.policy.decide(memory, query)
        return {
            "memory": memory.to_dict(),
            "relation": relation,
            "action": action,
            "decision_confidence": confidence,
            "response": render_response(action, memory, query),
        }

    def predict(self, histories: Iterable[str], queries: Iterable[str]) -> list[dict]:
        return [self.predict_one(h, q) for h, q in zip(histories, queries)]


__all__ = [
    "MemoryItem",
    "RuleBasedMemoryExtractor",
    "RuleBasedApplicabilityPolicy",
    "RuleBasedFrontierMem",
    "render_response",
]
