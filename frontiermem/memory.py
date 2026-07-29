from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import re

from .data import SPECS, ScenarioSpec


@dataclass
class MemoryItem:
    preference: str
    owner: str
    information_type: str
    applies_when: str
    does_not_apply_when: str
    temporal_validity: str
    confidence: float
    evidence: List[str]
    permission: str = "response_personalization"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class FrontierMemoryExtractor:
    """Lightweight, interpretable extractor for the initial proof-of-concept.

    The extractor intentionally uses transparent lexical evidence rather than an
    external LLM so the complete demo can run offline. It is designed as a drop-in
    interface: a future LLM/SFT extractor can return the same MemoryItem schema.
    """

    STABLE_CUES = (
        "generally", "consistently", "persistent", "stable", "across", "almost always",
        "even when", "every subject", "default", "long-term", "for several years",
    )
    SCOPED_CUES = (
        "only", "this trip", "until", "depends on", "temporary", "tied to", "limited to",
        "when i am", "for professors", "my sister", "rather than", "role-specific",
        "current", "strict", "because", "during", "not a general", "not all",
    )

    def __init__(self) -> None:
        self.spec_by_family = {spec.family: spec for spec in SPECS}

    @staticmethod
    def _sentences(history: str) -> List[str]:
        return [line.split(":", 1)[-1].strip() for line in history.splitlines() if line.strip()]

    def detect_family(self, history: str, query: str = "") -> Tuple[str, float]:
        text = f"{history}\n{query}".lower()
        best_family = "unknown"
        best_score = -1
        for spec in SPECS:
            tokens = set(re.findall(r"[a-z]+", spec.preference_label.lower()))
            anchor_tokens = set(re.findall(r"[a-z]+", spec.surface_anchor.lower()))
            terms = tokens | {t for t in anchor_tokens if len(t) > 4}
            score = sum(text.count(term) for term in terms)
            # Domain/family-specific discriminators.
            score += 3 * sum(1 for term in list(spec.match_terms) + list(spec.mismatch_terms) if term.lower() in text)
            if score > best_score:
                best_score = score
                best_family = spec.family
        confidence = min(0.99, 0.55 + 0.04 * max(best_score, 0))
        return best_family, confidence

    def infer_profile_variant(self, history: str) -> Tuple[str, float]:
        text = history.lower()
        stable_score = sum(text.count(cue) for cue in self.STABLE_CUES)
        scoped_score = sum(text.count(cue) for cue in self.SCOPED_CUES)
        if stable_score >= scoped_score:
            margin = stable_score - scoped_score
            return "stable", min(0.98, 0.72 + 0.05 * margin)
        margin = scoped_score - stable_score
        return "scoped", min(0.98, 0.72 + 0.05 * margin)

    def infer_relation(self, family: str, query: str) -> Tuple[str, float]:
        if family not in self.spec_by_family:
            return "unknown", 0.34
        spec = self.spec_by_family[family]
        q = query.lower()
        match_hits = sum(term.lower() in q for term in spec.match_terms)
        mismatch_hits = sum(term.lower() in q for term in spec.mismatch_terms)
        if match_hits > mismatch_hits and match_hits > 0:
            return "match", min(0.98, 0.76 + 0.06 * match_hits)
        if mismatch_hits > match_hits and mismatch_hits > 0:
            return "mismatch", min(0.98, 0.76 + 0.06 * mismatch_hits)
        return "unknown", 0.58

    def extract(self, history: str, query: str = "") -> MemoryItem:
        family, family_conf = self.detect_family(history, query)
        variant, variant_conf = self.infer_profile_variant(history)
        spec = self.spec_by_family.get(family)
        sentences = self._sentences(history)
        evidence = []
        if spec is not None:
            keywords = [w for w in re.findall(r"[a-z]+", spec.preference_label.lower()) if len(w) > 4]
            evidence = [s for s in sentences if any(k in s.lower() for k in keywords)]
            if not evidence:
                evidence = sentences[1:4]
        else:
            evidence = sentences[:3]

        confidence = round((family_conf + variant_conf) / 2, 3)
        if variant == "stable":
            return MemoryItem(
                preference=spec.preference_label if spec else "unknown preference",
                owner="user",
                information_type="trait_preference",
                applies_when="broadly across relevant situations",
                does_not_apply_when="only when explicitly overridden by the user",
                temporal_validity="persistent",
                confidence=confidence,
                evidence=evidence[:3],
            )

        scoped_type = spec.scoped_type if spec else "contextual_preference"
        owner = "other" if scoped_type == "other_owner" else "user"
        temporal = "temporary" if scoped_type == "temporary_constraint" else "context-dependent"
        return MemoryItem(
            preference=spec.preference_label if spec else "unknown preference",
            owner=owner,
            information_type=scoped_type,
            applies_when=spec.applies_when if spec else "unknown",
            does_not_apply_when=spec.not_when if spec else "unknown",
            temporal_validity=temporal,
            confidence=confidence,
            evidence=evidence[:3],
        )


class FrontierPolicy:
    def __init__(self, extractor: FrontierMemoryExtractor | None = None) -> None:
        self.extractor = extractor or FrontierMemoryExtractor()

    def predict_one(self, history: str, query: str) -> Dict[str, object]:
        memory = self.extractor.extract(history, query)
        family, _ = self.extractor.detect_family(history, query)
        relation, relation_conf = self.extractor.infer_relation(family, query)

        if memory.information_type == "trait_preference":
            action = "APPLY"
        elif relation == "match":
            action = "APPLY"
        elif relation == "mismatch":
            action = "IGNORE"
        else:
            action = "CLARIFY"

        return {
            "prediction": action,
            "memory": memory.to_dict(),
            "relation": relation,
            "relation_confidence": relation_conf,
            "family": family,
        }

    def predict(self, histories: List[str], queries: List[str]) -> List[str]:
        return [self.predict_one(h, q)["prediction"] for h, q in zip(histories, queries)]
