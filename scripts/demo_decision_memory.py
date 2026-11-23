"""Deterministic toy example: NOT a trained model or empirical research result."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontiermem.decision_memory import Evidence, DecisionProbe, greedy_backward_select, heldout_comparison

def toy_predict(items, probe):
    ids = {e.evidence_id for e in items}
    if "preference" not in ids:
        return "IGNORE"
    if "scope" in ids and probe.context == "exam":
        return "IGNORE"
    return "APPLY"

def main():
    ev = [
        Evidence("preference", "style", "Prefers concise explanations", 6, "toy:1"),
        Evidence("scope", "style", "Only during debugging", 4, "toy:2"),
        Evidence("irrelevant", "style", "Thanks", 7, "toy:3"),
    ]
    select = [
        DecisionProbe("s1", "sel1", "style", "debug", "Fix my Python error", "APPLY", "selection", "toy:s1", "sel-pair", "human_validated"),
        DecisionProbe("s2", "sel2", "style", "exam", "Prove this theorem", "IGNORE", "selection", "toy:s2", "sel-pair", "human_validated"),
    ]
    heldout = [
        DecisionProbe("h1", "test1", "style", "debug", "Debug this function", "APPLY", "heldout", "toy:h1", "test-pair", "human_validated"),
        DecisionProbe("h2", "test2", "style", "exam", "Explain this proof", "IGNORE", "heldout", "toy:h2", "test-pair", "human_validated"),
    ]
    result = greedy_backward_select(ev, select, toy_predict, budget_tokens=10)
    evaluation = heldout_comparison(ev, result["selected_evidence_ids"], select, heldout, toy_predict)
    print(json.dumps({"status": "DEMO_ONLY_NOT_RESEARCH_EVIDENCE", "selection": result, "heldout": evaluation}, indent=2))

if __name__ == "__main__":
    main()
