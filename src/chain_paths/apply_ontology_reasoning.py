"""Ontology and tactic-aware reasoning layer for predictions."""
from typing import Any, Dict, List

from .predictor import Predictor


def apply_ontology_reasoning(features: Dict[str, Any], predictor: Predictor) -> Dict[str, Any]:
    """Leverage tactic transitions to prioritize compatible techniques."""

    tactic_history: List[str] = features.get("tactic_history", []) or []
    allowed_tactics = set()
    if tactic_history:
        last_tactic = tactic_history[-1]
        transitions = predictor.tactic_transition_probs.get(last_tactic, {})
        allowed_tactics = set(transitions.keys())

    return {
        "allowed_tactics": allowed_tactics,
        "tactic_history": tactic_history,
        "technique_to_tactics": predictor.technique_to_tactics,
    }
