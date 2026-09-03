"""Evidence fusion and candidate prioritization helpers."""
from typing import Dict, Iterable, List, Tuple

from .predict_next_step import Prediction


def fuse_evidence(
    model_output: Iterable[Prediction],
    reasoning_output: Dict[str, object],
    *,
    max_candidates: int,
) -> List[Prediction]:
    """Combine model scores with ontology constraints."""

    allowed_tactics = reasoning_output.get("allowed_tactics") or set()
    technique_to_tactics = reasoning_output.get("technique_to_tactics") or {}

    filtered: List[Prediction] = []
    for entry in model_output:
        technique, score, _ = entry
        tactics = technique_to_tactics.get(technique, [])
        if allowed_tactics and not any(t in allowed_tactics for t in tactics):
            continue
        filtered.append(entry)

    def _score(entry: Prediction) -> Tuple[int, float]:
        technique, score, _ = entry
        tactics = technique_to_tactics.get(technique, [])
        compatible = 1 if allowed_tactics and any(t in allowed_tactics for t in tactics) else 0
        return (-compatible, -score)

    ranked = sorted(filtered, key=_score)
    return ranked[:max_candidates]


def prioritize_candidates(
    model_output: Iterable[Prediction],
    reasoning_output: Dict[str, object],
    *,
    max_candidates: int,
) -> List[Prediction]:
    """Alias for clarity in the staged pipeline."""

    return fuse_evidence(model_output, reasoning_output, max_candidates=max_candidates)
