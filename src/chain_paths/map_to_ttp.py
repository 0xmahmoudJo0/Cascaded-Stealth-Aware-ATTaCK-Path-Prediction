"""Map ranked predictions to ATT&CK-oriented payloads."""
from typing import Dict, Iterable, List

from .predict_next_step import Prediction
from .predictor import Predictor


def map_to_ttp(ranked_predictions: Iterable[Prediction], predictor: Predictor) -> List[Dict[str, object]]:
    """Convert internal prediction tuples into structured outputs."""

    mapped: List[Dict[str, object]] = []
    for technique_id, score, details in ranked_predictions:
        mapped.append(
            {
                "technique_id": technique_id,
                "score": score,
                "tactics": predictor.technique_to_tactics.get(technique_id, []),
                "details": details,
            }
        )
    return mapped
