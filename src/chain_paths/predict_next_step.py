"""Prediction wrapper that isolates model inference from the CLI."""
from typing import Any, List, Optional, Tuple

from .predictor import Predictor


Prediction = Tuple[str, float, dict]


def predict_next_step(
    preproc_seq_history: List[str],
    predictor: Predictor,
    *,
    group_context: Optional[Any] = None,
    top_n_candidates: int = 50,
) -> List[Prediction]:
    """Run the core predictor for the next step."""

    return predictor.next_probabilities(
        preproc_seq_history,
        group_context=group_context,
        top_n_candidates=top_n_candidates,
    )
