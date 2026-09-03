"""Feature extraction step for the staged attack pipeline."""
from typing import Any, Dict, Optional

from .attack_sequence import AttackSequence
from .predictor import Predictor


def extract_features(preproc_seq: AttackSequence, predictor: Predictor, *, group_context: Optional[Any] = None) -> Dict[str, Any]:
    """Derive features for downstream reasoning and prediction."""

    tactic_history = []
    for technique in preproc_seq.path:
        tactic_history.extend(predictor.technique_to_tactics.get(technique, []))

    features: Dict[str, Any] = {
        "history": list(preproc_seq.path),
        "tactic_history": tactic_history,
        "group_context": group_context,
    }

    base_model = getattr(predictor, "base_model", None)
    if hasattr(base_model, "augment_features"):
        try:
            neural_features = base_model.augment_features(preproc_seq.path, tactic_history)
            features.update(neural_features)
        except Exception:
            # Non-fatal: the neural model may be unavailable in lightweight envs
            pass

    return features
