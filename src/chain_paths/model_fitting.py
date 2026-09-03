"""Learning routines for the log-linear predictor weights.

Three-pillar architecture: LSTM + Tactic Prior + Stealth Score
Removed components (redundant with LSTM): count, embedding, prior, causal, mcdm
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config as cfg
from .predictor import Predictor, WEIGHT_ORDER


@dataclass
class TrainingExample:
    """Container for a single next-step prediction context."""

    features: List[Tuple[str, Dict[str, float]]]
    target: str


def _build_group_label_index(metadata: Optional[pd.DataFrame], n_sequences: int) -> List[Optional[str]]:
    # Return empty labels since we're not using group context anymore
    return [None] * n_sequences


def _chronological_indices(metadata: Optional[pd.DataFrame], n_sequences: int) -> List[int]:
    if metadata is None or 'start_time' not in metadata.columns:
        return list(range(n_sequences))

    meta = metadata[['sequence_id', 'start_time']].dropna()
    if meta.empty:
        return list(range(n_sequences))

    meta = meta.copy()
    meta['start_time'] = pd.to_datetime(meta['start_time'], errors='coerce')
    meta = meta.dropna(subset=['start_time']).sort_values('start_time')
    return meta['sequence_id'].astype(int).tolist()


def _gather_training_examples(
    predictor: Predictor,
    sequences: Sequence[Sequence[str]],
    sequence_order: Sequence[int],
    group_labels: Sequence[Optional[str]],
    top_n_candidates: int = 50,
) -> List[TrainingExample]:
    examples: List[TrainingExample] = []

    for seq_idx in sequence_order:
        sequence = sequences[seq_idx]
        if len(sequence) < 2:
            continue

        for pos in range(1, len(sequence)):
            history = list(sequence[:pos])
            target = sequence[pos]

            candidate_components = predictor.next_probabilities(
                history,
                group_context=None,
                top_n_candidates=top_n_candidates,
            )

            if not candidate_components:
                continue

            component_lookup = {cand: details for cand, _, details in candidate_components}
            if target not in component_lookup:
                continue

            feature_tuples = []
            for cand, _, comp in candidate_components:
                log_terms = comp.get('log_terms', {})
                # Three-pillar: LSTM + Tactic + Stealth
                feature_tuples.append(
                    (
                        cand,
                        {
                            'bias': 1.0,
                            'log_bilstm': log_terms.get('log_p_bilstm', 0.0),
                            'log_tactic': log_terms.get('log_p_tactic', 0.0),
                            'log_stealth': log_terms.get('log_p_stealth', 0.0),
                        },
                    )
                )

            examples.append(TrainingExample(feature_tuples, target))

    return examples


def _vectorise_weights(weights: Dict[str, float]) -> np.ndarray:
    return np.array([weights.get(key, 0.0) for key in WEIGHT_ORDER], dtype=np.float64)


def _devectorise_weights(vector: np.ndarray) -> Dict[str, float]:
    return {key: float(value) for key, value in zip(WEIGHT_ORDER, vector)}


def _objective_and_gradient(
    weight_vector: np.ndarray,
    examples: Sequence[TrainingExample],
    regularization: float,
) -> Tuple[float, np.ndarray]:
    total_loss = 0.0
    gradient = np.zeros_like(weight_vector)

    for example in examples:
        scores = []
        feature_matrix = []
        labels = []

        for candidate, feature_dict in example.features:
            # Three-pillar feature vector: bias + LSTM + Tactic + Stealth
            feature_vector = np.array([
                feature_dict['bias'],
                feature_dict['log_bilstm'],
                feature_dict['log_tactic'],
                feature_dict['log_stealth'],
            ], dtype=np.float64)
            feature_matrix.append(feature_vector)
            labels.append(candidate)

        feature_matrix_np = np.vstack(feature_matrix)
        logits = feature_matrix_np @ weight_vector
        max_logit = logits.max()
        exp_logits = np.exp(logits - max_logit)
        denom = exp_logits.sum()
        probs = exp_logits / denom

        try:
            target_index = labels.index(example.target)
        except ValueError:
            continue

        total_loss -= math.log(probs[target_index])

        expected_features = probs @ feature_matrix_np
        target_features = feature_matrix_np[target_index]
        gradient += expected_features - target_features

    total_loss += regularization * float(weight_vector @ weight_vector)
    gradient += 2 * regularization * weight_vector

    return total_loss, gradient


def fit_component_weights(
    predictor: Predictor,
    sequences: Sequence[Sequence[str]],
    metadata: Optional[pd.DataFrame] = None,
    train_ratio: float = 0.8,
    top_n_candidates: int = 50,
    regularization: Optional[float] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Fit the log-linear weights by maximising the likelihood on training data."""

    n_sequences = len(sequences)
    ordered_indices = _chronological_indices(metadata, n_sequences)
    if not ordered_indices:
        ordered_indices = list(range(n_sequences))

    n_train = max(1, int(len(ordered_indices) * train_ratio))
    train_indices = ordered_indices[:n_train]

    group_labels = _build_group_label_index(metadata, n_sequences)

    examples = _gather_training_examples(
        predictor,
        sequences,
        train_indices,
        group_labels,
        top_n_candidates=top_n_candidates,
    )

    if not examples:
        raise ValueError("Unable to assemble training examples. Check data coverage and candidate generation.")

    print(f"Collected {len(examples)} training examples for weight fitting")

    current_weights = predictor.params.get('component_weights', cfg.DEFAULT_PARAMS['component_weights'])
    weight_vector = _vectorise_weights(current_weights)
    reg = regularization if regularization is not None else predictor.params.get('logit_regularization', 1.0)

    def objective(vec: np.ndarray) -> Tuple[float, np.ndarray]:
        loss, grad = _objective_and_gradient(vec, examples, reg)
        return loss, grad

    result = minimize(
        fun=lambda v: objective(v)[0],
        x0=weight_vector,
        method='L-BFGS-B',
        jac=lambda v: objective(v)[1],
        options={'maxiter': 200, 'disp': True},
    )

    if not result.success:
        raise RuntimeError(f"Weight fitting failed: {result.message}")

    learned_weights = _devectorise_weights(result.x)
    predictor.update_component_weights(learned_weights)

    output_path = output_path or cfg.COMPONENT_WEIGHTS_JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(learned_weights, f, indent=2)

    print(f"Saved learned component weights to {output_path}")
    return learned_weights
