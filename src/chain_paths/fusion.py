"""Fusion layers for combining base predictors with auxiliary signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class FusionWeights:
    bias: float = 0.0
    log_interp: float = 0.0
    log_count: float = 0.0
    log_embedding: float = 0.0
    log_prior: float = 0.0
    log_stealth: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            'bias': self.bias,
            'log_interp': self.log_interp,
            'log_count': self.log_count,
            'log_embedding': self.log_embedding,
            'log_prior': self.log_prior,
            'log_stealth': self.log_stealth,
        }


class LogLinearFusion:
    """Log-linear fusion of heterogeneous features."""

    def __init__(self, weights: Dict[str, float], epsilon: float = 1e-12) -> None:
        self.weights = dict(weights)
        self.epsilon = epsilon

    def score(self, features: Dict[str, float]) -> float:
        w = self.weights
        return (
            w.get('bias', 0.0)
            + w.get('log_interp', 0.0) * features['log_p_interp']
            + w.get('log_count', 0.0) * features['log_p_count']
            + w.get('log_embedding', 0.0) * features['log_p_emb']
            + w.get('log_prior', 0.0) * features['log_p_prior']
            + w.get('log_stealth', 0.0) * features['log_stealth']
        )

    def normalize(self, scores: Iterable[float]) -> np.ndarray:
        array = np.asarray(list(scores), dtype=np.float64)
        if array.size == 0:
            return array
        max_score = float(array.max())
        exp_scores = np.exp(array - max_score)
        normalizer = float(exp_scores.sum())
        if normalizer <= 0.0 or not np.isfinite(normalizer):
            return np.full_like(array, 1.0 / len(array))
        return exp_scores / normalizer

    def score_candidates(self, features: List[Dict[str, float]]) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        scores = []
        augmented: List[Dict[str, float]] = []
        for feat in features:
            score = self.score(feat)
            copy = dict(feat)
            copy['logit'] = score
            scores.append(score)
            augmented.append(copy)
        probs = self.normalize(scores)
        for prob, feat in zip(probs, augmented):
            feat['probability'] = float(prob)
        return probs, augmented


__all__ = ["LogLinearFusion", "FusionWeights"]
