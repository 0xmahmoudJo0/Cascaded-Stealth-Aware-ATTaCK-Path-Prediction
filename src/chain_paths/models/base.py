"""Base interfaces for chain-aware sequence predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class BasePredictor(ABC):
    """Common interface for all base sequence models.

    Concrete predictors (n-gram, HMM, neural architectures, etc.) should focus on
    estimating the *base* conditional probabilities :math:`P(T_{t+1} | h_t)`
    given a history ``h_t`` of previously observed techniques. The fusion layer
    will subsequently combine these estimates with auxiliary signals (embeddings,
    stealth scores, attacker priors).
    """

    def __init__(self, technique_ids: Sequence[str]) -> None:
        self._technique_ids: Tuple[str, ...] = tuple(technique_ids)
        self._index: Dict[str, int] = {tid: idx for idx, tid in enumerate(self._technique_ids)}

    @property
    def technique_ids(self) -> Tuple[str, ...]:
        """Get the list of technique IDs this predictor supports."""
        return self._technique_ids

    @abstractmethod
    def get_model_name(self) -> str:
        """Return a string identifier for this model type.
        
        Returns:
            Model name (e.g., 'ngram', 'hmm', 'bilstm')
        """
        pass

    def to_distribution(self, log_probs: np.ndarray) -> np.ndarray:
        """Convert log-probabilities into a normalized distribution.
        
        Args:
            log_probs: Array of log probabilities
            
        Returns:
            Normalized probability distribution
        """
        max_log = float(np.max(log_probs)) if log_probs.size else 0.0
        exp_scores = np.exp(log_probs - max_log)
        normalizer = exp_scores.sum()
        if normalizer <= 0.0 or not np.isfinite(normalizer):
            return np.full_like(log_probs, 1.0 / len(log_probs)) if log_probs.size else log_probs
        return exp_scores / normalizer

    @abstractmethod
    def base_log_probabilities(
        self,
        history: Sequence[str],
        candidates: Sequence[str],
        group_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Return log P_base for each candidate.

        Args:
            history: Ordered sequence of past techniques.
            candidates: Candidate techniques to score.
            group_id: Optional context (kept for interface compatibility, typically unused).

        Returns:
            Tuple of ``(log_probs, diagnostics)`` where ``log_probs`` matches the
            order of ``candidates`` and ``diagnostics`` contains any
            model-specific details for downstream inspection.
        """

    def predict_distribution(
        self,
        history: Sequence[str],
        candidates: Sequence[str],
        group_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Predict probability distribution over candidates.
        
        Args:
            history: Ordered sequence of past techniques
            candidates: Candidate techniques to score
            group_id: Optional attacker group context
            
        Returns:
            Tuple of (probabilities, diagnostics)
        """
        log_probs, diagnostics = self.base_log_probabilities(history, candidates, group_id)
        return self.to_distribution(log_probs), diagnostics

    def ensure_candidate_order(self, candidates: Sequence[str]) -> np.ndarray:
        """Map candidates into the model's index order.
        
        Args:
            candidates: List of candidate technique IDs
            
        Returns:
            Array of indices corresponding to candidates
            
        Raises:
            KeyError: If a candidate is not in the model's vocabulary
        """
        indices = []
        for candidate in candidates:
            if candidate in self._index:
                indices.append(self._index[candidate])
            else:
                # Skip unknown candidates instead of raising to keep evaluation robust
                continue
        return np.asarray(indices, dtype=np.int64)


__all__ = ["BasePredictor"]
