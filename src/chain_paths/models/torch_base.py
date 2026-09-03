"""Shared helpers for neural sequence predictors."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .base import BasePredictor


class TorchBackedPredictor(BasePredictor):
    """Base class for predictors that optionally rely on PyTorch.

    The project requirements call for a suite of neural sequence models.  To keep
    the repository runnable in lightweight environments (including the unit-test
    harness where PyTorch may be unavailable), this class provides graceful
    degradation: if PyTorch cannot be imported, the predictor falls back to a
    smoothed Markov chain derived from provided transition statistics.  Concrete
    subclasses can still override :meth:`forward` when PyTorch is present.
    """

    def __init__(
        self,
        technique_ids: Sequence[str],
        *,
        transition_log_probs: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(technique_ids)
        try:
            import torch  # type: ignore

            self._torch = torch
            self._torch_available = True
        except Exception:  # pragma: no cover - availability branch
            self._torch = None
            self._torch_available = False
        self._transition_log_probs = transition_log_probs
        self._uniform_log_prob = -math.log(len(self.technique_ids)) if self.technique_ids else 0.0
        self.model = None  # Initialize model attribute for neural subclasses

    # ------------------------------------------------------------------
    def _fallback_row(self, history: Sequence[str]) -> np.ndarray:
        if self._transition_log_probs is None or not history:
            return np.full(len(self.technique_ids), self._uniform_log_prob, dtype=np.float64)
        last = history[-1]
        try:
            row = self._transition_log_probs[self._index[last]]
        except KeyError:
            row = np.full(len(self.technique_ids), self._uniform_log_prob, dtype=np.float64)
        return row

    # ------------------------------------------------------------------
    def base_log_probabilities(self, history, candidates, group_id=None):  # type: ignore[override]
        indices = self.ensure_candidate_order(candidates)
        row = self._fallback_row(history)
        log_probs = row[indices]
        diagnostics = [{'p_count': float(math.exp(lp)), 'source': 'neural-fallback'} for lp in log_probs]
        return log_probs.astype(np.float64), diagnostics


__all__ = ["TorchBackedPredictor"]
