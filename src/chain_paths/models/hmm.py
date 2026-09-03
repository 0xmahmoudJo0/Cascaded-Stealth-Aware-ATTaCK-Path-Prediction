"""Simple multinomial HMM predictor built on discrete transition counts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import BasePredictor


@dataclass
class HMMConfig:
    n_states: int = 4
    smoothing: float = 1.0


class HMMPredictor(BasePredictor):
    """Lightweight hidden Markov model over ATT&CK techniques.

    The implementation intentionally stays self-contained (no hmmlearn dependency)
    and treats the emission alphabet as the ATT&CK technique IDs.  Hidden states
    are trained via a simple count-based Baum-Welch approximation that reduces to
    a regularised bigram model, which keeps the behaviour intuitive while still
    supporting probabilistic smoothing.
    """

    def __init__(
        self,
        technique_ids: Sequence[str],
        transition_log_probs: np.ndarray,
        initial_log_probs: np.ndarray,
        config: HMMConfig | None = None,
    ) -> None:
        super().__init__(technique_ids)
        self.config = config or HMMConfig()
        self.transition_log_probs = transition_log_probs
        self.initial_log_probs = initial_log_probs

    def get_model_name(self) -> str:
        """Return model identifier."""
        return 'hmm'

    # ------------------------------------------------------------------
    @classmethod
    def from_sequences(
        cls,
        sequences: Sequence[Sequence[str]],
        technique_ids: Sequence[str],
        *,
        config: HMMConfig | None = None,
    ) -> "HMMPredictor":
        cfg = config or HMMConfig()
        index = {tid: i for i, tid in enumerate(technique_ids)}
        n = len(index)
        smoothing = cfg.smoothing
        transitions = np.full((n, n), smoothing, dtype=np.float64)
        initials = np.full(n, smoothing, dtype=np.float64)

        for seq in sequences:
            if not seq:
                continue
            first = seq[0]
            if first in index:
                initials[index[first]] += 1.0
            for prev, nxt in zip(seq, seq[1:]):
                if prev in index and nxt in index:
                    transitions[index[prev], index[nxt]] += 1.0

        transition_log_probs = np.log(transitions / transitions.sum(axis=1, keepdims=True))
        initial_log_probs = np.log(initials / initials.sum())
        return cls(technique_ids, transition_log_probs, initial_log_probs, cfg)

    # ------------------------------------------------------------------
    def base_log_probabilities(
        self,
        history: Sequence[str],
        candidates: Sequence[str],
        group_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        if not candidates:
            return np.asarray([], dtype=np.float64), []
        if history:
            last = history[-1]
            try:
                row = self.transition_log_probs[self._index[last]]
            except KeyError:
                row = np.full(len(self.technique_ids), -math.log(len(self.technique_ids)))
        else:
            row = self.initial_log_probs
        indices = self.ensure_candidate_order(candidates)
        log_probs = row[indices]
        diagnostics: List[Dict[str, float]] = [
            {'p_count': float(math.exp(lp)), 'source': 'hmm'} for lp in log_probs
        ]
        return log_probs.astype(np.float64), diagnostics


__all__ = ["HMMPredictor", "HMMConfig"]
