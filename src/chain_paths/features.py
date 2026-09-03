"""Feature extraction utilities shared across base predictors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class FeatureConfig:
    epsilon: float = 1e-12
    rho: float = 1.0
    gamma: float = 1.0


class FeatureExtractor:
    """Constructs fusion features for each candidate technique."""

    def __init__(
        self,
        sigma_scores: Dict[str, float],
        ps_scores: Dict[str, float],
        *,
        config: FeatureConfig | None = None,
    ) -> None:
        self.sigma_scores = sigma_scores
        self.ps_scores = ps_scores
        self.config = config or FeatureConfig()

    # ------------------------------------------------------------------
    def _stealth_multiplier(self, technique: str) -> float:
        return float(self.sigma_scores.get(technique, 1.0))

    def _priority_score(self, technique: str) -> float:
        return float(self.ps_scores.get(technique, 1.0))

    # ------------------------------------------------------------------
    def build_features(
        self,
        history: Sequence[str],
        candidates: Sequence[str],
        base_log_probs: np.ndarray,
        base_diagnostics: List[Dict[str, float]],
        *,
        embedding_probs: Optional[Dict[str, float]] = None,
        mapped_similarities: Optional[Dict[str, float]] = None,
        priors: Optional[Dict[str, float]] = None,
        support: float | None = None,
        omega: float | None = None,
    ) -> List[Dict[str, float]]:
        """Combine base diagnostics with auxiliary features."""

        epsilon = self.config.epsilon
        rho = self.config.rho
        gamma = self.config.gamma

        results: List[Dict[str, float]] = []
        embedding_probs = embedding_probs or {}
        mapped_similarities = mapped_similarities or {}
        priors = priors or {}

        for idx, candidate in enumerate(candidates):
            diag = base_diagnostics[idx] if idx < len(base_diagnostics) else {}
            p_count = float(diag.get('p_count', math.exp(base_log_probs[idx])))
            p_emb = float(embedding_probs.get(candidate, 0.0))
            mapped = float(mapped_similarities.get(candidate, 0.5))
            p_prior = float(priors.get(candidate, diag.get('p_prior', 1.0)))
            stealth = self._stealth_multiplier(candidate)
            p_interp = float(diag.get('p_interp', p_count))
            feat = {
                'candidate': candidate,
                'p_count': p_count,
                'p_emb': p_emb,
                'p_interp': p_interp,
                'p_prior': p_prior,
                'stealth': stealth,
                'support': float(support or diag.get('support', 0.0)),
                'omega': float(omega or diag.get('omega', 1.0)),
                'm_emb': mapped ** rho,
                's_stealth': stealth ** gamma,
                'log_p_count': math.log(max(p_count, epsilon)),
                'log_p_emb': math.log(max(p_emb, epsilon)),
                'log_p_interp': math.log(max(p_interp, epsilon)),
                'log_p_prior': math.log(max(p_prior, epsilon)),
                'log_stealth': math.log(max(stealth, epsilon)),
            }
            results.append(feat)
        return results


__all__ = ["FeatureExtractor", "FeatureConfig"]
