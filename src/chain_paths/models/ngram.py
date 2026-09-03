"""N-gram predictor that mirrors the legacy count-based model."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..counts import (
    get_count_for_context,
    get_total_context_support,
    get_top_candidates_for_context,
)
from ..embeddings import get_top_similar_techniques
from .base import BasePredictor


class NGramPredictor(BasePredictor):
    """Maximum-likelihood n-gram model with Dirichlet priors."""

    def __init__(
        self,
        counts_data: Dict[int, pd.DataFrame],
        ps_scores: Dict[str, float],
        sigma_scores: Dict[str, float],
        embeddings: np.ndarray,
        tech_to_idx: Dict[str, int],
        index,
        params: Dict[str, float],
    ) -> None:
        super().__init__(tech_to_idx.keys())
        self.counts_data = counts_data
        self.ps_scores = ps_scores
        self.sigma_scores = sigma_scores
        self.embeddings = embeddings
        self.tech_to_idx = tech_to_idx
        self.index = index
        self.params = dict(params)
        self._epsilon = self.params.get('epsilon', 1e-12)

    def get_model_name(self) -> str:
        """Return model identifier."""
        return 'ngram'

    # --- Priors -----------------------------------------------------------------
    def _global_prior(self, technique: str) -> float:
        unigram_df = self.counts_data.get(1, pd.DataFrame())
        if unigram_df.empty:
            return 1.0 / max(len(self.technique_ids), 1)
        tech_count = unigram_df.loc[unigram_df['candidate'] == technique, 'count']
        total = float(unigram_df['count'].sum())
        if tech_count.empty or total == 0:
            return 0.0
        return float(tech_count.iloc[0] / total)

    # --- Counts -----------------------------------------------------------------
    def _context_order(self, history: Sequence[str]) -> Tuple[int, List[str]]:
        k = min(len(history), self.params.get('k_history', 3))
        context = list(history[-k:]) if k > 0 else []
        return k, context

    def _count_probability(
        self, history: Sequence[str], candidate: str, group_id: Optional[str]
    ) -> Tuple[float, Dict[str, float]]:
        k, context = self._context_order(history)
        if k == 0:
            count = get_count_for_context(candidate, [], self.counts_data[1])
        elif k == 1:
            count = get_count_for_context(candidate, context, self.counts_data[2])
        else:
            lookup_ctx = context if k == 2 else context[-2:]
            count = get_count_for_context(candidate, lookup_ctx, self.counts_data[3])

        lambda_param = self.params['lambda']
        ps_last = self.ps_scores.get(context[-1] if context else '', 1.0)
        p_global = self._global_prior(candidate)
        alpha = lambda_param * ps_last * p_global

        if k == 0:
            total_support = float(self.counts_data[1]['count'].sum())
        elif k == 1:
            total_support = float(get_total_context_support(context, self.counts_data[2]))
        else:
            total_support = float(get_total_context_support(context[-2:], self.counts_data[3]))

        numerator = count + alpha
        denominator = total_support + lambda_param * ps_last
        prob = float(numerator / denominator) if denominator > 0 else 0.0
        # Group-level prior is not currently modeled; default to 1.0 for diagnostics.
        p_group = 1.0
        return prob, {
            'count': float(count),
            'alpha': float(alpha),
            'total_support': float(total_support),
            'ps_last': float(ps_last),
            'p_group': float(p_group),
            'p_global': float(p_global),
        }

    # --- Embeddings --------------------------------------------------------------
    def _embedding_distribution(
        self, history: Sequence[str], candidates: Sequence[str]
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        k, context = self._context_order(history)
        if not context or not candidates:
            zero_components = {
                candidate: {'similarity': 0.0, 'mapped_similarity': 0.5, 'tau': self.params['tau'], 'scaled': 0.0}
                for candidate in candidates
            }
            return {candidate: 0.0 for candidate in candidates}, zero_components

        context_indices = [self.tech_to_idx[tech] for tech in context if tech in self.tech_to_idx]
        if not context_indices:
            zero_components = {
                candidate: {'similarity': 0.0, 'mapped_similarity': 0.5, 'tau': self.params['tau'], 'scaled': 0.0}
                for candidate in candidates
            }
            return {candidate: 0.0 for candidate in candidates}, zero_components

        context_vec = np.mean(self.embeddings[context_indices], axis=0)
        context_norm = np.linalg.norm(context_vec)
        if context_norm == 0:
            zero_components = {
                candidate: {'similarity': 0.0, 'mapped_similarity': 0.5, 'tau': self.params['tau'], 'scaled': 0.0}
                for candidate in candidates
            }
            return {candidate: 0.0 for candidate in candidates}, zero_components

        tau = self.params['tau']
        similarities: List[float] = []
        components: Dict[str, Dict[str, float]] = {}
        for candidate in candidates:
            if candidate in self.tech_to_idx:
                cand_vec = self.embeddings[self.tech_to_idx[candidate]]
                cand_norm = np.linalg.norm(cand_vec)
                similarity = float(np.dot(context_vec, cand_vec) / (context_norm * cand_norm)) if cand_norm else 0.0
            else:
                similarity = 0.0
            similarity = float(np.clip(similarity, -1.0, 1.0))
            mapped = (1.0 + similarity) / 2.0
            scaled = tau * mapped
            similarities.append(scaled)
            components[candidate] = {
                'similarity': similarity,
                'mapped_similarity': mapped,
                'tau': tau,
                'scaled': scaled,
            }

        sim_array = np.asarray(similarities, dtype=np.float64)
        max_score = float(sim_array.max())
        exp_scores = np.exp(sim_array - max_score)
        norm = float(exp_scores.sum())
        if norm <= 0.0 or not np.isfinite(norm):
            probs = {candidate: 1.0 / len(candidates) for candidate in candidates}
            for candidate in candidates:
                components[candidate]['normalizer'] = float(len(candidates))
            return probs, components

        probs = {
            candidate: float(exp_scores[idx] / norm)
            for idx, candidate in enumerate(candidates)
        }
        for candidate in candidates:
            components[candidate]['normalizer'] = norm
        return probs, components

    # --- Candidate generation ----------------------------------------------------
    def candidate_set(self, history: Sequence[str], top_n: int) -> List[str]:
        candidates = set()
        k, context = self._context_order(history)
        if k == 0:
            top_counts = self.counts_data[1].nlargest(50, 'count')['candidate'].tolist()
        elif k == 1:
            top_counts = get_top_candidates_for_context(context, self.counts_data[2], 50)
        else:
            lookup_ctx = context if k == 2 else context[-2:]
            top_counts = get_top_candidates_for_context(lookup_ctx, self.counts_data[3], 50)
        candidates.update(top_counts)

        if context and self.index is not None:
            candidates.update(
                get_top_similar_techniques(self.embeddings, self.tech_to_idx, self.index, context, 50)
            )

        if len(candidates) < top_n:
            unigram_df = self.counts_data.get(1, pd.DataFrame())
            if not unigram_df.empty:
                ordered = unigram_df.sort_values('count', ascending=False)['candidate'].tolist()
            else:
                ordered = list(sorted(self.tech_to_idx.keys()))
            for tech in ordered:
                if tech not in candidates:
                    candidates.add(tech)
                if len(candidates) >= top_n:
                    break
            if len(candidates) < top_n:
                for tech in sorted(self.tech_to_idx.keys()):
                    if tech not in candidates:
                        candidates.add(tech)
                    if len(candidates) >= top_n:
                        break
        return list(candidates)[:top_n]

    # --- Base interface ---------------------------------------------------------
    def base_log_probabilities(
        self,
        history: Sequence[str],
        candidates: Sequence[str],
        group_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        log_probs: List[float] = []
        diagnostics: List[Dict[str, float]] = []
        for candidate in candidates:
            p_count, count_components = self._count_probability(history, candidate, group_id)
            epsilon = self._epsilon
            log_probs.append(math.log(max(p_count, epsilon)))
            diag = dict(count_components)
            diag['p_count'] = p_count
            diagnostics.append(diag)
        return np.asarray(log_probs, dtype=np.float64), diagnostics

    def enriched_features(
        self,
        history: Sequence[str],
        group_id: Optional[str] = None,
        top_n_candidates: int = 200,
    ) -> List[Tuple[str, Dict[str, float]]]:
        candidates = self.candidate_set(history, top_n_candidates)
        if not candidates:
            return []
        embedding_probs, embedding_components = self._embedding_distribution(history, candidates)
        k, context = self._context_order(history)
        if k == 0:
            support = float(self.counts_data[1]['count'].sum())
        elif k == 1:
            support = float(get_total_context_support(context, self.counts_data[2]))
        else:
            support = float(get_total_context_support(context[-2:], self.counts_data[3]))
        kappa = self.params['kappa']
        omega = support / (support + kappa) if (support + kappa) > 0 else 0.0
        rho = self.params['rho']
        gamma = self.params['gamma']
        epsilon = self._epsilon
        results: List[Tuple[str, Dict[str, float]]] = []
        for candidate in candidates:
            p_count, count_components = self._count_probability(history, candidate, group_id)
            p_emb = embedding_probs.get(candidate, 0.0)
            emb_comp = embedding_components.get(candidate, {
                'similarity': 0.0,
                'mapped_similarity': 0.5,
                'tau': self.params['tau'],
                'scaled': 0.0,
                'normalizer': 1.0,
            })
            mapped_similarity = emb_comp.get('mapped_similarity', 0.5)
            stealth_score = self.sigma_scores.get(candidate, 1.0)
            p_prior = self.params['eta'] * count_components['p_group'] + (1 - self.params['eta']) * count_components['p_global']
            p_interp = omega * p_count + (1 - omega) * p_emb
            features = {
                'candidate': candidate,
                'p_count': p_count,
                'p_emb': p_emb,
                'p_interp': p_interp,
                'p_prior': p_prior,
                'stealth': stealth_score,
                'm_emb': mapped_similarity ** rho,
                's_stealth': stealth_score ** gamma,
                'omega': omega,
                'support': support,
                'log_p_count': math.log(max(p_count, epsilon)),
                'log_p_emb': math.log(max(p_emb, epsilon)),
                'log_p_interp': math.log(max(p_interp, epsilon)),
                'log_p_prior': math.log(max(p_prior, epsilon)),
                'log_stealth': math.log(max(stealth_score, epsilon)),
                'count_components': count_components,
                'emb_components': emb_comp,
            }
            results.append((candidate, features))
        return results


__all__ = ["NGramPredictor"]
