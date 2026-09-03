"""Chain-aware predictor implementation.

This module implements a streamlined predictor that combines:
  - LSTM: Neural sequence model (primary predictor)
  - Tactic Transitions: ATT&CK kill-chain structure
  - Stealth Scores: Detection difficulty from Sigma rules

The architecture removes redundant components (count-based n-grams, embedding
similarity, global priors, causal reasoning) that are subsumed by the LSTM
or lack sufficient annotation data.

The predictor maintains an explicit log-linear probability model whose weights
can be fit from data, enabling statistical interpretability and rigorous evaluation.
"""

import math
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Sequence, Mapping, Set

import numpy as np
import pandas as pd

from . import config as cfg
from .io import load_parquet, load_csv, load_numpy, load_pickle
from .embeddings import load_embeddings_and_index
from .models.base import BasePredictor
from .models.checkpoint import load_torch_model
from .tactics import load_tactic_mapping, load_tactic_transitions
from .platforms import load_platform_mapping
from .io import load_sequences_for_source
from .models.factory import ModelFactory

# Legacy counts imports (retained for backward-compatible helper methods)
try:
    from .counts import get_count_for_context, get_total_context_support
    _HAS_COUNTS = True
except ImportError:
    _HAS_COUNTS = False

try:
    from .embeddings import get_top_similar_techniques
    _HAS_EMB_SEARCH = True
except ImportError:
    _HAS_EMB_SEARCH = False


# Three-pillar weight order: LSTM + Tactic + Stealth
WEIGHT_ORDER: Tuple[str, ...] = (
    'bias',
    'log_bilstm',
    'log_tactic',
    'log_stealth',
)


def _normalize_tid(value: Any) -> str:
    return str(value).strip().upper()


def _normalize_tactic_name(value: Any) -> str:
    return str(value).strip().lower()


class Predictor:
    """
    Streamlined chain-aware predictor for attack technique sequences.
    
    Combines LSTM neural predictions with tactic transitions and stealth scores
    to predict next techniques in attack chains.
    
    Architecture (3 orthogonal signals):
    - LSTM: Learns sequential patterns from training data
    - Tactic Transitions: Enforces ATT&CK kill-chain logic
    - Stealth: Detection difficulty from Sigma rules (defender's perspective)
    """
    
    def __init__(
        self,
        counts_data: Dict[int, pd.DataFrame],
        ps_scores: Dict[str, float],
        sigma_scores: Dict[str, float],
        embeddings: np.ndarray,
        tech_to_idx: Dict[str, int],
        index: Any,
        params: Optional[Dict[str, float]] = None,
        base_model: Optional[BasePredictor] = None,
        tactic_mapping: Optional[Mapping[str, Sequence[str]]] = None,
        tactic_transitions: Optional[Mapping[str, Mapping[str, float]]] = None,
        platform_mapping: Optional[Mapping[str, Sequence[str]]] = None,
    ):
        """
        Initialize predictor.
        
        Args:
            counts_data: Dictionary mapping n-gram size to counts DataFrame (legacy, kept for compatibility)
            ps_scores: Dictionary mapping technique_id to priority score
            sigma_scores: Dictionary mapping technique_id to stealth score
            embeddings: Technique embeddings array (used by BiLSTM internally)
            tech_to_idx: Technique to index mapping
            index: ANN index for similarity search (legacy, kept for compatibility)
            params: Hyperparameters dictionary
            base_model: Base model for probability computation (BiLSTM recommended)
            tactic_mapping: Mapping from technique IDs to ATT&CK tactics.
            tactic_transitions: Transition probabilities between tactics.
            platform_mapping: Mapping from technique IDs to platforms.
        """
        self.counts_data = counts_data
        self.ps_scores = ps_scores
        self.sigma_scores = sigma_scores
        self.embeddings = embeddings
        self.tech_to_idx = tech_to_idx
        self.index = index
        self.base_model = base_model

        # Set parameters
        self.params = deepcopy(cfg.DEFAULT_PARAMS)
        if params:
            for key, value in params.items():
                if key == 'component_weights':
                    merged = deepcopy(self.params.get('component_weights', {}))
                    merged.update(value)
                    self.params['component_weights'] = merged
                else:
                    self.params[key] = value

        self._epsilon = self.params.get('epsilon', 1e-12)
        self.params.setdefault('use_scenario_priors', True)
        
        # Initialize verbose early
        self.verbose = bool(self.params.get('verbose', False))

        # Create reverse mapping
        self.idx_to_tech = {idx: tech for tech, idx in tech_to_idx.items()}

        self._start_tactic_key = "__start__"
        self._known_tactics: Set[str] = set()
        self.technique_to_tactics: Dict[str, List[str]] = {}
        self.tactic_transition_probs: Dict[str, Dict[str, float]] = {}
        self._has_tactic_model = False
        self._initialize_tactic_model(tactic_mapping, tactic_transitions)

        self.technique_to_platforms: Dict[str, List[str]] = {}
        self._allowed_platforms = {
            _normalize_tactic_name(p) for p in self.params.get('allowed_platforms', []) if str(p).strip()
        }
        self._use_platform_filtering = bool(self.params.get('use_platform_filtering', True))
        self._initialize_platform_model(platform_mapping)

        model_name = base_model.get_model_name() if base_model else 'ngram'
        print(f"Initialized predictor with {len(tech_to_idx)} techniques (base model: {model_name})")
        if self.verbose:
            print(f"Parameters: {self.params}")

    def _initialize_tactic_model(
        self,
        tactic_mapping: Optional[Mapping[str, Sequence[str]]],
        tactic_transitions: Optional[Mapping[str, Mapping[str, float]]],
    ) -> None:
        """
        Initialize tactic transition model from loaded data.
        
        Populates:
        - technique_to_tactics: Maps technique IDs to their tactics
        - tactic_transition_probs: P(next_tactic | current_tactic)
        - _known_tactics: Set of all known tactics
        - _has_tactic_model: True if data available
        
        Args:
            tactic_mapping: Mapping from technique IDs to list of tactics
            tactic_transitions: Mapping from (current_tactic -> dict of next_tactic -> prob)
        """
        if not tactic_mapping:
            self._has_tactic_model = False
            return
        
        # Build technique_to_tactics mapping
        for technique_id, tactics in tactic_mapping.items():
            normalized_tid = _normalize_tid(technique_id)
            normalized_tactics = [_normalize_tactic_name(t) for t in tactics]
            self.technique_to_tactics[normalized_tid] = normalized_tactics
            self._known_tactics.update(normalized_tactics)
        
        # Build tactic transition probabilities
        if tactic_transitions:
            for current_tactic, next_tactics in tactic_transitions.items():
                normalized_current = _normalize_tactic_name(current_tactic)
                normalized_next = {
                    _normalize_tactic_name(tactic): prob
                    for tactic, prob in next_tactics.items()
                }
                self.tactic_transition_probs[normalized_current] = normalized_next
                self._known_tactics.add(normalized_current)
                self._known_tactics.update(normalized_next.keys())
        
        self._has_tactic_model = bool(self.tactic_transition_probs)
        
        if self._has_tactic_model:
            print(
                f"Initialized tactic model with {len(self.technique_to_tactics)} "
                f"technique mappings and {len(self.tactic_transition_probs)} "
                f"tactic transitions across {len(self._known_tactics)} tactics"
            )

    def _initialize_platform_model(
        self,
        platform_mapping: Optional[Mapping[str, Sequence[str]]],
    ) -> None:
        """Populate technique->platform lookup."""

        if not platform_mapping:
            return

        for technique_id, platforms in platform_mapping.items():
            normalized_tid = _normalize_tid(technique_id)
            normalized_platforms = [
                _normalize_tactic_name(p) for p in platforms if str(p).strip()
            ]
            if normalized_platforms:
                self.technique_to_platforms[normalized_tid] = normalized_platforms

    def _extract_context_fields(
        self, group_context: Optional[Any]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract APT group, scenario phase, and persistence focus from group context.
        
        Args:
            group_context: Optional context object with attributes:
                - apt_group or threat_actor or group
                - scenario_phase or phase
                - persistence_focus or persistence
        
        Returns:
            Tuple of (apt_group, scenario_phase, persistence_focus), all Optional[str]
        """
        if not group_context:
            return None, None, None
        
        # Try multiple attribute names for compatibility
        apt = None
        for attr in ['apt_group', 'threat_actor', 'group', 'actor']:
            if hasattr(group_context, attr):
                apt = getattr(group_context, attr)
                break
        
        phase = None
        for attr in ['scenario_phase', 'phase', 'tactic_phase']:
            if hasattr(group_context, attr):
                phase = getattr(group_context, attr)
                break
        
        persistence = None
        for attr in ['persistence_focus', 'persistence', 'focus']:
            if hasattr(group_context, attr):
                persistence = getattr(group_context, attr)
                break
        
        return apt, phase, persistence

    def _get_tactic_context(
        self, history: Sequence[str]
    ) -> Tuple[str, Dict[str, float]]:
        """
        Extract current tactic from history and get P(next_tactic | current).
        
        Args:
            history: Attack chain so far
        
        Returns:
            Tuple of (current_tactic, P_next_tactic_dict)
            where P_next_tactic_dict maps tactic names to probabilities
        """
        if not self._has_tactic_model:
            # No tactic model available
            uniform_dist = {tactic: 1.0 / max(len(self._known_tactics), 1) 
                           for tactic in self._known_tactics}
            return self._start_tactic_key, uniform_dist
        
        # Get current tactic from last technique in history
        if not history:
            current_tactic = self._start_tactic_key
        else:
            last_technique = _normalize_tid(history[-1])
            tactics_for_last = self.technique_to_tactics.get(last_technique, [])
            # Use primary tactic (first in list)
            current_tactic = tactics_for_last[0] if tactics_for_last else self._start_tactic_key
        
        # Get transition probabilities for current tactic
        next_tactic_probs = self.tactic_transition_probs.get(current_tactic, {})
        
        if not next_tactic_probs:
            # Uniform distribution if no transitions observed
            uniform_dist = {tactic: 1.0 / max(len(self._known_tactics), 1) 
                           for tactic in self._known_tactics}
            return current_tactic, uniform_dist
        
        return current_tactic, next_tactic_probs

    def _compute_tactic_transition_prob(
        self, history: Sequence[str], candidate: str
    ) -> Optional[float]:
        """
        Compute P(tactic_next | tactic_current) for a candidate technique.
        
        Args:
            history: Attack chain so far
            candidate: Candidate technique to evaluate
        
        Returns:
            Probability that candidate's tactic follows current tactic, or None if no model
        """
        if not self._has_tactic_model:
            return None
        
        current_tactic, next_tactic_probs = self._get_tactic_context(history)
        
        # Get candidate's tactics
        normalized_candidate = _normalize_tid(candidate)
        candidate_tactics = self.technique_to_tactics.get(normalized_candidate, [])
        
        if not candidate_tactics:
            # Unknown technique, return low probability
            return 0.001
        
        # Use primary tactic (first in list)
        candidate_tactic = candidate_tactics[0]
        
        # Return probability for this tactic transition
        prob = next_tactic_probs.get(candidate_tactic, 0.001)
        
        return prob

    def _global_prior(self, technique: str) -> float:
        df = self.counts_data.get(1)
        if df is None or df.empty:
            total = max(len(self.tech_to_idx), 1)
            return 1.0 / total
        row = df.loc[df['candidate'] == technique, 'count']
        total = float(df['count'].sum())
        if row.empty or total == 0.0:
            return 0.0
        return float(row.iloc[0] / total)

    def _count_probability_generic(
        self,
        history: Sequence[str],
        candidate: str,
        use_scenario_priors: Optional[bool] = None,
    ) -> Tuple[float, Dict[str, float]]:
        if not self.counts_data:
            total = max(len(self.tech_to_idx), 1)
            prob = 1.0 / total
            return prob, {'p_global': prob}
        k = min(len(history), self.params['k_history'])
        history_context = list(history[-k:]) if k > 0 else []
        lambda_param = self.params.get('lambda', 5.0)

        if k == 0:
            unigram_df = self.counts_data.get(1, pd.DataFrame())
            count = get_count_for_context(candidate, [], unigram_df)
            total_support = float(unigram_df['count'].sum()) if not unigram_df.empty else 0.0
            last = ''
        elif k == 1:
            bigram_df = self.counts_data.get(2, pd.DataFrame())
            count = get_count_for_context(candidate, history_context, bigram_df)
            total_support = get_total_context_support(history_context, bigram_df)
            last = history_context[-1] if history_context else ''
        else:
            trigram_df = self.counts_data.get(3, pd.DataFrame())
            trigram_context = history_context[-2:] if len(history_context) >= 2 else history_context
            count = get_count_for_context(candidate, trigram_context, trigram_df)
            total_support = get_total_context_support(trigram_context, trigram_df)
            last = history_context[-1] if history_context else ''

        # Get group-aware Dirichlet prior (optionally disable scenario priors)
        lambda_param = self.params['lambda']
        # When use_scenario_priors=False (legacy mode), ignore MCDM scores entirely
        if use_scenario_priors is False:
            ps_last = 1.0  # Baseline: no MCDM influence
        else:
            ps_last = self.ps_scores.get(last, 1.0)  # Scenario-aware: use MCDM scores

        p_global = self._global_prior(candidate)

        # Group-aware Dirichlet prior
        p_prior = p_global
        alpha = lambda_param * ps_last * p_prior

        # Compute probability
        numerator = count + alpha
        denominator = total_support + lambda_param * ps_last

        prob = numerator / denominator if denominator > 0 else 0.0

        variance = 0.0
        if denominator > 1:
            posterior_total = denominator
            posterior_alpha = numerator
            variance = (
                posterior_alpha * (posterior_total - posterior_alpha)
            ) / (posterior_total ** 2 * (posterior_total + 1))

        # Components for debugging
        components = {
            'count': count,
            'alpha': alpha,
            'total_support': total_support,
            'ps_last': ps_last,
            'p_global': p_global,
            'p_prior': p_prior,
            'posterior_total': denominator,
            'posterior_variance': variance,
        }

        return prob, components

    def compute_count_probability(
        self,
        history: Sequence[str],
        candidate: str,
        group_context: Optional[Any] = None,
        use_scenario_priors: Optional[bool] = None,
    ) -> Tuple[float, Dict[str, float]]:
        return self._count_probability_generic(
            history,
            candidate,
            use_scenario_priors=use_scenario_priors,
        )
    
    def compute_embedding_distribution(
        self, history: List[str], candidates: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Compute a normalized embedding-based probability distribution."""

        k = min(len(history), self.params['k_history'])
        history_context = history[-k:] if k > 0 else []

        if not history_context or not candidates:
            zero_components = {candidate: {'similarity': 0.0, 'mapped_similarity': 0.5, 'tau': self.params['tau'], 'scaled': 0.0} for candidate in candidates}
            return {candidate: 0.0 for candidate in candidates}, zero_components

        context_indices = [self.tech_to_idx[tech] for tech in history_context if tech in self.tech_to_idx]
        if not context_indices:
            zero_components = {candidate: {'similarity': 0.0, 'mapped_similarity': 0.5, 'tau': self.params['tau'], 'scaled': 0.0} for candidate in candidates}
            return {candidate: 0.0 for candidate in candidates}, zero_components

        context_vector = np.mean(self.embeddings[context_indices], axis=0)
        context_norm = np.linalg.norm(context_vector)
        if context_norm == 0:
            zero_components = {candidate: {'similarity': 0.0, 'mapped_similarity': 0.5, 'tau': self.params['tau'], 'scaled': 0.0} for candidate in candidates}
            return {candidate: 0.0 for candidate in candidates}, zero_components

        tau = self.params['tau']
        similarities = []
        components: Dict[str, Dict[str, float]] = {}

        for candidate in candidates:
            if candidate in self.tech_to_idx:
                candidate_vec = self.embeddings[self.tech_to_idx[candidate]]
                candidate_norm = np.linalg.norm(candidate_vec)
                if candidate_norm == 0:
                    similarity = 0.0
                else:
                    similarity = float(np.dot(context_vector, candidate_vec) / (context_norm * candidate_norm))
            else:
                similarity = 0.0

            similarity = float(np.clip(similarity, -1.0, 1.0))
            mapped_similarity = (1.0 + similarity) / 2.0
            scaled = tau * mapped_similarity
            similarities.append(scaled)
            components[candidate] = {
                'similarity': similarity,
                'mapped_similarity': mapped_similarity,
                'tau': tau,
                'scaled': scaled,
            }

        similarities_array = np.array(similarities, dtype=np.float64)
        max_score = similarities_array.max()
        exp_scores = np.exp(similarities_array - max_score)
        normalization = exp_scores.sum()

        if normalization == 0 or not np.isfinite(normalization):
            probabilities = {candidate: 1.0 / len(candidates) for candidate in candidates}
            for candidate in candidates:
                components[candidate]['normalizer'] = float(len(candidates))
            return probabilities, components

        probabilities = {
            candidate: float(exp_scores[idx] / normalization)
            for idx, candidate in enumerate(candidates)
        }

        for candidate in candidates:
            components[candidate]['normalizer'] = float(normalization)

        return probabilities, components
    
    def get_candidate_set(
        self,
        history: List[str],
        top_n: int = 200,
        use_tactic_filtering: Optional[bool] = None,
        tactic_threshold: Optional[float] = None,
        use_platform_filtering: Optional[bool] = None,
        allowed_platforms: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """
        Get candidate set for prediction with optional tactic-aware filtering.
        
        Args:
            history: History of techniques
            top_n: Number of candidates to return
            use_tactic_filtering: Whether to filter by tactic (default from config)
            tactic_threshold: Minimum tactic probability (default from config)
            
        Returns:
            List of candidate techniques
        """
        # Use config defaults if not specified
        if use_tactic_filtering is None:
            use_tactic_filtering = self.params.get('use_tactic_filtering', True)
        if tactic_threshold is None:
            tactic_threshold = self.params.get('tactic_threshold', 0.01)
        if use_platform_filtering is None:
            use_platform_filtering = self._use_platform_filtering

        allowed_platforms_set: Set[str] = set(self._allowed_platforms)
        if allowed_platforms:
            allowed_platforms_set = {
                _normalize_tactic_name(p) for p in allowed_platforms if str(p).strip()
            }
        
        # Get tactic context for filtering
        current_tactic, next_tactic_probs = self._get_tactic_context(history)
        
        # Filter to likely next tactics if filtering enabled
        likely_tactics = set()
        if use_tactic_filtering and self._has_tactic_model:
            likely_tactics = {
                tactic for tactic, prob in next_tactic_probs.items()
                if prob >= tactic_threshold
            }
            
            if likely_tactics and self.verbose:
                # Debug logging
                print(f"[Tactic Filtering] Current: {current_tactic}")
                print(f"[Tactic Filtering] Likely next tactics: {likely_tactics}")
        
        candidates = set()

        def _filter_platform(candidate_list: List[str]) -> List[str]:
            if use_platform_filtering and allowed_platforms_set:
                return [
                    tech for tech in candidate_list
                    if self._is_technique_in_platforms(tech, allowed_platforms_set)
                ]
            return candidate_list
        
        # Primary: Get candidates from BiLSTM vocabulary (all known techniques)
        # The BiLSTM will score them, so we just need a good candidate pool
        if self.base_model is not None:
            base_model_index = getattr(self.base_model, "_index", {})
            all_techs = list(base_model_index.keys())
        else:
            all_techs = list(self.tech_to_idx.keys())
        
        # Filter by tactic if enabled
        if use_tactic_filtering and likely_tactics:
            filtered_techs = [
                tech for tech in all_techs
                if self._is_technique_in_tactics(tech, likely_tactics)
            ]
            filtered_techs = _filter_platform(filtered_techs)
            candidates.update(filtered_techs)
        else:
            candidates.update(_filter_platform(all_techs))
        
        # Backfill from vocabulary if tactic filtering yielded too few candidates
        if len(candidates) < top_n:
            for tech in sorted(self.tech_to_idx.keys()):
                if tech not in candidates:
                    if use_tactic_filtering and likely_tactics:
                        if not self._is_technique_in_tactics(tech, likely_tactics):
                            continue
                    if use_platform_filtering and allowed_platforms_set:
                        if not self._is_technique_in_platforms(tech, allowed_platforms_set):
                            continue
                    candidates.add(tech)
                if len(candidates) >= top_n:
                    break

        after_filter = len(candidates)
        
        # Debug logging
        if use_tactic_filtering and likely_tactics and self.verbose:
            print(f"[Tactic Filtering] Candidates before: ~100, after filtering: {after_filter}")
        
        return list(candidates)[:top_n]
    
    def _is_technique_in_tactics(self, technique: str, tactics: Set[str]) -> bool:
        """
        Check if a technique belongs to any of the given tactics.
        
        Args:
            technique: Technique ID to check
            tactics: Set of tactic names to check against
        
        Returns:
            True if technique is in one of the tactics
        """
        normalized_tid = _normalize_tid(technique)
        technique_tactics = self.technique_to_tactics.get(normalized_tid, [])
        return any(tactic in tactics for tactic in technique_tactics)

    def _is_technique_in_platforms(self, technique: str, platforms: Set[str]) -> bool:
        """Check platform compatibility for a technique."""

        normalized_tid = _normalize_tid(technique)
        technique_platforms = self.technique_to_platforms.get(normalized_tid, [])
        if not technique_platforms:
            return True
        return any(platform in platforms for platform in technique_platforms)
    

    def compute_candidate_components(
        self,
        history: List[str],
        group_context: Optional[Any] = None,
        top_n_candidates: int = 200,
        use_scenario_priors: Optional[bool] = None,
        use_tactic_filtering: Optional[bool] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Compute scores for candidate techniques using simplified architecture.
        
        Architecture: LSTM + Tactic Prior + Stealth Score (3 orthogonal signals)
        
        Args:
            history: Attack chain so far
            group_context: Optional context (legacy, currently unused)
            top_n_candidates: Number of candidates to consider
            use_scenario_priors: Whether to use scenario-specific priors
            use_tactic_filtering: Override tactic pre-filtering (None = use config)
            
        Returns:
            List of (technique, probability, components) tuples sorted by probability
        """
        # Log pure model mode once
        if not hasattr(self, '_pure_model_logged'):
            is_pure = (
                self.params.get('w_tactic', 0.1) == 0.0 and
                self.params.get('w_stealth', 0.1) == 0.0
            )
            if is_pure:
                print("[*] PURE MODEL MODE: All fusion weights disabled (academic evaluation)")
            self._pure_model_logged = True

        if use_scenario_priors is None:
            use_scenario_priors = self.params.get('use_scenario_priors', True)
        
        candidates = self.get_candidate_set(
            history, top_n_candidates,
            use_tactic_filtering=use_tactic_filtering,
        )
        if not candidates:
            return []

        candidate_scores = []
        epsilon = self.params.get('epsilon', 1e-9)

        # Get BiLSTM scores for all candidates in one batch call
        base_model_name = (
            self.base_model.get_model_name().lower()
            if self.base_model and hasattr(self.base_model, "get_model_name")
            else "ngram"
        )
        base_log_scores: Dict[str, float] = {}
        if self.base_model is not None:
            base_model_index = getattr(self.base_model, "_index", {})
            known_candidates = [c for c in candidates if c in base_model_index]

            if known_candidates:
                log_probs, _ = self.base_model.base_log_probabilities(
                    history, known_candidates, None
                )
                for idx, candidate in enumerate(known_candidates):
                    lp = float(log_probs[idx]) if idx < len(log_probs) else float("-inf")
                    base_log_scores[candidate] = lp

        # ── Optional per-step signal standardisation (Section 5.4) ──────────
        # The three log-signals occupy very different numerical ranges: the LSTM
        # log-probability spans roughly [log(eps), 0] while the log-stealth term
        # is confined to about [-1.6, 0]. Under raw log-linear fusion no choice
        # of scalar weights can therefore let stealth compete with the sequence
        # model. With fusion_normalization='zscore' (or 'minmax') each signal is
        # standardised across the candidate pool at this step BEFORE weighting,
        # which removes the scale confound and gives the fused baseline a fair
        # opportunity to trade the signals off. Default 'none' reproduces v1.1.
        fusion_norm = self.params.get('fusion_normalization', 'none')

        raw_terms = []
        for candidate in candidates:
            log_base = base_log_scores.get(candidate, math.log(epsilon))
            tactic_prob = self._compute_tactic_transition_prob(history, candidate)
            base_tactic = tactic_prob if tactic_prob is not None else 1.0
            stealth_score = self.sigma_scores.get(candidate, 1.0)
            raw_terms.append((
                candidate,
                log_base,
                math.log(max(base_tactic, epsilon)),
                math.log(max(stealth_score if stealth_score > 0 else epsilon, epsilon)),
                tactic_prob, base_tactic, stealth_score,
            ))

        # Shannon-entropy weighting (Abo-alian, Youssef & Badr, Sci. Rep. 2025).
        # Each signal is normalised across the candidate set into a probability
        # distribution, its Shannon entropy e_j computed, and its weight set from
        # the divergence d_j = 1 - e_j. Signals whose values are more dispersed
        # across candidates - i.e. more discriminative - receive greater weight.
        # This is scale-invariant by construction, because the column
        # normalisation precedes the entropy computation, so it addresses the
        # dynamic-range confound without any manually chosen coefficient.
        entropy_w = None
        if fusion_norm in ('entropy', 'critic', 'rank') and len(raw_terms) > 1:
            m = len(raw_terms)
            raw_cols = [[math.exp(r[pos]) for r in raw_terms] for pos in (1, 2, 3)]

            if fusion_norm == 'rank':
                # Rank-normalise each signal to [0,1]. Every signal is then
                # uniformly dispersed by construction, so a compressed scale
                # (the Sigma stealth score) is not penalised for its incidental
                # range. Weights are equal because no pillar is privileged.
                mats = []
                for v in raw_cols:
                    idx = sorted(range(m), key=lambda i: v[i])
                    rr = [0.0] * m
                    for pos_, i in enumerate(idx):
                        rr[i] = pos_ / max(m - 1, 1)
                    s = sum(rr) or 1.0
                    mats.append([x / s for x in rr])
                entropy_w = [1 / 3, 1 / 3, 1 / 3]
            else:
                mats = []
                for v in raw_cols:
                    s = sum(v)
                    mats.append([x / s for x in v] if s > 0 else [1.0 / m] * m)
                ent = []
                for p in mats:
                    e = -sum(x * math.log(max(x, 1e-12)) for x in p) / math.log(m)
                    ent.append(max(0.0, 1.0 - e))            # divergence d_j
                if fusion_norm == 'entropy':
                    tot = sum(ent)
                    entropy_w = [c / tot for c in ent] if tot > 0 else [1 / 3] * 3
                else:
                    # CRITIC (Diakoulaki et al.): weight = dispersion x conflict,
                    # where conflict is 1 - mean correlation with the other
                    # criteria, so a signal duplicating another is discounted.
                    def _corr(a, b):
                        ma, mb = sum(a) / m, sum(b) / m
                        va = sum((x - ma) ** 2 for x in a) ** .5
                        vb = sum((x - mb) ** 2 for x in b) ** .5
                        if va < 1e-12 or vb < 1e-12:
                            return 0.0
                        return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)
                    crit = []
                    for j in range(3):
                        conflict = sum(1 - _corr(mats[j], mats[k])
                                       for k in range(3) if k != j)
                        crit.append(ent[j] * conflict)
                    tot = sum(crit)
                    entropy_w = [c / tot for c in crit] if tot > 0 else [1 / 3] * 3
            norm_terms = {'base': mats[0], 'tactic': mats[1], 'stealth': mats[2]}
            self._last_entropy_w = entropy_w

        norm_terms = locals().get('norm_terms', {}) if entropy_w else {}
        if fusion_norm in ('zscore', 'minmax') and len(raw_terms) > 1:
            for pos, key in ((1, 'base'), (2, 'tactic'), (3, 'stealth')):
                vals = [r[pos] for r in raw_terms]
                finite = [v for v in vals if v > float('-inf')]
                lo = min(finite) if finite else 0.0
                vals = [v if v > float('-inf') else lo for v in vals]
                if fusion_norm == 'zscore':
                    mu = sum(vals) / len(vals)
                    var = sum((v - mu) ** 2 for v in vals) / len(vals)
                    sd = math.sqrt(var)
                    scaled = [((v - mu) / sd) if sd > 1e-12 else 0.0 for v in vals]
                else:
                    lo_v, hi_v = min(vals), max(vals)
                    rng = hi_v - lo_v
                    scaled = [((v - lo_v) / rng) if rng > 1e-12 else 0.0 for v in vals]
                norm_terms[key] = scaled

        for _i, (candidate, log_base, log_tactic, log_stealth,
                 tactic_prob, base_tactic, stealth_score) in enumerate(raw_terms):
            # 1. BiLSTM score (primary predictor)
            p_base = math.exp(log_base) if log_base > float("-inf") else epsilon

            # Values actually entering the weighted sum
            if norm_terms:
                s_base = norm_terms['base'][_i]
                s_tactic = norm_terms['tactic'][_i]
                s_stealth = norm_terms['stealth'][_i]
            else:
                s_base, s_tactic, s_stealth = log_base, log_tactic, log_stealth

            # Log-linear combination: LSTM + Tactic + Stealth
            log_terms = {
                'log_p_base': log_base,
                'log_p_bilstm': log_base,
                'log_p_tactic': log_tactic,
                'log_p_stealth': log_stealth,
            }

            if base_model_name != 'bilstm':
                log_terms[f'log_p_{base_model_name}'] = log_base

            if entropy_w is not None:
                # weights are derived from the data, not supplied
                log_score = math.log(max(
                    entropy_w[0] * s_base + entropy_w[1] * s_tactic
                    + entropy_w[2] * s_stealth, 1e-12))
            else:
                log_score = (
                    self.params.get(f'w_{base_model_name}', self.params.get('w_bilstm', 1.0)) * s_base
                    + self.params.get('w_tactic', 0.1) * s_tactic
                    + self.params.get('w_stealth', 0.1) * s_stealth
                )

            # Component breakdown for 3-factor display
            components = {
                'P_bilstm': p_base,
                'P_tactic': base_tactic,
                'P_stealth': stealth_score,
                'tactic_transition_prob': tactic_prob,
                'stealth_prior': stealth_score,
                'log_terms': log_terms,
                'base_model': base_model_name,
                # Factor scores for display (raw log scores)
                'factor_scores': {
                    'lstm': log_base,
                    'tactic': log_tactic,
                    'stealth': log_stealth,
                },
                # Weights applied
                'factor_weights': {
                    'lstm': self.params.get(f'w_{base_model_name}', self.params.get('w_bilstm', 1.0)),
                    'tactic': self.params.get('w_tactic', 0.1),
                    'stealth': self.params.get('w_stealth', 0.1),
                },
            }

            if base_model_name != 'bilstm':
                components[f'P_{base_model_name}'] = p_base

            candidate_scores.append((candidate, log_score, components))

        if not candidate_scores:
            return []

        # Softmax normalization
        max_log = max(score for _, score, _ in candidate_scores)
        exp_scores = [math.exp(score - max_log) for _, score, _ in candidate_scores]
        total_exp = sum(exp_scores)

        normalized_scores = []
        probs_buffer = []
        for (tech, log_score, components), exp_score in zip(candidate_scores, exp_scores):
            prob = exp_score / total_exp if total_exp > 0 else 1.0 / len(candidate_scores)
            components['log_score'] = log_score
            probs_buffer.append(prob)
            normalized_scores.append((tech, prob, components))

        # Compute entropy for uncertainty estimation
        entropy = 0.0
        for prob in probs_buffer:
            if prob > 0:
                entropy -= prob * math.log(prob)

        for idx in range(len(normalized_scores)):
            tech, prob, components = normalized_scores[idx]
            components['local_entropy'] = entropy
            normalized_scores[idx] = (tech, prob, components)

        # Sort by probability descending
        normalized_scores.sort(key=lambda x: x[1], reverse=True)

        return normalized_scores

    def next_probabilities(
        self,
        history: List[str],
        group_context: Optional[Any] = None,
        top_n_candidates: int = 200,
        use_scenario_priors: Optional[bool] = None,
        use_tactic_filtering: Optional[bool] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        return self.compute_candidate_components(
            history,
            group_context=group_context,
            top_n_candidates=top_n_candidates,
            use_scenario_priors=use_scenario_priors,
            use_tactic_filtering=use_tactic_filtering,
        )

    def predict_next_tactics(
        self,
        history: List[str],
        top_k: int = 3,
        group_context: Optional[Any] = None,
        tactic_prior_alpha: float = 1.0,
        seen_tactics: Optional[List[str]] = None,
        novelty_bonus: float = 4.0,
        recurrence_penalty: float = 0.3,
        max_recurrence: int = 1,
    ) -> List[Tuple[str, float]]:
        """
        Predict top-K next tactics (not techniques) with probabilities.

        Aggregates technique probabilities by tactic to produce a tactic-level
        distribution, then applies two complementary regulators:

        (A) Tactic-transition matrix prior (Bayesian multiplicative term).
            P_final(τ) ∝ P_matrix(τ|τ_current)^alpha × Σ_{t∈τ} P_LSTM(t|history)
            Zeroes out structurally impossible transitions (α=1.0) while
            preserving genuine minority patterns with α<1.0.

        (B) Tactic novelty/recurrence adjustment (diversity enforcement).
            Tactics NOT yet seen in the path receive a multiplicative bonus
            `novelty_bonus`, encouraging kill-chain progression into unexplored
            phases (discovery, lateral-movement, collection).
            Tactics that have already appeared ≥ max_recurrence times in the
            path are penalised by `recurrence_penalty^(count - max_recurrence + 1)`,
            suppressing degenerate loops (e.g., execution returning after
            defence-evasion has been established) without hardcoding any
            specific technique IDs.

        Args:
            history: Attack chain so far.
            top_k: Number of top tactics to return.
            group_context: Optional context for group-aware predictions.
            tactic_prior_alpha: Temperature for the matrix prior (1.0 = full enforcement).
            seen_tactics: Ordered list of tactics already committed to in the
                          current path (from the beam search state). Used for
                          novelty / recurrence scoring.
            novelty_bonus: Multiplicative weight applied to tactics not yet in
                           seen_tactics. Values > 1.0 surface underexplored phases.
            recurrence_penalty: Per-excess-recurrence multiplier applied to
                                 tactics seen more than max_recurrence times.
            max_recurrence: Number of times a tactic may appear before the
                            recurrence penalty activates.

        Returns:
            List of (tactic, probability) tuples sorted by probability descending.
        """
        from collections import defaultdict, Counter

        # Get technique-level predictions over FULL vocabulary.
        # Tactic pre-filtering is disabled so softmax normalises across
        # all techniques; tactic aggregation then gives unbiased estimates.
        next_probs = self.next_probabilities(
            history,
            group_context=group_context,
            top_n_candidates=300,
            use_tactic_filtering=False,
        )

        # Aggregate probabilities by tactic.
        # Fix bug #5: split probability mass across ALL tactics the technique
        # belongs to (weighted equally) so multi-tactic techniques like T1091
        # (initial-access + lateral-movement) contribute to every relevant tactic.
        tactic_scores: Dict[str, float] = defaultdict(float)
        for tech, prob, _ in next_probs:
            normalized_tid = _normalize_tid(tech)
            technique_tactics = self.technique_to_tactics.get(normalized_tid, [])
            if not technique_tactics:
                continue
            share = prob / len(technique_tactics)  # split evenly across all tactics
            for tac in technique_tactics:
                tactic_scores[tac] += share

        # --- (A) Tactic-transition matrix prior ---
        if self._has_tactic_model:
            _, transition_probs = self._get_tactic_context(history)
            for tac in list(tactic_scores.keys()):
                matrix_prob = transition_probs.get(tac, 0.0)
                tactic_scores[tac] *= (matrix_prob ** tactic_prior_alpha)

        # --- (B) Novelty bonus + recurrence penalty ---
        # This operates purely on the tactic labels of the current path; no
        # technique-level identifiers are referenced.
        if seen_tactics:
            tactic_counts = Counter(seen_tactics)
            for tac in list(tactic_scores.keys()):
                count = tactic_counts.get(tac, 0)
                if count == 0:
                    # Not yet explored — reward kill-chain progression
                    tactic_scores[tac] *= novelty_bonus
                elif count >= max_recurrence:
                    # Appeared too many times — attenuate to break loops
                    excess = count - max_recurrence + 1
                    tactic_scores[tac] *= (recurrence_penalty ** excess)

        # Normalize to probability distribution
        total = sum(tactic_scores.values())
        if total == 0:
            known_tactics = list(self._known_tactics)
            uniform_prob = 1.0 / max(len(known_tactics), 1)
            tactic_probs = [(tactic, uniform_prob) for tactic in known_tactics[:top_k]]
        else:
            tactic_probs = [(tactic, score / total) for tactic, score in tactic_scores.items()]

        # Sort by probability descending and return top-K
        tactic_probs.sort(key=lambda x: x[1], reverse=True)
        return tactic_probs[:top_k]

    def next_probabilities_for_tactic(
        self,
        history: List[str],
        target_tactic: str,
        top_n: int = 5,
        group_context: Optional[Any] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Get top-N techniques within a specific tactic.
        
        Filters technique predictions to only those belonging to the target tactic.
        Used for tactic-aware branching to select best techniques per tactic.
        
        Args:
            history: Attack chain so far
            target_tactic: Tactic to filter for (e.g., 'execution', 'persistence')
            top_n: Number of top techniques to return
            group_context: Optional context for group-aware predictions
        
        Returns:
            List of (technique, probability, components) tuples for the target tactic
        """
        # Get all technique predictions over FULL vocabulary.
        # Tactic pre-filtering is disabled so softmax normalises across
        # all techniques; the per-tactic filter + renormalise below
        # then gives probabilities consistent with the three-pillar model.
        all_probs = self.next_probabilities(
            history,
            group_context=group_context,
            top_n_candidates=300,
            use_tactic_filtering=False,
        )
        
        # Normalize target tactic
        normalized_target = _normalize_tactic_name(target_tactic)
        
        # Filter to target tactic
        filtered = []
        for tech, prob, components in all_probs:
            normalized_tid = _normalize_tid(tech)
            technique_tactics = self.technique_to_tactics.get(normalized_tid, [])
            
            # Check if technique belongs to target tactic
            if normalized_target in technique_tactics:
                filtered.append((tech, prob, components))
        
        # If no techniques found in this tactic, return empty
        if not filtered:
            return []
        
        # Renormalize probabilities within this tactic
        total_prob = sum(prob for _, prob, _ in filtered)
        if total_prob > 0:
            filtered = [
                (tech, prob / total_prob, components)
                for tech, prob, components in filtered
            ]
        
        # Return top-N
        return filtered[:top_n]

    def load_component_weights_from_disk(self, weights_path: Optional[Path] = None) -> None:
        """Load and apply component weights from a JSON file."""
        weights_path = weights_path or cfg.COMPONENT_WEIGHTS_JSON
        if weights_path.exists():
            print(f"Loading component weights from {weights_path}")
            try:
                with open(weights_path, 'r', encoding='utf-8') as f:
                    weights = json.load(f)
                self.update_component_weights(weights)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load or parse weights file: {e}")
        else:
            print("Using default component weights from config.")

    def update_component_weights(self, new_weights: Dict[str, float]) -> None:
        """Update the log-linear component weights."""
        if 'component_weights' not in self.params:
            self.params['component_weights'] = {}
        self.params['component_weights'].update(new_weights)
        print("Predictor component weights updated.")


def load_predictor(
    data_dir: str = "data",
    outputs_dir: str = "outputs",
    model_type: Optional[str] = None,
    params: Optional[Dict[str, float]] = None,
    data_source: str = "legacy",
    checkpoint_path: Optional[str] = None,
) -> Predictor:
    """
    Load predictor from saved artifacts.
    
    Args:
        data_dir: Data directory path
        outputs_dir: Outputs directory path
        model_type: Optional model type ('ngram', 'hmm', 'bilstm', etc.). 
                   If None, uses default n-gram behavior.
        params: Optional parameters dictionary
        data_source: Telemetry corpus identifier used when auxiliary models
            require raw sequences (e.g., training the HMM baseline).
        checkpoint_path: Optional path to a saved model artifact. Supports
            torch checkpoints (state_dict payload) and pickle-serialized models.
        
    Returns:
        Loaded Predictor instance
    """
    
    data_path = Path(data_dir)
    outputs_path = Path(outputs_dir)

    # Counts-based n-gram data intentionally not loaded.
    # The three-pillar model (LSTM + Tactic Prior + Stealth) scores the full
    # vocabulary via softmax; no pre-filtering by empirical frequency needed.
    counts_data: Dict[int, pd.DataFrame] = {}

    sigma_scores: Dict[str, float] = {}
    sigma_file = data_path / "sigma_stealth.csv"
    if sigma_file.exists():
        sigma_df = load_csv(sigma_file)
        sigma_scores = dict(zip(sigma_df['technique_id'], sigma_df['stealth_S']))

    ps_scores: Dict[str, float] = {}
     
    # Load embeddings (FAISS removed - using NPY only)
    embeddings_path = outputs_path / "tech_embeddings.npy"
    tech_index_path = outputs_path / "tech_index.csv"
    tech_to_idx_json = outputs_path / "tech_to_idx.json"
    
    # Prefer tech_to_idx.json if it exists (more comprehensive vocabulary)
    if tech_to_idx_json.exists():
        import json
        with open(tech_to_idx_json, 'r') as f:
            tech_to_idx = json.load(f)
        print(f"Loaded technique vocabulary from {tech_to_idx_json}: {len(tech_to_idx)} techniques")
        # Load embeddings directly since we have tech_to_idx
        embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
        index = None
    else:
        # Fallback to CSV-based loading
        embeddings, tech_to_idx, index = load_embeddings_and_index(
            embeddings_path, 
            outputs_path / "emb_index.faiss",  # Path kept for API compatibility, not used
            tech_index_path
        )

    tactic_mapping = load_tactic_mapping(cfg.TACTIC_MAPPING_JSON)
    tactic_transitions = load_tactic_transitions(cfg.TACTIC_TRANSITIONS_JSON)
    platform_mapping = load_platform_mapping(cfg.ENTERPRISE_ATTACK_JSON)
    tactic_ids = sorted({t for v in (tactic_mapping or {}).values() for t in v})

    technique_ids = list(tech_to_idx.keys())

    def _build_model_kwargs(model_key: str, config: Optional[Any] = None) -> Dict[str, Any]:
        model_key = model_key.lower()
        if model_key == 'ngram':
            return {
                'counts_data': counts_data,
                'ps_scores': ps_scores,
                'sigma_scores': sigma_scores,
                'embeddings': embeddings,
                'tech_to_idx': tech_to_idx,
                'index': index,
                'params': params or cfg.DEFAULT_PARAMS,
            }
        if model_key == 'hmm':
            sequences = load_sequences_for_source(data_source)
            return {'sequences': sequences}
        if model_key in {'bilstm', 'lstm'}:
            kwargs = {'embeddings': embeddings, 'tech_to_idx': tech_to_idx}
            if config is not None:
                kwargs['config'] = config
            return kwargs
        if model_key == 'transformer':
            kwargs = {'tactic_mapping': tactic_mapping, 'tactic_ids': tactic_ids}
            if config is not None:
                kwargs['config'] = config
            return kwargs
        return {}

    def _load_torch_checkpoint(model_key: str, config: Optional[Any] = None) -> BasePredictor:
        model_kwargs = _build_model_kwargs(model_key, config=config)
        try:
            return load_torch_model(
                checkpoint,
                model_type=model_key,
                **model_kwargs,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "PyTorch is required to load this checkpoint; install torch to proceed."
            ) from exc

    base_model = None
    resolved_model_type = model_type.lower() if model_type else None
    checkpoint = Path(checkpoint_path) if checkpoint_path else None

    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        torch_payload = None
        torch_error = None
        try:  # pragma: no cover - torch import is optional
            import torch

            torch_payload = torch.load(checkpoint, map_location="cpu")
        except Exception as exc:  # pragma: no cover - defensive
            torch_error = exc

        # Torch checkpoints (state_dict) or torch-loaded BasePredictor
        if isinstance(torch_payload, BasePredictor):
            base_model = torch_payload
            if resolved_model_type is None and hasattr(base_model, "get_model_name"):
                resolved_model_type = base_model.get_model_name()
        elif isinstance(torch_payload, dict) and 'state_dict' in torch_payload:
            if resolved_model_type is None:
                resolved_model_type = str(torch_payload.get('model_type', '')).lower()
            if not resolved_model_type:
                raise ValueError("Unable to determine model type from checkpoint; pass --model-type")

            base_model = _load_torch_checkpoint(resolved_model_type)
        elif torch_payload is not None and hasattr(torch_payload, 'keys'):
            # Raw state_dict (OrderedDict) - infer config and load
            from collections import OrderedDict
            if isinstance(torch_payload, (dict, OrderedDict)):
                print("[*] Detected raw state_dict format (no metadata)")
                
                # Require explicit model type for raw state_dict
                if resolved_model_type is None:
                    # Try to infer from keys - check if LSTM and whether bidirectional
                    has_lstm = any(k.startswith('lstm.') for k in torch_payload.keys())
                    has_embedding = 'embedding.weight' in torch_payload
                    if has_lstm and has_embedding:
                        # Check if bidirectional (has _reverse weights)
                        has_reverse = any('_reverse' in k for k in torch_payload.keys())
                        resolved_model_type = 'bilstm' if has_reverse else 'lstm'
                        print(f"[*] Auto-detected model type: {resolved_model_type}")
                    else:
                        raise ValueError(
                            "Cannot auto-detect model type from raw state_dict. "
                            "Please specify --model-type (e.g., bilstm, lstm, gru)"
                        )
                
                # Infer config from state_dict for BiLSTM/LSTM
                inferred_config = None
                inferred_embeddings = None
                if resolved_model_type in {'bilstm', 'lstm'}:
                    from .models.checkpoint import infer_bilstm_config_from_state_dict
                    try:
                        inferred_config = infer_bilstm_config_from_state_dict(torch_payload)
                        
                        # Validate vocab size matches
                        vocab_size_in_weights = torch_payload['embedding.weight'].shape[0]
                        vocab_size_in_index = len(tech_to_idx)
                        if vocab_size_in_weights != vocab_size_in_index:
                            raise ValueError(
                                f"Vocab size mismatch: state_dict has {vocab_size_in_weights} techniques, "
                                f"but tech_to_idx has {vocab_size_in_index}. "
                                f"Ensure you're using the same technique vocabulary that was used during training."
                            )
                        
                        # Extract embeddings from state_dict (they're part of the model)
                        # We'll use dummy embeddings since they're in the state_dict
                        embedding_dim = torch_payload['embedding.weight'].shape[1]
                        inferred_embeddings = np.zeros((vocab_size_in_weights, embedding_dim), dtype=np.float32)
                        print(f"[*] Using {embedding_dim}-dim embeddings from state_dict")
                        
                    except Exception as e:
                        raise ValueError(f"Failed to infer config from raw state_dict: {e}")
                
                # Create model skeleton and load weights
                # Use inferred embeddings to match the model's expected dimensions
                saved_embeddings = embeddings
                if inferred_embeddings is not None:
                    embeddings = inferred_embeddings
                    
                model_kwargs = _build_model_kwargs(resolved_model_type, config=inferred_config)
                base_model = ModelFactory.create_model(resolved_model_type, technique_ids, **model_kwargs)
                
                # Restore original embeddings for predictor use
                embeddings = saved_embeddings
                
                if not hasattr(base_model, '_build_model'):
                    raise ValueError(f"Model type {resolved_model_type} doesn't support torch loading")
                
                base_model.model = base_model._build_model()
                if base_model.model is None:
                    raise RuntimeError("Failed to build model skeleton")
                
                # Load weights
                base_model.model.load_state_dict(torch_payload)
                base_model.model.eval()
                print(f"[OK] Loaded raw state_dict into {resolved_model_type} model")
        else:
            # torch.load failed (likely missing torch). Try pickle as a fallback.
            pass

        if base_model is None:
            import pickle

            with open(checkpoint, 'rb') as f:
                payload = pickle.load(f)

            if isinstance(payload, BasePredictor):
                base_model = payload
                if resolved_model_type is None and hasattr(base_model, "get_model_name"):
                    resolved_model_type = base_model.get_model_name()
            elif isinstance(payload, dict) and 'state_dict' in payload:
                if resolved_model_type is None:
                    resolved_model_type = str(payload.get('model_type', '')).lower()
                if not resolved_model_type:
                    raise ValueError("Unable to determine model type from checkpoint; pass --model-type")
                base_model = _load_torch_checkpoint(resolved_model_type)
            else:
                raise ValueError(
                    "Unsupported checkpoint format. Expected a serialized BasePredictor or a torch state_dict payload."
                )
        elif resolved_model_type is None and torch_error:
            # We loaded a model but couldn't resolve type and torch errored
            print(f"Loaded model from checkpoint but could not resolve type automatically ({torch_error})")

    # Create base model from scratch if requested and not loaded from checkpoint
    if base_model is None and resolved_model_type and resolved_model_type != 'ngram':
        try:
            model_kwargs = _build_model_kwargs(resolved_model_type)
            base_model = ModelFactory.create_model(resolved_model_type, technique_ids, **model_kwargs)
        except Exception as e:
            print(f"Warning: Failed to create base model {resolved_model_type}: {e}")
            print("Falling back to n-gram behavior")
            base_model = None
    
    predictor = Predictor(
        counts_data,
        ps_scores,
        sigma_scores,
        embeddings,
        tech_to_idx,
        index,
        params=params,
        base_model=base_model,
        tactic_mapping=tactic_mapping,
        tactic_transitions=tactic_transitions,
        platform_mapping=platform_mapping,
    )

    predictor.load_component_weights_from_disk()

    return predictor
