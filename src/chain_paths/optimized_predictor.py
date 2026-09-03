"""Optimized prediction module with streamlined tactic branching.

This module provides a simplified, efficient prediction pipeline using:
- LSTM + Tactic Prior + Stealth Score (3 orthogonal signals)
- Configurable tactic branching for diverse path generation
- Full probability tracking for each step and path
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math

from .predictor import Predictor


@dataclass
class TacticBranchConfig:
    """Configuration for tactic-aware path generation."""
    num_tactic_branches: int = 3       # Number of tactical directions to explore
    techniques_per_branch: int = 2      # Techniques to consider per tactic
    max_depth: int = 6                  # Maximum path length
    max_paths: int = 10                 # Maximum total paths to generate
    diversity_penalty: float = 0.2      # Penalty for redundant paths
    min_probability: float = 1e-9       # Minimum probability threshold

    # ── Ablation switches (Section 5.3). All default to the full architecture,
    #    so production behaviour is byte-identical unless a switch is flipped. ──
    use_lstm: bool = True               # Stage 1. False -> candidates are ranked
                                        # by corpus unigram frequency instead of
                                        # by the neural conditional.
    use_tactic_gate: bool = True        # Stage 2. False -> no tactic partitioning;
                                        # candidates are drawn from the full
                                        # Windows-applicable vocabulary.
    use_diversity_pruning: bool = True   # False -> the active beam is pruned by
                                        # global log-probability only (classical
                                        # beam search), not per-tactic buckets.
    use_per_tactic_cap: bool = True      # False -> the ceil(max_paths/B) cap on
                                        # completed paths per first-step tactic
                                        # is not enforced.


@dataclass
class StepProbability:
    """Probability components for a single prediction step."""
    position: int            # Step position in path (0-indexed)
    technique: str
    tactic: str
    prev_tactic: str         # Previous tactic for transition tracking
    p_bilstm: float          # BiLSTM neural probability
    p_tactic: float          # Tactic-level probability P(τ | history)
    p_stealth: float         # Stealth score (detection difficulty)
    p_combined: float        # Full-vocab combined probability P(tech | history)
    p_within_tactic: float = 0.0  # Within-tactic renormalized P(tech | τ, history)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'position': self.position,
            'technique': self.technique,
            'tactic': self.tactic,
            'prev_tactic': self.prev_tactic,
            'P_bilstm': self.p_bilstm,
            'P_tactic': self.p_tactic,
            'P_stealth': self.p_stealth,
            'P_combined': self.p_combined,
            'P_within_tactic': self.p_within_tactic,
        }


@dataclass  
class PathResult:
    """Complete result for a generated attack path."""
    path_id: str
    techniques: List[str]
    tactic_sequence: List[str]
    step_probabilities: List[StepProbability]
    path_probability: float           # Product of step probabilities
    log_probability: float            # Sum of log probabilities
    tactic_transition_probs: List[float]  # P(tactic_i+1 | tactic_i)
    dominant_tactic: str
    branch_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'path_id': self.path_id,
            'techniques': self.techniques,
            'tactic_sequence': self.tactic_sequence,
            'step_details': [sp.to_dict() for sp in self.step_probabilities],
            'path_probability': self.path_probability,
            'log_probability': self.log_probability,
            'tactic_transition_probs': self.tactic_transition_probs,
            'dominant_tactic': self.dominant_tactic,
            'branch_id': self.branch_id,
            'path_length': len(self.techniques),
        }


class OptimizedPredictor:
    """
    Streamlined predictor with efficient tactic branching.
    
    Uses simplified scoring: LSTM + Tactic Prior + Stealth Score
    """
    
    def __init__(self, predictor: Predictor, config: Optional[TacticBranchConfig] = None, verbose: bool = False, scoring_mode: str = 'log_linear', tactic_prior_alpha: float = 1.0):
        self.predictor = predictor
        self.config = config or TacticBranchConfig()
        self.verbose = verbose
        self.scoring_mode = scoring_mode  # 'log_linear' or 'cascaded'
        self.tactic_prior_alpha = tactic_prior_alpha
        
    def predict_next_step(
        self,
        history: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Predict next techniques with full probability breakdown.
        
        Returns:
            List of (technique, probability, components) tuples
        """
        return self.predictor.next_probabilities(
            history,
            top_n_candidates=max(top_k * 3, 50),  # Get more candidates for filtering
        )[:top_k]
    
    def get_technique_tactic(self, technique: str) -> str:
        """Get the most operationally representative tactic for a technique.
        
        Fix bug #4: for multi-tactic techniques (e.g. T1091 has
        ['initial-access', 'lateral-movement']), return the tactic that
        appears latest in the kill-chain ordering so beam diversity
        tracking correctly labels the path phase.
        """
        TACTIC_ORDER = [
            'reconnaissance', 'resource-development', 'initial-access',
            'execution', 'persistence', 'privilege-escalation',
            'defense-evasion', 'credential-access', 'discovery',
            'lateral-movement', 'collection', 'command-and-control',
            'exfiltration', 'impact',
        ]
        tactics = self.predictor.technique_to_tactics.get(technique.upper(), [])
        if not tactics:
            return 'unknown'
        # Return the latest-phase tactic (highest kill-chain position)
        def _order(t: str) -> int:
            try:
                return TACTIC_ORDER.index(t)
            except ValueError:
                return -1
        return max(tactics, key=_order)
    
    def compute_tactic_transition_prob(
        self,
        from_tactic: str,
        to_tactic: str
    ) -> float:
        """Get P(to_tactic | from_tactic) from transition matrix."""
        probs = self.predictor.tactic_transition_probs.get(from_tactic, {})
        return probs.get(to_tactic, 0.01)  # Small default for unseen transitions

    # ── Ablation support (Section 5.3) ────────────────────────────────────
    # Sentinel used by the -Stage-2 ablation to denote "no tactic constraint".
    _ANY_TACTIC = '__any__'

    def _unigram(self) -> Dict[str, float]:
        """Corpus unigram distribution over the technique vocabulary, used as the
        detection-agnostic, sequence-agnostic replacement for the LSTM in the
        -Stage-1 ablation. Cached on first use."""
        cached = getattr(self, '_unigram_cache', None)
        if cached is not None:
            return cached
        counts: Dict[str, float] = {}
        # Prefer an empirical count table if the predictor exposes one; fall back
        # to a uniform distribution over the vocabulary.
        src = getattr(self.predictor, 'technique_counts', None)
        if src:
            counts = {str(k).upper(): float(v) for k, v in src.items() if v > 0}
        if not counts:
            try:
                import pandas as pd
                from . import config as _cfg
                df = pd.read_parquet(_cfg.SEQUENCES_PARQUET)
                from collections import Counter as _C
                c = _C()
                for s in df['sequence']:
                    for t in s:
                        c[str(t).upper()] += 1
                counts = {k: float(v) for k, v in c.items()}
            except Exception:
                counts = {str(t).upper(): 1.0 for t in self.predictor.tech_to_idx}
        vocab = {str(t).upper() for t in self.predictor.tech_to_idx}
        counts = {k: v for k, v in counts.items() if k in vocab}
        total = sum(counts.values()) or 1.0
        self._unigram_cache = {k: v / total for k, v in counts.items()}
        return self._unigram_cache

    def _frequency_candidates(
        self,
        target_tactic: str,
        top_n: int,
        exclude: Optional[List[str]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Candidate pool ranked by corpus unigram frequency instead of by the
        LSTM conditional. Returns the same (technique, prob, components) triples
        the neural path produces, so downstream stages are unchanged."""
        uni = self._unigram()
        skip = {str(t).upper() for t in (exclude or [])}
        # Apply the SAME Windows/AD platform restriction the neural path applies
        # via Predictor.get_candidate_set, so the only difference between this
        # ablation arm and the full architecture is the ranking signal.
        p_ok = getattr(self.predictor, '_use_platform_filtering', False)
        p_set = set(getattr(self.predictor, '_allowed_platforms', set()) or set())
        pool = []
        for tech, p in uni.items():
            if tech in skip:
                continue
            if p_ok and p_set and not self.predictor._is_technique_in_platforms(tech, p_set):
                continue
            if target_tactic is not self._ANY_TACTIC and target_tactic != self._ANY_TACTIC:
                tacs = self.predictor.technique_to_tactics.get(tech, [])
                if target_tactic not in tacs:
                    continue
            pool.append((tech, p))
        if not pool:
            return []
        pool.sort(key=lambda x: x[1], reverse=True)
        pool = pool[:top_n]
        z = sum(p for _, p in pool) or 1.0
        out = []
        for tech, p in pool:
            stealth = self.predictor.sigma_scores.get(tech, 1.0)
            out.append((tech, p / z, {
                'P_bilstm': p / z, 'P_lstm': p / z,
                'P_stealth': stealth, 'stealth_prior': stealth,
            }))
        return out
    
    def generate_paths_with_tactic_branching(
        self,
        seed_techniques: List[str],
        config: Optional[TacticBranchConfig] = None,
    ) -> List[PathResult]:
        """
        Generate diverse attack paths using tactic-aware branching.
        
        Algorithm:
        1. Start with seed techniques
        2. At each step, get top-K next tactics
        3. For each tactic, get top-N techniques
        4. Expand paths while tracking probabilities
        5. Return top paths sorted by probability
        
        Args:
            seed_techniques: Starting techniques for the path
            config: Optional config override
            
        Returns:
            List of PathResult with full probability tracking
        """
        cfg = config or self.config
        
        # Initialize with seed path
        seed_tactics = [self.get_technique_tactic(t) for t in seed_techniques]
        seed_len = len(seed_techniques)
        
        # ── True tactic-branching beam search ────────────────────────────────
        # Begin with a single seed path.  The main loop below expands every
        # active path into num_tactic_branches tactic directions at EACH step,
        # creating a genuine divergent tree: paths share common prefixes until
        # a better branch appears, then diverge.  max_per_tactic (computed
        # below) ensures the dominant tactic cannot monopolise all output slots.
        active_paths: List[Tuple[List[str], List[str], float, List[StepProbability], List[float], str]] = [
            (list(seed_techniques), list(seed_tactics), 0.0, [], [], 'seed')
        ]
        
        completed_paths: List[PathResult] = []
        path_counter = 0
        
        # Each first-step tactic is allowed at most max_per_tactic completed paths
        # so the dominant tactic cannot monopolise all output slots.
        # Ablation: with use_per_tactic_cap=False the cap is lifted entirely.
        max_per_tactic = (
            math.ceil(cfg.max_paths / max(cfg.num_tactic_branches, 1))
            if getattr(cfg, 'use_per_tactic_cap', True) else float('inf')
        )
        tactic_completions: dict = {}
        
        while active_paths and len(completed_paths) < cfg.max_paths:
            # Sort by probability and take best path
            active_paths.sort(key=lambda x: x[2], reverse=True)
            current = active_paths.pop(0)
            
            techniques, tactics, log_prob, step_probs, tactic_trans, branch_id = current
            
            # Identify first post-seed tactic for diversity tracking
            first_tactic = tactics[seed_len] if len(tactics) > seed_len else '__seed__'
            
            # Check if path is complete
            if len(techniques) >= cfg.max_depth:
                # Skip exact duplicate technique sequences
                seq_key = tuple(techniques)
                seen_seqs = {tuple(r.techniques) for r in completed_paths}
                if seq_key in seen_seqs:
                    continue
                # Accept only if this tactic still has capacity
                if tactic_completions.get(first_tactic, 0) < max_per_tactic:
                    path_counter += 1
                    result = self._create_path_result(
                        path_counter, techniques, tactics, log_prob, 
                        step_probs, tactic_trans, branch_id
                    )
                    completed_paths.append(result)
                    tactic_completions[first_tactic] = tactic_completions.get(first_tactic, 0) + 1
                continue
            
            # Get top-K next tactics (standard per-step branching: num_tactic_branches=3)
            # Fix bug #3: pass the beam's current tactic history so novelty_bonus
            # and recurrence_penalty fire correctly during every expansion step.
            beam_tactics = tactics[seed_len:]  # tactics committed after the seed
            if getattr(cfg, 'use_tactic_gate', True):
                tactic_predictions = self.predictor.predict_next_tactics(
                    techniques,
                    top_k=cfg.num_tactic_branches,
                    tactic_prior_alpha=self.tactic_prior_alpha,
                    seen_tactics=beam_tactics if beam_tactics else None,
                )
            else:
                # Ablation (-Stage 2): no empirical tactic-transition gate. A
                # single pseudo-branch draws candidates from the whole vocabulary,
                # so structurally unattested transitions become reachable.
                tactic_predictions = [(self._ANY_TACTIC, 1.0)]
            
            if not tactic_predictions:
                # No valid tactics, complete this path
                path_counter += 1
                result = self._create_path_result(
                    path_counter, techniques, tactics, log_prob,
                    step_probs, tactic_trans, branch_id
                )
                completed_paths.append(result)
                continue
            
            # For each tactic branch
            for tactic_idx, (next_tactic, tactic_prob) in enumerate(tactic_predictions):
                # In cascaded mode: fetch a small pool of LSTM-credible candidates,
                # then re-rank by stealth. Pool = T+1 so Stage 3 can only swap
                # LSTM-rank-1 for LSTM-rank-2/3, never for zero-probability tail mass.
                if self.scoring_mode == 'cascaded':
                    fetch_n = cfg.techniques_per_branch + 2
                else:
                    fetch_n = cfg.techniques_per_branch

                # Get the Stage-1 candidate pool for this branch.
                if not getattr(cfg, 'use_lstm', True):
                    # Ablation (-Stage 1): the neural conditional is removed
                    # entirely. Candidates come from the corpus unigram
                    # distribution restricted to the branch tactic, with the
                    # same pool size, so only the ranking signal changes and
                    # not the search budget.
                    tech_probs = self._frequency_candidates(next_tactic, fetch_n,
                                                            exclude=techniques)
                elif next_tactic is self._ANY_TACTIC:
                    tech_probs = self.predictor.next_probabilities(
                        techniques, top_n_candidates=300, use_tactic_filtering=False,
                    )[:fetch_n]
                else:
                    tech_probs = self.predictor.next_probabilities_for_tactic(
                        techniques,
                        target_tactic=next_tactic,
                        top_n=fetch_n,
                    )

                if not tech_probs:
                    continue

                # Cascaded: re-rank by stealth score, take top techniques_per_branch
                if self.scoring_mode == 'cascaded' and len(tech_probs) > cfg.techniques_per_branch:
                    # Position-aware delivery-technique filter:
                    # T1204 (User Execution) and T1566 (Phishing) are initial-delivery
                    # techniques — operationally unrealistic at depth >=3 where the
                    # adversary already has persistent system access.
                    chain_depth = len(techniques) - seed_len
                    if chain_depth >= 2:
                        DELIVERY_PREFIXES = ('T1204', 'T1566')
                        filtered_tech = [
                            (t, p, c) for t, p, c in tech_probs
                            if not any(t.startswith(pre) for pre in DELIVERY_PREFIXES)
                        ]
                        if filtered_tech:  # only apply filter if alternatives exist
                            tech_probs = filtered_tech
                    tech_probs.sort(
                        key=lambda x: x[2].get('P_stealth', x[2].get('stealth_prior', 0.0)),
                        reverse=True,
                    )
                    tech_probs = tech_probs[:cfg.techniques_per_branch]
                
                # Create new paths for each technique
                for tech, prob, components in tech_probs:
                    # Skip if technique already in path (no cycles)
                    if tech in techniques:
                        continue
                    
                    if prob < cfg.min_probability:
                        continue
                    
                    # prob is within-tactic renormalized: P(tech | τ, history)
                    # Full-vocab probability: P(tech | history) = P(tech | τ, hist) × P(τ | hist)
                    full_vocab_prob = prob * tactic_prob
                    
                    # Extract component probabilities
                    p_bilstm = components.get('P_bilstm', components.get('P_lstm', prob))
                    p_stealth = components.get('P_stealth', components.get('stealth_prior', 1.0))
                    
                    # When the Stage-2 gate is ablated there is no branch tactic,
                    # so the step is labelled with the technique's own resolved
                    # tactic; the transition probability is then read off the
                    # empirical matrix purely for reporting, not for filtering.
                    step_tactic = (self.get_technique_tactic(tech)
                                   if next_tactic == self._ANY_TACTIC else next_tactic)

                    # Compute tactic transition probability
                    current_tactic = tactics[-1] if tactics else '__start__'
                    p_tactic_trans = self.compute_tactic_transition_prob(current_tactic, step_tactic)

                    # Create step probability record (position is length after seed)
                    step_position = len(techniques) - len(seed_techniques) + 1
                    step_prob = StepProbability(
                        position=step_position,
                        technique=tech,
                        tactic=step_tactic,
                        prev_tactic=current_tactic,
                        p_bilstm=p_bilstm,
                        p_tactic=tactic_prob,
                        p_stealth=p_stealth,
                        p_combined=full_vocab_prob,
                        p_within_tactic=prob,
                    )
                    
                    # Update path — rank by full-vocab probability
                    new_techniques = techniques + [tech]
                    new_tactics = tactics + [step_tactic]
                    new_log_prob = log_prob + math.log(max(full_vocab_prob, cfg.min_probability))
                    new_step_probs = step_probs + [step_prob]
                    new_tactic_trans = tactic_trans + [p_tactic_trans]
                    new_branch_id = f"{branch_id}_{step_tactic[:3]}_{tactic_idx}"
                    
                    active_paths.append((
                        new_techniques, new_tactics, new_log_prob,
                        new_step_probs, new_tactic_trans, new_branch_id
                    ))
            
            # Prune active paths to prevent explosion — diversity-preserving.
            # Keep the top paths per first-step tactic so minority branches
            # (discovery, lateral-movement) are not squeezed out by the dominant
            # defense-evasion branch.
            if len(active_paths) > cfg.max_paths * 10:
                if getattr(cfg, 'use_diversity_pruning', True):
                    from collections import defaultdict
                    buckets: dict = defaultdict(list)
                    for pth in active_paths:
                        key = pth[1][seed_len] if len(pth[1]) > seed_len else '__seed__'
                        buckets[key].append(pth)
                    per_bucket = max(3, (cfg.max_paths * 5) // max(len(buckets), 1))
                    pruned: list = []
                    for bucket in buckets.values():
                        bucket.sort(key=lambda x: x[2], reverse=True)
                        pruned.extend(bucket[:per_bucket])
                    active_paths = pruned
                else:
                    # Ablation (-diversity pruning): classical beam search keeps
                    # the globally highest-probability paths, with no per-tactic
                    # bucketing, so the dominant corridor can crowd out the rest.
                    # The retained budget matches the diversity-preserving branch
                    # so only the selection rule differs.
                    keep = max(3, cfg.max_paths * 5)
                    active_paths.sort(key=lambda x: x[2], reverse=True)
                    active_paths = active_paths[:keep]
        
        # Add remaining incomplete active paths as completed (longest first — closest to max_depth)
        remaining = sorted(active_paths, key=lambda x: -len(x[0]))
        for path_data in remaining:
            if len(completed_paths) >= cfg.max_paths:
                break
            techniques, tactics, log_prob, step_probs, tactic_trans, branch_id = path_data
            first_tactic = tactics[seed_len] if len(tactics) > seed_len else '__seed__'
            seq_key = tuple(techniques)
            seen_seqs = {tuple(r.techniques) for r in completed_paths}
            if seq_key in seen_seqs:
                continue
            if tactic_completions.get(first_tactic, 0) < max_per_tactic:
                path_counter += 1
                result = self._create_path_result(
                    path_counter, techniques, tactics, log_prob,
                    step_probs, tactic_trans, branch_id
                )
                completed_paths.append(result)
                tactic_completions[first_tactic] = tactic_completions.get(first_tactic, 0) + 1
        
        # Sort by probability
        completed_paths.sort(key=lambda x: x.log_probability, reverse=True)
        
        return completed_paths[:cfg.max_paths]
    
    def _create_path_result(
        self,
        path_id: int,
        techniques: List[str],
        tactics: List[str],
        log_prob: float,
        step_probs: List[StepProbability],
        tactic_trans: List[float],
        branch_id: str,
    ) -> PathResult:
        """Create a PathResult from path components."""
        # Compute dominant tactic
        from collections import Counter
        tactic_counts = Counter(tactics)
        dominant = tactic_counts.most_common(1)[0][0] if tactic_counts else 'unknown'
        
        # Compute path probability
        try:
            path_prob = math.exp(log_prob)
        except OverflowError:
            path_prob = 0.0
        
        return PathResult(
            path_id=f"path_{path_id:03d}",
            techniques=techniques,
            tactic_sequence=tactics,
            step_probabilities=step_probs,
            path_probability=path_prob,
            log_probability=log_prob,
            tactic_transition_probs=tactic_trans,
            dominant_tactic=dominant,
            branch_id=branch_id,
        )
    
    def compute_path_confidence(
        self,
        techniques: List[str],
        n_seed: int = 1,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Compute path confidence as mean per-step rank percentile.

        At each post-seed step, scores ALL techniques via the three-pillar
        log-linear model and finds the rank of the chosen technique.
        Percentile = (V - rank) / (V - 1) where V = vocabulary size.

        Returns:
            (mean_percentile, per_step_details)
        """
        vocab_size = len(self.predictor.tech_to_idx)
        step_details: List[Dict[str, Any]] = []

        for i in range(n_seed, len(techniques)):
            history = techniques[:i]
            chosen = techniques[i]

            all_scores = self.predictor.next_probabilities(
                history,
                top_n_candidates=vocab_size,
                use_tactic_filtering=False,
            )

            rank = vocab_size  # worst-case default
            for r, (tech, _prob, _comp) in enumerate(all_scores, 1):
                if tech == chosen:
                    rank = r
                    break

            percentile = (vocab_size - rank) / (vocab_size - 1) * 100.0
            step_details.append({
                'step': i - n_seed + 1,
                'technique': chosen,
                'rank': rank,
                'vocab_size': vocab_size,
                'percentile': percentile,
            })

        mean_pct = (
            sum(d['percentile'] for d in step_details) / len(step_details)
            if step_details else 0.0
        )
        return mean_pct, step_details

    def predict_single_step_detailed(
        self,
        history: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get detailed predictions for a single step.
        
        Returns predictions with full probability breakdown.
        """
        predictions = self.predict_next_step(history, top_k=top_k)
        
        results = []
        current_tactic = self.get_technique_tactic(history[-1]) if history else '__start__'
        
        for tech, prob, components in predictions:
            next_tactic = self.get_technique_tactic(tech)
            tactic_trans_prob = self.compute_tactic_transition_prob(current_tactic, next_tactic)
            
            results.append({
                'technique': tech,
                'tactic': next_tactic,
                'probability': prob,
                'P_bilstm': components.get('P_bilstm', components.get('P_lstm', prob)),
                'P_tactic': components.get('P_tactic', tactic_trans_prob),
                'P_stealth': components.get('P_stealth', components.get('stealth_prior', 1.0)),
                'tactic_transition_prob': tactic_trans_prob,
                'from_tactic': current_tactic,
                'to_tactic': next_tactic,
            })
        
        return results


def generate_attack_paths(
    predictor: Predictor,
    seed_techniques: List[str],
    num_tactic_branches: int = 3,
    techniques_per_branch: int = 2,
    max_depth: int = 6,
    max_paths: int = 10,
) -> List[Dict[str, Any]]:
    """
    Convenience function to generate attack paths with tactic branching.
    
    Args:
        predictor: Trained predictor instance
        seed_techniques: Starting techniques
        num_tactic_branches: Number of tactical directions (default: 3)
        techniques_per_branch: Techniques per tactic (default: 2)
        max_depth: Maximum path length (default: 6)
        max_paths: Maximum paths to generate (default: 10)
        
    Returns:
        List of path dictionaries with full probability tracking
    """
    config = TacticBranchConfig(
        num_tactic_branches=num_tactic_branches,
        techniques_per_branch=techniques_per_branch,
        max_depth=max_depth,
        max_paths=max_paths,
    )
    
    opt_predictor = OptimizedPredictor(predictor, config)
    paths = opt_predictor.generate_paths_with_tactic_branching(seed_techniques)
    
    return [p.to_dict() for p in paths]
