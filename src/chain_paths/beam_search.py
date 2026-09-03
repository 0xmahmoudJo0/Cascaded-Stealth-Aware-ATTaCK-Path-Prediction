"""
Beam search implementation for Top-K attack path enumeration.

This module implements beam search to find the most probable attack paths
using the chain-aware predictor. Supports both standard beam search and
multi-tactic branching for diverse path generation.
"""

import math
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Any, Set, Tuple

import numpy as np
from tqdm import tqdm

from .predictor import Predictor
from .io import save_paths_json, ensure_dir


def _coalesce_component(components: Dict[str, Any], *keys: str) -> Any:
    """Return the first present key from ``components``.

    Beam search historically consumed predictor diagnostics that used
    uppercase keys (``P_count``, ``M_emb``, ``S``).  The refactored
    predictor now emits lowercase snake_case names instead.  To maintain
    compatibility with artifacts saved prior to the refactor—and to
    tolerate mixtures of the two naming conventions—we coalesce values by
    checking both the new and legacy keys before falling back to ``None``.
    """

    if not isinstance(components, dict):
        return None

    for key in keys:
        if key in components:
            return components[key]
    return None


def _normalize_components(components: Dict[str, Any], tech: str, step: int, prob: float) -> Dict[str, Any]:
    """Normalize predictor diagnostics into a stable schema.

    Older checkpoints stored uppercase keys, whereas the refactored
    predictor emits lowercase names.  Some callers also expect derived
    values such as ``S`` or ``probability`` to be present.  This helper
    harmonises all of those possibilities and back-fills sensible defaults
    so downstream consumers never trigger ``KeyError`` even when the
    predictor omits an optional diagnostic.
    """

    normalized: Dict[str, Any] = {
        'step': step,
        'tech': tech,
        'p_count': _coalesce_component(components, 'p_count', 'P_count'),
        'p_emb': _coalesce_component(components, 'p_emb', 'P_emb'),
        'p_interp': _coalesce_component(components, 'p_interp', 'P_interp'),
        'p_prior': _coalesce_component(components, 'p_prior', 'P_prior'),
        'm_emb': _coalesce_component(components, 'm_emb', 'M_emb') or 1.0,
        'omega': _coalesce_component(components, 'omega', 'Omega') or 1.0,
        'support': _coalesce_component(components, 'support'),
        'prob': prob,
        'probability': _coalesce_component(components, 'probability', 'prob', 'P') or prob,
        'logit': _coalesce_component(components, 'logit'),
        'log_p_count': _coalesce_component(components, 'log_p_count', 'log_P_count'),
        'log_p_emb': _coalesce_component(components, 'log_p_emb', 'log_P_emb'),
        'log_p_interp': _coalesce_component(components, 'log_p_interp', 'log_P_interp'),
        'log_p_prior': _coalesce_component(components, 'log_p_prior', 'log_P_prior'),
        'log_stealth': _coalesce_component(components, 'log_stealth', 'log_S'),
    }

    stealth = _coalesce_component(components, 's_stealth', 'stealth', 'S')
    if stealth is None:
        stealth = 1.0

    normalized['s_stealth'] = stealth
    normalized['stealth'] = _coalesce_component(components, 'stealth', 'S') or stealth
    normalized['S'] = stealth

    if normalized['logit'] is None and prob > 0:
        normalized['logit'] = math.log(prob)

    if normalized['log_stealth'] is None and stealth not in (None, 0):
        normalized['log_stealth'] = math.log(stealth)

    return normalized


@dataclass
class TacticBranch:
    """Metadata for a tactical branch in multi-tactic beam search."""
    branch_id: str
    primary_tactic: str
    branch_probability: float
    tactic_sequence: List[str]


class BeamEntry:
    """Represents a single beam entry during search."""

    def __init__(self, path: List[str], log_probability: float,
                 per_step_components: List[Dict[str, Any]],
                 branch: Optional[TacticBranch] = None):
        """
        Initialize beam entry.
        
        Args:
            path: Current path (list of techniques)
            log_probability: Cumulative log-probability
            per_step_components: Components for each step
            branch: Optional tactical branch metadata
        """
        self.path = path
        self.log_probability = log_probability
        self.per_step_components = per_step_components
        self.branch = branch

    def __lt__(self, other):
        """For sorting by probability (descending)."""
        return self.log_probability > other.log_probability


class BeamSearch:
    """
    Beam search implementation for attack path enumeration.
    """
    
    def __init__(self, predictor: Predictor, beam_width: int = 200, 
                 max_depth: int = 6, top_k_paths: int = 100,
                 all_sequences: Optional[List[List[str]]] = None):
        """
        Initialize beam search.
        
        Args:
            predictor: Chain-aware predictor
            beam_width: Beam width for search
            max_depth: Maximum path depth
            top_k_paths: Number of top paths to return
            all_sequences: All sequences from the dataset for finding examples.
        """
        self.predictor = predictor
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.top_k_paths = top_k_paths
        
        print(f"Initialized beam search: width={beam_width}, max_depth={max_depth}, top_k={top_k_paths}")

        # Pre-process sequences for fast lookup
        self.sequences_str = None
        if all_sequences:
            print(f"Indexing {len(all_sequences)} sequences for example lookup...")
            # Store sequences as space-separated strings for fast substring search
            self.sequences_str = [" " + " ".join(seq) + " " for seq in all_sequences]
            self.original_sequences = all_sequences

    
    def expand_beam(
        self,
        beam: List[BeamEntry],
        group_context: Optional[Any] = None,
    ) -> List[BeamEntry]:
        """
        Expand beam by one step.
        
        Args:
            beam: Current beam
            group_context: Optional attacker context for priors
            
        Returns:
            Expanded beam
        """
        new_beam = []
        
        for entry in beam:
            # Get next probabilities
            next_probs = self.predictor.next_probabilities(
                entry.path,
                group_context=group_context,
                top_n_candidates=50,
            )

            # Create new entries
            for tech, prob, components in next_probs:
                # Avoid cycles (don't repeat techniques in path)
                if tech in entry.path:
                    continue

                if prob <= 0:
                    continue

                # Create new path
                new_path = entry.path + [tech]
                
                # Compute new probability
                new_log_prob = entry.log_probability + math.log(prob)
                
                # Create new components
                new_components = entry.per_step_components + [{
                    'step': len(new_path),
                    'tech': tech,
                    'P_count': components.get('P_count', components.get('P_bilstm', 0.0)),
                    'P_emb': components.get('P_emb', 0.0),
                    'M_emb': components.get('M_emb', 0.0),
                    'S': components.get('stealth_prior', components.get('P_stealth', 0.0)),
                    'prob': prob
                }]
                
                # Create new beam entry
                new_entry = BeamEntry(new_path, new_log_prob, new_components)
                new_beam.append(new_entry)
        
        # Sort by probability and keep top beam_width
        new_beam.sort()
        return new_beam[:self.beam_width]
    
    def _compute_path_similarity(self, path1: List[str], path2: List[str]) -> float:
        """
        Compute similarity between two paths using Jaccard similarity.
        
        Args:
            path1: First path
            path2: Second path
        
        Returns:
            Similarity score between 0.0 (completely different) and 1.0 (identical)
        """
        set1 = set(path1)
        set2 = set(path2)
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_diversity_bonus(
        self,
        candidate_path: List[str],
        existing_paths: List[List[str]],
        diversity_weight: float = 0.3
    ) -> float:
        """
        Compute diversity bonus for a candidate path.
        
        Penalizes paths that are too similar to existing paths in the beam.
        Encourages exploration of genuinely different tactical directions.
        
        Args:
            candidate_path: Path to evaluate
            existing_paths: Paths already in beam
            diversity_weight: Weight λ for diversity penalty
        
        Returns:
            Diversity bonus (negative value = penalty)
        """
        if not existing_paths:
            return 0.0
        
        # Compute maximum similarity to any existing path
        max_similarity = max(
            self._compute_path_similarity(candidate_path, existing_path)
            for existing_path in existing_paths
        )
        
        # Apply penalty proportional to maximum similarity
        diversity_bonus = -diversity_weight * max_similarity
        
        return diversity_bonus
    
    def expand_with_tactic_branching(
        self,
        beam: List[BeamEntry],
        group_context: Optional[Any] = None,
        branching_factor: int = 3,
        techniques_per_tactic: int = 5,
        diversity_weight: float = 0.3,
    ) -> List[BeamEntry]:
        """
        Expand beam by branching on top-K tactics.
        
        For each beam entry:
          1. Predict top-K next tactics
          2. For each tactic, get top-N techniques
          3. Create new beam entries (branching)
          4. Apply diversity penalty
          5. Keep top beam_width entries globally
        
        Args:
            beam: Current beam
            group_context: Optional attacker context for priors
            branching_factor: Number of tactics to branch on (default: 3)
            techniques_per_tactic: Number of techniques per tactic (default: 5)
            diversity_weight: Weight for diversity penalty (default: 0.3)
        
        Returns:
            Expanded beam with tactical branches
        """
        new_beam = []
        existing_paths = [entry.path for entry in beam]
        
        for entry in beam:
            # Get top-K next tactics
            tactic_probs = self.predictor.predict_next_tactics(
                entry.path,
                top_k=branching_factor,
                group_context=group_context
            )
            
            # For each tactic, get top-N techniques
            for tactic, tactic_prob in tactic_probs:
                tech_probs = self.predictor.next_probabilities_for_tactic(
                    entry.path,
                    target_tactic=tactic,
                    top_n=techniques_per_tactic,
                    group_context=group_context
                )
                
                # Create branch for this tactic direction
                for tech, tech_prob, components in tech_probs:
                    # Avoid cycles
                    if tech in entry.path:
                        continue
                    
                    if tech_prob <= 0:
                        continue
                    
                    # Create new path
                    new_path = entry.path + [tech]
                    
                    # Joint probability: P(tactic) * P(tech | tactic)
                    joint_prob = tactic_prob * tech_prob
                    new_log_prob = entry.log_probability + math.log(joint_prob)
                    
                    # Compute diversity bonus
                    diversity_bonus = self._compute_diversity_bonus(
                        new_path,
                        existing_paths,
                        diversity_weight
                    )
                    
                    # Adjusted score with diversity
                    adjusted_log_prob = new_log_prob + diversity_bonus
                    
                    # Update or create branch metadata
                    if entry.branch:
                        # Continue existing branch
                        tactic_sequence = entry.branch.tactic_sequence + [tactic]
                        branch_id = entry.branch.branch_id
                    else:
                        # Start new branch
                        tactic_sequence = [tactic]
                        branch_id = f"branch_{tactic}"
                    
                    branch = TacticBranch(
                        branch_id=branch_id,
                        primary_tactic=tactic,
                        branch_probability=tactic_prob,
                        tactic_sequence=tactic_sequence
                    )
                    
                    # Create new components with tactic info
                    new_components = entry.per_step_components + [{
                        'step': len(new_path),
                        'tech': tech,
                        'tactic': tactic,
                        'tactic_prob': tactic_prob,
                        'tech_prob': tech_prob,
                        'joint_prob': joint_prob,
                        'diversity_bonus': diversity_bonus,
                        'P_count': components.get('P_count', 0.0),
                        'P_emb': components.get('P_emb', 0.0),
                        'M_emb': components.get('M_emb', 1.0),
                        'S': components.get('stealth_prior', 1.0),
                        'prob': joint_prob
                    }]
                    
                    # Create new beam entry (use adjusted score for sorting)
                    new_entry = BeamEntry(new_path, adjusted_log_prob, new_components, branch)
                    new_entry._original_log_prob = new_log_prob  # Store original for output
                    new_beam.append(new_entry)
        
        # Sort by adjusted probability and keep top beam_width
        new_beam.sort()
        pruned_beam = new_beam[:self.beam_width]
        
        # Restore original probabilities for final scoring
        for entry in pruned_beam:
            if hasattr(entry, '_original_log_prob'):
                entry.log_probability = entry._original_log_prob
        
        return pruned_beam
    
    def search_paths(
        self,
        seed_history: List[str],
        group_context: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for top-K attack paths.
        
        Args:
            seed_history: Seed history (starting techniques)
            group_context: Optional attacker context for priors
            
        Returns:
            List of top-K paths with metadata
        """
        print(f"Starting beam search with seed: {seed_history}")
        
        # Initialize beam with seed
        initial_components = []
        for i, tech in enumerate(seed_history):
            initial_components.append({
                'step': i + 1,
                'tech': tech,
                'p_count': 1.0,  # Dummy values for seed
                'p_emb': 1.0,
                'p_interp': 1.0,
                'p_prior': 1.0,
                'm_emb': 1.0,
                's_stealth': 1.0,
                'stealth': 1.0,
                'S': 1.0,
                'omega': 1.0,
                'support': None,
                'prob': 1.0,
                'probability': 1.0
            })

        beam = [BeamEntry(seed_history, 0.0, initial_components)]
        
        # Expand beam for each depth
        for depth in tqdm(range(len(seed_history), self.max_depth), desc="Beam search"):
            beam = self.expand_beam(beam, group_context)
            
            if not beam:
                print(f"No valid paths found at depth {depth}")
                break
        
        # Convert to output format
        paths = []
        for i, entry in enumerate(beam[:self.top_k_paths]):
            path_data = self.create_path_output(entry, i)
            paths.append(path_data)
        
        print(f"Found {len(paths)} paths")
        return paths
    
    def create_path_output(self, entry: BeamEntry, path_id: int) -> Dict[str, Any]:
        """
        Create output dictionary for a path.
        
        Args:
            entry: Beam entry
            path_id: Path identifier
            
        Returns:
            Path output dictionary
        """
        # Compute path statistics
        log_p_seq = entry.log_probability
        try:
            p_seq = math.exp(log_p_seq)
        except OverflowError:
            p_seq = float('inf')

        # Compute PS cumulative (mean of PS scores)
        ps_scores = [self.predictor.ps_scores.get(tech, 1.0) or 1.0 for tech in entry.path]
        ps_cum = np.mean(ps_scores) if ps_scores else 1.0

        # Compute stealth cumulative (product of stealth scores)
        stealth_scores = [self.predictor.sigma_scores.get(tech, 1.0) or 1.0 for tech in entry.path]
        s_cum = np.prod(stealth_scores) if stealth_scores else 1.0

        # Compute embedding multiplier cumulative
        m_emb_scores = [(comp.get('m_emb') or 1.0) for comp in entry.per_step_components]
        m_emb_cum = np.prod(m_emb_scores) if m_emb_scores else 1.0

        # Compute support sum
        support_sum = sum((comp.get('support') or 0.0) for comp in entry.per_step_components)
        
        # Find example campaigns (simplified - in practice, would look up in sequences)
        example_campaigns = [f"c_{np.random.randint(10000, 99999)}" for _ in range(min(5, len(entry.path)))]
        
        example_campaigns = []
        if self.sequences_str:
            path_str = " " + " ".join(entry.path) + " "
            found_count = 0
            for i, seq_str in enumerate(self.sequences_str):
                if path_str in seq_str:
                    example_campaigns.append(f"c_{i}") # Use index as campaign ID
                    found_count += 1
                    if found_count >= 5:
                        break
        return {
            'path_id': f"p{path_id:04d}",
            'sequence': entry.path,
            'score': p_seq,
            'log_score': log_p_seq,
            'P_seq': p_seq,
            'PS_cum': ps_cum,
            'S_cum': s_cum,
            'M_emb_cum': m_emb_cum,
            'support_sum': support_sum,
            'per_step': entry.per_step_components,
            'example_campaigns': []
        }
    
    def search_paths_with_tactic_branching(
        self,
        seed_history: List[str],
        group_context: Optional[Any] = None,
        branching_factor: int = 3,
        techniques_per_tactic: int = 5,
        diversity_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Search for top-K attack paths using multi-tactic branching.
        
        Explores multiple tactical directions simultaneously by branching on
        top-K tactics at each step, generating diverse attack paths.
        
        Args:
            seed_history: Seed history (starting techniques)
            group_context: Optional attacker context for priors
            branching_factor: Number of tactics to branch on (default: 3)
            techniques_per_tactic: Number of techniques per tactic (default: 5)
            diversity_weight: Weight for diversity penalty (default: 0.3)
            
        Returns:
            List of top-K paths with branch metadata
        """
        print(f"Starting multi-tactic branching search with seed: {seed_history}")
        print(f"Branching factor: {branching_factor}, Techniques per tactic: {techniques_per_tactic}")
        
        # Initialize beam with seed
        initial_components = []
        for i, tech in enumerate(seed_history):
            initial_components.append({
                'step': i + 1,
                'tech': tech,
                'tactic': None,  # Will be filled if available
                'p_count': 1.0,
                'p_emb': 1.0,
                'p_interp': 1.0,
                'p_prior': 1.0,
                'm_emb': 1.0,
                's_stealth': 1.0,
                'stealth': 1.0,
                'S': 1.0,
                'omega': 1.0,
                'support': None,
                'prob': 1.0,
                'probability': 1.0,
                'tactic_prob': None,
                'tech_prob': None,
                'joint_prob': None,
                'diversity_bonus': 0.0,
            })

        beam = [BeamEntry(seed_history, 0.0, initial_components, branch=None)]
        
        # Expand beam for each depth using tactic branching
        for depth in tqdm(range(len(seed_history), self.max_depth), desc="Multi-tactic beam search"):
            beam = self.expand_with_tactic_branching(
                beam,
                group_context=group_context,
                branching_factor=branching_factor,
                techniques_per_tactic=techniques_per_tactic,
                diversity_weight=diversity_weight,
            )
            
            if not beam:
                print(f"No valid paths found at depth {depth}")
                break
        
        # Convert to output format with branch metadata
        paths = []
        for i, entry in enumerate(beam[:self.top_k_paths]):
            path_data = self.create_path_output_with_branches(entry, i)
            paths.append(path_data)
        
        print(f"Found {len(paths)} paths across {len(self.group_paths_by_tactic(paths))} tactical branches")
        return paths
    
    def create_path_output_with_branches(self, entry: BeamEntry, path_id: int) -> Dict[str, Any]:
        """
        Create output dictionary for a path with branch metadata.
        
        Args:
            entry: Beam entry with branch information
            path_id: Path identifier
            
        Returns:
            Path output dictionary with tactical branch metadata
        """
        # Get base path data
        base_output = self.create_path_output(entry, path_id)
        
        # Add branch metadata
        if entry.branch:
            base_output['branch_id'] = entry.branch.branch_id
            base_output['primary_tactic'] = entry.branch.primary_tactic
            base_output['branch_probability'] = entry.branch.branch_probability
            base_output['tactic_sequence'] = entry.branch.tactic_sequence
            base_output['dominant_tactic'] = self._get_dominant_tactic(entry.branch.tactic_sequence)
        else:
            base_output['branch_id'] = None
            base_output['primary_tactic'] = None
            base_output['branch_probability'] = None
            base_output['tactic_sequence'] = []
            base_output['dominant_tactic'] = None
        
        return base_output
    
    def _get_dominant_tactic(self, tactic_sequence: List[str]) -> str:
        """
        Get the most frequent tactic in a sequence.
        
        Args:
            tactic_sequence: List of tactics in the path
        
        Returns:
            Most common tactic, or first tactic if tie
        """
        if not tactic_sequence:
            return "unknown"
        
        from collections import Counter
        tactic_counts = Counter(tactic_sequence)
        return tactic_counts.most_common(1)[0][0]
    
    def group_paths_by_tactic(self, paths: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group paths by their dominant tactic.
        
        Clusters paths based on their tactical focus, enabling visualization
        of different attack strategies (e.g., persistence-focused vs lateral-movement-focused).
        
        Args:
            paths: List of path dictionaries with branch metadata
        
        Returns:
            Dictionary mapping tactic names to lists of paths
        """
        grouped = defaultdict(list)
        
        for path in paths:
            dominant_tactic = path.get('dominant_tactic', 'unknown')
            grouped[dominant_tactic].append(path)
        
        return dict(grouped)
    
    def search_multiple_seeds(
        self,
        seed_histories: List[List[str]],
        group_context: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search paths for multiple seed histories.
        
        Args:
            seed_histories: List of seed histories
            group_context: Optional attacker context for priors
            
        Returns:
            Combined list of top-K paths
        """
        all_paths = []
        
        for seed in tqdm(seed_histories, desc="Processing seeds"):
            paths = self.search_paths(seed, group_context)
            all_paths.extend(paths)
        
        # Sort all paths by score and return top-K
        all_paths.sort(key=lambda x: x['score'], reverse=True)
        return all_paths[:self.top_k_paths]


def run_beam_search(predictor: Predictor, output_file: str = "outputs/paths_topK.json",
                    beam_width: int = 200, max_depth: int = 6, top_k: int = 100,
                    custom_seeds_str: Optional[str] = None, all_sequences: Optional[List[List[str]]] = None) -> List[Dict[str, Any]]:
    """
    Run beam search and save results.
    
    Args:
        predictor: Chain-aware predictor
        output_file: Output file path
        beam_width: Beam width
        max_depth: Maximum depth
        top_k: Number of top paths
        custom_seeds_str: A string of custom seeds, e.g., "T1000,T1001;T1002"
        all_sequences: All sequences from the dataset for finding examples.
        
    Returns:
        List of top-K paths
    """
    print("Running beam search...")
    
    # Initialize beam search
    beam_search = BeamSearch(predictor, beam_width, max_depth, top_k, all_sequences=all_sequences)
    
    if custom_seeds_str:
        print(f"Using custom seeds: {custom_seeds_str}")
        # Parse the custom seeds string
        # Example: "T1000,T1001;T1002" -> [['T1000', 'T1001'], ['T1002']]
        seed_histories = [
            path.split(',') for path in custom_seeds_str.split(';')
        ]
    else:
        print("Using default seed techniques.")
        # Define default seed histories (common starting techniques)
        seed_histories = [
            ['T1059.001'],  # PowerShell
            ['T1105'],      # Ingress Tool Transfer
            ['T1053.005'],  # Scheduled Task
            ['T1566.001'],  # Phishing: Spearphishing Attachment
            ['T1071.001'],  # Application Layer Protocol: Web Protocols
            ['T1059.001', 'T1105'],  # PowerShell + Ingress Tool Transfer
            ['T1566.001', 'T1059.001'],  # Phishing + PowerShell
        ]
    
    # Search paths
    paths = beam_search.search_multiple_seeds(seed_histories)
    
    # Save results
    from pathlib import Path
    ensure_dir(Path(output_file).parent)
    save_paths_json(paths, output_file)
    
    print(f"Saved {len(paths)} paths to {output_file}")
    
    return paths


def run_multi_tactic_beam_search(
    predictor: Predictor,
    output_file: str = "outputs/paths_multi_tactic.json",
    beam_width: int = 200,
    max_depth: int = 6,
    top_k: int = 100,
    branching_factor: int = 3,
    techniques_per_tactic: int = 5,
    diversity_weight: float = 0.3,
    custom_seeds_str: Optional[str] = None,
    all_sequences: Optional[List[List[str]]] = None
) -> Dict[str, Any]:
    """
    Run multi-tactic branching beam search and save results.
    
    Generates diverse attack paths by exploring multiple tactical directions
    simultaneously. Each path is labeled with its tactical focus.
    
    Args:
        predictor: Chain-aware predictor
        output_file: Output file path
        beam_width: Beam width (200 total, dynamically allocated)
        max_depth: Maximum path depth
        top_k: Number of top paths to return
        branching_factor: Number of tactics to branch on (default: 3)
        techniques_per_tactic: Number of techniques per tactic (default: 5)
        diversity_weight: Weight for diversity penalty (default: 0.3)
        custom_seeds_str: A string of custom seeds, e.g., "T1000,T1001;T1002"
        all_sequences: All sequences from the dataset for finding examples
        
    Returns:
        Dictionary with paths and tactical grouping
    """
    print("Running multi-tactic branching beam search...")
    print(f"Configuration: branching_factor={branching_factor}, techniques_per_tactic={techniques_per_tactic}")
    print(f"Diversity weight: {diversity_weight}")
    
    # Initialize beam search
    beam_search = BeamSearch(predictor, beam_width, max_depth, top_k, all_sequences=all_sequences)
    
    if custom_seeds_str:
        print(f"Using custom seeds: {custom_seeds_str}")
        seed_histories = [
            path.split(',') for path in custom_seeds_str.split(';')
        ]
    else:
        print("Using default seed techniques.")
        seed_histories = [
            ['T1566.001'],  # Phishing: Spearphishing Attachment
            ['T1059.001'],  # PowerShell
            ['T1105'],      # Ingress Tool Transfer
        ]
    
    # Search paths with tactic branching
    all_paths = []
    for seed in tqdm(seed_histories, desc="Processing seeds"):
        paths = beam_search.search_paths_with_tactic_branching(
            seed,
            group_context=None,
            branching_factor=branching_factor,
            techniques_per_tactic=techniques_per_tactic,
            diversity_weight=diversity_weight,
        )
        all_paths.extend(paths)
    
    # Sort all paths by score and keep top-K
    all_paths.sort(key=lambda x: x['score'], reverse=True)
    top_paths = all_paths[:top_k]
    
    # Group by tactic
    grouped_by_tactic = beam_search.group_paths_by_tactic(top_paths)
    
    # Create structured output
    output = {
        'metadata': {
            'total_paths': len(top_paths),
            'branching_factor': branching_factor,
            'techniques_per_tactic': techniques_per_tactic,
            'diversity_weight': diversity_weight,
            'beam_width': beam_width,
            'max_depth': max_depth,
            'num_tactical_branches': len(grouped_by_tactic),
        },
        'paths': top_paths,
        'branches': [
            {
                'tactic': tactic,
                'num_paths': len(paths),
                'paths': paths
            }
            for tactic, paths in grouped_by_tactic.items()
        ]
    }
    
    # Save results
    from pathlib import Path
    import json
    ensure_dir(Path(output_file).parent)
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nMulti-tactic branching complete!")
    print(f"Generated {len(top_paths)} paths across {len(grouped_by_tactic)} tactical branches:")
    for tactic, paths in grouped_by_tactic.items():
        print(f"  - {tactic}: {len(paths)} paths")
    print(f"\nSaved results to {output_file}")
    
    return output


def main():
    """Main function to run beam search."""
    from .predictor import load_predictor
    
    # Load predictor
    predictor = load_predictor()
    
    # Run beam search
    paths = run_beam_search(predictor)
    
    print(f"Beam search complete. Generated {len(paths)} paths.")


if __name__ == "__main__":
    main()
