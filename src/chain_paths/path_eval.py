"""Path-level evaluation for predicted attack chains.

Compares predicted multi-step attack paths against ground-truth test sequences
to measure next-j step accuracy, tactic coherence, and tactical diversity.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import config as cfg


def _normalize_tid(value: str) -> str:
    return str(value).strip().upper()


class PathEvaluator:
    """Evaluate predicted attack paths against ground-truth sequences."""

    def __init__(
        self,
        predicted_paths: List[Dict[str, Any]],
        test_sequences: Sequence[Sequence[str]],
        technique_to_tactics: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize path evaluator.
        
        Args:
            predicted_paths: List of predicted path dictionaries with 'techniques' key
            test_sequences: Ground-truth test sequences
            technique_to_tactics: Mapping from technique IDs to tactics
        """
        self.predicted_paths = predicted_paths
        self.test_sequences = [list(seq) for seq in test_sequences]
        self.technique_to_tactics = technique_to_tactics or {}
        
        # Extract predicted technique sequences
        self.predicted_sequences = []
        for path in predicted_paths:
            techniques = path.get('techniques', [])
            if techniques:
                self.predicted_sequences.append([_normalize_tid(t) for t in techniques])

    def compute_next_j_metrics(
        self,
        j_values: Optional[Sequence[int]] = None,
        k_values: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Compute next-j step prediction accuracy metrics.
        
        For each predicted path, finds matching test sequences with the same prefix,
        then measures how many of the next j techniques appear in the predicted path.
        
        Args:
            j_values: Lookahead steps to evaluate (default: [1, 2, 3, 5])
            k_values: Top-k values for Hit@k metric (default: [1, 3, 5, 10])
        
        Returns:
            Dictionary with next-j metrics for each j and k
        """
        if j_values is None:
            j_values = [1, 2, 3, 5]
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        metrics = {
            'next_j_metrics': {},
            'metadata': {
                'num_predicted_paths': len(self.predicted_sequences),
                'num_test_sequences': len(self.test_sequences),
                'j_values': list(j_values),
                'k_values': list(k_values),
            }
        }
        
        for j in j_values:
            j_metrics = self._compute_for_j(j, k_values)
            metrics['next_j_metrics'][f'j={j}'] = j_metrics
        
        return metrics

    def _compute_for_j(self, j: int, k_values: Sequence[int]) -> Dict[str, Any]:
        """Compute metrics for a specific j-step lookahead."""
        
        hit_counts = {k: 0 for k in k_values}
        precision_scores = []
        recall_scores = []
        total_evaluated = 0
        
        for pred_path in self.predicted_sequences:
            if len(pred_path) < 2:
                continue
            
            # Extract seed (first technique) and predicted continuation
            seed = pred_path[0]
            predicted_next_j = set(pred_path[1:min(len(pred_path), 1 + j)])
            
            # Find test sequences starting with this seed
            matching_tests = []
            for test_seq in self.test_sequences:
                if len(test_seq) > 0 and _normalize_tid(test_seq[0]) == seed:
                    matching_tests.append(test_seq)
            
            if not matching_tests:
                continue
            
            # For each matching test, check if predicted path captures next j steps
            for test_seq in matching_tests:
                if len(test_seq) < 2:
                    continue
                
                # Get actual next j techniques from test
                actual_next_j = set(_normalize_tid(t) for t in test_seq[1:min(len(test_seq), 1 + j)])
                
                if not actual_next_j:
                    continue
                
                total_evaluated += 1
                
                # Compute overlap
                overlap = predicted_next_j & actual_next_j
                
                # Precision: what fraction of predicted are correct
                precision = len(overlap) / len(predicted_next_j) if predicted_next_j else 0.0
                precision_scores.append(precision)
                
                # Recall: what fraction of actual are predicted
                recall = len(overlap) / len(actual_next_j) if actual_next_j else 0.0
                recall_scores.append(recall)
                
                # Hit@k: did we predict at least one correct next technique
                for k in k_values:
                    predicted_top_k = set(pred_path[1:min(len(pred_path), 1 + k)])
                    if predicted_top_k & actual_next_j:
                        hit_counts[k] += 1
        
        if total_evaluated == 0:
            return {
                'total_evaluated': 0,
                'precision': {'mean': 0.0, 'std': 0.0},
                'recall': {'mean': 0.0, 'std': 0.0},
                'hit_at_k': {k: 0.0 for k in k_values},
            }
        
        return {
            'total_evaluated': total_evaluated,
            'precision': {
                'mean': float(np.mean(precision_scores)),
                'std': float(np.std(precision_scores)),
            },
            'recall': {
                'mean': float(np.mean(recall_scores)),
                'std': float(np.std(recall_scores)),
            },
            'hit_at_k': {
                k: hit_counts[k] / total_evaluated for k in k_values
            },
        }

    def analyze_tactic_phases(self) -> Dict[str, Any]:
        """Analyze multi-tactic phases and identify dominant tactical patterns.
        
        Clusters predicted paths by their tactic sequence patterns and identifies
        the top tactical phases represented in the predictions.
        
        Returns:
            Dictionary with tactic phase analysis
        """
        if not self.technique_to_tactics:
            return {
                'error': 'No tactic mapping available',
                'tactic_sequences': [],
                'top_phases': [],
            }
        
        # Extract tactic sequences from predicted paths
        tactic_sequences = []
        for pred_path in self.predicted_sequences:
            tactic_seq = []
            for tech in pred_path:
                tactics = self.technique_to_tactics.get(tech, [])
                if tactics:
                    # Use primary tactic (first in list)
                    tactic_seq.append(tactics[0])
            
            if tactic_seq:
                tactic_sequences.append(tuple(tactic_seq))
        
        # Count tactic sequence patterns
        sequence_counts = Counter(tactic_sequences)
        
        # Identify top-2 tactical phases
        top_phases = []
        for tactic_seq, count in sequence_counts.most_common(2):
            phase_repr = ' -> '.join(tactic_seq)
            top_phases.append({
                'phase_pattern': phase_repr,
                'tactic_sequence': list(tactic_seq),
                'count': count,
                'fraction': count / len(tactic_sequences) if tactic_sequences else 0.0,
            })
        
        # Compute tactic diversity (entropy)
        tactic_diversity = 0.0
        if sequence_counts:
            total = sum(sequence_counts.values())
            for count in sequence_counts.values():
                p = count / total
                if p > 0:
                    tactic_diversity -= p * np.log2(p)
        
        return {
            'num_unique_patterns': len(sequence_counts),
            'tactic_diversity_bits': float(tactic_diversity),
            'top_phases': top_phases,
            'all_patterns': [
                {
                    'pattern': ' -> '.join(seq),
                    'count': count,
                }
                for seq, count in sequence_counts.most_common(10)
            ],
        }

    def generate_report(
        self,
        j_values: Optional[Sequence[int]] = None,
        k_values: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive path evaluation report.
        
        Args:
            j_values: Lookahead steps to evaluate
            k_values: Top-k values for Hit@k metric
        
        Returns:
            Complete evaluation report
        """
        next_j_metrics = self.compute_next_j_metrics(j_values, k_values)
        tactic_analysis = self.analyze_tactic_phases()
        
        return {
            'path_evaluation': {
                'next_j_accuracy': next_j_metrics,
                'tactic_phase_analysis': tactic_analysis,
            },
            'summary': {
                'num_predicted_paths': len(self.predicted_sequences),
                'num_test_sequences': len(self.test_sequences),
                'avg_predicted_path_length': float(np.mean([len(p) for p in self.predicted_sequences])) if self.predicted_sequences else 0.0,
                'avg_test_sequence_length': float(np.mean([len(s) for s in self.test_sequences])) if self.test_sequences else 0.0,
            }
        }


def evaluate_predicted_paths(
    predicted_paths_file: Path,
    test_sequences: Sequence[Sequence[str]],
    technique_to_tactics: Optional[Dict[str, List[str]]] = None,
    output_file: Optional[Path] = None,
    j_values: Optional[Sequence[int]] = None,
    k_values: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Evaluate predicted paths from JSON file.
    
    Args:
        predicted_paths_file: Path to JSON file with predicted paths
        test_sequences: Ground-truth test sequences
        technique_to_tactics: Mapping from technique IDs to tactics
        output_file: Optional path to save evaluation report
        j_values: Lookahead steps to evaluate
        k_values: Top-k values for Hit@k metric
    
    Returns:
        Evaluation report dictionary
    """
    # Load predicted paths
    with open(predicted_paths_file, 'r', encoding='utf-8') as f:
        prediction_data = json.load(f)
    
    # Extract paths from the nested structure
    predicted_paths = []
    if 'paths' in prediction_data and isinstance(prediction_data['paths'], dict):
        for branch_paths in prediction_data['paths'].values():
            if isinstance(branch_paths, list):
                predicted_paths.extend(branch_paths)
    elif isinstance(prediction_data, list):
        predicted_paths = prediction_data
    
    # Create evaluator
    evaluator = PathEvaluator(
        predicted_paths=predicted_paths,
        test_sequences=test_sequences,
        technique_to_tactics=technique_to_tactics,
    )
    
    # Generate report
    report = evaluator.generate_report(j_values, k_values)
    
    # Add metadata
    report['metadata'] = {
        'predicted_paths_file': str(predicted_paths_file),
        'num_test_sequences': len(test_sequences),
    }
    
    # Save if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"Path evaluation report saved to {output_path}")
    
    return report


__all__ = ["PathEvaluator", "evaluate_predicted_paths"]
