"""
Parallel processing utilities for multicore evaluation and prediction.

This module provides worker-safe multiprocessing for CPU-intensive operations.
"""

import multiprocessing as mp
from typing import List, Dict, Any, Callable, Optional
from functools import partial
import sys


class ParallelExecutor:
    """
    Manages parallel execution with worker-safe model initialization.
    
    Each worker maintains its own predictor instance to avoid pickle issues.
    Uses multiprocessing.Pool for true parallelism (bypasses Python GIL).
    """
    
    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize parallel executor.
        
        Args:
            n_workers: Number of worker processes. None = auto (cpu_count - 1)
        """
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)
        elif n_workers <= 0:
            n_workers = mp.cpu_count()
        
        self.n_workers = min(n_workers, mp.cpu_count())
        print(f"ParallelExecutor: Using {self.n_workers} worker processes")
    
    def map(self, func: Callable, items: List[Any], chunksize: Optional[int] = None) -> List[Any]:
        """
        Execute function in parallel across items.
        
        Args:
            func: Worker function (must be picklable)
            items: List of items to process
            chunksize: Items per worker batch (None = auto)
            
        Returns:
            List of results (same order as items)
        """
        if not items:
            return []
        
        if self.n_workers == 1:
            # Sequential fallback
            return [func(item) for item in items]
        
        # Auto-calculate chunksize for better load balancing
        if chunksize is None:
            chunksize = max(1, len(items) // (self.n_workers * 4))
        
        try:
            with mp.Pool(processes=self.n_workers) as pool:
                results = pool.map(func, items, chunksize=chunksize)
            return results
        except Exception as e:
            print(f"⚠️  Parallel execution failed: {e}", file=sys.stderr)
            print("⚠️  Falling back to sequential processing...", file=sys.stderr)
            return [func(item) for item in items]


# Global worker state (initialized once per worker process)
_worker_predictor = None


def _init_worker_predictor(checkpoint_path: str, mcdm_scores_path: Optional[str] = None):
    """Initialize predictor in worker process (called once per worker)."""
    global _worker_predictor
    
    if _worker_predictor is None:
        # Force CPU usage in worker processes to avoid CUDA fork issues
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        try:
            import torch
            torch.set_num_threads(1)  # Avoid thread contention in workers
        except ImportError:
            pass
        
        from .predictor import load_predictor

        _worker_predictor = load_predictor(checkpoint_path=checkpoint_path)
        _worker_predictor._checkpoint_path = checkpoint_path
        
        # Move model to CPU explicitly if it has a torch model
        if hasattr(_worker_predictor, 'base_model') and _worker_predictor.base_model is not None:
            base = _worker_predictor.base_model
            if hasattr(base, 'model') and base.model is not None:
                try:
                    base.model = base.model.cpu()
                    base.model.eval()
                except Exception:
                    pass


def _evaluate_single_sequence_worker(args: tuple) -> Dict[str, Any]:
    """
    Worker function to evaluate a single sequence.
    
    Args:
        args: (seq_idx, sequence, context, k_values, use_scenario_priors)
        
    Returns:
        Dictionary with evaluation results for this sequence
    """
    seq_idx, sequence, context, k_values, use_scenario_priors = args
    
    global _worker_predictor
    if _worker_predictor is None:
        raise RuntimeError("Worker predictor not initialized")
    
    import numpy as np
    
    # Initialize result containers
    result = {
        'seq_idx': seq_idx,
        'observations': 0,
        'predictions_generated': 0,
        'predictions_empty': 0,
        'model_precisions': {k: [] for k in k_values},
        'baseline_precisions': {k: [] for k in k_values},
        'mrr_model': [],
        'mrr_baseline': [],
    }
    
    if len(sequence) < 2:
        return result
    
    # Evaluate each prediction step
    for i in range(1, len(sequence)):
        history = sequence[:i]
        target = sequence[i]
        
        predictions = _worker_predictor.next_probabilities(
            history,
            group_context=context,
            top_n_candidates=100,
            use_scenario_priors=use_scenario_priors,
        )
        
        if not predictions:
            result['predictions_empty'] += 1
            continue
        
        result['predictions_generated'] += 1
        
        candidate_names = [pred[0] for pred in predictions]
        predicted_techs = candidate_names
        
        # Baseline: count-only ordering
        # Handle both old architecture (P_count) and new architecture (P_bilstm, P_lstm, etc.)
        count_scores = []
        for pred in predictions:
            components = pred[2]
            # Try to get count-based score from various possible keys
            score = (
                components.get('P_count') or 
                components.get('P_bilstm') or 
                components.get('P_lstm') or
                components.get('P_gru') or
                components.get('P_ngram') or
                components.get('P_hmm') or
                1.0 / len(predictions)  # Uniform fallback
            )
            count_scores.append(float(score))
        
        count_scores = np.array(count_scores, dtype=float)
        if count_scores.sum() > 0:
            baseline_probs = count_scores / count_scores.sum()
        else:
            baseline_probs = np.ones_like(count_scores) / len(count_scores)
        baseline_order_idx = np.argsort(-baseline_probs)
        baseline_order = [candidate_names[idx] for idx in baseline_order_idx]
        
        # Compute precision@k
        for k in k_values:
            if k <= len(predicted_techs):
                model_hit = 1.0 if target in predicted_techs[:k] else 0.0
                baseline_hit = 1.0 if target in baseline_order[:k] else 0.0
                result['model_precisions'][k].append(model_hit)
                result['baseline_precisions'][k].append(baseline_hit)
        
        # Compute MRR
        if target in predicted_techs:
            rank = predicted_techs.index(target) + 1
            result['mrr_model'].append(1.0 / rank)
        else:
            result['mrr_model'].append(0.0)
        
        if target in baseline_order:
            baseline_rank = baseline_order.index(target) + 1
            result['mrr_baseline'].append(1.0 / baseline_rank)
        else:
            result['mrr_baseline'].append(0.0)
        
        result['observations'] += 1
    
    return result


def _evaluate_single_path_worker(args: tuple) -> Dict[str, Any]:
    """
    Worker function to evaluate a single path using beam search.
    
    Args:
        args: (seq_idx, sequence, context, k_values, beam_width, max_depth, top_k_paths)
        
    Returns:
        Dictionary with hit results for each k value
    """
    seq_idx, sequence, context, k_values, beam_width, max_depth, top_k_paths = args
    
    global _worker_predictor
    if _worker_predictor is None:
        raise RuntimeError("Worker predictor not initialized")
    
    from .beam_search import BeamSearch
    
    result = {
        'seq_idx': seq_idx,
        'hits': {k: 0.0 for k in k_values},
        'valid': False,
    }
    
    # Skip sequences that are too short
    if len(sequence) < 3:
        return result
    
    # Use first 1-3 techniques as seed
    seed_length = min(3, len(sequence) - 1)
    seed = sequence[:seed_length]
    target_path = sequence[seed_length:]
    
    if not target_path:
        return result
    
    # Get predictions using beam search
    beam_search = BeamSearch(_worker_predictor, beam_width=beam_width, max_depth=max_depth, top_k_paths=top_k_paths)
    
    try:
        predicted_paths = beam_search.search_paths(seed, group_context=context)
    except Exception as e:
        # Beam search failed - return zeros
        return result
    
    result['valid'] = True
    
    # Check if target path appears in predictions
    for k in k_values:
        if k <= len(predicted_paths):
            hit = False
            for pred_path in predicted_paths[:k]:
                pred_sequence = pred_path['sequence']
                # Check if target path is a subsequence of predicted path
                if _is_subsequence(target_path, pred_sequence):
                    hit = True
                    break
            
            result['hits'][k] = 1.0 if hit else 0.0
    
    return result


def _is_subsequence(subseq: List[str], seq: List[str]) -> bool:
    """Check if subseq is a subsequence of seq."""
    if not subseq:
        return True
    if len(subseq) > len(seq):
        return False
    
    sub_idx = 0
    for item in seq:
        if item == subseq[sub_idx]:
            sub_idx += 1
            if sub_idx >= len(subseq):
                return True
    return False
