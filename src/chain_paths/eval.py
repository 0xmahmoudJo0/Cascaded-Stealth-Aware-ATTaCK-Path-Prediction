"""
Evaluation module for the chain-aware predictor.

This module implements evaluation metrics for next-step prediction and path-level evaluation.
"""

import os
import random
from contextlib import contextmanager
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config as cfg
from .predictor import Predictor
from .io import save_json, ensure_dir
from .beam_search import BeamSearch


def _bootstrap_confidence_interval(samples: List[float], n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    if not samples:
        return 0.0, 0.0

    data = np.array(samples, dtype=np.float64)
    if len(data) == 1:
        return float(data[0]), float(data[0])

    idx = np.random.randint(0, len(data), size=(n_bootstrap, len(data)))
    boot_means = data[idx].mean(axis=1)
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def _paired_bootstrap_difference(
    base_samples: List[float],
    variant_samples: List[float],
    n_bootstrap: int = 1000,
) -> Optional[Dict[str, float]]:
    if not base_samples or not variant_samples:
        return None

    base = np.array(base_samples, dtype=np.float64)
    variant = np.array(variant_samples, dtype=np.float64)

    n = min(len(base), len(variant))
    if n == 0:
        return None

    base = base[:n]
    variant = variant[:n]
    diffs = base - variant

    if np.allclose(diffs, 0):
        return {'mean_diff': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0, 'p_value': 1.0}

    idx = np.random.randint(0, n, size=(n_bootstrap, n))
    boot_means = diffs[idx].mean(axis=1)
    mean_diff = float(diffs.mean())
    lower = float(np.percentile(boot_means, 2.5))
    upper = float(np.percentile(boot_means, 97.5))
    p_value = float(np.mean(np.abs(boot_means) >= abs(mean_diff)))

    return {
        'mean_diff': mean_diff,
        'ci_lower': lower,
        'ci_upper': upper,
        'p_value': p_value,
    }


class Evaluator:
    """
    Evaluator for the chain-aware predictor.
    """
    
    def __init__(self, predictor: Predictor, sequences: List[List[str]],
                 sequence_metadata: Optional[pd.DataFrame] = None):
        """
        Initialize evaluator.
        
        Args:
            predictor: Chain-aware predictor
            sequences: List of technique sequences
            sequence_metadata: Optional DataFrame with per-sequence metadata (including temporal fields)
        """
        self.predictor = predictor
        self.sequences = sequences
        self.sequence_metadata = sequence_metadata

        self.sequence_contexts: List[Dict[str, Optional[str]]] = self._build_sequence_contexts(
            sequence_metadata
        )
        self._last_split_contexts: Dict[str, List[Optional[Dict[str, Optional[str]]]]] = {
            'train': [],
            'test': [],
        }
        self._last_split_indices: Dict[str, List[int]] = {'train': [], 'test': []}

        if self.sequence_metadata is not None:
            if 'sequence_id' not in self.sequence_metadata.columns:
                raise ValueError("sequence_metadata must include a 'sequence_id' column")
            if len(self.sequence_metadata) != len(self.sequences):
                print(
                    "Warning: sequence_metadata length does not match number of sequences. "
                    "Entries will be aligned by 'sequence_id'."
                )

        # Track how many sequences are eligible for temporal evaluation
        # Academic Note: Three-way split ensures proper model selection without test-set leakage
        self._temporal_valid_indices: List[int] = list(range(len(self.sequences)))
        self._temporal_train_labels: List[Optional[str]] = []
        self._temporal_val_labels: List[Optional[str]] = []
        self._temporal_test_labels: List[Optional[str]] = []
        self._temporal_train_indices: List[int] = []
        self._temporal_val_indices: List[int] = []  # NEW: validation split for hyperparameter selection
        self._temporal_test_indices: List[int] = []

        # Set random seed for reproducibility
        random.seed(cfg.DEFAULT_PARAMS['RANDOM_SEED'])
        np.random.seed(cfg.DEFAULT_PARAMS['RANDOM_SEED'])

        print(f"Initialized evaluator with {len(sequences)} sequences")

    @contextmanager
    def _temporary_predictor_settings(
        self,
        disable_filters: bool,
        pure_model_eval: bool,
    ):
        prev_params: Dict[str, Tuple[bool, Any]] = {}
        prev_platform_filtering = None
        has_platform_filtering = hasattr(self.predictor, "_use_platform_filtering")

        def _set_param(key: str, value: Any) -> None:
            prev_params[key] = (key in self.predictor.params, self.predictor.params.get(key))
            self.predictor.params[key] = value

        try:
            if disable_filters:
                _set_param('use_tactic_filtering', False)
                if has_platform_filtering:
                    prev_platform_filtering = self.predictor._use_platform_filtering
                    self.predictor._use_platform_filtering = False

            if pure_model_eval:
                for key in ('w_tactic', 'w_stealth'):
                    _set_param(key, 0.0)

            yield
        finally:
            for key, (existed, value) in prev_params.items():
                if existed:
                    self.predictor.params[key] = value
                else:
                    self.predictor.params.pop(key, None)
            if has_platform_filtering and prev_platform_filtering is not None:
                self.predictor._use_platform_filtering = prev_platform_filtering
    
    def split_sequences(self, train_ratio: float = 0.8, temporal_split: bool = True, random_seed: int = None) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Split sequences into train, validation, and test sets.
        
        Academic Note:
        - When temporal_split=True and sequence_metadata contains 'start_time':
          Uses STRICT TEMPORAL ORDERING (no shuffling) to prevent data leakage.
          Split ratios: Train=60-70%, Validation=10-20%, Test=20%
          Ensures max(train_time) < min(val_time) < min(test_time)
        - When temporal_split=False or no timestamps:
          Falls back to random split (legacy behavior for backward compatibility)
        - When train_ratio=0.0:
          Uses all sequences as test (for checkpoint evaluation)
        
        Args:
            train_ratio: Ratio of sequences for training (0.0 = no re-split, use all as test)
            temporal_split: If True and timestamps available, use temporal ordering
            random_seed: Random seed for reproducible random splits
            
        Returns:
            Tuple of (train_sequences, test_sequences)
            Note: Validation sequences stored in self._temporal_val_indices
        """
        import random
        
        n_total = len(self.sequences)

        # Track whether a temporal split was successfully produced
        temporal_done = False

        # Special case: train_ratio=0.0 means use all as test WITHOUT re-shuffling
        # This preserves sequence order when evaluating checkpoint on saved test set
        if train_ratio == 0.0:
            train_indices = []
            val_indices = []
            test_indices = list(range(n_total))  # Keep original order
            temporal_done = True  # Split decided
            print(f"Evaluating on {len(test_indices)} pre-split test sequences (train/val/test split already applied at training time)")

        # TRUE TEMPORAL SPLIT: Sort by start_time and split chronologically
        if (not temporal_done) and temporal_split and self.sequence_metadata is not None and 'start_time' in self.sequence_metadata.columns:
            print("[TEMPORAL SPLIT] Using chronological ordering by start_time...")
            
            # Filter sequences with valid timestamps
            metadata = self.sequence_metadata.copy()
            metadata['start_time_parsed'] = pd.to_datetime(metadata['start_time'], errors='coerce')
            metadata = metadata[metadata['start_time_parsed'].notna()].copy()
            
            if len(metadata) < n_total:
                print(f"  Warning: {n_total - len(metadata)} sequences have invalid timestamps, excluding from temporal split")
            
            if len(metadata) < 3:
                print(f"  Error: Too few sequences with valid timestamps ({len(metadata)}), falling back to random split")
            else:
                # Sort by timestamp (earliest to latest)
                metadata = metadata.sort_values('start_time_parsed')
                sorted_indices = metadata.index.tolist()
                
                # Split: 70% train, 10% val, 20% test (academic standard for temporal data)
                n_valid = len(sorted_indices)
                n_train = int(n_valid * 0.70)
                n_val = int(n_valid * 0.10)
                # Remaining goes to test (ensures at least 20%)
                
                train_indices = sorted_indices[:n_train]
                val_indices = sorted_indices[n_train:n_train + n_val]
                test_indices = sorted_indices[n_train + n_val:]
                
                # Store temporal metadata
                self._temporal_train_indices = train_indices
                self._temporal_val_indices = val_indices
                self._temporal_test_indices = test_indices
                temporal_done = True
                
                # Verify temporal ordering
                train_times = metadata.loc[train_indices, 'start_time_parsed']
                val_times = metadata.loc[val_indices, 'start_time_parsed']
                test_times = metadata.loc[test_indices, 'start_time_parsed']
                
                print(f"  Train: {len(train_indices)} sequences [{train_times.min()} to {train_times.max()}]")
                print(f"  Val:   {len(val_indices)} sequences [{val_times.min()} to {val_times.max()}]")
                print(f"  Test:  {len(test_indices)} sequences [{test_times.min()} to {test_times.max()}]")
                print(f"  ✓ Temporal ordering: train < val < test (no data leakage)")

        # FALLBACK: Random split (legacy behavior) if temporal not completed
        if not temporal_done:
            if temporal_split:
                print("[RANDOM SPLIT] Temporal split unavailable, using random split with train/val/test")
            if random_seed is not None:
                random.seed(random_seed)
            indices = list(range(n_total))
            random.shuffle(indices)
            # 70% train, 10% val, 20% test (same ratios as temporal)
            n_train = int(n_total * 0.70)
            n_val = int(n_total * 0.10)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
            print(f"Split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test (random seed={random_seed})")

        train_sequences = [self.sequences[i] for i in train_indices]
        test_sequences = [self.sequences[i] for i in test_indices]
        self._last_split_contexts = {
            'train': [self.sequence_contexts[i] for i in train_indices],
            'val': [self.sequence_contexts[i] for i in val_indices] if 'val_indices' in locals() else [],
            'test': [self.sequence_contexts[i] for i in test_indices],
        }
        self._last_split_indices = {
            'train': train_indices,
            'val': val_indices if 'val_indices' in locals() else [],
            'test': test_indices
        }

        return train_sequences, test_sequences

    def _normalize_context_value(self, value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        if isinstance(value, (float, int)) and pd.isna(value):
            return None
        cleaned = str(value).strip()
        return cleaned or None

    def _build_sequence_contexts(
        self,
        sequence_metadata: Optional[pd.DataFrame],
    ) -> List[Dict[str, Optional[str]]]:
        metadata_map: Dict[int, Dict[str, Any]] = {}
        if sequence_metadata is not None and not sequence_metadata.empty:
            meta = sequence_metadata.copy()
            if 'sequence_id' not in meta.columns:
                meta['sequence_id'] = np.arange(len(meta))
            meta = meta.sort_values('sequence_id')
            metadata_map = meta.set_index('sequence_id').to_dict('index')

        contexts: List[Dict[str, Optional[str]]] = []
        for idx in range(len(self.sequences)):
            row = metadata_map.get(idx, {})
            phase = self._normalize_context_value(
                row.get('scenario_phase') or row.get('phase')
            )
            persistence = self._normalize_context_value(
                row.get('persistence_focus') or row.get('persistence')
            )
            contexts.append(
                {
                    'scenario_phase': phase,
                    'persistence_focus': persistence,
                }
            )
        return contexts

    def _bootstrap_summary(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {'mean': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}
        arr = np.array(values, dtype=float)
        mean = float(arr.mean())
        if len(arr) < 2:
            return {'mean': mean, 'ci_lower': mean, 'ci_upper': mean}
        rounds = int(self.predictor.params.get('bootstrap_rounds', cfg.DEFAULT_PARAMS.get('bootstrap_rounds', 1000)))
        alpha = float(self.predictor.params.get('significance_alpha', cfg.DEFAULT_PARAMS.get('significance_alpha', 0.05)))
        rng = np.random.default_rng(cfg.DEFAULT_PARAMS['RANDOM_SEED'])
        boot_means = []
        for _ in range(rounds):
            sample = rng.choice(arr, size=len(arr), replace=True)
            boot_means.append(sample.mean())
        lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
        upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return {'mean': mean, 'ci_lower': lower, 'ci_upper': upper}

    def _paired_bootstrap_test(self, model: List[float], baseline: List[float]) -> float:
        if not model or not baseline or len(model) != len(baseline):
            return 1.0
        model_arr = np.array(model, dtype=float)
        baseline_arr = np.array(baseline, dtype=float)
        rounds = int(self.predictor.params.get('bootstrap_rounds', cfg.DEFAULT_PARAMS.get('bootstrap_rounds', 1000)))
        rng = np.random.default_rng(cfg.DEFAULT_PARAMS['RANDOM_SEED'])
        diffs = []
        for _ in range(rounds):
            indices = rng.integers(0, len(model_arr), size=len(model_arr))
            diffs.append((model_arr[indices] - baseline_arr[indices]).mean())
        diffs = np.array(diffs)
        p_value = float((np.sum(diffs <= 0) + 1) / (len(diffs) + 1))
        return p_value
    
    def _compute_imbalance_aware_metrics(
        self,
        predictions_per_class: Dict[str, List[int]],  # class -> list of binary hits (0 or 1)
    ) -> Dict[str, float]:
        """
        Compute imbalance-aware metrics for classification.
        
        Academic Note:
        - Macro metrics: Compute per-class metric, then average (treats all classes equally)
        - Balanced Accuracy: Average of per-class recall (robust to class imbalance)
        - Addresses limitation of micro-averaged accuracy which is dominated by frequent classes
        
        Args:
            predictions_per_class: Dict mapping class_id to list of binary predictions (1=correct, 0=incorrect)
            
        Returns:
            Dict with macro_recall, macro_precision, macro_f1, balanced_accuracy
        """
        from collections import defaultdict
        
        if not predictions_per_class:
            return {
                'macro_recall': 0.0,
                'macro_precision': 0.0,
                'macro_f1': 0.0,
                'balanced_accuracy': 0.0,
            }
        
        per_class_recall = []
        per_class_precision = []
        
        for class_id, hits in predictions_per_class.items():
            if not hits:
                continue
            
            # For next-step prediction: recall = precision = accuracy for binary relevance
            # (each prediction has exactly 1 relevant item)
            class_acc = np.mean(hits)
            per_class_recall.append(class_acc)
            per_class_precision.append(class_acc)
        
        macro_recall = float(np.mean(per_class_recall)) if per_class_recall else 0.0
        macro_precision = float(np.mean(per_class_precision)) if per_class_precision else 0.0
        
        # Macro F1 = harmonic mean of macro precision and macro recall
        if macro_precision + macro_recall > 0:
            macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        else:
            macro_f1 = 0.0
        
        # Balanced accuracy = macro recall (average of per-class recall)
        balanced_accuracy = macro_recall
        
        return {
            'macro_recall': macro_recall,
            'macro_precision': macro_precision,
            'macro_f1': macro_f1,
            'balanced_accuracy': balanced_accuracy,
        }
    
    def _compute_tail_metrics(
        self,
        all_predictions: List[Tuple[str, List[str], int]],  # (target, ranked_predictions, hit_at_1)
        frequency_threshold: int = 100,
    ) -> Dict[str, Any]:
        """
        Compute metrics specifically for tail (rare) techniques.
        
        Academic Note:
        Tail classes are often underrepresented in training and suffer from poor performance.
        Separate tail metrics reveal whether the model can handle rare techniques,
        which is critical for cybersecurity (novel/rare attack techniques).
        
        Args:
            all_predictions: List of (target_technique, ranked_prediction_list, hit_at_1)
            frequency_threshold: Techniques with frequency < this are considered "tail"
            
        Returns:
            Dict with tail_mrr, tail_hit_at_5, tail_hit_at_10, tail_technique_count, tail_prediction_count
        """
        from collections import Counter
        
        # Count target frequencies
        target_freq = Counter([target for target, _, _ in all_predictions])
        tail_techniques = {tech for tech, freq in target_freq.items() if freq < frequency_threshold}
        
        tail_mrr_values = []
        tail_hit_at_5 = []
        tail_hit_at_10 = []
        
        for target, ranked_preds, _ in all_predictions:
            if target not in tail_techniques:
                continue
            
            # Compute MRR
            if target in ranked_preds:
                rank = ranked_preds.index(target) + 1
                tail_mrr_values.append(1.0 / rank)
            else:
                tail_mrr_values.append(0.0)
            
            # Compute Hit@K
            tail_hit_at_5.append(1.0 if target in ranked_preds[:5] else 0.0)
            tail_hit_at_10.append(1.0 if target in ranked_preds[:10] else 0.0)
        
        return {
            'tail_mrr': float(np.mean(tail_mrr_values)) if tail_mrr_values else 0.0,
            'tail_hit_at_5': float(np.mean(tail_hit_at_5)) if tail_hit_at_5 else 0.0,
            'tail_hit_at_10': float(np.mean(tail_hit_at_10)) if tail_hit_at_10 else 0.0,
            'tail_technique_count': len(tail_techniques),
            'tail_prediction_count': len(tail_mrr_values),
            'frequency_threshold': frequency_threshold,
        }
    
    def _compute_frequency_binned_metrics(
        self,
        all_predictions: List[Tuple[str, List[str], int]],  # (target, ranked_predictions, hit_at_1)
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Compute metrics binned by target technique frequency.
        
        Academic Note:
        Frequency-stratified evaluation reveals performance across the frequency spectrum.
        Common pattern: high-frequency techniques have inflated metrics, tail techniques struggle.
        This diagnostic is essential for understanding model biases.
        
        Args:
            all_predictions: List of (target_technique, ranked_prediction_list, hit_at_1)
            n_bins: Number of frequency bins (default: 10 for deciles)
            
        Returns:
            Dict with bins (each bin has: frequency_range, mrr, hit_at_5, count)
        """
        from collections import Counter
        
        # Count target frequencies
        target_freq = Counter([target for target, _, _ in all_predictions])
        
        # Sort techniques by frequency
        sorted_techniques = sorted(target_freq.items(), key=lambda x: x[1])
        
        # Create bins (log-scale for better distribution)
        freq_values = [freq for _, freq in sorted_techniques]
        if not freq_values:
            return {'bins': []}
        
        # Use quantile-based binning for equal-sized bins
        bin_edges = np.percentile(freq_values, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)  # Remove duplicates
        
        bins = []
        for i in range(len(bin_edges) - 1):
            bin_min = bin_edges[i]
            bin_max = bin_edges[i + 1]
            
            # Get techniques in this bin
            bin_techniques = {
                tech for tech, freq in target_freq.items()
                if bin_min <= freq <= bin_max
            }
            
            # Compute metrics for this bin
            bin_mrr = []
            bin_hit_at_5 = []
            bin_count = 0
            
            for target, ranked_preds, _ in all_predictions:
                if target not in bin_techniques:
                    continue
                
                bin_count += 1
                
                if target in ranked_preds:
                    rank = ranked_preds.index(target) + 1
                    bin_mrr.append(1.0 / rank)
                else:
                    bin_mrr.append(0.0)
                
                bin_hit_at_5.append(1.0 if target in ranked_preds[:5] else 0.0)
            
            if bin_count > 0:
                bins.append({
                    'frequency_range': f"{int(bin_min)}-{int(bin_max)}",
                    'frequency_min': int(bin_min),
                    'frequency_max': int(bin_max),
                    'mrr': float(np.mean(bin_mrr)),
                    'hit_at_5': float(np.mean(bin_hit_at_5)),
                    'technique_count': len(bin_techniques),
                    'prediction_count': bin_count,
                })
        
        return {'bins': bins}

    def analyze_temporal_leakage(self) -> Dict[str, Any]:
        """
        Verify temporal split integrity and detect data leakage.
        
        Academic Note:
        For valid temporal evaluation, we require:
        1. No overlap of sequence IDs between train/val/test
        2. Strict temporal ordering: max(train_time) < min(val_time) < min(test_time)
        
        Returns:
            Dict with:
                checked: bool - Whether leakage check was performed
                leakage_detected: bool - Whether any leakage was found
                temporal_ordering_valid: bool - Whether temporal ordering is correct
                sequence_overlap_detected: bool - Whether sequence IDs overlap
                [train/val/test]_sequences: int - Number of sequences in each split
        """
        if (
            self.sequence_metadata is None
            or not self._temporal_train_indices
            or 'start_time' not in self.sequence_metadata.columns
        ):
            return {
                'checked': False,
                'leakage_detected': False,
                'reason': 'No temporal metadata or indices available'
            }

        metadata = self.sequence_metadata.copy()
        if 'sequence_id' not in metadata.columns:
            metadata['sequence_id'] = metadata.index
        
        metadata = metadata.set_index('sequence_id')
        
        train_meta = metadata.loc[self._temporal_train_indices] if self._temporal_train_indices else pd.DataFrame()
        val_meta = metadata.loc[self._temporal_val_indices] if self._temporal_val_indices else pd.DataFrame()
        test_meta = metadata.loc[self._temporal_test_indices] if self._temporal_test_indices else pd.DataFrame()

        # Check 1: Sequence ID overlap (must be disjoint sets)
        train_ids = set(self._temporal_train_indices)
        val_ids = set(self._temporal_val_indices)
        test_ids = set(self._temporal_test_indices)
        
        overlap_train_val = train_ids & val_ids
        overlap_train_test = train_ids & test_ids
        overlap_val_test = val_ids & test_ids
        sequence_overlap_detected = bool(overlap_train_val or overlap_train_test or overlap_val_test)
        
        # Check 2: Temporal ordering (train < val < test)
        temporal_ordering_valid = True
        temporal_details = {}
        
        if not train_meta.empty:
            train_times = pd.to_datetime(train_meta['start_time'], errors='coerce')
            train_max = train_times.max()
            temporal_details['train_max_time'] = str(train_max) if pd.notna(train_max) else None
        else:
            train_max = None
            
        if not val_meta.empty:
            val_times = pd.to_datetime(val_meta['start_time'], errors='coerce')
            val_min = val_times.min()
            val_max = val_times.max()
            temporal_details['val_min_time'] = str(val_min) if pd.notna(val_min) else None
            temporal_details['val_max_time'] = str(val_max) if pd.notna(val_max) else None
            
            # Check train < val
            if pd.notna(train_max) and pd.notna(val_min):
                if train_max >= val_min:
                    temporal_ordering_valid = False
                    temporal_details['train_val_leakage'] = f"train_max ({train_max}) >= val_min ({val_min})"
        else:
            val_max = None
            
        if not test_meta.empty:
            test_times = pd.to_datetime(test_meta['start_time'], errors='coerce')
            test_min = test_times.min()
            temporal_details['test_min_time'] = str(test_min) if pd.notna(test_min) else None
            
            # Check val < test
            if val_max is not None and pd.notna(val_max) and pd.notna(test_min):
                if val_max >= test_min:
                    temporal_ordering_valid = False
                    temporal_details['val_test_leakage'] = f"val_max ({val_max}) >= test_min ({test_min})"
            # Check train < test (if no val)
            elif train_max is not None and pd.notna(train_max) and pd.notna(test_min):
                if train_max >= test_min:
                    temporal_ordering_valid = False
                    temporal_details['train_test_leakage'] = f"train_max ({train_max}) >= test_min ({test_min})"

        leakage_detected = sequence_overlap_detected or not temporal_ordering_valid
        
        result = {
            'checked': True,
            'leakage_detected': leakage_detected,
            'sequence_overlap_detected': sequence_overlap_detected,
            'temporal_ordering_valid': temporal_ordering_valid,
            'train_sequences': len(train_meta),
            'val_sequences': len(val_meta),
            'test_sequences': len(test_meta),
        }
        result.update(temporal_details)
        
        if leakage_detected:
            print("⚠️  WARNING: Temporal leakage detected!")
            if sequence_overlap_detected:
                print(f"   - Sequence overlap: train∩val={len(overlap_train_val)}, train∩test={len(overlap_train_test)}, val∩test={len(overlap_val_test)}")
            if not temporal_ordering_valid:
                print(f"   - Temporal ordering violation: {temporal_details}")
        
        return result

    def evaluate_next_step(
        self,
        test_sequences: List[List[str]],
        test_contexts: Optional[List[Optional[Dict[str, Optional[str]]]]] = None,
        k_values: List[int] = [1, 3, 5, 10, 20],
        use_scenario_priors: Optional[bool] = None,
        return_details: bool = False,
        n_workers: Optional[int] = None,
        top_n_candidates: Optional[int] = None,
        full_vocab_eval: bool = False,
        disable_filters: bool = False,
        pure_model_eval: bool = False,
        include_baseline: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate next-step prediction performance.
        
        Academic Note:\n        - Metrics renamed: precision@k → recall@k or hit@k (IR-standard naming)
        - Added imbalance-aware metrics: macro-recall, macro-F1, balanced accuracy
        - Added tail-specific metrics for rare techniques (frequency < 100)
        - Added frequency-binned diagnostics for stratified analysis
        
        Args:
            test_sequences: Test sequences to evaluate
            test_contexts: Optional context for each sequence
            k_values: K values for recall@k/hit@k metrics (IR standard: these are recall metrics for single-item relevance)
            use_scenario_priors: Whether to use scenario priors
            return_details: Whether to return detailed samples
            n_workers: Number of parallel workers (None=auto, 1=sequential)
        """

        print("Evaluating next-step prediction...")
        print(f"  Test sequences: {len(test_sequences)}")
        
        # Count sequences with sufficient length
        valid_sequences = [s for s in test_sequences if len(s) >= 2]
        print(f"  Valid sequences (len >= 2): {len(valid_sequences)}/{len(test_sequences)}")

        if test_contexts is None:
            test_contexts = self._last_split_contexts.get('test', [])
        if not test_contexts or len(test_contexts) != len(test_sequences):
            padded_contexts: List[Optional[Dict[str, Optional[str]]]] = [None] * len(test_sequences)
            for idx in range(min(len(test_sequences), len(test_contexts))):
                padded_contexts[idx] = test_contexts[idx]
            test_contexts = padded_contexts

        metrics: Dict[str, Any] = {
            'model': {'recall_at_k': {}, 'hit_at_k': {}, 'MRR': {}},  # IR-standard naming: recall@k = hit@k for single relevant item
            'baselines': {'count_only': {'recall_at_k': {}, 'hit_at_k': {}, 'MRR': {}}} if include_baseline else {},
            'significance': {'count_only': {'recall_at_k': {}, 'MRR': {}}} if include_baseline else {},
            'leakage': self.analyze_temporal_leakage(),
            'observations': 0,
        }

        model_hits = {k: [] for k in k_values}  # renamed from model_precisions
        baseline_hits = {k: [] for k in k_values}  # renamed from baseline_precisions
        mrr_model: List[float] = []
        mrr_baseline: List[float] = []
        
        # For imbalance-aware metrics: track per-class predictions
        predictions_per_class: Dict[str, List[int]] = {}  # class_id -> list of hits (1 or 0)
        all_predictions_for_tail: List[Tuple[str, List[str], int]] = []  # (target, ranked_preds, hit_at_1)
        
        predictions_generated = 0
        predictions_empty = 0
        
        # Full-vocab evaluation forces large candidate sets and disables filtering
        if full_vocab_eval:
            effective_top_n = len(self.predictor.tech_to_idx)
        else:
            effective_top_n = top_n_candidates or 100

        # Use parallel processing only when eval settings are default
        can_parallel = (
            n_workers and n_workers > 1
            and not full_vocab_eval
            and not disable_filters
            and not pure_model_eval
            and include_baseline
        )

        if can_parallel:
            print(f"  Using parallel processing with {n_workers} workers...")
            result = self._evaluate_parallel(
                test_sequences, test_contexts, k_values, use_scenario_priors, n_workers
            )
            model_hits = result['model_hits']  # renamed from model_precisions
            baseline_hits = result['baseline_hits']  # renamed from baseline_precisions
            mrr_model = result['mrr_model']
            mrr_baseline = result['mrr_baseline']
            metrics['observations'] = result['observations']
            predictions_generated = result['predictions_generated']
            predictions_empty = result['predictions_empty']
            predictions_per_class = result['predictions_per_class']
            all_predictions_for_tail = result['all_predictions_for_tail']
        else:
            if n_workers == 1:
                print("  Using sequential processing (--n-workers=1)...")
            else:
                print("  Using sequential processing...")

            with self._temporary_predictor_settings(disable_filters, pure_model_eval):
                for seq_idx, sequence in enumerate(tqdm(test_sequences, desc="Evaluating sequences")):
                    if len(sequence) < 2:
                        continue

                    context = test_contexts[seq_idx] if seq_idx < len(test_contexts) else None

                    for i in range(1, len(sequence)):
                        history = sequence[:i]
                        target = sequence[i]

                        predictions = self.predictor.next_probabilities(
                            history,
                            group_context=context,
                            top_n_candidates=effective_top_n,
                            use_scenario_priors=use_scenario_priors,
                        )
                        if not predictions:
                            predictions_empty += 1
                            continue
                        
                        predictions_generated += 1

                        candidate_names = [pred[0] for pred in predictions]
                        predicted_techs = candidate_names

                        if include_baseline:
                            baseline_scores = [
                                self.predictor.compute_count_probability(
                                    history,
                                    candidate,
                                    use_scenario_priors=False,
                                )[0]
                                for candidate in candidate_names
                            ]
                            baseline_order_idx = np.argsort(-np.array(baseline_scores))
                            baseline_order = [candidate_names[idx] for idx in baseline_order_idx]

                        for k in k_values:
                            if k <= len(predicted_techs):
                                model_hit = 1.0 if target in predicted_techs[:k] else 0.0
                                model_hits[k].append(model_hit)
                                if include_baseline:
                                    baseline_hit = 1.0 if target in baseline_order[:k] else 0.0
                                    baseline_hits[k].append(baseline_hit)
                        
                        # Track per-class hits for imbalance-aware metrics
                        if target not in predictions_per_class:
                            predictions_per_class[target] = []
                        hit_at_1 = 1.0 if target in predicted_techs[:1] else 0.0
                        predictions_per_class[target].append(int(hit_at_1))
                        
                        # Track for tail analysis and frequency binning
                        all_predictions_for_tail.append((target, predicted_techs, int(hit_at_1)))

                        if target in predicted_techs:
                            rank = predicted_techs.index(target) + 1
                            mrr_model.append(1.0 / rank)
                        else:
                            mrr_model.append(0.0)

                        if include_baseline:
                            if target in baseline_order:
                                baseline_rank = baseline_order.index(target) + 1
                                mrr_baseline.append(1.0 / baseline_rank)
                            else:
                                mrr_baseline.append(0.0)

                        metrics['observations'] += 1
        
        print(f"  Predictions generated: {predictions_generated}, empty: {predictions_empty}")
        print(f"  Total observations: {metrics['observations']}")

        # Store raw samples for bootstrap CI (using new naming)
        hit_samples_model = {k: list(model_hits[k]) for k in k_values}
        hit_samples_baseline = {k: list(baseline_hits[k]) for k in k_values}

        # Compute standard Hit@K / Recall@K metrics (IR naming: these are equivalent for binary relevance)
        for k in k_values:
            metrics['model']['recall_at_k'][k] = self._bootstrap_summary(model_hits[k])
            metrics['model']['hit_at_k'][k] = metrics['model']['recall_at_k'][k]  # Alias: hit@k = recall@k
            if include_baseline:
                metrics['baselines']['count_only']['recall_at_k'][k] = self._bootstrap_summary(baseline_hits[k])
                metrics['baselines']['count_only']['hit_at_k'][k] = metrics['baselines']['count_only']['recall_at_k'][k]
                metrics['significance']['count_only']['recall_at_k'][k] = {
                    'p_value': self._paired_bootstrap_test(model_hits[k], baseline_hits[k])
                }

        # Accuracy (top-1) is an alias for Recall@1 / Hit@1
        if 1 in metrics['model']['recall_at_k'] or '1' in metrics['model']['recall_at_k']:
            acc_model = metrics['model']['recall_at_k'].get(1, metrics['model']['recall_at_k'].get('1'))
        else:
            acc_model = self._bootstrap_summary(model_hits.get(1, []))
        metrics['model']['accuracy'] = acc_model

        if include_baseline:
            if 1 in metrics['baselines']['count_only']['recall_at_k'] or '1' in metrics['baselines']['count_only']['recall_at_k']:
                acc_base = metrics['baselines']['count_only']['recall_at_k'].get(1, metrics['baselines']['count_only']['recall_at_k'].get('1'))
            else:
                acc_base = self._bootstrap_summary(baseline_hits.get(1, []))
            metrics['baselines']['count_only']['accuracy'] = acc_base

            metrics['significance']['count_only']['accuracy'] = {
                'p_value': self._paired_bootstrap_test(model_hits.get(1, []), baseline_hits.get(1, []))
            }

        # MRR
        metrics['model']['MRR'] = self._bootstrap_summary(mrr_model)
        if include_baseline:
            metrics['baselines']['count_only']['MRR'] = self._bootstrap_summary(mrr_baseline)
            metrics['significance']['count_only']['MRR'] = {
                'p_value': self._paired_bootstrap_test(mrr_model, mrr_baseline)
            }
        
        # ADDED: Imbalance-aware metrics (macro-recall, macro-F1, balanced accuracy)
        print("  Computing imbalance-aware metrics...")
        imbalance_metrics = self._compute_imbalance_aware_metrics(predictions_per_class)
        metrics['model']['macro_recall'] = {'mean': imbalance_metrics['macro_recall']}
        metrics['model']['macro_f1'] = {'mean': imbalance_metrics['macro_f1']}
        metrics['model']['balanced_accuracy'] = {'mean': imbalance_metrics['balanced_accuracy']}
        
        # ADDED: Tail-specific metrics (rare techniques with frequency < 100)
        print("  Computing tail-specific metrics...")
        tail_metrics = self._compute_tail_metrics(all_predictions_for_tail, frequency_threshold=100)
        metrics['model']['tail_mrr'] = {'mean': tail_metrics['tail_mrr']}
        metrics['model']['tail_hit_at_5'] = {'mean': tail_metrics['tail_hit_at_5']}
        metrics['model']['tail_hit_at_10'] = {'mean': tail_metrics['tail_hit_at_10']}
        metrics['tail_analysis'] = {
            'tail_technique_count': tail_metrics['tail_technique_count'],
            'tail_prediction_count': tail_metrics['tail_prediction_count'],
            'frequency_threshold': tail_metrics['frequency_threshold'],
        }
        
        # ADDED: Frequency-binned diagnostics
        print("  Computing frequency-binned metrics...")
        freq_binned = self._compute_frequency_binned_metrics(all_predictions_for_tail, n_bins=10)
        metrics['frequency_binned_analysis'] = freq_binned

        print("Next-step evaluation results:")
        acc_summary = metrics['model'].get('accuracy', {'mean': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0})
        print(
            f"  Accuracy (Hit@1): {acc_summary['mean']:.4f}"
            f" (95% CI [{acc_summary['ci_lower']:.4f}, {acc_summary['ci_upper']:.4f}])"
        )
        for k in k_values:
            summary = metrics['model']['recall_at_k'][k]
            print(
                f"  Recall@{k} (Hit@{k}): {summary['mean']:.4f}"
                f" (95% CI [{summary['ci_lower']:.4f}, {summary['ci_upper']:.4f}])"
            )
        mrr_summary = metrics['model']['MRR']
        print(
            f"  MRR: {mrr_summary['mean']:.4f}"
            f" (95% CI [{mrr_summary['ci_lower']:.4f}, {mrr_summary['ci_upper']:.4f}])"
        )
        # Print imbalance-aware metrics
        print(f"  Macro-Recall: {metrics['model']['macro_recall']['mean']:.4f}")
        print(f"  Macro-F1: {metrics['model']['macro_f1']['mean']:.4f}")
        print(f"  Balanced Accuracy: {metrics['model']['balanced_accuracy']['mean']:.4f}")
        # Print tail metrics
        print(f"  Tail MRR (freq<{tail_metrics['frequency_threshold']}): {metrics['model']['tail_mrr']['mean']:.4f}")
        print(f"  Tail Hit@5: {metrics['model']['tail_hit_at_5']['mean']:.4f}")
        print(f"  Tail Hit@10: {metrics['model']['tail_hit_at_10']['mean']:.4f}")

        if return_details:
            details = {
                'hit_samples': hit_samples_model,  # renamed from precision_samples
                'baseline_hit_samples': hit_samples_baseline,
                'mrr_samples': list(mrr_model),
                'baseline_mrr_samples': list(mrr_baseline),
                'use_scenario_priors': (
                    use_scenario_priors
                    if use_scenario_priors is not None
                    else self.predictor.params.get('use_scenario_priors', True)
                ),
            }
            return metrics, details

        return metrics
    
    def _evaluate_parallel(
        self,
        test_sequences: List[List[str]],
        test_contexts: List[Optional[Dict[str, Optional[str]]]],
        k_values: List[int],
        use_scenario_priors: Optional[bool],
        n_workers: int,
    ) -> Dict[str, Any]:
        """
        Parallel evaluation using multiprocessing.
        
        Each worker processes a subset of sequences independently.
        Results are aggregated after all workers complete.
        """
        from .parallel import _init_worker_predictor, _evaluate_single_sequence_worker
        import multiprocessing as mp
        
        # Prepare worker arguments
        worker_args = []
        for seq_idx, (sequence, context) in enumerate(zip(test_sequences, test_contexts)):
            worker_args.append((seq_idx, sequence, context, k_values, use_scenario_priors))
        
        # Save checkpoint path for worker initialization
        checkpoint_path = getattr(self.predictor, '_checkpoint_path', None)
        if checkpoint_path is None:
            # Need to save temporary checkpoint for workers
            print("⚠️  Warning: No checkpoint path found. Parallel evaluation requires saved checkpoint.")
            print("⚠️  Falling back to sequential processing...")
            return self._evaluate_sequential_impl(test_sequences, test_contexts, k_values, use_scenario_priors)
        
        # Create pool with initializer - use spawn to avoid fork issues with CUDA
        print(f"  Initializing {n_workers} worker processes...")
        
        # Use smaller chunksize for better progress visibility
        chunksize = max(1, min(100, len(worker_args) // (n_workers * 10)))
        print(f"  Using chunksize={chunksize} for {len(worker_args)} sequences")
        
        with mp.Pool(
            processes=n_workers,
            initializer=_init_worker_predictor,
            initargs=(checkpoint_path,)
        ) as pool:
            # Map work to workers with progress bar - use imap_unordered for faster feedback
            results = list(tqdm(
                pool.imap_unordered(_evaluate_single_sequence_worker, worker_args, chunksize=chunksize),
                total=len(worker_args),
                desc="Evaluating sequences (parallel)"
            ))
        
        # Aggregate results from all workers
        model_hits = {k: [] for k in k_values}
        baseline_hits = {k: [] for k in k_values}
        mrr_model = []
        mrr_baseline = []
        total_observations = 0
        total_predictions_generated = 0
        total_predictions_empty = 0
        predictions_per_class: Dict[str, List[int]] = {}
        all_predictions_for_tail: List[Tuple[str, List[str], int]] = []
        
        for result in results:
            total_observations += result['observations']
            total_predictions_generated += result['predictions_generated']
            total_predictions_empty += result['predictions_empty']
            
            for k in k_values:
                model_hits[k].extend(result['model_hits'][k])
                baseline_hits[k].extend(result['baseline_hits'][k])
            
            mrr_model.extend(result['mrr_model'])
            mrr_baseline.extend(result['mrr_baseline'])
            
            # Aggregate per-class predictions
            for class_id, hits in result.get('predictions_per_class', {}).items():
                if class_id not in predictions_per_class:
                    predictions_per_class[class_id] = []
                predictions_per_class[class_id].extend(hits)
            
            # Aggregate tail predictions
            all_predictions_for_tail.extend(result.get('all_predictions_for_tail', []))
        
        print(f"  Predictions generated: {total_predictions_generated}, empty: {total_predictions_empty}")
        print(f"  Total observations: {total_observations}")
        
        return {
            'model_hits': model_hits,
            'baseline_hits': baseline_hits,
            'mrr_model': mrr_model,
            'mrr_baseline': mrr_baseline,
            'observations': total_observations,
            'predictions_generated': total_predictions_generated,
            'predictions_empty': total_predictions_empty,
            'predictions_per_class': predictions_per_class,
            'all_predictions_for_tail': all_predictions_for_tail,
        }
    
    def _evaluate_sequential_impl(
        self,
        test_sequences: List[List[str]],
        test_contexts: List[Optional[Dict[str, Optional[str]]]],
        k_values: List[int],
        use_scenario_priors: Optional[bool],
    ) -> Dict[str, Any]:
        """Sequential evaluation implementation (extracted for code reuse)."""
        model_hits = {k: [] for k in k_values}
        baseline_hits = {k: [] for k in k_values}
        mrr_model = []
        mrr_baseline = []
        predictions_generated = 0
        predictions_empty = 0
        observations = 0
        predictions_per_class: Dict[str, List[int]] = {}
        all_predictions_for_tail: List[Tuple[str, List[str], int]] = []
        
        for seq_idx, sequence in enumerate(tqdm(test_sequences, desc="Evaluating sequences")):
            if len(sequence) < 2:
                continue

            context = test_contexts[seq_idx] if seq_idx < len(test_contexts) else None

            for i in range(1, len(sequence)):
                history = sequence[:i]
                target = sequence[i]

                predictions = self.predictor.next_probabilities(
                    history,
                    group_context=context,
                    top_n_candidates=100,
                    use_scenario_priors=use_scenario_priors,
                )
                if not predictions:
                    predictions_empty += 1
                    continue
                
                predictions_generated += 1

                candidate_names = [pred[0] for pred in predictions]
                predicted_techs = candidate_names

                # Use BiLSTM scores for baseline comparison (fallback to uniform)
                bilstm_scores = np.array([
                    pred[2].get('P_bilstm', pred[2].get('P_lstm', pred[2].get('P_count', 1.0)))
                    for pred in predictions
                ], dtype=float)
                if bilstm_scores.sum() > 0:
                    baseline_probs = bilstm_scores / bilstm_scores.sum()
                else:
                    baseline_probs = np.ones_like(bilstm_scores) / len(bilstm_scores)
                baseline_order_idx = np.argsort(-baseline_probs)
                baseline_order = [candidate_names[idx] for idx in baseline_order_idx]

                for k in k_values:
                    if k <= len(predicted_techs):
                        model_hit = 1.0 if target in predicted_techs[:k] else 0.0
                        baseline_hit = 1.0 if target in baseline_order[:k] else 0.0
                        model_hits[k].append(model_hit)
                        baseline_hits[k].append(baseline_hit)
                
                # Track per-class hits for imbalance-aware metrics
                if target not in predictions_per_class:
                    predictions_per_class[target] = []
                hit_at_1 = 1.0 if target in predicted_techs[:1] else 0.0
                predictions_per_class[target].append(int(hit_at_1))
                
                # Track for tail analysis
                all_predictions_for_tail.append((target, predicted_techs, int(hit_at_1)))

                if target in predicted_techs:
                    rank = predicted_techs.index(target) + 1
                    mrr_model.append(1.0 / rank)
                else:
                    mrr_model.append(0.0)

                if target in baseline_order:
                    baseline_rank = baseline_order.index(target) + 1
                    mrr_baseline.append(1.0 / baseline_rank)
                else:
                    mrr_baseline.append(0.0)

                observations += 1
        
        print(f"  Predictions generated: {predictions_generated}, empty: {predictions_empty}")
        print(f"  Total observations: {observations}")
        
        return {
            'model_hits': model_hits,
            'baseline_hits': baseline_hits,
            'mrr_model': mrr_model,
            'mrr_baseline': mrr_baseline,
            'observations': observations,
            'predictions_generated': predictions_generated,
            'predictions_empty': predictions_empty,
            'predictions_per_class': predictions_per_class,
            'all_predictions_for_tail': all_predictions_for_tail,
        }
    
    def evaluate_path_level(
        self,
        test_sequences: List[List[str]],
        test_contexts: Optional[List[Optional[Dict[str, Optional[str]]]]] = None,
        k_values: List[int] = [5, 10, 20],
        n_workers: Optional[int] = None,
        beam_width: int = 100,
        max_depth: int = 6,
        top_k_paths: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate path-level prediction performance.

        Args:
            test_sequences: Test sequences
            test_contexts: Optional context for each sequence
            k_values: K values for Hit@K
            n_workers: Number of parallel workers (None=auto, 1=sequential)
            beam_width: Beam search width (default: 100)
            max_depth: Maximum search depth (default: 6)
            top_k_paths: Number of top paths to generate (default: 100)

        Returns:
            Dictionary of evaluation metrics
        """
        print("Evaluating path-level prediction...")

        if test_contexts is None:
            test_contexts = self._last_split_contexts.get('test', [])
        if not test_contexts or len(test_contexts) != len(test_sequences):
            padded_contexts: List[Optional[Dict[str, Optional[str]]]] = [None] * len(test_sequences)
            for idx in range(min(len(test_sequences), len(test_contexts))):
                padded_contexts[idx] = test_contexts[idx]
            test_contexts = padded_contexts

        metrics = {}
        all_hits = {k: [] for k in k_values}
        
        # Use parallel processing if n_workers > 1
        if n_workers and n_workers > 1:
            print(f"  Using parallel processing with {n_workers} workers for path-level...")
            results = self._evaluate_path_level_parallel(
                test_sequences, test_contexts, k_values, n_workers, beam_width, max_depth, top_k_paths
            )
            # Aggregate results
            for result in results:
                if result['valid']:
                    for k in k_values:
                        all_hits[k].append(result['hits'][k])
        else:
            # Sequential processing (original code)
            if n_workers == 1:
                print("  Using sequential processing for path-level...")
            else:
                print("  Using sequential processing for path-level...")

            for seq_idx, sequence in enumerate(tqdm(test_sequences, desc="Evaluating paths")):
                if len(sequence) < 3:
                    continue

                # Use first 1-3 techniques as seed
                seed_length = min(3, len(sequence) - 1)
                seed = sequence[:seed_length]
                target_path = sequence[seed_length:]

                if not target_path:
                    continue

                # Get predictions using beam search
                beam_search = BeamSearch(self.predictor, beam_width=beam_width, max_depth=max_depth, top_k_paths=top_k_paths)

                context = test_contexts[seq_idx] if seq_idx < len(test_contexts) else None
                try:
                    predicted_paths = beam_search.search_paths(seed, group_context=context)
                except Exception as e:
                    print(f"Warning: Beam search failed for sequence: {e}")
                    continue
                
                # Check if target path appears in predictions
                for k in k_values:
                    if k <= len(predicted_paths):
                        hit = False
                        for pred_path in predicted_paths[:k]:
                            pred_sequence = pred_path['sequence']
                            # Check if target path is a subsequence of predicted path
                            if self.is_subsequence(target_path, pred_sequence):
                                hit = True
                                break
                        
                        all_hits[k].append(1.0 if hit else 0.0)
        
        # Compute average metrics
        for k in k_values:
            if all_hits[k]:
                metrics[f'Hit@{k}'] = np.mean(all_hits[k])
            else:
                metrics[f'Hit@{k}'] = 0.0
        
        print("Path-level evaluation results:")
        for k in k_values:
            print(f"  Hit@{k}: {metrics[f'Hit@{k}']:.4f}")
        
        return metrics
    
    def _evaluate_path_level_parallel(
        self,
        test_sequences: List[List[str]],
        test_contexts: List[Optional[Dict[str, Optional[str]]]],
        k_values: List[int],
        n_workers: int,
        beam_width: int,
        max_depth: int,
        top_k_paths: int,
    ) -> List[Dict[str, Any]]:
        """
        Parallel path-level evaluation using multiprocessing.
        
        Each worker processes paths independently using beam search.
        """
        from .parallel import _init_worker_predictor, _evaluate_single_path_worker
        import multiprocessing as mp
        
        # Prepare worker arguments
        worker_args = []
        for seq_idx, (sequence, context) in enumerate(zip(test_sequences, test_contexts)):
            worker_args.append((seq_idx, sequence, context, k_values, beam_width, max_depth, top_k_paths))
        
        # Get checkpoint path for worker initialization
        checkpoint_path = getattr(self.predictor, '_checkpoint_path', None)
        if checkpoint_path is None:
            print("⚠️  Warning: No checkpoint path found. Falling back to sequential processing...")
            # Return empty results - will fall back to sequential
            return []
        
        # Create pool with initializer
        print(f"  Initializing {n_workers} worker processes for path-level evaluation...")
        
        # Use smaller chunksize for better progress visibility
        chunksize = max(1, min(50, len(worker_args) // (n_workers * 10)))
        print(f"  Using chunksize={chunksize} for {len(worker_args)} sequences")
        
        with mp.Pool(
            processes=n_workers,
            initializer=_init_worker_predictor,
            initargs=(checkpoint_path,)
        ) as pool:
            # Map work to workers with progress bar
            results = list(tqdm(
                pool.imap_unordered(_evaluate_single_path_worker, worker_args, chunksize=chunksize),
                total=len(worker_args),
                desc="Evaluating paths (parallel)"
            ))
        
        return results
    
    def is_subsequence(self, subseq: List[str], seq: List[str]) -> bool:
        """
        Check if subseq is a subsequence of seq.
        
        Args:
            subseq: Subsequence to find
            seq: Sequence to search in
            
        Returns:
            True if subseq is a subsequence of seq
        """
        if not subseq:
            return True
        
        i = 0
        for item in seq:
            if item == subseq[i]:
                i += 1
                if i == len(subseq):
                    return True
        
        return False

    def evaluate_path_probability(
        self,
        test_sequences: List[List[str]],
        test_contexts: Optional[List[Optional[Dict[str, Optional[str]]]]] = None,
        top_n_candidates: int = 100,
        use_scenario_priors: Optional[bool] = None,
        epsilon: float = 1e-12,
    ) -> Dict[str, Any]:
        """
        Compute probability metrics for the full ground-truth path.

        For each test sequence, we iterate each next step and collect the
        model probability assigned to the true next technique, then aggregate
        per-path average log-probability and perplexity.

        Args:
            test_sequences: Test sequences
            test_contexts: Optional context metadata aligned to sequences
            top_n_candidates: Number of candidates to request per step
            use_scenario_priors: Override predictor scenario-prior usage
            epsilon: Floor probability for unseen targets to avoid -inf

        Returns:
            Dict with aggregate metrics:
              - mean_avg_log_prob: Mean per-step log-probability averaged per path
              - mean_perplexity: exp(-mean_avg_log_prob)
              - paths_evaluated, total_steps, zero_prob_steps
        """
        print("Computing path probability metrics (full ground-truth paths)...")

        if test_contexts is None:
            test_contexts = self._last_split_contexts.get('test', [])
        if not test_contexts or len(test_contexts) != len(test_sequences):
            padded_contexts: List[Optional[Dict[str, Optional[str]]]] = [None] * len(test_sequences)
            for idx in range(min(len(test_sequences), len(test_contexts))):
                padded_contexts[idx] = test_contexts[idx]
            test_contexts = padded_contexts

        per_path_avg_logp: List[float] = []
        total_steps = 0
        zero_prob_steps = 0

        for seq_idx, sequence in enumerate(tqdm(test_sequences, desc="Path prob")):
            if len(sequence) < 2:
                continue

            context = test_contexts[seq_idx] if seq_idx < len(test_contexts) else None
            step_logps: List[float] = []

            for i in range(1, len(sequence)):
                history = sequence[:i]
                target = sequence[i]

                preds = self.predictor.next_probabilities(
                    history,
                    group_context=context,
                    top_n_candidates=top_n_candidates,
                    use_scenario_priors=use_scenario_priors,
                )

                # Find probability assigned to the true next technique
                prob = 0.0
                if preds:
                    for tech, p, _ in preds:
                        if tech == target:
                            prob = float(p)
                            break

                if prob <= 0.0:
                    zero_prob_steps += 1
                    prob = epsilon

                step_logps.append(float(np.log(prob)))
                total_steps += 1

            if step_logps:
                per_path_avg_logp.append(float(np.mean(step_logps)))

        mean_avg_logp = float(np.mean(per_path_avg_logp)) if per_path_avg_logp else 0.0
        mean_perplexity = float(np.exp(-mean_avg_logp)) if per_path_avg_logp else float('inf')

        metrics = {
            'paths_evaluated': len(per_path_avg_logp),
            'total_steps': total_steps,
            'zero_prob_steps': zero_prob_steps,
            'mean_avg_log_prob': mean_avg_logp,
            'mean_perplexity': mean_perplexity,
        }

        print("Path probability evaluation:")
        print(f"  Paths evaluated: {metrics['paths_evaluated']}")
        print(f"  Total steps: {metrics['total_steps']} (zero-prob steps: {metrics['zero_prob_steps']})")
        print(f"  Mean avg log-prob per step: {metrics['mean_avg_log_prob']:.6f}")
        print(f"  Mean perplexity: {metrics['mean_perplexity']:.4f}")

        return metrics
    
    def run_ablation_study(self, test_sequences: List[List[str]]) -> Dict[str, Dict[str, Any]]:
        """
        Run ablation study by removing different components.
        
        Args:
            test_sequences: Test sequences
            
        Returns:
            Dictionary of ablation results
        """
        print("Running ablation study...")
        
        ablation_results: Dict[str, Dict[str, Any]] = {}

        # --- Original model ---
        original_params = deepcopy(self.predictor.params)

        print("Evaluating original model...")
        test_contexts = self._last_split_contexts.get('test', [])
        original_metrics, original_details = self.evaluate_next_step(
            test_sequences,
            test_contexts=test_contexts,
            return_details=True,
        )
        ablation_results['original'] = {
            'metrics': original_metrics,
            'confidence_intervals': self._summarise_confidence_intervals(original_details),
        }
        
        # --- Ablation 1: Remove embeddings (set kappa to a very high value) ---
        print("Evaluating without embeddings...")
        original_kappa = self.predictor.params.get('kappa', 10.0)
        self.predictor.params['kappa'] = 1e9
        print(f"  Changed kappa: {original_kappa} → {self.predictor.params['kappa']}")
        metrics, details = self.evaluate_next_step(
            test_sequences,
            test_contexts=test_contexts,
            return_details=True,
        )
        ablation_results['no_embeddings'] = {
            'metrics': metrics,
            'confidence_intervals': self._summarise_confidence_intervals(details),
            'mrr_difference': _paired_bootstrap_difference(original_details['mrr_samples'], details['mrr_samples'])
        }
        self.predictor.params['kappa'] = original_kappa
        
        # --- Ablation 2: Remove stealth scores (set gamma to 0) ---
        print("Evaluating without stealth scores...")
        original_gamma = self.predictor.params.get('gamma', 1.0)
        self.predictor.params['gamma'] = 0.0
        metrics, details = self.evaluate_next_step(
            test_sequences,
            test_contexts=test_contexts,
            return_details=True,
        )
        ablation_results['no_stealth'] = {
            'metrics': metrics,
            'confidence_intervals': self._summarise_confidence_intervals(details),
            'mrr_difference': _paired_bootstrap_difference(original_details['mrr_samples'], details['mrr_samples'])
        }
        self.predictor.params['gamma'] = original_gamma
        
        # --- Ablation 3: Remove group priors (set eta to 0) ---
        print("Evaluating without group priors...")
        original_eta = self.predictor.params.get('eta', 0.7)
        self.predictor.params['eta'] = 0.0
        metrics, details = self.evaluate_next_step(
            test_sequences,
            test_contexts=test_contexts,
            return_details=True,
        )
        ablation_results['no_group_priors'] = {
            'metrics': metrics,
            'confidence_intervals': self._summarise_confidence_intervals(details),
            'mrr_difference': _paired_bootstrap_difference(original_details['mrr_samples'], details['mrr_samples'])
        }
        self.predictor.params['eta'] = original_eta

        # --- Ablation 4: Remove embedding multiplier (set rho to 0) ---
        print("Evaluating without embedding multiplier...")
        original_rho = self.predictor.params.get('rho', 2.0)
        self.predictor.params['rho'] = 0.0
        metrics, details = self.evaluate_next_step(
            test_sequences,
            test_contexts=test_contexts,
            return_details=True,
        )
        ablation_results['no_emb_multiplier'] = {
            'metrics': metrics,
            'confidence_intervals': self._summarise_confidence_intervals(details),
            'mrr_difference': _paired_bootstrap_difference(original_details['mrr_samples'], details['mrr_samples'])
        }
        self.predictor.params['rho'] = original_rho

        # --- Ablation 5: Counts only (disable all other features) ---
        print("Evaluating with counts only...")
        self.predictor.params['kappa'] = 1e9
        self.predictor.params['gamma'] = 0.0
        self.predictor.params['rho'] = 0.0
        metrics, details = self.evaluate_next_step(
            test_sequences,
            test_contexts=test_contexts,
            return_details=True,
        )
        ablation_results['counts_only'] = {
            'metrics': metrics,
            'confidence_intervals': self._summarise_confidence_intervals(details),
            'mrr_difference': _paired_bootstrap_difference(original_details['mrr_samples'], details['mrr_samples'])
        }
        self.predictor.params = deepcopy(original_params)

        return ablation_results

    def evaluate_prior_variants(
        self,
        train_sequences: List[List[str]],
        val_sequences: List[List[str]],
        test_sequences: List[List[str]],
        train_contexts: Optional[List[Optional[Dict[str, Optional[str]]]]] = None,
        val_contexts: Optional[List[Optional[Dict[str, Optional[str]]]]] = None,
        test_contexts: Optional[List[Optional[Dict[str, Optional[str]]]]] = None,
        n_workers: Optional[int] = None,
        full_vocab_eval: bool = False,
        disable_filters: bool = False,
        pure_model_eval: bool = False,
        include_baseline: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare legacy priors with scenario-enhanced priors.
        
        Academic Note:
        CRITICAL FIX: Model selection now uses VALIDATION SET ONLY.
        - Variants evaluated on validation set
        - Best variant selected based on validation MRR
        - Selected variant evaluated ONCE on test set
        - Test set NEVER used for model selection (prevents data leakage)
        
        Args:
            train_sequences: Training sequences (not used for evaluation)
            val_sequences: Validation sequences (used for model selection)
            test_sequences: Test sequences (used only for final evaluation of selected model)
            [contexts]: Optional contexts for each split
        """

        if val_contexts is None:
            val_contexts = self._last_split_contexts.get('val', [])
        if test_contexts is None:
            test_contexts = self._last_split_contexts.get('test', [])

        print(f"[VALIDATION-BASED SELECTION] Evaluating prior variants...")
        print(f"  Validation sequences: {len(val_sequences)}")
        print(f"  Test sequences: {len(test_sequences)} (held out for final evaluation)")
        
        variants = {
            'scenario_priors': True,
            'legacy_priors': False,
        }

        comparison: Dict[str, Any] = {}
        original_setting = self.predictor.params.get('use_scenario_priors', True)

        # STEP 1: Evaluate all variants on VALIDATION SET
        print("\n[Step 1/2] Evaluating variants on VALIDATION set...")
        for name, flag in variants.items():
            print(f"  Evaluating {name}... (use_scenario_priors={flag})")
            val_metrics, val_details = self.evaluate_next_step(
                val_sequences,
                test_contexts=val_contexts,
                use_scenario_priors=flag,
                return_details=True,
                n_workers=n_workers,
                full_vocab_eval=full_vocab_eval,
                disable_filters=disable_filters,
                pure_model_eval=pure_model_eval,
                include_baseline=include_baseline,
            )
            val_mrr_mean = val_metrics.get('model', {}).get('MRR', {}).get('mean', 0.0)
            val_obs_count = val_metrics.get('observations', 0)
            print(f"    {name}: Validation MRR={val_mrr_mean:.4f}, observations={val_obs_count}")
            comparison[name] = {
                'validation_metrics': val_metrics,
                'validation_details': val_details,
            }

        # STEP 2: Select best variant based on VALIDATION MRR
        def _val_mrr_mean(entry: Dict[str, Any]) -> float:
            metrics = entry.get('validation_metrics', {})
            return metrics.get('model', {}).get('MRR', {}).get('mean', 0.0)

        selected_variant, selected_entry = max(
            comparison.items(), key=lambda item: _val_mrr_mean(item[1])
        )
        selected_flag = variants[selected_variant]
        self.predictor.params['use_scenario_priors'] = selected_flag
        print(f"\n[Step 2/2] Selected variant: {selected_variant} (Validation MRR={_val_mrr_mean(selected_entry):.4f})")

        # STEP 3: Evaluate selected variant ONCE on TEST SET
        print(f"  Evaluating selected variant on TEST set...")
        test_metrics, test_details = self.evaluate_next_step(
            test_sequences,
            test_contexts=test_contexts,
            use_scenario_priors=selected_flag,
            return_details=True,
            n_workers=n_workers,
            full_vocab_eval=full_vocab_eval,
            disable_filters=disable_filters,
            pure_model_eval=pure_model_eval,
            include_baseline=include_baseline,
        )
        test_mrr_mean = test_metrics.get('model', {}).get('MRR', {}).get('mean', 0.0)
        print(f"    Test MRR: {test_mrr_mean:.4f}")

        comparison['selected_variant'] = selected_variant
        comparison['selected_validation_metrics'] = selected_entry['validation_metrics']
        comparison['selected_validation_details'] = selected_entry['validation_details']
        comparison['selected_test_metrics'] = test_metrics  # Final test performance
        comparison['selected_test_details'] = test_details
        comparison['original_setting'] = original_setting
        comparison['selection_criterion'] = 'validation_mrr'  # Document selection method
        return comparison
    
    def evaluate_all(self, train_ratio: float = 0.8, temporal_split: bool = True,
                     skip_ablation: bool = False, n_workers: Optional[int] = None,
                     skip_prior_comparison: bool = False, skip_path_level: bool = False,
                     full_vocab_eval: bool = False, disable_filters: bool = False,
                     pure_model_eval: bool = False, include_baseline: bool = True) -> Dict[str, Any]:
        """
        Run complete evaluation.
        
        Args:
            train_ratio: Ratio of sequences for training (0.0 = use pre-split test set without re-shuffling)
            temporal_split: Whether to use temporal split (deprecated, ignored)
            skip_ablation: If True, skips the ablation study (saves 6 evaluation passes)
            n_workers: Number of parallel workers (None=auto, 1=sequential)
            skip_prior_comparison: If True, skips prior variant comparison (saves 1 pass)
            skip_path_level: If True, skips path-level beam search evaluation (saves time)
            
        Returns:
            Dictionary of all evaluation results
            
        Note:
            When train_ratio=0.0 (used for checkpoint evaluation), sequences are NOT re-shuffled.
            This preserves the original sequence order from the checkpoint's saved split indices.
            Metrics are computed on the exact test set used during model training.
            
        Performance:
            Full evaluation runs 8 passes over test set:
            - 2 for prior comparison (scenario_priors vs legacy_priors)
            - 6 for ablation study (original + 5 ablations)
            Use --skip-ablation --skip-prior-comparison --skip-path-level for fastest evaluation.
        """
        print("Running complete evaluation...")
        print(f"  Total test sequences: {len(self.sequences)}")
        
        # Count evaluation passes for user info
        num_passes = 1  # Base evaluation
        if not skip_prior_comparison:
            num_passes += 1  # Prior comparison adds 1 pass (2 variants)
        if not skip_ablation:
            num_passes += 6  # Ablation adds 6 passes
        path_info = "" if skip_path_level else " + path-level beam search"
        print(f"  Evaluation passes: {num_passes}{path_info}")
        print(f"  Use --skip-ablation --skip-prior-comparison --skip-path-level for fastest evaluation")
        
        # Split sequences (for checkpoint evaluation, train_ratio=0.0 uses all as test)
        train_sequences, test_sequences = self.split_sequences(train_ratio, temporal_split)
        
        # Extract validation sequences from the split
        val_indices = self._last_split_indices.get('val', [])
        val_sequences = [self.sequences[i] for i in val_indices] if val_indices else []
        val_contexts = self._last_split_contexts.get('val', [])
        train_contexts = self._last_split_contexts.get('train', [])
        
        if len(test_sequences) == 0:
            print("⚠️  WARNING: Test set is empty after split!")
            return {
                'next_step_metrics': {},
                'next_step_confidence_intervals': {},
                'path_metrics': {},
                'ablation_results': {},
                'prior_comparison': {},
                'dataset_info': {'test_sequences': 0},
            }
        
        # Log sequence distribution
        print(f"  Train sequences: {len(train_sequences)}")
        print(f"  Val sequences: {len(val_sequences)}")
        print(f"  Test sequences: {len(test_sequences)}")
        test_lengths = [len(seq) for seq in test_sequences]
        print(f"  Test sequence lengths: min={min(test_lengths)}, max={max(test_lengths)}, avg={sum(test_lengths)/len(test_lengths):.1f}")
        print(f"  Sequences with len >= 2: {sum(1 for s in test_sequences if len(s) >= 2)}")
        print(f"  Sequences with len >= 3: {sum(1 for s in test_sequences if len(s) >= 3)}")
        
        test_contexts = self._last_split_contexts.get('test', [])

        # Compare prior variants using validation set for selection
        if skip_prior_comparison:
            print("Skipping prior comparison, running single evaluation on test set...")
            next_step_metrics, next_step_details = self.evaluate_next_step(
                test_sequences,
                test_contexts=test_contexts,
                return_details=True,
                n_workers=n_workers,
                full_vocab_eval=full_vocab_eval,
                disable_filters=disable_filters,
                pure_model_eval=pure_model_eval,
                include_baseline=include_baseline,
            )
            next_step_ci = self._summarise_confidence_intervals(next_step_details)
            prior_comparison = {
                'selected_variant': 'default',
                'selected_test_metrics': next_step_metrics,
                'selected_test_details': next_step_details,
                'skipped': True,
            }
        else:
            # Use validation set for model selection, then evaluate on test
            if not val_sequences:
                print("⚠️  WARNING: No validation set available, falling back to test-set selection (not recommended)")
                val_sequences = test_sequences
                val_contexts = test_contexts
            
            prior_comparison = self.evaluate_prior_variants(
                train_sequences,
                val_sequences,
                test_sequences,
                train_contexts=train_contexts,
                val_contexts=val_contexts,
                test_contexts=test_contexts,
                n_workers=n_workers,
                full_vocab_eval=full_vocab_eval,
                disable_filters=disable_filters,
                pure_model_eval=pure_model_eval,
                include_baseline=include_baseline,
            )
            next_step_metrics = prior_comparison['selected_test_metrics']
            next_step_details = prior_comparison['selected_test_details']
            next_step_ci = self._summarise_confidence_intervals(next_step_details)

        # Evaluate path-level prediction (very slow - runs beam search per sequence)
        path_metrics = {}
        if not skip_path_level:
            path_metrics = self.evaluate_path_level(test_sequences, test_contexts=test_contexts, n_workers=n_workers)
        else:
            print("Skipping path-level evaluation as requested.")

        # Evaluate path probability metrics (fast; no beam search)
        path_prob_metrics = self.evaluate_path_probability(
            test_sequences,
            test_contexts=test_contexts,
            use_scenario_priors=self.predictor.params.get('use_scenario_priors', True),
        )

        # Run ablation study
        ablation_results = {}
        if not skip_ablation:
            if pure_model_eval:
                print("Skipping ablation study because pure_model_eval is enabled.")
                ablation_results = {}
            else:
                ablation_results = self.run_ablation_study(test_sequences)
        else:
            print("Skipping ablation study as requested.")
        
        # Combine results
        dataset_info = {
            'total_sequences': len(self.sequences),
            'train_sequences': len(train_sequences),
            'test_sequences': len(test_sequences),
            'avg_sequence_length': float(np.mean([len(seq) for seq in self.sequences])) if self.sequences else 0.0,
            'temporal_split': temporal_split,
        }

        if temporal_split and self.sequence_metadata is not None:
            dataset_info['temporal_sequences_used'] = len(self._temporal_valid_indices)

            temporal_metadata = self.sequence_metadata[self.sequence_metadata['sequence_id'].isin(self._temporal_valid_indices)]
            if not temporal_metadata.empty:
                dataset_info['temporal_start'] = temporal_metadata['start_time'].min().isoformat() if pd.notna(temporal_metadata['start_time'].min()) else None
                dataset_info['temporal_end'] = temporal_metadata['start_time'].max().isoformat() if pd.notna(temporal_metadata['start_time'].max()) else None

        results = {
            'next_step_metrics': next_step_metrics,
            'next_step_confidence_intervals': next_step_ci,
            'path_metrics': path_metrics,
            'path_probability_metrics': path_prob_metrics,
            'ablation_results': ablation_results,
            'prior_comparison': prior_comparison,
            'dataset_info': dataset_info,
        }

        return results

    def _summarise_confidence_intervals(self, details: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        mrr_samples = details.get('mrr_samples', [])
        lower, upper = _bootstrap_confidence_interval(mrr_samples)
        summary['MRR'] = {'lower': lower, 'upper': upper}

        # Support both old naming (precision_samples) and new naming (hit_samples)
        hit_samples = details.get('hit_samples', details.get('precision_samples', {}))
        for k, samples in hit_samples.items():
            l, u = _bootstrap_confidence_interval(samples)
            # Use new IR-standard naming
            summary[f'recall_at_{k}'] = {'lower': l, 'upper': u}
            summary[f'hit_at_{k}'] = {'lower': l, 'upper': u}  # Alias

        return summary

def run_evaluation(predictor: Predictor, sequences: List[List[str]],
                  output_file: str = "outputs/metrics.json",
                  skip_ablation: bool = False,
                  sequence_metadata: Optional[pd.DataFrame] = None,
                  skip_prior_comparison: bool = False,
                  skip_path_level: bool = False,
                  full_vocab_eval: bool = False,
                  disable_filters: bool = False,
                  pure_model_eval: bool = False,
                  include_baseline: bool = True,
                  n_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Run evaluation and save results.
    
    Args:
        predictor: Chain-aware predictor
        sequences: List of technique sequences
        output_file: Output file path
        skip_ablation: If True, skips the ablation study.
        sequence_metadata: Optional DataFrame with sequence-level metadata for temporal splitting.
        full_vocab_eval: When False (default), enables parallelism; set True only if you need full-vocab scoring
        disable_filters: When False (default), keeps filtering on and allows parallelism
        pure_model_eval: When False (default), keeps normal weighting and allows parallelism
        include_baseline: Keep True to compare against count baseline (required for parallel path)
        n_workers: Number of CPU workers for parallel eval (None=auto: uses os.cpu_count())
        
    Returns:
        Evaluation results
    """
    print("Running evaluation...")
    
    # Initialize evaluator
    evaluator = Evaluator(predictor, sequences, sequence_metadata=sequence_metadata)
    
    # Run evaluation
    # Auto-select workers if not provided
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)

    results = evaluator.evaluate_all(
        skip_ablation=skip_ablation,
        skip_prior_comparison=skip_prior_comparison,
        skip_path_level=skip_path_level,
        full_vocab_eval=full_vocab_eval,
        disable_filters=disable_filters,
        pure_model_eval=pure_model_eval,
        include_baseline=include_baseline,
        n_workers=n_workers,
    )
    
    # Save results
    ensure_dir(output_file)
    save_json(results, output_file)
    
    print(f"Saved evaluation results to {output_file}")
    
    return results


def main():
    """Main function to run evaluation."""
    import argparse
    from .predictor import load_predictor
    from .io import (
        load_sequences_for_source,
        load_sequence_metadata_for_source,
    )
    from .config import DEFAULT_DATA_SOURCE

    parser = argparse.ArgumentParser(description="Evaluate chain-aware predictor")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (e.g., LSTM pkl)')
    parser.add_argument('--n-workers', type=int, default=None, help='Number of CPU workers (default: auto)')
    parser.add_argument('--output-file', type=str, default="outputs/metrics.json", help='Output metrics file')
    parser.add_argument('--skip-ablation', action='store_true', help='Skip ablation study')
    parser.add_argument('--skip-prior-comparison', action='store_true', help='Skip prior variant comparison')
    parser.add_argument('--skip-path-level', action='store_true', help='Skip path-level beam search evaluation')
    parser.add_argument('--full-vocab-eval', action='store_true', help='Use full vocabulary evaluation (disables parallel speedups)')
    parser.add_argument('--disable-filters', action='store_true', help='Disable filtering (disables some optimizations)')
    parser.add_argument('--pure-model-eval', action='store_true', help='Zero out heuristic weights (disables some optimizations)')
    args = parser.parse_args()

    # Load predictor and sequences
    predictor = load_predictor(data_source=DEFAULT_DATA_SOURCE, checkpoint_path=args.checkpoint)
    # Ensure worker processes can reload the checkpoint
    if args.checkpoint:
        predictor._checkpoint_path = args.checkpoint

    sequences = load_sequences_for_source(DEFAULT_DATA_SOURCE)
    metadata = load_sequence_metadata_for_source(DEFAULT_DATA_SOURCE)

    # Run evaluation
    results = run_evaluation(
        predictor,
        sequences,
        output_file=args.output_file,
        sequence_metadata=metadata,
        skip_ablation=args.skip_ablation,
        skip_prior_comparison=args.skip_prior_comparison,
        skip_path_level=args.skip_path_level,
        full_vocab_eval=args.full_vocab_eval,
        disable_filters=args.disable_filters,
        pure_model_eval=args.pure_model_eval,
        include_baseline=True,
        n_workers=args.n_workers,  # auto when None
    )
    
    print("Evaluation complete!")


def generate_plotting_csvs(
    evaluator,
    test_sequences: List[List[str]],
    test_contexts: Optional[List[Optional[Dict[str, Optional[str]]]]] = None,
    k_values: List[int] = [1, 3, 5, 10, 20, 50],
    output_dir: str = "outputs/evaluation_plots",
    use_scenario_priors: Optional[bool] = None,
) -> Dict[str, str]:
    """
    Generate CSV files ready for plotting from evaluation results.
    
    Returns:
        Dictionary mapping plot name to CSV file path
    """
    from pathlib import Path
    from collections import defaultdict, Counter
    import csv
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Generating Plotting Data")
    print(f"{'='*60}")
    
    # Initialize data structures
    per_prediction_results = []
    per_technique_stats = defaultdict(lambda: {'correct': 0, 'total_as_target': 0, 'total_predicted': 0})
    per_sequence_length_stats = defaultdict(lambda: {
        'count': 0,
        'hits_at_k': {k: 0 for k in k_values},
        'mrr_sum': 0.0
    })
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    confidence_bins = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # Metrics for each K
    top_k_hits = {k: [] for k in k_values}
    top_k_precisions = {k: [] for k in k_values}
    top_k_recalls = {k: [] for k in k_values}
    mrr_values = []
    
    # Prepare contexts
    if test_contexts is None:
        test_contexts = [None] * len(test_sequences)
    
    print(f"Processing {len(test_sequences)} test sequences...")
    
    # Evaluate each prediction
    for seq_idx, sequence in enumerate(tqdm(test_sequences, desc="Computing detailed metrics")):
        if len(sequence) < 2:
            continue
        
        context = test_contexts[seq_idx] if seq_idx < len(test_contexts) else None
        seq_length = len(sequence)
        
        for step_idx in range(1, len(sequence)):
            history = sequence[:step_idx]
            actual_next = sequence[step_idx]
            
            # Get predictions
            predictions = evaluator.predictor.next_probabilities(
                history,
                group_context=context,
                top_n_candidates=max(k_values) if k_values else 50,
                use_scenario_priors=use_scenario_priors,
            )
            
            if not predictions:
                continue
            
            # Extract technique IDs and probabilities
            predicted_techniques = [pred[0] for pred in predictions]
            predicted_probs = [pred[1] for pred in predictions]
            
            # Find rank of correct technique
            if actual_next in predicted_techniques:
                actual_rank = predicted_techniques.index(actual_next) + 1
                reciprocal_rank = 1.0 / actual_rank
                top_prob = predicted_probs[predicted_techniques.index(actual_next)]
            else:
                actual_rank = None
                reciprocal_rank = 0.0
                top_prob = 0.0
            
            # Update per-prediction results
            top_k_preds = []
            for rank, (tech, prob, _) in enumerate(predictions[:max(k_values)], 1):
                top_k_preds.append({
                    'technique': tech,
                    'probability': float(prob),
                    'rank': rank
                })
            
            per_prediction_results.append({
                'sequence_id': seq_idx,
                'step_idx': step_idx,
                'history': ','.join(history),
                'history_length': len(history),
                'actual_next': actual_next,
                'predicted_rank': actual_rank if actual_rank else -1,
                'reciprocal_rank': reciprocal_rank,
                'sequence_length': seq_length,
                'top_predictions': top_k_preds[:10]  # Store top 10
            })
            
            # Update metrics
            mrr_values.append(reciprocal_rank)
            
            for k in k_values:
                hit = 1.0 if actual_next in predicted_techniques[:k] else 0.0
                top_k_hits[k].append(hit)
                
                # Precision@K = # relevant in top-K / K
                precision = hit / k  # Binary relevance: either 1/k or 0
                top_k_precisions[k].append(precision)
                
                # Recall@K = # relevant in top-K / total relevant
                recall = hit  # Only 1 relevant item (next technique)
                top_k_recalls[k].append(recall)
                
                # Update per-sequence-length stats
                per_sequence_length_stats[seq_length]['hits_at_k'][k] += int(hit)
            
            per_sequence_length_stats[seq_length]['count'] += 1
            per_sequence_length_stats[seq_length]['mrr_sum'] += reciprocal_rank
            
            # Update per-technique stats
            per_technique_stats[actual_next]['total_as_target'] += 1
            if actual_rank and actual_rank <= 1:  # Top-1 correct
                per_technique_stats[actual_next]['correct'] += 1
            
            # Confusion matrix (top-1 prediction vs actual)
            if predicted_techniques:
                top_1_pred = predicted_techniques[0]
                confusion_matrix[actual_next][top_1_pred] += 1
                per_technique_stats[top_1_pred]['total_predicted'] += 1
            
            # Confidence calibration
            if predicted_techniques and actual_next in predicted_techniques:
                prob_bin = int(top_prob * 10) / 10.0  # 0.0, 0.1, 0.2, ..., 0.9
                confidence_bins[prob_bin]['correct'] += 1
                confidence_bins[prob_bin]['total'] += 1
            elif predicted_techniques:
                top_1_prob = predicted_probs[0]
                prob_bin = int(top_1_prob * 10) / 10.0
                confidence_bins[prob_bin]['total'] += 1
    
    print(f"✓ Processed {len(per_prediction_results)} predictions")
    
    # Generate CSV files
    csv_files = {}
    
    # 1. Top-K Metrics CSV
    print("Generating top_k_metrics.csv...")
    top_k_csv = output_path / "top_k_metrics.csv"
    with open(top_k_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['k', 'accuracy', 'precision', 'recall', 'f1', 'sample_size'])
        for k in k_values:
            acc = np.mean(top_k_hits[k]) if top_k_hits[k] else 0.0
            prec = np.mean(top_k_precisions[k]) if top_k_precisions[k] else 0.0
            rec = np.mean(top_k_recalls[k]) if top_k_recalls[k] else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            writer.writerow([k, f"{acc:.6f}", f"{prec:.6f}", f"{rec:.6f}", f"{f1:.6f}", len(top_k_hits[k])])
    csv_files['top_k_metrics'] = str(top_k_csv)
    
    # 2. Per-Technique Metrics CSV
    print("Generating per_technique_metrics.csv...")
    per_tech_csv = output_path / "per_technique_metrics.csv"
    with open(per_tech_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['technique_id', 'frequency_as_target', 'correct_predictions', 'accuracy',
                        'total_predicted', 'precision', 'recall', 'f1'])
        
        # Sort by frequency
        sorted_techniques = sorted(per_technique_stats.items(),
                                  key=lambda x: x[1]['total_as_target'], reverse=True)
        
        for tech, stats in sorted_techniques:
            freq = stats['total_as_target']
            correct = stats['correct']
            acc = correct / freq if freq > 0 else 0.0
            predicted = stats['total_predicted']
            
            prec = correct / predicted if predicted > 0 else 0.0
            rec = correct / freq if freq > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            writer.writerow([tech, freq, correct, f"{acc:.6f}", predicted,
                           f"{prec:.6f}", f"{rec:.6f}", f"{f1:.6f}"])
    csv_files['per_technique_metrics'] = str(per_tech_csv)
    
    # 3. Sequence Length Metrics CSV
    print("Generating sequence_length_metrics.csv...")
    seq_len_csv = output_path / "sequence_length_metrics.csv"
    with open(seq_len_csv, 'w', newline='') as f:
        # Create columns for different K values
        k_cols = [f'accuracy_top{k}' for k in k_values[:5]]  # Top 5 K values
        writer = csv.writer(f)
        writer.writerow(['sequence_length', 'count'] + k_cols + ['mrr'])
        
        sorted_lengths = sorted(per_sequence_length_stats.items())
        for length, stats in sorted_lengths:
            count = stats['count']
            mrr_avg = stats['mrr_sum'] / count if count > 0 else 0.0
            
            row = [length, count]
            for k in k_values[:5]:
                acc = stats['hits_at_k'][k] / count if count > 0 else 0.0
                row.append(f"{acc:.6f}")
            row.append(f"{mrr_avg:.6f}")
            
            writer.writerow(row)
    csv_files['sequence_length_metrics'] = str(seq_len_csv)
    
    # 4. Confusion Matrix CSV (Top 20 techniques)
    print("Generating confusion_matrix_top20.csv...")
    top_20_techniques = [tech for tech, _ in sorted(per_technique_stats.items(),
                                                     key=lambda x: x[1]['total_as_target'],
                                                     reverse=True)[:20]]
    
    conf_matrix_csv = output_path / "confusion_matrix_top20.csv"
    with open(conf_matrix_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['actual', 'predicted', 'count'])
        
        for actual in top_20_techniques:
            for predicted in top_20_techniques:
                count = confusion_matrix[actual][predicted]
                if count > 0:
                    writer.writerow([actual, predicted, count])
    csv_files['confusion_matrix'] = str(conf_matrix_csv)
    
    # 5. Confidence Calibration CSV
    print("Generating confidence_calibration.csv...")
    conf_cal_csv = output_path / "confidence_calibration.csv"
    with open(conf_cal_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['probability_bin', 'accuracy', 'count'])
        
        for bin_val in sorted(confidence_bins.keys()):
            stats = confidence_bins[bin_val]
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            writer.writerow([f"{bin_val:.1f}", f"{acc:.6f}", stats['total']])
    csv_files['confidence_calibration'] = str(conf_cal_csv)
    
    # 6. Cumulative MRR CSV
    print("Generating cumulative_mrr.csv...")
    cumulative_mrr_csv = output_path / "cumulative_mrr.csv"
    with open(cumulative_mrr_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prediction_count', 'cumulative_mrr'])
        
        cumsum = 0.0
        step = max(1, len(mrr_values) // 100)  # 100 data points
        for i in range(step, len(mrr_values) + 1, step):
            cumsum = np.mean(mrr_values[:i])
            writer.writerow([i, f"{cumsum:.6f}"])
    csv_files['cumulative_mrr'] = str(cumulative_mrr_csv)
    
    # 7. Per-Prediction Results CSV (optional, can be large)
    print("Generating per_prediction_results.csv (detailed)...")
    per_pred_csv = output_path / "per_prediction_results.csv"
    with open(per_pred_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence_id', 'step_idx', 'history', 'history_length', 'actual_next',
                        'predicted_rank', 'reciprocal_rank', 'sequence_length',
                        'top1_pred', 'top1_prob', 'top5_preds'])
        
        for result in per_prediction_results:
            top_preds = result['top_predictions']
            top1 = top_preds[0] if top_preds else {'technique': 'N/A', 'probability': 0.0}
            top5_str = ';'.join([f"{p['technique']}({p['probability']:.3f})"
                                for p in top_preds[:5]])
            
            writer.writerow([
                result['sequence_id'],
                result['step_idx'],
                result['history'],
                result['history_length'],
                result['actual_next'],
                result['predicted_rank'],
                f"{result['reciprocal_rank']:.6f}",
                result['sequence_length'],
                top1['technique'],
                f"{top1['probability']:.6f}",
                top5_str
            ])
    csv_files['per_prediction_results'] = str(per_pred_csv)
    
    print(f"\n{'='*60}")
    print("CSV Generation Complete")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    for name, path in csv_files.items():
        print(f"  • {name}: {Path(path).name}")
    
    return csv_files


if __name__ == "__main__":
    main()
