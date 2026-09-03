"""Model comparison framework for evaluating multiple models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from . import config as cfg
from .models.base import BasePredictor
from .models.factory import ModelFactory
from .predictor import Predictor
from .eval import Evaluator
from .trainers.trainer_factory import TrainerFactory
from .io import load_parquet, load_csv, load_numpy
from .embeddings import load_embeddings_and_index
from .tactics import load_tactic_mapping, load_tactic_transitions
from .platforms import load_platform_mapping
from .ontology_reasoning import OntologyReasoner


class ModelComparator:
    """Framework for training and comparing multiple models.
    
    This class provides methods to train all available models, evaluate them
    on a test set, and generate comparison reports.
    """

    def __init__(
        self,
        data_dir: Path = cfg.DATA_DIR,
        outputs_dir: Path = cfg.OUTPUTS_DIR,
    ):
        """Initialize model comparator.
        
        Args:
            data_dir: Directory containing data files
            outputs_dir: Directory for output files
        """
        self.data_dir = Path(data_dir)
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def train_all_models(
        self,
        sequences: Sequence[Sequence[str]],
        metadata: Optional[pd.DataFrame] = None,
        train_ratio: float = 0.8,
        model_types: Optional[List[str]] = None,
        fast_train: Optional[int] = None,
        random_seed: Optional[int] = None,
        **training_kwargs: Any,
    ) -> Tuple[Dict[str, BasePredictor], Dict[str, Any]]:
        """Train all specified models.
        
        Args:
            sequences: Training sequences
            metadata: Optional sequence metadata
            train_ratio: Ratio of sequences to use for training
            model_types: List of model types to train. If None, trains all available.
            fast_train: If set to an integer N, use only N examples for training (fast iteration).
                       If None, uses all available examples.
            random_seed: Random seed for reproducibility (uses config default if None)
            **training_kwargs: Additional training parameters
            
        Returns:
            Tuple of (trained_models_dict, split_info_dict)
        """
        from .eval import Evaluator
        import random
        import numpy as np
        from . import config as cfg
        
        if model_types is None:
            model_types = ModelFactory.get_available_models()
        
        # Set random seed for reproducibility
        if random_seed is None:
            random_seed = cfg.DEFAULT_PARAMS.get('RANDOM_SEED', 42)
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Use Evaluator.split_sequences for proper splitting
        # Create temporary evaluator just for splitting
        from .predictor import Predictor  # Dummy predictor for splitting
        temp_predictor = None  # Not needed for splitting
        temp_evaluator = Evaluator(temp_predictor, list(sequences), metadata)
        
        # Perform split (always use random split, no temporal)
        train_sequences, test_sequences = temp_evaluator.split_sequences(
            train_ratio=train_ratio,
            temporal_split=False,
            random_seed=random_seed
        )
        
        # Get indices from evaluator
        train_indices = temp_evaluator._last_split_indices.get('train', [])
        test_indices = temp_evaluator._last_split_indices.get('test', [])
        
        # Store split info
        split_info = {
            'train_indices': train_indices,
            'test_indices': test_indices,
            'train_ratio': train_ratio,
            'random_seed': random_seed,
            'n_train': len(train_sequences),
            'n_test': len(test_sequences),
        }
        
        print(f"Split info: {len(train_sequences)} train, {len(test_sequences)} test (seed={random_seed})")
        
        # Apply fast-train sampling if requested
        if fast_train is not None and fast_train > 0:
            if fast_train < len(train_sequences):
                # Stratified sampling to preserve technique distribution
                sampled_indices = random.sample(range(len(train_sequences)), min(fast_train, len(train_sequences)))
                train_sequences = [train_sequences[i] for i in sampled_indices]
                # Update train_indices to reflect sampling
                split_info['train_indices'] = [train_indices[i] for i in sampled_indices]
                split_info['fast_train_sampled'] = True
                print(f"Fast-train mode: using {len(train_sequences)} examples (sampled from {split_info['n_train']})")
        
        trained_models = {}
        
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"Training {model_type.upper()} model")
            print(f"{'='*60}")
            
            try:
                model = self._train_model(
                    model_type, train_sequences, metadata, **training_kwargs
                )
                trained_models[model_type] = model
                print(f"Successfully trained {model_type}")
            except Exception as e:
                print(f"Failed to train {model_type}: {e}")
                import traceback
                traceback.print_exc()
                # Don't add to trained_models so caller knows it failed
        
        return trained_models, split_info

    def _train_model(
        self,
        model_type: str,
        sequences: Sequence[Sequence[str]],
        metadata: Optional[pd.DataFrame],
        **kwargs: Any,
    ) -> BasePredictor:
        """Train a single model.
        
        Args:
            model_type: Type of model to train
            sequences: Training sequences
            metadata: Optional metadata
            **kwargs: Training parameters
            
        Returns:
            Trained model instance
        """
        # Get technique IDs from sequences
        all_techniques = set()
        for seq in sequences:
            all_techniques.update(seq)
        technique_ids = sorted(list(all_techniques))
        
        # Prepare model kwargs
        model_kwargs = {'technique_ids': technique_ids}
        
        if model_type == 'ngram':
            counts_data = {}
            
            sigma_scores = {}
            if cfg.SIGMA_STEALTH_CSV.exists():
                sigma_df = load_csv(cfg.SIGMA_STEALTH_CSV)
                sigma_scores = dict(zip(sigma_df['technique_id'], sigma_df['stealth_S']))

            ps_scores = {}
            
            embeddings, tech_to_idx, _ = load_embeddings_and_index(
                cfg.TECH_EMBEDDINGS_NPY,
                cfg.TECH_EMBEDDINGS_NPY.parent / "emb_index.faiss",
                cfg.TECH_INDEX_CSV
            )
            
            model_kwargs.update({
                'counts_data': counts_data,
                'ps_scores': ps_scores,
                'sigma_scores': sigma_scores,
                'embeddings': embeddings,
                'tech_to_idx': tech_to_idx,
                'index': None,  # FAISS removed - using embeddings only
                'params': cfg.DEFAULT_PARAMS,
            })
        
        elif model_type == 'hmm':
            model_kwargs['sequences'] = sequences
        
        elif model_type in ['bilstm', 'lstm', 'gru', 'tcn', 'transformer']:
            # Load embeddings for neural models
            if cfg.TECH_EMBEDDINGS_NPY.exists():
                embeddings, tech_to_idx, _ = load_embeddings_and_index(
                    cfg.TECH_EMBEDDINGS_NPY,
                    cfg.TECH_EMBEDDINGS_NPY.parent / "emb_index.faiss",
                    cfg.TECH_INDEX_CSV
                )
                # All neural models use embeddings for initialization
                model_kwargs['embeddings'] = embeddings
                model_kwargs['tech_to_idx'] = tech_to_idx
        
        # Create model
        model = ModelFactory.create_model(model_type, **model_kwargs)
        
        # Train if needed
        trainer = TrainerFactory.create_trainer(model, **kwargs)
        if trainer and trainer.needs_training(model):
            model = trainer.train(model, sequences, **kwargs)
        
        return model

    def evaluate_all_models(
        self,
        models: Dict[str, BasePredictor],
        test_sequences: Sequence[Sequence[str]],
        metadata: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all models on test sequences.
        
        Args:
            models: Dictionary of trained models
            test_sequences: Test sequences for evaluation
            metadata: Optional sequence metadata
            
        Returns:
            Dictionary mapping model names to evaluation results
        """
        results = {}
        
        for model_name, base_model in models.items():
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name.upper()} model")
            print(f"{'='*60}")
            
            try:
                # Create a Predictor wrapper around the base model
                predictor = self._create_predictor_for_model(base_model)
                
                # Evaluate
                evaluator = Evaluator(predictor, test_sequences, sequence_metadata=metadata)
                model_results = evaluator.evaluate_all()
                
                results[model_name] = model_results
                print(f"Completed evaluation for {model_name}")
                
            except Exception as e:
                print(f"Failed to evaluate {model_name}: {e}")
                import traceback
                traceback.print_exc()
                results[model_name] = {'error': str(e)}
        
        return results

    def _create_predictor_for_model(self, base_model: BasePredictor) -> Predictor:
        """Create a Predictor instance wrapping a base model.
        
        Args:
            base_model: Base model instance
            
        Returns:
            Predictor instance
        """
        # Counts intentionally not loaded — pure vocabulary scoring
        counts_data = {}
        
        sigma_scores = {}
        if cfg.SIGMA_STEALTH_CSV.exists():
            sigma_df = load_csv(cfg.SIGMA_STEALTH_CSV)
            sigma_scores = dict(zip(sigma_df['technique_id'], sigma_df['stealth_S']))

        ps_scores = {}

        tactic_mapping = load_tactic_mapping(cfg.TACTIC_MAPPING_JSON)
        tactic_transitions = load_tactic_transitions(cfg.TACTIC_TRANSITIONS_JSON)
        platform_mapping = load_platform_mapping(cfg.ENTERPRISE_ATTACK_JSON)

        ontology_reasoner = None
        try:
            if cfg.ONTOLOGY_JSON.exists():
                ontology_reasoner = OntologyReasoner.from_files(
                    cfg.ONTOLOGY_JSON, cfg.ONTOLOGY_RULES_JSON
                )
        except Exception:
            ontology_reasoner = None

        embeddings, tech_to_idx, index = load_embeddings_and_index(
            cfg.TECH_EMBEDDINGS_NPY,
            cfg.TECH_EMBEDDINGS_NPY.parent / "emb_index.faiss",
            cfg.TECH_INDEX_CSV
        )
        
        return Predictor(
            counts_data=counts_data,
            ps_scores=ps_scores,
            sigma_scores=sigma_scores,
            embeddings=embeddings,
            tech_to_idx=tech_to_idx,
            index=index,
            base_model=base_model,
            tactic_mapping=tactic_mapping,
            tactic_transitions=tactic_transitions,
            platform_mapping=platform_mapping,
            ontology_reasoner=ontology_reasoner,
        )

    def compare_results(
        self,
        results: Dict[str, Dict[str, Any]],
        output_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate comparison report from evaluation results.
        
        Args:
            results: Dictionary of evaluation results per model
            output_file: Optional path to save comparison report
            
        Returns:
            Comparison report dictionary
        """
        comparison = {
            'models': {},
            'summary': {},
        }
        
        # Extract metrics for each model
        for model_name, model_results in results.items():
            if 'error' in model_results:
                comparison['models'][model_name] = {'error': model_results['error']}
                continue
            
            metrics = model_results.get('next_step_metrics', {})
            path_metrics = model_results.get('path_metrics', {})
            
            comparison['models'][model_name] = {
                'mrr': metrics.get('MRR', {}).get('mean', 0.0),
                'precision_at_1': metrics.get('precision_at_k', {}).get('1', {}).get('mean', 0.0),
                'precision_at_5': metrics.get('precision_at_k', {}).get('5', {}).get('mean', 0.0),
                'precision_at_10': metrics.get('precision_at_k', {}).get('10', {}).get('mean', 0.0),
                'hit_at_1': path_metrics.get('Hit@1', 0.0),
                'hit_at_5': path_metrics.get('Hit@5', 0.0),
                'hit_at_10': path_metrics.get('Hit@10', 0.0),
            }
        
        # Generate summary
        if comparison['models']:
            mrr_scores = {
                name: data.get('mrr', 0.0)
                for name, data in comparison['models'].items()
                if 'error' not in data
            }
            if mrr_scores:
                best_model = max(mrr_scores.items(), key=lambda x: x[1])
                comparison['summary'] = {
                    'best_model': best_model[0],
                    'best_mrr': best_model[1],
                    'all_mrr': mrr_scores,
                }
        
        # Save if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\nComparison report saved to {output_file}")
        
        return comparison

    def select_best_model(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: str = 'MRR',
    ) -> Optional[str]:
        """Select the best model based on a metric.
        
        Args:
            results: Dictionary of evaluation results
            metric: Metric to use for selection ('MRR', 'precision_at_1', etc.)
            
        Returns:
            Name of the best model, or None if no valid results
        """
        model_scores = {}
        
        for model_name, model_results in results.items():
            if 'error' in model_results:
                continue
            
            if metric == 'MRR':
                score = model_results.get('next_step_metrics', {}).get('MRR', {}).get('mean', 0.0)
            elif metric.startswith('precision_at_'):
                k = metric.split('_')[-1]
                score = model_results.get('next_step_metrics', {}).get('precision_at_k', {}).get(k, {}).get('mean', 0.0)
            elif metric.startswith('Hit@'):
                k = metric.split('@')[-1]
                score = model_results.get('path_metrics', {}).get(f'Hit@{k}', 0.0)
            else:
                continue
            
            model_scores[model_name] = score
        
        if not model_scores:
            return None
        
        return max(model_scores.items(), key=lambda x: x[1])[0]


__all__ = ["ModelComparator"]

