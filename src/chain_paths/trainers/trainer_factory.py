"""Factory for creating appropriate trainers based on model type."""

from __future__ import annotations

from typing import Optional

from ..models.base import BasePredictor
from .base import BaseTrainer
from .neural_trainer import NeuralTrainer


class TrainerFactory:
    """Factory for creating trainers based on model type."""

    @staticmethod
    def create_trainer(
        model: BasePredictor,
        batch_size: int = 32,
        epochs: int = 10,
        learning_rate: float = 3e-4,
        device: Optional[str] = None,
        num_workers: int = 0,
        early_stopping: bool = False,
        patience: int = 5,
        min_delta: float = 0.0001,
        validation_split: float = 0.15,
        restore_best_weights: bool = True,
        early_stopping_metric: str = 'loss',
    ) -> Optional[BaseTrainer]:
        """Create an appropriate trainer for the given model.
        
        Args:
            model: Model instance to train
            batch_size: Training batch size (for neural models)
            epochs: Maximum number of training epochs (for neural models)
            learning_rate: Learning rate (for neural models)
            device: Device to train on (for neural models)
            early_stopping: Enable early stopping (for neural models)
            patience: Early stopping patience (for neural models)
            min_delta: Minimum improvement threshold (for neural models)
            validation_split: Validation data fraction (for neural models)
            restore_best_weights: Restore best weights after training (for neural models)
            early_stopping_metric: Metric to monitor for early stopping
            
        Returns:
            Trainer instance, or None if model doesn't need training
        """
        model_name = model.get_model_name()
        
        # Neural models need training
        if model_name in ['bilstm', 'lstm', 'gru', 'tcn', 'transformer']:
            return NeuralTrainer(
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                device=device,
                num_workers=num_workers,
                early_stopping=early_stopping,
                patience=patience,
                min_delta=min_delta,
                validation_split=validation_split,
                restore_best_weights=restore_best_weights,
                early_stopping_metric=early_stopping_metric,
            )
        
        # N-gram and HMM models don't need separate training
        # (they're trained via from_sequences or use pre-computed counts)
        elif model_name in ['ngram', 'hmm']:
            return None
        
        else:
            raise ValueError(f"Unknown model type for training: {model_name}")

    @staticmethod
    def model_needs_training(model: BasePredictor) -> bool:
        """Check if a model needs training.
        
        Args:
            model: Model instance to check
            
        Returns:
            True if model needs training, False otherwise
        """
        model_name = model.get_model_name()
        
        if model_name in ['bilstm', 'lstm', 'gru', 'tcn', 'transformer']:
            trainer = NeuralTrainer()
            return trainer.needs_training(model)
        
        return False


__all__ = ["TrainerFactory"]

