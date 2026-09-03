"""Base trainer interface for model training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..models.base import BasePredictor


class BaseTrainer(ABC):
    """Abstract base class for model trainers.
    
    Trainers handle the training process for different model types, including
    data preparation, training loops, validation, and model persistence.
    """

    @abstractmethod
    def train(
        self,
        model: BasePredictor,
        sequences: Sequence[Sequence[str]],
        **kwargs: Any,
    ) -> BasePredictor:
        """Train a model on sequences.
        
        Args:
            model: Model instance to train
            sequences: Training sequences
            **kwargs: Training-specific parameters (learning rate, epochs, etc.)
            
        Returns:
            Trained model instance
        """
        pass

    @abstractmethod
    def needs_training(self, model: BasePredictor) -> bool:
        """Check if a model needs training.
        
        Args:
            model: Model instance to check
            
        Returns:
            True if model requires training, False if already trained
        """
        pass


__all__ = ["BaseTrainer"]

