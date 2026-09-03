"""Factory for creating model instances."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .base import BasePredictor
from .ngram import NGramPredictor
from .hmm import HMMPredictor, HMMConfig
from .bilstm import BiLSTMPredictor, BiLSTMConfig
from .lstm import LSTMPredictor, LSTMConfig
from .gru import GRUPredictor, GRUConfig
from .tcn import TCNPredictor, TCNConfig
from .transformer import TransformerPredictor, TransformerConfig


class ModelFactory:
    """Factory for creating and initializing model instances.
    
    This factory handles the creation of different model types (n-gram, HMM,
    neural architectures) with their specific initialization requirements.
    """

    @staticmethod
    def create_model(
        model_type: str,
        technique_ids: Sequence[str],
        **kwargs: Any,
    ) -> BasePredictor:
        """Create a model instance of the specified type.
        
        Args:
            model_type: Type of model to create ('ngram', 'hmm', 'bilstm', 'lstm', 'gru', 'tcn', 'transformer')
            technique_ids: List of all technique IDs in the vocabulary
            **kwargs: Model-specific initialization parameters
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If model_type is not recognized
        """
        model_type = model_type.lower()
        
        if model_type == 'ngram':
            return ModelFactory._create_ngram(technique_ids, **kwargs)
        elif model_type == 'hmm':
            return ModelFactory._create_hmm(technique_ids, **kwargs)
        elif model_type == 'bilstm':
            return ModelFactory._create_bilstm(technique_ids, **kwargs)
        elif model_type == 'lstm':
            return ModelFactory._create_lstm(technique_ids, **kwargs)
        elif model_type == 'gru':
            return ModelFactory._create_gru(technique_ids, **kwargs)
        elif model_type == 'tcn':
            return ModelFactory._create_tcn(technique_ids, **kwargs)
        elif model_type == 'transformer':
            return ModelFactory._create_transformer(technique_ids, **kwargs)
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Supported types: ngram, hmm, bilstm, lstm, gru, tcn, transformer"
            )

    @staticmethod
    def _create_ngram(technique_ids: Sequence[str], **kwargs: Any) -> BasePredictor:
        """Create an N-gram model."""
        required = ['counts_data', 'ps_scores', 'sigma_scores', 'embeddings', 
                   'tech_to_idx', 'index', 'params']
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters for NGramPredictor: {missing}")
        
        return NGramPredictor(
            counts_data=kwargs['counts_data'],
            ps_scores=kwargs['ps_scores'],
            sigma_scores=kwargs['sigma_scores'],
            embeddings=kwargs['embeddings'],
            tech_to_idx=kwargs['tech_to_idx'],
            index=kwargs['index'],
            params=kwargs['params'],
        )

    @staticmethod
    def _create_hmm(technique_ids: Sequence[str], **kwargs: Any) -> BasePredictor:
        """Create an HMM model."""
        if 'sequences' in kwargs:
            # Train from sequences
            config = kwargs.get('config', HMMConfig())
            return HMMPredictor.from_sequences(
                sequences=kwargs['sequences'],
                technique_ids=technique_ids,
                config=config,
            )
        elif 'transition_log_probs' in kwargs and 'initial_log_probs' in kwargs:
            # Initialize from pre-computed probabilities
            config = kwargs.get('config', HMMConfig())
            return HMMPredictor(
                technique_ids=technique_ids,
                transition_log_probs=kwargs['transition_log_probs'],
                initial_log_probs=kwargs['initial_log_probs'],
                config=config,
            )
        else:
            raise ValueError(
                "HMMPredictor requires either 'sequences' for training or "
                "'transition_log_probs' and 'initial_log_probs' for initialization"
            )

    @staticmethod
    def _create_bilstm(technique_ids: Sequence[str], **kwargs: Any) -> BasePredictor:
        """Create a BiLSTM model."""
        config = kwargs.get('config', BiLSTMConfig())
        embeddings = kwargs.get('embeddings')
        tech_to_idx = kwargs.get('tech_to_idx')
        transition_log_probs = kwargs.get('transition_log_probs')
        
        if embeddings is None:
            raise ValueError("BiLSTMPredictor requires 'embeddings' parameter")
        if tech_to_idx is None:
            raise ValueError("BiLSTMPredictor requires 'tech_to_idx' mapping")
        
        return BiLSTMPredictor(
            technique_ids=technique_ids,
            embeddings=embeddings,
            tech_to_idx=tech_to_idx,
            transition_log_probs=transition_log_probs,
            config=config,
        )

    @staticmethod
    def _create_lstm(technique_ids: Sequence[str], **kwargs: Any) -> BasePredictor:
        """Create a unidirectional LSTM model."""
        config = kwargs.get('config', LSTMConfig())
        embeddings = kwargs.get('embeddings')
        tech_to_idx = kwargs.get('tech_to_idx')
        transition_log_probs = kwargs.get('transition_log_probs')

        if embeddings is None:
            raise ValueError("LSTMPredictor requires 'embeddings' parameter")
        if tech_to_idx is None:
            raise ValueError("LSTMPredictor requires 'tech_to_idx' mapping")

        return LSTMPredictor(
            technique_ids=technique_ids,
            embeddings=embeddings,
            tech_to_idx=tech_to_idx,
            transition_log_probs=transition_log_probs,
            config=config,
        )

    @staticmethod
    def _create_gru(technique_ids: Sequence[str], **kwargs: Any) -> BasePredictor:
        """Create a GRU model."""
        config = kwargs.get('config', GRUConfig())
        transition_log_probs = kwargs.get('transition_log_probs')
        
        return GRUPredictor(
            technique_ids=technique_ids,
            transition_log_probs=transition_log_probs,
            config=config,
        )

    @staticmethod
    def _create_tcn(technique_ids: Sequence[str], **kwargs: Any) -> BasePredictor:
        """Create a TCN model."""
        config = kwargs.get('config', TCNConfig())
        transition_log_probs = kwargs.get('transition_log_probs')
        
        return TCNPredictor(
            technique_ids=technique_ids,
            transition_log_probs=transition_log_probs,
            config=config,
        )

    @staticmethod
    def _create_transformer(technique_ids: Sequence[str], **kwargs: Any) -> BasePredictor:
        """Create a Transformer model."""
        config = kwargs.get('config', TransformerConfig())
        transition_log_probs = kwargs.get('transition_log_probs')
        tactic_mapping = kwargs.get('tactic_mapping')
        tactic_ids = kwargs.get('tactic_ids')

        return TransformerPredictor(
            technique_ids=technique_ids,
            transition_log_probs=transition_log_probs,
            config=config,
            tactic_mapping=tactic_mapping,
            tactic_ids=tactic_ids,
        )

    @staticmethod
    def get_available_models() -> list[str]:
        """Get list of available model types.
        
        Returns:
            List of model type strings
        """
        return ['ngram', 'hmm', 'bilstm', 'lstm', 'gru', 'tcn', 'transformer']


__all__ = ["ModelFactory"]

