"""Checkpoint utilities for saving/loading model artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import asdict, is_dataclass

import numpy as np

from .factory import ModelFactory
from .bilstm import BiLSTMConfig
from .lstm import LSTMConfig
from .gru import GRUConfig
from .tcn import TCNConfig
from .transformer import TransformerConfig
from .hmm import HMMConfig


_CONFIG_BUILDERS = {
    "bilstm": BiLSTMConfig,
    "lstm": LSTMConfig,
    "gru": GRUConfig,
    "tcn": TCNConfig,
    "transformer": TransformerConfig,
    "hmm": HMMConfig,
}


def infer_bilstm_config_from_state_dict(state_dict: Dict[str, Any]) -> BiLSTMConfig:
    """
    Infer BiLSTM architecture from raw state_dict tensor shapes.
    
    Args:
        state_dict: PyTorch state_dict (OrderedDict) with model weights
        
    Returns:
        BiLSTMConfig with inferred architecture parameters
        
    Raises:
        ValueError: If state_dict doesn't contain expected BiLSTM keys
    """
    required_keys = ['embedding.weight', 'output.weight']
    missing = [k for k in required_keys if k not in state_dict]
    if missing:
        raise ValueError(f"State dict missing required keys for BiLSTM: {missing}")
    
    try:
        # Extract dimensions from tensor shapes
        vocab_size = state_dict['embedding.weight'].shape[0]
        embedding_dim = state_dict['embedding.weight'].shape[1]
        hidden_size = state_dict['output.weight'].shape[1]
        
        # Count LSTM layers by checking weight keys
        num_layers = sum(1 for k in state_dict.keys() if k.startswith('lstm.weight_ih_l'))
        
        print(f"[*] Inferred BiLSTM config: vocab={vocab_size}, emb_dim={embedding_dim}, "
              f"hidden={hidden_size}, layers={num_layers}")
        
        return BiLSTMConfig(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.1,  # Default
            max_seq_len=256,  # Default
        )
    except (KeyError, IndexError, AttributeError) as e:
        raise ValueError(f"Failed to infer BiLSTM config from state_dict: {e}")


def _restore_config(model_type: str, cfg_payload: Any) -> Any:
    """Attempt to rebuild dataclass config from payload."""
    if cfg_payload is None:
        return None
    builder = _CONFIG_BUILDERS.get(model_type)
    if builder and isinstance(cfg_payload, dict):
        try:
            return builder(**cfg_payload)
        except Exception:
            return cfg_payload
    return cfg_payload


def save_model_checkpoint(
    model: Any,
    path: str | Path,
    model_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a model checkpoint with metadata.
    
    Args:
        model: The model instance to save
        path: Output path (.pkl or .pt)
        model_type: Model type identifier (bilstm, ngram, hmm, etc.)
        metadata: Optional training metadata (epochs, batch_size, etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint payload
    checkpoint = {
        'model_type': model_type,
        'technique_ids': list(getattr(model, 'technique_ids', [])),
        'metadata': metadata or {},
    }
    
    # Torch models: save state_dict
    try:
        import torch
        if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
            checkpoint['state_dict'] = {k: v.cpu() for k, v in model.model.state_dict().items()}
            cfg_obj = getattr(model, 'config', None)
            checkpoint['config'] = asdict(cfg_obj) if is_dataclass(cfg_obj) else cfg_obj
            checkpoint['tech_to_idx'] = getattr(model, 'tech_to_idx', None)
            checkpoint['is_torch'] = True
        else:
            checkpoint['is_torch'] = False
            checkpoint['model_obj'] = model
    except ImportError:
        checkpoint['is_torch'] = False
        checkpoint['model_obj'] = model
    
    # Save
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"[OK] Saved {model_type} checkpoint to {path}")


def load_model_checkpoint(
    path: str | Path,
    embeddings: Optional[np.ndarray] = None,
    tech_to_idx: Optional[Dict[str, int]] = None,
) -> Any:
    """
    Load a model checkpoint (auto-detects model type from file).
    
    Args:
        path: Path to checkpoint file
        embeddings: Optional embeddings (for torch models that need them)
        tech_to_idx: Optional technique index mapping
        
    Returns:
        Reconstructed model instance
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    # Log checkpoint info
    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[*] Loading checkpoint from: {path} ({file_size_mb:.1f} MB)")
    
    # For .pt files, try torch.load first (might be raw state_dict)
    if path.suffix == '.pt':
        try:
            import torch
            torch_payload = torch.load(path, map_location='cpu')
            
            # Check if it's a raw state_dict (OrderedDict without metadata)
            from collections import OrderedDict
            if isinstance(torch_payload, (dict, OrderedDict)) and 'model_type' not in torch_payload:
                # This is a raw state_dict - delegate to load_predictor logic
                # For now, we'll raise an informative error
                raise ValueError(
                    f"Raw state_dict detected in {path}. "
                    "Please use --checkpoint with load_predictor() via CLI, "
                    "or convert to full checkpoint format first."
                )
        except ImportError:
            pass  # Fall through to pickle loading
        except Exception as e:
            # If torch.load fails, fall through to pickle
            print(f"[*] torch.load failed, trying pickle: {e}")
    
    # Load checkpoint via pickle
    try:
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to unpickle checkpoint {path}: {e}")
    
    model_type = checkpoint.get('model_type')
    if not model_type:
        raise ValueError(f"Checkpoint missing 'model_type' field: {path}")
    
    print(f"[*] Checkpoint type: {model_type}")
    print(f"[*] Techniques in checkpoint: {len(checkpoint.get('technique_ids', []))}")
    
    # Non-torch models: return the pickled object directly
    if not checkpoint.get('is_torch', False):
        print(f"[*] Loading non-torch {model_type} model...")
        if 'model_obj' not in checkpoint:
            raise ValueError(f"Non-torch checkpoint missing 'model_obj' field: {path}")
        return checkpoint['model_obj']
    
    # Torch models: reconstruct from state_dict
    print(f"[*] Loading torch {model_type} model...")
    return load_torch_model(
        path=path,
        model_type=model_type,
        checkpoint=checkpoint,
        embeddings=embeddings,
        tech_to_idx=tech_to_idx,
    )


def load_torch_model(
    path: str | Path,
    model_type: Optional[str] = None,
    checkpoint: Optional[Dict] = None,
    embeddings: Optional[np.ndarray] = None,
    tech_to_idx: Optional[Dict[str, int]] = None,
    **model_kwargs: Any,
) -> Any:
    """
    Load a PyTorch model from checkpoint.
    
    Args:
        path: Checkpoint path
        model_type: Model type (bilstm, gru, tcn, transformer); auto-detected if None
        checkpoint: Pre-loaded checkpoint dict (optional)
        embeddings: Embeddings array
        tech_to_idx: Technique to index mapping
        **model_kwargs: Additional model creation kwargs
        
    Returns:
        Reconstructed model with loaded weights
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required to load torch models. "
            "Install: pip install torch"
        )
    
    # Load checkpoint if not provided
    if checkpoint is None:
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
    
    # Auto-detect model type
    if model_type is None:
        model_type = checkpoint.get('model_type', '').lower()
    if not model_type:
        raise ValueError(f"Cannot determine model_type from checkpoint: {path}")
    
    technique_ids = checkpoint.get('technique_ids')
    if not technique_ids:
        raise ValueError("technique_ids missing from checkpoint payload")
    
    state_dict = checkpoint.get('state_dict')
    if state_dict is None:
        raise ValueError(f"Checkpoint missing state_dict: {path}")
    
    cfg_payload = _restore_config(model_type, checkpoint.get('config'))
    if cfg_payload is not None:
        model_kwargs.setdefault('config', cfg_payload)
    
    # Use saved tech_to_idx if not provided
    if tech_to_idx is None:
        tech_to_idx = checkpoint.get('tech_to_idx')
    
    # Add required kwargs
    if embeddings is not None:
        model_kwargs.setdefault('embeddings', embeddings)
    if tech_to_idx is not None:
        model_kwargs.setdefault('tech_to_idx', tech_to_idx)
    
    # Build model skeleton
    model = ModelFactory.create_model(model_type, technique_ids, **model_kwargs)
    if not hasattr(model, "_build_model"):
        raise ValueError(f"Model type {model_type} is not torch-backed or lacks _build_model")
    
    # Instantiate and load weights
    model.model = model._build_model()
    if model.model is None:
        raise RuntimeError("Failed to build model while loading checkpoint")
    
    model.model.load_state_dict(state_dict)
    model.model.eval()
    
    print(f"[OK] Loaded {model_type} model with {len(technique_ids)} techniques")
    
    return model


__all__ = ["save_model_checkpoint", "load_model_checkpoint", "load_torch_model"]
