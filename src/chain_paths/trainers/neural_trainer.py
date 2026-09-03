"""Neural model trainer for PyTorch-based predictors."""

from __future__ import annotations

import copy
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable=None, **kwargs):
        return iterable or []

from ..models.base import BasePredictor
from ..models.torch_base import TorchBackedPredictor
from .base import BaseTrainer


class EarlyStoppingTracker:
    """Tracks validation metrics and determines when to stop training."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0001,
        metric: str = 'loss',
        mode: str = 'min',
    ):
        """Initialize early stopping tracker.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            metric: Metric to monitor ('loss', 'mrr', 'f1', 'recall', 'precision')
            mode: 'min' for loss, 'max' for other metrics
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.counter = 0
        self.best_state_dict = None
        self.history: List[Dict[str, float]] = []
        
    def _is_improvement(self, current_value: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == 'min':
            return current_value < (self.best_value - self.min_delta)
        else:  # mode == 'max'
            return current_value > (self.best_value + self.min_delta)
    
    def update(
        self,
        metrics: Dict[str, float],
        epoch: int,
        model_state_dict: dict,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Update tracker with new metrics.
        
        Args:
            metrics: Dictionary of metric values
            epoch: Current epoch number
            model_state_dict: Current model state dictionary
            
        Returns:
            Tuple of (should_stop, info_dict)
        """
        current_value = metrics.get(self.metric, float('inf') if self.mode == 'min' else float('-inf'))
        
        # Store history
        self.history.append({
            'epoch': epoch,
            **metrics,
        })
        
        # Check if this is an improvement
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model_state_dict)
            
            info = {
                'improved': True,
                'best_value': self.best_value,
                'best_epoch': self.best_epoch,
            }
            return False, info
        else:
            self.counter += 1
            info = {
                'improved': False,
                'counter': self.counter,
                'patience': self.patience,
                'best_value': self.best_value,
                'best_epoch': self.best_epoch,
            }
            
            if self.counter >= self.patience:
                return True, info  # Stop training
            return False, info
    
    def get_best_weights(self):
        """Return the best model weights."""
        return self.best_state_dict


class NeuralTrainer(BaseTrainer):
    """Trainer for neural network models (BiLSTM, GRU, TCN, Transformer).
    
    Handles batching, padding, training loops, and validation for PyTorch models.
    """

    def __init__(
        self,
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
    ):
        """Initialize neural trainer.
        
        Args:
            batch_size: Training batch size
            epochs: Maximum number of training epochs
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cuda', 'cpu', or None for auto)
            early_stopping: Whether to use early stopping
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            validation_split: Fraction of training data for validation (0.15 = 15%)
            restore_best_weights: Whether to restore best model weights
            early_stopping_metric: Metric to monitor ('loss', 'mrr', 'f1', 'recall', 'precision')
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.validation_split = validation_split
        self.restore_best_weights = restore_best_weights
        self.early_stopping_metric = early_stopping_metric
        self.num_workers = num_workers

    def needs_training(self, model: BasePredictor) -> bool:
        """Check if model needs training.
        
        Neural models always need training unless they have a trained model attribute.
        """
        if not isinstance(model, TorchBackedPredictor):
            return False
        
        # Check if model has a trained PyTorch module
        if hasattr(model, 'model') and model.model is not None:
            # Check if it's the fallback (not a real neural model)
            if not model._torch_available:
                return True
            # For now, assume neural models always need training
            # In production, you might check if weights are initialized
            return True
        
        return True

    def train(
        self,
        model: BasePredictor,
        sequences: Sequence[Sequence[str]],
        **kwargs: Any,
    ) -> BasePredictor:
        """Train a neural model.
        
        Args:
            model: Neural model instance to train
            sequences: Training sequences
            **kwargs: Additional training parameters
            
        Returns:
            Trained model instance
        """
        if not isinstance(model, TorchBackedPredictor):
            raise ValueError(f"NeuralTrainer can only train TorchBackedPredictor models, got {type(model)}")
        
        if not model._torch_available:
            raise RuntimeError("PyTorch is not available. Cannot train neural models.")
        
        torch = model._torch
        device = self._get_device(torch)
        pin_memory = torch.cuda.is_available()
        
        # Prepare training data
        all_examples = self._prepare_training_examples(sequences, model)
        
        if not all_examples:
            raise ValueError("No training examples generated from sequences")
        
        # Split into train/validation if early stopping enabled
        if self.early_stopping:
            train_examples, val_examples = self._split_train_val(all_examples)
            print(f"Early stopping enabled: {len(train_examples)} train, {len(val_examples)} validation examples")
            print(f"  Monitoring: {self.early_stopping_metric}, Patience: {self.patience}, Min delta: {self.min_delta}")
        else:
            train_examples = all_examples
            val_examples = []
        
        # Build model if not already built
        if model.model is None:
            model.model = model._build_model() if hasattr(model, '_build_model') else None
        
        if model.model is None:
            raise RuntimeError("Failed to build model architecture")
        
        model.model = model.model.to(device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(model.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        batch_size = kwargs.get('batch_size', self.batch_size)
        epochs = kwargs.get('epochs', self.epochs)

        num_batches_total = (len(train_examples) + batch_size - 1) // batch_size
        print(f"Training {model.get_model_name()} model: {len(train_examples)} examples, {num_batches_total} batches/epoch, {epochs} epochs")
        effective_workers = 0
        if self.num_workers and self.num_workers > 0:
            cpu_cap = max(1, (os.cpu_count() or 1) - 1)
            effective_workers = min(self.num_workers, cpu_cap)
            if effective_workers < self.num_workers:
                print(f"  ⚠ Reducing num_workers from {self.num_workers} to {effective_workers} based on CPU availability")
        
        # Initialize early stopping tracker
        early_stopping_tracker = None
        if self.early_stopping and val_examples:
            mode = 'min' if self.early_stopping_metric == 'loss' else 'max'
            early_stopping_tracker = EarlyStoppingTracker(
                patience=self.patience,
                min_delta=self.min_delta,
                metric=self.early_stopping_metric,
                mode=mode,
            )
        
        # Track epoch times for ETA calculation
        epoch_times = []
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0.0
            n_batches = 0
            
            random.shuffle(train_examples)

            # Optional parallel DataLoader for batch prep
            use_dataloader = effective_workers > 0
            if use_dataloader:
                torch = model._torch
                class _SeqDataset(torch.utils.data.Dataset):
                    def __init__(self, examples):
                        self.examples = examples
                    def __len__(self):
                        return len(self.examples)
                    def __getitem__(self, idx):
                        return self.examples[idx]
                def _collate(batch):
                    return self._prepare_batch(batch, model, device)
                ds = _SeqDataset(train_examples)
                loader = torch.utils.data.DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=effective_workers,
                    collate_fn=_collate,
                    pin_memory=pin_memory,
                )
                try:
                    iter_loader = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", disable=False)
                except TypeError:
                    iter_loader = loader
            else:
                batch_indices = range(0, len(train_examples), batch_size)
                try:
                    batch_indices = tqdm(batch_indices, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", disable=False)
                except TypeError:
                    pass  # tqdm fallback
                def _simple_iter():
                    for bs in batch_indices:
                        yield self._prepare_batch(train_examples[bs:bs + batch_size], model, device)
                iter_loader = _simple_iter()

            for histories, targets in iter_loader:
                if histories is None or targets is None:
                    continue

                optimizer.zero_grad()
                logits = model.model(histories)

                if len(logits.shape) == 3:  # (batch, seq_len, vocab)
                    logits = logits[:, -1, :]

                loss = criterion(logits, targets)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
            
            # Evaluate on validation set if early stopping enabled
            val_metrics = {}
            stop_training = False
            early_stop_info = {}
            
            if early_stopping_tracker is not None and val_examples:
                # Calculate validation metrics
                val_metrics = self._calculate_metrics(model, val_examples, device)
                
                # Check early stopping
                stop_training, early_stop_info = early_stopping_tracker.update(
                    metrics=val_metrics,
                    epoch=epoch,
                    model_state_dict=model.model.state_dict(),
                )
            
            # Print epoch summary
            remaining_epochs = epochs - epoch - 1
            if val_metrics:
                # Print with validation metrics
                metrics_str = f"Train Loss: {avg_loss:.4f}"
                metrics_str += f" | Val Loss: {val_metrics['loss']:.4f}"
                metrics_str += f" | MRR: {val_metrics['mrr']:.4f}"
                metrics_str += f" | F1: {val_metrics['f1']:.4f}"
                metrics_str += f" | Recall@5: {val_metrics['recall_at_k']:.4f}"
                metrics_str += f" | Hit@1: {val_metrics['hit_at_1']:.4f}"
                
                if epoch_times and remaining_epochs > 0:
                    avg_epoch_time = sum(epoch_times) / len(epoch_times)
                    eta_seconds = avg_epoch_time * remaining_epochs
                    eta_min = int(eta_seconds / 60)
                    eta_sec = int(eta_seconds % 60)
                    print(f"  Epoch {epoch + 1}/{epochs}: {metrics_str}, Time: {epoch_time:.1f}s, ETA: {eta_min}m {eta_sec}s")
                else:
                    print(f"  Epoch {epoch + 1}/{epochs}: {metrics_str}, Time: {epoch_time:.1f}s")
                
                # Print early stopping info
                if early_stop_info.get('improved'):
                    print(f"    ✓ New best {self.early_stopping_metric}: {early_stop_info['best_value']:.4f}")
                elif not stop_training:
                    counter = early_stop_info.get('counter', 0)
                    patience = early_stop_info.get('patience', self.patience)
                    print(f"    ⚠ No improvement ({counter}/{patience})")
            else:
                # Print without validation metrics
                if epoch_times and remaining_epochs > 0:
                    avg_epoch_time = sum(epoch_times) / len(epoch_times)
                    eta_seconds = avg_epoch_time * remaining_epochs
                    eta_min = int(eta_seconds / 60)
                    eta_sec = int(eta_seconds % 60)
                    print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s, ETA: {eta_min}m {eta_sec}s")
                else:
                    print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")
            
            # Check if we should stop early
            if stop_training:
                print(f"\n  🛑 Early stopping triggered at epoch {epoch + 1}")
                print(f"  📊 Best {self.early_stopping_metric}: {early_stop_info['best_value']:.4f} at epoch {early_stop_info['best_epoch'] + 1}")
                break
        
        # Restore best weights if early stopping was used
        if early_stopping_tracker is not None and self.restore_best_weights:
            if early_stopping_tracker.best_state_dict is not None:
                model.model.load_state_dict(early_stopping_tracker.best_state_dict)
                print(f"  ✓ Restored best model weights from epoch {early_stopping_tracker.best_epoch + 1}")
        
        total_time = time.time() - start_time
        total_min = int(total_time / 60)
        total_sec = int(total_time % 60)
        print(f"Training complete for {model.get_model_name()} ({total_min}m {total_sec}s total)")
        return model

    def _split_train_val(
        self,
        train_examples: List[Tuple[List[str], str]],
    ) -> Tuple[List[Tuple[List[str], str]], List[Tuple[List[str], str]]]:
        """Split training examples into train and validation sets.
        
        Args:
            train_examples: All training examples
            
        Returns:
            Tuple of (train_examples, val_examples)
        """
        if not self.early_stopping or self.validation_split <= 0:
            return train_examples, []
        
        n_val = int(len(train_examples) * self.validation_split)
        if n_val == 0:
            return train_examples, []
        
        # Shuffle before split for randomness
        random.seed(42)  # For reproducibility
        shuffled = train_examples.copy()
        random.shuffle(shuffled)
        
        val_examples = shuffled[:n_val]
        train_examples = shuffled[n_val:]
        
        return train_examples, val_examples

    def _calculate_metrics(
        self,
        model: BasePredictor,
        examples: List[Tuple[List[str], str]],
        device: Any,
        k: int = 5,
    ) -> Dict[str, float]:
        """Calculate comprehensive validation metrics.
        
        Args:
            model: Model instance
            examples: Validation examples
            device: PyTorch device
            k: Top-k for metrics calculation
            
        Returns:
            Dictionary of metrics
        """
        if not examples:
            return {}
        
        torch = model._torch
        criterion = torch.nn.CrossEntropyLoss()
        
        model.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        # Metrics accumulators
        mrr_scores = []
        recall_at_k = []
        precision_at_k = []
        hits_at_1 = []
        hits_at_k = []
        
        with torch.no_grad():
            for batch_start in range(0, len(examples), self.batch_size):
                batch = examples[batch_start:batch_start + self.batch_size]
                histories, targets = self._prepare_batch(batch, model, device)
                
                if histories is None or targets is None:
                    continue
                
                logits = model.model(histories)
                if len(logits.shape) == 3:
                    logits = logits[:, -1, :]
                
                # Calculate loss
                loss = criterion(logits, targets)
                total_loss += loss.item()
                n_batches += 1
                
                # Calculate ranking metrics
                probs = torch.softmax(logits, dim=-1)
                
                for i, target_idx in enumerate(targets.cpu().numpy()):
                    pred_probs = probs[i].cpu().numpy()
                    
                    # Rank all techniques by probability
                    ranked_indices = np.argsort(-pred_probs)  # Descending order
                    
                    # Find position of target
                    target_position = np.where(ranked_indices == target_idx)[0]
                    if len(target_position) > 0:
                        rank = target_position[0] + 1  # 1-indexed rank
                        
                        # MRR (Mean Reciprocal Rank)
                        mrr_scores.append(1.0 / rank)
                        
                        # Hit@1 and Hit@k
                        hits_at_1.append(1.0 if rank == 1 else 0.0)
                        hits_at_k.append(1.0 if rank <= k else 0.0)
                        
                        # Recall@k: was target in top-k?
                        recall_at_k.append(1.0 if rank <= k else 0.0)
                        
                        # Precision@k: 1/k if target in top-k, else 0
                        precision_at_k.append(1.0 / k if rank <= k else 0.0)
        
        model.model.train()
        
        # Aggregate metrics
        metrics = {
            'loss': total_loss / n_batches if n_batches > 0 else float('inf'),
            'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
            'recall_at_k': np.mean(recall_at_k) if recall_at_k else 0.0,
            'precision_at_k': np.mean(precision_at_k) if precision_at_k else 0.0,
            'hit_at_1': np.mean(hits_at_1) if hits_at_1 else 0.0,
            'hit_at_k': np.mean(hits_at_k) if hits_at_k else 0.0,
        }
        
        # Calculate F1 score from precision and recall
        if metrics['precision_at_k'] + metrics['recall_at_k'] > 0:
            metrics['f1'] = 2 * (metrics['precision_at_k'] * metrics['recall_at_k']) / \
                           (metrics['precision_at_k'] + metrics['recall_at_k'])
        else:
            metrics['f1'] = 0.0
        
        return metrics

    def _get_device(self, torch) -> Any:
        """Get the appropriate device for training."""
        if self.device:
            return torch.device(self.device)
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def _prepare_training_examples(
        self,
        sequences: Sequence[Sequence[str]],
        model: BasePredictor,
    ) -> List[tuple[List[str], str]]:
        """Prepare training examples from sequences.
        
        Args:
            sequences: List of technique sequences
            model: Model instance (for vocabulary access)
            
        Returns:
            List of (history, target) tuples
        """
        examples = []
        for seq in sequences:
            if len(seq) < 2:
                continue
            for i in range(1, len(seq)):
                history = list(seq[:i])
                target = seq[i]
                # Only include if target is in vocabulary
                if target in model._index:
                    examples.append((history, target))
        return examples

    def _prepare_batch(
        self,
        batch: List[tuple[List[str], str]],
        model: BasePredictor,
        device: Any,
    ) -> tuple[Any, Any]:
        """Prepare a batch for training.
        
        Args:
            batch: List of (history, target) tuples
            model: Model instance
            device: PyTorch device
            
        Returns:
            Tuple of (history_tensor, target_tensor) or (None, None) if error
        """
        if not batch:
            return None, None
        
        torch = model._torch
        
        # Find max length for padding
        max_len = max(len(h) for h, _ in batch)
        
        histories = []
        targets = []
        
        for history, target in batch:
            # Convert to indices
            hist_indices = [model._index.get(t, 0) for t in history]
            # Pad to max length
            hist_indices = hist_indices + [0] * (max_len - len(hist_indices))
            histories.append(hist_indices)
            targets.append(model._index[target])
        
        try:
            hist_tensor = torch.LongTensor(histories).to(device)
            target_tensor = torch.LongTensor(targets).to(device)
            return hist_tensor, target_tensor
        except Exception as e:
            print(f"Warning: Failed to prepare batch: {e}")
            return None, None


__all__ = ["NeuralTrainer"]

