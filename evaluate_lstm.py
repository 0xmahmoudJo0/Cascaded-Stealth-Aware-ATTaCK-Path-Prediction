"""
Comprehensive Evaluation Script for Trained LSTM Model.

This script evaluates the LSTM model trained by train_lstm.py with metrics:
- Top-K Accuracy (Hit@1, Hit@3, Hit@5, Hit@10)
- Mean Reciprocal Rank (MRR)
- Precision@K
- Recall@K
- F1@K

Usage:
    python evaluate_lstm.py --checkpoint outputs/lstm_weights.pt --data data/sequences.parquet
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL ARCHITECTURE (must match train_lstm.py)
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM model matching the training architecture."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        last_output = lstm_out[:, -1, :]
        dropped = self.dropout(last_output)
        logits = self.output(dropped)
        return logits


# ============================================================================
# DATASET
# ============================================================================

class AttackSequenceDataset(Dataset):
    """PyTorch Dataset for attack technique sequences."""
    
    def __init__(self, sequences: List[List[int]], targets: List[int]):
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.targets = torch.as_tensor(targets, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class MetricsCalculator:
    """Calculate comprehensive evaluation metrics."""
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.k_values = k_values
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.all_predictions = []  # Top-K predictions for each sample
        self.all_targets = []
        self.all_probs = []  # Probabilities for MRR calculation
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch predictions.
        
        Args:
            logits: Model output logits (batch_size, vocab_size)
            targets: True target indices (batch_size,)
        """
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get top-K predictions (use max K)
        max_k = max(self.k_values)
        topk_probs, topk_indices = torch.topk(probs, k=max_k, dim=-1)
        
        # Store for later calculation
        self.all_predictions.append(topk_indices.cpu().numpy())
        self.all_targets.append(targets.cpu().numpy())
        self.all_probs.append(topk_probs.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        # Concatenate all batches
        predictions = np.concatenate(self.all_predictions, axis=0)  # (N, max_K)
        targets = np.concatenate(self.all_targets, axis=0)  # (N,)
        probs = np.concatenate(self.all_probs, axis=0)  # (N, max_K)
        
        n_samples = len(targets)
        metrics = {}
        
        # Calculate Top-K Accuracy (Hit@K)
        for k in self.k_values:
            topk_preds = predictions[:, :k]
            hits = np.any(topk_preds == targets[:, None], axis=1)
            accuracy = hits.mean()
            metrics[f'hit@{k}'] = float(accuracy)
        
        # Calculate MRR (Mean Reciprocal Rank)
        reciprocal_ranks = []
        for i in range(n_samples):
            target = targets[i]
            # Find rank of correct prediction (1-indexed)
            matches = np.where(predictions[i] == target)[0]
            if len(matches) > 0:
                rank = matches[0] + 1  # Convert to 1-indexed
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        metrics['mrr'] = float(np.mean(reciprocal_ranks))
        
        # Calculate Precision@K, Recall@K, F1@K
        for k in self.k_values:
            topk_preds = predictions[:, :k]
            
            # Precision@K: proportion of top-K that are correct
            # For single-label classification, this is just hit@K / K
            precision_scores = []
            recall_scores = []
            
            for i in range(n_samples):
                target = targets[i]
                preds_k = topk_preds[i]
                
                # Check if target is in top-K
                is_correct = target in preds_k
                
                # Precision: 1/K if correct, 0 otherwise (single label)
                precision = 1.0 / k if is_correct else 0.0
                precision_scores.append(precision)
                
                # Recall: 1 if correct, 0 otherwise (single label)
                recall = 1.0 if is_correct else 0.0
                recall_scores.append(recall)
            
            precision = np.mean(precision_scores)
            recall = np.mean(recall_scores)
            
            # F1@K
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            metrics[f'precision@{k}'] = float(precision)
            metrics[f'recall@{k}'] = float(recall)
            metrics[f'f1@{k}'] = float(f1)
        
        return metrics


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

def load_data_and_preprocess(
    test_data_path: str,
    vocab_path: str,
    max_seq_len: int = 20,
) -> Tuple[DataLoader, Dict[str, int], int]:
    """Load raw test sequences and process them (matches training logic)."""
    
    # 1. Load vocabulary (must match training!)
    logger.info(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'r') as f:
        tech_to_idx = json.load(f)
    vocab_size = len(tech_to_idx)
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # 2. Load raw test sequences (saved by train_lstm.py)
    logger.info(f"Loading raw test sequences from {test_data_path}...")
    test_df = pd.read_parquet(test_data_path)
    
    # Validate format
    if 'sequence' not in test_df.columns:
        raise ValueError(
            f"Expected column 'sequence' in {test_data_path}, found {list(test_df.columns)}. "
            "Make sure you are using the file saved by train_lstm.py"
        )
    
    sequences = test_df['sequence'].tolist()
    logger.info(f"Loaded {len(sequences)} raw test sequences")
    
    # 3. Process sequences: create sliding windows (same logic as trainer)
    logger.info("Creating sliding window pairs from raw test sequences...")
    histories = []
    targets = []
    
    for seq in sequences:
        # Convert technique strings to indices
        seq_indices = [tech_to_idx.get(t, 0) for t in seq]  # Use 0 (padding) for unknown
        
        if len(seq_indices) < 2:
            continue  # Skip sequences that are too short
        
        # Create sliding windows: [A, B, C] -> ([A], B), ([A, B], C)
        for i in range(1, len(seq_indices)):
            history = seq_indices[:i]
            target = seq_indices[i]
            
            # Pad history to max_seq_len
            if len(history) < max_seq_len:
                history = [0] * (max_seq_len - len(history)) + history
            else:
                history = history[-max_seq_len:]  # Truncate if too long
            
            histories.append(history)
            targets.append(target)
    
    # 4. Convert to arrays
    X_test = np.array(histories)
    y_test = np.array(targets)
    logger.info(f"Generated {len(X_test)} evaluation samples from sliding windows")
    
    # 5. Create DataLoader
    test_dataset = AttackSequenceDataset(X_test.tolist(), y_test.tolist())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return test_loader, tech_to_idx, vocab_size


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Evaluate model on test set."""
    
    model.eval()
    metrics_calc = MetricsCalculator(k_values=k_values)
    
    logger.info("Evaluating model on test set...")
    with torch.no_grad():
        for sequences, targets in tqdm(test_loader, desc="Evaluating"):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(sequences)
            
            # Update metrics
            metrics_calc.update(logits, targets)
    
    # Compute final metrics
    metrics = metrics_calc.compute()
    
    return metrics


def print_metrics_table(metrics: Dict[str, float]):
    """Print metrics in a formatted table."""
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Top-K Accuracy
    print("\n📊 Top-K Accuracy (Hit@K):")
    print("-" * 40)
    for k in [1, 3, 5, 10]:
        key = f'hit@{k}'
        if key in metrics:
            print(f"  Hit@{k:<2}: {metrics[key]:.4f} ({metrics[key]*100:.2f}%)")
    
    # MRR
    print("\n📈 Mean Reciprocal Rank (MRR):")
    print("-" * 40)
    print(f"  MRR: {metrics['mrr']:.4f}")
    
    # Precision, Recall, F1
    print("\n📏 Precision, Recall, F1:")
    print("-" * 40)
    for k in [1, 3, 5, 10]:
        precision = metrics.get(f'precision@{k}', 0)
        recall = metrics.get(f'recall@{k}', 0)
        f1 = metrics.get(f'f1@{k}', 0)
        print(f"  K={k:<2} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    
    print("\n" + "="*60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LSTM model")
    parser.add_argument('--checkpoint', type=str, default='outputs/lstm_weights.pt',
                       help='Path to model weights')
    parser.add_argument('--vocab', type=str, default='outputs/tech_to_idx.json',
                       help='Path to vocabulary JSON')
    parser.add_argument('--test-data', type=str, default='data/test_sequences_final.parquet',
                       help='Path to test split parquet file (saved by train_lstm.py)')
    # Backward compatibility: allow --data as an alias for --test-data
    parser.add_argument('--data', dest='test_data', help='Alias for --test-data (path to test parquet)')
    parser.add_argument('--output', type=str, default='outputs/lstm_eval_metrics.json',
                       help='Output path for metrics JSON')
    parser.add_argument('--max-seq-len', type=int, default=20,
                       help='Maximum sequence length')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--hidden-size', type=int, default=256,
                       help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    test_loader, tech_to_idx, vocab_size = load_data_and_preprocess(
        args.test_data,
        args.vocab,
        args.max_seq_len,
    )
    
    # Create model
    logger.info("Creating model...")
    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    
    # Load weights
    logger.info(f"Loading weights from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    logger.info("Model loaded successfully")
    
    # Evaluate on test set
    logger.info("\n--- Test Set Evaluation ---")
    test_metrics = evaluate_model(model, test_loader, device)
    print_metrics_table(test_metrics)
    
    # Save metrics
    output_data = {
        'test': test_metrics,
        'model_info': {
            'checkpoint': args.checkpoint,
            'vocab_size': vocab_size,
            'embedding_dim': args.embedding_dim,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
        }
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n[OK] Metrics saved to {output_path}")


if __name__ == '__main__':
    main()
