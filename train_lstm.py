"""
Complete Training Pipeline for LSTM Attack Technique Sequence Predictor.

This script handles:
1. Loading sequences from parquet file
2. Tokenizing technique IDs
3. Creating sliding window (history, target) pairs
4. Padding sequences to fixed length
5. Splitting data without sequence leakage
6. Training with class weights to handle imbalance
7. Validation with Top-1 and Top-3 accuracy
8. Early stopping and learning rate scheduling
9. Saving model weights and vocabulary
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training configuration parameters."""
    
    # Data
    data_path: str = "data/sequences.parquet"
    max_seq_len: int = 20
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Pre-trained embeddings (Word2Vec)
    use_pretrained_embeddings: bool = True
    embeddings_path: str = "outputs/tech_embeddings.npy"
    embeddings_index_path: str = "outputs/tech_index.csv"
    
    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Model architecture (must match LSTMPredictor)
    embedding_dim: int = 128
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    
    # Training strategies
    patience: int = 5  # Early stopping patience
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    
    # Output
    output_dir: str = "outputs"
    weights_path: str = "outputs/lstm_weights.pt"
    vocab_path: str = "outputs/tech_to_idx.json"
    
    def __post_init__(self):
        """Ensure output directory exists."""
        Path(self.output_dir).mkdir(exist_ok=True)


# ============================================================================
# DATASET
# ============================================================================

class AttackSequenceDataset(Dataset):
    """PyTorch Dataset for attack technique sequences."""
    
    def __init__(
        self,
        sequences: List[List[int]],
        targets: List[int],
        max_seq_len: int = 20,
    ):
        """
        Initialize dataset.
        
        Args:
            sequences: List of padded sequences (as token indices)
            targets: List of target technique indices
            max_seq_len: Maximum sequence length (for validation)
        """
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.targets = torch.as_tensor(targets, dtype=torch.long)
        self.max_seq_len = max_seq_len
        
        if len(self.sequences) != len(self.targets):
            raise ValueError("Sequences and targets must have same length")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class SequencePreprocessor:
    """Handles tokenization, sliding windows, and padding."""
    
    def __init__(self, max_seq_len: int = 20):
        """
        Initialize preprocessor.
        
        Args:
            max_seq_len: Maximum sequence length for padding
        """
        self.max_seq_len = max_seq_len
        self.tech_to_idx: Dict[str, int] = {""  : 0}  # Reserve 0 for padding
        self.idx_to_tech: Dict[int, str] = {0: ""}
        self.vocab_size = 1
        self._vocab_built = False
    
    def build_vocabulary(self, sequences: List[List[str]]) -> None:
        """
        Build vocabulary from all sequences.
        
        Args:
            sequences: List of technique sequences
        """
        logger.info("Building vocabulary...")
        unique_techniques = set()
        
        for seq in sequences:
            unique_techniques.update(seq)
        
        for technique in sorted(unique_techniques):
            self.tech_to_idx[technique] = self.vocab_size
            self.idx_to_tech[self.vocab_size] = technique
            self.vocab_size += 1
        
        self._vocab_built = True
        logger.info(f"Vocabulary size: {self.vocab_size} (including padding token)")
    
    def create_sliding_windows(
        self,
        sequences: List[List[str]],
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Create (history, target) pairs from sequences using sliding window.
        
        Example: [A, B, C] -> ([A], B), ([A, B], C)
        
        Args:
            sequences: List of technique sequences
        
        Returns:
            Tuple of (sequence_indices, target_indices)
        """
        if not self._vocab_built:
            raise ValueError("Must build vocabulary first")
        
        logger.info("Creating sliding window pairs...")
        histories = []
        targets = []
        
        for seq in sequences:
            # Create (history, target) pairs
            for i in range(1, len(seq)):
                history = seq[:i]
                target = seq[i]
                
                histories.append(history)
                targets.append(self.tech_to_idx[target])
        
        logger.info(f"Created {len(targets)} (history, target) pairs")
        return histories, targets
    
    def pad_sequences(self, histories: List[List[str]]) -> List[List[int]]:
        """
        Pad sequences to fixed length using pre-padding with 0 (padding token).
        
        Args:
            histories: List of technique history sequences (strings)
        
        Returns:
            List of padded sequences (as token indices)
        """
        logger.info(f"Padding sequences to length {self.max_seq_len}...")
        padded = []
        
        for history in histories:
            # Convert to indices
            indices = [self.tech_to_idx[t] for t in history]
            
            # Pre-pad with 0s (padding token)
            if len(indices) < self.max_seq_len:
                indices = [0] * (self.max_seq_len - len(indices)) + indices
            else:
                # Truncate if longer
                indices = indices[-self.max_seq_len:]
            
            padded.append(indices)
        
        return padded
    
    def process(
        self,
        sequences: List[List[str]],
    ) -> Tuple[List[List[int]], List[int], int]:
        """
        Full preprocessing pipeline.
        
        Args:
            sequences: List of technique sequences
        
        Returns:
            Tuple of (padded_sequences, targets, vocab_size)
        """
        self.build_vocabulary(sequences)
        histories, target_indices = self.create_sliding_windows(sequences)
        padded_sequences = self.pad_sequences(histories)
        
        return padded_sequences, target_indices, self.vocab_size


# ============================================================================
# LSTM MODEL
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM model for attack sequence prediction with optional pre-trained embedding initialization."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        pretrained_embeddings: Optional[np.ndarray] = None,
    ):
        """
        Initialize LSTM model.
        
        Args:
            vocab_size: Size of vocabulary (including padding)
            embedding_dim: Embedding dimension
            hidden_size: Hidden size of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            pretrained_embeddings: Optional pre-trained embeddings (from Word2Vec)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Initialize with pre-trained embeddings if provided
        if pretrained_embeddings is not None:
            logger.info(f"Initializing embeddings with pre-trained Word2Vec weights")
            pretrained_tensor = torch.from_numpy(pretrained_embeddings).float()
            
            # Handle embedding dimension mismatch
            if pretrained_tensor.shape[1] != embedding_dim:
                logger.warning(f"Embedding dim mismatch: W2V={pretrained_tensor.shape[1]}, LSTM={embedding_dim}")
                # Use only the first embedding_dim dimensions or pad
                if pretrained_tensor.shape[1] > embedding_dim:
                    pretrained_tensor = pretrained_tensor[:, :embedding_dim]
                    logger.info(f"  Truncated embedding dim to {embedding_dim}")
                else:
                    padding = torch.randn(pretrained_tensor.shape[0], embedding_dim - pretrained_tensor.shape[1]) * 0.02
                    pretrained_tensor = torch.cat([pretrained_tensor, padding], dim=1)
                    logger.info(f"  Padded embedding dim to {embedding_dim}")
            
            # Handle vocab size mismatch
            if pretrained_tensor.shape[0] < vocab_size:
                padding_size = vocab_size - pretrained_tensor.shape[0]
                random_padding = torch.randn(padding_size, embedding_dim) * 0.02
                pretrained_tensor = torch.cat([pretrained_tensor, random_padding], dim=0)
                logger.info(f"  Padded vocab from {pretrained_tensor.shape[0] - padding_size} to {vocab_size}")
            elif pretrained_tensor.shape[0] > vocab_size:
                pretrained_tensor = pretrained_tensor[:vocab_size]
                logger.info(f"  Truncated vocab to {vocab_size}")
            
            # Set embedding weights (keep padding token at index 0 as zeros)
            with torch.no_grad():
                self.embedding.weight[1:].copy_(pretrained_tensor[1:])
                self.embedding.weight[0].zero_()
            logger.info(f"  ✓ Embedding initialized with pre-trained weights: {self.embedding.weight.shape}")
        
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
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_size)
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        dropped = self.dropout(last_output)
        logits = self.output(dropped)  # (batch_size, vocab_size)
        return logits


def load_pretrained_embeddings(
    config: TrainingConfig,
    tech_to_idx: Dict[str, int]
) -> Optional[np.ndarray]:
    """
    Load pre-trained Word2Vec embeddings and align with LSTM vocabulary.
    
    Args:
        config: Training configuration
        tech_to_idx: LSTM vocabulary mapping (technique -> index)
    
    Returns:
        Aligned embeddings array or None if not available
    """
    if not config.use_pretrained_embeddings:
        logger.info("Pre-trained embeddings disabled in config")
        return None
    
    emb_path = Path(config.embeddings_path)
    idx_path = Path(config.embeddings_index_path)
    
    if not emb_path.exists():
        logger.warning(f"Pre-trained embeddings not found at {emb_path}")
        logger.warning("Run: python -m src.chain_paths.cli emb")
        logger.warning("Continuing with random initialization...")
        return None
    
    if not idx_path.exists():
        logger.warning(f"Embedding index not found at {idx_path}")
        logger.warning("Continuing with random initialization...")
        return None
    
    try:
        # Load Word2Vec embeddings
        w2v_embeddings = np.load(emb_path)
        logger.info(f"Loaded Word2Vec embeddings: {w2v_embeddings.shape}")
        
        # Load Word2Vec vocabulary
        w2v_index = pd.read_csv(idx_path)
        w2v_tech_to_idx = dict(zip(w2v_index['technique_id'], w2v_index['index']))
        logger.info(f"Loaded Word2Vec vocabulary: {len(w2v_tech_to_idx)} techniques")
        
        # Create aligned embeddings for LSTM vocabulary
        lstm_vocab_size = len(tech_to_idx)
        embedding_dim = w2v_embeddings.shape[1]
        aligned_embeddings = np.random.randn(lstm_vocab_size, embedding_dim).astype(np.float32) * 0.02
        
        # Copy embeddings for techniques that exist in both vocabularies
        matched = 0
        for tech, lstm_idx in tech_to_idx.items():
            if tech in w2v_tech_to_idx:
                w2v_idx = w2v_tech_to_idx[tech]
                if w2v_idx < w2v_embeddings.shape[0]:
                    aligned_embeddings[lstm_idx] = w2v_embeddings[w2v_idx]
                    matched += 1
        
        # Keep padding token as zeros
        aligned_embeddings[0] = 0.0
        
        logger.info(f"✓ Aligned embeddings: {matched}/{lstm_vocab_size} techniques matched with Word2Vec")
        logger.info(f"  Coverage: {matched/max(lstm_vocab_size, 1)*100:.1f}%")
        
        return aligned_embeddings
        
    except Exception as e:
        logger.warning(f"Failed to load embeddings: {e}")
        logger.warning("Continuing with random initialization...")
        return None


# ============================================================================
#       self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
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
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        # Embedding
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(emb)  # (batch_size, seq_len, hidden_size)
        
        # Take last timestep
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Dropout
        out = self.dropout(last_out)
        
        # Output layer
        logits = self.output(out)  # (batch_size, vocab_size)
        
        return logits


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

class Trainer:
    """Handles training and evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Loss function (will set weight later)
        self.criterion = None
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    def set_loss_weights(self, class_weights: torch.Tensor) -> None:
        """
        Set loss function with class weights for imbalance handling.
        
        Args:
            class_weights: Tensor of shape (vocab_size,) with inverse frequency weights
        """
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.criterion.to(self.device)
        logger.info("Loss function initialized with class weights")
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for sequences, targets in pbar:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            logits = self.model(sequences)
            loss = self.criterion(logits, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def evaluate(self, loader: DataLoader, set_name: str = "Validation") -> Tuple[float, float, float]:
        """
        Evaluate on a dataset.
        
        Args:
            loader: Data loader
            set_name: Name of the set (for logging)
        
        Returns:
            Tuple of (loss, top1_accuracy, top3_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct_top1 = 0
        correct_top3 = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"{set_name} Evaluation")
            for sequences, targets in pbar:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits = self.model(sequences)
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                
                # Top-1 accuracy
                predictions = logits.argmax(dim=1)
                correct_top1 += (predictions == targets).sum().item()
                
                # Top-3 accuracy
                _, top3_preds = logits.topk(3, dim=1)
                correct_top3 += (top3_preds == targets.unsqueeze(1)).any(dim=1).sum().item()
                
                total_samples += targets.size(0)
                pbar.set_postfix({
                    'loss': loss.item(),
                    'top1_acc': correct_top1 / total_samples,
                    'top3_acc': correct_top3 / total_samples,
                })
        
        avg_loss = total_loss / len(loader)
        top1_accuracy = correct_top1 / total_samples
        top3_accuracy = correct_top3 / total_samples
        
        logger.info(
            f"{set_name} - Loss: {avg_loss:.4f}, Top-1 Acc: {top1_accuracy:.4f}, "
            f"Top-3 Acc: {top3_accuracy:.4f}"
        )
        
        return avg_loss, top1_accuracy, top3_accuracy
    
    def train(self) -> Dict:
        """
        Full training loop with early stopping.
        
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_top1_acc': [],
            'val_top3_acc': [],
            'test_loss': [],
            'test_top1_acc': [],
            'test_top3_acc': [],
        }
        
        logger.info(f"Training for max {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)
            logger.info(f"Training Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_top1, val_top3 = self.evaluate(self.val_loader, "Validation")
            history['val_loss'].append(val_loss)
            history['val_top1_acc'].append(val_top1)
            history['val_top3_acc'].append(val_top3)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                logger.info("✓ Validation loss improved!")
            else:
                self.patience_counter += 1
                logger.info(
                    f"Validation loss did not improve. "
                    f"Patience: {self.patience_counter}/{self.config.patience}"
                )
                
                if self.patience_counter >= self.config.patience:
                    logger.info("Early stopping triggered!")
                    break
        
        # Restore best model
        if self.best_model_state is not None:
            logger.info("Restoring best model...")
            self.model.load_state_dict(self.best_model_state)
        
        # Evaluate on test set
        logger.info(f"\n{'='*60}")
        logger.info("Final Evaluation on Test Set")
        logger.info(f"{'='*60}")
        test_loss, test_top1, test_top3 = self.evaluate(self.test_loader, "Test")
        history['test_loss'].append(test_loss)
        history['test_top1_acc'].append(test_top1)
        history['test_top3_acc'].append(test_top3)
        
        return history


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline."""
    
    # Setup
    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading sequences from {config.data_path}...")
    df = pd.read_parquet(config.data_path)
    sequences = df['sequence'].tolist()
    logger.info(f"Loaded {len(sequences)} sequences")
    
    # ============================================================================
    # FIX: Shuffle and split sequences FIRST (prevents data leakage)
    # ============================================================================
    import random
    random.seed(42)
    random.shuffle(sequences)  # Shuffle attack sequences (not windows!)
    
    logger.info("Splitting sequences to prevent data leakage...")
    n_sequences = len(sequences)
    n_train_seq = int(n_sequences * config.train_split)
    n_val_seq = int(n_sequences * config.val_split)
    
    train_sequences = sequences[:n_train_seq]
    val_sequences = sequences[n_train_seq:n_train_seq + n_val_seq]
    test_sequences = sequences[n_train_seq + n_val_seq:]
    
    logger.info(f"Split sequences: Train={len(train_sequences)}, Val={len(val_sequences)}, Test={len(test_sequences)}")
    
    # Build vocabulary on ALL data (ensures no OOV during evaluation)
    # NOTE: This is acceptable since we're not learning embeddings from val/test
    preprocessor = SequencePreprocessor(max_seq_len=config.max_seq_len)
    preprocessor.build_vocabulary(sequences)  # Use ALL sequences
    vocab_size = preprocessor.vocab_size
    
    # Create sliding windows for each split separately
    logger.info("Creating sliding windows for each split...")
    train_histories, train_targets = preprocessor.create_sliding_windows(train_sequences)
    val_histories, val_targets = preprocessor.create_sliding_windows(val_sequences)
    test_histories, test_targets = preprocessor.create_sliding_windows(test_sequences)
    
    # Pad sequences
    X_train = np.array(preprocessor.pad_sequences(train_histories))
    X_val = np.array(preprocessor.pad_sequences(val_histories))
    X_test = np.array(preprocessor.pad_sequences(test_histories))
    
    y_train = np.array(train_targets)
    y_val = np.array(val_targets)
    y_test = np.array(test_targets)
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Save RAW test sequences (strings) for evaluation - NOT tokenized indices
    logger.info("Saving test split for evaluation...")
    test_split_path = "data/test_sequences_final.parquet"
    test_df = pd.DataFrame({'sequence': test_sequences})  # Save original string sequences
    test_df.to_parquet(test_split_path, index=False)
    logger.info(f"✓ Saved {len(test_sequences)} raw test sequences to {test_split_path}")
    
    # Calculate class weights (inverse frequency)
    logger.info("Computing class weights...")
    class_counts = np.bincount(y_train, minlength=vocab_size)
    class_weights = np.zeros(vocab_size)
    mask = class_counts > 0
    class_weights[mask] = 1.0 / class_counts[mask]
    # Normalize so they sum to vocab_size
    class_weights = class_weights / class_weights.sum() * vocab_size
    class_weights = torch.from_numpy(class_weights).float().to(device)
    logger.info(f"Class weights computed: min={class_weights.min():.4f}, max={class_weights.max():.4f}")
    
    # Create datasets and dataloaders
    train_dataset = AttackSequenceDataset(X_train, y_train, config.max_seq_len)
    val_dataset = AttackSequenceDataset(X_val, y_val, config.max_seq_len)
    test_dataset = AttackSequenceDataset(X_test, y_test, config.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Load pre-trained embeddings if available
    logger.info("\n" + "="*60)
    logger.info("Loading Pre-trained Embeddings")
    logger.info("="*60)
    pretrained_embeddings = load_pretrained_embeddings(config, preprocessor.tech_to_idx)
    
    # Create model
    logger.info("\n" + "="*60)
    logger.info("Creating LSTM Model")
    logger.info("="*60)
    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pretrained_embeddings=pretrained_embeddings,
    )
    model.to(device)
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )
    trainer.set_loss_weights(class_weights)
    
    # Train
    history = trainer.train()
    
    # Save model
    logger.info(f"\nSaving model weights to {config.weights_path}...")
    torch.save(model.state_dict(), config.weights_path)
    logger.info(f"✓ Saved to {config.weights_path}")
    
    # Save vocabulary
    logger.info(f"Saving vocabulary to {config.vocab_path}...")
    with open(config.vocab_path, 'w') as f:
        json.dump(preprocessor.tech_to_idx, f, indent=2)
    logger.info(f"✓ Saved to {config.vocab_path}")
    
    # Save training history
    history_path = "outputs/training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"✓ Training history saved to {history_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Training complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
