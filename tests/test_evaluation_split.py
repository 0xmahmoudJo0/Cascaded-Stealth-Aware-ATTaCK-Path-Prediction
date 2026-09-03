"""
Test for train/test split correctness in evaluation.

Verifies that:
1. train_ratio=0.0 preserves sequence order (no shuffle)
2. train_ratio=0.8 shuffles sequences reproducibly
3. Checkpoint split indices are used correctly
"""

from typing import List
from src.chain_paths.eval import Evaluator
from src.chain_paths.predictor import Predictor


class MockPredictor:
    """Mock predictor for testing evaluation logic."""
    
    def __init__(self):
        self.params = {
            'use_scenario_priors': True,
            'bootstrap_rounds': 100,
            'significance_alpha': 0.05,
        }


def create_test_evaluator(n_sequences: int = 100) -> Evaluator:
    """Create evaluator with mock data."""
    sequences = [[f"T{i:04d}"] for i in range(n_sequences)]
    predictor = MockPredictor()
    evaluator = Evaluator(predictor, sequences)
    return evaluator


def test_split_sequences_with_zero_ratio_preserves_order():
    """
    Test that train_ratio=0.0 preserves original sequence order.
    
    This is critical for checkpoint evaluation - we want to evaluate on the EXACT
    same test set that was used during training, in the same order.
    """
    evaluator = create_test_evaluator(n_sequences=100)
    
    # Call with train_ratio=0.0 (no re-split)
    train_seqs, test_seqs = evaluator.split_sequences(train_ratio=0.0, random_seed=None)
    
    # Verify no training set
    assert len(train_seqs) == 0, f"Expected 0 train sequences, got {len(train_seqs)}"
    
    # Verify all sequences in test set
    assert len(test_seqs) == 100, f"Expected 100 test sequences, got {len(test_seqs)}"
    
    # Verify order is preserved (not shuffled)
    for i, seq in enumerate(test_seqs):
        expected = [f"T{i:04d}"]
        assert seq == expected, f"Sequence {i} order not preserved: got {seq}, expected {expected}"
    
    # Verify split indices are sequential (not shuffled)
    split_indices = evaluator._last_split_indices['test']
    expected_indices = list(range(100))
    assert split_indices == expected_indices, f"Indices not in order: {split_indices}"
    
    print("✅ train_ratio=0.0 correctly preserves sequence order")


def test_split_sequences_with_normal_ratio_shuffles():
    """
    Test that train_ratio != 0.0 shuffles sequences with seed.
    """
    evaluator = create_test_evaluator(n_sequences=100)
    
    # Call with train_ratio=0.8 (normal split)
    train_seqs, test_seqs = evaluator.split_sequences(train_ratio=0.8, random_seed=42)
    
    # Verify split ratio
    assert len(train_seqs) == 80, f"Expected 80 train sequences, got {len(train_seqs)}"
    assert len(test_seqs) == 20, f"Expected 20 test sequences, got {len(test_seqs)}"
    
    # Verify indices are shuffled (not sequential)
    split_indices = evaluator._last_split_indices['test']
    expected_indices = list(range(100))[80:]  # Would be sequential without shuffle
    assert split_indices != expected_indices, "Indices should be shuffled but weren't"
    
    print("✅ train_ratio=0.8 correctly shuffles sequences")


def test_split_sequences_reproducibility():
    """
    Test that same random seed produces same split.
    """
    evaluator1 = create_test_evaluator(n_sequences=100)
    evaluator2 = create_test_evaluator(n_sequences=100)
    
    # Split with same seed
    train1, test1 = evaluator1.split_sequences(train_ratio=0.8, random_seed=42)
    train2, test2 = evaluator2.split_sequences(train_ratio=0.8, random_seed=42)
    
    # Verify same split
    indices1 = evaluator1._last_split_indices
    indices2 = evaluator2._last_split_indices
    
    assert indices1['train'] == indices2['train'], "Train indices should match with same seed"
    assert indices1['test'] == indices2['test'], "Test indices should match with same seed"
    
    print("✅ Split reproducibility verified (same seed = same split)")


def test_split_sequences_different_seeds_differ():
    """
    Test that different random seeds produce different splits.
    """
    evaluator1 = create_test_evaluator(n_sequences=100)
    evaluator2 = create_test_evaluator(n_sequences=100)
    
    # Split with different seeds
    train1, test1 = evaluator1.split_sequences(train_ratio=0.8, random_seed=42)
    train2, test2 = evaluator2.split_sequences(train_ratio=0.8, random_seed=123)
    
    # Verify different splits
    indices1 = evaluator1._last_split_indices
    indices2 = evaluator2._last_split_indices
    
    # They should differ (with very high probability)
    assert indices1['test'] != indices2['test'], "Test indices should differ with different seeds"
    
    print("✅ Different seeds produce different splits")


def test_checkpoint_evaluation_scenario():
    """
    Simulate the actual checkpoint evaluation scenario:
    1. Train creates test_indices [1, 5, 23, 42, ...]
    2. Evaluation loads same sequences with train_ratio=0.0
    3. Should use exact same order
    """
    evaluator = create_test_evaluator(n_sequences=100)
    
    # Simulate checkpoint's saved test indices (from a hypothetical training run)
    checkpoint_test_indices = [1, 5, 23, 42, 67, 88, 12, 34, 56, 78]
    
    # Evaluation creates evaluator with just the test sequences
    test_sequences_from_checkpoint = [evaluator.sequences[i] for i in checkpoint_test_indices]
    predictor = MockPredictor()
    eval_evaluator = Evaluator(predictor, test_sequences_from_checkpoint)
    
    # Call evaluate_all with train_ratio=0.0 (don't re-split)
    train_seqs, test_seqs = eval_evaluator.split_sequences(train_ratio=0.0, random_seed=None)
    
    # Verify no training set
    assert len(train_seqs) == 0
    
    # Verify test sequences are in original order (preserved from checkpoint)
    assert len(test_seqs) == 10
    
    # Verify indices are sequential (because we're not reshuffling)
    split_indices = eval_evaluator._last_split_indices['test']
    expected = list(range(10))
    assert split_indices == expected, f"Expected {expected}, got {split_indices}"
    
    print("✅ Checkpoint evaluation scenario verified")


if __name__ == "__main__":
    test_split_sequences_with_zero_ratio_preserves_order()
    test_split_sequences_with_normal_ratio_shuffles()
    test_split_sequences_reproducibility()
    test_split_sequences_different_seeds_differ()
    test_checkpoint_evaluation_scenario()
    print("\n✅ All evaluation split tests passed!")
