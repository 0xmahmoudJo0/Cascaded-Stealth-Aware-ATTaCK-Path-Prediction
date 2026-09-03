"""
N-gram counts computation for the chain-aware predictor.

This module computes unigram, bigram, and trigram counts from technique sequences
and saves them as Parquet files for efficient access during prediction.
"""

from typing import List

import pandas as pd

from . import config as cfg
from .io import (
    ensure_dir,
    load_parquet,
    load_sequences_for_source,
    save_parquet,
)
from .preprocess import compute_ngram_counts


def compute_all_counts(sequences: List[List[str]]) -> None:
    """
    Compute and save all n-gram counts.
    
    Args:
        sequences: List of technique sequences
    """
    print("Computing all n-gram counts...")
    
    # Ensure output directories exist
    ensure_dir(cfg.COUNTS_UNIGRAM_PARQUET.parent)
    
    # Compute unigram counts
    print("\n=== Computing Unigram Counts ===")
    unigram_df = compute_ngram_counts(sequences, n=1)
    save_parquet(unigram_df, cfg.COUNTS_UNIGRAM_PARQUET)
    print(f"Saved unigram counts to {cfg.COUNTS_UNIGRAM_PARQUET}")
    
    # Compute bigram counts
    print("\n=== Computing Bigram Counts ===")
    bigram_df = compute_ngram_counts(sequences, n=2)
    save_parquet(bigram_df, cfg.COUNTS_BIGRAM_PARQUET)
    print(f"Saved bigram counts to {cfg.COUNTS_BIGRAM_PARQUET}")
    
    # Compute trigram counts
    print("\n=== Computing Trigram Counts ===")
    trigram_df = compute_ngram_counts(sequences, n=3)
    save_parquet(trigram_df, cfg.COUNTS_TRIGRAM_PARQUET)
    print(f"Saved trigram counts to {cfg.COUNTS_TRIGRAM_PARQUET}")
    
    # Print summary statistics
    print("\n=== Count Summary ===")
    print(f"Unigrams: {len(unigram_df)} unique, {unigram_df['count'].sum()} total")
    print(f"Bigrams: {len(bigram_df)} unique, {bigram_df['count'].sum()} total")
    print(f"Trigrams: {len(trigram_df)} unique, {trigram_df['count'].sum()} total")


def load_counts(n: int) -> pd.DataFrame:
    """
    Load n-gram counts from Parquet file.
    
    Args:
        n: N-gram size (1, 2, or 3)
        
    Returns:
        DataFrame with n-gram counts
    """
    if n == 1:
        return load_parquet(cfg.COUNTS_UNIGRAM_PARQUET)
    elif n == 2:
        return load_parquet(cfg.COUNTS_BIGRAM_PARQUET)
    elif n == 3:
        return load_parquet(cfg.COUNTS_TRIGRAM_PARQUET)
    else:
        raise ValueError(f"Unsupported n-gram size: {n}")


def get_count_for_context(candidate: str, context: List[str], counts_df: pd.DataFrame) -> int:
    """
    Get count for a specific context-candidate pair.
    
    Args:
        candidate: Candidate technique
        context: Context (list of previous techniques)
        counts_df: N-gram counts DataFrame
        
    Returns:
        Count value (0 if not found)
    """
    # Convert context to tuple for comparison
    context_tuple = tuple(context)
    
    # Find matching row
    mask = (counts_df['context'].apply(tuple) == context_tuple) & (counts_df['candidate'] == candidate)
    matching_rows = counts_df[mask]
    
    if len(matching_rows) > 0:
        return matching_rows['count'].iloc[0]
    else:
        return 0


def get_top_candidates_for_context(context: List[str], counts_df: pd.DataFrame, top_k: int = 50) -> List[str]:
    """
    Get top-K candidates for a given context.
    
    Args:
        context: Context (list of previous techniques)
        counts_df: N-gram counts DataFrame
        top_k: Number of top candidates to return
        
    Returns:
        List of top candidate techniques
    """
    # Convert context to tuple for comparison
    context_tuple = tuple(context)
    
    # Find matching rows
    mask = counts_df['context'].apply(tuple) == context_tuple
    matching_rows = counts_df[mask]
    
    if len(matching_rows) > 0:
        # Sort by count and return top candidates
        top_candidates = matching_rows.nlargest(top_k, 'count')['candidate'].tolist()
        return top_candidates
    else:
        return []


def get_total_context_support(context: List[str], counts_df: pd.DataFrame) -> int:
    """
    Get total support (sum of counts) for a given context.
    
    Args:
        context: Context (list of previous techniques)
        counts_df: N-gram counts DataFrame
        
    Returns:
        Total support value
    """
    # Convert context to tuple for comparison
    context_tuple = tuple(context)
    
    # Find matching rows
    mask = counts_df['context'].apply(tuple) == context_tuple
    matching_rows = counts_df[mask]
    
    return matching_rows['count'].sum()


def main():
    """Main function to compute n-gram counts."""
    sequences = load_sequences_for_source(cfg.DEFAULT_DATA_SOURCE)
    print(f"Loaded {len(sequences)} sequences")

    # Compute all counts
    compute_all_counts(sequences)


if __name__ == "__main__":
    main()
