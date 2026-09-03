import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors

from src.chain_paths.config import DEFAULT_PARAMS, SIGHTINGS_CSV
from src.chain_paths.predictor import Predictor
from src.chain_paths.preprocess import (
    build_sequences,
    build_technique_mapping,
    apply_technique_mapping,
    compute_ngram_counts,
)
from src.chain_paths.eval import Evaluator
from src.chain_paths.io import load_sightings_data


def test_real_dataset_evaluation_smoke():
    if not SIGHTINGS_CSV.exists():
        pytest.skip("SIGHTINGS dataset not available in test environment")

    try:
        raw_df = load_sightings_data(SIGHTINGS_CSV).head(400)
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"Skipping real dataset smoke test: {exc}")

    if 'campaign_id' not in raw_df.columns:
        raw_df['campaign_id'] = (np.arange(len(raw_df)) // 5).astype(str)

    mapping = build_technique_mapping(raw_df)
    mapped_df = apply_technique_mapping(raw_df, mapping)
    sequences, metadata = build_sequences(mapped_df)
    assert sequences, "Expected at least one sequence from real dataset"

    # Limit to manageable number for test runtime
    sequences = sequences[:20]
    metadata = metadata.head(len(sequences)).reset_index(drop=True)
    metadata['sequence_id'] = list(range(len(sequences)))

    counts_data = {
        1: compute_ngram_counts(sequences, 1),
        2: compute_ngram_counts(sequences, 2),
        3: compute_ngram_counts(sequences, 3),
    }

    techniques = sorted({tech for seq in sequences for tech in seq})
    tech_to_idx = {tech: idx for idx, tech in enumerate(techniques)}
    embeddings = np.eye(len(techniques), dtype=np.float32)
    index = NearestNeighbors(metric='cosine')
    if len(techniques) > 0:
        index.fit(embeddings)

    ps_scores = {tech: 1.0 for tech in techniques}
    sigma_scores = {tech: 1.0 for tech in techniques}

    predictor = Predictor(
        counts_data=counts_data,
        ps_scores=ps_scores,
        sigma_scores=sigma_scores,
        embeddings=embeddings,
        tech_to_idx=tech_to_idx,
        index=index,
        attacker_profiles=None,
        params=DEFAULT_PARAMS.copy(),
    )

    evaluator = Evaluator(predictor=predictor, sequences=sequences, sequence_metadata=metadata)
    train_sequences, test_sequences = evaluator.split_sequences(train_ratio=0.5, temporal_split=True)
    metrics = evaluator.evaluate_next_step(test_sequences, k_values=[1])

    assert metrics['observations'] > 0
    assert metrics['model']['precision_at_k'][1]['mean'] >= 0.0
    assert 'leakage' in metrics
