import json

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors

from src.chain_paths.model_fitting import fit_component_weights
from src.chain_paths.predictor import Predictor
from src.chain_paths.config import DEFAULT_PARAMS


@pytest.fixture
def synthetic_training_predictor(tmp_path):
    techs = ['T1000', 'T1001', 'T1002']
    tech_to_idx = {tech: i for i, tech in enumerate(techs)}

    counts_unigram = pd.DataFrame({'candidate': techs, 'count': [50, 30, 20]})
    counts_bigram = pd.DataFrame({
        'context': [('T1000',), ('T1001',)],
        'candidate': ['T1001', 'T1002'],
        'count': [25, 15]
    })
    counts_trigram = pd.DataFrame(columns=['context', 'candidate', 'count'])

    counts_data = {1: counts_unigram, 2: counts_bigram, 3: counts_trigram}

    ps_scores = {tech: 1.0 for tech in techs}
    sigma_scores = {tech: 0.9 for tech in techs}

    embeddings = np.eye(len(techs), dtype=np.float32)
    nn_model = NearestNeighbors(n_neighbors=len(techs), metric='cosine')
    nn_model.fit(embeddings)

    class SklearnIndex:
        def __init__(self, model):
            self.model = model

        def search(self, query, top_k):
            top = min(top_k, self.model._fit_X.shape[0])
            distances, indices = self.model.kneighbors(query, n_neighbors=top)
            scores = 1 - distances
            return scores, indices

    index = SklearnIndex(nn_model)

    predictor = Predictor(
        counts_data,
        ps_scores,
        sigma_scores,
        embeddings,
        tech_to_idx,
        index,
        attacker_profiles=None,
        params=DEFAULT_PARAMS,
    )

    predictor.params['component_weights'] = DEFAULT_PARAMS['component_weights'].copy()

    yield predictor


def test_fit_component_weights_updates_predictor(synthetic_training_predictor, tmp_path):
    sequences = [
        ['T1000', 'T1001', 'T1002'],
        ['T1000', 'T1001'],
    ]

    metadata = pd.DataFrame({
        'sequence_id': [0, 1],
        'group_id': ['', ''],
        'start_time': pd.to_datetime(['2021-01-01', '2021-06-01']),
    })

    output_path = tmp_path / 'weights.json'

    initial_weights = synthetic_training_predictor.params['component_weights'].copy()

    learned = fit_component_weights(
        synthetic_training_predictor,
        sequences,
        metadata=metadata,
        train_ratio=1.0,
        top_n_candidates=3,
        regularization=0.1,
        output_path=output_path,
    )

    assert output_path.exists()

    with open(output_path, 'r', encoding='utf-8') as f:
        on_disk = json.load(f)

    for key in on_disk:
        assert key in learned

    assert any(abs(learned[k] - initial_weights.get(k, 0.0)) > 1e-6 for k in learned)

    predictions = synthetic_training_predictor.next_probabilities(['T1000'])
    assert predictions
