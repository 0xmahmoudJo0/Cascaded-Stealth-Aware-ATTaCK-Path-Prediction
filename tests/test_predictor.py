"""
Unit tests for the Predictor class.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Dict

from src.chain_paths.predictor import Predictor
from src.chain_paths.config import DEFAULT_PARAMS
from src.chain_paths.attacker_profiles import format_profile_id


@pytest.fixture
def synthetic_predictor():
    """Creates a Predictor instance with a small, synthetic dataset."""
    # Techniques and mappings
    techs = ['T1059', 'T1105', 'T1071', 'T1548']
    tech_to_idx = {tech: i for i, tech in enumerate(techs)}

    # 1. Counts data
    unigrams = pd.DataFrame({'candidate': ['T1059', 'T1105', 'T1071'], 'count': [100, 80, 50]})
    bigrams = pd.DataFrame({
        'context': [('T1059',), ('T1059',), ('T1105',)],
        'candidate': ['T1105', 'T1071', 'T1548'],
        'count': [50, 10, 5]
    })
    counts_data = {1: unigrams, 2: bigrams, 3: pd.DataFrame(columns=['context', 'candidate', 'count'])}

    # 2. PS scores
    ps_scores = {'T1059': 0.9, 'T1105': 0.8, 'T1071': 0.7, 'T1548': 0.6}

    # 3. Sigma (stealth) scores
    # T1105 is stealthy, T1071 is noisy
    sigma_scores = {'T1059': 0.8, 'T1105': 0.95, 'T1071': 0.5, 'T1548': 0.7}

    # 4. Embeddings
    # T1059 and T1105 are similar. T1071 is different.
    embeddings = np.array([
        [0.9, 0.1, 0.1],  # T1059
        [0.8, 0.2, 0.1],  # T1105
        [0.1, 0.9, 0.2],  # T1071
        [0.5, 0.5, 0.5],  # T1548
    ], dtype=np.float32)

    # 5. ANN Index
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

    # 6. Attacker profiles (optional, not used in this test)
    attacker_profiles = None

    # 7. Parameters
    params = DEFAULT_PARAMS.copy()

    # Initialize Predictor
    predictor = Predictor(
        counts_data=counts_data,
        ps_scores=ps_scores,
        sigma_scores=sigma_scores,
        embeddings=embeddings,
        tech_to_idx=tech_to_idx,
        index=index,
        attacker_profiles=attacker_profiles,
        params=params
    )
    return predictor


def test_predictor_ranking(synthetic_predictor):
    """
    Tests if the predictor ranks next techniques as expected based on synthetic data.
    History: ['T1059']
    - T1105 should be ranked highest: high bigram count, high embedding similarity, high stealth.
    - T1071 should be ranked lower: low bigram count, low embedding similarity, low stealth.
    - T1548 should be ranked last: no bigram count, relies on embedding backoff.
    """
    history = ['T1059']
    
    # Get next step probabilities
    predictions = synthetic_predictor.next_probabilities(history, top_n_candidates=4)

    assert len(predictions) > 0, "Predictor should return predictions"

    # Extract just the technique IDs from the predictions
    predicted_order = [p[0] for p in predictions]

    # We expect T1105 to be the top prediction
    assert predicted_order[0] == 'T1105', "T1105 should be the top-ranked prediction"

    # Check the relative order
    try:
        rank_t1105 = predicted_order.index('T1105')
        rank_t1071 = predicted_order.index('T1071')
        rank_t1548 = predicted_order.index('T1548')
        
        assert rank_t1105 < rank_t1071, "T1105 should be ranked higher than T1071"
        assert rank_t1071 < rank_t1548, "T1071 should be ranked higher than T1548"

    except ValueError as e:
        pytest.fail(f"A predicted technique was not found in the output: {e}. Predictions: {predicted_order}")


def test_embedding_distribution_normalizes(synthetic_predictor):
    history = ['T1059']
    candidates = ['T1105', 'T1071', 'T1548']

    probs, components = synthetic_predictor.compute_embedding_distribution(history, candidates)

    assert pytest.approx(1.0) == sum(probs.values())
    for candidate in candidates:
        assert 0.0 <= probs[candidate] <= 1.0
        assert 'normalizer' in components[candidate]


@pytest.fixture
def predictor_with_priors():
    techs = ['T1000', 'T2000', 'T3000']
    tech_to_idx = {tech: i for i, tech in enumerate(techs)}

    counts_data = {
        1: pd.DataFrame({'candidate': techs, 'count': [10, 10, 10]}),
        2: pd.DataFrame(columns=['context', 'candidate', 'count']),
        3: pd.DataFrame(columns=['context', 'candidate', 'count']),
    }

    ps_scores: Dict[str, float] = {}
    sigma_scores: Dict[str, float] = {}

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

    attacker_profiles = pd.DataFrame([
        {
            'profile_id': format_profile_id('apt_phase', ('APT-Alpha', 'lateral')),
            'p_map': {'T2000': 0.9},
            'source': 'sightings',
        },
        {
            'profile_id': format_profile_id('apt', ('APT-Alpha',)),
            'p_map': {'T1000': 0.6, 'T2000': 0.4},
            'source': 'sightings',
        },
        {
            'profile_id': format_profile_id('global', ('all',)),
            'p_map': {'T1000': 0.33, 'T2000': 0.33, 'T3000': 0.34},
            'source': 'sightings',
        },
    ])

    params = DEFAULT_PARAMS.copy()
    params['use_scenario_priors'] = True

    predictor = Predictor(
        counts_data=counts_data,
        ps_scores=ps_scores,
        sigma_scores=sigma_scores,
        embeddings=embeddings,
        tech_to_idx=tech_to_idx,
        index=index,
        attacker_profiles=attacker_profiles,
        params=params,
    )

    return predictor


def test_group_prior_uniform_without_profiles():
    techs = ['T4000', 'T5000']
    tech_to_idx = {tech: i for i, tech in enumerate(techs)}

    counts_data = {
        1: pd.DataFrame({'candidate': techs, 'count': [5, 5]}),
        2: pd.DataFrame(columns=['context', 'candidate', 'count']),
        3: pd.DataFrame(columns=['context', 'candidate', 'count']),
    }

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
        counts_data=counts_data,
        ps_scores={},
        sigma_scores={},
        embeddings=embeddings,
        tech_to_idx=tech_to_idx,
        index=index,
        attacker_profiles=None,
        params={'use_scenario_priors': True},
    )

    prior = predictor.get_group_prior('T4000', {'apt': 'APT-Zed'})
    assert pytest.approx(0.5) == prior


def test_group_prior_falls_back_to_global(predictor_with_priors):
    prior = predictor_with_priors.get_group_prior(
        'T3000',
        {'apt': 'APT-Alpha', 'scenario_phase': 'lateral'},
    )

    global_profile = predictor_with_priors._prior_maps[predictor_with_priors._global_profile_id]
    assert pytest.approx(global_profile['T3000']) == prior


def test_group_prior_legacy_mode_uses_apt_profile(predictor_with_priors):
    context = {'apt': 'APT-Alpha', 'scenario_phase': 'lateral'}

    scenario_prior = predictor_with_priors.get_group_prior(
        'T2000',
        context,
        use_scenario_priors=True,
    )
    assert pytest.approx(0.9) == scenario_prior

    legacy_prior = predictor_with_priors.get_group_prior(
        'T2000',
        context,
        use_scenario_priors=False,
    )
    assert pytest.approx(0.4) == legacy_prior