import pandas as pd
import pytest

from src.chain_paths.eval import Evaluator


class DummyPredictor:
    def __init__(self):
        # Keep bootstrap rounds small for fast tests
        self.params = {
            'use_scenario_priors': True,
            'bootstrap_rounds': 10,
            'significance_alpha': 0.05,
        }

    def next_probabilities(
        self,
        history,
        group_context=None,
        top_n_candidates=100,
        use_scenario_priors=None,
    ):
        flag = (
            self.params['use_scenario_priors']
            if use_scenario_priors is None
            else bool(use_scenario_priors)
        )

        if flag:
            return [
                ('T2000', 0.9, {'P_count': 0.6}),
                ('T3000', 0.1, {'P_count': 0.4}),
            ]

        return [
            ('T3000', 0.9, {'P_count': 0.6}),
            ('T2000', 0.1, {'P_count': 0.4}),
        ]


def _dummy_sequences():
    return [
        ['T1000', 'T1001'],
        ['T2000', 'T2001'],
        ['T3000', 'T3001'],
    ]


def _dummy_metadata():
    return pd.DataFrame({
        'sequence_id': [0, 1, 2],
        'campaign_id': ['c0', 'c1', 'c2'],
        'group_id': ['', '', ''],
        'sequence_length': [2, 2, 2],
        'unique_techniques': [2, 2, 2],
        'start_time': pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01']),
        'end_time': pd.to_datetime(['2022-01-02', '2022-02-02', '2022-03-02']),
    })


def test_temporal_split_orders_sequences():
    sequences = _dummy_sequences()
    metadata = _dummy_metadata()

    evaluator = Evaluator(predictor=None, sequences=sequences, sequence_metadata=metadata)
    train, test = evaluator.split_sequences(train_ratio=2/3, temporal_split=True)

    assert train == sequences[:2]
    assert test == sequences[2:]


def test_temporal_split_without_metadata_raises():
    sequences = _dummy_sequences()
    evaluator = Evaluator(predictor=None, sequences=sequences)

    with pytest.raises(ValueError):
        evaluator.split_sequences(temporal_split=True)


def test_prior_variant_comparison_prefers_scenario():
    sequences = [['T1000', 'T2000']]
    predictor = DummyPredictor()
    evaluator = Evaluator(predictor=predictor, sequences=sequences)

    contexts = [
        {
            'apt': 'APT-X',
            'scenario_phase': 'lateral',
            'persistence_focus': 'credentials',
        }
    ]

    comparison = evaluator.evaluate_prior_variants(sequences, test_contexts=contexts)

    assert comparison['selected_variant'] == 'scenario_priors'
    assert evaluator.predictor.params['use_scenario_priors'] is True

    scenario_mrr = comparison['scenario_priors']['metrics']['model']['MRR']['mean']
    legacy_mrr = comparison['legacy_priors']['metrics']['model']['MRR']['mean']
    assert scenario_mrr > legacy_mrr

    assert comparison['scenario_priors']['details']['use_scenario_priors'] is True
    assert comparison['legacy_priors']['details']['use_scenario_priors'] is False
