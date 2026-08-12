'''Offline retrieval evaluation candidate tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.retrieval.evaluation import (
    evaluate_retrieval_query,
    select_outcome_evaluable_candidates,
)
from industrial_alarm_copilot.retrieval.features import (
    fit_alarm_tfidf,
    transform_alarm_documents,
)


def test_select_outcome_evaluable_candidates_applies_second_time_gate():
    incidents = pd.DataFrame(
        {
            'incident_id': [
                'old_complete',
                'outcome_equal_boundary',
                'outcome_incomplete',
                'query',
                'future_episode',
            ],
            'start_time': pd.to_datetime(
                [
                    '2020-01-01 07:00:00',
                    '2020-01-01 08:00:00',
                    '2020-01-01 08:30:00',
                    '2020-01-01 10:00:00',
                    '2020-01-01 11:00:00',
                ]
            ),
            'end_time': pd.to_datetime(
                [
                    '2020-01-01 07:30:00',
                    '2020-01-01 08:30:00',
                    '2020-01-01 09:00:00',
                    '2020-01-01 10:30:00',
                    '2020-01-01 11:30:00',
                ]
            ),
            'split': ['train', 'train', 'train', 'test', 'train'],
        }
    )
    outcomes = pd.DataFrame(
        {
            'incident_id': incidents['incident_id'],
            'outcome_end_time': pd.to_datetime(
                [
                    '2020-01-01 09:00:00',
                    '2020-01-01 10:00:00',
                    '2020-01-01 09:30:00',
                    '2020-01-01 12:00:00',
                    '2020-01-01 13:00:00',
                ]
            ),
            'outcome_is_complete': [True, True, False, True, True],
        }
    )

    candidates = select_outcome_evaluable_candidates(
        incidents,
        outcomes,
        query_incident_id='query',
    )

    assert candidates['incident_id'].tolist() == ['old_complete']


def _build_query_evaluation_inputs():
    incidents = pd.DataFrame(
        {
            'incident_id': [
                'candidate_relevant',
                'candidate_irrelevant',
                'candidate_outcome_too_recent',
                'query',
            ],
            'machine_id': ['4', '6', '7', '4'],
            'start_time': pd.to_datetime(
                [
                    '2020-01-01 07:00:00',
                    '2020-01-01 08:00:00',
                    '2020-01-01 09:00:00',
                    '2020-01-01 10:00:00',
                ]
            ),
            'end_time': pd.to_datetime(
                [
                    '2020-01-01 07:30:00',
                    '2020-01-01 08:30:00',
                    '2020-01-01 09:30:00',
                    '2020-01-01 10:30:00',
                ]
            ),
            'split': ['train', 'train', 'train', 'test'],
        }
    )
    documents = pd.DataFrame(
        {
            'incident_id': incidents['incident_id'],
            'alarm_document': ['26', '98 11', '98 11', '98 11'],
        }
    )
    fitted_tfidf = fit_alarm_tfidf(documents, incidents)
    features = transform_alarm_documents(fitted_tfidf, documents)
    outcomes = pd.DataFrame(
        {
            'incident_id': incidents['incident_id'],
            'outcome_end_time': pd.to_datetime(
                [
                    '2020-01-01 08:30:00',
                    '2020-01-01 09:30:00',
                    '2020-01-01 10:00:00',
                    '2020-01-01 11:30:00',
                ]
            ),
            'outcome_is_complete': [True, True, True, True],
            'has_future_alarms': [True, True, True, True],
            'future_alarm_codes': [
                ('11', '98'),
                ('26',),
                ('11', '98'),
                ('11', '98'),
            ],
        }
    )
    return incidents, documents, features, outcomes


def test_evaluate_retrieval_query_uses_full_pool_and_ranked_top_five():
    incidents, documents, features, outcomes = (
        _build_query_evaluation_inputs()
    )

    evaluation = evaluate_retrieval_query(
        incidents,
        documents,
        features,
        outcomes,
        query_incident_id='query',
        relevance_threshold=0.5,
        top_k=5,
    )

    assert evaluation.status == 'evaluated'
    assert evaluation.evaluable_candidate_count == 2
    assert evaluation.total_relevant_candidate_count == 1
    assert evaluation.ranked_results[
        'candidate_incident_id'
    ].tolist() == ['candidate_irrelevant', 'candidate_relevant']
    assert evaluation.ranked_results['is_relevant'].tolist() == [
        False,
        True,
    ]
    assert evaluation.metrics is not None
    assert evaluation.metrics.hit_at_k is True
    assert evaluation.metrics.precision_at_k == pytest.approx(1 / 5)
    assert evaluation.metrics.recall_at_k == pytest.approx(1.0)
    assert evaluation.metrics.reciprocal_rank == pytest.approx(1 / 2)


def test_evaluate_retrieval_query_preserves_incomplete_query_coverage():
    incidents, documents, features, outcomes = (
        _build_query_evaluation_inputs()
    )
    outcomes.loc[
        outcomes['incident_id'].eq('query'),
        'outcome_is_complete',
    ] = False

    evaluation = evaluate_retrieval_query(
        incidents,
        documents,
        features,
        outcomes,
        query_incident_id='query',
        relevance_threshold=0.5,
        top_k=5,
    )

    assert evaluation.status == 'query_outcome_incomplete'
    assert evaluation.metrics is None
    assert evaluation.ranked_results.empty
