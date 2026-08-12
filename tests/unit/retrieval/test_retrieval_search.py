'''Exact episode retrieval tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.retrieval.features import (
    fit_alarm_tfidf,
    transform_alarm_documents,
)
from industrial_alarm_copilot.retrieval.search import (
    RETRIEVAL_RESULT_COLUMNS,
    retrieve_similar_episodes,
)


def _build_retrieval_inputs():
    incidents = pd.DataFrame(
        {
            'incident_id': [
                'inc_query',
                'inc_future_perfect',
                'inc_tie_b',
                'inc_old_best',
                'inc_equal_boundary',
                'inc_tie_a',
            ],
            'machine_id': ['4', '4', '6', '4', '4', '7'],
            'start_time': pd.to_datetime(
                [
                    '2020-01-01 10:00:00',
                    '2020-01-01 11:00:00',
                    '2020-01-01 09:00:00',
                    '2020-01-01 08:00:00',
                    '2020-01-01 09:45:00',
                    '2020-01-01 09:05:00',
                ]
            ),
            'end_time': pd.to_datetime(
                [
                    '2020-01-01 10:15:00',
                    '2020-01-01 11:05:00',
                    '2020-01-01 09:30:00',
                    '2020-01-01 08:10:00',
                    '2020-01-01 10:00:00',
                    '2020-01-01 09:30:00',
                ]
            ),
            'split': [
                'test',
                'train',
                'train',
                'train',
                'train',
                'train',
            ],
        }
    )
    documents = pd.DataFrame(
        {
            'incident_id': incidents['incident_id'],
            'alarm_document': [
                '98 11',
                '98 11',
                '98 26',
                '98 11',
                '98 11',
                '98 26',
            ],
        }
    )
    fitted_tfidf = fit_alarm_tfidf(documents, incidents)
    features = transform_alarm_documents(fitted_tfidf, documents)
    return incidents, documents, features


def test_retrieve_similar_episodes_is_time_safe_and_deterministic():
    incidents, documents, features = _build_retrieval_inputs()

    results = retrieve_similar_episodes(
        incidents,
        documents,
        features,
        query_incident_id='inc_query',
        top_k=10,
    )

    assert results['candidate_incident_id'].tolist() == [
        'inc_old_best',
        'inc_tie_a',
        'inc_tie_b',
    ]
    assert results['rank'].tolist() == [1, 2, 3]
    assert results['same_machine'].tolist() == [True, False, False]
    assert results['shared_alarm_codes'].tolist() == [
        ('11', '98'),
        ('98',),
        ('98',),
    ]
    assert results.loc[0, 'similarity_score'] == pytest.approx(1.0)
    assert 'inc_future_perfect' not in set(
        results['candidate_incident_id']
    )
    assert 'inc_equal_boundary' not in set(
        results['candidate_incident_id']
    )


def test_retrieve_similar_episodes_returns_empty_when_history_is_empty():
    incidents, documents, features = _build_retrieval_inputs()

    results = retrieve_similar_episodes(
        incidents,
        documents,
        features,
        query_incident_id='inc_old_best',
    )

    assert results.empty
    assert results.columns.tolist() == RETRIEVAL_RESULT_COLUMNS
