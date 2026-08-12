'''Proxy relevance tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.retrieval.relevance import (
    jaccard_alarm_codes,
    label_retrieval_relevance,
)


def test_jaccard_alarm_codes_compares_unique_alarm_sets():
    assert jaccard_alarm_codes(
        ['11', '98', '98'],
        ['98', '26'],
    ) == pytest.approx(1 / 3)
    assert jaccard_alarm_codes([], []) == 0.0


def test_label_retrieval_relevance_tracks_evaluation_eligibility():
    retrieval_results = pd.DataFrame(
        {
            'query_incident_id': ['query', 'query', 'query'],
            'candidate_incident_id': [
                'candidate_good',
                'candidate_too_recent',
                'candidate_incomplete',
            ],
            'rank': [1, 2, 3],
        }
    )
    incidents = pd.DataFrame(
        {
            'incident_id': ['query'],
            'start_time': pd.to_datetime(['2020-01-02 10:00:00']),
        }
    )
    outcomes = pd.DataFrame(
        {
            'incident_id': [
                'query',
                'candidate_good',
                'candidate_too_recent',
                'candidate_incomplete',
            ],
            'outcome_end_time': pd.to_datetime(
                [
                    '2020-01-02 12:00:00',
                    '2020-01-02 09:00:00',
                    '2020-01-02 10:00:00',
                    '2020-01-02 08:00:00',
                ]
            ),
            'outcome_is_complete': [True, True, True, False],
            'has_future_alarms': [True, True, True, True],
            'future_alarm_codes': [
                ('11', '98'),
                ('98', '26'),
                ('11', '98'),
                ('11', '98'),
            ],
        }
    )

    labeled = label_retrieval_relevance(
        retrieval_results,
        outcomes,
        incidents,
        relevance_threshold=0.3,
    )

    assert labeled['evaluation_eligible'].tolist() == [
        True,
        False,
        False,
    ]
    assert labeled.loc[0, 'outcome_jaccard'] == pytest.approx(1 / 3)
    assert bool(labeled.loc[0, 'is_relevant']) is True
    assert pd.isna(labeled.loc[1, 'is_relevant'])
    assert pd.isna(labeled.loc[2, 'is_relevant'])
    assert labeled[
        'candidate_outcome_available_before_query'
    ].tolist() == [True, False, True]
