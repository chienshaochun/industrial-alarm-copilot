'''Time-safe retrieval candidate selection unit tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.retrieval.candidates import (
    select_historical_candidates,
)


def _build_incidents():
    return pd.DataFrame(
        {
            'incident_id': pd.Series(
                [
                    'inc_train_old',
                    'inc_validation_old',
                    'inc_equal_boundary',
                    'inc_overlap',
                    'inc_query',
                    'inc_future',
                ],
                dtype='string',
            ),
            'split': [
                'train',
                'validation',
                'train',
                'train',
                'test',
                'test',
            ],
            'start_time': pd.to_datetime(
                [
                    '2019-01-01 08:00:00',
                    '2019-01-01 08:30:00',
                    '2019-01-01 09:00:00',
                    '2019-01-01 09:30:00',
                    '2019-01-01 10:00:00',
                    '2019-01-01 11:00:00',
                ]
            ),
            'end_time': pd.to_datetime(
                [
                    '2019-01-01 08:10:00',
                    '2019-01-01 08:40:00',
                    '2019-01-01 10:00:00',
                    '2019-01-01 10:05:00',
                    '2019-01-01 10:30:00',
                    '2019-01-01 11:10:00',
                ]
            ),
        }
    )


def test_expanding_history_keeps_only_strictly_earlier_episodes():
    candidates = select_historical_candidates(
        _build_incidents(),
        'inc_query',
        policy='expanding_history',
    )

    assert candidates['incident_id'].tolist() == [
        'inc_train_old',
        'inc_validation_old',
    ]


def test_train_only_excludes_non_train_history():
    candidates = select_historical_candidates(
        _build_incidents(),
        'inc_query',
        policy='train_only',
    )

    assert candidates['incident_id'].tolist() == ['inc_train_old']


def test_candidate_selection_rejects_unknown_query():
    with pytest.raises(KeyError, match='inc_missing'):
        select_historical_candidates(_build_incidents(), 'inc_missing')
