'''Offline retrieval evaluation candidate tests.'''

import pandas as pd

from industrial_alarm_copilot.retrieval.evaluation import (
    select_outcome_evaluable_candidates,
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
