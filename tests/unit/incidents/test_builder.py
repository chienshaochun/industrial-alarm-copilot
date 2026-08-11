'''Derived alarm episode builder unit tests.'''

import pandas as pd

from industrial_alarm_copilot.incidents.builder import (
    assign_incident_numbers,
    mark_incident_starts,
)


def test_mark_incident_starts_respects_gap_split_and_machine_boundaries():
    events = pd.DataFrame(
        {
            'timestamp': pd.to_datetime(
                [
                    '2019-01-01 11:06:00',
                    '2019-01-01 09:00:00',
                    '2019-01-01 10:00:00',
                    '2019-01-01 10:05:00',
                    '2019-01-01 10:35:00',
                    '2019-01-01 11:05:01',
                    '2019-01-01 11:07:00',
                ]
            ),
            'machine_id': ['4', '6', '4', '4', '4', '4', '4'],
            'source_row': [4, 6, 0, 1, 2, 3, 5],
            'split': [
                'validation',
                'train',
                'train',
                'train',
                'train',
                'train',
                'validation',
            ],
            'gap_seconds': [
                59.0,
                float('nan'),
                float('nan'),
                300.0,
                1800.0,
                1801.0,
                60.0,
            ],
        }
    )

    result = mark_incident_starts(events, gap_minutes=30)

    assert result['source_row'].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert result['is_incident_start'].tolist() == [
        True,
        False,
        False,
        True,
        True,
        False,
        True,
    ]


def test_assign_incident_numbers_increments_only_at_episode_starts():
    events = pd.DataFrame(
        {
            'timestamp': pd.to_datetime(
                [
                    '2019-01-01 10:00:00',
                    '2019-01-01 10:05:00',
                    '2019-01-01 10:35:01',
                    '2019-01-01 10:36:00',
                ]
            ),
            'machine_id': ['4', '4', '4', '6'],
            'source_row': [0, 1, 2, 3],
            'split': ['train', 'train', 'train', 'train'],
            'gap_seconds': [float('nan'), 300.0, 1801.0, float('nan')],
        }
    )

    result = assign_incident_numbers(events, gap_minutes=30)

    assert result['incident_number'].tolist() == [0, 0, 1, 2]
