'''Incident event mapping unit tests.'''

import pandas as pd

from industrial_alarm_copilot.incidents.mapping import (
    build_incident_event_mapping,
)


def test_build_incident_event_mapping_preserves_rows_and_episode_order():
    events = pd.DataFrame(
        {
            'timestamp': pd.to_datetime(
                [
                    '2019-01-01 09:01:00',
                    '2019-01-01 10:05:00',
                    '2019-01-01 10:00:00',
                    '2019-01-01 10:35:01',
                    '2019-01-01 09:00:00',
                ]
            ),
            'machine_id': ['6', '4', '4', '4', '6'],
            'source_row': [4, 1, 0, 2, 3],
            'split': ['train', 'train', 'train', 'train', 'train'],
            'gap_seconds': [60.0, 300.0, float('nan'), 1801.0, float('nan')],
        }
    )

    mapping = build_incident_event_mapping(events, gap_minutes=30)

    assert mapping.columns.tolist() == [
        'incident_id',
        'source_row',
        'event_position',
    ]
    assert mapping['source_row'].tolist() == [0, 1, 2, 3, 4]
    assert mapping['event_position'].tolist() == [0, 1, 0, 0, 1]
    assert mapping['incident_id'].nunique() == 3
    assert mapping['source_row'].is_unique
