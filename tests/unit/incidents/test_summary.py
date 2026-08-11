'''Incident summary unit tests.'''

import pandas as pd

from industrial_alarm_copilot.incidents.summary import build_incident_summary


def test_build_incident_summary_calculates_episode_time_span_and_size():
    events = pd.DataFrame(
        {
            'timestamp': pd.to_datetime(
                [
                    '2019-01-01 10:50:00',
                    '2019-01-01 09:00:00',
                    '2019-01-01 10:00:00',
                    '2019-01-01 11:20:01',
                    '2019-01-01 10:25:00',
                ]
            ),
            'alarm_code': ['98', '26', '98', '11', '11'],
            'machine_id': ['4', '6', '4', '4', '4'],
            'source_row': [2, 4, 0, 3, 1],
            'split': ['train', 'train', 'train', 'train', 'train'],
            'gap_seconds': [1500.0, float('nan'), float('nan'), 1801.0, 1500.0],
            'is_exact_duplicate': [True, False, False, False, True],
        }
    )

    summary = build_incident_summary(events, gap_minutes=30)

    assert summary['machine_id'].tolist() == ['4', '4', '6']
    assert summary['event_count'].tolist() == [3, 1, 1]
    assert summary['duration_seconds'].tolist() == [3000.0, 0.0, 0.0]
    assert summary.loc[0, 'start_time'] == pd.Timestamp(
        '2019-01-01 10:00:00'
    )
    assert summary.loc[0, 'end_time'] == pd.Timestamp(
        '2019-01-01 10:50:00'
    )
    assert summary.loc[0, 'distinct_alarm_count'] == 2
    assert summary.loc[0, 'first_alarm_code'] == '98'
    assert summary.loc[0, 'last_alarm_code'] == '98'
    assert summary.loc[0, 'dominant_alarm_code'] == '98'
    assert summary.loc[0, 'duplicate_event_count'] == 2
    assert summary['incident_id'].is_unique


def test_build_incident_summary_breaks_dominant_alarm_ties_by_string():
    events = pd.DataFrame(
        {
            'timestamp': pd.to_datetime(
                ['2019-01-01 10:00:00', '2019-01-01 10:01:00']
            ),
            'alarm_code': ['2', '11'],
            'machine_id': ['4', '4'],
            'source_row': [0, 1],
            'split': ['train', 'train'],
            'gap_seconds': [float('nan'), 60.0],
            'is_exact_duplicate': [False, False],
        }
    )

    summary = build_incident_summary(events, gap_minutes=30)

    assert summary.loc[0, 'dominant_alarm_code'] == '11'
