'''Observed episode assembly tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.application.observed import (
    build_observed_episode,
)


def test_build_observed_episode_restores_sequence_and_upper_tail_facts():
    incidents = pd.DataFrame(
        {
            'incident_id': ['inc_other', 'inc_query'],
            'machine_id': ['9', '4'],
            'split': ['train', 'test'],
            'start_time': pd.to_datetime(
                ['2020-05-01 08:00:00', '2020-05-18 10:32:05']
            ),
            'end_time': pd.to_datetime(
                ['2020-05-01 08:00:00', '2020-05-18 11:14:05']
            ),
            'duration_seconds': [0.0, 2520.0],
            'event_count': [1, 3],
            'distinct_alarm_count': [1, 2],
            'is_high_event_count': [False, False],
            'is_high_duration_seconds': [False, True],
            'is_high_distinct_alarm_count': [False, False],
            'is_upper_tail': [False, True],
        }
    )
    events = pd.DataFrame(
        {
            'source_row': [12, 99, 10, 11],
            'timestamp': pd.to_datetime(
                [
                    '2020-05-18 11:14:05',
                    '2020-05-01 08:00:00',
                    '2020-05-18 10:32:05',
                    '2020-05-18 10:32:48',
                ]
            ),
            'alarm_code': ['98', '26', '11', '98'],
            'gap_seconds': [2477.0, None, 2400.0, 43.0],
        }
    )
    mapping = pd.DataFrame(
        {
            'incident_id': [
                'inc_query',
                'inc_other',
                'inc_query',
                'inc_query',
            ],
            'source_row': [11, 99, 12, 10],
            'event_position': [1, 0, 2, 0],
        }
    )

    observed = build_observed_episode(
        incidents,
        events,
        mapping,
        incident_id='inc_query',
    )

    assert observed.incident_id == 'inc_query'
    assert observed.machine_id == '4'
    assert observed.split == 'test'
    assert observed.event_count == 3
    assert observed.is_upper_tail is True
    assert observed.upper_tail_flags == ('high_duration',)
    assert [event.alarm_code for event in observed.alarm_sequence] == [
        '11',
        '98',
        '98',
    ]
    assert [event.gap_seconds for event in observed.alarm_sequence] == [
        None,
        43.0,
        2477.0,
    ]


def _single_incident_inputs():
    incidents = pd.DataFrame(
        {
            'incident_id': ['inc_query'],
            'machine_id': ['4'],
            'split': ['test'],
            'start_time': pd.to_datetime(['2020-05-18 10:32:05']),
            'end_time': pd.to_datetime(['2020-05-18 10:34:05']),
            'duration_seconds': [120.0],
            'event_count': [3],
            'distinct_alarm_count': [2],
            'is_high_event_count': [False],
            'is_high_duration_seconds': [False],
            'is_high_distinct_alarm_count': [False],
            'is_upper_tail': [False],
        }
    )
    events = pd.DataFrame(
        {
            'source_row': [10, 11, 12],
            'timestamp': pd.to_datetime(
                [
                    '2020-05-18 10:32:05',
                    '2020-05-18 10:33:05',
                    '2020-05-18 10:34:05',
                ]
            ),
            'alarm_code': ['11', '98', '98'],
            'gap_seconds': [2400.0, 60.0, 60.0],
        }
    )
    mapping = pd.DataFrame(
        {
            'incident_id': ['inc_query'] * 3,
            'source_row': [10, 11, 12],
            'event_position': [0, 1, 2],
        }
    )
    return incidents, events, mapping


def test_build_observed_episode_rejects_unknown_incident():
    incidents, events, mapping = _single_incident_inputs()

    with pytest.raises(KeyError, match='missing_incident'):
        build_observed_episode(
            incidents,
            events,
            mapping,
            incident_id='missing_incident',
        )


def test_build_observed_episode_rejects_missing_mapped_event():
    incidents, events, mapping = _single_incident_inputs()
    mapping.loc[1, 'source_row'] = 999

    with pytest.raises(ValueError, match='source_row'):
        build_observed_episode(
            incidents,
            events,
            mapping,
            incident_id='inc_query',
        )


def test_build_observed_episode_rejects_noncontiguous_positions():
    incidents, events, mapping = _single_incident_inputs()
    mapping['event_position'] = [0, 2, 3]

    with pytest.raises(ValueError, match='event_position'):
        build_observed_episode(
            incidents,
            events,
            mapping,
            incident_id='inc_query',
        )
