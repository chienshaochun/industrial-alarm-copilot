'''Time-safe multi-label forecasting target tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.forecasting.labels import (
    build_forecast_labels,
)


def test_build_forecast_labels_uses_open_closed_window_and_same_machine():
    incidents = pd.DataFrame(
        {
            'incident_id': ['query'],
            'machine_id': ['4'],
            'split': ['train'],
            'end_time': pd.to_datetime(['2020-01-01 10:00:00']),
        }
    )
    events = pd.DataFrame(
        {
            'machine_id': ['4', '4', '4', '4', '4', '7'],
            'split': [
                'train',
                'train',
                'train',
                'train',
                'train',
                'train',
            ],
            'timestamp': pd.to_datetime(
                [
                    '2020-01-01 10:00:00',
                    '2020-01-01 10:00:01',
                    '2020-01-01 10:30:00',
                    '2020-01-01 11:00:00',
                    '2020-01-01 11:00:01',
                    '2020-01-01 10:30:00',
                ]
            ),
            'alarm_code': ['1', '98', '98', '11', '26', '137'],
        }
    )

    labels = build_forecast_labels(
        incidents,
        events,
        forecast_horizon_hours=1,
    ).set_index('incident_id')

    assert labels.loc['query', 'future_event_count'] == 3
    assert labels.loc['query', 'future_alarm_codes'] == ('11', '98')
    assert labels.loc['query', 'future_alarm_counts'] == (
        ('11', 1),
        ('98', 2),
    )
    assert bool(labels.loc['query', 'outcome_is_complete']) is True


def test_build_forecast_labels_does_not_cross_split_boundary():
    incidents = pd.DataFrame(
        {
            'incident_id': ['validation_query'],
            'machine_id': ['4'],
            'split': ['validation'],
            'end_time': pd.to_datetime(['2020-01-01 10:00:00']),
        }
    )
    events = pd.DataFrame(
        {
            'machine_id': ['4', '4', '4'],
            'split': ['validation', 'test', 'test'],
            'timestamp': pd.to_datetime(
                [
                    '2020-01-01 10:15:00',
                    '2020-01-01 10:30:00',
                    '2020-01-01 12:00:00',
                ]
            ),
            'alarm_code': ['11', '98', '26'],
        }
    )

    labels = build_forecast_labels(
        incidents,
        events,
        forecast_horizon_hours=1,
    ).set_index('incident_id')

    assert labels.loc['validation_query', 'future_event_count'] == 1
    assert labels.loc['validation_query', 'future_alarm_codes'] == ('11',)
    assert bool(
        labels.loc['validation_query', 'outcome_is_complete']
    ) is False


def test_build_forecast_labels_keeps_complete_empty_outcome():
    incidents = pd.DataFrame(
        {
            'incident_id': ['quiet_query'],
            'machine_id': ['4'],
            'split': ['train'],
            'end_time': pd.to_datetime(['2020-01-01 10:00:00']),
        }
    )
    events = pd.DataFrame(
        {
            'machine_id': ['4'],
            'split': ['train'],
            'timestamp': pd.to_datetime(['2020-01-01 12:00:00']),
            'alarm_code': ['26'],
        }
    )

    labels = build_forecast_labels(
        incidents,
        events,
        forecast_horizon_hours=1,
    ).set_index('incident_id')

    assert bool(labels.loc['quiet_query', 'outcome_is_complete']) is True
    assert labels.loc['quiet_query', 'future_event_count'] == 0
    assert labels.loc['quiet_query', 'distinct_future_alarm_count'] == 0
    assert labels.loc['quiet_query', 'future_alarm_codes'] == ()
    assert labels.loc['quiet_query', 'future_alarm_counts'] == ()
    assert bool(labels.loc['quiet_query', 'has_future_alarms']) is False


@pytest.mark.parametrize('invalid_horizon', [0, -1, float('nan')])
def test_build_forecast_labels_rejects_invalid_horizon(invalid_horizon):
    incidents = pd.DataFrame(
        {
            'incident_id': ['query'],
            'machine_id': ['4'],
            'split': ['train'],
            'end_time': pd.to_datetime(['2020-01-01 10:00:00']),
        }
    )
    events = pd.DataFrame(
        columns=['machine_id', 'split', 'timestamp', 'alarm_code']
    )

    with pytest.raises(ValueError, match='finite and positive'):
        build_forecast_labels(
            incidents,
            events,
            forecast_horizon_hours=invalid_horizon,
        )


def test_build_forecast_labels_rejects_duplicate_incident_id():
    incidents = pd.DataFrame(
        {
            'incident_id': ['duplicate', 'duplicate'],
            'machine_id': ['4', '4'],
            'split': ['train', 'train'],
            'end_time': pd.to_datetime(
                ['2020-01-01 10:00:00', '2020-01-01 11:00:00']
            ),
        }
    )
    events = pd.DataFrame(
        columns=['machine_id', 'split', 'timestamp', 'alarm_code']
    )

    with pytest.raises(ValueError, match='incident_id'):
        build_forecast_labels(
            incidents,
            events,
            forecast_horizon_hours=1,
        )
