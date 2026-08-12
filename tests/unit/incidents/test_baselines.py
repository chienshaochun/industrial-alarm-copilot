'''Deterministic incident baseline unit tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.incidents.baselines import (
    fit_global_incident_baseline,
    fit_machine_incident_baselines,
)


def test_fit_global_incident_baseline_uses_train_split_only():
    incidents = pd.DataFrame(
        {
            'split': [
                'train',
                'train',
                'train',
                'train',
                'validation',
            ],
            'event_count': [1, 2, 3, 4, 999],
            'duration_seconds': [0.0, 10.0, 20.0, 30.0, 99999.0],
            'distinct_alarm_count': [1, 1, 2, 3, 99],
        }
    )

    baseline = fit_global_incident_baseline(incidents, quantile=0.95)

    assert baseline.quantile == 0.95
    assert baseline.fit_split == 'train'
    assert baseline.incident_count == 4
    assert baseline.event_count_threshold == 4.0
    assert baseline.duration_seconds_threshold == 30.0
    assert baseline.distinct_alarm_count_threshold == 3.0


def test_fit_global_incident_baseline_requires_train_incidents():
    incidents = pd.DataFrame(
        {
            'split': ['validation'],
            'event_count': [10],
            'duration_seconds': [60.0],
            'distinct_alarm_count': [2],
        }
    )

    with pytest.raises(ValueError, match='train incident'):
        fit_global_incident_baseline(incidents)


def test_fit_machine_incident_baselines_uses_each_machines_train_history():
    incidents = pd.DataFrame(
        {
            'machine_id': ['6', '4', '4', '6', '4', '6'],
            'split': [
                'train',
                'validation',
                'train',
                'train',
                'train',
                'validation',
            ],
            'event_count': [10, 999, 1, 20, 4, 999],
            'duration_seconds': [100.0, 9999.0, 10.0, 200.0, 40.0, 9999.0],
            'distinct_alarm_count': [2, 99, 1, 5, 3, 99],
        }
    )

    baselines = fit_machine_incident_baselines(
        incidents,
        quantile=0.95,
        minimum_incident_count=2,
    )

    assert baselines['machine_id'].tolist() == ['4', '6']
    assert baselines['incident_count'].tolist() == [2, 2]
    assert baselines['minimum_incident_count'].eq(2).all()
    assert baselines['has_sufficient_support'].eq(True).all()
    assert baselines['event_count_threshold'].tolist() == [4.0, 20.0]
    assert baselines['duration_seconds_threshold'].tolist() == [40.0, 200.0]
    assert baselines['distinct_alarm_count_threshold'].tolist() == [
        3.0,
        5.0,
    ]
    assert baselines['fit_split'].eq('train').all()


def test_fit_machine_incident_baselines_marks_insufficient_support():
    incidents = pd.DataFrame(
        {
            'machine_id': ['4', '4', '19'],
            'split': ['train', 'train', 'train'],
            'event_count': [2, 4, 10],
            'duration_seconds': [10.0, 40.0, 100.0],
            'distinct_alarm_count': [1, 3, 5],
        }
    )

    baselines = fit_machine_incident_baselines(
        incidents,
        minimum_incident_count=2,
    )

    assert baselines['machine_id'].tolist() == ['19', '4']
    assert baselines['incident_count'].tolist() == [1, 2]
    assert baselines['has_sufficient_support'].tolist() == [False, True]
