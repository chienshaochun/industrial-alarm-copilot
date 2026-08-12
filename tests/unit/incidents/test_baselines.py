'''Deterministic incident baseline unit tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.incidents.baselines import (
    GlobalIncidentBaseline,
    apply_incident_baseline_flags,
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


def test_apply_incident_baseline_flags_uses_global_fallback():
    incidents = pd.DataFrame(
        {
            'incident_id': ['inc_4', 'inc_19', 'inc_new'],
            'machine_id': ['4', '19', '99'],
            'event_count': [5, 11, 10],
            'duration_seconds': [40.0, 100.0, 101.0],
            'distinct_alarm_count': [3, 3, 4],
        }
    )
    global_baseline = GlobalIncidentBaseline(
        quantile=0.95,
        fit_split='train',
        incident_count=1000,
        event_count_threshold=10.0,
        duration_seconds_threshold=100.0,
        distinct_alarm_count_threshold=3.0,
    )
    machine_baselines = pd.DataFrame(
        {
            'machine_id': ['4', '19'],
            'incident_count': [217, 11],
            'has_sufficient_support': [True, False],
            'event_count_threshold': [4.0, 999.0],
            'duration_seconds_threshold': [40.0, 999.0],
            'distinct_alarm_count_threshold': [2.0, 99.0],
        }
    )

    flagged = apply_incident_baseline_flags(
        incidents,
        global_baseline,
        machine_baselines,
    )

    assert flagged['baseline_scope'].tolist() == [
        'machine',
        'global_fallback',
        'global_fallback',
    ]
    assert flagged['machine_train_incident_count'].tolist() == [217, 11, 0]
    assert flagged['baseline_train_incident_count'].tolist() == [
        217,
        1000,
        1000,
    ]
    assert flagged['event_count_threshold'].tolist() == [4.0, 10.0, 10.0]
    assert flagged['is_high_event_count'].tolist() == [True, True, False]
    assert flagged['is_high_duration_seconds'].tolist() == [
        False,
        False,
        True,
    ]
    assert flagged['is_high_distinct_alarm_count'].tolist() == [
        True,
        False,
        True,
    ]
    assert flagged['upper_tail_flag_count'].tolist() == [2, 1, 2]
    assert flagged['is_upper_tail'].eq(True).all()
