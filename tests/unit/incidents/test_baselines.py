'''Deterministic incident baseline unit tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.incidents.baselines import (
    fit_global_incident_baseline,
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
