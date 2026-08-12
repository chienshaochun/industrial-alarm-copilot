'''Incident gap sensitivity profile unit tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.incidents.sensitivity import (
    build_incident_sensitivity_profile,
)


def test_build_incident_sensitivity_profile_compares_gap_resolutions():
    events = pd.DataFrame(
        {
            'timestamp': pd.to_datetime(
                [
                    '2019-01-01 10:00:00',
                    '2019-01-01 10:10:00',
                    '2019-01-01 10:40:00',
                    '2019-01-01 11:40:00',
                ]
            ),
            'alarm_code': ['11', '11', '26', '98'],
            'machine_id': ['4', '4', '4', '4'],
            'source_row': [0, 1, 2, 3],
            'split': ['train', 'train', 'train', 'train'],
            'gap_seconds': [float('nan'), 600.0, 1800.0, 3600.0],
            'is_exact_duplicate': [False, False, False, False],
        }
    )

    profile = build_incident_sensitivity_profile(events, [15, 30, 60])

    assert profile['gap_minutes'].tolist() == [15.0, 30.0, 60.0]
    assert profile['incident_count'].tolist() == [3, 2, 1]
    assert profile['singleton_incident_count'].tolist() == [2, 1, 0]
    assert profile['singleton_incident_share'].tolist() == pytest.approx(
        [2 / 3, 0.5, 0.0]
    )
    assert profile['median_event_count'].tolist() == [1.0, 2.0, 4.0]
    assert profile['median_duration_seconds'].tolist() == [
        0.0,
        1200.0,
        6000.0,
    ]
    assert profile['mean_distinct_alarm_count'].tolist() == [
        1.0,
        1.5,
        3.0,
    ]
