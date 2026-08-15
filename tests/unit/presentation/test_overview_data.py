'''Overview view-model tests.'''

import pandas as pd

from industrial_alarm_copilot.presentation.data import build_overview_data


def test_build_overview_data_aggregates_without_mutating_artifacts():
    events = pd.DataFrame(
        {
            'timestamp': pd.to_datetime(
                ['2020-01-01', '2020-01-02', '2020-02-01']
            ),
            'machine_id': ['1', '1', '2'],
            'alarm_code': ['11', '11', '98'],
        }
    )
    incidents = pd.DataFrame(
        {'split': ['train', 'validation', 'test', 'test']}
    )

    overview = build_overview_data(events, incidents)

    assert overview.event_count == 3
    assert overview.machine_count == 2
    assert overview.alarm_code_count == 2
    assert overview.incident_count == 4
    assert overview.monthly_events['event_count'].tolist() == [2, 1]
    assert overview.top_alarms.iloc[0].to_dict() == {
        'alarm_code': '11',
        'event_count': 2,
    }
    assert overview.split_incidents['incident_count'].tolist() == [1, 1, 2]
