'''Future alarm outcome tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.retrieval.outcomes import (
    build_future_alarm_outcomes,
)


def test_build_future_alarm_outcomes_respects_machine_and_boundaries():
    incidents = pd.DataFrame(
        {
            'incident_id': ['inc_1', 'inc_without_future'],
            'machine_id': ['4', '9'],
            'end_time': pd.to_datetime(
                ['2020-01-01 10:00:00', '2020-01-01 12:00:00']
            ),
        }
    )
    events = pd.DataFrame(
        {
            'machine_id': ['4', '4', '4', '4', '4', '7'],
            'timestamp': pd.to_datetime(
                [
                    '2020-01-01 10:00:00',
                    '2020-01-01 10:00:01',
                    '2020-01-01 10:30:00',
                    '2020-01-01 11:00:00',
                    '2020-01-01 11:00:00.001',
                    '2020-01-01 10:10:00',
                ],
                format='mixed',
            ),
            'alarm_code': ['1', '98', '98', '11', '26', '137'],
        }
    )

    outcomes = build_future_alarm_outcomes(
        incidents,
        events,
        future_horizon_hours=1,
    ).set_index('incident_id')

    assert outcomes.loc['inc_1', 'future_event_count'] == 3
    assert outcomes.loc['inc_1', 'distinct_future_alarm_count'] == 2
    assert outcomes.loc['inc_1', 'future_alarm_codes'] == ('11', '98')
    assert bool(outcomes.loc['inc_1', 'has_future_alarms']) is True
    assert outcomes.loc['inc_1', 'outcome_end_time'] == pd.Timestamp(
        '2020-01-01 11:00:00'
    )
    assert bool(outcomes.loc['inc_1', 'outcome_is_complete']) is True
    assert outcomes.loc['inc_without_future', 'future_alarm_codes'] == ()
    assert bool(
        outcomes.loc['inc_without_future', 'has_future_alarms']
    ) is False
    assert bool(
        outcomes.loc['inc_without_future', 'outcome_is_complete']
    ) is False


def test_build_future_alarm_outcomes_rejects_nonpositive_horizon():
    incidents = pd.DataFrame(
        {
            'incident_id': ['inc_1'],
            'machine_id': ['4'],
            'end_time': pd.to_datetime(['2020-01-01 10:00:00']),
        }
    )
    events = pd.DataFrame(
        columns=['machine_id', 'timestamp', 'alarm_code']
    )

    with pytest.raises(ValueError, match='greater than zero'):
        build_future_alarm_outcomes(
            incidents,
            events,
            future_horizon_hours=0,
        )
