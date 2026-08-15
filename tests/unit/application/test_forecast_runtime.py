'''Portable forecast artifact runtime tests.'''

from datetime import datetime

import pytest

from industrial_alarm_copilot.application.contracts import (
    AlarmEventFact,
    ObservedEpisode,
)
from industrial_alarm_copilot.application.forecast import (
    parse_runtime_forecast_model,
    predict_future_alarms,
)


def _payload():
    return {
        'model_version': 'transition_frequency_v1',
        'alarm_codes': ['11', '26', '98'],
        'global_scores': [0.4, 0.2, 0.8],
        'machine_scores': [
            {'machine_id': '4', 'scores': [0.7, 0.3, 0.5]},
        ],
        'machine_train_sample_counts': [
            {'machine_id': '4', 'sample_count': 500},
            {'machine_id': '9', 'sample_count': 300},
        ],
        'transition_scores': [
            {
                'machine_id': '4',
                'last_alarm_code': '98',
                'scores': [0.2, 0.9, 0.6],
            }
        ],
        'transition_train_sample_counts': [
            {
                'machine_id': '4',
                'last_alarm_code': '98',
                'sample_count': 120,
            }
        ],
    }


def _observed(machine_id='4', last_alarm_code='98'):
    timestamp = datetime(2020, 5, 18, 10, 32)
    return ObservedEpisode(
        incident_id='inc_query',
        machine_id=machine_id,
        split='test',
        start_time=timestamp,
        end_time=timestamp,
        duration_seconds=0.0,
        event_count=1,
        distinct_alarm_count=1,
        is_upper_tail=False,
        upper_tail_flags=(),
        alarm_sequence=(
            AlarmEventFact(timestamp, last_alarm_code, None),
        ),
    )


@pytest.mark.parametrize(
    ('machine_id', 'last_alarm_code', 'scope', 'support', 'top_alarm'),
    [
        ('4', '98', 'transition', 120, '26'),
        ('4', 'unknown', 'machine_fallback', 500, '11'),
        ('new', 'unknown', 'global_fallback', 800, '98'),
    ],
)
def test_forecast_runtime_uses_hierarchical_fallback(
    machine_id,
    last_alarm_code,
    scope,
    support,
    top_alarm,
):
    model = parse_runtime_forecast_model(_payload())

    predictions = predict_future_alarms(
        model,
        _observed(machine_id, last_alarm_code),
        top_k=2,
    )

    assert [prediction.rank for prediction in predictions] == [1, 2]
    assert predictions[0].alarm_code == top_alarm
    assert {prediction.baseline_scope for prediction in predictions} == {scope}
    assert {prediction.train_support for prediction in predictions} == {support}


def test_forecast_runtime_rejects_misaligned_score_vector():
    payload = _payload()
    payload['global_scores'] = [0.1, 0.2]

    with pytest.raises(ValueError, match='global_scores'):
        parse_runtime_forecast_model(payload)
