'''Investigation service output contract tests.'''

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from industrial_alarm_copilot.application.contracts import (
    AlarmEventFact,
    ForecastPrediction,
    InvestigationResult,
    ObservedEpisode,
    RetrievedEpisodeEvidence,
)


def test_investigation_result_keeps_evidence_types_separate_and_immutable():
    observed = ObservedEpisode(
        incident_id='inc_query',
        machine_id='4',
        split='test',
        start_time=datetime(2020, 5, 18, 10, 32, 5),
        end_time=datetime(2020, 5, 18, 11, 14, 5),
        duration_seconds=2520.0,
        event_count=2,
        distinct_alarm_count=2,
        is_upper_tail=True,
        upper_tail_flags=('high_duration',),
        alarm_sequence=(
            AlarmEventFact(
                timestamp=datetime(2020, 5, 18, 10, 32, 5),
                alarm_code='11',
                gap_seconds=None,
            ),
            AlarmEventFact(
                timestamp=datetime(2020, 5, 18, 11, 14, 5),
                alarm_code='98',
                gap_seconds=2520.0,
            ),
        ),
    )
    evidence = RetrievedEpisodeEvidence(
        rank=1,
        incident_id='inc_history',
        machine_id='4',
        start_time=datetime(2019, 11, 3, 14, 21),
        end_time=datetime(2019, 11, 3, 15, 8),
        similarity_score=0.73,
        shared_alarm_codes=('11', '98'),
        future_alarm_codes=('26',),
    )
    prediction = ForecastPrediction(
        rank=1,
        alarm_code='98',
        model_score=0.42,
        model_version='transition_frequency_v1',
        forecast_horizon_hours=6.0,
        baseline_scope='transition',
        train_support=459,
    )

    result = InvestigationResult(
        observed=observed,
        retrieved_evidence=(evidence,),
        predictions=(prediction,),
        limitations=(
            'Alarm codes are anonymous.',
            'Predictions are statistical candidates.',
        ),
    )

    assert result.observed.incident_id == 'inc_query'
    assert result.retrieved_evidence[0].incident_id == 'inc_history'
    assert result.predictions[0].alarm_code == '98'
    assert result.observed.alarm_sequence[1].gap_seconds == 2520.0
    with pytest.raises(FrozenInstanceError):
        result.observed.event_count = 3


@pytest.mark.parametrize(
    ('field_name', 'field_value'),
    [
        ('rank', 0),
        ('similarity_score', 1.5),
    ],
)
def test_retrieved_evidence_rejects_invalid_ranking_values(
    field_name,
    field_value,
):
    values = {
        'rank': 1,
        'incident_id': 'inc_history',
        'machine_id': '4',
        'start_time': datetime(2019, 11, 3, 14, 21),
        'end_time': datetime(2019, 11, 3, 15, 8),
        'similarity_score': 0.73,
        'shared_alarm_codes': ('11', '98'),
        'future_alarm_codes': ('26',),
    }
    values[field_name] = field_value

    with pytest.raises(ValueError, match=field_name):
        RetrievedEpisodeEvidence(**values)


def test_observed_episode_rejects_inconsistent_counts():
    with pytest.raises(ValueError, match='distinct_alarm_count'):
        ObservedEpisode(
            incident_id='inc_query',
            machine_id='4',
            split='test',
            start_time=datetime(2020, 5, 18, 10, 32),
            end_time=datetime(2020, 5, 18, 10, 33),
            duration_seconds=60.0,
            event_count=1,
            distinct_alarm_count=2,
            is_upper_tail=False,
            upper_tail_flags=(),
            alarm_sequence=(
                AlarmEventFact(
                    timestamp=datetime(2020, 5, 18, 10, 32),
                    alarm_code='11',
                    gap_seconds=None,
                ),
            ),
        )


def test_alarm_event_fact_rejects_negative_gap():
    with pytest.raises(ValueError, match='gap_seconds'):
        AlarmEventFact(
            timestamp=datetime(2020, 5, 18, 10, 32),
            alarm_code='11',
            gap_seconds=-1.0,
        )


@pytest.mark.parametrize(
    ('field_name', 'field_value'),
    [
        ('rank', 0),
        ('train_support', -1),
        ('baseline_scope', 'future_lookup'),
    ],
)
def test_forecast_prediction_rejects_invalid_provenance(
    field_name,
    field_value,
):
    values = {
        'rank': 1,
        'alarm_code': '98',
        'model_score': 0.42,
        'model_version': 'transition_frequency_v1',
        'forecast_horizon_hours': 6.0,
        'baseline_scope': 'transition',
        'train_support': 459,
    }
    values[field_name] = field_value

    with pytest.raises(ValueError, match=field_name):
        ForecastPrediction(**values)
