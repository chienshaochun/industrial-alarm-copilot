'''Convert application contracts into display-only tables.'''

from dataclasses import asdict

import pandas as pd

from industrial_alarm_copilot.application.contracts import (
    InvestigationResult,
)


def observed_event_frame(result: InvestigationResult) -> pd.DataFrame:
    rows = [
        {
            'event_position': position,
            'timestamp': event.timestamp,
            'alarm_code': event.alarm_code,
            'gap_seconds': event.gap_seconds,
        }
        for position, event in enumerate(
            result.observed.alarm_sequence,
            start=1,
        )
    ]
    return pd.DataFrame(rows)


def retrieved_evidence_frame(result: InvestigationResult) -> pd.DataFrame:
    rows = [
        {
            'rank': evidence.rank,
            'incident_id': evidence.incident_id,
            'machine_id': evidence.machine_id,
            'start_time': evidence.start_time,
            'similarity_score': evidence.similarity_score,
            'shared_alarm_codes': ', '.join(evidence.shared_alarm_codes),
            'future_alarm_codes': ', '.join(evidence.future_alarm_codes),
            'outcome_is_complete': evidence.outcome_is_complete,
            'future_horizon_hours': evidence.future_horizon_hours,
        }
        for evidence in result.retrieved_evidence
    ]
    return pd.DataFrame(rows)


def forecast_prediction_frame(result: InvestigationResult) -> pd.DataFrame:
    return pd.DataFrame(
        [asdict(prediction) for prediction in result.predictions]
    )
