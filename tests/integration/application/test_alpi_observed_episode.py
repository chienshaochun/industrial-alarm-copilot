'''Observed episode assembly against the complete ALPI dataset.'''

from pathlib import Path

from industrial_alarm_copilot.application.observed import (
    build_observed_episode,
)
from industrial_alarm_copilot.data.loader import load_alarm_events
from industrial_alarm_copilot.data.pipeline import build_processed_events
from industrial_alarm_copilot.incidents.pipeline import (
    build_incident_analysis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_CSV_PATH = PROJECT_ROOT / 'data' / 'raw' / 'alarms.csv'


def test_complete_alpi_episode_can_be_reconstructed_for_presentation():
    events = load_alarm_events(RAW_CSV_PATH)
    processed_events = build_processed_events(events)
    analysis = build_incident_analysis(
        processed_events,
        gap_minutes=30,
        baseline_quantile=0.95,
        minimum_incident_count=200,
    )
    test_incidents = analysis.incidents.loc[
        analysis.incidents['split'].eq('test')
    ]
    selected = test_incidents.sort_values(
        ['event_count', 'incident_id'],
        ascending=[False, True],
        kind='stable',
    ).iloc[0]

    observed = build_observed_episode(
        analysis.incidents,
        processed_events,
        analysis.incident_events,
        incident_id=str(selected['incident_id']),
    )

    assert observed.incident_id == str(selected['incident_id'])
    assert observed.machine_id == str(selected['machine_id'])
    assert observed.split == 'test'
    assert len(observed.alarm_sequence) == int(selected['event_count'])
    assert observed.alarm_sequence[0].gap_seconds is None
    timestamps = [event.timestamp for event in observed.alarm_sequence]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == observed.start_time
    assert timestamps[-1] == observed.end_time
    assert all(
        event.gap_seconds is None or event.gap_seconds >= 0
        for event in observed.alarm_sequence
    )
