'''Complete ALPI incident analysis acceptance test.'''

from pathlib import Path

from industrial_alarm_copilot.data.loader import load_alarm_events
from industrial_alarm_copilot.data.pipeline import build_processed_events
from industrial_alarm_copilot.incidents.pipeline import (
    build_incident_analysis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_CSV_PATH = PROJECT_ROOT / 'data' / 'raw' / 'alarms.csv'


def test_complete_alpi_incident_analysis_is_consistent():
    events = load_alarm_events(RAW_CSV_PATH)
    processed_events = build_processed_events(events)
    analysis = build_incident_analysis(
        processed_events,
        gap_minutes=30,
        baseline_quantile=0.95,
        minimum_incident_count=200,
    )
    incidents = analysis.incidents
    mapping = analysis.incident_events

    assert incidents['split'].value_counts().to_dict() == {
        'train': 25_936,
        'test': 6_364,
        'validation': 5_853,
    }
    assert incidents['incident_id'].is_unique
    assert incidents['event_count'].sum() == 444_834
    assert len(mapping) == 444_834
    assert mapping['source_row'].is_unique
    assert set(mapping['incident_id']) == set(incidents['incident_id'])

    mapped_counts = mapping.groupby('incident_id', observed=True).size()
    incident_counts = incidents.set_index('incident_id')['event_count']
    assert mapped_counts.sort_index().equals(incident_counts.sort_index())
    assert mapping.groupby('incident_id', observed=True)[
        'event_position'
    ].min().eq(0).all()
    assert (
        mapping.groupby('incident_id', observed=True)['event_position']
        .max()
        .add(1)
        .eq(mapped_counts)
        .all()
    )

    assert analysis.global_baseline.incident_count == 25_936
    assert analysis.global_baseline.event_count_threshold == 42.0
    assert analysis.global_baseline.duration_seconds_threshold == 9_040.763
    assert (
        analysis.global_baseline.distinct_alarm_count_threshold == 6.0
    )

    machine_19 = analysis.machine_baselines.loc[
        analysis.machine_baselines['machine_id'].eq('19')
    ].iloc[0]
    assert machine_19['incident_count'] == 11
    assert not bool(machine_19['has_sufficient_support'])
    assert incidents['baseline_scope'].value_counts().to_dict() == {
        'machine': 38_138,
        'global_fallback': 15,
    }
    assert incidents.groupby('split', observed=True)[
        'is_upper_tail'
    ].sum().to_dict() == {
        'test': 434,
        'train': 2_050,
        'validation': 437,
    }
