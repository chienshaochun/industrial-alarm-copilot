'''Episode-level statistical summary construction.'''

import pandas as pd

from industrial_alarm_copilot.incidents.builder import assign_incident_ids


TEMPORAL_SUMMARY_COLUMNS = [
    'incident_id',
    'machine_id',
    'split',
    'start_time',
    'end_time',
    'duration_seconds',
    'event_count',
]


def build_incident_summary(
    events: pd.DataFrame,
    gap_minutes: float = 30.0,
) -> pd.DataFrame:
    '''Build one temporal summary row per derived alarm episode.'''
    identified_events = assign_incident_ids(
        events,
        gap_minutes=gap_minutes,
    )
    grouped_events = identified_events.groupby(
        'incident_number',
        observed=True,
        sort=False,
    )
    summary = grouped_events.agg(
        incident_id=('incident_id', 'first'),
        machine_id=('machine_id', 'first'),
        split=('split', 'first'),
        start_time=('timestamp', 'first'),
        end_time=('timestamp', 'last'),
        event_count=('source_row', 'size'),
    ).reset_index(drop=True)
    summary['duration_seconds'] = (
        summary['end_time'] - summary['start_time']
    ).dt.total_seconds()

    return summary[TEMPORAL_SUMMARY_COLUMNS]
