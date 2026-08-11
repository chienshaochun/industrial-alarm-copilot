'''Episode-level statistical summary construction.'''

import pandas as pd

from industrial_alarm_copilot.incidents.builder import assign_incident_ids


INCIDENT_SUMMARY_COLUMNS = [
    'incident_id',
    'machine_id',
    'split',
    'start_time',
    'end_time',
    'duration_seconds',
    'event_count',
    'distinct_alarm_count',
    'first_alarm_code',
    'last_alarm_code',
    'dominant_alarm_code',
    'duplicate_event_count',
]


def _select_dominant_alarm(alarm_codes: pd.Series) -> str:
    '''Select the most frequent code with a deterministic string tie-break.'''
    alarm_counts = alarm_codes.astype('string').value_counts(sort=False)
    highest_count = alarm_counts.max()
    tied_codes = alarm_counts.index[alarm_counts.eq(highest_count)]
    return min(str(code) for code in tied_codes)


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
        distinct_alarm_count=('alarm_code', 'nunique'),
        first_alarm_code=('alarm_code', 'first'),
        last_alarm_code=('alarm_code', 'last'),
        dominant_alarm_code=('alarm_code', _select_dominant_alarm),
        duplicate_event_count=('is_exact_duplicate', 'sum'),
    ).reset_index(drop=True)
    summary['duration_seconds'] = (
        summary['end_time'] - summary['start_time']
    ).dt.total_seconds()
    summary['duplicate_event_count'] = summary[
        'duplicate_event_count'
    ].astype('int64')

    return summary[INCIDENT_SUMMARY_COLUMNS]
