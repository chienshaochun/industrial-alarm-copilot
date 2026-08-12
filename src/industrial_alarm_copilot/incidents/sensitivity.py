'''Sensitivity analysis for time-gap-derived alarm episodes.'''

from collections.abc import Iterable

import pandas as pd

from industrial_alarm_copilot.incidents.summary import build_incident_summary


SENSITIVITY_COLUMNS = [
    'gap_minutes',
    'incident_count',
    'singleton_incident_count',
    'singleton_incident_share',
    'median_event_count',
    'p95_event_count',
    'median_duration_seconds',
    'p95_duration_seconds',
    'mean_distinct_alarm_count',
]


def build_incident_sensitivity_profile(
    events: pd.DataFrame,
    gap_minutes_values: Iterable[float],
) -> pd.DataFrame:
    '''Compare episode statistics across multiple time-gap thresholds.'''
    profile_rows = []

    for gap_minutes in gap_minutes_values:
        incidents = build_incident_summary(
            events,
            gap_minutes=float(gap_minutes),
        )
        singleton_mask = incidents['event_count'].eq(1)
        profile_rows.append(
            {
                'gap_minutes': float(gap_minutes),
                'incident_count': len(incidents),
                'singleton_incident_count': int(singleton_mask.sum()),
                'singleton_incident_share': float(singleton_mask.mean()),
                'median_event_count': float(
                    incidents['event_count'].median()
                ),
                'p95_event_count': float(
                    incidents['event_count'].quantile(0.95)
                ),
                'median_duration_seconds': float(
                    incidents['duration_seconds'].median()
                ),
                'p95_duration_seconds': float(
                    incidents['duration_seconds'].quantile(0.95)
                ),
                'mean_distinct_alarm_count': float(
                    incidents['distinct_alarm_count'].mean()
                ),
            }
        )

    return pd.DataFrame(profile_rows, columns=SENSITIVITY_COLUMNS)
