'''Deterministic statistical baselines for derived alarm episodes.'''

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GlobalIncidentBaseline:
    '''Train-only global upper-tail thresholds for episode statistics.'''

    quantile: float
    fit_split: str
    incident_count: int
    event_count_threshold: float
    duration_seconds_threshold: float
    distinct_alarm_count_threshold: float


def fit_global_incident_baseline(
    incidents: pd.DataFrame,
    quantile: float = 0.95,
) -> GlobalIncidentBaseline:
    '''Fit deterministic upper-tail thresholds using train incidents only.'''
    if not 0 < quantile < 1:
        raise ValueError('quantile must be between 0 and 1')

    train_incidents = incidents.loc[incidents['split'].eq('train')]
    if train_incidents.empty:
        raise ValueError('at least one train incident is required')

    return GlobalIncidentBaseline(
        quantile=float(quantile),
        fit_split='train',
        incident_count=len(train_incidents),
        event_count_threshold=float(
            train_incidents['event_count'].quantile(
                quantile,
                interpolation='higher',
            )
        ),
        duration_seconds_threshold=float(
            train_incidents['duration_seconds'].quantile(
                quantile,
                interpolation='higher',
            )
        ),
        distinct_alarm_count_threshold=float(
            train_incidents['distinct_alarm_count'].quantile(
                quantile,
                interpolation='higher',
            )
        ),
    )
