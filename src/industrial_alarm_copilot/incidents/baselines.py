'''Deterministic statistical baselines for derived alarm episodes.'''

from dataclasses import dataclass

import pandas as pd


MACHINE_BASELINE_COLUMNS = [
    'machine_id',
    'quantile',
    'fit_split',
    'incident_count',
    'minimum_incident_count',
    'has_sufficient_support',
    'event_count_threshold',
    'duration_seconds_threshold',
    'distinct_alarm_count_threshold',
]
BASELINE_METRICS = (
    'event_count',
    'duration_seconds',
    'distinct_alarm_count',
)


@dataclass(frozen=True)
class GlobalIncidentBaseline:
    '''Train-only global upper-tail thresholds for episode statistics.'''

    quantile: float
    fit_split: str
    incident_count: int
    event_count_threshold: float
    duration_seconds_threshold: float
    distinct_alarm_count_threshold: float


def _higher_quantile(values: pd.Series, quantile: float) -> float:
    return float(values.quantile(quantile, interpolation='higher'))


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
        event_count_threshold=_higher_quantile(
            train_incidents['event_count'],
            quantile,
        ),
        duration_seconds_threshold=_higher_quantile(
            train_incidents['duration_seconds'],
            quantile,
        ),
        distinct_alarm_count_threshold=_higher_quantile(
            train_incidents['distinct_alarm_count'],
            quantile,
        ),
    )


def fit_machine_incident_baselines(
    incidents: pd.DataFrame,
    quantile: float = 0.95,
    minimum_incident_count: int = 200,
) -> pd.DataFrame:
    '''Fit one train-only upper-tail baseline per machine.'''
    if not 0 < quantile < 1:
        raise ValueError('quantile must be between 0 and 1')
    if minimum_incident_count <= 0:
        raise ValueError('minimum_incident_count must be greater than zero')

    train_incidents = incidents.loc[incidents['split'].eq('train')]
    if train_incidents.empty:
        raise ValueError('at least one train incident is required')

    baseline_rows = []
    for machine_id, machine_incidents in train_incidents.groupby(
        'machine_id',
        observed=True,
        sort=True,
    ):
        incident_count = len(machine_incidents)
        baseline_rows.append(
            {
                'machine_id': str(machine_id),
                'quantile': float(quantile),
                'fit_split': 'train',
                'incident_count': incident_count,
                'minimum_incident_count': int(minimum_incident_count),
                'has_sufficient_support': (
                    incident_count >= minimum_incident_count
                ),
                'event_count_threshold': _higher_quantile(
                    machine_incidents['event_count'],
                    quantile,
                ),
                'duration_seconds_threshold': _higher_quantile(
                    machine_incidents['duration_seconds'],
                    quantile,
                ),
                'distinct_alarm_count_threshold': _higher_quantile(
                    machine_incidents['distinct_alarm_count'],
                    quantile,
                ),
            }
        )

    return pd.DataFrame(baseline_rows, columns=MACHINE_BASELINE_COLUMNS)


def apply_incident_baseline_flags(
    incidents: pd.DataFrame,
    global_baseline: GlobalIncidentBaseline,
    machine_baselines: pd.DataFrame,
) -> pd.DataFrame:
    '''Apply machine thresholds with a global fallback to all incidents.'''
    normalized_baselines = machine_baselines.copy()
    normalized_baselines['machine_id'] = normalized_baselines[
        'machine_id'
    ].astype('string')
    if normalized_baselines['machine_id'].duplicated().any():
        raise ValueError('machine baselines must contain unique machine_id')

    baseline_by_machine = normalized_baselines.set_index('machine_id')
    machine_keys = incidents['machine_id'].astype('string')
    machine_support = machine_keys.map(
        baseline_by_machine['incident_count']
    )
    use_machine_baseline = (
        machine_keys.map(baseline_by_machine['has_sufficient_support'])
        .eq(True)
    )

    flagged_incidents = incidents.copy()
    flagged_incidents['baseline_scope'] = pd.Series(
        'global_fallback',
        index=flagged_incidents.index,
        dtype='string',
    )
    flagged_incidents.loc[use_machine_baseline, 'baseline_scope'] = 'machine'
    flagged_incidents['machine_train_incident_count'] = (
        machine_support.fillna(0).astype('int64')
    )
    flagged_incidents['baseline_train_incident_count'] = (
        machine_support.where(
            use_machine_baseline,
            global_baseline.incident_count,
        ).astype('int64')
    )
    flagged_incidents['baseline_quantile'] = global_baseline.quantile

    flag_columns = []
    for metric in BASELINE_METRICS:
        threshold_column = f'{metric}_threshold'
        flag_column = f'is_high_{metric}'
        machine_threshold = machine_keys.map(
            baseline_by_machine[threshold_column]
        )
        flagged_incidents[threshold_column] = machine_threshold.where(
            use_machine_baseline,
            getattr(global_baseline, threshold_column),
        ).astype(float)
        flagged_incidents[flag_column] = flagged_incidents[metric].gt(
            flagged_incidents[threshold_column]
        )
        flag_columns.append(flag_column)

    flagged_incidents['upper_tail_flag_count'] = (
        flagged_incidents[flag_columns].sum(axis=1).astype('int64')
    )
    flagged_incidents['is_upper_tail'] = flagged_incidents[
        'upper_tail_flag_count'
    ].gt(0)
    return flagged_incidents
