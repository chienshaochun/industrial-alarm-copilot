'''Forecast label construction with explicit split boundaries.'''

import math

import numpy as np
import pandas as pd


FORECAST_LABEL_COLUMNS = [
    'incident_id',
    'machine_id',
    'split',
    'prediction_time',
    'forecast_end_time',
    'outcome_observed_through',
    'outcome_is_complete',
    'forecast_horizon_hours',
    'future_event_count',
    'distinct_future_alarm_count',
    'future_alarm_codes',
    'future_alarm_counts',
    'has_future_alarms',
]


def build_forecast_labels(
    incidents: pd.DataFrame,
    events: pd.DataFrame,
    forecast_horizon_hours: float,
) -> pd.DataFrame:
    '''Collect same-machine, same-split alarms in an open-closed window.'''
    if (
        not math.isfinite(float(forecast_horizon_hours))
        or forecast_horizon_hours <= 0
    ):
        raise ValueError('forecast_horizon_hours must be finite and positive')
    if not incidents['incident_id'].is_unique:
        raise ValueError('incidents incident_id must be unique')

    ordered_events = events[
        ['machine_id', 'split', 'timestamp', 'alarm_code']
    ].copy()
    ordered_events['machine_id'] = ordered_events[
        'machine_id'
    ].astype(str)
    ordered_events['split'] = ordered_events['split'].astype(str)
    ordered_events['alarm_code'] = ordered_events['alarm_code'].astype(str)
    ordered_events = ordered_events.sort_values(
        ['machine_id', 'split', 'timestamp'],
        kind='stable',
    )
    events_by_scope = {
        (str(machine_id), str(split)): (
            scope_events['timestamp'].to_numpy(dtype='datetime64[ns]'),
            scope_events['alarm_code'].to_numpy(dtype=str),
        )
        for (machine_id, split), scope_events in ordered_events.groupby(
            ['machine_id', 'split'],
            observed=True,
            sort=False,
        )
    }

    incident_count = len(incidents)
    incident_ids = incidents['incident_id'].astype(str).to_numpy()
    incident_machines = incidents['machine_id'].astype(str).to_numpy()
    incident_splits = incidents['split'].astype(str).to_numpy()
    prediction_times = incidents['end_time'].to_numpy(
        dtype='datetime64[ns]'
    )
    horizon = pd.to_timedelta(
        float(forecast_horizon_hours),
        unit='h',
    ).to_timedelta64()
    forecast_end_times = prediction_times + horizon
    outcome_observed_through = np.full(
        incident_count,
        np.datetime64('NaT', 'ns'),
        dtype='datetime64[ns]',
    )
    outcome_is_complete = np.zeros(incident_count, dtype=bool)
    future_event_count = np.zeros(incident_count, dtype=np.int64)
    distinct_future_alarm_count = np.zeros(
        incident_count,
        dtype=np.int64,
    )
    future_alarm_codes = [()] * incident_count
    future_alarm_counts = [()] * incident_count

    incident_rows_by_scope: dict[tuple[str, str], list[int]] = {}
    for row_number, scope in enumerate(
        zip(incident_machines, incident_splits, strict=True)
    ):
        incident_rows_by_scope.setdefault(scope, []).append(row_number)

    for scope, row_numbers_list in incident_rows_by_scope.items():
        scope_events = events_by_scope.get(scope)
        if scope_events is None:
            continue

        timestamps, alarm_codes = scope_events
        row_numbers = np.asarray(row_numbers_list, dtype=np.int64)
        left_boundaries = np.searchsorted(
            timestamps,
            prediction_times[row_numbers],
            side='right',
        )
        right_boundaries = np.searchsorted(
            timestamps,
            forecast_end_times[row_numbers],
            side='right',
        )
        observed_through = timestamps[-1]
        outcome_observed_through[row_numbers] = observed_through
        outcome_is_complete[row_numbers] = (
            observed_through >= forecast_end_times[row_numbers]
        )
        future_event_count[row_numbers] = (
            right_boundaries - left_boundaries
        )

        for row_number, left, right in zip(
            row_numbers,
            left_boundaries,
            right_boundaries,
            strict=True,
        ):
            distinct_codes, code_counts = np.unique(
                alarm_codes[left:right],
                return_counts=True,
            )
            future_alarm_codes[row_number] = tuple(distinct_codes.tolist())
            future_alarm_counts[row_number] = tuple(
                (str(code), int(count))
                for code, count in zip(
                    distinct_codes,
                    code_counts,
                    strict=True,
                )
            )
            distinct_future_alarm_count[row_number] = len(distinct_codes)

    return pd.DataFrame(
        {
            'incident_id': incident_ids,
            'machine_id': incident_machines,
            'split': incident_splits,
            'prediction_time': prediction_times,
            'forecast_end_time': forecast_end_times,
            'outcome_observed_through': outcome_observed_through,
            'outcome_is_complete': outcome_is_complete,
            'forecast_horizon_hours': float(forecast_horizon_hours),
            'future_event_count': future_event_count,
            'distinct_future_alarm_count': distinct_future_alarm_count,
            'future_alarm_codes': future_alarm_codes,
            'future_alarm_counts': future_alarm_counts,
            'has_future_alarms': future_event_count > 0,
        },
        columns=FORECAST_LABEL_COLUMNS,
    )
