'''Future alarm outcomes used only for offline retrieval evaluation.'''

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


FUTURE_OUTCOME_COLUMNS = [
    'incident_id',
    'outcome_start_time',
    'outcome_end_time',
    'outcome_observed_through',
    'outcome_is_complete',
    'future_horizon_hours',
    'future_event_count',
    'distinct_future_alarm_count',
    'future_alarm_codes',
    'has_future_alarms',
]


@dataclass(frozen=True)
class OutcomeAlarmMatrix:
    '''Sparse binary future-alarm sets aligned with incident IDs.'''

    incident_ids: tuple[str, ...]
    alarm_codes: tuple[str, ...]
    matrix: sparse.csr_matrix
    alarm_counts: np.ndarray
    row_by_incident_id: dict[str, int]


def build_future_alarm_outcomes(
    incidents: pd.DataFrame,
    events: pd.DataFrame,
    future_horizon_hours: float,
) -> pd.DataFrame:
    '''Collect same-machine alarms after each episode within a fixed horizon.'''
    if future_horizon_hours <= 0:
        raise ValueError('future_horizon_hours must be greater than zero')
    if not incidents['incident_id'].is_unique:
        raise ValueError('incidents incident_id must be unique')

    ordered_events = events[
        ['machine_id', 'timestamp', 'alarm_code']
    ].copy()
    ordered_events['machine_id'] = ordered_events[
        'machine_id'
    ].astype(str)
    ordered_events['alarm_code'] = ordered_events['alarm_code'].astype(str)
    ordered_events = ordered_events.sort_values(
        ['machine_id', 'timestamp'],
        kind='stable',
    )
    events_by_machine = {
        str(machine_id): (
            machine_events['timestamp'].to_numpy(dtype='datetime64[ns]'),
            machine_events['alarm_code'].to_numpy(dtype=str),
        )
        for machine_id, machine_events in ordered_events.groupby(
            'machine_id',
            observed=True,
            sort=False,
        )
    }

    incident_count = len(incidents)
    incident_ids = incidents['incident_id'].astype(str).to_numpy()
    incident_machines = incidents['machine_id'].astype(str).to_numpy()
    outcome_start_times = incidents['end_time'].to_numpy(
        dtype='datetime64[ns]'
    )
    horizon = pd.to_timedelta(
        float(future_horizon_hours),
        unit='h',
    ).to_timedelta64()
    outcome_end_times = outcome_start_times + horizon
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

    incident_rows_by_machine: dict[str, list[int]] = {}
    for row_number, machine_id in enumerate(incident_machines):
        incident_rows_by_machine.setdefault(machine_id, []).append(row_number)

    for machine_id, row_numbers_list in incident_rows_by_machine.items():
        machine_events = events_by_machine.get(machine_id)
        if machine_events is None:
            continue

        timestamps, alarm_codes = machine_events
        row_numbers = np.asarray(row_numbers_list, dtype=np.int64)
        left_boundaries = np.searchsorted(
            timestamps,
            outcome_start_times[row_numbers],
            side='right',
        )
        right_boundaries = np.searchsorted(
            timestamps,
            outcome_end_times[row_numbers],
            side='right',
        )
        observed_through = timestamps[-1]
        outcome_observed_through[row_numbers] = observed_through
        outcome_is_complete[row_numbers] = (
            observed_through >= outcome_end_times[row_numbers]
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
            distinct_codes = tuple(
                np.unique(alarm_codes[left:right]).tolist()
            )
            future_alarm_codes[row_number] = distinct_codes
            distinct_future_alarm_count[row_number] = len(distinct_codes)

    return pd.DataFrame(
        {
            'incident_id': incident_ids,
            'outcome_start_time': outcome_start_times,
            'outcome_end_time': outcome_end_times,
            'outcome_observed_through': outcome_observed_through,
            'outcome_is_complete': outcome_is_complete,
            'future_horizon_hours': float(future_horizon_hours),
            'future_event_count': future_event_count,
            'distinct_future_alarm_count': distinct_future_alarm_count,
            'future_alarm_codes': future_alarm_codes,
            'has_future_alarms': future_event_count > 0,
        },
        columns=FUTURE_OUTCOME_COLUMNS,
    )


def build_outcome_alarm_matrix(
    outcomes: pd.DataFrame,
) -> OutcomeAlarmMatrix:
    '''Encode exact future alarm sets as a sparse binary matrix.'''
    if not outcomes['incident_id'].is_unique:
        raise ValueError('outcomes incident_id must be unique')

    incident_ids = tuple(outcomes['incident_id'].astype(str))
    normalized_sets = [
        tuple(sorted(set(str(code) for code in alarm_codes)))
        for alarm_codes in outcomes['future_alarm_codes']
    ]
    alarm_codes = tuple(
        sorted({code for codes in normalized_sets for code in codes})
    )
    column_by_alarm_code = {
        alarm_code: column_number
        for column_number, alarm_code in enumerate(alarm_codes)
    }
    row_numbers = []
    column_numbers = []
    for row_number, codes in enumerate(normalized_sets):
        row_numbers.extend([row_number] * len(codes))
        column_numbers.extend(
            column_by_alarm_code[code] for code in codes
        )

    matrix = sparse.csr_matrix(
        (
            np.ones(len(row_numbers), dtype=np.int32),
            (row_numbers, column_numbers),
        ),
        shape=(len(incident_ids), len(alarm_codes)),
        dtype=np.int32,
    )
    alarm_counts = np.asarray(matrix.sum(axis=1)).ravel()
    return OutcomeAlarmMatrix(
        incident_ids=incident_ids,
        alarm_codes=alarm_codes,
        matrix=matrix,
        alarm_counts=alarm_counts,
        row_by_incident_id={
            incident_id: row_number
            for row_number, incident_id in enumerate(incident_ids)
        },
    )


def compute_outcome_jaccard_scores(
    outcome_matrix: OutcomeAlarmMatrix,
    query_incident_id: str,
    candidate_incident_ids: tuple[str, ...],
) -> np.ndarray:
    '''Vectorize exact Jaccard scores from one query to many candidates.'''
    try:
        query_row = outcome_matrix.row_by_incident_id[
            str(query_incident_id)
        ]
        candidate_rows = [
            outcome_matrix.row_by_incident_id[str(incident_id)]
            for incident_id in candidate_incident_ids
        ]
    except KeyError as error:
        raise ValueError('every incident must exist in outcome matrix') from error

    intersections = (
        outcome_matrix.matrix[candidate_rows]
        @ outcome_matrix.matrix[query_row].transpose()
    ).toarray().ravel()
    unions = (
        outcome_matrix.alarm_counts[candidate_rows]
        + outcome_matrix.alarm_counts[query_row]
        - intersections
    )
    return np.divide(
        intersections,
        unions,
        out=np.zeros(len(candidate_rows), dtype=float),
        where=unions != 0,
    )
