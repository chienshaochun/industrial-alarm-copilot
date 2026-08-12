'''Future alarm outcomes used only for offline retrieval evaluation.'''

import pandas as pd


FUTURE_OUTCOME_COLUMNS = [
    'incident_id',
    'outcome_start_time',
    'outcome_end_time',
    'future_horizon_hours',
    'future_event_count',
    'distinct_future_alarm_count',
    'future_alarm_codes',
    'has_future_alarms',
]


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
    ordered_events = ordered_events.sort_values(
        ['machine_id', 'timestamp'],
        kind='stable',
    )
    events_by_machine = {
        str(machine_id): (
            machine_events['timestamp'].reset_index(drop=True),
            machine_events['alarm_code'].astype(str).reset_index(drop=True),
        )
        for machine_id, machine_events in ordered_events.groupby(
            'machine_id',
            observed=True,
            sort=False,
        )
    }

    horizon = pd.to_timedelta(float(future_horizon_hours), unit='h')
    records = []
    for incident in incidents.itertuples(index=False):
        outcome_start_time = incident.end_time
        outcome_end_time = outcome_start_time + horizon
        machine_events = events_by_machine.get(str(incident.machine_id))

        if machine_events is None:
            future_codes = []
        else:
            timestamps, alarm_codes = machine_events
            left = timestamps.searchsorted(
                outcome_start_time,
                side='right',
            )
            right = timestamps.searchsorted(
                outcome_end_time,
                side='right',
            )
            future_codes = alarm_codes.iloc[left:right].tolist()

        distinct_codes = tuple(sorted(set(future_codes)))
        records.append(
            {
                'incident_id': str(incident.incident_id),
                'outcome_start_time': outcome_start_time,
                'outcome_end_time': outcome_end_time,
                'future_horizon_hours': float(future_horizon_hours),
                'future_event_count': len(future_codes),
                'distinct_future_alarm_count': len(distinct_codes),
                'future_alarm_codes': distinct_codes,
                'has_future_alarms': bool(future_codes),
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=FUTURE_OUTCOME_COLUMNS,
    )
