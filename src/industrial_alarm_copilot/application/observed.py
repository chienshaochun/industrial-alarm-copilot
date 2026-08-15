'''Assemble presentation-safe observed facts from processed artifacts.'''

import pandas as pd

from industrial_alarm_copilot.application.contracts import (
    AlarmEventFact,
    ObservedEpisode,
)


UPPER_TAIL_COLUMNS = (
    ('is_high_event_count', 'high_event_count'),
    ('is_high_duration_seconds', 'high_duration'),
    ('is_high_distinct_alarm_count', 'high_alarm_diversity'),
)


def build_observed_episode(
    incidents: pd.DataFrame,
    events: pd.DataFrame,
    incident_events: pd.DataFrame,
    incident_id: str,
) -> ObservedEpisode:
    '''Reconstruct one ordered episode without recalculating Stage 4 facts.'''
    normalized_incident_id = str(incident_id)
    selected_incident = incidents.loc[
        incidents['incident_id'].astype(str).eq(normalized_incident_id)
    ]
    if len(selected_incident) != 1:
        raise KeyError(f'unknown or duplicate incident_id: {incident_id}')
    incident = selected_incident.iloc[0]

    selected_mapping = incident_events.loc[
        incident_events['incident_id'].astype(str).eq(
            normalized_incident_id
        )
    ][['source_row', 'event_position']]
    mapped_events = selected_mapping.merge(
        events[['source_row', 'timestamp', 'alarm_code', 'gap_seconds']],
        on='source_row',
        how='left',
        sort=False,
        validate='one_to_one',
        indicator=True,
    ).sort_values('event_position', kind='stable')
    if not mapped_events['_merge'].eq('both').all():
        missing_source_rows = mapped_events.loc[
            mapped_events['_merge'].ne('both'),
            'source_row',
        ].astype(str)
        raise ValueError(
            'source_row does not reference an event: '
            + ', '.join(missing_source_rows)
        )

    expected_event_count = int(incident['event_count'])
    expected_positions = list(range(expected_event_count))
    actual_positions = mapped_events['event_position'].tolist()
    if actual_positions != expected_positions:
        raise ValueError(
            'event_position must be contiguous and match event_count'
        )

    alarm_sequence = tuple(
        AlarmEventFact(
            timestamp=row.timestamp.to_pydatetime(),
            alarm_code=str(row.alarm_code),
            gap_seconds=(
                None
                if row.event_position == 0
                else float(row.gap_seconds)
            ),
        )
        for row in mapped_events.itertuples(index=False)
    )
    upper_tail_flags = tuple(
        flag_name
        for column_name, flag_name in UPPER_TAIL_COLUMNS
        if bool(incident[column_name])
    )
    return ObservedEpisode(
        incident_id=normalized_incident_id,
        machine_id=str(incident['machine_id']),
        split=str(incident['split']),
        start_time=incident['start_time'].to_pydatetime(),
        end_time=incident['end_time'].to_pydatetime(),
        duration_seconds=float(incident['duration_seconds']),
        event_count=int(incident['event_count']),
        distinct_alarm_count=int(incident['distinct_alarm_count']),
        is_upper_tail=bool(incident['is_upper_tail']),
        upper_tail_flags=upper_tail_flags,
        alarm_sequence=alarm_sequence,
    )
