'''Event-to-episode mapping table construction.'''

import pandas as pd

from industrial_alarm_copilot.incidents.builder import assign_incident_ids


MAPPING_COLUMNS = ['incident_id', 'source_row', 'event_position']


def build_incident_event_mapping(
    events: pd.DataFrame,
    gap_minutes: float = 30.0,
) -> pd.DataFrame:
    '''Map every alarm event to one episode and its position within it.'''
    identified_events = assign_incident_ids(
        events,
        gap_minutes=gap_minutes,
    )
    mapping = identified_events[['incident_id', 'source_row']].copy()
    mapping['event_position'] = (
        identified_events.groupby(
            'incident_number',
            observed=True,
            sort=False,
        )
        .cumcount()
        .astype('int64')
    )
    return mapping[MAPPING_COLUMNS]
