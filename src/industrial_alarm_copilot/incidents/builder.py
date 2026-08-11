'''Build time-gap-derived alarm episodes.'''

import pandas as pd


def mark_incident_starts(
    events: pd.DataFrame,
    gap_minutes: float = 30.0,
) -> pd.DataFrame:
    '''Sort events and mark the first event of each derived episode.'''
    if gap_minutes <= 0:
        raise ValueError('gap_minutes must be greater than zero')

    marked_events = events.sort_values(
        ['machine_id', 'timestamp', 'source_row'],
        kind='stable',
    ).reset_index(drop=True)

    machine_groups = marked_events.groupby(
        'machine_id',
        observed=True,
        sort=False,
    )
    first_for_machine = machine_groups.cumcount().eq(0)
    previous_split = machine_groups['split'].shift(1)
    split_changed = previous_split.notna() & marked_events['split'].ne(
        previous_split
    )
    gap_exceeded = marked_events['gap_seconds'].gt(gap_minutes * 60)

    marked_events['is_incident_start'] = (
        first_for_machine | split_changed | gap_exceeded.fillna(False)
    )
    return marked_events


def assign_incident_numbers(
    events: pd.DataFrame,
    gap_minutes: float = 30.0,
) -> pd.DataFrame:
    '''Assign a zero-based number to each derived alarm episode.'''
    numbered_events = mark_incident_starts(events, gap_minutes=gap_minutes)
    numbered_events['incident_number'] = (
        numbered_events['is_incident_start'].cumsum().sub(1).astype('int64')
    )
    return numbered_events
