'''Build alarm-code documents from incident event mappings.'''

import pandas as pd


ALARM_DOCUMENT_COLUMNS = [
    'incident_id',
    'alarm_document',
    'alarm_token_count',
]


def build_alarm_documents(
    events: pd.DataFrame,
    incident_events: pd.DataFrame,
) -> pd.DataFrame:
    '''Create one ordered, duplicate-preserving alarm document per episode.'''
    if not events['source_row'].is_unique:
        raise ValueError('events source_row must be unique')
    if not incident_events['source_row'].is_unique:
        raise ValueError('incident_events source_row must be unique')

    mapped_events = incident_events.merge(
        events[['source_row', 'alarm_code']],
        on='source_row',
        how='left',
        sort=False,
        validate='one_to_one',
    )
    if mapped_events['alarm_code'].isna().any():
        raise ValueError('every mapped source_row must reference an event')

    mapped_events = mapped_events.sort_values(
        ['incident_id', 'event_position'],
        kind='stable',
    )
    documents = (
        mapped_events.groupby('incident_id', observed=True, sort=False)
        .agg(
            alarm_document=(
                'alarm_code',
                lambda codes: ' '.join(codes.astype('string')),
            ),
            alarm_token_count=('alarm_code', 'size'),
        )
        .reset_index()
    )
    documents['incident_id'] = documents['incident_id'].astype('string')
    documents['alarm_document'] = documents['alarm_document'].astype(
        'string'
    )
    documents['alarm_token_count'] = documents[
        'alarm_token_count'
    ].astype('int64')
    return documents[ALARM_DOCUMENT_COLUMNS]
