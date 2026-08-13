'''Alarm document construction unit tests.'''

import pandas as pd

from industrial_alarm_copilot.retrieval.documents import (
    build_alarm_documents,
)


def test_build_alarm_documents_restores_order_and_preserves_duplicates():
    events = pd.DataFrame(
        {
            'source_row': [12, 10, 13, 11],
            'alarm_code': pd.Series(
                ['1', '98', '26', '98'],
                dtype='string',
            ),
        }
    )
    incident_events = pd.DataFrame(
        {
            'incident_id': pd.Series(
                ['inc_a', 'inc_b', 'inc_a', 'inc_a'],
                dtype='string',
            ),
            'source_row': [11, 13, 10, 12],
            'event_position': [1, 0, 0, 2],
        }
    )

    documents = build_alarm_documents(events, incident_events)

    assert documents['incident_id'].tolist() == ['inc_a', 'inc_b']
    assert documents['alarm_document'].tolist() == ['98 98 1', '26']
    assert documents['alarm_token_count'].tolist() == [3, 1]
