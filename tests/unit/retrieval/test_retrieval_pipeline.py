'''Retrieval artifact loading tests.'''

import pandas as pd

from industrial_alarm_copilot.retrieval.pipeline import (
    load_retrieval_experiment_inputs,
)


def test_load_retrieval_experiment_inputs_builds_aligned_documents(
    monkeypatch,
):
    tables = {
        'events.parquet': pd.DataFrame(
            {
                'source_row': [0, 1, 2],
                'alarm_code': ['98', '11', '26'],
            }
        ),
        'incidents.parquet': pd.DataFrame(
            {'incident_id': ['inc_a', 'inc_b']}
        ),
        'incident_events.parquet': pd.DataFrame(
            {
                'incident_id': ['inc_a', 'inc_a', 'inc_b'],
                'source_row': [0, 1, 2],
                'event_position': [0, 1, 0],
            }
        ),
    }

    def fake_read_parquet(path):
        return tables[str(path)].copy()

    monkeypatch.setattr(pd, 'read_parquet', fake_read_parquet)

    inputs = load_retrieval_experiment_inputs(
        'events.parquet',
        'incidents.parquet',
        'incident_events.parquet',
    )

    assert inputs.documents['incident_id'].tolist() == ['inc_a', 'inc_b']
    assert inputs.documents['alarm_document'].tolist() == ['98 11', '26']
