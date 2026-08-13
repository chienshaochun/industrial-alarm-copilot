'''Retrieval artifact loading tests.'''

import pandas as pd

import industrial_alarm_copilot.retrieval.pipeline as pipeline
from industrial_alarm_copilot.retrieval.pipeline import (
    RetrievalExperimentInputs,
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


def test_run_selected_test_from_artifacts_passes_locked_settings(
    monkeypatch,
):
    inputs = RetrievalExperimentInputs(
        events=pd.DataFrame({'table': ['events']}),
        incidents=pd.DataFrame({'table': ['incidents']}),
        incident_events=pd.DataFrame({'table': ['incident_events']}),
        documents=pd.DataFrame({'table': ['documents']}),
    )
    parsed_settings = object()
    received = {}

    def fake_load(*paths):
        received['paths'] = paths
        return inputs

    def fake_evaluate(incidents, documents, events, settings):
        received['evaluation_args'] = (
            incidents,
            documents,
            events,
            settings,
        )
        return pd.DataFrame({'evaluation_split': ['test']})

    monkeypatch.setattr(
        pipeline,
        'load_retrieval_experiment_inputs',
        fake_load,
    )
    monkeypatch.setattr(
        pipeline,
        'parse_retrieval_settings',
        lambda settings: parsed_settings,
    )
    monkeypatch.setattr(
        pipeline,
        'run_selected_test_evaluation',
        fake_evaluate,
    )

    result = pipeline.run_selected_test_from_artifacts(
        'events.parquet',
        'incidents.parquet',
        'incident_events.parquet',
        {'retrieval': {'selected_feature_version': 'locked'}},
    )

    assert received['paths'] == (
        'events.parquet',
        'incidents.parquet',
        'incident_events.parquet',
    )
    evaluation_args = received['evaluation_args']
    assert evaluation_args[0] is inputs.incidents
    assert evaluation_args[1] is inputs.documents
    assert evaluation_args[2] is inputs.events
    assert evaluation_args[3] is parsed_settings
    assert result['evaluation_split'].tolist() == ['test']
