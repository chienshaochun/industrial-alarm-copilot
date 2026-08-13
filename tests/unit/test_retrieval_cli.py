'''Validate-retrieval CLI unit tests.'''

import pandas as pd

import industrial_alarm_copilot.__main__ as cli
from industrial_alarm_copilot.retrieval.artifacts import (
    RetrievalTestArtifactPaths,
    RetrievalValidationArtifactPaths,
)


def test_validate_retrieval_command_passes_artifacts_and_query_limit(
    monkeypatch,
    tmp_path,
    capsys,
):
    events_path = tmp_path / 'events.parquet'
    incidents_path = tmp_path / 'incidents.parquet'
    incident_events_path = tmp_path / 'incident_events.parquet'
    config_path = tmp_path / 'default.toml'
    settings = {'retrieval': {'top_k': 5}}
    received = {}
    received_artifact = {}

    monkeypatch.setattr(cli, 'load_pipeline_settings', lambda path: settings)
    monkeypatch.setattr(cli, 'get_git_commit', lambda path: 'abc1234')

    def fake_run_validation_from_artifacts(**kwargs):
        received.update(kwargs)
        return pd.DataFrame(
            {
                'selection_split': ['validation'],
                'feature_version': ['alarm_tfidf_v1'],
                'future_horizon_hours': [6.0],
                'relevance_threshold': [0.3],
                'evaluation_coverage': [0.8],
                'relevant_candidate_query_share': [0.7],
                'mean_relevant_candidate_share': [0.2],
                'mean_hit_at_k': [0.6],
                'mean_precision_at_k': [0.3],
                'mean_expected_random_precision_at_k': [0.2],
                'mean_precision_lift_at_k': [1.5],
                'mean_recall_at_k': [0.01],
                'mean_maximum_recall_at_k': [0.02],
                'mean_recall_efficiency_at_k': [0.5],
                'mean_reciprocal_rank': [0.4],
                'mean_ndcg_at_k': [0.35],
            }
        )

    monkeypatch.setattr(
        cli,
        'run_validation_from_artifacts',
        fake_run_validation_from_artifacts,
    )

    def fake_write_retrieval_validation_artifacts(
        experiment_results,
        **kwargs,
    ):
        received_artifact['experiment_results'] = experiment_results
        received_artifact.update(kwargs)
        return RetrievalValidationArtifactPaths(
            results_csv=tmp_path / 'retrieval_validation_results.csv',
            metadata_json=tmp_path / 'retrieval_validation.metadata.json',
        )

    monkeypatch.setattr(
        cli,
        'write_retrieval_validation_artifacts',
        fake_write_retrieval_validation_artifacts,
    )

    exit_code = cli.main(
        [
            'validate-retrieval',
            '--events-parquet',
            str(events_path),
            '--incidents-parquet',
            str(incidents_path),
            '--incident-events-parquet',
            str(incident_events_path),
            '--config',
            str(config_path),
            '--max-validation-queries',
            '10',
            '--diagnostics-only',
            '--output-dir',
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert received == {
        'events_parquet_path': events_path,
        'incidents_parquet_path': incidents_path,
        'incident_events_parquet_path': incident_events_path,
        'pipeline_settings': settings,
        'max_validation_queries': 10,
    }
    assert received_artifact['source_paths'] == {
        'events': events_path,
        'incidents': incidents_path,
        'incident_events': incident_events_path,
    }
    assert received_artifact['retrieval_settings'] == settings['retrieval']
    assert received_artifact['code_version'] == 'abc1234'
    assert received_artifact['output_dir'] == tmp_path
    assert received_artifact['query_limit'] == 10
    output = capsys.readouterr().out
    assert 'alarm_tfidf_v1' in output
    assert 'recall_efficiency_at_5' in output
    assert '0.5' in output
    assert 'smoke-run validation query limit: 10' in output
    assert 'retrieval_validation_results.csv' in output
    assert 'retrieval_validation.metadata.json' in output
    assert 'abc1234' in output


def test_test_retrieval_command_uses_locked_settings_and_writes_artifact(
    monkeypatch,
    tmp_path,
    capsys,
):
    events_path = tmp_path / 'events.parquet'
    incidents_path = tmp_path / 'incidents.parquet'
    incident_events_path = tmp_path / 'incident_events.parquet'
    config_path = tmp_path / 'default.toml'
    settings = {'retrieval': {'top_k': 5}}
    received = {}
    received_artifact = {}

    monkeypatch.setattr(cli, 'load_pipeline_settings', lambda path: settings)
    monkeypatch.setattr(cli, 'get_git_commit', lambda path: 'locked123')

    def fake_run_selected_test_from_artifacts(**kwargs):
        received.update(kwargs)
        return pd.DataFrame(
            {
                'evaluation_split': ['test'],
                'feature_version': ['alarm_plus_shape_v1'],
                'future_horizon_hours': [6.0],
                'relevance_threshold': [0.3],
                'evaluation_coverage': [0.8],
                'relevant_candidate_query_share': [0.7],
                'mean_relevant_candidate_share': [0.2],
                'mean_hit_at_k': [0.6],
                'mean_precision_at_k': [0.3],
                'mean_expected_random_precision_at_k': [0.2],
                'mean_precision_lift_at_k': [1.5],
                'mean_recall_at_k': [0.01],
                'mean_maximum_recall_at_k': [0.02],
                'mean_recall_efficiency_at_k': [0.5],
                'mean_reciprocal_rank': [0.4],
                'mean_ndcg_at_k': [0.35],
            }
        )

    monkeypatch.setattr(
        cli,
        'run_selected_test_from_artifacts',
        fake_run_selected_test_from_artifacts,
    )

    def fake_write_retrieval_test_artifacts(test_results, **kwargs):
        received_artifact['test_results'] = test_results
        received_artifact.update(kwargs)
        return RetrievalTestArtifactPaths(
            results_csv=tmp_path / 'retrieval_test_results.csv',
            metadata_json=tmp_path / 'retrieval_test.metadata.json',
        )

    monkeypatch.setattr(
        cli,
        'write_retrieval_test_artifacts',
        fake_write_retrieval_test_artifacts,
    )

    exit_code = cli.main(
        [
            'test-retrieval',
            '--events-parquet',
            str(events_path),
            '--incidents-parquet',
            str(incidents_path),
            '--incident-events-parquet',
            str(incident_events_path),
            '--config',
            str(config_path),
            '--output-dir',
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert received == {
        'events_parquet_path': events_path,
        'incidents_parquet_path': incidents_path,
        'incident_events_parquet_path': incident_events_path,
        'pipeline_settings': settings,
    }
    assert received_artifact['source_paths'] == {
        'events': events_path,
        'incidents': incidents_path,
        'incident_events': incident_events_path,
    }
    assert received_artifact['retrieval_settings'] == settings['retrieval']
    assert received_artifact['code_version'] == 'locked123'
    assert received_artifact['output_dir'] == tmp_path
    output = capsys.readouterr().out
    assert 'alarm_plus_shape_v1' in output
    assert 'retrieval_test_results.csv' in output
    assert 'retrieval_test.metadata.json' in output
    assert 'locked123' in output
