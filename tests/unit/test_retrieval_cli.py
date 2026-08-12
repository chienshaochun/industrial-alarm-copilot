'''Validate-retrieval CLI unit tests.'''

import pandas as pd

import industrial_alarm_copilot.__main__ as cli


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

    monkeypatch.setattr(cli, 'load_pipeline_settings', lambda path: settings)

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
    output = capsys.readouterr().out
    assert 'alarm_tfidf_v1' in output
    assert 'recall_efficiency_at_5' in output
    assert '0.5' in output
    assert 'smoke-run validation query limit: 10' in output
