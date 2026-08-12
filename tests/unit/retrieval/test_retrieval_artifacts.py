'''Retrieval validation artifact tests.'''

import json
from datetime import UTC, datetime

import pandas as pd

from industrial_alarm_copilot.retrieval.artifacts import (
    write_retrieval_validation_artifacts,
)


def test_write_retrieval_validation_artifacts_records_smoke_run_scope(
    tmp_path,
):
    source_paths = {}
    for source_name in ('events', 'incidents', 'incident_events'):
        source_path = tmp_path / f'{source_name}.parquet'
        source_path.write_bytes(source_name.encode('utf-8'))
        source_paths[source_name] = source_path
    results = pd.DataFrame(
        {
            'selection_split': ['validation', 'validation'],
            'feature_version': ['alarm_tfidf_v1', 'alarm_plus_shape_v1'],
            'future_horizon_hours': [6.0, 6.0],
            'relevance_threshold': [0.3, 0.3],
            'query_count': [100, 100],
            'mean_precision_at_k': [0.2, 0.1],
        }
    )

    paths = write_retrieval_validation_artifacts(
        results,
        source_paths=source_paths,
        retrieval_settings={'top_k': 5},
        code_version='abc1234',
        output_dir=tmp_path / 'output',
        query_limit=100,
        generated_at=datetime(2026, 8, 12, tzinfo=UTC),
    )

    written_results = pd.read_csv(paths.results_csv)
    metadata = json.loads(paths.metadata_json.read_text(encoding='utf-8'))
    pd.testing.assert_frame_equal(written_results, results)
    assert metadata['selection_split'] == 'validation'
    assert metadata['query_limit'] == 100
    assert metadata['is_complete_validation'] is False
    assert metadata['results']['row_count'] == 2
    assert metadata['results']['query_count'] == 100
    assert metadata['results']['feature_versions'] == [
        'alarm_plus_shape_v1',
        'alarm_tfidf_v1',
    ]
    assert set(metadata['sources']) == {
        'events',
        'incidents',
        'incident_events',
    }
