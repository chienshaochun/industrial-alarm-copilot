'''Incident artifact writer unit tests.'''

import json
from datetime import UTC, datetime

import pandas as pd

from industrial_alarm_copilot.incidents.artifacts import (
    prepare_incident_artifacts,
    write_incident_analysis_artifacts,
)
from industrial_alarm_copilot.incidents.baselines import (
    GlobalIncidentBaseline,
)
from industrial_alarm_copilot.incidents.pipeline import IncidentAnalysis


def test_write_incident_analysis_artifacts_preserves_tables_and_provenance(
    tmp_path,
):
    incidents = pd.DataFrame(
        {
            'incident_id': pd.Series(['inc_a', 'inc_b'], dtype='string'),
            'split': ['train', 'validation'],
            'is_upper_tail': [False, True],
        }
    )
    incident_events = pd.DataFrame(
        {
            'incident_id': pd.Series(
                ['inc_a', 'inc_a', 'inc_b'],
                dtype='string',
            ),
            'source_row': [0, 1, 2],
            'event_position': [0, 1, 0],
        }
    )
    global_baseline = GlobalIncidentBaseline(
        quantile=0.95,
        fit_split='train',
        incident_count=1,
        event_count_threshold=2.0,
        duration_seconds_threshold=300.0,
        distinct_alarm_count_threshold=1.0,
    )
    machine_baselines = pd.DataFrame(
        {
            'machine_id': ['4'],
            'incident_count': [1],
            'has_sufficient_support': [False],
        }
    )
    analysis = IncidentAnalysis(
        incidents=incidents,
        incident_events=incident_events,
        global_baseline=global_baseline,
        machine_baselines=machine_baselines,
    )
    source_events_path = tmp_path / 'events.parquet'
    source_events_path.write_bytes(b'processed-events')

    paths = write_incident_analysis_artifacts(
        analysis,
        source_events_path=source_events_path,
        output_dir=tmp_path / 'processed',
        settings={
            'gap_minutes': 30,
            'baseline_quantile': 0.95,
            'minimum_incident_count': 200,
        },
        code_version='abc1234',
        generated_at=datetime(2026, 8, 12, 10, 0, tzinfo=UTC),
    )

    written_incidents = pd.read_parquet(paths.incidents_parquet)
    written_mapping = pd.read_parquet(paths.incident_events_parquet)
    metadata = json.loads(paths.baselines_json.read_text(encoding='utf-8'))

    assert written_incidents['incident_id'].tolist() == ['inc_a', 'inc_b']
    assert written_mapping['source_row'].tolist() == [0, 1, 2]
    assert metadata['generated_at_utc'] == '2026-08-12T10:00:00+00:00'
    assert metadata['code_version'] == 'abc1234'
    assert len(metadata['source_events']['sha256']) == 64
    assert metadata['incidents']['row_count'] == 2
    assert metadata['incident_events']['row_count'] == 3
    assert metadata['incidents']['split_profile']['validation'][
        'upper_tail_share'
    ] == 1.0
    assert metadata['baselines']['global']['quantile'] == 0.95
    assert metadata['baselines']['machines'][0]['machine_id'] == '4'


def test_prepare_incident_artifacts_reads_events_and_settings(tmp_path):
    events = pd.DataFrame(
        {
            'timestamp': pd.to_datetime(
                [
                    '2019-01-01 10:00:00',
                    '2019-01-01 10:05:00',
                    '2019-01-01 10:40:00',
                    '2019-01-01 10:41:00',
                ]
            ),
            'alarm_code': ['11', '11', '26', '98'],
            'machine_id': ['4', '4', '4', '4'],
            'source_row': [0, 1, 2, 3],
            'split': ['train', 'train', 'train', 'validation'],
            'gap_seconds': [float('nan'), 300.0, 2100.0, 60.0],
            'is_exact_duplicate': [False] * 4,
        }
    )
    events_path = tmp_path / 'events.parquet'
    events.to_parquet(events_path, index=False)

    paths = prepare_incident_artifacts(
        events_parquet_path=events_path,
        output_dir=tmp_path / 'processed',
        pipeline_settings={
            'incidents': {'gap_minutes': 30},
            'baselines': {
                'quantile': 0.95,
                'minimum_incident_count': 2,
            },
        },
        code_version='abc1234',
        generated_at=datetime(2026, 8, 12, 10, 0, tzinfo=UTC),
    )

    incidents = pd.read_parquet(paths.incidents_parquet)
    metadata = json.loads(paths.baselines_json.read_text(encoding='utf-8'))

    assert len(incidents) == 3
    assert incidents['baseline_scope'].eq('machine').all()
    assert metadata['settings'] == {
        'gap_minutes': 30.0,
        'baseline_quantile': 0.95,
        'minimum_incident_count': 2,
    }
