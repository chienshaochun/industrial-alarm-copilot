'''Write reproducible derived-episode artifacts.'''

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from industrial_alarm_copilot.data.artifacts import (
    calculate_file_sha256,
    write_metadata_json,
)
from industrial_alarm_copilot.incidents.pipeline import (
    IncidentAnalysis,
    build_incident_analysis,
)


@dataclass(frozen=True)
class IncidentArtifactPaths:
    '''Paths produced by the incident artifact writer.'''

    incidents_parquet: Path
    incident_events_parquet: Path
    baselines_json: Path


def _write_parquet(table: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(
        output_path,
        engine='pyarrow',
        compression='zstd',
        index=False,
    )
    return output_path


def build_incident_artifact_metadata(
    analysis: IncidentAnalysis,
    source_events_path: str | Path,
    settings: dict[str, Any],
    code_version: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    '''Build JSON-serializable provenance for incident artifacts.'''
    generated_at = generated_at or datetime.now(UTC)
    if generated_at.tzinfo is None:
        raise ValueError('generated_at must include a timezone')

    split_profile = {}
    for split_name, split_incidents in analysis.incidents.groupby(
        'split',
        observed=True,
        sort=False,
    ):
        split_profile[str(split_name)] = {
            'incident_count': int(len(split_incidents)),
            'upper_tail_count': int(split_incidents['is_upper_tail'].sum()),
            'upper_tail_share': float(
                split_incidents['is_upper_tail'].mean()
            ),
        }

    machine_baselines = json.loads(
        analysis.machine_baselines.to_json(orient='records')
    )
    source_events_path = Path(source_events_path)
    return {
        'artifact_schema_version': 1,
        'generated_at_utc': generated_at.astimezone(UTC).isoformat(),
        'code_version': code_version,
        'source_events': {
            'path': source_events_path.as_posix(),
            'sha256': calculate_file_sha256(source_events_path),
        },
        'settings': settings,
        'incidents': {
            'row_count': int(len(analysis.incidents)),
            'columns': {
                column: str(dtype)
                for column, dtype in analysis.incidents.dtypes.items()
            },
            'split_profile': split_profile,
        },
        'incident_events': {
            'row_count': int(len(analysis.incident_events)),
            'columns': {
                column: str(dtype)
                for column, dtype in analysis.incident_events.dtypes.items()
            },
        },
        'baselines': {
            'global': asdict(analysis.global_baseline),
            'machines': machine_baselines,
        },
    }


def write_incident_analysis_artifacts(
    analysis: IncidentAnalysis,
    source_events_path: str | Path,
    output_dir: str | Path,
    settings: dict[str, Any],
    code_version: str,
    generated_at: datetime | None = None,
) -> IncidentArtifactPaths:
    '''Write incident tables and their reproducibility metadata.'''
    output_dir = Path(output_dir)
    incidents_path = output_dir / 'incidents.parquet'
    incident_events_path = output_dir / 'incident_events.parquet'
    baselines_path = output_dir / 'incident_baselines.json'

    metadata = build_incident_artifact_metadata(
        analysis,
        source_events_path=source_events_path,
        settings=settings,
        code_version=code_version,
        generated_at=generated_at,
    )
    _write_parquet(analysis.incidents, incidents_path)
    _write_parquet(analysis.incident_events, incident_events_path)
    write_metadata_json(metadata, baselines_path)

    return IncidentArtifactPaths(
        incidents_parquet=incidents_path,
        incident_events_parquet=incident_events_path,
        baselines_json=baselines_path,
    )


def prepare_incident_artifacts(
    events_parquet_path: str | Path,
    output_dir: str | Path,
    pipeline_settings: dict[str, Any],
    code_version: str,
    generated_at: datetime | None = None,
) -> IncidentArtifactPaths:
    '''Build incident analysis from processed events and write artifacts.'''
    incident_settings = pipeline_settings['incidents']
    baseline_settings = pipeline_settings['baselines']
    selected_settings = {
        'gap_minutes': float(incident_settings['gap_minutes']),
        'baseline_quantile': float(baseline_settings['quantile']),
        'minimum_incident_count': int(
            baseline_settings['minimum_incident_count']
        ),
    }

    events = pd.read_parquet(events_parquet_path)
    analysis = build_incident_analysis(
        events,
        gap_minutes=selected_settings['gap_minutes'],
        baseline_quantile=selected_settings['baseline_quantile'],
        minimum_incident_count=selected_settings[
            'minimum_incident_count'
        ],
    )
    return write_incident_analysis_artifacts(
        analysis,
        source_events_path=events_parquet_path,
        output_dir=output_dir,
        settings=selected_settings,
        code_version=code_version,
        generated_at=generated_at,
    )
