'''Reproducible artifacts for retrieval validation experiments.'''

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from industrial_alarm_copilot.data.artifacts import (
    calculate_file_sha256,
    write_metadata_json,
)


@dataclass(frozen=True)
class RetrievalValidationArtifactPaths:
    '''Files produced by one retrieval validation run.'''

    results_csv: Path
    metadata_json: Path


def build_retrieval_validation_metadata(
    experiment_results: pd.DataFrame,
    source_paths: dict[str, str | Path],
    retrieval_settings: dict[str, Any],
    code_version: str,
    query_limit: int | None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    '''Describe experiment inputs, scope, settings, and result schema.'''
    if experiment_results.empty:
        raise ValueError('experiment_results cannot be empty')
    if set(experiment_results['selection_split']) != {'validation'}:
        raise ValueError('selection_split must contain only validation')

    generated_at = generated_at or datetime.now(UTC)
    if generated_at.tzinfo is None:
        raise ValueError('generated_at must include a timezone')
    sources = {
        source_name: {
            'path': Path(source_path).as_posix(),
            'sha256': calculate_file_sha256(source_path),
        }
        for source_name, source_path in source_paths.items()
    }
    return {
        'artifact_schema_version': 1,
        'generated_at_utc': generated_at.astimezone(UTC).isoformat(),
        'code_version': code_version,
        'selection_split': 'validation',
        'query_limit': query_limit,
        'is_complete_validation': query_limit is None,
        'sources': sources,
        'retrieval_settings': retrieval_settings,
        'results': {
            'row_count': int(len(experiment_results)),
            'query_count': int(experiment_results['query_count'].max()),
            'feature_versions': sorted(
                experiment_results['feature_version'].unique().tolist()
            ),
            'future_horizon_hours': sorted(
                float(value)
                for value in experiment_results[
                    'future_horizon_hours'
                ].unique()
            ),
            'relevance_thresholds': sorted(
                float(value)
                for value in experiment_results[
                    'relevance_threshold'
                ].unique()
            ),
            'columns': {
                column: str(dtype)
                for column, dtype in experiment_results.dtypes.items()
            },
        },
    }


def write_retrieval_validation_artifacts(
    experiment_results: pd.DataFrame,
    source_paths: dict[str, str | Path],
    retrieval_settings: dict[str, Any],
    code_version: str,
    output_dir: str | Path,
    query_limit: int | None = None,
    generated_at: datetime | None = None,
) -> RetrievalValidationArtifactPaths:
    '''Write full-grid CSV results and provenance metadata.'''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'retrieval_validation_results.csv'
    metadata_path = output_dir / 'retrieval_validation.metadata.json'

    metadata = build_retrieval_validation_metadata(
        experiment_results,
        source_paths=source_paths,
        retrieval_settings=retrieval_settings,
        code_version=code_version,
        query_limit=query_limit,
        generated_at=generated_at,
    )
    experiment_results.to_csv(
        results_path,
        index=False,
        encoding='utf-8',
        lineterminator='\n',
    )
    write_metadata_json(metadata, metadata_path)
    return RetrievalValidationArtifactPaths(
        results_csv=results_path,
        metadata_json=metadata_path,
    )
