'''Forecast model and final test artifacts with provenance.'''

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from industrial_alarm_copilot.data.artifacts import (
    calculate_file_sha256,
    write_metadata_json,
)
from industrial_alarm_copilot.forecasting.baselines import (
    TransitionFrequencyBaseline,
)
from industrial_alarm_copilot.forecasting.pipeline import SelectedForecastRun


@dataclass(frozen=True)
class ForecastArtifactPaths:
    '''Files produced by the locked forecast test run.'''

    test_results_csv: Path
    support_groups_csv: Path
    scope_profile_csv: Path
    model_json: Path
    metadata_json: Path


def serialize_transition_baseline(
    baseline: TransitionFrequencyBaseline,
) -> dict[str, Any]:
    '''Convert the selected deterministic model into portable JSON data.'''
    machine = baseline.machine_baseline
    return {
        'artifact_schema_version': 1,
        'model_version': baseline.model_version,
        'alarm_codes': list(machine.alarm_codes),
        'minimum_machine_train_samples': machine.minimum_machine_train_samples,
        'minimum_transition_train_samples': baseline.minimum_transition_train_samples,
        'global_scores': machine.global_scores.tolist(),
        'machine_scores': [
            {'machine_id': machine_id, 'scores': scores.tolist()}
            for machine_id, scores in machine.machine_scores
        ],
        'machine_train_sample_counts': [
            {'machine_id': machine_id, 'sample_count': sample_count}
            for machine_id, sample_count in machine.machine_train_sample_counts
        ],
        'transition_scores': [
            {
                'machine_id': state[0],
                'last_alarm_code': state[1],
                'scores': scores.tolist(),
            }
            for state, scores in baseline.transition_scores
        ],
        'transition_train_sample_counts': [
            {
                'machine_id': state[0],
                'last_alarm_code': state[1],
                'sample_count': sample_count,
            }
            for state, sample_count in baseline.transition_train_sample_counts
        ],
    }


def write_forecast_test_artifacts(
    run: SelectedForecastRun,
    source_paths: dict[str, str | Path],
    forecast_settings: dict[str, Any],
    code_version: str,
    output_dir: str | Path,
    generated_at: datetime | None = None,
) -> ForecastArtifactPaths:
    '''Write one final test result, portable model, and provenance metadata.'''
    split_metrics = run.evaluation.split_metrics
    if len(split_metrics) != 1 or set(split_metrics['split']) != {'test'}:
        raise ValueError('forecast artifacts require exactly one test metric row')
    generated_at = generated_at or datetime.now(UTC)
    if generated_at.tzinfo is None:
        raise ValueError('generated_at must include a timezone')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_results_path = output_dir / 'forecast_test_results.csv'
    support_groups_path = output_dir / 'forecast_test_support_groups.csv'
    scope_profile_path = output_dir / 'forecast_test_scope_profile.csv'
    model_path = output_dir / 'forecast_model.json'
    metadata_path = output_dir / 'forecast_test.metadata.json'
    split_metrics.to_csv(test_results_path, index=False, lineterminator='\n')
    run.support_group_metrics.to_csv(
        support_groups_path, index=False, lineterminator='\n'
    )
    run.scope_profile.to_csv(scope_profile_path, index=False, lineterminator='\n')
    model_payload = serialize_transition_baseline(run.baseline)
    write_metadata_json(model_payload, model_path)

    sources = {
        name: {
            'path': Path(path).as_posix(),
            'sha256': calculate_file_sha256(path),
        }
        for name, path in source_paths.items()
    }
    result = split_metrics.iloc[0]
    metadata = {
        'artifact_schema_version': 1,
        'generated_at_utc': generated_at.astimezone(UTC).isoformat(),
        'code_version': code_version,
        'evaluation_split': 'test',
        'is_final_test': True,
        'sources': sources,
        'forecast_settings': forecast_settings,
        'model': {
            'model_version': run.baseline.model_version,
            'alarm_code_count': len(run.baseline.machine_baseline.alarm_codes),
            'transition_state_count': len(run.baseline.transition_scores),
        },
        'results': {
            'episode_count': int(result['episode_count']),
            'complete_outcome_count': int(result['complete_outcome_count']),
            'columns': {
                column: str(dtype) for column, dtype in split_metrics.dtypes.items()
            },
        },
    }
    write_metadata_json(metadata, metadata_path)
    return ForecastArtifactPaths(
        test_results_csv=test_results_path,
        support_groups_csv=support_groups_path,
        scope_profile_csv=scope_profile_path,
        model_json=model_path,
        metadata_json=metadata_path,
    )
