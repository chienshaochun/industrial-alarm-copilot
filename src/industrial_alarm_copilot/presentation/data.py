'''Small, deterministic view models for the Streamlit pages.'''

from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class OverviewData:
    '''Pre-aggregated data used by the overview page.'''

    event_count: int
    machine_count: int
    alarm_code_count: int
    incident_count: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    monthly_events: pd.DataFrame
    top_alarms: pd.DataFrame
    machine_events: pd.DataFrame
    split_incidents: pd.DataFrame


@dataclass(frozen=True)
class EvaluationData:
    '''Locked Stage 5 and Stage 6 test results for portfolio reporting.'''

    retrieval: dict[str, object]
    forecasting: dict[str, object]
    support_groups: pd.DataFrame


ARTIFACT_DIRECTORY_ENV = 'INDUSTRIAL_ALARM_ARTIFACT_DIR'
CORE_ARTIFACT_FILENAMES = (
    'events.parquet',
    'incidents.parquet',
    'incident_events.parquet',
    'forecast_model.json',
)


def resolve_artifact_directory(project_root: Path) -> Path:
    '''Prefer local processed artifacts, then the deployment snapshot.'''
    override = os.environ.get(ARTIFACT_DIRECTORY_ENV)
    if override:
        override_path = Path(override)
        return (
            override_path
            if override_path.is_absolute()
            else project_root / override_path
        )
    processed = project_root / 'data' / 'processed'
    if all((processed / filename).is_file() for filename in CORE_ARTIFACT_FILENAMES):
        return processed
    return project_root / 'data' / 'deployment'


def required_artifact_paths(project_root: Path) -> dict[str, Path]:
    artifact_directory = resolve_artifact_directory(project_root)
    return {
        'events': artifact_directory / 'events.parquet',
        'incidents': artifact_directory / 'incidents.parquet',
        'incident_events': artifact_directory / 'incident_events.parquet',
        'forecast_model': artifact_directory / 'forecast_model.json',
    }


def missing_artifacts(project_root: Path) -> tuple[str, ...]:
    return tuple(
        name
        for name, path in required_artifact_paths(project_root).items()
        if not path.is_file()
    )


def build_overview_data(
    events: pd.DataFrame,
    incidents: pd.DataFrame,
) -> OverviewData:
    '''Aggregate UI profiles without changing Stage 3 or 4 artifacts.'''
    monthly = (
        events.assign(
            month_start=events['timestamp'].dt.to_period('M').dt.to_timestamp()
        )
        .groupby('month_start', observed=True)
        .size()
        .rename('event_count')
        .reset_index()
    )
    top_alarms = (
        events.groupby('alarm_code', observed=True)
        .size()
        .rename('event_count')
        .sort_values(ascending=False)
        .head(12)
        .reset_index()
    )
    machine_events = (
        events.groupby('machine_id', observed=True)
        .size()
        .rename('event_count')
        .sort_values(ascending=False)
        .reset_index()
    )
    split_incidents = (
        incidents.groupby('split', observed=True)
        .size()
        .rename('incident_count')
        .reindex(['train', 'validation', 'test'])
        .fillna(0)
        .astype({'incident_count': int})
        .reset_index()
    )
    return OverviewData(
        event_count=len(events),
        machine_count=events['machine_id'].nunique(),
        alarm_code_count=events['alarm_code'].nunique(),
        incident_count=len(incidents),
        start_time=events['timestamp'].min(),
        end_time=events['timestamp'].max(),
        monthly_events=monthly,
        top_alarms=top_alarms,
        machine_events=machine_events,
        split_incidents=split_incidents,
    )


def build_evaluation_data(
    retrieval_results: pd.DataFrame,
    forecast_results: pd.DataFrame,
    support_groups: pd.DataFrame,
) -> EvaluationData:
    '''Validate the single locked test rows before presentation.'''
    if len(retrieval_results) != 1 or len(forecast_results) != 1:
        raise ValueError('evaluation artifacts must contain one locked row')
    retrieval = retrieval_results.iloc[0].to_dict()
    forecasting = forecast_results.iloc[0].to_dict()
    if retrieval['evaluation_split'] != 'test':
        raise ValueError('retrieval evaluation must use test split')
    if forecasting['split'] != 'test':
        raise ValueError('forecast evaluation must use test split')
    expected_groups = {'rare', 'medium', 'common'}
    if set(support_groups['support_group']) != expected_groups:
        raise ValueError('forecast support groups are incomplete')
    return EvaluationData(
        retrieval=retrieval,
        forecasting=forecasting,
        support_groups=support_groups.copy(),
    )
