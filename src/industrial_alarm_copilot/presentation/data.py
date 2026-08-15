'''Small, deterministic view models for the Streamlit pages.'''

from dataclasses import dataclass
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


def required_artifact_paths(project_root: Path) -> dict[str, Path]:
    processed = project_root / 'data' / 'processed'
    return {
        'events': processed / 'events.parquet',
        'incidents': processed / 'incidents.parquet',
        'incident_events': processed / 'incident_events.parquet',
        'forecast_model': processed / 'forecast_model.json',
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
