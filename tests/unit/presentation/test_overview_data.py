'''Overview view-model tests.'''

from pathlib import Path

import pandas as pd

from industrial_alarm_copilot.presentation.data import (
    ARTIFACT_DIRECTORY_ENV,
    CORE_ARTIFACT_FILENAMES,
    build_overview_data,
    resolve_artifact_directory,
)


def test_build_overview_data_aggregates_without_mutating_artifacts():
    events = pd.DataFrame(
        {
            'timestamp': pd.to_datetime(
                ['2020-01-01', '2020-01-02', '2020-02-01']
            ),
            'machine_id': ['1', '1', '2'],
            'alarm_code': ['11', '11', '98'],
        }
    )
    incidents = pd.DataFrame(
        {'split': ['train', 'validation', 'test', 'test']}
    )

    overview = build_overview_data(events, incidents)

    assert overview.event_count == 3
    assert overview.machine_count == 2
    assert overview.alarm_code_count == 2
    assert overview.incident_count == 4
    assert overview.monthly_events['event_count'].tolist() == [2, 1]
    assert overview.top_alarms.iloc[0].to_dict() == {
        'alarm_code': '11',
        'event_count': 2,
    }
    assert overview.split_incidents['incident_count'].tolist() == [1, 1, 2]


def _touch_core_artifacts(directory: Path) -> None:
    directory.mkdir(parents=True)
    for filename in CORE_ARTIFACT_FILENAMES:
        (directory / filename).touch()


def test_resolve_artifact_directory_prefers_local_processed(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(ARTIFACT_DIRECTORY_ENV, raising=False)
    processed = tmp_path / 'data' / 'processed'
    _touch_core_artifacts(processed)
    _touch_core_artifacts(tmp_path / 'data' / 'deployment')

    assert resolve_artifact_directory(tmp_path) == processed


def test_resolve_artifact_directory_falls_back_to_deployment(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(ARTIFACT_DIRECTORY_ENV, raising=False)
    deployment = tmp_path / 'data' / 'deployment'
    _touch_core_artifacts(deployment)

    assert resolve_artifact_directory(tmp_path) == deployment


def test_resolve_artifact_directory_honors_relative_override(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(ARTIFACT_DIRECTORY_ENV, 'fixtures/cloud')

    assert resolve_artifact_directory(tmp_path) == (
        tmp_path / 'fixtures' / 'cloud'
    )
