"""Raw CSV 到 processed artifacts 的整合測試。"""

import json
from datetime import UTC, datetime

import pandas as pd

from industrial_alarm_copilot.data.pipeline import prepare_data_artifacts


def test_prepare_data_artifacts_writes_parquet_and_metadata(tmp_path):
    raw_csv_path = tmp_path / "raw" / "alarms.csv"
    raw_csv_path.parent.mkdir(parents=True)
    raw_csv_path.write_text(
        "timestamp,alarm,serial\n"
        "2019-02-21 10:00:00.000,139,4\n"
        "2019-02-21 10:00:00.000,139,4\n"
        "2019-02-21 10:05:00.000,31,4\n"
        "2019-02-21 10:40:00.000,11,4\n"
        "2019-02-21 10:45:00.000,26,4\n",
        encoding="utf-8",
    )
    pipeline_settings = {
        "incidents": {
            "gap_minutes": 30,
            "sensitivity_gap_minutes": [15, 30, 60],
        },
        "split": {
            "strategy": "per_machine_unique_timestamp",
            "train_ratio": 0.50,
            "validation_ratio": 0.25,
            "test_ratio": 0.25,
        },
    }

    artifact_paths = prepare_data_artifacts(
        raw_csv_path=raw_csv_path,
        output_dir=tmp_path / "processed",
        pipeline_settings=pipeline_settings,
        code_version="abc1234",
        generated_at=datetime(2026, 8, 10, 12, 0, tzinfo=UTC),
    )

    processed_events = pd.read_parquet(artifact_paths.events_parquet)
    metadata = json.loads(
        artifact_paths.metadata_json.read_text(encoding="utf-8")
    )
    assert processed_events["duplicate_group_size"].tolist() == [
        2,
        2,
        1,
        1,
        1,
    ]
    assert processed_events["split"].tolist() == [
        "train",
        "train",
        "train",
        "validation",
        "test",
    ]
    assert metadata["code_version"] == "abc1234"
    assert metadata["events"]["row_count"] == 5
