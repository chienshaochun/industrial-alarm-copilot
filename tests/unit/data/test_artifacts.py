"""資料 artifact 寫入功能的單元測試。"""

import hashlib
import json
from datetime import UTC, datetime

import pandas as pd

from industrial_alarm_copilot.data.artifacts import (
    build_events_metadata,
    write_events_parquet,
    write_metadata_json,
)


def test_write_events_parquet_preserves_values_and_dtypes(tmp_path):
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2019-02-21 10:00:00", "2019-02-21 10:05:00"]
            ),
            "alarm_code": pd.Series(["139", "31"], dtype="string"),
            "machine_id": pd.Series(["4", "4"], dtype="string"),
            "source_row": [0, 1],
            "split": ["train", "train"],
        }
    )
    output_path = tmp_path / "processed" / "events.parquet"

    written_path = write_events_parquet(events, output_path)
    restored_events = pd.read_parquet(written_path, engine="pyarrow")

    assert written_path == output_path
    assert written_path.is_file()
    pd.testing.assert_frame_equal(restored_events, events)


def test_build_and_write_events_metadata(tmp_path):
    source_path = tmp_path / "alarms.csv"
    source_bytes = b"timestamp,alarm,serial\n"
    source_path.write_bytes(source_bytes)
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2019-01-01", "2019-01-02", "2019-01-03"]
            ),
            "alarm_code": pd.Series(["139", "31", "11"], dtype="string"),
            "machine_id": pd.Series(["4", "4", "4"], dtype="string"),
            "source_row": [0, 1, 2],
            "split": ["train", "validation", "test"],
        }
    )

    metadata = build_events_metadata(
        events,
        source_path=source_path,
        pipeline_settings={"split": [0.70, 0.15, 0.15]},
        code_version="abc1234",
        generated_at=datetime(2026, 8, 10, 12, 0, tzinfo=UTC),
    )
    output_path = tmp_path / "processed" / "events.metadata.json"
    written_path = write_metadata_json(metadata, output_path)
    restored_metadata = json.loads(written_path.read_text(encoding="utf-8"))

    assert restored_metadata["source"]["sha256"] == hashlib.sha256(
        source_bytes
    ).hexdigest()
    assert restored_metadata["generated_at_utc"] == "2026-08-10T12:00:00+00:00"
    assert restored_metadata["code_version"] == "abc1234"
    assert restored_metadata["events"]["row_count"] == 3
    assert restored_metadata["events"]["split_boundaries"]["test"] == {
        "event_count": 1,
        "start_time": "2019-01-03T00:00:00",
        "end_time": "2019-01-03T00:00:00",
    }
