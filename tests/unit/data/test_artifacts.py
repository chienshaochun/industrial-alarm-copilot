"""資料 artifact 寫入功能的單元測試。"""

import pandas as pd

from industrial_alarm_copilot.data.artifacts import write_events_parquet


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
