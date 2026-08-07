"""ALPI CSV 載入器的單元測試。"""

import pandas as pd

from industrial_alarm_copilot.data.loader import load_alarm_events


def test_load_alarm_events_normalizes_columns_and_preserves_rows(tmp_path):
    csv_path = tmp_path / "alarms.csv"
    csv_path.write_text(
        "timestamp,alarm,serial\n"
        "2019-02-21 19:57:57.532,139,4\n"
        "2019-02-21 19:58:28.293,31,4\n",
        encoding="utf-8",
    )

    events = load_alarm_events(csv_path)

    assert list(events.columns) == [
        "timestamp",
        "alarm_code",
        "machine_id",
        "source_row",
    ]
    assert events["alarm_code"].tolist() == ["139", "31"]
    assert events["machine_id"].tolist() == ["4", "4"]
    assert events["source_row"].tolist() == [0, 1]
    assert events["timestamp"].tolist() == [
        pd.Timestamp("2019-02-21 19:57:57.532"),
        pd.Timestamp("2019-02-21 19:58:28.293"),
    ]
