"""ALPI CSV 載入器的單元測試。"""

import pandas as pd
import pytest

from industrial_alarm_copilot.data.loader import load_alarm_events
from industrial_alarm_copilot.data.schema import DataContractError


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


def test_load_alarm_events_rejects_missing_required_column(tmp_path):
    csv_path = tmp_path / "missing_serial.csv"
    csv_path.write_text(
        "timestamp,alarm\n"
        "2019-02-21 19:57:57.532,139\n",
        encoding="utf-8",
    )

    with pytest.raises(DataContractError, match="serial"):
        load_alarm_events(csv_path)


def test_load_alarm_events_rejects_missing_value(tmp_path):
    csv_path = tmp_path / "missing_alarm_value.csv"
    csv_path.write_text(
        "timestamp,alarm,serial\n"
        "2019-02-21 19:57:57.532,,4\n",
        encoding="utf-8",
    )

    with pytest.raises(DataContractError, match="alarm"):
        load_alarm_events(csv_path)


def test_load_alarm_events_rejects_invalid_timestamp(tmp_path):
    csv_path = tmp_path / "invalid_timestamp.csv"
    csv_path.write_text(
        "timestamp,alarm,serial\n"
        "not-a-timestamp,139,4\n",
        encoding="utf-8",
    )

    with pytest.raises(DataContractError, match="timestamp"):
        load_alarm_events(csv_path)


@pytest.mark.parametrize(
    ("alarm", "serial", "invalid_column"),
    [
        ("not-an-alarm", "4", "alarm"),
        ("139", "not-a-machine", "serial"),
    ],
)
def test_load_alarm_events_rejects_invalid_identifier(
    tmp_path, alarm, serial, invalid_column
):
    csv_path = tmp_path / f"invalid_{invalid_column}.csv"
    csv_path.write_text(
        "timestamp,alarm,serial\n"
        f"2019-02-21 19:57:57.532,{alarm},{serial}\n",
        encoding="utf-8",
    )

    with pytest.raises(DataContractError, match=invalid_column):
        load_alarm_events(csv_path)
