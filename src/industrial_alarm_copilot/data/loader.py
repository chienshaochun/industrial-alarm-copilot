"""讀取 ALPI 原始警報資料。"""

from pathlib import Path

import pandas as pd


def load_alarm_events(csv_path: str | Path) -> pd.DataFrame:
    """讀取 ALPI CSV，並轉換為專案使用的事件欄位。"""
    events = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"],
        dtype={"alarm": "string", "serial": "string"},
    )

    events = events.rename(
        columns={"alarm": "alarm_code", "serial": "machine_id"}
    )
    events["source_row"] = range(len(events))

    return events[["timestamp", "alarm_code", "machine_id", "source_row"]]
