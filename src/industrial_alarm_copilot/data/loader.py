"""讀取 ALPI 原始警報資料。"""

from pathlib import Path

import pandas as pd

from industrial_alarm_copilot.data.schema import (
    DataContractError,
    validate_raw_data,
)


def load_alarm_events(csv_path: str | Path) -> pd.DataFrame:
    """讀取 ALPI CSV，並轉換為專案使用的事件欄位。"""
    events = pd.read_csv(
        csv_path,
        dtype={
            "timestamp": "string",
            "alarm": "string",
            "serial": "string",
        },
        encoding="utf-8",
    )
    validate_raw_data(events)

    try:
        events["timestamp"] = pd.to_datetime(
            events["timestamp"],
            format="%Y-%m-%d %H:%M:%S.%f",
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise DataContractError(
            "timestamp 包含無法解析的日期時間"
        ) from error

    events = events.rename(
        columns={"alarm": "alarm_code", "serial": "machine_id"}
    )
    events["source_row"] = range(len(events))

    return events[["timestamp", "alarm_code", "machine_id", "source_row"]]
