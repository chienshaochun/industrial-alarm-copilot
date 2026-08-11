"""ALPI 原始資料的 schema 驗證規則。"""

from collections.abc import Iterable

import pandas as pd


REQUIRED_RAW_COLUMNS = frozenset({"timestamp", "alarm", "serial"})


class DataContractError(ValueError):
    """資料不符合 ALPI 資料契約。"""


def validate_raw_columns(columns: Iterable[str]) -> None:
    """驗證原始資料是否包含必要欄位。"""
    actual_columns = set(columns)
    missing_columns = REQUIRED_RAW_COLUMNS - actual_columns
    unexpected_columns = actual_columns - REQUIRED_RAW_COLUMNS

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise DataContractError(f"缺少必要欄位：{missing_names}")

    if unexpected_columns:
        unexpected_names = ", ".join(sorted(unexpected_columns))
        raise DataContractError(f"出現未定義欄位：{unexpected_names}")


def validate_raw_data(events: pd.DataFrame) -> None:
    """驗證 ALPI 原始資料的欄位與缺值。"""
    validate_raw_columns(events.columns)

    missing_counts = events[list(sorted(REQUIRED_RAW_COLUMNS))].isna().sum()
    missing_values = {
        column: int(count)
        for column, count in missing_counts.items()
        if count > 0
    }

    if missing_values:
        details = ", ".join(
            f"{column}={count}"
            for column, count in sorted(missing_values.items())
        )
        raise DataContractError(f"必要欄位含有缺值：{details}")

    for column in ("alarm", "serial"):
        values = events[column].astype("string")
        invalid_count = int((~values.str.fullmatch(r"\d+")).sum())

        if invalid_count:
            raise DataContractError(
                f"{column} 含有非十進位數字識別碼：{invalid_count} 筆"
            )
