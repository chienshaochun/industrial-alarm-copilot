"""ALPI schema 驗證的單元測試。"""

import pytest

from industrial_alarm_copilot.data.schema import (
    DataContractError,
    validate_raw_columns,
)


def test_validate_raw_columns_rejects_missing_column():
    columns = ["timestamp", "alarm"]

    with pytest.raises(DataContractError, match="serial"):
        validate_raw_columns(columns)


def test_validate_raw_columns_rejects_unexpected_column():
    columns = ["timestamp", "alarm", "serial", "status"]

    with pytest.raises(DataContractError, match="status"):
        validate_raw_columns(columns)
