"""設備內時間序列切分的單元測試。"""

import pandas as pd
import pytest

from industrial_alarm_copilot.data.splitter import (
    assign_chronological_splits,
    validate_split_integrity,
)


def test_assign_chronological_splits_separates_each_machine_without_time_ties():
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2019-01-03",
                    "2019-01-04",
                    "2019-01-01",
                    "2019-01-02",
                    "2019-01-02",
                    "2019-01-04",
                    "2019-01-01",
                    "2019-01-02",
                    "2019-01-03",
                ]
            ),
            "alarm_code": ["11"] * 9,
            "machine_id": ["4", "6", "4", "4", "4", "4", "6", "6", "6"],
            "source_row": list(range(9)),
        }
    )

    result = assign_chronological_splits(
        events,
        train_ratio=0.50,
        validation_ratio=0.25,
        test_ratio=0.25,
    )

    machine_4 = result.loc[result["machine_id"] == "4"]
    machine_6 = result.loc[result["machine_id"] == "6"]
    assert machine_4["split"].tolist() == [
        "train",
        "train",
        "train",
        "validation",
        "test",
    ]
    assert machine_6["split"].tolist() == [
        "train",
        "train",
        "validation",
        "test",
    ]
    assert (
        result.groupby(["machine_id", "timestamp"])["split"].nunique()
        == 1
    ).all()


def test_validate_split_integrity_rejects_time_leakage():
    split_events = pd.DataFrame(
        {
            "machine_id": ["4", "4", "4"],
            "timestamp": pd.to_datetime(
                ["2019-01-02", "2019-01-01", "2019-01-03"]
            ),
            "split": ["train", "validation", "test"],
        }
    )

    with pytest.raises(ValueError, match="4"):
        validate_split_integrity(split_events)
