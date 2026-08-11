"""Processed event 組合流程的單元測試。"""

import pandas as pd

from industrial_alarm_copilot.data.pipeline import build_processed_events


def test_build_processed_events_adds_duplicates_gaps_and_splits():
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2019-02-21 10:00:00",
                    "2019-02-21 10:00:00",
                    "2019-02-21 10:05:00",
                    "2019-02-21 10:40:00",
                    "2019-02-21 10:45:00",
                ]
            ),
            "alarm_code": pd.Series(
                ["139", "139", "31", "11", "26"],
                dtype="string",
            ),
            "machine_id": pd.Series(["4"] * 5, dtype="string"),
            "source_row": [0, 1, 2, 3, 4],
        }
    )

    result = build_processed_events(
        events,
        train_ratio=0.50,
        validation_ratio=0.25,
        test_ratio=0.25,
    )

    assert result["source_row"].tolist() == [0, 1, 2, 3, 4]
    assert result["duplicate_group_size"].tolist() == [2, 2, 1, 1, 1]
    assert result["is_exact_duplicate"].tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert result["gap_seconds"].iloc[1:].tolist() == [
        0.0,
        300.0,
        2100.0,
        300.0,
    ]
    assert result["split"].tolist() == [
        "train",
        "train",
        "train",
        "validation",
        "test",
    ]
