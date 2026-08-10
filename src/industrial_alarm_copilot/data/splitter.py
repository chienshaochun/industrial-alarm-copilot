"""依設備與時間切分警報事件。"""

from math import isclose

import pandas as pd


SPLIT_NAMES = ("train", "validation", "test")


def validate_split_integrity(split_events: pd.DataFrame) -> None:
    """確認每台設備都有三個 split，且時間邊界依序前進。"""
    invalid_machines = []

    for machine_id, machine_events in split_events.groupby(
        "machine_id",
        observed=True,
        sort=False,
    ):
        split_names = set(machine_events["split"])
        if split_names != set(SPLIT_NAMES):
            invalid_machines.append(str(machine_id))
            continue

        train_max = machine_events.loc[
            machine_events["split"] == "train", "timestamp"
        ].max()
        validation_times = machine_events.loc[
            machine_events["split"] == "validation", "timestamp"
        ]
        test_min = machine_events.loc[
            machine_events["split"] == "test", "timestamp"
        ].min()

        if not (train_max < validation_times.min()):
            invalid_machines.append(str(machine_id))
            continue
        if not (validation_times.max() < test_min):
            invalid_machines.append(str(machine_id))

    if invalid_machines:
        machine_names = ", ".join(sorted(invalid_machines))
        raise ValueError(
            f"chronological split 時間邊界無效：{machine_names}"
        )


def assign_chronological_splits(
    events: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """在每台設備內依唯一 timestamp 指派 train、validation、test。"""
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios) or not isclose(sum(ratios), 1.0):
        raise ValueError("split 比例必須全部大於 0，且總和等於 1")

    sorted_events = events.sort_values(
        ["machine_id", "timestamp", "source_row"],
        kind="stable",
    ).reset_index(drop=True)
    unique_timestamps = sorted_events[
        ["machine_id", "timestamp"]
    ].drop_duplicates()

    grouped_timestamps = unique_timestamps.groupby(
        "machine_id",
        observed=True,
        sort=False,
    )
    timestamp_position = grouped_timestamps.cumcount()
    timestamp_count = grouped_timestamps["timestamp"].transform("size")
    train_end = (timestamp_count * train_ratio).astype(int)
    validation_end = (
        timestamp_count * (train_ratio + validation_ratio)
    ).astype(int)

    unique_timestamps["split"] = "test"
    unique_timestamps.loc[
        timestamp_position < validation_end,
        "split",
    ] = "validation"
    unique_timestamps.loc[
        timestamp_position < train_end,
        "split",
    ] = "train"

    split_events = sorted_events.merge(
        unique_timestamps,
        on=["machine_id", "timestamp"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    validate_split_integrity(split_events)

    return split_events
