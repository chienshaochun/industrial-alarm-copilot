"""ALPI 警報事件的探索性資料統計。"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetOverview:
    """資料集的全域概況。"""

    event_count: int
    machine_count: int
    alarm_code_count: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp


@dataclass(frozen=True)
class DataQualityOverview:
    """重複事件與時間順序的資料品質概況。"""

    exact_duplicate_group_count: int
    rows_in_duplicate_groups: int
    duplicate_excess_row_count: int
    global_time_reversal_count: int
    within_machine_time_reversal_count: int


def build_dataset_overview(events: pd.DataFrame) -> DatasetOverview:
    """由正規化警報事件建立全域概況。"""
    return DatasetOverview(
        event_count=len(events),
        machine_count=int(events["machine_id"].nunique()),
        alarm_code_count=int(events["alarm_code"].nunique()),
        start_time=events["timestamp"].min(),
        end_time=events["timestamp"].max(),
    )


def build_data_quality_overview(events: pd.DataFrame) -> DataQualityOverview:
    """統計完全相同事件與相鄰時間回退次數。"""
    event_key = ["timestamp", "alarm_code", "machine_id"]
    in_duplicate_group = events.duplicated(subset=event_key, keep=False)
    duplicate_excess = events.duplicated(subset=event_key, keep="first")
    duplicate_group_count = events.loc[in_duplicate_group, event_key].drop_duplicates()

    global_reversals = events["timestamp"].diff().lt(pd.Timedelta(0))
    within_machine_reversals = (
        events.groupby("machine_id", observed=True, sort=False)["timestamp"]
        .diff()
        .lt(pd.Timedelta(0))
    )

    return DataQualityOverview(
        exact_duplicate_group_count=len(duplicate_group_count),
        rows_in_duplicate_groups=int(in_duplicate_group.sum()),
        duplicate_excess_row_count=int(duplicate_excess.sum()),
        global_time_reversal_count=int(global_reversals.sum()),
        within_machine_time_reversal_count=int(
            within_machine_reversals.sum()
        ),
    )


def build_machine_profile(events: pd.DataFrame) -> pd.DataFrame:
    """依設備彙整警報事件分布。"""
    profile = (
        events.groupby("machine_id", observed=True)
        .agg(
            event_count=("alarm_code", "size"),
            alarm_code_count=("alarm_code", "nunique"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
        )
        .reset_index()
    )
    profile["event_share"] = profile["event_count"] / len(events)

    return (
        profile[
            [
                "machine_id",
                "event_count",
                "event_share",
                "alarm_code_count",
                "start_time",
                "end_time",
            ]
        ]
        .sort_values(
            ["event_count", "machine_id"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def build_alarm_profile(events: pd.DataFrame) -> pd.DataFrame:
    """依警報代碼彙整事件分布。"""
    profile = (
        events.groupby("alarm_code", observed=True)
        .agg(
            event_count=("machine_id", "size"),
            machine_count=("machine_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["event_count", "alarm_code"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    profile["event_share"] = profile["event_count"] / len(events)
    profile["cumulative_share"] = profile["event_share"].cumsum()

    return profile[
        [
            "alarm_code",
            "event_count",
            "event_share",
            "cumulative_share",
            "machine_count",
        ]
    ]


def build_monthly_profile(events: pd.DataFrame) -> pd.DataFrame:
    """依月份彙整警報事件分布。"""
    monthly_events = events.assign(
        month_start=events["timestamp"].dt.to_period("M").dt.to_timestamp()
    )
    profile = (
        monthly_events.groupby("month_start", observed=True)
        .agg(
            event_count=("alarm_code", "size"),
            machine_count=("machine_id", "nunique"),
            alarm_code_count=("alarm_code", "nunique"),
        )
        .reset_index()
        .sort_values("month_start", kind="stable")
        .reset_index(drop=True)
    )
    profile["event_share"] = profile["event_count"] / len(events)

    return profile[
        [
            "month_start",
            "event_count",
            "event_share",
            "machine_count",
            "alarm_code_count",
        ]
    ]
