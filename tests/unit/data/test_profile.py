"""探索性資料統計的單元測試。"""

import pandas as pd

from industrial_alarm_copilot.data.profile import (
    DataQualityOverview,
    DatasetOverview,
    build_alarm_profile,
    build_data_quality_overview,
    build_dataset_overview,
    build_machine_profile,
    build_monthly_profile,
)


def test_build_dataset_overview_counts_events_machines_and_alarms():
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2019-02-21 10:00:00",
                    "2019-02-21 11:00:00",
                    "2019-02-22 12:00:00",
                ]
            ),
            "alarm_code": ["139", "31", "139"],
            "machine_id": ["4", "4", "13"],
            "source_row": [0, 1, 2],
        }
    )

    overview = build_dataset_overview(events)

    assert overview == DatasetOverview(
        event_count=3,
        machine_count=2,
        alarm_code_count=2,
        start_time=pd.Timestamp("2019-02-21 10:00:00"),
        end_time=pd.Timestamp("2019-02-22 12:00:00"),
    )


def test_build_data_quality_overview_counts_duplicates_and_reversals():
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2019-02-21 10:00:00",
                    "2019-02-21 10:00:00",
                    "2019-02-21 09:00:00",
                    "2019-02-21 12:00:00",
                    "2019-02-21 11:00:00",
                ]
            ),
            "alarm_code": ["139", "139", "31", "139", "31"],
            "machine_id": ["4", "4", "4", "13", "13"],
            "source_row": [0, 1, 2, 3, 4],
        }
    )

    overview = build_data_quality_overview(events)

    assert overview == DataQualityOverview(
        exact_duplicate_group_count=1,
        rows_in_duplicate_groups=2,
        duplicate_excess_row_count=1,
        global_time_reversal_count=2,
        within_machine_time_reversal_count=2,
    )


def test_build_machine_profile_summarizes_each_machine():
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2019-02-21 10:00:00",
                    "2019-02-21 11:00:00",
                    "2019-02-22 12:00:00",
                ]
            ),
            "alarm_code": ["139", "31", "139"],
            "machine_id": ["4", "4", "13"],
            "source_row": [0, 1, 2],
        }
    )

    profile = build_machine_profile(events)

    expected = pd.DataFrame(
        {
            "machine_id": ["4", "13"],
            "event_count": [2, 1],
            "event_share": [2 / 3, 1 / 3],
            "alarm_code_count": [2, 1],
            "start_time": pd.to_datetime(
                ["2019-02-21 10:00:00", "2019-02-22 12:00:00"]
            ),
            "end_time": pd.to_datetime(
                ["2019-02-21 11:00:00", "2019-02-22 12:00:00"]
            ),
        }
    )
    pd.testing.assert_frame_equal(profile, expected)


def test_build_alarm_profile_summarizes_frequency_and_reach():
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2019-02-21 10:00:00",
                    "2019-02-21 11:00:00",
                    "2019-02-22 12:00:00",
                ]
            ),
            "alarm_code": ["139", "31", "139"],
            "machine_id": ["4", "4", "13"],
            "source_row": [0, 1, 2],
        }
    )

    profile = build_alarm_profile(events)

    expected = pd.DataFrame(
        {
            "alarm_code": ["139", "31"],
            "event_count": [2, 1],
            "event_share": [2 / 3, 1 / 3],
            "cumulative_share": [2 / 3, 1.0],
            "machine_count": [2, 1],
        }
    )
    pd.testing.assert_frame_equal(profile, expected)


def test_build_monthly_profile_summarizes_temporal_distribution():
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2019-02-21 10:00:00",
                    "2019-02-28 11:00:00",
                    "2019-03-01 12:00:00",
                ]
            ),
            "alarm_code": ["139", "31", "139"],
            "machine_id": ["4", "4", "13"],
            "source_row": [0, 1, 2],
        }
    )

    profile = build_monthly_profile(events)

    expected = pd.DataFrame(
        {
            "month_start": pd.to_datetime(["2019-02-01", "2019-03-01"]),
            "event_count": [2, 1],
            "event_share": [2 / 3, 1 / 3],
            "machine_count": [1, 1],
            "alarm_code_count": [2, 1],
        }
    )
    pd.testing.assert_frame_equal(profile, expected)
