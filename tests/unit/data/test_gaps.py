"""同設備相鄰警報時間間隔的單元測試。"""

import pandas as pd

from industrial_alarm_copilot.data.gaps import (
    add_inter_event_gaps,
    build_gap_threshold_profile,
)


def test_add_inter_event_gaps_sorts_and_never_crosses_machines():
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2019-02-21 10:00:00",
                    "2019-02-21 09:00:00",
                    "2019-02-21 10:05:00",
                    "2019-02-21 09:02:00",
                ]
            ),
            "alarm_code": ["139", "31", "11", "26"],
            "machine_id": ["4", "6", "4", "6"],
            "source_row": [0, 1, 2, 3],
        }
    )

    result = add_inter_event_gaps(events)

    assert result["machine_id"].tolist() == ["4", "4", "6", "6"]
    assert result["source_row"].tolist() == [0, 2, 1, 3]
    assert result["previous_timestamp"].isna().tolist() == [
        True,
        False,
        True,
        False,
    ]
    assert result["gap_seconds"].tolist()[1::2] == [300.0, 120.0]


def test_build_gap_threshold_profile_counts_incident_breaks():
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2019-02-21 10:00:00",
                    "2019-02-21 09:00:00",
                    "2019-02-21 10:05:00",
                    "2019-02-21 09:02:00",
                ]
            ),
            "alarm_code": ["139", "31", "11", "26"],
            "machine_id": ["4", "6", "4", "6"],
            "source_row": [0, 1, 2, 3],
        }
    )

    profile = build_gap_threshold_profile(events, [1, 3, 5])

    expected = pd.DataFrame(
        {
            "threshold_minutes": [1.0, 3.0, 5.0],
            "within_threshold_count": [0, 1, 2],
            "within_threshold_share": [0.0, 0.5, 1.0],
            "incident_break_count": [2, 1, 0],
            "estimated_incident_count": [4, 3, 2],
        }
    )
    pd.testing.assert_frame_equal(profile, expected)
