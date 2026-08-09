"""計算同一設備內相鄰警報的時間間隔。"""

import pandas as pd


def add_inter_event_gaps(events: pd.DataFrame) -> pd.DataFrame:
    """穩定排序事件，並加入前一事件時間與間隔秒數。"""
    events_with_gaps = events.sort_values(
        ["machine_id", "timestamp", "source_row"],
        kind="stable",
    ).reset_index(drop=True)

    events_with_gaps["previous_timestamp"] = events_with_gaps.groupby(
        "machine_id",
        observed=True,
        sort=False,
    )["timestamp"].shift(1)
    events_with_gaps["gap_seconds"] = (
        events_with_gaps["timestamp"]
        - events_with_gaps["previous_timestamp"]
    ).dt.total_seconds()

    return events_with_gaps


def build_gap_threshold_profile(
    events: pd.DataFrame,
    thresholds_minutes: list[float],
) -> pd.DataFrame:
    """比較不同 incident gap 門檻產生的事件切分規模。"""
    events_with_gaps = add_inter_event_gaps(events)
    gap_minutes = events_with_gaps["gap_seconds"].dropna() / 60
    machine_count = int(events_with_gaps["machine_id"].nunique())

    rows = []
    for threshold_minutes in thresholds_minutes:
        within_threshold_count = int(
            gap_minutes.le(threshold_minutes).sum()
        )
        incident_break_count = int(
            gap_minutes.gt(threshold_minutes).sum()
        )
        rows.append(
            {
                "threshold_minutes": float(threshold_minutes),
                "within_threshold_count": within_threshold_count,
                "within_threshold_share": (
                    within_threshold_count / len(gap_minutes)
                ),
                "incident_break_count": incident_break_count,
                "estimated_incident_count": (
                    machine_count + incident_break_count
                ),
            }
        )

    return pd.DataFrame(rows)
