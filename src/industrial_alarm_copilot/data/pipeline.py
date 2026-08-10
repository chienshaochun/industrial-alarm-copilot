"""組合可重現的 processed event 轉換流程。"""

import pandas as pd

from industrial_alarm_copilot.data.gaps import add_inter_event_gaps
from industrial_alarm_copilot.data.splitter import assign_chronological_splits


EVENT_KEY = ["timestamp", "alarm_code", "machine_id"]


def build_processed_events(
    events: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """加入重複標記、相鄰時間間隔與 chronological split。"""
    processed_events = events.copy()
    processed_events["duplicate_group_size"] = (
        processed_events.groupby(
            EVENT_KEY,
            observed=True,
            dropna=False,
        )["source_row"]
        .transform("size")
        .astype("int64")
    )
    processed_events["is_exact_duplicate"] = processed_events[
        "duplicate_group_size"
    ].gt(1)

    processed_events = add_inter_event_gaps(processed_events)

    return assign_chronological_splits(
        processed_events,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
