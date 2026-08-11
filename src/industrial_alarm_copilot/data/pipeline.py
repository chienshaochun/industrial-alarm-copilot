"""組合可重現的 processed event 轉換流程。"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from industrial_alarm_copilot.data.artifacts import (
    build_events_metadata,
    write_events_parquet,
    write_metadata_json,
)
from industrial_alarm_copilot.data.gaps import add_inter_event_gaps
from industrial_alarm_copilot.data.loader import load_alarm_events
from industrial_alarm_copilot.data.splitter import assign_chronological_splits


EVENT_KEY = ["timestamp", "alarm_code", "machine_id"]


@dataclass(frozen=True)
class PreparedArtifactPaths:
    """Prepare-data 流程產生的 artifact 路徑。"""

    events_parquet: Path
    metadata_json: Path


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


def prepare_data_artifacts(
    raw_csv_path: str | Path,
    output_dir: str | Path,
    pipeline_settings: dict[str, Any],
    code_version: str,
    generated_at: datetime | None = None,
) -> PreparedArtifactPaths:
    """由原始 ALPI CSV 產生 processed events 與 metadata。"""
    split_settings = pipeline_settings["split"]
    events = load_alarm_events(raw_csv_path)
    processed_events = build_processed_events(
        events,
        train_ratio=float(split_settings["train_ratio"]),
        validation_ratio=float(split_settings["validation_ratio"]),
        test_ratio=float(split_settings["test_ratio"]),
    )

    output_dir = Path(output_dir)
    events_path = output_dir / "events.parquet"
    metadata_path = output_dir / "events.metadata.json"
    metadata = build_events_metadata(
        processed_events,
        source_path=raw_csv_path,
        pipeline_settings=pipeline_settings,
        code_version=code_version,
        generated_at=generated_at,
    )

    write_events_parquet(processed_events, events_path)
    write_metadata_json(metadata, metadata_path)

    return PreparedArtifactPaths(
        events_parquet=events_path,
        metadata_json=metadata_path,
    )
