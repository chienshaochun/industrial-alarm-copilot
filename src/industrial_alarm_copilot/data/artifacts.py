"""寫入可重新產生的資料 artifacts。"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def write_events_parquet(
    events: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """將處理後事件寫入 Zstandard 壓縮的 Parquet 檔案。"""
    parquet_path = Path(output_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(
        parquet_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    return parquet_path


def calculate_file_sha256(file_path: str | Path) -> str:
    """以分塊讀取方式計算檔案 SHA-256。"""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_events_metadata(
    events: pd.DataFrame,
    source_path: str | Path,
    pipeline_settings: dict[str, Any],
    code_version: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """建立 processed events artifact 的可序列化 metadata。"""
    generated_at = generated_at or datetime.now(UTC)
    if generated_at.tzinfo is None:
        raise ValueError("generated_at 必須包含時區")

    split_boundaries = {}
    for split_name, split_events in events.groupby("split", sort=False):
        split_boundaries[str(split_name)] = {
            "event_count": int(len(split_events)),
            "start_time": split_events["timestamp"].min().isoformat(),
            "end_time": split_events["timestamp"].max().isoformat(),
        }

    return {
        "artifact_schema_version": 1,
        "generated_at_utc": generated_at.astimezone(UTC).isoformat(),
        "code_version": code_version,
        "source": {
            "path": Path(source_path).as_posix(),
            "sha256": calculate_file_sha256(source_path),
        },
        "pipeline_settings": pipeline_settings,
        "events": {
            "row_count": int(len(events)),
            "machine_count": int(events["machine_id"].nunique()),
            "alarm_code_count": int(events["alarm_code"].nunique()),
            "columns": {
                column: str(dtype)
                for column, dtype in events.dtypes.items()
            },
            "split_boundaries": split_boundaries,
        },
    }


def write_metadata_json(
    metadata: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """以 UTF-8 JSON 寫入 artifact metadata。"""
    metadata_path = Path(output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return metadata_path
