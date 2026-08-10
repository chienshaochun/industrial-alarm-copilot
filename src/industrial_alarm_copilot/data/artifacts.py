"""寫入可重新產生的資料 artifacts。"""

from pathlib import Path

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
