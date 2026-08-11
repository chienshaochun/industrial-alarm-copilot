"""完整 ALPI 原始資料與 processed events 的契約驗收。"""

from pathlib import Path

from industrial_alarm_copilot.data.artifacts import calculate_file_sha256
from industrial_alarm_copilot.data.loader import load_alarm_events
from industrial_alarm_copilot.data.pipeline import build_processed_events


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "alarms.csv"
EXPECTED_RAW_SHA256 = (
    "53bd4414a6fb5b6875a9535f1be622d"
    "fb2dfba69de407d071a52d0d304160d1a"
)


def test_complete_alpi_dataset_matches_processed_event_contract():
    assert calculate_file_sha256(RAW_CSV_PATH) == EXPECTED_RAW_SHA256

    events = load_alarm_events(RAW_CSV_PATH)
    processed_events = build_processed_events(events)

    assert processed_events.shape == (444_834, 9)
    assert processed_events["machine_id"].nunique() == 20
    assert processed_events["alarm_code"].nunique() == 154
    assert int(processed_events["is_exact_duplicate"].sum()) == 166
    assert int(processed_events["gap_seconds"].isna().sum()) == 20
    assert processed_events["gap_seconds"].dropna().ge(0).all()
    assert processed_events["source_row"].is_unique
    assert processed_events["split"].value_counts().to_dict() == {
        "train": 311_368,
        "test": 66_739,
        "validation": 66_727,
    }
