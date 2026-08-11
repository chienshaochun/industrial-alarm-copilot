"""Industrial Alarm Copilot CLI 的單元測試。"""

from pathlib import Path

import industrial_alarm_copilot.__main__ as cli
from industrial_alarm_copilot.data.pipeline import PreparedArtifactPaths


def test_prepare_data_command_passes_paths_settings_and_version(
    monkeypatch,
    tmp_path,
    capsys,
):
    raw_csv_path = tmp_path / "raw.csv"
    config_path = tmp_path / "default.toml"
    output_dir = tmp_path / "processed"
    settings = {"split": {"train_ratio": 0.70}}
    received = {}

    monkeypatch.setattr(
        cli,
        "load_pipeline_settings",
        lambda path: settings,
    )
    monkeypatch.setattr(
        cli,
        "get_git_commit",
        lambda path: "abc1234",
    )

    def fake_prepare_data_artifacts(**kwargs):
        received.update(kwargs)
        return PreparedArtifactPaths(
            events_parquet=output_dir / "events.parquet",
            metadata_json=output_dir / "events.metadata.json",
        )

    monkeypatch.setattr(
        cli,
        "prepare_data_artifacts",
        fake_prepare_data_artifacts,
    )

    exit_code = cli.main(
        [
            "prepare-data",
            "--raw-csv",
            str(raw_csv_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert received == {
        "raw_csv_path": raw_csv_path,
        "output_dir": output_dir,
        "pipeline_settings": settings,
        "code_version": "abc1234",
    }
    output = capsys.readouterr().out
    assert str(output_dir / "events.parquet") in output
    assert "abc1234" in output
