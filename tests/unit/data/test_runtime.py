"""Pipeline 設定與版本資訊的單元測試。"""

import subprocess

from industrial_alarm_copilot.data.runtime import (
    get_git_commit,
    load_pipeline_settings,
)


def test_load_pipeline_settings_reads_toml(tmp_path):
    config_path = tmp_path / "default.toml"
    config_path.write_text(
        "[split]\n"
        "train_ratio = 0.70\n"
        "validation_ratio = 0.15\n"
        "test_ratio = 0.15\n",
        encoding="utf-8",
    )

    settings = load_pipeline_settings(config_path)

    assert settings["split"] == {
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
    }


def test_get_git_commit_returns_stripped_hash(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        assert command == ["git", "rev-parse", "HEAD"]
        assert kwargs["cwd"] == tmp_path
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout="abc1234\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert get_git_commit(tmp_path) == "abc1234"
