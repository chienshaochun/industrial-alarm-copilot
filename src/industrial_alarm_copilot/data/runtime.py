"""Prepare-data 流程的設定與版本資訊。"""

import subprocess
import tomllib
from pathlib import Path
from typing import Any


def load_pipeline_settings(config_path: str | Path) -> dict[str, Any]:
    """讀取 TOML pipeline 設定。"""
    with Path(config_path).open("rb") as config_file:
        return tomllib.load(config_file)


def get_git_commit(repository_path: str | Path) -> str:
    """取得 repository 目前 HEAD 的完整 commit hash。"""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError("Git HEAD commit 不可為空")

    return commit
