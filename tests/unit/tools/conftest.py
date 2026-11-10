# conftest.py

"""Fixtures for the tests of the scripts under `tools/`: a directory of stub commands put first on
PATH, the environment a script runs in (git isolated from the user's configuration), and a
scratch git repository with one commit."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def stubs(tmp_path: Path) -> Path:
    """An empty directory that `tool_env` puts first on PATH, for stub commands."""
    directory = tmp_path / "stubs"
    directory.mkdir()
    return directory


@pytest.fixture
def tool_env(tmp_path: Path, stubs: Path) -> dict[str, str]:
    """This process's environment with `stubs` first on PATH, no inherited `GIT_*` variable, a
    scratch HOME and global git config, and a fixed commit identity."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    home = tmp_path / "home"
    home.mkdir()
    env.update(
        HOME=str(home),
        GIT_CONFIG_GLOBAL=str(home / ".gitconfig"),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="test",
        GIT_AUTHOR_EMAIL="test@example.com",
        GIT_COMMITTER_NAME="test",
        GIT_COMMITTER_EMAIL="test@example.com",
        PATH=f"{stubs}{os.pathsep}{os.environ['PATH']}",
    )
    return env


@pytest.fixture
def repo(tmp_path: Path, tool_env: dict[str, str]) -> tuple[Path, str]:
    """A git repository with one commit holding `tests/test_placeholder.py`, and that commit's
    full sha."""
    path = tmp_path / "repo"
    (path / "tests").mkdir(parents=True)
    (path / "tests" / "test_placeholder.py").write_text("")

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=path, env=tool_env, capture_output=True, text=True, check=True
        )
        return done.stdout.strip()

    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "initial")
    return path, git("rev-parse", "HEAD")
