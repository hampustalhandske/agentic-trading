# conftest.py

"""Fixtures for the trial ledger tests: `tmp_path` is the repository root of every test."""

from pathlib import Path

import pytest


@pytest.fixture
def registry_file(tmp_path: Path) -> Path:
    """The registry file `research/registry/demo-v1.yaml` under `tmp_path`."""
    path = tmp_path / "research" / "registry" / "demo-v1.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("id: demo-v1\n# kept byte for byte\n")
    return path
