"""Paths and helpers shared by the tests of the scripts under `tools/`."""

from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "tools"


def write_stub(directory: Path, name: str, body: str) -> Path:
    """Writes an executable bash script `name` into `directory` running `body`; returns its path."""
    path = directory / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)
    return path
