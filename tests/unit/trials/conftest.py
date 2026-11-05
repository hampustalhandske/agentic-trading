# conftest.py

"""Fixtures for the trial ledger tests: `tmp_path` is the repository root of every test."""

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def registry_file(tmp_path: Path) -> Path:
    """The registry file `research/registry/demo-v1.yaml` under `tmp_path`."""
    path = tmp_path / "research" / "registry" / "demo-v1.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("id: demo-v1\n# kept byte for byte\n")
    return path


@pytest.fixture
def prior_exposure() -> dict[str, Any]:
    """A made-up record in docs/prior_exposure.json's shape: two windows whose results were
    seen, the first under two issues, plus one data-only and one decisions-after-results
    entry, which count as no trial."""
    return {
        "results_seen": [
            {
                "start": "2010-01-14",
                "end": "2010-01-21",
                "universe": ["TLT", "SHY", "IEF"],
                "issues": ["AT-1", "AT-2 part 1"],
                "seen": "2026-10-28",
                "printed": "firm NAV, Sharpe; per-pod return",
            },
            {
                "start": "2011-03-01",
                "end": "2011-03-04",
                "universe": ["SPY"],
                "issues": ["AT-3"],
                "seen": "2026-10-29",
                "printed": "firm Sharpe",
            },
        ],
        "decisions_after_results": [
            {
                "start": "2012-01-03",
                "end": "2012-01-06",
                "universe": ["SPY"],
                "issues": ["AT-4"],
                "seen": "2026-10-30",
                "printed": "nothing",
            }
        ],
        "data_only": [
            {
                "start": "2013-01-02",
                "end": "2013-01-04",
                "universe": ["GLD"],
                "issues": ["AT-5"],
                "seen": "2026-10-31",
                "printed": "bar counts",
            }
        ],
    }
