# edgelab_live.py

"""The live edgelab drift checks, and the guard that keeps every other skip out of the suite.

A live check compares a vendored edgelab file with an edgelab checkout. It is marked
`edgelab_live` and finds the checkout through `edgelab_path`, which skips it with a reason
starting with `EDGELAB_ABSENT` when the checkout or the file it needs is not there. The vendored
copies themselves are pinned by sha256 in tests that always run
(`tests/unit/test_edgelab_fixture_pins.py`).

`configure(config)`, called from `tests/conftest.py`'s `pytest_configure`, registers the marker
and a `SkipGuard`: the run fails when any test or collector is skipped other than an
`edgelab_live` test skipped for `EDGELAB_ABSENT`, and its terminal summary reports how many live
checks were skipped that way and lists every other skip."""

import os
from pathlib import Path

import pytest

MARKER = "edgelab_live"
EDGELAB_ABSENT = "edgelab checkout absent"


def edgelab_dir() -> Path | None:
    """The edgelab checkout: `EDGELAB_DIR` when set, else `edgelab` beside this repo's parent
    directory; `None` when the repo sits too near the filesystem root to have one."""
    configured = os.environ.get("EDGELAB_DIR")
    if configured:
        return Path(configured)
    parents = Path(__file__).resolve().parents
    return parents[3] / "edgelab" if len(parents) > 3 else None


def edgelab_path(relative: Path) -> Path:
    """`relative` inside the edgelab checkout; skips the calling test with a reason starting
    with `EDGELAB_ABSENT` when there is no checkout or nothing at that path."""
    edgelab = edgelab_dir()
    path = edgelab / relative if edgelab is not None else None
    if path is None or not path.exists():
        pytest.skip(f"{EDGELAB_ABSENT}: no {relative}; set EDGELAB_DIR to check drift")
    return path


def _skip_reason(report: pytest.TestReport | pytest.CollectReport) -> str:
    """The message a skipped report carries."""
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2]).removeprefix("Skipped: ")
    return str(longrepr)


class SkipGuard:
    """Sorts the run's skips into live edgelab checks skipped for `EDGELAB_ABSENT` and all
    others; any other skip fails the run."""

    def __init__(self) -> None:
        self.live: list[str] = []
        self.other: list[str] = []

    def _record(self, report: pytest.TestReport | pytest.CollectReport, live_check: bool) -> None:
        reason = _skip_reason(report)
        live = live_check and reason.startswith(EDGELAB_ABSENT)
        (self.live if live else self.other).append(f"{report.nodeid}: {reason}")

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.skipped:
            self._record(report, MARKER in report.keywords)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        """A skipped module or class is never a live check."""
        if report.skipped:
            self._record(report, live_check=False)

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        passing = (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED)
        if self.other and session.exitstatus in passing:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        terminalreporter.write_line(
            f"{MARKER}: {len(self.live)} live edgelab check(s) skipped ({EDGELAB_ABSENT})"
        )
        if self.other:
            terminalreporter.write_line(
                f"skip guard: {len(self.other)} skip(s) not allowed; the run fails:", red=True
            )
            for entry in self.other:
                terminalreporter.write_line(f"  {entry}", red=True)


def configure(config: pytest.Config) -> None:
    """Registers the `edgelab_live` marker and a `SkipGuard` on `config`."""
    config.addinivalue_line(
        "markers",
        f"{MARKER}: compares a vendored edgelab file with an edgelab checkout; the one kind of "
        f"test allowed to skip, and only for '{EDGELAB_ABSENT}'",
    )
    config.pluginmanager.register(SkipGuard(), "edgelab-skip-guard")
