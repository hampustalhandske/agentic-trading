"""Unit tests for agentic_trading.logging_config.

configure_logging() is deliberately a do-once-per-process gate (see its own
docstring), and the whole test session runs with logging disabled from the
very first import (see tests/conftest.py) — so every test in this file
resets the module's internal state itself (`_configured`, the shared
"agentic_trading" logger's handlers, and the in-memory GUI buffer) via the
`_reset_logging_state` fixture below, and redirects the file handler to a
pytest tmp_path rather than the real log/ directory. This is the standard
way to test a process-wide singleton-style config module: reach into its
internals directly rather than relying on the self-configuring behavior
every other module in the codebase depends on.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

import agentic_trading.logging_config as logging_config
from agentic_trading.logging_config import (
    configure_logging,
    get_logger,
    log_progress,
    recent_gui_log_lines,
)


@pytest.fixture
def _reset_logging_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Resets configure_logging()'s do-once gate and the shared logger's
    handlers before each test, redirects the log file to tmp_path so no
    test ever touches the real log/agentic_trading.log, and restores
    everything afterward so later tests (and the rest of the suite) aren't
    affected by whatever a given test configured."""
    logger = logging.getLogger("agentic_trading")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    original_configured = logging_config._configured
    original_explicitly_configured = logging_config._explicitly_configured
    original_buffer = list(logging_config._gui_buffer)

    log_dir = tmp_path / "log"
    monkeypatch.setattr(logging_config, "_LOG_DIR", log_dir)
    monkeypatch.setattr(logging_config, "_LOG_FILE", log_dir / "agentic_trading.log")
    logger.handlers.clear()
    logging_config._configured = False
    logging_config._explicitly_configured = False
    logging_config._gui_buffer.clear()

    yield log_dir

    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.handlers.extend(original_handlers)
    logger.setLevel(original_level)
    logger.propagate = original_propagate
    logging_config._configured = original_configured
    logging_config._explicitly_configured = original_explicitly_configured
    logging_config._gui_buffer.clear()
    logging_config._gui_buffer.extend(original_buffer)


def _read_log_file(log_dir: Path) -> str:
    log_file = log_dir / "agentic_trading.log"
    return log_file.read_text() if log_file.exists() else ""


def test_configure_logging_creates_the_log_directory_and_file(
    _reset_logging_state: Path,
) -> None:
    log_dir = _reset_logging_state
    assert not log_dir.exists()

    configure_logging()
    get_logger("agentic_trading.test_logging_config").info("hello")

    assert log_dir.exists()
    assert (log_dir / "agentic_trading.log").exists()
    assert "hello" in _read_log_file(log_dir)


def test_configure_logging_appends_across_multiple_process_runs(
    _reset_logging_state: Path,
) -> None:
    """Simulates two separate process runs writing to the same file: the
    second must not truncate what the first wrote."""
    log_dir = _reset_logging_state

    configure_logging()
    get_logger("agentic_trading.test_logging_config").info("first run message")
    logging.getLogger("agentic_trading").handlers.clear()
    logging_config._configured = False
    logging_config._explicitly_configured = False

    configure_logging()
    get_logger("agentic_trading.test_logging_config").info("second run message")

    content = _read_log_file(log_dir)
    assert "first run message" in content
    assert "second run message" in content


def test_get_logger_self_configures_when_nothing_called_configure_logging(
    _reset_logging_state: Path,
) -> None:
    log_dir = _reset_logging_state

    logger = get_logger("agentic_trading.some_module")
    logger.info("auto-configured")

    assert "auto-configured" in _read_log_file(log_dir)


def test_configure_logging_second_call_is_a_no_op(_reset_logging_state: Path) -> None:
    """A repeat call (e.g. the GUI re-executing render_dashboard() on every
    rerun) must not add duplicate handlers, which would duplicate every
    subsequent log line."""
    configure_logging()
    handler_count_after_first_call = len(logging.getLogger("agentic_trading").handlers)

    configure_logging(verbose=True)  # different args — still must no-op

    assert len(logging.getLogger("agentic_trading").handlers) == handler_count_after_first_call


def test_explicit_configure_logging_overrides_a_prior_lazy_get_logger_call(
    _reset_logging_state: Path,
) -> None:
    """Regression test for a real bug: every module in the codebase calls
    get_logger(__name__) at module scope, so importing cli.py (which
    imports backtest/engine.py, which imports ...) triggers get_logger()'s
    lazy self-configuration *before* cli.py's own run() command body has
    parsed --verbose/--no-log from argv — imports always happen before
    argument parsing runs. If the lazy configuration were allowed to "win"
    permanently, an entry point's own explicit configure_logging() call
    would silently no-op and --no-log/--verbose would have zero effect —
    exactly what happened before this two-tier _configured/
    _explicitly_configured gate was added. This test simulates that import
    order directly: a lazy get_logger() call first (as module imports would
    trigger), then the entry point's explicit call with real flags, which
    must actually take effect."""
    log_dir = _reset_logging_state

    # Simulates a module-level `logger = get_logger(__name__)` running
    # during import, before any entry point has parsed CLI flags.
    get_logger("agentic_trading.some_imported_module")

    # Simulates the entry point's own explicit call, now that --no-log has
    # been parsed from argv.
    configure_logging(disable=True)
    get_logger("agentic_trading.test_logging_config").error("must not appear anywhere")

    # The lazy get_logger() call above may have already created the empty
    # log/ directory (see configure_logging()'s docstring) — what actually
    # matters is that the log *file* inside it, which would hold real
    # content, was never created (FileHandler's delay=True defers opening
    # it until first emit, and no emit ever happened through that
    # now-torn-down handler).
    assert not (log_dir / "agentic_trading.log").exists()
    assert recent_gui_log_lines() == []


def test_disable_prevents_any_file_or_console_output(_reset_logging_state: Path) -> None:
    log_dir = _reset_logging_state

    configure_logging(disable=True)
    get_logger("agentic_trading.test_logging_config").error("should not appear anywhere")

    assert not log_dir.exists()
    assert recent_gui_log_lines() == []


def test_verbose_makes_debug_messages_visible_via_the_gui_buffer(
    _reset_logging_state: Path,
) -> None:
    configure_logging(verbose=True)
    get_logger("agentic_trading.test_logging_config").debug("debug detail")

    assert any("debug detail" in line for line in recent_gui_log_lines())


def test_non_verbose_hides_debug_messages_from_the_gui_buffer(_reset_logging_state: Path) -> None:
    configure_logging(verbose=False)
    get_logger("agentic_trading.test_logging_config").debug("debug detail")

    assert recent_gui_log_lines() == []


def test_log_progress_reaches_console_and_gui_buffer_but_not_the_file(
    _reset_logging_state: Path,
) -> None:
    log_dir = _reset_logging_state
    configure_logging()

    log_progress(get_logger("agentic_trading.test_logging_config"), "day %d of %d", 3, 10)

    assert any("day 3 of 10" in line for line in recent_gui_log_lines())
    assert "day 3 of 10" not in _read_log_file(log_dir)


def test_ordinary_info_message_does_reach_the_file(_reset_logging_state: Path) -> None:
    """Contrast with log_progress: a plain logger.info() call (a run
    starting/finishing, not a routine per-step tick) must land in the file —
    only progress-tagged records are excluded."""
    log_dir = _reset_logging_state
    configure_logging()

    get_logger("agentic_trading.test_logging_config").info("run started")

    assert "run started" in _read_log_file(log_dir)


def test_warning_and_error_reach_the_file_even_without_verbose(
    _reset_logging_state: Path,
) -> None:
    log_dir = _reset_logging_state
    configure_logging(verbose=False)

    get_logger("agentic_trading.test_logging_config").warning("falling back to synthetic data")
    get_logger("agentic_trading.test_logging_config").error("something broke")

    content = _read_log_file(log_dir)
    assert "falling back to synthetic data" in content
    assert "something broke" in content


def test_recent_gui_log_lines_respects_limit(_reset_logging_state: Path) -> None:
    configure_logging()
    logger = get_logger("agentic_trading.test_logging_config")
    for i in range(10):
        logger.info("line %d", i)

    lines = recent_gui_log_lines(limit=3)

    assert len(lines) == 3
    assert "line 9" in lines[-1]


def test_get_logger_returns_a_logger_named_after_its_argument(_reset_logging_state: Path) -> None:
    logger = get_logger("agentic_trading.some.module")

    assert logger.name == "agentic_trading.some.module"
