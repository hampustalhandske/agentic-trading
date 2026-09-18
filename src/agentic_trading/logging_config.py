# logging_config.py

"""Central logging setup for agentic_trading, shared by the CLI and the GUI.

Usage: import `get_logger` at module scope and use it directly — never pass
a logger as a function argument.

    from agentic_trading.logging_config import get_logger

    logger = get_logger(__name__)

    def run() -> None:
        logger.info("starting")

An entry point (the CLI's `run` command, the GUI's `render_dashboard()`)
calls `configure_logging()` — that's what wires up the actual handlers
(file + console + the GUI's in-memory buffer), using the real
`--verbose`/`--no-log` flags where applicable. `get_logger()` also
self-configures with defaults if nothing has configured anything yet, so a
module doing `logger = get_logger(__name__)` at import time always works —
but that self-configuration is tracked separately and does NOT block a
later *explicit* `configure_logging()` call from overriding it. This
distinction matters in practice: every module in this codebase calls
`get_logger(__name__)` at module scope, so importing `cli.py` (which
imports `backtest/engine.py`, which imports ...) triggers self-
configuration *before* `cli.py`'s own `run()` command body has parsed
`--verbose`/`--no-log` from argv — imports always happen before argument
parsing. Without this two-tier gate, the lazy default configuration would
win permanently and the CLI's flags would silently do nothing.

Two destinations, deliberately not treated the same:
- The log file (`log/agentic_trading.log`, created relative to the project
  root, opened in append mode — never truncated, so it accumulates across
  every run for as long as the file exists) is a lean, permanent record:
  a marker for each run starting/finishing, WARNINGs (wherever code falls
  back to a different behavior than the one it tried first — e.g. an
  analyst reporting flat because a lakehouse table has no rows), and
  ERRORs. It
  deliberately does NOT receive routine per-day/per-step progress lines —
  those would make the file grow by hundreds of lines on every single run
  for no lasting value.
- The console (stdout) and the GUI's in-memory buffer (see
  `recent_gui_log_lines()`) get everything the file gets, *plus* the
  routine progress lines (via `log_progress()`), so a user watching either
  one sees live "what's happening right now" feedback. `verbose=True` (the
  CLI's `--verbose`) raises just these two to DEBUG, which is where
  detailed per-step timing is logged — the file's level is untouched by
  verbose, so turning it on doesn't bloat the permanent record.
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from pathlib import Path

_PACKAGE_LOGGER_NAME = "agentic_trading"
_LOG_DIR = Path(__file__).resolve().parents[2] / "log"
_LOG_FILE = _LOG_DIR / "agentic_trading.log"
_LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_GUI_BUFFER_MAX_LINES = 500

_configured = False
_explicitly_configured = False
_gui_buffer: deque[str] = deque(maxlen=_GUI_BUFFER_MAX_LINES)


class _ExcludeProgressFilter(logging.Filter):
    """Keeps records logged via `log_progress()` (tagged
    `extra={"progress": True}`) out of the file handler — see the module
    docstring on why the file and the live views are held to different
    standards."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "progress", False)


class _GuiBufferHandler(logging.Handler):
    """Captures formatted log lines into an in-memory ring buffer the GUI
    renders inside the app itself (see `recent_gui_log_lines()`).
    Deliberately not a Streamlit call: `st.*` element calls made from
    inside an `@st.cache_data`-wrapped function break that function's
    cache-replay on a later cache hit (see `gui/dashboard.py`'s
    `_simulate_desk` docstring) — writing to a plain in-memory list has no
    such restriction, so logging is safe to do from inside cached
    simulation code even though rendering a live Streamlit widget from
    there is not.
    """

    def emit(self, record: logging.LogRecord) -> None:
        _gui_buffer.append(self.format(record))


def _apply_configuration(*, verbose: bool, disable: bool) -> None:
    """Does the actual handler setup — shared by the lazy (`get_logger()`)
    and explicit (`configure_logging()`) paths. Always tears down whatever
    handlers are currently attached first: the explicit path may be
    overriding a lazy default configuration that already opened a file
    handler, and leaving that open while attaching a second one would leak
    a file descriptor for the lifetime of the process."""
    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False

    if disable:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL + 1)
        return

    logger.setLevel(logging.DEBUG)  # handlers below each filter to their own level
    console_level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    # delay=True: don't open (create) the file until the first record is
    # actually emitted through this handler. Without this, get_logger()'s
    # lazy default configuration (see the module docstring) would create an
    # empty log file the instant any module is imported, even in a process
    # where an entry point's own explicit configure_logging(disable=True)
    # arrives moments later, before anything was ever logged — --no-log
    # should mean the file never comes into existence, not "exists but
    # stays empty."
    file_handler = logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8", delay=True)
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(_ExcludeProgressFilter())
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    gui_handler = _GuiBufferHandler()
    gui_handler.setLevel(console_level)
    gui_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(gui_handler)


def configure_logging(*, verbose: bool = False, disable: bool = False) -> None:
    """Configures the shared "agentic_trading" logger's handlers using the
    real `verbose`/`disable` flags. Only the *first explicit* call in a
    process does anything — a second explicit call (e.g. the GUI
    re-executing render_dashboard() on every rerun) is a no-op, so it's
    always safe to call this unconditionally at the top of an entry point
    without worrying about duplicate handlers piling up. This first-explicit
    call always wins over `get_logger()`'s lazy self-configuration (see the
    module docstring on why that race exists and why explicit must win),
    even if lazy configuration already happened by the time this runs.

    disable=True (the CLI's --no-log) turns off logging entirely: no
    handlers are attached and the log *file* is never created, and nothing
    is written anywhere for the lifetime of the process. (The empty log/
    directory itself may already exist if a lazy default configuration ran
    first — see the module docstring — but the file inside it, the thing
    that would actually accumulate content, never does.)

    verbose=True (the CLI's --verbose) raises the console and GUI-buffer
    handlers to DEBUG, surfacing detailed per-step timing. The file
    handler's level is independent of this flag — it always stays at INFO.
    """
    global _configured, _explicitly_configured
    if _explicitly_configured:
        return
    _explicitly_configured = True
    _configured = True
    _apply_configuration(verbose=verbose, disable=disable)


def get_logger(name: str) -> logging.Logger:
    """Returns a logger under the shared "agentic_trading" namespace.
    Self-configures with defaults (enabled, not verbose) on first use if
    nothing has configured anything yet. This lazy configuration is
    deliberately overridable by a later explicit `configure_logging()`
    call from an entry point — see the module docstring."""
    global _configured
    if not _configured:
        _configured = True
        _apply_configuration(verbose=False, disable=False)
    return logging.getLogger(name)


def log_progress(logger: logging.Logger, message: str, *args: object) -> None:
    """Logs a routine progress message (e.g. "day 5 of 41 simulated") at
    INFO — visible on the console and in the GUI's log panel by default,
    but excluded from the log file (see `_ExcludeProgressFilter`) so the
    file stays a lean run/warning/error record instead of accumulating
    hundreds of these lines on every single run."""
    logger.info(message, *args, extra={"progress": True})


def recent_gui_log_lines(limit: int = 200) -> list[str]:
    """The most recently captured formatted log lines, for the GUI's
    in-app "Run log" panel. Empty if logging is disabled (--no-log) or
    nothing has logged yet."""
    return list(_gui_buffer)[-limit:]


def log_file_path() -> Path:
    """Where the log file lives (or would live, if nothing has logged yet) —
    for callers that want to display the path to the user, e.g. the GUI's
    "Run log" panel."""
    return _LOG_FILE
