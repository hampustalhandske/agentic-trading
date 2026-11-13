# test_lakehouse_client_quiet_period.py

"""After a lake request times out, or a run is killed with a request unfinished, the lake may
keep working on it, so `Lake` makes no new call for `settings.lake.request_quiet_seconds`. Each
send writes a marker file in the client's state directory while it runs and clears it when
it ends; a timeout is recorded there too, with its process. A call that finds a timeout this
process recorded inside the window waits it out; one that finds a timeout of another process,
or a leftover marker (one no live request owns) started inside it, raises
`LakeQuietPeriodError` naming the earliest UTC time a call is allowed. A send after a timeout
within one call waits out the window as its retry wait. The timeout, the retry and the refusal
are logged. Every clock here is injected; nothing sleeps."""

import io
import json
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pyarrow as pa
import pytest

from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data import lakehouse_client
from agentic_trading.data.lakehouse_client import Lake, LakehouseError, LakeQuietPeriodError

_T0 = datetime(2026, 11, 10, 12, 0, 0, tzinfo=UTC)
_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh client state directory, named by the client's environment variable."""
    directory = tmp_path / "lake-state"
    monkeypatch.setenv(lakehouse_client.STATE_DIR_ENV_VAR, str(directory))
    return directory


class _Clock:
    """A settable UTC clock; `sleep` records each wait in `waits` and moves the clock on by it."""

    def __init__(self, now: datetime) -> None:
        self.now = now
        self.waits: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += timedelta(seconds=seconds)


def _arrow_body() -> bytes:
    sink = io.BytesIO()
    table = pa.table({"one": [1]})
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def _ok() -> httpx.Response:
    return httpx.Response(200, content=_arrow_body())


def _lake(handler: Callable[[httpx.Request], httpx.Response], clock: _Clock) -> Lake:
    return Lake(
        base_url="http://lake.test",
        token="t",
        transport=httpx.MockTransport(handler),
        sleep=clock.sleep,
        clock=clock,
    )


def _timing_out(sent: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    """A handler that records each request in `sent` and times out on the first, answering
    every later one with rows."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if len(sent) == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return _ok()

    return handler


def _markers(state_dir: Path) -> list[Path]:
    directory = state_dir / lakehouse_client.MARKER_DIR
    return sorted(directory.glob("*.json")) if directory.exists() else []


def _plant_marker(state_dir: Path, pid: int, started_at: datetime) -> Path:
    directory = state_dir / lakehouse_client.MARKER_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{pid}-planted.json"
    payload = {
        "pid": pid,
        "started_at": started_at.isoformat(),
        "method": "POST",
        "path": "/v1/sql",
    }
    path.write_text(json.dumps(payload))
    return path


def _dead_pid() -> int:
    """The PID of a process that has exited."""
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _quiet_seconds() -> float:
    return settings.lake.request_quiet_seconds


def test_the_quiet_period_is_the_lakes_deadline_plus_a_margin() -> None:
    """The window defaults to 330 s: the lake's 300 s request deadline plus 30 s."""
    assert _quiet_seconds() == 330.0


def test_a_timeout_is_recorded_logged_and_waited_out_before_the_send_again(
    state_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A request that times out leaves no marker and records the timeout with this process;
    the call waits out the quiet period, then sends again and returns the rows, and the timeout
    and the retry are both logged."""
    sent: list[httpx.Request] = []
    clock = _Clock(_T0)
    lake = _lake(_timing_out(sent), clock)
    with caplog.at_level(logging.WARNING, logger=lakehouse_client.__name__):
        assert lake.sql("SELECT 1 AS one")["one"].tolist() == [1]

    assert len(sent) == 2
    assert clock.waits[0] >= _quiet_seconds()
    assert _markers(state_dir) == []
    recorded = json.loads((state_dir / lakehouse_client.TIMEOUT_FILE).read_text())
    assert recorded["pid"] == os.getpid()
    messages = [record.getMessage() for record in caplog.records]
    assert any("timed out" in message for message in messages)
    assert any("sending again" in message and "ReadTimeout" in message for message in messages)


def test_this_processs_timeout_is_waited_out_by_a_new_call(state_dir: Path) -> None:
    """A timeout this process recorded (as before an `os.execv` restart, which keeps the pid)
    makes a new call wait until the window has passed, then send, rather than refuse."""
    state_dir.mkdir(parents=True)
    at = _T0 - timedelta(seconds=30)
    (state_dir / lakehouse_client.TIMEOUT_FILE).write_text(
        json.dumps({"at": at.isoformat(), "pid": os.getpid()})
    )
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return _ok()

    clock = _Clock(_T0)
    lake = _lake(handler, clock)
    assert lake.sql("SELECT 1 AS one")["one"].tolist() == [1]
    assert clock.waits == [pytest.approx(_quiet_seconds() - 30)]
    assert len(sent) == 1


def test_a_timeout_of_a_process_that_has_exited_refuses_a_new_call(state_dir: Path) -> None:
    """A timeout another process recorded inside the window refuses the call before anything
    is sent, as for a new process started after a kill."""
    state_dir.mkdir(parents=True)
    (state_dir / lakehouse_client.TIMEOUT_FILE).write_text(
        json.dumps({"at": _T0.isoformat(), "pid": _dead_pid()})
    )
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return _ok()

    lake = _lake(handler, _Clock(_T0 + timedelta(seconds=100)))
    with pytest.raises(LakeQuietPeriodError) as refused:
        lake.sql("SELECT 1 AS one")
    assert refused.value.earliest == _T0 + timedelta(seconds=_quiet_seconds())
    assert sent == []


def test_a_call_once_the_window_has_passed_is_sent(state_dir: Path) -> None:
    """At another process's timeout plus the window, the call is sent and returns its rows."""
    state_dir.mkdir(parents=True)
    (state_dir / lakehouse_client.TIMEOUT_FILE).write_text(json.dumps({"at": _T0.isoformat()}))
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return _ok()

    lake = _lake(handler, _Clock(_T0 + timedelta(seconds=_quiet_seconds())))
    assert lake.sql("SELECT 1 AS one")["one"].tolist() == [1]
    assert len(sent) == 1


def test_a_timeout_while_reading_rows_is_recorded(state_dir: Path) -> None:
    """A read that times out after the response has started is a cut-off answer: it is sent
    again at most `LIMITED_RETRIES` times, each after the quiet period, then raises the cut-off
    `LakehouseError`; the timeout is recorded."""

    class _TimingOutStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            raise httpx.ReadTimeout("timed out")
            yield b""

    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, stream=_TimingOutStream())

    clock = _Clock(_T0)
    lake = _lake(handler, clock)
    with pytest.raises(LakehouseError, match="cut off"):
        lake.sql("SELECT 1 AS one")
    assert len(sent) == 1 + lakehouse_client.LIMITED_RETRIES
    assert all(wait >= _quiet_seconds() for wait in clock.waits)
    assert (state_dir / lakehouse_client.TIMEOUT_FILE).exists()


def test_a_planted_timeout_refuses_another_client(state_dir: Path) -> None:
    """A timeout recorded by another process (a planted record) refuses a new client's first
    call, of any endpoint, within the window, and none after it."""
    state_dir.mkdir(parents=True)
    (state_dir / lakehouse_client.TIMEOUT_FILE).write_text(
        json.dumps({"at": (_T0 - timedelta(seconds=30)).isoformat()})
    )
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"status": "ok"})

    clock = _Clock(_T0)
    lake = _lake(handler, clock)
    with pytest.raises(LakeQuietPeriodError) as refused:
        lake.health()
    assert refused.value.earliest == _T0 + timedelta(seconds=_quiet_seconds() - 30)
    assert sent == []

    clock.now = refused.value.earliest
    assert lake.health() == {"status": "ok"}


def test_the_window_is_the_setting_read_at_call_time(state_dir: Path) -> None:
    """An override of `lake.request_quiet_seconds` moves the earliest allowed time."""
    state_dir.mkdir(parents=True)
    (state_dir / lakehouse_client.TIMEOUT_FILE).write_text(json.dumps({"at": _T0.isoformat()}))
    lake = _lake(lambda request: _ok(), _Clock(_T0 + timedelta(seconds=10)))
    with temporary_overrides({"lake.request_quiet_seconds": 60.0}):
        with pytest.raises(LakeQuietPeriodError) as refused:
            lake.sql("SELECT 1 AS one")
    assert refused.value.earliest == _T0 + timedelta(seconds=60)


@pytest.mark.parametrize("owner", ["dead process", "this process"])
def test_a_leftover_marker_refuses_until_its_start_plus_the_window(
    state_dir: Path, owner: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A marker no live request owns (its process has exited, or it names this process but no
    request of it) is a request a killed run left unfinished: a call within the window of its
    start is refused and logged; a call after it is sent and removes the marker."""
    pid = _dead_pid() if owner == "dead process" else os.getpid()
    started = _T0 - timedelta(seconds=200)
    marker = _plant_marker(state_dir, pid, started)
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return _ok()

    clock = _Clock(_T0)
    lake = _lake(handler, clock)
    with caplog.at_level(logging.WARNING, logger=lakehouse_client.__name__):
        with pytest.raises(LakeQuietPeriodError) as refused:
            lake.sql("SELECT 1 AS one")
    earliest = started + timedelta(seconds=_quiet_seconds())
    assert refused.value.earliest == earliest
    assert str(pid) in str(refused.value)
    assert sent == []
    assert any("refused" in record.getMessage() for record in caplog.records)

    clock.now = earliest
    assert lake.sql("SELECT 1 AS one")["one"].tolist() == [1]
    assert not marker.exists()
    assert _markers(state_dir) == []


def test_a_marker_of_a_live_other_process_is_a_request_in_flight(state_dir: Path) -> None:
    """A marker owned by another live process is that process's request in flight, not a
    leftover: the call is sent and the marker is left alone."""
    marker = _plant_marker(state_dir, os.getppid(), _T0 - timedelta(seconds=5))
    lake = _lake(lambda request: _ok(), _Clock(_T0))

    assert lake.sql("SELECT 1 AS one")["one"].tolist() == [1]
    assert marker.exists()


def test_a_request_holds_a_marker_while_in_flight_and_clears_it_after(state_dir: Path) -> None:
    """While a streamed result is read the state directory holds one marker naming this
    process and the request's start; it is gone after the read and after a refusal (an HTTP 500
    on every send) alike."""
    clock = _Clock(_T0)
    failed = [httpx.Response(500, json={"detail": "query failed"}) for _ in range(3)]
    answers = [_ok(), *failed]
    lake = _lake(lambda request: answers.pop(0), clock)
    for _ in lake.sql_batches("SELECT 1 AS one"):
        markers = _markers(state_dir)
        assert len(markers) == 1
        payload = json.loads(markers[0].read_text())
        assert payload["pid"] == os.getpid()
        assert datetime.fromisoformat(payload["started_at"]) == _T0
    assert _markers(state_dir) == []
    with pytest.raises(LakehouseError, match="HTTP 500"):
        lake.sql("SELECT 1 AS one")
    assert _markers(state_dir) == []
    assert not (state_dir / lakehouse_client.TIMEOUT_FILE).exists()


def test_concurrent_requests_of_one_process_never_refuse_each_other(state_dir: Path) -> None:
    """Eight threads whose requests are all in flight at once (the transport holds each until
    every one has been sent, with the query cap raised to eight) each get their rows: a request
    of this process in flight is never a leftover marker to another."""
    threads = 8
    barrier = threading.Barrier(threads, timeout=10)
    in_flight_markers: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        barrier.wait()
        in_flight_markers.append(len(_markers(state_dir)))
        return _ok()

    lake = _lake(handler, _Clock(_T0))
    results: list[object] = []
    lock = threading.Lock()

    def call() -> None:
        try:
            outcome: object = lake.sql("SELECT 1 AS one")["one"].tolist()
        except Exception as error:
            outcome = error
        with lock:
            results.append(outcome)

    workers = [threading.Thread(target=call) for _ in range(threads)]
    with temporary_overrides({"lake.max_concurrent_queries": threads}):
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=20)

    assert results == [[1]] * threads
    assert max(in_flight_markers) == threads
    assert _markers(state_dir) == []


def test_the_default_state_directory_is_per_user_outside_the_repo_and_tmp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without the environment variable the state directory is under the user's home, outside
    the repository and outside /tmp; the variable replaces it."""
    monkeypatch.delenv(lakehouse_client.STATE_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    default = lakehouse_client.state_dir().resolve()

    assert default.is_relative_to(Path.home().resolve())
    assert not default.is_relative_to(_REPO_ROOT)
    for temporary in (Path("/tmp"), Path("/private/tmp")):
        assert not default.is_relative_to(temporary)

    monkeypatch.setenv(lakehouse_client.STATE_DIR_ENV_VAR, str(tmp_path / "elsewhere"))
    assert lakehouse_client.state_dir() == tmp_path / "elsewhere"


def test_an_interrupted_request_leaves_its_marker(state_dir: Path) -> None:
    """A request interrupted by `KeyboardInterrupt` is one the run abandoned unfinished: its
    marker stays, and the next call inside the window of its start is refused."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise KeyboardInterrupt

    clock = _Clock(_T0)
    lake = _lake(handler, clock)
    with pytest.raises(KeyboardInterrupt):
        lake.sql("SELECT 1 AS one")
    assert len(_markers(state_dir)) == 1
    clock.now = _T0 + timedelta(seconds=10)

    with pytest.raises(LakeQuietPeriodError) as refused:
        lake.sql("SELECT 1 AS one")
    assert refused.value.earliest == _T0 + timedelta(seconds=_quiet_seconds())
