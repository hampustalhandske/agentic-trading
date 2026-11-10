# test_lakehouse_client_quiet_period.py

"""After a lake request times out, or a run is killed with a request unfinished, the lake may
keep working on it, so `Lake` makes no new call for `settings.lake.request_quiet_seconds`. Each
request writes a marker file in the client's state directory while it runs and clears it when
it ends; a timeout is recorded there too. A call that finds a timeout recorded inside the window,
or a leftover marker (one no live request owns) started inside it, raises
`LakeQuietPeriodError` naming the earliest UTC time a call is allowed, and both the timeout and
the refusal are logged. Every clock here is injected; nothing sleeps."""

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
    """A settable UTC clock."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


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
        busy_retry_delays=(),
        sleep=lambda _: None,
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


def test_a_timeout_is_recorded_logged_and_refuses_the_next_call_inside_the_window(
    state_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A request that times out raises as before, leaves no marker and records the timeout; a
    call 100 s later is refused before anything is sent, naming the earliest allowed UTC time,
    and both are logged."""
    sent: list[httpx.Request] = []
    clock = _Clock(_T0)
    lake = _lake(_timing_out(sent), clock)
    with caplog.at_level(logging.WARNING, logger=lakehouse_client.__name__):
        with pytest.raises(httpx.ReadTimeout):
            lake.sql("SELECT 1 AS one")
        clock.now = _T0 + timedelta(seconds=100)
        with pytest.raises(LakeQuietPeriodError) as refused:
            lake.sql("SELECT 1 AS one")

    earliest = _T0 + timedelta(seconds=_quiet_seconds())
    assert refused.value.earliest == earliest
    assert earliest.isoformat() in str(refused.value)
    assert len(sent) == 1
    assert _markers(state_dir) == []
    assert (state_dir / lakehouse_client.TIMEOUT_FILE).exists()
    messages = [record.getMessage() for record in caplog.records]
    assert any("timed out" in message for message in messages)
    assert any("refused" in message and earliest.isoformat() in message for message in messages)


def test_a_call_once_the_window_has_passed_is_sent(state_dir: Path) -> None:
    """At the timeout plus the window, the next call is sent and returns its rows."""
    sent: list[httpx.Request] = []
    clock = _Clock(_T0)
    lake = _lake(_timing_out(sent), clock)
    with pytest.raises(httpx.ReadTimeout):
        lake.sql("SELECT 1 AS one")
    clock.now = _T0 + timedelta(seconds=_quiet_seconds())

    assert lake.sql("SELECT 1 AS one")["one"].tolist() == [1]
    assert len(sent) == 2


def test_a_timeout_while_reading_rows_is_recorded(state_dir: Path) -> None:
    """A read that times out after the response has started raises the cut-off
    `LakehouseError` and records the timeout, so the next call inside the window is refused."""

    class _TimingOutStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            raise httpx.ReadTimeout("timed out")
            yield b""

    clock = _Clock(_T0)
    lake = _lake(lambda request: httpx.Response(200, stream=_TimingOutStream()), clock)
    with pytest.raises(LakehouseError, match="cut off"):
        lake.sql("SELECT 1 AS one")
    clock.now = _T0 + timedelta(seconds=1)

    with pytest.raises(LakeQuietPeriodError):
        lake.sql("SELECT 1 AS one")


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
    process and the request's start; it is gone after the read, after a refusal and after a
    timeout alike."""
    clock = _Clock(_T0)
    answers = [_ok(), httpx.Response(500, json={"detail": "query failed"})]
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
    every one has been sent) each get their rows: a request of this process in flight is never
    a leftover marker to another."""
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
