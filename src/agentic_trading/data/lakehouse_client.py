"""Python client for the lakehouse HTTP API: the transport behind every data read in
`data/loaders/lakehouse_loader.py`.

Needs `httpx`, `pyarrow` and `pandas`. Reads the service URL and this project's token from the
environment (`LAKEHOUSE_API_URL`, `LAKEHOUSE_API_TOKEN`) unless given explicitly:

    from agentic_trading.data.lakehouse_client import Lake

    lake = Lake()
    df = lake.sql("SELECT ticker, date, close FROM instrument_prices WHERE ticker = ? AND date >= ?",
                  ["AAPL", "2020-01-01"])              # pandas DataFrame, every row
    answer = lake.sql_answer("SELECT ...", [...])       # the Arrow table and its lake version
    for batch in lake.sql_batches("SELECT * FROM reddit_comments"):   # results larger than memory
        ...
    lake.tables()                                       # what exists, with coverage and status
    lake.describe("instrument_prices")                  # columns, keys, point-in-time rule, caveats
    lake.ask("What was the score in Liverpool vs Arsenal on 11 May 2025?")["answer"]

Dates come back as dates and timestamps as UTC timestamps. A query that fails part-way raises
instead of returning a partial result: a result is returned only when its response body reached
its end. `lake.last_version` holds the `X-Lakehouse-Version` header of the most recent response
(`None` before the first request or when the header is absent).

Retries. Reads are idempotent, so a call is sent again after a failure the lake recovers from,
until `settings.lake.retry_budget_seconds` (read at each call) has passed since the call began;
the error of the last send is raised when the next wait would end past the budget:

- sent again within the budget: an HTTP 502, 503 (any detail) or 504, and an httpx transport
  error before any response arrived (connection refused or reset, a protocol or read error, a
  timeout), as when launchd is starting or restarting the API;
- sent again at most `LIMITED_RETRIES` times in a call: an HTTP 500 or any other 5xx, and a
  response cut off part-way (`LakeStreamCutError`: the lake's whole-query deadline or memory
  limit, or the connection lost while reading), which is usually deterministic;
- raised at once: every 4xx (bad SQL, a missing build, 401, a 409 pin refusal, a 423 read
  freeze) and any other error.

The wait before send n + 1 is `RETRY_FIRST_WAIT_SECONDS * 2**(n - 1)`, capped at
`RETRY_MAX_WAIT_SECONDS`, scaled by a random factor within `RETRY_JITTER`, and never shorter than
the response's `Retry-After` seconds; after a timeout it is never shorter than the quiet period.
Each retry is logged as a warning naming the attempt, the error and the wait.

`/v1/sql` requests of the process, over every client and thread, are capped at
`settings.lake.max_concurrent_queries` in flight; a further one waits for a slot before it is
sent. A slot is held from the send until the response is done, never during a retry's wait.

Quiet period. After a request times out, or a run is killed with a request unfinished, the lake
may keep working on it, so no new call is sent for `settings.lake.request_quiet_seconds` (read at
each call). The rule holds across processes through files in the client's state directory
(`state_dir()`: the directory `AGENTIC_LAKE_STATE_DIR` names, else `agentic-trading/lake` under
the user's XDG state directory, `~/.local/state` by default). Each send writes a marker file under
`requests/` naming its process and start time when it starts, and removes it when it ends; a send
that times out (`httpx.TimeoutException`, also when it cuts off a streamed result) removes its
marker and records the time and its process in `last_timeout.json`, and one interrupted by
`KeyboardInterrupt` or `SystemExit` leaves its marker. Before the first send of a call the client
looks for a recorded timeout within the window, and for leftover markers (one whose process has
exited, or one naming this process that no request of it holds) started within it. A timeout this
process recorded (same pid, which an `os.execv` restart keeps) is waited out; any other cause
refuses the call with `LakeQuietPeriodError`, naming the earliest UTC time a call is allowed, as
for a new process started after a kill. A leftover marker older than the window is removed. A
marker of another live process is that process's request in flight and never refuses. Within a
call, a send after a timeout waits out the quiet period as part of its retry wait. The timeout,
the wait and the refusal are each logged as a warning.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_URL = "http://localhost:8600"
VERSION_HEADER = "X-Lakehouse-Version"
_SQL_PATH = "/v1/sql"

# Seconds before the second send of a call; each later wait doubles, up to the cap.
RETRY_FIRST_WAIT_SECONDS = 2.0
# The longest backoff wait between two sends (a `Retry-After` or the quiet period may be longer).
RETRY_MAX_WAIT_SECONDS = 60.0
# Each backoff wait is scaled by a factor drawn uniformly from [1 - jitter, 1 + jitter].
RETRY_JITTER = 0.2
# Sends again at most this many times in one call after a 5xx other than 502/503/504 or a cut-off
# response: such a failure usually repeats, and splitting the query is the fix.
LIMITED_RETRIES = 2
# Statuses that mean the API is unavailable for now (busy query slots, restarting).
_UNAVAILABLE_STATUSES = frozenset({502, 503, 504})
# Names the client's state directory (see `state_dir`).
STATE_DIR_ENV_VAR = "AGENTIC_LAKE_STATE_DIR"
# Under the state directory: one marker file per request in flight, and the last client timeout.
MARKER_DIR = "requests"
TIMEOUT_FILE = "last_timeout.json"


class LakehouseError(RuntimeError):
    """The API refused or failed the request; the message is the server's explanation. `status`
    is the HTTP status (`None` when no response carried one) and `retry_after` the response's
    `Retry-After` seconds (0 when absent)."""

    def __init__(self, message: str, status: int | None = None, retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class LakeStreamCutError(LakehouseError):
    """A response was cut off before its end: the lake stopped the query after it had started
    answering, or the connection was lost while the body was read. No partial result is
    returned."""


class LakeQuietPeriodError(RuntimeError):
    """A call refused before it was sent: a lake request timed out in another process, or a run
    left one unfinished, less than `settings.lake.request_quiet_seconds` ago, so the lake may
    still be working on it. `earliest` is the first UTC time a call is allowed; the message names
    it and each cause."""

    def __init__(self, message: str, earliest: datetime) -> None:
        super().__init__(message)
        self.earliest = earliest


class LakeAnswer(BaseModel):
    """One `/v1/sql` answer read to its end: `table` holds every row as the lake sent it, and
    `version` the response's `X-Lakehouse-Version` (`None` when it carried none)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    table: pa.Table
    version: str | None


def state_dir() -> Path:
    """The directory of the client's request markers and its last timeout: the one
    `AGENTIC_LAKE_STATE_DIR` names when it is set and non-empty, else `agentic-trading/lake` under
    `XDG_STATE_HOME`, or under `~/.local/state` when that is unset. Read at each call."""
    named = os.environ.get(STATE_DIR_ENV_VAR)
    if named:
        return Path(named).expanduser()
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base).expanduser() / "agentic-trading" / "lake"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _is_timeout(error: BaseException) -> bool:
    """Whether `error`, or an error it was raised from, is an httpx timeout."""
    seen: BaseException | None = error
    while seen is not None:
        if isinstance(seen, httpx.TimeoutException):
            return True
        seen = seen.__cause__
    return False


def _process_alive(pid: int) -> bool:
    """Whether a process with `pid` exists (one of another user counts)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Writes `payload` to `path` whole: through a temporary file beside it, then a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(json.dumps(payload))
    os.replace(partial, path)


class _ByteStream(io.RawIOBase):
    """A readable file object over an iterator of byte chunks (the HTTP response body)."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = chunks
        self._buffer = b""

    def readable(self) -> bool:
        return True

    def readinto(self, target) -> int:  # type: ignore[override]
        while not self._buffer:
            try:
                self._buffer = next(self._chunks)
            except StopIteration:
                return 0
            except httpx.TransportError as exc:
                raise LakeStreamCutError(
                    "the response was cut off: the query failed or timed out on the server after it had "
                    "started returning rows; no partial result is returned"
                ) from exc
        n = min(len(target), len(self._buffer))
        target[:n] = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return n


def _arrow_reader(resp: httpx.Response) -> pa.RecordBatchStreamReader:
    """An Arrow stream reader over `resp`'s body."""
    try:
        return pa.ipc.open_stream(io.BufferedReader(_ByteStream(resp.iter_bytes())))
    except pa.ArrowInvalid as exc:
        raise LakeStreamCutError(f"the response is not a whole Arrow stream: {exc}") from exc


def _read_arrow(resp: httpx.Response) -> pa.Table:
    """Every row of `resp`'s Arrow body; `LakeStreamCutError` when the body ends part-way."""
    reader = _arrow_reader(resp)
    try:
        return reader.read_all()
    except pa.ArrowInvalid as exc:
        raise LakeStreamCutError(f"the response ended part-way through the Arrow stream: {exc}") from exc


class _QuietCause(BaseModel):
    """One reason no call may be sent before `until`: `own` when this process recorded it."""

    until: datetime
    text: str
    own: bool


class _CallBudget:
    """The retry state of one call: when it began, the seconds it has waited, the backoff waits
    taken and the limited retries used."""

    def __init__(self, began: datetime) -> None:
        self.began = began
        self.waited = 0.0
        self.backoffs = 0
        self.limited = 0
        self.sends = 1

    def elapsed(self, now: datetime) -> float:
        """Seconds the call has taken: the time since it began, and at least its waits."""
        return max((now - self.began).total_seconds(), self.waited)


class Lake:
    """A client of the lakehouse API at `base_url` with `token`. `transport` replaces the HTTP
    transport (tests), `sleep` the function that waits and `clock` the one that gives the current
    aware time for the quiet period and the retry budget. `in_flight` counts this client's
    requests sent and not yet finished, over every thread, and `Lake.in_flight_in_process()`
    those of every client in the process."""

    _process_in_flight = 0
    _process_lock = threading.Lock()
    # The marker files of this process's requests in flight, over every client.
    _process_markers: set[Path] = set()
    # The `/v1/sql` requests of this process in flight, over every client, and its waiters.
    _query_slots_held = 0
    _query_slots = threading.Condition()

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 600.0,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.base_url = (base_url or os.getenv("LAKEHOUSE_API_URL", DEFAULT_URL)).rstrip("/")
        token = token or os.getenv("LAKEHOUSE_API_TOKEN", "")
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
        )
        self._sleep = sleep
        self._clock = clock
        self._in_flight = 0
        self._in_flight_lock = threading.Lock()
        self.last_version: str | None = None

    @property
    def in_flight(self) -> int:
        """The number of requests sent and not yet finished (their responses not yet fully read
        or closed), over every thread."""
        with self._in_flight_lock:
            return self._in_flight

    @classmethod
    def in_flight_in_process(cls) -> int:
        """The number of requests every `Lake` of this process has sent and not yet finished."""
        with cls._process_lock:
            return cls._process_in_flight

    @classmethod
    def queries_in_flight_in_process(cls) -> int:
        """The number of `/v1/sql` requests every `Lake` of this process holds a slot for."""
        with cls._query_slots:
            return cls._query_slots_held

    def _count_in_flight(self, change: int) -> None:
        with self._in_flight_lock:
            self._in_flight += change
        with Lake._process_lock:
            Lake._process_in_flight += change

    # -- data -----------------------------------------------------------------------

    def sql(self, query: str, params: list[Any] | None = None):
        """Every row of a read-only SELECT as a pandas DataFrame (dates as `datetime64`)."""
        return self.sql_arrow(query, params).to_pandas(date_as_object=False)

    def sql_arrow(self, query: str, params: list[Any] | None = None) -> pa.Table:
        """Every row as a pyarrow Table (convert with `.to_pandas()`, or `polars.from_arrow`)."""
        return self.sql_answer(query, params).table

    def sql_answer(self, query: str, params: list[Any] | None = None) -> LakeAnswer:
        """Every row as a pyarrow Table, with the lake version of the response that held it."""
        body = {"sql": query, "params": params or [], "format": "arrow"}

        def consume(resp: httpx.Response) -> LakeAnswer:
            return LakeAnswer(table=_read_arrow(resp), version=resp.headers.get(VERSION_HEADER))

        return self._request("POST", _SQL_PATH, consume, json=body)

    def sql_batches(self, query: str, params: list[Any] | None = None) -> Iterator[pa.RecordBatch]:
        """The result one record batch at a time, for results too large to hold at once. A
        failure before the first batch is retried like any call; one after it raises."""
        body = {"sql": query, "params": params or [], "format": "arrow"}
        budget = self._begin_call()
        while True:
            yielded = False
            try:
                with self._sent("POST", _SQL_PATH, json=body) as resp:
                    for batch in _arrow_reader(resp):
                        yielded = True
                        yield batch
                    return
            except Exception as error:
                if yielded or not self._wait_to_retry(error, budget, "POST", _SQL_PATH):
                    raise

    def to_parquet(self, query: str, path: str | Path, params: list[Any] | None = None) -> Path:
        """Write the result to one Parquet file at `path` without holding it in memory."""
        path = Path(path)
        body = {"sql": query, "params": params or [], "format": "parquet"}
        tmp = path.with_suffix(path.suffix + ".partial")

        def consume(resp: httpx.Response) -> Path:
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
            tmp.replace(path)
            return path

        return self._request("POST", _SQL_PATH, consume, json=body)

    # -- catalogue and questions ------------------------------------------------------

    def tables(self, include_logs: bool = False) -> list[dict[str, Any]]:
        return self._get("/v1/tables", params={"include_logs": include_logs})

    def describe(self, table: str) -> dict[str, Any]:
        return self._get(f"/v1/tables/{table}")

    def status(self) -> dict[str, Any]:
        """What the lake holds now: tables with coverage and status, backfills in progress, storage."""
        return self._get("/v1/status")

    def ask(self, question: str, as_of: str | None = None, include_evidence: bool = False) -> dict[str, Any]:
        """A natural-language answer with the SQL it rests on (needs the `ask` scope)."""
        body = {"question": question, "as_of": as_of, "include_evidence": include_evidence}
        return self._request("POST", "/v1/ask", _json_body, json=body)

    def health(self) -> dict[str, Any]:
        return self._get("/v1/health")

    # -- helpers ----------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, _json_body, params=params)

    def _request[T](self, method: str, path: str, consume: Callable[[httpx.Response], T], **kwargs: Any) -> T:
        """`consume` applied to the response to `method` `path`, sent again after each failure
        the retry rules allow (see the module docstring)."""
        budget = self._begin_call()
        while True:
            try:
                with self._sent(method, path, **kwargs) as resp:
                    return consume(resp)
            except Exception as error:
                if not self._wait_to_retry(error, budget, method, path):
                    raise

    def _begin_call(self) -> _CallBudget:
        """Starts a call: waits out a quiet period this process recorded, or raises
        `LakeQuietPeriodError` for one it did not (see the module docstring)."""
        now = self._now()
        budget = _CallBudget(now)
        causes = self._quiet_causes(state_dir(), now)
        if not causes:
            return budget
        earliest = max(cause.until for cause in causes)
        message = (
            f"lake call refused until {earliest.isoformat()} "
            f"({settings.lake.request_quiet_seconds:g} s quiet period): "
            + "; ".join(cause.text for cause in causes)
        )
        wait = (earliest - now).total_seconds()
        if any(not cause.own for cause in causes) or wait > settings.lake.retry_budget_seconds:
            logger.warning("%s", message)
            raise LakeQuietPeriodError(message, earliest)
        logger.warning(
            "Lake call waits %.1f s, until %s, for the quiet period after this process's timeout",
            wait,
            earliest.isoformat(),
        )
        self._pause(wait, budget)
        return budget

    def _pause(self, seconds: float, budget: _CallBudget) -> None:
        budget.waited += seconds
        self._sleep(seconds)

    def _wait_to_retry(self, error: Exception, budget: _CallBudget, method: str, path: str) -> bool:
        """Waits before the next send of a call that failed with `error` and returns `True`, or
        returns `False` when the error is raised at once or the call is out of retries."""
        status = error.status if isinstance(error, LakehouseError) else None
        unavailable = status in _UNAVAILABLE_STATUSES or isinstance(error, httpx.TransportError)
        limited = isinstance(error, LakeStreamCutError) or (
            status is not None and status >= 500 and not unavailable
        )
        if limited:
            if budget.limited >= LIMITED_RETRIES:
                return False
            budget.limited += 1
        elif not unavailable:
            return False
        backoff = min(RETRY_MAX_WAIT_SECONDS, RETRY_FIRST_WAIT_SECONDS * 2.0**budget.backoffs)
        wait = backoff * random.uniform(1.0 - RETRY_JITTER, 1.0 + RETRY_JITTER)
        if isinstance(error, LakehouseError):
            wait = max(wait, error.retry_after)
        if _is_timeout(error):
            wait = max(wait, settings.lake.request_quiet_seconds)
        if budget.elapsed(self._now()) + wait > settings.lake.retry_budget_seconds:
            logger.warning(
                "Lake %s %s failed on send %d (%s: %s); the retry budget of %g s is spent",
                method,
                path,
                budget.sends,
                type(error).__name__,
                error,
                settings.lake.retry_budget_seconds,
            )
            return False
        logger.warning(
            "Lake %s %s failed on send %d (%s: %s); sending again in %.1f s",
            method,
            path,
            budget.sends,
            type(error).__name__,
            error,
            wait,
        )
        budget.backoffs += 1
        budget.sends += 1
        self._pause(wait, budget)
        return True

    @contextlib.contextmanager
    def _query_slot(self, path: str) -> Iterator[None]:
        """Holds one of the process's `/v1/sql` slots while the block runs (any other path
        holds none), waiting for one while `settings.lake.max_concurrent_queries` are held."""
        if path != _SQL_PATH:
            yield
            return
        with Lake._query_slots:
            while Lake._query_slots_held >= settings.lake.max_concurrent_queries:
                Lake._query_slots.wait()
            Lake._query_slots_held += 1
        try:
            yield
        finally:
            with Lake._query_slots:
                Lake._query_slots_held -= 1
                Lake._query_slots.notify_all()

    @contextlib.contextmanager
    def _sent(self, method: str, path: str, **kwargs: Any) -> Iterator[httpx.Response]:
        """One send of `method` `path`: the streamed response, after recording its version and
        raising `LakehouseError` for a refusal. Holds a query slot and a request marker until
        the response is done; a transport error while the block reads the body raises
        `LakeStreamCutError`. A timeout is recorded for the quiet period."""
        with self._query_slot(path):
            marker = self._begin_request(method, path)
            keep_marker = False
            self._count_in_flight(1)
            try:
                with self._http.stream(method, path, **kwargs) as resp:
                    self._record_version(resp)
                    self._raise_for(resp)
                    try:
                        yield resp
                    except httpx.TransportError as exc:
                        raise LakeStreamCutError(
                            f"the response to {method} {path} was cut off while it was read: {exc}"
                        ) from exc
            except BaseException as error:
                if _is_timeout(error):
                    self._record_timeout(method, path)
                elif isinstance(error, KeyboardInterrupt | SystemExit):
                    keep_marker = True
                raise
            finally:
                self._count_in_flight(-1)
                self._end_request(marker, keep_marker)

    # -- quiet period -------------------------------------------------------------------

    def _now(self) -> datetime:
        return self._clock().astimezone(UTC)

    def _begin_request(self, method: str, path: str) -> Path:
        """Writes this send's marker and returns its path. The marker is held in this
        process's set before it is written, so no other thread reads it as left over."""
        marker = state_dir() / MARKER_DIR / f"{os.getpid()}-{uuid.uuid4().hex}.json"
        with Lake._process_lock:
            Lake._process_markers.add(marker)
        payload = {
            "pid": os.getpid(),
            "started_at": self._now().isoformat(),
            "method": method,
            "path": path,
        }
        try:
            _write_json(marker, payload)
        except BaseException:
            with Lake._process_lock:
                Lake._process_markers.discard(marker)
            raise
        return marker

    @staticmethod
    def _end_request(marker: Path, keep: bool) -> None:
        """Removes `marker` unless `keep`, then lets it go from this process's set, so a marker
        on disk is always either held or left over."""
        if not keep:
            marker.unlink(missing_ok=True)
        with Lake._process_lock:
            Lake._process_markers.discard(marker)

    def _record_timeout(self, method: str, path: str) -> None:
        """Records a client timeout of `method` `path` now in the state directory and logs it."""
        now = self._now()
        _write_json(
            state_dir() / TIMEOUT_FILE,
            {"at": now.isoformat(), "pid": os.getpid(), "method": method, "path": path},
        )
        resume = now + timedelta(seconds=settings.lake.request_quiet_seconds)
        logger.warning(
            "Lakehouse request %s %s timed out at %s; the lake may still be working on it, so no "
            "lake call is sent before %s",
            method,
            path,
            now.isoformat(),
            resume.isoformat(),
        )

    @staticmethod
    def _marker_start(marker: Path) -> tuple[int | None, datetime] | None:
        """The process and start time `marker` records, `None` when it is gone. A marker that
        cannot be read gives no process and its file's modification time."""
        try:
            text = marker.read_text()
            modified = datetime.fromtimestamp(marker.stat().st_mtime, UTC)
        except FileNotFoundError:
            return None
        try:
            payload = json.loads(text)
            return int(payload["pid"]), datetime.fromisoformat(payload["started_at"]).astimezone(UTC)
        except (ValueError, KeyError, TypeError):
            return None, modified

    def _quiet_causes(self, directory: Path, now: datetime) -> list[_QuietCause]:
        """Each recorded timeout and leftover marker less than the quiet period before `now`;
        removes each leftover marker older than that."""
        window = timedelta(seconds=settings.lake.request_quiet_seconds)
        causes: list[_QuietCause] = []
        timeout = directory / TIMEOUT_FILE
        if timeout.exists():
            pid: int | None
            try:
                recorded = json.loads(timeout.read_text())
                at = datetime.fromisoformat(recorded["at"]).astimezone(UTC)
                pid = recorded.get("pid")
            except (ValueError, KeyError, TypeError, AttributeError):
                at = datetime.fromtimestamp(timeout.stat().st_mtime, UTC)
                pid = None
            if now < at + window:
                causes.append(
                    _QuietCause(
                        until=at + window,
                        text=f"a lake request timed out at {at.isoformat()}",
                        own=pid == os.getpid(),
                    )
                )
        markers = directory / MARKER_DIR
        for marker in sorted(markers.glob("*.json")) if markers.is_dir() else []:
            with Lake._process_lock:
                if marker in Lake._process_markers:
                    continue
            started_by = self._marker_start(marker)
            if started_by is None:
                continue
            owner_pid, started = started_by
            if owner_pid is not None and owner_pid != os.getpid() and _process_alive(owner_pid):
                continue
            owner = "an unreadable marker" if owner_pid is None else f"process {owner_pid}"
            if now < started + window:
                causes.append(
                    _QuietCause(
                        until=started + window,
                        text=f"{owner} left a lake request unfinished that started at "
                        f"{started.isoformat()} ({marker})",
                        own=False,
                    )
                )
            else:
                marker.unlink(missing_ok=True)
                logger.info(
                    "Removed the marker of a lake request %s left unfinished at %s, older than "
                    "the quiet period",
                    owner,
                    started.isoformat(),
                )
        return causes

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float:
        """The response's `Retry-After` header in seconds, 0 when it is absent or not a number."""
        try:
            return max(float(resp.headers.get("Retry-After", "0")), 0.0)
        except ValueError:
            return 0.0

    def _record_version(self, resp: httpx.Response) -> None:
        """Sets `last_version` from `resp`'s `X-Lakehouse-Version` header (header names are
        case-insensitive), or to `None` when the response carries none."""
        self.last_version = resp.headers.get(VERSION_HEADER)

    @classmethod
    def _raise_for(cls, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        resp.read()
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise LakehouseError(
            f"HTTP {resp.status_code}: {detail}",
            status=resp.status_code,
            retry_after=cls._retry_after(resp),
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Lake:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _json_body(resp: httpx.Response) -> Any:
    """`resp`'s body read whole and parsed as JSON."""
    resp.read()
    return resp.json()
