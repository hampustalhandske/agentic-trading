"""Python client for the lakehouse HTTP API: the transport behind every data read in
`data/loaders/lakehouse_loader.py`.

Needs `httpx`, `pyarrow` and `pandas`. Reads the service URL and this project's token from the
environment (`LAKEHOUSE_API_URL`, `LAKEHOUSE_API_TOKEN`) unless given explicitly:

    from agentic_trading.data.lakehouse_client import Lake

    lake = Lake()
    df = lake.sql("SELECT ticker, date, close FROM instrument_prices WHERE ticker = ? AND date >= ?",
                  ["AAPL", "2020-01-01"])              # pandas DataFrame, every row
    for batch in lake.sql_batches("SELECT * FROM reddit_comments"):   # results larger than memory
        ...
    lake.tables()                                       # what exists, with coverage and status
    lake.describe("instrument_prices")                  # columns, keys, point-in-time rule, caveats
    lake.ask("What was the score in Liverpool vs Arsenal on 11 May 2025?")["answer"]

Dates come back as dates and timestamps as UTC timestamps. A query that fails part-way raises
instead of returning a partial result. `lake.last_version` holds the `X-Lakehouse-Version` header
of the most recent response (`None` before the first request or when the header is absent).

An HTTP 503 whose detail says the lake's query slots are busy is sent again after each of
`BUSY_RETRY_DELAYS`, each scaled by a random factor within `BUSY_RETRY_JITTER` and never shorter
than the response's `Retry-After` seconds. The lake itself waits up to 30 s for a slot before it
answers busy, and holds the slot of a request whose client went away until the request's 300 s
deadline, so the seven sends span at least about six and a half minutes. Each retry is logged as
a warning, and the answer to the last send raises like any other. Every other refusal or failure
raises at once.

After a request times out, or a run is killed with a request unfinished, the lake may keep working
on it, so no new call is sent for `settings.lake.request_quiet_seconds` (read at each call). The
rule holds across processes through files in the client's state directory (`state_dir()`: the
directory `AGENTIC_LAKE_STATE_DIR` names, else `agentic-trading/lake` under the user's XDG state
directory, `~/.local/state` by default). Each request writes a marker file under `requests/`
naming its process and start time when it starts, and removes it when it ends; a request that
times out (`httpx.TimeoutException`, also when it cuts off a streamed result) removes its marker
and records the time in `last_timeout.json`, and one interrupted by `KeyboardInterrupt` or
`SystemExit` leaves its marker. Before each call the client refuses with `LakeQuietPeriodError`,
naming the earliest UTC time a call is allowed, when the recorded timeout is within the window,
or when a leftover marker (one whose process has exited, or one naming this process that no
request of it holds) started within the window. A leftover marker older than the window is
removed. A marker of another live process is that process's request in flight and never refuses.
The timeout and the refusal are each logged as a warning.
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

from agentic_trading.config import settings
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_URL = "http://localhost:8600"
VERSION_HEADER = "X-Lakehouse-Version"

# Seconds to wait before each new send of a request the lake answered with busy query slots: seven
# sends in all, four minutes of waiting before the last (see the module docstring).
BUSY_RETRY_DELAYS: tuple[float, ...] = (10.0, 20.0, 30.0, 60.0, 60.0, 60.0)
# Each delay is scaled by a factor drawn uniformly from [1 - jitter, 1 + jitter].
BUSY_RETRY_JITTER = 0.2
# The status and the text of its detail that mean the lake's query slots are all in use.
_BUSY_STATUS = 503
_BUSY_DETAIL = "query slots are busy"
# Names the client's state directory (see `state_dir`).
STATE_DIR_ENV_VAR = "AGENTIC_LAKE_STATE_DIR"
# Under the state directory: one marker file per request in flight, and the last client timeout.
MARKER_DIR = "requests"
TIMEOUT_FILE = "last_timeout.json"


class LakehouseError(RuntimeError):
    """The API refused or failed the request; the message is the server's explanation."""


class LakeQuietPeriodError(RuntimeError):
    """A call refused before it was sent: a lake request timed out, or a run left one unfinished,
    less than `settings.lake.request_quiet_seconds` ago, so the lake may still be working on it.
    `earliest` is the first UTC time a call is allowed; the message names it and each cause."""

    def __init__(self, message: str, earliest: datetime) -> None:
        super().__init__(message)
        self.earliest = earliest


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
                raise LakehouseError(
                    "the response was cut off: the query failed or timed out on the server after it had "
                    "started returning rows; no partial result is returned"
                ) from exc
        n = min(len(target), len(self._buffer))
        target[:n] = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return n


class Lake:
    """A client of the lakehouse API at `base_url` with `token`. `transport` replaces the HTTP
    transport (tests), `busy_retry_delays` the waits before each new send of a busy-slots answer
    (see the module docstring), `sleep` the function that waits and `clock` the one that gives the
    current aware time for the quiet period. `in_flight` counts this client's requests sent and
    not yet finished, over every thread, and `Lake.in_flight_in_process()` those of every client
    in the process."""

    _process_in_flight = 0
    _process_lock = threading.Lock()
    # The marker files of this process's requests in flight, over every client.
    _process_markers: set[Path] = set()

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 600.0,
        *,
        transport: httpx.BaseTransport | None = None,
        busy_retry_delays: tuple[float, ...] = BUSY_RETRY_DELAYS,
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
        self._busy_retry_delays = busy_retry_delays
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
        with self._arrow_reader(query, params) as reader:
            return reader.read_all()

    def sql_batches(self, query: str, params: list[Any] | None = None) -> Iterator[pa.RecordBatch]:
        """The result one record batch at a time, for results too large to hold at once."""
        with self._arrow_reader(query, params) as reader:
            yield from reader

    @contextlib.contextmanager
    def _arrow_reader(self, query: str, params: list[Any] | None) -> Iterator[pa.RecordBatchStreamReader]:
        body = {"sql": query, "params": params or [], "format": "arrow"}
        with self._response("POST", "/v1/sql", json=body) as resp:
            yield pa.ipc.open_stream(io.BufferedReader(_ByteStream(resp.iter_bytes())))

    def to_parquet(self, query: str, path: str | Path, params: list[Any] | None = None) -> Path:
        """Write the result to one Parquet file at `path` without holding it in memory."""
        path = Path(path)
        body = {"sql": query, "params": params or [], "format": "parquet"}
        tmp = path.with_suffix(path.suffix + ".partial")
        with self._response("POST", "/v1/sql", json=body) as resp:
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
        tmp.replace(path)
        return path

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
        with self._response("POST", "/v1/ask", json=body) as resp:
            resp.read()
            return resp.json()

    def health(self) -> dict[str, Any]:
        return self._get("/v1/health")

    # -- helpers ----------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        with self._response("GET", path, params=params) as resp:
            resp.read()
            return resp.json()

    @contextlib.contextmanager
    def _response(self, method: str, path: str, **kwargs: Any) -> Iterator[httpx.Response]:
        """The streamed response to `method` `path`, after recording its version and raising
        `LakehouseError` for a refusal. Raises `LakeQuietPeriodError` before sending inside the
        quiet period, and holds a request marker until the response is done (see the module
        docstring). A busy-slots answer is sent again after each of the busy retry delays, each
        logged; the answer after the last delay raises like any other."""
        marker = self._begin_request(method, path)
        keep_marker = False
        try:
            with self._sent(method, path, **kwargs) as resp:
                yield resp
        except BaseException as error:
            if _is_timeout(error):
                self._record_timeout(method, path)
            elif isinstance(error, KeyboardInterrupt | SystemExit):
                keep_marker = True
            raise
        finally:
            self._end_request(marker, keep_marker)

    @contextlib.contextmanager
    def _sent(self, method: str, path: str, **kwargs: Any) -> Iterator[httpx.Response]:
        """The sends of `_response`, busy retries included."""
        delays = self._busy_retry_delays
        for attempt in range(len(delays) + 1):
            self._count_in_flight(1)
            try:
                with self._http.stream(method, path, **kwargs) as resp:
                    self._record_version(resp)
                    if attempt == len(delays) or not self._is_busy(resp):
                        self._raise_for(resp)
                        yield resp
                        return
            finally:
                self._count_in_flight(-1)
            wait = max(
                delays[attempt] * random.uniform(1.0 - BUSY_RETRY_JITTER, 1.0 + BUSY_RETRY_JITTER),
                self._retry_after(resp),
            )
            logger.warning(
                "Lakehouse query slots busy (HTTP 503) on %s %s; send %d of %d in %.1f s",
                method,
                path,
                attempt + 2,
                len(delays) + 1,
                wait,
            )
            self._sleep(wait)

    # -- quiet period -------------------------------------------------------------------

    def _now(self) -> datetime:
        return self._clock().astimezone(UTC)

    def _begin_request(self, method: str, path: str) -> Path:
        """Refuses inside the quiet period, then writes this request's marker and returns its
        path. The marker is held in this process's set before it is written, so no other thread
        reads it as left over."""
        now = self._now()
        directory = state_dir()
        self._refuse_inside_quiet_period(directory, now)
        marker = directory / MARKER_DIR / f"{os.getpid()}-{uuid.uuid4().hex}.json"
        with Lake._process_lock:
            Lake._process_markers.add(marker)
        payload = {"pid": os.getpid(), "started_at": now.isoformat(), "method": method, "path": path}
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

    def _refuse_inside_quiet_period(self, directory: Path, now: datetime) -> None:
        """Raises `LakeQuietPeriodError` when the recorded timeout, or a leftover marker's start,
        is less than the quiet period before `now`; removes each leftover marker older than that."""
        window = timedelta(seconds=settings.lake.request_quiet_seconds)
        causes: list[tuple[datetime, str]] = []
        timeout = directory / TIMEOUT_FILE
        if timeout.exists():
            try:
                at = datetime.fromisoformat(json.loads(timeout.read_text())["at"]).astimezone(UTC)
            except (ValueError, KeyError, TypeError):
                at = datetime.fromtimestamp(timeout.stat().st_mtime, UTC)
            if now < at + window:
                causes.append((at + window, f"a lake request timed out at {at.isoformat()}"))
        markers = directory / MARKER_DIR
        for marker in sorted(markers.glob("*.json")) if markers.is_dir() else []:
            with Lake._process_lock:
                if marker in Lake._process_markers:
                    continue
            recorded = self._marker_start(marker)
            if recorded is None:
                continue
            pid, started = recorded
            if pid is not None and pid != os.getpid() and _process_alive(pid):
                continue
            owner = "an unreadable marker" if pid is None else f"process {pid}"
            if now < started + window:
                causes.append(
                    (
                        started + window,
                        f"{owner} left a lake request unfinished that started at "
                        f"{started.isoformat()} ({marker})",
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
        if causes:
            earliest = max(when for when, _ in causes)
            message = (
                f"lake call refused until {earliest.isoformat()} "
                f"({settings.lake.request_quiet_seconds:g} s quiet period): "
                + "; ".join(cause for _, cause in causes)
            )
            logger.warning("%s", message)
            raise LakeQuietPeriodError(message, earliest)

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float:
        """The response's `Retry-After` header in seconds, 0 when it is absent or not a number."""
        try:
            return max(float(resp.headers.get("Retry-After", "0")), 0.0)
        except ValueError:
            return 0.0

    @staticmethod
    def _is_busy(resp: httpx.Response) -> bool:
        """Whether `resp` is the lake's answer that every query slot is in use."""
        if resp.status_code != _BUSY_STATUS:
            return False
        resp.read()
        return _BUSY_DETAIL in resp.text

    def _record_version(self, resp: httpx.Response) -> None:
        """Sets `last_version` from `resp`'s `X-Lakehouse-Version` header (header names are
        case-insensitive), or to `None` when the response carries none."""
        self.last_version = resp.headers.get(VERSION_HEADER)

    @staticmethod
    def _raise_for(resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        resp.read()
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise LakehouseError(f"HTTP {resp.status_code}: {detail}")

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Lake:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
