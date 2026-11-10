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
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa

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


class LakehouseError(RuntimeError):
    """The API refused or failed the request; the message is the server's explanation."""


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
    (see the module docstring) and `sleep` the function that waits. `in_flight` counts this
    client's requests sent and not yet finished, over every thread, and
    `Lake.in_flight_in_process()` those of every client in the process."""

    _process_in_flight = 0
    _process_lock = threading.Lock()

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 600.0,
        *,
        transport: httpx.BaseTransport | None = None,
        busy_retry_delays: tuple[float, ...] = BUSY_RETRY_DELAYS,
        sleep: Callable[[float], None] = time.sleep,
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
        `LakehouseError` for a refusal. A busy-slots answer is sent again after each of the busy
        retry delays, each logged; the answer after the last delay raises like any other."""
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
