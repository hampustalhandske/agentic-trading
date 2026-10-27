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
instead of returning a partial result.
"""

from __future__ import annotations

import contextlib
import io
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa

DEFAULT_URL = "http://localhost:8600"


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
    def __init__(self, base_url: str | None = None, token: str | None = None, timeout: float = 600.0) -> None:
        self.base_url = (base_url or os.getenv("LAKEHOUSE_API_URL", DEFAULT_URL)).rstrip("/")
        token = token or os.getenv("LAKEHOUSE_API_TOKEN", "")
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

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
        with self._http.stream("POST", "/v1/sql", json=body) as resp:
            self._raise_for(resp)
            yield pa.ipc.open_stream(io.BufferedReader(_ByteStream(resp.iter_bytes())))

    def to_parquet(self, query: str, path: str | Path, params: list[Any] | None = None) -> Path:
        """Write the result to one Parquet file at `path` without holding it in memory."""
        path = Path(path)
        body = {"sql": query, "params": params or [], "format": "parquet"}
        tmp = path.with_suffix(path.suffix + ".partial")
        with self._http.stream("POST", "/v1/sql", json=body) as resp:
            self._raise_for(resp)
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
        resp = self._http.post("/v1/ask", json={"question": question, "as_of": as_of, "include_evidence": include_evidence})
        self._raise_for(resp)
        return resp.json()

    def health(self) -> dict[str, Any]:
        return self._get("/v1/health")

    # -- helpers ----------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._http.get(path, params=params)
        self._raise_for(resp)
        return resp.json()

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
