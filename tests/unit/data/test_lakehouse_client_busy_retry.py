# test_lakehouse_client_busy_retry.py

"""`Lake` sends a request again when the lake answers that its query slots are busy (HTTP 503
with "query slots are busy"), after each of its busy retry delays, never waiting less than the
answer's `Retry-After`, and raises `LakehouseError` when the last send is still busy. Every other
refusal raises at once. Each retry is logged as a warning."""

import io
import logging

import httpx
import pyarrow as pa
import pytest

from agentic_trading.data import lakehouse_client
from agentic_trading.data.lakehouse_client import BUSY_RETRY_JITTER, Lake, LakehouseError

_BUSY = {"detail": "all 2 query slots are busy; retry shortly"}
_DELAYS = (10.0, 20.0, 30.0)


def _arrow_body() -> bytes:
    """An Arrow IPC stream of one column `one` holding 1."""
    sink = io.BytesIO()
    table = pa.table({"one": [1]})
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def _lake(answers: list[httpx.Response], sent: list[httpx.Request], waits: list[float]) -> Lake:
    """A `Lake` whose transport gives `answers` in order, recording each request in `sent`
    and each wait in `waits`, with three busy retry delays."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return answers[len(sent) - 1]

    return Lake(
        base_url="http://lake.test",
        token="t",
        transport=httpx.MockTransport(handler),
        busy_retry_delays=_DELAYS,
        sleep=waits.append,
    )


def _busy(retry_after: str | None = "10") -> httpx.Response:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.Response(503, json=_BUSY, headers=headers)


def _ok_sql() -> httpx.Response:
    return httpx.Response(200, content=_arrow_body(), headers={"X-Lakehouse-Version": "v9"})


def test_busy_then_ok_returns_the_rows_and_logs_each_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two busy answers, then rows: the rows come back after two waits, each within the jitter
    of its delay and never below Retry-After, and each retry is logged."""
    sent: list[httpx.Request] = []
    waits: list[float] = []
    lake = _lake([_busy(), _busy(), _ok_sql()], sent, waits)
    with caplog.at_level(logging.WARNING, logger=lakehouse_client.__name__):
        frame = lake.sql("SELECT 1 AS one")
    assert frame["one"].tolist() == [1]
    assert len(sent) == 3
    assert len(waits) == 2
    for wait, delay in zip(waits, _DELAYS, strict=False):
        assert max(delay * (1 - BUSY_RETRY_JITTER), 10.0) <= wait <= delay * (1 + BUSY_RETRY_JITTER)
    retries = [r for r in caplog.records if "query slots busy" in r.getMessage()]
    assert len(retries) == 2
    assert lake.last_version == "v9"


def test_busy_past_the_last_delay_raises() -> None:
    """Busy on every send: one send per delay plus the first, then `LakehouseError` with the
    lake's message."""
    sent: list[httpx.Request] = []
    waits: list[float] = []
    lake = _lake([_busy() for _ in range(len(_DELAYS) + 1)], sent, waits)
    with pytest.raises(LakehouseError, match="HTTP 503: all 2 query slots are busy"):
        lake.sql("SELECT 1 AS one")
    assert len(sent) == len(_DELAYS) + 1
    assert len(waits) == len(_DELAYS)


def test_retry_after_longer_than_the_delay_is_honored() -> None:
    """A `Retry-After` above the jittered delay sets the wait."""
    sent: list[httpx.Request] = []
    waits: list[float] = []
    lake = _lake([_busy("45"), _ok_sql()], sent, waits)
    lake.sql("SELECT 1 AS one")
    assert waits == [45.0]


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(503, json={"detail": "the lake is restarting"}),
        httpx.Response(500, json={"detail": "query failed"}),
        httpx.Response(400, json={"detail": "bad SQL"}),
        httpx.Response(409, json={"detail": "pin cannot be served"}),
    ],
)
def test_any_other_refusal_raises_at_once(answer: httpx.Response) -> None:
    """A refusal that is not the busy-slots answer is sent once and raises."""
    sent: list[httpx.Request] = []
    waits: list[float] = []
    lake = _lake([answer], sent, waits)
    with pytest.raises(LakehouseError, match=f"HTTP {answer.status_code}"):
        lake.sql("SELECT 1 AS one")
    assert len(sent) == 1
    assert waits == []


def test_describe_retries_a_busy_answer() -> None:
    """The JSON endpoints share the retry: `describe` returns the table after one busy answer."""
    sent: list[httpx.Request] = []
    waits: list[float] = []
    lake = _lake([_busy(), httpx.Response(200, json={"kind": "current"})], sent, waits)
    assert lake.describe("instrument_prices") == {"kind": "current"}
    assert len(sent) == 2
    assert sent[0].url.path == "/v1/tables/instrument_prices"


def test_the_default_delays_span_the_lakes_abandoned_slot_deadline() -> None:
    """With the lake's own 30 s wait for a slot on each busy send, the default delays at their
    shortest jitter still span more than the lake's 300 s deadline for an abandoned request."""
    delays = lakehouse_client.BUSY_RETRY_DELAYS
    shortest = sum(delays) * (1 - BUSY_RETRY_JITTER) + 30.0 * (len(delays) + 1)
    assert shortest > 300.0 + 60.0


def test_in_flight_counts_an_open_response_and_returns_to_zero() -> None:
    """`in_flight` is 1 while a streamed result is being read and 0 after it, after a busy
    retry and after a refusal alike."""
    sent: list[httpx.Request] = []
    waits: list[float] = []
    lake = _lake([_busy(), _ok_sql(), httpx.Response(500, json={"detail": "x"})], sent, waits)
    seen = []
    for batch in lake.sql_batches("SELECT 1 AS one"):
        seen.append((batch.num_rows, lake.in_flight))
    assert seen == [(1, 1)]
    assert lake.in_flight == 0
    with pytest.raises(LakehouseError):
        lake.sql("SELECT 1 AS one")
    assert lake.in_flight == 0


def test_the_process_count_covers_every_client() -> None:
    """`Lake.in_flight_in_process()` counts an open response of any client."""
    before = Lake.in_flight_in_process()
    lake = _lake([_ok_sql()], [], [])
    for _ in lake.sql_batches("SELECT 1 AS one"):
        assert Lake.in_flight_in_process() == before + 1
    assert Lake.in_flight_in_process() == before
