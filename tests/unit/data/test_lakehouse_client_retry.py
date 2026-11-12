# test_lakehouse_client_retry.py

"""`Lake` sends a call again after a failure the lake recovers from, with a jittered
exponential backoff that honours `Retry-After`, until `settings.lake.retry_budget_seconds` has
passed: an HTTP 502/503/504 and an httpx transport error before any response (connection
refused or reset, a read error, a timeout, which also waits out the quiet period) within the
budget; an HTTP 500 (or another 5xx) and a response cut off part-way at most `LIMITED_RETRIES`
times; every 4xx never. Each retry is logged as a warning naming the attempt, the error and the
wait. At most `settings.lake.max_concurrent_queries` `/v1/sql` requests of the process are in
flight. Every clock here is injected; nothing sleeps."""

import io
import logging
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta

import httpx
import pyarrow as pa
import pytest

from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data import lakehouse_client
from agentic_trading.data.lakehouse_client import (
    LIMITED_RETRIES,
    RETRY_FIRST_WAIT_SECONDS,
    RETRY_JITTER,
    Lake,
    LakeConnectionError,
    LakehouseError,
    LakeStreamCutError,
)

_T0 = datetime(2026, 11, 10, 12, 0, 0, tzinfo=UTC)


class _Clock:
    """A UTC clock that `sleep` moves on, recording each wait."""

    def __init__(self) -> None:
        self.now = _T0
        self.waits: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += timedelta(seconds=seconds)


def _arrow_body(table: pa.Table | None = None) -> bytes:
    """An Arrow IPC stream of `table` (one column `one` holding 1 by default)."""
    table = pa.table({"one": [1]}) if table is None else table
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def _ok() -> httpx.Response:
    return httpx.Response(200, content=_arrow_body(), headers={"X-Lakehouse-Version": "v9"})


Answer = httpx.Response | Exception | Callable[[httpx.Request], httpx.Response]


def _lake(answers: list[Answer], sent: list[httpx.Request], clock: _Clock) -> Lake:
    """A `Lake` whose transport gives `answers` in order (an exception is raised, a callable is
    called with the request), recording each request in `sent`."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        answer = answers[len(sent) - 1]
        if isinstance(answer, Exception):
            raise answer
        if callable(answer) and not isinstance(answer, httpx.Response):
            return answer(request)
        return answer

    return Lake(
        base_url="http://lake.test",
        token="t",
        transport=httpx.MockTransport(handler),
        sleep=clock.sleep,
        clock=clock,
    )


def _retries(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "sending again" in r.getMessage()]


class _CutStream(httpx.SyncByteStream):
    """The first half of an Arrow body, then the connection lost."""

    def __iter__(self) -> Iterator[bytes]:
        body = _arrow_body()
        yield body[: len(body) // 2]
        raise httpx.RemoteProtocolError("peer closed connection without sending complete body")


def _cut(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, stream=_CutStream())


def test_a_503_then_rows_waits_at_least_retry_after_and_logs_the_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A busy answer, then rows: one wait, never below Retry-After, logged with the attempt,
    the error and the wait."""
    sent: list[httpx.Request] = []
    clock = _Clock()
    busy = httpx.Response(
        503, json={"detail": "query slots are busy"}, headers={"Retry-After": "10"}
    )
    lake = _lake([busy, _ok()], sent, clock)
    with caplog.at_level(logging.WARNING, logger=lakehouse_client.__name__):
        assert lake.sql("SELECT 1 AS one")["one"].tolist() == [1]

    assert len(sent) == 2
    assert clock.waits == [pytest.approx(10.0)]
    (message,) = _retries(caplog)
    assert "send 1" in message and "HTTP 503" in message and "10.0 s" in message
    assert lake.last_version == "v9"


@pytest.mark.parametrize(
    "refusal",
    [
        httpx.Response(503, json={"detail": "the lake is restarting"}),
        httpx.Response(502, text="bad gateway"),
        httpx.Response(504, text="gateway timeout"),
    ],
)
def test_an_unavailable_answer_of_any_detail_is_sent_again(refusal: httpx.Response) -> None:
    """A 503 of any detail, a 502 and a 504 are sent again after a backoff wait."""
    sent: list[httpx.Request] = []
    clock = _Clock()
    lake = _lake([refusal, _ok()], sent, clock)
    assert lake.sql("SELECT 1 AS one")["one"].tolist() == [1]
    assert len(sent) == 2
    first = RETRY_FIRST_WAIT_SECONDS
    assert first * (1 - RETRY_JITTER) <= clock.waits[0] <= first * (1 + RETRY_JITTER)


def test_connection_refused_three_times_then_rows_backs_off_exponentially(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Three refused connections (the API restarting), then rows: three waits, each about
    double the one before, each logged."""
    sent: list[httpx.Request] = []
    clock = _Clock()
    refused = [httpx.ConnectError("Connection refused") for _ in range(3)]
    lake = _lake([*refused, _ok()], sent, clock)
    with caplog.at_level(logging.WARNING, logger=lakehouse_client.__name__):
        assert lake.sql("SELECT 1 AS one")["one"].tolist() == [1]

    assert len(sent) == 4
    for attempt, wait in enumerate(clock.waits):
        nominal = RETRY_FIRST_WAIT_SECONDS * 2**attempt
        assert nominal * (1 - RETRY_JITTER) <= wait <= nominal * (1 + RETRY_JITTER)
    messages = _retries(caplog)
    assert len(messages) == 3
    assert all("ConnectError" in message for message in messages)
    assert [f"send {n}" in message for n, message in enumerate(messages, start=1)] == [True] * 3


@pytest.mark.parametrize(
    "error",
    [
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
        httpx.ReadError("Connection reset by peer"),
    ],
)
def test_a_transport_error_before_any_response_is_sent_again(error: Exception) -> None:
    """A connection reset or a protocol error before the response arrives is sent again."""
    sent: list[httpx.Request] = []
    lake = _lake([error, _ok()], sent, _Clock())
    assert lake.sql("SELECT 1 AS one")["one"].tolist() == [1]
    assert len(sent) == 2


def test_a_timeout_then_rows_waits_out_the_quiet_period() -> None:
    """A timed-out request is sent again only after the quiet period, then returns its rows."""
    sent: list[httpx.Request] = []
    clock = _Clock()
    lake = _lake([httpx.ReadTimeout("timed out"), _ok()], sent, clock)
    assert lake.sql("SELECT 1 AS one")["one"].tolist() == [1]
    assert len(sent) == 2
    assert clock.waits == [pytest.approx(settings.lake.request_quiet_seconds)]


def test_the_budget_spent_raises_the_last_error(caplog: pytest.LogCaptureFixture) -> None:
    """A lake refusing connections on every send: the call sends again until the next wait
    would end past the budget, then raises `LakeConnectionError` from the error of its last
    send, and logs that the budget is spent."""
    sent: list[httpx.Request] = []
    clock = _Clock()
    refused = [httpx.ConnectError(f"Connection refused {n}") for n in range(100)]
    lake = _lake(refused, sent, clock)
    with temporary_overrides({"lake.retry_budget_seconds": 120.0}):
        with caplog.at_level(logging.WARNING, logger=lakehouse_client.__name__):
            with pytest.raises(LakeConnectionError) as raised:
                lake.sql("SELECT 1 AS one")

    assert isinstance(raised.value.__cause__, httpx.ConnectError)
    assert str(raised.value.__cause__) == f"Connection refused {len(sent) - 1}"
    assert f"Connection refused {len(sent) - 1}" in str(raised.value)
    assert 3 <= len(sent) < 100
    assert sum(clock.waits) <= 120.0
    assert any("budget" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    ("status", "detail"),
    [
        (400, "Binder Error: column x does not exist"),
        (400, "the 300 s query deadline passed before any batch"),
        (401, "unauthorized"),
        (409, "the pin precedes the build"),
        (423, "reads are frozen"),
    ],
)
def test_a_4xx_raises_at_once(status: int, detail: str) -> None:
    """A 4xx is sent once and raises with the lake's detail; nothing waits."""
    sent: list[httpx.Request] = []
    clock = _Clock()
    lake = _lake([httpx.Response(status, json={"detail": detail})], sent, clock)
    with pytest.raises(LakehouseError, match=f"HTTP {status}: ") as raised:
        lake.sql("SELECT 1 AS one")
    assert raised.value.status == status
    assert len(sent) == 1
    assert clock.waits == []


def test_a_500_is_sent_again_at_most_limited_retries_times() -> None:
    """An HTTP 500 on every send: `LIMITED_RETRIES` more sends, then the 500 raises."""
    sent: list[httpx.Request] = []
    failed = [httpx.Response(500, json={"detail": "query failed"}) for _ in range(10)]
    lake = _lake(failed, sent, _Clock())
    with pytest.raises(LakehouseError, match="HTTP 500: query failed"):
        lake.sql("SELECT 1 AS one")
    assert len(sent) == 1 + LIMITED_RETRIES


def test_a_500_then_rows_returns_the_rows() -> None:
    sent: list[httpx.Request] = []
    lake = _lake([httpx.Response(500, json={"detail": "x"}), _ok()], sent, _Clock())
    assert lake.sql("SELECT 1 AS one")["one"].tolist() == [1]


def test_a_cut_stream_is_sent_again_at_most_limited_retries_times() -> None:
    """A body cut off part-way on every send raises `LakeStreamCutError` after
    `LIMITED_RETRIES` more sends; once whole, the rows come back."""
    sent: list[httpx.Request] = []
    lake = _lake([_cut] * 10, sent, _Clock())
    with pytest.raises(LakeStreamCutError, match="cut off"):
        lake.sql_answer("SELECT 1 AS one")
    assert len(sent) == 1 + LIMITED_RETRIES

    sent.clear()
    lake = _lake([_cut, _ok()], sent, _Clock())
    answer = lake.sql_answer("SELECT 1 AS one")
    assert answer.table.to_pydict() == {"one": [1]}
    assert answer.version == "v9"


def test_json_endpoints_share_the_retry() -> None:
    """`describe` returns the table after a refused connection and a busy answer."""
    sent: list[httpx.Request] = []
    answers: list[Answer] = [
        httpx.ConnectError("Connection refused"),
        httpx.Response(503, json={"detail": "busy"}),
        httpx.Response(200, json={"kind": "current"}),
    ]
    lake = _lake(answers, sent, _Clock())
    assert lake.describe("instrument_prices") == {"kind": "current"}
    assert [request.url.path for request in sent] == ["/v1/tables/instrument_prices"] * 3


def test_a_third_concurrent_query_waits_for_a_slot() -> None:
    """With two `/v1/sql` requests held open by the transport, a third call waits unsent until
    one of them is done, then is sent."""
    release = threading.Event()
    lock = threading.Lock()
    sent: list[httpx.Request] = []
    peak = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            sent.append(request)
            peak[0] = max(peak[0], Lake.queries_in_flight_in_process())
        release.wait(timeout=10)
        return _ok()

    lake = Lake(base_url="http://lake.test", token="t", transport=httpx.MockTransport(handler))
    results: list[list[int]] = []

    def call() -> None:
        results.append(lake.sql("SELECT 1 AS one")["one"].tolist())

    assert settings.lake.max_concurrent_queries == 2
    workers = [threading.Thread(target=call) for _ in range(3)]
    for worker in workers:
        worker.start()
    deadline = datetime.now(UTC) + timedelta(seconds=10)
    while len(sent) < 2 and datetime.now(UTC) < deadline:
        threading.Event().wait(0.01)
    threading.Event().wait(0.2)
    assert len(sent) == 2
    assert Lake.queries_in_flight_in_process() == 2
    release.set()
    for worker in workers:
        worker.join(timeout=10)

    assert results == [[1]] * 3
    assert len(sent) == 3
    assert peak[0] == 2
    assert Lake.queries_in_flight_in_process() == 0


def test_in_flight_counts_an_open_response_and_returns_to_zero() -> None:
    """`in_flight` is 1 while a streamed result is being read and 0 after it, after a retry
    and after a refusal alike."""
    sent: list[httpx.Request] = []
    answers: list[Answer] = [
        httpx.Response(503, json={"detail": "busy"}),
        _ok(),
        httpx.Response(400, json={"detail": "x"}),
    ]
    lake = _lake(answers, sent, _Clock())
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
    lake = _lake([_ok()], [], _Clock())
    for _ in lake.sql_batches("SELECT 1 AS one"):
        assert Lake.in_flight_in_process() == before + 1
    assert Lake.in_flight_in_process() == before
