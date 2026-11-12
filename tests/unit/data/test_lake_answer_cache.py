# test_lake_answer_cache.py

"""Inside `lake_answer_cache`, every `/v1/sql` read the loader makes in a `lake_session` is
answered from disk when the cache holds it and stored there otherwise. A stored answer converts
to the very frame a fresh read gives, dtypes and time zones included, empty answers too. An
answer is stored under the lake version of the response it came from, and looked up under the
session's version only, so no answer is served under another version. Entries are written
whole; a corrupt entry is discarded, logged and read again; a cut-off answer, an answer with no
version and a read with no session open are never stored.

The lake here is the real `Lake` client over a mock transport that answers the catalogue and
`/v1/sql` from tables the test sets."""

import decimal
import io
import json
import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pyarrow as pa
import pytest

from agentic_trading.data import lake_answer_cache as cache_module
from agentic_trading.data.lakehouse_client import Lake, LakeStreamCutError
from agentic_trading.data.loaders import lakehouse_loader as ll

_T = datetime(2026, 11, 6, tzinfo=UTC)
_QUERY = "SELECT * FROM fred_vintages_log WHERE series_id = ?"
_PARAMS: list[Any] = ["DGS3MO"]
_ROWS = pa.table(
    {
        "id": pa.array([1, None, 3], pa.int64()),
        "small": pa.array([1, 2, 3], pa.int32()),
        "value": pa.array([1.5, float("nan"), None], pa.float64()),
        "name": pa.array(["a", None, "c"], pa.string()),
        "flag": pa.array([True, None, False], pa.bool_()),
        "day": pa.array([date(2020, 1, 2), None, date(2020, 1, 6)], pa.date32()),
        "scraped_at": pa.array(
            [datetime(2020, 1, 2, 21, tzinfo=UTC), None, datetime(2020, 1, 6, 21, 30, tzinfo=UTC)],
            pa.timestamp("us", tz="UTC"),
        ),
        "open_time": pa.array([1, 2, None], pa.timestamp("ns")),
        "amount": pa.array(
            [decimal.Decimal("1.250"), None, decimal.Decimal("-3.000")], pa.decimal128(18, 3)
        ),
    }
)


def _arrow_body(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


class _CutStream(httpx.SyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __iter__(self) -> Iterator[bytes]:
        yield self._body[: len(self._body) // 2]
        raise httpx.RemoteProtocolError("peer closed connection without sending complete body")


class _MockLake:
    """The lake's HTTP API: `describe` of any table (a rebuilt-whole one when `builds` names
    it), and `/v1/sql` answering `answers[sql]` or, for a build read, the table's build.
    `version` is the header of every response (none when `None`), `sql_version` that of SQL
    answers when set. `refuse` makes every SQL read an HTTP 400 and `cut` cuts every SQL answer
    off part-way. `sql_sent` records each SQL text sent."""

    def __init__(self) -> None:
        self.version: str | None = "v1"
        self.sql_version: str | None = None
        self.answers: dict[str, pa.Table] = {_QUERY: _ROWS}
        self.builds: dict[str, datetime] = {}
        self.refuse = False
        self.cut = False
        self.sql_sent: list[str] = []

    def _headers(self, version: str | None) -> dict[str, str]:
        return {} if version is None else {"X-Lakehouse-Version": version}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/v1/tables/"):
            table = path.rsplit("/", 1)[1]
            described = {
                "kind": "current",
                "key_columns": [],
                "rebuilt_whole": table in self.builds,
                "last_write": "2020-01-01T00:00:00Z",
                "tie_order": [],
            }
            return httpx.Response(200, json=described, headers=self._headers(self.version))
        assert path == "/v1/sql"
        sql = json.loads(request.content)["sql"]
        self.sql_sent.append(sql)
        if self.refuse:
            return httpx.Response(400, json={"detail": "the lake refuses every read"})
        prefix = "SELECT max(scraped_at) AS build FROM "
        if sql.startswith(prefix):
            build = self.builds[sql.removeprefix(prefix)]
            table = pa.table({"build": pa.array([build], pa.timestamp("us", tz="UTC"))})
        else:
            table = self.answers[sql]
        headers = self._headers(self.sql_version or self.version)
        if self.cut:
            return httpx.Response(200, stream=_CutStream(_arrow_body(table)), headers=headers)
        return httpx.Response(200, content=_arrow_body(table), headers=headers)

    def queries_sent(self) -> list[str]:
        """The SQL reads sent, build reads aside."""
        return [sql for sql in self.sql_sent if not sql.startswith("SELECT max(scraped_at)")]


@pytest.fixture
def lake(monkeypatch: pytest.MonkeyPatch) -> _MockLake:
    """A `_MockLake` behind a real `Lake` client, as the loader's lake."""
    mock = _MockLake()
    client = Lake(
        base_url="http://lake.test",
        token="t",
        transport=httpx.MockTransport(mock.handler),
        sleep=lambda _: None,
    )
    monkeypatch.setattr(ll, "_lake", lambda: client)
    return mock


def _entries(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.parquet")) if directory.exists() else []


def _read(directory: Path | None, query: str = _QUERY) -> pd.DataFrame:
    """`_query` of `query` in a session at `_T`, through the cache at `directory` when given."""
    if directory is None:
        with ll.lake_session(_T):
            return ll._query(query, _PARAMS)
    with ll.lake_answer_cache(directory), ll.lake_session(_T):
        return ll._query(query, _PARAMS)


@pytest.mark.parametrize("rows", [_ROWS, _ROWS.slice(0, 0)], ids=["rows", "empty"])
def test_a_cached_answer_equals_a_fresh_one(
    lake: _MockLake, tmp_path: Path, rows: pa.Table
) -> None:
    """The frame of a read with no cache, of a read that stores the answer and of one served
    from the cache while the lake refuses every read are equal, dtypes and time zones
    included; the entry holds the Arrow table the lake sent."""
    lake.answers[_QUERY] = rows
    fresh = _read(None)
    stored = _read(tmp_path)
    lake.refuse = True
    cached = _read(tmp_path)

    for frame in (stored, cached):
        pd.testing.assert_frame_equal(frame, fresh, check_exact=True)
        assert list(frame.dtypes) == list(fresh.dtypes)
    assert str(cached["scraped_at"].dtype) == "datetime64[us, UTC]"
    assert len(lake.queries_sent()) == 2
    (entry,) = _entries(tmp_path)
    answers = cache_module.LakeAnswerCache(tmp_path)
    inputs = cache_module.answer_inputs("v1", _T, {}, _QUERY, _PARAMS)
    table = answers.get(inputs)
    assert table is not None
    assert table.schema.equals(rows.schema, check_metadata=True)
    assert table.to_pandas().equals(rows.to_pandas())
    assert entry.parent.name == cache_module.answer_key(inputs)[:2]


def test_an_answer_is_never_served_under_another_version(lake: _MockLake, tmp_path: Path) -> None:
    """An answer stored at version v1 is not served to a session at v2, which reads the lake
    and stores its own; a later session at v1 is served the v1 answer."""
    lake.answers[_QUERY] = pa.table({"value": [1.0]})
    _read(tmp_path)
    lake.version = "v2"
    lake.answers[_QUERY] = pa.table({"value": [2.0]})
    assert _read(tmp_path)["value"].tolist() == [2.0]
    lake.version = "v1"
    lake.refuse = True
    assert _read(tmp_path)["value"].tolist() == [1.0]

    assert len(lake.queries_sent()) == 2
    assert len(_entries(tmp_path)) == 2


def test_with_no_version_known_the_read_goes_to_the_lake_and_is_stored_under_its_version(
    lake: _MockLake, tmp_path: Path
) -> None:
    """When the catalogue responses carry no version the session opens without one: its first
    read goes to the lake even with the answer on disk, records the response's version, and
    is stored under it; the next read of the session is served from the cache."""
    lake.version = None
    lake.sql_version = "v1"
    with ll.lake_answer_cache(tmp_path), ll.lake_session(_T) as session:
        assert session.lake_version is None
        ll._query(_QUERY, _PARAMS)
        assert session.lake_version == "v1"
        ll._query(_QUERY, _PARAMS)
    with ll.lake_answer_cache(tmp_path), ll.lake_session(_T):
        ll._query(_QUERY, _PARAMS)

    assert len(lake.queries_sent()) == 2
    (entry,) = _entries(tmp_path)
    assert entry.stem == cache_module.answer_key(
        cache_module.answer_inputs("v1", _T, {}, _QUERY, _PARAMS)
    )


def test_a_response_of_another_version_raises_and_stores_nothing(
    lake: _MockLake, tmp_path: Path
) -> None:
    """A SQL answer whose version differs from the session's raises `LakeTimeError` before
    it is stored."""
    lake.sql_version = "v2"
    with pytest.raises(ll.LakeTimeError, match="version changed from v1 to v2"):
        _read(tmp_path)
    assert _entries(tmp_path) == []


def test_an_answer_without_a_version_is_never_stored(lake: _MockLake, tmp_path: Path) -> None:
    lake.version = None
    with ll.lake_answer_cache(tmp_path) as answers, ll.lake_session(_T):
        ll._query(_QUERY, _PARAMS)
        assert answers.counts().unstored == 1
    assert _entries(tmp_path) == []


def test_a_corrupt_entry_is_discarded_logged_and_read_again(
    lake: _MockLake, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An entry that is not a whole Parquet file is removed and logged, the answer is read from
    the lake again and stored whole."""
    expected = _read(tmp_path)
    (entry,) = _entries(tmp_path)
    entry.write_bytes(entry.read_bytes()[:100])
    with caplog.at_level(logging.WARNING, logger=cache_module.__name__):
        with ll.lake_answer_cache(tmp_path) as answers, ll.lake_session(_T):
            frame = ll._query(_QUERY, _PARAMS)
            counts = answers.counts()

    pd.testing.assert_frame_equal(frame, expected)
    assert (counts.discarded, counts.stored, counts.hits) == (1, 1, 0)
    assert any("Discarded" in record.getMessage() for record in caplog.records)
    assert len(lake.queries_sent()) == 2
    lake.refuse = True
    pd.testing.assert_frame_equal(_read(tmp_path), expected)


def test_an_entry_recording_other_inputs_is_discarded(lake: _MockLake, tmp_path: Path) -> None:
    """A file found under a key that its recorded inputs do not give is never served."""
    other = "SELECT * FROM fred_vintages_log WHERE series_id = ? AND 1 = 1"
    lake.answers[other] = pa.table({"value": [9.0]})
    _read(tmp_path, other)
    (entry,) = _entries(tmp_path)
    key = cache_module.answer_key(cache_module.answer_inputs("v1", _T, {}, _QUERY, _PARAMS))
    target = tmp_path / key[:2] / f"{key}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(entry.read_bytes())

    pd.testing.assert_frame_equal(_read(tmp_path), _ROWS.to_pandas(date_as_object=False))
    assert lake.queries_sent().count(_QUERY) == 1


def test_a_write_interrupted_part_way_leaves_no_entry(
    lake: _MockLake, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An answer whose write stops part-way (half a file written, then an interrupt) leaves
    nothing under its key and no partial file; the next read goes to the lake and stores the
    answer whole."""

    def interrupted(table: pa.Table, where: Any, **kwargs: Any) -> None:
        Path(where).write_bytes(b"PAR1 half a file")
        raise KeyboardInterrupt

    with monkeypatch.context() as patched:
        patched.setattr(cache_module.pq, "write_table", interrupted)
        with pytest.raises(KeyboardInterrupt):
            _read(tmp_path)
    assert len(lake.queries_sent()) == 1
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []
    _read(tmp_path)
    assert len(lake.queries_sent()) == 2
    assert len(_entries(tmp_path)) == 1


def test_a_cut_answer_is_never_stored(lake: _MockLake, tmp_path: Path) -> None:
    """An answer cut off part-way on every send raises and leaves the cache empty."""
    lake.cut = True
    with pytest.raises(LakeStreamCutError):
        _read(tmp_path)
    assert _entries(tmp_path) == []


def test_a_read_with_no_session_open_is_never_cached(lake: _MockLake, tmp_path: Path) -> None:
    with ll.lake_answer_cache(tmp_path), ll.unpinned_lake_reads("test"):
        ll._query(_QUERY, _PARAMS)
        ll._query(_QUERY, _PARAMS)
    assert len(lake.queries_sent()) == 2
    assert not tmp_path.joinpath("lake_cache").exists() and _entries(tmp_path) == []


def test_an_empty_answer_naming_an_always_populated_table_is_not_stored(
    lake: _MockLake, tmp_path: Path
) -> None:
    """An empty answer to a query naming a table of `ALWAYS_POPULATED_TABLES` or its log may be
    a read that landed on a rebuild, so it is not stored; one naming another table is."""
    always = [
        "SELECT 1 AS present FROM sp500_membership LIMIT 1",
        "SELECT 1 AS present FROM sp500_membership_log WHERE scraped_at <= ? LIMIT 1",
    ]
    for query in always:
        lake.answers[query] = pa.table({"present": pa.array([], pa.int32())})
    lake.answers[_QUERY] = _ROWS.slice(0, 0)
    with ll.lake_answer_cache(tmp_path) as answers, ll.lake_session(_T):
        for query in always:
            ll._query(query, [])
        ll._query(_QUERY, _PARAMS)
        assert (answers.counts().stored, answers.counts().unstored) == (1, 2)
    assert len(_entries(tmp_path)) == 1


def test_a_rebuilt_whole_tables_build_is_part_of_its_reads_key(
    lake: _MockLake, tmp_path: Path
) -> None:
    """A read naming a rebuilt-whole table, which the session reads unpinned, is stored under
    the build the session recorded: after a rebuild it is read from the lake again, while a
    read naming no such table is still served from the cache."""
    whole = "SELECT * FROM sp500_member_count"
    lake.answers[whole] = pa.table({"n": [500]})
    lake.builds["sp500_member_count"] = datetime(2026, 1, 1, tzinfo=UTC)
    with ll.lake_answer_cache(tmp_path), ll.lake_session(_T):
        ll._query(whole, [])
        ll._query(_QUERY, _PARAMS)
    lake.builds["sp500_member_count"] = datetime(2026, 2, 1, tzinfo=UTC)
    with ll.lake_answer_cache(tmp_path), ll.lake_session(_T):
        ll._query(whole, [])
        ll._query(_QUERY, _PARAMS)

    assert lake.queries_sent() == [whole, _QUERY, whole]


def test_a_cache_cannot_be_opened_inside_another(tmp_path: Path) -> None:
    with ll.lake_answer_cache(tmp_path / "a"):
        with pytest.raises(RuntimeError, match="open already"):
            with ll.lake_answer_cache(tmp_path / "b"):
                pass
    assert ll.lake_answer_cache_counts() is None
