# test_holdout_end.py

"""AT-110 precompute spec item 2: `lake_session(data_end=...)` is a hard cut on knowledge time.
Every public loader raises `HoldoutReadError` before it queries when its knowledge-time argument
(`as_of`/`as_of_date`, or `end` for bar-dated ranges) is later than the session's `data_end`; at
`data_end`, and with no `data_end`, loaders read as they do without the cut."""

import inspect
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from agentic_trading.data.loader import load_price_history_with_fallback
from agentic_trading.data.loaders import lakehouse_loader as ll

_T = datetime(2026, 10, 29, tzinfo=UTC)
_DATA_END = date(2024, 12, 31)
_PAST_END = _DATA_END + timedelta(days=1)
_KNOWLEDGE_TIME = frozenset({"as_of", "as_of_date", "end", "end_date"})

# Public functions that read the lake but take no knowledge-time argument, so the guard has no
# date to compare with data_end.
_NO_KNOWLEDGE_TIME = frozenset(
    {
        # One current SIC classification per ticker with no date column to gate on: the read is
        # bounded only by the lake time T, the documented look-ahead limit of SIC screening.
        "load_sic_code_from_lakehouse",
        # Reads only through load_sic_code_from_lakehouse.
        "same_sic_candidate_pairs",
        # `quote_date` is the date of a chain row from an option loader already guarded on its
        # as_of_date. The factor reads split ratios with later ex-dates, up to T, only to put
        # that row on the split-adjusted close basis; it carries no signal value of its own.
        "split_adjustment_factor",
        # A table's row count, recorded as provenance; no row values are returned.
        "adjustment_table_row_count",
        # First and last date per source for the preflight; it returns coverage dates, not data.
        "source_coverage",
        # The lake version and status document; no market data.
        "lake_provenance",
        # Re-reads the builds and last writes of tables the session read; no market data.
        "verify_rebuilt_whole_builds",
    }
)
_DUMMY_BY_NAME: dict[str, Any] = {"option_type": "call", "series_id": "DGS10"}
_DUMMY_BY_TYPE: dict[Any, Any] = {
    date: _DATA_END,
    str: "AAPL",
    int: 10,
    float: 100.0,
    list[str]: ["AAPL"],
    list[int]: [1],
}


def _public_functions() -> dict[str, Callable[..., Any]]:
    """Every public callable defined in lakehouse_loader itself, classes excluded."""
    return {
        name: value
        for name, value in vars(ll).items()
        if not name.startswith("_")
        and callable(value)
        and not inspect.isclass(value)
        and getattr(value, "__module__", None) == ll.__name__
    }


_GUARDED = sorted(
    name
    for name, function in _public_functions().items()
    if _KNOWLEDGE_TIME & set(inspect.signature(function).parameters)
)


def _arguments(function: Callable[..., Any]) -> dict[str, Any]:
    """Each knowledge-time parameter set to the day after data_end, every other required one to
    a minimal dummy of its name or annotation."""
    arguments: dict[str, Any] = {}
    for name, parameter in inspect.signature(function, eval_str=True).parameters.items():
        if name in _KNOWLEDGE_TIME:
            arguments[name] = _PAST_END
        elif parameter.default is not inspect.Parameter.empty:
            continue
        elif name in _DUMMY_BY_NAME:
            arguments[name] = _DUMMY_BY_NAME[name]
        elif parameter.annotation in _DUMMY_BY_TYPE:
            arguments[name] = _DUMMY_BY_TYPE[parameter.annotation]
        else:
            pytest.fail(f"no dummy for {function.__name__}({name}: {parameter.annotation})")
    return arguments


def test_lake_session_data_end_is_exposed_and_joins_must_agree(fake_lake: Any) -> None:
    """The session carries data_end; a join without one inherits it, a different one raises."""
    with ll.lake_session(_T, data_end=_DATA_END) as session:
        assert session.data_end == _DATA_END
        with ll.lake_session() as joined:
            assert joined.data_end == _DATA_END
        with ll.lake_session(_T, data_end=_DATA_END) as same:
            assert same.data_end == _DATA_END
        for other in (_PAST_END, _DATA_END - timedelta(days=1)):
            with pytest.raises(ll.LakeTimeError), ll.lake_session(data_end=other):
                pass
        assert ll.current_lake_time() == _T
    with ll.lake_session(_T) as open_ended:
        assert open_ended.data_end is None
        with pytest.raises(ll.LakeTimeError), ll.lake_session(data_end=_DATA_END):
            pass
    assert ll.current_lake_time() is None


def test_guarded_loader_enumeration_is_complete() -> None:
    """At least 10 loaders take a knowledge-time argument; every other load_* is allow-listed."""
    public = set(_public_functions())
    loaders = {name for name in public if name.startswith("load_")}

    assert len(_GUARDED) >= 10
    assert loaders - set(_GUARDED) <= _NO_KNOWLEDGE_TIME
    assert _NO_KNOWLEDGE_TIME <= public - set(_GUARDED)


@pytest.mark.parametrize("name", _GUARDED)
def test_loader_past_data_end_raises_before_querying(name: str, fake_lake: Any) -> None:
    """A knowledge-time argument one day after data_end raises HoldoutReadError, no query sent."""
    loader = getattr(ll, name)
    fake_lake.error = AssertionError(f"{name} queried the lake past data_end")

    with ll.lake_session(_T, data_end=_DATA_END), pytest.raises(ll.HoldoutReadError):
        loader(**_arguments(loader))

    assert fake_lake.calls == []


def test_reads_at_data_end_pass_the_guard(preflight_lake: Any) -> None:
    """A knowledge-time argument equal to data_end is queried; the empty tables give None."""
    reads = {
        "fred": lambda: ll.load_fred_series_from_lakehouse("DGS10", _DATA_END),
        "dividends": lambda: ll.load_dividends_from_lakehouse("AAPL", _DATA_END),
        "prices": lambda: ll.load_price_history_from_lakehouse(
            ["AAPL"], date(2024, 12, 2), _DATA_END
        ),
    }

    with ll.lake_session(_T, data_end=_DATA_END):
        for label, read in reads.items():
            before = len(preflight_lake.calls)
            assert read() is None, label
            assert len(preflight_lake.calls) > before, label


def test_no_data_end_means_no_guard(preflight_lake: Any) -> None:
    """With no session, or a session whose data_end is None, a read of any as_of date is made."""
    late = date(2030, 1, 2)

    def read() -> Any:
        return ll.load_fred_series_from_lakehouse("DGS10", late)

    assert read() is None
    with ll.lake_session(_T) as session:
        assert session.data_end is None
        assert read() is None
    with ll.lake_session(_T, data_end=None) as session:
        assert session.data_end is None
        assert read() is None
    assert len(preflight_lake.calls) >= 3


def test_price_history_with_fallback_past_data_end_raises(fake_lake: Any) -> None:
    """The CLI's price entry point raises HoldoutReadError for an end after data_end, unqueried."""
    fake_lake.error = AssertionError("queried the lake past data_end")

    with ll.lake_session(_T, data_end=_DATA_END), pytest.raises(ll.HoldoutReadError):
        load_price_history_with_fallback(["AAPL"], date(2024, 12, 2), _PAST_END)

    assert fake_lake.calls == []
