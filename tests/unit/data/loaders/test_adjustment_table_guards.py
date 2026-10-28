# test_adjustment_table_guards.py

"""Empty adjustment tables raise, unfactored split-sized moves are flagged, and row counts are
recorded in the lake provenance."""

import logging
import math
from datetime import date
from typing import Any

import pandas as pd
import pytest

from agentic_trading.data.loaders import lakehouse_loader
from agentic_trading.data.loaders.lakehouse_loader import (
    EmptyAdjustmentTableError,
    adjustment_table_row_count,
    load_security_log_returns,
    split_adjustment_factor,
    unfactored_split_moves,
)

_BARS = {
    "ticker": "VARCHAR",
    "date": "DATE",
    "close": "DOUBLE",
    "official_close": "DOUBLE",
    "security_id": "BIGINT",
}
_FACTORS = {
    "ticker": "VARCHAR",
    "effective_date": "DATE",
    "split_factor": "DOUBLE",
    "cum_factor": "DOUBLE",
    "security_id": "BIGINT",
}
_SPLITS = {"ticker": "VARCHAR", "ex_date": "DATE", "ratio": "DOUBLE", "in_close_basis": "BOOLEAN"}

_MON, _TUE = date(2024, 3, 4), date(2024, 3, 5)


def _seed_bars(lake: Any) -> None:
    lake.add_empty("equity_daily_from_bars", _BARS)
    for day, close in ((_MON, 100.0), (_TUE, 101.0)):
        lake._connection.execute(
            "INSERT INTO equity_daily_from_bars VALUES ('AAA', ?, ?, NULL, 1)", [day, close]
        )


def _count_queries(lake: Any, table: str) -> list[str]:
    return [q for q, _ in lake.calls if "count(" in q.lower() and table in q]


def test_empty_factors_table_raises_from_load_security_log_returns(fake_lake: Any) -> None:
    """A read against an `equity_adjustment_factors` table with no rows raises."""
    _seed_bars(fake_lake)
    fake_lake.add_empty("equity_adjustment_factors", _FACTORS)

    with pytest.raises(EmptyAdjustmentTableError):
        load_security_log_returns([1], _MON, _TUE)
    assert issubclass(EmptyAdjustmentTableError, RuntimeError)


def test_empty_close_basis_table_raises_from_split_adjustment_factor(fake_lake: Any) -> None:
    """A read against a `yahoo_split_close_basis` table with no rows raises."""
    fake_lake.add_empty("yahoo_split_close_basis", _SPLITS)

    with pytest.raises(EmptyAdjustmentTableError):
        split_adjustment_factor("AAPL", date(2024, 1, 2))


def test_security_without_factor_rows_is_unadjusted_when_table_is_non_empty(
    fake_lake: Any,
) -> None:
    """A security or ticker with no factor rows keeps raw returns and factor 1.0."""
    _seed_bars(fake_lake)
    fake_lake.add(
        "equity_adjustment_factors",
        pd.DataFrame(
            {
                "ticker": ["BBB"],
                "effective_date": ["2024-03-05"],
                "split_factor": [2.0],
                "cum_factor": [0.5],
                "security_id": [2],
            }
        ),
        date_columns=("effective_date",),
    )
    fake_lake.add(
        "yahoo_split_close_basis",
        pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "ex_date": ["2024-06-01"],
                "ratio": [0.5],
                "in_close_basis": [True],
            }
        ),
        date_columns=("ex_date",),
    )

    series = load_security_log_returns([1], _MON, _TUE)[1]

    assert series.iloc[0] == pytest.approx(math.log(101 / 100))
    assert split_adjustment_factor("AAPL", date(2024, 1, 2)) == 1.0


def test_row_count_is_returned_and_cached_per_lake_client(fake_lake: Any) -> None:
    """The count is the table's row total and two calls issue one count query."""
    fake_lake.add_empty("equity_adjustment_factors", _FACTORS)
    for security_id in (1, 2, 3):
        fake_lake._connection.execute(
            "INSERT INTO equity_adjustment_factors VALUES ('X', ?, 2.0, 0.5, ?)",
            [_TUE, security_id],
        )

    first = adjustment_table_row_count("equity_adjustment_factors")
    second = adjustment_table_row_count("equity_adjustment_factors")

    assert first == second == 3
    assert len(_count_queries(fake_lake, "equity_adjustment_factors")) == 1


def test_unfactored_split_moves_flags_only_split_sized_moves_without_a_factor_date() -> None:
    """A 2:1 drop is flagged unless its date has a factor row; a 3% move is never flagged."""
    index = pd.DatetimeIndex(["2024-03-04", "2024-03-05", "2024-03-06"])
    returns = {1: pd.Series([0.01, math.log(0.5), 0.03], index=index)}

    flagged = unfactored_split_moves(returns, {})
    covered = unfactored_split_moves(returns, {1: {date(2024, 3, 5)}})
    other_security = unfactored_split_moves(returns, {2: {date(2024, 3, 5)}})

    assert len(flagged) == 1
    security_id, day, ratio = flagged[0]
    assert (security_id, day) == (1, date(2024, 3, 5))
    assert min(ratio, 1 / ratio) == pytest.approx(0.5)
    assert covered == []
    assert other_security == flagged
    assert unfactored_split_moves({}, {}) == []


def test_lake_provenance_records_both_adjustment_table_row_counts(fake_lake: Any) -> None:
    """`lake_provenance().adjustment_table_rows` holds each table's total row count."""
    fake_lake.add_empty("equity_adjustment_factors", _FACTORS)
    fake_lake._connection.execute(
        "INSERT INTO equity_adjustment_factors VALUES ('X', ?, 2.0, 0.5, 1)", [_TUE]
    )
    fake_lake.add_empty("yahoo_split_close_basis", _SPLITS)
    for ex_date in (_MON, _TUE):
        fake_lake._connection.execute(
            "INSERT INTO yahoo_split_close_basis VALUES ('X', ?, 0.5, TRUE)", [ex_date]
        )
    fake_lake.status = lambda: {"generated_at": None}
    fake_lake.health = lambda: {"version": "abc123"}

    provenance = lakehouse_loader.lake_provenance()

    assert provenance.adjustment_table_rows == {
        "equity_adjustment_factors": 1,
        "yahoo_split_close_basis": 2,
    }
    assert lakehouse_loader.LakeProvenance.model_fields["adjustment_table_rows"].default == {}


@pytest.fixture
def package_log(caplog: pytest.LogCaptureFixture) -> Any:
    """Routes the package logger (which does not propagate) into `caplog` at WARNING."""
    package = logging.getLogger("agentic_trading")
    package.addHandler(caplog.handler)
    previous_level = package.level
    package.setLevel(logging.WARNING)
    yield caplog
    package.setLevel(previous_level)
    package.removeHandler(caplog.handler)


def test_load_security_log_returns_warns_on_an_unfactored_split_and_keeps_returns(
    fake_lake: Any, package_log: pytest.LogCaptureFixture
) -> None:
    """A planted 2:1 drop with no factor row logs one WARNING naming it; returns are unchanged."""
    fake_lake.add_empty("equity_daily_from_bars", _BARS)
    for day, close in ((_MON, 100.0), (_TUE, 50.0)):
        fake_lake._connection.execute(
            "INSERT INTO equity_daily_from_bars VALUES ('AAA', ?, ?, NULL, 1)", [day, close]
        )
    fake_lake.add_empty("equity_adjustment_factors", _FACTORS)
    fake_lake._connection.execute(
        "INSERT INTO equity_adjustment_factors VALUES ('BBB', ?, 2.0, 0.5, 2)", [_TUE]
    )

    series = load_security_log_returns([1], _MON, _TUE)[1]

    assert series.iloc[0] == pytest.approx(math.log(0.5))
    warnings = [r.getMessage() for r in package_log.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "(1, 2024-03-05, 0.5000)" in warnings[0]


def test_load_security_log_returns_does_not_warn_when_the_split_has_a_factor_row(
    fake_lake: Any, package_log: pytest.LogCaptureFixture
) -> None:
    """A 2:1 split with a factor row on its ex-date is adjusted and not flagged."""
    fake_lake.add_empty("equity_daily_from_bars", _BARS)
    for day, close in ((_MON, 100.0), (_TUE, 50.0)):
        fake_lake._connection.execute(
            "INSERT INTO equity_daily_from_bars VALUES ('AAA', ?, ?, NULL, 1)", [day, close]
        )
    fake_lake.add_empty("equity_adjustment_factors", _FACTORS)
    fake_lake._connection.execute(
        "INSERT INTO equity_adjustment_factors VALUES ('AAA', ?, 2.0, 0.5, 1)", [_TUE]
    )

    series = load_security_log_returns([1], _MON, _TUE)[1]

    assert series.iloc[0] == pytest.approx(0.0)
    assert not [r for r in package_log.records if r.levelno == logging.WARNING]
