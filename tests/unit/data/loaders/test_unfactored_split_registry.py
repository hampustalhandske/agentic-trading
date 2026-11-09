# test_unfactored_split_registry.py

"""`lakehouse_loader` keeps a per-process registry of unfactored split-sized moves, and
`BacktestResult` and `extract_run_summary` report it."""

from collections.abc import Iterator
from datetime import date
from typing import Any

import pytest

from agentic_trading.agents.schemas import BacktestResult, UnfactoredSplitMove
from agentic_trading.backtest.run_summary import extract_run_summary
from agentic_trading.data.loaders import lakehouse_loader
from agentic_trading.data.loaders.lakehouse_loader import (
    clear_unfactored_split_move_registry,
    load_security_log_returns,
    unfactored_split_move_registry,
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

_MON, _TUE, _WED = date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 6)


@pytest.fixture(autouse=True)
def _empty_registry() -> Iterator[None]:
    """The registry is empty before and after each test."""
    clear_unfactored_split_move_registry()
    yield
    clear_unfactored_split_move_registry()


def _plant_two_for_one(lake: Any, security_ids: tuple[int, ...]) -> None:
    """Seeds each security with a 2:1 close drop on Tuesday and no factor row for it."""
    lake.add_empty("equity_daily_from_bars", _BARS)
    for security_id in security_ids:
        for day, close in ((_MON, 100.0), (_TUE, 50.0), (_WED, 51.0)):
            lake._connection.execute(
                "INSERT INTO equity_daily_from_bars VALUES ('X', ?, ?, NULL, ?)",
                [day, close, security_id],
            )
    lake.add_empty("equity_adjustment_factors", _FACTORS)
    lake._connection.execute(
        "INSERT INTO equity_adjustment_factors VALUES ('OTHER', ?, 2.0, 0.5, 99)", [_TUE]
    )


def test_overlapping_reads_of_one_planted_move_register_it_once_in_sorted_order(
    fake_lake: Any,
) -> None:
    """Rereading overlapping windows adds no duplicate; entries are sorted by id then date."""
    _plant_two_for_one(fake_lake, (1, 2))

    load_security_log_returns([2], _MON, _TUE)
    load_security_log_returns([1], _MON, _TUE)
    load_security_log_returns([1], _MON, _WED)
    load_security_log_returns([1, 2], _TUE, _WED)

    registry = unfactored_split_move_registry()
    assert [(m.security_id, m.as_of_date) for m in registry] == [(1, _TUE), (2, _TUE)]
    assert all(isinstance(move, UnfactoredSplitMove) for move in registry)
    assert all(min(m.ratio, 1 / m.ratio) == pytest.approx(0.5) for m in registry)


def test_clearing_the_registry_empties_it(fake_lake: Any) -> None:
    """After a flagged read the registry is non-empty; clearing returns it to []."""
    assert unfactored_split_move_registry() == []
    _plant_two_for_one(fake_lake, (1,))
    load_security_log_returns([1], _MON, _TUE)
    assert len(unfactored_split_move_registry()) == 1

    lakehouse_loader.clear_unfactored_split_move_registry()

    assert unfactored_split_move_registry() == []


def test_backtest_result_defaults_to_no_unfactored_split_moves() -> None:
    """`BacktestResult.unfactored_split_moves` defaults to an empty list."""
    field = BacktestResult.model_fields["unfactored_split_moves"]
    assert field.default_factory is not None and field.default_factory() == []  # type: ignore[call-arg]
    move = UnfactoredSplitMove(security_id=1, as_of_date=_TUE, ratio=0.5)
    assert (move.security_id, move.as_of_date, move.ratio) == (1, _TUE, 0.5)


def _block(**overrides: object) -> dict:
    block = {
        "start_date": "2018-04-02",
        "end_date": "2018-04-03",
        "portfolio_value_history": [["2018-04-02", 10_000.0], ["2018-04-03", 10_050.0]],
        "final_sharpe_ratio": 0.5,
        "final_max_drawdown": -0.01,
        "final_sortino_ratio": 0.6,
        "final_calmar_ratio": 0.4,
        "final_information_ratio": None,
        "final_hit_rate": 0.55,
        "final_profit_factor": 1.1,
        "signal_health": {},
        "source_performance": {},
    }
    block.update(overrides)
    return block


def test_run_summary_counts_unfactored_split_moves_per_security() -> None:
    """Each period reports counts keyed by security id string, and {} when there are none."""
    moves = [
        {"security_id": 7, "as_of_date": "2018-04-02", "ratio": 0.5},
        {"security_id": 7, "as_of_date": "2018-04-03", "ratio": 2.0},
        {"security_id": 12, "as_of_date": "2018-04-02", "ratio": 0.33},
    ]
    result = {
        "experiment": {},
        "market_data_source": "lakehouse data",
        "calibration": _block(),
        "test": _block(unfactored_split_moves=moves),
    }

    summary = extract_run_summary(result)

    assert summary["test"]["unfactored_split_moves_by_security"] == {"7": 2, "12": 1}
    assert summary["calibration"]["unfactored_split_moves_by_security"] == {}


def test_run_summary_reports_none_for_a_resumed_period() -> None:
    """A period whose `unfactored_split_moves` is null (a resumed run) reports `None`, not `{}`;
    a block without the key reports `{}`."""
    result = {
        "experiment": {},
        "market_data_source": "lakehouse data",
        "calibration": _block(),
        "test": _block(unfactored_split_moves=None),
    }

    summary = extract_run_summary(result)

    assert summary["test"]["unfactored_split_moves_by_security"] is None
    assert summary["calibration"]["unfactored_split_moves_by_security"] == {}
