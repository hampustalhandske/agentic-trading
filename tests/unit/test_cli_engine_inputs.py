# test_cli_engine_inputs.py

"""AT-23: every entry point that runs the backtest engine builds its inputs with
`backtest.engine_inputs.load_engine_inputs`, so the single-phase CLI and the firm record hand
the engine the same inputs for the same universe and window: the prices, the risk-free rate,
the BIS policy rates spot FX carry accrues from, the benchmark and the book's costs."""

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from agentic_trading.agents.roster import FIRM_V1_EQUITY_SECURITY_IDS, FIRM_V1_UNIVERSE
from agentic_trading.agents.schemas import BacktestResult
from agentic_trading.backtest.engine_inputs import EngineInputs, load_engine_inputs
from agentic_trading.backtest.preflight import PreflightReport
from agentic_trading.cli import app, firm_record_app
from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loader import DEFAULT_BACKTEST_WARMUP_DAYS
from agentic_trading.firm_record.runner import FirmRecordResult, run_firm_record

runner = CliRunner()
_START, _END = date(2010, 1, 4), date(2010, 1, 5)
_MARKET = pd.DataFrame(
    {("SPY", "close"): [100.0, 101.0]}, index=pd.DatetimeIndex(["2010-01-04", "2010-01-05"])
)
_RISK_FREE = pd.Series(0.0001, index=pd.date_range("2010-01-01", "2010-01-05"))
_POLICY_RATES = pd.DataFrame(
    {currency: [0.01] for currency in ("AUD", "EUR", "GBP", "JPY", "USD")},
    index=pd.DatetimeIndex(["2009-12-31"], name="date"),
)
_RECORD = FirmRecordResult(
    spec_hash="0" * 64, days_written=0, last_completed=None, complete=False, counted=False
)

# How the engine receives each field of `EngineInputs`, read off its constructor's keyword
# arguments; a new field fails `test_every_input_field_reaches_the_engine` until it is listed.
_ENGINE_VIEW: dict[str, Callable[[dict[str, Any]], Any]] = {
    "assets": lambda kwargs: list(kwargs["assets"]),
    "market_data": lambda kwargs: kwargs["data_loader"]([], _START, _END),
    "market_data_source": lambda kwargs: kwargs["market_data_source"],
    "risk_free_rate_series": lambda kwargs: kwargs["risk_free_rate_series"],
    "policy_rates": lambda kwargs: kwargs["policy_rates"],
    "benchmark_returns": lambda kwargs: kwargs["benchmark_returns"],
    "starting_cash": lambda kwargs: kwargs["portfolio"].cash,
    "commission_rate": lambda kwargs: kwargs["portfolio"].commission_rate,
    "slippage_bps": lambda kwargs: kwargs["portfolio"].slippage_bps,
    "execution_lag_days": lambda kwargs: kwargs["execution_lag_days"],
}


def _result() -> BacktestResult:
    return BacktestResult(
        start_date=_START,
        end_date=_END,
        portfolio_value_history=[(_START, 1.0), (_END, 1.0)],
        final_sharpe_ratio=0.0,
        final_max_drawdown=0.0,
        final_sortino_ratio=0.0,
        final_calmar_ratio=0.0,
        final_hit_rate=0.0,
        final_profit_factor=0.0,
    )


def _same(left: Any, right: Any) -> None:
    """Assert two engine inputs are equal: frames and series by content, anything else by ==."""
    if isinstance(left, pd.DataFrame):
        pd.testing.assert_frame_equal(left, right, check_freq=False)
    elif isinstance(left, pd.Series):
        pd.testing.assert_series_equal(left, right, check_freq=False)
    else:
        assert left == right


def test_every_input_field_reaches_the_engine() -> None:
    """The field-by-field view below covers every field of `EngineInputs`."""
    assert set(_ENGINE_VIEW) == set(EngineInputs.model_fields)


def test_both_cli_paths_hand_the_engine_the_same_inputs(
    tmp_path: Path, preflight_lake: Any
) -> None:
    """The single-phase CLI and the firm record over the firm's universe and one window both
    call load_engine_inputs, each with its own alignment and warm-up, and the engine each
    builds receives the same value of every input field: the risk-free and policy rates read
    once per run for the window, the prices, the source and the firm's costs."""
    with (
        temporary_overrides({"firm.commission_rate": 0.002, "firm.slippage_bps": 3.0}),
        patch(
            "agentic_trading.backtest.engine_inputs.load_price_history_with_fallback",
            return_value=(_MARKET, "lakehouse data"),
        ),
        patch(
            "agentic_trading.backtest.engine_inputs.load_risk_free_rate_series",
            return_value=_RISK_FREE,
        ) as risk_free,
        patch(
            "agentic_trading.backtest.engine_inputs.load_fx_policy_rates",
            return_value=_POLICY_RATES,
        ) as policy,
        patch("agentic_trading.cli.load_engine_inputs", wraps=load_engine_inputs) as loads,
        patch(
            "agentic_trading.cli.check_data_availability",
            return_value=PreflightReport(checked_at=_START, results=[]),
        ),
        patch("agentic_trading.cli.BacktestEngine") as single_engine,
        patch("agentic_trading.cli.run_firm_record", return_value=_RECORD) as record,
        patch("agentic_trading.cli.check_price_coverage", return_value=[]),
    ):
        single_engine.return_value.run.return_value = _result()
        single = runner.invoke(
            app,
            [
                *FIRM_V1_UNIVERSE,
                "--start-date",
                _START.isoformat(),
                "--end-date",
                _END.isoformat(),
                "--yes",
            ],
        )
        firm = runner.invoke(
            firm_record_app,
            [
                "--start",
                _START.isoformat(),
                "--end",
                _END.isoformat(),
                "--store-dir",
                str(tmp_path),
            ],
        )
        firm_inputs: EngineInputs = record.call_args.args[5]
        # The engine the firm record builds from those inputs.
        with patch("agentic_trading.firm_record.runner.BacktestEngine") as firm_engine:
            run_firm_record(
                record.call_args.args[0],
                record.call_args.args[1],
                _START,
                _END,
                tmp_path / "engine",
                firm_inputs,
                lake_time=record.call_args.kwargs["lake_time"],
                lake_version="lake",
                quantcore_version="q",
            )

    assert single.exit_code == 0, single.output
    assert firm.exit_code == 0, firm.output
    assert loads.call_count == 2
    single_call, firm_call = loads.call_args_list
    assert [list(call.args[0]) for call in (single_call, firm_call)] == [
        list(FIRM_V1_UNIVERSE),
        list(FIRM_V1_UNIVERSE),
    ]
    assert [call.args[1:] for call in (single_call, firm_call)] == [(_START, _END)] * 2
    assert single_call.kwargs == {
        "warmup_days": DEFAULT_BACKTEST_WARMUP_DAYS,
        "align": "intersection",
    }
    assert firm_call.kwargs == {
        "warmup_days": settings.stage_a.warmup_days,
        "align": "sessions",
        "calendar": settings.stage_a.session_calendar,
        "security_ids": FIRM_V1_EQUITY_SECURITY_IDS,
    }
    assert risk_free.call_count == policy.call_count == 2
    for call in risk_free.call_args_list:
        assert call.args == (_START, _END)
    for call in policy.call_args_list:
        assert (list(call.args[0]), *call.args[1:]) == (list(FIRM_V1_UNIVERSE), _START, _END)

    single_kwargs = single_engine.call_args.kwargs
    firm_kwargs = firm_engine.call_args.kwargs
    for view in _ENGINE_VIEW.values():
        _same(view(single_kwargs), view(firm_kwargs))
    assert single_kwargs["policy_rates"] is _POLICY_RATES
    assert firm_kwargs["policy_rates"] is _POLICY_RATES
    assert single_kwargs["risk_free_rate_series"] is _RISK_FREE
    assert firm_kwargs["risk_free_rate_series"] is _RISK_FREE
    assert single_kwargs["portfolio"].commission_rate == pytest.approx(0.002)
    assert firm_kwargs["portfolio"].slippage_bps == pytest.approx(3.0)


def test_load_engine_inputs_reads_every_input_for_the_run() -> None:
    """Prices over the window and its warm-up with the given alignment, the risk-free rate and
    the policy rates over the window, the benchmark's close-to-close returns, and the costs as
    `settings.firm` holds them at the call."""
    benchmark = pd.DataFrame(
        {("^GSPC", "close"): [100.0, 102.0, 101.0]},
        index=pd.DatetimeIndex(["2010-01-04", "2010-01-05", "2010-01-06"]),
    )
    prices = MagicMock(side_effect=[(_MARKET, "lakehouse data"), (benchmark, "lakehouse data")])
    with (
        temporary_overrides({"firm.execution_lag_days": 0, "firm.starting_cash": 5_000.0}),
        patch("agentic_trading.backtest.engine_inputs.load_price_history_with_fallback", prices),
        patch(
            "agentic_trading.backtest.engine_inputs.load_risk_free_rate_series",
            return_value=_RISK_FREE,
        ) as risk_free,
        patch(
            "agentic_trading.backtest.engine_inputs.load_fx_policy_rates",
            return_value=_POLICY_RATES,
        ) as policy,
    ):
        inputs = load_engine_inputs(
            ["SPY", "EURUSD=X"],
            _START,
            _END,
            warmup_days=30,
            align="sessions",
            calendar="XNYS",
            benchmark="^GSPC",
        )

    assert prices.call_args_list[0].args == (["SPY", "EURUSD=X"], _START, _END)
    assert prices.call_args_list[0].kwargs == {
        "warmup_days": 30,
        "align": "sessions",
        "calendar": "XNYS",
        "security_ids": None,
    }
    assert prices.call_args_list[1].args == (["^GSPC"], _START, _END)
    assert prices.call_args_list[1].kwargs == {"warmup_days": 0, "align": "intersection"}
    risk_free.assert_called_once_with(_START, _END)
    policy.assert_called_once_with(["SPY", "EURUSD=X"], _START, _END)
    assert inputs.market_data is _MARKET
    assert inputs.risk_free_rate_series is _RISK_FREE
    assert inputs.policy_rates is _POLICY_RATES
    assert inputs.benchmark_returns is not None
    assert inputs.benchmark_returns.tolist() == pytest.approx([0.02, 101.0 / 102.0 - 1.0])
    assert (inputs.starting_cash, inputs.execution_lag_days) == (5_000.0, 0)
    assert inputs.data_loader([], _START, _END) is _MARKET
    book = inputs.portfolio()
    assert (book.cash, book.positions, book.commission_rate, book.slippage_bps) == (
        5_000.0,
        {},
        settings.firm.commission_rate,
        settings.firm.slippage_bps,
    )


def test_engine_inputs_refuse_a_pair_without_policy_rates() -> None:
    """Engine inputs over spot FX pairs without policy rates raise, naming every pair; without
    a pair they need none."""
    with pytest.raises(ValueError, match=r"\['EURUSD=X', 'USDJPY=X'\]"):
        EngineInputs(
            assets=["SPY", "EURUSD=X", "USDJPY=X"],
            market_data=_MARKET,
            market_data_source="lakehouse data",
        )

    assert (
        EngineInputs(assets=["SPY"], market_data=_MARKET, market_data_source="x").policy_rates
        is None
    )
