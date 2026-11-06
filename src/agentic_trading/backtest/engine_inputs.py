# engine_inputs.py

"""Every input a `BacktestEngine` run takes from the lake and from the
firm's cost settings, loaded in one place (`load_engine_inputs`) so each
entry point that runs the engine (the single-phase CLI, the experiment and
the firm record) hands it the same inputs.

`EngineInputs` holds the price frame and its source label, the risk-free
rate series, the BIS policy rates spot FX carry accrues from, an optional
benchmark return series, and the book's starting cash. The commission,
slippage and execution lag are not inputs: `EngineInputs.portfolio` and
every engine read them from `settings.firm` at use time, as the no-trade
band, the settings fingerprint and the provenance do. An `EngineInputs`
whose assets hold a spot FX pair always carries policy rates
(`require_policy_rates`), so no run holds a pair without its carry."""

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.config import settings
from agentic_trading.data.instruments import currency_pair
from agentic_trading.data.loader import load_price_history_with_fallback
from agentic_trading.data.loaders.macro_loader import (
    load_fx_policy_rates,
    load_risk_free_rate_series,
)


def require_policy_rates(assets: Sequence[str], policy_rates: pd.DataFrame | None) -> None:
    """Raise `ValueError` naming every spot FX pair among `assets` when
    `policy_rates` is `None`; return when there is no pair or rates are
    given."""
    pairs = [asset for asset in assets if currency_pair(asset) is not None]
    if pairs and policy_rates is None:
        raise ValueError(
            f"spot FX pair(s) {pairs} need BIS policy rates to accrue carry, and none were given"
        )


class EngineInputs(BaseModel):
    """The inputs of one `BacktestEngine` run over `assets`.

    `market_data` is the price frame in `load_price_history_with_fallback`'s
    (asset, field) column shape covering the run and its warm-up, read from
    `market_data_source`. `risk_free_rate_series` is the daily decimal rate
    cash accrues (`load_risk_free_rate_series`), `policy_rates` the annual
    policy rate per currency spot FX carry accrues from
    (`load_fx_policy_rates`, `None` only when `assets` holds no pair), and
    `benchmark_returns` the daily returns the information ratio is measured
    against, `None` for none. `starting_cash` defaults to
    `settings.firm.starting_cash` when the model is built. Construction raises a `ValueError`
    (`pydantic.ValidationError`) naming the spot FX pairs of `assets` when
    `policy_rates` is `None`."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    assets: list[str]
    market_data: pd.DataFrame
    market_data_source: str
    risk_free_rate_series: pd.Series | None = None
    policy_rates: pd.DataFrame | None = None
    benchmark_returns: pd.Series | None = None
    starting_cash: float = Field(default_factory=lambda: settings.firm.starting_cash)

    @model_validator(mode="after")
    def _fx_pairs_have_rates(self) -> "EngineInputs":
        require_policy_rates(self.assets, self.policy_rates)
        return self

    def portfolio(self) -> Portfolio:
        """A fresh, empty book holding `starting_cash`, filled at
        `settings.firm.commission_rate` and `settings.firm.slippage_bps` as
        they stand at the call."""
        return Portfolio(
            cash=self.starting_cash,
            positions={},
            commission_rate=settings.firm.commission_rate,
            slippage_bps=settings.firm.slippage_bps,
        )

    def data_loader(self, _assets: list[str], _start: date, _end: date) -> pd.DataFrame:
        """The engine's `data_loader`: `market_data`, whatever is asked."""
        return self.market_data


def load_engine_inputs(
    assets: Sequence[str],
    start: date,
    end: date,
    *,
    warmup_days: int,
    align: Literal["sessions", "intersection"],
    calendar: str = "XNYS",
    security_ids: Mapping[str, int] | None = None,
    benchmark: str | None = None,
) -> EngineInputs:
    """Read the `EngineInputs` of a run over `assets` from `start` to `end`
    from the lake, inside the caller's `lake_session` and its window store.

    The price frame is `load_price_history_with_fallback(assets, start, end,
    warmup_days=warmup_days, align=align, calendar=calendar,
    security_ids=security_ids)`. The risk-free series and the policy rates
    are read for `start` to `end` (`load_risk_free_rate_series`,
    `load_fx_policy_rates`, which reads its own look-back before `start` so
    the rate in force on `start` is there). With a `benchmark` ticker, the
    benchmark returns are its daily close-to-close returns from `start` to
    `end`, aligned on its own bars.

    Raises `ValueError` when the lake has no policy rate for a currency of a
    spot FX pair among `assets` over the run; every loader's own errors
    propagate."""
    tickers = list(assets)
    market_data, source_label = load_price_history_with_fallback(
        tickers,
        start,
        end,
        warmup_days=warmup_days,
        align=align,
        calendar=calendar,
        security_ids=security_ids,
    )
    benchmark_returns = None
    if benchmark is not None:
        benchmark_frame, _ = load_price_history_with_fallback(
            [benchmark], start, end, warmup_days=0, align="intersection"
        )
        benchmark_returns = benchmark_frame[benchmark]["close"].dropna().pct_change().dropna()
    return EngineInputs(
        assets=tickers,
        market_data=market_data,
        market_data_source=source_label,
        risk_free_rate_series=load_risk_free_rate_series(start, end),
        policy_rates=load_fx_policy_rates(tickers, start, end),
        benchmark_returns=benchmark_returns,
    )
