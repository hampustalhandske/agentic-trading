# _firm.py

"""The small firm the determinism tests run, in the test process and in the subprocesses of
`pipeline_entry.py`: three real price-only analysts (risk minimizing, GBM momentum, OU mean
reversion) under real managers and `PortfolioConstructor`s, and a FRED analyst reading
`DGS3MO` vintages from a `FakeLake` inside the run's lake session, over four synthetic
instruments with a seeded warm-up and a three-week XNYS window.

`PLANT_ENV` names a defect planted in the FRED analyst for a negative control, read when the
firm is built: `random` (an unseeded random draw added to its confidence), `unsaved_state`
(its confidence depends on a count of its calls kept in a module global, which the engine
checkpoint, holding only the agents' private attributes, does not carry), `thread` (its
confidence depends on whether the main thread asks it) or `hash_order` (its confidence depends
on where its series falls in the iteration order of a set of strings, which the string-hash
seed decides).

`LAKE_FAULT_ENV`, when set and non-empty, makes the lake fail under the FRED analyst: from the
window's `LAKE_FAILS_AFTER_DAYS + 1`-th session on, its read raises the lake's HTTP 400 as a
`LakehouseError`. It is read at each call and is not part of the firm, so a pass without it
resumes the same trial."""

import os
import random
import threading
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd

from agentic_trading.agents.analysts.risk_pod.gbm_momentum_analyst import GBMMomentumAnalyst
from agentic_trading.agents.analysts.risk_pod.risk_minimizing import RiskMinimizingAnalyst
from agentic_trading.agents.analysts.stat_arb_pod.ou_mean_reversion_analyst import (
    OUMeanReversionAnalyst,
)
from agentic_trading.agents.base import AnalystAgent, DataRequirement, ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.managers.alt_data_manager import AltDataManager
from agentic_trading.agents.managers.risk_manager import RiskManager
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.backtest.engine_inputs import EngineInputs
from agentic_trading.config import settings
from agentic_trading.data.lakehouse_client import LakehouseError
from agentic_trading.data.loaders import lakehouse_loader
from agentic_trading.data.trading_calendar import sessions_between
from agentic_trading.scoring.weighting import SharpeWeight, ShrinkageWeight

PLANT_ENV = "AGENTIC_DETERMINISM_PLANT"
LAKE_FAULT_ENV = "AGENTIC_DETERMINISM_LAKE_FAULT"
PLANTS = ("random", "unsaved_state", "thread", "hash_order")

LAKE_TIME = datetime(2026, 11, 6, tzinfo=UTC)
WINDOW_START, WINDOW_END = date(2016, 1, 4), date(2016, 1, 22)
UNIVERSE = ["SPY", "QQQ", "IWM", "TLT"]
_WARMUP = pd.DatetimeIndex(sessions_between(date(2015, 3, 2), date(2015, 12, 31)))
_SESSIONS = pd.DatetimeIndex(sessions_between(WINDOW_START, WINDOW_END))
# Window sessions the lake fault (`LAKE_FAULT_ENV`) lets through before the FRED read fails.
LAKE_FAILS_AFTER_DAYS = 4
_FRED_SERIES = "DGS3MO"
# Iterated by the `hash_order` defect; the series' place in it differs between string-hash
# seeds 0 and 1.
_SERIES_SET = frozenset({"DGS3MO", "DGS10", "DGS2", "DTB3", "T10Y2Y"})
# Every call of a FRED analyst in this process; the `unsaved_state` defect reads it.
_PROCESS_CALLS = [0]


def plant() -> str | None:
    """The planted defect `PLANT_ENV` names, `None` when it names none."""
    name = os.environ.get(PLANT_ENV) or None
    if name is not None and name not in PLANTS:
        raise ValueError(f"{PLANT_ENV}={name!r} is not one of {PLANTS}")
    return name


def market_frame() -> pd.DataFrame:
    """Seeded open/close/log-return columns per instrument over the warm-up and the window."""
    index = _WARMUP.append(_SESSIONS)
    rng = np.random.default_rng(11)
    columns: dict[tuple[str, str], pd.Series] = {}
    for asset in UNIVERSE:
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(index)))), index)
        columns[(asset, "open")] = close.shift(1).fillna(close.iloc[0])
        columns[(asset, "close")] = close
        columns[(asset, "log_return")] = np.log(close / close.shift(1))
    return pd.DataFrame(columns)


def engine_inputs() -> EngineInputs:
    """The engine inputs of a record over `UNIVERSE` on `market_frame`, with no rate series."""
    return EngineInputs(
        assets=UNIVERSE, market_data=market_frame(), market_data_source="lakehouse data"
    )


def fred_vintages() -> pd.DataFrame:
    """One `DGS3MO` vintage per weekday observation of the warm-up and window, published the
    next calendar day, with seeded values."""
    days = pd.bdate_range("2015-10-01", WINDOW_END)
    rng = np.random.default_rng(5)
    values = 0.2 + np.cumsum(rng.normal(0.0, 0.01, len(days)))
    return pd.DataFrame(
        {
            "series_id": _FRED_SERIES,
            "date": days.date,
            "realtime_start": (days + pd.Timedelta(days=1)).date,
            "value": values,
        }
    )


def add_fred_vintages(lake: Any) -> None:
    """Adds `fred_vintages` to `lake`, a `FakeLake`, as a current table keyed on series,
    observation date and vintage date."""
    lake.add(
        "fred_vintages",
        fred_vintages(),
        ("date", "realtime_start"),
        key_columns=["series_id", "date", "realtime_start"],
    )


class FredTrendAnalyst(AnalystAgent):
    """Long when DGS3MO, as known before the decision day, fell over its last five
    observations, short when it rose, at a confidence growing with the move; flat and
    unavailable without five observations. Its `source_as_of` is the newest `known_at` it read."""

    agent_id: str = "fred_trend_stub_analyst"
    data_requirements: tuple[DataRequirement, ...] = (
        DataRequirement(
            source="fred", identifier=_FRED_SERIES, asset_scoped=False, description="3M bill"
        ),
    )
    plant: str | None = None

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        _PROCESS_CALLS[0] += 1
        if os.environ.get(LAKE_FAULT_ENV) and as_of_date >= _SESSIONS[LAKE_FAILS_AFTER_DAYS]:
            raise LakehouseError("HTTP 400: the query failed on the lake", status=400)
        vintages = lakehouse_loader.load_fred_vintages_from_lakehouse(
            _FRED_SERIES, as_of_date.date()
        )
        if vintages is None or len(vintages) < 5:
            return self.unavailable_signal(asset, as_of_date, "fewer than five observations")
        values = vintages["value"].to_numpy()
        move = float(values[-1] - values[-5])
        confidence = min(1.0, 0.2 + 10.0 * abs(move))
        if self.plant == "random":
            confidence = min(1.0, confidence + 0.01 * random.random())
        elif self.plant == "unsaved_state":
            confidence = min(1.0, 0.2 + 0.05 * (_PROCESS_CALLS[0] % 7))
        elif self.plant == "thread":
            if threading.current_thread() is not threading.main_thread():
                confidence = min(1.0, confidence + 0.01)
        elif self.plant == "hash_order":
            confidence = min(1.0, confidence + 0.01 * list(_SERIES_SET).index(_FRED_SERIES))
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long" if move < 0 else "short",
            confidence=confidence,
            rationale="DGS3MO five-observation move",
            source_as_of=pd.Timestamp(vintages["known_at"].max()).date(),
            sample_count=len(values),
        )


class FredPodManager(ManagerAgent):
    """A pod that passes its analyst's confidence through unscaled."""

    agent_id: str = "fred_pod_manager"

    def scale_factor(self, signal, market_data, as_of_date) -> float:  # type: ignore[no-untyped-def]
        return 1.0


def _constructor(pod: str) -> PortfolioConstructor:
    return PortfolioConstructor(**settings.pod_sizing[pod].model_dump())


def analysts() -> list[AnalystAgent]:
    """One of each of the firm's analysts, the defect `PLANT_ENV` names planted in the FRED
    analyst."""
    return [
        RiskMinimizingAnalyst(),
        GBMMomentumAnalyst(),
        OUMeanReversionAnalyst(),
        FredTrendAnalyst(plant=plant()),
    ]


def build_commander(parallel_workers: int = 1) -> Commander:
    """The firm: a risk pod (risk minimizing, GBM momentum), an alt-data pod (OU mean
    reversion) and a FRED pod, allocated by shrunk trailing Sharpe, with the defect
    `PLANT_ENV` names planted in the FRED analyst."""
    risk_minimizing, gbm_momentum, ou_mean_reversion, fred_trend = analysts()
    return Commander(
        managers=[
            RiskManager(
                managed_analysts=[risk_minimizing, gbm_momentum],
                weighting_strategy=ShrinkageWeight(SharpeWeight()),
                portfolio_constructor=_constructor("risk_manager"),
            ),
            AltDataManager(
                managed_analysts=[ou_mean_reversion],
                weighting_strategy=ShrinkageWeight(SharpeWeight()),
                portfolio_constructor=_constructor("alt_data_manager"),
            ),
            FredPodManager(
                managed_analysts=[fred_trend],
                weighting_strategy=ShrinkageWeight(SharpeWeight()),
                portfolio_constructor=_constructor("macro_rates_manager"),
            ),
        ],
        weighting_strategy=ShrinkageWeight(SharpeWeight()),
        parallel_workers=parallel_workers,
    )
