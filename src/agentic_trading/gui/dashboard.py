from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import cast

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.managers import (
    AltDataManager,
    AnomalyNetworkManager,
    CVaRManager,
    EventCreditManager,
    FactorManager,
    MacroRatesManager,
    RiskManager,
    StatArbManager,
)
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.roster import (
    ALT_DATA_MANAGER_ANALYST_IDS,
    ANOMALY_NETWORK_MANAGER_ANALYST_IDS,
    CVAR_MANAGER_ANALYST_IDS,
    EVENT_CREDIT_MANAGER_ANALYST_IDS,
    FACTOR_MANAGER_ANALYST_IDS,
    MACRO_RATES_MANAGER_ANALYST_IDS,
    RISK_MANAGER_ANALYST_IDS,
    STAT_ARB_MANAGER_ANALYST_IDS,
    build_default_analysts,
    build_default_managers,
)
from agentic_trading.agents.schemas import (
    BacktestResult,
    ManagerSignal,
    PortfolioDecision,
    TradeRecord,
)
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.data.cache import CACHE_ROOT
from agentic_trading.data.db import read_fetch_log
from agentic_trading.data.loader import (
    DEFAULT_BACKTEST_WARMUP_DAYS,
    load_price_history_with_fallback,
)
from agentic_trading.data.ticker_catalog import TICKER_GROUPS, TickerGroup, tickers_by_asset_class
from agentic_trading.logging_config import configure_logging, get_logger
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import Hedge

# Hardcoded rather than get_logger(__name__): Streamlit execs this file
# directly, so __name__ is "__main__" here, not "agentic_trading.gui.dashboard".
logger = get_logger("agentic_trading.gui.dashboard")

HOLDINGS_EMPTY_PLACEHOLDER = "—"
SP500_TICKER = "^GSPC"
AUTOPLAY_INTERVAL_SECONDS = 0.9


@dataclass(slots=True)
class DeskRun:
    """Everything one simulation run produced: the multi-asset backtest
    itself (via the shared `BacktestEngine` — see `_simulate_desk`, there is
    no separate GUI-only simulation loop) plus the two benchmarks it's
    judged against throughout the GUI."""

    tickers: list[str]
    group_name: str
    manager_mode: str
    source_label: str
    manager_ids: list[str]
    analyst_ids: list[str]
    price_history: pd.DataFrame
    trade_blotter: pd.DataFrame
    backtest_result: BacktestResult
    track_record: TrackRecord
    scoreboard: PerformanceScoreboard
    portfolio_construction_method: str | None = None
    sp500_benchmark: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    basket_benchmark: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def decisions_by_day(self) -> dict[date, list[PortfolioDecision]]:
        return self.backtest_result.decisions_by_day


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root { color-scheme: dark; }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(14, 165, 233, 0.10), transparent 24%),
                    radial-gradient(circle at top right, rgba(59, 130, 246, 0.08), transparent 22%),
                    linear-gradient(180deg, #08111f 0%, #0b1220 46%, #0a0f1b 100%);
                color: #e5e7eb;
            }

            .block-container {
                padding-top: 3.5rem;
                padding-bottom: 2rem;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.80));
                border-right: 1px solid rgba(148, 163, 184, 0.15);
            }

            .mini-note {
                color: #94a3b8;
                font-size: 0.82rem;
            }

            .section-caption {
                color: #94a3b8;
                font-size: 0.88rem;
                margin-top: -0.4rem;
                margin-bottom: 0.6rem;
            }

            div[data-testid="stTabs"] button[data-baseweb="tab"] {
                font-size: 0.95rem;
                font-weight: 600;
            }

            .flow-layer-label {
                color: #94a3b8;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }

            .agent-node {
                display: block;
                border-radius: 8px;
                padding: 0.35rem 0.55rem;
                margin-bottom: 0.4rem;
                font-size: 0.8rem;
                border: 1px solid rgba(148, 163, 184, 0.25);
            }
            .agent-node-name {
                font-weight: 600;
                display: block;
            }
            .agent-node-agree {
                background: rgba(34, 197, 94, 0.12);
                border-color: rgba(34, 197, 94, 0.45);
            }
            .agent-node-disagree {
                background: rgba(239, 68, 68, 0.10);
                border-color: rgba(239, 68, 68, 0.4);
            }
            .agent-node-flat {
                background: rgba(100, 116, 139, 0.12);
                border-color: rgba(100, 116, 139, 0.35);
            }
            .agent-node-unavailable {
                background: rgba(245, 158, 11, 0.12);
                border-color: rgba(245, 158, 11, 0.45);
            }
            .agent-node-error {
                background: rgba(239, 68, 68, 0.12);
                border-color: rgba(239, 68, 68, 0.55);
            }

            .status-pill {
                display: inline-block;
                border-radius: 999px;
                padding: 0.15rem 0.6rem;
                margin: 0.15rem 0.3rem 0.15rem 0;
                font-size: 0.76rem;
                font-weight: 600;
                border: 1px solid rgba(148, 163, 184, 0.3);
            }
            .status-pill-live {
                background: rgba(34, 197, 94, 0.14);
                color: #4ade80;
                border-color: rgba(34, 197, 94, 0.4);
            }
            .status-pill-cached {
                background: rgba(59, 130, 246, 0.14);
                color: #60a5fa;
                border-color: rgba(59, 130, 246, 0.4);
            }
            .status-pill-fallback {
                background: rgba(234, 179, 8, 0.14);
                color: #facc15;
                border-color: rgba(234, 179, 8, 0.4);
            }
            .status-pill-unused {
                background: rgba(100, 116, 139, 0.10);
                color: #94a3b8;
                border-color: rgba(100, 116, 139, 0.3);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_analysts() -> list[AnalystAgent]:
    return build_default_analysts()


ALL_PODS_MODE = "All 8 pods"

# Single-pod modes: each hands its manager only the analysts belonging to
# its own pod (via roster.py's *_ANALYST_IDS), the same filtering
# build_default_managers() does — not the full roster, since a single pod
# is meant to isolate that pod's own signals for inspection.
_SINGLE_POD_MANAGERS: dict[
    str, tuple[Callable[[list[AnalystAgent]], ManagerAgent], frozenset[str]]
] = {
    "Risk Manager": (
        lambda owned: RiskManager(managed_analysts=owned, weighting_strategy=Hedge()),
        RISK_MANAGER_ANALYST_IDS,
    ),
    "CVaR Manager": (
        lambda owned: CVaRManager(managed_analysts=owned, weighting_strategy=Hedge()),
        CVAR_MANAGER_ANALYST_IDS,
    ),
    "StatArb Manager": (
        lambda owned: StatArbManager(managed_analysts=owned, weighting_strategy=Hedge()),
        STAT_ARB_MANAGER_ANALYST_IDS,
    ),
    "Macro/Rates Manager": (
        lambda owned: MacroRatesManager(managed_analysts=owned, weighting_strategy=Hedge()),
        MACRO_RATES_MANAGER_ANALYST_IDS,
    ),
    "Factor Manager": (
        lambda owned: FactorManager(managed_analysts=owned, weighting_strategy=Hedge()),
        FACTOR_MANAGER_ANALYST_IDS,
    ),
    "Event/Credit Manager": (
        lambda owned: EventCreditManager(managed_analysts=owned, weighting_strategy=Hedge()),
        EVENT_CREDIT_MANAGER_ANALYST_IDS,
    ),
    "Alt-Data Manager": (
        lambda owned: AltDataManager(managed_analysts=owned, weighting_strategy=Hedge()),
        ALT_DATA_MANAGER_ANALYST_IDS,
    ),
    "Anomaly/Network Manager": (
        lambda owned: AnomalyNetworkManager(managed_analysts=owned, weighting_strategy=Hedge()),
        ANOMALY_NETWORK_MANAGER_ANALYST_IDS,
    ),
}
MANAGER_MODES: tuple[str, ...] = (*_SINGLE_POD_MANAGERS, ALL_PODS_MODE)


def _build_managers(manager_mode: str, analysts: list[AnalystAgent]) -> list[ManagerAgent]:
    """`ALL_PODS_MODE` wires up the full 8-pod roster (see
    agents/roster.py), splitting ownership of `analysts` across all eight
    managers rather than duplicating every analyst under each — that would
    double-count each signal in Commander.decide(). Every other mode
    isolates a single pod, handing its manager only that pod's own
    analysts filtered out of the full roster."""
    if manager_mode == ALL_PODS_MODE:
        return build_default_managers(analysts)
    build_manager, analyst_ids = _SINGLE_POD_MANAGERS[manager_mode]
    owned = [analyst for analyst in analysts if analyst.agent_id in analyst_ids]
    return [build_manager(owned)]


def _load_basket_data(
    tickers: tuple[str, ...],
    start: date,
    end: date,
    warmup_days: int = 0,
    allow_synthetic_fallback: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Thin wrapper around the shared `data.loader.load_price_history_with_fallback`
    — the exact same function the CLI uses, so both front ends fall back to
    synthetic data (and log the same WARNING) identically on a live-data
    failure. Deliberately uncached: every run fetches fresh from yfinance for
    whatever tickers/date range the user has currently selected, rather than
    silently serving a stale result from an earlier selection."""
    return load_price_history_with_fallback(
        list(tickers),
        start,
        end,
        warmup_days=warmup_days,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )


def _normalized_return_pct(close: pd.Series) -> pd.Series:
    """`close` rebased to a % return from its first valid observation — the
    common unit every performance line in this dashboard is plotted in."""
    valid = close.dropna()
    if valid.empty or valid.iloc[0] == 0:
        return pd.Series(dtype=float)
    base = float(cast(float, valid.iloc[0]))
    return (close / base - 1.0) * 100.0


def _sp500_benchmark(start: date, end: date) -> pd.Series:
    frame, _ = _load_basket_data((SP500_TICKER,), start, end)
    return _normalized_return_pct(frame[SP500_TICKER]["close"])


def _basket_index_benchmark(
    price_history: pd.DataFrame, tickers: list[str], start: date | None = None
) -> pd.Series:
    """An equal-weighted, buy-and-hold index of `tickers` — "if I'd just
    bought and held this basket" rather than run the desk on it."""
    per_ticker = []
    for ticker in tickers:
        close = price_history[ticker]["close"]
        if start is not None:
            close = close.loc[close.index >= pd.Timestamp(start)]
        per_ticker.append(_normalized_return_pct(close))
    per_ticker = [series for series in per_ticker if not series.empty]
    if not per_ticker:
        return pd.Series(dtype=float)
    return pd.concat(per_ticker, axis=1).mean(axis=1, skipna=True)


def _portfolio_pct_series(
    portfolio_value_history: list[tuple[date, float]], initial_cash: float
) -> pd.Series:
    if not portfolio_value_history or initial_cash == 0:
        return pd.Series(dtype=float)
    index = [pd.Timestamp(day) for day, _ in portfolio_value_history]
    values = [(value / initial_cash - 1.0) * 100.0 for _, value in portfolio_value_history]
    return pd.Series(values, index=index)


def _cumulative_pct_series(track_record: TrackRecord, source_id: str) -> pd.Series:
    """One source's (analyst or manager) full-period contribution curve:
    daily attributable_return summed across every asset it touched that
    day, then cumulatively summed across days and expressed as a
    percentage — "this source's running share of portfolio performance,"
    per PerformanceRecord's own attribution convention."""
    entries = track_record.entries_for(source_id)
    if not entries:
        return pd.Series(dtype=float)
    daily_totals: dict[date, float] = {}
    for entry in entries:
        daily_totals[entry.as_of_date] = (
            daily_totals.get(entry.as_of_date, 0.0) + entry.attributable_return
        )
    ordered_dates = sorted(daily_totals)
    cumulative = np.cumsum([daily_totals[day] for day in ordered_dates]) * 100.0
    return pd.Series(cumulative, index=[pd.Timestamp(day) for day in ordered_dates])


def _combined_chart_frame(
    primary_series: dict[str, pd.Series],
    sp500: pd.Series,
    basket_index: pd.Series,
) -> pd.DataFrame:
    """Combines named performance lines with the two standing benchmarks
    into one chart-ready frame. Every series here is already Timestamp-
    indexed (never plain `date`) — mixing the two index types is exactly
    what makes `.sort_index()` raise comparing Timestamp to date."""
    columns = dict(primary_series)
    if not sp500.empty:
        columns["S&P 500"] = sp500
    if not basket_index.empty:
        columns["Basket index"] = basket_index
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns).sort_index()


def _holdings_by_day(
    decisions_by_day: dict[date, list[PortfolioDecision]],
    source_id: str,
    id_field: str,
    empty_placeholder: str = HOLDINGS_EMPTY_PLACEHOLDER,
) -> dict[date, str]:
    """One source's (desk/manager or analyst) compact "holdings" string per
    day — "AAPL long (0.62) · MSFT short (0.41)" — built from every signal
    attributable to `source_id` (matched on `id_field`) that day. This is
    the content behind each point's hover tooltip on the Desks/Analysts
    performance charts. Days with no matching signal map to
    `empty_placeholder` rather than an empty string, so the tooltip always
    shows something."""
    result: dict[date, str] = {}
    for day, decisions in decisions_by_day.items():
        parts = [
            f"{decision.asset} {signal.direction} ({signal.confidence:.2f})"
            for decision in decisions
            for signal in decision.contributing_signals
            if getattr(signal, id_field) == source_id
        ]
        result[day] = " · ".join(parts) if parts else empty_placeholder
    return result


def _long_format_chart_frame(
    chart_frame: pd.DataFrame,
    holdings_by_series: dict[str, dict[date, str]] | None = None,
    empty_placeholder: str = HOLDINGS_EMPTY_PLACEHOLDER,
) -> pd.DataFrame:
    """Reshapes `_combined_chart_frame`'s wide (date-indexed, one column per
    series) output into long format (one row per date/series/value triple)
    with an added `holdings` column — the shape Altair's color/tooltip
    encoding needs, since it draws one mark per row rather than one line
    per column.

    `holdings_by_series` maps a series name (e.g. an analyst or manager id)
    to that source's own `_holdings_by_day()` output; series with no entry
    there — the S&P 500 and basket-index benchmark columns, which have no
    per-asset signals of their own — fall back to `empty_placeholder`."""
    if chart_frame.empty:
        return pd.DataFrame(columns=["date", "series", "value", "holdings"])

    holdings_by_series = holdings_by_series or {}
    long_frame = chart_frame.reset_index(names="date").melt(
        id_vars="date", var_name="series", value_name="value"
    )
    long_frame = long_frame.dropna(subset=["value"]).reset_index(drop=True)

    def _holdings_for_row(row: pd.Series) -> str:
        day_map = holdings_by_series.get(str(row["series"]))
        if not day_map:
            return empty_placeholder
        timestamp = row["date"]
        as_of_date = timestamp.date() if isinstance(timestamp, pd.Timestamp) else timestamp
        return day_map.get(as_of_date, empty_placeholder)

    long_frame["holdings"] = long_frame.apply(_holdings_for_row, axis=1)
    return long_frame


def _render_altair_performance_chart(
    chart_frame: pd.DataFrame, holdings_by_series: dict[str, dict[date, str]]
) -> None:
    """Thin rendering call: hands `_long_format_chart_frame`'s output to
    `st.altair_chart`. All of the actual shaping/lookup logic lives in the
    pure functions above so it's testable without a Streamlit runtime —
    this function itself does nothing but build the Altair spec and render
    it."""
    long_frame = _long_format_chart_frame(chart_frame, holdings_by_series)
    if long_frame.empty:
        st.info("Nothing selected to plot yet.")
        return
    chart = (
        alt.Chart(long_frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Cumulative %"),
            color=alt.Color("series:N", title="Series"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title="Value", format=".2f"),
                alt.Tooltip("holdings:N", title="Holdings that day"),
            ],
        )
        .interactive()
    )
    st.altair_chart(chart, width="stretch", height=320)


def _build_trade_blotter(
    trade_log: list[TradeRecord], decisions_by_day: dict[date, list[PortfolioDecision]]
) -> pd.DataFrame:
    """Reshapes BacktestEngine's TradeRecord log into the wide, display-
    ready frame the Overview tab renders — joined against decisions_by_day
    (matched on date+asset) purely to attribute each trade to the
    managers/analysts that produced it, for display; the trade mechanics
    themselves (target_shares, delta_shares, notional, ...) come entirely
    from TradeRecord, not re-derived here."""
    decision_lookup = {
        (day, decision.asset): decision
        for day, decisions in decisions_by_day.items()
        for decision in decisions
    }
    rows: list[dict[str, object]] = []
    for trade in trade_log:
        decision = decision_lookup.get((trade.as_of_date, trade.asset))
        contributing = decision.contributing_signals if decision is not None else []
        rows.append(
            {
                "date": trade.as_of_date,
                "asset": trade.asset,
                "action": trade.action,
                "target_weight": trade.target_weight,
                "current_weight": trade.previous_weight,
                "target_shares": trade.target_shares,
                "delta_shares": trade.delta_shares,
                "notional": trade.notional,
                "close": trade.price,
                "manager_ids": ", ".join(sorted({s.manager_id for s in contributing})),
                "analyst_ids": ", ".join(sorted({s.source_agent_id for s in contributing})),
                "rationale": contributing[0].rationale
                if contributing
                else "No contributing signals",
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["date", "asset"]).reset_index(drop=True)
    return frame


def _simulate_desk(
    tickers: list[str],
    group_name: str,
    start: date,
    end: date,
    initial_cash: float,
    manager_mode: str,
    portfolio_construction_method: str | None = None,
    on_day_complete: object = None,
) -> DeskRun:
    """Runs the desk via `BacktestEngine` — the same class the CLI calls,
    not a second, GUI-only simulation loop. Nothing is cached: every call
    fetches fresh market data and runs a fresh backtest for whatever
    tickers/dates/manager mode are currently selected. `on_day_complete`, if
    given, is forwarded straight to `BacktestEngine.run()` to drive a live
    progress display — see `_run_simulation_with_progress()`.

    `portfolio_construction_method`, if given, enables the optional
    post-Commander `PortfolioConstructor` stage with that risk-budget
    method — mirroring the CLI's `--portfolio-construction` flag, off by
    default (`None`) so existing runs are unaffected."""
    if start > end:
        start, end = end, start

    analysts = _build_analysts()
    managers = _build_managers(manager_mode, analysts)
    commander = Commander(managers=managers, weighting_strategy=Hedge(), parallel_workers=4)
    portfolio = Portfolio(cash=initial_cash, positions={})
    track_record = TrackRecord()
    market_data, source_label = _load_basket_data(
        tuple(tickers),
        start,
        end,
        warmup_days=DEFAULT_BACKTEST_WARMUP_DAYS,
        allow_synthetic_fallback=False,
    )
    portfolio_constructor = (
        PortfolioConstructor(enabled=True, method=portfolio_construction_method)
        if portfolio_construction_method is not None
        else None
    )

    engine = BacktestEngine(
        commander=commander,
        portfolio=portfolio,
        track_record=track_record,
        data_loader=lambda _assets, _start, _end: market_data,
        start_date=start,
        end_date=end,
        assets=tickers,
        portfolio_constructor=portfolio_constructor,
        market_data_source=source_label,
    )
    backtest_result = engine.run(on_day_complete=on_day_complete)  # type: ignore[arg-type]

    return DeskRun(
        tickers=tickers,
        group_name=group_name,
        manager_mode=manager_mode,
        source_label=source_label,
        manager_ids=[manager.agent_id for manager in managers],
        analyst_ids=[analyst.agent_id for analyst in analysts],
        portfolio_construction_method=portfolio_construction_method,
        price_history=market_data,
        trade_blotter=_build_trade_blotter(
            backtest_result.trade_log, backtest_result.decisions_by_day
        ),
        backtest_result=backtest_result,
        track_record=track_record,
        scoreboard=PerformanceScoreboard(track_record),
        sp500_benchmark=_sp500_benchmark(start, end),
        basket_benchmark=_basket_index_benchmark(market_data, tickers, start),
    )


def _run_simulation_with_progress(
    tickers: list[str],
    group_name: str,
    start: date,
    end: date,
    initial_cash: float,
    manager_mode: str,
    portfolio_construction_method: str | None = None,
) -> DeskRun:
    """Runs the desk behind a live, center-of-screen progress display.
    `st.status()` streams each day's update to the browser as it happens,
    driven by `BacktestEngine.run()`'s `on_day_complete` callback."""
    with st.status(
        f"Simulating {len(tickers)} ticker{'s' if len(tickers) != 1 else ''}...",
        expanded=True,
    ) as status:
        progress_bar = st.progress(0.0)
        detail = st.empty()

        def _on_day_complete(
            day_index: int, total_days: int, as_of_date: date, portfolio_value: float
        ) -> None:
            fraction = (day_index + 1) / total_days if total_days else 1.0
            progress_bar.progress(fraction)
            detail.markdown(
                f"**Day {day_index + 1} of {total_days}** — {as_of_date.isoformat()} — "
                f"portfolio value **${portfolio_value:,.2f}**"
            )

        desk_run = _simulate_desk(
            tickers,
            group_name,
            start,
            end,
            initial_cash,
            manager_mode,
            portfolio_construction_method=portfolio_construction_method,
            on_day_complete=_on_day_complete,
        )
        status.update(
            label=f"Simulation complete — {desk_run.source_label}.",
            state="complete",
            expanded=False,
        )
    return desk_run


def _set_current_day_index(index: int) -> None:
    """Single write path for `current_day_index` — also keeps the day-scrubber
    slider's own widget state (a separate session_state key) in sync, since
    Streamlit widgets don't follow a plain session_state variable on their own."""
    st.session_state["current_day_index"] = index
    st.session_state["day_scrubber"] = index


def _step_day(delta: int) -> None:
    backtest_result = st.session_state.get("backtest_result")
    if backtest_result is None:
        return
    last_index = max(len(backtest_result.portfolio_value_history) - 1, 0)
    _set_current_day_index(max(0, min(last_index, st.session_state["current_day_index"] + delta)))


def _jump_to_day(index: int) -> None:
    _set_current_day_index(index)


def _manual_step_day(delta: int) -> None:
    """Step-forward/backward button callback. Sets a flag consumed by
    `_render_overview_tab`'s autoplay-tick check, so a manual click while
    playing is distinguishable from the run_every timer firing — otherwise
    both would apply in the same rerun and double-advance the day."""
    st.session_state["_manual_nav_this_run"] = True
    _step_day(delta)


def _manual_jump_to_day(index: int) -> None:
    """Day-scrubber on_change callback — see _manual_step_day for why this
    sets the flag instead of calling _jump_to_day directly."""
    st.session_state["_manual_nav_this_run"] = True
    _jump_to_day(index)


def _current_day(backtest_result: BacktestResult, current_day_index: int) -> tuple[date, float]:
    if not backtest_result.portfolio_value_history:
        _set_current_day_index(0)
        return date.today(), 0.0

    current_day_index = max(
        0, min(current_day_index, len(backtest_result.portfolio_value_history) - 1)
    )
    _set_current_day_index(current_day_index)
    return backtest_result.portfolio_value_history[current_day_index]


def _current_playback_date() -> date | None:
    """Resolves "what day is it" for playback purposes purely by reading
    (never writing) `st.session_state["current_day_index"]` — the single
    source of truth for playback position that `_current_day()` also reads,
    reused here rather than reimplemented. Deliberately does not call
    `_current_day()` itself: that function also writes
    `current_day_index`/`day_scrubber` back into session_state, and the
    Desks/Analysts tabs render *after* the Overview tab's fragment has
    already instantiated the `day_scrubber` slider widget in the same
    script run — writing to a widget's key post-instantiation raises
    StreamlitAPIException. Returns None when no backtest is loaded yet."""
    backtest_result: BacktestResult | None = st.session_state.get("backtest_result")
    if backtest_result is None or not backtest_result.portfolio_value_history:
        return None
    current_day_index = int(st.session_state.get("current_day_index", 0))
    current_day_index = max(
        0, min(current_day_index, len(backtest_result.portfolio_value_history) - 1)
    )
    resolved_date, _ = backtest_result.portfolio_value_history[current_day_index]
    return resolved_date


def _clip_series_to_date(series: pd.Series, as_of: date) -> pd.Series:
    """Clips a Timestamp-indexed series to observations on or before
    `as_of` — the mechanism behind every "live," playback-following chart
    in this dashboard. `as_of` is converted to a Timestamp explicitly
    before comparing against the series' Timestamp index; comparing a
    Timestamp against a plain `date` directly raises TypeError."""
    if series.empty:
        return series
    return series.loc[series.index <= pd.Timestamp(as_of)]


def _seed_defaults() -> None:
    st.session_state.setdefault("current_day_index", 0)
    st.session_state.setdefault("is_playing", False)
    st.session_state.setdefault("backtest_result", None)
    st.session_state.setdefault("scoreboard", None)
    st.session_state.setdefault("desk_run", None)
    st.session_state.setdefault("_manual_nav_this_run", False)


def _reset_desk() -> None:
    """Clears the loaded run back to an empty state."""
    st.session_state["desk_run"] = None
    st.session_state["backtest_result"] = None
    st.session_state["scoreboard"] = None
    st.session_state["is_playing"] = False
    _set_current_day_index(0)


def _realized_log_return(price_history: pd.DataFrame, asset: str, day: date) -> float | None:
    """The realized log return for `asset` on `day`, read from
    `price_history`'s MultiIndex ((ticker, "log_return")) columns. None if
    `day`/`asset` isn't present (e.g. the very first simulated day, which
    has no prior close to compute a return from)."""
    try:
        value = price_history.loc[pd.Timestamp(day), (asset, "log_return")]
    except KeyError:
        return None
    if pd.isna(value):
        return None
    return float(cast(float, value))


def _signal_history_rows(
    decisions_by_day: dict[date, list[PortfolioDecision]],
    source_id: str,
    id_field: str,
    price_history: pd.DataFrame | None = None,
    track_record: TrackRecord | None = None,
) -> pd.DataFrame:
    """Every recorded signal attributable to `source_id` (matched on
    `id_field`, "manager_id" or "source_agent_id"), across the whole
    backtest — the decision-making history behind one desk or analyst,
    what the "when it bought/sold" drill-down dialog is built from.

    `price_history` and `track_record` are optional so this stays callable
    with just `decisions_by_day` for narrow, filtering-only tests; when
    supplied, each row is enriched with that day's realized log return for
    the asset (from `price_history`) and `source_id`'s own cumulative
    contribution % as of that day (via `_cumulative_pct_series`, reused
    rather than re-summed here a second way).
    """
    cumulative_series = (
        _cumulative_pct_series(track_record, source_id)
        if track_record is not None
        else pd.Series(dtype=float)
    )
    rows: list[dict[str, object]] = []
    for day, decisions in sorted(decisions_by_day.items()):
        cumulative_contribution_pct: float | None = None
        if not cumulative_series.empty:
            as_of = cumulative_series.loc[cumulative_series.index <= pd.Timestamp(day)]
            if not as_of.empty:
                cumulative_contribution_pct = float(as_of.iloc[-1])
        for decision in decisions:
            for signal in decision.contributing_signals:
                if getattr(signal, id_field) != source_id:
                    continue
                rows.append(
                    {
                        "date": day,
                        "asset": decision.asset,
                        "manager": signal.manager_id,
                        "analyst": signal.source_agent_id,
                        "direction": signal.direction,
                        "status": signal.status,
                        "confidence": signal.confidence,
                        "scale_applied": signal.scale_applied,
                        "target_weight": decision.target_weight,
                        "realized_log_return": (
                            _realized_log_return(price_history, decision.asset, day)
                            if price_history is not None
                            else None
                        ),
                        "cumulative_contribution_pct": cumulative_contribution_pct,
                        "rationale": signal.rationale,
                    }
                )
    return pd.DataFrame(rows)


def _source_summary_frame(
    desk_run: DeskRun, source_ids: list[str], id_field: str, label_singular: str
) -> pd.DataFrame:
    """The Desks/Analysts tabs' summary table: one row per source, with
    contribution %, how many signals it produced, its average confidence,
    and what it's currently signaling — everything needed to judge each
    source at a glance before drilling into one via the dialog below."""
    rows: list[dict[str, object]] = []
    for source_id in source_ids:
        history = _signal_history_rows(desk_run.decisions_by_day, source_id, id_field)
        series = _cumulative_pct_series(desk_run.track_record, source_id)
        contribution_pct = float(series.iloc[-1]) if not series.empty else 0.0
        if history.empty:
            rows.append(
                {
                    label_singular: source_id,
                    "contribution_%": contribution_pct,
                    "signals": 0,
                    "avg_confidence": 0.0,
                    "last_direction": "—",
                    "active": 0,
                    "unavailable": 0,
                    "errors": 0,
                }
            )
            continue
        last_row = history.sort_values("date").iloc[-1]
        rows.append(
            {
                label_singular: source_id,
                "contribution_%": contribution_pct,
                "signals": len(history),
                "avg_confidence": float(history["confidence"].mean()),
                "last_direction": f"{last_row['direction']} ({last_row['asset']})",
                "active": int((history["status"] == "active").sum()),
                "unavailable": int((history["status"] == "unavailable").sum()),
                "errors": int((history["status"] == "error").sum()),
            }
        )
    return pd.DataFrame(rows)


def _majority_direction(directions: list[str]) -> str:
    """The most common direction among `directions`, "flat" for an empty
    or perfectly-tied input — the reference point every node in the
    agent-flow view is colored agree/disagree against."""
    if not directions:
        return "flat"
    counts = {direction: directions.count(direction) for direction in set(directions)}
    top_count = max(counts.values())
    tied = sorted(direction for direction, count in counts.items() if count == top_count)
    return tied[0] if len(tied) > 1 else next(iter(tied))


@dataclass(slots=True)
class AgentFlowNode:
    """One node in the agent-flow diagram: a name, its current
    direction/confidence output, and whether that direction agrees with
    the day's majority-vote direction (used purely for the node's color)."""

    name: str
    direction: str
    confidence: float
    agrees_with_majority: bool
    status: str = "active"


@dataclass(slots=True)
class AgentFlowSnapshot:
    """The four-layer pipeline's state for one asset on one day: every
    contributing analyst signal, every manager's aggregated read across
    the analysts it owns, and the Commander's final call — everything the
    agent-flow view renders, computed once so the render function stays a
    thin loop over already-shaped data."""

    analysts: list[AgentFlowNode]
    managers: list[AgentFlowNode]
    commander: AgentFlowNode | None
    majority_direction: str


def _build_agent_flow_snapshot(
    decisions_by_day: dict[date, list[PortfolioDecision]],
    asset: str,
    as_of_date: date,
) -> AgentFlowSnapshot:
    """Reshapes one day's already-recorded `ManagerSignal`s for `asset`
    into the three-layer node structure `_render_agent_flow_view` draws.
    Analyst-layer confidence is each `ManagerSignal`'s confidence — already
    dampened by its owning manager, since a bare `AgentSignal` (pre-
    dampening) is never persisted past `ManagerAgent.evaluate()` — labeled
    accordingly in the UI rather than implying it's the analyst's raw
    output. The manager layer aggregates by mean confidence and majority
    direction across the analysts each manager owns for this asset/day."""
    decisions = [
        decision for decision in decisions_by_day.get(as_of_date, []) if decision.asset == asset
    ]
    signals: list[ManagerSignal] = [
        signal for decision in decisions for signal in decision.contributing_signals
    ]
    if not signals:
        return AgentFlowSnapshot(
            analysts=[], managers=[], commander=None, majority_direction="flat"
        )

    majority = _majority_direction([signal.direction for signal in signals])

    analyst_nodes = [
        AgentFlowNode(
            name=signal.source_agent_id,
            direction=signal.direction,
            confidence=signal.confidence,
            agrees_with_majority=signal.direction == majority,
            status=signal.status,
        )
        for signal in sorted(signals, key=lambda s: s.source_agent_id)
    ]

    by_manager: dict[str, list[ManagerSignal]] = {}
    for signal in signals:
        by_manager.setdefault(signal.manager_id, []).append(signal)
    manager_nodes = [
        AgentFlowNode(
            name=manager_id,
            direction=_majority_direction([s.direction for s in owned]),
            confidence=sum(s.confidence for s in owned) / len(owned),
            agrees_with_majority=_majority_direction([s.direction for s in owned]) == majority,
            status=(
                "error"
                if any(s.status == "error" for s in owned)
                else "unavailable"
                if any(s.status == "unavailable" for s in owned)
                else "active"
            ),
        )
        for manager_id, owned in sorted(by_manager.items())
    ]

    commander_node = None
    if decisions:
        target_weight = decisions[0].target_weight
        commander_direction = (
            "long" if target_weight > 0 else "short" if target_weight < 0 else "flat"
        )
        commander_node = AgentFlowNode(
            name="Commander",
            direction=commander_direction,
            confidence=min(abs(target_weight), 1.0),
            agrees_with_majority=commander_direction == majority,
        )

    return AgentFlowSnapshot(
        analysts=analyst_nodes,
        managers=manager_nodes,
        commander=commander_node,
        majority_direction=majority,
    )


def _node_css_class(node: AgentFlowNode) -> str:
    if node.status == "error":
        return "agent-node-error"
    if node.status == "unavailable":
        return "agent-node-unavailable"
    if node.direction == "flat":
        return "agent-node-flat"
    return "agent-node-agree" if node.agrees_with_majority else "agent-node-disagree"


def _render_flow_node(node: AgentFlowNode) -> None:
    st.markdown(
        f'<div class="agent-node {_node_css_class(node)}">'
        f'<span class="agent-node-name">{node.name}</span>'
        f"{node.direction.upper()} · {node.confidence:.2f} · {node.status}</div>",
        unsafe_allow_html=True,
    )


def _render_agent_flow_view(desk_run: DeskRun, as_of_date: date) -> None:
    """The four-layer pipeline (Analysts → Pod Managers → Commander →
    optional Portfolio Construction) for one asset on the current playback
    day, each node colored by whether it agrees with that day's
    majority-vote direction — this is what makes "many heterogeneous
    agents debating" visible rather than just asserted in prose."""
    st.markdown("### Agent flow")
    st.markdown(
        '<div class="section-caption">Every contributing signal for one asset, on the '
        "currently-selected playback day, grouped by pipeline layer. A node's color shows "
        "whether it agrees with that day's majority-vote direction across every analyst "
        "signal for this asset — not a claim about which side is actually right."
        "</div>",
        unsafe_allow_html=True,
    )
    if not desk_run.tickers:
        st.info("No tickers in this run.")
        return
    asset = st.selectbox("Asset", options=desk_run.tickers, key="agent_flow_asset")
    snapshot = _build_agent_flow_snapshot(desk_run.decisions_by_day, asset, as_of_date)
    if not snapshot.analysts:
        st.info(f"No contributing signals recorded for {asset} on {as_of_date.isoformat()}.")
        return

    col_analysts, col_managers, col_commander, col_l4 = st.columns(4)
    with col_analysts:
        st.markdown(
            '<div class="flow-layer-label">Layer 1 · Analysts</div>', unsafe_allow_html=True
        )
        for node in snapshot.analysts:
            _render_flow_node(node)
    with col_managers:
        st.markdown(
            '<div class="flow-layer-label">Layer 2 · Pod managers</div>', unsafe_allow_html=True
        )
        for node in snapshot.managers:
            _render_flow_node(node)
    with col_commander:
        st.markdown(
            '<div class="flow-layer-label">Layer 3 · Commander</div>', unsafe_allow_html=True
        )
        if snapshot.commander is not None:
            _render_flow_node(snapshot.commander)
    with col_l4:
        st.markdown(
            '<div class="flow-layer-label">Layer 4 · Portfolio construction</div>',
            unsafe_allow_html=True,
        )
        if desk_run.portfolio_construction_method is not None:
            st.markdown(
                '<div class="agent-node agent-node-flat">'
                f'<span class="agent-node-name">{desk_run.portfolio_construction_method}'
                "</span>enabled — reshapes weights across the whole basket, not just "
                f"{asset}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="agent-node agent-node-flat">'
                '<span class="agent-node-name">Off</span>Commander\'s decision passes '
                "through unchanged</div>",
                unsafe_allow_html=True,
            )


def _trust_panel_frame(desk_run: DeskRun, source_ids: list[str], as_of_date: date) -> pd.DataFrame:
    """One row per analyst/manager: trailing Sharpe and cumulative
    contribution from `scoring/scoreboard.py`, side by side with its
    current scoring-derived portfolio weight from the same `Hedge`
    weighting strategy `ManagerAgent.evaluate()`/`Commander.decide()`
    actually use — so the panel shows the same weight the backtest itself
    applied, not a separately-reconstructed approximation of it."""
    if not source_ids:
        return pd.DataFrame(columns=["source", "weight", "sharpe_30d", "contribution_%"])

    weights = Hedge().get_weights(desk_run.scoreboard, source_ids, as_of_date)
    rows = []
    for source_id in source_ids:
        try:
            sharpe = desk_run.scoreboard.sharpe_ratio(source_id, as_of_date, 30)
        except ValueError:
            sharpe = 0.0
        series = _cumulative_pct_series(desk_run.track_record, source_id)
        as_of_series = _clip_series_to_date(series, as_of_date)
        contribution_pct = float(as_of_series.iloc[-1]) if not as_of_series.empty else 0.0
        rows.append(
            {
                "source": source_id,
                "weight": weights.get(source_id, 1.0),
                "sharpe_30d": sharpe,
                "contribution_%": contribution_pct,
            }
        )
    return pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)


def _render_trust_panel(
    desk_run: DeskRun, source_ids: list[str], as_of_date: date, label_plural: str
) -> None:
    """Trailing performance next to current scoring-derived weight, for
    every source in `source_ids` — makes it visually obvious that trust in
    this project is earned from a measured track record, never hardcoded
    by agent type (see `scoring/weighting.py`)."""
    st.markdown(f"#### {label_plural} trust panel")
    st.markdown(
        '<div class="section-caption">Trailing 30-day Sharpe and contribution next to the '
        "same scoring-derived weight this backtest actually applied that day — weight "
        "1.0 is average; above/below reflects out/underperformance relative to peers."
        "</div>",
        unsafe_allow_html=True,
    )
    frame = _trust_panel_frame(desk_run, source_ids, as_of_date)
    if frame.empty:
        st.info("No track record yet for this source group.")
        return
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "weight": st.column_config.ProgressColumn(
                "Weight",
                min_value=0.0,
                max_value=max(2.0, float(frame["weight"].max())),
                format="%.2f",
            ),
            "sharpe_30d": st.column_config.NumberColumn("Sharpe (30d)", format="%.2f"),
            "contribution_%": st.column_config.NumberColumn("Contribution %", format="%.2f"),
        },
    )


_KNOWN_DATA_SOURCES: tuple[str, ...] = (
    "fred",
    "finnhub_news",
    "sec_edgar",
    "sec_fulltext",
    "finra_short_interest",
    "fama_french",
    "tiingo",
    "alpha_vantage",
    "fmp",
    "google_trends",
    "wikimedia_pageviews",
    "reddit",
    "cboe",
)


def _data_source_status_rows(
    fetch_log: pd.DataFrame, known_sources: tuple[str, ...], today: date
) -> list[dict[str, str]]:
    """Per-source status for the strip: "live" if that source's most
    recent recorded fetch happened today (real wall-clock date, matching
    how `data/cache.py` keys most of these sources), "cached" if it has a
    fetch on record from an earlier day, "unused" if this run never
    touched it at all. Pure function over an already-loaded fetch log
    frame so it's testable without a real SQLite file."""
    latest_by_source: dict[str, str] = {}
    if not fetch_log.empty:
        for source, group in fetch_log.groupby("source"):
            latest_by_source[str(source)] = str(group["as_of_date"].max())

    rows = []
    for source in known_sources:
        latest = latest_by_source.get(source)
        if latest is None:
            status = "unused"
        elif latest == today.isoformat():
            status = "live"
        else:
            status = "cached"
        rows.append({"source": source, "status": status, "last_fetched": latest or "—"})
    return rows


def _render_data_source_status_strip(desk_run: DeskRun) -> None:
    """Which external sources this run actually touched, and whether each
    was a fresh live fetch today or served from an earlier cache entry —
    surfaces the multi-source data layer (Task 1) at a glance instead of
    leaving it implicit in the README."""
    st.markdown("#### Data sources")
    price_status = "live" if "live" in desk_run.source_label else "fallback"
    price_class = {"live": "status-pill-live", "fallback": "status-pill-fallback"}[price_status]
    st.markdown(
        f'<span class="status-pill {price_class}">market data: {desk_run.source_label}</span>',
        unsafe_allow_html=True,
    )

    try:
        fetch_log = read_fetch_log(CACHE_ROOT / "fetch_log.db")
    except Exception:
        logger.debug("Could not read the fetch log for the data-source status strip", exc_info=True)
        fetch_log = pd.DataFrame(columns=["source", "identifier", "as_of_date"])

    rows = _data_source_status_rows(fetch_log, _KNOWN_DATA_SOURCES, date.today())
    pill_class = {
        "live": "status-pill-live",
        "cached": "status-pill-cached",
        "unused": "status-pill-unused",
    }
    badges = "".join(
        f'<span class="status-pill {pill_class[row["status"]]}" '
        f'title="last fetched: {row["last_fetched"]}">{row["source"]}: {row["status"]}</span>'
        for row in rows
    )
    st.markdown(badges, unsafe_allow_html=True)


@st.dialog("Signal & trade history", width="large")
def _show_source_history_dialog(source_id: str, id_field: str, label_singular: str) -> None:
    """Drill-down modal: every signal `source_id` produced, split into one
    tab per asset it touched, so "when it bought and when it sold for all
    the different stocks" is answerable at a glance per stock rather than
    one long mixed table. Reads the loaded desk_run from session_state
    (the standard way to parameterize a Streamlit dialog, since the dialog
    function itself is called with no way to smuggle extra render-time
    context past what its own arguments carry)."""
    desk_run: DeskRun | None = st.session_state.get("desk_run")
    st.markdown(f"#### {label_singular.capitalize()}: `{source_id}`")
    if desk_run is None:
        st.write("No simulation state is loaded yet.")
        return

    history = _signal_history_rows(
        desk_run.decisions_by_day,
        source_id,
        id_field,
        price_history=desk_run.price_history,
        track_record=desk_run.track_record,
    )
    if history.empty:
        st.write("No recorded signals for this source yet.")
        return

    assets = sorted(history["asset"].unique())
    display_columns = [
        "date",
        "direction",
        "confidence",
        "scale_applied",
        "target_weight",
        "realized_log_return",
        "cumulative_contribution_pct",
        "rationale",
    ]
    if len(assets) == 1:
        st.dataframe(history[display_columns].sort_values("date"), width="stretch", hide_index=True)
        return

    for asset, tab in zip(assets, st.tabs(assets), strict=True):
        with tab:
            asset_history = history[history["asset"] == asset][display_columns].sort_values("date")
            st.dataframe(asset_history, width="stretch", hide_index=True)


def _render_overview_tab() -> None:
    """The playback surface: current-day metrics, the portfolio-vs-benchmark
    chart, day navigation (step/scrub/play), and that day's trade detail.
    This whole function is the body of a fragment (see render_dashboard())
    so autoplay can tick it on a timer without blocking the rest of the app —
    the sidebar, tabs, and every other control stay clickable throughout."""
    desk_run: DeskRun | None = st.session_state.get("desk_run")
    backtest_result: BacktestResult | None = st.session_state.get("backtest_result")
    if backtest_result is None or desk_run is None:
        st.info("No backtest result is loaded yet.")
        return

    last_day_index = max(len(backtest_result.portfolio_value_history) - 1, 0)
    is_playing = st.session_state.get("is_playing", False)
    current_day_index = st.session_state.get("current_day_index", 0)

    # Distinguishes a run_every autoplay tick from a rerun caused by a manual
    # click (on_click/on_change callbacks set this flag before the fragment
    # body reruns) — skips the auto-tick below when the manual click already
    # applied this run's day change.
    manual_nav_happened = st.session_state.get("_manual_nav_this_run", False)
    st.session_state["_manual_nav_this_run"] = False
    if is_playing:
        if current_day_index >= last_day_index:
            st.session_state["is_playing"] = False
            is_playing = False
            st.rerun()  # full rerun: stop the fragment's run_every ticking
        elif not manual_nav_happened:
            _step_day(1)
            current_day_index = st.session_state["current_day_index"]

    current_date, current_value = _current_day(backtest_result, current_day_index)
    previous_value = (
        current_value
        if current_day_index == 0
        else backtest_result.portfolio_value_history[current_day_index - 1][1]
    )
    daily_change = current_value - previous_value
    daily_change_pct = 0.0 if previous_value == 0 else (daily_change / previous_value) * 100.0

    with st.container(border=True):
        metric_left, metric_mid, metric_right, metric_far = st.columns(4)
        metric_left.metric(
            "Portfolio value",
            f"${current_value:,.2f}",
            f"{daily_change:,.2f} / {daily_change_pct:.2f}%",
        )
        metric_mid.metric("Sharpe", f"{backtest_result.final_sharpe_ratio:.2f}")
        metric_right.metric("Max drawdown", f"{backtest_result.final_max_drawdown:,.2f}")
        metric_far.metric("Desk date", current_date.isoformat())

        nav_back, nav_fwd, nav_play, nav_note = st.columns([1, 1, 1, 5])
        with nav_back:
            st.button("Step backward", on_click=_manual_step_day, args=(-1,), width="stretch")
        with nav_fwd:
            st.button("Step forward", on_click=_manual_step_day, args=(1,), width="stretch")
        with nav_play:
            if st.button(
                "Pause" if is_playing else "Play",
                width="stretch",
                type="primary" if is_playing else "secondary",
            ):
                st.session_state["is_playing"] = not is_playing
                st.rerun()  # full rerun: (re)wire the fragment's run_every
        with nav_note:
            st.markdown(
                "<div class='mini-note'>Basket: "
                f"<strong>{desk_run.group_name}</strong> ({', '.join(desk_run.tickers)}) • "
                f"Day {current_day_index + 1} of {last_day_index + 1}</div>",
                unsafe_allow_html=True,
            )

        if last_day_index > 0:
            # No value= here: _current_day() always seeds "day_scrubber" into
            # session_state (via _set_current_day_index) before this widget
            # is created, so also passing value= just triggers Streamlit's
            # conflicting-default warning for no benefit.
            st.slider(
                "Scrub to day",
                min_value=0,
                max_value=last_day_index,
                key="day_scrubber",
                on_change=lambda: _manual_jump_to_day(st.session_state["day_scrubber"]),
                label_visibility="collapsed",
            )

    with st.container(border=True):
        _render_data_source_status_strip(desk_run)

    with st.container(border=True):
        _render_agent_flow_view(desk_run, current_date)

    st.markdown("### Portfolio vs. benchmarks")
    st.markdown(
        '<div class="section-caption">Cumulative % return since the start of the window — '
        "the desk against the S&P 500 and a passive, equal-weighted hold of its own basket."
        "</div>",
        unsafe_allow_html=True,
    )
    initial_cash = float(st.session_state.get("initial_cash", 10_000.0))
    portfolio_series = _portfolio_pct_series(backtest_result.portfolio_value_history, initial_cash)
    # Clip every series to the current playback day so this chart grows as
    # the user steps/scrubs/plays forward, rather than always showing the
    # full precomputed window — this is what makes playback feel live.
    chart_frame = _combined_chart_frame(
        {"Portfolio": _clip_series_to_date(portfolio_series, current_date)},
        _clip_series_to_date(desk_run.sp500_benchmark, current_date),
        _clip_series_to_date(desk_run.basket_benchmark, current_date),
    )
    if chart_frame.empty:
        st.info("Not enough data yet to plot a performance curve.")
    else:
        st.line_chart(chart_frame, height=320)

    action_left, action_right = st.columns([1, 1])
    with action_left:
        st.markdown("### Current day action")
        if desk_run.trade_blotter.empty:
            st.write("No trade blotter is available for the loaded result.")
        else:
            day_blotter = desk_run.trade_blotter[desk_run.trade_blotter["date"] == current_date]
            if day_blotter.empty:
                st.write("No trades on this day.")
            else:
                st.dataframe(
                    day_blotter[
                        [
                            "asset",
                            "action",
                            "target_weight",
                            "delta_shares",
                            "notional",
                            "manager_ids",
                            "analyst_ids",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

    with action_right:
        st.markdown("### Signal breakdown")
        day_decisions = desk_run.decisions_by_day.get(current_date, [])
        if not day_decisions:
            st.write("No contributing signals recorded for this day.")
        else:
            signal_rows = [
                {
                    "asset": decision.asset,
                    "target_weight": decision.target_weight,
                    "manager": signal.manager_id,
                    "analyst": signal.source_agent_id,
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "rationale": signal.rationale,
                }
                for decision in day_decisions
                for signal in decision.contributing_signals
            ]
            st.dataframe(pd.DataFrame(signal_rows), width="stretch", hide_index=True)

    with st.expander("Holdings and full trade blotter", expanded=False):
        holdings_left, holdings_right = st.columns([0.9, 1.1])
        with holdings_left:
            st.markdown("#### Holdings as of this day")
            if not desk_run.trade_blotter.empty:
                current_rows = desk_run.trade_blotter[
                    desk_run.trade_blotter["date"] == current_date
                ]
                if current_rows.empty:
                    st.write("No holdings to display for this day.")
                else:
                    holdings = current_rows[
                        ["asset", "action", "target_weight", "notional", "manager_ids"]
                    ].copy()
                    st.dataframe(holdings, width="stretch", hide_index=True)
            else:
                st.write("No simulation state is loaded yet.")

        with holdings_right:
            st.markdown("#### Full trade blotter")
            if desk_run.trade_blotter.empty:
                st.write("No trade blotter available.")
            else:
                st.dataframe(desk_run.trade_blotter, width="stretch", hide_index=True)


def _render_performance_tab(
    *,
    label_plural: str,
    label_singular: str,
    source_ids: list[str],
    id_field: str,
    multiselect_key: str,
    table_key: str,
) -> None:
    """Shared body for the Desks and Analysts tabs: a toggleable multi-line
    performance chart against both benchmarks, a summary table, and a
    click-a-row drill-down (opens `_show_source_history_dialog`) into one
    source's recorded decision-making history. Purely a read/inspect
    surface — nothing here changes which desks or analysts run, per the
    dashboard's own "not a control panel" design."""
    desk_run: DeskRun | None = st.session_state.get("desk_run")
    if desk_run is None or not source_ids:
        st.info("No simulation state is loaded yet.")
        return

    st.markdown(f"### {label_plural} performance")
    st.markdown(
        '<div class="section-caption">Cumulative attributable contribution, as a % of '
        f"starting value — each {label_singular}'s own share of portfolio performance, "
        "not an absolute return.</div>",
        unsafe_allow_html=True,
    )

    current_date_for_trust = _current_playback_date()
    with st.expander(f"{label_plural} trust panel", expanded=True):
        _render_trust_panel(
            desk_run,
            source_ids,
            current_date_for_trust if current_date_for_trust is not None else date.today(),
            label_plural,
        )

    all_series = {sid: _cumulative_pct_series(desk_run.track_record, sid) for sid in source_ids}

    visible = st.multiselect(
        f"Show {label_plural.lower()}",
        options=source_ids,
        default=source_ids,
        key=multiselect_key,
    )
    # Clip every series (including both benchmarks) to the current playback
    # day, same as the Overview tab, so this chart also grows with playback
    # rather than always showing the full precomputed window.
    current_date = _current_playback_date()
    if current_date is None:
        sp500_benchmark = desk_run.sp500_benchmark
        basket_benchmark = desk_run.basket_benchmark
    else:
        sp500_benchmark = _clip_series_to_date(desk_run.sp500_benchmark, current_date)
        basket_benchmark = _clip_series_to_date(desk_run.basket_benchmark, current_date)

    clipped_series = {
        sid: (
            all_series[sid]
            if current_date is None
            else _clip_series_to_date(all_series[sid], current_date)
        )
        for sid in visible
    }
    chart_frame = _combined_chart_frame(clipped_series, sp500_benchmark, basket_benchmark)
    holdings_by_series = {
        sid: _holdings_by_day(desk_run.decisions_by_day, sid, id_field) for sid in visible
    }
    _render_altair_performance_chart(chart_frame, holdings_by_series)

    st.markdown(f"#### {label_plural} summary")
    st.markdown(
        '<div class="section-caption">Click a row to drill into that '
        f"{label_singular}'s full signal & trade history.</div>",
        unsafe_allow_html=True,
    )
    summary = _source_summary_frame(desk_run, source_ids, id_field, label_singular)
    selection = st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=table_key,
    )
    selected_rows = getattr(getattr(selection, "selection", None), "rows", [])
    if selected_rows:
        selected_source_id = str(summary.iloc[selected_rows[0]][label_singular])
        _show_source_history_dialog(selected_source_id, id_field, label_singular)


def _render_tabbed_content(desk_run: DeskRun | None) -> None:
    """Creates the three tabs and renders all of them. This whole function
    is the body of the run_every-wired fragment created in render_dashboard()
    — not just the Overview tab. Desks/Analysts must live inside the same
    fragment as Overview: a fragment-scoped rerun (the autoplay timer tick,
    or a Step/scrub click inside Overview) only re-executes code inside that
    fragment, so rendering Desks/Analysts outside it would leave their
    playback-clipped charts frozen during autoplay.

    Re-seeds session-state defaults on every call: a fragment-scoped rerun
    re-executes only this function, not render_dashboard() above it, so
    defaults must be available here independently."""
    _seed_defaults()
    tab_overview, tab_desks, tab_analysts = st.tabs(["📊 Overview", "🗂️ Desks", "🔎 Analysts"])

    with tab_overview:
        _render_overview_tab()

    with tab_desks:
        _render_performance_tab(
            label_plural="Desks",
            label_singular="desk",
            source_ids=desk_run.manager_ids if desk_run is not None else [],
            id_field="manager_id",
            multiselect_key="visible_desks",
            table_key="desks_summary_table",
        )

    with tab_analysts:
        _render_performance_tab(
            label_plural="Analysts",
            label_singular="analyst",
            source_ids=desk_run.analyst_ids if desk_run is not None else [],
            id_field="source_agent_id",
            multiselect_key="visible_analysts",
            table_key="analysts_summary_table",
        )


def render_dashboard() -> None:
    """Dark-mode trading desk: a tabbed view (Overview / Desks / Analysts)
    over a multi-asset backtest, benchmarked against the S&P 500 and a
    passive hold of the selected ticker basket throughout."""
    # No-op after the first call in this process (see configure_logging()'s
    # docstring) — safe to call unconditionally even though render_dashboard()
    # re-executes on every Streamlit rerun. Uses the CLI's same defaults
    # (enabled, not verbose); there's no per-session CLI-flag equivalent in
    # the GUI, so verbosity here isn't user-adjustable from the sidebar.
    configure_logging()
    st.set_page_config(
        page_title="Agentic Trading Desk",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    _seed_defaults()

    with st.sidebar:
        st.markdown("## Trading desk")
        st.caption("Configure the desk and rerun the simulation in-place.")

        st.markdown("#### Universe")
        group_names = [group.name for group in TICKER_GROUPS]
        selected_group_name = st.selectbox(
            "Ticker group", group_names, index=0, key="selected_group"
        )
        selected_group: TickerGroup = next(
            group for group in TICKER_GROUPS if group.name == selected_group_name
        )
        st.caption(selected_group.description)

        # The group selectbox is only a starting point — the multiselect
        # below is what the rest of the app actually reads. Keying it per
        # group (rather than one shared key) means switching groups resets
        # it to that group's own defaults, while switching back to a group
        # customized earlier in this session remembers that customization,
        # since each group's edits live under their own key. The
        # setdefault() below must run before the multiselect widget is
        # instantiated in this script run — writing to a widget's key after
        # it's been created raises StreamlitAPIException.
        ticker_multiselect_key = f"ticker_multiselect__{selected_group_name}"
        st.session_state.setdefault(ticker_multiselect_key, list(selected_group.symbols))
        all_stock_symbols = [info.symbol for info in tickers_by_asset_class("stock")]
        selected_tickers = st.multiselect(
            "Tickers",
            options=all_stock_symbols,
            key=ticker_multiselect_key,
        )

        st.markdown("#### Backtest window")
        start_date = st.date_input("Start date", value=date(2024, 1, 2), key="start_date")
        end_date = st.date_input("End date", value=date(2024, 3, 1), key="end_date")
        initial_cash = st.number_input(
            "Initial cash",
            min_value=1_000.0,
            max_value=1_000_000.0,
            value=10_000.0,
            step=500.0,
            key="initial_cash",
        )

        st.markdown("#### Desk configuration")
        manager_mode = st.radio(
            "Manager stack",
            options=MANAGER_MODES,
            index=0,
            key="manager_mode",
        )

        portfolio_construction_enabled = st.checkbox(
            "Portfolio construction (Layer 4)",
            value=False,
            key="portfolio_construction_enabled",
            help="Optional post-Commander stage: Black-Litterman views from the Commander's "
            "own decisions, a risk-budget optimizer, then Kelly leverage scaling. Off by "
            "default, same as the CLI's --portfolio-construction flag.",
        )
        portfolio_construction_method = st.selectbox(
            "Risk budget method",
            options=("risk_parity", "min_variance", "mean_variance"),
            index=0,
            key="portfolio_construction_method",
            disabled=not portfolio_construction_enabled,
        )

        st.divider()
        run_clicked = st.button("Run simulation", width="stretch", type="primary")
        reset_clicked = st.button("Reset desk", width="stretch")

    if reset_clicked:
        _reset_desk()
        st.rerun()

    if start_date > end_date:
        st.warning("Start date is after end date; swapping the range for this run.")
        start_date, end_date = end_date, start_date

    tickers = list(selected_tickers)
    if not tickers:
        st.warning("Select at least one ticker in the sidebar before running a simulation.")

    # Only ever runs on an explicit click — never on page load and never as
    # a side effect of any other widget interaction (ticker/date/manager
    # changes just update what a *future* click will run against).
    if run_clicked and tickers:
        logger.info(
            "GUI-initiated backtest requested: tickers=%s, group=%s, %s to %s, manager_mode=%s",
            tickers,
            selected_group.name,
            start_date,
            end_date,
            manager_mode,
        )
        new_run = _run_simulation_with_progress(
            tickers,
            selected_group.name,
            start_date,
            end_date,
            float(initial_cash),
            manager_mode,
            portfolio_construction_method=(
                portfolio_construction_method if portfolio_construction_enabled else None
            ),
        )
        st.session_state["desk_run"] = new_run
        st.session_state["backtest_result"] = new_run.backtest_result
        st.session_state["scoreboard"] = new_run.scoreboard
        st.session_state["is_playing"] = False
        _set_current_day_index(0)

    desk_run: DeskRun | None = st.session_state.get("desk_run")

    if desk_run is not None:
        portfolio_construction_caption = (
            f"Portfolio construction: {desk_run.portfolio_construction_method} "
            if desk_run.portfolio_construction_method
            else "Portfolio construction: off "
        )
        st.caption(
            f"Simulation source: {desk_run.source_label} • Manager stack: {desk_run.manager_mode} "
            f"• {portfolio_construction_caption}"
            f"• Basket: {desk_run.group_name}"
        )

    if st.session_state.get("backtest_result") is None:
        st.info("Configure the desk in the sidebar and click **Run simulation** to begin.")
        return

    is_playing = st.session_state.get("is_playing", False)
    tabs_fragment = st.fragment(run_every=AUTOPLAY_INTERVAL_SECONDS if is_playing else None)(
        _render_tabbed_content
    )
    tabs_fragment(desk_run)


def main() -> None:
    render_dashboard()


if __name__ == "__main__":
    render_dashboard()
