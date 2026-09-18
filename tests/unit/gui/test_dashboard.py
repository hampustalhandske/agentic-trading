"""Unit tests for agentic_trading.gui.dashboard.

Two layers of coverage, mirroring how the dashboard itself is split:

1. Pure-function tests (no Streamlit runtime) for the benchmark/series/chart
   helpers introduced for the tabbed, multi-asset, benchmarked dashboard —
   fast, precise, and the right place to pin down the actual math.
2. A small set of `AppTest`-based integration tests for session-state
   behavior (day navigation, autoplay toggling, tab rendering) that can only
   be observed by actually running the script.

For (2), the seed fixture is a full `DeskRun`, not just a `BacktestResult` —
unlike the single-ticker, single-view predecessor of this dashboard, every
tab here (Overview/Desks/Analysts) reads `st.session_state["desk_run"]`
directly (for price history, benchmarks, track record, decisions_by_day),
so pre-seeding only `backtest_result`/`scoreboard` no longer exercises any
tab's real content — see render_dashboard()'s own `desk_run is None` guard.
"""

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Button, Multiselect

import agentic_trading.gui.dashboard as dashboard_module
from agentic_trading.agents.schemas import (
    BacktestResult,
    ManagerSignal,
    PerformanceRecord,
    PortfolioDecision,
)
from agentic_trading.scoring.scoreboard import PerformanceScoreboard
from agentic_trading.scoring.track_record import TrackRecord

# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_normalized_return_pct_rebases_to_zero_at_first_observation() -> None:
    close = pd.Series([100.0, 110.0, 90.0])

    result = dashboard_module._normalized_return_pct(close)

    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] == pytest.approx(10.0)
    assert result.iloc[2] == pytest.approx(-10.0)


def test_normalized_return_pct_returns_empty_series_for_all_nan_input() -> None:
    close = pd.Series([float("nan"), float("nan")])

    result = dashboard_module._normalized_return_pct(close)

    assert result.empty


def test_normalized_return_pct_returns_empty_series_when_base_is_zero() -> None:
    close = pd.Series([0.0, 5.0, 10.0])

    result = dashboard_module._normalized_return_pct(close)

    assert result.empty


def test_portfolio_pct_series_matches_hand_computed_percentages() -> None:
    history = [(date(2024, 1, 2), 10_000.0), (date(2024, 1, 3), 11_000.0)]

    result = dashboard_module._portfolio_pct_series(history, initial_cash=10_000.0)

    assert result.loc[pd.Timestamp(date(2024, 1, 2))] == pytest.approx(0.0)
    assert result.loc[pd.Timestamp(date(2024, 1, 3))] == pytest.approx(10.0)


def test_portfolio_pct_series_empty_for_empty_history() -> None:
    assert dashboard_module._portfolio_pct_series([], initial_cash=10_000.0).empty


def test_cumulative_pct_series_sums_within_a_day_and_across_days() -> None:
    """A source posting on two assets the same day must be summed, then the
    running total carried forward cumulatively across subsequent days."""
    track_record = TrackRecord()
    day1, day2 = date(2024, 1, 1), date(2024, 1, 2)
    track_record.record(
        PerformanceRecord(
            source_id="analyst-1",
            source_type="analyst",
            asset="AAA",
            as_of_date=day1,
            attributable_return=0.01,
            raw_attributable_return=0.01,
        )
    )
    track_record.record(
        PerformanceRecord(
            source_id="analyst-1",
            source_type="analyst",
            asset="BBB",
            as_of_date=day1,
            attributable_return=0.02,
            raw_attributable_return=0.02,
        )
    )
    track_record.record(
        PerformanceRecord(
            source_id="analyst-1",
            source_type="analyst",
            asset="AAA",
            as_of_date=day2,
            attributable_return=-0.01,
            raw_attributable_return=-0.01,
        )
    )

    result = dashboard_module._cumulative_pct_series(track_record, "analyst-1")

    assert result.loc[pd.Timestamp(day1)] == pytest.approx(3.0)
    assert result.loc[pd.Timestamp(day2)] == pytest.approx(2.0)


def test_cumulative_pct_series_empty_for_unknown_source() -> None:
    track_record = TrackRecord()

    assert dashboard_module._cumulative_pct_series(track_record, "never-seen").empty


def test_combined_chart_frame_includes_benchmarks_when_present() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    primary = {"Portfolio": pd.Series([0.0, 1.0], index=index)}
    sp500 = pd.Series([0.0, 0.5], index=index)
    basket = pd.Series([0.0, 0.8], index=index)

    frame = dashboard_module._combined_chart_frame(primary, sp500, basket)

    assert set(frame.columns) == {"Portfolio", "S&P 500", "Basket index"}


def test_combined_chart_frame_omits_empty_benchmarks() -> None:
    index = pd.to_datetime(["2024-01-02"])
    primary = {"Portfolio": pd.Series([0.0], index=index)}

    frame = dashboard_module._combined_chart_frame(
        primary, pd.Series(dtype=float), pd.Series(dtype=float)
    )

    assert list(frame.columns) == ["Portfolio"]


def test_combined_chart_frame_empty_when_nothing_to_plot() -> None:
    frame = dashboard_module._combined_chart_frame(
        {}, pd.Series(dtype=float), pd.Series(dtype=float)
    )

    assert frame.empty


def _signal(manager_id: str, source_agent_id: str) -> ManagerSignal:
    return ManagerSignal(
        manager_id=manager_id,
        source_agent_id=source_agent_id,
        asset="AAA",
        as_of_date=date(2024, 1, 2),
        direction="long",
        confidence=0.5,
        raw_confidence=0.5,
        scale_applied=1.0,
        rationale="test rationale",
    )


def test_signal_history_rows_filters_by_manager_id() -> None:
    decision = PortfolioDecision(
        asset="AAA",
        as_of_date=date(2024, 1, 2),
        target_weight=0.1,
        contributing_signals=[
            _signal("manager-1", "analyst-1"),
            _signal("manager-2", "analyst-1"),
        ],
    )
    decisions_by_day = {date(2024, 1, 2): [decision]}

    rows = dashboard_module._signal_history_rows(decisions_by_day, "manager-1", "manager_id")

    assert len(rows) == 1
    assert rows.iloc[0]["manager"] == "manager-1"
    assert rows.iloc[0]["target_weight"] == pytest.approx(0.1)
    # No price_history/track_record supplied — enriched columns exist but
    # carry no value.
    assert rows.iloc[0]["realized_log_return"] is None
    assert rows.iloc[0]["cumulative_contribution_pct"] is None


def test_signal_history_rows_filters_by_source_agent_id() -> None:
    decision = PortfolioDecision(
        asset="AAA",
        as_of_date=date(2024, 1, 2),
        target_weight=0.1,
        contributing_signals=[
            _signal("manager-1", "analyst-1"),
            _signal("manager-1", "analyst-2"),
        ],
    )
    decisions_by_day = {date(2024, 1, 2): [decision]}

    rows = dashboard_module._signal_history_rows(decisions_by_day, "analyst-2", "source_agent_id")

    assert len(rows) == 1
    assert rows.iloc[0]["analyst"] == "analyst-2"


def test_signal_history_rows_empty_for_unmatched_source() -> None:
    decisions_by_day = {
        date(2024, 1, 2): [
            PortfolioDecision(
                asset="AAA",
                as_of_date=date(2024, 1, 2),
                target_weight=0.1,
                contributing_signals=[_signal("manager-1", "analyst-1")],
            )
        ]
    }

    rows = dashboard_module._signal_history_rows(decisions_by_day, "never-seen", "manager_id")

    assert rows.empty


def _multiindex_price_history(
    asset: str, day: date, close: float, log_return: float | None
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"close": [close], "log_return": [log_return]}, index=pd.to_datetime([day])
    )
    return pd.concat({asset: frame}, axis=1)


def test_signal_history_rows_enriches_with_realized_return_and_cumulative_pct_when_supplied() -> (
    None
):
    """When `price_history`/`track_record` are supplied, each row is
    enriched with that day's realized log return for the asset and the
    source's own cumulative contribution % as of that day — reusing
    `_cumulative_pct_series` rather than re-summing the math a second way."""
    decision = PortfolioDecision(
        asset="AAA",
        as_of_date=date(2024, 1, 2),
        target_weight=0.3,
        contributing_signals=[_signal("manager-1", "analyst-1")],
    )
    decisions_by_day = {date(2024, 1, 2): [decision]}
    price_history = _multiindex_price_history("AAA", date(2024, 1, 2), 10.0, 0.05)
    track_record = TrackRecord()
    track_record.record(
        PerformanceRecord(
            source_id="analyst-1",
            source_type="analyst",
            asset="AAA",
            as_of_date=date(2024, 1, 2),
            attributable_return=0.02,
            raw_attributable_return=0.02,
        )
    )

    rows = dashboard_module._signal_history_rows(
        decisions_by_day,
        "analyst-1",
        "source_agent_id",
        price_history=price_history,
        track_record=track_record,
    )

    assert rows.iloc[0]["target_weight"] == pytest.approx(0.3)
    assert rows.iloc[0]["realized_log_return"] == pytest.approx(0.05)
    assert rows.iloc[0]["cumulative_contribution_pct"] == pytest.approx(2.0)


def test_signal_history_rows_realized_return_none_when_day_missing_from_price_history() -> None:
    decision = PortfolioDecision(
        asset="AAA",
        as_of_date=date(2024, 1, 2),
        target_weight=0.1,
        contributing_signals=[_signal("manager-1", "analyst-1")],
    )
    decisions_by_day = {date(2024, 1, 2): [decision]}
    price_history = _multiindex_price_history("AAA", date(2024, 1, 3), 10.0, 0.05)

    rows = dashboard_module._signal_history_rows(
        decisions_by_day, "analyst-1", "source_agent_id", price_history=price_history
    )

    assert rows.iloc[0]["realized_log_return"] is None


def test_clip_series_to_date_keeps_observations_on_or_before_cutoff() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    series = pd.Series([1.0, 2.0, 3.0], index=index)

    clipped = dashboard_module._clip_series_to_date(series, date(2024, 1, 3))

    assert list(clipped.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert list(clipped.to_numpy()) == [1.0, 2.0]


def test_clip_series_to_date_drops_everything_before_the_first_observation() -> None:
    index = pd.to_datetime(["2024-01-05", "2024-01-06"])
    series = pd.Series([1.0, 2.0], index=index)

    clipped = dashboard_module._clip_series_to_date(series, date(2024, 1, 2))

    assert clipped.empty


def test_clip_series_to_date_returns_empty_for_empty_input() -> None:
    result = dashboard_module._clip_series_to_date(pd.Series(dtype=float), date(2024, 1, 2))

    assert result.empty


def test_holdings_by_day_builds_compact_string_for_matching_signals() -> None:
    day = date(2024, 1, 2)
    decisions = [
        PortfolioDecision(
            asset="AAA",
            as_of_date=day,
            target_weight=0.1,
            contributing_signals=[_signal("manager-1", "analyst-1")],
        ),
        PortfolioDecision(
            asset="BBB",
            as_of_date=day,
            target_weight=0.2,
            contributing_signals=[_signal("manager-1", "analyst-1")],
        ),
    ]

    result = dashboard_module._holdings_by_day({day: decisions}, "analyst-1", "source_agent_id")

    assert result[day] == "AAA long (0.50) · BBB long (0.50)"


def test_holdings_by_day_uses_placeholder_when_nothing_matches() -> None:
    day = date(2024, 1, 2)
    decisions = [
        PortfolioDecision(
            asset="AAA",
            as_of_date=day,
            target_weight=0.1,
            contributing_signals=[_signal("manager-1", "analyst-1")],
        )
    ]

    result = dashboard_module._holdings_by_day({day: decisions}, "never-seen", "source_agent_id")

    assert result[day] == dashboard_module.HOLDINGS_EMPTY_PLACEHOLDER


def test_long_format_chart_frame_reshapes_wide_frame_and_attaches_holdings() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    chart_frame = pd.DataFrame({"analyst-1": [0.0, 1.0], "S&P 500": [0.0, 0.5]}, index=index)
    holdings = {
        "analyst-1": {
            date(2024, 1, 2): "AAA long (0.50)",
            date(2024, 1, 3): "AAA short (0.20)",
        }
    }

    long_frame = dashboard_module._long_format_chart_frame(chart_frame, holdings)

    assert set(long_frame.columns) == {"date", "series", "value", "holdings"}
    analyst_rows = long_frame[long_frame["series"] == "analyst-1"].sort_values("date")
    assert list(analyst_rows["holdings"]) == ["AAA long (0.50)", "AAA short (0.20)"]
    sp500_rows = long_frame[long_frame["series"] == "S&P 500"]
    assert (sp500_rows["holdings"] == dashboard_module.HOLDINGS_EMPTY_PLACEHOLDER).all()


def test_long_format_chart_frame_empty_for_empty_input() -> None:
    result = dashboard_module._long_format_chart_frame(pd.DataFrame())

    assert result.empty
    assert list(result.columns) == ["date", "series", "value", "holdings"]


def test_long_format_chart_frame_drops_rows_with_nan_values() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    chart_frame = pd.DataFrame({"a": [1.0, float("nan")]}, index=index)

    result = dashboard_module._long_format_chart_frame(chart_frame)

    assert len(result) == 1
    assert result.iloc[0]["value"] == pytest.approx(1.0)


@patch("agentic_trading.data.loader.load_price_history_from_lakehouse")
def test_load_basket_data_wraps_single_ticker_into_multiindex(mock_lakehouse: MagicMock) -> None:
    """_load_basket_data always returns MultiIndex columns, even for a
    one-ticker basket, so every downstream consumer only handles one
    shape — lakehouse-only, no live API/synthetic path to exercise."""
    index = pd.bdate_range(date(2024, 1, 2), date(2024, 1, 5))
    mock_lakehouse.return_value = pd.concat(
        {"AAPL": pd.DataFrame({"close": [1.0] * len(index), "log_return": 0.0}, index=index)},
        axis=1,
    )

    frame, _ = dashboard_module._load_basket_data(("AAPL",), date(2024, 1, 2), date(2024, 1, 5))

    assert isinstance(frame.columns, pd.MultiIndex)
    assert "AAPL" in frame.columns.get_level_values(0)


def test_majority_direction_picks_the_most_common_direction() -> None:
    assert dashboard_module._majority_direction(["long", "long", "short"]) == "long"


def test_majority_direction_breaks_ties_deterministically() -> None:
    assert dashboard_module._majority_direction(["short", "long"]) == "long"


def test_majority_direction_empty_input_is_flat() -> None:
    assert dashboard_module._majority_direction([]) == "flat"


def _flow_signal(manager_id: str, source_agent_id: str, direction: str, confidence: float = 0.6):
    return ManagerSignal(
        manager_id=manager_id,
        source_agent_id=source_agent_id,
        asset="AAA",
        as_of_date=date(2024, 1, 2),
        direction=direction,
        confidence=confidence,
        raw_confidence=confidence,
        scale_applied=1.0,
        rationale="test rationale",
    )


def test_build_agent_flow_snapshot_colors_nodes_by_majority_agreement() -> None:
    decisions_by_day = {
        date(2024, 1, 2): [
            PortfolioDecision(
                asset="AAA",
                as_of_date=date(2024, 1, 2),
                target_weight=0.5,
                contributing_signals=[
                    _flow_signal("manager-1", "analyst-1", "long"),
                    _flow_signal("manager-1", "analyst-2", "long"),
                    _flow_signal("manager-2", "analyst-3", "short"),
                ],
            )
        ]
    }

    snapshot = dashboard_module._build_agent_flow_snapshot(
        decisions_by_day, "AAA", date(2024, 1, 2)
    )

    assert snapshot.majority_direction == "long"
    by_name = {node.name: node for node in snapshot.analysts}
    assert by_name["analyst-1"].agrees_with_majority is True
    assert by_name["analyst-3"].agrees_with_majority is False
    assert {node.name for node in snapshot.managers} == {"manager-1", "manager-2"}
    assert snapshot.commander is not None
    assert snapshot.commander.direction == "long"


def test_build_agent_flow_snapshot_empty_for_unmatched_asset() -> None:
    decisions_by_day = {
        date(2024, 1, 2): [
            PortfolioDecision(
                asset="AAA",
                as_of_date=date(2024, 1, 2),
                target_weight=0.5,
                contributing_signals=[_flow_signal("manager-1", "analyst-1", "long")],
            )
        ]
    }

    snapshot = dashboard_module._build_agent_flow_snapshot(
        decisions_by_day, "ZZZ", date(2024, 1, 2)
    )

    assert snapshot.analysts == []
    assert snapshot.commander is None
    assert snapshot.majority_direction == "flat"


# ---------------------------------------------------------------------------
# AppTest integration tests
# ---------------------------------------------------------------------------

_TICKERS = ["AAA", "BBB"]
_DAYS = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]


def _fake_price_history() -> pd.DataFrame:
    index = pd.to_datetime(_DAYS)
    per_ticker = {
        "AAA": pd.DataFrame({"close": [10.0, 10.5, 10.2], "log_return": [None, 0.05, -0.03]}),
        "BBB": pd.DataFrame({"close": [20.0, 19.5, 20.4], "log_return": [None, -0.03, 0.05]}),
    }
    for frame in per_ticker.values():
        frame.index = index
    return pd.concat(per_ticker, axis=1)


def _fake_decisions_by_day() -> dict[date, list[PortfolioDecision]]:
    decisions: dict[date, list[PortfolioDecision]] = {}
    for day in _DAYS:
        decisions[day] = [
            PortfolioDecision(
                asset=ticker,
                as_of_date=day,
                target_weight=0.2,
                contributing_signals=[_signal("manager-1", "analyst-1")],
            )
            for ticker in _TICKERS
        ]
    return decisions


def _fake_track_record() -> TrackRecord:
    track_record = TrackRecord()
    for day, attributable_return in zip(_DAYS, [0.01, -0.005, 0.02], strict=True):
        for source_id, source_type in (("analyst-1", "analyst"), ("manager-1", "manager")):
            track_record.record(
                PerformanceRecord(
                    source_id=source_id,
                    source_type=source_type,
                    asset="AAA",
                    as_of_date=day,
                    attributable_return=attributable_return,
                    raw_attributable_return=attributable_return,
                )
            )
    return track_record


def _fake_desk_run(days: list[date] | None = None) -> dashboard_module.DeskRun:
    days = days if days is not None else _DAYS
    portfolio_value_history = [
        (day, value)
        for day, value in zip(days, [10_000.0, 10_100.0, 9_950.0][: len(days)], strict=True)
    ]
    track_record = _fake_track_record()
    index = pd.to_datetime(days)
    benchmark = pd.Series([0.0, 0.4, -0.2][: len(days)], index=index)
    return dashboard_module.DeskRun(
        tickers=_TICKERS,
        group_name="Test Group",
        manager_mode="Risk Manager",
        source_label="synthetic fallback data",
        manager_ids=["manager-1"],
        analyst_ids=["analyst-1"],
        price_history=_fake_price_history(),
        trade_blotter=pd.DataFrame(
            {
                "date": [days[0], days[0]],
                "asset": _TICKERS,
                "action": ["buy", "buy"],
                "target_weight": [0.2, 0.2],
                "current_weight": [0.0, 0.0],
                "target_shares": [10.0, 5.0],
                "delta_shares": [10.0, 5.0],
                "notional": [100.0, 100.0],
                "close": [10.0, 20.0],
                "manager_ids": ["manager-1", "manager-1"],
                "analyst_ids": ["analyst-1", "analyst-1"],
                "rationale": ["test", "test"],
            }
        ),
        backtest_result=BacktestResult(
            start_date=days[0],
            end_date=days[-1],
            portfolio_value_history=portfolio_value_history,
            final_sharpe_ratio=0.8,
            final_max_drawdown=-0.02,
            final_sortino_ratio=0.7,
            final_calmar_ratio=0.6,
            final_hit_rate=0.5,
            final_profit_factor=1.2,
            decisions_by_day=_fake_decisions_by_day() if len(days) == len(_DAYS) else {},
        ),
        track_record=track_record,
        scoreboard=PerformanceScoreboard(track_record),
        sp500_benchmark=benchmark,
        basket_benchmark=benchmark,
    )


def _seeded_app() -> AppTest:
    at = AppTest.from_file(dashboard_module.__file__)
    desk_run = _fake_desk_run()
    at.session_state["desk_run"] = desk_run
    at.session_state["backtest_result"] = desk_run.backtest_result
    at.session_state["scoreboard"] = desk_run.scoreboard
    return at


def _find_button(at: AppTest, label: str) -> Button:
    for button in at.button:
        if button.label == label:
            return button
    raise AssertionError(f"no button labelled {label!r} was rendered")


def test_app_does_not_crash_with_minimal_synthetic_fixture() -> None:
    """A single-day desk run must not raise inside the script."""
    at = AppTest.from_file(dashboard_module.__file__)
    desk_run = _fake_desk_run(days=[date(2024, 1, 2)])
    at.session_state["desk_run"] = desk_run
    at.session_state["backtest_result"] = desk_run.backtest_result
    at.session_state["scoreboard"] = desk_run.scoreboard

    at.run(timeout=30)

    assert not at.exception


def test_initial_current_day_index_is_zero() -> None:
    """A freshly-run dashboard starts at the first simulated day, not the last."""
    at = _seeded_app().run(timeout=30)

    assert not at.exception
    assert at.session_state["current_day_index"] == 0


def test_step_forward_increments_and_clamps_at_last_available_day() -> None:
    """Clicking past the end of the history must not run current_day_index out of bounds."""
    at = _seeded_app().run(timeout=30)
    last_index = len(_fake_desk_run().backtest_result.portfolio_value_history) - 1

    for _ in range(last_index + 3):
        _find_button(at, "Step forward").click().run(timeout=30)

    assert not at.exception
    assert at.session_state["current_day_index"] == last_index


def test_step_backward_clamps_at_zero() -> None:
    """Stepping backward from day 0 must not go negative."""
    at = _seeded_app().run(timeout=30)

    _find_button(at, "Step backward").click().run(timeout=30)

    assert not at.exception
    assert at.session_state["current_day_index"] == 0


def test_play_button_toggles_is_playing_and_advances_the_day() -> None:
    """Clicking Play flips is_playing and (via the fragment's own autoplay
    tick, resolved within the same rerun) steps the day forward."""
    at = _seeded_app().run(timeout=30)

    _find_button(at, "Play").click().run(timeout=30)

    assert not at.exception
    assert at.session_state["is_playing"] is True
    assert at.session_state["current_day_index"] >= 0


def test_play_button_autostops_and_relabels_at_the_last_day() -> None:
    """Starting playback already on the last day must stop it immediately
    rather than tick forever past the end of the history."""
    at = _seeded_app().run(timeout=30)
    last_index = len(_fake_desk_run().backtest_result.portfolio_value_history) - 1
    for _ in range(last_index):
        _find_button(at, "Step forward").click().run(timeout=30)
    assert at.session_state["current_day_index"] == last_index

    _find_button(at, "Play").click().run(timeout=30)

    assert not at.exception
    assert at.session_state["is_playing"] is False
    assert at.session_state["current_day_index"] == last_index


def test_reset_desk_clears_session_state() -> None:
    at = _seeded_app().run(timeout=30)
    _find_button(at, "Step forward").click().run(timeout=30)

    _find_button(at, "Reset desk").click().run(timeout=30)

    assert not at.exception
    assert at.session_state["desk_run"] is None
    assert at.session_state["backtest_result"] is None
    assert at.session_state["current_day_index"] == 0


def test_desks_and_analysts_tabs_render_without_exception() -> None:
    """Desks/Analysts tab content executes every run regardless of which
    tab is visually active (Streamlit renders every tab's body), so a
    seeded run must not raise anywhere in either tab — including the new
    Altair charts, which replaced the old st.line_chart in these two tabs."""
    at = _seeded_app().run(timeout=30)

    assert not at.exception
    multiselect_labels = {box.label for box in at.multiselect}
    assert "Show desks" in multiselect_labels
    assert "Show analysts" in multiselect_labels
    assert len(at.dataframe) >= 2


def test_agent_flow_view_renders_without_exception() -> None:
    """The Overview tab's agent-flow diagram (the new highest-value panel)
    must render for a seeded run without raising, and must expose the
    per-asset selector it's built around."""
    at = _seeded_app().run(timeout=30)

    assert not at.exception
    flow_selectboxes = [box for box in at.selectbox if box.key == "agent_flow_asset"]
    assert len(flow_selectboxes) == 1
    assert set(flow_selectboxes[0].options) == set(_TICKERS)


def test_trust_panel_renders_without_exception() -> None:
    """The Desks/Analysts trust panels must render for a seeded run
    without raising, surfacing both tabs' scoring-derived weight tables."""
    at = _seeded_app().run(timeout=30)

    assert not at.exception
    markdown_texts = " ".join(md.value for md in at.markdown)
    assert "Desks trust panel" in markdown_texts
    assert "Analysts trust panel" in markdown_texts


# ---------------------------------------------------------------------------
# Saved-run loader (📁 Saved Run tab)
# ---------------------------------------------------------------------------


def _fake_run_summary() -> dict:
    return {
        "experiment": {"assets": ["AAPL"], "excluded_analysts": ["sec_fundamentals_analyst"]},
        "market_data_source": "lakehouse data",
        "calibration": {
            "start_date": "2018-01-02",
            "end_date": "2018-03-30",
            "portfolio_value_history": [["2018-01-02", 10_000.0], ["2018-03-30", 10_500.0]],
            "final_sharpe_ratio": 0.8,
            "final_max_drawdown": -0.1,
            "final_sortino_ratio": 1.0,
            "final_calmar_ratio": 1.5,
            "final_information_ratio": None,
            "final_hit_rate": 0.55,
            "final_profit_factor": 1.1,
            "signal_health": {
                "risk_minimizing_analyst": {"active": 10, "unavailable": 0, "error": 0}
            },
            "source_performance": {
                "risk_minimizing_analyst": {
                    "cumulative_return": 0.02,
                    "sharpe_ratio": 0.5,
                    "max_drawdown": -0.01,
                    "observations": 10.0,
                },
                "risk_manager": {
                    "cumulative_return": 0.03,
                    "sharpe_ratio": 0.6,
                    "max_drawdown": -0.02,
                    "observations": 20.0,
                },
            },
        },
        "test": {
            "start_date": "2018-04-02",
            "end_date": "2018-04-04",
            "portfolio_value_history": [
                ["2018-04-02", 10_000.0],
                ["2018-04-03", 10_100.0],
                ["2018-04-04", 9_950.0],
            ],
            "final_sharpe_ratio": 0.97,
            "final_max_drawdown": -0.31,
            "final_sortino_ratio": 1.36,
            "final_calmar_ratio": 0.51,
            "final_information_ratio": None,
            "final_hit_rate": 0.554,
            "final_profit_factor": 1.22,
            "signal_health": {
                "risk_minimizing_analyst": {"active": 10, "unavailable": 0, "error": 0},
                "cir_rate_regime_analyst": {"active": 7, "unavailable": 0, "error": 3},
            },
            "source_performance": {
                "risk_minimizing_analyst": {
                    "cumulative_return": -0.02,
                    "sharpe_ratio": -0.1,
                    "max_drawdown": -0.03,
                    "observations": 10.0,
                },
                "cvar_manager": {
                    "cumulative_return": -0.9,
                    "sharpe_ratio": -0.14,
                    "max_drawdown": -1.1,
                    "observations": 30.0,
                },
            },
        },
        "baselines": {
            "equal_weight": {
                "start_date": "2018-04-02",
                "end_date": "2018-04-04",
                "portfolio_value_history": [
                    ["2018-04-02", 10_000.0],
                    ["2018-04-03", 10_050.0],
                    ["2018-04-04", 10_020.0],
                ],
                "final_sharpe_ratio": 1.1,
                "final_max_drawdown": -0.01,
                "final_sortino_ratio": 1.3,
                "final_calmar_ratio": 1.0,
                "final_information_ratio": None,
                "final_hit_rate": 0.5,
                "final_profit_factor": 1.05,
            },
        },
    }


def test_saved_run_tab_prompts_for_a_load_when_nothing_is_loaded_yet() -> None:
    at = _seeded_app().run(timeout=30)

    assert not at.exception
    info_texts = " ".join(info.value for info in at.info)
    assert "Load a saved run" in info_texts


def test_saved_run_tab_renders_loaded_summary_without_exception() -> None:
    at = _seeded_app()
    at.session_state["saved_run_summary"] = _fake_run_summary()
    at.session_state["saved_run_source_path"] = "results/fake_run/experiment_result.json"

    at.run(timeout=30)

    assert not at.exception
    markdown_texts = " ".join(md.value for md in at.markdown)
    assert "results/fake_run/experiment_result.json" in markdown_texts


def test_load_saved_run_summary_auto_extracts_a_full_result_file(tmp_path) -> None:
    """A raw experiment_result.json (decisions_by_day/trade_log present)
    must be auto-reduced to the summary shape on load, not handed to the
    rest of the panel unmodified."""
    full_result = _fake_run_summary()
    for period in ("calibration", "test"):
        full_result[period]["decisions_by_day"] = {"2018-04-02": [{"asset": "AAPL"}]}
        full_result[period]["trade_log"] = [{"asset": "AAPL"}]
    path = tmp_path / "experiment_result.json"
    path.write_text(json.dumps(full_result))

    loaded = dashboard_module._load_saved_run_summary(path)

    assert "decisions_by_day" not in loaded["test"]
    assert "trade_log" not in loaded["test"]
    assert loaded["test"]["final_sharpe_ratio"] == 0.97


def test_load_saved_run_summary_accepts_an_already_extracted_summary(tmp_path) -> None:
    summary = _fake_run_summary()
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))

    loaded = dashboard_module._load_saved_run_summary(path)

    assert loaded == summary


def test_load_saved_run_summary_raises_for_a_payload_missing_period_blocks(tmp_path) -> None:
    path = tmp_path / "not_a_run.json"
    path.write_text(json.dumps({"foo": "bar"}))

    with pytest.raises(ValueError, match=r"test.*calibration"):
        dashboard_module._load_saved_run_summary(path)


def test_stepping_forward_advances_current_day_index_that_every_chart_clips_by() -> None:
    at = _seeded_app().run(timeout=30)
    desk_run = _fake_desk_run()
    initial_index = at.session_state["current_day_index"]
    initial_date = desk_run.backtest_result.portfolio_value_history[initial_index][0]

    _find_button(at, "Step forward").click().run(timeout=30)

    assert not at.exception
    new_index = at.session_state["current_day_index"]
    assert new_index == initial_index + 1
    new_date = desk_run.backtest_result.portfolio_value_history[new_index][0]
    assert new_date > initial_date

    clipped_before = dashboard_module._clip_series_to_date(desk_run.sp500_benchmark, initial_date)
    clipped_after = dashboard_module._clip_series_to_date(desk_run.sp500_benchmark, new_date)
    assert len(clipped_after) > len(clipped_before)


def test_performance_tabs_are_called_only_from_inside_the_fragment_body() -> None:
    import inspect

    render_dashboard_source = inspect.getsource(dashboard_module.render_dashboard)
    tabbed_content_source = inspect.getsource(dashboard_module._render_tabbed_content)

    assert "_render_overview_tab(" not in render_dashboard_source
    assert "_render_performance_tab(" not in render_dashboard_source
    assert "_render_tabbed_content" in render_dashboard_source
    assert "_render_overview_tab(" in tabbed_content_source
    assert tabbed_content_source.count("_render_performance_tab(") == 2


def _find_multiselect(at: AppTest, key: str) -> Multiselect:
    for element in at.multiselect:
        if element.key == key:
            return element
    raise AssertionError(f"no multiselect with key {key!r} was rendered")


def test_ticker_multiselect_changes_the_basket_that_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Editing the ticker multiselect beyond the preset group's symbols must
    change the resolved ticker list the desk actually runs on"""
    monkeypatch.setattr("agentic_trading.agents.roster.build_default_analysts", lambda: [])

    def _explicit_test_data(
        tickers: tuple[str, ...], start: date, end: date, **_kwargs: object
    ) -> tuple[pd.DataFrame, str]:
        from agentic_trading.data.loader import _synthetic_price_history

        return (
            pd.concat(
                {ticker: _synthetic_price_history(ticker, start, end) for ticker in tickers},
                axis=1,
            ),
            "explicit test fixture",
        )

    monkeypatch.setattr(dashboard_module, "_load_basket_data", _explicit_test_data)

    at = AppTest.from_file(dashboard_module.__file__)
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["desk_run"] is None

    default_group = dashboard_module.TICKER_GROUPS[0]
    _find_button(at, "Run simulation").click().run(timeout=30)
    assert not at.exception
    assert at.session_state["desk_run"].tickers == list(default_group.symbols)

    multiselect_key = f"ticker_multiselect__{default_group.name}"
    extended_tickers = [*default_group.symbols, "NVDA"]
    _find_multiselect(at, multiselect_key).set_value(extended_tickers).run(timeout=30)
    _find_button(at, "Run simulation").click().run(timeout=30)

    assert not at.exception
    assert at.session_state["desk_run"].tickers == extended_tickers


def test_empty_ticker_selection_warns_and_does_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clearing the ticker multiselect entirely must warn, not attempt a
    zero-ticker simulation."""
    monkeypatch.setattr("agentic_trading.agents.roster.build_default_analysts", lambda: [])

    at = AppTest.from_file(dashboard_module.__file__)
    at.run(timeout=30)
    assert not at.exception

    default_group = dashboard_module.TICKER_GROUPS[0]
    multiselect_key = f"ticker_multiselect__{default_group.name}"
    _find_multiselect(at, multiselect_key).set_value([]).run(timeout=30)

    assert not at.exception
    warning_texts = [warning.value for warning in at.warning]
    assert any("at least one ticker" in text for text in warning_texts)


def _fake_desk_run_with_days(days: list[date]) -> dashboard_module.DeskRun:
    """A minimal DeskRun spanning an arbitrary number of days — _fake_desk_run
    is hardcoded to _DAYS' three entries (its portfolio-value/benchmark lists
    are literal 3-value lists sliced by len(days)), so it can't represent a
    longer run. Used for the autoplay-vs-manual-nav regression test below,
    which needs enough days of runway to step forward and then back again
    without hitting either boundary."""
    portfolio_value_history = [(day, 10_000.0 + 10.0 * i) for i, day in enumerate(days)]
    index = pd.to_datetime(days)
    benchmark = pd.Series([0.0] * len(days), index=index)
    return dashboard_module.DeskRun(
        tickers=_TICKERS,
        group_name="Test Group",
        manager_mode="Risk Manager",
        source_label="synthetic fallback data",
        manager_ids=["manager-1"],
        analyst_ids=["analyst-1"],
        price_history=_fake_price_history(),
        trade_blotter=pd.DataFrame(),
        backtest_result=BacktestResult(
            start_date=days[0],
            end_date=days[-1],
            portfolio_value_history=portfolio_value_history,
            final_sharpe_ratio=0.8,
            final_max_drawdown=-0.02,
            final_sortino_ratio=0.7,
            final_calmar_ratio=0.6,
            final_hit_rate=0.5,
            final_profit_factor=1.2,
            decisions_by_day={day: [] for day in days},
        ),
        track_record=TrackRecord(),
        scoreboard=PerformanceScoreboard(TrackRecord()),
        sp500_benchmark=benchmark,
        basket_benchmark=benchmark,
    )


def test_manual_step_while_playing_advances_by_exactly_one_not_two() -> None:
    """Regression test: clicking Step forward while autoplay is active used"""
    days = [date(2024, 1, day) for day in range(2, 9)]  # 7 days
    desk_run = _fake_desk_run_with_days(days)
    at = AppTest.from_file(dashboard_module.__file__)
    at.session_state["desk_run"] = desk_run
    at.session_state["backtest_result"] = desk_run.backtest_result
    at.session_state["scoreboard"] = desk_run.scoreboard
    at.run(timeout=30)

    _find_button(at, "Play").click().run(timeout=30)
    assert at.session_state["is_playing"] is True
    index_after_play = at.session_state["current_day_index"]

    _find_button(at, "Step forward").click().run(timeout=30)
    assert not at.exception
    assert at.session_state["current_day_index"] == index_after_play + 1

    index_before_backward = at.session_state["current_day_index"]
    _find_button(at, "Step backward").click().run(timeout=30)
    assert not at.exception
    assert at.session_state["current_day_index"] == index_before_backward - 1
