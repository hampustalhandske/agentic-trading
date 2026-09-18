"""Unit tests for StatArbManager's quarterly walk-forward pair
re-screening. `_pairs_screening.rescreen_pairs`'s own Engle-Granger/ADF
correctness is a statistics question, not re-tested here — these tests
pin down the orchestration: does the manager gate on
`pairs_refit_interval_days`, does it apply the admit/keep/drop
hysteresis to the verdicts, and does it push the resulting wiring and
hedge ratios into the two consuming analysts only when it re-screens."""

from datetime import timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from agentic_trading.agents.analysts.stat_arb_pod._pairs_screening import PairVerdict
from agentic_trading.agents.analysts.stat_arb_pod.kalman_hedge_ratio_analyst import (
    KalmanHedgeRatioAnalyst,
)
from agentic_trading.agents.analysts.stat_arb_pod.pairs_cointegration_analyst import (
    PairsCointegrationAnalyst,
)
from agentic_trading.agents.managers.stat_arb_manager import (
    StatArbManager,
)
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.config import settings
from agentic_trading.scoring.weighting import Hedge

AS_OF = pd.Timestamp("2024-06-20")
ASSETS = ["AAPL", "AMZN", "MSFT", "ADBE"]
REFIT = timedelta(days=settings.managers.pairs_refit_interval_days)


def _manager() -> tuple[StatArbManager, PairsCointegrationAnalyst, KalmanHedgeRatioAnalyst]:
    pairs_analyst = PairsCointegrationAnalyst()
    kalman_analyst = KalmanHedgeRatioAnalyst()
    manager = StatArbManager(
        managed_analysts=[pairs_analyst, kalman_analyst],
        weighting_strategy=Hedge(),
        portfolio_constructor=PortfolioConstructor(),
    )
    return manager, pairs_analyst, kalman_analyst


def _verdict(
    p_value: float = 0.01, half_life: float = 20.0, beta: float = 1.5, dependent: str = "AAPL"
) -> PairVerdict:
    return PairVerdict(p_value=p_value, beta=beta, dependent=dependent, half_life=half_life)


@patch("agentic_trading.agents.managers.stat_arb_manager.resolve_candidate_pairs")
@patch("agentic_trading.agents.managers.stat_arb_manager.rescreen_pairs")
def test_first_call_admits_passing_pairs_and_wires_hedge_ratios_symmetrically(
    mock_rescreen, mock_resolve
) -> None:
    mock_resolve.return_value = (("AAPL", "AMZN"), ("MSFT", "ADBE"))
    mock_rescreen.return_value = {
        ("AAPL", "AMZN"): _verdict(beta=2.0, dependent="AAPL"),
        ("MSFT", "ADBE"): _verdict(p_value=0.08),  # below keep, not below admit
    }
    manager, pairs_analyst, kalman_analyst = _manager()

    manager._maybe_rescreen_pairs(ASSETS, pd.DataFrame(), AS_OF)

    assert mock_rescreen.call_count == 1
    assert pairs_analyst.partner_tickers_by_asset == {"AAPL": ("AMZN",), "AMZN": ("AAPL",)}
    assert kalman_analyst.partner_ticker_by_asset == {"AAPL": "AMZN", "AMZN": "AAPL"}
    assert pairs_analyst.hedge_ratio_by_pair == {("AAPL", "AMZN"): 2.0, ("AMZN", "AAPL"): 0.5}


@patch("agentic_trading.agents.managers.stat_arb_manager.resolve_candidate_pairs")
@patch("agentic_trading.agents.managers.stat_arb_manager.rescreen_pairs")
@pytest.mark.parametrize("half_life", [2.0, 80.0])
def test_admission_requires_half_life_within_tradeable_band(
    mock_rescreen, mock_resolve, half_life: float
) -> None:
    mock_resolve.return_value = (("AAPL", "AMZN"),)
    mock_rescreen.return_value = {("AAPL", "AMZN"): _verdict(half_life=half_life)}
    manager, pairs_analyst, _ = _manager()

    manager._maybe_rescreen_pairs(ASSETS, pd.DataFrame(), AS_OF)

    assert pairs_analyst.partner_tickers_by_asset == {}


@patch("agentic_trading.agents.managers.stat_arb_manager.resolve_candidate_pairs")
@patch("agentic_trading.agents.managers.stat_arb_manager.rescreen_pairs")
def test_second_call_within_refit_interval_does_not_rescreen_again(
    mock_rescreen, mock_resolve
) -> None:
    mock_resolve.return_value = (("AAPL", "AMZN"),)
    mock_rescreen.return_value = {("AAPL", "AMZN"): _verdict()}
    manager, pairs_analyst, _ = _manager()

    manager._maybe_rescreen_pairs(ASSETS, pd.DataFrame(), AS_OF)
    mock_rescreen.return_value = {}  # would change the wiring if re-run twice
    manager._maybe_rescreen_pairs(ASSETS, pd.DataFrame(), AS_OF + timedelta(days=10))

    assert mock_rescreen.call_count == 1
    assert pairs_analyst.partner_tickers_by_asset == {"AAPL": ("AMZN",), "AMZN": ("AAPL",)}


@patch("agentic_trading.agents.managers.stat_arb_manager.resolve_candidate_pairs")
@patch("agentic_trading.agents.managers.stat_arb_manager.rescreen_pairs")
def test_held_pair_survives_one_failed_rescreen_and_drops_after_two(
    mock_rescreen, mock_resolve
) -> None:
    mock_resolve.return_value = (("AAPL", "AMZN"),)
    mock_rescreen.return_value = {("AAPL", "AMZN"): _verdict()}
    manager, pairs_analyst, kalman_analyst = _manager()

    manager._maybe_rescreen_pairs(ASSETS, pd.DataFrame(), AS_OF)
    mock_rescreen.return_value = {("AAPL", "AMZN"): _verdict(p_value=0.20)}
    manager._maybe_rescreen_pairs(ASSETS, pd.DataFrame(), AS_OF + REFIT)

    assert mock_rescreen.call_count == 2
    assert pairs_analyst.partner_tickers_by_asset == {"AAPL": ("AMZN",), "AMZN": ("AAPL",)}

    mock_rescreen.return_value = {}
    manager._maybe_rescreen_pairs(ASSETS, pd.DataFrame(), AS_OF + 2 * REFIT)

    assert mock_rescreen.call_count == 3
    assert pairs_analyst.partner_tickers_by_asset == {}
    assert pairs_analyst.hedge_ratio_by_pair == {}
    assert kalman_analyst.partner_ticker_by_asset == {}


@patch("agentic_trading.agents.managers.stat_arb_manager.resolve_candidate_pairs")
@patch("agentic_trading.agents.managers.stat_arb_manager.rescreen_pairs")
def test_held_pair_is_kept_between_admit_and_keep_pvalues(mock_rescreen, mock_resolve) -> None:
    mock_resolve.return_value = (("AAPL", "AMZN"),)
    mock_rescreen.return_value = {("AAPL", "AMZN"): _verdict()}
    manager, pairs_analyst, _ = _manager()

    manager._maybe_rescreen_pairs(ASSETS, pd.DataFrame(), AS_OF)
    mock_rescreen.return_value = {("AAPL", "AMZN"): _verdict(p_value=0.08, half_life=75.0)}
    manager._maybe_rescreen_pairs(ASSETS, pd.DataFrame(), AS_OF + REFIT)
    mock_rescreen.return_value = {("AAPL", "AMZN"): _verdict(p_value=0.08, half_life=75.0)}
    manager._maybe_rescreen_pairs(ASSETS, pd.DataFrame(), AS_OF + 2 * REFIT)

    assert pairs_analyst.partner_tickers_by_asset == {"AAPL": ("AMZN",), "AMZN": ("AAPL",)}


@patch("agentic_trading.agents.managers.stat_arb_manager.resolve_candidate_pairs")
@patch("agentic_trading.agents.managers.stat_arb_manager.rescreen_pairs")
def test_held_pair_drops_immediately_above_drop_half_life(mock_rescreen, mock_resolve) -> None:
    mock_resolve.return_value = (("AAPL", "AMZN"),)
    mock_rescreen.return_value = {("AAPL", "AMZN"): _verdict()}
    manager, pairs_analyst, _ = _manager()

    manager._maybe_rescreen_pairs(ASSETS, pd.DataFrame(), AS_OF)
    mock_rescreen.return_value = {
        ("AAPL", "AMZN"): _verdict(
            p_value=0.01, half_life=settings.analysts.pair_drop_half_life_days + 1.0
        )
    }
    manager._maybe_rescreen_pairs(ASSETS, pd.DataFrame(), AS_OF + REFIT)

    assert pairs_analyst.partner_tickers_by_asset == {}


@patch("agentic_trading.agents.managers.stat_arb_manager.resolve_candidate_pairs")
@patch("agentic_trading.agents.managers.stat_arb_manager.rescreen_pairs")
def test_held_pairs_are_ranked_by_half_life_and_capped_at_max_pairs(
    mock_rescreen, mock_resolve, monkeypatch
) -> None:
    monkeypatch.setattr(settings.analysts, "max_pairs", 1)
    mock_resolve.return_value = (("AAPL", "AMZN"), ("MSFT", "ADBE"))
    mock_rescreen.return_value = {
        ("AAPL", "AMZN"): _verdict(half_life=40.0, dependent="AAPL"),
        ("MSFT", "ADBE"): _verdict(half_life=10.0, dependent="MSFT"),
    }
    manager, pairs_analyst, _ = _manager()

    manager._maybe_rescreen_pairs(ASSETS, pd.DataFrame(), AS_OF)

    assert pairs_analyst.partner_tickers_by_asset == {"MSFT": ("ADBE",), "ADBE": ("MSFT",)}
