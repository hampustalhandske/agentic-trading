from unittest.mock import patch

import numpy as np
import pandas as pd

from agentic_trading.agents.analysts.stat_arb_pod.ou_mean_reversion_analyst import (
    OUMeanReversionAnalyst,
)


def _market_data(asset: str, n: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(100.0 + np.arange(n) * 0.5, index=dates)
    return pd.concat({asset: pd.DataFrame({"close": close})}, axis=1)


@patch("agentic_trading.agents.analysts.stat_arb_pod.ou_mean_reversion_analyst.ou_half_life")
def test_non_mean_reverting_spread_holds_gracefully_instead_of_raising(mock_half_life) -> None:
    """quantcore's `ou_half_life` raises `ValueError` for a spread that
    doesn't mean-revert (phi >= 0, diverging) — a real, non-hypothetical
    case that surfaced in an actual backtest run. This must produce a
    clean flat/0-confidence hold like every other analyst's own
    try/except handles its failure modes, not an unhandled exception that
    only the generic base.py safety net catches."""
    mock_half_life.side_effect = ValueError(
        "estimated series does not exhibit mean reversion (phi >= 0, diverging)"
    )
    market_data = _market_data("AAPL")
    analyst = OUMeanReversionAnalyst()

    signal = analyst.generate_signal("AAPL", market_data, market_data.index[-1])

    assert signal.direction == "flat"
    assert signal.confidence == 0.0
    assert "not mean-reverting" in signal.rationale
