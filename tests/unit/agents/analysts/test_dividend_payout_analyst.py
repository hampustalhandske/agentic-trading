# test_dividend_payout_analyst.py

"""Tests for `DividendPayoutAnalyst`, with the lake loader monkeypatched."""

import pandas as pd

from agentic_trading.agents.analysts.risk_pod import dividend_payout_analyst as mod


def test_two_equal_payouts_are_flat_with_zero_confidence(monkeypatch) -> None:
    """Equal consecutive split-adjusted payouts give a flat, zero-confidence signal."""
    payouts = pd.Series([0.205, 0.205], index=pd.to_datetime(["2020-08-07", "2020-11-06"]))
    monkeypatch.setattr(mod, "load_dividends_from_lakehouse", lambda *_a: payouts)

    signal = mod.DividendPayoutAnalyst().generate_signal(
        "AAPL", pd.DataFrame(), pd.Timestamp("2020-11-06")
    )

    assert signal.direction == "flat"
    assert signal.confidence == 0.0
    assert signal.status == "active"
