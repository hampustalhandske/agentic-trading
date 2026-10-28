# test_option_spot_basis.py

import importlib

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.analysts.cvar_pod import _options_data

ASSET = "XOM"
AS_OF = pd.Timestamp("2020-01-02")
ADJ_CLOSE = 52.61
RAW_CLOSE = 70.90
STRIKE_STEP = 2.5
MODULES = (
    "black_scholes_valuation_analyst",
    "greeks_delta_hedge_analyst",
    "heston_vol_surface_analyst",
)


def _market_data() -> pd.DataFrame:
    """Price frame in the loader's (ticker, field) shape with a dividend-adjusted close
    below raw_close."""
    dates = pd.bdate_range(end=AS_OF, periods=300)
    returns = np.random.default_rng(1).normal(0.0, 0.01, dates.size)
    frame = pd.DataFrame(
        {"close": ADJ_CLOSE, "raw_close": RAW_CLOSE, "log_return": returns}, index=dates
    )
    return pd.concat({ASSET: frame}, axis=1)


def _snapshot(asset: str, as_of) -> pd.DataFrame:
    """Call and put quotes for strikes 50..75 in steps of 2.5 across three expiries."""
    rows = []
    for expiration in pd.to_datetime(["2020-01-24", "2020-02-21", "2020-03-20"]):
        for strike in np.arange(50.0, 75.01, STRIKE_STEP):
            for option_type in ("call", "put"):
                rows.append(
                    {
                        "date": AS_OF,
                        "expiration": expiration,
                        "strike": float(strike),
                        "option_type": option_type,
                        "bid": 1.0,
                        "ask": 1.1,
                        "implied_volatility": 0.22 + 0.001 * strike / 10,
                        "delta": 0.5 if option_type == "call" else -0.5,
                        "gamma": 0.04,
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("module_name", MODULES)
def test_options_analysts_pick_atm_strike_on_raw_close(monkeypatch, module_name: str) -> None:
    """Each options analyst asks for the ATM row with the raw close and gets a strike next to it."""
    mod = importlib.import_module(f"agentic_trading.agents.analysts.cvar_pod.{module_name}")
    monkeypatch.setattr(_options_data, "load_options_snapshot_from_lakehouse", _snapshot)
    monkeypatch.setattr(mod, "annualized_garch_volatility", lambda *_a, **_k: 0.25, raising=False)
    monkeypatch.setattr(
        mod, "conditional_daily_volatility", lambda r, *_a, **_k: float(np.std(r)), raising=False
    )
    calls: list[tuple[float, pd.DataFrame]] = []

    def _recording(asset, as_of_date, spot):
        result = _options_data.real_atm_options_by_expiration(asset, as_of_date, spot)
        calls.append((spot, result))
        return result

    monkeypatch.setattr(mod, "real_atm_options_by_expiration", _recording)
    analyst = getattr(mod, _class_name(module_name))()
    analyst.generate_signal(ASSET, _market_data(), AS_OF)

    assert calls, "analyst never requested the ATM option rows"
    for spot, atm in calls:
        assert spot == pytest.approx(RAW_CLOSE)
        assert spot != pytest.approx(ADJ_CLOSE)
        assert atm is not None and not atm.empty
        assert set(atm["strike"]) <= {70.0, 72.5}
        assert (atm["strike"] - RAW_CLOSE).abs().max() <= STRIKE_STEP


def test_black_scholes_prices_off_raw_close(monkeypatch) -> None:
    """The Black-Scholes pricer receives the raw close as spot and a strike near the raw close."""
    from agentic_trading.agents.analysts.cvar_pod import black_scholes_valuation_analyst as mod

    monkeypatch.setattr(_options_data, "load_options_snapshot_from_lakehouse", _snapshot)
    monkeypatch.setattr(mod, "conditional_daily_volatility", lambda r, *_a, **_k: 0.01)
    priced: list[tuple[float, float]] = []

    def _record(spot, strike, *_args, **_kwargs):
        priced.append((float(spot), float(strike)))
        return 1.0

    monkeypatch.setattr(mod, "black_scholes_call", _record)
    monkeypatch.setattr(mod, "black_scholes_put", _record)
    mod.BlackScholesValuationAnalyst().generate_signal(ASSET, _market_data(), AS_OF)

    assert priced
    for spot, strike in priced:
        assert spot == pytest.approx(RAW_CLOSE)
        assert abs(strike - RAW_CLOSE) <= STRIKE_STEP
        assert strike in (70.0, 72.5) or strike == pytest.approx(RAW_CLOSE)


def _class_name(module_name: str) -> str:
    """CamelCase analyst class name for a snake_case module name."""
    return "".join(part.capitalize() for part in module_name.split("_"))
