# test_date_aligned_regressions.py

import re

import numpy as np
import pandas as pd
import pytest

from agentic_trading.data.loaders.lakehouse_loader import IndexMember

FACTORS = ["Mkt-RF", "SMB", "HML"]


def _ols(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Intercept-first least-squares coefficients."""
    design = np.column_stack([np.ones(len(y)), x])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def _factor_table(dates: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """Trending Fama-French style factors so that shifting them by days changes any regression."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame(rng.normal(0.0, 0.01, (dates.size, 3)), index=dates, columns=FACTORS)
    table["Mkt-RF"] += np.linspace(-0.004, 0.004, dates.size)
    table["SMB"] += np.linspace(0.003, -0.003, dates.size)
    table["RF"] = 0.0001
    return table


def _recorder(monkeypatch, mod, name: str) -> list[tuple[tuple, object]]:
    """Wraps `mod.<name>` so each call's inputs are stored while the real result is returned."""
    real = getattr(mod, name)
    calls: list[tuple[tuple, object]] = []

    def _wrapped(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append((tuple(np.asarray(a, dtype=float) for a in args), result))
        return result

    monkeypatch.setattr(mod, name, _wrapped)
    return calls


def test_fama_french_regression_joins_returns_and_stale_factors_by_date(monkeypatch) -> None:
    """A factor table ending 15 days early and a missing return day give the date-joined
    OLS beta."""
    from agentic_trading.agents.analysts.factor_pod import fama_french_factor_analyst as mod

    dates = pd.bdate_range(end="2021-06-30", periods=320)
    full = _factor_table(dates, seed=0)
    noise = np.random.default_rng(1).normal(0.0, 0.002, dates.size)
    returns = pd.Series(
        1.2 * full["Mkt-RF"] + 0.4 * full["SMB"] - 0.3 * full["HML"] + full["RF"] + noise,
        index=dates,
    )
    returns.iloc[-100] = np.nan
    market_data = pd.concat({"AAPL": pd.DataFrame({"log_return": returns})}, axis=1)

    window = returns.dropna().iloc[-250:]
    stale = full.loc[window.index[0] : dates[-16]]
    monkeypatch.setattr(mod, "load_fama_french_factors_from_lakehouse", lambda *_a, **_k: stale)
    calls = _recorder(monkeypatch, mod, "factor_loadings")

    signal = mod.FamaFrenchFactorAnalyst().generate_signal("AAPL", market_data, dates[-1])

    joined = pd.concat([window.rename("ret"), stale], axis=1, join="inner")
    expected_y = (joined["ret"] - joined["RF"]).to_numpy()
    expected_x = joined[FACTORS].to_numpy()
    expected = _ols(expected_y, expected_x)
    assert len(joined) < len(window)

    (y, x), result = calls[0]
    np.testing.assert_allclose(y, expected_y, atol=1e-12)
    np.testing.assert_allclose(x, expected_x, atol=1e-12)
    np.testing.assert_allclose(result.loadings, expected, atol=1e-9)
    assert signal.sample_count == len(joined)

    length = len(stale)
    positional_y = window.to_numpy()[-length:] - stale["RF"].to_numpy()
    positional = _ols(positional_y, stale[FACTORS].to_numpy())
    assert abs(positional[1] - expected[1]) > 0.1
    assert abs(result.loadings[1] - positional[1]) > 0.1


def _patch_member_returns(monkeypatch, mod, tickers: list[str], frame: pd.DataFrame) -> None:
    """Serves `tickers` as S&P members (security ids by position) whose
    returns are `frame`'s `log_return` columns, gaps dropped as the
    security-keyed loader returns them."""
    members = [IndexMember(security_id=i, ticker=t) for i, t in enumerate(tickers)]
    monkeypatch.setattr(mod, "load_sp500_member_securities_asof", lambda *_a: members)
    monkeypatch.setattr(
        mod,
        "load_security_log_returns",
        lambda ids, *_a, **_k: {
            m.security_id: frame[(m.ticker, "log_return")].dropna()
            for m in members
            if m.security_id in ids and m.ticker in frame.columns.get_level_values(0)
        },
    )


def _universe_frame(
    n_tickers: int, n_days: int, seed: int, hole: tuple[str, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide (ticker, log_return) frame and factor table on one business-day index, with one
    NaN return. RF varies by day so that subtracting it by position instead of by date
    changes excess returns."""
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    factors = _factor_table(dates, seed)
    factors["RF"] = np.random.default_rng(seed + 2).uniform(0.00005, 0.00015, n_days)
    rng = np.random.default_rng(seed + 1)
    columns = {}
    for i in range(n_tickers):
        betas = rng.normal(1.0, 0.5, 3)
        columns[f"T{i:03d}"] = factors[FACTORS].to_numpy() @ betas + rng.normal(0, 0.004, n_days)
    returns = pd.DataFrame(columns, index=dates)
    returns.iloc[hole[1], returns.columns.get_loc(hole[0])] = np.nan
    frame = pd.concat({t: pd.DataFrame({"log_return": returns[t]}) for t in returns}, axis=1)
    return frame, factors


def test_fama_macbeth_stages_use_date_joined_panel(monkeypatch) -> None:
    """A one-day hole in the asset's returns removes that date for every ticker and the factors."""
    from agentic_trading.agents.analysts.factor_pod import fama_macbeth_analyst as mod

    frame, factors = _universe_frame(35, 100, seed=10, hole=("T000", 60))
    tickers = list(frame.columns.get_level_values(0).unique())
    _patch_member_returns(monkeypatch, mod, tickers, frame)
    monkeypatch.setattr(mod, "load_price_history_from_lakehouse", lambda *_a, **_k: frame)
    monkeypatch.setattr(mod, "load_fama_french_factors_from_lakehouse", lambda *_a, **_k: factors)
    loadings_calls = _recorder(monkeypatch, mod, "factor_loadings")
    premia_calls = _recorder(monkeypatch, mod, "fama_macbeth_regression")

    signal = mod.FamaMacbethAnalyst().generate_signal("T000", pd.DataFrame(), factors.index[-1])

    wide = pd.DataFrame({t: frame[(t, "log_return")] for t in tickers})
    joined = pd.concat([wide, factors], axis=1, join="inner").dropna()
    assert len(joined) == len(factors) - 1
    expected_x = joined[FACTORS].to_numpy()
    expected_panel = joined[tickers].sub(joined["RF"], axis=0).to_numpy()
    expected_betas = np.vstack(
        [_ols(expected_panel[:, i], expected_x)[1:] for i in range(len(tickers))]
    )

    assert len(loadings_calls) == len(tickers)
    for i, ((y, x), _result) in enumerate(loadings_calls):
        np.testing.assert_allclose(y, expected_panel[:, i], atol=1e-12)
        np.testing.assert_allclose(x, expected_x, atol=1e-12)
    (panel, betas), (premia, _t_stats) = premia_calls[0]
    np.testing.assert_allclose(panel, expected_panel, atol=1e-12)
    np.testing.assert_allclose(betas, expected_betas, atol=1e-9)

    positional_y = wide["T000"].dropna().to_numpy() - factors["RF"].to_numpy()[-99:]
    assert not np.allclose(positional_y, expected_panel[:, 0], atol=1e-6)

    predicted = float(expected_betas[0] @ np.asarray(premia))
    reported = float(re.search(r"predicted return=(-?\d+\.\d+)", signal.rationale).group(1))
    assert reported == pytest.approx(predicted, abs=1e-5)
    assert signal.direction == ("long" if predicted >= 0.0 else "short")


def test_network_centrality_correlations_use_date_joined_returns(monkeypatch) -> None:
    """A one-day hole in the asset's returns shifts the lead-lag window to the date-joined rows."""
    from agentic_trading.agents.analysts.alt_data_pod import network_centrality_analyst as mod

    n_days, window, lag = 150, mod.WINDOW, mod.LAG
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    rng = np.random.default_rng(5)
    wide = pd.DataFrame(
        rng.normal(0.0, 0.005, (n_days, 35)), index=dates, columns=[f"T{i:03d}" for i in range(35)]
    )
    wide["T000"] = 0.5 * wide["T001"].shift(1).fillna(0.0) + 0.5 * wide["T002"].shift(2).fillna(0.0)
    wide["T000"] += rng.normal(0.0, 0.003, n_days)
    wide.iloc[n_days - 20, 0] = np.nan
    tickers = list(wide.columns)
    frame = pd.concat({t: pd.DataFrame({"log_return": wide[t]}) for t in tickers}, axis=1)
    _patch_member_returns(monkeypatch, mod, tickers, frame)

    analyst = mod.LeadLagNetworkCentralityAnalyst()
    signal = analyst.generate_signal("T000", pd.DataFrame(), dates[-1])

    def _confidence(columns: dict[str, np.ndarray]) -> tuple[float, float]:
        current = columns["T000"][lag:]
        corrs = {
            t: float(np.corrcoef(v[:-lag], current)[0, 1])
            for t, v in columns.items()
            if t != "T000"
        }
        predicted = sum(c * columns[t][-1] for t, c in corrs.items())
        centrality = float(np.mean(np.abs(list(corrs.values()))))
        return min(centrality, 1.0) * min(abs(predicted) / mod.RETURN_CAP, 1.0), predicted

    joined = wide.dropna().iloc[-(window + lag) :]
    expected, predicted = _confidence({t: joined[t].to_numpy() for t in tickers})
    tails = {t: wide[t].dropna().to_numpy()[-(window + lag) :] for t in tickers}
    positional, _ = _confidence(tails)

    assert 0.0 < expected < 1.0 and abs(expected - positional) > 1e-6
    assert signal.confidence == pytest.approx(expected, abs=1e-9)
    assert signal.direction == ("long" if predicted > 0.0 else "short")


def test_fama_macbeth_drops_a_gappy_universe_ticker_not_its_date(monkeypatch) -> None:
    """A one-day hole in a non-asset ticker drops that ticker; every other name keeps all dates."""
    from agentic_trading.agents.analysts.factor_pod import fama_macbeth_analyst as mod

    frame, factors = _universe_frame(35, 100, seed=20, hole=("T005", 60))
    tickers = list(frame.columns.get_level_values(0).unique())
    _patch_member_returns(monkeypatch, mod, tickers, frame)
    monkeypatch.setattr(mod, "load_price_history_from_lakehouse", lambda *_a, **_k: frame)
    monkeypatch.setattr(mod, "load_fama_french_factors_from_lakehouse", lambda *_a, **_k: factors)
    loadings_calls = _recorder(monkeypatch, mod, "factor_loadings")
    premia_calls = _recorder(monkeypatch, mod, "fama_macbeth_regression")

    signal = mod.FamaMacbethAnalyst().generate_signal("T000", pd.DataFrame(), factors.index[-1])

    kept = [t for t in tickers if t != "T005"]
    wide = pd.DataFrame({t: frame[(t, "log_return")] for t in kept})
    expected_x = factors[FACTORS].to_numpy()
    expected_panel = wide.sub(factors["RF"], axis=0).to_numpy()
    assert expected_panel.shape == (len(factors), len(tickers) - 1)

    assert len(loadings_calls) == len(kept)
    for i, ((y, x), _result) in enumerate(loadings_calls):
        np.testing.assert_allclose(y, expected_panel[:, i], atol=1e-12)
        np.testing.assert_allclose(x, expected_x, atol=1e-12)
    (panel, _betas), _premia = premia_calls[0]
    np.testing.assert_allclose(panel, expected_panel, atol=1e-12)
    assert signal.status == "active"
    assert signal.sample_count == len(factors)
    assert f"{len(kept)}-name cross-section" in signal.rationale


def test_network_centrality_drops_a_gappy_universe_ticker_not_its_date(monkeypatch) -> None:
    """A one-day hole in a non-asset ticker drops that ticker from the lead-lag panel; the asset
    and every other name keep the asset's last WINDOW + LAG dates."""
    from agentic_trading.agents.analysts.alt_data_pod import network_centrality_analyst as mod

    n_days, window, lag = 150, mod.WINDOW, mod.LAG
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    rng = np.random.default_rng(7)
    wide = pd.DataFrame(
        rng.normal(0.0, 0.005, (n_days, 35)), index=dates, columns=[f"T{i:03d}" for i in range(35)]
    )
    wide["T000"] = 0.5 * wide["T001"].shift(1).fillna(0.0) + 0.5 * wide["T002"].shift(2).fillna(0.0)
    wide["T000"] += rng.normal(0.0, 0.003, n_days)
    wide.iloc[n_days - 20, wide.columns.get_loc("T005")] = np.nan
    tickers = list(wide.columns)
    frame = pd.concat({t: pd.DataFrame({"log_return": wide[t]}) for t in tickers}, axis=1)
    _patch_member_returns(monkeypatch, mod, tickers, frame)

    signal = mod.LeadLagNetworkCentralityAnalyst().generate_signal(
        "T000", pd.DataFrame(), dates[-1]
    )

    def _confidence(panel: pd.DataFrame) -> float:
        current = panel["T000"].to_numpy()[lag:]
        corrs = {
            t: float(np.corrcoef(panel[t].to_numpy()[:-lag], current)[0, 1])
            for t in panel.columns
            if t != "T000"
        }
        predicted = sum(c * float(panel[t].iloc[-1]) for t, c in corrs.items())
        centrality = float(np.mean(np.abs(list(corrs.values()))))
        return min(centrality, 1.0) * min(abs(predicted) / mod.RETURN_CAP, 1.0)

    kept = [t for t in tickers if t != "T005"]
    expected = _confidence(wide[kept].iloc[-(window + lag) :])
    date_dropping = _confidence(wide.dropna().iloc[-(window + lag) :])

    assert abs(expected - date_dropping) > 1e-9
    assert signal.confidence == pytest.approx(expected, abs=1e-9)
    assert signal.sample_count == window + lag
    assert "'T005'" not in signal.rationale
    assert f"{len(kept)} of {len(tickers)} universe names" in signal.rationale
