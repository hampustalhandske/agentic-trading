# test_signal_status_meanings.py

"""Every round-1 analyst's non-signal branches carry the status that says what happened:
`unavailable` when the data the analyst needs is not there, `active` flat at confidence 0 when
the model has the data and declines to bet, and `error` when the model or a loader fails. Each
branch is reached by stubbing a loader or a quantcore fit on the analyst's own module, so no
test reads the lake. A loader exception counts as `error` whether the analyst returns an error
signal or lets it propagate to `ManagerAgent.collect_signals`, which records it as one."""

import ast
import dataclasses
import math
from datetime import date
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from quantcore.core.stochastic import CIRNoMeanReversionError
from quantcore.risk import ConvergenceError
from quantcore.risk.egarch import EGARCHParams, GJRGARCHParams

from agentic_trading.agents.analysts.alt_data_pod import (
    calendar_seasonality_analyst,
    insider_cluster_buying_analyst,
    network_centrality_analyst,
)
from agentic_trading.agents.analysts.credit_pod import credit_spread_signal_analyst
from agentic_trading.agents.analysts.factor_pod import (
    earnings_surprise_analyst,
    fama_french_factor_analyst,
    fama_macbeth_analyst,
    newey_west_significance_analyst,
)
from agentic_trading.agents.analysts.macro_rates_pod import (
    arima_forecast_analyst,
    cir_rate_regime_analyst,
    cot_positioning_analyst,
    egarch_asymmetric_vol_analyst,
    gjr_garch_crash_risk_analyst,
    macro_var_spillover_analyst,
)
from agentic_trading.agents.analysts.macro_rates_pod._garch_checks import (
    FINAL_SIGMA_RATIO_BOUNDS,
    MAX_SIGMA_PATH_RANGE_RATIO,
    MIN_EGARCH_BETA,
)
from agentic_trading.agents.analysts.stat_arb_pod import (
    kalman_hedge_ratio_analyst,
    ou_mean_reversion_analyst,
    pairs_cointegration_analyst,
    regime_hmm_analyst,
)
from agentic_trading.agents.base import ManagerAgent
from agentic_trading.agents.managers.cross_asset_manager import FX_ASSETS
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings
from agentic_trading.data.loaders.lakehouse_loader import IndexMember, ken_french_public_date
from agentic_trading.data.loaders.macro_loader import DGS3MO, DGS10, VIXCLS

AS_OF = pd.Timestamp("2024-06-14")
# The `source_as_of` of the dated stubs below: their newest value becomes known on AS_OF.
PUBLISHED = date(2024, 6, 14)
# A power of two: copies of it sum and average exactly, so their sample std is exactly 0.
EXACT = 2.0**-7


class _Recorder:
    """A stand-in loader or fit that records each call and returns `value`, or raises `error`."""

    def __init__(self, value: Any = None, error: BaseException | None = None) -> None:
        self.value = value
        self.error = error
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(args)
        if self.error is not None:
            raise self.error
        return self.value


def _stub(outcome: Any) -> _Recorder:
    """A recorder that raises `outcome` when it is an exception and returns it otherwise."""
    if isinstance(outcome, BaseException):
        return _Recorder(error=outcome)
    return _Recorder(outcome)


def _noise(size: int, seed: int, scale: float = 0.01) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, size)


def _market(returns: np.ndarray, *, close: bool = True) -> pd.DataFrame:
    """A single-asset frame of business-day `log_return`s ending AS_OF, with closes if `close`."""
    index = pd.bdate_range(end=AS_OF, periods=len(returns))
    frame = pd.DataFrame({"log_return": np.asarray(returns, dtype=np.float64)}, index=index)
    if close:
        frame["close"] = 100.0 * np.exp(np.cumsum(frame["log_return"].to_numpy()))
    return frame


def _vintages(values: Any, index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """`values` as FRED vintages, each known the day after its date (default: days before AS_OF)."""
    if index is None:
        index = pd.bdate_range(end=AS_OF - pd.offsets.BDay(1), periods=len(values))
    known = index + pd.Timedelta(days=1)
    return pd.DataFrame(
        {"value": np.asarray(values, dtype=np.float64), "realtime_start": known, "known_at": known},
        index=index,
    )


def _recorded(analyst: Any, asset: str, market_data: pd.DataFrame) -> AgentSignal:
    """The signal `ManagerAgent.collect_signals` records for `analyst` on AS_OF."""
    holder = SimpleNamespace(managed_analysts=[analyst])
    return ManagerAgent.collect_signals(holder, asset, market_data, AS_OF)[0]


def _assert_flat(
    signal: AgentSignal,
    status: str,
    source_as_of: date | None = None,
    sample_count: int | None = None,
) -> None:
    """`signal` is flat at confidence 0 with `status`, and the provenance given when not None."""
    assert (signal.status, signal.direction, signal.confidence) == (status, "flat", 0.0)
    if source_as_of is not None:
        assert signal.source_as_of == source_as_of
    if sample_count is not None:
        assert signal.sample_count == sample_count


# alt_data_pod


@pytest.mark.parametrize(
    ("market", "status", "sample_count"),
    [
        (pd.DataFrame({"close": [100.0]}, index=[AS_OF]), "unavailable", None),
        (pd.concat({"MSFT": _market(_noise(300, 1))}, axis=1), "unavailable", None),
        (_market(np.full(10, 0.001)), "unavailable", 2),
        (_market(np.full(1500, EXACT)), "active", None),
        (_market(np.full(1500, -EXACT)), "active", None),
        (_market(np.zeros(1500)), "active", None),
    ],
    ids=[
        "no_returns",
        "asset_not_in_frame",
        "few_regime_days",
        "zero_std_positive_mean",
        "zero_std_negative_mean",
        "zero_returns",
    ],
)
def test_calendar_seasonality(market: pd.DataFrame, status: str, sample_count: int | None) -> None:
    """No returns or too few regime days is unavailable; a zero-variance regime mean abstains."""
    analyst = calendar_seasonality_analyst.CalendarSeasonalityAnalyst()
    _assert_flat(analyst.generate_signal("AAPL", market, AS_OF), status, sample_count=sample_count)


def _filings(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    days = pd.to_datetime([row[2] for row in rows])
    return pd.DataFrame(
        {
            "owner_cik": [row[0] for row in rows],
            "transaction_code": [row[1] for row in rows],
            "filed_date": days,
            "transaction_date": days,
            "shares": [100.0] * len(rows),
        }
    )


@pytest.mark.parametrize(
    ("asset", "filings", "status", "source_as_of", "sample_count"),
    [
        ("GC=F", None, "unavailable", None, None),
        ("AAPL", None, "unavailable", None, None),
        ("AAPL", _filings([]), "unavailable", None, None),
        (
            "AAPL",
            _filings([(1, "P", "2024-06-05"), (1, "P", "2024-06-10")]),
            "active",
            date(2024, 6, 10),
            2,
        ),
    ],
    ids=["non_equity", "no_coverage", "no_filings", "single_buyer"],
)
def test_insider_cluster_buying(
    monkeypatch: pytest.MonkeyPatch,
    asset: str,
    filings: pd.DataFrame | None,
    status: str,
    source_as_of: date | None,
    sample_count: int | None,
) -> None:
    """A non-equity or no Form 4 data is unavailable; buyers under the minimum abstain, dated."""
    module = insider_cluster_buying_analyst
    monkeypatch.setattr(module, "load_insider_transactions_from_lakehouse", _Recorder(filings))

    signal = module.InsiderClusterBuyingAnalyst().generate_signal(
        asset, _market(np.zeros(30)), AS_OF
    )

    _assert_flat(signal, status, source_as_of, sample_count)


_LEAD_LAG_DATES = pd.bdate_range(end=AS_OF, periods=71)
_LEAD_LAG_MEMBERS = [
    IndexMember(security_id=n, ticker=t)
    for n, t in enumerate(["AAPL", *(f"T{n:03d}" for n in range(34))])
]


# Kinds of member returns: "full" (every member on the same 71 days), "short" (10 days each),
# "thin" (too few members with enough history), "disjoint" (the other members only on days
# AAPL never trades) and "zero_others" (the other members flat at zero).
def _lead_lag_universe(kind: str) -> dict[int, pd.Series]:
    """Member returns by `security_id` for the lead-lag cross-section of the given `kind`."""
    old_dates = pd.bdate_range(end="2023-06-30", periods=71)
    returns = {}
    for member in _LEAD_LAG_MEMBERS:
        other = member.ticker != "AAPL"
        seed = member.security_id
        values = np.zeros(71) if kind == "zero_others" and other else _noise(71, seed)
        dates = old_dates if kind == "disjoint" and other else _LEAD_LAG_DATES
        returns[member.security_id] = pd.Series(values, index=dates)
    if kind == "short":
        returns = {security: series.iloc[-10:] for security, series in returns.items()}
    if kind == "thin":
        returns = dict(list(returns.items())[: network_centrality_analyst._MIN_UNIVERSE_SIZE - 1])
    return returns


@pytest.mark.parametrize(
    ("asset", "members", "kind", "status", "source_as_of"),
    [
        ("AAPL", None, "full", "unavailable", None),
        ("AAPL", _LEAD_LAG_MEMBERS, "short", "unavailable", None),
        ("AAPL", _LEAD_LAG_MEMBERS, "thin", "unavailable", None),
        ("SPY", _LEAD_LAG_MEMBERS, "full", "unavailable", None),
        ("AAPL", _LEAD_LAG_MEMBERS, "disjoint", "unavailable", None),
        ("AAPL", _LEAD_LAG_MEMBERS, "zero_others", "active", AS_OF.date()),
    ],
    ids=[
        "no_membership",
        "short_history",
        "too_few_members",
        "not_a_member",
        "no_other_names",
        "zero_forecast",
    ],
)
def test_lead_lag_cross_section(
    monkeypatch: pytest.MonkeyPatch,
    asset: str,
    members: list[IndexMember] | None,
    kind: str,
    status: str,
    source_as_of: date | None,
) -> None:
    """No membership, too few usable members, a non-member or no overlap is unavailable."""
    module = network_centrality_analyst
    monkeypatch.setattr(module, "load_sp500_member_securities_asof", _Recorder(members))
    monkeypatch.setattr(module, "load_security_log_returns", _Recorder(_lead_lag_universe(kind)))

    signal = _recorded(module.LeadLagNetworkCentralityAnalyst(), asset, pd.DataFrame())

    _assert_flat(signal, status, source_as_of)


@pytest.mark.parametrize(
    ("failing", "error"),
    [
        ("load_sp500_member_securities_asof", RuntimeError("lake down")),
        ("load_sp500_member_securities_asof", ValueError("bad response")),
        ("load_security_log_returns", RuntimeError("lake down")),
        ("load_security_log_returns", ValueError("bad response")),
    ],
)
def test_lead_lag_broad_fetch_failure(
    monkeypatch: pytest.MonkeyPatch, failing: str, error: Exception
) -> None:
    """A membership or returns fetch that raises is an error."""
    module = network_centrality_analyst
    monkeypatch.setattr(module, "load_sp500_member_securities_asof", _Recorder(_LEAD_LAG_MEMBERS))
    monkeypatch.setattr(module, "load_security_log_returns", _Recorder(_lead_lag_universe("full")))
    monkeypatch.setattr(module, failing, _Recorder(error=error))

    signal = _recorded(module.LeadLagNetworkCentralityAnalyst(), "AAPL", pd.DataFrame())

    _assert_flat(signal, "error")


# stat_arb_pod


@pytest.mark.parametrize(
    ("rows", "half_life", "direction", "status"),
    [
        (5, 10.0, "long", "unavailable"),
        (120, ValueError("phi >= 0, diverging"), "long", "active"),
        (120, 10.0, "flat", "active"),
    ],
    ids=["short_spread", "not_mean_reverting", "inside_exit_band"],
)
def test_ou_mean_reversion(
    monkeypatch: pytest.MonkeyPatch, rows: int, half_life: Any, direction: str, status: str
) -> None:
    """A short spread is unavailable; no mean reversion or no entry abstains; each closes out."""
    module = ou_mean_reversion_analyst
    monkeypatch.setattr(module, "ou_half_life", _stub(half_life))
    monkeypatch.setattr(module, "spread_direction", _Recorder(direction))
    analyst = module.OUMeanReversionAnalyst()
    analyst._open_direction_by_asset["AAPL"] = "long"

    _assert_flat(analyst.generate_signal("AAPL", _market(_noise(rows, 7)), AS_OF), status)
    assert "AAPL" not in analyst._open_direction_by_asset


@pytest.mark.parametrize("direction", ["long", "short"])
def test_ou_mean_reversion_simulation_disagreement(
    monkeypatch: pytest.MonkeyPatch, direction: str
) -> None:
    """Simulated OU paths that do not revert in the trade's direction give active flat 0 and
    close out the open direction."""
    module = ou_mean_reversion_analyst
    monkeypatch.setattr(module, "ou_half_life", _Recorder(10.0))
    monkeypatch.setattr(module, "spread_direction", _Recorder(direction))
    monkeypatch.setattr(
        module, "simulate_ou_paths", lambda _t, _m, _s, start, **_k: np.full((500, 51), start)
    )
    analyst = module.OUMeanReversionAnalyst()
    analyst._open_direction_by_asset["AAPL"] = direction

    signal = analyst.generate_signal("AAPL", _market(_noise(120, 7)), AS_OF)

    _assert_flat(signal, "active")
    assert "AAPL" not in analyst._open_direction_by_asset


def _pair_analyst(kind: str, paired: bool) -> Any:
    if kind == "pairs":
        partners = {"AAPL": ("MSFT",)} if paired else {}
        return pairs_cointegration_analyst.PairsCointegrationAnalyst(
            partner_tickers_by_asset=partners
        )
    partner = {"AAPL": "MSFT"} if paired else {}
    return kalman_hedge_ratio_analyst.KalmanHedgeRatioAnalyst(partner_ticker_by_asset=partner)


@pytest.mark.parametrize("kind", ["pairs", "kalman"])
@pytest.mark.parametrize(
    ("paired", "rows", "status"),
    [(False, 120, "unavailable"), (True, 20, "unavailable"), (True, 120, "active")],
    ids=["no_partner", "short_overlap", "inside_exit_band"],
)
def test_pair_spread_analysts(
    monkeypatch: pytest.MonkeyPatch, kind: str, paired: bool, rows: int, status: str
) -> None:
    """No partner or a short overlap is unavailable; no entry abstains; each closes out."""
    module = pairs_cointegration_analyst if kind == "pairs" else kalman_hedge_ratio_analyst
    monkeypatch.setattr(module, "spread_direction", _Recorder("flat"))
    monkeypatch.setattr(
        kalman_hedge_ratio_analyst, "dynamic_hedge_ratio", lambda y, _x: np.ones(len(y))
    )
    market = pd.concat({"AAPL": _market(_noise(rows, 3)), "MSFT": _market(_noise(rows, 4))}, axis=1)
    analyst = _pair_analyst(kind, paired)
    analyst._open_direction_by_asset["AAPL"] = "long"

    _assert_flat(analyst.generate_signal("AAPL", market, AS_OF), status)
    assert "AAPL" not in analyst._open_direction_by_asset


_HMM_TRANSITION = ((0.95, 0.05), (0.10, 0.90))
_HMM_VARIANCES = (1e-4, 4e-4)


def _hmm_fit(
    transition: Any = _HMM_TRANSITION, variances: Any = _HMM_VARIANCES
) -> tuple[np.ndarray, ...]:
    """Two-state parameters as `hmm_fit` returns them: transitions, means, variances, initial."""
    return (
        np.asarray(transition, dtype=float),
        np.array([0.001, -0.002]),
        np.asarray(variances, dtype=float),
        np.array([0.5, 0.5]),
    )


def _hmm_decoder(posterior: Any) -> Any:
    """A stand-in `hmm_predict_proba` that raises `posterior` or repeats it on every row."""

    def _decode(observations: np.ndarray, *_params: Any) -> np.ndarray:
        if isinstance(posterior, BaseException):
            raise posterior
        return np.tile(np.asarray(posterior, dtype=float), (len(observations), 1))

    return _decode


@pytest.mark.parametrize(
    ("rows", "fit", "posterior", "status"),
    [
        (
            settings.analysts.hmm_min_observations - 1,
            AssertionError("no fit below the observation floor"),
            (0.5, 0.5),
            "unavailable",
        ),
        (600, RuntimeError("EM did not converge"), (0.5, 0.5), "error"),
        (600, ValueError("singular covariance"), (0.5, 0.5), "error"),
        (600, np.linalg.LinAlgError("SVD did not converge"), (0.5, 0.5), "error"),
        (600, _hmm_fit(), ValueError("decode failed"), "error"),
        (600, _hmm_fit(variances=(1e-4, 1.2e-4)), (0.5, 0.5), "active"),
        (600, _hmm_fit(), (0.98, 0.02), "active"),
        (600, _hmm_fit(transition=((0.95, 0.05), (0.5, 0.5))), (0.5, 0.5), "active"),
    ],
    ids=[
        "short_history",
        "fit_runtime_error",
        "fit_value_error",
        "fit_linalg_error",
        "decode_error",
        "variance_ratio",
        "state_occupancy",
        "transition_diagonal",
    ],
)
def test_regime_hmm(
    monkeypatch: pytest.MonkeyPatch, rows: int, fit: Any, posterior: Any, status: str
) -> None:
    """Short history is unavailable; a failed fit or decode errors; a degenerate fit abstains."""
    module = regime_hmm_analyst
    monkeypatch.setattr(module, "hmm_fit", _stub(fit))
    monkeypatch.setattr(module, "hmm_predict_proba", _hmm_decoder(posterior))

    signal = module.RegimeHMMAnalyst().generate_signal("AAPL", _market(_noise(rows, 6)), AS_OF)

    _assert_flat(signal, status)


# macro_rates_pod: GARCH family

_GARCH = {
    "egarch": (
        egarch_asymmetric_vol_analyst,
        "EGARCHAsymmetricVolAnalyst",
        "egarch_fit",
        "egarch_11_variance",
    ),
    "gjr": (
        gjr_garch_crash_risk_analyst,
        "GJRGARCHCrashRiskAnalyst",
        "gjr_garch_fit",
        "gjr_garch_11_variance",
    ),
}
_GARCH_RETURNS = _noise(800, 11)
_GARCH_SHORT = _GARCH_RETURNS[-(settings.analysts.min_garch_observations - 1) :]
_GARCH_STD = float(np.std(_GARCH_RETURNS[-settings.analysts.risk_window_garch :], ddof=1))
_GARCH_PATH = np.full(settings.analysts.risk_window_garch, _GARCH_STD)
# EGARCH's implied unconditional vol is exp(omega / (2 (1 - beta))); at beta 0.9 this times
# log(vol) is the omega that implies `vol`.
_EGARCH_OMEGA_PER_LOG_VOL = 2.0 * (1.0 - 0.9)
_LOW, _HIGH = FINAL_SIGMA_RATIO_BOUNDS


def _admissible(name: str) -> EGARCHParams | GJRGARCHParams:
    """Parameters passing every check, implying the sample std, with no EGARCH leverage view."""
    if name == "egarch":
        return EGARCHParams(
            omega=_EGARCH_OMEGA_PER_LOG_VOL * math.log(_GARCH_STD),
            alpha=0.1,
            gamma=math.copysign(0.1, _GARCH_RETURNS[-1]),
            beta=0.9,
            log_likelihood=0.0,
        )
    return GJRGARCHParams(
        omega=(1.0 - 0.95) * _GARCH_STD**2, alpha=0.05, gamma=0.1, beta=0.85, log_likelihood=0.0
    )


def _variance_output(name: str, conditional_std: np.ndarray) -> np.ndarray:
    """The recursion output giving `conditional_std`: log variance (EGARCH) or variance (GJR)."""
    squared = np.square(np.asarray(conditional_std, dtype=np.float64))
    if name == "egarch":
        with np.errstate(divide="ignore"):
            return np.log(squared)
    return squared


def _garch_signal(
    monkeypatch: pytest.MonkeyPatch, name: str, fit: Any, variance: Any, returns: np.ndarray
) -> AgentSignal:
    """The `name` analyst's signal on `returns`, its fit and variance recursion stubbed."""
    module, class_name, fit_name, variance_name = _GARCH[name]
    monkeypatch.setattr(module, fit_name, _stub(fit))
    monkeypatch.setattr(module, variance_name, _stub(variance))
    return getattr(module, class_name)().generate_signal("AAPL", _market(returns), AS_OF)


def _convergence_error(params: tuple[float, ...] | None) -> ConvergenceError:
    return ConvergenceError(
        "no start converged",
        params=params,
        log_likelihood=0.0,
        optimizer_message="Iteration limit reached",
    )


@pytest.mark.parametrize("name", ["egarch", "gjr"])
@pytest.mark.parametrize(
    ("fit", "variance", "returns", "status"),
    [
        (AssertionError("no fit below the floor"), None, _GARCH_SHORT, "unavailable"),
        (ValueError("bad input"), None, _GARCH_RETURNS, "error"),
        (RuntimeError("optimizer failed"), None, _GARCH_RETURNS, "error"),
        (_convergence_error(None), None, _GARCH_RETURNS, "error"),
        (_convergence_error((1e-6, 0.05, 0.1, 0.8)), None, _GARCH_RETURNS, "error"),
        ("admissible", ValueError("recursion failed"), _GARCH_RETURNS, "error"),
        ("admissible", RuntimeError("recursion failed"), _GARCH_RETURNS, "error"),
        ("admissible", ZeroDivisionError("recursion failed"), _GARCH_RETURNS, "error"),
    ],
    ids=[
        "short_history",
        "fit_value_error",
        "fit_runtime_error",
        "no_converged_attempt",
        "best_attempt_not_converged",
        "variance_value_error",
        "variance_runtime_error",
        "variance_zero_division",
    ],
)
def test_garch_fit_failures(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    fit: Any,
    variance: Any,
    returns: np.ndarray,
    status: str,
) -> None:
    """Under the observation floor is unavailable; a failed or non-converged fit errors."""
    fitted = _admissible(name) if isinstance(fit, str) else fit
    _assert_flat(_garch_signal(monkeypatch, name, fitted, variance, returns), status)


@pytest.mark.parametrize(
    ("name", "changes", "status"),
    [
        ("egarch", {"beta": MIN_EGARCH_BETA - 0.1}, "active"),
        ("egarch", {"beta": (1.0 + settings.analysts.max_garch_persistence) / 2.0}, "active"),
        ("egarch", {"omega": _EGARCH_OMEGA_PER_LOG_VOL * math.log(10.0 * _GARCH_STD)}, "active"),
        ("egarch", {"omega": _EGARCH_OMEGA_PER_LOG_VOL * math.log(_GARCH_STD / 10.0)}, "active"),
        ("gjr", {"omega": 0.0}, "error"),
        ("gjr", {"omega": -1e-7}, "error"),
        ("gjr", {"alpha": -0.01}, "error"),
        ("gjr", {"beta": -0.1}, "error"),
        ("gjr", {"gamma": -0.1}, "error"),
        ("gjr", {"alpha": 0.1, "beta": 0.9}, "active"),
        ("gjr", {"omega": (1.0 - 0.95) * (10.0 * _GARCH_STD) ** 2}, "active"),
        ("gjr", {"omega": (1.0 - 0.95) * (_GARCH_STD / 10.0) ** 2}, "active"),
    ],
    ids=[
        "egarch_beta_below_floor",
        "egarch_beta_above_max",
        "egarch_vol_too_high",
        "egarch_vol_too_low",
        "gjr_zero_omega",
        "gjr_negative_omega",
        "gjr_negative_alpha",
        "gjr_negative_beta",
        "gjr_negative_alpha_plus_gamma",
        "gjr_over_persistent",
        "gjr_vol_too_high",
        "gjr_vol_too_low",
    ],
)
def test_garch_parameter_checks(
    monkeypatch: pytest.MonkeyPatch, name: str, changes: dict[str, float], status: str
) -> None:
    """An inadmissible GJR fit errors; every other failed parameter check abstains."""
    fitted = dataclasses.replace(_admissible(name), **changes)
    variance = AssertionError("no variance path after a failed parameter check")
    _assert_flat(_garch_signal(monkeypatch, name, fitted, variance, _GARCH_RETURNS), status)


def _with(index: int, value: float) -> np.ndarray:
    path = _GARCH_PATH.copy()
    path[index] = value
    return path


@pytest.mark.parametrize("name", ["egarch", "gjr"])
@pytest.mark.parametrize(
    ("conditional_std", "status"),
    [
        (np.array([]), "error"),
        (_with(10, math.nan), "error"),
        (_with(10, math.inf), "error"),
        (_with(10, 0.0), "error"),
        (_with(0, _GARCH_STD / (MAX_SIGMA_PATH_RANGE_RATIO + 5.0)), "active"),
        (_GARCH_PATH * (_HIGH + 1.0), "active"),
        (_GARCH_PATH * (_LOW / 2.0), "active"),
        (_GARCH_PATH, "active"),
    ],
    ids=["empty", "nan", "inf", "zero", "range", "final_high", "final_low", "no_view"],
)
def test_garch_sigma_path_checks(
    monkeypatch: pytest.MonkeyPatch, name: str, conditional_std: np.ndarray, status: str
) -> None:
    """A broken conditional-vol path errors; a range or final breach, or no view, abstains."""
    variance = _variance_output(name, conditional_std)
    signal = _garch_signal(monkeypatch, name, _admissible(name), variance, _GARCH_RETURNS)
    _assert_flat(signal, status)


# macro_rates_pod: ARIMA, VAR, CIR, COT


def _ljung_box(*p_values: float) -> Any:
    """A stand-in `ljung_box_test` returning `p_values` in order, one per call."""
    remaining = iter(p_values)
    return lambda *_args, **_kwargs: (0.0, next(remaining))


_ARIMA_FIT = (np.array([0.1]), np.array([]), 1e-4)
_ARIMA_MARKET = _market(_noise(300, 5))


@pytest.mark.parametrize(
    ("market", "p_values", "order", "fit", "status"),
    [
        (_market(_noise(300, 5), close=False), (), (1, 0, 0), _ARIMA_FIT, "unavailable"),
        (_market(_noise(20, 5)), (), (1, 0, 0), _ARIMA_FIT, "unavailable"),
        (
            _ARIMA_MARKET,
            (0.5,),
            AssertionError("no order selection without serial dependence"),
            AssertionError("no fit without serial dependence"),
            "active",
        ),
        (_ARIMA_MARKET, (0.001,), RuntimeError("grid failed"), _ARIMA_FIT, "error"),
        (_ARIMA_MARKET, (0.001,), ValueError("grid failed"), _ARIMA_FIT, "error"),
        (_ARIMA_MARKET, (0.001,), (1, 0, 0), _convergence_error((0.1,)), "error"),
        (_ARIMA_MARKET, (0.001,), (1, 0, 0), ValueError("non-stationary"), "error"),
        (_ARIMA_MARKET, (0.001, 0.001), (1, 0, 0), _ARIMA_FIT, "active"),
    ],
    ids=[
        "no_close",
        "short_history",
        "no_serial_dependence",
        "order_runtime_error",
        "order_value_error",
        "fit_not_converged",
        "fit_value_error",
        "residuals_autocorrelated",
    ],
)
def test_arima_forecast(
    monkeypatch: pytest.MonkeyPatch,
    market: pd.DataFrame,
    p_values: tuple[float, ...],
    order: Any,
    fit: Any,
    status: str,
) -> None:
    """Missing data is unavailable; a failed Ljung-Box gate abstains; a failed fit errors."""
    module = arima_forecast_analyst
    monkeypatch.setattr(module, "ljung_box_test", _ljung_box(*p_values))
    monkeypatch.setattr(module, "select_arima_order", _stub(order))
    monkeypatch.setattr(module, "arima_fit", _stub(fit))
    monkeypatch.setattr(module, "arima_residuals", _Recorder(np.zeros(249)))
    analyst = module.ARIMAForecastAnalyst(agent_id="arima_forecast_analyst_rates")

    _assert_flat(analyst.generate_signal("TLT", market, AS_OF), status)


_VAR_DATES = pd.bdate_range(end=AS_OF, periods=150)
_VAR_LEVELS = ((VIXCLS, 20.0), (DGS10, 4.0), (DGS3MO, 3.0))


def _var_macro(
    index: pd.DatetimeIndex = _VAR_DATES[:-1], missing: str | None = None
) -> dict[str, pd.DataFrame | None]:
    """VIXCLS, DGS10 and DGS3MO vintages on `index`, with `missing` absent from the lake."""
    frames: dict[str, pd.DataFrame | None] = {
        series_id: _vintages(level + _noise(len(index), seed, 1.0), index)
        for seed, (series_id, level) in enumerate(_VAR_LEVELS)
    }
    if missing is not None:
        frames[missing] = None
    return frames


_VAR_MARKET = _market(_noise(150, 8))
_NO_VAR_FIT = AssertionError("no VAR fit without Granger causality")


@pytest.mark.parametrize(
    ("market", "macro", "granger", "var_fit", "status", "source_as_of"),
    [
        (_market(_noise(20, 8)), _var_macro(), (0.0, 0.01), _NO_VAR_FIT, "unavailable", None),
        (_VAR_MARKET, _var_macro(missing=DGS10), (0.0, 0.01), _NO_VAR_FIT, "unavailable", None),
        (_VAR_MARKET[["close"]], _var_macro(), (0.0, 0.01), _NO_VAR_FIT, "unavailable", None),
        (
            _VAR_MARKET,
            _var_macro(_VAR_DATES[-21:-1]),
            (0.0, 0.01),
            _NO_VAR_FIT,
            "unavailable",
            None,
        ),
        (_VAR_MARKET, _var_macro(), (0.0, 0.9), _NO_VAR_FIT, "active", PUBLISHED),
        (_VAR_MARKET, _var_macro(), ValueError("granger failed"), _NO_VAR_FIT, "error", None),
        (
            _VAR_MARKET,
            _var_macro(),
            (0.0, 0.01),
            np.linalg.LinAlgError("SVD did not converge"),
            "error",
            None,
        ),
    ],
    ids=[
        "short_asset_history",
        "missing_macro_series",
        "no_log_returns",
        "short_overlap",
        "no_granger_causality",
        "granger_error",
        "var_fit_error",
    ],
)
def test_macro_var_spillover(
    monkeypatch: pytest.MonkeyPatch,
    market: pd.DataFrame,
    macro: dict[str, pd.DataFrame | None],
    granger: Any,
    var_fit: Any,
    status: str,
    source_as_of: date | None,
) -> None:
    """Missing data is unavailable; no Granger causality abstains, dated; a failed fit errors."""
    module = macro_var_spillover_analyst
    monkeypatch.setattr(
        module, "load_fred_vintages_from_lakehouse", lambda series_id, _as_of: macro[series_id]
    )
    monkeypatch.setattr(module, "select_var_lag_order", _Recorder(1))
    monkeypatch.setattr(module, "granger_causality_test", _stub(granger))
    monkeypatch.setattr(module, "var_fit", _stub(var_fit))

    signal = module.MacroVarSpilloverAnalyst().generate_signal("SPY", market, AS_OF)

    _assert_flat(signal, status, source_as_of)


_WINDOW = settings.analysts.risk_window_long
_CIR_FIT = SimpleNamespace(kappa=1.0, theta=0.03, sigma=0.05)
# The messages quantcore's `cir_fit` raises for a fit outside the CIR parameter space. Only
# kappa <= 0 (no mean reversion in the data, `CIRNoMeanReversionError`) is an abstention.
_CIR_KAPPA = (
    "cir_fit: estimated kappa <= 0 -- input series shows no discernible "
    "mean reversion, so no admissible CIR fit exists for this data"
)
_CIR_THETA = "cir_fit: estimated theta <= 0 -- not an admissible CIR long-run mean"
_CIR_SIGMA = "cir_fit: estimated sigma^2 <= 0 -- not an admissible CIR diffusion coefficient"


def _rates(size: int = 300, at: int | None = None, value: float = 0.0) -> np.ndarray:
    """DGS3MO in percent around 3.0, with `value` at position `at`."""
    rates = 3.0 + _noise(size, 12, 0.02)
    if at is not None:
        rates[at] = value
    return rates


@pytest.mark.parametrize(
    ("vintages", "fit", "status", "source_as_of", "fit_called"),
    [
        (None, _CIR_FIT, "unavailable", None, False),
        (_vintages(_rates(20)), _CIR_FIT, "unavailable", None, False),
        (_vintages(_rates(at=-1)), _CIR_FIT, "active", PUBLISHED, False),
        (_vintages(_rates(at=-_WINDOW)), _CIR_FIT, "active", PUBLISHED, False),
        (_vintages(_rates(at=-100, value=-0.01)), _CIR_FIT, "active", PUBLISHED, False),
        (_vintages(_rates()), CIRNoMeanReversionError(_CIR_KAPPA), "active", PUBLISHED, True),
        (_vintages(_rates()), RuntimeError(_CIR_THETA), "error", None, True),
        (_vintages(_rates()), RuntimeError(_CIR_SIGMA), "error", None, True),
        (_vintages(_rates()), RuntimeError("cir_fit: unexpected"), "error", None, True),
        (_vintages(_rates()), ValueError("bad input"), "error", None, True),
    ],
    ids=[
        "no_coverage",
        "short_history",
        "zero_rate_latest",
        "zero_rate_window_start",
        "negative_rate",
        "no_mean_reversion",
        "inadmissible_theta",
        "inadmissible_sigma",
        "other_runtime_error",
        "fit_value_error",
    ],
)
def test_cir_rate_regime(
    monkeypatch: pytest.MonkeyPatch,
    vintages: pd.DataFrame | None,
    fit: Any,
    status: str,
    source_as_of: date | None,
    fit_called: bool,
) -> None:
    """A rate at or below zero in the window abstains without a fit. A fit with no mean reversion
    (kappa <= 0) abstains; any other fit failure, an inadmissible theta or sigma included, is an
    error."""
    module = cir_rate_regime_analyst
    fitter = _stub(fit)
    monkeypatch.setattr(module, "load_fred_vintages_from_lakehouse", _Recorder(vintages))
    monkeypatch.setattr(module, "cir_fit", fitter)
    monkeypatch.setattr(module, "simulate_cir_paths", _Recorder(np.full((500, 31), 0.03)))

    signal = module.CIRRateRegimeAnalyst().generate_signal("TLT", pd.DataFrame(), AS_OF)

    _assert_flat(signal, status, source_as_of)
    assert bool(fitter.calls) is fit_called


def test_cir_zero_rate_before_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero rate older than the trailing window does not stop the fit or the view."""
    module = cir_rate_regime_analyst
    fitter = _Recorder(_CIR_FIT)
    vintages = _vintages(_rates(at=-(_WINDOW + 1)))
    monkeypatch.setattr(module, "load_fred_vintages_from_lakehouse", _Recorder(vintages))
    monkeypatch.setattr(module, "cir_fit", fitter)
    monkeypatch.setattr(module, "simulate_cir_paths", _Recorder(np.full((500, 31), 0.03)))

    signal = module.CIRRateRegimeAnalyst().generate_signal("TLT", pd.DataFrame(), AS_OF)

    assert len(fitter.calls) == 1
    assert signal.status == "active"
    assert signal.direction in {"long", "short"}


def _cot_reports(values: list[float]) -> pd.DataFrame:
    """Weekly reports dated Tuesdays to 2024-06-11, each released three days later."""
    reports = pd.date_range(end="2024-06-11", periods=len(values), freq="W-TUE")
    return pd.DataFrame(
        {
            "net_positioning_ratio": np.asarray(values, dtype=np.float64),
            "published_date": reports + pd.Timedelta(days=3),
        },
        index=reports,
    )


@pytest.mark.parametrize(
    ("asset", "reports", "status", "source_as_of"),
    [
        ("GC=F", _cot_reports([0.25, 0.75] * 15), "unavailable", None),
        ("EURUSD=X", None, "unavailable", None),
        ("EURUSD=X", _cot_reports([0.25, 0.75] * 4), "unavailable", None),
        ("EURUSD=X", _cot_reports([0.25] * 30), "active", PUBLISHED),
        ("EURUSD=X", _cot_reports([0.25, 0.75] * 15 + [0.5]), "active", PUBLISHED),
    ],
    ids=["outside_asset_class", "no_coverage", "short_history", "zero_variance", "zero_z_score"],
)
def test_cot_positioning(
    monkeypatch: pytest.MonkeyPatch,
    asset: str,
    reports: pd.DataFrame | None,
    status: str,
    source_as_of: date | None,
) -> None:
    """No coverage or history is unavailable; zero variance or a zero z-score abstains, dated."""
    module = cot_positioning_analyst
    monkeypatch.setattr(module, "load_cot_reports_from_lakehouse", _Recorder(reports))
    analyst = module.COTPositioningAnalyst(
        agent_id="cot_positioning_analyst_fx", covered_assets=FX_ASSETS
    )

    _assert_flat(analyst.generate_signal(asset, pd.DataFrame(), AS_OF), status, source_as_of)


# credit_pod


@pytest.mark.parametrize(
    ("vintages", "status", "source_as_of"),
    [
        (None, "unavailable", None),
        (_vintages([4.0] * 10), "unavailable", None),
        (_vintages([4.0] * 120), "active", PUBLISHED),
        (_vintages([3.5, 4.5] * 30 + [4.0]), "active", PUBLISHED),
    ],
    ids=["no_coverage", "short_history", "zero_variance", "zero_z_score"],
)
def test_credit_spread_signal(
    monkeypatch: pytest.MonkeyPatch,
    vintages: pd.DataFrame | None,
    status: str,
    source_as_of: date | None,
) -> None:
    """No coverage or history is unavailable; zero variance or a zero z-score abstains, dated."""
    module = credit_spread_signal_analyst
    monkeypatch.setattr(module, "load_fred_vintages_from_lakehouse", _Recorder(vintages))

    signal = module.CreditSpreadSignalAnalyst().generate_signal("HYG", pd.DataFrame(), AS_OF)

    _assert_flat(signal, status, source_as_of)


# factor_pod


def _announced(days_ago: int) -> pd.Timestamp:
    return AS_OF - pd.Timedelta(days=days_ago)


@pytest.mark.parametrize(
    ("result", "status", "source_as_of"),
    [
        (None, "unavailable", None),
        ((1.10, 1.00, _announced(61)), "active", _announced(61).date()),
        ((0.02, 0.005, _announced(10)), "active", _announced(10).date()),
        ((-0.02, -0.009, _announced(10)), "active", _announced(10).date()),
        ((0.03, 0.0, _announced(10)), "active", _announced(10).date()),
        ((1.00, 1.00, _announced(10)), "active", _announced(10).date()),
    ],
    ids=[
        "no_coverage",
        "past_drift_window",
        "tiny_positive_consensus",
        "tiny_negative_consensus",
        "zero_consensus",
        "zero_surprise",
    ],
)
def test_earnings_surprise(
    monkeypatch: pytest.MonkeyPatch, result: Any, status: str, source_as_of: date | None
) -> None:
    """No coverage is unavailable; a stale announcement, tiny consensus or no surprise abstains."""
    module = earnings_surprise_analyst
    monkeypatch.setattr(module, "load_earnings_surprise_from_lakehouse", _Recorder(result))

    signal = module.EarningsSurpriseAnalyst().generate_signal("AAPL", pd.DataFrame(), AS_OF)

    _assert_flat(signal, status, source_as_of)


def _factors(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """Daily Ken French 3-factor returns on `index`."""
    values = np.random.default_rng(seed).normal(0.0, 0.01, (len(index), 3))
    frame = pd.DataFrame(values, index=index, columns=["Mkt-RF", "SMB", "HML"])
    frame["RF"] = 0.0001
    return frame


_FF_FACTORS = _factors(pd.bdate_range("2022-01-03", "2024-04-30"), seed=14)
_FF_SHORT = _factors(pd.bdate_range("2024-03-01", "2024-04-30"), seed=14)
_FF_PUBLIC = ken_french_public_date(date(2024, 4, 30))
_INSIGNIFICANT = SimpleNamespace(loadings=np.array([0.0, 0.5, 0.5, 0.5]), t_stats=np.zeros(4))
_NEGLIGIBLE = SimpleNamespace(
    loadings=np.array([0.0, 1e-9, 0.0, 0.0]), t_stats=np.array([0.0, 10.0, 0.0, 0.0])
)


@pytest.mark.parametrize(
    ("factors", "regression", "status", "source_as_of", "sample_count"),
    [
        (None, _INSIGNIFICANT, "unavailable", None, None),
        (_FF_SHORT, _INSIGNIFICANT, "unavailable", None, len(_FF_SHORT)),
        (_FF_FACTORS, _INSIGNIFICANT, "active", _FF_PUBLIC, None),
        (_FF_FACTORS, _NEGLIGIBLE, "active", _FF_PUBLIC, None),
    ],
    ids=["no_coverage", "short_overlap", "no_significant_loading", "under_flat_threshold"],
)
def test_fama_french_factor(
    monkeypatch: pytest.MonkeyPatch,
    factors: pd.DataFrame | None,
    regression: SimpleNamespace,
    status: str,
    source_as_of: date | None,
    sample_count: int | None,
) -> None:
    """No factors or overlap is unavailable; no significant loading or a tiny z-score abstains."""
    module = fama_french_factor_analyst
    monkeypatch.setattr(module, "load_fama_french_factors_from_lakehouse", _Recorder(factors))
    monkeypatch.setattr(module, "factor_loadings", _Recorder(regression))

    signal = module.FamaFrenchFactorAnalyst().generate_signal(
        "AAPL", _market(_noise(300, 13)), AS_OF
    )

    _assert_flat(signal, status, source_as_of, sample_count)


_FM_DATES = pd.bdate_range(end=AS_OF, periods=250)
_FM_TICKERS = [f"T{n:03d}" for n in range(35)]
_FM_RETURNS = pd.DataFrame(
    np.random.default_rng(15).normal(0.0, 0.01, (len(_FM_DATES), len(_FM_TICKERS))),
    index=_FM_DATES,
    columns=_FM_TICKERS,
)
_FM_MEMBERS = [IndexMember(security_id=n, ticker=t) for n, t in enumerate(_FM_TICKERS)]
_FM_FACTORS = _factors(_FM_DATES, seed=16)
_FM_DEFAULTS: dict[str, Any] = {
    "load_sp500_member_securities_asof": _FM_MEMBERS,
    "load_security_log_returns": {m.security_id: _FM_RETURNS[m.ticker] for m in _FM_MEMBERS},
    "load_price_history_from_lakehouse": None,
    "load_fama_french_factors_from_lakehouse": _FM_FACTORS,
}
_NO_BROAD = {"load_sp500_member_securities_asof": None}
_THIN_BROAD = {
    "load_security_log_returns": {
        m.security_id: _FM_RETURNS[m.ticker]
        for m in _FM_MEMBERS[: fama_macbeth_analyst._MIN_UNIVERSE_SIZE - 1]
    }
}


@pytest.mark.parametrize(
    ("asset", "overrides", "status"),
    [
        ("T000", {"load_sp500_member_securities_asof": RuntimeError("lake down")}, "error"),
        ("T000", {"load_security_log_returns": ValueError("bad response")}, "error"),
        ("ZZZ", {}, "unavailable"),
        ("ZZZ", {"load_price_history_from_lakehouse": RuntimeError("lake down")}, "error"),
        ("T000", _NO_BROAD, "unavailable"),
        ("T000", _THIN_BROAD, "unavailable"),
        ("T000", {"load_fama_french_factors_from_lakehouse": None}, "unavailable"),
        (
            "T000",
            {"load_fama_french_factors_from_lakehouse": _FM_FACTORS.iloc[-20:]},
            "unavailable",
        ),
    ],
    ids=[
        "membership_raises",
        "security_returns_raise",
        "asset_not_in_lake",
        "asset_lookup_raises",
        "no_membership",
        "too_few_members",
        "no_factor_coverage",
        "short_factor_overlap",
    ],
)
def test_fama_macbeth(
    monkeypatch: pytest.MonkeyPatch, asset: str, overrides: dict[str, Any], status: str
) -> None:
    """A loader that raises errors; missing names, factors or history are unavailable."""
    for name, outcome in {**_FM_DEFAULTS, **overrides}.items():
        monkeypatch.setattr(fama_macbeth_analyst, name, _stub(outcome))

    signal = _recorded(fama_macbeth_analyst.FamaMacbethAnalyst(), asset, pd.DataFrame())

    _assert_flat(signal, status)


@pytest.mark.parametrize(
    ("rows", "covariance", "status"),
    [
        (30, np.eye(2), "unavailable"),
        (120, np.eye(2), "active"),
        (120, np.zeros((2, 2)), "active"),
    ],
    ids=["short_history", "insignificant", "zero_standard_error"],
)
def test_newey_west_significance(
    monkeypatch: pytest.MonkeyPatch, rows: int, covariance: np.ndarray, status: str
) -> None:
    """Too little history is unavailable; a HAC t-stat under the threshold abstains."""
    module = newey_west_significance_analyst
    monkeypatch.setattr(module, "ols", _Recorder((np.array([0.0, 0.001]), np.zeros(2), 0.0)))
    monkeypatch.setattr(module, "newey_west_optimal_lags", _Recorder(1))
    monkeypatch.setattr(module, "newey_west_cov", _Recorder(covariance))

    signal = module.NeweyWestSignificanceAnalyst().generate_signal(
        "AAPL", _market(_noise(rows, 18)), AS_OF
    )

    _assert_flat(signal, status)


# every round-1 module

_ROUND_ONE_MODULES: tuple[ModuleType, ...] = (
    calendar_seasonality_analyst,
    insider_cluster_buying_analyst,
    network_centrality_analyst,
    ou_mean_reversion_analyst,
    pairs_cointegration_analyst,
    kalman_hedge_ratio_analyst,
    regime_hmm_analyst,
    egarch_asymmetric_vol_analyst,
    gjr_garch_crash_risk_analyst,
    arima_forecast_analyst,
    macro_var_spillover_analyst,
    cir_rate_regime_analyst,
    cot_positioning_analyst,
    fama_french_factor_analyst,
    fama_macbeth_analyst,
    newey_west_significance_analyst,
    earnings_surprise_analyst,
    credit_spread_signal_analyst,
)
_BUILDERS = frozenset({"unavailable_signal", "abstain_signal", "error_signal"})


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_round_one_modules_build_status_signals_through_the_helpers() -> None:
    """No round-1 `AgentSignal(...)` call passes `status=`; each module calls a status helper."""
    assert len({module.__name__ for module in _ROUND_ONE_MODULES}) == 18
    offenders = []
    for module in _ROUND_ONE_MODULES:
        assert module.__file__ is not None
        path = Path(module.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        offenders += [
            f"{path.name}:{call.lineno}"
            for call in calls
            if _called_name(call) == "AgentSignal"
            and any(keyword.arg == "status" for keyword in call.keywords)
        ]
        assert any(_called_name(call) in _BUILDERS for call in calls), module.__name__
    assert offenders == []


def test_cir_no_mean_reversion_matches_quantcore() -> None:
    """quantcore's real `cir_fit` on a steadily rising rate series raises
    `CIRNoMeanReversionError`, which the analyst classifies as no mean reversion; a plain
    `RuntimeError` is not, whatever its message."""
    from quantcore.core.stochastic import cir_fit

    rising = np.linspace(0.01, 0.05, 250) * np.exp(_noise(250, 3, 0.001))
    with pytest.raises(CIRNoMeanReversionError) as raised:
        cir_fit(rising, 1.0 / 365.0)
    assert cir_rate_regime_analyst.is_no_mean_reversion(raised.value)
    assert not cir_rate_regime_analyst.is_no_mean_reversion(RuntimeError(_CIR_THETA))
    assert not cir_rate_regime_analyst.is_no_mean_reversion(RuntimeError(_CIR_SIGMA))
    assert not cir_rate_regime_analyst.is_no_mean_reversion(RuntimeError(_CIR_KAPPA))
