# test_signal_status_helpers.py

"""The three signal builders on `AnalystAgent` (`unavailable_signal`, `abstain_signal`,
`error_signal`) and the `CheckFailure` verdicts the GARCH sanity checks return: `error` for an
inadmissible fit or a broken variance path, `abstain` for a fit or path the model declines to
trade on."""

import dataclasses
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError
from quantcore.risk.egarch import EGARCHParams, GJRGARCHParams

from agentic_trading.agents.analysts import ARIMAForecastAnalyst
from agentic_trading.agents.analysts.macro_rates_pod._garch_checks import (
    FINAL_SIGMA_RATIO_BOUNDS,
    MAX_SIGMA_PATH_RANGE_RATIO,
    MIN_EGARCH_BETA,
    CheckFailure,
    check_egarch_params,
    check_gjr_params,
    check_sigma_path,
)
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.config import settings

AS_OF = pd.Timestamp("2024-06-14 15:30")
# A power of two, so every path built from it below has exact ratios.
SAMPLE_STD = 2.0**-7
_S = SAMPLE_STD
_LOW, _HIGH = FINAL_SIGMA_RATIO_BOUNDS
_MAX_PERSISTENCE = settings.analysts.max_garch_persistence


class _StubAnalyst(AnalystAgent):
    """A minimal concrete analyst for exercising the base-class builders."""

    agent_id: str = "stub_analyst"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        return self.abstain_signal(asset, as_of_date, "stub")


@pytest.mark.parametrize(
    ("builder", "status"),
    [
        ("unavailable_signal", "unavailable"),
        ("abstain_signal", "active"),
        ("error_signal", "error"),
    ],
)
def test_signal_builders(builder: str, status: str) -> None:
    """Each builder gives a flat 0-confidence signal with its status and the provenance passed."""
    analysts = (_StubAnalyst(), ARIMAForecastAnalyst(agent_id="arima_forecast_analyst_rates"))
    for analyst in analysts:
        build = getattr(analyst, builder)
        signal = build(
            asset="TLT",
            as_of_date=AS_OF,
            rationale="why",
            source_as_of=date(2024, 6, 13),
            sample_count=42,
        )
        assert isinstance(signal, AgentSignal)
        assert signal.model_dump() == {
            "agent_id": analyst.agent_id,
            "asset": "TLT",
            "as_of_date": date(2024, 6, 14),
            "direction": "flat",
            "confidence": 0.0,
            "rationale": "why",
            "status": status,
            "source_as_of": date(2024, 6, 13),
            "sample_count": 42,
        }

        bare = build("TLT", AS_OF, "why")
        assert (bare.status, bare.source_as_of, bare.sample_count) == (status, None, None)
        assert build("TLT", AS_OF, "why", sample_count=0).sample_count == 0
        with pytest.raises(TypeError):
            build("TLT", AS_OF, "why", date(2024, 6, 13))


def test_check_failure_model() -> None:
    """`CheckFailure` is a Pydantic model with `kind` "error" or "abstain" and a `reason`."""
    assert issubclass(CheckFailure, BaseModel)
    for kind in ("error", "abstain"):
        failure = CheckFailure(kind=kind, reason="why")
        assert (failure.kind, failure.reason) == (kind, "why")
    with pytest.raises(ValidationError):
        CheckFailure(kind="unavailable", reason="why")


def _assert_verdict(result: CheckFailure | None, kind: str | None) -> None:
    """`result` is None when `kind` is None, else a `CheckFailure` of `kind` with a reason."""
    if kind is None:
        assert result is None
        return
    assert isinstance(result, CheckFailure)
    assert result.kind == kind
    assert isinstance(result.reason, str) and result.reason


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ([], "error"),
        ([_S, math.nan, _S], "error"),
        ([_S, math.inf, _S], "error"),
        ([_S, 0.0, _S], "error"),
        ([_S, -_S, _S], "error"),
        ([_S / 100.0, math.nan, _S], "error"),
        ([_S / 100.0, 0.0, _S * 10.0], "error"),
        ([_S / 4.0, _S / 4.0 * MAX_SIGMA_PATH_RANGE_RATIO, _S], "abstain"),
        ([_S / 32.0, _S, _S], "abstain"),
        ([_S / 4.0, _S / 4.0 * (MAX_SIGMA_PATH_RANGE_RATIO - 1.0), _S], None),
        ([_S, _S, _S * (_HIGH + 1.0)], "abstain"),
        ([_S, _S, _S * _LOW / 2.0], "abstain"),
        ([_S / 100.0, _S, _S * (_HIGH + 1.0)], "abstain"),
        ([_S, _S, _S * (_HIGH - 1.0)], None),
        ([_S] * 10, None),
    ],
    ids=[
        "empty",
        "nan",
        "inf",
        "zero",
        "negative",
        "nan_before_range",
        "zero_before_range",
        "range_at_limit",
        "range_beyond_limit",
        "range_inside_limit",
        "final_above_band",
        "final_below_band",
        "range_and_final",
        "final_inside_band",
        "constant",
    ],
)
def test_check_sigma_path(path: list[float], kind: str | None) -> None:
    """An empty, non-finite or non-positive path is an error; a range or final breach abstains."""
    _assert_verdict(check_sigma_path(np.asarray(path, dtype=np.float64), SAMPLE_STD), kind)


_GJR = GJRGARCHParams(
    omega=(1.0 - 0.95) * _S**2, alpha=0.05, gamma=0.1, beta=0.85, log_likelihood=0.0
)


@pytest.mark.parametrize(
    ("changes", "kind"),
    [
        ({}, None),
        ({"omega": 0.0}, "error"),
        ({"omega": -1e-7}, "error"),
        ({"alpha": -0.01}, "error"),
        ({"beta": -0.1}, "error"),
        ({"gamma": -0.1}, "error"),
        ({"omega": 0.0, "alpha": 0.1, "beta": 0.9}, "error"),
        ({"alpha": 0.5, "gamma": -0.6, "beta": 0.9}, "error"),
        ({"alpha": 0.1, "beta": 0.9}, "abstain"),
        ({"omega": (1.0 - 0.95) * (10.0 * _S) ** 2}, "abstain"),
        ({"omega": (1.0 - 0.95) * (_S / 10.0) ** 2}, "abstain"),
    ],
    ids=[
        "admissible",
        "zero_omega",
        "negative_omega",
        "negative_alpha",
        "negative_beta",
        "negative_alpha_plus_gamma",
        "omega_checked_before_persistence",
        "alpha_plus_gamma_checked_before_persistence",
        "over_persistent",
        "unconditional_vol_too_high",
        "unconditional_vol_too_low",
    ],
)
def test_check_gjr_params(changes: dict[str, float], kind: str | None) -> None:
    """An inadmissible GJR fit is an error; a persistence or implied-vol breach abstains."""
    _assert_verdict(check_gjr_params(dataclasses.replace(_GJR, **changes), SAMPLE_STD), kind)


_EGARCH = EGARCHParams(
    omega=2.0 * (1.0 - 0.9) * math.log(_S), alpha=0.1, gamma=-0.1, beta=0.9, log_likelihood=0.0
)


@pytest.mark.parametrize(
    ("changes", "kind"),
    [
        ({}, None),
        ({"beta": MIN_EGARCH_BETA - 0.1}, "abstain"),
        ({"beta": MIN_EGARCH_BETA - 0.1, "omega": 50.0}, "abstain"),
        ({"beta": (1.0 + _MAX_PERSISTENCE) / 2.0}, "abstain"),
        ({"omega": 2.0 * (1.0 - 0.9) * math.log(10.0 * _S)}, "abstain"),
        ({"omega": 2.0 * (1.0 - 0.9) * math.log(_S / 10.0)}, "abstain"),
        ({"omega": 1e6}, "abstain"),
    ],
    ids=[
        "admissible",
        "beta_below_floor",
        "beta_below_floor_with_absurd_omega",
        "beta_above_max_persistence",
        "unconditional_vol_too_high",
        "unconditional_vol_too_low",
        "unconditional_vol_overflows",
    ],
)
def test_check_egarch_params(changes: dict[str, float], kind: str | None) -> None:
    """Every EGARCH parameter-check failure abstains."""
    _assert_verdict(check_egarch_params(dataclasses.replace(_EGARCH, **changes), SAMPLE_STD), kind)


@pytest.mark.parametrize("direction", ["long", "short", "flat"])
def test_zero_confidence_signal_is_flat(direction: str) -> None:
    """A signal at confidence 0 carries no view, so its direction is flat whatever the analyst
    passed; the status is kept."""
    signal = AgentSignal(
        agent_id="a",
        asset="SPY",
        as_of_date=date(2016, 1, 4),
        direction=direction,  # type: ignore[arg-type]
        confidence=0.0,
        rationale="r",
    )
    assert signal.direction == "flat"
    assert signal.status == "active"


@pytest.mark.parametrize("direction", ["long", "short"])
def test_positive_confidence_keeps_its_direction(direction: str) -> None:
    """A signal with any positive confidence keeps the direction it was given."""
    signal = AgentSignal(
        agent_id="a",
        asset="SPY",
        as_of_date=date(2016, 1, 4),
        direction=direction,  # type: ignore[arg-type]
        confidence=1e-9,
        rationale="r",
    )
    assert signal.direction == direction
