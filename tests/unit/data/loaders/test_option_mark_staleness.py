# test_option_mark_staleness.py

"""Unit tests for the staleness handling in agentic_trading.data.loaders.option_marks.

Part A reads the DGS3MO risk-free rate through a `fake_lake` `fred_vintages`
table. Part B replaces `option_marks._query_chain`, `_query_rate` and
`load_option_quote_from_lakehouse` at the module boundary: the latest snapshot
(dated S) does not list the held contract's expiry, the contract's own last
quote (dated Q < S) carries a different implied volatility, and the mark must
use the volatility read off snapshot S's surface instead.
"""

import math
from collections.abc import Callable, Iterator
from datetime import date
from typing import Any

import pandas as pd
import pytest
from quantcore.pricing.black_scholes import black_scholes_call, black_scholes_put

from agentic_trading.agents.schemas import OptionContract
from agentic_trading.config import temporary_overrides
from agentic_trading.data.loaders import option_marks
from agentic_trading.data.loaders.option_marks import _query_rate, load_option_mark

UNDERLYING = "AAPL"
RATE = 0.05
SPOT = 180.0
STRIKE = 180.0
QUOTE_DATE = date(2024, 3, 1)
SNAPSHOT_DATE = date(2024, 3, 5)
AS_OF = date(2024, 3, 7)
HELD_EXPIRY = date(2024, 4, 5)
NEAR_EXPIRY = date(2024, 3, 15)
FAR_EXPIRY = date(2024, 4, 19)
QUOTE_IV = 0.80
VINTAGE_DATES = ("date", "realtime_start")


def _vintages(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["series_id", "date", "realtime_start", "value"])


def test_query_rate_returns_latest_published_observation_as_a_fraction(fake_lake: Any) -> None:
    """The rate is the latest DGS3MO value published by as_of, divided by 100."""
    fake_lake.add(
        "fred_vintages",
        _vintages(
            [
                ("DGS3MO", "2024-03-05", "2024-03-05", 5.20),
                ("DGS3MO", "2024-03-06", "2024-03-06", 5.25),
                ("DGS3MO", "2024-03-08", "2024-03-08", 4.00),
            ]
        ),
        date_columns=VINTAGE_DATES,
    )

    assert _query_rate(AS_OF) == pytest.approx(0.0525, rel=1e-9)


@pytest.mark.parametrize(
    "rows",
    [
        [("DGS3MO", "2024-03-08", "2024-03-08", 5.25)],
        [("DGS10", "2024-03-05", "2024-03-05", 4.10)],
    ],
    ids=["only-later-vintage", "only-other-series"],
)
def test_query_rate_raises_when_no_dgs3mo_is_published(
    fake_lake: Any, rows: list[tuple[str, str, str, float]]
) -> None:
    """With no DGS3MO row published by as_of the rate lookup raises instead of returning 0.0."""
    fake_lake.add("fred_vintages", _vintages(rows), date_columns=VINTAGE_DATES)

    day = AS_OF.isoformat()
    with pytest.raises(ValueError, match=f"DGS3MO.*{day}|{day}.*DGS3MO"):
        _query_rate(AS_OF)


def _rows(
    expiry: date,
    option_type: str,
    ivs: dict[float, float],
    snapshot_date: date = SNAPSHOT_DATE,
    bid: float = 3.9,
    ask: float = 4.1,
) -> list[dict[str, Any]]:
    return [
        {
            "date": pd.Timestamp(snapshot_date),
            "expiration": pd.Timestamp(expiry),
            "option_type": option_type,
            "strike": strike,
            "bid": bid,
            "ask": ask,
            "implied_volatility": iv,
            "delta": 0.5,
            "gamma": 0.02,
            "theta": -0.05,
            "vega": 0.1,
            "rho": 0.01,
        }
        for strike, iv in ivs.items()
    ]


Setup = Callable[..., None]


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch, fake_lake: Any) -> Iterator[Setup]:
    """Installs a snapshot (or none) and the contract's own last quote at the module boundary."""
    option_marks.clear_cache()
    monkeypatch.setattr(option_marks, "_query_rate", lambda as_of_date: RATE)

    def install(rows: list[dict[str, Any]] | None, option_type: str = "call") -> None:
        frame = pd.DataFrame(rows) if rows else None
        monkeypatch.setattr(option_marks, "_query_chain", lambda underlying, as_of_date: frame)
        quote = pd.DataFrame(_rows(HELD_EXPIRY, option_type, {STRIKE: QUOTE_IV}, QUOTE_DATE))
        monkeypatch.setattr(
            option_marks, "load_option_quote_from_lakehouse", lambda *_a, **_k: quote
        )

    yield install
    option_marks.clear_cache()


def _contract(option_type: str = "call", strike: float = STRIKE) -> OptionContract:
    return OptionContract(
        underlying=UNDERLYING, expiry=HELD_EXPIRY, strike=strike, option_type=option_type
    )


def _days(start: date, end: date) -> float:
    return (end - start).days / 365


def _bs(option_type: str, sigma: float, strike: float = STRIKE, as_of: date = AS_OF) -> float:
    pricer = black_scholes_call if option_type == "call" else black_scholes_put
    return pricer(SPOT, strike, RATE, sigma, _days(as_of, HELD_EXPIRY))


def test_contract_listed_in_the_snapshot_is_marked_from_its_own_row(setup: Setup) -> None:
    """A contract quoted in a same-day snapshot keeps its quoted bid/ask and is not a model mark."""
    setup(_rows(HELD_EXPIRY, "call", {STRIKE: 0.25}))

    mark = load_option_mark(_contract(), SNAPSHOT_DATE, SPOT)

    assert mark is not None
    assert (mark.bid, mark.ask) == (pytest.approx(3.9), pytest.approx(4.1))
    assert mark.snapshot_date == SNAPSHOT_DATE
    assert not mark.is_model_mark


def test_stale_contract_is_remarked_at_the_interpolated_total_variance_vol(setup: Setup) -> None:
    """At a listed strike, sigma interpolates total variance linearly in tenor between expiries."""
    setup(_rows(NEAR_EXPIRY, "call", {STRIKE: 0.20}) + _rows(FAR_EXPIRY, "call", {STRIKE: 0.30}))
    tau_near, tau_far = _days(SNAPSHOT_DATE, NEAR_EXPIRY), _days(SNAPSHOT_DATE, FAR_EXPIRY)
    tau = _days(SNAPSHOT_DATE, HELD_EXPIRY)
    w_near, w_far = 0.20**2 * tau_near, 0.30**2 * tau_far
    w = w_near + (tau - tau_near) / (tau_far - tau_near) * (w_far - w_near)
    sigma = math.sqrt(w / tau)

    mark = load_option_mark(_contract(), AS_OF, SPOT)

    assert mark is not None
    assert 0.20 < sigma < 0.30
    assert mark.implied_volatility == pytest.approx(sigma, rel=1e-9)
    assert mark.mid == pytest.approx(_bs("call", sigma), rel=1e-9)
    assert mark.is_model_mark
    assert mark.snapshot_date == QUOTE_DATE


@pytest.mark.parametrize(
    ("strike", "expected_iv"),
    [(180.0, 0.25), (175.0, 0.225), (160.0, 0.20), (200.0, 0.30)],
    ids=["midpoint", "quarter", "flat-below", "flat-above"],
)
def test_stale_contract_interpolates_in_strike_and_is_flat_outside_the_quoted_range(
    setup: Setup, strike: float, expected_iv: float
) -> None:
    """Within one listed expiry IV is linear in strike and flat beyond the quoted strikes."""
    setup(_rows(NEAR_EXPIRY, "call", {170.0: 0.20, 190.0: 0.30}))

    mark = load_option_mark(_contract(strike=strike), AS_OF, SPOT)

    assert mark is not None
    assert mark.implied_volatility == pytest.approx(expected_iv, rel=1e-9)
    assert mark.mid == pytest.approx(_bs("call", expected_iv, strike), rel=1e-9)


@pytest.mark.parametrize(
    ("listed", "expected_iv"),
    [
        ((date(2024, 3, 8), date(2024, 3, 15)), 0.30),
        ((NEAR_EXPIRY, FAR_EXPIRY), 0.20),
    ],
    ids=["beyond-last-expiry", "before-first-expiry"],
)
def test_stale_contract_outside_the_listed_tenors_uses_the_nearest_expiry_vol(
    setup: Setup, listed: tuple[date, date], expected_iv: float
) -> None:
    """Outside the listed tenor range the nearest expiry's IV is used flat."""
    expiry = HELD_EXPIRY if expected_iv == 0.30 else date(2024, 3, 10)
    setup(_rows(listed[0], "call", {STRIKE: 0.20}) + _rows(listed[1], "call", {STRIKE: 0.30}))
    contract = OptionContract(
        underlying=UNDERLYING, expiry=expiry, strike=STRIKE, option_type="call"
    )

    mark = load_option_mark(contract, AS_OF, SPOT)

    assert mark is not None
    assert mark.implied_volatility == pytest.approx(expected_iv, rel=1e-9)


def test_stale_put_reads_the_put_surface_only(setup: Setup) -> None:
    """A put is re-marked from put rows even when calls quote a very different IV."""
    setup(
        _rows(NEAR_EXPIRY, "call", {STRIKE: 0.50}) + _rows(NEAR_EXPIRY, "put", {STRIKE: 0.22}),
        option_type="put",
    )

    mark = load_option_mark(_contract("put"), AS_OF, SPOT)

    assert mark is not None
    assert mark.implied_volatility == pytest.approx(0.22, rel=1e-9)
    assert mark.mid == pytest.approx(_bs("put", 0.22), rel=1e-9)


def test_stale_mark_is_tradeable_only_within_the_snapshot_age_window_of_the_quote(
    setup: Setup,
) -> None:
    """Tradeability depends on the age of the contract's own quote, not of the surface snapshot."""
    setup(_rows(NEAR_EXPIRY, "call", {STRIKE: 0.25}))

    with temporary_overrides({"options.max_snapshot_age_days": 10}):
        at_limit = load_option_mark(_contract(), date(2024, 3, 11), SPOT)
        beyond = load_option_mark(_contract(), date(2024, 3, 12), SPOT)

    assert at_limit is not None and beyond is not None
    assert (date(2024, 3, 11) - QUOTE_DATE).days == 10
    assert at_limit.tradeable
    assert not beyond.tradeable
    assert beyond.is_model_mark
    assert beyond.implied_volatility == pytest.approx(0.25, rel=1e-9)


def test_stale_contract_falls_back_to_its_own_iv_without_a_usable_surface(setup: Setup) -> None:
    """Without same-type positive-IV rows, or without a snapshot, the quote's own IV is used."""
    unusable = _rows(NEAR_EXPIRY, "call", {STRIKE: 0.25}) + _rows(NEAR_EXPIRY, "put", {STRIKE: 0.0})
    for rows in (unusable, None):
        setup(rows, option_type="put")
        option_marks.clear_cache()

        mark = load_option_mark(_contract("put"), AS_OF, SPOT)

        assert mark is not None
        assert mark.implied_volatility == pytest.approx(QUOTE_IV, rel=1e-9)
        assert mark.mid == pytest.approx(_bs("put", QUOTE_IV), rel=1e-9)
        assert mark.is_model_mark
        assert mark.snapshot_date == QUOTE_DATE
