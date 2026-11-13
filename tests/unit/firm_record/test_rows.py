# test_rows.py

"""AT-125 row fields: `firm_day_rows` turns one day's `PodTarget`s into decision rows. A row
is active when it has a nonzero weight or is carried, or any contributing signal is active, and
unavailable or error, at weight 0, only when every signal is. Its source and source_as_of are
those of the contributing input (signals, and an option row's chain snapshot) with the latest
effective known time, a date from a source that can publish after the close counting from the
next session; every input passes the stage (a) audit on its own, and a held target keeps the
provenance of the row that set it. An instrument carried over a day without its bar is written
at its carried targets; one without a bar and without a target gets one unavailable row at
weight 0 per pod whose scope holds it. A held put is written at its held premium, and a future
under its root. `firm_sources` is the union over the roster."""

from datetime import date
from typing import Any

import pandas as pd
import pytest

from agentic_trading.agents.base import AnalystAgent, DataRequirement
from agentic_trading.agents.schemas import AgentSignal, ManagerSignal, OptionContract, PodTarget
from agentic_trading.backtest.portfolio import OptionPosition, Portfolio
from agentic_trading.config import settings
from agentic_trading.firm_record.rows import (
    firm_day_rows,
    firm_sources,
    osi_symbol,
    row_instrument_id,
    row_provenance,
)
from agentic_trading.firm_record.runner import _held_contracts
from agentic_trading.stage_a.decisions import LookAheadError

_DAY = date(2016, 1, 6)
_KINDS: dict[str, Any] = {"SPY": "equity", "CL=F": "future", "BTC-USD": "crypto"}
_SOURCES = {"fred_analyst": "fred", "chain_analyst": "options_chain", "price_analyst": None}


def _signal(
    analyst: str, status: str = "active", source_as_of: date | None = None, pod: str = "a_pod"
) -> ManagerSignal:
    return ManagerSignal(
        manager_id=pod,
        source_agent_id=analyst,
        asset="SPY",
        as_of_date=_DAY,
        direction="long" if status == "active" else "flat",
        confidence=0.5 if status == "active" else 0.0,
        raw_confidence=0.5 if status == "active" else 0.0,
        scale_applied=1.0,
        rationale="stub",
        status=status,
        source_as_of=source_as_of,
    )


def _target(
    weight: float, *signals: ManagerSignal, pod: str = "a_pod", asset: str = "SPY", **fields: Any
) -> PodTarget:
    return PodTarget(
        pod_id=pod,
        asset=asset,
        as_of_date=_DAY,
        target_weight=weight,
        contributing_signals=list(signals),
        **fields,
    )


def _rows(*targets: PodTarget, absent: dict[str, list[str]] | None = None) -> list[Any]:
    return firm_day_rows(
        _DAY,
        list(targets),
        absent=absent or {},
        analyst_sources=_SOURCES,
        instrument_kinds=_KINDS,
        calendar="XNYS",
    )


@pytest.mark.parametrize(
    ("statuses", "weight", "expected"),
    [
        (["active", "unavailable"], 0.1, "active"),
        (["active", "error"], 0.0, "active"),
        (["unavailable", "unavailable"], 0.0, "unavailable"),
        (["unavailable", "error"], 0.0, "error"),
        (["error"], 0.0, "error"),
        (["unavailable"], 0.2, "active"),
    ],
)
def test_row_status(statuses: list[str], weight: float, expected: str) -> None:
    """Active with any active signal or a held weight; else error when a signal errored, else
    unavailable; a non-active row has weight 0."""
    signals = [_signal("price_analyst", status) for status in statuses]
    (row,) = _rows(_target(weight, *signals))

    assert row.status == expected
    assert row.target_weight == weight
    assert row.pod_id == "a_pod"


def test_target_without_signals_or_weight_writes_no_row() -> None:
    """A pod that neither ran an analyst on the instrument nor holds it writes no row; one that
    holds it without a signal writes an active row with no source."""
    assert _rows(_target(0.0)) == []
    (row,) = _rows(_target(0.05))
    assert (row.status, row.source, row.source_as_of) == ("active", None, None)


@pytest.mark.parametrize(
    ("fred_as_of", "chain_as_of", "source", "source_as_of"),
    [
        # Equal dates: fred, published after the close, counts from 2016-01-05 and wins.
        (date(2016, 1, 4), date(2016, 1, 4), "fred", date(2016, 1, 4)),
        # Equal effective day (2016-01-05): the after-close source wins.
        (date(2016, 1, 4), date(2016, 1, 5), "fred", date(2016, 1, 4)),
        # The chain snapshot of the decision day is the latest known input.
        (date(2016, 1, 4), date(2016, 1, 6), "options_chain", date(2016, 1, 6)),
        # Friday's fred value counts from Monday 2016-01-04, before Tuesday's chain.
        (date(2016, 1, 1), date(2016, 1, 5), "options_chain", date(2016, 1, 5)),
    ],
)
def test_source_is_the_latest_effective_input(
    fred_as_of: date, chain_as_of: date, source: str, source_as_of: date
) -> None:
    """The row names the contributing input with the latest effective known time."""
    (row,) = _rows(
        _target(
            0.1,
            _signal("fred_analyst", source_as_of=fred_as_of),
            _signal("chain_analyst", source_as_of=chain_as_of),
            _signal("price_analyst"),
        )
    )

    assert (row.source, row.source_as_of) == (source, source_as_of)


@pytest.mark.parametrize(
    "signal",
    [
        _signal("fred_analyst", source_as_of=_DAY),
        _signal("chain_analyst", source_as_of=date(2016, 1, 7)),
        _signal("fred_analyst"),
        _signal("price_analyst", source_as_of=date(2016, 1, 4)),
    ],
)
def test_every_signal_passes_the_stage_a_audit(signal: ManagerSignal) -> None:
    """One contributing signal the close cannot know, an active signal of a dated source with
    no source_as_of, or a date from an analyst with no dated source, refuses the day, even when
    another signal on the row would pass."""
    with pytest.raises(LookAheadError):
        _rows(_target(0.1, _signal("chain_analyst", source_as_of=_DAY), signal))


def test_inactive_signal_of_a_dated_source_may_carry_no_date() -> None:
    """An unavailable signal from a dated source with no source_as_of passes, as in stage (a)."""
    (row,) = _rows(_target(0.0, _signal("fred_analyst", status="unavailable")))

    assert (row.status, row.source, row.source_as_of) == ("unavailable", None, None)


def test_unknown_analyst_is_refused() -> None:
    """A signal from an analyst outside the roster's source map is refused."""
    with pytest.raises(ValueError, match="ghost_analyst"):
        _rows(_target(0.1, _signal("ghost_analyst")))


def test_absent_instruments_get_unavailable_rows_per_pod() -> None:
    """An instrument without a bar gets one unavailable, weight-0 row per pod in its scope."""
    rows = _rows(absent={"BTC-USD": ["risk_manager", "cross_asset_manager"]})

    assert [(row.instrument_id, row.pod_id, row.status, row.target_weight) for row in rows] == [
        ("BTC-USD", "risk_manager", "unavailable", 0.0),
        ("BTC-USD", "cross_asset_manager", "unavailable", 0.0),
    ]
    assert {row.instrument_kind for row in rows} == {"crypto"}


def test_absent_instrument_with_a_target_is_refused() -> None:
    """A pod target for an instrument listed as without a bar is a contradiction and refused."""
    with pytest.raises(ValueError, match="CL=F"):
        _rows(_target(0.1, _signal("price_analyst"), asset="CL=F"), absent={"CL=F": ["a_pod"]})


def test_option_and_kind_mapping() -> None:
    """An option row is of kind option, keyed by its OSI symbol, with its delta-equivalent
    weight; a ticker's kind comes from the mapping, and a ticker without one is refused."""
    put = OptionContract(
        underlying="SPY", expiry=date(2016, 3, 18), strike=180.0, option_type="put"
    )
    (option,) = _rows(
        _target(
            0.004,
            _signal("chain_analyst", source_as_of=_DAY),
            pod="tail_hedge_manager",
            asset=put.contract_id,
            instrument=put,
            delta_equiv_weight=-0.3,
        )
    )
    (future,) = _rows(_target(0.1, _signal("price_analyst"), asset="CL=F"))

    assert (option.instrument_kind, option.instrument_id) == ("option", "SPY160318P00180000")
    assert option.delta_equiv_weight == -0.3
    assert future.instrument_kind == "future"
    with pytest.raises(ValueError, match="GLD"):
        _rows(_target(0.1, _signal("price_analyst"), asset="GLD"))


class _Declaring(AnalystAgent):
    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        raise NotImplementedError


def _requirement(source: str) -> DataRequirement:
    return DataRequirement(
        source=source,
        identifier="x",
        asset_scoped=False,
        description="stub",  # type: ignore[arg-type]
    )


def test_firm_sources_are_the_union_over_the_roster() -> None:
    """Every dated source any analyst declares, by name, with can_publish_after_close, then each
    undated source with its limitation."""
    analysts = [
        _Declaring(agent_id="a", data_requirements=(_requirement("fred"), _requirement("sec_sic"))),
        _Declaring(
            agent_id="b", data_requirements=(_requirement("options_chain"), _requirement("fred"))
        ),
        _Declaring(agent_id="c"),
    ]
    sources = firm_sources(analysts)

    assert [entry["name"] for entry in sources] == ["fred", "options_chain", "sec_sic"]
    assert sources[0] == {"name": "fred", "can_publish_after_close": True}
    assert sources[1] == {"name": "options_chain", "can_publish_after_close": False}
    assert sources[2]["undated"] is True and sources[2]["limitation"]


_PUT = OptionContract(underlying="SPY", expiry=date(2016, 3, 18), strike=180.0, option_type="put")


def test_held_target_keeps_the_provenance_of_the_row_that_set_it() -> None:
    """A held target names the source and source_as_of of the pod's row the day before, not its
    own signals'; a target the pod sets names its own; both pass the audit."""
    previous = {("SPY", "a_pod"): ("fred", date(2015, 12, 30))}
    fresh = _signal("fred_analyst", source_as_of=date(2016, 1, 5))
    (held,) = firm_day_rows(
        _DAY,
        [_target(0.2, fresh, held=True)],
        absent={},
        analyst_sources=_SOURCES,
        instrument_kinds=_KINDS,
        calendar="XNYS",
        previous=previous,
    )
    (set_today,) = firm_day_rows(
        _DAY,
        [_target(0.2, fresh)],
        absent={},
        analyst_sources=_SOURCES,
        instrument_kinds=_KINDS,
        calendar="XNYS",
        previous=previous,
    )

    assert (held.source, held.source_as_of) == ("fred", date(2015, 12, 30))
    assert (set_today.source, set_today.source_as_of) == ("fred", date(2016, 1, 5))


def test_carried_target_is_active_and_suppresses_the_unavailable_rows() -> None:
    """An instrument without a bar that carries its targets is written at them, active even at
    weight 0, with no unavailable rows; one without a target gets them."""
    rows = _rows(
        _target(0.0, _signal("price_analyst", "unavailable"), held=True, carried=True),
        _target(0.3, _signal("price_analyst", "unavailable"), pod="b_pod", held=True, carried=True),
        absent={"SPY": ["a_pod", "b_pod"], "BTC-USD": ["a_pod"]},
    )

    assert sorted((r.instrument_id, r.pod_id, r.status, r.target_weight) for r in rows) == [
        ("BTC-USD", "a_pod", "unavailable", 0.0),
        ("SPY", "a_pod", "active", 0.0),
        ("SPY", "b_pod", "active", 0.3),
    ]


def test_option_row_counts_its_chain_snapshot() -> None:
    """An option row's chain snapshot is one of its inputs: when its date is the latest it
    names options_chain; an older one leaves the signal's source."""
    signal = _signal("price_analyst", source_as_of=None)
    newest = _signal("fred_analyst", source_as_of=date(2016, 1, 5))
    (chain,) = _rows(
        _target(0.01, signal, asset=_PUT.contract_id, instrument=_PUT, mark_as_of=_DAY)
    )
    (fred,) = _rows(
        _target(0.01, newest, asset=_PUT.contract_id, instrument=_PUT, mark_as_of=date(2016, 1, 4))
    )

    assert (chain.source, chain.source_as_of) == ("options_chain", _DAY)
    assert (fred.source, fred.source_as_of) == ("fred", date(2016, 1, 5))


def test_held_put_is_written_at_its_held_premium() -> None:
    """After the put's mark rose fivefold, the held put's row is the held contracts' premium
    weight at the decision day's close mark, not the nominal weight, and its delta-equivalent
    weight scales with it."""
    nominal, contracts, nav = 0.004, 4.0, 1_000_000.0
    entry_mid = nominal * nav / (contracts * settings.options.contract_multiplier)
    (row,) = firm_day_rows(
        _DAY,
        [
            _target(
                nominal,
                _signal("price_analyst"),
                pod="tail_hedge_manager",
                asset=_PUT.contract_id,
                instrument=_PUT,
                delta_equiv_weight=-0.1,
                mark_mid=5 * entry_mid,
                hold_contracts=True,
                held=True,
            )
        ],
        absent={},
        analyst_sources=_SOURCES,
        instrument_kinds=_KINDS,
        calendar="XNYS",
        held_contracts={_PUT.contract_id: contracts},
        nav=nav,
    )

    assert row.target_weight == pytest.approx(5 * nominal)
    assert row.delta_equiv_weight == pytest.approx(-0.5)


def test_held_contracts_read_the_book_not_its_fill_mid() -> None:
    """The row prices the held contracts at the decision day's close mark: the position's own
    last mid (the lag-1 fill's open mid) does not enter."""
    portfolio = Portfolio(cash=100_000.0, positions={})
    portfolio.option_positions[_PUT.contract_id] = OptionPosition(
        contract=_PUT, contracts=2.0, entry_price=1.0, last_mid=9.0
    )
    held = _held_contracts(portfolio)
    (row,) = firm_day_rows(
        _DAY,
        [
            _target(
                0.004,
                _signal("price_analyst"),
                pod="tail_hedge_manager",
                asset=_PUT.contract_id,
                instrument=_PUT,
                mark_mid=5.0,
                hold_contracts=True,
            )
        ],
        absent={},
        analyst_sources=_SOURCES,
        instrument_kinds=_KINDS,
        calendar="XNYS",
        held_contracts=held,
        nav=100_000.0,
    )

    assert held == {_PUT.contract_id: 2.0}
    assert row.target_weight == pytest.approx(
        2.0 * 5.0 * settings.options.contract_multiplier / 100_000.0
    )


@pytest.mark.parametrize(
    ("ticker", "row_id"),
    [
        ("GC=F", "GC"),
        ("CL=F", "CL"),
        ("SI=F", "SI"),
        ("NG=F", "NG"),
        ("HG=F", "HG"),
        ("SPY", "SPY"),
        ("TLT", "TLT"),
        ("EURUSD=X", "EURUSD=X"),
        ("BTC-USD", "BTC-USD"),
    ],
)
def test_row_instrument_id(ticker: str, row_id: str) -> None:
    """A future's row is keyed by its root; every other instrument by its own id."""
    assert row_instrument_id(ticker) == row_id


def test_an_option_row_is_keyed_by_its_osi_symbol() -> None:
    """With its contract, an option's row id is the contract's OSI symbol."""
    assert row_instrument_id(_PUT.contract_id, _PUT) == "SPY160318P00180000"


@pytest.mark.parametrize(
    ("contract", "symbol"),
    [
        (
            OptionContract(
                underlying="SPY", expiry=date(2015, 3, 20), strike=190.0, option_type="put"
            ),
            "SPY150320P00190000",
        ),
        (
            OptionContract(
                underlying="SPY", expiry=date(2016, 12, 16), strike=72.5, option_type="call"
            ),
            "SPY161216C00072500",
        ),
        (
            OptionContract(
                underlying="SPY", expiry=date(2017, 1, 20), strike=72.375, option_type="put"
            ),
            "SPY170120P00072375",
        ),
        (
            OptionContract(
                underlying="AAPL", expiry=date(2026, 12, 18), strike=5.0, option_type="call"
            ),
            "AAPL261218C00005000",
        ),
        (
            OptionContract(
                underlying="SPY", expiry=date(2016, 3, 18), strike=99999.999, option_type="put"
            ),
            "SPY160318P99999999",
        ),
    ],
)
def test_osi_symbol(contract: OptionContract, symbol: str) -> None:
    """The unpadded OSI symbol: root, yymmdd expiry, C or P, strike x1000 in 8 digits."""
    assert osi_symbol(contract) == symbol


@pytest.mark.parametrize(
    ("underlying", "expiry", "strike"),
    [
        ("SPY", date(2015, 3, 20), 190.0005),
        ("SPY", date(2015, 3, 20), 100000.0),
        ("spy", date(2015, 3, 20), 190.0),
        ("ABCDEFG", date(2015, 3, 20), 190.0),
        ("BRK.B", date(2015, 3, 20), 190.0),
        ("SPY", date(1999, 3, 19), 190.0),
    ],
)
def test_osi_symbol_refuses_what_osi_cannot_carry(
    underlying: str, expiry: date, strike: float
) -> None:
    """A strike with more than 3 dp or over 8 digits x1000, a root that is not 1-6 uppercase
    letters, or an expiry outside 2000-2099 is refused, never rounded."""
    contract = OptionContract(
        underlying=underlying, expiry=expiry, strike=strike, option_type="put"
    )
    with pytest.raises(ValueError, match="OSI"):
        osi_symbol(contract)


def test_a_held_option_keeps_its_provenance_under_its_osi_symbol() -> None:
    """A held option's row keeps the previous day's provenance, looked up by its OSI symbol."""
    kept = ("fred", date(2016, 1, 4))
    (row,) = firm_day_rows(
        _DAY,
        [
            _target(
                0.004,
                _signal("price_analyst"),
                pod="tail_hedge_manager",
                asset=_PUT.contract_id,
                instrument=_PUT,
                held=True,
            )
        ],
        absent={},
        analyst_sources=_SOURCES,
        instrument_kinds=_KINDS,
        calendar="XNYS",
        previous={("SPY160318P00180000", "tail_hedge_manager"): kept},
    )

    assert row.instrument_id == "SPY160318P00180000"
    assert (row.source, row.source_as_of) == kept


def test_every_futures_row_is_keyed_by_its_root() -> None:
    """Every row of a future, a target's or an unavailable one, carries the root as its id and
    the future kind."""
    kinds = {**_KINDS, "GC=F": "future"}
    rows = firm_day_rows(
        _DAY,
        [_target(0.1, _signal("price_analyst"), asset="CL=F")],
        absent={"GC=F": ["a_pod"]},
        analyst_sources=_SOURCES,
        instrument_kinds=kinds,
        calendar="XNYS",
    )

    assert sorted((row.instrument_id, row.instrument_kind) for row in rows) == [
        ("CL", "future"),
        ("GC", "future"),
    ]


def test_carried_row_provenance_depends_on_whether_a_limit_scaled_it() -> None:
    """A carried row at its carried weight keeps the setting decision's provenance; a carried
    row a firm limit scaled takes the latest of the day's inputs, other instruments' included."""
    previous = {("SPY", "a_pod"): ("fred", date(2015, 12, 28))}
    old_signal = _signal("fred_analyst", source_as_of=date(2015, 12, 28))
    today = _signal("fred_analyst", source_as_of=date(2016, 1, 5), pod="b_pod")
    other = _target(0.1, today, pod="b_pod", asset="CL=F")

    def spy_row(held: bool) -> Any:
        rows = firm_day_rows(
            _DAY,
            [_target(0.2, old_signal, held=held, carried=True), other],
            absent={"SPY": ["a_pod"]},
            analyst_sources=_SOURCES,
            instrument_kinds=_KINDS,
            calendar="XNYS",
            previous=previous,
        )
        return next(row for row in rows if row.instrument_id == "SPY")

    kept, scaled = spy_row(held=True), spy_row(held=False)

    assert (kept.source, kept.source_as_of, kept.status) == ("fred", date(2015, 12, 28), "active")
    assert (scaled.source, scaled.source_as_of, scaled.status) == (
        "fred",
        date(2016, 1, 5),
        "active",
    )


def test_a_held_option_carries_its_provenance_from_one_day_to_the_next() -> None:
    """The provenance of day 1's option row, keyed by its OSI symbol, is what day 2's held
    target keeps; keyed by the contract id instead, it would not be found."""
    signal = _signal("fred_analyst", source_as_of=date(2016, 1, 5))
    day1 = firm_day_rows(
        _DAY,
        [_target(0.004, signal, pod="tail_hedge_manager", asset=_PUT.contract_id, instrument=_PUT)],
        absent={},
        analyst_sources=_SOURCES,
        instrument_kinds=_KINDS,
        calendar="XNYS",
    )
    previous = row_provenance(day1)
    assert set(previous) == {("SPY160318P00180000", "tail_hedge_manager")}

    day2 = date(2016, 1, 7)
    (row,) = firm_day_rows(
        day2,
        [
            _target(
                0.004,
                _signal("price_analyst"),
                pod="tail_hedge_manager",
                asset=_PUT.contract_id,
                instrument=_PUT,
                held=True,
            ).model_copy(update={"as_of_date": day2})
        ],
        absent={},
        analyst_sources=_SOURCES,
        instrument_kinds=_KINDS,
        calendar="XNYS",
        previous=previous,
    )
    assert (row.source, row.source_as_of) == previous[("SPY160318P00180000", "tail_hedge_manager")]
