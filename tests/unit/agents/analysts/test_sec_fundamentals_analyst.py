# test_sec_fundamentals_analyst.py

"""Tests for the discrete-quarter construction and trend signal of `SecFundamentalsAnalyst`."""

from datetime import date

import pandas as pd
import pytest

from agentic_trading.agents.analysts.event_credit_pod import sec_fundamentals_analyst as mod
from agentic_trading.data.loaders import lakehouse_loader as ll
from agentic_trading.stage_a.decisions import check_source_as_of, row_source

_CONCEPT = "EarningsPerShareDiluted"
_AS_OF = pd.Timestamp("2024-06-30")

# (period_start, period_end, value); filed 35 days after period_end.
_FACTS = [
    ("2021-09-26", "2021-12-25", 2.10),
    ("2021-12-26", "2022-03-26", 1.52),
    ("2022-03-27", "2022-06-25", 1.20),
    ("2021-09-26", "2022-09-24", 6.11),
    ("2022-09-25", "2022-12-31", 1.88),
    ("2023-01-01", "2023-04-01", 1.52),
    ("2023-04-02", "2023-07-01", 1.26),
    ("2022-09-25", "2023-09-30", 6.13),
    ("2023-10-01", "2023-12-30", 2.18),
    ("2023-12-31", "2024-03-30", 1.53),
    ("2022-09-25", "2023-04-01", 3.40),
    ("2022-09-25", "2023-06-24", 4.66),
]
_EXPECTED = [1.20, 1.29, 1.88, 1.52, 1.26, 1.47, 2.18, 1.53]


def _facts(rows: list[tuple[str, str, float]] = _FACTS) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["period_start", "period_end", "value"])
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    frame["concept"] = _CONCEPT
    frame["unit"] = "USD/shares"
    frame["filed"] = frame["period_end"] + pd.Timedelta(days=35)
    frame["available_date"] = frame["filed"]
    return frame


def test_discrete_quarters_derive_q4_and_exclude_ytd_facts() -> None:
    """Q4 is the annual minus the three inside quarters; 188- and 272-day YTD facts are skipped."""
    facts = _facts()
    other = facts.iloc[:2].assign(concept="Revenues", value=999.0)

    quarters = mod.discrete_quarters(pd.concat([facts, other]), _CONCEPT)

    assert list(quarters.columns) == ["period_end", "value", "filed"]
    assert quarters["period_end"].is_monotonic_increasing
    assert list(quarters["value"].tail(8)) == pytest.approx(_EXPECTED, abs=1e-9)
    assert not quarters["value"].isin([3.40, 4.66, 999.0, 6.11, 6.13]).any()
    derived = quarters.loc[quarters["period_end"] == pd.Timestamp("2023-09-30")].iloc[0]
    assert derived["filed"] == pd.Timestamp("2023-09-30") + pd.Timedelta(days=35)


def test_discrete_quarters_span_boundaries_and_missing_inside_quarter() -> None:
    """80- and 100-day spans are discrete, 79 and 101 are not; Q4 needs all three quarters."""
    rows = []
    for i, (start, days) in enumerate([("2024-01-01", 79), ("2024-01-01", 80)]):
        end = pd.Timestamp(start) + pd.Timedelta(days=days)
        rows.append((start, str(end.date()), float(i)))
    for i, (start, days) in enumerate([("2024-06-01", 100), ("2024-06-01", 101)], start=2):
        end = pd.Timestamp(start) + pd.Timedelta(days=days)
        rows.append((start, str(end.date()), float(i)))

    quarters = mod.discrete_quarters(_facts(rows), _CONCEPT)
    assert list(quarters["value"]) == [1.0, 2.0]

    missing = [r for r in _FACTS if (r[0], r[1]) != ("2023-01-01", "2023-04-01")]
    ends = set(mod.discrete_quarters(_facts(missing), _CONCEPT)["period_end"])
    assert pd.Timestamp("2023-09-30") not in ends
    assert pd.Timestamp("2022-09-24") in ends


def test_sec_fundamentals_signal_reports_provenance_from_discrete_quarters(monkeypatch) -> None:
    """Eight discrete quarters give an active signal whose source_as_of is the latest filed."""
    monkeypatch.setattr(mod, "load_company_facts_from_lakehouse", lambda _a, _d: _facts())

    signal = mod.SecFundamentalsAnalyst().generate_signal("AAPL", pd.DataFrame(), _AS_OF)

    assert signal.status == "active"
    assert signal.sample_count == 8
    assert signal.source_as_of == (pd.Timestamp("2024-03-30") + pd.Timedelta(days=35)).date()


def test_sec_fundamentals_unavailable_with_fewer_than_four_discrete_quarters(monkeypatch) -> None:
    """YTD facts do not count towards the four-quarter minimum."""
    rows = [_FACTS[5], _FACTS[6], _FACTS[8], _FACTS[10], _FACTS[11]]
    monkeypatch.setattr(mod, "load_company_facts_from_lakehouse", lambda _a, _d: _facts(rows))

    signal = mod.SecFundamentalsAnalyst().generate_signal("AAPL", pd.DataFrame(), _AS_OF)

    assert signal.status == "unavailable"
    assert signal.direction == "flat"


def test_discrete_quarters_keeps_latest_filed_known_fact_per_quarter() -> None:
    """Of two facts for one quarter, the later filing (X, filed 2024-03-15, available
    2024-03-18) wins over the earlier one accepted later (Y, filed 2024-03-12, available
    2024-03-21), in either row order."""
    rows = [("2023-10-01", "2023-12-30", 100.0), ("2023-10-01", "2023-12-30", 200.0)]
    facts = _facts(rows)
    facts["filed"] = pd.to_datetime(["2024-03-12", "2024-03-15"])
    facts["available_date"] = pd.to_datetime(["2024-03-21", "2024-03-18"])

    for frame in (facts, facts.iloc[::-1]):
        quarters = mod.discrete_quarters(frame, _CONCEPT)
        assert list(quarters["value"]) == [200.0]
        assert list(quarters["filed"]) == [pd.Timestamp("2024-03-15")]


def _fetched_quarters(rows: list[tuple[str, str, float, str, str]]) -> pd.DataFrame:
    """`_fetch_company_facts` rows of discrete quarters of `_CONCEPT`, one per (period_start,
    period_end, value, filed date, America/New_York acceptance time)."""
    return pd.DataFrame(
        {
            "concept": _CONCEPT,
            "unit": "USD/shares",
            "period_start": pd.to_datetime([r[0] for r in rows]),
            "period_end": pd.to_datetime([r[1] for r in rows]),
            "value": [r[2] for r in rows],
            "fiscal_year": 2023,
            "fiscal_period": "Q",
            "form": "10-Q",
            "filed": pd.to_datetime([r[3] for r in rows]),
            "accepted_at": [
                pd.Timestamp(r[4], tz="America/New_York").tz_convert("UTC") for r in rows
            ],
        }
    )


def test_signal_from_loader_selection_passes_source_as_of_audit(monkeypatch) -> None:
    """A 10-Q filed and accepted before the close on the decision day is not yet known, so the
    signal's `source_as_of` passes the after-close audit of the analyst's declared source on
    that day, and on the next session, when the filing counts."""
    fetched = _fetched_quarters(
        [
            ("2023-01-01", "2023-04-01", 1.00, "2023-05-05", "2023-05-05 08:00"),
            ("2023-04-02", "2023-07-01", 2.10, "2023-08-04", "2023-08-04 18:00"),
            ("2023-07-02", "2023-09-30", 2.90, "2023-11-03", "2023-11-03 09:00"),
            ("2023-10-01", "2023-12-30", 4.05, "2024-02-02", "2024-02-02 07:00"),
            ("2023-12-31", "2024-03-30", 5.00, "2024-05-03", "2024-05-03 10:00"),
        ]
    )
    monkeypatch.setattr(
        mod,
        "load_company_facts_from_lakehouse",
        lambda _ticker, as_of: ll._select_company_facts(fetched, as_of),
    )
    analyst = mod.SecFundamentalsAnalyst()
    source = row_source(analyst)
    assert source == "sec_edgar_facts"

    decision_day = pd.Timestamp("2024-05-03")
    signal = analyst.generate_signal("AAPL", pd.DataFrame(), decision_day)
    assert signal.status == "active"
    assert signal.source_as_of == date(2024, 2, 2)
    assert signal.sample_count == 4
    check_source_as_of(signal.source_as_of, decision_day.date(), source)

    next_session = pd.Timestamp("2024-05-06")
    later = analyst.generate_signal("AAPL", pd.DataFrame(), next_session)
    assert later.source_as_of == date(2024, 5, 3)
    assert later.sample_count == 5
    check_source_as_of(later.source_as_of, next_session.date(), source)
