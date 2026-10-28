# test_sec_fundamentals_analyst.py

"""Tests for the discrete-quarter construction and trend signal of `SecFundamentalsAnalyst`."""

import pandas as pd
import pytest

from agentic_trading.agents.analysts.event_credit_pod import sec_fundamentals_analyst as mod

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
