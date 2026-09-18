"""Unit tests for agentic_trading.data.loaders.lakehouse_loader's
point-in-time self-assertion.

Each loader gates its own SQL query on `<= as_of_date`; these tests
monkeypatch `duckdb.connect` to return a canned DataFrame directly (the
query text itself is never executed), so they can independently confirm
`assert_point_in_time` raises if that DataFrame ever contained a
future-dated row that the SQL gate should have excluded.
"""

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from agentic_trading.config import temporary_overrides
from agentic_trading.data.loaders import lakehouse_loader


class _FakeResult:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def df(self) -> pd.DataFrame:
        return self._frame


class _FakeConnection:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def execute(self, _query: str) -> _FakeResult:
        return _FakeResult(self._frame)

    def close(self) -> None:
        pass


def _touch_table(lake_dir: Path, table: str) -> None:
    partition_dir = lake_dir / f"table={table}" / "partition=x"
    partition_dir.mkdir(parents=True, exist_ok=True)
    (partition_dir / "data.parquet").touch()


def _patch_connect(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> None:
    def _fake_connect(*_args: Any, **_kwargs: Any) -> _FakeConnection:
        return _FakeConnection(frame)

    monkeypatch.setattr(lakehouse_loader.duckdb, "connect", _fake_connect)


def test_load_short_interest_returns_valid_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid, in-window dissemination dates pass straight through."""
    with temporary_overrides({"data.lake_dir": tmp_path}):
        _touch_table(tmp_path, "finra_short_interest")
        frame = pd.DataFrame(
            {
                "settlement_date": ["2024-01-01"],
                "dissemination_date": ["2024-01-10"],
                "short_interest_shares": [1000],
                "days_to_cover": [1.5],
                "avg_daily_volume": [500],
            }
        )
        _patch_connect(monkeypatch, frame)

        result = lakehouse_loader.load_short_interest_from_lakehouse("AAPL", date(2024, 1, 15))

    assert result is not None
    assert result.loc[0, "short_interest_shares"] == 1000


def test_load_short_interest_raises_on_future_dissemination_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dissemination_date after as_of_date is a broken SQL gate — must raise, not pass through."""
    with temporary_overrides({"data.lake_dir": tmp_path}):
        _touch_table(tmp_path, "finra_short_interest")
        frame = pd.DataFrame(
            {
                "settlement_date": ["2024-01-01"],
                "dissemination_date": ["2024-01-20"],
                "short_interest_shares": [1000],
                "days_to_cover": [1.5],
                "avg_daily_volume": [500],
            }
        )
        _patch_connect(monkeypatch, frame)

        with pytest.raises(ValueError, match="2024-01-20"):
            lakehouse_loader.load_short_interest_from_lakehouse("AAPL", date(2024, 1, 15))


def test_load_company_facts_returns_valid_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid, in-window filed dates pass straight through."""
    with temporary_overrides({"data.lake_dir": tmp_path}):
        _touch_table(tmp_path, "sec_company_facts")
        frame = pd.DataFrame(
            {
                "concept": ["Revenues"],
                "value": [1_000_000.0],
                "fiscal_period": ["Q1"],
                "filed": ["2024-01-10"],
            }
        )
        _patch_connect(monkeypatch, frame)

        result = lakehouse_loader.load_company_facts_from_lakehouse("AAPL", date(2024, 1, 15))

    assert result is not None
    assert result.loc[0, "concept"] == "Revenues"


def test_load_company_facts_raises_on_future_filed_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filed date after as_of_date is a broken SQL gate — must raise, not pass through."""
    with temporary_overrides({"data.lake_dir": tmp_path}):
        _touch_table(tmp_path, "sec_company_facts")
        frame = pd.DataFrame(
            {
                "concept": ["Revenues"],
                "value": [1_000_000.0],
                "fiscal_period": ["Q1"],
                "filed": ["2024-01-20"],
            }
        )
        _patch_connect(monkeypatch, frame)

        with pytest.raises(ValueError, match="2024-01-20"):
            lakehouse_loader.load_company_facts_from_lakehouse("AAPL", date(2024, 1, 15))


def test_load_dividends_returns_valid_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid, in-window ex-dates pass straight through."""
    with temporary_overrides({"data.lake_dir": tmp_path}):
        _touch_table(tmp_path, "dolthub_stock_dividends")
        frame = pd.DataFrame({"ex_date": ["2024-01-10"], "amount": [0.25]})
        _patch_connect(monkeypatch, frame)

        result = lakehouse_loader.load_dividends_from_lakehouse("AAPL", date(2024, 1, 15))

    assert result is not None
    assert result.iloc[0] == 0.25


def test_load_dividends_raises_on_future_ex_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ex-date after as_of_date in the index is a broken SQL gate — must raise."""
    with temporary_overrides({"data.lake_dir": tmp_path}):
        _touch_table(tmp_path, "dolthub_stock_dividends")
        frame = pd.DataFrame({"ex_date": ["2024-01-20"], "amount": [0.25]})
        _patch_connect(monkeypatch, frame)

        with pytest.raises(ValueError, match="2024-01-20"):
            lakehouse_loader.load_dividends_from_lakehouse("AAPL", date(2024, 1, 15))


def test_load_fred_series_returns_valid_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid, in-window dates in the index pass straight through."""
    with temporary_overrides({"data.lake_dir": tmp_path}):
        _touch_table(tmp_path, "fred_series_observations")
        frame = pd.DataFrame({"date": ["2024-01-10"], "value": [5.25]})
        _patch_connect(monkeypatch, frame)

        result = lakehouse_loader.load_fred_series_from_lakehouse("DGS10", date(2024, 1, 15))

    assert result is not None
    assert result.iloc[0] == 5.25


def test_load_fred_series_raises_on_future_index_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A date after as_of_date in the index is a broken SQL gate — must raise."""
    with temporary_overrides({"data.lake_dir": tmp_path}):
        _touch_table(tmp_path, "fred_series_observations")
        frame = pd.DataFrame({"date": ["2024-01-20"], "value": [5.25]})
        _patch_connect(monkeypatch, frame)

        with pytest.raises(ValueError, match="2024-01-20"):
            lakehouse_loader.load_fred_series_from_lakehouse("DGS10", date(2024, 1, 15))
