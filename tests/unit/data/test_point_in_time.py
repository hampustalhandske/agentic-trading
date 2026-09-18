"""Unit tests for agentic_trading.data.point_in_time.

No mocking needed here — both functions operate purely on in-memory
pandas objects.
"""

from datetime import date

import pandas as pd
import pytest

from agentic_trading.data.point_in_time import assert_point_in_time, get_point_in_time_data


def _sample_data() -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"])
    return pd.DataFrame({"close": [10.0, 20.0, 30.0, 40.0]}, index=index)


def test_data_after_as_of_date_is_excluded() -> None:
    """No row dated after as_of_date may appear in the result — the no-look-ahead guarantee."""
    result = get_point_in_time_data(_sample_data(), pd.Timestamp("2024-01-05"))

    assert (result.index <= pd.Timestamp("2024-01-05")).all()
    assert pd.Timestamp("2024-01-08") not in result.index


def test_data_exactly_on_as_of_date_is_included() -> None:
    """The as_of_date's own row is visible that day, not withheld until the next simulated day."""
    result = get_point_in_time_data(_sample_data(), pd.Timestamp("2024-01-05"))

    assert pd.Timestamp("2024-01-05") in result.index
    assert result.loc[pd.Timestamp("2024-01-05"), "close"] == 30.0


def test_as_of_date_with_no_exact_row_returns_all_prior_rows() -> None:
    """A non-trading as_of_date with no row of its own still acts as an inclusive `<=` range filter.

    The docstring's only stated guarantee is that no row after as_of_date appears in the
    result — it does not require an exact index match to exist. The natural reading is
    therefore a `<=` range filter: 2024-01-04 has no row of its own, so the result should
    contain every row strictly before it (2024-01-02, 2024-01-03) and nothing from
    2024-01-05 onward.
    """
    result = get_point_in_time_data(_sample_data(), pd.Timestamp("2024-01-04"))

    assert list(result.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_as_of_date_before_all_data_returns_empty_result() -> None:
    """An as_of_date earlier than any available data yields an empty frame, not an error."""
    result = get_point_in_time_data(_sample_data(), pd.Timestamp("2020-01-01"))

    assert result.empty
    assert isinstance(result, pd.DataFrame)


def test_assert_point_in_time_passes_on_in_window_dataframe_column() -> None:
    """A DataFrame whose date column stays within as_of_date is a no-op."""
    frame = pd.DataFrame(
        {"filed_date": pd.to_datetime(["2024-01-02", "2024-01-05"]), "value": [1.0, 2.0]}
    )

    assert_point_in_time(frame, date(2024, 1, 5), date_column="filed_date")


def test_assert_point_in_time_raises_on_out_of_window_dataframe_column() -> None:
    """A row after as_of_date in the named column raises, naming both dates."""
    frame = pd.DataFrame(
        {"filed_date": pd.to_datetime(["2024-01-02", "2024-01-08"]), "value": [1.0, 2.0]}
    )

    with pytest.raises(ValueError, match="2024-01-08") as excinfo:
        assert_point_in_time(frame, date(2024, 1, 5), date_column="filed_date")
    assert "2024-01-05" in str(excinfo.value)


def test_assert_point_in_time_raises_on_out_of_window_index() -> None:
    """No date_column given checks the index instead."""
    series = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-02", "2024-01-08"]))

    with pytest.raises(ValueError, match="2024-01-08"):
        assert_point_in_time(series, date(2024, 1, 5))


def test_assert_point_in_time_passes_on_in_window_series_index() -> None:
    """A Series whose index stays within as_of_date is a no-op."""
    series = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-02", "2024-01-05"]))

    assert_point_in_time(series, date(2024, 1, 5))


def test_assert_point_in_time_coerces_string_dtype_column() -> None:
    """A date column stored as plain strings (not yet datetime64) is coerced before checking."""
    frame = pd.DataFrame({"dissemination_date": ["2024-01-02", "2024-01-08"], "value": [1.0, 2.0]})

    with pytest.raises(ValueError, match="2024-01-08"):
        assert_point_in_time(frame, date(2024, 1, 5), date_column="dissemination_date")


def test_assert_point_in_time_is_noop_on_none() -> None:
    """Many loaders return None on no coverage — that's a valid, non-violating result."""
    assert_point_in_time(None, date(2024, 1, 5))


def test_assert_point_in_time_is_noop_on_list_of_tickers() -> None:
    """load_sp500_constituents_asof returns tickers, not dates — nothing to check."""
    assert_point_in_time(["AAPL", "MSFT"], date(2024, 1, 5))


def test_assert_point_in_time_is_noop_on_empty_dataframe() -> None:
    """An empty frame/series has no offending rows by construction."""
    frame = pd.DataFrame({"filed_date": pd.to_datetime([]), "value": []})

    assert_point_in_time(frame, date(2024, 1, 5), date_column="filed_date")


def test_assert_point_in_time_message_lists_at_most_three_offending_dates() -> None:
    """Many violating rows still produce a short, readable message naming only the first three."""
    frame = pd.DataFrame(
        {"filed_date": pd.to_datetime(["2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09"])}
    )

    with pytest.raises(ValueError) as excinfo:
        assert_point_in_time(frame, date(2024, 1, 5), date_column="filed_date")

    message = str(excinfo.value)
    assert "2024-01-06" in message
    assert "2024-01-07" in message
    assert "2024-01-08" in message
    assert "2024-01-09" not in message
