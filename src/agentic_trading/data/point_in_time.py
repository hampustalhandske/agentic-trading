# point_in_time.py

"""Two structural enforcement points for the project's hard no-look-ahead
rule: on simulated day t, no agent may access data from day t+1 onward.

`get_point_in_time_data` is the shared one — every caller that hands the
engine's price-history DataFrame to an agent routes it through this first.

`assert_point_in_time` is the second, generic one: every lakehouse loader
in `data/loaders/lakehouse_loader.py` calls it on its own result right
before returning, on top of that loader's own SQL `WHERE ... <= as_of_date`
gate. The SQL gate is what actually filters the rows; this assertion is a
structural backstop that raises immediately if that gate is ever wrong or
missing, rather than letting a look-ahead leak pass through silently.

Together, these two functions are what CLAUDE.md's per-source point-in-time
table describes: price data enforced via `get_point_in_time_data`, every
other source self-asserting via `assert_point_in_time`.
"""

from datetime import date

import pandas as pd


def get_point_in_time_data(data: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Returns the subset of `data` visible as of `as_of_date` — no row
    with a date after `as_of_date` may appear in the result.

    This is the structural enforcement point for the project's hard
    no-look-ahead rule: on simulated day t, no agent may access data from
    day t+1 onward. Every caller that hands market data to an agent must
    route it through this function first; agents themselves must not be
    trusted to self-enforce this.
    """
    return data.loc[data.index <= as_of_date]


def assert_point_in_time(
    data: pd.DataFrame | pd.Series | list[str] | None,
    as_of_date: date,
    date_column: str | None = None,
) -> None:
    """Raises `ValueError` if `data` contains any date strictly after
    `as_of_date` — a lakehouse loader's self-check, called on its own
    result right before returning it.

    `data` may be a `pd.DataFrame`, a `pd.Series`, a `list[str]`, or
    `None`. For a `DataFrame`/`Series`, the dates checked come from
    `date_column` if given (coerced to datetime if not already), or from
    the index otherwise; an empty frame/series is a no-op. `list[str]` is
    a no-op — it's the shape `load_sp500_constituents_asof` returns
    (tickers, not dates), so there is nothing here to check. `None` is
    also a no-op — many loaders return `None` to mean "no coverage."

    Comparison is by calendar date, not timestamp, since some loaders
    store plain dates (`date`/`filed_date`/`dissemination_date`) while
    `as_of_date` here is always a `datetime.date`.
    """
    if data is None or isinstance(data, list):
        return

    dates = data.index if date_column is None else data[date_column]

    if len(dates) == 0:
        return

    dates = pd.to_datetime(pd.Index(dates))
    offending = dates[dates.date > as_of_date]
    if len(offending) > 0:
        offending_dates = sorted({d.date().isoformat() for d in offending[:3]})
        raise ValueError(
            f"Point-in-time violation: found date(s) {offending_dates} after "
            f"as_of_date={as_of_date.isoformat()}"
        )
