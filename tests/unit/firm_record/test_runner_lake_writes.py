# test_runner_lake_writes.py

"""A firm record reads the lake as it stood at its lake time T, whatever the lake writes while
it runs: a new build of a rebuilt-whole table and `crypto_klines_1m` minutes written after T
during a pass that stops for memory, and read again by the resumed pass, leave the record's
rows those of a record of an unchanged lake."""

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import PrivateAttr

from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.data.loaders import lakehouse_loader as ll
from agentic_trading.scoring.weighting import EqualWeight
from unit.firm_record.test_runner import (
    _T,
    _PassThroughManager,
    _rows,
    _run,
    _ViewConstructor,
)

_BUILD_AT_T = "2026-11-05T03:00:00Z"
_AFTER_T = "2026-11-07T03:00:00Z"
_LATER_T = pd.Timestamp("2026-11-08", tz="UTC").to_pydatetime()
_KLINE_DAY = date(2015, 12, 30)
_SPLIT_COLUMNS = ["ticker", "ex_date", "ratio", "in_close_basis"]
# Lake writes a test schedules: call number of `_LakeReadingAnalyst.generate_signal` to the
# write run before that call's reads. Kept outside the analyst, whose private attributes the
# engine checkpoint pickles.
_WRITES_BEFORE_CALL: dict[int, Callable[[], None]] = {}


class _LakeReadingAnalyst(AnalystAgent):
    """Long at a confidence set by what it reads from the lake each day: AAPL's split ex-dates
    known on the decision day and the BTC-USD volume of `_KLINE_DAY`. Before the reads of a
    call it runs the write `_WRITES_BEFORE_CALL` holds for the call's number, once."""

    agent_id: str = "lake_reading_stub_analyst"
    _calls: int = PrivateAttr(default=0)

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        self._calls += 1
        write = _WRITES_BEFORE_CALL.pop(self._calls, None)
        if write is not None:
            write()
        splits = ll.load_split_ex_dates("AAPL", as_of_date.date())
        bars = ll.load_price_history_from_lakehouse(["BTC-USD"], _KLINE_DAY, _KLINE_DAY)
        volume = 0.0 if bars is None else float(bars["BTC-USD"]["volume"].sum())
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=min(1.0, 0.1 * len(splits) + 0.01 * volume),
            rationale="stub",
        )


def _commander() -> Commander:
    return Commander(
        managers=[
            _PassThroughManager(
                agent_id="lake_manager",
                managed_analysts=[_LakeReadingAnalyst()],
                weighting_strategy=EqualWeight(),
                portfolio_constructor=_ViewConstructor(),
            )
        ],
        weighting_strategy=EqualWeight(),
    )


def _minute(open_time: str, close: float, volume: float) -> dict[str, Any]:
    stamp = pd.Timestamp(open_time)
    return {
        "symbol": "BTCUSDT",
        "open_time": stamp,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": volume,
        "quote_volume": close * volume,
        "n_trades": 1,
        "taker_buy_base_volume": volume,
        "taker_buy_quote_volume": close * volume,
        "dt": stamp.normalize(),
        "table": "crypto_klines_1m",
    }


def _seed(lake: Any) -> None:
    """The lake at T: one AAPL split, and the first and last session minute of `_KLINE_DAY`."""
    lake.add(
        "yahoo_split_close_basis",
        pd.DataFrame([("AAPL", "2015-12-01", 0.5, True)], columns=_SPLIT_COLUMNS),
        ("ex_date",),
        scraped_at=_BUILD_AT_T,
        rebuilt_whole=True,
    )
    minutes = [_minute("2015-12-30 14:30", 400.0, 2.0), _minute("2015-12-30 20:59", 410.0, 3.0)]
    lake.add(
        "crypto_klines_1m",
        pd.DataFrame(minutes),
        ("dt",),
        scraped_at=_BUILD_AT_T,
        kind="append-only",
    )


def _write_after_t(lake: Any) -> None:
    """What the lake writes after T: a new split build with a second AAPL split, and a late
    minute of `_KLINE_DAY`."""
    lake.add_build(
        "yahoo_split_close_basis",
        pd.DataFrame(
            [("AAPL", "2015-12-01", 0.5, True), ("AAPL", "2015-12-15", 0.5, True)],
            columns=_SPLIT_COLUMNS,
        ),
        _AFTER_T,
    )
    lake.add_log_rows(
        "crypto_klines_1m", pd.DataFrame([_minute("2015-12-30 18:00", 405.0, 30.0)]), _AFTER_T
    )


def test_a_record_restarted_across_lake_writes_ends_with_the_unbroken_rows(
    tmp_path: Path, fake_lake: Any
) -> None:
    """A pass that sees the lake write in its second call, stops for memory after three days
    and is resumed by a fresh firm writes exactly the rows of a record that ran through on the
    lake as it stood at T."""
    _seed(fake_lake)
    unbroken = _run(tmp_path / "unbroken", commander=_commander())

    _WRITES_BEFORE_CALL[2] = lambda: _write_after_t(fake_lake)
    try:
        paused = _run(
            tmp_path / "paused",
            commander=_commander(),
            restart_above_rss_mb=100.0,
            restart_min_days=3,
            rss_mb=lambda: 101.0,
        )
    finally:
        _WRITES_BEFORE_CALL.clear()
    assert (paused.days_written, paused.paused_for_memory) == (3, True)
    assert fake_lake.version == "fake-code-1.fake-builds-2"
    resumed = _run(tmp_path / "paused", commander=_commander())

    assert resumed.complete and not resumed.paused_for_memory
    assert _rows(tmp_path / "paused", resumed) == _rows(tmp_path / "unbroken", unbroken)


def test_a_record_at_a_later_lake_time_reads_the_writes(tmp_path: Path, fake_lake: Any) -> None:
    """Negative control: a record at a lake time after the writes decides on the new build and
    the late minute, so its targets differ from a record at T."""
    _seed(fake_lake)
    _write_after_t(fake_lake)

    at_t = _run(tmp_path / "at_t", commander=_commander())
    later = _run(tmp_path / "later", commander=_commander(), lake_time=_LATER_T)

    def targets(store_dir: Path, result: Any) -> list[Any]:
        return [row["target_weight"] for row in _rows(store_dir, result)]

    assert targets(tmp_path / "at_t", at_t) != targets(tmp_path / "later", later)
    assert _T < _LATER_T
