"""The availability record summed from the signal counts the engine checkpoint stores with each
day equals the one counted from the stored decisions, down to the bytes of
analyst_availability.json."""

import json
from datetime import date
from pathlib import Path

from agentic_trading.agents.schemas import (
    ManagerSignal,
    OptionContract,
    PortfolioDecision,
    SignalStatus,
    VolatilityEstimator,
)
from agentic_trading.backtest import checkpoint_store
from agentic_trading.firm_record.availability import (
    analyst_availability,
    analyst_availability_of_counts,
    analyst_availability_of_payloads,
    day_signal_counts,
)
from agentic_trading.firm_record.submission import (
    ANALYST_AVAILABILITY_FILE,
    AgenticMeta,
    write_analyst_availability,
)
from agentic_trading.provenance import runtime_environment

_ROSTER = ["vol_analyst", "plain_analyst", "silent_analyst"]
_DAYS = [date(2015, 12, 30), date(2015, 12, 31), date(2016, 1, 4), date(2016, 1, 5)]


def _signal(
    analyst: str,
    day: date,
    status: SignalStatus = "active",
    estimator: VolatilityEstimator | None = None,
    asset: str = "SPY",
    pod: str = "pod",
) -> ManagerSignal:
    return ManagerSignal(
        manager_id=pod,
        source_agent_id=analyst,
        asset=asset,
        as_of_date=day,
        direction="long" if status == "active" else "flat",
        confidence=0.2 if status == "active" else 0.0,
        raw_confidence=0.2 if status == "active" else 0.0,
        scale_applied=1.0,
        rationale="stub",
        status=status,
        volatility_estimator=estimator,
    )


def _decisions(index: int, day: date) -> list[PortfolioDecision]:
    """A day's decisions: signals of every status and estimator, an analyst outside the roster,
    one analyst in two pods, a stale signal of the day before, and an option decision repeating
    its underlying's signals; the last day has no counted signal at all."""
    if index == len(_DAYS) - 1:
        return [
            PortfolioDecision(
                asset="SPY", as_of_date=day, target_weight=0.0, contributing_signals=[]
            )
        ]
    signals = [
        _signal("vol_analyst", day, estimator="ewma" if index % 2 else "garch_11"),
        _signal("vol_analyst", day, "unavailable", estimator=None, pod="other_pod"),
        _signal("plain_analyst", day, "error" if index == 1 else "active"),
        _signal("outside_analyst", day, "unavailable"),
        _signal("plain_analyst", _DAYS[0] if index else day, "active"),
    ]
    put = OptionContract(
        underlying="SPY", expiry=date(2016, 3, 18), strike=180.0, option_type="put"
    )
    return [
        PortfolioDecision(
            asset="SPY", as_of_date=day, target_weight=0.1, contributing_signals=signals
        ),
        PortfolioDecision(
            asset="QQQ",
            as_of_date=day,
            target_weight=0.0,
            contributing_signals=[_signal("vol_analyst", day, estimator="ewma", asset="QQQ")],
        ),
        PortfolioDecision(
            asset=put.contract_id,
            as_of_date=day,
            target_weight=0.01,
            contributing_signals=signals,
            instrument=put,
        ),
    ]


def _checkpoint(path: Path) -> dict[date, list[PortfolioDecision]]:
    """Appends every day of `_DAYS` to a checkpoint at `path`; returns the decisions by day."""
    connection = checkpoint_store.connect(path)
    by_day = {day: _decisions(index, day) for index, day in enumerate(_DAYS)}
    for index, (day, decisions) in enumerate(by_day.items()):
        checkpoint_store.append_day(
            connection,
            as_of_date=day,
            portfolio_value=1.0,
            new_track_record=[],
            decisions=decisions,
            new_trades=[],
            next_day_index=index + 1,
            start_date=_DAYS[0],
            end_date=_DAYS[-1],
            assets=["SPY", "QQQ"],
            portfolio={"cash": 1.0},
            signal_health={},
        )
    connection.close()
    return by_day


def test_day_signal_counts_count_what_the_record_counts() -> None:
    """A day's counts hold each counted signal once per (analyst, status, estimator): the option
    decision and the stale signal are left out."""
    counts = day_signal_counts(_DAYS[1], _decisions(1, _DAYS[1]))

    assert counts == {
        ("vol_analyst", "active", "ewma"): 2,
        ("vol_analyst", "unavailable", None): 1,
        ("plain_analyst", "error", None): 1,
        ("outside_analyst", "unavailable", None): 1,
    }


def test_the_stored_tally_gives_the_record_of_the_stored_decisions(tmp_path: Path) -> None:
    """The record summed from the checkpoint's stored counts equals the record counted from its
    stored decisions and from the decisions themselves, and writes the same
    analyst_availability.json bytes."""
    path = tmp_path / "checkpoint.db"
    by_day = _checkpoint(path)
    connection = checkpoint_store.connect(path, resume=True)
    days = [date.fromisoformat(day) for day in checkpoint_store.decision_days(connection)]
    tally = [
        (date.fromisoformat(day), analyst_id, status, estimator, signals)
        for day, analyst_id, status, estimator, signals in checkpoint_store.iter_signal_tally(
            connection
        )
    ]
    payloads = [
        (date.fromisoformat(day), payload)
        for day, payload in checkpoint_store.iter_decision_payloads(connection)
    ]
    connection.close()

    from_counts = analyst_availability_of_counts(days, tally, _ROSTER)
    from_payloads = analyst_availability_of_payloads(payloads, _ROSTER)

    assert days == _DAYS
    assert from_counts == from_payloads == analyst_availability(by_day, _ROSTER)
    assert set(from_counts.root) == {*_ROSTER, "outside_analyst"}
    assert list(from_counts.root["vol_analyst"]) == ["2015", "2016", "total"]
    written = []
    for availability in (from_counts, from_payloads):
        meta = AgenticMeta(
            counted=False,
            src_tree_clean=True,
            git_sha="0" * 40,
            declared_n_from="uncounted_default",
            environment=runtime_environment(),
        )
        trial = tmp_path / str(len(written))
        trial.mkdir()
        write_analyst_availability(trial, availability, meta)
        written.append((trial / ANALYST_AVAILABILITY_FILE).read_bytes())
    assert written[0] == written[1]
    assert json.loads(written[0])["silent_analyst"]["total"]["available_share"] == 0.0
