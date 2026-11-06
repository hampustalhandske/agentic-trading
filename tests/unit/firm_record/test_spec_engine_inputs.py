# test_spec_engine_inputs.py

"""AT-23: the firm spec hashes the engine-input builder (`backtest.engine_inputs`) and the
loaders it reads prices and rates through, whether or not they are loaded, so a record whose
engine inputs come from other code gets another spec hash, and a store paused under the old
hash is never resumed by it: the new record starts at the window's first session in its own
trial directory and leaves the old one as it was."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from agentic_trading.agents.base import AnalystAgent, ManagerAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.roster import build_default_analysts, build_default_managers
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.backtest.engine_inputs import EngineInputs
from agentic_trading.data.trading_calendar import sessions_between
from agentic_trading.firm_record import spec as spec_module
from agentic_trading.firm_record.runner import FirmRecordResult, run_firm_record
from agentic_trading.firm_record.spec import build_firm_spec
from agentic_trading.scoring.weighting import EqualWeight, Hedge, ShrinkageWeight
from agentic_trading.stage_a.spec import FirmSpec, firm_spec_hash

_ENGINE_INPUTS = "agentic_trading.backtest.engine_inputs"
_LOADERS = ("agentic_trading.data.loader", "agentic_trading.data.loaders.lakehouse_loader")
_OTHER_DIGEST = "f" * 64
_T = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


def _default_commander() -> Commander:
    analysts = build_default_analysts(excluded_analyst_ids={"pairs_cointegration_analyst"})
    return Commander(
        managers=build_default_managers(analysts), weighting_strategy=ShrinkageWeight(Hedge())
    )


def _default_spec() -> FirmSpec:
    """The spec of the default roster over a short 2010 window."""
    return build_firm_spec(
        _default_commander(), ["SPY", "AAPL"], date(2010, 1, 4), date(2010, 1, 29), _T
    )


def _swap_digest(monkeypatch: pytest.MonkeyPatch, module: str, digest: str) -> None:
    """Make the spec record `digest` as `module`'s hash, as if its source had changed, both
    where it is hashed as a loaded module and where it is reached from a hashed root."""
    loaded = spec_module._loaded_module_hashes
    reached = spec_module._module_hashes

    def swapped(hashes: dict[str, str]) -> dict[str, str]:
        return {name: digest if name == module else value for name, value in hashes.items()}

    def swapped_loaded() -> dict[str, str]:
        return swapped(loaded())

    def swapped_reached(root: str) -> dict[str, str]:
        return swapped(reached(root))

    monkeypatch.setattr(spec_module, "_loaded_module_hashes", swapped_loaded)
    monkeypatch.setattr(spec_module, "_module_hashes", swapped_reached)


def test_engine_inputs_and_its_loaders_are_hashed_without_being_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no loaded module counted, the spec still lists the engine-input builder and the
    price and rate loaders: they are reached from the hashed roots, not by being imported."""
    loaded = _default_spec()
    monkeypatch.setattr(spec_module, "_loaded_module_hashes", dict)
    reached = _default_spec()

    for module in (_ENGINE_INPUTS, *_LOADERS):
        assert module in reached.module_hashes, module
        assert reached.module_hashes[module] == loaded.module_hashes[module], module


@pytest.mark.parametrize("module", [_ENGINE_INPUTS, *_LOADERS])
def test_another_engine_input_source_is_another_spec(
    monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    """Two specs that differ only in the recorded hash of the engine-input builder, or of a
    loader it reads through, have different spec hashes."""
    base = _default_spec()
    _swap_digest(monkeypatch, module, _OTHER_DIGEST)
    changed = _default_spec()

    differing = {
        name
        for name in base.module_hashes.keys() | changed.module_hashes.keys()
        if base.module_hashes.get(name) != changed.module_hashes.get(name)
    }
    assert differing == {module}
    assert changed.module_hashes[module] == _OTHER_DIGEST
    assert firm_spec_hash(changed) != firm_spec_hash(base)


_UNIVERSE = ["SPY"]
_WARMUP = pd.DatetimeIndex(sessions_between(date(2015, 12, 1), date(2015, 12, 31)))
_START, _END = date(2016, 1, 4), date(2016, 1, 15)
_SESSIONS = pd.DatetimeIndex(sessions_between(_START, _END))


class _LongAnalyst(AnalystAgent):
    agent_id: str = "long_stub_analyst"

    def generate_signal(
        self, asset: str, market_data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            asset=asset,
            as_of_date=as_of_date.date(),
            direction="long",
            confidence=0.5,
            rationale="stub",
        )


class _ViewConstructor(PortfolioConstructor):
    def construct(self, views, returns_history, as_of_date, previous_weights=None):
        return {asset: 0.3 * view.net_signal for asset, view in views.items()}


class _PassThroughManager(ManagerAgent):
    def scale_factor(self, signal, market_data, as_of_date) -> float:
        return 1.0


def _record_commander() -> Commander:
    return Commander(
        managers=[
            _PassThroughManager(
                agent_id="stub_manager",
                managed_analysts=[_LongAnalyst()],
                weighting_strategy=EqualWeight(),
                portfolio_constructor=_ViewConstructor(),
            )
        ],
        weighting_strategy=EqualWeight(),
    )


def _inputs() -> EngineInputs:
    index = _WARMUP.append(_SESSIONS)
    rng = np.random.default_rng(5)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(index)))), index)
    frame = pd.DataFrame(
        {
            ("SPY", "open"): close,
            ("SPY", "close"): close,
            ("SPY", "log_return"): np.log(close / close.shift(1)),
        }
    )
    return EngineInputs(assets=_UNIVERSE, market_data=frame, market_data_source="lakehouse data")


def _run(store_dir: Path, max_days: int | None = None) -> FirmRecordResult:
    return run_firm_record(
        _record_commander(),
        _UNIVERSE,
        _START,
        _END,
        store_dir,
        _inputs(),
        lake_time=_T,
        lake_version="lake",
        quantcore_version="0.3.1",
        max_days=max_days,
    )


def _snapshot(paths: list[Path]) -> dict[Path, bytes]:
    """Every file under `paths` (a directory's files recursively) to its bytes."""
    files: list[Path] = []
    for path in paths:
        files += [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return {file: file.read_bytes() for file in sorted(files)}


def _decision_dates(trial: Path) -> list[date]:
    rows = pq.read_table(trial / "decisions.parquet").to_pylist()
    return sorted({row["decision_date"] for row in rows})


def test_a_paused_store_is_not_resumed_by_other_engine_input_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record paused after 2 days, run again on the same store directory once the
    engine-input builder's hash differs, gets a new spec hash, computes every session of the
    window from the first, and leaves the paused trial and its checkpoint byte for byte as
    they were."""
    paused = _run(tmp_path, max_days=2)
    assert (paused.days_written, paused.complete) == (2, False)
    assert paused.last_completed == _SESSIONS[1].date()
    old_trial = tmp_path / paused.spec_hash
    old_checkpoint = tmp_path / f"{paused.spec_hash}.checkpoint.sqlite"
    assert old_trial.is_dir() and old_checkpoint.is_file()
    before = _snapshot([old_trial, old_checkpoint])

    _swap_digest(monkeypatch, _ENGINE_INPUTS, _OTHER_DIGEST)
    fresh = _run(tmp_path)

    assert fresh.spec_hash != paused.spec_hash
    assert (fresh.days_written, fresh.complete) == (len(_SESSIONS), True)
    new_trial = tmp_path / fresh.spec_hash
    assert _decision_dates(new_trial) == [day.date() for day in _SESSIONS]
    assert (tmp_path / f"{fresh.spec_hash}.checkpoint.sqlite").is_file()
    new_spec = json.loads((new_trial / "spec.json").read_text())
    assert new_spec["spec"]["module_hashes"][_ENGINE_INPUTS] == _OTHER_DIGEST
    assert _snapshot([old_trial, old_checkpoint]) == before
    assert _decision_dates(old_trial) == [day.date() for day in _SESSIONS[:2]]


def test_the_same_engine_input_code_resumes_the_paused_store(tmp_path: Path) -> None:
    """Without a change to the engine-input builder the paused record resumes under its own
    spec hash and computes only the sessions it had not written."""
    paused = _run(tmp_path, max_days=2)
    resumed = _run(tmp_path)

    assert resumed.spec_hash == paused.spec_hash
    assert (resumed.days_written, resumed.complete) == (len(_SESSIONS) - 2, True)
