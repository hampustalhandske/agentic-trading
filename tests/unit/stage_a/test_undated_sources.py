# test_undated_sources.py

"""An undated source (`decisions.UNDATED_SOURCES`, today only `sec_sic`) is an input with no
publication date, such as a current classification applied to past dates. A decision row never
names one: `source_of` and `row_source` skip it, so a row resting only on prices and an undated
input is price-only and passes the look-ahead audit with no `source_as_of`. Each trial declares
its undated inputs once in sources.json, with the limitation they bring."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from agentic_trading.agents.analysts import KalmanHedgeRatioAnalyst, PairsCointegrationAnalyst
from agentic_trading.agents.analysts.macro_rates_pod.cir_rate_regime_analyst import (
    CIRRateRegimeAnalyst,
)
from agentic_trading.agents.base import AnalystAgent, DataRequirement
from agentic_trading.agents.schemas import AgentSignal
from agentic_trading.stage_a.decisions import (
    SOURCES,
    UNDATED_SOURCES,
    LookAheadError,
    row_source,
    source_of,
    undated_sources_of,
)
from agentic_trading.stage_a.spec import SpecRecord, build_spec, spec_hash
from agentic_trading.stage_a.store import SignalStore

_DAY = date(2016, 3, 1)
_SIC_LIMITATION = (
    "current SIC code applied to past dates in the manager's pair screen to choose candidate "
    "partners; signals use prices only"
)
_SIC_DECLARATION = {"name": "sec_sic", "undated": True, "limitation": _SIC_LIMITATION}


def _store(store_dir: Path, analyst: AnalystAgent) -> tuple[SignalStore, Path]:
    spec = build_spec(analyst, ["AAPL"], _DAY, _DAY)
    record = SpecRecord(
        spec=spec,
        spec_hash=spec_hash(spec),
        lake_time=datetime(2026, 11, 5, tzinfo=UTC),
        lake_version="v",
        quantcore_version="0",
        quantcore_sha=None,
    )
    store = SignalStore(store_dir, record)
    store.open()
    return store, store_dir / record.spec_hash


def _requires(*sources: str) -> CIRRateRegimeAnalyst:
    """A CIR analyst whose data requirements read `sources`, one requirement each."""
    requirements = tuple(
        DataRequirement(source=source, identifier=f"id{n}", asset_scoped=False, description="")
        for n, source in enumerate(sources)
    )
    return CIRRateRegimeAnalyst(data_requirements=requirements)


def test_sec_sic_is_undated_with_its_limitation() -> None:
    """sec_sic is undated, carries the limitation line, and has no publish-time flag."""
    assert UNDATED_SOURCES == {"sec_sic": _SIC_LIMITATION}
    assert "sec_sic" not in SOURCES


def test_source_of_skips_undated_sources() -> None:
    """A row never names an undated source; a dated one beside it still names the row."""
    assert source_of(["sec_sic"]) is None
    assert source_of(["sec_sic", "fred"]) == "fred"
    assert source_of(["sec_sic", "dividends"]) == "dividends"
    with pytest.raises(LookAheadError, match="not_a_source"):
        source_of(["sec_sic", "not_a_source"])


def test_stat_arb_rows_are_price_only() -> None:
    """The pairs and kalman analysts declare only sec_sic, so their rows carry no source."""
    for analyst in (PairsCointegrationAnalyst(), KalmanHedgeRatioAnalyst()):
        assert row_source(analyst) is None, analyst.agent_id
        assert undated_sources_of(analyst) == [_SIC_DECLARATION], analyst.agent_id


def test_undated_sources_of_a_dated_analyst_is_empty() -> None:
    """An analyst with only dated sources declares no undated input."""
    assert undated_sources_of(CIRRateRegimeAnalyst()) == []
    assert undated_sources_of(_requires("fred", "sec_sic")) == [_SIC_DECLARATION]


def test_sources_json_declares_undated_inputs(tmp_path: Path) -> None:
    """sources.json lists the row source first, then each undated input with its limitation."""
    _, pairs_trial = _store(tmp_path / "pairs", PairsCointegrationAnalyst())
    assert json.loads((pairs_trial / "sources.json").read_text()) == [_SIC_DECLARATION]

    _, mixed_trial = _store(tmp_path / "mixed", _requires("fred", "sec_sic"))
    assert json.loads((mixed_trial / "sources.json").read_text()) == [
        {"name": "fred", "can_publish_after_close": True},
        _SIC_DECLARATION,
    ]

    _, cir_trial = _store(tmp_path / "cir", CIRRateRegimeAnalyst())
    assert json.loads((cir_trial / "sources.json").read_text()) == [
        {"name": "fred", "can_publish_after_close": True}
    ]


def test_store_writes_a_stat_arb_active_row_without_source_as_of(tmp_path: Path) -> None:
    """A pairs or kalman active row is price-only: written with no source and no source_as_of,
    and refused if it claims a source_as_of."""
    for analyst in (PairsCointegrationAnalyst(), KalmanHedgeRatioAnalyst()):
        store, _ = _store(tmp_path / analyst.agent_id, analyst)
        row = AgentSignal(
            agent_id=analyst.agent_id,
            asset="AAPL",
            as_of_date=_DAY,
            direction="long",
            confidence=0.5,
            rationale="",
        )
        store.write_day(_DAY, [row], {"AAPL": "AAPL"}, "equity")
        assert store.last_completed() == _DAY, analyst.agent_id

        dated_store, _ = _store(tmp_path / f"{analyst.agent_id}-dated", analyst)
        dated = row.model_copy(update={"source_as_of": _DAY})
        with pytest.raises(LookAheadError):
            dated_store.write_day(_DAY, [dated], {"AAPL": "AAPL"}, "equity")
