# test_broad_universe_securities.py

"""The Fama-MacBeth and lead-lag centrality broad universes are built from member securities."""

import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agentic_trading.agents.analysts.alt_data_pod import network_centrality_analyst as network
from agentic_trading.agents.analysts.alt_data_pod.network_centrality_analyst import (
    LeadLagNetworkCentralityAnalyst,
)
from agentic_trading.agents.analysts.factor_pod import fama_macbeth_analyst as fama
from agentic_trading.agents.analysts.factor_pod.fama_macbeth_analyst import FamaMacbethAnalyst
from agentic_trading.data.loaders.lakehouse_loader import IndexMember

_AS_OF = pd.Timestamp("2024-03-08")
_SHORT_MEMBERS = 2


def _forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("ticker-keyed loader must not be used for the broad universe")


def _series(size: int, seed: int) -> pd.Series:
    index = pd.bdate_range(end=_AS_OF, periods=size)
    return pd.Series(np.random.default_rng(seed).normal(0, 0.01, size), index=index)


def _install(
    monkeypatch: pytest.MonkeyPatch, module: Any, min_size: int, usable_size: int, members: int
) -> list[tuple[list[int], date, date]]:
    """Patches the module with `members` full-history securities plus short-history ones."""
    total = members + _SHORT_MEMBERS
    roster = [IndexMember(security_id=100 + i, ticker=f"T{i:03d}") for i in range(total)]
    calls: list[tuple[list[int], date, date]] = []

    def returns(ids: list[int], start: date, end: date) -> dict[int, pd.Series]:
        calls.append((list(ids), start, end))
        return {
            m.security_id: _series(usable_size if i < members else min_size - 1, i)
            for i, m in enumerate(roster)
            if m.security_id in ids
        }

    monkeypatch.setattr(module, "load_sp500_member_securities_asof", lambda d: roster)
    monkeypatch.setattr(module, "load_security_log_returns", returns)
    monkeypatch.setattr(module, "load_price_history_from_lakehouse", _forbidden, raising=False)
    monkeypatch.setattr(module, "load_sp500_constituents_asof", _forbidden, raising=False)
    return calls


def _info_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]


@pytest.fixture
def package_log(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    package = logging.getLogger("agentic_trading")
    package.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="agentic_trading")
    yield caplog
    package.removeHandler(caplog.handler)


def test_fama_macbeth_universe_keyed_by_ticker_and_logs_exclusions(
    monkeypatch: pytest.MonkeyPatch, package_log: pytest.LogCaptureFixture
) -> None:
    """Members with enough returns are keyed by ticker; the excluded count is logged at INFO."""
    min_size = fama._RISK_WINDOW // 2
    calls = _install(monkeypatch, fama, min_size, min_size, fama._MIN_UNIVERSE_SIZE)

    result = FamaMacbethAnalyst()._load_broad_lakehouse_universe(_AS_OF)

    assert result is not None
    tickers, returns = result
    expected = {f"T{i:03d}" for i in range(fama._MIN_UNIVERSE_SIZE)}
    assert set(tickers) == expected == set(returns)
    assert all(series.size >= min_size for series in returns.values())
    assert calls and calls[0][2] == _AS_OF.date()
    assert any(str(_SHORT_MEMBERS) in message for message in _info_messages(package_log))


def test_fama_macbeth_universe_none_when_membership_missing_or_too_thin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No membership, or fewer usable members than the minimum, gives None."""
    min_size = fama._RISK_WINDOW // 2
    _install(monkeypatch, fama, min_size, min_size, fama._MIN_UNIVERSE_SIZE - 1)
    assert FamaMacbethAnalyst()._load_broad_lakehouse_universe(_AS_OF) is None

    monkeypatch.setattr(fama, "load_sp500_member_securities_asof", lambda d: None)
    assert FamaMacbethAnalyst()._load_broad_lakehouse_universe(_AS_OF) is None


def test_network_universe_keyed_by_ticker_and_logs_exclusions(
    monkeypatch: pytest.MonkeyPatch, package_log: pytest.LogCaptureFixture
) -> None:
    """Members with enough returns are keyed by ticker; the excluded count is logged at INFO."""
    min_size = network.WINDOW + network.LAG
    calls = _install(monkeypatch, network, min_size, min_size, network._MIN_UNIVERSE_SIZE)

    result = LeadLagNetworkCentralityAnalyst()._load_broad_lakehouse_universe(_AS_OF)

    assert result is not None
    assert set(result) == {f"T{i:03d}" for i in range(network._MIN_UNIVERSE_SIZE)}
    assert all(series.size >= min_size for series in result.values())
    assert calls and calls[0][1:] == (
        _AS_OF.date() - timedelta(days=network._UNIVERSE_HISTORY_DAYS),
        _AS_OF.date(),
    )
    assert any(str(_SHORT_MEMBERS) in message for message in _info_messages(package_log))


def test_network_universe_none_when_membership_missing_or_too_thin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No membership, or fewer usable members than the minimum, gives None."""
    min_size = network.WINDOW + network.LAG
    _install(monkeypatch, network, min_size, min_size, network._MIN_UNIVERSE_SIZE - 1)
    assert LeadLagNetworkCentralityAnalyst()._load_broad_lakehouse_universe(_AS_OF) is None

    monkeypatch.setattr(network, "load_sp500_member_securities_asof", lambda d: None)
    assert LeadLagNetworkCentralityAnalyst()._load_broad_lakehouse_universe(_AS_OF) is None
