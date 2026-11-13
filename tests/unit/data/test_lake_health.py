# test_lake_health.py

"""`require_lake_health` reads `GET /v1/health` and refuses a run unless the lake's `status` is
"ok" and every table the run reads that has an entry under `write_times.tables` is `ok`. The
lake answers HTTP 200 when degraded too, so only those fields decide; a table without an entry
needs nothing."""

from typing import Any

import httpx
import pytest

from agentic_trading.data.lakehouse_client import Lake
from agentic_trading.data.loaders import lakehouse_loader as ll


def test_a_healthy_lake_passes(fake_lake: Any) -> None:
    """Status ok, a read table ok, and an unread table not ok: the run may start."""
    fake_lake.write_times = {
        "instrument_prices": {"ok": True},
        "reddit_comments": {"ok": False},
    }

    ll.require_lake_health()


def test_a_read_table_without_an_entry_needs_nothing(fake_lake: Any) -> None:
    fake_lake.write_times = {}

    ll.require_lake_health()


@pytest.mark.parametrize("table", ["instrument_prices", "fred_vintages_log"])
def test_a_degraded_read_table_refuses_naming_it(fake_lake: Any, table: str) -> None:
    """A read table, or the log a pinned read of it goes through, whose entry is not ok refuses
    the run, naming the table, while the lake's overall status is still ok."""
    fake_lake.write_times = {table: {"ok": False, "age_hours": 50.0}, "sec_filings": {"ok": True}}

    with pytest.raises(ll.LakeHealthError, match=table) as refused:
        ll.require_lake_health()

    assert "sec_filings" not in str(refused.value)


def test_an_entry_without_ok_true_refuses(fake_lake: Any) -> None:
    """An entry whose `ok` is missing is not ok."""
    fake_lake.write_times = {"bis_observations": {"age_hours": 1.0}}

    with pytest.raises(ll.LakeHealthError, match="bis_observations"):
        ll.require_lake_health()


@pytest.mark.parametrize("status", ["degraded", None])
def test_a_status_other_than_ok_refuses_naming_it(fake_lake: Any, status: str | None) -> None:
    fake_lake.health_status = status

    with pytest.raises(ll.LakeHealthError, match=f"status {status!r}"):
        ll.require_lake_health()


def test_a_degraded_http_200_answer_from_the_service_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service answers HTTP 200 with status "degraded" when a table lacks a write time it
    needs: the real client returns that body, and the check refuses on its fields, naming the
    status and the table."""
    body = {
        "status": "degraded",
        "version": "abc",
        "write_times": {"tables": {"finra_short_interest": {"ok": False}}},
    }
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=body)

    lake = Lake(base_url="http://lake.test", token="t", transport=httpx.MockTransport(handler))
    monkeypatch.setattr(ll, "_lake", lambda: lake)

    with pytest.raises(ll.LakeHealthError, match="degraded") as refused:
        ll.require_lake_health()

    assert "finra_short_interest" in str(refused.value)
    assert [request.url.path for request in sent] == ["/v1/health"]
