from pathlib import Path

from agentic_trading.config import settings, temporary_overrides
from agentic_trading.data.loaders import lakehouse_loader


def test_lake_dir_override_changes_where_the_loader_looks(tmp_path: Path) -> None:
    """The lakehouse location is configuration, read at use time: pointing
    `data.lake_dir` at an empty directory makes the lake unavailable."""
    with temporary_overrides({"data.lake_dir": tmp_path}):
        assert Path(settings.data.lake_dir) == tmp_path
        assert not lakehouse_loader.lakehouse_available()
        assert lakehouse_loader.load_price_history_from_lakehouse(["AAPL"], *_range()) is None


def _range() -> tuple:
    from datetime import date

    return date(2024, 1, 2), date(2024, 1, 5)
