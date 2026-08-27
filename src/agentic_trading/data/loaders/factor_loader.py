# factor_loader.py

"""Ken French Data Library daily factor returns, cached via `cached_fetch`.

The library publishes monthly, so caching by `(source, identifier, as_of_date)`
means this only actually re-downloads once per calendar-month bucket per
`as_of_date` in practice."""

import io
import zipfile
from datetime import date

import pandas as pd
import requests

from agentic_trading.data.cache import cached_fetch
from agentic_trading.logging_config import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
_FILES = {
    "3factor": "F-F_Research_Data_Factors_daily_CSV.zip",
    "5factor": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
}


def _download_factor_csv(model: str) -> pd.DataFrame:
    filename = _FILES[model]
    logger.info("Downloading Ken French %s factor file: %s", model, filename)
    response = requests.get(f"{_BASE_URL}/{filename}", timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(csv_name) as handle:
            raw_lines = handle.read().decode("utf-8", errors="ignore").splitlines()

    data_start = next(
        index
        for index, line in enumerate(raw_lines)
        if line.strip().startswith("19") or line.strip().startswith("20")
    )
    header_line = raw_lines[data_start - 1]
    columns = ["date", *[column.strip() for column in header_line.split(",") if column.strip()]]

    data_lines = []
    for line in raw_lines[data_start:]:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            break
        data_lines.append(stripped)

    frame = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None, names=columns)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    frame = frame.set_index("date")
    return frame.astype(float) / 100.0


def load_fama_french_factors(as_of_date: pd.Timestamp, model: str = "3factor") -> pd.DataFrame:
    """Daily Fama-French factor returns for `model` ("3factor" or
    "5factor"), truncated to rows with date `<= as_of_date`."""
    if model not in _FILES:
        raise ValueError(f"Unsupported factor model {model!r}; supported: {sorted(_FILES)}")

    frame = cached_fetch(
        source="fama_french",
        identifier=model,
        as_of_date=date(as_of_date.year, as_of_date.month, 1),
        fetch_fn=lambda: _download_factor_csv(model),
    )
    return frame.loc[frame.index <= as_of_date]
