# cli.py

from datetime import date

import typer

from agentic_trading.agents.commander import Commander
from agentic_trading.agents.portfolio_construction import PortfolioConstructor
from agentic_trading.agents.roster import build_default_analysts, build_default_managers
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.data.loader import load_price_history_with_fallback

# NOTE: adjust this import to wherever ticker_catalog.py actually lives in the
# repo (it reads as a data/ module from its own docstring — "data/loader.py"
# is its neighbor above — but confirm the real path before running this).
from agentic_trading.data.ticker_catalog import (
    TICKER_GROUPS,
    AssetClass,
    TickerGroup,
    tickers_by_asset_class,
)
from agentic_trading.logging_config import configure_logging, get_logger
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import Hedge

app = typer.Typer()

# Every asset class CATALOG actually classifies into (see ticker_catalog.py's
# _QUOTE_TYPE_TO_ASSET_CLASS) — options are deliberately excluded there, so
# they're excluded here too, not an oversight.
_ASSET_CLASSES: tuple[AssetClass, ...] = ("stock", "index", "commodity", "forex", "crypto")

_DATE_FORMAT_HINT = "YYYY-MM-DD"
_DATE_EXAMPLE = "2023-01-01"


# --------------------------------------------------------------------------
# Interactive prompts — only reached when the CLI wasn't given enough on the
# command line to run non-interactively. Each one loops on bad input rather
# than crashing, since a mistyped date or an out-of-range menu number is the
# most common way a person actually interacts with these.
# --------------------------------------------------------------------------


def _prompt_menu_choice(prompt: str, valid: set[str], default: str) -> str:
    while True:
        raw = typer.prompt(prompt, default=default)
        choice = str(raw.strip())
        if choice in valid:
            return choice
        typer.echo(f"Enter one of: {', '.join(sorted(valid))}.")


def _prompt_ticker_group() -> list[str]:
    typer.echo()
    for i, group in enumerate(TICKER_GROUPS, start=1):
        symbols_preview = ", ".join(group.symbols)
        typer.echo(f"  {i:>2}. {group.name} — {group.description}")
        typer.echo(f"      [{symbols_preview}]")
    typer.echo()
    while True:
        raw = typer.prompt(f"Pick a basket (1-{len(TICKER_GROUPS)})")
        if raw.strip().isdigit() and 1 <= int(raw) <= len(TICKER_GROUPS):
            chosen: TickerGroup = TICKER_GROUPS[int(raw) - 1]
            return list(chosen.symbols)
        typer.echo(f"Enter a number between 1 and {len(TICKER_GROUPS)}.")


def _prompt_by_asset_class() -> list[str]:
    typer.echo()
    for i, asset_class in enumerate(_ASSET_CLASSES, start=1):
        count = len(tickers_by_asset_class(asset_class))
        typer.echo(f"  {i}. {asset_class} ({count} in catalog)")
    typer.echo()
    while True:
        raw = typer.prompt(f"Pick an asset class (1-{len(_ASSET_CLASSES)})")
        if raw.strip().isdigit() and 1 <= int(raw) <= len(_ASSET_CLASSES):
            asset_class = _ASSET_CLASSES[int(raw) - 1]
            break
        typer.echo(f"Enter a number between 1 and {len(_ASSET_CLASSES)}.")

    entries = tickers_by_asset_class(asset_class)
    typer.echo()
    for i, entry in enumerate(entries, start=1):
        typer.echo(f"  {i:>2}. {entry.symbol:<10} {entry.name}")
    typer.echo()
    raw = typer.prompt("Pick one or more (comma-separated numbers, e.g. 1,3,5 — or 'all')")
    if raw.strip().lower() == "all":
        return [entry.symbol for entry in entries]

    selected = []
    for token in raw.split(","):
        token = token.strip()
        if token.isdigit() and 1 <= int(token) <= len(entries):
            selected.append(entries[int(token) - 1].symbol)
    if not selected:
        typer.echo("No valid selections — falling back to custom entry.")
        return _prompt_custom_symbols()
    return selected


def _prompt_custom_symbols() -> list[str]:
    while True:
        raw = typer.prompt("Enter ticker symbols, comma- or space-separated (e.g. AAPL, ...)")
        symbols = [s.strip().upper() for s in raw.replace(",", " ").split() if s.strip()]
        if symbols:
            return symbols
        typer.echo("Enter at least one symbol.")


def _prompt_for_assets() -> list[str]:
    typer.echo()
    typer.echo("Which assets should this backtest cover?")
    typer.echo("  1. Pick a preset basket (recommended)")
    typer.echo("  2. Browse the catalog by asset class")
    typer.echo("  3. Enter custom ticker symbols")
    choice = _prompt_menu_choice("Choice", {"1", "2", "3"}, default="1")
    if choice == "1":
        return _prompt_ticker_group()
    if choice == "2":
        return _prompt_by_asset_class()
    return _prompt_custom_symbols()


def _prompt_for_date(label: str, *, after: date | None = None) -> date:
    while True:
        raw = typer.prompt(f"{label} (format: {_DATE_FORMAT_HINT}, e.g. {_DATE_EXAMPLE})")
        try:
            parsed = date.fromisoformat(raw.strip())
        except ValueError:
            typer.echo(f"Not a valid date — use {_DATE_FORMAT_HINT}, e.g. {_DATE_EXAMPLE}.")
            continue
        if after is not None and parsed <= after:
            typer.echo(f"Must be after the start date ({after.isoformat()}).")
            continue
        return parsed


def _prompt_for_portfolio_construction() -> bool:
    return typer.confirm(
        "Run the optional portfolio-construction stage "
        "(Black-Litterman + risk parity, after the Commander)?",
        default=False,
    )


def _parse_date_flag(value: str, flag_name: str) -> date:
    """Validates a date string supplied via a CLI flag (not interactively) —
    fails fast with a clear message rather than silently falling through to
    an interactive prompt, which would hang a scripted/CI invocation."""
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise typer.BadParameter(
            f"{flag_name} must be in {_DATE_FORMAT_HINT} format (e.g. {_DATE_EXAMPLE}), "
            f"got {value!r}."
        ) from exc


# --------------------------------------------------------------------------
# Command
# --------------------------------------------------------------------------


@app.command()
def run(
    assets: list[str] | None = typer.Argument(  # noqa: B008
        None,
        help="Asset symbols to include in the backtest (e.g. AAPL MSFT GOOGL). "
        "Omit to pick interactively from the ticker catalog.",
    ),
    start_date: str | None = typer.Option(
        None,
        help=f"Backtest start date, format {_DATE_FORMAT_HINT} (e.g. {_DATE_EXAMPLE}). "
        "Omit to be prompted.",
    ),
    end_date: str | None = typer.Option(
        None,
        help=f"Backtest end date, format {_DATE_FORMAT_HINT} (e.g. {_DATE_EXAMPLE}). "
        "Omit to be prompted.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Log detailed per-step timing (console/log-file only)."
    ),
    no_log: bool = typer.Option(
        False, "--no-log", help="Disable logging entirely — nothing to console or the log file."
    ),
    portfolio_construction: bool | None = typer.Option(
        None,
        "--portfolio-construction/--no-portfolio-construction",
        help="Run the optional post-Commander portfolio construction stage (risk parity by "
        "default). Omit to be asked interactively; defaults to off in a fully "
        "non-interactive (all-flags-given) run.",
    ),
) -> None:
    """Wires together Portfolio, Commander, and BacktestEngine for the
    given assets and date range, runs the backtest, and prints a summary
    of the resulting BacktestResult.

    Run with every argument given (assets + --start-date + --end-date) for a
    fully non-interactive, scriptable invocation — nothing is prompted, and a
    malformed date fails fast with a clear error. Omit any of the three and
    this drops into an interactive prompt for whatever's missing, including a
    picker over the curated ticker catalog (preset baskets, browse-by-asset-
    class, or free-form entry).

    Examples:

        # Fully interactive — prompts for everything
        uv run agentic-trading-cli

        # Assets given, prompts only for dates
        uv run agentic-trading-cli AAPL MSFT GOOGL

        # Fully non-interactive, e.g. for scripting/CI
        uv run agentic-trading-cli AAPL MSFT GOOGL \\
            --start-date 2023-01-01 --end-date 2023-12-31
    """
    # True whenever anything essential wasn't supplied on the command line —
    # this is also what gates the portfolio-construction prompt, so a fully
    # scripted invocation that omits --portfolio-construction gets the old
    # silent-default-False behavior rather than an unexpected prompt.
    interactive = not assets or start_date is None or end_date is None

    if interactive:
        typer.echo("Some run parameters weren't given on the command line — let's fill them in.")

    resolved_assets = (
        [symbol.strip().upper() for symbol in assets] if assets else _prompt_for_assets()
    )

    parsed_start_date = (
        _parse_date_flag(start_date, "--start-date")
        if start_date is not None
        else _prompt_for_date("Start date")
    )
    parsed_end_date = (
        _parse_date_flag(end_date, "--end-date")
        if end_date is not None
        else _prompt_for_date("End date", after=parsed_start_date)
    )
    if parsed_end_date <= parsed_start_date:
        # Only reachable when both dates came in via flags (the interactive
        # prompt already enforces this itself via `after=`).
        raise typer.BadParameter(
            f"--end-date ({parsed_end_date.isoformat()}) must be after "
            f"--start-date ({parsed_start_date.isoformat()})."
        )

    if portfolio_construction is None:
        portfolio_construction = _prompt_for_portfolio_construction() if interactive else False

    typer.echo()
    typer.echo(f"  Assets:                 {', '.join(resolved_assets)}")
    typer.echo(
        f"  Date range:             {parsed_start_date.isoformat()} → {parsed_end_date.isoformat()}"
    )
    typer.echo(f"  Portfolio construction: {'on' if portfolio_construction else 'off'}")
    typer.echo()

    configure_logging(verbose=verbose, disable=no_log)
    logger = get_logger(__name__)

    logger.info(
        "CLI invoked: assets=%s, start_date=%s, end_date=%s, portfolio_construction=%s",
        resolved_assets,
        parsed_start_date,
        parsed_end_date,
        portfolio_construction,
    )
    analysts = build_default_analysts()
    commander = Commander(
        managers=build_default_managers(analysts),
        weighting_strategy=Hedge(),
    )
    # Pre-fetched once (with the same synthetic-data safety net the GUI
    # uses — load_price_history_with_fallback) rather than passed to
    # BacktestEngine as a lazy callable: this way the CLI can log which
    # source_label ("live yfinance data" vs "synthetic fallback data") the
    # run actually used, and the engine's data_loader stays a trivial
    # pass-through.
    market_data, source_label = load_price_history_with_fallback(
        resolved_assets, parsed_start_date, parsed_end_date
    )
    logger.info("Market data source: %s", source_label)
    engine = BacktestEngine(
        commander=commander,
        portfolio=Portfolio(cash=10_000.0, positions={}),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: market_data,
        start_date=parsed_start_date,
        end_date=parsed_end_date,
        assets=resolved_assets,
        portfolio_constructor=PortfolioConstructor(enabled=True)
        if portfolio_construction
        else None,
    )
    try:
        result = engine.run()
    except Exception:
        logger.exception(
            "CLI run failed for assets=%s (%s to %s)",
            resolved_assets,
            parsed_start_date,
            parsed_end_date,
        )
        raise
    typer.echo(result.model_dump_json(indent=2))


def main(args: list[str] | None = None) -> None:
    app(args=args)
