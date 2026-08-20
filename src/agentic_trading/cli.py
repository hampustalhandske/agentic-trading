# cli.py

from datetime import date

import typer

from agentic_trading.agents.commander import Commander
from agentic_trading.agents.roster import build_default_analysts, build_default_managers
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.data.loader import load_price_history_with_fallback
from agentic_trading.logging_config import configure_logging, get_logger
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import Hedge

app = typer.Typer()


@app.command()
def run(
    assets: list[str] = typer.Argument(  # noqa: B008
        ..., help="Asset symbols to include in the backtest."
    ),
    start_date: str = typer.Option(..., help="Backtest start date."),
    end_date: str = typer.Option(..., help="Backtest end date."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Log detailed per-step timing (console/log-file only)."
    ),
    no_log: bool = typer.Option(
        False, "--no-log", help="Disable logging entirely — nothing to console or the log file."
    ),
) -> None:
    """Wires together Portfolio, Commander, and BacktestEngine for the
    given assets and date range, runs the backtest, and prints a summary
    of the resulting BacktestResult."""
    configure_logging(verbose=verbose, disable=no_log)
    logger = get_logger(__name__)

    logger.info("CLI invoked: assets=%s, start_date=%s, end_date=%s", assets, start_date, end_date)
    analysts = build_default_analysts()
    parsed_start_date = date.fromisoformat(start_date)
    parsed_end_date = date.fromisoformat(end_date)
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
        assets, parsed_start_date, parsed_end_date
    )
    logger.info("Market data source: %s", source_label)
    engine = BacktestEngine(
        commander=commander,
        portfolio=Portfolio(cash=10_000.0, positions={}),
        track_record=TrackRecord(),
        data_loader=lambda _assets, _start, _end: market_data,
        start_date=parsed_start_date,
        end_date=parsed_end_date,
        assets=assets,
    )
    try:
        result = engine.run()
    except Exception:
        logger.exception("CLI run failed for assets=%s (%s to %s)", assets, start_date, end_date)
        raise
    typer.echo(result.model_dump_json(indent=2))


def main(args: list[str] | None = None) -> None:
    app(args=args)
