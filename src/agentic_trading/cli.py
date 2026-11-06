# cli.py

import json
from datetime import date, datetime, timedelta
from importlib import metadata
from pathlib import Path
from typing import cast, get_args

import typer

from agentic_trading import provenance
from agentic_trading.agents.base import AnalystAgent
from agentic_trading.agents.commander import Commander
from agentic_trading.agents.managers.cross_asset_manager import CRYPTO_ASSETS
from agentic_trading.agents.roster import (
    FIRM_V1_EQUITY_SECURITY_IDS,
    FIRM_V1_EXCLUDED_ANALYST_IDS,
    FIRM_V1_UNIVERSE,
    FIRM_V1_WINDOW_END,
    FIRM_V1_WINDOW_START,
    build_default_analysts,
    build_default_managers,
)
from agentic_trading.agents.schemas import BacktestResult
from agentic_trading.backtest import checkpoint_store
from agentic_trading.backtest.baselines import BaselineCommander, SingleAnalystCommander
from agentic_trading.backtest.engine import BacktestEngine
from agentic_trading.backtest.portfolio import Portfolio
from agentic_trading.backtest.preflight import check_data_availability, check_price_coverage
from agentic_trading.config import apply_overrides, parse_override, settings
from agentic_trading.data.loader import (
    DEFAULT_BACKTEST_WARMUP_DAYS,
    load_price_history_with_fallback,
)
from agentic_trading.data.loaders.lakehouse_loader import (
    declare_loader_window,
    lake_session,
    loader_query_counts,
    loader_store_usage,
    reset_loader_query_counts,
    verify_rebuilt_whole_builds,
)
from agentic_trading.data.loaders.macro_loader import (
    load_fx_policy_rates,
    load_risk_free_rate_series,
)

# NOTE: adjust this import to wherever ticker_catalog.py actually lives in the
# repo (it reads as a data/ module from its own docstring — "data/loader.py"
# is its neighbor above — but confirm the real path before running this).
from agentic_trading.data.ticker_catalog import (
    TICKER_GROUPS,
    AssetClass,
    TickerGroup,
    tickers_by_asset_class,
)
from agentic_trading.firm_record.runner import run_firm_record
from agentic_trading.logging_config import configure_logging, get_logger
from agentic_trading.provenance import build_run_provenance
from agentic_trading.scoring.track_record import TrackRecord
from agentic_trading.scoring.weighting import Hedge, ShrinkageWeight
from agentic_trading.stage_a.runner import EXCLUDED_ANALYSTS, run_batch
from agentic_trading.stage_a.store import InstrumentKind
from agentic_trading.trials.ledger import TrialLedger

app = typer.Typer()
experiment_app = typer.Typer()
stage_a_app = typer.Typer()
firm_record_app = typer.Typer()

# Every asset class CATALOG actually classifies into (see ticker_catalog.py's
# AssetClass) — options are deliberately excluded there, so they're
# excluded here too, not an oversight.
_ASSET_CLASSES: tuple[AssetClass, ...] = ("stock", "index", "commodity", "forex", "crypto")

_DATE_FORMAT_HINT = "YYYY-MM-DD"

# The market benchmark `experiment()` computes final_information_ratio
# against — independent of whatever assets the run itself trades.
SP500_BENCHMARK_TICKER = "^GSPC"
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


_LAKE_TIME_EXAMPLE = "2024-06-01T12:00:00+00:00"
_LAKE_TIME_HELP = (
    "Lake time T every lakehouse read sees, as ISO 8601 with a UTC offset "
    f"(e.g. {_LAKE_TIME_EXAMPLE} or 2024-06-01T12:00:00Z). Rerun at an earlier run's "
    "recorded lake_time to read the same lake state. Default: 1 s after the newest "
    "write of the tables the run reads."
)

_LOADER_STORE_HELP = (
    "Serve the loaders' per-day reads from data read once over the run's window "
    "(the window store); --no-loader-store makes every read query the lake itself. "
    "Both give the same result."
)

# Prefix of the stderr line holding the run's loader query counts as JSON.
LOADER_QUERY_COUNTS_PREFIX = "loader-query-counts "
# Prefix of the stderr line holding the window store's memory as JSON.
LOADER_STORE_USAGE_PREFIX = "loader-store-usage "


def declare_run_window(enabled: bool, start: date, end: date, warmup_days: int) -> None:
    """Resets the loader query counts and, when `enabled`, declares the
    open session's window store for a run from `start` to `end` with
    `warmup_days` calendar days of warm-up: from `start - warmup_days`
    to `end`."""
    reset_loader_query_counts()
    if enabled:
        declare_loader_window(start - timedelta(days=warmup_days), end)


def echo_loader_query_counts() -> None:
    """Writes one stderr line: `LOADER_QUERY_COUNTS_PREFIX` and the JSON
    list of `loader_query_counts()`."""
    counts = [count.model_dump() for count in loader_query_counts()]
    typer.echo(LOADER_QUERY_COUNTS_PREFIX + json.dumps(counts), err=True)


def echo_loader_store_usage() -> None:
    """Writes one stderr line: `LOADER_STORE_USAGE_PREFIX` and the JSON of
    `loader_store_usage()` (`null` when no window store is declared)."""
    usage = loader_store_usage()
    text = "null" if usage is None else usage.model_dump_json()
    typer.echo(LOADER_STORE_USAGE_PREFIX + text, err=True)


def _parse_lake_time_flag(value: str | None) -> datetime | None:
    """The `--lake-time` value as an aware `datetime`, or `None` when the
    flag was not given. A value that is not ISO 8601, or that has no UTC
    offset, is a usage error."""
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise typer.BadParameter(
            f"--lake-time must be ISO 8601 with a UTC offset (e.g. {_LAKE_TIME_EXAMPLE}), "
            f"got {value!r}."
        ) from exc
    if parsed.utcoffset() is None:
        raise typer.BadParameter(
            f"--lake-time {value!r} has no UTC offset; add one, e.g. {_LAKE_TIME_EXAMPLE}."
        )
    return parsed


def _checkpoint_lake_time(checkpoint_paths: list[str | None]) -> datetime | None:
    """The lake time stored in the given checkpoint files (missing paths
    and files without one are skipped), or `None` when none stores one.
    Two checkpoints storing different lake times are a usage error: one
    experiment reads at one lake time."""
    stored = {
        lake_time
        for path in checkpoint_paths
        if path is not None
        and (lake_time := checkpoint_store.read_lake_time(Path(path))) is not None
    }
    if len(stored) > 1:
        listed = ", ".join(sorted(lake_time.isoformat() for lake_time in stored))
        raise typer.BadParameter(
            f"the checkpoints were written at different lake times ({listed}); "
            "resume them separately or pass --lake-time"
        )
    return next(iter(stored), None)


def resolve_roster_after_preflight(
    analysts: list[AnalystAgent],
    assets: list[str],
    start_date: date,
    end_date: date,
    *,
    warmup_days: int,
    auto_confirm: bool,
) -> list[AnalystAgent]:
    """Runs `check_data_availability` against the final roster/ticker
    list over the run window plus its `warmup_days` of history, prints a
    report for anything that can't run fully with real lakehouse data,
    and — unless `auto_confirm` — asks once (a single
    batched prompt covering every failing analyst, not one per analyst)
    whether to drop them and continue. Exits the process on a decline,
    or on any failure at all when `auto_confirm` is False and the run
    isn't interactive enough to ask (Typer's `confirm` itself handles
    the actual prompting)."""
    report = check_data_availability(
        analysts, assets, start_date, end_date, warmup_days=warmup_days
    )
    if not report.failing_analyst_ids:
        return analysts

    typer.echo()
    for analyst_id in sorted(report.failing_analyst_ids):
        typer.echo(f"  ⚠ {report.failure_summary(analyst_id)}", err=True)
    typer.echo()

    if not auto_confirm and not typer.confirm(
        f"{len(report.failing_analyst_ids)} analyst(s) cannot run fully with real data. "
        "Drop them and continue?",
        default=False,
    ):
        typer.echo("Aborting — no real-data-backed roster to run.", err=True)
        raise typer.Exit(code=1)

    return [analyst for analyst in analysts if analyst.agent_id in report.passing_analyst_ids]


# --------------------------------------------------------------------------
# Command
# --------------------------------------------------------------------------


# Settings read once at import (`firm.max_leverage` bounds the Pydantic
# schemas in `agents/schemas.py`), so a `--set` for them would be recorded
# in provenance without changing what the run trades on.
_IMPORT_TIME_SETTINGS = frozenset({"firm.max_leverage"})


def _apply_cli_overrides(tokens: list[str]) -> dict[str, str]:
    """Parse repeated `--set group.field=value` tokens and apply them to
    `config.settings` before any analyst/manager is built; a bad key,
    an out-of-range value, or a key in `_IMPORT_TIME_SETTINGS` is a usage
    error, not a silent default."""
    overrides: dict[str, str] = {}
    for token in tokens:
        try:
            key, value = parse_override(token)
            if key in _IMPORT_TIME_SETTINGS:
                raise ValueError(
                    f"{key} is read at import and cannot be overridden with --set; "
                    "change its default in config.py instead"
                )
            apply_overrides({key: value})
        except (KeyError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        overrides[key] = value
    return overrides


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
    parallel_workers: int = typer.Option(
        4, min=1, help="Concurrent manager-signal workers used within each simulated day."
    ),
    set_override: list[str] = typer.Option(  # noqa: B008
        [],
        "--set",
        help="Override one strategy constant for this run, as group.field=value "
        "(e.g. commander.drawdown_trip_threshold=-0.15, "
        "pod_sizing.stat_arb_manager.leverage_cap=4.0); repeat for several. "
        "See agentic_trading/config.py for every key.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Auto-drop analysts that fail the real-data preflight check instead of "
        "prompting — required for scripted/CI use.",
    ),
    lake_time: str | None = typer.Option(None, "--lake-time", help=_LAKE_TIME_HELP),
    loader_store: bool = typer.Option(
        True, "--loader-store/--no-loader-store", help=_LOADER_STORE_HELP
    ),
) -> None:
    """Wires together Portfolio, Commander, and BacktestEngine for the
    given assets and date range, runs the backtest, and prints a summary
    of the resulting BacktestResult. Every lakehouse read (the preflight,
    the price history, the engine run and the provenance) happens inside
    one lake session, at `--lake-time` or the default lake time, and the
    provenance records it. Unless `--no-loader-store`, the session's
    window store is declared from the start date minus the warm-up to the
    end date. The loader query counts and the window store's memory go to
    stderr as one `LOADER_QUERY_COUNTS_PREFIX` and one
    `LOADER_STORE_USAGE_PREFIX` line.

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
    requested_lake_time = _parse_lake_time_flag(lake_time)
    # True whenever anything essential wasn't supplied on the command line —
    # this is also what gates the portfolio-construction prompt, so a fully
    # scripted invocation that omits --portfolio-construction silently
    # defaults to False rather than prompting.
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

    typer.echo()
    typer.echo(f"  Assets:                 {', '.join(resolved_assets)}")
    typer.echo(
        f"  Date range:             {parsed_start_date.isoformat()} → {parsed_end_date.isoformat()}"
    )
    typer.echo()

    configure_logging(verbose=verbose, disable=no_log)
    _apply_cli_overrides(set_override)
    logger = get_logger(__name__)

    logger.info(
        "CLI invoked: assets=%s, start_date=%s, end_date=%s",
        resolved_assets,
        parsed_start_date,
        parsed_end_date,
    )
    with lake_session(requested_lake_time) as session:
        logger.info("Reading the lake at lake time %s", session.lake_time.isoformat())
        declare_run_window(
            loader_store, parsed_start_date, parsed_end_date, DEFAULT_BACKTEST_WARMUP_DAYS
        )
        analysts = resolve_roster_after_preflight(
            build_default_analysts(),
            resolved_assets,
            parsed_start_date,
            parsed_end_date,
            warmup_days=DEFAULT_BACKTEST_WARMUP_DAYS,
            auto_confirm=yes,
        )
        managers = build_default_managers(analysts)
        provenance = build_run_provenance(
            {
                "assets": resolved_assets,
                "start_date": parsed_start_date.isoformat(),
                "end_date": parsed_end_date.isoformat(),
                "parallel_workers": parallel_workers,
                "set": list(set_override),
                "yes": yes,
                "verbose": verbose,
                "no_log": no_log,
                "lake_time": lake_time,
                "loader_store": loader_store,
            },
            [analyst.agent_id for analyst in analysts],
            [manager.agent_id for manager in managers],
        )
        commander = Commander(
            managers=managers,
            weighting_strategy=ShrinkageWeight(Hedge()),
            parallel_workers=parallel_workers,
        )
        # Pre-fetched once, lakehouse-only, rather than passed to
        # BacktestEngine as a lazy callable: this way the CLI can log the
        # source_label the run actually used, and the engine's data_loader
        # stays a trivial pass-through.
        market_data, source_label = load_price_history_with_fallback(
            resolved_assets,
            parsed_start_date,
            parsed_end_date,
            warmup_days=DEFAULT_BACKTEST_WARMUP_DAYS,
            align="intersection",
        )
        logger.info("Market data source: %s", source_label)
        engine = BacktestEngine(
            commander=commander,
            portfolio=Portfolio(cash=settings.firm.starting_cash, positions={}),
            track_record=TrackRecord(),
            data_loader=lambda _assets, _start, _end: market_data,
            start_date=parsed_start_date,
            end_date=parsed_end_date,
            assets=resolved_assets,
            market_data_source=source_label,
            execution_lag_days=settings.firm.execution_lag_days,
            risk_free_rate_series=load_risk_free_rate_series(parsed_start_date, parsed_end_date),
            policy_rates=load_fx_policy_rates(resolved_assets, parsed_start_date, parsed_end_date),
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
        verify_rebuilt_whole_builds()
        echo_loader_query_counts()
        echo_loader_store_usage()
        typer.echo(
            json.dumps(
                {
                    "provenance": provenance.model_dump(mode="json"),
                    "result": result.model_dump(mode="json"),
                },
                indent=2,
            )
        )


@experiment_app.command()
def experiment(
    assets: list[str] = typer.Argument(..., help="Asset symbols for the experiment."),  # noqa: B008
    train_start: str = typer.Option(..., help="Calibration period start, YYYY-MM-DD."),
    train_end: str = typer.Option(..., help="Calibration period end, YYYY-MM-DD."),
    test_start: str = typer.Option(..., help="Out-of-sample test period start, YYYY-MM-DD."),
    test_end: str = typer.Option(..., help="Out-of-sample test period end, YYYY-MM-DD."),
    warmup_days: int = typer.Option(
        DEFAULT_BACKTEST_WARMUP_DAYS,
        min=0,
        help="Extra historical days loaded before the training period.",
    ),
    parallel_workers: int = typer.Option(
        4, min=1, help="Concurrent manager-signal workers per simulated day."
    ),
    commission_rate: float | None = typer.Option(
        None,
        min=0.0,
        help="Commission as a fraction of traded notional "
        "(default: firm.commission_rate, after --set).",
    ),
    slippage_bps: float | None = typer.Option(
        None,
        min=0.0,
        help="Execution slippage in basis points per trade "
        "(default: firm.slippage_bps, after --set).",
    ),
    execution_lag_days: int | None = typer.Option(
        None,
        min=0,
        max=1,
        help="Bars between signal formation and execution "
        "(default: firm.execution_lag_days, after --set).",
    ),
    calibration_checkpoint: str | None = typer.Option(
        None, help="JSON checkpoint path for calibration; add --resume to continue it."
    ),
    test_checkpoint: str | None = typer.Option(
        None, help="JSON checkpoint path for out-of-sample testing; add --resume to continue it."
    ),
    resume: bool = typer.Option(False, help="Resume from the supplied checkpoint files."),
    include_baselines: bool = typer.Option(
        True, help="Include simple benchmark strategies in the test report."
    ),
    include_isolated: bool = typer.Option(
        False,
        help="Run every analyst and manager as an isolated test strategy; "
        "slower but more diagnostic.",
    ),
    exclude_analyst: list[str] = typer.Option(  # noqa: B008
        [],
        "--exclude-analyst",
        help="Analyst id to omit; repeat the option for multiple analysts.",
    ),
    set_override: list[str] = typer.Option(  # noqa: B008
        [],
        "--set",
        help="Override one strategy constant for this run, as group.field=value "
        "(e.g. commander.drawdown_trip_threshold=-0.15, "
        "pod_sizing.stat_arb_manager.leverage_cap=4.0); repeat for several. "
        "See agentic_trading/config.py for every key.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Auto-drop analysts that fail the real-data preflight check instead of "
        "prompting — required for scripted/CI use.",
    ),
    lake_time: str | None = typer.Option(
        None,
        "--lake-time",
        help=_LAKE_TIME_HELP + " With --resume and no --lake-time, the lake time stored "
        "in the checkpoints.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Log detailed per-step timing (log-file only)."
    ),
    no_log: bool = typer.Option(
        False, "--no-log", help="Disable logging entirely — nothing is written to the log file."
    ),
    loader_store: bool = typer.Option(
        True, "--loader-store/--no-loader-store", help=_LOADER_STORE_HELP
    ),
) -> None:
    """Run a calibration phase followed by a separate out-of-sample test phase.

    The calibration phase establishes analyst and manager trust weights; it does
    not train a machine-learning model.
    The test phase starts with fresh portfolio holdings but carries those
    learned weights forward. Both phases use strict real market data.

    Both phases, the baselines and the provenance read the lake inside one
    lake session: at `--lake-time` when given; otherwise, with `--resume`,
    at the lake time stored in the checkpoints; otherwise at the default
    lake time. Unless `--no-loader-store`, the session's window store is
    declared from the training start minus `--warmup-days` to the test
    end.

    The JSON result is the only thing on stdout; progress messages and
    one `LOADER_QUERY_COUNTS_PREFIX` line of loader query counts and one
    `LOADER_STORE_USAGE_PREFIX` line of window-store memory go to
    stderr, and the run log to `log/agentic_trading.log` (none with
    `--no-log`).
    """
    parsed_train_start = _parse_date_flag(train_start, "--train-start")
    parsed_train_end = _parse_date_flag(train_end, "--train-end")
    parsed_test_start = _parse_date_flag(test_start, "--test-start")
    parsed_test_end = _parse_date_flag(test_end, "--test-end")
    if parsed_train_end >= parsed_test_start:
        raise typer.BadParameter("--train-end must be before --test-start")
    if parsed_test_end <= parsed_test_start:
        raise typer.BadParameter("--test-end must be after --test-start")
    requested_lake_time = _parse_lake_time_flag(lake_time)
    if requested_lake_time is None and resume:
        requested_lake_time = _checkpoint_lake_time([calibration_checkpoint, test_checkpoint])

    resolved_assets = [symbol.strip().upper() for symbol in assets]
    configure_logging(verbose=verbose, disable=no_log, console=False)
    overrides = _apply_cli_overrides(set_override)
    # An explicit flag wins; otherwise the firm setting as `--set` left it.
    resolved_commission_rate = (
        settings.firm.commission_rate if commission_rate is None else commission_rate
    )
    resolved_slippage_bps = settings.firm.slippage_bps if slippage_bps is None else slippage_bps
    resolved_execution_lag_days = (
        settings.firm.execution_lag_days if execution_lag_days is None else execution_lag_days
    )
    # The resolved values go back onto `settings.firm`, so the no-trade band
    # and the provenance read the same costs the fills pay.
    apply_overrides(
        {
            "firm.commission_rate": resolved_commission_rate,
            "firm.slippage_bps": resolved_slippage_bps,
            "firm.execution_lag_days": resolved_execution_lag_days,
        }
    )
    excluded = frozenset(analyst_id.strip() for analyst_id in exclude_analyst)
    analysts = build_default_analysts(excluded_analyst_ids=excluded)
    typer.echo(
        f"Loading real data for {len(resolved_assets)} assets; excluded analysts: "
        f"{', '.join(sorted(excluded)) or 'none'}",
        err=True,
    )
    with lake_session(requested_lake_time) as session:
        typer.echo(f"Reading the lake at lake time {session.lake_time.isoformat()}", err=True)
        declare_run_window(loader_store, parsed_train_start, parsed_test_end, warmup_days)
        analysts = resolve_roster_after_preflight(
            analysts,
            resolved_assets,
            parsed_train_start,
            parsed_test_end,
            warmup_days=warmup_days,
            auto_confirm=yes,
        )
        managers = build_default_managers(analysts)
        provenance = build_run_provenance(
            {
                "assets": resolved_assets,
                "train_start": parsed_train_start.isoformat(),
                "train_end": parsed_train_end.isoformat(),
                "test_start": parsed_test_start.isoformat(),
                "test_end": parsed_test_end.isoformat(),
                "warmup_days": warmup_days,
                "parallel_workers": parallel_workers,
                "commission_rate": resolved_commission_rate,
                "slippage_bps": resolved_slippage_bps,
                "execution_lag_days": resolved_execution_lag_days,
                "calibration_checkpoint": calibration_checkpoint,
                "test_checkpoint": test_checkpoint,
                "resume": resume,
                "include_baselines": include_baselines,
                "include_isolated": include_isolated,
                "exclude_analyst": sorted(excluded),
                "set": list(set_override),
                "yes": yes,
                "verbose": verbose,
                "no_log": no_log,
                "lake_time": lake_time,
                "loader_store": loader_store,
            },
            [analyst.agent_id for analyst in analysts],
            [manager.agent_id for manager in managers],
        )
        commander = Commander(
            managers=managers,
            weighting_strategy=ShrinkageWeight(Hedge()),
            parallel_workers=parallel_workers,
        )
        market_data, source_label = load_price_history_with_fallback(
            resolved_assets,
            parsed_train_start,
            parsed_test_end,
            warmup_days=warmup_days,
            align="intersection",
        )
        typer.echo(
            f"Market data ready: {source_label}; {len(market_data)} aligned rows. "
            "Starting calibration...",
            err=True,
        )
        track_record = TrackRecord()

        # Every Sharpe/Sortino/Information-Ratio this experiment reports (system,
        # baselines, and any --include-isolated run) shares these two inputs, so
        # they're loaded once here rather than per-engine: a real 3-month T-bill
        # rate (FRED DGS3MO from the lakehouse) for excess-return Sharpe/Sortino, and an
        # S&P 500 buy-and-hold's own daily returns as the market benchmark
        # information_ratio is computed against.
        risk_free_rate_series = load_risk_free_rate_series(parsed_train_start, parsed_test_end)
        # The BIS policy rates every FX pair's carry accrues from, shared the same way.
        policy_rates = load_fx_policy_rates(resolved_assets, parsed_train_start, parsed_test_end)
        sp500_frame, _sp500_source = load_price_history_with_fallback(
            [SP500_BENCHMARK_TICKER],
            parsed_train_start,
            parsed_test_end,
            warmup_days=0,
            align="intersection",
        )
        benchmark_returns = (
            sp500_frame[SP500_BENCHMARK_TICKER]["close"].dropna().pct_change().dropna()
        )

        def build_engine(start: date, end: date) -> BacktestEngine:
            return BacktestEngine(
                commander=commander,
                portfolio=Portfolio(
                    cash=settings.firm.starting_cash,
                    positions={},
                    commission_rate=resolved_commission_rate,
                    slippage_bps=resolved_slippage_bps,
                ),
                track_record=track_record,
                data_loader=lambda _assets, _start, _end: market_data,
                start_date=start,
                end_date=end,
                assets=resolved_assets,
                market_data_source=source_label,
                execution_lag_days=resolved_execution_lag_days,
                checkpoint_path=(
                    calibration_checkpoint if start == parsed_train_start else test_checkpoint
                ),
                resume=resume,
                risk_free_rate_series=risk_free_rate_series,
                policy_rates=policy_rates,
                benchmark_returns=benchmark_returns,
            )

        def run_phase(label: str, start: date, end: date) -> BacktestResult:
            def on_day_complete(
                day_index: int, total_days: int, as_of_date: date, value: float
            ) -> None:
                typer.echo(
                    f"{label}: day {day_index + 1}/{total_days} ({as_of_date}) "
                    f"portfolio=${value:,.2f}",
                    err=True,
                )

            result = build_engine(start, end).run(on_day_complete=on_day_complete)
            typer.echo(f"{label} complete. Starting next phase...", err=True)
            return result

        calibration_result = run_phase("CALIBRATION", parsed_train_start, parsed_train_end)
        test_result = run_phase("TEST", parsed_test_start, parsed_test_end)
        baseline_results: dict[str, dict[str, object]] = {}
        if include_baselines:
            for strategy in ("equal_weight", "buy_and_hold", "momentum", "volatility_target"):
                baseline_engine = BacktestEngine(
                    commander=BaselineCommander(strategy),
                    portfolio=Portfolio(
                        cash=settings.firm.starting_cash,
                        positions={},
                        commission_rate=resolved_commission_rate,
                        slippage_bps=resolved_slippage_bps,
                    ),
                    track_record=TrackRecord(),
                    data_loader=lambda _assets, _start, _end: market_data,
                    start_date=parsed_test_start,
                    end_date=parsed_test_end,
                    assets=resolved_assets,
                    market_data_source=source_label,
                    execution_lag_days=resolved_execution_lag_days,
                    risk_free_rate_series=risk_free_rate_series,
                    policy_rates=policy_rates,
                    benchmark_returns=benchmark_returns,
                )
                baseline_results[strategy] = baseline_engine.run().model_dump(mode="json")
        isolated_results: dict[str, dict[str, object]] = {}
        if include_isolated:
            for analyst in analysts:
                isolated_engine = BacktestEngine(
                    commander=SingleAnalystCommander(analyst),
                    portfolio=Portfolio(
                        cash=settings.firm.starting_cash,
                        positions={},
                        commission_rate=resolved_commission_rate,
                        slippage_bps=resolved_slippage_bps,
                    ),
                    track_record=TrackRecord(),
                    data_loader=lambda _assets, _start, _end: market_data,
                    start_date=parsed_test_start,
                    end_date=parsed_test_end,
                    assets=resolved_assets,
                    market_data_source=source_label,
                    execution_lag_days=resolved_execution_lag_days,
                    risk_free_rate_series=risk_free_rate_series,
                    policy_rates=policy_rates,
                    benchmark_returns=benchmark_returns,
                )
                isolated_results[f"analyst:{analyst.agent_id}"] = isolated_engine.run().model_dump(
                    mode="json"
                )
            for manager in build_default_managers(analysts):
                isolated_engine = BacktestEngine(
                    commander=Commander(
                        managers=[manager],
                        weighting_strategy=ShrinkageWeight(Hedge()),
                        parallel_workers=parallel_workers,
                    ),
                    portfolio=Portfolio(
                        cash=settings.firm.starting_cash,
                        positions={},
                        commission_rate=resolved_commission_rate,
                        slippage_bps=resolved_slippage_bps,
                    ),
                    track_record=TrackRecord(),
                    data_loader=lambda _assets, _start, _end: market_data,
                    start_date=parsed_test_start,
                    end_date=parsed_test_end,
                    assets=resolved_assets,
                    market_data_source=source_label,
                    execution_lag_days=resolved_execution_lag_days,
                    risk_free_rate_series=risk_free_rate_series,
                    policy_rates=policy_rates,
                    benchmark_returns=benchmark_returns,
                )
                isolated_results[f"manager:{manager.agent_id}"] = isolated_engine.run().model_dump(
                    mode="json"
                )
        verify_rebuilt_whole_builds()
        echo_loader_query_counts()
        echo_loader_store_usage()
        typer.echo(
            json.dumps(
                {
                    "provenance": provenance.model_dump(mode="json"),
                    "experiment": {
                        "assets": resolved_assets,
                        "train_start": parsed_train_start.isoformat(),
                        "train_end": parsed_train_end.isoformat(),
                        "test_start": parsed_test_start.isoformat(),
                        "test_end": parsed_test_end.isoformat(),
                        "warmup_days": warmup_days,
                        "excluded_analysts": sorted(excluded),
                        "config_overrides": overrides,
                    },
                    "market_data_source": source_label,
                    "calibration": calibration_result.model_dump(mode="json"),
                    "test": test_result.model_dump(mode="json"),
                    "baselines": baseline_results,
                    "isolated": isolated_results,
                },
                indent=2,
            )
        )


# The repository root; a stage (a) store must lie outside it, and the trial
# ledger's relative paths are resolved against it.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Exit code of `agentic-trading-stage-a` when the batch stopped for resources.
_EXIT_STOPPED_FOR_RESOURCES = 3


def _stage_a_analysts(agent_ids: list[str]) -> list[AnalystAgent]:
    """The default-roster analysts with `agent_ids`, in that order. An id
    the roster does not hold, an id in `stage_a.runner.EXCLUDED_ANALYSTS`
    (reported with its reason), or an id given twice, is a usage error."""
    roster = {analyst.agent_id: analyst for analyst in build_default_analysts()}
    unknown = [agent_id for agent_id in agent_ids if agent_id not in roster]
    if unknown:
        raise typer.BadParameter(
            f"no analyst in the default roster has id(s) {unknown}", param_hint="--analyst"
        )
    excluded = [agent_id for agent_id in agent_ids if agent_id in EXCLUDED_ANALYSTS]
    if excluded:
        raise typer.BadParameter(
            "excluded from stage (a): "
            + "; ".join(f"{agent_id} {EXCLUDED_ANALYSTS[agent_id]}" for agent_id in excluded),
            param_hint="--analyst",
        )
    repeated = sorted({agent_id for agent_id in agent_ids if agent_ids.count(agent_id) > 1})
    if repeated:
        raise typer.BadParameter(f"analyst id(s) given twice: {repeated}", param_hint="--analyst")
    return [roster[agent_id] for agent_id in agent_ids]


def _stage_a_instrument_kind(value: str) -> InstrumentKind:
    """`value` as a `decision_file.v1` instrument kind; any other value is a
    usage error."""
    kinds: tuple[str, ...] = get_args(InstrumentKind)
    if value not in kinds:
        raise typer.BadParameter(
            f"must be one of {', '.join(kinds)}, got {value!r}", param_hint="--instrument-kind"
        )
    return cast(InstrumentKind, value)


@stage_a_app.command()
def stage_a(
    analyst: list[str] = typer.Option(  # noqa: B008
        ...,
        "--analyst",
        help="Default-roster analyst id to run; repeat for several, run in the order given.",
    ),
    universe: list[str] = typer.Option(  # noqa: B008
        ..., "--universe", help="Instrument id (ticker) to run over; repeat for several."
    ),
    store_dir: Path = typer.Option(  # noqa: B008
        ...,
        "--store-dir",
        help="Directory outside the repository holding one subdirectory per trial spec hash.",
    ),
    instrument_kind: str = typer.Option(
        "equity",
        "--instrument-kind",
        help="decision_file.v1 instrument kind every row is written as: "
        "equity, etf, future, fx, crypto or option.",
    ),
    window_start: str | None = typer.Option(
        None,
        "--window-start",
        help=f"First decision date, {_DATE_FORMAT_HINT} (default: stage_a.window_start).",
    ),
    window_end: str | None = typer.Option(
        None,
        "--window-end",
        help=f"Last decision date, {_DATE_FORMAT_HINT}, at most stage_a.data_end "
        "(default: stage_a.data_end).",
    ),
    lake_time: str | None = typer.Option(None, "--lake-time", help=_LAKE_TIME_HELP),
    set_override: list[str] = typer.Option(  # noqa: B008
        [],
        "--set",
        help="Override one strategy constant for this batch, as group.field=value "
        "(e.g. stage_a.session_calendar=XSTO); repeat for several. Hashed settings "
        "enter each trial's spec; trials.ledger_path moves the trial ledger. See "
        "agentic_trading/config.py for every key.",
    ),
) -> None:
    """Compute and store stage (a) signals and decisions for the given
    analysts over the universe, resuming each trial from its store, and
    record each batch in the trial ledger.

    Loads the universe's prices from the window start minus
    `stage_a.warmup_days` calendar days to the window end, then runs `stage_a.runner.run_batch`, all
    inside one lake session cut at `stage_a.data_end`, so the prices and
    every analyst read see the same lake time. The spec records that
    lake time, the lake version and the installed quantcore version and
    commit. Each analyst's hypothesis is registered in the trial ledger at
    `trials.ledger_path` (relative to the repository root unless absolute)
    before its first day is computed, and each analyst that wrote a day
    records a trial row there, whose spec holds the batch's window. Prints
    the batch result as JSON. Exits 0 when every analyst finished, 3 when
    the batch stopped for low memory or disk (rerun to resume), and 2 on a
    usage error: an unknown analyst id or one excluded from stage (a)
    (`stage_a.runner.EXCLUDED_ANALYSTS`), a missing
    --store-dir or one inside the repository, a naive --lake-time, a bad
    --set, or a window that is reversed or ends after stage_a.data_end.

    `--set` overrides apply before any setting is read. Prices are aligned
    on the sessions of `stage_a.session_calendar`, the calendar the batch
    decides on, so an instrument without a bar on a session is stored as
    unavailable that day rather than removing the day for every instrument.
    """
    _apply_cli_overrides(set_override)
    requested_lake_time = _parse_lake_time_flag(lake_time)
    kind = _stage_a_instrument_kind(instrument_kind)
    data_end = settings.stage_a.data_end
    start = (
        settings.stage_a.window_start
        if window_start is None
        else _parse_date_flag(window_start, "--window-start")
    )
    end = data_end if window_end is None else _parse_date_flag(window_end, "--window-end")
    if start > end:
        raise typer.BadParameter(
            f"--window-start ({start.isoformat()}) is after --window-end ({end.isoformat()})."
        )
    if end > data_end:
        raise typer.BadParameter(
            f"--window-end ({end.isoformat()}) is after stage_a.data_end "
            f"({data_end.isoformat()}); the holdout stays locked."
        )
    if store_dir.resolve().is_relative_to(_REPO_ROOT):
        raise typer.BadParameter(
            f"{store_dir} is inside the repository; give a directory outside it.",
            param_hint="--store-dir",
        )
    analysts = _stage_a_analysts([agent_id.strip() for agent_id in analyst])
    tickers = [ticker.strip() for ticker in universe]

    configure_logging()
    logger = get_logger(__name__)
    logger.info(
        "Stage (a) batch: analysts=%s, universe=%s, %s to %s, store %s",
        [agent.agent_id for agent in analysts],
        tickers,
        start,
        end,
        store_dir,
    )
    with lake_session(requested_lake_time, data_end=data_end) as session:
        typer.echo(f"Reading the lake at lake time {session.lake_time.isoformat()}", err=True)
        market_data, source_label = load_price_history_with_fallback(
            tickers,
            start,
            end,
            warmup_days=settings.stage_a.warmup_days,
            align="sessions",
            calendar=settings.stage_a.session_calendar,
        )
        typer.echo(f"Market data ready: {source_label}; {len(market_data)} rows.", err=True)
        lake_version = (
            session.lake_version
            if session.lake_version is not None
            else provenance.lake_provenance().lake_version
        )
        result = run_batch(
            analysts,
            tickers,
            kind,
            start,
            end,
            store_dir,
            market_data,
            lake_time=session.lake_time,
            lake_version=lake_version,
            quantcore_version=metadata.version("quantcore"),
            ledger=TrialLedger(_REPO_ROOT / settings.trials.ledger_path, _REPO_ROOT),
            repo_root=_REPO_ROOT,
        )
    typer.echo(result.model_dump_json(indent=2))
    if result.stopped_for_resources:
        raise typer.Exit(code=_EXIT_STOPPED_FOR_RESOURCES)


@firm_record_app.command()
def firm_record(
    start: str = typer.Option(
        ...,
        "--start",
        help=f"First decision date, {_DATE_FORMAT_HINT}; the registered window starts "
        f"{FIRM_V1_WINDOW_START.isoformat()}.",
    ),
    end: str = typer.Option(
        ...,
        "--end",
        help=f"Last decision date, {_DATE_FORMAT_HINT}, at most stage_a.data_end; the "
        f"registered window ends {FIRM_V1_WINDOW_END.isoformat()}.",
    ),
    store_dir: Path = typer.Option(  # noqa: B008
        ...,
        "--store-dir",
        help="Directory outside the repository holding one subdirectory per firm spec hash.",
    ),
    max_days: int | None = typer.Option(
        None,
        "--max-days",
        min=1,
        help="Write at most this many days in this call; rerun to resume (default: all).",
    ),
    lake_time: str | None = typer.Option(None, "--lake-time", help=_LAKE_TIME_HELP),
    parallel_workers: int = typer.Option(
        4,
        min=1,
        help="Concurrent manager-signal workers per decision day; not part of the spec.",
    ),
    set_override: list[str] = typer.Option(  # noqa: B008
        [],
        "--set",
        help="Override one strategy constant for this record, as group.field=value; repeat "
        "for several. Hashed settings enter the firm's spec; trials.ledger_path moves the "
        "trial ledger. See agentic_trading/config.py for every key.",
    ),
    loader_store: bool = typer.Option(
        True, "--loader-store/--no-loader-store", help=_LOADER_STORE_HELP
    ),
) -> None:
    """Record the registered firm (agentic-firm-v1) from --start to --end:
    the Commander's daily pod targets written as a `decision_file.v1` trial
    under --store-dir, resuming from its store.

    The firm is the default roster without `roster.FIRM_V1_EXCLUDED_ANALYST_IDS`
    and the default managers under a Commander weighting pods by
    `ShrinkageWeight(Hedge())`, over `roster.FIRM_V1_UNIVERSE`. Loads the
    universe's prices aligned on `stage_a.session_calendar`, from --start
    minus `stage_a.warmup_days` calendar days to --end, each of
    `roster.FIRM_V1_EQUITIES` by its `roster.FIRM_V1_EQUITY_SECURITY_IDS`
    id from `equity_daily_from_bars` and every other instrument by ticker.
    Before anything is written, checks the universe's price coverage
    (`backtest.preflight.check_price_coverage`): each firm equity needs a
    bar on every session from --start to --end bar the
    `firm.known_price_gaps`, and every other instrument a bar near both
    ends of the window, the crypto pairs near its end only. It raises
    `PriceCoverageError` listing each failing instrument. An instrument
    whose bars start inside the warm-up passes, and its `ShortWarmup` goes
    to agentic_meta.json. Then loads the risk-free rate and runs
    `firm_record.runner.run_firm_record`, all inside one lake session cut
    at `stage_a.data_end`. With --loader-store (the
    default) the session's loader window store is declared over the same
    span, --start minus `stage_a.warmup_days` to --end
    (`declare_run_window`); --no-loader-store leaves every loader call
    fetching for itself. Before the session closes, the loader query
    counts and the store's usage are written to stderr
    (`echo_loader_query_counts`, `echo_loader_store_usage`). The firm's hypothesis is
    registered in the trial ledger at `trials.ledger_path` from its
    registry file before the first day; without that file the record is
    uncounted. Prints the result as JSON, which holds no performance
    figure. Exits 2 on a usage error: a reversed window, one ending after
    stage_a.data_end, a --store-dir inside the repository, a naive
    --lake-time or a bad --set. `--set` overrides apply before any setting
    is read."""
    _apply_cli_overrides(set_override)
    requested_lake_time = _parse_lake_time_flag(lake_time)
    data_end = settings.stage_a.data_end
    window_start = _parse_date_flag(start, "--start")
    window_end = _parse_date_flag(end, "--end")
    if window_start > window_end:
        raise typer.BadParameter(
            f"--start ({window_start.isoformat()}) is after --end ({window_end.isoformat()})."
        )
    if window_end > data_end:
        raise typer.BadParameter(
            f"--end ({window_end.isoformat()}) is after stage_a.data_end "
            f"({data_end.isoformat()}); the holdout stays locked."
        )
    if store_dir.resolve().is_relative_to(_REPO_ROOT):
        raise typer.BadParameter(
            f"{store_dir} is inside the repository; give a directory outside it.",
            param_hint="--store-dir",
        )
    analysts = build_default_analysts(excluded_analyst_ids=FIRM_V1_EXCLUDED_ANALYST_IDS)
    commander = Commander(
        managers=build_default_managers(analysts),
        weighting_strategy=ShrinkageWeight(Hedge()),
        parallel_workers=parallel_workers,
    )
    tickers = list(FIRM_V1_UNIVERSE)

    configure_logging()
    logger = get_logger(__name__)
    logger.info(
        "Firm record: %d instruments, %s to %s, store %s",
        len(tickers),
        window_start,
        window_end,
        store_dir,
    )
    with lake_session(requested_lake_time, data_end=data_end) as session:
        typer.echo(f"Reading the lake at lake time {session.lake_time.isoformat()}", err=True)
        declare_run_window(loader_store, window_start, window_end, settings.stage_a.warmup_days)
        market_data, source_label = load_price_history_with_fallback(
            tickers,
            window_start,
            window_end,
            warmup_days=settings.stage_a.warmup_days,
            align="sessions",
            calendar=settings.stage_a.session_calendar,
            security_ids=FIRM_V1_EQUITY_SECURITY_IDS,
        )
        typer.echo(f"Market data ready: {source_label}; {len(market_data)} rows.", err=True)
        coverage = check_price_coverage(
            market_data,
            tickers,
            window_start - timedelta(days=settings.stage_a.warmup_days),
            window_start,
            window_end,
            calendar=settings.stage_a.session_calendar,
            complete=FIRM_V1_EQUITY_SECURITY_IDS,
            known_gaps=settings.firm.known_price_gaps,
            start_exempt=CRYPTO_ASSETS,
        )
        risk_free_rate_series = load_risk_free_rate_series(window_start, window_end)
        lake_version = (
            session.lake_version
            if session.lake_version is not None
            else provenance.lake_provenance().lake_version
        )
        result = run_firm_record(
            commander,
            tickers,
            window_start,
            window_end,
            store_dir,
            market_data,
            lake_time=session.lake_time,
            lake_version=lake_version,
            quantcore_version=metadata.version("quantcore"),
            risk_free_rate_series=risk_free_rate_series,
            max_days=max_days,
            ledger=TrialLedger(_REPO_ROOT / settings.trials.ledger_path, _REPO_ROOT),
            repo_root=_REPO_ROOT,
            short_warmup=[entry.short_warmup for entry in coverage if entry.short_warmup],
        )
        echo_loader_query_counts()
        echo_loader_store_usage()
    typer.echo(result.model_dump_json(indent=2))


def main(args: list[str] | None = None) -> None:
    app(args=args)


def experiment_main(args: list[str] | None = None) -> None:
    experiment_app(args=args)


def stage_a_main(args: list[str] | None = None) -> None:
    stage_a_app(args=args)


def firm_record_main(args: list[str] | None = None) -> None:
    firm_record_app(args=args)
