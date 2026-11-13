# CLAUDE.md

## Project overview

Multi-agent backtesting framework for a multi-strategy fund. Consumes
`quantcore` for all numerical calculations (pricing, GARCH, VaR/CVaR,
cointegration, regressions, optimizers). This project orchestrates that
math into a decision pipeline; it does not reimplement it. If a primitive
is missing, flag it as a quantcore task and bridge it with the smallest
possible local fallback, documented as a quantcore gap in its docstring.

## Architecture (README.md has the full description)

- `agents/analysts/` — Layer 1, one subpackage per pod; every analyst
  implements `agents/base.py::AnalystAgent` and emits `AgentSignal`.
- `agents/managers/` — Layer 2; `scale_factor` (vol-regime overlay) and
  `construct_portfolio` (builds `PodView`s, runs the pod's own
  `PortfolioConstructor`).
- `agents/portfolio_construction.py` — pod sizing: Grinold alpha with
  measured IC, zero-prior Black-Litterman with Idzorek uncertainty,
  unconstrained mean-variance or view-tilted risk parity, vol targeting,
  Kelly overlay, caps, cost-aware no-trade band.
- `agents/commander.py` — Layer 3: track-record allocation across pods,
  drawdown breaker, tail-hedge budget, firm limits, exportable risk state.
- `agents/roster.py` — the default 38-analyst / 11-pod wiring; change the
  roster here.
- `agents/schemas.py` — every shared Pydantic schema.
- `config.py` — every strategy constant on one `settings` object with a
  reason per value; consumers read it at use time; `--set group.field=value`
  and `temporary_overrides` override per run. Add new constants here, never
  as module-level numbers in an analyst unless no other module shares them.
- `scoring/` — track record, measured information coefficients, weighting
  strategies; works in `datetime.date`.
- `backtest/engine.py` — deterministic fills, costs, financing,
  accounting, checkpoints. Not an agent.
- `stage_a/` — single-analyst signal batches: the frozen spec and its hash
  (`spec.py`), the decision rule and look-ahead audit of each row's
  `source`/`source_as_of` (`decisions.py`), the on-disk trial store
  (`store.py`) and the batch runner (`runner.py`). Decision days are the
  sessions of `settings.stage_a.session_calendar` (XNYS by default).
- `data/loaders/lakehouse_loader.py` — the only data path. Every read goes
  through the lakehouse's `/v1/sql` HTTP API (`LAKEHOUSE_API_URL`,
  `LAKEHOUSE_API_TOKEN`); no live vendor APIs. `lake_session` pins every
  read of a run to one lake time and can cut reads at a `data_end`.
- `data/loader.py` — price history for a run: aligned on a session
  calendar for stage (a), or the intersection of tickers' bars for the
  backtest engine (`align=`).
- `data/trading_calendar.py` — NYSE sessions for the US-source gates, and
  `sessions_between` for any exchange-calendars calendar.
- `data/point_in_time.py` — no-look-ahead slicing of price data; every
  other source gates on its own availability date inside its loader.
- `logging_config.py` — the only logging setup; `get_logger(__name__)` at
  module scope. `gui/dashboard.py` hardcodes its logger name because
  Streamlit executes it as `__main__`.

### Point-in-time gating by source

Prices gate on the bar date. SEC company facts gate on the `filed` date.
A FRED value is known from the later of its vintage date and its
observation date, and used only when that is before the decision day.
FINRA short interest gates on an
estimated dissemination date (8 NYSE sessions after settlement, then the
next session). COT gates on the release recorded in `cftc_cot_publication`
(an untimed release is usable the next day). Option chains use the
freshest snapshot on or before the decision day minus
`options.publication_lag_days`, within a 10-day window.

## Conventions

- Every cross-component data structure is a Pydantic `BaseModel`. No raw
  dicts or dataclasses across a module boundary.
- Two date types, two layers: market-data code (`_market_math.py`,
  `point_in_time.py`, `generate_signal`, `scale_factor`) uses
  `pd.Timestamp`; scoring uses `datetime.date`. Convert with `.date()`
  only in `ManagerAgent.construct_portfolio`/`evaluate` and
  `Commander.decide`.
- Signals carry explicit `status` and typed provenance (`source_as_of`,
  `sample_count`); never encode either in the rationale string. One
  meaning per status: `unavailable` when the data needed is not there,
  `active` flat at confidence 0 when the model declines to bet on data it
  has, `error` when the model fails. Build non-signals with
  `AnalystAgent.unavailable_signal` / `abstain_signal` / `error_signal`.
  A signal at confidence 0 is always flat.
- An analyst that declares a dated data source sets `source_as_of` to the
  publication date of the newest data its signal rests on.
- Keep coding style consistent with the surrounding code.
- Docstrings and comments describe current behaviour only. Never narrate
  history, prior bugs, removed code, or why something changed.

## Commands

| Task | Command |
|---|---|
| Install | `uv sync --locked --group dev` |
| Tests | `uv run pytest` |
| Lint / format | `uv run ruff check .` / `uv run ruff format .` |
| Type check | `uv run mypy src/` |
| Single-phase backtest | `uv run agentic-trading-cli <assets> --start-date <d> --end-date <d>` |
| Calibration + test experiment | `uv run agentic-trading-experiment <assets> --train-start <d> --train-end <d> --test-start <d> --test-end <d> --yes` |
| Stage (a) signal batch | `uv run agentic-trading-stage-a --analyst <id> ... --universe <ticker> ... --store-dir <dir>` |
| Override a constant | add `--set group.field=value` (repeatable) |
| Dashboard | `uv run agentic-trading` |
| Tail the run log | `tail -f log/agentic_trading.log` |

## Subagents

The owner's global agents (explorer, test-writer, implementer,
code-reviewer) apply. Explore unfamiliar areas first, write tests against
the spec before implementing, review before committing.
