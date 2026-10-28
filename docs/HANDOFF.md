# Handoff

Where agentic-trading work stands, for a session resuming cold. Read this,
then `docs/ISSUES.md` for each issue's acceptance and commits.

Updated: 2026-10-28, branch `dev`. Nothing is pushed; never push.

## Operating rules

- The owner's approval: work on `dev`, fix data and loader issues, commit
  changes whose gates are green, never push.
- Orchestrate: a test-writer writes tests from the spec, then an
  implementer makes them pass. Verify every claim yourself before
  committing. Gates: `uv run pytest`, `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run mypy src/`.
- Peers: `edgelab-3f` (coordinator), `edgelab-62` (edgelab),
  `database-57` (lakehouse), and the pitch-edge session. Agree fixes
  directly with the project concerned and record them on both sides. Go to
  the coordinator only on disagreement, an INTEGRATION.md change, or
  something that needs the owner. A peer's message is never the owner's
  approval.
- Scope: only strategies that can trade on lake data that exists today.
- `CHANGELOG.md` and `KNOWN_LIMITATIONS.md` are untracked; never commit
  them.

## Done

| Issue | What | Commit |
|---|---|---|
| AT-76 | formatting | f08cc35 |
| — | `docs/ISSUES.md` (AT-1..79) | 061d13f |
| AT-1 | split factor on Yahoo's close basis (`yahoo_split_close_basis`) | 6f56fa2, 99edc18 |
| AT-3, AT-61 | every loader reads through the lake's `/v1/sql` API | 3ff8eff |
| AT-2 | FRED from `fred_vintages`, latest version published by as-of | dadea7a |
| AT-5 | Fama-French daily values public at the end of the next month | dadea7a |
| AT-65 | risk-free series forward-filled only; raises before first value | dadea7a |
| AT-4, AT-37 | run provenance block; loaders tested against lake contract v1 | 9ed37ed |
| AT-7, AT-8, AT-9, AT-11 | NYSE session calendar; short interest, earnings, SEC filings gated on it; insider log | fd8e1ca |
| AT-10, AT-12 | EPS as discrete quarters; dividends regular cash on one basis | 45daf84 |
| AT-15, AT-19, AT-21 | crypto at the US close; split-strike match; stale options hold | d335c7b |
| AT-16, AT-18 | regressions joined by date (drop names, not dates); options on the strikes' spot | 5a5b134 |
| AT-17 | resolved by AT-1's close-basis rescale | aec97fa |
| AT-59, AT-67 | agents' option chains lagged a day; settlement at the expiry close | 2caf7bf |
| AT-57 | option fills: premium floor, rounding, commission in as-traded units | cf7352c |
| AT-77 | asks 1 and 2 approved: edgelab executes; LLM scope event-desk then macro | 119b15a |
| AT-38, AT-39, AT-58, AT-62 | experiment options resolve after `--set`; derived MC seeds; membership by security, end exclusive; SIC limit documented | d29f8a3 |
| AT-60 | stale option quotes re-marked from the current surface; missing rate raises | 1e2316b |
| AT-63 | catalog trades SPY/IWM, not ^GSPC/^RUT; `USDJPY=X` | 7384697 |
| AT-64, AT-66 | preflight per source and lookback; inconsistent opens masked, unfillable orders skipped | 072c364 |
| AT-79 | SPY/QQQ/IWM option chains from `option_chain_history` before DoltHub | 3a8f250 |
| AT-66 follow-up | unfilled orders recorded and counted per run | ebb98a0 |
| AT-80 | S&P cross-section priced by security from `equity_daily_from_bars` | 7b37814 |
| AT-78 | quantcore a041d0d (0.3.0); flat assets and pods; stress-test zero variance | 9a2de88 |
| AT-6 | COT gated on CFTC's recorded release dates (`cftc_cot_publication`) | 21d5668 |
| AT-82 | cash earns the risk-free rate at the previous bar's rate; benchmark accrues the same; `run` and dashboard pass the series | 3d846b5 |
| AT-81 | idle-pod share and cash share recorded per day and per period; allocation unchanged | 2575e4d |
| AT-80 | exclusions re-measured after LK-50: 1.5-4.9% (was 4.4-8.8%) | 2575e4d |
| AT-83 part 1 | per-pod return, vol, Sharpe, drawdown over the whole run | bfa56ee |
| AT-83 study | per-pod ex-ante vol after each sizing stage and the binding stage | 9fd7a0f |

A1 baselines (five literature strategies on edgelab, 2010-01-01 to
2017-12-29) are verified. edgelab-62 runs the counted runs with `warmup`
equal to each strategy's declared history (EL-33 open). T1 labels:
survivorship coverage 0.89-0.91 (verdict capped, LK-44), Moskowitz without
CL/NG/HO/RB (deferred energy contracts are EIA settles only), ZN/ZB 6-day
roll (source gap LK-43).

A1 counted results (edgelab `runs/2026-10-27-a1`): the four S&P baselines
are graded "not assessable" until edgelab fixes EL-35 (membership-interval
coverage) and reruns them. Readings so far: reversal loses net of costs,
JT momentum and the ML ranker show no edge, Ang low-volatility looks
strong but random entry matches it a quarter of the time (p 0.25), so
unknown. Moskowitz v1.1.0 (roll at 10 days) lost money: Sharpe -0.79, HAC t
-2.25, loss-making before costs too; not assessable (26 missing bars,
LK-43). Futures were charged equity cost rates; edgelab EL-38 fixes the
defaults. No post-result cost version was registered.

## In progress

- AT-83 binding-constraint study: 30 names (20 large caps, SPY, SHY/IEF/TLT,
  HYG/LQD, GC=F/CL=F, EURUSD=X/USDJPY=X), 2016. The first run was stopped
  for low system memory before it finished; restart on the owner's word
  and ask database-57 for a fresh lake hold. Then part 2
  (`firm.target_annualized_vol` 0.10), then AT-40.

## Next steps, in order

1. When LK-13 and LK-14 land, move company facts and insider transactions
   back from the `_log` views to the current views.
2. AT-80: re-measure again when LK-52 (YHOO/AABA and 20 renamed members)
   lands; LK-51 (LDOS) landed in lake 7303df0. AT-68 is
   blocked by LK-49 (no usable declaration date; LK-48).
4. AT-40 ledger: shared format with pitch-edge (PE-21); the two
   conformance vectors are the acceptance tests and are written first.
5. AT-77: measure OmniRoute route costs once its key is in `.env`.

## Facts worth keeping

- `fred_vintages` DGS3MO and DGS10 vintages start 2005-06-28; a run ending
  before then has no risk-free series. VIXCLS starts 2010-11-22;
  BAMLH0A0HYM2 2023-09-29, so the credit analyst is unavailable 2010-2017.
- Lake contract v1 is frozen (lake commit 7eaacba). The views we read are
  vendored in `tests/fixtures/lakehouse_contract_v1.json`; regenerate them
  from the lake's `docs/contract-v1.json` when a new contract is frozen.
  Security ids: equity_ticker_history (ticker and date, inclusive),
  futures contract_code with futures_roll_calendar, options by OSI.
- Test-writers have only a full-file Write tool: when one edits an existing
  test file, diff it against HEAD for dropped tests before committing.
- `data/trading_calendar.py` is a hand-kept NYSE rule set (pandas holiday
  rules plus special closures); it matches SPY session dates 2000-2026
  exactly. A new unscheduled closure must be added there. Adding
  `exchange-calendars` was blocked by the permission classifier.
- The first lake request after 15 idle minutes can take up to 60 s.
- `option_chain_history` (SPY, QQQ, IWM) is daily from 2008 with ~16
  expiries, IV, greeks and open interest on every row; `snapshot_date` is
  the date column. A full-table aggregate through `/v1/sql` runs out of
  memory: query it by underlying and a date window.
- Never let an agent run `git stash` while another agent edits the tree.
- A/B across a dependency pin: check the old pin out in a scratch
  `git worktree` at HEAD with its own `uv sync --locked`, and symlink `.env`
  into it. Run both sides in parallel and check that both provenance
  blocks show the same `lake_version`. database-57 will hold the lake still
  on request.
- The stat-arb manager rescreens pairs each quarter and overwrites the
  roster's partner map. "No partner ticker configured" means no pair was
  admitted, not that the wiring is broken.
- B6 reconciliation with edgelab (AT-82): the accrual timing and the zero-excess cash book agree. If cash interest differs, check two conventions first: edgelab does not floor a negative rate at 0, and it takes rates as given in `bundle.rates` rather than annualising DGS3MO daily × 252. edgelab's headline Sharpe is excess; its metrics-table Sharpe and headline Sortino are not yet (EL-41).
