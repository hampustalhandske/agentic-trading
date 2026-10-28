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

Nothing in flight.

## Next steps, in order

1. When LK-13 and LK-14 land, move company facts and insider transactions
   back from the `_log` views to the current views.
2. AT-6 waits for LK-15 (COT `published_at`); until then the `+3 days`
   gate stands.
3. AT-78: re-pin quantcore to a041d0d and record every numeric change.
   AT-80's residual exclusions (table in ISSUES) shrink when LK-50 lands;
   re-measure then. AT-68 is blocked by LK-49 (no
   usable declaration date; LK-48).
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
