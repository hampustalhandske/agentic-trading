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
| AT-77 | asks 1 and 2 approved: edgelab executes; LLM scope event-desk then macro | 119b15a |

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
unknown. Moskowitz failed on the 2014-03-18 ZN/ZB gap because a roll fills
two sessions after its decision; the folder now rolls at 10 days
(v1.1.0) and awaits edgelab's rerun.

## In progress

- AT-7, AT-8, AT-9, AT-11 (one batch, Opus implementer after a
  test-writer). Agreed with database-57:
  - AT-7: gate on settlement + 8 NYSE sessions (+1 day), holiday-aware;
    the lake counts weekdays (LK-47). Rows with `revision_flag='R'` are
    available only from the next dissemination; originals are not in the
    log.
  - AT-8: "After market close" or null `when` is available the next
    trading day.
  - AT-9: SEC filings available on `filed_date` only if `accepted_at`
    (join `sec_filings` on `accession_number`) is before 16:00 ET,
    otherwise the next trading day.
  - AT-11: read `sec_insider_transactions_log` with DISTINCT over every
    column except scraped_at, dt and table until LK-14 lands (the `_log`
    views are a supported read).
  - Needs an NYSE session calendar (exchange_calendars XNYS is the plan).

## Next steps, in order

1. The AT-7/8/9/11 batch above, then AT-10 (discrete quarters from
   company facts) and AT-12 (dividends from `equity_distributions`).
2. AT-6 waits for LK-15 (COT `published_at`); until then the `+3 days`
   gate stands.
3. Remaining data and point-in-time musts: AT-15..19, 21, 38, 39,
   57..60, 62..64, 66..68, AT-79, AT-78 (re-pin quantcore to a041d0d).
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
- The first lake request after 15 idle minutes can take up to 60 s.
