# test_row_pick_census.py

"""Census of the row picks in `agentic_trading`.

A row pick keeps one row, or one order of rows, out of several. When the rows tie on the keys the
pick orders by, which row it keeps depends on the order the rows arrive in, and the lake returns
rows in no fixed order. Such a pick can make two runs over the same data differ.

The census reads the source of every module under `src/agentic_trading` with `ast`, without
importing it, and finds:

- SQL picks, in every string literal that is not a bare string statement (a docstring), with an
  f-string's constant parts and the constant parts of a `+` chain joined around a placeholder for
  each non-constant part: `arg_max`/`arg_min`/`max_by`/`min_by` (and DuckDB's `argmax`/`argmin`),
  `first(`/`last(`/`any_value(`/`arbitrary(`, `first_value(`/`last_value(`/`nth_value(`,
  `row_number(`/`rank(`/`dense_rank(`, `QUALIFY`, `DISTINCT ON`, a `LIMIT` with its row count
  (`order_by_limit` when an `ORDER BY` comes before it in the same string, `limit` when none
  does) and `ASOF` joins;
- pandas and numpy picks, as calls of an attribute or a name: `drop_duplicates`, `duplicated`,
  `first`, `last`, `nth`, `head`, `tail`, `idxmax`, `idxmin`, `nlargest`, `nsmallest`,
  `sort_values`, `sort_index`, `argsort`, `argmax`, `argmin` and `merge_asof`. Every such call
  is a candidate whatever its receiver; one that picks no row is allowed with that reason.

A sort's kind carries `(stable)` when it passes `kind="stable"` or `kind="mergesort"`. That is
information only: a stable sort keeps tied rows in their input order, so it is deterministic only
when its input order is. A sort on one key without a stable kind is a quicksort, which keeps no
order between tied rows; pandas sorts a frame on two or more columns with a stable lexsort
whatever `kind` it is given.

A site is keyed `<module>:<qualname>:<kind>#<n>`, where the qualname is the enclosing classes and
functions (`<module>` at top level) and `n` counts the sites of that kind in that scope in source
order, so moving lines does not change a key. Modules in `_NOT_FIRM_RECORD` (the dashboard and
the command-line entry points) are scanned too and tagged as outside the firm record's decision
path.

The sites found must be exactly those in `_ALLOWED`, each with the total-order key, the reason a
tie cannot occur or the reason the pick does not matter, and those in `KNOWN_UNTIED`, each naming
the issue that tracks it and what can tie; no site is on both lists."""

import ast
import importlib.util
import pathlib
import re
from typing import NamedTuple

# Modules outside the firm record's decision path, which its spec does not hash: the Streamlit
# dashboard and the command-line entry points (`cli.firm_record_main` starts a firm record and
# takes no decision of its own).
_NOT_FIRM_RECORD = (
    "agentic_trading.gui",
    "agentic_trading.cli",
    "agentic_trading.main",
    "agentic_trading.__main__",
)

# SQL patterns, matched case-insensitively against a string's text, and the kind each names.
_SQL_PATTERNS: tuple[tuple[re.Pattern[str], str | None], ...] = (
    (re.compile(r"\b(arg_max|arg_min|max_by|min_by|argmax|argmin)\s*\(", re.IGNORECASE), None),
    (
        re.compile(
            r"\b(first|last|any_value|arbitrary|first_value|last_value|nth_value)\s*\(",
            re.IGNORECASE,
        ),
        None,
    ),
    (re.compile(r"\b(row_number|rank|dense_rank)\s*\(", re.IGNORECASE), None),
    (re.compile(r"\bQUALIFY\b", re.IGNORECASE), "qualify"),
    (re.compile(r"\bDISTINCT\s+ON\b", re.IGNORECASE), "distinct_on"),
    (re.compile(r"\bASOF\b", re.IGNORECASE), "asof_join"),
)
# A `LIMIT` followed by its row count: a number, a parameter or an interpolated value.
_SQL_LIMIT = re.compile(r"\bLIMIT\s+(?:\d|\?|\{\}|\$\d|:\w)", re.IGNORECASE)
_SQL_ORDER_BY = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)

# Called attribute or name: the kind of pandas/numpy pick it names.
_CALL_KINDS: dict[str, str] = {
    "drop_duplicates": "pandas.drop_duplicates",
    "duplicated": "pandas.duplicated",
    "first": "pandas.first",
    "last": "pandas.last",
    "nth": "pandas.nth",
    "head": "pandas.head",
    "tail": "pandas.tail",
    "idxmax": "pandas.idxmax",
    "idxmin": "pandas.idxmin",
    "nlargest": "pandas.nlargest",
    "nsmallest": "pandas.nsmallest",
    "sort_values": "pandas.sort_values",
    "sort_index": "pandas.sort_index",
    "merge_asof": "pandas.merge_asof",
    "argsort": "numpy.argsort",
    "argmax": "numpy.argmax",
    "argmin": "numpy.argmin",
}
_SORTS = frozenset({"sort_values", "sort_index", "argsort"})
_STABLE_KINDS = frozenset({"stable", "mergesort"})


class Site(NamedTuple):
    """One row pick: its module, enclosing qualname and kind, and where it is: a call at the end
    of its method or function name, so the calls of a chain are in source order, and an SQL pick
    at the start of its string."""

    module: str
    qualname: str
    kind: str
    line: int
    column: int


def _string_text(node: ast.expr) -> str | None:
    """The text of a string expression: a string constant, an f-string with `{}` for each
    interpolated value, or a `+` chain with at least one string part and `{}` for each other
    part. `None` for any other expression."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value if isinstance(part, ast.Constant) else "{}" for part in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _string_text(node.left), _string_text(node.right)
        if left is None and right is None:
            return None
        return (left if left is not None else "{}") + (right if right is not None else "{}")
    return None


def _sql_kinds(text: str) -> list[str]:
    """The kind of each SQL pick in `text`, in the order of the patterns."""
    kinds: list[str] = []
    for pattern, kind in _SQL_PATTERNS:
        kinds.extend(f"sql.{kind or match.group(1).lower()}" for match in pattern.finditer(text))
    for match in _SQL_LIMIT.finditer(text):
        ordered = _SQL_ORDER_BY.search(text, 0, match.start()) is not None
        kinds.append("sql.order_by_limit" if ordered else "sql.limit")
    return kinds


def _call_kind(node: ast.Call) -> str | None:
    """The kind of pandas/numpy pick `node` calls, with `(stable)` on a stable sort, or `None`."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name not in _CALL_KINDS:
        return None
    kind = _CALL_KINDS[name]
    if name in _SORTS and any(
        keyword.arg == "kind"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value in _STABLE_KINDS
        for keyword in node.keywords
    ):
        kind += "(stable)"
    return kind


class _Scanner(ast.NodeVisitor):
    """Collects the `Site`s of one module's tree in source order."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.scope: list[str] = []
        self.sites: list[Site] = []

    def _add(self, kind: str, line: int, column: int) -> None:
        qualname = ".".join(self.scope) or "<module>"
        self.sites.append(Site(self.module, qualname, kind, line, column))

    def _visit_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        # A bare string statement is a docstring, not a query.
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        kind = _call_kind(node)
        if kind is not None:
            func = node.func
            self._add(kind, func.end_lineno or func.lineno, func.end_col_offset or func.col_offset)
        self.generic_visit(node)

    def _visit_string(self, node: ast.expr) -> None:
        """Scans the text of the string expression `node` as SQL once, then visits the
        expressions interpolated into it."""
        text = _string_text(node)
        assert text is not None
        for kind in _sql_kinds(text):
            self._add(kind, node.lineno, node.col_offset)
        self._visit_string_parts(node)

    def _visit_string_parts(self, node: ast.expr) -> None:
        if isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.FormattedValue):
                    self.visit(part.value)
        elif isinstance(node, ast.BinOp) and _string_text(node) is not None:
            self._visit_string_parts(node.left)
            self._visit_string_parts(node.right)
        elif not isinstance(node, ast.Constant):
            self.visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self._visit_string(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self._visit_string(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if _string_text(node) is None:
            self.generic_visit(node)
        else:
            self._visit_string(node)


def scan_source(source: str, module: str) -> list[Site]:
    """Every row pick in the module `module` whose source is `source`, in source order."""
    scanner = _Scanner(module)
    scanner.visit(ast.parse(source))
    return sorted(scanner.sites, key=lambda site: (site.line, site.column))


def scan_package(root: pathlib.Path, package: str) -> list[Site]:
    """Every row pick in every module of the package `package` whose directory is `root`."""
    sites: list[Site] = []
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module = ".".join((package, *parts))
        sites.extend(scan_source(path.read_text(encoding="utf-8"), module))
    return sites


def site_keys(sites: list[Site]) -> dict[str, Site]:
    """Each site under its key `<module>:<qualname>:<kind>#<n>`, `n` counting from 1 the sites
    of the same module, qualname and kind in the order given."""
    keyed: dict[str, Site] = {}
    counts: dict[tuple[str, str, str], int] = {}
    for site in sites:
        scope = (site.module, site.qualname, site.kind)
        counts[scope] = counts.get(scope, 0) + 1
        keyed[f"{site.module}:{site.qualname}:{site.kind}#{counts[scope]}"] = site
    return keyed


def _in_firm_record(module: str) -> bool:
    return not any(
        module == package or module.startswith(f"{package}.") for package in _NOT_FIRM_RECORD
    )


def _census() -> dict[str, Site]:
    """Every row pick in `agentic_trading`, keyed by `site_keys`."""
    spec = importlib.util.find_spec("agentic_trading")
    assert spec is not None and spec.submodule_search_locations
    root = pathlib.Path(next(iter(spec.submodule_search_locations)))
    return site_keys(scan_package(root, "agentic_trading"))


_CVAR = "agentic_trading.agents.analysts.cvar_pod"
_EVENT_CREDIT = "agentic_trading.agents.analysts.event_credit_pod"
_HMM = "agentic_trading.agents.analysts.stat_arb_pod.regime_hmm_analyst"
_VIX = "agentic_trading.agents.analysts.tail_hedge_pod.vix_tail_hedge_analyst"
_LOADER = "agentic_trading.data.loaders.lakehouse_loader"
_OPTION_MARKS = "agentic_trading.data.loaders.option_marks"
_DASHBOARD = "agentic_trading.gui.dashboard"

_CHAIN_ORDER = (
    "load_options_snapshot_from_lakehouse sorts the chain on date and then every column it "
    "returns, so of equally distant strikes the first, the lower strike, is kept"
)
_MARKET_DATA_DATES = (
    "keys unique: the price frame has one row per date (data.loader aligns it on sessions or "
    "on the intersection of tickers' unique date indexes)"
)
_PIVOT_INDEX = "keys unique: the index is the row index of a pivot, one row per date"

# "<module>:<qualname>:<kind>#<n>": why the pick's result is the same whatever order tied rows
# arrive in.
_ALLOWED: dict[str, str] = {
    f"{_CVAR}._options_data:real_atm_options_by_expiration:pandas.idxmin#1": _CHAIN_ORDER,
    f"{_CVAR}._options_data:real_atm_options_by_expiration:pandas.sort_values(stable)#1": (
        "sorted on days_to_expiration then expiration: one row per expiration, so the key is unique"
    ),
    f"{_CVAR}.heston_vol_surface_analyst:_interpolated_atm_iv:numpy.argmin#1": (
        "a tie cannot occur: this branch runs only with every expiry on one side of the target, "
        "and the expiries it gets are distinct and at least MIN_NEAREST_DTE days out, so their "
        "distances to the target differ"
    ),
    f"{_EVENT_CREDIT}.sec_fundamentals_analyst:discrete_quarters:pandas.sort_values(stable)#1": (
        "stable sort on filed, available_date of rows in a total order: "
        "load_company_facts_from_lakehouse fetches them sorted on every column and reorders "
        "them only with stable sorts"
    ),
    f"{_EVENT_CREDIT}.sec_fundamentals_analyst:discrete_quarters:pandas.drop_duplicates#1": (
        "keep='last' after the stable sort on filed, available_date: tied facts keep the total "
        "order of load_company_facts_from_lakehouse"
    ),
    f"{_EVENT_CREDIT}.sec_fundamentals_analyst:discrete_quarters:pandas.drop_duplicates#2": (
        "keep='last' after the stable sort on filed, available_date: tied facts keep the total "
        "order of load_company_facts_from_lakehouse"
    ),
    f"{_EVENT_CREDIT}.sec_fundamentals_analyst:discrete_quarters:pandas.sort_values(stable)#2": (
        "keys unique: quarters are deduplicated on period_end, and a derived Q4 takes the "
        "period_end of an annual fact (deduplicated on it) that no quarter ends on"
    ),
    f"{_EVENT_CREDIT}.sec_fundamentals_analyst:SecFundamentalsAnalyst.generate_signal"
    ":pandas.tail#1": "keys unique: discrete_quarters returns one row per period_end, sorted",
    f"{_EVENT_CREDIT}.short_interest_squeeze_analyst:ShortInterestSqueezeAnalyst"
    ".generate_signal:pandas.sort_values(stable)#1": (
        "stable sort on settlement_date of rows in a total order: _fetch_short_interest orders "
        "by settlement_date, revision_flag NULLS FIRST and every other column it returns"
    ),
    f"{_HMM}:RegimeHMMAnalyst.generate_signal:numpy.argmin#1": (
        "picks between the two fitted HMM states, in hmm_fit's order, not between rows; reached "
        "only when the variances are equal"
    ),
    f"{_HMM}:RegimeHMMAnalyst.generate_signal:numpy.argmax#1": (
        "picks between the two fitted HMM states, in hmm_fit's order, not between rows; equal "
        "variances take the argmin branch"
    ),
    f"{_VIX}:VixTailHedgeAnalyst.generate_signal:pandas.tail#1": (
        "keys unique: load_price_history_from_lakehouse indexes each ticker by its unique dates"
    ),
    "agentic_trading.agents.commander:_apply_firm_limits:pandas.tail#1": _MARKET_DATA_DATES,
    "agentic_trading.backtest.checkpoint_store:read_recent:sql.order_by_limit#1": (
        "ORDER BY as_of_date DESC, rowid DESC: rowid is unique"
    ),
    "agentic_trading.backtest.engine:BacktestEngine._carry_rates:pandas.sort_index#1": (
        "keys unique: policy_rates is load_policy_rates_from_lakehouse's frame, a pivot with "
        "one row per date"
    ),
    "agentic_trading.backtest.engine:_liquidity_inputs:pandas.tail#1": _MARKET_DATA_DATES,
    "agentic_trading.backtest.preflight:check_price_coverage:pandas.sort_values#1": (
        "sorts the values of a DatetimeIndex: tied values are equal"
    ),
    f"{_LOADER}:_deduplicated_log_sql:sql.row_number#1": (
        "ORDER BY scraped_at DESC then the tie_order the lake serves for the table; a current "
        "table it serves none for is refused (AT-157, AT-175)"
    ),
    f"{_LOADER}:_require_rows:sql.limit#1": (
        "SELECT 1 AS present ... LIMIT 1: every row is the constant 1"
    ),
    f"{_LOADER}:_select_sp500_members:pandas.sort_values(stable)#1": (
        "rows tied on start_date and ticker are deduplicated on (security_id, unresolved "
        "ticker) and only security_id and ticker are kept, so of two tied rows of one key "
        "either gives the same member"
    ),
    f"{_LOADER}:_select_sp500_members:pandas.drop_duplicates#1": (
        "only security_id and ticker of the kept row are used, and tied rows of one key share "
        "both after the sort on start_date DESC, ticker"
    ),
    f"{_LOADER}:_select_sp500_members:pandas.sort_values(stable)#2": (
        "keys unique: after the dedupe on (security_id, unresolved ticker) no two rows share "
        "ticker and security_id"
    ),
    f"{_LOADER}:unfactored_split_moves:pandas.sort_index#1": (
        "keys unique: its only caller passes _log_return_series output, one row per date per "
        "security after the dedupe on (security_id, date)"
    ),
    f"{_LOADER}:_fetch_security_bars:sql.asof_join#1": (
        "keys unique: the factors are grouped to one row per (security_id, effective_date), "
        "and a joined group with more than one cum_factor is refused"
    ),
    f"{_LOADER}:_refuse_factor_collisions:pandas.drop_duplicates#1": (
        "drop_duplicates over all columns (security_id, factor_date): tied rows are identical"
    ),
    f"{_LOADER}:_select_security_bars:pandas.drop_duplicates#1": (
        "ORDER BY security_id, date, raw_close (then open, volume when read): rows tied on "
        "these are equal in every column returned"
    ),
    f"{_LOADER}:_log_return_series:pandas.sort_index#1": (
        "keys unique: _select_security_bars deduplicates on (security_id, date)"
    ),
    f"{_LOADER}:_equity_price_rows:pandas.sort_index#1": (
        "keys unique: value_counts gives one row per ticker"
    ),
    f"{_LOADER}:load_price_history_from_lakehouse:pandas.sort_index#1": (
        "keys unique: instrument_prices is keyed on (ticker, date) and crypto rows aggregate "
        "to one per (symbol, date)"
    ),
    f"{_LOADER}:_select_fred_vintages:pandas.drop_duplicates#1": (
        "ORDER BY date, realtime_start NULLS FIRST, value NULLS FIRST; known_at follows from "
        "date and realtime_start, so rows tied on the order are equal"
    ),
    f"{_LOADER}:_select_policy_rates:pandas.sort_values(stable)#1": (
        "the frame is pivoted to one cell per (date, currency), which pivot requires unique, "
        "so the row order does not reach the result"
    ),
    f"{_LOADER}:_select_policy_rates:pandas.sort_index#1": _PIVOT_INDEX,
    f"{_LOADER}:_select_fama_french:pandas.sort_index#1": (
        "keys unique: the index is the row index of a pivot_table, one row per date"
    ),
    f"{_LOADER}:_select_company_facts:pandas.sort_values(stable)#1": (
        "stable sort of rows fetched in a total order (ORDER BY every column returned)"
    ),
    f"{_LOADER}:_select_company_facts:pandas.drop_duplicates#1": (
        "keep='last' after a stable sort of rows fetched in a total order (ORDER BY every "
        "column returned)"
    ),
    f"{_LOADER}:_select_company_facts:pandas.sort_values(stable)#2": (
        "stable sort of rows in a total order: the fetch's ORDER BY every column, reordered "
        "only by stable sorts"
    ),
    f"{_OPTION_MARKS}:_surface_volatility:pandas.sort_index#1": (
        "keys unique: the index is a groupby on strike"
    ),
    f"{_OPTION_MARKS}:nearest_strike:pandas.idxmin#1": _CHAIN_ORDER,
    "agentic_trading.stage_a.store:SignalStore.counts:pandas.sort_index#1": _PIVOT_INDEX,
    "agentic_trading.trials.ledger:_append:sql.order_by_limit#1": (
        "ORDER BY seq DESC LIMIT 1: seq is the ledger_chain primary key"
    ),
    f"{_DASHBOARD}:_combined_chart_frame:pandas.sort_index#1": (
        "keys unique: the index is the union of its series' date indexes"
    ),
    f"{_DASHBOARD}:_build_trade_blotter:pandas.sort_values#1": (
        "a sort on two columns is a stable lexsort; ties keep the order of the engine's trade log"
    ),
    f"{_DASHBOARD}:_render_saved_run_tab:pandas.sort_values#1": (
        "a sort on two columns is a stable lexsort; ties keep the order of the run's "
        "signal_health mapping"
    ),
}

# "<module>:<qualname>:<kind>#<n>": the issue tracking a pick whose result can depend on the order
# tied rows arrive in, and what can tie.
KNOWN_UNTIED: dict[str, str] = {
    f"{_LOADER}:load_sp500_member_count:pandas.sort_index#1": (
        "AT-176: two sp500_member_count rows of one date tie, in the lake's order"
    ),
    f"{_DASHBOARD}:_source_summary_frame:pandas.sort_values#1": (
        "AT-176: a source's signals on one date (one per asset or pod) tie, and the last is shown"
    ),
    f"{_DASHBOARD}:_trust_panel_frame:pandas.sort_values#1": (
        "AT-176: sources of equal weight tie in the displayed order"
    ),
    f"{_DASHBOARD}:_saved_run_pod_analyst_frame:pandas.sort_values#1": (
        "AT-176: analysts of equal cumulative_return tie in the displayed order"
    ),
    f"{_DASHBOARD}:_saved_run_pod_frame:pandas.sort_values#1": (
        "AT-176: pods of equal cumulative_return tie in the displayed order"
    ),
    f"{_DASHBOARD}:_show_source_history_dialog:pandas.sort_values#1": (
        "AT-176: a source's signals on one date (one per pod) tie in the displayed order"
    ),
    f"{_DASHBOARD}:_show_source_history_dialog:pandas.sort_values#2": (
        "AT-176: a source's signals on one date (one per pod) tie in the displayed order"
    ),
}


def _unlisted(found: dict[str, Site]) -> list[str]:
    """The keys of `found` on neither list, each with its line and, for a module in
    `_NOT_FIRM_RECORD`, a tag saying it is outside the firm record's decision path."""
    listed = set(_ALLOWED) | set(KNOWN_UNTIED)
    return [
        f"{key} (line {site.line}"
        + ("" if _in_firm_record(site.module) else ", outside the firm record")
        + ")"
        for key, site in found.items()
        if key not in listed
    ]


def test_census_is_exactly_the_two_lists() -> None:
    """The row picks in `agentic_trading` are exactly `_ALLOWED` and `KNOWN_UNTIED` together: a
    new pick must be listed, and a listed pick that is gone must be removed."""
    found = _census()
    unlisted = _unlisted(found)
    assert not unlisted, (
        "row picks on neither list; break their ties on a total-order key and add them to "
        "_ALLOWED with that key, or add them to KNOWN_UNTIED with their issue: "
        + ", ".join(unlisted)
    )
    stale_allowed = sorted(set(_ALLOWED) - set(found))
    assert not stale_allowed, (
        f"_ALLOWED names picks the census no longer finds: {', '.join(stale_allowed)}"
    )
    stale_known = sorted(set(KNOWN_UNTIED) - set(found))
    assert not stale_known, (
        f"KNOWN_UNTIED names picks the census no longer finds: {', '.join(stale_known)}"
    )


def test_allowed_and_known_untied_are_disjoint() -> None:
    """No pick is both allowed and known to be untied."""
    assert not set(_ALLOWED) & set(KNOWN_UNTIED)


def test_every_known_untied_pick_names_its_issue() -> None:
    """Every `KNOWN_UNTIED` entry names the issue that tracks it and says what can tie."""
    for key, issue in KNOWN_UNTIED.items():
        assert re.match(r"AT-\d+: \S", issue), key


_PLANTED_SOURCE = '''
"""A planted module with one untied row pick of each family."""

import pandas as pd


def latest_closes(lake):
    return lake.sql("SELECT ticker, arg_max(close, date) AS close FROM prices GROUP BY ticker")


def first_quote(lake, table):
    return lake.sql(f"SELECT * FROM {table} ORDER BY quote_date LIMIT 1")


def one_row_per_day(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop_duplicates(subset=["date"])


def first_per_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("ticker").first()


def by_score(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("score")
'''


def test_census_reports_every_planted_untied_pick(tmp_path: pathlib.Path) -> None:
    """A planted package with an SQL `arg_max` without a tie-break, an `ORDER BY ... LIMIT 1`
    on a non-unique column, a `drop_duplicates` on a subset, a `groupby().first()` and a
    single-key `sort_values` has every one of them reported as on neither list."""
    package = tmp_path / "planted"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "picks.py").write_text(_PLANTED_SOURCE, encoding="utf-8")

    found = site_keys(scan_package(package, "planted"))

    assert sorted(_unlisted(found)) == sorted(
        [
            "planted.picks:latest_closes:sql.arg_max#1 (line 8)",
            "planted.picks:first_quote:sql.order_by_limit#1 (line 12)",
            "planted.picks:one_row_per_day:pandas.drop_duplicates#1 (line 16)",
            "planted.picks:first_per_ticker:pandas.first#1 (line 20)",
            "planted.picks:by_score:pandas.sort_values#1 (line 24)",
        ]
    )


_PROBE_SOURCE = '''
"""Prose is not SQL: ORDER BY date LIMIT 1, arg_max(x, y)."""


def picks(frame, table, n):
    """Neither is this: QUALIFY row_number() = 1."""
    split = f"SELECT * FROM {table} ORDER BY d " f"LIMIT {n}"
    chained = "SELECT * FROM t ORDER BY d" + " LIMIT 1"
    windowed = (
        "SELECT * FROM t "
        "QUALIFY row_number() OVER (PARTITION BY k ORDER BY d) = 1"
    )
    nearest = "SELECT * FROM a ASOF JOIN b ON a.k = b.k AND a.d >= b.d"
    distinct = "SELECT DISTINCT ON (k) * FROM t"
    message = "the firm gross limit binds"
    ordered = frame.sort_values("d", kind="stable").sort_index(kind="mergesort")
    return ordered.sort_values("e").tail(1), np.argsort(frame["d"].to_numpy(), kind="stable")
'''


def test_scanner_reads_every_form_of_string_and_call() -> None:
    """SQL is read across implicitly joined f-strings and `+` chains, never in docstrings or in
    prose using a keyword; chained calls come in source order, and a sort passing a stable
    `kind` is marked `(stable)`."""
    sites = scan_source(_PROBE_SOURCE, "probe")

    assert [(site.qualname, site.kind) for site in sites] == [
        ("picks", "sql.order_by_limit"),
        ("picks", "sql.order_by_limit"),
        ("picks", "sql.row_number"),
        ("picks", "sql.qualify"),
        ("picks", "sql.asof_join"),
        ("picks", "sql.distinct_on"),
        ("picks", "pandas.sort_values(stable)"),
        ("picks", "pandas.sort_index(stable)"),
        ("picks", "pandas.sort_values"),
        ("picks", "pandas.tail"),
        ("picks", "numpy.argsort(stable)"),
    ]
    assert list(site_keys(sites))[:2] == [
        "probe:picks:sql.order_by_limit#1",
        "probe:picks:sql.order_by_limit#2",
    ]
