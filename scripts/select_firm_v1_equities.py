# select_firm_v1_equities.py

"""Selects the equity part of the registered firm's universe (agentic-firm-v1):
the S&P 500 members on 2009-12-31 with the highest average daily dollar
volume over the 2009 XNYS sessions, using only data known by 2009-12-31.

Dollar volume on a session is the bar's raw (unadjusted) close times its
raw volume, from `equity_daily_from_bars`; a session with no bar or no
volume is left out of that member's average. Members are ranked by that
average, ties broken by ticker, and the first `--count` are printed as JSON
(security_id, ticker on 2009-12-31, average dollar volume, sessions used),
with the lake time the reads were pinned to.

Usage: uv run python scripts/select_firm_v1_equities.py [--count 30] [--out PATH]
"""

import argparse
import json
from datetime import date
from pathlib import Path

from agentic_trading.data.loaders import lakehouse_loader

_AS_OF = date(2009, 12, 31)
_START = date(2009, 1, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with lakehouse_loader.lake_session(data_end=_AS_OF) as session:
        members = lakehouse_loader.load_sp500_member_securities_asof(_AS_OF)
        if not members:
            raise SystemExit(f"no S&P 500 membership on {_AS_OF}")
        bars = lakehouse_loader.load_security_daily_bars(
            [member.security_id for member in members], _START, _AS_OF
        )
        if bars is None:
            raise SystemExit("no bars for the members")
        ranked = []
        for member in members:
            if member.security_id not in bars.columns.get_level_values("security_id"):
                continue
            frame = bars[member.security_id]
            dollar_volume = (frame["raw_close"] * frame["volume"]).dropna()
            if dollar_volume.empty:
                continue
            ranked.append(
                {
                    "security_id": member.security_id,
                    "ticker": member.ticker,
                    "avg_dollar_volume": float(dollar_volume.mean()),
                    "sessions": int(dollar_volume.size),
                }
            )
        ranked.sort(key=lambda row: (-row["avg_dollar_volume"], row["ticker"]))
        result = {
            "as_of": _AS_OF.isoformat(),
            "window": [_START.isoformat(), _AS_OF.isoformat()],
            "lake_time": session.lake_time.isoformat() if session.lake_time else None,
            "members": len(members),
            "ranked_members": len(ranked),
            "selected": ranked[: args.count],
        }
    text = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
