#!/usr/bin/env python3
"""Download free historical option chains, closes, and dividends from DoltHub.

Usage: fetch_dolthub_chains.py <out_dir> [--symbol SPY] [--start 2019-01-01] [--end 2026-12-31]
                               [--weekdays 0,1,2,3,4] [--workers 8] [--as-of today]
                               [--settle-days 7]

Source: the public DoltHub databases post-no-preference/options (end-of-day option bid/ask,
about four expirations near the money, mostly three days a week from 2019 and more often later)
and post-no-preference/stocks (daily OHLCV and cash dividends). Both are free and need no
account. They are read through DoltHub's SQL API, which caps each response and times out long
scans: even listing one month's chain dates for one symbol can exceed its limit. Point lookups by
date and symbol use the tables' key, so each candidate weekday is probed: its chain, then the
underlying's close on that day.

Writes <out_dir>/chains/<SYMBOL>/<date>.csv, one per chain day, with the day's close repeated in
an `underlying_close` column so that a day is one file, and <out_dir>/<SYMBOL>_dividends.csv.
A day with no chain gets an empty <date>.none marker. A day with a chain but no close gets an
empty <date>.noclose marker: the options table has chains on some market holidays (2020-01-20,
2020-04-10, 2020-05-25) when the stocks table, rightly, has no close, so a retry cannot succeed.
Such a day is kept out of the data, since the reader would have no spot check, and is listed
rather than counted as a failure. Existing day files and markers are kept, so an interrupted run
resumes; only transport or API errors fail the run and are retried by rerunning.
The source posts a day after it ends, so a day within --settle-days of --as-of that has no chain
or no close may only be unpublished: it gets no marker and is probed again by the next run.
Each file is written to a temporary name and renamed, so a partial file never looks complete.
Keep the output outside version control, like all vendor data here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

API = "https://www.dolthub.com/api/v1alpha1"
OPTIONS = "post-no-preference/options"
STOCKS = "post-no-preference/stocks"
CHAIN_COLUMNS = ("date", "expiration", "strike", "call_put", "bid", "ask", "vol", "delta", "gamma")
DAY_COLUMNS = (*CHAIN_COLUMNS, "underlying_close")
# A day's chain is about 200 rows. A response this large may have been cut at the API's cap.
_ROW_CAP = 1000
# fetch_day's result for a recent day the source may not have published yet: nothing is written.
PENDING = -1


class FetchError(RuntimeError):
    pass


def query(repo: str, sql: str, *, attempts: int = 5) -> list[dict[str, str | None]]:
    url = f"{API}/{repo}/master?" + urllib.parse.urlencode({"q": sql})
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == attempts:
                raise FetchError(f"{repo}: {exc}") from exc
            time.sleep(delay)
            delay *= 2
            continue
        status = payload.get("query_execution_status")
        if status == "Success":
            return list(payload.get("rows", []))
        message = payload.get("query_execution_message", "")
        if attempt == attempts:
            raise FetchError(f"{repo}: {status}: {message}")
        time.sleep(delay)
        delay *= 2
    raise AssertionError("unreachable")


def _write_atomic(path: Path, header: tuple[str, ...], rows: list[dict[str, str | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if row.get(c) is None else row[c] for c in header])
    tmp.replace(path)


def weekdays(start: date, end: date, weekday_set: frozenset[int]) -> list[date]:
    """Candidate days. Even a one-month listing of chain dates can exceed the API's time limit,
    while a point lookup by date and symbol uses the table's key, so every candidate is probed."""
    span = (end - start).days + 1
    return [
        start + timedelta(days=k)
        for k in range(span)
        if (start + timedelta(days=k)).weekday() in weekday_set
    ]


MARKERS = (".none", ".noclose")


def fetch_day(symbol: str, day: date, target: Path, *, settled: bool = True) -> int | None:
    """Rows written for the day; 0 with an empty `.none` marker when it has no chain; None with
    an empty `.noclose` marker when it has a chain but the stocks table has no close for it.

    A day that is not `settled` may be unpublished rather than missing, so either gap returns
    PENDING and writes nothing, and the next run probes the day again.
    """
    rows = query(
        OPTIONS,
        f"SELECT {', '.join(CHAIN_COLUMNS)} FROM option_chain "
        f"WHERE date = '{day}' AND act_symbol = '{symbol}' "
        "ORDER BY expiration, strike, call_put",
    )
    if not rows:
        if not settled:
            return PENDING
        target.with_suffix(".none").touch()
        return 0
    if len(rows) >= _ROW_CAP:
        raise FetchError(f"{symbol} {day}: {len(rows)} rows, which may be truncated at the cap")
    closes = query(
        STOCKS, f"SELECT close FROM ohlcv WHERE date = '{day}' AND act_symbol = '{symbol}'"
    )
    if len(closes) != 1 or closes[0].get("close") in (None, ""):
        if not settled:
            return PENDING
        # Kept out rather than written without a close: the reader would have no spot check.
        target.with_suffix(".noclose").touch()
        return None
    close = closes[0]["close"]
    _write_atomic(target, DAY_COLUMNS, [{**row, "underlying_close": close} for row in rows])
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2019, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="default: --as-of")
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date.today(),
        help="the day the run stands on, today unless replaying; sets the settle window",
    )
    parser.add_argument(
        "--settle-days",
        type=int,
        default=7,
        help="days within this many of --as-of with no chain or close are not marked, since the "
        "source may not have posted them yet",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--weekdays",
        default="0,1,2,3,4",
        help="Monday is 0. The source mostly samples Mon, Wed, and Fri, so 0,2,4 gets most "
        "days. The registered run probed every weekday, the default",
    )
    args = parser.parse_args()
    symbol = args.symbol.upper()
    if not symbol.isalnum():
        raise SystemExit(f"symbol must be alphanumeric: {symbol!r}")

    dividends = query(
        STOCKS,
        f"SELECT ex_date, amount FROM dividend WHERE act_symbol = '{symbol}' ORDER BY ex_date",
    )
    if len(dividends) >= _ROW_CAP:
        raise SystemExit(f"{symbol}: {len(dividends)} dividends, which may be truncated")
    _write_atomic(args.out / f"{symbol}_dividends.csv", ("ex_date", "amount"), dividends)
    print(f"{symbol}: {len(dividends)} dividends", flush=True)

    if args.settle_days < 0:
        raise SystemExit("--settle-days must be zero or more")
    end = args.end if args.end is not None else args.as_of
    weekday_set = frozenset(int(x) for x in args.weekdays.split(","))
    days = weekdays(args.start, end, weekday_set)
    folder = args.out / "chains" / symbol
    folder.mkdir(parents=True, exist_ok=True)
    todo = [
        d
        for d in days
        if not (folder / f"{d}.csv").exists()
        and not any((folder / f"{d}{m}").exists() for m in MARKERS)
    ]
    print(f"{symbol}: {len(days)} candidate days, {len(todo)} to probe", flush=True)
    found = 0
    no_close: list[date] = []
    pending: list[date] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        futures = {
            d: pool.submit(
                fetch_day,
                symbol,
                d,
                folder / f"{d}.csv",
                settled=(args.as_of - d).days > args.settle_days,
            )
            for d in todo
        }
        for index, (day, future) in enumerate(futures.items(), 1):
            try:
                count = future.result()
            except FetchError as exc:
                failures.append(str(exc))
                continue
            if count is None:
                no_close.append(day)
            elif count == PENDING:
                pending.append(day)
            else:
                found += count > 0
            if index % 50 == 0 or index == len(todo):
                print(f"{index}/{len(todo)} {day} chains_found={found}", flush=True)
    if no_close:
        listed = ", ".join(str(d) for d in no_close)
        print(f"{len(no_close)} chain days have no close and are kept out: {listed}", flush=True)
    if pending:
        listed = ", ".join(str(d) for d in pending)
        print(f"{len(pending)} recent days are not posted yet and will be probed again: {listed}")
    if failures:
        print(f"{len(failures)} days failed; rerun to retry them:", file=sys.stderr)
        for line in failures[:20]:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
