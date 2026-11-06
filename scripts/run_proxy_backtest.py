#!/usr/bin/env python3
"""Run the pre-registered proxy backtest. Every variant below is reported; none may be dropped.

Usage: run_proxy_backtest.py <SP500.csv> <VIXCLS.csv> <DGS3MO.csv> --out results.json

The variants were fixed in this file before the first run. Changing them later means adding to the
list and recording the reason in docs/evidence/phase4/ledger.md, so the selection is visible.

Spot, VIX, and rates are real FRED closes. Option prices are modeled: see backtest/proxy.py.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from greeks_neutralizer._version import __version__
from greeks_neutralizer.backtest.engine import BacktestResult, run_backtest
from greeks_neutralizer.backtest.policies import registered_policy
from greeks_neutralizer.backtest.proxy import (
    SYMBOL,
    HistoryRow,
    LongPutOverlayBook,
    ProxyMarket,
    ShortStrangleBook,
    SpreadModel,
    join_history,
    load_fred_series,
)
from greeks_neutralizer.backtest.report import compare, policy_stats, stress_days

# Dropped from the report while the trailing long-run vol level is still forming.
WARMUP_DAYS = 252
STRESS_DAYS = (date(2018, 2, 5), date(2020, 3, 16), date(2024, 8, 5))


@dataclass(frozen=True, slots=True)
class Variant:
    book: str
    policy: str
    spread_multiple: float


VARIANTS: tuple[Variant, ...] = (
    *(
        Variant("short_strangle", policy, 1.0)
        for policy in (
            "unhedged",
            "delta_daily",
            "delta_band",
            "opt_dg",
            "opt_dgv",
            "opt_dgv_cheap",
            "opt_dgv_cvar",
        )
    ),
    *(
        Variant("short_strangle", policy, 3.0)
        for policy in ("unhedged", "delta_daily", "delta_band", "opt_dgv", "opt_dgv_cheap")
    ),
    *(
        Variant("long_put_overlay", policy, 1.0)
        for policy in ("unhedged", "delta_daily", "opt_dgv", "opt_dgv_cheap")
    ),
)


def _run(args: tuple[Variant, tuple[HistoryRow, ...]]) -> tuple[Variant, BacktestResult, float]:
    variant, history = args
    start = time.perf_counter()
    market = ProxyMarket(history, spreads=SpreadModel(multiple=variant.spread_multiple))
    book = ShortStrangleBook() if variant.book == "short_strangle" else LongPutOverlayBook()
    result = run_backtest(market, book, registered_policy(variant.policy, SYMBOL))
    return variant, result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sp500", type=Path)
    parser.add_argument("vix", type=Path)
    parser.add_argument("rates", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()

    history = join_history(
        load_fred_series(args.sp500), load_fred_series(args.vix), load_fred_series(args.rates)
    )
    print(f"history {history[0].day} to {history[-1].day}: {len(history)} days", flush=True)

    with mp.Pool(args.processes) as pool:
        finished = []
        for variant, result, seconds in pool.imap_unordered(
            _run, [(variant, history) for variant in VARIANTS]
        ):
            print(f"{variant} {seconds:.0f}s failures={len(result.failures)}", flush=True)
            finished.append((variant, result))

    groups: dict[tuple[str, float], dict[str, BacktestResult]] = {}
    for variant, result in finished:
        groups.setdefault((variant.book, variant.spread_multiple), {})[variant.policy] = result

    output: dict[str, object] = {
        "warning": "Spot, VIX, and rates are real. Option prices and spreads are modeled.",
        "code_version": __version__,
        "history": [history[0].day.isoformat(), history[-1].day.isoformat(), len(history)],
        "n_variants": len(VARIANTS),
        "groups": [],
    }
    for (book, multiple), results in sorted(groups.items()):
        # Sorted, because workers finish in any order and the file should not depend on it.
        results = dict(sorted(results.items()))
        trimmed = {name: r.after_warmup(WARMUP_DAYS) for name, r in results.items()}
        comparisons = [
            compare(trimmed[name], trimmed[baseline]).as_dict()
            for baseline in ("unhedged", "delta_daily")
            for name in sorted(trimmed)
            if name != baseline and baseline in trimmed
        ]
        output["groups"].append(  # type: ignore[attr-defined]
            {
                "book": book,
                "spread_multiple": multiple,
                "stats": [policy_stats(r).as_dict() for _, r in sorted(trimmed.items())],
                "comparisons": comparisons,
                "stress_days": [
                    {"day": s.day, "pnl": s.pnl_by_policy}
                    for s in stress_days(list(trimmed.values()), STRESS_DAYS)
                ],
                "first_failures": {
                    name: r.failures[:3] for name, r in results.items() if r.failures
                },
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
