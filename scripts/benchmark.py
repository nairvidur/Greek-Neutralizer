#!/usr/bin/env python3
"""Measure the provisional performance budgets. Repeats each case and reports percentiles.

Usage: benchmark.py [--out results.json]

Budgets being tested are provisional targets from the product brief, not achievements. Run on an
otherwise idle machine, since results from a busy one say nothing.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from greeks_neutralizer._version import __version__
from greeks_neutralizer.domain import (
    ExerciseStyle,
    LinearInstrument,
    OptionContract,
    OptionRight,
    Position,
    Quote,
    SettlementKind,
)
from greeks_neutralizer.hedging.optimizer import solve_hedge
from greeks_neutralizer.hedging.problem import (
    GreekName,
    GreekTarget,
    HedgeCandidate,
    HedgeConfig,
    HedgeProblem,
)
from greeks_neutralizer.market import DiscountCurve, FlatVol, MarketSnapshot, UnderlyingMarket
from greeks_neutralizer.pricing.black import black_greeks
from greeks_neutralizer.pricing.book import value_options
from greeks_neutralizer.risk.scenarios import HISTORICAL_STRESSES, book_scenario_pnl

AS_OF = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
SNAPSHOT = MarketSnapshot(
    AS_OF, {"XYZ": UnderlyingMarket("XYZ", 100.0, DiscountCurve.flat(0.05), FlatVol(0.25))}
)


def contract(index: int, style: ExerciseStyle) -> OptionContract:
    right = OptionRight.CALL if index % 2 == 0 else OptionRight.PUT
    days = 7 + (index % 360)
    strike = 60.0 + (index % 800) * 0.1
    settlement = SettlementKind.CASH if style is ExerciseStyle.EUROPEAN else SettlementKind.PHYSICAL
    return OptionContract(
        "XYZ", right, strike, AS_OF + timedelta(days=days), style, 100.0, settlement
    )


def measure(fn: Callable[[], object], repeats: int) -> dict[str, float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    data = np.array(samples)
    return {
        "repeats": repeats,
        "p50_ms": float(np.percentile(data, 50) * 1e3),
        "p99_ms": float(np.percentile(data, 99) * 1e3),
        "max_ms": float(data.max() * 1e3),
    }


def universe(n: int) -> tuple[HedgeCandidate, ...]:
    out = [
        HedgeCandidate(LinearInstrument("XYZ", "XYZ"), Quote(99.99, 100.01, 10**6, 10**6, AS_OF))
    ]
    prices = value_options(
        [contract(i, ExerciseStyle.EUROPEAN) for i in range(n)], SNAPSHOT
    ).greeks.price
    for i in range(n):
        mid = max(float(prices[i]), 0.03)
        out.append(
            HedgeCandidate(
                contract(i, ExerciseStyle.EUROPEAN),
                Quote(max(mid - 0.02, 0.01), mid + 0.02, 500, 500, AS_OF),
            )
        )
    return tuple(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    n = 100_000
    strikes = np.linspace(60.0, 140.0, n)
    times = np.linspace(0.01, 2.0, n)
    omega = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)

    euro_100k = [contract(i, ExerciseStyle.EUROPEAN) for i in range(100_000)]
    euro_5k = euro_100k[:5000]
    amer_1k = [contract(i, ExerciseStyle.AMERICAN) for i in range(1000)]
    amer_5k = [contract(i, ExerciseStyle.AMERICAN) for i in range(5000)]

    book = (
        Position(contract(100, ExerciseStyle.EUROPEAN), -50),
        Position(contract(101, ExerciseStyle.EUROPEAN), -50),
    )
    targets = tuple(
        GreekTarget("XYZ", g) for g in (GreekName.DELTA, GreekName.GAMMA, GreekName.VEGA)
    )
    big_book = [Position(c, 1 + i % 5) for i, c in enumerate(euro_5k)]

    def hedge(size: int, stability: bool = False) -> Callable[[], object]:
        candidates = universe(size)
        problem = HedgeProblem(book, candidates, targets, (), HedgeConfig(max_legs=6))
        return lambda: solve_hedge(problem, SNAPSHOT, run_stability=stability)

    cases: dict[str, tuple[Callable[[], object], int, str | None]] = {
        "kernel_100k_vectorized": (
            lambda: black_greeks(100.0, strikes, times, 0.05, 0.0, 0.25, omega),
            20,
            None,
        ),
        "value_100k_european_contracts": (
            lambda: value_options(euro_100k, SNAPSHOT),
            3,
            "under 1000 ms",
        ),
        "value_5k_european_contracts": (lambda: value_options(euro_5k, SNAPSHOT), 10, None),
        "value_1k_american_contracts": (lambda: value_options(amer_1k, SNAPSHOT), 3, None),
        "value_5k_american_contracts": (lambda: value_options(amer_5k, SNAPSHOT), 2, None),
        "stress_5k_positions_7_scenarios": (
            lambda: book_scenario_pnl(big_book, SNAPSHOT, HISTORICAL_STRESSES),
            3,
            None,
        ),
        "hedge_100_candidates": (hedge(100), 5, None),
        "hedge_1000_candidates": (hedge(1000), 5, None),
        "hedge_5000_candidates": (hedge(5000), 3, "under 1000 ms at p99"),
        "hedge_100_candidates_with_stability": (hedge(100, True), 2, None),
    }
    results: dict[str, object] = {
        "code_version": __version__,
        "machine": platform.machine(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "note": "Single process. p99 from few repeats is close to the maximum, not a real tail.",
        "cases": {},
    }
    for name, (fn, repeats, budget) in cases.items():
        stats = measure(fn, repeats)
        stats_with_budget: dict[str, object] = dict(stats)
        if budget is not None:
            stats_with_budget["budget"] = budget
        results["cases"][name] = stats_with_budget  # type: ignore[index]
        print(f"{name}: p50 {stats['p50_ms']:.1f} ms, max {stats['max_ms']:.1f} ms", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
