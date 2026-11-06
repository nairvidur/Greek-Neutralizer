#!/usr/bin/env python3
"""Run the pre-registered backtest on historical option chains. Every variant is reported.

Usage: run_chain_backtest.py <chain_dir> <dividends.csv> <DGS3MO.csv> --out results.json

<chain_dir> holds one <date>.csv per day from scripts/fetch_dolthub_chains.py, and DGS3MO.csv is
FRED's 3-month Treasury series. The books, policies, and spread multiples are the proxy's
pre-registered ones, mapped to SPY at the same notional: one SPX point is about ten SPY dollars,
so 100 SPY lots stand in for 10 SPX lots. They were fixed before this script first ran, and
changing them means recording the reason in docs/evidence/phase4/ledger.md.

Option bid and ask are real, so a spread multiple of 1 is the quoted spread, and 3 is a stress.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from greeks_neutralizer._version import __version__
from greeks_neutralizer.backtest.chains import (
    ChainMarket,
    ChainPutOverlay,
    ChainShortStrangle,
    MarkAudit,
    SkippedDay,
    load_chain_day,
    load_dividends,
)
from greeks_neutralizer.backtest.engine import BacktestResult, BookStrategy, run_backtest
from greeks_neutralizer.backtest.policies import registered_policy
from greeks_neutralizer.backtest.proxy import load_fred_series
from greeks_neutralizer.backtest.report import compare, policy_stats, stress_days
from greeks_neutralizer.errors import GNError

SYMBOL = "SPY"
# The chain market has no trailing state to form, unlike the proxy's long-run vol level.
WARMUP_DAYS = 0
# The proxy's stress dates that fall inside the chain history, fixed in advance.
STRESS_DAYS = (date(2020, 3, 16), date(2024, 8, 5))
# Block length in observations. Chain days are about three a week, so this is about 7 weeks,
# against the proxy's 20 trading days, to keep clustered stress periods inside one block.
BOOTSTRAP_BLOCK = 20
# Every fifth chain day is audited for out-of-sample mark error.
AUDIT_EVERY = 5


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


@dataclass(frozen=True, slots=True)
class Inputs:
    files: tuple[Path, ...]
    rates_pct: dict[date, float]
    dividends: tuple[tuple[date, float], ...]


@dataclass(frozen=True, slots=True)
class Outcome:
    variant: Variant
    result: BacktestResult
    skipped: tuple[SkippedDay, ...]
    modeled_quotes: int
    close_gaps: tuple[float, ...]
    seconds: float


def book_for(name: str) -> BookStrategy:
    if name == "short_strangle":
        return ChainShortStrangle(SYMBOL, lots=100)
    if name == "long_put_overlay":
        return ChainPutOverlay(SYMBOL, notional_shares=100_000)
    raise ValueError(name)


def _run(args: tuple[Variant, Inputs]) -> Outcome:
    variant, inputs = args
    start = time.perf_counter()
    market = ChainMarket(
        SYMBOL,
        inputs.files,
        inputs.rates_pct,
        inputs.dividends,
        spread_multiple=variant.spread_multiple,
    )
    result = run_backtest(market, book_for(variant.book), registered_policy(variant.policy, SYMBOL))
    return Outcome(
        variant,
        result,
        tuple(market.skipped),
        len(market.modeled_quotes),
        tuple(market.close_gaps),
        time.perf_counter() - start,
    )


def _audit(args: tuple[Path, Inputs]) -> MarkAudit | str:
    path, inputs = args
    market = ChainMarket(SYMBOL, (path,), inputs.rates_pct, inputs.dividends)
    try:
        return market.mark_audit(load_chain_day(path))
    except GNError as exc:
        return f"{path.stem}: {type(exc).__name__}: {exc}"


def summarize_errors(pairs: Sequence[tuple[float, float]]) -> dict[str, float | int]:
    """Model minus quoted mid, per share, against the quoted half-spread."""
    if not pairs:
        return {"contracts": 0}
    error = np.array([e for e, _ in pairs])
    half = np.array([h for _, h in pairs])
    return {
        "contracts": int(error.size),
        "median_abs_error": float(np.median(np.abs(error))),
        "p90_abs_error": float(np.quantile(np.abs(error), 0.9)),
        "mean_error": float(error.mean()),
        "rms_error": float(np.sqrt(np.mean(error**2))),
        "median_half_spread": float(np.median(half)),
        "share_within_half_spread": float(np.mean(np.abs(error) <= half)),
    }


def mark_noise_variance(result: BacktestResult, rms_error: float) -> float:
    """Mean P&L variance per period that model-mark errors of this size would add on their own,
    treating each mark's error as independent. An estimate, used only to read the results."""
    return float(np.mean([d.modeled_units_sq for d in result.days])) * rms_error**2


def noise_adjusted_reduction(
    policy: BacktestResult, baseline: BacktestResult, rms_error: float
) -> float:
    """1 - var(policy) / var(baseline), with each variance net of its estimated mark noise."""
    a = float(np.var(policy.pnl(), ddof=1)) - mark_noise_variance(policy, rms_error)
    b = float(np.var(baseline.pnl(), ddof=1)) - mark_noise_variance(baseline, rms_error)
    return 1.0 - a / b if b > 0.0 else float("nan")


def model_marked_share(result: BacktestResult) -> float:
    held = sum(d.option_positions for d in result.days)
    return sum(d.model_marked for d in result.days) / held if held else 0.0


def select_variants(spec: str | None) -> tuple[Variant, ...]:
    if not spec:
        return VARIANTS
    wanted = []
    for item in spec.split(","):
        book, policy, multiple = item.split(":")
        variant = Variant(book, policy, float(multiple))
        if variant not in VARIANTS:
            raise SystemExit(f"not a registered variant: {item}")
        wanted.append(variant)
    return tuple(wanted)


SERIES_COLUMNS = (
    "as_of",
    "pnl",
    "book_pnl",
    "hedge_pnl",
    "financing",
    "carry_income",
    "hedge_cost",
    "hedge_option_cost",
    "book_cost",
    "hedge_option_contracts",
    "hedge_lots",
    "model_marked",
    "option_positions",
    "modeled_units_sq",
)


def write_series(folder: Path, outcome: Outcome) -> None:
    v = outcome.variant
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{v.book}__{v.policy}__{v.spread_multiple:g}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SERIES_COLUMNS)
        for day in outcome.result.days:
            writer.writerow(
                [day.as_of.date().isoformat(), *(getattr(day, c) for c in SERIES_COLUMNS[1:])]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chains", type=Path)
    parser.add_argument("dividends", type=Path)
    parser.add_argument("rates", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument(
        "--series",
        type=Path,
        help="Also write each variant's per-period results to this directory, as CSV",
    )
    parser.add_argument(
        "--variants",
        help="Run only these, as book:policy:multiple separated by commas. For reruns and "
        "diagnostics; the registered result is the full set",
    )
    args = parser.parse_args()

    files = tuple(sorted(args.chains.glob("*.csv")))
    if not files:
        raise SystemExit(f"no chain files in {args.chains}")
    inputs = Inputs(files, load_fred_series(args.rates), load_dividends(args.dividends))
    print(f"{len(files)} chain days, {files[0].stem} to {files[-1].stem}", flush=True)

    selected = select_variants(args.variants)
    outcomes: list[Outcome] = []
    with mp.Pool(args.processes) as pool:
        for outcome in pool.imap_unordered(_run, [(v, inputs) for v in selected]):
            print(
                f"{outcome.variant} {outcome.seconds:.0f}s days={len(outcome.result.days)} "
                f"failures={len(outcome.result.failures)} skipped={len(outcome.skipped)} "
                f"modeled_quotes={outcome.modeled_quotes}",
                flush=True,
            )
            outcomes.append(outcome)
        audits = pool.map(_audit, [(path, inputs) for path in files[::AUDIT_EVERY]])

    skipped_sets = {tuple(s.day for s in o.skipped) for o in outcomes}
    if len(skipped_sets) != 1:
        raise SystemExit("variants skipped different days, so they are not comparable")
    skipped = outcomes[0].skipped
    reasons = Counter(s.reason.split(":")[0] for s in skipped)
    scored = [a for a in audits if isinstance(a, MarkAudit)]
    strike_errors = summarize_errors([x for a in scored for x in a.strike_holdout])
    expiry_errors = summarize_errors([x for a in scored for x in a.expiry_holdout])
    noise_levels = {
        "strike_held_out": float(strike_errors.get("rms_error", 0.0)),
        "expiry_held_out": float(expiry_errors.get("rms_error", 0.0)),
    }
    gaps = np.abs(np.array(outcomes[0].close_gaps))

    groups: dict[tuple[str, float], dict[str, Outcome]] = {}
    for outcome in outcomes:
        key = (outcome.variant.book, outcome.variant.spread_multiple)
        groups.setdefault(key, {})[outcome.variant.policy] = outcome

    output: dict[str, object] = {
        "warning": (
            "Option bid and ask are real end-of-day quotes from a free source. Quote sizes, fees, "
            "and future dividend amounts are assumed. P&L periods run from one chain day to the "
            "next, about three a week, so per-period figures are not daily ones."
        ),
        "source": "DoltHub post-no-preference/options and post-no-preference/stocks; FRED DGS3MO",
        "code_version": __version__,
        "chain_days": [files[0].stem, files[-1].stem, len(files)],
        "skipped_days": len(skipped),
        "skipped_by_reason": dict(sorted(reasons.items())),
        "first_skipped": [f"{s.day}: {s.reason}" for s in skipped[:5]],
        "close_gap": {
            "note": "Parity spot over the close, minus one. Spot comes from parity.",
            "median_abs": float(np.median(gaps)),
            "p99_abs": float(np.quantile(gaps, 0.99)),
            "max_abs": float(gaps.max()),
        },
        "mark_audit": {
            "note": (
                "Out-of-sample marks, as the backtest uses for contracts the source did not "
                "quote: model minus quoted mid, in dollars a share."
            ),
            "days_audited": len(scored),
            "days_not_auditable": len(audits) - len(scored),
            "first_not_auditable": [a for a in audits if isinstance(a, str)][:3],
            "strike_held_out": strike_errors,
            "expiry_held_out": expiry_errors,
        },
        "n_variants": len(VARIANTS),
        "variants_run": len(selected),
        "bootstrap_block": BOOTSTRAP_BLOCK,
        "groups": [],
    }
    for (book, multiple), by_policy in sorted(groups.items()):
        # Sorted, because workers finish in any order and the file should not depend on it.
        ordered = dict(sorted(by_policy.items()))
        trimmed = {name: o.result.after_warmup(WARMUP_DAYS) for name, o in ordered.items()}
        comparisons = [
            compare(trimmed[name], trimmed[baseline], block=BOOTSTRAP_BLOCK).as_dict()
            for baseline in ("unhedged", "delta_daily")
            for name in sorted(trimmed)
            if name != baseline and baseline in trimmed
        ]
        output["groups"].append(  # type: ignore[attr-defined]
            {
                "book": book,
                "spread_multiple": multiple,
                "stats": [
                    {
                        **policy_stats(r).as_dict(),
                        "model_marked_share": model_marked_share(r),
                        "modeled_quotes": ordered[name].modeled_quotes,
                        "mark_noise_variance_share": {
                            level: mark_noise_variance(r, rms) / float(np.var(r.pnl(), ddof=1))
                            for level, rms in noise_levels.items()
                        },
                    }
                    for name, r in trimmed.items()
                ],
                "noise_adjusted_variance_reduction_vs_delta": {
                    name: {
                        level: noise_adjusted_reduction(r, trimmed["delta_daily"], rms)
                        for level, rms in noise_levels.items()
                    }
                    for name, r in trimmed.items()
                    if name != "delta_daily" and "delta_daily" in trimmed
                },
                "comparisons": comparisons,
                "stress_days": [
                    {"day": s.day, "pnl": s.pnl_by_policy}
                    for s in stress_days(list(trimmed.values()), STRESS_DAYS)
                ],
                "first_failures": {
                    name: o.result.failures[:3] for name, o in ordered.items() if o.result.failures
                },
            }
        )
    if args.series is not None:
        for outcome in outcomes:
            write_series(args.series, outcome)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
