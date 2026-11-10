#!/usr/bin/env python3
"""Run the pre-registered backtest on historical option chains. Every variant is reported.

Usage: run_chain_backtest.py <chain_dir> <dividends.csv> <DGS3MO.csv> --out results.json
       [--symbol SPY] [--start YYYY-MM-DD]

<chain_dir> holds one <date>.csv per day from scripts/fetch_dolthub_chains.py, and DGS3MO.csv is
FRED's 3-month Treasury series. The books, policies, and spread multiples are the proxy's
pre-registered ones, mapped to SPY at the same notional: one SPX point is about ten SPY dollars,
so 100 SPY lots stand in for 10 SPX lots. They were fixed before this script first ran, and
changing them means recording the reason in docs/evidence/phase4/ledger.md.

Option bid and ask are real, so a spread multiple of 1 is the quoted spread, and 3 is a stress.

The position-aware optimizer variants were added after the SPY no-go and designed on SPY, so their
SPY results are in-sample. Their test is on DIA and XLF, held out until the variants were entered
in the ledger. Every symbol uses the same book sizes and settings.

`--start` drops chain days before a date, which is how the forward pilot in
docs/evidence/phase6/ scores only days that did not exist when it was registered. A run too short
for the block bootstrap reports its statistics without intervals, rather than failing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import subprocess
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
from greeks_neutralizer.backtest.policies import POSITION_AWARE_POLICIES, registered_policy
from greeks_neutralizer.backtest.proxy import load_fred_series
from greeks_neutralizer.backtest.report import compare, policy_stats, stress_days
from greeks_neutralizer.errors import GNError

DEFAULT_SYMBOL = "SPY"
# The history starts at the first chain day after which no gap to the next exceeds this. SPY's
# source has one chain day in 2019, then none until 2020, which made a first "period" of eight
# months that was about 40% of daily delta's variance in the first chain run.
MAX_GAP_DAYS = 15
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
    # Added after the SPY no-go, as entered in the ledger before the held-out runs.
    *(
        Variant(book, policy, multiple)
        for book, multiple in (
            ("short_strangle", 1.0),
            ("short_strangle", 3.0),
            ("long_put_overlay", 1.0),
        )
        for policy in POSITION_AWARE_POLICIES
    ),
)


@dataclass(frozen=True, slots=True)
class Inputs:
    symbol: str
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


def book_for(name: str, symbol: str) -> BookStrategy:
    if name == "short_strangle":
        return ChainShortStrangle(symbol, lots=100)
    if name == "long_put_overlay":
        return ChainPutOverlay(symbol, notional_shares=100_000)
    raise ValueError(name)


def continuous_history(files: Sequence[Path], max_gap_days: int = MAX_GAP_DAYS) -> int:
    """Index of the first chain day after which no gap to the next day exceeds max_gap_days."""
    days = [date.fromisoformat(p.stem) for p in files]
    start = 0
    for i in range(1, len(days)):
        if (days[i] - days[i - 1]).days > max_gap_days:
            start = i
    return start


def select_history(files: Sequence[Path], start: date | None) -> tuple[list[Path], list[Path]]:
    """The chain files scored, and those dropped by the gap rule. Days before `start` are not
    data for this run at all, so the gap rule sees only the rest."""
    kept = [p for p in files if start is None or date.fromisoformat(p.stem) >= start]
    first = continuous_history(kept)
    return kept[first:], kept[:first]


def inputs_digest(paths: Sequence[Path]) -> str:
    """SHA-256 over every input file's name and bytes, in name order. The free sources revise
    their data, so a readout records exactly what it read."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: (p.name, str(p))):
        digest.update(path.name.encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def code_identity(folder: Path) -> tuple[str | None, bool | None]:
    """The commit checked out where `folder` lives, and whether tracked files under src/ or
    scripts/ differ from it. Both are None without git or outside a git checkout."""

    def git(*argv: str) -> str:
        return subprocess.run(
            ["git", "-C", str(folder), *argv], capture_output=True, text=True, check=True
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        changes = git("status", "--porcelain", "--untracked-files=no", "--", ":/src", ":/scripts")
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(changes.strip())


def comparisons_for(
    results: dict[str, BacktestResult], block: int
) -> tuple[list[dict[str, object]], str | None]:
    """Each policy against unhedged and daily delta, or none, with the reason, while the run is
    shorter than the block bootstrap needs."""
    periods = min(r.pnl().size for r in results.values())
    if periods < 2 * block:
        return [], (
            f"{periods} periods, fewer than the {2 * block} the block bootstrap needs, so no "
            "comparison or interval is reported yet"
        )
    return [
        compare(results[name], results[baseline], block=block).as_dict()
        for baseline in ("unhedged", "delta_daily")
        for name in sorted(results)
        if name != baseline and baseline in results
    ], None


def _run(args: tuple[Variant, Inputs]) -> Outcome:
    variant, inputs = args
    start = time.perf_counter()
    market = ChainMarket(
        inputs.symbol,
        inputs.files,
        inputs.rates_pct,
        inputs.dividends,
        spread_multiple=variant.spread_multiple,
    )
    result = run_backtest(
        market,
        book_for(variant.book, inputs.symbol),
        registered_policy(variant.policy, inputs.symbol),
    )
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
    market = ChainMarket(inputs.symbol, (path,), inputs.rates_pct, inputs.dividends)
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
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        help="score only chain days on or after this date, as the forward pilot does",
    )
    parser.add_argument(
        "--series",
        type=Path,
        help="Also write each variant's per-period results to this directory, as CSV",
    )
    parser.add_argument(
        "--variants",
        help="Run only these, as book:policy:multiple separated by commas. For diagnostics "
        "only: a registered result runs every variant, n_variants in the output, and a subset "
        "run records how many ran in variants_run",
    )
    args = parser.parse_args()

    every = tuple(sorted(args.chains.glob("*.csv")))
    if not every:
        raise SystemExit(f"no chain files in {args.chains}")
    files, dropped = select_history(every, args.start)
    if len(files) < 3:
        raise SystemExit(f"fewer than two periods of continuous chain history in {args.chains}")
    inputs = Inputs(
        args.symbol, tuple(files), load_fred_series(args.rates), load_dividends(args.dividends)
    )
    print(
        f"{args.symbol}: {len(files)} chain days, {files[0].stem} to {files[-1].stem}, "
        f"{len(dropped)} leading days dropped",
        flush=True,
    )

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

    git_commit, git_dirty = code_identity(Path(__file__).resolve().parent)
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
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "symbol": args.symbol,
        "start": args.start.isoformat() if args.start else None,
        "inputs_sha256": inputs_digest([*files, args.dividends, args.rates]),
        "chain_days": [files[0].stem, files[-1].stem, len(files)],
        "dropped_leading_days": {
            "rule": f"history starts after the last gap over {MAX_GAP_DAYS} calendar days",
            "days": [p.stem for p in dropped],
        },
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
        comparisons, comparisons_note = comparisons_for(trimmed, BOOTSTRAP_BLOCK)
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
                "comparisons_note": comparisons_note,
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
