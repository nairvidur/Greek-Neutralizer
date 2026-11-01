"""Backtest statistics, with confidence intervals so noise is not reported as an edge."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np

from greeks_neutralizer.backtest.engine import BacktestResult
from greeks_neutralizer.errors import ValidationError
from greeks_neutralizer.pricing.black import FloatArray
from greeks_neutralizer.risk.measures import expected_shortfall


@dataclass(frozen=True, slots=True)
class PolicyStats:
    policy: str
    days: int
    total_pnl: float
    daily_stdev: float
    worst_day: float
    es_975: float
    max_drawdown: float
    hedge_cost: float
    book_cost: float
    hedge_lots: int
    failed_days: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def max_drawdown(pnl: FloatArray) -> float:
    equity = np.cumsum(pnl)
    return float(np.max(np.maximum.accumulate(np.maximum(equity, 0.0)) - equity))


def policy_stats(result: BacktestResult) -> PolicyStats:
    pnl = result.pnl()
    if pnl.size == 0:
        raise ValidationError(f"{result.policy}: no days to report on")
    return PolicyStats(
        policy=result.policy,
        days=int(pnl.size),
        total_pnl=float(pnl.sum()),
        daily_stdev=float(pnl.std(ddof=1)),
        worst_day=float(np.min(pnl)),
        es_975=expected_shortfall(pnl, 0.975),
        max_drawdown=max_drawdown(pnl),
        hedge_cost=sum(d.hedge_cost for d in result.days),
        book_cost=sum(d.book_cost for d in result.days),
        hedge_lots=sum(d.hedge_lots for d in result.days),
        failed_days=len(result.failures),
    )


@dataclass(frozen=True, slots=True)
class Comparison:
    """How `policy` compares with `baseline` on the same days, with block-bootstrap intervals.

    Variance reduction is 1 - var(policy) / var(baseline), and ES improvement is the fall in 97.5%
    expected shortfall in dollars. Intervals are 95% from a circular block bootstrap, so
    autocorrelated and clustered days are resampled together. If an interval includes zero, the data
    do not establish a difference.
    """

    policy: str
    baseline: str
    variance_reduction: float
    variance_reduction_ci: tuple[float, float]
    es_improvement: float
    es_improvement_ci: tuple[float, float]
    mean_pnl_difference_per_day: float
    mean_pnl_difference_ci: tuple[float, float]
    policy_failed_days: int = 0
    baseline_failed_days: int = 0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _block_indices(
    n: int, block: int, draws: int, rng: np.random.Generator
) -> np.ndarray[tuple[int, int], np.dtype[np.intp]]:
    blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets) % n
    return np.asarray(idx.reshape(draws, -1)[:, :n], dtype=np.intp)


def compare(
    result: BacktestResult,
    baseline: BacktestResult,
    *,
    block: int = 20,
    draws: int = 2000,
    seed: int = 20261101,
) -> Comparison:
    a, b = result.pnl(), baseline.pnl()
    if a.shape != b.shape:
        raise ValidationError("comparison needs results over the same days")
    if result.dates() != baseline.dates():
        raise ValidationError("comparison needs results on the same dates, and these differ")
    if a.size < 2 * block:
        raise ValidationError(f"need at least {2 * block} days, got {a.size}")
    rng = np.random.default_rng(seed)
    idx = _block_indices(a.size, block, draws, rng)
    sample_a, sample_b = a[idx], b[idx]

    var_a = sample_a.var(axis=1, ddof=1)
    var_b = sample_b.var(axis=1, ddof=1)
    reduction = 1.0 - var_a / var_b

    def es(samples: FloatArray) -> FloatArray:
        k = max(int(np.floor(samples.shape[1] * 0.025)), 1)
        return np.asarray(-np.sort(samples, axis=1)[:, :k].mean(axis=1), dtype=np.float64)

    es_gain = es(sample_b) - es(sample_a)
    mean_diff = sample_a.mean(axis=1) - sample_b.mean(axis=1)

    def ci(values: FloatArray) -> tuple[float, float]:
        low, high = np.quantile(values, [0.025, 0.975])
        return float(low), float(high)

    return Comparison(
        policy=result.policy,
        baseline=baseline.policy,
        variance_reduction=float(1.0 - a.var(ddof=1) / b.var(ddof=1)),
        variance_reduction_ci=ci(reduction),
        es_improvement=expected_shortfall(b, 0.975) - expected_shortfall(a, 0.975),
        es_improvement_ci=ci(es_gain),
        mean_pnl_difference_per_day=float(a.mean() - b.mean()),
        mean_pnl_difference_ci=ci(mean_diff),
        policy_failed_days=len(result.failures),
        baseline_failed_days=len(baseline.failures),
    )


@dataclass(frozen=True, slots=True)
class StressDay:
    day: str
    pnl_by_policy: dict[str, float]


def stress_days(results: Sequence[BacktestResult], days: Sequence[date]) -> tuple[StressDay, ...]:
    """Each policy's P&L on specific dates, read from the results. Absent dates are omitted."""
    out: list[StressDay] = []
    for target in days:
        pnl = {}
        for result in results:
            for row in result.days:
                if row.as_of.date() == target:
                    pnl[result.policy] = row.pnl
                    break
        if pnl:
            out.append(StressDay(target.isoformat(), pnl))
    return tuple(out)
