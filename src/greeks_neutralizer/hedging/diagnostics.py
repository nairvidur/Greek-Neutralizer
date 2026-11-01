"""Stability, frontier, baselines, and dynamic rebalancing policy.

A hedge is only as good as its inputs, and the surface and spot are estimates. Stability asks two
questions: does the held hedge keep working if the inputs were slightly wrong, and would a slightly
different input have produced a very different trade? The second is the sign of a hedge that fits
noise, typically large offsetting positions in nearly identical instruments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from greeks_neutralizer.domain import Position
from greeks_neutralizer.errors import HedgeInfeasibleError, ValidationError
from greeks_neutralizer.hedging.problem import (
    GreekName,
    GreekTarget,
    HedgeConfig,
    HedgeProblem,
    HedgeProposal,
    StabilityReport,
)
from greeks_neutralizer.market import MarketSnapshot
from greeks_neutralizer.pricing.black import FloatArray
from greeks_neutralizer.pricing.book import Exposure, value_book
from greeks_neutralizer.risk.scenarios import Scenario, shock_snapshot

if TYPE_CHECKING:
    from greeks_neutralizer.hedging.optimizer import _Program, _Usable

PERTURBATIONS: tuple[Scenario, ...] = (
    Scenario("spot +0.25%", spot_shock=0.0025),
    Scenario("spot -0.25%", spot_shock=-0.0025),
    Scenario("vol +1pt", vol_shock=0.01),
    Scenario("vol -1pt", vol_shock=-0.01),
    Scenario("skew steeper", skew_shock=-0.02),
    Scenario("skew flatter", skew_shock=0.02),
)

_CONDITION_LIMIT = 1e6


def _penalized_residual(
    targets: Sequence[GreekTarget], by_underlying: dict[str, Exposure]
) -> float:
    total = 0.0
    for target in targets:
        exposure = by_underlying.get(target.underlying, Exposure())
        total += target.effective_weight * abs(float(getattr(exposure, target.greek.value)))
    return total


def stability_report(
    problem: HedgeProblem,
    snapshot: MarketSnapshot,
    lots: NDArray[np.int64],
    usable: list[_Usable],
    program: _Program,
) -> StabilityReport:
    from greeks_neutralizer.hedging.optimizer import _filter_candidates, _Program

    used = np.flatnonzero(lots)
    notes: list[str] = []
    condition = _condition_number(problem, usable, used, program)
    if condition > _CONDITION_LIMIT:
        notes.append(
            f"hedge legs are nearly dependent (condition number {condition:.2e}): "
            "they may be large offsetting positions in similar instruments"
        )

    hedge_positions = [
        Position(usable[int(i)].candidate.instrument, int(lots[int(i)])) for i in used
    ]
    hedged = (*problem.book, *hedge_positions)
    base_residual = _penalized_residual(problem.targets, value_book(hedged, snapshot).by_underlying)

    drift: dict[str, float] = {}
    worst_change = 0.0
    base_lots = max(float(np.abs(lots).sum()), 1.0)
    for perturbation in PERTURBATIONS:
        shocked = shock_snapshot(snapshot, perturbation)
        held = _penalized_residual(problem.targets, value_book(hedged, shocked).by_underlying)
        drift[perturbation.name] = held - base_residual
        try:
            moved_usable, _ = _filter_candidates(problem, shocked)
            book = value_book(problem.book, shocked)
            moved = _Program(
                problem,
                moved_usable,
                dict(book.by_underlying),
                program.book_scenario,
                program.unit_scenario,
            ).solve()
        except HedgeInfeasibleError:
            notes.append(f"infeasible under {perturbation.name}: the bands are fragile")
            worst_change = max(worst_change, 1.0)
            continue
        if moved.lots.shape == lots.shape:
            change = float(np.abs(moved.lots - lots).sum()) / base_lots
            worst_change = max(worst_change, change)
    threshold = problem.config.instability_threshold
    if worst_change > threshold:
        notes.append(
            f"a small input change moves the trade by {worst_change:.0%} of its lots, "
            f"over the {threshold:.0%} limit"
        )
    return StabilityReport(
        condition_number=condition,
        lot_change_ratio=worst_change,
        drift=drift,
        is_stable=not notes,
        notes=tuple(notes),
    )


def _condition_number(
    problem: HedgeProblem,
    usable: list[_Usable],
    used: NDArray[np.intp],
    program: _Program,
) -> float:
    """Condition number of the used legs' Greek matrix, with rows scaled to comparable dollars."""
    if used.size == 0 or not problem.targets:
        return 1.0
    matrix = np.array(
        [
            [
                target.effective_weight * program._exposure_column(target, usable[int(i)])
                for i in used
            ]
            for target in problem.targets
        ],
        dtype=np.float64,
    )
    columns = np.linalg.norm(matrix, axis=0)
    columns[columns == 0.0] = 1.0
    singular = np.linalg.svd(matrix / columns, compute_uv=False)
    smallest = float(singular[-1])
    if smallest <= 0.0:
        return float("inf")
    return float(singular[0]) / smallest


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    label: str
    risk_aversion: float | None
    transaction_cost: float
    gross_lots: int
    n_legs: int
    penalized_residual: float
    worst_scenario_pnl: float | None
    proposal: HedgeProposal | None
    infeasible_reason: str | None = None


def _worst(values: dict[str, float] | None) -> float | None:
    return min(values.values()) if values else None


def hedge_frontier(
    problem: HedgeProblem,
    snapshot: MarketSnapshot,
    risk_aversions: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
) -> tuple[FrontierPoint, ...]:
    """Cost against residual risk, always including the baselines a user must be able to beat.

    Doing nothing and delta-only are always present, so a recommendation can be judged against the
    trivial alternatives.
    """
    from greeks_neutralizer.hedging.optimizer import solve_hedge

    book = value_book(problem.book, snapshot)
    before_scenarios = {}
    if problem.scenarios:
        from greeks_neutralizer.risk.scenarios import book_scenario_pnl

        pnl = book_scenario_pnl(problem.book, snapshot, problem.scenarios)
        before_scenarios = {s.name: float(v) for s, v in zip(problem.scenarios, pnl, strict=True)}
    points = [
        FrontierPoint(
            label="do nothing",
            risk_aversion=None,
            transaction_cost=0.0,
            gross_lots=0,
            n_legs=0,
            penalized_residual=_penalized_residual(problem.targets, dict(book.by_underlying)),
            worst_scenario_pnl=_worst(before_scenarios),
            proposal=None,
        )
    ]
    delta_only = _delta_only_problem(problem)
    if delta_only is not None:
        points.append(_point("delta only", None, delta_only, snapshot, problem.targets))
    for aversion in risk_aversions:
        adjusted = replace(problem, config=replace(problem.config, risk_aversion=aversion))
        try:
            proposal = solve_hedge(adjusted, snapshot)
        except HedgeInfeasibleError as exc:
            points.append(
                FrontierPoint(
                    f"risk aversion {aversion:g}",
                    aversion,
                    0.0,
                    0,
                    0,
                    float("nan"),
                    None,
                    None,
                    "; ".join(exc.binding) or str(exc),
                )
            )
            continue
        points.append(_from_proposal(f"risk aversion {aversion:g}", aversion, proposal, problem))
    return tuple(points)


def _delta_only_problem(problem: HedgeProblem) -> HedgeProblem | None:
    """Same book and scenarios, but only linear hedges, with a very strong penalty on delta."""
    from greeks_neutralizer.domain import LinearInstrument

    linear = tuple(c for c in problem.candidates if isinstance(c.instrument, LinearInstrument))
    if not linear:
        return None
    underlyings = sorted({c.instrument.underlying for c in linear})
    targets = tuple(GreekTarget(u, GreekName.DELTA, weight=1.0) for u in underlyings)
    config = HedgeConfig(
        risk_aversion=1.0,
        max_quote_age=problem.config.max_quote_age,
        require_quote_size=problem.config.require_quote_size,
    )
    return HedgeProblem(problem.book, linear, targets, problem.scenarios, config)


def _point(
    label: str,
    aversion: float | None,
    problem: HedgeProblem,
    snapshot: MarketSnapshot,
    report_targets: Sequence[GreekTarget],
) -> FrontierPoint:
    from greeks_neutralizer.hedging.optimizer import solve_hedge

    proposal = solve_hedge(problem, snapshot)
    return _from_proposal(
        label, aversion, proposal, replace(problem, targets=tuple(report_targets))
    )


def _from_proposal(
    label: str, aversion: float | None, proposal: HedgeProposal, problem: HedgeProblem
) -> FrontierPoint:
    return FrontierPoint(
        label=label,
        risk_aversion=aversion,
        transaction_cost=proposal.transaction_cost,
        gross_lots=proposal.gross_lots,
        n_legs=len(proposal.legs),
        penalized_residual=_penalized_residual(problem.targets, dict(proposal.after)),
        worst_scenario_pnl=_worst(dict(proposal.scenario_pnl_after)),
        proposal=proposal,
    )


@dataclass(frozen=True, slots=True)
class NoTradeBand:
    """Whalley-Wilmott: delta tolerated without trading, in shares either side of target."""

    half_width_shares: float
    target_delta_shares: float

    def should_rebalance(self, delta_shares: float) -> bool:
        return abs(delta_shares - self.target_delta_shares) > self.half_width_shares


def whalley_wilmott_band(
    gamma_shares: float,
    spot: float,
    proportional_cost: float,
    risk_aversion: float,
    discount: float = 1.0,
    *,
    target_delta_shares: float = 0.0,
) -> NoTradeBand:
    """Asymptotic no-trade half-width, (3/2 * D * cost * S * gamma^2 / risk_aversion)^(1/3).

    `gamma_shares` is the book's total change in delta shares per $1 of spot. `proportional_cost` is
    the one-way cost as a fraction of notional. `risk_aversion` is the exponential utility
    coefficient per dollar. The asymptotic is for small costs. It says more gamma means a wider band
    in absolute delta, and a lower cost means trading more often.
    """
    if spot <= 0.0 or proportional_cost < 0.0 or risk_aversion <= 0.0 or discount <= 0.0:
        raise ValidationError("invalid inputs to the no-trade band")
    width = 1.5 * discount * proportional_cost * spot * gamma_shares * gamma_shares / risk_aversion
    return NoTradeBand(float(np.cbrt(width)), target_delta_shares)


@dataclass(frozen=True, slots=True)
class DecayStep:
    days_forward: float
    exposure: Exposure


def hedge_decay_forecast(
    positions: Sequence[Position],
    snapshot: MarketSnapshot,
    underlying: str,
    days: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 21.0),
) -> tuple[DecayStep, ...]:
    """How the book's exposures drift as time passes with spot and vol unchanged."""
    steps: list[DecayStep] = []
    for day in days:
        rolled = shock_snapshot(snapshot, Scenario(f"+{day:g}d", days_forward=day))
        book = value_book(positions, rolled)
        steps.append(DecayStep(day, book.by_underlying.get(underlying, Exposure())))
    return tuple(steps)


def effective_number_of_bets(values: FloatArray) -> float:
    """Herfindahl-based concentration of absolute exposures. 1 means all in one position."""
    weights = np.abs(np.asarray(values, dtype=np.float64))
    total = float(weights.sum())
    if total == 0.0:
        return 0.0
    shares = weights / total
    return float(1.0 / np.sum(shares * shares))
