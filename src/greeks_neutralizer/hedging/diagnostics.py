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

from greeks_neutralizer.domain import LinearInstrument, Position, require_symbol
from greeks_neutralizer.errors import (
    HedgeInfeasibleError,
    SearchLimitError,
    SolverRangeError,
    ValidationError,
)
from greeks_neutralizer.hedging.problem import (
    GreekName,
    GreekTarget,
    HedgeConfig,
    HedgeProblem,
    HedgeProposal,
    StabilityReport,
)
from greeks_neutralizer.market import MarketSnapshot
from greeks_neutralizer.pricing.book import Exposure, value_book
from greeks_neutralizer.risk.scenarios import (
    Scenario,
    book_scenario_pnl,
    scenario_pnl_matrix,
    shock_snapshot,
)

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

_MIN_LOTS_FOR_RATIO = 10.0


def _penalized_residual(
    targets: Sequence[GreekTarget], by_underlying: dict[str, Exposure]
) -> float:
    total = 0.0
    for target in targets:
        exposure = by_underlying.get(target.underlying, Exposure())
        total += target.effective_weight * abs(float(getattr(exposure, target.greek.value)))
    return total


def _contract_lots(usable: Sequence[_Usable], lots: NDArray[np.int64]) -> dict[str, int]:
    """Lots by instrument key, leaving out fine-grained linear hedges.

    Shares and contracts are different units, and a share hedge should move with spot: that is its
    job, not instability. Keys rather than positions, because a perturbed market can change which
    candidates are usable.
    """
    out: dict[str, int] = {}
    for row, quantity in zip(usable, lots, strict=True):
        instrument = row.candidate.instrument
        fine = isinstance(instrument, LinearInstrument) and instrument.multiplier == 1.0
        if quantity != 0 and not fine:
            out[instrument.key] = int(quantity)
    return out


def stability_report(
    problem: HedgeProblem,
    snapshot: MarketSnapshot,
    lots: NDArray[np.int64],
    usable: list[_Usable],
    program: _Program,
) -> StabilityReport:
    """Re-solve under small market perturbations and measure how far the option trade moves.

    A hedge is called unstable only if the trade moves and that move matters. Near-substitutes can
    swap freely at no cost: two calls on one expiry have exactly proportional gamma and vega, which
    also makes the Greek matrix singular for a sensible hedge. So `condition_number` is reported
    for inspection only, and `regret_ratio` measures what it would cost to be wrong.
    """
    from greeks_neutralizer.hedging.optimizer import _filter_candidates, _Program

    used = np.flatnonzero(lots)
    notes: list[str] = []
    hit_time_limit = False
    # Only what the base solve accepted. The re-solves skip the quote-to-model gates because the
    # market is shocked on purpose, and a quote rejected today must not come back through that gap.
    accepted = replace(problem, candidates=tuple(row.candidate for row in usable))
    condition = _condition_number(problem, usable, used, program)

    hedge_positions = [
        Position(usable[int(i)].candidate.instrument, int(lots[int(i)])) for i in used
    ]
    hedged = (*problem.book, *hedge_positions)
    base_residual = _penalized_residual(problem.targets, value_book(hedged, snapshot).by_underlying)

    drift: dict[str, float] = {}
    worst_change = 0.0
    worst_regret = 0.0
    base = _contract_lots(usable, lots)
    # A floor, so that moving one or two lots of a very small trade is not reported as instability.
    denominator = max(float(sum(abs(q) for q in base.values())), _MIN_LOTS_FOR_RATIO)
    for perturbation in PERTURBATIONS:
        shocked = shock_snapshot(snapshot, perturbation)
        held = _penalized_residual(problem.targets, value_book(hedged, shocked).by_underlying)
        drift[perturbation.name] = held - base_residual
        try:
            moved_usable, _ = _filter_candidates(accepted, shocked, compare_quotes_to_model=False)
            book = value_book(problem.book, shocked)
            moved_scenario = book_scenario_pnl(problem.book, shocked, problem.scenarios)
            moved_units = (
                scenario_pnl_matrix(
                    [u.candidate.instrument for u in moved_usable], shocked, problem.scenarios
                )
                if moved_usable and problem.scenarios
                else np.zeros((len(problem.scenarios), len(moved_usable)), dtype=np.float64)
            )
            moved_program = _Program(
                problem, moved_usable, dict(book.by_underlying), moved_scenario, moved_units
            )
            moved = moved_program.solve()
        except HedgeInfeasibleError:
            notes.append(f"infeasible under {perturbation.name}: the bands are fragile")
            worst_change = max(worst_change, 1.0)
            worst_regret = max(worst_regret, float("inf"))
            continue
        except SolverRangeError as exc:
            notes.append(f"could not be re-solved under {perturbation.name}: {exc}")
            worst_change = max(worst_change, 1.0)
            continue
        except SearchLimitError as exc:
            raise SearchLimitError(
                f"the stability check could not finish under {perturbation.name}: {exc}"
            ) from exc
        hit_time_limit = hit_time_limit or moved.hit_time_limit
        other = _contract_lots(moved_usable, moved.lots)
        change = sum(abs(base.get(key, 0) - other.get(key, 0)) for key in {*base, *other})
        worst_change = max(worst_change, change / denominator)
        worst_regret = max(
            worst_regret,
            _regret(moved_program, moved_usable, moved.objective, usable, lots),
        )
    threshold = problem.config.instability_threshold
    regret_limit = problem.config.regret_threshold
    if not np.isfinite(worst_regret):
        # However little the trade moves, a held hedge outside a hard limit is not stable.
        notes.append(
            "under a small input change this hedge is no longer feasible: it breaks a hard limit, "
            "or its instruments are no longer usable"
        )
    elif worst_change > threshold and worst_regret > regret_limit:
        notes.append(
            f"a small input change moves the option trade by {worst_change:.0%} of its contracts, "
            f"and holding this hedge would give up {worst_regret:.0%} of its benefit against "
            f"the hedge that market calls for, over the {regret_limit:.0%} limit"
        )
    return StabilityReport(
        condition_number=condition,
        lot_change_ratio=worst_change,
        regret_ratio=worst_regret,
        drift=drift,
        is_stable=not notes,
        notes=tuple(notes),
        hit_time_limit=hit_time_limit,
    )


def _regret(
    program: _Program,
    moved_usable: Sequence[_Usable],
    best_objective: float,
    usable: Sequence[_Usable],
    lots: NDArray[np.int64],
) -> float:
    """How much worse today's hedge is than re-solving, if the market were really the moved one.

    Evaluated with the delta-one leg re-tuned, since a desk would do that anyway, and infinite when
    the held hedge is not feasible there. As a share of the benefit of hedging in that market: the
    objective of doing nothing less the best objective, floored at $1. Dividing by the objective
    alone would make any tight hedge look fragile, since a good hedge has a very small one.
    """
    held = {
        row.candidate.instrument.key: int(q) for row, q in zip(usable, lots, strict=True) if q != 0
    }
    index = {row.candidate.instrument.key: i for i, row in enumerate(moved_usable)}
    mapped = np.zeros(len(moved_usable), dtype=np.int64)
    for key, quantity in held.items():
        position = index.get(key)
        if position is None:
            return float("inf")
        mapped[position] = quantity
    tuned = program._delta_retune(mapped)
    value = min(program.evaluate(mapped), program.evaluate(tuned) if tuned is not None else np.inf)
    do_nothing = program.evaluate(np.zeros(len(moved_usable), dtype=np.int64))
    benefit = do_nothing - best_objective if np.isfinite(do_nothing) else abs(best_objective)
    return max(value - best_objective, 0.0) / max(benefit, 1.0)


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
    # Why this point has no proposal. `status` then says which kind: infeasible, search_limit, or
    # out_of_range.
    unsolved_reason: str | None = None
    status: str | None = None
    issues: tuple[str, ...] = ()


def _worst(values: dict[str, float] | None) -> float | None:
    return min(values.values()) if values else None


def hedge_frontier(
    problem: HedgeProblem,
    snapshot: MarketSnapshot,
    risk_aversions: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    *,
    gate_baseline: HedgeConfig | None = None,
) -> tuple[FrontierPoint, ...]:
    """Cost against residual risk, always including the baselines a user must be able to beat.

    Doing nothing and delta-only are always present, so a recommendation can be judged against the
    trivial alternatives. `gate_baseline` is what each point's gate settings are disclosed against.
    """
    book = value_book(problem.book, snapshot)
    before_scenarios = {}
    if problem.scenarios:
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
            issues=book.issues,
        )
    ]
    delta_only = _delta_only_problem(problem, snapshot)
    if delta_only is not None:
        points.append(
            _point(
                "delta only",
                None,
                delta_only,
                snapshot,
                problem.targets,
                gate_baseline,
                book.issues,
            )
        )
    for aversion in risk_aversions:
        adjusted = replace(problem, config=replace(problem.config, risk_aversion=aversion))
        points.append(
            _point(
                f"risk aversion {aversion:g}",
                aversion,
                adjusted,
                snapshot,
                problem.targets,
                gate_baseline,
                book.issues,
            )
        )
    return tuple(points)


def _delta_only_problem(problem: HedgeProblem, snapshot: MarketSnapshot) -> HedgeProblem | None:
    """Same book and scenarios, but only linear hedges, with a very strong penalty on delta.

    A candidate with no market data cannot trade, as in the hedge itself, so it gets no target.
    """
    linear = tuple(
        c
        for c in problem.candidates
        if isinstance(c.instrument, LinearInstrument)
        and c.instrument.underlying in snapshot.underlyings
    )
    if not linear:
        return None
    underlyings = sorted({c.instrument.underlying for c in linear})
    targets = tuple(GreekTarget(u, GreekName.DELTA, weight=1.0) for u in underlyings)
    config = replace(
        problem.config,
        risk_aversion=1.0,
        cvar_weight=0.0,
        max_legs=None,
        ticket_cost=0.0,
        premium_budget=None,
    )
    return HedgeProblem(problem.book, linear, targets, problem.scenarios, config)


def _point(
    label: str,
    aversion: float | None,
    problem: HedgeProblem,
    snapshot: MarketSnapshot,
    report_targets: Sequence[GreekTarget],
    gate_baseline: HedgeConfig | None,
    book_issues: tuple[str, ...],
) -> FrontierPoint:
    from greeks_neutralizer.hedging.optimizer import solve_hedge

    try:
        proposal = solve_hedge(problem, snapshot, gate_baseline=gate_baseline)
    except HedgeInfeasibleError as exc:
        reason = "; ".join(exc.binding) or str(exc)
        return _unsolved_point(label, aversion, "infeasible", reason, book_issues)
    except SearchLimitError as exc:
        return _unsolved_point(label, aversion, "search_limit", str(exc), book_issues)
    except SolverRangeError as exc:
        return _unsolved_point(label, aversion, "out_of_range", str(exc), book_issues)
    return _from_proposal(label, aversion, proposal, report_targets)


def _unsolved_point(
    label: str, aversion: float | None, status: str, reason: str, issues: tuple[str, ...]
) -> FrontierPoint:
    return FrontierPoint(
        label,
        aversion,
        0.0,
        0,
        0,
        float("nan"),
        None,
        None,
        unsolved_reason=reason,
        status=status,
        issues=issues,
    )


def _from_proposal(
    label: str,
    aversion: float | None,
    proposal: HedgeProposal,
    report_targets: Sequence[GreekTarget],
) -> FrontierPoint:
    return FrontierPoint(
        label=label,
        risk_aversion=aversion,
        transaction_cost=proposal.transaction_cost,
        gross_lots=proposal.gross_lots,
        n_legs=len(proposal.legs),
        penalized_residual=_penalized_residual(report_targets, dict(proposal.after)),
        worst_scenario_pnl=_worst(dict(proposal.scenario_pnl_after)),
        proposal=proposal,
        status=proposal.status.value,
        issues=tuple(proposal.issues),
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


@dataclass(frozen=True, slots=True)
class DecayForecast:
    steps: tuple[DecayStep, ...]
    issues: tuple[str, ...]


def hedge_decay_forecast(
    positions: Sequence[Position],
    snapshot: MarketSnapshot,
    underlying: str,
    days: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 21.0),
) -> DecayForecast:
    """How the book's exposures drift as time passes with spot and vol unchanged.

    An underlying the market does not have raises, since a misspelled one would read as no exposure.
    `issues` holds today's valuation issues, then those first appearing at a step, marked with it.
    """
    require_symbol("underlying", underlying)
    snapshot.market(underlying)
    today = value_book(positions, snapshot).issues
    issues = list(today)
    seen = set(today)
    steps: list[DecayStep] = []
    for day in days:
        rolled = shock_snapshot(snapshot, Scenario(f"+{day:g}d", days_forward=day))
        book = value_book(positions, rolled)
        steps.append(DecayStep(day, book.by_underlying.get(underlying, Exposure())))
        issues.extend(f"{issue} (at +{day:g}d)" for issue in book.issues if issue not in seen)
        seen.update(book.issues)
    return DecayForecast(tuple(steps), tuple(issues))
