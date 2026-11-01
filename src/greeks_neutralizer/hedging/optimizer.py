"""Hedge optimizer: a mixed-integer linear program over any number of instruments.

Decision variables for each usable candidate i:
    buy_touch_i, buy_deep_i, sell_touch_i, sell_deep_i >= 0   integer lots, by liquidity tier
    used_i in {0, 1}                                          only with max_legs or a ticket cost
For each target t: pos_t, neg_t >= 0, the absolute value of the residual exposure.
For CVaR (Rockafellar-Uryasev): VaR level v (free) and tail excess u_s >= 0 per scenario.

Solved by HiGHS through scipy.optimize.milp. Every proposal is then re-verified through a separate
path: the hedged book is repriced from scratch, and Greeks and bands are checked against what the
solver believed. A proposal that fails verification is an exception, not a warning.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import __version__ as scipy_version
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from greeks_neutralizer._version import __version__
from greeks_neutralizer.data.quality import QuoteFlag, assess_quote, is_usable
from greeks_neutralizer.domain import (
    ExerciseStyle,
    LinearInstrument,
    LinearKind,
    OptionContract,
    Position,
)
from greeks_neutralizer.errors import (
    GNError,
    HedgeInfeasibleError,
    MarketDataError,
    SearchLimitError,
    SolverRangeError,
    ValidationError,
)
from greeks_neutralizer.hedging.problem import (
    ExcludedCandidate,
    GreekName,
    GreekTarget,
    HedgeCandidate,
    HedgeConfig,
    HedgeLeg,
    HedgeProblem,
    HedgeProposal,
    ProposalStatus,
    ResidualLine,
    RunManifest,
    VerificationReport,
    gate_overrides,
)
from greeks_neutralizer.market import MarketSnapshot
from greeks_neutralizer.pricing.book import (
    Exposure,
    is_expired_instrument,
    linear_unit_exposure,
    unit_exposures,
    value_book,
    value_options,
)
from greeks_neutralizer.risk.scenarios import book_scenario_pnl, scenario_pnl_matrix

_DEFAULT_MAX_UNITS = 1_000_000
_MIN_WINDOW_LOTS = 25.0
_MAX_INTEGER_CANDIDATES = 40
_POLISH_PASSES = 30
_ROUNDING_REPAIR_RANGE = 5
_MIN_MODEL_QUOTE_GAP = 0.05
_EXCLUDED_ISSUE_SHARE = 0.25
_WINDOW_FRACTION = 0.5
_TRADING_DAYS = 252.0


class VerificationError(GNError):
    """The solver's solution did not survive independent repricing. Nothing may trade on it."""


@dataclass(frozen=True, slots=True)
class _Usable:
    candidate: HedgeCandidate
    exposure: Exposure
    multiplier: float
    mid_value: float
    buy_cost_touch: float
    sell_cost_touch: float
    buy_touch_cap: int
    sell_touch_cap: int
    buy_total_cap: int
    sell_total_cap: int
    basis_risk_per_unit_1d: float


def _filter_candidates(
    problem: HedgeProblem, snapshot: MarketSnapshot, *, compare_quotes_to_model: bool = True
) -> tuple[list[_Usable], list[ExcludedCandidate]]:
    """Apply quote gates and build per-unit economics. Exclusions carry their reason.

    `compare_quotes_to_model=False` is for stability re-solves, which shock the market and hold
    quotes fixed: there, model and quotes disagree on purpose, and it is not evidence of bad data.
    """
    config = problem.config
    policy = config.quality_policy()
    excluded: list[ExcludedCandidate] = []
    live: list[HedgeCandidate] = []
    for candidate in problem.candidates:
        name = candidate.instrument.describe()
        if candidate.instrument.underlying not in snapshot.underlyings:
            excluded.append(ExcludedCandidate(name, "no market data for its underlying"))
            continue
        instrument = candidate.instrument
        if is_expired_instrument(instrument, snapshot.as_of):
            excluded.append(ExcludedCandidate(name, "expired"))
            continue
        flags = assess_quote(candidate.quote, snapshot.as_of, policy=policy)
        if not is_usable(flags):
            excluded.append(
                ExcludedCandidate(
                    name, "unusable quote: " + ", ".join(sorted(f.value for f in flags))
                )
            )
            continue
        live.append(candidate)

    if compare_quotes_to_model:
        live = _linear_quote_gate(live, excluded, snapshot, config)
    live = _model_gates(live, excluded, snapshot, config, compare_quotes_to_model)

    exposures = unit_exposures([c.instrument for c in live], snapshot) if live else ()
    usable = [
        _economics(candidate, exposure, config)
        for candidate, exposure in zip(live, exposures, strict=True)
    ]
    return usable, excluded


def _linear_quote_gate(
    live: list[HedgeCandidate],
    excluded: list[ExcludedCandidate],
    snapshot: MarketSnapshot,
    config: HedgeConfig,
) -> list[HedgeCandidate]:
    """Compare each stock or future quote with what the market says it is worth.

    Spot times the delta ratio, or the forward for a future. Options have the model to catch a wrong
    spot, but a delta-only hedge has no other evidence, and its limit price reaches the ticket. A
    proxy's price has no such reference, as a leveraged fund's exposure is not its price, so it
    passes here unchecked and a proposal using one says so.
    """
    kept: list[HedgeCandidate] = []
    for candidate in live:
        instrument = candidate.instrument
        if isinstance(instrument, LinearInstrument) and instrument.kind is not LinearKind.PROXY:
            market = snapshot.market(instrument.underlying)
            fair, _ = linear_unit_exposure(instrument, market, snapshot.as_of)
            gap = abs(candidate.quote.mid / fair - 1.0)
            if gap > config.max_linear_quote_gap:
                excluded.append(
                    ExcludedCandidate(
                        instrument.describe(),
                        f"unusable quote: mid {candidate.quote.mid:g} is {gap:.0%} from the "
                        f"{fair:g} the market implies",
                    )
                )
                continue
        kept.append(candidate)
    return kept


def _model_gates(
    live: list[HedgeCandidate],
    excluded: list[ExcludedCandidate],
    snapshot: MarketSnapshot,
    config: HedgeConfig,
    compare_quotes_to_model: bool = True,
) -> list[HedgeCandidate]:
    """Compare each option's quote with the model, which catches errors repricing cannot.

    Verification reprices using the same market, so a vol given as 25.0 rather than 0.25 agrees with
    itself. The live quotes are the only independent evidence. One far-off quote is excluded as an
    outlier. When they broadly disagree, the market data is suspect, and this fails closed.
    """
    options = [c for c in live if isinstance(c.instrument, OptionContract)]
    contracts = [c.instrument for c in options if isinstance(c.instrument, OptionContract)]
    if not contracts:
        return live
    valuation = value_options(contracts, snapshot, tree_steps=25)
    rejected: set[int] = set()
    outliers = 0
    for row, (candidate, contract) in enumerate(zip(options, contracts, strict=True)):
        market = snapshot.market(contract.underlying)
        quote = candidate.quote
        name = contract.describe()
        # European options can trade below spot intrinsic, but not below discounted intrinsic.
        if contract.style is ExerciseStyle.AMERICAN:
            floor = max(contract.right.omega * (market.spot - contract.strike), 0.0)
        else:
            discount = float(market.curve.discount(float(valuation.T[row])))
            forward = float(valuation.forward[row])
            floor = discount * max(contract.right.omega * (forward - contract.strike), 0.0)
        if valuation.extrapolated[row] and not config.allow_extrapolated_vol:
            excluded.append(ExcludedCandidate(name, "strike beyond the surface's quotes"))
            rejected.add(id(candidate))
            continue
        if not compare_quotes_to_model:
            continue
        policy = config.quality_policy()
        flags = assess_quote(quote, snapshot.as_of, intrinsic=floor, policy=policy)
        if QuoteFlag.BELOW_INTRINSIC in flags:
            excluded.append(ExcludedCandidate(name, "unusable quote: below_intrinsic"))
            rejected.add(id(candidate))
            continue
        gap = abs(float(valuation.greeks.price[row]) - quote.mid)
        vega = max(float(valuation.greeks.vega[row]), 1e-9)
        tradable_gap = 3.0 * quote.half_spread + _MIN_MODEL_QUOTE_GAP
        if gap > tradable_gap and gap / vega > config.max_model_quote_vol_gap:
            excluded.append(
                ExcludedCandidate(
                    name,
                    f"unusable quote: {QuoteFlag.IV_OUTLIER.value}, about "
                    f"{gap / vega * 100.0:.1f} vol points from the model",
                )
            )
            rejected.add(id(candidate))
            outliers += 1
    if outliers >= max(2, (len(options) + 1) // 2):
        raise MarketDataError(
            f"model and quotes disagree on {outliers} of {len(options)} candidates: "
            "check vol units, rates, and spot"
        )
    return [c for c in live if id(c) not in rejected]


def _economics(candidate: HedgeCandidate, exposure: Exposure, config: HedgeConfig) -> _Usable:
    instrument = candidate.instrument
    multiplier = instrument.multiplier
    quote = candidate.quote
    half = quote.half_spread * multiplier
    lot = candidate.lot_size
    # A zero bid means nobody will buy it from you at any price.
    can_sell = quote.bid > 0.0

    def in_lots(units: int) -> int:
        return units // lot

    # Nothing displayed on a side means no market on that side, and by default a hedge is limited
    # to the display. The exception is a source declared not to populate sizes, where the caps are
    # the only limit there can be.
    sizes_absent = not config.require_quote_size and quote.bid_size == 0 and quote.ask_size == 0
    default_buy = _DEFAULT_MAX_UNITS if sizes_absent else quote.ask_size
    default_sell = _DEFAULT_MAX_UNITS if sizes_absent else quote.bid_size
    buy_max = default_buy if candidate.max_buy is None else candidate.max_buy
    sell_max = default_sell if candidate.max_sell is None else candidate.max_sell
    if not sizes_absent:
        if quote.ask_size == 0:
            buy_max = 0
        if quote.bid_size == 0:
            sell_max = 0
    # A hard stop on every contract-sized leg, whatever the objective says, so that no modeling
    # error can size a leg absurdly. Fine-grained linear hedges are measured in shares instead.
    fine = isinstance(instrument, LinearInstrument) and instrument.multiplier == 1.0
    if not fine:
        buy_max = min(buy_max, config.max_lots_per_leg)
        sell_max = min(sell_max, config.max_lots_per_leg)
    if not can_sell:
        sell_max = 0
    basis = 0.0
    if isinstance(instrument, LinearInstrument) and instrument.kind is LinearKind.PROXY:
        unit_notional = abs(exposure.delta_dollars)
        basis = unit_notional * instrument.basis_vol / np.sqrt(_TRADING_DAYS)
    return _Usable(
        candidate=candidate,
        exposure=exposure,
        multiplier=multiplier,
        # A future costs no premium: its notional is exposure, not cash paid.
        mid_value=0.0
        if isinstance(instrument, LinearInstrument) and instrument.kind is LinearKind.FUTURE
        else quote.mid * multiplier,
        buy_cost_touch=half + candidate.fee_per_unit,
        sell_cost_touch=half + candidate.fee_per_unit,
        buy_touch_cap=in_lots(buy_max if sizes_absent else min(quote.ask_size, buy_max)),
        sell_touch_cap=in_lots(sell_max if sizes_absent else min(quote.bid_size, sell_max)),
        buy_total_cap=in_lots(buy_max),
        sell_total_cap=in_lots(sell_max),
        basis_risk_per_unit_1d=float(basis),
    )


_MAX_SOLVER_COST = 1e15
# A row is scaled by the book, up to this many times its largest hedge. HiGHS drops a coefficient
# under its small-matrix value, so an uncapped large book made every hedge on the row invisible.
_MAX_BOOK_TO_HEDGE_SCALE = 1e4
_HIGHS_SMALL_MATRIX_VALUE = 1e-9
_HIGHS_INFINITY = 1e20
# A dropped coefficient matters if the instrument at capacity could do this share of what the row's
# largest hedge can at capacity. Below it, Black gamma and vega deep in a strike grid are dropped
# harmlessly, and a real chain nearly always has such options.
_MATERIAL_SHARE_OF_REACH = 1e-3


def _reached_clock(result: Any) -> bool:
    return bool(result.status == 1 and "time limit" in str(result.message).lower())


@dataclass(frozen=True, slots=True)
class _Solution:
    lots: NDArray[np.int64]
    objective: float
    gap: float
    lower_bound: float
    hit_time_limit: bool = False


class _Program:
    """Builds and solves the MILP. Column layout is stored so results can be read back by name."""

    def __init__(
        self,
        problem: HedgeProblem,
        usable: list[_Usable],
        book_by_underlying: dict[str, Exposure],
        book_scenario: NDArray[np.float64],
        unit_scenario: NDArray[np.float64],
    ) -> None:
        self.problem = problem
        self.usable = usable
        self.book_by_underlying = book_by_underlying
        self.book_scenario = book_scenario
        self.unit_scenario = unit_scenario
        self.targets = problem.targets
        self.n = len(usable)
        self.nt = len(self.targets)
        self.ns = len(problem.scenarios) if problem.config.cvar_weight > 0.0 else 0
        config = problem.config
        self.use_binary = config.max_legs is not None or config.ticket_cost > 0.0

        # Exposures by target and candidate, so that evaluating a trial touches only held legs.
        self._book_row = np.array([self.book_value(t) for t in self.targets], dtype=np.float64)
        self._exposures = np.array(
            [[self._exposure_column(t, row) for row in usable] for t in self.targets],
            dtype=np.float64,
        ).reshape(self.nt, self.n)

        # Columns: 4n trade tiers, optional n binaries, 2nt abs parts, then CVaR v and u_s.
        self.i_trade = 0
        self.i_used = 4 * self.n
        self.i_abs = self.i_used + (self.n if self.use_binary else 0)
        self.i_v = self.i_abs + 2 * self.nt
        self.i_u = self.i_v + (1 if self.ns else 0)
        self.size = self.i_u + self.ns

    def _exposure_column(self, target: GreekTarget, row: _Usable) -> float:
        if row.candidate.instrument.underlying != target.underlying:
            return 0.0
        return float(getattr(row.exposure, target.greek.value))

    def _row_scale(self, target: GreekTarget) -> float:
        """Largest exposure on this target's row, so scaled coefficients are at most about 1.

        Delta dollars per contract are around 1e6 next to unit coefficients on the absolute-value
        columns, which leaves the matrix badly conditioned and made HiGHS report numerical trouble.
        The book counts only up to a limit, so that it cannot scale the hedges out of the solver.
        """
        hedge = max((abs(self._exposure_column(target, row)) for row in self.usable), default=0.0)
        book = abs(self.book_value(target))
        return max(max(hedge, min(book, hedge * _MAX_BOOK_TO_HEDGE_SCALE)), 1.0)

    def book_value(self, target: GreekTarget) -> float:
        exposure = self.book_by_underlying.get(target.underlying, Exposure())
        return float(getattr(exposure, target.greek.value))

    def _costs(self) -> NDArray[np.float64]:
        config = self.problem.config
        c = np.zeros(self.size, dtype=np.float64)
        for i, row in enumerate(self.usable):
            lot = row.candidate.lot_size
            impact = row.candidate.impact_per_unit
            basis = config.risk_aversion * row.basis_risk_per_unit_1d
            c[4 * i + 0] = lot * (row.buy_cost_touch + basis)
            c[4 * i + 1] = lot * (row.buy_cost_touch + impact + basis)
            c[4 * i + 2] = lot * (row.sell_cost_touch + basis)
            c[4 * i + 3] = lot * (row.sell_cost_touch + impact + basis)
            if self.use_binary:
                c[self.i_used + i] = config.ticket_cost
        for t, target in enumerate(self.targets):
            # The absolute-value columns are in units of the scaled row.
            penalty = config.risk_aversion * target.effective_weight * self._row_scale(target)
            c[self.i_abs + 2 * t] = penalty
            c[self.i_abs + 2 * t + 1] = penalty
        if self.ns:
            c[self.i_v] = config.cvar_weight
            tail = config.cvar_weight / ((1.0 - config.cvar_alpha) * self.ns)
            c[self.i_u : self.i_u + self.ns] = tail
        if not np.all(np.abs(c) <= _MAX_SOLVER_COST):
            raise SolverRangeError(
                f"an objective cost is past {_MAX_SOLVER_COST:g}, where the solver cannot be "
                "relied on: check risk_aversion, cvar_weight, ticket_cost, target weights, and "
                "candidate fees, impact, and lot sizes"
            )
        return c

    def _variable_bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lb = np.zeros(self.size, dtype=np.float64)
        ub = np.full(self.size, np.inf, dtype=np.float64)
        for i, row in enumerate(self.usable):
            ub[4 * i + 0] = row.buy_touch_cap
            ub[4 * i + 1] = max(row.buy_total_cap - row.buy_touch_cap, 0)
            ub[4 * i + 2] = row.sell_touch_cap
            ub[4 * i + 3] = max(row.sell_total_cap - row.sell_touch_cap, 0)
            if self.use_binary:
                ub[self.i_used + i] = 1.0
        # The VaR level stays at or above zero, which floors CVaR at zero. Otherwise a scenario set
        # where a hedge always profits makes CVaR negative without limit, and pays for more size.
        return lb, ub

    def _net_units(self, i: int) -> dict[int, float]:
        lot = float(self.usable[i].candidate.lot_size)
        return {4 * i + 0: lot, 4 * i + 1: lot, 4 * i + 2: -lot, 4 * i + 3: -lot}

    def _constraints(self, elastic: bool = False) -> tuple[list[LinearConstraint], list[str]]:
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        lo: list[float] = []
        hi: list[float] = []
        names: list[str] = []

        def add(entries: dict[int, float], low: float, high: float, name: str) -> None:
            r = len(lo)
            for col, value in entries.items():
                rows.append(r)
                cols.append(col)
                vals.append(value)
            lo.append(low)
            hi.append(high)
            names.append(name)

        for t, target in enumerate(self.targets):
            scale = self._row_scale(target)
            base = self.book_value(target) / scale
            terms: dict[int, float] = {}
            for i, row in enumerate(self.usable):
                exposure = self._exposure_column(target, row) / scale
                if exposure != 0.0:
                    for col, sign in self._net_units(i).items():
                        terms[col] = sign * exposure
            # residual = base + sum(terms) = pos - neg
            definition = dict(terms)
            definition[self.i_abs + 2 * t] = -1.0
            definition[self.i_abs + 2 * t + 1] = 1.0
            add(definition, -base, -base, f"residual:{target.underlying}:{target.greek.value}")
            if not elastic and (target.lower is not None or target.upper is not None):
                low = -np.inf if target.lower is None else target.lower / scale - base
                high = np.inf if target.upper is None else target.upper / scale - base
                add(terms, low, high, f"band:{target.underlying}:{target.greek.value}")

        config = self.problem.config
        if config.premium_budget is not None and not elastic:
            terms = {}
            for i, row in enumerate(self.usable):
                for col, sign in self._net_units(i).items():
                    terms[col] = sign * row.mid_value
            add(terms, -np.inf, config.premium_budget, "premium_budget")

        if self.use_binary:
            for i, row in enumerate(self.usable):
                cap = float(max(row.buy_total_cap, row.sell_total_cap, 1))
                terms = {4 * i + k: 1.0 for k in range(4)}
                terms[self.i_used + i] = -cap
                add(terms, -np.inf, 0.0, f"used:{i}")
            if config.max_legs is not None and not elastic:
                terms = {self.i_used + i: 1.0 for i in range(self.n)}
                add(terms, 0.0, float(config.max_legs), "max_legs")

        for s in range(self.ns):
            # u_s >= loss_s - v, where loss_s = -(book_pnl_s + sum x_i pnl_is)
            terms = {self.i_u + s: 1.0, self.i_v: 1.0}
            for i in range(self.n):
                pnl = float(self.unit_scenario[s, i])
                if pnl != 0.0:
                    for col, sign in self._net_units(i).items():
                        terms[col] = sign * pnl
            add(terms, -float(self.book_scenario[s]), np.inf, f"cvar:{s}")

        if not lo:
            return [], []
        matrix = sparse.csr_array((vals, (rows, cols)), shape=(len(lo), self.size))
        return [LinearConstraint(matrix, np.array(lo), np.array(hi))], names

    def _book_description(self) -> str:
        book = self.problem.book
        size = max((abs(p.quantity) for p in book), default=0)
        multiplier = max((p.instrument.multiplier for p in book), default=0.0)
        return (
            f"a book of up to {size:g} units with a multiplier up to {multiplier:g}: check the "
            "book size, spot, and multiplier, and any band or budget"
        )

    def _check_solver_range(self, constraints: list[LinearConstraint]) -> None:
        """Refuse what HiGHS would silently change: a coefficient it drops as zero where that
        matters, or a right-hand side it reads as infinite, which can even give a false proof of
        infeasibility.

        What matters is in the row's own dollars, |exposure| * capacity * risk_aversion * weight,
        against the same for its largest hedge, so the shared price of the row cancels. An option
        that could move the objective by cents does not refuse the request: the solver leaves it at
        zero, and verification reprices the result as it is.
        """
        config = self.problem.config
        for target in self.targets:
            unpriced = config.risk_aversion * target.effective_weight == 0.0
            if unpriced and target.lower is None and target.upper is None:
                continue
            scale = self._row_scale(target)
            reach = [
                abs(self._exposure_column(target, row))
                * max(row.buy_total_cap, row.sell_total_cap)
                * row.candidate.lot_size
                for row in self.usable
            ]
            largest = max(reach, default=0.0)
            for row, row_reach in zip(self.usable, reach, strict=True):
                coefficient = abs(self._exposure_column(target, row)) / scale
                if (
                    0.0 < coefficient < _HIGHS_SMALL_MATRIX_VALUE
                    and row_reach >= _MATERIAL_SHARE_OF_REACH * largest
                ):
                    raise SolverRangeError(
                        f"{row.candidate.instrument.describe()} is too small against the rest of "
                        f"the {target.greek.value} row for the solver to see it, though it could "
                        f"matter at capacity, with {self._book_description()}"
                    )
        for limit in constraints:
            for side in (np.asarray(limit.lb), np.asarray(limit.ub)):
                finite = np.abs(side[np.isfinite(side)])
                if finite.size and float(finite.max()) >= _HIGHS_INFINITY:
                    raise SolverRangeError(
                        f"a limit or scenario loss of {float(finite.max()):g} is past what the "
                        f"solver treats as finite, with {self._book_description()}"
                    )

    def _beats_the_lower_bound(self, solution: _Solution) -> str | None:
        """A candidate at capacity that scores below the LP bound, which no feasible hedge can do.

        Nothing else notices a solver that cannot see an instrument, since verification reprices
        the lots it chose and the LP bound shares the same blind spot.
        """
        slack = 1e-6 * max(abs(solution.lower_bound), 1.0)
        for i, row in enumerate(self.usable):
            lot = row.candidate.lot_size
            for quantity in (row.buy_total_cap * lot, -row.sell_total_cap * lot):
                trial = solution.lots.copy()
                trial[i] = quantity
                if self.evaluate(trial) < solution.lower_bound - slack:
                    name = row.candidate.instrument.describe()
                    return f"{quantity} of {name} beats the lower bound the solver reported"
        return None

    def _integrality(self, *, all_whole: bool = False) -> NDArray[np.int64]:
        """Integer lots, except fine-grained linear hedges, which are solved then rounded.

        Branching on share counts up to a million costs real solve time and usually buys nothing:
        the rounded result is what gets verified and reported. `all_whole` is for when it does.
        """
        kinds = np.zeros(self.size, dtype=np.int64)
        for i, row in enumerate(self.usable):
            instrument = row.candidate.instrument
            fine = isinstance(instrument, LinearInstrument) and instrument.multiplier == 1.0
            if all_whole or not fine:
                kinds[4 * i : 4 * i + 4] = 1
        if self.use_binary:
            kinds[self.i_used : self.i_used + self.n] = 1
        return kinds

    def _net_lots(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array(
            [x[4 * i] + x[4 * i + 1] - x[4 * i + 2] - x[4 * i + 3] for i in range(self.n)],
            dtype=np.float64,
        )

    def _windowed_bounds(
        self,
        lp_lots: NDArray[np.float64],
        widen: float,
        screened: NDArray[np.bool_] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Bounds for the integer search: near the LP solution, and always able to drop a leg.

        An integer hedge worth having sits close to the continuous optimum, since moving far from
        it costs more than integrality can repay. Without this, HiGHS gets lot bounds of a million
        and a weak big-M, and at its time limit it returned an incumbent 80% worse than a tighter
        problem's: a constrained solve scored better than the unconstrained one.
        """
        lb, ub = self._variable_bounds()
        for i, lots in enumerate(lp_lots):
            window = widen * max(_MIN_WINDOW_LOTS, _WINDOW_FRACTION * abs(float(lots)))
            high = max(float(lots) + window, 0.0)
            low = min(float(lots) - window, 0.0)
            ub[4 * i + 0] = min(ub[4 * i + 0], np.ceil(high))
            ub[4 * i + 1] = min(ub[4 * i + 1], np.ceil(high))
            ub[4 * i + 2] = min(ub[4 * i + 2], np.ceil(-low))
            ub[4 * i + 3] = min(ub[4 * i + 3], np.ceil(-low))
            if screened is not None and not screened[i]:
                ub[4 * i : 4 * i + 4] = 0.0
        return lb, ub

    def evaluate(self, lots: NDArray[np.int64]) -> float:
        """The exact objective of net `lots` (in units), or infinity if any constraint fails.

        An independent statement of the MILP's objective and constraints in terms of net positions.
        A test requires it to reproduce the solver's objective, which cross-checks the formulation.
        """
        config = self.problem.config
        total = 0.0
        premium = 0.0
        legs = 0
        held = np.flatnonzero(lots)
        for index in held:
            row = self.usable[int(index)]
            q = int(lots[int(index)])
            lot = row.candidate.lot_size
            if q % lot != 0:
                return float("inf")
            size = abs(q)
            buying = q > 0
            cap_lots = row.buy_total_cap if buying else row.sell_total_cap
            touch_lots = row.buy_touch_cap if buying else row.sell_touch_cap
            if size > cap_lots * lot:
                return float("inf")
            touch_cost = row.buy_cost_touch if buying else row.sell_cost_touch
            deep_units = max(size - touch_lots * lot, 0)
            total += size * (touch_cost + config.risk_aversion * row.basis_risk_per_unit_1d)
            total += deep_units * row.candidate.impact_per_unit
            premium += q * row.mid_value
            legs += 1
        if config.premium_budget is not None and premium > config.premium_budget + 1e-9:
            return float("inf")
        if config.max_legs is not None and legs > config.max_legs:
            return float("inf")
        if self.use_binary:
            total += legs * config.ticket_cost
        for t, target in enumerate(self.targets):
            value = self._book_row[t] + float(self._exposures[t, held] @ lots[held])
            if target.lower is not None and value < target.lower - 1e-9:
                return float("inf")
            if target.upper is not None and value > target.upper + 1e-9:
                return float("inf")
            total += config.risk_aversion * target.effective_weight * abs(value)
        if self.ns:
            total += config.cvar_weight * self._cvar(lots)
        return total

    def _cvar(self, lots: NDArray[np.int64]) -> float:
        """Rockafellar-Uryasev with v >= 0: the minimum is at zero or at one of the positive losses.

        Flooring at zero means protection earns nothing once the tail loss is gone. Without it, the
        proxy backtest bought 11.6 billion lots against a crash-heavy scenario list.
        """
        alpha = self.problem.config.cvar_alpha
        losses = -(self.book_scenario + self.unit_scenario @ lots.astype(np.float64))
        best = float("inf")
        for v in (0.0, *(float(x) for x in losses if x > 0.0)):
            excess = np.maximum(losses - v, 0.0)
            best = min(best, float(v + excess.sum() / ((1.0 - alpha) * losses.size)))
        return best

    def _delta_retune(self, lots: NDArray[np.int64]) -> NDArray[np.int64] | None:
        """These lots with the fine-grained linear hedge moved to flatten delta, if one exists."""
        for target in self.targets:
            if target.greek is not GreekName.DELTA:
                continue
            for i, row in enumerate(self.usable):
                instrument = row.candidate.instrument
                if (
                    isinstance(instrument, LinearInstrument)
                    and instrument.multiplier == 1.0
                    and instrument.underlying == target.underlying
                ):
                    unit = self._exposure_column(target, row)
                    if unit == 0.0:
                        continue
                    t = self.targets.index(target)
                    held = np.flatnonzero(lots)
                    held = held[held != i]
                    others = self._book_row[t] + float(self._exposures[t, held] @ lots[held])
                    tuned = lots.copy()
                    tuned[i] = round(-others / unit)
                    return tuned
        return None

    def _repair_rounding(self, lots: NDArray[np.int64]) -> NDArray[np.int64]:
        """Rounding a fine-grained hedge can leave a band or budget, so try its neighbours."""
        if np.isfinite(self.evaluate(lots)):
            return lots
        fine = [
            i
            for i, row in enumerate(self.usable)
            if isinstance(row.candidate.instrument, LinearInstrument)
            and row.candidate.instrument.multiplier == 1.0
        ]
        best, best_value = lots, float("inf")
        for i in fine:
            for step in range(-_ROUNDING_REPAIR_RANGE, _ROUNDING_REPAIR_RANGE + 1):
                trial = lots.copy()
                trial[i] += step
                value = self.evaluate(trial)
                if value < best_value:
                    best, best_value = trial, value
        return best

    def polish(
        self, lots: NDArray[np.int64], allowed: NDArray[np.bool_]
    ) -> tuple[NDArray[np.int64], float]:
        """Deterministic local search over single and paired lot moves from a feasible start.

        Lot rounding makes this a lattice problem: the best integer point near the LP solution is
        generally not what branch and bound reaches inside its time limit. On a 5-instrument option
        hedge, HiGHS's incumbent at the limit scored 358 against 252 for a more constrained problem,
        which is impossible at an optimum. Only strict improvements are taken, so this terminates
        and never leaves a worse answer than the solver's.
        """
        current = self._repair_rounding(lots)
        best = self.evaluate(current)
        steps = [i for i in range(self.n) if allowed[i]]
        if not steps or not np.isfinite(best):
            return current, best
        for _ in range(_POLISH_PASSES):
            improved = False
            held = [i for i in steps if current[i] != 0]
            for i in steps:
                unit = self.usable[i].candidate.lot_size
                for direction in (unit, -unit):
                    trial = current.copy()
                    trial[i] += direction
                    pool = [trial]
                    for j in held:
                        if j == i:
                            continue
                        other = self.usable[j].candidate.lot_size
                        for second in (other, -other):
                            paired = trial.copy()
                            paired[j] += second
                            pool.append(paired)
                    for candidate in pool:
                        tuned = self._delta_retune(candidate)
                        for option in (candidate, tuned):
                            if option is None:
                                continue
                            value = self.evaluate(option)
                            if value < best - 1e-9:
                                current, best = option, value
                                improved = True
            if not improved:
                break
        return current, best

    def _relax(self, costs: NDArray[np.float64], constraints: list[LinearConstraint]) -> Any:
        """Solve the LP relaxation with linprog, which returns the reduced costs used to screen."""
        lb, ub = self._variable_bounds()
        bounds = list(zip(lb, ub, strict=True))
        kwargs: dict[str, Any] = {}
        if constraints:
            matrix = sparse.csr_array(constraints[0].A)
            low = np.asarray(constraints[0].lb, dtype=np.float64)
            high = np.asarray(constraints[0].ub, dtype=np.float64)
            equal = low == high
            upper = ~equal & np.isfinite(high)
            lower = ~equal & np.isfinite(low)
            if np.any(equal):
                kwargs["A_eq"] = matrix[np.flatnonzero(equal)]
                kwargs["b_eq"] = high[equal]
            if np.any(upper) or np.any(lower):
                parts = []
                rhs = []
                if np.any(upper):
                    parts.append(matrix[np.flatnonzero(upper)])
                    rhs.append(high[upper])
                if np.any(lower):
                    parts.append(-matrix[np.flatnonzero(lower)])
                    rhs.append(-low[lower])
                kwargs["A_ub"] = sparse.vstack(parts, format="csr")
                kwargs["b_ub"] = np.concatenate(rhs)
        return linprog(
            costs,
            bounds=bounds,
            method="highs",
            options={"time_limit": self.problem.config.time_limit_seconds, "disp": False},
            **kwargs,
        )

    def _screen(self, relaxed: Any, lp_lots: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Candidates the integer search may use: the LP's choices, plus the closest alternatives.

        The reduced cost of a column is how far its cost is from being worth using. Small values
        mark instruments that are nearly as good as those chosen, which is what integer rounding
        needs. Thousands of candidates with large reduced costs only slow the search: with 2,144 on
        a real SPX chain, the time limit left a frontier that was not monotone in risk aversion.
        """
        keep = np.abs(lp_lots) > 1e-9
        if self.n <= _MAX_INTEGER_CANDIDATES:
            return np.ones(self.n, dtype=bool)
        marginals = getattr(getattr(relaxed, "lower", None), "marginals", None)
        if marginals is None:
            return np.ones(self.n, dtype=bool)
        reduced = np.asarray(marginals, dtype=np.float64)[: 4 * self.n].reshape(self.n, 4)
        tradable = self._variable_bounds()[1][: 4 * self.n].reshape(self.n, 4) > 0.0
        score = np.where(tradable, reduced, np.inf).min(axis=1)
        order = np.argsort(score, kind="stable")
        keep[order[:_MAX_INTEGER_CANDIDATES]] = True
        return keep

    def solve(self) -> _Solution:
        """LP relaxation, then an integer search over screened candidates near its solution.

        The relaxation is fast, gives the true lower bound on the objective, and decides
        feasibility. If the window excludes every integer solution, the search widens once and then
        drops the screen. If rounding a share leg still cannot meet a band, shares are branched on
        too, since a share can be worth more than the band is wide. Only the last solve, all whole
        numbers at full capacity, can prove infeasibility: every earlier one is a restriction, so
        its failure proves nothing. `gap` is measured against the LP bound, so it reports what
        integrality cost in total and cannot be fooled by an incumbent returned at the time limit.
        """
        config = self.problem.config
        costs = self._costs()
        constraints, _ = self._constraints()
        self._check_solver_range(constraints)
        relaxed = self._relax(costs, constraints)
        if relaxed.status == 2:
            raise HedgeInfeasibleError(
                "no hedge satisfies the hard constraints", self.diagnose_infeasibility()
            )
        if relaxed.status == 1:
            raise SearchLimitError(
                f"the continuous relaxation did not finish inside {config.time_limit_seconds:g}s, "
                "which says nothing about whether a hedge exists: raise time_limit_seconds, or "
                "reduce the universe"
            )
        if relaxed.status != 0 or relaxed.x is None:
            raise GNError(f"hedge solver failed on the relaxation: {relaxed.message}")
        lower_bound = float(relaxed.fun)
        lp_lots = self._net_lots(np.asarray(relaxed.x, dtype=np.float64))
        screened = self._screen(relaxed, lp_lots)

        fine = self._integrality()
        whole = self._integrality(all_whole=True)
        plan = (
            (self._windowed_bounds(lp_lots, 1.0, screened), fine),
            (self._windowed_bounds(lp_lots, 4.0, screened), fine),
            (self._windowed_bounds(lp_lots, 4.0), fine),
            (self._windowed_bounds(lp_lots, 4.0), whole),
            (self._variable_bounds(), whole),
        )
        result: Any = None
        reached_clock = False
        for (lb, ub), integrality in plan:
            result = milp(
                c=costs,
                constraints=constraints,
                bounds=Bounds(lb, ub),
                integrality=integrality,
                options={
                    "mip_rel_gap": config.mip_gap,
                    "node_limit": config.mip_node_limit,
                    "time_limit": config.time_limit_seconds,
                    "disp": False,
                },
            )
            reached_clock = reached_clock or _reached_clock(result)
            if result.x is None:
                continue
            lots, objective = self.polish(self._read_lots(result.x), screened)
            if np.isfinite(objective):
                gap = (objective - lower_bound) / max(abs(objective), 1.0)
                return _Solution(lots, objective, max(gap, 0.0), lower_bound, reached_clock)
        if result.x is None and result.status == 2:
            raise HedgeInfeasibleError(
                "the continuous hedge is feasible but no whole number of lots and shares is, at "
                "full capacity: widen a band by about one unit's exposure",
                self.diagnose_infeasibility(),
            )
        raise SearchLimitError(
            f"the integer search stopped with no hedge, inside mip_node_limit="
            f"{config.mip_node_limit} and time_limit_seconds={config.time_limit_seconds:g}. "
            "That does not show that no hedge exists: raise mip_node_limit or "
            f"time_limit_seconds, or reduce the universe. Solver: {result.message}"
        )

    def _read_lots(self, x: NDArray[np.float64]) -> NDArray[np.int64]:
        lots = np.zeros(self.n, dtype=np.int64)
        for i, row in enumerate(self.usable):
            buys = round(x[4 * i]) + round(x[4 * i + 1])
            sells = round(x[4 * i + 2]) + round(x[4 * i + 3])
            lots[i] = (buys - sells) * row.candidate.lot_size
        return lots

    def diagnose_infeasibility(self) -> tuple[str, ...]:
        """Which hard constraints cannot be met, found by solving with explicit slack on each.

        The relaxation minimizes total slack, so non-zero slack names the binding constraints
        and how far each one is from achievable, in its own units.
        """
        config = self.problem.config
        banded = [t for t in self.targets if t.lower is not None or t.upper is not None]
        budget = config.premium_budget
        extra = 2 * len(banded) + (1 if budget is not None else 0)
        base_constraints, _ = self._constraints(elastic=True)
        size = self.size + extra
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        lo: list[float] = []
        hi: list[float] = []
        labels: list[str] = []
        next_slack = self.size

        def net_terms(column_of: Any) -> dict[int, float]:
            terms: dict[int, float] = {}
            for i, row in enumerate(self.usable):
                value = float(column_of(row))
                if value != 0.0:
                    for col, sign in self._net_units(i).items():
                        terms[col] = sign * value
            return terms

        for target in banded:
            base = self.book_value(target)
            terms = net_terms(lambda row, t=target: self._exposure_column(t, row))
            low = -np.inf if target.lower is None else target.lower - base
            high = np.inf if target.upper is None else target.upper - base
            r = len(lo)
            for col, value in terms.items():
                rows.append(r)
                cols.append(col)
                vals.append(value)
            rows += [r, r]
            cols += [next_slack, next_slack + 1]
            vals += [1.0, -1.0]
            lo.append(low)
            hi.append(high)
            label = f"{target.greek.value} on {target.underlying}"
            labels += [
                f"{label}: lower bound short by {{:.2f}}",
                f"{label}: upper bound over by {{:.2f}}",
            ]
            next_slack += 2
        if budget is not None:
            terms = net_terms(lambda row: row.mid_value)
            r = len(lo)
            for col, value in terms.items():
                rows.append(r)
                cols.append(col)
                vals.append(value)
            rows.append(r)
            cols.append(next_slack)
            vals.append(-1.0)
            lo.append(-np.inf)
            hi.append(budget)
            labels.append("premium budget over by {:.2f}")
            next_slack += 1

        if not lo:
            return ("infeasible with no bands or budget: check candidate capacity",)

        lb, ub = self._variable_bounds()
        lb = np.concatenate([lb, np.zeros(extra)])
        ub = np.concatenate([ub, np.full(extra, np.inf)])
        # Scale each slack to its own constraint so a large-unit bound cannot dominate the choice.
        cost = np.zeros(size, dtype=np.float64)
        cost[self.size :] = 1.0
        constraints = [
            LinearConstraint(
                sparse.csr_array((vals, (rows, cols)), shape=(len(lo), size)),
                np.array(lo),
                np.array(hi),
            )
        ]
        for original in base_constraints:
            matrix = sparse.hstack(
                [original.A, sparse.csr_array((original.A.shape[0], extra))], format="csr"
            )
            constraints.append(LinearConstraint(matrix, original.lb, original.ub))
        result = milp(
            c=cost,
            constraints=constraints,
            bounds=Bounds(lb, ub),
            options={"time_limit": config.time_limit_seconds, "disp": False},
        )
        if result.x is None and result.status != 2:
            return (f"the binding constraints could not be diagnosed: {result.message}",)
        if result.x is None:
            return ("constraints conflict even with all bands relaxed: check capacity",)
        slack = np.asarray(result.x[self.size :], dtype=np.float64)
        found = tuple(
            label.format(value) for label, value in zip(labels, slack, strict=True) if value > 1e-6
        )
        return found or ("infeasible only under integer lots: widen a band by one contract's size",)


def _problem_fingerprint(problem: HedgeProblem, run_stability: bool) -> str:
    """Over the dataclass repr, so that every field of every input is covered by construction.

    The stability flag is an input too, since it changes a proposal's status and issues.
    """
    return hashlib.sha256(f"{problem!r}|run_stability={run_stability}".encode()).hexdigest()


def verify_proposal(
    problem: HedgeProblem,
    snapshot: MarketSnapshot,
    legs: tuple[HedgeLeg, ...],
    predicted: dict[tuple[str, str], float],
    predicted_scenario: dict[str, float],
) -> VerificationReport:
    """Reprice the hedged book from scratch and compare with what the solver believed.

    This uses position-level valuation rather than the unit matrix the solver saw, so a bookkeeping
    error between lots, signs, multipliers, or underlyings would surface here.
    """
    hedged = (*problem.book, *(Position(leg.instrument, leg.quantity) for leg in legs))
    book = value_book(hedged, snapshot)
    worst_greek = 0.0
    violations: list[str] = _limit_violations(problem, legs)
    for target in problem.targets:
        exposure = book.by_underlying.get(target.underlying, Exposure())
        actual = float(getattr(exposure, target.greek.value))
        expected = predicted[(target.underlying, target.greek.value)]
        if not (np.isfinite(actual) and np.isfinite(expected)):
            violations.append(f"{target.greek.value} is not finite: {actual} against {expected}")
            continue
        worst_greek = max(worst_greek, abs(actual - expected))
        tolerance = 1e-6 * max(1.0, abs(actual))
        if target.lower is not None and actual < target.lower - tolerance:
            violations.append(f"{target.greek.value} {actual:.4f} below {target.lower}")
        if target.upper is not None and actual > target.upper + tolerance:
            violations.append(f"{target.greek.value} {actual:.4f} above {target.upper}")

    worst_scenario = 0.0
    if problem.scenarios:
        repriced = book_scenario_pnl(hedged, snapshot, problem.scenarios)
        for scenario, value in zip(problem.scenarios, repriced, strict=True):
            expected = predicted_scenario[scenario.name]
            if not (np.isfinite(value) and np.isfinite(expected)):
                violations.append(f"scenario {scenario.name} P&L is not finite")
                continue
            worst_scenario = max(worst_scenario, abs(float(value) - expected))

    # Dollars of tolerance: a small fraction of a cent per dollar of exposure, from float error.
    finite_predicted = [abs(v) for v in predicted.values() if np.isfinite(v)]
    finite_scenarios = [abs(v) for v in predicted_scenario.values() if np.isfinite(v)]
    scale = 1e-6 * max(1.0, sum(finite_predicted))
    scenario_scale = 1e-6 * max(1.0, sum(finite_scenarios))
    passed = worst_greek <= scale and worst_scenario <= scenario_scale and not violations
    return VerificationReport(passed, worst_greek, worst_scenario, tuple(violations))


def _limit_violations(problem: HedgeProblem, legs: tuple[HedgeLeg, ...]) -> list[str]:
    """Check hard limits from the legs themselves, independently of the program that chose them."""
    config = problem.config
    out: list[str] = []
    premium = sum(leg.premium for leg in legs)
    if config.premium_budget is not None and premium > config.premium_budget + 1e-6:
        out.append(f"net premium {premium:,.2f} is over the budget of {config.premium_budget:,.2f}")
    if config.max_legs is not None and len(legs) > config.max_legs:
        out.append(f"{len(legs)} legs is over the limit of {config.max_legs} legs")
    candidates = {c.instrument.key: c for c in problem.candidates}
    for leg in legs:
        candidate = candidates.get(leg.instrument.key)
        size = abs(leg.quantity)
        name = leg.instrument.describe()
        if candidate is None:
            out.append(f"{name} was not offered as a candidate")
            continue
        if leg.quantity % candidate.lot_size != 0:
            out.append(f"{name}: {leg.quantity} is not a multiple of lot size {candidate.lot_size}")
        limit = candidate.max_buy if leg.quantity > 0 else candidate.max_sell
        if limit is not None and size > limit:
            out.append(f"{name}: {size} is over its limit of {limit}")
        fine = isinstance(leg.instrument, LinearInstrument) and leg.instrument.multiplier == 1.0
        if not fine and size > config.max_lots_per_leg:
            out.append(f"{name}: {size} is over the leg limit of {config.max_lots_per_leg}")
    return out


def _check_targets(problem: HedgeProblem, snapshot: MarketSnapshot) -> None:
    """A target that refers to nothing is an error. Silently, a typo would read as a flat book."""
    held = {p.instrument.underlying for p in problem.book}
    offered = {c.instrument.underlying for c in problem.candidates}
    for target in problem.targets:
        if target.underlying not in snapshot.underlyings:
            raise ValidationError(f"target on {target.underlying!r}, which has no market data")
        if target.underlying not in held and target.underlying not in offered:
            raise ValidationError(
                f"target on {target.underlying!r}, with no position or candidate on it"
            )


def _uncovered_underlyings(problem: HedgeProblem, usable: list[_Usable]) -> list[str]:
    """Targeted underlyings with nothing that can trade, so the hedge cannot act on them at all."""
    tradable = {
        row.candidate.instrument.underlying
        for row in usable
        if row.buy_total_cap > 0 or row.sell_total_cap > 0
    }
    return sorted({t.underlying for t in problem.targets} - tradable)


def _coverage_issues(
    problem: HedgeProblem,
    snapshot: MarketSnapshot,
    usable: list[_Usable],
    excluded: list[ExcludedCandidate],
) -> list[str]:
    issues: list[str] = []
    if problem.candidates and not usable:
        issues.append(
            f"NO_USABLE_CANDIDATES all {len(problem.candidates)} candidates were excluded, so this "
            "says nothing about whether a hedge is needed"
        )
    else:
        for underlying in _uncovered_underlyings(problem, usable):
            offered = sum(c.instrument.underlying == underlying for c in problem.candidates)
            issues.append(
                f"NO_USABLE_CANDIDATES {underlying}: none of its {offered} candidates can trade, "
                "so its residual is unhedged, which says nothing about whether a hedge is needed"
            )
    if usable and excluded:
        share = len(excluded) / max(len(problem.candidates), 1)
        if share >= _EXCLUDED_ISSUE_SHARE:
            issues.append(
                f"EXCLUDED {len(excluded)} of {len(problem.candidates)} candidates: check the feed"
            )
    targeted = {t.underlying for t in problem.targets}
    for underlying in sorted({p.instrument.underlying for p in problem.book} - targeted):
        if problem.targets:
            issues.append(f"UNTARGETED {underlying} is in the book, with no target on it")
    for underlying, market in sorted(snapshot.underlyings.items()):
        if underlying in targeted and getattr(market.surface, "quoted_ranges", None) == ():
            issues.append(
                f"UNVERIFIED_SURFACE {underlying}: this surface has no record of its quotes, "
                "so extrapolation cannot be detected"
            )
    return issues


def _depth_issues(legs: tuple[HedgeLeg, ...], problem: HedgeProblem) -> list[str]:
    candidates = {c.instrument.key: c for c in problem.candidates}
    issues: list[str] = []
    for leg in legs:
        quote = candidates[leg.instrument.key].quote
        displayed = quote.ask_size if leg.quantity > 0 else quote.bid_size
        if displayed and abs(leg.quantity) > displayed:
            issues.append(
                f"OVER_DISPLAYED {leg.instrument.describe()}: {abs(leg.quantity)} against "
                f"{displayed} displayed, so the expected price is optimistic for the excess"
            )
    return issues


def solve_hedge(
    problem: HedgeProblem,
    snapshot: MarketSnapshot,
    *,
    run_stability: bool = False,
    gate_baseline: HedgeConfig | None = None,
) -> HedgeProposal:
    """Solve, read back, verify independently, and report. Raises rather than returning a bad plan.

    Raises HedgeInfeasibleError with the binding constraints when nothing satisfies the bands, and
    VerificationError if the solution fails its independent repricing. `gate_baseline` is what gate
    settings are disclosed against: a service's limits, or the library defaults if omitted.
    """
    from greeks_neutralizer.hedging.diagnostics import stability_report

    _check_targets(problem, snapshot)
    usable, excluded = _filter_candidates(problem, snapshot)
    book = value_book(problem.book, snapshot)
    issues = list(book.issues)
    config = problem.config
    overrides = gate_overrides(config, gate_baseline)
    issues.extend(_coverage_issues(problem, snapshot, usable, excluded))
    if overrides:
        shown = ", ".join(f"{name}={value}" for name, value in overrides.items())
        issues.append(f"GATE_SETTINGS gates differ from their standing values: {shown}")

    book_scenario = book_scenario_pnl(problem.book, snapshot, problem.scenarios)
    unit_scenario = (
        scenario_pnl_matrix([u.candidate.instrument for u in usable], snapshot, problem.scenarios)
        if usable and problem.scenarios
        else np.zeros((len(problem.scenarios), len(usable)), dtype=np.float64)
    )

    program = _Program(problem, usable, dict(book.by_underlying), book_scenario, unit_scenario)
    solution = program.solve()
    inadequate = program._beats_the_lower_bound(solution)
    if inadequate is not None:
        raise VerificationError(f"the solver missed a better hedge: {inadequate}")

    legs = tuple(
        _build_leg(row, int(quantity), problem)
        for row, quantity in zip(usable, solution.lots, strict=True)
        if quantity != 0
    )

    predicted: dict[tuple[str, str], float] = {}
    residuals: list[ResidualLine] = []
    for target in problem.targets:
        before = program.book_value(target)
        after = before + sum(
            float(quantity) * program._exposure_column(target, row)
            for row, quantity in zip(usable, solution.lots, strict=True)
        )
        predicted[(target.underlying, target.greek.value)] = after
        residuals.append(
            ResidualLine(
                target.underlying, target.greek.value, before, after, target.lower, target.upper
            )
        )

    scenario_after = (
        book_scenario + unit_scenario @ solution.lots.astype(np.float64)
        if usable and problem.scenarios
        else book_scenario
    )
    predicted_scenario = {
        s.name: float(v) for s, v in zip(problem.scenarios, scenario_after, strict=True)
    }
    verification = verify_proposal(problem, snapshot, legs, predicted, predicted_scenario)
    if not verification.passed:
        raise VerificationError(
            "solution failed independent repricing: "
            f"greek error {verification.max_abs_greek_error:.6g}, "
            f"scenario error {verification.max_abs_scenario_error:.6g}, "
            f"violations {verification.band_violations}"
        )

    hedged_book = value_book(
        (*problem.book, *(Position(leg.instrument, leg.quantity) for leg in legs)), snapshot
    )
    issues.extend(i for i in hedged_book.issues if i not in issues)
    issues.extend(_depth_issues(legs, problem))
    issues.extend(
        f"QUOTE_UNCHECKED {leg.instrument.describe()}: a proxy has no market reference, so its "
        "quote was not compared with the market"
        for leg in legs
        if isinstance(leg.instrument, LinearInstrument) and leg.instrument.kind is LinearKind.PROXY
    )
    transaction_cost = sum(leg.transaction_cost for leg in legs)
    stability = None
    if run_stability and legs:
        stability = stability_report(problem, snapshot, solution.lots, usable, program)
    basis_1d = float(
        sum(
            abs(int(q)) * row.basis_risk_per_unit_1d
            for row, q in zip(usable, solution.lots, strict=True)
        )
    )
    if basis_1d > 0.0:
        issues.append(f"BASIS_RISK proxy hedges carry about {basis_1d:,.0f} of 1-day basis risk")
    status = ProposalStatus.NO_TRADE if not legs else ProposalStatus.OPTIMAL
    if not usable:
        status = ProposalStatus.NO_USABLE_CANDIDATES
    if solution.hit_time_limit or (stability is not None and stability.hit_time_limit):
        issues.append(
            f"SOLVER_TIME_LIMIT the search stopped at the {config.time_limit_seconds:g}s backstop, "
            "so this result depends on machine speed and will not replay exactly"
        )
    if legs and solution.gap > config.mip_gap:
        status = ProposalStatus.WITHIN_GAP
        issues.append(
            f"INTEGRALITY_GAP objective {solution.objective:,.2f} against a continuous lower bound "
            f"of {solution.lower_bound:,.2f}: integer lots cost at most {solution.gap:.1%}"
        )
    if usable and _uncovered_underlyings(problem, usable):
        status = ProposalStatus.PARTIAL_COVERAGE
    if stability is not None and not stability.is_stable:
        status = ProposalStatus.UNSTABLE
        issues.extend(f"UNSTABLE {note}" for note in stability.notes)

    return HedgeProposal(
        status=status,
        legs=legs,
        objective=solution.objective,
        transaction_cost=transaction_cost,
        net_premium=sum(leg.premium for leg in legs),
        gross_lots=int(sum(abs(leg.quantity) for leg in legs)),
        residuals=tuple(residuals),
        before=dict(book.by_underlying),
        after=dict(hedged_book.by_underlying),
        scenario_pnl_before={
            s.name: float(v) for s, v in zip(problem.scenarios, book_scenario, strict=True)
        },
        scenario_pnl_after=predicted_scenario,
        excluded=tuple(excluded),
        verification=verification,
        stability=stability,
        basis_risk_1d=basis_1d,
        solver_gap=solution.gap,
        objective_lower_bound=solution.lower_bound,
        issues=tuple(issues),
        manifest=RunManifest(
            code_version=__version__,
            market_fingerprint=snapshot.fingerprint(),
            problem_fingerprint=_problem_fingerprint(problem, run_stability),
            as_of=snapshot.as_of,
            solver="scipy.optimize.milp/HiGHS",
            scipy_version=scipy_version,
            numpy_version=np.__version__,
            run_stability=run_stability,
            gate_overrides=overrides,
        ),
    )


def _build_leg(row: _Usable, quantity: int, problem: HedgeProblem) -> HedgeLeg:
    candidate = row.candidate
    quote = candidate.quote
    instrument = candidate.instrument
    buying = quantity > 0
    size = abs(quantity)
    price = quote.ask if buying else quote.bid
    touch_units = (row.buy_touch_cap if buying else row.sell_touch_cap) * candidate.lot_size
    deep_units = max(size - touch_units, 0)
    half_and_fee = row.buy_cost_touch if buying else row.sell_cost_touch
    cost = size * half_and_fee + deep_units * candidate.impact_per_unit
    premium = quantity * row.mid_value
    return HedgeLeg(
        instrument=instrument,
        quantity=quantity,
        expected_price=price,
        premium=premium,
        transaction_cost=cost,
        rationale=_rationale(row, quantity, problem, cost),
    )


def _rationale(row: _Usable, quantity: int, problem: HedgeProblem, cost: float) -> str:
    """One line of plain language: what this leg does to the risks the user asked about."""
    parts: list[str] = []
    for target in problem.targets:
        if target.underlying != row.candidate.instrument.underlying:
            continue
        effect = quantity * float(getattr(row.exposure, target.greek.value))
        if abs(effect) > 1e-9:
            parts.append(f"{target.greek.value} {effect:+,.0f}")
    verb = "Buy" if quantity > 0 else "Sell"
    effects = ", ".join(parts) if parts else "scenario protection only"
    return (
        f"{verb} {abs(quantity)} {row.candidate.instrument.describe()}: {effects}; cost {cost:,.2f}"
    )


__all__ = ["HedgeConfig", "VerificationError", "solve_hedge", "verify_proposal"]
