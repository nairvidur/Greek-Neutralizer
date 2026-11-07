"""Hedge policies: the baselines a hedge must beat, and the optimizer."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from greeks_neutralizer.backtest.engine import DayMarket, HedgePolicy
from greeks_neutralizer.domain import LinearInstrument, OptionContract, Position
from greeks_neutralizer.errors import HedgeInfeasibleError, SearchLimitError
from greeks_neutralizer.hedging.diagnostics import whalley_wilmott_band
from greeks_neutralizer.hedging.optimizer import solve_hedge
from greeks_neutralizer.hedging.problem import (
    GreekName,
    GreekTarget,
    HedgeCandidate,
    HedgeConfig,
    HedgeProblem,
)
from greeks_neutralizer.pricing.book import Exposure, linear_unit_exposure, value_book
from greeks_neutralizer.risk.scenarios import HISTORICAL_STRESSES, SURFACE_SHAPE_SHOCKS, Scenario


@dataclass(frozen=True, slots=True)
class Unhedged:
    name: str = "unhedged"

    def trades(
        self, day: DayMarket, book: Sequence[Position], hedges: Sequence[Position]
    ) -> Sequence[Position]:
        return ()


def _unit_delta(day: DayMarket, instrument: LinearInstrument) -> float:
    """Delta shares of one unit. For a future that includes the carry, so it is not its size."""
    market = day.snapshot.market(instrument.underlying)
    _, delta = linear_unit_exposure(instrument, market, day.as_of)
    return instrument.multiplier * delta


def _linear(day: DayMarket, underlying: str) -> LinearInstrument:
    for candidate in day.candidates:
        instrument = candidate.instrument
        if isinstance(instrument, LinearInstrument) and instrument.underlying == underlying:
            return instrument
    raise ValueError(f"no linear hedge available for {underlying}")


@dataclass(frozen=True, slots=True)
class DeltaDaily:
    """Flatten delta in the linear instrument every day: the industry's default baseline."""

    underlying: str
    name: str = "delta_daily"

    def trades(
        self, day: DayMarket, book: Sequence[Position], hedges: Sequence[Position]
    ) -> Sequence[Position]:
        instrument = _linear(day, self.underlying)
        delta = _exposure(day, (*book, *hedges), self.underlying).delta_shares
        unit = _unit_delta(day, instrument)
        return (Position(instrument, -round(delta / unit)),)


@dataclass(frozen=True, slots=True)
class DeltaBand:
    """Rebalance only outside a Whalley-Wilmott no-trade band: delta with cost awareness."""

    underlying: str
    proportional_cost: float
    risk_aversion: float
    name: str = "delta_band"

    def trades(
        self, day: DayMarket, book: Sequence[Position], hedges: Sequence[Position]
    ) -> Sequence[Position]:
        instrument = _linear(day, self.underlying)
        exposure = _exposure(day, (*book, *hedges), self.underlying)
        band = whalley_wilmott_band(
            exposure.gamma_shares,
            day.snapshot.market(self.underlying).spot,
            self.proportional_cost,
            self.risk_aversion,
        )
        if not band.should_rebalance(exposure.delta_shares):
            return ()
        # Trade back to the edge of the band, the optimal rule under proportional costs.
        edge = band.half_width_shares if exposure.delta_shares > 0 else -band.half_width_shares
        unit = _unit_delta(day, instrument)
        return (Position(instrument, -round((exposure.delta_shares - edge) / unit)),)


@dataclass(frozen=True, slots=True)
class OptimizerPolicy:
    """The product: the hedge optimizer, seeing current hedges as part of the book.

    Current hedges are offered back as candidates, so unwinding one pays its real spread.

    `gross_multiple` caps the hedges' gross option contracts at that multiple of the book's own,
    so hedges can neither stack up nor grow past the risk they hedge. Held hedges are passed as
    the problem's `held_hedges`, so exit costs and the cap fall on them and never on the book.
    `option_band` trades options only when residual gamma or vega is outside that share of the
    book's own: inside it, only the linear hedge is offered, and the day's trade is a delta hedge.
    """

    name: str
    targets: tuple[GreekTarget, ...]
    config: HedgeConfig
    scenarios: tuple[Scenario, ...] = ()
    gross_multiple: float | None = None
    option_band: float | None = None

    def trades(
        self, day: DayMarket, book: Sequence[Position], hedges: Sequence[Position]
    ) -> Sequence[Position]:
        offered = {c.instrument.key: c for c in day.candidates}
        for position in hedges:
            key = position.instrument.key
            if key not in offered:
                offered[key] = day.quote_for(position.instrument)
        candidates: tuple[HedgeCandidate, ...] = tuple(offered.values())
        options_offered = self.option_band is None or not self._inside_band(day, book, hedges)
        if not options_offered:
            candidates = tuple(c for c in candidates if isinstance(c.instrument, LinearInstrument))
        problem = HedgeProblem(
            (*book, *hedges),
            candidates,
            self.targets,
            self.scenarios,
            self.config,
            tuple(hedges),
        )
        # With no options offered the cap cannot be acted on, so it is left off rather than
        # failing a delta-only day on hedges held from before the book shrank.
        if self.gross_multiple is None or not options_offered:
            proposal = solve_hedge(problem, day.snapshot)
        else:
            own = _option_contracts(book)
            capped = replace(
                self.config, max_gross_option_contracts=math.floor(self.gross_multiple * own)
            )
            try:
                proposal = solve_hedge(replace(problem, config=capped), day.snapshot)
            except HedgeInfeasibleError:
                # The book shrank below its hedges, by more than the displayed size can unwind in a
                # day. The hedges may then shrink but not grow, which is always feasible for the
                # cap, so the day's delta hedge is not lost with it.
                held = replace(self.config, max_gross_option_contracts=_option_contracts(hedges))
                proposal = solve_hedge(replace(problem, config=held), day.snapshot)
        timed_out = [i for i in proposal.issues if i.startswith("SOLVER_TIME_LIMIT")]
        if timed_out:
            # Its hedge depends on how loaded the machine was, so trading it would make the
            # backtest irreproducible. The day is recorded as failed instead.
            raise SearchLimitError(timed_out[0])
        return tuple(Position(leg.instrument, leg.quantity) for leg in proposal.legs)

    def _inside_band(
        self, day: DayMarket, book: Sequence[Position], hedges: Sequence[Position]
    ) -> bool:
        """Residual gamma and vega both within option_band of the book's own, per underlying."""
        assert self.option_band is not None
        underlyings = {t.underlying for t in self.targets}
        for underlying in underlyings:
            own = _exposure(day, book, underlying)
            residual = _exposure(day, (*book, *hedges), underlying)
            for greek in (GreekName.GAMMA, GreekName.VEGA):
                if not any(t.greek is greek and t.underlying == underlying for t in self.targets):
                    continue
                limit = self.option_band * abs(float(getattr(own, greek.value)))
                if abs(float(getattr(residual, greek.value))) > limit:
                    return False
        return True


def _option_contracts(positions: Sequence[Position]) -> int:
    """Absolute option contracts, netted per instrument."""
    net: dict[str, int] = {}
    for position in positions:
        if isinstance(position.instrument, OptionContract):
            key = position.instrument.key
            net[key] = net.get(key, 0) + position.quantity
    return sum(abs(q) for q in net.values())


def _exposure(day: DayMarket, positions: Sequence[Position], underlying: str) -> Exposure:
    if not positions:
        return Exposure()
    return value_book(positions, day.snapshot).by_underlying.get(underlying, Exposure())


# The pre-registered policies, fixed before the first proxy run. Every backtest builds them here,
# so the proxy and the historical-chain runs test exactly the same policies.
REGISTERED_POLICIES = (
    "unhedged",
    "delta_daily",
    "delta_band",
    "opt_dg",
    "opt_dgv",
    "opt_dgv_cheap",
    "opt_dgv_cvar",
)


# The optimizer variants added after the phase 4 no-go, which bound the position the hedge leaves
# behind. They were designed on the SPY chains the no-go came from, so SPY results for them are
# in-sample. Their test is a run on an underlying held out until they were fixed, as entered in
# docs/evidence/phase4/ledger.md before that run.
POSITION_AWARE_POLICIES = (
    "opt_dgv_bounded",
    "opt_dgv_bounded_band",
    "opt_dgv_surface",
)
# Hedge option contracts up to the book's own, so the hedged book holds at most twice the book's.
_GROSS_MULTIPLE = 1.0
_OPTION_BAND = 0.5


# The search stops at HedgeConfig's node limit, which is deterministic. The clock is only a
# backstop, set far above any solve here so that it never binds on a loaded machine. At the
# default 30s it did bind on a starved machine, and changed an early chain run's results.
_BACKTEST_TIME_LIMIT_SECONDS = 600.0


def _registered_config(
    risk_aversion: float, cvar_weight: float = 0.0, *, price_exit: bool = False
) -> HedgeConfig:
    return HedgeConfig(
        risk_aversion=risk_aversion,
        cvar_weight=cvar_weight,
        cvar_alpha=0.8,
        mip_gap=1e-2,
        max_legs=6,
        time_limit_seconds=_BACKTEST_TIME_LIMIT_SECONDS,
        price_exit=price_exit,
    )


def registered_policy(name: str, symbol: str) -> HedgePolicy:
    """A pre-registered policy on `symbol`. Unknown names raise, so none is invented later."""
    delta, gamma, vega = GreekName.DELTA, GreekName.GAMMA, GreekName.VEGA

    def targets(*greeks: GreekName) -> tuple[GreekTarget, ...]:
        return tuple(GreekTarget(symbol, greek) for greek in greeks)

    if name == "unhedged":
        return Unhedged()
    if name == "delta_daily":
        return DeltaDaily(symbol)
    if name == "delta_band":
        return DeltaBand(symbol, proportional_cost=5e-5, risk_aversion=1e-6)
    if name == "opt_dg":
        return OptimizerPolicy(name, targets(delta, gamma), _registered_config(1.0))
    if name == "opt_dgv":
        return OptimizerPolicy(name, targets(delta, gamma, vega), _registered_config(1.0))
    if name == "opt_dgv_cheap":
        return OptimizerPolicy(name, targets(delta, gamma, vega), _registered_config(0.2))
    if name == "opt_dgv_cvar":
        return OptimizerPolicy(
            name, targets(delta, gamma, vega), _registered_config(0.2, 0.2), HISTORICAL_STRESSES
        )
    dgv = targets(delta, gamma, vega)
    bounded = _registered_config(1.0, price_exit=True)
    if name == "opt_dgv_bounded":
        return OptimizerPolicy(name, dgv, bounded, gross_multiple=_GROSS_MULTIPLE)
    if name == "opt_dgv_bounded_band":
        return OptimizerPolicy(
            name, dgv, bounded, gross_multiple=_GROSS_MULTIPLE, option_band=_OPTION_BAND
        )
    if name == "opt_dgv_surface":
        return OptimizerPolicy(
            name,
            dgv,
            _registered_config(1.0, 1.0, price_exit=True),
            SURFACE_SHAPE_SHOCKS,
            gross_multiple=_GROSS_MULTIPLE,
            option_band=_OPTION_BAND,
        )
    raise ValueError(f"not a registered policy: {name!r}")


__all__ = [
    "POSITION_AWARE_POLICIES",
    "REGISTERED_POLICIES",
    "DeltaBand",
    "DeltaDaily",
    "OptimizerPolicy",
    "Unhedged",
    "registered_policy",
]
