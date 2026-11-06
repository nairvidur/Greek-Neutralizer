"""Hedge policies: the baselines a hedge must beat, and the optimizer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from greeks_neutralizer.backtest.engine import DayMarket, HedgePolicy
from greeks_neutralizer.domain import LinearInstrument, Position
from greeks_neutralizer.errors import SearchLimitError
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
from greeks_neutralizer.risk.scenarios import HISTORICAL_STRESSES, Scenario


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
    """

    name: str
    targets: tuple[GreekTarget, ...]
    config: HedgeConfig
    scenarios: tuple[Scenario, ...] = ()

    def trades(
        self, day: DayMarket, book: Sequence[Position], hedges: Sequence[Position]
    ) -> Sequence[Position]:
        offered = {c.instrument.key: c for c in day.candidates}
        for position in hedges:
            key = position.instrument.key
            if key not in offered:
                offered[key] = day.quote_for(position.instrument)
        candidates: tuple[HedgeCandidate, ...] = tuple(offered.values())
        problem = HedgeProblem(
            (*book, *hedges),
            candidates,
            self.targets,
            self.scenarios,
            self.config,
        )
        proposal = solve_hedge(problem, day.snapshot)
        timed_out = [i for i in proposal.issues if i.startswith("SOLVER_TIME_LIMIT")]
        if timed_out:
            # Its hedge depends on how loaded the machine was, so trading it would make the
            # backtest irreproducible. The day is recorded as failed instead.
            raise SearchLimitError(timed_out[0])
        return tuple(Position(leg.instrument, leg.quantity) for leg in proposal.legs)


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


# The search stops at HedgeConfig's node limit, which is deterministic. The clock is only a
# backstop, set far above any solve here so that it never binds on a loaded machine. At the
# default 30s it did bind on a starved machine, and changed an early chain run's results.
_BACKTEST_TIME_LIMIT_SECONDS = 600.0


def _registered_config(risk_aversion: float, cvar_weight: float = 0.0) -> HedgeConfig:
    return HedgeConfig(
        risk_aversion=risk_aversion,
        cvar_weight=cvar_weight,
        cvar_alpha=0.8,
        mip_gap=1e-2,
        max_legs=6,
        time_limit_seconds=_BACKTEST_TIME_LIMIT_SECONDS,
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
    raise ValueError(f"not a registered policy: {name!r}")


__all__ = [
    "REGISTERED_POLICIES",
    "DeltaBand",
    "DeltaDaily",
    "OptimizerPolicy",
    "Unhedged",
    "registered_policy",
]
