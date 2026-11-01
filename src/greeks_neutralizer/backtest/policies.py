"""Hedge policies: the baselines a hedge must beat, and the optimizer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from greeks_neutralizer.backtest.engine import DayMarket
from greeks_neutralizer.domain import LinearInstrument, Position
from greeks_neutralizer.hedging.diagnostics import whalley_wilmott_band
from greeks_neutralizer.hedging.optimizer import solve_hedge
from greeks_neutralizer.hedging.problem import (
    GreekTarget,
    HedgeCandidate,
    HedgeConfig,
    HedgeProblem,
)
from greeks_neutralizer.pricing.book import Exposure, linear_unit_exposure, value_book
from greeks_neutralizer.risk.scenarios import Scenario


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
        return tuple(Position(leg.instrument, leg.quantity) for leg in proposal.legs)


def _exposure(day: DayMarket, positions: Sequence[Position], underlying: str) -> Exposure:
    if not positions:
        return Exposure()
    return value_book(positions, day.snapshot).by_underlying.get(underlying, Exposure())


__all__ = ["DeltaBand", "DeltaDaily", "OptimizerPolicy", "Unhedged"]
