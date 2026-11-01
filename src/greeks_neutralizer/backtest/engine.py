"""Daily hedging backtest with honest accounting.

No look-ahead by construction: `run_backtest` pulls one `DayMarket` at a time from an iterator, and
decisions at day t receive only that day's market. P&L from t to t+1 is measured by marking the
positions held after t's trades in t+1's market.

Daily P&L = sum of qty * (mark(t+1) - mark(t)) - financing on the marked value - costs paid at t.

Trades pay half the bid/ask plus fees, with marks at mid. An option expiring by t+1 is marked at
intrinsic in t+1's market and then removed, so settlement needs no separate cash path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np

from greeks_neutralizer.domain import Instrument, OptionContract, Position
from greeks_neutralizer.errors import ValidationError
from greeks_neutralizer.hedging.problem import HedgeCandidate
from greeks_neutralizer.market import MarketSnapshot
from greeks_neutralizer.pricing.black import FloatArray
from greeks_neutralizer.pricing.book import value_book
from greeks_neutralizer.time_model import calendar_years


@dataclass(frozen=True, slots=True)
class DayMarket:
    as_of: datetime
    snapshot: MarketSnapshot
    candidates: tuple[HedgeCandidate, ...]
    quote_for: Callable[[Instrument], HedgeCandidate]


class BookStrategy(Protocol):
    """The risk being hedged. Trades here are the same for every policy, so differences in results
    come only from hedging."""

    def trades(self, day: DayMarket, holdings: Mapping[str, Position]) -> Sequence[Position]: ...


class HedgePolicy(Protocol):
    name: str

    def trades(
        self, day: DayMarket, book: Sequence[Position], hedges: Sequence[Position]
    ) -> Sequence[Position]: ...


@dataclass(slots=True)
class DayResult:
    as_of: datetime
    pnl: float
    book_pnl: float
    hedge_pnl: float
    financing: float
    hedge_cost: float
    book_cost: float
    hedge_lots: int
    failures: int


@dataclass(slots=True)
class BacktestResult:
    policy: str
    days: list[DayResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def pnl(self) -> FloatArray:
        return np.array([d.pnl for d in self.days], dtype=np.float64)

    def dates(self) -> list[datetime]:
        return [d.as_of for d in self.days]


def _merge(holdings: dict[str, Position], trades: Iterable[Position]) -> None:
    for trade in trades:
        key = trade.instrument.key
        current = holdings.get(key)
        quantity = trade.quantity + (current.quantity if current else 0)
        if quantity == 0:
            holdings.pop(key, None)
        else:
            holdings[key] = Position(trade.instrument, quantity)


def _trade_cost(day: DayMarket, trades: Iterable[Position]) -> float:
    total = 0.0
    for trade in trades:
        if trade.quantity == 0:
            continue
        candidate = day.quote_for(trade.instrument)
        half = candidate.quote.half_spread * trade.instrument.multiplier
        total += abs(trade.quantity) * (half + candidate.fee_per_unit)
    return total


def _value(holdings: Mapping[str, Position], snapshot: MarketSnapshot) -> dict[str, float]:
    """Mark of each position, by key. Marks use the model; a real source can supply mids."""
    positions = list(holdings.values())
    if not positions:
        return {}
    book = value_book(positions, snapshot)
    return {
        line.position.instrument.key: _linear_aware_value(
            line.position, line.unit_price, line.exposure.value
        )
        for line in book.lines
    }


def _linear_aware_value(position: Position, unit_price: float, exposure_value: float) -> float:
    """A future's exposure value is zero at inception, but its mark moves with its price."""
    instrument = position.instrument
    if isinstance(instrument, OptionContract):
        return exposure_value
    return position.quantity * instrument.multiplier * unit_price


def _drop_expired(holdings: dict[str, Position], as_of: datetime) -> None:
    for key in [
        k
        for k, position in holdings.items()
        if isinstance(position.instrument, OptionContract) and position.instrument.expiry <= as_of
    ]:
        del holdings[key]


def run_backtest(
    market: Iterable[DayMarket],
    book_strategy: BookStrategy,
    policy: HedgePolicy,
    *,
    fail_fast: bool = False,
) -> BacktestResult:
    """Run one policy. A day where the policy cannot produce a hedge holds its previous hedge.

    Such days are recorded, since an unavailable hedge is a real outcome of a hedging policy. With
    fail_fast they raise instead.
    """
    result = BacktestResult(policy=policy.name)
    book: dict[str, Position] = {}
    hedges: dict[str, Position] = {}
    previous: DayMarket | None = None
    book_marks: dict[str, float] = {}
    hedge_marks: dict[str, float] = {}
    pending_book_cost = 0.0
    pending_hedge_cost = 0.0
    pending_lots = 0
    pending_failures = 0

    for day in market:
        if previous is not None and day.as_of <= previous.as_of:
            raise ValidationError("market days must be strictly increasing")
        if previous is not None:
            book_now = _value(book, day.snapshot)
            hedge_now = _value(hedges, day.snapshot)
            book_pnl = sum(book_now[k] - book_marks[k] for k in book_now)
            hedge_pnl = sum(hedge_now[k] - hedge_marks[k] for k in hedge_now)
            financing = _financing(book_marks, hedge_marks, previous, day)
            total_cost = pending_book_cost + pending_hedge_cost
            result.days.append(
                DayResult(
                    as_of=day.as_of,
                    pnl=book_pnl + hedge_pnl - financing - total_cost,
                    book_pnl=book_pnl,
                    hedge_pnl=hedge_pnl,
                    financing=financing,
                    hedge_cost=pending_hedge_cost,
                    book_cost=pending_book_cost,
                    hedge_lots=pending_lots,
                    failures=pending_failures,
                )
            )
        _drop_expired(book, day.as_of)
        _drop_expired(hedges, day.as_of)

        book_trades = [t for t in book_strategy.trades(day, book) if t.quantity != 0]
        pending_book_cost = _trade_cost(day, book_trades)
        _merge(book, book_trades)

        pending_failures = 0
        try:
            hedge_trades = [
                t
                for t in policy.trades(day, list(book.values()), list(hedges.values()))
                if t.quantity != 0
            ]
            pending_hedge_cost = _trade_cost(day, hedge_trades)
            pending_lots = sum(abs(t.quantity) for t in hedge_trades)
            _merge(hedges, hedge_trades)
        except Exception as exc:
            if fail_fast:
                raise
            pending_hedge_cost = 0.0
            pending_lots = 0
            pending_failures = 1
            result.failures.append(f"{day.as_of.date()}: {type(exc).__name__}: {exc}")

        book_marks = _value(book, day.snapshot)
        hedge_marks = _value(hedges, day.snapshot)
        previous = day
    return result


def _financing(
    book_marks: Mapping[str, float],
    hedge_marks: Mapping[str, float],
    start: DayMarket,
    end: DayMarket,
) -> float:
    """Interest on the marked value held over the period, so all policies are self-financing."""
    dt = calendar_years(start.as_of, end.as_of)
    held = sum(book_marks.values()) + sum(hedge_marks.values())
    rates = [m.curve.zero_rate(np.array([0.25]))[0] for m in start.snapshot.underlyings.values()]
    rate = float(rates[0]) if rates else 0.0
    return held * rate * dt
