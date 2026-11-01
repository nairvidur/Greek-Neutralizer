"""Daily hedging backtest with honest accounting.

No look-ahead by construction: `run_backtest` pulls one `DayMarket` at a time from an iterator, and
decisions at day t receive only that day's market. P&L from t to t+1 is measured by marking the
positions held after t's trades in t+1's market.

Daily P&L = sum of qty * (mark(t+1) - mark(t)) - financing + carry income - costs paid at t.

Financing is on each position's marked value, at its underlying's rate. A future has no financed
value: its carry is already in the forward's roll. A shareholder receives the carry yield and any
dividends that go ex in the period, and a short pays them. Without that, a stock hedge across an
ex-date would book the price drop as profit.

Trades pay half the bid/ask plus fees, with marks at mid. An option expiring by t+1 is marked at
intrinsic in t+1's market and then removed, so settlement needs no separate cash path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np

from greeks_neutralizer.domain import Instrument, LinearInstrument, LinearKind, Position
from greeks_neutralizer.errors import GNError, ValidationError
from greeks_neutralizer.hedging.problem import HedgeCandidate
from greeks_neutralizer.market import MarketSnapshot
from greeks_neutralizer.pricing.black import FloatArray
from greeks_neutralizer.pricing.book import is_expired_instrument, mark_value, value_book
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
    carry_income: float = 0.0


@dataclass(slots=True)
class BacktestResult:
    policy: str
    days: list[DayResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    final_day_cost: float = 0.0  # paid on the last day, with no later day to report it against

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
    """Half-spread and fees on all units, plus impact on the units past the displayed size."""
    total = 0.0
    for trade in trades:
        if trade.quantity == 0:
            continue
        candidate = day.quote_for(trade.instrument)
        half = candidate.quote.half_spread * trade.instrument.multiplier
        size = abs(trade.quantity)
        displayed = candidate.quote.ask_size if trade.quantity > 0 else candidate.quote.bid_size
        total += size * (half + candidate.fee_per_unit)
        total += max(size - displayed, 0) * candidate.impact_per_unit
    return total


def _value(holdings: Mapping[str, Position], snapshot: MarketSnapshot) -> dict[str, float]:
    """Mark of each position, by key. Marks use the model; a real source can supply mids."""
    positions = list(holdings.values())
    if not positions:
        return {}
    book = value_book(positions, snapshot)
    return {
        line.position.instrument.key: mark_value(
            line.position, line.unit_price, line.exposure.value
        )
        for line in book.lines
    }


def _drop_expired(holdings: dict[str, Position], as_of: datetime) -> None:
    for key in [
        k for k, position in holdings.items() if is_expired_instrument(position.instrument, as_of)
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
            held = {**book, **hedges}
            marks = {**book_marks, **hedge_marks}
            if len(held) != len(book) + len(hedges):
                # One instrument in both the book and the hedges: sum its marks and quantities.
                marks = {
                    key: book_marks.get(key, 0.0) + hedge_marks.get(key, 0.0)
                    for key in {*book_marks, *hedge_marks}
                }
                held = dict(book)
                _merge(held, hedges.values())
            financing = _financing(held, marks, previous, day)
            income = _carry_income(held, previous, day)
            total_cost = pending_book_cost + pending_hedge_cost
            result.days.append(
                DayResult(
                    as_of=day.as_of,
                    pnl=book_pnl + hedge_pnl - financing + income - total_cost,
                    book_pnl=book_pnl,
                    hedge_pnl=hedge_pnl,
                    financing=financing,
                    hedge_cost=pending_hedge_cost,
                    book_cost=pending_book_cost,
                    hedge_lots=pending_lots,
                    failures=pending_failures,
                    carry_income=income,
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
        except GNError as exc:
            # Expected failures to hedge. A programming error still raises, rather than being
            # counted as a day when no hedge was available.
            if fail_fast:
                raise
            pending_hedge_cost = 0.0
            pending_lots = 0
            pending_failures = 1
            result.failures.append(f"{day.as_of.date()}: {type(exc).__name__}: {exc}")

        book_marks = _value(book, day.snapshot)
        hedge_marks = _value(hedges, day.snapshot)
        previous = day
    result.final_day_cost = pending_book_cost + pending_hedge_cost
    return result


def _is_future(instrument: Instrument) -> bool:
    return isinstance(instrument, LinearInstrument) and instrument.kind is LinearKind.FUTURE


def _financing(
    held: Mapping[str, Position], marks: Mapping[str, float], start: DayMarket, end: DayMarket
) -> float:
    """Interest on the marked value over the period, at each underlying's own rate."""
    dt = calendar_years(start.as_of, end.as_of)
    total = 0.0
    for key, position in held.items():
        instrument = position.instrument
        if _is_future(instrument):
            continue
        market = start.snapshot.market(instrument.underlying)
        rate = float(market.curve.zero_rate(np.array([max(dt, 1e-6)]))[0])
        total += marks[key] * rate * dt
    return total


def _carry_income(held: Mapping[str, Position], start: DayMarket, end: DayMarket) -> float:
    """Dividends and carry yield on stock and proxy holdings. Negative for a short."""
    dt = calendar_years(start.as_of, end.as_of)
    total = 0.0
    for position in held.values():
        instrument = position.instrument
        if not isinstance(instrument, LinearInstrument) or _is_future(instrument):
            continue
        market = start.snapshot.market(instrument.underlying)
        units = position.quantity * instrument.multiplier * instrument.delta_ratio
        total += units * market.spot * market.carry_yield * dt
        for dividend in market.dividends:
            if start.as_of < dividend.ex_date <= end.as_of:
                total += units * dividend.amount
    return total
