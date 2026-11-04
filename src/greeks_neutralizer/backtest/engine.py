"""Daily hedging backtest with honest accounting.

No look-ahead by construction: `run_backtest` pulls one `DayMarket` at a time from an iterator, and
decisions at day t receive only that day's market. P&L from t to t+1 is measured by marking the
positions held after t's trades in t+1's market.

Daily P&L = sum of qty * (mark(t+1) - mark(t)) - financing + carry income - costs paid at t.

Financing is on each position's marked value, at its underlying's rate. A future has no financed
value: its carry is already in the forward's roll. A shareholder receives the carry yield and any
dividends that go ex in the period, and a short pays them. Without that, a stock hedge across an
ex-date would book the price drop as profit.

Trades pay half the bid/ask plus fees. Marks come from the model, unless the market supplies an
observed mid for an option, as a historical chain does: then that option is marked at its mid, so
P&L is measured on market prices. Each period counts the option positions whose mark at either
end came from the model, so a report can say how much of its P&L rests on the model.

An option expiring by t+1 is marked at intrinsic in t+1's market and then removed, so settlement
needs no separate cash path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np

from greeks_neutralizer.domain import (
    Instrument,
    LinearInstrument,
    LinearKind,
    OptionContract,
    Position,
)
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
    # The observed mid per unit of an option quoted that day, or None where it was not quoted.
    # Left unset, every mark comes from the model.
    observed_mid: Callable[[OptionContract], float | None] | None = None


class BookStrategy(Protocol):
    """The risk being hedged. Trades here are the same for every policy, so differences in results
    come only from hedging."""

    def trades(self, day: DayMarket, holdings: Mapping[str, Position]) -> Sequence[Position]: ...


class HedgePolicy(Protocol):
    # Read-only, so immutable policies such as frozen dataclasses satisfy the protocol.
    @property
    def name(self) -> str: ...

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
    # Option positions held over the period whose mark at either end came from the model, out of
    # all option positions held over it. Both are 0 when the market supplies no observed mids.
    model_marked: int = 0
    option_positions: int = 0
    # Sum over model-marked option positions of (units held)^2 times the number of its two marks
    # that came from the model. Times the variance of a model mark's error per unit, it is the
    # P&L variance that mark errors alone add to this period, if those errors are independent.
    modeled_units_sq: float = 0.0


@dataclass(slots=True)
class BacktestResult:
    policy: str
    days: list[DayResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    final_day_cost: float = 0.0  # paid on the last day, with no later day to report it against
    # The decision day of each entry in `failures`, in the same order.
    failure_dates: list[datetime] = field(default_factory=list)

    def after_warmup(self, warmup_days: int) -> BacktestResult:
        """The result with the first `warmup_days` P&L days dropped.

        A failure is kept only if it could affect a kept day: its decision day is on or after the
        decision day of the first kept P&L day, which is the market day before it, that is, the
        as_of of the last dropped day. Failures during the warm-up are dropped with it.
        """
        if warmup_days < 0:
            raise ValidationError("warmup_days must be non-negative")
        if len(self.failure_dates) != len(self.failures):
            raise ValidationError(f"{self.policy}: failures are not dated, so cannot be trimmed")
        if warmup_days == 0:
            return BacktestResult(
                self.policy,
                list(self.days),
                list(self.failures),
                self.final_day_cost,
                list(self.failure_dates),
            )
        if warmup_days > len(self.days):
            raise ValidationError(
                f"{self.policy}: warm-up of {warmup_days} days exceeds {len(self.days)} days"
            )
        cutoff = self.days[warmup_days - 1].as_of
        kept = [
            (m, d) for m, d in zip(self.failures, self.failure_dates, strict=True) if d >= cutoff
        ]
        return BacktestResult(
            self.policy,
            self.days[warmup_days:],
            [m for m, _ in kept],
            self.final_day_cost,
            [d for _, d in kept],
        )

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


@dataclass(frozen=True, slots=True)
class _Marks:
    values: dict[str, float]
    # Keys of option positions marked by the model although the market supplies observed mids.
    modeled: frozenset[str]


def _value(holdings: Mapping[str, Position], day: DayMarket) -> _Marks:
    """Mark of each position, by key: the observed mid where there is one, else the model."""
    positions = list(holdings.values())
    if not positions:
        return _Marks({}, frozenset())
    book = value_book(positions, day.snapshot)
    values = {
        line.position.instrument.key: mark_value(
            line.position, line.unit_price, line.exposure.value
        )
        for line in book.lines
    }
    modeled: set[str] = set()
    if day.observed_mid is not None:
        for key, position in holdings.items():
            instrument = position.instrument
            if not isinstance(instrument, OptionContract):
                continue
            mid = None if instrument.expiry <= day.as_of else day.observed_mid(instrument)
            if mid is None:
                # Expired options settle at intrinsic, which is a market value, not a model one.
                if instrument.expiry > day.as_of:
                    modeled.add(key)
                continue
            values[key] = position.quantity * instrument.multiplier * mid
    return _Marks(values, frozenset(modeled))


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
    modeled_at_start: frozenset[str] = frozenset()
    pending_book_cost = 0.0
    pending_hedge_cost = 0.0
    pending_lots = 0
    pending_failures = 0

    for day in market:
        if previous is not None and day.as_of <= previous.as_of:
            raise ValidationError("market days must be strictly increasing")
        if previous is not None:
            book_end = _value(book, day)
            hedge_end = _value(hedges, day)
            book_now, hedge_now = book_end.values, hedge_end.values
            book_pnl = sum(book_now[k] - book_marks[k] for k in book_now)
            hedge_pnl = sum(hedge_now[k] - hedge_marks[k] for k in hedge_now)
            option_keys = {
                k
                for k, p in (*book.items(), *hedges.items())
                if isinstance(p.instrument, OptionContract)
            }
            modeled = (modeled_at_start | book_end.modeled | hedge_end.modeled) & option_keys
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
            end_modeled = book_end.modeled | hedge_end.modeled
            modeled_units_sq = sum(
                (position.quantity * position.instrument.multiplier) ** 2
                * ((key in modeled_at_start) + (key in end_modeled))
                for key, position in held.items()
                if key in modeled
            )
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
                    model_marked=len(modeled) if day.observed_mid is not None else 0,
                    option_positions=len(option_keys) if day.observed_mid is not None else 0,
                    modeled_units_sq=float(modeled_units_sq),
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
            result.failure_dates.append(day.as_of)

        book_start = _value(book, day)
        hedge_start = _value(hedges, day)
        book_marks, hedge_marks = book_start.values, hedge_start.values
        modeled_at_start = book_start.modeled | hedge_start.modeled
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
