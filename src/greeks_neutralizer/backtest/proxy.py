"""A proxy option market built from real index, VIX, and rate history.

THIS IS NOT OPTION DATA. Spot, VIX, and rates are real closes. Option prices are not: they come from
a parametric smile anchored to the VIX, with modeled spreads, so the market is free of noise, skew
dynamics, and dislocations that real quotes have. Hedge effectiveness here is therefore likely
overstated for option hedges. Use it to exercise the system and rank policies on real paths. It does
not replace a backtest on historical chains, and nothing here may be presented as if it were.

Every value on day t uses only data up to t: the long-run vol level is a trailing mean, and the VIX
is the same-day close.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np

from greeks_neutralizer.backtest.engine import DayMarket
from greeks_neutralizer.domain import (
    ExerciseStyle,
    Instrument,
    LinearInstrument,
    OptionContract,
    OptionRight,
    Quote,
    SettlementKind,
)
from greeks_neutralizer.errors import MarketDataError, ValidationError
from greeks_neutralizer.hedging.problem import HedgeCandidate
from greeks_neutralizer.market import DiscountCurve, MarketSnapshot, UnderlyingMarket
from greeks_neutralizer.pricing.black import FloatArray, black_premium
from greeks_neutralizer.time_model import NEW_YORK, calendar_years, us_expiry

SYMBOL = "SPX"
_TERM_DAYS = 60.0 / 365.0
_REF_TAU = 30.0 / 365.0


@dataclass(frozen=True, slots=True)
class ProxySmile:
    """vol(k, tau) = atm(tau) * (1 + skew z + curvature z^2) in z = k / (atm sqrt(tau)), with ATM
    mean-reverting across tenor from the 30-day level to a trailing long-run level. Wings are capped
    so total variance keeps a sane slope. Not fitted to anything: a stylized surface."""

    atm_30d: float
    long_run: float
    skew: float = -0.10
    curvature: float = 0.02

    def atm(self, tau: FloatArray) -> FloatArray:
        weight = np.exp(-np.maximum(tau - _REF_TAU, 0.0) / _TERM_DAYS)
        return np.asarray(self.long_run + (self.atm_30d - self.long_run) * weight, dtype=np.float64)

    def vol(self, k: FloatArray, tau: FloatArray) -> FloatArray:
        k_, tau_ = np.broadcast_arrays(
            np.asarray(k, dtype=np.float64), np.asarray(tau, dtype=np.float64)
        )
        atm = self.atm(tau_)
        z = np.clip(k_ / (atm * np.sqrt(np.maximum(tau_, 1e-8))), -4.0, 4.0)
        return np.asarray(
            np.maximum(atm * (1.0 + self.skew * z + self.curvature * z * z), 0.03),
            dtype=np.float64,
        )

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "kind": "proxy_smile",
            "atm_30d": self.atm_30d,
            "long_run": self.long_run,
            "skew": self.skew,
            "curvature": self.curvature,
        }


@dataclass(frozen=True, slots=True)
class HistoryRow:
    day: date
    spot: float
    vix: float
    rate: float


def load_fred_series(path: Path) -> dict[date, float]:
    """A FRED CSV of date and value. Missing observations ('.' or empty) are left out."""
    out: dict[date, float] = {}
    with path.open() as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 2 or row[1] in ("", "."):
                continue
            out[date.fromisoformat(row[0])] = float(row[1])
    if not out:
        raise MarketDataError(f"{path}: no observations")
    return out


def join_history(
    spot: dict[date, float], vix: dict[date, float], rates_pct: dict[date, float]
) -> tuple[HistoryRow, ...]:
    """Days with spot and VIX. Rates carry forward from the last known value: never backward."""
    rate_days = sorted(rates_pct)
    rows: list[HistoryRow] = []
    last_rate: float | None = None
    cursor = 0
    for day in sorted(set(spot) & set(vix)):
        while cursor < len(rate_days) and rate_days[cursor] <= day:
            last_rate = rates_pct[rate_days[cursor]] / 100.0
            cursor += 1
        if last_rate is not None:
            rows.append(HistoryRow(day, spot[day], vix[day] / 100.0, last_rate))
    if not rows:
        raise MarketDataError("history join is empty")
    return tuple(rows)


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def monthly_expiries(day: date, count: int) -> list[date]:
    """Next `count` third Fridays strictly after `day`."""
    out: list[date] = []
    year, month = day.year, day.month
    while len(out) < count:
        candidate = third_friday(year, month)
        if candidate > day:
            out.append(candidate)
        month = month % 12 + 1
        year += 1 if month == 1 else 0
    return out


@dataclass(frozen=True, slots=True)
class SpreadModel:
    """Half-spread in index points: base plus a share of premium, scaled with the VIX level.

    Assumption, not a measurement. Its level is set near the SPX half-spreads in the 2026-10-30
    chain (0.2 to 1.0 points at a VIX near 15), and the scaling with vol is assumed. Costs scale
    with `multiple`, so results can be tested for robustness to it.
    """

    base: float = 0.15
    premium_share: float = 0.002
    reference_vix: float = 0.15
    multiple: float = 1.0
    linear_bps: float = 0.5
    min_half_spread: float = 0.05

    def option_half(self, premium: float, vix: float) -> float:
        half = (self.base + self.premium_share * premium) * max(vix / self.reference_vix, 1.0)
        return max(half, self.min_half_spread) * self.multiple

    def linear_half(self, spot: float) -> float:
        return spot * self.linear_bps * 1e-4 * self.multiple


class ProxyMarket:
    """Iterates over `DayMarket`s. Holds no future data: each day is built only from rows so far."""

    def __init__(
        self,
        history: Sequence[HistoryRow],
        *,
        spreads: SpreadModel | None = None,
        atm_ratio: float = 0.90,
        long_run_days: int = 252,
        strike_grid_stdevs: Sequence[float] = (-2.0, -1.25, -0.5, 0.5, 1.25, 2.0),
        n_expiries: int = 2,
    ) -> None:
        if not history:
            raise ValidationError("history is empty")
        self.history = tuple(history)
        self.spreads = spreads or SpreadModel()
        self.atm_ratio = atm_ratio
        self.long_run_days = long_run_days
        self.grid = tuple(strike_grid_stdevs)
        self.n_expiries = n_expiries
        self.linear = LinearInstrument(SYMBOL, SYMBOL)

    def __iter__(self) -> Iterator[DayMarket]:
        for index in range(len(self.history)):
            yield self.day(index)

    def day(self, index: int) -> DayMarket:
        row = self.history[index]
        as_of = datetime.combine(row.day, time(16, 0), tzinfo=NEW_YORK)
        window = self.history[max(0, index - self.long_run_days + 1) : index + 1]
        long_run = float(np.mean([r.vix for r in window])) * self.atm_ratio
        smile = ProxySmile(row.vix * self.atm_ratio, long_run)
        market = UnderlyingMarket(SYMBOL, row.spot, DiscountCurve.flat(row.rate), smile)
        snapshot = MarketSnapshot(as_of, {SYMBOL: market})

        def quote_for(instrument: Instrument) -> HedgeCandidate:
            return self._candidate(instrument, snapshot, row)

        candidates: list[HedgeCandidate] = [quote_for(self.linear)]
        for expiry_day in monthly_expiries(row.day, self.n_expiries):
            expiry = us_expiry(expiry_day)
            tau = calendar_years(as_of, expiry)
            stdev = float(smile.atm(np.array(tau))) * np.sqrt(tau)
            forward = market.forward(as_of, expiry)
            for z in self.grid:
                strike = _round_strike(forward * np.exp(z * stdev))
                right = OptionRight.PUT if z < 0.0 else OptionRight.CALL
                candidates.append(quote_for(_spx_option(right, strike, expiry)))
        return DayMarket(as_of, snapshot, tuple(candidates), quote_for)

    def _candidate(
        self, instrument: Instrument, snapshot: MarketSnapshot, row: HistoryRow
    ) -> HedgeCandidate:
        as_of = snapshot.as_of
        if isinstance(instrument, LinearInstrument):
            half = self.spreads.linear_half(row.spot)
            quote = Quote(row.spot - half, row.spot + half, 1_000_000, 1_000_000, as_of)
            return HedgeCandidate(instrument, quote)
        assert isinstance(instrument, OptionContract)
        market = snapshot.market(instrument.underlying)
        if instrument.expiry <= as_of:
            intrinsic = max(instrument.right.omega * (row.spot - instrument.strike), 0.0)
            half = self.spreads.min_half_spread
            quote = Quote(max(intrinsic - half, 0.0), intrinsic + half, 500, 500, as_of)
            return HedgeCandidate(instrument, quote)
        premium = _model_premium(instrument, market, as_of)
        half = self.spreads.option_half(premium, row.vix)
        quote = Quote(max(premium - half, 0.0), premium + half, 500, 500, as_of)
        return HedgeCandidate(instrument, quote)


def _round_strike(strike: float, step: float = 5.0) -> float:
    return float(max(round(strike / step) * step, step))


def _spx_option(right: OptionRight, strike: float, expiry: datetime) -> OptionContract:
    return OptionContract(
        SYMBOL, right, strike, expiry, ExerciseStyle.EUROPEAN, settlement=SettlementKind.CASH
    )


def _model_premium(contract: OptionContract, market: UnderlyingMarket, as_of: datetime) -> float:
    T = calendar_years(as_of, contract.expiry)
    forward = market.forward(as_of, contract.expiry)
    k = np.array([np.log(contract.strike / forward)])
    tau = np.array([T])
    vol = market.surface.vol(k, tau)
    return float(
        black_premium(
            np.array([forward]),
            np.array([contract.strike]),
            market.curve.discount(tau),
            vol * np.sqrt(tau),
            np.array([float(contract.right.omega)]),
        )[0]
    )


@dataclass(slots=True)
class ShortStrangleBook:
    """Short strangle at about +/-1 stdev in the second listed monthly, rolled at `roll_days` left.

    A common vol-selling book, and a hard one to hedge since it is short gamma and short vega into
    exactly the moves that matter.
    """

    lots: int = 10
    roll_days: int = 10
    stdevs: float = 1.0

    def trades(self, day: DayMarket, holdings):  # type: ignore[no-untyped-def]
        from greeks_neutralizer.domain import Position

        live = [
            p
            for p in holdings.values()
            if isinstance(p.instrument, OptionContract) and p.instrument.expiry > day.as_of
        ]
        if live and all(
            calendar_years(day.as_of, p.instrument.expiry) * 365.0 > self.roll_days for p in live
        ):
            return ()
        trades = [Position(p.instrument, -p.quantity) for p in live]
        market = day.snapshot.market(SYMBOL)
        expiries = monthly_expiries(day.as_of.date(), 2)
        target = next(
            (e for e in expiries if (e - day.as_of.date()).days > self.roll_days + 5), expiries[-1]
        )
        expiry = us_expiry(target)
        tau = calendar_years(day.as_of, expiry)
        surface = market.surface
        atm = float(surface.vol(np.array([0.0]), np.array([tau]))[0])
        forward = market.forward(day.as_of, expiry)
        stdev = atm * np.sqrt(tau)
        put = _spx_option(
            OptionRight.PUT, _round_strike(forward * np.exp(-self.stdevs * stdev)), expiry
        )
        call = _spx_option(
            OptionRight.CALL, _round_strike(forward * np.exp(self.stdevs * stdev)), expiry
        )
        trades += [Position(put, -self.lots), Position(call, -self.lots)]
        return trades


@dataclass(slots=True)
class LongPutOverlayBook:
    """Long equity index exposure with a monthly protective put: a typical buy-side overlay."""

    notional_shares: int = 10_000
    put_stdevs: float = 1.5
    roll_days: int = 10

    def trades(self, day: DayMarket, holdings):  # type: ignore[no-untyped-def]
        from greeks_neutralizer.domain import Position

        trades: list[Position] = []
        linear = LinearInstrument(SYMBOL, SYMBOL)
        if linear.key not in holdings:
            trades.append(Position(linear, self.notional_shares))
        live = [
            p
            for p in holdings.values()
            if isinstance(p.instrument, OptionContract) and p.instrument.expiry > day.as_of
        ]
        if live and all(
            calendar_years(day.as_of, p.instrument.expiry) * 365.0 > self.roll_days for p in live
        ):
            return trades
        trades += [Position(p.instrument, -p.quantity) for p in live]
        market = day.snapshot.market(SYMBOL)
        expiries = monthly_expiries(day.as_of.date(), 2)
        target = next(
            (e for e in expiries if (e - day.as_of.date()).days > self.roll_days + 5), expiries[-1]
        )
        expiry = us_expiry(target)
        tau = calendar_years(day.as_of, expiry)
        atm = float(market.surface.vol(np.array([0.0]), np.array([tau]))[0])
        forward = market.forward(day.as_of, expiry)
        strike = _round_strike(forward * np.exp(-self.put_stdevs * atm * np.sqrt(tau)))
        lots = max(self.notional_shares // 100, 1)
        trades.append(Position(_spx_option(OptionRight.PUT, strike, expiry), lots))
        return trades
