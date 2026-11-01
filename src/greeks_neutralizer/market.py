"""Market state: curves, dividends, surfaces, and the immutable snapshot everything prices off.

A snapshot is the only way market data reaches the pricing code, which is what makes any result
replayable: `MarketSnapshot.fingerprint()` hashes the complete state for the run manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import pairwise
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from greeks_neutralizer.domain import require_aware, require_finite, require_finite_positive
from greeks_neutralizer.errors import MarketDataError, ValidationError
from greeks_neutralizer.pricing.black import FloatArray
from greeks_neutralizer.time_model import CALENDAR_CLOCK, VarianceClock, calendar_years

_MAX_PLAUSIBLE_RATE = 1.0


@dataclass(frozen=True, slots=True)
class DiscountCurve:
    """Continuously-compounded zero curve, linear in zero rate * time (log-linear discount)."""

    times: tuple[float, ...]
    zero_rates: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.times or len(self.times) != len(self.zero_rates):
            raise ValidationError("curve needs equal, non-empty times and zero_rates")
        if any(b <= a for a, b in pairwise(self.times)):
            raise ValidationError("curve times must be strictly increasing")
        if self.times[0] <= 0.0:
            raise ValidationError("curve times must be positive")
        for rate in self.zero_rates:
            require_finite("zero rate", rate)
            if abs(rate) > _MAX_PLAUSIBLE_RATE:
                raise ValidationError(
                    f"zero rate {rate} is implausible: rates are decimals, so 5% is 0.05"
                )

    @staticmethod
    def flat(rate: float) -> DiscountCurve:
        return DiscountCurve((1.0,), (rate,))

    def discount(self, t: ArrayLike) -> FloatArray:
        arr = np.asarray(t, dtype=np.float64)
        return np.asarray(np.exp(-self.zero_rate(arr) * arr), dtype=np.float64)

    def zero_rate(self, t: ArrayLike) -> FloatArray:
        """Zero rate: flat beyond the curve's ends, with log-discount interpolation inside."""
        arr = np.asarray(t, dtype=np.float64)
        times = np.asarray(self.times)
        rates = np.asarray(self.zero_rates)
        if times.size == 1:
            return np.full_like(arr, rates[0], dtype=np.float64)
        log_df = np.interp(arr, times, rates * times)
        short = arr < times[0]
        long = arr > times[-1]
        fallback = np.full_like(arr, rates[0], dtype=np.float64)
        out = np.divide(log_df, arr, out=fallback, where=arr > 0)
        return np.asarray(
            np.where(short, rates[0], np.where(long, rates[-1], out)), dtype=np.float64
        )

    def bumped(self, shift: float) -> DiscountCurve:
        return DiscountCurve(self.times, tuple(r + shift for r in self.zero_rates))


@dataclass(frozen=True, slots=True)
class CashDividend:
    ex_date: datetime
    amount: float

    def __post_init__(self) -> None:
        require_aware("ex_date", self.ex_date)
        require_finite_positive("dividend amount", self.amount)


@runtime_checkable
class VolSurface(Protocol):
    """Implied vol as a function of log-moneyness k = ln(K / F) and variance-time tau."""

    def vol(self, k: FloatArray, tau: FloatArray) -> FloatArray: ...

    def fingerprint_payload(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class FlatVol:
    """Constant vol, for tests and for marks where no surface is available."""

    sigma: float

    def __post_init__(self) -> None:
        require_finite("sigma", self.sigma)
        if self.sigma < 0.0:
            raise ValidationError("sigma must be non-negative")

    def vol(self, k: FloatArray, tau: FloatArray) -> FloatArray:
        return np.full(np.broadcast(k, tau).shape, self.sigma, dtype=np.float64)

    def fingerprint_payload(self) -> dict[str, object]:
        return {"kind": "flat", "sigma": self.sigma}


@dataclass(frozen=True, slots=True)
class UnderlyingMarket:
    """Everything needed to value options on one underlying.

    `carry_yield` is the continuous dividend yield plus stock borrow fee. Discrete dividends are
    escrowed: volatility applies to spot minus the PV of dividends paid before expiry.
    """

    symbol: str
    spot: float
    curve: DiscountCurve
    surface: VolSurface
    carry_yield: float = 0.0
    dividends: tuple[CashDividend, ...] = ()
    clock: VarianceClock = CALENDAR_CLOCK

    def __post_init__(self) -> None:
        require_finite_positive("spot", self.spot)
        require_finite("carry_yield", self.carry_yield)
        if not isinstance(self.surface, VolSurface):
            raise ValidationError("surface must implement VolSurface")

    def pv_dividends(self, as_of: datetime, until: datetime) -> float:
        """PV at as_of of cash dividends with as_of < ex_date <= until."""
        pv = 0.0
        for div in self.dividends:
            if as_of < div.ex_date <= until:
                pv += div.amount * float(self.curve.discount(calendar_years(as_of, div.ex_date)))
        return pv

    def pv_dividends_time_weighted(self, as_of: datetime, until: datetime) -> float:
        """Sum of t_i * PV_i, which is minus the derivative of PV(dividends) in the rate."""
        total = 0.0
        for div in self.dividends:
            if as_of < div.ex_date <= until:
                t = calendar_years(as_of, div.ex_date)
                total += t * div.amount * float(self.curve.discount(t))
        return total

    def forward(self, as_of: datetime, expiry: datetime) -> float:
        T = calendar_years(as_of, expiry)
        escrowed = self.spot - self.pv_dividends(as_of, expiry)
        if escrowed <= 0.0:
            raise MarketDataError(f"{self.symbol}: dividends PV exceeds spot before {expiry}")
        df = float(self.curve.discount(T))
        return float(escrowed * np.exp(-self.carry_yield * T) / df)


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    as_of: datetime
    underlyings: dict[str, UnderlyingMarket] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_aware("as_of", self.as_of)

    def market(self, symbol: str) -> UnderlyingMarket:
        try:
            return self.underlyings[symbol]
        except KeyError:
            raise MarketDataError(f"no market data for underlying {symbol!r}") from None

    def replaced(self, symbol: str, **changes: object) -> MarketSnapshot:
        updated = dict(self.underlyings)
        updated[symbol] = replace(self.market(symbol), **changes)  # type: ignore[arg-type]
        return MarketSnapshot(self.as_of, updated)

    def fingerprint(self) -> str:
        """Stable SHA-256 of the full market state, recorded in every run manifest."""
        payload = {
            "as_of": self.as_of.isoformat(),
            "underlyings": {
                sym: {
                    "spot": m.spot,
                    "times": m.curve.times,
                    "rates": m.curve.zero_rates,
                    "carry": m.carry_yield,
                    "dividends": [(d.ex_date.isoformat(), d.amount) for d in m.dividends],
                    "surface": m.surface.fingerprint_payload(),
                    "clock": [
                        m.clock.session_weight,
                        m.clock.overnight_weight,
                        m.clock.closed_day_weight,
                        [(e[0].isoformat(), e[1]) for e in m.clock.events],
                        getattr(m.clock.is_session_day, "__qualname__", "custom"),
                    ],
                }
                for sym, m in sorted(self.underlyings.items())
            },
        }
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()
