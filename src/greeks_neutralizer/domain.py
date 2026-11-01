"""Immutable domain model. Constructors enforce invariants, so an invalid object cannot exist.

Sign convention: `Position.quantity` is a signed integer count of contracts or shares; positive is
long. There is deliberately no separate "side" field, so sign cannot disagree with side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias

from greeks_neutralizer.errors import ValidationError


class OptionRight(StrEnum):
    CALL = "call"
    PUT = "put"

    @property
    def omega(self) -> int:
        """+1 for a call, -1 for a put."""
        return 1 if self is OptionRight.CALL else -1


class ExerciseStyle(StrEnum):
    EUROPEAN = "european"
    AMERICAN = "american"


class SettlementKind(StrEnum):
    """Physical delivery (single names, ETFs) or cash (indices)."""

    PHYSICAL = "physical"
    CASH = "cash"


class LinearKind(StrEnum):
    SPOT = "spot"
    FUTURE = "future"
    PROXY = "proxy"


def require_finite_positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value <= 0.0:
        raise ValidationError(f"{name} must be finite and positive, got {value!r}")


def require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value!r}")


def require_aware(name: str, value: datetime) -> None:
    if not isinstance(value, datetime):
        raise ValidationError(f"{name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{name} must be timezone-aware, got naive {value!r}")


@dataclass(frozen=True, slots=True)
class OptionContract:
    """A listed or OTC vanilla option.

    `expiry` is the instant the option's value is fixed: for example 09:30 ET on the settlement date
    for AM-settled SPX, and 16:00 ET for PM-settled contracts. See `conventions.us_expiry`.
    """

    underlying: str
    right: OptionRight
    strike: float
    expiry: datetime
    style: ExerciseStyle = ExerciseStyle.AMERICAN
    multiplier: float = 100.0
    settlement: SettlementKind = SettlementKind.PHYSICAL
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.underlying:
            raise ValidationError("underlying must be non-empty")
        if not isinstance(self.right, OptionRight):
            raise ValidationError(f"right must be an OptionRight, got {self.right!r}")
        if not isinstance(self.style, ExerciseStyle):
            raise ValidationError(f"style must be an ExerciseStyle, got {self.style!r}")
        if not isinstance(self.settlement, SettlementKind):
            raise ValidationError(f"settlement must be a SettlementKind, got {self.settlement!r}")
        require_finite_positive("strike", self.strike)
        require_finite_positive("multiplier", self.multiplier)
        require_aware("expiry", self.expiry)

    @property
    def key(self) -> str:
        return (
            f"{self.underlying}|{self.expiry.isoformat()}|{self.right.value}|"
            f"{self.strike:.6f}|{self.style.value}|{self.multiplier:g}"
        )

    def describe(self) -> str:
        return (
            f"{self.underlying} {self.expiry.date().isoformat()} "
            f"{self.strike:g}{self.right.value[0].upper()}"
        )


@dataclass(frozen=True, slots=True)
class LinearInstrument:
    """A delta-one hedge: the underlying itself, a future on it, or a proxy such as an ETF.

    `delta_ratio` is the units of underlying exposure per unit of price-1 multiplier, used for
    proxies (one SPY share is about 0.1 SPX points) and `basis_vol` is the annualized volatility of
    the untradable basis, which is carried into proposals as a quantified residual risk.
    A future's `expiry` makes its spot delta the carry factor F/S.
    """

    symbol: str
    underlying: str
    kind: LinearKind = LinearKind.SPOT
    multiplier: float = 1.0
    delta_ratio: float = 1.0
    basis_vol: float = 0.0
    expiry: datetime | None = None
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.symbol or not self.underlying:
            raise ValidationError("symbol and underlying must be non-empty")
        if not isinstance(self.kind, LinearKind):
            raise ValidationError(f"kind must be a LinearKind, got {self.kind!r}")
        require_finite_positive("multiplier", self.multiplier)
        require_finite_positive("delta_ratio", self.delta_ratio)
        require_finite("basis_vol", self.basis_vol)
        if self.basis_vol < 0.0:
            raise ValidationError("basis_vol must be non-negative")
        if self.kind is LinearKind.FUTURE:
            if self.expiry is None:
                raise ValidationError("a future requires an expiry")
            require_aware("expiry", self.expiry)
        if self.kind is LinearKind.PROXY and self.basis_vol == 0.0:
            raise ValidationError("a proxy hedge must state a positive basis_vol")

    @property
    def key(self) -> str:
        return f"{self.symbol}|{self.kind.value}|{self.multiplier:g}"

    def describe(self) -> str:
        return f"{self.symbol} ({self.kind.value})"


Instrument: TypeAlias = OptionContract | LinearInstrument


@dataclass(frozen=True, slots=True)
class Position:
    """A signed holding. Zero is allowed so a flat line stays visible in a book."""

    instrument: Instrument
    quantity: int

    def __post_init__(self) -> None:
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise ValidationError(
                f"quantity must be a signed int, got {type(self.quantity).__name__}"
            )
        if not isinstance(self.instrument, (OptionContract, LinearInstrument)):
            raise ValidationError(f"unknown instrument type {type(self.instrument).__name__}")


@dataclass(frozen=True, slots=True)
class Quote:
    """A two-sided market with sizes and the time it was observed. Size is in contracts."""

    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: datetime

    def __post_init__(self) -> None:
        require_finite("bid", self.bid)
        require_finite("ask", self.ask)
        if self.bid < 0.0 or self.ask < 0.0:
            raise ValidationError(f"quote prices must be non-negative, got {self.bid}/{self.ask}")
        if self.bid_size < 0 or self.ask_size < 0:
            raise ValidationError("quote sizes must be non-negative")
        require_aware("timestamp", self.timestamp)

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def half_spread(self) -> float:
        return 0.5 * (self.ask - self.bid)
