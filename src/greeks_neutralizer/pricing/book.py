"""Book valuation: contracts to per-share Greeks, positions to scaled exposures, and aggregates.

Fail-closed rules:
  - A position whose underlying has no market data raises MarketDataError.
  - An expired option keeps its intrinsic value at spot and is reported in `issues`.
  - A vol surface returning NaN or a negative value raises MarketDataError.

Reporting units (`Exposure`), all for the whole position:
  delta_shares    underlying shares equivalent
  delta_dollars   delta_shares * spot
  gamma_shares    change in delta_shares per $1 of spot
  gamma_1pct      change in delta_dollars for a +1% spot move
  vega_1pt        P&L per +1 vol point (0.01)
  theta_1d        P&L per calendar day
  rho_1pct        P&L per +1% rate
  vanna_1pt       change in delta_shares per +1 vol point
  volga_1pt       change in vega_1pt per +1 vol point
  charm_1d        change in delta_shares per calendar day
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from greeks_neutralizer.domain import (
    ExerciseStyle,
    Instrument,
    LinearInstrument,
    LinearKind,
    OptionContract,
    Position,
)
from greeks_neutralizer.errors import MarketDataError
from greeks_neutralizer.market import MarketSnapshot, UnderlyingMarket
from greeks_neutralizer.pricing.american import (
    AmericanBatch,
    american_greeks,
    exercise_is_ever_optimal,
)
from greeks_neutralizer.pricing.black import GREEK_NAMES, FloatArray, GreekArrays, black_greeks
from greeks_neutralizer.time_model import calendar_years

EXPOSURE_NAMES: tuple[str, ...] = (
    "value",
    "delta_shares",
    "delta_dollars",
    "gamma_shares",
    "gamma_1pct",
    "vega_1pt",
    "theta_1d",
    "rho_1pct",
    "vanna_1pt",
    "volga_1pt",
    "charm_1d",
)


class SmileRegime(StrEnum):
    """How the smile responds to spot, which changes the delta you should actually hedge.

    STICKY_STRIKE: vol at a fixed strike does not move (Black delta).
    STICKY_MONEYNESS: the smile moves with the forward (also called sticky delta).
    """

    STICKY_STRIKE = "sticky_strike"
    STICKY_MONEYNESS = "sticky_moneyness"


@dataclass(frozen=True, slots=True)
class OptionValuation:
    """Per-share results aligned with the input contracts."""

    greeks: GreekArrays
    vol: FloatArray
    forward: FloatArray
    T: FloatArray
    expired: NDArray[np.bool_]
    american_tree: NDArray[np.bool_]
    exercise_premium: FloatArray
    smile_slope: FloatArray  # d sigma / d k at the contract's strike


@dataclass(frozen=True, slots=True)
class Exposure:
    """Position-scaled exposures. See the module docstring for units."""

    value: float = 0.0
    delta_shares: float = 0.0
    delta_dollars: float = 0.0
    gamma_shares: float = 0.0
    gamma_1pct: float = 0.0
    vega_1pt: float = 0.0
    theta_1d: float = 0.0
    rho_1pct: float = 0.0
    vanna_1pt: float = 0.0
    volga_1pt: float = 0.0
    charm_1d: float = 0.0

    def __add__(self, other: Exposure) -> Exposure:
        return Exposure(*(getattr(self, n) + getattr(other, n) for n in EXPOSURE_NAMES))

    def as_dict(self) -> dict[str, float]:
        return {n: float(getattr(self, n)) for n in EXPOSURE_NAMES}


@dataclass(frozen=True, slots=True)
class PositionValuation:
    position: Position
    unit_price: float
    vol: float | None
    exposure: Exposure


@dataclass(frozen=True, slots=True)
class BookValuation:
    as_of: datetime
    market_fingerprint: str
    lines: tuple[PositionValuation, ...]
    by_underlying: dict[str, Exposure]
    total_value: float
    total_delta_dollars: float
    total_gamma_1pct: float
    total_vega_1pt: float
    total_theta_1d: float
    issues: tuple[str, ...] = field(default_factory=tuple)


def _group_by_underlying(contracts: Sequence[OptionContract]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, contract in enumerate(contracts):
        groups.setdefault(contract.underlying, []).append(idx)
    return groups


def _smile(
    market: UnderlyingMarket, k: FloatArray, tau: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Vol and d(vol)/dk from the surface, rejecting unusable values. Slope by central difference
    on log-moneyness, which is the only thing the protocol guarantees."""
    vol = np.asarray(market.surface.vol(k, tau), dtype=np.float64)
    if not np.all(np.isfinite(vol)) or np.any(vol < 0.0):
        raise MarketDataError(f"{market.symbol}: vol surface returned NaN or a negative vol")
    h = 1e-4
    up = market.surface.vol(k + h, tau)
    dn = market.surface.vol(k - h, tau)
    return vol, np.asarray((up - dn) / (2.0 * h), dtype=np.float64)


@dataclass(slots=True)
class _Buffers:
    """Output columns for `value_options`, filled one underlying at a time."""

    greeks: dict[str, FloatArray]
    vol: FloatArray
    forward: FloatArray
    T: FloatArray
    expired: NDArray[np.bool_]
    tree: NDArray[np.bool_]
    premium: FloatArray
    slope: FloatArray

    @staticmethod
    def zeros(n: int) -> _Buffers:
        def col() -> FloatArray:
            return np.zeros(n, dtype=np.float64)

        return _Buffers(
            greeks={name: col() for name in GREEK_NAMES},
            vol=col(),
            forward=col(),
            T=col(),
            expired=np.zeros(n, dtype=bool),
            tree=np.zeros(n, dtype=bool),
            premium=col(),
            slope=col(),
        )


def value_options(
    contracts: Sequence[OptionContract], snapshot: MarketSnapshot, *, tree_steps: int = 101
) -> OptionValuation:
    """Value and risk per share for each contract, dispatching on exercise style."""
    out = _Buffers.zeros(len(contracts))
    for symbol, indices in _group_by_underlying(contracts).items():
        market = snapshot.market(symbol)
        _value_group(contracts, indices, market, snapshot.as_of, out, tree_steps)
    return OptionValuation(
        greeks=GreekArrays(**out.greeks),
        vol=out.vol,
        forward=out.forward,
        T=out.T,
        expired=out.expired,
        american_tree=out.tree,
        exercise_premium=out.premium,
        smile_slope=out.slope,
    )


def _value_group(
    contracts: Sequence[OptionContract],
    indices: list[int],
    market: UnderlyingMarket,
    as_of: datetime,
    out: _Buffers,
    tree_steps: int,
) -> None:
    idx = np.asarray(indices, dtype=np.intp)
    group = [contracts[i] for i in indices]
    T = np.array([calendar_years(as_of, c.expiry) for c in group], dtype=np.float64)
    tau = np.array([market.clock.tau(as_of, c.expiry) for c in group], dtype=np.float64)
    strikes = np.array([c.strike for c in group], dtype=np.float64)
    omega = np.array([c.right.omega for c in group], dtype=np.float64)
    forwards = np.array([market.forward(as_of, c.expiry) for c in group], dtype=np.float64)
    live = T > 0.0
    r = np.zeros_like(T)
    if np.any(live):
        r[live] = -np.log(market.curve.discount(T[live])) / T[live]

    k = np.log(strikes / forwards)
    vol = np.zeros_like(T)
    slope = np.zeros_like(T)
    if np.any(live):
        vol[live], slope[live] = _smile(market, k[live], tau[live])

    # The escrowed spot and carry yield reproduce each contract's forward in the closed form.
    pv_div = np.array([market.pv_dividends(as_of, c.expiry) for c in group], dtype=np.float64)
    q = np.full_like(T, market.carry_yield)
    greeks = black_greeks(market.spot - pv_div, strikes, T, r, q, vol, omega, tau=tau)
    premium = np.zeros_like(T)

    is_american = np.array([c.style is ExerciseStyle.AMERICAN for c in group], dtype=bool)
    needs_tree = (
        is_american & live & (vol > 0.0) & exercise_is_ever_optimal(omega, r, q, pv_div > 0.0)
    )
    if np.any(needs_tree):
        sel = np.flatnonzero(needs_tree)
        batch = _american_batch(market, as_of, group, sel, T, tau, r, q, vol, strikes, omega)
        am = american_greeks(batch, steps=tree_steps)
        for name in ("price", "delta", "gamma", "vega", "theta", "rho", "phi", "vanna", "volga"):
            getattr(greeks, name)[sel] = getattr(am, name)
        premium[sel] = am.exercise_premium

    # Expired contracts report intrinsic at spot, not at the escrowed forward, with zero risk.
    if np.any(~live):
        dead = np.flatnonzero(~live)
        for name in GREEK_NAMES:
            getattr(greeks, name)[dead] = 0.0
        greeks.price[dead] = np.maximum(omega[dead] * (market.spot - strikes[dead]), 0.0)

    for name in GREEK_NAMES:
        out.greeks[name][idx] = getattr(greeks, name)
    out.vol[idx] = vol
    out.forward[idx] = forwards
    out.T[idx] = T
    out.expired[idx] = ~live
    out.tree[idx] = needs_tree
    out.premium[idx] = premium
    out.slope[idx] = slope


def _american_batch(
    market: UnderlyingMarket,
    as_of: datetime,
    group: list[OptionContract],
    sel: NDArray[np.intp],
    T: FloatArray,
    tau: FloatArray,
    r: FloatArray,
    q: FloatArray,
    vol: FloatArray,
    strikes: FloatArray,
    omega: FloatArray,
) -> AmericanBatch:
    divs = [d for d in market.dividends if d.ex_date > as_of]
    m = len(divs)
    div_times = np.full((sel.size, m), np.inf, dtype=np.float64)
    div_amounts = np.zeros((sel.size, m), dtype=np.float64)
    for row, i in enumerate(sel):
        for col, d in enumerate(divs):
            if d.ex_date <= group[int(i)].expiry:
                div_times[row, col] = calendar_years(as_of, d.ex_date)
                div_amounts[row, col] = d.amount
    return AmericanBatch(
        spot=np.full(sel.size, market.spot, dtype=np.float64),
        strike=strikes[sel],
        T=T[sel],
        tau=tau[sel],
        r=r[sel],
        q=q[sel],
        sigma=vol[sel],
        omega=omega[sel],
        div_times=div_times,
        div_amounts=div_amounts,
    )


def linear_unit_exposure(
    instrument: LinearInstrument, market: UnderlyingMarket, as_of: datetime
) -> tuple[float, float]:
    """(unit price, spot delta per unit before multiplier) for a delta-one instrument."""
    if instrument.kind is LinearKind.FUTURE:
        assert instrument.expiry is not None
        forward = market.forward(as_of, instrument.expiry)
        return forward, forward / market.spot
    return market.spot * instrument.delta_ratio, instrument.delta_ratio


def value_book(positions: Sequence[Position], snapshot: MarketSnapshot) -> BookValuation:
    """Value and risk for a mixed book of options and linear instruments."""
    options = [p for p in positions if isinstance(p.instrument, OptionContract)]
    option_contracts: list[OptionContract] = [
        p.instrument for p in options if isinstance(p.instrument, OptionContract)
    ]
    valuation = value_options(option_contracts, snapshot) if option_contracts else None

    lines: list[PositionValuation] = []
    by_underlying: dict[str, Exposure] = {}
    issues: list[str] = []
    option_row = 0
    for position in positions:
        instrument = position.instrument
        if isinstance(instrument, OptionContract):
            assert valuation is not None
            line = _option_line(position, instrument, option_row, valuation, snapshot)
            if valuation.expired[option_row]:
                issues.append(
                    f"EXPIRED {instrument.describe()} qty {position.quantity}: "
                    "carried at intrinsic with zero risk; settle or remove it"
                )
            underlying = instrument.underlying
            option_row += 1
        else:
            underlying = instrument.underlying
            line = _linear_line(position, instrument, snapshot)
        lines.append(line)
        by_underlying[underlying] = by_underlying.get(underlying, Exposure()) + line.exposure

    totals = {
        "value": 0.0,
        "delta_dollars": 0.0,
        "gamma_1pct": 0.0,
        "vega_1pt": 0.0,
        "theta_1d": 0.0,
    }
    for exposure in by_underlying.values():
        for key in totals:
            totals[key] += getattr(exposure, key)
    return BookValuation(
        as_of=snapshot.as_of,
        market_fingerprint=snapshot.fingerprint(),
        lines=tuple(lines),
        by_underlying=by_underlying,
        total_value=totals["value"],
        total_delta_dollars=totals["delta_dollars"],
        total_gamma_1pct=totals["gamma_1pct"],
        total_vega_1pt=totals["vega_1pt"],
        total_theta_1d=totals["theta_1d"],
        issues=tuple(issues),
    )


def scale_option_exposure(
    greeks: GreekArrays, row: int, quantity: float, multiplier: float, spot: float
) -> Exposure:
    """Scale per-share Greeks to a position. The single definition of every reported unit."""
    size = quantity * multiplier
    delta_shares = size * float(greeks.delta[row])
    gamma_shares = size * float(greeks.gamma[row])
    return Exposure(
        value=size * float(greeks.price[row]),
        delta_shares=delta_shares,
        delta_dollars=delta_shares * spot,
        gamma_shares=gamma_shares,
        gamma_1pct=gamma_shares * spot * spot * 0.01,
        vega_1pt=size * float(greeks.vega[row]) * 0.01,
        theta_1d=size * float(greeks.theta[row]) / 365.0,
        rho_1pct=size * float(greeks.rho[row]) * 0.01,
        vanna_1pt=size * float(greeks.vanna[row]) * 0.01,
        volga_1pt=size * float(greeks.volga[row]) * 1e-4,
        charm_1d=size * float(greeks.charm[row]) / 365.0,
    )


def _option_line(
    position: Position,
    contract: OptionContract,
    row: int,
    valuation: OptionValuation,
    snapshot: MarketSnapshot,
) -> PositionValuation:
    spot = snapshot.market(contract.underlying).spot
    exposure = scale_option_exposure(
        valuation.greeks, row, position.quantity, contract.multiplier, spot
    )
    return PositionValuation(
        position=position,
        unit_price=float(valuation.greeks.price[row]),
        vol=float(valuation.vol[row]) if not valuation.expired[row] else None,
        exposure=exposure,
    )


def _linear_line(
    position: Position, instrument: LinearInstrument, snapshot: MarketSnapshot
) -> PositionValuation:
    market = snapshot.market(instrument.underlying)
    price, unit_delta = linear_unit_exposure(instrument, market, snapshot.as_of)
    size = position.quantity * instrument.multiplier
    delta_shares = size * unit_delta
    # A future has zero value at inception; report the notional-free mark as zero value.
    value = 0.0 if instrument.kind is LinearKind.FUTURE else size * price
    return PositionValuation(
        position=position,
        unit_price=price,
        vol=None,
        exposure=Exposure(
            value=value,
            delta_shares=delta_shares,
            delta_dollars=delta_shares * market.spot,
        ),
    )


def unit_exposures(
    instruments: Sequence[Instrument], snapshot: MarketSnapshot
) -> tuple[Exposure, ...]:
    """Exposure of holding one unit of each instrument, for building hedge matrices."""
    book = value_book([Position(i, 1) for i in instruments], snapshot)
    return tuple(line.exposure for line in book.lines)
