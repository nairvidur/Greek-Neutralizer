"""VaR and ES with backtesting, P&L attribution, ladders, pin and assignment, and limits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from scipy.stats import chi2

from greeks_neutralizer.domain import (
    NEW_YORK,
    ExerciseStyle,
    OptionContract,
    OptionRight,
    Position,
    SettlementKind,
    require_finite,
    require_symbol,
)
from greeks_neutralizer.errors import ValidationError
from greeks_neutralizer.market import MarketSnapshot
from greeks_neutralizer.pricing.black import FloatArray
from greeks_neutralizer.pricing.book import Exposure, mark_value, value_book, value_options
from greeks_neutralizer.time_model import calendar_years


def historical_var(pnl: FloatArray, alpha: float = 0.99) -> float:
    """Loss not exceeded with probability alpha, as a positive number. Fails on too few samples."""
    values = _clean_pnl(pnl, alpha)
    return float(-np.quantile(values, 1.0 - alpha, method="lower"))


def expected_shortfall(pnl: FloatArray, alpha: float = 0.975) -> float:
    """Mean loss in the worst (1 - alpha) tail, as a positive number."""
    values = np.sort(_clean_pnl(pnl, alpha))
    tail = max(int(np.floor(values.size * (1.0 - alpha))), 1)
    return float(-values[:tail].mean())


def _clean_pnl(pnl: FloatArray, alpha: float) -> FloatArray:
    values = np.asarray(pnl, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValidationError("P&L contains NaN or inf")
    if not 0.0 < alpha < 1.0:
        raise ValidationError("alpha must be in (0, 1)")
    if values.size < int(np.ceil(1.0 / (1.0 - alpha))):
        raise ValidationError(f"{values.size} samples cannot support alpha={alpha}")
    return values


@dataclass(frozen=True, slots=True)
class VarBacktest:
    n: int
    exceptions: int
    expected: float
    kupiec_lr: float
    kupiec_p_value: float

    @property
    def rejected_at_5pct(self) -> bool:
        return self.kupiec_p_value < 0.05


def backtest_var(pnl: FloatArray, var: FloatArray, alpha: float = 0.99) -> VarBacktest:
    """Kupiec proportion-of-failures test. `var` is the positive VaR forecast for each day."""
    realized = np.asarray(pnl, dtype=np.float64)
    forecast = np.asarray(var, dtype=np.float64)
    if realized.shape != forecast.shape or realized.size == 0:
        raise ValidationError("pnl and var must be non-empty and the same shape")
    if not (np.all(np.isfinite(realized)) and np.all(np.isfinite(forecast))):
        raise ValidationError("pnl and var must be finite")
    n = int(realized.size)
    x = int(np.sum(realized < -forecast))
    p = 1.0 - alpha
    if x == 0:
        lr = -2.0 * n * np.log(1.0 - p)
    elif x == n:
        lr = -2.0 * n * np.log(p)
    else:
        rate = x / n
        lr = -2.0 * ((n - x) * np.log((1.0 - p) / (1.0 - rate)) + x * np.log(p / rate))
    return VarBacktest(n, x, n * p, float(lr), float(chi2.sf(lr, df=1)))


@dataclass(frozen=True, slots=True)
class PnLAttribution:
    total: float
    delta: float
    gamma: float
    vega: float
    vanna: float
    volga: float
    theta: float
    unexplained: float

    @property
    def unexplained_ratio(self) -> float:
        """Unexplained as a share of gross explained moves. The control metric to track daily."""
        gross = sum(
            abs(x) for x in (self.delta, self.gamma, self.vega, self.vanna, self.volga, self.theta)
        )
        return abs(self.unexplained) / gross if gross > 0.0 else 0.0


def attribute_pnl(
    positions: Sequence[Position], start: MarketSnapshot, end: MarketSnapshot
) -> PnLAttribution:
    """Actual P&L between two snapshots, decomposed by start-of-period Greeks per contract."""
    start_book = value_book(positions, start)
    end_book = value_book(positions, end)
    contracts = [p for p in positions if isinstance(p.instrument, OptionContract)]
    dvol = np.zeros(len(positions))
    option_rows = [i for i, p in enumerate(positions) if isinstance(p.instrument, OptionContract)]
    if contracts:
        insts = [p.instrument for p in contracts if isinstance(p.instrument, OptionContract)]
        v0 = value_options(insts, start)
        v1 = value_options(insts, end)
        # An option that expired has no vol. Its loss of time value is theta, and not a vol move.
        alive = ~(v0.expired | v1.expired)
        dvol[option_rows] = np.where(alive, v1.vol - v0.vol, 0.0)

    days = calendar_years(start.as_of, end.as_of) * 365.0
    terms = dict.fromkeys(("delta", "gamma", "vega", "vanna", "volga", "theta"), 0.0)
    for i, line in enumerate(start_book.lines):
        e = line.exposure
        underlying = line.position.instrument.underlying
        ds = end.market(underlying).spot - start.market(underlying).spot
        pts = dvol[i] * 100.0
        terms["delta"] += e.delta_shares * ds
        terms["gamma"] += 0.5 * e.gamma_shares * ds * ds
        terms["vega"] += e.vega_1pt * pts
        terms["vanna"] += e.vanna_1pt * pts * ds
        terms["volga"] += 0.5 * e.volga_1pt * pts * pts
        terms["theta"] += e.theta_1d * days
    total = sum(
        mark_value(after.position, after.unit_price, after.exposure.value)
        - mark_value(before.position, before.unit_price, before.exposure.value)
        for before, after in zip(start_book.lines, end_book.lines, strict=True)
    )
    explained = sum(terms.values())
    return PnLAttribution(
        total=total,
        unexplained=total - explained,
        **terms,
    )


@dataclass(frozen=True, slots=True)
class LadderBucket:
    expiry: date
    n_lines: int
    exposure: Exposure


def expiry_ladder(
    positions: Sequence[Position], snapshot: MarketSnapshot
) -> tuple[LadderBucket, ...]:
    """Exposures by New York expiry date, shortest first, the same date tickets and descriptions
    use. Linear lines are excluded: they do not expire."""
    book = value_book(positions, snapshot)
    buckets: dict[date, tuple[int, Exposure]] = {}
    for line in book.lines:
        instrument = line.position.instrument
        if not isinstance(instrument, OptionContract):
            continue
        day = instrument.expiry.astimezone(NEW_YORK).date()
        count, total = buckets.get(day, (0, Exposure()))
        buckets[day] = (count + 1, total + line.exposure)
    return tuple(
        LadderBucket(day, count, exposure) for day, (count, exposure) in sorted(buckets.items())
    )


@dataclass(frozen=True, slots=True)
class ExpiryRisk:
    kind: str
    contract: str
    quantity: int
    detail: str


def expiry_risks(
    positions: Sequence[Position],
    snapshot: MarketSnapshot,
    *,
    pin_window_days: float = 2.0,
    pin_band: float = 0.01,
    assignment_window_days: float = 7.0,
) -> tuple[ExpiryRisk, ...]:
    """Flags risks a Greek report does not show: pin risk and likely early assignment.

    Pin risk: a near-expiry physically-settled option close to the strike, where the final share
    position is uncertain. Assignment: a short American ITM call before an ex-date where the
    dividend exceeds the remaining time value, and deep ITM short puts with little time value.
    """
    flags: list[ExpiryRisk] = []
    options = [p for p in positions if isinstance(p.instrument, OptionContract)]
    if not options:
        return ()
    contracts = [p.instrument for p in options if isinstance(p.instrument, OptionContract)]
    valuation = value_options(contracts, snapshot)
    for row, (position, contract) in enumerate(zip(options, contracts, strict=True)):
        market = snapshot.market(contract.underlying)
        days = float(valuation.T[row]) * 365.0
        moneyness = abs(market.spot / contract.strike - 1.0)
        if (
            0.0 < days <= pin_window_days
            and moneyness <= pin_band
            and contract.settlement is SettlementKind.PHYSICAL
        ):
            flags.append(
                ExpiryRisk(
                    "pin",
                    contract.describe(),
                    position.quantity,
                    f"{moneyness:.2%} from strike with {days:.1f} days left; up to "
                    f"{abs(position.quantity) * contract.multiplier:g} shares of settlement risk",
                )
            )
        if position.quantity >= 0 or contract.style is not ExerciseStyle.AMERICAN:
            continue
        intrinsic = max(contract.right.omega * (market.spot - contract.strike), 0.0)
        time_value = float(valuation.greeks.price[row]) - intrinsic
        if intrinsic <= 0.0:
            continue
        if contract.right is OptionRight.CALL:
            horizon = snapshot.as_of + timedelta(days=assignment_window_days)
            for dividend in market.dividends:
                if (
                    snapshot.as_of < dividend.ex_date <= min(horizon, contract.expiry)
                    and dividend.amount > time_value
                ):
                    flags.append(
                        ExpiryRisk(
                            "assignment",
                            contract.describe(),
                            position.quantity,
                            f"short ITM call: dividend {dividend.amount:g} exceeds time value "
                            f"{time_value:.3f} before ex-date "
                            f"{dividend.ex_date.astimezone(NEW_YORK).date()}",
                        )
                    )
        elif time_value < 0.001 * contract.strike:
            flags.append(
                ExpiryRisk(
                    "assignment",
                    contract.describe(),
                    position.quantity,
                    f"short deep ITM put: only {time_value:.3f} of time value",
                )
            )
    return tuple(flags)


@dataclass(frozen=True, slots=True)
class RiskLimit:
    """Bounds on one exposure for one underlying, or on book totals when underlying is None."""

    metric: str
    lower: float
    upper: float
    underlying: str | None = None

    def __post_init__(self) -> None:
        valid = Exposure().as_dict()
        if not isinstance(self.metric, str) or self.metric not in valid:
            raise ValidationError(f"unknown risk metric {self.metric!r}; valid: {sorted(valid)}")
        require_finite(f"limit on {self.metric}: lower", self.lower)
        require_finite(f"limit on {self.metric}: upper", self.upper)
        if self.underlying is not None:
            require_symbol("underlying", self.underlying)
        if self.lower > self.upper:
            raise ValidationError(f"limit on {self.metric}: lower exceeds upper")


@dataclass(frozen=True, slots=True)
class LimitBreach:
    limit: RiskLimit
    value: float
    utilization: float


def check_limits(
    positions: Sequence[Position], snapshot: MarketSnapshot, limits: Sequence[RiskLimit]
) -> tuple[LimitBreach, ...]:
    """Breached limits. This only reports: it does not store, escalate, or block anything.

    An underlying the market does not have raises, since a misspelled one would read as no exposure
    and pass. A value that is not a number counts as a breach.
    """
    book = value_book(positions, snapshot)
    breaches: list[LimitBreach] = []
    for limit in limits:
        if limit.underlying is None:
            value = 0.0
            for exposure in book.by_underlying.values():
                value += float(getattr(exposure, limit.metric))
        else:
            snapshot.market(limit.underlying)
            exposure = book.by_underlying.get(limit.underlying, Exposure())
            value = float(getattr(exposure, limit.metric))
        if not limit.lower <= value <= limit.upper:
            bound = limit.upper if value > limit.upper else limit.lower
            utilization = abs(value / bound) if bound != 0.0 else float("inf")
            breaches.append(LimitBreach(limit, float(value), utilization))
    return tuple(breaches)


@dataclass(frozen=True, slots=True)
class LiquidityEstimate:
    instrument: str
    quantity: int
    half_spread_cost: float
    days_to_unwind: float


def liquidity_adjusted_cost(
    position: Position, half_spread: float, adv_contracts: float, *, participation: float = 0.1
) -> LiquidityEstimate:
    """Cost of crossing half the spread to exit and the time to do so at a capped participation."""
    require_finite("half_spread", half_spread)
    require_finite("adv_contracts", adv_contracts)
    require_finite("participation", participation)
    if adv_contracts <= 0.0 or not 0.0 < participation <= 1.0:
        raise ValidationError("adv must be positive and participation in (0, 1]")
    if half_spread < 0.0:
        raise ValidationError("half_spread must be non-negative")
    multiplier = position.instrument.multiplier
    size = abs(position.quantity)
    return LiquidityEstimate(
        position.instrument.describe(),
        position.quantity,
        size * multiplier * half_spread,
        size / (adv_contracts * participation),
    )
