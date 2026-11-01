"""Full-revaluation scenarios: every number here reprices the book rather than expanding Greeks.

Vol shocks are specified at a 30-day reference tenor and decay with tenor as (30d / tau)^power,
since short-dated vol moves more than long-dated vol. Skew shocks tilt the smile linearly in
log-moneyness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import timedelta

import numpy as np
from numpy.typing import NDArray

from greeks_neutralizer.domain import (
    ExerciseStyle,
    Instrument,
    OptionContract,
    Position,
)
from greeks_neutralizer.errors import MarketDataError, ValidationError
from greeks_neutralizer.market import MarketSnapshot, UnderlyingMarket, VolSurface
from greeks_neutralizer.pricing.american import (
    AmericanBatch,
    american_price,
    exercise_is_ever_optimal,
)
from greeks_neutralizer.pricing.black import FloatArray, black_premium
from greeks_neutralizer.pricing.book import _american_batch, linear_unit_exposure
from greeks_neutralizer.time_model import calendar_years

_REF_TAU = 30.0 / 365.0
_MIN_VOL = 1e-3
_MAX_TERM_FACTOR = 3.0


@dataclass(frozen=True, slots=True)
class Scenario:
    """A market move. spot_shock is relative (-0.05 is -5%); vol_shock is in vol (0.05 is +5)."""

    name: str
    spot_shock: float = 0.0
    vol_shock: float = 0.0
    skew_shock: float = 0.0
    term_power: float = 0.5
    days_forward: float = 0.0
    sticky_strike: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        for value in (self.spot_shock, self.vol_shock, self.skew_shock, self.days_forward):
            if not np.isfinite(value):
                raise ValidationError(f"scenario {self.name}: shocks must be finite")
        if self.spot_shock <= -1.0:
            raise ValidationError(f"scenario {self.name}: spot_shock must exceed -100%")
        if self.days_forward < 0.0:
            raise ValidationError(f"scenario {self.name}: days_forward must be non-negative")


@dataclass(frozen=True, slots=True)
class ShockedSurface:
    base: VolSurface
    vol_shock: float
    skew_shock: float
    term_power: float
    k_shift: float

    def vol(self, k: FloatArray, tau: FloatArray) -> FloatArray:
        base_vol = self.base.vol(k + self.k_shift, tau)
        factor = np.minimum((_REF_TAU / np.maximum(tau, 1e-9)) ** self.term_power, _MAX_TERM_FACTOR)
        shocked = base_vol + self.vol_shock * factor + self.skew_shock * k
        return np.asarray(np.maximum(shocked, _MIN_VOL), dtype=np.float64)

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "kind": "shocked",
            "base": self.base.fingerprint_payload(),
            "vol_shock": self.vol_shock,
            "skew_shock": self.skew_shock,
            "term_power": self.term_power,
            "k_shift": self.k_shift,
        }


def shock_market(market: UnderlyingMarket, scenario: Scenario) -> UnderlyingMarket:
    # Sticky strike keeps vol at a fixed strike: a move in the forward shifts log-moneyness, so the
    # base surface is read at the strike's old k. Sticky moneyness reads it at the new k.
    k_shift = float(np.log1p(scenario.spot_shock)) if scenario.sticky_strike else 0.0
    surface = ShockedSurface(
        market.surface, scenario.vol_shock, scenario.skew_shock, scenario.term_power, k_shift
    )
    return replace(market, spot=market.spot * (1.0 + scenario.spot_shock), surface=surface)


def shock_snapshot(
    snapshot: MarketSnapshot, scenario: Scenario, symbols: Sequence[str] | None = None
) -> MarketSnapshot:
    """A new snapshot with the scenario applied to the given underlyings, or all of them."""
    chosen = set(snapshot.underlyings if symbols is None else symbols)
    return MarketSnapshot(
        snapshot.as_of + timedelta(days=scenario.days_forward),
        {
            sym: shock_market(m, scenario) if sym in chosen else m
            for sym, m in snapshot.underlyings.items()
        },
    )


def price_instruments(
    instruments: Sequence[Instrument], snapshot: MarketSnapshot, *, tree_steps: int = 51
) -> FloatArray:
    """Whole-contract value of one unit of each instrument. Price only, so it is fast.

    Futures are marked at the forward: only differences are meaningful, which is how
    `scenario_pnl_matrix` uses them.
    """
    values = np.zeros(len(instruments), dtype=np.float64)
    groups: dict[str, list[int]] = {}
    for i, instrument in enumerate(instruments):
        if isinstance(instrument, OptionContract):
            groups.setdefault(instrument.underlying, []).append(i)
        else:
            market = snapshot.market(instrument.underlying)
            price, _ = linear_unit_exposure(instrument, market, snapshot.as_of)
            values[i] = price * instrument.multiplier
    for symbol, indices in groups.items():
        contracts = [instruments[i] for i in indices]
        assert all(isinstance(c, OptionContract) for c in contracts)
        prices = _price_group(
            [c for c in contracts if isinstance(c, OptionContract)],
            snapshot.market(symbol),
            snapshot,
            tree_steps,
        )
        for idx, price, contract in zip(indices, prices, contracts, strict=True):
            assert isinstance(contract, OptionContract)
            values[idx] = float(price) * contract.multiplier
    return values


def _price_group(
    group: list[OptionContract], market: UnderlyingMarket, snapshot: MarketSnapshot, steps: int
) -> FloatArray:
    as_of = snapshot.as_of
    T = np.array([calendar_years(as_of, c.expiry) for c in group], dtype=np.float64)
    tau = np.array([market.clock.tau(as_of, c.expiry) for c in group], dtype=np.float64)
    K = np.array([c.strike for c in group], dtype=np.float64)
    omega = np.array([c.right.omega for c in group], dtype=np.float64)
    live = T > 0.0
    prices = np.maximum(omega * (market.spot - K), 0.0)
    if not np.any(live):
        return prices
    forwards = np.array(
        [market.forward(as_of, c.expiry) if t > 0 else 1.0 for c, t in zip(group, T, strict=True)],
        dtype=np.float64,
    )
    pv = np.array([market.pv_dividends(as_of, c.expiry) for c in group], dtype=np.float64)
    df = np.ones_like(T)
    df[live] = market.curve.discount(T[live])
    r = np.zeros_like(T)
    r[live] = -np.log(df[live]) / T[live]
    vol = np.zeros_like(T)
    raw = market.surface.vol(np.log(K[live] / forwards[live]), tau[live])
    if not np.all(np.isfinite(raw)) or np.any(raw < 0.0):
        raise MarketDataError(f"{market.symbol}: vol surface returned NaN or a negative vol")
    vol[live] = raw
    s = vol * np.sqrt(tau)
    prices[live] = black_premium(forwards[live], K[live], df[live], s[live], omega[live])

    q = np.full_like(T, market.carry_yield)
    is_american = np.array([c.style is ExerciseStyle.AMERICAN for c in group], dtype=bool)
    needs_tree = is_american & live & (vol > 0.0) & exercise_is_ever_optimal(omega, r, q, pv > 0.0)
    if np.any(needs_tree):
        sel = np.flatnonzero(needs_tree)
        batch: AmericanBatch = _american_batch(
            market, as_of, group, sel, T, tau, r, q, vol, K, omega
        )
        prices[sel] = american_price(batch, steps)
    return prices


def scenario_pnl_matrix(
    instruments: Sequence[Instrument],
    snapshot: MarketSnapshot,
    scenarios: Sequence[Scenario],
    *,
    tree_steps: int = 51,
) -> NDArray[np.float64]:
    """P&L of holding one unit, shape (n_scenarios, n_instruments), by full repricing."""
    base = price_instruments(instruments, snapshot, tree_steps=tree_steps)
    out = np.zeros((len(scenarios), len(instruments)), dtype=np.float64)
    for row, scenario in enumerate(scenarios):
        shocked = shock_snapshot(snapshot, scenario)
        out[row] = price_instruments(instruments, shocked, tree_steps=tree_steps) - base
    return out


def book_scenario_pnl(
    positions: Sequence[Position], snapshot: MarketSnapshot, scenarios: Sequence[Scenario]
) -> FloatArray:
    """Total book P&L per scenario by full repricing."""
    if not positions:
        return np.zeros(len(scenarios), dtype=np.float64)
    instruments = [p.instrument for p in positions]
    qty = np.array([p.quantity for p in positions], dtype=np.float64)
    return np.asarray(scenario_pnl_matrix(instruments, snapshot, scenarios) @ qty, dtype=np.float64)


# Representative single-day moves. Spot is the S&P 500 close-to-close change and vol is the change
# in the VIX close, checked against FRED VIXCLS where that series exists: see docs/risk/stress.md.
# 1987 predates the VIX, so that vol shock is a judgment value and is labeled as one.
HISTORICAL_STRESSES: tuple[Scenario, ...] = (
    Scenario(
        "1987-10-19 Black Monday",
        spot_shock=-0.2047,
        vol_shock=0.60,
        skew_shock=-0.10,
        note="Spot is the S&P 500 close. Vol shock is a judgment value; the VIX did not exist.",
    ),
    Scenario(
        "2008-09-29 TARP rejection",
        spot_shock=-0.0879,
        vol_shock=0.1198,
        skew_shock=-0.05,
    ),
    Scenario(
        "2008-10-15 post-rally selloff",
        spot_shock=-0.0903,
        vol_shock=0.1349,
        skew_shock=-0.05,
    ),
    Scenario(
        "2015-08-24 flash crash open",
        spot_shock=-0.0394,
        vol_shock=0.1274,
        skew_shock=-0.03,
    ),
    Scenario(
        "2018-02-05 Volmageddon",
        spot_shock=-0.0410,
        vol_shock=0.2001,
        skew_shock=-0.05,
    ),
    Scenario(
        "2020-03-16 COVID limit-down",
        spot_shock=-0.1198,
        vol_shock=0.2486,
        skew_shock=-0.08,
    ),
    Scenario(
        "2024-08-05 yen carry unwind",
        spot_shock=-0.0300,
        vol_shock=0.1518,
        skew_shock=-0.04,
    ),
)

HYPOTHETICAL_SHOCKS: tuple[Scenario, ...] = (
    Scenario("gap up 5% vol down 3", spot_shock=0.05, vol_shock=-0.03),
    Scenario(
        "melt up 10% vol up 5", spot_shock=0.10, vol_shock=0.05, note="Spot-up vol-up regime."
    ),
    Scenario("vol crush 5 pts flat spot", vol_shock=-0.05),
    Scenario("skew steepens", skew_shock=-0.10),
    Scenario("skew flattens", skew_shock=0.10),
    Scenario("term inversion", vol_shock=0.08, term_power=1.0),
    Scenario("one week of decay", days_forward=7.0),
)


def spot_vol_grid(
    spot_shocks: Sequence[float] = (-0.2, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1),
    vol_shocks: Sequence[float] = (-0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2),
) -> tuple[Scenario, ...]:
    return tuple(
        Scenario(f"spot {s:+.0%} vol {v * 100:+.0f}pt", spot_shock=s, vol_shock=v)
        for s in spot_shocks
        for v in vol_shocks
    )


@dataclass(frozen=True, slots=True)
class TaylorCheck:
    """Greek-based P&L against full revaluation, per scenario. Large errors mean Greeks mislead."""

    scenario: str
    full_reval: float
    taylor: float
    error: float


def taylor_vs_full(
    positions: Sequence[Position], snapshot: MarketSnapshot, scenarios: Sequence[Scenario]
) -> tuple[TaylorCheck, ...]:
    """Second-order expansion in spot and vol (with vanna, volga, and theta) versus repricing."""
    from greeks_neutralizer.pricing.book import value_book

    book = value_book(positions, snapshot)
    full = book_scenario_pnl(positions, snapshot, scenarios)
    out: list[TaylorCheck] = []
    for scenario, actual in zip(scenarios, full, strict=True):
        estimate = 0.0
        for symbol, e in book.by_underlying.items():
            spot = snapshot.market(symbol).spot
            ds = spot * scenario.spot_shock
            dv_pts = scenario.vol_shock * 100.0
            estimate += (
                e.delta_shares * ds
                + 0.5 * e.gamma_shares * ds * ds
                + e.vega_1pt * dv_pts
                + 0.5 * e.volga_1pt * dv_pts * dv_pts
                + e.vanna_1pt * dv_pts * ds
                + e.theta_1d * scenario.days_forward
            )
        out.append(TaylorCheck(scenario.name, float(actual), estimate, estimate - float(actual)))
    return tuple(out)
