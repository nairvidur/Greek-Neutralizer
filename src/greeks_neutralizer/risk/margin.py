"""Margin ESTIMATES. These are not official calculations and must not be used to meet a margin call.

Implemented:
  - Reg T style strategy-free short option requirement: the standard listed-options formula. Offsets
    from recognized spreads are not modeled, so this overstates requirements for spreaded books.
  - Portfolio margin in the style of FINRA Rule 4210(g): the worst loss across a price scan with
    no vol shock and a per-contract minimum, using the rule's standard scan ranges. TIMS-derived vol
    assumptions and class offsets are not modeled, so this will not match a broker's number.

Not implemented: CME SPAN and SPAN 2, OCC STANS, and ISDA SIMM. Those need licensed parameters or
files from the CCP or vendor (SIMM requires an ISDA license), and approximating them would present
invented numbers as margin. See docs/risk/margin.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from greeks_neutralizer.domain import OptionContract, OptionRight, Position
from greeks_neutralizer.market import MarketSnapshot
from greeks_neutralizer.pricing.book import value_book
from greeks_neutralizer.risk.scenarios import Scenario, book_scenario_pnl


class UnderlyingClass(StrEnum):
    EQUITY = "equity"
    BROAD_INDEX = "broad_index"


@dataclass(frozen=True, slots=True)
class MarginEstimate:
    method: str
    requirement: float
    detail: str
    is_estimate: bool = True


def _base_rate(cls: UnderlyingClass) -> float:
    return 0.15 if cls is UnderlyingClass.BROAD_INDEX else 0.20


def reg_t_short_option_estimate(
    positions: Sequence[Position],
    snapshot: MarketSnapshot,
    classes: dict[str, UnderlyingClass] | None = None,
) -> MarginEstimate:
    """Sum over short options of premium + max(rate * S - OTM amount, 10% of S or K). No offsets."""
    classes = classes or {}
    book = value_book(positions, snapshot)
    total = 0.0
    for line in book.lines:
        instrument = line.position.instrument
        if not isinstance(instrument, OptionContract) or line.position.quantity >= 0:
            continue
        spot = snapshot.market(instrument.underlying).spot
        rate = _base_rate(classes.get(instrument.underlying, UnderlyingClass.EQUITY))
        if instrument.right is OptionRight.CALL:
            otm = max(instrument.strike - spot, 0.0)
            minimum = 0.10 * spot
        else:
            otm = max(spot - instrument.strike, 0.0)
            minimum = 0.10 * instrument.strike
        per_share = line.unit_price + max(rate * spot - otm, minimum)
        total += abs(line.position.quantity) * instrument.multiplier * per_share
    return MarginEstimate(
        "reg_t_style_no_offsets",
        total,
        "Strategy-free short option formula. Recognized spread offsets are not applied.",
    )


# FINRA 4210(g) scan ranges: +/-15% for equities, and +6%/-8% for broad-based index products.
_SCAN: dict[UnderlyingClass, tuple[float, float]] = {
    UnderlyingClass.EQUITY: (-0.15, 0.15),
    UnderlyingClass.BROAD_INDEX: (-0.08, 0.06),
}
_MIN_PER_CONTRACT = 37.50


def portfolio_margin_style_estimate(
    positions: Sequence[Position],
    snapshot: MarketSnapshot,
    classes: dict[str, UnderlyingClass] | None = None,
    *,
    points: int = 10,
) -> MarginEstimate:
    """Worst loss across an evenly spaced price scan per underlying, with a per-contract minimum."""
    classes = classes or {}
    requirement = 0.0
    by_underlying: dict[str, list[Position]] = {}
    for position in positions:
        by_underlying.setdefault(position.instrument.underlying, []).append(position)
    details: list[str] = []
    for symbol, group in sorted(by_underlying.items()):
        cls = classes.get(symbol, UnderlyingClass.EQUITY)
        low, high = _SCAN[cls]
        scan = tuple(
            Scenario(f"scan {move:+.2%}", spot_shock=float(move))
            for move in np.linspace(low, high, points + 1)
        )
        single = MarketSnapshot(snapshot.as_of, {symbol: snapshot.market(symbol)})
        pnl = book_scenario_pnl(group, single, scan)
        worst = max(-float(np.min(pnl)), 0.0)
        contracts = sum(abs(p.quantity) for p in group if isinstance(p.instrument, OptionContract))
        floor = contracts * _MIN_PER_CONTRACT
        requirement += max(worst, floor)
        details.append(f"{symbol}: scan loss {worst:,.0f}, floor {floor:,.0f}")
    return MarginEstimate(
        "portfolio_margin_style_price_scan",
        requirement,
        "; ".join(details) + ". Price scan only: no vol shock or class offsets.",
    )
