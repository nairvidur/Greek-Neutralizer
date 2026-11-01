from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from greeks_neutralizer.domain import ExerciseStyle, OptionContract, OptionRight, SettlementKind
from greeks_neutralizer.market import DiscountCurve, FlatVol, MarketSnapshot, UnderlyingMarket

AS_OF = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)


def expiry_in(days: float) -> datetime:
    return AS_OF + timedelta(days=days)


def euro(right: str, strike: float, days: float, underlying: str = "XYZ") -> OptionContract:
    return OptionContract(
        underlying,
        OptionRight(right),
        strike,
        expiry_in(days),
        ExerciseStyle.EUROPEAN,
        settlement=SettlementKind.CASH,
    )


def american(right: str, strike: float, days: float, underlying: str = "XYZ") -> OptionContract:
    return OptionContract(underlying, OptionRight(right), strike, expiry_in(days))


def flat_snapshot(
    spot: float = 100.0, rate: float = 0.05, vol: float = 0.25, carry: float = 0.0
) -> MarketSnapshot:
    market = UnderlyingMarket("XYZ", spot, DiscountCurve.flat(rate), FlatVol(vol), carry)
    return MarketSnapshot(AS_OF, {"XYZ": market})


@pytest.fixture
def snapshot() -> MarketSnapshot:
    return flat_snapshot()
