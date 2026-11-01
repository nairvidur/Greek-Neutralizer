"""Book valuation scaling, linear instruments, dispatch, and the time model."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
from numpy.testing import assert_allclose

from greeks_neutralizer.domain import LinearInstrument, LinearKind, Position
from greeks_neutralizer.errors import MarketDataError, ValidationError
from greeks_neutralizer.market import (
    CashDividend,
    DiscountCurve,
    FlatVol,
    MarketSnapshot,
    UnderlyingMarket,
)
from greeks_neutralizer.pricing.black import black_greeks
from greeks_neutralizer.pricing.book import value_book, value_options
from greeks_neutralizer.time_model import VarianceClock, calendar_years, us_expiry
from tests.conftest import AS_OF, american, euro, expiry_in, flat_snapshot


def test_position_scaling_units_against_hand_values():
    book = value_book([Position(euro("call", 100, 365), 10)], flat_snapshot())
    g = black_greeks(100.0, 100.0, 1.0, 0.05, 0.0, 0.25, 1)
    e = book.lines[0].exposure
    assert_allclose(e.value, 1000 * g.price[0])
    assert_allclose(e.delta_shares, 1000 * g.delta[0])
    assert_allclose(e.delta_dollars, 1000 * g.delta[0] * 100)
    assert_allclose(e.gamma_shares, 1000 * g.gamma[0])
    assert_allclose(e.gamma_1pct, 1000 * g.gamma[0] * 100 * 100 * 0.01)
    assert_allclose(e.vega_1pt, 1000 * g.vega[0] * 0.01)
    assert_allclose(e.theta_1d, 1000 * g.theta[0] / 365)
    assert_allclose(e.rho_1pct, 1000 * g.rho[0] * 0.01)


def test_gamma_1pct_means_what_it_says():
    # Delta dollars after a +1% spot move changes by about gamma_1pct, plus a small spot term.
    contract = euro("call", 100, 30)
    base = value_book([Position(contract, 10)], flat_snapshot(spot=100.0))
    up = value_book([Position(contract, 10)], flat_snapshot(spot=100.0001))
    shares_per_dollar = (
        up.lines[0].exposure.delta_shares - base.lines[0].exposure.delta_shares
    ) / 0.0001
    assert_allclose(shares_per_dollar * 100 * 100 * 0.01, base.total_gamma_1pct, rtol=1e-3)


def test_long_and_short_cancel():
    c = euro("call", 100, 30)
    book = value_book([Position(c, 7), Position(c, -7)], flat_snapshot())
    assert_allclose(book.total_value, 0.0, atol=1e-12)
    assert_allclose(book.total_delta_dollars, 0.0, atol=1e-12)
    assert_allclose(book.total_vega_1pt, 0.0, atol=1e-12)


def test_order_of_positions_does_not_change_totals():
    legs = [Position(euro("call", 100, 30), 3), Position(american("put", 95, 60), -2)]
    a = value_book(legs, flat_snapshot())
    b = value_book(list(reversed(legs)), flat_snapshot())
    assert_allclose(a.total_value, b.total_value, rtol=1e-12)
    assert_allclose(a.total_vega_1pt, b.total_vega_1pt, rtol=1e-12)


def test_american_routes_to_tree_only_where_exercise_has_value():
    snapshot = flat_snapshot()  # positive rate, no carry yield
    v = value_options([american("call", 100, 90), american("put", 100, 90)], snapshot)
    assert v.american_tree.tolist() == [False, True]
    assert v.exercise_premium[0] == 0.0
    assert v.exercise_premium[1] > 0.0


def test_missing_market_fails_closed():
    with pytest.raises(MarketDataError, match="no market data"):
        value_book([Position(euro("call", 100, 30, underlying="ABC"), 1)], flat_snapshot())


def test_dividend_lowers_forward_and_call_value():
    plain = flat_snapshot()
    market = plain.market("XYZ")
    with_div = UnderlyingMarket(
        "XYZ",
        100.0,
        market.curve,
        market.surface,
        dividends=(CashDividend(expiry_in(30), 2.0),),
    )
    snap = MarketSnapshot(AS_OF, {"XYZ": with_div})
    contract = euro("call", 100, 90)
    assert with_div.forward(AS_OF, contract.expiry) < market.forward(AS_OF, contract.expiry)
    assert (
        value_options([contract], snap).greeks.price[0]
        < value_options([contract], plain).greeks.price[0]
    )


def test_dividends_exceeding_spot_fail_closed():
    market = UnderlyingMarket(
        "XYZ",
        1.0,
        DiscountCurve.flat(0.0),
        FlatVol(0.2),
        dividends=(CashDividend(expiry_in(1), 5.0),),
    )
    with pytest.raises(MarketDataError, match="exceeds spot"):
        market.forward(AS_OF, expiry_in(10))


def test_spot_hedge_has_unit_delta():
    stock = LinearInstrument("XYZ", "XYZ")
    book = value_book([Position(stock, 250)], flat_snapshot())
    e = book.lines[0].exposure
    assert (e.delta_shares, e.gamma_shares, e.vega_1pt) == (250.0, 0.0, 0.0)
    assert_allclose(e.value, 25000.0)


def test_future_delta_is_the_carry_factor():
    future = LinearInstrument(
        "XYZF", "XYZ", LinearKind.FUTURE, multiplier=50.0, expiry=expiry_in(365)
    )
    book = value_book([Position(future, 1)], flat_snapshot(rate=0.05))
    assert_allclose(book.lines[0].exposure.delta_shares, 50.0 * np.exp(0.05), rtol=1e-12)
    assert book.lines[0].exposure.value == 0.0


def test_proxy_requires_basis_vol_and_uses_ratio():
    with pytest.raises(ValidationError, match="basis_vol"):
        LinearInstrument("SPY", "SPX", LinearKind.PROXY, delta_ratio=0.1)
    proxy = LinearInstrument("SPY", "XYZ", LinearKind.PROXY, delta_ratio=0.1, basis_vol=0.01)
    book = value_book([Position(proxy, 100)], flat_snapshot())
    assert_allclose(book.lines[0].exposure.delta_shares, 10.0)


def test_future_requires_expiry():
    with pytest.raises(ValidationError, match="expiry"):
        LinearInstrument("F", "XYZ", LinearKind.FUTURE)


def test_curve_interpolation_is_exact_at_nodes_and_flat_beyond():
    curve = DiscountCurve((0.25, 1.0), (0.04, 0.05))
    assert_allclose(curve.zero_rate(np.array([0.25, 1.0])), [0.04, 0.05])
    assert_allclose(curve.zero_rate(np.array([0.1, 5.0])), [0.04, 0.05])
    assert_allclose(curve.discount(np.array([1.0])), [np.exp(-0.05)])
    with pytest.raises(ValidationError, match="strictly increasing"):
        DiscountCurve((1.0, 0.5), (0.04, 0.05))


def test_snapshot_fingerprint_detects_any_change():
    a = flat_snapshot()
    assert a.fingerprint() == flat_snapshot().fingerprint()
    dividend = (CashDividend(AS_OF + timedelta(days=10), 1.0),)
    changed = {
        "spot": flat_snapshot(spot=100.01),
        "vol": flat_snapshot(vol=0.2501),
        "rate": flat_snapshot(rate=0.0501),
        "carry": flat_snapshot(carry=0.0001),
        "dividends": a.replaced("XYZ", dividends=dividend),
        "dividend amount": a.replaced(
            "XYZ", dividends=(CashDividend(AS_OF + timedelta(days=10), 1.01),)
        ),
        "curve times": a.replaced("XYZ", curve=DiscountCurve((2.0,), (0.05,))),
        "clock": a.replaced("XYZ", clock=VarianceClock(session_weight=2.0)),
        "as_of": MarketSnapshot(AS_OF + timedelta(seconds=1), dict(a.underlyings)),
    }
    prints = {a.fingerprint()}
    for name, snapshot in changed.items():
        assert snapshot.fingerprint() not in prints, name
        prints.add(snapshot.fingerprint())


def test_naive_datetimes_rejected():
    with pytest.raises(ValidationError, match="timezone-aware"):
        calendar_years(datetime(2026, 1, 1), datetime(2026, 2, 1))


def test_expiry_is_timezone_correct_across_dst():
    # US DST ended 2026-11-01, so 16:00 ET is 20:00 UTC before it and 21:00 UTC after.
    before = us_expiry(date(2026, 10, 30)).astimezone(UTC)
    after = us_expiry(date(2026, 11, 2)).astimezone(UTC)
    assert (before.hour, after.hour) == (20, 21)
    assert us_expiry(date(2026, 11, 20), am_settled=True).astimezone(UTC).hour == 14


def test_default_clock_equals_calendar_time():
    clock = VarianceClock()
    start, end = AS_OF, AS_OF + timedelta(days=17, hours=3)
    assert clock.tau(start, end) == calendar_years(start, end)


def test_session_clock_removes_weekend_variance():
    clock = VarianceClock(session_weight=1.0, overnight_weight=0.1, closed_day_weight=0.0)
    friday_close = us_expiry(date(2026, 1, 9))
    monday_open = us_expiry(date(2026, 1, 12), am_settled=True)
    # Only Friday evening and early Monday accrue, at the overnight weight.
    weekend = clock.tau(friday_close, monday_open)
    session = clock.tau(monday_open, us_expiry(date(2026, 1, 12)))
    assert session > weekend > 0.0


def test_event_adds_variance_only_when_spanned():
    event = AS_OF + timedelta(days=5)
    clock = VarianceClock(events=((event, 0.01),))
    base = VarianceClock(session_weight=1.0, overnight_weight=1.0, closed_day_weight=0.999999)
    assert clock.tau(AS_OF, AS_OF + timedelta(days=3)) < 0.01
    with_event = clock.tau(AS_OF, AS_OF + timedelta(days=10))
    assert with_event - base.tau(AS_OF, AS_OF + timedelta(days=10)) == pytest.approx(0.01, rel=0.05)


def test_expired_and_live_mixed_book():
    expired = euro("put", 100, 0)
    live = euro("put", 100, 30)
    book = value_book([Position(expired, 1), Position(live, 1)], flat_snapshot(spot=90.0))
    assert_allclose(book.lines[0].exposure.value, 1000.0)
    assert book.lines[0].exposure.gamma_shares == 0.0
    assert book.lines[1].exposure.gamma_shares > 0.0
    assert len(book.issues) == 1


def test_book_flags_a_contract_priced_beyond_the_quotes():
    from greeks_neutralizer.surface import SviSlice, SviSurface

    slice_ = SviSlice(0.25, 0.005, 0.06, -0.6, 0.02, 0.15)
    surface = SviSurface((slice_,), ((-0.1, 0.1),))
    market = UnderlyingMarket("XYZ", 100.0, DiscountCurve.flat(0.0), surface)
    snap = MarketSnapshot(AS_OF, {"XYZ": market})
    inside = value_book([Position(euro("call", 100, 91), 1)], snap)
    outside = value_book([Position(euro("call", 200, 91), 1)], snap)
    assert inside.issues == ()
    assert any("EXTRAPOLATED_VOL" in issue for issue in outside.issues)
