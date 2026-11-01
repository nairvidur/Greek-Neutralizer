"""Regression pins for defects found by independent review. Each failed before its fix."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from numpy.testing import assert_allclose

from greeks_neutralizer.backtest.engine import DayMarket, run_backtest
from greeks_neutralizer.backtest.policies import DeltaDaily, Unhedged
from greeks_neutralizer.domain import (
    NEW_YORK,
    ExerciseStyle,
    LinearInstrument,
    LinearKind,
    Position,
    Quote,
)
from greeks_neutralizer.hedging.optimizer import solve_hedge
from greeks_neutralizer.hedging.problem import (
    DEFAULT_WEIGHTS,
    GreekName,
    GreekTarget,
    HedgeCandidate,
    HedgeConfig,
    HedgeProblem,
)
from greeks_neutralizer.market import (
    CashDividend,
    DiscountCurve,
    FlatVol,
    MarketSnapshot,
    UnderlyingMarket,
)
from greeks_neutralizer.pricing.american import exercise_is_ever_optimal
from greeks_neutralizer.pricing.book import value_book, value_options
from greeks_neutralizer.risk.measures import attribute_pnl
from greeks_neutralizer.risk.scenarios import Scenario, shock_snapshot, taylor_vs_full
from greeks_neutralizer.surface import SviSlice, SviSurface
from greeks_neutralizer.time_model import VarianceClock, calendar_years
from tests.conftest import AS_OF, american, euro, expiry_in


def snapshot_with(
    spot=100.0, rate=0.05, vol=0.25, carry=0.0, dividends=(), curve=None, as_of=AS_OF
) -> MarketSnapshot:
    market = UnderlyingMarket(
        "XYZ", spot, curve or DiscountCurve.flat(rate), FlatVol(vol), carry, dividends
    )
    return MarketSnapshot(as_of, {"XYZ": market})


def future(days=365, multiplier=1.0, symbol="XYZF"):
    return LinearInstrument(
        symbol, "XYZ", LinearKind.FUTURE, multiplier=multiplier, expiry=expiry_in(days)
    )


STOCK = LinearInstrument("XYZ", "XYZ")


class Days:
    """A deterministic market from a list of snapshots, with zero-spread quotes."""

    def __init__(self, snapshots, half=0.0):
        self.snapshots, self.half = snapshots, half

    def __iter__(self):
        for snap in self.snapshots:

            def quote_for(instrument, snap=snap):
                ts = snap.as_of
                return HedgeCandidate(instrument, Quote(1.0, 1.0 + 2 * self.half, 10**6, 10**6, ts))

            yield DayMarket(snap.as_of, snap, (quote_for(STOCK),), quote_for)


class Hold:
    def __init__(self, position):
        self.position = position

    def trades(self, day, holdings):
        return () if holdings else (self.position,)


def _path(spots, step_days=30, **kwargs):
    return [
        snapshot_with(spot=s, as_of=AS_OF + timedelta(days=i * step_days), **kwargs)
        for i, s in enumerate(spots)
    ]


# --- Backtest accounting ---------------------------------------------------------------------


def test_future_notional_is_not_financed():
    # A future costs nothing to hold. Its carry is already in the forward's roll.
    result = run_backtest(Days(_path([100.0, 100.0])), Hold(Position(future(), 1)), Unhedged())
    assert result.days[0].financing == 0.0
    assert_allclose(result.days[0].pnl, result.days[0].book_pnl)


def test_shareholder_is_paid_the_carry_yield():
    days = _path([100.0, 100.0], rate=0.0, carry=0.02)
    result = run_backtest(Days(days), Hold(Position(STOCK, 100)), Unhedged())
    assert_allclose(result.days[0].pnl, 100 * 100.0 * 0.02 * 30 / 365, rtol=1e-9)


def test_short_stock_across_an_ex_date_pays_the_dividend():
    # Spot falls by the dividend: true P&L is zero, not +500.
    ex_date = AS_OF + timedelta(days=10)
    start = snapshot_with(spot=100.0, rate=0.0, dividends=(CashDividend(ex_date, 5.0),))
    end = snapshot_with(spot=95.0, rate=0.0, as_of=AS_OF + timedelta(days=30))
    result = run_backtest(Days([start, end]), Hold(Position(STOCK, -100)), Unhedged())
    assert_allclose(result.days[0].pnl, 0.0, atol=1e-9)


def test_trade_cost_includes_impact_beyond_displayed_size():
    class Thin(Days):
        def __iter__(self):
            for day in super().__iter__():

                def quote_for(instrument, ts=day.as_of):
                    return HedgeCandidate(
                        instrument, Quote(1.0, 2.0, 10, 10, ts), impact_per_unit=2.0
                    )

                yield replace(day, quote_for=quote_for)

    contract = euro("call", 100, 90)
    result = run_backtest(
        Thin(_path([100.0, 100.0], step_days=1)), Hold(Position(contract, 50)), Unhedged()
    )
    assert_allclose(result.days[0].book_cost, 50 * 100 * 0.5 + 40 * 2.0)


def test_last_day_trade_costs_are_reported():
    class TradeLast:
        def __init__(self, n):
            self.n, self.seen = n, 0

        def trades(self, day, holdings):
            self.seen += 1
            return (Position(STOCK, 10),) if self.seen == self.n else ()

    result = run_backtest(
        Days(_path([100.0, 100.0], step_days=1), half=0.5), TradeLast(2), Unhedged()
    )
    assert_allclose(result.final_day_cost, 10 * 0.5)


def test_delta_policy_sizes_a_future_by_its_delta():
    # A one-year future has delta F/S, so sizing it one-for-one overhedges by the carry.
    contract = euro("call", 100, 365)
    snap = snapshot_with()

    def quote_for(instrument):
        return HedgeCandidate(instrument, Quote(1.0, 1.0, 10**6, 10**6, AS_OF))

    instrument = future()
    day = DayMarket(AS_OF, snap, (quote_for(instrument),), quote_for)
    book = [Position(contract, 100)]
    trades = DeltaDaily("XYZ").trades(day, book, [])
    after = value_book([*book, *trades], snap).by_underlying["XYZ"].delta_shares
    assert abs(after) < 1.0


# --- Futures ---------------------------------------------------------------------------------


def test_expired_future_has_no_delta_and_is_flagged():
    old = LinearInstrument(
        "F", "XYZ", LinearKind.FUTURE, multiplier=50.0, expiry=AS_OF - timedelta(days=10)
    )
    book = value_book([Position(old, -10)], snapshot_with())
    assert book.lines[0].exposure.delta_shares == 0.0
    assert any("EXPIRED" in issue for issue in book.issues)


def test_expired_future_is_not_a_candidate_and_leaves_the_book():
    old = LinearInstrument("F", "XYZ", LinearKind.FUTURE, expiry=AS_OF - timedelta(days=1))
    problem = HedgeProblem(
        (Position(euro("call", 100, 30), 1),),
        (
            HedgeCandidate(old, Quote(99.0, 101.0, 10, 10, AS_OF)),
            HedgeCandidate(STOCK, Quote(99.9, 100.1, 10**6, 10**6, AS_OF)),
        ),
        (GreekTarget("XYZ", GreekName.DELTA, weight=1.0),),
    )
    proposal = solve_hedge(problem, snapshot_with())
    assert any(item.reason == "expired" for item in proposal.excluded)
    assert all(leg.instrument != old for leg in proposal.legs)

    class OpenOnce:
        def __init__(self):
            self.opened = False

        def trades(self, day, holdings):
            if self.opened:
                return ()
            self.opened = True
            return (Position(future(days=1), 1),)

    # Spot keeps moving after expiry, so a position that had not left would keep making P&L.
    days = _path([100.0, 110.0, 120.0, 130.0], step_days=5)
    result = run_backtest(Days(days), OpenOnce(), Unhedged())
    assert [day.book_pnl for day in result.days[1:]] == [0.0, 0.0]


def test_future_delta_with_dividends_matches_finite_difference():
    divs = (CashDividend(expiry_in(30), 5.0),)
    instrument = future(days=90)
    delta = (
        value_book([Position(instrument, 1)], snapshot_with(dividends=divs))
        .lines[0]
        .exposure.delta_shares
    )

    def price(spot):
        return (
            snapshot_with(spot=spot, dividends=divs).market("XYZ").forward(AS_OF, instrument.expiry)
        )

    assert_allclose(delta, (price(100.01) - price(99.99)) / 0.02, rtol=1e-9)


def test_futures_report_rho_theta_and_charm():
    instrument = future(multiplier=50.0)
    e = value_book([Position(instrument, 1)], snapshot_with()).lines[0].exposure
    forward = lambda r, as_of=AS_OF: (  # noqa: E731
        snapshot_with(rate=r, as_of=as_of).market("XYZ").forward(as_of, instrument.expiry)
    )
    bump = 1e-4
    rho = 50.0 * (forward(0.05 + bump) - forward(0.05 - bump)) / (2 * bump) * 0.01
    assert_allclose(e.rho_1pct, rho, rtol=1e-6)
    theta = 50.0 * (forward(0.05, AS_OF + timedelta(days=1)) - forward(0.05))
    assert_allclose(e.theta_1d, theta, rtol=1e-3)
    assert e.charm_1d < 0.0


def test_premium_budget_ignores_futures_notional():
    instrument = future(multiplier=50.0)
    problem = HedgeProblem(
        (Position(euro("call", 100, 30), -100),),
        (HedgeCandidate(instrument, Quote(104.9, 105.1, 10**5, 10**5, AS_OF)),),
        (GreekTarget("XYZ", GreekName.DELTA, weight=1.0),),
        (),
        HedgeConfig(premium_budget=1000.0),
    )
    proposal = solve_hedge(problem, snapshot_with())
    assert proposal.legs
    assert proposal.net_premium == 0.0


def test_two_futures_with_different_expiries_have_different_keys():
    assert future(days=30, symbol="ES").key != future(days=120, symbol="ES").key


# --- Time --------------------------------------------------------------------------------------


def test_time_to_expiry_is_elapsed_time_across_dst():
    # 2026-11-01 ends US DST: Oct 30 16:00 ET to Nov 2 16:00 ET is 73 hours, not 72.
    start = datetime(2026, 10, 30, 16, 0, tzinfo=NEW_YORK)
    end = datetime(2026, 11, 2, 16, 0, tzinfo=NEW_YORK)
    assert_allclose(calendar_years(start, end) * 365 * 24, 73.0)
    assert calendar_years(start.astimezone(UTC), end) == calendar_years(start, end)


def test_same_instant_gives_one_price_in_any_zone():
    et = datetime(2026, 10, 30, 16, 0, tzinfo=NEW_YORK)
    contract = replace(euro("call", 100, 30), expiry=datetime(2026, 11, 2, 16, 0, tzinfo=NEW_YORK))
    a = value_options([contract], snapshot_with(as_of=et)).greeks.price[0]
    b = value_options([contract], snapshot_with(as_of=et.astimezone(UTC))).greeks.price[0]
    assert_allclose(a, b, rtol=1e-14)


def test_weighted_clock_counts_elapsed_time_across_dst():
    clock = VarianceClock(session_weight=1.0, overnight_weight=1.0, closed_day_weight=0.999999)
    # Noon to noon is 25 elapsed hours on the day DST ends, and 24 on an ordinary day.
    across = clock.tau(
        datetime(2026, 10, 31, 12, 0, tzinfo=NEW_YORK),
        datetime(2026, 11, 1, 12, 0, tzinfo=NEW_YORK),
    )
    ordinary = clock.tau(
        datetime(2026, 10, 29, 12, 0, tzinfo=NEW_YORK),
        datetime(2026, 10, 30, 12, 0, tzinfo=NEW_YORK),
    )
    assert_allclose(across / ordinary, 25.0 / 24.0, rtol=1e-3)


# --- Attribution and Taylor -------------------------------------------------------------------


def test_attribution_total_includes_futures():
    start = snapshot_with(spot=100.0)
    end = snapshot_with(spot=110.0)
    a = attribute_pnl([Position(future(), 100)], start, end)
    assert a.total > 0.0
    assert_allclose(a.total, a.delta, rtol=1e-9)
    assert abs(a.unexplained) < 1e-6


def test_expired_option_is_not_a_vol_move():
    start = snapshot_with()
    end = snapshot_with(as_of=AS_OF + timedelta(days=31))
    a = attribute_pnl([Position(euro("call", 100, 30), 100)], start, end)
    assert a.vega == 0.0


def test_taylor_uses_the_scenarios_own_term_structure_and_skew():
    book = [Position(euro("call", 100, 365), 10)]
    term = taylor_vs_full(book, snapshot_with(), (Scenario("v", vol_shock=0.05),))[0]
    assert abs(term.error) < 0.05 * abs(term.full_reval)
    skew = taylor_vs_full(
        [Position(euro("put", 90, 60), 10)], snapshot_with(), (Scenario("s", skew_shock=-0.05),)
    )[0]
    assert abs(skew.error) < 0.1 * abs(skew.full_reval)


# --- Hedge configuration and capacity ------------------------------------------------------------


def test_default_gamma_weight_is_the_pnl_of_a_3pct_move():
    contract = euro("call", 100, 30)
    e = value_book([Position(contract, 1)], snapshot_with()).lines[0].exposure
    move = 100.0 * 0.03
    assert_allclose(
        DEFAULT_WEIGHTS[GreekName.GAMMA] * e.gamma_1pct, 0.5 * e.gamma_shares * move * move
    )


def test_no_displayed_size_on_one_side_blocks_that_side():
    contract = euro("call", 105, 60)
    # A fair quote, so that nothing but its size can keep it out of the hedge.
    fair_value = value_options([contract], snapshot_with()).greeks.price[0]

    def hedge(book_quantity, bid_size, ask_size, **caps):
        quote = Quote(fair_value - 0.05, fair_value + 0.05, bid_size, ask_size, AS_OF)
        return solve_hedge(
            HedgeProblem(
                (Position(euro("call", 100, 30), book_quantity),),
                (HedgeCandidate(contract, quote, **caps),),
                (GreekTarget("XYZ", GreekName.VEGA, weight=10.0),),
            ),
            snapshot_with(),
        )

    # A short vega book buys this option, and a long one sells it, when the size is there.
    assert [leg.side for leg in hedge(-10, 100, 100).legs] == ["BUY"]
    assert [leg.side for leg in hedge(10, 100, 100).legs] == ["SELL"]
    for book_quantity, bid_size, ask_size in ((-10, 100, 0), (10, 0, 100)):
        # Allowing depth does not help: with nothing displayed, there is no market to go deep into.
        for caps in ({}, {"max_buy": 100, "max_sell": 100}):
            blocked = hedge(book_quantity, bid_size, ask_size, **caps)
            assert blocked.legs == ()
            assert blocked.excluded == ()


def test_a_source_without_sizes_can_still_trade_when_policy_allows():
    contract = euro("call", 105, 60)
    fair_value = value_options([contract], snapshot_with()).greeks.price[0]
    candidate = HedgeCandidate(contract, Quote(fair_value - 0.05, fair_value + 0.05, 0, 0, AS_OF))
    problem = HedgeProblem(
        (Position(euro("call", 100, 30), -10),),
        (candidate,),
        (GreekTarget("XYZ", GreekName.VEGA, weight=10.0),),
        (),
        HedgeConfig(require_quote_size=False),
    )
    assert solve_hedge(problem, snapshot_with()).legs


# --- American engine -------------------------------------------------------------------------


def test_tree_runs_where_negative_rates_or_carry_make_exercise_valuable():
    assert exercise_is_ever_optimal(
        np.array([1.0, -1.0]),
        np.array([-0.02, -0.01]),
        np.array([0.0, -0.03]),
        np.array([False, False]),
    ).tolist() == [True, True]


@pytest.mark.parametrize(
    ("contract", "kwargs", "intrinsic"),
    [
        (american("call", 60, 365), {"rate": -0.02}, 40.0),
        (american("put", 140, 365), {"rate": -0.01, "carry": -0.03}, 40.0),
        (american("put", 100, 365), {"spot": 80.0, "vol": 0.0}, 20.0),
    ],
    ids=["negative-rate-call", "negative-carry-put", "zero-vol-put"],
)
def test_american_is_never_below_intrinsic(contract, kwargs, intrinsic):
    price = value_options([contract], snapshot_with(**kwargs)).greeks.price[0]
    assert price >= intrinsic - 1e-9


def test_american_charm_is_flat_in_the_exercise_region():
    deep = value_options([american("put", 200, 180)], snapshot_with())
    assert_allclose(deep.greeks.delta[0], -1.0, atol=1e-6)
    assert abs(deep.greeks.charm[0]) < 1e-4


def test_american_charm_matches_finite_difference_of_delta():
    contract = american("put", 105, 180)
    now = value_options([contract], snapshot_with(), tree_steps=201).greeks
    later = value_options(
        [contract], snapshot_with(as_of=AS_OF + timedelta(days=1)), tree_steps=201
    ).greeks
    numeric = (later.delta[0] - now.delta[0]) * 365.0
    assert_allclose(now.charm[0], numeric, atol=0.02, rtol=0.1)


def test_american_theta_does_not_jump_before_an_ex_date():
    contract = american("call", 100, 30)
    far = snapshot_with(dividends=(CashDividend(AS_OF + timedelta(days=5), 2.0),))
    near = snapshot_with(dividends=(CashDividend(AS_OF + timedelta(hours=12), 2.0),))
    far_theta = value_options([contract], far).greeks.theta[0]
    near_theta = value_options([contract], near).greeks.theta[0]
    assert far_theta < 0.0
    assert near_theta < 0.0
    # Holding S fixed across the 2.00 ex-date would add about delta * 2.00 per day, against a theta
    # of about 0.05, so a bump stepping over the date moves theta many times its own size.
    assert abs(near_theta - far_theta) < 0.5 * abs(far_theta)


def test_tree_discounts_dividends_on_the_curve():
    # Exercise premium is the difference from the European, so both must agree on the dividends.
    curve = DiscountCurve((0.25, 1.0), (0.01, 0.06))
    divs = (CashDividend(expiry_in(30), 5.0), CashDividend(expiry_in(120), 5.0))
    snap = snapshot_with(curve=curve, dividends=divs)
    american_call = american("call", 110, 365)
    european_call = replace(american_call, style=ExerciseStyle.EUROPEAN)
    am = value_options([american_call], snap)
    eu = value_options([european_call], snap)
    gap = am.greeks.price[0] - eu.greeks.price[0]
    assert_allclose(gap, am.exercise_premium[0], atol=2e-3)
    # The same cash flows in another form: one flat rate at the contract's own zero rate, with each
    # dividend's amount scaled to keep its PV on the curve. They are one option, so the tree must
    # price them alike. A tree discounting the curve case's dividends at the flat rate would not.
    expiry_t = calendar_years(AS_OF, american_call.expiry)
    flat_rate = -np.log(float(curve.discount(expiry_t))) / expiry_t
    restated = []
    for dividend in divs:
        t = calendar_years(AS_OF, dividend.ex_date)
        pv = dividend.amount * float(curve.discount(t))
        restated.append(CashDividend(dividend.ex_date, pv * np.exp(flat_rate * t)))
    flat_snap = snapshot_with(curve=DiscountCurve.flat(flat_rate), dividends=tuple(restated))
    flat_am = value_options([american_call], flat_snap)
    assert_allclose(am.greeks.price[0], flat_am.greeks.price[0], rtol=1e-9)
    assert_allclose(am.exercise_premium[0], flat_am.exercise_premium[0], rtol=1e-7)


def test_european_theta_and_rho_include_the_dividend_pv():
    divs = (CashDividend(expiry_in(60), 3.0), CashDividend(expiry_in(150), 3.0))
    contract = euro("call", 100, 180)

    def value(rate=0.05, as_of=AS_OF):
        return value_options([contract], snapshot_with(rate=rate, dividends=divs, as_of=as_of))

    base = value()
    bump = 1e-4
    rho = (value(0.05 + bump).greeks.price[0] - value(0.05 - bump).greeks.price[0]) / (2 * bump)
    assert_allclose(base.greeks.rho[0], rho, rtol=1e-3)
    hour = 1.0 / (365.0 * 24.0)
    later = value(as_of=AS_OF + timedelta(hours=1)).greeks.price[0]
    assert_allclose(base.greeks.theta[0], (later - base.greeks.price[0]) / hour, rtol=2e-3)


# --- Scenarios ---------------------------------------------------------------------------------


def _smile_surface():
    return SviSurface((SviSlice(0.25, 0.005, 0.06, -0.6, 0.02, 0.15),), ((-0.1, 0.1),))


def test_sticky_strike_holds_vol_at_a_strike_with_dividends():
    divs = (CashDividend(AS_OF + timedelta(days=30), 8.0),)
    base = snapshot_with(dividends=divs).replaced("XYZ", surface=_smile_surface())
    shocked = shock_snapshot(base, Scenario("s", spot_shock=-0.2, sticky_strike=True))
    contract = euro("put", 90, 90)
    before = value_options([contract], base).vol[0]
    after = value_options([contract], shocked).vol[0]
    assert_allclose(after, before, atol=1e-6)


def test_shocked_surface_keeps_the_extrapolation_flag():
    base = snapshot_with().replaced("XYZ", surface=_smile_surface())
    shocked = shock_snapshot(base, Scenario("s", spot_shock=0.0025))
    outside = euro("put", 50, 91)
    assert value_options([outside], base).extrapolated[0]
    assert value_options([outside], shocked).extrapolated[0]


# --- Fingerprint -------------------------------------------------------------------------------


def test_fingerprint_covers_clock_and_quoted_ranges():
    a = snapshot_with().replaced("XYZ", clock=VarianceClock(session_weight=2.0))
    b = snapshot_with().replaced("XYZ", clock=VarianceClock(session_weight=5.0))
    assert a.fingerprint() != b.fingerprint()
    s = _smile_surface()
    c = snapshot_with().replaced("XYZ", surface=s)
    d = snapshot_with().replaced("XYZ", surface=SviSurface(s.slices, ((-0.2, 0.2),)))
    assert c.fingerprint() != d.fingerprint()
