"""Scenarios, VaR and ES, attribution, ladders, flags, limits, and margin estimates."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
from numpy.testing import assert_allclose

from greeks_neutralizer.domain import Position
from greeks_neutralizer.errors import ValidationError
from greeks_neutralizer.market import CashDividend, DiscountCurve, FlatVol, MarketSnapshot
from greeks_neutralizer.pricing.black import black_greeks
from greeks_neutralizer.pricing.book import value_book
from greeks_neutralizer.risk.margin import (
    UnderlyingClass,
    portfolio_margin_style_estimate,
    reg_t_short_option_estimate,
)
from greeks_neutralizer.risk.measures import (
    RiskLimit,
    attribute_pnl,
    backtest_var,
    check_limits,
    expected_shortfall,
    expiry_ladder,
    expiry_risks,
    historical_var,
    liquidity_adjusted_cost,
)
from greeks_neutralizer.risk.scenarios import (
    HISTORICAL_STRESSES,
    Scenario,
    book_scenario_pnl,
    scenario_pnl_matrix,
    shock_snapshot,
    spot_vol_grid,
    taylor_vs_full,
)
from tests.conftest import AS_OF, american, euro, expiry_in, flat_snapshot


def test_scenario_pnl_matches_hand_repricing():
    contract = euro("call", 100, 91.25)
    pnl = book_scenario_pnl(
        [Position(contract, 2)],
        flat_snapshot(),
        [Scenario("x", spot_shock=-0.1, vol_shock=0.05, term_power=0.0)],
    )[0]
    T = 91.25 / 365
    before = black_greeks(100.0, 100.0, T, 0.05, 0.0, 0.25, 1).price[0]
    after = black_greeks(90.0, 100.0, T, 0.05, 0.0, 0.30, 1).price[0]
    assert_allclose(pnl, 200 * (after - before), rtol=1e-10)


def test_empty_book_has_zero_pnl():
    assert_allclose(book_scenario_pnl([], flat_snapshot(), [Scenario("x", spot_shock=0.1)]), [0.0])


def test_no_shock_means_no_pnl():
    book = [Position(euro("put", 95, 30), -3), Position(american("put", 100, 60), 2)]
    assert_allclose(book_scenario_pnl(book, flat_snapshot(), [Scenario("flat")]), [0.0], atol=1e-9)


def test_short_tenor_moves_more_than_long_under_term_shock():
    scenario = Scenario("v", vol_shock=0.05, term_power=0.5)
    shocked = shock_snapshot(flat_snapshot(), scenario)
    s_sigma = shocked.market("XYZ").surface.vol(np.array([0.0]), np.array([7 / 365]))[0]
    l_sigma = shocked.market("XYZ").surface.vol(np.array([0.0]), np.array([1.0]))[0]
    # By hand, from docs/risk/stress.md's (30d / tenor)^0.5: 30/7 is 4.285714, whose root is
    # 2.070197, so 0.25 + 0.05 * 2.070197. And 30/365 is 0.082192, whose root is 0.286691.
    assert_allclose(s_sigma, 0.353510, atol=1e-6)
    assert_allclose(l_sigma, 0.264335, atol=1e-6)


def test_vol_cannot_go_negative_under_a_large_crush():
    shocked = shock_snapshot(flat_snapshot(), Scenario("crush", vol_shock=-0.9))
    assert shocked.market("XYZ").surface.vol(np.array([0.0]), np.array([0.1]))[0] > 0.0


def test_days_forward_reduces_time_and_decays_long_options():
    pnl = book_scenario_pnl(
        [Position(euro("call", 100, 30), 1)], flat_snapshot(), [Scenario("d", days_forward=7.0)]
    )[0]
    assert pnl < 0.0


def test_sticky_strike_and_moneyness_differ_only_with_a_smile():
    from greeks_neutralizer.surface import SviSlice, SviSurface

    surface = SviSurface((SviSlice(0.25, 0.005, 0.06, -0.6, 0.02, 0.15),))
    snap = flat_snapshot().replaced("XYZ", surface=surface)
    book = [Position(euro("put", 90, 60), 1)]
    strike = book_scenario_pnl(book, snap, [Scenario("s", spot_shock=-0.05, sticky_strike=True)])[0]
    money = book_scenario_pnl(book, snap, [Scenario("m", spot_shock=-0.05, sticky_strike=False)])[0]
    assert abs(strike - money) > 1.0
    flat = flat_snapshot()
    a = book_scenario_pnl(book, flat, [Scenario("s", spot_shock=-0.05, sticky_strike=True)])[0]
    b = book_scenario_pnl(book, flat, [Scenario("m", spot_shock=-0.05, sticky_strike=False)])[0]
    assert_allclose(a, b, rtol=1e-10)


def test_stress_library_contains_required_events_and_short_gamma_loses():
    names = " ".join(s.name for s in HISTORICAL_STRESSES)
    for year in ("1987", "2008", "2015-08-24", "2018-02-05", "2020-03", "2024-08-05"):
        assert year in names
    book = [Position(euro("put", 95, 30), -10)]
    assert np.all(book_scenario_pnl(book, flat_snapshot(), HISTORICAL_STRESSES) < 0.0)


def test_grid_shape_and_matrix_shape():
    grid = spot_vol_grid((-0.1, 0.0), (0.0, 0.1, 0.2))
    assert len(grid) == 6
    matrix = scenario_pnl_matrix(
        [euro("call", 100, 30), euro("put", 90, 30)], flat_snapshot(), grid
    )
    assert matrix.shape == (6, 2)


def test_taylor_error_grows_with_shock_size():
    book = [Position(euro("call", 100, 30), 10)]
    checks = taylor_vs_full(
        book,
        flat_snapshot(),
        (
            Scenario("a", spot_shock=0.005),
            Scenario("b", spot_shock=0.05),
            Scenario("c", spot_shock=0.3),
        ),
    )
    errors = [abs(c.error) for c in checks]
    assert errors[0] < errors[1] < errors[2]


@pytest.mark.parametrize(
    "bad", [{"spot_shock": -1.0}, {"days_forward": -1.0}, {"vol_shock": np.nan}]
)
def test_invalid_scenarios_rejected(bad):
    with pytest.raises(ValidationError):
        Scenario("bad", **bad)


def test_var_and_es_on_a_known_distribution():
    rng = np.random.default_rng(42)
    pnl = rng.normal(0.0, 1.0, 200_000)
    assert_allclose(historical_var(pnl, 0.99), 2.326, atol=0.03)
    assert_allclose(expected_shortfall(pnl, 0.975), 2.338, atol=0.03)
    assert expected_shortfall(pnl, 0.99) > historical_var(pnl, 0.99)


def test_var_refuses_nan_and_too_few_samples():
    with pytest.raises(ValidationError, match="NaN"):
        historical_var(np.array([1.0, np.nan] * 100), 0.9)
    with pytest.raises(ValidationError, match="samples"):
        historical_var(np.arange(5.0), 0.99)


def test_kupiec_accepts_correct_var_and_rejects_bad_var():
    rng = np.random.default_rng(1)
    pnl = rng.normal(size=2000)
    good = backtest_var(pnl, np.full(2000, 2.326), 0.99)
    bad = backtest_var(pnl, np.full(2000, 1.0), 0.99)
    assert not good.rejected_at_5pct
    assert bad.rejected_at_5pct
    assert good.expected == pytest.approx(20.0)
    zero = backtest_var(np.zeros(300), np.full(300, 1.0), 0.99)
    assert zero.exceptions == 0
    assert zero.rejected_at_5pct


def test_attribution_sums_and_explains_a_small_move():
    book = [Position(euro("call", 100, 30), 10)]
    start = flat_snapshot()
    end = MarketSnapshot(
        AS_OF + timedelta(days=1),
        {"XYZ": replace(start.market("XYZ"), spot=100.5, surface=FlatVol(0.255))},
    )
    a = attribute_pnl(book, start, end)
    explained = a.delta + a.gamma + a.vega + a.vanna + a.volga + a.theta
    assert_allclose(explained + a.unexplained, a.total, atol=1e-8)
    assert a.unexplained_ratio < 0.02
    assert a.delta > 0.0 and a.gamma > 0.0 and a.vega > 0.0 and a.theta < 0.0


def test_ladder_groups_by_expiry_sorted():
    book = [
        Position(euro("call", 100, 60), 1),
        Position(euro("put", 100, 30), 1),
        Position(euro("call", 105, 30), 2),
    ]
    ladder = expiry_ladder(book, flat_snapshot()).buckets
    assert [b.n_lines for b in ladder] == [2, 1]
    assert ladder[0].expiry < ladder[1].expiry


def test_pin_risk_flags_physical_near_strike_near_expiry_only():
    near = Position(american("call", 100, 1), 5)
    far = Position(american("call", 120, 1), 5)
    later = Position(american("call", 100, 30), 5)
    cash = Position(euro("call", 100, 1), 5)
    flags = expiry_risks([near, far, later, cash], flat_snapshot())
    pins = [f for f in flags if f.kind == "pin"]
    assert len(pins) == 1
    assert "500" in pins[0].detail


def test_assignment_flagged_on_short_itm_call_before_dividend():
    market = replace(
        flat_snapshot(spot=120.0, vol=0.15).market("XYZ"),
        dividends=(CashDividend(expiry_in(3), 5.0),),
    )
    snap = MarketSnapshot(AS_OF, {"XYZ": market})
    flags = expiry_risks([Position(american("call", 100, 20), -5)], snap)
    assert any(f.kind == "assignment" and "dividend" in f.detail for f in flags)
    long_side = expiry_risks([Position(american("call", 100, 20), 5)], snap)
    assert not any(f.kind == "assignment" for f in long_side)


def test_assignment_flagged_on_short_deep_itm_put():
    snap = flat_snapshot(spot=50.0, vol=0.2)
    flags = expiry_risks([Position(american("put", 100, 20), -1)], snap)
    assert any(f.kind == "assignment" and "put" in f.detail for f in flags)


def test_limits_report_only_breaches_and_reject_unknown_metrics():
    book = [Position(euro("call", 100, 30), 10)]
    snap = flat_snapshot()
    delta = value_book(book, snap).by_underlying["XYZ"].delta_shares
    breached = check_limits(book, snap, [RiskLimit("delta_shares", -1.0, 1.0, "XYZ")]).breaches
    assert len(breached) == 1
    assert_allclose(breached[0].value, delta)
    assert breached[0].utilization > 1.0
    assert check_limits(book, snap, [RiskLimit("delta_shares", -1e9, 1e9)]).breaches == ()
    with pytest.raises(ValidationError, match="unknown risk metric"):
        check_limits(book, snap, [RiskLimit("delta", -1.0, 1.0)])


def test_liquidity_cost_and_time():
    estimate = liquidity_adjusted_cost(Position(euro("call", 100, 30), -500), 0.05, 1000.0)
    assert_allclose(estimate.half_spread_cost, 500 * 100 * 0.05)
    assert_allclose(estimate.days_to_unwind, 5.0)


def test_reg_t_matches_a_worked_example():
    # Short 1 put, K=95, S=100, premium P: P + max(20%*100 - 5, 10%*95) = P + 15 per share.
    snap = flat_snapshot()
    put = euro("put", 95, 30)
    premium = value_book([Position(put, 1)], snap).lines[0].unit_price
    estimate = reg_t_short_option_estimate([Position(put, -1)], snap)
    assert_allclose(estimate.requirement, 100 * (premium + 15.0))
    assert estimate.is_estimate


def test_reg_t_minimum_binds_deep_otm_and_index_rate_is_lower():
    snap = flat_snapshot()
    deep = euro("put", 70, 30)
    premium = value_book([Position(deep, 1)], snap).lines[0].unit_price
    est = reg_t_short_option_estimate([Position(deep, -1)], snap)
    assert_allclose(est.requirement, 100 * (premium + 7.0))
    atm = [Position(euro("put", 100, 30), -1)]
    equity = reg_t_short_option_estimate(atm, snap).requirement
    index = reg_t_short_option_estimate(atm, snap, {"XYZ": UnderlyingClass.BROAD_INDEX}).requirement
    assert index < equity


def test_long_options_need_no_reg_t_margin():
    assert (
        reg_t_short_option_estimate(
            [Position(euro("call", 100, 30), 5)], flat_snapshot()
        ).requirement
        == 0.0
    )


def test_portfolio_margin_style_uses_scan_and_floor():
    snap = flat_snapshot()
    short = [Position(euro("put", 95, 30), -10)]
    est = portfolio_margin_style_estimate(short, snap)
    worst = -book_scenario_pnl(short, snap, [Scenario("x", spot_shock=-0.15)])[0]
    assert_allclose(est.requirement, worst, rtol=1e-6)
    far_long = [Position(euro("put", 40, 7), 10)]
    floor = portfolio_margin_style_estimate(far_long, snap)
    assert_allclose(floor.requirement, 10 * 37.5)
    assert floor.is_estimate


def test_curve_change_flows_through_snapshot_replace():
    snap = flat_snapshot()
    higher = snap.replaced("XYZ", curve=DiscountCurve.flat(0.10))
    assert higher.market("XYZ").curve.zero_rates == (0.10,)
