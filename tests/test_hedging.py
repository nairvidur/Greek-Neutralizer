"""Hedge optimizer: exact neutralization, integer lots, costs, constraints, and verification."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from greeks_neutralizer.domain import LinearInstrument, LinearKind, Position, Quote
from greeks_neutralizer.errors import HedgeInfeasibleError
from greeks_neutralizer.hedging.diagnostics import (
    hedge_decay_forecast,
    hedge_frontier,
    whalley_wilmott_band,
)
from greeks_neutralizer.hedging.optimizer import solve_hedge
from greeks_neutralizer.hedging.problem import (
    GreekName,
    GreekTarget,
    HedgeCandidate,
    HedgeConfig,
    HedgeProblem,
    ProposalStatus,
)
from greeks_neutralizer.pricing.black import black_greeks
from greeks_neutralizer.pricing.book import value_book
from greeks_neutralizer.risk.scenarios import HISTORICAL_STRESSES, Scenario
from tests.conftest import AS_OF, euro, flat_snapshot

STOCK = LinearInstrument("XYZ", "XYZ")


def q(mid, half=0.02, size=1000, age=0.0):
    return Quote(mid - half, mid + half, size, size, AS_OF - timedelta(seconds=age))


def option_candidate(right, strike, days, half=0.02, size=1000, **kw):
    contract = euro(right, strike, days)
    T = days / 365
    price = black_greeks(100.0, strike, T, 0.05, 0.0, 0.25, (right == "call" and 1) or -1).price[0]
    return HedgeCandidate(contract, q(price, half, size), **kw)


def stock_candidate(half=0.01, size=1_000_000, **kw):
    return HedgeCandidate(STOCK, q(100.0, half, size), **kw)


BOOK = (
    Position(euro("call", 100, 30), 10),
    Position(euro("put", 95, 30), -5),
    Position(euro("call", 110, 60), 8),
)
CANDIDATES = (
    stock_candidate(),
    option_candidate("put", 95, 60),
    option_candidate("call", 105, 60),
    option_candidate("call", 100, 90),
    option_candidate("put", 90, 90),
    option_candidate("call", 110, 90),
)
DGV = tuple(
    GreekTarget("XYZ", g, weight=w)
    for g, w in ((GreekName.DELTA, 1.0), (GreekName.GAMMA, 1.0), (GreekName.VEGA, 1.0))
)


def solve(**kw):
    config = kw.pop("config", HedgeConfig(risk_aversion=10.0))
    problem = HedgeProblem(
        kw.pop("book", BOOK),
        kw.pop("candidates", CANDIDATES),
        kw.pop("targets", DGV),
        kw.pop("scenarios", ()),
        config,
    )
    return solve_hedge(problem, flat_snapshot(), **kw)


def test_high_risk_aversion_reduces_all_greeks_within_one_contract():
    proposal = solve()
    for line in proposal.residuals:
        assert abs(line.after) < abs(line.before)
    after = proposal.after["XYZ"]
    # Within what one integer contract of the smallest hedge can resolve.
    assert abs(after.delta_shares) < 100.0
    assert abs(after.vega_1pt) < 15.0


def test_quantities_are_integer_lots():
    proposal = solve()
    assert all(isinstance(leg.quantity, int) for leg in proposal.legs)


def test_more_than_three_instruments_and_linear_hedge_are_supported():
    # Baseline defect: the old solver accepted exactly three option hedges and no underlying.
    proposal = solve()
    instruments = {leg.instrument.key for leg in proposal.legs}
    assert STOCK.key in instruments
    assert len(CANDIDATES) > 3


def test_premium_is_accounted_for_and_separate_from_cost():
    # Baseline defect: net value ignored the premium paid for the hedge.
    proposal = solve()
    assert proposal.net_premium != 0.0
    expected = sum(
        leg.quantity * leg.instrument.multiplier * c.quote.mid
        for leg in proposal.legs
        for c in CANDIDATES
        if c.instrument == leg.instrument
    )
    assert_allclose(proposal.net_premium, expected)
    assert proposal.transaction_cost > 0.0
    assert proposal.transaction_cost < abs(proposal.net_premium)


def test_transaction_cost_is_spread_times_size_plus_fees():
    candidates = (stock_candidate(half=0.05, fee_per_unit=0.01),)
    targets = (GreekTarget("XYZ", GreekName.DELTA, weight=1.0),)
    p = solve(candidates=candidates, targets=targets)
    leg = p.legs[0]
    assert_allclose(p.transaction_cost, abs(leg.quantity) * 0.06)


def test_low_risk_aversion_trades_less_than_high():
    cheap = solve(config=HedgeConfig(risk_aversion=0.001))
    keen = solve(config=HedgeConfig(risk_aversion=10.0))
    # Only cost is monotone. Gross lots is not: an option can replace many cheap shares.
    assert cheap.transaction_cost <= keen.transaction_cost


def test_flat_book_needs_no_trade():
    empty = solve(book=())
    assert empty.status is ProposalStatus.NO_TRADE
    assert empty.legs == ()


def test_hard_band_is_met_and_verified():
    targets = (
        GreekTarget("XYZ", GreekName.DELTA, lower=-500.0, upper=500.0, weight=0.0),
        GreekTarget("XYZ", GreekName.VEGA, lower=-5.0, upper=5.0, weight=0.0),
    )
    p = solve(targets=targets)
    after = p.after["XYZ"]
    assert -500.0 <= after.delta_dollars <= 500.0
    assert -5.0 <= after.vega_1pt <= 5.0
    assert p.verification.passed


def test_infeasible_band_names_the_binding_constraint():
    targets = (GreekTarget("XYZ", GreekName.VEGA, lower=-1.0, upper=1.0),)
    candidates = (stock_candidate(),)  # stock has no vega, so vega cannot be hedged
    with pytest.raises(HedgeInfeasibleError) as err:
        solve(candidates=candidates, targets=targets)
    assert any("vega_1pt" in text for text in err.value.binding)


def test_premium_budget_is_enforced():
    # The unconstrained hedge sets the level, so that the budget is certain to bind.
    unconstrained = solve(candidates=CANDIDATES[1:])
    budget = unconstrained.net_premium - 300.0
    capped = solve(
        candidates=CANDIDATES[1:],
        config=HedgeConfig(risk_aversion=10.0, premium_budget=budget),
    )
    assert capped.net_premium <= budget + 1e-6


def test_a_constraint_cannot_improve_the_objective():
    # Found by this suite: at its time limit the solver returned an incumbent so poor that adding a
    # constraint appeared to improve the answer, which is impossible at an optimum.
    free = solve(candidates=CANDIDATES[1:])
    capped = solve(
        candidates=CANDIDATES[1:],
        config=HedgeConfig(risk_aversion=10.0, premium_budget=free.net_premium - 300.0),
    )
    assert capped.objective >= free.objective * (1.0 - 0.05)
    assert capped.objective_lower_bound >= free.objective_lower_bound - 1e-6


def test_objective_never_beats_its_continuous_lower_bound():
    for config in (HedgeConfig(risk_aversion=0.1), HedgeConfig(risk_aversion=10.0)):
        p = solve(candidates=CANDIDATES[1:], config=config)
        assert p.objective >= p.objective_lower_bound - 1e-6
        assert 0.0 <= p.solver_gap < 1.0


def test_integrality_cost_is_reported_when_material():
    p = solve(candidates=CANDIDATES[1:], config=HedgeConfig(risk_aversion=10.0, mip_gap=1e-6))
    assert p.solver_gap > 1e-6
    assert p.status is ProposalStatus.WITHIN_GAP
    assert any("INTEGRALITY_GAP" in issue for issue in p.issues)


def test_max_legs_is_respected():
    p = solve(config=HedgeConfig(risk_aversion=10.0, max_legs=2))
    assert len(p.legs) <= 2


def test_ticket_cost_reduces_legs():
    free = solve(config=HedgeConfig(risk_aversion=0.5))
    dear = solve(config=HedgeConfig(risk_aversion=0.5, ticket_cost=200.0))
    assert len(dear.legs) <= len(free.legs)


def test_max_buy_and_sell_caps_are_respected():
    capped = (
        replace(CANDIDATES[0], max_buy=50, max_sell=50),
        *(replace(c, max_buy=3, max_sell=3) for c in CANDIDATES[1:]),
    )
    p = solve(candidates=capped)
    for leg in p.legs:
        limit = 50 if isinstance(leg.instrument, LinearInstrument) else 3
        assert abs(leg.quantity) <= limit


def test_lot_size_is_respected():
    candidates = (replace(CANDIDATES[0], lot_size=100),)
    targets = (GreekTarget("XYZ", GreekName.DELTA, weight=1.0),)
    p = solve(candidates=candidates, targets=targets)
    assert p.legs[0].quantity % 100 == 0


def test_impact_tier_is_used_only_past_displayed_size():
    deep = stock_candidate(half=0.01, size=100, impact_per_unit=0.5)
    targets = (GreekTarget("XYZ", GreekName.DELTA, weight=1.0),)
    p = solve(candidates=(deep,), targets=targets)
    size = abs(p.legs[0].quantity)
    assert size > 100
    expected = size * 0.01 + (size - 100) * 0.5
    assert_allclose(p.transaction_cost, expected, rtol=1e-9)


def test_zero_bid_cannot_be_sold():
    worthless = HedgeCandidate(euro("call", 200, 60), Quote(0.0, 0.05, 100, 100, AS_OF))
    p = solve(candidates=(worthless, stock_candidate()))
    for leg in p.legs:
        if leg.instrument == worthless.instrument:
            assert leg.quantity > 0


@pytest.mark.parametrize(
    ("quote", "reason"),
    [
        (Quote(1.2, 1.1, 10, 10, AS_OF), "crossed"),
        (Quote(1.0, 1.1, 10, 10, AS_OF - timedelta(minutes=5)), "stale"),
        (Quote(1.0, 1.1, 0, 0, AS_OF), "no_size"),
    ],
    ids=["crossed", "stale", "no-size"],
)
def test_bad_quotes_are_excluded_with_reasons(quote, reason):
    bad = HedgeCandidate(euro("call", 105, 60), quote)
    p = solve(candidates=(bad, stock_candidate()))
    assert any(reason in item.reason for item in p.excluded)
    assert all(leg.instrument != bad.instrument for leg in p.legs)


def test_expired_candidate_excluded():
    expired = HedgeCandidate(euro("call", 105, 0), q(1.0))
    p = solve(candidates=(expired, stock_candidate()))
    assert any(item.reason == "expired" for item in p.excluded)


def test_verification_runs_on_every_proposal():
    assert solve().verification.passed


def test_cvar_objective_improves_worst_tail():
    scenarios = HISTORICAL_STRESSES
    book = (Position(euro("put", 95, 30), -50),)
    candidates = (option_candidate("put", 90, 30), option_candidate("put", 85, 60))
    targets = (GreekTarget("XYZ", GreekName.DELTA, weight=0.0),)
    unhedged = solve(
        book=book,
        candidates=candidates,
        targets=targets,
        scenarios=scenarios,
        config=HedgeConfig(risk_aversion=0.0),
    )
    hedged = solve(
        book=book,
        candidates=candidates,
        targets=targets,
        scenarios=scenarios,
        config=HedgeConfig(risk_aversion=0.0, cvar_weight=1.0, cvar_alpha=0.8),
    )
    assert min(hedged.scenario_pnl_after.values()) > min(unhedged.scenario_pnl_after.values())
    assert hedged.verification.passed


def test_proxy_hedge_reports_basis_risk():
    proxy = LinearInstrument("PROXY", "XYZ", LinearKind.PROXY, delta_ratio=1.0, basis_vol=0.05)
    candidates = (HedgeCandidate(proxy, q(100.0, 0.01, 1_000_000)),)
    targets = (GreekTarget("XYZ", GreekName.DELTA, weight=1.0),)
    p = solve(candidates=candidates, targets=targets)
    assert p.basis_risk_1d > 0.0
    assert any("BASIS_RISK" in issue for issue in p.issues)


def test_multi_underlying_targets_do_not_cross_hedge():
    from greeks_neutralizer.market import DiscountCurve, FlatVol, MarketSnapshot, UnderlyingMarket

    other = UnderlyingMarket("ABC", 50.0, DiscountCurve.flat(0.05), FlatVol(0.3))
    base = flat_snapshot()
    snapshot = MarketSnapshot(AS_OF, {"XYZ": base.market("XYZ"), "ABC": other})
    book = (Position(euro("call", 100, 30), 10),)
    abc_stock = LinearInstrument("ABC", "ABC")
    problem = HedgeProblem(
        book,
        (HedgeCandidate(abc_stock, q(50.0, 0.01, 100000)), stock_candidate()),
        (
            GreekTarget("XYZ", GreekName.DELTA, weight=1.0),
            GreekTarget("ABC", GreekName.DELTA, weight=1.0),
        ),
    )
    p = solve_hedge(problem, snapshot)
    assert {leg.instrument.symbol for leg in p.legs} == {"XYZ"}


def test_manifest_changes_with_inputs_and_replay_is_exact():
    a = solve()
    b = solve()
    assert a.quantities == b.quantities
    assert a.manifest.problem_fingerprint == b.manifest.problem_fingerprint
    c = solve(config=HedgeConfig(risk_aversion=9.0))
    assert a.manifest.problem_fingerprint != c.manifest.problem_fingerprint


def test_stability_detects_nearly_identical_hedges():
    twins = (
        option_candidate("call", 100, 60),
        option_candidate("call", 100.01, 60),
        option_candidate("call", 100.02, 60),
    )
    p = solve(candidates=twins, config=HedgeConfig(risk_aversion=10.0), run_stability=True)
    assert p.stability is not None
    if len(p.legs) > 1:
        assert p.stability.condition_number > 100.0


def test_stability_report_on_diversified_hedges():
    p = solve(run_stability=True)
    assert p.stability is not None
    assert len(p.stability.drift) == 6


def test_frontier_contains_baselines_and_cost_rises_with_aversion():
    problem = HedgeProblem(BOOK, CANDIDATES, DGV)
    points = hedge_frontier(problem, flat_snapshot(), risk_aversions=(0.01, 1.0, 10.0))
    labels = [p.label for p in points]
    assert labels[0] == "do nothing"
    assert labels[1] == "delta only"
    optimized = [p for p in points if p.risk_aversion is not None]
    costs = [p.transaction_cost for p in optimized]
    assert costs == sorted(costs)
    # The highest aversion must leave no more residual than doing nothing or delta-only.
    assert optimized[-1].penalized_residual <= points[0].penalized_residual
    assert optimized[-1].penalized_residual <= points[1].penalized_residual


def test_no_trade_band_formula_and_monotonicity():
    band = whalley_wilmott_band(
        gamma_shares=100.0, spot=100.0, proportional_cost=0.001, risk_aversion=1e-4
    )
    expected = np.cbrt(1.5 * 0.001 * 100.0 * 100.0**2 / 1e-4)
    assert_allclose(band.half_width_shares, expected)
    wider = whalley_wilmott_band(100.0, 100.0, 0.002, 1e-4)
    assert wider.half_width_shares > band.half_width_shares
    assert band.should_rebalance(band.half_width_shares + 1.0)
    assert not band.should_rebalance(band.half_width_shares - 1.0)


def test_decay_forecast_gamma_grows_atm_into_expiry():
    book = (Position(euro("call", 100, 10), 10),)
    steps = hedge_decay_forecast(book, flat_snapshot(), "XYZ", days=(1.0, 5.0, 9.0))
    assert steps[0].exposure.gamma_shares < steps[-1].exposure.gamma_shares


@given(
    qty=st.integers(-200, 200),
    strike=st.sampled_from([90.0, 100.0, 110.0]),
    days=st.sampled_from([20, 40, 80]),
    aversion=st.sampled_from([0.1, 1.0, 10.0]),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_properties_of_every_proposal(qty, strike, days, aversion):
    book = (Position(euro("call", strike, days), qty),)
    p = solve(book=book, config=HedgeConfig(risk_aversion=aversion))
    assert p.verification.passed
    assert p.transaction_cost >= 0.0
    # An optimal solution cannot leave the penalized objective above doing nothing.
    before = value_book(book, flat_snapshot()).by_underlying["XYZ"]
    do_nothing = aversion * sum(
        t.effective_weight * abs(getattr(before, t.greek.value)) for t in DGV
    )
    assert p.objective <= do_nothing * (1 + 1e-3) + 1e-6


def test_empty_problem_rejected():
    from greeks_neutralizer.errors import ValidationError

    with pytest.raises(ValidationError):
        HedgeProblem(BOOK, CANDIDATES, ())
    with pytest.raises(ValidationError):
        HedgeProblem(BOOK, CANDIDATES, DGV, (), HedgeConfig(cvar_weight=1.0))


def test_taylor_check_on_small_shock_is_accurate():
    from greeks_neutralizer.risk.scenarios import taylor_vs_full

    small = taylor_vs_full(BOOK, flat_snapshot(), (Scenario("small", spot_shock=0.001),))
    assert abs(small[0].error) < 1.0
    large = taylor_vs_full(BOOK, flat_snapshot(), (Scenario("crash", spot_shock=-0.2),))
    assert abs(large[0].error) > abs(small[0].error)
