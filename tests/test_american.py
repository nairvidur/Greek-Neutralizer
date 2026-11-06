"""American engine: reconciled to QuantLib, with exercise bounds and dividend behavior."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from greeks_neutralizer.pricing.american import (
    AmericanBatch,
    american_greeks,
    american_price,
    exercise_is_ever_optimal,
)
from greeks_neutralizer.pricing.black import black_greeks


def batch(S, K, T, r, q, v, om, divs=()):
    n = np.size(K)
    S_, K_, T_, r_, q_, v_, om_ = (
        np.broadcast_to(np.asarray(x, dtype=float), (n,)).copy() for x in (S, K, T, r, q, v, om)
    )
    m = len(divs)
    times = np.full((n, m), np.inf)
    amounts = np.zeros((n, m))
    for col, (t, a) in enumerate(divs):
        times[:, col] = np.where(t <= T_, t, np.inf)
        amounts[:, col] = np.where(t <= T_, a, 0.0)
    return AmericanBatch(S_, K_, T_, T_.copy(), r_, q_, v_, om_, times, amounts)


CASES = [
    # S, K, T, r, q, vol, omega
    (100.0, 100.0, 0.25, 0.05, 0.0, 0.30, -1),
    (100.0, 110.0, 1.0, 0.05, 0.0, 0.25, -1),
    (100.0, 80.0, 2.0, 0.08, 0.0, 0.40, -1),
    (100.0, 90.0, 0.5, 0.03, 0.06, 0.30, 1),
    (100.0, 120.0, 1.0, 0.02, 0.03, 0.50, 1),
    (50.0, 45.0, 0.05, 0.04, 0.0, 0.60, -1),
    (400.0, 395.0, 7 / 365, 0.045, 0.012, 0.18, -1),
]


def _ql_reference(S, K, days, r, q, v, om, divs=()):
    """QuantLib reference. No dividends: its high-precision ALO engine. Dividends: finite
    differences under the Escrowed model, the same assumption this engine uses. QuantLib's default
    Spot model prices these about 0.12 higher, which is a modeling choice and not an error."""
    ql = pytest.importorskip("QuantLib")
    today = ql.Date(5, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), v, dc)),
    )
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if om == 1 else ql.Option.Put, K)
    option = ql.VanillaOption(payoff, ql.AmericanExercise(today, today + days))
    if divs:
        schedule = ql.DividendVector([today + d for d, _ in divs], [a for _, a in divs])
        engine = ql.FdBlackScholesVanillaEngine(
            process,
            schedule,
            2000,
            2000,
            0,
            ql.FdmSchemeDesc.Douglas(),
            False,
            -ql.nullDouble(),
            ql.FdBlackScholesVanillaEngine.Escrowed,
        )
    else:
        engine = ql.QdFpAmericanEngine(process, ql.QdFpAmericanEngine.highPrecisionScheme())
    option.setPricingEngine(engine)
    return option.NPV()


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES, ids=[str(i) for i in range(len(CASES))])
def test_price_reconciles_to_quantlib(case):
    S, K, T, r, q, v, om = case
    days = round(T * 365)
    reference = _ql_reference(S, K, days, r, q, v, om)
    ours = american_price(batch(S, K, days / 365, r, q, v, om))[0]
    assert_allclose(ours, reference, atol=1e-3 * max(1.0, S / 100), rtol=0)


@pytest.mark.reference
def test_discrete_dividend_put_and_call_reconcile_to_quantlib():
    days = 182
    T = days / 365
    divs_days = [(60, 1.5), (150, 1.5)]
    divs = [(d / 365, a) for d, a in divs_days]
    for om, K in [(1, 95.0), (1, 105.0), (-1, 100.0), (-1, 110.0)]:
        reference = _ql_reference(100.0, K, days, 0.04, 0.0, 0.25, om, divs_days)
        ours = american_price(batch(100.0, K, T, 0.04, 0.0, 0.25, om, divs))[0]
        # A quarter of a cent: tree steps do not land exactly on ex-dates.
        assert_allclose(ours, reference, atol=0.0025, rtol=0, err_msg=f"{om} {K}")


@pytest.mark.reference
def test_greeks_reconcile_to_quantlib_finite_differences():
    ql = pytest.importorskip("QuantLib")
    today = ql.Date(5, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    S, K, days, r, q, v, om = 100.0, 105.0, 182, 0.05, 0.01, 0.3, -1
    spot = ql.SimpleQuote(S)
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(spot),
        ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), v, dc)),
    )
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Put, K),
        ql.AmericanExercise(today, today + days),
    )
    option.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, 1600, 1600))
    g = american_greeks(batch(S, K, days / 365, r, q, v, om))
    assert_allclose(g.price[0], option.NPV(), atol=2e-3)
    assert_allclose(g.delta[0], option.delta(), atol=2e-3)
    assert_allclose(g.gamma[0], option.gamma(), atol=2e-3)


def test_exercise_floor_and_european_lower_bound():
    Ks = np.linspace(60, 160, 26)
    for om in (1, -1):
        am = american_price(batch(100.0, Ks, 0.5, 0.05, 0.03, 0.3, om))
        eu = black_greeks(100.0, Ks, 0.5, 0.05, 0.03, 0.3, om).price
        assert np.all(am >= eu - 1e-12)
        assert np.all(am >= om * (100.0 - Ks) - 1e-12)


def test_call_with_no_carry_has_no_exercise_premium():
    am = american_price(batch(100.0, 100.0, 1.0, 0.05, 0.0, 0.3, 1))[0]
    eu = black_greeks(100.0, 100.0, 1.0, 0.05, 0.0, 0.3, 1).price[0]
    assert_allclose(am, eu, atol=1e-6)


def test_dividend_makes_call_exercise_valuable_before_ex_date():
    # A large dividend just before expiry on a deep ITM call: exercising first captures it.
    with_div = american_greeks(batch(100.0, 70.0, 0.25, 0.01, 0.0, 0.2, 1, [(0.2, 5.0)]))
    assert with_div.exercise_premium[0] > 0.5


def test_greeks_are_sane_in_expected_direction():
    g = american_greeks(batch(100.0, 100.0, 0.5, 0.05, 0.0, 0.25, -1))
    assert -1.0 < g.delta[0] < 0.0
    assert g.gamma[0] > 0.0
    assert g.vega[0] > 0.0
    assert g.theta[0] < 0.0
    assert g.rho[0] < 0.0


def test_even_steps_rejected():
    with pytest.raises(ValueError, match="odd"):
        american_price(batch(100.0, 100.0, 0.5, 0.05, 0.0, 0.25, -1), steps=200)


def test_never_optimal_routing_rule():
    mask = exercise_is_ever_optimal(
        np.array([1.0, 1.0, 1.0, -1.0, -1.0]),
        np.array([0.05, 0.05, 0.05, 0.05, -0.01]),
        np.array([0.0, 0.02, 0.0, 0.0, 0.0]),
        np.array([False, False, True, False, False]),
    )
    assert mask.tolist() == [False, True, True, True, False]


@pytest.mark.parametrize("rate", [0.0, -0.01])
def test_dividend_after_expiry_at_non_positive_rates_is_silent_and_ignored(rate):
    """A dividend past expiry is padded as an infinite time. At r <= 0 that made -r * inf
    non-finite, and the repo treats the resulting RuntimeWarning as an error. Oracle: a dividend
    after expiry cannot change an option's value, so the price must equal the no-dividend price."""
    late = batch(100.0, [95.0, 105.0], 0.25, rate, 0.0, 0.3, [1, -1], divs=[(0.1, 0.5), (0.5, 1.0)])
    early_only = batch(100.0, [95.0, 105.0], 0.25, rate, 0.0, 0.3, [1, -1], divs=[(0.1, 0.5)])
    assert np.isinf(late.div_times[:, 1]).all()
    assert_allclose(american_price(late), american_price(early_only), rtol=0, atol=1e-12)
