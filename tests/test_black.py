"""Black kernel: analytic Greeks against finite differences, parity, bounds, limits, QuantLib."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from greeks_neutralizer.errors import ValidationError
from greeks_neutralizer.pricing.black import black_greeks, normal_cdf

GRID = [
    # S, K, T, r, q, vol, omega, tau multiplier
    (100.0, 100.0, 0.25, 0.05, 0.0, 0.30, 1, 1.0),
    (100.0, 100.0, 0.25, 0.05, 0.0, 0.30, -1, 1.0),
    (100.0, 80.0, 1.0, 0.03, 0.02, 0.20, 1, 1.0),
    (100.0, 130.0, 0.5, 0.01, 0.04, 0.45, -1, 1.0),
    (4000.0, 3900.0, 2.0 / 365, 0.04, 0.012, 0.18, 1, 0.8),
    (50.0, 55.0, 3.0, -0.005, 0.0, 0.60, 1, 1.0),
]


def _price(S, K, T, r, q, v, om, kappa):
    return black_greeks(S, K, T, r, q, v, om, tau=T * kappa).price[0]


def _d_dt(f, T):
    # Sensitivity to calendar time is minus the derivative with respect to time to expiry.
    h = T * 1e-4
    return -(f(T + h) - f(T - h)) / (2 * h)


@pytest.mark.parametrize("case", GRID, ids=[str(i) for i in range(len(GRID))])
def test_all_greeks_match_finite_differences(case):
    S, K, T, r, q, v, om, kappa = case
    g = black_greeks(S, K, T, r, q, v, om, tau=T * kappa)
    # Scale the spot bump to the option's own stdev so truncation stays O(h^2) when short-dated.
    hs = S * min(2e-4, 0.02 * v * np.sqrt(T * kappa))
    hv = 1e-4
    hr = 1e-5

    def sg(name, s=S, vol=v, t=T):
        return getattr(black_greeks(s, K, t, r, q, vol, om, tau=t * kappa), name)[0]

    checks = {
        "delta": (sg("price", s=S + hs) - sg("price", s=S - hs)) / (2 * hs),
        "gamma": (sg("price", s=S + hs) - 2 * sg("price") + sg("price", s=S - hs)) / hs**2,
        "vega": (sg("price", vol=v + hv) - sg("price", vol=v - hv)) / (2 * hv),
        "theta": _d_dt(lambda t: sg("price", t=t), T),
        "rho": (_price(S, K, T, r + hr, q, v, om, kappa) - _price(S, K, T, r - hr, q, v, om, kappa))
        / (2 * hr),
        "phi": (_price(S, K, T, r, q + hr, v, om, kappa) - _price(S, K, T, r, q - hr, v, om, kappa))
        / (2 * hr),
        "vanna": (sg("vega", s=S + hs) - sg("vega", s=S - hs)) / (2 * hs),
        "volga": (sg("vega", vol=v + hv) - sg("vega", vol=v - hv)) / (2 * hv),
        "charm": _d_dt(lambda t: sg("delta", t=t), T),
        "veta": _d_dt(lambda t: sg("vega", t=t), T),
        "speed": (sg("gamma", s=S + hs) - sg("gamma", s=S - hs)) / (2 * hs),
        "zomma": (sg("gamma", vol=v + hv) - sg("gamma", vol=v - hv)) / (2 * hv),
        "color": _d_dt(lambda t: sg("gamma", t=t), T),
    }
    for name, numeric in checks.items():
        analytic = getattr(g, name)[0]
        assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-6, err_msg=name)


@given(
    st.floats(0.5, 2000),
    st.floats(0.7, 1.4),
    st.floats(1e-3, 5),
    st.floats(-0.02, 0.15),
    st.floats(0, 0.1),
    st.floats(0.01, 2),
)
@settings(max_examples=300, deadline=None)
def test_parity_and_bounds(S, moneyness, T, r, q, v):
    K = S * moneyness
    call = black_greeks(S, K, T, r, q, v, 1)
    put = black_greeks(S, K, T, r, q, v, -1)
    parity = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert_allclose(call.price[0] - put.price[0], parity, rtol=1e-10, atol=1e-10 * S)
    assert call.price[0] >= 0.0
    assert put.price[0] >= 0.0
    assert_allclose(call.gamma[0], put.gamma[0], rtol=1e-12, atol=1e-300)
    assert call.price[0] <= S * np.exp(-q * T) * (1 + 1e-12)
    assert put.price[0] <= K * np.exp(-r * T) * (1 + 1e-12)


@pytest.mark.parametrize(
    ("K", "T", "vol", "reference", "rtol"),
    [
        # Reference: S N(d1) - K N(d2) at 80 digits with mpmath, r = q = 0, S = 100. At these
        # tolerances, that difference in double precision fails: it is off by 5.6e-12 and 1.5e-9.
        (200.0, 0.05, 0.2, 7.1013615483982070581e-55, 1e-12),
        (110.0, 0.01, 0.03, 8.293222808927537721e-224, 1e-11),
    ],
)
def test_far_wing_premium_keeps_relative_accuracy(K, T, vol, reference, rtol):
    assert_allclose(black_greeks(100.0, K, T, 0.0, 0.0, vol, 1).price[0], reference, rtol=rtol)


def test_zero_vol_is_discounted_forward_intrinsic():
    g = black_greeks(120.0, 100.0, 1.0, 0.05, 0.0, 0.0, 1)
    assert_allclose(g.price[0], 120.0 - 100.0 * np.exp(-0.05), rtol=1e-14)
    assert g.delta[0] == 1.0
    assert g.gamma[0] == 0.0
    assert g.vega[0] == 0.0


def test_expired_is_intrinsic():
    g = black_greeks(120.0, 100.0, 0.0, 0.05, 0.0, 0.25, 1)
    assert g.price[0] == 20.0


def test_cdf_matches_erfc_to_machine_precision():
    from math import erfc, sqrt

    for x in (-30.0, -8.0, -1.0, 0.06, 1.0, 8.0):
        assert_allclose(float(normal_cdf(x)), 0.5 * erfc(-x / sqrt(2)), rtol=1e-12)


@pytest.mark.parametrize(
    "bad",
    [
        {"sigma": float("nan")},
        {"spot": float("inf")},
        {"spot": -1.0},
        {"strike": 0.0},
        {"T": -0.1},
        {"sigma": -0.2},
        {"omega": 2.0},
    ],
    ids=lambda d: next(iter(d)),
)
def test_rejects_invalid_inputs(bad):
    args = {"spot": 100.0, "strike": 100.0, "T": 1.0, "r": 0.0, "q": 0.0, "sigma": 0.2, "omega": 1}
    args.update(bad)
    with pytest.raises(ValidationError, match=r"omega|sigma|spot|time to expiry"):
        black_greeks(**args)


@pytest.mark.reference
def test_matches_quantlib_analytic_engine():
    ql = pytest.importorskip("QuantLib")
    today = ql.Date(5, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    worst = 0.0
    for S, K, days, r, q, v, om in [
        (100, 100, 91, 0.05, 0.0, 0.3, 1),
        (100, 120, 365, 0.03, 0.02, 0.25, -1),
        (4000, 3900, 7, 0.04, 0.012, 0.18, 1),
        (50, 30, 730, 0.01, 0.0, 0.8, -1),
    ]:
        T = days / 365.0
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(S)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), v, dc)),
        )
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call if om == 1 else ql.Option.Put, K),
            ql.EuropeanExercise(today + days),
        )
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        g = black_greeks(S, K, T, r, q, v, om)
        for name, ours, theirs in [
            ("npv", g.price[0], option.NPV()),
            ("delta", g.delta[0], option.delta()),
            ("gamma", g.gamma[0], option.gamma()),
            ("vega", g.vega[0], option.vega()),
            ("theta", g.theta[0], option.theta()),
            ("rho", g.rho[0], option.rho()),
        ]:
            assert_allclose(ours, theirs, rtol=1e-10, err_msg=name)
            worst = max(worst, abs(ours - theirs) / max(abs(theirs), 1e-12))
    assert worst < 1e-10
