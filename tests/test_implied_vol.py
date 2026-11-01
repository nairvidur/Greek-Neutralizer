"""Implied vol: dense-grid round trips, statuses for unattainable prices, QuantLib agreement."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from greeks_neutralizer.errors import ValidationError
from greeks_neutralizer.pricing.black import black_premium
from greeks_neutralizer.pricing.implied_vol import IVStatus, implied_vol


def _premium(F, K, df, vol, tau, om):
    s = np.asarray(vol * np.sqrt(tau), dtype=float).reshape(-1)
    return black_premium(
        np.asarray([F], dtype=float).reshape(-1),
        np.asarray([K], dtype=float).reshape(-1),
        np.asarray([df], dtype=float).reshape(-1),
        s,
        np.asarray([om], dtype=float).reshape(-1),
    )


def test_round_trip_dense_grid_within_bar():
    F = 100.0
    strikes = F * np.exp(np.linspace(-1.5, 1.5, 61))
    vols = np.array([0.03, 0.1, 0.2, 0.4, 0.8, 1.6, 3.0])
    taus = np.array([1 / 365 / 24, 1 / 365, 0.02, 0.25, 1.0, 5.0])
    K, V, T, W = np.meshgrid(strikes, vols, taus, [1.0, -1.0], indexing="ij")
    K, V, T, W = K.ravel(), V.ravel(), T.ravel(), W.ravel()
    forward = np.full_like(K, F)
    df = np.full_like(K, 0.97)
    price = black_premium(forward, K, df, V * np.sqrt(T), W)
    result = implied_vol(price, forward, K, df, T, W)

    # Vol recovery is only well-posed where time value is resolvable at double precision.
    intrinsic = np.maximum(W * (forward - K), 0.0)
    time_value = price / df - intrinsic
    assert np.all(result.status[time_value > 1e-11 * F] == IVStatus.OK)
    resolvable = time_value > 1e-9 * F
    assert resolvable.sum() > 1000
    assert_allclose(result.vol[resolvable], V[resolvable], rtol=1e-7, atol=0)

    # Wherever a vol is returned, repricing it must reproduce the input premium.
    ok = result.is_ok
    repriced = black_premium(forward[ok], K[ok], df[ok], result.vol[ok] * np.sqrt(T[ok]), W[ok])
    assert_allclose(repriced, price[ok], rtol=1e-11, atol=1e-11)


def test_within_bar_where_vega_is_meaningful():
    K = np.array([80.0, 90.0, 100.0, 110.0, 125.0])
    forward = np.full_like(K, 100.0)
    tau = np.full_like(K, 0.25)
    vol = np.array([0.35, 0.28, 0.25, 0.23, 0.27])
    om = np.ones_like(K)
    df = np.ones_like(K)
    price = black_premium(forward, K, df, vol * np.sqrt(tau), om)
    result = implied_vol(price, forward, K, df, tau, om)
    assert_allclose(result.vol, vol, atol=1e-10, rtol=0)


@given(st.floats(0.5, 2.0), st.floats(0.02, 2.5), st.floats(0.003, 4.0), st.sampled_from([1, -1]))
@settings(max_examples=400, deadline=None)
def test_round_trip_property(moneyness, vol, tau, om):
    F, K, df = 100.0, 100.0 * moneyness, 0.98
    price = _premium(F, K, df, vol, tau, om)[0]
    intrinsic = max(om * (F - K), 0.0)
    assume(price / df - intrinsic >= 1e-9)
    result = implied_vol(price, F, K, df, tau, om)
    assert result.status[0] == IVStatus.OK
    assert_allclose(result.vol[0], vol, rtol=1e-6)


def test_below_intrinsic_is_reported_not_solved():
    result = implied_vol(5.0, 120.0, 100.0, 1.0, 0.25, 1)
    assert result.status[0] == IVStatus.BELOW_INTRINSIC
    assert np.isnan(result.vol[0])


def test_above_maximum_is_reported():
    result = implied_vol(150.0, 100.0, 100.0, 1.0, 0.25, 1)
    assert result.status[0] == IVStatus.ABOVE_MAXIMUM
    assert np.isnan(result.vol[0])


def test_at_intrinsic_is_reported():
    result = implied_vol(20.0, 120.0, 100.0, 1.0, 0.25, 1)
    assert result.status[0] == IVStatus.AT_INTRINSIC


def test_nan_price_rejected():
    with pytest.raises(ValidationError, match="NaN or inf"):
        implied_vol(float("nan"), 100.0, 100.0, 1.0, 0.25, 1)


@pytest.mark.reference
def test_agrees_with_quantlib_black_formula_implied():
    ql = pytest.importorskip("QuantLib")
    for F, K, s, om in [(100.0, 100.0, 0.1, 1), (100.0, 130.0, 0.3, 1), (100.0, 80.0, 0.25, -1)]:
        ql_type = ql.Option.Call if om == 1 else ql.Option.Put
        price = ql.blackFormula(ql_type, K, F, s, 1.0)
        ql_s = ql.blackFormulaImpliedStdDev(ql_type, K, F, price, 1.0)
        ours = implied_vol(price, F, K, 1.0, 1.0, om)
        assert_allclose(ours.vol[0], s, rtol=1e-10)
        assert_allclose(ours.vol[0], ql_s, rtol=1e-6)
