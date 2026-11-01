"""SVI fits, arbitrage detection, and interpolation behavior."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from greeks_neutralizer.errors import MarketDataError, ValidationError
from greeks_neutralizer.surface import (
    SviSlice,
    SviSurface,
    check_grid,
    fit_surface,
    fit_svi_slice,
)

TRUE = SviSlice(0.25, 0.005, 0.06, -0.6, 0.02, 0.15)


def test_recovers_known_slice_within_a_hundredth_of_a_vol_point():
    k = np.linspace(-0.5, 0.35, 25)
    vol = np.sqrt(TRUE.total_variance(k) / TRUE.tau)
    fit = fit_svi_slice(k, vol, TRUE.tau)
    assert fit.rmse_vol < 1e-4
    assert fit.max_abs_error_vol < 3e-4
    assert fit.butterfly_min_g >= -1e-6


def test_fit_is_deterministic():
    k = np.linspace(-0.4, 0.3, 15)
    rng = np.random.default_rng(7)
    vol = np.sqrt(TRUE.total_variance(k) / TRUE.tau) + rng.normal(0, 0.002, k.size)
    assert fit_svi_slice(k, vol, TRUE.tau).slice == fit_svi_slice(k, vol, TRUE.tau).slice


def test_noisy_quotes_do_not_produce_butterfly_arbitrage():
    k = np.linspace(-0.6, 0.4, 21)
    rng = np.random.default_rng(11)
    vol = np.sqrt(TRUE.total_variance(k) / TRUE.tau) + rng.normal(0, 0.01, k.size)
    fit = fit_svi_slice(k, vol, TRUE.tau)
    assert fit.butterfly_min_g > -1e-3


def test_known_butterfly_arbitrage_is_detected():
    # The Axel Vogt slice from Gatheral-Jacquier is the standard arbitrageable SVI example.
    vogt = SviSlice(1.0, -0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
    report = SviSurface((vogt,)).arbitrage_report()
    assert report.wing_min_g < 0.0
    assert not report.wings_clean


def test_core_grid_is_scaled_from_at_the_money_variance():
    # With m far from zero, variance at m is much larger than at the money. Sizing the grid from it
    # flagged unquoted strikes as core arbitrage on real one-week SPX slices.
    skewed = SviSlice(0.0137, -0.00238, 0.16692, -0.963, -0.1528, 0.05789)
    atm_stdev = float(np.sqrt(skewed.total_variance(np.array([0.0]))[0]))
    assert_allclose(check_grid(skewed, 3.0).max(), 3.0 * atm_stdev)


def test_weighted_error_ignores_low_weight_wing():
    k = np.linspace(-0.2, 0.2, 21)
    vol = np.full_like(k, 0.2)
    vol[-1] = 0.5
    weights = np.ones_like(k)
    weights[-1] = 1e-6
    fit = fit_svi_slice(k, vol, 0.25, weights)
    assert fit.weighted_rmse_vol < fit.rmse_vol
    assert fit.weighted_rmse_vol < 0.002


def test_crossing_slices_are_detected():
    low = SviSlice(0.5, 0.02, 0.05, -0.5, 0.0, 0.2)
    crossing = SviSlice(1.0, 0.01, 0.05, -0.5, 0.0, 0.2)
    report = SviSurface((low, crossing)).arbitrage_report()
    assert report.core_calendar_gap < 0.0
    assert not report.is_clean


def test_fit_surface_keeps_calendar_order():
    quotes = []
    for tau, level in ((0.1, 0.30), (0.25, 0.27), (0.5, 0.25), (1.0, 0.24)):
        k = np.linspace(-0.5, 0.4, 19)
        vol = level * (1.0 - 0.4 * k + 0.8 * k * k)
        quotes.append((tau, k, vol, None))
    surface, fits = fit_surface(quotes)
    report = surface.arbitrage_report()
    assert report.is_clean or report.core_calendar_gap > -1e-4
    assert all(f.rmse_vol < 0.01 for f in fits)


def test_interpolation_hits_slices_and_is_monotone_in_total_variance():
    a = SviSlice(0.25, 0.01, 0.05, -0.5, 0.0, 0.2)
    b = SviSlice(1.0, 0.05, 0.08, -0.4, 0.0, 0.3)
    surface = SviSurface((a, b))
    k = np.array([-0.2, 0.0, 0.2])
    assert_allclose(surface.total_variance(k, np.full(3, 0.25)), a.total_variance(k))
    assert_allclose(surface.total_variance(k, np.full(3, 1.0)), b.total_variance(k))
    taus = np.linspace(0.02, 2.0, 40)
    w = surface.total_variance(np.zeros_like(taus), taus)
    assert np.all(np.diff(w) > 0.0)


def test_short_extrapolation_holds_vol_constant():
    surface = SviSurface((TRUE,))
    k = np.array([0.0])
    assert_allclose(surface.vol(k, np.array([0.001])), surface.vol(k, np.array([TRUE.tau])))


def test_too_few_points_fails_closed():
    with pytest.raises(MarketDataError, match="usable points"):
        fit_svi_slice(np.array([0.0, 0.1]), np.array([0.2, 0.2]), 0.25)


def test_nan_quotes_are_excluded_not_fitted():
    k = np.linspace(-0.4, 0.3, 15)
    vol = np.sqrt(TRUE.total_variance(k) / TRUE.tau)
    vol[3] = np.nan
    fit = fit_svi_slice(k, vol, TRUE.tau)
    assert fit.n_points == 14


def test_invalid_slice_rejected():
    with pytest.raises(ValidationError):
        SviSlice(0.25, 0.01, 0.05, -1.0, 0.0, 0.2)
    with pytest.raises(ValidationError):
        SviSlice(0.25, -1.0, 0.05, 0.0, 0.0, 0.2)
