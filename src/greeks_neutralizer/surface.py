"""Arbitrage-checked implied volatility surfaces built from raw SVI slices.

A slice gives total variance w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)), with k the
log-moneyness ln(K/F). Slices are combined linearly in total variance at fixed k, which keeps the
surface free of calendar arbitrage whenever the slices themselves do not cross (Gatheral-Jacquier).

Calibration penalizes Durrleman's butterfly condition g(k) >= 0 and crossing the previous slice.
Remaining violations are measured and returned, so a fit that had to compromise cannot pass
silently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.optimize import least_squares

from greeks_neutralizer.errors import MarketDataError, ValidationError
from greeks_neutralizer.pricing.black import FloatArray

_CHECK_GRID = np.linspace(-1.5, 1.5, 241)


@dataclass(frozen=True, slots=True)
class SviSlice:
    tau: float
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        values = (self.tau, self.a, self.b, self.rho, self.m, self.sigma)
        if not all(np.isfinite(v) for v in values):
            raise ValidationError("SVI parameters must be finite")
        if self.tau <= 0.0 or self.b < 0.0 or self.sigma <= 0.0 or abs(self.rho) >= 1.0:
            raise ValidationError(f"SVI parameters out of domain: {self}")
        if self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2) < 0.0:
            raise ValidationError("SVI minimum total variance is negative")

    def total_variance(self, k: FloatArray) -> FloatArray:
        x = np.asarray(k, dtype=np.float64) - self.m
        return np.asarray(
            self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma * self.sigma)),
            dtype=np.float64,
        )

    def durrleman_g(self, k: FloatArray) -> FloatArray:
        """Risk-neutral density is non-negative only where this is >= 0."""
        x = np.asarray(k, dtype=np.float64) - self.m
        root = np.sqrt(x * x + self.sigma * self.sigma)
        w = self.a + self.b * (self.rho * x + root)
        w1 = self.b * (self.rho + x / root)
        w2 = self.b * self.sigma * self.sigma / root**3
        w_safe = np.maximum(w, 1e-12)
        kw = np.asarray(k, dtype=np.float64) * w1 / (2.0 * w_safe)
        return np.asarray(
            (1.0 - kw) ** 2 - 0.25 * w1 * w1 * (0.25 + 1.0 / w_safe) + 0.5 * w2, dtype=np.float64
        )


@dataclass(frozen=True, slots=True)
class ArbitrageReport:
    butterfly_min_g: float
    calendar_min_gap: float

    @property
    def is_clean(self) -> bool:
        return self.butterfly_min_g >= -1e-9 and self.calendar_min_gap >= -1e-9


@dataclass(frozen=True, slots=True)
class SviSurface:
    """Surface over variance-time. Before the first slice, total variance scales linearly with tau
    (constant vol per k); beyond the last, it grows at the last slice's implied rate."""

    slices: tuple[SviSlice, ...]

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValidationError("a surface needs at least one slice")
        taus = [s.tau for s in self.slices]
        if any(b <= a for a, b in pairwise(taus)):
            raise ValidationError("slices must be strictly increasing in tau")

    def total_variance(self, k: FloatArray, tau: FloatArray) -> FloatArray:
        k_, tau_ = np.broadcast_arrays(
            np.asarray(k, dtype=np.float64), np.asarray(tau, dtype=np.float64)
        )
        k_f, tau_f = k_.ravel(), tau_.ravel()
        taus = np.array([s.tau for s in self.slices])
        result = np.zeros_like(k_f, dtype=np.float64)
        first = tau_f <= taus[0]
        if np.any(first):
            s0 = self.slices[0]
            result[first] = s0.total_variance(k_f[first]) * tau_f[first] / s0.tau
        last = tau_f >= taus[-1]
        if np.any(last):
            sl = self.slices[-1]
            result[last] = sl.total_variance(k_f[last]) * tau_f[last] / sl.tau
        middle = ~(first | last)
        if np.any(middle):
            upper = np.searchsorted(taus, tau_f[middle])
            for hi in np.unique(upper):
                mask = np.flatnonzero(middle)[upper == hi]
                s_lo, s_hi = self.slices[hi - 1], self.slices[hi]
                weight = (tau_f[mask] - s_lo.tau) / (s_hi.tau - s_lo.tau)
                result[mask] = (1.0 - weight) * s_lo.total_variance(
                    k_f[mask]
                ) + weight * s_hi.total_variance(k_f[mask])
        return result.reshape(k_.shape)

    def vol(self, k: FloatArray, tau: FloatArray) -> FloatArray:
        w = self.total_variance(k, tau)
        tau_arr = np.asarray(tau, dtype=np.float64)
        if np.any(tau_arr <= 0.0):
            raise MarketDataError("vol requested at non-positive tau")
        if np.any(w < 0.0):
            raise MarketDataError("surface produced negative total variance")
        return np.asarray(np.sqrt(w / tau_arr), dtype=np.float64)

    def arbitrage_report(self) -> ArbitrageReport:
        g_min = min(float(np.min(s.durrleman_g(_CHECK_GRID))) for s in self.slices)
        gap = float("inf")
        for lo, hi in pairwise(self.slices):
            gap = min(
                gap, float(np.min(hi.total_variance(_CHECK_GRID) - lo.total_variance(_CHECK_GRID)))
            )
        return ArbitrageReport(butterfly_min_g=g_min, calendar_min_gap=gap)

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "kind": "svi",
            "slices": [(s.tau, s.a, s.b, s.rho, s.m, s.sigma) for s in self.slices],
        }


@dataclass(frozen=True, slots=True)
class SliceFit:
    slice: SviSlice
    rmse_vol: float
    max_abs_error_vol: float
    n_points: int
    butterfly_min_g: float


_LOWER = np.array([-1.0, 1e-5, -0.999, -2.0, 1e-4])
_UPPER = np.array([5.0, 5.0, 0.999, 2.0, 5.0])


def fit_svi_slice(
    k: FloatArray,
    vol: FloatArray,
    tau: float,
    weights: FloatArray | None = None,
    *,
    floor_slice: SviSlice | None = None,
    min_points: int = 5,
) -> SliceFit:
    """Fit raw SVI to vols at log-moneyness k by weighted least squares in vol, with penalties for
    butterfly arbitrage and for crossing below `floor_slice`. Multi-start for robustness; every
    start is deterministic, so the same quotes always give the same fit."""
    k_ = np.asarray(k, dtype=np.float64)
    v_ = np.asarray(vol, dtype=np.float64)
    if k_.size != v_.size:
        raise ValidationError("k and vol must have equal length")
    finite = np.isfinite(k_) & np.isfinite(v_) & (v_ > 0.0)
    k_, v_ = k_[finite], v_[finite]
    if k_.size < min_points:
        raise MarketDataError(
            f"cannot fit SVI at tau={tau:.5f}: {k_.size} usable points < {min_points}"
        )
    w_ = np.ones_like(k_) if weights is None else np.asarray(weights, dtype=np.float64)[finite]
    w_ = np.sqrt(np.maximum(w_, 0.0) / max(float(np.max(w_)), 1e-300))

    grid = np.linspace(min(float(np.min(k_)), -0.5) * 1.5, max(float(np.max(k_)), 0.5) * 1.5, 61)
    floor = None if floor_slice is None else floor_slice.total_variance(grid)
    weight_penalty = 10.0

    def residuals(p: FloatArray) -> FloatArray:
        s = _unchecked_slice(tau, p)
        total = np.maximum(s.total_variance(k_), 1e-12)
        fit_error = (np.sqrt(total / tau) - v_) * w_
        g = s.durrleman_g(grid)
        parts = [fit_error, weight_penalty * np.minimum(g, 0.0)]
        if floor is not None:
            parts.append(weight_penalty * np.minimum(s.total_variance(grid) - floor, 0.0))
        parts.append(weight_penalty * np.minimum(p[0] + p[1] * p[4] * np.sqrt(1 - p[2] ** 2), 0.0))
        return np.concatenate([np.atleast_1d(x) for x in parts])

    atm_var = float(np.interp(0.0, np.sort(k_), v_[np.argsort(k_)]) ** 2 * tau)
    best = None
    for rho0 in (-0.7, -0.3, 0.0, 0.3):
        for sig0 in (0.05, 0.2, 0.5):
            start = np.array([0.5 * atm_var, 0.3 * max(np.sqrt(atm_var), 0.02), rho0, 0.0, sig0])
            start = np.clip(start, _LOWER + 1e-9, _UPPER - 1e-9)
            sol = least_squares(residuals, start, bounds=(_LOWER, _UPPER), max_nfev=200)
            if best is None or sol.cost < best.cost:
                best = sol
    assert best is not None
    slice_ = _feasible_slice(tau, best.x)
    errors = np.sqrt(np.maximum(slice_.total_variance(k_), 0.0) / tau) - v_
    return SliceFit(
        slice=slice_,
        rmse_vol=float(np.sqrt(np.mean(errors**2))),
        max_abs_error_vol=float(np.max(np.abs(errors))),
        n_points=int(k_.size),
        butterfly_min_g=float(np.min(slice_.durrleman_g(_CHECK_GRID))),
    )


def _unchecked_slice(tau: float, p: FloatArray) -> SviSlice:
    """Build a slice during optimization, where a may transiently violate the variance floor."""
    obj = object.__new__(SviSlice)
    for name, value in zip(("tau", "a", "b", "rho", "m", "sigma"), (tau, *p), strict=True):
        object.__setattr__(obj, name, float(value))
    return obj


def _feasible_slice(tau: float, p: FloatArray) -> SviSlice:
    a, b, rho, m, sigma = (float(x) for x in p)
    floor = -b * sigma * np.sqrt(1.0 - rho * rho)
    return SviSlice(tau, max(a, floor), b, rho, m, sigma)


def fit_surface(
    quotes: Sequence[tuple[float, FloatArray, FloatArray, FloatArray | None]],
) -> tuple[SviSurface, tuple[SliceFit, ...]]:
    """Fit an arbitrage-penalized surface from (tau, k, vol, weights) per expiry, shortest first."""
    ordered = sorted(quotes, key=lambda q: q[0])
    fits: list[SliceFit] = []
    floor: SviSlice | None = None
    for tau, k, vol, weights in ordered:
        fit = fit_svi_slice(k, vol, tau, weights, floor_slice=floor)
        fits.append(fit)
        floor = fit.slice
    return SviSurface(tuple(f.slice for f in fits)), tuple(fits)
