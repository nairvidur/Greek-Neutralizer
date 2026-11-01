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
from numpy.typing import NDArray
from scipy.optimize import least_squares

from greeks_neutralizer.errors import MarketDataError, ValidationError
from greeks_neutralizer.pricing.black import FloatArray

_CORE_STDEVS = 3.0
_WING_STDEVS = 6.0
_CALENDAR_TOLERANCE = 0.01


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


def check_grid(s: SviSlice, stdevs: float = _WING_STDEVS) -> FloatArray:
    """Log-moneyness grid scaled to a slice's own stdev, capped at |k| = 1.5.

    A fixed log-moneyness range is wrong across tenors. At one week, k = 1.5 is dozens of stdevs
    out, where no option trades and the SVI wing is pure extrapolation.
    """
    atm = max(float(s.total_variance(np.array([0.0]))[0]), 1e-10)
    half_width = min(stdevs * np.sqrt(atm), 1.5)
    return np.linspace(-half_width, half_width, 241)


@dataclass(frozen=True, slots=True)
class ArbitrageReport:
    """Minimum Durrleman g and calendar gap, in the tradable core and out in the wings.

    Measured on real SPX: the core is clean while unquoted wings can show violations, and blending
    them would either hide real arbitrage or fail surfaces over strikes nobody can trade. A core
    violation means do not use this surface. A wing violation means do not price beyond the quotes.
    """

    core_min_g: float
    wing_min_g: float
    core_calendar_gap: float
    wing_calendar_gap: float

    @property
    def is_clean(self) -> bool:
        return self.core_min_g >= -1e-9 and self.core_calendar_gap >= -1e-9

    @property
    def wings_clean(self) -> bool:
        return self.wing_min_g >= -1e-9 and self.wing_calendar_gap >= -1e-9


@dataclass(frozen=True, slots=True)
class SviSurface:
    """Surface over variance-time. Before the first slice, total variance scales linearly with tau
    (constant vol per k); beyond the last, it grows at the last slice's implied rate.

    `quoted_ranges` holds each slice's quoted log-moneyness range when the surface came from a fit.
    Arbitrage is verified only inside it, so `extrapolating` lets callers flag anything outside.
    """

    slices: tuple[SviSlice, ...]
    quoted_ranges: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValidationError("a surface needs at least one slice")
        taus = [s.tau for s in self.slices]
        if any(b <= a for a, b in pairwise(taus)):
            raise ValidationError("slices must be strictly increasing in tau")
        if self.quoted_ranges and len(self.quoted_ranges) != len(self.slices):
            raise ValidationError("quoted_ranges must match slices one to one")

    def extrapolating(self, k: FloatArray, tau: FloatArray) -> NDArray[np.bool_]:
        """True where (k, tau) is beyond the quotes of the nearest slice in tau. All False when the
        surface has no recorded ranges, such as one built directly from parameters."""
        k_ = np.atleast_1d(np.asarray(k, dtype=np.float64))
        tau_ = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        if not self.quoted_ranges:
            return np.zeros(k_.shape, dtype=bool)
        taus = np.array([s.tau for s in self.slices])
        nearest = np.argmin(np.abs(taus[:, None] - tau_[None, :]), axis=0)
        ranges = np.asarray(self.quoted_ranges, dtype=np.float64)
        return np.asarray((k_ < ranges[nearest, 0]) | (k_ > ranges[nearest, 1]), dtype=np.bool_)

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
        def min_g(stdevs: float) -> float:
            return min(float(np.min(s.durrleman_g(check_grid(s, stdevs)))) for s in self.slices)

        def min_gap(stdevs: float) -> float:
            gap = float("inf")
            for lo, hi in pairwise(self.slices):
                grid = check_grid(lo, stdevs)
                gap = min(gap, float(np.min(hi.total_variance(grid) - lo.total_variance(grid))))
            return gap

        return ArbitrageReport(
            core_min_g=min_g(_CORE_STDEVS),
            wing_min_g=min_g(_WING_STDEVS),
            core_calendar_gap=min_gap(_CORE_STDEVS),
            wing_calendar_gap=min_gap(_WING_STDEVS),
        )

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "kind": "svi",
            "slices": [(s.tau, s.a, s.b, s.rho, s.m, s.sigma) for s in self.slices],
        }


@dataclass(frozen=True, slots=True)
class SliceFit:
    """Fit quality. `weighted_rmse_vol` is the headline: it uses the calibration weights, so wings
    with tiny premium do not dominate. `rmse_vol` and `max_abs_error_vol` stay visible for them."""

    slice: SviSlice
    weighted_rmse_vol: float
    rmse_vol: float
    max_abs_error_vol: float
    n_points: int
    butterfly_min_g: float
    k_min: float
    k_max: float


_LOWER = np.array([-1.0, 1e-7, -0.999, -2.0, 1e-5])
_UPPER = np.array([5.0, 5.0, 0.999, 2.0, 5.0])


def fit_svi_slice(
    k: FloatArray,
    vol: FloatArray,
    tau: float,
    weights: FloatArray | None = None,
    *,
    floor_slice: SviSlice | None = None,
    floor_range: tuple[float, float] | None = None,
    min_points: int = 5,
) -> SliceFit:
    """Fit raw SVI to vols at log-moneyness k by weighted least squares in vol, with penalties for
    butterfly arbitrage and for crossing below `floor_slice` within `floor_range`.

    The calendar floor applies only where both slices are quoted, since a shorter slice's wing is
    extrapolation and says nothing about what the longer slice must clear.

    Multi-start for robustness; every start is deterministic, so the same quotes give the same fit.
    """
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

    # Penalize over the quoted range plus a margin in stdevs, matching `check_grid`.
    order = np.argsort(k_)
    atm_var = max(float(np.interp(0.0, k_[order], v_[order]) ** 2 * tau), 1e-10)
    margin = _WING_STDEVS * np.sqrt(atm_var)
    grid = np.linspace(min(float(np.min(k_)), -margin), max(float(np.max(k_)), margin), 61)
    floor_grid: FloatArray | None = None
    floor: FloatArray | None = None
    if floor_slice is not None:
        low, high = floor_range if floor_range is not None else (-margin, margin)
        low, high = max(low, float(np.min(k_))), min(high, float(np.max(k_)))
        if high > low:
            floor_grid = np.linspace(low, high, 41)
            floor = floor_slice.total_variance(floor_grid)
    weight_penalty = 10.0

    def residuals(p: FloatArray) -> FloatArray:
        s = _unchecked_slice(tau, p)
        total = np.maximum(s.total_variance(k_), 1e-12)
        fit_error = (np.sqrt(total / tau) - v_) * w_
        g = s.durrleman_g(grid)
        parts = [fit_error, weight_penalty * np.minimum(g, 0.0)]
        if floor is not None and floor_grid is not None:
            # Relative to at-the-money variance: raw total variance is around 1e-4 at one week,
            # which would leave the penalty far too weak against errors measured in vol.
            gap = s.total_variance(floor_grid) - floor
            parts.append(weight_penalty * np.minimum(gap, 0.0) / atm_var)
        parts.append(weight_penalty * np.minimum(p[0] + p[1] * p[4] * np.sqrt(1 - p[2] ** 2), 0.0))
        return np.concatenate([np.atleast_1d(x) for x in parts])

    best = None
    for rho0 in (-0.7, -0.3, 0.2):
        for sig0 in (0.1, 0.4):
            root_var = np.sqrt(atm_var)
            start = np.array(
                [0.5 * atm_var, 0.5 * root_var, rho0, 0.0, max(sig0 * root_var * 4.0, 2e-4)]
            )
            start = np.clip(start, _LOWER + 1e-9, _UPPER - 1e-9)
            sol = least_squares(
                residuals, start, bounds=(_LOWER, _UPPER), x_scale="jac", max_nfev=200
            )
            if best is None or sol.cost < best.cost:
                best = sol
    assert best is not None
    slice_ = _feasible_slice(tau, best.x)
    errors = np.sqrt(np.maximum(slice_.total_variance(k_), 0.0) / tau) - v_
    return SliceFit(
        slice=slice_,
        weighted_rmse_vol=float(np.sqrt(np.average(errors**2, weights=w_**2))),
        rmse_vol=float(np.sqrt(np.mean(errors**2))),
        max_abs_error_vol=float(np.max(np.abs(errors))),
        n_points=int(k_.size),
        butterfly_min_g=float(np.min(slice_.durrleman_g(check_grid(slice_)))),
        k_min=float(np.min(k_)),
        k_max=float(np.max(k_)),
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


@dataclass(frozen=True, slots=True)
class QuotedArbitrageReport:
    """Arbitrage measured only where the market quotes: each slice's range and each pair's overlap.

    This is the calibration gate. `SviSurface.arbitrage_report` also covers extrapolated strikes.
    """

    min_g: float
    min_calendar_gap: float
    min_relative_calendar_gap: float
    worst_pair: tuple[float, float] | None

    @property
    def is_clean(self) -> bool:
        return self.min_g >= -1e-9 and self.min_relative_calendar_gap >= -_CALENDAR_TOLERANCE


def quoted_arbitrage_report(fits: Sequence[SliceFit]) -> QuotedArbitrageReport:
    """Butterfly over each quoted range, and calendar over each adjacent pair's quoted overlap.

    A relative calendar gap within `_CALENDAR_TOLERANCE` counts as clean: expiries hours apart (AM
    and PM settled on one date) legitimately have nearly equal variance, inside quote noise.
    """
    min_g = float("inf")
    for fit in fits:
        grid = np.linspace(fit.k_min, fit.k_max, 241)
        min_g = min(min_g, float(np.min(fit.slice.durrleman_g(grid))))
    min_gap = float("inf")
    min_relative = float("inf")
    worst: tuple[float, float] | None = None
    for lo, hi in pairwise(fits):
        low, high = max(lo.k_min, hi.k_min), min(lo.k_max, hi.k_max)
        if high <= low:
            continue
        grid = np.linspace(low, high, 241)
        lower = lo.slice.total_variance(grid)
        gap = hi.slice.total_variance(grid) - lower
        relative = gap / np.maximum(lower, 1e-12)
        min_gap = min(min_gap, float(np.min(gap)))
        if float(np.min(relative)) < min_relative:
            min_relative = float(np.min(relative))
            worst = (lo.slice.tau, hi.slice.tau)
    return QuotedArbitrageReport(min_g, min_gap, min_relative, worst)


def fit_surface(
    quotes: Sequence[tuple[float, FloatArray, FloatArray, FloatArray | None]],
) -> tuple[SviSurface, tuple[SliceFit, ...]]:
    """Fit an arbitrage-penalized surface from (tau, k, vol, weights) per expiry, shortest first."""
    ordered = sorted(quotes, key=lambda q: q[0])
    fits: list[SliceFit] = []
    previous: SliceFit | None = None
    for tau, k, vol, weights in ordered:
        fit = fit_svi_slice(
            k,
            vol,
            tau,
            weights,
            floor_slice=None if previous is None else previous.slice,
            floor_range=None if previous is None else (previous.k_min, previous.k_max),
        )
        fits.append(fit)
        previous = fit
    surface = SviSurface(tuple(f.slice for f in fits), tuple((f.k_min, f.k_max) for f in fits))
    return surface, tuple(fits)
