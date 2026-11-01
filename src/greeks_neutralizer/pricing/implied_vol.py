"""Vectorized implied volatility from option premium.

Works on the normalized OTM premium b(s), which is monotone in total stdev s with a single
inflection at s* = sqrt(2 |x|) (Jaeckel). Newton from the inflection point, in log space for very
small premiums, with a maintained bracket and a bisection fallback. Every element reports a status
so unattainable prices surface as such rather than as a silently wrong vol.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike

from greeks_neutralizer.errors import ValidationError
from greeks_neutralizer.pricing.black import FloatArray, normalized_otm_premium


class IVStatus(IntEnum):
    OK = 0
    BELOW_INTRINSIC = 1
    ABOVE_MAXIMUM = 2
    NO_CONVERGENCE = 3
    AT_INTRINSIC = 4


@dataclass(frozen=True, slots=True)
class ImpliedVolResult:
    """`vol` is NaN wherever `status` is not OK; `is_ok` gives the mask."""

    vol: FloatArray
    status: np.ndarray[tuple[int, ...], np.dtype[np.int8]]

    @property
    def is_ok(self) -> np.ndarray[tuple[int, ...], np.dtype[np.bool_]]:
        return np.asarray(self.status == IVStatus.OK, dtype=np.bool_)


_MAX_ITER = 100
_S_MIN = 1e-10
_S_MAX = 40.0


def implied_vol(
    price: ArrayLike,
    forward: ArrayLike,
    strike: ArrayLike,
    df: ArrayLike,
    tau: ArrayLike,
    omega: ArrayLike,
) -> ImpliedVolResult:
    """Invert discounted European premium to Black vol. `tau` is variance-time in years."""
    P, F, K, D, tau_, om = (
        np.atleast_1d(a).astype(np.float64)
        for a in np.broadcast_arrays(
            *(np.asarray(x, dtype=np.float64) for x in (price, forward, strike, df, tau, omega))
        )
    )
    for name, arr in {"price": P, "forward": F, "strike": K, "df": D, "tau": tau_}.items():
        if not np.all(np.isfinite(arr)):
            raise ValidationError(f"{name} contains NaN or inf")
    if np.any(F <= 0.0) or np.any(K <= 0.0) or np.any(D <= 0.0) or np.any(tau_ <= 0.0):
        raise ValidationError("forward, strike, df, and tau must be positive")

    undiscounted = P / D
    intrinsic = np.maximum(om * (F - K), 0.0)
    otm_value = undiscounted - intrinsic
    root_fk = np.sqrt(F * K)
    target = otm_value / root_fk
    abs_x = np.abs(np.log(F / K))
    upper = np.exp(-0.5 * abs_x)  # limit of normalized OTM premium as s tends to infinity

    # Tolerance scales with the premium so round-off in `price - intrinsic` is not read as an error.
    noise = 4.0 * np.finfo(np.float64).eps * np.maximum(undiscounted, intrinsic) / root_fk

    vol = np.full_like(P, np.nan)
    status = np.full(P.shape, IVStatus.NO_CONVERGENCE, dtype=np.int8)
    status[target < -noise] = IVStatus.BELOW_INTRINSIC
    status[(target >= -noise) & (target <= noise)] = IVStatus.AT_INTRINSIC
    status[target >= upper] = IVStatus.ABOVE_MAXIMUM

    active = np.flatnonzero(status == IVStatus.NO_CONVERGENCE)
    if active.size:
        s, converged = _solve(target[active], abs_x[active])
        vol[active[converged]] = s[converged] / np.sqrt(tau_[active[converged]])
        status[active[converged]] = IVStatus.OK
    return ImpliedVolResult(vol=vol, status=status)


def _solve(target: FloatArray, abs_x: FloatArray) -> tuple[FloatArray, np.ndarray]:
    lo = np.full_like(target, _S_MIN)
    hi = np.full_like(target, _S_MAX)
    s = np.clip(np.sqrt(2.0 * abs_x), 1e-4, _S_MAX)
    # Near the money the inflection is at ~0, so seed with the ATM approximation b = s * phi(0).
    atm_seed = target * np.sqrt(2.0 * np.pi)
    s = np.where(abs_x < 1e-6, np.clip(atm_seed, 1e-6, _S_MAX), s)
    log_target = np.log(np.maximum(target, 1e-300))
    converged = np.zeros_like(target, dtype=bool)
    for _ in range(_MAX_ITER):
        open_ = np.flatnonzero(~converged)
        if open_.size == 0:
            break
        s_o = s[open_]
        x_o = abs_x[open_]
        b = normalized_otm_premium(x_o, s_o)
        h = -x_o / s_o
        t = 0.5 * s_o
        db = np.exp(-0.5 * (h * h + t * t)) / np.sqrt(2.0 * np.pi)
        # Maintain the bracket: premium is strictly increasing in s.
        above = b > target[open_]
        hi[open_] = np.where(above, s_o, hi[open_])
        lo[open_] = np.where(above, lo[open_], s_o)
        small = target[open_] < 1e-8
        with np.errstate(divide="ignore", invalid="ignore"):
            step_plain = (b - target[open_]) / db
            step_log = (np.log(np.maximum(b, 1e-300)) - log_target[open_]) * b / db
        step = np.where(small, step_log, step_plain)
        new = s_o - step
        bad = ~np.isfinite(new) | (new <= lo[open_]) | (new >= hi[open_])
        new = np.where(bad, 0.5 * (lo[open_] + hi[open_]), new)
        done = np.abs(new - s_o) <= 1e-14 * np.maximum(new, 1e-4)
        s[open_] = new
        converged[open_] = done
    return s, converged
