"""Vectorized generalized Black-Scholes kernel on a forward, with all Greeks to third order.

All outputs are raw per-share sensitivities in natural units: vega per 1.00 of vol, theta per year
of calendar time, rho per 1.00 of rate. Scaling for reporting happens in `pricing.book`.

Time inputs are split so 0DTE and event-weighted clocks are exact:
    T   : calendar years, used for discounting and carry.
    tau : variance-time years, used for total stdev s = sigma * sqrt(tau).
The ratio kappa = tau / T is taken as locally constant when differentiating with respect to time.

Limits are explicit rather than accidental. s == 0 (zero vol) gives the discounted forward
intrinsic and T == 0 gives spot intrinsic, each with step delta and zero higher Greeks.

The premium is computed from the out-of-the-money side using erfcx so that deep wing values keep
relative accuracy and cannot go negative; in-the-money values add discounted intrinsic.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erfcx, ndtr

from greeks_neutralizer.errors import ValidationError

FloatArray = NDArray[np.float64]

_SQRT2 = np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def normal_pdf(x: FloatArray) -> FloatArray:
    return np.asarray(_INV_SQRT_2PI * np.exp(-0.5 * x * x), dtype=np.float64)


def normal_cdf(x: ArrayLike) -> FloatArray:
    """Standard normal CDF, accurate to machine precision including the far tails."""
    return np.asarray(ndtr(np.asarray(x, dtype=np.float64)), dtype=np.float64)


def normalized_otm_premium(abs_x: FloatArray, s: FloatArray) -> FloatArray:
    """Undiscounted OTM premium divided by sqrt(F K), for x = |ln(F/K)| and total stdev s > 0.

    Deep OTM uses the scaled complementary error function, which preserves relative precision where
    the textbook difference of two CDFs cancels to noise (and can even turn negative).
    """
    h = -abs_x / s
    t = 0.5 * s
    deep = (h + t) < -1.0
    out = np.empty_like(s)
    if np.any(deep):
        hd, td = h[deep], t[deep]
        out[deep] = (
            0.5
            * np.exp(-0.5 * (hd * hd + td * td))
            * (erfcx(-(hd + td) / _SQRT2) - erfcx(-(hd - td) / _SQRT2))
        )
    near = ~deep
    if np.any(near):
        half_x = 0.5 * abs_x[near]
        out[near] = np.exp(-half_x) * ndtr(h[near] + t[near]) - np.exp(half_x) * ndtr(
            h[near] - t[near]
        )
    return np.maximum(out, 0.0)


def black_premium(
    forward: FloatArray, strike: FloatArray, df: FloatArray, s: FloatArray, omega: FloatArray
) -> FloatArray:
    """Discounted option premium. Requires s >= 0; s == 0 returns discounted intrinsic."""
    intrinsic = np.maximum(omega * (forward - strike), 0.0)
    positive = s > 0.0
    time_value = np.zeros_like(s)
    if np.any(positive):
        abs_x = np.abs(np.log(forward[positive] / strike[positive]))
        time_value[positive] = np.sqrt(forward[positive] * strike[positive]) * (
            normalized_otm_premium(abs_x, s[positive])
        )
    return np.asarray(df * (intrinsic + time_value), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class GreekArrays:
    """Per-share value and sensitivities, one array per Greek, all the same shape."""

    price: FloatArray
    delta: FloatArray
    gamma: FloatArray
    vega: FloatArray
    theta: FloatArray
    rho: FloatArray
    phi: FloatArray
    vanna: FloatArray
    volga: FloatArray
    charm: FloatArray
    veta: FloatArray
    speed: FloatArray
    zomma: FloatArray
    color: FloatArray


GREEK_NAMES: tuple[str, ...] = tuple(f.name for f in fields(GreekArrays))


def _broadcast(*args: ArrayLike) -> tuple[FloatArray, ...]:
    arrays = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in args))
    return tuple(np.atleast_1d(a).astype(np.float64) for a in arrays)


def black_greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    omega: ArrayLike,
    tau: ArrayLike | None = None,
) -> GreekArrays:
    """Value and Greeks under generalized Black-Scholes with continuous rate r and carry yield q.

    Raises ValidationError on NaN or out-of-domain inputs instead of propagating them.
    """
    tau_in = T if tau is None else tau
    S, K, T_, r_, q_, vol, om, tau_ = _broadcast(spot, strike, T, r, q, sigma, omega, tau_in)
    _validate(S, K, T_, r_, q_, vol, om, tau_)

    df = np.exp(-r_ * T_)
    G = np.exp(-q_ * T_)
    forward = S * G / df
    sqrt_tau = np.sqrt(tau_)
    s = vol * sqrt_tau

    price = black_premium(forward, K, df, s, om)

    live = s > 0.0
    itm = om * (forward - K) > 0.0
    delta = np.where(itm, om * G, 0.0)
    gamma = np.zeros_like(S)
    vega = np.zeros_like(S)
    # Zero-vol forward value is df*max(w(F-K),0), so carry terms follow the ITM indicator.
    theta = np.where(itm, om * (q_ * S * G - r_ * K * df), 0.0)
    rho = np.where(itm, om * K * T_ * df, 0.0)
    phi = np.where(itm, -om * S * T_ * G, 0.0)
    vanna = np.zeros_like(S)
    volga = np.zeros_like(S)
    charm = np.where(itm, om * q_ * G, 0.0)
    veta = np.zeros_like(S)
    speed = np.zeros_like(S)
    zomma = np.zeros_like(S)
    color = np.zeros_like(S)

    if np.any(live):
        S_, K_, T_l, r_l, q_l = S[live], K[live], T_[live], r_[live], q_[live]
        v, om_l, G_l, df_l, s_l = vol[live], om[live], G[live], df[live], s[live]
        rt = sqrt_tau[live]
        kappa = tau_[live] / T_l
        b = r_l - q_l
        m = np.log(S_ / K_) + b * T_l
        d1 = m / s_l + 0.5 * s_l
        d2 = d1 - s_l
        pdf = normal_pdf(d1)
        n_w_d1 = normal_cdf(om_l * d1)
        n_w_d2 = normal_cdf(om_l * d2)

        gam = G_l * pdf / (S_ * s_l)
        veg = S_ * G_l * pdf * rt
        ddt = b / s_l - d2 / (2.0 * T_l)  # partial d1 / partial T

        delta[live] = om_l * G_l * n_w_d1
        gamma[live] = gam
        vega[live] = veg
        theta[live] = (
            -S_ * G_l * pdf * v * kappa / (2.0 * rt)
            + om_l * q_l * S_ * G_l * n_w_d1
            - om_l * r_l * K_ * df_l * n_w_d2
        )
        rho[live] = om_l * K_ * T_l * df_l * n_w_d2
        phi[live] = -om_l * S_ * T_l * G_l * n_w_d1
        vanna[live] = -G_l * pdf * d2 / v
        volga[live] = veg * d1 * d2 / v
        charm[live] = om_l * q_l * G_l * n_w_d1 - G_l * pdf * ddt
        veta[live] = -veg * (-q_l - d1 * ddt + 1.0 / (2.0 * T_l))
        speed[live] = -gam / S_ * (1.0 + d1 / s_l)
        zomma[live] = gam * (d1 * d2 - 1.0) / v
        color[live] = -gam * (-q_l - d1 * ddt - 1.0 / (2.0 * T_l))

    return GreekArrays(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        phi=phi,
        vanna=vanna,
        volga=volga,
        charm=charm,
        veta=veta,
        speed=speed,
        zomma=zomma,
        color=color,
    )


def _validate(
    S: FloatArray,
    K: FloatArray,
    T: FloatArray,
    r: FloatArray,
    q: FloatArray,
    vol: FloatArray,
    om: FloatArray,
    tau: FloatArray,
) -> None:
    named = {"spot": S, "strike": K, "T": T, "r": r, "q": q, "sigma": vol, "tau": tau}
    for name, arr in named.items():
        if not np.all(np.isfinite(arr)):
            raise ValidationError(f"{name} contains NaN or inf")
    if np.any(S <= 0.0) or np.any(K <= 0.0):
        raise ValidationError("spot and strike must be positive")
    if np.any(T < 0.0) or np.any(tau < 0.0):
        raise ValidationError("time to expiry must be non-negative")
    if np.any((T == 0.0) != (tau == 0.0)):
        raise ValidationError("T and tau must be zero together")
    if np.any(vol < 0.0):
        raise ValidationError("sigma must be non-negative")
    if not np.all(np.isin(om, (-1.0, 1.0))):
        raise ValidationError("omega must be +1 or -1")
