"""Batched American option valuation with discrete dividends.

Method: Leisen-Reimer binomial tree (odd steps) on the escrowed spot, vectorized across options so
a full chain is one tensor sweep, with a European control variate: the tree's own European error is
removed using the exact closed form, which cuts error and makes bumped Greeks smooth.

Discrete dividends use the escrowed model consistent with `market.UnderlyingMarket.forward`: the
tree evolves S* = S - PV(dividends before expiry), and the exercise value at each node adds back the
PV of dividends not yet gone ex. Exercise just before an ex-date therefore captures the dividend.

Greeks to second order, and charm, come from central bumps through the control-variate price.
Veta, speed, zomma, and color stay at their European values for American rows: those rows are
marked by `OptionValuation.american_tree`, and these Greeks are not reported in `Exposure`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from greeks_neutralizer.pricing.black import FloatArray, black_premium

DEFAULT_STEPS = 201
DEFAULT_GREEK_STEPS = 101


@dataclass(frozen=True, slots=True)
class AmericanBatch:
    """One option per row. All arrays have shape (n,). Dividends are per-option padded tables."""

    spot: FloatArray
    strike: FloatArray
    T: FloatArray
    tau: FloatArray
    r: FloatArray
    q: FloatArray
    sigma: FloatArray
    omega: FloatArray
    # shape (n, m): time of each dividend (inf where unused) and its cash amount.
    div_times: FloatArray
    div_amounts: FloatArray


def _peizer_pratt(z: FloatArray, n: int) -> FloatArray:
    """Peizer-Pratt method 2 inversion of the normal into a binomial probability."""
    denom = n + 1.0 / 3.0 + 0.1 / (n + 1.0)
    return np.asarray(
        0.5 + np.sign(z) * 0.5 * np.sqrt(1.0 - np.exp(-((z / denom) ** 2) * (n + 1.0 / 6.0))),
        dtype=np.float64,
    )


def _pv_divs_at(b: AmericanBatch, t: FloatArray) -> FloatArray:
    """PV at time t of dividends after t and on or before expiry, with flat rate r in the batch."""
    if b.div_times.shape[1] == 0:
        return np.zeros_like(t)
    live = (b.div_times > t[:, None]) & (b.div_times <= b.T[:, None])
    pv = b.div_amounts * np.exp(-b.r[:, None] * np.maximum(b.div_times - t[:, None], 0.0))
    return np.asarray(np.sum(np.where(live, pv, 0.0), axis=1), dtype=np.float64)


def _apply_exercise_at_ex_date(
    b: AmericanBatch,
    american: FloatArray,
    nodes: FloatArray,
    pv_now: FloatArray,
    dt: FloatArray,
    i: int,
) -> FloatArray:
    """Value calls for the strategy "exercise immediately before the next ex-date".

    That instant is where a call's early exercise is optimal, but it generally falls between tree
    steps. From a node at time t_i with the ex-date delta later, committing to exercise then is
    worth S* exp(-q delta) + PV(dividends) - K exp(-r delta). Without this floor, the tree charges
    interest on the strike over the gap and undervalues the call by about r K dt.
    """
    t_i = dt * i
    with np.errstate(invalid="ignore"):
        gaps = np.where(b.div_times > t_i[:, None], b.div_times - t_i[:, None], np.inf)
    delta = gaps.min(axis=1)
    rows = np.flatnonzero((delta <= dt) & (b.omega > 0.0))
    if rows.size == 0:
        return american
    d = delta[rows][:, None]
    strategy = (
        nodes[rows] * np.exp(-b.q[rows][:, None] * d)
        + pv_now[rows]
        - b.strike[rows][:, None] * np.exp(-b.r[rows][:, None] * d)
    )
    out = american.copy()
    out[rows] = np.maximum(american[rows], strategy)
    return out


def _tree_values(b: AmericanBatch, steps: int) -> tuple[FloatArray, FloatArray]:
    """(American, European) tree values per row. Requires T > 0, sigma > 0."""
    n = steps
    pv0 = _pv_divs_at(b, np.zeros_like(b.T))
    s_star = b.spot - pv0
    s = b.sigma * np.sqrt(b.tau)
    dt = b.T / n
    carry = b.r - b.q
    d1 = (np.log(s_star / b.strike) + carry * b.T) / s + 0.5 * s
    d2 = d1 - s
    # Clamped: at very low vol the inversion reaches 0 or 1 in floating point, leaving the tree
    # undefined, and such an option has no exercise premium to resolve anyway.
    p = np.clip(_peizer_pratt(d2, n), 1e-12, 1.0 - 1e-12)
    p_prime = np.clip(_peizer_pratt(d1, n), 1e-12, 1.0 - 1e-12)
    growth = np.exp(carry * dt)
    u = growth * p_prime / p
    d = (growth - p * u) / (1.0 - p)
    disc = np.exp(-b.r * dt)
    pu = (disc * p)[:, None]
    pd = (disc * (1.0 - p))[:, None]
    omega = b.omega[:, None]
    strike = b.strike[:, None]
    has_divs = b.div_times.shape[1] > 0 and bool(np.any(np.isfinite(b.div_times)))

    # Terminal nodes: index j counts up-moves. No dividends remain at expiry.
    j = np.arange(n + 1, dtype=np.float64)
    nodes = s_star[:, None] * u[:, None] ** j * d[:, None] ** (n - j)
    american = np.maximum(omega * (nodes - strike), 0.0)
    european = american.copy()
    inv_d = (1.0 / d)[:, None]
    for i in range(n - 1, -1, -1):
        american = pd * american[:, :-1] + pu * american[:, 1:]
        european = pd * european[:, :-1] + pu * european[:, 1:]
        nodes = nodes[:, :-1] * inv_d
        exercise_spot = nodes
        if has_divs:
            pv_now = _pv_divs_at(b, dt * i)[:, None]
            exercise_spot = nodes + pv_now
            american = _apply_exercise_at_ex_date(b, american, nodes, pv_now, dt, i)
        american = np.maximum(american, omega * (exercise_spot - strike))
    return american[:, 0], european[:, 0]


def european_closed_form(b: AmericanBatch) -> FloatArray:
    s_star = b.spot - _pv_divs_at(b, np.zeros_like(b.T))
    forward = s_star * np.exp((b.r - b.q) * b.T)
    df = np.exp(-b.r * b.T)
    return black_premium(forward, b.strike, df, b.sigma * np.sqrt(b.tau), b.omega)


def _exercise_premium(batch: AmericanBatch, steps: int) -> FloatArray:
    tree_am, tree_eu = _tree_values(batch, steps)
    return np.maximum(tree_am - tree_eu, 0.0)


def _select(b: AmericanBatch, rows: NDArray[np.intp]) -> AmericanBatch:
    return AmericanBatch(
        spot=b.spot[rows],
        strike=b.strike[rows],
        T=b.T[rows],
        tau=b.tau[rows],
        r=b.r[rows],
        q=b.q[rows],
        sigma=b.sigma[rows],
        omega=b.omega[rows],
        div_times=b.div_times[rows],
        div_amounts=b.div_amounts[rows],
    )


def american_price(batch: AmericanBatch, steps: int = DEFAULT_STEPS) -> FloatArray:
    """American value: exact European closed form plus the tree's early-exercise premium.

    Without dividends the premium converges smoothly at O(1/n) against QuantLib's high-precision
    ALO engine, so premiums at n and about n/2 are Richardson-extrapolated.

    With dividends, the error is not smooth in n (measured non-monotone, because node placement
    around the exercise boundary near each ex-date dominates), so extrapolating would amplify it.
    Those rows use a single finer tree of 2n - 1 steps instead. At the default n this is within
    0.25 cents of QuantLib's escrowed finite-difference engine in the reconciled cases.

    `steps` must be odd for Leisen-Reimer.
    """
    if steps % 2 == 0:
        raise ValueError("Leisen-Reimer requires an odd step count")
    if batch.spot.size == 0:
        return np.zeros(0, dtype=np.float64)
    premium = np.zeros_like(batch.spot)
    has_div = np.any(batch.div_amounts > 0.0, axis=1) if batch.div_amounts.shape[1] else None
    with_div = np.zeros(0, dtype=np.intp) if has_div is None else np.flatnonzero(has_div)
    plain = np.setdiff1d(np.arange(batch.spot.size), with_div)
    if plain.size:
        sub = _select(batch, plain)
        fine = _exercise_premium(sub, steps)
        coarse = _exercise_premium(sub, steps // 2 | 1)
        premium[plain] = np.maximum(2.0 * fine - coarse, 0.0)
    if with_div.size:
        premium[with_div] = _exercise_premium(_select(batch, with_div), 2 * steps - 1)
    value = european_closed_form(batch) + premium
    immediate = batch.omega * (batch.spot - batch.strike)
    return np.asarray(np.maximum(value, immediate), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AmericanGreeks:
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
    exercise_premium: FloatArray


def _with(b: AmericanBatch, **changes: FloatArray) -> AmericanBatch:
    fields = {
        "spot": b.spot,
        "strike": b.strike,
        "T": b.T,
        "tau": b.tau,
        "r": b.r,
        "q": b.q,
        "sigma": b.sigma,
        "omega": b.omega,
        "div_times": b.div_times,
        "div_amounts": b.div_amounts,
    }
    fields.update(changes)
    return AmericanBatch(**fields)


def american_greeks(batch: AmericanBatch, steps: int = DEFAULT_GREEK_STEPS) -> AmericanGreeks:
    """Value and bump Greeks. Bumps are relative to the option's own scale so short-dated contracts
    get a spot bump that resolves their gamma, without crossing into a different tree regime."""
    b = batch
    p0 = american_price(b, steps)
    reported_price = american_price(b, DEFAULT_STEPS)
    stdev_move = b.spot * b.sigma * np.sqrt(b.tau)
    h = np.clip(0.05 * stdev_move, 1e-4 * b.spot, 0.01 * b.spot)
    dv = np.full_like(b.sigma, 1e-3)
    dr = np.full_like(b.r, 1e-4)

    up = american_price(_with(b, spot=b.spot + h), steps)
    dn = american_price(_with(b, spot=b.spot - h), steps)
    v_up = american_price(_with(b, sigma=b.sigma + dv), steps)
    v_dn = american_price(_with(b, sigma=np.maximum(b.sigma - dv, 1e-6)), steps)
    r_up = american_price(_with(b, r=b.r + dr), steps)
    r_dn = american_price(_with(b, r=b.r - dr), steps)
    q_up = american_price(_with(b, q=b.q + dr), steps)
    q_dn = american_price(_with(b, q=b.q - dr), steps)

    # Theta: a calendar roll of at most 1 day, half the remaining life, and half the time to the
    # next ex-date. Rolling across one with spot held flat would report the dividend as decay.
    next_ex = np.full_like(b.T, np.inf)
    if b.div_times.shape[1]:
        next_ex = np.min(np.where(b.div_amounts > 0.0, b.div_times, np.inf), axis=1)
    dt = np.minimum(np.minimum(1.0 / 365.0, 0.5 * b.T), 0.5 * next_ex)
    ratio = 1.0 - dt / b.T
    rolled_batch = _with(
        b,
        T=b.T * ratio,
        tau=b.tau * ratio,
        div_times=np.where(np.isfinite(b.div_times), b.div_times - dt[:, None], b.div_times),
    )
    rolled = american_price(rolled_batch, steps)
    rolled_up = american_price(_with(rolled_batch, spot=b.spot + h), steps)
    rolled_dn = american_price(_with(rolled_batch, spot=b.spot - h), steps)

    up_v = american_price(_with(b, spot=b.spot + h, sigma=b.sigma + dv), steps)
    dn_v = american_price(_with(b, spot=b.spot - h, sigma=b.sigma + dv), steps)
    up_w = american_price(_with(b, spot=b.spot + h, sigma=np.maximum(b.sigma - dv, 1e-6)), steps)
    dn_w = american_price(_with(b, spot=b.spot - h, sigma=np.maximum(b.sigma - dv, 1e-6)), steps)

    delta = (up - dn) / (2.0 * h)
    return AmericanGreeks(
        price=reported_price,
        delta=delta,
        gamma=np.maximum((up - 2.0 * p0 + dn) / (h * h), 0.0),
        vega=(v_up - v_dn) / (2.0 * dv),
        theta=(rolled - p0) / dt,
        rho=(r_up - r_dn) / (2.0 * dr),
        phi=(q_up - q_dn) / (2.0 * dr),
        vanna=(up_v - dn_v - up_w + dn_w) / (4.0 * h * dv),
        volga=(v_up - 2.0 * p0 + v_dn) / (dv * dv),
        charm=((rolled_up - rolled_dn) / (2.0 * h) - delta) / dt,
        exercise_premium=np.maximum(reported_price - european_closed_form(b), 0.0),
    )


def exercise_is_ever_optimal(
    omega: FloatArray, r: FloatArray, q: FloatArray, has_dividends: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """False only where early exercise cannot have value, so the European closed form is exact.

    A call needs r >= 0, q <= 0, and no dividends: with a negative rate, deferring the strike costs
    money. A put needs r <= 0 and q >= 0, for the same reason on the other side.
    """
    call = omega > 0
    call_never = call & (r >= 0.0) & (q <= 0.0) & ~has_dividends
    put_never = ~call & (r <= 0.0) & (q >= 0.0)
    return ~(call_never | put_never)
