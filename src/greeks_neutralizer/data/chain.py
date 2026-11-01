"""Option chains: quotes per contract, implied forwards from parity, and surface calibration.

The forward comes from the market rather than from assumed dividends: at each expiry, put-call
parity across near-the-money strikes gives F and the discount factor, by regressing C - P on K.
This is the standard way to avoid guessing dividends and borrow for index and ETF options.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from greeks_neutralizer.data.quality import QualityPolicy, QuoteFlag, assess_quote, is_usable
from greeks_neutralizer.domain import ExerciseStyle, OptionContract, OptionRight, Quote
from greeks_neutralizer.errors import MarketDataError
from greeks_neutralizer.pricing.american import (
    AmericanBatch,
    american_price,
    european_closed_form,
)
from greeks_neutralizer.pricing.black import FloatArray
from greeks_neutralizer.pricing.implied_vol import implied_vol
from greeks_neutralizer.surface import (
    QuotedArbitrageReport,
    SliceFit,
    SviSurface,
    fit_surface,
    quoted_arbitrage_report,
)
from greeks_neutralizer.time_model import CALENDAR_CLOCK, VarianceClock, calendar_years


@dataclass(frozen=True, slots=True)
class ChainQuote:
    contract: OptionContract
    quote: Quote
    open_interest: int = 0
    volume: int = 0
    vendor_iv: float | None = None
    vendor_delta: float | None = None
    vendor_gamma: float | None = None


@dataclass(frozen=True, slots=True)
class ParityForward:
    expiry: datetime
    T: float
    forward: float
    discount: float
    n_strikes: int
    residual_rms: float


def _pairs(
    quotes: Iterable[ChainQuote], expiry: datetime, spot: float, window: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Strike, C - P mid, and combined spread for strikes with two-sided call and put markets."""
    calls: dict[float, Quote] = {}
    puts: dict[float, Quote] = {}
    for item in quotes:
        c = item.contract
        if c.expiry != expiry or item.quote.bid <= 0.0 or item.quote.ask <= item.quote.bid:
            continue
        if abs(c.strike / spot - 1.0) > window:
            continue
        (calls if c.right is OptionRight.CALL else puts)[c.strike] = item.quote
    strikes = sorted(set(calls) & set(puts))
    K = np.array(strikes, dtype=np.float64)
    diff = np.array([calls[s].mid - puts[s].mid for s in strikes], dtype=np.float64)
    spread = np.array(
        [(calls[s].ask - calls[s].bid) + (puts[s].ask - puts[s].bid) for s in strikes],
        dtype=np.float64,
    )
    return K, diff, spread


def implied_forward(
    quotes: Iterable[ChainQuote],
    expiry: datetime,
    as_of: datetime,
    spot: float,
    *,
    discount: float,
    window: float = 0.05,
) -> ParityForward:
    """Forward at one expiry from C - P = D (F - K), given the discount factor D.

    D is an input rather than regressed, because regressing it from American quotes is biased by
    early exercise (measured on SPY: implied rates of -6% at two months). Weighted toward tight
    markets, using strikes near spot. Fails closed with fewer than 2 usable pairs.
    """
    K, diff, spread = _pairs(quotes, expiry, spot, window)
    if K.size < 2:
        raise MarketDataError(f"parity forward at {expiry}: only {K.size} usable pairs")
    weights = 1.0 / np.maximum(spread, 1e-3) ** 2
    F = float(np.average(diff / discount + K, weights=weights))
    if F <= 0.0:
        raise MarketDataError(f"parity forward at {expiry}: non-positive forward {F}")
    residual = diff - discount * (F - K)
    return ParityForward(
        expiry=expiry,
        T=calendar_years(as_of, expiry),
        forward=F,
        discount=discount,
        n_strikes=int(K.size),
        residual_rms=float(np.sqrt(np.mean(residual**2))),
    )


def box_implied_rate(
    quotes: Iterable[ChainQuote], as_of: datetime, spot: float, *, min_T: float = 0.25
) -> float:
    """Financing rate implied by European put-call parity: the slope of C - P on K is -D.

    Median across expiries of at least `min_T`, since the slope is unresolvable when short-dated.
    Fails closed on American chains, where parity holds only as an inequality.
    """
    items = list(quotes)
    if any(i.contract.style is not ExerciseStyle.EUROPEAN for i in items):
        raise MarketDataError("box-implied rate requires a European chain")
    rates: list[float] = []
    for expiry in sorted({i.contract.expiry for i in items}):
        T = calendar_years(as_of, expiry)
        if T < min_T:  # noqa: SIM300
            continue
        K, diff, spread = _pairs(items, expiry, spot, 0.15)
        if K.size < 5:
            continue
        sqrt_w = 1.0 / np.maximum(spread, 1e-3)
        design = np.column_stack([np.ones_like(K), -K])
        solution = np.linalg.lstsq(design * sqrt_w[:, None], diff * sqrt_w, rcond=None)[0]
        if solution[1] > 0.0:
            rates.append(float(-np.log(solution[1]) / T))
    if not rates:
        raise MarketDataError("no expiry resolves a box-implied rate")
    return float(np.median(rates))


@dataclass(frozen=True, slots=True)
class ChainCalibration:
    surface: SviSurface
    fits: tuple[SliceFit, ...]
    forwards: tuple[ParityForward, ...]
    n_quotes_total: int
    n_quotes_used: int
    excluded_expiries: tuple[str, ...]
    flags: dict[str, int]
    quoted_arbitrage: QuotedArbitrageReport
    n_unconverged_fits: int


def _deamericanize(
    prices: FloatArray,
    forward: float,
    discount: float,
    strikes: FloatArray,
    omega: FloatArray,
    T: float,
    tau: float,
    spot: float,
    european_vol: FloatArray,
) -> FloatArray:
    """Remove the early-exercise premium from American mids so a European surface can be fit.

    Fixed-point: price an American at the current vol, measure its premium over European, and
    re-invert. Two passes are enough since the premium is small and smooth in vol for OTM quotes.
    """
    r = -np.log(discount) / T
    q = r - np.log(forward / spot) / T
    vol = np.maximum(np.nan_to_num(european_vol, nan=0.3), 1e-4)
    european_prices = prices
    n = strikes.size
    for _ in range(2):
        batch = AmericanBatch(
            spot=np.full(n, spot),
            strike=strikes,
            T=np.full(n, T),
            tau=np.full(n, tau),
            r=np.full(n, r),
            q=np.full(n, q),
            sigma=vol,
            omega=omega,
            div_times=np.zeros((n, 0)),
            div_amounts=np.zeros((n, 0)),
        )
        premium = np.maximum(american_price(batch, 51) - european_closed_form(batch), 0.0)
        european_prices = np.maximum(prices - premium, 1e-8)
        result = implied_vol(european_prices, forward, strikes, discount, tau, omega)
        vol = np.where(result.is_ok, result.vol, vol)
    return np.asarray(european_prices, dtype=np.float64)


def calibrate_chain(
    chain: Iterable[ChainQuote],
    as_of: datetime,
    spot: float,
    *,
    discount_rate: float,
    clock: VarianceClock = CALENDAR_CLOCK,
    policy: QualityPolicy | None = None,
    min_points_per_expiry: int = 8,
    max_abs_log_moneyness_stdevs: float = 4.0,
    max_weighted_fit_error: float = 0.02,
    max_parity_residual_share: float = 0.005,
) -> ChainCalibration:
    """Quality gate, parity forwards, OTM implied vols, then an arbitrage-penalized SVI surface.

    Uses OTM options only, where vega is resolvable and early exercise matters least.

    Fails closed. An expiry is excluded, with its reason recorded, if parity does not hold or its
    fit error is over the limit. With no expiry left, or with arbitrage inside the quoted strikes,
    this raises rather than returning a surface that looks usable.
    """
    items = list(chain)
    if not items:
        raise MarketDataError("empty chain")
    by_expiry: dict[datetime, list[ChainQuote]] = defaultdict(list)
    flag_counts: dict[str, int] = defaultdict(int)
    usable = 0
    for item in items:
        flags = assess_quote(item.quote, as_of, policy=policy)
        for flag in flags:
            flag_counts[flag.value] += 1
        if (
            is_usable(flags)
            and QuoteFlag.ZERO_BID not in flags
            and QuoteFlag.WIDE_SPREAD not in flags
            and item.contract.expiry > as_of
        ):
            by_expiry[item.contract.expiry].append(item)
            usable += 1

    slices: list[tuple[float, FloatArray, FloatArray, FloatArray | None]] = []
    forwards: list[ParityForward] = []
    excluded: list[str] = []
    used = 0
    for expiry in sorted(by_expiry):
        group = by_expiry[expiry]
        T = calendar_years(as_of, expiry)
        tau = clock.tau(as_of, expiry)
        discount = float(np.exp(-discount_rate * T))
        try:
            parity = implied_forward(group, expiry, as_of, spot, discount=discount)
            data = _expiry_vols(group, parity, tau, spot, max_abs_log_moneyness_stdevs)
        except MarketDataError as exc:
            excluded.append(f"{expiry.date()}: {exc}")
            continue
        if parity.residual_rms > max_parity_residual_share * parity.forward:
            excluded.append(
                f"{expiry.date()}: parity residual {parity.residual_rms:.3f} on a forward of "
                f"{parity.forward:.2f}"
            )
            continue
        k, vol, vega_weight = data
        if k.size < min_points_per_expiry:
            excluded.append(f"{expiry.date()}: only {k.size} usable OTM points")
            continue
        slices.append((tau, k, vol, vega_weight))
        forwards.append(parity)
        used += int(k.size)
    if not slices:
        raise MarketDataError(f"no expiry could be calibrated: {excluded[:3]}")
    _, all_fits = fit_surface(slices)
    kept: list[SliceFit] = []
    kept_forwards: list[ParityForward] = []
    for fit, parity in zip(all_fits, forwards, strict=True):
        if fit.weighted_rmse_vol > max_weighted_fit_error:
            excluded.append(
                f"{parity.expiry.date()}: fit error {fit.weighted_rmse_vol:.4f} is over the limit"
            )
            used -= fit.n_points
            continue
        kept.append(fit)
        kept_forwards.append(parity)
    if not kept:
        raise MarketDataError(f"no expiry could be calibrated: {excluded[:3]}")
    surface = SviSurface(tuple(f.slice for f in kept), tuple((f.k_min, f.k_max) for f in kept))
    fits = tuple(kept)
    forwards = kept_forwards
    report = quoted_arbitrage_report(fits)
    if not report.is_clean:
        raise MarketDataError(
            f"surface has arbitrage inside the quoted strikes: minimum g {report.min_g:.4f}, "
            f"calendar gap {report.min_relative_calendar_gap:.4%}"
        )
    return ChainCalibration(
        surface=surface,
        fits=fits,
        forwards=tuple(forwards),
        n_quotes_total=len(items),
        n_quotes_used=used,
        excluded_expiries=tuple(excluded),
        flags=dict(sorted(flag_counts.items())),
        quoted_arbitrage=report,
        n_unconverged_fits=sum(1 for fit in fits if not fit.converged),
    )


def _expiry_vols(
    group: list[ChainQuote],
    parity: ParityForward,
    tau: float,
    spot: float,
    max_stdevs: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    F, D = parity.forward, parity.discount
    otm = [
        item
        for item in group
        if (item.contract.right is OptionRight.CALL) == (item.contract.strike >= F)
    ]
    if not otm:
        raise MarketDataError("no OTM quotes")
    strikes = np.array([i.contract.strike for i in otm])
    omega = np.array([float(i.contract.right.omega) for i in otm])
    mids = np.array([i.quote.mid for i in otm])
    spreads = np.array([i.quote.ask - i.quote.bid for i in otm])
    result = implied_vol(mids, F, strikes, D, tau, omega)
    if any(i.contract.style is ExerciseStyle.AMERICAN for i in otm):
        european = _deamericanize(mids, F, D, strikes, omega, parity.T, tau, spot, result.vol)
        result = implied_vol(european, F, strikes, D, tau, omega)
    ok = result.is_ok
    vol = result.vol[ok]
    if vol.size == 0:
        raise MarketDataError("no implied vols solved")
    k = np.log(strikes[ok] / F)
    # The window is set by at-the-money vol. A median across OTM quotes is inflated by the wings,
    # which on a short expiry admits $0.05 markets 8 stdevs out, whose vols are noise.
    order = np.argsort(k)
    atm_vol = float(np.interp(0.0, k[order], vol[order]))
    within = np.abs(k) <= max_stdevs * atm_vol * np.sqrt(tau)
    # Vega-scaled confidence: tight markets on meaningful-vega strikes carry the fit.
    s = vol * np.sqrt(tau)
    d1 = -k / s + 0.5 * s
    vega = np.sqrt(F * strikes[ok]) * np.exp(-0.5 * d1 * d1) * np.sqrt(tau)
    weight = (vega / np.maximum(spreads[ok], 0.01)) ** 2
    return k[within], vol[within], weight[within]
