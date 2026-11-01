#!/usr/bin/env python3
"""Reconcile a real option chain: quality, parity forwards, surface fit, and vendor Greeks.

Usage: reconcile_chain.py <payload.json> [--rate R] [--out summary.json]

Reads a saved Cboe delayed-quotes payload. Quote files stay outside version control because market
data redistribution is license-restricted. Only aggregate statistics are written, and those are
what docs/evidence commits.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np

from greeks_neutralizer.data.cboe import parse_cboe_delayed
from greeks_neutralizer.data.chain import box_implied_rate, calibrate_chain
from greeks_neutralizer.data.quality import QualityPolicy
from greeks_neutralizer.domain import OptionContract, OptionRight
from greeks_neutralizer.market import DiscountCurve, MarketSnapshot, UnderlyingMarket
from greeks_neutralizer.pricing.book import value_options
from greeks_neutralizer.surface import quoted_arbitrage_report
from greeks_neutralizer.time_model import calendar_years


def reconcile(payload_path: Path, rate: float | None) -> dict[str, object]:
    parsed = parse_cboe_delayed(json.loads(payload_path.read_text()))
    # The payload is one instant, and this source does not always populate quote sizes.
    policy = QualityPolicy(max_age=timedelta(days=30), require_size=parsed.sizes_available)
    near = [
        q
        for q in parsed.quotes
        if 2 / 365 <= calendar_years(parsed.as_of, q.contract.expiry) <= 1.5
        and abs(np.log(q.contract.strike / parsed.spot)) < 0.5
    ]
    if rate is None:
        rate = box_implied_rate(near, parsed.as_of, parsed.spot)
    calibration = calibrate_chain(
        near, parsed.as_of, parsed.spot, discount_rate=rate, policy=policy
    )
    # The last index print goes stale against option quotes, so spot comes from nearest parity.
    nearest = calibration.forwards[0]
    implied_spot = nearest.forward * nearest.discount
    market = UnderlyingMarket(
        parsed.symbol, implied_spot, DiscountCurve.flat(rate), calibration.surface
    )
    snapshot = MarketSnapshot(parsed.as_of, {parsed.symbol: market})

    forwards = {p.expiry: p for p in calibration.forwards}
    contracts: list[OptionContract] = []
    mids: list[float] = []
    half_spreads: list[float] = []
    vendor_iv: list[float] = []
    vendor_delta: list[float] = []
    vendor_gamma: list[float] = []
    model_forwards: list[float] = []
    for item in near:
        c = item.contract
        parity = forwards.get(c.expiry)
        if parity is None or item.quote.bid <= 0.0:
            continue
        otm = (c.right is OptionRight.CALL) == (c.strike >= parity.forward)
        if not otm or abs(np.log(c.strike / parity.forward)) > 0.25:
            continue
        contracts.append(c)
        mids.append(item.quote.mid)
        half_spreads.append(item.quote.half_spread)
        vendor_iv.append(np.nan if item.vendor_iv is None else item.vendor_iv)
        vendor_delta.append(np.nan if item.vendor_delta is None else item.vendor_delta)
        vendor_gamma.append(np.nan if item.vendor_gamma is None else item.vendor_gamma)
        model_forwards.append(parity.forward)

    greeks, model_vol = _value_on_parity(contracts, snapshot, model_forwards, market)
    model_price, model_delta, model_gamma = greeks.price, greeks.delta, greeks.gamma

    mid = np.asarray(mids)
    half = np.asarray(half_spreads)
    inside = np.abs(model_price - mid) <= half
    v_iv, v_delta, v_gamma = map(np.asarray, (vendor_iv, vendor_delta, vendor_gamma))
    valid_iv = np.isfinite(v_iv) & (v_iv > 0.0)
    valid_delta = np.isfinite(v_delta) & (np.abs(v_delta) > 0.0)
    valid_gamma = np.isfinite(v_gamma) & (v_gamma > 0.0)

    report = calibration.surface.arbitrage_report()
    quoted = quoted_arbitrage_report(calibration.fits)
    return {
        "symbol": parsed.symbol,
        "as_of": parsed.as_of.isoformat(),
        "last_print": parsed.spot,
        "parity_implied_spot": implied_spot,
        "discount_rate_used": rate,
        "rows_non_standard_deliverable": parsed.n_non_standard,
        "sizes_available": parsed.sizes_available,
        "quotes_in_payload": len(parsed.quotes),
        "rows_rejected_by_parser": parsed.n_rejected,
        "quotes_in_scope": len(near),
        "quotes_used_in_fit": calibration.n_quotes_used,
        "quality_flags": calibration.flags,
        "expiries_fit": len(calibration.fits),
        "expiries_excluded": len(calibration.excluded_expiries),
        "fit_weighted_rmse_vol_median": float(
            np.median([f.weighted_rmse_vol for f in calibration.fits])
        ),
        "fit_weighted_rmse_vol_max": float(np.max([f.weighted_rmse_vol for f in calibration.fits])),
        "fit_rmse_vol_median": float(np.median([f.rmse_vol for f in calibration.fits])),
        "fit_rmse_vol_max": float(np.max([f.rmse_vol for f in calibration.fits])),
        "arbitrage_quoted_clean": quoted.is_clean,
        "butterfly_quoted_min_g": quoted.min_g,
        "calendar_quoted_min_gap": quoted.min_calendar_gap,
        "calendar_quoted_min_relative_gap": quoted.min_relative_calendar_gap,
        "calendar_quoted_worst_pair_tau": quoted.worst_pair,
        "arbitrage_core_clean": report.is_clean,
        "butterfly_core_min_g": report.core_min_g,
        "calendar_core_min_gap": report.core_calendar_gap,
        "butterfly_wing_min_g": report.wing_min_g,
        "calendar_wing_min_gap": report.wing_calendar_gap,
        "contracts_repriced": len(contracts),
        "share_model_price_inside_bid_ask": float(np.mean(inside)),
        "median_abs_price_error_in_half_spreads": float(
            np.median(np.abs(model_price - mid) / np.maximum(half, 1e-6))
        ),
        "vendor_iv_compared": int(valid_iv.sum()),
        "median_abs_iv_diff_vs_vendor": float(np.median(np.abs(model_vol - v_iv)[valid_iv])),
        "vendor_delta_compared": int(valid_delta.sum()),
        "median_abs_delta_diff_vs_vendor": float(
            np.median(np.abs(model_delta - v_delta)[valid_delta])
        ),
        "vendor_gamma_compared": int(valid_gamma.sum()),
        "median_rel_gamma_diff_vs_vendor": float(
            np.median(np.abs(model_gamma[valid_gamma] / v_gamma[valid_gamma] - 1.0))
        ),
    }


def _value_on_parity(contracts, snapshot, forwards, market):
    """Value each contract with a carry yield that reproduces its expiry's parity forward."""
    from greeks_neutralizer.pricing.black import GREEK_NAMES, GreekArrays

    n = len(contracts)
    buffers = {name: np.zeros(n) for name in GREEK_NAMES}
    vol = np.zeros(n)
    groups: dict[object, list[int]] = {}
    for i, c in enumerate(contracts):
        groups.setdefault(c.expiry, []).append(i)
    for expiry, idx in groups.items():
        T = calendar_years(snapshot.as_of, expiry)
        df = float(market.curve.discount(T))
        q = float(-np.log(forwards[idx[0]] * df / market.spot) / T)
        shifted = snapshot.replaced(market.symbol, carry_yield=q)
        result = value_options([contracts[i] for i in idx], shifted, tree_steps=51)
        for name in GREEK_NAMES:
            buffers[name][idx] = getattr(result.greeks, name)
        vol[idx] = result.vol
    return GreekArrays(**buffers), vol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("--rate", type=float, help="discount rate; default is box-implied")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summary = reconcile(args.payload, args.rate)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
