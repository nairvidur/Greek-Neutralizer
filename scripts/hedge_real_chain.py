#!/usr/bin/env python3
"""End to end on a real chain: calibrate, hedge a book with live quotes, verify with QuantLib.

Usage: hedge_real_chain.py <payload.json> --rate R [--out summary.json]

Book: short 50 of a roughly 1-stdev strangle at the expiry nearest 35 days.
Hedges: delta-one, plus every quoted option in a strike and tenor window around that expiry.
Costs come from the real quotes. Greeks of the hedged book are recomputed in QuantLib, a separate
implementation, using the same vol per contract. Only aggregates are written.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import QuantLib as ql

from greeks_neutralizer.data.cboe import parse_cboe_delayed
from greeks_neutralizer.data.chain import ChainQuote, calibrate_chain
from greeks_neutralizer.data.quality import QualityPolicy
from greeks_neutralizer.domain import LinearInstrument, OptionContract, OptionRight, Position, Quote
from greeks_neutralizer.hedging.diagnostics import hedge_frontier
from greeks_neutralizer.hedging.optimizer import solve_hedge
from greeks_neutralizer.hedging.problem import (
    GreekName,
    GreekTarget,
    HedgeCandidate,
    HedgeConfig,
    HedgeProblem,
)
from greeks_neutralizer.market import DiscountCurve, MarketSnapshot, UnderlyingMarket
from greeks_neutralizer.pricing.book import value_book, value_options
from greeks_neutralizer.risk.margin import UnderlyingClass, reg_t_short_option_estimate
from greeks_neutralizer.risk.measures import expiry_risks
from greeks_neutralizer.risk.scenarios import (
    HISTORICAL_STRESSES,
    HYPOTHETICAL_SHOCKS,
    book_scenario_pnl,
    taylor_vs_full,
)
from greeks_neutralizer.time_model import calendar_years


def nearest_expiry_with_quotes(quotes: tuple[ChainQuote, ...], as_of, days: float):  # type: ignore[no-untyped-def]
    expiries = sorted({q.contract.expiry for q in quotes if q.contract.expiry > as_of})
    return min(expiries, key=lambda e: abs(calendar_years(as_of, e) * 365.0 - days))


def quantlib_greeks(
    positions: list[Position], snapshot: MarketSnapshot, symbol: str
) -> dict[str, float]:
    """Delta shares, gamma shares, and vega per point for the book, computed by QuantLib."""
    market = snapshot.market(symbol)
    as_of = snapshot.as_of
    contracts = [p.instrument for p in positions if isinstance(p.instrument, OptionContract)]
    vols = value_options(contracts, snapshot).vol
    today = ql.Date(as_of.day, as_of.month, as_of.year)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    totals = {"delta_shares": 0.0, "gamma_shares": 0.0, "vega_1pt": 0.0, "value": 0.0}
    row = 0
    for position in positions:
        instrument = position.instrument
        if not isinstance(instrument, OptionContract):
            totals["delta_shares"] += position.quantity * instrument.multiplier
            totals["value"] += position.quantity * instrument.multiplier * market.spot
            continue
        T = calendar_years(as_of, instrument.expiry)
        rate = float(market.curve.zero_rate(np.array([T]))[0])
        # QuantLib dates are whole days. Scaling rate, yield, and vol to its time keeps r T, q T,
        # and sigma sqrt(T) exactly equal to ours, so the comparison tests formulas, not rounding.
        days = max(round(T * 365.0), 1)
        ratio = T / (days / 365.0)
        scale = float(np.sqrt(ratio))
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(market.spot)),
            ql.YieldTermStructureHandle(
                ql.FlatForward(today, market.carry_yield * ratio, day_count)
            ),
            ql.YieldTermStructureHandle(ql.FlatForward(today, rate * ratio, day_count)),
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, ql.NullCalendar(), float(vols[row]) * scale, day_count)
            ),
        )
        kind = ql.Option.Call if instrument.right is OptionRight.CALL else ql.Option.Put
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(kind, instrument.strike), ql.EuropeanExercise(today + days)
        )
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        size = position.quantity * instrument.multiplier
        totals["value"] += size * option.NPV()
        totals["delta_shares"] += size * option.delta()
        totals["gamma_shares"] += size * option.gamma()
        # QuantLib's vega is per its (time-scaled) vol. Convert to per point of the original vol.
        totals["vega_1pt"] += size * option.vega() * scale * 0.01
        row += 1
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    parsed = parse_cboe_delayed(json.loads(args.payload.read_text()))
    as_of = parsed.as_of
    policy = QualityPolicy(max_age=timedelta(days=30), require_size=parsed.sizes_available)
    near = [
        q
        for q in parsed.quotes
        if 2 / 365 <= calendar_years(as_of, q.contract.expiry) <= 0.6
        and abs(np.log(q.contract.strike / parsed.spot)) < 0.3
    ]
    calibration = calibrate_chain(near, as_of, parsed.spot, discount_rate=args.rate, policy=policy)
    nearest = calibration.forwards[0]
    spot = nearest.forward * nearest.discount
    # The parity forward carries the dividend yield, so recover it for pricing.
    target_forward = min(calibration.forwards, key=lambda p: abs(p.T - 35 / 365))
    carry = args.rate - float(np.log(target_forward.forward / spot) / target_forward.T)
    market = UnderlyingMarket(
        parsed.symbol, spot, DiscountCurve.flat(args.rate), calibration.surface, carry
    )
    snapshot = MarketSnapshot(as_of, {parsed.symbol: market})

    expiry = nearest_expiry_with_quotes(tuple(near), as_of, 35.0)
    T = calendar_years(as_of, expiry)
    atm = float(calibration.surface.vol(np.array([0.0]), np.array([T]))[0])
    forward = market.forward(as_of, expiry)
    width = atm * np.sqrt(T)
    same = [q for q in near if q.contract.expiry == expiry and q.quote.bid > 0.0]

    def closest(right: OptionRight, strike: float) -> OptionContract:
        options = [q.contract for q in same if q.contract.right is right]
        return min(options, key=lambda c: abs(c.strike - strike))

    book = (
        Position(closest(OptionRight.PUT, forward * np.exp(-width)), -50),
        Position(closest(OptionRight.CALL, forward * np.exp(width)), -50),
    )

    held = {p.instrument.key for p in book}
    candidates: list[HedgeCandidate] = [
        HedgeCandidate(
            LinearInstrument(parsed.symbol, parsed.symbol),
            Quote(spot - 0.5, spot + 0.5, 1_000_000, 1_000_000, as_of),
        )
    ]
    for item in near:
        contract = item.contract
        days = calendar_years(as_of, contract.expiry) * 365.0
        moneyness = abs(np.log(contract.strike / forward))
        if (
            20.0 <= days <= 90.0
            and moneyness <= 2.0 * width
            and item.quote.bid > 0.0
            and contract.strike % 25 == 0
            and contract.key not in held
        ):
            sized = Quote(item.quote.bid, item.quote.ask, 200, 200, as_of)
            candidates.append(HedgeCandidate(contract, sized, fee_per_unit=0.7))

    targets = tuple(
        GreekTarget(parsed.symbol, greek)
        for greek in (GreekName.DELTA, GreekName.GAMMA, GreekName.VEGA)
    )
    scenarios = (*HISTORICAL_STRESSES, *HYPOTHETICAL_SHOCKS)
    config = HedgeConfig(
        risk_aversion=1.0,
        max_legs=6,
        max_quote_age=timedelta(days=30),
    )
    problem = HedgeProblem(book, tuple(candidates), targets, scenarios, config)

    started = time.perf_counter()
    proposal = solve_hedge(problem, snapshot, run_stability=True)
    elapsed = time.perf_counter() - started

    hedged = list(proposal.hedged_book(book))
    ours = value_book(hedged, snapshot).by_underlying[parsed.symbol]
    theirs = quantlib_greeks(hedged, snapshot, parsed.symbol)
    comparison = {
        name: {
            "ours": float(getattr(ours, name)),
            "quantlib": theirs[name],
            "difference": float(getattr(ours, name)) - theirs[name],
        }
        for name in ("value", "delta_shares", "gamma_shares", "vega_1pt")
    }

    frontier = hedge_frontier(
        HedgeProblem(book, tuple(candidates), targets, scenarios, config),
        snapshot,
        risk_aversions=(0.05, 0.2, 1.0, 5.0),
    )

    summary = {
        "symbol": parsed.symbol,
        "as_of": as_of.isoformat(),
        "spot": spot,
        "book": [p.instrument.describe() + f" x {p.quantity}" for p in book],
        "n_candidates": len(candidates),
        "n_excluded": len(proposal.excluded),
        "status": proposal.status.value,
        "seconds_to_solve_with_stability": elapsed,
        "legs": [leg.rationale for leg in proposal.legs],
        "transaction_cost": proposal.transaction_cost,
        "net_premium": proposal.net_premium,
        "objective": proposal.objective,
        "objective_lower_bound": proposal.objective_lower_bound,
        "residuals": [asdict(line) for line in proposal.residuals],
        "verification_passed": proposal.verification.passed,
        "quantlib_check": comparison,
        "stability": None
        if proposal.stability is None
        else {
            "condition_number": proposal.stability.condition_number,
            "lot_change_ratio": proposal.stability.lot_change_ratio,
            "regret_ratio": proposal.stability.regret_ratio,
            "is_stable": proposal.stability.is_stable,
            "notes": list(proposal.stability.notes),
        },
        "worst_stress_before": min(proposal.scenario_pnl_before.values()),
        "worst_stress_after": min(proposal.scenario_pnl_after.values()),
        "stress": {
            name: {
                "before": proposal.scenario_pnl_before[name],
                "after": proposal.scenario_pnl_after[name],
            }
            for name in proposal.scenario_pnl_before
        },
        "taylor_error_after_worst": max(
            (abs(c.error) for c in taylor_vs_full(hedged, snapshot, scenarios)), default=0.0
        ),
        "frontier": [
            {
                "label": point.label,
                "transaction_cost": point.transaction_cost,
                "n_legs": point.n_legs,
                "penalized_residual": point.penalized_residual,
                "worst_scenario_pnl": point.worst_scenario_pnl,
            }
            for point in frontier
        ],
        "reg_t_estimate_before": reg_t_short_option_estimate(
            book, snapshot, {parsed.symbol: UnderlyingClass.BROAD_INDEX}
        ).requirement,
        "reg_t_estimate_after": reg_t_short_option_estimate(
            hedged, snapshot, {parsed.symbol: UnderlyingClass.BROAD_INDEX}
        ).requirement,
        "expiry_risks": [asdict(flag) for flag in expiry_risks(hedged, snapshot)],
        "issues": list(proposal.issues),
        "manifest": {
            "code_version": proposal.manifest.code_version,
            "market_fingerprint": proposal.manifest.market_fingerprint,
            "problem_fingerprint": proposal.manifest.problem_fingerprint,
        },
    }
    assert book_scenario_pnl(book, snapshot, scenarios).size == len(scenarios)
    text = json.dumps(summary, indent=2, default=float)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
