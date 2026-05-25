#!/usr/bin/env python3
"""
examples/advanced_analysis.py
════════════════════════════════════════════════════════════════
Advanced usage: sensitivity analysis and scenario comparison.

Demonstrates:
  1. Greeks sensitivity across a range of spot prices
  2. Comparing hedge effectiveness across IV scenarios
  3. Multi-leg strategy analysis (iron condor)

Run from project root:
    python examples/advanced_analysis.py
════════════════════════════════════════════════════════════════
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.greeks_engine import (
    OptionContract,
    compute_greeks,
    aggregate_portfolio_greeks,
    solve_neutralization,
)


def sensitivity_analysis():
    """Show how portfolio delta changes across spot prices."""
    print("=" * 64)
    print("  SENSITIVITY ANALYSIS: Portfolio Delta vs. Spot Price")
    print("=" * 64)

    r = 0.05
    portfolio = [
        OptionContract("call", strike=100, expiry=0.25, iv=0.30, quantity=10, side="long"),
        OptionContract("put",  strike=95,  expiry=0.25, iv=0.32, quantity=5,  side="short"),
    ]

    print(f"\n  {'Spot':>6}  {'Delta':>10}  {'Gamma':>10}  {'Vega':>10}  {'Theta':>10}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for spot in range(80, 121, 2):
        S = float(spot)
        _, agg = aggregate_portfolio_greeks(S, r, portfolio)
        print(f"  {S:6.0f}  {agg.delta:+10.4f}  {agg.gamma:+10.4f}  {agg.vega:+10.4f}  {agg.theta:+10.4f}")


def iv_scenario_comparison():
    """Compare hedge quantities across low/mid/high IV environments."""
    print(f"\n{'=' * 64}")
    print("  IV SCENARIO COMPARISON: Hedge Quantities")
    print("=" * 64)

    S, r = 100.0, 0.05
    scenarios = {
        "Low IV  (15%)": 0.15,
        "Mid IV  (30%)": 0.30,
        "High IV (50%)": 0.50,
    }

    for label, iv in scenarios.items():
        portfolio = [
            OptionContract("call", 100, 0.25, iv, quantity=10, side="long"),
        ]
        hedges = [
            OptionContract("put",  95,  0.50, iv),
            OptionContract("call", 105, 0.50, iv),
            OptionContract("call", 100, 0.75, iv),
        ]
        result = solve_neutralization(S, r, portfolio, hedges)
        if result.success:
            q = result.quantities
            print(f"\n  {label}:")
            print(f"    Portfolio Δ={result.portfolio_greeks.delta:+.4f}  "
                  f"Γ={result.portfolio_greeks.gamma:+.4f}  "
                  f"V={result.portfolio_greeks.vega:+.4f}")
            print(f"    Hedge: q₁={q[0]:+.2f}  q₂={q[1]:+.2f}  q₃={q[2]:+.2f}")
            print(f"    Net Θ = {result.net_greeks.theta:+.6f}/day")


def iron_condor_analysis():
    """Analyze an iron condor and neutralize its residual Greeks."""
    print(f"\n{'=' * 64}")
    print("  IRON CONDOR: Multi-Leg Strategy Analysis")
    print("=" * 64)

    S, r = 100.0, 0.05

    # Iron condor: sell put spread + sell call spread
    iron_condor = [
        OptionContract("put",  strike=90,  expiry=0.25, iv=0.28, quantity=10, side="short"),  # Sell OTM put
        OptionContract("put",  strike=85,  expiry=0.25, iv=0.30, quantity=10, side="long"),   # Buy lower put (protection)
        OptionContract("call", strike=110, expiry=0.25, iv=0.26, quantity=10, side="short"),  # Sell OTM call
        OptionContract("call", strike=115, expiry=0.25, iv=0.28, quantity=10, side="long"),   # Buy higher call (protection)
    ]

    positions, aggregate = aggregate_portfolio_greeks(S, r, iron_condor)

    print(f"\n  Iron Condor Structure (S={S}, {len(iron_condor)} legs):")
    for i, pos in enumerate(positions):
        c = pos.contract
        print(f"    {c.side.upper():>5} {c.quantity}× {c.strike}{c.option_type[0].upper()} @{c.expiry}y σ={c.iv:.0%}")

    print(f"\n  Aggregate Greeks:")
    print(f"    Δ = {aggregate.delta:+.6f}")
    print(f"    Γ = {aggregate.gamma:+.6f}")
    print(f"    V = {aggregate.vega:+.6f}")
    print(f"    Θ = {aggregate.theta:+.6f}")

    # Neutralize
    hedges = [
        OptionContract("put",  strike=95,  expiry=0.50, iv=0.30),
        OptionContract("call", strike=105, expiry=0.50, iv=0.28),
        OptionContract("call", strike=100, expiry=0.75, iv=0.30),
    ]

    result = solve_neutralization(S, r, iron_condor, hedges)
    if result.success:
        print(f"\n  Neutralization Solution:")
        for i, qty in enumerate(result.quantities):
            h = hedges[i]
            action = "BUY " if qty >= 0 else "SELL"
            print(f"    H{i+1}: {action} {abs(qty):.4f} × {h.strike}{h.option_type[0].upper()} @{h.expiry}y")

        print(f"\n  Verification:")
        print(f"    Net Δ = {result.net_greeks.delta:+.10f}  {'✓' if abs(result.net_greeks.delta) < 1e-6 else '✗'}")
        print(f"    Net Γ = {result.net_greeks.gamma:+.10f}  {'✓' if abs(result.net_greeks.gamma) < 1e-6 else '✗'}")
        print(f"    Net V = {result.net_greeks.vega:+.10f}  {'✓' if abs(result.net_greeks.vega) < 1e-6 else '✗'}")
        print(f"    Net Θ = {result.net_greeks.theta:+.6f}")


if __name__ == "__main__":
    sensitivity_analysis()
    iv_scenario_comparison()
    iron_condor_analysis()
    print()
