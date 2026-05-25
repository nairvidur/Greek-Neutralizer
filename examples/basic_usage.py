#!/usr/bin/env python3
"""
examples/basic_usage.py
════════════════════════════════════════════════════════════════
Basic usage of the Greeks Neutralizer engine.

Demonstrates:
  1. Computing Greeks for a single option
  2. Building a multi-position portfolio
  3. Aggregating portfolio Greeks
  4. Solving for Δ-Γ-V neutral hedge quantities
  5. Verifying the solution

Run from project root:
    python examples/basic_usage.py
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
    format_greeks_table,
)


def main():
    # ─────────────────────────────────────────────────
    # 1. Single Option Greeks
    # ─────────────────────────────────────────────────
    print("=" * 60)
    print("  EXAMPLE 1: Single Option Greeks")
    print("=" * 60)

    g = compute_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.30, option_type="call")
    print(f"\n  ATM Call (S=100, K=100, T=0.25y, σ=30%)")
    print(f"  Price:  ${g.price:.4f}")
    print(f"  Delta:  {g.delta:+.6f}")
    print(f"  Gamma:  {g.gamma:+.6f}")
    print(f"  Vega:   {g.vega:+.6f}  (per 1% IV)")
    print(f"  Theta:  {g.theta:+.6f}  (per day)")
    print(f"  Rho:    {g.rho:+.6f}  (per 1% rate)")

    # ─────────────────────────────────────────────────
    # 2. Multi-Position Portfolio
    # ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  EXAMPLE 2: Multi-Position Portfolio")
    print("=" * 60)

    S, r = 100.0, 0.05

    portfolio = [
        OptionContract("call", strike=100, expiry=0.25, iv=0.30, quantity=10, side="long"),
        OptionContract("put",  strike=95,  expiry=0.25, iv=0.32, quantity=5,  side="short"),
        OptionContract("call", strike=110, expiry=0.50, iv=0.25, quantity=8,  side="long"),
    ]

    positions, aggregate = aggregate_portfolio_greeks(S, r, portfolio)

    print(f"\n  Portfolio ({len(portfolio)} positions):")
    for i, pos in enumerate(positions):
        c = pos.contract
        print(f"    #{i+1}: {c.side.upper()} {c.quantity}× {c.strike}{c.option_type[0].upper()} @{c.expiry}y σ={c.iv:.0%}")
        print(f"         {format_greeks_table(pos.position_greeks)}")

    print(f"\n  Aggregate:")
    print(f"  {format_greeks_table(aggregate, 'TOTAL')}")

    # ─────────────────────────────────────────────────
    # 3. Neutralization
    # ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  EXAMPLE 3: Δ-Γ-V Neutralization")
    print("=" * 60)

    hedges = [
        OptionContract("put",  strike=95,  expiry=0.50, iv=0.32),
        OptionContract("call", strike=105, expiry=0.50, iv=0.28),
        OptionContract("call", strike=100, expiry=0.75, iv=0.30),
    ]

    result = solve_neutralization(S, r, portfolio, hedges)

    if result.success:
        print("\n  ✓ Solution found!")
        print("\n  Hedge quantities:")
        for i, qty in enumerate(result.quantities):
            h = hedges[i]
            action = "BUY " if qty >= 0 else "SELL"
            print(f"    H{i+1}: {action} {abs(qty):.4f} × {h.strike}{h.option_type[0].upper()} @{h.expiry}y")

        print(f"\n  Net Greeks (after hedging):")
        net = result.net_greeks
        print(f"    Delta: {net.delta:+.10f}  {'✓ NEUTRAL' if abs(net.delta) < 1e-6 else '✗'}")
        print(f"    Gamma: {net.gamma:+.10f}  {'✓ NEUTRAL' if abs(net.gamma) < 1e-6 else '✗'}")
        print(f"    Vega:  {net.vega:+.10f}  {'✓ NEUTRAL' if abs(net.vega) < 1e-6 else '✗'}")
        print(f"    Theta: {net.theta:+.6f}   (daily hedge cost)")
    else:
        print(f"\n  ✗ {result.error_message}")

    print()


if __name__ == "__main__":
    main()
