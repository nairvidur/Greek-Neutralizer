#!/usr/bin/env python3
"""
main.py
════════════════════════════════════════════════════════════════
Greeks Neutralizer — Interactive CLI Application

Demonstrates multi-position Delta-Gamma-Vega neutralization
with a rich terminal interface.

Usage:
    python main.py              # Run with default example portfolio
    python main.py --interactive  # Enter positions interactively

════════════════════════════════════════════════════════════════
"""

import argparse
import sys
from typing import List

from src.greeks_engine import (
    OptionContract,
    compute_greeks,
    aggregate_portfolio_greeks,
    solve_neutralization,
    format_greeks_table,
)


# ─────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────

SEPARATOR = "═" * 64
THIN_SEP = "─" * 64

def banner(text: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {text}")
    print(SEPARATOR)

def section(text: str) -> None:
    print(f"\n{THIN_SEP}")
    print(f"  {text}")
    print(THIN_SEP)


def print_position(idx: int, contract: OptionContract, greeks) -> None:
    side_str = contract.side.upper()
    type_str = contract.option_type[0].upper()
    print(
        f"  #{idx+1}  {side_str:<5} {contract.quantity:>3}× "
        f"{contract.strike}{type_str} @{contract.expiry}y σ={contract.iv:.0%}"
    )
    print(format_greeks_table(greeks, label=""))


def print_hedge(idx: int, contract: OptionContract, greeks) -> None:
    type_str = contract.option_type[0].upper()
    print(
        f"  H{idx+1}  {contract.strike}{type_str} "
        f"@{contract.expiry}y σ={contract.iv:.0%}"
    )
    print(format_greeks_table(greeks, label=""))


# ─────────────────────────────────────────────────────────────
# DEFAULT EXAMPLE PORTFOLIO
# ─────────────────────────────────────────────────────────────

def default_portfolio() -> tuple:
    """Return a sample portfolio, hedge set, and market params."""
    S = 100.0
    r = 0.05

    portfolio = [
        OptionContract("call", strike=100, expiry=0.25, iv=0.30, quantity=10, side="long"),
        OptionContract("put",  strike=95,  expiry=0.25, iv=0.32, quantity=5,  side="short"),
        OptionContract("call", strike=110, expiry=0.50, iv=0.25, quantity=8,  side="long"),
    ]

    hedges = [
        OptionContract("put",  strike=95,  expiry=0.50, iv=0.32),
        OptionContract("call", strike=105, expiry=0.50, iv=0.28),
        OptionContract("call", strike=100, expiry=0.75, iv=0.30),
    ]

    return S, r, portfolio, hedges


# ─────────────────────────────────────────────────────────────
# INTERACTIVE INPUT
# ─────────────────────────────────────────────────────────────

def get_float(prompt: str, default: float = None) -> float:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"    {prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("    ⚠ Please enter a valid number.")


def get_choice(prompt: str, choices: list, default: str = None) -> str:
    choice_str = "/".join(choices)
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"    {prompt} ({choice_str}){suffix}: ").strip().lower()
        if not raw and default:
            return default
        if raw in choices:
            return raw
        print(f"    ⚠ Choose from: {choice_str}")


def get_int(prompt: str, default: int = None) -> int:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"    {prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            print("    ⚠ Please enter a valid integer.")


def interactive_input() -> tuple:
    """Gather portfolio and hedge parameters interactively."""
    banner("MARKET PARAMETERS")
    S = get_float("Spot price (S)", 100.0)
    r = get_float("Risk-free rate (r)", 0.05)

    # Portfolio positions
    section("PORTFOLIO POSITIONS")
    print("  Enter your option positions. Type 'done' when finished.\n")

    portfolio: List[OptionContract] = []
    idx = 1
    while True:
        print(f"  ── Position #{idx} ──")
        raw = input("    Add position? (yes/done) [yes]: ").strip().lower()
        if raw == "done":
            if not portfolio:
                print("    ⚠ Need at least one position.")
                continue
            break

        opt_type = get_choice("Type", ["call", "put"], "call")
        side = get_choice("Side", ["long", "short"], "long")
        strike = get_float("Strike (K)", 100.0)
        expiry = get_float("Expiry (T years)", 0.25)
        iv = get_float("IV (σ, e.g. 0.30)", 0.30)
        qty = get_int("Quantity", 10)

        portfolio.append(OptionContract(opt_type, strike, expiry, iv, qty, side))
        print(f"    ✓ Added: {side.upper()} {qty}× {strike}{opt_type[0].upper()} @{expiry}y σ={iv}")
        idx += 1
        print()

    # Hedging instruments (always 3)
    section("HEDGING INSTRUMENTS (need exactly 3)")
    hedges: List[OptionContract] = []
    for i in range(3):
        print(f"\n  ── Hedge #{i+1} ──")
        opt_type = get_choice("Type", ["call", "put"], "call" if i > 0 else "put")
        strike = get_float("Strike (K)", 95.0 + i * 5)
        expiry = get_float("Expiry (T years)", 0.50)
        iv = get_float("IV (σ)", 0.30)
        hedges.append(OptionContract(opt_type, strike, expiry, iv))
        print(f"    ✓ Added: {strike}{opt_type[0].upper()} @{expiry}y σ={iv}")

    return S, r, portfolio, hedges


# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────

def run(S: float, r: float, portfolio: List[OptionContract], hedges: List[OptionContract]) -> None:
    """Execute the full neutralization pipeline and display results."""

    banner("GREEKS NEUTRALIZER — MULTI-POSITION ENGINE")
    print(f"\n  Market Parameters: S = {S}, r = {r}")
    print(f"  Portfolio Legs:    {len(portfolio)}")
    print(f"  Hedge Instruments: {len(hedges)}")

    # Step 1: Position Greeks
    section("INDIVIDUAL POSITION GREEKS")
    positions, aggregate = aggregate_portfolio_greeks(S, r, portfolio)

    for i, pos in enumerate(positions):
        print_position(i, pos.contract, pos.position_greeks)
        print()

    # Step 2: Aggregate Greeks
    section("AGGREGATE PORTFOLIO GREEKS")
    print(format_greeks_table(aggregate, label="TOTAL"))
    print()

    # Step 3: Hedge instrument Greeks
    section("HEDGING INSTRUMENT GREEKS (per unit)")
    hedge_greeks = []
    for i, hc in enumerate(hedges):
        g = compute_greeks(S, hc.strike, hc.expiry, r, hc.iv, hc.option_type)
        hedge_greeks.append(g)
        print_hedge(i, hc, g)
        print()

    # Step 4: Solve
    section("SOLVING 3×3 LINEAR SYSTEM  [Δ, Γ, V]")
    result = solve_neutralization(S, r, portfolio, hedges)

    if not result.success:
        print(f"\n  ✗ FAILED: {result.error_message}")
        print("    Try instruments with more diverse strikes, expirations, or types.")
        return

    print("  ✓ Solution found!\n")

    # Step 5: Display quantities
    section("HEDGE QUANTITIES")
    for i, qty in enumerate(result.quantities):
        action = "BUY " if qty >= 0 else "SELL"
        type_str = hedges[i].option_type[0].upper()
        print(
            f"  H{i+1}  {action} {abs(qty):>10.4f} contracts of "
            f"{hedges[i].strike}{type_str} @{hedges[i].expiry}y"
        )

    # Step 6: Verification
    section("VERIFICATION — NET PORTFOLIO GREEKS")
    net = result.net_greeks
    print(format_greeks_table(net, label="NET"))
    print()

    checks = [
        ("Delta neutral", abs(net.delta) < 1e-6),
        ("Gamma neutral", abs(net.gamma) < 1e-6),
        ("Vega  neutral", abs(net.vega) < 1e-6),
    ]
    for name, passed in checks:
        symbol = "✓" if passed else "✗"
        status = "YES" if passed else "NO"
        print(f"    {symbol} {name}: {status}")

    print(f"\n    Theta (hedge cost): {net.theta:+.6f} per day")
    print(f"    Rho exposure:      {net.rho:+.6f} per 1% rate")

    print(f"\n{SEPARATOR}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Options Greeks Neutralizer — Multi-Position Δ-Γ-V Hedging Engine"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enter positions interactively instead of using the default example."
    )
    args = parser.parse_args()

    if args.interactive:
        S, r, portfolio, hedges = interactive_input()
    else:
        S, r, portfolio, hedges = default_portfolio()

    run(S, r, portfolio, hedges)


if __name__ == "__main__":
    main()
