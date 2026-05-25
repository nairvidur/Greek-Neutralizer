# 📊 Options Greeks Neutralizer

**A multi-position Delta-Gamma-Vega neutralization engine for European options portfolios.**

Build hedged portfolios that are simultaneously neutral to directional moves (Δ), convexity (Γ), and volatility (V) — the three most critical risk dimensions in options trading.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-40%20passed-brightgreen.svg)](#testing)

---

## 🎯 What This Does

Options portfolios carry **multi-dimensional risk** that a single delta hedge can't capture. A portfolio hedged for directional moves remains exposed to large spot swings (gamma risk) and volatility shifts (vega risk).

This engine solves the **simultaneous neutralization** of all three Greeks by computing exact hedge quantities across three hedging instruments, supporting **arbitrarily complex multi-leg portfolios**.

```
Portfolio_Δ + q₁·Δ₁ + q₂·Δ₂ + q₃·Δ₃ = 0   (directional neutrality)
Portfolio_Γ + q₁·Γ₁ + q₂·Γ₂ + q₃·Γ₃ = 0   (convexity neutrality)
Portfolio_V + q₁·V₁ + q₂·V₂ + q₃·V₃ = 0   (volatility neutrality)
```

---

## ✨ Key Features

- **Multi-position support** — model any number of option legs (spreads, condors, butterflies, custom structures)
- **Full Greek suite** — Delta, Gamma, Vega, Theta, Rho, and theoretical price via Black-Scholes
- **Automatic aggregation** — per-position and portfolio-level Greeks
- **3×3 neutralization solver** — Gaussian elimination with partial pivoting
- **Zero dependencies** — pure Python, no NumPy/SciPy required
- **Interactive CLI** — enter positions interactively or use programmatic API
- **Comprehensive test suite** — 40+ tests covering math, aggregation, and end-to-end neutralization

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/greeks-neutralizer.git
cd greeks-neutralizer
```

No dependencies needed for core functionality. For development:

```bash
pip install -r requirements.txt
```

### Run the Example

```bash
python main.py
```

Output:
```
═══════════════════════════════════════════════════
  GREEKS NEUTRALIZER — MULTI-POSITION ENGINE
═══════════════════════════════════════════════════

  Market Parameters: S = 100.0, r = 0.05
  Portfolio Legs:    3
  Hedge Instruments: 3

─── INDIVIDUAL POSITION GREEKS ─────────────────────
  #1  LONG  10× 100C @0.25y σ=30%
  #2  SHORT  5× 95P  @0.25y σ=32%
  #3  LONG   8× 110C @0.50y σ=25%

─── VERIFICATION — NET PORTFOLIO GREEKS ────────────
    ✓ Delta neutral: YES
    ✓ Gamma neutral: YES
    ✓ Vega  neutral: YES
```

### Interactive Mode

```bash
python main.py --interactive
```

Guides you through entering positions and hedge instruments step by step.

---

## 📖 Usage as a Library

```python
from src.greeks_engine import (
    OptionContract, compute_greeks,
    aggregate_portfolio_greeks, solve_neutralization
)

# Compute Greeks for a single option
greeks = compute_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.30, option_type="call")
print(f"Delta: {greeks.delta:.4f}, Gamma: {greeks.gamma:.4f}, Vega: {greeks.vega:.4f}")

# Build a multi-position portfolio
portfolio = [
    OptionContract("call", strike=100, expiry=0.25, iv=0.30, quantity=10, side="long"),
    OptionContract("put",  strike=95,  expiry=0.25, iv=0.32, quantity=5,  side="short"),
    OptionContract("call", strike=110, expiry=0.50, iv=0.25, quantity=8,  side="long"),
]

# Aggregate portfolio Greeks
positions, aggregate = aggregate_portfolio_greeks(100, 0.05, portfolio)
print(f"Portfolio Delta: {aggregate.delta:+.4f}")

# Solve for Δ-Γ-V neutralization
hedges = [
    OptionContract("put",  strike=95,  expiry=0.50, iv=0.32),
    OptionContract("call", strike=105, expiry=0.50, iv=0.28),
    OptionContract("call", strike=100, expiry=0.75, iv=0.30),
]

result = solve_neutralization(100, 0.05, portfolio, hedges)
if result.success:
    for i, qty in enumerate(result.quantities):
        action = "BUY" if qty >= 0 else "SELL"
        print(f"  H{i+1}: {action} {abs(qty):.4f} contracts")
```

---

## 📁 Project Structure

```
greeks-neutralizer/
├── main.py                          # CLI application (default + interactive mode)
├── setup.py                         # Package configuration
├── requirements.txt                 # Dependencies
├── LICENSE                          # MIT License
├── .gitignore
│
├── src/
│   ├── __init__.py                  # Package exports
│   └── greeks_engine.py             # Core engine: BS Greeks, solver, aggregation
│
├── tests/
│   └── test_greeks_engine.py        # 40+ unit tests
│
├── examples/
│   ├── basic_usage.py               # Getting started examples
│   └── advanced_analysis.py         # Sensitivity analysis, IV scenarios, iron condor
│
└── report/
    ├── generate_report.py           # Word document generator
    └── Project_Report.docx          # Comprehensive project report
```

---

## 🔬 Mathematical Framework

### Black-Scholes Greeks

| Greek | Formula | Unit |
|-------|---------|------|
| **Delta (Δ)** | N(d₁) for calls, N(d₁)−1 for puts | per $1 spot move |
| **Gamma (Γ)** | n(d₁) / (Sσ√T) | per $1 spot move |
| **Vega (V)** | Sn(d₁)√T / 100 | per 1% IV move |
| **Theta (Θ)** | −[Sn(d₁)σ/(2√T) + rKe⁻ʳᵀN(d₂)] / 365 | per calendar day |
| **Rho (ρ)** | KTe⁻ʳᵀN(d₂) / 100 | per 1% rate move |

### Neutralization

The 3×3 system **Aq = −b** is solved via **Gaussian elimination with partial pivoting**:

```
A = | Δ₁  Δ₂  Δ₃ |      b = | Portfolio_Δ |
    | Γ₁  Γ₂  Γ₃ |          | Portfolio_Γ |
    | V₁  V₂  V₃ |          | Portfolio_V |
```

A unique solution exists iff the three hedging instruments have **linearly independent** Greek profiles (different strikes, expirations, or types).

---

## 🧪 Testing

```bash
# With pytest
pytest tests/ -v

# Without pytest
python tests/test_greeks_engine.py
```

Test coverage includes:
- Normal CDF/PDF accuracy and symmetry
- Put-call parity verification
- Greek boundary conditions (deep ITM/OTM, expired options)
- Long/short offset cancellation
- Linear solver correctness and singular matrix detection
- End-to-end neutralization with 1, 3, and 5 position portfolios

---

## 📋 API Reference

### `compute_greeks(S, K, T, r, sigma, option_type) → Greeks`

Black-Scholes Greeks for a single European option.

### `aggregate_portfolio_greeks(S, r, contracts) → (positions, aggregate)`

Compute and aggregate Greeks across multiple option positions.

### `solve_neutralization(S, r, portfolio, hedges) → NeutralizationResult`

Solve for Δ-Γ-V neutral hedge quantities. Requires exactly 3 hedging instruments.

### `solve_linear_system(A, b) → list | None`

General n×n solver using Gaussian elimination with partial pivoting.

### Data Classes

- **`OptionContract`** — option parameters (type, strike, expiry, IV, quantity, side)
- **`Greeks`** — container with arithmetic operations (`+`, `.scale()`)
- **`NeutralizationResult`** — solution, net Greeks, error info

---

## ⚠️ Limitations

- **European options only** — no early exercise (American option) support
- **Static hedge** — valid at computation instant; requires rebalancing
- **No transaction costs** — bid-ask spreads and commissions not modeled
- **Fractional contracts** — solver returns real-valued quantities
- **No higher-order Greeks** — vanna, volga, charm not neutralized

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
