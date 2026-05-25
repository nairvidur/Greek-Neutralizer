"""
greeks_engine.py
════════════════════════════════════════════════════════════════
Black-Scholes Option Pricing & Greeks Calculation Engine

Provides closed-form solutions for European option pricing
and first/second-order risk sensitivities (Greeks), plus a
linear algebra solver for multi-Greek neutralization.

Author:  Greeks Neutralizer Project
License: MIT
════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class OptionContract:
    """Represents a single European option contract."""
    option_type: str          # "call" or "put"
    strike: float             # Strike price (K)
    expiry: float             # Time to expiration in years (T)
    iv: float                 # Implied volatility (σ, annualized)
    quantity: int = 1         # Number of contracts
    side: str = "long"        # "long" or "short"

    def direction_sign(self) -> int:
        """Returns +1 for long, -1 for short."""
        return 1 if self.side == "long" else -1


@dataclass
class Greeks:
    """Container for Black-Scholes Greeks and theoretical price."""
    delta: float = 0.0       # ∂V/∂S          (per $1 spot move)
    gamma: float = 0.0       # ∂²V/∂S²        (per $1 spot move)
    vega: float = 0.0        # ∂V/∂σ / 100    (per 1% IV move)
    theta: float = 0.0       # ∂V/∂t / 365    (per calendar day)
    rho: float = 0.0         # ∂V/∂r / 100    (per 1% rate move)
    price: float = 0.0       # BS theoretical price

    def __add__(self, other: "Greeks") -> "Greeks":
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
            rho=self.rho + other.rho,
            price=self.price + other.price,
        )

    def scale(self, factor: float) -> "Greeks":
        """Return a new Greeks object scaled by the given factor."""
        return Greeks(
            delta=self.delta * factor,
            gamma=self.gamma * factor,
            vega=self.vega * factor,
            theta=self.theta * factor,
            rho=self.rho * factor,
            price=self.price * factor,
        )

    def __repr__(self) -> str:
        return (
            f"Greeks(Δ={self.delta:+.6f}, Γ={self.gamma:+.6f}, "
            f"V={self.vega:+.6f}, Θ={self.theta:+.6f}, "
            f"ρ={self.rho:+.6f}, $={self.price:+.4f})"
        )


@dataclass
class NeutralizationResult:
    """Result of the Delta-Gamma-Vega neutralization solver."""
    success: bool
    quantities: Optional[List[float]] = None
    net_greeks: Optional[Greeks] = None
    portfolio_greeks: Optional[Greeks] = None
    hedge_greeks: Optional[List[Greeks]] = None
    error_message: str = ""


@dataclass
class Position:
    """A portfolio position: contract + computed Greeks."""
    contract: OptionContract
    per_unit_greeks: Greeks = field(default_factory=Greeks)
    position_greeks: Greeks = field(default_factory=Greeks)


# ─────────────────────────────────────────────────────────────
# STATISTICAL FUNCTIONS
# ─────────────────────────────────────────────────────────────

def normal_cdf(x: float) -> float:
    """
    Cumulative standard normal distribution.

    Uses the Abramowitz & Stegun (1964) rational approximation.
    Maximum error: |ε| < 1.5 × 10⁻⁷

    Parameters
    ----------
    x : float
        Standard normal variate.

    Returns
    -------
    float
        P(Z ≤ x)
    """
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = -1.0 if x < 0 else 1.0
    ax = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-ax * ax)
    return 0.5 * (1.0 + sign * y)


def normal_pdf(x: float) -> float:
    """
    Standard normal probability density function.

    Parameters
    ----------
    x : float
        Standard normal variate.

    Returns
    -------
    float
        φ(x)
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ─────────────────────────────────────────────────────────────
# BLACK-SCHOLES ENGINE
# ─────────────────────────────────────────────────────────────

def compute_d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> Tuple[float, float]:
    """
    Compute the Black-Scholes d₁ and d₂ parameters.

    Parameters
    ----------
    S     : Current spot price
    K     : Strike price
    T     : Time to expiration (years)
    r     : Risk-free rate (annualized, continuous)
    sigma : Implied volatility (annualized)

    Returns
    -------
    (d1, d2) : tuple of float
    """
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """
    Compute Black-Scholes theoretical option price.

    Parameters
    ----------
    S           : Spot price
    K           : Strike price
    T           : Time to expiry (years)
    r           : Risk-free rate
    sigma       : Implied volatility
    option_type : "call" or "put"

    Returns
    -------
    float : Theoretical option price
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    if sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    exp_rT = math.exp(-r * T)

    if option_type == "call":
        return S * normal_cdf(d1) - K * exp_rT * normal_cdf(d2)
    else:
        return K * exp_rT * normal_cdf(-d2) - S * normal_cdf(-d1)


def compute_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> Greeks:
    """
    Compute all Black-Scholes Greeks for a European option.

    Returns
    -------
    Greeks
        delta  : ∂V/∂S          (per $1 move in spot)
        gamma  : ∂²V/∂S²        (per $1 move in spot)
        vega   : ∂V/∂σ / 100    (per 1% move in IV)
        theta  : ∂V/∂t / 365    (per calendar day)
        rho    : ∂V/∂r / 100    (per 1% move in rate)
        price  : BS theoretical price
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return Greeks()

    sqrt_T = math.sqrt(T)
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    Nd1 = normal_cdf(d1)
    Nd2 = normal_cdf(d2)
    nd1 = normal_pdf(d1)
    exp_rT = math.exp(-r * T)

    # Delta
    delta = Nd1 if option_type == "call" else Nd1 - 1.0

    # Gamma (same for calls and puts)
    gamma = nd1 / (S * sigma * sqrt_T)

    # Vega (per 1% IV move)
    vega = S * nd1 * sqrt_T / 100.0

    # Theta (per calendar day)
    if option_type == "call":
        theta = (-(S * nd1 * sigma) / (2.0 * sqrt_T) - r * K * exp_rT * Nd2) / 365.0
    else:
        theta = (-(S * nd1 * sigma) / (2.0 * sqrt_T) + r * K * exp_rT * normal_cdf(-d2)) / 365.0

    # Rho (per 1% rate move)
    if option_type == "call":
        rho = K * T * exp_rT * Nd2 / 100.0
    else:
        rho = -K * T * exp_rT * normal_cdf(-d2) / 100.0

    # Price
    price = black_scholes_price(S, K, T, r, sigma, option_type)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho, price=price)


# ─────────────────────────────────────────────────────────────
# PORTFOLIO AGGREGATION
# ─────────────────────────────────────────────────────────────

def compute_position_greeks(
    S: float, r: float, contract: OptionContract
) -> Tuple[Greeks, Greeks]:
    """
    Compute per-unit and scaled position Greeks for an option contract.

    Parameters
    ----------
    S        : Spot price
    r        : Risk-free rate
    contract : OptionContract with strike, expiry, IV, quantity, side

    Returns
    -------
    (per_unit_greeks, position_greeks) : tuple of Greeks
        per_unit_greeks: Greeks for a single contract
        position_greeks: Greeks scaled by quantity × direction sign
    """
    per_unit = compute_greeks(S, contract.strike, contract.expiry, r, contract.iv, contract.option_type)
    factor = contract.quantity * contract.direction_sign()
    position = per_unit.scale(factor)
    return per_unit, position


def aggregate_portfolio_greeks(
    S: float, r: float, contracts: List[OptionContract]
) -> Tuple[List[Position], Greeks]:
    """
    Compute and aggregate Greeks across multiple option positions.

    Parameters
    ----------
    S         : Spot price
    r         : Risk-free rate
    contracts : List of OptionContract objects

    Returns
    -------
    (positions, aggregate) : tuple
        positions: List of Position objects with individual Greeks
        aggregate: Greeks summed across all positions
    """
    positions = []
    aggregate = Greeks()

    for contract in contracts:
        per_unit, scaled = compute_position_greeks(S, r, contract)
        positions.append(Position(contract=contract, per_unit_greeks=per_unit, position_greeks=scaled))
        aggregate = aggregate + scaled

    return positions, aggregate


# ─────────────────────────────────────────────────────────────
# LINEAR ALGEBRA SOLVER
# ─────────────────────────────────────────────────────────────

def solve_linear_system(
    A: List[List[float]], b: List[float], tol: float = 1e-12
) -> Optional[List[float]]:
    """
    Solve the linear system Ax = b using Gaussian elimination
    with partial pivoting.

    Parameters
    ----------
    A   : n×n coefficient matrix (list of lists)
    b   : n-element right-hand side vector
    tol : Singularity tolerance for pivot magnitude

    Returns
    -------
    list[float] | None
        Solution vector x, or None if the matrix is singular.
    """
    n = len(A)
    # Build augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col
        max_val = abs(M[col][col])
        for row in range(col + 1, n):
            if abs(M[row][col]) > max_val:
                max_val = abs(M[row][col])
                max_row = row

        if max_val < tol:
            return None  # Singular matrix

        # Swap rows
        if max_row != col:
            M[col], M[max_row] = M[max_row], M[col]

        # Eliminate below
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]

    return x


# ─────────────────────────────────────────────────────────────
# NEUTRALIZATION SOLVER
# ─────────────────────────────────────────────────────────────

def solve_neutralization(
    S: float,
    r: float,
    portfolio_contracts: List[OptionContract],
    hedge_contracts: List[OptionContract],
) -> NeutralizationResult:
    """
    Solve for Delta-Gamma-Vega neutral hedge quantities.

    Given a portfolio of option positions and exactly 3 hedging
    instruments, solves the 3×3 linear system:

        Aq = -b

    where A is the matrix of per-unit hedge Greeks [Δ, Γ, V]
    and b is the aggregate portfolio Greeks vector.

    Parameters
    ----------
    S                   : Current spot price
    r                   : Risk-free rate
    portfolio_contracts : List of OptionContract (the portfolio)
    hedge_contracts     : List of exactly 3 OptionContract (hedging instruments)

    Returns
    -------
    NeutralizationResult
        success    : Whether a valid solution was found
        quantities : Hedge quantities [q₁, q₂, q₃]
        net_greeks : Portfolio Greeks after hedging
        ...
    """
    if len(hedge_contracts) != 3:
        return NeutralizationResult(
            success=False,
            error_message=f"Exactly 3 hedging instruments required, got {len(hedge_contracts)}."
        )

    # Aggregate portfolio Greeks
    positions, portfolio_greeks = aggregate_portfolio_greeks(S, r, portfolio_contracts)

    # Compute per-unit hedge Greeks
    hedge_greeks_list = []
    for hc in hedge_contracts:
        g = compute_greeks(S, hc.strike, hc.expiry, r, hc.iv, hc.option_type)
        hedge_greeks_list.append(g)

    # Build 3×3 coefficient matrix
    A = [
        [hedge_greeks_list[0].delta, hedge_greeks_list[1].delta, hedge_greeks_list[2].delta],
        [hedge_greeks_list[0].gamma, hedge_greeks_list[1].gamma, hedge_greeks_list[2].gamma],
        [hedge_greeks_list[0].vega,  hedge_greeks_list[1].vega,  hedge_greeks_list[2].vega],
    ]
    b = [-portfolio_greeks.delta, -portfolio_greeks.gamma, -portfolio_greeks.vega]

    quantities = solve_linear_system(A, b)

    if quantities is None:
        return NeutralizationResult(
            success=False,
            portfolio_greeks=portfolio_greeks,
            hedge_greeks=hedge_greeks_list,
            error_message="Coefficient matrix is singular. Hedging instruments are linearly dependent."
        )

    # Compute net Greeks after hedging
    net = Greeks(
        delta=portfolio_greeks.delta,
        gamma=portfolio_greeks.gamma,
        vega=portfolio_greeks.vega,
        theta=portfolio_greeks.theta,
        rho=portfolio_greeks.rho,
        price=portfolio_greeks.price,
    )
    for i, qty in enumerate(quantities):
        net.delta += qty * hedge_greeks_list[i].delta
        net.gamma += qty * hedge_greeks_list[i].gamma
        net.vega  += qty * hedge_greeks_list[i].vega
        net.theta += qty * hedge_greeks_list[i].theta
        net.rho   += qty * hedge_greeks_list[i].rho

    return NeutralizationResult(
        success=True,
        quantities=quantities,
        net_greeks=net,
        portfolio_greeks=portfolio_greeks,
        hedge_greeks=hedge_greeks_list,
    )


# ─────────────────────────────────────────────────────────────
# CONVENIENCE: PRETTY-PRINT
# ─────────────────────────────────────────────────────────────

def format_greeks_table(greeks: Greeks, label: str = "") -> str:
    """Format Greeks as an aligned table row."""
    prefix = f"  {label:<12}" if label else "  "
    return (
        f"{prefix}"
        f"Δ={greeks.delta:+10.6f}  "
        f"Γ={greeks.gamma:+10.6f}  "
        f"V={greeks.vega:+10.6f}  "
        f"Θ={greeks.theta:+10.6f}  "
        f"ρ={greeks.rho:+10.6f}"
    )
