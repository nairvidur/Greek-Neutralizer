"""
test_greeks_engine.py
════════════════════════════════════════════════════════════════
Comprehensive test suite for the Greeks computation engine.

Run:  pytest tests/ -v
════════════════════════════════════════════════════════════════
"""

import math
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.greeks_engine import (
    normal_cdf,
    normal_pdf,
    compute_d1_d2,
    black_scholes_price,
    compute_greeks,
    aggregate_portfolio_greeks,
    solve_linear_system,
    solve_neutralization,
    OptionContract,
    Greeks,
)


# ─────────────────────────────────────────────────────────────
# STATISTICAL FUNCTIONS
# ─────────────────────────────────────────────────────────────

class TestNormalCDF:
    """Tests for the cumulative normal distribution function."""

    def test_standard_values(self):
        assert abs(normal_cdf(0.0) - 0.5) < 1e-6
        assert abs(normal_cdf(1.0) - 0.8413447) < 1e-5
        assert abs(normal_cdf(-1.0) - 0.1586553) < 1e-5
        assert abs(normal_cdf(2.0) - 0.9772499) < 1e-5

    def test_symmetry(self):
        for x in [0.5, 1.0, 1.5, 2.0, 3.0]:
            assert abs(normal_cdf(x) + normal_cdf(-x) - 1.0) < 1e-7

    def test_extreme_values(self):
        assert normal_cdf(10.0) > 0.9999
        assert normal_cdf(-10.0) < 0.0001

    def test_monotonicity(self):
        prev = normal_cdf(-5.0)
        for x in range(-4, 6):
            curr = normal_cdf(float(x))
            assert curr > prev
            prev = curr


class TestNormalPDF:
    """Tests for the standard normal PDF."""

    def test_peak(self):
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert abs(normal_pdf(0.0) - expected) < 1e-10

    def test_symmetry(self):
        for x in [0.5, 1.0, 2.0, 3.0]:
            assert abs(normal_pdf(x) - normal_pdf(-x)) < 1e-10

    def test_positive(self):
        for x in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            assert normal_pdf(x) > 0


# ─────────────────────────────────────────────────────────────
# BLACK-SCHOLES PRICING
# ─────────────────────────────────────────────────────────────

class TestBlackScholesPrice:
    """Tests for Black-Scholes option pricing."""

    def test_atm_call(self):
        price = black_scholes_price(100, 100, 1.0, 0.05, 0.20, "call")
        assert 8.0 < price < 12.0  # ATM 1Y call ~ $10

    def test_put_call_parity(self):
        """C - P = S - K * exp(-rT)"""
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.30
        call = black_scholes_price(S, K, T, r, sigma, "call")
        put = black_scholes_price(S, K, T, r, sigma, "put")
        parity = S - K * math.exp(-r * T)
        assert abs((call - put) - parity) < 1e-6

    def test_deep_itm_call(self):
        price = black_scholes_price(200, 100, 0.5, 0.05, 0.20, "call")
        intrinsic = 200 - 100 * math.exp(-0.05 * 0.5)
        assert price > intrinsic * 0.99

    def test_expired_option(self):
        assert black_scholes_price(105, 100, 0.0, 0.05, 0.20, "call") == 5.0
        assert black_scholes_price(95, 100, 0.0, 0.05, 0.20, "call") == 0.0
        assert black_scholes_price(95, 100, 0.0, 0.05, 0.20, "put") == 5.0

    def test_zero_vol(self):
        assert black_scholes_price(100, 100, 1.0, 0.05, 0.0, "call") == 0.0


# ─────────────────────────────────────────────────────────────
# GREEKS COMPUTATION
# ─────────────────────────────────────────────────────────────

class TestComputeGreeks:
    """Tests for Black-Scholes Greeks calculation."""

    def test_call_delta_range(self):
        g = compute_greeks(100, 100, 0.25, 0.05, 0.30, "call")
        assert 0.0 < g.delta < 1.0

    def test_put_delta_range(self):
        g = compute_greeks(100, 100, 0.25, 0.05, 0.30, "put")
        assert -1.0 < g.delta < 0.0

    def test_call_put_delta_relationship(self):
        """Call delta - Put delta = 1 (approximately)"""
        call_g = compute_greeks(100, 100, 0.25, 0.05, 0.30, "call")
        put_g = compute_greeks(100, 100, 0.25, 0.05, 0.30, "put")
        assert abs((call_g.delta - put_g.delta) - 1.0) < 1e-4

    def test_gamma_positive(self):
        for opt_type in ["call", "put"]:
            g = compute_greeks(100, 100, 0.5, 0.05, 0.30, opt_type)
            assert g.gamma > 0

    def test_gamma_same_for_call_put(self):
        call_g = compute_greeks(100, 100, 0.5, 0.05, 0.30, "call")
        put_g = compute_greeks(100, 100, 0.5, 0.05, 0.30, "put")
        assert abs(call_g.gamma - put_g.gamma) < 1e-10

    def test_vega_positive(self):
        for opt_type in ["call", "put"]:
            g = compute_greeks(100, 100, 0.5, 0.05, 0.30, opt_type)
            assert g.vega > 0

    def test_vega_same_for_call_put(self):
        call_g = compute_greeks(100, 100, 0.5, 0.05, 0.30, "call")
        put_g = compute_greeks(100, 100, 0.5, 0.05, 0.30, "put")
        assert abs(call_g.vega - put_g.vega) < 1e-10

    def test_theta_negative_for_long(self):
        g = compute_greeks(100, 100, 0.5, 0.05, 0.30, "call")
        assert g.theta < 0  # Time decay is negative

    def test_invalid_inputs(self):
        g = compute_greeks(100, 100, 0.0, 0.05, 0.30, "call")
        assert g.delta == 0.0
        assert g.gamma == 0.0

    def test_deep_itm_call_delta(self):
        g = compute_greeks(200, 100, 0.5, 0.05, 0.20, "call")
        assert g.delta > 0.95

    def test_deep_otm_call_delta(self):
        g = compute_greeks(50, 100, 0.25, 0.05, 0.20, "call")
        assert g.delta < 0.01


# ─────────────────────────────────────────────────────────────
# PORTFOLIO AGGREGATION
# ─────────────────────────────────────────────────────────────

class TestPortfolioAggregation:
    """Tests for multi-position portfolio aggregation."""

    def test_single_position(self):
        contracts = [OptionContract("call", 100, 0.25, 0.30, quantity=10, side="long")]
        positions, agg = aggregate_portfolio_greeks(100, 0.05, contracts)
        assert len(positions) == 1
        assert abs(agg.delta - positions[0].position_greeks.delta) < 1e-10

    def test_long_short_offset(self):
        """Long and short same option should cancel out."""
        contracts = [
            OptionContract("call", 100, 0.25, 0.30, quantity=10, side="long"),
            OptionContract("call", 100, 0.25, 0.30, quantity=10, side="short"),
        ]
        _, agg = aggregate_portfolio_greeks(100, 0.05, contracts)
        assert abs(agg.delta) < 1e-10
        assert abs(agg.gamma) < 1e-10
        assert abs(agg.vega) < 1e-10

    def test_three_position_aggregation(self):
        contracts = [
            OptionContract("call", 100, 0.25, 0.30, quantity=10, side="long"),
            OptionContract("put",  95,  0.25, 0.32, quantity=5,  side="short"),
            OptionContract("call", 110, 0.50, 0.25, quantity=8,  side="long"),
        ]
        positions, agg = aggregate_portfolio_greeks(100, 0.05, contracts)
        assert len(positions) == 3

        # Verify aggregation = sum of individual
        total_delta = sum(p.position_greeks.delta for p in positions)
        assert abs(agg.delta - total_delta) < 1e-10

    def test_direction_sign(self):
        long = OptionContract("call", 100, 0.25, 0.30, quantity=5, side="long")
        short = OptionContract("call", 100, 0.25, 0.30, quantity=5, side="short")
        assert long.direction_sign() == 1
        assert short.direction_sign() == -1


# ─────────────────────────────────────────────────────────────
# LINEAR SOLVER
# ─────────────────────────────────────────────────────────────

class TestLinearSolver:
    """Tests for Gaussian elimination with partial pivoting."""

    def test_identity_system(self):
        A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = [3.0, 5.0, 7.0]
        x = solve_linear_system(A, b)
        assert x is not None
        assert abs(x[0] - 3.0) < 1e-10
        assert abs(x[1] - 5.0) < 1e-10
        assert abs(x[2] - 7.0) < 1e-10

    def test_known_solution(self):
        """2x + y = 5, x + 3y = 10 → x=1, y=3"""
        A = [[2, 1], [1, 3]]
        b = [5.0, 10.0]
        x = solve_linear_system(A, b)
        assert x is not None
        assert abs(x[0] - 1.0) < 1e-10
        assert abs(x[1] - 3.0) < 1e-10

    def test_singular_matrix(self):
        A = [[1, 2, 3], [2, 4, 6], [1, 1, 1]]
        b = [1.0, 2.0, 3.0]
        x = solve_linear_system(A, b)
        assert x is None

    def test_3x3_system(self):
        A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
        b = [8.0, -11.0, -3.0]
        x = solve_linear_system(A, b)
        assert x is not None
        assert abs(x[0] - 2.0) < 1e-10
        assert abs(x[1] - 3.0) < 1e-10
        assert abs(x[2] - -1.0) < 1e-10

    def test_needs_pivoting(self):
        """System where pivoting is necessary."""
        A = [[0, 1], [1, 0]]
        b = [3.0, 5.0]
        x = solve_linear_system(A, b)
        assert x is not None
        assert abs(x[0] - 5.0) < 1e-10
        assert abs(x[1] - 3.0) < 1e-10


# ─────────────────────────────────────────────────────────────
# FULL NEUTRALIZATION
# ─────────────────────────────────────────────────────────────

class TestNeutralization:
    """Tests for the end-to-end neutralization solver."""

    def test_successful_neutralization(self):
        portfolio = [
            OptionContract("call", 100, 0.25, 0.30, quantity=10, side="long"),
            OptionContract("put",  95,  0.25, 0.32, quantity=5,  side="short"),
            OptionContract("call", 110, 0.50, 0.25, quantity=8,  side="long"),
        ]
        hedges = [
            OptionContract("put",  95,  0.50, 0.32),
            OptionContract("call", 105, 0.50, 0.28),
            OptionContract("call", 100, 0.75, 0.30),
        ]

        result = solve_neutralization(100, 0.05, portfolio, hedges)
        assert result.success is True
        assert result.quantities is not None
        assert len(result.quantities) == 3

        # Verify neutralization
        assert abs(result.net_greeks.delta) < 1e-8
        assert abs(result.net_greeks.gamma) < 1e-8
        assert abs(result.net_greeks.vega) < 1e-8

    def test_wrong_number_of_hedges(self):
        portfolio = [OptionContract("call", 100, 0.25, 0.30, quantity=10, side="long")]
        hedges = [
            OptionContract("put",  95,  0.50, 0.32),
            OptionContract("call", 105, 0.50, 0.28),
        ]
        result = solve_neutralization(100, 0.05, portfolio, hedges)
        assert result.success is False

    def test_singular_hedges(self):
        """Three identical instruments → singular matrix."""
        portfolio = [OptionContract("call", 100, 0.25, 0.30, quantity=10, side="long")]
        hedges = [
            OptionContract("call", 100, 0.50, 0.30),
            OptionContract("call", 100, 0.50, 0.30),
            OptionContract("call", 100, 0.50, 0.30),
        ]
        result = solve_neutralization(100, 0.05, portfolio, hedges)
        assert result.success is False

    def test_single_position_portfolio(self):
        portfolio = [OptionContract("call", 100, 0.25, 0.30, quantity=1, side="long")]
        hedges = [
            OptionContract("put",  90,  0.50, 0.35),
            OptionContract("call", 110, 0.50, 0.25),
            OptionContract("call", 100, 1.00, 0.30),
        ]
        result = solve_neutralization(100, 0.05, portfolio, hedges)
        assert result.success is True
        assert abs(result.net_greeks.delta) < 1e-8
        assert abs(result.net_greeks.gamma) < 1e-8
        assert abs(result.net_greeks.vega) < 1e-8

    def test_five_position_portfolio(self):
        """Verify neutralization works with many positions."""
        portfolio = [
            OptionContract("call", 100, 0.25, 0.30, quantity=10, side="long"),
            OptionContract("put",  95,  0.25, 0.32, quantity=5,  side="short"),
            OptionContract("call", 110, 0.50, 0.25, quantity=8,  side="long"),
            OptionContract("put",  90,  0.50, 0.35, quantity=3,  side="long"),
            OptionContract("call", 105, 0.75, 0.28, quantity=7,  side="short"),
        ]
        hedges = [
            OptionContract("put",  85,  0.50, 0.40),
            OptionContract("call", 115, 0.75, 0.22),
            OptionContract("call", 100, 1.00, 0.30),
        ]
        result = solve_neutralization(100, 0.05, portfolio, hedges)
        assert result.success is True
        assert abs(result.net_greeks.delta) < 1e-8
        assert abs(result.net_greeks.gamma) < 1e-8
        assert abs(result.net_greeks.vega) < 1e-8


# ─────────────────────────────────────────────────────────────
# GREEKS DATACLASS
# ─────────────────────────────────────────────────────────────

class TestGreeksDataclass:
    """Tests for the Greeks dataclass operations."""

    def test_addition(self):
        g1 = Greeks(delta=1.0, gamma=2.0, vega=3.0)
        g2 = Greeks(delta=0.5, gamma=1.5, vega=2.5)
        result = g1 + g2
        assert result.delta == 1.5
        assert result.gamma == 3.5
        assert result.vega == 5.5

    def test_scale(self):
        g = Greeks(delta=1.0, gamma=2.0, vega=3.0)
        scaled = g.scale(-2.0)
        assert scaled.delta == -2.0
        assert scaled.gamma == -4.0
        assert scaled.vega == -6.0

    def test_repr(self):
        g = Greeks(delta=0.5, gamma=0.02)
        s = repr(g)
        assert "Δ=" in s
        assert "Γ=" in s


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simple runner if pytest isn't available
    import traceback

    test_classes = [
        TestNormalCDF, TestNormalPDF, TestBlackScholesPrice,
        TestComputeGreeks, TestPortfolioAggregation, TestLinearSolver,
        TestNeutralization, TestGreeksDataclass,
    ]

    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    passed += 1
                    print(f"  ✓ {cls.__name__}.{method_name}")
                except Exception as e:
                    failed += 1
                    print(f"  ✗ {cls.__name__}.{method_name}: {e}")
                    traceback.print_exc()

    print(f"\n  Results: {passed} passed, {failed} failed, {passed + failed} total")
    sys.exit(1 if failed else 0)
