"""RED gate: each baseline defect stated as the behavior a correct engine must have.

Run against the legacy `src/greeks_engine.py` to capture the failures in
docs/evidence/phase1/red_legacy.txt. Removed with the legacy engine; the same
behaviors are pinned against the new API in tests/test_regressions.py.
"""

import math

import pytest

from src.greeks_engine import (
    OptionContract,
    black_scholes_price,
    compute_greeks,
    normal_cdf,
    solve_neutralization,
)

PORT = [OptionContract("call", 100, 0.25, 0.30, 10, "long")]
HEDGES = [
    OptionContract("put", 95, 0.5, 0.32),
    OptionContract("call", 105, 0.5, 0.28),
    OptionContract("call", 100, 0.75, 0.30),
]


def test_nan_iv_must_not_report_success():
    bad = [OptionContract("call", 100, 0.25, float("nan"), 10, "long")]
    try:
        result = solve_neutralization(100, 0.05, bad, HEDGES)
    except ValueError:
        return
    assert not result.success


def test_unknown_option_type_rejected():
    with pytest.raises(ValueError):
        black_scholes_price(100, 100, 1, 0.05, 0.2, "CALL")


def test_unknown_side_rejected():
    with pytest.raises(ValueError):
        OptionContract("call", 100, 1, 0.2, 1, "lnog")


def test_negative_quantity_rejected():
    with pytest.raises(ValueError):
        OptionContract("call", 100, 1, 0.2, -5, "short")


def test_expired_itm_keeps_intrinsic():
    assert compute_greeks(120, 100, 0, 0.05, 0.2, "call").price == pytest.approx(20.0)


def test_zero_vol_itm_is_discounted_intrinsic():
    expected = 120 - 100 * math.exp(-0.05)
    assert black_scholes_price(120, 100, 1, 0.05, 0.0, "call") == pytest.approx(expected)


def test_cdf_accuracy_to_machine_precision():
    for x in (-3.0, -1.0, 0.06, 1.0, 3.0):
        assert normal_cdf(x) == pytest.approx(0.5 * math.erfc(-x / math.sqrt(2)), rel=1e-14)


def test_delta_matches_finite_difference_of_own_price():
    h = 1e-3
    fd = (
        black_scholes_price(100 + h, 100, 0.25, 0.05, 0.3, "call")
        - black_scholes_price(100 - h, 100, 0.25, 0.05, 0.3, "call")
    ) / (2 * h)
    assert compute_greeks(100, 100, 0.25, 0.05, 0.3, "call").delta == pytest.approx(fd, abs=1e-8)


def test_far_otm_price_never_negative():
    assert min(black_scholes_price(100, k, 1 / 365, 0.05, 0.2, "call") for k in range(101, 200)) >= 0


def test_net_value_includes_hedge_premium():
    result = solve_neutralization(100, 0.05, PORT, HEDGES)
    assert result.net_greeks.price != pytest.approx(result.portfolio_greeks.price)


def test_contract_multiplier_in_position_delta():
    result = solve_neutralization(100, 0.05, PORT, HEDGES)
    assert result.portfolio_greeks.delta == pytest.approx(562.9, abs=1.0)


def test_can_hedge_with_underlying_or_more_than_three_instruments():
    four = [*HEDGES, OptionContract("put", 90, 0.5, 0.33)]
    assert solve_neutralization(100, 0.05, PORT, four).success
