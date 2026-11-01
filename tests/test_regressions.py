"""Regression pins for the baseline defects in docs/evidence/phase1/red_legacy.txt.

Each test states a behavior the old engine got wrong, as that file records.
The hedge-solver behaviors (premium accounting, >3 instruments) are pinned in test_hedging.py.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from greeks_neutralizer.domain import OptionContract, OptionRight, Position
from greeks_neutralizer.errors import ValidationError
from greeks_neutralizer.market import DiscountCurve, FlatVol, MarketSnapshot, UnderlyingMarket
from greeks_neutralizer.pricing.black import black_greeks, normal_cdf
from greeks_neutralizer.pricing.book import value_book
from tests.conftest import AS_OF, euro, expiry_in, flat_snapshot


def test_nan_vol_cannot_enter_the_system():
    with pytest.raises(ValidationError, match="finite"):
        FlatVol(float("nan"))
    with pytest.raises(ValidationError, match="NaN or inf"):
        black_greeks(100.0, 100.0, 0.25, 0.05, 0.0, float("nan"), 1)


def test_nan_from_a_surface_fails_closed():
    class BrokenSurface:
        def vol(self, k, tau):
            return np.full_like(k, np.nan)

        def fingerprint_payload(self):
            return {"kind": "broken"}

    from greeks_neutralizer.errors import MarketDataError

    market = UnderlyingMarket("XYZ", 100.0, DiscountCurve.flat(0.05), BrokenSurface())
    snapshot = MarketSnapshot(AS_OF, {"XYZ": market})
    with pytest.raises(MarketDataError, match="NaN"):
        value_book([Position(euro("call", 100, 30), 1)], snapshot)


@pytest.mark.parametrize("right", ["CALL", "c", "", None])
def test_unknown_option_right_rejected(right):
    with pytest.raises(ValidationError, match="OptionRight"):
        OptionContract("XYZ", right, 100.0, expiry_in(30))


def test_sign_lives_only_in_quantity_so_side_cannot_disagree():
    assert Position(euro("call", 100, 30), -5).quantity == -5
    with pytest.raises(ValidationError, match="signed int"):
        Position(euro("call", 100, 30), 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="signed int"):
        Position(euro("call", 100, 30), True)
    # There is nowhere for a second statement of the side to go.
    with pytest.raises(TypeError, match="side"):
        Position(euro("call", 100, 30), -5, side="BUY")  # type: ignore[call-arg]


def test_expired_itm_keeps_intrinsic_and_is_flagged():
    book = value_book(
        [Position(replace(euro("call", 100, 30), expiry=AS_OF), 1000)],
        flat_snapshot(spot=120.0),
    )
    assert_allclose(book.total_value, 20.0 * 100 * 1000)
    assert book.lines[0].exposure.delta_shares == 0.0
    assert any("EXPIRED" in issue for issue in book.issues)


def test_cdf_matches_erfc_to_machine_precision():
    from math import erfc, sqrt

    for x in (-3.0, -1.0, 0.06, 1.0, 3.0):
        assert_allclose(float(normal_cdf(x)), 0.5 * erfc(-x / sqrt(2)), rtol=1e-14)


def test_delta_equals_finite_difference_of_own_price():
    h = 1e-3
    up = black_greeks(100 + h, 100, 0.25, 0.05, 0.0, 0.3, 1).price[0]
    dn = black_greeks(100 - h, 100, 0.25, 0.05, 0.0, 0.3, 1).price[0]
    delta = black_greeks(100, 100, 0.25, 0.05, 0.0, 0.3, 1).delta[0]
    assert_allclose(delta, (up - dn) / (2 * h), atol=1e-8)


def test_far_otm_price_never_negative():
    strikes = np.arange(101.0, 400.0)
    assert np.all(black_greeks(100.0, strikes, 1 / 365, 0.05, 0.0, 0.2, 1).price >= 0.0)


def test_contract_multiplier_scales_position_delta():
    book = value_book(
        [Position(euro("call", 100, 91.25), 10)],
        flat_snapshot(vol=0.30),
    )
    assert_allclose(book.lines[0].exposure.delta_shares, 562.9, atol=1.0)


def test_right_enum_values():
    assert OptionRight("call").omega == 1
    assert OptionRight("put").omega == -1
