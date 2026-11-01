"""Data layer: symbology, quality gates, parity forwards, calibration, and the Cboe adapter."""

from __future__ import annotations

import importlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import least_squares as real_least_squares

from greeks_neutralizer import surface as surface_module
from greeks_neutralizer.data.cboe import ParsedChain, parse_cboe_delayed
from greeks_neutralizer.data.chain import (
    ChainQuote,
    box_implied_rate,
    calibrate_chain,
    implied_forward,
)
from greeks_neutralizer.data.quality import (
    QualityPolicy,
    QuoteFlag,
    assess_quote,
    is_usable,
)
from greeks_neutralizer.data.symbology import parse_osi
from greeks_neutralizer.domain import (
    ExerciseStyle,
    OptionContract,
    OptionRight,
    Quote,
    SettlementKind,
)
from greeks_neutralizer.errors import MarketDataError, ValidationError
from greeks_neutralizer.pricing.black import black_premium
from greeks_neutralizer.surface import fit_svi_slice

NOW = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)


def quote(bid=1.0, ask=1.1, bs=10, a_s=10, age=0.0):
    return Quote(bid, ask, bs, a_s, NOW - timedelta(seconds=age))


@pytest.mark.parametrize(
    ("text", "root", "right", "strike"),
    [
        ("SPXW261113C07765000", "SPXW", OptionRight.CALL, 7765.0),
        ("SPY   270129P00741500", "SPY", OptionRight.PUT, 741.5),
        ("B1SP261113C00700000", "B1SP", OptionRight.CALL, 700.0),
    ],
    ids=["compact", "padded", "adjusted-root"],
)
def test_parse_osi(text, root, right, strike):
    parsed = parse_osi(text)
    assert (parsed.root, parsed.right, parsed.strike) == (root, right, strike)


@pytest.mark.parametrize("bad", ["", "SPY", "SPY261399C00100000", "SPY261113X00100000"])
def test_bad_osi_rejected(bad):
    with pytest.raises(ValidationError):
        parse_osi(bad)


@pytest.mark.parametrize(
    ("q", "flag", "usable"),
    [
        (quote(1.2, 1.1), QuoteFlag.CROSSED, False),
        (quote(1.1, 1.1), QuoteFlag.LOCKED, True),
        (quote(0.0, 0.05), QuoteFlag.ZERO_BID, True),
        (quote(bs=0, a_s=0), QuoteFlag.NO_SIZE, False),
        (quote(1.0, 3.0), QuoteFlag.WIDE_SPREAD, True),
        (quote(age=120), QuoteFlag.STALE, False),
        (quote(age=-60), QuoteFlag.FUTURE_TIMESTAMP, False),
    ],
    ids=["crossed", "locked", "zero-bid", "no-size", "wide", "stale", "future"],
)
def test_quality_flags(q, flag, usable):
    flags = assess_quote(q, NOW)
    assert flag in flags
    assert is_usable(flags) is usable


def test_below_intrinsic_flagged_and_blocks():
    flags = assess_quote(quote(5.0, 5.1), NOW, intrinsic=20.0)
    assert QuoteFlag.BELOW_INTRINSIC in flags
    assert not is_usable(flags)


def test_clean_quote_has_no_flags():
    assert assess_quote(quote(), NOW) == frozenset()


def test_source_without_sizes_needs_explicit_policy():
    q = quote(bs=0, a_s=0)
    assert not is_usable(assess_quote(q, NOW))
    assert is_usable(assess_quote(q, NOW, policy=QualityPolicy(require_size=False)))


def _chain(F=100.0, D=0.99, T=0.25, spread=0.02, vol_fn=None, style=ExerciseStyle.EUROPEAN):
    expiry = NOW + timedelta(days=T * 365)
    out = []
    vol_fn = vol_fn or (lambda k: 0.2 - 0.1 * k + 0.3 * k * k)
    for K in np.arange(0.7 * F, 1.3 * F + 0.1, 0.02 * F):
        k = np.log(K / F)
        for right in OptionRight:
            price = black_premium(
                np.array([F]),
                np.array([K]),
                np.array([D]),
                np.array([vol_fn(k) * np.sqrt(T)]),
                np.array([float(right.omega)]),
            )[0]
            contract = OptionContract(
                "XYZ", right, float(K), expiry, style, settlement=SettlementKind.CASH
            )
            q = Quote(max(price - spread, 0.0), price + spread, 10, 10, NOW)
            out.append(ChainQuote(contract, q))
    return out


def test_parity_forward_recovers_truth():
    chain = _chain(F=101.5, D=0.99)
    result = implied_forward(chain, chain[0].contract.expiry, NOW, 100.0, discount=0.99)
    assert_allclose(result.forward, 101.5, rtol=1e-6)
    assert result.residual_rms < 1e-6


def test_parity_forward_fails_closed_without_pairs():
    chain = [q for q in _chain() if q.contract.right is OptionRight.CALL]
    with pytest.raises(MarketDataError, match="usable pairs"):
        implied_forward(chain, chain[0].contract.expiry, NOW, 100.0, discount=0.99)


def test_box_rate_recovers_financing_from_european_chain():
    T = 0.5
    D = np.exp(-0.04 * T)
    rate = box_implied_rate(_chain(F=100.0, D=D, T=T), NOW, 100.0)
    assert_allclose(rate, 0.04, atol=1e-5)


def test_box_rate_refuses_american_chain():
    with pytest.raises(MarketDataError, match="European"):
        box_implied_rate(_chain(style=ExerciseStyle.AMERICAN), NOW, 100.0)


def test_calibration_recovers_smile_from_quotes():
    chain = _chain(F=100.0, D=np.exp(-0.04 * 0.25))
    calibration = calibrate_chain(chain, NOW, 100.0, discount_rate=0.04)
    assert_allclose(calibration.forwards[0].forward, 100.0, atol=1e-3)
    fit = calibration.fits[0]
    assert fit.rmse_vol < 0.002
    k = np.array([-0.1, 0.0, 0.1])
    vols = calibration.surface.vol(k, np.full(3, 0.25))
    assert_allclose(vols, 0.2 - 0.1 * k + 0.3 * k * k, atol=0.002)


def _scipy_reports(monkeypatch, *, success):
    # Set at the SciPy boundary: the fit itself stays good, so that no quality gate intervenes.
    def reported(*args, **kwargs):
        result = real_least_squares(*args, **kwargs)
        result.success = success
        return result

    monkeypatch.setattr(surface_module, "least_squares", reported)


def _unconverged_least_squares(monkeypatch):
    _scipy_reports(monkeypatch, success=False)


@pytest.mark.parametrize("success", [True, False])
def test_converged_is_scipys_own_status_whatever_the_gradient(monkeypatch, success):
    def reported(*args, **kwargs):
        result = real_least_squares(*args, **kwargs)
        result.success = success
        # Near zero and large: a threshold on this in either direction would change the answer.
        result.optimality = 1e-12 if not success else 1.0
        return result

    monkeypatch.setattr(surface_module, "least_squares", reported)
    k = np.linspace(-0.2, 0.2, 21)
    assert fit_svi_slice(k, 0.2 - 0.1 * k + 0.3 * k * k, 0.25).converged is success


def test_a_fit_given_room_converges_and_one_stopped_early_does_not(monkeypatch):
    # No patched result: SciPy's own status, with evaluations to spare and with two.
    k = np.linspace(-0.2, 0.2, 21)
    vols = 0.2 - 0.1 * k + 0.3 * k * k

    def with_evaluations(limit):
        def solver(*args, **kwargs):
            return real_least_squares(*args, **{**kwargs, "max_nfev": limit})

        return solver

    monkeypatch.setattr(surface_module, "least_squares", with_evaluations(20_000))
    assert fit_svi_slice(k, vols, 0.25).converged
    monkeypatch.setattr(surface_module, "least_squares", with_evaluations(2))
    assert not fit_svi_slice(k, vols, 0.25).converged


def test_a_converged_calibration_has_no_issue(monkeypatch):
    _scipy_reports(monkeypatch, success=True)
    calibration = calibrate_chain(_chain(D=np.exp(-0.04 * 0.25)), NOW, 100.0, discount_rate=0.04)
    assert calibration.n_unconverged_fits == 0
    assert calibration.issues == ()


def test_a_fit_that_does_not_converge_raises_an_issue(monkeypatch):
    _unconverged_least_squares(monkeypatch)
    calibration = calibrate_chain(_chain(D=np.exp(-0.04 * 0.25)), NOW, 100.0, discount_rate=0.04)
    assert calibration.n_unconverged_fits == 1
    assert len(calibration.issues) == 1
    assert calibration.issues[0].startswith("UNCONVERGED_FIT 1 of 1 expiries")


def test_the_reconciliation_output_reports_a_fit_that_did_not_converge(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    reconcile_chain = importlib.import_module("reconcile_chain")
    chain = [
        replace(item, vendor_iv=0.2, vendor_delta=0.5, vendor_gamma=0.01)
        for item in _chain(D=np.exp(-0.04 * 0.25))
    ]
    parsed = ParsedChain(chain[0].contract.underlying, 100.0, NOW, tuple(chain), 0, 0, True)
    monkeypatch.setattr(reconcile_chain, "parse_cboe_delayed", lambda _: parsed)
    payload = tmp_path / "payload.json"
    payload.write_text("{}")

    _scipy_reports(monkeypatch, success=True)
    clean = reconcile_chain.reconcile(payload, 0.04)
    assert clean["fits_unconverged"] == 0
    assert clean["issues"] == []

    _unconverged_least_squares(monkeypatch)
    failed = reconcile_chain.reconcile(payload, 0.04)
    assert failed["fits_unconverged"] == 1
    assert [issue.split()[0] for issue in failed["issues"]] == ["UNCONVERGED_FIT"]


def test_calibration_excludes_blocked_quotes_and_counts_them():
    good = _chain(D=np.exp(-0.04 * 0.25))
    stale = [
        ChainQuote(item.contract, Quote(1.0, 1.1, 5, 5, NOW - timedelta(hours=2)))
        for item in _chain(T=0.5)
    ]
    calibration = calibrate_chain(good + stale, NOW, 100.0, discount_rate=0.04)
    assert calibration.flags["stale"] == len(stale)
    assert len(calibration.fits) == 1


def test_calibration_with_nothing_usable_fails_closed():
    stale = [
        ChainQuote(item.contract, Quote(1.0, 1.1, 5, 5, NOW - timedelta(hours=2)))
        for item in _chain()
    ]
    with pytest.raises(MarketDataError, match="no expiry"):
        calibrate_chain(stale, NOW, 100.0, discount_rate=0.04)
    with pytest.raises(MarketDataError, match="empty"):
        calibrate_chain([], NOW, 100.0, discount_rate=0.04)


def _payload(options, **data):
    return {
        "timestamp": "2026-10-31 22:21:00",
        "data": {
            "symbol": "SPY",
            "current_price": 755.72,
            "last_trade_time": "2026-10-30T16:00:00",
            "options": options,
            **data,
        },
    }


ROW = {
    "option": "SPY261113C00760000",
    "bid": 7.0,
    "ask": 7.1,
    "bid_size": 5.0,
    "ask_size": 5.0,
    "iv": 0.15,
    "delta": 0.45,
    "gamma": 0.01,
    "last_trade_time": "2026-10-30T16:14:00",
}


def test_cboe_as_of_is_the_quote_time_not_the_fetch_time():
    parsed = parse_cboe_delayed(_payload([ROW]))
    assert parsed.as_of.isoformat() == "2026-10-30T16:14:00-04:00"


def test_cboe_style_follows_root():
    spx = dict(ROW, option="SPXW261113C07600000")
    parsed = parse_cboe_delayed(_payload([spx], symbol="^SPX"))
    assert parsed.symbol == "SPX"
    assert parsed.quotes[0].contract.style is ExerciseStyle.EUROPEAN
    american = parse_cboe_delayed(_payload([ROW])).quotes[0].contract
    assert american.style is ExerciseStyle.AMERICAN


def test_cboe_am_settled_root_fixes_at_open():
    monthly = dict(ROW, option="SPX261120C07600000")
    contract = parse_cboe_delayed(_payload([monthly], symbol="^SPX")).quotes[0].contract
    assert contract.expiry.hour == 9
    assert contract.expiry.minute == 30


def test_cboe_excludes_and_counts_non_standard_deliverables():
    adjusted = dict(ROW, option="B1SP261113C00760000")
    parsed = parse_cboe_delayed(_payload([ROW, adjusted]))
    assert len(parsed.quotes) == 1
    assert parsed.n_non_standard == 1


def test_cboe_counts_malformed_rows():
    rows = [*([ROW] * 40), {"option": "garbage"}, dict(ROW, bid="x")]
    assert parse_cboe_delayed(_payload(rows)).n_rejected == 2


def test_cboe_detects_missing_sizes():
    row = dict(ROW, bid_size=0.0, ask_size=0.0)
    assert parse_cboe_delayed(_payload([row])).sizes_available is False
    assert parse_cboe_delayed(_payload([ROW])).sizes_available is True


@pytest.mark.parametrize(
    "bad",
    [
        {},
        ["not an object"],
        {"timestamp": "x"},
        {"data": {}, "timestamp": 1},
        {"data": {}, "timestamp": "not a time"},
        {"data": {"options": 5}, "timestamp": "2026-01-05T15:00:00"},
        _payload([]),
        _payload(["nope"]),
    ],
    ids=[
        "empty",
        "not-an-object",
        "no-data",
        "bad-ts",
        "unreadable-ts",
        "options-not-a-list",
        "no-options",
        "bad-rows",
    ],
)
def test_cboe_malformed_payload_fails_closed(bad):
    with pytest.raises(MarketDataError):
        parse_cboe_delayed(bad)
