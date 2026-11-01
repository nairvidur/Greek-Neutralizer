"""The fail-closed rule, end to end. Each case was reproduced as a silent failure by review."""

from __future__ import annotations

import contextlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from greeks_neutralizer.backtest.engine import BacktestResult, DayResult
from greeks_neutralizer.backtest.report import compare, policy_stats
from greeks_neutralizer.data.cboe import parse_cboe_delayed
from greeks_neutralizer.data.chain import ChainQuote, calibrate_chain
from greeks_neutralizer.data.quality import QuoteFlag, assess_quote, is_usable
from greeks_neutralizer.domain import (
    ExerciseStyle,
    LinearInstrument,
    OptionContract,
    OptionRight,
    Position,
    Quote,
    SettlementKind,
)
from greeks_neutralizer.errors import (
    GNError,
    HedgeInfeasibleError,
    MarketDataError,
    ValidationError,
)
from greeks_neutralizer.examples import example_request
from greeks_neutralizer.hedging.diagnostics import hedge_frontier
from greeks_neutralizer.hedging.optimizer import solve_hedge, verify_proposal
from greeks_neutralizer.hedging.problem import (
    GreekName,
    GreekTarget,
    HedgeCandidate,
    HedgeConfig,
    HedgeLeg,
    HedgeProblem,
    ProposalStatus,
)
from greeks_neutralizer.market import DiscountCurve, FlatVol, MarketSnapshot, UnderlyingMarket
from greeks_neutralizer.pricing.black import black_greeks
from greeks_neutralizer.pricing.book import value_book
from greeks_neutralizer.schema import parse_hedge_request, proposal_to_json
from greeks_neutralizer.service.app import create_app
from greeks_neutralizer.surface import SviSlice, SviSurface
from greeks_neutralizer.workflow import (
    AuditLog,
    ProposalStore,
    WorkflowError,
    WorkflowPolicy,
    tickets_csv,
    tickets_fix,
    verify_audit_log,
)
from tests.conftest import AS_OF, euro, expiry_in, flat_snapshot

STOCK = LinearInstrument("XYZ", "XYZ")
DELTA = (GreekTarget("XYZ", GreekName.DELTA, weight=1.0),)
BOOK = (Position(euro("call", 100, 30), -10),)


def fair(right, strike, days, half=0.02, size=500, age=0.0, vol=0.25, rate=0.05):
    contract = euro(right, strike, days)
    omega = 1 if right == "call" else -1
    price = black_greeks(100.0, strike, days / 365, rate, 0.0, vol, omega).price[0]
    ts = AS_OF - timedelta(seconds=age)
    return HedgeCandidate(contract, Quote(max(price - half, 0.0), price + half, size, size, ts))


def stock(age=0.0):
    ts = AS_OF - timedelta(seconds=age)
    return HedgeCandidate(STOCK, Quote(99.99, 100.01, 10**6, 10**6, ts))


UNIVERSE = (
    stock(),
    fair("put", 95, 60),
    fair("call", 105, 60),
    fair("call", 100, 90),
    fair("put", 90, 90),
)
DGV = tuple(GreekTarget("XYZ", g) for g in (GreekName.DELTA, GreekName.GAMMA, GreekName.VEGA))


# --- Nothing usable is not "no trade needed" -----------------------------------------------------


def test_no_usable_candidates_is_not_reported_as_no_trade():
    stale = tuple(
        replace(c, quote=replace(c.quote, timestamp=AS_OF - timedelta(hours=1))) for c in UNIVERSE
    )
    p = solve_hedge(HedgeProblem(BOOK, stale, DGV), flat_snapshot())
    assert p.status is ProposalStatus.NO_USABLE_CANDIDATES
    assert any("NO_USABLE_CANDIDATES" in issue for issue in p.issues)


def test_real_no_trade_is_still_no_trade():
    p = solve_hedge(HedgeProblem((), UNIVERSE, DGV), flat_snapshot())
    assert p.status is ProposalStatus.NO_TRADE


def test_many_exclusions_are_raised_as_an_issue():
    half_stale = (
        *UNIVERSE[:2],
        *(
            replace(c, quote=replace(c.quote, timestamp=AS_OF - timedelta(hours=1)))
            for c in UNIVERSE[2:]
        ),
    )
    p = solve_hedge(HedgeProblem(BOOK, half_stale, DGV), flat_snapshot())
    assert any("EXCLUDED" in issue for issue in p.issues)


# --- Targets --------------------------------------------------------------------------------------


def test_target_on_unknown_underlying_is_an_error():
    typo = (GreekTarget("XYX", GreekName.DELTA),)
    with pytest.raises(ValidationError, match="XYX"):
        solve_hedge(HedgeProblem(BOOK, UNIVERSE, typo), flat_snapshot())


def test_target_with_nothing_on_it_is_an_error():
    other = UnderlyingMarket("ABC", 50.0, DiscountCurve.flat(0.05), FlatVol(0.3))
    snap = MarketSnapshot(AS_OF, {**flat_snapshot().underlyings, "ABC": other})
    with pytest.raises(ValidationError, match="ABC"):
        solve_hedge(HedgeProblem(BOOK, UNIVERSE, (GreekTarget("ABC", GreekName.DELTA),)), snap)


def test_book_underlying_without_a_target_is_flagged():
    other = UnderlyingMarket("ABC", 50.0, DiscountCurve.flat(0.05), FlatVol(0.3))
    snap = MarketSnapshot(AS_OF, {**flat_snapshot().underlyings, "ABC": other})
    book = (*BOOK, Position(euro("call", 50, 30, underlying="ABC"), 5))
    p = solve_hedge(HedgeProblem(book, UNIVERSE, DGV), snap)
    assert any("UNTARGETED" in issue and "ABC" in issue for issue in p.issues)


# --- Rounding must not hide a breached limit ------------------------------------------------------


def test_rounding_cannot_breach_the_premium_budget():
    problem = HedgeProblem(BOOK, (stock(),), DELTA, (), HedgeConfig(premium_budget=25050.5))
    p = solve_hedge(problem, flat_snapshot())
    assert p.net_premium <= 25050.5


def test_rounding_finds_a_feasible_neighbour_for_a_band():
    band = (GreekTarget("XYZ", GreekName.DELTA, lower=20.0, upper=120.0, weight=0.0),)
    p = solve_hedge(HedgeProblem((), (stock(),), band), flat_snapshot())
    assert 20.0 <= p.after["XYZ"].delta_dollars <= 120.0


def test_unreachable_after_rounding_is_infeasible_not_a_breach():
    band = (GreekTarget("XYZ", GreekName.DELTA, lower=20.0, upper=30.0, weight=0.0),)
    with pytest.raises(HedgeInfeasibleError):
        solve_hedge(HedgeProblem((), (stock(),), band), flat_snapshot())


def test_verification_checks_limits_and_lots_itself():
    problem = HedgeProblem(BOOK, (stock(),), DELTA, (), HedgeConfig(premium_budget=1.0, max_legs=1))
    leg = HedgeLeg(STOCK, 100, 100.01, 10_000.0, 1.0, "test")
    report = verify_proposal(
        problem, flat_snapshot(), (leg, leg), {("XYZ", "delta_dollars"): 0.0}, {}
    )
    assert not report.passed
    joined = " ".join(report.band_violations)
    assert "budget" in joined
    assert "legs" in joined


def test_verification_rejects_non_finite_values():
    problem = HedgeProblem(BOOK, (stock(),), DELTA)
    report = verify_proposal(
        problem, flat_snapshot(), (), {("XYZ", "delta_dollars"): float("nan")}, {}
    )
    assert not report.passed


# --- Quotes ---------------------------------------------------------------------------------------


def test_a_zero_offer_is_blocked_not_free():
    flags = assess_quote(Quote(0.0, 0.0, 10, 10, AS_OF), AS_OF)
    assert QuoteFlag.NO_OFFER in flags
    assert not is_usable(flags)
    free = HedgeCandidate(euro("call", 105, 60), Quote(0.0, 0.0, 100, 100, AS_OF))
    p = solve_hedge(HedgeProblem(BOOK, (free, stock()), DGV), flat_snapshot())
    assert all(leg.instrument != free.instrument for leg in p.legs)


def test_size_beyond_the_displayed_quote_is_not_free_by_default():
    thin = fair("call", 105, 60, size=1)
    p = solve_hedge(
        HedgeProblem(BOOK, (thin,), (GreekTarget("XYZ", GreekName.VEGA, weight=10.0),)),
        flat_snapshot(),
    )
    assert abs(p.legs[0].quantity) <= 1


def test_explicit_depth_past_the_display_is_flagged():
    deep = replace(fair("call", 105, 60, size=1), max_buy=50)
    p = solve_hedge(
        HedgeProblem(BOOK, (deep,), (GreekTarget("XYZ", GreekName.VEGA, weight=10.0),)),
        flat_snapshot(),
    )
    assert abs(p.legs[0].quantity) > 1
    assert any("OVER_DISPLAYED" in issue for issue in p.issues)


def test_quote_below_intrinsic_is_excluded():
    contract = euro("call", 50, 60)
    bad = HedgeCandidate(contract, Quote(1.0, 1.1, 100, 100, AS_OF))
    p = solve_hedge(HedgeProblem(BOOK, (bad, stock()), DELTA), flat_snapshot())
    assert any("below_intrinsic" in item.reason for item in p.excluded)


def test_single_quote_far_from_the_model_is_excluded():
    fine = fair("call", 105, 60)
    bad = replace(fair("call", 110, 60), quote=Quote(9.0, 9.2, 500, 500, AS_OF))
    p = solve_hedge(HedgeProblem(BOOK, (fine, bad, stock()), DGV), flat_snapshot())
    assert any("iv_outlier" in item.reason for item in p.excluded)


def test_percent_for_decimal_vol_is_caught():
    # sigma=25.0 where 0.25 was meant: repricing agrees with itself, so only quotes can catch it.
    with pytest.raises(MarketDataError):
        solve_hedge(HedgeProblem(BOOK, UNIVERSE, DGV), flat_snapshot(vol=25.0))


def test_rate_for_decimal_is_caught_by_plausibility():
    with pytest.raises(ValidationError):
        DiscountCurve.flat(5.0)


def test_model_and_quotes_disagreeing_broadly_fails_closed():
    with pytest.raises(MarketDataError, match="disagree"):
        solve_hedge(HedgeProblem(BOOK, UNIVERSE, DGV), flat_snapshot(vol=0.60))


# --- Config ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"instability_threshold": float("nan")},
        {"regret_threshold": float("nan")},
        {"time_limit_seconds": float("nan")},
        {"time_limit_seconds": -1.0},
        {"ticket_cost": float("nan")},
        {"risk_aversion": float("inf")},
        {"allow_extrapolated_vol": "false"},
        {"require_quote_size": "no"},
    ],
    ids=lambda d: next(iter(d)),
)
def test_config_rejects_nan_and_wrong_types(bad):
    with pytest.raises(ValidationError):
        HedgeConfig(**bad)


@pytest.mark.parametrize("field", ["fee_per_unit", "impact_per_unit"])
def test_candidate_rejects_nan(field):
    with pytest.raises(ValidationError):
        replace(stock(), **{field: float("nan")})


# --- Stability is a gate ---


def test_hedge_that_breaks_a_band_after_a_small_move_is_unstable():
    book = (Position(euro("call", 100, 30), -1000), Position(euro("put", 95, 30), -1000))
    universe = tuple(replace(c, max_buy=5000, max_sell=5000) for c in UNIVERSE)
    band = (*DGV[:2], GreekTarget("XYZ", GreekName.VEGA, lower=-5.0, upper=5.0))
    p = solve_hedge(HedgeProblem(book, universe, band), flat_snapshot(), run_stability=True)
    assert p.stability is not None
    assert not np.isfinite(p.stability.regret_ratio)
    assert not p.stability.is_stable
    assert p.status is ProposalStatus.UNSTABLE
    data = proposal_to_json(p)
    assert data["stability"]["regret_unbounded"] is True
    assert json.dumps(data, allow_nan=False)


def _verified_proposal(stability=True):
    problem, snapshot, _ = parse_hedge_request(example_request())
    return solve_hedge(problem, snapshot, run_stability=stability)


class Clock:
    def __init__(self, now=datetime(2026, 1, 5, 15, 0, tzinfo=UTC)):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def store(tmp_path):
    return ProposalStore(AuditLog(tmp_path / "audit.jsonl"), clock=Clock())


def test_proposal_with_legs_must_have_run_stability(store):
    with pytest.raises(WorkflowError, match="stability"):
        store.submit(_verified_proposal(stability=False), "alice")


# --- Workflow -----------------------------------------------------------------------------------


def test_stale_market_cannot_be_submitted_approved_or_exported(tmp_path):
    clock = Clock()
    store = ProposalStore(
        AuditLog(tmp_path / "a.jsonl"),
        WorkflowPolicy(max_market_age=timedelta(minutes=2), approval_valid_for=timedelta(hours=1)),
        clock,
    )
    proposal = _verified_proposal()
    record = store.submit(proposal, "alice")
    clock.now += timedelta(minutes=3)
    with pytest.raises(WorkflowError, match="market"):
        store.approve(record.id, "bob")
    with pytest.raises(WorkflowError, match="market"):
        store.submit(proposal, "alice")


def test_audit_failure_blocks_the_action(store, monkeypatch):
    record = store.submit(_verified_proposal(), "alice")

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store.audit, "append", fail)
    with pytest.raises(OSError):
        store.approve(record.id, "bob")
    assert store.get(record.id).state.value == "proposed"
    with pytest.raises(OSError):
        store.submit(_verified_proposal(), "alice")
    assert len(store.list()) == 1


def test_broken_chain_blocks_approval_and_export(tmp_path):
    path = tmp_path / "a.jsonl"
    store = ProposalStore(AuditLog(path), clock=Clock())
    record = store.submit(_verified_proposal(), "alice")
    path.write_text(path.read_text().replace("alice", "mallory"))
    with pytest.raises(WorkflowError, match="audit"):
        store.approve(record.id, "bob")


def test_truncating_the_tail_is_detected(tmp_path):
    path = tmp_path / "a.jsonl"
    store = ProposalStore(AuditLog(path), clock=Clock())
    record = store.submit(_verified_proposal(), "alice")
    store.approve(record.id, "bob")
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + "\n")
    check = verify_audit_log(path)
    assert not check.ok
    assert "tail" in check.reason


def test_deleting_the_log_is_detected(tmp_path):
    path = tmp_path / "a.jsonl"
    ProposalStore(AuditLog(path), clock=Clock()).submit(_verified_proposal(), "alice")
    path.unlink()
    assert not verify_audit_log(path).ok


def test_approve_cannot_overwrite_a_concurrent_rejection(store):
    record = store.submit(_verified_proposal(), "alice")
    original = store.clock

    def reject_during_approve():
        # The window between approve's state check and its transition.
        store.clock = original
        store.reject(record.id, "carol", "no")
        return original()

    store.clock = reject_during_approve
    with contextlib.suppress(WorkflowError):
        store.approve(record.id, "bob")
    assert store.get(record.id).state.value in {"approved", "rejected"}
    events = [json.loads(line)["event"] for line in store.audit.path.read_text().splitlines()]
    assert not ("rejected" in events and "approved" in events)


# --- Tickets --------------------------------------------------------------------------------------


def _leg(contract):
    return HedgeLeg(contract, 5, 1.25, 625.0, 2.0, "test")


def test_fix_maturity_is_in_exchange_time():
    # The same instant in another zone falls on another calendar date.
    # 16:00 in New York on March 6 is already 06:00 on March 7 in Tokyo.
    tokyo = datetime(2026, 3, 6, 21, 0, tzinfo=UTC).astimezone(
        __import__("zoneinfo").ZoneInfo("Asia/Tokyo")
    )
    contract = OptionContract(
        "SPX",
        OptionRight.CALL,
        5000.0,
        tokyo,
        ExerciseStyle.EUROPEAN,
        settlement=SettlementKind.CASH,
    )
    proposal = replace(_verified_proposal(), legs=(_leg(contract),))
    assert "541=20260306|" in tickets_fix(proposal, "abcdefgh")


def test_tickets_name_the_listed_root_and_full_terms():
    contract = OptionContract(
        "SPX",
        OptionRight.CALL,
        5000.0,
        expiry_in(30),
        ExerciseStyle.EUROPEAN,
        settlement=SettlementKind.CASH,
        exchange_symbol="SPXW",
    )
    proposal = replace(_verified_proposal(), legs=(_leg(contract),))
    assert "55=SPXW|" in tickets_fix(proposal, "abcdefgh")
    text = tickets_csv(proposal, "abc")
    header = text.splitlines()[0]
    for column in ("multiplier", "style", "settlement"):
        assert column in header
    assert "SPXW" in text.splitlines()[1]


def test_cboe_keeps_the_listed_root():
    payload = {
        "timestamp": "2026-10-31 22:21:00",
        "data": {
            "symbol": "^SPX",
            "current_price": 7580.0,
            "last_trade_time": "2026-10-30T16:00:00",
            "options": [
                {
                    "option": "SPXW261113C07600000",
                    "bid": 7.0,
                    "ask": 7.1,
                    "last_trade_time": "2026-10-30T16:14:00",
                }
            ],
        },
    }
    assert parse_cboe_delayed(payload).quotes[0].contract.exchange_symbol == "SPXW"


# --- Calibration ---


def _noisy_chain(seed=3):
    rng = np.random.default_rng(seed)
    expiry = AS_OF + timedelta(days=91)
    out = []
    for K in np.arange(70.0, 131.0, 2.0):
        for right in OptionRight:
            vol = float(rng.uniform(0.05, 1.5))
            price = black_greeks(100.0, K, 0.25, 0.04, 0.0, vol, right.omega).price[0]
            contract = OptionContract(
                "XYZ",
                right,
                float(K),
                expiry,
                ExerciseStyle.EUROPEAN,
                settlement=SettlementKind.CASH,
            )
            out.append(
                ChainQuote(contract, Quote(max(price - 0.02, 0.01), price + 0.02, 10, 10, AS_OF))
            )
    return out


def test_garbage_quotes_cannot_produce_a_surface():
    with pytest.raises(MarketDataError):
        calibrate_chain(_noisy_chain(), AS_OF, 100.0, discount_rate=0.04)


# --- Surfaces -----------------------------------------------------------------------------------


def test_extrapolation_flag_covers_tenor():
    surface = SviSurface((SviSlice(0.002, 0.00005, 0.002, -0.3, 0.0, 0.02),), ((-0.05, 0.05),))
    assert surface.extrapolating(np.array([0.0]), np.array([60 / 365]))[0]


def test_allowed_extrapolation_is_still_reported():
    surface = SviSurface((SviSlice(0.25, 0.005, 0.06, -0.6, 0.02, 0.15),), ((-0.2, 0.2),))
    snap = flat_snapshot().replaced("XYZ", surface=surface)
    far = HedgeCandidate(euro("put", 50, 91), Quote(0.01, 0.03, 500, 500, AS_OF))
    book = (Position(euro("put", 95, 91), -10),)
    p = solve_hedge(
        HedgeProblem(
            book,
            (far,),
            (GreekTarget("XYZ", GreekName.VEGA, weight=10.0),),
            (),
            HedgeConfig(allow_extrapolated_vol=True),
        ),
        snap,
    )
    if p.legs:
        assert any("EXTRAPOLATED_VOL" in issue for issue in p.issues)


def test_surface_without_quoted_ranges_is_flagged():
    surface = SviSurface((SviSlice(0.25, 0.005, 0.06, -0.6, 0.02, 0.15),))
    snap = flat_snapshot().replaced("XYZ", surface=surface)
    p = solve_hedge(HedgeProblem(BOOK, (stock(),), DELTA), snap)
    assert any("UNVERIFIED_SURFACE" in issue for issue in p.issues)


# --- Cboe -----------------------------------------------------------------------------------------


GOOD = {
    "option": "SPY261113C00760000",
    "bid": 7.0,
    "ask": 7.1,
    "bid_size": 5.0,
    "ask_size": 5.0,
    "last_trade_time": "2026-10-30T16:14:00",
}


def _payload(rows, **data):
    return {
        "timestamp": "2026-10-31 22:21:00",
        "data": {"symbol": "SPY", "current_price": 755.0, "options": rows, **data},
    }


def test_no_quote_time_fails_closed():
    row = {k: v for k, v in GOOD.items() if k != "last_trade_time"}
    with pytest.raises(MarketDataError, match="quote time"):
        parse_cboe_delayed(_payload([row]))


def test_high_rejection_rate_fails_closed():
    with pytest.raises(MarketDataError, match="rejected"):
        parse_cboe_delayed(_payload([GOOD, *([{"option": "garbage"}] * 50)]))


def test_crossed_underlying_fails_closed():
    with pytest.raises(MarketDataError, match="underlying"):
        parse_cboe_delayed(_payload([GOOD], current_price=None, bid=510.0, ask=490.0))


def test_non_numeric_open_interest_counts_as_rejected():
    rows = [*([GOOD] * 40), dict(GOOD, open_interest="N/A")]
    assert parse_cboe_delayed(_payload(rows)).n_rejected == 1


# --- Frontier -------------------------------------------------------------------------------------


def test_frontier_points_carry_status_and_issues():
    points = hedge_frontier(
        HedgeProblem(BOOK, UNIVERSE, DGV), flat_snapshot(), risk_aversions=(1.0,)
    )
    assert points[-1].status == "optimal" or points[-1].status == "within_gap"
    assert isinstance(points[-1].issues, tuple)


# --- Backtest reporting ---


def _fake(name, pnl, start=datetime(2024, 1, 1, tzinfo=UTC), failures=()):
    days = [
        DayResult(start + timedelta(days=i), float(x), 0, 0, 0, 0, 0, 0, 0)
        for i, x in enumerate(pnl)
    ]
    return BacktestResult(name, days, list(failures))


def test_failed_days_count_includes_the_last_day():
    assert policy_stats(_fake("x", np.arange(50.0), failures=["a", "b"])).failed_days == 2


def test_comparison_requires_the_same_dates():
    other = datetime(2025, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError, match="dates"):
        compare(_fake("a", np.ones(60)), _fake("b", np.ones(60), start=other))


def test_comparison_reports_failed_days():
    a = _fake("a", np.random.default_rng(1).normal(size=60), failures=["x"] * 60)
    b = _fake("b", np.random.default_rng(2).normal(size=60))
    assert compare(a, b).policy_failed_days == 60


# --- Service --------------------------------------------------------------------------------------


ALICE = {"Authorization": "Bearer a"}


@pytest.fixture
def client(tmp_path):
    app = create_app({"a": "alice"}, tmp_path / "audit.jsonl", 200_000)
    return TestClient(app, raise_server_exceptions=False)


def test_body_without_a_length_is_refused(client):
    def body():
        yield b"{}"

    assert client.post("/v1/value", headers=ALICE, content=body()).status_code == 411


def test_broken_chain_makes_health_fail(tmp_path):
    path = tmp_path / "a.jsonl"
    app = create_app({"a": "alice"}, path)
    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/healthz").status_code == 200
    path.write_text('{"not": "a chain"}\n')
    assert client.get("/healthz").status_code == 503


@pytest.mark.parametrize("path", ["/v1/value", "/v1/stress", "/v1/grid"])
def test_typo_in_a_request_field_is_rejected(client, path):
    request = example_request()
    body = {"book": request["book"], "market": request["market"], "sett": "historical"}
    response = client.post(path, headers=ALICE, json=body)
    assert response.status_code == 422
    assert "unknown field" in response.text


def test_uncaught_errors_are_counted(tmp_path, monkeypatch):
    import greeks_neutralizer.service.app as module

    app = create_app({"a": "alice"}, tmp_path / "audit.jsonl")
    client = TestClient(app, raise_server_exceptions=False)
    monkeypatch.setattr(
        module, "value_book", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    )
    request = example_request()
    response = client.post(
        "/v1/value", headers=ALICE, json={"book": request["book"], "market": request["market"]}
    )
    assert response.status_code == 500
    assert "gn_server_errors_total 1" in client.get("/metrics").text


def test_gn_error_type_stays_available():
    assert issubclass(HedgeInfeasibleError, GNError)
    assert value_book([], flat_snapshot()).lines == ()
