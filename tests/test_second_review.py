"""Regression tests from the second independent review. Each failed before its fix.

Oracles are stated where they are not obvious: the FIX 4.4 tag dictionary for tickets, a settled
instrument's definition for scenario P&L, and constructed quote and audit states elsewhere.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.testing import assert_allclose

from greeks_neutralizer import workflow
from greeks_neutralizer.cli import main
from greeks_neutralizer.domain import (
    LinearInstrument,
    LinearKind,
    OptionContract,
    OptionRight,
    Position,
    Quote,
)
from greeks_neutralizer.errors import ValidationError
from greeks_neutralizer.examples import DEFAULT_AS_OF, example_book, example_market, example_request
from greeks_neutralizer.hedging.optimizer import solve_hedge
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
from greeks_neutralizer.pricing.book import value_book
from greeks_neutralizer.risk.scenarios import Scenario, book_scenario_pnl, scenario_pnl_matrix
from greeks_neutralizer.schema import parse_hedge_request, parse_instrument, proposal_to_json
from greeks_neutralizer.service.app import create_app
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

ALICE = {"Authorization": "Bearer a-token"}
BOB = {"Authorization": "Bearer b-token"}
EXPIRY = AS_OF + timedelta(days=30)
# 01:00 UTC on the 21st is 21:00 on the 20th in New York, so the zone decides the date.
EVENING_EXPIRY = datetime(2026, 3, 21, 1, 0, tzinfo=UTC)
STOCK_QUOTE = Quote(99.9, 100.1, 10**6, 10**6, AS_OF)


def option(underlying: str = "SPX", root: str = "") -> OptionContract:
    return OptionContract(underlying, OptionRight.CALL, 100.0, EXPIRY, exchange_symbol=root)


@pytest.fixture(scope="module")
def proposal():
    problem, snapshot, _ = parse_hedge_request(example_request())
    return solve_hedge(problem, snapshot, run_stability=True)


def _leg(instrument, quantity: int = 5) -> HedgeLeg:
    return HedgeLeg(instrument, quantity, 1.25, 0.0, 0.0, "why")


# --- Ticket fields ------------------------------------------------------------------------------

HOSTILE = (
    "XYZ|54=2|38=99999",
    '=HYPERLINK("http://x")',
    "+1",
    "-1+1",
    "@SUM(A1)",
    "A B",
    "A\n",
    "A\x01B",
    "A" * 33,
)


@pytest.mark.parametrize("bad", HOSTILE)
def test_ticket_symbols_refuse_hostile_text_at_construction(bad):
    with pytest.raises(ValidationError, match="exchange_symbol must be"):
        option(root=bad)
    with pytest.raises(ValidationError, match="underlying must be"):
        option(underlying=bad)
    with pytest.raises(ValidationError, match="symbol must be"):
        LinearInstrument(bad, "XYZ")
    with pytest.raises(ValidationError, match="underlying must be"):
        LinearInstrument("XYZ", bad)


@pytest.mark.parametrize("bad", ["", 5, None, ["SPX"]])
def test_required_symbols_must_be_non_empty_text(bad):
    with pytest.raises(ValidationError, match="underlying must be"):
        option(underlying=bad)
    with pytest.raises(ValidationError, match="symbol must be"):
        LinearInstrument(bad, "XYZ")


@pytest.mark.parametrize("good", ["SPX", "SPXW", "BRK.B", "BF-B", "B1SP", "^SPX", "_SPX", "A" * 32])
def test_real_roots_and_index_symbols_are_admitted(good):
    assert option(root=good).exchange_symbol == good
    assert option(underlying=good).underlying == good
    assert LinearInstrument(good, good).symbol == good


def test_injected_root_never_reaches_a_ticket_through_the_schema():
    request = example_request()
    request["candidates"][1]["instrument"]["exchange_symbol"] = "XYZ|54=2|38=99999"
    with pytest.raises(ValidationError, match="exchange_symbol"):
        parse_hedge_request(request)


def test_the_approver_sees_the_root_the_ticket_names(proposal):
    contract = OptionContract(
        "SPX", OptionRight.CALL, 100.0, EVENING_EXPIRY, exchange_symbol="SPXW"
    )
    fields = dict(
        part.split("=")
        for part in tickets_fix(replace(proposal, legs=(_leg(contract),)), "abc", AS_OF).split("|")
        if part.strip()
    )
    assert fields["55"] == "SPXW"
    assert fields["55"] in contract.describe()
    # Same calendar date on the screen and on the ticket: tag 541 is YYYYMMDD in exchange time.
    assert fields["541"] == "20260320"
    assert "2026-03-20" in contract.describe()


def test_future_ticket_identifies_the_contract(proposal):
    future = LinearInstrument(
        "ES", "SPX", LinearKind.FUTURE, multiplier=50.0, expiry=EVENING_EXPIRY
    )
    message = tickets_fix(replace(proposal, legs=(_leg(future),)), "abc", AS_OF).strip().rstrip("|")
    fields = dict(part.split("=") for part in message.split("|"))
    # FIX 4.4: 167 security type, 541 maturity date in exchange time, 231 contract multiplier.
    assert fields["167"] == "FUT"
    assert fields["541"] == "20260320"
    assert fields["231"] == "50"
    assert "2026-03-20" in future.describe()


def test_stock_ticket_has_no_maturity(proposal):
    message = tickets_fix(
        replace(proposal, legs=(_leg(LinearInstrument("XYZ", "XYZ")),)), "abc", AS_OF
    )
    fields = dict(part.split("=") for part in message.strip().rstrip("|").split("|"))
    assert fields["167"] == "CS"
    assert "541" not in fields
    assert "231" not in fields


@pytest.mark.parametrize("symbol", ["=1+1", "+1", "-1", "@SUM(A1)", "=HYPERLINK(1)", "A,B", "A B"])
def test_csv_ticket_cells_cannot_start_a_formula(symbol):
    # Spreadsheets run a cell starting = + - or @. The ticket writes symbols as they are, so such a
    # symbol must never become an instrument, by any of the three fields a ticket prints.
    with pytest.raises(ValidationError, match="symbol must be"):
        LinearInstrument(symbol, "XYZ")
    with pytest.raises(ValidationError, match="underlying must be"):
        LinearInstrument("XYZ", symbol)
    with pytest.raises(ValidationError, match="exchange_symbol"):
        option(root=symbol)


def test_csv_ticket_prints_the_symbols_a_trader_expects(proposal):
    legs = (_leg(option(root="SPXW")), _leg(LinearInstrument("^VIX", "^VIX")))
    rows = list(csv.DictReader(io.StringIO(tickets_csv(replace(proposal, legs=legs), "abc"))))
    assert [row["symbol"] for row in rows] == ["SPXW", "^VIX"]


@pytest.mark.parametrize(
    ("kind", "basis_vol"), [(LinearKind.SPOT, 0.0), (LinearKind.PROXY, 0.1)], ids=["spot", "proxy"]
)
def test_only_a_future_has_an_expiry(kind, basis_vol):
    with pytest.raises(ValidationError, match="only a future has an expiry"):
        LinearInstrument("XYZ", "XYZ", kind, basis_vol=basis_vol, expiry=EXPIRY)


# --- Currency -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: OptionContract("XYZ", OptionRight.CALL, 100.0, EXPIRY, currency="EUR"),
        lambda: LinearInstrument("XYZ", "XYZ", currency="EUR"),
        lambda: parse_instrument(
            {
                "type": "option",
                "underlying": "XYZ",
                "right": "call",
                "strike": 100.0,
                "expiry": EXPIRY.isoformat(),
                "currency": "EUR",
            }
        ),
        lambda: parse_instrument(
            {"type": "linear", "symbol": "XYZ", "underlying": "XYZ", "currency": "GBP"}
        ),
    ],
)
def test_non_usd_fails_closed_until_fx_exists(build):
    with pytest.raises(ValidationError, match="USD"):
        build()


def test_explicit_usd_is_accepted():
    assert LinearInstrument("XYZ", "XYZ", currency="USD").currency == "USD"


# --- Market clock -------------------------------------------------------------------------------


class Clock:
    def __init__(self, now: datetime = DEFAULT_AS_OF) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def store(tmp_path):
    return ProposalStore(AuditLog(tmp_path / "audit.jsonl"), clock=Clock())


def test_market_dated_in_the_future_cannot_be_submitted(store, proposal):
    store.clock.now = DEFAULT_AS_OF - timedelta(days=365)
    with pytest.raises(WorkflowError, match="future"):
        store.submit(proposal, "alice")


def test_market_dated_in_the_future_cannot_be_approved_or_exported(store, proposal):
    record = store.submit(proposal, "alice")
    store.clock.now = DEFAULT_AS_OF - timedelta(minutes=1)
    with pytest.raises(WorkflowError, match="future"):
        store.approve(record.id, "bob")
    store.clock.now = DEFAULT_AS_OF
    store.approve(record.id, "bob")
    store.clock.now = DEFAULT_AS_OF - timedelta(minutes=1)
    with pytest.raises(WorkflowError, match="future"):
        store.export_tickets(record.id, "bob")


def test_clock_skew_inside_the_quote_tolerance_is_not_an_incident(store, proposal):
    store.clock.now = DEFAULT_AS_OF - timedelta(milliseconds=500)
    assert store.submit(proposal, "alice").proposal is proposal


# --- Coverage -----------------------------------------------------------------------------------


def _two_underlyings() -> MarketSnapshot:
    curve = DiscountCurve.flat(0.05)
    return MarketSnapshot(
        AS_OF,
        {
            "XYZ": UnderlyingMarket("XYZ", 100.0, curve, FlatVol(0.25)),
            "AAA": UnderlyingMarket("AAA", 100.0, curve, FlatVol(0.25)),
        },
    )


def _coverage_problem(aaa_candidates: tuple[HedgeCandidate, ...]) -> HedgeProblem:
    return HedgeProblem(
        (
            Position(euro("call", 100, 30), 10),
            Position(euro("call", 100, 30, underlying="AAA"), 10),
        ),
        (HedgeCandidate(LinearInstrument("XYZ", "XYZ"), STOCK_QUOTE), *aaa_candidates),
        (
            GreekTarget("XYZ", GreekName.DELTA, weight=1.0),
            GreekTarget("AAA", GreekName.DELTA, weight=1.0),
        ),
    )


@pytest.mark.parametrize(
    "aaa_candidates",
    [
        pytest.param(
            (
                HedgeCandidate(
                    LinearInstrument("AAA", "AAA"),
                    replace(STOCK_QUOTE, timestamp=AS_OF - timedelta(hours=1)),
                ),
            ),
            id="every-candidate-excluded",
        ),
        pytest.param((), id="none-offered"),
        pytest.param(
            (HedgeCandidate(LinearInstrument("AAA", "AAA"), STOCK_QUOTE, max_buy=0, max_sell=0),),
            id="usable-with-no-capacity",
        ),
    ],
)
def test_targeted_underlying_with_nothing_tradable_cannot_read_as_clean(aaa_candidates):
    p = solve_hedge(_coverage_problem(aaa_candidates), _two_underlyings())
    assert {leg.instrument.underlying for leg in p.legs} == {"XYZ"}
    assert p.status.value == "partial_coverage"
    assert any(i.startswith("NO_USABLE_CANDIDATES AAA") for i in p.issues)
    assert not any(i.startswith("NO_USABLE_CANDIDATES XYZ") for i in p.issues)


def test_fully_covered_book_stays_clean():
    candidates = (HedgeCandidate(LinearInstrument("AAA", "AAA"), STOCK_QUOTE),)
    p = solve_hedge(_coverage_problem(candidates), _two_underlyings())
    assert {leg.instrument.underlying for leg in p.legs} == {"XYZ", "AAA"}
    assert p.status.value in ("optimal", "within_gap")
    assert not any("NO_USABLE_CANDIDATES" in i for i in p.issues)


def test_all_excluded_keeps_its_own_status_with_one_issue():
    stale = replace(STOCK_QUOTE, timestamp=AS_OF - timedelta(hours=1))
    problem = HedgeProblem(
        (Position(euro("call", 100, 30), 10),),
        (HedgeCandidate(LinearInstrument("XYZ", "XYZ"), stale),),
        (GreekTarget("XYZ", GreekName.DELTA, weight=1.0),),
    )
    p = solve_hedge(problem, flat_snapshot())
    assert p.status.value == "no_usable_candidates"
    assert sum("NO_USABLE_CANDIDATES" in i for i in p.issues) == 1


# --- Expired instruments ------------------------------------------------------------------------


def test_settled_instruments_do_not_move_in_any_scenario():
    future = LinearInstrument(
        "ES", "XYZ", LinearKind.FUTURE, multiplier=50.0, expiry=AS_OF - timedelta(days=1)
    )
    call = euro("call", 80, -1)
    scenarios = (Scenario("down", spot_shock=-0.1), Scenario("later", -0.1, days_forward=7.0))
    snapshot = flat_snapshot()
    assert_allclose(scenario_pnl_matrix([future, call], snapshot, scenarios), np.zeros((2, 2)))
    book = (Position(future, 10), Position(call, -3))
    assert_allclose(book_scenario_pnl(book, snapshot, scenarios), [0.0, 0.0])


def test_live_future_still_moves_with_its_forward():
    expiry = expiry_in(365)
    future = LinearInstrument("ES", "XYZ", LinearKind.FUTURE, multiplier=50.0, expiry=expiry)
    pnl = scenario_pnl_matrix([future], flat_snapshot(), (Scenario("down", spot_shock=-0.1),))
    # F = S exp(rT) with no dividends, so -10% on spot is -10% on the forward.
    assert_allclose(pnl[0, 0], -0.1 * 100.0 * math.exp(0.05) * 50.0, rtol=1e-12)


def test_instrument_expiring_inside_the_horizon_settles_at_the_shocked_spot():
    call = euro("call", 80, 3)
    snapshot = flat_snapshot()
    base = value_book([Position(call, 1)], snapshot).lines[0].exposure.value
    pnl = scenario_pnl_matrix([call], snapshot, (Scenario("x", -0.1, days_forward=7.0),))
    assert_allclose(pnl[0, 0], (90.0 - 80.0) * 100.0 - base, rtol=1e-9)


# --- Keys and fingerprints ----------------------------------------------------------------------


def test_duplicate_candidates_are_refused():
    candidate = HedgeCandidate(euro("call", 100, 30), Quote(3.0, 3.1, 100, 100, AS_OF), max_buy=3)
    with pytest.raises(ValidationError, match="duplicate candidate"):
        HedgeProblem((), (candidate, candidate), (GreekTarget("XYZ", GreekName.DELTA),))


def test_duplicate_scenario_names_are_refused():
    with pytest.raises(ValidationError, match="duplicate scenario"):
        HedgeProblem((), (), (), (Scenario("a", spot_shock=-0.1), Scenario("a", spot_shock=-0.2)))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (
            LinearInstrument("SPY", "SPX", LinearKind.PROXY, delta_ratio=0.1, basis_vol=0.01),
            LinearInstrument("SPY", "SPX", LinearKind.PROXY, delta_ratio=0.2, basis_vol=0.01),
        ),
        (
            LinearInstrument("SPY", "SPX", LinearKind.PROXY, basis_vol=0.01),
            LinearInstrument("SPY", "NDX", LinearKind.PROXY, basis_vol=0.01),
        ),
        (
            LinearInstrument("SPY", "SPX", LinearKind.PROXY, basis_vol=0.01),
            LinearInstrument("SPY", "SPX", LinearKind.PROXY, basis_vol=0.02),
        ),
        (option(root="SPX"), option(root="SPXW")),
    ],
)
def test_distinct_instruments_have_distinct_keys(a, b):
    assert a.key != b.key


def test_fingerprint_covers_every_input_that_changes_the_answer():
    base_problem = HedgeProblem(
        (Position(euro("call", 100, 30), 1),),
        (HedgeCandidate(LinearInstrument("XYZ", "XYZ"), STOCK_QUOTE),),
        (GreekTarget("XYZ", GreekName.DELTA, weight=1.0),),
        (Scenario("s", spot_shock=-0.1),),
    )
    snapshot = flat_snapshot()

    def fingerprint(problem):
        return solve_hedge(problem, snapshot).manifest.problem_fingerprint

    base = fingerprint(base_problem)
    assert base == fingerprint(replace(base_problem))
    candidate = base_problem.candidates[0]
    proxy = LinearInstrument("SPY", "XYZ", LinearKind.PROXY, basis_vol=0.01)
    changes = {
        "book quantity": {"book": (Position(euro("call", 100, 30), 2),)},
        "book strike": {"book": (Position(euro("call", 101, 30), 1),)},
        "candidate fee": {"candidates": (replace(candidate, fee_per_unit=0.01),)},
        "candidate quote": {
            "candidates": (replace(candidate, quote=replace(STOCK_QUOTE, bid=99.8)),)
        },
        "candidate instrument": {"candidates": (replace(candidate, instrument=proxy),)},
        "target weight": {"targets": (GreekTarget("XYZ", GreekName.DELTA, weight=2.0),)},
        "target band": {
            "targets": (GreekTarget("XYZ", GreekName.DELTA, lower=-1e7, upper=1e7, weight=1.0),)
        },
        "scenario shock": {"scenarios": (Scenario("s", spot_shock=-0.2),)},
        "scenario sticky": {"scenarios": (Scenario("s", spot_shock=-0.1, sticky_strike=False),)},
        "config aversion": {"config": HedgeConfig(risk_aversion=2.0)},
        "config gate": {"config": HedgeConfig(max_lots_per_leg=4000)},
    }
    seen = {base}
    for name, change in changes.items():
        changed = fingerprint(replace(base_problem, **change))
        assert changed not in seen, name
        seen.add(changed)
    stability = solve_hedge(base_problem, snapshot, run_stability=True)
    assert stability.manifest.problem_fingerprint != base


# --- Audit log ----------------------------------------------------------------------------------


def _seed(path: Path, records: int = 1) -> AuditLog:
    log = AuditLog(path)
    for n in range(records):
        log.append("event", "alice", {"n": n}, DEFAULT_AS_OF)
    return log


def test_verification_beside_appends_never_raises_a_false_alarm(tmp_path, monkeypatch):
    # The verify starts in the one moment the log is ahead of its sidecar, between the two writes
    # of an append. With the lock it blocks until they agree. Without the lock it reads that moment
    # and reports a broken chain. The append holds there until one or the other has happened.
    log = _seed(tmp_path / "a.jsonl")
    between_the_writes = threading.Event()
    verify_blocked_or_done = threading.Event()
    real_write = workflow._write_sidecar

    class WatchedLock:
        def __init__(self, lock):
            self.lock = lock

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                verify_blocked_or_done.set()
                self.lock.acquire()

        def __exit__(self, *exc):
            self.lock.release()

    def write(*args):
        between_the_writes.set()
        assert verify_blocked_or_done.wait(30.0)
        real_write(*args)

    log._lock = WatchedLock(log._lock)
    monkeypatch.setattr(workflow, "_write_sidecar", write)
    results = []

    def verify():
        assert between_the_writes.wait(30.0)
        results.append(log.verify())
        verify_blocked_or_done.set()

    verifier = threading.Thread(target=verify)
    verifier.start()
    log.append("event", "alice", {"n": 1}, DEFAULT_AS_OF)
    verifier.join(timeout=30.0)
    assert not verifier.is_alive()
    assert [check.ok for check in results] == [True]
    assert results[0].records == 2


def test_crash_before_the_head_is_replaced_keeps_it_readable_and_is_named(tmp_path, monkeypatch):
    path = tmp_path / "a.jsonl"
    log = _seed(path)
    head_before = (tmp_path / "a.jsonl.head").read_text()

    def crash(*_: object) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr("greeks_neutralizer.workflow.os.replace", crash)
        with pytest.raises(OSError, match="disk full"):
            log.append("event", "alice", {}, DEFAULT_AS_OF)
    assert (tmp_path / "a.jsonl.head").read_text() == head_before
    check = verify_audit_log(path)
    assert not check.ok
    assert "longer than" in check.reason
    assert "tail" not in check.reason
    assert check.first_bad_line == 2


def test_interrupted_write_still_blocks_after_reopening(tmp_path, proposal, monkeypatch):
    path = tmp_path / "a.jsonl"
    log = _seed(path)
    with monkeypatch.context() as patched:
        patched.setattr(
            "greeks_neutralizer.workflow.os.replace",
            lambda *_: (_ for _ in ()).throw(OSError("crash")),
        )
        with pytest.raises(OSError, match="crash"):
            log.append("event", "alice", {}, DEFAULT_AS_OF)
    store = ProposalStore(AuditLog(path), clock=Clock())
    with pytest.raises(WorkflowError, match="longer than"):
        store.submit(proposal, "alice")


@pytest.mark.parametrize("content", ["", "{", "[1]", '{"head": "x"}', '{"head": 1, "count": "2"}'])
def test_unreadable_sidecar_is_a_failed_verification(tmp_path, content):
    path = tmp_path / "a.jsonl"
    _seed(path)
    (tmp_path / "a.jsonl.head").write_text(content)
    check = verify_audit_log(path)
    assert not check.ok
    assert "sidecar" in check.reason


def test_deleted_sidecar_is_detected(tmp_path):
    path = tmp_path / "a.jsonl"
    _seed(path, 2)
    (tmp_path / "a.jsonl.head").unlink()
    check = verify_audit_log(path)
    assert not check.ok
    assert check.first_bad_line == 1


def test_wrong_head_hash_with_the_right_count_is_detected(tmp_path):
    path = tmp_path / "a.jsonl"
    _seed(path)
    (tmp_path / "a.jsonl.head").write_text(json.dumps({"head": "0" * 64, "count": 1}))
    check = verify_audit_log(path)
    assert not check.ok
    assert "hash" in check.reason


def test_rejection_cannot_paper_over_a_broken_chain(tmp_path, proposal):
    path = tmp_path / "a.jsonl"
    store = ProposalStore(AuditLog(path), clock=Clock())
    first = store.submit(proposal, "alice")
    second = store.submit(proposal, "alice")
    assert first.id != second.id
    # Removing the tail leaves a valid prefix, which only the sidecar can catch. An append would
    # rewrite the sidecar and hide that, so rejecting has to be blocked as well.
    path.write_text(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(WorkflowError, match="audit log"):
        store.reject(first.id, "bob", "no")
    assert not verify_audit_log(path).ok


DAMAGE = [
    pytest.param(b"not json\n", id="text"),
    pytest.param(b"[1, 2]\n", id="list"),
    pytest.param(b"5\n", id="number"),
    pytest.param(b'"text"\n', id="string"),
    pytest.param(b'{"no_hash": 1}\n', id="no-hash"),
    pytest.param(b'{"hash": 5}\n', id="hash-not-text"),
    pytest.param(b"\xff\xfe\n", id="binary"),
    pytest.param(b"[" * 200_000 + b"\n", id="deep-nesting"),
]


@pytest.mark.parametrize("line", DAMAGE)
def test_damaged_line_is_reported_at_every_entry_point(tmp_path, line):
    path = tmp_path / "audit.jsonl"
    _seed(path)
    with path.open("ab") as handle:
        handle.write(line)
    check = verify_audit_log(path)
    assert not check.ok
    assert check.first_bad_line == 2

    app = create_app({"a-token": "alice"}, path, clock=lambda: DEFAULT_AS_OF)
    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/healthz").status_code == 503
    response = client.get("/v1/audit", headers=ALICE)
    assert response.status_code == 200
    body = response.json()
    assert body["chain_ok"] is False
    assert body["first_bad_line"] == 2
    assert body["unreadable_lines"] == 1
    assert [r["event"] for r in body["records"]] == ["event"]


def test_audit_endpoint_returns_a_line_with_nan_as_json(tmp_path):
    path = tmp_path / "audit.jsonl"
    _seed(path)
    with path.open("a") as handle:
        handle.write('{"hash": "x", "value": NaN}\n')
    app = create_app({"a-token": "alice"}, path, clock=lambda: DEFAULT_AS_OF)
    response = TestClient(app, raise_server_exceptions=False).get("/v1/audit", headers=ALICE)
    assert response.status_code == 200
    assert response.json()["records"][0]["value"] is None


@pytest.fixture
def client(tmp_path):
    app = create_app(
        {"a-token": "alice", "b-token": "bob"},
        tmp_path / "audit.jsonl",
        200_000,
        clock=lambda: DEFAULT_AS_OF,
    )
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("limit", [0, -1])
def test_audit_limit_must_be_positive(client, limit):
    assert client.get(f"/v1/audit?limit={limit}", headers=ALICE).status_code == 422


def test_audit_limit_returns_the_newest_records(tmp_path):
    path = tmp_path / "audit.jsonl"
    _seed(path, 3)
    app = create_app({"a-token": "alice"}, path, clock=lambda: DEFAULT_AS_OF)
    response = TestClient(app).get("/v1/audit?limit=1", headers=ALICE)
    assert [r["detail"]["n"] for r in response.json()["records"]] == [2]


# --- Typed input --------------------------------------------------------------------------------

SLICE = {"tau": 0.25, "a": 0.005, "b": 0.06, "rho": -0.6, "m": 0.02, "sigma": 0.15}


def _config(**fields):
    def apply(request):
        request["config"].update(fields)

    return apply


def _xyz(**fields):
    def apply(request):
        market = request["market"]["underlyings"]["XYZ"]
        market.pop("rate")
        market.update(fields)

    return apply


def _target(field, value):
    def apply(request):
        request["targets"][0][field] = value

    return apply


def _candidate(field, value):
    def apply(request):
        request["candidates"][1][field] = value

    return apply


def _quote(field, value):
    def apply(request):
        request["candidates"][1]["quote"][field] = value

    return apply


def _scenario(**fields):
    def apply(request):
        request["scenarios"] = [{"name": "s", **fields}]

    return apply


def _set(field, value):
    def apply(request):
        request[field] = value

    return apply


def _surface(surface):
    def apply(request):
        request["market"]["underlyings"]["XYZ"]["surface"] = surface

    return apply


def _spot(value):
    def apply(request):
        request["market"]["underlyings"]["XYZ"]["spot"] = value

    return apply


BAD_REQUESTS = {
    "max_legs-text": _config(max_legs="x"),
    "max_legs-fraction": _config(max_legs=1.5),
    "max_legs-bool": _config(max_legs=True),
    "max_lots-text": _config(max_lots_per_leg="x"),
    "max_quote_age-text": _config(max_quote_age_seconds="x"),
    "max_quote_age-nan": _config(max_quote_age_seconds=float("nan")),
    "max_quote_age-huge": _config(max_quote_age_seconds=1e30),
    "curve-times-number": _xyz(curve={"times": 5, "zero_rates": [0.05]}),
    "curve-times-text": _xyz(curve={"times": ["a"], "zero_rates": [0.05]}),
    "curve-times-string": _xyz(curve={"times": "1", "zero_rates": [0.05]}),
    "curve-times-nan": _xyz(curve={"times": [float("nan")], "zero_rates": [0.05]}),
    "curve-rates-number": _xyz(curve={"times": [1.0], "zero_rates": 5}),
    "curve-not-object": _xyz(curve=5),
    "quoted-ranges-number": _surface({"type": "svi", "slices": [SLICE], "quoted_ranges": [1]}),
    "quoted-ranges-text": _surface(
        {"type": "svi", "slices": [SLICE], "quoted_ranges": [["a", "b"]]}
    ),
    "quoted-ranges-nan": _surface(
        {"type": "svi", "slices": [SLICE], "quoted_ranges": [[float("nan"), 1.0]]}
    ),
    "slice-text": _surface({"type": "svi", "slices": [{**SLICE, "a": "x"}]}),
    "sigma-text": _surface({"type": "flat", "sigma": "x"}),
    "spot-text": _spot("x"),
    "target-upper-text": _target("upper", "abc"),
    "target-lower-text": _target("lower", "abc"),
    "target-upper-nan": _target("upper", float("nan")),
    "target-weight-nan": _target("weight", float("nan")),
    "target-weight-inf": _target("weight", float("inf")),
    "target-weight-text": _target("weight", "x"),
    "target-underlying-number": _target("underlying", 5),
    "target-underlying-list": _target("underlying", ["XYZ"]),
    "lot-size-fraction": _candidate("lot_size", 1.5),
    "lot-size-text": _candidate("lot_size", "2"),
    "max-buy-fraction": _candidate("max_buy", 1.5),
    "max-sell-text": _candidate("max_sell", "x"),
    "bid-size-fraction": _quote("bid_size", 1.5),
    "ask-size-text": _quote("ask_size", "x"),
    "term-power-text": _scenario(term_power="x"),
    "term-power-nan": _scenario(term_power=float("nan")),
    "sticky-strike-text": _scenario(sticky_strike="no"),
    "spot-shock-text": _scenario(spot_shock="x"),
    "scenario-name-number": _scenario(name=5),
    "run-stability-text": _set("run_stability", "false"),
    "run-stability-number": _set("run_stability", 1),
}


@pytest.mark.parametrize("mutate", BAD_REQUESTS.values(), ids=BAD_REQUESTS.keys())
def test_wrongly_typed_input_is_a_validation_error(mutate):
    request = copy.deepcopy(example_request())
    mutate(request)
    with pytest.raises(
        ValidationError, match=r"must be|quoted range|could not convert|too large|not supported"
    ):
        parse_hedge_request(request)


@pytest.mark.parametrize("mutate", BAD_REQUESTS.values(), ids=BAD_REQUESTS.keys())
def test_wrongly_typed_input_is_a_422_from_the_service(client, mutate):
    request = copy.deepcopy(example_request())
    mutate(request)
    # The server's JSON parser accepts NaN and Infinity, so they do reach the schema.
    response = client.post(
        "/v1/hedge",
        headers={**ALICE, "Content-Type": "application/json"},
        content=json.dumps(request),
    )
    assert response.status_code == 422, response.text


def test_wrongly_typed_input_is_a_structured_cli_error(tmp_path, capfd):
    request = copy.deepcopy(example_request())
    request["config"]["max_legs"] = "x"
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    assert main(["hedge", "--request", str(path)]) == 1
    assert "max_legs" in capfd.readouterr().out


@pytest.mark.parametrize(
    "build",
    [
        lambda: HedgeConfig(max_quote_age="x"),
        lambda: HedgeCandidate(LinearInstrument("XYZ", "XYZ"), STOCK_QUOTE, lot_size=1.5),
        lambda: Quote(1.0, 2.0, 1.5, 1, AS_OF),
        lambda: DiscountCurve(("a",), (0.05,)),
    ],
)
def test_constructors_refuse_wrong_types_directly(build):
    with pytest.raises(ValidationError, match="must be"):
        build()


def test_numpy_integers_are_still_integers():
    assert Quote(1.0, 2.0, np.int64(3), np.int64(4), AS_OF).ask_size == 4


def test_decision_reason_must_be_text(client):
    submitted = client.post("/v1/proposals", headers=ALICE, json=example_request())
    assert submitted.status_code == 201, submitted.text
    record = submitted.json()["id"]
    for action in ("approve", "reject"):
        response = client.post(f"/v1/proposals/{record}/{action}", headers=BOB, json={"reason": 5})
        assert response.status_code == 422, response.text
    assert client.get(f"/v1/proposals/{record}", headers=BOB).json()["state"] == "proposed"


def test_scenario_set_must_be_a_known_name(client):
    body = {"book": example_book(), "market": example_market(), "set": []}
    assert client.post("/v1/stress", headers=ALICE, json=body).status_code == 422


def test_grid_refuses_a_scenario_set_it_would_ignore(client):
    body = {"book": example_book(), "market": example_market(), "set": "historical"}
    assert client.post("/v1/grid", headers=ALICE, json=body).status_code == 422


# --- Policy ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    [
        WorkflowPolicy(),
        WorkflowPolicy(approval_valid_for=timedelta(days=1), max_market_age=timedelta(days=1)),
        WorkflowPolicy(approval_valid_for=timedelta(seconds=1), max_market_age=timedelta(1)),
    ],
    ids=["default", "generous", "short"],
)
def test_no_policy_lets_a_proposer_approve_or_an_unstable_proposal_through(
    tmp_path, proposal, policy
):
    # The requirement, whatever a policy allows: four eyes, and no approval of an unstable hedge.
    store = ProposalStore(AuditLog(tmp_path / "a.jsonl"), policy, lambda: DEFAULT_AS_OF)
    record = store.submit(proposal, "alice")
    with pytest.raises(WorkflowError, match="four-eyes"):
        store.approve(record.id, "alice")
    unstable = store.submit(replace(proposal, status=ProposalStatus.UNSTABLE), "alice")
    with pytest.raises(WorkflowError, match="unstable"):
        store.approve(unstable.id, "bob")
    assert store.approve(record.id, "bob").state.value == "approved"


def test_status_survives_serialization():
    candidates = (
        HedgeCandidate(LinearInstrument("AAA", "AAA"), STOCK_QUOTE, max_buy=0, max_sell=0),
    )
    p = solve_hedge(_coverage_problem(candidates), _two_underlyings())
    assert proposal_to_json(p)["status"] == "partial_coverage"
