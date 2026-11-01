"""Regression tests from the fourth review round. Each failed before its fix.

Oracles: a brute-force search over lots, confirmed by repricing, shows a hedge exists; RFC 9112
framing through a real h11 server for the body limit; and each gate's direction from the meaning of
the gate. The solver is replaced only at its SciPy boundary, to produce statuses too rare to build.
"""

from __future__ import annotations

import itertools
import json
import socket
import threading
import time
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient
from scipy.optimize import OptimizeResult
from scipy.optimize import milp as real_milp

from greeks_neutralizer.cli import main
from greeks_neutralizer.domain import LinearInstrument, LinearKind, Position, Quote
from greeks_neutralizer.errors import HedgeInfeasibleError, SearchLimitError
from greeks_neutralizer.examples import example_request
from greeks_neutralizer.hedging import optimizer
from greeks_neutralizer.hedging.diagnostics import hedge_frontier
from greeks_neutralizer.hedging.optimizer import solve_hedge
from greeks_neutralizer.hedging.problem import (
    GreekName,
    GreekTarget,
    HedgeCandidate,
    HedgeConfig,
    HedgeProblem,
    gate_overrides,
)
from greeks_neutralizer.market import DiscountCurve, FlatVol, MarketSnapshot, UnderlyingMarket
from greeks_neutralizer.pricing.book import unit_exposures, value_book, value_options
from greeks_neutralizer.schema import parse_hedge_request
from greeks_neutralizer.service.app import create_app
from greeks_neutralizer.workflow import AuditLog, ProposalStore
from tests.conftest import AS_OF, euro, expiry_in, flat_snapshot

SPOT = 755.0
BAND = 100.0
STOCK = LinearInstrument("XYZ", "XYZ")
HIGH_PRICE = MarketSnapshot(
    AS_OF, {"XYZ": UnderlyingMarket("XYZ", SPOT, DiscountCurve.flat(0.05), FlatVol(0.2))}
)
OPTIONS = (
    euro("call", 780, 45),
    euro("put", 700, 45),
    euro("call", 755, 60),
    euro("put", 730, 60),
)
ALICE = {"Authorization": "Bearer a"}


def _quoted(contract):
    price = float(value_options([contract], HIGH_PRICE).greeks.price[0])
    return HedgeCandidate(contract, Quote(max(price - 0.05, 0.0), price + 0.05, 1000, 1000, AS_OF))


def _share_too_coarse_for_the_band(quantity=1):
    """One share is $755 of delta, so no share count alone can land inside a $100 band."""
    book = (Position(euro("call", 755, 30), -quantity), Position(euro("put", 730, 30), -quantity))
    stock = HedgeCandidate(STOCK, Quote(754.99, 755.01, 10**6, 10**6, AS_OF))
    targets = (
        GreekTarget("XYZ", GreekName.DELTA, lower=-BAND, upper=BAND),
        GreekTarget("XYZ", GreekName.GAMMA),
        GreekTarget("XYZ", GreekName.VEGA),
    )
    return HedgeProblem(book, (stock, *(_quoted(c) for c in OPTIONS)), targets)


def _brute_force_hedge(problem):
    """Any whole-number hedge inside the band, found with no solver, or None."""
    start = value_book(problem.book, HIGH_PRICE).by_underlying["XYZ"].delta_dollars
    units = [e.delta_dollars for e in unit_exposures(OPTIONS, HIGH_PRICE)]
    for lots in itertools.product(range(-6, 7), repeat=len(OPTIONS)):
        before_shares = start + float(np.dot(lots, units))
        shares = round(-before_shares / SPOT)
        if abs(before_shares + shares * SPOT) <= BAND:
            return shares, lots
    return None


@pytest.mark.parametrize("quantity", range(1, 11))
def test_a_band_tighter_than_a_share_finds_the_hedge_that_exists(quantity):
    problem = _share_too_coarse_for_the_band(quantity)
    found = _brute_force_hedge(problem)
    assert found is not None
    shares, lots = found
    positions = (
        *problem.book,
        Position(STOCK, shares),
        *(Position(c, q) for c, q in zip(OPTIONS, lots, strict=True) if q),
    )
    assert abs(value_book(positions, HIGH_PRICE).by_underlying["XYZ"].delta_dollars) <= BAND

    proposal = solve_hedge(problem, HIGH_PRICE)
    assert abs(proposal.after["XYZ"].delta_dollars) <= BAND
    assert proposal.verification.passed


@pytest.mark.parametrize("quantity", range(1, 11))
def test_stability_does_not_call_such_a_band_fragile(quantity):
    proposal = solve_hedge(_share_too_coarse_for_the_band(quantity), HIGH_PRICE, run_stability=True)
    assert proposal.stability is not None
    assert not any("fragile" in note for note in proposal.stability.notes)


@pytest.mark.parametrize("quantity", range(1, 11))
def test_frontier_does_not_call_such_a_band_infeasible(quantity):
    problem = _share_too_coarse_for_the_band(quantity)
    point = hedge_frontier(problem, HIGH_PRICE, risk_aversions=(1.0,))[-1]
    assert point.status != "infeasible"


def _script_milp(monkeypatch, script):
    calls = {"n": 0}

    def fake(*args, **kwargs):
        step = script[min(calls["n"], len(script) - 1)]
        calls["n"] += 1
        return step(real_milp(*args, **kwargs))

    monkeypatch.setattr(optimizer, "milp", fake)


def _infeasible_here(result):
    return OptimizeResult(x=None, status=2, message="The problem is infeasible.", fun=None)


def test_infeasible_in_every_window_is_not_a_proof_until_the_full_solve_says_so(monkeypatch):
    _script_milp(monkeypatch, [_infeasible_here] * 4 + [lambda result: result])
    assert solve_hedge(_share_too_coarse_for_the_band(), HIGH_PRICE).verification.passed


def test_integer_answers_that_keep_missing_the_band_are_not_called_infeasible(monkeypatch):
    def flat(result):
        return OptimizeResult(
            x=np.zeros_like(result.x), status=0, message=result.message, fun=result.fun
        )

    _script_milp(monkeypatch, [flat])
    with pytest.raises(SearchLimitError, match="mip_node_limit") as err:
        solve_hedge(_share_too_coarse_for_the_band(), HIGH_PRICE)
    assert not isinstance(err.value, HedgeInfeasibleError)


# --- Body limit, through a real h11 server --------------------------------------------------------


@pytest.fixture
def live_port(tmp_path):
    app = create_app({"a": "alice"}, tmp_path / "audit.jsonl", 1000)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, http="h11", log_level="critical")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started
    yield port
    server.should_exit = True
    thread.join(timeout=10.0)


def _raw_post(port, headers, body):
    head = "".join(f"{name}: {value}\r\n" for name, value in headers.items())
    request = (
        f"POST /v1/value HTTP/1.1\r\nHost: test\r\nAuthorization: Bearer a\r\n"
        f"Connection: close\r\n{head}\r\n"
    )
    with socket.create_connection(("127.0.0.1", port), timeout=10.0) as conn:
        conn.sendall(request.encode() + body)
        status_line = conn.makefile("rb").readline().decode()
    return int(status_line.split()[1])


def _chunked(size):
    payload = b"x" * size
    return f"{size:x}\r\n".encode() + payload + b"\r\n0\r\n\r\n"


@pytest.mark.parametrize(
    ("headers", "body", "expected"),
    [
        ({"Content-Length": "1", "Transfer-Encoding": "chunked"}, _chunked(50_000), 411),
        ({"Transfer-Encoding": "chunked"}, _chunked(50_000), 411),
        ({"Content-Length": "50000"}, b"x" * 50_000, 413),
    ],
    ids=["both-headers", "chunked-alone", "oversized-length"],
)
def test_a_body_past_the_limit_never_reaches_the_app(live_port, headers, body, expected):
    assert _raw_post(live_port, headers, body) == expected


def test_a_small_declared_body_still_reaches_the_app(live_port):
    assert _raw_post(live_port, {"Content-Length": "2"}, b"{}") == 422


# --- Gates: enforced at the service, disclosed everywhere -----------------------------------------


LOOSER = {
    "max_quote_age_seconds": 1e9,
    "instability_threshold": 0.9,
    "regret_threshold": 0.9,
    "max_model_quote_vol_gap": 0.5,
    "max_lots_per_leg": 5001,
    "allow_extrapolated_vol": True,
    "require_quote_size": False,
}
TIGHTER = {
    "max_quote_age_seconds": 5.0,
    "instability_threshold": 0.4,
    "regret_threshold": 0.05,
    "max_model_quote_vol_gap": 0.04,
    "max_lots_per_leg": 4000,
}


def _client(tmp_path, **kwargs):
    app = create_app({"a": "alice"}, tmp_path / "audit.jsonl", **kwargs)
    return TestClient(app, raise_server_exceptions=False)


def _request(**config):
    request = example_request()
    request["config"] = {**request.get("config", {}), **config}
    return request


@pytest.mark.parametrize("path", ["/v1/hedge", "/v1/frontier", "/v1/proposals"])
@pytest.mark.parametrize(("name", "value"), LOOSER.items(), ids=list(LOOSER))
def test_a_request_cannot_loosen_a_gate_by_default(tmp_path, path, name, value):
    response = _client(tmp_path).post(path, headers=ALICE, json=_request(**{name: value}))
    assert response.status_code == 422
    assert name in response.json()["error"]


@pytest.mark.parametrize(("name", "value"), TIGHTER.items(), ids=list(TIGHTER))
def test_tightening_a_gate_is_allowed_and_disclosed(tmp_path, name, value):
    response = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=_request(**{name: value}))
    assert response.status_code == 200
    body = response.json()
    assert body["manifest"]["gate_overrides"] == {name: value}
    assert any(i.startswith("GATE_SETTINGS") and name in i for i in body["issues"])


def test_a_deployment_can_raise_its_limit_from_the_environment_and_no_further(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GN_GATE_MAX_QUOTE_AGE_SECONDS", "3600")
    monkeypatch.setenv("GN_GATE_REQUIRE_QUOTE_SIZE", "false")
    client = _client(tmp_path)
    allowed = _request(max_quote_age_seconds=600.0, require_quote_size=False)
    response = client.post("/v1/hedge", headers=ALICE, json=allowed)
    assert response.status_code == 200
    assert response.json()["manifest"]["gate_overrides"] == {
        "max_quote_age_seconds": 600.0,
        "require_quote_size": False,
    }
    past = client.post("/v1/hedge", headers=ALICE, json=_request(max_quote_age_seconds=7200.0))
    assert past.status_code == 422
    assert "max_quote_age_seconds" in past.json()["error"]


def test_an_unreadable_gate_limit_stops_the_service_from_starting(tmp_path, monkeypatch):
    monkeypatch.setenv("GN_GATE_ALLOW_EXTRAPOLATED_VOL", "maybe")
    with pytest.raises(RuntimeError, match="GN_GATE_ALLOW_EXTRAPOLATED_VOL"):
        _client(tmp_path)


def test_a_default_request_has_no_gate_issue():
    problem, snapshot, _ = parse_hedge_request(example_request())
    proposal = solve_hedge(problem, snapshot)
    assert not any(i.startswith("GATE_SETTINGS") for i in proposal.issues)
    assert proposal.manifest.gate_overrides == {}


def test_a_stale_market_passed_by_a_library_override_says_so():
    problem, snapshot, _ = parse_hedge_request(example_request())
    stale = tuple(
        replace(c, quote=replace(c.quote, timestamp=snapshot.as_of - timedelta(hours=10)))
        for c in problem.candidates
    )
    blocked = solve_hedge(replace(problem, candidates=stale), snapshot)
    assert blocked.status.value == "no_usable_candidates"
    config = replace(problem.config, max_quote_age=timedelta(days=30))
    allowed = solve_hedge(replace(problem, candidates=stale, config=config), snapshot)
    assert allowed.legs
    assert any(i.startswith("GATE_SETTINGS") for i in allowed.issues)
    assert allowed.manifest.gate_overrides == {"max_quote_age_seconds": 2_592_000.0}


def test_the_audit_record_lists_the_gate_settings(tmp_path):
    problem, snapshot, _ = parse_hedge_request(_request(max_lots_per_leg=4000))
    proposal = solve_hedge(problem, snapshot, run_stability=True)
    path = tmp_path / "audit.jsonl"
    ProposalStore(AuditLog(path), clock=lambda: snapshot.as_of).submit(proposal, "alice")
    detail = json.loads(path.read_text().splitlines()[0])["detail"]
    assert detail["gate_overrides"] == {"max_lots_per_leg": 4000}


def test_the_cli_shows_a_gate_setting(tmp_path, capfd):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(_request(max_lots_per_leg=4000)))
    assert main(["hedge", "--request", str(path)]) == 0
    assert "GATE_SETTINGS" in capfd.readouterr().out


def test_every_non_default_gate_is_listed_whichever_way_it_moved():
    config = HedgeConfig(require_quote_size=False, regret_threshold=0.05)
    assert gate_overrides(config) == {"regret_threshold": 0.05, "require_quote_size": False}


def test_the_delta_only_baseline_keeps_the_lot_limit():
    future = LinearInstrument("XF", "XYZ", LinearKind.FUTURE, multiplier=50.0, expiry=expiry_in(30))
    candidate = HedgeCandidate(future, Quote(100.0, 100.1, 10_000, 10_000, AS_OF))
    book = (Position(euro("call", 100, 30), -100),)
    target = (GreekTarget("XYZ", GreekName.DELTA, weight=1.0),)
    problem = HedgeProblem(book, (candidate,), target, (), HedgeConfig(max_lots_per_leg=2))
    delta_only = next(
        p for p in hedge_frontier(problem, flat_snapshot(), ()) if p.label == "delta only"
    )
    assert delta_only.proposal is not None
    assert abs(delta_only.proposal.legs[0].quantity) == 2
