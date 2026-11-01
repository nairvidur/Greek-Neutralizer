"""Regression tests from the fifth review round. Each failed before its fix.

Oracles: the FIX 4.4 required-field list for NewOrderSingle and its UTCTimestamp format; calendar
arithmetic for the New York date of an expiry; hand-worked dollar exposures of a share position and
hand-worked spread and volume arithmetic; and each gate's direction from the meaning of the gate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from greeks_neutralizer.cli import main
from greeks_neutralizer.domain import (
    LinearInstrument,
    LinearKind,
    OptionContract,
    OptionRight,
    Position,
    Quote,
)
from greeks_neutralizer.errors import SolverRangeError, ValidationError
from greeks_neutralizer.examples import (
    DEFAULT_AS_OF,
    example_book,
    example_market,
    example_request,
)
from greeks_neutralizer.hedging import optimizer
from greeks_neutralizer.hedging.diagnostics import PERTURBATIONS, hedge_decay_forecast
from greeks_neutralizer.hedging.optimizer import VerificationError, solve_hedge
from greeks_neutralizer.hedging.problem import (
    GreekName,
    GreekTarget,
    HedgeCandidate,
    HedgeConfig,
    HedgeLeg,
    HedgeProblem,
)
from greeks_neutralizer.pricing.black import black_greeks
from greeks_neutralizer.pricing.book import value_book
from greeks_neutralizer.risk.measures import RiskLimit, expiry_ladder, liquidity_adjusted_cost
from greeks_neutralizer.schema import (
    parse_book,
    parse_hedge_request,
    parse_market,
    proposal_to_json,
)
from greeks_neutralizer.service.app import create_app
from greeks_neutralizer.workflow import AuditLog, ProposalStore
from tests.conftest import AS_OF, euro, flat_snapshot

ALICE = {"Authorization": "Bearer a"}
BOB = {"Authorization": "Bearer b"}
STOCK = {"type": "linear", "symbol": "XYZ", "underlying": "XYZ"}
HUNDRED_SHARES = [{"instrument": STOCK, "quantity": 100}]


def _client(tmp_path, clock=lambda: DEFAULT_AS_OF, **kwargs):
    app = create_app({"a": "alice", "b": "bob"}, tmp_path / "audit.jsonl", clock=clock, **kwargs)
    return TestClient(app, raise_server_exceptions=False)


def _request(**config):
    request = example_request()
    request["config"] = {**request["config"], **config}
    return request


# --- The frontier of a CVaR-only request ----------------------------------------------------------


def test_a_cvar_only_request_has_a_frontier_with_its_baselines(tmp_path):
    request = example_request()
    request.update(targets=[], scenarios="historical", config={"cvar_weight": 1.0})
    client = _client(tmp_path)
    assert client.post("/v1/hedge", headers=ALICE, json=request).status_code == 200
    response = client.post("/v1/frontier", headers=ALICE, json=request)
    assert response.status_code == 200, response.text
    points = {point["label"]: point for point in response.json()["points"]}
    assert points["do nothing"]["n_legs"] == 0
    assert points["delta only"]["status"] in {"optimal", "within_gap"}
    assert points["delta only"]["unsolved_reason"] is None
    # No targets were asked for, so no point has a residual to report.
    assert {point["penalized_residual"] for point in points.values()} == {0.0}


# --- Solver controls are gates --------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/v1/hedge", "/v1/frontier", "/v1/proposals"])
def test_a_request_cannot_read_as_optimal_by_widening_the_gap(tmp_path, path):
    response = _client(tmp_path).post(path, headers=ALICE, json=_request(mip_gap=0.99))
    assert response.status_code == 422
    assert "mip_gap" in response.json()["error"]


@pytest.mark.parametrize(
    ("variable", "name", "limit", "over"),
    [
        ("GN_GATE_MIP_GAP", "mip_gap", 0.01, 0.02),
        ("GN_GATE_TIME_LIMIT_SECONDS", "time_limit_seconds", 60.0, 61.0),
        ("GN_GATE_MIP_NODE_LIMIT", "mip_node_limit", 500, 501),
    ],
)
def test_a_deployment_sets_each_solver_limit(tmp_path, monkeypatch, variable, name, limit, over):
    monkeypatch.setenv(variable, str(limit))
    client = _client(tmp_path)
    allowed = client.post("/v1/hedge", headers=ALICE, json=_request(**{name: limit}))
    assert allowed.status_code == 200
    assert allowed.json()["manifest"]["gate_overrides"] == {}
    refused = client.post("/v1/hedge", headers=ALICE, json=_request(**{name: over}))
    assert refused.status_code == 422
    assert name in refused.json()["error"]


def test_a_tighter_solver_setting_reaches_the_audit_record(tmp_path):
    client = _client(tmp_path)
    response = client.post("/v1/proposals", headers=ALICE, json=_request(time_limit_seconds=10.0))
    assert response.status_code == 201
    first = (tmp_path / "audit.jsonl").read_text().splitlines()[0]
    assert json.loads(first)["detail"]["gate_overrides"] == {"time_limit_seconds": 10.0}


# --- An omitted gate takes the deployment's limit -------------------------------------------------


def test_tightening_a_limit_does_not_break_the_services_own_sample(tmp_path, monkeypatch):
    monkeypatch.setenv("GN_GATE_MAX_LOTS_PER_LEG", "3")
    client = _client(tmp_path)
    sample = client.get("/v1/example", headers=ALICE).json()
    response = client.post("/v1/hedge", headers=ALICE, json=sample)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["manifest"]["gate_overrides"] == {}
    assert not any(issue.startswith("GATE_SETTINGS") for issue in body["issues"])
    options = [leg for leg in body["legs"] if leg["instrument"]["type"] == "option"]
    assert options
    assert all(abs(leg["quantity"]) <= 3 for leg in options)


def test_the_frontier_has_no_false_gate_issue_under_a_tighter_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("GN_GATE_MAX_LOTS_PER_LEG", "1000")
    monkeypatch.setenv("GN_GATE_MIP_GAP", "0.001")
    client = _client(tmp_path)
    response = client.post("/v1/frontier", headers=ALICE, json=example_request())
    assert response.status_code == 200, response.text
    issues = [issue for point in response.json()["points"] for issue in point["issues"]]
    assert not any(issue.startswith("GATE_SETTINGS") for issue in issues)


def test_a_request_is_compared_with_the_limit_in_both_directions(tmp_path, monkeypatch):
    monkeypatch.setenv("GN_GATE_MAX_LOTS_PER_LEG", "1000")
    client = _client(tmp_path)
    tighter = client.post("/v1/hedge", headers=ALICE, json=_request(max_lots_per_leg=500))
    assert tighter.status_code == 200
    assert tighter.json()["manifest"]["gate_overrides"] == {"max_lots_per_leg": 500}
    assert any(issue.startswith("GATE_SETTINGS") for issue in tighter.json()["issues"])
    # The library default, but looser than this deployment allows.
    looser = client.post("/v1/hedge", headers=ALICE, json=_request(max_lots_per_leg=5000))
    assert looser.status_code == 422


def test_with_no_deployment_the_comparison_is_with_the_library_defaults():
    problem, snapshot, _ = parse_hedge_request(_request(mip_gap=0.001))
    assert solve_hedge(problem, snapshot).manifest.gate_overrides == {"mip_gap": 0.001}


# --- FIX 4.4 NewOrderSingle -----------------------------------------------------------------------


def test_a_fix_ticket_has_every_required_field_and_the_audited_hash_covers_it(tmp_path):
    problem, snapshot, _ = parse_hedge_request(example_request())
    proposal = solve_hedge(problem, snapshot, run_stability=True)
    assert proposal.legs
    # 10:00:30 at UTC-05:00 is 15:00:30 UTC, and the ticket must not carry the clock's own zone.
    clock = DEFAULT_AS_OF.astimezone(timezone(timedelta(hours=-5))) + timedelta(seconds=30)
    path = tmp_path / "audit.jsonl"
    store = ProposalStore(AuditLog(path), clock=lambda: clock)
    record = store.submit(proposal, "alice")
    store.approve(record.id, "bob")
    text = store.export_tickets(record.id, "bob", "fix")

    # FIX 4.4 NewOrderSingle: ClOrdID, Side, TransactTime, OrdType and OrderQty, with Symbol from
    # the Instrument block, and Price because the order type is limit.
    required = {"35", "11", "55", "54", "60", "38", "40", "44"}
    for message in text.strip().splitlines():
        fields = dict(part.split("=") for part in message.rstrip("|").split("|"))
        assert required <= set(fields)
        assert fields["60"] == "20260105-15:00:30"

    exported = json.loads(path.read_text().splitlines()[-1])
    assert exported["detail"]["sha256"] == hashlib.sha256(text.encode()).hexdigest()


# --- Risk reports ---------------------------------------------------------------------------------


def test_the_ladder_buckets_on_the_new_york_date():
    # 16:00 in New York on 20 March is 05:00 on 21 March at +09:00.
    tokyo = timezone(timedelta(hours=9))
    expiry = datetime(2026, 3, 21, 5, 0, tzinfo=tokyo)
    contract = OptionContract("XYZ", OptionRight.CALL, 100.0, expiry)
    ladder = expiry_ladder([Position(contract, 1)], flat_snapshot())
    assert [bucket.expiry.isoformat() for bucket in ladder.buckets] == ["2026-03-20"]


def test_the_ladder_endpoint_totals_each_date(tmp_path):
    payload = {"book": [*example_book(), *HUNDRED_SHARES], "market": example_market()}
    response = _client(tmp_path).post("/v1/ladder", headers=ALICE, json=payload)
    assert response.status_code == 200
    body = response.json()
    # Both sample options expire 30 days after 15:00 UTC on 5 January, and shares do not expire.
    assert body["count"] == 1
    assert body["buckets"][0]["expiry"] == "2026-02-04"
    assert body["buckets"][0]["n_lines"] == 2


def test_limit_breaches_are_reported_with_hand_worked_values(tmp_path):
    limits = [
        {"metric": "delta_dollars", "lower": -5000.0, "upper": 5000.0, "underlying": "XYZ"},
        {"metric": "delta_shares", "lower": 0.0, "upper": 1000.0, "underlying": "XYZ"},
        {"metric": "vega_1pt", "lower": 0.0, "upper": 0.0},
        {"metric": "delta_shares", "lower": 0.0, "upper": 0.0},
    ]
    payload = {"book": HUNDRED_SHARES, "market": example_market(), "limits": limits}
    response = _client(tmp_path).post("/v1/limits", headers=ALICE, json=payload)
    assert response.status_code == 200
    body = response.json()
    assert (body["checked"], body["breached"]) == (4, 2)
    dollars, shares = body["breaches"]
    # 100 shares at a spot of 100 is $10,000 of delta, twice a limit of $5,000.
    assert dollars["metric"] == "delta_dollars"
    assert dollars["value"] == pytest.approx(10_000.0)
    assert dollars["utilization"] == pytest.approx(2.0)
    assert not dollars["utilization_unbounded"]
    # A band of zero width has no finite utilization, and says so.
    assert shares["utilization"] is None
    assert shares["utilization_unbounded"]


@pytest.mark.parametrize(
    ("limit", "fragment"),
    [
        ({"metric": "delta", "lower": -1.0, "upper": 1.0}, "unknown risk metric"),
        ({"metric": "delta_shares", "lower": -1.0, "upper": 1.0, "underlying": "NOPE"}, "NOPE"),
        ({"metric": "delta_shares", "lower": -1.0, "upper": 1.0, "extra": 1}, "extra"),
        ({"metric": "delta_shares", "lower": "low", "upper": 1.0}, "lower"),
    ],
)
def test_a_limit_that_cannot_be_checked_is_refused_even_on_an_empty_book(tmp_path, limit, fragment):
    payload = {"book": [], "market": example_market(), "limits": [limit]}
    response = _client(tmp_path).post("/v1/limits", headers=ALICE, json=payload)
    assert response.status_code == 422
    assert fragment in response.json()["error"]


def test_a_not_a_number_limit_cannot_be_built():
    with pytest.raises(ValidationError, match="lower"):
        RiskLimit("delta_shares", float("nan"), 1.0)


def test_liquidity_reports_hand_worked_cost_time_and_totals(tmp_path):
    option = example_book()[0]["instrument"]
    lines = [
        {"instrument": option, "quantity": -500, "half_spread": 0.05, "adv_contracts": 1000.0},
        {
            "instrument": STOCK,
            "quantity": 300,
            "half_spread": 0.01,
            "adv_contracts": 1000.0,
            "participation": 0.5,
        },
    ]
    response = _client(tmp_path).post("/v1/liquidity", headers=ALICE, json={"lines": lines})
    assert response.status_code == 200
    body = response.json()
    # 500 contracts x 100 x 0.05 = 2,500, over 500 / (1,000 x 10%) = 5 days.
    assert body["lines"][0]["half_spread_cost"] == pytest.approx(2_500.0)
    assert body["lines"][0]["days_to_unwind"] == pytest.approx(5.0)
    # 300 shares x 0.01 = 3, over 300 / (1,000 x 50%) = 0.6 days.
    assert body["total_half_spread_cost"] == pytest.approx(2_503.0)
    assert body["longest_days_to_unwind"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("half_spread", "adv"),
    [(float("nan"), 1000.0), (0.05, float("inf")), (-0.01, 1000.0), (0.05, 0.0)],
)
def test_unusable_liquidity_inputs_are_refused(half_spread, adv):
    with pytest.raises(ValidationError, match=r"half_spread|adv"):
        liquidity_adjusted_cost(
            Position(OptionContract("XYZ", OptionRight.CALL, 100.0, AS_OF), 1), half_spread, adv
        )


def test_a_liquidity_line_with_an_unknown_field_is_refused(tmp_path):
    line = {"instrument": STOCK, "quantity": 1, "half_spread": 0.01, "adv_contracts": 1, "x": 1}
    response = _client(tmp_path).post("/v1/liquidity", headers=ALICE, json={"lines": [line]})
    assert response.status_code == 422
    assert "unknown field(s) ['x']" in response.json()["error"]


def test_decay_shows_an_option_expiring_and_shares_staying_put(tmp_path):
    expiring = {**example_book()[0]["instrument"]}
    expiring["expiry"] = (DEFAULT_AS_OF + timedelta(days=5)).isoformat()
    payload = {
        "book": [{"instrument": expiring, "quantity": 1}, *HUNDRED_SHARES],
        "market": example_market(),
        "underlying": "XYZ",
        "days": [1, 10],
    }
    response = _client(tmp_path).post("/v1/decay", headers=ALICE, json=payload)
    assert response.status_code == 200, response.text
    before, after = response.json()["steps"]
    assert (before["days_forward"], after["days_forward"]) == (1.0, 10.0)
    assert before["exposure"]["vega_1pt"] > 0.0
    # Past its expiry the option has no vega, and what is left is the 100 shares at a spot of 100.
    assert after["exposure"]["vega_1pt"] == 0.0
    assert after["exposure"]["delta_dollars"] == pytest.approx(10_000.0)


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"underlying": "NOPE"}, "NOPE"),
        ({"underlying": 5}, "underlying"),
        ({"days": [-1.0]}, "days_forward"),
        ({"days": ["soon"]}, "days"),
        ({"extra": 1}, "extra"),
    ],
)
def test_a_decay_request_that_cannot_be_answered_is_refused(tmp_path, change, fragment):
    payload = {"book": HUNDRED_SHARES, "market": example_market(), "underlying": "XYZ", **change}
    response = _client(tmp_path).post("/v1/decay", headers=ALICE, json=payload)
    assert response.status_code == 422
    assert fragment in response.json()["error"]


@pytest.mark.parametrize("path", ["/v1/ladder", "/v1/limits", "/v1/liquidity", "/v1/decay"])
def test_risk_reports_require_a_token(tmp_path, path):
    assert _client(tmp_path).post(path, json={}).status_code == 401


# --- The same reports from the CLI, on the commands the README gives ------------------------------


@pytest.fixture
def files(tmp_path, capfd):
    def example(kind: str) -> str:
        assert main(["example", "--kind", kind]) == 0
        path = tmp_path / f"{kind}.json"
        path.write_text(capfd.readouterr().out)
        return str(path)

    return {kind: example(kind) for kind in ("book", "market", "limits", "liquidity")}


def test_gn_ladder(files, capfd):
    assert main(["ladder", "--book", files["book"], "--market", files["market"]]) == 0
    out = capfd.readouterr().out
    assert "2026-02-04" in out
    assert "--full" in out


def test_gn_limits_reports_a_definite_answer(files, tmp_path, capfd):
    wide = tmp_path / "wide.json"
    wide.write_text(json.dumps([{"metric": "delta_dollars", "lower": -1e12, "upper": 1e12}]))
    args = ["limits", "--book", files["book"], "--market", files["market"], "--limits"]
    assert main([*args, str(wide)]) == 0
    assert "0 of 1 limits breached" in capfd.readouterr().out
    tight = tmp_path / "tight.json"
    tight.write_text(json.dumps([{"metric": "delta_dollars", "lower": 1e9, "upper": 1e12}]))
    assert main([*args, str(tight)]) == 0
    assert "breached: 1" in capfd.readouterr().out
    assert main([*args, files["limits"]]) == 0


def test_gn_limits_fails_on_a_bad_limit(files, tmp_path, capfd):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"metric": "delta", "lower": -1.0, "upper": 1.0}]))
    args = ["limits", "--book", files["book"], "--market", files["market"], "--limits", str(bad)]
    assert main(args) == 1
    assert "unknown risk metric" in capfd.readouterr().out


def test_gn_liquidity(files, capfd):
    assert main(["liquidity", "--lines", files["liquidity"]]) == 0
    # 2 lines of 10 contracts x 100 x 0.05 = 100.
    assert "total_half_spread_cost: 100" in capfd.readouterr().out


def test_gn_decay(files, capfd):
    args = ["decay", "--book", files["book"], "--market", files["market"], "--underlying", "XYZ"]
    assert main([*args, "--days", "1", "40"]) == 0
    out = capfd.readouterr().out
    assert "underlying: XYZ" in out
    rows = {
        float(row[0]): [float(cell) for cell in row[1:]]
        for row in csv.reader(line.strip() for line in out.splitlines() if line.startswith("  "))
    }
    # days, delta_dollars, gamma_1pct, vega_1pt, theta_1d, of a short 30-day book. It is short vega
    # at +1 day, and past expiry by +40, when it has no exposure left at all.
    assert rows[1.0][2] < -100.0
    assert rows[40.0] == [0.0, 0.0, 0.0, 0.0]
    assert main([*args, "--full"]) == 0
    assert "vanna_1pt" in capfd.readouterr().out
    assert (
        main(["decay", "--book", files["book"], "--market", files["market"], "--underlying", "X"])
        == 1
    )
    assert "no market data" in capfd.readouterr().out


# --- Listing proposals ----------------------------------------------------------------------------


def _store_with(tmp_path, count, clock):
    client = _client(tmp_path, clock=clock)
    problem, snapshot, _ = parse_hedge_request(example_request())
    proposal = solve_hedge(problem, snapshot, run_stability=True)
    ids = [client.app.state.store.submit(proposal, "alice").id for _ in range(count)]
    return client, ids, proposal


def test_the_list_is_summaries_with_a_total_and_a_default_limit(tmp_path):
    client, _, proposal = _store_with(tmp_path, 51, lambda: DEFAULT_AS_OF)
    body = client.get("/v1/proposals", headers=BOB).json()
    assert (body["total"], body["count"], len(body["proposals"])) == (51, 50, 50)
    assert body["proposals"][0] == {
        "id": body["proposals"][0]["id"],
        "state": "proposed",
        "status": proposal.status.value,
        "issue_count": len(proposal.issues),
        "created_by": "alice",
        "created_at": DEFAULT_AS_OF.isoformat(),
        "leg_count": len(proposal.legs),
        "transaction_cost": pytest.approx(proposal.transaction_cost),
    }


def test_the_list_is_newest_first_under_a_limit_and_open_returns_the_whole_proposal(tmp_path):
    times = iter(DEFAULT_AS_OF + timedelta(seconds=n) for n in range(10_000))
    client, ids, proposal = _store_with(tmp_path, 3, lambda: next(times))
    body = client.get("/v1/proposals?limit=2", headers=BOB).json()
    assert (body["total"], body["count"]) == (3, 2)
    assert [row["id"] for row in body["proposals"]] == [ids[2], ids[1]]
    opened = client.get(f"/v1/proposals/{ids[2]}", headers=BOB).json()
    assert len(opened["proposal"]["legs"]) == len(proposal.legs)


@pytest.mark.parametrize("limit", [0, -1, 501, "many"])
def test_an_unusable_list_limit_is_refused(tmp_path, limit):
    response = _client(tmp_path).get(f"/v1/proposals?limit={limit}", headers=ALICE)
    assert response.status_code == 422


def test_an_empty_list_is_definite(tmp_path):
    body = _client(tmp_path).get("/v1/proposals", headers=ALICE).json()
    assert body == {"total": 0, "count": 0, "proposals": []}


# --- Reports carry the book's valuation issues ----------------------------------------------------


def _expired_book():
    option = {**example_book()[0]["instrument"]}
    option["expiry"] = (DEFAULT_AS_OF - timedelta(days=1)).isoformat()
    return [{"instrument": option, "quantity": -1}]


@pytest.mark.parametrize(
    ("path", "extra"),
    [
        ("/v1/ladder", {}),
        ("/v1/limits", {"limits": [{"metric": "delta_shares", "lower": -1.0, "upper": 1.0}]}),
        ("/v1/decay", {"underlying": "XYZ", "days": [1]}),
        ("/v1/decay", {"underlying": "XYZ", "days": []}),
        ("/v1/stress", {}),
        ("/v1/grid", {}),
    ],
)
def test_an_expired_option_is_named_on_every_report(tmp_path, path, extra):
    payload = {"book": _expired_book(), "market": example_market(), **extra}
    response = _client(tmp_path).post(path, headers=ALICE, json=payload)
    assert response.status_code == 200, response.text
    assert any(issue.startswith("EXPIRED") for issue in response.json()["issues"])


def test_a_clean_book_has_no_issues_on_its_reports(tmp_path):
    payload = {"book": example_book(), "market": example_market()}
    client = _client(tmp_path)
    assert client.post("/v1/ladder", headers=ALICE, json=payload).json()["issues"] == []
    assert client.post("/v1/stress", headers=ALICE, json=payload).json()["issues"] == []


def test_decay_says_at_which_step_an_option_expires(tmp_path):
    expiring = {**example_book()[0]["instrument"]}
    expiring["expiry"] = (DEFAULT_AS_OF + timedelta(days=5)).isoformat()
    payload = {
        "book": [{"instrument": expiring, "quantity": 1}],
        "market": example_market(),
        "underlying": "XYZ",
        "days": [1, 10, 20],
    }
    issues = _client(tmp_path).post("/v1/decay", headers=ALICE, json=payload).json()["issues"]
    assert len(issues) == 1
    assert issues[0].startswith("EXPIRED")
    assert issues[0].endswith("(at +10d)")


@pytest.mark.parametrize(
    "command",
    [
        ["ladder"],
        ["stress"],
        ["decay", "--underlying", "XYZ", "--days", "1"],
        ["limits", "--limits", "LIMITS"],
    ],
    ids=["ladder", "stress", "decay", "limits"],
)
def test_an_expired_option_is_named_by_every_cli_report(tmp_path, capfd, command):
    book = tmp_path / "book.json"
    book.write_text(json.dumps(_expired_book()))
    market = tmp_path / "market.json"
    market.write_text(json.dumps(example_market()))
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps([{"metric": "delta_shares", "lower": -1.0, "upper": 1.0}]))
    args = [str(limits) if part == "LIMITS" else part for part in command]
    assert main([args[0], "--book", str(book), "--market", str(market), *args[1:]]) == 0
    assert "EXPIRED" in capfd.readouterr().out


# --- Days forward has a stated maximum ------------------------------------------------------------


@pytest.mark.parametrize("days", [36_526.0, 1e9, 1e12, 1e308])
def test_a_huge_horizon_is_a_422_on_decay(tmp_path, days):
    payload = {
        "book": HUNDRED_SHARES,
        "market": example_market(),
        "underlying": "XYZ",
        "days": [days],
    }
    response = _client(tmp_path).post("/v1/decay", headers=ALICE, json=payload)
    assert response.status_code == 422
    assert "days_forward" in response.json()["error"]


@pytest.mark.parametrize("path", ["/v1/hedge", "/v1/frontier", "/v1/proposals"])
def test_a_huge_horizon_is_a_422_in_a_request_scenario(tmp_path, path):
    request = example_request()
    request["scenarios"] = [{"name": "far", "days_forward": 1e9}]
    response = _client(tmp_path).post(path, headers=ALICE, json=request)
    assert response.status_code == 422
    assert "days_forward" in response.json()["error"]


@pytest.mark.parametrize("days", ["1e7", "1e12"])
def test_a_huge_horizon_is_a_structured_cli_error(tmp_path, capfd, days):
    book = tmp_path / "book.json"
    book.write_text(json.dumps(HUNDRED_SHARES))
    market = tmp_path / "market.json"
    market.write_text(json.dumps(example_market()))
    args = ["decay", "--book", str(book), "--market", str(market), "--underlying", "XYZ"]
    assert main([*args, "--days", days]) == 1
    out = capfd.readouterr().out
    assert out.startswith("error:")
    assert "days_forward" in out


@pytest.mark.parametrize(
    ("path", "extra"), [("/v1/stress", {}), ("/v1/decay", {"underlying": "XYZ"})]
)
def test_a_market_date_with_no_room_for_a_horizon_is_a_422(tmp_path, path, extra):
    market = {**example_market(), "as_of": "9999-12-30T15:00:00+00:00"}
    payload = {"book": HUNDRED_SHARES, "market": market, **extra}
    response = _client(tmp_path).post(path, headers=ALICE, json=payload)
    assert response.status_code == 422
    assert "as_of must be from 1900 to 2300" in response.json()["error"]


# These overflow on conversion to UTC or New York, which every report does.
BOUNDARY_DATES = ["9999-12-31T23:59:59-12:00", "0001-01-01T00:00:00+14:00", "1899-12-31T00:00:00Z"]


@pytest.mark.parametrize("path", ["/v1/value", "/v1/ladder", "/v1/stress", "/v1/grid"])
@pytest.mark.parametrize("expiry", BOUNDARY_DATES)
def test_an_expiry_past_either_end_of_time_is_a_422(tmp_path, path, expiry):
    option = {**example_book()[0]["instrument"], "expiry": expiry}
    payload = {"book": [{"instrument": option, "quantity": 1}], "market": example_market()}
    response = _client(tmp_path).post(path, headers=ALICE, json=payload)
    assert response.status_code == 422
    assert "expiry must be from 1900 to 2300" in response.json()["error"]


def test_the_date_window_leaves_room_for_the_assignment_window(tmp_path):
    # expiry_risks adds 7 days to as_of, which ends past the window here, and its stated rule still
    # applies: a short in-the-money American call, with a dividend above its time value before the
    # ex-date, is flagged. Strike 50 on a spot of 100 has next to no time value against a 2.00 cash
    # dividend. A window too narrow for that arithmetic would refuse this, or fail with a 500.
    last = "2300-01-01T00:00:00+00:00"
    call = {**example_book()[0]["instrument"], "style": "american", "strike": 50.0, "expiry": last}
    market = example_market()
    market["as_of"] = "2299-12-31T15:00:00+00:00"
    market["underlyings"]["XYZ"]["dividends"] = [{"ex_date": last, "amount": 2.0}]
    payload = {"book": [{"instrument": call, "quantity": -1}], "market": market}
    response = _client(tmp_path).post("/v1/value", headers=ALICE, json=payload)
    assert response.status_code == 200, response.text
    assert [risk["kind"] for risk in response.json()["expiry_risks"]] == ["assignment"]


@pytest.mark.parametrize("command", ["value", "ladder"])
def test_an_expiry_past_the_end_of_time_is_a_structured_cli_error(tmp_path, capfd, command):
    option = {**example_book()[0]["instrument"], "expiry": BOUNDARY_DATES[0]}
    book = tmp_path / "book.json"
    book.write_text(json.dumps([{"instrument": option, "quantity": 1}]))
    market = tmp_path / "market.json"
    market.write_text(json.dumps(example_market()))
    assert main([command, "--book", str(book), "--market", str(market)]) == 1
    out = capfd.readouterr().out
    assert out.startswith("error:")
    assert "expiry must be from 1900 to 2300" in out


def test_the_last_day_allowed_still_values_every_horizon(tmp_path):
    # The far end of the window, and the longest horizon on top of it, must both be real dates.
    option = {**example_book()[0]["instrument"], "expiry": "2300-01-01T00:00:00+00:00"}
    market = {**example_market(), "as_of": "2300-01-01T00:00:00+00:00"}
    payload = {
        "book": [{"instrument": option, "quantity": 1}],
        "market": market,
        "underlying": "XYZ",
        "days": [36_525],
    }
    assert _client(tmp_path).post("/v1/decay", headers=ALICE, json=payload).status_code == 200


def _enormous_lot_request():
    # A lot size past what a float can hold reaches arithmetic the parser does not do.
    request = example_request()
    request["candidates"][1]["lot_size"] = 10**400
    return request


def test_arithmetic_out_of_range_on_inputs_is_a_422(tmp_path):
    response = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=_enormous_lot_request())
    assert response.status_code == 422
    assert "out of range" in response.json()["error"]


def test_arithmetic_out_of_range_on_inputs_is_a_structured_cli_error(tmp_path, capfd):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(_enormous_lot_request()))
    assert main(["hedge", "--request", str(path)]) == 1
    out = capfd.readouterr().out
    assert out.startswith("error:")
    assert "out of range" in out


# --- Every frontier point names the book's issues -------------------------------------------------


def _options_only_frontier_request(**target):
    request = example_request()
    option = {**example_book()[0]["instrument"]}
    option["expiry"] = (DEFAULT_AS_OF - timedelta(days=1)).isoformat()
    request["book"] = [*example_book(), {"instrument": option, "quantity": -1}]
    request["candidates"] = [
        c for c in request["candidates"] if c["instrument"]["type"] == "option"
    ]
    request["targets"] = [{"underlying": "XYZ", "greek": "delta_dollars", **target}]
    return request


@pytest.mark.parametrize(
    "target",
    [{"lower": 1e12, "upper": 2e12}, {}],
    ids=["every-solve-infeasible", "every-solve-solved"],
)
def test_every_frontier_point_names_an_expired_option(tmp_path, target):
    request = _options_only_frontier_request(**target)
    response = _client(tmp_path).post("/v1/frontier", headers=ALICE, json=request)
    assert response.status_code == 200, response.text
    points = response.json()["points"]
    # 7 risk aversions, the do-nothing baseline, and no delta-only point with no linear candidate.
    assert len(points) == 8
    assert points[0]["label"] == "do nothing"
    assert "delta only" not in [point["label"] for point in points]
    for point in points:
        assert any(issue.startswith("EXPIRED") for issue in point["issues"]), point["label"]
    solves = points[1:]
    if target:
        assert all(p["status"] == "infeasible" for p in solves)
        assert all(p["unsolved_reason"] for p in solves)
    else:
        assert all(p["status"] != "infeasible" and p["unsolved_reason"] is None for p in solves)


def test_the_cli_frontier_prints_the_issues_themselves(tmp_path, capfd):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(_options_only_frontier_request(lower=1e12, upper=2e12)))
    assert main(["frontier", "--request", str(path)]) == 0
    out = capfd.readouterr().out
    assert out.count("EXPIRED") == 1


def test_the_longest_horizon_allowed_still_values_a_book():
    # 100 years, and a stated maximum rather than the point where dates happen to overflow.
    book = [Position(LinearInstrument("XYZ", "XYZ"), 100)]
    forecast = hedge_decay_forecast(book, flat_snapshot(), "XYZ", (36_525.0,))
    assert forecast.steps[0].exposure.delta_shares == 100.0


# --- Limit utilization ----------------------------------------------------------------------------


def _breach(tmp_path, quantity, **bounds):
    payload = {
        "book": [{"instrument": STOCK, "quantity": quantity}],
        "market": example_market(),
        "limits": [{"metric": "delta_shares", **bounds}],
    }
    response = _client(tmp_path).post("/v1/limits", headers=ALICE, json=payload)
    assert response.status_code == 200, response.text
    return response.json()["breaches"]


def test_utilization_is_distance_from_the_centre_over_the_half_width(tmp_path):
    # Centre 550, half-width 450, and 1 share is 549 from the centre: 549 / 450 = 1.22.
    assert _breach(tmp_path, 1, lower=100.0, upper=1000.0)[0]["utilization"] == pytest.approx(1.22)
    # Centre 500, half-width 500, and -100 shares is 600 from the centre: 600 / 500 = 1.2.
    assert _breach(tmp_path, -100, lower=0.0, upper=1000.0)[0]["utilization"] == pytest.approx(1.2)


def test_a_worse_breach_of_a_floor_has_a_higher_utilization(tmp_path):
    near = _breach(tmp_path, 90, lower=100.0, upper=1000.0)[0]["utilization"]
    far = _breach(tmp_path, 1, lower=100.0, upper=1000.0)[0]["utilization"]
    assert 1.0 < near < far


def test_utilization_does_not_overflow_on_very_large_bounds():
    limit = RiskLimit("delta_shares", 1e308, 1.7e308)
    assert limit.utilization(1.0) == pytest.approx(1.35 / 0.35)
    assert RiskLimit("delta_shares", -1.7e308, 1.7e308).utilization(0.0) == 0.0


@pytest.mark.parametrize(
    ("quantity", "bounds"),
    [
        (50, {"lower": 100.0}),
        (150, {"upper": 100.0}),
        (150, {"lower": 100.0, "upper": 100.0}),
    ],
    ids=["floor-only", "cap-only", "zero-width"],
)
def test_no_ratio_exists_for_an_open_or_zero_width_band(tmp_path, quantity, bounds):
    breach = _breach(tmp_path, quantity, **bounds)[0]
    assert breach["utilization"] is None
    assert breach["utilization_unbounded"] is True
    assert breach["value"] == pytest.approx(float(quantity))


def test_a_one_sided_limit_that_holds_is_not_a_breach(tmp_path):
    assert _breach(tmp_path, 150, lower=100.0) == []
    assert _breach(tmp_path, 50, upper=100.0) == []


@pytest.mark.parametrize("raw", ["Infinity", "-Infinity", "NaN"])
def test_a_bound_that_is_not_finite_is_refused(tmp_path, raw):
    limit = f'{{"metric": "delta_shares", "lower": {raw}}}'
    body = json.dumps({"book": HUNDRED_SHARES, "market": example_market(), "limits": []})
    response = _client(tmp_path).post(
        "/v1/limits",
        headers={**ALICE, "Content-Type": "application/json"},
        content=body.replace('"limits": []', f'"limits": [{limit}]'),
    )
    assert response.status_code == 422
    assert "lower" in response.json()["error"]


def test_a_limit_with_no_bound_is_refused(tmp_path):
    payload = {
        "book": HUNDRED_SHARES,
        "market": example_market(),
        "limits": [{"metric": "delta_shares"}],
    }
    response = _client(tmp_path).post("/v1/limits", headers=ALICE, json=payload)
    assert response.status_code == 422
    assert "bound" in response.json()["error"]


# --- Dated futures in the ladder ------------------------------------------------------------------


def _future(days):
    return {
        "type": "linear",
        "symbol": "XYF",
        "underlying": "XYZ",
        "kind": "future",
        "multiplier": 50.0,
        "expiry": (DEFAULT_AS_OF + timedelta(days=days)).isoformat(),
    }


def test_a_book_of_futures_has_a_ladder(tmp_path):
    payload = {"book": [{"instrument": _future(20), "quantity": 5}], "market": example_market()}
    body = _client(tmp_path).post("/v1/ladder", headers=ALICE, json=payload).json()
    assert body["count"] == 1
    assert body["buckets"][0]["expiry"] == "2026-01-25"
    # dF/dS is exp(rT) with no dividends: 5 x 50 x 100 x exp(0.05 x 20/365).
    expected = 5 * 50 * 100.0 * math.exp(0.05 * 20 / 365)
    exposure = body["buckets"][0]["exposure"]
    assert exposure["delta_dollars"] == pytest.approx(expected, rel=1e-6)


def test_a_date_shows_options_and_futures_net(tmp_path):
    futures_only = {"instrument": _future(30), "quantity": 2}
    payload = {"book": [*example_book(), futures_only], "market": example_market()}
    client = _client(tmp_path)
    body = client.post("/v1/ladder", headers=ALICE, json=payload).json()
    assert body["count"] == 1
    assert body["buckets"][0]["n_lines"] == 3
    # The two lines of options plus the future are the whole book, valued independently.
    whole = client.post("/v1/value", headers=ALICE, json=payload).json()
    expected = whole["by_underlying"]["XYZ"]["delta_dollars"]
    assert body["buckets"][0]["exposure"]["delta_dollars"] == pytest.approx(expected)


def test_a_future_is_bucketed_on_its_new_york_date():
    # 16:00 in New York on 20 March is 05:00 on 21 March at +09:00.
    expiry = datetime(2026, 3, 21, 5, 0, tzinfo=timezone(timedelta(hours=9)))
    future = LinearInstrument("XYF", "XYZ", LinearKind.FUTURE, multiplier=50.0, expiry=expiry)
    ladder = expiry_ladder([Position(future, 1)], flat_snapshot())
    assert [bucket.expiry.isoformat() for bucket in ladder.buckets] == ["2026-03-20"]


def test_shares_alone_have_no_ladder(tmp_path, capfd):
    payload = {"book": HUNDRED_SHARES, "market": example_market()}
    body = _client(tmp_path).post("/v1/ladder", headers=ALICE, json=payload).json()
    assert (body["count"], body["buckets"]) == (0, [])
    book = tmp_path / "book.json"
    book.write_text(json.dumps(HUNDRED_SHARES))
    market = tmp_path / "market.json"
    market.write_text(json.dumps(example_market()))
    assert main(["ladder", "--book", str(book), "--market", str(market)]) == 0
    assert "0 dated lines" in capfd.readouterr().out


# --- Market inputs have plausibility bounds, as rates and vols do ---------------------------------


def _stock_only_request(**market_fields):
    request = example_request()
    request["candidates"] = request["candidates"][:1]
    request["targets"] = [{"underlying": "XYZ", "greek": "delta_dollars"}]
    request["scenarios"] = []
    request["run_stability"] = False
    request["market"]["underlyings"]["XYZ"].update(market_fields)
    return request


def test_a_real_carry_yield_still_hedges(tmp_path):
    # 2% a year: no dividend mistake, and a short strangle's delta is hedged by buying shares.
    request = _stock_only_request(carry_yield=0.02)
    body = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=request).json()
    assert [leg["side"] for leg in body["legs"]] == ["BUY"]


@pytest.mark.parametrize("carry", [2.0, -2.0, -1e6, 1.0000001])
@pytest.mark.parametrize("path", ["/v1/hedge", "/v1/frontier", "/v1/value", "/v1/stress"])
def test_a_carry_yield_in_percent_is_refused(tmp_path, path, carry):
    # 2.0 meant as 2% once flipped this hedge to selling shares, with verification passing.
    request = _stock_only_request(carry_yield=carry)
    if path in {"/v1/value", "/v1/stress"}:
        request = {"book": request["book"], "market": request["market"]}
    response = _client(tmp_path).post(path, headers=ALICE, json=request)
    assert response.status_code == 422
    error = response.json()["error"]
    assert "carry_yield" in error
    assert "rates are decimals" in error


def test_the_cli_refuses_a_carry_yield_in_percent(tmp_path, capfd):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(_stock_only_request(carry_yield=2.0)))
    assert main(["hedge", "--request", str(path)]) == 1
    assert "rates are decimals" in capfd.readouterr().out


def test_a_full_carry_yield_is_at_the_bound_and_allowed(tmp_path):
    payload = _stock_only_request(carry_yield=1.0)
    assert _client(tmp_path).post("/v1/hedge", headers=ALICE, json=payload).status_code == 200


def _value_at_spot(tmp_path, spot):
    # A strike at spot, so that the only thing out of the ordinary is the size of the numbers.
    option = {**example_book()[0]["instrument"], "strike": spot}
    market = example_market()
    market["underlyings"]["XYZ"]["spot"] = spot
    payload = {"book": [{"instrument": option, "quantity": 1}], "market": market}
    return _client(tmp_path).post("/v1/value", headers=ALICE, json=payload)


def test_spot_is_limited_at_the_documented_1e9(tmp_path):
    assert _value_at_spot(tmp_path, 1e9).status_code == 200
    # This returned 200 at 1e300, with a book value of about -1e303 and no issue.
    for spot in (1.0000001e9, 1e10, 1e300):
        response = _value_at_spot(tmp_path, spot)
        assert response.status_code == 422
        assert "spot" in response.json()["error"]


def test_a_scenario_taking_spot_past_the_limit_says_which_and_what_was_entered(tmp_path):
    payload = {"book": HUNDRED_SHARES, "market": example_market()}
    payload["market"]["underlyings"]["XYZ"]["spot"] = 1e9
    # The set's own stresses include a melt-up, which goes past the limit from exactly 1e9.
    response = _client(tmp_path).post("/v1/stress", headers=ALICE, json=payload)
    assert response.status_code == 422
    error = response.json()["error"]
    assert "spot 1e+09 reaches" in error
    assert "under" in error


# --- A frontier stands wherever the hedge stands --------------------------------------------------


def _request_with_a_candidate_off_the_market(**changes):
    request = example_request()
    off_market = {
        "instrument": {"type": "linear", "symbol": "QQQ", "underlying": "QQQ"},
        "quote": request["candidates"][0]["quote"],
    }
    request["candidates"].append(off_market)
    request["run_stability"] = False
    request.update(changes)
    return request


def test_a_candidate_with_no_market_is_excluded_from_hedge_and_frontier_alike(tmp_path):
    request = _request_with_a_candidate_off_the_market()
    client = _client(tmp_path)
    hedge = client.post("/v1/hedge", headers=ALICE, json=request)
    assert hedge.status_code == 200, hedge.text
    assert [e["reason"] for e in hedge.json()["excluded"]] == ["no market data for its underlying"]
    frontier = client.post("/v1/frontier", headers=ALICE, json=request)
    assert frontier.status_code == 200, frontier.text
    points = {point["label"]: point for point in frontier.json()["points"]}
    # The stock on XYZ still makes a delta-only baseline, and only it is traded there.
    assert points["delta only"]["n_legs"] == 1
    assert points["delta only"]["unsolved_reason"] is None


def test_no_delta_only_baseline_when_no_linear_candidate_has_a_market(tmp_path):
    request = _request_with_a_candidate_off_the_market()
    request["candidates"] = request["candidates"][1:]
    response = _client(tmp_path).post("/v1/frontier", headers=ALICE, json=request)
    assert response.status_code == 200, response.text
    assert "delta only" not in [point["label"] for point in response.json()["points"]]


def test_the_cli_frontier_takes_a_candidate_with_no_market(tmp_path, capfd):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(_request_with_a_candidate_off_the_market()))
    assert main(["frontier", "--request", str(path)]) == 0
    assert '"delta only"' in capfd.readouterr().out


# --- Inputs too large for the solver are named, not reported as solver failures -------------------


COST_INPUTS = {
    "risk_aversion": lambda r: r["config"].update(risk_aversion=1e308),
    "risk_aversion_past_what_highs_accepts": lambda r: r["config"].update(risk_aversion=1e20),
    "cvar_weight": lambda r: r["config"].update(cvar_weight=1e300),
    "ticket_cost": lambda r: r["config"].update(ticket_cost=1e300),
    "target_weight": lambda r: r["targets"][0].update(weight=1e308),
    "fee": lambda r: r["candidates"][1].update(fee_per_unit=1e300),
    "impact": lambda r: r["candidates"][1].update(impact_per_unit=1e300),
    "lot_size": lambda r: r["candidates"][1].update(lot_size=10**300),
}


@pytest.mark.parametrize("change", COST_INPUTS.values(), ids=COST_INPUTS)
def test_a_cost_past_what_the_solver_can_use_is_a_422(tmp_path, capfd, change):
    request = example_request()
    change(request)
    response = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=request)
    assert response.status_code == 422
    assert response.json()["type"] == "validation"
    assert "objective cost" in response.json()["error"]
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    assert main(["hedge", "--request", str(path)]) == 1
    out = capfd.readouterr().out
    assert out.startswith("error:")
    assert "objective cost" in out


@pytest.mark.parametrize("quantity", [10**20, -(10**20), 10**300, 10**400])
@pytest.mark.parametrize("path", ["/v1/hedge", "/v1/value"])
def test_a_book_quantity_past_the_limit_is_a_422(tmp_path, path, quantity):
    request = example_request()
    request["book"][0]["quantity"] = quantity
    if path == "/v1/value":
        request = {"book": request["book"], "market": request["market"]}
    response = _client(tmp_path).post(path, headers=ALICE, json=request)
    assert response.status_code == 422
    assert "quantity must be at most" in response.json()["error"]


def test_the_cli_names_a_book_quantity_past_the_limit(tmp_path, capfd):
    request = example_request()
    request["book"][0]["quantity"] = 10**300
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    assert main(["hedge", "--request", str(path)]) == 1
    assert "quantity must be at most" in capfd.readouterr().out


def _big_book_request(quantity, *, stock_only=False):
    request = example_request()
    request["book"] = [{**request["book"][0], "quantity": quantity}]
    request["run_stability"] = False
    request["scenarios"] = []
    request["config"] = {"risk_aversion": 1.0}
    if stock_only:
        request["candidates"] = request["candidates"][:1]
        request["targets"] = request["targets"][:1]
    return request


@pytest.mark.parametrize("quantity", [2 * 10**7, 10**8, 10**10, 10**12])
def test_a_large_book_gets_a_hedge_at_capacity_not_one_too_small(tmp_path, quantity):
    # Long calls with far more delta than the market can offset. Oracle: every hedge that helps
    # sells all it can, 100,000 shares and each option's displayed 500. It was once 30, optimal.
    response = _client(tmp_path).post(
        "/v1/hedge", headers=ALICE, json=_big_book_request(quantity, stock_only=True)
    )
    assert response.status_code == 200, response.text
    assert [leg["quantity"] for leg in response.json()["legs"]] == [-100_000]
    full = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=_big_book_request(quantity))
    assert full.status_code == 200, full.text
    legs = sorted(leg["quantity"] for leg in full.json()["legs"])
    assert legs == [-100_000, -500, -500, -500, -500, -500]


def test_the_position_limit_is_exactly_the_documented_1e12(tmp_path):
    client = _client(tmp_path)
    accepted = client.post(
        "/v1/hedge", headers=ALICE, json=_big_book_request(10**12, stock_only=True)
    )
    assert accepted.status_code == 200
    assert [leg["quantity"] for leg in accepted.json()["legs"]] == [-100_000]
    for quantity in (10**12 + 1, -(10**12) - 1):
        refused = client.post(
            "/v1/hedge", headers=ALICE, json=_big_book_request(quantity, stock_only=True)
        )
        assert refused.status_code == 422
        assert "quantity must be at most" in refused.json()["error"]


def _cvar_only_request(quantity, spot):
    request = _big_book_request(quantity, stock_only=True)
    request["book"][0]["instrument"] = {**request["book"][0]["instrument"], "strike": spot}
    request["market"]["underlyings"]["XYZ"]["spot"] = spot
    quote = request["candidates"][0]["quote"]
    quote.update(bid=spot * 0.9999, ask=spot * 1.0001)
    request["targets"] = []
    request["scenarios"] = [{"name": "down", "spot_shock": -0.2, "vol_shock": 0.3}]
    request["config"] = {"cvar_weight": 1.0}
    return request


def test_cvar_at_a_size_the_solver_can_read_is_solved(tmp_path):
    request = _cvar_only_request(10**12, 1e7)
    response = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=request)
    assert response.status_code == 200, response.text
    assert [leg["quantity"] for leg in response.json()["legs"]] == [-100_000]


def test_cvar_past_what_the_solver_can_read_is_not_called_infeasible(tmp_path):
    # A scenario loss of about 2.5e20 reads as infinite to HiGHS, and gave a false proof of
    # infeasibility telling the user to check candidate capacity.
    request = _cvar_only_request(10**12, 1e8)
    response = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=request)
    assert response.status_code == 422
    body = response.json()
    assert body["type"] == "validation"
    assert "treats as finite" in body["error"]
    assert "book size, spot, and multiplier" in body["error"]


def _stock_beside_a_giant_future(book_shares, *, future_lots, stock_shares, band=None):
    # A future of 1e6 a point is about 1e8 of delta a contract, against $100 a share for the stock.
    request = example_request()
    stock = request["candidates"][0]
    request["book"] = [{"instrument": stock["instrument"], "quantity": book_shares}]
    request["run_stability"] = False
    request["scenarios"] = []
    request["config"] = {"risk_aversion": 1.0}
    request["targets"] = [{"underlying": "XYZ", "greek": "delta_dollars", **(band or {})}]
    stock["quote"].update(bid_size=stock_shares, ask_size=stock_shares)
    future = {
        "type": "linear",
        "symbol": "XYF",
        "underlying": "XYZ",
        "kind": "future",
        "multiplier": 1e6,
        "expiry": (DEFAULT_AS_OF + timedelta(days=30)).isoformat(),
    }
    quote = {
        **stock["quote"],
        "bid": 100.4,
        "ask": 100.42,
        "bid_size": future_lots,
        "ask_size": future_lots,
    }
    request["candidates"] = [stock, {"instrument": future, "quote": quote}]
    return request


GIANT_FUTURE_BOOKS = {
    # Once a false proof of infeasibility: this band is reachable, as the unbanded hedge shows.
    "banded": _stock_beside_a_giant_future(
        -(10**9 + 500_000),
        future_lots=5000,
        stock_shares=1_000_000,
        band={"lower": -1e6, "upper": 1e6},
    ),
    "unbanded": _stock_beside_a_giant_future(
        -(10**9 + 500_000), future_lots=5000, stock_shares=1_000_000
    ),
    # Once 30 shares and 5,000 futures, reported as optimal, with the stock worth using in full.
    "largest-book": _stock_beside_a_giant_future(
        -(10**12), future_lots=100_000, stock_shares=100_000
    ),
}


@pytest.mark.parametrize("request_body", GIANT_FUTURE_BOOKS.values(), ids=GIANT_FUTURE_BOOKS)
def test_a_material_instrument_too_small_for_the_solver_is_refused_by_name(tmp_path, request_body):
    client = _client(tmp_path)
    response = client.post("/v1/hedge", headers=ALICE, json=request_body)
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["type"] == "validation"
    assert "XYZ (spot) is too small" in body["error"]
    assert "book size, spot, and multiplier" in body["error"]
    frontier = client.post("/v1/frontier", headers=ALICE, json=request_body)
    assert frontier.status_code == 200, frontier.text
    solves = [p for p in frontier.json()["points"] if p["label"].startswith("risk aversion")]
    assert {p["status"] for p in solves} == {"out_of_range"}


def test_the_same_two_instruments_are_fine_against_an_ordinary_book(tmp_path):
    request = _stock_beside_a_giant_future(-10_000, future_lots=5000, stock_shares=1_000_000)
    response = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=request)
    assert response.status_code == 200, response.text
    assert [leg["quantity"] for leg in response.json()["legs"]] == [10_000]


def test_nothing_able_to_trade_cannot_be_refused_for_being_too_small():
    # The far option has a vega near 1e-29 and no capacity, like the other one. This is the
    # partial coverage path: one blocked underlying must not cost the whole answer a refusal.
    candidates = tuple(
        HedgeCandidate(
            euro("call", strike, days), Quote(bid, ask, 100, 100, AS_OF), max_buy=0, max_sell=0
        )
        for strike, days, bid, ask in ((100.0, 60, 4.5, 4.6), (150.0, 7, 0.01, 0.02))
    )
    problem = HedgeProblem(
        (Position(euro("call", 100, 30), -10),),
        candidates,
        (GreekTarget("XYZ", GreekName.VEGA),),
    )
    proposal = solve_hedge(problem, flat_snapshot())
    assert proposal.status.value == "partial_coverage"
    assert proposal.legs == ()
    assert any("NO_USABLE_CANDIDATES" in issue for issue in proposal.issues)


def test_float_error_on_a_huge_offset_book_is_not_a_failed_verification():
    # 4e9 + 7 shares hedged down to under $50 with a future of about 1e8 a contract and some stock.
    # About 8e11 of gross delta carries float error like the 6.1e-05 reported from this book, an
    # HTTP 500 telling the user to report a bug, when what remains would only have allowed 5e-05.
    request = _stock_beside_a_giant_future(4 * 10**9 + 7, future_lots=5000, stock_shares=10_000)
    problem, snapshot, _ = parse_hedge_request(request)
    stock, future = (candidate.instrument for candidate in problem.candidates)
    book_delta = value_book(problem.book, snapshot).by_underlying["XYZ"].delta_dollars
    future_delta = value_book([Position(future, 1)], snapshot).by_underlying["XYZ"].delta_dollars
    futures = -round(book_delta / future_delta)
    remainder = book_delta + futures * future_delta
    shares = -round(remainder / 100.0)
    legs = (
        HedgeLeg(stock, shares, 100.01, shares * 100.0, 0.0, "stock"),
        HedgeLeg(future, futures, 100.4, 0.0, 0.0, "future"),
    )
    hedged = value_book(
        [*problem.book, Position(stock, shares), Position(future, futures)], snapshot
    )
    after = hedged.by_underlying["XYZ"].delta_dollars
    assert abs(after) <= 50.0
    within_float = {("XYZ", "delta_dollars"): after + 6.1e-05}
    assert optimizer.verify_proposal(problem, snapshot, legs, within_float, {}).passed
    # A dollar is not float error at this size, where float error is under a millionth of that.
    beyond_float = {("XYZ", "delta_dollars"): after + 1.0}
    assert not optimizer.verify_proposal(problem, snapshot, legs, beyond_float, {}).passed


def test_verification_still_catches_a_small_error_on_an_ordinary_book():
    # The gross-exposure allowance is only for float error. A cent off is not float error.
    problem, snapshot, _ = parse_hedge_request(example_request())
    proposal = solve_hedge(problem, snapshot)
    predicted = {(line.underlying, line.greek): line.after for line in proposal.residuals}
    scenarios = dict(proposal.scenario_pnl_after)
    assert optimizer.verify_proposal(problem, snapshot, proposal.legs, predicted, scenarios).passed
    off = {key: value + 0.01 for key, value in predicted.items()}
    assert not optimizer.verify_proposal(problem, snapshot, proposal.legs, off, scenarios).passed
    # Scenario P&L here is tens of thousands, so it has a few cents of room, and a dollar is out.
    worse = {name: value + 1.0 for name, value in scenarios.items()}
    assert not optimizer.verify_proposal(problem, snapshot, proposal.legs, predicted, worse).passed


def test_a_solver_blind_to_an_instrument_cannot_read_as_verified(monkeypatch):
    # The old scale, which hid every hedge from HiGHS, stands in for any such blind spot. Selling
    # the stock's capacity then beats the bound the solver reports, which no feasible hedge can do.
    problem, snapshot, _ = parse_hedge_request(_big_book_request(10**8, stock_only=True))
    monkeypatch.setattr(optimizer._Program, "_row_scale", lambda self, target: 1e11)
    with pytest.raises(VerificationError, match="missed a better hedge"):
        solve_hedge(problem, snapshot)


def _strike_grid_candidates(spot, strikes, day_counts, *, sizes=10, **caps):
    """Fair quotes on every strike and expiry, on the same flat 25% vol as `flat_snapshot`."""
    candidates = []
    for days in day_counts:
        for strike in strikes:
            for right in ("call", "put"):
                omega = 1 if right == "call" else -1
                price = black_greeks(spot, strike, days / 365, 0.05, 0.0, 0.25, omega).price[0]
                if price < 0.05:
                    continue
                half = max(0.01 * price, 0.02)
                quote = Quote(max(price - half, 0.01), price + half, sizes, sizes, AS_OF)
                candidates.append(HedgeCandidate(euro(right, strike, days), quote, **caps))
    return tuple(candidates)


DGV = tuple(
    GreekTarget("XYZ", greek) for greek in (GreekName.DELTA, GreekName.GAMMA, GreekName.VEGA)
)


def _strike_grid_hedge(spot, step, candidates_kwargs=None, *, lots=50, **config):
    # Eight expiries, and strikes from about 73% to 133% of spot, on the shape of a real chain.
    strikes = np.arange(round(spot * 0.73 / step) * step, spot * 1.34, step)
    days = (7, 14, 30, 45, 60, 90, 120, 180)
    kwargs = candidates_kwargs or {"max_buy": 5000, "max_sell": 5000}
    candidates = _strike_grid_candidates(spot, strikes, days, **kwargs)
    book = (
        Position(euro("call", float(strikes[len(strikes) // 2]), 30), -lots),
        Position(euro("put", float(strikes[len(strikes) // 3]), 30), -lots),
    )
    problem = HedgeProblem(book, candidates, DGV, (), HedgeConfig(max_legs=6, **config))
    return solve_hedge(problem, flat_snapshot(spot=spot))


@pytest.mark.parametrize("lots", [50, 500, 5000])
@pytest.mark.parametrize(("spot", "step"), [(6000.0, 25.0), (600.0, 2.5), (150.0, 1.0)])
def test_a_real_strike_grid_with_depth_gets_a_hedge(spot, step, lots):
    # Deep options here have gamma and vega under 1e-9 of the largest on their rows, and at 5,000
    # lots these requests were refused for a size the book does not have, for cents at stake.
    proposal = _strike_grid_hedge(spot, step, lots=lots)
    assert 1 <= len(proposal.legs) <= 6
    assert proposal.verification.passed


def test_a_real_strike_grid_with_absent_sizes_gets_a_hedge():
    # The Cboe source leaves SPX sizes empty, which allows every leg up to the 5,000 lot limit.
    proposal = _strike_grid_hedge(
        6000.0, 25.0, {"sizes": 0}, require_quote_size=False, risk_aversion=1.0
    )
    assert 1 <= len(proposal.legs) <= 6


def test_ordinary_books_on_a_strike_grid_are_not_refused():
    # A seeded fuzz of ordinary books, with the count recorded: none of these may be refused.
    rng = np.random.default_rng(11)
    spot = 755.0
    strikes = np.arange(550.0, 1005.0, 5.0)
    days = (7, 14, 30, 60, 90, 180)
    candidates = _strike_grid_candidates(spot, strikes, days, max_buy=5000, max_sell=5000)
    refusals = []
    trials = 25
    for _ in range(trials):
        book = tuple(
            Position(
                euro(
                    str(rng.choice(["call", "put"])),
                    float(rng.choice(strikes)),
                    int(rng.choice(days)),
                ),
                int(rng.integers(-500, 501)) or 1,
            )
            for _ in range(int(rng.integers(1, 5)))
        )
        problem = HedgeProblem(book, candidates, DGV, (), HedgeConfig(max_legs=6))
        try:
            solve_hedge(problem, flat_snapshot(spot=spot))
        except SolverRangeError as exc:
            refusals.append(str(exc))
    assert refusals == [], f"{len(refusals)} of {trials} refused"


def _re_solves_out_of_range(monkeypatch):
    # A market move can take a request past the solver's range when its base solve was fine.
    real_check = optimizer._Program._check_solver_range
    calls = []

    def after_the_base_solve(program, constraints):
        calls.append(1)
        if len(calls) > 1:
            raise SolverRangeError("planted")
        real_check(program, constraints)

    monkeypatch.setattr(optimizer._Program, "_check_solver_range", after_the_base_solve)


def test_a_re_solve_out_of_range_is_its_own_outcome(monkeypatch):
    # Unknown is neither a 422 for a request that has an answer, nor unstable, nor called stable.
    problem, snapshot, _ = parse_hedge_request(example_request())
    _re_solves_out_of_range(monkeypatch)
    proposal = solve_hedge(problem, snapshot, run_stability=True)
    assert proposal.status.value != "unstable"
    assert proposal.legs
    assert proposal.stability.is_stable is False
    assert proposal.stability.notes == ()
    assert proposal.stability.regret_ratio != float("inf")
    assert len(proposal.stability.out_of_range) == len(PERTURBATIONS)
    incomplete = [i for i in proposal.issues if i.startswith("STABILITY_INCOMPLETE")]
    assert len(incomplete) == len(PERTURBATIONS)
    assert "planted" in incomplete[0]
    assert proposal_to_json(proposal)["stability"]["out_of_range"]


def test_a_held_hedge_breaking_its_band_stays_unstable_when_the_re_solve_is_out_of_range(
    monkeypatch,
):
    # The block must survive this path. Gamma cannot be re-tuned with shares, and a band this tight
    # around today's hedge fits it now and not after a small spot move. That is a fact about the
    # held hedge, whatever the solver can say.
    problem, snapshot, _ = parse_hedge_request(example_request())
    today = solve_hedge(problem, snapshot).after["XYZ"].gamma_1pct
    request = example_request()
    request["targets"][1].update(lower=today - 1.0, upper=today + 1.0)
    problem, snapshot, _ = parse_hedge_request(request)
    assert solve_hedge(problem, snapshot).legs
    _re_solves_out_of_range(monkeypatch)
    proposal = solve_hedge(problem, snapshot, run_stability=True)
    assert proposal.status.value == "unstable"
    assert proposal.stability.regret_ratio == float("inf")
    assert proposal.stability.out_of_range


# Reach of a few hundred dollars of typical shock: under the refusal, yet deciding the answer.
BAND_ONLY_THE_STOCK_REACHES = _stock_beside_a_giant_future(
    -5_020_589_925, future_lots=5000, stock_shares=500, band={"lower": -1e4, "upper": 1e4}
)


def test_a_band_only_an_unseen_instrument_can_reach_is_not_called_infeasible(tmp_path):
    # The oracle is a hedge by hand: 5,000 futures and 300 shares sold, repriced from scratch, are
    # inside both the band and the caps, so "no whole number of lots and shares is" would be false.
    problem, snapshot, _ = parse_hedge_request(BAND_ONLY_THE_STOCK_REACHES)
    stock, future = (candidate.instrument for candidate in problem.candidates)
    hedged = [*problem.book, Position(future, 5000), Position(stock, -300)]
    assert abs(value_book(hedged, snapshot).by_underlying["XYZ"].delta_dollars) < 1e4

    client = _client(tmp_path)
    response = client.post("/v1/hedge", headers=ALICE, json=BAND_ONLY_THE_STOCK_REACHES)
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["type"] == "validation"
    assert "XYZ (spot) is too small" in body["error"]
    assert "not a proof" in body["error"]
    frontier = client.post("/v1/frontier", headers=ALICE, json=BAND_ONLY_THE_STOCK_REACHES)
    assert frontier.status_code == 200, frontier.text
    statuses = {p["status"] for p in frontier.json()["points"] if p["label"].startswith("risk")}
    assert statuses == {"out_of_range"}


def test_a_better_hedge_the_solver_could_not_see_is_not_reported_as_a_bug(tmp_path):
    # The oracle: 1,000 futures and every share on offer sold improves on the solver's bound, which
    # cannot be, so this is what the solver could not see. It was a 500 saying to report a bug.
    request = _stock_beside_a_giant_future(-1_004_500_000, future_lots=1000, stock_shares=500)
    client = _client(tmp_path)
    response = client.post("/v1/hedge", headers=ALICE, json=request)
    assert response.status_code == 422, response.text
    assert response.json()["type"] == "validation"
    assert "XYZ (spot) is too small" in response.json()["error"]
    frontier = client.post("/v1/frontier", headers=ALICE, json=request)
    assert frontier.status_code == 200, frontier.text
    statuses = [p["status"] for p in frontier.json()["points"]]
    assert "out_of_range" in statuses
    assert "infeasible" not in statuses


def test_a_lower_bound_miss_by_a_visible_instrument_is_still_reported_as_a_bug(monkeypatch):
    # The distinction must not swallow a real defect: nothing here is unseen, and the bound is
    # wrong.
    problem, snapshot, _ = parse_hedge_request(_stock_only_request())
    real_solve = optimizer._Program.solve

    def lower(program):
        return replace(real_solve(program), lower_bound=1e12)

    monkeypatch.setattr(optimizer._Program, "solve", lower)
    with pytest.raises(VerificationError, match="missed a better hedge"):
        solve_hedge(problem, snapshot)


def test_dust_on_a_real_chain_is_seen_as_dropped_and_still_gets_a_hedge(monkeypatch):
    # A result that stands is returned with deep options unseen: the rule is about proofs.
    real_dropped = optimizer._Program._dropped_by_the_solver
    seen = []

    def recording(program):
        seen.append(real_dropped(program))
        return seen[-1]

    monkeypatch.setattr(optimizer._Program, "_dropped_by_the_solver", recording)
    proposal = _strike_grid_hedge(6000.0, 25.0, lots=5000)
    assert any(seen)
    assert 1 <= len(proposal.legs) <= 6
    assert proposal.verification.passed


def test_a_truly_impossible_band_on_a_chain_with_dust_is_a_refusal_not_a_proof(monkeypatch):
    # The price of the rule, pinned: with deep options unseen, the solver cannot prove this either.
    # 1e15 of delta is past all of this universe at capacity, about 1e12, so it is truly impossible.
    spot = 6000.0
    strikes = np.arange(4400.0, 8000.0, 25.0)
    candidates = _strike_grid_candidates(spot, strikes, (7, 30, 90), max_buy=5000, max_sell=5000)
    book = (Position(euro("call", 6000.0, 30), -5000),)
    band = (GreekTarget("XYZ", GreekName.DELTA, lower=1e15, upper=2e15), *DGV[1:])
    problem = HedgeProblem(book, candidates, band, (), HedgeConfig())
    with pytest.raises(SolverRangeError, match="not a proof"):
        solve_hedge(problem, flat_snapshot(spot=spot))


def test_one_frontier_point_out_of_range_leaves_the_others_standing(tmp_path):
    # A delta weight where a risk aversion of 10 passes the 1e15 cost limit and 3 does not: the
    # row's scale is the book's delta, 32,781, so the penalty at 1 is about 2e14.
    request = example_request()
    request["targets"] = [{"underlying": "XYZ", "greek": "delta_dollars", "weight": 6e9}]
    response = _client(tmp_path).post("/v1/frontier", headers=ALICE, json=request)
    assert response.status_code == 200, response.text
    points = {point["label"]: point for point in response.json()["points"]}
    assert points["risk aversion 10"]["status"] == "out_of_range"
    assert "objective cost" in points["risk aversion 10"]["unsolved_reason"]
    assert points["risk aversion 0.01"]["unsolved_reason"] is None
    assert points["do nothing"]["n_legs"] == 0


def test_an_ordinary_hedge_is_not_called_inadequate():
    problem, snapshot, _ = parse_hedge_request(example_request())
    assert solve_hedge(problem, snapshot, run_stability=True).legs


COST_LIMIT_REQUEST = {"max_legs": 5}


@pytest.mark.parametrize(
    ("ticket_cost", "accepted"),
    [(1e15, True), (1.0000001e15, False)],
    ids=["at-the-limit", "just-past-it"],
)
def test_the_cost_limit_is_exactly_the_documented_1e15(tmp_path, ticket_cost, accepted):
    request = example_request()
    request["run_stability"] = False
    request["config"] = {**COST_LIMIT_REQUEST, "ticket_cost": ticket_cost}
    response = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=request)
    assert response.status_code == (200 if accepted else 422), response.text


# --- A delta-one quote must agree with the market -------------------------------------------------


def _stock_quoted_at(mid, *, options=False, **config):
    request = example_request()
    if not options:
        request["candidates"] = request["candidates"][:1]
        request["targets"] = request["targets"][:1]
    request["candidates"][0]["quote"].update(bid=mid - 0.01, ask=mid + 0.01)
    request["run_stability"] = False
    request["scenarios"] = []
    request["config"] = {"risk_aversion": 1.0, **config}
    return request


@pytest.mark.parametrize("mid", [200.0, 10.0, 126.0, 74.0])
def test_a_stock_quote_far_from_spot_is_excluded_with_its_reason(tmp_path, mid):
    # This hedged 328 shares at any of these prices, with the limit reaching the ticket.
    response = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=_stock_quoted_at(mid))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "no_usable_candidates"
    assert body["legs"] == []
    assert len(body["excluded"]) == 1
    assert "the market implies" in body["excluded"][0]["reason"]


@pytest.mark.parametrize("mid", [100.05, 99.95, 124.0, 76.0])
def test_a_stock_quote_near_spot_is_used(tmp_path, mid):
    response = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=_stock_quoted_at(mid))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["excluded"] == []
    assert [leg["side"] for leg in body["legs"]] == ["BUY"]


def test_options_still_hedge_when_only_the_stock_is_excluded(tmp_path):
    request = _stock_quoted_at(200.0, options=True)
    body = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=request).json()
    assert [e["instrument"] for e in body["excluded"]] == ["XYZ (spot)"]
    assert body["legs"]
    assert all(leg["instrument"]["type"] == "option" for leg in body["legs"])


@pytest.mark.parametrize(
    ("mid", "usable"),
    # Forward by hand: 100 * exp(0.05 * 90/365) = 101.24, so 200 is far and 101.2 is near. A check
    # against spot alone could not tell a future at 101.2 from one at 100.
    [(101.2, True), (200.0, False), (10.0, False)],
)
def test_a_future_is_compared_with_the_forward(tmp_path, mid, usable):
    request = _stock_quoted_at(100.0)
    request["candidates"][0]["instrument"] = {
        "type": "linear",
        "symbol": "XYF",
        "underlying": "XYZ",
        "kind": "future",
        "multiplier": 50.0,
        "expiry": (DEFAULT_AS_OF + timedelta(days=90)).isoformat(),
    }
    request["candidates"][0]["quote"].update(bid=mid - 0.01, ask=mid + 0.01)
    body = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=request).json()
    assert bool(body["legs"]) is usable
    assert len(body["excluded"]) == (0 if usable else 1)


def _spxl_request():
    # SPXL at 200 on an SPX at 6000: triple leverage, so 3 * 200 / 6000 is a delta ratio of 0.1,
    # while spot times that ratio is 600. A correct quote, a long way from what the rule would say.
    request = _stock_quoted_at(200.0)
    request["candidates"][0]["instrument"] = {
        "type": "linear",
        "symbol": "SPXL",
        "underlying": "XYZ",
        "kind": "proxy",
        "delta_ratio": 0.1,
        "basis_vol": 0.05,
    }
    request["book"] = [{**request["book"][0], "quantity": -10}]
    request["book"][0]["instrument"] = {**request["book"][0]["instrument"], "strike": 6000.0}
    request["market"]["underlyings"]["XYZ"]["spot"] = 6000.0
    return request


def test_a_leveraged_proxy_is_not_refused_and_says_it_was_not_checked(tmp_path):
    body = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=_spxl_request()).json()
    assert body["excluded"] == []
    assert [leg["instrument"]["symbol"] for leg in body["legs"]] == ["SPXL"]
    unchecked = [i for i in body["issues"] if i.startswith("QUOTE_UNCHECKED")]
    assert len(unchecked) == 1
    assert "SPXL" in unchecked[0]


def test_a_proxy_only_offered_raises_no_unchecked_issue(tmp_path):
    # The sample hedge with a proxy on offer but too dear to use: nothing to warn the approver of.
    request = _stock_quoted_at(100.0, options=True)
    request["candidates"].append(
        {
            "instrument": {
                "type": "linear",
                "symbol": "XYP",
                "underlying": "XYZ",
                "kind": "proxy",
                "basis_vol": 5.0,
            },
            "quote": {**request["candidates"][0]["quote"], "bid": 50.0, "ask": 150.0},
        }
    )
    body = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=request).json()
    assert "XYP" not in [leg["instrument"].get("symbol") for leg in body["legs"]]
    assert not any(i.startswith("QUOTE_UNCHECKED") for i in body["issues"])


def test_the_quote_gate_can_be_set_by_a_deployment_only(tmp_path, monkeypatch):
    refused = _client(tmp_path).post(
        "/v1/hedge", headers=ALICE, json=_stock_quoted_at(200.0, max_linear_quote_gap=2.0)
    )
    assert refused.status_code == 422
    assert "max_linear_quote_gap" in refused.json()["error"]
    monkeypatch.setenv("GN_GATE_MAX_LINEAR_QUOTE_GAP", "2.0")
    allowed = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=_stock_quoted_at(200.0))
    assert allowed.json()["excluded"] == []


def test_the_cli_shows_the_exclusion(tmp_path, capfd):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(_stock_quoted_at(200.0)))
    assert main(["hedge", "--request", str(path)]) == 0
    out = capfd.readouterr().out
    assert "status: no_usable_candidates" in out
    assert "1 candidates excluded" in out
    assert main(["hedge", "--request", str(path), "--full"]) == 0
    assert "the market implies" in capfd.readouterr().out


# --- A proxy's mark is called what it is ----------------------------------------------------------


def _proxy_market_and_book():
    proxy = {
        "type": "linear",
        "symbol": "SPXL",
        "underlying": "XYZ",
        "kind": "proxy",
        "delta_ratio": 0.1,
        "basis_vol": 0.05,
    }
    market = example_market()
    market["underlyings"]["XYZ"]["spot"] = 6000.0
    return market, [{"instrument": proxy, "quantity": 1000}]


@pytest.mark.parametrize(
    ("path", "extra"),
    [
        ("/v1/value", {}),
        ("/v1/ladder", {}),
        ("/v1/limits", {"limits": [{"metric": "delta_shares", "lower": -1e9, "upper": 1e9}]}),
        ("/v1/decay", {"underlying": "XYZ", "days": [1]}),
        ("/v1/stress", {}),
        ("/v1/grid", {}),
    ],
)
def test_a_proxy_in_the_book_is_named_as_marked_at_its_ratio(tmp_path, path, extra):
    market, book = _proxy_market_and_book()
    payload = {"book": book, "market": market, **extra}
    response = _client(tmp_path).post(path, headers=ALICE, json=payload)
    assert response.status_code == 200, response.text
    marks = [i for i in response.json()["issues"] if i.startswith("PROXY_MARK")]
    assert len(marks) == 1
    assert "SPXL" in marks[0]
    assert "spot times its delta ratio" in marks[0]


def test_the_mark_itself_is_spot_times_the_ratio():
    # 1,000 units at 6000 * 0.1, the convention documented, and not the fund's 200 quote.
    market, book = _proxy_market_and_book()
    valued = value_book(parse_book(book), parse_market(market))
    assert valued.total_value == pytest.approx(600_000.0)


def test_a_proposal_holding_or_trading_a_proxy_says_how_it_is_marked(tmp_path):
    market, book = _proxy_market_and_book()
    request = _spxl_request()
    request["book"] = [*request["book"], *book]
    request["market"] = market
    body = _client(tmp_path).post("/v1/hedge", headers=ALICE, json=request).json()
    assert any(i.startswith("PROXY_MARK") and "SPXL" in i for i in body["issues"])


def test_a_stock_has_no_proxy_mark_issue(tmp_path):
    payload = {"book": HUNDRED_SHARES, "market": example_market()}
    body = _client(tmp_path).post("/v1/value", headers=ALICE, json=payload).json()
    assert body["issues"] == []
