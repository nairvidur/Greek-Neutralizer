"""Regression tests from the fifth review round. Each failed before its fix.

Oracles: the FIX 4.4 required-field list for NewOrderSingle and its UTCTimestamp format; calendar
arithmetic for the New York date of an expiry; hand-worked dollar exposures of a share position and
hand-worked spread and volume arithmetic; and each gate's direction from the meaning of the gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from greeks_neutralizer.cli import main
from greeks_neutralizer.domain import (
    LinearInstrument,
    LinearKind,
    OptionContract,
    OptionRight,
    Position,
)
from greeks_neutralizer.errors import ValidationError
from greeks_neutralizer.examples import (
    DEFAULT_AS_OF,
    example_book,
    example_market,
    example_request,
)
from greeks_neutralizer.hedging.diagnostics import hedge_decay_forecast
from greeks_neutralizer.hedging.optimizer import solve_hedge
from greeks_neutralizer.risk.measures import RiskLimit, expiry_ladder, liquidity_adjusted_cost
from greeks_neutralizer.schema import parse_hedge_request
from greeks_neutralizer.service.app import create_app
from greeks_neutralizer.workflow import AuditLog, ProposalStore
from tests.conftest import AS_OF, flat_snapshot

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
    assert points["delta only"]["status"] is not None
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
    with pytest.raises(ValidationError):
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


def test_a_short_american_call_on_a_far_market_date_is_a_422(tmp_path):
    # The assignment window adds 7 days to as_of, which ran off the end of the calendar here.
    call = {**example_book()[0]["instrument"], "style": "american", "strike": 50.0}
    call["expiry"] = "9999-12-31T00:00:00+00:00"
    market = {**example_market(), "as_of": "9999-12-30T15:00:00+00:00"}
    payload = {"book": [{"instrument": call, "quantity": -1}], "market": market}
    assert _client(tmp_path).post("/v1/value", headers=ALICE, json=payload).status_code == 422


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


# A quantity far past what a float can hold reaches arithmetic the parser does not do.
ENORMOUS = 10**400


def test_arithmetic_out_of_range_on_inputs_is_a_422(tmp_path):
    payload = {
        "book": [{"instrument": STOCK, "quantity": ENORMOUS}],
        "market": example_market(),
    }
    response = _client(tmp_path).post("/v1/value", headers=ALICE, json=payload)
    assert response.status_code == 422
    assert "out of range" in response.json()["error"]


def test_arithmetic_out_of_range_on_inputs_is_a_structured_cli_error(tmp_path, capfd):
    book = tmp_path / "book.json"
    book.write_text(json.dumps([{"instrument": STOCK, "quantity": ENORMOUS}]))
    market = tmp_path / "market.json"
    market.write_text(json.dumps(example_market()))
    assert main(["value", "--book", str(book), "--market", str(market)]) == 1
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
    if target:
        assert all(p["status"] == "infeasible" for p in points[1:])


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
