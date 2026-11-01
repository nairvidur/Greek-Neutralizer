"""HTTP service: auth, validation, the hedge flow, four-eyes, audit, and safe failure."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from greeks_neutralizer.examples import (
    DEFAULT_AS_OF,
    example_book,
    example_market,
    example_request,
)
from greeks_neutralizer.service.app import _clean, _parse_tokens, create_app

ALICE = {"Authorization": "Bearer a-token"}
BOB = {"Authorization": "Bearer b-token"}


@pytest.fixture
def client(tmp_path):
    app = create_app(
        {"a-token": "alice", "b-token": "bob"},
        tmp_path / "audit.jsonl",
        200_000,
        clock=lambda: DEFAULT_AS_OF,
    )
    return TestClient(app, raise_server_exceptions=False)


def quick_request():
    request = example_request()
    request["scenarios"] = "historical"
    return request


def test_refuses_to_start_without_tokens(tmp_path, monkeypatch):
    monkeypatch.delenv("GN_TOKENS", raising=False)
    with pytest.raises(RuntimeError):
        create_app(audit_path=tmp_path / "a.jsonl")
    with pytest.raises(RuntimeError):
        create_app({}, tmp_path / "a.jsonl")


def test_token_parsing():
    assert _parse_tokens("x=alice, y=bob") == {"x": "alice", "y": "bob"}
    with pytest.raises(RuntimeError):
        _parse_tokens("justatoken")


@pytest.mark.parametrize("path", ["/v1/whoami", "/v1/example", "/v1/proposals", "/v1/audit"])
def test_data_endpoints_require_auth(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401


@pytest.mark.parametrize("path", ["/v1/value", "/v1/hedge", "/v1/stress", "/v1/proposals"])
def test_mutating_endpoints_require_auth(client, path):
    assert client.post(path, json={}).status_code == 401


def test_health_and_ui_are_open_but_hold_no_data(client):
    created = client.post("/v1/proposals", headers=ALICE, json=quick_request())
    assert created.status_code == 201
    record = created.json()
    secrets = (
        record["id"],
        "alice",
        "a-token",
        record["proposal"]["manifest"]["market_fingerprint"],
    )
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert set(health.json()) == {"status", "version", "audit_records", "detail"}
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    page = client.get("/")
    assert page.status_code == 200
    assert "Greeks Neutralizer" in page.text
    for response in (health, metrics, page):
        for secret in secrets:
            assert secret not in response.text


def test_identity_comes_from_the_token(client):
    assert client.get("/v1/whoami", headers=ALICE).json() == {"actor": "alice"}


def test_value_endpoint(client):
    response = client.post(
        "/v1/value", headers=ALICE, json={"book": example_book(), "market": example_market()}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["totals"]["delta_dollars"] != 0.0
    assert len(body["lines"]) == 2


def test_stress_and_grid(client):
    body = {"book": example_book(), "market": example_market(), "set": "historical"}
    stress = client.post("/v1/stress", headers=ALICE, json=body)
    assert stress.status_code == 200
    assert len(stress.json()["pnl"]) == 7
    grid = client.post(
        "/v1/grid", headers=ALICE, json={"book": body["book"], "market": body["market"]}
    )
    assert grid.status_code == 200
    data = grid.json()
    assert len(data["pnl"]) == len(data["spot_shocks"])
    assert len(data["pnl"][0]) == len(data["vol_shocks"])


def test_hedge_endpoint_returns_a_verified_proposal(client):
    response = client.post("/v1/hedge", headers=ALICE, json=quick_request())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["verification"]["passed"] is True
    assert body["manifest"]["problem_fingerprint"]


def test_frontier_endpoint(client):
    response = client.post("/v1/frontier", headers=ALICE, json=quick_request())
    assert response.status_code == 200, response.text
    assert response.json()["points"][0]["label"] == "do nothing"


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda r: r.update(typo=1), "unknown field"),
        (lambda r: r["market"].update(as_of="2026-01-05T15:00:00"), "UTC offset"),
        (lambda r: r["book"][0].update(quantity="ten"), "signed int"),
        (
            lambda r: r["market"]["underlyings"]["XYZ"].update(spot=float("nan")),
            "",
        ),
    ],
    ids=["typo", "naive", "qty", "nan-spot"],
)
def test_bad_input_is_a_422_not_a_crash(client, mutate, fragment):
    request = quick_request()
    mutate(request)
    content = json.dumps(request)
    response = client.post(
        "/v1/hedge", headers={**ALICE, "content-type": "application/json"}, content=content
    )
    assert response.status_code == 422
    assert fragment in response.text


def test_infeasible_reports_binding_constraints(client):
    request = quick_request()
    request["candidates"] = request["candidates"][:1]
    request["targets"] = [{"underlying": "XYZ", "greek": "vega_1pt", "lower": -1.0, "upper": 1.0}]
    response = client.post("/v1/hedge", headers=ALICE, json=request)
    assert response.status_code == 422
    body = response.json()
    assert body["type"] == "infeasible"
    assert any("vega_1pt" in text for text in body["binding"])


def test_oversized_body_rejected(client):
    response = client.post(
        "/v1/hedge",
        headers={**ALICE, "content-type": "application/json"},
        content="x" * 300_000,
    )
    assert response.status_code == 413


def test_request_id_is_returned(client):
    response = client.get("/healthz", headers={"x-request-id": "abc"})
    assert response.headers["x-request-id"] == "abc"


def test_four_eyes_flow_over_http(client):
    created = client.post("/v1/proposals", headers=ALICE, json=quick_request())
    assert created.status_code == 201, created.text
    record_id = created.json()["id"]

    own = client.post(f"/v1/proposals/{record_id}/approve", headers=ALICE, json={})
    assert own.status_code == 409
    assert "four-eyes" in own.text

    early = client.post(f"/v1/proposals/{record_id}/tickets", headers=BOB)
    assert early.status_code == 409

    approved = client.post(
        f"/v1/proposals/{record_id}/approve", headers=BOB, json={"reason": "checked"}
    )
    assert approved.status_code == 200
    assert approved.json()["decided_by"] == "bob"

    tickets = client.post(f"/v1/proposals/{record_id}/tickets", headers=BOB)
    assert tickets.status_code == 200
    assert tickets.text.startswith("proposal_id,side")

    fix = client.post(f"/v1/proposals/{record_id}/tickets?fmt=fix", headers=BOB)
    assert "35=D|" in fix.text

    audit = client.get("/v1/audit", headers=ALICE).json()
    assert audit["chain_ok"] is True
    assert [r["event"] for r in audit["records"]][-3:] == ["exported", "approved", "submitted"]


def test_rejection_flow(client):
    record_id = client.post("/v1/proposals", headers=ALICE, json=quick_request()).json()["id"]
    assert client.post(f"/v1/proposals/{record_id}/reject", headers=BOB, json={}).status_code == 422
    rejected = client.post(
        f"/v1/proposals/{record_id}/reject", headers=BOB, json={"reason": "too big"}
    )
    assert rejected.json()["state"] == "rejected"


def test_unknown_proposal_is_a_conflict_not_a_crash(client):
    assert client.get("/v1/proposals/nope", headers=ALICE).status_code == 409


def test_broken_audit_chain_shows_in_health(tmp_path):
    path = tmp_path / "audit.jsonl"
    app = create_app({"a-token": "alice"}, path, clock=lambda: DEFAULT_AS_OF)
    client = TestClient(app)
    client.post("/v1/proposals", headers=ALICE, json=quick_request())
    path.write_text(path.read_text().replace("alice", "mallory"))
    assert client.get("/healthz").json()["status"] == "audit_chain_broken"
    assert client.get("/v1/audit", headers=ALICE).json()["chain_ok"] is False


def test_clean_makes_values_json_safe():
    import numpy as np

    cleaned = _clean(
        {
            "a": float("nan"),
            "b": [np.float64(1.5), float("inf")],
            "c": np.int64(3),
            "d": np.bool_(True),
        }
    )
    assert cleaned == {"a": None, "b": [1.5, None], "c": 3, "d": True}
    assert json.dumps(cleaned)
