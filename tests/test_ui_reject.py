"""The approval screen's reject flow, run from the page's own script against a live service."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient

from greeks_neutralizer.examples import DEFAULT_AS_OF, example_request
from greeks_neutralizer.service.app import create_app

DRIVER = Path(__file__).with_name("ui_reject_driver.cjs")
ALICE = {"Authorization": "Bearer a-token"}
BOB = {"Authorization": "Bearer b-token"}


@pytest.fixture
def service(tmp_path: Path) -> Iterator[tuple[str, TestClient]]:
    app = create_app(
        {"a-token": "alice", "b-token": "bob", "c-token": "carol"},
        tmp_path / "audit.jsonl",
        200_000,
        clock=lambda: DEFAULT_AS_OF,
    )
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="critical"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", TestClient(app)
    server.should_exit = True
    thread.join(timeout=10.0)


def test_each_rejection_records_only_the_reason_given_for_it(
    service: tuple[str, TestClient],
) -> None:
    # A real browser run recorded one approver's reason against a second proposal with no typing,
    # in an audit log that cannot be corrected. The oracle is that log, read back over HTTP.
    base_url, client = service
    node = shutil.which("node")
    if node is None:
        if os.environ.get("CI"):
            pytest.fail("node is required in CI, so this test cannot pass by being skipped")
        pytest.skip("node is not installed")

    request = example_request()
    request["scenarios"] = "historical"
    for _ in range(3):
        created = client.post("/v1/proposals", headers=ALICE, json=request)
        assert created.status_code == 201, created.text

    reasons = ["quote looks stale", "cost looks wrong"]
    done = subprocess.run(
        [node, str(DRIVER), base_url, "b-token", "c-token", *reasons],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout)

    # A used reason leaves the field, and an unused one stays for the approver to retry with.
    assert seen["afterRejecting"] == ""
    assert "a rejection needs a reason" in seen["errorWithoutTyping"]
    assert "cannot reject a proposal that is approved" in seen["errorOnRefusal"]
    assert seen["afterRefusal"] == "cost looks wrong"

    audit = client.get("/v1/audit", headers=BOB, params={"limit": 25}).json()
    records = [r["detail"] for r in audit["records"] if r["event"] == "rejected"]
    rejected = {detail["id"]: detail["reason"] for detail in records}
    assert rejected == {seen["first"]: "quote looks stale", seen["second"]: "cost looks wrong"}
