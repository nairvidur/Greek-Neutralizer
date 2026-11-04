"""Run the operational drills end to end, against real service processes, at a small size.

The evidence in docs/evidence/phase5 comes from the same scripts at full size. These keep them
working, and hold the service to what the runbook promises about backup, restore, and damage.
"""

from __future__ import annotations

import importlib
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx2
import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.slow


def run_script(name: str, out: Path, *args: str) -> tuple[int, dict[str, Any]]:
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), "--out", str(out), *args],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert out.exists(), done.stdout + done.stderr
    return done.returncode, json.loads(out.read_text())


def test_restore_drill_recovers_and_refuses_damaged_backups(tmp_path: Path) -> None:
    code, result = run_script(
        "restore_drill.py", tmp_path / "drill.json", "--proposals", "2", "--crashes", "1"
    )

    restore = result["restore"]
    # Two proposals, each submitted and approved, are four records, and one more after restoring.
    assert restore["records_backed_up"] == 4
    assert restore["records_after_restore"] == 4
    assert restore["records_after_new_submission"] == 5
    assert restore["healthz_after_second_restart"] == 200

    # Every kind of damaged restore must be visible to a monitor and must block new work.
    cases = result["damaged_restores"]["cases"]
    assert set(cases) == {
        "sidecar_missing",
        "sidecar_from_an_earlier_time",
        "log_truncated",
        "record_edited",
    }
    for case in cases.values():
        assert (case["healthz"], case["healthz_status"]) == (503, "audit_chain_broken")
        assert case["submit"] == 409

    # A SIGKILL may leave an unsynced sidecar, but must never lose an acknowledged submission.
    crash = result["crash_recovery"]
    assert crash["attempts"][0]["acknowledged"] > 0
    assert crash["acknowledged_but_lost"] == 0
    assert code == 0


def _script(monkeypatch: pytest.MonkeyPatch, name: str) -> Any:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module(name)


def _record_started(monkeypatch: pytest.MonkeyPatch, load_test: Any) -> list[Any]:
    started: list[Any] = []
    real_start = load_test.start_service

    def start(*args: Any) -> Any:
        started.append(real_start(*args))
        return started[-1]

    monkeypatch.setattr(load_test, "start_service", start)
    return started


def test_a_service_that_never_becomes_ready_is_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_test = _script(monkeypatch, "load_test")
    started = _record_started(monkeypatch, load_test)
    monkeypatch.setattr(load_test, "READY_TIMEOUT_SECONDS", 3.0)
    with (
        pytest.raises(RuntimeError, match="not ready"),
        load_test.running_service(tmp_path / "audit.jsonl", frozenset({418})),
    ):
        pytest.fail("a service that never answered as accepted must not be yielded")
    assert len(started) == 1
    assert started[0].poll() is not None


def test_a_crash_drill_that_fails_early_stops_its_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_test = _script(monkeypatch, "load_test")
    restore_drill = _script(monkeypatch, "restore_drill")
    started = _record_started(monkeypatch, load_test)

    def fail(_: Any) -> None:
        raise RuntimeError("the drill failed before its crash")

    monkeypatch.setattr(restore_drill, "_hedge_request", fail)
    with pytest.raises(RuntimeError, match="before its crash"):
        restore_drill.crash_recovery(tmp_path, 1)
    assert len(started) == 1
    assert started[0].poll() is not None


def test_a_port_taken_before_the_bind_starts_again_on_a_new_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_test = _script(monkeypatch, "load_test")
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        taken_port = taken.getsockname()[1]
        real_free_port = load_test.free_port
        given: list[int] = []

        def first_the_taken_one() -> int:
            given.append(real_free_port() if given else taken_port)
            return given[-1]

        monkeypatch.setattr(load_test, "free_port", first_the_taken_one)
        with load_test.running_service(tmp_path / "audit.jsonl") as base:
            assert httpx2.get(f"{base}/healthz", timeout=10.0).status_code == 200
            # It tried the taken port, failed to bind, and came up on the next one.
            assert len(given) == 2
            assert given[0] == taken_port
            assert base.endswith(f":{given[1]}")


def test_load_test_accounts_for_every_action(tmp_path: Path) -> None:
    submissions = 6
    code, result = run_script(
        "load_test.py",
        tmp_path / "load.json",
        "--seconds",
        "1",
        "--rates",
        "2",
        "--clients",
        "2",
        "--submissions",
        str(submissions),
        "--growth",
        "",
    )

    integrity = result["integrity"]
    assert integrity["submit_statuses"] == {"201": submissions}
    # Each proposal is raced by two approvers, and only one can win.
    assert integrity["audit_events"] == {"approved": submissions, "submitted": submissions}
    assert integrity["chain_records"] == 2 * submissions
    assert integrity["chain_ok"]

    # Latency at a fixed rate runs from the schedule, so its count is what the schedule implies.
    assert result["fixed_rate_hedge"][0]["requests"] == 2
    assert code == 0


def test_open_loop_throughput_falls_below_the_offered_rate_when_saturated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oracle: a server that handles one request at a time, taking 50 ms each, completes at most
    20 a second. Offered 50 a second for 0.4 s, the measured throughput must show that limit.
    Dividing by the nominal duration reported exactly the offered rate, 50, whatever happened."""
    import threading
    import time

    load_test = _script(monkeypatch, "load_test")
    server = threading.Lock()

    class Done:
        status_code = 200

    def send(_: object) -> Done:
        with server:
            time.sleep(0.05)
        return Done()

    result = load_test.open_loop("http://127.0.0.1:9", send, rate=50.0, seconds=0.4)
    assert result["requests"] == 20
    assert result["offered_per_second"] == 50.0
    assert result["throughput_per_second"] < 21.0
    assert result["throughput_per_second"] > 10.0
