"""Run the operational drills end to end, against real service processes, at a small size.

The evidence in docs/evidence/phase5 comes from the same scripts at full size. These keep them
working, and hold the service to what the runbook promises about backup, restore, and damage.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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
