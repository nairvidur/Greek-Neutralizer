#!/usr/bin/env python3
"""Load test a real service process over HTTP, and check that nothing was lost under load.

Usage: load_test.py [--out results.json] [--seconds 20] [--rates 1,2,4] [--submissions 500]

It starts uvicorn on a free port with a new audit log, and measures three things:

- Capacity: closed-loop throughput per endpoint and client count. Latency here understates what a
  caller sees near saturation, since a slow server slows the clients that are measuring it.
- Latency at a fixed arrival rate: open-loop, timed from each request's scheduled start, so that a
  stall is charged to every request that had to wait for it (no coordinated omission).
- Integrity: concurrent submissions and approval races, then the audit log is verified from disk.
  Every acknowledged action must have exactly one record, and a race must have one winner.

Results describe the machine and the sample proposal used, not a production deployment.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx2
import numpy as np

from greeks_neutralizer._version import __version__
from greeks_neutralizer.workflow import verify_audit_log

TOKENS = {"alice": "load-token-alice", "bob": "load-token-bob", "carol": "load-token-carol"}
READY_TIMEOUT_SECONDS = 60.0
START_ATTEMPTS = 5
REQUEST_TIMEOUT_SECONDS = 120.0
GROWTH_REFRESH_EVERY = 200


class PortTakenError(RuntimeError):
    """The service could not bind: another process took its port after it was found free."""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_service(audit_log: Path, port: int, output: Any) -> subprocess.Popen[bytes]:
    """A real uvicorn process, configured through its environment as an operator would."""
    env = {
        **os.environ,
        "GN_TOKENS": ",".join(f"{token}={name}" for name, token in TOKENS.items()),
        "GN_AUDIT_LOG": str(audit_log),
    }
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "--factory",
        "greeks_neutralizer.service.app:factory",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    return subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)


def wait_until_answering(
    base: str, process: subprocess.Popen[bytes], output: Any, accepted: frozenset[int]
) -> None:
    """Until /healthz returns an accepted status. A failed start raises with its output."""
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output.seek(0)
            log = output.read().decode()
            kind = PortTakenError if "address already in use" in log.lower() else RuntimeError
            raise kind(f"the service exited at startup: {log[-2000:]}")
        with contextlib.suppress(httpx2.TransportError):
            status = httpx2.get(f"{base}/healthz", timeout=2.0).status_code
            if status in accepted:
                return
        time.sleep(0.1)
    raise RuntimeError(f"the service was not ready within {READY_TIMEOUT_SECONDS:.0f}s")


def stop_service(process: subprocess.Popen[bytes]) -> None:
    """SIGTERM, then SIGKILL if it does not exit in time. Safe on a process already gone."""
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def start_on_a_free_port(
    audit_log: Path, output: Any, accepted: frozenset[int]
) -> tuple[subprocess.Popen[bytes], str]:
    """A started service and its base URL. A port can be taken between finding it and binding it,
    so a failed bind starts again on a new port. Any other failure raises at once. A process that
    fails to start is always stopped, so the caller owns only one that started."""
    for attempt in range(1, START_ATTEMPTS + 1):
        port = free_port()
        output.seek(0)
        output.truncate()
        process = start_service(audit_log, port, output)
        base = f"http://127.0.0.1:{port}"
        try:
            wait_until_answering(base, process, output, accepted)
        except BaseException as exc:
            stop_service(process)
            if not isinstance(exc, PortTakenError) or attempt == START_ATTEMPTS:
                raise
        else:
            return process, base
    raise AssertionError("unreachable: the last attempt returns or raises")


@contextlib.contextmanager
def running_service(audit_log: Path, accepted: frozenset[int] = frozenset({200})) -> Iterator[str]:
    """Started on a free port, and stopped with SIGTERM. Yields its base URL."""
    with tempfile.TemporaryFile() as output:
        process, base = start_on_a_free_port(audit_log, output, accepted)
        try:
            yield base
        finally:
            stop_service(process)


def client(base: str, actor: str = "alice") -> httpx2.Client:
    return httpx2.Client(
        base_url=base,
        headers={"Authorization": f"Bearer {TOKENS[actor]}"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def summarize(latencies: list[float], statuses: Counter[int], seconds: float) -> dict[str, Any]:
    data = np.asarray(latencies) * 1000.0 if latencies else np.asarray([np.nan])
    return {
        "requests": len(latencies),
        "throughput_per_second": len(latencies) / seconds,
        "statuses": {str(code): count for code, count in sorted(statuses.items())},
        "p50_ms": float(np.percentile(data, 50)),
        "p95_ms": float(np.percentile(data, 95)),
        "p99_ms": float(np.percentile(data, 99)),
        "max_ms": float(np.max(data)),
    }


Sender = Callable[[httpx2.Client], httpx2.Response]


def closed_loop(base: str, send: Sender, clients: int, seconds: float) -> dict[str, Any]:
    """Each client sends its next request when the last one returns."""
    lock = threading.Lock()
    latencies: list[float] = []
    statuses: Counter[int] = Counter()
    stop_at = time.monotonic() + seconds

    def work() -> None:
        with client(base) as http:
            while time.monotonic() < stop_at:
                started = time.perf_counter()
                code = send(http).status_code
                elapsed = time.perf_counter() - started
                with lock:
                    latencies.append(elapsed)
                    statuses[code] += 1

    started = time.monotonic()
    with ThreadPoolExecutor(clients) as pool:
        for future in [pool.submit(work) for _ in range(clients)]:
            future.result()
    return {"clients": clients, **summarize(latencies, statuses, time.monotonic() - started)}


def open_loop(base: str, send: Sender, rate: float, seconds: float) -> dict[str, Any]:
    """Requests start on a fixed schedule, and latency runs from the schedule, not the send."""
    count = int(rate * seconds)
    lock = threading.Lock()
    latencies: list[float] = []
    statuses: Counter[int] = Counter()
    local = threading.local()
    begin = time.perf_counter() + 0.2

    def one(index: int) -> None:
        scheduled = begin + index / rate
        delay = scheduled - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        if not hasattr(local, "http"):
            local.http = client(base)
        code = send(local.http).status_code
        elapsed = time.perf_counter() - scheduled
        with lock:
            latencies.append(elapsed)
            statuses[code] += 1

    # One thread per request that could be outstanding, so that the generator is not the queue.
    with ThreadPoolExecutor(min(count, 256)) as pool:
        for future in [pool.submit(one, index) for index in range(count)]:
            future.result()
    # Throughput over the time the work actually took, from the first scheduled start to the last
    # completion. Dividing by the nominal duration would always return the offered rate, so it
    # could never show the service falling behind.
    elapsed = time.perf_counter() - begin
    return {"offered_per_second": rate, **summarize(latencies, statuses, elapsed)}


def fresh_sample(base: str) -> dict[str, Any]:
    """The sample request, dated now, since quotes and markets age out during a long run."""
    with client(base) as http:
        sample: dict[str, Any] = http.get("/v1/example").json()
    return sample


def integrity_run(base: str, audit_log: Path, submissions: int, clients: int) -> dict[str, Any]:
    """Concurrent submissions, then every approval raced by two approvers."""
    hedge_request = fresh_sample(base)

    def submit(_: int) -> tuple[int, str | None]:
        with client(base, "alice") as http:
            response = http.post("/v1/proposals", json=hedge_request)
            body = response.json() if response.status_code == 201 else {}
            return response.status_code, body.get("id")

    started = time.monotonic()
    with ThreadPoolExecutor(clients) as pool:
        outcomes = list(pool.map(submit, range(submissions)))
    submit_seconds = time.monotonic() - started
    accepted = [record_id for code, record_id in outcomes if code == 201 and record_id]

    def approve(args: tuple[str, str]) -> int:
        record_id, approver = args
        with client(base, approver) as http:
            path = f"/v1/proposals/{record_id}/approve"
            return http.post(path, json={"reason": "load test"}).status_code

    races = [(record_id, name) for record_id in accepted for name in ("bob", "carol")]
    with ThreadPoolExecutor(clients) as pool:
        approvals = Counter(pool.map(approve, races))

    with client(base) as http:
        health = http.get("/healthz")
    check = verify_audit_log(audit_log)
    events: Counter[str] = Counter()
    for line in audit_log.read_text().splitlines():
        events[json.loads(line)["event"]] += 1
    return {
        "submissions_sent": submissions,
        "submit_statuses": {str(k): v for k, v in sorted(Counter(c for c, _ in outcomes).items())},
        "submit_seconds": submit_seconds,
        "submits_per_second": submissions / submit_seconds,
        "approval_race_statuses": {str(k): v for k, v in sorted(approvals.items())},
        "audit_events": dict(sorted(events.items())),
        "chain_ok": check.ok,
        "chain_records": check.records,
        "chain_reason": check.reason,
        "healthz_status": health.status_code,
        "audit_log_bytes": audit_log.stat().st_size,
        "every_submission_recorded_once": events["submitted"] == len(accepted),
        "one_winner_per_race": events["approved"] == len(accepted)
        and approvals[200] == len(accepted),
    }


def audit_growth(base: str, audit_log: Path, checkpoints: tuple[int, ...]) -> list[dict[str, Any]]:
    """Latency of a submission as the audit log grows, sequentially so that nothing else queues."""
    rows = []
    with client(base) as http:
        for target in checkpoints:
            current = verify_audit_log(audit_log).records
            for index in range(max(0, target - current)):
                if index % GROWTH_REFRESH_EVERY == 0:
                    request = fresh_sample(base)
                http.post("/v1/proposals", json=request).raise_for_status()
            request = fresh_sample(base)
            samples = []
            for _ in range(20):
                started = time.perf_counter()
                http.post("/v1/proposals", json=request).raise_for_status()
                samples.append((time.perf_counter() - started) * 1000.0)
            started = time.perf_counter()
            http.get("/healthz").raise_for_status()
            healthz_ms = (time.perf_counter() - started) * 1000.0
            rows.append(
                {
                    "audit_records": verify_audit_log(audit_log).records,
                    "submit_p50_ms": float(np.percentile(samples, 50)),
                    "healthz_ms": healthz_ms,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--rates", default="1,2,4,8")
    parser.add_argument("--clients", default="1,4,16")
    parser.add_argument("--submissions", type=int, default=500)
    parser.add_argument("--growth", default="1000,5000")
    args = parser.parse_args()

    client_counts = [int(c) for c in args.clients.split(",")]
    rates = [float(r) for r in args.rates.split(",")]
    growth = tuple(int(g) for g in args.growth.split(",") if g)
    results: dict[str, Any] = {
        "code_version": __version__,
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "note": "One uvicorn worker on the same machine as the load generator, on loopback.",
        "seconds_per_case": args.seconds,
    }

    with tempfile.TemporaryDirectory() as scratch:
        audit_log = Path(scratch) / "audit.jsonl"
        with running_service(audit_log) as base:
            # Each sender builds on a request dated at the start of its own case.
            def stage(name: str) -> Sender:
                sample = fresh_sample(base)
                body = sample
                if name != "hedge":
                    body = {"book": sample["book"], "market": sample["market"]}
                return lambda http: http.post(f"/v1/{name}", json=body)

            results["capacity"] = {
                name: [closed_loop(base, stage(name), n, args.seconds) for n in client_counts]
                for name in ("value", "stress", "hedge")
            }
            results["fixed_rate_hedge"] = [
                open_loop(base, stage("hedge"), rate, args.seconds) for rate in rates
            ]
            results["integrity"] = integrity_run(
                base, audit_log, args.submissions, max(client_counts)
            )
            if growth:
                results["audit_growth"] = audit_growth(base, audit_log, growth)

    text = json.dumps(results, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    integrity = results["integrity"]
    passed = (
        integrity["chain_ok"]
        and integrity["every_submission_recorded_once"]
        and integrity["one_winner_per_race"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
