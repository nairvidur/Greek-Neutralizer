"""CLI behavior (AXI rules), request schema validation, approvals, audit chain, and tickets."""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from greeks_neutralizer._version import __version__
from greeks_neutralizer.cli import main
from greeks_neutralizer.domain import OptionContract, OptionRight
from greeks_neutralizer.errors import ValidationError
from greeks_neutralizer.examples import example_book, example_market, example_request
from greeks_neutralizer.hedging.optimizer import solve_hedge
from greeks_neutralizer.hedging.problem import HedgeLeg, ProposalStatus
from greeks_neutralizer.pricing.black import black_greeks
from greeks_neutralizer.risk.scenarios import HISTORICAL_STRESSES, book_scenario_pnl
from greeks_neutralizer.schema import (
    parse_book,
    parse_hedge_request,
    parse_instrument,
    parse_market,
    proposal_to_json,
)
from greeks_neutralizer.toon import encode, scalar
from greeks_neutralizer.workflow import (
    AuditLog,
    ProposalStore,
    State,
    WorkflowError,
    tickets_csv,
    tickets_fix,
    verify_audit_log,
)
from tests.conftest import AS_OF


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "greeks_neutralizer.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    paths = {name: tmp_path / f"{name}.json" for name in ("request", "book", "market")}
    paths["request"].write_text(json.dumps(example_request()))
    paths["book"].write_text(json.dumps(example_book()))
    paths["market"].write_text(json.dumps(example_market()))
    return paths


@pytest.mark.parametrize("flag", ["--version", "-v", "-V"])
def test_version_fast_path_prints_bare_version(flag, capsys):
    assert main([flag]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_version_does_not_load_the_engine():
    code = (
        "import sys; from greeks_neutralizer.cli import main; main(['--version']); "
        "assert 'numpy' not in sys.modules and 'scipy' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_home_view_shows_content_not_a_manual():
    result = run_cli()
    assert result.returncode == 0
    assert "description:" in result.stdout
    assert "gn hedge" in result.stdout


def test_unknown_flag_is_rejected_with_valid_flags_on_stdout():
    result = run_cli("hedge", "--requst", "x.json", "--request", "y.json")
    assert result.returncode == 2
    assert "error:" in result.stdout
    assert "--request" in result.stdout


def test_missing_required_flag_is_a_usage_error():
    result = run_cli("hedge")
    assert result.returncode == 2
    assert "--request" in result.stdout


def test_missing_file_is_a_structured_error():
    result = run_cli("hedge", "--request", "/nonexistent/x.json")
    assert result.returncode == 1
    assert result.stdout.startswith("error:")
    assert "Traceback" not in result.stdout + result.stderr


def test_bad_json_is_a_structured_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    result = run_cli("hedge", "--request", str(bad))
    assert result.returncode == 1
    assert "not valid JSON" in result.stdout


def test_value_runs_end_to_end(files):
    result = run_cli("value", "--book", str(files["book"]), "--market", str(files["market"]))
    assert result.returncode == 0, result.stdout + result.stderr
    # Short 10 of each by closed form, 30 days, 5%, 25%, 100 a contract at a spot of 100.
    call = black_greeks(100.0, 100.0, 30 / 365, 0.05, 0.0, 0.25, 1).delta[0]
    put = black_greeks(100.0, 95.0, 30 / 365, 0.05, 0.0, 0.25, -1).delta[0]
    printed = next(
        line.split(":")[1] for line in result.stdout.splitlines() if "delta_dollars:" in line
    )
    assert float(printed) == pytest.approx(-10 * 100 * (call + put) * 100.0, rel=1e-8)
    assert "--full" in result.stdout


def test_value_full_lists_lines(files):
    args = ("value", "--book", str(files["book"]), "--market", str(files["market"]), "--full")
    assert "lines[2]" in run_cli(*args).stdout


def test_empty_book_says_so_definitively(tmp_path, files):
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    result = run_cli("value", "--book", str(empty), "--market", str(files["market"]))
    assert result.returncode == 0
    assert "0 positions" in result.stdout


def test_stress_is_sorted_worst_first(files):
    result = run_cli(
        "stress",
        "--book",
        str(files["book"]),
        "--market",
        str(files["market"]),
        "--set",
        "historical",
    )
    assert result.returncode == 0
    lines = result.stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("pnl[")) + 1
    rows = list(csv.reader(line.strip() for line in lines[start:] if line.startswith("  ")))
    shown = [(row[0], float(row[1])) for row in rows]
    # The oracle for "worst" is the smallest P&L from a separate repricing call, not the CLI's sort.
    book = parse_book(json.loads(files["book"].read_text()))
    market = parse_market(json.loads(files["market"].read_text()))
    truth = {
        s.name: float(v)
        for s, v in zip(
            HISTORICAL_STRESSES, book_scenario_pnl(book, market, HISTORICAL_STRESSES), strict=True
        )
    }
    assert len(shown) == len(truth)
    assert [pnl for _, pnl in shown] == sorted(pnl for _, pnl in shown)
    worst = min(truth, key=lambda name: truth[name])
    assert shown[0][0] == worst
    assert shown[0][1] == pytest.approx(truth[worst], rel=1e-8)
    assert f'name: "{worst}"' in result.stdout


def test_stress_rejects_unknown_set(files):
    result = run_cli(
        "stress", "--book", str(files["book"]), "--market", str(files["market"]), "--set", "nope"
    )
    assert result.returncode == 1
    assert "unknown scenario set" in result.stdout


def test_hedge_runs_verified_end_to_end(files):
    result = run_cli("hedge", "--request", str(files["request"]))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "verified: true" in result.stdout
    assert "legs[" in result.stdout


def test_infeasible_hedge_names_binding_constraints(tmp_path):
    request = example_request()
    request["candidates"] = request["candidates"][:1]
    request["targets"] = [{"underlying": "XYZ", "greek": "vega_1pt", "lower": -1.0, "upper": 1.0}]
    request["scenarios"] = []
    request["run_stability"] = False
    path = tmp_path / "r.json"
    path.write_text(json.dumps(request))
    result = run_cli("hedge", "--request", str(path))
    assert result.returncode == 1
    assert "binding" in result.stdout
    assert "vega_1pt" in result.stdout


def test_frontier_has_baselines(files):
    result = run_cli("frontier", "--request", str(files["request"]))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "do nothing" in result.stdout
    assert "delta only" in result.stdout


def test_example_round_trips_through_the_parser():
    result = run_cli("example", "--kind", "hedge")
    assert result.returncode == 0
    problem, snapshot, stability = parse_hedge_request(json.loads(result.stdout))
    assert problem.book
    assert snapshot.underlyings
    assert stability is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.update(max_legz=3), "unknown field"),
        (lambda r: r["config"].update(risk_aversionn=1.0), "unknown field"),
        (lambda r: r.pop("targets"), "missing field"),
        (lambda r: r["market"].update(as_of="2026-01-05T15:00:00"), "UTC offset"),
        (lambda r: r["book"][0].update(quantity=1.5), "signed int"),
        (lambda r: r["book"][0]["instrument"].update(right="CALL"), "CALL"),
        (lambda r: r["targets"][0].update(greek="delta"), "unknown greek"),
        (lambda r: r.update(schema_version=99), "schema_version"),
        (lambda r: r.update(scenarios="nope"), "unknown scenario set"),
        (lambda r: r["market"]["underlyings"]["XYZ"].pop("rate"), "no default rate"),
    ],
    ids=[
        "typo-top",
        "typo-config",
        "missing",
        "naive-time",
        "fractional",
        "bad-right",
        "bad-greek",
        "version",
        "bad-set",
        "no-rate",
    ],
)
def test_schema_rejects_bad_requests_with_a_useful_message(mutate, message):
    request = example_request()
    mutate(request)
    with pytest.raises((ValidationError, ValueError)) as err:
        parse_hedge_request(request)
    assert message in str(err.value)


def test_schema_parsers_build_valid_objects():
    assert parse_book(example_book())[0].quantity == -10
    assert parse_market(example_market()).market("XYZ").spot == 100.0
    assert parse_instrument({"type": "linear", "symbol": "A", "underlying": "A"}).symbol == "A"
    with pytest.raises(ValidationError, match="must be 'option' or 'linear'"):
        parse_instrument({"type": "swap"})


def test_proposal_json_is_serializable():
    problem, snapshot, _ = parse_hedge_request(example_request())
    data = proposal_to_json(solve_hedge(problem, snapshot))
    assert json.loads(json.dumps(data))["verification"]["passed"] is True


def test_toon_encoding():
    assert scalar(True) == "true"
    assert scalar(None) == "null"
    assert scalar(float("nan")) == "null"
    assert scalar("has space") == '"has space"'
    assert scalar("null") == '"null"'
    out = encode({"rows": [{"a": 1, "b": "x"}, {"a": 2, "b": "y z"}], "tags": ["p", "q"], "e": []})
    assert 'rows[2]{a,b}:\n  1,x\n  2,"y z"' in out
    assert "tags[2]: p,q" in out
    assert "e[0]:" in out


# --- Workflow ------------------------------------------------------------------------------------


class Clock:
    def __init__(self):
        self.now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)

    def __call__(self):
        return self.now


@pytest.fixture
def proposal():
    problem, snapshot, _ = parse_hedge_request(example_request())
    return solve_hedge(problem, snapshot, run_stability=True)


@pytest.fixture
def store(tmp_path):
    return ProposalStore(AuditLog(tmp_path / "audit.jsonl"), clock=Clock())


def test_full_lifecycle_is_audited(store, proposal, tmp_path):
    record = store.submit(proposal, "alice")
    assert record.state is State.PROPOSED
    assert store.approve(record.id, "bob", "looks right").state is State.APPROVED
    tickets = store.export_tickets(record.id, "bob")
    assert record.id in tickets
    assert store.get(record.id).state is State.EXPORTED
    check = verify_audit_log(tmp_path / "audit.jsonl")
    assert check.ok
    assert check.records == 3


def test_four_eyes_blocks_self_approval(store, proposal):
    record = store.submit(proposal, "alice")
    with pytest.raises(WorkflowError, match="four-eyes"):
        store.approve(record.id, "alice")
    assert store.get(record.id).state is State.PROPOSED


def test_stale_proposal_cannot_be_approved_or_exported(store, proposal):
    record = store.submit(proposal, "alice")
    store.clock.now += timedelta(minutes=6)
    with pytest.raises(WorkflowError, match="re-propose"):
        store.approve(record.id, "bob")

    fresh = store.submit(proposal, "alice")
    store.approve(fresh.id, "bob")
    store.clock.now += timedelta(minutes=6)
    with pytest.raises(WorkflowError, match="re-propose"):
        store.export_tickets(fresh.id, "bob")


def test_unapproved_cannot_export_and_rejected_cannot_approve(store, proposal):
    record = store.submit(proposal, "alice")
    with pytest.raises(WorkflowError, match="only an approved"):
        store.export_tickets(record.id, "bob")
    store.reject(record.id, "bob", "wrong size")
    with pytest.raises(WorkflowError, match="rejected"):
        store.approve(record.id, "bob")


def test_rejection_needs_a_reason_and_is_idempotent(store, proposal):
    record = store.submit(proposal, "alice")
    with pytest.raises(ValidationError, match="reason"):
        store.reject(record.id, "bob", "  ")
    store.reject(record.id, "bob", "no")
    records = store.audit.path.read_text().splitlines()
    assert store.reject(record.id, "bob", "no").state is State.REJECTED
    assert store.audit.path.read_text().splitlines() == records


def test_repeat_approval_is_a_no_op(store, proposal):
    record = store.submit(proposal, "alice")
    first = store.approve(record.id, "bob")
    records = store.audit.path.read_text().splitlines()
    assert store.approve(record.id, "bob") == first
    assert store.audit.path.read_text().splitlines() == records


def test_unstable_proposal_cannot_be_approved(store, proposal):
    unstable = replace(proposal, status=ProposalStatus.UNSTABLE)
    record = store.submit(unstable, "alice")
    with pytest.raises(WorkflowError, match="unstable"):
        store.approve(record.id, "bob")


def test_failed_verification_cannot_be_submitted(store, proposal):
    bad = replace(proposal, verification=replace(proposal.verification, passed=False))
    with pytest.raises(WorkflowError, match="verification"):
        store.submit(bad, "alice")


def test_actor_is_required(store, proposal):
    with pytest.raises(ValidationError, match="actor"):
        store.submit(proposal, " ")


def test_unknown_proposal_and_format(store, proposal):
    with pytest.raises(WorkflowError, match="no proposal"):
        store.get("nope")
    record = store.submit(proposal, "alice")
    store.approve(record.id, "bob")
    with pytest.raises(ValidationError, match="ticket format"):
        store.export_tickets(record.id, "bob", "xml")


@pytest.mark.parametrize(
    "tamper",
    [
        lambda lines: [lines[0].replace('"alice"', '"mallory"'), *lines[1:]],
        lambda lines: lines[1:],
        lambda lines: [lines[1], lines[0], *lines[2:]],
    ],
    ids=["edit", "remove-first", "reorder"],
)
def test_audit_chain_detects_tampering(store, proposal, tmp_path, tamper):
    record = store.submit(proposal, "alice")
    store.approve(record.id, "bob")
    store.export_tickets(record.id, "bob")
    path = tmp_path / "audit.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(tamper(lines)) + "\n")
    check = verify_audit_log(path)
    assert not check.ok
    assert check.first_bad_line is not None


def test_audit_chain_continues_after_restart(tmp_path, proposal):
    path = tmp_path / "audit.jsonl"
    ProposalStore(AuditLog(path), clock=Clock()).submit(proposal, "alice")
    ProposalStore(AuditLog(path), clock=Clock()).submit(proposal, "carol")
    check = verify_audit_log(path)
    assert check.ok
    assert check.records == 2


def test_ticket_side_quantity_and_limit_are_what_the_legs_say(proposal):
    # Hand-built: buy 5 at 1.25 and sell 3 at 1.10, in the units a broker reads them.
    contract = OptionContract("XYZ", OptionRight.CALL, 100.0, AS_OF + timedelta(days=30))
    legs = (
        HedgeLeg(contract, 5, 1.25, 625.0, 2.0, "buy"),
        HedgeLeg(contract, -3, 1.10, -330.0, 1.2, "sell"),
    )
    subject = replace(proposal, legs=legs)
    rows = list(csv.DictReader(io.StringIO(tickets_csv(subject, "abc"))))
    assert [(r["side"], r["quantity"], r["limit_price"]) for r in rows] == [
        ("BUY", "5", "1.25"),
        ("SELL", "3", "1.1"),
    ]
    messages = tickets_fix(subject, "abcdefghij", AS_OF).strip().splitlines()
    fields = [dict(part.split("=") for part in m.rstrip("|").split("|")) for m in messages]
    # FIX 4.4: Side 1 is buy and 2 is sell, with OrderQty unsigned and Price at four places here.
    assert [(f["54"], f["38"], f["44"]) for f in fields] == [
        ("1", "5", "1.2500"),
        ("2", "3", "1.1000"),
    ]


def test_missing_audit_log_is_empty_and_ok(tmp_path):
    assert verify_audit_log(tmp_path / "none.jsonl").ok


def test_tickets_match_the_proposal(proposal):
    text = tickets_csv(proposal, "abc")
    rows = text.strip().splitlines()
    assert proposal.legs
    assert len(rows) == len(proposal.legs) + 1
    assert rows[0].startswith("proposal_id,side,quantity")
    fix = tickets_fix(proposal, "abcdefghij", AS_OF).strip().splitlines()
    assert len(fix) == len(proposal.legs)
    for message, leg in zip(fix, proposal.legs, strict=True):
        assert "35=D|" in message
        assert f"38={abs(leg.quantity)}|" in message
        assert ("54=1|" if leg.quantity > 0 else "54=2|") in message
        assert "40=2|" in message


def test_request_can_turn_on_the_position_terms():
    """A request reaches the optimizer's position terms, which are off unless asked for."""
    request = example_request()
    assert parse_hedge_request(request)[0].config.price_exit is False
    request["config"].update(price_exit=True, max_gross_option_contracts=40)
    request["held_hedges"] = []
    config = parse_hedge_request(request)[0].config
    assert config.price_exit is True
    assert config.max_gross_option_contracts == 40


@pytest.mark.parametrize(
    "terms",
    [{"price_exit": True}, {"max_gross_option_contracts": 40}],
    ids=["price-exit", "gross-cap"],
)
def test_position_terms_need_the_held_hedges_named(terms):
    """Left out, the terms would see the trade alone and quietly ignore every hedge already held,
    so a request that turns them on must list its held hedges, [] when it holds none."""
    request = example_request()
    request["config"].update(terms)
    with pytest.raises(ValidationError, match=r"held_hedges.*\[\] if none"):
        parse_hedge_request(request)
    request["held_hedges"] = []
    assert parse_hedge_request(request)[0].held_hedges == ()


def test_position_terms_switched_off_need_no_held_hedges():
    request = example_request()
    request["config"].update(price_exit=False)
    assert parse_hedge_request(request)[0].held_hedges == ()
