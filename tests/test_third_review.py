"""Regression tests from the third review round. Each failed before its fix.

Oracles are stated where they are not obvious: a constructed hedge whose existence is checked by
repricing, the no-arbitrage identity for stock across an ex-date, Black theta for a hedged book, and
option payoffs at expiry. The solver is replaced only at its SciPy boundary, to produce statuses
that are too rare to construct, using the status and message SciPy documents for them.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.testing import assert_allclose
from scipy.optimize import OptimizeResult
from scipy.optimize import linprog as real_linprog
from scipy.optimize import milp as real_milp

from greeks_neutralizer.cli import main
from greeks_neutralizer.domain import LinearInstrument, LinearKind, Position, Quote
from greeks_neutralizer.errors import (
    HedgeInfeasibleError,
    MarketDataError,
    SearchLimitError,
    ValidationError,
)
from greeks_neutralizer.examples import example_request
from greeks_neutralizer.hedging import diagnostics, optimizer
from greeks_neutralizer.hedging.diagnostics import hedge_frontier
from greeks_neutralizer.hedging.optimizer import solve_hedge
from greeks_neutralizer.hedging.problem import (
    GreekName,
    GreekTarget,
    HedgeCandidate,
    HedgeConfig,
    HedgeProblem,
    ProposalStatus,
)
from greeks_neutralizer.market import CashDividend, MarketSnapshot
from greeks_neutralizer.pricing.black import black_greeks
from greeks_neutralizer.pricing.book import value_book, value_options
from greeks_neutralizer.risk.scenarios import (
    HISTORICAL_STRESSES,
    Scenario,
    book_scenario_pnl,
    scenario_pnl_matrix,
    shock_snapshot,
)
from greeks_neutralizer.schema import parse_hedge_request, proposal_to_json
from greeks_neutralizer.service.app import create_app
from greeks_neutralizer.surface import SviSlice, SviSurface
from greeks_neutralizer.workflow import AuditLog, ProposalStore
from tests.conftest import AS_OF, american, euro, expiry_in, flat_snapshot

STOCK = LinearInstrument("XYZ", "XYZ")
DECAY = Scenario("decay", days_forward=7.0)
CLOCK_MESSAGE = "Time limit reached. (HiGHS Status 13: model_status is Time limit reached)"
NODE_MESSAGE = "The HiGHS status code was not recognized. (HiGHS Status 16: Solution limit reached)"


def fair(right, strike, days, half=0.02, size=1000, vol=0.25):
    omega = 1 if right == "call" else -1
    price = black_greeks(100.0, strike, days / 365, 0.05, 0.0, vol, omega).price[0]
    quote = Quote(max(price - half, 0.0), price + half, size, size, AS_OF)
    return HedgeCandidate(euro(right, strike, days), quote)


def stock_candidate():
    return HedgeCandidate(STOCK, Quote(99.99, 100.01, 1_000_000, 1_000_000, AS_OF))


BOOK = (Position(euro("call", 100, 30), 10), Position(euro("put", 95, 30), -5))
UNIVERSE = (
    stock_candidate(),
    fair("put", 95, 60),
    fair("call", 105, 60),
    fair("call", 100, 90),
    fair("put", 90, 90),
)
DGV = tuple(
    GreekTarget("XYZ", g, weight=1.0) for g in (GreekName.DELTA, GreekName.GAMMA, GreekName.VEGA)
)


def problem(**config):
    return HedgeProblem(BOOK, UNIVERSE, DGV, (), HedgeConfig(risk_aversion=10.0, **config))


# --- A search limit is not infeasibility ----------------------------------------------------------


def _known_feasible_problem(node_limit):
    """A book that is the exact negative of an integer hedge, so that a feasible point exists."""
    candidates = tuple(
        fair(right, strike, days)
        for days in (30, 45, 60, 90, 120, 180, 365)
        for strike, right in (
            (85, "put"),
            (90, "put"),
            (95, "put"),
            (105, "call"),
            (110, "call"),
            (115, "call"),
        )
    )
    lots = np.random.default_rng(7).integers(-40, 41, len(candidates))
    book = tuple(
        Position(c.instrument, -int(q)) for c, q in zip(candidates, lots, strict=True) if q
    )
    targets = tuple(GreekTarget("XYZ", g, lower=-0.5, upper=0.5) for g in GreekName)
    hedge = tuple(
        Position(c.instrument, int(q)) for c, q in zip(candidates, lots, strict=True) if q
    )
    config = HedgeConfig(mip_node_limit=node_limit)
    return HedgeProblem(book, candidates, targets, (), config), hedge


def test_search_limit_on_a_known_feasible_problem_is_not_reported_as_infeasible():
    hedged_problem, hedge = _known_feasible_problem(node_limit=1)
    # The oracle: repricing shows that this integer hedge meets every band, so infeasible is false.
    flat = value_book((*hedged_problem.book, *hedge), flat_snapshot()).by_underlying["XYZ"]
    for target in hedged_problem.targets:
        assert abs(getattr(flat, target.greek.value)) <= 0.5
    with pytest.raises(SearchLimitError, match="mip_node_limit=1"):
        solve_hedge(hedged_problem, flat_snapshot())


def _results(monkeypatch, name, script):
    """Replace a SciPy solver: each call takes the next step, and the last step repeats."""
    real = {"milp": real_milp, "linprog": real_linprog}[name]
    calls = {"n": 0}

    def fake(*args, **kwargs):
        step = script[min(calls["n"], len(script) - 1)]
        calls["n"] += 1
        return step(real(*args, **kwargs))

    monkeypatch.setattr(optimizer, name, fake)
    return calls


def passthrough(result):
    return result


def no_incumbent(status, message):
    return lambda result: OptimizeResult(x=None, status=status, message=message, fun=None)


def clock_with_incumbent(result):
    return OptimizeResult(x=result.x, status=1, message=CLOCK_MESSAGE, fun=result.fun)


@pytest.mark.parametrize(
    ("status", "message"), [(4, NODE_MESSAGE), (1, CLOCK_MESSAGE)], ids=["node-limit", "clock"]
)
def test_every_attempt_stopping_with_no_incumbent_is_its_own_error(monkeypatch, status, message):
    _results(monkeypatch, "milp", [no_incumbent(status, message)])
    with pytest.raises(SearchLimitError, match="mip_node_limit") as err:
        solve_hedge(problem(), flat_snapshot())
    assert not isinstance(err.value, HedgeInfeasibleError)


def test_proved_infeasible_integer_search_stays_infeasible(monkeypatch):
    _results(monkeypatch, "milp", [no_incumbent(2, "The problem is infeasible.")])
    with pytest.raises(HedgeInfeasibleError):
        solve_hedge(problem(), flat_snapshot())


def test_reaching_the_clock_on_an_earlier_attempt_is_flagged(monkeypatch):
    clean = solve_hedge(problem(), flat_snapshot())
    assert not any(i.startswith("SOLVER_TIME_LIMIT") for i in clean.issues)
    _results(monkeypatch, "milp", [no_incumbent(1, CLOCK_MESSAGE), passthrough])
    flagged = solve_hedge(problem(), flat_snapshot())
    assert any(i.startswith("SOLVER_TIME_LIMIT") for i in flagged.issues)


def test_relaxation_stopped_at_the_clock_is_not_used_as_a_lower_bound(monkeypatch):
    _results(monkeypatch, "linprog", [clock_with_incumbent])
    with pytest.raises(SearchLimitError, match="time_limit_seconds"):
        solve_hedge(problem(), flat_snapshot())


def test_stability_resolve_reaching_the_clock_is_flagged(monkeypatch):
    _results(monkeypatch, "milp", [passthrough, clock_with_incumbent])
    proposal = solve_hedge(problem(), flat_snapshot(), run_stability=True)
    assert proposal.stability is not None
    assert proposal.stability.hit_time_limit
    assert any(i.startswith("SOLVER_TIME_LIMIT") for i in proposal.issues)


def test_stability_resolve_with_no_incumbent_is_an_error_and_not_instability(monkeypatch):
    _results(monkeypatch, "milp", [passthrough, no_incumbent(4, NODE_MESSAGE)])
    with pytest.raises(SearchLimitError, match="stability check could not finish"):
        solve_hedge(problem(), flat_snapshot(), run_stability=True)


def test_frontier_reports_a_search_limit_as_such_and_keeps_the_baselines(monkeypatch):
    _results(monkeypatch, "milp", [no_incumbent(4, NODE_MESSAGE)])
    points = hedge_frontier(problem(), flat_snapshot(), risk_aversions=(1.0,))
    assert points[0].label == "do nothing"
    assert points[-1].status == "search_limit"
    assert "mip_node_limit" in (points[-1].unsolved_reason or "")


def test_frontier_still_reports_a_proved_infeasible_point():
    band = (GreekTarget("XYZ", GreekName.VEGA, lower=-1.0, upper=1.0),)
    book_problem = HedgeProblem(BOOK, (stock_candidate(),), band)
    point = hedge_frontier(book_problem, flat_snapshot(), risk_aversions=(1.0,))[-1]
    assert point.status == "infeasible"
    assert "vega_1pt" in (point.unsolved_reason or "")


def test_infeasibility_diagnosis_stopped_at_the_clock_does_not_claim_a_conflict(monkeypatch):
    _results(monkeypatch, "milp", [no_incumbent(1, CLOCK_MESSAGE)])
    band = (GreekTarget("XYZ", GreekName.VEGA, lower=-1.0, upper=1.0),)
    with pytest.raises(HedgeInfeasibleError) as err:
        solve_hedge(HedgeProblem(BOOK, (stock_candidate(),), band), flat_snapshot())
    assert any("could not be diagnosed" in text for text in err.value.binding)
    assert not any("conflict" in text for text in err.value.binding)


def test_service_returns_a_search_limit_as_a_structured_client_error(monkeypatch, tmp_path):
    _results(monkeypatch, "milp", [no_incumbent(4, NODE_MESSAGE)])
    app = create_app({"t": "alice"}, tmp_path / "audit.jsonl")
    with TestClient(app) as client:
        response = client.post(
            "/v1/hedge", json=example_request(), headers={"Authorization": "Bearer t"}
        )
    assert response.status_code == 422
    assert response.json()["type"] == "search_limit"
    assert "mip_node_limit" in response.json()["error"]


# --- Stability uses only what the base solve accepted ---------------------------------------------


def test_a_quote_the_base_solve_rejected_does_not_change_the_stability_report():
    rich = fair("call", 102, 60, vol=0.50)
    without = problem()
    with_rich = replace(without, candidates=(*UNIVERSE, rich))
    base = solve_hedge(without, flat_snapshot(), run_stability=True)
    proposal = solve_hedge(with_rich, flat_snapshot(), run_stability=True)
    assert any("iv_outlier" in item.reason for item in proposal.excluded)
    assert proposal.quantities == base.quantities
    assert proposal.stability == base.stability


def test_stability_measures_a_perturbation_that_removes_a_held_candidate(monkeypatch):
    """Regression: lots were compared only with equal candidate counts, and it reported 0.0."""
    surface = SviSurface(
        (SviSlice(60 / 365, 0.0625 * 60 / 365, 0.001, 0.0, 0.0, 0.1),), ((-0.15, 0.15),)
    )
    base = flat_snapshot()
    snapshot = MarketSnapshot(AS_OF, {"XYZ": replace(base.market("XYZ"), surface=surface)})

    def fair_on_surface(contract):
        price = float(value_options([contract], snapshot).greeks.price[0])
        return HedgeCandidate(contract, Quote(price - 0.02, price + 0.02, 1000, 1000, AS_OF))

    # A spot rise under sticky moneyness carries the 90 put's strike beyond the quoted range.
    near_edge, inside = euro("put", 90, 60), euro("call", 115, 60)
    monkeypatch.setattr(
        diagnostics,
        "PERTURBATIONS",
        (Scenario("spot up, sticky moneyness", spot_shock=0.2, sticky_strike=False),),
    )
    book = (Position(near_edge, -20),)
    proposal = solve_hedge(
        HedgeProblem(
            book,
            (fair_on_surface(near_edge), fair_on_surface(inside), stock_candidate()),
            DGV,
            (),
            HedgeConfig(risk_aversion=10.0),
        ),
        snapshot,
        run_stability=True,
    )
    held = proposal.quantities[near_edge.key]
    assert held > 0
    contracts = sum(abs(q) for key, q in proposal.quantities.items() if key != STOCK.key)
    assert proposal.stability is not None
    # All of that leg has to move once it cannot trade, whatever else the re-solve does.
    assert proposal.stability.lot_change_ratio >= abs(held) / max(contracts, 10)
    assert proposal.stability.regret_ratio == float("inf")
    assert not proposal.stability.is_stable


# --- A horizon that crosses an ex-date ------------------------------------------------------------


def _with_dividend(amount=2.0, days=3.0):
    base = flat_snapshot()
    dividend = CashDividend(expiry_in(days), amount)
    return MarketSnapshot(AS_OF, {"XYZ": replace(base.market("XYZ"), dividends=(dividend,))})


def test_stock_across_an_ex_date_earns_its_price_fall_back_as_a_dividend():
    # No arbitrage: with spot otherwise flat, ex-dividend price fall plus dividend is exactly zero.
    instruments = [
        STOCK,
        LinearInstrument("SPY", "XYZ", LinearKind.PROXY, delta_ratio=0.1, basis_vol=0.01),
    ]
    assert_allclose(scenario_pnl_matrix(instruments, _with_dividend(), [DECAY]), 0.0, atol=1e-9)
    assert shock_snapshot(_with_dividend(), DECAY).market("XYZ").spot == pytest.approx(98.0)


def test_a_hedged_book_decays_across_an_ex_date_instead_of_earning_the_dividend():
    call = euro("call", 100, 30)
    snapshot = _with_dividend()
    start = value_book([Position(call, 10)], snapshot).by_underlying["XYZ"]
    hedged = (Position(call, 10), Position(STOCK, -round(start.delta_shares)))
    pnl = float(book_scenario_pnl(hedged, snapshot, [DECAY])[0])
    # A delta-hedged long call loses about theta at flat spot. Before the fix this was about +559.
    assert pnl < 0.0
    assert pnl == pytest.approx(start.theta_1d * 7.0, rel=0.1)


@pytest.mark.parametrize(
    ("contract", "payoff"),
    [(euro("call", 90, 2), 10.0), (euro("put", 110, 2), 10.0), (american("put", 110, 2), 10.0)],
    ids=["euro-call", "euro-put", "american-put"],
)
def test_option_expiring_before_the_ex_date_settles_at_spot_on_its_expiry(contract, payoff):
    snapshot = _with_dividend(days=5.0)
    start = float(value_options([contract], snapshot).greeks.price[0])
    pnl = scenario_pnl_matrix([contract], snapshot, [DECAY])[0, 0]
    # A dollar is well below the $200 that settling at the ex-dividend spot would be wrong by.
    assert pnl == pytest.approx((payoff - start) * contract.multiplier, abs=1.0)


def test_future_expiring_before_the_ex_date_settles_at_spot_on_its_expiry():
    future = LinearInstrument("XF", "XYZ", LinearKind.FUTURE, expiry=expiry_in(2.0))
    snapshot = _with_dividend(days=5.0)
    start = snapshot.market("XYZ").forward(AS_OF, expiry_in(2.0))
    assert scenario_pnl_matrix([future], snapshot, [DECAY])[0, 0] == pytest.approx(100.0 - start)


def test_dividends_beyond_a_shocked_spot_fail_closed():
    crash = Scenario("crash", spot_shock=-0.99, days_forward=7.0)
    with pytest.raises(MarketDataError, match="exceed spot"):
        shock_snapshot(_with_dividend(), crash)


# --- Scenarios must act on something --------------------------------------------------------------


def test_scenarios_with_no_targets_and_no_cvar_are_rejected():
    with pytest.raises(ValidationError, match="cvar_weight"):
        HedgeProblem(BOOK, UNIVERSE, (), HISTORICAL_STRESSES)
    request = example_request()
    request.update(targets=[], scenarios="historical")
    with pytest.raises(ValidationError, match="cvar_weight"):
        parse_hedge_request(request)


def test_cvar_only_problem_with_every_quote_stale_reports_no_usable_candidates():
    stale = tuple(
        replace(c, quote=replace(c.quote, timestamp=AS_OF - timedelta(hours=1))) for c in UNIVERSE
    )
    scenario_problem = HedgeProblem(
        BOOK, stale, (), HISTORICAL_STRESSES, HedgeConfig(cvar_weight=1.0)
    )
    proposal = solve_hedge(scenario_problem, flat_snapshot())
    assert proposal.status is ProposalStatus.NO_USABLE_CANDIDATES


# --- What a replay needs is recorded --------------------------------------------------------------


def test_manifest_records_the_library_versions_and_the_stability_flag():
    with_check = solve_hedge(problem(), flat_snapshot(), run_stability=True)
    without = solve_hedge(problem(), flat_snapshot(), run_stability=False)
    manifest = proposal_to_json(with_check)["manifest"]
    assert manifest["scipy_version"] == importlib.metadata.version("scipy")
    assert manifest["numpy_version"] == importlib.metadata.version("numpy")
    assert manifest["run_stability"] is True
    assert without.manifest.run_stability is False
    assert with_check.manifest.problem_fingerprint != without.manifest.problem_fingerprint


def test_audit_record_carries_the_library_versions_and_the_stability_flag(tmp_path):
    path = tmp_path / "audit.jsonl"
    parsed, snapshot, _ = parse_hedge_request(example_request())
    proposal = solve_hedge(parsed, snapshot, run_stability=True)
    store = ProposalStore(AuditLog(path), clock=lambda: snapshot.as_of)
    store.submit(proposal, "alice")
    detail = json.loads(path.read_text().splitlines()[0])["detail"]
    assert detail["scipy_version"] == importlib.metadata.version("scipy")
    assert detail["numpy_version"] == importlib.metadata.version("numpy")
    assert detail["run_stability"] is True


# --- The CLI's contract: structured errors and help on stdout -------------------------------------


@pytest.mark.parametrize(
    "content",
    [b"\xff\xfe", b"1" * 5000, b"[" * 200_000],
    ids=["not-utf8", "integer-too-long", "too-deep"],
)
def test_unloadable_request_file_is_a_structured_error(tmp_path, capfd, content):
    path = tmp_path / "request.json"
    path.write_bytes(content)
    assert main(["hedge", "--request", str(path)]) == 1
    captured = capfd.readouterr()
    assert "not valid JSON" in captured.out
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("argv", [["--help"], ["hedge", "--help"]], ids=["top", "subcommand"])
def test_help_is_written_to_the_real_stdout(argv):
    # A real process, because the redirect under test is of file descriptor 1.
    done = subprocess.run(
        [sys.executable, "-m", "greeks_neutralizer.cli", *argv],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "usage: gn" in done.stdout
    assert done.stderr == ""


def test_usage_errors_stay_on_stdout(capfd):
    assert main(["hedge", "--nonsense"]) == 2
    assert "valid flags" in capfd.readouterr().out


# --- An accepted field must do something ----------------------------------------------------------


def test_market_with_both_rate_and_curve_is_rejected():
    request = copy.deepcopy(example_request())
    request["market"]["underlyings"]["XYZ"]["curve"] = {"times": [1.0], "zero_rates": [0.01]}
    with pytest.raises(ValidationError, match="exactly one"):
        parse_hedge_request(request)


def test_future_with_a_delta_ratio_is_rejected():
    with pytest.raises(ValidationError, match="delta_ratio"):
        LinearInstrument("XF", "XYZ", LinearKind.FUTURE, delta_ratio=0.5, expiry=expiry_in(30))


@pytest.mark.parametrize(
    "build",
    [
        lambda: LinearInstrument("XYZ", "XYZ", LinearKind.SPOT, basis_vol=0.1),
        lambda: LinearInstrument(
            "XF", "XYZ", LinearKind.FUTURE, basis_vol=0.1, expiry=expiry_in(30)
        ),
    ],
    ids=["spot", "future"],
)
def test_basis_vol_on_anything_but_a_proxy_is_rejected(build):
    with pytest.raises(ValidationError, match="basis_vol"):
        build()
